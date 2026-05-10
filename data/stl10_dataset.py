"""STL-10 数据集加载与分层划分工具。

设计要点：
    - **内存缓存**：一次性把全部 PNG 解码并预处理为 float32 张量，常驻 RAM。
      train 集 5 000 张 96x96x3 约 132 MB；test 集 8 000 张约 211 MB，CPU 内存完全可承受。
    - **分层划分**：仅对 train 目录做 80/20 分层切分（每类内部独立切分），
      划分结果落盘到 split_index.json，多次实验自动复用，确保可复现。
    - **不接触 test**：build_train_valid_loaders 仅读取训练目录；build_test_loader 在评估阶段单独调用。
    - **零增强**：train / valid / test 使用同一 transform（ToTensor + 可选 Normalize）。
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

# STL-10 训练集（仅 5000 张）的通道均值方差，事先在标准 STL-10 上测得，可直接使用。
# 注：这是输入预处理意义上的 Normalize，不属于 BatchNorm/LayerNorm 等"网络归一化层"。
STL10_MEAN = (0.4467, 0.4398, 0.4066)
STL10_STD = (0.2603, 0.2566, 0.2713)


# ---------------------------------------------------------------------------
# Dataset & transform
# ---------------------------------------------------------------------------


class InMemoryDataset(Dataset):
    """基于已经在内存中的图像张量与标签张量构造的 Dataset。

    Args:
        images: 形状为 (N, C, H, W) 的 float32 张量，已完成 ToTensor + 可选 Normalize。
        labels: 形状为 (N,) 的 long 张量。
        indices: 可选索引子集；为 None 时使用全部样本。
    """

    def __init__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        indices: Sequence[int] | None = None,
    ) -> None:
        assert images.shape[0] == labels.shape[0], "images / labels 数量必须一致"
        self.images = images
        self.labels = labels
        if indices is None:
            self.indices = torch.arange(images.shape[0], dtype=torch.long)
        else:
            self.indices = torch.as_tensor(list(indices), dtype=torch.long)

    def __len__(self) -> int:
        return int(self.indices.numel())

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = int(self.indices[i].item())
        return self.images[idx], self.labels[idx]


def build_transform(normalize: bool = True) -> transforms.Compose:
    """构造与 train/valid/test 共享的预处理流程（无任何数据增强）。"""
    ops: list = [transforms.ToTensor()]
    if normalize:
        ops.append(transforms.Normalize(mean=STL10_MEAN, std=STL10_STD))
    return transforms.Compose(ops)


def build_augment_transform() -> transforms.Compose:
    """构造训练专用数据增强流程（RandomCrop + HorizontalFlip + ColorJitter）。

    注意：增强操作在 ToTensor 之前应用于 PIL Image。
    返回的 Compose 在 __getitem__ 中对 [0,1] float32 tensor 生效。
    """
    return transforms.Compose([
        transforms.RandomCrop(96, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.Normalize(mean=STL10_MEAN, std=STL10_STD),
    ])


# ---------------------------------------------------------------------------
# Augmented dataset（内存缓存原始 [0,1] 图像 + __getitem__ 时做随机增强）
# ---------------------------------------------------------------------------


class AugmentedInMemoryDataset(Dataset):
    """基于内存中的原始 [0,1] float32 图像张量构造的 Dataset。

    与 InMemoryDataset 的关键区别：
        - 存储的是未归一化的 [0,1] 图像（仅 ToTensor，无 Normalize）
        - 在 __getitem__ 中动态应用 augmentation + Normalize，每次索引产生不同的随机变换

    Args:
        images: 形状为 (N, C, H, W) 的 float32 张量，值域 [0, 1]。
        labels: 形状为 (N,) 的 long 张量。
        augment: 数据增强 + 归一化 transform（应用于 tensor）。
        indices: 可选索引子集；为 None 时使用全部样本。
    """

    def __init__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        augment: transforms.Compose,
        indices: Sequence[int] | None = None,
    ) -> None:
        assert images.shape[0] == labels.shape[0], "images / labels 数量必须一致"
        self.images = images
        self.labels = labels
        self.augment = augment
        if indices is None:
            self.indices = torch.arange(images.shape[0], dtype=torch.long)
        else:
            self.indices = torch.as_tensor(list(indices), dtype=torch.long)

    def __len__(self) -> int:
        return int(self.indices.numel())

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = int(self.indices[i].item())
        img = self.images[idx]  # [0, 1] float32 tensor
        img = self.augment(img)  # 随机增强 + 归一化
        return img, self.labels[idx]


# ---------------------------------------------------------------------------
# Memory loader
# ---------------------------------------------------------------------------


@dataclass
class LoadedImageFolder:
    """`load_imagefolder_to_memory` 的返回结构。"""

    images: torch.Tensor   # (N, 3, 96, 96) float32
    labels: torch.Tensor   # (N,) long
    classes: list[str]     # 按 class_to_idx 排序的类名列表
    class_to_idx: dict[str, int]


def load_imagefolder_to_memory(
    root: str | os.PathLike,
    normalize: bool = True,
    verbose: bool = True,
) -> LoadedImageFolder:
    """以 ImageFolder 协议读取 root 下全部图像，预处理后整体堆叠到一个张量。

    Args:
        root: 形如 ``STL10/train`` 或 ``STL10/test`` 的目录，下含每类一个子目录。
        normalize: 是否对张量做通道均值方差标准化。
        verbose: 是否打印加载进度。
    """
    root = str(root)
    transform = build_transform(normalize=normalize)
    base = datasets.ImageFolder(root=root, transform=transform)

    n = len(base)
    if n == 0:
        raise RuntimeError(f"目录 '{root}' 下未找到任何图像样本")

    sample_img, _ = base[0]
    c, h, w = sample_img.shape
    images = torch.empty((n, c, h, w), dtype=torch.float32)
    labels = torch.empty((n,), dtype=torch.long)

    log_every = max(1, n // 10)
    for i in range(n):
        img, lbl = base[i]
        images[i] = img
        labels[i] = lbl
        if verbose and (i + 1) % log_every == 0:
            print(f"  [load] {root}: {i + 1}/{n}")

    if verbose:
        mb = images.element_size() * images.numel() / (1024 ** 2)
        print(
            f"  [load] {root}: 完成，N={n}, shape={tuple(images.shape)}, "
            f"占用约 {mb:.1f} MB"
        )

    return LoadedImageFolder(
        images=images,
        labels=labels,
        classes=list(base.classes),
        class_to_idx=dict(base.class_to_idx),
    )


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------


def stratified_split_indices(
    labels: Iterable[int],
    valid_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """对每个类别独立做 (1-valid_ratio)/valid_ratio 的随机切分。

    返回两组互不相交的索引列表（train_indices, valid_indices）。
    """
    if not 0.0 < valid_ratio < 1.0:
        raise ValueError(f"valid_ratio 必须在 (0, 1) 之间，收到 {valid_ratio}")

    labels_arr = np.asarray(list(labels))
    rng = random.Random(seed)

    train_idx: list[int] = []
    valid_idx: list[int] = []
    for cls in np.unique(labels_arr):
        cls_indices = np.where(labels_arr == cls)[0].tolist()
        rng.shuffle(cls_indices)
        n_valid = max(1, int(round(len(cls_indices) * valid_ratio)))
        valid_idx.extend(cls_indices[:n_valid])
        train_idx.extend(cls_indices[n_valid:])

    # 整体再次按 seed 打乱（仅影响样本访问顺序，不影响划分）
    rng.shuffle(train_idx)
    rng.shuffle(valid_idx)
    return train_idx, valid_idx


def _persist_split(
    path: str | os.PathLike,
    classes: list[str],
    train_idx: list[int],
    valid_idx: list[int],
    seed: int,
    valid_ratio: float,
) -> None:
    payload = {
        "seed": seed,
        "valid_ratio": valid_ratio,
        "classes": classes,
        "train_indices": train_idx,
        "valid_indices": valid_idx,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _try_load_split(
    path: str | os.PathLike,
    expected_seed: int,
    expected_valid_ratio: float,
    expected_classes: list[str],
) -> tuple[list[int], list[int]] | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if (
            payload.get("seed") != expected_seed
            or abs(float(payload.get("valid_ratio", -1)) - expected_valid_ratio) > 1e-9
            or payload.get("classes") != expected_classes
        ):
            return None
        return list(payload["train_indices"]), list(payload["valid_indices"])
    except Exception:
        return None


# ---------------------------------------------------------------------------
# DataLoader builders
# ---------------------------------------------------------------------------


def build_augmented_train_valid_loaders(
    train_root: str | os.PathLike,
    *,
    valid_ratio: float = 0.2,
    batch_size: int = 128,
    seed: int = 42,
    num_workers: int = 0,
    normalize: bool = True,
    split_index_path: str | os.PathLike | None = None,
    verbose: bool = True,
) -> tuple[DataLoader, DataLoader, list[str], dict]:
    """构造带数据增强的 train / valid DataLoader。

    - train：使用 AugmentedInMemoryDataset（RandomCrop + HorizontalFlip + ColorJitter）
    - valid：使用 InMemoryDataset（无增强，仅 ToTensor + 可选 Normalize）
    - 划分逻辑与 build_train_valid_loaders 相同（分层 80/20）
    """
    # 加载原始 [0,1] 图像（不做归一化，归一化在 augment transform 中完成）
    loaded = load_imagefolder_to_memory(
        train_root, normalize=False, verbose=verbose
    )

    # 1) 解析或重新生成划分
    train_idx: list[int] | None = None
    valid_idx: list[int] | None = None
    reused = False
    if split_index_path is not None:
        cached = _try_load_split(
            split_index_path,
            expected_seed=seed,
            expected_valid_ratio=valid_ratio,
            expected_classes=loaded.classes,
        )
        if cached is not None:
            train_idx, valid_idx = cached
            reused = True

    if train_idx is None or valid_idx is None:
        train_idx, valid_idx = stratified_split_indices(
            loaded.labels.tolist(), valid_ratio=valid_ratio, seed=seed
        )
        if split_index_path is not None:
            _persist_split(
                split_index_path,
                classes=loaded.classes,
                train_idx=train_idx,
                valid_idx=valid_idx,
                seed=seed,
                valid_ratio=valid_ratio,
            )

    if verbose:
        print(
            f"  [split] reused={reused}, train={len(train_idx)}, valid={len(valid_idx)}, "
            f"valid_ratio={valid_ratio}, seed={seed}"
        )

    # 2) 构造 augment transform
    augment_transform = build_augment_transform()

    # 3) 构造 Dataset：train 用增强版，valid 用预归一化版
    train_ds = AugmentedInMemoryDataset(
        loaded.images, loaded.labels, augment=augment_transform, indices=train_idx
    )

    # valid：需要预先对 loaded.images 做归一化
    if normalize:
        norm = transforms.Normalize(mean=STL10_MEAN, std=STL10_STD)
        valid_images = torch.stack([norm(img) for img in loaded.images])
    else:
        valid_images = loaded.images
    valid_ds = InMemoryDataset(valid_images, loaded.labels, indices=valid_idx)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
        generator=g,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )

    info = {
        "classes": loaded.classes,
        "class_to_idx": loaded.class_to_idx,
        "n_train": len(train_idx),
        "n_valid": len(valid_idx),
        "valid_ratio": valid_ratio,
        "seed": seed,
        "normalize": normalize,
        "augmentation": True,
        "reused_split": reused,
        "split_index_path": str(split_index_path) if split_index_path else None,
    }
    return train_loader, valid_loader, loaded.classes, info


def build_train_valid_loaders(
    train_root: str | os.PathLike,
    *,
    valid_ratio: float = 0.2,
    batch_size: int = 128,
    seed: int = 42,
    num_workers: int = 0,
    normalize: bool = True,
    split_index_path: str | os.PathLike | None = None,
    verbose: bool = True,
) -> tuple[DataLoader, DataLoader, list[str], dict]:
    """构造 train / valid 两个 DataLoader（基于内存缓存 + 分层划分）。

    Args:
        train_root: 形如 ``STL10/train`` 的目录路径。
        valid_ratio: 每类内部划作验证集的比例。
        batch_size: 两个 loader 的批大小。
        seed: 随机种子，控制划分与 train_loader 的 shuffle。
        num_workers: DataLoader worker 数；数据已在内存，建议 0。
        normalize: 是否对图像做通道标准化。
        split_index_path: 划分索引落盘路径；若文件存在且 seed/比例/类别一致则直接复用。

    Returns:
        (train_loader, valid_loader, class_names, info_dict)
    """
    loaded = load_imagefolder_to_memory(train_root, normalize=normalize, verbose=verbose)

    # 1) 解析或重新生成划分
    train_idx: list[int] | None = None
    valid_idx: list[int] | None = None
    reused = False
    if split_index_path is not None:
        cached = _try_load_split(
            split_index_path,
            expected_seed=seed,
            expected_valid_ratio=valid_ratio,
            expected_classes=loaded.classes,
        )
        if cached is not None:
            train_idx, valid_idx = cached
            reused = True

    if train_idx is None or valid_idx is None:
        train_idx, valid_idx = stratified_split_indices(
            loaded.labels.tolist(), valid_ratio=valid_ratio, seed=seed
        )
        if split_index_path is not None:
            _persist_split(
                split_index_path,
                classes=loaded.classes,
                train_idx=train_idx,
                valid_idx=valid_idx,
                seed=seed,
                valid_ratio=valid_ratio,
            )

    if verbose:
        print(
            f"  [split] reused={reused}, train={len(train_idx)}, valid={len(valid_idx)}, "
            f"valid_ratio={valid_ratio}, seed={seed}"
        )

    # 2) 构造 Dataset / DataLoader
    train_ds = InMemoryDataset(loaded.images, loaded.labels, indices=train_idx)
    valid_ds = InMemoryDataset(loaded.images, loaded.labels, indices=valid_idx)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
        generator=g,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )

    info = {
        "classes": loaded.classes,
        "class_to_idx": loaded.class_to_idx,
        "n_train": len(train_idx),
        "n_valid": len(valid_idx),
        "valid_ratio": valid_ratio,
        "seed": seed,
        "normalize": normalize,
        "reused_split": reused,
        "split_index_path": str(split_index_path) if split_index_path else None,
    }
    return train_loader, valid_loader, loaded.classes, info


def build_test_loader(
    test_root: str | os.PathLike,
    *,
    batch_size: int = 128,
    num_workers: int = 0,
    normalize: bool = True,
    verbose: bool = True,
) -> tuple[DataLoader, list[str], dict]:
    """构造 test DataLoader（基于内存缓存）。仅在最终评估阶段调用。"""
    loaded = load_imagefolder_to_memory(test_root, normalize=normalize, verbose=verbose)
    ds = InMemoryDataset(loaded.images, loaded.labels)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )
    info = {
        "classes": loaded.classes,
        "class_to_idx": loaded.class_to_idx,
        "n_test": len(ds),
        "normalize": normalize,
    }
    return loader, loaded.classes, info
