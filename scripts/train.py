"""BaselineCNN 训练脚本（CPU 优化版）。

使用示例：
    python -m scripts.train --epochs 40 --batch-size 128 --lr 0.01 --momentum 0.9 \
                            --optimizer sgd --early-stop-patience 8 \
                            --output-dir outputs/baseline --seed 42

合规承诺：
    - 严禁使用 Dropout / BatchNorm / weight_decay / 数据增强 / label smoothing
    - 严禁在训练 / 验证过程中接触 STL10/test/ 下任何数据
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# 允许 `python scripts/train.py` 与 `python -m scripts.train` 两种调用方式
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data.stl10_dataset import build_train_valid_loaders  # noqa: E402
from models.baseline_cnn import BaselineCNN, count_parameters  # noqa: E402


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train BaselineCNN on STL-10 (no Dropout / BN, CPU-friendly)."
    )
    parser.add_argument("--data-root", type=str, default="STL10/train",
                        help="STL-10 训练集目录（仅训练 + 验证使用，不含 test）")
    parser.add_argument("--output-dir", type=str, default="outputs/baseline",
                        help="所有训练产物的输出根目录")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="SGD 动量；--optimizer adam 时忽略")
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam"],
                        help="baseline 默认使用 SGD；Adam 留作后续对比实验")
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="数据已在内存，建议 0（Windows 友好）")
    parser.add_argument("--no-normalize", action="store_true",
                        help="禁用通道均值方差标准化（仅 ToTensor）")
    parser.add_argument("--early-stop-patience", type=int, default=8,
                        help="valid_acc 连续未刷新最高值的最大 epoch 数；<=0 关闭早停")
    parser.add_argument("--threads", type=int, default=0,
                        help="torch.set_num_threads 的取值；0=使用 os.cpu_count()")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Reproducibility & logging
# ---------------------------------------------------------------------------


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------


def _run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    train: bool,
) -> tuple[float, float]:
    """运行一个 epoch，返回 (avg_loss, accuracy)。

    train=True 时执行反向传播；否则仅做前向 + 指标累计。
    """
    model.train(mode=train)
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device, non_blocking=False)
            labels = labels.to(device, non_blocking=False)

            logits = model(images)
            loss = criterion(logits, labels)

            if train:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            preds = logits.argmax(dim=1)
            bs = labels.size(0)
            total_loss += float(loss.detach().item()) * bs
            total_correct += int((preds == labels).sum().item())
            total_count += bs

    avg_loss = total_loss / max(1, total_count)
    accuracy = total_correct / max(1, total_count)
    return avg_loss, accuracy


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = _setup_logger(output_dir / "train.log")
    logger.info("=" * 70)
    logger.info("BaselineCNN training (no Dropout / BN, CPU-friendly)")
    logger.info("=" * 70)
    logger.info("args = %s", vars(args))

    # ---- 1) 复现性与线程 ----
    _set_seed(args.seed)
    threads = args.threads if args.threads > 0 else (os.cpu_count() or 1)
    torch.set_num_threads(threads)
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))
    logger.info("torch.get_num_threads() = %d", torch.get_num_threads())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device = %s, torch = %s, python = %s, platform = %s",
                device, torch.__version__, platform.python_version(), platform.platform())

    # ---- 2) 数据 ----
    logger.info("loading dataset into memory ...")
    t0 = time.time()
    train_loader, valid_loader, classes, split_info = build_train_valid_loaders(
        train_root=args.data_root,
        valid_ratio=args.valid_ratio,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        normalize=(not args.no_normalize),
        split_index_path=output_dir / "split_index.json",
        verbose=True,
    )
    logger.info("dataset ready in %.1fs | classes=%s", time.time() - t0, classes)
    logger.info("split_info = %s", split_info)

    # ---- 3) 模型 / 损失 / 优化器 ----
    model = BaselineCNN(num_classes=len(classes)).to(device)
    n_params = count_parameters(model)
    logger.info("model = BaselineCNN | trainable params = %s", f"{n_params:,}")

    criterion = nn.CrossEntropyLoss()
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=0.0,  # 显式禁用 L2 正则
        )
    else:  # adam
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=0.0
        )
    logger.info("optimizer = %s", optimizer)

    # ---- 4) 持久化训练配置 ----
    config = {
        "args": vars(args),
        "classes": classes,
        "n_params": n_params,
        "split_info": split_info,
        "device": str(device),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "threads": threads,
    }
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    # ---- 5) metrics.csv 表头 ----
    metrics_path = output_dir / "metrics.csv"
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc",
                         "valid_loss", "valid_acc", "lr", "elapsed_sec"])

    # ---- 6) 训练循环 ----
    best_valid_acc = -1.0
    best_epoch = -1
    no_improve = 0
    early_stop_patience = args.early_stop_patience

    best_ckpt_path = output_dir / "best_model.pt"
    last_ckpt_path = output_dir / "last_model.pt"

    logger.info("start training: epochs=%d, batch_size=%d", args.epochs, args.batch_size)
    overall_t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        ep_t0 = time.time()

        train_loss, train_acc = _run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        valid_loss, valid_acc = _run_epoch(
            model, valid_loader, criterion, None, device, train=False
        )

        elapsed = time.time() - ep_t0
        cur_lr = optimizer.param_groups[0]["lr"]

        with open(metrics_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                epoch,
                f"{train_loss:.6f}", f"{train_acc:.6f}",
                f"{valid_loss:.6f}", f"{valid_acc:.6f}",
                f"{cur_lr:.6f}", f"{elapsed:.2f}",
            ])

        improved = valid_acc > best_valid_acc + 1e-9
        marker = "  *new best*" if improved else ""
        logger.info(
            "epoch %3d/%d | train loss=%.4f acc=%.4f | valid loss=%.4f acc=%.4f | "
            "lr=%.4g | %.1fs%s",
            epoch, args.epochs, train_loss, train_acc, valid_loss, valid_acc,
            cur_lr, elapsed, marker,
        )

        # 保存 last
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "valid_acc": valid_acc,
            "classes": classes,
            "args": vars(args),
        }, last_ckpt_path)

        if improved:
            best_valid_acc = valid_acc
            best_epoch = epoch
            no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "valid_acc": valid_acc,
                "classes": classes,
                "args": vars(args),
            }, best_ckpt_path)
        else:
            no_improve += 1

        if early_stop_patience > 0 and no_improve >= early_stop_patience:
            logger.info(
                "early stopping triggered: valid_acc not improving for %d epochs "
                "(best=%.4f @ epoch %d)",
                no_improve, best_valid_acc, best_epoch,
            )
            break

    total = time.time() - overall_t0
    logger.info(
        "training done in %.1fs | best valid_acc=%.4f @ epoch %d | "
        "best ckpt: %s",
        total, best_valid_acc, best_epoch, best_ckpt_path,
    )

    # 在 config.json 中追加 best 信息，便于后续 eval 读取
    config["best_valid_acc"] = best_valid_acc
    config["best_epoch"] = best_epoch
    config["total_train_seconds"] = total
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
