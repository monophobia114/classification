"""CNN 模型可解释性可视化脚本。

支持三种可视化方法：
    - Grad-CAM：类别激活热力图，揭示模型关注区域
    - Saliency Map：像素级梯度显著图，显示每个像素对分类的影响
    - t-SNE：特征空间降维可视化，观察类别聚类

使用示例：
    # 只生成 Grad-CAM
    python -m scripts.visualize --checkpoint outputs/combined/best_model.pt \
        --data-root STL10/test --output-dir outputs/combined/visualizations

    # 所有方法
    python -m scripts.visualize --checkpoint outputs/combined/best_model.pt \
        --data-root STL10/test --output-dir outputs/combined/visualizations \
        --methods grad-cam,saliency,tsne --samples-per-class 3

合规承诺：
    - 仅对测试集做前向推断，不进行任何训练/梯度更新/超参调优。
"""

from __future__ import annotations

import argparse
import csv
import importlib
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap

# 允许 python scripts/visualize.py 与 python -m scripts.visualize
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data.stl10_dataset import (  # noqa: E402
    STL10_MEAN,
    STL10_STD,
    build_test_loader,
)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

TARGET_CONV_LAYER = "features.8"  # 最后一个 Conv2d (64→128)，输出 (128, 24, 24)
TSNE_FEATURE_LAYER = "classifier.1"  # FC1 输出 (128维)，分类前的最终特征
IMAGE_SIZE = 96
NUM_CLASSES = 10
CLASS_NAMES = [
    "airplane", "bird", "car", "cat", "deer",
    "dog", "horse", "monkey", "ship", "truck",
]


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CNN 模型可解释性可视化：Grad-CAM / Saliency / t-SNE"
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default="outputs/combined/best_model.pt",
        help="模型 checkpoint 路径",
    )
    parser.add_argument(
        "--model", type=str, default="",
        help="模型类路径（如 models.combined_cnn.CombinedCNN）；"
             "留空则从 checkpoint 的 args 字段自动推断",
    )
    parser.add_argument(
        "--data-root", type=str, default="STL10/test",
        help="STL-10 测试集目录",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="outputs/combined/visualizations",
        help="可视化输出目录",
    )
    parser.add_argument(
        "--methods", type=str,
        default="grad-cam,saliency,tsne",
        help="要生成的可视化方法，逗号分隔：grad-cam,saliency,tsne",
    )
    parser.add_argument(
        "--samples-per-class", type=int, default=3,
        help="每类选取的样本数量（仅针对 grad-cam 和 saliency）",
    )
    parser.add_argument(
        "--batch-size", type=int, default=128,
    )
    parser.add_argument(
        "--num-workers", type=int, default=0,
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="控制样本选取与 t-SNE 的随机种子",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 模型加载
# ---------------------------------------------------------------------------

def _load_model(checkpoint_path: str, model_path: str, device: torch.device):
    """加载模型与 checkpoint 权重，返回 (model, classes, checkpoint_info)。"""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # 解析模型类
    if model_path:
        cls_path = model_path
    elif isinstance(ckpt, dict) and "args" in ckpt:
        cls_path = ckpt["args"].get("model", "models.combined_cnn.CombinedCNN")
    else:
        cls_path = "models.combined_cnn.CombinedCNN"

    ModelCls = _resolve_model_cls(cls_path)

    # 解析类别
    if isinstance(ckpt, dict) and "classes" in ckpt:
        classes = list(ckpt["classes"])
    else:
        classes = list(CLASS_NAMES)

    model = ModelCls(num_classes=len(classes)).to(device)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    info = {
        "epoch": ckpt.get("epoch", "?") if isinstance(ckpt, dict) else "?",
        "valid_acc": ckpt.get("valid_acc", None) if isinstance(ckpt, dict) else None,
        "model_cls": cls_path,
    }
    return model, classes, info


def _resolve_model_cls(model_path: str) -> type[nn.Module]:
    parts = model_path.rsplit(".", 1)
    if len(parts) != 2:
        raise ValueError(f"无效的模型路径：'{model_path}'，应为 module.ClassName")
    module_name, cls_name = parts
    module = importlib.import_module(module_name)
    return getattr(module, cls_name)


# ---------------------------------------------------------------------------
# 反归一化（用于可视化显示）
# ---------------------------------------------------------------------------

def _denormalize(tensor: torch.Tensor) -> np.ndarray:
    """将归一化后的图像张量反归一化到 [0, 1]，返回 (H, W, 3) numpy 数组。"""
    t = tensor.detach()  # 断开梯度图，避免 requires_grad 导致后续 .numpy() 失败
    mean = torch.tensor(STL10_MEAN, device=t.device).view(3, 1, 1)
    std = torch.tensor(STL10_STD, device=t.device).view(3, 1, 1)
    img = t * std + mean  # (3, H, W)
    img = torch.clamp(img, 0.0, 1.0)
    img = img.permute(1, 2, 0).cpu().numpy()
    return img


# ===================================================================
# 方法 1：Grad-CAM
# ===================================================================

class GradCAM:
    """Grad-CAM：对指定卷积层的特征图做梯度加权组合。

    Attributes:
        activations: 前向传播时缓存的激活图 (C, H, W)。
        gradients: 反向传播时缓存的梯度 (C, H, W)。
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        # 注册钩子
        self._forward_handle = target_layer.register_forward_hook(self._save_activation)
        self._backward_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()  # (B, C, H, W)

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()  # (B, C, H, W)

    def remove_hooks(self):
        self._forward_handle.remove()
        self._backward_handle.remove()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> np.ndarray:
        """生成 Grad-CAM 热力图。

        Args:
            input_tensor: 单张图像 (1, 3, 96, 96)。
            target_class: 目标类别索引；None 则使用模型预测的类别。

        Returns:
            shape (96, 96) float32 热力图，值域 [0, 1]。
        """
        assert input_tensor.dim() == 4 and input_tensor.size(0) == 1, \
            "input_tensor 必须是 (1, 3, 96, 96)"

        # 清空
        self.activations = None
        self.gradients = None
        self.model.zero_grad()

        # 前向
        input_tensor = input_tensor.requires_grad_(True)
        logits = self.model(input_tensor)  # (1, num_classes)

        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        # 反向（只对目标类）
        score = logits[0, target_class]
        self.model.zero_grad()
        score.backward(retain_graph=False)

        # --- 计算 Grad-CAM ---
        activations = self.activations  # (1, C, H_cam, W_cam)
        gradients = self.gradients      # (1, C, H_cam, W_cam)

        if activations is None or gradients is None:
            raise RuntimeError("钩子未能捕获激活值/梯度，请确认 target_layer 正确")

        # 通道权重 = 梯度的全局平均池化
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # 加权组合 + ReLU
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, H_cam, W_cam)
        cam = F.relu(cam)

        # 上采样到输入尺寸
        cam = F.interpolate(cam, size=(IMAGE_SIZE, IMAGE_SIZE),
                            mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()  # (96, 96)

        # 归一化到 [0, 1]
        cam_max = cam.max()
        if cam_max > 1e-8:
            cam = cam / cam_max
        else:
            cam = np.zeros_like(cam)

        return cam


# ===================================================================
# 方法 2：Saliency Map
# ===================================================================

def generate_saliency_map(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int | None = None,
) -> np.ndarray:
    """生成像素级显著图。

    Args:
        model: 已加载的模型（eval 模式）。
        input_tensor: 单张图像 (1, 3, 96, 96)。
        target_class: 目标类别索引；None 则使用预测类别。

    Returns:
        shape (96, 96) float32 显著图，值域 [0, 1]。
    """
    model.zero_grad()
    input_tensor = input_tensor.detach().requires_grad_(True)

    logits = model(input_tensor)
    if target_class is None:
        target_class = logits.argmax(dim=1).item()

    score = logits[0, target_class]
    model.zero_grad()
    score.backward()

    grad = input_tensor.grad  # (1, 3, 96, 96)
    if grad is None:
        raise RuntimeError("未能获取输入梯度，请检查 requires_grad")

    # 跨通道取最大绝对值
    saliency, _ = grad.abs().max(dim=1)  # (1, 96, 96)
    saliency = saliency.squeeze().cpu().numpy()

    # 归一化
    s_max = saliency.max()
    if s_max > 1e-8:
        saliency = saliency / s_max
    else:
        saliency = np.zeros_like(saliency)

    return saliency


# ===================================================================
# 方法 3：t-SNE 特征空间可视化
# ===================================================================

def _extract_features(
    model: nn.Module,
    loader,
    device: torch.device,
    target_layer_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """从指定层提取全测试集的特征向量。

    Returns:
        features: (N, D) float32 特征矩阵。
        labels:   (N,) long 标签数组。
    """
    # 如果 target_layer_name 包含 '.'，需要遍历获取子模块
    if "." in target_layer_name:
        parts = target_layer_name.split(".")
        layer = model
        for p in parts:
            layer = getattr(layer, p)
    else:
        layer = getattr(model, target_layer_name)

    features_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []

    def hook_fn(module, inp, out):
        features_list.append(out.detach().cpu().numpy())

    handle = layer.register_forward_hook(hook_fn)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            model(images)
            labels_list.append(labels.numpy())

    handle.remove()

    features = np.concatenate(features_list, axis=0)  # (N, D)
    labels = np.concatenate(labels_list, axis=0)       # (N,)
    return features, labels


def generate_tsne(
    model: nn.Module,
    loader,
    device: torch.device,
    classes: list[str],
    output_dir: Path,
    seed: int = 42,
):
    """生成 t-SNE 降维散点图。"""
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("[tsne] ERROR: 需要 scikit-learn。pip install scikit-learn")
        return

    print("[tsne] 提取特征向量 ...")
    t0 = time.time()
    features, labels = _extract_features(model, loader, device, TSNE_FEATURE_LAYER)
    print(f"[tsne] 特征提取完成：{features.shape}，耗时 {time.time() - t0:.1f}s")

    # 如果特征维度 > 256，先用 PCA 降维（加速 t-SNE）
    if features.shape[1] > 256:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=128, random_state=seed)
        features = pca.fit_transform(features)
        print(f"[tsne] PCA: {features.shape[1]} dims")

    print(f"[tsne] 运行 t-SNE（可能需要数分钟）...")
    t0 = time.time()
    tsne = TSNE(n_components=2, random_state=seed, perplexity=30, max_iter=1000,
                init="pca", learning_rate="auto")
    embedding = tsne.fit_transform(features)
    print(f"[tsne] t-SNE 完成，耗时 {time.time() - t0:.1f}s")

    # 绘图
    fig, ax = plt.subplots(figsize=(12, 9), dpi=150)
    cmap = plt.colormaps.get_cmap("tab10")
    colors = [cmap(i) for i in range(len(classes))]

    for i, cls_name in enumerate(classes):
        mask = labels == i
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=[colors[i]], label=cls_name,
            s=22, alpha=0.85, edgecolors="black", linewidths=0.3,
        )

    ax.set_title("t-SNE Feature Space Visualization\n"
                 f"(feature layer: {TSNE_FEATURE_LAYER})", fontsize=14, fontweight="bold")
    ax.set_xlabel("t-SNE dimension 1", fontsize=11)
    ax.set_ylabel("t-SNE dimension 2", fontsize=11)
    ax.legend(markerscale=1.5, fontsize=10, loc="lower right",
              ncol=2 if len(classes) > 5 else 1, framealpha=0.9)

    fig.tight_layout()
    out_path = output_dir / "tsne_feature_space.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[tsne] 散点图已保存：{out_path}")

    # 同时保存嵌入坐标 CSV（方便后续自定义分析）
    csv_path = output_dir / "tsne_coordinates.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_name", "class_id", "x", "y"])
        for i in range(len(labels)):
            writer.writerow([
                classes[labels[i]], int(labels[i]),
                f"{embedding[i, 0]:.6f}", f"{embedding[i, 1]:.6f}",
            ])
    print(f"[tsne] 坐标 CSV 已保存：{csv_path}")

    # ---- 量化指标 ----
    print(f"[tsne] 计算聚类质量指标 ...")
    _compute_cluster_metrics(
        embedding, labels, classes, output_dir, "tsne_embedding",
    )
    # 同时在原始特征空间计算（更有参考价值）
    _compute_cluster_metrics(
        features, labels, classes, output_dir, "original_features",
    )


# ===================================================================
# 聚类质量量化指标
# ===================================================================

def _compute_cluster_metrics(
    points: np.ndarray,            # (N, D) 特征矩阵
    labels: np.ndarray,            # (N,) 标签
    classes: list[str],
    output_dir: Path,
    prefix: str,
) -> None:
    """计算每类的聚类紧密度、分离度和全局指标，写入 CSV。

    指标包括：
        - **每类质心**
        - **类内紧密度**（intra-class compactness）：每点到类质心的平均距离
        - **类间分离度**（inter-class separation）：本类质心到最近邻类质心的距离
        - **最近邻类**：与本类质心最近的另一类
        - **全局指标**：Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index
    """
    from sklearn.metrics import (
        silhouette_score,
        davies_bouldin_score,
        calinski_harabasz_score,
    )
    from scipy.spatial.distance import cdist

    n_classes = len(classes)
    # 计算每类质心
    centroids = np.zeros((n_classes, points.shape[1]), dtype=np.float64)
    intra_dist = np.zeros(n_classes, dtype=np.float64)
    for i in range(n_classes):
        mask = labels == i
        if mask.sum() == 0:
            centroids[i] = np.nan
            intra_dist[i] = np.nan
            continue
        cls_points = points[mask]
        centroids[i] = cls_points.mean(axis=0)
        intra_dist[i] = np.mean(np.linalg.norm(cls_points - centroids[i], axis=1))

    # 质心间距离矩阵 + 最近邻类
    centroid_dist = cdist(centroids, centroids, metric="euclidean")
    np.fill_diagonal(centroid_dist, np.inf)
    nearest_neighbor = np.argmin(centroid_dist, axis=1)  # (n_classes,)
    inter_separation = np.min(centroid_dist, axis=1)      # 到最近邻质心的距离

    # 全局指标（仅用无 nan 的类）
    valid_mask = ~np.isnan(intra_dist)
    valid_indices = np.where(valid_mask)[0]
    if valid_mask.sum() >= 2:
        valid_labels = labels[np.isin(labels, valid_indices)]
        valid_points = points[np.isin(labels, valid_indices)]
        try:
            sil = silhouette_score(valid_points, valid_labels, sample_size=min(500, len(valid_labels)),
                                   random_state=42)
        except Exception:
            sil = float("nan")
        try:
            db = davies_bouldin_score(valid_points, valid_labels)
        except Exception:
            db = float("nan")
        try:
            ch = calinski_harabasz_score(valid_points, valid_labels)
        except Exception:
            ch = float("nan")
    else:
        sil = float("nan")
        db = float("nan")
        ch = float("nan")

    # 写入 CSV
    csv_path = output_dir / f"{prefix}_cluster_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "class", "intra_compactness", "inter_separation",
            "nearest_neighbor_class", "nearest_neighbor_distance",
            "separation_to_compactness_ratio",
        ])
        for i, cls_name in enumerate(classes):
            nn_idx = int(nearest_neighbor[i])
            nn_name = classes[nn_idx] if nn_idx < len(classes) else "?"
            ratio = float(inter_separation[i]) / max(float(intra_dist[i]), 1e-12)
            writer.writerow([
                cls_name,
                f"{intra_dist[i]:.6f}",
                f"{inter_separation[i]:.6f}",
                nn_name,
                f"{centroid_dist[i, nn_idx]:.6f}",
                f"{ratio:.4f}",
            ])

        writer.writerow([])
        writer.writerow(["global_metric", "value", "note", "", "", ""])
        writer.writerow(["Silhouette Score", f"{sil:.4f}", "[-1,1], 越高越好", "", "", ""])
        writer.writerow(["Davies-Bouldin Index", f"{db:.4f}", ">=0, 越低越好", "", "", ""])
        writer.writerow(["Calinski-Harabasz Index", f"{ch:.2f}", "越高越好", "", "", ""])

    print(f"[tsne] {prefix}_cluster_metrics.csv 已保存")
    print(f"         Silhouette = {sil:.4f} | DB = {db:.4f} | CH = {ch:.2f}")


# ===================================================================
# 样本选取
# ===================================================================

def _select_samples(
    model: nn.Module,
    loader,
    device: torch.device,
    classes: list[str],
    per_class: int,
    seed: int,
) -> dict[str, list[dict]]:
    """从测试集中选取样本。

    - corrected: 每类选取 per_class 张正确分类的样本
    - misclassified: 每类选取 max(1, per_class // 2) 张错误分类的样本（如有）

    Returns:
        {
            "corrected": [
                {"image": (3, 96, 96) tensor, "label": int, "pred": int, "idx": int},
                ...
            ],
            "misclassified": [...],
        }
    """
    rng = np.random.RandomState(seed)

    # 收集所有预测结果
    all_images: list[torch.Tensor] = []
    all_labels: list[int] = []
    all_preds: list[int] = []

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu()

            all_images.extend([img.cpu() for img in images])
            all_labels.extend(labels.tolist())
            all_preds.extend(preds.tolist())

    n = len(all_labels)
    all_correct = [all_preds[i] == all_labels[i] for i in range(n)]

    # 按类别组织
    corrected: list[dict] = []
    misclassified: list[dict] = []

    for cls_id in range(len(classes)):
        cls_indices = [i for i in range(n) if all_labels[i] == cls_id]

        # 正确分类的
        cls_correct = [i for i in cls_indices if all_correct[i]]
        rng.shuffle(cls_correct)
        for idx in cls_correct[:per_class]:
            corrected.append({
                "image": all_images[idx],
                "label": all_labels[idx],
                "pred": all_preds[idx],
                "class_name": classes[cls_id],
                "pred_name": classes[all_preds[idx]],
            })

        # 错误分类的
        cls_wrong = [i for i in cls_indices if not all_correct[i]]
        rng.shuffle(cls_wrong)
        n_wrong = max(1, per_class // 2)
        for idx in cls_wrong[:n_wrong]:
            misclassified.append({
                "image": all_images[idx],
                "label": all_labels[idx],
                "pred": all_preds[idx],
                "class_name": classes[cls_id],
                "pred_name": classes[all_preds[idx]],
            })

    return {"corrected": corrected, "misclassified": misclassified}


# ===================================================================
# 可视化绘制
# ===================================================================

def _create_heatmap_overlay(
    image_np: np.ndarray,  # (H, W, 3) float32 [0, 1]
    heatmap: np.ndarray,    # (H, W) float32 [0, 1]
    alpha: float = 0.5,
) -> np.ndarray:
    """将热力图叠加到原图上。

    Returns:
        (H, W, 3) float32 [0, 1] 叠加后的图像。
    """
    # 将热力图映射为 jet 颜色
    cmap = plt.cm.get_cmap("jet")
    heatmap_colored = cmap(heatmap)[:, :, :3]  # (H, W, 3) [0, 1]

    # Alpha 混合
    overlay = (1 - alpha) * image_np + alpha * heatmap_colored
    overlay = np.clip(overlay, 0.0, 1.0)
    return overlay


def _save_heatmap_grid(
    samples: list[dict],
    model: nn.Module,
    grad_cam: GradCAM,
    device: torch.device,
    output_dir: Path,
    filename_prefix: str,
    title: str,
    method: str = "gradcam",
):
    """生成多张热力图并拼成网格保存。

    For each sample, generates:
    - Original image
    - Grad-CAM (or Saliency) heatmap overlay

    Saves individual images + a summary grid.
    """
    if not samples:
        print(f"[{method}] {title}: 无可用样本，跳过")
        return

    n_samples = len(samples)
    # 网格：每行 3 列（原图 | 热力图叠加 | 热力图纯色）
    n_cols = 3
    n_rows = n_samples

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(12, 4 * n_rows),
        dpi=120,
        squeeze=False,
    )

    for i, sample in enumerate(samples):
        img_tensor = sample["image"].unsqueeze(0).to(device)  # (1, 3, 96, 96)
        true_label = sample["class_name"]
        pred_label = sample["pred_name"]
        img_np = _denormalize(sample["image"])  # (96, 96, 3) display

        # 生成热力图
        if method == "gradcam":
            heatmap = grad_cam.generate(img_tensor, target_class=sample["label"])
            heatmap_title = "Grad-CAM"
        else:
            heatmap = generate_saliency_map(model, img_tensor, target_class=sample["label"])
            heatmap_title = "Saliency"

        overlay_np = _create_heatmap_overlay(img_np, heatmap)

        # 列 0：原图
        ax = axes[i, 0]
        ax.imshow(img_np)
        ax.set_title(f"{true_label} (true: {true_label})", fontsize=9)
        ax.axis("off")

        # 列 1：叠加图
        ax = axes[i, 1]
        ax.imshow(overlay_np)
        ax.set_title(f"{heatmap_title} → pred: {pred_label}", fontsize=9)
        ax.axis("off")

        # 列 2：纯热力图
        ax = axes[i, 2]
        ax.imshow(heatmap, cmap="jet", vmin=0, vmax=1)
        ax.set_title(f"{heatmap_title} (raw)", fontsize=9)
        ax.axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    grid_path = output_dir / f"{filename_prefix}_grid.png"
    fig.savefig(grid_path, dpi=120)
    plt.close(fig)
    print(f"[{method}] 网格图已保存：{grid_path} ({n_samples} 样本)")

    # 同时保存单张放大图（便于仔细分析）
    for i, sample in enumerate(samples):
        img_tensor = sample["image"].unsqueeze(0).to(device)
        img_np = _denormalize(sample["image"])
        true_label = sample["class_name"]
        pred_label = sample["pred_name"]

        if method == "gradcam":
            heatmap = grad_cam.generate(img_tensor, target_class=sample["label"])
        else:
            heatmap = generate_saliency_map(model, img_tensor, target_class=sample["label"])

        overlay_np = _create_heatmap_overlay(img_np, heatmap)

        # 2×1：原图 + 叠加图
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), dpi=120)
        ax1.imshow(img_np)
        ax1.set_title(f"True: {true_label}", fontsize=10)
        ax1.axis("off")
        ax2.imshow(overlay_np)
        ax2.set_title(f"Pred: {pred_label} | {method.upper()}", fontsize=10)
        ax2.axis("off")
        fig2.tight_layout()

        single_path = output_dir / f"{filename_prefix}_sample{i+1}_{true_label}.png"
        fig2.savefig(single_path, dpi=120)
        plt.close(fig2)


# ===================================================================
# 主入口
# ===================================================================

def main() -> None:
    args = _parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[visualize] device = {device}")

    methods = [m.strip().lower() for m in args.methods.split(",")]

    # ---- 1) 加载模型 ----
    print(f"[visualize] 加载模型：{args.checkpoint}")
    model, classes, ckpt_info = _load_model(args.checkpoint, args.model, device)
    model.eval()
    print(f"[visualize] model_cls={ckpt_info['model_cls']}, "
          f"epoch={ckpt_info['epoch']}, valid_acc={ckpt_info['valid_acc']}")
    print(f"[visualize] classes={classes}")

    # ---- 2) 加载测试集 ----
    print(f"[visualize] 加载测试集：{args.data_root}")
    test_loader, ds_classes, ds_info = build_test_loader(
        test_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        normalize=True,
        verbose=True,
    )
    print(f"[visualize] 测试集信息：{ds_info}")

    # ---- 3) Grad-CAM & Saliency ----
    need_samples = any(m in methods for m in ("grad-cam", "saliency"))
    samples: dict | None = None

    if need_samples:
        print(f"[visualize] 选取样本（每类 {args.samples_per_class} 张）...")
        samples = _select_samples(
            model, test_loader, device, classes,
            per_class=args.samples_per_class, seed=args.seed,
        )
        n_correct = len(samples["corrected"])
        n_wrong = len(samples["misclassified"])
        print(f"[visualize] 正确分类：{n_correct}，错误分类：{n_wrong}")

    # Grad-CAM
    if "grad-cam" in methods:
        print("[grad-cam] 初始化 ...")
        # 解析目标层
        target_layer = model
        for part in TARGET_CONV_LAYER.split("."):
            target_layer = getattr(target_layer, part)
        print(f"[grad-cam] 目标层：{TARGET_CONV_LAYER} ({type(target_layer).__name__})")

        gc = GradCAM(model, target_layer)

        try:
            if samples:
                _save_heatmap_grid(
                    samples["corrected"], model, gc, device, output_dir,
                    "gradcam_corrected", "Grad-CAM: Correctly Classified Samples",
                    method="gradcam",
                )
                _save_heatmap_grid(
                    samples["misclassified"], model, gc, device, output_dir,
                    "gradcam_misclassified", "Grad-CAM: Misclassified Samples",
                    method="gradcam",
                )
        finally:
            gc.remove_hooks()

    # Saliency Map
    if "saliency" in methods:
        print("[saliency] 生成 ...")
        if samples:
            _save_heatmap_grid(
                samples["corrected"], model, None, device, output_dir,
                "saliency_corrected", "Saliency Map: Correctly Classified Samples",
                method="saliency",
            )
            _save_heatmap_grid(
                samples["misclassified"], model, None, device, output_dir,
                "saliency_misclassified", "Saliency Map: Misclassified Samples",
                method="saliency",
            )

    # t-SNE
    if "tsne" in methods:
        print("[tsne] 生成 ...")
        generate_tsne(model, test_loader, device, classes, output_dir, seed=args.seed)

    print(f"\n[visualize] 全部完成。输出目录：{output_dir.resolve()}")
    print("[visualize] 生成的文件：")
    for f in sorted(output_dir.iterdir()):
        if f.is_file():
            size_kb = f.stat().st_size / 1024
            print(f"  {f.name} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
