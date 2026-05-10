"""在 STL-10 test 集上评估训练好的模型，输出完整分类报告与混淆矩阵。

使用示例：
    python -m scripts.eval --model models.baseline_cnn.BaselineCNN \
                           --checkpoint outputs/baseline/best_model.pt \
                           --data-root  STL10/test \
                           --output-dir outputs/baseline

合规承诺：
    - 测试集仅在本脚本中读取，且仅用于"前向 + 指标统计"，绝不参与梯度更新。
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
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
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# 允许 `python scripts/eval.py` 与 `python -m scripts.eval` 两种调用方式
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data.stl10_dataset import build_test_loader  # noqa: E402


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate CNN on STL-10 test set; "
                    "produce classification report & confusion matrix."
    )
    parser.add_argument("--model", type=str, default="models.baseline_cnn.BaselineCNN",
                        help="模型类路径，格式 module.ClassName（必须与训练时一致）")
    parser.add_argument("--checkpoint", type=str,
                        default="outputs/baseline/best_model.pt")
    parser.add_argument("--data-root", type=str, default="STL10/test",
                        help="STL-10 测试集目录")
    parser.add_argument("--output-dir", type=str, default="outputs/baseline")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-normalize", action="store_true",
                        help="禁用通道均值方差标准化（必须与训练时保持一致）")
    parser.add_argument("--top-k", type=int, default=3,
                        help="top-k accuracy 中的 k")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def _infer(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """返回 (y_true, y_pred, y_prob, mean_latency_ms_per_image)。"""
    model.eval()
    ys: list[np.ndarray] = []
    ps: list[np.ndarray] = []
    probs: list[np.ndarray] = []

    total_imgs = 0
    total_time = 0.0
    first_batch = True

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels_np = labels.numpy()

            t0 = time.perf_counter()
            logits = model(images)
            t1 = time.perf_counter()

            # 跳过第 1 个 batch 的预热时间，单独累计后续 batch 的耗时
            if not first_batch:
                total_time += (t1 - t0)
                total_imgs += images.size(0)
            first_batch = False

            preds = logits.argmax(dim=1).cpu().numpy()
            prob = F.softmax(logits, dim=1).cpu().numpy()

            ys.append(labels_np)
            ps.append(preds)
            probs.append(prob)

    y_true = np.concatenate(ys, axis=0)
    y_pred = np.concatenate(ps, axis=0)
    y_prob = np.concatenate(probs, axis=0)

    mean_latency_ms = (total_time / max(1, total_imgs)) * 1000.0 if total_imgs > 0 else 0.0
    return y_true, y_pred, y_prob, mean_latency_ms


# ---------------------------------------------------------------------------
# Reports & plots
# ---------------------------------------------------------------------------


def _save_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list[str],
    out_dir: Path,
) -> dict:
    """生成 classification_report.txt / .csv，并返回结构化 dict。"""
    text: str = classification_report(  # type: ignore[assignment]
        y_true, y_pred, target_names=classes, digits=4, zero_division=0,
    )
    (out_dir / "classification_report.txt").write_text(text, encoding="utf-8")

    report_dict: dict = classification_report(  # type: ignore[assignment]
        y_true, y_pred, target_names=classes, digits=4, zero_division=0,
        output_dict=True,
    )

    # 转 CSV
    csv_rows: list[list[str]] = [["class", "precision", "recall", "f1-score", "support"]]
    for cls in classes:
        d = report_dict[cls]
        csv_rows.append([cls, f"{d['precision']:.4f}", f"{d['recall']:.4f}",
                         f"{d['f1-score']:.4f}", str(int(d["support"]))])
    for avg_key in ("macro avg", "weighted avg"):
        d = report_dict[avg_key]
        csv_rows.append([avg_key, f"{d['precision']:.4f}", f"{d['recall']:.4f}",
                         f"{d['f1-score']:.4f}", str(int(d["support"]))])
    total_support = int(sum(int(report_dict[c]["support"]) for c in classes))
    csv_rows.append(["accuracy", "", "", f"{report_dict['accuracy']:.4f}",
                     str(total_support)])
    with open(out_dir / "classification_report.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(csv_rows)

    return report_dict


def _save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list[str],
    out_dir: Path,
) -> np.ndarray:
    """保存混淆矩阵 csv + 两张热力图（绝对计数 / 行归一化）。返回原始计数矩阵。"""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))

    # CSV
    with open(out_dir / "confusion_matrix.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *classes])
        for i, cls in enumerate(classes):
            writer.writerow([cls, *cm[i].tolist()])

    # PNG (counts)
    _plot_confusion_matrix(
        cm, classes,
        title="Confusion Matrix (counts)",
        out_path=out_dir / "confusion_matrix.png",
        fmt="d", normalize=False,
    )

    # PNG (row-normalized)
    cm_norm = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    _plot_confusion_matrix(
        cm_norm, classes,
        title="Confusion Matrix (row-normalized)",
        out_path=out_dir / "confusion_matrix_normalized.png",
        fmt=".2f", normalize=True,
    )

    return cm


def _plot_confusion_matrix(
    matrix: np.ndarray,
    classes: list[str],
    *,
    title: str,
    out_path: Path,
    fmt: str,
    normalize: bool,
) -> None:
    n = len(classes)
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * n + 3), max(5, 0.55 * n + 2)), dpi=130)
    im = ax.imshow(matrix, cmap="Blues", vmin=0,
                   vmax=1.0 if normalize else float(matrix.max() if matrix.size else 1))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    # 在每个单元格内标注数值
    threshold = (matrix.max() if matrix.size else 0) / 2.0 if not normalize else 0.5
    for i in range(n):
        for j in range(n):
            v = matrix[i, j]
            text = format(v, fmt)
            ax.text(j, i, text,
                    ha="center", va="center",
                    color="white" if v > threshold else "black",
                    fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _save_per_class_errors(
    cm: np.ndarray,
    classes: list[str],
    out_dir: Path,
) -> list[dict]:
    """根据混淆矩阵导出每类错误分析。"""
    rows: list[dict] = []
    for i, cls in enumerate(classes):
        support = int(cm[i].sum())
        correct = int(cm[i, i])
        error_count = support - correct
        error_rate = error_count / support if support > 0 else 0.0

        # 该类被混最多的另一类（排除自身）
        confused_with = ""
        confused_count = 0
        if error_count > 0:
            row_copy = cm[i].copy()
            row_copy[i] = -1
            j = int(np.argmax(row_copy))
            confused_with = classes[j]
            confused_count = int(cm[i, j])

        rows.append({
            "class": cls,
            "support": support,
            "correct": correct,
            "error_count": error_count,
            "error_rate": error_rate,
            "top_confused_with": confused_with,
            "top_confused_count": confused_count,
        })

    rows_sorted = sorted(rows, key=lambda r: r["error_rate"], reverse=True)
    with open(out_dir / "per_class_errors.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_sorted[0].keys()))
        writer.writeheader()
        for r in rows_sorted:
            r_out = dict(r)
            r_out["error_rate"] = f"{r_out['error_rate']:.4f}"
            writer.writerow(r_out)
    return rows_sorted


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _load_checkpoint_classes(ckpt: dict, fallback: list[str]) -> list[str]:
    cls = ckpt.get("classes")
    if isinstance(cls, list) and cls:
        return list(cls)
    return list(fallback)


def _resolve_model_cls(model_path: str) -> type[nn.Module]:
    """从 'module.ClassName' 路径动态导入模型类。"""
    parts = model_path.rsplit(".", 1)
    if len(parts) != 2:
        raise ValueError(f"无效的 --model 格式：'{model_path}'，应为 module.ClassName")
    module_name, cls_name = parts
    module = importlib.import_module(module_name)
    return getattr(module, cls_name)


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] device = {device}")

    # 1) 读取测试集（仅此处一次）
    test_loader, classes_dataset, info = build_test_loader(
        test_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        normalize=(not args.no_normalize),
        verbose=True,
    )
    print(f"[eval] test info = {info}")

    # 2) 加载 checkpoint
    ckpt_path = Path(args.checkpoint)
    print(f"[eval] loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    classes = _load_checkpoint_classes(ckpt, fallback=classes_dataset)
    if classes != classes_dataset:
        print(
            f"[eval] WARNING: checkpoint classes != dataset classes; "
            f"using checkpoint order: {classes}"
        )

    ModelCls = _resolve_model_cls(args.model)
    model = ModelCls(num_classes=len(classes)).to(device)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state)

    # 3) 推理
    print("[eval] running inference on test set ...")
    t0 = time.time()
    y_true, y_pred, y_prob, latency_ms = _infer(model, test_loader, device)
    elapsed = time.time() - t0

    # 4) 总体指标
    accuracy = float((y_pred == y_true).mean())
    top_k = max(1, min(args.top_k, len(classes)))
    topk_idx = np.argsort(-y_prob, axis=1)[:, :top_k]
    topk_acc = float(np.mean([y_true[i] in topk_idx[i] for i in range(len(y_true))]))

    macro_p = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_r = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    macro_f = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_p = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    weighted_r = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    weighted_f = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # 5) 报告 / 混淆矩阵 / per-class 错误
    report_dict = _save_classification_report(y_true, y_pred, classes, out_dir)
    cm = _save_confusion_matrix(y_true, y_pred, classes, out_dir)
    per_class_errors = _save_per_class_errors(cm, classes, out_dir)

    summary = {
        "checkpoint": str(ckpt_path),
        "data_root": str(args.data_root),
        "n_test": int(len(y_true)),
        "classes": classes,
        "accuracy": accuracy,
        f"top_{top_k}_accuracy": topk_acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f,
        "weighted_precision": weighted_p,
        "weighted_recall": weighted_r,
        "weighted_f1": weighted_f,
        "mean_inference_latency_ms_per_image": latency_ms,
        "total_eval_seconds": elapsed,
        "best_valid_acc": ckpt.get("valid_acc") if isinstance(ckpt, dict) else None,
        "best_epoch": ckpt.get("epoch") if isinstance(ckpt, dict) else None,
    }
    with open(out_dir / "test_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 6) 控制台总结
    print("\n========== Test Summary ==========")
    print(f"  N_test             : {len(y_true)}")
    print(f"  Accuracy (top-1)   : {accuracy:.4f}")
    print(f"  Top-{top_k} Accuracy : {topk_acc:.4f}")
    print(f"  macro     P/R/F1   : {macro_p:.4f} / {macro_r:.4f} / {macro_f:.4f}")
    print(f"  weighted  P/R/F1   : {weighted_p:.4f} / {weighted_r:.4f} / {weighted_f:.4f}")
    print(f"  Latency / image    : {latency_ms:.2f} ms")
    print(f"  Total eval time    : {elapsed:.1f} s")
    print("==================================\n")

    print("[eval] artifacts saved to:")
    for name in ["classification_report.txt", "classification_report.csv",
                 "confusion_matrix.csv", "confusion_matrix.png",
                 "confusion_matrix_normalized.png",
                 "per_class_errors.csv", "test_summary.json"]:
        print(f"  - {out_dir / name}")

    # 简单的 top-3 易错类提示（便于讨论）
    print("\nTop-3 most error-prone classes:")
    for r in per_class_errors[:3]:
        print(f"  {r['class']:>10s}: error_rate={r['error_rate']:.4f}, "
              f"most confused with '{r['top_confused_with']}' "
              f"({r['top_confused_count']} times)")

    _ = report_dict  # 已落盘，无需进一步使用


if __name__ == "__main__":
    main()
