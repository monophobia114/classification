"""根据训练阶段生成的 metrics.csv 绘制 loss / accuracy 曲线。

使用示例：
    python -m scripts.plot_curves --metrics-csv outputs/baseline/metrics.csv \
                                  --output-dir  outputs/baseline
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无显示设备时也能保存图片
import matplotlib.pyplot as plt


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot train / valid loss & accuracy curves from metrics.csv."
    )
    parser.add_argument("--metrics-csv", type=str, default="outputs/baseline/metrics.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/baseline")
    parser.add_argument("--title-prefix", type=str, default="BaselineCNN")
    return parser.parse_args()


def _load_metrics(path: str) -> dict[str, list[float]]:
    cols = {"epoch": [], "train_loss": [], "train_acc": [],
            "valid_loss": [], "valid_acc": [], "lr": [], "elapsed_sec": []}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k in cols:
                cols[k].append(float(row[k]))
    return cols


def _plot_loss(ax, m: dict[str, list[float]], title: str) -> None:
    ax.plot(m["epoch"], m["train_loss"], "-",  label="train", color="#1f77b4")
    ax.plot(m["epoch"], m["valid_loss"], "--", label="valid", color="#d62728")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (cross-entropy)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()


def _plot_acc(ax, m: dict[str, list[float]], title: str) -> None:
    ax.plot(m["epoch"], m["train_acc"], "-",  label="train", color="#1f77b4")
    ax.plot(m["epoch"], m["valid_acc"], "--", label="valid", color="#d62728")
    # 标注 valid 最高点
    if m["valid_acc"]:
        best_i = max(range(len(m["valid_acc"])), key=lambda i: m["valid_acc"][i])
        best_ep = m["epoch"][best_i]
        best_acc = m["valid_acc"][best_i]
        ax.scatter([best_ep], [best_acc], s=60, color="red", zorder=5,
                   label=f"best valid={best_acc:.4f} @ ep {int(best_ep)}")
        ax.annotate(f"{best_acc:.4f}",
                    xy=(best_ep, best_acc),
                    xytext=(5, -12), textcoords="offset points",
                    fontsize=9, color="red")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()


def main() -> None:
    args = _parse_args()
    csv_path = Path(args.metrics_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"metrics csv not found: {csv_path}")

    m = _load_metrics(str(csv_path))
    if not m["epoch"]:
        raise RuntimeError(f"no rows found in {csv_path}")

    # 1) Loss only
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=120)
    _plot_loss(ax, m, f"{args.title_prefix} — Train/Valid Loss")
    fig.tight_layout()
    fig.savefig(out_dir / "curve_loss.png")
    plt.close(fig)

    # 2) Accuracy only
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=120)
    _plot_acc(ax, m, f"{args.title_prefix} — Train/Valid Accuracy")
    fig.tight_layout()
    fig.savefig(out_dir / "curve_acc.png")
    plt.close(fig)

    # 3) Combined side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=120)
    _plot_loss(axes[0], m, "Train/Valid Loss")
    _plot_acc(axes[1], m, "Train/Valid Accuracy")
    fig.suptitle(args.title_prefix)
    fig.tight_layout()
    fig.savefig(out_dir / "curves.png")
    plt.close(fig)

    print(f"[plot_curves] saved:")
    print(f"  - {out_dir / 'curve_loss.png'}")
    print(f"  - {out_dir / 'curve_acc.png'}")
    print(f"  - {out_dir / 'curves.png'}")


if __name__ == "__main__":
    main()
