"""STL-10 baseline CNN 模型定义。

设计要点（详见 plans/baseline_cnn_design.md）：
    - 输入：RGB 图像，形状 (N, 3, 96, 96)
    - 3 个卷积层 + 4 个最大池化层 + 2 个全连接层
    - 严格不含任何正则化 / 归一化（无 Dropout、无 BatchNorm、无 weight decay）
    - 总参数量约 0.69 M，CPU 训练友好

逐层张量形状推导：
    Input              : (N, 3,   96, 96)
    Conv1 (3 -> 32)    : (N, 32,  96, 96)
    ReLU + MaxPool 2x2 : (N, 32,  48, 48)
    Conv2 (32 -> 64)   : (N, 64,  48, 48)
    ReLU + MaxPool 2x2 : (N, 64,  24, 24)
    Conv3 (64 -> 128)  : (N, 128, 24, 24)
    ReLU + MaxPool 2x2 : (N, 128, 12, 12)
    MaxPool 2x2 (压参) : (N, 128,  6,  6)
    Flatten            : (N, 4608)
    FC1 (4608 -> 128)  : (N, 128)
    ReLU
    FC2 (128 -> 10)    : (N, 10)   # logits
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    """STL-10 baseline CNN（无 Dropout / 无 BatchNorm）。"""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        # ------- 特征提取：3 个卷积层 + 4 个最大池化层 -------
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels=3, out_channels=32,
                      kernel_size=3, stride=1, padding=1),  # -> (32, 96, 96)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),          # -> (32, 48, 48)

            # Block 2
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=3, stride=1, padding=1),  # -> (64, 48, 48)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),          # -> (64, 24, 24)

            # Block 3
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=3, stride=1, padding=1),  # -> (128, 24, 24)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),          # -> (128, 12, 12)

            # 额外的下采样：进一步把空间尺寸压到 6x6，以减小后续 FC1 的参数量
            nn.MaxPool2d(kernel_size=2, stride=2),          # -> (128, 6, 6)
        )

        # ------- 分类头：2 个全连接层 -------
        self.classifier = nn.Sequential(
            nn.Flatten(),                                   # -> (N, 4608)
            nn.Linear(in_features=128 * 6 * 6, out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=num_classes),  # logits
        )

        # 显式初始化（不属于正则化 / 归一化范畴，仅是参数初始值设定）
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """采用 Kaiming 初始化，与 ReLU 激活函数匹配，便于无 BN 时仍能稳定训练。"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 形状为 (N, 3, 96, 96) 的输入张量。

        Returns:
            形状为 (N, num_classes) 的 logits 张量（未经过 softmax）。
        """
        x = self.features(x)
        x = self.classifier(x)
        return x


def count_parameters(model: nn.Module) -> int:
    """统计模型可训练参数总数。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 简单的形状与参数量自检：python -m models.baseline_cnn
    model = BaselineCNN(num_classes=10)
    dummy = torch.randn(2, 3, 96, 96)
    logits = model(dummy)
    print(model)
    print(f"\nInput  shape: {tuple(dummy.shape)}")
    print(f"Output shape: {tuple(logits.shape)}  (expected: (2, 10))")
    print(f"Trainable parameters: {count_parameters(model):,}")
