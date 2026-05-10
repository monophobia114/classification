"""Dropout 正则化实验模型。

在 BaselineCNN 的 FC1 → ReLU 之后添加 Dropout(p=0.5)，以抑制过拟合。

对比重点（vs baseline）：
    - 过拟合改善：valid_loss 在 epoch 10+ 后是否不再持续上升
    - train/valid accuracy 差距是否缩小
    - test_acc 是否提升
    - best epoch 是否后移（Dropout 通常需要更多 epoch 收敛）
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DropoutCNN(nn.Module):
    """在 FC1 后添加 Dropout 的 CNN。

    3 Conv + 4 MaxPool + 2 FC，FC1→ReLU→Dropout(0.5)→FC2。
    无 BatchNorm。
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 额外下采样
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Kaiming 初始化，与 ReLU 匹配。"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x
