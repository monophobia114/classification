"""集大成组合模型。

综合以下改进措施：
    - 激活函数：Tanh（替换 ReLU）
    - 归一化：BatchNorm2d（每个 Conv 后、Tanh 前）
    - 正则化：Dropout(p=0.3)（FC1 → Tanh 后）
    - 池化：保持 MaxPool2d 不变
    - 权重初始化：Xavier（适配 Tanh）

训练时配合：
    - 优化器：Adam (lr=0.001)
    - 数据增强：RandomCrop + HorizontalFlip + ColorJitter (--use-augmentation)
    - 学习率调度：CosineAnnealingLR (--cosine-lr)
    - Epochs：80，关闭早停 (--early-stop-patience 0)

架构：
    Conv2d → BatchNorm2d → Tanh → MaxPool2d  (×3)
    → MaxPool2d → Flatten
    → FC1 → Tanh → Dropout(0.3) → FC2 (logits)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CombinedCNN(nn.Module):
    """集大成模型：BatchNorm + Tanh + Dropout + MaxPool。"""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 额外下采样
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 128),
            nn.Tanh(),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Xavier 初始化，适配 Tanh 激活函数。"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x
