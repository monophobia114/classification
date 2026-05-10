# STL-10 BaselineCNN 实验

> 基础卷积神经网络 baseline（**无 Dropout / 无 BatchNorm / 无 weight_decay / 无数据增强**）。
> 模型设计见 [`plans/baseline_cnn_design.md`](plans/baseline_cnn_design.md)；实验方案见 [`plans/baseline_experiment_plan.md`](plans/baseline_experiment_plan.md)。

## 目录结构

```
classification/
├── STL10/                          # 数据集（不可修改）
│   ├── train/<class>/*.png         # 训练 + 验证使用（每类 700 张，共 7 000）
│   └── test/<class>/*.png          # 仅最终测试，训练全程不可见
├── models/
│   └── baseline_cnn.py             # BaselineCNN 模型定义（≈ 0.69M 参数）
├── data/
│   └── stl10_dataset.py            # 内存缓存数据集 + 分层 80/20 切分
├── scripts/
│   ├── train.py                    # 训练 + 早停 + best/last 权重 + metrics.csv
│   ├── plot_curves.py              # loss / acc 曲线
│   └── eval.py                     # test 集分类报告 + 混淆矩阵
├── plans/
│   ├── baseline_cnn_design.md
│   └── baseline_experiment_plan.md
└── outputs/baseline/               # 训练 + 评估全部产物落到这里
```

## 运行环境

- Python ≥ 3.10
- PyTorch（CPU 版即可）：`pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`
- 其他依赖：`pip install scikit-learn matplotlib numpy`

本机已验证可用版本：torch 2.11.0+cpu、torchvision 0.26.0+cpu、scikit-learn 1.8、matplotlib 3.10、numpy 2.4。

## 三步运行

### 第 1 步：训练

```bash
python -m scripts.train --epochs 40 --batch-size 128 --lr 0.01 --momentum 0.9 --optimizer sgd --early-stop-patience 8 --output-dir outputs/baseline --seed 42
```

> Windows cmd 下行末用 `^`，PowerShell 下用反引号 `` ` ``，bash 下用 `\`。也可全部写成一行。

产物（落到 `outputs/baseline/`）：
- `split_index.json` —— 80/20 train/valid 划分索引（首次生成，复用一致）
- `config.json` —— 训练配置 + best 信息
- `train.log` —— 完整训练日志
- `metrics.csv` —— 每 epoch 的 train/valid loss & acc & lr & 用时
- `best_model.pt` —— **valid_acc 最高的权重**（用于 eval）
- `last_model.pt` —— 最后一个 epoch 的权重（备份）

参考速度（16 线程 CPU 实测）：
- 启动数据加载：约 100–120 秒（一次性 PNG 解码）
- 单 epoch：约 24–30 秒
- 40 epoch + 早停：约 15–25 分钟可完成

常用可调参数：
| 参数 | 含义 |
|------|------|
| `--epochs` | 训练轮数上限（默认 40） |
| `--batch-size` | 默认 128，越大对 BLAS 越友好 |
| `--lr` / `--momentum` | SGD 学习率 / 动量 |
| `--optimizer {sgd,adam}` | baseline 用 sgd；adam 留作改进对比 |
| `--early-stop-patience N` | valid_acc 连续 N 个 epoch 不刷新最高即停；`0` 关闭 |
| `--no-normalize` | 禁用通道标准化（仅 ToTensor） |
| `--threads` | 显式设置 torch 线程数；默认 `os.cpu_count()` |
| `--seed` | 随机种子（默认 42） |

### 第 2 步：绘制 loss / acc 曲线

```bash
python -m scripts.plot_curves --metrics-csv outputs/baseline/metrics.csv --output-dir  outputs/baseline
```

产物：
- `curve_loss.png`，`curve_acc.png`，`curves.png`（左右合并图，valid 最高点已标注）

### 第 3 步：在 test 集上评估并生成完整分类报告

> ⚠️ test 集仅在本步骤中读取一次。

```bash
python -m scripts.eval --checkpoint outputs/baseline/best_model.pt --data-root  STL10/test --output-dir outputs/baseline
```

产物：
- `classification_report.txt` / `.csv` —— 每类 Precision / Recall / F1 / Support + macro/weighted 平均
- `confusion_matrix.csv` —— 混淆矩阵数值表
- `confusion_matrix.png` —— 绝对计数热力图
- `confusion_matrix_normalized.png` —— 行归一化热力图
- `per_class_errors.csv` —— 每类错误率排名 + 最易混淆的另一个类
- `test_summary.json` —— Accuracy、Top-3 Accuracy、macro/weighted P/R/F1、平均推理耗时

控制台还会打印 Top-3 易错类（便于在实验报告中讨论）。

## 复现性

- 随机种子全程固定为 `--seed 42`（影响 split + train shuffle + 权重初始化）；
- 划分一次后保存到 `split_index.json`，重跑训练自动复用，不会出现"每次训练数据不同"；
- 训练阶段不导入 `STL10/test/`；评估阶段只读取一次。

## 合规性声明（baseline 阶段严格遵守）

| 项目 | 是否使用 |
|------|----------|
| Dropout / DropPath | ❌ |
| BatchNorm / LayerNorm / GroupNorm / InstanceNorm | ❌ |
| `weight_decay` (L2) | ❌（显式置 0） |
| 数据增强（Crop / Flip / ColorJitter / Mixup …） | ❌ |
| Label Smoothing | ❌ |
| 学习率调度 | ❌（保持 baseline 简洁） |
| 输入 `Normalize`（通道均值方差） | ✅（属于输入预处理，可用 `--no-normalize` 关闭） |
| Kaiming 权重初始化 | ✅（仅参数初始值，不属于正则化 / 归一化） |
