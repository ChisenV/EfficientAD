# EfficientAD 项目架构文档

## 项目简介

EfficientAD: Accurate Visual Anomaly Detection 论文的非官方 PyTorch 实现，用于工业视觉异常检测（如检测产品表面缺陷）。

- 论文: <https://arxiv.org/abs/2303.14535>
- 核心思想: 教师-学生知识蒸馏 + 自编码器双分支异常检测

## 目录结构

```
EfficientAD/
├── common.py                  # 模型定义 + 数据加载工具
├── pretraining.py              # 教师网络预训练（知识蒸馏阶段 1）
├── efficientad.py              # 主训练 + 推理（知识蒸馏阶段 2）
├── benchmark.py                # 推理速度基准测试
├── models/                     # 预训练好的教师网络权重
│   ├── teacher_small.pth       # PDN-Small（~10MB）
│   └── teacher_medium.pth      # PDN-Medium（~32MB）
├── results/                    # 论文复现结果
│   ├── mvtec_ad_small.json
│   └── mvtec_ad_medium.json
├── mvtec_loco_ad_evaluation/   # MVTec LOCO 官方评测工具
└── output/                     # 训练输出（异常图、模型权重、指标）
```

## 核心网络模型

| 网络 | 定义位置 | 作用 | 输入 | 输出 |
|------|---------|------|------|------|
| 教师 PDN | `common.py:get_pdn_small/medium` | 提取正常样本特征（冻结） | 256×256×3 | 64×64×384 |
| 学生 PDN | `common.py:get_pdn_small/medium` | 学习模仿教师（训练） | 256×256×3 | 64×64×768 |
| 自编码器 | `common.py:get_autoencoder` | 重构图像特征（训练） | 256×256×3 | 64×64×384 |

### 教师/学生网络结构

```
PDN-Small:                        PDN-Medium:
Conv(3→128, k4) + ReLU           Conv(3→256, k4) + ReLU
AvgPool2d(2)                      AvgPool2d(2)
Conv(128→256, k4) + ReLU         Conv(256→512, k4) + ReLU
AvgPool2d(2)                      AvgPool2d(2)
Conv(256→256, k3) + ReLU         Conv(512→512, k1) + ReLU
Conv(256→out, k4)                Conv(512→512, k3) + ReLU
                                  Conv(512→out, k4) + ReLU
                                  Conv(out→out, k1)
```

- 教师输出 384 通道
- 学生输出 768 通道（前 384 = ST 分支，后 384 = AE 分支）

### 自编码器结构

```
编码器 (下采样 256→1):            解码器 (上采样 1→64):
Conv(3→32, k4s2) + ReLU         Upsample→3, Conv(64→64) + ReLU + Dropout
Conv(32→32, k4s2) + ReLU        Upsample→8, Conv(64→64) + ReLU + Dropout
Conv(32→64, k4s2) + ReLU        Upsample→15, Conv(64→64) + ReLU + Dropout
Conv(64→64, k4s2) + ReLU        Upsample→32, Conv(64→64) + ReLU + Dropout
Conv(64→64, k4s2) + ReLU        Upsample→63, Conv(64→64) + ReLU + Dropout
Conv(64→64, k8)                 Upsample→127, Conv(64→64) + ReLU + Dropout
                                 Upsample→56, Conv(64→64) + ReLU
                                 Conv(64→384, k3)
```

## 预训练教师网络 (Feature Extractor)

`pretraining.py` 中使用 Wide ResNet-101-2 的 layer2 和 layer3 作为特征提取 backbone:

```
Wide ResNet-101-2 (冻结, ImageNet 预训练)
    ├── layer2 → 特征图
    ├── layer3 → 特征图
    ├── PatchMaker (patchify, patchsize=3, stride=1)
    ├── Preprocessing (MeanMapper, 自适应池化到 1024 维)
    └── Aggregator (聚合到 384 维)
              ↓
         64×64×384 特征图（训练目标）
```

相当于用一个大模型的中间层特征来训练轻量级 PDN，使 PDN 学会提取与 WR101 相似的语义特征。

## 工具类

| 类/函数 | 位置 | 说明 |
|---------|------|------|
| `ImageFolderWithoutTarget` | `common.py` | 从 ImageFolder 读取图像，丢弃标签 |
| `ImageFolderWithPath` | `common.py` | 从 ImageFolder 读取图像，返回路径 |
| `InfiniteDataloader` | `common.py` | 无限循环的 DataLoader 包装器 |
