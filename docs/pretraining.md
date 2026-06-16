# pretraining.py 详解

## 文件总体结构

```
pretraining.py
├── 命令行参数解析        get_argparse()
├── 超参数/常量定义        全局变量区
├── 数据变换函数           train_transform()
├── 主训练入口             main()
├── 特征归一化             feature_normalization()
└── 核心类（5 个）
    ├── FeatureExtractor              特征提取器（顶层封装）
    ├── PatchMaker                    特征图 → patch 切分
    ├── Preprocessing                 多尺度特征统一维度
    │   └── MeanMapper               自适应池化
    ├── Aggregator                    多层级特征聚合
    ├── NetworkFeatureAggregator      Backbone 特征提取（hook 机制）
    ├── ForwardHook                  注册 hook，提前终止前向
    └── LastLayerToExtractReachedException  终止信号异常
```

---

## 训练流程

```
ImageNet 训练图像
    │
    ├── grayscale_transform (RandomGrayscale 0.1)  ← 共享灰度化
    │
    ├── extractor_transform:                ├── pdn_transform:
    │   Resize(512) + Normalize               │   Resize(256) + Normalize
    │        ↓                                │        ↓
    │   Wide ResNet-101-2 (冻结)              │   PDN (训练中)
    │        ↓                                │        ↓
    │   layer2 特征 → hook                   │   prediction
    │   layer3 特征 → hook                   │   64×64×384
    │        ↓                                │
    │   PatchMaker.patchify()                 │
    │   (3×3 patches, stride=1)               │
    │        ↓                                │
    │   Preprocessing (MeanMapper)            │
    │   各层特征 → 统一 1024 维               │
    │        ↓                                │
    │   Aggregator                           │
    │   2 层×1024 → 聚合为 384 维            │
    │        ↓                                │
    │   Reshape → 64×64×384  ← teacher 目标    │
    │        ↓                                │
    │   feature_normalization()               │
    │   (channel_mean, channel_std)           │
    │        ↓                                │
    └────────┤                                │
             ↓                                ↓
        target (归一化后的 WR101 特征)     prediction (PDN 输出)
             └────────────┬────────────┘
                          ↓
              loss = MSE(target, prediction)
                          ↓
                   loss.backward()
                   optimizer.step()    ← 仅更新 PDN
                          ↓
                   每 10000步保存权重
                          ↓
                   60,000 步后 → teacher_*_final.pth
```

---

## FeatureExtractor 内部数据流（embed 方法）

这是整个预训练最核心的部分，决定了"教师应该学什么特征":

```
输入: images [B, 3, 512, 512]
    │
    ▼
NetworkFeatureAggregator.forward()
    ├── 清空 outputs 字典
    ├── 调用 backbone(images)，逐层前向传播
    │   ├── layer1 → ...  (不提取)
    │   ├── layer2 → ForwardHook 触发 → outputs['layer2'] = 特征 [B, 512, 64, 64]
    │   └── layer3 → ForwardHook 触发 → outputs['layer3'] = 特征 [B, 1024, 32, 32]
    │       └── 到达 last_layer → 抛出 LastLayerToExtractReachedException
    │           提前终止（不再计算 layer4 和后续）
    └── 返回 outputs = {'layer2': ..., 'layer3': ...}
    │
    ▼
对每层特征分别处理:
    │
    ├── PatchMaker.patchify(feature, return_spatial_info=True)
    │   torch.nn.Unfold(kernel=3, stride=1, padding=1)
    │   layer2: [B, 512, 64, 64] → [B, 4096, 512, 3, 3]      shape [64, 64]
    │   layer3: [B, 1024, 32, 32] → [B, 1024, 1024, 3, 3]     shape [32, 32]
    │
    ├── 多尺度对齐（将 layer3 的 32×32 上采样到 64×64）:
    │   features[i] → reshape → permute → interpolate(bilinear, 64×64)
    │                  → permute → reshape
    │   最终: layer2=[B×4096, 512, 3, 3], layer3=[B×4096, 1024, 3, 3]
    │
    ├── Preprocessing:
    │   layer2: MeanMapper: [N, 512, 3, 3] → [N, 1024]
    │   layer3: MeanMapper: [N, 1024, 3, 3] → [N, 1024]
    │   结果: [N, 2, 1024]  (2 = 层数)
    │
    └── Aggregator:
        自适应平均池化: [N, 2, 1024] → [N, 384]
        最终: [B×4096, 384] → reshape → [B, 64, 64, 384] → permute → [B, 384, 64, 64]
```

**关键设计点**:
- Hook 机制让 backbone 跑到 layer3 就**提前终止**，节省不必要的 layer4 计算
- Patchify 后每个空间位置变成 3×3 的 patch，带有上下文信息
- 多尺度对齐保证了 layer2（高分辨率）和 layer3（低分辨率高语义）在空间上对应

---

## 核心类详解

### 1. NetworkFeatureAggregator — Hook 驱动的特征提取

**工作方式**:

1. 遍历 `layers_to_extract_from = ['layer2', 'layer3']`
2. 对每一层，通过 `._modules` 递归解析到具体的子模块
3. 对 `Sequential` 取最后一层注册 forward hook，对单层直接注册
4. Hook 触发时保存输出到 `self.outputs` 字典
5. 到达 `last_layer_to_extract`（layer3）时，hook 抛出 `LastLayerToExtractReachedException`，立即终止前向传播

**层名解析逻辑**:

| 输入 | 解析方式 |
|------|---------|
| `"layer2"` | `backbone._modules['layer2']` → Sequential，取 `[-1]` |
| `"layer3.5"` | `backbone._modules['layer3'][5]` → 按索引 |
| `"layer3.bn2"` | `backbone._modules['layer3']._modules['bn2']` → 按名字 |

### 2. PatchMaker — Unfold 实现重叠切块

```python
class PatchMaker:
    def __init__(self, patchsize=3, stride=1):
```

使用 `torch.nn.Unfold(kernel=3, stride=1, padding=1)` 将特征图转为 patch 序列:

```
输入: [B, C, H, W]
    ↓ Unfold: 在每个空间位置展开 3×3 邻域
输出: [B, C×9, H×W] → reshape → [B*H*W, C, 3, 3]
```

`padding=1` 确保输出空间尺寸与输入一致（same padding）。

### 3. Preprocessing + MeanMapper — 维度统一

不同层的特征通道数不同（layer2=512, layer3=1024），需要统一到相同维度:

```python
class MeanMapper(nn.Module):
    def forward(self, features):
        features = features.reshape(len(features), 1, -1)  # [N, 1, 512×9]
        return F.adaptive_avg_pool1d(features, self.preprocessing_dim).squeeze(1)
        # → [N, 1024]
```

**本质**: 对每个 3×3 patch 内的元素做自适应平均池化，压缩到 1024 维。

### 4. Aggregator — 多层级特征融合

```python
class Aggregator(nn.Module):
    def forward(self, features):
        features = features.reshape(len(features), 1, -1)  # [N, 1, 2×1024]
        features = F.adaptive_avg_pool1d(features, self.target_dim)  # → [N, 384]
        return features.reshape(len(features), -1)
```

将两层的 1024 维特征拼接后池化为 384 维，最终和 PDN 教师网络的输出通道数一致。

### 5. ForwardHook — 带终止的前向钩子

```python
class ForwardHook:
    def __init__(self, hook_dict, layer_name, last_layer_to_extract):
        self.raise_exception_to_break = (layer_name == last_layer_to_extract)

    def __call__(self, module, input, output):
        self.hook_dict[self.layer_name] = output
        if self.raise_exception_to_break:
            raise LastLayerToExtractReachedException()
```

---

## 特征归一化详解

```python
def feature_normalization(extractor, train_loader, steps=10000):
```

**目的**: 让 Wide ResNet 特征图的各通道均值为 0、标准差为 1。

**两轮遍历**（各 10000 张图，batch_size=16 所以约 625 步）:

| 轮次 | 计算 | 统计量 |
|------|------|--------|
| 第一轮 | `mean(output, dim=[0, 2, 3])` → 每个通道的均值 | `channel_mean: [1, 384, 1, 1]` |
| 第二轮 | `mean((output - mean)^2, dim=[0, 2, 3])` → 每个通道的方差 | `channel_std: [1, 384, 1, 1]` |

训练时归一化: `target = (extractor_output - channel_mean) / channel_std`

---

## 训练超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| batch_size | 16 | 比 efficientad.py 大（那里 batch=1） |
| optimizer | Adam(lr=1e-4, weight_decay=1e-5) | 标准设置 |
| 训练步数 | 60,000 | 比 efficientad.py 少 10,000 步 |
| 学习率衰减 | 无显式 scheduler | 全程常数 lr |
| backbone | Wide ResNet-101-2, ImageNet1K_V1 权重 | 冻结，不更新 |
| 输入尺寸 (extractor) | 512×512 | WR101 需要更大分辨率 |
| 输入尺寸 (PDN) | 256×256 | 与下游任务一致 |
| 保存频率 | 每 10,000 步 | 同时保存完整模型和 state_dict |

---

## 与 efficientad.py 的关系

```
pretraining.py 产出                    efficientad.py 使用
─────────────────────────────────────────────────────────
teacher_small_final.pth      →        --weights models/teacher_small.pth
teacher_medium_final.pth     →        --weights models/teacher_medium.pth
                                       加载为 teacher (冻结) + student (需训练)
```

两个文件使用同一个 `get_pdn_small()` / `get_pdn_medium()` 工厂函数（来自 `common.py`），但参数不同:

| 参数 | pretraining | efficientad |
|------|-------------|-------------|
| `padding` | `True` | `False` (默认) |
| 原因 | 512×512 输入，下采样后需对齐到 64×64 | 256×256 输入，经两次 AvgPool2d 后已是 64×64，无需额外 padding |
