# 训练与推理流程详解

## 整体流程概览

```
阶段1: pretraining.py
  ImageNet → Wide ResNet-101-2 → 特征图 → 训练 PDN 教师网络
                                              ↓
阶段2: efficientad.py                    teacher_*.pth
  MVTec 正常样本 → 教师(冻结) + 学生 + 自编码器 联合训练
                                              ↓
推理: efficientad.py (test 函数)
  测试图像 → 三网络前向 → 异常热力图 → ROC AUC
```

---

## 阶段 1: 教师预训练

### 入口

```bash
python pretraining.py -o output/pretraining/1/
```

### 目标

用 Wide ResNet-101-2 的知识训练一个轻量 PDN，使其学会提取通用的、有判别力的视觉特征。

### 数据流

```
ImageNet 训练图像
    ↓ RandomGrayscale(0.1)
    ├── extractor_transform (512×512) → Wide ResNet-101-2 → 64×64×384 特征
    └── pdn_transform (256×256) → PDN 教师 → 64×64×384 预测
              ↓
    loss = MSE(target, prediction)
    loss.backward() → 仅更新 PDN
```

### 关键步骤

1. **FeatureExtractor.embed()**: 从 WR101 的 layer2、layer3 提取特征，patchify、预处理、聚合为 64×64×384
2. **feature_normalization()**: 计算 WR101 输出的通道均值和标准差（10000 步采样）
3. **训练循环**: 60,000 步，MSE 损失，Adam(lr=1e-4)

### 输出

- `teacher_small_final.pth` / `teacher_medium_final.pth`

---

## 阶段 2: 异常检测训练

### 入口

```bash
python efficientad.py -d mvtec_ad -s bottle -m medium \
    -w models/teacher_medium.pth -i ./ILSVRC/Data/CLS-LOC/train
```

### 数据准备

| 数据集 | 训练集 | 验证集 | 测试集 |
|--------|--------|--------|--------|
| Mvtec AD | 90% train 目录 | 10% train 目录 | test 目录 |
| Mvtec LOCO | train 目录 | validation 目录 | test 目录 |

### 三路数据增强

```python
# 原始图像 → default_transform (Resize 256 + Normalize) → image_st
# 原始图像 → 颜色抖动 + default_transform → image_ae
# ImageNet图像 → penalty_transform (512→crop→256 + RandomGray) → image_penalty
```

### 训练循环 (70,000 步)

每个迭代处理三个分支:

```
┌──────────────────────────────────────────────────────────────┐
│  ST 分支 (Student-Teacher)                                    │
│  ┌─────────────┐    ┌──────────┐                             │
│  │ image_st     │───→│ teacher  │ → teacher_output (归一化)    │
│  │              │    └──────────┘                             │
│  │              │    ┌──────────┐                             │
│  │              │───→│ student  │ → student_output[:, :384]   │
│  └─────────────┘    └──────────┘                             │
│  distance_st = (teacher_output - student_output_st)^2         │
│  loss_hard = mean(top 0.1% distance_st)  ← 硬样本挖掘          │
│                                                              │
│  Penalty 分支 (可选，需 ImageNet)                              │
│  ┌─────────────┐    ┌──────────┐                             │
│  │image_penalty│───→│ student  │ → student_output[:, :384]   │
│  └─────────────┘    └──────────┘                             │
│  loss_penalty = mean(student_output_penalty^2)                │
│                                                              │
│  AE 分支 (Autoencoder)                                       │
│  ┌─────────────┐    ┌──────────┐                             │
│  │ image_ae     │───→│ teacher  │ → teacher_output (归一化)    │
│  │              │    └──────────┘                             │
│  │              │    ┌──────────┐                             │
│  │              │───→│ student  │ → student_output[:, 384:]  │
│  │              │    └──────────┘                             │
│  │              │    ┌──────────┐                             │
│  │              │───→│autoencodr│ → ae_output                 │
│  └─────────────┘    └──────────┘                             │
│  loss_ae   = mean((teacher_output - ae_output)^2)            │
│  loss_stae = mean((ae_output - student_output_ae)^2)         │
└──────────────────────────────────────────────────────────────┘

loss_total = loss_hard + loss_penalty + loss_ae + loss_stae
```

### 验证与推理

每 10,000 步进行一次中间评估:

1. **map_normalization()**: 在验证集上计算 ST 和 AE 异常图的分位数边界
   - `q_st_start = quantile(maps_st, 0.9)`
   - `q_st_end = quantile(maps_st, 0.995)`
   - `q_ae_start = quantile(maps_ae, 0.9)`
   - `q_ae_end = quantile(maps_ae, 0.995)`

2. **test()**: 在测试集上生成异常图并计算图像级 ROC AUC

---

## 推理阶段

### predict() 函数

```python
# 输入: 单张 256×256 图像
teacher_output = normalize(teacher(image))
student_output = student(image)

map_st = mean((teacher - student[:, :384])^2)     # ST 异常图
map_ae = mean((autoencoder(image) - student[:, 384:])^2)  # AE 异常图

# 量化归一化 (使用验证集分位数)
map_st = 0.1 * (map_st - q_st_start) / (q_st_end - q_st_start)
map_ae = 0.1 * (map_ae - q_ae_start) / (q_ae_end - q_ae_start)

map_combined = 0.5 * map_st + 0.5 * map_ae        # 融合异常图
```

### 后处理

1. Padding 到 64+8=72 → Interpolate 回原始尺寸
2. 保存为 `.tiff` 文件（每像素异常分数）
3. 图像级分数 = `max(map_combined)` → AUC 计算

---

## 关键设计决策

| 设计 | 说明 | 位置 |
|------|------|------|
| 硬样本挖掘 | 仅对距离最大的 0.1% 像素计算损失，迫使网络关注最难区域 | `efficientad.py:179-180` |
| ImageNet 惩罚 | 让学生对通用图像输出接近 0，防止对异常也学出有效特征 | `efficientad.py:183-184` |
| 量化归一化 | 将 ST 和 AE 分支映射到同一尺度，使其可加和 | `efficientad.py:311-314` |
| 颜色抖动输入 | AE 分支输入经过颜色增强，使自编码器学习更鲁棒的重构 | `common.py:54-58` |
| SGD 阶梯衰减 | 在 95% 训练步数时学习率降为 0.1 倍 | `efficientad.py:164-165` |
