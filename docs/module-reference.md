# 模块与函数参考

## common.py — 模型定义与数据工具

### 模型工厂函数

#### `get_autoencoder(out_channels=384) → nn.Sequential`

构建编解码器网络。编码器 6 层逐步下采样 (256→128→64→32→16→8→1)，解码器 7 次上采样恢复至 64×64 特征图。

#### `get_pdn_small(out_channels=384, padding=False) → nn.Sequential`

构建 PDN-Small 网络。4 层卷积 + 2 层 AvgPool，适合轻量部署。

#### `get_pdn_medium(out_channels=384, padding=False) → nn.Sequential`

构建 PDN-Medium 网络。6 层卷积 + 2 层 AvgPool，精度更高。

### 数据工具类

#### `ImageFolderWithoutTarget(ImageFolder)`

继承 `torchvision.datasets.ImageFolder`，覆盖 `__getitem__` 丢弃标签只返回图像。

#### `ImageFolderWithPath(ImageFolder)`

继承 `ImageFolder`，返回 `(image, target, path)` 三元组，用于测试时保存异常图。

#### `InfiniteDataloader(loader) → generator`

将有限 DataLoader 包装为无限循环生成器，避免手动管理 epoch。

---

## pretraining.py — 教师预训练

### 核心类

#### `FeatureExtractor(nn.Module)`

**作用**: 从 Wide ResNet-101-2 提取特征并处理为统一格式。

**流程**: WR101 前向 → layer2/layer3 特征 → PatchMaker → Preprocessing → Aggregator → 64×64×384

**关键方法**: `embed(images)` — 返回特征嵌入

#### `PatchMaker`

**作用**: 将特征图切分为重叠 patch。

**参数**: `patchsize=3, stride=1`，用 `torch.nn.Unfold` 实现。

#### `Preprocessing(nn.Module)`

**作用**: 将不同层的不同维度特征统一到相同维度 (1024)。

使用 `MeanMapper` (自适应平均池化) 实现。

#### `Aggregator(nn.Module)`

**作用**: 将多层特征聚合到目标维度 (out_channels=384)。

#### `NetworkFeatureAggregator(nn.Module)`

**作用**: 通过 forward hook 高效提取 backbone 中间层特征，到达最后一层时抛出 `LastLayerToExtractReachedException` 提前终止前向传播。

#### `ForwardHook`

**作用**: 注册到目标层的 forward hook，保存输出到字典。到达指定最后一层时抛出异常停止计算。

### 辅助函数

#### `feature_normalization(extractor, train_loader, steps=10000)`

计算特征提取器输出的通道均值和标准差，用于归一化训练目标。

**返回**: `(channel_mean, channel_std)` — shape `[1, C, 1, 1]`

---

## efficientad.py — 主训练与推理

### 入口函数

#### `main()`

解析参数、加载数据、创建模型、执行训练循环 (70,000 步)、最终评估。

关键参数 (`get_argparse()`):

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | mvtec_ad | 数据集选择: mvtec_ad / mvtec_loco |
| `--subdataset` | bottle | 子数据集 (15 个 AD 类别或 5 个 LOCO 类别) |
| `--model_size` | small | 模型规模: small / medium |
| `--weights` | models/teacher_small.pth | 预训练教师权重路径 |
| `--imagenet_train_path` | none | ImageNet 路径，"none" 禁用预训练惩罚 |
| `--train_steps` | 70000 | 训练步数 |

### 训练相关函数

#### `teacher_normalization(teacher, train_loader) → (mean, std)`

在训练集上计算教师输出的通道均值和标准差。两轮遍历: 第一轮算均值, 第二轮算标准差。

#### `train_transform(image) → (image_st, image_ae)`

对同一张图生成两个变体: 标准变换用于 ST 分支, 颜色抖动用于 AE 分支。

### 推理相关函数

#### `predict(image, teacher, student, autoencoder, teacher_mean, teacher_std, q_st_start, q_st_end, q_ae_start, q_ae_end) → (map_combined, map_st, map_ae)`

核心推理函数，生成三张异常图:

- `map_st`: 教师-学生差异图
- `map_ae`: 自编码器-学生差异图
- `map_combined`: 融合异常图 = 0.5×map_st + 0.5×map_ae

各分图经过量化归一化: `0.1 * (map - q_start) / (q_end - q_start)`

#### `map_normalization(validation_loader, ...) → (q_st_start, q_st_end, q_ae_start, q_ae_end)`

在验证集上计算异常图的分位数边界，用于推理时的量化归一化。

#### `test(test_set, ...) → auc`

在完整测试集上运行推理，生成异常图文件 (`.tiff`)，计算图像级 ROC AUC。

### 损失函数

| 损失 | 公式 | 权重 | 说明 |
|------|------|------|------|
| `loss_hard` | `mean(dist[dist >= Q_{0.999}])` | 1.0 | ST 硬样本损失 |
| `loss_penalty` | `mean(student_penalty^2)` | 1.0 | ImageNet 正则化 |
| `loss_ae` | `mean((teacher - ae)^2)` | 1.0 | AE 重构损失 |
| `loss_stae` | `mean((ae - student_ae)^2)` | 1.0 | 学生-AE 对齐损失 |

---

## benchmark.py — 推理速度测试

独立测试脚本，量化推理延迟。

评估指标: 单张 256×256 图像推理时间（不含 I/O）。

使用 FP16 精度 (`.half().cuda()`) 提高速度，取最后 1000 次平均。
