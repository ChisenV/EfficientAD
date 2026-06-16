# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Unofficial PyTorch implementation of [EfficientAD](https://arxiv.org/abs/2303.14535) — an anomaly detection method achieving accurate visual anomaly detection at millisecond-level latencies using a Teacher-Student + Autoencoder architecture.

## Commands

### Training and Inference

```bash
# EfficientAD-S on a single MVTec AD sub-dataset
python efficientad.py --dataset mvtec_ad --subdataset bottle

# EfficientAD-M with ImageNet penalty term (full paper setup)
python efficientad.py --dataset mvtec_ad --subdataset bottle \
    --model_size medium --weights models/teacher_medium.pth \
    --imagenet_train_path ./ILSVRC/Data/CLS-LOC/train

# MVTec LOCO dataset
python efficientad.py --dataset mvtec_loco --subdataset breakfast_box
```

### Evaluation

#### MVTec AD (requires separately downloaded eval code)

```bash
python mvtec_ad_evaluation/evaluate_experiment.py \
    --dataset_base_dir './mvtec_anomaly_detection/' \
    --anomaly_maps_dir './output/1/anomaly_maps/mvtec_ad/' \
    --output_dir './output/1/metrics/mvtec_ad/' \
    --evaluated_objects bottle
```

#### MVTec LOCO (`mvtec_loco_ad_evaluation/` 已包含在仓库中)

异常图要求：每个测试样本必须有对应的 `.tiff` 文件，目录结构为 `<anomaly_maps_dir>/<object_name>/test/<defect_name>/<image_id>.tiff`。

**单个对象评估**（计算 sPRO 曲线下面积 + 图像级 AUC-ROC）：

```bash
python mvtec_loco_ad_evaluation/evaluate_experiment.py \
    --object_name breakfast_box \
    --dataset_base_dir './mvtec_loco_anomaly_detection/' \
    --anomaly_maps_dir './output/1/anomaly_maps/mvtec_loco/' \
    --output_dir './output/1/metrics/mvtec_loco/'
```

可选参数：`--num_parallel_workers N`（并行CPU数）、`--curve_max_distance 0.001`（sPRO曲线精度）。

**批量评估多个实验**（需要 `config.json` 定义实验路径）：

```bash
python mvtec_loco_ad_evaluation/evaluate_multiple_experiments.py \
    --dataset_base_dir './mvtec_loco_anomaly_detection/' \
    --experiment_configs 'experiment_configs.json' \
    --output_dir './output/1/metrics/mvtec_loco/'
```

**查看结果表格**（在评估完成后使用）：

```bash
# 像素级定位结果 (AUC-sPRO)
python mvtec_loco_ad_evaluation/print_metrics.py \
    --metrics_folder './output/1/metrics/mvtec_loco/' \
    --metric_type localization --integration_limit 0.3

# 图像级分类结果 (AUC-ROC)
python mvtec_loco_ad_evaluation/print_metrics.py \
    --metrics_folder './output/1/metrics/mvtec_loco/' \
    --metric_type classification
```

### Teacher Pretraining

```bash
python pretraining.py --output_folder output/pretraining/1/
```

Note: `model_size` and `imagenet_train_path` must be edited directly in `pretraining.py` (hardcoded variables, not CLI args).

### Benchmark

```bash
python benchmark.py
```

## Dependencies

Python 3.10, torch 1.13.0, torchvision 0.14.0, tifffile, tqdm, scikit-learn 1.2.2. No `requirements.txt` or `setup.py` exists — install manually.

## Dataset Setup

MVTec AD and LOCO datasets must be downloaded separately. Set paths via `--mvtec_ad_path` / `--mvtec_loco_path` (defaults: `./mvtec_anomaly_detection`, `./mvtec_loco_anomaly_detection`). MVTec evaluation code is also a separate download (see README for URLs).

## Gotchas

- **Output directory**: `efficientad.py` calls `os.makedirs()` without `exist_ok=True` — re-running with the same `--output_dir` and `--subdataset` will crash. Delete or rename the existing output directory first.
- **Quick test runs**: Use `--train_steps 100` to verify setup before a full 70k-step run.

## Architecture

### Three-Network Design

1. **Teacher (PDN)** — Frozen patch description network pretrained to distill Wide-ResNet-101-2 features. Provides target representations. Pre-trained weights are in `models/`.
2. **Student (PDN)** — Same architecture as teacher but with doubled output channels (768 vs 384). First 384 channels learn to mimic the teacher; last 384 channels learn to mimic the autoencoder.
3. **Autoencoder** — Encoder-decoder with a tight bottleneck. Learns to reconstruct teacher features; captures structural/logical anomalies.

### Anomaly Scoring

Two anomaly maps are computed and combined 50/50 after quantile normalization:
- **Student-Teacher map**: MSE between teacher output and student's first 384 channels (texture/pattern anomalies)
- **AE-Student map**: MSE between autoencoder output and student's last 384 channels (logical/structural anomalies)

Image-level anomaly score = max of the combined map.

### Code Layout

- **`efficientad.py`** — Main entry point: training loop, evaluation, anomaly map generation. Contains all CLI argument parsing, loss computation (hard-example MSE + optional ImageNet penalty + AE losses), teacher output normalization, and quantile-based map normalization.
- **`common.py`** — Model definitions (`get_pdn_small`, `get_pdn_medium`, `get_autoencoder`) and dataset utilities (`ImageFolderWithoutTarget`, `ImageFolderWithPath`, `InfiniteDataloader`).
- **`pretraining.py`** — Teacher PDN pretraining from Wide-ResNet-101-2 using `NetworkFeatureAggregator`, `Preprocessing`, `Aggregator`, and `PatchMaker` classes (all defined within this file).
- **`benchmark.py`** — GPU/CPU inference latency measurement.

### Key Training Details

- Default 70,000 steps, batch size 1, Adam optimizer (lr=1e-4, weight_decay=1e-5)
- Hard-example mining: only top 0.1% hardest pixels contribute to student-teacher loss (`torch.quantile(..., q=0.999)`)
- Teacher outputs are channel-wise normalized (mean/std from training set) before use
- Checkpoints saved every 1,000 steps; intermediate evaluation every 10,000 steps
- Input resolution: 256×256, output channels: 384

### Dataset Handling

Supports `mvtec_ad` (15 sub-datasets) and `mvtec_loco` (5 sub-datasets). MVTec AD uses a random 10% train split for validation; MVTec LOCO has a separate `validation/` directory. Test outputs are saved as TIFF anomaly maps under `output/1/anomaly_maps/`.

## No Test or Lint Infrastructure

There are no tests, no linting configuration, and no CI/CD pipeline in this repository.
