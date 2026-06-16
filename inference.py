#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse
import os
import numpy as np
import tifffile
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import common

out_channels = 384
image_size = 256
default_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True,
                        help="Directory with teacher_final.pth, student_final.pth, autoencoder_final.pth")
    parser.add_argument("--normal_dir", required=True,
                        help="Directory of normal images for calibration (e.g. train/good)")
    parser.add_argument("--input", required=True,
                        help="Single image path or directory of images to infer")
    parser.add_argument("--output_dir", default="output/inference",
                        help="Output directory for .tiff anomaly maps")
    parser.add_argument("--model_size", default="small", choices=["small", "medium"])
    return parser.parse_args()

def build_models(model_size):
    if model_size == "small":
        teacher = common.get_pdn_small(out_channels)
        student = common.get_pdn_small(2 * out_channels)
    elif model_size == "medium":
        teacher = common.get_pdn_medium(out_channels)
        student = common.get_pdn_medium(2 * out_channels)
    else:
        raise ValueError(f"Unknown model_size: {model_size}")
    autoencoder = common.get_autoencoder(out_channels)
    return teacher, student, autoencoder


@torch.no_grad()
def teacher_normalization(teacher, normal_loader, device):
    mean_outputs = []
    for image in tqdm(normal_loader, desc="Computing teacher mean"):
        image = image.to(device)
        teacher_output = teacher(image)
        mean_output = torch.mean(teacher_output, dim=[0, 2, 3])
        mean_outputs.append(mean_output)
    channel_mean = torch.mean(torch.stack(mean_outputs), dim=0)
    channel_mean = channel_mean[None, :, None, None]

    mean_distances = []
    for image in tqdm(normal_loader, desc="Computing teacher std"):
        image = image.to(device)
        teacher_output = teacher(image)
        distance = (teacher_output - channel_mean) ** 2
        mean_distance = torch.mean(distance, dim=[0, 2, 3])
        mean_distances.append(mean_distance)
    channel_var = torch.mean(torch.stack(mean_distances), dim=0)
    channel_var = channel_var[None, :, None, None]
    channel_std = torch.sqrt(channel_var)
    return channel_mean, channel_std


@torch.no_grad()
def predict(image_tensor, teacher, student, autoencoder,
            teacher_mean, teacher_std,
            q_st_start=None, q_st_end=None,
            q_ae_start=None, q_ae_end=None):
    teacher_output = teacher(image_tensor)
    teacher_output = (teacher_output - teacher_mean) / teacher_std
    student_output = student(image_tensor)
    autoencoder_output = autoencoder(image_tensor)
    map_st = torch.mean(
        (teacher_output - student_output[:, :out_channels]) ** 2,
        dim=1, keepdim=True)
    map_ae = torch.mean(
        (autoencoder_output - student_output[:, out_channels:]) ** 2,
        dim=1, keepdim=True)
    if q_st_start is not None:
        map_st = 0.1 * (map_st - q_st_start) / (q_st_end - q_st_start)
    if q_ae_start is not None:
        map_ae = 0.1 * (map_ae - q_ae_start) / (q_ae_end - q_ae_start)
    map_combined = 0.5 * map_st + 0.5 * map_ae
    return map_combined, map_st, map_ae


@torch.no_grad()
def map_normalization(normal_loader, teacher, student, autoencoder,
                      teacher_mean, teacher_std, device):
    maps_st = []
    maps_ae = []
    for image in tqdm(normal_loader, desc="Map normalization"):
        image = image.to(device)
        _, map_st, map_ae = predict(
            image, teacher, student, autoencoder,
            teacher_mean, teacher_std)
        maps_st.append(map_st)
        maps_ae.append(map_ae)
    maps_st = torch.cat(maps_st)
    maps_ae = torch.cat(maps_ae)
    q_st_start = torch.quantile(maps_st, q=0.9)
    q_st_end = torch.quantile(maps_st, q=0.995)
    q_ae_start = torch.quantile(maps_ae, q=0.9)
    q_ae_end = torch.quantile(maps_ae, q=0.995)
    return q_st_start, q_st_end, q_ae_start, q_ae_end


def collect_image_paths(input_path):
    if os.path.isfile(input_path):
        return [input_path]
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
    paths = []
    for root, _, files in os.walk(input_path):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                paths.append(os.path.join(root, f))
    return sorted(paths)

def main():
    config = get_argparse()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load models
    teacher, student, autoencoder = build_models(config.model_size)
    teacher = torch.load(os.path.join(config.model_dir, "teacher_final.pth"),
                         map_location="cpu", weights_only=False)
    student = torch.load(os.path.join(config.model_dir, "student_final.pth"),
                         map_location="cpu", weights_only=False)
    autoencoder = torch.load(os.path.join(config.model_dir, "autoencoder_final.pth"),
                             map_location="cpu", weights_only=False)

    teacher.eval().to(device)
    student.eval().to(device)
    autoencoder.eval().to(device)

    # calibration: compute teacher normalization + map quantiles from normal images
    normal_set = common.ImageFolderWithoutTarget(
        config.normal_dir, transform=default_transform)
    normal_loader = DataLoader(normal_set, batch_size=1, shuffle=False, num_workers=0)
    print(f"Normal images for calibration: {len(normal_set)}")

    teacher_mean, teacher_std = teacher_normalization(teacher, normal_loader, device)
    q_st_start, q_st_end, q_ae_start, q_ae_end = map_normalization(
        normal_loader, teacher, student, autoencoder, teacher_mean, teacher_std, device)

    # inference
    image_paths = collect_image_paths(config.input)
    print(f"Images to infer: {len(image_paths)}")
    os.makedirs(config.output_dir, exist_ok=True)

    from PIL import Image
    anomaly_scores = []
    for img_path in tqdm(image_paths, desc="Inference"):
        pil_image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = pil_image.size
        image_tensor = default_transform(pil_image)
        image_tensor = image_tensor[None].to(device)

        map_combined, _, _ = predict(
            image_tensor, teacher, student, autoencoder,
            teacher_mean, teacher_std,
            q_st_start, q_st_end, q_ae_start, q_ae_end)

        # pad from 64x64 -> 72x72, then resize to original
        map_combined = torch.nn.functional.pad(map_combined, (4, 4, 4, 4))
        map_combined = torch.nn.functional.interpolate(
            map_combined, (orig_h, orig_w), mode="bilinear")
        map_combined = map_combined[0, 0].cpu().numpy()

        base = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(config.output_dir, base + ".tiff")
        tifffile.imwrite(out_path, map_combined)

        score = float(np.max(map_combined))
        anomaly_scores.append((base, score))

    print("\n--- Inference Summary ---")
    for name, score in anomaly_scores:
        print(f"  {name}: anomaly_score={score:.6f}")
    print(f"\nOutput saved to: {config.output_dir}")


if __name__ == "__main__":
    main()
