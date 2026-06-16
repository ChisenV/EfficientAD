#!/usr/bin/python
import argparse
import os
import numpy as np
import cv2
import tifffile
from PIL import Image, ImageDraw, ImageFont

def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--anomaly_dir', required=True,
                        help='Directory of .tiff anomaly maps with test/*/ structure')
    parser.add_argument('--original_dir', required=True,
                        help='Directory of original test images with same subdir structure')
    parser.add_argument('--output_dir', default='output/overlay',
                        help='Where to save composite PNGs')
    parser.add_argument('--max_images', type=int, default=0,
                        help='Limit images per class (0=all)')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Percentile threshold for bbox (e.g. 0.995 = top 0.5%%). '
                             'Omitting disables boxes.')
    parser.add_argument('--min_area', type=int, default=0,
                        help='Minimum connected component area (pixels) to draw a box')
    parser.add_argument('--alpha', type=float, default=0.4,
                        help='Heatmap overlay opacity (default 0.4)')
    return parser.parse_args()

def jet_colormap(normalized):
    x = normalized * 255
    r = np.clip(np.minimum(4 * x - 384, 512 - 4 * x), 0, 255).astype(np.uint8)
    g = np.clip(np.minimum(4 * x - 128, 384 - 4 * x), 0, 255).astype(np.uint8)
    b = np.clip(np.minimum(4 * x, 768 - 4 * x), 0, 255).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)

def heatmap_to_image(anomaly_map):
    vmin, vmax = np.min(anomaly_map), np.max(anomaly_map)
    if vmax - vmin < 1e-8:
        return Image.new("RGB", (anomaly_map.shape[1], anomaly_map.shape[0]), (0, 0, 0))
    normalized = np.clip((anomaly_map - vmin) / (vmax - vmin), 0, 1)
    return Image.fromarray(jet_colormap(normalized))

def overlay_heatmap(original_rgb, anomaly_map, alpha=0.4):
    vmin, vmax = np.min(anomaly_map), np.max(anomaly_map)
    if vmax - vmin < 1e-8:
        return Image.fromarray(original_rgb)
    normalized = np.clip((anomaly_map - vmin) / (vmax - vmin), 0, 1)
    heatmap_rgb = jet_colormap(normalized)
    mask = (normalized[:, :, None] * alpha).astype(np.float32)
    blended = original_rgb.astype(np.float32) * (1 - mask) + heatmap_rgb.astype(np.float32) * mask
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))

def find_original(anomaly_maps_root, original_dir):
    mapping = {}
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    for root, _, files in os.walk(anomaly_maps_root):
        for f in files:
            if not f.lower().endswith(".tiff"):
                continue
            tiff_path = os.path.join(root, f)
            base = os.path.splitext(f)[0]
            rel_dir = os.path.relpath(root, anomaly_maps_root)
            if rel_dir == ".":
                rel_dir = ""
            for ext in exts:
                candidate = os.path.join(original_dir, rel_dir, base + ext)
                if os.path.exists(candidate):
                    mapping[tiff_path] = candidate
                    break
    return mapping

def make_composite(original, heatmap_img, overlay_img, score, bboxes=None):
    w, h = original.size
    pad = 4
    label_h = 24
    total_w = w * 3 + pad * 2
    total_h = h + label_h
    composite = Image.new("RGB", (total_w, total_h), (40, 40, 40))
    composite.paste(original, (0, 0))
    composite.paste(heatmap_img, (w + pad, 0))
    composite.paste(overlay_img, (2 * (w + pad), 0))
    try:
        font = ImageFont.truetype("consola.ttf", 13)
    except Exception:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(composite)
    draw.text((4, h + 4), f"score: {score:.4f}", fill=(210, 210, 210), font=font)
    if bboxes:
        overlay_offset_x = 2 * (w + pad)
        composite_np = np.array(composite)
        for bx, by, bw, bh in bboxes:
            cv2.rectangle(composite_np, (bx, by), (bx + bw - 1, by + bh - 1),
                          (255, 0, 0), 2)
            cv2.rectangle(composite_np,
                          (overlay_offset_x + bx, by),
                          (overlay_offset_x + bx + bw - 1, by + bh - 1),
                          (255, 0, 0), 2)
        composite = Image.fromarray(composite_np)
    return composite

def find_connected_components(binary_mask):
    """Return list of (x, y, w, h) bounding boxes using OpenCV connectedComponentsWithStats."""
    binary_u8 = binary_mask.astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_u8, connectivity=4)
    bboxes = []
    for i in range(1, num_labels):  # skip label 0 (background)
        x, y, w, h, area = stats[i]
        bboxes.append((x, y, w, h))
    return bboxes

def main():
    config = get_argparse()
    mapping = find_original(config.anomaly_dir, config.original_dir)
    print(f"Anomaly maps found: {len(mapping)}")
    if not mapping:
        print("No .tiff files found!")
        return
    os.makedirs(config.output_dir, exist_ok=True)
    classes = {}
    for tiff_path in mapping:
        cls = os.path.basename(os.path.dirname(tiff_path))
        classes.setdefault(cls, []).append(tiff_path)
    for cls, tiff_paths in sorted(classes.items()):
        cls_out = os.path.join(config.output_dir, cls)
        os.makedirs(cls_out, exist_ok=True)
        limit = config.max_images if config.max_images > 0 else len(tiff_paths)
        count = 0
        for tiff_path in sorted(tiff_paths):
            if count >= limit:
                break
            anomaly_map = tifffile.imread(tiff_path)
            orig_path = mapping[tiff_path]
            original = Image.open(orig_path).convert("RGB")
            if original.size != (anomaly_map.shape[1], anomaly_map.shape[0]):
                original = original.resize(
                    (anomaly_map.shape[1], anomaly_map.shape[0]), Image.BILINEAR)
            heatmap_img = heatmap_to_image(anomaly_map).resize(
                original.size, Image.NEAREST)
            overlay_img = overlay_heatmap(np.array(original), anomaly_map, alpha=config.alpha)
            score = float(np.max(anomaly_map))
            bboxes = None
            if config.threshold is not None:
                thresh_val = np.quantile(anomaly_map, config.threshold)
                binary = anomaly_map > thresh_val
                bboxes = find_connected_components(binary)
                if config.min_area > 0:
                    bboxes = [(x, y, w, h) for x, y, w, h in bboxes
                              if w * h >= config.min_area]
            composite = make_composite(original, heatmap_img, overlay_img, score, bboxes)
            base = os.path.splitext(os.path.basename(tiff_path))[0]
            if bboxes:
                print(f"    {base}: {len(bboxes)} bbox(es)")
            composite.save(os.path.join(cls_out, base + ".png"))
            count += 1
        print(f"  {cls}: {count} images")
    print(f"\nDone. Output: {config.output_dir}")


if __name__ == "__main__":
    main()
