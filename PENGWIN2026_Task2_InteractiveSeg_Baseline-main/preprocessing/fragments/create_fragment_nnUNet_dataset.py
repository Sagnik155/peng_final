#!/usr/bin/env python3

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk


# -----------------------------
# CLICK TYPES
# -----------------------------
CLICK_TYPES = [
    "uniformly_sampled",
    "euclidean_distance_transform",
    "center_of_mass",
    "boundary_internal_margin",
]




# -----------------------------
# IO
# -----------------------------
def load_img(path):
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    return img, arr


def save_img(arr, ref, path):
    img = sitk.GetImageFromArray(arr.astype(np.float32))
    img.CopyInformation(ref)
    img = sitk.Cast(img, sitk.sitkFloat32)
    sitk.WriteImage(img, str(path))


def save_label(arr, ref, path):
    img = sitk.GetImageFromArray(arr.astype(np.uint8))
    img.CopyInformation(ref)
    img = sitk.Cast(img, sitk.sitkUInt8)
    sitk.WriteImage(img, str(path))




# -----------------------------
# LOAD JSON POINTS
# -----------------------------
def load_points(json_path):
    if not json_path.exists():
        return [], []

    with open(json_path) as f:
        data = json.load(f)

    pts = [p["point"] for p in data.get("points", [])]
    names = [p["name"].split(" ")[0] for p in data.get("points", [])]

    return pts, names


# -----------------------------
# FRAGMENT FROM LABEL
# -----------------------------
def get_fragment_from_label(label_arr, point):
    z, y, x = map(int, point)

    if not (
        0 <= z < label_arr.shape[0]
        and 0 <= y < label_arr.shape[1]
        and 0 <= x < label_arr.shape[2]
    ):
        return None, None

    label_value = label_arr[z, y, x]

    if label_value == 0:
        return None, None

    return (label_arr == label_value), label_value


# -----------------------------
# FAST GAUSSIAN (NO MESHGRID PER CALL)
# -----------------------------
def gaussian_from_coords(Z, Y, X, center, sigma=1.0):
    cz, cy, cx = center
    dist_sq = (Z - cz) ** 2 + (Y - cy) ** 2 + (X - cx) ** 2
    return np.exp(-dist_sq / (2 * sigma ** 2))


# -----------------------------
# BG HEATMAP (VECTORISED)
# -----------------------------
def heatmap_from_points(Z, Y, X, points, sigma=1.0):
    heatmap = np.zeros(Z.shape, dtype=np.float32)

    for p in points:
        z, y, x = p
        dist_sq = (Z - z) ** 2 + (Y - y) ** 2 + (X - x) ** 2
        heatmap += np.exp(-dist_sq / (2 * sigma ** 2))

    m = heatmap.max()
    if m > 0:
        heatmap /= m

    return heatmap


# -----------------------------
# MAIN PIPELINE
# -----------------------------
def process_case(ct_path, label_path, clicks_root, out_img, out_lbl, case_id):

    t0 = time.time()
    print(f"\n========== {case_id} ==========")

    # -------------------------
    # LOAD
    # -------------------------
    t = time.time()
    ct_img, ct = load_img(ct_path)
    _, labels = load_img(label_path)

    shape = ct.shape
    print(f"[LOAD] {time.time() - t:.3f}s | shape={shape}")

    # -------------------------
    # PRECOMPUTE GRID (🔥 BIG SPEEDUP)
    # -------------------------
    Z, Y, X = np.meshgrid(
        np.arange(shape[0]),
        np.arange(shape[1]),
        np.arange(shape[2]),
        indexing="ij",
    )

    # -------------------------
    # CLICK TYPE
    # -------------------------
    click_type = random.choice(CLICK_TYPES)
    print(f"[CLICK TYPE] {click_type}")

    json_path = clicks_root / click_type / case_id / "peripelvic-fragment-clicks.json"
    points, names = load_points(json_path)
    name_counts = {}
    for n in names:
        name_counts[n] = name_counts.get(n, 0) + 1

    print(f"[POINTS] {len(points)} clicks")

    # -------------------------
    # PROCESS EACH CLICK
    # -------------------------
    for i, pt in enumerate(points):

        frag_id = f"case_{case_id}_frag_{names[i]}_{i:03d}"
        if name_counts[names[i]] <= 1:
            print(f"\n [SKIP]  -> {frag_id}")
            continue
        print(f"\n  -> {frag_id}")


        frag_mask, label_value = get_fragment_from_label(labels, pt)

        if frag_mask is None:
            print("     skip invalid/background click")
            continue

        print(f"     label = {label_value}")

        label_out = frag_mask.astype(np.uint8)

        # -------------------------
        # FG (fast vectorised)
        # -------------------------
        fg = gaussian_from_coords(Z, Y, X, pt, sigma=1.0)

        # -------------------------
        # BG (all other clicks)
        # -------------------------
        other_points = points[:i] + points[i + 1 :]
        bg = heatmap_from_points(Z, Y, X, other_points, sigma=1.0)

        # -------------------------
        # SAVE
        # -------------------------
        sitk.WriteImage(ct_img, str(out_img / f"{frag_id}_0000.mha"))
        save_img(fg, ct_img, out_img / f"{frag_id}_0001.mha")
        save_img(bg, ct_img, out_img / f"{frag_id}_0002.mha")
        save_label(label_out, ct_img, out_lbl / f"{frag_id}.mha")

    print(f"\nTOTAL CASE TIME: {time.time() - t0:.3f}s")


# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--clicks_root", required=True)

    parser.add_argument("--out_images", required=True)
    parser.add_argument("--out_labels", required=True)

    args = parser.parse_args()

    images = Path(args.images)
    labels = Path(args.labels)
    clicks_root = Path(args.clicks_root)

    out_images = Path(args.out_images)
    out_labels = Path(args.out_labels)

    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    ct_files = sorted(images.glob("*_0000.mha"))

    print(f"Found {len(ct_files)} CT cases")

    for ct in ct_files:
        case_id = ct.name.replace("_0000.mha", "")
        label_path = labels / f"{case_id}.mha"

        if not label_path.exists():
            print(f"[WARN] missing label {case_id}")
            continue

        process_case(
            ct,
            label_path,
            clicks_root,
            out_images,
            out_labels,
            case_id,
        )


if __name__ == "__main__":
    main()