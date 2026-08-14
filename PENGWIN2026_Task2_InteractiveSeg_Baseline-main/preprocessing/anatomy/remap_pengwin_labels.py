#!/usr/bin/env python3

import argparse
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import os
import shutil


def remap_labels(arr):
    """
    Map:
      0        -> 0 (background)
      1–50     -> 1 (sacrum)
      51–100   -> 2 (left hip)
      101–150  -> 3 (right hip)
      151–200  -> 4 (femur)
    """
    out = np.zeros_like(arr, dtype=np.uint8)

    out[(arr >= 1) & (arr <= 50)] = 1
    out[(arr >= 51) & (arr <= 100)] = 2
    out[(arr >= 101) & (arr <= 150)] = 3
    out[(arr >= 151) & (arr <= 200)] = 4

    return out


def process_file(in_path, out_path):
    img = sitk.ReadImage(str(in_path))
    arr = sitk.GetArrayFromImage(img)

    remapped = remap_labels(arr)

    out_img = sitk.GetImageFromArray(remapped)
    out_img.CopyInformation(img)

    sitk.WriteImage(out_img, str(out_path))


def main():
    parser = argparse.ArgumentParser(
        description="Remap PENGWIN labels to 0–4 classes"
    )

    parser.add_argument(
        "--input_labels",
        type=str,
        required=True,
        help="Path to labelsTr directory (input)"
    )

    parser.add_argument(
        "--output_labels",
        type=str,
        required=True,
        help="Path to labelsTr directory (output)"
    )

    args = parser.parse_args()

    input_dir = Path(args.input_labels)
    output_dir = Path(args.output_labels)

    os.makedirs(args.input_labels.rstrip('/') + "_unmapped", exist_ok=True)

    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.mha"))

    print(f"Found {len(files)} label files")

    for f in files:
        shutil.copy(f, os.path.join(args.input_labels.rstrip('/') + "_unmapped", os.path.basename(f)))
        out_path = output_dir / f.name
        process_file(f, out_path)
        print(f"[OK] {f.name}")


if __name__ == "__main__":
    main()
