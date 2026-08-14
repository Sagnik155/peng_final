import argparse
import json
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import random


CLICK_TYPES = [
    "uniformly_sampled",
    "euclidean_distance_transform",
    "center_of_mass",
    "boundary_internal_margin",
]

CLASS_MAP = {
    "sacrum": 1,
    "left hip": 2,
    "right hip": 3,
    "femur": 4,
}


def gaussian_3d(shape, center, sigma=1.0):
    """Create a 3D Gaussian heatmap."""
    zz, yy, xx = np.meshgrid(
        np.arange(shape[0]),
        np.arange(shape[1]),
        np.arange(shape[2]),
        indexing="ij",
    )

    cz, cy, cx = center

    dist_sq = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2
    return np.exp(-dist_sq / (2 * sigma ** 2))


def parse_points(json_path):
    """Parse JSON and group points by class."""
    with open(json_path) as f:
        data = json.load(f)

    grouped = {1: [], 2: [], 3: [], 4: []}

    for p in data["points"]:
        name = p["name"].lower()
        coord = p["point"]  # [x, y, z]

        # convert to z, y, x (numpy indexing)
        coord = [coord[2], coord[1], coord[0]]

        for key, cls_id in CLASS_MAP.items():
            if key in name:
                grouped[cls_id].append(coord)

    return grouped


def create_heatmaps(image, grouped_points):
    """Create 4 heatmaps."""
    shape = sitk.GetArrayFromImage(image).shape  # z,y,x

    heatmaps = {}

    for cls_id in range(1, 5):
        heat = np.zeros(shape, dtype=np.float32)

        for pt in grouped_points[cls_id]:
            heat += gaussian_3d(shape, pt, sigma=1.0)

        heat = np.clip(heat, 0, 1)

        heatmaps[cls_id] = heat

    return heatmaps


def save_heatmap(heat, reference_img, out_path):
    img = sitk.GetImageFromArray(heat)
    img.CopyInformation(reference_img)
    sitk.WriteImage(img, str(out_path))


def main():
    parser = argparse.ArgumentParser(description="Add Gaussian heatmaps to PENGWIN nnU-Net dataset")

    parser.add_argument("--input_images", type=str, required=True,
                        help="Path to imagesTr (nnU-Net format)")
    parser.add_argument("--clicks_root", type=str, required=True,
                        help="Path to PENGWIN26_task2_train_clicks")
    parser.add_argument("--output_images", type=str, required=True,
                        help="Output imagesTr with heatmaps")


    args = parser.parse_args()

    input_images = Path(args.input_images)
    clicks_root = Path(args.clicks_root)
    output_images = Path(args.output_images)

    output_images.mkdir(parents=True, exist_ok=True)



    cases = sorted([p for p in input_images.glob("*_0000.mha")])


    print(f"Found {len(cases)} cases")

    # assign click type (1/4 each)
    random.seed(42)
    random.shuffle(cases)

    assignments = {}
    for i, case in enumerate(cases):
        click_type = CLICK_TYPES[i % 4]
        case_id = case.stem.replace("_0000", "")
        assignments[case_id] = click_type

    for case_path in cases:
        case_id = case_path.stem.replace("_0000", "")
        click_type = assignments[case_id]

        print(f"[{case_id}] using {click_type}")

        # load image
        img = sitk.ReadImage(str(case_path))

        json_path = (
            clicks_root
            / click_type
            / case_id
            / "peripelvic-fragment-clicks.json"
        )

        if not json_path.exists():
            print(f"[WARN] Missing JSON for {case_id}")
            grouped = {1: [], 2: [], 3: [], 4: []}
        else:
            grouped = parse_points(json_path)

        heatmaps = create_heatmaps(img, grouped)

        # copy original image
        sitk.WriteImage(img, str(output_images / f"{case_id}_0000.mha"))

        # save 4 channels
        for cls_id in range(1, 5):
            out_path = output_images / f"{case_id}_000{cls_id}.mha"
            save_heatmap(heatmaps[cls_id], img, out_path)


if __name__ == "__main__":
    main()
