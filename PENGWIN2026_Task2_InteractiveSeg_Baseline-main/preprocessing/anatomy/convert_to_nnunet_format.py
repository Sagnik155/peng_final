import argparse
from pathlib import Path
import shutil


def copy_case(case_dir, case_id, imagesTr, labelsTr):
    image_path = case_dir / "image.mha"
    label_path = case_dir / "label.mha"

    if not image_path.exists() or not label_path.exists():
        print(f"[WARN] Missing files in {case_dir}")
        return

    img_out = imagesTr / f"{case_id}_0000.mha"
    lbl_out = labelsTr / f"{case_id}.mha"

    shutil.copy2(image_path, img_out)
    shutil.copy2(label_path, lbl_out)

    print(f"[OK] {case_id}")


def main():
    parser = argparse.ArgumentParser(description="Convert PENGWIN dataset to nnU-Net format")

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Root directory containing PENGWIN parts"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output nnU-Net dataset directory"
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    imagesTr = output_dir / "imagesTr"
    labelsTr = output_dir / "labelsTr"

    imagesTr.mkdir(parents=True, exist_ok=True)
    labelsTr.mkdir(parents=True, exist_ok=True)

    # find all part folders
    part_dirs = sorted(input_dir.glob("PENGWIN26_task1_2_train_part*"))

    print(f"Found {len(part_dirs)} parts")

    for part_dir in part_dirs:
        if not part_dir.is_dir():
            continue

        print(f"Processing {part_dir.name}")

        # each subfolder is a case (001, 002, ...)
        for case_dir in sorted(part_dir.iterdir()):
            if not case_dir.is_dir():
                continue

            case_id = case_dir.name  # already like 001, 002, etc.

            copy_case(case_dir, case_id, imagesTr, labelsTr)


if __name__ == "__main__":
    main()
