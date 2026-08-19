import os
import sys
from pathlib import Path
import torch

# 1. Add 'method' directory to Python path so internal imports work
base_dir = Path(__file__).resolve().parent
sys.path.append(os.path.abspath(os.path.join(base_dir, "method")))

from baseline_adapter.run_baseline import get_anatomy_mask
from fragment_seg.unet3d import LightweightFragmentUNet
from fusion.merge_overlap import merge_overlapping_predictions
from fusion.resolve_conflicts import resolve_and_pack_instances
from parallel_infer.batch_runner import run_inference_on_clicks
from shared.click_parser import parse_clicks
from shared.io_utils import read_mha, write_mha
from shared.pelvic_femur_router import classify_pelvic_femur, get_image_info


def run_case_pipeline(
    image_path: Path, click_path: Path, output_path: Path, model_path: Path
):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  print(f"--- Running Prediction on {image_path.name} ---")

  # Load Data & Model
  img_array, metadata = read_mha(image_path)
  clicks = parse_clicks(click_path)

  model = LightweightFragmentUNet(
      in_channels=2, out_channels=2, base_filters=16
  ).to(device)
  model.load_state_dict(torch.load(model_path, map_location=device))

  # Pipeline Execution
  print("Running 3D UNet ROI forward passes...")
  roi_predictions = run_inference_on_clicks(img_array, clicks, model, device)

  print("Merging overlapping predictions...")
  merged_predictions = merge_overlapping_predictions(
      roi_predictions, iou_threshold=0.5
  )

  print("Resolving boundary conflicts and packing instance IDs...")
  anatomy_mask, anatomy_mapping = get_anatomy_mask(image_path)
  final_instance_mask = resolve_and_pack_instances(
      merged_predictions, anatomy_mask, anatomy_mapping
  )

  # Ensure the output is an integer type (range 0-200)
  final_instance_mask = final_instance_mask.astype("uint8")

  # Write Final Output
  write_mha(final_instance_mask, metadata, output_path)
  print(f"Successfully saved output to: {output_path}\n")


if __name__ == "__main__":
  input_dir = Path("/input")

  # 1. Official PENGWIN Image Input Directory
  image_input_dir = input_dir / "images" / "peripelvic-fracture-ct"
  if not image_input_dir.exists():
    image_input_dir = input_dir  # Fallback search

  # 2. Official PENGWIN Output Directory
  output_dir = (
      Path("/output") / "images" / "peripelvic-fracture-ct-segmentation"
  )
  output_dir.mkdir(parents=True, exist_ok=True)

  # Locate the input CT image (.mha)
  try:
    image_path = list(image_input_dir.rglob("*.mha"))[0]
  except IndexError:
    print(f"No .mha file found in {image_input_dir}")
    sys.exit(1)

  # Locate the click annotations (.json)
  try:
    # Searches entire /input for the click json
    click_path = [
        f for f in input_dir.rglob("*.json") if f.name != "inputs.json"
    ][0]
  except IndexError:
    # If only inputs.json exists, fall back to any JSON
    try:
      click_path = list(input_dir.rglob("*.json"))[0]
    except IndexError:
      print("No click JSON found in /input")
      sys.exit(1)

  # Save output using the standard challenge naming convention
  output_path = output_dir / "output.mha"

  # Path to the decoupled model checkpoint
  model_path = Path("/opt/ml/model/fragment_unet_test.pth")

  run_case_pipeline(image_path, click_path, output_path, model_path)