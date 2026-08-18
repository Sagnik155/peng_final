import torch
from pathlib import Path
import sys
import os

from shared.io_utils import read_mha, write_mha
from shared.click_parser import parse_clicks
from shared.pelvic_femur_router import get_image_info, classify_pelvic_femur
from fragment_seg.unet3d import LightweightFragmentUNet
from parallel_infer.batch_runner import run_inference_on_clicks
from fusion.merge_overlap import merge_overlapping_predictions
from fusion.resolve_conflicts import resolve_and_pack_instances
from baseline_adapter.run_baseline import get_anatomy_mask

def run_case_pipeline(case_id: str, click_strategy: str, model_path: Path, output_dir: Path):
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

    base_dir = Path(__file__).resolve().parent.parent
    raw_dir = base_dir / "data_main" / "raw" / case_id
    image_path = raw_dir / "image.mha"
    click_path = base_dir / "data_main" / "clicks" / click_strategy / case_id / "peripelvic-fragment-clicks.json"
    
    if not image_path.exists() or not click_path.exists():
        print(f"Skipping {case_id}: missing image or clicks.")
        return
        
    print(f"--- Running Prediction for Case {case_id} ({click_strategy}) ---")

    img_info = get_image_info(image_path)
    routing_class = classify_pelvic_femur(
        img_info["spacing_x"], img_info["spacing_y"], img_info["spacing_z"],
        img_info["physical_x_mm"], img_info["physical_z_mm"]
    )
    print(f"Case Routing: {routing_class}")
    
    img_array, metadata = read_mha(image_path)
    clicks = parse_clicks(click_path)
    
    model = LightweightFragmentUNet(in_channels=2, out_channels=2, base_filters=16).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
  
    print("Running 3D UNet ROI forward passes...")
    roi_predictions = run_inference_on_clicks(img_array, clicks, model, device)
    
    print("Merging overlapping predictions...")
    merged_predictions = merge_overlapping_predictions(roi_predictions, iou_threshold=0.5)

    print("Resolving boundary conflicts and packing instance IDs...")
    anatomy_mask, anatomy_mapping = get_anatomy_mask(image_path)
    final_instance_mask = resolve_and_pack_instances(merged_predictions, anatomy_mask, anatomy_mapping)
    
    output_path = output_dir / case_id / "pelvic-fracture-segmentation.mha"
    write_mha(final_instance_mask, metadata, output_path)
    print(f"Successfully saved output to: {output_path}\n")

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    test_model_path = base_dir / "method" / "fragment_seg" / "fragment_unet_test.pth"
    output_directory = base_dir / "data_main" / "predictions"
    raw_directory = base_dir / "data_main" / "raw"
    
    click_strategy = "uniformly_sampled"
    
    if not raw_directory.exists():
        print(f"Raw directory not found at {raw_directory}")
        sys.exit(1)

    case_folders = sorted([d.name for d in raw_directory.iterdir() if d.is_dir()])
    
    if not case_folders:
        print("No cases found in the raw directory.")
    else:
        print(f"Found {len(case_folders)} cases. Starting batch inference...\n")

        for case_id in case_folders:
            run_case_pipeline(case_id, click_strategy, test_model_path, output_directory)
            
        print("=========================================")
        print("All cases processed successfully!")
        print("=========================================")
