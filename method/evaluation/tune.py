import numpy as np
import torch
from pathlib import Path
import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared.io_utils import read_mha
from shared.click_parser import parse_clicks
from fragment_seg.unet3d import LightweightFragmentUNet
from parallel_infer.batch_runner import run_inference_on_clicks
from fusion.merge_overlap import merge_overlapping_predictions
from baseline_adapter.run_baseline import get_anatomy_mask
from fusion.resolve_conflicts import resolve_and_pack_instances
from evaluation.pengwin_metrics import evaluate_full_case

def run_grid_search(val_cases: list[str], model_path: Path):
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    base_dir = Path(__file__).resolve().parent.parent.parent
    
    # Load model once
    model = LightweightFragmentUNet(in_channels=2, out_channels=2, base_filters=16).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Hyperparameter Grid
    confidence_thresholds = [0.4, 0.5, 0.6]
    iou_merge_thresholds = [0.3, 0.5, 0.7]
    
    results_log = []
    
    for conf_thresh in confidence_thresholds:
        for iou_thresh in iou_merge_thresholds:
            print(f"\n--- Testing Config: Confidence > {conf_thresh}, Merge IoU > {iou_thresh} ---")
            
            config_iou_scores = []
            config_hd95_scores = []
            
            for case_id in val_cases:
                image_path = base_dir / "data_main" / "raw" / case_id / "image.mha"
                gt_path = base_dir / "data_main" / "raw" / case_id / "label.mha"
                click_path = base_dir / "data_main" / "clicks" / "uniformly_sampled" / case_id / "peripelvic-fragment-clicks.json"
                
                if not image_path.exists() or not gt_path.exists():
                    continue
                    
                img_array, metadata = read_mha(image_path)
                gt_array, _ = read_mha(gt_path)
                clicks = parse_clicks(click_path)
                
                # 1. Run inference with variable confidence
                model.eval()
                roi_predictions = []
                with torch.no_grad():
                    for click in clicks:
                        if click['anatomy'] == "background":
                            continue
                        from roi.roi_extractor import extract_inference_roi, place_roi_back
                        x_tensor, placement_info = extract_inference_roi(img_array, click['z'], click['y'], click['x'])
                        logits = model(x_tensor.to(device))
                        probs = torch.sigmoid(logits[0, 0, ...])
                        
                        # Apply Grid Search Confidence Threshold
                        pred_mask = probs.cpu().numpy() > conf_thresh
                        global_mask = place_roi_back(pred_mask, placement_info)
                        roi_predictions.append({'global_mask': global_mask, 'anatomy': click['anatomy']})
                
                # 2. Apply Grid Search Merge Threshold
                merged_predictions = merge_overlapping_predictions(roi_predictions, iou_threshold=iou_thresh)
                
                # 3. Resolve & Pack
                anatomy_mask, anatomy_mapping = get_anatomy_mask(image_path)
                final_mask = resolve_and_pack_instances(merged_predictions, anatomy_mask, anatomy_mapping)
                
                # 4. Score Case
                spacing = metadata.get('spacing', (1.0, 1.0, 1.0)) # Default to 1mm if spacing missing
                metrics = evaluate_full_case(final_mask, gt_array, spacing)
                
                config_iou_scores.append(metrics["Mean_IoU"])
                config_hd95_scores.append(metrics["Mean_HD95"])
                
            mean_config_iou = np.mean(config_iou_scores)
            mean_config_hd95 = np.mean([x for x in config_hd95_scores if not np.isinf(x)])
            
            print(f"Result -> Mean IoU: {mean_config_iou:.4f} | Mean HD95: {mean_config_hd95:.2f}mm")
            
            results_log.append({
                "confidence": conf_thresh,
                "iou_merge": iou_thresh,
                "Mean_IoU": mean_config_iou,
                "Mean_HD95": mean_config_hd95
            })
            
    # Find and print the best configuration based on highest IoU
    best_config = max(results_log, key=lambda x: x['Mean_IoU'])
    print("\n=========================================")
    print(f"OPTIMAL HYPERPARAMETERS FOUND:")
    print(f"Confidence Threshold: {best_config['confidence']}")
    print(f"Merge IoU Threshold: {best_config['iou_merge']}")
    print(f"Expected Challenge IoU: {best_config['Mean_IoU']:.4f}")
    print("=========================================")

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent.parent
    test_model = base_dir / "method" / "fragment_seg" / "fragment_unet_test.pth"
    
    # Run optimization on a small validation subset (e.g., cases 001 and 002)
    val_subset = ["001", "002"] 
    run_grid_search(val_subset, test_model)