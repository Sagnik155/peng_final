import numpy as np
from pathlib import Path
import SimpleITK as sitk
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared.io_utils import read_mha

def compute_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Calculates the Dice Similarity Coefficient between two boolean masks."""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = pred_mask.sum() + gt_mask.sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return 2.0 * float(intersection) / float(union)

def evaluate_case(pred_array: np.ndarray, gt_array: np.ndarray):
    """
    Evaluates a single case for Fracture Dice, Merge Errors, and Split Errors.
    """
    pred_ids = np.unique(pred_array)
    pred_ids = pred_ids[pred_ids != 0]
    
    gt_ids = np.unique(gt_array)
    gt_ids = gt_ids[gt_ids != 0]

    # 1. Fracture Dice (Average across all matched ground truth fragments)
    dice_scores = []
    for gt_id in gt_ids:
        gt_mask = (gt_array == gt_id)
        
        # Find the predicted fragment that overlaps the most with this GT fragment
        best_dice = 0.0
        for pred_id in pred_ids:
            pred_mask = (pred_array == pred_id)
            dice = compute_dice(pred_mask, gt_mask)
            if dice > best_dice:
                best_dice = dice
                
        dice_scores.append(best_dice)
        
    avg_fracture_dice = np.mean(dice_scores) if dice_scores else 0.0

    # 2. Merge & Split Error Detection
    merge_errors = 0
    split_errors = 0
    
    # Merge Error: One predicted fragment overlaps with multiple GT fragments
    for pred_id in pred_ids:
        pred_mask = (pred_array == pred_id)
        overlapped_gt_fragments = 0
        for gt_id in gt_ids:
            gt_mask = (gt_array == gt_id)
            if np.logical_and(pred_mask, gt_mask).sum() > 0:
                overlapped_gt_fragments += 1
        if overlapped_gt_fragments > 1:
            merge_errors += (overlapped_gt_fragments - 1)

    # Split Error: Multiple predicted fragments overlap with a single GT fragment
    for gt_id in gt_ids:
        gt_mask = (gt_array == gt_id)
        overlapped_pred_fragments = 0
        for pred_id in pred_ids:
            pred_mask = (pred_array == pred_id)
            if np.logical_and(gt_mask, pred_mask).sum() > 0:
                overlapped_pred_fragments += 1
        if overlapped_pred_fragments > 1:
            split_errors += (overlapped_pred_fragments - 1)

    return {
        "Fracture_Dice": avg_fracture_dice,
        "Merge_Errors": merge_errors,
        "Split_Errors": split_errors
    }

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent.parent
    case_id = "001"
    
    pred_path = base_dir / "data_main" / "predictions" / case_id / "pelvic-fracture-segmentation.mha"
    gt_path = base_dir / "data_main" / "raw" / case_id / "label.mha"
    
    if pred_path.exists() and gt_path.exists():
        pred_arr, _ = read_mha(pred_path)
        gt_arr, _ = read_mha(gt_path)
        
        print(f"--- Evaluation Results for Case {case_id} ---")
        metrics = evaluate_case(pred_arr, gt_arr)
        
        for metric, value in metrics.items():
            if "Dice" in metric:
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")
    else:
        print("Prediction or Ground Truth file missing.")