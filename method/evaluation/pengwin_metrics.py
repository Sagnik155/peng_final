import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree
import SimpleITK as sitk
from pathlib import Path
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared.io_utils import read_mha

def extract_surface(mask: np.ndarray) -> np.ndarray:
    eroded = binary_erosion(mask, iterations=1)
    boundary = np.logical_xor(mask, eroded)
    return np.argwhere(boundary)

def compute_surface_distances(pred_mask: np.ndarray, gt_mask: np.ndarray, spacing: tuple) -> tuple:
    surf_pred = extract_surface(pred_mask)
    surf_gt = extract_surface(gt_mask)
    
    if len(surf_pred) == 0 or len(surf_gt) == 0:
        return np.array([np.inf]), np.array([np.inf])

    surf_pred_mm = surf_pred * np.array(spacing)
    surf_gt_mm = surf_gt * np.array(spacing)
    tree_pred = cKDTree(surf_pred_mm)
    tree_gt = cKDTree(surf_gt_mm)
    
    dists_pred_to_gt, _ = tree_gt.query(surf_pred_mm)
    dists_gt_to_pred, _ = tree_pred.query(surf_gt_mm)
    
    return dists_pred_to_gt, dists_gt_to_pred

def compute_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray, spacing: tuple) -> dict:
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    iou = float(intersection) / float(union) if union > 0 else 0.0
    
    d1, d2 = compute_surface_distances(pred_mask, gt_mask, spacing)
    
    if np.isinf(d1[0]) or np.isinf(d2[0]):
        return {"IoU": 0.0, "HD95": np.inf, "ASSD": np.inf}
        
    all_dists = np.concatenate([d1, d2])
    hd95 = np.percentile(all_dists, 95)
    assd = (np.sum(d1) + np.sum(d2)) / (len(d1) + len(d2))
    
    return {"IoU": iou, "HD95": hd95, "ASSD": assd}

def evaluate_full_case(pred_array: np.ndarray, gt_array: np.ndarray, spacing: tuple) -> dict:
    pred_ids = np.unique(pred_array)[1:] # Skip background 0
    gt_ids = np.unique(gt_array)[1:]
    
    case_metrics = {"IoU": [], "HD95": [], "ASSD": []}
    
    for gt_id in gt_ids:
        gt_mask = (gt_array == gt_id)
        best_iou = 0.0
        best_hd95 = np.inf
        best_assd = np.inf
        
        for pred_id in pred_ids:
            pred_mask = (pred_array == pred_id)
            metrics = compute_metrics(pred_mask, gt_mask, spacing)
            
            if metrics["IoU"] > best_iou:
                best_iou = metrics["IoU"]
                best_hd95 = metrics["HD95"]
                best_assd = metrics["ASSD"]
                
        case_metrics["IoU"].append(best_iou)
        if not np.isinf(best_hd95):
            case_metrics["HD95"].append(best_hd95)
            case_metrics["ASSD"].append(best_assd)
            
    return {
        "Mean_IoU": np.mean(case_metrics["IoU"]) if case_metrics["IoU"] else 0.0,
        "Mean_HD95": np.mean(case_metrics["HD95"]) if case_metrics["HD95"] else np.inf,
        "Mean_ASSD": np.mean(case_metrics["ASSD"]) if case_metrics["ASSD"] else np.inf
    }
