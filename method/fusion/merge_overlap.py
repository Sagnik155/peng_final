import numpy as np

def calculate_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Calculates Intersection over Union for two boolean arrays."""
    intersection = np.logical_and(mask1, mask2).sum()
    if intersection == 0:
        return 0.0
    union = np.logical_or(mask1, mask2).sum()
    return float(intersection) / float(union)

def merge_overlapping_predictions(roi_predictions: list[dict], iou_threshold: float = 0.5) -> list[dict]:
    """
    roi_predictions: list of dicts containing:
        - 'global_mask': boolean numpy array of the prediction mapped back to the full CT shape
        - 'anatomy': str (e.g., 'femur')
        
    Returns a collapsed list where overlapping masks of the SAME anatomy are unioned.
    """
    merged_predictions = []
    
    for current_pred in roi_predictions:
        merged = False
        for existing_pred in merged_predictions:
            # Only merge if they belong to the same anatomical region
            if current_pred['anatomy'] != existing_pred['anatomy']:
                continue
                
            iou = calculate_iou(current_pred['global_mask'], existing_pred['global_mask'])
            
            if iou > iou_threshold:
                # Merge the masks using a logical OR
                existing_pred['global_mask'] = np.logical_or(
                    existing_pred['global_mask'], 
                    current_pred['global_mask']
                )
                merged = True
                break
                
        if not merged:
            merged_predictions.append(current_pred)
            
    return merged_predictions