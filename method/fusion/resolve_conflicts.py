import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared.label_conventions import LABEL_RANGES

def resolve_and_pack_instances(merged_predictions: list[dict], anatomy_mask: np.ndarray, anatomy_mapping: dict) -> np.ndarray:
    """
    merged_predictions: list of distinct fragment dictionaries.
    anatomy_mask: 3D numpy array from the Baseline Phase 1 (5-class mask or fallback).
    anatomy_mapping: dict mapping anatomy string to baseline's integer class.
    
    Returns the final instance segmentation volume.
    """
    final_volume = np.zeros_like(anatomy_mask, dtype=np.uint16)
    
    # Track the next available ID for each anatomy to pack them tightly
    next_id = {
        "sacrum": LABEL_RANGES["sacrum"][0],
        "left_hipbone": LABEL_RANGES["left_hipbone"][0],
        "right_hipbone": LABEL_RANGES["right_hipbone"][0],
        "femur": LABEL_RANGES["femur"][0]
    }
    
    for pred in merged_predictions:
        anatomy = pred['anatomy']
        if anatomy == "background":
            continue
            
        current_id = next_id[anatomy]
        
        # Ensure we don't exceed the maximum ID for this anatomy
        if current_id > LABEL_RANGES[anatomy][1]:
            print(f"Warning: Ran out of instance IDs for {anatomy}!")
            continue
            
        frag_mask = pred['global_mask']
        
        # --- THE MISSING VARIABLE ASSIGNMENT ---
        baseline_class_idx = anatomy_mapping.get(anatomy, -1)
        
        # 1. Anatomy Masking: Fragment must only exist where Phase 1 says the bone exists
        # (Or where our universal fallback mask detects raw bone during testing)
        valid_bone_mask = (anatomy_mask == baseline_class_idx) | (anatomy_mask == anatomy_mapping.get("universal_fallback", -1))
        frag_mask = np.logical_and(frag_mask, valid_bone_mask)
        
        # 2. Boundary Conflict Resolution: If final_volume already has a fragment here,
        # we let the first one keep the voxels.
        unoccupied_mask = (final_volume == 0)
        final_placement = np.logical_and(frag_mask, unoccupied_mask)
        
        final_volume[final_placement] = current_id
        next_id[anatomy] += 1
        
    return final_volume