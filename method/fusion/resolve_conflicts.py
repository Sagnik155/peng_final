import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared.label_conventions import LABEL_RANGES

def resolve_and_pack_instances(merged_predictions: list[dict], anatomy_mask: np.ndarray, anatomy_mapping: dict) -> np.ndarray:
    final_volume = np.zeros_like(anatomy_mask, dtype=np.uint16)

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

        if current_id > LABEL_RANGES[anatomy][1]:
            print(f"Warning: Ran out of instance IDs for {anatomy}!")
            continue
            
        frag_mask = pred['global_mask']

        baseline_class_idx = anatomy_mapping.get(anatomy, -1)

        valid_bone_mask = (anatomy_mask == baseline_class_idx) | (anatomy_mask == anatomy_mapping.get("universal_fallback", -1))
        frag_mask = np.logical_and(frag_mask, valid_bone_mask)

        unoccupied_mask = (final_volume == 0)
        final_placement = np.logical_and(frag_mask, unoccupied_mask)
        
        final_volume[final_placement] = current_id
        next_id[anatomy] += 1
        
    return final_volume
