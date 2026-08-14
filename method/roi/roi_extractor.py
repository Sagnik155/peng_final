import numpy as np
import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from roi.roi_config import ROI_SHAPE, CLICK_CHANNEL_VALUE

def normalize_ct(img_roi: np.ndarray) -> np.ndarray:
    """Clips CT Hounsfield Units to bone window [-1000, 1000] and scales to [0, 1]."""
    img_roi = np.clip(img_roi, -1000, 1000)
    img_roi = (img_roi + 1000) / 2000.0
    return img_roi.astype(np.float32)

def extract_inference_roi(img_array: np.ndarray, click_z: int, click_y: int, click_x: int):
    """
    Extracts the ROI around a click and returns the tensor alongside the spatial metadata
    needed to map the prediction back to the global volume.
    """
    z, y, x = click_z, click_y, click_x
    dz, dy, dx = ROI_SHAPE
    
    z_min, z_max = z - dz//2, z + dz//2
    y_min, y_max = y - dy//2, y + dy//2
    x_min, x_max = x - dx//2, x + dx//2
    
    roi = np.zeros(ROI_SHAPE, dtype=img_array.dtype)
    
    valid_z_min, valid_z_max = max(0, z_min), min(img_array.shape[0], z_max)
    valid_y_min, valid_y_max = max(0, y_min), min(img_array.shape[1], y_max)
    valid_x_min, valid_x_max = max(0, x_min), min(img_array.shape[2], x_max)
    
    dest_z_min = valid_z_min - z_min
    dest_z_max = dest_z_min + (valid_z_max - valid_z_min)
    dest_y_min = valid_y_min - y_min
    dest_y_max = dest_y_min + (valid_y_max - valid_y_min)
    dest_x_min = valid_x_min - x_min
    dest_x_max = dest_x_min + (valid_x_max - valid_x_min)
    
    roi[dest_z_min:dest_z_max, dest_y_min:dest_y_max, dest_x_min:dest_x_max] = \
        img_array[valid_z_min:valid_z_max, valid_y_min:valid_y_max, valid_x_min:valid_x_max]
        
    click_roi = np.zeros(ROI_SHAPE, dtype=np.float32)
    click_roi[ROI_SHAPE[0]//2, ROI_SHAPE[1]//2, ROI_SHAPE[2]//2] = CLICK_CHANNEL_VALUE
    
    # Return both the stacked tensor and the bounding box info needed for placing it back
    x_tensor = torch.from_numpy(np.stack([roi, click_roi], axis=0)).float().unsqueeze(0) # (1, 2, Z, Y, X)
    
    placement_info = {
        'z_min': z_min, 'z_max': z_max,
        'y_min': y_min, 'y_max': y_max,
        'x_min': x_min, 'x_max': x_max,
        'valid_z_min': valid_z_min, 'valid_z_max': valid_z_max,
        'valid_y_min': valid_y_min, 'valid_y_max': valid_y_max,
        'valid_x_min': valid_x_min, 'valid_x_max': valid_x_max,
        'dest_z_min': dest_z_min, 'dest_z_max': dest_z_max,
        'dest_y_min': dest_y_min, 'dest_y_max': dest_y_max,
        'dest_x_min': dest_x_min, 'dest_x_max': dest_x_max,
        'global_shape': img_array.shape
    }
    
    return x_tensor, placement_info

def place_roi_back(roi_mask: np.ndarray, placement_info: dict) -> np.ndarray:
    """Maps the 128^3 prediction mask back into a full-sized boolean array."""
    global_mask = np.zeros(placement_info['global_shape'], dtype=bool)
    
    valid_roi = roi_mask[
        placement_info['dest_z_min']:placement_info['dest_z_max'],
        placement_info['dest_y_min']:placement_info['dest_y_max'],
        placement_info['dest_x_min']:placement_info['dest_x_max']
    ]
    
    global_mask[
        placement_info['valid_z_min']:placement_info['valid_z_max'],
        placement_info['valid_y_min']:placement_info['valid_y_max'],
        placement_info['valid_x_min']:placement_info['valid_x_max']
    ] = valid_roi
    
    return global_mask