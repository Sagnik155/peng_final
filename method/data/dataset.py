import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from shared.io_utils import read_mha
from shared.click_parser import parse_clicks
from roi.roi_config import ROI_SHAPE, CLICK_CHANNEL_VALUE

def normalize_ct(img_roi: np.ndarray) -> np.ndarray:
    """Clips CT Hounsfield Units to bone window [-1000, 1000] and scales to [0, 1]."""
    img_roi = np.clip(img_roi, -1000, 1000)
    img_roi = (img_roi + 1000) / 2000.0
    return img_roi.astype(np.float32)

class PengwinFragmentDataset(Dataset):
    def __init__(self, case_ids: list[str], click_strategy: str, is_train: bool = True):
        self.case_ids = case_ids
        self.click_strategy = click_strategy
        self.is_train = is_train
        self.samples = self._build_index()

    def _build_index(self):
        """Creates a flat list of all valid clicks across all provided cases."""
        samples = []
        for case_id in self.case_ids:
            click_path = config.CLICKS_DIR / self.click_strategy / case_id / "peripelvic-fragment-clicks.json"
            if not click_path.exists():
                continue
                
            clicks = parse_clicks(click_path)
            for click in clicks:
                samples.append({
                    "case_id": case_id,
                    "click": click
                })
        return samples

    def _extract_roi(self, array: np.ndarray, center: tuple[int, int, int]):
        """Crops the array to ROI_SHAPE centered around the click, with zero-padding if needed."""
        z, y, x = center
        dz, dy, dx = ROI_SHAPE
        
        z_min, z_max = z - dz//2, z + dz//2
        y_min, y_max = y - dy//2, y + dy//2
        x_min, x_max = x - dx//2, x + dx//2
        
        roi = np.zeros(ROI_SHAPE, dtype=array.dtype)
        
        valid_z_min, valid_z_max = max(0, z_min), min(array.shape[0], z_max)
        valid_y_min, valid_y_max = max(0, y_min), min(array.shape[1], y_max)
        valid_x_min, valid_x_max = max(0, x_min), min(array.shape[2], x_max)
        
        dest_z_min = valid_z_min - z_min
        dest_z_max = dest_z_min + (valid_z_max - valid_z_min)
        dest_y_min = valid_y_min - y_min
        dest_y_max = dest_y_min + (valid_y_max - valid_y_min)
        dest_x_min = valid_x_min - x_min
        dest_x_max = dest_x_min + (valid_x_max - valid_x_min)
        
        roi[dest_z_min:dest_z_max, dest_y_min:dest_y_max, dest_x_min:dest_x_max] = \
            array[valid_z_min:valid_z_max, valid_y_min:valid_y_max, valid_x_min:valid_x_max]
            
        return roi

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        case_id = sample["case_id"]
        click = sample["click"]
        
        image_path = config.RAW_DATA_DIR / case_id / "image.mha"
        img_array, _ = read_mha(image_path)
        
        z = max(0, min(click["z"], img_array.shape[0] - 1))
        y = max(0, min(click["y"], img_array.shape[1] - 1))
        x = max(0, min(click["x"], img_array.shape[2] - 1))
        center = (z, y, x)
        
        # Extract & normalize Image ROI
        img_roi = self._extract_roi(img_array, center)
        img_roi_normalized = normalize_ct(img_roi)
        
        # Create Click Channel ROI
        click_roi = np.zeros(ROI_SHAPE, dtype=np.float32)
        click_roi[ROI_SHAPE[0]//2, ROI_SHAPE[1]//2, ROI_SHAPE[2]//2] = CLICK_CHANNEL_VALUE
        
        # Stack into 2-channel tensor (C, Z, Y, X)
        x_tensor = torch.from_numpy(np.stack([img_roi_normalized, click_roi], axis=0)).float()
        
        result = {"input": x_tensor, "case_id": case_id, "anatomy": click["anatomy"]}
        
        if self.is_train:
            label_path = config.RAW_DATA_DIR / case_id / "label.mha"
            label_array, _ = read_mha(label_path)
            
            target_id = label_array[center[0], center[1], center[2]]
            
            label_roi = self._extract_roi(label_array, center)
            binary_label_roi = (label_roi == target_id).astype(np.float32)
            
            y_tensor = torch.from_numpy(binary_label_roi).unsqueeze(0) 
            result["label"] = y_tensor
            
        return result