import SimpleITK as sitk
import numpy as np
from pathlib import Path

prediction_path = Path("data_main/predictions/001/pelvic-fracture-segmentation.mha")
raw_image_path = Path("data_main/raw/001/image.mha")

if not prediction_path.exists():
    print(f"File not found: {prediction_path}")
else:
    # 1. Load Prediction
    pred_img = sitk.ReadImage(str(prediction_path))
    pred_arr = sitk.GetArrayFromImage(pred_img)
    
    # 2. Load Raw CT for Metadata Comparison
    raw_img = sitk.ReadImage(str(raw_image_path))
    
    unique_labels = np.unique(pred_arr)
    non_zero_voxels = np.count_nonzero(pred_arr)
    
    print("--- PREDICTION HEALTH CHECK ---")
    print(f"Prediction Array Shape: {pred_arr.shape}")
    print(f"Unique Label IDs Present: {unique_labels}")
    print(f"Total Non-Zero Predicted Voxels: {non_zero_voxels}")
    
    # Verify spatial alignment metadata
    print(f"Spacing Match: {pred_img.GetSpacing() == raw_img.GetSpacing()}")
    print(f"Origin Match:  {pred_img.GetOrigin() == raw_img.GetOrigin()}")