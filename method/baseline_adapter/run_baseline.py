import numpy as np
import SimpleITK as sitk
from pathlib import Path
import subprocess
import tempfile
import shutil
import os

def get_anatomy_mask(image_path: Path) -> tuple[np.ndarray, dict]:

    anatomy_mapping = {
        "sacrum": 1,
        "left_hipbone": 2,
        "right_hipbone": 3,
        "femur": 4,
        "universal_fallback": 255
    }

    base_dir = Path(__file__).resolve().parent.parent.parent
    weights_dir = base_dir / "method" / "baseline_adapter" / "weights" / "phase1"
    
    if weights_dir.exists() and any(weights_dir.iterdir()):
        try:
            print("Running Baseline Phase 1 nnUNet...")
            with tempfile.TemporaryDirectory() as temp_dir:
                input_dir = Path(temp_dir) / "input"
                output_dir = Path(temp_dir) / "output"
                input_dir.mkdir()
                output_dir.mkdir()
                
                temp_input_path = input_dir / f"case_0000.mha"
                shutil.copy(image_path, temp_input_path)
                
                cmd = [
                    "nnUNetv2_predict",
                    "-i", str(input_dir),
                    "-o", str(output_dir),
                    "-d", "DatasetXXX_Pengwin", 
                    "-c", "3d_fullres",
                    "-f", "0"
                ]

                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                output_file = output_dir / "case.mha"
                if output_file.exists():
                    img = sitk.ReadImage(str(output_file))
                    anatomy_mask = sitk.GetArrayFromImage(img)
                    return anatomy_mask, anatomy_mapping
                    
        except Exception as e:
            print(f"Phase 1 nnUNet execution failed or weights missing. Reason: {e}")
            print("Falling back to local testing mask...")

    img = sitk.ReadImage(str(image_path))
    arr = sitk.GetArrayFromImage(img)
    
    anatomy_mask = np.zeros_like(arr, dtype=np.uint8)
    anatomy_mask[arr > 200] = 255 
    
    return anatomy_mask, anatomy_mapping
