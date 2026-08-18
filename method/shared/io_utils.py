import SimpleITK as sitk
import numpy as np
from pathlib import Path

def read_mha(file_path: Path) -> tuple[np.ndarray, dict]:
    img = sitk.ReadImage(str(file_path))
    arr = sitk.GetArrayFromImage(img)
    metadata = {
        "spacing": img.GetSpacing(),
        "origin": img.GetOrigin(),
        "direction": img.GetDirection()
    }
    return arr, metadata

def write_mha(array: np.ndarray, metadata: dict, output_path: Path) -> None:
    img = sitk.GetImageFromArray(array)
    img.SetSpacing(metadata["spacing"])
    img.SetOrigin(metadata["origin"])
    img.SetDirection(metadata["direction"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(output_path), useCompression=True)
