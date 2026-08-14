import SimpleITK as sitk

def get_image_info(image_nii_path) -> dict:
    img = sitk.ReadImage(str(image_nii_path))
    arr = sitk.GetArrayFromImage(img)
    sp = img.GetSpacing()
    return {
        "dim_z": arr.shape[0], "dim_y": arr.shape[1], "dim_x": arr.shape[2],
        "spacing_z": sp[0], "spacing_y": sp[1], "spacing_x": sp[2],
        "physical_z_mm": sp[0] * arr.shape[0], "physical_x_mm": sp[2] * arr.shape[2],
    }

def classify_pelvic_femur(spacing_x, spacing_y, spacing_z, physical_x_mm, physical_z_mm):
    if physical_x_mm <= 285.35:
        if spacing_x <= 0.71:
            return "pelvic"
        elif spacing_z <= 0.98:
            return "femur"
        else:
            return "pelvic" if spacing_y <= 0.91 else "femur"
    else:
        if spacing_z <= 0.68:
            return "pelvic" if physical_z_mm <= 193.55 else "femur"
        else:
            return "pelvic" if physical_z_mm <= 390.78 else "femur"