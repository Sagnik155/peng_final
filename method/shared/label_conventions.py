# Anatomical label ranges assigned by PENGWIN 2026
LABEL_RANGES = {
    "sacrum": (1, 50),
    "left_hipbone": (51, 100),
    "right_hipbone": (101, 150),
    "femur": (151, 200)
}

def get_anatomy_from_label(label_id: int) -> str:
    for anatomy, (min_val, max_val) in LABEL_RANGES.items():
        if min_val <= label_id <= max_val:
            return anatomy
    return "background"