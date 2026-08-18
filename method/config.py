from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_ROOT = BASE_DIR / "data_main"
RAW_DATA_DIR = DATA_ROOT / "raw"
CLICKS_DIR = DATA_ROOT / "clicks"

CLICK_STRATEGIES = [
    "boundary_internal_margin",
    "center_of_mass",
    "euclidean_distance_transform",
    "uniformly_sampled"
]
