import random
from pathlib import Path
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

def get_train_val_splits(val_ratio: float = 0.2, seed: int = 42) -> tuple[list[str], list[str]]:
   
    case_dirs = [d for d in config.RAW_DATA_DIR.iterdir() if d.is_dir()]
    case_ids = sorted([d.name for d in case_dirs])
    
    if not case_ids:
        raise ValueError(f"No cases found in {config.RAW_DATA_DIR}")

    random.seed(seed)
    random.shuffle(case_ids)
    
    num_val = int(len(case_ids) * val_ratio)
    val_cases = case_ids[:num_val]
    train_cases = case_ids[num_val:]
    
    return train_cases, val_cases

if __name__ == "__main__":
    train, val = get_train_val_splits()
    print(f"Total cases: {len(train) + len(val)}")
    print(f"Train cases: {len(train)}")
    print(f"Val cases: {len(val)}")
