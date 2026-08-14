import json
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from collections import defaultdict
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

def get_fragment_bounding_box(label_array, target_id):
    indices = np.where(label_array == target_id)
    if len(indices[0]) == 0:
        return None
    return (
        np.min(indices[0]), np.max(indices[0]),
        np.min(indices[1]), np.max(indices[1]),
        np.min(indices[2]), np.max(indices[2])
    )

def run_experiment():
    # Expanding ROI sizes to catch large pelvic bones
    test_roi_sizes = [64, 96, 128, 160, 192, 256]
    
    results = defaultdict(lambda: defaultdict(list))
    overlap_counts = defaultdict(int)
    fragment_max_dims = []

    for case_dir in config.RAW_DATA_DIR.iterdir():
        if not case_dir.is_dir(): continue
        case_id = case_dir.name
        
        label_path = case_dir / "label.mha"
        if not label_path.exists(): continue
            
        print(f"Processing Case {case_id}...")
        label_img = sitk.ReadImage(str(label_path))
        label_array = sitk.GetArrayFromImage(label_img) # (z, y, x)
        
        unique_labels = np.unique(label_array)
        fragment_data = {}
        for lab in unique_labels:
            if lab == 0: continue
            bbox = get_fragment_bounding_box(label_array, lab)
            
            # Track how big these fragments actually are in voxels
            z_len = bbox[1] - bbox[0]
            y_len = bbox[3] - bbox[2]
            x_len = bbox[5] - bbox[4]
            max_dim = max(z_len, y_len, x_len)
            fragment_max_dims.append(max_dim)
            
            volume = np.sum(label_array == lab)
            fragment_data[lab] = {"bbox": bbox, "volume": volume, "max_dim": max_dim}

        for strategy in config.CLICK_STRATEGIES:
            click_json_path = config.CLICKS_DIR / strategy / case_id / "peripelvic-fragment-clicks.json"
            if not click_json_path.exists(): continue
                
            with open(click_json_path, 'r') as f:
                click_data = json.load(f)
                
            clicks_per_fragment = defaultdict(int)

            for point_info in click_data.get("points", []):
                coords = point_info["point"]
                
                # The PENGWIN JSON natively stores [z, y, x] voxel coordinates
                idx_z, idx_y, idx_x = int(coords[0]), int(coords[1]), int(coords[2])
                
                # Safe clamping just in case
                idx_z = max(0, min(idx_z, label_array.shape[0] - 1))
                idx_y = max(0, min(idx_y, label_array.shape[1] - 1))
                idx_x = max(0, min(idx_x, label_array.shape[2] - 1))
                
                landed_label = label_array[idx_z, idx_y, idx_x]
                if landed_label == 0:
                    continue 
                    
                clicks_per_fragment[landed_label] += 1
                
                frag_info = fragment_data[landed_label]
                z_min, z_max, y_min, y_max, x_min, x_max = frag_info["bbox"]
                
                for roi_size in test_roi_sizes:
                    half_roi = roi_size // 2
                    contained = (
                        (idx_z - half_roi <= z_min) and (idx_z + half_roi >= z_max) and
                        (idx_y - half_roi <= y_min) and (idx_y + half_roi >= y_max) and
                        (idx_x - half_roi <= x_min) and (idx_x + half_roi >= x_max)
                    )
                    
                    results[strategy][roi_size].append({
                        "contained": contained,
                        "volume": frag_info["volume"]
                    })
            
            for lab, count in clicks_per_fragment.items():
                if count > 1:
                    overlap_counts[strategy] += 1

    print("\n--- EXPERIMENT RESULTS ---")
    if fragment_max_dims:
        print(f"Average Fragment Max Dimension: {np.mean(fragment_max_dims):.1f} voxels")
        print(f"95th Percentile Max Dimension: {np.percentile(fragment_max_dims, 95):.1f} voxels")
        print(f"Absolute Largest Fragment: {np.max(fragment_max_dims)} voxels")

    for strategy in config.CLICK_STRATEGIES:
        print(f"\nStrategy: {strategy}")
        print(f"Fragments with >1 click (Merge Risk): {overlap_counts[strategy]}")
        for roi_size in test_roi_sizes:
            data = results[strategy][roi_size]
            if not data: continue
            
            containment_rate = sum(1 for d in data if d["contained"]) / len(data)
            print(f"  ROI {roi_size}^3 Containment Rate: {containment_rate*100:.1f}%")

if __name__ == "__main__":
    run_experiment()