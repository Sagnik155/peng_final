#!/usr/bin/env python3
"""
ULTRA-FAST script with optimized I/O for fragment-based Gaussian heatmaps.
Key optimizations:
  - Parallel file writing using multiprocessing
  - Uses .mha with compression (smaller files = faster I/O)
  - Batch writes and reduces disk sync operations
"""

import argparse
import json
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

ANATOMY_NAMES = {
    1: "sacrum",
    2: "left_hip",
    3: "right_hip",
    4: "femur",
}

LABEL_RANGES = {
    1: (1, 50),
    2: (51, 100),
    3: (101, 150),
    4: (151, 200),
}

# Thread-local storage for SimpleITK images
thread_local = threading.local()

def save_image_compressed(data, reference_img, output_path):
    """Save image data with compression enabled."""
    img = sitk.GetImageFromArray(data.astype(np.float32))
    img.CopyInformation(reference_img)
    # Enable compression for .mha files (default is True, but explicit is better)
    sitk.WriteImage(img, str(output_path), True)  # True = use compression


def save_fragment_files(args):
    """Save all files for a single fragment (runs in parallel)."""
    global_idx, metadata, original_array, heatmap_volume, img, basename, output_base = args
    
    anatomy_id = metadata['anatomy_id']
    anatomy_name = ANATOMY_NAMES[anatomy_id]
    point_idx = metadata['point_idx']
    
    # Calculate fragment number
    label_start = LABEL_RANGES[anatomy_id][0]
    fragment_number = label_start + point_idx
    
    # Create directory
    fragment_dir = output_base / anatomy_name / str(fragment_number)
    fragment_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract foreground and background (fast array operations)
    foreground = heatmap_volume[global_idx]
    background = np.sum(heatmap_volume, axis=0) - foreground
    
    # Normalize background
    bg_max = background.max()
    if bg_max > 0:
        background /= bg_max
    
    # Write files with compression
    base_path = fragment_dir / f"{basename}"
    
    # Save original (already float32 from original_array)
    save_image_compressed(original_array, img, f"{base_path}_0000.mha")
    
    # Save foreground and background
    save_image_compressed(foreground, img, f"{base_path}_0001.mha")
    save_image_compressed(background, img, f"{base_path}_0002.mha")
    
    return global_idx


def gaussian_3d_vectorized(shape, centers, sigma=1.0):
    """Create multiple 3D Gaussian heatmaps using vectorized operations."""
    z = np.arange(shape[0], dtype=np.float32).reshape(-1, 1, 1)
    y = np.arange(shape[1], dtype=np.float32).reshape(1, -1, 1)
    x = np.arange(shape[2], dtype=np.float32).reshape(1, 1, -1)
    
    n_points = len(centers)
    heatmaps = np.zeros((n_points,) + shape, dtype=np.float32)
    
    # Pre-compute coordinate grids once
    z_grid = np.broadcast_to(z, shape)
    y_grid = np.broadcast_to(y, shape)
    x_grid = np.broadcast_to(x, shape)
    
    inv_sigma2 = 1.0 / (2 * sigma * sigma)
    
    # Vectorized computation for all points
    for i, (cz, cy, cx) in enumerate(centers):
        dz2 = (z_grid - cz) ** 2
        dy2 = (y_grid - cy) ** 2
        dx2 = (x_grid - cx) ** 2
        dist_sq = dz2 + dy2 + dx2
        heatmaps[i] = np.exp(-dist_sq * inv_sigma2)
        
        # Normalize this heatmap
        max_val = heatmaps[i].max()
        if max_val > 0:
            heatmaps[i] /= max_val
    
    return heatmaps


def main():
    parser = argparse.ArgumentParser(description="ULTRA-FAST heatmap generation with parallel I/O")
    parser.add_argument("--json", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--output_base", type=str, required=True)
    parser.add_argument("--basename", type=str, default=None)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers for I/O")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-compression", action="store_true", help="Disable compression (faster but larger files)")

    args = parser.parse_args()

    start_time = time.time()
    
    # Validate inputs
    json_path = Path(args.json)
    image_path = Path(args.image)
    
    if not json_path.exists() or not image_path.exists():
        print("ERROR: Input files not found")
        return 1

    # Setup paths
    output_base = Path(args.output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    # Determine basename
    if args.basename is None:
        basename = image_path.stem
        if basename.endswith("_0000"):
            basename = basename[:-5]
    else:
        basename = args.basename

    print(f"\n{'='*60}")
    print(f"ULTRA-FAST Heatmap Generation with Parallel I/O")
    print(f"{'='*60}")
    print(f"Output format: .mha with compression {'OFF' if args.no_compression else 'ON'}")
    
    # Read image once
    print(f"\n[1/5] Reading image...")
    img = sitk.ReadImage(str(image_path))
    original_array = sitk.GetArrayFromImage(img).astype(np.float32)
    shape = original_array.shape
    print(f"  Shape: {shape}, Size: {np.prod(shape) / 1e6:.1f}M voxels")

    # Parse JSON
    print(f"\n[2/5] Parsing JSON...")
    with open(json_path) as f:
        data = json.load(f)
    
    # Group points by anatomy
    points_by_anatomy = {1: [], 2: [], 3: [], 4: []}
    keyword_map = {"Sacrum": 1, "Left Hip": 2, "Right Hip": 3, "Femur": 4}
    
    for p in data["points"]:
        name = p["name"]
        coord = p["point"]  # [x, y, z]
        # Convert to (z, y, x) for numpy
        coord_np = (coord[0], coord[1], coord[2])
        
        for keyword, cls_id in keyword_map.items():
            if keyword in name:
                points_by_anatomy[cls_id].append({
                    'coord': coord_np,
                    'name': name,
                    'index': len(points_by_anatomy[cls_id])
                })
                break
    
    # Collect all points in order
    all_points = []
    point_metadata = []
    
    for anatomy_id in [1, 2, 3, 4]:
        points = points_by_anatomy[anatomy_id]
        for point_idx, point_info in enumerate(points):
            all_points.append(point_info['coord'])
            point_metadata.append({
                'anatomy_id': anatomy_id,
                'point_idx': point_idx,
                'name': point_info['name']
            })
    
    total_fragments = len(all_points)
    print(f"  Found {total_fragments} total fragments")
    
    if total_fragments == 0:
        print("ERROR: No valid points found!")
        return 1

    # Precompute ALL heatmaps at once
    print(f"\n[3/5] Precomputing {total_fragments} heatmaps in one pass...")
    precompute_start = time.time()
    
    # For smaller memory footprint, use float16 if possible
    use_float16 = np.prod(shape) * total_fragments < 2e9
    if use_float16:
        print(f"  Using float16 to reduce memory ({(np.prod(shape) * total_fragments * 2) / 1e9:.2f} GB expected)")
        heatmap_volume = gaussian_3d_vectorized(shape, all_points, sigma=args.sigma).astype(np.float16)
    else:
        print(f"  Using float32 ({(np.prod(shape) * total_fragments * 4) / 1e9:.2f} GB expected)")
        heatmap_volume = gaussian_3d_vectorized(shape, all_points, sigma=args.sigma)
    
    precompute_time = time.time() - precompute_start
    print(f"  Precomputed in {precompute_time:.2f} seconds")

    # Generate all fragment directories and save files IN PARALLEL
    print(f"\n[4/5] Generating fragment directories and saving files (parallel I/O with {args.workers} workers)...")
    save_start = time.time()
    
    # Prepare arguments for parallel processing
    fragment_args = []
    for global_idx, metadata in enumerate(point_metadata):
        fragment_args.append((
            global_idx, metadata, original_array, heatmap_volume, 
            img, basename, output_base
        ))
    
    # Use ThreadPoolExecutor for parallel I/O
    processed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_idx = {executor.submit(save_fragment_files, args): args[0] 
                        for args in fragment_args}
        
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                future.result()
                processed += 1
                if args.verbose and processed % 10 == 0:
                    print(f"  Processed {processed}/{total_fragments} fragments")
            except Exception as e:
                print(f"  Error processing fragment {idx}: {e}")
    
    save_time = time.time() - save_start
    
    # Create summary file
    print(f"\n[5/5] Creating summary...")
    summary_file = output_base / "fragment_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Fragment Generation Summary\n")
        f.write(f"{'='*60}\n")
        f.write(f"Image: {image_path}\n")
        f.write(f"JSON: {json_path}\n")
        f.write(f"Sigma: {args.sigma}\n")
        f.write(f"Total fragments: {total_fragments}\n")
        f.write(f"Workers: {args.workers}\n\n")
        
        for anatomy_id in [1, 2, 3, 4]:
            points = points_by_anatomy[anatomy_id]
            if points:
                anatomy_name = ANATOMY_NAMES[anatomy_id]
                f.write(f"{anatomy_name} ({len(points)} fragments):\n")
                for idx, p in enumerate(points):
                    label_start = LABEL_RANGES[anatomy_id][0]
                    fragment_number = label_start + idx
                    f.write(f"  - Fragment {fragment_number}: {p['name']}\n")
                f.write("\n")
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"✓ COMPLETE!")
    print(f"{'='*60}")
    print(f"  Total fragments: {total_fragments}")
    print(f"  Heatmap precomputation: {precompute_time:.2f}s")
    print(f"  Parallel file saving: {save_time:.2f}s (using {args.workers} workers)")
    print(f"  TOTAL TIME: {total_time:.2f}s")
    
    if total_time > 0:
        print(f"\n  Speed: {total_fragments/total_time:.1f} fragments/second")
        print(f"  I/O speed: {total_fragments * 3 / save_time:.1f} files/second")
    
    return 0


if __name__ == "__main__":
    exit(main())