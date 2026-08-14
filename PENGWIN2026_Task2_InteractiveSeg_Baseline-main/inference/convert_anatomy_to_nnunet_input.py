#!/usr/bin/env python3
"""
Script to create Gaussian heatmaps from a single JSON clicks file and a single image.
Outputs: 
  - {basename}_0000.{ext} (original image copy)
  - {basename}_0001.{ext} (sacrum heatmap)
  - {basename}_0002.{ext} (left hip heatmap)
  - {basename}_0003.{ext} (right hip heatmap)
  - {basename}_0004.{ext} (femur heatmap)
"""

import argparse
import json
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter

CLASS_MAP = {
    "Sacrum": 1,
    "Left Hip": 2,
    "Right Hip": 3,
    "Femur": 4,
}

# Keywords to match in point names
KEYWORD_MAP = {
    "Sacrum": 1,
    "Left Hip": 2,
    "Right Hip": 3,
    "Femur": 4,
}


def gaussian_3d_fast(shape, center, sigma=1.0):
    """
    Create a 3D Gaussian heatmap efficiently using pre-computed coordinates.
    This avoids creating full meshgrid for each point.
    """
    # Create coordinate arrays for each dimension
    z = np.arange(shape[0], dtype=np.float32)
    y = np.arange(shape[1], dtype=np.float32)
    x = np.arange(shape[2], dtype=np.float32)
    
    # Center coordinates
    cx, cy, cz = center
    
    # Compute squared distances using broadcasting (much faster than meshgrid)
    dz2 = (z - cz) ** 2
    dy2 = (y - cy) ** 2
    dx2 = (x - cx) ** 2
    
    # Add dimensions for broadcasting: (z,1,1) + (1,y,1) + (1,1,x)
    dist_sq = dz2[:, np.newaxis, np.newaxis] + dy2[np.newaxis, :, np.newaxis] + dx2[np.newaxis, np.newaxis, :]
    
    return np.exp(-dist_sq / (2 * sigma ** 2))


def gaussian_filter_method(shape, center, sigma=1.0):
    """
    Alternative method: Create a delta function at center and apply Gaussian filter.
    This can be faster for large sigma values.
    """
    heat = np.zeros(shape, dtype=np.float32)
    cz, cy, cx = center
    # Ensure coordinates are within bounds
    if 0 <= cz < shape[0] and 0 <= cy < shape[1] and 0 <= cx < shape[2]:
        heat[int(round(cz)), int(round(cy)), int(round(cx))] = 1.0
    return gaussian_filter(heat, sigma=sigma, mode='constant', cval=0.0)


def parse_points(json_path):
    """Parse JSON and group points by class based on name keywords."""
    with open(json_path) as f:
        data = json.load(f)

    grouped = {1: [], 2: [], 3: [], 4: []}

    for p in data["points"]:
        name = p["name"]
        coord = p["point"]  # [x, y, z]

        # convert to z, y, x (numpy indexing)
        coord = [coord[2], coord[1], coord[0]]

        # Check which class this point belongs to
        for keyword, cls_id in KEYWORD_MAP.items():
            if keyword in name:
                grouped[cls_id].append(coord)
                break

    return grouped


def create_heatmaps_fast(image, grouped_points, sigma=1.0):
    """Create 4 heatmaps for each class using vectorized operations."""
    shape = sitk.GetArrayFromImage(image).shape  # z,y,x
    heatmaps = {}
    
    # Pre-allocate heatmaps as zeros
    for cls_id in range(1, 5):
        if len(grouped_points[cls_id]) > 0:
            # Start with zeros
            heat = np.zeros(shape, dtype=np.float32)
            
            # Accumulate all Gaussians for this class
            for pt in grouped_points[cls_id]:
                heat += gaussian_3d_fast(shape, pt, sigma=sigma)
            
            # Normalize if needed
            if np.max(heat) > 0:
                heat = heat / np.max(heat)
            
            heatmaps[cls_id] = heat
        else:
            # No points for this class, return zeros
            heatmaps[cls_id] = np.zeros(shape, dtype=np.float32)
    
    return heatmaps


def create_heatmaps_filter_method(image, grouped_points, sigma=1.0):
    """
    Alternative method using Gaussian filter.
    This can be faster when there are many points per class.
    """
    shape = sitk.GetArrayFromImage(image).shape
    heatmaps = {}
    
    for cls_id in range(1, 5):
        if len(grouped_points[cls_id]) > 0:
            heat = np.zeros(shape, dtype=np.float32)
            
            # Place delta functions at each point
            for pt in grouped_points[cls_id]:
                cz, cy, cx = pt
                if 0 <= cz < shape[0] and 0 <= cy < shape[1] and 0 <= cx < shape[2]:
                    heat[int(round(cz)), int(round(cy)), int(round(cx))] += 1.0
            
            # Apply Gaussian filter
            heat = gaussian_filter(heat, sigma=sigma, mode='constant', cval=0.0)
            
            # Normalize
            if np.max(heat) > 0:
                heat = heat / np.max(heat)
            
            heatmaps[cls_id] = heat
        else:
            heatmaps[cls_id] = np.zeros(shape, dtype=np.float32)
    
    return heatmaps


def save_image(data, reference_img, out_path):
    """Save image data using reference image metadata."""
    img = sitk.GetImageFromArray(data.astype(np.float32))
    img.CopyInformation(reference_img)
    
    # Determine compression for nii.gz
    if str(out_path).endswith('.nii.gz'):
        sitk.WriteImage(img, str(out_path), True)
    else:
        sitk.WriteImage(img, str(out_path), True)
    
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create Gaussian heatmaps from single JSON clicks file and image",
        epilog="Outputs: {basename}_0000.{ext} (image) and {basename}_0001-0004.{ext} (heatmaps)"
    )

    parser.add_argument(
        "--json",
        type=str,
        required=True,
        help="Path to JSON file containing click points"
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image (.mha or .nii.gz)"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for the converted files"
    )

    parser.add_argument(
        "--basename",
        type=str,
        default=None,
        help="Basename for output files (default: stem of image filename)"
    )

    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="Sigma for Gaussian heatmap (default: 1.0)"
    )

    parser.add_argument(
        "--method",
        type=str,
        choices=["direct", "filter"],
        default="direct",
        help="Method for generating heatmaps: 'direct' (faster for small sigma/few points) or 'filter' (faster for large sigma/many points)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information"
    )

    args = parser.parse_args()

    # Validate input files
    json_path = Path(args.json)
    if not json_path.exists():
        print(f"ERROR: JSON file not found: {json_path}")
        return 1

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"ERROR: Image file not found: {image_path}")
        return 1

    # Set output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine basename
    if args.basename is None:
        basename = image_path.stem
        # Remove trailing _0000 if present (nnU-Net format)
        if basename.endswith("_0000"):
            basename = basename[:-5]
    else:
        basename = args.basename

    # Get file extension
    extension = image_path.suffix
    if image_path.name.endswith('.nii.gz'):
        extension = '.nii.gz'

    print(f"\nProcessing:")
    print(f"  JSON: {json_path}")
    print(f"  Image: {image_path}")
    print(f"  Output dir: {output_dir}")
    print(f"  Basename: {basename}")
    print(f"  Extension: {extension}")
    print(f"  Sigma: {args.sigma}")
    print(f"  Method: {args.method}")

    # Read image
    print(f"\nReading image...")
    img = sitk.ReadImage(str(image_path))
    data_shape = sitk.GetArrayFromImage(img).shape
    print(f"  Image shape (z,y,x): {data_shape}")

    # Parse JSON
    print(f"\nParsing JSON...")
    try:
        grouped_points = parse_points(json_path)
        
        # Print point counts
        print(f"  Points found:")
        for cls_id, cls_name in CLASS_MAP.items():
            count = len(grouped_points[cls_name])
            print(f"    {cls_id}: {count} point(s)")
            
            if args.verbose and count > 0:
                for pt in grouped_points[cls_id]:
                    print(f"      - (z={pt[0]}, y={pt[1]}, x={pt[2]})")
    
    except Exception as e:
        print(f"ERROR parsing JSON: {e}")
        return 1

    # Check if any points were found
    total_points = sum(len(pts) for pts in grouped_points.values())
    if total_points == 0:
        print(f"\nWARNING: No valid points found in JSON!")
        print(f"  Please check that point names contain: Sacrum, Left Hip, Right Hip, or Femur")

    # Create heatmaps
    print(f"\nCreating heatmaps...")
    
    import time
    start_time = time.time()
    
    if args.method == "filter":
        heatmaps = create_heatmaps_filter_method(img, grouped_points, sigma=args.sigma)
    else:
        heatmaps = create_heatmaps_fast(img, grouped_points, sigma=args.sigma)
    
    elapsed = time.time() - start_time
    print(f"  Heatmaps created in {elapsed:.2f} seconds")

    # Save original image as _0000
    output_image_path = output_dir / f"{basename}_0000{extension}"
    print(f"\nSaving files:")
    save_image(sitk.GetArrayFromImage(img), img, output_image_path)

    # Save heatmaps as _0001 to _0004
    for cls_id in range(1, 5):
        cls_name = [k for k, v in CLASS_MAP.items() if v == cls_id][0]
        output_heatmap_path = output_dir / f"{basename}_000{cls_id}{extension}"
        save_image(heatmaps[cls_id], img, output_heatmap_path)
        
        if args.verbose:
            heatmap_data = heatmaps[cls_id]
            nonzero_voxels = np.sum(heatmap_data > 0)
            print(f"    {cls_name} heatmap: range=[{heatmap_data.min():.4f}, {heatmap_data.max():.4f}], mean={heatmap_data.mean():.4f}, nonzero={nonzero_voxels}")

    print(f"\n✓ Done! Files saved to: {output_dir}")
    print(f"  - {basename}_0000{extension} (original image)")
    print(f"  - {basename}_0001{extension} (sacrum heatmap)")
    print(f"  - {basename}_0002{extension} (left hip heatmap)")
    print(f"  - {basename}_0003{extension} (right hip heatmap)")
    print(f"  - {basename}_0004{extension} (femur heatmap)")

    return 0


if __name__ == "__main__":
    exit(main())