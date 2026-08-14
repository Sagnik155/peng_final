#!/usr/bin/env python3
"""
Pre-resample images to nnUNet target spacing
OPTIMIZED: Only _0000.mha files are identical, so only resample those once.
Heatmaps (_0001, _0002) are unique and must be resampled individually.

Target spacing: [1.0, 0.78125, 0.78125] (z, y, x)
"""

import argparse
from pathlib import Path
import SimpleITK as sitk
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import shutil

TARGET_SPACING = [0.78125, 0.78125, 1.0]  

def resample_image_fast(input_path, output_path, target_spacing, use_nearest=False):
    """
    Resample a single image to target spacing
    
    Args:
        input_path: Path to input image
        output_path: Path to output image
        target_spacing: Target spacing (z, y, x)
        use_nearest: If True, use nearest neighbor and convert to uint8
    """
    # Read image
    img = sitk.ReadImage(str(input_path))
    original_spacing = img.GetSpacing()
    original_size = img.GetSize()
    
    # Calculate new size
    old_size_physical = [original_size[i] * original_spacing[i] for i in range(3)]
    new_size = [int(round(old_size_physical[i] / target_spacing[i])) for i in range(3)]
    
    # Configure resampler
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetOutputDirection(img.GetDirection())
    
    # Choose interpolation method
    if use_nearest:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampled = resampler.Execute(img)
        
        # Convert to uint8
        resampled_array = sitk.GetArrayFromImage(resampled)
        resampled_array = np.clip(resampled_array, 0, 255).astype(np.uint8)
        
        # Create new image with same geometry as resampled
        output_img = sitk.GetImageFromArray(resampled_array)
        output_img.SetSpacing(resampler.GetOutputSpacing())
        output_img.SetOrigin(img.GetOrigin())
        output_img.SetDirection(img.GetDirection())
    else:
        resampler.SetInterpolator(sitk.sitkLinear)
        output_img = resampler.Execute(img)
    
    # Write with compression
    sitk.WriteImage(output_img, str(output_path), True)
    
    return new_size



def main():
    parser = argparse.ArgumentParser(description="Pre-resample images to nnUNet target spacing (OPTIMIZED)")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory with .mha files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for resampled files")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--target", type=str, default="anatomy", help="Anatomy or fragments")
    parser.add_argument("--load_spacing", type=str, help="Load pre-saved spacing npy file")
    parser.add_argument("--spacing", type=float, nargs=3, default=[0.78125, 0.78125, 1.0],
                       help="Target spacing (default: 0.78125 0.78125 1.0)")
    
    args = parser.parse_args()
    
    if args.target == 'fragments':
        args.spacing = [0.7820000052452087, 0.7820000052452087, 0.801025390625]

    if args.load_spacing is not None:
        args.spacing = [float(el) for el in np.load(args.load_spacing)]
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all input files
    input_files = sorted(input_dir.glob("*.mha"))
    
    if not input_files:
        print(f"No .mha files found in {input_dir}")
        return 1
    
    # Separate files by type
    files_0000 = [f for f in input_files if '_0000.mha' in f.name]
    files_heatmaps = [f for f in input_files if '_0001.mha' in f.name or '_0002.mha' in f.name]
    files_other = [f for f in input_files if f not in files_0000 and f not in files_heatmaps]
    
    print(f"\n{'='*60}")
    print(f"Pre-resampling Images to nnUNet Target Spacing (OPTIMIZED)")
    print(f"{'='*60}")
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Total files:      {len(input_files)}")
    print(f"  _0000 files:    {len(files_0000)} (IDENTICAL - will resample once)")
    print(f"  Heatmaps:       {len(files_heatmaps)} (UNIQUE - each must be resampled)")
    print(f"  Other:          {len(files_other)}")
    print(f"Target spacing:   {args.spacing[0]:.4f} x {args.spacing[1]:.4f} x {args.spacing[2]:.4f} (z,y,x)")
    print(f"Workers:          {args.workers}")
    print(f"{'='*60}\n")
    
    # Read one sample to show size reduction
    if files_0000:
        sample = sitk.ReadImage(str(files_0000[0]))
        original_spacing = sample.GetSpacing()
        original_size = sample.GetSize()
        old_physical = [original_size[i] * original_spacing[i] for i in range(3)]
        new_size = [int(round(old_physical[i] / args.spacing[i])) for i in range(3)]
        
        original_voxels = original_size[0] * original_size[1] * original_size[2]
        new_voxels = new_size[0] * new_size[1] * new_size[2]
        reduction = (1 - new_voxels / original_voxels) * 100
        
        print(f"Sample image info:")
        print(f"  Original: {original_size[0]}x{original_size[1]}x{original_size[2]} voxels")
        print(f"    Spacing: {original_spacing[0]:.4f}x{original_spacing[1]:.4f}x{original_spacing[2]:.4f} mm")
        print(f"  New:      {new_size[0]}x{new_size[1]}x{new_size[2]} voxels")
        print(f"    Spacing: {args.spacing[0]:.4f}x{args.spacing[1]:.4f}x{args.spacing[2]:.4f} mm")
        print(f"  Voxel reduction: {reduction:.1f}% ({original_voxels:,} → {new_voxels:,} voxels)\n")
        
        # Save original spacing
        save_spacing = output_dir / 'original_spacing.npy'
        np.save(str(save_spacing), np.array(original_spacing))
    
    start_time = time.time()
    
    # Step 1: Resample ONLY ONE _0000 file (they are all identical)
    print("Step 1: Resampling _0000 files (only one needed - they are identical)...")
    print("-" * 40)
    
    copy_0000_time = 0
    if files_0000:
        resample_start = time.time()
        representative_0000 = files_0000[0]
        temp_resampled_0000 = output_dir / "temp_resampled_0000.mha"
        resample_image_fast(representative_0000, temp_resampled_0000, args.spacing, use_nearest=False)
        resample_time = time.time() - resample_start
        print(f"  Resampled 1 representative _0000 file in {resample_time:.1f}s")
        
        # Copy to all _0000 output files
        copy_start = time.time()
        for input_file in files_0000:
            output_file = output_dir / input_file.name
            shutil.copy2(temp_resampled_0000, output_file)
        copy_0000_time = time.time() - copy_start
        print(f"  Copied to {len(files_0000)} _0000 files in {copy_0000_time:.1f}s\n")
        
        # Cleanup temp file
        temp_resampled_0000.unlink()
    else:
        print("  No _0000 files found\n")
        resample_time = 0
    
    # Step 2: Resample ALL heatmap files (they are UNIQUE per fragment)
    print("Step 2: Resampling heatmap files (_0001, _0002) - each is UNIQUE...")
    print("-" * 40)
    
    heatmap_start = time.time()
    successful_heatmaps = 0
    failed_heatmaps = 0
    
    if files_heatmaps:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            for input_file in files_heatmaps:
                output_file = output_dir / input_file.name
                # Use linear interpolation for heatmaps (they contain probabilities)
                future = executor.submit(resample_image_fast, input_file, output_file, args.spacing, use_nearest=False)
                futures[future] = input_file
            
            for future in as_completed(futures):
                input_file = futures[future]
                try:
                    new_size = future.result()
                    successful_heatmaps += 1
                    if successful_heatmaps % 10 == 0:
                        print(f"  Resampled {successful_heatmaps}/{len(files_heatmaps)} heatmap files...")
                except Exception as e:
                    failed_heatmaps += 1
                    print(f"  ✗ Failed: {input_file.name} - {e}")
        
        heatmap_time = time.time() - heatmap_start
        print(f"\n  Resampled {successful_heatmaps} heatmap files in {heatmap_time:.1f}s")
        if failed_heatmaps > 0:
            print(f"  Failed: {failed_heatmaps}")
    else:
        print("  No heatmap files found")
        heatmap_time = 0
        successful_heatmaps = 0
    
    # Step 3: Handle any other files (if they exist)
    successful_other = 0
    failed_other = 0
    other_time = 0
    
    if files_other:
        print("\nStep 3: Resampling other files...")
        print("-" * 40)
        other_start = time.time()
        
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            for input_file in files_other:
                output_file = output_dir / input_file.name
                use_nearest = 'prediction' in str(input_file)
                future = executor.submit(resample_image_fast, input_file, output_file, args.spacing, use_nearest=use_nearest)
                futures[future] = input_file
            
            for future in as_completed(futures):
                input_file = futures[future]
                try:
                    new_size = future.result()
                    successful_other += 1
                except Exception as e:
                    failed_other += 1
                    print(f"  ✗ Failed: {input_file.name} - {e}")
        
        other_time = time.time() - other_start
        print(f"  Resampled {successful_other} other files in {other_time:.1f}s")
        if failed_other > 0:
            print(f"  Failed: {failed_other}")
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"Resampling Complete!")
    print(f"{'='*60}")
    print(f"Summary:")
    print(f"  _0000 files:     {len(files_0000)} (resampled 1, copied {len(files_0000)-1})")
    print(f"  Heatmap files:   {successful_heatmaps}/{len(files_heatmaps)} resampled individually")
    print(f"  Other files:     {successful_other}/{len(files_other)} resampled")
    print(f"  Total time:      {total_time:.2f} seconds")
    
    if files_0000 and len(files_0000) > 1:
        time_saved = (len(files_0000) - 1) * resample_time
        print(f"\n  Time saved:      ~{time_saved:.1f}s (by not re-resampling identical _0000 files)")
    
    print(f"\nOutput saved to: {output_dir}")
    
    # Verify output
    if files_0000:
        output_sample = sitk.ReadImage(str(output_dir / files_0000[0].name))
        print(f"\nVerification - Output file spacing: {output_sample.GetSpacing()}")
        print(f"Expected spacing: {args.spacing}")
    
    return 0


if __name__ == "__main__":
    main()