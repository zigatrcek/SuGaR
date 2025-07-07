#!/usr/bin/env python3
"""
Script to prepare a COLMAP dataset from images with optional resizing.

This script:
1. Takes a directory containing images
2. Optionally resizes them by a specified factor
3. Calls gaussian_splatting/convert.py to generate COLMAP dataset

Usage:
    python prepare_colmap_dataset.py -s /path/to/images --resize_factor 0.5
    python prepare_colmap_dataset.py -s /path/to/images --max_size 1024
    python prepare_colmap_dataset.py -s /path/to/images  # No resizing
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
from PIL import Image
import shutil
from typing import Optional, Tuple

def get_image_files(directory: Path) -> list:
    """Get all image files from directory."""
    print(f"🔍 Scanning for image files in: {directory}")
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    for ext in image_extensions:
        found_lower = list(directory.glob(f'*{ext}'))
        found_upper = list(directory.glob(f'*{ext.upper()}'))
        if found_lower:
            print(f"   Found {len(found_lower)} {ext} files")
        if found_upper:
            print(f"   Found {len(found_upper)} {ext.upper()} files")
        image_files.extend(found_lower)
        image_files.extend(found_upper)
    
    return sorted(image_files)

def calculate_new_size(original_size: Tuple[int, int], resize_factor: Optional[float] = None, 
                      max_size: Optional[int] = None) -> Tuple[int, int]:
    """Calculate new image size based on resize factor or max size constraint."""
    width, height = original_size
    
    if resize_factor is not None:
        new_width = int(width * resize_factor)
        new_height = int(height * resize_factor)
    elif max_size is not None:
        # Resize maintaining aspect ratio so largest dimension = max_size
        if max(width, height) <= max_size:
            return original_size  # No need to resize
        
        if width > height:
            new_width = max_size
            new_height = int(height * max_size / width)
        else:
            new_height = max_size
            new_width = int(width * max_size / height)
    else:
        return original_size
    
    # Ensure dimensions are at least 1
    new_width = max(1, new_width)
    new_height = max(1, new_height)
    
    return new_width, new_height

def resize_image(input_path: Path, output_path: Path, new_size: Tuple[int, int], quality: int = 95):
    """Resize a single image."""
    try:
        with Image.open(input_path) as img:
            # Convert to RGB if necessary (handles RGBA, P mode, etc.)
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')
            
            # Resize with high-quality resampling
            resized_img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Save with high quality
            if output_path.suffix.lower() in ['.jpg', '.jpeg']:
                resized_img.save(output_path, 'JPEG', quality=quality, optimize=True)
            else:
                resized_img.save(output_path)
            
        return True
    except Exception as e:
        print(f"Error resizing {input_path}: {e}")
        return False

def prepare_input_directory(source_dir: Path, target_dir: Path, resize_factor: Optional[float] = None,
                          max_size: Optional[int] = None, quality: int = 95) -> bool:
    """Prepare input directory with optionally resized images."""
    
    print(f"\n{'='*60}")
    print(f"PREPARING INPUT DIRECTORY")
    print(f"{'='*60}")
    
    # Create input directory
    input_dir = target_dir / "input"
    print(f"Creating input directory: {input_dir}")
    input_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    print(f"Scanning for image files in: {source_dir}")
    image_files = get_image_files(source_dir)
    
    if not image_files:
        print(f"❌ No image files found in {source_dir}")
        print(f"   Supported formats: .jpg, .jpeg, .png, .bmp, .tiff, .tif")
        return False
    
    print(f"✅ Found {len(image_files)} images in {source_dir}")
    
    # Show first few files for verification
    print(f"\nFirst few image files:")
    for i, img_file in enumerate(image_files[:5]):
        print(f"  {i+1}. {img_file.name}")
    if len(image_files) > 5:
        print(f"  ... and {len(image_files) - 5} more")
    
    # Processing info
    if resize_factor:
        print(f"\n📐 Will resize images by factor: {resize_factor}")
    elif max_size:
        print(f"\n📐 Will limit max dimension to: {max_size} pixels")
    else:
        print(f"\n📐 No resizing - copying images as-is")
    
    print(f"💾 JPEG quality setting: {quality}")
    print(f"\nProcessing images...")
    print(f"-" * 60)
    
    # Process images
    successful = 0
    total_original_size = 0
    total_new_size = 0
    
    for i, img_path in enumerate(image_files):
        try:
            # Get original size
            with Image.open(img_path) as img:
                original_size = img.size
                original_area = original_size[0] * original_size[1]
                total_original_size += original_area
            
            # Calculate new size
            new_size = calculate_new_size(original_size, resize_factor, max_size)
            new_area = new_size[0] * new_size[1]
            total_new_size += new_area
            
            # Output path
            output_path = input_dir / img_path.name
            
            # Resize or copy
            if new_size != original_size:
                print(f"🔄 [{i+1:3d}/{len(image_files)}] Resizing: {img_path.name}")
                print(f"    {original_size[0]}x{original_size[1]} -> {new_size[0]}x{new_size[1]} "
                      f"({new_area/original_area:.2%} of original)")
                
                if resize_image(img_path, output_path, new_size, quality):
                    successful += 1
                    file_size = output_path.stat().st_size / (1024*1024)  # MB
                    print(f"    ✅ Saved ({file_size:.1f} MB)")
                else:
                    print(f"    ❌ Failed to resize")
            else:
                print(f"📋 [{i+1:3d}/{len(image_files)}] Copying: {img_path.name}")
                print(f"    {original_size[0]}x{original_size[1]} (no resize needed)")
                
                shutil.copy2(img_path, output_path)
                successful += 1
                file_size = output_path.stat().st_size / (1024*1024)  # MB
                print(f"    ✅ Copied ({file_size:.1f} MB)")
                
        except Exception as e:
            print(f"❌ [{i+1:3d}/{len(image_files)}] Error processing {img_path.name}: {e}")
    
    print(f"\n{'='*60}")
    print(f"IMAGE PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successfully processed: {successful}/{len(image_files)} images")
    
    if resize_factor or max_size:
        reduction = (1 - total_new_size / total_original_size) * 100
        print(f"📊 Total pixel reduction: {reduction:.1f}%")
        print(f"   Original total pixels: {total_original_size:,}")
        print(f"   New total pixels: {total_new_size:,}")
    
    # Check output directory
    output_files = list(input_dir.glob("*"))
    total_size_mb = sum(f.stat().st_size for f in output_files if f.is_file()) / (1024*1024)
    print(f"💾 Total output size: {total_size_mb:.1f} MB")
    print(f"📁 Output directory: {input_dir}")
    
    return successful > 0

def run_colmap_conversion(dataset_path: Path, skip_matching: bool = False, no_gpu: bool = False) -> bool:
    """Run gaussian_splatting/convert.py on the dataset."""
    
    print(f"\n{'='*60}")
    print(f"RUNNING COLMAP CONVERSION")
    print(f"{'='*60}")
    
    convert_script = Path(__file__).parent / "gaussian_splatting" / "convert.py"
    
    print(f"🔍 Looking for conversion script at: {convert_script}")
    
    if not convert_script.exists():
        print(f"❌ Error: COLMAP conversion script not found at {convert_script}")
        print(f"   Please make sure the gaussian_splatting submodule is properly initialized")
        return False
    
    print(f"✅ Conversion script found")
    
    cmd = ["python3", str(convert_script), "-s", str(dataset_path)]
    
    if skip_matching:
        cmd.append("--skip_matching")
        print(f"⚡ Skip matching enabled - will skip feature matching step")
    
    if no_gpu:
        cmd.append("--no_gpu")
        print(f"🔌 GPU usage disabled for COLMAP")
    
    print(f"🚀 Running command: {' '.join(cmd)}")
    print(f"📁 Dataset path: {dataset_path}")
    print(f"\nCOLMAP Output:")
    print(f"-" * 40)
    
    try:
        # Run with real-time output
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Print output in real-time
        output_lines = []
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(f"COLMAP: {output.strip()}")
                output_lines.append(output)
        
        return_code = process.poll()
        
        if return_code == 0:
            print(f"\n{'='*60}")
            print(f"✅ COLMAP conversion completed successfully!")
            print(f"{'='*60}")
            
            # Check what was created
            sparse_dir = dataset_path / "sparse" / "0"
            if sparse_dir.exists():
                print(f"📁 Sparse reconstruction created at: {sparse_dir}")
                
                # Count cameras and points
                cameras_file = sparse_dir / "cameras.bin"
                images_file = sparse_dir / "images.bin"
                points_file = sparse_dir / "points3D.bin"
                
                if cameras_file.exists():
                    print(f"📷 Camera parameters: {cameras_file}")
                if images_file.exists():
                    print(f"🖼️  Image data: {images_file}")
                if points_file.exists():
                    print(f"🎯 3D points: {points_file}")
            
            return True
        else:
            print(f"\n{'='*60}")
            print(f"❌ COLMAP conversion failed with return code {return_code}")
            print(f"{'='*60}")
            return False
            
    except Exception as e:
        print(f"\n❌ Error running COLMAP conversion: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Prepare COLMAP dataset from images with optional resizing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Resize images to half size and create COLMAP dataset
  python prepare_colmap_dataset.py -s /path/to/images --resize_factor 0.5
  
  # Limit max dimension to 1024 pixels
  python prepare_colmap_dataset.py -s /path/to/images --max_size 1024
  
  # No resizing, just create COLMAP dataset
  python prepare_colmap_dataset.py -s /path/to/images
  
  # Specify custom output directory
  python prepare_colmap_dataset.py -s /path/to/images -o /custom/output --resize_factor 0.75
        """
    )
    
    parser.add_argument("-s", "--source", type=str, required=True,
                        help="Path to directory containing source images")
    parser.add_argument("-o", "--output", type=str,
                        help="Output directory for COLMAP dataset (default: source directory)")
    parser.add_argument("--resize_factor", type=float,
                        help="Resize factor (e.g., 0.5 for half size, 2.0 for double size)")
    parser.add_argument("--max_size", type=int,
                        help="Maximum size for largest dimension (maintains aspect ratio)")
    parser.add_argument("--quality", type=int, default=95,
                        help="JPEG quality for resized images (default: 95)")
    parser.add_argument("--skip_matching", action="store_true",
                        help="Skip feature matching in COLMAP (useful for re-runs)")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Disable GPU usage for COLMAP")

    args = parser.parse_args()

    # Validate arguments
    if args.resize_factor is not None and args.max_size is not None:
        print("Error: Cannot specify both --resize_factor and --max_size")
        sys.exit(1)
    
    if args.resize_factor is not None and args.resize_factor <= 0:
        print("Error: resize_factor must be positive")
        sys.exit(1)
    
    if args.max_size is not None and args.max_size <= 0:
        print("Error: max_size must be positive")
        sys.exit(1)
    
    # Set up paths
    source_dir = Path(args.source).resolve()
    if not source_dir.exists():
        print(f"❌ Error: Source directory {source_dir} does not exist")
        sys.exit(1)
    
    output_dir = Path(args.output).resolve() if args.output else source_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"COLMAP DATASET PREPARATION")
    print(f"{'='*60}")
    print(f"📁 Source directory: {source_dir}")
    print(f"📁 Output directory: {output_dir}")
    print(f"🔧 JPEG quality: {args.quality}")
    
    if args.resize_factor:
        print(f"📐 Resize factor: {args.resize_factor}")
        if args.resize_factor < 1.0:
            print(f"   → Images will be {args.resize_factor:.1%} of original size (smaller)")
        else:
            print(f"   → Images will be {args.resize_factor:.1%} of original size (larger)")
    elif args.max_size:
        print(f"📐 Max size: {args.max_size} pixels")
        print(f"   → Largest dimension will be limited to {args.max_size}px")
    else:
        print(f"📐 No resizing will be performed")
    
    if args.skip_matching:
        print(f"⚡ Skip matching: Enabled (faster re-runs)")
    else:
        print(f"🔍 Feature matching: Enabled (full COLMAP processing)")
        
    print(f"\nStarting processing...")

    # Prepare input directory
    if not prepare_input_directory(source_dir, output_dir, args.resize_factor, 
                                 args.max_size, args.quality):
        print(f"\n❌ Failed to prepare input directory")
        sys.exit(1)
    
    # Run COLMAP conversion
    if not run_colmap_conversion(output_dir, args.skip_matching, args.no_gpu):
        print(f"\n❌ COLMAP conversion failed")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"🎉 DATASET PREPARATION COMPLETE!")
    print(f"{'='*60}")
    print(f"📁 COLMAP dataset ready at: {output_dir}")
    print(f"🚀 You can now run SuGaR with:")
    print(f"   python train_full_pipeline.py -s {output_dir} -r dn_consistency --high_poly True")
    print(f"\n💡 Tip: Use 'dn_consistency' regularization for best mesh quality!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
