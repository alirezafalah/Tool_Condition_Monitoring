#!/usr/bin/env python3
"""
mask_cleaner.py

Post-processes binary mask images in a directory:
  1. Fills holes (enclosed background regions within the mask, similar to bucket fill).
  2. Keeps only the largest contour (removes stray error dots and noise).
  
Command-line toggles allow running both operations, or either one individually,
with the default order: fill holes first, then keep the largest contour.
"""

import os
import sys
import glob
import re
import time
import argparse
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, List, Optional

import cv2
import numpy as np
from PIL import Image

VALID_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp')


def extract_frame_num(filepath: str) -> float:
    """Extract numeric value from filename for natural numerical sorting."""
    basename = os.path.basename(filepath)
    name = os.path.splitext(basename)[0]
    match = re.match(r"^(\d+\.?\d*)", name)
    if match:
        return float(match.group(1))
    parts = name.split("_")
    for part in reversed(parts):
        try:
            return float(part)
        except ValueError:
            continue
    return 0.0


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """
    Fills enclosed background regions (holes) in a binary mask.
    All pixels inside the outer boundary of external contours are filled with 255.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask.copy()
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def keep_largest_contour(mask: np.ndarray) -> np.ndarray:
    """
    Keeps only the largest connected contour in the binary mask,
    eliminating smaller disconnected error dots and noise blobs.
    If the largest contour has holes and they have not been filled,
    those holes are preserved.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(mask)
    largest_contour = max(contours, key=cv2.contourArea)
    largest_mask = np.zeros_like(mask)
    cv2.drawContours(largest_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    return cv2.bitwise_and(mask, largest_mask)


def clean_mask(
    mask: np.ndarray,
    do_fill_holes: bool = True,
    do_keep_largest: bool = True,
    order: str = "fill-first",
    threshold: int = 127
) -> Tuple[np.ndarray, dict]:
    """
    Post-processes a single binary mask with hole filling and/or largest contour selection.
    
    Args:
        mask: Grayscale or binary numpy array.
        do_fill_holes: Whether to fill enclosed holes.
        do_keep_largest: Whether to keep only the largest contour.
        order: "fill-first" (fill holes then keep largest) or "largest-first".
        threshold: Threshold value to ensure mask is strictly binary (0 or 255).
        
    Returns:
        (cleaned_mask, stats_dict)
    """
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
    _, binary_mask = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)
    orig_white_count = int(cv2.countNonZero(binary_mask))
    
    current = binary_mask.copy()
    had_holes = False
    had_error_dots = False
    hole_pixels_filled = 0
    noise_pixels_removed = 0
    
    def apply_fill(m: np.ndarray) -> np.ndarray:
        nonlocal had_holes, hole_pixels_filled
        before = int(cv2.countNonZero(m))
        filled = fill_holes(m)
        after = int(cv2.countNonZero(filled))
        diff = after - before
        if diff > 0:
            had_holes = True
            hole_pixels_filled += diff
        return filled

    def apply_keep_largest(m: np.ndarray) -> np.ndarray:
        nonlocal had_error_dots, noise_pixels_removed
        before = int(cv2.countNonZero(m))
        largest = keep_largest_contour(m)
        after = int(cv2.countNonZero(largest))
        diff = before - after
        if diff > 0:
            had_error_dots = True
            noise_pixels_removed += diff
        return largest

    if order == "fill-first":
        if do_fill_holes:
            current = apply_fill(current)
        if do_keep_largest:
            current = apply_keep_largest(current)
    else:  # largest-first
        if do_keep_largest:
            current = apply_keep_largest(current)
        if do_fill_holes:
            current = apply_fill(current)

    final_white_count = int(cv2.countNonZero(current))
    stats = {
        "orig_pixels": orig_white_count,
        "final_pixels": final_white_count,
        "had_holes": had_holes,
        "hole_pixels_filled": hole_pixels_filled,
        "had_error_dots": had_error_dots,
        "noise_pixels_removed": noise_pixels_removed,
        "changed": (orig_white_count != final_white_count) or had_holes or had_error_dots
    }
    return current, stats


def read_image(path: str) -> Optional[np.ndarray]:
    """Reads an image as grayscale numpy array, handling diverse formats via OpenCV/PIL."""
    try:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return img
    except Exception:
        pass
    try:
        pil_img = Image.open(path).convert('L')
        return np.array(pil_img)
    except Exception as e:
        print(f"Error reading image '{path}': {e}", file=sys.stderr)
        return None


def write_image(path: str, mask: np.ndarray) -> bool:
    """Writes binary mask to disk."""
    try:
        success = cv2.imwrite(path, mask)
        if success:
            return True
    except Exception:
        pass
    try:
        out_img = Image.fromarray(mask, mode='L')
        out_img.save(path)
        return True
    except Exception as e:
        print(f"Error saving image '{path}': {e}", file=sys.stderr)
        return False


def process_single_file(
    filepath: str,
    output_dir: Optional[str],
    do_fill_holes: bool,
    do_keep_largest: bool,
    order: str,
    threshold: int,
    dry_run: bool
) -> Tuple[str, bool, dict]:
    """Processes a single mask file and optionally writes it to output_dir."""
    mask = read_image(filepath)
    if mask is None:
        return (os.path.basename(filepath), False, {})
        
    cleaned, stats = clean_mask(
        mask,
        do_fill_holes=do_fill_holes,
        do_keep_largest=do_keep_largest,
        order=order,
        threshold=threshold
    )
    
    if dry_run or output_dir is None:
        return (os.path.basename(filepath), True, stats)
        
    out_path = os.path.join(output_dir, os.path.basename(filepath))
    saved = write_image(out_path, cleaned)
    return (os.path.basename(filepath), saved, stats)


def process_mask_directory(
    input_dir: str,
    output_dir: Optional[str] = None,
    do_fill_holes: bool = True,
    do_keep_largest: bool = True,
    order: str = "fill-first",
    threshold: int = 127,
    dry_run: bool = False,
    num_workers: Optional[int] = None
) -> dict:
    """
    Processes all masks in a directory with specified toggles.
    """
    if not os.path.isdir(input_dir):
        raise ValueError(f"Input directory does not exist: {input_dir}")
        
    files = []
    for ext in VALID_EXTENSIONS:
        files.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
        files.extend(glob.glob(os.path.join(input_dir, f"*{ext.upper()}")))
        
    files = sorted(list(set(files)), key=extract_frame_num)
    if not files:
        print(f"No mask images found in '{input_dir}' matching extensions: {VALID_EXTENSIONS}")
        return {}
        
    if not dry_run and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    print(f"\n{'=' * 60}")
    print(f"  Mask Cleaner Pipeline")
    print(f"{'=' * 60}")
    print(f"  Input Directory : {input_dir}")
    print(f"  Output Directory: {'(Dry Run - No save)' if dry_run else output_dir}")
    print(f"  Total Masks     : {len(files)}")
    print(f"  Fill Holes      : {'Enabled' if do_fill_holes else 'Disabled'}")
    print(f"  Keep Largest    : {'Enabled' if do_keep_largest else 'Disabled'}")
    if do_fill_holes and do_keep_largest:
        print(f"  Operation Order : {order} ({'1. Fill holes -> 2. Keep largest' if order == 'fill-first' else '1. Keep largest -> 2. Fill holes'})")
    print(f"  Threshold       : {threshold}")
    print(f"{'=' * 60}\n")
    
    start_time = time.time()
    total_files = len(files)
    success_count = 0
    masks_with_holes_filled = 0
    masks_with_dots_removed = 0
    total_holes_pixels = 0
    total_noise_pixels = 0
    
    workers = num_workers or min(8, max(1, os.cpu_count() or 1))
    
    def task(fp):
        return process_single_file(
            fp, output_dir, do_fill_holes, do_keep_largest, order, threshold, dry_run
        )
        
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for i, (fname, success, stats) in enumerate(executor.map(task, files), start=1):
            if success:
                success_count += 1
                if stats.get("had_holes"):
                    masks_with_holes_filled += 1
                    total_holes_pixels += stats.get("hole_pixels_filled", 0)
                if stats.get("had_error_dots"):
                    masks_with_dots_removed += 1
                    total_noise_pixels += stats.get("noise_pixels_removed", 0)
                    
            if i % 25 == 0 or i == total_files:
                pct = (i / total_files) * 100.0
                print(f"  [{i}/{total_files}] ({pct:5.1f}%) processed...", end="\r", flush=True)
                
    elapsed = time.time() - start_time
    throughput = total_files / elapsed if elapsed > 0 else 0
    
    print(f"\n\n{'=' * 60}")
    print(f"  Processing Summary")
    print(f"{'=' * 60}")
    print(f"  Total Processed     : {success_count}/{total_files} masks")
    print(f"  Time Elapsed        : {elapsed:.2f}s ({throughput:.1f} masks/sec)")
    print(f"  Masks with Holes    : {masks_with_holes_filled} (total {total_holes_pixels:,} hole pixels filled)")
    print(f"  Masks with Error Dots: {masks_with_dots_removed} (total {total_noise_pixels:,} noise pixels removed)")
    if not dry_run and output_dir:
        print(f"  Cleaned Masks Saved : {output_dir}")
    print(f"{'=' * 60}\n")
    
    return {
        "total_files": total_files,
        "success_count": success_count,
        "masks_with_holes_filled": masks_with_holes_filled,
        "masks_with_dots_removed": masks_with_dots_removed,
        "total_holes_pixels": total_holes_pixels,
        "total_noise_pixels": total_noise_pixels,
        "elapsed_seconds": elapsed
    }


def main():
    parser = argparse.ArgumentParser(
        description="Clean binary masks by filling enclosed holes and/or removing error dots (keeping largest contour).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run both operations (fill holes, then keep largest contour):
  python mask_cleaner.py /path/to/masks

  # Run only hole filling:
  python mask_cleaner.py /path/to/masks --fill-holes

  # Run only keeping largest contour (error dot removal):
  python mask_cleaner.py /path/to/masks --keep-largest

  # Specify output directory explicitly:
  python mask_cleaner.py /path/to/masks -o /path/to/cleaned_masks

  # Overwrite original masks in-place:
  python mask_cleaner.py /path/to/masks --in-place
        """
    )
    
    parser.add_argument(
        "input_dir",
        type=str,
        help="Path to folder containing binary mask images."
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Output folder to save cleaned masks. Defaults to '<input_dir>_cleaned' if not specified."
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite original masks in input_dir in-place instead of creating a new output directory."
    )
    parser.add_argument(
        "--fill-holes",
        action="store_true",
        default=None,
        help="Fill enclosed background holes inside masks."
    )
    parser.add_argument(
        "--keep-largest",
        action="store_true",
        default=None,
        help="Keep only the largest connected contour, removing stray error dots."
    )
    parser.add_argument(
        "--no-fill-holes",
        action="store_true",
        help="Explicitly disable hole filling."
    )
    parser.add_argument(
        "--no-keep-largest",
        action="store_true",
        help="Explicitly disable keeping largest contour."
    )
    parser.add_argument(
        "--order",
        choices=["fill-first", "largest-first"],
        default="fill-first",
        help="Execution order when both operations are enabled. Default: fill-first (fill holes, then keep largest)."
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=127,
        help="Binarization threshold (0-255). Values > threshold become 255. Default: 127."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze masks and display statistics without writing any files."
    )
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=None,
        help="Number of parallel worker threads. Defaults to CPU count."
    )
    
    args = parser.parse_args()
    
    # Resolve toggles
    if args.no_fill_holes:
        do_fill = False
    elif args.fill_holes is True:
        do_fill = True
    else:
        do_fill = None

    if args.no_keep_largest:
        do_keep = False
    elif args.keep_largest is True:
        do_keep = True
    else:
        do_keep = None

    # If neither toggle was explicitly specified on command line, default to BOTH
    if do_fill is None and do_keep is None:
        do_fill = True
        do_keep = True
    else:
        if do_fill is None:
            do_fill = False
        if do_keep is None:
            do_keep = False

    if not do_fill and not do_keep:
        print("Error: Both --no-fill-holes and --no-keep-largest were specified. Nothing to do.", file=sys.stderr)
        sys.exit(1)
        
    input_dir = os.path.abspath(args.input_dir)
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)
        
    if args.in_place:
        output_dir = input_dir
    elif args.output_dir:
        output_dir = os.path.abspath(args.output_dir)
    else:
        output_dir = input_dir.rstrip("/\\") + "_cleaned"
        
    process_mask_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        do_fill_holes=do_fill,
        do_keep_largest=do_keep,
        order=args.order,
        threshold=args.threshold,
        dry_run=args.dry_run,
        num_workers=args.jobs
    )


if __name__ == "__main__":
    main()
