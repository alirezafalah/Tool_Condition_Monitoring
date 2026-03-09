"""
Left-Right Symmetry Analysis for 2-Edge Tools
==============================================
This script analyzes tool symmetry by comparing the left and right halves
of each mask image for the first 90 degrees of rotation.

For 2-edge tools, this captures asymmetry that might be lost when only
analyzing the 1D area-vs-angle profile.

Workflow:
1. Load final masks for a tool (first 90 frames = 0-89 degrees)
2. Find the global ROI (most bottom white pixel across all frames)
3. For each frame, divide the image into left/right halves at center column
4. Count white pixels in each half (within ROI region)
5. Compare left vs right to detect asymmetry
"""

import os
import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DIR = r"c:\Users\alrfa\OneDrive - Eotvos Lorand Tudomanyegyetem Informatikai Kar\PhD\Dataset\CCD_DATA\DATA"
MASKS_DIR = os.path.join(BASE_DIR, "masks")
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))  # Same directory as script

# Tool to analyze (can be changed)
TOOL_ID = "tool070"

# Starting frame (0 = start from frame 0, 80 = start from frame 80, etc.)
START_FRAME = 0

# Number of frames to analyze (first N degrees)
NUM_FRAMES = 90

# ROI height (rows above the bottom-most white pixel)
ROI_HEIGHT = 200

# Output formats: Choose from 'png', 'svg', 'eps', 'jpg' or any combination
OUTPUT_FORMATS = ['png']

# Rotation angle in degrees (0.0 = no rotation, -1.0 = -1 degree, etc.)
ROTATION_ANGLE_DEG = -1.0

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_mask_files(tool_id):
    """Get all final mask files for a tool, sorted by frame number."""
    mask_folder = os.path.join(MASKS_DIR, f"{tool_id}gain10paperBG_final_masks")
    
    # Try alternate naming patterns
    if not os.path.exists(mask_folder):
        mask_folder = os.path.join(MASKS_DIR, f"{tool_id}gain10_final_masks")
    if not os.path.exists(mask_folder):
        mask_folder = os.path.join(MASKS_DIR, f"{tool_id}_final_masks")
    
    if not os.path.exists(mask_folder):
        raise FileNotFoundError(f"Could not find mask folder for {tool_id}. Tried patterns in {MASKS_DIR}")
    
    # Get all tiff files
    pattern = os.path.join(mask_folder, "*.tiff")
    files = glob.glob(pattern)
    
    if not files:
        pattern = os.path.join(mask_folder, "*.tif")
        files = glob.glob(pattern)
    
    if not files:
        raise FileNotFoundError(f"No mask files found in {mask_folder}")
    
    # Sort by frame number (extract number from filename)
    def extract_frame_num(filepath):
        basename = os.path.basename(filepath)
        # Extract the last number before .tiff
        parts = basename.replace('.tiff', '').replace('.tif', '').split('_')
        for part in reversed(parts):
            if part.isdigit():
                return int(part)
        return 0
    
    files = sorted(files, key=extract_frame_num)
    return files, mask_folder

def find_global_roi_bottom(mask_files, num_frames):
    """Find the most bottom white pixel across all frames (global ROI)."""
    global_bottom = 0
    
    for i, mask_path in enumerate(mask_files[:num_frames]):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        
        # Find white pixels (255)
        white_pixels = np.where(mask == 255)
        if len(white_pixels[0]) > 0:
            bottom_row = np.max(white_pixels[0])
            global_bottom = max(global_bottom, bottom_row)
    
    return global_bottom

def rotate_image(image, angle_deg):
    """Rotate image around center, preserving size."""
    if angle_deg == 0.0:
        return image
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(
        image,
        mat,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

def analyze_left_right_symmetry(mask_path, global_roi_bottom, roi_height):
    """
    Analyze a single mask image for left-right symmetry.
    
    The center column is determined by the center of the tool's white pixel contour,
    not the image center. The ROI is defined as roi_height rows above global_roi_bottom.
    
    Returns:
        dict with left_count, right_count, difference, ratio, center_col
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    
    # Apply rotation if configured
    if ROTATION_ANGLE_DEG != 0.0:
        mask = rotate_image(mask, ROTATION_ANGLE_DEG)
    
    height, width = mask.shape
    
    # Define ROI region (roi_height rows above global bottom)
    roi_top = max(0, global_roi_bottom - roi_height)
    roi_bottom = global_roi_bottom + 1  # inclusive
    
    # Extract ROI region
    roi_mask = mask[roi_top:roi_bottom, :]
    
    # Find the center column of the tool's white pixels in this frame
    white_pixels = np.where(roi_mask == 255)
    if len(white_pixels[1]) == 0:
        return None
    
    # Center column is the middle of the white pixel columns
    left_col = np.min(white_pixels[1])
    right_col = np.max(white_pixels[1])
    center_col = (left_col + right_col) // 2
    
    # Divide into left and right halves based on tool center
    # Left half: from left_col to center_col (exclusive)
    # Right half: from center_col+1 to right_col (inclusive)
    left_half = roi_mask[:, left_col:center_col]
    right_half = roi_mask[:, center_col + 1:right_col + 1]
    
    # Count white pixels in each half
    left_count = np.sum(left_half == 255)
    right_count = np.sum(right_half == 255)
    
    # Calculate difference and ratio
    diff = abs(left_count - right_count)
    total = left_count + right_count
    ratio = diff / total if total > 0 else 0
    
    # Normalized difference (by half area for scale invariance)
    half_width = center_col - left_col
    half_area = roi_height * half_width if half_width > 0 else 1
    normalized_diff = diff / half_area
    
    return {
        'left_count': left_count,
        'right_count': right_count,
        'difference': diff,
        'ratio': ratio,
        'normalized_diff': normalized_diff,
        'center_col': center_col,
        'left_col': left_col,
        'right_col': right_col
    }

def plot_analysis_results(results_df, tool_id, output_dir):
    """Plot the left-right symmetry analysis results."""
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10
    })
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Left vs Right pixel counts
    ax1 = axes[0, 0]
    ax1.plot(results_df['Frame'], results_df['Left Count'], label='Left Half', color='blue', alpha=0.7, linewidth=1.5)
    ax1.plot(results_df['Frame'], results_df['Right Count'], label='Right Half', color='red', alpha=0.7, linewidth=1.5)
    ax1.set_title(f'{tool_id}: White Pixel Count per Half')
    ax1.set_xlabel('Frame (Degrees)')
    ax1.set_ylabel('White Pixel Count')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Absolute difference
    ax2 = axes[0, 1]
    ax2.plot(results_df['Frame'], results_df['Difference'], color='purple', linewidth=1.5)
    ax2.fill_between(results_df['Frame'], results_df['Difference'], color='purple', alpha=0.2)
    ax2.set_title('Absolute Difference (Left - Right)')
    ax2.set_xlabel('Frame (Degrees)')
    ax2.set_ylabel('Pixel Count Difference')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Ratio (normalized by total)
    ax3 = axes[1, 0]
    ax3.plot(results_df['Frame'], results_df['Ratio'], color='green', linewidth=1.5)
    ax3.set_title('Asymmetry Ratio (|L-R| / Total)')
    ax3.set_xlabel('Frame (Degrees)')
    ax3.set_ylabel('Ratio')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Normalized difference
    ax4 = axes[1, 1]
    ax4.plot(results_df['Frame'], results_df['Normalized Diff'], color='orange', linewidth=1.5)
    ax4.set_title('Normalized Difference (by half area)')
    ax4.set_xlabel('Frame (Degrees)')
    ax4.set_ylabel('Normalized Difference')
    ax4.grid(True, alpha=0.3)
    
    # Add overall statistics
    mean_ratio = results_df['Ratio'].mean()
    max_ratio = results_df['Ratio'].max()
    fig.suptitle(f'{tool_id} Left-Right Symmetry Analysis\nMean Asymmetry Ratio: {mean_ratio:.4f}, Max: {max_ratio:.4f}', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    for fmt in OUTPUT_FORMATS:
        path = os.path.join(output_dir, f'{tool_id}_left_right_analysis.{fmt}')
        plt.savefig(path, format=fmt, dpi=300)
    
    plt.close()

def plot_sample_frames(mask_files, global_roi_bottom, roi_height, tool_id, output_dir, sample_frames=[0, 22, 45, 67, 89]):
    """Plot sample frames showing the left-right division within ROI."""
    n_samples = len(sample_frames)
    fig, axes = plt.subplots(1, n_samples, figsize=(4 * n_samples, 6))
    
    if n_samples == 1:
        axes = [axes]
    
    roi_top = max(0, global_roi_bottom - roi_height)
    
    for idx, frame_idx in enumerate(sample_frames):
        if frame_idx >= len(mask_files):
            continue
        
        mask = cv2.imread(mask_files[frame_idx], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        
        height, width = mask.shape
        
        # Find tool center from white pixels in ROI
        roi_mask = mask[roi_top:global_roi_bottom + 1, :]
        white_pixels = np.where(roi_mask == 255)
        
        if len(white_pixels[1]) > 0:
            left_col = np.min(white_pixels[1])
            right_col = np.max(white_pixels[1])
            center_col = (left_col + right_col) // 2
        else:
            center_col = width // 2
            left_col = 0
            right_col = width - 1
        
        # Create visualization with ROI region and center line
        vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        
        # Draw ROI region (top and bottom horizontal lines)
        cv2.line(vis, (0, roi_top), (width, roi_top), (0, 255, 0), 2)
        cv2.line(vis, (0, global_roi_bottom), (width, global_roi_bottom), (0, 255, 0), 2)
        
        # Draw tool boundary lines (vertical, at left and right edges of tool)
        cv2.line(vis, (left_col, roi_top), (left_col, global_roi_bottom), (255, 255, 0), 1)
        cv2.line(vis, (right_col, roi_top), (right_col, global_roi_bottom), (255, 255, 0), 1)
        
        # Draw center line (vertical, red, at tool center)
        cv2.line(vis, (center_col, roi_top), (center_col, global_roi_bottom), (255, 0, 0), 2)
        
        axes[idx].imshow(vis)
        axes[idx].set_title(f'Frame {frame_idx}°\nCenter: col {center_col}')
        axes[idx].axis('off')
    
    fig.suptitle(f'{tool_id}: Sample Frames\nROI (green), Tool bounds (yellow), Center (red)', fontsize=12)
    plt.tight_layout()
    
    for fmt in OUTPUT_FORMATS:
        path = os.path.join(output_dir, f'{tool_id}_sample_frames.{fmt}')
        plt.savefig(path, format=fmt, dpi=300)
    
    plt.close()

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("LEFT-RIGHT SYMMETRY ANALYSIS FOR 2-EDGE TOOLS")
    print("=" * 70)
    print(f"Tool: {TOOL_ID}")
    print(f"Frame Range: {START_FRAME} to {START_FRAME + NUM_FRAMES - 1} degrees")
    print(f"ROI Height: {ROI_HEIGHT} rows")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("=" * 70)
    
    # Get mask files
    print("\nLoading mask files...")
    try:
        mask_files, mask_folder = get_mask_files(TOOL_ID)
        print(f"Found {len(mask_files)} mask files in {mask_folder}")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return
    
    # Check if we have enough frames
    end_frame = min(START_FRAME + NUM_FRAMES, len(mask_files))
    actual_num_frames = end_frame - START_FRAME
    
    if actual_num_frames < 10:
        print(f"ERROR: Not enough frames. Need at least 10, have {actual_num_frames} in range.")
        return
    
    if actual_num_frames < NUM_FRAMES:
        print(f"WARNING: Only {actual_num_frames} frames available in range, analyzing those.")
    
    # Find global ROI bottom for the analyzed range
    print("\nFinding global ROI (most bottom white pixel)...")
    global_roi_bottom = 0
    for i in range(START_FRAME, end_frame):
        mask = cv2.imread(mask_files[i], cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            white_pixels = np.where(mask == 255)
            if len(white_pixels[0]) > 0:
                bottom_row = np.max(white_pixels[0])
                global_roi_bottom = max(global_roi_bottom, bottom_row)
    
    roi_top = max(0, global_roi_bottom - ROI_HEIGHT)
    print(f"Global ROI bottom row: {global_roi_bottom}")
    print(f"ROI region: rows {roi_top} to {global_roi_bottom} ({ROI_HEIGHT} rows)")
    
    # Analyze each frame
    print("\nAnalyzing left-right symmetry for each frame...")
    results = []
    
    for i in range(START_FRAME, end_frame):
        result = analyze_left_right_symmetry(mask_files[i], global_roi_bottom, ROI_HEIGHT)
        if result:
            results.append({
                'Frame': i,
                'Left Count': result['left_count'],
                'Right Count': result['right_count'],
                'Difference': result['difference'],
                'Ratio': result['ratio'],
                'Normalized Diff': result['normalized_diff'],
                'Center Col': result['center_col']
            })
        
        if (i - START_FRAME + 1) % 10 == 0:
            print(f"  Processed {i - START_FRAME + 1}/{actual_num_frames} frames...")
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate summary statistics
    mean_ratio = results_df['Ratio'].mean()
    max_ratio = results_df['Ratio'].max()
    std_ratio = results_df['Ratio'].std()
    mean_diff = results_df['Difference'].mean()
    
    print(f"\n--- Summary Statistics ---")
    print(f"Mean Asymmetry Ratio: {mean_ratio:.4f}")
    print(f"Max Asymmetry Ratio:  {max_ratio:.4f}")
    print(f"Std Asymmetry Ratio:  {std_ratio:.4f}")
    print(f"Mean Pixel Difference: {mean_diff:.1f}")
    
    # Save results to CSV
    csv_path = os.path.join(OUTPUT_DIR, f'{TOOL_ID}_left_right_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # Generate plots
    print("\nGenerating plots...")
    plot_analysis_results(results_df, TOOL_ID, OUTPUT_DIR)
    
    # Sample frames for visualization (spread across the analyzed range)
    sample_frames = [START_FRAME + int(i * actual_num_frames / 5) for i in range(5)]
    plot_sample_frames(mask_files, global_roi_bottom, ROI_HEIGHT, TOOL_ID, OUTPUT_DIR, sample_frames)
    print("Plots saved.")
    
    # Save metadata
    metadata = {
        'tool_id': TOOL_ID,
        'start_frame': START_FRAME,
        'end_frame': end_frame - 1,
        'num_frames_analyzed': len(results),
        'roi_height': ROI_HEIGHT,
        'global_roi_bottom': int(global_roi_bottom),
        'roi_top': int(roi_top),
        'mean_asymmetry_ratio': float(mean_ratio),
        'max_asymmetry_ratio': float(max_ratio),
        'std_asymmetry_ratio': float(std_ratio),
        'mean_pixel_difference': float(mean_diff),
        'analysis_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'mask_folder': mask_folder
    }
    
    meta_path = os.path.join(OUTPUT_DIR, f'{TOOL_ID}_analysis_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nAnalysis complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
