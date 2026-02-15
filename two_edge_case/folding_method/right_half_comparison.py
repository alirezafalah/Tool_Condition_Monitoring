"""
Right Half Comparison Analysis for 2-Edge Tools
================================================
This script analyzes tool symmetry by comparing the RIGHT HALF ONLY 
from frames 0-90° with frames at ANGLE_OFFSET (e.g., 182-271°).

Instead of left vs right within the same frame, this compares:
- Right half of frame i (0-89°) vs Right half of frame i+ANGLE_OFFSET

This approach tests if the same edge appears similar at opposite rotations.
The optimal offset is found via find_optimal_offset.py script.

Workflow:
1. Load final masks for frames 0-89° and ANGLE_OFFSET to ANGLE_OFFSET+89°
2. Find the global ROI (most bottom white pixel across all frames)
3. For each frame pair (i, i+ANGLE_OFFSET), extract the RIGHT HALF only
4. Count white pixels in each right half (within ROI region)
5. Compare to detect asymmetry
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

# Tools to analyze
TOOL_IDS = ["tool012", "tool115"]

# Angle offset for comparison (found via optimal offset search)
ANGLE_OFFSET = 182

# Number of frames to analyze (90 frames from 0 and 90 frames from ANGLE_OFFSET)
NUM_FRAMES = 90

# ROI height (rows above the bottom-most white pixel)
ROI_HEIGHT = 200

# Output formats: Choose from 'png', 'svg', 'eps', 'jpg' or any combination
OUTPUT_FORMATS = ['png']

# Rotation angles to test (will run analysis for each)
ROTATION_ANGLES_TO_TEST = [0.0]

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

def find_global_roi_bottom(mask_files, num_frames, rotation_angle_deg, angle_offset):
    """Find the most bottom white pixel across frames 0-89 and angle_offset to angle_offset+89."""
    global_bottom = 0
    
    # Check frames 0-89
    for i in range(min(num_frames, len(mask_files))):
        mask = cv2.imread(mask_files[i], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        
        # Apply rotation if configured
        if rotation_angle_deg != 0.0:
            mask = rotate_image(mask, rotation_angle_deg)
        
        # Find white pixels (255)
        white_pixels = np.where(mask == 255)
        if len(white_pixels[0]) > 0:
            bottom_row = np.max(white_pixels[0])
            global_bottom = max(global_bottom, bottom_row)
    
    # Check frames at offset
    for i in range(angle_offset, min(angle_offset + num_frames, len(mask_files))):
        mask = cv2.imread(mask_files[i], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        
        # Apply rotation if configured
        if rotation_angle_deg != 0.0:
            mask = rotate_image(mask, rotation_angle_deg)
        
        # Find white pixels (255)
        white_pixels = np.where(mask == 255)
        if len(white_pixels[0]) > 0:
            bottom_row = np.max(white_pixels[0])
            global_bottom = max(global_bottom, bottom_row)
    
    return global_bottom

def extract_right_half(mask, global_roi_bottom, roi_height):
    """
    Extract the RIGHT HALF of the mask within ROI region.
    
    Returns:
        dict with right_count, center_col, left_col, right_col
    """
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
    
    # Extract RIGHT HALF only: from center_col+1 to right_col (inclusive)
    right_half = roi_mask[:, center_col + 1:right_col + 1]
    
    # Count white pixels in right half
    right_count = np.sum(right_half == 255)
    
    # Calculate half area for normalization
    half_width = right_col - center_col
    half_area = roi_height * half_width if half_width > 0 else 1
    
    return {
        'right_count': right_count,
        'center_col': center_col,
        'left_col': left_col,
        'right_col': right_col,
        'half_area': half_area
    }

def analyze_right_half_comparison(mask_files, global_roi_bottom, roi_height, num_frames, rotation_angle_deg, angle_offset):
    """
    Compare right half from frames 0-89° with frames at angle_offset.
    
    Returns DataFrame with comparison results.
    """
    results = []
    
    for i in range(num_frames):
        # Frame from 0-90 range
        frame_0_90 = mask_files[i]
        # Frame from offset range (i+angle_offset)
        frame_offset_idx = i + angle_offset
        
        if frame_offset_idx >= len(mask_files):
            print(f"Warning: Frame {frame_offset_idx} does not exist for comparison with frame {i}")
            continue
        
        frame_offset = mask_files[frame_offset_idx]
        
        # Load and process first frame (0-90 range)
        mask1 = cv2.imread(frame_0_90, cv2.IMREAD_GRAYSCALE)
        if mask1 is None:
            continue
        if rotation_angle_deg != 0.0:
            mask1 = rotate_image(mask1, rotation_angle_deg)
        
        # Load and process second frame (offset range)
        mask2 = cv2.imread(frame_offset, cv2.IMREAD_GRAYSCALE)
        if mask2 is None:
            continue
        if rotation_angle_deg != 0.0:
            mask2 = rotate_image(mask2, rotation_angle_deg)
        
        # Extract right half from both frames
        result1 = extract_right_half(mask1, global_roi_bottom, roi_height)
        result2 = extract_right_half(mask2, global_roi_bottom, roi_height)
        
        if result1 is None or result2 is None:
            continue
        
        # Compare right halves
        count_0_90 = result1['right_count']
        count_offset = result2['right_count']
        
        diff = abs(count_0_90 - count_offset)
        total = count_0_90 + count_offset
        ratio = diff / total if total > 0 else 0
        
        # Normalized difference (by average half area)
        avg_area = (result1['half_area'] + result2['half_area']) / 2
        normalized_diff = diff / avg_area if avg_area > 0 else 0
        
        results.append({
            'Frame': i,
            'Angle_deg': i,
            'Count_0_90': count_0_90,
            'Count_Offset': count_offset,
            'Difference': diff,
            'Ratio': ratio,
            'Normalized_Diff': normalized_diff
        })
    
    return pd.DataFrame(results)

def plot_analysis_results(results_df, tool_id, output_dir, rotation_angle_deg, angle_offset):
    """Plot the right half comparison analysis results."""
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10
    })
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: 0-90° vs offset pixel counts
    ax1 = axes[0, 0]
    ax1.plot(results_df['Frame'], results_df['Count_0_90'], label='Right Half (0-90°)', color='blue', alpha=0.7, linewidth=1.5)
    ax1.plot(results_df['Frame'], results_df['Count_Offset'], label=f'Right Half ({angle_offset}-{angle_offset+89}°)', color='red', alpha=0.7, linewidth=1.5)
    ax1.set_title(f'{tool_id}: White Pixel Count (Right Half Only)')
    ax1.set_xlabel('Frame (Degrees)')
    ax1.set_ylabel('White Pixel Count')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Absolute difference
    ax2 = axes[0, 1]
    ax2.plot(results_df['Frame'], results_df['Difference'], color='purple', linewidth=1.5)
    ax2.fill_between(results_df['Frame'], results_df['Difference'], color='purple', alpha=0.2)
    ax2.set_title(f'Absolute Difference (0-90° vs {angle_offset}-{angle_offset+89}°)')
    ax2.set_xlabel('Frame (Degrees)')
    ax2.set_ylabel('Pixel Count Difference')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Ratio (normalized by total)
    ax3 = axes[1, 0]
    ax3.plot(results_df['Frame'], results_df['Ratio'], color='green', linewidth=1.5)
    ax3.set_title(f'Asymmetry Ratio (|0-90 - {angle_offset}-{angle_offset+89}| / Total)')
    ax3.set_xlabel('Frame (Degrees)')
    ax3.set_ylabel('Ratio')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Normalized difference
    ax4 = axes[1, 1]
    ax4.plot(results_df['Frame'], results_df['Normalized_Diff'], color='orange', linewidth=1.5)
    ax4.set_title('Normalized Difference (by half area)')
    ax4.set_xlabel('Frame (Degrees)')
    ax4.set_ylabel('Normalized Difference')
    ax4.grid(True, alpha=0.3)
    
    # Add overall statistics
    mean_ratio = results_df['Ratio'].mean()
    max_ratio = results_df['Ratio'].max()
    rotation_text = f" (Rotation: {rotation_angle_deg}°)" if rotation_angle_deg != 0.0 else ""
    fig.suptitle(f'{tool_id} Right Half Comparison (0-90° vs {angle_offset}-{angle_offset+89}°){rotation_text}\nMean Asymmetry Ratio: {mean_ratio:.4f}, Max: {max_ratio:.4f}', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    offset_suffix = f"_offset{angle_offset}"
    rotation_suffix = f"_rot{rotation_angle_deg:.1f}" if rotation_angle_deg != 0.0 else ""
    
    for fmt in OUTPUT_FORMATS:
        output_path = os.path.join(output_dir, f'{tool_id}_right_half_comparison{offset_suffix}{rotation_suffix}.{fmt}')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    plt.close()

def save_results_csv(results_df, tool_id, output_dir, rotation_angle_deg, angle_offset):
    """Save analysis results to CSV."""
    offset_suffix = f"_offset{angle_offset}"
    rotation_suffix = f"_rot{rotation_angle_deg:.1f}" if rotation_angle_deg != 0.0 else ""
    output_path = os.path.join(output_dir, f'{tool_id}_right_half_comparison{offset_suffix}{rotation_suffix}.csv')
    results_df.to_csv(output_path, index=False)
    print(f"Saved CSV: {output_path}")

def save_metadata(tool_id, mask_folder, global_roi_bottom, num_frames, output_dir, rotation_angle_deg, angle_offset):
    """Save analysis metadata to JSON."""
    metadata = {
        'tool_id': tool_id,
        'mask_folder': mask_folder,
        'global_roi_bottom': int(global_roi_bottom),
        'roi_height': ROI_HEIGHT,
        'num_frames': num_frames,
        'rotation_angle_deg': rotation_angle_deg,
        'angle_offset': angle_offset,
        'analysis_type': 'right_half_comparison',
        'frame_range_1': '0-89 degrees',
        'frame_range_2': f'{angle_offset}-{angle_offset+89} degrees',
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    offset_suffix = f"_offset{angle_offset}"
    rotation_suffix = f"_rot{rotation_angle_deg:.1f}" if rotation_angle_deg != 0.0 else ""
    output_path = os.path.join(output_dir, f'{tool_id}_right_half_comparison{offset_suffix}{rotation_suffix}_metadata.json')
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata: {output_path}")

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("RIGHT HALF COMPARISON ANALYSIS")
    print(f"Comparing right half from 0-90° vs {ANGLE_OFFSET}-{ANGLE_OFFSET+89}°")
    print("=" * 70)
    print(f"Angle Offset: {ANGLE_OFFSET}°")
    print(f"ROI Height: {ROI_HEIGHT} pixels")
    print(f"Number of frames: {NUM_FRAMES}")
    print(f"Rotation angles to test: {ROTATION_ANGLES_TO_TEST}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Tools to analyze: {', '.join(TOOL_IDS)}")
    print("=" * 70)
    
    for ROTATION_ANGLE_DEG in ROTATION_ANGLES_TO_TEST:
        print(f"\n{'#'*70}")
        print(f"# TESTING WITH ROTATION ANGLE: {ROTATION_ANGLE_DEG}°")
        print(f"{'#'*70}\n")
        
        for tool_id in TOOL_IDS:
            print(f"\n{'='*70}")
            print(f"Analyzing {tool_id} with rotation {ROTATION_ANGLE_DEG}°...")
            print(f"{'='*70}")
            
            try:
                # Get mask files
                mask_files, mask_folder = get_mask_files(tool_id)
                print(f"Found {len(mask_files)} mask files in {mask_folder}")
                
                # Check if we have enough frames
                required_frames = ANGLE_OFFSET + NUM_FRAMES
                if len(mask_files) < required_frames:
                    print(f"ERROR: Need at least {required_frames} frames, but only {len(mask_files)} available")
                    continue
                
                # Find global ROI bottom
                print("\nFinding global ROI bottom...")
                global_roi_bottom = find_global_roi_bottom(mask_files, NUM_FRAMES, ROTATION_ANGLE_DEG, ANGLE_OFFSET)
                print(f"Global ROI bottom: {global_roi_bottom} (ROI top: {global_roi_bottom - ROI_HEIGHT})")
                
                # Analyze right half comparison
                print(f"\nAnalyzing right half comparison (0-90° vs {ANGLE_OFFSET}-{ANGLE_OFFSET+NUM_FRAMES-1}°)...")
                results_df = analyze_right_half_comparison(mask_files, global_roi_bottom, ROI_HEIGHT, NUM_FRAMES, ROTATION_ANGLE_DEG, ANGLE_OFFSET)
                
                if len(results_df) == 0:
                    print(f"ERROR: No valid results for {tool_id}")
                    continue
                
                print(f"Successfully analyzed {len(results_df)} frame pairs")
                
                # Display summary statistics
                print("\nSummary Statistics:")
                print(f"  Mean difference: {results_df['Difference'].mean():.2f} pixels")
                print(f"  Max difference: {results_df['Difference'].max():.2f} pixels")
                print(f"  Mean ratio: {results_df['Ratio'].mean():.4f}")
                print(f"  Max ratio: {results_df['Ratio'].max():.4f}")
                
                # Save results
                print("\nSaving results...")
                save_results_csv(results_df, tool_id, OUTPUT_DIR, ROTATION_ANGLE_DEG, ANGLE_OFFSET)
                save_metadata(tool_id, mask_folder, global_roi_bottom, NUM_FRAMES, OUTPUT_DIR, ROTATION_ANGLE_DEG, ANGLE_OFFSET)
                
                # Plot results
                print("\nGenerating plots...")
                plot_analysis_results(results_df, tool_id, OUTPUT_DIR, ROTATION_ANGLE_DEG, ANGLE_OFFSET)
                
                print(f"\n✓ Completed analysis for {tool_id} with rotation {ROTATION_ANGLE_DEG}°")
                
            except Exception as e:
                print(f"\nERROR analyzing {tool_id}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
