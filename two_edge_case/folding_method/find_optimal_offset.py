"""
Find Optimal Angle Offset for Right Half Comparison
====================================================
This script searches for the best angle offset for comparing the right half
of frames 0-89° with another 90-frame sequence.

Since the tool is healthy (symmetric), the best offset should minimize the
difference between the two sequences, indicating they are viewing the same edge.

The script tests different offsets (e.g., 178°, 179°, 180°, 181°, 182°, etc.)
and finds which gives the most similar right-half pixel counts.

Workflow:
1. Load masks for frames 0-89° 
2. Test different offset angles (e.g., 176-184°)
3. For each offset, compare right halves and calculate similarity metrics
4. Find the offset with minimum mean difference and ratio
5. Generate detailed comparison plots and results
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

# Number of frames to analyze (90 frames = 0-89°)
NUM_FRAMES = 90

# ROI height (rows above the bottom-most white pixel)
ROI_HEIGHT = 200

# Range of offset angles to test (in degrees)
# For example, test offsets from 176° to 184° (around 180°)
OFFSET_MIN = 176
OFFSET_MAX = 186

# Output formats
OUTPUT_FORMATS = ['png']

# Rotation angle (if needed)
ROTATION_ANGLE_DEG = 0.0

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
        raise FileNotFoundError(f"Could not find mask folder for {tool_id}")
    
    # Get all tiff files
    pattern = os.path.join(mask_folder, "*.tiff")
    files = glob.glob(pattern)
    
    if not files:
        pattern = os.path.join(mask_folder, "*.tif")
        files = glob.glob(pattern)
    
    if not files:
        raise FileNotFoundError(f"No mask files found in {mask_folder}")
    
    # Sort by frame number
    def extract_frame_num(filepath):
        basename = os.path.basename(filepath)
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

def find_global_roi_bottom(mask_files, num_frames, offset_range, rotation_angle_deg):
    """Find the most bottom white pixel across all frames that will be analyzed."""
    global_bottom = 0
    
    # Check frames 0-89
    for i in range(min(num_frames, len(mask_files))):
        mask = cv2.imread(mask_files[i], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        
        if rotation_angle_deg != 0.0:
            mask = rotate_image(mask, rotation_angle_deg)
        
        white_pixels = np.where(mask == 255)
        if len(white_pixels[0]) > 0:
            bottom_row = np.max(white_pixels[0])
            global_bottom = max(global_bottom, bottom_row)
    
    # Check all possible offset frames
    for offset in range(offset_range[0], offset_range[1] + 1):
        for i in range(num_frames):
            frame_idx = i + offset
            if frame_idx >= len(mask_files):
                continue
            
            mask = cv2.imread(mask_files[frame_idx], cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            
            if rotation_angle_deg != 0.0:
                mask = rotate_image(mask, rotation_angle_deg)
            
            white_pixels = np.where(mask == 255)
            if len(white_pixels[0]) > 0:
                bottom_row = np.max(white_pixels[0])
                global_bottom = max(global_bottom, bottom_row)
    
    return global_bottom

def extract_right_half_count(mask, global_roi_bottom, roi_height):
    """Extract white pixel count from right half of ROI."""
    height, width = mask.shape
    
    # Define ROI region
    roi_top = max(0, global_roi_bottom - roi_height)
    roi_bottom = global_roi_bottom + 1
    
    # Extract ROI
    roi_mask = mask[roi_top:roi_bottom, :]
    
    # Find center column
    white_pixels = np.where(roi_mask == 255)
    if len(white_pixels[1]) == 0:
        return None
    
    left_col = np.min(white_pixels[1])
    right_col = np.max(white_pixels[1])
    center_col = (left_col + right_col) // 2
    
    # Extract right half
    right_half = roi_mask[:, center_col + 1:right_col + 1]
    right_count = np.sum(right_half == 255)
    
    return right_count

def test_offset(mask_files, offset, global_roi_bottom, roi_height, num_frames, rotation_angle_deg):
    """Test a specific offset angle and return comparison metrics."""
    differences = []
    ratios = []
    
    for i in range(num_frames):
        # First sequence frame (0-89)
        frame1_idx = i
        # Second sequence frame (offset + i)
        frame2_idx = i + offset
        
        if frame2_idx >= len(mask_files):
            continue
        
        # Load frames
        mask1 = cv2.imread(mask_files[frame1_idx], cv2.IMREAD_GRAYSCALE)
        mask2 = cv2.imread(mask_files[frame2_idx], cv2.IMREAD_GRAYSCALE)
        
        if mask1 is None or mask2 is None:
            continue
        
        # Apply rotation
        if rotation_angle_deg != 0.0:
            mask1 = rotate_image(mask1, rotation_angle_deg)
            mask2 = rotate_image(mask2, rotation_angle_deg)
        
        # Extract right half counts
        count1 = extract_right_half_count(mask1, global_roi_bottom, roi_height)
        count2 = extract_right_half_count(mask2, global_roi_bottom, roi_height)
        
        if count1 is None or count2 is None:
            continue
        
        # Calculate metrics
        diff = abs(count1 - count2)
        total = count1 + count2
        ratio = diff / total if total > 0 else 0
        
        differences.append(diff)
        ratios.append(ratio)
    
    if len(differences) == 0:
        return None
    
    return {
        'offset': offset,
        'mean_difference': np.mean(differences),
        'std_difference': np.std(differences),
        'mean_ratio': np.mean(ratios),
        'std_ratio': np.std(ratios),
        'max_difference': np.max(differences),
        'max_ratio': np.max(ratios),
        'num_valid_frames': len(differences)
    }

def find_optimal_offset(mask_files, global_roi_bottom, roi_height, num_frames, 
                        offset_min, offset_max, rotation_angle_deg):
    """Test all offsets and find the one with minimum difference."""
    results = []
    
    print(f"Testing offsets from {offset_min}° to {offset_max}°...")
    for offset in range(offset_min, offset_max + 1):
        print(f"  Testing offset {offset}°...", end=" ")
        result = test_offset(mask_files, offset, global_roi_bottom, roi_height, 
                            num_frames, rotation_angle_deg)
        
        if result is not None:
            results.append(result)
            print(f"Mean diff: {result['mean_difference']:.2f}, Mean ratio: {result['mean_ratio']:.6f}")
        else:
            print("No valid data")
    
    df = pd.DataFrame(results)
    
    # Find optimal offset (minimum mean ratio)
    optimal_idx = df['mean_ratio'].idxmin()
    optimal_offset = df.loc[optimal_idx, 'offset']
    
    return df, optimal_offset

def plot_offset_comparison(results_df, tool_id, output_dir, rotation_angle_deg):
    """Plot comparison of different offsets."""
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10
    })
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Mean difference vs offset
    ax1 = axes[0, 0]
    ax1.plot(results_df['offset'], results_df['mean_difference'], 'o-', color='blue', linewidth=2, markersize=6)
    optimal_idx = results_df['mean_ratio'].idxmin()
    optimal_offset = results_df.loc[optimal_idx, 'offset']
    ax1.axvline(optimal_offset, color='red', linestyle='--', linewidth=2, label=f'Optimal: {optimal_offset}°')
    ax1.set_title('Mean Difference vs Offset Angle')
    ax1.set_xlabel('Offset Angle (degrees)')
    ax1.set_ylabel('Mean Pixel Count Difference')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Mean ratio vs offset
    ax2 = axes[0, 1]
    ax2.plot(results_df['offset'], results_df['mean_ratio'], 'o-', color='green', linewidth=2, markersize=6)
    ax2.axvline(optimal_offset, color='red', linestyle='--', linewidth=2, label=f'Optimal: {optimal_offset}°')
    ax2.set_title('Mean Asymmetry Ratio vs Offset Angle')
    ax2.set_xlabel('Offset Angle (degrees)')
    ax2.set_ylabel('Mean Ratio')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Max difference vs offset
    ax3 = axes[1, 0]
    ax3.plot(results_df['offset'], results_df['max_difference'], 'o-', color='purple', linewidth=2, markersize=6)
    ax3.axvline(optimal_offset, color='red', linestyle='--', linewidth=2, label=f'Optimal: {optimal_offset}°')
    ax3.set_title('Max Difference vs Offset Angle')
    ax3.set_xlabel('Offset Angle (degrees)')
    ax3.set_ylabel('Max Pixel Count Difference')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Std deviation vs offset
    ax4 = axes[1, 1]
    ax4.plot(results_df['offset'], results_df['std_difference'], 'o-', color='orange', linewidth=2, markersize=6)
    ax4.axvline(optimal_offset, color='red', linestyle='--', linewidth=2, label=f'Optimal: {optimal_offset}°')
    ax4.set_title('Std Dev of Difference vs Offset Angle')
    ax4.set_xlabel('Offset Angle (degrees)')
    ax4.set_ylabel('Std Dev')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    rotation_text = f" (Rotation: {rotation_angle_deg}°)" if rotation_angle_deg != 0.0 else ""
    fig.suptitle(f'{tool_id} Optimal Offset Search{rotation_text}\nOptimal Offset: {optimal_offset}° (frames {optimal_offset}-{optimal_offset+89})',
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    rotation_suffix = f"_rot{rotation_angle_deg:.1f}" if rotation_angle_deg != 0.0 else ""
    
    for fmt in OUTPUT_FORMATS:
        output_path = os.path.join(output_dir, f'{tool_id}_optimal_offset_search{rotation_suffix}.{fmt}')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    plt.close()

def save_results(results_df, tool_id, optimal_offset, output_dir, rotation_angle_deg):
    """Save offset search results to CSV and JSON."""
    rotation_suffix = f"_rot{rotation_angle_deg:.1f}" if rotation_angle_deg != 0.0 else ""
    
    # Save CSV
    csv_path = os.path.join(output_dir, f'{tool_id}_optimal_offset_search{rotation_suffix}.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")
    
    # Save metadata
    optimal_row = results_df[results_df['offset'] == optimal_offset].iloc[0]
    metadata = {
        'tool_id': tool_id,
        'optimal_offset': int(optimal_offset),
        'optimal_frame_range': f'{optimal_offset}-{optimal_offset + NUM_FRAMES - 1}',
        'mean_difference': float(optimal_row['mean_difference']),
        'mean_ratio': float(optimal_row['mean_ratio']),
        'roi_height': ROI_HEIGHT,
        'num_frames': NUM_FRAMES,
        'rotation_angle_deg': rotation_angle_deg,
        'offset_range_tested': f'{OFFSET_MIN}-{OFFSET_MAX}',
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    json_path = os.path.join(output_dir, f'{tool_id}_optimal_offset_search{rotation_suffix}_metadata.json')
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata: {json_path}")

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("OPTIMAL OFFSET FINDER")
    print("Finding best angle offset for right half comparison")
    print("=" * 70)
    print(f"ROI Height: {ROI_HEIGHT} pixels")
    print(f"Number of frames: {NUM_FRAMES}")
    print(f"Offset range to test: {OFFSET_MIN}° to {OFFSET_MAX}°")
    print(f"Rotation angle: {ROTATION_ANGLE_DEG}°")
    print(f"Tools to analyze: {', '.join(TOOL_IDS)}")
    print("=" * 70)
    
    for tool_id in TOOL_IDS:
        print(f"\n{'='*70}")
        print(f"Analyzing {tool_id}...")
        print(f"{'='*70}")
        
        try:
            # Get mask files
            mask_files, mask_folder = get_mask_files(tool_id)
            print(f"Found {len(mask_files)} mask files")
            
            # Check if we have enough frames
            required_frames = OFFSET_MAX + NUM_FRAMES
            if len(mask_files) < required_frames:
                print(f"ERROR: Need at least {required_frames} frames, but only {len(mask_files)} available")
                continue
            
            # Find global ROI bottom
            print("\nFinding global ROI bottom...")
            global_roi_bottom = find_global_roi_bottom(
                mask_files, NUM_FRAMES, (OFFSET_MIN, OFFSET_MAX), ROTATION_ANGLE_DEG
            )
            print(f"Global ROI bottom: {global_roi_bottom}")
            
            # Find optimal offset
            print("\nSearching for optimal offset...")
            results_df, optimal_offset = find_optimal_offset(
                mask_files, global_roi_bottom, ROI_HEIGHT, NUM_FRAMES,
                OFFSET_MIN, OFFSET_MAX, ROTATION_ANGLE_DEG
            )
            
            print(f"\n{'='*70}")
            print(f"OPTIMAL OFFSET FOUND: {optimal_offset}°")
            print(f"Frame range: {optimal_offset}-{optimal_offset + NUM_FRAMES - 1}")
            optimal_row = results_df[results_df['offset'] == optimal_offset].iloc[0]
            print(f"Mean difference: {optimal_row['mean_difference']:.2f} pixels")
            print(f"Mean ratio: {optimal_row['mean_ratio']:.6f}")
            print(f"{'='*70}")
            
            # Create output folder for this tool
            rotation_suffix = f"_rot{ROTATION_ANGLE_DEG:.1f}" if ROTATION_ANGLE_DEG != 0.0 else ""
            tool_output_dir = os.path.join(OUTPUT_DIR, f"{tool_id}_optimal_offset_search{rotation_suffix}")
            os.makedirs(tool_output_dir, exist_ok=True)
            
            # Save results
            print("\nSaving results...")
            save_results(results_df, tool_id, optimal_offset, tool_output_dir, ROTATION_ANGLE_DEG)
            
            # Plot results
            print("\nGenerating plots...")
            plot_offset_comparison(results_df, tool_id, tool_output_dir, ROTATION_ANGLE_DEG)
            
            print(f"\n✓ Completed analysis for {tool_id}")
            print(f"  Results saved to: {tool_output_dir}")
            
        except Exception as e:
            print(f"\nERROR analyzing {tool_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
