"""
Test Script for Left-Right Symmetry Analysis
=============================================
Run analysis on specific tools with custom parameters per tool.
Useful for testing different ROI heights, thresholds, etc.
"""

import os
import glob
import re
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json

# ============================================================================
# CONFIGURATION - EDIT THIS SECTION
# ============================================================================
BASE_DIR = r"c:\Users\alrfa\OneDrive - Eotvos Lorand Tudomanyegyetem Informatikai Kar\PhD\Dataset\CCD_DATA\DATA"
MASKS_DIR = os.path.join(BASE_DIR, "masks")
OUTPUT_DIR = os.path.join(BASE_DIR, "threshold_analysis", "left_right_method", "test_results")

# Starting frame
START_FRAME = 0

# Number of frames to analyze (90 degrees)
NUM_FRAMES = 90

# Default ROI height (can be overridden per tool)
DEFAULT_ROI_HEIGHT = 200

# Outlier threshold
WHITE_RATIO_OUTLIER_THRESHOLD = 0.8

# Threshold for classification
ASYMMETRY_THRESHOLD = 0.033

# Output formats
OUTPUT_FORMATS = ['png']

# ============================================================================
# TOOLS TO TEST - CONFIGURE HERE
# ============================================================================
# Format: tool_id -> custom ROI height (or None for default)
# Comment out tools you don't want to test

TOOLS_TO_TEST = {
    'tool002': 500,   # Test with ROI 500
    'tool028': 500,   # Test with ROI 500
    # 'tool069': 200,
    # 'tool070': 200,
    # 'tool071': 200,
    # 'tool072': 200,
    # 'tool062': 200,
}

# ============================================================================
# HELPER FUNCTIONS (same as run_all_tools_analysis.py)
# ============================================================================

def get_largest_contour_mask(mask):
    """Keep only the largest contour in the mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
    largest_contour = max(contours, key=cv2.contourArea)
    cleaned_mask = np.zeros_like(mask)
    cv2.drawContours(cleaned_mask, [largest_contour], -1, 255, -1)
    return cleaned_mask

def get_mask_folder(tool_id):
    """Get the mask folder path for a tool."""
    patterns = [
        f"{tool_id}_final_masks",
        f"{tool_id}gain10paperBG_final_masks",
        f"{tool_id}gain10_final_masks"
    ]
    for pattern in patterns:
        folder = os.path.join(MASKS_DIR, pattern)
        if os.path.exists(folder):
            return folder
    return None

def get_mask_files(mask_folder):
    """Get all mask files sorted by degree."""
    files = glob.glob(os.path.join(mask_folder, "*.tiff"))
    if not files:
        files = glob.glob(os.path.join(mask_folder, "*.tif"))
    if not files:
        return []
    
    def extract_frame_num(filepath):
        basename = os.path.basename(filepath)
        name = basename.replace('.tiff', '').replace('.tif', '')
        match = re.match(r'^(\d+\.?\d*)', name)
        if match:
            return float(match.group(1))
        return 0.0
    
    return sorted(files, key=extract_frame_num)

def find_global_roi_bottom(mask_files, start_frame, num_frames, roi_height):
    """Find ROI bottom using median, skipping outliers."""
    bottom_rows = []
    end_frame = min(start_frame + num_frames, len(mask_files))
    
    for i in range(start_frame, end_frame):
        mask = cv2.imread(mask_files[i], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        
        cleaned_mask = get_largest_contour_mask(mask)
        
        # Check for outlier
        height = cleaned_mask.shape[0]
        check_area = cleaned_mask[max(0, height - 400):, :]
        check_total = check_area.shape[0] * check_area.shape[1]
        check_white = np.sum(check_area == 255)
        white_ratio = check_white / check_total if check_total > 0 else 0
        
        if white_ratio > WHITE_RATIO_OUTLIER_THRESHOLD:
            continue
        
        white_pixels = np.where(cleaned_mask == 255)
        if len(white_pixels[0]) > 0:
            bottom_rows.append(np.max(white_pixels[0]))
    
    if not bottom_rows:
        return 0
    return int(np.median(bottom_rows))

def analyze_frame(mask_path, global_roi_bottom, roi_height):
    """Analyze a single frame."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    
    mask = get_largest_contour_mask(mask)
    height, width = mask.shape
    
    roi_top = max(0, global_roi_bottom - roi_height)
    roi_bottom = global_roi_bottom + 1
    roi_mask = mask[roi_top:roi_bottom, :]
    
    # Check for outlier
    roi_area = roi_mask.shape[0] * roi_mask.shape[1]
    white_count = np.sum(roi_mask == 255)
    white_ratio = white_count / roi_area if roi_area > 0 else 0
    
    if white_ratio > WHITE_RATIO_OUTLIER_THRESHOLD:
        return None
    
    # Find center
    white_pixels = np.where(roi_mask == 255)
    if len(white_pixels[1]) == 0:
        return None
    
    left_col = np.min(white_pixels[1])
    right_col = np.max(white_pixels[1])
    center_col = (left_col + right_col) // 2
    
    # Count pixels in each half
    left_half = roi_mask[:, left_col:center_col]
    right_half = roi_mask[:, center_col + 1:right_col + 1]
    
    left_count = np.sum(left_half == 255)
    right_count = np.sum(right_half == 255)
    
    diff = abs(left_count - right_count)
    total = left_count + right_count
    ratio = diff / total if total > 0 else 0
    
    return {
        'left_count': left_count,
        'right_count': right_count,
        'difference': diff,
        'ratio': ratio,
        'center_col': center_col
    }

def analyze_tool(tool_id, roi_height):
    """Analyze a single tool and return results."""
    mask_folder = get_mask_folder(tool_id)
    if not mask_folder:
        return None, f"No mask folder found"
    
    mask_files = get_mask_files(mask_folder)
    if len(mask_files) < START_FRAME + 10:
        return None, f"Not enough frames ({len(mask_files)})"
    
    # Find ROI bottom
    global_roi_bottom = find_global_roi_bottom(mask_files, START_FRAME, NUM_FRAMES, roi_height)
    if global_roi_bottom == 0:
        return None, "Could not find ROI bottom"
    
    end_frame = min(START_FRAME + NUM_FRAMES, len(mask_files))
    
    # Analyze each frame
    frame_data = []
    for i in range(START_FRAME, end_frame):
        result = analyze_frame(mask_files[i], global_roi_bottom, roi_height)
        if result:
            frame_data.append({
                'frame': i,
                **result
            })
    
    if not frame_data:
        return None, "No valid frames"
    
    ratios = [f['ratio'] for f in frame_data]
    diffs = [f['difference'] for f in frame_data]
    
    return {
        'mean_ratio': np.mean(ratios),
        'max_ratio': np.max(ratios),
        'std_ratio': np.std(ratios),
        'mean_diff': np.mean(diffs),
        'frames_analyzed': len(frame_data),
        'global_roi_bottom': global_roi_bottom,
        'roi_height': roi_height,
        'frame_data': frame_data,
        'mask_files': mask_files
    }, None

def plot_tool_results(tool_id, stats, output_dir):
    """Generate plot for a single tool."""
    frame_data = stats['frame_data']
    frames = [f['frame'] for f in frame_data]
    left_counts = [f['left_count'] for f in frame_data]
    right_counts = [f['right_count'] for f in frame_data]
    differences = [f['difference'] for f in frame_data]
    ratios = [f['ratio'] for f in frame_data]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Left vs Right
    ax1 = axes[0, 0]
    ax1.plot(frames, left_counts, label='Left Half', color='blue', alpha=0.7)
    ax1.plot(frames, right_counts, label='Right Half', color='red', alpha=0.7)
    ax1.set_title(f'{tool_id}: White Pixel Count per Half')
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Count')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Difference
    ax2 = axes[0, 1]
    ax2.plot(frames, differences, color='purple')
    ax2.fill_between(frames, differences, color='purple', alpha=0.2)
    ax2.set_title('Absolute Difference')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Difference')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Ratio with threshold
    ax3 = axes[1, 0]
    ax3.plot(frames, ratios, color='green')
    ax3.axhline(y=ASYMMETRY_THRESHOLD, color='red', linestyle='--', label=f'Threshold ({ASYMMETRY_THRESHOLD})')
    ax3.set_title('Asymmetry Ratio per Frame')
    ax3.set_xlabel('Frame')
    ax3.set_ylabel('Ratio')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    prediction = "Damaged" if stats['mean_ratio'] > ASYMMETRY_THRESHOLD else "Good"
    stats_text = f"""
    Tool ID: {tool_id}
    ROI Height: {stats['roi_height']}
    ROI Bottom: {stats['global_roi_bottom']}
    
    Frames Analyzed: {stats['frames_analyzed']}
    
    Mean Ratio: {stats['mean_ratio']:.4f}
    Max Ratio: {stats['max_ratio']:.4f}
    Std Ratio: {stats['std_ratio']:.4f}
    Mean Diff: {stats['mean_diff']:.1f}
    
    Threshold: {ASYMMETRY_THRESHOLD}
    Prediction: {prediction}
    """
    ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    ax4.set_title('Summary')
    
    fig.suptitle(f'{tool_id} - Test Analysis (ROI={stats["roi_height"]})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    for fmt in OUTPUT_FORMATS:
        path = os.path.join(output_dir, f'{tool_id}_roi{stats["roi_height"]}_test.{fmt}')
        plt.savefig(path, format=fmt, dpi=300)
    plt.close()

def plot_sample_frames(tool_id, stats, output_dir):
    """Plot sample frames showing ROI and division."""
    mask_files = stats['mask_files']
    global_roi_bottom = stats['global_roi_bottom']
    roi_height = stats['roi_height']
    
    end_frame = min(START_FRAME + NUM_FRAMES, len(mask_files))
    actual_frames = end_frame - START_FRAME
    sample_indices = [START_FRAME + int(i * actual_frames / 5) for i in range(5)]
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 6))
    roi_top = max(0, global_roi_bottom - roi_height)
    
    for idx, frame_idx in enumerate(sample_indices):
        if frame_idx >= len(mask_files):
            continue
        
        mask = cv2.imread(mask_files[frame_idx], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        
        mask = get_largest_contour_mask(mask)
        height, width = mask.shape
        
        # Find center
        roi_mask = mask[roi_top:global_roi_bottom + 1, :]
        white_pixels = np.where(roi_mask == 255)
        
        if len(white_pixels[1]) > 0:
            left_col = np.min(white_pixels[1])
            right_col = np.max(white_pixels[1])
            center_col = (left_col + right_col) // 2
        else:
            center_col = width // 2
            left_col, right_col = 0, width - 1
        
        # Visualization
        vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        cv2.line(vis, (0, roi_top), (width, roi_top), (0, 255, 0), 2)
        cv2.line(vis, (0, global_roi_bottom), (width, global_roi_bottom), (0, 255, 0), 2)
        cv2.line(vis, (left_col, roi_top), (left_col, global_roi_bottom), (255, 255, 0), 1)
        cv2.line(vis, (right_col, roi_top), (right_col, global_roi_bottom), (255, 255, 0), 1)
        cv2.line(vis, (center_col, roi_top), (center_col, global_roi_bottom), (255, 0, 0), 2)
        
        axes[idx].imshow(vis)
        axes[idx].set_title(f'Frame {frame_idx}°')
        axes[idx].axis('off')
    
    fig.suptitle(f'{tool_id}: Sample Frames (ROI={roi_height})\nGreen=ROI, Yellow=Tool bounds, Red=Center', fontsize=12)
    plt.tight_layout()
    
    for fmt in OUTPUT_FORMATS:
        path = os.path.join(output_dir, f'{tool_id}_roi{roi_height}_samples.{fmt}')
        plt.savefig(path, format=fmt, dpi=300)
    plt.close()

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("TEST SCRIPT: LEFT-RIGHT SYMMETRY ANALYSIS")
    print("=" * 70)
    print(f"Tools to test: {list(TOOLS_TO_TEST.keys())}")
    print(f"Default ROI Height: {DEFAULT_ROI_HEIGHT}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("=" * 70)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    results = []
    
    for tool_id, custom_roi in TOOLS_TO_TEST.items():
        roi_height = custom_roi if custom_roi else DEFAULT_ROI_HEIGHT
        
        print(f"\nProcessing {tool_id} (ROI={roi_height})...", end=" ")
        
        stats, error = analyze_tool(tool_id, roi_height)
        
        if error:
            print(f"FAILED: {error}")
            continue
        
        prediction = "Damaged" if stats['mean_ratio'] > ASYMMETRY_THRESHOLD else "Good"
        print(f"Mean Ratio: {stats['mean_ratio']:.4f} -> {prediction}")
        
        # Generate plots
        plot_tool_results(tool_id, stats, OUTPUT_DIR)
        plot_sample_frames(tool_id, stats, OUTPUT_DIR)
        
        results.append({
            'Tool ID': tool_id,
            'ROI Height': roi_height,
            'Mean Ratio': round(stats['mean_ratio'], 6),
            'Max Ratio': round(stats['max_ratio'], 6),
            'Std Ratio': round(stats['std_ratio'], 6),
            'Mean Diff': round(stats['mean_diff'], 2),
            'Frames Analyzed': stats['frames_analyzed'],
            'ROI Bottom': stats['global_roi_bottom'],
            'Prediction': prediction
        })
    
    # Summary
    if results:
        results_df = pd.DataFrame(results)
        csv_path = os.path.join(OUTPUT_DIR, 'test_results.csv')
        results_df.to_csv(csv_path, index=False)
        
        print("\n" + "=" * 70)
        print("TEST RESULTS")
        print("=" * 70)
        print(results_df.to_string(index=False))
        print("=" * 70)
        print(f"\nResults saved to: {csv_path}")
    
    print("\nTest complete!")

if __name__ == "__main__":
    main()
