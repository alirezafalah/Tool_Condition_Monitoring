"""
Phase-Shifted Half-Profile Comparison Analysis (0-90 vs 180-270)
=================================================================
This script analyzes tool condition by comparing one ROI half profile
(right by default) at angle theta with the same half at theta + 180°.

Notes:
1. A small internal frame correction is used for capture phase alignment.
2. All reported labels use the nominal geometric relation (theta + 180°).
3. Metric is absolute pixel difference (no asymmetry ratio).
4. No image rotation is applied.
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
import csv

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DIR = r"c:\Users\alrfa\OneDrive - Eotvos Lorand Tudomanyegyetem Informatikai Kar\PhD\Dataset\CCD_DATA\DATA"
MASKS_DIR = os.path.join(BASE_DIR, "masks")
TOOLS_METADATA_PATH = os.path.join(BASE_DIR, "tools_metadata.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "threshold_analysis", "phase_shift_method")

# Starting frame
START_FRAME = 0

# Number of frames to analyze (90 degrees)
NUM_FRAMES = 90

# Internal frame offset (capture phase correction)
PHASE_SHIFT = 182

# Displayed geometric phase shift in figures/text
DISPLAY_PHASE_SHIFT_DEG = 180

# Comparison side: 'right' (default) or 'left'
COMPARE_HALF = 'right'

# ROI height configuration
DEFAULT_ROI_HEIGHT = 200

# Special ROI heights for specific tools
SPECIAL_ROI_TOOLS = {
    'tool002': 500,
    'tool028': 500,
}

# Tools to skip
SKIP_TOOLS = ['tool016', 'tool066', 'tool069']

# Threshold for Absolute Difference
# This will need to be tuned based on the data. 
# Initial guess based on observation or will be updated after first run.
# Let's start with a value and we can refine it. 
ABSOLUTE_DIFF_THRESHOLD = 500  # Placeholder, will adjust based on results

# Outlier threshold
WHITE_RATIO_OUTLIER_THRESHOLD = 0.8

# Output formats (vector)
OUTPUT_FORMATS = ['svg', 'eps']

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_tools_metadata():
    """Load the tools metadata CSV into a dictionary keyed by tool_id."""
    metadata = {}
    if os.path.exists(TOOLS_METADATA_PATH):
        with open(TOOLS_METADATA_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                metadata[row['tool_id']] = row
    return metadata

def get_two_edge_tools(tools_metadata):
    """Get list of tool IDs that have 2 edges AND have mask folders with images."""
    two_edge_tools = []
    for tool_id, meta in tools_metadata.items():
        if tool_id in SKIP_TOOLS:
            continue
        edges = meta.get('edges', '')
        try:
            if int(edges) == 2:
                mask_folder = get_mask_folder(tool_id)
                if mask_folder is not None:
                    mask_files = get_mask_files(mask_folder)
                    # We need enough frames for the phase shift (e.g. 0 + 90 + 182 approx 272 frames)
                    if len(mask_files) > START_FRAME + NUM_FRAMES + PHASE_SHIFT:
                        two_edge_tools.append(tool_id)
        except:
            pass
    return sorted(two_edge_tools)

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
    """Get all mask files from a folder, sorted by frame number."""
    pattern = os.path.join(mask_folder, "*.tiff")
    files = glob.glob(pattern)
    if not files:
        pattern = os.path.join(mask_folder, "*.tif")
        files = glob.glob(pattern)
    if not files:
        return []
    
    def extract_frame_num(filepath):
        basename = os.path.basename(filepath)
        name = basename.replace('.tiff', '').replace('.tif', '')
        # Try finding the frame number
        import re
        match = re.search(r'(\d+)', name)
        if match:
             # If strictly numeric or ends with number
             return float(match.group(1))
        return 0.0
    
    # Sort carefully - some filenames might be trickier
    files = sorted(files, key=extract_frame_num)
    return files

def get_largest_contour_mask(mask):
    """Keep only large contours."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
    largest_contour = max(contours, key=cv2.contourArea)
    cleaned_mask = np.zeros_like(mask)
    cv2.drawContours(cleaned_mask, [largest_contour], -1, 255, -1)
    return cleaned_mask

def find_global_roi_bottom(mask_files, start_frame, num_frames, phase_shift, roi_height):
    """
    Find global ROI bottom considering both the 0-90 and 180-270 ranges.
    """
    bottom_rows = []
    
    # Ranges to check: 0 to 90 AND 182 to 272 (approx)
    ranges = [
        range(start_frame, start_frame + num_frames),
        range(start_frame + phase_shift, start_frame + phase_shift + num_frames)
    ]
    
    for r in ranges:
        for i in r:
            if i >= len(mask_files): continue
            
            mask = cv2.imread(mask_files[i], cv2.IMREAD_GRAYSCALE)
            if mask is None: continue
            
            cleaned_mask = get_largest_contour_mask(mask)
            
            # Outlier check
            height = cleaned_mask.shape[0]
            check_area = cleaned_mask[max(0, height - roi_height * 2):, :]
            check_total = check_area.size
            check_white = np.sum(check_area == 255)
            if check_total > 0 and (check_white / check_total) > WHITE_RATIO_OUTLIER_THRESHOLD:
                continue
            
            white_pixels = np.where(cleaned_mask == 255)
            if len(white_pixels[0]) > 0:
                bottom_rows.append(np.max(white_pixels[0]))
            
    if not bottom_rows:
        return 0
        
    return int(np.median(bottom_rows))

def get_half_pixel_count(mask_path, global_roi_bottom, roi_height, compare_half='right'):
    """
    Reads a mask, extracts ROI, splits at center, returns selected half count.
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    
    mask = get_largest_contour_mask(mask)
    
    # ROI
    roi_top = max(0, global_roi_bottom - roi_height)
    roi_bottom = global_roi_bottom + 1
    roi_mask = mask[roi_top:roi_bottom, :]
    
    # Outlier check
    if roi_mask.size > 0:
        ratio = np.sum(roi_mask == 255) / roi_mask.size
        if ratio > WHITE_RATIO_OUTLIER_THRESHOLD:
            return None
            
    # Center
    white_pixels = np.where(roi_mask == 255)
    if len(white_pixels[1]) == 0:
        return 0 # No tool visible?
        
    left_col = np.min(white_pixels[1])
    right_col = np.max(white_pixels[1])
    center_col = (left_col + right_col) // 2
    
    left_half = roi_mask[:, left_col:center_col]
    right_half = roi_mask[:, center_col + 1:right_col + 1]

    if compare_half.lower() == 'left':
        return np.sum(left_half == 255)
    return np.sum(right_half == 255)

def analyze_tool_phase_shift(tool_id, mask_files, start_frame, num_frames, phase_shift):
    if len(mask_files) < start_frame + phase_shift + num_frames:
        return None
        
    roi_height = SPECIAL_ROI_TOOLS.get(tool_id, DEFAULT_ROI_HEIGHT)
    global_roi_bottom = find_global_roi_bottom(mask_files, start_frame, num_frames, phase_shift, roi_height)
    
    if global_roi_bottom == 0:
        return None
        
    frame_data = []
    
    for i in range(start_frame, start_frame + num_frames):
        idx1 = i
        idx2 = i + phase_shift
        
        # Check indices
        if idx2 >= len(mask_files):
            break
            
        count1 = get_half_pixel_count(mask_files[idx1], global_roi_bottom, roi_height, COMPARE_HALF)
        count2 = get_half_pixel_count(mask_files[idx2], global_roi_bottom, roi_height, COMPARE_HALF)
        
        if count1 is None or count2 is None:
            continue
            
        abs_diff = abs(count1 - count2)
        
        frame_data.append({
            'frame_idx': i,
            'frame1': idx1,
            'frame2': idx2,
            'count1': count1,
            'count2': count2,
            'abs_diff': abs_diff
        })
        
    if not frame_data:
        return None
        
    diffs = [f['abs_diff'] for f in frame_data]
    mean_diff = np.mean(diffs)
    
    return {
        'mean_diff': mean_diff,
        'frame_data': frame_data,
        'roi_height': roi_height,
        'global_roi_bottom': global_roi_bottom
    }

def plot_tool_analysis(tool_id, stats, condition, output_dir):
    plt.rcParams.update({'font.size': 12})
    
    data = stats['frame_data']
    frames = [x['frame_idx'] - START_FRAME for x in data]
    count1 = [x['count1'] for x in data]
    count2 = [x['count2'] for x in data]
    diffs = [x['abs_diff'] for x in data]
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot 1: Signal Comparison
    ax1 = axes[0]
    half_name = COMPARE_HALF.capitalize()
    ax1.plot(frames, count1, label=f'{half_name} (0-90°)', color='blue')
    ax1.plot(frames, count2, label=f'{half_name} (180-270°)', color='green', linestyle='--')
    ax1.set_title(f'{tool_id} ({condition}): {half_name}-Half Signal Comparison')
    ax1.set_xlabel('Angle θ (deg)')
    ax1.set_ylabel('Pixel Count')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Absolute difference
    ax2 = axes[1]
    ax2.plot(frames, diffs, color='red')
    ax2.set_title(f'Absolute Difference |P(θ) - P(θ+{DISPLAY_PHASE_SHIFT_DEG}°)|')
    ax2.set_xlabel('Angle θ (deg)')
    ax2.set_ylabel('Absolute Difference')
    ax2.grid(True, alpha=0.3)
    
    # Add stats text
    stats_text = f"Mean Abs Diff: {stats['mean_diff']:.2f}"
    ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes, 
             bbox=dict(facecolor='white', alpha=0.8), verticalalignment='top')
    
    plt.tight_layout()
    for fmt in OUTPUT_FORMATS:
        plt.savefig(os.path.join(output_dir, f'{tool_id}_phase_analysis.{fmt}'))
    plt.close()

def plot_summary_bar(results_df, output_dir):
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = []
    for cond in results_df['Condition']:
        c = str(cond).lower()
        if 'fractured' in c or 'deposit' in c: colors.append('red')
        elif 'used' in c: colors.append('orange')
        elif 'new' in c: colors.append('green')
        else: colors.append('gray')
        
    ax.bar(results_df['Tool ID'], results_df['Mean Diff'], color=colors, edgecolor='black')
    ax.set_title(f'Mean Absolute Difference (0-90 vs 180-270, {COMPARE_HALF.capitalize()} Half)')
    ax.set_ylabel('Mean Absolute Difference (Pixels)')
    ax.set_xlabel('Tool ID')
    plt.xticks(rotation=45, ha='right')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='New'),
        Patch(facecolor='orange', label='Used'),
        Patch(facecolor='red', label='Fractured'),
    ]
    ax.legend(handles=legend_elements)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    for fmt in OUTPUT_FORMATS:
        plt.savefig(os.path.join(output_dir, f'summary_bar_chart.{fmt}'))
    plt.close()

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("PHASE SHIFT ANALYSIS START")
    print(f"Comparison Half: {COMPARE_HALF}")
    print(f"Display Phase Shift: +{DISPLAY_PHASE_SHIFT_DEG}°")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    tools_meta = load_tools_metadata()
    two_edge_tools = get_two_edge_tools(tools_meta)
    
    print(f"Found {len(two_edge_tools)} valid tools.")
    
    results = []
    
    for tool_id in two_edge_tools:
        print(f"Analyzing {tool_id}...", end=" ")
        mask_folder = get_mask_folder(tool_id)
        mask_files = get_mask_files(mask_folder)
        
        meta = tools_meta.get(tool_id, {})
        condition = meta.get('condition', 'Unknown')
        
        stats = analyze_tool_phase_shift(tool_id, mask_files, START_FRAME, NUM_FRAMES, PHASE_SHIFT)
        
        if stats:
            print(f"Mean Diff: {stats['mean_diff']:.2f}")
            plot_tool_analysis(tool_id, stats, condition, OUTPUT_DIR)
            results.append({
                'Tool ID': tool_id,
                'Condition': condition,
                'Mean Diff': stats['mean_diff']
            })
        else:
            print("Failed (insufficient data or error)")
            
    if not results:
        return
        
    df = pd.DataFrame(results)
    df = df.sort_values('Mean Diff')
    
    csv_path = os.path.join(OUTPUT_DIR, 'phase_shift_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")
    
    plot_summary_bar(df, OUTPUT_DIR)
    
    # Suggest threshold based on separation
    # Simple logic: midpoint between max 'Used' and min 'Fractured' if possible
    try:
        fractured = df[df['Condition'].str.contains('Fractured', case=False, na=False)]
        functional = df[~df['Condition'].str.contains('Fractured', case=False, na=False)]
        
        if not fractured.empty and not functional.empty:
            min_frac = fractured['Mean Diff'].min()
            max_func = functional['Mean Diff'].max()
            print(f"\nMax Functional Diff: {max_func:.2f}")
            print(f"Min Fractured Diff:  {min_frac:.2f}")
            
            if min_frac > max_func:
                suggested_thresh = (min_frac + max_func) / 2
                print(f"Suggested Threshold: {suggested_thresh:.2f}")
            else:
                print("Classes overlap.")
    except Exception as e:
        print(f"Could not calc threshold: {e}")

if __name__ == "__main__":
    main()
