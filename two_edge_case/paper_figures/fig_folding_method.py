"""
Figure A: New Tool Symmetry (4 Angles) - Local Save
===================================================
Generates a figure showing the blue/red split for a NEW tool
at 0, 30, 60, and 90 degrees.
Saves the output image in the current directory.
"""

import os
import glob
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ============================================================================
# CONFIGURATION
# ============================================================================
# Path to your data (Keep this pointing to where your data actually lives)
BASE_DIR = r"c:\Users\alrfa\OneDrive - Eotvos Lorand Tudomanyegyetem Informatikai Kar\PhD\Dataset\CCD_DATA\DATA"
MASKS_DIR = os.path.join(BASE_DIR, "masks")

# Output directory: "." means current folder where script is running
OUTPUT_DIR = "."

TOOL_ID = "tool002"
TOOL_LABEL = "New Tool (tool002)"
TARGET_DEGREES = [0, 30, 60, 90]

# Dynamic ROI coefficient: roi_height = tool_width * ROI_WIDTH_COEFF
ROI_WIDTH_COEFF = 0.45

# Outlier filter (same as analysis)
WHITE_RATIO_OUTLIER_THRESHOLD = 0.8

# Colors (RGB)
COL_LEFT  = np.array([68, 114, 196]) / 255.0   # Steel Blue
COL_RIGHT = np.array([204, 68, 75]) / 255.0    # Muted Red

# ============================================================================
# HELPERS
# ============================================================================
def get_mask_folder(tool_id):
    patterns = [f"{tool_id}_final_masks", f"{tool_id}gain10paperBG_final_masks", f"{tool_id}gain10_final_masks"]
    for pat in patterns:
        f = os.path.join(MASKS_DIR, pat)
        if os.path.exists(f): return f
    return None

def get_file_for_degree(folder, target_degree):
    files = glob.glob(os.path.join(folder, "*.tif*"))
    best_file = None
    min_diff = 999
    
    for f in files:
        name = os.path.basename(f)
        m = re.match(r"^(\d+\.?\d*)", name)
        deg = float(m.group(1)) if m else 0
        
        diff = abs(deg - target_degree)
        if diff < min_diff:
            min_diff = diff
            best_file = f
            
    return best_file, min_diff

def get_largest_contour_mask(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return mask
    largest = max(contours, key=cv2.contourArea)
    out = np.zeros_like(mask)
    cv2.drawContours(out, [largest], -1, 255, -1)
    return out

def find_global_roi_bottom(mask_files, roi_height):
    """Find the median bottom-most white pixel across all frames (same as analysis)."""
    bottoms = []
    for fp in mask_files:
        m = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        m = get_largest_contour_mask(m)
        h = m.shape[0]
        check = m[max(0, h - roi_height * 2):, :]
        ratio = np.sum(check == 255) / max(check.size, 1)
        if ratio > WHITE_RATIO_OUTLIER_THRESHOLD:
            continue
        wp = np.where(m == 255)
        if len(wp[0]) > 0:
            bottoms.append(np.max(wp[0]))
    return int(np.median(bottoms)) if bottoms else 0

def create_split_image(mask, global_roi_bottom, roi_height):
    """Color the tool halves based on center found from white pixels within the ROI."""
    mask = get_largest_contour_mask(mask)
    y_idxs, x_idxs = np.where(mask == 255)
    if len(x_idxs) == 0: return None
    
    h, w = mask.shape
    
    # Define ROI (same logic as analysis)
    roi_top = max(0, global_roi_bottom - roi_height)
    roi_bottom = global_roi_bottom + 1
    
    # Find center from white pixels WITHIN the ROI
    roi_mask = mask[roi_top:roi_bottom, :]
    roi_white = np.where(roi_mask == 255)
    if len(roi_white[1]) == 0:
        return None
    
    left_col = np.min(roi_white[1])
    right_col = np.max(roi_white[1])
    center_x = (left_col + right_col) // 2
    
    # Full extent for cropping
    min_x = np.min(x_idxs)
    max_x = np.max(x_idxs)
    
    img = np.zeros((h, w, 3), dtype=np.float32) # Black background
    
    # Color Left Blue
    left_mask = np.zeros_like(mask)
    left_mask[:, left_col:center_x] = mask[:, left_col:center_x]
    img[left_mask == 255] = COL_LEFT
    
    # Color Right Red
    right_mask = np.zeros_like(mask)
    right_mask[:, center_x+1:right_col+1] = mask[:, center_x+1:right_col+1]
    img[right_mask == 255] = COL_RIGHT
    
    # Yellow Center Line (solid)
    img[:, center_x] = [1.0, 1.0, 0.0]
    
    # Crop
    pad = 40
    crop_y_min = max(0, np.min(y_idxs) - pad)
    crop_y_max = min(h, np.max(y_idxs) + pad)
    crop_x_min = max(0, min_x - pad)
    crop_x_max = min(w, max_x + pad)
    
    return img[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

# ============================================================================
# MAIN
# ============================================================================
def main():
    folder = get_mask_folder(TOOL_ID)
    if not folder:
        print(f"Folder not found for {TOOL_ID}")
        return

    # Compute tool width from first valid frame to derive dynamic ROI height
    sample_files = sorted(glob.glob(os.path.join(folder, "*.tif*")))
    tool_width = 0
    for sf in sample_files:
        sm = cv2.imread(sf, cv2.IMREAD_GRAYSCALE)
        if sm is None:
            continue
        sm = get_largest_contour_mask(sm)
        wp = np.where(sm == 255)
        if len(wp[1]) > 0:
            tool_width = int(np.max(wp[1])) - int(np.min(wp[1]))
            break
    roi_height = int(tool_width * ROI_WIDTH_COEFF) if tool_width > 0 else 200

    # Compute global ROI bottom from all mask files (same as analysis)
    all_mask_files = sorted(glob.glob(os.path.join(folder, "*.tif*")))
    global_roi_bottom = find_global_roi_bottom(all_mask_files, roi_height)
    if global_roi_bottom == 0:
        print(f"Could not determine ROI bottom for {TOOL_ID}")
        return

    # Create Figure: 1 Row, 4 Columns
    fig, axes = plt.subplots(1, 4, figsize=(12, 4), constrained_layout=False)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.85, bottom=0.18, wspace=0.02)
    
    for i, target in enumerate(TARGET_DEGREES):
        fpath, diff = get_file_for_degree(folder, target)
        ax = axes[i]
        
        if fpath:
            raw = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            viz = create_split_image(raw, global_roi_bottom, roi_height)
            if viz is not None:
                ax.imshow(viz)
                ax.set_title(f"{target}°", fontsize=12, fontweight='bold')
                ax.axis('off')
        
    # Shared Title and Legend
    fig.suptitle(f"{TOOL_LABEL} - Spatial Symmetry Check", fontsize=14, fontweight='bold', y=0.96)
    
    legend_elements = [
        Patch(facecolor=COL_LEFT, label='Left Hemisphere'),
        Patch(facecolor=COL_RIGHT, label='Right Hemisphere'),
        Patch(facecolor='yellow', label='Center in ROI'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
               frameon=True, fontsize=10, bbox_to_anchor=(0.5, 0.02))
    
    # Save to current directory
    out_path = f"{TOOL_ID}_symmetry_series.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {os.path.abspath(out_path)}")
    plt.show()

if __name__ == "__main__":
    main()