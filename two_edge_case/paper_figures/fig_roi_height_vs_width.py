"""
Paper Figure: ROI Height vs Tool Width
======================================
Creates one figure per tool showing:
  - the mask
  - a dashed line spanning the tool width
  - the ROI height measured from the bottom-most white pixel
  - annotation of width, ROI height, and coefficient

This is for figure generation only (does not modify analysis outputs).

Usage:
    python fig_roi_height_vs_width.py
"""

import os
import glob
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DIR = r"c:\Users\alrfa\OneDrive - Eotvos Lorand Tudomanyegyetem Informatikai Kar\PhD\Dataset\CCD_DATA\DATA"
MASKS_DIR = os.path.join(BASE_DIR, "masks")
OUTPUT_DIR = os.path.join(BASE_DIR, "threshold_analysis", "left_right_method", "paper_figures")

TOOLS = [
    {"id": "tool062", "label": "Tool062 (fractured)", "target_roi": 200},
    {"id": "tool002", "label": "Tool002 (new)", "target_roi": 500},
]

# Dynamic ROI coefficient: roi_height = tool_width * ROI_WIDTH_COEFF
ROI_WIDTH_COEFF = 0.45

# Frame selection (use closest to these angles)
TARGET_DEGREES = [0, 30, 60, 90]

# Outlier filter (same as analysis)
WHITE_RATIO_OUTLIER_THRESHOLD = 0.8

# Styling
COLOR_TOOL = (1.0, 1.0, 1.0)  # white mask
COLOR_ROI = (0.2, 0.8, 0.2)   # green
COLOR_WIDTH = (1.0, 0.8, 0.2) # yellow-orange
COLOR_CENTER = (1.0, 0.0, 0.0)

# ============================================================================
# HELPERS
# ============================================================================

def get_mask_folder(tool_id):
    patterns = [
        f"{tool_id}_final_masks",
        f"{tool_id}gain10paperBG_final_masks",
        f"{tool_id}gain10_final_masks",
    ]
    for pat in patterns:
        folder = os.path.join(MASKS_DIR, pat)
        if os.path.exists(folder):
            return folder
    return None


def get_mask_files(mask_folder):
    files = glob.glob(os.path.join(mask_folder, "*.tiff"))
    if not files:
        files = glob.glob(os.path.join(mask_folder, "*.tif"))
    if not files:
        return []

    def _key(fp):
        name = os.path.basename(fp).replace(".tiff", "").replace(".tif", "")
        m = re.match(r"^(\d+\.?\d*)", name)
        return float(m.group(1)) if m else 0.0

    return sorted(files, key=_key)


def extract_degree(filepath):
    name = os.path.basename(filepath).replace(".tiff", "").replace(".tif", "")
    m = re.match(r"^(\d+\.?\d*)", name)
    return float(m.group(1)) if m else 0.0


def get_largest_contour_mask(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
    largest = max(contours, key=cv2.contourArea)
    out = np.zeros_like(mask)
    cv2.drawContours(out, [largest], -1, 255, -1)
    return out


def find_global_roi_bottom(mask_files, roi_height):
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


def find_closest_frame(mask_files, target_degrees):
    all_deg = [(f, extract_degree(f)) for f in mask_files]
    # pick the one closest to 60 degrees for a representative frame
    target = target_degrees[2] if len(target_degrees) >= 3 else target_degrees[0]
    return min(all_deg, key=lambda x: abs(x[1] - target))


def compute_tool_width(mask):
    wp = np.where(mask == 255)
    if len(wp[1]) == 0:
        return 0, 0, 0
    min_x = int(np.min(wp[1]))
    max_x = int(np.max(wp[1]))
    return max_x - min_x, min_x, max_x


# ============================================================================
# FIGURE GENERATION
# ============================================================================

def create_width_roi_figure(tool_cfg):
    tool_id = tool_cfg["id"]
    label = tool_cfg["label"]
    target_roi = tool_cfg["target_roi"]

    mask_folder = get_mask_folder(tool_id)
    if not mask_folder:
        print(f"No mask folder for {tool_id}")
        return

    mask_files = get_mask_files(mask_folder)
    if not mask_files:
        print(f"No mask files for {tool_id}")
        return

    # Use closest-to-60° frame for a clean view
    frame_path, deg = find_closest_frame(mask_files, TARGET_DEGREES)
    raw = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
    if raw is None:
        print(f"Could not read frame for {tool_id}")
        return

    mask = get_largest_contour_mask(raw)
    height, width = mask.shape

    # Width from full tool
    tool_width, min_x, max_x = compute_tool_width(mask)
    roi_height = int(tool_width * ROI_WIDTH_COEFF)

    # ROI bottom from all frames (median of bottom-most white pixel)
    roi_bottom = find_global_roi_bottom(mask_files, target_roi)
    if roi_bottom == 0:
        roi_bottom = int(np.max(np.where(mask == 255)[0])) if np.any(mask == 255) else height - 1
    roi_top = max(0, roi_bottom - roi_height)

    # Build RGB image (white mask on black background)
    rgb = np.zeros((height, width, 3), dtype=np.float32)
    rgb[mask == 255] = COLOR_TOOL

    # Plot
    fig, ax = plt.subplots(figsize=(6, 9))
    ax.imshow(rgb, interpolation="nearest")

    # Draw dashed width line at roi_bottom
    ax.plot([min_x, max_x], [roi_bottom, roi_bottom],
            color=COLOR_WIDTH, linewidth=2, linestyle=(0, (5, 4)))

    # Draw ROI top/bottom lines
    ax.plot([min_x, max_x], [roi_top, roi_top], color=COLOR_ROI, linewidth=2)
    ax.plot([min_x, max_x], [roi_bottom, roi_bottom], color=COLOR_ROI, linewidth=2)

    # Draw vertical ROI sides
    ax.plot([min_x, min_x], [roi_top, roi_bottom], color=COLOR_ROI, linewidth=1.5)
    ax.plot([max_x, max_x], [roi_top, roi_bottom], color=COLOR_ROI, linewidth=1.5)

    # Annotations
    text = (
        f"Width = {tool_width}px\n"
        f"ROI height = {roi_height}px\n"
        f"Coeff = {ROI_WIDTH_COEFF:.2f}"
    )
    ax.text(0.03, 0.03, text, transform=ax.transAxes,
            fontsize=11, color="white",
            bbox=dict(boxstyle="round", facecolor="black", alpha=0.6))

    ax.set_title(f"{label}  |  Frame {deg:.1f}°", fontsize=14, color="white", pad=10)
    ax.axis("off")
    fig.patch.set_facecolor("black")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_name = f"{tool_id}_roi_width_dynamic.png"
    fig.savefig(os.path.join(OUTPUT_DIR, out_name), dpi=300, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"Saved: {out_name}")


def main():
    for tool_cfg in TOOLS:
        create_width_roi_figure(tool_cfg)


if __name__ == "__main__":
    main()
