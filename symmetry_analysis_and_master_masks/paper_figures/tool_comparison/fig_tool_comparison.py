"""
Paper Figure: Tool Comparison (tool028 vs tool062)
=================================================
Creates side-by-side comparison figures for:
  1) White pixel counts per half
  2) Absolute difference (left vs right)
  3) Asymmetry ratio per frame

This script uses a fixed ROI height: ROI_HEIGHT = 200.
No threshold lines or threshold text are included.
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
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

TOOLS = [
    {"id": "tool115", "label": "New tool (115)"},
    {"id": "tool062", "label": "Fractured tool (062)"},
]

START_FRAME = 0
NUM_FRAMES = 90
ROI_HEIGHT = 200
WHITE_RATIO_OUTLIER_THRESHOLD = 0.8

OUTPUT_FORMATS = ["png", "svg"]

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


def get_largest_contour_mask(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
    largest = max(contours, key=cv2.contourArea)
    out = np.zeros_like(mask)
    cv2.drawContours(out, [largest], -1, 255, -1)
    return out


def find_global_roi_bottom(mask_files, start_frame, num_frames, roi_height):
    bottoms = []
    end_frame = min(start_frame + num_frames, len(mask_files))
    for i in range(start_frame, end_frame):
        m = cv2.imread(mask_files[i], cv2.IMREAD_GRAYSCALE)
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


def analyze_frame(mask_path, global_roi_bottom, roi_height):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None

    mask = get_largest_contour_mask(mask)
    roi_top = max(0, global_roi_bottom - roi_height)
    roi_bottom = global_roi_bottom + 1
    roi_mask = mask[roi_top:roi_bottom, :]

    roi_area = roi_mask.shape[0] * roi_mask.shape[1]
    white_pixel_count = np.sum(roi_mask == 255)
    white_ratio = white_pixel_count / roi_area if roi_area > 0 else 0
    if white_ratio > WHITE_RATIO_OUTLIER_THRESHOLD:
        return None

    white_pixels = np.where(roi_mask == 255)
    if len(white_pixels[1]) == 0:
        return None

    left_col = np.min(white_pixels[1])
    right_col = np.max(white_pixels[1])
    center_col = (left_col + right_col) // 2

    left_half = roi_mask[:, left_col:center_col]
    right_half = roi_mask[:, center_col + 1:right_col + 1]

    left_count = np.sum(left_half == 255)
    right_count = np.sum(right_half == 255)
    diff = abs(left_count - right_count)
    total = left_count + right_count
    ratio = diff / total if total > 0 else 0

    return {
        "left_count": left_count,
        "right_count": right_count,
        "difference": diff,
        "ratio": ratio,
    }


def analyze_tool(tool_cfg):
    tool_id = tool_cfg["id"]
    mask_folder = get_mask_folder(tool_id)
    if not mask_folder:
        print(f"No mask folder for {tool_id}")
        return None

    mask_files = get_mask_files(mask_folder)
    if not mask_files:
        print(f"No mask files for {tool_id}")
        return None

    roi_height = ROI_HEIGHT
    global_roi_bottom = find_global_roi_bottom(mask_files, START_FRAME, NUM_FRAMES, roi_height)
    if global_roi_bottom == 0:
        print(f"Could not determine ROI bottom for {tool_id}")
        return None

    end_frame = min(START_FRAME + NUM_FRAMES, len(mask_files))
    frame_data = []

    for i in range(START_FRAME, end_frame):
        result = analyze_frame(mask_files[i], global_roi_bottom, roi_height)
        if result is None:
            continue
        frame_data.append({
            "frame": i,
            "left_count": result["left_count"],
            "right_count": result["right_count"],
            "difference": result["difference"],
            "ratio": result["ratio"],
        })

    if not frame_data:
        print(f"No valid frames for {tool_id}")
        return None

    return {
        "tool_id": tool_id,
        "label": tool_cfg["label"],
        "roi_height": roi_height,
        "frame_data": frame_data,
    }

# ============================================================================
# PLOTTING
# ============================================================================

def plot_white_counts(left_tool, right_tool):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, tool in zip(axes, [left_tool, right_tool]):
        frames = [f["frame"] for f in tool["frame_data"]]
        left_counts = [f["left_count"] for f in tool["frame_data"]]
        right_counts = [f["right_count"] for f in tool["frame_data"]]

        ax.plot(frames, left_counts, label="Left Half", color="#4472C4", linewidth=1.5)
        ax.plot(frames, right_counts, label="Right Half", color="#CC444B", linewidth=1.5)
        ax.set_title(tool["label"])
        ax.set_xlabel("Frame")
        ax.set_ylabel("White Pixel Count")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("White Pixel Count per Half")
    fig.tight_layout()

    for fmt in OUTPUT_FORMATS:
        out_path = os.path.join(OUTPUT_DIR, f"compare_white_counts.{fmt}")
        fig.savefig(out_path, format=fmt, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_absolute_difference(left_tool, right_tool):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, tool in zip(axes, [left_tool, right_tool]):
        frames = [f["frame"] for f in tool["frame_data"]]
        diffs = [f["difference"] for f in tool["frame_data"]]

        ax.plot(frames, diffs, color="#7B3FA1", linewidth=1.5)
        ax.fill_between(frames, diffs, color="#7B3FA1", alpha=0.2)
        ax.set_title(tool["label"])
        ax.set_xlabel("Frame")
        ax.set_ylabel("Absolute Difference")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Absolute Difference (Left vs Right)")
    fig.tight_layout()

    for fmt in OUTPUT_FORMATS:
        out_path = os.path.join(OUTPUT_DIR, f"compare_absolute_difference.{fmt}")
        fig.savefig(out_path, format=fmt, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_asymmetry_ratio(left_tool, right_tool):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, tool in zip(axes, [left_tool, right_tool]):
        frames = [f["frame"] for f in tool["frame_data"]]
        ratios = [f["ratio"] for f in tool["frame_data"]]

        ax.plot(frames, ratios, color="#2E8B57", linewidth=1.5)
        ax.set_title(tool["label"])
        ax.set_xlabel("Frame")
        ax.set_ylabel("Asymmetry Ratio")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Asymmetry Ratio per Frame")
    fig.tight_layout()

    for fmt in OUTPUT_FORMATS:
        out_path = os.path.join(OUTPUT_DIR, f"compare_asymmetry_ratio.{fmt}")
        fig.savefig(out_path, format=fmt, dpi=300, bbox_inches="tight")
    plt.close(fig)

# ============================================================================
# MAIN
# ============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = []
    for tool_cfg in TOOLS:
        result = analyze_tool(tool_cfg)
        if result:
            results.append(result)

    if len(results) != 2:
        print("Could not analyze both tools. Check mask availability.")
        return

    left_tool, right_tool = results

    plot_white_counts(left_tool, right_tool)
    plot_absolute_difference(left_tool, right_tool)
    plot_asymmetry_ratio(left_tool, right_tool)

    print("Saved comparison figures to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
