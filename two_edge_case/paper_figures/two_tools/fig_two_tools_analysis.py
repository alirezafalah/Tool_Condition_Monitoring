"""
Paper Figure: Two-Tool Analysis (tool115 vs tool062)
===================================================
Generates 2x2 analysis plots per tool, mirroring run_all_tools_analysis,
with the threshold line removed. Output is saved in paper_figures.

Changes vs run_all_tools_analysis:
- Only tool115 and tool062 are processed.
- No threshold line in the asymmetry ratio plot.
- Summary text removes ROI height, threshold, and prediction.
"""

import os
import glob
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DIR = r"c:\Users\alrfa\OneDrive - Eotvos Lorand Tudomanyegyetem Informatikai Kar\PhD\Dataset\CCD_DATA\DATA"
MASKS_DIR = os.path.join(BASE_DIR, "masks")
TOOLS_METADATA_PATH = os.path.join(BASE_DIR, "tools_metadata.csv")
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

TOOLS = ["tool115", "tool062"]
TOOL_LABELS = {
    "tool115": "New tool (115)",
    "tool062": "Fractured tool (062)",
}

START_FRAME = 0
NUM_FRAMES = 90
ROI_HEIGHT = 200
WHITE_RATIO_OUTLIER_THRESHOLD = 0.8
OUTPUT_FORMATS = ["png", "svg"]

# ============================================================================
# HELPERS
# ============================================================================

def load_tools_metadata():
    metadata = {}
    if os.path.exists(TOOLS_METADATA_PATH):
        with open(TOOLS_METADATA_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                metadata[row["tool_id"]] = row
    return metadata


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


def analyze_left_right_symmetry(mask_path, global_roi_bottom, roi_height):
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


def analyze_tool(tool_id, mask_files, start_frame, num_frames):
    end_frame = min(start_frame + num_frames, len(mask_files))
    if end_frame - start_frame < 10:
        return None

    global_roi_bottom = find_global_roi_bottom(mask_files, start_frame, num_frames, ROI_HEIGHT)
    if global_roi_bottom == 0:
        return None

    frame_data = []
    for i in range(start_frame, end_frame):
        result = analyze_left_right_symmetry(mask_files[i], global_roi_bottom, ROI_HEIGHT)
        if result:
            frame_data.append({
                "frame": i,
                "left_count": result["left_count"],
                "right_count": result["right_count"],
                "difference": result["difference"],
                "ratio": result["ratio"],
            })

    if not frame_data:
        return None

    ratios = [f["ratio"] for f in frame_data]
    diffs = [f["difference"] for f in frame_data]

    return {
        "frames_analyzed": len(frame_data),
        "mean_ratio": float(np.mean(ratios)),
        "max_ratio": float(np.max(ratios)),
        "std_ratio": float(np.std(ratios)),
        "mean_diff": float(np.mean(diffs)),
        "frame_data": frame_data,
    }

# ============================================================================
# PLOTTING
# ============================================================================

def plot_tool_analysis(tool_id, stats, condition, label):
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    })

    frame_data = stats["frame_data"]
    frames = [f["frame"] for f in frame_data]
    left_counts = [f["left_count"] for f in frame_data]
    right_counts = [f["right_count"] for f in frame_data]
    differences = [f["difference"] for f in frame_data]
    ratios = [f["ratio"] for f in frame_data]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax1 = axes[0, 0]
    ax1.plot(frames, left_counts, label="Left Half", color="blue", alpha=0.7, linewidth=1.5)
    ax1.plot(frames, right_counts, label="Right Half", color="red", alpha=0.7, linewidth=1.5)
    ax1.set_title(f"{label}: White Pixel Count per Half")
    ax1.set_xlabel("Frame")
    ax1.set_ylabel("White Pixel Count")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    ax2.plot(frames, differences, color="purple", linewidth=1.5)
    ax2.fill_between(frames, differences, color="purple", alpha=0.2)
    ax2.set_title("Absolute Difference (Left - Right)")
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("Pixel Count Difference")
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    ax3.plot(frames, ratios, color="green", linewidth=1.5)
    ax3.set_title("Asymmetry Ratio per Frame")
    ax3.set_xlabel("Frame")
    ax3.set_ylabel("Ratio (|L-R| / Total)")
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    ax4.axis("off")
    stats_text = (
        f"Tool ID: {tool_id}\n"
        f"Condition: {condition}\n\n"
        f"Frame Range: {frames[0]} - {frames[-1]}\n"
        f"Frames Analyzed: {stats['frames_analyzed']}\n\n"
        f"Mean Asymmetry Ratio: {stats['mean_ratio']:.4f}\n"
        f"Max Asymmetry Ratio: {stats['max_ratio']:.4f}\n"
        f"Std Asymmetry Ratio: {stats['std_ratio']:.4f}\n"
        f"Mean Pixel Difference: {stats['mean_diff']:.1f}"
    )
    ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment="center", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.5))
    ax4.set_title("Summary Statistics")

    fig.suptitle(f"{label} Left-Right Symmetry Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()

    for fmt in OUTPUT_FORMATS:
        path = os.path.join(OUTPUT_DIR, f"{tool_id}_left_right_analysis_paper.{fmt}")
        plt.savefig(path, format=fmt, dpi=300)

    plt.close(fig)

# ============================================================================
# MAIN
# ============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tools_meta = load_tools_metadata()

    for tool_id in TOOLS:
        meta = tools_meta.get(tool_id, {})
        condition = meta.get("condition", "N/A")
        label = TOOL_LABELS.get(tool_id, tool_id)

        mask_folder = get_mask_folder(tool_id)
        if not mask_folder:
            print(f"{tool_id}: No mask folder found, skipping.")
            continue

        mask_files = get_mask_files(mask_folder)
        if len(mask_files) < START_FRAME + 10:
            print(f"{tool_id}: Not enough frames ({len(mask_files)} total), skipping.")
            continue

        stats = analyze_tool(tool_id, mask_files, START_FRAME, NUM_FRAMES)
        if stats is None:
            print(f"{tool_id}: Analysis failed.")
            continue

        plot_tool_analysis(tool_id, stats, condition, label)
        print(f"Saved: {tool_id}_left_right_analysis_paper.{OUTPUT_FORMATS[0]}")


if __name__ == "__main__":
    main()
