"""
Paper Figure: Sample Frames with Colored ROI Split
=================================================
Creates a multi-panel figure of sample frames for each tool,
coloring the ROI left/right halves and showing ROI bounds and center.

Uses a fixed ROI height: ROI_HEIGHT = 500.
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
N_SAMPLES = 5
ROI_HEIGHT = 500
WHITE_RATIO_OUTLIER_THRESHOLD = 0.8
OUTPUT_FORMATS = ["png", "svg"]

# Styling (match fig_roi_height_vs_width)
COLOR_TOOL = (1.0, 1.0, 1.0)  # white mask
COLOR_ROI = (0.2, 0.8, 0.2)   # green
COLOR_LEFT = (68/255, 114/255, 196/255)  # blue
COLOR_RIGHT = (204/255, 68/255, 75/255)  # red
COLOR_CENTER = (0.95, 0.80, 0.15)        # gold

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


def build_colored_frame(mask, roi_top, roi_bottom):
    """Return an RGB image with ROI left/right halves colored."""
    mask = get_largest_contour_mask(mask)
    h, w = mask.shape

    roi_mask = mask[roi_top:roi_bottom + 1, :]
    roi_white = np.where(roi_mask == 255)
    if len(roi_white[1]) == 0:
        return None, None, None

    left_col = int(np.min(roi_white[1]))
    right_col = int(np.max(roi_white[1]))
    center_x = (left_col + right_col) // 2

    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[mask == 255] = COLOR_TOOL

    left_mask = np.zeros_like(roi_mask)
    right_mask = np.zeros_like(roi_mask)
    left_mask[:, left_col:center_x] = roi_mask[:, left_col:center_x]
    right_mask[:, center_x + 1:right_col + 1] = roi_mask[:, center_x + 1:right_col + 1]

    rgb_roi = rgb[roi_top:roi_bottom + 1, :]
    rgb_roi[left_mask == 255] = COLOR_LEFT
    rgb_roi[right_mask == 255] = COLOR_RIGHT
    rgb[roi_top:roi_bottom + 1, :] = rgb_roi

    return rgb, left_col, right_col, center_x


# ============================================================================
# MAIN
# ============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for tool_cfg in TOOLS:
        tool_id = tool_cfg["id"]
        label = tool_cfg["label"]

        mask_folder = get_mask_folder(tool_id)
        if not mask_folder:
            print(f"No mask folder for {tool_id}")
            continue

        mask_files = get_mask_files(mask_folder)
        if not mask_files:
            print(f"No mask files for {tool_id}")
            continue

        global_roi_bottom = find_global_roi_bottom(mask_files, START_FRAME, NUM_FRAMES, ROI_HEIGHT)
        if global_roi_bottom == 0:
            print(f"Could not determine ROI bottom for {tool_id}")
            continue

        roi_top = max(0, global_roi_bottom - ROI_HEIGHT)
        roi_bottom = global_roi_bottom

        end_frame = min(START_FRAME + NUM_FRAMES, len(mask_files))
        actual_frames = end_frame - START_FRAME
        if actual_frames <= 0:
            print(f"No frames in range for {tool_id}")
            continue

        sample_indices = [START_FRAME + int(i * actual_frames / N_SAMPLES) for i in range(N_SAMPLES)]

        fig, axes = plt.subplots(1, N_SAMPLES, figsize=(4 * N_SAMPLES, 6))
        if N_SAMPLES == 1:
            axes = [axes]

        for ax, frame_idx in zip(axes, sample_indices):
            if frame_idx >= len(mask_files):
                ax.axis("off")
                continue

            mask = cv2.imread(mask_files[frame_idx], cv2.IMREAD_GRAYSCALE)
            if mask is None:
                ax.axis("off")
                continue

            rgb, left_col, right_col, center_x = build_colored_frame(mask, roi_top, roi_bottom)
            if rgb is None:
                ax.axis("off")
                continue

            ax.imshow(rgb, interpolation="nearest")

            # ROI box lines
            ax.plot([left_col, right_col], [roi_top, roi_top], color=COLOR_ROI, linewidth=2)
            ax.plot([left_col, right_col], [roi_bottom, roi_bottom], color=COLOR_ROI, linewidth=2)
            ax.plot([left_col, left_col], [roi_top, roi_bottom], color=COLOR_ROI, linewidth=1.5)
            ax.plot([right_col, right_col], [roi_top, roi_bottom], color=COLOR_ROI, linewidth=1.5)

            # Center line
            ax.plot([center_x, center_x], [roi_top, roi_bottom], color=COLOR_CENTER, linewidth=1)

            ax.set_title(f"Frame {frame_idx}°")
            ax.axis("off")

        fig.suptitle(f"{label}: Sample Frames (ROI={ROI_HEIGHT})", fontsize=12, y=0.97)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        for fmt in OUTPUT_FORMATS:
            out_path = os.path.join(OUTPUT_DIR, f"{tool_id}_sample_frames_colored.{fmt}")
            fig.savefig(out_path, format=fmt, dpi=300, bbox_inches="tight", facecolor="white")

        plt.close(fig)
        print(f"Saved: {tool_id}_sample_frames_colored.png/.svg")


if __name__ == "__main__":
    main()
