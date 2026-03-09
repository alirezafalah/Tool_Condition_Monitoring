import json
import os
import random
import re

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle as MplRect

from build_master_masks_all_two_edge_tools import get_boundaries, select_widest_rows, fit_lines


# -----------------------------------------------------------------------------
# USER CONFIG (edit these at the top)
# -----------------------------------------------------------------------------
TOOL_ID = "tool002"  # Set to "all" to process every folder under DATA/masks_tilted.
FRAME_INDEX = None  # None -> middle frame (or random frame if USE_RANDOM_FRAME is True).
USE_RANDOM_FRAME = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CCD_DATA_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
BASE_DATA_DIR = os.path.join(CCD_DATA_ROOT, "DATA")
MASKS_TILTED_DIR = os.path.join(BASE_DATA_DIR, "masks_tilted")

ALPHA = 0.35
MAX_METHOD2_ITERS = 12


def normalize_tool_id(name):
    match = re.search(r"(tool\d+)", name, flags=re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return name.strip().lower().replace(" ", "_")


def to_binary(gray_or_bgr):
    if gray_or_bgr is None:
        return None
    if gray_or_bgr.ndim == 3:
        gray = cv2.cvtColor(gray_or_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = gray_or_bgr
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return binary


def get_line_center(rotated_master):
    ys, left_x, right_x = get_boundaries(rotated_master)
    if ys is None:
        return None
    ys_fit, left_fit, right_fit = select_widest_rows(ys, left_x, right_x)
    _, _, line_center = fit_lines(ys_fit, left_fit, right_fit)
    return line_center


def row_span_in_roi_rows(binary_mask, top, bottom):
    min_left = None
    max_right = None
    top = max(0, int(top))
    bottom = min(binary_mask.shape[0], int(bottom))
    if bottom <= top:
        return None, None

    for y in range(top, bottom):
        cols = np.where(binary_mask[y, :] == 255)[0]
        if cols.size == 0:
            continue
        row_left = int(cols[0])
        row_right = int(cols[-1])
        if min_left is None or row_left < min_left:
            min_left = row_left
        if max_right is None or row_right > max_right:
            max_right = row_right

    return min_left, max_right


def resolve_center_x(meta, line_center, roi_top, roi_bottom, roi_left, roi_right):
    if meta.get("centerline_column_px") is not None:
        center_x = int(round(float(meta["centerline_column_px"])))
        source = "metadata:centerline_column_px"
    elif line_center is not None:
        m_c, b_c = line_center
        center_x = int(round(m_c * ((roi_top + roi_bottom) / 2.0) + b_c))
        source = "computed:line_fit_on_rotated_master"
    else:
        center_x = int(round((roi_left + roi_right) / 2.0))
        source = "fallback:roi_midpoint"

    center_x = max(roi_left, min(center_x, roi_right))
    return center_x, source


def compute_roi_method1(rotated_master, meta, line_center):
    ys = np.where(rotated_master.any(axis=1))[0]
    if len(ys) == 0:
        return None

    white_cols = np.where(rotated_master.any(axis=0))[0]
    if len(white_cols) < 2:
        return None

    bottom_y = int(ys[-1])
    roi_left = int(white_cols[0])
    roi_right = int(white_cols[-1])
    width = int(roi_right - roi_left)
    roi_height = max(1, int(round(0.45 * width)))
    roi_top = max(0, bottom_y - roi_height)
    roi_bottom = bottom_y

    center_x, center_src = resolve_center_x(meta, line_center, roi_top, roi_bottom, roi_left, roi_right)

    return {
        "method": "ROI1_global_master_span",
        "width_px": width,
        "roi_height_px": roi_height,
        "roi_top_px": roi_top,
        "roi_bottom_px": roi_bottom,
        "roi_left_px": roi_left,
        "roi_right_px": roi_right,
        "centerline_column_px": center_x,
        "centerline_source": center_src,
    }


def compute_roi_method2(rotated_master, meta, line_center):
    ys = np.where(rotated_master.any(axis=1))[0]
    if len(ys) == 0:
        return None

    white_cols = np.where(rotated_master.any(axis=0))[0]
    if len(white_cols) < 2:
        return None

    bottom_y = int(ys[-1])
    width = int(white_cols[-1] - white_cols[0])
    roi_left = int(white_cols[0])
    roi_right = int(white_cols[-1])

    for _ in range(MAX_METHOD2_ITERS):
        roi_height = max(1, int(round(0.45 * width)))
        roi_top = max(0, bottom_y - roi_height)
        roi_bottom = bottom_y

        rows_left, rows_right = row_span_in_roi_rows(rotated_master, roi_top, roi_bottom)
        if rows_left is None or rows_right is None or rows_right <= rows_left:
            break

        new_width = int(rows_right - rows_left)
        roi_left, roi_right = int(rows_left), int(rows_right)
        if new_width == width:
            break
        width = new_width

    roi_height = max(1, int(round(0.45 * width)))
    roi_top = max(0, bottom_y - roi_height)
    roi_bottom = bottom_y

    rows_left, rows_right = row_span_in_roi_rows(rotated_master, roi_top, roi_bottom)
    if rows_left is not None and rows_right is not None and rows_right > rows_left:
        roi_left, roi_right = int(rows_left), int(rows_right)
        width = int(roi_right - roi_left)

    center_x, center_src = resolve_center_x(meta, line_center, roi_top, roi_bottom, roi_left, roi_right)

    return {
        "method": "ROI2_rowwise_span_within_roi_rows",
        "width_px": width,
        "roi_height_px": roi_height,
        "roi_top_px": roi_top,
        "roi_bottom_px": roi_bottom,
        "roi_left_px": roi_left,
        "roi_right_px": roi_right,
        "centerline_column_px": center_x,
        "centerline_source": center_src,
    }


def render_roi_debug(frame_bin, roi, title, out_path):
    h, w = frame_bin.shape
    roi_top = max(0, min(int(roi["roi_top_px"]), h - 1))
    roi_bottom = max(roi_top + 1, min(int(roi["roi_bottom_px"]), h))
    roi_left = max(0, min(int(roi["roi_left_px"]), w - 1))
    roi_right = max(roi_left + 1, min(int(roi["roi_right_px"]), w - 1))
    center_x = max(roi_left, min(int(roi["centerline_column_px"]), roi_right))

    vis = cv2.cvtColor(frame_bin, cv2.COLOR_GRAY2RGB).astype(np.float64) / 255.0

    blue = np.zeros_like(vis)
    red = np.zeros_like(vis)
    blue[roi_top:roi_bottom, roi_left:center_x] = [0.0, 0.0, 1.0]
    red[roi_top:roi_bottom, center_x:roi_right] = [1.0, 0.0, 0.0]

    blue_mask = blue.sum(axis=2) > 0
    red_mask = red.sum(axis=2) > 0
    vis[blue_mask] = vis[blue_mask] * (1.0 - ALPHA) + blue[blue_mask] * ALPHA
    vis[red_mask] = vis[red_mask] * (1.0 - ALPHA) + red[red_mask] * ALPHA

    fig, ax = plt.subplots(figsize=(8, 10), dpi=300)
    ax.imshow(vis, origin="upper")

    rect = MplRect(
        (roi_left, roi_top),
        roi_right - roi_left,
        roi_bottom - roi_top,
        linewidth=2.5,
        edgecolor="lime",
        facecolor="none",
    )
    ax.add_patch(rect)
    ax.plot([center_x, center_x], [roi_top, roi_bottom], color="yellow", linewidth=2.5)

    handles = [
        MplRect((0, 0), 1, 1, facecolor="blue", alpha=0.5, label="Left ROI"),
        MplRect((0, 0), 1, 1, facecolor="red", alpha=0.5, label="Right ROI"),
        Line2D([0], [0], color="yellow", linewidth=2.5, label="Centerline"),
        MplRect((0, 0), 1, 1, edgecolor="lime", facecolor="none", linewidth=2, label="ROI Box"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True, fontsize=10)

    subtitle = (
        f"width={roi['width_px']} px, roi_h={roi['roi_height_px']} px, "
        f"center_x={roi['centerline_column_px']} ({roi['centerline_source']})"
    )
    ax.set_title(f"{title}\n{subtitle}", fontsize=12)
    ax.set_axis_off()
    fig.tight_layout(pad=0.15)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def process_tool_folder(tool_dir):
    tool_folder_name = os.path.basename(tool_dir)
    tool_id = normalize_tool_id(tool_folder_name)
    info_dir = os.path.join(tool_dir, "information")

    if not os.path.isdir(info_dir):
        print(f"[SKIP] {tool_id}: missing information folder")
        return

    master_candidates = [
        f for f in os.listdir(info_dir)
        if f.lower().endswith(".png") and "master_mask" in f.lower()
    ]
    if not master_candidates:
        print(f"[SKIP] {tool_id}: no master mask png found")
        return

    meta = {}
    meta_candidates = [f for f in os.listdir(info_dir) if f.endswith("_tilt_metadata.json")]
    if meta_candidates:
        with open(os.path.join(info_dir, meta_candidates[0]), "r", encoding="utf-8") as file:
            meta = json.load(file)

    master_path = os.path.join(info_dir, master_candidates[0])
    master = cv2.imread(master_path, cv2.IMREAD_GRAYSCALE)
    master_bin = to_binary(master)
    if master_bin is None:
        print(f"[SKIP] {tool_id}: cannot read master mask")
        return

    h, w = master_bin.shape
    rotation_angle = float(meta.get("rotation_angle_deg", 0.0))
    rot_mat = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), rotation_angle, 1.0)
    rotated_master = cv2.warpAffine(master_bin, rot_mat, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)

    frame_files = sorted(
        f for f in os.listdir(tool_dir)
        if f.lower().endswith(".png") and os.path.isfile(os.path.join(tool_dir, f))
    )
    if not frame_files:
        print(f"[SKIP] {tool_id}: no tilted png frames")
        return

    if FRAME_INDEX is not None:
        frame_idx = max(0, min(int(FRAME_INDEX), len(frame_files) - 1))
    elif USE_RANDOM_FRAME:
        frame_idx = random.randint(0, len(frame_files) - 1)
    else:
        frame_idx = len(frame_files) // 2

    frame_path = os.path.join(tool_dir, frame_files[frame_idx])
    frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
    frame_bin = to_binary(frame)
    if frame_bin is None:
        print(f"[SKIP] {tool_id}: cannot read frame {frame_files[frame_idx]}")
        return

    line_center = get_line_center(rotated_master)

    roi1 = compute_roi_method1(rotated_master, meta, line_center)
    roi2 = compute_roi_method2(rotated_master, meta, line_center)
    if roi1 is None or roi2 is None:
        print(f"[SKIP] {tool_id}: failed to compute ROI geometry")
        return

    roi1_dir = os.path.join(info_dir, "ROI1")
    roi2_dir = os.path.join(info_dir, "ROI2")
    roi1_png = os.path.join(roi1_dir, f"{tool_id}_roi1_debug.png")
    roi2_png = os.path.join(roi2_dir, f"{tool_id}_roi2_debug.png")

    title_base = f"{tool_id} | frame={frame_files[frame_idx]}"
    render_roi_debug(frame_bin, roi1, f"{title_base} | ROI1 global-span", roi1_png)
    render_roi_debug(frame_bin, roi2, f"{title_base} | ROI2 row-wise span", roi2_png)

    compare = {
        "tool_id": tool_id,
        "frame_file": frame_files[frame_idx],
        "rotation_angle_deg": rotation_angle,
        "roi1": roi1,
        "roi2": roi2,
        "delta_width_px": int(roi2["width_px"] - roi1["width_px"]),
        "delta_roi_height_px": int(roi2["roi_height_px"] - roi1["roi_height_px"]),
        "metadata_centerline_column_px": meta.get("centerline_column_px"),
    }

    with open(os.path.join(roi1_dir, f"{tool_id}_roi_compare.json"), "w", encoding="utf-8") as file:
        json.dump(compare, file, indent=2)
    with open(os.path.join(roi2_dir, f"{tool_id}_roi_compare.json"), "w", encoding="utf-8") as file:
        json.dump(compare, file, indent=2)

    print(
        f"[OK] {tool_id} | ROI1 width={roi1['width_px']} roi_h={roi1['roi_height_px']} | "
        f"ROI2 width={roi2['width_px']} roi_h={roi2['roi_height_px']} | "
        f"center_x={roi1['centerline_column_px']}"
    )
    print(f"      saved: {roi1_png}")
    print(f"      saved: {roi2_png}")


def list_tool_dirs():
    if not os.path.isdir(MASKS_TILTED_DIR):
        return []
    dirs = []
    for name in sorted(os.listdir(MASKS_TILTED_DIR)):
        full = os.path.join(MASKS_TILTED_DIR, name)
        if os.path.isdir(full) and name.lower().endswith("_final_masks"):
            dirs.append(full)
    return dirs


def main():
    tool_dirs = list_tool_dirs()
    if not tool_dirs:
        print(f"No tool folders found in: {MASKS_TILTED_DIR}")
        return

    selected = []
    requested = normalize_tool_id(TOOL_ID)
    if requested == "all":
        selected = tool_dirs
    else:
        for folder in tool_dirs:
            folder_tid = normalize_tool_id(os.path.basename(folder))
            if folder_tid == requested:
                selected.append(folder)
        if not selected:
            print(f"Tool '{TOOL_ID}' not found under {MASKS_TILTED_DIR}")
            return

    print("=" * 80)
    print("ROI DEBUG PLAYGROUND")
    print("=" * 80)
    print(f"Base data dir : {BASE_DATA_DIR}")
    print(f"Masks tilted  : {MASKS_TILTED_DIR}")
    print(f"Requested tool: {TOOL_ID}")
    print(f"Selected tools: {len(selected)}")
    print("=" * 80)

    for folder in selected:
        process_tool_folder(folder)

    print("=" * 80)
    print("Done.")
    print("Outputs are saved under each tool information folder in ROI1/ and ROI2/.")
    print("=" * 80)


if __name__ == "__main__":
    main()
