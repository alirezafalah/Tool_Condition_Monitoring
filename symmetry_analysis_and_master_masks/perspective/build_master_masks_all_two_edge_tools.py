import argparse
import csv
import glob
import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_BASE_DATA_DIR = r"c:\Users\alrfa\OneDrive - Eotvos Lorand Tudomanyegyetem Informatikai Kar\PhD\Dataset\CCD_DATA\DATA"
DEFAULT_MASKS_DIR = os.path.join(DEFAULT_BASE_DATA_DIR, "masks")
DEFAULT_TOOLS_METADATA_PATH = os.path.join(DEFAULT_BASE_DATA_DIR, "tools_metadata.csv")
DEFAULT_MASKS_TILTED_ROOT = os.path.join(DEFAULT_BASE_DATA_DIR, "masks_tilted")

FIG_DPI = 300
WIDTH_PERCENTILE_FOR_FIT = 50.0
MIN_ROWS_FOR_FIT = 20


# ============================================================================
# UTILITY
# ============================================================================
def recommended_worker_count():
    cpu_total = os.cpu_count() or 4
    reserve_cores = 2
    bounded = max(1, cpu_total - reserve_cores)
    fractional = max(1, int(round(cpu_total * 0.6)))
    return max(1, min(8, bounded, fractional))


def read_tools_metadata(path):
    metadata = {}
    if not os.path.exists(path):
        return metadata
    with open(path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            metadata[row.get("tool_id", "")] = row
    return metadata


def find_tiff_files(mask_folder):
    files = glob.glob(os.path.join(mask_folder, "*.tiff"))
    files.extend(glob.glob(os.path.join(mask_folder, "*.tif")))
    if not files:
        return []

    def extract_frame_num(filepath):
        basename = os.path.basename(filepath)
        name = basename.replace(".tiff", "").replace(".tif", "")
        match = re.match(r"^(\d+\.?\d*)", name)
        if match:
            return float(match.group(1))
        parts = name.split("_")
        for part in reversed(parts):
            if part.isdigit():
                return float(part)
            try:
                return float(part)
            except ValueError:
                continue
        return 0.0

    return sorted(files, key=extract_frame_num)


def to_binary_mask(mask_img):
    if mask_img.ndim == 3:
        gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = mask_img
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return binary


def is_all_white(binary):
    return np.all(binary == 255)


def normalize_tool_id(folder_name):
    match = re.search(r"(tool\d+)", folder_name, flags=re.IGNORECASE)
    if match:
        return match.group(1).lower()
    cleaned = folder_name.strip().replace(" ", "_")
    return cleaned.lower()


# ============================================================================
# MASTER MASK (skip all-white frames)
# ============================================================================
def build_master_mask(file_list):
    first_img = cv2.imread(file_list[0], cv2.IMREAD_UNCHANGED)
    if first_img is None:
        return None, 0

    master_mask = np.zeros(first_img.shape[:2], dtype=np.uint8)
    skipped = 0
    for file_path in file_list:
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        binary = to_binary_mask(img)
        if is_all_white(binary):
            skipped += 1
            continue
        master_mask = cv2.bitwise_or(master_mask, binary)

    return master_mask, skipped


# ============================================================================
# BOUNDARY / LINE FITTING (shared by figures)
# ============================================================================
def get_boundaries(binary_mask):
    h, _ = binary_mask.shape
    ys, left_x, right_x = [], [], []
    for y in range(h):
        white = np.where(binary_mask[y, :] == 255)[0]
        if white.size > 0:
            ys.append(y)
            left_x.append(white[0])
            right_x.append(white[-1])
    if len(ys) < 2:
        return None, None, None
    return (
        np.array(ys, dtype=np.float64),
        np.array(left_x, dtype=np.float64),
        np.array(right_x, dtype=np.float64),
    )


def select_widest_rows(ys, left_x, right_x):
    widths = right_x - left_x
    threshold = np.percentile(widths, WIDTH_PERCENTILE_FOR_FIT)
    keep = widths >= threshold
    if np.sum(keep) < MIN_ROWS_FOR_FIT:
        top_idx = np.argsort(widths)[-MIN_ROWS_FOR_FIT:]
        keep = np.zeros_like(widths, dtype=bool)
        keep[top_idx] = True
    return ys[keep], left_x[keep], right_x[keep]


def fit_lines(ys, left_x, right_x):
    m_left, b_left = np.polyfit(ys, left_x, 1)
    m_right, b_right = np.polyfit(ys, right_x, 1)
    m_center = (m_left + m_right) / 2.0
    b_center = (b_left + b_right) / 2.0
    return (m_left, b_left), (m_right, b_right), (m_center, b_center)


def compute_tilt_deg(m_center):
    return float(np.degrees(np.arctan(m_center)))


# ============================================================================
# MATPLOTLIB DEBUG FIGURES
# ============================================================================
def render_tilt_angle_figure(binary_mask, ys, line_left, line_right, line_center, tilt_deg, out_path):
    h, w = binary_mask.shape
    y_plot = np.array([ys.min(), ys.max()])
    m_l, b_l = line_left
    m_r, b_r = line_right
    m_c, b_c = line_center
    x_left = m_l * y_plot + b_l
    x_right = m_r * y_plot + b_r
    x_center = m_c * y_plot + b_c
    vertical_x = w / 2.0

    fig, ax = plt.subplots(figsize=(8, 10), dpi=FIG_DPI)
    ax.imshow(binary_mask, cmap="gray", origin="upper")
    ax.plot(x_left, y_plot, color="red", linewidth=2.5, label="Outer Boundaries (Left/Right)")
    ax.plot(x_right, y_plot, color="red", linewidth=2.5)
    ax.plot(x_center, y_plot, color="green", linewidth=2.8, label="Center Bisector")
    ax.plot([vertical_x, vertical_x], [y_plot.min(), y_plot.max()],
            color="blue", linewidth=2.8, linestyle="--", label="True Vertical Reference")
    ax.text(0.03, 0.97, f"Tilt angle: {tilt_deg:.3f}\u00b0",
            transform=ax.transAxes, va="top", ha="left", fontsize=16,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=0.9))
    ax.legend(loc="lower right", frameon=True, fontsize=12)
    ax.set_title("Tilt Angle Calculation from Master Mask", fontsize=18)
    ax.set_axis_off()
    fig.tight_layout(pad=0.15)
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def render_centerline_figure(rotated_mask, ys, line_left, line_right, out_path):
    y_plot = np.array([ys.min(), ys.max()])
    m_l, b_l = line_left
    m_r, b_r = line_right
    x_left = m_l * y_plot + b_l
    x_right = m_r * y_plot + b_r
    center_x = (x_left + x_right) / 2.0

    fig, ax = plt.subplots(figsize=(8, 10), dpi=FIG_DPI)
    ax.imshow(rotated_mask, cmap="gray", origin="upper")
    ax.plot(x_left, y_plot, color="red", linewidth=2.5, label="Outer Boundaries")
    ax.plot(x_right, y_plot, color="red", linewidth=2.5)
    ax.plot(center_x, y_plot, color="magenta", linewidth=3.2, linestyle="--", label="Geometric Centerline")
    ax.legend(loc="lower right", frameon=True, fontsize=12)
    ax.set_title("Centerline Determination on Straightened Master Mask", fontsize=18)
    ax.set_axis_off()
    fig.tight_layout(pad=0.15)
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ============================================================================
# ROI FIGURE
# ============================================================================
def render_roi_figure(
    sample_frame_binary,
    master_mask,
    rotated_master,
    line_center_rotated,
    tool_id,
    tool_type,
    caption_override,
    out_path,
):
    h, w = rotated_master.shape
    ys_rm = np.where(rotated_master.any(axis=1))[0]
    if len(ys_rm) == 0:
        return

    bottom_y = int(ys_rm[-1])

    mask_white_cols = np.where(rotated_master.any(axis=0))[0]
    if len(mask_white_cols) < 2:
        return
    mask_width = int(mask_white_cols[-1] - mask_white_cols[0])
    roi_height = int(round(0.45 * mask_width))

    roi_top = max(0, bottom_y - roi_height)
    roi_bottom = bottom_y

    m_c, b_c = line_center_rotated
    center_x_at_mid = m_c * ((roi_top + roi_bottom) / 2.0) + b_c
    center_x = int(round(center_x_at_mid))

    roi_left = int(mask_white_cols[0])
    roi_right = int(mask_white_cols[-1])

    vis = cv2.cvtColor(sample_frame_binary, cv2.COLOR_GRAY2RGB).astype(np.float64) / 255.0

    blue_overlay = np.zeros_like(vis)
    blue_overlay[roi_top:roi_bottom, roi_left:center_x] = [0.0, 0.0, 1.0]
    red_overlay = np.zeros_like(vis)
    red_overlay[roi_top:roi_bottom, center_x:roi_right] = [1.0, 0.0, 0.0]

    alpha = 0.35
    mask_left = blue_overlay.sum(axis=2) > 0
    mask_right = red_overlay.sum(axis=2) > 0
    vis[mask_left] = vis[mask_left] * (1 - alpha) + blue_overlay[mask_left] * alpha
    vis[mask_right] = vis[mask_right] * (1 - alpha) + red_overlay[mask_right] * alpha

    if caption_override:
        title_text = caption_override
    else:
        ttype = (tool_type or "Tool").capitalize()
        tid_num = re.sub(r"[^0-9]", "", tool_id)
        title_text = f"{ttype} Tool ({tid_num})"

    fig, ax = plt.subplots(figsize=(8, 10), dpi=FIG_DPI)
    ax.imshow(vis, origin="upper")

    from matplotlib.patches import Rectangle as MplRect
    from matplotlib.lines import Line2D

    rect = MplRect((roi_left, roi_top), roi_right - roi_left, roi_bottom - roi_top,
                    linewidth=2.5, edgecolor="lime", facecolor="none", label="ROI Box")
    ax.add_patch(rect)

    ax.plot([center_x, center_x], [roi_top, roi_bottom],
            color="yellow", linewidth=2.5, linestyle="-", label="Centerline in ROI")

    legend_handles = [
        MplRect((0, 0), 1, 1, facecolor="blue", alpha=0.5, label="Left ROI"),
        MplRect((0, 0), 1, 1, facecolor="red", alpha=0.5, label="Right ROI"),
        Line2D([0], [0], color="yellow", linewidth=2.5, label="Centerline in ROI"),
        MplRect((0, 0), 1, 1, edgecolor="lime", facecolor="none", linewidth=2, label="ROI Box"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=True, fontsize=11)
    ax.set_title(title_text, fontsize=18)
    ax.set_axis_off()
    fig.tight_layout(pad=0.15)
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ============================================================================
# TILT CALCULATION (returns structured info + generates figures)
# ============================================================================
def calculate_tilt_and_figures(master_mask, info_dir, tool_id):
    ys, left_x, right_x = get_boundaries(master_mask)
    if ys is None:
        return {
            "tilt_angle_deg": 0.0,
            "rotation_angle_deg": 0.0,
            "status": "empty_master_mask",
        }

    ys_fit, left_fit, right_fit = select_widest_rows(ys, left_x, right_x)
    line_left, line_right, line_center = fit_lines(ys_fit, left_fit, right_fit)
    tilt_deg = compute_tilt_deg(line_center[0])
    rotation_angle = -tilt_deg

    fig1_path = os.path.join(info_dir, f"{tool_id}_angle_calculation.png")
    render_tilt_angle_figure(master_mask, ys, line_left, line_right, line_center, tilt_deg, fig1_path)

    h, w = master_mask.shape
    center = (w / 2.0, h / 2.0)
    rot_mat = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    rotated = cv2.warpAffine(master_mask, rot_mat, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)

    ys2, left2, right2 = get_boundaries(rotated)
    result = {
        "tilt_angle_deg": tilt_deg,
        "rotation_angle_deg": rotation_angle,
        "m_center": float(line_center[0]),
        "b_center": float(line_center[1]),
        "status": "ok",
    }

    if ys2 is not None:
        ys2_fit, left2_fit, right2_fit = select_widest_rows(ys2, left2, right2)
        line_left2, line_right2, line_center2 = fit_lines(ys2_fit, left2_fit, right2_fit)
        fig2_path = os.path.join(info_dir, f"{tool_id}_centerline_determination.png")
        render_centerline_figure(rotated, ys2, line_left2, line_right2, fig2_path)
        result["rotated_line_center"] = (float(line_center2[0]), float(line_center2[1]))
    else:
        result["rotated_line_center"] = None

    return result


# ============================================================================
# PER-TOOL PROCESSING
# ============================================================================
def process_selected_folder(
    mask_folder,
    tools_metadata,
    masks_tilted_root,
    sample_frame_index=None,
    caption_override=None,
):
    folder_name = os.path.basename(mask_folder.rstrip("/\\"))
    tool_id = normalize_tool_id(folder_name)

    mask_files = find_tiff_files(mask_folder)
    if not mask_files:
        return {"tool_id": tool_id, "mask_folder": mask_folder, "status": "no_tiff_files"}

    master_mask, skipped_white = build_master_mask(mask_files)
    if master_mask is None:
        return {"tool_id": tool_id, "mask_folder": mask_folder, "status": "failed_to_build_master_mask"}

    out_tool_dir = os.path.join(masks_tilted_root, f"{tool_id}_final_masks")
    out_info_dir = os.path.join(out_tool_dir, "information")
    os.makedirs(out_tool_dir, exist_ok=True)
    os.makedirs(out_info_dir, exist_ok=True)

    master_mask_path = os.path.join(out_info_dir, f"{tool_id}_MASTER_MASK.png")
    cv2.imwrite(master_mask_path, master_mask)

    tilt_info = calculate_tilt_and_figures(master_mask, out_info_dir, tool_id)

    rotation_angle = tilt_info.get("rotation_angle_deg", 0.0)
    height, width = master_mask.shape
    center = (width // 2, height // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)

    rotated_master = cv2.warpAffine(master_mask, rot_matrix, (width, height),
                                     flags=cv2.INTER_NEAREST, borderValue=0)

    written_count = 0
    for file_path in mask_files:
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        b_mask = to_binary_mask(img)
        rotated_mask = cv2.warpAffine(b_mask, rot_matrix, (width, height), flags=cv2.INTER_NEAREST)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        out_path = os.path.join(out_tool_dir, f"{base_name}.png")
        if cv2.imwrite(out_path, rotated_mask):
            written_count += 1

    # ROI figure on a sample frame
    rotated_center = tilt_info.get("rotated_line_center")
    if rotated_center is not None:
        if sample_frame_index is not None and 0 <= sample_frame_index < len(mask_files):
            frame_idx = sample_frame_index
        else:
            frame_idx = random.randint(0, len(mask_files) - 1)

        sample_img = cv2.imread(mask_files[frame_idx], cv2.IMREAD_UNCHANGED)
        if sample_img is not None:
            sample_bin = to_binary_mask(sample_img)
            sample_rotated = cv2.warpAffine(sample_bin, rot_matrix, (width, height),
                                             flags=cv2.INTER_NEAREST, borderValue=0)
            meta = tools_metadata.get(tool_id, {})
            tool_type = meta.get("type", "")
            roi_fig_path = os.path.join(out_info_dir, f"{tool_id}_roi_visualization.png")
            render_roi_figure(
                sample_rotated, master_mask, rotated_master,
                rotated_center, tool_id, tool_type, caption_override, roi_fig_path,
            )

    meta = tools_metadata.get(tool_id, {})
    per_tool_metadata = {
        "tool_id": tool_id,
        "tool_type": meta.get("type", ""),
        "condition": meta.get("condition", ""),
        "edges": meta.get("edges", ""),
        "input_mask_folder": mask_folder,
        "num_input_frames": len(mask_files),
        "num_skipped_all_white": skipped_white,
        "num_output_png_frames": written_count,
        "output_tool_folder": out_tool_dir,
        "output_information_folder": out_info_dir,
        "master_mask_path": master_mask_path,
        "tilt_angle_deg": tilt_info.get("tilt_angle_deg", 0.0),
        "rotation_angle_deg": tilt_info.get("rotation_angle_deg", 0.0),
        "status": tilt_info.get("status", "ok"),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    metadata_path = os.path.join(out_info_dir, f"{tool_id}_tilt_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(per_tool_metadata, file, indent=2)

    return {
        "tool_id": tool_id,
        "mask_folder": mask_folder,
        "num_input_frames": len(mask_files),
        "num_skipped_all_white": skipped_white,
        "num_output_png_frames": written_count,
        "rotation_angle_deg": tilt_info.get("rotation_angle_deg", 0.0),
        "status": tilt_info.get("status", "ok"),
        "metadata_path": metadata_path,
    }


# ============================================================================
# FOLDER LISTING
# ============================================================================
def list_all_mask_subfolders(masks_dir):
    if not os.path.isdir(masks_dir):
        return []
    subfolders = []
    for name in sorted(os.listdir(masks_dir)):
        folder_path = os.path.join(masks_dir, name)
        if os.path.isdir(folder_path):
            subfolders.append(folder_path)
    return subfolders


# ============================================================================
# CLI
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Tilt-calibrate selected mask folders and write rotated PNG masks to "
            "DATA/masks_tilted/{tool_id}_final_masks."
        )
    )
    parser.add_argument("--base-data-dir", default=DEFAULT_BASE_DATA_DIR,
                        help="Path to CCD_DATA/DATA directory.")
    parser.add_argument("--mask-folder", action="append", default=[],
                        help="Absolute path to one input mask folder. Repeat for multiple.")
    parser.add_argument("--max-workers", type=int, default=recommended_worker_count(),
                        help="Max parallel tool folders to process.")
    parser.add_argument("--disable-opencl", action="store_true",
                        help="Disable OpenCL acceleration attempt.")
    return parser.parse_args()


def main(
    base_data_dir=DEFAULT_BASE_DATA_DIR,
    selected_mask_folders=None,
    max_workers=None,
    enable_opencl=True,
):
    masks_dir = os.path.join(base_data_dir, "masks")
    tools_metadata_path = os.path.join(base_data_dir, "tools_metadata.csv")
    masks_tilted_root = os.path.join(base_data_dir, "masks_tilted")

    if selected_mask_folders:
        folders = [f for f in selected_mask_folders if os.path.isdir(f)]
    else:
        folders = list_all_mask_subfolders(masks_dir)

    if not folders:
        print("No valid input mask folders found. Nothing to process.")
        return 1

    workers = max(1, int(max_workers or recommended_worker_count()))

    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

    opencl_enabled = False
    if enable_opencl:
        try:
            cv2.ocl.setUseOpenCL(True)
            opencl_enabled = bool(cv2.ocl.useOpenCL())
        except Exception:
            opencl_enabled = False

    unique_by_tool_id = {}
    duplicate_folders = []
    for folder in folders:
        tool_id = normalize_tool_id(os.path.basename(folder.rstrip("/\\")))
        if tool_id in unique_by_tool_id:
            duplicate_folders.append(folder)
            continue
        unique_by_tool_id[tool_id] = folder

    folders = list(unique_by_tool_id.values())
    tools_metadata = read_tools_metadata(tools_metadata_path)

    print("=" * 80)
    print("TILTED MASK GENERATION")
    print("=" * 80)
    print(f"Base DATA dir: {base_data_dir}")
    print(f"Input masks dir: {masks_dir}")
    print(f"Output dir: {masks_tilted_root}")
    print(f"Selected tool folders: {len(folders)}")
    print(f"Max workers: {workers}")
    print(f"OpenCL active: {opencl_enabled}")
    if duplicate_folders:
        print(f"Skipped duplicate tool folders: {len(duplicate_folders)}")
    print("=" * 80)

    os.makedirs(masks_tilted_root, exist_ok=True)

    succeeded = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_folder = {
            executor.submit(process_selected_folder, folder, tools_metadata, masks_tilted_root): folder
            for folder in folders
        }
        for idx, future in enumerate(as_completed(future_to_folder), start=1):
            folder = future_to_folder[future]
            folder_name = os.path.basename(folder)
            try:
                result = future.result()
            except Exception as exc:
                failed += 1
                print(f"[{idx}/{len(folders)}] {folder_name}: FAILED | {exc}")
                continue

            status = result.get("status", "unknown")
            if status in {"ok", "insufficient_points_for_fit", "empty_master_mask"}:
                succeeded += 1
                skipped_w = result.get("num_skipped_all_white", 0)
                print(
                    f"[{idx}/{len(folders)}] {folder_name}: OK | "
                    f"frames={result.get('num_output_png_frames', 0)} | "
                    f"skipped_white={skipped_w} | "
                    f"rotation={result.get('rotation_angle_deg', 0.0):.3f} deg | "
                    f"status={status}"
                )
            else:
                failed += 1
                print(f"[{idx}/{len(folders)}] {folder_name}: FAILED | status={status}")

    print("\n" + "=" * 80)
    print("DONE")
    print(f"Succeeded: {succeeded}")
    print(f"Failed: {failed}")
    print("=" * 80)

    return 0 if succeeded > 0 else 1


if __name__ == "__main__":
    args = parse_args()
    exit_code = main(
        base_data_dir=args.base_data_dir,
        selected_mask_folders=args.mask_folder,
        max_workers=args.max_workers,
        enable_opencl=not args.disable_opencl,
    )
    raise SystemExit(exit_code)
