#!/usr/bin/env python3
"""
Full-Degree Centerline and Pixel Count Debug Suite for the 6 Outlier Tools.

Tools:
- tool008 (New, Endmill 4.0mm)
- tool006 (Used, Endmill 7.0mm)
- tool073 (Used, Drill 2.8mm)
- tool057 (Worn, Chamfer 6.0mm)
- tool076 (Worn, Reamer 4.0mm)
- tool092 (Fractured, Endmill 12.0mm)

Features:
1. Generates a debug plot for EVERY SINGLE FRAME/DEGREE in the rotational sequence.
2. Accurately calculates Left Pixels, Right Pixels, Total Pixels, and Centerline xc for each degree.
3. Identifies the exact angular frame pair causing the Peak DSI for tool008, tool006, and tool073.
4. Generates side-by-side Peak DSI diagnostic figures and CSV databases.
5. Uses multiprocessing for high performance.
"""

import os
import sys
import glob
import json
import re
import cv2
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "DejaVu Sans"

def extract_angle(filename, idx, total_frames):
    base = os.path.basename(filename)
    m = re.search(r"(\d+\.\d+)_degrees", base)
    if m:
        return float(m.group(1))
    m2 = re.search(r"(\d+)_degrees", base)
    if m2:
        return float(m2.group(1))
    return float(idx) * 360.0 / total_frames

def process_single_frame(args):
    (tool_id, fpath, idx, total_frames, deg, roi_top, roi_bottom, roi_height,
     mask_width, static_cx, edges, pitch, span, out_dir) = args

    fname = os.path.basename(fpath)
    img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    h_img, w_img = img.shape
    y_start = max(0, roi_top - 10)
    y_end = roi_top
    if y_end - y_start < 5:
        y_start = roi_top
        y_end = min(h_img, roi_top + 10)

    # 10-row above-ROI centerline
    mids = []
    for y in range(y_start, y_end):
        w_idx = np.where(img[y, :] == 255)[0]
        if len(w_idx) > 1:
            mids.append((w_idx[0] + w_idx[-1]) / 2.0)
    cx = float(np.mean(mids)) if mids else static_cx

    # Count pixels in ROI
    roi_img = img[roi_top:roi_bottom, :]
    r_cols = np.arange(roi_img.shape[1])[None, :] > cx
    r_cnt = int(np.count_nonzero((roi_img == 255) & r_cols))
    l_cnt = int(np.count_nonzero((roi_img == 255) & ~r_cols))
    tot = r_cnt + l_cnt
    diff = abs(r_cnt - l_cnt)
    asym_pct = (200.0 * diff / tot) if tot > 0 else 0.0

    # Determine region
    region_label = "Inter-Flute Gap"
    for k in range(edges):
        r_start = k * pitch
        r_end = r_start + span
        if r_start <= deg < r_end:
            region_label = f"Region R{k+1} [{r_start:.0f}°-{r_end:.0f}°]"
            break

    # Dynamic crop parameters for fast crisp rendering
    half_w = int(round(mask_width / 2.0))
    x_margin = int(max(half_w + 120, 480))
    y_min_crop = max(0, roi_top - 160)
    y_max_crop = min(h_img, roi_bottom + 80)
    x_min_crop = max(0, int(cx - x_margin))
    x_max_crop = min(w_img, int(cx + x_margin))

    cropped = img[y_min_crop:y_max_crop, x_min_crop:x_max_crop]

    fig, ax = plt.subplots(figsize=(7, 8), dpi=120)
    ax.imshow(cropped, cmap="gray", origin="upper", extent=[x_min_crop, x_max_crop, y_max_crop, y_min_crop])

    # Overlay markings
    ax.axhline(roi_top, color="gold", linewidth=1.8, linestyle="--", label=f"ROI Top (y={roi_top})")
    ax.axhline(roi_bottom, color="gold", linewidth=1.8, linestyle="-", label=f"ROI Bottom (y={roi_bottom})")

    # 10-row band
    rect_band = Rectangle((x_min_crop, y_start), x_max_crop - x_min_crop, roi_top - y_start,
                          facecolor="lime", alpha=0.45, edgecolor="green", linewidth=1.2, label="10-Row Calib Band")
    ax.add_patch(rect_band)
    ax.axvline(cx, color="lime", linewidth=2.2, label=f"Centerline xc={cx:.1f}px")

    # Right-half ROI
    rect_roi = Rectangle((cx, roi_top), x_max_crop - cx, roi_height,
                         facecolor="cyan", alpha=0.3, edgecolor="cyan", linewidth=1.5, label="Right Half ROI")
    ax.add_patch(rect_roi)

    ax.set_xlim(x_min_crop, x_max_crop)
    ax.set_ylim(y_max_crop, y_min_crop)

    ax.set_title(f"{tool_id} — {fname} [{deg:05.2f}°] (Frame {idx+1}/{total_frames})\n"
                 f"{region_label} | xc = {cx:.2f} px\n"
                 f"Right = {r_cnt:,} px  |  Left = {l_cnt:,} px  |  Total = {tot:,} px\n"
                 f"|R - L| = {diff:,} px  (Single-Frame Asymmetry = {asym_pct:.2f}%)",
                 fontsize=10.5, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8.0, frameon=True)

    out_png = os.path.join(out_dir, f"{tool_id}_deg_{deg:06.2f}_frame_{idx:03d}.png")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    return {
        "tool_id": tool_id,
        "frame_idx": idx,
        "frame_name": fname,
        "angle_degrees": round(deg, 3),
        "region_label": region_label,
        "centerline_xc": round(cx, 2),
        "left_pixels": l_cnt,
        "right_pixels": r_cnt,
        "total_pixels": tot,
        "abs_diff_left_right": diff,
        "asymmetry_percent": round(asym_pct, 4),
        "image_file": os.path.basename(out_png)
    }

def process_tool(tool_id, base_dir, out_parent):
    tool_dir = os.path.join(out_parent, tool_id)
    os.makedirs(tool_dir, exist_ok=True)

    # 1. Read metadata
    meta_json = os.path.join(base_dir, "pixel_comparison_per_tool", f"{tool_id}_symmetry_metadata.json")
    if not os.path.exists(meta_json):
        print(f"Error: {meta_json} not found")
        return None

    with open(meta_json, "r") as fp:
        meta = json.load(fp)

    roi_meta = meta["dynamic_roi"]
    roi_top = roi_meta["roi_top_px"]
    roi_bottom = roi_meta["roi_bottom_px"]
    roi_height = roi_meta["roi_height_px"]
    mask_width = roi_meta["mask_width_px"]
    static_cx = float(meta["tilt_calibration"].get("static_centerline_intercept", 1300.0))

    t_info = meta["tool_info"]
    edges = int(t_info["edges"])
    pitch = 360.0 / edges
    span = 180.0 / edges

    # 2. Get frames
    raw_files = sorted(glob.glob(os.path.join(base_dir, "masks_tilted", tool_id, "*.png")))
    total_frames = len(raw_files)
    file_angles = [extract_angle(f, i, total_frames) for i, f in enumerate(raw_files)]
    sorted_pairs = sorted(zip(file_angles, raw_files), key=lambda x: x[0])
    angles = [p[0] for p in sorted_pairs]
    frame_files = [p[1] for p in sorted_pairs]

    print(f"\nProcessing {tool_id}: {total_frames} frames (edges={edges}, ROI top={roi_top}, bot={roi_bottom})")

    tasks = []
    for idx, (deg, fpath) in enumerate(zip(angles, frame_files)):
        tasks.append((
            tool_id, fpath, idx, total_frames, deg, roi_top, roi_bottom, roi_height,
            mask_width, static_cx, edges, pitch, span, tool_dir
        ))

    # Parallel execution
    workers = min(10, cpu_count())
    with Pool(workers) as pool:
        records = pool.map(process_single_frame, tasks)

    records = [r for r in records if r is not None]
    df_counts = pd.DataFrame(records)
    csv_path = os.path.join(tool_dir, f"{tool_id}_all_degrees_pixel_counts.csv")
    df_counts.to_csv(csv_path, index=False)
    print(f"  -> Generated {len(df_counts)} frame figures and CSV for {tool_id}")

    # 3. Peak DSI Localization
    abs_diff_csv = os.path.join(base_dir, "pixel_comparison_per_tool", f"{tool_id}_abs_diff_per_angle.csv")
    if os.path.exists(abs_diff_csv):
        df_diff = pd.read_csv(abs_diff_csv)
        max_idx = df_diff["dsi_percent_per_angle"].idxmax()
        peak_row = df_diff.loc[max_idx]

        # Save peak report CSV
        peak_df = pd.DataFrame([{
            "tool_id": tool_id,
            "peak_dsi_percent": round(float(peak_row["dsi_percent_per_angle"]), 4),
            "flute_pair": peak_row["pair_key"],
            "flute_i": peak_row["region_i"],
            "angle_i": peak_row["angle_i"],
            "frame_i": peak_row["frame_i"],
            "count_i": round(float(peak_row["count_i"]), 2),
            "flute_j": peak_row["region_j"],
            "angle_j": peak_row["angle_j"],
            "frame_j": peak_row["frame_j"],
            "count_j": round(float(peak_row["count_j"]), 2),
            "abs_difference": round(float(peak_row["abs_difference"]), 2),
        }])
        peak_df.to_csv(os.path.join(tool_dir, f"{tool_id}_peak_dsi_report.csv"), index=False)

        # Generate side-by-side Peak DSI comparison plot
        fpath_i = os.path.join(base_dir, "masks_tilted", tool_id, peak_row["frame_i"])
        fpath_j = os.path.join(base_dir, "masks_tilted", tool_id, peak_row["frame_j"])
        img_i = cv2.imread(fpath_i, cv2.IMREAD_GRAYSCALE)
        img_j = cv2.imread(fpath_j, cv2.IMREAD_GRAYSCALE)

        if img_i is not None and img_j is not None:
            fig_sbs, (ax_i, ax_j) = plt.subplots(1, 2, figsize=(14, 8), dpi=150)
            
            # Find centerlines for both peak frames
            def get_cx(img):
                mids = []
                for y in range(max(0, roi_top - 10), roi_top):
                    w_idx = np.where(img[y, :] == 255)[0]
                    if len(w_idx) > 1:
                        mids.append((w_idx[0] + w_idx[-1]) / 2.0)
                return float(np.mean(mids)) if mids else static_cx

            cxi = get_cx(img_i)
            cxj = get_cx(img_j)

            half_w = int(round(mask_width / 2.0))
            x_m = int(max(half_w + 100, 480))
            y_min = max(0, roi_top - 150)
            y_max = min(img_i.shape[0], roi_bottom + 80)

            for ax, im, cx_val, fn, reg, ang, cnt_val in [
                (ax_i, img_i, cxi, peak_row["frame_i"], peak_row["region_i"], peak_row["angle_i"], peak_row["count_i"]),
                (ax_j, img_j, cxj, peak_row["frame_j"], peak_row["region_j"], peak_row["angle_j"], peak_row["count_j"])
            ]:
                crp = im[y_min:y_max, max(0, int(cx_val - x_m)):min(im.shape[1], int(cx_val + x_m))]
                ax.imshow(crp, cmap="gray", origin="upper",
                          extent=[max(0, int(cx_val - x_m)), min(im.shape[1], int(cx_val + x_m)), y_max, y_min])
                ax.axhline(roi_top, color="gold", linestyle="--", linewidth=1.8, label=f"ROI Top (y={roi_top})")
                ax.axhline(roi_bottom, color="gold", linestyle="-", linewidth=1.8, label=f"ROI Bottom (y={roi_bottom})")
                ax.add_patch(Rectangle((cx_val - x_m, max(0, roi_top - 10)), 2 * x_m, 10,
                                       facecolor="lime", alpha=0.4, edgecolor="green", linewidth=1.2, label="10-Row Band"))
                ax.axvline(cx_val, color="lime", linewidth=2.2, label=f"xc={cx_val:.1f}px")
                ax.add_patch(Rectangle((cx_val, roi_top), im.shape[1] - cx_val, roi_height,
                                       facecolor="cyan", alpha=0.3, edgecolor="cyan", linewidth=1.5, label="Right Half ROI"))
                ax.set_xlim(cx_val - x_m, cx_val + x_m)
                ax.set_ylim(y_max, y_min)
                ax.set_title(f"{reg} Flute [{ang:.2f}°]\nFrame: {fn}\nRight Pixels = {cnt_val:,.1f}", fontsize=11, fontweight="bold")
                ax.legend(loc="upper right", fontsize=8.5)

            fig_sbs.suptitle(f"{tool_id} — PEAK DSI DIAGNOSTIC COMPARISON\n"
                             f"Flute Pair: {peak_row['pair_key']}  |  Peak DSI = {peak_row['dsi_percent_per_angle']:.4f}%\n"
                             f"Pixel Difference = {peak_row['abs_difference']:,.1f} px",
                             fontsize=13, fontweight="bold", y=0.98)
            sbs_png = os.path.join(tool_dir, f"{tool_id}_PEAK_DSI_SIDE_BY_SIDE_COMPARISON.png")
            fig_sbs.savefig(sbs_png, bbox_inches="tight")
            plt.close(fig_sbs)
            print(f"  -> Generated Peak DSI side-by-side comparison for {tool_id}: {os.path.basename(sbs_png)}")

    return tool_id

def main():
    base_dir = "/home/alifalah/Projects/DATA"
    out_parent = os.path.join(base_dir, "six_tools_full_degree_debug")
    os.makedirs(out_parent, exist_ok=True)

    six_tools = ["tool008", "tool006", "tool073", "tool057", "tool076", "tool092"]

    print("=== Starting Full-Degree Centerline Debug Generation for 6 Tools ===")
    print(f"Target Directory: {out_parent}")
    print(f"Tools to process: {', '.join(six_tools)}\n")

    for tid in six_tools:
        process_tool(tid, base_dir, out_parent)

    print("\n=== All 6 Tools Processed Successfully! ===")
    print(f"Browse results in: {out_parent}")

if __name__ == "__main__":
    main()
