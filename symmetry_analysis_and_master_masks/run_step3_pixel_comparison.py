#!/usr/bin/env python3
"""
Step 3: Right-Side White-Pixel Comparison per Tool with Individual Above-ROI Centerlines.

Key Methodology:
1. Dynamic Cutting-Tip ROI: (H_ROI = 0.45 * W_master) defined globally for each tool.
2. Per-Frame Individual Centerline:
   - Evaluated across the 10-row calibration band IMMEDIATELY ABOVE the Dynamic ROI:
     y in [max(0, roi_top - 10), roi_top].
   - Immune to cutting edge chipping, fracture, and wear occurring in the active cutting zone.
   - Near-field reference (offset < 10 px) eliminating angular lever-arm error.
   - Dynamically tracks fixture runout / spindle wobble frame-by-frame.
3. Degree-Based Flute Segmentation:
   - Span = 180 / N, Pitch = 360 / N.
   - Sequential rotational mapping for raw Basler camera frames.
4. Signal Smoothing:
   - Centered rolling moving average (window W = 20).
5. Dimensionless Symmetry Index (DSI):
   - All pairwise flute comparisons (N choose 2) and global overall tool DSI.
6. Publication Outputs:
   - Main outputs saved flat in DATA/pixel_comparison_per_tool/ (PNG 300 DPI + vector PDF).
   - Detailed debug analysis folders saved in DATA/centerline_debug/temp_<tool_id>_analysis/
     containing 8-10 sample-angle ROI bisector figures, side-by-side flute comparisons,
     and fixed runout stacked figures.
"""

import os
import sys
import glob
import re
import json
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# TrueType font export for Inkscape compatibility
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "DejaVu Sans"

# Add Tool_Condition_Monitoring to sys.path
sys.path.append("/home/alifalah/Projects/Tool_Condition_Monitoring")
from symmetry_analysis_and_master_masks.perspective.find_optimal_offset import (
    OffsetAnalysisConfig,
    _smooth_values,
    _overall_dsi_percent,
    _plot_overlay_dsi_stacked,
)


def extract_angle(fpath, file_idx=0, total_files=1):
    basename = os.path.basename(fpath)
    m = re.search(r"([0-9]+\.?[0-9]*)_degrees", basename)
    if m:
        return float(m.group(1))
    m = re.match(r"^([0-9]+\.?[0-9]*)", basename)
    if m:
        return float(m.group(1))
    # Fallback for sequential raw camera frame filenames
    return float(file_idx) * (360.0 / max(1, float(total_files)))


def process_single_tool(tool_id, base_dir, out_dir, debug_parent_dir, meta_dict):
    t0 = time.time()
    tool_info = meta_dict.get(tool_id, {})
    edges = int(tool_info.get("edges", 2))
    tool_type = str(tool_info.get("type", "unknown"))
    condition = str(tool_info.get("condition", "unknown"))
    diameter_mm = tool_info.get("diameter_mm", None)
    coating = tool_info.get("coating", None)
    material = tool_info.get("material", None)

    # 1. Per-tool tilt
    tilt_json_path = os.path.join(base_dir, "tilt_calculation_per_tool", f"{tool_id}_tilt.json")
    if not os.path.exists(tilt_json_path):
        return tool_id, False, f"Missing tilt file: {tilt_json_path}", None

    with open(tilt_json_path, "r", encoding="utf-8") as f:
        tilt_data = json.load(f)
    tilt_deg = float(tilt_data.get("tilt_angle_degrees", 0.0))
    rot_angle = -tilt_deg
    static_cx = float(tilt_data.get("centerline", {}).get("intercept", 1350.0))

    TOOL_ROI_HEIGHT_OVERRIDES = {
        "tool073": 400,
        "tool092": 180,
    }

    # 2. Dynamic ROI bounds
    sym_meta_path = os.path.join(out_dir, f"{tool_id}_symmetry_metadata.json")
    if os.path.exists(sym_meta_path):
        with open(sym_meta_path, "r", encoding="utf-8") as f:
            prev_meta = json.load(f)
        roi_bottom = int(prev_meta["dynamic_roi"]["roi_bottom_px"])
        if tool_id in TOOL_ROI_HEIGHT_OVERRIDES:
            roi_height = int(TOOL_ROI_HEIGHT_OVERRIDES[tool_id])
            roi_top = max(0, roi_bottom - roi_height)
        else:
            roi_top = int(prev_meta["dynamic_roi"]["roi_top_px"])
            roi_height = int(prev_meta["dynamic_roi"]["roi_height_px"])
        mask_width = int(prev_meta["dynamic_roi"]["mask_width_px"])
    else:
        # Fallback to computing from masks_tilted bounds
        mask_tilted_dir = os.path.join(base_dir, "masks_tilted", tool_id)
        sample_files = sorted(glob.glob(os.path.join(mask_tilted_dir, "*.png")))[::15]
        max_bot = 0
        min_lx, max_rx = 9999, 0
        for sf in sample_files:
            sim = cv2.imread(sf, cv2.IMREAD_GRAYSCALE)
            if sim is None: continue
            ys = np.where(sim.any(axis=1))[0]
            xs = np.where(sim.any(axis=0))[0]
            if len(ys) > 0: max_bot = max(max_bot, ys[-1])
            if len(xs) > 0:
                min_lx = min(min_lx, xs[0])
                max_rx = max(max_rx, xs[-1])
        mask_width = int(max_rx - min_lx)
        roi_height = int(round(0.45 * mask_width))
        roi_bottom = int(max_bot)
        roi_top = max(0, roi_bottom - roi_height)

    # 3. Load tilted frames
    mask_tilted_dir = os.path.join(base_dir, "masks_tilted", tool_id)
    raw_files = sorted(glob.glob(os.path.join(mask_tilted_dir, "*.png")))
    if not raw_files:
        return tool_id, False, f"No frames in {mask_tilted_dir}", None

    total_tool_frames = len(raw_files)
    file_angles = [extract_angle(f, i, total_tool_frames) for i, f in enumerate(raw_files)]
    sorted_pairs = sorted(zip(file_angles, raw_files), key=lambda x: x[0])
    angles = [p[0] for p in sorted_pairs]
    frame_files = [p[1] for p in sorted_pairs]

    pitch = 360.0 / edges
    span = 180.0 / edges

    region_indices = []
    for k in range(edges):
        r_start = k * pitch
        r_end = r_start + span
        k_indices = [i for i, a in enumerate(angles) if r_start <= a < r_end]
        region_indices.append(k_indices)

    pair_count = min(len(ki) for ki in region_indices)
    if pair_count < 2:
        return tool_id, False, f"Insufficient aligned frames (pair_count={pair_count})", None

    display_ranges = tuple((int(round(k * pitch)), int(round(k * pitch + span))) for k in range(edges))
    region_ranges = tuple((0, pair_count - 1) for _ in range(edges))

    cfg = OffsetAnalysisConfig(
        output_formats=("png", "pdf"),
        smoothing_enabled=True,
        smoothing_window=20,
        smoothing_strength=1.0,
        stack_overlay_abs_diff=True,
        manual_legend_ranges=True,
        legend_ranges=display_ranges,
        include_top_caption=True,
    )

    # 4. Process all frames with 10-row above-ROI centerline
    counts_rows = []
    frame_cache = {}  # for debug figure generation

    for idx in range(pair_count):
        row = {"pair_idx": idx}
        for k in range(edges):
            f_idx = region_indices[k][idx]
            fpath = frame_files[f_idx]
            fname = os.path.basename(fpath)
            deg = angles[f_idx]
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)

            if img is None:
                cx = static_cx
                cnt = 0
                l_cnt = 0
            else:
                h_img, w_img = img.shape
                # 10 rows immediately above roi_top: [roi_top - 10, roi_top]
                y_start = max(0, roi_top - 10)
                y_end = roi_top
                if y_end - y_start < 5:
                    y_start = roi_top
                    y_end = min(h_img, roi_top + 10)

                mids = []
                for y in range(y_start, y_end):
                    w_idx = np.where(img[y, :] == 255)[0]
                    if len(w_idx) > 1:
                        mids.append((w_idx[0] + w_idx[-1]) / 2.0)
                cx = float(np.mean(mids)) if mids else static_cx

                # Right half ROI count
                roi_img = img[roi_top:roi_bottom, :]
                r_cols = np.arange(roi_img.shape[1])[None, :] > cx
                cnt = int(np.count_nonzero((roi_img == 255) & r_cols))
                l_cnt = int(np.count_nonzero((roi_img == 255) & ~r_cols))

            row[f"frame_r{k+1}"] = fname
            row[f"angle_r{k+1}"] = round(deg, 3)
            row[f"count_r{k+1}"] = cnt
            row[f"centerline_r{k+1}"] = round(cx, 2)

            frame_cache[fname] = {
                "fpath": fpath,
                "angle": deg,
                "cx": cx,
                "r_cnt": cnt,
                "l_cnt": l_cnt,
            }
        counts_rows.append(row)

    counts_df = pd.DataFrame(counts_rows)
    for k in range(edges):
        counts_df[f"processed_count_r{k+1}"] = _smooth_values(counts_df[f"count_r{k+1}"], cfg)

    # 5. Pairwise metrics
    pair_rows = []
    for _, row in counts_df.iterrows():
        for i in range(edges):
            for j in range(i + 1, edges):
                pi = float(row[f"processed_count_r{i+1}"])
                pj = float(row[f"processed_count_r{j+1}"])
                diff = abs(pi - pj)
                tot = pi + pj
                dsi = (200.0 * diff / tot) if tot > 0 else 0.0
                pair_rows.append({
                    "pair_idx": int(row["pair_idx"]),
                    "pair_key": f"R{i+1}_vs_R{j+1}",
                    "region_i": f"R{i+1}",
                    "region_j": f"R{j+1}",
                    "frame_i": row[f"frame_r{i+1}"],
                    "frame_j": row[f"frame_r{j+1}"],
                    "angle_i": row[f"angle_r{i+1}"],
                    "angle_j": row[f"angle_r{j+1}"],
                    "count_i": pi,
                    "count_j": pj,
                    "abs_difference": diff,
                    "ratio": (diff / tot) if tot > 0 else 0.0,
                    "dsi_percent_per_angle": dsi,
                })

    pairwise_df = pd.DataFrame(pair_rows)
    overall_dsi = _overall_dsi_percent(pairwise_df)
    mean_abs_diff = float(pairwise_df["abs_difference"].mean())

    summary_df = (
        pairwise_df.groupby("pair_key", as_index=False)
        .agg(
            mean_difference=("abs_difference", "mean"),
            std_difference=("abs_difference", "std"),
            max_difference=("abs_difference", "max"),
            mean_dsi_percent=("dsi_percent_per_angle", "mean"),
            max_dsi_percent=("dsi_percent_per_angle", "max"),
        )
    )

    # 6. Output to pixel_comparison_per_tool (Flat)
    out_prefix = os.path.join(out_dir, tool_id)
    _plot_overlay_dsi_stacked(counts_df, pairwise_df, tool_id, out_prefix, cfg, region_ranges, display_ranges)

    pixel_counts_csv = f"{out_prefix}_right_side_pixel_counts.csv"
    counts_df.to_csv(pixel_counts_csv, index=False)

    abs_diff_csv = f"{out_prefix}_abs_diff_per_angle.csv"
    pairwise_df.to_csv(abs_diff_csv, index=False)

    comp_summary_csv = f"{out_prefix}_comparison_summary.csv"
    summary_df.to_csv(comp_summary_csv, index=False)

    metadata = {
        "tool_id": tool_id,
        "tool_info": {
            "type": tool_type,
            "diameter_mm": diameter_mm,
            "edges": edges,
            "condition": condition,
            "material": material,
            "coating": coating,
        },
        "tilt_calibration": {
            "tilt_angle_deg": tilt_deg,
            "rotation_angle_deg": rot_angle,
            "centerline_method": "individual_above_roi_10_rows",
            "reference_band": f"y in [{max(0, roi_top-10)}, {roi_top}]",
            "static_centerline_intercept": static_cx,
        },
        "dynamic_roi": {
            "roi_top_px": roi_top,
            "roi_bottom_px": roi_bottom,
            "roi_height_px": roi_height,
            "mask_width_px": mask_width,
            "factor": 0.45,
        },
        "analysis_settings": {
            "smoothing_enabled": True,
            "smoothing_window": 20,
            "smoothing_strength": 1.0,
            "pitch_deg": pitch,
            "span_deg": span,
            "display_regions_deg": [f"{r[0]}-{r[1]}" for r in display_ranges],
            "pair_count": pair_count,
        },
        "results": {
            "mean_abs_diff": mean_abs_diff,
            "overall_dsi_percent": overall_dsi,
            "pairwise_summary": summary_df.to_dict(orient="records"),
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(f"{out_prefix}_symmetry_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # 7. Output to centerline_debug/temp_<tool_id>_analysis/
    debug_dir = os.path.join(debug_parent_dir, f"temp_{tool_id}_analysis")
    os.makedirs(debug_dir, exist_ok=True)

    # 7.1 Sample angles for debug figures
    sample_angles = []
    # Region 1 angles (start, 33%, 67%, end)
    for frac in [0.0, 0.33, 0.67, 1.0]:
        sample_angles.append(0.0 + frac * span)
    # Region 2 start and end
    sample_angles.append(pitch)
    sample_angles.append(pitch + span)
    # Opposite region angles
    opp_k = edges // 2
    for frac in [0.0, 0.33, 0.67, 1.0]:
        sample_angles.append(opp_k * pitch + frac * span)

    sample_angles = sorted(list(set([round(a, 1) for a in sample_angles])))

    for ang in sample_angles:
        best_idx = min(range(len(angles)), key=lambda i: abs(angles[i] - ang))
        fpath = frame_files[best_idx]
        actual_ang = angles[best_idx]
        fname = os.path.basename(fpath)
        fdata = frame_cache.get(fname)
        if fdata is None: continue

        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        cx_val = fdata["cx"]
        r_val = fdata["r_cnt"]
        l_val = fdata["l_cnt"]

        # Dynamic zoom margin based on actual tool width
        half_w = int(round(mask_width / 2.0))
        x_margin = int(max(half_w + 120, 500))
        rect_w = int(w_img - cx_val)

        fig, ax = plt.subplots(figsize=(9, 10), dpi=150)
        ax.imshow(img, cmap="gray", origin="upper")
        ax.axhline(roi_top, color="gold", linewidth=2.0, linestyle="--", label=f"ROI Top (y={roi_top})")
        ax.axhline(roi_bottom, color="gold", linewidth=2.0, linestyle="-", label=f"ROI Bottom (y={roi_bottom})")

        # 10-row band spanning entire zoom window
        y_start = max(0, roi_top - 10)
        rect_band = Rectangle((cx_val - x_margin, y_start), 2 * x_margin, roi_top - y_start,
                              facecolor="lime", alpha=0.45, edgecolor="green", linewidth=1.2, label="10-Row Calib Band")
        ax.add_patch(rect_band)
        ax.axvline(cx_val, color="lime", linewidth=2.2, label=f"Centerline (x={cx_val:.1f})")

        rect_roi = Rectangle((cx_val, roi_top), rect_w, roi_height,
                             facecolor="cyan", alpha=0.3, edgecolor="cyan", linewidth=1.5, label=f"Right Half ROI (H={roi_height}px)")
        ax.add_patch(rect_roi)

        ax.set_xlim(cx_val - x_margin, cx_val + x_margin)
        ax.set_ylim(roi_bottom + 80, roi_top - 160)
        ax.set_title(f"{tool_id} — {fname} [{actual_ang:.1f}°]\nROI H={roi_height}px | Right Pixels={r_val:,} | Left Pixels={l_val:,}\nCenterline xc={cx_val:.1f}px",
                     fontsize=11, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8.5, frameon=True)

        out_png = os.path.join(debug_dir, f"{tool_id}_{actual_ang:05.1f}deg_roi_bisector.png")
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)

    # 7.2 Side-by-side comparison (Flute 1 vs Opposite Flute)
    f0_idx = region_indices[0][0]
    fopp_idx = region_indices[opp_k][0]
    f0_name = os.path.basename(frame_files[f0_idx])
    fopp_name = os.path.basename(frame_files[fopp_idx])

    img0 = cv2.imread(frame_files[f0_idx], cv2.IMREAD_GRAYSCALE)
    imgopp = cv2.imread(frame_files[fopp_idx], cv2.IMREAD_GRAYSCALE)

    if img0 is not None and imgopp is not None:
        fig_sbs, axes_sbs = plt.subplots(1, 2, figsize=(15, 9), dpi=200)
        half_w = int(round(mask_width / 2.0))
        x_margin_sbs = int(max(half_w + 100, 480))
        for ax, im, fn, reg_name in [(axes_sbs[0], img0, f0_name, f"Region R1 [{angles[f0_idx]:.1f}°]"),
                                     (axes_sbs[1], imgopp, fopp_name, f"Opposite Region R{opp_k+1} [{angles[fopp_idx]:.1f}°]")]:
            fc = frame_cache[fn]
            cx_v = fc["cx"]
            ax.imshow(im, cmap="gray", origin="upper")
            ax.axhline(roi_top, color="gold", linewidth=2.0, linestyle="--", label=f"ROI Top (y={roi_top})")
            ax.axhline(roi_bottom, color="gold", linewidth=2.0, linestyle="-", label=f"ROI Bottom (y={roi_bottom})")
            y_start = max(0, roi_top - 10)
            ax.add_patch(Rectangle((cx_v - x_margin_sbs, y_start), 2 * x_margin_sbs, roi_top - y_start, facecolor="lime", alpha=0.4, edgecolor="green", linewidth=1.2, label="10-Row Calib Band"))
            ax.axvline(cx_v, color="lime", linewidth=2.2, label=f"Centerline (x={cx_v:.1f})")
            ax.add_patch(Rectangle((cx_v, roi_top), im.shape[1] - cx_v, roi_height, facecolor="cyan", alpha=0.3, edgecolor="cyan", linewidth=1.5, label=f"Right Half ROI"))
            ax.set_xlim(cx_v - x_margin_sbs, cx_v + x_margin_sbs)
            ax.set_ylim(roi_bottom + 80, roi_top - 150)
            ax.set_title(f"{reg_name}: {fn}\nRight Pixels={fc['r_cnt']:,} | Left Pixels={fc['l_cnt']:,}\nCenterline={cx_v:.1f}px", fontsize=11, fontweight="bold")
            ax.legend(loc="upper right", fontsize=8.5, frameon=True)

        fig_sbs.suptitle(f"{tool_id}: Opposing Flute Symmetry Alignment (Runout-Compensated)", fontsize=13, fontweight="bold", y=0.98)
        sbs_path = os.path.join(debug_dir, f"{tool_id}_side_by_side_flutes.png")
        fig_sbs.savefig(sbs_path, bbox_inches="tight")
        plt.close(fig_sbs)

    # 7.3 Save stacked overlay & CSVs inside debug dir
    debug_prefix = os.path.join(debug_dir, f"{tool_id}_fixed_runout")
    _plot_overlay_dsi_stacked(counts_df, pairwise_df, tool_id, debug_prefix, cfg, region_ranges, display_ranges)
    counts_df.to_csv(f"{debug_prefix}_right_side_pixel_counts.csv", index=False)
    pairwise_df.to_csv(f"{debug_prefix}_abs_diff_per_angle.csv", index=False)
    summary_df.to_csv(f"{debug_prefix}_comparison_summary.csv", index=False)

    summary_entry = {
        "tool_id": tool_id,
        "type": tool_type,
        "diameter_mm": diameter_mm,
        "edges": edges,
        "condition": condition,
        "material": material,
        "coating": coating,
        "tilt_angle_deg": round(tilt_deg, 4),
        "rotation_angle_deg": round(rot_angle, 4),
        "pair_count": pair_count,
        "overall_dsi_percent": round(overall_dsi, 4),
        "mean_abs_diff": round(mean_abs_diff, 2),
        "roi_height_px": roi_height,
        "duration_sec": round(time.time() - t0, 2),
    }

    return tool_id, True, f"Success (DSI={overall_dsi:.3f}%)", summary_entry


def main():
    parser = argparse.ArgumentParser(description="Step 3: Run right-side pixel comparison with individual centerlines.")
    parser.add_argument("--base_dir", default="/home/alifalah/Projects/DATA")
    parser.add_argument("--tools", nargs="*", default=None, help="Specific tools to process")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    args = parser.parse_args()

    base_dir = args.base_dir
    out_dir = os.path.join(base_dir, "pixel_comparison_per_tool")
    debug_parent_dir = os.path.join(base_dir, "centerline_debug")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(debug_parent_dir, exist_ok=True)

    meta_path = os.path.join(base_dir, "tools_metadata.csv")
    meta_df = pd.read_csv(meta_path)
    meta_dict = {}
    for _, r in meta_df.iterrows():
        meta_dict[r["tool_id"]] = r.to_dict()

    if args.tools:
        tools_to_run = args.tools
    else:
        all_dirs = sorted(glob.glob(os.path.join(base_dir, "masks_tilted", "tool*")))
        tools_to_run = [os.path.basename(d) for d in all_dirs if os.path.isdir(d)]
        # Exclude tool077
        tools_to_run = [t for t in tools_to_run if t != "tool077"]

    print(f"=== Starting Step 3 Processing with Individual Above-ROI Centerlines ===")
    print(f"Total tools to process: {len(tools_to_run)}")
    print(f"Output directory (Main):  {out_dir}")
    print(f"Output directory (Debug): {debug_parent_dir}")
    print(f"Parallel workers: {args.workers}")

    t_start = time.time()
    summary_records = []
    success_count = 0
    fail_count = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_map = {
            executor.submit(process_single_tool, tid, base_dir, out_dir, debug_parent_dir, meta_dict): tid
            for tid in tools_to_run
        }
        for future in as_completed(future_map):
            tid = future_map[future]
            try:
                tool_id, ok, msg, record = future.result()
                if ok:
                    success_count += 1
                    summary_records.append(record)
                    print(f"[{success_count + fail_count}/{len(tools_to_run)}] [OK] {tool_id}: {msg} ({record['duration_sec']}s)")
                else:
                    fail_count += 1
                    print(f"[{success_count + fail_count}/{len(tools_to_run)}] [FAIL] {tool_id}: {msg}")
            except Exception as e:
                fail_count += 1
                print(f"[{success_count + fail_count}/{len(tools_to_run)}] [ERROR] {tid}: {str(e)}")

    # Save compiled summary table
    if summary_records:
        new_sum_df = pd.DataFrame(summary_records)
        summary_csv = os.path.join(out_dir, "all_tools_step3_summary.csv")
        if os.path.exists(summary_csv) and args.tools:
            existing_df = pd.read_csv(summary_csv)
            # Remove updated tools and append new records
            updated_ids = set(new_sum_df["tool_id"])
            existing_df = existing_df[~existing_df["tool_id"].isin(updated_ids)]
            sum_df = pd.concat([existing_df, new_sum_df], ignore_index=True)
        else:
            sum_df = new_sum_df
        sum_df.sort_values(by="tool_id", inplace=True)
        sum_df.to_csv(summary_csv, index=False)
        print(f"\nSaved updated population summary to: {summary_csv}")

    total_time = time.time() - t_start
    print(f"\n========================================================")
    print(f"Step 3 Complete in {total_time:.2f} seconds ({total_time / 60.0:.2f} min)")
    print(f"Success: {success_count}/{len(tools_to_run)}")
    print(f"Failed:  {fail_count}/{len(tools_to_run)}")
    print(f"========================================================")


if __name__ == "__main__":
    main()
