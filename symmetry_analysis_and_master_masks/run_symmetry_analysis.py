#!/usr/bin/env python3
"""
Half-Tool Symmetry Analysis Pipeline for Tool Condition Monitoring.

Computes:
1. Master centerline (m_center, b_center) fitted to rotated master masks.
2. Dynamic ROI half-tool signal extraction (right half, left half, and total).
3. Moving-average circular wrap smoothing.
4. N-edge cutting-flute region segmentation and pairwise symmetry comparisons (DSI %).
5. Comprehensive publication-quality PNG (300 DPI) and vector PDF figures.
6. Per-tool metadata JSON, frame CSVs, and pairwise CSVs.
7. Global master summary table: symmetry_summary.csv.
"""

import os
import re
import glob
import json
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import convolve1d

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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


def select_widest_rows(ys, left_x, right_x, width_percentile=50.0, min_rows=20):
    widths = right_x - left_x
    threshold = np.percentile(widths, width_percentile)
    keep = widths >= threshold
    if np.sum(keep) < min_rows:
        top_idx = np.argsort(widths)[-min_rows:]
        keep = np.zeros_like(widths, dtype=bool)
        keep[top_idx] = True
    return ys[keep], left_x[keep], right_x[keep]


def fit_centerline(master_mask):
    """Fit outer boundary lines and center bisector line to master mask."""
    ys, lx, rx = get_boundaries(master_mask)
    if ys is None or len(ys) < 20:
        h, w = master_mask.shape
        return 0.0, float(w / 2.0)
    ys_k, lx_k, rx_k = select_widest_rows(ys, lx, rx)
    m_l, b_l = np.polyfit(ys_k, lx_k, 1)
    m_r, b_r = np.polyfit(ys_k, rx_k, 1)
    m_c = float((m_l + m_r) / 2.0)
    b_c = float((b_l + b_r) / 2.0)
    return m_c, b_c


def extract_angle_from_filename(filename):
    basename = os.path.basename(filename)
    m = re.search(r"([0-9]+\.?[0-9]*)_degrees", basename)
    if m:
        return float(m.group(1))
    m = re.match(r"^([0-9]+\.?[0-9]*)", basename)
    if m:
        return float(m.group(1))
    return 0.0


def process_tool_symmetry(tool_id, base_dir, out_root, meta_dict):
    """Process a single tool: half signals, symmetry index, plots, and metadata."""
    t0 = time.time()
    tool_out_dir = os.path.join(out_root, tool_id)
    os.makedirs(tool_out_dir, exist_ok=True)

    tool_info = meta_dict.get(tool_id, {})
    edges = int(tool_info.get("edges", 2))
    tool_type = str(tool_info.get("type", "unknown"))
    condition = str(tool_info.get("condition", "unknown"))
    diameter_mm = tool_info.get("diameter_mm", None)
    coating = tool_info.get("coating", None)
    material = tool_info.get("material", None)
    color = tool_info.get("color", None)
    inspection_status = tool_info.get("inspection_status", None)

    # 1. Master mask & Centerline
    master_path = os.path.join(base_dir, "master_masks_tilted", f"{tool_id}_master_mask.png")
    if not os.path.exists(master_path):
        return tool_id, False, f"Missing master mask: {master_path}", None
    master_mask = cv2.imread(master_path, cv2.IMREAD_GRAYSCALE)
    if master_mask is None:
        return tool_id, False, f"Could not read master mask: {master_path}", None

    h, w = master_mask.shape
    m_c, b_c = fit_centerline(master_mask)

    # 2. Dynamic ROI bounds
    roi_meta_path = os.path.join(base_dir, "whole_tool_dynamic_roi_signals", tool_id, f"{tool_id}_metadata.json")
    y_top, y_bot = None, None
    if os.path.exists(roi_meta_path):
        try:
            with open(roi_meta_path, "r", encoding="utf-8") as f:
                roi_data = json.load(f).get("dynamic_roi", {})
                y_top = int(roi_data["roi_top_px"])
                y_bot = int(roi_data["roi_bottom_px"])
        except Exception:
            pass

    if y_top is None or y_bot is None:
        ys_rm = np.where(master_mask.any(axis=1))[0]
        white_cols = np.where(master_mask.any(axis=0))[0]
        if len(ys_rm) == 0 or len(white_cols) < 2:
            return tool_id, False, "Empty master mask boundaries", None
        bottom_y = int(ys_rm[-1])
        mask_width = int(white_cols[-1] - white_cols[0])
        roi_height = int(round(0.45 * mask_width))
        y_top = max(0, bottom_y - roi_height)
        y_bot = bottom_y

    roi_height = y_bot - y_top

    # Precompute fast right-half boolean mask for the ROI rows
    y_coords = np.arange(y_top, y_bot)
    cx_coords = np.clip(np.round(m_c * y_coords + b_c).astype(int), 0, w - 1)
    right_mask = np.arange(w)[None, :] > cx_coords[:, None]

    # 3. Process tilted frames
    tilted_frames_dir = os.path.join(base_dir, "masks_tilted", tool_id)
    frame_files = sorted(glob.glob(os.path.join(tilted_frames_dir, "*.png")), key=extract_angle_from_filename)
    if not frame_files:
        return tool_id, False, f"No tilted frames found in {tilted_frames_dir}", None

    angles = []
    frame_names = []
    r_counts_raw = []
    l_counts_raw = []

    for fpath in frame_files:
        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        roi_img = (img[y_top:y_bot, :] == 255)
        r_cnt = int(np.count_nonzero(roi_img & right_mask))
        l_cnt = int(np.count_nonzero(roi_img & ~right_mask))
        angles.append(extract_angle_from_filename(fpath))
        frame_names.append(os.path.basename(fpath))
        r_counts_raw.append(r_cnt)
        l_counts_raw.append(l_cnt)

    if not angles:
        return tool_id, False, "No valid frames read", None

    angles = np.array(angles, dtype=float)
    r_counts_raw = np.array(r_counts_raw, dtype=float)
    l_counts_raw = np.array(l_counts_raw, dtype=float)
    tot_counts_raw = r_counts_raw + l_counts_raw

    # Moving average wrap-around smoothing (window = 10)
    window = 10
    weights = np.ones(window) / float(window)
    r_smooth = convolve1d(r_counts_raw, weights, mode="wrap")
    l_smooth = convolve1d(l_counts_raw, weights, mode="wrap")
    tot_smooth = r_smooth + l_smooth

    # Save half signals CSV
    half_signals_df = pd.DataFrame({
        "frame_idx": np.arange(len(angles)),
        "angle_deg": angles,
        "frame_filename": frame_names,
        "right_count_raw": r_counts_raw.astype(int),
        "left_count_raw": l_counts_raw.astype(int),
        "total_count_raw": tot_counts_raw.astype(int),
        "right_count_smooth": np.round(r_smooth, 2),
        "left_count_smooth": np.round(l_smooth, 2),
        "total_count_smooth": np.round(tot_smooth, 2),
    })
    half_signals_csv = os.path.join(tool_out_dir, f"{tool_id}_half_signals.csv")
    half_signals_df.to_csv(half_signals_csv, index=False)

    # 4. Region segmentation & Pairwise Symmetry
    phase_step = 360.0 / edges
    span = 180.0 / edges
    num_phase_points = max(10, int(round(len(angles) / (2.0 * edges))))
    phase_grid = np.linspace(0, span, num=num_phase_points, endpoint=False)

    region_signals = []
    region_ranges = {}
    for k in range(edges):
        r_start = k * phase_step
        r_end = r_start + span
        region_ranges[f"R{k+1}"] = [float(r_start), float(r_end)]
        mask_k = (angles >= r_start) & (angles < r_end)
        k_angles = angles[mask_k] - r_start
        k_vals = r_smooth[mask_k]
        if len(k_angles) >= 2:
            interp_vals = np.interp(phase_grid, k_angles, k_vals)
        else:
            interp_vals = np.zeros_like(phase_grid)
        region_signals.append(interp_vals)

    pairwise_cols = {
        "phase_idx": np.arange(num_phase_points),
        "phase_angle_deg": np.round(phase_grid, 4),
    }
    for k in range(edges):
        pairwise_cols[f"edge_R{k+1}_count"] = np.round(region_signals[k], 2)

    pair_dsi_dict = {}
    pairwise_summary = {}
    all_dsi_values = []

    for i in range(edges):
        for j in range(i + 1, edges):
            pair_key = f"R{i+1}_vs_R{j+1}"
            s_i = region_signals[i]
            s_j = region_signals[j]
            diff = np.abs(s_i - s_j)
            tot = s_i + s_j
            ratio = np.divide(diff, tot, out=np.zeros_like(diff), where=tot > 0)
            dsi = 200.0 * ratio

            pair_dsi_dict[pair_key] = dsi
            all_dsi_values.append(dsi)

            pairwise_cols[f"abs_diff_{pair_key}"] = np.round(diff, 2)
            pairwise_cols[f"ratio_{pair_key}"] = np.round(ratio, 6)
            pairwise_cols[f"dsi_percent_{pair_key}"] = np.round(dsi, 4)

            pairwise_summary[pair_key] = {
                "mean_dsi_percent": float(np.mean(dsi)),
                "max_dsi_percent": float(np.max(dsi)),
                "std_dsi_percent": float(np.std(dsi)),
                "mean_abs_diff": float(np.mean(diff)),
                "max_abs_diff": float(np.max(diff)),
            }

    pairwise_df = pd.DataFrame(pairwise_cols)
    pairwise_csv = os.path.join(tool_out_dir, f"{tool_id}_pairwise_symmetry.csv")
    pairwise_df.to_csv(pairwise_csv, index=False)

    concat_dsi = np.concatenate(all_dsi_values) if all_dsi_values else np.array([0.0])
    global_mean_dsi = float(np.mean(concat_dsi))
    global_max_dsi = float(np.max(concat_dsi))

    # 5. Metadata JSON
    metadata = {
        "tool_id": tool_id,
        "metadata": {
            "type": tool_type,
            "diameter_mm": diameter_mm,
            "edges": edges,
            "condition": condition,
            "material": material,
            "coating": coating,
            "color": color,
            "inspection_status": inspection_status,
        },
        "centerline": {
            "slope": m_c,
            "intercept": b_c,
            "center_x_at_mid_px": float(m_c * (y_top + y_bot) / 2.0 + b_c),
            "reference_master_mask": f"{tool_id}_master_mask.png",
        },
        "dynamic_roi": {
            "roi_top_px": y_top,
            "roi_bottom_px": y_bot,
            "roi_height_px": roi_height,
        },
        "symmetry_analysis": {
            "num_edges": edges,
            "phase_step_deg": phase_step,
            "comparison_span_deg": span,
            "region_ranges_deg": region_ranges,
            "global_mean_dsi_percent": global_mean_dsi,
            "global_max_dsi_percent": global_max_dsi,
            "pairwise_summary": pairwise_summary,
        },
    }
    metadata_path = os.path.join(tool_out_dir, f"{tool_id}_symmetry_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # 6. Plotting (PNG 300 DPI + Vector PDF)
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), dpi=300)
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]

    # Panel 1: Full 360 degree half-tool signals
    axes[0].plot(angles, r_smooth, label="Right Half (ROI)", color="#1f77b4", linewidth=1.8)
    axes[0].plot(angles, l_smooth, label="Left Half (ROI)", color="#ff7f0e", linewidth=1.5, alpha=0.75, linestyle="--")
    for k in range(edges):
        r_start = k * phase_step
        r_end = r_start + span
        axes[0].axvspan(r_start, r_end, color=palette[k % len(palette)], alpha=0.15,
                        label=f"Region R{k+1}" if k < 6 else None)
    axes[0].set_xlim(0, 360)
    axes[0].set_ylabel("White Pixels in ROI", fontsize=11, fontweight="bold")
    axes[0].set_title("1. Full 360° Rotational Half-Tool Signals with Cutting Edge Regions", fontsize=12, fontweight="bold")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="upper right", frameon=True, fontsize=9, ncol=min(edges + 2, 5))

    # Panel 2: Superimposed Edge Profiles over Phase Span
    for k in range(edges):
        r_start = k * phase_step
        r_end = r_start + span
        axes[1].plot(phase_grid, region_signals[k],
                     label=f"Edge R{k+1} [{r_start:.0f}° - {r_end:.0f}°]",
                     color=palette[k % len(palette)], linewidth=2.0)
    axes[1].set_xlim(0, span)
    axes[1].set_ylabel("Right Half Pixels", fontsize=11, fontweight="bold")
    axes[1].set_title(f"2. Superimposed Cutting Edge Profiles over Common Phase Span (0° - {span:.1f}°)", fontsize=12, fontweight="bold")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(loc="best", frameon=True, fontsize=9, ncol=min(edges, 4))

    # Panel 3: Pairwise DSI
    for pair_key, dsi in pair_dsi_dict.items():
        axes[2].plot(phase_grid, dsi, label=f"{pair_key} (mean: {np.mean(dsi):.1f}%)", linewidth=1.6)
    axes[2].axhline(global_mean_dsi, color="black", linestyle="--", linewidth=1.5,
                    label=f"Global Mean DSI: {global_mean_dsi:.2f}%")
    axes[2].set_xlim(0, span)
    axes[2].set_xlabel("Relative Phase Angle (Degrees)", fontsize=11, fontweight="bold")
    axes[2].set_ylabel("DSI (%)", fontsize=11, fontweight="bold")
    axes[2].set_title("3. Pairwise Dissymmetry Index (DSI) Across Cutting Edges", fontsize=12, fontweight="bold")
    axes[2].grid(True, linestyle=":", alpha=0.6)
    axes[2].legend(loc="upper right", frameon=True, fontsize=8, ncol=min(len(pair_dsi_dict) + 1, 4))

    cond_str = condition.upper()
    type_str = tool_type.capitalize()
    fig.suptitle(f"{tool_id} Symmetry Analysis — {edges}-Edge {type_str} (Condition: {cond_str})\n"
                 f"Global Mean DSI: {global_mean_dsi:.2f}% | Max DSI: {global_max_dsi:.2f}% | Centerline: x = {m_c:.5f}·y + {b_c:.1f}",
                 fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plot_png = os.path.join(tool_out_dir, f"{tool_id}_symmetry_plot.png")
    plot_pdf = os.path.join(tool_out_dir, f"{tool_id}_symmetry_plot.pdf")
    fig.savefig(plot_png, dpi=300)
    fig.savefig(plot_pdf)
    plt.close(fig)

    summary_entry = {
        "tool_id": tool_id,
        "type": tool_type,
        "diameter_mm": diameter_mm,
        "edges": edges,
        "condition": condition,
        "material": material,
        "coating": coating,
        "centerline_slope": m_c,
        "centerline_intercept": b_c,
        "roi_top_px": y_top,
        "roi_bottom_px": y_bot,
        "roi_height_px": roi_height,
        "global_mean_dsi_percent": global_mean_dsi,
        "global_max_dsi_percent": global_max_dsi,
        "num_frames": len(angles),
        "duration_sec": round(time.time() - t0, 2),
    }

    return tool_id, True, "Success", summary_entry


def main():
    parser = argparse.ArgumentParser(description="Run Half-Tool Symmetry Analysis Pipeline")
    parser.add_argument("--base-dir", default="/home/alifalah/Projects/DATA", help="Base DATA directory")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel processes")
    args = parser.parse_args()

    base_dir = args.base_dir
    out_root = os.path.join(base_dir, "symmetry")
    os.makedirs(out_root, exist_ok=True)

    meta_df = pd.read_csv(os.path.join(base_dir, "tools_metadata.csv"))
    meta_df = meta_df.where(pd.notnull(meta_df), None)
    meta_dict = meta_df.set_index("tool_id").to_dict("index")

    tilted_root = os.path.join(base_dir, "masks_tilted")
    tool_dirs = sorted([d for d in os.listdir(tilted_root) if os.path.isdir(os.path.join(tilted_root, d))])
    print(f"Starting Half-Tool Symmetry Analysis for {len(tool_dirs)} tools using {args.workers} workers...")
    print(f"Output directory: {out_root}")

    t_start = time.time()
    summary_entries = []
    completed = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_tool_symmetry, tid, base_dir, out_root, meta_dict): tid
            for tid in tool_dirs
        }

        for fut in as_completed(futures):
            tid, success, msg, entry = fut.result()
            if success:
                completed += 1
                summary_entries.append(entry)
            else:
                failed += 1
                print(f"ERROR on {tid}: {msg}")

            if (completed + failed) % 15 == 0 or (completed + failed) == len(tool_dirs):
                elapsed = time.time() - t_start
                print(f"Progress: {completed + failed}/{len(tool_dirs)} tools completed ({completed} ok, {failed} failed, {elapsed:.1f}s)")

    summary_df = pd.DataFrame(summary_entries)
    summary_df = summary_df.sort_values("tool_id")
    summary_csv = os.path.join(out_root, "symmetry_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    print(f"\nAll tools processed in {time.time() - t_start:.1f}s.")
    print(f"Summary table saved to: {summary_csv}")
    print(f"Total tools: {len(summary_df)} (Success: {completed}, Failed: {failed})")


if __name__ == "__main__":
    main()
