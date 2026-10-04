#!/usr/bin/env python3
"""
Calculate Tilt Angle per Tool from Master Masks with Academic Paper Debug Figures.

Fits outer boundaries and center bisector line to each tool's master mask to find:
- Tilt angle (degrees)
- Recommended rotation angle (-tilt_angle)
- Centerline slope and intercept

Generates:
1. High-resolution (300 DPI) debug figure: <tool_id>_tilt_angle_calculation.png
2. Structured metadata: <tool_id>_tilt.json
3. Plaintext angle file: <tool_id>_tilt_angle.txt
4. Master CSV summary: all_tools_tilt_summary.csv

All files are saved flat in a single parent directory for convenient arrow-key navigation.
"""

import os
import glob
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

FIG_DPI = 300
WIDTH_PERCENTILE_FOR_FIT = 50.0
MIN_ROWS_FOR_FIT = 20


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


def select_widest_rows(ys, left_x, right_x, width_percentile=WIDTH_PERCENTILE_FOR_FIT, min_rows=MIN_ROWS_FOR_FIT):
    widths = right_x - left_x
    threshold = np.percentile(widths, width_percentile)
    keep = widths >= threshold
    if np.sum(keep) < min_rows:
        top_idx = np.argsort(widths)[-min_rows:]
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


def render_tilt_angle_figure(binary_mask, ys, line_left, line_right, line_center, tilt_deg, out_path, tool_id=None, extra_info=""):
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
    ax.plot(x_center, y_plot, color="lime", linewidth=2.8, label="Center Bisector")
    ax.plot([vertical_x, vertical_x], [y_plot.min(), y_plot.max()],
            color="dodgerblue", linewidth=2.8, linestyle="--", label="True Vertical Reference")

    text_str = f"Tilt angle: {tilt_deg:+.3f}°\nRotation: {-tilt_deg:+.3f}°"
    if extra_info:
        text_str += f"\n{extra_info}"

    ax.text(0.03, 0.97, text_str,
            transform=ax.transAxes, va="top", ha="left", fontsize=14,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="black", alpha=0.92))

    ax.legend(loc="lower right", frameon=True, fontsize=11, facecolor="white", edgecolor="black")
    title = f"{tool_id}: Tilt Angle Calculation from Master Mask" if tool_id else "Tilt Angle Calculation from Master Mask"
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_axis_off()
    fig.tight_layout(pad=0.15)
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def process_single_tool(mask_path, out_dir, meta_dict):
    fname = os.path.basename(mask_path)
    tool_id = fname.replace("_master_mask.png", "")

    img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return tool_id, False, "Cannot read image", None

    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    ys, left_x, right_x = get_boundaries(binary)
    if ys is None or len(ys) < MIN_ROWS_FOR_FIT:
        return tool_id, False, "Empty or insufficient mask boundaries", None

    ys_fit, left_fit, right_fit = select_widest_rows(ys, left_x, right_x)
    if len(ys_fit) < 2:
        return tool_id, False, "Insufficient rows for linear fit", None

    line_left, line_right, line_center = fit_lines(ys_fit, left_fit, right_fit)
    tilt_deg = compute_tilt_deg(line_center[0])
    rotation_angle = -tilt_deg

    tool_info = meta_dict.get(tool_id, {})
    extra_info = f"Type: {tool_info.get('type', '')} | Cond: {tool_info.get('condition', '')} | Edges: {tool_info.get('edges', '')}"

    # 1. Debug figure
    fig_path = os.path.join(out_dir, f"{tool_id}_tilt_angle_calculation.png")
    render_tilt_angle_figure(binary, ys, line_left, line_right, line_center, tilt_deg, fig_path, tool_id=tool_id, extra_info=extra_info)

    # 2. Text file
    txt_path = os.path.join(out_dir, f"{tool_id}_tilt_angle.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Tilt angle (degrees): {tilt_deg:.6f}\n")
        f.write(f"Rotation angle (degrees): {rotation_angle:.6f}\n")

    # 3. JSON metadata
    json_path = os.path.join(out_dir, f"{tool_id}_tilt.json")
    tool_meta = {
        "tool_id": tool_id,
        "tilt_angle_degrees": tilt_deg,
        "rotation_angle_degrees": rotation_angle,
        "centerline": {
            "slope": float(line_center[0]),
            "intercept": float(line_center[1]),
        },
        "left_boundary": {
            "slope": float(line_left[0]),
            "intercept": float(line_left[1]),
        },
        "right_boundary": {
            "slope": float(line_right[0]),
            "intercept": float(line_right[1]),
        },
        "tool_info": {
            "type": tool_info.get("type"),
            "diameter_mm": tool_info.get("diameter_mm"),
            "edges": tool_info.get("edges"),
            "condition": tool_info.get("condition"),
            "material": tool_info.get("material"),
            "coating": tool_info.get("coating"),
        }
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(tool_meta, f, indent=2)

    summary_row = {
        "tool_id": tool_id,
        "type": tool_info.get("type"),
        "diameter_mm": tool_info.get("diameter_mm"),
        "edges": tool_info.get("edges"),
        "condition": tool_info.get("condition"),
        "material": tool_info.get("material"),
        "coating": tool_info.get("coating"),
        "tilt_angle_deg": round(tilt_deg, 6),
        "rotation_angle_deg": round(rotation_angle, 6),
        "centerline_slope": round(float(line_center[0]), 8),
        "centerline_intercept": round(float(line_center[1]), 4),
        "left_slope": round(float(line_left[0]), 8),
        "left_intercept": round(float(line_left[1]), 4),
        "right_slope": round(float(line_right[0]), 8),
        "right_intercept": round(float(line_right[1]), 4),
    }

    return tool_id, True, "Success", summary_row


def main():
    parser = argparse.ArgumentParser(description="Calculate tilt angle per tool with debug figures")
    parser.add_argument("--base-dir", default="/home/alifalah/Projects/DATA", help="Base DATA directory")
    parser.add_argument("--out-dir", default=None, help="Output directory for debug figures and tilt files")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    args = parser.parse_args()

    base_dir = args.base_dir
    src_dir = os.path.join(base_dir, "master_masks")
    out_dir = args.out_dir or os.path.join(base_dir, "tilt_calculation_per_tool")
    os.makedirs(out_dir, exist_ok=True)

    meta_df = pd.read_csv(os.path.join(base_dir, "tools_metadata.csv"))
    meta_df = meta_df.where(pd.notnull(meta_df), None)
    meta_dict = meta_df.set_index("tool_id").to_dict("index")

    mask_files = sorted(glob.glob(os.path.join(src_dir, "*_master_mask.png")))
    print(f"Calculating tilt angle for {len(mask_files)} tools into: {out_dir}")

    t0 = time.time()
    summary_rows = []
    completed = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_single_tool, p, out_dir, meta_dict): p
            for p in mask_files
        }

        for fut in as_completed(futures):
            tid, success, msg, row = fut.result()
            if success:
                completed += 1
                summary_rows.append(row)
            else:
                failed += 1
                print(f"ERROR on {tid}: {msg}")

            if (completed + failed) % 20 == 0 or (completed + failed) == len(mask_files):
                print(f"Progress: {completed + failed}/{len(mask_files)} completed ({time.time() - t0:.1f}s)")

    summary_df = pd.DataFrame(summary_rows).sort_values("tool_id")
    summary_csv = os.path.join(out_dir, "all_tools_tilt_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    print(f"\nFinished processing all {len(summary_df)} tools in {time.time() - t0:.1f}s.")
    print(f"Summary table saved to: {summary_csv}")
    print(f"Total debug figures: {len(glob.glob(os.path.join(out_dir, '*_tilt_angle_calculation.png')))}")


if __name__ == "__main__":
    main()
