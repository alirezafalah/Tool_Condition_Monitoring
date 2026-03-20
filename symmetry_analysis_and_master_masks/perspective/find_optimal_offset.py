"""
Optimal offset and multi-region comparison for tilted mask folders.

Designed to be called from the perspective GUI (Tab 3).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Iterable, Optional

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


VALID_OUTPUT_FORMATS = ("png", "svg", "pdf")
VALID_ANALYSIS_MODES = ("search_offset", "fixed_ranges")


@dataclass(frozen=True)
class OffsetAnalysisConfig:
    analysis_mode: str = "search_offset"

    # Search mode parameters.
    num_frames: int = 90
    offset_min: int = 176
    offset_max: int = 186
    search_num_regions: int = 2

    # Fixed mode fallback (2 ranges, inclusive frame indices).
    range_a_start: int = 0
    range_a_end: int = 89
    range_b_start: int = 182
    range_b_end: int = 271

    # Fixed mode preferred (supports many regions).
    region_ranges: tuple[tuple[int, int], ...] = ()

    # ROI height behavior.
    roi_height: int = 200
    use_metadata_roi_height: bool = True

    # Figure and output.
    output_formats: tuple[str, ...] = ("png",)
    title_font_size: int = 14
    axis_label_font_size: int = 12
    tick_font_size: int = 10
    legend_font_size: int = 10
    include_top_caption: bool = True
    stack_overlay_abs_diff: bool = False

    # Display degree labels in fixed-range figure.
    manual_legend_ranges: bool = False
    legend_a_start_deg: int = 0
    legend_a_end_deg: int = 90
    legend_b_start_deg: int = 180
    legend_b_end_deg: int = 270
    legend_ranges: tuple[tuple[int, int], ...] = ()


def _normalize_output_formats(values: Iterable[str]) -> tuple[str, ...]:
    normalized = []
    for value in values:
        fmt = str(value).strip().lower()
        if fmt in VALID_OUTPUT_FORMATS and fmt not in normalized:
            normalized.append(fmt)
    return tuple(normalized)


def _extract_frame_num(path: str) -> int:
    base = os.path.basename(path)
    stem, _ext = os.path.splitext(base)
    parts = stem.split("_")
    for part in reversed(parts):
        if part.isdigit():
            return int(part)
    match = re.search(r"(\d+)$", stem)
    return int(match.group(1)) if match else 0


def _read_binary_mask(path: str) -> Optional[np.ndarray]:
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    return np.where(mask >= 127, 255, 0).astype(np.uint8)


def get_tilted_mask_files(tool_dir: str) -> list[str]:
    if not os.path.isdir(tool_dir):
        raise FileNotFoundError(f"Tool directory not found: {tool_dir}")

    candidates = []
    for name in os.listdir(tool_dir):
        path = os.path.join(tool_dir, name)
        if not os.path.isfile(path):
            continue
        lname = name.lower()
        if lname.endswith(".png") or lname.endswith(".tif") or lname.endswith(".tiff"):
            candidates.append(path)

    if not candidates:
        raise FileNotFoundError(f"No mask image files found in {tool_dir}")

    candidates.sort(key=_extract_frame_num)
    return candidates


def _iter_inclusive(start_idx: int, end_idx: int) -> list[int]:
    if end_idx < start_idx:
        raise ValueError(f"Invalid range: start ({start_idx}) must be <= end ({end_idx}).")
    return list(range(start_idx, end_idx + 1))


def _range_to_str(rng: tuple[int, int]) -> str:
    return f"{rng[0]}-{rng[1]}"


def _extract_right_half_stats(
    mask: np.ndarray,
    global_roi_bottom: int,
    roi_height: int,
    center_col: int,
) -> Optional[tuple[int, int]]:
    roi_top = max(0, global_roi_bottom - roi_height)
    roi_bottom = global_roi_bottom + 1
    roi_mask = mask[roi_top:roi_bottom, :]

    white_pixels = np.where(roi_mask == 255)
    if len(white_pixels[1]) == 0:
        return None

    _left_col = int(np.min(white_pixels[1]))
    right_col = int(np.max(white_pixels[1]))

    width = int(roi_mask.shape[1])
    fixed_center_col = int(np.clip(center_col, 0, max(0, width - 1)))
    start_col = fixed_center_col + 1
    end_col = min(width, right_col + 1)
    right_half = roi_mask[:, start_col:end_col]
    right_count = int(np.sum(right_half == 255))

    half_width = max(1, end_col - start_col)
    half_area = max(1, roi_height * half_width)
    return right_count, half_area


def _find_global_roi_bottom_for_indices(mask_files: list[str], indices: Iterable[int]) -> int:
    global_bottom = 0
    for idx in indices:
        if idx < 0 or idx >= len(mask_files):
            continue
        mask = _read_binary_mask(mask_files[idx])
        if mask is None:
            continue
        white_pixels = np.where(mask == 255)
        if len(white_pixels[0]) > 0:
            global_bottom = max(global_bottom, int(np.max(white_pixels[0])))
    return global_bottom


def _find_global_roi_bottom_for_search(mask_files: list[str], num_frames: int, offset_min: int, offset_max: int) -> int:
    indices = set(range(min(num_frames, len(mask_files))))
    for offset in range(offset_min, offset_max + 1):
        for i in range(num_frames):
            indices.add(i + offset)
    return _find_global_roi_bottom_for_indices(mask_files, sorted(indices))


def _test_offset(
    mask_files: list[str],
    offset: int,
    global_roi_bottom: int,
    roi_height: int,
    num_frames: int,
    center_col: int,
) -> Optional[dict]:
    differences = []
    ratios = []

    for i in range(num_frames):
        frame1_idx = i
        frame2_idx = i + offset
        if frame2_idx >= len(mask_files):
            continue

        mask1 = _read_binary_mask(mask_files[frame1_idx])
        mask2 = _read_binary_mask(mask_files[frame2_idx])
        if mask1 is None or mask2 is None:
            continue

        stats1 = _extract_right_half_stats(mask1, global_roi_bottom, roi_height, center_col)
        stats2 = _extract_right_half_stats(mask2, global_roi_bottom, roi_height, center_col)
        if stats1 is None or stats2 is None:
            continue

        count1, _ = stats1
        count2, _ = stats2

        diff = abs(count1 - count2)
        total = count1 + count2
        ratio = (diff / total) if total > 0 else 0.0

        differences.append(diff)
        ratios.append(ratio)

    if not differences:
        return None

    return {
        "offset": int(offset),
        "mean_difference": float(np.mean(differences)),
        "std_difference": float(np.std(differences)),
        "mean_ratio": float(np.mean(ratios)),
        "std_ratio": float(np.std(ratios)),
        "max_difference": float(np.max(differences)),
        "max_ratio": float(np.max(ratios)),
        "num_valid_frames": int(len(differences)),
    }


def _find_optimal_offset(
    mask_files: list[str],
    global_roi_bottom: int,
    cfg: OffsetAnalysisConfig,
    roi_height: int,
    center_col: int,
    log_fn: Optional[Callable[[str], None]] = None,
) -> tuple[pd.DataFrame, int]:
    rows = []
    for offset in range(cfg.offset_min, cfg.offset_max + 1):
        if log_fn:
            log_fn(f"  Testing offset {offset} deg... ")
        result = _test_offset(mask_files, offset, global_roi_bottom, roi_height, cfg.num_frames, center_col)
        if result is None:
            if log_fn:
                log_fn("no valid data\n")
            continue
        rows.append(result)
        if log_fn:
            log_fn(f"mean diff={result['mean_difference']:.2f}, mean ratio={result['mean_ratio']:.6f}\n")

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No valid data was produced for the selected offset range.")

    optimal_idx = int(df["mean_ratio"].idxmin())
    optimal_offset = int(df.loc[optimal_idx, "offset"])
    return df, optimal_offset


def _resolve_fixed_regions(cfg: OffsetAnalysisConfig) -> list[tuple[int, int]]:
    if cfg.region_ranges:
        return [tuple(map(int, r)) for r in cfg.region_ranges]
    return [
        (int(cfg.range_a_start), int(cfg.range_a_end)),
        (int(cfg.range_b_start), int(cfg.range_b_end)),
    ]


def _compare_regions(
    mask_files: list[str],
    region_ranges: list[tuple[int, int]],
    global_roi_bottom: int,
    roi_height: int,
    center_col: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[list[int]]]:
    if len(region_ranges) < 2:
        raise ValueError("At least two regions are required for comparison.")

    region_indices = [_iter_inclusive(r[0], r[1]) for r in region_ranges]
    pair_count = min(len(idx_list) for idx_list in region_indices)
    if pair_count <= 0:
        raise ValueError("Fixed ranges produced no comparable frame pairs.")

    region_indices = [idx_list[:pair_count] for idx_list in region_indices]

    counts_rows = []
    pair_rows = []

    for pair_idx in range(pair_count):
        frame_indices = [idx_list[pair_idx] for idx_list in region_indices]
        if any(idx < 0 or idx >= len(mask_files) for idx in frame_indices):
            continue

        counts = []
        areas = []
        valid = True
        for frame_idx in frame_indices:
            mask = _read_binary_mask(mask_files[frame_idx])
            if mask is None:
                valid = False
                break
            stats = _extract_right_half_stats(mask, global_roi_bottom, roi_height, center_col)
            if stats is None:
                valid = False
                break
            count, area = stats
            counts.append(count)
            areas.append(area)

        if not valid:
            continue

        count_row = {"pair_idx": int(pair_idx)}
        for r_i, frame_idx in enumerate(frame_indices):
            count_row[f"frame_r{r_i+1}"] = int(frame_idx)
            count_row[f"count_r{r_i+1}"] = int(counts[r_i])
        counts_rows.append(count_row)

        for i in range(len(region_ranges)):
            for j in range(i + 1, len(region_ranges)):
                diff = abs(counts[i] - counts[j])
                total = counts[i] + counts[j]
                ratio = (diff / total) if total > 0 else 0.0
                avg_area = max(1.0, (areas[i] + areas[j]) / 2.0)
                normalized_diff = diff / avg_area
                pair_key = f"R{i+1}_vs_R{j+1}"
                pair_rows.append(
                    {
                        "pair_idx": int(pair_idx),
                        "pair_key": pair_key,
                        "region_i": f"R{i+1}",
                        "region_j": f"R{j+1}",
                        "region_i_range": _range_to_str(region_ranges[i]),
                        "region_j_range": _range_to_str(region_ranges[j]),
                        "frame_i": int(frame_indices[i]),
                        "frame_j": int(frame_indices[j]),
                        "count_i": int(counts[i]),
                        "count_j": int(counts[j]),
                        "abs_difference": int(diff),
                        "ratio": float(ratio),
                        "normalized_diff": float(normalized_diff),
                    }
                )

    counts_df = pd.DataFrame(counts_rows)
    pairwise_df = pd.DataFrame(pair_rows)

    if counts_df.empty or pairwise_df.empty:
        raise ValueError("No valid paired frames were produced for region comparison.")

    summary_df = (
        pairwise_df.groupby(["pair_key", "region_i_range", "region_j_range"], as_index=False)
        .agg(
            mean_abs_diff=("abs_difference", "mean"),
            std_abs_diff=("abs_difference", "std"),
            max_abs_diff=("abs_difference", "max"),
            mean_ratio=("ratio", "mean"),
            max_ratio=("ratio", "max"),
            pair_samples=("pair_idx", "count"),
        )
        .fillna(0.0)
    )

    return counts_df, pairwise_df, summary_df, region_indices


def _resolve_roi_height(tool_dir: str, cfg: OffsetAnalysisConfig, log_fn: Optional[Callable[[str], None]] = None) -> int:
    if cfg.use_metadata_roi_height:
        info_dir = os.path.join(tool_dir, "information")
        if os.path.isdir(info_dir):
            meta_files = sorted(
                f for f in os.listdir(info_dir)
                if f.endswith("_tilt_metadata.json") and os.path.isfile(os.path.join(info_dir, f))
            )
            for name in meta_files:
                path = os.path.join(info_dir, name)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    candidate = data.get("roi_height_px", data.get("roi_height"))
                    if candidate is not None:
                        roi_h = int(candidate)
                        if roi_h > 0:
                            if log_fn:
                                log_fn(f"ROI height loaded from metadata: {roi_h} px ({path})\n")
                            return roi_h
                except Exception:
                    continue
        if log_fn:
            log_fn("ROI height metadata not found; using manual ROI Height value.\n")

    return max(1, int(cfg.roi_height))


def _resolve_metadata_centerline(
    tool_dir: str,
    mask_files: list[str],
    log_fn: Optional[Callable[[str], None]] = None,
) -> int:
    """Resolve a constant centerline.

    Priority:
    1) Read centerline from existing tilt metadata.
    2) Fallback: build a master mask from available frames, estimate centerline,
       and persist it into tool metadata for future runs.
    """
    info_dir = os.path.join(tool_dir, "information")
    os.makedirs(info_dir, exist_ok=True)

    tool_folder_name = os.path.basename(os.path.normpath(tool_dir))
    match = re.search(r"(tool\d+)", tool_folder_name, re.IGNORECASE)
    tool_id = match.group(1).lower() if match else tool_folder_name

    meta_files = sorted(
        f for f in os.listdir(info_dir)
        if f.endswith("_tilt_metadata.json") and os.path.isfile(os.path.join(info_dir, f))
    )

    # 1) Metadata path first.
    for name in meta_files:
        path = os.path.join(info_dir, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            candidate = data.get("centerline_column_px", data.get("centerline_column"))
            if candidate is None:
                continue
            center_col = int(candidate)
            if center_col >= 0:
                if log_fn:
                    log_fn(f"Centerline column loaded from metadata: x={center_col} ({path})\n")
                return center_col
        except Exception:
            continue

    # 2) Fallback path: estimate from a master mask built from the current frames.
    if log_fn:
        log_fn("Centerline not found in metadata; building fallback master mask to estimate centerline.\n")

    if not mask_files:
        raise ValueError("Cannot estimate fallback centerline: no mask files available.")

    first_mask = None
    master_mask = None
    used_frames = 0
    for path in mask_files:
        mask = _read_binary_mask(path)
        if mask is None:
            continue
        if first_mask is None:
            first_mask = mask
            master_mask = np.zeros_like(mask, dtype=np.uint8)
        # Keep behavior close to previous tooling: skip all-white frames.
        if np.all(mask == 255):
            continue
        master_mask = cv2.bitwise_or(master_mask, mask)
        used_frames += 1

    if first_mask is None or master_mask is None or used_frames == 0:
        raise ValueError("Failed to build fallback master mask for centerline estimation.")

    ys = []
    left_x = []
    right_x = []
    h, w = master_mask.shape
    for y in range(h):
        white = np.where(master_mask[y, :] == 255)[0]
        if white.size > 0:
            ys.append(float(y))
            left_x.append(float(white[0]))
            right_x.append(float(white[-1]))

    if len(ys) < 2:
        white_cols = np.where(master_mask.any(axis=0))[0]
        if len(white_cols) >= 2:
            center_col = int(round((float(white_cols[0]) + float(white_cols[-1])) / 2.0))
        else:
            center_col = w // 2
    else:
        ys_arr = np.array(ys, dtype=np.float64)
        left_arr = np.array(left_x, dtype=np.float64)
        right_arr = np.array(right_x, dtype=np.float64)

        widths = right_arr - left_arr
        threshold = np.percentile(widths, 50.0)
        keep = widths >= threshold
        if int(np.sum(keep)) < min(20, len(widths)):
            take_n = min(20, len(widths))
            top_idx = np.argsort(widths)[-take_n:]
            keep = np.zeros_like(widths, dtype=bool)
            keep[top_idx] = True

        ys_fit = ys_arr[keep]
        left_fit = left_arr[keep]
        right_fit = right_arr[keep]

        m_left, b_left = np.polyfit(ys_fit, left_fit, 1)
        m_right, b_right = np.polyfit(ys_fit, right_fit, 1)
        m_center = (m_left + m_right) / 2.0
        b_center = (b_left + b_right) / 2.0

        y_mid = float((ys_arr.min() + ys_arr.max()) / 2.0)
        center_col = int(round(m_center * y_mid + b_center))

        white_cols = np.where(master_mask.any(axis=0))[0]
        if len(white_cols) >= 2:
            center_col = int(np.clip(center_col, int(white_cols[0]), int(white_cols[-1])))
        else:
            center_col = int(np.clip(center_col, 0, max(0, w - 1)))

    master_mask_path = os.path.join(info_dir, f"{tool_id}_MASTER_MASK_fallback.png")
    cv2.imwrite(master_mask_path, master_mask)

    target_meta_name = meta_files[0] if meta_files else f"{tool_id}_tilt_metadata.json"
    target_meta_path = os.path.join(info_dir, target_meta_name)
    try:
        with open(target_meta_path, "r", encoding="utf-8") as f:
            meta_data = json.load(f)
    except Exception:
        meta_data = {}

    meta_data["centerline_column_px"] = int(center_col)
    meta_data["centerline_source"] = "fallback_master_mask_fit"
    meta_data["centerline_fallback_master_mask_path"] = master_mask_path
    meta_data["centerline_fallback_used_frames"] = int(used_frames)
    meta_data["centerline_generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(target_meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_data, f, indent=2)

    if log_fn:
        log_fn(
            f"Fallback centerline estimated and saved: x={center_col} ({target_meta_path}); "
            f"master mask: {master_mask_path}\n"
        )

    return int(center_col)


def _save_plot_formats(
    fig,
    out_prefix: str,
    cfg: OffsetAnalysisConfig,
    log_fn: Optional[Callable[[str], None]] = None,
) -> list[str]:
    formats = _normalize_output_formats(cfg.output_formats)
    if not formats:
        raise ValueError("No valid output formats selected.")

    saved = []
    for fmt in formats:
        path = f"{out_prefix}.{fmt}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        saved.append(path)
        if log_fn:
            log_fn(f"Saved plot: {path}\n")

    plt.close(fig)
    return saved


def _plot_search(
    results_df: pd.DataFrame,
    tool_id: str,
    out_prefix: str,
    cfg: OffsetAnalysisConfig,
    optimal_offset: int,
    log_fn: Optional[Callable[[str], None]] = None,
) -> list[str]:
    plt.rcParams.update(
        {
            "font.size": cfg.axis_label_font_size,
            "axes.titlesize": cfg.title_font_size,
            "axes.labelsize": cfg.axis_label_font_size,
            "xtick.labelsize": cfg.tick_font_size,
            "ytick.labelsize": cfg.tick_font_size,
            "legend.fontsize": cfg.legend_font_size,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax1 = axes[0, 0]
    ax1.plot(results_df["offset"], results_df["mean_difference"], "o-", color="blue", linewidth=2, markersize=6)
    ax1.axvline(optimal_offset, color="red", linestyle="--", linewidth=2, label=f"Optimal: {optimal_offset} deg")
    ax1.set_title("Mean Difference vs Offset")
    ax1.set_xlabel("Offset")
    ax1.set_ylabel("Mean Pixel Count Difference")
    ax1.legend(fontsize=cfg.legend_font_size)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    ax2.plot(results_df["offset"], results_df["mean_ratio"], "o-", color="green", linewidth=2, markersize=6)
    ax2.axvline(optimal_offset, color="red", linestyle="--", linewidth=2, label=f"Optimal: {optimal_offset} deg")
    ax2.set_title("Mean Asymmetry Ratio vs Offset")
    ax2.set_xlabel("Offset")
    ax2.set_ylabel("Mean Ratio")
    ax2.legend(fontsize=cfg.legend_font_size)
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    ax3.plot(results_df["offset"], results_df["max_difference"], "o-", color="purple", linewidth=2, markersize=6)
    ax3.axvline(optimal_offset, color="red", linestyle="--", linewidth=2, label=f"Optimal: {optimal_offset} deg")
    ax3.set_title("Max Difference vs Offset")
    ax3.set_xlabel("Offset")
    ax3.set_ylabel("Max Difference")
    ax3.legend(fontsize=cfg.legend_font_size)
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    ax4.plot(results_df["offset"], results_df["std_difference"], "o-", color="orange", linewidth=2, markersize=6)
    ax4.axvline(optimal_offset, color="red", linestyle="--", linewidth=2, label=f"Optimal: {optimal_offset} deg")
    ax4.set_title("Std Dev of Difference vs Offset")
    ax4.set_xlabel("Offset")
    ax4.set_ylabel("Std Dev")
    ax4.legend(fontsize=cfg.legend_font_size)
    ax4.grid(True, alpha=0.3)

    for axis in (ax1, ax2, ax3, ax4):
        axis.tick_params(axis="both", labelsize=cfg.tick_font_size)

    if cfg.include_top_caption:
        fig.suptitle(
            f"{tool_id} Optimal Offset Search\n"
            f"Optimal Offset: {optimal_offset} deg (frames {optimal_offset}-{optimal_offset + cfg.num_frames - 1})",
            fontsize=cfg.title_font_size,
            fontweight="bold",
        )
        plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    else:
        plt.tight_layout()

    return _save_plot_formats(fig, out_prefix, cfg, log_fn=log_fn)


def _resolve_display_ranges(
    cfg: OffsetAnalysisConfig,
    pair_count: int,
    region_count: int,
) -> list[tuple[int, int]]:
    if cfg.manual_legend_ranges:
        if cfg.legend_ranges and len(cfg.legend_ranges) == region_count:
            return [tuple(map(int, r)) for r in cfg.legend_ranges]
        if region_count == 2:
            return [
                (int(cfg.legend_a_start_deg), int(cfg.legend_a_end_deg)),
                (int(cfg.legend_b_start_deg), int(cfg.legend_b_end_deg)),
            ]

    # Default canonical display labels for N regions.
    labels = []
    for i in range(region_count):
        start = 180 * i
        end = start + pair_count
        labels.append((start, end))
    return labels


def _plot_overlay_pixel_counts(
    counts_df: pd.DataFrame,
    tool_id: str,
    out_prefix: str,
    cfg: OffsetAnalysisConfig,
    region_ranges: list[tuple[int, int]],
    display_ranges: list[tuple[int, int]],
    log_fn: Optional[Callable[[str], None]] = None,
) -> list[str]:
    """Figure 1: Overlay of right-half pixel counts for all regions.

    X-axis = angle (degrees, from first display range), Y-axis = pixel count.
    Each region is a separate line using its legend label.
    """
    plt.rcParams.update(
        {
            "font.size": cfg.axis_label_font_size,
            "axes.titlesize": cfg.title_font_size,
            "axes.labelsize": cfg.axis_label_font_size,
            "xtick.labelsize": cfg.tick_font_size,
            "ytick.labelsize": cfg.tick_font_size,
            "legend.fontsize": cfg.legend_font_size,
        }
    )

    region_count = len(region_ranges)
    pair_count = int(len(counts_df))

    # X-axis: progression index within each range (0 .. pair_count-1).
    x_progression = np.arange(pair_count)

    fig, ax = plt.subplots(figsize=(12, 6))
    for r_i in range(region_count):
        y_col = f"count_r{r_i + 1}"
        label = f"P({display_ranges[r_i][0]}\u00b0\u2013{display_ranges[r_i][1]}\u00b0)"
        ax.plot(x_progression, counts_df[y_col], linewidth=1.5, label=label)

    ax.set_xlabel(r"$\theta$ progression (frame index within range)")
    ax.set_ylabel("White Pixel Count (Right Half)")
    ax.legend(fontsize=cfg.legend_font_size)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", labelsize=cfg.tick_font_size)

    if cfg.include_top_caption:
        ax.set_title(
            f"{tool_id} \u2014 Right-Half Pixel Count Overlay\n"
            f"Regions: {', '.join(f'{d[0]}\u00b0\u2013{d[1]}\u00b0' for d in display_ranges)}",
            fontsize=cfg.title_font_size,
            fontweight="bold",
        )

    plt.tight_layout()
    return _save_plot_formats(fig, f"{out_prefix}_overlay", cfg, log_fn=log_fn)


def _plot_abs_diff(
    pairwise_df: pd.DataFrame,
    tool_id: str,
    out_prefix: str,
    cfg: OffsetAnalysisConfig,
    display_ranges: list[tuple[int, int]],
    log_fn: Optional[Callable[[str], None]] = None,
) -> list[str]:
    """Figure 2: Absolute difference |P(θ) - P(θ+offset)| per angle.

    X-axis = angle (degrees), Y-axis = absolute difference.
    Mean absolute difference is shown in each legend entry.
    """
    plt.rcParams.update(
        {
            "font.size": cfg.axis_label_font_size,
            "axes.titlesize": cfg.title_font_size,
            "axes.labelsize": cfg.axis_label_font_size,
            "xtick.labelsize": cfg.tick_font_size,
            "ytick.labelsize": cfg.tick_font_size,
            "legend.fontsize": cfg.legend_font_size,
        }
    )

    pair_count = int(pairwise_df["pair_idx"].nunique())
    x_progression = np.arange(pair_count)

    fig, ax = plt.subplots(figsize=(12, 6))

    pair_groups = list(pairwise_df.groupby("pair_key", sort=True))
    # Keep all series in a red family and fill under each line with a lighter shade.
    red_shades = plt.cm.Reds(np.linspace(0.58, 0.9, max(1, len(pair_groups))))

    for idx, (pair_key, grp) in enumerate(pair_groups):
        grp = grp.sort_values("pair_idx")
        mean_val = float(grp["abs_difference"].mean())
        # Legend shows only pair identifier and mean value.
        parts = pair_key.replace("R", "").split("_vs_")
        if len(parts) == 2:
            ri, rj = int(parts[0]) - 1, int(parts[1]) - 1
            lbl = (
                f"R{ri+1} vs R{rj+1}  "
                f"(mean\u2009=\u2009{mean_val:.2f})"
            )
        else:
            lbl = f"{pair_key}  (mean={mean_val:.2f})"

        line_color = red_shades[idx]
        y_values = grp["abs_difference"].values
        ax.plot(
            x_progression,
            y_values,
            linewidth=1.8,
            color=line_color,
            label=lbl,
        )
        ax.fill_between(
            x_progression,
            y_values,
            0,
            color=line_color,
            alpha=0.18,
            linewidth=0,
        )

    ax.set_xlabel(r"$\theta$ progression (frame index within range)")
    ax.set_ylabel("Absolute Difference")
    ax.legend(fontsize=cfg.legend_font_size)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", labelsize=cfg.tick_font_size)

    if cfg.include_top_caption:
        overall_mean = float(pairwise_df["abs_difference"].mean())
        ax.set_title(
            f"{tool_id} \u2014 Absolute Difference per Angle\n"
            f"Overall Mean Abs Diff: {overall_mean:.2f}",
            fontsize=cfg.title_font_size,
            fontweight="bold",
        )

    plt.tight_layout()
    return _save_plot_formats(fig, f"{out_prefix}_abs_diff", cfg, log_fn=log_fn)


def _plot_overlay_abs_diff_stacked(
    counts_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    tool_id: str,
    out_prefix: str,
    cfg: OffsetAnalysisConfig,
    region_ranges: list[tuple[int, int]],
    display_ranges: list[tuple[int, int]],
    log_fn: Optional[Callable[[str], None]] = None,
) -> list[str]:
    """Combined figure: overlay on top, absolute difference below."""
    plt.rcParams.update(
        {
            "font.size": cfg.axis_label_font_size,
            "axes.titlesize": cfg.title_font_size,
            "axes.labelsize": cfg.axis_label_font_size,
            "xtick.labelsize": cfg.tick_font_size,
            "ytick.labelsize": cfg.tick_font_size,
            "legend.fontsize": cfg.legend_font_size,
        }
    )

    region_count = len(region_ranges)
    pair_count = int(len(counts_df))
    x_progression = np.arange(pair_count)

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    for r_i in range(region_count):
        y_col = f"count_r{r_i + 1}"
        label = f"P({display_ranges[r_i][0]}\u00b0\u2013{display_ranges[r_i][1]}\u00b0)"
        ax_top.plot(x_progression, counts_df[y_col], linewidth=1.5, label=label)

    ax_top.set_ylabel("White Pixel Count (Right Half)")
    ax_top.legend(fontsize=cfg.legend_font_size)
    ax_top.grid(True, alpha=0.3)
    ax_top.tick_params(axis="both", labelsize=cfg.tick_font_size)
    ax_top.set_title("Overlay Pixel Counts", fontsize=cfg.axis_label_font_size, fontweight="bold")

    pair_groups = list(pairwise_df.groupby("pair_key", sort=True))
    red_shades = plt.cm.Reds(np.linspace(0.58, 0.9, max(1, len(pair_groups))))

    for idx, (pair_key, grp) in enumerate(pair_groups):
        grp = grp.sort_values("pair_idx")
        mean_val = float(grp["abs_difference"].mean())
        parts = pair_key.replace("R", "").split("_vs_")
        if len(parts) == 2:
            ri, rj = int(parts[0]) - 1, int(parts[1]) - 1
            lbl = f"R{ri+1} vs R{rj+1}  (mean\u2009=\u2009{mean_val:.2f})"
        else:
            lbl = f"{pair_key}  (mean={mean_val:.2f})"

        line_color = red_shades[idx]
        y_values = grp["abs_difference"].values
        ax_bottom.plot(x_progression, y_values, linewidth=1.8, color=line_color, label=lbl)
        ax_bottom.fill_between(x_progression, y_values, 0, color=line_color, alpha=0.18, linewidth=0)

    ax_bottom.set_xlabel(r"$\theta$ progression (frame index within range)")
    ax_bottom.set_ylabel("Absolute Difference")
    ax_bottom.legend(fontsize=cfg.legend_font_size)
    ax_bottom.grid(True, alpha=0.3)
    ax_bottom.tick_params(axis="both", labelsize=cfg.tick_font_size)
    ax_bottom.set_title("Absolute Difference per Angle", fontsize=cfg.axis_label_font_size, fontweight="bold")

    if cfg.include_top_caption:
        overall_mean = float(pairwise_df["abs_difference"].mean())
        fig.suptitle(
            f"{tool_id} \u2014 Overlay and Absolute Difference\n"
            f"Overall Mean Abs Diff: {overall_mean:.2f}",
            fontsize=cfg.title_font_size,
            fontweight="bold",
        )
        plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    else:
        plt.tight_layout()

    return _save_plot_formats(fig, f"{out_prefix}_overlay_abs_diff_stacked", cfg, log_fn=log_fn)


def _plot_fixed_ranges(
    counts_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    tool_id: str,
    out_prefix: str,
    cfg: OffsetAnalysisConfig,
    region_ranges: list[tuple[int, int]],
    log_fn: Optional[Callable[[str], None]] = None,
) -> tuple[list[str], list[tuple[int, int]]]:
    region_count = len(region_ranges)
    pair_count = int(len(counts_df))
    display_ranges = _resolve_display_ranges(cfg, pair_count, region_count)

    saved: list[str] = []

    # Figure 1: Overlay of pixel counts for each region.
    saved.extend(
        _plot_overlay_pixel_counts(
            counts_df, tool_id, out_prefix, cfg, region_ranges, display_ranges, log_fn=log_fn,
        )
    )

    # Figure 2: Absolute difference per angle with mean in legend.
    saved.extend(
        _plot_abs_diff(
            pairwise_df, tool_id, out_prefix, cfg, display_ranges, log_fn=log_fn,
        )
    )

    if cfg.stack_overlay_abs_diff:
        saved.extend(
            _plot_overlay_abs_diff_stacked(
                counts_df,
                pairwise_df,
                tool_id,
                out_prefix,
                cfg,
                region_ranges,
                display_ranges,
                log_fn=log_fn,
            )
        )

    return saved, display_ranges


def _save_json(path: str, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _try_load_cached_search_result(
    metadata_path: str,
    sweep_csv_path: str,
    cfg: OffsetAnalysisConfig,
    center_col: int,
    log_fn: Optional[Callable[[str], None]] = None,
) -> tuple[Optional[int], Optional[pd.DataFrame], Optional[int]]:
    """Load previously computed search results when settings match.

    Returns: (optimal_offset, sweep_df, global_roi_bottom)
    """
    if not os.path.isfile(metadata_path):
        return None, None, None

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return None, None, None

    if meta.get("analysis_mode") != "search_offset":
        return None, None, None

    expected_range = f"{cfg.offset_min}-{cfg.offset_max}"
    if str(meta.get("offset_range_tested", "")) != expected_range:
        return None, None, None

    try:
        if int(meta.get("num_frames", -1)) != int(cfg.num_frames):
            return None, None, None
        if int(meta.get("centerline_column_px_used", -1)) != int(center_col):
            return None, None, None
        optimal_offset = int(meta.get("optimal_offset"))
    except Exception:
        return None, None, None

    if optimal_offset < cfg.offset_min or optimal_offset > cfg.offset_max:
        return None, None, None

    global_roi_bottom = None
    try:
        global_roi_bottom = int(meta.get("global_roi_bottom"))
    except Exception:
        global_roi_bottom = None

    sweep_df = None
    if os.path.isfile(sweep_csv_path):
        try:
            candidate_df = pd.read_csv(sweep_csv_path)
            if not candidate_df.empty:
                sweep_df = candidate_df
        except Exception:
            sweep_df = None

    if log_fn:
        log_fn(
            f"Reusing cached search result from metadata: optimal_offset={optimal_offset}, "
            f"offset_range={expected_range}, num_frames={cfg.num_frames}, centerline_x={center_col}.\n"
        )

    return optimal_offset, sweep_df, global_roi_bottom


def _update_tool_tilt_metadata(
    info_dir: str,
    updates: dict,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Optional[str]:
    if not os.path.isdir(info_dir):
        return None

    meta_files = sorted(
        f for f in os.listdir(info_dir)
        if f.endswith("_tilt_metadata.json") and os.path.isfile(os.path.join(info_dir, f))
    )
    if not meta_files:
        return None

    path = os.path.join(info_dir, meta_files[0])
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    data.update(updates)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    if log_fn:
        log_fn(f"Updated tool metadata: {path}\n")
    return path


def run_optimal_offset_analysis_for_tool(
    tool_dir: str,
    cfg: OffsetAnalysisConfig,
    log_fn: Optional[Callable[[str], None]] = None,
    symmetry_dir: Optional[str] = None,
) -> dict:
    if cfg.analysis_mode not in VALID_ANALYSIS_MODES:
        raise ValueError(f"Unsupported analysis mode: {cfg.analysis_mode}")

    tool_folder_name = os.path.basename(os.path.normpath(tool_dir))
    match = re.search(r"(tool\d+)", tool_folder_name, re.IGNORECASE)
    tool_id = match.group(1).lower() if match else tool_folder_name

    # Tilt metadata lives in tool_dir/information/ (created by Tab 1).
    info_dir = os.path.join(tool_dir, "information")
    # Analysis outputs go to symmetry_dir when provided.
    out_dir = symmetry_dir if symmetry_dir else info_dir
    os.makedirs(out_dir, exist_ok=True)

    mask_files = get_tilted_mask_files(tool_dir)
    roi_height = _resolve_roi_height(tool_dir, cfg, log_fn=log_fn)
    center_col = _resolve_metadata_centerline(tool_dir, mask_files, log_fn=log_fn)

    if log_fn:
        log_fn(f"Found {len(mask_files)} frames in {tool_folder_name}.\n")

    if cfg.analysis_mode == "search_offset":
        required_frames = cfg.offset_max + cfg.num_frames
        if len(mask_files) < required_frames:
            raise ValueError(
                f"Need at least {required_frames} frames for offsets up to {cfg.offset_max}, "
                f"but found {len(mask_files)}."
            )
        out_prefix = os.path.join(out_dir, tool_id)
        sweep_csv_path = f"{out_prefix}_search_sweep.csv"
        metadata_path = f"{out_prefix}_symmetry_metadata.json"

        used_cached_search = False
        optimal_offset, sweep_df, cached_global_roi_bottom = _try_load_cached_search_result(
            metadata_path,
            sweep_csv_path,
            cfg,
            center_col,
            log_fn=log_fn,
        )

        if optimal_offset is None:
            if log_fn:
                log_fn("Finding global ROI bottom...\n")
            global_roi_bottom = _find_global_roi_bottom_for_search(mask_files, cfg.num_frames, cfg.offset_min, cfg.offset_max)
            if log_fn:
                log_fn(f"Global ROI bottom: {global_roi_bottom}\n")
                log_fn(f"Testing offsets {cfg.offset_min}..{cfg.offset_max}...\n")

            sweep_df, optimal_offset = _find_optimal_offset(
                mask_files,
                global_roi_bottom,
                cfg,
                roi_height,
                center_col,
                log_fn=log_fn,
            )
            sweep_df.to_csv(sweep_csv_path, index=False)
        else:
            used_cached_search = True
            if cached_global_roi_bottom is None:
                if log_fn:
                    log_fn("Cached search found but global ROI bottom missing; recomputing ROI bottom for search indices.\n")
                global_roi_bottom = _find_global_roi_bottom_for_search(mask_files, cfg.num_frames, cfg.offset_min, cfg.offset_max)
            else:
                global_roi_bottom = int(cached_global_roi_bottom)
                if log_fn:
                    log_fn(f"Using cached global ROI bottom: {global_roi_bottom}\n")

        # Build N regions from the optimal offset (default 2, user may request more).
        region_ranges = []
        for r_i in range(max(2, int(cfg.search_num_regions))):
            start = r_i * optimal_offset
            end = start + cfg.num_frames - 1
            if end >= len(mask_files):
                if log_fn:
                    log_fn(f"Region {r_i+1} (frames {start}-{end}) exceeds available frames; using {max(2, int(cfg.search_num_regions)) - 1} regions.\n")
                break
            region_ranges.append((start, end))
        if len(region_ranges) < 2:
            region_ranges = [(0, cfg.num_frames - 1), (optimal_offset, optimal_offset + cfg.num_frames - 1)]
        counts_df, pairwise_df, summary_df, _region_indices = _compare_regions(
            mask_files,
            region_ranges,
            global_roi_bottom,
            roi_height,
            center_col,
        )
        abs_diff_csv_path = f"{out_prefix}_abs_diff_per_angle.csv"
        pairwise_df.to_csv(abs_diff_csv_path, index=False)

        plot_paths = []
        if sweep_df is not None and not sweep_df.empty:
            plot_paths.extend(_plot_search(sweep_df, tool_id, f"{out_prefix}_search_sweep", cfg, optimal_offset, log_fn=log_fn))

        # Also generate the two key comparison figures for the best offset found.
        pair_count = int(len(counts_df))
        display_ranges = _resolve_display_ranges(cfg, pair_count, len(region_ranges))
        plot_paths.extend(
            _plot_overlay_pixel_counts(
                counts_df, tool_id, out_prefix, cfg, region_ranges, display_ranges, log_fn=log_fn,
            )
        )
        plot_paths.extend(
            _plot_abs_diff(
                pairwise_df, tool_id, out_prefix, cfg, display_ranges, log_fn=log_fn,
            )
        )
        if cfg.stack_overlay_abs_diff:
            plot_paths.extend(
                _plot_overlay_abs_diff_stacked(
                    counts_df,
                    pairwise_df,
                    tool_id,
                    out_prefix,
                    cfg,
                    region_ranges,
                    display_ranges,
                    log_fn=log_fn,
                )
            )

        mean_abs_diff = float(pairwise_df["abs_difference"].mean())

        metadata = {
            "analysis_mode": "search_offset",
            "tool_id": tool_id,
            "num_frames": int(cfg.num_frames),
            "offset_range_tested": f"{cfg.offset_min}-{cfg.offset_max}",
            "optimal_offset": int(optimal_offset),
            "optimal_frame_range": f"{optimal_offset}-{optimal_offset + cfg.num_frames - 1}",
            "roi_height_px": int(roi_height),
            "centerline_column_px_used": int(center_col),
            "global_roi_bottom": int(global_roi_bottom),
            "use_metadata_roi_height": bool(cfg.use_metadata_roi_height),
            "used_cached_search": bool(used_cached_search),
            "mean_abs_diff": mean_abs_diff,
            "output_formats": list(_normalize_output_formats(cfg.output_formats)),
            "stack_overlay_abs_diff": bool(cfg.stack_overlay_abs_diff),
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        _save_json(metadata_path, metadata)

        _update_tool_tilt_metadata(
            info_dir,
            {
                "optimal_offset": int(optimal_offset),
                "optimal_offset_mean_abs_diff": float(mean_abs_diff),
            },
            log_fn=log_fn,
        )

        if log_fn:
            log_fn(f"Outputs saved to: {out_dir}\n")

        return {
            "analysis_mode": "search_offset",
            "tool_id": tool_id,
            "tool_folder_name": tool_folder_name,
            "output_dir": out_dir,
            "roi_height_px": int(roi_height),
            "centerline_column_px": int(center_col),
            "global_roi_bottom": int(global_roi_bottom),
            "optimal_offset": int(optimal_offset),
            "used_cached_search": bool(used_cached_search),
            "frame_range": f"{optimal_offset}-{optimal_offset + cfg.num_frames - 1}",
            "mean_abs_diff": mean_abs_diff,
            "metadata_path": metadata_path,
            "plot_paths": plot_paths,
        }

    # Fixed-range mode (supports N regions).
    region_ranges = _resolve_fixed_regions(cfg)
    if len(region_ranges) < 2:
        raise ValueError("Fixed mode requires at least two regions.")

    all_indices = []
    for rng in region_ranges:
        all_indices.extend(_iter_inclusive(rng[0], rng[1]))

    if any(idx < 0 or idx >= len(mask_files) for idx in all_indices):
        raise ValueError(
            f"One or more frame indices are out of bounds for this tool (available: 0-{len(mask_files)-1})."
        )

    if log_fn:
        log_fn("Finding global ROI bottom...\n")
    global_roi_bottom = _find_global_roi_bottom_for_indices(mask_files, sorted(set(all_indices)))
    if log_fn:
        log_fn(f"Global ROI bottom: {global_roi_bottom}\n")
        log_fn(f"Comparing {len(region_ranges)} regions: {', '.join(_range_to_str(r) for r in region_ranges)}\n")

    counts_df, pairwise_df, summary_df, _region_indices = _compare_regions(
        mask_files,
        region_ranges,
        global_roi_bottom,
        roi_height,
        center_col,
    )

    out_prefix = os.path.join(out_dir, tool_id)

    abs_diff_csv_path = f"{out_prefix}_abs_diff_per_angle.csv"
    pairwise_df.to_csv(abs_diff_csv_path, index=False)

    plot_paths, display_ranges = _plot_fixed_ranges(
        counts_df,
        pairwise_df,
        summary_df,
        tool_id,
        out_prefix,
        cfg,
        region_ranges,
        log_fn=log_fn,
    )

    mean_abs_diff = float(pairwise_df["abs_difference"].mean())

    metadata = {
        "analysis_mode": "fixed_ranges",
        "tool_id": tool_id,
        "roi_height_px": int(roi_height),
        "centerline_column_px_used": int(center_col),
        "global_roi_bottom": int(global_roi_bottom),
        "use_metadata_roi_height": bool(cfg.use_metadata_roi_height),
        "internal_regions": [_range_to_str(r) for r in region_ranges],
        "display_regions_deg": [f"{r[0]}-{r[1]}" for r in display_ranges],
        "manual_legend_ranges": bool(cfg.manual_legend_ranges),
        "pair_count": int(len(counts_df)),
        "mean_abs_diff": mean_abs_diff,
        "output_formats": list(_normalize_output_formats(cfg.output_formats)),
        "stack_overlay_abs_diff": bool(cfg.stack_overlay_abs_diff),
        "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    metadata_path = f"{out_prefix}_symmetry_metadata.json"
    _save_json(metadata_path, metadata)

    _update_tool_tilt_metadata(
        info_dir,
        {
            "fixed_regions": [_range_to_str(r) for r in region_ranges],
            "fixed_regions_mean_abs_diff": float(mean_abs_diff),
        },
        log_fn=log_fn,
    )

    if log_fn:
        log_fn(f"Outputs saved to: {out_dir}\n")

    return {
        "analysis_mode": "fixed_ranges",
        "tool_id": tool_id,
        "tool_folder_name": tool_folder_name,
        "output_dir": out_dir,
        "roi_height_px": int(roi_height),
        "centerline_column_px": int(center_col),
        "global_roi_bottom": int(global_roi_bottom),
        "region_count": int(len(region_ranges)),
        "internal_regions": [_range_to_str(r) for r in region_ranges],
        "display_regions": [f"{r[0]}-{r[1]}" for r in display_ranges],
        "pair_count": int(len(counts_df)),
        "mean_abs_diff": mean_abs_diff,
        "metadata_path": metadata_path,
        "plot_paths": plot_paths,
    }


# ============================================================================
# TAB 4: SYMMETRY SUMMARY BAR CHART
# ============================================================================

CONDITION_ORDER = {"new": 0, "used": 1, "deposit": 2, "fractured": 3, "broken": 4}
CONDITION_COLORS = {
    "new": "#2ca02c",
    "used": "#ff7f0e",
    "deposit": "#d62728",
    "fractured": "#d62728",
    "broken": "#7f0000",
}


def _condition_sort_key(cond: str) -> int:
    c = str(cond).strip().lower()
    for key, val in CONDITION_ORDER.items():
        if key in c:
            return val
    return 99


def _condition_color(cond: str) -> str:
    c = str(cond).strip().lower()
    for key, color in CONDITION_COLORS.items():
        if key in c:
            return color
    return "#888888"


def run_symmetry_summary(
    symmetry_root: str,
    tools_metadata_path: str,
    cfg: OffsetAnalysisConfig,
    log_fn: Optional[Callable[[str], None]] = None,
    include_tools: Optional[list[str]] = None,
    threshold_value: Optional[float] = None,
    show_threshold: bool = False,
) -> dict:
    """Generate a summary bar chart from Tab 3 results.

    Scans ``symmetry_root/<tool_id>/*_symmetry_metadata.json`` for
    ``mean_abs_diff``, joins with *tools_metadata.csv* for condition,
    creates a bar chart grouped by condition (new → fractured → broken).

    ``include_tools``: if provided, only tool IDs in this list are included.
    """
    if not os.path.isdir(symmetry_root):
        raise FileNotFoundError(f"Symmetry directory not found: {symmetry_root}")

    # Load tools metadata for condition info.
    tools_meta: dict[str, dict] = {}
    if os.path.isfile(tools_metadata_path):
        meta_df = pd.read_csv(tools_metadata_path)
        for _, row in meta_df.iterrows():
            tools_meta[str(row["tool_id"]).strip()] = row.to_dict()

    # Scan symmetry folders for results.
    results: list[dict] = []
    for entry in sorted(os.listdir(symmetry_root)):
        tool_sub = os.path.join(symmetry_root, entry)
        if not os.path.isdir(tool_sub):
            continue

        meta_files = [
            f for f in os.listdir(tool_sub) if f.endswith("_symmetry_metadata.json")
        ]
        if not meta_files:
            continue

        meta_path = os.path.join(tool_sub, meta_files[0])
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            continue

        tool_id = meta.get("tool_id", entry)
        mean_abs_diff = meta.get("mean_abs_diff")
        if mean_abs_diff is None:
            continue

        # Skip if not in the user-selected include list.
        if include_tools is not None and tool_id not in include_tools:
            continue

        tool_meta = tools_meta.get(tool_id, {})
        condition = str(tool_meta.get("condition", "unknown")).strip().lower()

        results.append(
            {
                "tool_id": tool_id,
                "condition": condition,
                "mean_abs_diff": float(mean_abs_diff),
                "analysis_mode": meta.get("analysis_mode", "unknown"),
            }
        )

    if not results:
        raise ValueError("No symmetry analysis results found in the symmetry folder.")

    df = pd.DataFrame(results)
    df["_sort"] = df["condition"].apply(_condition_sort_key)
    # Primary grouping by condition, then by mean difference (not by tool name).
    df = df.sort_values(["_sort", "mean_abs_diff", "tool_id"], ascending=[True, True, True]).drop(columns=["_sort"]).reset_index(drop=True)

    # Save summary CSV.
    csv_path = os.path.join(symmetry_root, "symmetry_summary.csv")
    df[["tool_id", "condition", "mean_abs_diff", "analysis_mode"]].to_csv(csv_path, index=False)
    if log_fn:
        log_fn(f"Saved summary CSV: {csv_path}\n")

    # ---------- Bar chart ----------
    plt.rcParams.update(
        {
            "font.size": cfg.axis_label_font_size,
            "axes.titlesize": cfg.title_font_size,
            "axes.labelsize": cfg.axis_label_font_size,
            "xtick.labelsize": cfg.tick_font_size,
            "ytick.labelsize": cfg.tick_font_size,
            "legend.fontsize": cfg.legend_font_size,
        }
    )

    colors = [_condition_color(c) for c in df["condition"]]

    fig, ax = plt.subplots(figsize=(max(10, len(df) * 0.8), 7))
    ax.bar(range(len(df)), df["mean_abs_diff"], color=colors, edgecolor="black", linewidth=0.5)

    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["tool_id"], rotation=45, ha="right", fontsize=cfg.tick_font_size)
    ax.set_ylabel("Mean Absolute Difference (Pixels)")
    ax.set_xlabel("Tool ID")
    ax.grid(axis="y", alpha=0.3)

    ax.set_title(
        "Mean Absolute Difference (0-90 vs 180-270, Right Half)",
        fontsize=cfg.title_font_size,
        fontweight="bold",
    )

    # Build legend from actually-present conditions.
    from matplotlib.patches import Patch as _Patch
    from matplotlib.lines import Line2D as _Line2D

    seen: list[str] = []
    for cond in df["condition"]:
        key = None
        for k in CONDITION_ORDER:
            if k in str(cond).lower():
                key = k
                break
        if key is None:
            key = str(cond).lower()
        if key not in seen:
            seen.append(key)

    legend_elements = [
        _Patch(facecolor=CONDITION_COLORS.get(k, "#888888"), edgecolor="black", label=k.capitalize())
        for k in seen
    ]

    # Add threshold line to legend if enabled
    if show_threshold and threshold_value is not None:
        threshold_line = _Line2D([0], [0], color='blue', linestyle='--', linewidth=2, label=f'Threshold (T = {threshold_value})')
        legend_elements.append(threshold_line)
        ax.axhline(threshold_value, color='blue', linestyle='--', linewidth=2)  # Draw the line

    ax.legend(handles=legend_elements, fontsize=cfg.legend_font_size)

    plt.tight_layout()
    out_prefix = os.path.join(symmetry_root, "summary_bar_chart")
    plot_paths = _save_plot_formats(fig, out_prefix, cfg, log_fn=log_fn)

    if log_fn:
        log_fn(f"Summary: {len(df)} tools processed.\n")

    return {
        "csv_path": csv_path,
        "plot_paths": plot_paths,
        "tool_count": len(df),
        "results": df.to_dict("records"),
    }


def run_custom_summary_graph(
    symmetry_root: str,
    tools_metadata_path: str,
    cfg: OffsetAnalysisConfig,
    labels_config: list[dict],
    threshold_value: Optional[float] = None,
    show_threshold: bool = False,
    custom_title: Optional[str] = None,
    show_title: bool = False,
    log_fn: Optional[Callable[[str], None]] = None,
) -> dict:
    """Generate a custom summary bar chart with user-defined labels and colors.

    Scans ``symmetry_root/<tool_id>/*_symmetry_metadata.json`` for ``mean_abs_diff``,
    joins with *tools_metadata.csv*, and creates a bar chart with custom labels.

    ``labels_config``: list of dicts with keys:
        - "name": label name
        - "color": hex color code
        - "tools": list of tool_ids to include in this label
    """
    if not os.path.isdir(symmetry_root):
        raise FileNotFoundError(f"Symmetry directory not found: {symmetry_root}")

    # Load tools metadata for condition info (optional, for context).
    tools_meta: dict[str, dict] = {}
    if os.path.isfile(tools_metadata_path):
        try:
            meta_df = pd.read_csv(tools_metadata_path)
            for _, row in meta_df.iterrows():
                tools_meta[str(row["tool_id"]).strip()] = row.to_dict()
        except Exception:
            pass

    # Scan symmetry folders for results.
    results: list[dict] = []
    for entry in sorted(os.listdir(symmetry_root)):
        tool_sub = os.path.join(symmetry_root, entry)
        if not os.path.isdir(tool_sub):
            continue

        meta_files = [
            f for f in os.listdir(tool_sub) if f.endswith("_symmetry_metadata.json")
        ]
        if not meta_files:
            continue

        meta_path = os.path.join(tool_sub, meta_files[0])
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            continue

        tool_id = meta.get("tool_id", entry)
        mean_abs_diff = meta.get("mean_abs_diff")
        if mean_abs_diff is None:
            continue

        results.append(
            {
                "tool_id": tool_id,
                "mean_abs_diff": float(mean_abs_diff),
            }
        )

    if not results:
        raise ValueError("No symmetry analysis results found in the symmetry folder.")

    # Assign tools to labels based on config
    df_rows = []
    for label_config in labels_config:
        label_name = label_config["name"]
        label_tools = label_config["tools"]

        for result in results:
            if result["tool_id"] in label_tools:
                df_rows.append({
                    "tool_id": result["tool_id"],
                    "label": label_name,
                    "mean_abs_diff": result["mean_abs_diff"],
                })

    if not df_rows:
        raise ValueError("No tools matched the provided label configuration.")

    df = pd.DataFrame(df_rows)
    # Sort by label order (insertion order), then by mean_abs_diff
    label_order = {label_config["name"]: idx for idx, label_config in enumerate(labels_config)}
    df["_sort"] = df["label"].map(label_order)
    df = df.sort_values(["_sort", "mean_abs_diff", "tool_id"], ascending=[True, True, True]).drop(
        columns=["_sort"]
    ).reset_index(drop=True)

    # Save summary CSV
    csv_path = os.path.join(symmetry_root, "custom_summary.csv")
    df[["tool_id", "label", "mean_abs_diff"]].to_csv(csv_path, index=False)
    if log_fn:
        log_fn(f"Saved summary CSV: {csv_path}\n")

    # Create color mapping
    color_map = {label_config["name"]: label_config["color"] for label_config in labels_config}

    # Bar chart
    plt.rcParams.update(
        {
            "font.size": cfg.axis_label_font_size,
            "axes.titlesize": cfg.title_font_size,
            "axes.labelsize": cfg.axis_label_font_size,
            "xtick.labelsize": cfg.tick_font_size,
            "ytick.labelsize": cfg.tick_font_size,
            "legend.fontsize": cfg.legend_font_size,
        }
    )

    colors = [color_map.get(label, "#888888") for label in df["label"]]

    fig, ax = plt.subplots(figsize=(max(10, len(df) * 0.8), 7))
    ax.bar(range(len(df)), df["mean_abs_diff"], color=colors, edgecolor="black", linewidth=0.5)

    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["tool_id"], rotation=45, ha="right", fontsize=cfg.tick_font_size)
    ax.set_ylabel("Mean Absolute Difference (Pixels)")
    ax.set_xlabel("Tool ID")
    ax.grid(axis="y", alpha=0.3)
    
    # Set title only if enabled
    if show_title:
        title_to_use = custom_title if custom_title else "Custom Summary: Mean Absolute Difference"
        ax.set_title(
            title_to_use,
            fontsize=cfg.title_font_size,
            fontweight="bold",
        )

    # Build legend from labels_config
    from matplotlib.patches import Patch as _Patch
    from matplotlib.lines import Line2D as _Line2D

    legend_elements = [
        _Patch(facecolor=label_config["color"], edgecolor="black", label=label_config["name"])
        for label_config in labels_config
    ]

    # Add threshold line to legend if enabled
    if show_threshold and threshold_value is not None:
        ax.axhline(threshold_value, color='blue', linestyle='--', linewidth=2, label=f'Threshold (T = {threshold_value})')
        # Create Line2D for legend instead of relying on axhline label (more reliable)
        threshold_line = _Line2D([0], [0], color='blue', linestyle='--', linewidth=2, label=f'Threshold (T = {threshold_value})')
        legend_elements.append(threshold_line)
        ax.axhline(threshold_value, color='blue', linestyle='--', linewidth=2)  # Draw the line without label

    ax.legend(handles=legend_elements, fontsize=cfg.legend_font_size)

    plt.tight_layout()
    out_prefix = os.path.join(symmetry_root, "custom_summary_bar_chart")
    plot_paths = _save_plot_formats(fig, out_prefix, cfg, log_fn=log_fn)

    if log_fn:
        log_fn(f"Summary: {len(df)} tools processed.\n")

    return {
        "csv_path": csv_path,
        "plot_paths": plot_paths,
        "tool_count": len(df),
        "results": df.to_dict("records"),
    }

