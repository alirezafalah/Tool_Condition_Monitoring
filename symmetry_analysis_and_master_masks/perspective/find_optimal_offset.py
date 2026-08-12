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
from matplotlib.ticker import MultipleLocator

# Keep the Qt backend when this module is embedded in gui_main; use the
# non-interactive backend for command-line/batch execution.
if "qt" not in matplotlib.get_backend().lower():
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


VALID_OUTPUT_FORMATS = ("png", "svg", "pdf")
VALID_ANALYSIS_MODES = ("search_offset", "fixed_ranges")
# Also acts as the search-cache version: v8 rejects anchor candidates that
# cannot accommodate every selected edge phase within the recording.
CENTERLINE_MODE = "shared_rotated_master_fitted_line_v8"


@dataclass(frozen=True)
class OffsetAnalysisConfig:
    analysis_mode: str = "search_offset"

    # Search mode parameters.
    num_frames: int = 90
    offset_min: int = 176
    offset_max: int = 186
    search_num_regions: int = 2
    # Candidate start-frame range for the first matching symmetry phase
    # (Region 2).  It calibrates the recording's constant angular speed; all
    # later regions are calculated from this one anchor, not independently
    # searched at potentially inconsistent frame rates.
    search_target_start_ranges: tuple[tuple[int, int], ...] = ()
    # When the folder is known to contain one complete 360-degree recording,
    # its image count fixes the angular sampling rate. In this mode only the
    # phase position consistent with that rate is accepted.
    full_rotation_locked: bool = False
    # Even when the precise angular overrun is unknown, a valid anchor must
    # leave enough frames for all edge phases in one complete turn.
    require_all_edge_phases_within_recording: bool = True

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
    dynamic_roi_enabled: bool = False
    dynamic_roi_height_factor: float = 0.45

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
    legend_ranges: tuple[tuple[float, float], ...] = ()

    # Optional smoothing before offset scoring / absolute differences.
    smoothing_enabled: bool = True
    smoothing_window: int = 20
    smoothing_strength: float = 1.0


def _smooth_values(values: Iterable[float], cfg: OffsetAnalysisConfig) -> np.ndarray:
    raw = np.asarray(list(values), dtype=np.float64)
    if not cfg.smoothing_enabled or len(raw) < 2:
        return raw.copy()
    window = max(1, min(int(cfg.smoothing_window), len(raw)))
    strength = float(np.clip(cfg.smoothing_strength, 0.0, 1.0))
    if window <= 1 or strength <= 0.0:
        return raw.copy()
    smoothed = pd.Series(raw).rolling(window=window, center=True, min_periods=1).mean().to_numpy()
    return raw * (1.0 - strength) + smoothed * strength


def _round_half_up(value: float) -> int:
    """Round positive frame counts so 30.5 frames becomes 31, not 30."""
    return int(np.floor(float(value) + 0.5))


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
    shared_centerline: tuple[float, float],
) -> Optional[tuple[int, int, int]]:
    """Count right-half ROI pixels using one rotated-master fitted line."""
    roi_top = max(0, global_roi_bottom - roi_height)
    roi_bottom = global_roi_bottom + 1
    roi_mask = mask[roi_top:roi_bottom, :]
    if not np.any(roi_mask == 255):
        return None

    width = int(mask.shape[1])
    center_slope, center_intercept = shared_centerline
    right_count = 0
    half_area = 0
    for local_y, global_y in enumerate(range(roi_top, roi_bottom)):
        center_x = int(np.clip(round(center_slope * global_y + center_intercept), 0, width - 1))
        first_right_col = center_x + 1
        right_count += int(np.count_nonzero(roi_mask[local_y, first_right_col:] == 255))
        half_area += max(0, width - first_right_col)

    roi_mid_y = (roi_top + roi_bottom - 1) / 2.0
    center_at_roi_mid = int(np.clip(round(center_slope * roi_mid_y + center_intercept), 0, width - 1))
    return right_count, max(1, half_area), center_at_roi_mid


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
    shared_centerline: tuple[float, float],
    cfg: OffsetAnalysisConfig,
    stats_cache: Optional[dict[int, Optional[tuple[int, int, int]]]] = None,
) -> Optional[dict]:
    counts1 = []
    counts2 = []

    for i in range(num_frames):
        frame1_idx = i
        frame2_idx = i + offset
        if frame2_idx >= len(mask_files):
            continue

        def get_stats(frame_idx):
            if stats_cache is not None and frame_idx in stats_cache:
                return stats_cache[frame_idx]
            mask = _read_binary_mask(mask_files[frame_idx])
            stats = None if mask is None else _extract_right_half_stats(
                mask, global_roi_bottom, roi_height, shared_centerline
            )
            if stats_cache is not None:
                stats_cache[frame_idx] = stats
            return stats

        stats1 = get_stats(frame1_idx)
        stats2 = get_stats(frame2_idx)
        if stats1 is None or stats2 is None:
            continue

        count1, *_unused1 = stats1
        count2, *_unused2 = stats2

        counts1.append(count1)
        counts2.append(count2)

    if not counts1:
        return None

    processed1 = _smooth_values(counts1, cfg)
    processed2 = _smooth_values(counts2, cfg)
    differences = np.abs(processed1 - processed2)
    totals = processed1 + processed2
    ratios = np.divide(differences, totals, out=np.zeros_like(differences), where=totals > 0)

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
    shared_centerline: tuple[float, float],
    log_fn: Optional[Callable[[str], None]] = None,
    stats_cache: Optional[dict[int, Optional[tuple[int, int, int]]]] = None,
) -> tuple[pd.DataFrame, int]:
    rows = []
    if stats_cache is None:
        stats_cache = {}
    for offset in range(cfg.offset_min, cfg.offset_max + 1):
        if log_fn:
            log_fn(f"  Testing offset {offset} deg... ")
        result = _test_offset(
            mask_files, offset, global_roi_bottom, roi_height, cfg.num_frames,
            shared_centerline, cfg, stats_cache=stats_cache,
        )
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


def _find_anchor_and_calculate_region_starts(
    mask_files: list[str],
    global_roi_bottom: int,
    cfg: OffsetAnalysisConfig,
    roi_height: int,
    shared_centerline: tuple[float, float],
    anchor_start_range: tuple[int, int],
    log_fn: Optional[Callable[[str], None]] = None,
) -> tuple[pd.DataFrame, list[int]]:
    """Search one anchor phase and derive all later phases at the same rate.

    For an E-edge tool, Region 2 starts one symmetry phase after Region 1.
    Once its frame start is found, each subsequent symmetry phase starts at an
    integer multiple of it. This preserves the constant angular velocity of a
    single recording instead of fitting a separate apparent velocity per edge.
    """
    start_min, start_max = map(int, anchor_start_range)
    if start_max < start_min:
        raise ValueError(f"Invalid Region 2 anchor search range: {start_min}-{start_max}.")
    if int(cfg.search_num_regions) < 2:
        raise ValueError("Search mode needs at least two symmetry regions.")
    if log_fn:
        log_fn(f"Searching Region 2 anchor start {start_min}..{start_max}...\n")

    anchor_cfg = OffsetAnalysisConfig(
        **{**cfg.__dict__, "offset_min": start_min, "offset_max": start_max}
    )
    sweep, anchor_start = _find_optimal_offset(
        mask_files,
        global_roi_bottom,
        anchor_cfg,
        roi_height,
        shared_centerline,
        log_fn=log_fn,
    )
    sweep = sweep.copy()
    sweep.insert(0, "target_region", 2)
    sweep.insert(1, "target_start_min", start_min)
    sweep.insert(2, "target_start_max", start_max)

    if cfg.full_rotation_locked:
        return sweep, [int(anchor_start)]

    region_starts = [int(round(anchor_start * multiplier)) for multiplier in range(1, int(cfg.search_num_regions))]
    if log_fn and len(region_starts) > 1:
        log_fn(
            "Constant-rotation model: Region 2 anchor="
            f"{anchor_start}; calculated later starts="
            + ", ".join(str(start) for start in region_starts[1:])
            + ".\n"
        )
    return sweep, region_starts


def _resolve_fixed_regions(cfg: OffsetAnalysisConfig) -> list[tuple[int, int]]:
    if cfg.region_ranges:
        return [tuple(map(int, r)) for r in cfg.region_ranges]
    return [
        (int(cfg.range_a_start), int(cfg.range_a_end)),
        (int(cfg.range_b_start), int(cfg.range_b_end)),
    ]


def _overall_dsi_percent(pairwise_df: pd.DataFrame) -> float:
    """DSI = mean absolute difference / pooled mean profile count * 100."""
    numerator = float(pairwise_df["abs_difference"].sum())
    denominator = float((pairwise_df["count_i"] + pairwise_df["count_j"]).sum())
    return 200.0 * numerator / max(1.0, denominator)


def _compare_regions(
    mask_files: list[str],
    region_ranges: list[tuple[int, int]],
    global_roi_bottom: int,
    roi_height: int,
    shared_centerline: tuple[float, float],
    cfg: OffsetAnalysisConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[list[int]]]:
    if len(region_ranges) < 2:
        raise ValueError("At least two regions are required for comparison.")

    region_indices = [_iter_inclusive(r[0], r[1]) for r in region_ranges]
    pair_count = min(len(idx_list) for idx_list in region_indices)
    if pair_count <= 0:
        raise ValueError("Fixed ranges produced no comparable frame pairs.")

    region_indices = [idx_list[:pair_count] for idx_list in region_indices]

    counts_rows = []
    for pair_idx in range(pair_count):
        frame_indices = [idx_list[pair_idx] for idx_list in region_indices]
        if any(idx < 0 or idx >= len(mask_files) for idx in frame_indices):
            continue

        counts = []
        areas = []
        centerlines = []
        valid = True
        for frame_idx in frame_indices:
            mask = _read_binary_mask(mask_files[frame_idx])
            if mask is None:
                valid = False
                break
            stats = _extract_right_half_stats(mask, global_roi_bottom, roi_height, shared_centerline)
            if stats is None:
                valid = False
                break
            count, area, frame_centerline = stats
            counts.append(count)
            areas.append(area)
            centerlines.append(frame_centerline)

        if not valid:
            continue

        count_row = {
            "pair_idx": int(pair_idx),
            "shared_centerline_slope": float(shared_centerline[0]),
            "shared_centerline_intercept": float(shared_centerline[1]),
        }
        for r_i, frame_idx in enumerate(frame_indices):
            count_row[f"frame_r{r_i+1}"] = int(frame_idx)
            count_row[f"centerline_x_r{r_i+1}"] = int(centerlines[r_i])
            count_row[f"count_r{r_i+1}"] = int(counts[r_i])
            count_row[f"_area_r{r_i+1}"] = int(areas[r_i])
        counts_rows.append(count_row)

    counts_df = pd.DataFrame(counts_rows)
    if counts_df.empty:
        raise ValueError("No valid paired frames were produced for region comparison.")

    for r_i in range(len(region_ranges)):
        raw_col = f"count_r{r_i+1}"
        counts_df[f"processed_count_r{r_i+1}"] = _smooth_values(counts_df[raw_col], cfg)

    pair_rows = []
    for _, row in counts_df.iterrows():
        for i in range(len(region_ranges)):
            for j in range(i + 1, len(region_ranges)):
                processed_i = float(row[f"processed_count_r{i+1}"])
                processed_j = float(row[f"processed_count_r{j+1}"])
                diff = abs(processed_i - processed_j)
                total = processed_i + processed_j
                avg_area = max(1.0, (float(row[f"_area_r{i+1}"]) + float(row[f"_area_r{j+1}"])) / 2.0)
                pair_rows.append({
                    "pair_idx": int(row["pair_idx"]),
                    "pair_key": f"R{i+1}_vs_R{j+1}",
                    "region_i": f"R{i+1}", "region_j": f"R{j+1}",
                    "region_i_range": _range_to_str(region_ranges[i]),
                    "region_j_range": _range_to_str(region_ranges[j]),
                    "frame_i": int(row[f"frame_r{i+1}"]), "frame_j": int(row[f"frame_r{j+1}"]),
                    "centerline_x_i": int(row[f"centerline_x_r{i+1}"]),
                    "centerline_x_j": int(row[f"centerline_x_r{j+1}"]),
                    "raw_count_i": int(row[f"count_r{i+1}"]), "raw_count_j": int(row[f"count_r{j+1}"]),
                    "count_i": processed_i, "count_j": processed_j,
                    "abs_difference": diff,
                    "ratio": (diff / total) if total > 0 else 0.0,
                    "dsi_percent_per_angle": (200.0 * diff / total) if total > 0 else 0.0,
                    "normalized_diff": diff / avg_area,
                })
    counts_df = counts_df.drop(columns=[c for c in counts_df.columns if c.startswith("_area_")])
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
    dsi_by_pair = pairwise_df.groupby("pair_key").apply(
        lambda group: 200.0 * group["abs_difference"].sum()
        / max(1.0, (group["count_i"] + group["count_j"]).sum()),
        include_groups=False,
    )
    summary_df["dsi_percent"] = summary_df["pair_key"].map(dsi_by_pair)

    return counts_df, pairwise_df, summary_df, region_indices


def _resolve_roi_height(
    tool_dir: str,
    cfg: OffsetAnalysisConfig,
    mask_files: list[str],
    log_fn: Optional[Callable[[str], None]] = None,
) -> tuple[int, Optional[int]]:
    if cfg.dynamic_roi_enabled:
        master_mask = None
        for path in mask_files:
            mask = _read_binary_mask(path)
            if mask is None:
                continue
            if master_mask is None:
                master_mask = np.zeros_like(mask, dtype=np.uint8)
            if mask.shape != master_mask.shape:
                raise ValueError("Dynamic ROI requires all tilted masks to have the same dimensions.")
            cv2.bitwise_or(master_mask, mask, dst=master_mask)
        if master_mask is None:
            raise ValueError("Cannot calculate dynamic ROI: no readable mask frames.")
        white_columns = np.flatnonzero(np.any(master_mask == 255, axis=0))
        if not white_columns.size:
            raise ValueError("Cannot calculate dynamic ROI: master mask is empty.")
        master_width = int(white_columns[-1] - white_columns[0] + 1)
        factor = max(0.01, float(cfg.dynamic_roi_height_factor))
        roi_height = max(1, int(round(master_width * factor)))
        roi_height = min(roi_height, int(master_mask.shape[0]))
        if log_fn:
            log_fn(
                f"Dynamic ROI: master-mask width W={master_width}px, factor={factor:.3f}, "
                f"shared height H={roi_height}px.\n"
            )
        return roi_height, master_width

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
                            return roi_h, None
                except Exception:
                    continue
        if log_fn:
            log_fn("ROI height metadata not found; using manual ROI Height value.\n")

    return max(1, int(cfg.roi_height)), None


def _fit_master_centerline(master_mask: np.ndarray) -> tuple[float, float]:
    """Fit the centre bisector from the widest rows of one master mask."""
    ys = []
    left_x = []
    right_x = []
    for y in range(master_mask.shape[0]):
        cols = np.flatnonzero(master_mask[y] == 255)
        if cols.size:
            ys.append(float(y))
            left_x.append(float(cols[0]))
            right_x.append(float(cols[-1]))
    if len(ys) < 2:
        raise ValueError("Master mask has insufficient non-empty rows for centerline fitting.")

    ys_arr = np.asarray(ys, dtype=np.float64)
    left_arr = np.asarray(left_x, dtype=np.float64)
    right_arr = np.asarray(right_x, dtype=np.float64)
    widths = right_arr - left_arr
    keep = widths >= np.percentile(widths, 50.0)
    if np.count_nonzero(keep) < min(20, len(widths)):
        keep = np.zeros_like(widths, dtype=bool)
        keep[np.argsort(widths)[-min(20, len(widths)):]] = True

    left_slope, left_intercept = np.polyfit(ys_arr[keep], left_arr[keep], 1)
    right_slope, right_intercept = np.polyfit(ys_arr[keep], right_arr[keep], 1)
    return (
        float((left_slope + right_slope) / 2.0),
        float((left_intercept + right_intercept) / 2.0),
    )


def _resolve_shared_master_centerline(
    tool_dir: str,
    mask_files: list[str],
    log_fn: Optional[Callable[[str], None]] = None,
) -> tuple[float, float]:
    """Load the rotated-master line, or reconstruct it once from tilted masks.

    Normal operation consumes the slope/intercept saved by ``TiltWorker``. The
    fallback only supports existing tilted-mask folders created before that
    metadata existed; it ORs the already-tilted frames and fits the same master
    mask method, never individual frames.
    """
    info_parent = os.path.dirname(tool_dir) if os.path.basename(os.path.normpath(tool_dir)).lower() == "tilted_masks" else tool_dir
    info_dir = os.path.join(info_parent, "information")
    os.makedirs(info_dir, exist_ok=True)
    meta_files = sorted(
        f for f in os.listdir(info_dir)
        if f.endswith("_tilt_metadata.json") and os.path.isfile(os.path.join(info_dir, f))
    )
    for name in meta_files:
        path = os.path.join(info_dir, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            slope = data.get("shared_centerline_slope")
            intercept = data.get("shared_centerline_intercept")
            if slope is None or intercept is None:
                continue
            line = (float(slope), float(intercept))
            if np.isfinite(line[0]) and np.isfinite(line[1]):
                if log_fn:
                    log_fn(f"Shared rotated-master centerline loaded from metadata: {path}\n")
                return line
        except Exception:
            continue

    if not mask_files:
        raise ValueError("Cannot build fallback rotated master mask: no tilted mask files are available.")
    master_mask = None
    used_frames = 0
    for path in mask_files:
        mask = _read_binary_mask(path)
        if mask is None or np.all(mask == 255):
            continue
        if master_mask is None:
            master_mask = np.zeros_like(mask, dtype=np.uint8)
        if mask.shape != master_mask.shape:
            raise ValueError("All tilted masks must have matching dimensions for master centerline fitting.")
        cv2.bitwise_or(master_mask, mask, dst=master_mask)
        used_frames += 1
    if master_mask is None or used_frames == 0:
        raise ValueError("Failed to build fallback rotated master mask for shared centerline.")

    line = _fit_master_centerline(master_mask)
    folder_name = os.path.basename(os.path.normpath(tool_dir))
    identity = os.path.basename(os.path.dirname(os.path.dirname(tool_dir))) if folder_name.lower() == "tilted_masks" else folder_name
    match = re.search(r"(tool\d+)", identity, re.IGNORECASE)
    tool_id = match.group(1).lower() if match else identity
    master_path = os.path.join(info_dir, f"{tool_id}_ROTATED_MASTER_MASK_fallback.png")
    cv2.imwrite(master_path, master_mask)
    meta_path = os.path.join(info_dir, meta_files[0] if meta_files else f"{tool_id}_tilt_metadata.json")
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = {}
    data.update(
        {
            "master_centerline_source": "fallback_fitted_boundaries_on_tilted_master_mask",
            "shared_centerline_slope": line[0],
            "shared_centerline_intercept": line[1],
            "shared_centerline_fallback_master_mask_path": master_path,
            "shared_centerline_fallback_used_frames": int(used_frames),
            "shared_centerline_generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    if log_fn:
        log_fn(f"Shared rotated-master centerline rebuilt and saved: {meta_path}\n")
    return line


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
    optimal_offsets: list[int],
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

    if "target_region" not in results_df:
        results_df = results_df.copy()
        results_df["target_region"] = 2

    groups = list(results_df.groupby("target_region", sort=True))
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(groups))))
    optimal_by_region = {
        int(region): int(optimal_offsets[index])
        for index, (region, _group) in enumerate(groups)
        if index < len(optimal_offsets)
    }

    def draw_metric(ax, column, color_name, title, ylabel):
        for color, (region, group) in zip(colors, groups):
            group = group.sort_values("offset")
            optimum = optimal_by_region.get(int(region))
            ax.plot(
                group["offset"], group[column], "o-", color=color, linewidth=2, markersize=5,
                label=f"Region {int(region)} search",
            )
            if optimum is not None:
                ax.axvline(
                    optimum, color=color, linestyle="--", linewidth=1.6,
                    label=f"Region {int(region)} optimum: {optimum}",
                )
        ax.set_title(title)
        ax.set_xlabel("Candidate target start frame")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=max(7, cfg.legend_font_size - 1))
        ax.grid(True, alpha=0.3)

    ax1 = axes[0, 0]
    draw_metric(ax1, "mean_difference", "blue", "Mean Difference vs Target Start", "Mean Pixel Count Difference")

    ax2 = axes[0, 1]
    draw_metric(ax2, "mean_ratio", "green", "Mean Asymmetry Ratio vs Target Start", "Mean Ratio")

    ax3 = axes[1, 0]
    draw_metric(ax3, "max_difference", "purple", "Maximum Difference vs Target Start", "Maximum Difference")

    ax4 = axes[1, 1]
    draw_metric(ax4, "std_difference", "orange", "Difference Standard Deviation vs Target Start", "Standard Deviation")

    for axis in (ax1, ax2, ax3, ax4):
        axis.tick_params(axis="both", labelsize=cfg.tick_font_size)

    if cfg.include_top_caption:
        fig.suptitle(
            f"{tool_id} Optimal Offset Search\n"
            "Optimal target starts: " + ", ".join(
                f"R{region}={start} (frames {start}-{start + cfg.num_frames - 1})"
                for region, start in optimal_by_region.items()
            ),
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
) -> list[tuple[float, float]]:
    if cfg.manual_legend_ranges:
        if cfg.legend_ranges and len(cfg.legend_ranges) == region_count:
            return [tuple(map(float, r)) for r in cfg.legend_ranges]
        if region_count == 2:
            return [
                (float(cfg.legend_a_start_deg), float(cfg.legend_a_end_deg)),
                (float(cfg.legend_b_start_deg), float(cfg.legend_b_end_deg)),
            ]

    # Default canonical display labels for N regions.
    labels = []
    for i in range(region_count):
        start = 180 * i
        end = start + pair_count
        labels.append((start, end))
    return labels


def _format_degree(value: float) -> str:
    value = float(value)
    return f"{value:.0f}" if np.isclose(value, round(value)) else f"{value:.1f}"


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
        y_col = f"processed_count_r{r_i + 1}" if cfg.smoothing_enabled else f"count_r{r_i + 1}"
        label = f"P({_format_degree(display_ranges[r_i][0])}\u00b0\u2013{_format_degree(display_ranges[r_i][1])}\u00b0)"
        ax.plot(x_progression, counts_df[y_col], linewidth=1.5, label=label)

    ax.set_xlabel(r"$\theta$ progression (frame index within range)")
    ax.set_ylabel("White Pixel Count (Right Half)")
    ax.legend(fontsize=cfg.legend_font_size)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", labelsize=cfg.tick_font_size)

    if cfg.include_top_caption:
        ax.set_title(
            f"{tool_id} \u2014 Right-Half Pixel Count Overlay\n"
            f"Regions: {', '.join(f'{_format_degree(d[0])}\u00b0\u2013{_format_degree(d[1])}\u00b0' for d in display_ranges)}",
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
                f"{_format_degree(display_ranges[ri][0])}°-{_format_degree(display_ranges[ri][1])}° vs "
                f"{_format_degree(display_ranges[rj][0])}°-{_format_degree(display_ranges[rj][1])}°  "
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


def _plot_dsi(
    pairwise_df: pd.DataFrame,
    tool_id: str,
    out_prefix: str,
    cfg: OffsetAnalysisConfig,
    display_ranges: list[tuple[int, int]],
    log_fn: Optional[Callable[[str], None]] = None,
) -> list[str]:
    """Standalone per-angle normalized deviation with the tool-level DSI."""
    pair_count = int(pairwise_df["pair_idx"].nunique())
    x_progression = np.arange(pair_count)
    pair_groups = list(pairwise_df.groupby("pair_key", sort=True))
    blue_shades = plt.cm.Blues(np.linspace(0.58, 0.9, max(1, len(pair_groups))))
    fig, ax = plt.subplots(figsize=(12, 6))
    tool_dsi_values = []

    for idx, (pair_key, grp) in enumerate(pair_groups):
        grp = grp.sort_values("pair_idx")
        parts = pair_key.replace("R", "").split("_vs_")
        if len(parts) == 2:
            ri, rj = int(parts[0]) - 1, int(parts[1]) - 1
            pair_label = (
                f"{_format_degree(display_ranges[ri][0])}°-{_format_degree(display_ranges[ri][1])}° vs "
                f"{_format_degree(display_ranges[rj][0])}°-{_format_degree(display_ranges[rj][1])}°"
            )
        else:
            pair_label = pair_key.replace("_", " ")
        tool_dsi = _overall_dsi_percent(grp)
        tool_dsi_values.append(tool_dsi)
        values = grp["dsi_percent_per_angle"].to_numpy()
        label = f"{pair_label} (mean DSI={tool_dsi:.3f}%)"
        ax.plot(x_progression, values, linewidth=1.8, color=blue_shades[idx], label=label)
        ax.fill_between(x_progression, values, 0, color=blue_shades[idx], alpha=0.18, linewidth=0)

    ax.set_xlabel(r"$\theta$ progression (frame index within range)")
    ax.set_ylabel(r"Per-angle $\mathrm{DSI}_i$ (%)")
    ax.legend(fontsize=cfg.legend_font_size)
    ax.grid(True, alpha=0.3)
    if cfg.include_top_caption:
        caption = (
            f"{tool_dsi_values[0]:.3f}%" if len(tool_dsi_values) == 1
            else ", ".join(f"{value:.3f}%" for value in tool_dsi_values)
        )
        ax.set_title(
            f"{tool_id} - Dimensionless Symmetry Index per Angle\n"
            f"mean DSI over {pair_count} aligned angles: {caption}",
            fontsize=cfg.title_font_size,
            fontweight="bold",
        )
    plt.tight_layout()
    return _save_plot_formats(fig, f"{out_prefix}_dsi", cfg, log_fn=log_fn)


def _plot_overlay_dsi_stacked(
    counts_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    tool_id: str,
    out_prefix: str,
    cfg: OffsetAnalysisConfig,
    region_ranges: list[tuple[int, int]],
    display_ranges: list[tuple[int, int]],
    log_fn: Optional[Callable[[str], None]] = None,
) -> list[str]:
    """Combined figure: phase-aligned pixel-count overlay on top, DSI below."""
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

    pair_count = int(len(counts_df))
    x_progression = np.arange(pair_count)

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    for r_i in range(len(region_ranges)):
        y_col = f"processed_count_r{r_i + 1}" if cfg.smoothing_enabled else f"count_r{r_i + 1}"
        label = f"P({_format_degree(display_ranges[r_i][0])}°-{_format_degree(display_ranges[r_i][1])}°)"
        ax_top.plot(x_progression, counts_df[y_col], linewidth=1.5, label=label)

    ax_top.set_ylabel("White Pixel Count (Right Half)")
    ax_top.legend(fontsize=cfg.legend_font_size)
    ax_top.grid(True, alpha=0.3)
    ax_top.tick_params(axis="both", labelsize=cfg.tick_font_size)
    ax_top.set_title("Phase-Aligned Pixel-Count Overlay", fontsize=cfg.axis_label_font_size, fontweight="bold")

    pair_groups = list(pairwise_df.groupby("pair_key", sort=True))
    blue_shades = plt.cm.Blues(np.linspace(0.58, 0.9, max(1, len(pair_groups))))
    overall_dsi_labels = []

    for idx, (pair_key, grp) in enumerate(pair_groups):
        grp = grp.sort_values("pair_idx")
        parts = pair_key.replace("R", "").split("_vs_")
        if len(parts) == 2:
            ri, rj = int(parts[0]) - 1, int(parts[1]) - 1
            pair_label = (
                f"{_format_degree(display_ranges[ri][0])}°-{_format_degree(display_ranges[ri][1])}° vs "
                f"{_format_degree(display_ranges[rj][0])}°-{_format_degree(display_ranges[rj][1])}°"
            )
        else:
            pair_label = pair_key.replace("_", " ")

        total_profile = float((grp["count_i"] + grp["count_j"]).sum())
        overall_dsi = 200.0 * float(grp["abs_difference"].sum()) / max(1.0, total_profile)
        overall_dsi_labels.append(f"{pair_label}: {overall_dsi:.3f}%")
        dsi_values = grp["dsi_percent_per_angle"].values
        dsi_label = f"{pair_label} (mean DSI={overall_dsi:.3f}%)"
        ax_bottom.plot(x_progression, dsi_values, linewidth=1.8, color=blue_shades[idx], label=dsi_label)
        ax_bottom.fill_between(
            x_progression, dsi_values, 0, color=blue_shades[idx], alpha=0.18, linewidth=0
        )

    ax_bottom.set_xlabel(r"$\theta$ progression (frame index within range)")
    ax_bottom.set_ylabel(r"Per-angle $\mathrm{DSI}_i$ (%)")
    ax_bottom.legend(fontsize=cfg.legend_font_size)
    ax_bottom.grid(True, alpha=0.3)
    ax_bottom.tick_params(axis="both", labelsize=cfg.tick_font_size)
    ax_bottom.set_title(
        # r"Normalized Deviation: $\mathrm{DSI}_i=200|P_1-P_2|/(P_1+P_2)$",
        r"$\mathrm{DSI}_i$",
        fontsize=cfg.axis_label_font_size,
        fontweight="bold",
    )

    if cfg.include_top_caption:
        dsi_caption = (
            overall_dsi_labels[0].split(": ", 1)[1]
            if len(overall_dsi_labels) == 1 else ", ".join(overall_dsi_labels)
        )
        fig.suptitle(
            f"{tool_id} \u2014 Pixel-Count Overlay and Dimensionless Symmetry Index\n"
            f"mean DSI over {pair_count} aligned angles: {dsi_caption}",
            fontsize=cfg.title_font_size,
            fontweight="bold",
        )
        plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    else:
        plt.tight_layout()

    return _save_plot_formats(fig, f"{out_prefix}_overlay_dsi_stacked", cfg, log_fn=log_fn)


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
    saved.extend(
        _plot_dsi(
            pairwise_df, tool_id, out_prefix, cfg, display_ranges, log_fn=log_fn,
        )
    )

    if cfg.stack_overlay_abs_diff:
        saved.extend(
            _plot_overlay_dsi_stacked(
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
    resolved_roi_height: int,
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

    # Invalidate old caches produced with previous centerline behavior.
    if str(meta.get("centerline_mode", "")) != CENTERLINE_MODE:
        return None, None, None

    expected_range = f"{cfg.offset_min}-{cfg.offset_max}"
    if str(meta.get("offset_range_tested", "")) != expected_range:
        return None, None, None

    try:
        if int(meta.get("num_frames", -1)) != int(cfg.num_frames):
            return None, None, None
        if int(meta.get("edge_count", 2)) != int(cfg.search_num_regions):
            return None, None, None
        if bool(meta.get("full_rotation_locked", False)) != bool(cfg.full_rotation_locked):
            return None, None, None
        if bool(meta.get("require_all_edge_phases_within_recording", False)) != bool(
            cfg.require_all_edge_phases_within_recording
        ):
            return None, None, None
        if int(meta.get("roi_height_px", -1)) != int(resolved_roi_height):
            return None, None, None
        if bool(meta.get("dynamic_roi_enabled", False)) != bool(cfg.dynamic_roi_enabled):
            return None, None, None
        if not np.isclose(
            float(meta.get("dynamic_roi_height_factor", 0.45)),
            float(cfg.dynamic_roi_height_factor),
        ):
            return None, None, None
        if bool(meta.get("smoothing_enabled", False)) != bool(cfg.smoothing_enabled):
            return None, None, None
        if int(meta.get("smoothing_window", 1)) != int(cfg.smoothing_window):
            return None, None, None
        if not np.isclose(float(meta.get("smoothing_strength", 0.0)), float(cfg.smoothing_strength)):
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
            f"offset_range={expected_range}, num_frames={cfg.num_frames}; "
            "centerline uses the shared fitted line from the rotated master mask.\n"
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
    identity_name = tool_folder_name
    if tool_folder_name.lower() == "tilted_masks":
        analysis_parent = os.path.dirname(os.path.normpath(tool_dir))
        masks_parent = os.path.dirname(analysis_parent)
        identity_name = os.path.basename(masks_parent).removesuffix("_final_masks")
    match = re.search(r"(tool\d+)", identity_name, re.IGNORECASE)
    tool_id = match.group(1).lower() if match else identity_name

    # Keep metadata beside tilted_masks when the GUI passes that folder directly.
    info_parent = os.path.dirname(tool_dir) if tool_folder_name.lower() == "tilted_masks" else tool_dir
    info_dir = os.path.join(info_parent, "information")
    # Analysis outputs go to symmetry_dir when provided.
    out_dir = symmetry_dir if symmetry_dir else info_dir
    os.makedirs(out_dir, exist_ok=True)

    mask_files = get_tilted_mask_files(tool_dir)
    roi_height, dynamic_roi_master_width = _resolve_roi_height(
        tool_dir, cfg, mask_files, log_fn=log_fn
    )
    # One line is fitted to the rotated master mask and reused for every frame.
    # The ROI only limits the vertical rows counted on the right side of it.
    shared_centerline = _resolve_shared_master_centerline(tool_dir, mask_files, log_fn=log_fn)

    if log_fn:
        log_fn(f"Found {len(mask_files)} frames in {tool_folder_name}.\n")
        log_fn(
            "Shared centerline: fitted to the rotated master mask and reused for all frames "
            f"(slope={shared_centerline[0]:.6f}, intercept={shared_centerline[1]:.3f}); "
            "ROI limits pixel counting only.\n"
        )

    if cfg.analysis_mode == "search_offset":
        region_count = int(cfg.search_num_regions)
        if region_count < 2:
            raise ValueError("Search mode needs at least two symmetry regions.")
        full_rotation_locked = bool(cfg.full_rotation_locked)
        require_all_edge_phases = bool(cfg.require_all_edge_phases_within_recording)
        expected_phase_frames = len(mask_files) / float(region_count)
        expected_comparison_frames = expected_phase_frames / 2.0
        search_num_frames = max(1, _round_half_up(expected_comparison_frames)) if full_rotation_locked else int(cfg.num_frames)
        target_start_ranges = [
            (int(start), int(end)) for start, end in cfg.search_target_start_ranges
        ] or [(int(cfg.offset_min), int(cfg.offset_max))]
        anchor_start_range = target_start_ranges[0]
        if full_rotation_locked:
            # A known complete turn fixes the phase location. Do not let a
            # visually tempting but geometrically impossible offset redefine
            # the speed of rotation.
            expected_anchor = max(1, int(round(expected_phase_frames)))
            anchor_start_range = (expected_anchor, expected_anchor)
        elif require_all_edge_phases:
            # The recording can contain a few extra degrees, so we do not
            # force N/E exactly.  But an anchor of A frames for E edges must
            # fit E complete phase pitches inside N recorded frames.  Otherwise
            # it represents a false local pixel match, not the rotation rate.
            max_feasible_anchor = len(mask_files) // region_count
            if anchor_start_range[1] > max_feasible_anchor and log_fn:
                log_fn(
                    f"Rejecting Region 2 candidates above {max_feasible_anchor}: "
                    f"{region_count} edges × anchor must fit within {len(mask_files)} frames.\n"
                )
            anchor_start_range = (
                anchor_start_range[0],
                min(anchor_start_range[1], max_feasible_anchor),
            )
        if anchor_start_range[1] < anchor_start_range[0]:
            raise ValueError(
                "No candidate anchor can fit every selected edge phase within the available frames. "
                "Reduce the edge count, select the correct mask folder, or widen the captured rotation."
            )
        required_frames = anchor_start_range[1] + search_num_frames
        if len(mask_files) < required_frames:
            raise ValueError(
                f"Need at least {required_frames} frames for the Region 2 anchor search, "
                f"but found {len(mask_files)}."
            )
        out_prefix = os.path.join(out_dir, tool_id)
        sweep_csv_path = f"{out_prefix}_search_sweep.csv"
        metadata_path = f"{out_prefix}_symmetry_metadata.json"

        # The legacy two-region call can retain its single-offset cache.
        used_cached_search = False
        optimal_starts: Optional[list[int]] = None
        sweep_df = None
        cached_global_roi_bottom = None
        if len(target_start_ranges) == 1 and not cfg.search_target_start_ranges:
            cached_offset, sweep_df, cached_global_roi_bottom = _try_load_cached_search_result(
                metadata_path,
                sweep_csv_path,
                cfg,
                roi_height,
                log_fn=log_fn,
            )
            if cached_offset is not None:
                optimal_starts = [
                    int(round(cached_offset * multiplier))
                    for multiplier in range(1, int(cfg.search_num_regions))
                ]
                used_cached_search = True

        if optimal_starts is None:
            if log_fn:
                log_fn("Finding global ROI bottom...\n")
            search_indices = set(range(min(search_num_frames, len(mask_files))))
            for candidate_start in range(anchor_start_range[0], anchor_start_range[1] + 1):
                search_indices.update(range(candidate_start, candidate_start + search_num_frames))
            global_roi_bottom = _find_global_roi_bottom_for_indices(mask_files, sorted(search_indices))
            if log_fn:
                log_fn(f"Global ROI bottom: {global_roi_bottom}\n")
                if full_rotation_locked:
                    log_fn(
                        f"Full-rotation constraint: {len(mask_files)} frames / {region_count} edges → "
                        f"Region 2 phase fixed at frame {anchor_start_range[0]}; "
                        "checking its pixel similarity only.\n"
                    )
                else:
                    log_fn(
                        f"Testing Region 2 anchor range: {anchor_start_range[0]}-{anchor_start_range[1]}; "
                        "all later regions will be calculated at the same rotation rate. "
                        f"Candidates must satisfy {region_count} × anchor ≤ {len(mask_files)} frames.\n"
                    )

            search_cfg = OffsetAnalysisConfig(
                **{**cfg.__dict__, "num_frames": search_num_frames}
            )
            sweep_df, optimal_starts = _find_anchor_and_calculate_region_starts(
                mask_files,
                global_roi_bottom,
                search_cfg,
                roi_height,
                shared_centerline,
                anchor_start_range,
                log_fn=log_fn,
            )
            sweep_df.to_csv(sweep_csv_path, index=False)
        else:
            if cached_global_roi_bottom is None:
                if log_fn:
                    log_fn("Cached search found but global ROI bottom missing; recomputing ROI bottom for search indices.\n")
                global_roi_bottom = _find_global_roi_bottom_for_search(mask_files, cfg.num_frames, cfg.offset_min, cfg.offset_max)
            else:
                global_roi_bottom = int(cached_global_roi_bottom)
                if log_fn:
                    log_fn(f"Using cached global ROI bottom: {global_roi_bottom}\n")

        anchor_start = int(optimal_starts[0])
        if anchor_start <= 0:
            raise ValueError("The selected Region 2 anchor must be greater than frame 0.")
        phase_step_degrees = 360.0 / region_count
        comparison_span_degrees = phase_step_degrees / 2.0
        if full_rotation_locked:
            degrees_per_frame = 360.0 / len(mask_files)
            calibrated_num_frames = search_num_frames
            # Use the same fractional full-rotation grid for every phase;
            # individual integer starts differ by rounding only, not speed.
            optimal_starts = [
                int(round(expected_phase_frames * multiplier))
                for multiplier in range(1, region_count)
            ]
            rotation_model = "full_360_degree_frame_grid"
        else:
            degrees_per_frame = phase_step_degrees / anchor_start
            calibrated_num_frames = max(1, _round_half_up(comparison_span_degrees / degrees_per_frame))
            rotation_model = "constant_rate_calibrated_from_region_2_anchor"
        # Region 1 is the base range. Every later region comes from the same
        # calibrated frame rate: start_k = k * anchor_start.
        region_ranges = [(0, calibrated_num_frames - 1)] + [
            (start, start + calibrated_num_frames - 1) for start in optimal_starts
        ]
        all_region_indices = [idx for rng in region_ranges for idx in _iter_inclusive(*rng)]
        if any(idx < 0 or idx >= len(mask_files) for idx in all_region_indices):
            raise ValueError(
                "The constant-rotation calibration produces a region outside the available frames: "
                f"anchor={anchor_start}, calibrated frames/region={calibrated_num_frames}, "
                f"available=0-{len(mask_files) - 1}."
            )
        # Search scoring only needs the anchor candidates. The final comparison
        # needs a common ROI bottom covering every calculated region.
        global_roi_bottom = _find_global_roi_bottom_for_indices(mask_files, all_region_indices)
        if log_fn:
            log_fn(
                f"Rotation model: {rotation_model}; {degrees_per_frame:.8f}°/frame; "
                f"{calibrated_num_frames} frames per {comparison_span_degrees:g}° comparison span.\n"
            )
        counts_df, pairwise_df, summary_df, _region_indices = _compare_regions(
            mask_files,
            region_ranges,
            global_roi_bottom,
            roi_height,
            shared_centerline,
            cfg,
        )
        abs_diff_csv_path = f"{out_prefix}_abs_diff_per_angle.csv"
        pairwise_df.to_csv(abs_diff_csv_path, index=False)
        pixel_counts_csv_path = f"{out_prefix}_right_side_pixel_counts.csv"
        counts_df.to_csv(pixel_counts_csv_path, index=False)
        summary_csv_path = f"{out_prefix}_comparison_summary.csv"
        summary_df.to_csv(summary_csv_path, index=False)

        plot_paths = []
        if sweep_df is not None and not sweep_df.empty:
            plot_paths.extend(
                _plot_search(
                    sweep_df,
                    tool_id,
                    f"{out_prefix}_search_sweep",
                    cfg,
                    optimal_starts,
                    log_fn=log_fn,
                )
            )

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
        plot_paths.extend(
            _plot_dsi(
                pairwise_df, tool_id, out_prefix, cfg, display_ranges, log_fn=log_fn,
            )
        )
        if cfg.stack_overlay_abs_diff:
            plot_paths.extend(
                _plot_overlay_dsi_stacked(
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
        dsi_percent = _overall_dsi_percent(pairwise_df)

        metadata = {
            "analysis_mode": "search_offset",
            "tool_id": tool_id,
            "num_frames": int(cfg.num_frames),
            "offset_range_tested": f"{cfg.offset_min}-{cfg.offset_max}",
            "target_start_ranges": [list(anchor_start_range)],
            "rotation_model": rotation_model,
            "full_rotation_locked": full_rotation_locked,
            "require_all_edge_phases_within_recording": require_all_edge_phases,
            "maximum_feasible_anchor_frames": len(mask_files) // region_count,
            "edge_count": region_count,
            "symmetry_phase_step_degrees": phase_step_degrees,
            "comparison_span_degrees": comparison_span_degrees,
            "degrees_per_frame": degrees_per_frame,
            "frames_per_degree": 1.0 / degrees_per_frame,
            "calibrated_frames_per_region": calibrated_num_frames,
            "optimal_offset": int(optimal_starts[0]),
            "optimal_frame_range": f"{optimal_starts[0]}-{optimal_starts[0] + cfg.num_frames - 1}",
            "optimal_region_starts": [int(start) for start in optimal_starts],
            "optimal_region_ranges": [_range_to_str(rng) for rng in region_ranges],
            "roi_height_px": int(roi_height),
            "dynamic_roi_enabled": bool(cfg.dynamic_roi_enabled),
            "dynamic_roi_height_factor": float(cfg.dynamic_roi_height_factor),
            "dynamic_roi_master_width_px": dynamic_roi_master_width,
            "centerline_mode": CENTERLINE_MODE,
            "pixel_count_centerline": "one line fitted to the rotated master mask and reused for every frame; ROI only limits counted rows",
            "shared_centerline_slope": float(shared_centerline[0]),
            "shared_centerline_intercept": float(shared_centerline[1]),
            "smoothing_enabled": bool(cfg.smoothing_enabled),
            "smoothing_window": int(cfg.smoothing_window),
            "smoothing_strength": float(cfg.smoothing_strength),
            "global_roi_bottom": int(global_roi_bottom),
            "use_metadata_roi_height": bool(cfg.use_metadata_roi_height),
            "used_cached_search": bool(used_cached_search),
            "mean_abs_diff": mean_abs_diff,
            "dsi_percent": dsi_percent,
            "output_formats": list(_normalize_output_formats(cfg.output_formats)),
            "stack_overlay_abs_diff": bool(cfg.stack_overlay_abs_diff),
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        _save_json(metadata_path, metadata)

        _update_tool_tilt_metadata(
            info_dir,
            {
                "optimal_offset": int(optimal_starts[0]),
                "optimal_region_starts": [int(start) for start in optimal_starts],
                "constant_rotation_degrees_per_frame": float(degrees_per_frame),
                "constant_rotation_frames_per_region": int(calibrated_num_frames),
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
            "dynamic_roi_enabled": bool(cfg.dynamic_roi_enabled),
            "dynamic_roi_height_factor": float(cfg.dynamic_roi_height_factor),
            "dynamic_roi_master_width_px": dynamic_roi_master_width,
            "global_roi_bottom": int(global_roi_bottom),
            "optimal_offset": int(optimal_starts[0]),
            "optimal_offsets": [int(start) for start in optimal_starts],
            "rotation_model": rotation_model,
            "full_rotation_locked": full_rotation_locked,
            "require_all_edge_phases_within_recording": require_all_edge_phases,
            "degrees_per_frame": float(degrees_per_frame),
            "frames_per_degree": float(1.0 / degrees_per_frame),
            "calibrated_frames_per_region": int(calibrated_num_frames),
            "used_cached_search": bool(used_cached_search),
            "frame_range": f"{optimal_starts[0]}-{optimal_starts[0] + cfg.num_frames - 1}",
            "region_ranges": [_range_to_str(rng) for rng in region_ranges],
            "mean_abs_diff": mean_abs_diff,
            "dsi_percent": dsi_percent,
            "metadata_path": metadata_path,
            "pixel_counts_csv_path": pixel_counts_csv_path,
            "abs_diff_csv_path": abs_diff_csv_path,
            "summary_csv_path": summary_csv_path,
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
        shared_centerline,
        cfg,
    )

    out_prefix = os.path.join(out_dir, tool_id)

    abs_diff_csv_path = f"{out_prefix}_abs_diff_per_angle.csv"
    pairwise_df.to_csv(abs_diff_csv_path, index=False)
    pixel_counts_csv_path = f"{out_prefix}_right_side_pixel_counts.csv"
    counts_df.to_csv(pixel_counts_csv_path, index=False)
    summary_csv_path = f"{out_prefix}_comparison_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)

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
    dsi_percent = _overall_dsi_percent(pairwise_df)

    metadata = {
        "analysis_mode": "fixed_ranges",
        "tool_id": tool_id,
        "roi_height_px": int(roi_height),
        "dynamic_roi_enabled": bool(cfg.dynamic_roi_enabled),
        "dynamic_roi_height_factor": float(cfg.dynamic_roi_height_factor),
        "dynamic_roi_master_width_px": dynamic_roi_master_width,
        "centerline_mode": CENTERLINE_MODE,
        "pixel_count_centerline": "one line fitted to the rotated master mask and reused for every frame; ROI only limits counted rows",
        "shared_centerline_slope": float(shared_centerline[0]),
        "shared_centerline_intercept": float(shared_centerline[1]),
        "smoothing_enabled": bool(cfg.smoothing_enabled),
        "smoothing_window": int(cfg.smoothing_window),
        "smoothing_strength": float(cfg.smoothing_strength),
        "global_roi_bottom": int(global_roi_bottom),
        "use_metadata_roi_height": bool(cfg.use_metadata_roi_height),
        "internal_regions": [_range_to_str(r) for r in region_ranges],
        "display_regions_deg": [f"{r[0]}-{r[1]}" for r in display_ranges],
        "manual_legend_ranges": bool(cfg.manual_legend_ranges),
        "pair_count": int(len(counts_df)),
        "mean_abs_diff": mean_abs_diff,
        "dsi_percent": dsi_percent,
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
        "dynamic_roi_enabled": bool(cfg.dynamic_roi_enabled),
        "dynamic_roi_height_factor": float(cfg.dynamic_roi_height_factor),
        "dynamic_roi_master_width_px": dynamic_roi_master_width,
        "global_roi_bottom": int(global_roi_bottom),
        "region_count": int(len(region_ranges)),
        "internal_regions": [_range_to_str(r) for r in region_ranges],
        "display_regions": [f"{r[0]}-{r[1]}" for r in display_ranges],
        "pair_count": int(len(counts_df)),
        "mean_abs_diff": mean_abs_diff,
        "dsi_percent": dsi_percent,
        "metadata_path": metadata_path,
        "pixel_counts_csv_path": pixel_counts_csv_path,
        "abs_diff_csv_path": abs_diff_csv_path,
        "summary_csv_path": summary_csv_path,
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


def _apply_adaptive_summary_y_axis(ax, values: list[float], threshold_value: Optional[float] = None):
    """Apply readable y-axis scaling for both low and extreme-value bar charts.

    Behavior:
    - Base granularity is 500 px.
    - If the range is moderate, keep major ticks every 500.
    - For very large ranges, increase major step (2k / 5k / 10k) but keep minor grid at 500.
    """
    y_max = 0.0
    if values:
        y_max = max(y_max, float(np.max(values)))
    if threshold_value is not None:
        y_max = max(y_max, float(threshold_value))

    y_upper = max(500.0, float(np.ceil(y_max / 500.0) * 500.0))

    # Keep labels readable while preserving 500-step detail via minor ticks.
    tick_count_500 = int(y_upper / 500.0)
    if tick_count_500 <= 14:
        major_step = 500
    elif tick_count_500 <= 40:
        major_step = 2000
    elif tick_count_500 <= 100:
        major_step = 5000
    else:
        major_step = 10000

    ax.set_ylim(0, y_upper)
    ax.yaxis.set_major_locator(MultipleLocator(major_step))
    ax.yaxis.set_minor_locator(MultipleLocator(500))
    ax.grid(axis="y", which="major", alpha=0.35)
    ax.grid(axis="y", which="minor", alpha=0.12, linestyle=":")


def _normalize_threshold_lines(
    threshold_lines: Optional[list[dict]] = None,
    threshold_value: Optional[float] = None,
    show_threshold: bool = False,
) -> list[dict]:
    """Normalize old/new threshold inputs to a standard list format.

    Each entry in result has keys: value(float), color(str), label(str).
    """
    normalized: list[dict] = []

    # Preferred new API path.
    if threshold_lines:
        for i, item in enumerate(threshold_lines, start=1):
            try:
                value = float(item.get("value"))
            except Exception:
                continue
            color = str(item.get("color", "#1f77b4") or "#1f77b4").strip()
            label = str(item.get("label", "") or "").strip()
            if not label:
                label = f"Threshold {i} (T = {value:g})"
            normalized.append({"value": value, "color": color, "label": label})

    # Backward-compatible single-threshold path.
    if not normalized and show_threshold and threshold_value is not None:
        value = float(threshold_value)
        normalized.append({"value": value, "color": "#1f77b4", "label": f"Threshold (T = {value:g})"})

    return normalized


def _draw_threshold_lines(ax, threshold_lines: list[dict]):
    """Draw threshold lines and return legend entries."""
    from matplotlib.lines import Line2D as _Line2D

    legend_lines = []
    for item in threshold_lines:
        value = float(item["value"])
        color = str(item.get("color", "#1f77b4"))
        label = str(item.get("label", f"Threshold (T = {value:g})"))
        ax.axhline(value, color=color, linestyle="--", linewidth=2)
        legend_lines.append(_Line2D([0], [0], color=color, linestyle="--", linewidth=2, label=label))
    return legend_lines


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
    threshold_lines: Optional[list[dict]] = None,
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
    normalized_thresholds = _normalize_threshold_lines(
        threshold_lines=threshold_lines,
        threshold_value=threshold_value,
        show_threshold=show_threshold,
    )

    threshold_max = None
    if normalized_thresholds:
        threshold_max = max(float(t["value"]) for t in normalized_thresholds)

    _apply_adaptive_summary_y_axis(
        ax,
        df["mean_abs_diff"].tolist(),
        threshold_max,
    )

    ax.set_title(
        "Mean Absolute Difference (0-90 vs 180-270, Right Half)",
        fontsize=cfg.title_font_size,
        fontweight="bold",
    )

    # Build legend from actually-present conditions.
    from matplotlib.patches import Patch as _Patch

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

    # Add threshold lines to legend and plot.
    legend_elements.extend(_draw_threshold_lines(ax, normalized_thresholds))

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
    threshold_lines: Optional[list[dict]] = None,
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
    normalized_thresholds = _normalize_threshold_lines(
        threshold_lines=threshold_lines,
        threshold_value=threshold_value,
        show_threshold=show_threshold,
    )
    threshold_max = None
    if normalized_thresholds:
        threshold_max = max(float(t["value"]) for t in normalized_thresholds)

    _apply_adaptive_summary_y_axis(
        ax,
        df["mean_abs_diff"].tolist(),
        threshold_max,
    )
    
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

    legend_elements = [
        _Patch(facecolor=label_config["color"], edgecolor="black", label=label_config["name"])
        for label_config in labels_config
    ]

    # Add threshold lines to legend and plot.
    legend_elements.extend(_draw_threshold_lines(ax, normalized_thresholds))

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
