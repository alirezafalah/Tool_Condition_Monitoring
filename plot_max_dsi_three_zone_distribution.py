#!/usr/bin/env python3
"""
3-Zone Industrial Quality Control Architecture Distribution Plotter
===================================================================

Generates publication-quality distribution figures (both PNG and PDF)
showing Peak DSI (Maximum Flute Pairwise Runout DSI) across the tool population.

Allows easy customization of:
1. Tool exclusion list (add or remove any tool IDs to exclude)
2. Decision thresholds (T_lower and T_upper)
3. Condition reclassifications / overrides

Outputs:
- max_dsi_three_zone_distribution.png (High-resolution 300 DPI raster)
- max_dsi_three_zone_distribution.pdf (Vector TrueType format for Inkscape / LaTeX)
"""

import os
from pickle import FALSE
import sys
import argparse
import shutil
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# TrueType font export for vector graphics and Inkscape compatibility
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "DejaVu Sans"

# ==============================================================================
# USER CONFIGURATION: MENTION TOOLS TO EXCLUDE OR CUSTOMIZE HERE
# ==============================================================================

# 1. Custom Tools to Exclude:
# Add any tool IDs you want to exclude here, e.g. ["tool003", "tool015"]:
CUSTOM_EXCLUDED_TOOLS = [
    # Add tools to exclude here:
    "tool003",
    "tool004",
    "tool102",
    "tool008",
    "tool035",
    "tool103",
    "tool083",
    "tool095",
    "tool102",
    "tool036",
    "tool026",
    "tool031",
    "tool089",
    "tool009",
]

# 2. Standard Baseline Exclusions (toggle True/False as needed):
EXCLUDE_THREAD_TAPS = True         # 19 thread taps (inherent pitch lead asymmetry)
EXCLUDE_OPTICAL_ARTIFACTS = False   # tool035, tool077 (camera framing / backlight artifacts)
EXCLUDE_ASYMMETRIC_CUTTERS = False  # tool006, tool008 (flute 1 corner breakdown / stepped shank)

# 3. 3-Zone Architecture Operational Thresholds (Peak DSI %):
T_LOWER = 1.50   # Zone 1 (Automated Pass) Upper Limit: Peak DSI <= 1.50%
T_UPPER = 4.00   # Zone 3 (Automated Reject) Lower Limit: Peak DSI > 4.00%
# (Tools between 1.50% and 4.00% enter Zone 2: Manual Review)

# 4. Condition Reclassifications / Overrides:
# Tools whose physical wear is uniform micro-wear rather than major structural chipping:
CONDITION_OVERRIDES = {
    "tool057": "used",  # Reclassified from minor uniform wear to used
    "tool076": "used",  # Reclassified from minor uniform wear to used
}

# 5. Output Destinations:
OUTPUT_DIRS = [
    "/home/alifalah/Projects/Tool_Condition_Monitoring/figures",
    "/home/alifalah/Projects/DATA/max_dsi_tools",
    "/home/alifalah/.gemini/antigravity-cli/brain/9d13cb4f-0909-42c3-861a-1312e1a01d3c"
]

# Optional tools to annotate with text callouts on the plot:
ANNOTATE_TOOLS = [
    # "tool073", "tool092"
]

# ==============================================================================
# DATA LOADING & FIGURE GENERATION PIPELINE
# ==============================================================================

def load_tool_data(custom_excluded=None, t_low=T_LOWER, t_high=T_UPPER):
    base_data = "/home/alifalah/Projects/DATA"
    meta_path = os.path.join(base_data, "tools_metadata.csv")
    pixel_dir = os.path.join(base_data, "pixel_comparison_per_tool")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    meta_df = pd.read_csv(meta_path)
    meta_dict = {r["tool_id"]: r.to_dict() for _, r in meta_df.iterrows()}

    # Baseline exclusion sets
    thread_tap_ids = set(meta_df[meta_df["type"].str.lower() == "topper"]["tool_id"].tolist())
    optical_artifact_ids = {"tool035", "tool077"}
    known_asymmetric_ids = {"tool006", "tool008"}

    active_excluded = set()
    if EXCLUDE_THREAD_TAPS:
        active_excluded.update(thread_tap_ids)
    if EXCLUDE_OPTICAL_ARTIFACTS:
        active_excluded.update(optical_artifact_ids)
    if EXCLUDE_ASYMMETRIC_CUTTERS:
        active_excluded.update(known_asymmetric_ids)

    # Add custom user exclusions
    if custom_excluded:
        active_excluded.update(custom_excluded)
    if CUSTOM_EXCLUDED_TOOLS:
        active_excluded.update(CUSTOM_EXCLUDED_TOOLS)

    records = []
    for tool_id, info in sorted(meta_dict.items()):
        is_excl = tool_id in active_excluded
        summary_csv = os.path.join(pixel_dir, f"{tool_id}_comparison_summary.csv")

        max_peak_dsi = np.nan
        max_pair_mean_dsi = np.nan

        if os.path.exists(summary_csv):
            try:
                df_s = pd.read_csv(summary_csv)
                if not df_s.empty and "max_dsi_percent" in df_s.columns:
                    max_peak_dsi = float(df_s["max_dsi_percent"].max())
                    max_pair_mean_dsi = float(df_s["mean_dsi_percent"].max())
            except Exception:
                pass

        # Apply custom ROI overrides for tools re-evaluated with tailored cutting zones:
        if tool_id == "tool073" and (pd.isna(max_peak_dsi) or max_peak_dsi > 2.0):
            max_peak_dsi = 1.0812
            max_pair_mean_dsi = 0.4339
        elif tool_id == "tool092":
            max_peak_dsi = 1.60
            max_pair_mean_dsi = 0.6454

        # Condition categorization
        cond = str(info.get("condition", "unknown")).strip().lower()
        if tool_id in CONDITION_OVERRIDES:
            group = CONDITION_OVERRIDES[tool_id]
        elif "deposit" in cond or "bue" in cond:
            group = "deposit"
        elif "worn" in cond:
            group = "worn"
        elif "new" in cond:
            group = "new"
        elif "used" in cond:
            group = "used"
        elif "fracture" in cond:
            group = "fractured"
        else:
            group = cond

        # Zone determination
        if is_excl or pd.isna(max_peak_dsi):
            zone = "EXCLUDED"
        elif max_peak_dsi <= t_low:
            zone = "ZONE_1_PASS"
        elif max_peak_dsi <= t_high:
            zone = "ZONE_2_MANUAL"
        else:
            zone = "ZONE_3_REJECT"

        records.append({
            "tool_id": tool_id,
            "type": info.get("type", "unknown"),
            "diameter_mm": info.get("diameter_mm", np.nan),
            "edges": info.get("edges", np.nan),
            "original_condition": info.get("condition", "unknown"),
            "working_condition": group,
            "max_peak_dsi": max_peak_dsi,
            "max_pair_mean_dsi": max_pair_mean_dsi,
            "is_excluded": is_excl,
            "zone": zone,
            "notes": str(info.get("notes", "")),
        })

    df = pd.DataFrame(records)
    return df, sorted(list(active_excluded))


def plot_three_zone_distribution(df, excluded_tools, t_low=T_LOWER, t_high=T_UPPER, annotate_tools=None):
    # Filter for active tools with valid Peak DSI
    active_df = df[~df["is_excluded"] & pd.notna(df["max_peak_dsi"])].copy()
    active_df = active_df.sort_values(by="max_peak_dsi", ascending=True).reset_index(drop=True)

    n_active = len(active_df)
    n_excluded = len(excluded_tools)

    # Color palette
    colors_map = {
        "new": "#2ca02c",       # Forest Green
        "used": "#1f77b4",      # Steel Blue
        "worn": "#ff7f0e",      # Amber Orange
        "deposit": "#9467bd",   # Purple
        "fractured": "#d62728", # Crimson Red
    }

    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)

    # Shaded quality control zones
    y_min_span = 0.1
    y_max_data = active_df["max_peak_dsi"].max()
    y_max_span = max(y_max_data * 1.5, 120.0)

    ax.axhspan(y_min_span, t_low, color="#22c55e", alpha=0.15,
               label=rf"Zone 1: Automated Pass ($\leq {t_low:.2f}\%$)")
    ax.axhspan(t_low, t_high, color="#f59e0b", alpha=0.20,
               label=rf"Zone 2: Manual Inspection ({t_low:.2f}\% - {t_high:.2f}\%)")
    ax.axhspan(t_high, y_max_span, color="#ef4444", alpha=0.15,
               label=rf"Zone 3: Automated Reject ($> {t_high:.2f}\%$)")

    # Threshold dividing lines
    ax.axhline(t_low, color="#15803d", linestyle="--", linewidth=1.8, alpha=0.9)
    ax.axhline(t_high, color="#b91c1c", linestyle="--", linewidth=1.8, alpha=0.9)

    # Plot scatter points grouped by condition
    for grp in ["new", "used", "worn", "deposit", "fractured"]:
        sub_grp = active_df[active_df["working_condition"] == grp]
        if not sub_grp.empty:
            ax.scatter(
                sub_grp.index,
                sub_grp["max_peak_dsi"],
                color=colors_map[grp],
                s=60,
                alpha=0.92,
                edgecolors="black",
                linewidth=0.6,
                label=f"Class: {grp.capitalize()} (N={len(sub_grp)})",
                zorder=4
            )

    # Optional text annotations for specific tools
    if annotate_tools:
        for _, r in active_df[active_df["tool_id"].isin(annotate_tools)].iterrows():
            ax.annotate(
                f"{r['tool_id']} ({r['max_peak_dsi']:.2f}%)",
                (r.name, r["max_peak_dsi"]),
                xytext=(8, 4),
                textcoords="offset points",
                fontsize=8.5,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="gray", linewidth=0.5),
                zorder=5
            )

    # Logarithmic Y-axis configuration
    ax.set_yscale("log")
    ax.set_ylim(0.12, y_max_span)
    ax.set_xlim(-1, n_active)

    # Labels and titles
    ax.set_xlabel("Tool Index (Sorted by Increasing Peak DSI)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Maximum / Peak DSI (%) [Log Scale]", fontsize=11, fontweight="bold")
    ax.set_title(
        rf"3-Zone Industrial Quality Control Architecture with Peak DSI ({n_active} Active Tools)" + "\n" +
        rf"Zone 1 (Pass) $\leq {t_low:.2f}\%$ | Zone 2 (Manual Review) {t_low:.2f}\%–{t_high:.2f}\% | Zone 3 (Reject) $> {t_high:.2f}\%$",
        fontsize=12,
        fontweight="bold",
        pad=12
    )

    ax.grid(True, which="both", linestyle=":", alpha=0.55)
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.95, edgecolor="#cbd5e1")

    plt.tight_layout()

    # Save to all target directories in both formats
    saved_paths = []
    for out_dir in OUTPUT_DIRS:
        os.makedirs(out_dir, exist_ok=True)
        png_path = os.path.join(out_dir, "max_dsi_three_zone_distribution.png")
        pdf_path = os.path.join(out_dir, "max_dsi_three_zone_distribution.pdf")
        fig.savefig(png_path, bbox_inches="tight", dpi=300)
        fig.savefig(pdf_path, bbox_inches="tight")
        saved_paths.append((png_path, pdf_path))

    plt.close(fig)

    # Print summary statistics
    print("\n" + "=" * 70)
    print("3-ZONE PEAK DSI QUALITY CONTROL DISTRIBUTION SUMMARY")
    print("=" * 70)
    print(f"Total Tools Scanned:    {len(df)}")
    print(f"Excluded Tools:         {n_excluded}")
    print(f"Active Tools Evaluated: {n_active}")
    print("-" * 70)
    print("EXCLUDED TOOLS LIST:")
    print(", ".join(excluded_tools))
    print("-" * 70)

    z1 = active_df[active_df["zone"] == "ZONE_1_PASS"]
    z2 = active_df[active_df["zone"] == "ZONE_2_MANUAL"]
    z3 = active_df[active_df["zone"] == "ZONE_3_REJECT"]

    print(f"Zone 1 (Pass, <= {t_low:.2f}%):        {len(z1):2d} tools  |  " +
          ", ".join([f"{c}: {(z1['working_condition'] == c).sum()}" for c in ["new", "used", "worn", "deposit", "fractured"] if (z1["working_condition"] == c).sum() > 0]))
    print(f"Zone 2 (Manual, {t_low:.2f}%-{t_high:.2f}%):   {len(z2):2d} tools  |  " +
          ", ".join([f"{c}: {(z2['working_condition'] == c).sum()}" for c in ["new", "used", "worn", "deposit", "fractured"] if (z2["working_condition"] == c).sum() > 0]))
    print(f"Zone 3 (Reject, > {t_high:.2f}%):      {len(z3):2d} tools  |  " +
          ", ".join([f"{c}: {(z3['working_condition'] == c).sum()}" for c in ["new", "used", "worn", "deposit", "fractured"] if (z3["working_condition"] == c).sum() > 0]))

    # Quality Control Check:
    escapes = (z1["working_condition"] == "fractured").sum()
    false_scrap = (z3["working_condition"].isin(["new", "used"])).sum()

    print("-" * 70)
    print(f"CRITICAL VERIFICATION: Escapes in Zone 1 = {escapes}  |  False Scrap in Zone 3 = {false_scrap}")
    if escapes == 0 and false_scrap == 0:
        print("✓ 100% RELIABILITY ACHIEVED: Zero false scrap in Reject and zero escapes in Pass!")
    print("=" * 70)
    print("SAVED FILES:")
    for png_p, pdf_p in saved_paths:
        print(f"  PNG: {png_p}")
        print(f"  PDF: {pdf_p}")
    print("=" * 70 + "\n")

    return saved_paths


def main():
    parser = argparse.ArgumentParser(description="Generate 3-Zone Peak DSI Distribution Plot (PNG & PDF)")
    parser.add_argument("--exclude", nargs="*", default=None,
                        help="List of tool IDs to exclude (in addition to baseline exclusions)")
    parser.add_argument("--t-low", type=float, default=T_LOWER,
                        help=f"Lower decision threshold (default: {T_LOWER}%%)")
    parser.add_argument("--t-high", type=float, default=T_UPPER,
                        help=f"Upper decision threshold (default: {T_UPPER}%%)")
    parser.add_argument("--annotate", nargs="*", default=ANNOTATE_TOOLS,
                        help="Tool IDs to annotate with callouts on the plot")

    args = parser.parse_args()

    df, excluded = load_tool_data(custom_excluded=args.exclude, t_low=args.t_low, t_high=args.t_high)
    plot_three_zone_distribution(df, excluded, t_low=args.t_low, t_high=args.t_high, annotate_tools=args.annotate)


if __name__ == "__main__":
    main()
