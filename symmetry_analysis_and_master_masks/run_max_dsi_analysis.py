#!/usr/bin/env python3
"""
Comprehensive Maximum / Peak DSI Analysis, Threshold Optimization, and 3-Zone Reconfiguration.

Key Objectives:
1. Replace Mean Pairwise DSI with Maximum / Peak Pairwise DSI (max difference within comparison zones).
2. Compare single-threshold metrics (Accuracy, FP, FN, Sensitivity, Specificity, F1) against Mean DSI.
3. Test 3-Zone Quality Control Architecture under:
   - Configuration 1: Original Thresholds (T_lower = 1.5%, T_upper = 3.0%)
   - Configuration 2: Redefined Balanced Thresholds (T_lower = 1.5%, T_upper = 3.5%)
   - Configuration 3: Redefined Operational Thresholds (T_lower = 1.5%, T_upper = 4.0%)
4. Identify critical outliers and diagnose whether Max DSI makes the triage better or worse.
5. Save all outputs and browsable stacked images in DATA/max_dsi_tools/.
"""

import os
import sys
import glob
import json
import shutil
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# TrueType font export for Inkscape compatibility
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "DejaVu Sans"

def main():
    base_dir = "/home/alifalah/Projects/DATA"
    out_dir = os.path.join(base_dir, "max_dsi_tools")
    pixel_dir = os.path.join(base_dir, "pixel_comparison_per_tool")
    meta_path = os.path.join(base_dir, "tools_metadata.csv")

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    meta_df = pd.read_csv(meta_path)
    meta_dict = {r["tool_id"]: r.to_dict() for _, r in meta_df.iterrows()}

    topper_ids = set(meta_df[meta_df["type"].str.lower() == "topper"]["tool_id"].tolist())
    artifact_ids = {"tool035", "tool077"}
    asymmetric_ids = {"tool006", "tool008"}
    excluded_ids = topper_ids.union(artifact_ids).union(asymmetric_ids)

    print(f"Total tools in metadata: {len(meta_df)}")
    print(f"Total excluded tools: {len(excluded_ids)} (19 toppers + 2 artifacts + 2 asymmetric)")
    print(f"Total active tools: {len(meta_df) - len(excluded_ids)}")

    records = []
    for tool_id, info in sorted(meta_dict.items()):
        is_excluded = tool_id in excluded_ids
        summary_csv = os.path.join(pixel_dir, f"{tool_id}_comparison_summary.csv")

        max_pair_mean_dsi = np.nan
        max_peak_dsi = np.nan
        max_abs_diff_pixels = np.nan
        mean_abs_diff_pixels = np.nan

        if os.path.exists(summary_csv):
            df_s = pd.read_csv(summary_csv)
            max_pair_mean_dsi = float(df_s["mean_dsi_percent"].max())
            max_peak_dsi = float(df_s["max_dsi_percent"].max())
            max_abs_diff_pixels = float(df_s["max_difference"].max())
            mean_abs_diff_pixels = float(df_s["mean_difference"].max())

        meta_json = os.path.join(pixel_dir, f"{tool_id}_symmetry_metadata.json")
        overall_dsi = np.nan
        if os.path.exists(meta_json):
            with open(meta_json, "r") as fp:
                mj = json.load(fp)
            overall_dsi = float(mj.get("results", {}).get("overall_dsi_percent", np.nan))

        cond = str(info.get("condition", "unknown")).strip().lower()
        if tool_id in ["tool057", "tool076"]:
            group = "used"
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

        if is_excluded:
            if tool_id in topper_ids:
                exclusion_reason = "EXCLUDED_TOPPER"
            elif tool_id in asymmetric_ids:
                exclusion_reason = "EXCLUDED_ASYMMETRIC_TOOL"
            else:
                exclusion_reason = "EXCLUDED_SEGMENTATION_ARTIFACT"
            role = "excluded"
        else:
            exclusion_reason = "NONE"
            role = "calibration" if group in ["new", "used", "fractured"] else "secondary_evaluation"

        records.append({
            "tool_id": tool_id,
            "type": info.get("type"),
            "diameter_mm": info.get("diameter_mm"),
            "edges": info.get("edges"),
            "condition_raw": info.get("condition"),
            "group": group,
            "role": role,
            "exclusion_reason": exclusion_reason,
            "max_pair_mean_dsi": round(max_pair_mean_dsi, 4) if pd.notna(max_pair_mean_dsi) else np.nan,
            "max_peak_dsi": round(max_peak_dsi, 4) if pd.notna(max_peak_dsi) else np.nan,
            "max_abs_diff_pixels": round(max_abs_diff_pixels, 2) if pd.notna(max_abs_diff_pixels) else np.nan,
            "mean_abs_diff_pixels": round(mean_abs_diff_pixels, 2) if pd.notna(mean_abs_diff_pixels) else np.nan,
            "overall_dsi": round(overall_dsi, 4) if pd.notna(overall_dsi) else np.nan,
            "notes": str(info.get("notes", "")),
        })

    all_df = pd.DataFrame(records)

    # 1. Sweep on Calibration Set (83 tools) for Max Peak DSI
    calib_df = all_df[all_df["role"] == "calibration"].copy()
    calib_df["is_fractured"] = (calib_df["group"] == "fractured").astype(int)

    thresholds = np.arange(1.0, 6.05, 0.05)
    sweep_results = []
    best_th = 3.15
    min_errors = 999
    best_acc = 0.0

    for th in thresholds:
        th = round(th, 2)
        pred = (calib_df["max_peak_dsi"] > th).astype(int)
        actual = calib_df["is_fractured"]

        tp = int(((pred == 1) & (actual == 1)).sum())
        tn = int(((pred == 0) & (actual == 0)).sum())
        fp = int(((pred == 1) & (actual == 0)).sum())
        fn = int(((pred == 0) & (actual == 1)).sum())

        err = fp + fn
        acc = (tp + tn) / len(calib_df) * 100.0
        sens = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
        spec = (tn / (tn + fp) * 100.0) if (tn + fp) > 0 else 0.0
        prec = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
        f1 = (2 * prec * sens / (prec + sens)) if (prec + sens) > 0 else 0.0

        if err < min_errors:
            min_errors = err
            best_th = th
            best_acc = acc

        sweep_results.append({
            "threshold_peak_dsi_percent": th,
            "accuracy_percent": round(acc, 2),
            "total_errors": err,
            "false_positives": fp,
            "false_negatives": fn,
            "sensitivity_recall_percent": round(sens, 2),
            "specificity_percent": round(spec, 2),
            "precision_percent": round(prec, 2),
            "f1_score_percent": round(f1, 2),
        })

    sweep_df = pd.DataFrame(sweep_results)
    sweep_csv = os.path.join(out_dir, "threshold_sweep_max_dsi.csv")
    sweep_df.to_csv(sweep_csv, index=False)
    print(f"Optimal Single Threshold for Peak DSI: T = {best_th:.2f}% (Accuracy = {best_acc:.2f}%, Total Errors = {min_errors})")

    # 2. Assign 3-Zone Classifications
    # Configuration 1: Original Thresholds (T_low = 1.5%, T_high = 3.0%)
    # Configuration 2: Redefined Balanced Thresholds (T_low = 1.5%, T_high = 3.5%)
    # Configuration 3: Redefined Operational Thresholds (T_low = 1.5%, T_high = 4.0%)
    def get_zone(val, t_low, t_high, role):
        if role == "excluded":
            return "EXCLUDED"
        if pd.isna(val):
            return "UNKNOWN"
        if val <= t_low:
            return "ZONE_1_PASS"
        elif val <= t_high:
            return "ZONE_2_MANUAL"
        else:
            return "ZONE_3_REJECT"

    all_df["zone_orig_1p5_3p0"] = all_df.apply(lambda r: get_zone(r["max_peak_dsi"], 1.5, 3.0, r["role"]), axis=1)
    all_df["zone_redef_1p5_3p5"] = all_df.apply(lambda r: get_zone(r["max_peak_dsi"], 1.5, 3.5, r["role"]), axis=1)
    all_df["zone_redef_1p5_4p0"] = all_df.apply(lambda r: get_zone(r["max_peak_dsi"], 1.5, 4.0, r["role"]), axis=1)

    # Single threshold prediction at optimal T = 3.15%
    all_df["single_th_3p15_pred"] = (all_df["max_peak_dsi"] > 3.15).astype(int)

    def get_single_status(r):
        if r["role"] == "excluded":
            return f"EXCLUDED ({r['exclusion_reason']})"
        is_hi = r["max_peak_dsi"] > 3.15
        grp = r["group"]
        if grp in ["new", "used"]:
            return "FALSE_POSITIVE" if is_hi else "CORRECT_HEALTHY"
        elif grp == "fractured":
            return "CORRECT_FRACTURED" if is_hi else "FALSE_NEGATIVE"
        elif grp in ["worn", "deposit"]:
            return f"SECONDARY_{grp.upper()}_HIGH" if is_hi else f"SECONDARY_{grp.upper()}_LOW"
        return "UNKNOWN"

    all_df["single_th_status"] = all_df.apply(get_single_status, axis=1)

    # Save full population master CSV
    master_csv = os.path.join(out_dir, "full_population_max_dsi_classification.csv")
    all_df.to_csv(master_csv, index=False)

    # 3. Plot 1: Peak DSI Threshold Optimization Curve
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), dpi=300, sharex=True)
    ax1.plot(sweep_df["threshold_peak_dsi_percent"], sweep_df["accuracy_percent"], color="navy", linewidth=2.5, label="Overall Accuracy (%)")
    ax1.plot(sweep_df["threshold_peak_dsi_percent"], sweep_df["f1_score_percent"], color="teal", linewidth=2.0, linestyle="--", label="F1-Score (%)")
    ax1.axvline(best_th, color="red", linestyle=":", linewidth=2.0, label=f"Optimal Threshold T = {best_th:.2f}% (Acc={best_acc:.1f}%)")
    ax1.set_ylabel("Performance (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Single-Threshold Calibration for Peak DSI (33 New/Used vs 50 Fractured)", fontsize=13, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower left", fontsize=10, frameon=True)

    ax2.plot(sweep_df["threshold_peak_dsi_percent"], sweep_df["false_positives"], color="crimson", linewidth=2.2, label="False Positives (New/Used > T)")
    ax2.plot(sweep_df["threshold_peak_dsi_percent"], sweep_df["false_negatives"], color="darkorange", linewidth=2.2, label="False Negatives (Fractured <= T)")
    ax2.plot(sweep_df["threshold_peak_dsi_percent"], sweep_df["total_errors"], color="black", linewidth=2.5, linestyle="-.", label="Total Misclassifications")
    ax2.axvline(best_th, color="red", linestyle=":", linewidth=2.0, label=f"Optimal Threshold T = {best_th:.2f}% (Errors={min_errors})")
    ax2.set_xlabel("Decision Threshold T (% Maximum / Peak DSI)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Error Count (Tools)", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=10, frameon=True)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "max_dsi_threshold_optimization_curve.png"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "max_dsi_threshold_optimization_curve.pdf"), bbox_inches="tight")
    plt.close(fig)

    # 4. Plot 2: 3-Zone Architecture Distribution Plot for Peak DSI (using T_low=1.5%, T_high=4.0%)
    fig3, ax3 = plt.subplots(figsize=(12, 7), dpi=300)
    active_df = all_df[all_df["role"] != "excluded"].sort_values(by="max_peak_dsi").reset_index(drop=True)

    colors_map = {
        "new": "#2ca02c",       # Green
        "used": "#1f77b4",      # Blue
        "worn": "#ff7f0e",      # Orange
        "deposit": "#9467bd",   # Purple
        "fractured": "#d62728", # Red
    }

    t_low_chosen = 1.50
    t_high_chosen = 4.00

    ax3.axhspan(0.1, t_low_chosen, color="green", alpha=0.15, label=f"Zone 1: Automated Pass (<= {t_low_chosen:.1f}%)")
    ax3.axhspan(t_low_chosen, t_high_chosen, color="gold", alpha=0.20, label=f"Zone 2: Manual Inspection ({t_low_chosen:.1f}% - {t_high_chosen:.1f}%)")
    ax3.axhspan(t_high_chosen, max(active_df["max_peak_dsi"].max() + 10, 110), color="red", alpha=0.15, label=f"Zone 3: Automated Reject (> {t_high_chosen:.1f}%)")

    ax3.axhline(t_low_chosen, color="darkgreen", linestyle="--", linewidth=1.8)
    ax3.axhline(t_high_chosen, color="darkred", linestyle="--", linewidth=1.8)

    for grp in ["new", "used", "worn", "deposit", "fractured"]:
        sub_grp = active_df[active_df["group"] == grp]
        ax3.scatter(sub_grp.index, sub_grp["max_peak_dsi"],
                    color=colors_map[grp], s=55, alpha=0.9, edgecolors="k", linewidth=0.6,
                    label=f"Class: {grp.capitalize()} (N={len(sub_grp)})")

    ax3.set_yscale("log")
    ax3.set_xlabel("Tool Index (Sorted by Increasing Peak DSI)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Maximum / Peak DSI (%) [Log Scale]", fontsize=11, fontweight="bold")
    ax3.set_title(f"3-Zone Industrial Quality Control Architecture with Peak DSI (94 Active Tools)\nZone 1 (Pass) <= {t_low_chosen:.1f}% | Zone 2 (Manual Review) {t_low_chosen:.1f}%-{t_high_chosen:.1f}% | Zone 3 (Reject) > {t_high_chosen:.1f}%", fontsize=12, fontweight="bold")
    ax3.grid(True, which="both", linestyle=":", alpha=0.6)
    ax3.legend(loc="upper left", fontsize=9.5, framealpha=0.95)

    plt.tight_layout()
    fig3.savefig(os.path.join(out_dir, "max_dsi_three_zone_distribution.png"), bbox_inches="tight")
    fig3.savefig(os.path.join(out_dir, "max_dsi_three_zone_distribution.pdf"), bbox_inches="tight")
    plt.close(fig3)

    # 5. Plot 3: Scatter Plot Comparing Mean DSI vs Peak DSI directly
    fig4, ax4 = plt.subplots(figsize=(9, 8), dpi=300)
    for grp in ["new", "used", "worn", "deposit", "fractured"]:
        sub_grp = active_df[active_df["group"] == grp]
        ax4.scatter(sub_grp["max_pair_mean_dsi"], sub_grp["max_peak_dsi"],
                    color=colors_map[grp], s=60, alpha=0.85, edgecolors="k", linewidth=0.6,
                    label=f"{grp.capitalize()} (N={len(sub_grp)})")

    # Add diagonal y=x line
    max_val = 25.0
    ax4.plot([0, max_val], [0, max_val], color="gray", linestyle=":", label="y = x (Uniform Flute Profile)")
    ax4.axvline(1.5, color="green", linestyle="--", alpha=0.6, label="Mean DSI T_low (1.5%)")
    ax4.axvline(3.0, color="red", linestyle="--", alpha=0.6, label="Mean DSI T_high (3.0%)")
    ax4.axhline(1.5, color="darkgreen", linestyle="-.", alpha=0.6, label="Peak DSI T_low (1.5%)")
    ax4.axhline(4.0, color="darkred", linestyle="-.", alpha=0.6, label="Peak DSI T_high (4.0%)")

    # Annotate critical tools
    annot_tools = ["tool092", "tool056", "tool034", "tool084", "tool003", "tool006", "tool008", "tool073"]
    for _, r in active_df[active_df["tool_id"].isin(annot_tools)].iterrows():
        ax4.annotate(r["tool_id"], (r["max_pair_mean_dsi"], r["max_peak_dsi"]),
                     xytext=(6, 4), textcoords="offset points", fontsize=8.5, fontweight="bold")

    ax4.set_xlim(-0.2, 12.0)
    ax4.set_ylim(-0.2, 12.0)
    ax4.set_xlabel("Maximum Pairwise Mean DSI (%)", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Maximum / Peak DSI (%)", fontsize=11, fontweight="bold")
    ax4.set_title("Direct Metric Comparison: Mean DSI vs Peak DSI", fontsize=12, fontweight="bold")
    ax4.grid(True, linestyle=":", alpha=0.6)
    ax4.legend(loc="upper left", fontsize=9, framealpha=0.95)

    plt.tight_layout()
    fig4.savefig(os.path.join(out_dir, "mean_vs_max_dsi_scatter_comparison.png"), bbox_inches="tight")
    fig4.savefig(os.path.join(out_dir, "mean_vs_max_dsi_scatter_comparison.pdf"), bbox_inches="tight")
    plt.close(fig4)

    # 6. Organize Subfolders with Stacked Plot Copies
    subfolder_configs = {
        "ZONE_1_GREEN_AUTOMATED_PASS": all_df[all_df["zone_redef_1p5_4p0"] == "ZONE_1_PASS"],
        "ZONE_2_YELLOW_MANUAL_INSPECTION": all_df[all_df["zone_redef_1p5_4p0"] == "ZONE_2_MANUAL"],
        "ZONE_3_RED_AUTOMATED_REJECT": all_df[all_df["zone_redef_1p5_4p0"] == "ZONE_3_REJECT"],
        "CRITICAL_OUTLIERS_REFINED_ZONES": all_df[
            ((all_df["zone_redef_1p5_4p0"] == "ZONE_1_PASS") & (all_df["group"] == "fractured")) |
            ((all_df["zone_redef_1p5_4p0"] == "ZONE_3_REJECT") & (all_df["group"].isin(["new", "used"])))
        ],
        "CRITICAL_OUTLIERS_ORIGINAL_ZONES": all_df[
            ((all_df["zone_orig_1p5_3p0"] == "ZONE_1_PASS") & (all_df["group"] == "fractured")) |
            ((all_df["zone_orig_1p5_3p0"] == "ZONE_3_REJECT") & (all_df["group"].isin(["new", "used"])))
        ],
        "EXCLUDED_TOPPERS_AND_ARTIFACTS": all_df[all_df["role"] == "excluded"],
    }

    for sub_name, sub_df in subfolder_configs.items():
        sub_path = os.path.join(out_dir, sub_name)
        os.makedirs(sub_path, exist_ok=True)
        sub_df_sorted = sub_df.sort_values(by="max_peak_dsi", ascending=True)
        sub_csv = os.path.join(sub_path, f"{sub_name.lower()}_summary.csv")
        sub_df_sorted.to_csv(sub_csv, index=False)

        for tid in sub_df_sorted["tool_id"]:
            src_png = os.path.join(pixel_dir, f"{tid}_overlay_dsi_stacked.png")
            if os.path.exists(src_png):
                shutil.copy2(src_png, os.path.join(sub_path, f"{tid}_overlay_dsi_stacked.png"))

    # Copy plots to brain artifact directory
    artifact_dir = "/home/alifalah/.gemini/antigravity-cli/brain/9d13cb4f-0909-42c3-861a-1312e1a01d3c"
    for img_name in ["max_dsi_threshold_optimization_curve.png", "max_dsi_three_zone_distribution.png", "mean_vs_max_dsi_scatter_comparison.png"]:
        src = os.path.join(out_dir, img_name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(artifact_dir, img_name))

    print("\n=== Max DSI Analysis Completed Successfully! ===")
    print(f"Results archived in: {out_dir}")

if __name__ == "__main__":
    main()
