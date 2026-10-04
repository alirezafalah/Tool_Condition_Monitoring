#!/usr/bin/env python3
"""
Refined Outlier Analysis, Threshold Optimization, and 3-Zone Decision Architecture.

Changes in this version:
1. Complete removal of all 'topper' (threading tap) tools (19 tools):
   - Justification: Thread taps have asymmetric thread pitch lead, helical relief, and non-symmetric chamfers that violate rotational silhouette symmetry.
2. Complete removal of known segmentation/framing artifact tools:
   - tool035 (endmill) and tool077 (t_slot).
   - Total excluded tools: 21 tools (19 toppers + tool035 + tool077).
3. Active Dataset (94 tools):
   - Calibration Set (83 tools): 'new' (15) + 'used' (18) vs 'fractured' (50).
   - Secondary Evaluation Set (11 tools): 'worn' (8) + 'deposit' (3).
4. Dual Framework:
   - Framework A: Single-Threshold Binary Decision (Optimal T = 2.0%).
   - Framework B: 3-Zone Industrial Inspection Architecture:
     * Zone 1 (Green / Automated Pass): DSI <= T_lower (1.5%) -> Pristine symmetry, automated clearance.
     * Zone 2 (Yellow / Manual Inspection): T_lower < DSI <= T_upper (1.5% to 3.0%) -> Micro-chipping, fine wear; requires microscope/operator review.
     * Zone 3 (Red / Automated Reject): DSI > T_upper (3.0%) -> Severe fracture / catastrophic failure; automated scrapping.
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
    out_dir = os.path.join(base_dir, "outlier_analysis_and_thresholding")
    pixel_dir = os.path.join(base_dir, "pixel_comparison_per_tool")
    meta_path = os.path.join(base_dir, "tools_metadata.csv")

    # Clear previous contents if existing to prevent stale files
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    meta_df = pd.read_csv(meta_path)
    meta_dict = {r["tool_id"]: r.to_dict() for _, r in meta_df.iterrows()}

    # 1. Identify excluded tools
    topper_ids = set(meta_df[meta_df["type"].str.lower() == "topper"]["tool_id"].tolist())
    artifact_ids = {"tool035", "tool077"}
    excluded_ids = topper_ids.union(artifact_ids)

    print(f"Total tools in metadata: {len(meta_df)}")
    print(f"Total excluded tools: {len(excluded_ids)} (19 toppers + 2 artifact tools)")
    print(f"Total active tools: {len(meta_df) - len(excluded_ids)}")

    # 2. Gather tool metrics
    records = []
    for tool_id, info in sorted(meta_dict.items()):
        is_excluded = tool_id in excluded_ids
        summary_csv = os.path.join(pixel_dir, f"{tool_id}_comparison_summary.csv")

        max_pair_mean_dsi = np.nan
        max_peak_dsi = np.nan
        overall_dsi = np.nan

        if os.path.exists(summary_csv):
            df_s = pd.read_csv(summary_csv)
            max_pair_mean_dsi = float(df_s["mean_dsi_percent"].max())
            max_peak_dsi = float(df_s["max_dsi_percent"].max())

        meta_json = os.path.join(pixel_dir, f"{tool_id}_symmetry_metadata.json")
        if os.path.exists(meta_json):
            with open(meta_json, "r") as fp:
                mj = json.load(fp)
            overall_dsi = float(mj.get("results", {}).get("overall_dsi_percent", np.nan))

        cond = str(info.get("condition", "unknown")).strip().lower()
        if "deposit" in cond or "bue" in cond:
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

        t_type = str(info.get("type", "unknown")).strip().lower()
        if is_excluded:
            exclusion_reason = "EXCLUDED_TOPPER" if tool_id in topper_ids else "EXCLUDED_SEGMENTATION_ARTIFACT"
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
            "max_pair_mean_dsi": max_pair_mean_dsi,
            "max_peak_dsi": max_peak_dsi,
            "overall_dsi": overall_dsi,
            "notes": str(info.get("notes", "")),
        })

    all_df = pd.DataFrame(records)

    # 3. Calibration Set Sweep (83 tools: 33 new/used vs 50 fractured)
    calib_df = all_df[all_df["role"] == "calibration"].copy()
    calib_df["is_fractured"] = (calib_df["group"] == "fractured").astype(int)

    thresholds = np.arange(1.0, 5.05, 0.05)
    sweep_results = []

    best_th = 2.0
    min_errors = 999
    best_acc = 0.0

    for th in thresholds:
        th = round(th, 2)
        pred = (calib_df["max_pair_mean_dsi"] > th).astype(int)
        actual = calib_df["is_fractured"]

        tp = int(((pred == 1) & (actual == 1)).sum())
        tn = int(((pred == 0) & (actual == 0)).sum())
        fp = int(((pred == 1) & (actual == 0)).sum())
        fn = int(((pred == 0) & (actual == 1)).sum())

        acc = (tp + tn) / len(calib_df) * 100.0
        err = fp + fn
        sens = tp / (tp + fn) * 100.0 if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) * 100.0 if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) * 100.0 if (tp + fp) > 0 else 0.0
        f1 = (2 * prec * sens) / (prec + sens) if (prec + sens) > 0 else 0.0

        if err < min_errors:
            min_errors = err
            best_th = th
            best_acc = acc

        sweep_results.append({
            "threshold_dsi_percent": th,
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
    sweep_csv = os.path.join(out_dir, "threshold_sweep_metrics.csv")
    sweep_df.to_csv(sweep_csv, index=False)

    print(f"Optimal Single Threshold: T = {best_th:.2f}% (Accuracy = {best_acc:.2f}%, Total Errors = {min_errors})")

    # Fixed single threshold: 2.0%
    chosen_single_th = 2.00
    all_df["single_threshold"] = chosen_single_th
    all_df["single_th_prediction"] = (all_df["max_pair_mean_dsi"] > chosen_single_th).astype(int)

    # 4. Framework B: 3-Zone Architecture Definition
    # Zone 1: DSI <= 1.50% (Automated Pass)
    # Zone 2: 1.50% < DSI <= 3.00% (Manual Operator Inspection)
    # Zone 3: DSI > 3.00% (Automated Reject)
    t_lower = 1.50
    t_upper = 3.00

    def assign_three_zone(row):
        if row["role"] == "excluded":
            return "EXCLUDED"
        val = row["max_pair_mean_dsi"]
        if pd.isna(val):
            return "UNKNOWN"
        if val <= t_lower:
            return "ZONE_1_PASS"
        elif val <= t_upper:
            return "ZONE_2_INSPECTION"
        else:
            return "ZONE_3_REJECT"

    all_df["three_zone_assignment"] = all_df.apply(assign_three_zone, axis=1)

    # Classification status for single threshold
    def assign_single_status(row):
        if row["role"] == "excluded":
            return f"EXCLUDED ({row['exclusion_reason']})"
        grp = row["group"]
        is_high = row["max_pair_mean_dsi"] > chosen_single_th
        if grp in ["new", "used"]:
            return "FALSE_POSITIVE (High DSI Outlier)" if is_high else "CORRECT_HEALTHY"
        elif grp == "fractured":
            return "CORRECT_FRACTURED" if is_high else "FALSE_NEGATIVE (Low DSI Outlier)"
        elif grp in ["worn", "deposit"]:
            return f"SECONDARY_{grp.upper()}_DETECTED_HIGH" if is_high else f"SECONDARY_{grp.upper()}_SUB_THRESHOLD_LOW"
        return "UNKNOWN"

    all_df["single_th_status"] = all_df.apply(assign_single_status, axis=1)

    # Save full population master CSV
    master_csv = os.path.join(out_dir, "full_population_classification.csv")
    all_df.to_csv(master_csv, index=False)

    # 5. Plot 1: Single Threshold Optimization Curve
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), dpi=300, sharex=True)
    ax1.plot(sweep_df["threshold_dsi_percent"], sweep_df["accuracy_percent"], color="navy", linewidth=2.5, label="Overall Accuracy (%)")
    ax1.plot(sweep_df["threshold_dsi_percent"], sweep_df["f1_score_percent"], color="teal", linewidth=2.0, linestyle="--", label="F1-Score (%)")
    ax1.axvline(chosen_single_th, color="red", linestyle=":", linewidth=2.0, label=f"Selected Threshold T = {chosen_single_th:.1f}%")
    ax1.set_ylabel("Performance (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Single-Threshold Calibration (Excluding Toppers: 33 New/Used vs 50 Fractured)", fontsize=13, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower left", fontsize=10, frameon=True)

    ax2.plot(sweep_df["threshold_dsi_percent"], sweep_df["false_positives"], color="crimson", linewidth=2.2, label="False Positives (New/Used > T)")
    ax2.plot(sweep_df["threshold_dsi_percent"], sweep_df["false_negatives"], color="darkorange", linewidth=2.2, label="False Negatives (Fractured <= T)")
    ax2.plot(sweep_df["threshold_dsi_percent"], sweep_df["total_errors"], color="black", linewidth=2.5, linestyle="-.", label="Total Misclassifications")
    ax2.axvline(chosen_single_th, color="red", linestyle=":", linewidth=2.0, label=f"Selected Threshold T = {chosen_single_th:.1f}%")
    ax2.set_xlabel("Decision Threshold T (% Max Pairwise Mean DSI)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Error Count (Tools)", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=10, frameon=True)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "threshold_optimization_curve.png"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "threshold_optimization_curve.pdf"), bbox_inches="tight")
    plt.close(fig)

    # 6. Plot 2: 3-Zone Architecture Distribution Plot
    fig3, ax3 = plt.subplots(figsize=(12, 7), dpi=300)
    active_df = all_df[all_df["role"] != "excluded"].sort_values(by="max_pair_mean_dsi").reset_index(drop=True)
    
    colors_map = {
        "new": "#2ca02c",       # Green
        "used": "#1f77b4",      # Blue
        "worn": "#ff7f0e",      # Orange
        "deposit": "#9467bd",   # Purple
        "fractured": "#d62728", # Red
    }

    # Background shading for 3 zones
    ax3.axhspan(0, t_lower, color="green", alpha=0.15, label=f"Zone 1: Automated Pass (<= {t_lower:.1f}%)")
    ax3.axhspan(t_lower, t_upper, color="gold", alpha=0.20, label=f"Zone 2: Manual Inspection ({t_lower:.1f}% - {t_upper:.1f}%)")
    ax3.axhspan(t_upper, max(active_df["max_pair_mean_dsi"].max() + 5, 50), color="red", alpha=0.15, label=f"Zone 3: Automated Reject (> {t_upper:.1f}%)")

    # Threshold horizontal lines
    ax3.axhline(t_lower, color="darkgreen", linestyle="--", linewidth=1.8)
    ax3.axhline(t_upper, color="darkred", linestyle="--", linewidth=1.8)

    # Scatter points
    for grp in ["new", "used", "worn", "deposit", "fractured"]:
        sub_grp = active_df[active_df["group"] == grp]
        ax3.scatter(sub_grp.index, sub_grp["max_pair_mean_dsi"],
                    color=colors_map[grp], s=55, alpha=0.9, edgecolors="k", linewidth=0.6,
                    label=f"Class: {grp.capitalize()} (N={len(sub_grp)})")

    ax3.set_yscale("log")
    ax3.set_xlabel("Tool Index (Sorted by Increasing Max Pairwise DSI)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Max Pairwise Mean DSI (%) [Log Scale]", fontsize=11, fontweight="bold")
    ax3.set_title(f"3-Zone Industrial Quality Control Architecture across 94 Active Tools\nZone 1 (Pass) <= {t_lower:.1f}% | Zone 2 (Inspection) {t_lower:.1f}%-{t_upper:.1f}% | Zone 3 (Reject) > {t_upper:.1f}%", fontsize=12, fontweight="bold")
    ax3.grid(True, which="both", linestyle=":", alpha=0.6)
    ax3.legend(loc="upper left", fontsize=9.5, framealpha=0.95)

    plt.tight_layout()
    fig3.savefig(os.path.join(out_dir, "three_zone_distribution_plot.png"), bbox_inches="tight")
    fig3.savefig(os.path.join(out_dir, "three_zone_distribution_plot.pdf"), bbox_inches="tight")
    plt.close(fig3)

    # 7. Subfolder Population
    subfolders = {
        "ZONE_1_GREEN_AUTOMATED_PASS": all_df[all_df["three_zone_assignment"] == "ZONE_1_PASS"],
        "ZONE_2_YELLOW_MANUAL_INSPECTION": all_df[all_df["three_zone_assignment"] == "ZONE_2_INSPECTION"],
        "ZONE_3_RED_AUTOMATED_REJECT": all_df[all_df["three_zone_assignment"] == "ZONE_3_REJECT"],
        "SINGLE_THRESHOLD_FALSE_POSITIVES": all_df[all_df["single_th_status"] == "FALSE_POSITIVE (High DSI Outlier)"],
        "SINGLE_THRESHOLD_FALSE_NEGATIVES": all_df[all_df["single_th_status"] == "FALSE_NEGATIVE (Low DSI Outlier)"],
        "EXCLUDED_TOPPERS_AND_ARTIFACTS": all_df[all_df["role"] == "excluded"],
    }

    for sub_name, sub_df in subfolders.items():
        sub_path = os.path.join(out_dir, sub_name)
        os.makedirs(sub_path, exist_ok=True)

        if sub_name == "SINGLE_THRESHOLD_FALSE_POSITIVES":
            sub_df_sorted = sub_df.sort_values(by="max_pair_mean_dsi", ascending=False)
        elif sub_name in ["SINGLE_THRESHOLD_FALSE_NEGATIVES", "ZONE_1_GREEN_AUTOMATED_PASS"]:
            sub_df_sorted = sub_df.sort_values(by="max_pair_mean_dsi", ascending=True)
        elif sub_name == "ZONE_3_RED_AUTOMATED_REJECT":
            sub_df_sorted = sub_df.sort_values(by="max_pair_mean_dsi", ascending=False)
        else:
            sub_df_sorted = sub_df.sort_values(by="max_pair_mean_dsi", ascending=True)

        sub_csv = os.path.join(sub_path, f"{sub_name.lower()}_summary.csv")
        sub_df_sorted.to_csv(sub_csv, index=False)

        # Copy stacked plots for image viewer browsing
        for tid in sub_df_sorted["tool_id"]:
            src_png = os.path.join(pixel_dir, f"{tid}_overlay_dsi_stacked.png")
            if os.path.exists(src_png):
                shutil.copy2(src_png, os.path.join(sub_path, f"{tid}_overlay_dsi_stacked.png"))

    print("\n=== Refined Outlier & 3-Zone Analysis Complete! ===")
    print(f"Output Directory: {out_dir}")

if __name__ == "__main__":
    main()
