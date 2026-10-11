#!/usr/bin/env python3
"""
Generate Publication-Grade LaTeX Statistical Distribution Table for DSI values.
Reads active exclusions directly from plot_max_dsi_three_zone_distribution.py,
computes non-parametric percentiles (Min, 25th, Median, 75th, 90th, Max),
and exports formatted .tex tables into reports/tables/.
"""

import os
import sys
import numpy as np
import pandas as pd

# Add repo root to import configuration and data loader
REPO_DIR = "/home/alifalah/Projects/Tool_Condition_Monitoring"
sys.path.insert(0, REPO_DIR)

import plot_max_dsi_three_zone_distribution as pmd

OUTPUT_DIRS = [
    "/home/alifalah/Projects/DATA/reports/tables",
    "/home/alifalah/Projects/Tool_Condition_Monitoring/reports/tables",
]

def compute_group_stats(vals, label):
    vals_clean = np.array(vals)[~np.isnan(vals)]
    return {
        "Group": label,
        "Count": len(vals_clean),
        "Min": np.min(vals_clean),
        "25th": np.percentile(vals_clean, 25),
        "Median": np.median(vals_clean),
        "75th": np.percentile(vals_clean, 75),
        "90th": np.percentile(vals_clean, 90),
        "Max": np.max(vals_clean)
    }

def generate_latex_table(stats_rows, caption="Statistical distribution of the DSI values across the healthy and fractured tool groups.", label="tab:dsi_statistics"):
    lines = []
    lines.append(r"\begin{table}[htpb]")
    lines.append(r"  \centering")
    lines.append(f"  \\caption{{{caption}}}")
    lines.append(r"  \small")
    lines.append(r"  \renewcommand{\arraystretch}{1.15}")
    lines.append(r"  \begin{tabular}{@{}")
    lines.append(r"    >{\raggedright\arraybackslash}p{3.2cm}")
    lines.append(r"    >{\centering\arraybackslash}p{1.1cm}")
    lines.append(r"    >{\centering\arraybackslash}p{1.3cm}")
    lines.append(r"    >{\centering\arraybackslash}p{1.5cm}")
    lines.append(r"    >{\centering\arraybackslash}p{1.3cm}")
    lines.append(r"    >{\centering\arraybackslash}p{1.5cm}")
    lines.append(r"    >{\centering\arraybackslash}p{1.5cm}")
    lines.append(r"    >{\centering\arraybackslash}p{1.4cm}")
    lines.append(r"    @{}}")
    lines.append(r"    \toprule")
    lines.append(r"    \textbf{Tool Group} &")
    lines.append(r"    \textbf{Count} &")
    lines.append(r"    \textbf{Min} &")
    lines.append(r"    \textbf{25th Perc.} &")
    lines.append(r"    \textbf{Median} &")
    lines.append(r"    \textbf{75th Perc.} &")
    lines.append(r"    \textbf{90th Perc.} &")
    lines.append(r"    \textbf{Max} \\")
    lines.append(r"    \midrule")
    lines.append("")

    for row in stats_rows:
        grp = row["Group"]
        cnt = row["Count"]
        p_min = f"${row['Min']:.2f}\\%$"
        p_25 = f"${row['25th']:.2f}\\%$"
        p_med = f"${row['Median']:.2f}\\%$"
        p_75 = f"${row['75th']:.2f}\\%$"
        p_90 = f"${row['90th']:.2f}\\%$"
        p_max = f"${row['Max']:.2f}\\%$"
        lines.append(f"    {grp:20s} & {cnt:2d} & {p_min:7s} & {p_25:7s}  & {p_med:7s} & {p_75:7s}  & {p_90:7s}  & {p_max:8s}   \\\\")

    lines.append("")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(f"  \\label{{{label}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines)

def main():
    df, active_excluded = pmd.load_tool_data()
    df_excluded = df[df["is_excluded"]].copy()
    df_active = df[~df["is_excluded"] & df["max_peak_dsi"].notna()].copy()

    # Calculate statistics
    h_vals = df_active[df_active["working_condition"].isin(["new", "used"])]["max_peak_dsi"].values
    f_vals = df_active[df_active["working_condition"] == "fractured"]["max_peak_dsi"].values

    new_vals = df_active[df_active["working_condition"] == "new"]["max_peak_dsi"].values
    used_vals = df_active[df_active["working_condition"] == "used"]["max_peak_dsi"].values
    dep_vals = df_active[df_active["working_condition"] == "deposit"]["max_peak_dsi"].values

    stats_main = [
        compute_group_stats(h_vals, "Healthy (New + Used)"),
        compute_group_stats(f_vals, "Fractured"),
    ]

    stats_extended = [
        compute_group_stats(h_vals, "Healthy (New + Used)"),
        compute_group_stats(new_vals, "  -- New"),
        compute_group_stats(used_vals, "  -- Used"),
        compute_group_stats(f_vals, "Fractured"),
        compute_group_stats(dep_vals, "Secondary: Deposit"),
    ]

    latex_main = generate_latex_table(stats_main, label="tab:dsi_statistics")
    latex_ext = generate_latex_table(stats_extended, caption="Extended statistical distribution of DSI values by tool condition.", label="tab:dsi_statistics_extended")

    # Save outputs to all directories
    for d in OUTPUT_DIRS:
        os.makedirs(d, exist_ok=True)
        main_path = os.path.join(d, "tab_dsi_statistics.tex")
        ext_path = os.path.join(d, "tab_dsi_statistics_extended.tex")
        with open(main_path, "w") as fp:
            fp.write(latex_main + "\n")
        with open(ext_path, "w") as fp:
            fp.write(latex_ext + "\n")
        print(f"Saved: {main_path}")
        print(f"Saved: {ext_path}")

    print("\n" + "="*70)
    print("LATEX CODE FOR tab_dsi_statistics.tex:")
    print("="*70)
    print(latex_main)
    print("="*70)

if __name__ == "__main__":
    main()
