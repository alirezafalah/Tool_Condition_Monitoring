"""
Paper Figure: Left-Right Summary (No Threshold)
==============================================
Creates a summary bar chart of mean asymmetry ratios by tool.
Threshold line is intentionally omitted.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION
# ============================================================================
CSV_PATH = r"c:\Users\alrfa\OneDrive - Eotvos Lorand Tudomanyegyetem Informatikai Kar\PhD\Dataset\CCD_DATA\DATA\threshold_analysis\left_right_method\left_right_analysis_results.csv"
OUTPUT_DIR = r"c:\Users\alrfa\OneDrive - Eotvos Lorand Tudomanyegyetem Informatikai Kar\PhD\Dataset\CCD_DATA\Tool_Condition_Monitoring\two_edge_case\paper_figures\summary_bar"
OUTPUT_BASENAME = "left_right_summary_no_threshold"
OUTPUT_FORMATS = ["png", "svg"]

# ============================================================================
# MAIN
# ============================================================================

def main():
    if not os.path.exists(CSV_PATH):
        print("CSV not found. Run analysis first.")
        return

    df = pd.read_csv(CSV_PATH)
    if df.empty:
        print("CSV is empty.")
        return

    # Sort by mean ratio
    df_sorted = df.sort_values("Mean Ratio", ascending=True)

    # Color by actual condition
    colors = []
    for _, row in df_sorted.iterrows():
        cond = str(row.get("Condition", "")).lower()
        if cond in ["fractured", "deposit"]:
            colors.append("red")
        elif cond == "used":
            colors.append("orange")
        elif cond == "new":
            colors.append("green")
        else:
            colors.append("gray")

    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    })

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.bar(df_sorted["Tool ID"], df_sorted["Mean Ratio"], color=colors,
           alpha=0.7, edgecolor="black")

    ax.set_title("Left-Right Symmetry Analysis: Mean Asymmetry Ratio by Tool")
    ax.set_ylabel("Mean Asymmetry Ratio")
    ax.set_xlabel("Tool ID")
    plt.xticks(rotation=45, ha="right")

    # Legend for conditions
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="green", alpha=0.7, label="New"),
        Patch(facecolor="orange", alpha=0.7, label="Used"),
        Patch(facecolor="red", alpha=0.7, label="Fractured/Deposit"),
        Patch(facecolor="gray", alpha=0.7, label="Unknown"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for fmt in OUTPUT_FORMATS:
        out_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_BASENAME}.{fmt}")
        fig.savefig(out_path, format=fmt, dpi=300)

    plt.close(fig)
    print(f"Saved: {OUTPUT_BASENAME}.png/.svg")


if __name__ == "__main__":
    main()
