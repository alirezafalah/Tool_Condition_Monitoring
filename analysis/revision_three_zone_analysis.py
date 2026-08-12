#!/usr/bin/env python3
"""Create the three-zone DSI revision package for the two-edge dataset.

The BUE specimen remains in descriptive outputs, but is excluded before the
Functional-versus-Fractured calibration.  The safe/pass boundary is a
non-parametric 90th-percentile Functional reference limit; the fracture-alarm
boundary is selected by ROC/Youden analysis.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy import stats

from revision_dsi_analysis import (
    COLORS,
    auc_from_scores,
    bootstrap_auc,
    discover_results,
    roc_points,
    save_figure,
    wilson_interval,
)


SAFE_COLOR = "#66A61E"
WARNING_COLOR = "#F1B944"
ALARM_COLOR = "#D9534F"


def empirical_upper_quantile(values: np.ndarray, q: float) -> float:
    """Return an observed-value upper empirical quantile.

    ``method='higher'`` deliberately uses an observed Functional DSI as the
    boundary. With 12 Functional samples, q=.90 leaves 11/12 at or below the
    pass boundary and avoids an interpolated, unobserved decision limit.
    """
    try:
        return float(np.quantile(values, q, method="higher"))
    except TypeError:  # NumPy < 1.22 compatibility
        return float(np.quantile(values, q, interpolation="higher"))


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    lower = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2, k, n - k + 1))
    upper = 1.0 if k == n else float(stats.beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lower, upper


def assign_zone(dsi: float, safe_threshold: float, fracture_threshold: float) -> str:
    if dsi <= safe_threshold:
        return "Pass"
    if dsi < fracture_threshold:
        return "Warning"
    return "Fracture alarm"


def plot_three_zone_bar(
    all_tools: pd.DataFrame,
    safe_threshold: float,
    fracture_threshold: float,
    destination: Path,
) -> None:
    """DSI-sorted descriptive chart with a linear zoom and full-scale panel."""
    ordered = all_tools.sort_values(["DSI_percent", "tool_id"]).reset_index(drop=True)
    x = np.arange(len(ordered))
    colors = ordered["diagnostic_group"].map(COLORS)
    highest = float(ordered["DSI_percent"].max())
    zoom_max = max(3.0, fracture_threshold * 1.55)

    fig, (ax_high, ax_low) = plt.subplots(
        2,
        1,
        figsize=(12.5, 8.0),
        sharex=True,
        gridspec_kw={"height_ratios": [1.25, 1.7], "hspace": 0.06},
    )
    for ax in (ax_high, ax_low):
        ax.bar(x, ordered["DSI_percent"], color=colors, edgecolor="black", linewidth=0.45)
        ax.grid(axis="y", alpha=0.24)

    # Full-scale panel: preserves the severe-fracture magnitudes without a log transform.
    ax_high.set_ylim(zoom_max, highest * 1.06)
    ax_high.set_ylabel("DSI (%)")
    ax_high.set_title("Three-Zone Dimensionless Symmetry Index (DSI) Framework", fontweight="bold")

    # Expanded lower panel: makes the pass/warning/alarm thresholds legible.
    ax_low.axhspan(0, safe_threshold, color=SAFE_COLOR, alpha=0.13, zorder=0)
    ax_low.axhspan(safe_threshold, fracture_threshold, color=WARNING_COLOR, alpha=0.17, zorder=0)
    ax_low.axhspan(fracture_threshold, zoom_max, color=ALARM_COLOR, alpha=0.10, zorder=0)
    ax_low.axhline(safe_threshold, color="#4C7F18", linestyle="--", linewidth=1.5)
    ax_low.axhline(fracture_threshold, color="#B43C39", linestyle="--", linewidth=1.5)
    ax_low.set_ylim(0, zoom_max)
    ax_low.set_ylabel("DSI (%)")
    ax_low.set_xlabel("Tool specimen (ascending DSI)")
    ax_low.set_xticks(x, ordered["tool_id"], rotation=55)
    ax_low.text(len(x) - 0.45, safe_threshold / 2, "Pass zone", ha="right", va="center", color="#356111")
    ax_low.text(
        len(x) - 0.45,
        (safe_threshold + fracture_threshold) / 2,
        "Warning zone",
        ha="right",
        va="center",
        color="#725500",
    )
    ax_low.text(
        len(x) - 0.45,
        (fracture_threshold + zoom_max) / 2,
        "Fracture-alarm zone",
        ha="right",
        va="center",
        color="#8A2927",
    )

    # Diagonal marks identify the omitted middle section of the linear y-axis.
    marker = 0.012
    ax_high.plot((-marker, +marker), (-marker, +marker), transform=ax_high.transAxes, color="black", clip_on=False)
    ax_high.plot((1 - marker, 1 + marker), (-marker, +marker), transform=ax_high.transAxes, color="black", clip_on=False)
    ax_low.plot((-marker, +marker), (1 - marker, 1 + marker), transform=ax_low.transAxes, color="black", clip_on=False)
    ax_low.plot((1 - marker, 1 + marker), (1 - marker, 1 + marker), transform=ax_low.transAxes, color="black", clip_on=False)

    legend_handles = [
        Patch(facecolor=COLORS[group], edgecolor="black", label=group)
        for group in ("Functional", "Fractured", "BUE")
    ] + [
        Line2D([0], [0], color="#4C7F18", linestyle="--", label=f"Pass limit = {safe_threshold:.3f}%"),
        Line2D([0], [0], color="#B43C39", linestyle="--", label=f"Fracture limit = {fracture_threshold:.3f}%"),
    ]
    ax_high.legend(handles=legend_handles, ncol=2, frameon=True, fontsize=9, loc="upper left")
    save_figure(fig, destination)


def plot_functional_ecdf(
    functional: np.ndarray,
    bue: pd.DataFrame,
    safe_threshold: float,
    destination: Path,
) -> None:
    values = np.sort(functional)
    y = np.arange(1, len(values) + 1) / len(values)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.step(values, y, where="post", linewidth=2.0, color=COLORS["Functional"], label="Functional empirical CDF")
    ax.scatter(values, y, color=COLORS["Functional"], edgecolor="white", linewidth=.7, s=52, zorder=3)
    ax.axvline(safe_threshold, color="#4C7F18", linestyle="--", linewidth=1.5,
               label=f"90th-percentile pass limit = {safe_threshold:.3f}%")
    ax.axhline(.90, color="gray", linestyle=":", linewidth=1.2, label="90% reference quantile")
    if not bue.empty:
        bue_dsi = float(bue.iloc[0].DSI_percent)
        ax.scatter(
            [bue_dsi], [1.02], marker="*", s=180, color=COLORS["BUE"], edgecolor="black",
            linewidth=.5, zorder=4, label=f"BUE tool069 = {bue_dsi:.3f}%",
        )
    ax.set(xlabel="DSI (%)", ylabel="Empirical cumulative proportion", ylim=(0, 1.10))
    ax.grid(alpha=.25)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_title("Functional baseline used to define the pass limit", fontweight="bold")
    save_figure(fig, destination)


def plot_roc(roc: pd.DataFrame, threshold: float, auc: float, auc_ci: tuple[float, float], destination: Path) -> None:
    points = roc.sort_values(["false_positive_rate", "sensitivity"])
    best = roc.loc[np.isclose(roc.threshold_DSI_percent, threshold)].iloc[0]
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.step(points.false_positive_rate, points.sensitivity, where="post", color="#6A3D9A", linewidth=2)
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.scatter(best.false_positive_rate, best.sensitivity, color=ALARM_COLOR, s=70, zorder=3,
               label=f"Youden fracture limit = {threshold:.3f}%")
    ax.set(xlabel="False-positive rate (1 − specificity)", ylabel="True-positive rate (sensitivity)",
           xlim=(-.02, 1.02), ylim=(-.02, 1.02))
    ax.set_title(f"ROC curve: AUC = {auc:.3f} (95% bootstrap CI {auc_ci[0]:.3f}–{auc_ci[1]:.3f})")
    ax.grid(alpha=.25)
    ax.legend(loc="lower right", frameon=False)
    save_figure(fig, destination)


def plot_zone_disposition(zone_counts: pd.DataFrame, destination: Path) -> None:
    order = ["Functional", "BUE", "Fractured"]
    zones = ["Pass", "Warning", "Fracture alarm"]
    matrix = zone_counts.reindex(index=order, columns=zones, fill_value=0)
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    left = np.zeros(len(matrix))
    zone_colors = {"Pass": SAFE_COLOR, "Warning": WARNING_COLOR, "Fracture alarm": ALARM_COLOR}
    for zone in zones:
        values = matrix[zone].to_numpy()
        ax.bar(matrix.index, values, bottom=left, label=zone, color=zone_colors[zone], edgecolor="black", linewidth=.5)
        for i, value in enumerate(values):
            if value:
                ax.text(i, left[i] + value / 2, str(int(value)), ha="center", va="center", fontweight="bold")
        left += values
    ax.set_ylabel("Number of tools")
    ax.set_title("Three-zone disposition by known condition", fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.22)
    save_figure(fig, destination)


def write_zone_table(path: Path, zone_counts: pd.DataFrame, safe_threshold: float, fracture_threshold: float) -> None:
    counts = zone_counts.reindex(index=["Functional", "BUE", "Fractured"], columns=["Pass", "Warning", "Fracture alarm"], fill_value=0)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Three-zone disposition of the available two-edge tool specimens. Tool069 (BUE) was not used to calibrate either boundary.}",
        "\\label{tab:three_zone_disposition}",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Known condition & Pass & Warning & Fracture alarm \\\\",
        "\\midrule",
    ]
    for group in counts.index:
        row = counts.loc[group]
        lines.append(f"{group} & {int(row['Pass'])} & {int(row['Warning'])} & {int(row['Fracture alarm'])} \\\\")
    lines.extend([
        "\\bottomrule",
        "\\multicolumn{4}{l}{\\footnotesize Pass: DSI $\\leq$ %.3f\\%%; fracture alarm: DSI $\\geq$ %.3f\\%%.} \\\\" % (safe_threshold, fracture_threshold),
        "\\end{tabular}",
        "\\end{table}",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_per_tool_table(path: Path, all_tools: pd.DataFrame) -> None:
    """Write the descriptive table used beside the three-zone bar chart."""
    table = all_tools.copy()
    table["tool_number"] = table.tool_id.str.extract(r"(\d+)")[0].astype(int)
    table = table.sort_values("tool_number")
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Dimensionless Symmetry Index (DSI) and three-zone disposition for each available two-edge tool specimen.}",
        "\\label{tab:per_tool_dsi_three_zone}",
        "\\small",
        "\\setlength{\\tabcolsep}{7pt}",
        "\\renewcommand{\\arraystretch}{1.08}",
        "\\begin{tabular}{lllllr}",
        "\\toprule",
        "Tool ID & Type & Diameter (mm) & Color & Condition & DSI (\\%) \\\\",
        "\\midrule",
    ]
    for row in table.itertuples():
        condition = "BUE$^{*}$" if row.diagnostic_group == "BUE" else row.diagnostic_group
        tool_id = str(row.tool_id).replace("tool", "")
        tool_type = {
            "drill": "Drill",
            "endmill": "End mill",
            "central_drill": "Central drill",
        }.get(str(row.type), str(row.type).replace("_", " ").capitalize())
        color = str(row.color).capitalize()
        lines.append(
            f"{tool_id} & {tool_type} & {float(row.diameter_mm):.1f} & {color} & {condition} & {float(row.DSI_percent):.3f} \\\\")
    lines.extend([
        "\\bottomrule",
        "\\multicolumn{6}{l}{\\footnotesize $^{*}$BUE: built-up edge; retained descriptively but excluded from binary threshold calibration.} \\\\",
        "\\end{tabular}",
        "\\end{table*}",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_latex(path: Path, results: dict, missing: pd.DataFrame) -> None:
    missing_ids = ", ".join(missing.tool_id.tolist()) if not missing.empty else "none"
    text = rf"""% Auto-generated candidate text for the three-zone DSI revision.
% Requires: \usepackage{{booktabs}}
\subsection{{Three-Zone DSI Decision Framework}}
The binary Functional-versus-Fractured threshold is useful for detecting severe asymmetric material loss,
but it should not be interpreted as a guarantee that every tool below that boundary is operational. In
particular, specimen 069 exhibited built-up edge (BUE), a failure mode that can preserve the apparent
two-fold silhouette symmetry. It therefore produced DSI={results['bue_dsi']:.3f}\%, close to the
Functional observations, despite being unsuitable for normal operation. Specimen 069 was retained in
the descriptive figures as a separate BUE category but excluded a priori from calibration of the binary
Functional-versus-Fractured boundaries.

Accordingly, a three-zone decision framework was defined. The pass limit was the upper empirical 90th
percentile of the Functional reference DSI distribution, $T_\mathrm{{pass}}={results['safe_threshold']:.3f}\%$.
An observed-value (``higher'') empirical quantile was used, so the limit is conservative with respect to
the finite Functional sample rather than being an interpolated value. The 90th percentile is an explicit
screening policy corresponding to an approximately 10\% expected Functional warning rate, rather than a
universal safety limit. The fracture-alarm limit was
obtained independently by ROC analysis and maximum Youden's index,
$T_\mathrm{{fracture}}={results['fracture_threshold']:.3f}\%$. Thus,
\begin{{equation}}
\mathrm{{zone}}(\mathrm{{DSI}})=
\begin{{cases}}
\text{{Pass}}, & \mathrm{{DSI}}\leq T_\mathrm{{pass}},\\
\text{{Warning}}, & T_\mathrm{{pass}}<\mathrm{{DSI}}<T_\mathrm{{fracture}},\\
\text{{Fracture alarm}}, & \mathrm{{DSI}}\geq T_\mathrm{{fracture}}.
\end{{cases}}
\end{{equation}}
The pass zone supports automated acceptance within the observed Functional baseline; the warning zone
requires secondary inspection, such as human review or a complementary image feature; and the
fracture-alarm zone triggers a non-operational decision. This terminology is deliberately conservative:
the pass limit is an empirical screening boundary, not a universal safety guarantee.

\begin{{figure}}[t]
\centering
\includegraphics[width=\linewidth]{{phase1_three_zone_summary/three_zone_dsi_by_tool.pdf}}
\caption{{DSI values in ascending order. The lower panel expands the pass, warning, and fracture-alarm
zones without a logarithmic axis; the upper panel preserves the magnitude of severe fractures. Tool069
(BUE) lies in the warning zone.}}
\label{{fig:three_zone_dsi}}
\end{{figure}}

\subsection{{Statistical Basis of the Decision Limits}}
The Functional reference sample was non-normal by Shapiro--Wilk testing
($W={results['shapiro_functional_W']:.3f}$, $p={results['shapiro_functional_p']:.4f}$); therefore, the
pass boundary was derived non-parametrically. {results['functional_pass_count']} of
{results['n_functional']} Functional specimens ({results['functional_pass_rate']:.1f}\%) were in the
pass zone (exact 95\% binomial CI {results['functional_pass_ci_low']:.3f}--{results['functional_pass_ci_high']:.3f}).
The BUE specimen fell in the warning zone rather than the pass zone. For the independent binary
Functional-versus-Fractured comparison (excluding BUE), a two-sided exact Mann--Whitney U test showed
strong separation ($U={results['mann_whitney_U']:.1f}$, $p={results['mann_whitney_p']:.4g}$,
$r_{{rb}}={results['rank_biserial']:.3f}$). ROC analysis yielded AUC={results['auc']:.3f}
(95\% stratified-bootstrap CI {results['auc_ci_low']:.3f}--{results['auc_ci_high']:.3f}) and selected
$T_\mathrm{{fracture}}={results['fracture_threshold']:.3f}\%$ by maximum Youden's index
($J={results['youden_J']:.3f}$).

\begin{{figure}}[t]
\centering
\includegraphics[width=.75\linewidth]{{phase2_statistical_validation/functional_ecdf_pass_limit.pdf}}
\caption{{Functional empirical cumulative distribution and the non-parametric pass limit. The BUE value
is shown for descriptive comparison only and was not used to estimate the limit.}}
\label{{fig:functional_pass_limit}}
\end{{figure}}

\begin{{figure}}[t]
\centering
\includegraphics[width=.75\linewidth]{{phase3_threshold_determination/roc_curve_fracture_limit.pdf}}
\caption{{ROC curve for Functional-versus-Fractured calibration, excluding the BUE specimen.}}
\label{{fig:three_zone_roc}}
\end{{figure}}

\input{{three_zone_disposition_table}}

In this dataset, all {results['n_fractured']} conventionally fractured tools were assigned to the
fracture-alarm zone, while {results['functional_warning_count']} Functional tool was assigned to the
warning zone and no Functional tool was assigned to the fracture-alarm zone. The resulting fracture-alarm
performance was sensitivity={results['sensitivity']:.3f}, specificity={results['specificity']:.3f}, and
accuracy={results['accuracy']:.3f}; these are apparent in-sample estimates and require independent
validation. Tools {missing_ids} were unavailable because no final pixel-comparison output existed.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--masks-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output
    phase1 = output / "phase1_three_zone_summary"
    phase2 = output / "phase2_statistical_validation"
    phase3 = output / "phase3_threshold_determination"
    phase4 = output / "phase4_operational_performance"
    for folder in (phase1, phase2, phase3, phase4):
        folder.mkdir(parents=True, exist_ok=True)

    all_tools, missing = discover_results(args.metadata, args.masks_root)
    bue = all_tools.loc[all_tools.diagnostic_group == "BUE"].copy()
    calibration = all_tools.loc[all_tools.diagnostic_group != "BUE"].copy()
    functional = calibration.loc[calibration.diagnostic_group == "Functional", "DSI_percent"].to_numpy()
    fractured = calibration.loc[calibration.diagnostic_group == "Fractured", "DSI_percent"].to_numpy()
    if len(functional) < 3 or len(fractured) < 3 or len(bue) != 1:
        raise ValueError("Expected >=3 Functional, >=3 Fractured, and exactly one BUE tool.")

    safe_threshold = empirical_upper_quantile(functional, .90)
    y_true = (calibration.diagnostic_group == "Fractured").astype(int).to_numpy()
    scores = calibration.DSI_percent.to_numpy()
    roc = roc_points(y_true, scores)
    best = roc.loc[roc.youden_J == roc.youden_J.max()].sort_values("threshold_DSI_percent", ascending=False).iloc[0]
    fracture_threshold = float(best.threshold_DSI_percent)
    auc = auc_from_scores(y_true, scores)
    auc_ci = tuple(np.quantile(bootstrap_auc(y_true, scores), [.025, .975]))

    all_tools = all_tools.copy()
    all_tools["three_zone"] = all_tools.DSI_percent.map(
        lambda x: assign_zone(float(x), safe_threshold, fracture_threshold)
    )
    all_tools["included_in_binary_calibration"] = all_tools.diagnostic_group != "BUE"
    all_tools["binary_calibration_status"] = np.where(
        all_tools.diagnostic_group == "BUE", "Excluded: BUE", "Included"
    )

    pass_count = int(((calibration.diagnostic_group == "Functional") & (all_tools.loc[calibration.index, "three_zone"] == "Pass")).sum())
    pass_ci = clopper_pearson(pass_count, len(functional))
    zone_counts = pd.crosstab(all_tools.diagnostic_group, all_tools.three_zone)

    # Binary fracture-alarm performance, calculated only on the pre-specified calibration groups.
    predicted_alarm = calibration.DSI_percent >= fracture_threshold
    actual_fracture = calibration.diagnostic_group == "Fractured"
    tp = int((predicted_alarm & actual_fracture).sum())
    fn = int((~predicted_alarm & actual_fracture).sum())
    tn = int((~predicted_alarm & ~actual_fracture).sum())
    fp = int((predicted_alarm & ~actual_fracture).sum())
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    accuracy = (tp + tn) / len(calibration)
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0
    mw = stats.mannwhitneyu(fractured, functional, alternative="two-sided", method="exact")
    rank_biserial = 2 * float(mw.statistic) / (len(functional) * len(fractured)) - 1
    shapiro = stats.shapiro(functional)

    plot_three_zone_bar(all_tools, safe_threshold, fracture_threshold, phase1 / "three_zone_dsi_by_tool")
    plot_functional_ecdf(functional, bue, safe_threshold, phase2 / "functional_ecdf_pass_limit")
    plot_roc(roc, fracture_threshold, auc, auc_ci, phase3 / "roc_curve_fracture_limit")
    plot_zone_disposition(zone_counts, phase4 / "three_zone_disposition")

    all_tools.sort_values("DSI_percent").to_csv(phase1 / "per_tool_dsi_three_zone.csv", index=False)
    missing.to_csv(output / "missing_pixel_comparison_outputs.csv", index=False)
    all_tools.loc[all_tools.diagnostic_group == "BUE", [
        "tool_id", "condition", "DSI_percent", "three_zone"
    ]].to_csv(output / "bue_descriptive_exclusion.csv", index=False)
    pd.DataFrame([
        {
            "limit": "Pass limit",
            "DSI_percent": safe_threshold,
            "method": "90th percentile of Functional DSI, observed-value higher empirical quantile",
            "calibration_tools": len(functional),
        },
        {
            "limit": "Fracture-alarm limit",
            "DSI_percent": fracture_threshold,
            "method": "ROC threshold maximizing Youden's index (Functional vs Fractured; BUE excluded)",
            "calibration_tools": len(calibration),
        },
    ]).to_csv(phase3 / "three_zone_thresholds.csv", index=False)
    pd.DataFrame([
        {"test": "Shapiro-Wilk, Functional DSI", "statistic": float(shapiro.statistic), "p_value": float(shapiro.pvalue), "n": len(functional)},
        {"test": "Mann-Whitney U, Fractured vs Functional", "statistic": float(mw.statistic), "p_value": float(mw.pvalue), "n": len(calibration), "rank_biserial": rank_biserial},
    ]).to_csv(phase2 / "statistical_tests.csv", index=False)
    zone_counts.reindex(index=["Functional", "BUE", "Fractured"], columns=["Pass", "Warning", "Fracture alarm"], fill_value=0).to_csv(phase4 / "three_zone_disposition.csv")
    metric_rows = []
    for label, value, ci in (
        ("Sensitivity", sensitivity, wilson_interval(tp, tp + fn)),
        ("Specificity", specificity, wilson_interval(tn, tn + fp)),
        ("Accuracy", accuracy, wilson_interval(tp + tn, len(calibration))),
        ("Precision", precision, wilson_interval(tp, tp + fp)),
        ("F1-score", f1, (f1, f1)),
    ):
        metric_rows.append({"metric": label, "value": value, "ci_low": ci[0], "ci_high": ci[1]})
    pd.DataFrame(metric_rows).to_csv(phase4 / "fracture_alarm_performance.csv", index=False)
    pd.DataFrame([
        {"actual": "Functional", "predicted_no_alarm": tn, "predicted_fracture_alarm": fp},
        {"actual": "Fractured", "predicted_no_alarm": fn, "predicted_fracture_alarm": tp},
    ]).to_csv(phase4 / "fracture_alarm_confusion_matrix.csv", index=False)

    results = {
        "n_descriptive_total": int(len(all_tools)),
        "n_calibration": int(len(calibration)),
        "n_functional": int(len(functional)),
        "n_fractured": int(len(fractured)),
        "bue_dsi": float(bue.iloc[0].DSI_percent),
        "safe_threshold": safe_threshold,
        "fracture_threshold": fracture_threshold,
        "functional_pass_count": pass_count,
        "functional_pass_rate": 100 * pass_count / len(functional),
        "functional_pass_ci_low": pass_ci[0],
        "functional_pass_ci_high": pass_ci[1],
        "functional_warning_count": int(len(functional) - pass_count),
        "shapiro_functional_W": float(shapiro.statistic),
        "shapiro_functional_p": float(shapiro.pvalue),
        "mann_whitney_U": float(mw.statistic),
        "mann_whitney_p": float(mw.pvalue),
        "rank_biserial": rank_biserial,
        "auc": auc,
        "auc_ci_low": float(auc_ci[0]),
        "auc_ci_high": float(auc_ci[1]),
        "youden_J": float(best.youden_J),
        "TP": tp, "FN": fn, "TN": tn, "FP": fp,
        "sensitivity": sensitivity, "specificity": specificity, "accuracy": accuracy,
        "precision": precision, "f1": f1,
    }
    (output / "three_zone_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_zone_table(output / "three_zone_disposition_table.tex", zone_counts, safe_threshold, fracture_threshold)
    write_per_tool_table(output / "per_tool_dsi_three_zone_table.tex", all_tools)
    write_latex(output / "three_zone_revision_sections.tex", results, missing)
    shutil.copy2(Path(__file__), output / "revision_three_zone_analysis.py")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
