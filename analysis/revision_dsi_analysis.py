#!/usr/bin/env python3
"""Generate the DSI/statistical revision package for the two-edge dataset."""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


FUNCTIONAL_CONDITIONS = {"new", "used"}
RNG = np.random.default_rng(20260805)
COLORS = {"Functional": "#2878B5", "Fractured": "#D9534F", "BUE": "#E69F00"}


def classify_condition(value: str) -> str:
    condition = str(value).strip().lower()
    if condition in FUNCTIONAL_CONDITIONS:
        return "Functional"
    if "bue" in condition:
        return "BUE"
    if "fractured" in condition or condition == "worn":
        return "Fractured"
    raise ValueError(f"Unmapped condition: {value!r}")


def discover_results(metadata_csv: Path, masks_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = pd.read_csv(metadata_csv)
    included, excluded = [], []
    for record in metadata.to_dict("records"):
        tool_id = str(record["tool_id"])
        matches = sorted(glob.glob(str(
            masks_root / f"{tool_id}*" / "half_tool_analysis" / "pixel_comparison"
            / f"{tool_id}_right_side_pixel_counts.csv"
        )))
        if not matches:
            excluded.append({
                "tool_id": tool_id,
                "condition": record["condition"],
                "reason": "No final pixel_comparison output",
            })
            continue
        if len(matches) > 1:
            raise RuntimeError(f"Multiple final outputs found for {tool_id}: {matches}")

        counts_path = Path(matches[0])
        result_dir = counts_path.parent
        metadata_path = result_dir / f"{tool_id}_symmetry_metadata.json"
        counts = pd.read_csv(counts_path)
        analysis_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        processed_cols = sorted(
            (c for c in counts.columns if c.startswith("processed_count_r")),
            key=lambda c: int(c.rsplit("r", 1)[1]),
        )
        if len(processed_cols) != 2:
            raise ValueError(f"{tool_id}: expected exactly two processed region columns")
        if len(counts) != 90:
            raise ValueError(f"{tool_id}: expected 90 paired samples, found {len(counts)}")

        p1 = counts[processed_cols[0]].to_numpy(dtype=float)
        p2 = counts[processed_cols[1]].to_numpy(dtype=float)
        differences = np.abs(p1 - p2)
        d_bar = float(differences.mean())
        # Mean right-half profile area pooled across both evaluated phase ranges.
        p_bar = float(np.concatenate([p1, p2]).mean())
        dsi = 100.0 * d_bar / p_bar

        included.append({
            **record,
            "diagnostic_group": classify_condition(record["condition"]),
            "mean_abs_difference_px": d_bar,
            "mean_profile_area_px": p_bar,
            "DSI_percent": dsi,
            "n_angle_pairs": len(counts),
            "roi_height_px": analysis_meta.get("roi_height_px"),
            "frame_regions": "; ".join(analysis_meta.get("internal_regions", [])),
            "smoothing_enabled": analysis_meta.get("smoothing_enabled", False),
            "smoothing_window": analysis_meta.get("smoothing_window"),
            "smoothing_strength": analysis_meta.get("smoothing_strength"),
            "counts_path": str(counts_path),
        })
    return pd.DataFrame(included), pd.DataFrame(excluded)


def roc_points(y_true: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    unique = np.unique(scores)
    thresholds = np.r_[-np.inf, (unique[:-1] + unique[1:]) / 2.0, np.inf]
    rows = []
    for threshold in thresholds:
        predicted = scores > threshold
        tp = int(np.sum((y_true == 1) & predicted))
        fn = int(np.sum((y_true == 1) & ~predicted))
        tn = int(np.sum((y_true == 0) & ~predicted))
        fp = int(np.sum((y_true == 0) & predicted))
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        rows.append({
            "threshold_DSI_percent": threshold,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "false_positive_rate": 1.0 - specificity,
            "youden_J": sensitivity + specificity - 1.0,
            "TP": tp, "FN": fn, "TN": tn, "FP": fp,
        })
    return pd.DataFrame(rows)


def auc_from_scores(y_true: np.ndarray, scores: np.ndarray) -> float:
    positive = scores[y_true == 1]
    negative = scores[y_true == 0]
    comparisons = positive[:, None] - negative[None, :]
    return float((np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0)) / comparisons.size)


def bootstrap_auc(y_true: np.ndarray, scores: np.ndarray, iterations: int = 10_000) -> np.ndarray:
    positive = scores[y_true == 1]
    negative = scores[y_true == 0]
    values = np.empty(iterations)
    for i in range(iterations):
        p = RNG.choice(positive, size=len(positive), replace=True)
        n = RNG.choice(negative, size=len(negative), replace=True)
        values[i] = auc_from_scores(
            np.r_[np.ones(len(p), dtype=int), np.zeros(len(n), dtype=int)], np.r_[p, n]
        )
    return values


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return center - margin, center + margin


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def scatter_jitter(ax, x: float, values: np.ndarray, color: str) -> None:
    jitter = np.linspace(-0.075, 0.075, len(values)) if len(values) > 1 else np.array([0.0])
    ax.scatter(x + jitter, values, s=48, color=color, edgecolor="white", linewidth=0.7, zorder=3)


def generate_figures(all_tools: pd.DataFrame, df: pd.DataFrame, roc: pd.DataFrame, threshold: float, auc: float,
                     auc_ci: tuple[float, float], metrics: pd.DataFrame, output: Path,
                     mann_p: float) -> None:
    phase1 = output / "phase1_dsi"
    phase2 = output / "phase2_significance"
    phase3 = output / "phase3_roc_threshold"
    phase4 = output / "phase4_classification_metrics"
    for folder in (phase1, phase2, phase3, phase4):
        folder.mkdir(parents=True, exist_ok=True)

    # Descriptive chart includes the BUE specimen, but inferential analyses below do not.
    ordered = all_tools.copy()
    ordered = ordered.sort_values(["DSI_percent", "tool_id"])
    fig, ax = plt.subplots(figsize=(12, 5.6))
    colors = ordered["diagnostic_group"].map(COLORS)
    ax.bar(ordered["tool_id"], ordered["DSI_percent"], color=colors,
           edgecolor="black", linewidth=.35)
    ax.set_ylabel("Dimensionless Symmetry Index, DSI (%)")
    ax.set_xlabel("Tool specimen")
    ax.tick_params(axis="x", rotation=55)
    from matplotlib.patches import Patch
    ax.legend(
        handles=[Patch(facecolor=COLORS[group], edgecolor="black", label=group)
                 for group in ("Functional", "Fractured", "BUE")],
        frameon=False,
    )
    ax.grid(axis="y", alpha=.25)
    save_figure(fig, phase1 / "dsi_by_tool")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, metric, ylabel in (
        (axes[0], "mean_abs_difference_px", r"Mean absolute difference, $\overline{D}$ (pixels)"),
        (axes[1], "DSI_percent", "DSI (%)"),
    ):
        for group in ("Functional", "Fractured"):
            part = df[df.diagnostic_group == group]
            ax.scatter(part.diameter_mm, part[metric], label=group, color=COLORS[group], s=55,
                       edgecolor="white", linewidth=.7)
        rho, p = stats.spearmanr(df.diameter_mm, df[metric])
        ax.set_xlabel("Nominal tool diameter (mm)")
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        ax.text(.03, .96, rf"Spearman $\rho$={rho:.2f}, $p$={p:.3f}", transform=ax.transAxes,
                va="top")
        ax.grid(alpha=.25)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, phase1 / "absolute_and_normalized_vs_diameter")

    groups = [df.loc[df.diagnostic_group == g, "DSI_percent"].to_numpy() for g in ("Functional", "Fractured")]
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    bp = ax.boxplot(groups, positions=[1, 2], widths=.5, patch_artist=True, showfliers=False)
    for patch, group in zip(bp["boxes"], ("Functional", "Fractured")):
        patch.set_facecolor(COLORS[group]); patch.set_alpha(.35)
    scatter_jitter(ax, 1, groups[0], COLORS["Functional"])
    scatter_jitter(ax, 2, groups[1], COLORS["Fractured"])
    ax.set_xticks([1, 2], [f"Functional\n(n={len(groups[0])})", f"Fractured\n(n={len(groups[1])})"])
    ax.set_ylabel("DSI (%)")
    ax.set_yscale("log")
    ax.set_title(f"Mann–Whitney U test: p = {mann_p:.4g}")
    ax.grid(axis="y", alpha=.25)
    save_figure(fig, phase2 / "dsi_group_comparison")

    roc_plot = roc.sort_values(["false_positive_rate", "sensitivity"])
    best = roc.loc[np.isclose(roc.threshold_DSI_percent, threshold)].iloc[0]
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.step(roc_plot.false_positive_rate, roc_plot.sensitivity, where="post", color="#6A3D9A", linewidth=2)
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.scatter(best.false_positive_rate, best.sensitivity, color="#D9534F", s=70, zorder=3,
               label=f"Youden optimum: {threshold:.3f}%")
    ax.set(xlabel="False-positive rate (1 − specificity)", ylabel="True-positive rate (sensitivity)",
           xlim=(-.02, 1.02), ylim=(-.02, 1.02))
    ax.set_title(f"ROC curve: AUC = {auc:.3f} (95% bootstrap CI {auc_ci[0]:.3f}–{auc_ci[1]:.3f})")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(alpha=.25)
    save_figure(fig, phase3 / "roc_curve_youden")

    finite = roc[np.isfinite(roc.threshold_DSI_percent)].sort_values("threshold_DSI_percent")
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.plot(finite.threshold_DSI_percent, finite.youden_J, marker="o", markersize=3, color="#2A9D8F")
    ax.axvline(threshold, color="black", linestyle="--", label=f"Maximum J at {threshold:.3f}%")
    ax.set(xlabel="Candidate DSI threshold (%)", ylabel="Youden's J")
    ax.legend(frameon=False); ax.grid(alpha=.25)
    save_figure(fig, phase3 / "youden_index_by_threshold")

    confusion = np.array([[int(best.TN), int(best.FP)], [int(best.FN), int(best.TP)]])
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    image = ax.imshow(confusion, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(confusion[i, j]), ha="center", va="center", fontsize=18,
                    color="white" if confusion[i, j] > confusion.max()/2 else "black")
    ax.set_xticks([0, 1], ["Functional", "Fractured"])
    ax.set_yticks([0, 1], ["Functional", "Fractured"])
    ax.set_xlabel("Predicted class"); ax.set_ylabel("Ground-truth class")
    ax.set_title(f"Confusion matrix at DSI > {threshold:.3f}%")
    fig.colorbar(image, ax=ax, fraction=.046, pad=.04)
    save_figure(fig, phase4 / "confusion_matrix")

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    x = np.arange(len(metrics))
    values = metrics.value.to_numpy()
    # Wilson bounds can differ from an exact boundary estimate (for example,
    # sensitivity=1) by ~1e-16 due to floating-point rounding.  Matplotlib
    # rejects even these numerically insignificant negative error lengths.
    lower = np.maximum(0.0, values - metrics.ci_low.to_numpy())
    upper = np.maximum(0.0, metrics.ci_high.to_numpy() - values)
    ax.bar(x, values, color=["#D9534F", "#2878B5", "#4C956C", "#F4A261", "#6A3D9A"][:len(x)])
    ax.errorbar(x, values, yerr=np.vstack([lower, upper]), fmt="none", ecolor="black", capsize=4)
    ax.set_xticks(x, metrics.metric, rotation=20)
    ax.set_ylim(0, 1.08); ax.set_ylabel("Performance")
    ax.grid(axis="y", alpha=.25)
    save_figure(fig, phase4 / "classification_metrics")


def write_latex(path: Path, results: dict, excluded: pd.DataFrame) -> None:
    excluded_ids = ", ".join(excluded.tool_id.tolist()) if not excluded.empty else "none"
    text = rf"""% Auto-generated candidate revision text. Verify journal style and numbering before use.
\subsection{{Dimensionless Symmetry Index}}
For a two-edge tool, the right-half profiles separated by approximately $180^\circ$ were compared over
$N=90$ aligned angular samples. Let $P_{{1,i}}$ and $P_{{2,i}}$ denote the right-half ROI pixel counts
for the two phase-aligned regions. The mean absolute difference and pooled mean profile area were
\begin{{equation}}
\overline{{D}}=\frac{{1}}{{N}}\sum_{{i=1}}^N |P_{{1,i}}-P_{{2,i}}|,
\qquad
\overline{{P}}=\frac{{1}}{{2N}}\sum_{{i=1}}^N(P_{{1,i}}+P_{{2,i}}).
\end{{equation}}
The proposed Dimensionless Symmetry Index (DSI) was then defined as
\begin{{equation}}
\mathrm{{DSI}}=100\frac{{\overline{{D}}}}{{\overline{{P}}}}\;(\%).
\end{{equation}}
Unlike $\overline{{D}}$, DSI is normalized by the observed profile area and is therefore less directly
dependent on projected tool size and image scale. Before comparison, each angular pixel-count sequence
was smoothed independently using a centered {results['smoothing_window']}-sample moving-average window
at {100*results['smoothing_strength']:.0f}\% strength. Raw and processed values were retained in the exported data.
DSI was available for {results['n_descriptive_total']} tools. Specimen 069, which exhibited built-up edge
(BUE), was retained in the descriptive per-tool presentation as a separate condition but excluded from
the binary inferential, ROC, and classification analyses. Those analyses therefore included
{results['n_total']} tools ({results['n_functional']} Functional: New or Used; {results['n_fractured']}
Fractured). Specimens {excluded_ids} were excluded because no final comparison output was available.

\begin{{figure}}[t]
\centering
\includegraphics[width=\linewidth]{{phase1_dsi/dsi_by_tool.pdf}}
\caption{{DSI values by specimen and condition. The BUE specimen is shown descriptively but was excluded
from the binary Functional-versus-Fractured statistical analyses.}}
\label{{fig:dsi_by_tool}}
\end{{figure}}

\subsection{{Statistical Validation of DSI}}
Normality of the Functional baseline was assessed using the Shapiro--Wilk test
($W={results['shapiro_functional_W']:.3f}$, $p={results['shapiro_functional_p']:.4f}$).
Because the sample was small and normality-based assumptions were not required, a two-sided
Mann--Whitney U test was used for the group comparison. DSI differed between Functional and Fractured
tools ($U={results['mann_whitney_U']:.1f}$, $p={results['mann_whitney_p']:.4g}$), with rank-biserial
correlation $r_{{rb}}={results['rank_biserial']:.3f}$. This constitutes evidence of group separation in
the present dataset, rather than proof of universal separation.

\begin{{figure}}[t]
\centering
\includegraphics[width=.72\linewidth]{{phase2_significance/dsi_group_comparison.pdf}}
\caption{{Distribution of DSI in the Functional and Fractured groups. Points denote individual tools.}}
\label{{fig:dsi_groups}}
\end{{figure}}

\subsection{{ROC Analysis and Threshold Determination}}
To avoid selecting a threshold by visual inspection, all candidate DSI decision boundaries were evaluated
using receiver operating characteristic (ROC) analysis. The threshold maximizing Youden's index,
$J=\mathrm{{sensitivity}}+\mathrm{{specificity}}-1$, was {results['threshold']:.3f}\% DSI.
The area under the ROC curve was {results['auc']:.3f} (95\% stratified-bootstrap CI
{results['auc_ci_low']:.3f}--{results['auc_ci_high']:.3f}).

\begin{{figure}}[t]
\centering
\includegraphics[width=.72\linewidth]{{phase3_roc_threshold/roc_curve_youden.pdf}}
\caption{{ROC curve and the operating point selected by maximum Youden's index.}}
\label{{fig:dsi_roc}}
\end{{figure}}

\subsection{{Diagnostic Performance}}
Applying the rule $\mathrm{{DSI}}>{results['threshold']:.3f}\%$ to the same {results['n_total']}-tool
dataset yielded sensitivity {results['sensitivity']:.3f}, specificity {results['specificity']:.3f},
accuracy {results['accuracy']:.3f}, precision {results['precision']:.3f}, and F1-score
{results['f1']:.3f}. The confusion matrix contained TP={results['TP']}, FN={results['FN']},
TN={results['TN']}, and FP={results['FP']}.

\begin{{table}}[t]
\centering
\caption{{Apparent diagnostic performance at the Youden-optimal threshold.}}
\begin{{tabular}}{{lcc}}
\hline
Metric & Estimate & 95\% CI \\
\hline
Sensitivity & {results['sensitivity']:.3f} & {results['sensitivity_ci_low']:.3f}--{results['sensitivity_ci_high']:.3f} \\
Specificity & {results['specificity']:.3f} & {results['specificity_ci_low']:.3f}--{results['specificity_ci_high']:.3f} \\
Accuracy & {results['accuracy']:.3f} & {results['accuracy_ci_low']:.3f}--{results['accuracy_ci_high']:.3f} \\
Precision & {results['precision']:.3f} & {results['precision_ci_low']:.3f}--{results['precision_ci_high']:.3f} \\
F1-score & {results['f1']:.3f} & -- \\
\hline
\end{{tabular}}
\label{{tab:dsi_performance}}
\end{{table}}

Because threshold selection and performance estimation used the same small dataset, these values are
apparent (resubstitution) performance and may be optimistic. The threshold must be locked and evaluated
on an independent, prospectively collected tool cohort before industrial deployment. The previously
proposed Weibull baseline model remains appropriate for future large-scale functional-tool monitoring,
where its parameters and target false-alarm probability can be estimated reliably.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--masks-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    for name in (
        "phase1_dsi", "phase2_significance", "phase3_roc_threshold",
        "phase4_classification_metrics",
    ):
        (output / name).mkdir(parents=True, exist_ok=True)

    all_tools, excluded = discover_results(args.metadata, args.masks_root)
    bue_tools = all_tools.loc[all_tools.diagnostic_group == "BUE"].copy()
    df = all_tools.loc[all_tools.diagnostic_group != "BUE"].copy()
    functional = df.loc[df.diagnostic_group == "Functional", "DSI_percent"].to_numpy()
    fractured = df.loc[df.diagnostic_group == "Fractured", "DSI_percent"].to_numpy()
    if len(functional) < 3 or len(fractured) < 3:
        raise ValueError("Insufficient tools in one or both diagnostic groups")

    shapiro_functional = stats.shapiro(functional)
    shapiro_fractured = stats.shapiro(fractured)
    mw = stats.mannwhitneyu(fractured, functional, alternative="two-sided", method="exact")
    rank_biserial = 2.0 * float(mw.statistic) / (len(fractured) * len(functional)) - 1.0
    y_true = (df.diagnostic_group == "Fractured").astype(int).to_numpy()
    scores = df.DSI_percent.to_numpy()
    roc = roc_points(y_true, scores)
    best = roc.loc[roc.youden_J == roc.youden_J.max()].copy()
    # If equivalent operating points exist, choose the largest threshold (most conservative alarm rule).
    best = best.sort_values("threshold_DSI_percent", ascending=False).iloc[0]
    threshold = float(best.threshold_DSI_percent)
    auc = auc_from_scores(y_true, scores)
    auc_boot = bootstrap_auc(y_true, scores)
    auc_ci = tuple(np.quantile(auc_boot, [.025, .975]))

    tp, fn, tn, fp = map(int, [best.TP, best.FN, best.TN, best.FP])
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    accuracy = (tp + tn) / len(df)
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0
    sens_ci = wilson_interval(tp, tp + fn)
    spec_ci = wilson_interval(tn, tn + fp)
    acc_ci = wilson_interval(tp + tn, len(df))
    prec_ci = wilson_interval(tp, tp + fp)

    results = {
        "metric_name": "Dimensionless Symmetry Index (DSI)",
        "definition": "100 * mean(abs(P1-P2)) / mean(concatenate(P1,P2))",
        "n_descriptive_total": len(all_tools),
        "n_total": len(df), "n_functional": len(functional), "n_fractured": len(fractured),
        "n_bue_excluded_from_binary_analysis": len(bue_tools),
        "bue_tools_excluded_from_binary_analysis": bue_tools.tool_id.tolist(),
        "excluded_tools": excluded.tool_id.tolist() if not excluded.empty else [],
        "smoothing_window": int(df.smoothing_window.dropna().mode().iloc[0]),
        "smoothing_strength": float(df.smoothing_strength.dropna().mode().iloc[0]),
        "shapiro_functional_W": float(shapiro_functional.statistic),
        "shapiro_functional_p": float(shapiro_functional.pvalue),
        "shapiro_fractured_W": float(shapiro_fractured.statistic),
        "shapiro_fractured_p": float(shapiro_fractured.pvalue),
        "mann_whitney_U": float(mw.statistic), "mann_whitney_p": float(mw.pvalue),
        "rank_biserial": rank_biserial,
        "threshold": threshold, "youden_J": float(best.youden_J),
        "auc": auc, "auc_ci_low": float(auc_ci[0]), "auc_ci_high": float(auc_ci[1]),
        "TP": tp, "FN": fn, "TN": tn, "FP": fp,
        "sensitivity": sensitivity, "specificity": specificity, "accuracy": accuracy,
        "precision": precision, "f1": f1,
        "sensitivity_ci_low": sens_ci[0], "sensitivity_ci_high": sens_ci[1],
        "specificity_ci_low": spec_ci[0], "specificity_ci_high": spec_ci[1],
        "accuracy_ci_low": acc_ci[0], "accuracy_ci_high": acc_ci[1],
        "precision_ci_low": prec_ci[0], "precision_ci_high": prec_ci[1],
        "caution": "Threshold and performance are estimated on the same dataset and require external validation.",
    }

    predicted = np.where(scores > threshold, "Fractured", "Functional")
    df["predicted_group"] = predicted
    df["correct_classification"] = df.predicted_group == df.diagnostic_group
    df["youden_threshold_DSI_percent"] = threshold
    all_tools_output = all_tools.copy()
    all_tools_output["included_in_binary_analysis"] = all_tools_output.diagnostic_group != "BUE"
    all_tools_output = all_tools_output.merge(
        df[["tool_id", "predicted_group", "correct_classification", "youden_threshold_DSI_percent"]],
        on="tool_id", how="left",
    )
    all_tools_output.loc[all_tools_output.diagnostic_group == "BUE", "predicted_group"] = "Not evaluated"
    all_tools_output.sort_values("tool_id").to_csv(
        output / "phase1_dsi" / "per_tool_dsi.csv", index=False
    )
    excluded.to_csv(output / "excluded_tools.csv", index=False)
    pd.DataFrame([
        {
            "tool_id": row.tool_id,
            "condition": row.condition,
            "reason": "BUE retained descriptively but excluded from binary Functional-versus-Fractured analysis",
        }
        for row in bue_tools.itertuples()
    ]).to_csv(output / "binary_analysis_exclusions.csv", index=False)
    pd.DataFrame([
        {
            "group": "Functional",
            "test": "Shapiro-Wilk",
            "statistic": float(shapiro_functional.statistic),
            "p_value": float(shapiro_functional.pvalue),
            "n": len(functional),
        },
        {
            "group": "Fractured",
            "test": "Shapiro-Wilk",
            "statistic": float(shapiro_fractured.statistic),
            "p_value": float(shapiro_fractured.pvalue),
            "n": len(fractured),
        },
        {
            "group": "Fractured vs Functional",
            "test": "Mann-Whitney U (two-sided exact)",
            "statistic": float(mw.statistic),
            "p_value": float(mw.pvalue),
            "n": len(df),
            "effect_size_rank_biserial": rank_biserial,
        },
    ]).to_csv(output / "phase2_significance" / "statistical_tests.csv", index=False)
    roc.to_csv(output / "phase3_roc_threshold" / "roc_thresholds.csv", index=False)
    pd.DataFrame([{
        "threshold_DSI_percent": threshold,
        "youden_J": float(best.youden_J),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "false_positive_rate": 1.0 - specificity,
        "auc": auc,
        "auc_bootstrap_ci_low": float(auc_ci[0]),
        "auc_bootstrap_ci_high": float(auc_ci[1]),
    }]).to_csv(output / "phase3_roc_threshold" / "selected_threshold.csv", index=False)

    metric_rows = []
    for name, value, ci in (
        ("Sensitivity", sensitivity, sens_ci), ("Specificity", specificity, spec_ci),
        ("Accuracy", accuracy, acc_ci), ("Precision", precision, prec_ci),
        ("F1-score", f1, (f1, f1)),
    ):
        metric_rows.append({"metric": name, "value": value, "ci_low": ci[0], "ci_high": ci[1]})
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output / "phase4_classification_metrics" / "classification_metrics.csv", index=False)
    pd.DataFrame([
        {"actual": "Functional", "predicted_functional": tn, "predicted_fractured": fp},
        {"actual": "Fractured", "predicted_functional": fn, "predicted_fractured": tp},
    ]).to_csv(output / "phase4_classification_metrics" / "confusion_matrix.csv", index=False)
    df[[
        "tool_id", "condition", "diagnostic_group", "DSI_percent",
        "youden_threshold_DSI_percent", "predicted_group", "correct_classification",
    ]].sort_values("tool_id").to_csv(
        output / "phase4_classification_metrics" / "per_tool_predictions.csv", index=False
    )

    generate_figures(all_tools, df, roc, threshold, auc, auc_ci, metrics, output, float(mw.pvalue))
    (output / "analysis_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_latex(output / "revision_sections.tex", results, excluded)
    shutil.copy2(Path(__file__), output / "revision_dsi_analysis.py")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
