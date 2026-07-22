"""
Build a side-by-side baseline-vs-uncertainty comparison table from
graph_mode_summary.csv, AND generate paired per-protein boxplot
visualizations (overall, rank-decile, detection-quantile) comparing
baseline vs. uncertainty-weighted-loss for each mode.

Mirrors build_baseline_vs_cellfeat_table.py exactly, just pointed at
the *_uncertainty modes instead of *_cellfeat.
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe for cluster environments
import matplotlib.pyplot as plt

BASE = "/users/coh33/SpaMIE/results"
summary = pd.read_csv(os.path.join(BASE, "graph_mode_summary.csv"))

BASELINE_MODES = ["attention", "union", "intersection", "spatial", "feature"]

VIZ_DIR = os.path.join(BASE, "protein_boxplots_paired_uncertainty")
os.makedirs(VIZ_DIR, exist_ok=True)


############################################################
# Part 1: Summary table
############################################################
rows = []
for mode in BASELINE_MODES:
    baseline_row = summary[summary["Graph Mode"] == mode]
    uncertainty_row = summary[summary["Graph Mode"] == f"{mode}_uncertainty"]

    if baseline_row.empty or uncertainty_row.empty:
        print(f"[{mode}] Missing baseline or uncertainty row — skipping.")
        continue

    rows.append({
        "Graph Mode": mode,
        "Mean Pearson (baseline)": baseline_row["Mean Pearson"].values[0],
        "Mean Pearson (+uncertainty)": uncertainty_row["Mean Pearson"].values[0],
        "Mean RMSE (baseline)": baseline_row["Mean RMSE"].values[0],
        "Mean RMSE (+uncertainty)": uncertainty_row["Mean RMSE"].values[0],
    })

paired_df = pd.DataFrame(rows).round(4)
output_path = os.path.join(BASE, "graph_mode_baseline_vs_uncertainty.csv")
paired_df.to_csv(output_path, index=False)

print(paired_df.to_string(index=False))
print(f"\nSaved to {output_path}")


############################################################
# Part 2: Paired per-protein boxplots — baseline vs. uncertainty
############################################################
print("\n=== Generating paired boxplot visualizations ===")

for mode in BASELINE_MODES:
    baseline_file = os.path.join(BASE, f"protein_stratification_{mode}.csv")
    uncertainty_file = os.path.join(BASE, f"protein_stratification_{mode}_uncertainty.csv")

    if not os.path.exists(baseline_file) or not os.path.exists(uncertainty_file):
        print(f"[{mode}] Missing stratification file(s) — skipping visualization.")
        continue

    baseline_strat = pd.read_csv(baseline_file)
    uncertainty_strat = pd.read_csv(uncertainty_file)

    # --- 1) Overall boxplot: baseline vs. uncertainty, side by side ---
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.boxplot(
        [baseline_strat["Pearson"].dropna(), uncertainty_strat["Pearson"].dropna()],
        tick_labels=["Baseline", "+Uncertainty"],
    )
    ax.set_title(f"Overall Pearson distribution ({mode})")
    ax.set_ylabel("Pearson correlation")
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, f"pearson_overall_paired_{mode}.png"), dpi=150)
    plt.close()

    # --- 2) Rank-decile boxplot: baseline vs. uncertainty as separate series ---
    baseline_strat = baseline_strat.copy()
    uncertainty_strat = uncertainty_strat.copy()
    baseline_strat["Rank Decile"] = pd.qcut(
        baseline_strat["Pearson"].rank(method="first"), 10, labels=False
    )
    uncertainty_strat["Rank Decile"] = pd.qcut(
        uncertainty_strat["Pearson"].rank(method="first"), 10, labels=False
    )
    baseline_strat["Run"] = "Baseline"
    uncertainty_strat["Run"] = "+Uncertainty"
    combined_decile = pd.concat([
        baseline_strat[["Pearson", "Rank Decile", "Run"]],
        uncertainty_strat[["Pearson", "Rank Decile", "Run"]],
    ])

    fig, ax = plt.subplots(figsize=(11, 5))
    positions = []
    data = []
    for decile in range(10):
        for run, offset in [("Baseline", -0.15), ("+Uncertainty", 0.15)]:
            vals = combined_decile[
                (combined_decile["Rank Decile"] == decile) & (combined_decile["Run"] == run)
            ]["Pearson"]
            data.append(vals)
            positions.append(decile + offset)
    bp = ax.boxplot(data, positions=positions, widths=0.25, patch_artist=True)
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor("#B33A3A" if i % 2 == 0 else "#4E3629")
    ax.set_xticks(range(10))
    ax.set_xticklabels([f"D{i+1}" for i in range(10)])
    ax.set_title(f"Pearson by rank decile: baseline (red) vs. +uncertainty (brown) — {mode}")
    ax.set_ylabel("Pearson correlation")
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, f"pearson_by_rank_decile_paired_{mode}.png"), dpi=150)
    plt.close()

    # --- 3) Detection-rate quantile boxplot: baseline vs. uncertainty ---
    baseline_strat["Detection Quantile"] = pd.qcut(
        baseline_strat["Detection Rate"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"]
    )
    uncertainty_strat["Detection Quantile"] = pd.qcut(
        uncertainty_strat["Detection Rate"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"]
    )
    baseline_strat["Run"] = "Baseline"
    uncertainty_strat["Run"] = "+Uncertainty"
    combined_quant = pd.concat([
        baseline_strat[["Pearson", "Detection Quantile", "Run"]],
        uncertainty_strat[["Pearson", "Detection Quantile", "Run"]],
    ])

    fig, ax = plt.subplots(figsize=(9, 5))
    positions = []
    data = []
    quantiles = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    for qi, q in enumerate(quantiles):
        for run, offset in [("Baseline", -0.15), ("+Uncertainty", 0.15)]:
            vals = combined_quant[
                (combined_quant["Detection Quantile"] == q) & (combined_quant["Run"] == run)
            ]["Pearson"]
            data.append(vals)
            positions.append(qi + offset)
    bp = ax.boxplot(data, positions=positions, widths=0.25, patch_artist=True)
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor("#B33A3A" if i % 2 == 0 else "#4E3629")
    ax.set_xticks(range(5))
    ax.set_xticklabels(quantiles)
    ax.set_title(f"Pearson by detection-rate quantile: baseline (red) vs. +uncertainty (brown) — {mode}")
    ax.set_ylabel("Pearson correlation")
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, f"pearson_by_detection_quantile_paired_{mode}.png"), dpi=150)
    plt.close()

    print(f"[{mode}] Saved 3 paired boxplots to {VIZ_DIR}")

print(f"\nAll paired visualizations saved to {VIZ_DIR}")
