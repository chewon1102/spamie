"""
Build a side-by-side baseline-vs-cellfeat comparison table from
graph_mode_summary.csv (which has all 10 modes in long format:
5 baseline + 5 *_cellfeat), AND generate paired per-protein boxplot
visualizations (overall, rank-decile, detection-quantile) comparing
baseline vs. cellfeat for each mode.
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe for cluster environments
import matplotlib.pyplot as plt

BASE = "/users/coh33/SpaMIE/results"
summary = pd.read_csv(os.path.join(BASE, "graph_mode_summary.csv"))

BASELINE_MODES = ["attention", "union", "intersection", "spatial", "feature"]

VIZ_DIR = os.path.join(BASE, "protein_boxplots_paired")
os.makedirs(VIZ_DIR, exist_ok=True)


############################################################
# Part 1: Summary table (unchanged from before)
############################################################
rows = []
for mode in BASELINE_MODES:
    baseline_row = summary[summary["Graph Mode"] == mode]
    cellfeat_row = summary[summary["Graph Mode"] == f"{mode}_cellfeat"]

    if baseline_row.empty or cellfeat_row.empty:
        print(f"[{mode}] Missing baseline or cellfeat row — skipping.")
        continue

    rows.append({
        "Graph Mode": mode,
        "Mean Pearson (baseline)": baseline_row["Mean Pearson"].values[0],
        "Mean Pearson (+cellfeat)": cellfeat_row["Mean Pearson"].values[0],
        "Mean RMSE (baseline)": baseline_row["Mean RMSE"].values[0],
        "Mean RMSE (+cellfeat)": cellfeat_row["Mean RMSE"].values[0],
    })

paired_df = pd.DataFrame(rows).round(4)
output_path = os.path.join(BASE, "graph_mode_baseline_vs_cellfeat.csv")
paired_df.to_csv(output_path, index=False)

print(paired_df.to_string(index=False))
print(f"\nSaved to {output_path}")


############################################################
# Part 2: Paired per-protein boxplots — baseline vs. cellfeat
# Uses the protein_stratification_{mode}.csv files already
# produced by compare_graph_modes.py's Part 6.
############################################################
print("\n=== Generating paired boxplot visualizations ===")

for mode in BASELINE_MODES:
    baseline_file = os.path.join(BASE, f"protein_stratification_{mode}.csv")
    cellfeat_file = os.path.join(BASE, f"protein_stratification_{mode}_cellfeat.csv")

    if not os.path.exists(baseline_file) or not os.path.exists(cellfeat_file):
        print(f"[{mode}] Missing stratification file(s) — skipping visualization.")
        continue

    baseline_strat = pd.read_csv(baseline_file)
    cellfeat_strat = pd.read_csv(cellfeat_file)

    # --- 1) Overall boxplot: baseline vs. cellfeat, side by side ---
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.boxplot(
        [baseline_strat["Pearson"].dropna(), cellfeat_strat["Pearson"].dropna()],
        tick_labels=["Baseline", "+Cellfeat"],
    )
    ax.set_title(f"Overall Pearson distribution ({mode})")
    ax.set_ylabel("Pearson correlation")
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, f"pearson_overall_paired_{mode}.png"), dpi=150)
    plt.close()

    # --- 2) Rank-decile boxplot: baseline vs. cellfeat as separate series ---
    baseline_strat = baseline_strat.copy()
    cellfeat_strat = cellfeat_strat.copy()
    baseline_strat["Rank Decile"] = pd.qcut(
        baseline_strat["Pearson"].rank(method="first"), 10, labels=False
    )
    cellfeat_strat["Rank Decile"] = pd.qcut(
        cellfeat_strat["Pearson"].rank(method="first"), 10, labels=False
    )
    baseline_strat["Run"] = "Baseline"
    cellfeat_strat["Run"] = "+Cellfeat"
    combined_decile = pd.concat([
        baseline_strat[["Pearson", "Rank Decile", "Run"]],
        cellfeat_strat[["Pearson", "Rank Decile", "Run"]],
    ])

    fig, ax = plt.subplots(figsize=(11, 5))
    positions = []
    data = []
    labels = []
    for decile in range(10):
        for run, offset in [("Baseline", -0.15), ("+Cellfeat", 0.15)]:
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
    ax.set_title(f"Pearson by rank decile: baseline (red) vs. +cellfeat (brown) — {mode}")
    ax.set_ylabel("Pearson correlation")
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, f"pearson_by_rank_decile_paired_{mode}.png"), dpi=150)
    plt.close()

    # --- 3) Detection-rate quantile boxplot: baseline vs. cellfeat ---
    baseline_strat["Detection Quantile"] = pd.qcut(
        baseline_strat["Detection Rate"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"]
    )
    cellfeat_strat["Detection Quantile"] = pd.qcut(
        cellfeat_strat["Detection Rate"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"]
    )
    baseline_strat["Run"] = "Baseline"
    cellfeat_strat["Run"] = "+Cellfeat"
    combined_quant = pd.concat([
        baseline_strat[["Pearson", "Detection Quantile", "Run"]],
        cellfeat_strat[["Pearson", "Detection Quantile", "Run"]],
    ])

    fig, ax = plt.subplots(figsize=(9, 5))
    positions = []
    data = []
    quantiles = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    for qi, q in enumerate(quantiles):
        for run, offset in [("Baseline", -0.15), ("+Cellfeat", 0.15)]:
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
    ax.set_title(f"Pearson by detection-rate quantile: baseline (red) vs. +cellfeat (brown) — {mode}")
    ax.set_ylabel("Pearson correlation")
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, f"pearson_by_detection_quantile_paired_{mode}.png"), dpi=150)
    plt.close()

    print(f"[{mode}] Saved 3 paired boxplots to {VIZ_DIR}")

print(f"\nAll paired visualizations saved to {VIZ_DIR}")
