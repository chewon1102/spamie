"""
Compute per-protein Pearson, Spearman, and RMSE from the raw
prediction/truth CSVs produced by the cell-feature training runs,
in the same format as your existing protein_metrics.csv files
(so it plugs directly into your existing stratification/summary code).

Loops over all 5 graph modes (attention, union, intersection,
spatial, feature) — each trained with the cell-feature graph enabled.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

BASE_RESULTS_DIR = "/users/coh33/SpaMIE/results"
GRAPH_MODES = ["attention", "union", "intersection", "spatial", "feature"]


def compute_metrics_for_mode(mode):
    result_dir = os.path.join(BASE_RESULTS_DIR, f"human_skin_{mode}_cellfeat", "SpaMIE pred result")
    pred_file = os.path.join(result_dir, f"1human_skin_{mode}_cellfeat_pred.csv")
    true_file = os.path.join(result_dir, f"1human_skin_{mode}_cellfeat_truth.csv")

    output_dir = os.path.join(BASE_RESULTS_DIR, f"human_skin_{mode}_cellfeat", "evaluation")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "protein_metrics.csv")

    pred = pd.read_csv(pred_file)
    truth = pd.read_csv(true_file)

    if pred.shape != truth.shape:
        raise ValueError(
            f"[{mode}] Shape mismatch: pred is {pred.shape}, truth is {truth.shape}."
        )

    n_test_spots, n_proteins = pred.shape
    print(f"[{mode}] Loaded {n_test_spots} test spots x {n_proteins} proteins")

    records = []
    for col in pred.columns:
        y_pred = pred[col].values
        y_true = truth[col].values

        if np.std(y_pred) == 0 or np.std(y_true) == 0:
            r_pearson, r_spearman = np.nan, np.nan
        else:
            r_pearson, _ = pearsonr(y_pred, y_true)
            r_spearman, _ = spearmanr(y_pred, y_true)

        rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))

        records.append({
            "Protein": col,
            "Pearson": r_pearson,
            "Spearman": r_spearman,
            "RMSE": rmse,
        })

    metrics_df = pd.DataFrame(records)
    metrics_df["Protein"] = range(len(metrics_df))  # match existing plain-integer format
    metrics_df.to_csv(output_file, index=False)

    n_nan = metrics_df["Pearson"].isna().sum()
    if n_nan > 0:
        print(f"[{mode}] Warning: {n_nan} protein(s) had zero variance and got NaN correlation.")

    print(f"[{mode}] Mean Pearson: {np.nanmean(metrics_df['Pearson']):.4f} | "
          f"Mean RMSE: {metrics_df['RMSE'].mean():.4f}")
    print(f"[{mode}] Saved to {output_file}\n")

    return metrics_df


############################################################
# Run for all 5 modes and build a combined summary
############################################################
all_summaries = []

for mode in GRAPH_MODES:
    metrics_df = compute_metrics_for_mode(mode)
    all_summaries.append({
        "Graph Mode": f"{mode}_cellfeat",
        "Mean Pearson": np.nanmean(metrics_df["Pearson"]),
        "Median Pearson": np.nanmedian(metrics_df["Pearson"]),
        "Mean Spearman": np.nanmean(metrics_df["Spearman"]),
        "Median Spearman": np.nanmedian(metrics_df["Spearman"]),
        "Mean RMSE": metrics_df["RMSE"].mean(),
        "Median RMSE": metrics_df["RMSE"].median(),
    })

summary_df = pd.DataFrame(all_summaries).round(4)
summary_path = os.path.join(BASE_RESULTS_DIR, "graph_mode_cellfeat_summary.csv")
summary_df.to_csv(summary_path, index=False)

print("=== Cell-feature runs: summary across all 5 modes ===")
print(summary_df)
print(f"\nSaved combined cell-feat summary to {summary_path}")
