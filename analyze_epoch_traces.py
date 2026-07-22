"""
Analyze epoch-by-epoch training loss traces to check for patterns
near the end of training (plateau, still improving, or overfitting).

Loads each mode's {seed}_epoch_loss_trace.csv (saved automatically by
spamie_main_working.py / spamie_main_cellfeat.py during training),
plots train vs. validation RMSE over epochs, and prints a diagnostic
summary for each.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe for cluster environments
import matplotlib.pyplot as plt

BASE = "/users/coh33/SpaMIE/results"
SEED = 1  # matches seed=1 used in all training scripts so far

# Every mode we have an epoch trace for. Baseline (non-cellfeat) modes are
# NOT included here, because those were trained before epoch-loss tracing
# was added to spamie_main_working.py — their folders have no
# training_trace/ subfolder at all. Once you retrain the baselines with
# tracing enabled, add them here the same way.
#
# Includes both the original 50-epoch cellfeat runs AND the new 200-epoch
# runs (train_cell_feat_test_e200.py), so you can directly compare
# convergence behavior at the two epoch budgets.
MODES_WITH_TRACES = {
    # --- 50 epochs (original) ---
    "attention_cellfeat_e50": "human_skin_attention_cellfeat",
    "union_cellfeat_e50": "human_skin_union_cellfeat",
    "intersection_cellfeat_e50": "human_skin_intersection_cellfeat",
    "spatial_cellfeat_e50": "human_skin_spatial_cellfeat",
    "feature_cellfeat_e50": "human_skin_feature_cellfeat",
    # --- 200 epochs (new) ---
    "attention_cellfeat_e200": "human_skin_attention_e200_cellfeat",
    "union_cellfeat_e200": "human_skin_union_e200_cellfeat",
    "intersection_cellfeat_e200": "human_skin_intersection_e200_cellfeat",
    "spatial_cellfeat_e200": "human_skin_spatial_e200_cellfeat",
    "feature_cellfeat_e200": "human_skin_feature_e200_cellfeat",
}

VIZ_DIR = os.path.join(BASE, "epoch_trace_plots")
os.makedirs(VIZ_DIR, exist_ok=True)

# How many of the final epochs to look at when checking for a pattern
LAST_N_EPOCHS = 10


def diagnose_pattern(val_rmse, last_n=LAST_N_EPOCHS):
    """
    Simple heuristic check on the last `last_n` epochs of validation RMSE:
      - "still improving": val RMSE noticeably decreasing at the end
      - "plateaued": val RMSE roughly flat at the end
      - "overfitting": val RMSE increasing while train RMSE (checked
         separately) keeps decreasing
    This is a heuristic, not a formal statistical test — meant to give
    you a quick read, not a definitive claim.
    """
    if len(val_rmse) < last_n + 1:
        last_n = max(2, len(val_rmse) - 1)

    tail = val_rmse.iloc[-last_n:].values
    # Simple linear trend over the tail: positive slope = getting worse
    x = np.arange(len(tail))
    slope = np.polyfit(x, tail, 1)[0]

    # Normalize slope by the RMSE scale so the threshold is meaningful
    # regardless of the absolute RMSE magnitude
    relative_slope = slope / (np.mean(tail) + 1e-8)

    if relative_slope > 0.01:
        return "overfitting (val RMSE rising)"
    elif relative_slope < -0.01:
        return "still improving (val RMSE falling)"
    else:
        return "plateaued (val RMSE roughly flat)"


############################################################
# Loop over every mode with an available epoch trace
############################################################
summary_rows = []

for mode, folder in MODES_WITH_TRACES.items():
    trace_path = os.path.join(BASE, folder, "training_trace", f"{SEED}_epoch_loss_trace.csv")

    if not os.path.exists(trace_path):
        print(f"[{mode}] No epoch trace found at {trace_path} — skipping.")
        continue

    trace = pd.read_csv(trace_path)
    n_epochs = len(trace)

    print(f"\n=== [{mode}] Epoch trace ({n_epochs} epochs) ===")
    print(f"First epoch:  train_rmse={trace['train_rmse'].iloc[0]:.4f}, "
          f"val_rmse={trace['val_rmse'].iloc[0]:.4f}")
    print(f"Last epoch:   train_rmse={trace['train_rmse'].iloc[-1]:.4f}, "
          f"val_rmse={trace['val_rmse'].iloc[-1]:.4f}")

    val_pattern = diagnose_pattern(trace["val_rmse"])
    train_pattern = diagnose_pattern(trace["train_rmse"])
    print(f"Pattern over last {min(LAST_N_EPOCHS, n_epochs-1)} epochs: "
          f"train={train_pattern}, val={val_pattern}")

    # Flag the classic overfitting signature explicitly
    if "improving" in train_pattern and "overfitting" in val_pattern:
        print(f"  -> Overfitting signature detected: train still improving "
              f"while val RMSE rises.")

    summary_rows.append({
        "Graph Mode": mode,
        "N Epochs": n_epochs,
        "First Train RMSE": trace["train_rmse"].iloc[0],
        "Last Train RMSE": trace["train_rmse"].iloc[-1],
        "First Val RMSE": trace["val_rmse"].iloc[0],
        "Last Val RMSE": trace["val_rmse"].iloc[-1],
        "Train Pattern (last epochs)": train_pattern,
        "Val Pattern (last epochs)": val_pattern,
    })

    # --- Plot train vs val RMSE over epochs ---
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(trace["epoch"], trace["train_rmse"], label="Train RMSE", color="#B33A3A")
    ax.plot(trace["epoch"], trace["val_rmse"], label="Val RMSE", color="#4E3629")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE")
    ax.set_title(f"Training curve: {mode}")
    ax.legend()
    plt.tight_layout()
    plot_path = os.path.join(VIZ_DIR, f"epoch_trace_{mode}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved plot to {plot_path}")


############################################################
# Combined summary across all modes
############################################################
if summary_rows:
    summary_df = pd.DataFrame(summary_rows).round(4)
    summary_path = os.path.join(BASE, "epoch_trace_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n=== Epoch trace summary across all modes ===")
    print(summary_df.to_string(index=False))
    print(f"\nSaved to {summary_path}")
else:
    print("\nNo epoch traces found for any mode — nothing to summarize.")
