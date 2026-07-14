import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error

###############################################
# Load predictions
###############################################

pred = pd.read_csv(
    "/users/coh33/SpaMIE/results/human_skin_feature/SpaMIE pred result/1human_skin_feature_pred.csv"
)

truth = pd.read_csv(
    "/users/coh33/SpaMIE/results/human_skin_feature/SpaMIE pred result/1human_skin_feature_truth.csv"
)

print("Prediction shape:", pred.shape)
print("Truth shape:", truth.shape)

###############################################
# Evaluation
###############################################

pearsons = []
spearmans = []
rmses = []

constant_truth = []
constant_pred = []

for j in range(pred.shape[1]):

    y_true = truth.iloc[:, j]
    y_pred = pred.iloc[:, j]

    # Check for constant vectors
    if y_true.std() == 0:
        constant_truth.append(j)

    if y_pred.std() == 0:
        constant_pred.append(j)

    r_p, _ = pearsonr(y_true, y_pred)
    r_s, _ = spearmanr(y_true, y_pred)

    pearsons.append(r_p)
    spearmans.append(r_s)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    rmses.append(rmse)

print("Constant truth proteins:", constant_truth)
print("Constant predicted proteins:", constant_pred)
print("NaN Pearson:", np.isnan(pearsons).sum())
print("NaN Spearman:", np.isnan(spearmans).sum())
###############################################
# Summary statistics
###############################################

print("\n========== SUMMARY ==========")

print(f"Mean Pearson : {np.nanmean(pearsons):.4f}")
print(f"Median Pearson : {np.nanmedian(pearsons):.4f}")

print(f"Mean Spearman : {np.nanmean(spearmans):.4f}")
print(f"Median Spearman : {np.nanmedian(spearmans):.4f}")

print(f"Mean RMSE : {np.mean(rmses):.4f}")
print(f"Median RMSE : {np.median(rmses):.4f}")

###############################################
# Save results
###############################################

results = pd.DataFrame({
    "Protein": truth.columns,
    "Pearson": pearsons,
    "Spearman": spearmans,
    "RMSE": rmses
})

print("\nTop 10 largest RMSE:")
print(results.sort_values("RMSE", ascending=False).head(10))

outdir = "/users/coh33/SpaMIE/results/human_skin_feature/evaluation"
os.makedirs(outdir, exist_ok=True)

results.to_csv(
    os.path.join(outdir, "protein_metrics.csv"),
    index=False
)

###############################################
# Histogram: Pearson
###############################################

plt.figure(figsize=(6,4))
plt.hist(pearsons, bins=20)
plt.xlabel("Pearson correlation")
plt.ylabel("Number of proteins")
plt.title("SpaMIE Human Skin")
plt.tight_layout()
plt.savefig(os.path.join(outdir, "pearson_histogram.png"))
plt.close()

###############################################
# Histogram: Spearman
###############################################

plt.figure(figsize=(6,4))
plt.hist(spearmans, bins=20)
plt.xlabel("Spearman correlation")
plt.ylabel("Number of proteins")
plt.title("SpaMIE Human Skin")
plt.tight_layout()
plt.savefig(os.path.join(outdir, "spearman_histogram.png"))
plt.close()

###############################################
# Histogram: RMSE
###############################################

plt.figure(figsize=(6,4))
plt.hist(rmses, bins=20)
plt.xlabel("RMSE")
plt.ylabel("Number of proteins")
plt.title("SpaMIE Human Skin")
plt.tight_layout()
plt.savefig(os.path.join(outdir, "rmse_histogram.png"))
plt.close()

###############################################
# Ranked Pearson plot
###############################################

sorted_pearson = np.sort(pearsons)[::-1]

plt.figure(figsize=(8,4))
plt.plot(sorted_pearson)
plt.xlabel("Protein rank")
plt.ylabel("Pearson correlation")
plt.title("Pearson Correlation by Protein")
plt.tight_layout()
plt.savefig(os.path.join(outdir, "pearson_ranked.png"))
plt.close()

print("\nEvaluation complete!")





