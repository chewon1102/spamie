# Evaluation: Correlation
import pandas as pd
from scipy.stats import pearsonr

pred = pd.read_csv("/users/coh33/SpaMIE/results/my_data/SpaMIE pred result/1my_data_pred.csv", index_col=0)
truth = pd.read_csv("/users/coh33/SpaMIE/results/my_data/SpaMIE pred result/1my_data_truth.csv", index_col=0)

print(pred.shape)
print(truth.shape)

corrs = []

for col in pred.columns:
    r, _ = pearsonr(pred[col], truth[col])
    corrs.append(r)

print("Mean Pearson:", sum(corrs)/len(corrs))

# Evaluation: RMSE 
import numpy as np

diff = pred.values - truth.values
rmse = np.sqrt(np.mean(diff**2))

print("RMSE =", rmse)

with open("/users/coh33/SpaMIE/results/my_data/metrics.txt", "w") as f:
    f.write(f"Mean Pearson: {sum(corrs)/len(corrs):.4f}\n")
    f.write(f"RMSE: {rmse:.4f}\n")

# Predicted vs True Scatter Plot 
import matplotlib.pyplot as plt
import numpy as np

truth_all = truth.values.flatten()
pred_all = pred.values.flatten()

plt.figure(figsize=(6,6))
plt.scatter(truth_all, pred_all, alpha=0.3, s=5)

lims = [
    min(truth_all.min(), pred_all.min()),
    max(truth_all.max(), pred_all.max())
]

plt.plot(lims, lims, '--')
plt.xlabel("True Protein Expression")
plt.ylabel("Predicted Protein Expression")
plt.title("SpaMIE Prediction Performance")
plt.tight_layout()
plt.savefig("/users/coh33/SpaMIE/results/my_data/pred_vs_true_scatter.png")
plt.close()

# Per Protein Pearson Correlation Bar Plot 
from scipy.stats import pearsonr
import pandas as pd
import matplotlib.pyplot as plt

protein_corrs = []

for col in pred.columns:
    r, _ = pearsonr(pred[col], truth[col])
    protein_corrs.append((col, r))

corr_df = pd.DataFrame(
    protein_corrs,
    columns=["Protein", "Pearson"]
).sort_values("Pearson", ascending=False)

plt.figure(figsize=(10,5))
plt.bar(corr_df["Protein"], corr_df["Pearson"])
plt.xticks(rotation=90)
plt.ylabel("Pearson Correlation")
plt.title("Protein-wise Prediction Accuracy")
plt.tight_layout()
plt.savefig("/users/coh33/SpaMIE/results/my_data/protein_corr_barplot.png")
plt.close()

# Benchmark 



