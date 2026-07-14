"""
Merged SpaMIE human-skin analysis script.

This combines:
  1) Graph construction / edge counting across graph modes
     (attention, union, intersection, spatial, feature)
  2) Summary of previously-computed protein imputation metrics
     (Pearson, Spearman, RMSE) for each corresponding experiment folder

The two are joined on graph mode so the final table shows, per mode,
both the graph edge counts and the protein-level imputation performance.
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
import torch as th

from SpaMIE.preprocess import clr_normalize_each_cell, pca
from SpaMIE.create_graph_v2 import Sagegraph

device = th.device("cpu")

BASE = "/users/coh33/SpaMIE/results"
DATA_DIR = "/users/coh33/SpaMIE/data/human_skin"

# Maps each graph mode -> the results folder holding its protein_metrics.csv
EXPERIMENTS = {
    "attention": "human_skin_hvg",
    "union": "human_skin_union",
    "intersection": "human_skin_intersection",
    "spatial": "human_skin_spatial",
    "feature": "human_skin_feature",
}

GRAPH_MODES = ["attention", "union", "intersection", "spatial", "feature"]


############################################################
# Part 1: Load data
############################################################
rna = pd.read_csv(
    os.path.join(DATA_DIR, "GSM6578065_humanskin_RNA.tsv.gz"),
    sep="\t",
    index_col=0,
)
protein = pd.read_csv(
    os.path.join(DATA_DIR, "GSM6578074_humanskin_protein.tsv.gz"),
    sep="\t",
    index_col=0,
)
protein = protein.loc[rna.index]

adata_rna = AnnData(rna)
adata_protein = AnnData(protein)

coords = np.array([
    list(map(float, cell.split("x")))
    for cell in adata_rna.obs_names
])
adata_rna.obsm["spatial"] = coords
adata_protein.obsm["spatial"] = coords


############################################################
# Part 2: Preprocessing (same as experiments)
############################################################
sc.pp.filter_genes(adata_rna, min_cells=0)
sc.pp.filter_cells(adata_rna, min_genes=0)
sc.pp.highly_variable_genes(
    adata_rna,
    flavor="seurat_v3",
    n_top_genes=3000,
)
sc.pp.normalize_total(adata_rna, target_sum=1e4)
sc.pp.log1p(adata_rna)
adata_rna_high = adata_rna[:, adata_rna.var["highly_variable"]].copy()

adata_protein = clr_normalize_each_cell(adata_protein)

adata_rna.obsm["feat"] = pca(adata_rna_high, n_comps=32)
adata_protein.obsm["feat"] = pca(adata_protein, n_comps=adata_protein.n_vars - 1)


############################################################
# Part 3: Build graphs per mode & count edges
############################################################
edge_results = []
for mode in GRAPH_MODES:
    print(f"\n===== {mode.upper()} =====")
    modalities = [adata_rna.copy(), adata_protein.copy()]

    (
        g_spatial_rna,
        g_feature_rna,
        g_spatial_protein,
        g_feature_protein,
        _,
        _,
    ) = Sagegraph(
        modalities,
        device,
        datatype="my_data",
        batch=False,
        graph_mode=mode,
    )

    if mode == "attention":
        rna_spatial_edges = g_spatial_rna.num_edges()
        rna_feature_edges = g_feature_rna.num_edges()
        protein_spatial_edges = g_spatial_protein.num_edges()
        protein_feature_edges = g_feature_protein.num_edges()
        print(f"RNA: spatial={rna_spatial_edges}, feature={rna_feature_edges}")
        print(f"Protein: spatial={protein_spatial_edges}, feature={protein_feature_edges}")
    else:
        rna_spatial_edges = g_spatial_rna.num_edges()
        rna_feature_edges = ""
        protein_spatial_edges = g_spatial_protein.num_edges()
        protein_feature_edges = ""
        print(f"RNA edges: {rna_spatial_edges}")
        print(f"Protein edges: {protein_spatial_edges}")

    edge_results.append({
        "Graph Mode": mode,
        "RNA Spatial": rna_spatial_edges,
        "RNA Feature": rna_feature_edges,
        "Protein Spatial": protein_spatial_edges,
        "Protein Feature": protein_feature_edges,
    })

edge_df = pd.DataFrame(edge_results)
edge_df.to_csv(os.path.join(BASE, "graph_edge_counts.csv"), index=False)
print("\nSaved edge counts to graph_edge_counts.csv")


############################################################
# Part 4: Summarize protein imputation metrics per experiment
############################################################
metrics_results = []
for mode, folder in EXPERIMENTS.items():
    file = os.path.join(BASE, folder, "evaluation", "protein_metrics.csv")
    df = pd.read_csv(file)

    metrics_results.append({
        "Graph Mode": mode,
        "Mean Pearson": np.nanmean(df["Pearson"]),
        "Median Pearson": np.nanmedian(df["Pearson"]),
        "Mean Spearman": np.nanmean(df["Spearman"]),
        "Median Spearman": np.nanmedian(df["Spearman"]),
        "Mean RMSE": df["RMSE"].mean(),
        "Median RMSE": df["RMSE"].median(),
    })

metrics_df = pd.DataFrame(metrics_results).round(4)
metrics_df.to_csv(os.path.join(BASE, "graph_mode_summary.csv"), index=False)
print("\nSaved metrics summary to graph_mode_summary.csv")


############################################################
# Part 5: Merge edge counts + metrics into one combined table
############################################################
combined = pd.merge(edge_df, metrics_df, on="Graph Mode", how="outer")
combined_path = os.path.join(BASE, "graph_mode_combined_summary.csv")
combined.to_csv(combined_path, index=False)

print("\n=== Combined summary (edges + protein metrics) ===")
print(combined)
print(f"\nSaved combined summary to {combined_path}")
print("\nDone! :)")

############################################################
# Part 6: Stratify proteins by prediction accuracy to see
#          why some proteins are harder to predict than others
#          -- run for EVERY graph mode, not just attention --
############################################################
from scipy.stats import pearsonr

# --- Protein expression characteristics are mode-independent, so compute once ---
raw_protein_counts = protein  # loaded in Part 1, cells x proteins, raw counts
clr_values = pd.DataFrame(
    adata_protein.X,
    index=adata_protein.obs_names,
    columns=adata_protein.var_names,
)

protein_stats = pd.DataFrame({
    "Mean Raw Expression": raw_protein_counts.mean(axis=0),
    "Detection Rate": (raw_protein_counts > 0).mean(axis=0),
    "CLR Variance": clr_values.var(axis=0),
})
protein_stats.index.name = "Protein"
protein_stats = protein_stats.reset_index()
n_proteins = len(protein_stats)

all_tier_summaries = []   # collects one tier_summary per mode (for cross-mode comparison)
all_correlations = []     # collects r/p per mode per feature (for cross-mode comparison)

for mode in GRAPH_MODES:
    stratify_folder = EXPERIMENTS[mode]
    stratify_file = os.path.join(BASE, stratify_folder, "evaluation", "protein_metrics.csv")

    per_protein = pd.read_csv(stratify_file)
    per_protein = per_protein.rename(columns={"Protein": "Protein_Index"})

    # Identify the protein-name column (fallback logic, same as before)
    candidate_cols = [c for c in per_protein.columns if c not in ("Pearson", "Spearman", "RMSE")]
    PROTEIN_COL = candidate_cols[0] if candidate_cols else None

    # Decide join strategy: by name if it overlaps with real protein names,
    # otherwise fall back to positional alignment.
    if PROTEIN_COL is not None:
        name_overlap = per_protein[PROTEIN_COL].astype(str).isin(
            protein_stats["Protein"].astype(str)
        ).mean()
    else:
        name_overlap = 0

    if PROTEIN_COL is not None and per_protein[PROTEIN_COL].dtype == object and name_overlap > 0.5:
        stratified = per_protein.merge(
            protein_stats, left_on=PROTEIN_COL, right_on="Protein", how="left"
        )
    else:
        n_metrics = len(per_protein)
        if n_metrics != n_proteins:
            raise ValueError(
                f"[{mode}] Cannot positionally align: protein_metrics.csv has "
                f"{n_metrics} rows but adata_protein has {n_proteins} proteins."
            )
        stratified = pd.concat(
            [per_protein.reset_index(drop=True), protein_stats.reset_index(drop=True)],
            axis=1,
        )

    # --- Stratify into Low / Medium / High Pearson tertiles ---
    stratified["Pearson Tier"] = pd.qcut(
        stratified["Pearson"], q=3, labels=["Low", "Medium", "High"]
    )

    tier_summary = stratified.groupby("Pearson Tier", observed=True).agg(
        n_proteins=("Pearson", "size"),
        mean_pearson=("Pearson", "mean"),
        mean_rmse=("RMSE", "mean"),
        mean_raw_expression=("Mean Raw Expression", "mean"),
        mean_detection_rate=("Detection Rate", "mean"),
        mean_clr_variance=("CLR Variance", "mean"),
    ).round(4)

    print(f"\n=== Protein stratification by Pearson tier ({mode} mode) ===")
    print(tier_summary)

    tier_summary_for_combo = tier_summary.reset_index()
    tier_summary_for_combo.insert(0, "Graph Mode", mode)
    all_tier_summaries.append(tier_summary_for_combo)

    # --- Correlation between per-protein Pearson and expression features ---
    for feature in ["Mean Raw Expression", "Detection Rate", "CLR Variance"]:
        valid = stratified[["Pearson", feature]].dropna()
        r, p = pearsonr(valid["Pearson"], valid[feature])
        print(f"[{mode}] Correlation between per-protein Pearson and {feature}: r={r:.3f}, p={p:.3g}")
        all_correlations.append({
            "Graph Mode": mode, "Feature": feature, "r": round(r, 4), "p": p,
        })

    # --- Save full per-protein table (sorted worst-predicted first) ---
    stratified_sorted = stratified.sort_values("Pearson").reset_index(drop=True)
    stratified_path = os.path.join(BASE, f"protein_stratification_{mode}.csv")
    tier_summary_path = os.path.join(BASE, f"protein_stratification_tier_summary_{mode}.csv")

    stratified_sorted.to_csv(stratified_path, index=False)
    tier_summary.to_csv(tier_summary_path)

    print(f"[{mode}] Hardest-to-predict proteins (lowest Pearson):")
    print(stratified_sorted.head(10)[
        ["Protein", "Pearson", "RMSE", "Mean Raw Expression", "Detection Rate", "CLR Variance"]
    ])
    print(f"[{mode}] Saved per-protein stratification to {stratified_path}")
    print(f"[{mode}] Saved tier summary to {tier_summary_path}")

# --- Combined cross-mode comparison tables ---
combined_tier_summary = pd.concat(all_tier_summaries, ignore_index=True)
combined_tier_summary_path = os.path.join(BASE, "protein_stratification_tier_summary_ALL_MODES.csv")
combined_tier_summary.to_csv(combined_tier_summary_path, index=False)

combined_correlations = pd.DataFrame(all_correlations)
combined_correlations_path = os.path.join(BASE, "protein_stratification_correlations_ALL_MODES.csv")
combined_correlations.to_csv(combined_correlations_path, index=False)

print("\n=== Cross-mode tier summary comparison ===")
print(combined_tier_summary)
print(f"Saved to {combined_tier_summary_path}")

print("\n=== Cross-mode correlation comparison (does sparsity drive poor prediction in every mode?) ===")
print(combined_correlations.pivot(index="Feature", columns="Graph Mode", values="r").round(3))
print(f"Saved to {combined_correlations_path}")

print("\nDone! :)")
