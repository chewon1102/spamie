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
    # 5 separate cell-feature runs (one per graph mode), from
    # train_cell_feat_test.py + compute_cellfeat_metrics.py.
    "attention_cellfeat": "human_skin_attention_cellfeat",
    "union_cellfeat": "human_skin_union_cellfeat",
    "intersection_cellfeat": "human_skin_intersection_cellfeat",
    "spatial_cellfeat": "human_skin_spatial_cellfeat",
    "feature_cellfeat": "human_skin_feature_cellfeat",
    # 5 separate uncertainty-weighted-loss runs (one per graph mode),
    # from train_uncertainty_test.py + compute_uncertainty_metrics.py.
    "attention_uncertainty": "human_skin_attention_uncertainty",
    "union_uncertainty": "human_skin_union_uncertainty",
    "intersection_uncertainty": "human_skin_intersection_uncertainty",
    "spatial_uncertainty": "human_skin_spatial_uncertainty",
    "feature_uncertainty": "human_skin_feature_uncertainty",
}

# GRAPH_MODES: only the modes Sagegraph() actually knows how to build
# (used in Part 3, edge counting). The cellfeat/uncertainty variants are
# NOT real Sagegraph graph_mode values — they were trained through
# separate scripts, so they're excluded here.
GRAPH_MODES = ["attention", "union", "intersection", "spatial", "feature"]

# ALL_MODES: every mode with a protein_metrics.csv on disk, including
# all 5 cellfeat variants and all 5 uncertainty variants.
ALL_MODES = GRAPH_MODES + [
    "attention_cellfeat",
    "union_cellfeat",
    "intersection_cellfeat",
    "spatial_cellfeat",
    "feature_cellfeat",
    "attention_uncertainty",
    "union_uncertainty",
    "intersection_uncertainty",
    "spatial_uncertainty",
    "feature_uncertainty",
]

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

# --- NEW: check explained variance for both PCA embeddings ---
from sklearn.decomposition import PCA as SklearnPCA
import scipy.sparse

def pca_with_variance(adata, n_comps):
    X = adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X
    pca_model = SklearnPCA(n_components=n_comps)
    feat_pca = pca_model.fit_transform(X)
    return feat_pca, pca_model.explained_variance_ratio_

_, rna_var_ratio = pca_with_variance(adata_rna_high, n_comps=32)
_, protein_var_ratio = pca_with_variance(adata_protein, n_comps=adata_protein.n_vars - 1)

print(f"RNA: {rna_var_ratio.sum():.1%} total variance explained by 32 PCs")
print(f"Protein: {protein_var_ratio.sum():.1%} total variance explained by {adata_protein.n_vars-1} PCs")

pd.DataFrame({
    "Component": range(1, len(rna_var_ratio) + 1),
    "Explained Variance Ratio": rna_var_ratio,
    "Cumulative": np.cumsum(rna_var_ratio),
}).to_csv(os.path.join(BASE, "rna_pca_explained_variance.csv"), index=False)

pd.DataFrame({
    "Component": range(1, len(protein_var_ratio) + 1),
    "Explained Variance Ratio": protein_var_ratio,
    "Cumulative": np.cumsum(protein_var_ratio),
}).to_csv(os.path.join(BASE, "protein_pca_explained_variance.csv"), index=False)


############################################################
# Part 2b: Cell-feature bipartite graph construction (task 2)
# Standalone — builds the graph and reports stats only.
# Does NOT plug into SpaMIE_pred yet; that's a separate step
# once this structure is confirmed to look reasonable.
############################################################
import dgl
import scipy.sparse as sp


def build_cell_feature_graph(expr_matrix, min_expr=0, top_k_genes_per_spot=None):
    """
    Build a bipartite spot<->gene (or spot<->protein) graph from an
    expression matrix.

    expr_matrix: array-like, shape (n_spots, n_genes) — raw or normalized counts
    min_expr: minimum expression value to count as an edge (0 = any nonzero)
    top_k_genes_per_spot: if set, only keep each spot's top-k most
        expressed genes as edges (mirrors scMoGNN-style sparsification,
        since keeping every nonzero entry can make the graph very dense)

    Returns a dgl.heterograph with two node types: 'spot' and 'gene'
    """
    X = expr_matrix.toarray() if sp.issparse(expr_matrix) else np.asarray(expr_matrix)
    n_spots, n_genes = X.shape

    if top_k_genes_per_spot is not None:
        # Keep only each spot's top-k expressed genes as edges
        spot_ids, gene_ids = [], []
        for i in range(n_spots):
            row = X[i]
            if top_k_genes_per_spot < n_genes:
                top_idx = np.argpartition(row, -top_k_genes_per_spot)[-top_k_genes_per_spot:]
            else:
                top_idx = np.arange(n_genes)
            top_idx = top_idx[row[top_idx] > min_expr]
            spot_ids.extend([i] * len(top_idx))
            gene_ids.extend(top_idx.tolist())
    else:
        # Keep every edge above the expression threshold
        spot_ids, gene_ids = np.nonzero(X > min_expr)
        spot_ids, gene_ids = spot_ids.tolist(), gene_ids.tolist()

    spot_ids = th.tensor(spot_ids, dtype=th.int64)
    gene_ids = th.tensor(gene_ids, dtype=th.int64)

    g = dgl.heterograph(
        {
            ("spot", "expresses", "gene"): (spot_ids, gene_ids),
            ("gene", "expressed_by", "spot"): (gene_ids, spot_ids),
        },
        num_nodes_dict={"spot": n_spots, "gene": n_genes},
    )
    return g


# --- Build it for RNA and inspect basic stats ---
rna_expr = adata_rna_high.X  # HVG-filtered, normalized, log1p'd RNA matrix

g_cf_rna = build_cell_feature_graph(
    rna_expr,
    min_expr=0,
    top_k_genes_per_spot=50,  # cap to keep the graph a manageable size; None = keep all nonzero
)

print("\n=== RNA cell-feature bipartite graph ===")
print(g_cf_rna)
print(f"spot nodes: {g_cf_rna.num_nodes('spot')}")
print(f"gene nodes: {g_cf_rna.num_nodes('gene')}")
print(f"spot->gene edges: {g_cf_rna.num_edges('expresses')}")
print(f"avg genes per spot: {g_cf_rna.num_edges('expresses') / g_cf_rna.num_nodes('spot'):.1f}")
print(f"avg spots per gene: {g_cf_rna.num_edges('expressed_by') / g_cf_rna.num_nodes('gene'):.1f}")

# --- Same thing for protein (small panel, keep every nonzero edge) ---
protein_expr = adata_protein.X

g_cf_protein = build_cell_feature_graph(
    protein_expr,
    min_expr=0,
    top_k_genes_per_spot=None,
)

print("\n=== Protein cell-feature bipartite graph ===")
print(g_cf_protein)
print(f"spot nodes: {g_cf_protein.num_nodes('spot')}")
print(f"protein nodes: {g_cf_protein.num_nodes('gene')}")
print(f"spot->protein edges: {g_cf_protein.num_edges('expresses')}")
print(f"avg proteins per spot: {g_cf_protein.num_edges('expresses') / g_cf_protein.num_nodes('spot'):.1f}")
print(f"avg spots per protein: {g_cf_protein.num_edges('expressed_by') / g_cf_protein.num_nodes('gene'):.1f}")


############################################################
# Part 2c-prep: HeteroGraphConv encoder definition
# (standalone here for testing; will move into spamie_net.py
# once confirmed working)
############################################################
import torch.nn as nn
import dgl.nn.pytorch as dglnn


class HeteroCellFeatureEncoder(nn.Module):
    """
    Encodes a bipartite spot<->gene (or spot<->protein) graph into
    per-spot embeddings, using alternating spot->gene->spot message
    passing (dgl HeteroGraphConv).

    Gene/protein nodes have no natural continuous features, so they
    get a learnable embedding table (nn.Embedding), trained end-to-end
    — same idea scMoGNN uses for feature nodes.
    """

    def __init__(self, n_genes, in_feat_spot, n_hidden, n_layers=2,
                 dropout=0.2, activation=None, batchnorm=True):
        super().__init__()
        self.n_layers = n_layers
        self.activation = activation
        self.batchnorm = batchnorm

        # Learnable gene/protein embeddings (no real input features exist for them)
        self.gene_embed = nn.Embedding(n_genes, n_hidden)
        nn.init.xavier_uniform_(self.gene_embed.weight)

        # Project spot's real features (PCA) to n_hidden for the first layer
        self.spot_proj = nn.Linear(in_feat_spot, n_hidden)

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(
                dglnn.HeteroGraphConv({
                    "expresses":    dglnn.SAGEConv(n_hidden, n_hidden, "mean"),  # spot -> gene
                    "expressed_by": dglnn.SAGEConv(n_hidden, n_hidden, "mean"),  # gene -> spot
                }, aggregate="sum")
            )

        if batchnorm:
            self.bn_spot = nn.ModuleList([nn.BatchNorm1d(n_hidden) for _ in range(n_layers)])
            self.bn_gene = nn.ModuleList([nn.BatchNorm1d(n_hidden) for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

    def forward(self, g, feat_spot):
        """
        g: dgl.heterograph with node types 'spot' and 'gene'
        feat_spot: (n_spots, in_feat_spot) — the existing PCA embedding
        Returns: (n_spots, n_hidden) per-spot embedding
        """
        h_spot = self.spot_proj(feat_spot)
        h_gene = self.gene_embed.weight  # (n_genes, n_hidden), learnable

        h = {"spot": h_spot, "gene": h_gene}

        for i, layer in enumerate(self.layers):
            h = layer(g, h)

            if self.batchnorm:
                h["spot"] = self.bn_spot[i](h["spot"])
                h["gene"] = self.bn_gene[i](h["gene"])

            if self.activation is not None:
                h["spot"] = self.activation(h["spot"])
                h["gene"] = self.activation(h["gene"])

            if i < self.n_layers - 1:
                h["spot"] = self.dropout(h["spot"])
                h["gene"] = self.dropout(h["gene"])

        return h["spot"]  # only the spot-side embedding matters downstream


############################################################
# Part 2c: Quick standalone test of HeteroCellFeatureEncoder
# (sanity check only — not wired into training yet)
############################################################
encoder_test = HeteroCellFeatureEncoder(
    n_genes=g_cf_rna.num_nodes("gene"),
    in_feat_spot=adata_rna.obsm["feat"].shape[1],  # 32, matches RNA PCA dim
    n_hidden=256,
    n_layers=2,
    activation=nn.LeakyReLU(),
)

feat_spot_test = th.tensor(adata_rna.obsm["feat"], dtype=th.float32)
spot_embedding = encoder_test(g_cf_rna, feat_spot_test)

print("\n=== HeteroCellFeatureEncoder test ===")
print(f"Output shape: {spot_embedding.shape}")  # should be (1691, 256)
print(f"Output stats: mean={spot_embedding.mean().item():.4f}, std={spot_embedding.std().item():.4f}")


############################################################
# Part 2d: Standalone forward-pass test of SpaMIE_pred
#          with use_cellfeat=True (cell-feature graph enabled)
# Sanity check only — no training, no optimizer, single forward call.
############################################################
from SpaMIE.spamie_net_v2 import SpaMIE_pred

# Get one set of spatial/feature graphs to test with (attention mode,
# since that's your main model's graph combination style)
modalities_test = [adata_rna.copy(), adata_protein.copy()]
(
    g_spatial_rna_test,
    g_feature_rna_test,
    g_spatial_protein_test,
    g_feature_protein_test,
    _,
    _,
) = Sagegraph(
    modalities_test,
    device,
    datatype="my_data",
    batch=False,
    graph_mode="attention",
)

# Build the model with the cell-feature path turned on
model_test = SpaMIE_pred(
    in_feats=adata_rna.obsm["feat"].shape[1],   # 32
    n_hidden=256,
    out_feats=adata_protein.n_vars,             # 283
    wt=[0, 0, 2],
    activation=nn.LeakyReLU,
    sagetype="mean",
    layers_nums=3,
    res="res_add",
    batchnorm=True,
    dropout=0.2,
    use_cellfeat=True,
    n_genes_cellfeat=g_cf_rna.num_nodes("gene"),  # 3000, matches RNA HVGs
)

feat_omics1_test = th.tensor(adata_rna.obsm["feat"], dtype=th.float32)

# IMPORTANT: put model in eval mode for this sanity check.
# BatchNorm layers require batch size > 1 in train mode, and more
# importantly we don't want dropout/backprop noise for a pure shape check.
model_test.eval()

with th.no_grad():
    output, wt, alph, latents = model_test(
        g_spatial_rna_test,
        g_feature_rna_test,
        feat_omics1_test,
        weight=True,
        g_cellfeat_omics1=g_cf_rna,
    )

print("\n=== SpaMIE_pred forward-pass test (use_cellfeat=True) ===")
print(f"Output shape (predicted protein): {output.shape}")       # expect (1691, 283)
print(f"Latent embedding shape: {latents.shape}")                # expect (1691, 256)
print(f"wt (layer-combination weights): {[w.shape for w in wt]}")
print(f"Attention alpha shape: {alph.shape}")                     # expect (1691, 3) -- 3 = spatial/feature/cellfeat
print(f"Attention alpha sample (first spot): {alph[0]}")          # should sum to ~1.0
print(f"Attention alpha mean across all spots: {alph.mean(dim=0)}")  # avg weight given to each of the 3 graph types


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

for mode in ALL_MODES:
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

    # --- NEW: does RMSE measure the same thing as Pearson? ---
    valid_rmse = stratified[["Pearson", "RMSE"]].dropna()
    r_rmse, p_rmse = pearsonr(valid_rmse["Pearson"], valid_rmse["RMSE"])
    print(f"[{mode}] Correlation between per-protein Pearson and RMSE: r={r_rmse:.3f}, p={p_rmse:.3g}")
    all_correlations.append({
        "Graph Mode": mode, "Feature": "RMSE", "r": round(r_rmse, 4), "p": p_rmse,
    })

    # --- NEW: weighted RMSE, using each protein's detection rate as its weight ---
    # Proteins detected in more spots contribute more to this aggregate score.
    # Compare against the plain (unweighted) mean RMSE already in tier_summary.
    valid_w = stratified[["RMSE", "Detection Rate"]].dropna()
    weighted_rmse = np.sqrt(
        np.sum(valid_w["Detection Rate"] * valid_w["RMSE"] ** 2) / np.sum(valid_w["Detection Rate"])
    )
    plain_mean_rmse = stratified["RMSE"].mean()
    print(f"[{mode}] Plain mean RMSE: {plain_mean_rmse:.4f} | "
          f"Detection-rate-weighted RMSE: {weighted_rmse:.4f}")

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

    # --- NEW: per-protein visualizations — overall first, then ranked, then quantile ---
    import matplotlib
    matplotlib.use("Agg")  # headless-safe for cluster environments
    import matplotlib.pyplot as plt

    viz_dir = os.path.join(BASE, "protein_boxplots")
    os.makedirs(viz_dir, exist_ok=True)

    # 1) Overall boxplot — always shown first
    fig, ax = plt.subplots(figsize=(4, 5))
    ax.boxplot(stratified["Pearson"].dropna())
    ax.set_title(f"Overall Pearson distribution ({mode})")
    ax.set_ylabel("Pearson correlation")
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f"pearson_overall_boxplot_{mode}.png"), dpi=150)
    plt.close()

    # 2) By rank-defined decile
    stratified["Rank Decile"] = pd.qcut(stratified["Pearson"].rank(method="first"), 10, labels=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    stratified.boxplot(column="Pearson", by="Rank Decile", ax=ax)
    ax.set_title(f"Pearson by rank decile ({mode})")
    plt.suptitle("")
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f"pearson_by_rank_decile_{mode}.png"), dpi=150)
    plt.close()

    # 3) By quantile of detection rate (checks linearity of the relationship)
    stratified["Detection Quantile"] = pd.qcut(
        stratified["Detection Rate"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"]
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    stratified.boxplot(column="Pearson", by="Detection Quantile", ax=ax)
    ax.set_title(f"Pearson by detection-rate quantile ({mode})")
    plt.suptitle("")
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f"pearson_by_detection_quantile_{mode}.png"), dpi=150)
    plt.close()

    print(f"[{mode}] Saved 3 boxplot visualizations to {viz_dir}")

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
