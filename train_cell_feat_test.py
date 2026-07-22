"""
Train SpaMIE WITH the cell-feature bipartite graph enabled (task 2),
looping across all 5 graph modes (attention, union, intersection,
spatial, feature), on the human-skin CITE-seq dataset.

This mirrors the exact preprocessing used in compare_graph_modes.py,
then trains for real (not just a forward-pass sanity check) using
Sagewrapper from spamie_main_cellfeat. Each mode's results are saved
to its own output directory (human_skin_{mode}_cellfeat/), so none
of your existing baselines (human_skin_hvg, human_skin_union, etc.)
are touched.
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
import torch as th
import torch.nn as nn
import dgl
import scipy.sparse as sp

from SpaMIE.preprocess import clr_normalize_each_cell, pca
from SpaMIE.create_graph_v2 import Sagegraph
from SpaMIE.spamie_main_cellfeat import Sagewrapper

device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

DATA_DIR = "/users/coh33/SpaMIE/data/human_skin"
BASE_OUTPUT_DIR = "/users/coh33/SpaMIE/results"

# NEW: loop over all 5 modes instead of hardcoding just "attention"
GRAPH_MODES = ["attention", "union", "intersection", "spatial", "feature"]


############################################################
# Part 1: Load data (identical to compare_graph_modes.py)
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
# Part 2: Preprocessing (identical to compare_graph_modes.py)
# Done once — same PCA embeddings are reused for every mode below.
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
# Part 2b: Build the cell-feature bipartite graph
# (mode-independent — built once, reused for every mode below)
############################################################
def build_cell_feature_graph(expr_matrix, min_expr=0, top_k_genes_per_spot=None):
    X = expr_matrix.toarray() if sp.issparse(expr_matrix) else np.asarray(expr_matrix)
    n_spots, n_genes = X.shape

    if top_k_genes_per_spot is not None:
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


rna_expr = adata_rna_high.X
g_cf_rna = build_cell_feature_graph(
    rna_expr,
    min_expr=0,
    top_k_genes_per_spot=50,
)
# Move to the training device (GPU if available)
g_cf_rna = g_cf_rna.to(device)

print(f"Cell-feature graph (shared across all modes): "
      f"{g_cf_rna.num_nodes('spot')} spots, "
      f"{g_cf_rna.num_nodes('gene')} genes, "
      f"{g_cf_rna.num_edges('expresses')} spot->gene edges\n")


############################################################
# Part 3: Loop over all 5 graph modes, training with
#          use_cellfeat=True each time
# (this REPLACES the old single-mode "attention only" block)
############################################################
weight = [0, 0, 2]
in_feat = adata_rna.obsm["feat"].shape[1]
out_feat = adata_protein.X.shape[1]
n_genes_cellfeat = g_cf_rna.num_nodes("gene")

print("in_feat =", in_feat)
print("out_feat =", out_feat)
print("n_genes_cellfeat =", n_genes_cellfeat)

for mode in GRAPH_MODES:
    print(f"\n===== TRAINING {mode.upper()} + CELL-FEATURE =====")

    # output_dir = os.path.join(BASE_OUTPUT_DIR, f"human_skin_{mode}_cellfeat/")
    output_dir = os.path.join(BASE_OUTPUT_DIR, f"human_skin_{mode}_e200_cellfeat/")
    os.makedirs(output_dir, exist_ok=True)

    # Build spatial + feature graphs for this specific mode
    modalities = [adata_rna.copy(), adata_protein.copy()]

    (
        g_spatial_omics1,
        g_feature_omics1,
        g_spatial_omics2,
        g_feature_omics2,
        adata_omics1,
        adata_omics2,
    ) = Sagegraph(
        modalities,
        device,
        datatype="my_data",
        batch=False,
        graph_mode=mode,
    )

    model = Sagewrapper(
        seed=1,
        device=device,
        in_feat=in_feat,
        n_hidden=256,
        out_feat=out_feat,
        task="prediction",
        datatype="my_data",
        layers_nums=3,
        weight=weight,
        epoch=200,
        res_type="res_add",
        activation=nn.LeakyReLU,
        sagetype="mean",
        lr=2e-4,
        lr2=0.002,
        use_cellfeat=True,
        n_genes_cellfeat=n_genes_cellfeat,
    )

    result = model.fit(
        g_spatial_omics1,
        g_feature_omics1,
        g_spatial_omics2,
        g_feature_omics2,
        adata_omics1,
        adata_omics2,
        output_dir=output_dir,
        pred_name=f"human_skin_{mode}_cellfeat_pred.csv",
        true_name=f"human_skin_{mode}_cellfeat_truth.csv",
        weight=True,
        save_csv=True,
        g_cellfeat_omics1=g_cf_rna,
    )

    print(f"[{mode}] Finished. Results saved to {output_dir}")

print("\nAll 5 modes trained with cell-feature graph enabled.")
