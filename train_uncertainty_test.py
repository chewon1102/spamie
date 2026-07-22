"""
Train SpaMIE WITH per-protein uncertainty-weighted loss enabled
(Kendall et al. 2018 homoscedastic uncertainty), looping across all
5 graph modes (attention, union, intersection, spatial, feature),
on the human-skin CITE-seq dataset.

use_cellfeat=False here on purpose — this isolates the effect of
uncertainty-weighted loss on your baseline cell-cell graphs, as a
clean, single-variable comparison against my existing baseline
results (human_skin_hvg, human_skin_union, etc.). The cell-feature
and uncertainty-weighting features CAN be combined later (both flags
are independent), but keeping them separate first tells you which
change is actually responsible for any improvement.

Each mode's results are saved to its own output directory
(human_skin_{mode}_uncertainty/), so none of my existing baselines
or cell-feature results are touched.
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
import torch as th
import torch.nn as nn

from SpaMIE.preprocess import clr_normalize_each_cell, pca
from SpaMIE.create_graph_v2 import Sagegraph
from SpaMIE.spamie_main_uncertainty import Sagewrapper

device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

DATA_DIR = "/users/coh33/SpaMIE/data/human_skin"
BASE_OUTPUT_DIR = "/users/coh33/SpaMIE/results"

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
# Part 3: Loop over all 5 graph modes, training with
#          use_uncertainty=True each time (use_cellfeat=False)
############################################################
weight = [0, 0, 2]
in_feat = adata_rna.obsm["feat"].shape[1]
out_feat = adata_protein.X.shape[1]

print("in_feat =", in_feat)
print("out_feat =", out_feat)

for mode in GRAPH_MODES:
    print(f"\n===== TRAINING {mode.upper()} + UNCERTAINTY-WEIGHTED LOSS =====")

    output_dir = os.path.join(BASE_OUTPUT_DIR, f"human_skin_{mode}_uncertainty/")
    os.makedirs(output_dir, exist_ok=True)

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
        epoch=200,  # matches your e200 cellfeat runs, for a fair comparison
        res_type="res_add",
        activation=nn.LeakyReLU,
        sagetype="mean",
        lr=2e-4,
        lr2=0.002,
        use_cellfeat=False,       # isolate uncertainty-weighting effect only
        use_uncertainty=True,     # NEW
    )

    result = model.fit(
        g_spatial_omics1,
        g_feature_omics1,
        g_spatial_omics2,
        g_feature_omics2,
        adata_omics1,
        adata_omics2,
        output_dir=output_dir,
        pred_name=f"human_skin_{mode}_uncertainty_pred.csv",
        true_name=f"human_skin_{mode}_uncertainty_truth.csv",
        weight=True,
        save_csv=True,
        g_cellfeat_omics1=None,   # not used since use_cellfeat=False
    )

    print(f"[{mode}] Finished. Results saved to {output_dir}")

print("\nAll 5 modes trained with uncertainty-weighted loss enabled.")
