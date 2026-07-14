import scanpy as sc
import pandas as pd
import numpy as np
import torch as th
import torch.nn as nn
from anndata import AnnData

# from SpaMIE.create_graph import Sagegraph # original
from SpaMIE.create_graph_v2 import Sagegraph # modified versions 
from SpaMIE.spamie_main import Sagewrapper
from SpaMIE.preprocess import clr_normalize_each_cell, pca

device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

########################################################
# Load RNA and Protein
########################################################

rna = pd.read_csv(
    "/users/coh33/SpaMIE/data/human_skin/GSM6578065_humanskin_RNA.tsv.gz",
    sep="\t",
    index_col=0
)

protein = pd.read_csv(
    "/users/coh33/SpaMIE/data/human_skin/GSM6578074_humanskin_protein.tsv.gz",
    sep="\t",
    index_col=0
)

print("RNA shape:", rna.shape)
print("Protein shape:", protein.shape)

########################################################
# Align cells
########################################################

protein = protein.loc[rna.index]

assert np.all(rna.index == protein.index)

########################################################
# Create AnnData
########################################################

adata_omics1 = AnnData(rna)
adata_omics2 = AnnData(protein)

########################################################
# Spatial coordinates
########################################################

coords = np.array([
    list(map(float, cell.split("x")))
    for cell in adata_omics1.obs_names
])

adata_omics1.obsm["spatial"] = coords
adata_omics2.obsm["spatial"] = coords

########################################################
# SpaMIE Stereo-CITE-seq preprocessing
########################################################

print("\nPreprocessing RNA...")

sc.pp.filter_genes(
    adata_omics1,
    min_cells=0
)

sc.pp.filter_cells(
    adata_omics1,
    min_genes=0
)

sc.pp.highly_variable_genes(
    adata_omics1,
    flavor="seurat_v3",
    n_top_genes=3000
)

sc.pp.normalize_total(
    adata_omics1,
    target_sum=1e4
)

sc.pp.log1p(
    adata_omics1
)

adata_omics1_high = adata_omics1[
    :,
    adata_omics1.var["highly_variable"]
].copy()

print("Number of HVGs:", adata_omics1_high.n_vars)

########################################################
# Protein preprocessing
########################################################

print("Preprocessing protein (CLR)...")

adata_omics2 = clr_normalize_each_cell(
    adata_omics2
)

########################################################
# PCA (SpaMIE implementation)
########################################################

print("Running SpaMIE PCA...")

adata_omics1.obsm["feat"] = pca(
    adata_omics1_high,
    n_comps=32
)

adata_omics2.obsm["feat"] = pca(
    adata_omics2,
    n_comps=adata_omics2.n_vars - 1
)

########################################################
# Data summary
########################################################

print("\n========== DATA SUMMARY ==========")

print("RNA:", adata_omics1.shape)
print("Protein:", adata_omics2.shape)

print("RNA HVGs:", adata_omics1_high.n_vars)

print("Spatial:", adata_omics1.obsm["spatial"].shape)

print("RNA feat:", adata_omics1.obsm["feat"].shape)
print("Protein feat:", adata_omics2.obsm["feat"].shape)

########################################################
# Graph construction
########################################################

modalities = [
    adata_omics1,
    adata_omics2
]

(
    g_spatial_omics1,
    g_feature_omics1,
    g_spatial_omics2,
    g_feature_omics2,
    adata_omics1,
    adata_omics2
) = Sagegraph(
    modalities,
    device,
    datatype="my_data",
    batch=False, 
    graph_mode = "feature" 
    # if you want to make it original, remove the comma after batch= False and graph_mode 
)

########################################################
# Paper 3:1 train/test split
########################################################

train_size = int(0.75 * adata_omics1.n_obs)

print("\nTraining cells :", train_size)
print("Testing cells  :", adata_omics1.n_obs - train_size)

########################################################
# Build model
########################################################

weight = [0, 0, 2]

model = Sagewrapper(
    seed=1,
    device=device,
    in_feat=adata_omics1.obsm["feat"].shape[1],
    n_hidden=256,
    out_feat=adata_omics2.X.shape[1],
    task="prediction",
    datatype="my_data",
    layers_nums=3,
    weight=weight,
    epoch=350,
    res_type="res_add",
    activation=nn.LeakyReLU,
    sagetype="mean",
    lr=2e-4,
    lr2=0.002
)

########################################################
# Train
########################################################

model.fit(
    g_spatial_omics1,
    g_feature_omics1,
    g_spatial_omics2,
    g_feature_omics2,
    adata_omics1,
    adata_omics2,
    output_dir="/users/coh33/SpaMIE/results/human_skin_feature/",
    pred_name="human_skin_feature_pred.csv",
    true_name="human_skin_feature_truth.csv",
    train_size=train_size,
    weight=True,
    save_csv=True
)

print("\nFinished!")
