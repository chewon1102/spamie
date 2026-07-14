import scanpy as sc
import scvi
import numpy as np
import pandas as pd

adata_omics1 = sc.read_h5ad(
    "/users/coh33/SpaMIE/data/my_data/GSM8195494_A1_LN.h5ad"
)

adata_omics2 = sc.read_h5ad(
    "/users/coh33/SpaMIE/data/my_data/GSM8195498_A1_LN_Protein.h5ad"
)


adata = adata_omics1.copy()

protein_matrix = (
    adata_omics2.X.toarray()
    if hasattr(adata_omics2.X, "toarray")
    else adata_omics2.X
)

adata.obsm["protein_expression"] = protein_matrix

scvi.model.TOTALVI.setup_anndata(
    adata,
    protein_expression_obsm_key="protein_expression"
)

model = scvi.model.TOTALVI(
    adata,
    n_latent=20
)

model.train(max_epochs=200)

result = model.get_normalized_expression(
    n_samples=25
)

print(type(result))

if isinstance(result, tuple):
    rna_exp, protein_pred = result
else:
    protein_pred = result

protein_pred = pd.DataFrame(
    protein_pred,
    index=adata.obs_names,
    columns=adata_omics2.var_names
)

protein_pred.to_csv(
    "/users/coh33/SpaMIE/results/my_data/totalVI_pred.csv"
)

pd.DataFrame(
    protein_matrix,
    index=adata.obs_names,
    columns=adata_omics2.var_names
).to_csv(
    "/users/coh33/SpaMIE/results/my_data/totalVI_truth.csv"
)
