import scanpy as sc
import torch as th
import torch.nn as nn

from SpaMIE.create_graph import Sagegraph
from SpaMIE.spamie_main import Sagewrapper

device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')

# Load your data
# same section two modalitty files 
adata_omics1 = sc.read_h5ad(
    # "/users/coh33/SpaMIE/data/my_data/GSM8195494_A1_LN.h5ad"
    
)

adata_omics2 = sc.read_h5ad(
    # "/users/coh33/SpaMIE/data/my_data/GSM8195498_A1_LN_Protein.h5ad"
    
)

modalities = [adata_omics1, adata_omics2]

# Build graphs
# Computes a spatial KMN graph from coordinates and a feature KNN graphs from PCA embeddings 
# Then wraps both as DGL grpahs with node features attached. 
g_spatial_omics1, g_feature_omics1, \
g_spatial_omics2, g_feature_omics2, \
adata_omics1, adata_omics2 = Sagegraph(
    modalities,
    device,
    datatype='my_data',
    batch=False
    # Not applying batch correction across sections 
)

output_dir = "/users/coh33/SpaMIE/results/my_data/"
weight = [0, 0, 2]
# initial logits per layer embedding weighting 

in_feat = adata_omics1.obsm['feat'].shape[1]
out_feat = adata_omics2.X.shape[1]

print("in_feat =", in_feat)
# comes from omic1's PCA based feat matrix 
print("out_feat =", out_feat)
# raw # of features in omics 2's 

# Builds two optimizer param groups: wt1/wt2 get lr2 = 0.002 
# splits spots within omic1 itself first 75% 
# 3:1 SMO training : mono omics testing split, not the cross section transfer. 
model = Sagewrapper(
    seed=1,
    device=device,
    in_feat=in_feat,
    n_hidden=256,
    out_feat=out_feat,
    task='prediction',
    datatype='my_data',
    layers_nums=3,
    weight=weight,
    epoch=50,          # use 50 first for testing
    res_type='res_add',
    activation=nn.LeakyReLU,
    sagetype='mean',
    lr=2e-4,
    lr2=0.002
)

result = model.fit(
    g_spatial_omics1,
    g_feature_omics1,
    g_spatial_omics2,
    g_feature_omics2,
    adata_omics1,
    adata_omics2,
    output_dir=output_dir,
    pred_name="my_data_pred.csv",
    true_name="my_data_truth.csv",
    weight=True,
    save_csv=True
)
print("Finished")









