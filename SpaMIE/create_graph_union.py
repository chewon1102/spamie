import scanpy as sc
import dgl
import torch as th
import numpy as np
from scipy.sparse import coo_matrix

from SpaMIE.preprocess import preprocessing


def edge_list_to_sparse(adj_dict, n):
    """Convert SpaMIE edge list to scipy sparse matrix."""

    rows = np.array(adj_dict["x"])
    cols = np.array(adj_dict["y"])

    data = np.ones(len(rows))

    return coo_matrix((data, (rows, cols)), shape=(n, n))


def Sagegraph(
    modalities,
    device,
    task="Prediction",
    test_idx_name=None,
    y_pred_name=None,
    pred_joint=False,
    datatype="Stereo-CITE-seq",
    batch=False,
    graph_mode="attention"
):

    adata_omics1 = modalities[0]
    adata_omics2 = modalities[1]

    adata_omics1.var_names_make_unique()
    adata_omics2.var_names_make_unique()

    data = preprocessing(
        adata_omics1,
        adata_omics2,
        task,
        test_idx_name,
        y_pred_name,
        pred_joint=pred_joint,
        datatype=datatype,
        batch=batch,
    )

    adata_omics1 = data["adata_omics1"]
    adata_omics2 = data["adata_omics2"]

    ##################################################
    # helper
    ##################################################

    def build_graph(adata):

        n = adata.n_obs

        spatial = edge_list_to_sparse(
            adata.uns["adj_spatial"],
            n
        ).tocsr()

        feature = adata.obsm["adj_feature"].tocsr()

        ###############################################
        # choose graph
        ###############################################

        if graph_mode == "attention":

            g_spatial = dgl.graph(
                (
                    adata.uns["adj_spatial"]["x"],
                    adata.uns["adj_spatial"]["y"],
                )
            )

            feature = feature.tocoo()

            g_feature = dgl.graph(
                (
                    feature.row,
                    feature.col,
                )
            )

            return g_spatial, g_feature

        elif graph_mode == "spatial":

            g = dgl.graph(
                (
                    adata.uns["adj_spatial"]["x"],
                    adata.uns["adj_spatial"]["y"],
                )
            )

            return g, g

        elif graph_mode == "feature":

            feature = feature.tocoo()

            g = dgl.graph(
                (
                    feature.row,
                    feature.col,
                )
            )

            return g, g

        elif graph_mode == "union":

            union = spatial.maximum(feature)

            union = union.tocoo()

            g = dgl.graph(
                (
                    union.row,
                    union.col,
                )
            )

            return g, g

        elif graph_mode == "intersection":

            inter = spatial.multiply(feature)

            inter.data[:] = 1

            inter = inter.tocoo()

            g = dgl.graph(
                (
                    inter.row,
                    inter.col,
                )
            )

            return g, g

        else:

            raise ValueError(
                f"Unknown graph_mode: {graph_mode}"
            )

    ##################################################
    # omics1
    ##################################################

    g_spatial_omics1, g_feature_omics1 = build_graph(
        adata_omics1
    )

    g_spatial_omics1.ndata["feat"] = th.tensor(
        adata_omics1.obsm["feat"]
    )

    g_feature_omics1.ndata["feat"] = th.tensor(
        adata_omics1.obsm["feat"]
    )

    ##################################################
    # omics2
    ##################################################

    g_spatial_omics2, g_feature_omics2 = build_graph(
        adata_omics2
    )

    g_spatial_omics1 = dgl.to_bidirected(g_spatial_omics1).to(device)
    g_feature_omics1 = dgl.to_bidirected(g_feature_omics1).to(device)

    g_spatial_omics2 = dgl.to_bidirected(g_spatial_omics2).to(device)
    g_feature_omics2 = dgl.to_bidirected(g_feature_omics2).to(device)

    return (
        g_spatial_omics1,
        g_feature_omics1,
        g_spatial_omics2,
        g_feature_omics2,
        adata_omics1,
        adata_omics2,
    )
