import dgl
import torch as th
import numpy as np
from scipy.sparse import coo_matrix

from SpaMIE.preprocess import preprocessing


def edge_list_to_sparse(adj_dict, n):
    """
    Convert SpaMIE edge list stored in
    adata.uns["adj_spatial"]
    into a scipy sparse matrix.
    """

    rows = np.asarray(adj_dict["x"])
    cols = np.asarray(adj_dict["y"])

    data = np.ones(len(rows), dtype=np.float32)

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
    graph_mode="attention",
):

    ###########################################################
    # Load data
    ###########################################################

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

    ###########################################################
    # Graph builder
    ###########################################################

    def build_graph(adata):

        n = adata.n_obs

        spatial = edge_list_to_sparse(
            adata.uns["adj_spatial"],
            n,
        ).tocsr()

        feature = adata.obsm["adj_feature"].tocsr()

        #######################################################
        # Original SpaMIE
        #######################################################

        if graph_mode == "attention":

            g_spatial = dgl.graph(
                (
                    adata.uns["adj_spatial"]["x"],
                    adata.uns["adj_spatial"]["y"],
                ),
                num_nodes=n,
            )

            feature = feature.tocoo()

            g_feature = dgl.graph(
                (
                    feature.row,
                    feature.col,
                ),
                num_nodes=n,
            )

            print(f"[Attention] Spatial edges : {g_spatial.number_of_edges()}")
            print(f"[Attention] Feature edges : {g_feature.number_of_edges()}")

            return g_spatial, g_feature

        #######################################################
        # Spatial only
        #######################################################

        elif graph_mode == "spatial":

            g_spatial = dgl.graph(
                (
                    adata.uns["adj_spatial"]["x"],
                    adata.uns["adj_spatial"]["y"],
                ),
                num_nodes=n,
            )

            print(f"[Spatial] Edges : {g_spatial.number_of_edges()}")

            return g_spatial, g_spatial

        #######################################################
        # Feature only
        #######################################################

        elif graph_mode == "feature":

            feature = feature.tocoo()

            g_feature = dgl.graph(
                (
                    feature.row,
                    feature.col,
                ),
                num_nodes=n,
            )

            print(f"[Feature] Edges : {g_feature.number_of_edges()}")

            return g_feature, g_feature

        #######################################################
        # Union
        #######################################################

        elif graph_mode == "union":

            union = spatial.maximum(feature)
            union = union.tocoo()

            g_union = dgl.graph(
                (
                    union.row,
                    union.col,
                ),
                num_nodes=n,
            )

            print(f"[Union] Edges : {g_union.number_of_edges()}")

            return g_union, g_union

        #######################################################
        # Intersection
        #######################################################

        elif graph_mode == "intersection":

            inter = spatial.multiply(feature)
            inter.data[:] = 1
            inter = inter.tocoo()

            g_inter = dgl.graph(
                (
                    inter.row,
                    inter.col,
                ),
                num_nodes=n,
            )

            print(f"[Intersection] Edges : {g_inter.number_of_edges()}")

            return g_inter, g_inter

        #######################################################

        else:

            raise ValueError(
                f"Unknown graph_mode: {graph_mode}"
            )

    ###########################################################
    # Omics 1
    ###########################################################

    g_spatial_omics1, g_feature_omics1 = build_graph(
        adata_omics1
    )

    feat1 = th.tensor(
        adata_omics1.obsm["feat"],
        dtype=th.float32,
    )

    g_spatial_omics1.ndata["feat"] = feat1
    g_feature_omics1.ndata["feat"] = feat1

    ###########################################################
    # Omics 2
    ###########################################################

    g_spatial_omics2, g_feature_omics2 = build_graph(
        adata_omics2
    )

    feat2 = th.tensor(
        adata_omics2.obsm["feat"],
        dtype=th.float32,
    )

    g_spatial_omics2.ndata["feat"] = feat2
    g_feature_omics2.ndata["feat"] = feat2

    ###########################################################
    # Move to GPU
    ###########################################################

    g_spatial_omics1 = dgl.to_bidirected(g_spatial_omics1).to(device)
    g_feature_omics1 = dgl.to_bidirected(g_feature_omics1).to(device)

    g_spatial_omics2 = dgl.to_bidirected(g_spatial_omics2).to(device)
    g_feature_omics2 = dgl.to_bidirected(g_feature_omics2).to(device)

    ###########################################################
    # Return
    ###########################################################

    return (
        g_spatial_omics1,
        g_feature_omics1,
        g_spatial_omics2,
        g_feature_omics2,
        adata_omics1,
        adata_omics2,
    )
