import numpy as np
import torch
import scipy.sparse as sp
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ["OMP_NUM_THREADS"] = "1"

def generate_G_from_H(H, variable_weight=False):
    H = np.array(H)
    n_edge = H.shape[1]  # Number of columns of matrix = number of hyperedge
    W = np.ones(n_edge)    # Initialize weight vector

    DV = np.sum(H * W, axis=1)    # Degree of vertices
    DE = np.sum(H, axis=0)    # Degree of edges
    invDE = np.mat(np.diag(np.power(DE, float(-1))))    # Inverse degree of edges
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))    # Degree normalization for vertices

    W = np.mat(np.diag(W))    # Weighted adjacency
    H = np.mat(H)
    HT = H.T    # Transpose of H

    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        G = DV2 * H * W * invDE * HT * DV2    # Graph generation
        G = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(G))    # Convert to sparse tensor
        return G


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)    # Convert to COO format
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))    # Indices
    values = torch.from_numpy(sparse_mx.data)    # Values
    shape = torch.Size(sparse_mx.shape)    # Matrix shape
    return torch.sparse.FloatTensor(indices, values, shape)    # Return sparse tensor

