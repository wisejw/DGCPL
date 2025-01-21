import numpy as np
import scipy.sparse as sp
import torch
import pandas as pd


def generate_adj_matrices(data_file, num_nodes):
    adj_out = np.zeros((num_nodes, num_nodes))
    adj_in = np.zeros((num_nodes, num_nodes))

    df = pd.read_csv(data_file, header=0)
    head_ids = df['C1(head)_id']
    tail_ids = df['C2(tail)_id']
    Nums = df['Num']

    for head, tail, Num in zip(head_ids, tail_ids, Nums):
        adj_out[head - 1][tail - 1] += Num
        adj_in[tail - 1][head - 1] += Num

    adj_out = normalize(adj_out + sp.eye(num_nodes))
    adj_in = normalize(adj_in + sp.eye(num_nodes))

    adj_out = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(adj_out))
    adj_in = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(adj_in))

    return adj_out, adj_in


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
