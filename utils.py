#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility helpers for LightGCL-no-GNN.
Provides dataset handling, evaluation metrics, and sparse tensor helpers.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from scipy.sparse import coo_matrix

# -------------------- 1. Convert COO to torch.sparse --------------------
def scipy_sparse_mat_to_torch_sparse_tensor(sparse_mx: coo_matrix) -> torch.Tensor:
    """Convert a scipy COO matrix to torch.sparse_coo_tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)

# -------------------- 2. Sparse dropout --------------------
def sparse_dropout(mat: torch.Tensor, dropout: float) -> torch.Tensor:
    """Dropout only applied to non-zero values of a sparse tensor."""
    if dropout == 0.0:
        return mat
    indices = mat.indices()
    values = nn.functional.dropout(mat.values(), p=dropout)
    return torch.sparse_coo_tensor(indices, values, mat.size(), dtype=mat.dtype)

# -------------------- 3. Sparse-dense multiplication --------------------
def spmm(sp: torch.Tensor, emb: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Multiply sparse adjacency (COO) by dense embeddings (segment sum)."""
    sp = sp.coalesce()
    cols = sp.indices()[1]
    rows = sp.indices()[0]
    col_segs = emb[cols]
    result = torch.zeros((sp.shape[0], emb.shape[1]), device=device, dtype=emb.dtype)
    result.index_add_(0, rows, col_segs)
    return result

# -------------------- 4. Training dataset with on-the-fly negative sampling --------------------
class TrnData(data.Dataset):
    """Dataset for user-item interactions, samples negative items on-the-fly."""
    def __init__(self, coomat: coo_matrix):
        self.rows = coomat.row
        self.cols = coomat.col
        self.dokmat = coomat.todok()
        self.negs = np.zeros(len(self.rows), dtype=np.int32)

    def neg_sampling(self) -> None:
        """Sample a negative item for every positive interaction."""
        for i in range(len(self.rows)):
            u = self.rows[i]
            while True:
                neg = np.random.randint(self.dokmat.shape[1])
                if (u, neg) not in self.dokmat:
                    break
            self.negs[i] = neg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx], self.cols[idx], self.negs[idx]

# -------------------- 5. Precision@K --------------------
def precision(uids: np.ndarray, predictions: np.ndarray, topk: int, test_labels: list) -> float:
    hits, total = 0, 0
    for i, uid in enumerate(uids):
        pred = predictions[i][:topk].tolist()
        label = test_labels[uid]
        if len(label) > 0:
            hits += len(set(pred) & set(label))
            total += topk
    return hits / total if total > 0 else 0.0

# -------------------- 6. Mean Reciprocal Rank@K --------------------
def mrr(uids: np.ndarray, predictions: np.ndarray, topk: int, test_labels: list) -> float:
    mrr_sum, user_num = 0.0, 0
    for i, uid in enumerate(uids):
        pred = predictions[i][:topk].tolist()
        label = test_labels[uid]
        if len(label) == 0:
            continue
        for rank, item in enumerate(pred, 1):
            if item in label:
                mrr_sum += 1.0 / rank
                break
        user_num += 1
    return mrr_sum / user_num if user_num > 0 else 0.0
