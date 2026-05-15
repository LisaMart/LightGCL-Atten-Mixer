#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main training and evaluation script for LightGCL-no-GNN.
Pipeline:
1. Load sessions, remap IDs
2. Build COO interaction matrices
3. Normalise matrices for SVD
4. Compute SVD factors
5. Train LightGCL with BPR + contrastive loss
6. Evaluate Precision@K and MRR@K
"""

import os
import re
import time
import warnings
from collections import defaultdict
from typing import List

import numpy as np
import torch
import torch.utils.data as data
from scipy.sparse import coo_matrix
from tqdm import tqdm

from model(2) import LightGCL  # GNN layers removed
from parser import args
from utils(1) import TrnData, precision, mrr, scipy_sparse_mat_to_torch_sparse_tensor

warnings.filterwarnings("ignore", category=DeprecationWarning)

# -------------------- device setup --------------------
device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

# -------------------- hyper-parameters --------------------
d, l, temp = args.d, args.gnn_layer, args.temp  # l kept for compatibility
batch_u, inter_b = args.batch, args.inter_batch
epoch_no, lr = args.epoch, args.lr
lambda_1, lambda_2, dropout = args.lambda1, args.lambda2, args.dropout
svd_q = args.q

# -------------------- universal dataset path --------------------
base_path = os.path.join(os.path.dirname(__file__), 'datasets')
ds_path = os.path.join(base_path, args.data)

# -------------------- 1. Load sessions --------------------
def load_sessions(path: str) -> List[List[int]]:
    """Read session file into list of integer sequences."""
    for enc in ("utf-8", "utf-16", "latin1"):
        try:
            with open(path, encoding=enc) as f:
                return [[int(x) for x in re.findall(r"\d+", line)] for line in f if line.strip()]
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"cannot read {path}")

# -------------------- 2. Build COO interaction matrix --------------------
def build_coo(sessions: List[List[int]]) -> coo_matrix:
    """Convert session list into user-item COO sparse matrix."""
    if not sessions:
        return coo_matrix((0, 0), dtype=np.float32)
    rows, cols, vals = [], [], []
    for seq in sessions:
        uid = seq[0]
        for iid in seq[1:]:
            rows.append(uid)
            cols.append(iid)
            vals.append(1.0)
    return coo_matrix((vals, (rows, cols)), shape=(max(rows) + 1, max(cols) + 1), dtype=np.float32)

# -------------------- 3. Remap IDs --------------------
def remap_sessions(session_list: List[List[int]]):
    """Map raw user/item IDs to contiguous indices."""
    user_set = {s[0] for s in session_list}
    item_set = set()
    for s in session_list:
        item_set.update(s[1:])
    umap = {u: idx for idx, u in enumerate(sorted(user_set))}
    imap = {i: idx for idx, i in enumerate(sorted(item_set))}
    return [[umap[s[0]]] + [imap[i] for i in s[1:]] for s in session_list], umap, imap

# -------------------- 4. Load & preprocess --------------------
train_seq = load_sessions(os.path.join(ds_path, "train.txt"))
test_seq  = load_sessions(os.path.join(ds_path, "test.txt"))
print(f"loaded {len(train_seq)} train / {len(test_seq)} test sessions")

train_seq, umap, imap = remap_sessions(train_seq)
test_seq = [[umap.get(s[0], -1)] + [imap[i] for i in s[1:] if i in imap] for s in test_seq]
test_seq = [s for s in test_seq if s[0] >= 0 and len(s) >= 3]

# -------------------- 5. Interaction matrices --------------------
train_mat = build_coo(train_seq)
test_mat  = build_coo(test_seq)
print("train shape", train_mat.shape, " nnz=", train_mat.nnz)
print("test  shape", test_mat.shape, " nnz=", test_mat.nnz)

# -------------------- 6. Symmetric normalisation --------------------
train_coo = train_mat.tocoo()
row_sum = np.array(train_coo.sum(1)).squeeze()
col_sum = np.array(train_coo.sum(0)).squeeze()
for i in range(len(train_coo.data)):
    train_coo.data[i] /= np.sqrt(row_sum[train_coo.row[i]] * col_sum[train_coo.col[i]])
adj_norm = scipy_sparse_mat_to_torch_sparse_tensor(train_coo).coalesce().to(device)
train_csr = (train_mat != 0).astype(np.float32)

n_u, n_i = adj_norm.shape

# -------------------- 7. Test labels --------------------
test_labels: List[List[int]] = [[] for _ in range(n_u)]
for r, c in zip(test_mat.row, test_mat.col):
    if r < n_u:
        test_labels[r].append(c)

# -------------------- 8. SVD factors --------------------
svd_u, s, svd_v = torch.svd_lowrank(adj_norm, q=svd_q)
u_mul_s = svd_u @ torch.diag(s)
v_mul_s = svd_v @ torch.diag(s)
del s

# -------------------- 9. Model & optimizer --------------------
model = LightGCL(n_u, n_i, d, u_mul_s, v_mul_s, svd_u.T, svd_v.T, train_csr, adj_norm,
                  l, temp, lambda_1, lambda_2, dropout, batch_u, device).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5)

# -------------------- 10. Training loop --------------------
best_p5 = best_mrr5 = best_p10 = best_mrr10 = best_p20 = best_mrr20 = 0.0
best_ep: dict[str, int] = defaultdict(int)

train_loader = data.DataLoader(TrnData(train_mat), batch_size=inter_b, shuffle=True, num_workers=4)

for epoch in range(epoch_no):
    train_loader.dataset.neg_sampling()
    model.train()
    tot_loss = 0.0

    for uids, pos, neg in tqdm(train_loader, desc=f"epoch {epoch}"):
        uids = uids.long().to(device)
        pos, neg = pos.long().to(device), neg.long().to(device)
        optimizer.zero_grad()
        loss, loss_r, loss_s = model(uids, torch.cat([pos, neg]), pos, neg)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()

    scheduler.step()
    avg_loss = tot_loss / len(train_loader)
    print(f"epoch {epoch:03d} | loss = {avg_loss:.4f}")

    # -------------------- 11. Evaluation --------------------
    model.eval()
    test_uids = np.arange(n_u)
    batches = (len(test_uids) + batch_u - 1) // batch_u

    p5 = mrr5 = p10 = mrr10 = p20 = mrr20 = 0.0
    with torch.no_grad():
        for b in range(batches):
            s, e = b * batch_u, min((b + 1) * batch_u, len(test_uids))
            uu = torch.LongTensor(test_uids[s:e]).to(device)
            preds = model(uu, None, None, None, test=True).cpu().numpy()

            p5 += precision(test_uids[s:e], preds, 5, test_labels)
            mrr5 += mrr(test_uids[s:e], preds, 5, test_labels)
            p10 += precision(test_uids[s:e], preds, 10, test_labels)
            mrr10 += mrr(test_uids[s:e], preds, 10, test_labels)
            p20 += precision(test_uids[s:e], preds, 20, test_labels)
            mrr20 += mrr(test_uids[s:e], preds, 20, test_labels)

    # normalize by number of batches
    p5 /= batches
    mrr5 /= batches
    p10 /= batches
    mrr10 /= batches
    p20 /= batches
    mrr20 /= batches
