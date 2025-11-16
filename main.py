# main_no_gnn.py
import os, re, warnings, time
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from scipy.sparse import coo_matrix
from tqdm import tqdm
from parser import args
from model_no_gnn import LightGCL
from utils import TrnData, precision, mrr, scipy_sparse_mat_to_torch_sparse_tensor

warnings.filterwarnings("ignore", category=DeprecationWarning)
device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

# ---------------- 1. гипер-параметры ---------------- #
d, l, temp = args.d, args.gnn_layer, args.temp
batch_u, inter_b = args.batch, args.inter_batch
epoch_no, lr = args.epoch, args.lr
lambda_1, lambda_2, dropout = args.lambda1, args.lambda2, args.dropout
svd_q = args.q

# ---------------- 2. утилиты ---------------- #
def load_sessions(path):
    for enc in ("utf-8", "utf-16", "latin1"):
        try:
            with open(path, encoding=enc) as f:
                return [[int(x) for x in re.findall(r"\d+", line)] for line in f if line.strip()]
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"cannot read {path}")


def build_coo(sessions):
    if not sessions:
        return coo_matrix((0, 0), dtype=np.float32)
    rows, cols, vals = [], [], []
    for seq in sessions:
        uid = seq[0]
        for iid in seq[1:]:
            rows.append(uid)
            cols.append(iid)
            vals.append(1.0)
    return coo_matrix(
        (vals, (rows, cols)),
        shape=(max(rows) + 1, max(cols) + 1),
        dtype=np.float32,
    )


def remap_sessions(session_list):
    user_set = {s[0] for s in session_list}
    item_set = set()
    for s in session_list:
        item_set.update(s[1:])
    umap = {u: idx for idx, u in enumerate(sorted(user_set))}
    imap = {i: idx for idx, i in enumerate(sorted(item_set))}
    return [[umap[s[0]]] + [imap[i] for i in s[1:]] for s in session_list], umap, imap


def clean_short(sessions, min_len=3):
    return [s for s in sessions if len(s) >= min_len]


# ---------------- 3. читаем данные ---------------- #
base = "/home/lisa/LightGCL_am-gnn/data"
ds_path = os.path.join(base, args.data)
train_seq = load_sessions(os.path.join(ds_path, "train.txt"))
test_seq = load_sessions(os.path.join(ds_path, "test.txt"))
print(f"loaded {len(train_seq)} train / {len(test_seq)} test sessions")

# remap
train_seq, umap, imap = remap_sessions(train_seq)
test_seq = [[umap.get(s[0], -1)] + [imap[i] for i in s[1:] if i in imap] for s in test_seq]
test_seq = [s for s in test_seq if s[0] >= 0 and len(s) >= 3]

# ---------------- 4. строим матрицы ---------------- #
train_mat = build_coo(train_seq)
test_mat = build_coo(test_seq)
print("train shape", train_mat.shape, " nnz=", train_mat.nnz)
print("test  shape", test_mat.shape, " nnz=", test_mat.nnz)

# ---------------- 5. нормализация adj (нужна только для SVD) ---------------- #
train_coo = train_mat.tocoo()
row_sum = np.array(train_coo.sum(1)).squeeze()
col_sum = np.array(train_coo.sum(0)).squeeze()
for i in range(len(train_coo.data)):
    train_coo.data[i] /= np.sqrt(row_sum[train_coo.row[i]] * col_sum[train_coo.col[i]])

adj_norm = scipy_sparse_mat_to_torch_sparse_tensor(train_coo).coalesce().to(device)
train_csr = (train_mat != 0).astype(np.float32)

n_u, n_i = adj_norm.shape          # теперь n_u известен

# ---------------- 6. тест-лейблы (для всех train-пользователей) ---------------- #
test_labels = [[] for _ in range(n_u)]
for r, c in zip(test_mat.row, test_mat.col):
    if r < n_u:
        test_labels[r].append(c)

# ---------------- 7. SVD ---------------- #
svd_u, s, svd_v = torch.svd_lowrank(adj_norm, q=svd_q)
u_mul_s = svd_u @ torch.diag(s)
v_mul_s = svd_v @ torch.diag(s)
del s

# ---------------- 8. модель / оптимизатор ---------------- #
model = LightGCL(
    n_u, n_i, d,
    u_mul_s, v_mul_s,
    svd_u.T, svd_v.T,
    train_csr, adj_norm,
    l, temp, lambda_1, lambda_2,
    dropout, batch_u, device,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5)

# ---------------- 9. тренировка ---------------- #
best_p5 = best_mrr5 = best_p10 = best_mrr10 = best_p20 = best_mrr20 = 0.0
best_ep = defaultdict(int)

train_loader = data.DataLoader(
    TrnData(train_mat), batch_size=inter_b, shuffle=True, num_workers=4
)

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

    # ---------------- тест ---------------- #
    model.eval()
    test_uids = np.arange(n_u)
    batches = (len(test_uids) + batch_u - 1) // batch_u

    p5 = mrr5 = p10 = mrr10 = p20 = mrr20 = 0.0
    with torch.no_grad():
        for b in range(batches):
            s, e = b * batch_u, min((b + 1) * batch_u, len(test_uids))
            uu = torch.LongTensor(test_uids[s:e]).to(device)
            preds = model(uu, None, None, None, test=True).cpu().numpy()

            p5   += precision(test_uids[s:e], preds, 5,  test_labels)
            mrr5 += mrr(test_uids[s:e], preds, 5,  test_labels)
            p10  += precision(test_uids[s:e], preds, 10, test_labels)
            mrr10+= mrr(test_uids[s:e], preds, 10, test_labels)
            p20  += precision(test_uids[s:e], preds, 20, test_labels)
            mrr20+= mrr(test_uids[s:e], preds, 20, test_labels)

    p5/=batches; mrr5/=batches; p10/=batches; mrr10/=batches; p20/=batches; mrr20/=batches
    print(f"TEST  P@5={p5:.4f} MRR@5={mrr5:.4f} | P@10={p10:.4f} MRR@10={mrr10:.4f} | P@20={p20:.4f} MRR@20={mrr20:.4f}")

    # best
    if p5 > best_p5: best_p5, best_ep["p5"] = p5, epoch
    if mrr5 > best_mrr5: best_mrr5, best_ep["mrr5"] = mrr5, epoch
    if p10 > best_p10: best_p10, best_ep["p10"] = p10, epoch
    if mrr10 > best_mrr10: best_mrr10, best_ep["mrr10"] = mrr10, epoch
    if p20 > best_p20: best_p20, best_ep["p20"] = p20, epoch
    if mrr20 > best_mrr20: best_mrr20, best_ep["mrr20"] = mrr20, epoch

print("\n======== BEST ========")
print(f"P@5  = {best_p5:.4f}  (ep {best_ep['p5']})   MRR@5  = {best_mrr5:.4f}  (ep {best_ep['mrr5']})")
print(f"P@10 = {best_p10:.4f}  (ep {best_ep['p10']})  MRR@10 = {best_mrr10:.4f}  (ep {best_ep['mrr10']})")
print(f"P@20 = {best_p20:.4f}  (ep {best_ep['p20']})  MRR@20 = {best_mrr20:.4f}  (ep {best_ep['mrr20']})")