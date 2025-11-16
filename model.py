# model_no_gnn.py
import torch
import torch.nn as nn
from utils import sparse_dropout, spmm   # теперь используется только для SVD
import torch.nn.functional as F

class AttenMixer(nn.Module):
    """
    Внимательное смешивание L+1 эмбеддингов.
    Вход: list[Tensor] длины L+1, shape (B, d)
    Выход: (B, d)
    """
    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        self.query = nn.Parameter(torch.empty(num_layers, hidden_dim))
        nn.init.xavier_uniform_(self.query)

    def forward(self, emb_list):
        stack = torch.stack(emb_list, dim=1)          # (B, L+1, d)
        score = torch.einsum("bld,ld->bl", stack, self.query)
        alpha = torch.softmax(score, dim=1)
        out   = torch.einsum("bld,bl->bd", stack, alpha)
        return out


class LightGCL(nn.Module):
    def __init__(self, n_u, n_i, d, u_mul_s, v_mul_s, ut, vt, train_csr, adj_norm,
                 l, temp, lambda_1, lambda_2, dropout, batch_user, device):
        super().__init__()
        # начальные эмбеддинги
        self.E_u_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_u, d)))
        self.E_i_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_i, d)))

        self.train_csr = train_csr
        self.l = l
        self.temp = temp
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.dropout = dropout
        self.device = device
        self.batch_user = batch_user

        # SVD-матрицы
        self.u_mul_s = u_mul_s
        self.v_mul_s = v_mul_s
        self.ut = ut
        self.vt = vt

        # attention-микшеры
        self.atten_u = AttenMixer(d, l + 1)
        self.atten_i = AttenMixer(d, l + 1)

        # списки для слоёв
        self.E_u_list = [None] * (l + 1)
        self.E_i_list = [None] * (l + 1)
        self.G_u_list = [None] * (l + 1)
        self.G_i_list = [None] * (l + 1)
        self.E_u_list[0] = self.E_u_0
        self.E_i_list[0] = self.E_i_0
        self.G_u_list[0] = self.E_u_0
        self.G_i_list[0] = self.E_i_0

        self.E_u = None
        self.E_i = None

    def forward(self, uids, iids, pos, neg, test=False):
        if test:
            preds = self.E_u[uids] @ self.E_i.T
            mask = self.train_csr[uids.cpu().numpy()].toarray()
            mask = torch.Tensor(mask).to(self.device)
            preds = preds * (1 - mask) - 1e8 * mask
            return preds.argsort(descending=True)

        # --------- ONLY SVD-propagation --------- #
        for layer in range(1, self.l + 1):
            # SVD-ветка
            vt_ei = self.vt @ self.E_i_list[layer - 1]
            self.G_u_list[layer] = self.u_mul_s @ vt_ei
            ut_eu = self.ut @ self.E_u_list[layer - 1]
            self.G_i_list[layer] = self.v_mul_s @ ut_eu

            # **больше нет GNN-пропагации**
            self.E_u_list[layer] = self.G_u_list[layer]
            self.E_i_list[layer] = self.G_i_list[layer]

        self.G_u = sum(self.G_u_list)
        self.G_i = sum(self.G_i_list)

        # итоговые эмбеддинги – attention по слоям
        self.E_u = self.atten_u(self.E_u_list)
        self.E_i = self.atten_i(self.E_i_list)

        # contrastive loss
        G_u_norm = self.G_u
        E_u_norm = self.E_u
        G_i_norm = self.G_i
        E_i_norm = self.E_i

        neg_score = torch.log(torch.exp(G_u_norm[uids] @ E_u_norm.T / self.temp).sum(1) + 1e-8).mean()
        neg_score += torch.log(torch.exp(G_i_norm[iids] @ E_i_norm.T / self.temp).sum(1) + 1e-8).mean()
        pos_score = (torch.clamp((G_u_norm[uids] * E_u_norm[uids]).sum(1) / self.temp, -5, 5)).mean() + \
                    (torch.clamp((G_i_norm[iids] * E_i_norm[iids]).sum(1) / self.temp, -5, 5)).mean()
        loss_s = -pos_score + neg_score

        # BPR-loss
        u_emb = self.E_u[uids]
        pos_emb = self.E_i[pos]
        neg_emb = self.E_i[neg]
        pos_scores = (u_emb * pos_emb).sum(-1)
        neg_scores = (u_emb * neg_emb).sum(-1)
        loss_r = -(pos_scores - neg_scores).sigmoid().log().mean()

        # L2-reg
        loss_reg = 0
        for param in self.parameters():
            loss_reg += param.norm(2).square()
        loss_reg *= self.lambda_2

        loss = loss_r + self.lambda_1 * loss_s + loss_reg
        return loss, loss_r, self.lambda_1 * loss_s