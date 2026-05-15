import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import sparse_dropout, spmm  # imported but not used in GNN-free variant

# ------------------------------------------------------------------
# 1. Layer-wise Attention Mixer
# ------------------------------------------------------------------
class AttenMixer(nn.Module):
    """
    Performs weighted aggregation of multiple layer embeddings.
    Each layer l gets a learnable scalar weight alpha_l.
    Inputs: list of tensors [(B, d)]
    Outputs: tensor (B, d) - weighted sum
    """
    def __init__(self, hidden_dim: int, num_layers: int):
        super().__init__()
        # learnable query vector for each layer
        self.query = nn.Parameter(torch.empty(num_layers, hidden_dim))
        nn.init.xavier_uniform_(self.query)

    def forward(self, emb_list: list[torch.Tensor]) -> torch.Tensor:
        # stack layers into shape (B, L+1, d)
        stack = torch.stack(emb_list, dim=1)
        # compute raw attention scores per layer (B, L+1)
        score = torch.einsum("bld,ld->bl", stack, self.query)
        # softmax to get attention weights
        alpha = torch.softmax(score, dim=1)
        # weighted sum across layers to obtain final embedding
        out = torch.einsum("bld,bl->bd", stack, alpha)
        return out

# ------------------------------------------------------------------
# 2. LightGCL model - GNN-free version
# ------------------------------------------------------------------
class LightGCL(nn.Module):
    """
    Implements LightGCL without GNN propagation.
    Uses SVD views + layer-wise attention mixer.
    Computes Info-NCE contrastive loss + BPR pairwise ranking + L2 regularisation.
    """
    def __init__(self, n_u, n_i, d, u_mul_s, v_mul_s, ut, vt, train_csr, adj_norm,
                 l, temp, lambda_1, lambda_2, dropout, batch_user, device):
        super().__init__()

        # trainable base embeddings
        self.E_u_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_u, d)))
        self.E_i_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_i, d)))

        # store training mask
        self.train_csr = train_csr
        self.l = l  # number of SVD layers
        self.temp = temp
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.device = device
        self.batch_user = batch_user

        # SVD matrices (fixed, not trainable)
        self.register_buffer("u_mul_s", u_mul_s)
        self.register_buffer("v_mul_s", v_mul_s)
        self.register_buffer("ut", ut)
        self.register_buffer("vt", vt)

        # attention mixers for layer aggregation
        self.atten_u = AttenMixer(d, l + 1)
        self.atten_i = AttenMixer(d, l + 1)

        # lists to store per-layer embeddings for users/items
        self.E_u_list = [None] * (l + 1)
        self.E_i_list = [None] * (l + 1)
        self.G_u_list = [None] * (l + 1)
        self.G_i_list = [None] * (l + 1)

        # initialise layer 0 embeddings
        self.E_u_list[0] = self.E_u_0
        self.E_i_list[0] = self.E_i_0
        self.G_u_list[0] = self.E_u_0
        self.G_i_list[0] = self.E_i_0

        # final embeddings (updated each forward)
        self.E_u = None
        self.E_i = None

    # ------------------------------------------------------------------
    # 3. Forward pass
    # ------------------------------------------------------------------
    def forward(self, uids, iids, pos, neg, test=False):
        # ---------- TEST: return ranked lists ----------
        if test:
            scores = self.E_u[uids] @ self.E_i.T
            mask = torch.tensor(
                self.train_csr[uids.cpu().numpy()].toarray(), device=self.device, dtype=torch.float32
            )
            scores = scores * (1 - mask) - 1e8 * mask  # mask seen items
            return scores.argsort(dim=1, descending=True)

        # ---------- TRAIN: SVD-only propagation ----------
        for layer in range(1, self.l + 1):
            # user embeddings from item SVD view
            vt_ei = self.vt @ self.E_i_list[layer - 1]
            self.G_u_list[layer] = self.u_mul_s @ vt_ei

            # item embeddings from user SVD view
            ut_eu = self.ut @ self.E_u_list[layer - 1]
            self.G_i_list[layer] = self.v_mul_s @ ut_eu

            # assign embeddings (no GNN)
            self.E_u_list[layer] = self.G_u_list[layer]
            self.E_i_list[layer] = self.G_i_list[layer]

        # aggregate across layers
        self.G_u = sum(self.G_u_list)
        self.G_i = sum(self.G_i_list)

        # attention-weighted aggregation
        self.E_u = self.atten_u(self.E_u_list)
        self.E_i = self.atten_i(self.E_i_list)

        # ---------- 4. Contrastive loss (Info-NCE) ----------
        G_u_norm, E_u_norm = self.G_u, self.E_u
        G_i_norm, E_i_norm = self.G_i, self.E_i

        neg_score = torch.log(torch.exp(G_u_norm[uids] @ E_u_norm.T / self.temp).sum(1) + 1e-8).mean()
        neg_score += torch.log(torch.exp(G_i_norm[iids] @ E_i_norm.T / self.temp).sum(1) + 1e-8).mean()

        pos_score = (torch.clamp((G_u_norm[uids] * E_u_norm[uids]).sum(1) / self.temp, -5, 5)).mean()
        pos_score += (torch.clamp((G_i_norm[iids] * E_i_norm[iids]).sum(1) / self.temp, -5, 5)).mean()

        loss_s = -pos_score + neg_score

        # ---------- 5. BPR loss ----------
        u_emb = self.E_u[uids]
        pos_emb = self.E_i[pos]
        neg_emb = self.E_i[neg]

        pos_scores = (u_emb * pos_emb).sum(-1)
        neg_scores = (u_emb * neg_emb).sum(-1)
        loss_r = -(pos_scores - neg_scores).sigmoid().log().mean()

        # ---------- 6. L2 regularisation ----------
        loss_reg = self.lambda_2 * sum(p.norm(2).square() for p in self.parameters())

        # ---------- 7. Total loss ----------
        loss = loss_r + self.lambda_1 * loss_s + loss_reg
        return loss, loss_r, self.lambda_1 * loss_s
