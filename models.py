import numpy as np
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch
import torch.nn as nn
from typing import Optional

# ========== Model: Frozen GNN Embeddings + Transformer (prefix) + Classification Head ==========
class AccessSeqModel(nn.Module):
    def __init__(self,
                 user_embedding_matrix: np.ndarray,       # [U, d_gnn]
                 resource_embedding_matrix: np.ndarray,   # [R, d_gnn]
                 d_act: int = 8,
                 d_model: Optional[int] = None,           # Default  = d_gnn
                 nhead: int = 4,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 max_len_pos: int = 1024):
        super().__init__()
        U, d_gnn = user_embedding_matrix.shape
        R, d_gnn2 = resource_embedding_matrix.shape
        assert d_gnn == d_gnn2, "User and resource embedding dimensions must match"
        if d_model is None:
            d_model = d_gnn

        # Frozen graph-side embeddings (indexed by mapped ID)
        self.user_emb = nn.Embedding.from_pretrained(
            torch.tensor(user_embedding_matrix, dtype=torch.float32), freeze=True
        )
        self.resource_emb = nn.Embedding.from_pretrained(
            torch.tensor(resource_embedding_matrix, dtype=torch.float32), freeze=True
        )

        # Action embedding (0/1)
        self.action_emb = nn.Embedding(2, d_act)

        # token = [res_gnn || act_emb] → d_model
        self.token_proj = nn.Linear(d_gnn + d_act, d_model)

        # Learnable CLS token + positional embedding
        self.cls = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_emb = nn.Embedding(max_len_pos + 1, d_model)  # +1 for CLS position

        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                               dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Classification head：concat [user_gnn, seq_emb(CLS), last_resource_gnn]
        self.head = nn.Sequential(
            nn.Linear(d_gnn + d_model + d_gnn, 2 * d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, 1)
        )

    def forward(self,
                user_ids: torch.Tensor,
                prefix_res_ids: torch.Tensor,
                prefix_actions: torch.Tensor,
                attn_mask: torch.Tensor,
                last_resource_ids: torch.Tensor) -> torch.Tensor:
        B, T = prefix_res_ids.shape
        device = user_ids.device

        # Prefix  tokens
        res_tok = self.resource_emb(prefix_res_ids.clamp(min=0))   # pad=-1 → clamp to 0 (will be masked)
        act_tok = self.action_emb(prefix_actions.clamp(min=0))
        tok = self.token_proj(torch.cat([res_tok, act_tok], dim=-1))  # (B,T,d_model)

        # CLS + positional encoding
        cls_tok = self.cls.expand(B, 1, -1)
        tok = torch.cat([cls_tok, tok], dim=1)                        # (B,T+1,d_model)
        pos_ids = torch.arange(0, T + 1, device=device).unsqueeze(0).expand(B, -1)
        tok = tok + self.pos_emb(pos_ids)

        # padding mask: True=padding（note Transformer’s convention）
        key_padding_mask = torch.zeros(B, T + 1, dtype=torch.bool, device=device)
        key_padding_mask[:, 1:] = ~attn_mask

        enc = self.encoder(tok, src_key_padding_mask=key_padding_mask)
        seq_emb = enc[:, 0, :]                                       # CLS embedding

        # Graph-side vectors
        user_vec = self.user_emb(user_ids)
        last_res_vec = self.resource_emb(last_resource_ids)

        # Classification
        feat = torch.cat([user_vec, seq_emb, last_res_vec], dim=-1)
        logits = self.head(feat).squeeze(-1)                         # (B,)
        return logits

class OnlyGraphHead(nn.Module):
    """Use only GNN embeddings: [user || last_res] → MLP"""
    def __init__(self, user_embedding_matrix: np.ndarray,
                 resource_embedding_matrix: np.ndarray,
                 hidden: int = 256, dropout: float = .1):
        super().__init__()
        du = user_embedding_matrix.shape[1]
        dr = resource_embedding_matrix.shape[1]
        self.user_emb = nn.Embedding.from_pretrained(
            torch.tensor(user_embedding_matrix, dtype=torch.float32), freeze=True)
        self.res_emb = nn.Embedding.from_pretrained(
            torch.tensor(resource_embedding_matrix, dtype=torch.float32), freeze=True)
        self.head = nn.Sequential(
            nn.Linear(du + dr, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )

    def forward(self, user_ids, prefix_resources, prefix_actions, attn_mask, last_resource):
        u = self.user_emb(user_ids.long())
        r = self.res_emb(last_resource.long())
        x = torch.cat([u, r], dim=-1)
        logit = self.head(x).squeeze(-1)
        return logit

class OnlyTransformerHead(nn.Module):
    """No graph embeddings: use trainable embeddings for user/resource; CLS as sequence representation"""
    def __init__(self, num_users: int, num_resources: int, d_model: int = 128,
                 nhead: int = 4, num_layers: int = 2, dropout: float = .1, max_len_pos: int = 256):
        super().__init__()
        # Reserve index=0 for PAD
        self.user_emb = nn.Embedding(num_embeddings=num_users + 1, embedding_dim=d_model, padding_idx=0)
        self.res_emb  = nn.Embedding(num_embeddings=num_resources + 1, embedding_dim=d_model, padding_idx=0)
#
# class OnlyTransformerHead(nn.Module):
#     """No graph embeddings: use trainable embeddings for user/resource; CLS as sequence representation"""
#     def __init__(self, num_users: int, num_resources: int, d_model: int = 128,
#                  nhead: int = 4, num_layers: int = 2, dropout: float = .1, max_len_pos: int = 256):
#         super().__init__()
#         # Reserve index=0 for PAD
#         self.user_emb = nn.Embedding(num_embeddings=num_users + 1, embedding_dim=d_model, padding_idx=0)
#         self.res_emb  = nn.Embedding(num_embeddings=num_resources + 1, embedding_dim=d_model, padding_idx=0)
        # Action 0/1; reserve PAD=0, shift real actions +1 → indices 1/2
        d_act = max(8, d_model // 8)
        self.act_emb  = nn.Embedding(num_embeddings=3, embedding_dim=d_act, padding_idx=0)

        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                         dim_feedforward=4*d_model, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_emb = nn.Embedding(max_len_pos + 1, d_model)

        self.proj = nn.Linear(d_model + d_act, d_model)  # project [res_emb || act_emb] back to d_model
        self.head = nn.Sequential(
            nn.Linear(d_model, 2*d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(2*d_model, 1)
        )

    def forward(self, user_ids, prefix_resources, prefix_actions, attn_mask, last_resource):
        B, T = prefix_resources.shape

        # === Handle PAD and valid ranges ===
        uid_raw = user_ids.to(torch.long)
        res_raw = prefix_resources.to(torch.long)
        act_raw = prefix_actions.to(torch.long)
        last_raw = last_resource.to(torch.long)

        # Allow -1 as PAD in collate: clamp negatives to 0, then +1
        uid = torch.clamp(uid_raw, min=0) + 1
        res = torch.clamp(res_raw, min=0) + 1
        act = torch.clamp(act_raw, min=0) + 1  # action 0/1 → 1/2; PAD(-1)→0
        last_res = torch.clamp(last_raw, min=0) + 1

        # === Explicit range checking (raise with detailed values) ===
        def _check(name, x, limit):
            if torch.any(x >= limit):
                bad = x[x >= limit]
                vmax = int(bad.max())
                vmin = int(bad.min())
                count = int(bad.numel())
                raise IndexError(
                    f"[{name}] index out of range: found {count} value(s) in [{vmin}, {vmax}] "
                    f"but num_embeddings={limit} (padding_idx=0; values must be < {limit})."
                )

        _check("user_ids", uid, self.user_emb.num_embeddings)
        _check("prefix_resources", res, self.res_emb.num_embeddings)
        _check("last_resource", last_res, self.res_emb.num_embeddings)
        _check("prefix_actions", act, self.act_emb.num_embeddings)

        # === Embedding  ===
        u = self.user_emb(uid)  # [B, d]
        r = self.res_emb(res)  # [B, T, d]
        a = self.act_emb(act)  # [B, T, d_act]

        tok = self.proj(torch.cat([r, a], dim=-1))  # [B, T, d_model]

        # CLS + positional encoding
        cls_tok = self.cls.expand(B, 1, -1)
        seq = torch.cat([cls_tok, tok], dim=1)  # [B, T+1, d]
        pos = torch.arange(T + 1, device=seq.device).unsqueeze(0)
        seq = seq + self.pos_emb(pos)

        key_padding = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=seq.device),
                                 ~attn_mask], dim=1)
        h = self.encoder(seq, src_key_padding_mask=key_padding)
        h_cls = h[:, 0]
        logit = self.head(h_cls).squeeze(-1)
        return logit
