from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.grid2d_embedding import Grid2DEmbedding
from helper.grid_utils import local_window

# -----------------------------------------------------------------------------
# Utility
# -----------------------------------------------------------------------------

def subsequent_mask(sz: int) -> torch.Tensor:
    """Mask out subsequent positions (for auto‑regressive decoding)."""
    return torch.triu(torch.ones(sz, sz, dtype=torch.bool), diagonal=1)

# -----------------------------------------------------------------------------
# Differentiable top-k by projection to capped simplex (p=2)
# -----------------------------------------------------------------------------
def _find_tau_bisect(x: torch.Tensor, k: torch.Tensor, iters: int = 40):
    """
    x: [B, L]，k: [B] or [B,1]
    回傳 tau: [B, 1]，使 y=clip(x - tau, 0, 1) 的列和 ≈ k
    """
    B, L = x.shape
    k = k.view(B, 1).to(x)
    lo = x.min(dim=1, keepdim=True).values - 1.0
    hi = x.max(dim=1, keepdim=True).values + 1.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        y = (x - mid).clamp_(0.0, 1.0)
        s = y.sum(dim=1, keepdim=True)
        go_right = (s > k)  # sum 太大 → tau 要更大
        lo = torch.where(go_right, mid, lo)
        hi = torch.where(go_right, hi, mid)
    return (lo + hi) / 2

def diff_topk_proj_p2(scores: torch.Tensor, k_vec: torch.Tensor) -> torch.Tensor:
    """
    scores: [B, L]；k_vec: [B]
    回傳 y: [B, L]，0~1 且 Σy≈k（投影到 capped simplex）
    """
    tau = _find_tau_bisect(scores, k_vec)
    y = (scores - tau).clamp(0.0, 1.0)
    return y

# -----------------------------------------------------------------------------
# Main networks
# -----------------------------------------------------------------------------

class Compressor(nn.Module):
    """Encoder that produces *importance logits* for each input point.

    We simply apply a standard TransformerEncoder, then a linear layer to map
    each token representation to a single logit score. A higher score means
    the point is *more important* and will be more likely kept (e.g. via
    Gumbel‑Softmax or top‑k pruning).
    """

    def __init__(self, cfg):
        super().__init__()
        model_cfg = cfg["model"]
        compressor_cfg = model_cfg["compressor"]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_cfg["embedding"]["dim"],
            nhead=compressor_cfg["heads"],
            dim_feedforward=compressor_cfg["ffn_dim"],
            dropout=compressor_cfg["dropout"],
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=compressor_cfg["layers"])
        self.score_proj = nn.Linear(model_cfg["embedding"]["dim"], 1)

    def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Parameters
        ----------
        x : Tensor [B, L, D] – embedded trajectory sequence.
        src_key_padding_mask : BoolTensor [B, L] – True for PAD tokens.
        Returns
        -------
        h      : Tensor [B, L, D] – encoder features per token.
        scores : Tensor [B, L]    – importance logits per point.
        """
        h = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        scores = self.score_proj(h).squeeze(-1)
        return h, scores

class Constructor(nn.Module):
    """Decoder that tries to *reconstruct* the original trajectory given the
    retained points mask. Here we take a simple TransformerDecoder approach.
    """

    def __init__(self, cfg):
        super().__init__()
        model_cfg = cfg["model"]
        constructor_cfg = model_cfg["constructor"]
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=model_cfg["embedding"]["dim"],
            nhead=constructor_cfg["heads"],
            dim_feedforward=constructor_cfg["ffn_dim"],
            dropout=constructor_cfg["dropout"],
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=cfg["model"]["constructor"]["layers"])
        self.output_proj_x = nn.Linear(model_cfg["embedding"]["dim"], cfg["data"]["num_cells_x"])
        self.output_proj_y = nn.Linear(model_cfg["embedding"]["dim"], cfg["data"]["num_cells_y"])
        self.out_emb = nn.Embedding(cfg["data"]["num_cells_x"] *
                                    cfg["data"]["num_cells_y"],
                                    cfg["model"]["embedding"]["dim"])
        self.cfg = cfg

    def forward(
        self,
        tgt_emb: torch.Tensor,
        memory: torch.Tensor,
        grid_x: torch.Tensor,
        grid_y: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns per‑step logits for grid_x and grid_y."""
        h = self.decoder(
            tgt_emb, memory, tgt_mask=tgt_mask, memory_key_padding_mask=memory_key_padding_mask
        )
        logits_x = self.output_proj_x(h)
        logits_y = self.output_proj_y(h)
        return logits_x, logits_y
    
        local_ids, tgt_pos = local_window(
            grid_x, grid_y, win=self.cfg["model"]["local_window"])  # 自己設定 win
        local_emb = self.out_emb(local_ids)                    # [B,L,K,D]
        logits = (h.unsqueeze(-2) * local_emb).sum(-1)         # [B,L,K]
        return logits, tgt_pos

# -----------------------------------------------------------------------------
# Full model
# -----------------------------------------------------------------------------

class TrajSimplificationModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        model_cfg = cfg["model"]
        self.embedding = Grid2DEmbedding(
            num_cells_x=cfg["data"]["num_cells_x"],
            num_cells_y=cfg["data"]["num_cells_y"],
            embedding_dim=model_cfg["embedding"]["dim"],
            dropout=model_cfg["embedding"]["dropout"],
            use_layernorm=model_cfg["embedding"]["layernorm"],
            mode="sum",  # could be changed to "concat" if embedding_dim divisible by 2
        )
        self.compressor = Compressor(cfg)
        self.constructor = Constructor(cfg)
        self.diff_cfg = cfg["model"].get("diff_topk", {"enabled": False})

    # ------------------------------------------------------------------ utils
    def _sample_mask(self, scores: torch.Tensor, keep_ratio: float, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Turn importance logits into a binary mask (True=keep).
        scores   : [B, L]
        pad_mask : [B, L]  True 表 PAD
        """
        valid = scores.masked_fill(pad_mask, -1e9)
        lengths = (~pad_mask).sum(dim=1)                 # [B]
        k = (lengths.float() * keep_ratio).clamp(min=1).long()

        mask = torch.zeros_like(scores, dtype=torch.bool)

        # 逐條處理，避免 stack 尺寸不一
        for b, bk in enumerate(k):
            if bk.item() == 0:
                continue
            topk_idx = valid[b].topk(k=bk.item()).indices   # 長度 = bk
            mask[b, topk_idx] = True
        
        # --- 強制保留序列首尾的「有效」點 ---
        B, L = mask.shape
        if pad_mask is not None:
            valid = ~pad_mask
        else:
            valid = torch.ones_like(mask, dtype=torch.bool)

        first_idx = valid.float().argmax(dim=1)  # 每列第一個 True 的位置
        last_idx  = (L - 1) - torch.flip(valid, [1]).float().argmax(dim=1)

        row = torch.arange(B, device=mask.device)
        mask[row, first_idx] = True
        mask[row, last_idx]  = True
        return mask

    # ------------------------------------------------------------------ forward
    def forward(self, grid_x: torch.Tensor, grid_y: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """End‑to‑end forward pass.

        Parameters
        ----------
        grid_x / grid_y : Tensor [B, L] – discrete grid indices.
        Returns
        -------
        dict with keys: `scores`, `mask`, `logits_x`, `logits_y`, `recon_loss`.
        """
        # 1) embed
        emb = self.embedding(grid_x, grid_y)  # [B, L, D]

        # 2) compressor – 得到 h, scores
        h, scores = self.compressor(emb, src_key_padding_mask=pad_mask)  # [B,L,D], [B,L]

        magnitude = self.diff_cfg.get("magnitude", False)
        s_for_rank = scores.abs() if magnitude else scores
        # 是否走「軟 gating」？
        use_soft = self.diff_cfg.get("enabled", False) and (self.training or not self.diff_cfg.get("hard_inference", False))

        if use_soft:
            # --- (a) 決定每列的 k ---
            lengths = (~pad_mask).sum(dim=1) if pad_mask is not None else torch.full((h.size(0),), h.size(1), device=h.device)
            if self.diff_cfg.get("k_mode", "ratio") == "fixed_k":
                k_vec = torch.full_like(lengths, int(self.diff_cfg.get("fixed_k", 8)))
            else:
                # 沿用 bottleneck.compression_ratio 做預設
                keep_ratio = float(self.diff_cfg.get("keep_ratio", self.cfg["model"]["bottleneck"]["compression_ratio"]))
                k_vec = (lengths.float() * keep_ratio).clamp_min(1.0)

            # --- (b) 排序依據：值 or 幅度 ---
            s_for_rank = scores.abs() if self.diff_cfg.get("magnitude", False) else scores

            # --- (c) 產生可微權重 y（p=2 投影）---
            y = diff_topk_proj_p2(s_for_rank, k_vec)  # [B, L], 0~1, Σ≈k

            # --- (d) 首尾必留（選配，跟你原本一致）---
            if self.diff_cfg.get("pin_edges", True):
                B, L = y.shape
                valid = ~pad_mask if pad_mask is not None else torch.ones_like(y, dtype=torch.bool)
                first_idx = valid.float().argmax(dim=1)
                last_idx  = (L - 1) - torch.flip(valid, [1]).float().argmax(dim=1)
                row = torch.arange(B, device=y.device)
                y[row, first_idx] = 1.0
                y[row, last_idx]  = 1.0

            # --- (e) 用 y 當 gate：整段當 memory（可微）---
            memory_padded = h * y.unsqueeze(-1)   # [B, L, D]
            mem_pad_mask  = pad_mask              # [B, L]，沿用原始 pad

            # 為了和你現有流程相容，我們仍然回傳一個「觀察用」硬 mask（不參與梯度）
            with torch.no_grad():
                bk = k_vec.long().clamp_min(1)
                vis_mask = torch.zeros_like(scores, dtype=torch.bool)
                valid_sc = s_for_rank.masked_fill(pad_mask, float("-inf")) if pad_mask is not None else s_for_rank
                for b in range(scores.size(0)):
                    topk_idx = valid_sc[b].topk(bk[b].item()).indices
                    vis_mask[b, topk_idx] = True
                vis_mask[row, first_idx] = True
                vis_mask[row, last_idx]  = True
            mask = vis_mask

        else:
            # === 硬路徑（沿用你原本的實作）；只在 test 且 hard_inference=true 時會走到這裡 ===
            mask = self._sample_mask(scores, self.cfg["model"]["bottleneck"]["compression_ratio"], pad_mask)  # [B, L]

            B, L, D = h.size()
            retained = [h[b][mask[b]] for b in range(B)]
            max_kept = max(m.size(0) for m in retained) if len(retained) > 0 else 0
            memory_padded = torch.zeros(B, max_kept, D, device=h.device)  # [B, L_kept_max, D]
            mem_pad_mask = torch.ones(B, max_kept, dtype=torch.bool, device=h.device)
            for b, m in enumerate(retained):
                if m.numel() == 0: 
                    continue
                memory_padded[b, : m.size(0)] = m
                mem_pad_mask[b, : m.size(0)] = False  # False = valid token


        # 4) constructor – reconstruct full sequence autoregressively
        # For simplicity we feed the *original* embedding sequence as tgt but shifted.
        tgt_emb = F.pad(emb[:, :-1], (0, 0, 1, 0))  # prepend zero vector (start token)
        tgt_mask = subsequent_mask(L).to(emb.device)
        logits_x, logits_y = self.constructor(
            tgt_emb, memory_padded, grid_x, grid_y, tgt_mask=tgt_mask, memory_key_padding_mask=mem_pad_mask
        )  # each [B, L, num_cells]

        # 5) loss (teacher forcing)
        # if self.cfg.recon_loss == "mse":
        #     recon_loss = F.mse_loss(logits_x.argmax(-1).float(), grid_x.float()) + F.mse_loss(
        #         logits_y.argmax(-1).float(), grid_y.float()
        #     )
        # else:  # l1
        #     recon_loss = F.l1_loss(logits_x.argmax(-1).float(), grid_x.float()) + F.l1_loss(
        #         logits_y.argmax(-1).float(), grid_y.float()
        #     )

        return {
            "scores": scores,
            "mask": mask,
            "logits_x": logits_x,
            "logits_y": logits_y,
        }

# -----------------------------------------------------------------------------
# Quick smoke test
# -----------------------------------------------------------------------------
"""Trajectory Simplification – main model definition
This file wires together the **Grid2DEmbedding**, a *Compressor* (encoder)
and a *Constructor* (decoder) into the end‑to‑end architecture proposed in
"A Lightweight Framework for Fast Trajectory Simplification".

The code is intentionally modular – every block can be swapped out if you
want to experiment with different backbones (e.g. replace the Transformer
encoder with an MLP or a conformer).

Typical usage
-------------
>>> import yaml, torch
>>> from model import TrajSimplificationModel
>>> cfg = yaml.safe_load(open('config.yaml'))
>>> model = TrajSimplificationModel(cfg)
>>> x, y = torch.randint(0, cfg['data']['num_cells_x'], (2, 512)), torch.randint(0, cfg['data']['num_cells_y'], (2, 512))
>>> logits, recon = model(x, y)  # forward pass
"""
if __name__ == "__main__":
    from helper.load_config import load_config
    cfg = load_config("config/config.yaml")
    model = TrajSimplificationModel(cfg)
    batch_size, seq_len = 4, 1024
    gx = torch.randint(0, cfg["data"]["num_cells_x"], (batch_size, seq_len))
    gy = torch.randint(0, cfg["data"]["num_cells_y"], (batch_size, seq_len))
    out = model(gx, gy)
    print("Forward OK, recon loss:", out["recon_loss"].item())
