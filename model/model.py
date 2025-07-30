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

    def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Parameters
        ----------
        x : Tensor [B, L, D] – embedded trajectory sequence.
        src_key_padding_mask : BoolTensor [B, L] – True for PAD tokens.
        Returns
        -------
        scores : Tensor [B, L] – importance logits per point.
        """
        h = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        scores = self.score_proj(h).squeeze(-1)
        return scores

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
        # logits_x = self.output_proj_x(h)
        # logits_y = self.output_proj_y(h)
        # return logits_x, logits_y
    
        local_ids, tgt_pos = local_window(
            grid_x, grid_y, win=cfg["model"]["local_window"])  # 自己設定 win
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

    # ------------------------------------------------------------------ utils
    def _sample_mask(self, scores: torch.Tensor, keep_ratio: float, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Turn importance logits into a binary mask (True=keep).
        scores   : [B, L]
        pad_mask : [B, L]  True 表 PAD
        """
        # ① 給 PAD −∞，永遠不被選
        valid_scores = scores.masked_fill(pad_mask, -1e9)

        # ② 每個 batch 用「實際有效點數」算 k
        lengths = (~pad_mask).sum(dim=1)                  # [B]
        k = (lengths.float() * keep_ratio).clamp(min=1).long()

        topk_idx = torch.stack([
            valid_scores[b].topk(k=bk.item()).indices if bk.item() > 0 else torch.tensor([], dtype=torch.long, device=scores.device)
            for b, bk in enumerate(k)
        ]) #若 bk.item() == 0 會報錯，因此應加保護

        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask.scatter_(1, topk_idx, True)
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

        # 2) compressor – importance scores
        scores = self.compressor(emb,src_key_padding_mask = pad_mask)  # [B, L]
        mask = self._sample_mask(scores, self.cfg["model"]["bottleneck"]["compression_ratio"],pad_mask)  # [B, L]

        # 3) gather retained embeddings as memory for decoder
        # We keep original ordering
        B, L, D = emb.size()
        memory = torch.stack([
            emb[b][mask[b]] for b in range(B) #從emb裡的資料跑過每一個batch找出 mask = True 對應的數值
        ])  # List[Li,D] -> ragged; we pad to max retained length
        max_kept = max(m.size(0) for m in memory)
        memory_padded = torch.zeros(B, max_kept, D, device=emb.device) #[B, L_max, D]
        mem_pad_mask = torch.ones(B, max_kept, dtype=torch.bool, device=emb.device)
        for b, m in enumerate(memory):
            memory_padded[b, : m.size(0)] = m
            mem_pad_mask[b, : m.size(0)] = False  # False means valid token for Transformer

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
