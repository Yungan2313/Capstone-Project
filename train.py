import os, copy, time
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import yaml
import pandas as pd
from matplotlib.ticker import MaxNLocator

# ---------------- project modules ----------------
from model.model import TrajSimplificationModel
from data.dataloader import TrajDataModule
from helper.load_config import load_config
from helper.graph import draw_loss
# -----------------------------------------------------------------------------
#  ★ 1. training loop  ★
# -----------------------------------------------------------------------------
def _make_axis_centers(n: int, device):
    # 以格子中心為座標：0.5, 1.5, 2.5, ... (單位：格子)
    return (torch.arange(n, device=device, dtype=torch.float32) + 0.5)

def sed_from_two_logits(
    logits_x, logits_y,            # [B,L,Cx], [B,L,Cy]
    gx, gy,                         # [B,L],    [B,L]  (int)
    pad_mask,                       # [B,L] True=PAD
    x_centers, y_centers,           # [Cx], [Cy] (float)
    tau: float = 1.0,
    eps: float = 1e-6
):
    # soft-argmax → 連續座標
    px = F.softmax(logits_x / tau, dim=-1)          # [B,L,Cx]
    py = F.softmax(logits_y / tau, dim=-1)          # [B,L,Cy]
    pred_x = torch.matmul(px, x_centers)            # [B,L]
    pred_y = torch.matmul(py, y_centers)            # [B,L]

    # GT 連續座標（用中心）
    gt_x = x_centers[gx]                            # [B,L]
    gt_y = y_centers[gy]                            # [B,L]

    diff2 = (pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2
    sed   = torch.sqrt(diff2 + eps * eps)           # [B,L]
    if pad_mask is not None:
        sed = sed.masked_fill(pad_mask, 0.0)
        denom = (~pad_mask).sum().clamp_min(1)
    else:
        denom = torch.tensor(sed.numel(), device=sed.device)
    return sed.sum() / denom

def masked_ce(logits, target, pad_mask):
    """
    logits : [B, L, C]   (未轉置才符合 cross_entropy)
    target : [B, L]
    pad_mask : [B, L]    True = PAD
    """
    B, L, C = logits.shape
    loss = F.cross_entropy(
        logits.view(B * L, C),          # [B*L, C]
        target.view(B * L),             # [B*L]
        reduction="none"
    )                                   # => [B*L]
    loss = loss.view(B, L)
    loss = loss.masked_fill(pad_mask, 0.)
    return loss.sum() / (~pad_mask).sum()   # 只對有效 token 取平均
    
def train_one_epoch(model, loader, optimizer, device, cfg):
    model.train()
    total_loss = 0.0
    for gx, gy, pad_mask in tqdm(loader, desc="train", leave=True):
        gx, gy, pad_mask = gx.to(device), gy.to(device), pad_mask.to(device)
        out: Dict[str, torch.Tensor] = model(gx, gy, pad_mask)  # <-- forward
        Cx = out["logits_x"].size(-1)
        Cy = out["logits_y"].size(-1)
        x_centers = _make_axis_centers(Cx, device=out["logits_x"].device)
        y_centers = _make_axis_centers(Cy, device=out["logits_y"].device)
        loss = sed_from_two_logits(
            out["logits_x"], out["logits_y"],
            gx, gy, pad_mask,
            x_centers, y_centers,
            tau=(cfg["training"].get("softargmax_tau", 1.0) if isinstance(cfg, dict) and "training" in cfg else 1.0),
            eps=1e-6
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * gx.size(0)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device, cfg):
    model.eval()
    total_loss = 0.0
    for gx, gy, pad_mask in tqdm(loader, desc="valid", leave=False):
        gx, gy, pad_mask = gx.to(device), gy.to(device), pad_mask.to(device)
        out: Dict[str, torch.Tensor] = model(gx, gy, pad_mask)  # <-- forward
        Cx = out["logits_x"].size(-1)
        Cy = out["logits_y"].size(-1)
        x_centers = _make_axis_centers(Cx, device=out["logits_x"].device)
        y_centers = _make_axis_centers(Cy, device=out["logits_y"].device)
        loss = sed_from_two_logits(
            out["logits_x"], out["logits_y"],
            gx, gy, pad_mask,
            x_centers, y_centers,
            tau=getattr(cfg["training"], "softargmax_tau", 1.0) if isinstance(cfg, dict) and "training" in cfg else 1.0,
            eps=1e-6
        ) 
        total_loss += loss.item() * gx.size(0)
    return total_loss / len(loader.dataset) 


# -----------------------------------------------------------------------------
#  ★ 2. main  ★
# -----------------------------------------------------------------------------
def main():
    print("Starting training...")
    # ----------------------- paths & config -----------------------
    cfg = load_config("config/config.yaml")  # 讀取 config.yaml
    # data_cfg  = DataConfig.from_yaml(cfg["data"])
    # model_cfg = ModelConfig.from_yaml(cfg)
    train_cfg = cfg["training"]                # 新增 training 區
    res_dir = Path("result") / time.strftime("%Y%m%d-%H%M%S")
    res_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path("checkpoint") / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best.pt"                            # <<< ADD
    last_path = run_dir / "last.pt"                            # <<< ADD
    best_val = float("inf")                                    # <<< ADD
    # ----------------------- data -----------------------
    print("Preparing data...")
    data_args = cfg["data"]
    dm = TrajDataModule(
        data_dir        = cfg["data"]["dataset_dir"],
        split_ratio     = tuple(train_cfg["split_ratio"]),
        max_len         = cfg["model"]["max_seq_len"],
        batch_size      = train_cfg["batch_size"],
        skip_list_path  = cfg["data"]["skip_list"],
        num_workers     = train_cfg["num_workers"],
    )
    loaders = dm.loaders()
    print(f"Train: {len(loaders['train'].dataset)} samples")
    print(f"Val:   {len(loaders['val'].dataset)} samples")
    # ----------------------- model -----------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = TrajSimplificationModel(cfg).to(device)
    # print(model)  # 可先看看參數量
    # ----------------------- optim -----------------------
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if n.endswith("bias") or "norm" in n else decay).append(p)
    optimizer = torch.optim.AdamW(
        [{"params": decay,    "weight_decay": 1e-4},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                            patience=3, factor=0.5)

    # ----------------------- train loop -----------------------
    best_val = float("inf")
    patience, no_imp = 10, 0
    num_epochs = train_cfg["epochs"]
    train_losses = []
    val_losses = []
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}")
        train_loss = train_one_epoch(model, loaders["train"], optimizer, device, cfg)
        val_loss = evaluate(model, loaders["val"], device, cfg)
        scheduler.step(val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"  train loss {train_loss:.4f} | val loss {val_loss:.4f}")

        # --- checkpoint ---
        torch.save({"model": model.state_dict(),
                        "cfg": cfg}, last_path)
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(),
                        "cfg": cfg}, best_path)
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                print("Early stop 🚦")
                break
    draw_loss(train_losses, val_losses, run_dir)  # 畫出 Loss 圖

if __name__ == "__main__":
    main()
