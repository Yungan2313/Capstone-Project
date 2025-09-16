import os, time, copy, argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
import torch.nn.functional as F

from helper.load_config import load_config
from model.model import TrajSimplificationModel
from helper.grid_utils import grid_to_latlon
from data.dataloader import TrajDataModule

# 防止 OMP 衝突（和你 test.py 同步）
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ------------------------ CLI ------------------------
def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=False,
                   default="./checkpoint/20250910-184624_v2withsmalldecoder/best.pt",
                   help="path to best.pt")
    p.add_argument("--out_root", default="./result/eval",
                   help="base folder for all evaluate outputs")
    p.add_argument("--max_iter", type=int, default=200)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--nms", action="store_true", help="apply greedy NMS spacing on mask")
    p.add_argument("--min_gap", type=int, default=4, help="min index gap when NMS on")
    p.add_argument("--cratio", type=float, default=0.2,
                   help="override compression_ratio (e.g., 0.2). If None, use cfg.")
    p.add_argument("--save_html", action="store_true", help="also save HTML animation")
    p.add_argument("--save_gif", action="store_true", help="also save GIF animation")
    return p

# -------------------- utils: naming --------------------
def sample_name_from_dataset(dataset, global_idx: int) -> str:
    """
    優先用資料集的檔名/識別資訊；拿不到就用 sample_{index:05d}
    """
    # 嘗試常見欄位 / 介面
    for key in ["name", "filename", "file", "path", "id"]:
        v = getattr(dataset, key, None)
        if isinstance(v, list) and 0 <= global_idx < len(v):
            return Path(v[global_idx]).stem
        if isinstance(v, dict) and global_idx in v:
            return Path(v[global_idx]).stem
    # 有些 Dataset 會有 metas
    metas = getattr(dataset, "metas", None)
    if isinstance(metas, list) and 0 <= global_idx < len(metas):
        m = metas[global_idx]
        for key in ["name", "filename", "file", "path", "id"]:
            if key in m:
                return Path(m[key]).stem
    return f"sample_{global_idx:05d}"

# -------------------- utils: mask後處理 --------------------
def greedy_nms_mask(scores, pad_mask, k_vec, min_gap: int):
    """
    與你之前 test/evaluate 版本一致的最小間隔 NMS，保持 K 不變。
    """
    B, L = scores.shape
    valid = ~pad_mask
    sc = scores.masked_fill(pad_mask, float("-inf"))
    out = torch.zeros(B, L, dtype=torch.bool, device=scores.device)
    for b in range(B):
        K = int(k_vec[b])
        if K <= 0:
            continue
        order = torch.argsort(sc[b], descending=True).tolist()
        chosen = []
        for t in order:
            if not valid[b, t]:
                continue
            if not chosen or min(abs(t - u) for u in chosen) > min_gap:
                chosen.append(t)
            if len(chosen) >= K:
                break
        if len(chosen) < K:
            for t in order:
                if not valid[b, t] or t in chosen:
                    continue
                chosen.append(t)
                if len(chosen) >= K:
                    break
        out[b, chosen] = True
    # 首尾必留
    first = valid.float().argmax(1)
    last = (L - 1) - torch.flip(valid, [1]).float().argmax(1)
    row = torch.arange(B, device=scores.device)
    out[row, first] = True
    out[row, last] = True
    return out

# -------------------- utils: 動畫/圖表 --------------------
def save_kept_animation_gif(lats, lons, keep_idx_hist, loss_hist, out_path_gif: Path, title: str):
    if len(keep_idx_hist) == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    line, = ax.plot(lons, lats, '-', lw=1, alpha=0.5)
    scat = ax.scatter([], [], s=20)  # kept
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title)

    def _empty_offsets():
        return np.empty((0, 2), dtype=float)

    def init():
        scat.set_offsets(_empty_offsets())
        ax.legend([line, scat], ["original", "kept"], loc="best")
        return scat, line

    def update(i):
        idx = keep_idx_hist[i]
        if idx.size:
            XY = np.column_stack([lons[idx], lats[idx]])
        else:
            XY = _empty_offsets()
        scat.set_offsets(XY)
        if i < len(loss_hist):
            ax.set_xlabel(f"iter {i+1}/{len(keep_idx_hist)} | loss={loss_hist[i]:.4f} | kept={idx.size}")
        else:
            ax.set_xlabel(f"iter {i+1}/{len(keep_idx_hist)} | kept={idx.size}")
        return scat, line

    ani = animation.FuncAnimation(fig, update, frames=len(keep_idx_hist),
                                  init_func=init, blit=True, interval=600)
    try:
        ani.save(str(out_path_gif), writer=animation.PillowWriter(fps=2))
    except Exception as e:
        print(f"[warn] GIF save failed: {e}")
    plt.close(fig)

def save_kept_animation_html(lats, lons, keep_idx_hist, loss_hist, out_path_html: Path, title: str):
    if len(keep_idx_hist) == 0:
        return
    fig = plt.figure(figsize=(7, 7))
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title)
    ax.plot(lons, lats, '-', lw=1, alpha=0.5)
    plt.axis("off")

    ims = []
    for i, idx in enumerate(keep_idx_hist, start=1):
        if idx.size > 0:
            XY = np.column_stack([lons[idx], lats[idx]])
            art_kept = ax.scatter(XY[:, 0], XY[:, 1], s=20)
        else:
            art_kept = ax.scatter([], [], s=20)
        txt = ax.text(
            0.02, 0.02,
            f"iter {i}/{len(keep_idx_hist)}"
            + (f" | loss={loss_hist[i-1]:.4f}" if i-1 < len(loss_hist) else "")
            + f" | kept={idx.size}",
            transform=ax.transAxes, fontsize=10, color="#444"
        )
        ims.append([art_kept, txt])

    ani = animation.ArtistAnimation(fig, ims, interval=600, repeat_delay=1000, blit=True)
    try:
        html_str = ani.to_jshtml()
        with open(out_path_html, "w", encoding="utf-8") as f:
            f.write(html_str)
    except Exception as e:
        print(f"[warn] HTML save failed: {e}")
    plt.close(fig)

def save_final_plot(lats, lons, kept_lats, kept_lons, out_png: Path, title: str):
    plt.figure(figsize=(6, 6))
    plt.plot(lons, lats, '-', lw=1, alpha=0.5, label="original")
    plt.scatter(kept_lons, kept_lats, s=18, label="kept pts")
    plt.legend()
    plt.axis("equal")
    plt.title(title)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()

# -------------------- bin balance loss --------------------
def bin_balance_loss(y, pad_mask, bins=16, mode="under"):
    """
    y:        [B, L]  可微 top-k 權重(0~1)；可能是 None（硬化路徑）
    pad_mask: [B, L]  True=PAD, False=valid；也可能是 None
    bins:     int     分段數
    mode:     "mse" | "under"
    """
    # --- 防呆：若沒有 y（例如硬化或測試），直接回傳 0（在正確的 device 上） ---
    device = None
    if isinstance(pad_mask, torch.Tensor):
        device = pad_mask.device
    if y is None:
        return torch.tensor(0.0, device=device)

    B, L = y.shape
    if device is None:
        device = y.device

    valid = (~pad_mask).float() if isinstance(pad_mask, torch.Tensor) else torch.ones(B, L, device=device)
    y = y * valid  # 去掉 PAD 影響

    # 每個 token 的 bin 索引（按時間等寬切 bins 份）
    pos = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
    bin_idx = (pos * bins // L).clamp(max=bins - 1)  # [B, L]

    # 計算各 bin 的 "質量"
    bin_y = torch.zeros(B, bins, device=device)
    bin_v = torch.zeros(B, bins, device=device)
    bin_y.scatter_add_(1, bin_idx, y)        # Σ y_i in bin
    bin_v.scatter_add_(1, bin_idx, valid)    # Σ valid_i in bin

    # 轉成「比例」
    eps = 1e-8
    p_hat = bin_y / (bin_y.sum(dim=1, keepdim=True) + eps)  # 估計：y 的分佈
    p_tgt = bin_v / (bin_v.sum(dim=1, keepdim=True) + eps)  # 目標：長度分佈

    if mode == "mse":
        loss = ((p_hat - p_tgt) ** 2).sum(dim=1).mean()
    else:  # "under" 只懲罰不足的 bin
        loss = (F.relu(p_tgt - p_hat)).sum(dim=1).mean()
    return loss

# -------------------- main --------------------
def main():
    args = build_argparser().parse_args()

    # 1) 讀 ckpt & model
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt["cfg"]  # dict
    cfg["model"]["bottleneck"]["compression_ratio"] = float(args.cratio)

    model = TrajSimplificationModel(cfg)
    model.load_state_dict(ckpt["model"])
    base_state = copy.deepcopy(model.state_dict())  # 用來每條軌跡前 reset
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # 2) dataloader（沿用你 evaluate 中的作法）
    eva_cfg = cfg["training"]
    dm = TrajDataModule(
        data_dir       = cfg["data"]["dataset_dir"],
        split_ratio    = tuple(eva_cfg["split_ratio"]),
        max_len        = cfg["model"]["max_seq_len"],
        batch_size     = eva_cfg["batch_size"],
        skip_list_path = cfg["data"]["skip_list"],
        num_workers    = eva_cfg["num_workers"],
    )
    loaders = dm.loaders()
    test_loader = loaders["test"]
    test_ds = test_loader.dataset
    print(f"Test: {len(test_ds)} samples")

    # 3) 建立這次 evaluate 的時間戳輸出資料夾
    ts = time.strftime("%Y%m%d-%H%M")  # 例：20250911-1615
    session_dir = Path(args.out_root) / ts
    session_dir.mkdir(parents=True, exist_ok=True)
    ckpt_basename = Path(args.ckpt).name
    ckpt_fullpath = Path(args.ckpt).resolve().as_posix()
    try:
        (session_dir / "eval_checkpoint.txt").write_text(
            f"{ckpt_basename}\n{ckpt_fullpath}\n", encoding="utf-8"
        )
    except Exception as e:
        print(f"[warn] cannot write eval_checkpoint.txt in {session_dir}: {e}")


    # 小工具：從格點還原 Lat/Lon（若無原始 df 可用）
    def grid_to_ll_np(gx_1d: np.ndarray, gy_1d: np.ndarray):
        # 使用你專案既有的 grid_to_latlon
        lats, lons = grid_to_latlon(gx_1d, gy_1d)
        return np.asarray(lats), np.asarray(lons)

    # 4) 評估全部軌跡（支援 batch，但我們會逐樣本處理與存檔）
    global_index = 0
    pbar = tqdm(test_loader, desc="evaluate", leave=True)
    for batch in pbar:
        gx, gy, pad_mask = batch  # 形狀：[B, L]
        gx = gx.to(device)
        gy = gy.to(device)
        pad_mask = pad_mask.to(device)
        B, L = gx.shape

        # 逐樣本處理
        for b in range(B):
            # 重置模型到 checkpoint 狀態，以避免上一條微調污染這條
            model.load_state_dict(base_state)
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

            gx_b = gx[b:b+1]              # [1, L]
            gy_b = gy[b:b+1]
            pad_b = pad_mask[b:b+1]

            # 嘗試取得樣本名稱；拿不到就用 index
            name = sample_name_from_dataset(test_ds, global_index)

            # 輸出資料夾：為每條軌跡建子資料夾
            # 以原始檔案名稱的 stem 當作樣本資料夾名稱（可直接對回 ./data/datasets/{stem}.csv）
            stem = sample_name_from_dataset(test_ds, global_index)
            out_dir = session_dir / stem
            out_dir.mkdir(parents=True, exist_ok=True)

            # histories（每條軌跡各自重置）
            loss_hist: list[float] = []
            keep_hist: list[np.ndarray] = []

            best_mask = torch.zeros_like(gx_b, dtype=torch.bool)
            best_loss = float("inf")
            no_imp = 0
            start = time.time()

            # 迭代微調（與你先前單軌跡 test 一致）
            for it in range(1, args.max_iter + 1):
                out = model(gx_b, gy_b, pad_b)

                # 你新版 loss：SED（與 evaluate/test 的寫法一致）
                from train import sed_from_two_logits, _make_axis_centers
                Cx = out["logits_x"].size(-1)
                Cy = out["logits_y"].size(-1)
                x_centers = _make_axis_centers(Cx, device=out["logits_x"].device)
                y_centers = _make_axis_centers(Cy, device=out["logits_y"].device)
                loss = sed_from_two_logits(
                    out["logits_x"], out["logits_y"],
                    gx_b, gy_b, pad_b,
                    x_centers, y_centers,
                    tau=(cfg["training"].get("softargmax_tau", 1.0)
                         if isinstance(cfg, dict) and "training" in cfg else 1.0),
                    eps=1e-6
                )
                
                # ----- Modified Losses -----
                bb_cfg = cfg.get("losses", {}).get("bin_balance", {})
                if bb_cfg.get("enabled", False):
                    y_gate = out.get("soft_gate", None)  # 來自 model.forward
                    if y_gate is not None:               # 只有有軟 gate 時才加
                        bb_loss = bin_balance_loss(
                            y_gate, pad_mask,
                            bins=int(bb_cfg.get("bins", 16)),
                            mode=bb_cfg.get("mode", "under"),
                        )
                        loss = loss + float(bb_cfg.get("weight", 0.1)) * bb_loss
                    # else: 沒有 y_gate（硬化或測試）就不加這項 loss


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                cur_mask = out["mask"].detach()  # [1, L]
                if args.nms:
                    k_vec = cur_mask.sum(1)
                    cur_mask = greedy_nms_mask(out["scores"].detach().to(cur_mask.device),
                                               pad_b, k_vec, args.min_gap)

                # early stop 邏輯（用 loss 改善與否）
                cur_loss = float(loss.item())
                if best_loss <= cur_loss:
                    no_imp += 1
                else:
                    best_loss = cur_loss
                    best_mask = cur_mask.clone()
                    no_imp = 0

                # 紀錄當前 iter 的 kept 索引（純 numpy，給 GIF/HTML）
                idx_np = cur_mask.squeeze(0).nonzero(as_tuple=False).squeeze(1).detach().cpu().numpy()
                loss_hist.append(cur_loss)
                keep_hist.append(idx_np)

                pbar.set_postfix({"sample": name, "iter": it, "kept": idx_np.size, "loss": f"{cur_loss:.4f}"})
                if no_imp >= args.patience:
                    break

            end = time.time()
            # print(f"[{name}] iters={it} time={end-start:.2f}s best_loss={best_loss:.4f}")

            # 產生最終輸出：CSV / PNG / (可選) GIF / (可選) HTML
            keep_idx = best_mask.squeeze(0).nonzero(as_tuple=False).squeeze(1).cpu().numpy()  # [K]

            # 直接讀原始 CSV（與 test.py 一致，避免反推誤差）
            src_csv = None
            # 你的 TrajCSVDataset 有 files list（常見設計），直接拿來用：
            if hasattr(test_ds, "files"):
                src_csv = test_ds.files[global_index]
            if src_csv is None:
                # 萬一取不到，最後才 fallback（不建議）
                gx_np = gx_b.squeeze(0).detach().cpu().numpy()
                gy_np = gy_b.squeeze(0).detach().cpu().numpy()
                lats, lons = grid_to_ll_np(gx_np, gy_np)
            else:
                df_src = pd.read_csv(src_csv)
                lats = df_src["lat"].to_numpy()
                lons = df_src["lon"].to_numpy()

            kept_lats = lats[keep_idx]
            kept_lons = lons[keep_idx]

            # CSV（最終）
            out_csv = out_dir / f"{stem}_kept.csv"
            pd.DataFrame({
                "idx": keep_idx,
                "lat": lats[keep_idx],
                "lon": lons[keep_idx],
            }).to_csv(out_csv, index=False)

            # PNG（最終）
            out_png = out_dir / f"{stem}.png"
            save_final_plot(lats, lons, lats[keep_idx], lons[keep_idx], out_png, title=name)

            # GIF / HTML（可選）
            if args.save_gif:
                out_gif = out_dir / f"{stem}_kept_anim.gif"
                save_kept_animation_gif(lats, lons, keep_hist, loss_hist, out_gif, title=name)

            if args.save_html:
                out_html = out_dir / f"{stem}_kept_anim.html"
                save_kept_animation_html(lats, lons, keep_hist, loss_hist, out_html, title=name)


            global_index += 1

    print(f"\nAll done. Outputs saved under: {session_dir}")

if __name__ == "__main__":
    """
    執行指令範例：
    python evaluate.py --ckpt .\checkpoint\20250815-184727_v1.1/best.pt --out_root .\result\eval --max_iter 200 --patience 10 --cratio 0.2 --save_gif --save_html
    """
    main()
