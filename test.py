import torch, yaml, argparse, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from helper.load_config import load_config
from model.model import TrajSimplificationModel
from helper.grid_utils import latlon_to_grid, grid_to_latlon
from train import masked_ce
import os
import matplotlib.animation as animation
import numpy as np
import time
import torch.nn.functional as F


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# ------------------------ CLI ------------------------
# parser = argparse.ArgumentParser()
# parser.add_argument("--ckpt", required=True, help="./checkpoint/20250803-165333/best.pt")
# parser.add_argument("--csv",  required=True, help="./data/datasets/179_000022.csv")
# parser.add_argument("--max_iter", type=int, default=30, help="fine-tune rounds")
# parser.add_argument("--patience", type=int, default=5, help="stop if mask unchanged for N rounds")
# args = parser.parse_args()
ckpt_path = "./checkpoint/20250924-211027_v3.1_mse_withoutv2v3/best.pt"         # ← 換成你的 .pt 路徑
csv_path  = "./data/test/test.csv"       # ← 換成要測的軌跡
csv_path  = "./data/datasets/167_000008.csv"       # ← 換成要測的軌跡
# csv_path  = "./data/datasets/085_000094.csv"       # ← 換成要測的軌跡
# csv_path  = "./data/datasets/085_000083.csv"       # ← 換成要測的軌跡
# csv_path  = "./data/datasets/153_000052.csv"       # ← 換成要測的軌跡
max_iter  = 200
patience  = 10
save_path = Path("./result/test")
save_path.parent.mkdir(parents=True, exist_ok=True)         # <<< ADD: 建資料夾
mask_history, keep_idx_history, loss_history = [], [], []   # <<< ADD: 歷史記錄
scores_history = []                                         # <<< ADD:（可選）記錄 scores
NMS = False  # 是否使用 NMS 後處理（True = 使用）
class _Args: pass
args = _Args()
args.ckpt     = ckpt_path
args.csv      = csv_path
args.max_iter = max_iter
args.patience = patience
args.NMS = NMS
# -------------------- 1. load cfg & model --------------------
ckpt = torch.load(args.ckpt, map_location="cpu")
cfg  = ckpt["cfg"]                    # 已存成 dict
cfg['model']['bottleneck']['compression_ratio'] = 0.2  # 20% of input seq length
model = TrajSimplificationModel(cfg)
model.load_state_dict(ckpt["model"])
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# -------------------- 2. prepare single trajectory --------------------
df = pd.read_csv(args.csv)
gx, gy = latlon_to_grid(df["lat"].values, df["lon"].values,
                        win=cfg["data"]["fixed_window"])
gx = torch.LongTensor(gx).unsqueeze(0).to(device)    # [1,L]
gy = torch.LongTensor(gy).unsqueeze(0).to(device)
pad_mask = gx.eq(cfg["data"]["pad_idx"])             # [1,L]
if "seconds" in df.columns or "timestamp" in df.columns:
    # 嘗試把第 3 欄時間轉 datetime；失敗就當數值處理
    t_series = pd.to_datetime(df.iloc[:, 2], errors="ignore")
    if np.issubdtype(t_series.dtype, np.datetime64):
        t_sec_np = (t_series - t_series.iloc[0]).dt.total_seconds().to_numpy(dtype=float)
    else:
        t_sec_np = pd.to_numeric(df.iloc[:, 2], errors="coerce").to_numpy(dtype=float)
        t_sec_np = t_sec_np - np.nanmin(t_sec_np)
else:
    t_sec_np = np.arange(len(df), dtype=float)
t_b = torch.tensor(t_sec_np, dtype=torch.float32, device=gx.device).unsqueeze(0)  # [1,L]
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# -------------------- 3. iterative fine-tune --------------------
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

def greedy_nms_mask(scores, pad_mask, k_vec, min_gap):
    # scores: [B,L]（用 out["scores"]），pad_mask: [B,L]，k_vec: [B] 每列要保留的數量
    B, L = scores.shape
    valid = ~pad_mask
    sc = scores.masked_fill(pad_mask, float("-inf"))
    out = torch.zeros(B, L, dtype=torch.bool, device=scores.device)
    for b in range(B):
        K = int(k_vec[b])
        if K <= 0: continue
        order = torch.argsort(sc[b], descending=True).tolist()
        chosen = []
        for t in order:
            if not valid[b, t]: continue
            if not chosen or min(abs(t-u) for u in chosen) > min_gap:
                chosen.append(t)
            if len(chosen) >= K: break
        if len(chosen) < K:  # 不夠就補
            for t in order:
                if not valid[b, t] or t in chosen: continue
                chosen.append(t)
                if len(chosen) >= K: break
        out[b, chosen] = True
    # 首尾必留
    first = valid.float().argmax(1)
    last  = (L-1) - torch.flip(valid,[1]).float().argmax(1)
    row = torch.arange(B, device=scores.device)
    out[row, first] = True
    out[row, last]  = True
    return out

best_mask = torch.zeros_like(gx, dtype=torch.bool)
best_loss = float("inf")
no_imp = 0
start = time.time()
for it in range(1, args.max_iter + 1):
    model.train()
    out = model(gx, gy, pad_mask)

    # loss 一樣：交叉熵 (全域 logits)
    # from train import masked_ce               # 直接重用
    # loss_x = masked_ce(out["logits_x"], gx, pad_mask)
    # loss_y = masked_ce(out["logits_y"], gy, pad_mask)
    # loss   = loss_x + loss_y
    from train import sed_from_two_logits, _make_axis_centers, _softargmax_xy, _sed_time_sync_loss
    Cx = out["logits_x"].size(-1)
    Cy = out["logits_y"].size(-1)
    x_centers = _make_axis_centers(Cx, device=out["logits_x"].device)
    y_centers = _make_axis_centers(Cy, device=out["logits_y"].device)
    base = sed_from_two_logits(
        out["logits_x"], out["logits_y"],
        gx, gy, pad_mask,
        x_centers, y_centers,
        tau=(cfg["training"].get("softargmax_tau", 1.0) if isinstance(cfg, dict) and "training" in cfg else 1.0),
        eps=1e-6
    )
    tau = (cfg["training"].get("softargmax_tau", 1.0) if isinstance(cfg, dict) and "training" in cfg else 1.0)
    pred_x, pred_y = _softargmax_xy(out["logits_x"], out["logits_y"], x_centers, y_centers, tau=tau)
    
    loss = _sed_time_sync_loss(
        pred_x, pred_y, t_b,
        out["mask"].bool() if "mask" in out else (out["scores"] > 0),
        pad_mask
    ) + base
    # loss = _sed_time_sync_loss(
    #     pred_x, pred_y, t_b,
    #     out["mask"].bool() if "mask" in out else (out["scores"] > 0),
    #     pad_mask
    # )

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

    
    cur_loss = loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # # 取得此次的 mask (True = 保留點)
    cur_mask = out["mask"].detach()                 # tensor on same device as model
    
    # --- 後處理：最小間隔（例如 4），保持同樣 K ---
    if args.NMS:
        min_gap = 4
        k_vec = cur_mask.sum(1)
        cur_mask = greedy_nms_mask(out["scores"].detach().to(cur_mask.device), pad_mask, k_vec, min_gap)
    
    cur_mask = cur_mask.to(best_mask.device)        # ★ 確保跟 best_mask 在同一裝置
    changed = not torch.equal(cur_mask, best_mask)  # ← 不再因 CPU / CUDA 混用而報錯
    # # 檢查是否跟上回一致
    # if torch.equal(cur_mask, best_mask):
    #     no_imp += 1
    # else:
    #     best_mask = cur_mask.clone()
    #     no_imp = 0
    
    if best_loss < cur_loss:  # 若 loss 有改善
        no_imp += 1
    else:
        best_loss = cur_loss
        best_mask = cur_mask.clone()
        no_imp = 0

    
    # 記錄歷史（本 iter）
    loss_history.append(float(loss.item()))                            
    mask_history.append(cur_mask.squeeze(0).detach().cpu())            
    keep_idx_history.append(mask_history[-1].nonzero(as_tuple=False).squeeze(1))
    if "scores" in out:
        scores_history.append(out["scores"].detach().cpu())          
    print(f"iter {it:02d} | loss {loss.item():.4f} | kept {cur_mask.sum().item()} pts")

    if no_imp >= args.patience:
        print(f"mask stable for {args.patience} rounds → stop")
        break
end = time.time()
print(f"訓練結束，總共 {it} 迭代，耗時 {end-start:.2f} 秒，最低loss {best_loss:.4f}")
# -------------------- 3.5 產生動態 GIF 與每 iter CSV（新增） --------------------
def save_kept_animation_html(df, keep_idx_hist, loss_hist, base_path: Path, title: str):
    """
    產出僅顯示「當前保留點」的 HTML 動畫（不標新加入/被移除）。
    df 需含 'lat','lon'; keep_idx_hist 為 List[1D Tensor 索引]
    """
    if len(keep_idx_hist) == 0:
        print("no history to animate; skip HTML.")
        return

    import numpy as np
    import matplotlib.animation as animation

    lats = df["lat"].to_numpy()
    lons = df["lon"].to_numpy()

    fig = plt.figure(figsize=(7, 7))
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title)
    ax.plot(lons, lats, '-', lw=1, alpha=0.5)   # 原軌跡
    plt.axis("off")

    ims = []  # 每幀的 artist 列表

    for i, idx_t in enumerate(keep_idx_hist, start=1):
        idx_np = idx_t.detach().cpu().numpy().astype(int, copy=False).reshape(-1)
        if idx_np.size > 0:
            XY = np.column_stack([lons[idx_np], lats[idx_np]])
            art_kept = ax.scatter(XY[:,0], XY[:,1], s=20, c='tab:blue')
        else:
            art_kept = ax.scatter([], [], s=20)

        txt = ax.text(
            0.02, 0.02,
            f"iter {i}/{len(keep_idx_hist)}"
            + (f" | loss={loss_hist[i-1]:.4f}" if i-1 < len(loss_hist) else "")
            + f" | kept={idx_np.size}",
            transform=ax.transAxes, fontsize=10, color="#444"
        )

        ims.append([art_kept, txt])

    ani = animation.ArtistAnimation(fig, ims, interval=600, repeat_delay=1000, blit=True)
    html_path = base_path.with_name(base_path.name + "_kept_anim.html")
    try:
        html_str = ani.to_jshtml()
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_str)
        print(f" --> HTML animation saved to {html_path}")
    except Exception as e:
        print(f"[warn] HTML save failed: {e}")
    plt.close(fig)
def save_kept_animation(df, keep_idx_hist, loss_hist, base_path: Path, title: str):
    """
    產出僅顯示「當前保留點」的 GIF 動畫（不標新加入/被移除）。
    """
    if len(keep_idx_hist) == 0:
        print("no history to animate; skip GIF.")
        return

    import numpy as np
    import matplotlib.animation as animation

    lats = df["lat"].values
    lons = df["lon"].values

    fig, ax = plt.subplots(figsize=(6, 6))
    line, = ax.plot(lons, lats, '-', lw=1, alpha=0.5)   # 原軌跡
    scat_kept = ax.scatter([], [], s=20, c='tab:blue')  # 當前保留點
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title)

    def _empty_offsets():
        return np.empty((0, 2), dtype=float)

    def init():
        scat_kept.set_offsets(_empty_offsets())
        ax.legend([line, scat_kept], ["original", "kept"], loc="best")
        return scat_kept, line

    def update(i):
        idx = keep_idx_hist[i].cpu().numpy().astype(int, copy=False).reshape(-1)
        if idx.size:
            XY = np.column_stack([lons[idx], lats[idx]])
        else:
            XY = _empty_offsets()
        scat_kept.set_offsets(XY)

        if i < len(loss_hist):
            ax.set_xlabel(f"iter {i+1}/{len(keep_idx_hist)}  |  loss={loss_hist[i]:.4f}  |  kept={idx.size}")
        else:
            ax.set_xlabel(f"iter {i+1}/{len(keep_idx_hist)}  |  kept={idx.size}")
        return scat_kept, line

    gif_path = base_path.with_name(base_path.name + "_kept_anim.gif")
    try:
        ani = animation.FuncAnimation(fig, update, frames=len(keep_idx_hist),
                                      init_func=init, blit=True, interval=600)
        ani.save(str(gif_path), writer=animation.PillowWriter(fps=2))
        print(f" --> GIF saved to {gif_path}")
    except Exception as e:
        print(f"[warn] GIF save failed: {e}")
    plt.close(fig)
def save_per_iter_csvs(df, keep_idx_hist, base_path: Path):
    base = base_path.with_suffix("")  # ./result/test  → ./result/test
    for i, idx in enumerate(keep_idx_hist, start=1):
        idx_np = idx.cpu().numpy()
        # out_df_i = pd.DataFrame({
        #     "idx": idx_np,
        #     "lat": df["lat"].iloc[idx_np].values,
        #     "lon": df["lon"].iloc[idx_np].values
        # })
        out_df_i = pd.DataFrame({
            "idx": idx_np,
            "lat": df["lat"].iloc[idx_np].values,
            "lon": df["lon"].iloc[idx_np].values,
            "time": t_sec_np[idx_np]
        })
        
        csv_i = base.with_name(base.name + f"_iter_{i:02d}_kept.csv")
        out_df_i.to_csv(csv_i, index=False)
save_kept_animation_html(df, keep_idx_history, loss_history, save_path, Path(args.csv).stem)
save_kept_animation(df, keep_idx_history, loss_history, save_path, Path(args.csv).stem)
save_per_iter_csvs(df, keep_idx_history, save_path)
# -------------------- 4. 輸出保留點 --------------------
keep_idx = best_mask.squeeze(0).nonzero(as_tuple=False).squeeze(1).cpu()  # 1-D indices
lat_kept, lon_kept = df["lat"].iloc[keep_idx], df["lon"].iloc[keep_idx]
out_df = pd.DataFrame({"idx": keep_idx.numpy(),
                       "lat": lat_kept.values,
                       "lon": lon_kept.values,
                       "time": t_sec_np[keep_idx.numpy()]})
final_save_path = Path(save_path).with_suffix("")  # 變成沒有 .csv
final_save_path = final_save_path.with_name(final_save_path.name + "_kept.csv")
out_df.to_csv(final_save_path, index=False)
print(f"kept points saved to {save_path}  ({len(out_df)} pts)")

plt.figure(figsize=(6,6))
plt.plot(df.lon, df.lat, '-', lw=1, alpha=0.5, label="original")
plt.scatter(out_df.lon, out_df.lat, c="red", s=18, label="kept pts")
plt.legend(); plt.axis("equal"); plt.title(Path(args.csv).stem)
img_path = final_save_path.with_suffix(".png")
plt.savefig(img_path, dpi=150, bbox_inches="tight")
print(f" --> plot saved to {img_path}")
try:                 # 若在 notebook / VS Code 內跑，會直接顯示
    plt.show()
except:
    pass