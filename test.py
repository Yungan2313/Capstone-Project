import torch, yaml, argparse, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from helper.load_config import load_config
from model.model import TrajSimplificationModel
from helper.grid_utils import latlon_to_grid, grid_to_latlon
from train import masked_ce
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# ------------------------ CLI ------------------------
# parser = argparse.ArgumentParser()
# parser.add_argument("--ckpt", required=True, help="./checkpoint/20250803-165333/best.pt")
# parser.add_argument("--csv",  required=True, help="./data/datasets/179_000022.csv")
# parser.add_argument("--max_iter", type=int, default=30, help="fine-tune rounds")
# parser.add_argument("--patience", type=int, default=5, help="stop if mask unchanged for N rounds")
# args = parser.parse_args()
ckpt_path = "./checkpoint/20250803-165333/best.pt"         # ← 換成你的 .pt 路徑
# csv_path  = "./data/test/test.csv"       # ← 換成要測的軌跡
csv_path  = "./data/datasets/020_000029.csv"       # ← 換成要測的軌跡
max_iter  = 30
patience  = 5
save_path = Path("./result/test")
class _Args: pass
args = _Args()
args.ckpt     = ckpt_path
args.csv      = csv_path
args.max_iter = max_iter
args.patience = patience
# -------------------- 1. load cfg & model --------------------
ckpt = torch.load(args.ckpt, map_location="cpu")
cfg  = ckpt["cfg"]                    # 已存成 dict
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

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# -------------------- 3. iterative fine-tune --------------------
best_mask = torch.zeros_like(gx, dtype=torch.bool)
no_imp = 0
for it in range(1, args.max_iter + 1):
    model.train()
    out = model(gx, gy, pad_mask)

    # loss 一樣：交叉熵 (全域 logits)
    from train import masked_ce               # 直接重用
    loss_x = masked_ce(out["logits_x"], gx, pad_mask)
    loss_y = masked_ce(out["logits_y"], gy, pad_mask)
    loss   = loss_x + loss_y

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 取得此次的 mask (True = 保留點)
    cur_mask = out["mask"].detach()                 # tensor on same device as model
    cur_mask = cur_mask.to(best_mask.device)        # ★ 確保跟 best_mask 在同一裝置
    changed = not torch.equal(cur_mask, best_mask)  # ← 不再因 CPU / CUDA 混用而報錯
    best_mask = cur_mask.clone()
    # 檢查是否跟上回一致
    if torch.equal(cur_mask, best_mask):
        no_imp += 1
    else:
        best_mask = cur_mask.clone()
        no_imp = 0

    print(f"iter {it:02d} | loss {loss.item():.4f} | kept {cur_mask.sum().item()} pts")

    if no_imp >= args.patience:
        print(f"mask stable for {args.patience} rounds → stop")
        break

# -------------------- 4. 輸出保留點 --------------------
keep_idx = best_mask.squeeze(0).nonzero(as_tuple=False).squeeze(1).cpu()  # 1-D indices
lat_kept, lon_kept = df["lat"].iloc[keep_idx], df["lon"].iloc[keep_idx]
out_df = pd.DataFrame({
    "idx": keep_idx.numpy(),
    "lat": lat_kept.values,
    "lon": lon_kept.values
})
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