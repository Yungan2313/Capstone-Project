import os
import torch
import matplotlib.pyplot as plt
from model import build_transformer
from dataloader import TrajectoryDataset

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# ---------- 設定參數 ----------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "checkpoints/best_model_0410_2354.pth"  # <-- 修改為你要測試的模型檔案
DATA_PATH = "Compressed_Data"
EXAMPLE_INDEX = 2  # 選擇測試第幾筆資料

# ---------- 載入模型 ----------
model = build_transformer().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ---------- 載入資料 ----------
dataset = TrajectoryDataset(DATA_PATH)
src, tgt = dataset[EXAMPLE_INDEX]

# ---------- 製作 mask ----------
src_mask = (src[:, 0] != 0).int()
tgt_mask = (tgt[:, 0] != 0).int()

# ---------- 準備資料 ----------
src = src.unsqueeze(0).to(DEVICE)  # (1, S, 2)
tgt = tgt.unsqueeze(0).to(DEVICE)  # (1, T, 2)
src_mask = src_mask.unsqueeze(0).unsqueeze(1).unsqueeze(2).to(DEVICE)
tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(1).unsqueeze(2).to(DEVICE)

# ---------- 推論 ----------
with torch.no_grad():
    output = model(src, tgt, src_mask, tgt_mask)  # (1, T, 2)

# ---------- 輸出結果 ----------
pred_coords = output.squeeze(0).cpu().numpy()
tgt_coords = tgt.squeeze(0).cpu().numpy()
src_coords = src.squeeze(0).cpu().numpy()

print("\n🔹 Ground Truth:")
print(tgt_coords)
print("\n🔸 Model Output:")
print(pred_coords)

# ---------- 視覺化（雙子圖） ----------
fig, axs = plt.subplots(1, 2, figsize=(16, 7))

# Subplot 1: Input Trajectory
axs[0].plot(src_coords[:, 0], src_coords[:, 1], marker='o', linestyle='-', color='g', label='Input Trajectory')
axs[0].set_title('Input Trajectory')
axs[0].set_xlabel('Longitude')
axs[0].set_ylabel('Latitude')
axs[0].legend()
axs[0].grid(True)

# Subplot 2: Ground Truth vs Model Output
# axs[1].plot(tgt_coords[:, 0], tgt_coords[:, 1], marker='o', linestyle='-', color='r', label='Ground Truth')
axs[1].plot(pred_coords[:, 0], pred_coords[:, 1], marker='o', linestyle='-', color='b', label='Model Output')
# axs[1].plot(tgt_coords[0, 0], tgt_coords[0, 1], 'go', markersize=10, label='Start')
# axs[1].plot(tgt_coords[-1, 0], tgt_coords[-1, 1], 'ro', markersize=10, label='End')
axs[1].set_title('Ground Truth vs Model Output')
axs[1].set_xlabel('Longitude')
axs[1].set_ylabel('Latitude')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()
