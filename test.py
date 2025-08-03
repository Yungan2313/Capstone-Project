from calculation import compute_ade_fde
import torch
from model import TrajectoryTransformer
from dataloader import TrajectoryDataset
import matplotlib.pyplot as plt
import os

# 設定設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}")
if device.type == "cuda":
    print(f"🚀 GPU name: {torch.cuda.get_device_name(0)}")

# 建立輸出資料夾
os.makedirs("test_output", exist_ok=True)

# 載入模型
model = TrajectoryTransformer().to(device)
model.load_state_dict(torch.load("model.pt", map_location=device))
model.eval()

# 載入測試資料
dataset = TrajectoryDataset("C:/Users/User/Desktop/data/testing_data=20%")

all_ade = []
all_fde = []

for i in range(len(dataset)):
    input_seq, target_seq, mean, std = dataset[i]
    input_seq = input_seq.unsqueeze(0).to(device)
    target_seq = target_seq.unsqueeze(0).to(device)

    mean = mean.to(device)
    std = std.to(device)

    with torch.no_grad():
        pred = model(input_seq, target_seq[:, :-1, :])

    input_seq = input_seq.squeeze() * std + mean
    target_seq = target_seq.squeeze() * std + mean
    pred = pred.squeeze() * std + mean

    input_seq = input_seq.cpu().numpy()
    target_seq = target_seq.cpu().numpy()
    pred = pred.cpu().numpy()

    if input_seq.ndim == 1:
        input_seq = input_seq.reshape(1, 2)
    if target_seq.ndim == 1:
        target_seq = target_seq.reshape(1, 2)
    if pred.ndim == 1:
        pred = pred.reshape(1, 2)

    # 計算 Haversine ADE / FDE
    ade, fde = compute_ade_fde(pred, target_seq)
    all_ade.append(ade)
    all_fde.append(fde)

    # 繪圖
    plt.figure(figsize=(6, 6))
    plt.plot(input_seq[:, 0], input_seq[:, 1], 'bo-', label="Input (observed)")
    plt.plot(target_seq[:, 0], target_seq[:, 1], 'go-', label="Ground Truth")
    plt.plot(pred[:, 0], pred[:, 1], 'ro--', label="Predicted")

    plt.xlabel("Longitude (經度)", fontsize=12)
    plt.ylabel("Latitude (緯度)", fontsize=12)
    plt.title(f"Trajectory Prediction #{i}", fontsize=14)
    plt.legend()
    plt.grid(True)

    # ➕ 圖右上角顯示 ADE/FDE
    plt.text(0.98, 0.98,
             f"ADE: {ade:.2f} m\nFDE: {fde:.2f} m",
             transform=plt.gca().transAxes,
             fontsize=10,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.7))

    plt.savefig(f"test_output/trajectory_{i}.png")
    plt.close()

# ➕ 輸出平均誤差
avg_ade = sum(all_ade) / len(all_ade)
avg_fde = sum(all_fde) / len(all_fde)
with open("average.txt", "w") as f:
    f.write(f"average_ADE(m): {avg_ade:.4f}\n")
    f.write(f"average_FDE(m): {avg_fde:.4f}\n")

print(f"\n✅ 全部測試完成，圖檔在 test_output/，平均 ADE/FDE 寫入 average.txt")
'''
# 預測每一筆資料並畫圖
for i in range(len(dataset)):
    input_seq, target_seq, mean, std = dataset[i]
    input_seq = input_seq.unsqueeze(0).to(device)     # shape: (1, input_len, 2)
    target_seq = target_seq.unsqueeze(0).to(device)   # shape: (1, pred_len, 2)
    mean = mean.to(device)
    std = std.to(device)

    with torch.no_grad():
        pred = model(input_seq, target_seq[:, :-1, :])  # 預測 pred_len 筆

    # 還原標準化：x * std + mean
    input_seq = input_seq.squeeze() * std + mean
    target_seq = target_seq.squeeze() * std + mean
    pred = pred.squeeze() * std + mean

    # 轉 numpy
    input_seq = input_seq.cpu().numpy()
    target_seq = target_seq.cpu().numpy()
    pred = pred.cpu().numpy()

    # ✅ 畫圖（經度為橫軸，緯度為縱軸）
    plt.figure(figsize=(6, 6))
    plt.plot(input_seq[:, 0], input_seq[:, 1], 'bo-', label="Input (observed)")        # 藍色
    plt.plot(target_seq[:, 0], target_seq[:, 1], 'go-', label="Ground Truth")          # 綠色
    plt.plot(pred[:, 0], pred[:, 1], 'ro--', label="Predicted")                        # 紅色虛線

    plt.xlabel("Longitude (經度)", fontsize=12)
    plt.ylabel("Latitude (緯度)", fontsize=12)
    plt.title(f"Trajectory Prediction #{i}", fontsize=14)
    plt.legend()
    plt.grid(True)

    # 儲存圖檔
    plt.savefig(f"test_output/trajectory_{i}.png")
    plt.close()

print(f"✅ 已完成共 {len(dataset)} 筆的預測與畫圖，圖檔儲存於 test_output/")
'''