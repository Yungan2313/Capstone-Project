import torch
from torch.utils.data import DataLoader
from model import TrajectoryTransformer
from dataloader import TrajectoryDataset
import torch.nn as nn
import torch.optim as optim

# 自動選擇 GPU 或 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 顯示 GPU 名稱（如果有）
if device.type == "cuda":
    print(f"🚀 GPU name: {torch.cuda.get_device_name(0)}")

# 載入資料
dataset = TrajectoryDataset("C:/Users/User/Desktop/data/training_data=80%")
loader = DataLoader(dataset, batch_size=1, shuffle=True)

# 建立模型並移至 GPU
model = TrajectoryTransformer().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.MSELoss()

# 開始訓練
for epoch in range(50):
    total_loss = 0
    for input_seq, target_seq, _, _ in loader:
        input_seq = input_seq.to(device)
        target_seq = target_seq.to(device)

        output_seq = model(input_seq, target_seq[:, :-1, :])
        loss = criterion(output_seq, target_seq[:, 1:, :])

        if torch.isnan(loss):
            print("⚠️ Warning: NaN Loss Detected. Skipping batch.")
            continue

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

# 儲存模型
torch.save(model.state_dict(), "model.pt")
