import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from model import build_transformer
from dataloader import TrajectoryDataset, pad_collate_fn
import os
from datetime import datetime

# ---------- 訓練參數 ----------
BATCH_SIZE = 8
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- 儲存路徑與命名 ----------
time_str = datetime.now().strftime("%m%d_%H%M")  # e.g. 0410_2350
MODEL_DIR = "checkpoints"
os.makedirs(MODEL_DIR, exist_ok=True)
SAVE_PATH = os.path.join(MODEL_DIR, f"best_model_{time_str}.pth")

# ---------- 初始化模型與資料 ----------
dataset = TrajectoryDataset("Compressed_Data")
train_size = int(0.8 * len(dataset))
valid_size = len(dataset) - train_size
train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate_fn)

model = build_transformer().to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

best_valid_loss = float('inf')
best_model_state = None

# ---------- 訓練流程 ----------
for epoch in range(NUM_EPOCHS):
    # ---Training---
    model.train()
    total_loss = 0.0

    for batch_idx, (src, tgt, src_mask, tgt_mask) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")):
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        src_mask = src_mask.to(DEVICE).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)
        tgt_mask = tgt_mask.to(DEVICE).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)

        optimizer.zero_grad()
        output = model(src, tgt, src_mask, tgt_mask)  # (B, T, 2)

        # 搭配 mask 計算 loss（忽略 padding）
        mask = tgt_mask.squeeze(1).squeeze(1)  # (B, T)
        loss = criterion(output[mask], tgt[mask])

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {avg_train_loss:.6f}")

    # ---Validation---
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch_idx, (src, tgt, src_mask, tgt_mask) in enumerate(tqdm(valid_loader, desc=f"Epoch {epoch+1} [Valid]")):
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)
            src_mask = src_mask.to(DEVICE).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)
            tgt_mask = tgt_mask.to(DEVICE).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)

            output = model(src, tgt, src_mask, tgt_mask)  # (B, T, 2)
            mask = tgt_mask.squeeze(1).squeeze(1)  # (B, T)
            loss = criterion(output[mask], tgt[mask])

            total_loss += loss.item()

    avg_valid_loss = total_loss / len(valid_loader)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Valid Loss: {avg_valid_loss:.6f}")

    # 儲存最佳模型狀態（不馬上寫檔）
    if avg_valid_loss < best_valid_loss:
        best_valid_loss = avg_valid_loss
        best_model_state = model.state_dict()
        print(f"\nNew best model found at epoch {epoch+1} with Valid Loss: {best_valid_loss:.6f}\n")

# 訓練完成後儲存最佳模型
if best_model_state:
    torch.save(best_model_state, SAVE_PATH)
    print(f"\nTraining complete. Best model saved as {SAVE_PATH}\n")