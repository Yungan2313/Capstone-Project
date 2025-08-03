import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import Dataset

# 每一筆資料對應一個 CSV 檔（完整軌跡）
class TrajectoryDataset(Dataset):
    def __init__(self, data_dir, pred_len=5):
        self.samples = []  # 用來存放所有訓練樣本 (input_seq, pred_seq, mean, std)

        for fname in os.listdir(data_dir):
            # 忽略非 CSV 檔案
            if not fname.endswith(".csv"):
                continue

            fpath = os.path.join(data_dir, fname)

            try:
                # 讀取 CSV 並只保留經緯度欄位，lat 是緯度、lon 是經度
                df = pd.read_csv(fpath, encoding='utf-8')
                df = df.loc[:, ['lon', 'lat']]  # 取出指定欄位
                df.dropna(inplace=True)  # 移除缺失值
                coords = df.values.astype('float32')  # 轉為 float32 Numpy 陣列
            except Exception as e:
                print(f"❌ 無法處理檔案 {fpath}：{e}")
                continue

            # 若資料長度太短，無法切出 input + pred，則跳過
            if len(coords) <= pred_len:
                print(f"⚠️ 資料太短（{len(coords)} 筆），跳過：{fpath}")
                continue

            # 切出 input（前面）與預測目標（最後 pred_len 筆）
            input_seq = coords[:-pred_len]  # 前 N-5 筆
            pred_seq = coords[-pred_len:]  # 最後 5 筆作為 ground truth

            # 為了提升模型穩定性，我們會進行標準化（normalization）
            all_seq = np.vstack((input_seq, pred_seq))  # 合併 input + pred
            mean = all_seq.mean(axis=0)  # 算出均值 (lon, lat)
            std = all_seq.std(axis=0)    # 算出標準差 (lon, lat)
            std[std == 0] = 1.0          # 避免除以 0

            # 將 input 和 pred 序列進行標準化（可逆）
            input_seq = (input_seq - mean) / std
            pred_seq = (pred_seq - mean) / std

            # 儲存一筆資料樣本（含均值與標準差）
            self.samples.append((input_seq, pred_seq, mean, std))

    def __len__(self):
        # 回傳總樣本數
        return len(self.samples)

    def __getitem__(self, idx):
        # 根據索引取得對應樣本
        input_seq, pred_seq, mean, std = self.samples[idx]
        return (
            torch.tensor(input_seq),   # (seq_len, 2) tensor，標準化後的 input
            torch.tensor(pred_seq),   # (pred_len, 2) tensor，標準化後的預測目標
            torch.tensor(mean),       # 原始資料均值，用於還原標準化
            torch.tensor(std)         # 原始資料標準差
        )
