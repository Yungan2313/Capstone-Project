import os
import pandas as pd

DATASET_DIR = "data/datasets/"
CELL_SIZE = 0.001  # 約 10m 每格
SKIP_FILE_LIST = "data/notBeijing.txt"

all_lats, all_lons = [], []
max_len = 0

# 讀取要跳過的檔案清單
with open(SKIP_FILE_LIST, "r", encoding="utf-8") as f:
    skip_files = set(line.strip().lstrip("- ").strip() for line in f if line.strip())
f = ""
# 掃描所有 csv 檔案
for file in os.listdir(DATASET_DIR):
    if file in skip_files or not file.endswith(".csv"):
        continue
    df = pd.read_csv(os.path.join(DATASET_DIR, file))
    if len(df) > max_len:
        f = file
    max_len = max(max_len, len(df))
    all_lats.extend(df['lat'].tolist())
    all_lons.extend(df['lon'].tolist())

# 計算 grid x/y 的邊界
min_lat, max_lat = min(all_lats), max(all_lats)
min_lon, max_lon = min(all_lons), max(all_lons)

num_cells_x = int((max_lon - min_lon) / CELL_SIZE) + 1
num_cells_y = int((max_lat - min_lat) / CELL_SIZE) + 1
total_grids = num_cells_x * num_cells_y

# 顯示結果
print("最大軌跡長度:", max_len)
print("最大軌跡檔案:", f)
print("Grid X 數量（經度方向）:", num_cells_x)
print("Grid Y 數量（緯度方向）:", num_cells_y)
print("總格子數量 (X × Y):", total_grids)
print("經緯度範圍: lat({:.6f} ~ {:.6f}), lon({:.6f} ~ {:.6f})".format(
    min_lat, max_lat, min_lon, max_lon
))
