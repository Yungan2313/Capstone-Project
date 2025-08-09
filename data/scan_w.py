"""
scan_w.py  —  執行後印出 90% / 95% / 99% 分位的 w
"""
import pandas as pd, numpy as np, glob, os, yaml, math
from pathlib import Path
from helper.load_config import load_config
# 讀 cell_size, min_lat/lon
cfg = load_config("config/config.yaml")
CELL = cfg["data"]["cell_size"]
MIN_LAT, MIN_LON = cfg["data"]["min_lat"], cfg["data"]["min_lon"]

def latlon_to_grid(lat, lon):
    gx = np.floor((lon - MIN_LON) / CELL).astype(int)
    gy = np.floor((lat - MIN_LAT) / CELL).astype(int)
    return gx, gy

ws = []
for f in Path(cfg["data"]["dataset_dir"]).glob("*.csv"):
    df   = pd.read_csv(f)
    gx, gy = latlon_to_grid(df["lat"].values, df["lon"].values)
    # 每條軌跡在 x / y 方向上的「半寬」
    w_i = max((gx.max() - gx.min())//2, (gy.max() - gy.min())//2)
    ws.append(w_i)

for q in (0.90, 0.95, 0.99):
    print(f"{int(q*100)}%  quantile w  = {int(np.quantile(ws, q))}")
