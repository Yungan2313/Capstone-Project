# model/grid_utils.py
import numpy as np
import yaml
from pathlib import Path
from helper.load_config import load_config

# 讀全域設定 (cell size、min_lat/min_lon)；可在 config.yaml 的 data 區塊裡
_cfg = load_config()
CELL_SIZE = _cfg["data"]["cell_size"]          # 0.0001 之類
MIN_LAT   = _cfg["data"]["min_lat"]
MIN_LON   = _cfg["data"]["min_lon"]


def latlon_to_grid(lat_arr, lon_arr):
    """np.ndarray → np.ndarray (gx, gy)"""
    gx = np.floor((lon_arr - MIN_LON) / CELL_SIZE).astype(int)
    gy = np.floor((lat_arr - MIN_LAT) / CELL_SIZE).astype(int)
    return gx, gy


def grid_to_latlon(gx_arr, gy_arr):
    lat = gy_arr * CELL_SIZE + MIN_LAT + CELL_SIZE / 2    # 中心點
    lon = gx_arr * CELL_SIZE + MIN_LON + CELL_SIZE / 2
    return lat, lon
