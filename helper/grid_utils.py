import torch
import numpy as np
import yaml
from pathlib import Path
from helper.load_config import load_config

# 讀全域設定 (cell size、min_lat/min_lon)；可在 config.yaml 的 data 區塊裡
_cfg = load_config()
CELL_SIZE = _cfg["data"]["cell_size"]          # 0.0001 之類
MIN_LAT   = _cfg["data"]["min_lat"]
MIN_LON   = _cfg["data"]["min_lon"]
WIN       = _cfg["data"]["fixed_window"]


def latlon_to_grid(lat_arr, lon_arr, win: int = WIN):
    # 1) 絕對格子（仍用 cell_size 量化）
    gx_abs = np.floor((lon_arr - lon_arr[0]) / CELL_SIZE).astype(int)
    gy_abs = np.floor((lat_arr - lat_arr[0]) / CELL_SIZE).astype(int)
    # 2) shift + clip 到 0‥2w
    gx = np.clip(gx_abs + win, 0, 2*win)
    gy = np.clip(gy_abs + win, 0, 2*win)
    # if (gx_abs > win).any() or (gy_abs > win).any():
    #     print("⚠️ clip", gx_abs.max(), gy_abs.max()) # for debug
    return gx, gy


def grid_to_latlon(gx_arr, gy_arr):
    lat = gy_arr * CELL_SIZE + MIN_LAT + CELL_SIZE / 2    # 中心點
    lon = gx_arr * CELL_SIZE + MIN_LON + CELL_SIZE / 2
    return lat, lon

def local_window(grid_x, grid_y, win=3):
    """
    給定每個時間步的中心格子 (gx, gy)，回傳：
      local_ids   : [B, L, (2w+1)^2]  局部格子 id (flatten 到單一 id)
      tgt_pos_in_window : [B, L]       真實格子在 window 中的 offset
    """
    B, L = grid_x.shape
    device = grid_x.device
    rng = torch.arange(-win, win+1, device=device)
    dx, dy = torch.meshgrid(rng, rng, indexing="ij")          # (2w+1, 2w+1)
    dx, dy = dx.flatten(), dy.flatten()                       # K = (2w+1)^2

    local_x = grid_x[..., None] + dx        # [B, L, K]
    local_y = grid_y[..., None] + dy        # [B, L, K]

    # clamp 到合法範圍
    local_x = local_x.clamp(0, _cfg['data']['num_cells_x']-1)
    local_y = local_y.clamp(0, _cfg['data']['num_cells_y']-1)

    local_ids = local_y * _cfg['data']['num_cells_x'] + local_x      # flatten → id
    tgt_pos   = ((win) * (2*win+1) + win)                     # 中央位置
    tgt_pos   = torch.full((B, L), tgt_pos, device=device, dtype=torch.long)

    return local_ids, tgt_pos
