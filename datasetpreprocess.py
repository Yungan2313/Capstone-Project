#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import json

def convert_all_trajectories(walk_dir="datasets", out_file="trajs.txt"):
    # 建立輸出檔案
    with open(out_file, "w", encoding="utf-8") as fw:
        # 找到所有 csv，並依名稱排序
        files = sorted(f for f in os.listdir(walk_dir) if f.lower().endswith(".csv"))
        for fn in files:
            path = os.path.join(walk_dir, fn)
            # 讀取 CSV，只要 lat, lon
            df = pd.read_csv(path, usecols=["lat", "lon"])
            # 轉成 [[lon, lat], …]
            coords = [[float(r["lon"]), float(r["lat"])] for _, r in df.iterrows()]
            # 寫成一行 JSON 字串
            fw.write(json.dumps(coords, ensure_ascii=False) + "\n")
    print(f"完成！共處理 {len(files)} 支檔案，輸出到 {out_file}")

if __name__ == "__main__":
    convert_all_trajectories()
