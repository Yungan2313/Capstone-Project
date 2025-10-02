import os
import pandas as pd

DATASET_DIR = "data/datasets/"  
OUTPUT_FILE = "range_report.txt"

# 北京範圍
BEIJING_MIN_LAT, BEIJING_MAX_LAT = 39.40, 41.10
BEIJING_MIN_LON, BEIJING_MAX_LON = 115.25, 117.30

all_file_ranges = []
non_beijing_files = []

global_min_lat = float("inf")
global_max_lat = float("-inf")
global_min_lon = float("inf")
global_max_lon = float("-inf")

with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    for fname in os.listdir(DATASET_DIR):
        if not fname.endswith(".csv"):
            continue
        fpath = os.path.join(DATASET_DIR, fname)
        try:
            df = pd.read_csv(fpath)
            min_lat = df["lat"].min()
            max_lat = df["lat"].max()
            min_lon = df["lon"].min()
            max_lon = df["lon"].max()

            all_file_ranges.append({
                "file": fname,
                "min_lat": min_lat,
                "max_lat": max_lat,
                "min_lon": min_lon,
                "max_lon": max_lon,
                "lat_range": max_lat - min_lat,
                "lon_range": max_lon - min_lon
            })

            # 判斷是否完全落在北京範圍內
            if (min_lat < BEIJING_MIN_LAT or max_lat > BEIJING_MAX_LAT or
                min_lon < BEIJING_MIN_LON or max_lon > BEIJING_MAX_LON):
                non_beijing_files.append(fname)

            # 更新整體範圍
            global_min_lat = min(global_min_lat, min_lat)
            global_max_lat = max(global_max_lat, max_lat)
            global_min_lon = min(global_min_lon, min_lon)
            global_max_lon = max(global_max_lon, max_lon)

        except Exception as e:
            out.write(f"[ERROR] Failed reading {fname}: {e}\n")

    # 全體統計
    out.write("📊 Global Lat Range: {:.6f} ~ {:.6f}\n".format(global_min_lat, global_max_lat))
    out.write("📊 Global Lon Range: {:.6f} ~ {:.6f}\n".format(global_min_lon, global_max_lon))
    out.write("🌏 Total Lat Span: {:.6f}\n".format(global_max_lat - global_min_lat))
    out.write("🌍 Total Lon Span: {:.6f}\n\n".format(global_max_lon - global_min_lon))

    # 離群範圍資料（舊判斷）
    out.write("🔍 Possible outliers (check manually):\n")
    for r in all_file_ranges:
        if abs(r["min_lat"] - global_min_lat) > 1 or abs(r["max_lat"] - global_max_lat) > 1 \
           or abs(r["min_lon"] - global_min_lon) > 1 or abs(r["max_lon"] - global_max_lon) > 1:
            out.write(f"- {r['file']} | lat: {r['min_lat']:.3f}~{r['max_lat']:.3f}, "
                      f"lon: {r['min_lon']:.3f}~{r['max_lon']:.3f}\n")

    # 額外列出非北京的資料
    out.write("\n🚩 Files NOT in Beijing range:\n")
    for fname in non_beijing_files:
        out.write(f"- {fname}\n")

print(f"✅ 分析完成，結果已寫入：{OUTPUT_FILE}")
