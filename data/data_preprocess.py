from pathlib import Path
import pandas as pd
import math

original_folder = Path("data/")
folder = Path("data/datasets/")
SPLIT_SIZE = 2_000
split_list = []

for csv_path in folder.glob("*.csv"):
    df_iter = pd.read_csv(csv_path)  # 仍一次讀入；需要省 RAM 可用 chunksize
    total_rows = len(df_iter)

    if total_rows <= SPLIT_SIZE:
        continue  # 不用處理

    # === 解析檔名 ===
    stem = csv_path.stem                # e.g. '010_000001'
    prefix, suffix = stem.split("_", 1) # 只分第一個 '_'
    middle_num = int(suffix[:3])
    last_id    = suffix[3:6]

    # === 寫回原檔（前 2K 列）===
    df_iter.iloc[:SPLIT_SIZE].to_csv(csv_path, index=False)

    # === 分割其餘列 ===
    remaining = df_iter.iloc[SPLIT_SIZE:]
    n_parts = math.ceil(len(remaining) / SPLIT_SIZE)

    for i in range(n_parts):
        part     = remaining.iloc[i*SPLIT_SIZE:(i+1)*SPLIT_SIZE]
        new_mid  = f"{middle_num + i + 1:03d}"
        new_name = f"{prefix}_{new_mid}{last_id}.csv"
        part.to_csv(csv_path.with_name(new_name), index=False)

    split_list.append(csv_path.name)

# === 紀錄被切割的原始檔名 ===
with open(original_folder / "split_files.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(split_list))

print(f"Done! 已切割 {len(split_list)} 個檔案")
