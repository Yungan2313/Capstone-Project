import os
import pandas as pd
from pathlib import Path
from datetime import datetime

def find_users_with_label(root_dir):
    """
    回傳 root_dir 底下，所有含有 labels.txt 的使用者資料夾名稱列表。
    """
    users = []
    for name in os.listdir(root_dir):
        user_dir = os.path.join(root_dir, name)
        if os.path.isdir(user_dir) and os.path.isfile(os.path.join(user_dir, "labels.txt")):
            users.append(name)
    return users

def process_user_folder(root_dir, user_folder, out_dir):
    """
    處理單一使用者資料夾：
    1. 讀取 labels.txt，篩出 mode == 'walk' 的時段
    2. 根據 start time 對應 plt 檔案，擷取該時段內的點
    3. 計算相對秒差，輸出 CSV
    """
    user_dir   = Path(root_dir) / user_folder
    label_path = user_dir / "labels.txt"
    traj_dir   = user_dir / "Trajectory"

    # 讀 labels.txt
    df_lbl = pd.read_csv(label_path, sep="\t")
    df_walk = df_lbl[df_lbl["Transportation Mode"] == "walk"]
    if df_walk.empty:
        return

    out_idx = 1
    for _, row in df_walk.iterrows():
        # parse start/end time
        t_start = datetime.strptime(row["Start Time"], "%Y/%m/%d %H:%M:%S")
        t_end   = datetime.strptime(row["End Time"],   "%Y/%m/%d %H:%M:%S")

        # plt 檔名
        plt_name = t_start.strftime("%Y%m%d%H%M%S") + ".plt"
        plt_path = traj_dir / plt_name
        if not plt_path.exists():
            # print(f"[{user_folder}] 找不到 {plt_name}，跳過")
            continue

        # 讀 PLT
        df = pd.read_csv(
            plt_path,
            header=None,
            skiprows=6,
            usecols=[0,1,5,6],
            names=["lat","lon","date_str","time_str"]
        )
        # combine date+time → datetime
        df["dt"] = df.apply(
            lambda r: datetime.strptime(r["date_str"] + " " + r["time_str"],
                                        "%Y-%m-%d %H:%M:%S"),
            axis=1
        )
        # 篩選時間區段
        df_seg = df[(df["dt"] >= t_start) & (df["dt"] <= t_end)].copy()
        if df_seg.empty:
            # print(f"[{user_folder}] {plt_name} 無符合時間的點")
            continue

        # 計算 relative seconds
        t0 = df_seg["dt"].iloc[0]
        df_seg["seconds"] = df_seg["dt"].apply(lambda x: int((x - t0).total_seconds()))

        # 輸出 CSV
        out_name = f"{user_folder}_{out_idx:06d}.csv"
        out_path = Path(out_dir) / out_name
        df_seg[["lat","lon","seconds"]].to_csv(out_path, index=False)
        print(f"輸出：{out_path}  共 {len(df_seg)} 筆")
        out_idx += 1

def main():
    now_dir = os.path.dirname(os.path.abspath(__file__))
    DATA = os.path.join(now_dir, "Dataset", "Data") # 資料目錄
    OUT_DIR  = os.path.join(now_dir, "walk_traj") # 輸出目錄
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. 找出所有使用者資料夾
    user_folders = find_users_with_label(DATA)

    # 2. 逐個處理
    for uf in sorted(user_folders):
        process_user_folder(DATA, uf, OUT_DIR)

if __name__ == "__main__":
    main()
