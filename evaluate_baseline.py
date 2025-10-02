
# evaluate_baseline.py
# Ratio-driven evaluation for three algorithms:
#   - TDTR (SED) by ratio
#   - DP (min-number, SED) by ratio
#   - Error-Search (fixed K) as baseline
#
# Output structure:
#   ./result/eval/<YYYYmmdd-HHMMSS>_TDTR/<stem>/<stem>_kept.csv
#   ./result/eval/<YYYYmmdd-HHMMSS>_DP/<stem>/<stem>_kept.csv
#   ./result/eval/<YYYYmmdd-HHMMSS>_Error-Search/<stem>/<stem>_kept.csv

import argparse, time
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
import torch
from data.dataloader import TrajDataModule
from pathlib import Path


# Import the new ratio-driven APIs
from baseline.traj_simplify import tdtr_by_ratio, dp_by_ratio, error_search, Point

LAT_CANDS = ["lat", "latitude", "Lat", "Latitude"]
LON_CANDS = ["lon", "lng", "longitude", "Lon", "Lng", "Longitude"]
T_CANDS   = ["time", "timestamp", "t", "Time", "Timestamp"]

def files_from_dm_test_split(ckpt_path: str):
    """
    用 evaluate.py 相同的 cfg 建立 TrajDataModule，回傳 test split 的完整檔路徑 list 與資料夾路徑。
    """
    if not ckpt_path:
        return None, None

    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("cfg")
    if cfg is None:
        print("[WARN] ckpt 裡沒有 cfg，無法用 TrajDataModule 產生 test split，將 fallback 到原本邏輯。")
        return None, None

    data_dir       = Path(cfg["data"]["dataset_dir"])
    skip_list_path = cfg["data"].get("skip_list", None)
    split_ratio    = tuple(cfg["training"]["split_ratio"])
    batch_size     = 1
    num_workers    = cfg["training"].get("num_workers", 0)
    max_len        = cfg["model"]["max_seq_len"]

    dm = TrajDataModule(
        data_dir       = data_dir,
        split_ratio    = split_ratio,
        max_len        = max_len,
        batch_size     = batch_size,
        skip_list_path = skip_list_path,
        num_workers    = num_workers,
    )
    loaders = dm.loaders()
    test_loader = loaders["test"]
    test_ds = test_loader.dataset

    files = getattr(test_ds, "files", None)
    if not files:
        raise RuntimeError("test dataset 沒有 .files 屬性，請確認 TrajCSVDataset 的實作。")

    files = [str(Path(p)) for p in files]
    return files, data_dir

def read_original_csv(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    lat_col = next((c for c in LAT_CANDS if c in df.columns), None)
    lon_col = next((c for c in LON_CANDS if c in df.columns), None)
    t_col   = next((c for c in T_CANDS   if c in df.columns), None)
    if lat_col is None or lon_col is None:
        raise ValueError(f"[{path.name}] cannot find lat/lon columns; got {list(df.columns)}")
    lat = df[lat_col].to_numpy(dtype=float)
    lon = df[lon_col].to_numpy(dtype=float)
    if t_col is None:
        t = np.arange(len(df), dtype=float)
    else:
        # Allow numeric or datetime
        t_ser = pd.to_datetime(df[t_col], errors="ignore")
        if np.issubdtype(t_ser.dtype, np.datetime64):
            t = (t_ser - t_ser.iloc[0]).dt.total_seconds().to_numpy(dtype=float)
        else:
            t = pd.to_numeric(df[t_col], errors="coerce").to_numpy(dtype=float)
            t = t - np.nanmin(t)
    # Make time strictly increasing for stable SED interpolation
    t = np.maximum.accumulate(t.astype(float))
    t = t + np.arange(len(t), dtype=float)*1e-9
    return lat, lon, t

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def write_kept_csv(out_dir: Path, stem: str, kept_idx: List[int], lat: np.ndarray, lon: np.ndarray):
    kept_idx = sorted(set(int(i) for i in kept_idx))
    kept_df = pd.DataFrame({
        "idx": kept_idx,
        "lat": [float(lat[i]) for i in kept_idx],
        "lon": [float(lon[i]) for i in kept_idx],
    })
    ensure_dir(out_dir)
    kept_df.to_csv(out_dir / f"{stem}_kept.csv", index=False)

def list_csvs(datasets_dir: Path, filelist: Path | None) -> List[Path]:
    if filelist and filelist.exists():
        stems = [line.strip() for line in filelist.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [datasets_dir / (s if s.endswith(".csv") else f"{s}.csv") for s in stems]
    return sorted(datasets_dir.glob("*.csv"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets_dir", type=str, required=True,
                    help="folder of raw trajectory CSVs (e.g., 010_000004.csv)")
    ap.add_argument("--filelist", type=str, default=None,
                    help="optional: a txt file that lists filenames or stems to process (one per line)")
    ap.add_argument("--out_root", type=str, default="./result/eval",
                    help="output root directory")
    ap.add_argument("--keep_ratio", type=float, default=0.2,
                    help="ratio of points to keep (e.g., 0.2 means keep 20% of points)")
    # 修正這行：原本你的程式誤用 p.add_argument，這裡要用 ap.add_argument
    ap.add_argument("--ckpt", type=str, default=None,
                    help="(推薦) 與 evaluate.py 相同的 checkpoint；會讀其中的 cfg 來產生 TrajDataModule 的 test split")
    args = ap.parse_args()

    datasets_dir = Path(args.datasets_dir)
    filelist = Path(args.filelist) if args.filelist else None
    out_root = Path(args.out_root)
    ts = time.strftime("%Y%m%d-%H%M%S")

    session_dirs = {
        "TDTR": out_root / f"{ts}_TDTR",
        "DP": out_root / f"{ts}_DP",
        "Error-Search": out_root / f"{ts}_Error-Search",
    }
    for p in session_dirs.values():
        ensure_dir(p)

    # ===== 檔案清單決策順序 =====
    # 1) 若提供 --filelist：完全照檔單
    if filelist:
        csv_paths = list_csvs(datasets_dir, filelist)

    # 2) 否則，若提供 --ckpt：用 TrajDataModule 的 test split
    else:
        csv_paths, data_dir_from_dm = files_from_dm_test_split(args.ckpt)
        # 兼容：若 DM 流程失敗/未提供 ckpt，退回舊邏輯（整個資料夾）
        if not csv_paths:
            csv_paths = list_csvs(datasets_dir, None)
            data_dir_from_dm = None

    if not csv_paths:
        print(f"[Warn] No CSV found under {datasets_dir}")
        return

    # （可選）把這次實際要跑的 test 檔案清單存出去，方便與 evaluate.py 對比
    ensure_dir(out_root)
    (out_root / f"{ts}_baseline_test_list.txt").write_text(
        "\n".join(str(p) for p in csv_paths), encoding="utf-8"
    )
    if "data_dir_from_dm" in locals() and data_dir_from_dm:
        (out_root / f"{ts}_data_dir_used.txt").write_text(str(data_dir_from_dm), encoding="utf-8")
        print(f"[INFO] test split 來自 TrajDataModule | dataset_dir={data_dir_from_dm}")

    for csv_path in csv_paths:
        stem = Path(csv_path).stem
        lat, lon, t = read_original_csv(Path(csv_path))
        n = len(lat)
        if n < 2:
            print(f"[Skip] {stem}: too few points")
            continue
        target_kept = max(2, int(round(n * args.keep_ratio)))

        # Build points
        points: List[Point] = [(float(lat[i]), float(lon[i]), float(t[i])) for i in range(n)]

        # --- TD-TR (ratio) ---
        kept_td = tdtr_by_ratio(points, args.keep_ratio)
        write_kept_csv(session_dirs["TDTR"] / stem, stem, kept_td, lat, lon)

        # --- DP (ratio) ---
        kept_dp = dp_by_ratio(points, args.keep_ratio)
        write_kept_csv(session_dirs["DP"] / stem, stem, kept_dp, lat, lon)

        # --- Error-Search baseline: K = target_kept - 1 segments ---
        k_segments = max(1, target_kept - 1)
        eps_star, kept_es = error_search(points, k_segments)
        kept_es = sorted(set(kept_es) | {0, n - 1})
        write_kept_csv(session_dirs["Error-Search"] / stem, stem, kept_es, lat, lon)

        print(f"[OK] {stem} -> N={n}, target_keep={target_kept} | "
              f"TD-TR:{len(kept_td)}, DP:{len(kept_dp)}, ES:{len(kept_es)} (eps*={eps_star:.6f})")
        
        # print(f"[OK] {stem} -> N={n}, target_keep={target_kept} | "
        #       f"ES:{len(kept_es)} (eps*={eps_star:.6f})")
        
        # print(f"[OK] {stem} -> N={n}, target_keep={target_kept} | "
        #       f"TD-TR:{len(kept_td)}, DP:{len(kept_dp)}")
        

    print("\nDone.")
    for name, p in session_dirs.items():
        print(f"  {name}: {p}")


if __name__ == "__main__":
    """python -m evaluate_baseline \
    --datasets_dir ./data/datasets \
    --out_root ./result/eval \
    --keep_ratio 0.2 \
    --ckpt ./checkpoint/best.pt

    python -m evaluate_baseline --datasets_dir ./data/datasets --ckpt ./checkpoint/20250918-181249_v3.1_mse_withoutv2/best.pt --out_root ./result/eval --keep_ratio 0.2
    """
    main()
