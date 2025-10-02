
# -*- coding: utf-8 -*-
"""
Compute PED & SED for each simplified trajectory produced by evaluate(),
assuming each subfolder under session_dir is named by the original CSV's stem,
and the kept file is {stem}_kept.csv with columns idx,lat,lon.

Outputs:
- Per-sample:  session_dir/{stem}/errors.txt
- Session avg: session_dir/eval_summary.txt
python eval_matric.py --session_dir ./result/eval/2025xxxx-xxxx --datasets_dir ./data/datasets
python -m eval_matric --session_dir ./result/eval/20250923-1029 --datasets_dir ./data/datasets
python -m eval_matric --session_dir ./result/eval/20250921-192525_DP --datasets_dir ./data/datasets
python -m eval_matric --session_dir ./result/eval/20250921-192525_Error-Search --datasets_dir ./data/datasets
python -m eval_matric --session_dir ./result/eval/20250921-192525_TDTR --datasets_dir ./data/datasets
"""
import argparse, os, math, json
from pathlib import Path
import numpy as np
import pandas as pd

LAT_CANDS = ["lat", "latitude", "Lat", "Latitude"]
LON_CANDS = ["lon", "lng", "longitude", "Lon", "Lng", "Longitude"]
T_CANDS   = ["time", "timestamp", "t", "Time", "Timestamp"]

def _deg2rad(x): return x * math.pi / 180.0

def latlon_to_xy_m(lat, lon, lat0=None):
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    if lat0 is None:
        lat0 = float(np.mean(lat))
    R = 6371000.0
    x = R * math.cos(_deg2rad(lat0)) * _deg2rad(lon - np.mean(lon))
    y = R * _deg2rad(lat - np.mean(lat))
    return x, y

def read_original_csv(path: Path):
    df = pd.read_csv(path)
    lat_col = next((c for c in LAT_CANDS if c in df.columns), None)
    lon_col = next((c for c in LON_CANDS if c in df.columns), None)
    t_col   = next((c for c in T_CANDS   if c in df.columns), None)
    if lat_col is None or lon_col is None:
        raise ValueError(f"[{path.name}] cannot find lat/lon columns; got {list(df.columns)}")
    lat = df[lat_col].to_numpy()
    lon = df[lon_col].to_numpy()
    if t_col is None:
        t = np.arange(len(df), dtype=float)
    else:
        t = pd.to_datetime(df[t_col], errors="ignore")
        if np.issubdtype(t.dtype, np.datetime64):
            t = (t - t.iloc[0]).dt.total_seconds().to_numpy(dtype=float)
        else:
            t = pd.to_numeric(df[t_col], errors="coerce").to_numpy(dtype=float)
            t = t - np.nanmin(t)
    return lat, lon, t

def read_kept_csv(path: Path):
    df = pd.read_csv(path)
    if not {"idx","lat","lon"}.issubset(df.columns):
        raise ValueError(f"[{path.name}] expect columns idx,lat,lon; got {list(df.columns)}")
    return df

def ped_point_to_segment(px, py, x1, y1, x2, y2):
    vx, vy = x2 - x1, y2 - y1
    wx, wy = px - x1, py - y1
    L2 = vx*vx + vy*vy
    if L2 == 0.0:
        return math.hypot(wx, wy)
    t = max(0.0, min(1.0, (wx*vx + wy*vy) / L2))
    projx = x1 + t * vx
    projy = y1 + t * vy
    return math.hypot(px - projx, py - projy)

def sed_point_to_segment(px, py, tx, x1, y1, t1, x2, y2, t2):
    if t2 == t1:
        return ped_point_to_segment(px, py, x1, y1, x2, y2)
    alpha = (tx - t1) / (t2 - t1)
    alpha = min(1.0, max(0.0, alpha))
    sx = x1 + alpha * (x2 - x1)
    sy = y1 + alpha * (y2 - y1)
    return math.hypot(px - sx, py - sy)

def compute_errors_for_one(original_lat, original_lon, original_t, kept_idx):
    x, y = latlon_to_xy_m(original_lat, original_lon)
    t = original_t.astype(float)
    kept_idx = np.asarray(kept_idx, dtype=int)
    kept_idx = np.unique(kept_idx)
    kept_idx.sort()
    ped_list, sed_list = [], []
    for i in range(len(kept_idx) - 1):
        s, e = int(kept_idx[i]), int(kept_idx[i+1])
        if e <= s: 
            continue
        seg_disc = np.arange(s+1, e, dtype=int)
        if seg_disc.size == 0:
            continue
        x1, y1, t1 = x[s], y[s], t[s]
        x2, y2, t2 = x[e], y[e], t[e]
        for m in seg_disc:
            ped = ped_point_to_segment(x[m], y[m], x1, y1, x2, y2)
            sed = sed_point_to_segment(x[m], y[m], t[m], x1, y1, t1, x2, y2, t2)
            ped_list.append(ped); sed_list.append(sed)
    ped_arr = np.array(ped_list, dtype=float)
    sed_arr = np.array(sed_list, dtype=float)
    return {
        "n_discard": int(ped_arr.size),
        "ped_mean": float(ped_arr.mean()) if ped_arr.size else 0.0,
        "ped_max":  float(ped_arr.max())  if ped_arr.size else 0.0,
        "ped_med":  float(np.median(ped_arr)) if ped_arr.size else 0.0,
        "sed_mean": float(sed_arr.mean()) if sed_arr.size else 0.0,
        "sed_max":  float(sed_arr.max())  if sed_arr.size else 0.0,
        "sed_med":  float(np.median(sed_arr)) if sed_arr.size else 0.0,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session_dir", required=True, help="e.g. ./result/eval/2025xxxx-xxxx")
    ap.add_argument("--datasets_dir", default="./data/datasets", help="original CSV folder")
    ap.add_argument("--kept_suffix", default="_kept.csv", help="filename suffix of kept file")
    args = ap.parse_args()

    session_dir = Path(args.session_dir)
    data_dir = Path(args.datasets_dir)

    subdirs = [p for p in session_dir.iterdir() if p.is_dir()]
    if not subdirs:
        raise SystemExit(f"No sample folders found under {session_dir}")

    agg_ped, agg_sed, agg_cnt = 0.0, 0.0, 0
    missing = 0

    for d in sorted(subdirs):
        stem = d.name
        kept_csv = d / f"{stem}{args.kept_suffix}"
        orig_csv = data_dir / f"{stem}.csv"

        if not kept_csv.exists():
            candidates = list(d.glob("*_kept.csv"))
            if candidates:
                kept_csv = candidates[0]
        if not kept_csv.exists():
            missing += 1
            continue
        if not orig_csv.exists():
            alts = list(data_dir.glob(f"{stem}.*"))
            if not alts:
                with open(d / "errors.txt", "a", encoding="utf-8") as f:
                    f.write(f"[warn] original not found: {orig_csv}\n")
                missing += 1
                continue
            orig_csv = alts[0]

        kept_df = read_kept_csv(kept_csv)
        kept_idx = kept_df["idx"].to_numpy().astype(int)

        lat, lon, t = read_original_csv(orig_csv)
        L = len(lat)
        if kept_idx.min() >= 1 and (0 not in kept_idx) and kept_idx.max() - 1 < L:
            kept_idx = kept_idx - 1
        kept_idx = kept_idx[(kept_idx >= 0) & (kept_idx < L)]
        if 0 not in kept_idx:
            kept_idx = np.insert(kept_idx, 0, 0)
        if (L - 1) not in kept_idx:
            kept_idx = np.append(kept_idx, L - 1)
        kept_idx = np.unique(kept_idx)

        stats = compute_errors_for_one(lat, lon, t, kept_idx)

        with open(d / "errors.txt", "w", encoding="utf-8") as f:
            f.write(f"file: {orig_csv.name}\n")
            f.write(f"discarded_points: {stats['n_discard']}\n")
            f.write(f"PED_mean(m): {stats['ped_mean']:.6f}\n")
            f.write(f"PED_median(m): {stats['ped_med']:.6f}\n")
            f.write(f"PED_max(m): {stats['ped_max']:.6f}\n")
            f.write(f"SED_mean(m): {stats['sed_mean']:.6f}\n")
            f.write(f"SED_median(m): {stats['sed_med']:.6f}\n")
            f.write(f"SED_max(m): {stats['sed_max']:.6f}\n")

        if stats["n_discard"] > 0:
            agg_ped += stats["ped_mean"]
            agg_sed += stats["sed_mean"]
            agg_cnt += 1

    with open(session_dir / "eval_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"samples_total: {len(subdirs)}\n")
        f.write(f"samples_scored: {agg_cnt}\n")
        f.write(f"samples_missing_or_empty: {len(subdirs) - agg_cnt}\n")
        mean_ped = agg_ped / agg_cnt if agg_cnt else 0.0
        mean_sed = agg_sed / agg_cnt if agg_cnt else 0.0
        f.write(f"AVERAGE_PED_mean(m): {mean_ped:.6f}\n")
        f.write(f"AVERAGE_SED_mean(m): {mean_sed:.6f}\n")

    print(f"[done] wrote per-sample errors and eval_summary.txt in {session_dir}")

if __name__ == "__main__":
    main()
