"""
compare_models_ped_sed.py

Aggregate multiple evaluation *sessions* (each corresponds to one model's evaluate output
that has per-sample errors.txt from eval_matric.py), and produce a table:

Columns:
- file      : sample stem (original CSV filename without extension)
- best_model: which model is best for this sample under the chosen scoring rule
- For each model M, two columns: M_PED_mean, M_SED_mean

Scoring (lower is better):
- --pick-by sed        : compare by SED_mean only (default)
- --pick-by ped        : compare by PED_mean only
- --pick-by sum        : compare by (SED_mean + PED_mean)
- --pick-by wavg       : compare by w_sed * SED_mean + w_ped * PED_mean (set weights)

Usage examples:
python compare_models_ped_sed.py \
  --out ./result/eval/compare.csv \
  --model Baseline=./result/eval/2025-09-10-2230 \
  --model NewLoss=./result/eval/2025-09-12-0042 \
  --model DiffTopK=./result/eval/2025-09-13-1801 \
  --pick-by sed

python compare_models_ped_sed.py \
  --out ./result/eval/compare.csv \
  --model ./result/eval/2025-09-10-2230 \
  --model ./result/eval/2025-09-12-0042 \
  --model ./result/eval/2025-09-13-1801 \
  --pick-by wavg --w-sed 1.0 --w-ped 0.5

The script also writes a 'compare_summary.txt' next to the CSV with overall stats per model.
python -m matric_statsic --out ./result/compare.csv --model v1=./result/eval/20250917-0052 --model v3=./result/eval/20250916-2346_v3 --model v3nov2=./result/eval/20250917-1606 --pick-by sed

"""
import argparse, re, math
from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd
import numpy as np

ERR_KEYS = ("PED_mean", "SED_mean")

def parse_errors_txt(err_path: Path):
    """Return dict with PED_mean and SED_mean (float)."""
    text = err_path.read_text(encoding="utf-8", errors="ignore")
    kv = {}
    # lines like: "PED_mean(m): 3.241905"
    for key in ERR_KEYS:
        m = re.search(rf"{key}\(m\):\s*([0-9.+-eE]+)", text)
        if m:
            kv[key] = float(m.group(1))
    return kv if all(k in kv for k in ERR_KEYS) else None

def load_session(session_dir: Path):
    """Return mapping: sample_stem -> {'PED_mean': x, 'SED_mean': y} for this session."""
    result = {}
    if not Path(session_dir).exists():
        return result
    for sub in Path(session_dir).iterdir():
        if not sub.is_dir(): 
            continue
        err = sub / "errors.txt"
        if not err.exists():
            continue
        vals = parse_errors_txt(err)
        if not vals:
            continue
        stem = sub.name  # folder is sample stem per your evaluate change
        result[stem] = vals
    return result

def build_table(model_sessions, pick_by="sed", w_sed=1.0, w_ped=1.0):
    """
    model_sessions: list of (model_name, session_dir Path)
    Returns: (df, summary_txt)
    """
    # load
    data_by_model = {}
    all_samples = set()
    for name, sdir in model_sessions:
        data = load_session(Path(sdir))
        data_by_model[name] = data
        all_samples.update(data.keys())

    # rows
    rows = []
    for sample in sorted(all_samples):
        # collect metrics per model
        per_model = {}
        for name in data_by_model:
            vals = data_by_model[name].get(sample)
            if vals is not None:
                per_model[name] = (vals["PED_mean"], vals["SED_mean"])  # (PED, SED)
            else:
                per_model[name] = (math.nan, math.nan)

        # decide best model
        best_name, best_score = None, math.inf
        for name, (ped, sed) in per_model.items():
            if math.isnan(ped) or math.isnan(sed):
                continue
            if pick_by == "sed":
                score = sed
            elif pick_by == "ped":
                score = ped
            elif pick_by == "sum":
                score = ped + sed
            elif pick_by == "wavg":
                score = w_sed * sed + w_ped * ped
            else:
                raise ValueError(f"Unknown pick_by: {pick_by}")
            if score < best_score:
                best_score = score
                best_name = name

        row = {"file": sample, "best_model": best_name if best_name is not None else ""}
        # add metrics columns
        for name in data_by_model.keys():
            ped, sed = per_model[name]
            row[f"{name}_PED_mean"] = ped
            row[f"{name}_SED_mean"] = sed
        rows.append(row)

    df = pd.DataFrame(rows)

    # build overall summary
    lines = []
    lines.append(f"pick_by={pick_by}, w_sed={w_sed}, w_ped={w_ped}\n")
    # count wins
    win_counts = Counter(df["best_model"].dropna())
    total_present = len(df)
    for name in data_by_model.keys():
        wins = win_counts.get(name, 0)
        lines.append(f"{name}: wins={wins}/{total_present}")
    lines.append("")

    # overall means per model (across samples where the model has values)
    for name, data in data_by_model.items():
        ped_vals = [v["PED_mean"] for v in data.values() if "PED_mean" in v]
        sed_vals = [v["SED_mean"] for v in data.values() if "SED_mean" in v]
        ped_mean = float(np.mean(ped_vals)) if ped_vals else float("nan")
        sed_mean = float(np.mean(sed_vals)) if sed_vals else float("nan")
        lines.append(f"{name}: overall PED_mean={ped_mean:.6f}, SED_mean={sed_mean:.6f}")

    summary_txt = "\n".join(lines)
    return df, summary_txt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument("--pick-by", default="sed", choices=["sed","ped","sum","wavg"])
    ap.add_argument("--w-sed", type=float, default=1.0, help="weight for SED_mean when pick-by=wavg")
    ap.add_argument("--w-ped", type=float, default=1.0, help="weight for PED_mean when pick-by=wavg")
    ap.add_argument("--model", action="append", required=True,
                    help="Model mapping as Name=SessionDir OR just SessionDir; can repeat")
    args = ap.parse_args()

    model_sessions = []
    for spec in args.model:
        if "=" in spec:
            name, path = spec.split("=", 1)
        else:
            p = Path(spec.rstrip("/")).name  # use dir name as model name
            name, path = p, spec
        model_sessions.append((name, Path(path)))

    df, summary_txt = build_table(model_sessions, args.pick_by, args.w_sed, args.w_ped)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")
    # also write a sidecar summary
    sum_path = out_path.with_suffix(".summary.txt")
    sum_path.write_text(summary_txt, encoding="utf-8")
    print(f"[done] wrote table to {out_path}")
    print(f"[done] wrote summary to {sum_path}")

if __name__ == "__main__":
    main()
