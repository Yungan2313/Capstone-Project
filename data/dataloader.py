from __future__ import annotations
from pathlib import Path
from typing import Tuple, Dict, List
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from helper.grid_utils import latlon_to_grid
from helper.collate import collate_fn
from helper.load_config import load_config

class TrajCSVDataset(Dataset):
    """
    假設每檔案一條軌跡 csv 檔將單一 csv → (gx, gy, length) tensor
    """
    def __init__(self, files: List[Path], max_len: int, win):
        # 原始檔路徑清單
        self.files = files
        # 讓 evaluate.sample_name_from_dataset() 能直接抓到「原始檔名的 stem」
        # evaluate 會依序檢查 dataset 的屬性: name / filename / file / path / id / metas
        # 這裡同時提供 name / filename / path，避免找不到而退回 sample_00000…
        self.name = [p.stem for p in files]           # 例如 "073_000028"
        self.filename = [p.name for p in files]       # 例如 "073_000028.csv"
        self.path = [str(p) for p in files]           # 完整路徑字串
        self.max_len = max_len
        self.win = win

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        df = pd.read_csv(self.files[idx])
        gx, gy = latlon_to_grid(
            df["lat"].values,
            df["lon"].values,
            win=self.win
        )
        if len(gx) > self.max_len:        # truncate，但 **不 pad**
            gx, gy = gx[: self.max_len], gy[: self.max_len]
        return torch.LongTensor(gx), torch.LongTensor(gy), len(gx)


class TrajDataModule:
    """
    高階包裝：一次搞定 train/val/test split + DataLoader
    """
    def __init__(
        self,
        data_dir: str,
        split_ratio: Tuple[float, float, float],
        max_len: int,
        batch_size: int,
        num_workers: int = 4,
        skip_list_path: str | None = None,
        seed: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.ratio = split_ratio
        self.max_len = max_len
        self.batch = batch_size
        self.num_workers = num_workers
        self.skip = (
            set(Path(skip_list_path).read_text().split())
            if skip_list_path and Path(skip_list_path).exists()
            else set()
        )
        random.seed(seed)

        self._prepare()

    # ------------------------------------------------------------------
    def _prepare(self):
        cfg = load_config("config/config.yaml")
        w = cfg["data"]["fixed_window"]
        files = [f for f in self.data_dir.glob("*.csv") if f.name not in self.skip]
        random.shuffle(files)
        n = len(files)
        n_train = int(n * self.ratio[0])
        n_val   = int(n * self.ratio[1])
        self.train_ds = TrajCSVDataset(files[:n_train], self.max_len, w)
        self.val_ds   = TrajCSVDataset(files[n_train : n_train + n_val], self.max_len, w)
        self.test_ds  = TrajCSVDataset(files[n_train + n_val :], self.max_len, w)

    # ------------------------------------------------------------------
    def loaders(self) -> Dict[str, DataLoader]:
        return dict(
            train=DataLoader(
                self.train_ds, self.batch, shuffle=True, collate_fn=collate_fn, num_workers=self.num_workers
            ),
            val=DataLoader(
                self.val_ds, self.batch, shuffle=False, collate_fn=collate_fn, num_workers=self.num_workers
            ),
            test=DataLoader(
                self.test_ds, self.batch, shuffle=False, collate_fn=collate_fn, num_workers=self.num_workers
            ),
        )

if __name__ == "__main__":
    from pathlib import Path

    save_dir = "./data"
    data_dir = "./data/datasets"
    out_dir = Path(save_dir) / "check"
    out_dir.mkdir(parents=True, exist_ok=True)

    dm = TrajDataModule(
        data_dir=data_dir,
        split_ratio=(0.7, 0.2, 0.1),
        max_len=200,
        batch_size=32,
        num_workers=0,
        seed=42,
    )

    # 取出檔名（或想保留子資料夾層級就改成相對路徑）
    def to_relative_names(paths):
        base = Path(data_dir).resolve()
        return [str(Path(p).resolve().relative_to(base)) for p in paths]

    splits = {
        "train": to_relative_names(dm.train_ds.files),
        "val":   to_relative_names(dm.val_ds.files),
        "test":  to_relative_names(dm.test_ds.files),
    }

    for name, names in splits.items():
        (out_dir / f"{name}.txt").write_text("\n".join(names), encoding="utf-8")

    total = sum(len(v) for v in splits.values())
    print(f"[INFO] Saved lists to {out_dir} | total={total} "
          f"(train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])})")
