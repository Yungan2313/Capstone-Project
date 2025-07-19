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

class TrajCSVDataset(Dataset):
    """
    假設每檔案一條軌跡 csv 檔將單一 csv → (gx, gy, length) tensor
    """
    def __init__(self, files: List[Path], max_len: int):
        self.files = files
        self.max_len = max_len

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        df = pd.read_csv(self.files[idx])
        gx, gy = latlon_to_grid(
            df["lat"].values,
            df["lon"].values,
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
        files = [f for f in self.data_dir.glob("*.csv") if f.name not in self.skip]
        random.shuffle(files)
        n = len(files)
        n_train = int(n * self.ratio[0])
        n_val   = int(n * self.ratio[1])
        self.train_ds = TrajCSVDataset(files[:n_train], self.max_len)
        self.val_ds   = TrajCSVDataset(files[n_train : n_train + n_val], self.max_len)
        self.test_ds  = TrajCSVDataset(files[n_train + n_val :], self.max_len)

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
