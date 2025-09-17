import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class TrajectoryDataset(Dataset):
    def __init__(self, root_folder):
        self.files = []
        for folder in os.listdir(root_folder):
            folder_path = os.path.join(root_folder, folder)
            if os.path.isdir(folder_path):
                for file in os.listdir(folder_path):
                    if file.endswith(".npz"):
                        self.files.append(os.path.join(folder_path, file))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        # 只取 lat/lon 兩欄，移除 time
        original = torch.tensor(data["original"][:, :2], dtype=torch.float32)
        compressed = torch.tensor(data["compressed"][:, :2], dtype=torch.float32)
        return original, compressed

def pad_collate_fn(batch):
    originals, compresseds = zip(*batch)

    orig_lens = [x.size(0) for x in originals]
    comp_lens = [x.size(0) for x in compresseds]

    max_len_orig = max(orig_lens)
    max_len_comp = max(comp_lens)

    batch_size = len(batch)

    padded_original = torch.zeros(batch_size, max_len_orig, 2)
    padded_compressed = torch.zeros(batch_size, max_len_comp, 2)
    original_mask = torch.zeros(batch_size, max_len_orig, dtype=torch.bool)
    compressed_mask = torch.zeros(batch_size, max_len_comp, dtype=torch.bool)

    for i in range(batch_size):
        padded_original[i, :orig_lens[i]] = originals[i]
        padded_compressed[i, :comp_lens[i]] = compresseds[i]
        original_mask[i, :orig_lens[i]] = 1
        compressed_mask[i, :comp_lens[i]] = 1

    return padded_original, padded_compressed, original_mask, compressed_mask

# 使用範例（可放在 train.py 中測試）
if __name__ == "__main__":
    dataset = TrajectoryDataset("Compressed_Data")
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=pad_collate_fn
    )

    for src, tgt, src_mask, tgt_mask in dataloader:
        print("src:", src.shape)
        print("tgt:", tgt.shape)
        print("src_mask:", src_mask.shape)
        print("tgt_mask:", tgt_mask.shape)
        torch.set_printoptions(precision=10)
        print("src:", src[0])
        break