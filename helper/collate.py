# collate.py  —— 直接覆蓋 collate_fn 內容
import torch
from torch.nn.utils.rnn import pad_sequence
from helper.load_config import load_config

def collate_fn(batch, pad_val=-1):
    """
    batch: List[Tuple[gx, gy, t, L]]
    """
    PAD = load_config("config/config.yaml")["data"]["pad_idx"]
    pad_val = PAD

    gxs, gys, ts, _ = zip(*batch)  # <<<< 取出時間序列 t
    lens = [len(x) for x in gxs]
    max_len = max(lens)

    padded_gx = torch.full((len(batch), max_len), pad_val, dtype=torch.long)
    padded_gy = torch.full((len(batch), max_len), pad_val, dtype=torch.long)
    padded_t  = torch.zeros(len(batch), max_len, dtype=torch.float32)  # <<<< 新增
    pad_mask  = torch.ones (len(batch), max_len, dtype=torch.bool)     # True=PAD

    for i, (gx, gy, t) in enumerate(zip(gxs, gys, ts)):
        L = len(gx)
        padded_gx[i, :L] = gx
        padded_gy[i, :L] = gy
        padded_t[i,  :L] = t
        pad_mask[i, :L]  = False  # False=有效 token

    return padded_gx, padded_gy, padded_t, pad_mask
