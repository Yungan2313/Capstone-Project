import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch, pad_val=-1):
    """
    batch: List[Tuple[gx, gy]] where gx,gy = 1-D LongTensor
    """
    gxs, gys, _ = zip(*batch)
    lens = [len(x) for x in gxs]
    max_len = max(lens)

    padded_gx = torch.full((len(batch), max_len), pad_val, dtype=torch.long)
    padded_gy = torch.full((len(batch), max_len), pad_val, dtype=torch.long)
    pad_mask   = torch.ones (len(batch), max_len,  dtype=torch.bool)   # True=PAD

    for i, (gx, gy) in enumerate(zip(gxs, gys)):
        L = len(gx)
        padded_gx[i, :L] = gx
        padded_gy[i, :L] = gy
        pad_mask[i, :L]  = False       # False=有效 token

    return padded_gx, padded_gy, pad_mask

