import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    """
    batch: List[Tuple[gx, gy, length]]
    回傳：
      gx_pad, gy_pad   -> [B, L_max]
      pad_mask         -> [B, L_max]  True 表示「是 PAD」給 Transformer 用
    """
    gx_list, gy_list, lens = zip(*batch)
    gx_pad = pad_sequence(gx_list, batch_first=True, padding_value=0)
    gy_pad = pad_sequence(gy_list, batch_first=True, padding_value=0)
    # True = pad，False = valid
    lens = torch.tensor(lens)
    pad_mask = (torch.arange(gx_pad.size(1))[None, :].to(lens.device) >= lens[:, None])
    return gx_pad, gy_pad, pad_mask
