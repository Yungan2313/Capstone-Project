import numpy as np

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000  # 地球半徑（公尺）
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = phi2 - phi1
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def pad_to_length(arr, target_len=5):
    if len(arr) < target_len:
        last = arr[-1:]
        padding = np.repeat(last, target_len - len(arr), axis=0)
        return np.concatenate([arr, padding], axis=0)
    return arr

def compute_ade_fde(pred, target):
    # 確保 pred/target 至少有 5 點
    pred = pad_to_length(pred, 5)
    target = pad_to_length(target, 5)

    distances = haversine_distance(pred[:,1], pred[:,0], target[:,1], target[:,0])
    ade = distances.mean()
    fde = distances[-1]
    return ade, fde

'''
# calculate.py
import numpy as np

# Haversine distance (km)
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000  # 地球半徑（公尺）
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = phi2 - phi1
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c  # 回傳公尺

# 計算 ADE & FDE（傳入 shape = (N, 2) 的經緯度陣列）
def compute_ade_fde(pred, target):
    distances = haversine_distance(pred[:,1], pred[:,0], target[:,1], target[:,0])
    ade = distances.mean()
    fde = distances[-1]
    return ade, fde
    '''