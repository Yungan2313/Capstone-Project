from typing import List, Tuple, Literal, Optional
import math

Point = Tuple[float, float, float]  # (lat, lon, time)

# ---------- 基本幫手：SED 與段內最大 SED ----------
def _sed(s: Point, m: Point, e: Point) -> float:
    """Synchronous Euclidean Distance (time-ratio)."""
    (slat, slon, st), (mlat, mlon, mt), (elat, elon, et) = s, m, e
    den = (et - st)
    tr = 1.0 if den == 0 else (mt - st) / den
    ilat = slat + (elat - slat) * tr
    ilon = slon + (elon - slon) * tr
    return math.hypot(ilat - mlat, ilon - mlon)

def _segment_max_sed(points: List[Point], i: int, j: int) -> float:
    """i<j，回傳 i..j 由單一直線近似時，段內最大 SED。"""
    if j - i <= 1:
        return 0.0
    s, e = points[i], points[j]
    mx = 0.0
    for k in range(i + 1, j):
        mx = max(mx, _sed(s, points[k], e))
    return mx


# ---------- 1) TD-TR：Top-Down 時間比率（SED）----------
def td_tr(points: List[Point], epsilon: float) -> List[int]:
    """
    Top-Down SED 簡化。回傳保留點的索引(遞增)。
    對應 C++: TD-TR.cpp 的 SED 與遞迴分割（:contentReference[oaicite:3]{index=3}）。
    """
    n = len(points)
    if n == 0:
        return []
    kept = []

    def _recur(a: int, b: int):
        if b - a <= 1:
            kept.append(a)
            kept.append(b)
            return
        s, e = points[a], points[b]
        dmax, idx = -1.0, -1
        for i in range(a + 1, b):
            d = _sed(s, points[i], e)
            if d > dmax:
                dmax, idx = d, i
        if dmax > epsilon:
            _recur(a, idx)
            _recur(idx, b)
        else:
            kept.append(a)
            kept.append(b)

    _recur(0, n - 1)
    kept = sorted(set(kept))
    return kept


# ---------- 2) DP：兩種經典問題 ----------
# (A) Min-Number with fixed epsilon：給定 ε，最少段數（Imai-Iri 形式）
#     參考 C++ DP.cpp：appr_table + DP_Number（:contentReference[oaicite:4]{index=4}）
# (B) Min-Error with fixed K：給定 K，最小化最大段誤差（:contentReference[oaicite:5]{index=5}）

def _appr_table(points: List[Point], eps: float) -> List[List[bool]]:
    """O(n^3) 建立可近似表 appr[i][j]（i<=j），是否能用一段把 i..j 近似到 SED<=eps。"""
    n = len(points)
    appr = [[False]*n for _ in range(n)]
    for i in range(n):
        appr[i][i] = True
    for i in range(n):
        for j in range(i+1, n):
            appr[i][j] = (_segment_max_sed(points, i, j) <= eps)
    return appr

def _dp_min_number(points: List[Point], eps: float) -> Tuple[int, List[int]]:
    """
    回傳(最少段數, 索引列表)。重建路徑輸出保留點索引。
    """
    n = len(points)
    if n == 0:
        return (0, [])
    appr = _appr_table(points, eps)  # 依據 C++ 的 calc_appr_table1（:contentReference[oaicite:6]{index=6}）

    # PN[i] = 從 i 到 n-1 的最少段數
    PN = [math.inf]*n
    nxt = [-1]*n

    PN[n-1] = 1
    nxt[n-1] = n-1
    for i in range(n-2, -1, -1):
        # 直接一段到尾
        if appr[i][n-1]:
            PN[i] = 1
            nxt[i] = n-1
            continue
        # 枚舉下一個端點 j
        best = math.inf
        best_j = -1
        for j in range(i+1, n):
            if appr[i][j] and PN[j] + 0 < best:
                best = 1 + (PN[j] if j < n else 0)
                best_j = j
        PN[i] = best
        nxt[i] = best_j

    # 重建索引
    kept = []
    i = 0
    while i != -1:
        kept.append(i)
        j = nxt[i]
        if j == -1 or j == i:
            break
        i = j
    if kept[-1] != n-1:
        kept.append(n-1)
    return (int(PN[0]), kept)

def _dp_min_error(points: List[Point], k: int) -> Tuple[float, List[int]]:
    """
    給定段數 k，最小化最大段誤差（max-SED）。回傳(最小最大誤差 epsilon*, 索引列表)。
    參考 C++ min_error.cpp 的 DP_Error 思路（:contentReference[oaicite:7]{index=7}）。
    """
    n = len(points)
    if n == 0:
        return (0.0, [])
    # 預先算每個 (i,j) 的誤差（最大 SED）
    err = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            err[i][j] = _segment_max_sed(points, i, j)

    # E[i][s] = 從 i..n-1 用 s 段時的最小「最大段誤差」
    E = [[math.inf]*(k+1) for _ in range(n)]
    prv = [[-1]*(k+1) for _ in range(n)]

    # s=1：只能一段到底
    for i in range(n-1):
        E[i][1] = err[i][n-1]
    E[n-1][1] = 0.0

    for s in range(2, k+1):
        for i in range(n-1):
            best, best_h = math.inf, -1
            # i..h 一段，h.. 用 s-1 段
            for h in range(i+1, n):
                val = max(err[i][h], E[h][s-1])
                if val < best:
                    best, best_h = val, h
            E[i][s] = best
            prv[i][s] = best_h
        E[n-1][s] = 0.0

    # 回溯
    eps_star = E[0][k]
    kept = [0]
    i, s = 0, k
    while s > 1:
        h = prv[i][s]
        kept.append(h)
        i, s = h, s-1
    if kept[-1] != n-1:
        kept.append(n-1)
    return (eps_star, kept)

def dp(points: List[Point],
       objective: Literal["min_number", "min_error"],
       epsilon: Optional[float] = None,
       k: Optional[int] = None) -> Tuple[float, List[int]]:
    """
    通用 DP 入口：
      - objective="min_number" 需給 epsilon，回傳(段數, 索引)；第一個回傳值以 float 形態但其實是整數。
      - objective="min_error"  需給 k，回傳(最小最大誤差, 索引)。
    """
    if objective == "min_number":
        assert epsilon is not None, "min_number 需提供 epsilon"
        segs, kept = _dp_min_number(points, epsilon)
        return float(segs), kept
    else:
        assert k is not None and k >= 1, "min_error 需提供正整數 k"
        return _dp_min_error(points, k)


# ---------- 3) Error-Search：給定段數 K，用二分搜 ε ----------
# 以 DP(min_number, ε) 為黑箱，找最小 ε 使得 段數 ≤ K。
def error_search(points: List[Point],
                 k: int,
                 eps_low: float = 0.0,
                 eps_high: Optional[float] = None,
                 tol: float = 1e-6,
                 max_iter: int = 60) -> Tuple[float, List[int]]:
    """
    回傳 (最小 ε, 對應的索引)。
    參考：DP 最少段數的單調性 + 二分搜（DP.cpp 的 DP_Number 思路，與 min_error.cpp 的 DP 見解相輔，:contentReference[oaicite:8]{index=8} :contentReference[oaicite:9]{index=9}）。
    """
    n = len(points)
    if n == 0:
        return 0.0, []
    if eps_high is None:
        # 粗估上界：以所有 (i,j) 段的最大 SED 做上界
        mx = 0.0
        for i in range(n-1):
            for j in range(i+1, n):
                mx = max(mx, _segment_max_sed(points, i, j))
        eps_high = mx

    lo, hi = eps_low, eps_high
    best_eps, best_kept = hi, [0, n-1]
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        segs, kept = _dp_min_number(points, mid)
        if segs <= k:
            best_eps, best_kept = mid, kept
            hi = mid
        else:
            lo = mid
        if hi - lo <= tol:
            break
    return best_eps, best_kept

if __name__ == "__main__":
    # 簡單測試
    points = [(0, 0, 0), (1, 0, 1), (2, 0, 2), (3, 1, 3), (4, 0, 4), (5, 0, 5)]
    # 假設 points 是 [(lat, lon, time), ...]
    epsilon = 5.0
    kept_td = td_tr(points, epsilon)  # TD-TR（SED）簡化

    # DP：給定 ε，最少段數
    segs, kept_dp_eps = dp(points, objective="min_number", epsilon=epsilon)

    # DP：給定段數 K，最小最大誤差
    eps_star, kept_dp_k = dp(points, objective="min_error", k=20)

    # Error-Search：指定 K，用二分搜 ε + DP(min_number) 得到最小 ε 與切點
    eps_min, kept_es = error_search(points, k=20)