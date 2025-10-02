
from typing import List, Tuple, Literal, Optional, Callable
import math

# ------------------------------
# Types
# ------------------------------
Point = Tuple[float, float, float]  # (lat, lon, time)

# ------------------------------
# Core geometry helpers (SED)
# ------------------------------
def _sed(s: Point, m: Point, e: Point) -> float:
    """Synchronous Euclidean Distance (time-ratio between s and e)."""
    (slat, slon, st), (mlat, mlon, mt), (elat, elon, et) = s, m, e
    den = (et - st)
    # Guard division by zero in time
    tr = 1.0 if den == 0 else (mt - st) / den
    ilat = slat + (elat - slat) * tr
    ilon = slon + (elon - slon) * tr
    return math.hypot(ilat - mlat, ilon - mlon)

def _segment_max_sed(points: List[Point], i: int, j: int) -> float:
    """Return max SED within [i, j] if approximated by single segment (i->j)."""
    if j - i <= 1:
        return 0.0
    s, e = points[i], points[j]
    mx = 0.0
    for k in range(i + 1, j):
        mx = max(mx, _sed(s, points[k], e))
    return mx

# ------------------------------
# 1) TD-TR (Top-Down, SED)
# ------------------------------
def td_tr(points: List[Point], epsilon: float) -> List[int]:
    """
    Top-Down SED simplification.
    Returns sorted kept indices (always includes endpoints if len>=1).
    """
    n = len(points)
    if n == 0:
        return []
    kept: List[int] = []

    def _recur(a: int, b: int):
        if b - a <= 1:
            kept.append(a); kept.append(b)
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
            kept.append(a); kept.append(b)

    _recur(0, n - 1)
    kept = sorted(set(kept))
    return kept

# ------------------------------
# 2) DP variants
# ------------------------------
def _appr_table(points: List[Point], eps: float) -> List[List[bool]]:
    """
    O(n^3) table: appr[i][j]=True iff segment i->j approximates i..j within SED<=eps.
    """
    n = len(points)
    appr = [[False]*n for _ in range(n)]
    for i in range(n):
        appr[i][i] = True
    for i in range(n):
        for j in range(i+1, n):
            appr[i][j] = (_segment_max_sed(points, i, j) <= eps)
    return appr

def _dp_min_number(points: List[Point], eps: float):
    """
    Given epsilon, find minimum #segments to cover trajectory with SED<=eps per segment.
    Returns (min_segments, kept_indices).
    """
    n = len(points)
    if n == 0:
        return 0, []
    appr = _appr_table(points, eps)
    # PN[i] = min segments from i..(n-1)
    PN = [math.inf]*n
    nxt = [-1]*n
    PN[n-1], nxt[n-1] = 1, n-1
    for i in range(n-2, -1, -1):
        if appr[i][n-1]:
            PN[i] = 1; nxt[i] = n-1
            continue
        best, best_j = math.inf, -1
        for j in range(i+1, n):
            if appr[i][j]:
                cand = 1 + PN[j]
                if cand < best:
                    best, best_j = cand, j
        PN[i], nxt[i] = best, best_j

    # Reconstruct kept indices
    kept = []
    i = 0
    while i != -1 and i < n:
        kept.append(i)
        j = nxt[i]
        if j == -1 or j == i:
            break
        i = j
    if kept and kept[-1] != n-1:
        kept.append(n-1)
    elif not kept:
        kept = [0, n-1]
    return int(PN[0] if PN[0] != math.inf else 1), kept

def _dp_min_error(points: List[Point], k: int):
    """
    Given segment count k, minimize the maximum per-segment SED.
    Returns (epsilon_star, kept_indices).
    """
    n = len(points)
    if n == 0:
        return 0.0, []
    # Precompute error(i,j)
    err = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            err[i][j] = _segment_max_sed(points, i, j)

    # E[i][s] = minimal max-error for i..end using s segments
    E = [[math.inf]*(k+1) for _ in range(n)]
    prv = [[-1]*(k+1) for _ in range(n)]

    # base: s=1 -> one segment to end
    for i in range(n-1):
        E[i][1] = err[i][n-1]
    E[n-1][1] = 0.0

    for s in range(2, k+1):
        for i in range(n-1):
            best, best_h = math.inf, -1
            for h in range(i+1, n):
                val = max(err[i][h], E[h][s-1])
                if val < best:
                    best, best_h = val, h
            E[i][s], prv[i][s] = best, best_h
        E[n-1][s] = 0.0

    # Reconstruct
    eps_star = E[0][k]
    kept = [0]
    i, s = 0, k
    while s > 1:
        h = prv[i][s]
        kept.append(h)
        i, s = h, s-1
    if kept[-1] != n-1:
        kept.append(n-1)
    return eps_star, kept

def dp(points: List[Point],
       objective: Literal["min_number", "min_error"],
       epsilon: Optional[float] = None,
       k: Optional[int] = None):
    """
    Unified DP entry.
      - objective="min_number": supply epsilon -> returns (segments, kept)
      - objective="min_error":  supply k       -> returns (epsilon_star, kept)
    """
    if objective == "min_number":
        assert epsilon is not None, "min_number 需提供 epsilon"
        return _dp_min_number(points, epsilon)
    else:
        assert k is not None and k >= 1, "min_error 需提供正整數 k"
        return _dp_min_error(points, k)

# Error-Search: given K, binary-search epsilon with dp(min_number, eps)
def error_search(points: List[Point], k: int,
                 eps_low: float = 0.0,
                 eps_high: Optional[float] = None,
                 tol: float = 1e-6,
                 max_iter: int = 60):
    """
    Returns (epsilon_star, kept) achieving <=K segments (i.e., <=K+1 points).
    """
    n = len(points)
    if n == 0:
        return 0.0, []
    if eps_high is None:
        # robust upper bound: max SED over all (i,j)
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

# ------------------------------
# New: ratio-driven wrappers
# ------------------------------
def _segment_max_sed_all(points: List[Point]) -> float:
    """Upper bound epsilon that guarantees the whole trajectory can be one segment."""
    n = len(points)
    if n <= 2:
        return 0.0
    return _segment_max_sed(points, 0, n-1)

def _kept_via_ratio_with_eps_search(
    algo_fn: Callable[[List[Point], float], List[int]],
    points: List[Point],
    target_kept: int,
    eps_lo: float = 0.0,
    max_expand: int = 24,
    max_iter: int = 60,
    tol: float = 1e-9,
) -> List[int]:
    """
    Generic epsilon search to achieve kept_count <= target_kept (monotone relationship).
    Uses data-driven upper bound for robustness.
    """
    n = len(points)
    if n == 0:
        return []
    target_kept = max(2, min(int(round(target_kept)), n))

    # 1) robust upper bound: max SED over whole trajectory
    hi = _segment_max_sed_all(points)
    kept = algo_fn(points, hi)
    cnt = len(set(kept))

    # 2) in rare cases still too many points, expand
    expand = 0
    while cnt > target_kept and expand < max_expand:
        hi = (hi * 2.0) if hi > 0 else 1e-6
        kept = algo_fn(points, hi)
        cnt = len(set(kept))
        expand += 1

    # 3) binary search
    lo = eps_lo
    best_eps, best_kept = hi, kept
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        kept_mid = algo_fn(points, mid)
        cnt_mid  = len(set(kept_mid))
        if cnt_mid <= target_kept:
            best_eps, best_kept = mid, kept_mid
            hi = mid
        else:
            lo = mid
        if hi - lo <= tol:
            break

    # 4) enforce endpoints
    return sorted(set(best_kept) | {0, n-1})

def tdtr_by_ratio(points: List[Point], keep_ratio: float) -> List[int]:
    """TD-TR (SED) controlled by keep_ratio (ratio of points to keep)."""
    n = len(points)
    target_kept = max(2, int(round(n * keep_ratio)))
    return _kept_via_ratio_with_eps_search(td_tr, points, target_kept)

def dp_by_ratio(points: List[Point], keep_ratio: float) -> List[int]:
    """
    DP(min_number, SED) controlled by keep_ratio.
    If you switch metric, replace _dp_min_number/SED accordingly.
    """
    n = len(points)
    target_kept = max(2, int(round(n * keep_ratio)))
    # Wrap dp(min_number, eps) for epsilon search by providing a small adapter:
    def _dp_eps(points_: List[Point], eps: float) -> List[int]:
        _, kept_idx = _dp_min_number(points_, eps)
        return kept_idx
    return _kept_via_ratio_with_eps_search(_dp_eps, points, target_kept)


if __name__ == "__main__":
    # Simple smoke test
    pts = [(0,0,0),(1,0,1),(2,0,2),(3,1,3),(4,0,4),(5,0,5)]
    kept1 = tdtr_by_ratio(pts, 0.4)
    kept2 = dp_by_ratio(pts, 0.4)
    print("TDTR keep idx:", kept1)
    print("DP   keep idx:", kept2)
