"""
배치 신호 생성 numba 커널.

rule_engine.build_signals 가 이 모듈을 자동으로 사용한다.
numba 미설치 시 순수 파이썬 경로로 자동 폴백(느리지만 결과 동일).
"""
import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except Exception:                                   # pragma: no cover
    HAVE_NUMBA = False

    def njit(*a, **k):
        def deco(f):
            return f
        return deco if not a else a[0]

LN2 = np.log(2.0)


@njit(cache=True)
def _run(T, OBS, SRC, KEEP, KPOS, CODE, SIZES, HLS,
         hl_stale, shrink_m, clip, robust_k, n_keep):
    """
    시간순 단일 패스. 각 행에서 query(현재행 미포함) 후 update.
    CODE : (na, n) 축별 정수코드
    반환  : V(n_keep, na*nh), 그리고 축별 상태 리스트
    """
    na = CODE.shape[0]
    n = T.shape[0]
    nh = HLS.shape[0]
    S = [np.zeros((SIZES[j], nh)) for j in range(na)]
    W = [np.zeros((SIZES[j], nh)) for j in range(na)]
    R = [np.zeros((SIZES[j], nh)) for j in range(na)]
    L = [np.full((SIZES[j], nh), -1e18) for j in range(na)]
    V = np.zeros((n_keep, na * nh))
    d_st = LN2 / hl_stale
    dob = LN2 / HLS

    for i in range(n):
        t = T[i]
        if KEEP[i]:
            r = KPOS[i]
            p = 0
            for j in range(na):
                ci = CODE[j, i]
                for h in range(nh):
                    lt = L[j][ci, h]
                    if lt > -1e17:
                        dt = t - lt
                        if dt < 0.0:
                            dt = 0.0
                        we = W[j][ci, h] * np.exp(-d_st * dt)
                        V[r, p + h] = S[j][ci, h] * we / (we + shrink_m)
                p += nh
        if SRC[i]:
            o = OBS[i]
            if o > clip:
                o = clip
            elif o < -clip:
                o = -clip
            for j in range(na):
                ci = CODE[j, i]
                for h in range(nh):
                    lt = L[j][ci, h]
                    if lt < -1e17:
                        S[j][ci, h] = o
                        W[j][ci, h] = 1.0
                        R[j][ci, h] = 0.0
                    else:
                        dt = t - lt
                        if dt < 0.0:
                            dt = 0.0
                        wo = W[j][ci, h] * np.exp(-dob[h] * dt)
                        dev = o - S[j][ci, h]
                        if robust_k > 0.0:
                            lim = robust_k * (np.sqrt(R[j][ci, h]) + 1e-9)
                            if dev > lim:
                                dev = lim
                            elif dev < -lim:
                                dev = -lim
                        new = S[j][ci, h] + dev / (wo + 1.0)
                        R[j][ci, h] = (R[j][ci, h] * wo + (o - new) ** 2) / (wo + 1.0)
                        S[j][ci, h] = new
                        W[j][ci, h] = wo + 1.0
                    L[j][ci, h] = t
    return V, S, W, R, L


def run_batch(T, OBS, SRC, keep, codes, sizes, hls,
              hl_stale, shrink_m, clip, robust_k):
    kpos = (np.cumsum(keep) - 1).astype(np.int64)
    V, S, W, R, L = _run(
        T.astype(np.float64), np.nan_to_num(OBS).astype(np.float64),
        SRC.astype(np.bool_), keep.astype(np.bool_), kpos,
        codes.astype(np.int64), np.asarray(sizes, np.int64),
        np.asarray(hls, np.float64),
        float(hl_stale), float(shrink_m), float(clip), float(robust_k),
        int(keep.sum()))
    return V, S, W, R, L
