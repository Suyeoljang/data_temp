"""
배치 신호 생성 numba 커널.

rule_engine.build_signals 가 이 모듈을 자동으로 사용한다.
numba 미설치 시 순수 파이썬 경로로 자동 폴백(느리지만 결과 동일).

v3 변경
  - 모든 커널이 DAY(일 배치 번호)를 받는다. 같은 DAY 의 행은 "전일 배치까지 갱신된 상태"로
    조회한 뒤 하루치를 시간순으로 갱신한다(운영의 일 배치와 동일).
    DAY 를 행 번호로 주면 v2 와 같은 행 단위(조회→즉시 갱신) 동작이 된다.
  - 레벨 drift 추종(_run_level), L3 자기이력 EMA(_run_l3) 커널 추가.
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


# ============================================================
# 18축 x 8척도 EMA (잔차 RRES)
# ============================================================
@njit(cache=True)
def _run(T, DAY, OBS, SRC, KEEP, KPOS, CODE, SIZES, HLS,
         hl_stale, shrink_m, clip, robust_k, n_keep):
    """
    시간순 단일 패스. 행 i 조회 시점에는 DAY < DAY[i] 인 행까지만 상태에 반영돼 있다.
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
    V = np.zeros((n_keep, na * nh), np.float32)
    d_st = LN2 / hl_stale
    dob = LN2 / HLS
    u = 0
    for i in range(n + 1):
        # ---- 이전 배치(DAY < DAY[i]) 갱신 ----
        while u < n and (i == n or DAY[u] < DAY[i]):
            if SRC[u]:
                t = T[u]
                o = OBS[u]
                if o > clip:
                    o = clip
                elif o < -clip:
                    o = -clip
                for j in range(na):
                    ci = CODE[j, u]
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
            u += 1
        if i == n:
            break
        # ---- 조회 ----
        if KEEP[i]:
            t = T[i]
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
    return V, S, W, R, L


def run_batch(T, OBS, SRC, keep, codes, sizes, hls,
              hl_stale, shrink_m, clip, robust_k, day=None):
    kpos = (np.cumsum(keep) - 1).astype(np.int64)
    if day is None:
        day = np.arange(len(T), dtype=np.int64)          # v2 호환: 행 단위
    V, S, W, R, L = _run(
        T.astype(np.float64), np.asarray(day, np.int64),
        np.nan_to_num(OBS).astype(np.float64),
        SRC.astype(np.bool_), keep.astype(np.bool_), kpos,
        codes.astype(np.int64), np.asarray(sizes, np.int64),
        np.asarray(hls, np.float64),
        float(hl_stale), float(shrink_m), float(clip), float(robust_k),
        int(keep.sum()))
    return V, S, W, R, L


# ============================================================
# 레벨 drift 추종  log IDEAL = B[key, j] + D[group_j]
#   조회값 = D[group](지금) - KD[key, j](그 KEY 마지막 진행 시점의 D)
#   = "KEY 가 마지막으로 돈 이후 그 그룹의 노광 수준 변화"
# ============================================================
@njit(cache=True)
def _run_level(T, DAY, Y, SRC, KEEP, KPOS, GID, KID, NG, NKEY, HLS,
               clipv, key_alpha, min_alpha, n_keep):
    J = GID.shape[0]
    n = T.shape[0]
    D = np.zeros(NG)
    TG = np.full(NG, -1.0e18)
    B = np.zeros((NKEY, J))
    KD = np.zeros((NKEY, J))
    HK = np.zeros((NKEY, J), np.bool_)
    V = np.full((n_keep, J), np.nan)
    u = 0
    for i in range(n + 1):
        while u < n and (i == n or DAY[u] < DAY[i]):
            k = KID[u]
            for j in range(J):
                g = GID[j, u]
                if SRC[u]:
                    if HK[k, j]:
                        res = Y[u] - B[k, j] - D[g]
                        if res > clipv:
                            res = clipv
                        elif res < -clipv:
                            res = -clipv
                        a = 0.0
                        if TG[g] > -1e17:
                            dt = T[u] - TG[g]
                            if dt < 0.0:
                                dt = 0.0
                            a = 1.0 - np.exp(-LN2 * dt / HLS[j])
                        if a < min_alpha:
                            a = min_alpha
                        D[g] += a * res
                        TG[g] = T[u]
                        B[k, j] += key_alpha * (Y[u] - B[k, j] - D[g])
                    else:
                        B[k, j] = Y[u] - D[g]
                        HK[k, j] = True
                        if TG[g] < -1e17:
                            TG[g] = T[u]
                KD[k, j] = D[g]
            u += 1
        if i == n:
            break
        if KEEP[i]:
            k = KID[i]
            r = KPOS[i]
            for j in range(J):
                if HK[k, j]:
                    V[r, j] = D[GID[j, i]] - KD[k, j]
    return V, D, TG, B, KD, HK


# ============================================================
# L3 자기이력 EMA  (IDEAL 값 자체의 지수가중평균)
# ============================================================
@njit(cache=True)
def _run_l3(T, DAY, OBS, SRC, KEEP, KPOS, LID, NL, HLS, n_keep):
    J = HLS.shape[0]
    n = T.shape[0]
    S = np.zeros((NL, J))
    W = np.zeros((NL, J))
    L = np.full((NL, J), -1.0e18)
    V = np.full((n_keep, J), np.nan)
    u = 0
    for i in range(n + 1):
        while u < n and (i == n or DAY[u] < DAY[i]):
            if SRC[u]:
                k = LID[u]
                for j in range(J):
                    if L[k, j] > -1e17:
                        dt = T[u] - L[k, j]
                        if dt < 0.0:
                            dt = 0.0
                        w = W[k, j] * np.exp(-LN2 / HLS[j] * dt)
                        S[k, j] = (S[k, j] * w + OBS[u]) / (w + 1.0)
                        W[k, j] = w + 1.0
                    else:
                        S[k, j] = OBS[u]
                        W[k, j] = 1.0
                    L[k, j] = T[u]
            u += 1
        if i == n:
            break
        if KEEP[i]:
            k = LID[i]
            if L[k, 0] > -1e17:
                for j in range(J):
                    V[KPOS[i], j] = S[k, j]
    return V, S, W, L


# ============================================================
# β 배율 (세그먼트별 보정 배율, 계층 shrink)
#   세그먼트 l 의 상태: A = Σ w·s·u,  B = Σ w·s²   (w = exp(-λ·경과일))
#   조회: β_0 = 1,  β_l = (A_l + K·β_{l-1}) / (B_l + K)   → clip [lo, hi]
#   s = 엔진 원 보정량(u 단위, β·게이트 적용 전), u = APC 원값 기준 CD 오차(u 단위)
#   일 배치 의미: 행 i 조회 시점에는 DAY < DAY[i] 인 행까지만 반영
# ============================================================
@njit(cache=True)
def _beta_upd(j, DAY, KID, S, U, A, B, TT, lam):
    for l in range(KID.shape[1]):
        k = KID[j, l]
        if k < 0:
            continue
        if TT[l, k] > -1e17:
            f = np.exp(-lam * (DAY[j] - TT[l, k]))
            A[l, k] *= f
            B[l, k] *= f
        A[l, k] += S[j] * U[j]
        B[l, k] += S[j] * S[j]
        TT[l, k] = DAY[j]


@njit(cache=True)
def _run_beta(DAY, KID, S, U, QRY, UPD, NKMAX, lam, K, lo, hi):
    n, L = KID.shape
    A = np.zeros((L, NKMAX))
    B = np.zeros((L, NKMAX))
    TT = np.full((L, NKMAX), -1.0e18)
    beta = np.ones(n)
    g0 = 0
    for i in range(n):
        if i > 0 and DAY[i] != DAY[i - 1]:
            for j in range(g0, i):
                if UPD[j]:
                    _beta_upd(j, DAY, KID, S, U, A, B, TT, lam)
            g0 = i
        if QRY[i]:
            b = 1.0
            for l in range(L):
                k = KID[i, l]
                a = 0.0
                bb = 0.0
                if k >= 0 and TT[l, k] > -1e17:
                    f = np.exp(-lam * (DAY[i] - TT[l, k]))
                    a = A[l, k] * f
                    bb = B[l, k] * f
                b = (a + K * b) / (bb + K)
            beta[i] = min(max(b, lo), hi)
    for j in range(g0, n):
        if UPD[j]:
            _beta_upd(j, DAY, KID, S, U, A, B, TT, lam)
    return beta, A, B, TT
