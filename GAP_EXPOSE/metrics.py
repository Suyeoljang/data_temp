"""
평가 지표 — 1% 합격 판정과 3구간 판정 (측정 오차 감안)

3구간 (config.PASS_U / GRAY_U, 단위 = CD 오차 %)
  합격 : |CD 오차| <= PASS_U                  점수 2
  회색 : PASS_U < |CD 오차| <= GRAY_U         점수 1   측정 오차 범위. 합격·불량 어느 쪽으로도 단정하지 않음
  불량 : |CD 오차| >  GRAY_U                  점수 0   측정 오차로 설명되지 않는 확실한 이탈

보정 전 → 후 이동
  합격 → 불량 : 확실한 손실 (-2)      불량 → 합격 : 확실한 구제 (+2)
  합격 → 회색 : 경미한 악화 (-1)      회색 → 합격 : 경미한 개선 (+1)
  회색 → 불량 : 경미한 악화 (-1)      불량 → 회색 : 경미한 개선 (+1)
3구간 점수 = Σ(보정 후 점수 − 보정 전 점수). 구제와 손실을 같은 무게로 센다.
"""
import numpy as np
import config as C

ZONE_NAMES = ['불량', '회색', '합격']          # 점수 0, 1, 2


def pass_u():
    return float(getattr(C, 'PASS_U', 1.0))


def gray_u():
    return float(getattr(C, 'GRAY_U', 2.0))


def zone(u):
    """u(CD 오차 %, 부호 무관) -> 0 불량 / 1 회색 / 2 합격"""
    a = np.abs(np.asarray(u, float))
    return np.where(a <= pass_u(), 2, np.where(a <= gray_u(), 1, 0))


def zone_label(u):
    return np.array(ZONE_NAMES, dtype=object)[zone(u)]


def zone_judge(u0, u1):
    """Lot 별 3구간 판정 문자열"""
    z0, z1 = zone(u0), zone(u1)
    d = z1 - z0
    out = np.full(len(z0), '유지', dtype=object)
    out[d == 2] = '확실한 구제'
    out[d == -2] = '확실한 손실'
    out[d == 1] = '경미한 개선'
    out[d == -1] = '경미한 악화'
    return out


def zone_stats(u0, u1):
    """보정 전(u0)·후(u1) 배열 -> 3구간 지표 dict"""
    z0, z1 = zone(u0), zone(u1)
    n = max(len(z0), 1)
    M = np.zeros((3, 3), int)                     # [전 구간, 후 구간]
    np.add.at(M, (z0, z1), 1)
    return dict(
        Z_N=int(len(z0)), Z_MAT=M,
        Z_SCORE=int((z1 - z0).sum()),
        Z_HARD_RESCUE=int(M[0, 2]), Z_HARD_LOSS=int(M[2, 0]),
        Z_SOFT_UP=int(M[1, 2] + M[0, 1]), Z_SOFT_DOWN=int(M[2, 1] + M[1, 0]),
        Z_BAD0=float((z0 == 0).sum() / n * 100), Z_BAD1=float((z1 == 0).sum() / n * 100),
        Z_PASS0=float((z0 == 2).sum() / n * 100), Z_PASS1=float((z1 == 2).sum() / n * 100),
        Z_GRAY0=float((z0 == 1).sum() / n * 100), Z_GRAY1=float((z1 == 1).sum() / n * 100))


# ============================================================
# 결정 지표 — "확실한 불량"(> GRAY_U) 기준의 구제 / 손실
#   구제 : 기존 APC 로는 불량(> GRAY_U) → 보정 후 GRAY_U 이내
#   손실 : 기존 APC 로는 GRAY_U 이내 → 보정 후 불량
#   (2026-10 공정 엔지니어 협의: 측정 오차를 감안해 2% 초과를 확실한 불량으로 본다)
# ============================================================
def decision_spec():
    return gray_u()


def decision_stats(u0, u1, spec=None):
    sp = decision_spec() if spec is None else float(spec)
    a0 = np.abs(np.asarray(u0, float)) <= sp
    a1 = np.abs(np.asarray(u1, float)) <= sp
    n = max(len(a0), 1)
    g = int((~a0 & a1).sum())
    l = int((a0 & ~a1).sum())
    return dict(D_SPEC=sp, D_RESCUE=g, D_LOSS=l, D_NET=g - l,
                D_RATIO=(g / l) if l else float('inf'),
                D_LOSSRATE=l / max(int(a0.sum()), 1) * 100,
                D_BAD0=float((~a0).sum() / n * 100), D_BAD1=float((~a1).sum() / n * 100))


def scale_scenarios(u0, u1, scale_now, scales=None, spec=None):
    """
    지금 결과(u0 → u1, 배율 scale_now)를 배율만 바꿨을 때의 결정 지표로 환산.
    원 보정량 = (u0 - u1) / scale_now 로 복원해 배율을 곱한다. 노광량 최소단위 반올림은 무시(근사).
    반환: [(배율, decision_stats dict), ...]
    """
    import config as _C
    scales = list(getattr(_C, 'SCALE_SCENARIOS', [0.5, 0.65, 0.8, 1.0]) if scales is None else scales)
    u0 = np.asarray(u0, float)
    raw = (u0 - np.asarray(u1, float)) / max(float(scale_now), 1e-9)
    if float(scale_now) not in scales:
        scales = sorted(set(scales) | {float(scale_now)})
    return [(c, decision_stats(u0, u0 - c * raw, spec)) for c in sorted(scales)]
