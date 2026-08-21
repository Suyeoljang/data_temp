# -*- coding: utf-8 -*-
"""
FAB2 오버레이 — EMA 잔차 보정 (infer 결과 res_df에 적용)
========================================================
AutoGluon 예측(PRED_*)에 시간 드리프트 보정을 적용해 최종 예측을 만든다.

    최종 예측 = PRED_*  +  blend · exp(-gap일/decay) · EMA(잔차)
    EMA 갱신  : ema ← α·(CAL - PRED) + (1-α)·ema

전제: infer 단계에서 만든 res_df CSV (PRED_*, CAL_*, PROC_*, PROC_TIME,
      EQUIP, ROUTE_DESC, PROCESS_ID, RETICLE_ID 컬럼 포함)

실행:
    python fab2_ol_ema_correction.py
"""

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────────
INPUT_CSV = 'fab2_df_processing_OL_infer_260812_FAB2_OL_MODEL_260806.csv'
OUTPUT_CSV = 'fab2_df_processing_OL_infer_EMA.csv'

# 파라미터 선택 구간 / 평가 구간 분할 (낙관 편향 차단)
#   전반부에서 전략·파라미터 선택 → 후반부에서만 성능 측정
SPLIT_DATE = pd.Timestamp('2026-01-16')

# 'auto'  : 전반부에서 타겟별 최적 전략·파라미터 자동 선택 (권장)
# 'fixed' : 아래 FIXED_STRATEGY / FIXED_PARAM 그대로 사용
MODE = 'auto'

# LightGBM 실험에서 확정된 전략 (MODE='fixed'일 때 사용)
FIXED_STRATEGY = {
    'MSX': 'hier', 'MSY': 'hier', 'RR': 'hier', 'MAGX': 'hier',
    'MAGY': 'hier', 'SKEW': 'hier',
    'SCALX': 'ret', 'SCALY': 'ret', 'W_ROT': 'ret',
    'SA_MAG': 'ret', 'SA_ROT': 'ret',
    'ORT': 'none',
}
FIXED_PARAM = dict(alpha=0.3, blend=0.85, decay=30)

TARGETS = ['MSX', 'MSY', 'SCALX', 'SCALY', 'ORT', 'W_ROT',
           'RR', 'MAGX', 'MAGY', 'SKEW', 'SA_MAG', 'SA_ROT']

KEY_G1 = ['EQUIP', 'ROUTE_DESC', 'PROCESS_ID']
KEY_RET = ['EQUIP', 'ROUTE_DESC', 'PROCESS_ID', 'RETICLE_ID']

# 전략 = EMA 그룹키 우선순위 목록 (앞쪽 키에 상태가 있으면 그것을 사용)
STRATEGIES = {
    'none': [],                    # 보정 안 함
    'g1':   [KEY_G1],              # G1 키 단독
    'ret':  [KEY_RET],             # RETICLE 키 단독
    'hier': [KEY_RET, KEY_G1],     # RETICLE 우선 → G1 fallback → 무보정
}

GRID = [(a, b, d)
        for a in (0.2, 0.3, 0.4, 0.5)
        for b in (0.85, 1.0)
        for d in (30, 60)]


# ─────────────────────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────────────────────
def to_num(s):
    """'-' 및 공백을 NaN으로 변환 후 수치화 (Oracle 유래 컬럼 방어)."""
    return pd.to_numeric(s.astype(str).str.strip().replace('-', np.nan), errors='coerce')


def build_keys(df, cols):
    """그룹키 문자열 배열 생성."""
    return df[cols].astype(str).apply(lambda s: s.str.strip()).agg('|'.join, axis=1).values


def ema_correct(key_arrays, pred, cal, dt, alpha, blend, decay):
    """
    시간순 EMA 잔차 보정. 입력 순서를 그대로 유지한 배열을 반환한다.

    key_arrays : 그룹키 배열들의 우선순위 목록. 빈 목록이면 무보정.
                 예) [ret_keys, g1_keys] → RETICLE 상태가 있으면 그것을, 없으면 G1을 사용
    pred, cal  : float 배열 (cal이 NaN인 행은 상태 갱신에서 제외 — 계측 없음)
    dt         : numpy datetime64 배열
    """
    n = len(pred)
    out = pred.astype(float).copy()
    if not key_arrays:
        return out

    states = [dict() for _ in key_arrays]
    # 시간순 처리. 동일 타임스탬프(초 단위 동시 랏)는 입력 순서를 tie-breaker로 사용해
    # 행 순서가 바뀌어도 결과가 달라지지 않도록 고정한다.
    order = np.lexsort((np.arange(n), dt))

    for i in order:
        # 1) 보정: 우선순위가 가장 높은, 상태가 존재하는 키를 사용
        for lvl, karr in enumerate(key_arrays):
            st = states[lvl].get(karr[i])
            if st is not None:
                gap_days = (dt[i] - st[1]) / np.timedelta64(1, 'D')
                out[i] = pred[i] + blend * np.exp(-gap_days / decay) * st[0]
                break

        # 2) 상태 갱신: 계측값(CAL)이 있는 행만. 모든 계층을 함께 갱신
        if not np.isnan(cal[i]):
            resid = cal[i] - pred[i]
            for lvl, karr in enumerate(key_arrays):
                st = states[lvl].get(karr[i])
                new_ema = resid if st is None else alpha * resid + (1 - alpha) * st[0]
                states[lvl][karr[i]] = (new_ema, dt[i])

    return out


def mae(a, b, mask):
    return float(np.mean(np.abs(a[mask] - b[mask])))


# ─────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────
def main():
    df = pd.read_csv(INPUT_CSV, dtype=str)
    df.columns = [c.strip() for c in df.columns]

    need = ['PROC_TIME'] + KEY_RET
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f'필수 컬럼 누락: {missing}')

    dt_all = pd.to_datetime(df['PROC_TIME'].astype(str).str.strip(),
                            format='%Y%m%d%H%M%S', errors='coerce')
    if dt_all.isna().any():
        raise ValueError(f'PROC_TIME 파싱 실패 {int(dt_all.isna().sum())}건 — 공백/포맷 확인')
    dt_np = dt_all.values

    keys = {'g1': build_keys(df, KEY_G1), 'ret': build_keys(df, KEY_RET)}
    strat_keys = {
        'none': [],
        'g1':   [keys['g1']],
        'ret':  [keys['ret']],
        'hier': [keys['ret'], keys['g1']],
    }

    is_first = (dt_all < SPLIT_DATE).values      # 전반부: 선택용
    is_second = ~is_first                        # 후반부: 평가용

    report = []
    for t in TARGETS:
        c_pred, c_cal, c_proc = f'PRED_{t}', f'CAL_{t}', f'PROC_{t}'
        if not all(c in df.columns for c in (c_pred, c_cal, c_proc)):
            print(f'[{t}] 컬럼 없음, skip')
            continue

        pred = to_num(df[c_pred]).values
        cal = to_num(df[c_cal]).values
        proc = to_num(df[c_proc]).values

        valid = ~np.isnan(cal) & ~np.isnan(pred) & ~np.isnan(proc)
        m1, m2 = valid & is_first, valid & is_second
        if m1.sum() < 30 or m2.sum() < 30:
            print(f'[{t}] 유효 데이터 부족 (전반부 {m1.sum()}, 후반부 {m2.sum()}), skip')
            continue

        # ── 전략·파라미터 선택 ──
        if MODE == 'fixed':
            best_strat = FIXED_STRATEGY.get(t, 'hier')
            a, b, d = FIXED_PARAM['alpha'], FIXED_PARAM['blend'], FIXED_PARAM['decay']
        else:
            best = None
            for sname, karrs in strat_keys.items():
                cands = [(0.3, 0.85, 30)] if sname == 'none' else GRID
                for a_, b_, d_ in cands:
                    corr = ema_correct(karrs, pred, cal, dt_np, a_, b_, d_)
                    score = mae(cal, corr, m1)
                    if best is None or score < best[0]:
                        best = (score, sname, a_, b_, d_)
            _, best_strat, a, b, d = best

        # ── 최종 보정값 산출 (전체 구간을 시간순으로 순차 처리) ──
        corrected = ema_correct(strat_keys[best_strat], pred, cal, dt_np, a, b, d)

        df[f'PRED_EMA_{t}'] = corrected
        df[f'CAL-PREDEMA_{t}'] = np.where(valid, cal - corrected, np.nan)

        report.append({
            'target': t,
            'strategy': best_strat,
            'param': '-' if best_strat == 'none' else f'a{a}/b{b}/d{d}',
            'n_eval': int(m2.sum()),
            'APC_MAE': round(mae(cal, proc, m2), 3),
            'raw_MAE': round(mae(cal, pred, m2), 3),
            'EMA_MAE': round(mae(cal, corrected, m2), 3),
            'APC_std': round(float(np.std(cal[m2] - proc[m2])), 3),
            'EMA_std': round(float(np.std(cal[m2] - corrected[m2])), 3),
        })

    res = pd.DataFrame(report)
    res['vs_APC%'] = (100 * (res.APC_MAE - res.EMA_MAE) / res.APC_MAE).round(1)
    res['raw_vs_APC%'] = (100 * (res.APC_MAE - res.raw_MAE) / res.APC_MAE).round(1)

    pd.set_option('display.width', 200)
    print(f'\n===== EMA 보정 결과 (평가구간: {SPLIT_DATE.date()} 이후) =====')
    print(res[['target', 'strategy', 'param', 'n_eval', 'APC_MAE',
               'raw_MAE', 'EMA_MAE', 'raw_vs_APC%', 'vs_APC%',
               'APC_std', 'EMA_std']].to_string(index=False))
    print(f"\n역전: {(res['vs_APC%'] > 0).sum()}/{len(res)}  |  평균 vs APC: {res['vs_APC%'].mean():.2f}%")
    print(f"(보정 전 raw 기준: {(res['raw_vs_APC%'] > 0).sum()}/{len(res)}, 평균 {res['raw_vs_APC%'].mean():.2f}%)")

    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8_sig')
    res.to_csv('ema_eval_summary.csv', index=False, encoding='utf-8_sig')
    print(f'\n저장 완료: {OUTPUT_CSV} / ema_eval_summary.csv')


if __name__ == '__main__':
    main()
