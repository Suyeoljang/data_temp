"""
분석 — 과거 이력에 보정을 걸었다면 어땠을지 (운영에는 쓰지 않음)

  python analyze.py model/ lots.csv out.csv [--since 20260201]

용도
  1) 섀도우 분석 — 보정 적용 시 구제/손실/순증 사전 확인
  2) 모델 비교   — 재학습한 새 모델과 기존 모델을 같은 데이터로 대조
  3) 시각화 입력 — viz.py 가 이 결과를 쓴다

★ 인과적 재현: 각 Lot 은 자기 진행 시점까지의 이력만으로 보정된다(운영과 동일).
  따라서 입력에는 평가 구간 앞의 이력(최소 3개월)이 함께 있어야 상태가 쌓인다.
  학습에 쓴 기간을 평가하면 가중치가 in-sample 이므로 --since 로
  학습 이후 구간만 평가하는 것을 권한다.

GAP 은 입력 CSV 에 GAP_D 가 있으면 그것을, 없으면 config.KEY_COLS 로 산출한다.

v3: 보정 대상 = 유효 & 이력 있음 & GAP >= config.GAP_THRESHOLD (기본 0 = 전 GAP).
    요약은 Lot 기준 구제/손실/순증을 GAP 구간별·전체로 출력한다.
v3.1: 보정 대상에 적용 장비(config.APPLY_EQ_PREFIX, 기본 PTRD=DUV) 조건이 추가됐다.
      평가도 대상 장비 Lot 만 한다. 출력 컬럼 TARGET(대상 여부), BETA(β 배율), DN(원시 보정량),
      APPLIED(실제 노광량 변경 여부 — 작은 보정은 N).
"""
import argparse
import sys
import numpy as np
import pandas as pd
import config as C
from rule_engine import RuleEngine, derive, eq_target
import metrics as MT


def prepare(csv_in, model_dir):
    """보정 재현 결과 DataFrame 반환 (viz.py 에서도 사용)"""
    import glob
    files = sorted(glob.glob(csv_in)) or [csv_in]
    raw = pd.concat([pd.read_csv(f, dtype=str, low_memory=False) for f in files], ignore_index=True)
    if 'MATI_ID' in raw.columns and 'MAT_ID' not in raw.columns:
        raw = raw.rename(columns={'MATI_ID': 'MAT_ID'})
    if 'GAP_D' in raw.columns:
        raw['_GAP_IN'] = raw['GAP_D']
    df = derive(raw, verify_ideal=False)
    if '_GAP_IN' in df.columns:
        df['GAP_D'] = pd.to_numeric(df['_GAP_IN'], errors='coerce')
        print('[analyze] 입력 CSV 의 GAP_D 사용')
    eng = RuleEngine.load(model_dir)
    print(f'[analyze] 인과적 재현 중 ({len(df)}행, 보정 대상(GAP>={C.GAP_THRESHOLD:g}, 장비 '
          f'{C.APPLY_EQ_PREFIX or "전체"}) {int(eng.default_mask(df).sum())}건, {eng.version}/{eng.mode}, '
          f'β {"ON" if eng.use_beta else "OFF"}, 적용 임계 {eng.apply_min_u:.4f} CD%)')
    delta, info = eng.replay(df, return_info=True)
    df['TARGET'] = info['TARGET']
    df['LONG'] = df['TARGET']                   # viz 평가 대상 = 보정 대상(GAP·장비)
    df['BETA'] = np.round(info['BETA'], 4)
    df['DN'] = np.round(info['DN'], 5)
    df['APPLY_SCALE'] = C.APPLY_SCALE

    if 'EXPOSE_TYPE' in df.columns:
        q = df['EXPOSE_TYPE'].map(C.QUANT).fillna(C.QUANT_DEFAULT).values
    else:
        q = np.full(len(df), C.QUANT_DEFAULT)
    proc = df['PROC_EXPOSE'].values
    df['DELTA_PCT'] = delta * 100
    df['APPLIED'] = np.where(delta != 0, 'Y', 'N')
    df['EMA_EXPOSE'] = np.where(delta != 0, np.round(proc * (1 + delta) / q) * q, proc)
    logic = df['CD_LOGIC'] if 'CD_LOGIC' in df.columns else pd.Series('', index=df.index)
    df['EVAL'] = df['VALID'] & (logic != 'SC_IN_AI') & (df['CD_METHOD'] != 'FIX')
    df['U_PROC'] = (df['IDEAL_EXPOSE'] - df['PROC_EXPOSE']) / df['TOL']
    df['U_EMA'] = (df['IDEAL_EXPOSE'] - df['EMA_EXPOSE']) / df['TOL']
    return df


GAP_BANDS = [(0, 1, '같은 날(<1)'), (1, 3, '1~3'), (3, 7, '3~7'), (7, 14, '7~14'),
             (14, 30, '14~30'), (30, 1e9, '30 이상')]


def lot_summary(df, mask, used_col='U_EMA'):
    """Lot 기준 구제/손실/순증(PASS_U 판정) + 3구간 지표. 반환 dict"""
    pu = MT.pass_u()
    b = df.loc[mask, 'U_PROC'].abs() <= pu
    a = df.loc[mask, used_col].abs() <= pu
    z = MT.zone_stats(df.loc[mask, 'U_PROC'].values, df.loc[mask, used_col].values)
    g, l = int((~b & a).sum()), int((b & ~a).sum())
    ap = (df.loc[mask, 'APPLIED'].astype(str) == 'Y').mean() * 100 if 'APPLIED' in df.columns and mask.any() else np.nan
    return dict(N=int(mask.sum()), APC=b.mean() * 100 if len(b) else np.nan,
                COR=a.mean() * 100 if len(a) else np.nan, rescue=g, loss=l, net=g - l,
                lossrate=l / max(int(b.sum()), 1) * 100, applied=ap,
                z_score=z['Z_SCORE'], hard_rescue=z['Z_HARD_RESCUE'], hard_loss=z['Z_HARD_LOSS'],
                soft_up=z['Z_SOFT_UP'], soft_down=z['Z_SOFT_DOWN'], bad0=z['Z_BAD0'], bad1=z['Z_BAD1'],
                **_decision(df, mask, used_col))


def _decision(df, mask, used_col='U_EMA'):
    d = MT.decision_stats(df.loc[mask, 'U_PROC'].values, df.loc[mask, used_col].values)
    return dict(d_rescue=d['D_RESCUE'], d_loss=d['D_LOSS'], d_net=d['D_NET'],
                d_ratio=d['D_RATIO'], d_lossrate=d['D_LOSSRATE'])


def scenario_table(df, mask, used_col='U_EMA'):
    """배율 시나리오 (원 보정량 복원 근사). 반환 DataFrame"""
    sc_now = float(df['APPLY_SCALE'].iloc[0]) if 'APPLY_SCALE' in df.columns and len(df) else C.APPLY_SCALE
    rows = []
    for c, d in MT.scale_scenarios(df.loc[mask, 'U_PROC'].values, df.loc[mask, used_col].values, sc_now):
        rows.append(dict(배율=c, 현재=(abs(c - sc_now) < 1e-9), 구제=d['D_RESCUE'], 손실=d['D_LOSS'],
                         순증=d['D_NET'], 손실률=round(d['D_LOSSRATE'], 2), 구제손실비=round(d['D_RATIO'], 2),
                         불량률_후=round(d['D_BAD1'], 2)))
    return pd.DataFrame(rows)


def print_decision(df, mask, used_col='U_EMA', tag='[analyze]'):
    sp = MT.decision_spec()
    d = MT.decision_stats(df.loc[mask, 'U_PROC'].values, df.loc[mask, used_col].values)
    print(f'\n{tag} 결정 지표 — 확실한 불량 = |CD 오차| > {sp:g}%')
    print(f"  구제 {d['D_RESCUE']:,d} / 손실 {d['D_LOSS']:,d} / 순증 {d['D_NET']:+,d} · 손실률 {d['D_LOSSRATE']:.2f}% · "
          f"구제/손실 {d['D_RATIO']:.2f} · 불량률 {d['D_BAD0']:.2f}% -> {d['D_BAD1']:.2f}%")
    t = scenario_table(df, mask, used_col)
    print(f'{tag} 배율(APPLY_SCALE) 시나리오 — 원 보정량을 복원한 근사, 재학습 불필요')
    for r in t.itertuples():
        print(f"  {'▶' if r.현재 else ' '} 배율 {r.배율:4.2f}  구제 {r.구제:6,d} 손실 {r.손실:6,d} 순증 {r.순증:+7,d}  "
              f"손실률 {r.손실률:5.2f}%  구제/손실 {r.구제손실비:5.2f}  불량률(후) {r.불량률_후:5.2f}%")
    return t


def band_table(df, since=None, until=None, used_col='U_EMA'):
    m0 = df['EVAL'] & np.isfinite(df['GAP_D'].values)
    if 'TARGET' in df.columns:
        m0 &= df['TARGET'].astype(bool)
    elif 'PROC_EQ' in df.columns:
        m0 &= eq_target(df['PROC_EQ'])
    if since:
        m0 &= df['PROC_TRANS_DATE'] >= pd.Timestamp(since)
    if until:
        m0 &= df['PROC_TRANS_DATE'] < pd.Timestamp(until)
    rows = []
    for lo, hi, lbl in GAP_BANDS:
        m = m0 & (df['GAP_D'] >= lo) & (df['GAP_D'] < hi)
        if m.any():
            rows.append(dict(GAP=lbl, **lot_summary(df, m, used_col)))
    rows.append(dict(GAP='전체', **lot_summary(df, m0, used_col)))
    return pd.DataFrame(rows)


def summary(df, since=None, until=None):
    t = band_table(df, since, until)
    if t.empty:
        print('[analyze] 평가 대상 없음')
        return t
    rng = f" ({since or ''}~{until or ''})" if (since or until) else ''
    print(f'[analyze] Lot 기준 효과{rng} — FIX·SC_IN_AI 제외, 유효, 이력 있는 Lot · 합격 |CD 오차|<={MT.pass_u():g}%')
    print(f"  {'GAP(일)':10s} {'N':>8s} {'적용':>6s} {'APC':>7s} {'보정':>7s} {'Δ%p':>6s} {'구제':>7s} {'손실':>7s} {'순증':>7s} {'손실률':>6s}")
    for r in t.itertuples():
        print(f'  {r.GAP:10s} {r.N:8,d} {r.applied:5.1f}% {r.APC:6.2f}% {r.COR:6.2f}% {r.COR-r.APC:+6.2f} '
              f'{r.rescue:7,d} {r.loss:7,d} {r.net:+7,d} {r.lossrate:5.1f}%')
    print(f'\n[analyze] 3구간 판정 — 합격 <={MT.pass_u():g}% / 회색 ~{MT.gray_u():g}% (측정 오차 범위) / 불량 >{MT.gray_u():g}%')
    print(f"  {'GAP(일)':10s} {'3구간점수':>8s} {'확실구제':>7s} {'확실손실':>7s} {'경미개선':>7s} {'경미악화':>7s} {'불량률':>15s}")
    for r in t.itertuples():
        print(f'  {r.GAP:10s} {r.z_score:+8,d} {r.hard_rescue:7,d} {r.hard_loss:7,d} {r.soft_up:7,d} {r.soft_down:7,d} '
              f'{r.bad0:6.2f}% -> {r.bad1:5.2f}%')
    m0 = df['EVAL'] & np.isfinite(df['GAP_D'].values)
    if 'TARGET' in df.columns:
        m0 &= df['TARGET'].astype(bool)
    if since:
        m0 &= df['PROC_TRANS_DATE'] >= pd.Timestamp(since)
    if until:
        m0 &= df['PROC_TRANS_DATE'] < pd.Timestamp(until)
    print_decision(df, m0.values)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model'); ap.add_argument('csv_in'); ap.add_argument('csv_out')
    ap.add_argument('--since', help='평가 시작일 YYYYMMDD (학습 이후 구간 권장)')
    ap.add_argument('--until', help='평가 종료일 YYYYMMDD (미포함)')
    ap.add_argument('--band-csv', help='GAP 구간별 요약 저장 경로')
    a = ap.parse_args()
    df = prepare(a.csv_in, a.model)
    cols = [c for c in df.columns if not c.startswith('_')]
    df[cols].to_csv(a.csv_out, index=False)
    t = summary(df, a.since, a.until)
    if a.band_csv and len(t):
        t.to_csv(a.band_csv, index=False, encoding='utf-8-sig')
    return 0


if __name__ == '__main__':
    sys.exit(main())
