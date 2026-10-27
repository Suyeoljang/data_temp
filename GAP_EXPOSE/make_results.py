"""
RULE 엔진 결과 테이블 생성 — viz.py --longgap 과 같은 형식으로 저장

v3: 보정 대상 = 유효 & 이력 있음 & GAP >= config.GAP_THRESHOLD (기본 0 = 전 GAP).
    출력은 보정 대상 행 전체(--gap-min 으로 축소 가능). 요약은 GAP 구간별 Lot 기준 구제/손실/순증.

  python make_results.py "data/*_CD_V2_part*.csv" --out rule_longgap.csv
  python make_results.py "data/*_CD_V2_part*.csv" \\
         --old "apc_ema_longgap_part*.csv" --out rule_longgap.csv     # 기존 방식과 비교

절차
  1) 원본 병합 (분할 파일 glob)
  2) cutoff 이전 이력으로 학습 (가중치·캘리브레이션). cutoff 이후는 완전 홀드아웃
  3) 전 기간 인과적 재현 — 각 Lot 은 자기 진행 시점까지의 이력만 사용 (운영과 동일)
  4) 보정 대상 행(GAP >= --gap-min, 기본 GAP_THRESHOLD)만 추려 저장. PHASE 컬럼으로 TRAIN / TEST 구분
  5) --old 지정 시 기존 방식 EMA_EXPOSE 를 EMA_EXPOSE_OLD 로 붙임

출력 컬럼 (apc_ema_longgap.csv 와 호환)
  원본 컬럼 + IDEAL_EXPOSE, CD_REL_ERR, GAP_DAYS, EMA_EXPOSE,
  DELTA_PCT, APPLIED, PHASE, EMA_MODE, [EMA_EXPOSE_OLD]

옵션
  --cutoff 20260201          학습 종료일 (기본: 분석 때와 같은 2026-02-01)
  --key-cols DEV_ID,ROUTE,PROC_EQ
                             GAP 산출 키. 기본은 분석·기존 산출물과 같은 3컬럼.
                             운영 KEY(4컬럼)로 보려면 DEV_ID,PROC_EQ,ROUTE,RETICLE
  --model model_eval/        학습된 모델 저장 위치
"""
import argparse
import glob
import sys
import numpy as np
import pandas as pd
import config as C
from rule_engine import RuleEngine, derive, eq_target
from analyze import band_table

TS = '%Y%m%d%H%M%S'
HELPER = {'SGN', 'VALID', 'TOL', 'RES', 'RRES', 'RTOL', 'U', 'LONG', 'SRC', 'GAP_D', 'NPRIOR',
          'DAY', 'Y_LOG', 'EVAL', 'U_PROC', 'U_EMA'}


def read_glob(pattern):
    files = sorted(glob.glob(pattern),
                   key=lambda f: int(''.join(ch for ch in f.split('part')[-1] if ch.isdigit()) or 0)
                   if 'part' in f else 0)
    if not files:
        sys.exit(f'파일 없음: {pattern}')
    df = pd.concat([pd.read_csv(f, dtype=str, low_memory=False) for f in files], ignore_index=True)
    print(f'[make] {pattern}: {len(files)}개 파일 -> {len(df):,}행')
    return df


def match_key(d):
    return (d['LOT_ID'].astype(str) + '|' + d['ROUTE'].astype(str) + '|'
            + d['PROC_EQ'].astype(str) + '|' + d['PROC_TRANS_DATE'].astype(str))


def summarize(out, col, label, mask=None):
    m = out['_EVAL'] if mask is None else out['_EVAL'] & mask
    if not m.any():
        return
    tol = out.loc[m, 'MAIN_CONSTANT'] * out.loc[m, 'CD_TARGET']
    import metrics as MT
    b = (out.loc[m, 'IDEAL_EXPOSE'] - out.loc[m, 'PROC_EXPOSE']).abs() <= tol * MT.pass_u()
    a = (out.loc[m, 'IDEAL_EXPOSE'] - out.loc[m, col]).abs() <= tol * MT.pass_u()
    z = MT.zone_stats(((out.loc[m, 'IDEAL_EXPOSE'] - out.loc[m, 'PROC_EXPOSE']) / tol).values,
                      ((out.loc[m, 'IDEAL_EXPOSE'] - out.loc[m, col]) / tol).values)
    g, l = int((~b & a).sum()), int((b & ~a).sum())
    re0 = ((out.loc[m, 'PROC_EXPOSE'] - out.loc[m, 'IDEAL_EXPOSE']).abs() / out.loc[m, 'PROC_EXPOSE']).mean() * 100
    re1 = ((out.loc[m, col] - out.loc[m, 'IDEAL_EXPOSE']).abs() / out.loc[m, col]).mean() * 100
    print(f'  {label:26s} N={int(m.sum()):7,d}  CD≤{MT.pass_u():g}% {b.mean()*100:6.2f}% → {a.mean()*100:6.2f}% '
          f'({(a.mean()-b.mean())*100:+.2f}%p)  순증 {g-l:+6d}  손실률 {l/max(int(b.sum()),1)*100:5.2f}%'
          f'  상대오차 {re0:.3f}→{re1:.3f}%')
    print(f"  {'':26s} 3구간 점수 {z['Z_SCORE']:+d}  확실구제 {z['Z_HARD_RESCUE']} 확실손실 {z['Z_HARD_LOSS']}  "
          f"불량률 {z['Z_BAD0']:.2f}% → {z['Z_BAD1']:.2f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('data')
    ap.add_argument('--out', default='rule_longgap.csv')
    ap.add_argument('--cutoff', default='20260201')
    ap.add_argument('--key-cols', default='DEV_ID,ROUTE,PROC_EQ')
    ap.add_argument('--model', default='model_eval')
    ap.add_argument('--old', help='기존 방식 apc_ema_longgap CSV (glob 가능)')
    ap.add_argument('--gap-min', type=float, default=None, help='저장할 행의 GAP 하한(일). 기본 config.GAP_THRESHOLD')
    ap.add_argument('--band-csv', help='TEST 구간 GAP별 요약 저장 경로')
    ap.add_argument('--reuse-model', action='store_true', help='--model 에 이미 학습된 모델 사용(학습 생략)')
    a = ap.parse_args()

    C.KEY_COLS = [c.strip() for c in a.key_cols.split(',')]
    print(f'[make] GAP 산출 키 = {C.KEY_COLS}')

    raw = read_glob(a.data)
    if 'MATI_ID' in raw.columns and 'MAT_ID' not in raw.columns:
        raw = raw.rename(columns={'MATI_ID': 'MAT_ID'})
    ts = pd.to_datetime(raw['PROC_TRANS_DATE'].astype(str), format=TS, errors='coerce')
    cut = pd.Timestamp(a.cutoff)
    ntr = int((ts < cut).sum())
    print(f'[make] 학습 구간 < {cut:%Y-%m-%d}: {ntr:,}행 / 홀드아웃: {len(raw)-ntr:,}행')
    if ntr == 0:
        sys.exit('cutoff 이전 데이터가 없습니다.')

    # 2) 학습 (cutoff 이전만)
    if a.reuse_model:
        eng = RuleEngine.load(a.model)
        print(f'[make] 기존 모델 사용 {a.model} ({eng.version}/{eng.mode})')
    else:
        eng = RuleEngine().fit(raw[(ts < cut).values].copy())
        eng.save(a.model)

    # 3) 전 기간 인과적 재현
    df = derive(raw, verify_ideal=True)
    del raw
    import gc; gc.collect()
    print(f'[make] 인과적 재현 중 (보정 대상 {int(eng.default_mask(df).sum()):,}건, {eng.version}/{eng.mode})')
    delta, info = eng.replay(df, return_info=True)
    df['TARGET'] = info['TARGET']
    df['BETA'] = np.round(info['BETA'], 4)
    q = (df['EXPOSE_TYPE'].map(C.QUANT).fillna(C.QUANT_DEFAULT).values
         if 'EXPOSE_TYPE' in df.columns else np.full(len(df), C.QUANT_DEFAULT))
    proc = df['PROC_EXPOSE'].values
    df['EMA_EXPOSE'] = np.where(delta != 0, np.round(proc * (1 + delta) / q) * q, proc)
    df['DELTA_PCT'] = np.round(delta * 100, 5)
    df['APPLIED'] = np.where(delta != 0, 'Y', 'N')
    df['GAP_DAYS'] = df['GAP_D'].round(3)
    df['CD_REL_ERR'] = (df['MEAS_CD'] - df['CD_TARGET']).abs() / df['CD_TARGET'] * 100
    df['PHASE'] = np.where(df['PROC_TRANS_DATE'] < cut, 'TRAIN', 'TEST')
    df['EMA_MODE'] = f'RULE_{eng.version.upper()}_{eng.mode.upper()}'
    df['APPLY_SCALE'] = C.APPLY_SCALE
    logic = df['CD_LOGIC'] if 'CD_LOGIC' in df.columns else pd.Series('', index=df.index)
    df['_EVAL'] = df['VALID'] & np.isfinite(df['GAP_D'].values) & (df['CD_METHOD'] != 'FIX') & (logic != 'SC_IN_AI')
    df['EVAL'] = df['_EVAL']
    df['U_PROC'] = (df['IDEAL_EXPOSE'] - df['PROC_EXPOSE']) / df['TOL']
    df['U_EMA'] = (df['IDEAL_EXPOSE'] - df['EMA_EXPOSE']) / df['TOL']
    gmin = C.GAP_THRESHOLD if a.gap_min is None else a.gap_min

    # 4) 보정 대상 GAP 이상 & 적용 장비만
    out = df[np.isfinite(df['GAP_D'].values) & (df['GAP_D'].values >= gmin)
             & eq_target(df['PROC_EQ'])].copy()
    out['PROC_TRANS_DATE'] = out['PROC_TRANS_DATE'].dt.strftime(TS)

    # 5) 기존 방식 붙이기
    if a.old:
        old = read_glob(a.old)
        old['_K'] = match_key(old)
        out['_K'] = match_key(out)
        mp = old.drop_duplicates('_K').set_index('_K')['EMA_EXPOSE']
        out['EMA_EXPOSE_OLD'] = pd.to_numeric(out['_K'].map(mp), errors='coerce')
        hit = out['EMA_EXPOSE_OLD'].notna()
        print(f'[make] 기존 방식 매칭 {int(hit.sum()):,} / {len(out):,}행 ({hit.mean()*100:.1f}%)')

    cols = [c for c in out.columns if not c.startswith('_') and c not in HELPER] + ['GAP_DAYS']
    cols = list(dict.fromkeys(cols))
    out[cols].to_csv(a.out, index=False)
    print(f'[make] 저장 {a.out}  ({len(out):,}행, {len(cols)}컬럼)')

    print(f'\n[make] === 요약 (FIX·SC_IN_AI 제외, 유효, GAP>={gmin:g}, 장비 {C.APPLY_EQ_PREFIX or "전체"}) ===')
    for ph in ['TRAIN', 'TEST']:
        summarize(out, 'EMA_EXPOSE', f'RULE · {ph}', out['PHASE'] == ph)
    if a.old:
        both = out['EMA_EXPOSE_OLD'].notna()
        print('  --- 동일 행 비교 ---')
        for ph in ['TRAIN', 'TEST']:
            m = both & (out['PHASE'] == ph)
            summarize(out, 'EMA_EXPOSE_OLD', f'기존 방식 · {ph}', m)
            summarize(out, 'EMA_EXPOSE', f'RULE · {ph}', m)
    t = band_table(df, since=cut.strftime('%Y%m%d'))
    import metrics as MT
    print('\n[make] === TEST GAP 구간별 Lot 기준 (구제/손실/순증) ===')
    for r in t.itertuples():
        print(f'  {r.GAP:10s} N={r.N:8,d} 적용 {r.applied:5.1f}%  {r.APC:6.2f}% -> {r.COR:6.2f}% ({r.COR-r.APC:+.2f}%p)  '
              f'구제 {r.rescue:6,d} 손실 {r.loss:6,d} 순증 {r.net:+6,d}  손실률 {r.lossrate:4.1f}%')
    print(f'\n[make] === TEST 3구간 (합격 <={MT.pass_u():g}% / 회색 ~{MT.gray_u():g}% / 불량) ===')
    for r in t.itertuples():
        print(f'  {r.GAP:10s} 점수 {r.z_score:+6,d}  확실구제 {r.hard_rescue:5,d} 확실손실 {r.hard_loss:5,d}  '
              f'경미개선 {r.soft_up:5,d} 경미악화 {r.soft_down:5,d}  불량률 {r.bad0:5.2f}% -> {r.bad1:5.2f}%')
    from analyze import print_decision
    mt = (df['EVAL'] & np.isfinite(df['GAP_D'].values) & df['TARGET'].astype(bool)
          & (df['PROC_TRANS_DATE'] >= cut)).values
    sc = print_decision(df, mt, tag='[make] TEST')
    if a.band_csv:
        t.to_csv(a.band_csv, index=False, encoding='utf-8-sig')
        sc.to_csv(a.band_csv.replace('.csv', '_scale.csv'), index=False, encoding='utf-8-sig')
    print('\n  ※ TEST(cutoff 이후)가 정직한 성능입니다. TRAIN 은 가중치가 in-sample.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
