"""
RULE 엔진 결과 테이블 생성 — viz.py --longgap 과 같은 형식으로 저장

  python make_results.py "data/*_CD_V2_part*.csv" --out rule_longgap.csv
  python make_results.py "data/*_CD_V2_part*.csv" \\
         --old "apc_ema_longgap_part*.csv" --out rule_longgap.csv     # 기존 방식과 비교

절차
  1) 원본 병합 (분할 파일 glob)
  2) cutoff 이전 이력으로 학습 (가중치·캘리브레이션). cutoff 이후는 완전 홀드아웃
  3) 전 기간 인과적 재현 — 각 Lot 은 자기 진행 시점까지의 이력만 사용 (운영과 동일)
  4) 간만진행(GAP ≥ 30일) 행만 추려 저장. PHASE 컬럼으로 TRAIN / TEST 구분
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
from rule_engine import RuleEngine, derive

TS = '%Y%m%d%H%M%S'
HELPER = {'SGN', 'VALID', 'TOL', 'RES', 'RRES', 'RTOL', 'U', 'LONG', 'SRC', 'GAP_D', 'NPRIOR'}


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
    b = (out.loc[m, 'IDEAL_EXPOSE'] - out.loc[m, 'PROC_EXPOSE']).abs() <= tol
    a = (out.loc[m, 'IDEAL_EXPOSE'] - out.loc[m, col]).abs() <= tol
    g, l = int((~b & a).sum()), int((b & ~a).sum())
    re0 = ((out.loc[m, 'PROC_EXPOSE'] - out.loc[m, 'IDEAL_EXPOSE']).abs() / out.loc[m, 'PROC_EXPOSE']).mean() * 100
    re1 = ((out.loc[m, col] - out.loc[m, 'IDEAL_EXPOSE']).abs() / out.loc[m, col]).mean() * 100
    print(f'  {label:26s} N={int(m.sum()):7,d}  CD≤1% {b.mean()*100:6.2f}% → {a.mean()*100:6.2f}% '
          f'({(a.mean()-b.mean())*100:+.2f}%p)  순증 {g-l:+6d}  손실률 {l/max(int(b.sum()),1)*100:5.2f}%'
          f'  상대오차 {re0:.3f}→{re1:.3f}%')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('data')
    ap.add_argument('--out', default='rule_longgap.csv')
    ap.add_argument('--cutoff', default='20260201')
    ap.add_argument('--key-cols', default='DEV_ID,ROUTE,PROC_EQ')
    ap.add_argument('--model', default='model_eval')
    ap.add_argument('--old', help='기존 방식 apc_ema_longgap CSV (glob 가능)')
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
    eng = RuleEngine().fit(raw[(ts < cut).values].copy())
    eng.save(a.model)

    # 3) 전 기간 인과적 재현
    df = derive(raw, verify_ideal=True)
    print(f'[make] 인과적 재현 중 (간만진행 {int((df.LONG & df.VALID).sum()):,}건)')
    delta = eng.replay(df)
    q = (df['EXPOSE_TYPE'].map(C.QUANT).fillna(C.QUANT_DEFAULT).values
         if 'EXPOSE_TYPE' in df.columns else np.full(len(df), C.QUANT_DEFAULT))
    proc = df['PROC_EXPOSE'].values
    df['EMA_EXPOSE'] = np.where(delta != 0, np.round(proc * (1 + delta) / q) * q, proc)
    df['DELTA_PCT'] = np.round(delta * 100, 5)
    df['APPLIED'] = np.where(delta != 0, 'Y', 'N')
    df['GAP_DAYS'] = df['GAP_D'].round(3)
    df['CD_REL_ERR'] = (df['MEAS_CD'] - df['CD_TARGET']).abs() / df['CD_TARGET'] * 100
    df['PHASE'] = np.where(df['PROC_TRANS_DATE'] < cut, 'TRAIN', 'TEST')
    df['EMA_MODE'] = 'RULE_ROBUST_EMA'
    logic = df['CD_LOGIC'] if 'CD_LOGIC' in df.columns else pd.Series('', index=df.index)
    df['_EVAL'] = df['VALID'] & df['LONG'] & (df['CD_METHOD'] != 'FIX') & (logic != 'SC_IN_AI')

    # 4) 간만진행만
    out = df[df['LONG']].copy()
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

    print('\n[make] === 요약 (FIX·SC_IN_AI 제외, GAP≥30, 유효) ===')
    for ph in ['TRAIN', 'TEST']:
        summarize(out, 'EMA_EXPOSE', f'RULE · {ph}', out['PHASE'] == ph)
    if a.old:
        both = out['EMA_EXPOSE_OLD'].notna()
        print('  --- 동일 행 비교 ---')
        for ph in ['TRAIN', 'TEST']:
            m = both & (out['PHASE'] == ph)
            summarize(out, 'EMA_EXPOSE_OLD', f'기존 방식 · {ph}', m)
            summarize(out, 'EMA_EXPOSE', f'RULE · {ph}', m)
    print('\n  ※ TEST(cutoff 이후)가 정직한 성능입니다. TRAIN 은 가중치가 in-sample.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
