"""
원본 데이터 점검 — make_results.py / fit.py 전에 실행

  python check_data.py photo_apc_data_250101_260801.csv
"""
import sys
import numpy as np
import pandas as pd

f = sys.argv[1]
use = ['PROC_TRANS_DATE', 'CD_TYPE', 'CD_TARGET', 'MEAS_CD', 'MAIN_CONSTANT',
       'PROC_EXPOSE', 'CD_METHOD', 'CD_LOGIC', 'IDEAL_EXPOSE', 'PR_ID']
head = pd.read_csv(f, nrows=0).columns
df = pd.read_csv(f, usecols=[c for c in use if c in head], dtype=str, low_memory=False)
print(f'행 {len(df):,}  |  없는 컬럼: {[c for c in use if c not in head]}')
for c in ['CD_TARGET', 'MEAS_CD', 'MAIN_CONSTANT', 'PROC_EXPOSE']:
    df[c] = pd.to_numeric(df[c], errors='coerce')

print('\n[CD_TYPE 값] — SPACE/LINE(또는 S/L) 외 값이 있으면 부호가 틀어집니다')
print(df['CD_TYPE'].value_counts(dropna=False).head(10).to_string())

print('\n[무효 후보]')
for nm, m in [('CD_TARGET <= 0 / 결측', ~(df.CD_TARGET > 0)),
              ('MEAS_CD <= 0 / 결측', ~(df.MEAS_CD > 0)),
              ('MAIN_CONSTANT <= 0', ~(df.MAIN_CONSTANT > 0)),
              ('PROC_EXPOSE <= 0', ~(df.PROC_EXPOSE > 0))]:
    print(f'  {nm:24s} {int(m.sum()):>9,}  ({m.mean()*100:5.2f}%)')

ok = (df.CD_TARGET > 0) & (df.MEAS_CD > 0)
e = ((df.MEAS_CD - df.CD_TARGET).abs() / df.CD_TARGET)[ok] * 100
print('\n[|CD 오차| 분포, %]  — 분석에 쓴 V2 데이터는 최대 15.4%')
print('  ' + '  '.join(f'P{q}={np.percentile(e, q):.2f}' for q in [50, 90, 99, 99.9]) + f'  max={e.max():.1f}')
for t in [3, 10, 30, 50]:
    print(f'  > {t:2d}%  {int((e > t).sum()):>8,}  ({(e > t).mean()*100:5.2f}%)')

if 'IDEAL_EXPOSE' in df.columns:
    s = np.where(df.CD_TYPE.astype(str).str.upper().isin(['SPACE', 'S']), 1., -1.)
    calc = df.PROC_EXPOSE + s * (df.CD_TARGET - df.MEAS_CD) * 100 * df.MAIN_CONSTANT
    ref = pd.to_numeric(df.IDEAL_EXPOSE, errors='coerce')
    m = ok & (df.MAIN_CONSTANT > 0) & ref.notna()
    d = (calc[m] - ref[m]).abs()
    print(f'\n[부호 규약] IDEAL 역산 일치율 {(d < 1e-4).mean()*100:.2f}%  (100% 가 정상)')
else:
    print('\n[부호 규약] IDEAL_EXPOSE 컬럼이 없어 검증 불가')
