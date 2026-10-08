"""
합성 데이터로 파이프라인 전체를 점검한다.

  python selftest.py

실데이터·DB 없이 fit -> analyze 가 끝까지 도는지 확인하는 용도이며,
성능 수치는 의미 없다(무작위 데이터).
"""
import os
import shutil
import subprocess
import sys
import numpy as np
import pandas as pd

RNG = np.random.RandomState(0)
N = 60000


def make(n=N, start='2024-01-01', days=540):
    eq = [f'PTRA{i:02d}' for i in range(1, 13)]
    route = [f'R{i:02d}' for i in range(1, 16)]
    dev = [f'D{i:03d}' for i in range(1, 26)]
    ret = [f'RT{i:03d}' for i in range(1, 31)]
    t = pd.to_datetime(start) + pd.to_timedelta(
        np.sort(RNG.uniform(0, days, n)) * 86400, unit='s')
    cd_type = RNG.choice(['SPACE', 'LINE'], n, p=[.7, .3])
    tgt = np.where(cd_type == 'SPACE', RNG.choice([0.23, 0.27, 0.43], n),
                   RNG.choice([0.23, 0.25], n))
    mc = RNG.choice([0.9, 2.0, 2.3, 3.5], n)
    proc = RNG.choice([29.2, 33.5, 44.0, 245.2, 453.2], n) * RNG.uniform(.98, 1.02, n)
    # 장비별 느린 드리프트 + Lot 노이즈
    eqi = RNG.randint(0, len(eq), n)
    drift = 0.004 * np.sin(2 * np.pi * (t.dayofyear.values / 180.0) + eqi)
    rres = drift + RNG.normal(0, 0.012, n)
    s = np.where(cd_type == 'SPACE', 1.0, -1.0)
    meas = tgt - s * (rres * proc) / (100 * mc)
    df = pd.DataFrame({
        'PROC_TRANS_DATE': t.strftime('%Y%m%d%H%M%S'),
        'PROC_EQ': [eq[i] for i in eqi],
        'ROUTE': RNG.choice(route, n),
        'DEV_ID': RNG.choice(dev, n),
        'RETICLE': RNG.choice(ret, n),
        'TRACK_RECIPE': RNG.choice([f'T{i}' for i in range(1, 16)], n),
        'ROUTE_DESC': RNG.choice([f'L{i:02d}' for i in range(1, 21)], n),
        'ROUTE_PREFIX': RNG.choice([f'P{i:02d}' for i in range(1, 11)], n),
        'RETICLE_SUBFIX': RNG.choice([f'S{i:02d}' for i in range(1, 13)], n),
        'TECH': RNG.choice(['TA', 'TB', 'TC'], n),
        'PR_ID': RNG.choice(['PR1', 'PR2', 'PR3'], n),
        'MAT_ID': RNG.choice([f'M{i:03d}' for i in range(40)], n),
        'PROCESS_ID': RNG.choice([f'P{i:02d}' for i in range(12)], n),
        'EXPOSE_TYPE': RNG.choice(['ASML', 'STEPPER'], n),
        'CD_TYPE': cd_type,
        'CD_METHOD': RNG.choice(['APC', 'Logic', 'FIX'], n, p=[.9, .09, .01]),
        'CD_TARGET': np.round(tgt, 4),
        'MEAS_CD': np.round(meas, 5),
        'PROC_EXPOSE': np.round(proc, 2),
        'MAIN_CONSTANT': mc,
        'LOT_ID': [f'L{i:06d}' for i in range(n)],
    })
    return df


def main():
    os.makedirs('_selftest', exist_ok=True)
    os.chdir('_selftest')
    df = make()
    k = int(len(df) * 0.9)
    cut = np.arange(len(df)) < k
    df[cut].to_csv('hist.csv', index=False)
    df.to_csv('all.csv', index=False)          # analyze 는 GAP 산출 위해 전체 필요
    print(f'[selftest] 합성 이력 {int(cut.sum())} / 전체 {len(df)}')
    src = os.path.dirname(os.path.abspath(__file__))
    for step, script, args in [('fit', 'fit.py', ['hist.csv', 'model']),
                               ('analyze', 'analyze.py', ['model', 'all.csv', 'out.csv'])]:
        print(f'\n===== {step} =====')
        r = subprocess.run([sys.executable, os.path.join(src, script)] + args,
                           capture_output=True, text=True)
        print(r.stdout.strip()[-1200:])
        if r.returncode != 0:
            print(r.stderr.strip()[-2000:])
            print(f'[selftest] {step} 실패')
            return 1
    o = pd.read_csv('out.csv')
    n = int((o['APPLIED'].astype(str) == 'Y').sum())
    print(f'\n[selftest] 통과. 보정 적용 {n}/{len(o)}건')
    print('[selftest] 산출 컬럼:', [c for c in o.columns if c in
                                  ('CORRECTED_EXPOSE', 'DELTA_PCT', 'GAP_D', 'APPLIED')])
    return 0


if __name__ == '__main__':
    sys.exit(main())
