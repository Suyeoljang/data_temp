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
"""
import argparse
import sys
import numpy as np
import pandas as pd
import config as C
from rule_engine import RuleEngine, derive


def prepare(csv_in, model_dir):
    """보정 재현 결과 DataFrame 반환 (viz.py 에서도 사용)"""
    raw = pd.read_csv(csv_in, dtype=str, low_memory=False)
    if 'GAP_D' in raw.columns:
        raw['_GAP_IN'] = raw['GAP_D']
    df = derive(raw, verify_ideal=False)
    if '_GAP_IN' in df.columns:
        df['GAP_D'] = pd.to_numeric(df['_GAP_IN'], errors='coerce')
        df['LONG'] = np.isfinite(df['GAP_D'].values) & (df['GAP_D'].values >= C.GAP_THRESHOLD)
        print('[analyze] 입력 CSV 의 GAP_D 사용')
    eng = RuleEngine.load(model_dir)
    print(f'[analyze] 인과적 재현 중 ({len(df)}행, 간만진행 {int((df.LONG & df.VALID).sum())}건)')
    delta = eng.replay(df)

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


def summary(df, since=None):
    m = df['EVAL'] & df['LONG']
    if since:
        m &= df['PROC_TRANS_DATE'] >= pd.Timestamp(since)
    if not m.any():
        print('[analyze] 평가 대상 없음')
        return
    b = df.loc[m, 'U_PROC'].abs() <= 1.0
    a = df.loc[m, 'U_EMA'].abs() <= 1.0
    g, l = int((~b & a).sum()), int((b & ~a).sum())
    print(f'[analyze] 간만진행 평가 {int(m.sum())}건'
          + (f' (since {since})' if since else ''))
    print(f'  CD<=1%  {b.mean()*100:.2f}% -> {a.mean()*100:.2f}% ({(a.mean()-b.mean())*100:+.3f}%p)')
    print(f'  구제 {g} / 손실 {l} / 순증 {g-l:+d} / 손실률 {l/max(b.sum(),1)*100:.2f}%')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model'); ap.add_argument('csv_in'); ap.add_argument('csv_out')
    ap.add_argument('--since', help='평가 시작일 YYYYMMDD (학습 이후 구간 권장)')
    a = ap.parse_args()
    df = prepare(a.csv_in, a.model)
    cols = [c for c in df.columns if not c.startswith('_')]
    df[cols].to_csv(a.csv_out, index=False)
    summary(df, a.since)
    return 0


if __name__ == '__main__':
    sys.exit(main())
