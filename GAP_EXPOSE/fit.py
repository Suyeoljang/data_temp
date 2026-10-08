"""
학습 — 릿지 가중치 + 캘리브레이션 테이블 적합

  python fit.py "data/history_*.csv" model/

  초기 구축 시 1회, 이후 분기 1회 재학습.
  산출물: model/model.json (가중치·캘리브레이션), model/state.npz (EMA 상태)

  ※ 일상 운영에서는 실행하지 않는다. 운영은 main.py 하나로 돌아간다.
"""
import glob
import sys
import pandas as pd
from rule_engine import RuleEngine


def read_csv_glob(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(pattern)
    df = pd.concat([pd.read_csv(f, dtype=str, low_memory=False) for f in files],
                   ignore_index=True)
    print(f'[fit] {len(files)} files -> {df.shape}')
    return df


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    RuleEngine().fit(read_csv_glob(sys.argv[1])).save(sys.argv[2])
    return 0


if __name__ == '__main__':
    sys.exit(main())
