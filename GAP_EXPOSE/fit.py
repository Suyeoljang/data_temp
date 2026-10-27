"""
학습 — 릿지 가중치 + 캘리브레이션 테이블 적합

  python fit.py "data/history_*.csv" model/

  v3.1: 학습 끝에 β 상태와 적용 임계값(APPLY_RATIO)을 정해 함께 저장한다.
        마지막 일 배치(07:00 이후)는 제외되므로, 교체 전 daily_update --from <state_until> 로 공백을 채울 것.

  초기 구축 시 1회, 이후 월 1회 재학습 권장 (v3 검증: 월별 재학습이 고정 대비 +0.1%p).
  산출물: model/model.json (가중치·캘리브레이션·버전), model/state.npz (EMA + v3 추가 상태)
  학습 방식은 config.USE_V3 / FIT_MODE 로 결정된다 (기본 v3/pass = 순증 최대화).

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
    if 'MATI_ID' in df.columns and 'MAT_ID' not in df.columns:
        df = df.rename(columns={'MATI_ID': 'MAT_ID'})
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
