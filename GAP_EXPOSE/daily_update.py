"""
일 배치 — IDB PHOTO_APC_DATA 기반 상태 갱신

  python daily_update.py                     # 어제 07:00 ~ 오늘 07:00
  python daily_update.py --date 20260908     # 해당일 07:00 기준 창
  python daily_update.py --from 20260901070000 --to 20260908070000

수행
  1) IDB PHOTO_APC_DATA 조회 (어제 07:00 <= PROC_TRANS_DATE < 오늘 07:00)
  2) 계산결과 테이블과 대조해 APC 원값 / 실제 사용값 복원
  3) 잔차 RRES = (IDEAL - APC_EXPOSE) / APC_EXPOSE          ← APC 원값 기준
  4) 데이터 점검 -> EMA 상태 갱신 -> state.npz + 날짜 스냅샷
  5) 보정 테이블 재생성, 지표 출력

★ 잔차를 실제사용값 기준으로 계산하면 보정이 0으로 수렴해 시스템이 죽는다.
"""
import argparse
import datetime as dt
import os
import shutil
import sys
import numpy as np
import pandas as pd
import config as C
import config_db as D
import db
from rule_engine import RuleEngine

TS = '%Y%m%d%H%M%S'


def window(base_date=None):
    """어제 07:00 <= t < 오늘 07:00. base_date 는 '오늘'에 해당하는 날짜."""
    d = dt.datetime.strptime(base_date, '%Y%m%d') if base_date else dt.datetime.now()
    to = d.replace(hour=D.DAILY_WINDOW_HOUR, minute=0, second=0, microsecond=0)
    if base_date is None and d < to:
        to -= dt.timedelta(days=1)
    return (to - dt.timedelta(days=1)).strftime(TS), to.strftime(TS)


def resolve_expose(hist, res):
    """PHOTO_APC_DATA.PROC_EXPOSE 의 의미를 판정하고 APC/USED 두 컬럼을 만든다."""
    h = hist.copy()
    h['PROC_EXPOSE'] = pd.to_numeric(h['PROC_EXPOSE'], errors='coerce')
    if len(res):
        for c in ['RES_PROC_EXPOSE', 'RES_CAL_EXPOSE']:
            res[c] = pd.to_numeric(res[c], errors='coerce')
        h = h.merge(res, on=['LOT_ID', 'ROUTE'], how='left')
    else:
        h['RES_PROC_EXPOSE'] = np.nan
        h['RES_CAL_EXPOSE'] = np.nan

    matched = h['RES_PROC_EXPOSE'].notna()
    diff = (matched & ((h['RES_CAL_EXPOSE'] - h['RES_PROC_EXPOSE']).abs() > 1e-9)).fillna(False)
    mode = D.HIST_EXPOSE_MEANING
    if mode == 'AUTO':
        if diff.sum() < 20:
            mode = 'USED'
            print(f'[daily] 보정 대조 {int(diff.sum())}건 — 판정 불가, USED 가정')
        else:
            du = (h.loc[diff, 'PROC_EXPOSE'] - h.loc[diff, 'RES_CAL_EXPOSE']).abs().median()
            da = (h.loc[diff, 'PROC_EXPOSE'] - h.loc[diff, 'RES_PROC_EXPOSE']).abs().median()
            mode = 'USED' if du < da else 'APC'
            print(f'[daily] PROC_EXPOSE 의미 자동판정 = {mode} '
                  f'(대조 {int(diff.sum())}건, |Δ|중앙 USED {du:.4f} / APC {da:.4f})')

    if mode == 'USED':
        h['USED_EXPOSE'] = h['PROC_EXPOSE']
        h['APC_EXPOSE'] = h['RES_PROC_EXPOSE'].fillna(h['PROC_EXPOSE'])
    else:
        h['APC_EXPOSE'] = h['PROC_EXPOSE']
        h['USED_EXPOSE'] = h['RES_CAL_EXPOSE'].fillna(h['PROC_EXPOSE'])
    h['CORRECTED'] = diff
    return h, mode


def compute_residual(df):
    for c in ['CD_TARGET', 'MEAS_CD', 'MAIN_CONSTANT', 'APC_EXPOSE', 'USED_EXPOSE']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    t = df['PROC_TRANS_DATE']
    if not pd.api.types.is_datetime64_any_dtype(t):
        t = pd.to_datetime(t.astype(str), format=TS, errors='coerce')
    df['PROC_TRANS_DATE'] = t
    df = df.sort_values('PROC_TRANS_DATE', kind='mergesort').reset_index(drop=True)

    s = np.where(df['CD_TYPE'].astype(str).str.upper().eq('SPACE'),
                 C.SIGN_SPACE, C.SIGN_LINE)
    df['IDEAL'] = df['USED_EXPOSE'] + s * (df['CD_TARGET'] - df['MEAS_CD']) \
        * 100 * df['MAIN_CONSTANT']
    df['RRES'] = (df['IDEAL'] - df['APC_EXPOSE']) / df['APC_EXPOSE']
    df['_T'] = df['PROC_TRANS_DATE'].values.astype('datetime64[s]').astype(np.int64) / 86400.0
    bad = (df['MAIN_CONSTANT'] <= 0) | (df['APC_EXPOSE'] <= 0) | df['RRES'].isna()
    if bad.any():
        print(f'[daily] 무효 {int(bad.sum())}건 제외')
    return df[~bad].reset_index(drop=True)


def sanity_check(df):
    r = df['RRES']
    msg = []
    if len(df) < 50:
        msg.append(f'건수 부족 {len(df)}')
    if abs(r.mean()) > 0.05:
        msg.append(f'평균 잔차 이상 {r.mean()*100:.2f}%')
    if r.std() > 0.20:
        msg.append(f'산포 이상 {r.std()*100:.2f}%')
    if (r.abs() > 0.30).mean() > 0.10:
        msg.append(f'극단값 과다 {(r.abs()>0.30).mean()*100:.1f}%')
    if msg:
        print('[daily] 데이터 점검 경고: ' + ' / '.join(msg))
        return False
    return True


def update_state(engine, df):
    src = ~df['CD_METHOD'].astype(str).str.upper().eq('FIX')
    KA = {}
    for a, cc in engine.state.axes.items():
        k = df[cc[0]].astype(str)
        for c in cc[1:]:
            k = k.str.cat(df[c].astype(str), sep='\x01')
        KA[a] = k.values
    T, R = df['_T'].values, df['RRES'].values
    n = 0
    for i in range(len(df)):
        if src.iat[i] and np.isfinite(R[i]):
            engine.state.update_keys({a: KA[a][i] for a in KA}, float(T[i]), float(R[i]))
            n += 1
    return n


def save_state(engine, model_dir):
    engine.save(model_dir)
    stamp = dt.datetime.now().strftime('%Y%m%d')
    shutil.copy2(os.path.join(model_dir, 'state.npz'),
                 os.path.join(model_dir, f'state_{stamp}.npz'))
    olds = sorted(f for f in os.listdir(model_dir)
                  if f.startswith('state_') and f.endswith('.npz'))
    for f in olds[:-30]:
        os.remove(os.path.join(model_dir, f))
    print(f'[daily] 상태 저장 · 스냅샷 state_{stamp}.npz')


def build_correction_table(engine, out_csv):
    from service import compute
    try:
        snap = db.fetch_key_snapshot()
    except Exception as ex:
        print(f'[daily] KEY 스냅샷 조회 실패(생략): {ex}')
        return
    if snap.empty:
        return
    now = dt.datetime.now()
    snap['LAST_PROC_TIME'] = pd.to_datetime(snap['LAST_PROC_TIME'])
    snap['GAP_D'] = (now - snap['LAST_PROC_TIME']).dt.total_seconds() / 86400.0
    a0 = next(iter(engine.state.axes))
    rows = []
    for r in snap.to_dict('records'):
        try:
            cal, m = compute(engine, dict(r, REQ_TIME=now))
        except Exception:
            continue
        i = engine.state.idx[a0].get(
            '\x01'.join(str(r.get(c)) for c in engine.state.axes[a0]), -1)
        rows.append(dict(
            DEV_ID=r['DEV_ID'], PROC_EQ=r['PROC_EQ'], ROUTE=r['ROUTE'],
            RETICLE=r['RETICLE'], LAST_PROC_TIME=r['LAST_PROC_TIME'],
            GAP_D=round(r['GAP_D'], 2), PROC_EXPOSE=r.get('PROC_EXPOSE'),
            BIAS_MIX=round(m['bias_mix'], 6), DELTA_PCT=round(m['delta_pct'], 4),
            CORRECTED_EXPOSE=cal, APPLIED=m['applied'], SKIP_REASON=m['skip_reason'],
            CONFIDENCE=round(float(engine.state.W[a0][i].max()), 3) if i >= 0 else 0.0,
            N_OBS=int(r.get('N_OBS', 0)),
            UPDATED_AT=now.strftime('%Y-%m-%d %H:%M:%S'), MODEL_VER=D.MODEL_VER))
    t = pd.DataFrame(rows)
    t.to_csv(out_csv, index=False)
    ap = int((t.APPLIED == 'Y').sum())
    print(f'[daily] 보정 테이블 {len(t)} KEY (적용 {ap}, {ap/max(len(t),1)*100:.1f}%)')


def report(df):
    tol = df['MAIN_CONSTANT'] * df['CD_TARGET']
    ok = (tol > 0).values
    cd_used = ((df['IDEAL'] - df['USED_EXPOSE']).abs() / tol).values[ok]
    cd_apc = ((df['IDEAL'] - df['APC_EXPOSE']).abs() / tol).values[ok]
    print(f'\n[daily] === 지표 ({len(df)}건) ===')
    print(f'  잔차 평균 {df.RRES.mean()*100:+.4f}%  중앙 {df.RRES.median()*100:+.4f}%  '
          f'산포 {df.RRES.std()*100:.4f}%')
    print(f'  CD<=1% (APC 원값)   {(cd_apc<=1).mean()*100:.2f}%')
    print(f'  CD<=1% (실제 사용)  {(cd_used<=1).mean()*100:.2f}%')
    c = df['CORRECTED'].values[ok] if 'CORRECTED' in df.columns else np.zeros(ok.sum(), bool)
    if c.any():
        w, n = cd_apc[c] <= 1, cd_used[c] <= 1
        print(f'  보정 적용 {int(c.sum())}건: {w.mean()*100:.2f}% -> {n.mean()*100:.2f}% '
              f'({(n.mean()-w.mean())*100:+.3f}%p)')
        print(f'    구제 {int((~w & n).sum())} / 손실 {int((w & ~n).sum())} '
              f'/ 순증 {int((~w & n).sum() - (w & ~n).sum()):+d}')
    if (~c).any():
        print(f'  미적용 {int((~c).sum())}건(대조군): {(cd_apc[~c]<=1).mean()*100:.2f}%')


def run(model_dir=None, dt_from=None, dt_to=None, base_date=None,
        skip_table=False, engine=None):
    """
    반환 (rc, engine).  rc: 0 정상 / 1 대상없음 / 2 점검경고 / 3 오류
    engine 을 넘기면 그 객체 상태를 갱신해 돌려준다(서비스 내 호출용).
    """
    model_dir = model_dir or D.MODEL_DIR
    if dt_from is None:
        dt_from, dt_to = window(base_date)
    print(f'[daily] 구간 {dt_from} ~ {dt_to}')
    try:
        hist = db.fetch_history(dt_from, dt_to)
    except Exception as ex:
        print(f'[daily] IDB 조회 실패: {ex}')
        return 3, engine
    if hist.empty:
        print('[daily] 대상 없음')
        return 1, engine
    print(f'[daily] PHOTO_APC_DATA {len(hist)}건')

    hist = hist.rename(columns={v: k for k, v in D.HIST_COLS.items() if v != k})
    res = db.fetch_result_window(dt_from, dt_to)
    hist, mode = resolve_expose(hist, res)
    df = compute_residual(hist)
    print(f'[daily] 유효 {len(df)}건 (PROC_EXPOSE 해석={mode})')
    if not sanity_check(df):
        print('[daily] 상태 미갱신 종료. 계측 확인 후 --from/--to 로 재처리')
        return 2, engine

    eng = engine or RuleEngine.load(model_dir)
    n = update_state(eng, df)
    print(f'[daily] 상태 갱신 {n}건 (FIX 제외)')
    save_state(eng, model_dir)
    report(df)
    if not skip_table:
        build_correction_table(eng, os.path.join(model_dir, 'correction_table.csv'))
    return 0, eng


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', help='기준일 YYYYMMDD')
    ap.add_argument('--from', dest='dfrom', help='YYYYMMDDHH24MISS')
    ap.add_argument('--to', dest='dto', help='YYYYMMDDHH24MISS')
    ap.add_argument('--model', default=D.MODEL_DIR)
    ap.add_argument('--skip-table', action='store_true')
    a = ap.parse_args()
    rc, _ = run(model_dir=a.model, dt_from=a.dfrom, dt_to=a.dto,
                base_date=a.date, skip_table=a.skip_table)
    db.close_pool()
    return rc


if __name__ == '__main__':
    sys.exit(main())
