"""
실시간 보정 서비스

  python service.py            # 상시 폴링
  python service.py --once     # 1회만 처리 (cron/테스트용)
  python service.py --dry      # 계산만 하고 DB 기록 안 함 (섀도우 점검)

동작
  계산요청 테이블 폴링 -> 보정 계산 -> 계산결과 INSERT -> 요청행 DELETE

원칙
  보정은 부가 기능이다. 어떤 실패가 나도 요청을 막지 않고
  CAL_EXPOSE = PROC_EXPOSE (원값) 로 통과시킨다.
"""
import argparse
import datetime as dt
import os
import signal
import sys
import time
import traceback
import numpy as np
import config as C
import config_db as D
import db
from rule_engine import RuleEngine

_stop = False


def _sigterm(*a):
    global _stop
    _stop = True
    print('[svc] 종료 신호 수신, 현재 배치 마무리 후 정지')


def quant_of(proc_eq):
    for pfx, q in D.QUANT_BY_EQ_PREFIX.items():
        if str(proc_eq).startswith(pfx):
            return q
    return D.QUANT_FALLBACK


def to_days(ts):
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return float(ts)
    return ts.timestamp() / 86400.0


def compute(engine, req):
    """
    요청 1건 보정.
    반환: (cal_expose, meta)
      meta = dict(delta_pct, bias_mix, applied, skip_reason, clamped)
    """
    proc = float(req['PROC_EXPOSE'])
    meta = dict(delta_pct=0.0, bias_mix=0.0, applied='N',
                skip_reason=None, clamped='N')

    # --- 1) 유효성 ---
    mc = req.get('MAIN_CONSTANT')
    tgt = req.get('CD_TARGET')
    if mc is None or tgt is None or float(mc) <= 0 or float(tgt) <= 0 or proc <= 0:
        meta['skip_reason'] = 'INVALID_MC'
        return proc, meta

    # --- 2) GAP 판정 (요청 테이블 제공값 사용) ---
    gap = req.get('GAP_D')
    if gap is None:
        meta['skip_reason'] = 'NO_HIST'
        return proc, meta
    gap = float(gap)
    if gap < C.GAP_THRESHOLD:
        meta['skip_reason'] = 'SHORT_GAP'
        return proc, meta

    # --- 3) 상태 조회 + 델타 ---
    t = to_days(req.get('REQ_TIME')) or (dt.datetime.now().timestamp() / 86400.0)
    rtol = float(mc) * float(tgt) / proc
    vals = engine.state.query(req, t)
    if not np.any(vals):
        meta['skip_reason'] = 'NO_STATE'
        return proc, meta

    dn = float(np.nan_to_num(vals / rtol, nan=0.0, posinf=0.0, neginf=0.0) @ engine.weights)
    meta['bias_mix'] = dn
    if abs(dn) > C.CLAMP_RTOL:
        meta['clamped'] = 'Y'
    dn = float(np.clip(dn, -C.CLAMP_RTOL, C.CLAMP_RTOL))

    bins = np.asarray(C.CAL_BINS)
    b = int(np.clip(np.digitize(abs(dn), bins) - 1, 0, len(engine.cal_scale) - 1))
    delta = engine.cal_scale[b] * dn * rtol

    q = quant_of(req['PROC_EQ'])
    cal = round(proc * (1 + delta) / q) * q
    meta.update(delta_pct=delta * 100.0, applied='Y')
    return cal, meta


def run_once(engine, state_ver, dry=False):
    reqs = db.fetch_requests()
    if not reqs:
        return 0, 0
    items, nap = [], 0
    for r in reqs:
        try:
            cal, meta = compute(engine, r)
        except Exception:
            traceback.print_exc()
            cal = float(r['PROC_EXPOSE'])
            meta = dict(delta_pct=0.0, bias_mix=0.0, applied='N',
                        skip_reason='CALC_ERROR', clamped='N')
        if meta['applied'] == 'Y':
            nap += 1
        items.append((r, cal, meta))
        if dry:
            print(f"  [dry] {r['LOT_ID']:>12s} GAP={r.get('GAP_D')} "
                  f"{r['PROC_EXPOSE']:.2f} -> {cal:.2f} "
                  f"({meta['delta_pct']:+.4f}%) {meta['applied']}"
                  f"{'/' + meta['skip_reason'] if meta['skip_reason'] else ''}")
    if dry:
        return len(items), nap
    ok, ng = db.insert_results_many(items, state_ver=state_ver)
    if ng:
        print(f'[svc] {ok}건 성공 / {ng}건 실패')
    return ok, nap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--once', action='store_true')
    ap.add_argument('--dry', action='store_true')
    ap.add_argument('--model', default=D.MODEL_DIR)
    a = ap.parse_args()

    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT, _sigterm)

    engine = RuleEngine.load(a.model)
    sv = state_version(a.model)
    print(f'[svc] 모델 로드 {a.model} (state_ver={sv}, 축 {len(engine.axes)}개)')

    total = applied = 0
    t0 = time.time()
    while not _stop:
        try:
            n, na = run_once(engine, sv, dry=a.dry)
            total += n
            applied += na
        except Exception:
            traceback.print_exc()
            time.sleep(5)
            continue
        if a.once:
            break
        if n == 0:
            time.sleep(D.POLL_INTERVAL)
        if time.time() - t0 > 300:
            rate = applied / total * 100 if total else 0
            print(f'[svc] 누적 {total}건 처리, 보정 적용 {applied}건 ({rate:.1f}%)')
            t0 = time.time()
    print(f'[svc] 종료. 총 {total}건 처리, 보정 {applied}건')
    db.close_pool()


def state_version(model_dir):
    p = os.path.join(model_dir, 'state.npz')
    if not os.path.exists(p):
        return ''
    return dt.datetime.fromtimestamp(os.path.getmtime(p)).strftime('%Y%m%d')


if __name__ == '__main__':
    sys.exit(main())
