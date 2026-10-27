"""
통합 실행 — 실시간 보정 + 일 배치 스케줄링

  python main.py                  # 상시 실행 (권장)
  python main.py --dry            # 계산만, DB 기록 안 함
  python main.py --daily-now      # 기동 직후 일 배치 1회 실행

동작
  메인 스레드  : 계산요청 폴링 -> 보정 -> 계산결과 INSERT       (상시)
  스케줄 스레드 : 매일 07:10 일 배치 실행 -> 상태 갱신 -> 엔진 교체

엔진 교체 방식
  일 배치는 별도 엔진 객체에 상태를 갱신한 뒤, 참조만 바꿔치기한다.
  파이썬 참조 대입은 원자적이므로 실시간 처리가 멈추지 않고,
  교체 순간의 요청은 이전/이후 중 하나의 일관된 상태를 본다.

누락 보정
  프로세스가 꺼져 있어 07:10 을 놓쳤다면, 기동 시 마지막 실행일을 확인해
  누락된 날짜를 순서대로 따라잡는다(DAILY_CATCHUP).
"""
import argparse
import datetime as dt
import json
import os
import signal
import sys
import threading
import time
import traceback
import config_db as D
import db
import daily_update
from rule_engine import RuleEngine
from service import run_once, state_version

_stop = threading.Event()
_engine = None
_engine_lock = threading.Lock()
MARK = 'last_daily.json'


def _sig(*a):
    print('[main] 종료 신호 수신')
    _stop.set()


# ============================================================
# 마지막 일 배치 실행일 기록
# ============================================================
def mark_path(model_dir):
    return os.path.join(model_dir, MARK)


def read_mark(model_dir):
    p = mark_path(model_dir)
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p)).get('last_date')
    except Exception:
        return None


def write_mark(model_dir, datestr, rc):
    json.dump({'last_date': datestr, 'rc': rc,
               'at': dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')},
              open(mark_path(model_dir), 'w'), indent=1)


# ============================================================
# 일 배치 실행 + 엔진 교체
# ============================================================
def run_daily(model_dir, base_date=None, tag=''):
    global _engine
    label = base_date or dt.datetime.now().strftime('%Y%m%d')
    print(f'\n[main] ===== 일 배치 시작 {label} {tag} =====')
    t0 = time.time()
    try:
        # 현재 엔진을 건드리지 않도록 디스크에서 새로 로드해 갱신
        fresh = RuleEngine.load(model_dir)
        rc, eng = daily_update.run(model_dir=model_dir, base_date=base_date, engine=fresh)
        if rc == 0:
            with _engine_lock:
                _engine = eng                      # 참조 교체 (원자적)
            print(f'[main] 엔진 교체 완료 (state_ver={state_version(model_dir)})')
        else:
            print(f'[main] 일 배치 rc={rc} — 엔진 유지')
        write_mark(model_dir, label, rc)
    except Exception:
        traceback.print_exc()
        rc = 3
    print(f'[main] ===== 일 배치 종료 rc={rc} ({time.time()-t0:.1f}s) =====\n')
    return rc


def catchup(model_dir):
    """누락일 따라잡기. 최대 7일치."""
    last = read_mark(model_dir)
    today = dt.datetime.now()
    # 오늘 07:10 이전이면 오늘분은 아직 대상이 아님
    due = today.date() if today.hour * 60 + today.minute >= D.DAILY_HOUR * 60 + D.DAILY_MINUTE \
        else (today - dt.timedelta(days=1)).date()
    if last is None:
        print('[main] 최초 기동 — 누락 보정 생략 (필요 시 --daily-now)')
        write_mark(model_dir, due.strftime('%Y%m%d'), -1)
        return
    d = dt.datetime.strptime(last, '%Y%m%d').date() + dt.timedelta(days=1)
    miss = []
    while d <= due and len(miss) < 7:
        miss.append(d.strftime('%Y%m%d'))
        d += dt.timedelta(days=1)
    if not miss:
        return
    print(f'[main] 누락 {len(miss)}일 감지: {miss[0]} ~ {miss[-1]}')
    for ds in miss:
        if _stop.is_set():
            break
        run_daily(model_dir, base_date=ds, tag='(누락보정)')


# ============================================================
# 스케줄 스레드
# ============================================================
def scheduler(model_dir):
    while not _stop.is_set():
        now = dt.datetime.now()
        nxt = now.replace(hour=D.DAILY_HOUR, minute=D.DAILY_MINUTE,
                          second=0, microsecond=0)
        if nxt <= now:
            nxt += dt.timedelta(days=1)
        wait = (nxt - now).total_seconds()
        print(f'[sched] 다음 일 배치 {nxt:%Y-%m-%d %H:%M} ({wait/3600:.1f}h 후)')
        if _stop.wait(wait):
            break
        run_daily(model_dir)


# ============================================================
# 메인
# ============================================================
def main():
    global _engine
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=D.MODEL_DIR)
    ap.add_argument('--dry', action='store_true', help='DB 기록 없이 계산만')
    ap.add_argument('--daily-now', action='store_true', help='기동 직후 일 배치 1회')
    ap.add_argument('--no-schedule', action='store_true', help='실시간만 실행')
    a = ap.parse_args()

    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)

    _engine = RuleEngine.load(a.model)
    print(f'[main] 모델 로드 {a.model} '
          f'(축 {len(_engine.axes)}개, state_ver={state_version(a.model)})')

    if not a.no_schedule:
        if D.DAILY_CATCHUP:
            catchup(a.model)
        if a.daily_now or D.DAILY_ON_START:
            run_daily(a.model)
        threading.Thread(target=scheduler, args=(a.model,),
                         daemon=True, name='sched').start()

    total = applied = 0
    t_log = time.time()
    while not _stop.is_set():
        try:
            with _engine_lock:
                eng = _engine
            n, na = run_once(eng, state_version(a.model), dry=a.dry)
            total += n
            applied += na
            if n == 0:
                _stop.wait(D.POLL_INTERVAL)
        except Exception:
            traceback.print_exc()
            _stop.wait(5)
            continue
        if time.time() - t_log > 300:
            rate = applied / total * 100 if total else 0
            print(f'[main] 누적 {total}건 처리, 보정 {applied}건 ({rate:.1f}%)')
            t_log = time.time()

    print(f'[main] 종료. 총 {total}건 처리, 보정 {applied}건')
    db.close_pool()
    return 0


if __name__ == '__main__':
    sys.exit(main())
