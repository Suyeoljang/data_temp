"""
DB 접근 계층 — Oracle 연결, 요청 조회, 결과 INSERT, 요청 삭제.

oracledb (python-oracledb) 를 씁니다. thin 모드라 Oracle Client 설치가 필요 없습니다.
    pip install oracledb
"""
import datetime as dt
import oracledb
import config_db as D

_pool = None          # MES (계산요청/결과)
_pool_idb = None      # IDB (진행이력)


def get_pool():
    global _pool
    if _pool is None:
        dsn = oracledb.makedsn(D.DB_HOST, D.DB_PORT, service_name=D.DB_SERVICE)
        _pool = oracledb.create_pool(
            user=D.DB_USER, password=D.DB_PASSWORD, dsn=dsn,
            min=D.POOL_MIN, max=D.POOL_MAX, increment=1,
            getmode=oracledb.POOL_GETMODE_WAIT, timeout=D.POOL_TIMEOUT)
        print(f'[db] MES pool 생성 {D.DB_HOST}:{D.DB_PORT}/{D.DB_SERVICE}')
    return _pool


def get_pool_idb():
    """IDB(진행이력) 커넥션. 일 배치에서만 사용."""
    global _pool_idb
    if _pool_idb is None:
        dsn = oracledb.makedsn(D.IDB_HOST, D.IDB_PORT, service_name=D.IDB_SERVICE)
        _pool_idb = oracledb.create_pool(
            user=D.IDB_USER, password=D.IDB_PASSWORD, dsn=dsn,
            min=1, max=2, increment=1,
            getmode=oracledb.POOL_GETMODE_WAIT, timeout=D.POOL_TIMEOUT)
        print(f'[db] IDB pool 생성 {D.IDB_HOST}:{D.IDB_PORT}/{D.IDB_SERVICE}')
    return _pool_idb


def close_pool():
    global _pool, _pool_idb
    for p in ('_pool', '_pool_idb'):
        v = globals()[p]
        if v is not None:
            try:
                v.close()
            except Exception:
                pass
            globals()[p] = None


# ============================================================
# 계산요청 조회
# ============================================================
def fetch_requests(limit=None):
    """
    미처리 요청을 시간순으로 가져온다.
    계산결과에 이미 있는 LOT 은 제외(재처리 방지).
    반환: list[dict]  — 키는 REQ_COLS 의 논리명
    """
    limit = limit or D.BATCH_SIZE
    q = D.REQ_COLS
    sel = ', '.join(f'r.{v} AS {k}' for k, v in q.items())
    sql = f"""
        SELECT {sel}
        FROM   {D.TB_REQUEST} r
        WHERE  NOT EXISTS (
                 SELECT 1 FROM {D.TB_RESULT} a
                 WHERE  a.{D.RES_COLS['LOT_ID']} = r.{q['LOT_ID']}
                   AND  a.{D.RES_COLS['ROUTE']}  = r.{q['ROUTE']}
               )
        ORDER  BY r.{q['REQ_TIME']}
        FETCH  FIRST :lim ROWS ONLY
    """
    with get_pool().acquire() as con:
        cur = con.cursor()
        cur.execute(sql, lim=limit)
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, row)) for row in cur]


# ============================================================
# 계산결과 INSERT (+ 요청 삭제)
# ============================================================
def insert_result(req, cal_expose, *, delta_pct=0.0, bias_mix=0.0,
                  applied='N', skip_reason=None, clamped='N',
                  state_ver='', commit=True):
    """
    결과 1건 INSERT. DELETE_AFTER_INSERT 면 같은 트랜잭션에서 요청행 삭제.
    실패해도 요청을 막지 않도록 호출부에서 예외를 잡아 재시도하십시오.
    """
    r = D.RES_COLS
    q = D.REQ_COLS
    cols = [r['LOT_ID'], r['PROC_EQ'], r['DEV_ID'], r['ROUTE'], r['RETICLE'],
            r['PROC_EXPOSE'], r['CAL_EXPOSE'], r['GAP_D'],
            r['LAST_PROC_TIME'], r['REQ_TIME'], r['CAL_TIME']]
    binds = {
        'lot': req['LOT_ID'], 'eq': req['PROC_EQ'], 'dev': req['DEV_ID'],
        'rt': req['ROUTE'], 'ret': req['RETICLE'],
        'proc': float(req['PROC_EXPOSE']), 'cal': float(cal_expose),
        'gap': None if req.get('GAP_D') is None else float(req['GAP_D']),
        'last': req.get('LAST_PROC_TIME'), 'reqt': req.get('REQ_TIME'),
        'calt': dt.datetime.now(),
    }
    vals = [':lot', ':eq', ':dev', ':rt', ':ret',
            ':proc', ':cal', ':gap', ':last', ':reqt', ':calt']

    if D.RES_EXTRA_ENABLED:
        e = D.RES_EXTRA_COLS
        cols += [e['DELTA_PCT'], e['BIAS_MIX'], e['APPLIED'],
                 e['SKIP_REASON'], e['CLAMPED'], e['MODEL_VER'], e['STATE_VER']]
        vals += [':dpct', ':bmix', ':appl', ':srsn', ':clmp', ':mver', ':sver']
        binds.update(dpct=float(delta_pct), bmix=float(bias_mix),
                     appl=applied, srsn=skip_reason, clmp=clamped,
                     mver=D.MODEL_VER, sver=state_ver)

    sql_ins = (f"INSERT INTO {D.TB_RESULT} ({', '.join(cols)}) "
               f"VALUES ({', '.join(vals)})")
    sql_del = (f"DELETE FROM {D.TB_REQUEST} "
               f"WHERE {q['LOT_ID']} = :lot AND {q['ROUTE']} = :rt")

    with get_pool().acquire() as con:
        cur = con.cursor()
        cur.execute(sql_ins, binds)
        if D.DELETE_AFTER_INSERT:
            cur.execute(sql_del, lot=binds['lot'], rt=binds['rt'])
        if commit:
            con.commit()


def insert_results_many(items, state_ver=''):
    """
    다건 INSERT. items = [(req, cal_expose, meta_dict), ...]
    한 트랜잭션으로 묶어 처리하고, 실패 시 전체 롤백 후 건별 재시도.
    """
    try:
        with get_pool().acquire() as con:
            cur = con.cursor()
            for req, cal, meta in items:
                _exec_one(cur, req, cal, meta, state_ver)
            con.commit()
        return len(items), 0
    except Exception as ex:
        print(f'[db] 다건 실패 -> 건별 재시도: {ex}')
        ok = ng = 0
        for req, cal, meta in items:
            try:
                insert_result(req, cal, state_ver=state_ver, **meta)
                ok += 1
            except Exception as e2:
                ng += 1
                print(f'[db] LOT {req.get("LOT_ID")} 실패: {e2}')
        return ok, ng


def _exec_one(cur, req, cal, meta, state_ver):
    r, q = D.RES_COLS, D.REQ_COLS
    cols = [r['LOT_ID'], r['PROC_EQ'], r['DEV_ID'], r['ROUTE'], r['RETICLE'],
            r['PROC_EXPOSE'], r['CAL_EXPOSE'], r['GAP_D'],
            r['LAST_PROC_TIME'], r['REQ_TIME'], r['CAL_TIME']]
    vals = [':lot', ':eq', ':dev', ':rt', ':ret',
            ':proc', ':cal', ':gap', ':last', ':reqt', ':calt']
    binds = dict(lot=req['LOT_ID'], eq=req['PROC_EQ'], dev=req['DEV_ID'],
                 rt=req['ROUTE'], ret=req['RETICLE'],
                 proc=float(req['PROC_EXPOSE']), cal=float(cal),
                 gap=None if req.get('GAP_D') is None else float(req['GAP_D']),
                 last=req.get('LAST_PROC_TIME'), reqt=req.get('REQ_TIME'),
                 calt=dt.datetime.now())
    if D.RES_EXTRA_ENABLED:
        e = D.RES_EXTRA_COLS
        cols += [e['DELTA_PCT'], e['BIAS_MIX'], e['APPLIED'],
                 e['SKIP_REASON'], e['CLAMPED'], e['MODEL_VER'], e['STATE_VER']]
        vals += [':dpct', ':bmix', ':appl', ':srsn', ':clmp', ':mver', ':sver']
        binds.update(dpct=float(meta.get('delta_pct', 0.0)),
                     bmix=float(meta.get('bias_mix', 0.0)),
                     appl=meta.get('applied', 'N'),
                     srsn=meta.get('skip_reason'),
                     clmp=meta.get('clamped', 'N'),
                     mver=D.MODEL_VER, sver=state_ver)
    cur.execute(f"INSERT INTO {D.TB_RESULT} ({', '.join(cols)}) "
                f"VALUES ({', '.join(vals)})", binds)
    if D.DELETE_AFTER_INSERT:
        cur.execute(f"DELETE FROM {D.TB_REQUEST} "
                    f"WHERE {q['LOT_ID']} = :lot AND {q['ROUTE']} = :rt",
                    lot=binds['lot'], rt=binds['rt'])


# ============================================================
# 일 배치용 조회
# ============================================================
def fetch_dataframe(sql, pool=None, **binds):
    import pandas as pd
    with (pool or get_pool()).acquire() as con:
        cur = con.cursor()
        cur.execute(sql, binds)
        cols = [c[0] for c in cur.description]
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=cols)


def fetch_history(dt_from, dt_to):
    """
    IDB PHOTO_APC_DATA 조회.
    dt_from / dt_to : 'YYYYMMDDHH24MISS' 문자열
    """
    sql = D.SQL_HIST_STR if D.HIST_TIME_FORMAT == 'STR' else D.SQL_HIST_DATE
    return fetch_dataframe(sql, pool=get_pool_idb(), dt_from=dt_from, dt_to=dt_to)


def fetch_result_window(dt_from, dt_to):
    """같은 기간 우리 계산결과 (APC 원값/보정값 복원용). MES DB."""
    r = D.RES_COLS
    sql = D.SQL_RESULT_WINDOW.format(
        lot=r['LOT_ID'], route=r['ROUTE'],
        proc=r['PROC_EXPOSE'], cal=r['CAL_EXPOSE'], caltime=r['CAL_TIME'])
    try:
        return fetch_dataframe(sql, dt_from=dt_from, dt_to=dt_to)
    except Exception as ex:
        print(f'[db] 계산결과 조회 실패(무시): {ex}')
        import pandas as pd
        return pd.DataFrame(columns=['LOT_ID', 'ROUTE',
                                     'RES_PROC_EXPOSE', 'RES_CAL_EXPOSE'])


def fetch_key_snapshot():
    return fetch_dataframe(D.SQL_KEY_SNAPSHOT)
