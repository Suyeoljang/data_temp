"""
DB 연동 설정 — 운영 환경에 맞게 이 파일만 수정하면 된다.
"""
import os

# ============================================================
# 접속 정보 (운영에서는 환경변수 사용 권장)
# ============================================================
DB_USER = os.getenv('R2R_DB_USER', 'mesmgr')
DB_PASSWORD = os.getenv('R2R_DB_PASSWORD', 'mesmgr')
DB_HOST = os.getenv('R2R_DB_HOST', '10.147.1.112')
DB_PORT = int(os.getenv('R2R_DB_PORT', '1521'))
DB_SERVICE = os.getenv('R2R_DB_SERVICE', 'mesdb')

POOL_MIN = 1
POOL_MAX = 4
POOL_TIMEOUT = 60          # 초. 커넥션 획득 대기

# ------------------------------------------------------------
# IDB — 진행이력(PHOTO_APC_DATA) 조회 전용. 일 배치에서만 사용
# ------------------------------------------------------------
IDB_USER = os.getenv('R2R_IDB_USER', 'idbmgr')
IDB_PASSWORD = os.getenv('R2R_IDB_PASSWORD', 'idbmgr')
IDB_HOST = os.getenv('R2R_IDB_HOST', '10.148.1.133')
IDB_PORT = int(os.getenv('R2R_IDB_PORT', '1720'))
IDB_SERVICE = os.getenv('R2R_IDB_SERVICE', 'IDB')

# ============================================================
# 테이블명
# ============================================================
TB_REQUEST = 'CR2RPHTMEACAL'      # 계산요청
TB_RESULT = 'CR2RPHTMEAAIC'       # 계산결과

# ============================================================
# 컬럼명 매핑
#   좌변(논리명) : 우변(실제 DB 컬럼명)
#   운영 스키마와 다르면 우변만 고치면 된다.
# ============================================================
REQ_COLS = {
    'LOT_ID':         'LOT_ID',
    'PROC_EQ':        'PROC_EQ',
    'DEV_ID':         'DEV_ID',
    'ROUTE':          'ROUTE',          # = LPT
    'RETICLE':        'RETICLE',
    'TRACK_RECIPE':   'TRACK_RECIPE',
    'ROUTE_DESC':     'ROUTE_DESC',
    'MAT_ID':         'MATI_ID',        # 요청 테이블 실제 컬럼명
    'PROCESS_ID':     'PROCESS_ID',
    'CD_TYPE':        'CD_TYPE',
    'PR_ID':          'PR_ID',
    'CD_TARGET':      'CD_TARGET',
    'MAIN_CONSTANT':  'MAIN_CONSTANT',
    'GAP_D':          'GAP_D',
    'PROC_EXPOSE':    'PROC_EXPOSE',
    'LAST_PROC_TIME': 'LAST_PROC_TIME',  # KEY 최종진행시간
    'REQ_TIME':       'REQ_TIME',        # 계산요청시간
}

RES_COLS = {
    'LOT_ID':         'LOT_ID',
    'PROC_EQ':        'PROC_EQ',
    'DEV_ID':         'DEV_ID',
    'ROUTE':          'ROUTE',
    'RETICLE':        'RETICLE',
    'PROC_EXPOSE':    'PROC_EXPOSE',
    'CAL_EXPOSE':     'CAL_EXPOSE',      # 보정 후 값
    'GAP_D':          'GAP_D',
    'LAST_PROC_TIME': 'LAST_PROC_TIME',
    'REQ_TIME':       'REQ_TIME',
    'CAL_TIME':       'CAL_TIME',        # 계산완료시간
}

# 감사용 확장 컬럼. 테이블에 추가했다면 True 로 바꾸면 자동 기록된다.
# 권장: 도입 초기 원인 추적에 사실상 필수
RES_EXTRA_ENABLED = False
RES_EXTRA_COLS = {
    'DELTA_PCT':   'DELTA_PCT',      # 보정률 (%)
    'BIAS_MIX':    'BIAS_MIX',       # 캘리브레이션 전 원시 보정률
    'APPLIED':     'APPLIED',        # Y / N
    'SKIP_REASON': 'SKIP_REASON',    # 미적용 사유
    'CLAMPED':     'CLAMPED',        # Y / N
    'MODEL_VER':   'MODEL_VER',
    'STATE_VER':   'STATE_VER',
}

# ============================================================
# 서비스 동작
# ============================================================
POLL_INTERVAL = 2.0        # 초. 요청 테이블 폴링 주기
BATCH_SIZE = 50            # 1회 처리 건수
DELETE_AFTER_INSERT = True  # 계산완료 후 요청행 삭제
MODEL_DIR = os.getenv('R2R_MODEL_DIR', 'model')
MODEL_VER = 'v1.0'

# 노광량 양자화. EXPOSE_TYPE 이 요청에 없으므로 장비 접두어로 판정
QUANT_BY_EQ_PREFIX = {
    'PTRA': 0.01,
    'PTRD': 0.01,
}
QUANT_FALLBACK = 0.01

# ============================================================
# 일 배치
# ============================================================
DAILY_HOUR = 7             # 매일 07:10 실행
DAILY_MINUTE = 10
DAILY_WINDOW_HOUR = 7      # 어제 07:00 <= PROC_TRANS_DATE < 오늘 07:00
DAILY_ON_START = False     # 기동 직후 1회 실행 여부
DAILY_CATCHUP = True       # 누락일 감지 시 자동 보정 실행

# PHOTO_APC_DATA.PROC_EXPOSE 가 무엇을 담는가
#   'USED' : 실제 노광에 사용된 값 (보정 적용분이면 CAL_EXPOSE)
#   'APC'  : NormalAPC 원값 (보정 전)
#   'AUTO' : 계산결과 테이블과 대조해 자동 판정 (권장, 최초 며칠은 로그 확인)
HIST_EXPOSE_MEANING = 'AUTO'

# PHOTO_APC_DATA 시간 컬럼 형식
#   'STR' : 'yyyyMMddHHmmss' 문자열   /   'DATE' : DATE 타입
HIST_TIME_FORMAT = 'STR'

# IDB 진행이력 조회. PROC_TRANS_DATE 가 문자열이므로 문자열 비교
SQL_HIST_STR = """
SELECT *
FROM   PHOTO_APC_DATA
WHERE  PROC_TRANS_DATE >= :dt_from
  AND  PROC_TRANS_DATE <  :dt_to
ORDER  BY PROC_TRANS_DATE
"""
SQL_HIST_DATE = """
SELECT *
FROM   PHOTO_APC_DATA
WHERE  PROC_TRANS_DATE >= TO_DATE(:dt_from, 'YYYYMMDDHH24MISS')
  AND  PROC_TRANS_DATE <  TO_DATE(:dt_to,   'YYYYMMDDHH24MISS')
ORDER  BY PROC_TRANS_DATE
"""

# 같은 기간 우리 계산결과 (APC 원값 / 보정값 복원용). MES DB 조회
SQL_RESULT_WINDOW = f"""
SELECT {{lot}} AS LOT_ID, {{route}} AS ROUTE,
       {{proc}} AS RES_PROC_EXPOSE, {{cal}} AS RES_CAL_EXPOSE
FROM   {TB_RESULT}
WHERE  {{caltime}} >= TO_DATE(:dt_from, 'YYYYMMDDHH24MISS') - 2
  AND  {{caltime}} <  TO_DATE(:dt_to,   'YYYYMMDDHH24MISS') + 1
"""

# PHOTO_APC_DATA 컬럼명 매핑 (실제 스키마와 다르면 우변만 수정)
HIST_COLS = {
    'LOT_ID':        'LOT_ID',
    'PROC_TRANS_DATE': 'PROC_TRANS_DATE',
    'PROC_EQ':       'PROC_EQ',
    'DEV_ID':        'DEV_ID',
    'ROUTE':         'ROUTE',
    'RETICLE':       'RETICLE',
    'TRACK_RECIPE':  'TRACK_RECIPE',
    'ROUTE_DESC':    'ROUTE_DESC',
    'MAT_ID':        'MAT_ID',
    'PROCESS_ID':    'PROCESS_ID',
    'CD_TYPE':       'CD_TYPE',
    'PR_ID':         'PR_ID',
    'CD_TARGET':     'CD_TARGET',
    'MEAS_CD':       'MEAS_CD',
    'MAIN_CONSTANT': 'MAIN_CONSTANT',
    'PROC_EXPOSE':   'PROC_EXPOSE',
    'CD_METHOD':     'CD_METHOD',
    'CD_LOGIC':      'CD_LOGIC',
}

# 보정 테이블(CSV) 생성용 — KEY 목록 + 대표 축 값
SQL_KEY_SNAPSHOT = """
SELECT  DEV_ID, PROC_EQ, ROUTE, RETICLE,
        MAX(PROC_TRANS_DATE) AS LAST_PROC_TIME,
        COUNT(*)             AS N_OBS,
        MAX(TRACK_RECIPE) KEEP (DENSE_RANK LAST ORDER BY PROC_TRANS_DATE) AS TRACK_RECIPE,
        MAX(ROUTE_DESC)   KEEP (DENSE_RANK LAST ORDER BY PROC_TRANS_DATE) AS ROUTE_DESC,
        MAX(MAT_ID)       KEEP (DENSE_RANK LAST ORDER BY PROC_TRANS_DATE) AS MAT_ID,
        MAX(PROCESS_ID)   KEEP (DENSE_RANK LAST ORDER BY PROC_TRANS_DATE) AS PROCESS_ID,
        MAX(PR_ID)        KEEP (DENSE_RANK LAST ORDER BY PROC_TRANS_DATE) AS PR_ID,
        MAX(CD_TYPE)      KEEP (DENSE_RANK LAST ORDER BY PROC_TRANS_DATE) AS CD_TYPE,
        MAX(CD_TARGET)    KEEP (DENSE_RANK LAST ORDER BY PROC_TRANS_DATE) AS CD_TARGET,
        MAX(MAIN_CONSTANT)KEEP (DENSE_RANK LAST ORDER BY PROC_TRANS_DATE) AS MAIN_CONSTANT,
        MAX(PROC_EXPOSE)  KEEP (DENSE_RANK LAST ORDER BY PROC_TRANS_DATE) AS PROC_EXPOSE
FROM    진행이력테이블
WHERE   PROC_TRANS_DATE >= ADD_MONTHS(SYSDATE, -18)
GROUP   BY DEV_ID, PROC_EQ, ROUTE, RETICLE
"""
