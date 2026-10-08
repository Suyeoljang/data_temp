"""
PHOTO R2R 간만진행 EXPOSE 보정 — RULE 엔진 설정

이식 시 반드시 확인할 것
--------------------------------
1) 부호 규약 : FAB1 은 CAL = PROC − MEAS. 다른 라인은 반대일 수 있음.
                SIGN_SPACE / SIGN_LINE 로 조정하고, fit 단계에서 IDEAL 역산식이
                실데이터와 일치하는지 자동 검증됨(불일치 시 예외 발생).
2) 릿지 가중치 : 아래 값은 FAB1 데이터로 적합한 것. 다른 라인에서는
                반드시 fit_weights() 로 재적합할 것. 그대로 쓰면 성능 보장 안 됨.
3) MAIN_CONSTANT = 0 인 행은 IDEAL 계산 불가 → 상태갱신·평가 모두 제외.
"""

# ---------- 입력 컬럼 ----------
COL_TIME = 'PROC_TRANS_DATE'          # yyyymmddHHMMSS
COL_NUM = ['CD_TARGET', 'MEAS_CD', 'PROC_EXPOSE', 'MAIN_CONSTANT']
KEY_COLS = ['DEV_ID', 'PROC_EQ', 'ROUTE', 'RETICLE']   # 운영 KEY (GAP 산출용)

# ---------- 물리 규약 ----------
SIGN_SPACE = +1.0     # IDEAL = PROC + s*(CD_TARGET - MEAS_CD)*100*MAIN_CONSTANT
SIGN_LINE = -1.0
CD_SPEC = 0.01        # 합격 기준 |CD_ERROR| <= 1%

# 무효 측정 경계 — |CD_ERROR| 가 이보다 크면 계측 누락·오입력으로 보고 제외.
# ※ 1~3% 수준의 CD_ERROR 게이팅(선택편향, 성능 절반)과는 다르다. 이것은 물리적으로
#   불가능한 값(예: MEAS_CD=0 이면 CD_ERROR=100%)을 걸러내는 데이터 유효성 검사이며,
#   분석에 쓴 V2 데이터의 최댓값은 15.4% 라 이 경계로 제외되는 행이 없었다.
MAX_CD_REL_ERR = 0.30

# ---------- 적용 범위 ----------
GAP_THRESHOLD = 30.0  # 이 값 이상(일)인 Lot 에만 보정 적용
SRC_EXCLUDE_CD_METHOD = ['FIX']       # 상태 갱신에서 제외할 CD_METHOD
                                       # SC_IN_AI / MFG / Manual 은 포함이 유리(검증됨)

# ---------- EMA 추적 ----------
CLIP = 0.35           # 관측 윈저화 (상대잔차). CD_ERROR 기반 게이팅은 금지(선택편향)
HL_STALE = 120.0      # 미관측 기간 신뢰도 감쇠 half-life (days)
SHRINK_M = 12.0       # 유효표본 축소 계수
ROBUST_K = 2.0        # 편차를 축별 산포의 K배로 제한 (0 이면 비활성)

# 축 정의: name -> 컬럼 리스트
# ★ 계산요청 테이블(CR2RPHTMEACAL)에 실제로 있는 컬럼만으로 구성했다.
#   요청에 없는 컬럼(TECH, MAIN_PROCESS, RETICLE_SUBFIX, ROUTE_PREFIX 등)을
#   쓰려면 요청 테이블에 추가하거나 마스터 조인으로 채워야 한다.
#   축을 바꾸면 반드시 가중치를 재적합할 것.
AXES = {
    'EQ_ROUTE':  ['PROC_EQ', 'ROUTE'],        # 최대 비중
    'RETICLE':   ['RETICLE'],
    'DEV_ROUTE': ['DEV_ID', 'ROUTE'],
    'TRK':       ['TRACK_RECIPE'],
    'EQ_RDESC':  ['PROC_EQ', 'ROUTE_DESC'],
    'EQ_TRK':    ['PROC_EQ', 'TRACK_RECIPE'],
    'ROUTE':     ['ROUTE'],
    'RDESC':     ['ROUTE_DESC'],
    'EQ':        ['PROC_EQ'],
    'KEY':       ['DEV_ID', 'PROC_EQ', 'ROUTE', 'RETICLE'],
    'EQ_PRID':   ['PROC_EQ', 'PR_ID'],
    'PR':        ['PR_ID'],
    'EQ_PROCID': ['PROC_EQ', 'PROCESS_ID'],
    'PROCID':    ['PROCESS_ID'],
    'EQ_MAT':    ['PROC_EQ', 'MAT_ID'],
    'MAT':       ['MAT_ID'],
    'DEV_EQ':    ['DEV_ID', 'PROC_EQ'],
    'EQ_CDTYPE': ['PROC_EQ', 'CD_TYPE'],
}
# 축별 반감기(일). 드리프트는 단일 시간척도가 아니므로 여러 척도를 동시에 씀
HALF_LIVES = [1.75, 3.5, 7.0, 14.0, 28.0, 56.0, 112.0, 224.0]

# 컬럼이 없는 축은 자동으로 건너뜀 (라인마다 보유 컬럼이 다를 수 있음)
SKIP_MISSING_AXES = True

# ---------- 릿지 적합 ----------
RIDGE_L2 = 0.02
RIDGE_MAXITER = 500
FIT_LAMBDA = 1.3      # 캘리브레이션 손실가중. 클수록 보수적(손실↓ 순증↓)

# ---------- 보정 후처리 ----------
CLAMP_RTOL = 2.0      # |Δ| <= CLAMP_RTOL * RTOL  (CD 오차 2% 이동분)
CAL_BINS = [0.0, 0.2, 0.45, 0.8, 1.3, 2.0, 1e9]   # |Δ|/RTOL 구간

# 노광량 양자화 (EXPOSE_TYPE -> 최소단위). 없으면 QUANT_DEFAULT
QUANT = {'ASML': 0.01, 'ASML_TWIN': 0.01, 'SCANNER': 0.01, 'STEPPER': 0.02}
QUANT_DEFAULT = 0.01

# ---------- 시간 분할 (fit 시) ----------
WARM_MONTHS = 3       # 앞부분 워밍업 (상태 축적 전용, 적합 제외)
VAL_MONTHS = 2        # 뒤에서 이만큼을 캘리브레이션 적합용으로 사용
