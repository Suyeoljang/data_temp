"""
PHOTO R2R Normal APC EXPOSE 보정 — RULE 엔진 설정 (v3)

v3 변경 요약 (2026-10)
  - 적용 범위: 간만진행(GAP≥30) 전용 → 전 GAP (GAP_THRESHOLD = 0). 이력 없는 신규 KEY 만 제외
  - 신호: 18축 x 8척도 EMA 144개 + 추가 22개 (KEY 직전 이력, L3 자기이력 EMA, 레벨 drift)
  - 가중치: 순증(구제-손실) 최대화. FIT_MODE='pass' (pass율 직접 최적화)
  - 재현·학습: 운영과 같은 일 배치 의미(DAY_BOUNDARY_HOUR)로 상태 갱신

v3.1 변경 (2026-10-14)
  - 적용 장비: APPLY_EQ_PREFIX = ['PTRD'] (DUV). 학습·상태 갱신은 전 장비 이력 사용
  - β 배율: 세그먼트별(장비 → 장비+레이어 → 제품+경로 → KEY) 과거 "보정량 대비 실제 오차" 비율을
            일 배치로 추정해 보정량에 곱한다 (BETA_*)
  - 적용 비율: |β·보정량| 상위 APPLY_RATIO(0.75)만 적용. 임계값은 fit 시 정해 model.json 에 저장
  - 검증(2025 학습 → 2026-01~07, DUV 219,918 Lot): 순증 +4,537 → +5,359, 노광량 변경 Lot 97% → 79%

v3.1.1 ~ v3.1.4 (2026-10-15 ~ 10-20)
  - 장기 GAP(GAP_THRESHOLD = 30) 파일럿 지원. β 는 전 GAP Lot 으로 학습(BETA_GAP_MIN = 0)
  - 보정 배율 APPLY_SCALE. 운영 의사결정은 이 값 하나로 한다 (아래 표)
  - 평가: 3구간(합격 ≤PASS_U / 회색 ≤GRAY_U / 불량). 결정 지표는 GRAY_U(2%) 초과 = 확실한 불량 기준의 구제·손실
  - 학습 목표 FIT_PASS_BAND = 2.0 (2% 기준 순증 직접 최적화)
  - 파라미터 탐색(2026-10-20): 배율 0.5 / 0.65 / 0.8 / 1.0 어디서도 나머지 파라미터는 현재 값이 최적.
    배율별로 다른 값을 쓸 필요 없음. β 는 필수(끄면 순증 -10 ~ -13%)

  배율별 예상 (2025 학습 → 2026-01~07, DUV GAP≥30 38,639 Lot, 2% 기준)
    APPLY_SCALE  구제   손실  순증    손실률  구제/손실
       1.0       2,391   956  +1,435   3.1%    2.50     효과 최대
       0.8       2,024   697  +1,327   2.3%    2.90
       0.65      1,710   555  +1,155   1.8%    3.08     손실 = 예전 v2 수준
       0.5       1,366   407    +959   1.3%    3.36

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
CD_SPEC = 0.01        # (참고용) 합격 기준 |CD_ERROR| <= 1%. 평가는 아래 PASS_U / GRAY_U 를 쓴다

# ---------- 평가 기준 (3구간, 단위 = CD 오차 %) ----------
# 합격 : |CD 오차| <= PASS_U
# 회색 : PASS_U < |CD 오차| <= GRAY_U   (측정 오차 범위 — 합격·불량 어느 쪽으로도 단정하지 않음)
# 불량 : |CD 오차| >  GRAY_U
# 판정·지표는 metrics.py. analyze / make_results / viz 가 함께 쓴다.
# GRAY_U 는 계측 반복 오차(Gauge R&R)로 확정할 것. 학습 목표(±1%)는 바꾸지 않는다(바꿔도 보정값이 거의 같음, 검증됨).
PASS_U = 1.0
GRAY_U = 2.0

# 무효 측정 경계 — |CD_ERROR| 가 이보다 크면 계측 누락·오입력으로 보고 제외.
# ※ 1~3% 수준의 CD_ERROR 게이팅(선택편향, 성능 절반)과는 다르다. 이것은 물리적으로
#   불가능한 값(예: MEAS_CD=0 이면 CD_ERROR=100%)을 걸러내는 데이터 유효성 검사이며,
#   분석에 쓴 V2 데이터의 최댓값은 15.4% 라 이 경계로 제외되는 행이 없었다.
MAX_CD_REL_ERR = 0.30

# ---------- 적용 범위 ----------
# v3: 전 GAP 적용. 2026-01~07 홀드아웃에서 모든 GAP 구간의 순증이 양수였다.
#     30.0 으로 두면 v2 와 같이 간만진행에만 적용한다.
GAP_THRESHOLD = 30.0  # 이 값 이상(일)인 Lot 에만 보정 적용 (GAP 결측 = 신규 KEY 는 항상 제외). 파일럿 = 장기 GAP 30일
# 적용 장비: PROC_EQ 가 이 접두어로 시작하는 장비에만 보정을 적용한다 (PTRD = DUV / DUV_TWIN).
# 학습·상태 갱신은 전 장비 이력을 그대로 쓴다(신호 품질). [] 또는 None 이면 전 장비 적용.
APPLY_EQ_PREFIX = ['PTRD']
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

# ---------- v3 추가 신호 ----------
USE_V3 = True         # False 면 v2(144 신호, 간만진행 학습) 동작
# 레벨 drift: log IDEAL = B[KEY] + D[그룹]. KEY 마지막 진행 이후 그룹 수준 변화를 신호로 쓴다
LEVEL_GROUPS = {
    'EQ':    ['PROC_EQ'],
    'EQRD':  ['PROC_EQ', 'ROUTE_DESC'],
    'DEVRT': ['DEV_ID', 'ROUTE'],
    'RET':   ['RETICLE'],
    'PR':    ['PR_ID'],
    'TRK':   ['TRACK_RECIPE'],
    'EQTRK': ['PROC_EQ', 'TRACK_RECIPE'],
}
LEVEL_HLS = [3.0, 14.0]
LEVEL_CLIP = 0.05         # 레벨 잔차 clip (log)
LEVEL_KEY_ALPHA = 0.3     # KEY 기준선 갱신률
LEVEL_MIN_ALPHA = 0.002
# L3 자기이력 EMA: DEV+ROUTE+EQ 조합의 IDEAL EMA (BIAS_MIX8 의 블렌딩 항을 신호로 흡수)
L3_COLS = ['DEV_ID', 'ROUTE', 'PROC_EQ']
L3_HLS = [21.0, 180.0]
EXTRA_CLIP = 5.0          # 추가 신호(정규화 후) clip
FEAT_CLIP = 8.0           # 전체 신호(정규화 후) clip

# 일 배치 경계. 재현·학습에서 같은 "일"의 Lot 은 전일 배치까지의 상태만 본다 (daily_update 와 동일)
DAY_BOUNDARY_HOUR = 7

# ---------- 가중치 적합 ----------
# FIT_MODE
#   'pass'  : 순증(구제-손실) 최대화. 비음수 릿지로 초기화 후 pass율(|CD오차|<=1%) 직접 최적화,
#             출력은 선형(캘리브 스케일 1) + clip ±CLAMP_PASS          ← 권장 (순증 최대)
#   'ridge' : 비음수 릿지 + 구간 캘리브레이션(FIT_LAMBDA 로 손실 가중)   ← 손실률 우선 시
FIT_MODE = 'pass'
FIT_GAP_FULL = 14.0       # 이 GAP 이상은 전부, 미만은 FIT_SHORT_SAMPLE 비율만 학습에 사용
FIT_SHORT_SAMPLE = 0.25
NN_RIDGE_LAM = 0.1        # 비음수 릿지 L2 (144 EMA 신호는 >=0, 추가 신호는 부호 자유)
Y_CLIP = 5.0              # 릿지 목표값(u) clip
# 학습 목표 기준(CD%). 1.0 = ±1% 합격 순증 최대화, 2.0 = 불량(>2%) 기준 순증 최대화.
# 2% 를 확실한 불량으로 보는 평가(GRAY_U=2)에서는 2.0 이 유리하다
# (2025 학습 → 2026 DUV GAP>=30: 2% 기준 순증 +1,398 → +1,439, 손실 1,069 → 1,003. 2025 Q4 검증도 +3%)
FIT_PASS_BAND = 2.0
PASS_TAU = 0.3            # pass율 surrogate 평활도
PASS_LAM = 1.0            # 릿지 해로부터의 이탈 페널티
PASS_MAXITER = 300
CLAMP_PASS = 2.5

# v2 경로(USE_V3=False 또는 FIT_MODE='ridge' 의 캘리브) 설정
RIDGE_L2 = 0.02
RIDGE_MAXITER = 500
FIT_LAMBDA = 1.0      # 캘리브레이션 손실가중. 1.0 = 순증(구제-손실) 기준. 클수록 보수적

# ---------- 보정 후처리 ----------
CLAMP_RTOL = 2.0      # |Δ| <= CLAMP_RTOL * RTOL  (CD 오차 2% 이동분)
CAL_BINS = [0.0, 0.2, 0.45, 0.8, 1.3, 2.0, 1e9]   # |Δ|/RTOL 구간

# 노광량 양자화 (EXPOSE_TYPE -> 최소단위). 없으면 QUANT_DEFAULT
QUANT = {'ASML': 0.01, 'ASML_TWIN': 0.01, 'SCANNER': 0.01, 'STEPPER': 0.02}
QUANT_DEFAULT = 0.01

# ---------- β 배율 (세그먼트별 보정 배율, v3.1) ----------
# 세그먼트마다 과거 "보정량 대비 실제 오차" 비율 β 를 일 배치로 추정해 보정량에 곱한다.
#   β_l = (Σw·s·u + K·β_상위) / (Σw·s² + K),  w = 0.5^(경과일/BETA_HL)
#   s = 엔진 원 보정량(CD% 단위), u = APC 원값 기준 CD 오차(CD% 단위)
# 계층은 위 → 아래 순서로 shrink 된다(이력이 적은 세그먼트는 상위 β 를 따른다).
# 검증(2025 학습 → 2026-01~07, DUV 219,918 Lot): 순증 +4,537 → +5,334 (+18%), 16개월 모두 개선
BETA_ON = True
BETA_LEVELS = [
    ['PROC_EQ'],
    ['PROC_EQ', 'ROUTE_DESC'],
    ['DEV_ID', 'ROUTE'],
    ['DEV_ID', 'PROC_EQ', 'ROUTE', 'RETICLE'],
]
BETA_HL = 60.0        # 반감기(일)
BETA_K = 20.0         # 상위 β 쪽으로 당기는 강도 (Lot 수 환산)
BETA_MIN = 0.0
BETA_MAX = 1.5
BETA_UCLIP = 5.0      # β 추정에 쓰는 u clip
# β 학습에 쓰는 Lot 의 GAP 하한. 적용 범위(GAP_THRESHOLD)와 별개다.
# GAP_THRESHOLD=30 이어도 0 으로 두면 같은 장비의 전 GAP Lot 으로 β 를 추정한다
# (검증: 장기 GAP 만으로 추정할 때와 순증은 같고 손실률이 낮다 — DUV 14.3% → 12.5%)
BETA_GAP_MIN = 0.0
BETA_EXCLUDE_CD_LOGIC = ['SC_IN_AI']   # β 갱신 제외 (평가 기준과 동일; FIX 는 SRC_EXCLUDE_CD_METHOD)

# ---------- 적용 비율 (작은 보정 생략) ----------
# fit 시 학습 구간 대상 Lot 의 |β·보정량| 분포에서 상위 APPLY_RATIO 만 적용되도록 임계값(CD%)을 정해
# model.json(apply_min_u)에 저장한다. 임계값 미만은 보정하지 않는다(APC 원값 사용, SKIP=SMALL_CORR).
# 0.75: 순증은 전부 적용과 같고 노광량이 바뀌는 Lot 은 약 1/4 줄어든다 (DUV 검증, 임계값 약 0.075 CD%).
APPLY_RATIO = 0.75
APPLY_MIN_U = None    # 숫자를 주면 fit 결과 대신 이 임계값(CD%)을 쓴다. 0 = 전부 적용

# ---------- 보정 배율 (손실 억제) ----------
# 최종 보정량에 곱하는 배율. 1.0 = 엔진 추정치 그대로(순증 최대).
# 줄이면 구제·손실이 함께 줄고, 손실이 더 빨리 줄어 손실률이 크게 낮아진다.
# 적용 여부 판정(APPLY_RATIO 임계값)은 배율 적용 전 값으로 하므로 바꿔도 재학습이 필요 없다.
# 검증(2026, DUV GAP>=30, 적용 0.5): 1.0 → 순증 +1,544 / 손실률 10.7%,  0.5 → +1,045 / 4.6%,  0.3 → +695 / 2.5%

#APPLY_SCALE = 1.0
APPLY_SCALE = 0.65

# 리포트(analyze / make_results / viz)에 함께 보여줄 배율 시나리오. 결과에서 원 보정량을 복원해 근사 계산한다
SCALE_SCENARIOS = [0.5, 0.65, 0.8, 1.0]

# ---------- 시간 분할 (fit 시) ----------
WARM_MONTHS = 3       # 앞부분 워밍업 (상태 축적 전용, 적합 제외)
VAL_MONTHS = 2        # 뒤에서 이만큼을 캘리브레이션 적합용으로 사용 (FIT_MODE='ridge' 만)
