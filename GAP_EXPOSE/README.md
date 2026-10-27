# PHOTO R2R 간만진행 EXPOSE 보정 — RULE 엔진

ML 없이 EMA 계통편차만으로 간만진행(GAP ≥ 30일) Lot의 노광 에너지를 보정합니다.
계산요청/계산결과 테이블 기반으로 운영되며, 재학습 없이 일 1회 상태 갱신만으로 돕니다.

**성능** (홀드아웃 57,670건) — CD ≤1% 합격률 49.70% → 51.84% (+2.14%p), 순증 +1,232 Lot, 손실률 7.62%

---

## 1. 설치

```bash
pip install -r requirements.txt
python selftest.py          # 합성 데이터로 환경 확인 (DB·실데이터 불필요)
```

`numba`는 선택이지만 **강력히 권장**합니다. 없으면 학습이 수십 배 느려집니다
(70만 행 기준 수 분 → 수 시간). 미설치 시 자동 폴백하며 결과는 동일합니다.

---

## 2. 실행 — 운영에서 사람이 돌리는 것은 두 개뿐입니다

```bash
python fit.py "data/history_1y.csv" model/     # 구축 시 1회, 이후 분기 1회
python main.py                                  # 상시 (실시간 + 일 배치 자동)
```

| 스크립트 | 언제 | 빈도 |
|---|---|---|
| **`fit.py`** | 초기 구축, 재학습 | **분기 1회** |
| **`main.py`** | 상시 기동 | **계속** |
| ↳ 실시간 | 요청 들어올 때마다 | 자동 |
| ↳ 일 배치 | 매일 07:10 | 자동 |
| `analyze.py` | 섀도우 분석, 모델 비교 | 필요할 때만 |
| `make_results.py` | 원본 → RULE 결과 테이블 (도입 검토용) | 필요할 때만 |
| `viz.py` | 장비별 EMA 효과 시각화 | 필요할 때만 |
| `selftest.py` | 설치 직후 환경 확인 | 1회 |

`main.py` 옵션

```bash
python main.py --dry          # 계산만, DB 기록 안 함 (섀도우)
python main.py --daily-now    # 기동 직후 일 배치 1회
python main.py --no-schedule  # 실시간만
```

단독 실행도 가능합니다.

```bash
python service.py --once
python daily_update.py --date 20260908
python daily_update.py --from 20260901070000 --to 20260908070000
```

---

## 3. 전체 그림

```
[분기 1회]  fit.py
              → model.json  릿지 가중치 144개 + 캘리브레이션 테이블
              → state.npz   EMA 상태 초기값

[상시]      main.py
              ├ 실시간 : CR2RPHTMEACAL 폴링
              │            └ GAP_D 판정 → EMA 조회 → 델타 → 클램프·캘리브
              │                └ CR2RPHTMEAAIC INSERT + 요청 DELETE (동일 트랜잭션)
              │
              └ 07:10  : IDB PHOTO_APC_DATA (어제 07:00 ~ 오늘 07:00)
                           └ 계산결과와 대조 → APC/USED 복원
                               └ 잔차 → 상태 갱신 → state.npz 교체
                                   └ correction_table.csv → 지표 출력
                                          ↑
                                    실시간이 이 상태를 씀
```

**학습과 상태 갱신은 다릅니다.**

| | 바뀌는 것 | 주기 |
|---|---|---|
| 상태 갱신 (일 배치) | EMA 숫자 (편차·신뢰도·산포) | 일 1회 |
| 학습 (`fit.py`) | 릿지 가중치, 캘리브레이션 | 분기 1회 |

일 배치는 "지금 각 축이 얼마나 치우쳤나"만 갱신합니다.
"어느 축을 얼마나 믿을지"(로직)는 학습에서만 바뀝니다.

---

## 4. 입력 데이터

### 계산요청 `CR2RPHTMEACAL`

```
LOT_ID, PROC_EQ, DEV_ID, ROUTE, RETICLE, TRACK_RECIPE, ROUTE_DESC,
MATI_ID, PROCESS_ID, CD_TYPE, PR_ID, CD_TARGET, MAIN_CONSTANT,
GAP_D, PROC_EXPOSE, LAST_PROC_TIME, REQ_TIME
```

**실시간 보정에 `MEAS_CD`는 필요 없습니다.** 측정값은 노광 이후에나 나오므로
애초에 존재하지 않고, 실시간에는 허용치 환산만 하면 됩니다.

```
RTOL = (MAIN_CONSTANT × CD_TARGET) / PROC_EXPOSE
```

`CD_TYPE` 도 부호 용도로는 안 쓰고 `EQ_CDTYPE` 축의 키 문자열로만 씁니다.
부호는 일 배치에서 이미 반영돼 EMA 상태에 들어 있습니다.

### 계산결과 `CR2RPHTMEAAIC`

```
LOT_ID, PROC_EQ, DEV_ID, ROUTE, RETICLE,
PROC_EXPOSE,   -- ★ NormalAPC 원값. 되먹임에 필수
CAL_EXPOSE,    -- 보정 후 값 (미적용 시 PROC_EXPOSE 와 동일)
GAP_D, LAST_PROC_TIME, REQ_TIME, CAL_TIME
```

### 진행이력 `PHOTO_APC_DATA` (IDB)

일 배치가 `PROC_TRANS_DATE` 로 조회합니다. 필요 컬럼은 `config_db.HIST_COLS` 참조.

---

## 5. 설정

### 접속

```python
# config_db.py
DB_HOST  = '10.147.1.112'   # MES  — 계산요청/결과
IDB_HOST = '10.148.1.133'   # IDB  — 진행이력
```

비밀번호는 환경변수 `R2R_DB_PASSWORD`, `R2R_IDB_PASSWORD` 로 넘기는 것을 권합니다.

### 먼저 확인할 것

```python
HIST_COLS                  # PHOTO_APC_DATA 실제 컬럼명과 대조
HIST_TIME_FORMAT           # 'STR'(yyyyMMddHHmmss) / 'DATE'
HIST_EXPOSE_MEANING        # 'AUTO' 권장 (아래 참조)
REQ_COLS / RES_COLS        # 컬럼명 매핑
SQL_KEY_SNAPSHOT           # 보정 테이블용. 진행이력 테이블명 교체
DAILY_HOUR, DAILY_MINUTE   # 기본 07:10
```

### ★ PROC_EXPOSE 의미 자동 판정

`PHOTO_APC_DATA.PROC_EXPOSE` 가 **실제 사용값**인지 **NormalAPC 원값**인지에 따라
잔차 계산이 완전히 달라집니다. 스키마만으로는 알 수 없어 자동 판정을 넣었습니다.

```
[daily] PROC_EXPOSE 의미 자동판정 = USED  (대조 143건, |Δ|중앙 USED 0.0000 / APC 0.0821)
```

보정이 실제로 걸린 Lot 20건 이상을 계산결과와 대조해 판정합니다.
**도입 후 첫 며칠은 이 로그를 반드시 확인**하시고, 안정되면
`HIST_EXPOSE_MEANING` 을 `'USED'` 또는 `'APC'` 로 고정하십시오.

섀도우 기간에는 보정이 안 걸려 판정 근거가 없으므로 `USED` 로 가정합니다.

### 부호 규약

```python
# config.py
SIGN_SPACE = +1.0
SIGN_LINE  = -1.0
```

`IDEAL = PROC + s·(CD_TARGET − MEAS_CD)·100·MAIN_CONSTANT`

틀리면 `fit.py` 실행 시 IDEAL 역산 불일치로 **예외를 던집니다.** 조용히
잘못되는 일은 없으니, 예외가 나면 부호를 뒤집어 다시 시도하십시오.

### 감사 컬럼 추가 권장

```sql
ALTER TABLE CR2RPHTMEAAIC ADD (
  DELTA_PCT    NUMBER,
  BIAS_MIX     NUMBER,
  APPLIED      CHAR(1),
  SKIP_REASON  VARCHAR2(20),   -- SHORT_GAP/NO_HIST/NO_STATE/INVALID_MC/CALC_ERROR
  CLAMPED      CHAR(1),
  MODEL_VER    VARCHAR2(20),
  STATE_VER    VARCHAR2(20)
);
```

추가 후 `RES_EXTRA_ENABLED = True`. 없어도 동작하지만 도입 초기 원인 추적에는
사실상 필수입니다.

---

## 6. 동작 방식

### 실시간

```
1) GAP_D 판정        요청 테이블 제공값 사용. <30일 / NULL → 보정 없음
2) EMA 상태 조회      18축 × 8척도 = 144개 값
                     값 = state × w_eff/(w_eff+m),  w_eff = weight × exp(-ln2/120 × Δt)
3) 델타 계산          dn = Σ(가중치 × 값/RTOL)
4) 클램프            dn = clip(dn, ±2.0)          CD 오차 2% 이동분
5) 캘리브레이션       구간별 스케일 곱
6) 양자화            round(PROC × (1+Δ) / q) × q
```

### 일 배치

```
IDEAL = USED_EXPOSE + s·(CD_TARGET − MEAS_CD)·100·MC     물리적 정답
RRES  = (IDEAL − APC_EXPOSE) / APC_EXPOSE                ← APC 원값 기준
```

**잔차를 실제사용값 기준으로 계산하면 안 됩니다.** 보정이 잘 들수록 잔차가 0에
수렴해 편차 추정치가 죽고, 몇 주 만에 보정이 사라집니다.

`IDEAL` 은 우리가 무엇을 쐈든 같은 값입니다. 더 좋은 dose 를 쓰면 `USED` 가
커진 만큼 `(CD_TARGET − MEAS_CD)` 가 줄어 합이 불변이기 때문입니다.
덕분에 NormalAPC 의 `PROC_EXPOSE` 도 우리 보정에 오염되지 않습니다.

### 상태 갱신 대상

```
CD_METHOD != 'FIX' 인 전 Lot
```

| 구분 | 포함 | 이유 |
|---|:---:|---|
| 간만진행 (GAP≥30d) | ● | 당연 |
| **단기 (GAP<30d)** | **●** | **장비·공정 최신 상태의 주 정보원** |
| AI 연산 / MFG / Manual | ● | 검증 결과 포함이 유리 |
| `CD_METHOD='FIX'` | ✕ | 고정값이라 APC 제어를 반영 안 함 |

보정은 간만진행에만 적용하지만 **상태 갱신은 전체**로 해야 합니다.

### 엔진 교체

일 배치는 디스크에서 엔진을 새로 로드해 갱신한 뒤 참조만 바꿔치기합니다.
파이썬 참조 대입이 원자적이라 실시간이 멈추지 않고, 교체 순간의 요청은
이전/이후 중 하나의 일관된 상태를 봅니다. 배치 실패 시 기존 엔진을 유지합니다.

### 누락 보정

프로세스가 꺼져 07:10 을 놓치면 기동 시 `model/last_daily.json` 을 보고
누락 날짜를 **순서대로** 따라잡습니다(최대 7일). 시간순이어야 EMA 감쇠가 맞습니다.

---

## 7. 안전 동작

| 상황 | 처리 |
|---|---|
| 계산 실패·DB 오류·모델 없음 | **CAL_EXPOSE = PROC_EXPOSE 로 통과.** 요청을 막지 않음 |
| 다건 INSERT 실패 | 자동 건별 재시도 |
| 일 배치 데이터 이상 | 점검(건수·평균 잔차·산포·극단값) 미통과 시 **상태 미갱신** |
| 일 배치 실패 | 기존 엔진 유지 |
| 롤백 | `state_YYYYMMDD.npz` 30일 보관, 복사로 즉시 복원 |
| 보정량 이상 | `±2·RTOL` 클램프 (TEST 발동률 0.17%) |

---

## 8. 튜닝 파라미터

| 항목 | 기본값 | 설명 |
|---|---|---|
| `GAP_THRESHOLD` | 30일 | 90일로 올리면 순증 60% 감소(검증됨). 30일 권장 |
| `FIT_LAMBDA` | 1.3 | 손실 가중. 클수록 보수적(손실↓ 순증↓) |
| `ROBUST_K` | 2.0 | 편차를 축별 산포의 K배로 제한 |
| `CLIP` | ±0.35 | 관측 윈저화 |
| `SHRINK_M` | 12.0 | 유효표본 축소 |
| `CLAMP_RTOL` | 2.0 | 최대 보정량 = CD 오차 2% 이동분 |
| `RIDGE_L2` | 0.02 | 가중치 정규화 |

손실을 줄이려면 `GAP_THRESHOLD` 를 올리는 것보다 **`FIT_LAMBDA` 를 올리는 편이
효율적**입니다(검증됨).

---

## 9. 하면 안 되는 것

| 금지 | 결과 |
|---|---|
| 잔차를 `USED_EXPOSE` 기준으로 계산 | 피드백 루프 붕괴, 보정이 0으로 수렴 |
| `CD_ERROR` 로 상태갱신 대상 필터링 | 추정 대상 크기로 선별하는 선택편향, 성능 절반 |
| 간만진행 Lot 만으로 상태 갱신 | 단기 Lot 이 최신 상태 정보원이라 성능 하락 |
| 합격 판정에 `+1e-12` 추가 | `\|u\|` 가 정확히 1인 Lot 이 다수라 합격률 0.3%p 부풀려짐 |
| 일 배치를 시간순 아닌 순서로 처리 | EMA 시간감쇠가 어긋나 상태 오염 |

이상치 방어는 `CLIP`(윈저화)과 `ROBUST_K`(축별 적응 클리핑)가 담당합니다.

---

## 10. 도입 절차

**1단계 — 학습**

```bash
python fit.py "history_1y.csv" model/
```

간만진행 적합 2,000건 + 캘리브레이션 1,000건이 최소입니다. 부족하면 명시적
예외를 냅니다. 통상 12개월 이상 이력이 필요합니다.

**2단계 — 섀도우 (2~4주)**

```bash
python main.py --dry
```

보정값을 계산만 하고 DB에 넣지 않습니다. `analyze.py` 로 반사실 성능을 확인해
분석 결과(순증 +1,232, 손실률 7.6%)와 비슷한지 대조하십시오.

**3단계 — 제한 적용** — 특정 설비군부터. 미적용 대조군 유지

**4단계 — 전면 적용**

---

## 11. 모니터링

| 지표 | 정상 범위 | 이탈 시 |
|---|---|---|
| 적용률 | 학습 시점과 ±5%p | KEY 분포 변화 점검 |
| 보정량 중앙값 | 0.2~0.5% | 급변 시 상태 오염 의심 |
| 클램프 발동률 | < 1% | 초과 시 상태 재구축 검토 |
| **CD ≤1% (적용군)** | 대조군 대비 +2%p 이상 | 미달 시 재적합 |
| **손실률** | < 10% | 초과 시 `FIT_LAMBDA` 상향 |
| `NO_STATE` 비율 | < 5% | 초과 시 축 커버리지 점검 |

일 배치 로그가 이 지표를 출력합니다. 별도로 시간 단위 모니터링을 권합니다.

### 롤백 기준

- 2주 연속 적용군 합격률이 대조군 이하
- 클램프 발동률 5% 초과
- 특정 설비에서 합격률 3%p 이상 악화

---

## 12. 파일 구성

| 파일 | 역할 |
|---|---|
| `main.py` | **통합 실행 — 실시간 + 일 배치 스케줄러** |
| `fit.py` | 학습 (구축·분기 재학습) |
| `config.py` | 축 18개, 반감기 8개, 하이퍼파라미터, 부호 규약 |
| `config_db.py` | MES/IDB 접속, 테이블·컬럼 매핑, SQL, 스케줄 |
| `rule_engine.py` | `RuleState`(EMA 상태), `RuleEngine`(학습·보정) |
| `fast.py` | numba 배치 커널 (미설치 시 자동 폴백) |
| `db.py` | 커넥션 풀 2개, 요청 조회, 결과 INSERT, 요청 DELETE |
| `service.py` | 실시간 폴링 (`main.py` 가 내부 호출) |
| `daily_update.py` | 일 배치 (`main.py` 가 내부 호출) |
| `analyze.py` | 과거 이력 반사실 평가 — 인과적 재현 (운영 미사용) |
| `make_results.py` | 원본 → RULE 결과 테이블, 기존 방식과 매칭 (운영 미사용) |
| `viz.py` | 장비별 EMA 효과 HTML 리포트, 방식 비교 (운영 미사용) |
| `selftest.py` | 합성 데이터 환경 점검 |

### systemd 등록

```ini
[Unit]
Description=PHOTO R2R Rule Correction
After=network.target

[Service]
User=r2r
WorkingDirectory=/opt/r2r_rule
ExecStart=/usr/bin/python3 main.py
Restart=always
RestartSec=10
Environment=R2R_DB_PASSWORD=...
Environment=R2R_IDB_PASSWORD=...

[Install]
WantedBy=multi-user.target
```

---

## 13. 참고 — KEY 가 4개 컬럼이라는 점

운영 KEY 는 `DEV_ID + PROC_EQ + ROUTE + RETICLE` 입니다.
같은 제품·레이어라도 레티클이 다르면 별도 이력이 되므로

- 간만진행 비중이 늘어납니다
- **`NO_HIST`(첫 등장) 비중도 함께 늘어납니다**

도입 전 1년치로 `GAP_D` 분포를 확인하고, `NO_HIST` 가 지나치게 많으면
GAP 산출 키에서만 RETICLE 을 빼는 것을 검토하십시오.

---

## 14. 효과 분석과 시각화

### 반사실 평가 — `analyze.py`

```bash
python analyze.py model/ photo_apc_data.csv analyzed.csv --since 20260201
```

과거 이력에 보정을 걸었다면 구제/손실/순증이 어땠을지 계산합니다.

**인과적 재현을 씁니다.** 각 Lot 은 자기 진행 시점까지의 이력만으로 보정됩니다
(운영과 동일한 query → update 순서). 학습 종료 시점의 최종 상태로 과거 Lot 을
조회하면 미래 정보가 섞여 결과가 왜곡됩니다 — 합성 데이터 검증에서 같은 입력이
−0.28%p 와 +0.87%p 로 갈렸습니다.

- 입력에는 **평가 구간 앞 3개월 이상의 이력**이 함께 있어야 상태가 쌓입니다
- 학습에 쓴 기간은 가중치가 in-sample 이므로 `--since` 로 **학습 이후 구간만** 평가하십시오

### 장비별 시각화 — `viz.py`

```bash
# 장비 1대
python viz.py analyzed.csv --prepared --eq PTRD24 --since 20260201 --out PTRD24.html

# 전체 장비 요약 + 장비별 리포트
python viz.py analyzed.csv --prepared --all --since 20260201 --out reports/

# analyze 를 거치지 않고 바로 (내부에서 인과적 재현 수행)
python viz.py photo_apc_data.csv --model model/ --eq PTRD24 --out PTRD24.html
```

리포트 구성 (간만진행 대상)

| 구성 | 보여주는 것 |
|---|---|
| 요약 카드 | 합격률 전후, 순증, 손실률, 계통편향, \|u\| 평균 |
| ① 시간축 산점도 | u 의 시간 추이, ±1 합격 밴드, 이동 중앙값 — 드리프트를 EMA 가 따라가는가 |
| ② GAP축 비교 | 간격 구간별 \|u\| 중앙값·합격률 — 오래 쉴수록 EMA 효과가 커지는가 |
| ③ 분포 | u 히스토그램 — 밴드 안으로 모이는가 |
| ④ 구제/손실 2×2 | 최종 판정 근거 |
| ⑤ KEY 드릴다운 | KEY 하나의 전체 이력을 **mJ 원값**으로. 녹색 막대 = 그 Lot 의 합격 구간 |
| ⑦ 보상량 | 보정률·보정량(mJ) 요약 카드, 시간축·분포·GAP별 보정률 |
| Lot 별 시뮬레이션 CSV | `<리포트명>_simulation.csv` — 리포트에서 링크로 내려받기 |
| 전체 장비 대비 | 합격률 변화 막대, 해당 장비 강조 |

**왜 노광량 원값이 아니라 u 인가** — 한 장비에 여러 제품·레이어가 섞여 돌아
노광량이 30~50 mJ 대로 흩어지므로, 보정 차이(1~2%)가 제품 간 차이에 묻힙니다.
`u = (IDEAL − 사용값)/(MAIN_CONSTANT × CD_TARGET)` 는 CD 오차(%)와 같아 제품이
섞여도 한 축에서 비교되고 합격선이 ±1 로 고정됩니다. 반대로 KEY 하나(⑤)는
제품·레이어가 고정이라 원값이 더 직관적입니다.

**해석 주의** — 장비 1대의 간만진행은 월 수십~수백 건이라 짧은 기간으로 쪼개면
흔들립니다. N 이 100 미만인 장비는 참고만 하십시오. 판단은 전체 장비 요약에서
시작해 개선폭이 유독 작은 장비를 ①~⑤ 로 파고드는 순서를 권합니다.

### 지난 분석 산출물로 보기 — `--longgap`

```bash
python viz.py "apc_ema_longgap_part*.csv" --longgap --all --out reports/
python viz.py "apc_ema_longgap_part*.csv" --longgap --eq PTRD24 --out PTRD24.html
python viz.py "apc_ema_longgap_part*.csv" --longgap --all --since 20260201 --out reports_holdout/
```

`apc_ema_longgap.csv`(분할 파일 그대로 가능)를 읽어 같은 리포트를 만듭니다.
**이 파일의 `EMA_EXPOSE` 는 지난 분석의 기존 방식(L3_EMA + BIAS_MIX8) 값**이며
리포트 제목·범례에 방식 이름이 표시됩니다(`--label` 로 변경 가능).

- 간만진행 행만 들어 있으므로 KEY 드릴다운(⑤)에는 단기 이력이 나오지 않습니다
- 평가 대상은 `CD_METHOD != 'FIX'`, `CD_LOGIC != 'SC_IN_AI'`, `GAP_DAYS ≥ 30`,
  `MAIN_CONSTANT > 0`, `IDEAL_EXPOSE`·`EMA_EXPOSE` 존재 행입니다
- 전체 기간으로 돌리면 지난 보고서와 같은 모집단(134,291건)이 나와야 하며,
  요약 카드의 노광량 상대오차 평균이 2.561% → 2.317% 로 재현되면 정상입니다
- 기존 방식은 전 기간 데이터로 튜닝됐으므로 `--since 20260201` 로 뒷부분만 보면
  in-sample 효과를 덜어낸 모습을 볼 수 있습니다

`--all` 은 `plotly.min.js` 를 한 번만 저장하고 공유하므로 사내망 오프라인에서도
열립니다(폴더째 복사). 단일 리포트는 JS 를 내장해 파일 하나로 열립니다.

---

## 15. 도입 검토 — RULE 결과 테이블 만들고 기존 방식과 비교

```bash
# 1) 원본으로 RULE 결과 테이블 생성 (+ 기존 방식 값 매칭)
python make_results.py "data/*_CD_V2_part*.csv" \
       --old "apc_ema_longgap_part*.csv" --out rule_longgap.csv

# 2) 리포트
python viz.py rule_longgap.csv --longgap --all --since 20260201 --out reports/
python viz.py rule_longgap.csv --longgap --eq PTRD24 --since 20260201 --out PTRD24.html
```

### make_results.py 가 하는 일

1. 원본 병합 (분할 파일 glob)
2. **cutoff(기본 2026-02-01) 이전 이력으로만 학습** — 이후는 완전 홀드아웃
3. 전 기간 **인과적 재현** (각 Lot 은 자기 진행 시점까지의 이력만 사용)
4. 간만진행 행만 저장. `PHASE` 컬럼으로 `TRAIN` / `TEST` 구분
5. `--old` 를 주면 기존 방식 `EMA_EXPOSE` 를 `EMA_EXPOSE_OLD` 로 붙임
   (LOT_ID + ROUTE + PROC_EQ + PROC_TRANS_DATE 매칭, 매칭률 출력)

출력은 `apc_ema_longgap.csv` 와 같은 형식이라 `viz.py --longgap` 에 그대로 들어가고,
`EMA_MODE='RULE_ROBUST_EMA'` 로 방식이 자동 표기됩니다.

실행 후 콘솔에 TRAIN / TEST 별 합격률·순증·손실률·상대오차가, `--old` 를 주면
**같은 Lot 기준** 두 방식 비교가 함께 나옵니다. **TEST 가 정직한 성능**입니다.

### GAP 산출 키

```bash
--key-cols DEV_ID,ROUTE,PROC_EQ              # 기본 — 분석·기존 산출물과 동일
--key-cols DEV_ID,PROC_EQ,ROUTE,RETICLE      # 운영 KEY
```

기존 방식과 비교할 때는 기본값(3컬럼)으로 두어야 같은 간만진행 모집단이 됩니다.
운영 도입 시의 모습을 보려면 4컬럼으로 한 번 더 돌려보십시오.

### 비교 리포트 구성

결과 CSV 에 `EMA_EXPOSE_OLD` 가 있으면 자동으로 추가됩니다.

| 위치 | 내용 |
|---|---|
| 전체 요약 | 방식 비교표 (합격률·구제·손실·순증·손실률·계통편향), 장비별 두 방식 막대 |
| 장비 리포트 | 방식 비교표, ⑥ GAP 구간별 합격률 PROC / 기존 / RULE |

### 기대 수치 (분석 때 결과, 간만진행 홀드아웃)

| | CD≤1% 변화 | 손실률 |
|---|---:|---:|
| 기존 방식 | +2.86%p | 16.2% |
| RULE 엔진 | +2.14%p | 7.6% |

두 수치는 모집단이 약간 다릅니다(기존 61,589건 / RULE 57,670건). make_results.py 는 같은 Lot 으로 맞춰 비교하므로 그 결과를 기준으로 판단하십시오.

실데이터로 돌린 결과가 이와 크게 다르면 입력 컬럼이나 cutoff 설정을 점검하십시오.
기존 방식 값은 지난 분석 산출물 그대로이며, 계산 시 인과 순서를 지켰는지는
이 패키지에서 검증할 수 없습니다.

---

## 16. 현업 FAQ

**Q. 그래프 Y값은 CD 오차(%)인가?**
크기는 같습니다(|Y| = |CD 오차 %|, ±1 밴드가 합격). 부호는 노광량 기준으로
+ 는 노광량 부족, − 는 과다입니다. SPACE·LINE 이 섞여 있어 CD 부호(크다/작다)로
그리면 같은 원인이 반대 방향으로 찍히기 때문입니다.

**Q. 실제 보상치는 얼마인가?**
⑦ 보상량 패널과 시뮬레이션 CSV 의 `보정량(mJ)`, `보정률(%)` 입니다.
그래프에서 같은 Lot 의 주황 점과 파란 점 세로 간격이 CD 환산 보상량입니다.

**Q. EMA 를 적용했을 때 CD 오차를 시뮬레이션할 수 있나?**
Lot 별 시뮬레이션 CSV 가 그 결과입니다.

```
EMA적용시CD = 실측CD + s × (EMA노광량 − APC노광량) / (100 × MAIN_CONSTANT)
s = +1 (SPACE) / −1 (LINE)
```

APC 와 같은 노광량-CD 관계를 쓰며, 보정 상한(CD 2% 이동분) 안에서 직선 근사가
성립합니다. 엑셀에서 바로 열리도록 UTF-8(BOM)으로 저장합니다.
`판정` 컬럼: 유지합격 / 구제 / 손실 / 계속탈락

**Q. EMA 는 어떻게 집계되나?**

기존 방식 (L3_EMA + BIAS_MIX8)
```
보정값 = [(1−w) × 현재 APC값 + w × L3_EMA] × (1 + c × BIAS_MIX)
```
- L3_EMA: 그 Lot 의 L3 조합(제품+경로+장비) 고유의 과거 최적값. L3 마다 따로 보유
- BIAS_MIX: 8개 관점의 최근 계통편차 가중평균
  (장비+경로 39.5%, 장비+경로설명 17.8%, 장비+트랙 12.8%, 레티클 12.3%,
   경로설명 10.4%, 나머지 7.2%)
- w = 15~30%, c 는 GAP 이 길수록 커짐

RULE 엔진
- 18개 관점 × 8개 시간척도(1.75~224일) = 144개 값을 가중합
- L3 자체는 18개 중 하나로 비중이 작음 — 간만진행은 그 L3 이력이 낡은 것이 문제이므로
- 가중치는 1년치 데이터로 학습
