"""
장비별 EMA 보정 효과 시각화 (보정 대상 = GAP >= config.GAP_THRESHOLD, v3 기본 전 GAP)

  # 장비 1대 리포트
  python viz.py data.csv --model model/ --eq PTRD24 --out report_PTRD24.html

  # 전체 장비 요약 + 장비별 리포트 일괄 생성
  python viz.py data.csv --model model/ --all --out reports/

  # 이미 analyze.py 로 만든 결과를 재사용 (재현 계산 생략)
  python viz.py analyzed.csv --prepared --eq PTRD24 --out report.html

  # 지난 분석 산출물 apc_ema_longgap.csv (분할 파일 그대로 가능)
  python viz.py "apc_ema_longgap_part*.csv" --longgap --all --out reports/
  python viz.py "apc_ema_longgap_part*.csv" --longgap --eq PTRD24 --out PTRD24.html

옵션
  --since 20260201   평가 시작일. 학습 이후 구간만 보는 것을 권장
  --key "DEV|ROUTE"  드릴다운할 KEY 지정 (기본: 간만진행이 가장 많은 KEY)
  --window 40        이동 중앙값 창 크기(건)
  --max-points 6000  산점도 1개에 그릴 최대 점 수 (0 = 전부). 파일 크기 조절용
                     · 요약 수치·막대·분포·이동중앙값은 항상 전체 Lot 으로 계산
                     · 시간 순 균등 간격 표본 (보정률 그림은 상위 1% 큰 보정 Lot 을 우선 포함)
                     · 전체 Lot 값은 *_simulation.csv 에 그대로 저장

입력
  PHOTO_APC_DATA 추출 CSV (평가 구간 앞 3개월 이상 이력 포함 권장)
  필수: PROC_TRANS_DATE, PROC_EQ, DEV_ID, ROUTE, RETICLE, CD_TYPE, CD_TARGET,
        MEAS_CD, MAIN_CONSTANT, PROC_EXPOSE, CD_METHOD + 축 컬럼들

구성 (장비 1대)
  0) 요약 지표          합격률 전후, 구제/손실/순증, 계통편향
     결정 지표 카드     2% 초과 = 확실한 불량 — 구제/손실/순증/손실률, 배율 시나리오 표
     3구간 카드         합격(<=PASS_U) / 회색(~GRAY_U, 측정 오차 범위) / 불량 — 확실한 구제·손실, 불량률, 3구간 점수
  1) 시간축 산점도      u = (IDEAL-사용값)/TOL, ±PASS_U 합격 · ±GRAY_U 회색 밴드, 이동 중앙값
  2) GAP축 비교         GAP 구간별 |u| 중앙값·합격률
  3) 분포 비교          u 히스토그램
  4) 구제/손실 2x2 + 3구간 이동표(3x3)
  5) KEY 드릴다운       KEY 하나의 전체 이력을 mJ 원값으로
  ⑦ 보상량              보정률 분포·추이
  ⑧ 효과        Lot 노이즈에 묻히지 않는 지표 — 개선/악화 비율과 평균 개선(t),
                        주별 계통편향·합격률, 큰 오차 비율, 보정 크기별 효과, GAP 구간별 95% 신뢰구간

  u 는 CD 오차(%)와 같다. |u|<=PASS_U(기본 1) 이면 합격. 기준은 config.PASS_U / GRAY_U.
  시뮬레이션 CSV 에 구간_실측 / 구간_EMA적용 / 판정_3구간 / 판정_2% 컬럼이 추가된다.
  v3.1.5: 결정 지표 카드(GRAY_U 초과 = 확실한 불량 기준 구제·손실·순증·손실률)와
          배율(APPLY_SCALE) 시나리오 표 추가. 결과 CSV 의 APPLY_SCALE 컬럼(없으면 --scale, config)을
          현재 배율로 보고, 보정량에 배율만 바꿔 곱한 근사로 SCALE_SCENARIOS 를 다시 센다.
  장비 전체는 제품이 섞여 있어 mJ 원값으로는 비교가 안 되므로 u 로 정규화하고,
  KEY 하나(제품·레이어 고정)에서만 mJ 원값을 쓴다.
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    sys.exit('plotly 가 필요합니다:  pip install plotly')

C_PROC = '#eb6834'
C_EMA = '#2a78d6'
C_IDEAL = '#888780'
C_BAND = 'rgba(29,158,117,0.12)'
EMA_LBL = 'EMA 보정'
import config as _C
import metrics as MT
PU, GU = MT.pass_u(), MT.gray_u()        # 합격 / 회색 상한 (CD 오차 %)
DS = MT.decision_spec()                  # 결정 지표 기준: 이 값 초과 = 확실한 불량
_ALL_BINS = [0, 1, 3, 7, 14, 30, 45, 60, 90, 180, 365, 1e9]
_ALL_LBL = ['<1', '1-3', '3-7', '7-14', '14-30', '30-45', '45-60', '60-90', '90-180', '180-365', '365+']
# 보정 대상 GAP 하한부터의 구간만 사용 (v3 기본 0 = 전 구간, v2 = 30 이상)
_i0 = max(0, sum(1 for b in _ALL_BINS[:-1] if b <= _C.GAP_THRESHOLD) - 1)
GAP_BINS = _ALL_BINS[_i0:]
GAP_LBL = _ALL_LBL[_i0:]
SCOPE_LBL = '간만진행(GAP≥30일)' if _C.GAP_THRESHOLD >= 30 else (
    f'GAP≥{_C.GAP_THRESHOLD:g}일' if _C.GAP_THRESHOLD > 0 else '전 GAP(이력 있는 Lot)')
if getattr(_C, 'APPLY_EQ_PREFIX', None):
    SCOPE_LBL += ' · 장비 ' + '/'.join(f'{p}*' for p in _C.APPLY_EQ_PREFIX)


def _eq_target(s):
    pf = tuple(getattr(_C, 'APPLY_EQ_PREFIX', None) or ())
    return s.astype(str).str.startswith(pf) if pf else pd.Series(True, index=s.index)


# ============================================================
# 데이터 준비
# ============================================================
def load(args):
    if args.prepared:
        df = pd.read_csv(args.csv, low_memory=False)
        df['PROC_TRANS_DATE'] = pd.to_datetime(df['PROC_TRANS_DATE'])
        for c in ['U_PROC', 'U_EMA', 'GAP_D', 'PROC_EXPOSE', 'EMA_EXPOSE',
                  'IDEAL_EXPOSE', 'TOL', 'CD_TARGET']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        for c in ['EVAL', 'LONG', 'VALID']:
            df[c] = df[c].astype(str).str.lower().isin(['true', '1'])
    else:
        from analyze import prepare
        df = prepare(args.csv, args.model)
    for c in ['MEAS_CD', 'MAIN_CONSTANT']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return add_correction_cols(df)


def load_longgap(pattern):
    """
    지난 분석 산출물(apc_ema_longgap.csv) 로더. 분할 파일은 glob 으로 병합.
    간만진행 행만 들어 있으며 EMA_EXPOSE 는 기존 방식(L3_EMA + BIAS_MIX8) 값이다.
    """
    import glob
    files = sorted(glob.glob(pattern),
                   key=lambda f: int(''.join(ch for ch in f.split('part')[-1] if ch.isdigit()) or 0)
                   if 'part' in f else 0)
    if not files:
        sys.exit(f'파일 없음: {pattern}')
    df = pd.concat([pd.read_csv(f, dtype=str, low_memory=False) for f in files],
                   ignore_index=True)
    print(f'[viz] {len(files)}개 파일 병합 -> {len(df):,}행')
    for c in ['CD_TARGET', 'MEAS_CD', 'PROC_EXPOSE', 'MAIN_CONSTANT', 'IDEAL_EXPOSE',
              'EMA_EXPOSE', 'EMA_EXPOSE_OLD', 'GAP_DAYS', 'CD_REL_ERR']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df['PROC_TRANS_DATE'] = pd.to_datetime(df['PROC_TRANS_DATE'].astype(str),
                                           format='%Y%m%d%H%M%S', errors='coerce')
    df = df.sort_values('PROC_TRANS_DATE', kind='mergesort').reset_index(drop=True)
    df['GAP_D'] = df['GAP_DAYS']
    df['TOL'] = df['MAIN_CONSTANT'] * df['CD_TARGET']
    df['VALID'] = ((df['MAIN_CONSTANT'] > 0) & (df['PROC_EXPOSE'] > 0)
                   & df['IDEAL_EXPOSE'].notna() & df['EMA_EXPOSE'].notna())
    logic = df['CD_LOGIC'] if 'CD_LOGIC' in df.columns else pd.Series('', index=df.index)
    df['EVAL'] = df['VALID'] & (df['CD_METHOD'] != 'FIX') & (logic != 'SC_IN_AI')
    df['LONG'] = (df['GAP_D'] >= _C.GAP_THRESHOLD) & _eq_target(df['PROC_EQ'])   # v3 결과 파일은 전 GAP 행 포함
    df['U_PROC'] = (df['IDEAL_EXPOSE'] - df['PROC_EXPOSE']) / df['TOL']
    df['U_EMA'] = (df['IDEAL_EXPOSE'] - df['EMA_EXPOSE']) / df['TOL']
    if 'EMA_EXPOSE_OLD' in df.columns:
        df['U_OLD'] = (df['IDEAL_EXPOSE'] - df['EMA_EXPOSE_OLD']) / df['TOL']
        print(f"[viz] 기존 방식 비교 컬럼 감지 — 비교 그림 추가 "
              f"(매칭 {int(df['U_OLD'].notna().sum()):,}행)")
    df = add_correction_cols(df)
    ev = df['EVAL'] & df['LONG']
    print(f'[viz] 평가 대상(FIX·SC_IN_AI 제외, {SCOPE_LBL}, 유효) {int(ev.sum()):,}건 '
          f'| 제외: MAIN_CONSTANT≤0 {int((df.MAIN_CONSTANT <= 0).sum())}, '
          f'IDEAL 결측 {int(df.IDEAL_EXPOSE.isna().sum())}')
    return df


def add_correction_cols(df):
    """보상량·시뮬레이션 CD 계산용 공통 컬럼"""
    ct = df['CD_TYPE'].astype(str).str.strip().str.upper()
    df['SGN'] = np.where(ct.isin(['SPACE', 'S']), 1.0, -1.0)
    df['CORR_MJ'] = df['EMA_EXPOSE'] - df['PROC_EXPOSE']                   # 보정량 mJ
    df['CORR_PCT'] = df['CORR_MJ'] / df['PROC_EXPOSE'] * 100               # 보정률 %
    df['CORR_CD'] = df['CORR_MJ'] / df['TOL']                               # CD % 환산 이동량
    if 'MEAS_CD' in df.columns:
        mc = df['MAIN_CONSTANT']
        df['SIM_CD'] = df['MEAS_CD'] + df['SGN'] * df['CORR_MJ'] / (100 * mc)
        df['CDERR_ACT'] = (df['MEAS_CD'] - df['CD_TARGET']) / df['CD_TARGET'] * 100
        df['CDERR_SIM'] = (df['SIM_CD'] - df['CD_TARGET']) / df['CD_TARGET'] * 100
    return df


def target(df, since=None, eq=None):
    m = df['EVAL'] & df['LONG'] & np.isfinite(df['U_PROC']) & np.isfinite(df['U_EMA'])
    if since:
        m &= df['PROC_TRANS_DATE'] >= pd.Timestamp(since)
    if eq:
        m &= df['PROC_EQ'].astype(str) == eq
    return df[m].copy()


def stats(d):
    b = d['U_PROC'].abs() <= PU
    a = d['U_EMA'].abs() <= PU
    g, l = int((~b & a).sum()), int((b & ~a).sum())
    return dict(N=len(d), P0=b.mean() * 100, P1=a.mean() * 100,
                dP=(a.mean() - b.mean()) * 100, gain=g, lost=l, net=g - l,
                lossrate=l / max(int(b.sum()), 1) * 100,
                bias0=d['U_PROC'].mean(), bias1=d['U_EMA'].mean(),
                mae0=d['U_PROC'].abs().mean(), mae1=d['U_EMA'].abs().mean(),
                re0=((d['PROC_EXPOSE'] - d['IDEAL_EXPOSE']).abs() / d['PROC_EXPOSE']).mean() * 100,
                re1=((d['EMA_EXPOSE'] - d['IDEAL_EXPOSE']).abs() / d['EMA_EXPOSE']).mean() * 100,
                ww=int((b & a).sum()), ll=int((~b & ~a).sum()),
                **MT.zone_stats(d['U_PROC'].values, d['U_EMA'].values),
                **MT.decision_stats(d['U_PROC'].values, d['U_EMA'].values),
                SCALE=cur_scale(d))


def cur_scale(d):
    """결과에 적용된 APPLY_SCALE — 결과 CSV 의 APPLY_SCALE 컬럼 > --scale > config"""
    if SCALE_OVERRIDE is not None:
        return float(SCALE_OVERRIDE)
    if 'APPLY_SCALE' in d.columns and d['APPLY_SCALE'].notna().any():
        return float(d['APPLY_SCALE'].dropna().iloc[0])
    return float(getattr(_C, 'APPLY_SCALE', 1.0))


SCALE_OVERRIDE = None


# ============================================================
# 표시 점 줄이기 (파일 크기)
# ============================================================
MAX_POINTS = 6000          # main() 에서 --max-points 로 덮어씀. 0 이면 전부
MAX_LINE_POINTS = 1500     # 이동중앙값 선의 최대 꼭짓점 수
PRIORITY_SHARE = 0.2       # 우선 표시(극단값) 점이 차지할 수 있는 최대 비율


def _even(idx, n):
    """정렬된 인덱스 배열에서 n 개를 균등 간격으로 (결정적, 시간 분포 보존)"""
    if n <= 0:
        return idx[:0]
    if len(idx) <= n:
        return idx
    return idx[np.linspace(0, len(idx) - 1, n).round().astype(int)]


def thin_points(d, priority=None, max_pts=None):
    """
    시간순으로 정렬된 d 에서 표시할 행 위치(np.ndarray)를 고른다.
    priority(bool 배열) 행을 먼저 담고(넘치면 그 안에서 균등 표본), 남는 자리는 나머지에서 균등 표본.
    """
    max_pts = MAX_POINTS if max_pts is None else max_pts
    n = len(d)
    allpos = np.arange(n)
    if not max_pts or n <= max_pts:
        return allpos
    pr = np.zeros(n, bool) if priority is None else np.asarray(priority, bool)
    keep_p = _even(allpos[pr], int(max_pts * PRIORITY_SHARE))   # 우선 표시분은 예산의 일부만
    rest = _even(allpos[~pr], max_pts - len(keep_p))
    return np.sort(np.r_[keep_p, rest])


def thin_line(x, y, max_pts=MAX_LINE_POINTS):
    """이동중앙값처럼 매끄러운 선은 꼭짓점만 줄여도 모양이 같다"""
    if len(x) <= max_pts:
        return x, y
    i = np.linspace(0, len(x) - 1, max_pts).round().astype(int)
    return x.iloc[i] if hasattr(x, 'iloc') else x[i], y.iloc[i] if hasattr(y, 'iloc') else y[i]


def _ts(x):
    """시각을 짧은 문자열로 (HTML 용량 절감, 분 단위)"""
    return pd.to_datetime(x).dt.strftime('%Y-%m-%d %H:%M').values


def _f32(a, nd=3):
    return np.round(np.asarray(a, dtype=float), nd).astype(np.float32)


def _shown_note(n_show, n_all):
    if n_show >= n_all:
        return ''
    return (f' — 점 {n_show:,}/{n_all:,}개 표시(시간 순 균등 표본). '
            '수치·막대·분포·이동중앙값은 전체 Lot 기준')


def _hist_bars(f, vals, edges, name, color, pattern=None, opacity=.55, orientation='v', **kw):
    """원자료 대신 미리 센 막대를 넣는다 (그림은 동일, 용량은 수백 분의 1)"""
    cnt, e = np.histogram(np.asarray(vals, float)[np.isfinite(vals)], bins=edges)
    mid = (e[:-1] + e[1:]) / 2
    w = float(e[1] - e[0])
    mk = dict(color=color)
    if pattern:
        mk['pattern_shape'] = pattern
    if orientation == 'h':
        tr = go.Bar(y=_f32(mid, 4), x=cnt, width=w, orientation='h', name=name, marker=mk,
                    opacity=opacity, hovertemplate='%{y:.2f}: %{x}건<extra>' + name + '</extra>')
    else:
        tr = go.Bar(x=_f32(mid, 4), y=cnt, width=w, name=name, marker=mk, opacity=opacity,
                    hovertemplate='%{x:.2f}: %{y}건<extra>' + name + '</extra>')
    f.add_trace(tr, **kw)


# ============================================================
# 그림
# ============================================================
def _band(fig, row=None, col=None, x0=None, x1=None):
    kw = dict(row=row, col=col) if row else {}
    fig.add_hrect(y0=-GU, y1=GU, fillcolor='rgba(239,159,39,0.08)', line_width=0, layer='below', **kw)
    fig.add_hrect(y0=-PU, y1=PU, fillcolor=C_BAND, line_width=0, layer='below', **kw)
    fig.add_hline(y=0, line_width=1, line_color='#c3c2b7', **kw)


def fig_time(d, window):
    d = d.sort_values('PROC_TRANS_DATE').reset_index(drop=True)
    pos = thin_points(d)                                   # 시간 순 균등 표본 (분포 왜곡 없음)
    v = d.iloc[pos]
    x = _ts(v['PROC_TRANS_DATE'])
    f = go.Figure()
    _band(f)
    f.add_trace(go.Scattergl(x=x, y=_f32(v['U_PROC']), mode='markers',
                             name='PROC (NormalAPC)',
                             marker=dict(color='rgba(0,0,0,0)', size=6,
                                         line=dict(color=C_PROC, width=1.2)),
                             customdata=np.c_[v['DEV_ID'].astype(str), v['ROUTE'].astype(str),
                                              v['GAP_D'].round(1)],
                             hovertemplate='%{x}<br>u=%{y:.2f}<br>%{customdata[0]} / '
                                           '%{customdata[1]}<br>GAP %{customdata[2]}일'
                                           '<extra>PROC</extra>'))
    f.add_trace(go.Scattergl(x=x, y=_f32(v['U_EMA']), mode='markers',
                             name=EMA_LBL,
                             marker=dict(color=C_EMA, size=5, symbol='diamond', opacity=.75),
                             hovertemplate='%{x}<br>u=%{y:.2f}<extra>' + EMA_LBL + '</extra>'))
    w = max(5, min(window, len(d) // 3))
    xt = d['PROC_TRANS_DATE']
    for col, c, nm in [('U_PROC', C_PROC, 'PROC 이동중앙값'), ('U_EMA', C_EMA, 'EMA 이동중앙값')]:
        lx, ly = thin_line(xt, d[col].rolling(w, center=True, min_periods=w // 2).median())
        f.add_trace(go.Scatter(x=_ts(lx), y=_f32(ly), mode='lines', name=nm,
                               line=dict(color=c, width=2.5,
                                         dash='dot' if col == 'U_PROC' else 'solid')))
    f.update_layout(title=f'① 시간에 따른 CD 오차 환산값 (이동 중앙값 {w}건)'
                          + _shown_note(len(pos), len(d)),
                    yaxis_title='u = CD 오차 (%)', xaxis_title='진행시각',
                    yaxis=dict(range=[-4, 4]), height=420)
    return f


def fig_gap(d):
    d = d.copy()
    d['GB'] = pd.cut(d['GAP_D'], GAP_BINS, labels=GAP_LBL, right=False)
    g = d.groupby('GB', observed=True).agg(
        N=('U_PROC', 'size'),
        mP=('U_PROC', lambda s: s.abs().median()), mE=('U_EMA', lambda s: s.abs().median()),
        pP=('U_PROC', lambda s: (s.abs() <= PU).mean() * 100),
        pE=('U_EMA', lambda s: (s.abs() <= PU).mean() * 100)).reset_index()
    x = [f'{a}<br>(N={n})' for a, n in zip(g['GB'], g['N'])]
    f = make_subplots(rows=1, cols=2, subplot_titles=('|u| 중앙값 (낮을수록 좋음)',
                                                      f'CD≤{PU:g}% 합격률 (%)'))
    f.add_trace(go.Bar(x=x, y=g['mP'], name='PROC', marker_color=C_PROC,
                       marker_pattern_shape='/', legendgroup='p'), 1, 1)
    f.add_trace(go.Bar(x=x, y=g['mE'], name='EMA', marker_color=C_EMA, legendgroup='e'), 1, 1)
    f.add_trace(go.Bar(x=x, y=g['pP'], name='PROC', marker_color=C_PROC,
                       marker_pattern_shape='/', legendgroup='p', showlegend=False), 1, 2)
    f.add_trace(go.Bar(x=x, y=g['pE'], name='EMA', marker_color=C_EMA,
                       legendgroup='e', showlegend=False), 1, 2)
    f.update_layout(title='② 재진행 간격(GAP, 일)별 비교', barmode='group', height=400)
    return f


def fig_hist(d):
    f = go.Figure()
    edges = np.arange(-5, 5.0001, 0.2)
    _hist_bars(f, d['U_PROC'].clip(-5, 5), edges, 'PROC', C_PROC, pattern='/')
    _hist_bars(f, d['U_EMA'].clip(-5, 5), edges, 'EMA', C_EMA)
    f.add_vrect(x0=-GU, x1=GU, fillcolor='rgba(239,159,39,0.10)', line_width=0, layer='below')
    f.add_vrect(x0=-PU, x1=PU, fillcolor=C_BAND, line_width=0, layer='below')
    f.add_vline(x=0, line_width=1, line_color='#c3c2b7')
    f.update_layout(title=f'③ CD 오차 환산값 분포 (진한 음영 = 합격 ±{PU:g}, 옅은 음영 = 회색 ±{GU:g})', barmode='overlay',
                    xaxis_title='u = CD 오차 (%)', yaxis_title='Lot 수', height=380)
    return f


def fig_2x2(s):
    z = [[s['ww'], s['lost']], [s['gain'], s['ll']]]
    txt = [[f"유지 합격<br>{s['ww']}", f"손실<br>{s['lost']}"],
           [f"구제<br>{s['gain']}", f"계속 탈락<br>{s['ll']}"]]
    f = go.Figure(go.Heatmap(z=[[1, 3], [2, 0]], x=['보정 후 합격', '보정 후 탈락'],
                             y=['보정 전 합격', '보정 전 탈락'], text=txt,
                             texttemplate='%{text}', textfont=dict(size=16),
                             colorscale=[[0, '#f1efe8'], [.33, '#e6f1fb'],
                                         [.66, '#eaf3de'], [1, '#fcebeb']],
                             showscale=False, hoverinfo='skip'))
    f.update_layout(title=f"④ 구제/손실 — 순증 {s['net']:+d}, 손실률 {s['lossrate']:.1f}%",
                    height=320, yaxis=dict(autorange='reversed'))
    return f


def fig_key(full, eq, key=None, since=None):
    h = full[(full['PROC_EQ'].astype(str) == eq) & full['VALID']].copy()
    h['_K'] = h['DEV_ID'].astype(str) + '|' + h['ROUTE'].astype(str)
    lg = h[h['LONG'] & h['EVAL']]
    if since:
        lg = lg[lg['PROC_TRANS_DATE'] >= pd.Timestamp(since)]
    if key is None:
        if lg.empty:
            return None, None
        key = lg['_K'].value_counts().index[0]
    k = h[h['_K'] == key].sort_values('PROC_TRANS_DATE')
    if k.empty:
        return None, key
    long_ = k[k['LONG']]
    ann = k[k['GAP_D'] >= 30]                          # 주석은 간만진행 Lot 에만 (전 GAP 적용 시 과밀 방지)
    f = go.Figure()
    f.add_trace(go.Scatter(x=_ts(k['PROC_TRANS_DATE']), y=_f32(k['IDEAL_EXPOSE'], 4), mode='lines+markers',
                           name='IDEAL (정답)', line=dict(color=C_IDEAL, width=1.5),
                           marker=dict(size=4)))
    f.add_trace(go.Scatter(x=_ts(k['PROC_TRANS_DATE']), y=_f32(k['PROC_EXPOSE'], 4), mode='markers',
                           name='PROC (NormalAPC)',
                           marker=dict(color='rgba(0,0,0,0)', size=7,
                                       line=dict(color=C_PROC, width=1.5))))
    f.add_trace(go.Scatter(x=_ts(long_['PROC_TRANS_DATE']), y=_f32(long_['EMA_EXPOSE'], 4), mode='markers',
                           name=EMA_LBL,
                           marker=dict(color=C_EMA, size=11 if len(long_) <= 300 else 7, symbol='diamond',
                                       line=dict(color='white', width=1))))
    # 합격 구간(IDEAL ± TOL): Lot 마다 도형을 그리지 않고 error bar 하나로
    f.add_trace(go.Scatter(x=_ts(long_['PROC_TRANS_DATE']), y=_f32(long_['IDEAL_EXPOSE'], 4), mode='markers',
                           name='합격 구간 (IDEAL ± TOL)', marker=dict(size=1, color='rgba(0,0,0,0)'),
                           error_y=dict(type='data', array=_f32(long_['TOL'], 4), color='rgba(29,158,117,.45)',
                                        thickness=4, width=0), hoverinfo='skip'))
    for _, r in ann.iterrows():
        f.add_annotation(x=r['PROC_TRANS_DATE'].strftime('%Y-%m-%d %H:%M'), y=r['PROC_EXPOSE'],
                         text=f"GAP {r['GAP_D']:.0f}일", showarrow=True, arrowhead=0,
                         ay=-30, font=dict(size=11))
    dev, route = key.split('|', 1)
    only_long = bool((k['GAP_D'] >= 30).all())
    sub = ' · 간만진행 Lot 만 수록된 데이터' if only_long else ''
    f.update_layout(title=f'⑤ KEY 드릴다운 — {dev} / {route} (녹색 막대 = 해당 Lot 합격 구간{sub})',
                    yaxis_title='노광량 (mJ)', xaxis_title='진행시각', height=420)
    return f, key


def fig_all_eq(df, since, hi=None):
    rows = []
    for eq, d in target(df, since).groupby('PROC_EQ'):
        if len(d) >= 30:
            s = stats(d)
            rows.append(dict(EQ=str(eq), **s))
    t = pd.DataFrame(rows).sort_values('dP')
    col = [C_EMA if e == hi else ('#c3c2b7' if v >= 0 else '#e34948')
           for e, v in zip(t['EQ'], t['dP'])]
    f = go.Figure(go.Bar(x=t['dP'], y=t['EQ'], orientation='h', marker_color=col,
                         customdata=np.c_[t['N'], t['P0'], t['P1'], t['net']],
                         hovertemplate='%{y}<br>Δ %{x:+.2f}%p<br>N=%{customdata[0]}'
                                       '<br>%{customdata[1]:.1f}% → %{customdata[2]:.1f}%'
                                       '<br>순증 %{customdata[3]:+d}<extra></extra>'))
    f.add_vline(x=0, line_color='#888780')
    f.update_layout(title=f'장비별 CD≤{PU:g}% 합격률 변화 ({SCOPE_LBL}, N≥30)',
                    xaxis_title='합격률 변화 (%p)', height=max(360, 22 * len(t) + 120))
    return f, t


def _corr_cards(d):
    a = d['CORR_PCT']
    mj = d['CORR_MJ']
    up = (a > 0).mean() * 100
    def card(lbl, val, sub=''):
        return (f'<div class="card"><div class="lbl">{lbl}</div><div class="val">{val}</div>'
                f'<div class="sub">{sub}</div></div>')
    return ('<h2>⑦ 보상량 — EMA 가 노광량을 얼마나 움직였나</h2><div class="cards">'
            + card('보정률 |중앙값|', f'{a.abs().median():.3f}%', f'{mj.abs().median():.3f} mJ')
            + card('보정률 |P90|', f'{a.abs().quantile(.9):.3f}%', f'{mj.abs().quantile(.9):.3f} mJ')
            + card('보정률 |P99|', f'{a.abs().quantile(.99):.3f}%', f'{mj.abs().quantile(.99):.3f} mJ')
            + card('최대 보정', f'{a.abs().max():.2f}%', f'{mj.abs().max():.2f} mJ')
            + card('방향', f'증가 {up:.0f}% / 감소 {100-up:.0f}%', '노광량을 올린 Lot 비율')
            + card('CD 환산 이동량 |중앙값|', f"{d['CORR_CD'].abs().median():.2f}%",
                   '보정이 CD 를 움직인 크기')
            + '</div>')


def fig_corr(d, window):
    d = d.sort_values('PROC_TRANS_DATE').reset_index(drop=True)
    f = make_subplots(rows=1, cols=3, column_widths=[.5, .22, .28], horizontal_spacing=.07,
                      subplot_titles=('시간에 따른 보정률 (%)', '보정률 분포',
                                      'GAP 구간별 평균 보정률'))
    lim = float(np.nanpercentile(d['CORR_PCT'].abs(), 99.5)) if len(d) else 1
    lim = lim if lim > 0 else 1
    big = (d['CORR_PCT'].abs() >= np.nanpercentile(d['CORR_PCT'].abs(), 99)).values if len(d) else None
    pos = thin_points(d, priority=big)                     # 큰 보정 Lot 우선
    v = d.iloc[pos]
    f.add_trace(go.Scattergl(x=_ts(v['PROC_TRANS_DATE']), y=_f32(v['CORR_PCT'], 4), mode='markers',
                             name='Lot 별 보정률', marker=dict(color=C_EMA, size=4, opacity=.5),
                             customdata=np.c_[_f32(v['PROC_EXPOSE'], 3), _f32(v['EMA_EXPOSE'], 3),
                                              _f32(v['CORR_MJ'], 4), _f32(v['GAP_D'], 1)],
                             hovertemplate='%{x}<br>%{customdata[0]:.2f} → %{customdata[1]:.2f} mJ'
                                           '<br>보정 %{customdata[2]:+.3f} mJ (%{y:+.3f}%)'
                                           '<br>GAP %{customdata[3]:.0f}일<extra></extra>'), 1, 1)
    w = max(5, min(window, len(d) // 3))
    lx, ly = thin_line(d['PROC_TRANS_DATE'],
                       d['CORR_PCT'].rolling(w, center=True, min_periods=w // 2).median())
    f.add_trace(go.Scatter(x=_ts(lx), y=_f32(ly, 4), mode='lines', name=f'이동 중앙값 ({w}건)',
                           line=dict(color='#0c447c', width=2.5)), 1, 1)
    f.add_hline(y=0, line_width=1, line_color='#c3c2b7', row=1, col=1)
    _hist_bars(f, d['CORR_PCT'].clip(-lim, lim), np.linspace(-lim, lim, 51), '분포', C_EMA,
               opacity=.7, orientation='h', row=1, col=2)
    f.data[-1].showlegend = False
    d['GB'] = pd.cut(d['GAP_D'], GAP_BINS, labels=GAP_LBL, right=False)
    g = d.groupby('GB', observed=True).agg(N=('CORR_PCT', 'size'), m=('CORR_PCT', 'mean'),
                                           a=('CORR_PCT', lambda x: x.abs().mean())).reset_index()
    x = [f'{b}<br>(N={n})' for b, n in zip(g['GB'], g['N'])]
    f.add_trace(go.Bar(x=x, y=g['a'], name='|보정률| 평균', marker_color=C_EMA), 1, 3)
    f.add_trace(go.Bar(x=x, y=g['m'], name='보정률 평균(부호)', marker_color='#85b7eb'), 1, 3)
    f.update_yaxes(range=[-lim, lim], row=1, col=1)
    f.update_yaxes(range=[-lim, lim], showticklabels=False, row=1, col=2)
    f.update_yaxes(title_text='보정률 (%)', row=1, col=1)
    if len(pos) < len(d):
        f.layout.annotations[0].text = f'시간에 따른 보정률 (%) — 점 {len(pos):,}/{len(d):,}개 표시'
    return f


SIM_COLS = [('LOT_ID', 'LOT_ID'), ('PROC_TRANS_DATE', '진행시각'), ('DEV_ID', 'DEV_ID'),
            ('ROUTE', 'ROUTE'), ('RETICLE', 'RETICLE'), ('CD_TYPE', 'CD_TYPE'),
            ('GAP_D', 'GAP(일)'), ('CD_TARGET', 'CD_TARGET'), ('MAIN_CONSTANT', 'MAIN_CONSTANT'),
            ('PROC_EXPOSE', 'APC노광량'), ('EMA_EXPOSE', 'EMA노광량'), ('CORR_MJ', '보정량(mJ)'),
            ('CORR_PCT', '보정률(%)'), ('MEAS_CD', '실측CD'), ('SIM_CD', 'EMA적용시CD'),
            ('CDERR_ACT', 'CD오차_실측(%)'), ('CDERR_SIM', 'CD오차_EMA적용(%)')]


def write_simulation(d, path):
    """Lot 별 시뮬레이션 표. 엑셀에서 바로 열리도록 utf-8-sig."""
    if 'SIM_CD' not in d.columns:
        print('[viz] MEAS_CD 컬럼이 없어 시뮬레이션 CSV 생략')
        return None
    t = d.sort_values('PROC_TRANS_DATE').copy()
    b = t['CDERR_ACT'].abs() <= PU
    a = t['CDERR_SIM'].abs() <= PU
    t['합격_실측'] = np.where(b, 'Y', 'N')
    t['합격_EMA적용'] = np.where(a, 'Y', 'N')
    t['판정'] = np.select([b & a, ~b & a, b & ~a], ['유지합격', '구제', '손실'], '계속탈락')
    out = t[[c for c, _ in SIM_COLS if c in t.columns]].rename(columns=dict(SIM_COLS))
    out['진행시각'] = pd.to_datetime(out['진행시각']).dt.strftime('%Y-%m-%d %H:%M:%S')
    for c, nd in [('GAP(일)', 1), ('보정량(mJ)', 4), ('보정률(%)', 4), ('EMA적용시CD', 5),
                  ('CD오차_실측(%)', 3), ('CD오차_EMA적용(%)', 3)]:
        if c in out.columns:
            out[c] = out[c].round(nd)
    t['구간_실측'] = MT.zone_label(t['CDERR_ACT'].values)
    t['구간_EMA적용'] = MT.zone_label(t['CDERR_SIM'].values)
    t['판정_3구간'] = MT.zone_judge(t['CDERR_ACT'].values, t['CDERR_SIM'].values)
    b2 = t['CDERR_ACT'].abs() <= DS
    a2 = t['CDERR_SIM'].abs() <= DS
    t[f'판정_{DS:g}%'] = np.select([b2 & a2, ~b2 & a2, b2 & ~a2], ['유지', '구제', '손실'], '계속불량')
    out = pd.concat([out, t[['합격_실측', '합격_EMA적용', '판정', '구간_실측', '구간_EMA적용', '판정_3구간',
                             f'판정_{DS:g}%']]], axis=1)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    out.to_csv(path, index=False, encoding='cp949')
    print(f'[viz] 시뮬레이션 {path} ({len(out):,} Lot)')
    return os.path.basename(path)



# ============================================================
# ⑧ 효과 읽는 법 — Lot 노이즈에 묻히지 않는 지표
# ============================================================
EFFECT_NOTE = """<div class="note">
Lot 마다의 측정·공정 노이즈(CD 오차 표준편차 약 1%)가 보정량(보통 0.1~0.5%)보다 훨씬 큼.
EMA 보정은 노이즈가 아니라 장비·제품·트랙 쪽으로 쏠린 <b>계통 오차</b>를 덜어내므로,
Lot 하나하나로는 좋아진 Lot 과 나빠진 Lot 이 섞여 보이는 것이 정상.
효과는 아래처럼 <b>평균(계통편향)·큰 오차 비율·보정 크기별·구간 평균</b>으로 묶어 볼 때 드러남.</div>"""

CORR_BINS = [0, 0.1, 0.25, 0.5, 1.0, 1e9]
CORR_LBL = ['<0.1', '0.1~0.25', '0.25~0.5', '0.5~1', '≥1']


def effect_stats(d):
    e0, e1 = d['U_PROC'].abs(), d['U_EMA'].abs()
    diff = e0 - e1
    n = len(d)
    sd = float(diff.std()) if n > 1 else np.nan
    return dict(win=(diff > 0).mean() * 100, lose=(diff < 0).mean() * 100, same=(diff == 0).mean() * 100,
                dmean=float(diff.mean()), dsd=sd, t=float(diff.mean() / sd * np.sqrt(n)) if sd and sd > 0 else np.nan)


def _effect_cards(d):
    s = effect_stats(d)
    def card(lbl, val, sub=''):
        return (f'<div class="card"><div class="lbl">{lbl}</div><div class="val">{val}</div>'
                f'<div class="sub">{sub}</div></div>')
    tails = ''.join(card(f'|CD 오차| &gt; {th}% 비율',
                         f"{(d['U_PROC'].abs() > th).mean()*100:.1f}% → {(d['U_EMA'].abs() > th).mean()*100:.1f}%",
                         '큰 오차 Lot 감소') for th in (2, 3))
    sig = '우연으로 보기 어려움' if abs(s['t']) >= 3 else '우연 범위'
    return ('<h2>⑧ 효과 </h2>' + EFFECT_NOTE + '<div class="cards">'
            + card('Lot 별 개선 / 악화', f"{s['win']:.0f}% / {s['lose']:.0f}%", f"변화 없음 {s['same']:.0f}%")
            + card('|CD 오차| 평균 개선 (기존−EMA)', f"{s['dmean']:+.3f}%p",
                   f"표준편차 {s['dsd']:.2f} · t={s['t']:.1f} ({sig})")
            + card('계통편향 (CD 오차 평균, u)', f"{d['U_PROC'].mean():+.3f} → {d['U_EMA'].mean():+.3f}",
                   '0 에 가까울수록 쏠림 없음')
            + tails + '</div>')


def fig_bias_week(d):
    """주별 계통편향(부호 있는 평균)과 합격률"""
    w = d.set_index('PROC_TRANS_DATE').sort_index()
    g = pd.DataFrame({
        'b0': w['U_PROC'].resample('W').mean(), 'b1': w['U_EMA'].resample('W').mean(),
        'p0': (w['U_PROC'].abs() <= PU).resample('W').mean() * 100,
        'p1': (w['U_EMA'].abs() <= PU).resample('W').mean() * 100,
        'n': w['U_PROC'].resample('W').size()}).dropna()
    g = g[g['n'] >= 5]
    x = g.index.strftime('%Y-%m-%d')
    f = make_subplots(rows=1, cols=2, horizontal_spacing=.08,
                      subplot_titles=('주별 계통편향 = CD 오차(u) 평균 (0 에 가까울수록 좋음)', f'주별 CD≤{PU:g}% 합격률 (%)'))
    f.add_trace(go.Scatter(x=x, y=_f32(g['b0']), mode='lines+markers', name='PROC (NormalAPC)',
                           line=dict(color=C_PROC, width=2, dash='dot'), marker=dict(size=5),
                           customdata=g['n'], hovertemplate='%{x}<br>%{y:+.3f}<br>N=%{customdata}<extra>PROC</extra>',
                           legendgroup='p'), 1, 1)
    f.add_trace(go.Scatter(x=x, y=_f32(g['b1']), mode='lines+markers', name=EMA_LBL,
                           line=dict(color=C_EMA, width=2), marker=dict(size=5),
                           customdata=g['n'], hovertemplate='%{x}<br>%{y:+.3f}<br>N=%{customdata}<extra>EMA</extra>',
                           legendgroup='e'), 1, 1)
    f.add_hline(y=0, line_width=1, line_color='#888780', row=1, col=1)
    f.add_trace(go.Scatter(x=x, y=_f32(g['p0'], 2), mode='lines', name='PROC', showlegend=False,
                           line=dict(color=C_PROC, width=2, dash='dot'), legendgroup='p',
                           hovertemplate='%{x}<br>%{y:.1f}%<extra>PROC</extra>'), 1, 2)
    f.add_trace(go.Scatter(x=x, y=_f32(g['p1'], 2), mode='lines', name='EMA', showlegend=False,
                           line=dict(color=C_EMA, width=2), legendgroup='e',
                           hovertemplate='%{x}<br>%{y:.1f}%<extra>EMA</extra>'), 1, 2)
    f.update_layout(title='⑧-1 시간에 따른 쏠림과 합격률 (주 단위 평균, 주당 5 Lot 이상)', height=400,
                    legend=dict(orientation='h', y=-0.18, x=0))
    return f


def fig_tail(d):
    """큰 오차 Lot 비율"""
    th = [1, 1.5, 2, 3]
    x = [f'|오차| &gt; {t}%' for t in th]
    y0 = [(d['U_PROC'].abs() > t).mean() * 100 for t in th]
    y1 = [(d['U_EMA'].abs() > t).mean() * 100 for t in th]
    f = go.Figure()
    f.add_trace(go.Bar(x=x, y=y0, name='PROC (NormalAPC)', marker_color=C_PROC, marker_pattern_shape='/',
                       text=[f'{v:.1f}%' for v in y0], textposition='outside'))
    f.add_trace(go.Bar(x=x, y=y1, name=EMA_LBL, marker_color=C_EMA,
                       text=[f'{v:.1f}%' for v in y1], textposition='outside'))
    f.update_layout(title='⑧-2 큰 CD 오차 Lot 비율 (낮을수록 좋음)', barmode='group',
                    yaxis_title='Lot 비율 (%)', height=360)
    return f


def fig_by_corr(d):
    """보정 크기(CD 환산 이동량)별 — 개선 Lot 비율과 합격률 변화"""
    t = d.copy()
    t['CB'] = pd.cut(t['CORR_CD'].abs(), CORR_BINS, labels=CORR_LBL, right=False)
    b, a = t['U_PROC'].abs() <= PU, t['U_EMA'].abs() <= PU
    t['win'] = (t['U_PROC'].abs() - t['U_EMA'].abs()) > 0
    t['lose'] = (t['U_PROC'].abs() - t['U_EMA'].abs()) < 0
    t['g'] = ~b & a
    t['l'] = b & ~a
    g = t.groupby('CB', observed=True).agg(N=('win', 'size'), win=('win', 'mean'), lose=('lose', 'mean'),
                                          g=('g', 'sum'), l=('l', 'sum')).reset_index()
    g['dP'] = (g['g'] - g['l']) / g['N'] * 100
    x = [f'{c}<br>(N={n:,})' for c, n in zip(g['CB'], g['N'])]
    f = make_subplots(rows=1, cols=2, horizontal_spacing=.1,
                      subplot_titles=('개선 / 악화 Lot 비율 (%)', f'CD≤{PU:g}% 합격률 변화 (%p) = (구제−손실)/N'))
    f.add_trace(go.Bar(x=x, y=g['win'] * 100, name='개선 Lot', marker_color=C_EMA,
                       text=[f'{v*100:.0f}%' for v in g['win']], textposition='outside'), 1, 1)
    f.add_trace(go.Bar(x=x, y=g['lose'] * 100, name='악화 Lot', marker_color=C_PROC, marker_pattern_shape='/',
                       text=[f'{v*100:.0f}%' for v in g['lose']], textposition='outside'), 1, 1)
    f.add_hline(y=50, line_width=1, line_dash='dot', line_color='#888780', row=1, col=1)
    f.add_trace(go.Bar(x=x, y=g['dP'], name='합격률 변화', marker_color='#0c447c', showlegend=False,
                       customdata=np.c_[g['g'], g['l']],
                       text=[f'{v:+.1f}' for v in g['dP']], textposition='outside',
                       hovertemplate='%{x}<br>%{y:+.2f}%p<br>구제 %{customdata[0]} / 손실 %{customdata[1]}'
                                     '<extra></extra>'), 1, 2)
    f.update_xaxes(title_text='보정이 CD 를 움직인 크기 (%)', row=1, col=1)
    f.update_xaxes(title_text='보정이 CD 를 움직인 크기 (%)', row=1, col=2)
    f.update_layout(title='⑧-3 보정 크기별 효과 — 작은 보정은 동전 던지기, 큰 보정일수록 확실',
                    barmode='group', height=430, legend=dict(orientation='h', y=-0.32, x=0))
    return f


def fig_gap_ci(d):
    """GAP 구간별 합격률 변화와 95% 신뢰구간 (같은 Lot 쌍 비교)"""
    t = d.copy()
    t['GB'] = pd.cut(t['GAP_D'], GAP_BINS, labels=GAP_LBL, right=False)
    b, a = t['U_PROC'].abs() <= PU, t['U_EMA'].abs() <= PU
    t['g'] = (~b & a).astype(int)
    t['l'] = (b & ~a).astype(int)
    g = t.groupby('GB', observed=True).agg(N=('g', 'size'), g=('g', 'sum'), l=('l', 'sum')).reset_index()
    small = g[g['N'] < 30]
    g = g[g['N'] >= 30].reset_index(drop=True)          # 30 Lot 미만 구간은 신뢰구간이 너무 넓어 제외
    if g.empty:
        return None
    n = g['N'].astype(float)
    dlt = (g['g'] - g['l']) / n
    se = np.sqrt(np.maximum((g['g'] + g['l']) / n - dlt ** 2, 0) / n)
    x = [f'{c}<br>(N={k:,})' for c, k in zip(g['GB'], g['N'])]
    col = [C_EMA if lo > 0 else ('#e34948' if hi < 0 else '#c3c2b7')
           for lo, hi in zip((dlt - 1.96 * se) * 100, (dlt + 1.96 * se) * 100)]
    f = go.Figure(go.Bar(x=x, y=dlt * 100, marker_color=col,
                         error_y=dict(type='data', array=1.96 * se * 100, color='#444441', thickness=1.5),
                         customdata=np.c_[g['g'], g['l'], (dlt - 1.96 * se) * 100, (dlt + 1.96 * se) * 100],
                         text=[f'{v:+.1f}' for v in dlt * 100], textposition='outside',
                         hovertemplate='%{x}<br>%{y:+.2f}%p (95%% 구간 %{customdata[2]:+.2f} ~ %{customdata[3]:+.2f})'
                                       '<br>구제 %{customdata[0]} / 손실 %{customdata[1]}<extra></extra>'))
    f.add_hline(y=0, line_width=1, line_color='#888780')
    sub = (' · 제외(N<30): ' + ', '.join(f'{c}({k})' for c, k in zip(small['GB'], small['N']))) if len(small) else ''
    f.update_layout(title=f'⑧-4 GAP 구간별 CD≤{PU:g}% 합격률 변화 (%p, 오차막대 = 95% 신뢰구간 · 파랑 = 0 초과 확실)' + sub,
                    xaxis_title='재진행 간격 GAP (일)', yaxis_title='합격률 변화 (%p)', height=380)
    return f


def effect_parts(d, weekly=True):
    if len(d) < 30:
        return []
    parts = [_effect_cards(d)]
    if weekly:
        parts.append(fig_bias_week(d))
    parts += [fig_tail(d), fig_by_corr(d), fig_gap_ci(d)]
    return parts


SIM_NOTE = """<div class="note"><b>Lot 별 시뮬레이션</b> —
EMA적용시CD = 실측CD + s × (EMA노광량 − APC노광량) / (100 × MAIN_CONSTANT),
s = +1 (SPACE) / −1 (LINE). APC 가 쓰는 노광량-CD 관계(MAIN_CONSTANT)를 그대로 쓴 계산이며,
보정 상한(CD 2% 이동분) 안에서 직선 근사가 성립합니다. {link}</div>"""


C_OLD = '#1baf7a'
OLD_LBL = 'EMA 보정 (L3_EMA + BIAS_MIX8)'


def has_old(df):
    return 'U_OLD' in df.columns and df['U_OLD'].notna().any()


def _pass(u):
    return (u.abs() <= PU).mean() * 100


def fig_compare_eq(d):
    """장비 1대: GAP 구간별 합격률 — PROC / 기존 / RULE"""
    d = d[d['U_OLD'].notna()].copy()
    if d.empty:
        return None
    d['GB'] = pd.cut(d['GAP_D'], GAP_BINS, labels=GAP_LBL, right=False)
    g = d.groupby('GB', observed=True).agg(
        N=('U_PROC', 'size'), P=('U_PROC', _pass), O=('U_OLD', _pass), E=('U_EMA', _pass)).reset_index()
    x = ['전체<br>(N=%d)' % len(d)] + [f'{a}<br>(N={n})' for a, n in zip(g['GB'], g['N'])]
    yP = [_pass(d['U_PROC'])] + list(g['P'])
    yO = [_pass(d['U_OLD'])] + list(g['O'])
    yE = [_pass(d['U_EMA'])] + list(g['E'])
    f = go.Figure()
    f.add_trace(go.Bar(x=x, y=yP, name='PROC (NormalAPC)', marker_color=C_PROC, marker_pattern_shape='/'))
    f.add_trace(go.Bar(x=x, y=yO, name=OLD_LBL, marker_color=C_OLD, marker_pattern_shape='.'))
    f.add_trace(go.Bar(x=x, y=yE, name=EMA_LBL, marker_color=C_EMA))
    lo = min(yP + yO + yE)
    f.update_layout(title=f'⑥ 방식 비교 — GAP 구간별 CD≤{PU:g}% 합격률 (같은 Lot)', barmode='group',
                    yaxis=dict(title='합격률 (%)', range=[max(0, lo - 8), None]), height=400)
    return f


def fig_compare_all(df, since):
    """전체 장비: 장비별 합격률 변화 — 기존 vs RULE"""
    rows = []
    for eq, d in target(df, since).groupby('PROC_EQ'):
        d = d[d['U_OLD'].notna()]
        if len(d) >= 30:
            p0 = _pass(d['U_PROC'])
            rows.append(dict(EQ=str(eq), N=len(d), dO=_pass(d['U_OLD']) - p0, dE=_pass(d['U_EMA']) - p0))
    if not rows:
        return None, None
    t = pd.DataFrame(rows).sort_values('dE')
    f = go.Figure()
    f.add_trace(go.Bar(y=t['EQ'], x=t['dO'], orientation='h', name=OLD_LBL,
                       marker_color=C_OLD, marker_pattern_shape='.'))
    f.add_trace(go.Bar(y=t['EQ'], x=t['dE'], orientation='h', name=EMA_LBL, marker_color=C_EMA))
    f.add_vline(x=0, line_color='#888780')
    f.update_layout(title=f'방식 비교 — 장비별 CD≤{PU:g}% 합격률 변화 (같은 Lot, N≥30)', barmode='group',
                    xaxis_title='합격률 변화 (%p)', height=max(380, 30 * len(t) + 140))
    return f, t


def _cards_compare(d):
    d = d[d['U_OLD'].notna()]
    if d.empty:
        return ''
    def one(col, name):
        b, a = d['U_PROC'].abs() <= PU, d[col].abs() <= PU
        g, l = int((~b & a).sum()), int((b & ~a).sum())
        return (f'<tr><td>{name}</td><td>{a.mean()*100:.2f}%</td>'
                f'<td>{(a.mean()-b.mean())*100:+.2f}%p</td><td>{g}</td><td>{l}</td>'
                f'<td>{g-l:+d}</td><td>{l/max(int(b.sum()),1)*100:.1f}%</td>'
                f'<td>{d[col].mean():+.3f}</td></tr>')
    p = (d['U_PROC'].abs() <= PU).mean() * 100
    return ('<h2>방식 비교 (같은 Lot ' + f'{len(d):,}건)</h2>'
            f'<table class="cmp"><tr><th>방식</th><th>CD≤{PU:g}%</th><th>변화</th><th>구제</th>'
            '<th>손실</th><th>순증</th><th>손실률</th><th>계통편향 u</th></tr>'
            f'<tr><td>PROC (NormalAPC)</td><td>{p:.2f}%</td><td>—</td><td>—</td><td>—</td>'
            f'<td>—</td><td>—</td><td>{d["U_PROC"].mean():+.3f}</td></tr>'
            + one('U_OLD', OLD_LBL) + one('U_EMA', EMA_LBL) + '</table>')


# ============================================================
# HTML 조립
# ============================================================
def _cards(s, title):
    def card(lbl, val, sub=''):
        return (f'<div class="card"><div class="lbl">{lbl}</div><div class="val">{val}</div>'
                f'<div class="sub">{sub}</div></div>')
    return (f'<h2>{title}</h2><div class="cards">'
            + card('보정 대상 Lot', f"{s['N']:,}")
            + card(f'CD≤{PU:g}% 합격률', f"{s['P0']:.1f}% → {s['P1']:.1f}%", f"{s['dP']:+.2f}%p")
            + card('순증', f"{s['net']:+d}", f"구제 {s['gain']} / 손실 {s['lost']}")
            + card('손실률', f"{s['lossrate']:.1f}%", '원래 합격 중 탈락 비율')
            + card('계통편향 (u 평균)', f"{s['bias0']:+.2f} → {s['bias1']:+.2f}", '0에 가까울수록 좋음')
            + card('|u| 평균', f"{s['mae0']:.2f} → {s['mae1']:.2f}")
            + card('노광량 상대오차 평균', f"{s['re0']:.3f}% → {s['re1']:.3f}%",
                   f"{(s['re0']-s['re1'])/s['re0']*100:+.1f}% 개선")
            + '</div>' + _decision_cards(s) + _zone_cards(s))


ZONE_NOTE = (f'<div class="note"><b>3구간 판정 (측정 오차 감안)</b> — 합격 |CD 오차| ≤ {PU:g}%, '
             f'회색 {PU:g} ~ {GU:g}% (측정 오차 범위), 불량 &gt; {GU:g}%. '
             '합격↔불량을 건너뛴 이동만 <b>확실한 구제 / 확실한 손실</b>로 보고, 회색을 거치는 이동은 경미한 변화로 본다. '
             '3구간 점수 = Σ(보정 후 점수 − 보정 전 점수), 점수는 합격 2 · 회색 1 · 불량 0. 구제와 손실을 같은 무게로 센다.</div>')


def _zone_cards(s):
    if 'Z_SCORE' not in s:
        return ''
    def card(lbl, val, sub=''):
        return (f'<div class="card"><div class="lbl">{lbl}</div><div class="val">{val}</div>'
                f'<div class="sub">{sub}</div></div>')
    return ('<h2>3구간 판정 (측정 오차 감안)</h2>' + ZONE_NOTE + '<div class="cards">'
            + card('3구간 점수', f"{s['Z_SCORE']:+,d}", '+ 일수록 좋음')
            + card('확실한 구제', f"{s['Z_HARD_RESCUE']:,d}", f'불량 → 합격')
            + card('확실한 손실', f"{s['Z_HARD_LOSS']:,d}", f'합격 → 불량')
            + card('경미한 개선 / 악화', f"{s['Z_SOFT_UP']:,d} / {s['Z_SOFT_DOWN']:,d}", '회색 구간을 거친 이동')
            + card(f'불량률 (>{GU:g}%)', f"{s['Z_BAD0']:.1f}% → {s['Z_BAD1']:.1f}%",
                   f"{(s['Z_BAD1']-s['Z_BAD0']):+.2f}%p")
            + card(f'합격 / 회색 비율', f"{s['Z_PASS0']:.1f}/{s['Z_GRAY0']:.1f} → {s['Z_PASS1']:.1f}/{s['Z_GRAY1']:.1f}%")
            + '</div>')


DECISION_NOTE = (f'<div class="note"><b>결정 지표 ({DS:g}% 초과 = 확실한 불량)</b> — '
                 f'|CD 오차| &gt; {DS:g}% 를 확실한 불량으로 보고, 보정으로 불량 → {DS:g}% 이내가 된 Lot 을 구제, '
                 f'{DS:g}% 이내 → 불량이 된 Lot 을 손실로 센다. 운영 의사결정(배율 APPLY_SCALE)은 이 지표로 한다. '
                 '구제/손실 비율 자체를 최대화하면 보정을 거의 안 하는 쪽으로 쏠리므로, '
                 '<b>허용할 손실 수준(손실률)을 먼저 정하고 그 안에서 순증이 큰 배율</b>을 고른다.</div>')


def _decision_cards(s):
    if 'D_NET' not in s:
        return ''
    def card(lbl, val, sub=''):
        return (f'<div class="card"><div class="lbl">{lbl}</div><div class="val">{val}</div>'
                f'<div class="sub">{sub}</div></div>')
    r = s['D_RATIO']
    return (f'<h2>결정 지표 ({DS:g}% 초과 = 확실한 불량)</h2>' + DECISION_NOTE + '<div class="cards">'
            + card('적용 배율', f"{s['SCALE']:g}", 'config.APPLY_SCALE')
            + card('순증', f"{s['D_NET']:+,d}", f"구제 {s['D_RESCUE']:,d} / 손실 {s['D_LOSS']:,d}")
            + card('구제 : 손실', '∞' if not np.isfinite(r) else f'{r:.2f} : 1')
            + card('손실률', f"{s['D_LOSSRATE']:.1f}%", f'원래 {DS:g}% 이내 중 불량이 된 비율')
            + card(f'불량률 (>{DS:g}%)', f"{s['D_BAD0']:.1f}% → {s['D_BAD1']:.1f}%",
                   f"{(s['D_BAD1']-s['D_BAD0']):+.2f}%p")
            + '</div>')


def _scale_table(d, s=None):
    """배율(APPLY_SCALE)만 바꿨을 때의 결정 지표 — 재학습 없이 보정량에 배율만 곱한 근사"""
    if len(d) == 0:
        return ''
    cur = cur_scale(d)
    rows = MT.scale_scenarios(d['U_PROC'].values, d['U_EMA'].values, cur)
    h = ('<h2>배율(APPLY_SCALE) 시나리오</h2><div class="note">지금 결과의 보정량을 배율만 바꿔 다시 센 값이다. '
         '게이트(적용 여부)는 배율 적용 전에 판정되므로 적용 Lot 은 그대로이고 보정 크기만 줄어든다. '
         '노광량 최소단위 반올림은 무시한 근사. 손실 예산을 먼저 정하고 그 안에서 순증이 가장 큰 배율을 고른다. '
         f'<b>굵은 행</b> = 현재 배율 {cur:g}.</div>'
         '<table class="cmp"><tr><th>배율</th><th>구제</th><th>손실</th><th>순증</th>'
         f'<th>구제:손실</th><th>손실률</th><th>불량률 (&gt;{DS:g}%)</th></tr>')
    for c, x in rows:
        r = x['D_RATIO']
        st = ' style="font-weight:600;background:#eef4fb"' if abs(c - cur) < 1e-9 else ''
        h += (f'<tr{st}><td>{c:g}</td><td>{x["D_RESCUE"]:,d}</td><td>{x["D_LOSS"]:,d}</td>'
              f'<td>{x["D_NET"]:+,d}</td><td>{"∞" if not np.isfinite(r) else f"{r:.2f}"}</td>'
              f'<td>{x["D_LOSSRATE"]:.1f}%</td><td>{x["D_BAD0"]:.1f}% → {x["D_BAD1"]:.1f}%</td></tr>')
    return h + '</table>'


def fig_zone(s):
    """3x3 구간 이동표 (행 = 보정 전, 열 = 보정 후)"""
    M = s['Z_MAT'][::-1, ::-1]                    # 합격, 회색, 불량 순서
    names = ['합격', '회색', '불량']
    kind = [['유지', '경미한 악화', '확실한 손실'], ['경미한 개선', '유지', '경미한 악화'],
            ['확실한 구제', '경미한 개선', '유지']]
    col = [[0, -1, -2], [1, 0, -1], [2, 1, 0]]
    txt = [[f'{kind[i][j]}<br>{M[i, j]:,d}' for j in range(3)] for i in range(3)]
    f = go.Figure(go.Heatmap(z=col, x=[f'보정 후 {n}' for n in names], y=[f'보정 전 {n}' for n in names],
                             text=txt, texttemplate='%{text}', textfont=dict(size=14), zmin=-2, zmax=2,
                             colorscale=[[0, '#f4c7c3'], [.25, '#fcebeb'], [.5, '#f1efe8'],
                                         [.75, '#eaf3de'], [1, '#bfe3b4']],
                             showscale=False, hoverinfo='skip'))
    f.update_layout(title=f"④-2 3구간 이동표 — 점수 {s['Z_SCORE']:+,d} · 확실한 구제 {s['Z_HARD_RESCUE']} / "
                          f"확실한 손실 {s['Z_HARD_LOSS']}", height=360, yaxis=dict(autorange='reversed'))
    return f


CSS = """<style>
body{font-family:'Malgun Gothic',sans-serif;margin:24px;color:#222;max-width:1200px}
h1{font-size:22px;font-weight:500} h2{font-size:17px;font-weight:500;margin-top:28px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px}
.card{background:#f6f6f3;border-radius:8px;padding:12px 14px}
.lbl{font-size:12px;color:#777}.val{font-size:20px;margin-top:4px}.sub{font-size:12px;color:#777}
.note{font-size:13px;color:#555;line-height:1.7;background:#fafaf7;padding:10px 14px;border-radius:8px}
a{color:#2a78d6}
.cmp{border-collapse:collapse;font-size:13px;margin-top:6px}.cmp td,.cmp th{border-bottom:1px solid #e5e5e0;padding:6px 12px;text-align:right}.cmp td:first-child,.cmp th:first-child{text-align:left}
</style>"""

NOTE = """<div class="note">
u = (IDEAL − PROC) / (MAIN_CONSTANT × CD_TARGET) 는 CD 오차(%)와 같으며 |u|≤""" + f'{PU:g}' + """ 이면 합격 (회색 ≤""" + f'{GU:g}' + """).
장비 전체는 제품이 섞여 있어 노광량 원값으로는 비교가 되지 않으므로 u 로 정규화.
EMA 값은 각 Lot 진행 시점까지의 이력만으로 재현한 반사실 값.(운영과 동일한 인과 순서)
</div>"""


def write_html(path, title, parts, shared_js=False):
    """shared_js=True 면 같은 폴더의 plotly.min.js 를 참조 (일괄 생성용, 오프라인 동작)"""
    html = [f'<html><head><meta charset="utf-8"><title>{title}</title>{CSS}</head><body>']
    first = True
    for p in parts:
        if isinstance(p, str):
            html.append(p)
        elif p is not None:
            leg = dict(orientation='h', y=1.08) if p.layout.legend.y is None else {}
            p.update_layout(template='plotly_white', font=dict(size=12),
                            margin=dict(l=60, r=20, t=70, b=50 if not leg == {} else 90))
            if leg:
                p.update_layout(legend=leg)
            inc = ('plotly.min.js' if shared_js else True) if first else False
            if shared_js and first:
                html.append('<script src="plotly.min.js"></script>')
                inc = False
            html.append(p.to_html(full_html=False, include_plotlyjs=inc))
            first = False
    html.append('</body></html>')
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    open(path, 'w', encoding='utf-8').write('\n'.join(html))
    print(f'[viz] {path}')


def report_eq(df, eq, args, path, shared_js=False):
    d = target(df, args.since, eq)
    if len(d) < 10:
        print(f'[viz] {eq}: 보정 대상 {len(d)}건 — 생략')
        return None
    s = stats(d)
    fk, key = fig_key(df, eq, args.key, args.since)
    fa, _ = fig_all_eq(df, args.since, hi=eq)
    rng = f"{d['PROC_TRANS_DATE'].min():%Y-%m-%d} ~ {d['PROC_TRANS_DATE'].max():%Y-%m-%d}"
    rng += f" · 노광량 변경 Lot {(d['EMA_EXPOSE'] != d['PROC_EXPOSE']).mean()*100:.1f}%"
    sim_path = os.path.splitext(path)[0] + '_simulation.csv'
    sim = write_simulation(d, sim_path)
    sim_html = SIM_NOTE.format(link=(f'<a href="{sim}">Lot 별 시뮬레이션 표 내려받기 ({sim})</a>'
                                     if sim else ''))
    write_html(path, f'EMA 효과 {eq}', [
        f'<h1>EMA 보정 효과 — {eq}</h1><p>방식: {EMA_LBL} · {SCOPE_LBL} · 평가기간 {rng}</p>', NOTE,
        _cards(s, '요약'), _scale_table(d, s),
        fig_time(d, args.window), fig_gap(d), fig_hist(d), fig_2x2(s), fig_zone(s), fk,
        _corr_cards(d), fig_corr(d, args.window),
        *effect_parts(d),
        sim_html,
        _cards_compare(d) if has_old(d) else '',
        fig_compare_eq(d) if has_old(d) else None,
        '<h2>전체 장비 대비 위치</h2>', fa], shared_js=shared_js)
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('csv')
    ap.add_argument('--model', default='model')
    ap.add_argument('--prepared', action='store_true', help='analyze.py 결과 CSV 입력')
    ap.add_argument('--longgap', action='store_true',
                    help='지난 분석 산출물 apc_ema_longgap.csv 입력 (분할 glob 가능)')
    ap.add_argument('--label', help='보정 방식 이름 (범례·제목 표시)')
    ap.add_argument('--eq')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--since')
    ap.add_argument('--key', help='"DEV_ID|ROUTE"')
    ap.add_argument('--window', type=int, default=40)
    ap.add_argument('--max-points', type=int, default=6000,
                    help='산점도 최대 점 수 (0 = 전부)')
    ap.add_argument('--scale', type=float,
                    help='결과에 적용된 APPLY_SCALE (결과 CSV 에 APPLY_SCALE 컬럼이 없을 때만 지정)')
    ap.add_argument('--out', default='report.html')
    a = ap.parse_args()
    if not a.eq and not a.all:
        ap.error('--eq 또는 --all 을 지정하십시오')

    global EMA_LBL, MAX_POINTS, SCALE_OVERRIDE
    MAX_POINTS = a.max_points
    SCALE_OVERRIDE = a.scale
    if a.longgap:
        df = load_longgap(a.csv)
        mode = str(df['EMA_MODE'].dropna().iloc[0]) if 'EMA_MODE' in df.columns and df['EMA_MODE'].notna().any() else ''
        auto = 'EMA 보정 (RULE v3.1 + β)' if mode.startswith('RULE') else 'EMA 보정 (L3_EMA + BIAS_MIX8)'
        EMA_LBL = a.label or auto
    else:
        EMA_LBL = a.label or 'EMA 보정 (RULE v3.1 + β)'
        df = load(a)
    if a.all:
        os.makedirs(a.out, exist_ok=True)
        from plotly.offline import get_plotlyjs
        open(os.path.join(a.out, 'plotly.min.js'), 'w', encoding='utf-8').write(get_plotlyjs())
        fa, t = fig_all_eq(df, a.since)
        tot = stats(target(df, a.since))
        links = ''.join(f'<li><a href="{e}.html">{e}</a> — {DS:g}% 순증 {dn:+,d} '
                        f'(구제 {dr} / 손실 {dl}), Δ {v:+.2f}%p, 3구간 점수 {z:+,d}, N={n}</li>'
                        for e, dn, dr, dl, v, z, n in t.sort_values('D_NET', ascending=False)[
                            ['EQ', 'D_NET', 'D_RESCUE', 'D_LOSS', 'dP', 'Z_SCORE', 'N']].values)
        write_html(os.path.join(a.out, 'index.html'), 'EMA 효과 요약', [
            f'<h1>EMA 보정 효과 — 전체 장비</h1><p>방식: {EMA_LBL}</p>', NOTE,
            _cards(tot, '전체 요약'), _scale_table(target(df, a.since)), fig_zone(tot), fa,
            *effect_parts(target(df, a.since), weekly=True),
            _cards_compare(target(df, a.since)) if has_old(df) else '',
            fig_compare_all(df, a.since)[0] if has_old(df) else None,
            f'<h2>장비별 리포트</h2><ul>{links}</ul>'], shared_js=True)
        for eq in t['EQ']:
            report_eq(df, eq, a, os.path.join(a.out, f'{eq}.html'), shared_js=True)
    else:
        s = report_eq(df, a.eq, a, a.out)
        if s:
            print(f"[viz] {a.eq}: N={s['N']} {s['P0']:.1f}% -> {s['P1']:.1f}% "
                  f"({s['dP']:+.2f}%p) 순증 {s['net']:+d} · 3구간 점수 {s['Z_SCORE']:+d} "
                  f"(확실한 구제 {s['Z_HARD_RESCUE']} / 손실 {s['Z_HARD_LOSS']}) · "
                  f"{DS:g}% 기준 구제 {s['D_RESCUE']} / 손실 {s['D_LOSS']} / 순증 {s['D_NET']:+d} "
                  f"(배율 {s['SCALE']:g})")
    return 0


if __name__ == '__main__':
    sys.exit(main())
