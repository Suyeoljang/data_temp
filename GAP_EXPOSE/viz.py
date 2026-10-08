"""
장비별 EMA 보정 효과 시각화 (간만진행 대상)

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

입력
  PHOTO_APC_DATA 추출 CSV (평가 구간 앞 3개월 이상 이력 포함 권장)
  필수: PROC_TRANS_DATE, PROC_EQ, DEV_ID, ROUTE, RETICLE, CD_TYPE, CD_TARGET,
        MEAS_CD, MAIN_CONSTANT, PROC_EXPOSE, CD_METHOD + 축 컬럼들

구성 (장비 1대)
  0) 요약 지표          합격률 전후, 구제/손실/순증, 계통편향
  1) 시간축 산점도      u = (IDEAL-사용값)/TOL, ±1 합격 밴드, 이동 중앙값
  2) GAP축 비교         GAP 구간별 |u| 중앙값·합격률
  3) 분포 비교          u 히스토그램
  4) 구제/손실 2x2
  5) KEY 드릴다운       KEY 하나의 전체 이력을 mJ 원값으로

  u 는 CD 오차(%)와 같다. |u|<=1 이면 합격.
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
GAP_BINS = [30, 45, 60, 90, 180, 365, 1e9]
GAP_LBL = ['30-45', '45-60', '60-90', '90-180', '180-365', '365+']


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
    df['LONG'] = df['GAP_D'] >= 30
    df['U_PROC'] = (df['IDEAL_EXPOSE'] - df['PROC_EXPOSE']) / df['TOL']
    df['U_EMA'] = (df['IDEAL_EXPOSE'] - df['EMA_EXPOSE']) / df['TOL']
    if 'EMA_EXPOSE_OLD' in df.columns:
        df['U_OLD'] = (df['IDEAL_EXPOSE'] - df['EMA_EXPOSE_OLD']) / df['TOL']
        print(f"[viz] 기존 방식 비교 컬럼 감지 — 비교 그림 추가 "
              f"(매칭 {int(df['U_OLD'].notna().sum()):,}행)")
    df = add_correction_cols(df)
    ev = df['EVAL'] & df['LONG']
    print(f'[viz] 평가 대상(FIX·SC_IN_AI 제외, GAP≥30, 유효) {int(ev.sum()):,}건 '
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
    b = d['U_PROC'].abs() <= 1.0
    a = d['U_EMA'].abs() <= 1.0
    g, l = int((~b & a).sum()), int((b & ~a).sum())
    return dict(N=len(d), P0=b.mean() * 100, P1=a.mean() * 100,
                dP=(a.mean() - b.mean()) * 100, gain=g, lost=l, net=g - l,
                lossrate=l / max(int(b.sum()), 1) * 100,
                bias0=d['U_PROC'].mean(), bias1=d['U_EMA'].mean(),
                mae0=d['U_PROC'].abs().mean(), mae1=d['U_EMA'].abs().mean(),
                re0=((d['PROC_EXPOSE'] - d['IDEAL_EXPOSE']).abs() / d['PROC_EXPOSE']).mean() * 100,
                re1=((d['EMA_EXPOSE'] - d['IDEAL_EXPOSE']).abs() / d['EMA_EXPOSE']).mean() * 100,
                ww=int((b & a).sum()), ll=int((~b & ~a).sum()))


# ============================================================
# 그림
# ============================================================
def _band(fig, row=None, col=None, x0=None, x1=None):
    kw = dict(row=row, col=col) if row else {}
    fig.add_hrect(y0=-1, y1=1, fillcolor=C_BAND, line_width=0, layer='below', **kw)
    fig.add_hline(y=0, line_width=1, line_color='#c3c2b7', **kw)


def fig_time(d, window):
    d = d.sort_values('PROC_TRANS_DATE')
    f = go.Figure()
    _band(f)
    f.add_trace(go.Scattergl(x=d['PROC_TRANS_DATE'], y=d['U_PROC'], mode='markers',
                             name='PROC (NormalAPC)',
                             marker=dict(color='rgba(0,0,0,0)', size=6,
                                         line=dict(color=C_PROC, width=1.2)),
                             customdata=np.c_[d['DEV_ID'], d['ROUTE'], d['GAP_D']],
                             hovertemplate='%{x}<br>u=%{y:.2f}<br>%{customdata[0]} / '
                                           '%{customdata[1]}<br>GAP %{customdata[2]:.0f}일'
                                           '<extra>PROC</extra>'))
    f.add_trace(go.Scattergl(x=d['PROC_TRANS_DATE'], y=d['U_EMA'], mode='markers',
                             name=EMA_LBL,
                             marker=dict(color=C_EMA, size=5, symbol='diamond', opacity=.75),
                             hovertemplate='%{x}<br>u=%{y:.2f}<extra>' + EMA_LBL + '</extra>'))
    w = max(5, min(window, len(d) // 3))
    for col, c, nm in [('U_PROC', C_PROC, 'PROC 이동중앙값'), ('U_EMA', C_EMA, 'EMA 이동중앙값')]:
        f.add_trace(go.Scatter(x=d['PROC_TRANS_DATE'],
                               y=d[col].rolling(w, center=True, min_periods=w // 2).median(),
                               mode='lines', name=nm,
                               line=dict(color=c, width=2.5,
                                         dash='dot' if col == 'U_PROC' else 'solid')))
    f.update_layout(title=f'① 시간에 따른 CD 오차 환산값 (이동 중앙값 {w}건)',
                    yaxis_title='u = CD 오차 (%)', xaxis_title='진행시각',
                    yaxis=dict(range=[-4, 4]), height=420)
    return f


def fig_gap(d):
    d = d.copy()
    d['GB'] = pd.cut(d['GAP_D'], GAP_BINS, labels=GAP_LBL, right=False)
    g = d.groupby('GB', observed=True).agg(
        N=('U_PROC', 'size'),
        mP=('U_PROC', lambda s: s.abs().median()), mE=('U_EMA', lambda s: s.abs().median()),
        pP=('U_PROC', lambda s: (s.abs() <= 1).mean() * 100),
        pE=('U_EMA', lambda s: (s.abs() <= 1).mean() * 100)).reset_index()
    x = [f'{a}<br>(N={n})' for a, n in zip(g['GB'], g['N'])]
    f = make_subplots(rows=1, cols=2, subplot_titles=('|u| 중앙값 (낮을수록 좋음)',
                                                      'CD≤1% 합격률 (%)'))
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
    edges = dict(start=-5, end=5, size=0.2)
    f.add_trace(go.Histogram(x=d['U_PROC'].clip(-5, 5), xbins=edges, name='PROC',
                             marker_color=C_PROC, opacity=.55, marker_pattern_shape='/'))
    f.add_trace(go.Histogram(x=d['U_EMA'].clip(-5, 5), xbins=edges, name='EMA',
                             marker_color=C_EMA, opacity=.55))
    f.add_vrect(x0=-1, x1=1, fillcolor=C_BAND, line_width=0, layer='below')
    f.add_vline(x=0, line_width=1, line_color='#c3c2b7')
    f.update_layout(title='③ CD 오차 환산값 분포 (음영 = 합격 구간 ±1)', barmode='overlay',
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
    f = go.Figure()
    f.add_trace(go.Scatter(x=k['PROC_TRANS_DATE'], y=k['IDEAL_EXPOSE'], mode='lines+markers',
                           name='IDEAL (정답)', line=dict(color=C_IDEAL, width=1.5),
                           marker=dict(size=4)))
    f.add_trace(go.Scatter(x=k['PROC_TRANS_DATE'], y=k['PROC_EXPOSE'], mode='markers',
                           name='PROC (NormalAPC)',
                           marker=dict(color='rgba(0,0,0,0)', size=7,
                                       line=dict(color=C_PROC, width=1.5))))
    f.add_trace(go.Scatter(x=long_['PROC_TRANS_DATE'], y=long_['EMA_EXPOSE'], mode='markers',
                           name=EMA_LBL,
                           marker=dict(color=C_EMA, size=11, symbol='diamond',
                                       line=dict(color='white', width=1))))
    for _, r in long_.iterrows():
        f.add_annotation(x=r['PROC_TRANS_DATE'], y=r['PROC_EXPOSE'],
                         text=f"GAP {r['GAP_D']:.0f}일", showarrow=True, arrowhead=0,
                         ay=-30, font=dict(size=11))
        tol = r['TOL']
        f.add_shape(type='rect', x0=r['PROC_TRANS_DATE'], x1=r['PROC_TRANS_DATE'],
                    y0=r['IDEAL_EXPOSE'] - tol, y1=r['IDEAL_EXPOSE'] + tol,
                    line=dict(color='#1d9e75', width=6), opacity=.35)
    dev, route = key.split('|', 1)
    only_long = bool(k['LONG'].all())
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
    f.update_layout(title='장비별 CD≤1% 합격률 변화 (간만진행, N≥30)',
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
    d = d.sort_values('PROC_TRANS_DATE').copy()
    f = make_subplots(rows=1, cols=3, column_widths=[.5, .22, .28], horizontal_spacing=.07,
                      subplot_titles=('시간에 따른 보정률 (%)', '보정률 분포',
                                      'GAP 구간별 평균 보정률'))
    f.add_trace(go.Scattergl(x=d['PROC_TRANS_DATE'], y=d['CORR_PCT'], mode='markers',
                             name='Lot 별 보정률', marker=dict(color=C_EMA, size=4, opacity=.5),
                             customdata=np.c_[d['PROC_EXPOSE'], d['EMA_EXPOSE'], d['CORR_MJ'],
                                              d['GAP_D']],
                             hovertemplate='%{x}<br>%{customdata[0]:.2f} → %{customdata[1]:.2f} mJ'
                                           '<br>보정 %{customdata[2]:+.3f} mJ (%{y:+.3f}%)'
                                           '<br>GAP %{customdata[3]:.0f}일<extra></extra>'), 1, 1)
    w = max(5, min(window, len(d) // 3))
    f.add_trace(go.Scatter(x=d['PROC_TRANS_DATE'],
                           y=d['CORR_PCT'].rolling(w, center=True, min_periods=w // 2).median(),
                           mode='lines', name=f'이동 중앙값 ({w}건)',
                           line=dict(color='#0c447c', width=2.5)), 1, 1)
    f.add_hline(y=0, line_width=1, line_color='#c3c2b7', row=1, col=1)
    lim = float(np.nanpercentile(d['CORR_PCT'].abs(), 99.5)) if len(d) else 1
    f.add_trace(go.Histogram(y=d['CORR_PCT'].clip(-lim, lim), nbinsy=50, name='분포',
                             marker_color=C_EMA, opacity=.7, showlegend=False), 1, 2)
    d['GB'] = pd.cut(d['GAP_D'], GAP_BINS, labels=GAP_LBL, right=False)
    g = d.groupby('GB', observed=True).agg(N=('CORR_PCT', 'size'), m=('CORR_PCT', 'mean'),
                                           a=('CORR_PCT', lambda x: x.abs().mean())).reset_index()
    x = [f'{b}<br>(N={n})' for b, n in zip(g['GB'], g['N'])]
    f.add_trace(go.Bar(x=x, y=g['a'], name='|보정률| 평균', marker_color=C_EMA), 1, 3)
    f.add_trace(go.Bar(x=x, y=g['m'], name='보정률 평균(부호)', marker_color='#85b7eb'), 1, 3)
    f.update_yaxes(range=[-lim, lim], row=1, col=1)
    f.update_yaxes(range=[-lim, lim], showticklabels=False, row=1, col=2)
    f.update_yaxes(title_text='보정률 (%)', row=1, col=1)
    #f.update_layout(title='보정률 = (EMA 노광량 − APC 노광량) / APC 노광량', barmode='group', height=420)
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
    b = t['CDERR_ACT'].abs() <= 1.0
    a = t['CDERR_SIM'].abs() <= 1.0
    t['합격_실측'] = np.where(b, 'Y', 'N')
    t['합격_EMA적용'] = np.where(a, 'Y', 'N')
    t['판정'] = np.select([b & a, ~b & a, b & ~a], ['유지합격', '구제', '손실'], '계속탈락')
    out = t[[c for c, _ in SIM_COLS if c in t.columns]].rename(columns=dict(SIM_COLS))
    out['진행시각'] = pd.to_datetime(out['진행시각']).dt.strftime('%Y-%m-%d %H:%M:%S')
    for c, nd in [('GAP(일)', 1), ('보정량(mJ)', 4), ('보정률(%)', 4), ('EMA적용시CD', 5),
                  ('CD오차_실측(%)', 3), ('CD오차_EMA적용(%)', 3)]:
        if c in out.columns:
            out[c] = out[c].round(nd)
    out = pd.concat([out, t[['합격_실측', '합격_EMA적용', '판정']]], axis=1)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    out.to_csv(path, index=False, encoding='utf-8-sig')
    print(f'[viz] 시뮬레이션 {path} ({len(out):,} Lot)')
    return os.path.basename(path)


SIM_NOTE = """<div class="note"><b>Lot 별 시뮬레이션</b> —
EMA적용시CD = 실측CD + s × (EMA노광량 − APC노광량) / (100 × MAIN_CONSTANT),
s = +1 (SPACE) / −1 (LINE). APC 가 쓰는 노광량-CD 관계(MAIN_CONSTANT)를 그대로 쓴 계산이며,
보정 상한(CD 2% 이동분) 안에서 직선 근사가 성립합니다. {link}</div>"""


C_OLD = '#1baf7a'
OLD_LBL = 'EMA 보정 (L3_EMA + BIAS_MIX8)'


def has_old(df):
    return 'U_OLD' in df.columns and df['U_OLD'].notna().any()


def _pass(u):
    return (u.abs() <= 1.0).mean() * 100


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
    f.update_layout(title='⑥ 방식 비교 — GAP 구간별 CD≤1% 합격률 (같은 Lot)', barmode='group',
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
    f.update_layout(title='방식 비교 — 장비별 CD≤1% 합격률 변화 (같은 Lot, N≥30)', barmode='group',
                    xaxis_title='합격률 변화 (%p)', height=max(380, 30 * len(t) + 140))
    return f, t


def _cards_compare(d):
    d = d[d['U_OLD'].notna()]
    if d.empty:
        return ''
    def one(col, name):
        b, a = d['U_PROC'].abs() <= 1, d[col].abs() <= 1
        g, l = int((~b & a).sum()), int((b & ~a).sum())
        return (f'<tr><td>{name}</td><td>{a.mean()*100:.2f}%</td>'
                f'<td>{(a.mean()-b.mean())*100:+.2f}%p</td><td>{g}</td><td>{l}</td>'
                f'<td>{g-l:+d}</td><td>{l/max(int(b.sum()),1)*100:.1f}%</td>'
                f'<td>{d[col].mean():+.3f}</td></tr>')
    p = (d['U_PROC'].abs() <= 1).mean() * 100
    return ('<h2>방식 비교 (같은 Lot ' + f'{len(d):,}건)</h2>'
            '<table class="cmp"><tr><th>방식</th><th>CD≤1%</th><th>변화</th><th>구제</th>'
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
            + card('간만진행 Lot', f"{s['N']:,}")
            + card('CD≤1% 합격률', f"{s['P0']:.1f}% → {s['P1']:.1f}%", f"{s['dP']:+.2f}%p")
            + card('순증', f"{s['net']:+d}", f"구제 {s['gain']} / 손실 {s['lost']}")
            + card('손실률', f"{s['lossrate']:.1f}%", '원래 합격 중 탈락 비율')
            + card('계통편향 (u 평균)', f"{s['bias0']:+.2f} → {s['bias1']:+.2f}", '0에 가까울수록 좋음')
            + card('|u| 평균', f"{s['mae0']:.2f} → {s['mae1']:.2f}")
            + card('노광량 상대오차 평균', f"{s['re0']:.3f}% → {s['re1']:.3f}%",
                   f"{(s['re0']-s['re1'])/s['re0']*100:+.1f}% 개선")
            + '</div>')


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
u = (IDEAL − PROC) / (MAIN_CONSTANT × CD_TARGET) 는 CD 오차(%)와 같으며 |u|≤1 이면 합격.
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
            p.update_layout(template='plotly_white', font=dict(size=12),
                            legend=dict(orientation='h', y=1.08),
                            margin=dict(l=60, r=20, t=70, b=50))
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
        print(f'[viz] {eq}: 간만진행 {len(d)}건 — 생략')
        return None
    s = stats(d)
    fk, key = fig_key(df, eq, args.key, args.since)
    fa, _ = fig_all_eq(df, args.since, hi=eq)
    rng = f"{d['PROC_TRANS_DATE'].min():%Y-%m-%d} ~ {d['PROC_TRANS_DATE'].max():%Y-%m-%d}"
    sim_path = os.path.splitext(path)[0] + '_simulation.csv'
    sim = write_simulation(d, sim_path)
    sim_html = SIM_NOTE.format(link=(f'<a href="{sim}">Lot 별 시뮬레이션 표 내려받기 ({sim})</a>'
                                     if sim else ''))
    write_html(path, f'EMA 효과 {eq}', [
        f'<h1>EMA 보정 효과 — {eq}</h1><p>방식: {EMA_LBL} · 간만진행(GAP≥30일) · 평가기간 {rng}</p>', NOTE,
        _cards(s, '요약'),
        fig_time(d, args.window), fig_gap(d), fig_hist(d), fig_2x2(s), fk,
        _corr_cards(d), fig_corr(d, args.window), sim_html,
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
    ap.add_argument('--out', default='report.html')
    a = ap.parse_args()
    if not a.eq and not a.all:
        ap.error('--eq 또는 --all 을 지정하십시오')

    global EMA_LBL
    if a.longgap:
        df = load_longgap(a.csv)
        mode = str(df['EMA_MODE'].dropna().iloc[0]) if 'EMA_MODE' in df.columns and df['EMA_MODE'].notna().any() else ''
        auto = 'EMA 보정 (L3_EMA + BIAS_MIX8)' if mode.startswith('RULE') else 'EMA 보정 (L3_EMA + BIAS_MIX8)'
        EMA_LBL = a.label or auto
    else:
        EMA_LBL = a.label or 'EMA 보정 (L3_EMA + BIAS_MIX8)'
        df = load(a)
    if a.all:
        os.makedirs(a.out, exist_ok=True)
        from plotly.offline import get_plotlyjs
        open(os.path.join(a.out, 'plotly.min.js'), 'w', encoding='utf-8').write(get_plotlyjs())
        fa, t = fig_all_eq(df, a.since)
        tot = stats(target(df, a.since))
        links = ''.join(f'<li><a href="{e}.html">{e}</a> — Δ {v:+.2f}%p, N={n}</li>'
                        for e, v, n in t.sort_values('dP')[['EQ', 'dP', 'N']].values)
        write_html(os.path.join(a.out, 'index.html'), 'EMA 효과 요약', [
            f'<h1>EMA 보정 효과 — 전체 장비</h1><p>방식: {EMA_LBL}</p>', NOTE,
            _cards(tot, '전체 요약'), fa,
            _cards_compare(target(df, a.since)) if has_old(df) else '',
            fig_compare_all(df, a.since)[0] if has_old(df) else None,
            f'<h2>장비별 리포트</h2><ul>{links}</ul>'], shared_js=True)
        for eq in t['EQ']:
            report_eq(df, eq, a, os.path.join(a.out, f'{eq}.html'), shared_js=True)
    else:
        s = report_eq(df, a.eq, a, a.out)
        if s:
            print(f"[viz] {a.eq}: N={s['N']} {s['P0']:.1f}% -> {s['P1']:.1f}% "
                  f"({s['dP']:+.2f}%p) 순증 {s['net']:+d}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
