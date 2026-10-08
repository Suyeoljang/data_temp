"""
PHOTO R2R 간만진행 EXPOSE 보정 — RULE 엔진

구조
  RuleState  : 축별 EMA 상태 저장소. update(관측) / query(조회) 를 분리해
               "읽고 나서 갱신"하는 인과 순서를 강제한다.
  RuleEngine : 상태 + 릿지 가중치 + 캘리브레이션 테이블을 묶은 보정기.
               fit(이력) 으로 학습, predict(신규Lot) 로 보정량 산출.

인과성 원칙 (반드시 지킬 것)
  - 보정량을 계산할 때는 그 Lot 자신의 측정 결과가 상태에 들어가면 안 된다.
    운영에서는 자연히 지켜지지만(측정은 노광 이후), 오프라인 재현 시에는
    query -> update 순서를 반드시 지켜야 한다. 어기면 성능이 +26%p 로 부풀려진다.
"""
import json
import os
import numpy as np
import pandas as pd
import config as C


# ============================================================
# 파생 변수
# ============================================================
def derive(df, verify_ideal=True):
    """원본 CSV -> 계산에 필요한 파생 변수 부착. 시간순 정렬 포함."""
    df = df.copy()
    for c in C.COL_NUM:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    if not pd.api.types.is_datetime64_any_dtype(df[C.COL_TIME]):
        df[C.COL_TIME] = pd.to_datetime(df[C.COL_TIME].astype(str),
                                        format='%Y%m%d%H%M%S', errors='coerce')
    df = df.sort_values(C.COL_TIME, kind='mergesort').reset_index(drop=True)

    # CD_TYPE 표기 정규화 ('SPACE'/'S'/'space' 등)
    ctype = df['CD_TYPE'].astype(str).str.strip().str.upper()
    is_space = ctype.isin(['SPACE', 'S'])
    is_line = ctype.isin(['LINE', 'L'])
    unk = ~(is_space | is_line)
    if unk.any():
        print(f'[derive] ⚠ CD_TYPE 미인식 {int(unk.sum()):,}행 제외 — 값: '
              f'{ctype[unk].value_counts().head(5).to_dict()}')
    s = np.where(is_space, C.SIGN_SPACE, C.SIGN_LINE)

    ideal = df['PROC_EXPOSE'] + s * (df['CD_TARGET'] - df['MEAS_CD']) * 100 * df['MAIN_CONSTANT']
    if 'IDEAL_EXPOSE' in df.columns:
        ref = pd.to_numeric(df['IDEAL_EXPOSE'], errors='coerce')
        ok = (df['MAIN_CONSTANT'] > 0) & ref.notna() & ~unk
        if verify_ideal and ok.sum():
            err = float((ideal[ok] - ref[ok]).abs().max())
            if err > 1e-4:
                raise ValueError(f'IDEAL_EXPOSE 역산 불일치 (최대오차 {err:.3e}). '
                                 'config.SIGN_SPACE/SIGN_LINE 부호 규약을 확인하십시오.')
            print(f'[derive] IDEAL 역산 검증 통과 (최대오차 {err:.2e})')
    elif verify_ideal:
        print('[derive] ⚠ IDEAL_EXPOSE 컬럼 없음 — 부호 규약 검증 생략. '
              '첫 적용 전 IDEAL 이 있는 데이터로 한 번 확인하십시오.')
    df['IDEAL_EXPOSE'] = ideal
    df['SGN'] = s

    # 유효성 — 0 나눗셈과 무효 측정을 원천 차단
    cd_rel = (df['MEAS_CD'] - df['CD_TARGET']).abs() / df['CD_TARGET'].where(df['CD_TARGET'] > 0)
    checks = {
        'MAIN_CONSTANT<=0': ~(df['MAIN_CONSTANT'] > 0),
        'CD_TARGET<=0':     ~(df['CD_TARGET'] > 0),
        'MEAS_CD<=0/결측':  ~(df['MEAS_CD'] > 0),
        'PROC_EXPOSE<=0':   ~(df['PROC_EXPOSE'] > 0),
        'CD_TYPE 미인식':    unk,
        f'CD오차>{C.MAX_CD_REL_ERR*100:.0f}%': cd_rel > C.MAX_CD_REL_ERR,
    }
    bad = np.zeros(len(df), bool)
    msg = []
    for k, m in checks.items():
        m = m.fillna(True).values if hasattr(m, 'fillna') else np.asarray(m)
        n_new = int((m & ~bad).sum())
        if n_new:
            msg.append(f'{k} {n_new:,}')
        bad |= m
    if msg:
        print(f'[derive] 무효 {int(bad.sum()):,}행 제외 ({bad.mean()*100:.1f}%) — ' + ', '.join(msg))
    df['VALID'] = ~bad & df['IDEAL_EXPOSE'].notna().values
    df['TOL'] = df['MAIN_CONSTANT'] * df['CD_TARGET']
    df['RES'] = df['IDEAL_EXPOSE'] - df['PROC_EXPOSE']
    df['RRES'] = np.where(df['VALID'], df['RES'] / df['PROC_EXPOSE'], np.nan)
    df['RTOL'] = np.where(df['VALID'], df['TOL'] / df['PROC_EXPOSE'], np.nan)
    df['U'] = np.where(df['VALID'], df['RES'] / df['TOL'], np.nan)

    df['_KEY'] = df[C.KEY_COLS[0]].astype(str)
    for c in C.KEY_COLS[1:]:
        df['_KEY'] = df['_KEY'] + '|' + df[c].astype(str)
    df['_T'] = df[C.COL_TIME].values.astype('datetime64[s]').astype(np.int64) / 86400.0
    df['GAP_D'] = df['_T'] - df.groupby('_KEY', sort=False)['_T'].shift(1)
    df['LONG'] = np.isfinite(df['GAP_D'].values) & (df['GAP_D'].values >= C.GAP_THRESHOLD)
    df['SRC'] = df['VALID'] & ~df['CD_METHOD'].isin(C.SRC_EXCLUDE_CD_METHOD)
    return df


def active_axes(df):
    """보유 컬럼으로 만들 수 있는 축만 남긴다."""
    out = {}
    for nm, cols in C.AXES.items():
        if all(c in df.columns for c in cols):
            out[nm] = cols
        elif not C.SKIP_MISSING_AXES:
            raise KeyError(f'축 {nm} 에 필요한 컬럼 없음: {cols}')
        else:
            print(f'[axes] 건너뜀: {nm} (컬럼 없음)')
    return out


def axis_key(row, cols):
    return '\x01'.join(str(row[c]) for c in cols)


def signal_names(axes):
    return [f'{nm}@{hl:g}' for nm in axes for hl in C.HALF_LIVES]


# ============================================================
# EMA 상태 저장소
# ============================================================
LN2 = np.log(2.0)


class RuleState:
    """축 a 마다 S/W/R/L 을 (키수, 반감기수) 배열로 보관."""

    def __init__(self, axes):
        self.axes = dict(axes)
        self.hls = np.array(C.HALF_LIVES, float)
        self.nh = len(self.hls)
        self.sig = signal_names(self.axes)
        self.idx = {a: {} for a in self.axes}
        self.S, self.W, self.R, self.L = {}, {}, {}, {}
        for a in self.axes:
            self._alloc(a, 1024)

    def _alloc(self, a, n):
        self.S[a] = np.zeros((n, self.nh))
        self.W[a] = np.zeros((n, self.nh))
        self.R[a] = np.zeros((n, self.nh))
        self.L[a] = np.full((n, self.nh), -1e18)

    def _row(self, a, key, create=True):
        d = self.idx[a]
        i = d.get(key)
        if i is not None:
            return i
        if not create:
            return -1
        i = len(d)
        if i >= self.S[a].shape[0]:
            for M in (self.S, self.W, self.R):
                M[a] = np.vstack([M[a], np.zeros_like(M[a])])
            self.L[a] = np.vstack([self.L[a], np.full_like(self.L[a], -1e18)])
        d[key] = i
        return i

    def _keys(self, row):
        return {a: axis_key(row, cols) for a, cols in self.axes.items()}

    def query_keys(self, keys, t):
        out = np.zeros(len(self.axes) * self.nh)
        p = 0
        for a in self.axes:
            i = self.idx[a].get(keys[a], -1)
            if i >= 0:
                lt = self.L[a][i]
                live = lt > -1e17
                if live.any():
                    we = self.W[a][i] * np.exp(-LN2 / C.HL_STALE * np.maximum(t - lt, 0.0))
                    out[p:p + self.nh] = np.where(live, self.S[a][i] * we / (we + C.SHRINK_M), 0.0)
            p += self.nh
        return out

    def update_keys(self, keys, t, obs):
        if not np.isfinite(obs):
            return
        o = float(np.clip(obs, -C.CLIP, C.CLIP))
        for a in self.axes:
            i = self._row(a, keys[a])
            lt = self.L[a][i]
            fresh = lt < -1e17
            wo = np.where(fresh, 0.0,
                          self.W[a][i] * np.exp(-LN2 / self.hls * np.maximum(t - lt, 0.0)))
            dev = o - np.where(fresh, o, self.S[a][i])
            if C.ROBUST_K > 0:
                lim = C.ROBUST_K * (np.sqrt(self.R[a][i]) + 1e-9)
                dev = np.clip(dev, -lim, lim)
            new = np.where(fresh, o, self.S[a][i] + dev / (wo + 1.0))
            self.R[a][i] = np.where(fresh, 0.0, (self.R[a][i] * wo + (o - new) ** 2) / (wo + 1.0))
            self.S[a][i] = new
            self.W[a][i] = wo + 1.0
            self.L[a][i] = t

    # 편의 래퍼 (행 dict 입력)
    def query(self, row, t):
        return self.query_keys(self._keys(row), t)

    def update(self, row, t, obs):
        self.update_keys(self._keys(row), t, obs)

    def last_key_time(self, key_axis='KEY'):
        """KEY 축 키별 최종 진행시각 (GAP 산출용)"""
        if key_axis not in self.idx:
            return {}
        return {k: float(self.L[key_axis][i].max()) for k, i in self.idx[key_axis].items()}

    def save(self, path):
        d = {'__axes__': json.dumps(self.axes)}
        for a in self.axes:
            n = len(self.idx[a])
            d[f'{a}__keys'] = np.array(list(self.idx[a].keys()), dtype=object)
            for tag, M in (('S', self.S), ('W', self.W), ('R', self.R), ('L', self.L)):
                d[f'{a}__{tag}'] = M[a][:n]
        np.savez_compressed(path, **d)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=True)
        o = cls(json.loads(str(z['__axes__'])))
        for a in o.axes:
            keys = z[f'{a}__keys']
            if len(keys) == 0:
                continue
            o.idx[a] = {str(k): i for i, k in enumerate(keys)}
            for tag, M in (('S', o.S), ('W', o.W), ('R', o.R), ('L', o.L)):
                M[a] = z[f'{a}__{tag}']
        return o


# ============================================================
# 배치 신호 생성 (fit 용, 인과 순서 보장)
# ============================================================
def build_signals(df, axes, keep_mask=None, return_state=False):
    """
    시간순 단일 패스로 각 행 시점의 신호를 만든다 (query -> update, 인과성 보장).
    keep_mask 로 필요한 행의 신호만 반환해 메모리를 절약한다.
    numba 가 있으면 자동 가속(수십 배). 결과는 동일.
    """
    import fast
    st = RuleState(axes)
    n = len(df)
    keep = np.ones(n, bool) if keep_mask is None else np.asarray(keep_mask, bool)

    # 축별 키 -> 정수코드
    names, codes, sizes, uniqs = [], [], [], {}
    for a, cc in axes.items():
        k = df[cc[0]].astype(str)
        for c in cc[1:]:
            k = k.str.cat(df[c].astype(str), sep='\x01')
        c_, u_ = pd.factorize(k, sort=False)
        names.append(a); codes.append(c_.astype(np.int64))
        sizes.append(len(u_)); uniqs[a] = u_
    CODE = np.vstack(codes)

    OBS = df['RRES'].values.astype(np.float64)
    SRC = df['SRC'].values & np.isfinite(OBS)
    if fast.HAVE_NUMBA:
        print(f'  [signals] numba 가속 · {n}행 x {len(names)}축 x {st.nh}척도', flush=True)
    else:
        print(f'  [signals] 순수 파이썬(느림) · numba 설치 권장', flush=True)
    V, S, W, R, L = fast.run_batch(
        df['_T'].values, OBS, SRC, keep, CODE, sizes, C.HALF_LIVES,
        C.HL_STALE, C.SHRINK_M, C.CLIP, C.ROBUST_K)
    V = V.astype(np.float32)

    if return_state:
        for j, a in enumerate(names):
            live = (L[j] > -1e17).any(axis=1)
            ki = np.where(live)[0]
            st.idx[a] = {str(uniqs[a][i]): p for p, i in enumerate(ki)}
            st.S[a], st.W[a] = S[j][ki], W[j][ki]
            st.R[a], st.L[a] = R[j][ki], L[j][ki]
            if len(ki) == 0:
                st._alloc(a, 1024)
        return V, st
    return V


# ============================================================
# 보정기
# ============================================================
class RuleEngine:
    def __init__(self, axes=None, weights=None, cal_scale=None, state=None):
        self.axes = axes
        self.weights = None if weights is None else np.asarray(weights, float)
        self.cal_scale = None if cal_scale is None else np.asarray(cal_scale, float)
        self.state = state

    # ---------- 학습 ----------
    def fit(self, df, verbose=True):
        """이력 전체로 릿지 가중치 + 캘리브레이션 테이블 적합."""
        from scipy.optimize import minimize
        import gc
        df = derive(df)
        self.axes = active_axes(df)
        need = set(sum(self.axes.values(), [])) | {'_T', 'RRES', 'SRC', 'RTOL', 'U',
                                                   'LONG', 'VALID', C.COL_TIME}
        df = df[[c for c in df.columns if c in need]].copy()
        gc.collect()
        V, st = build_signals(df, self.axes, return_state=True)
        self.state = st

        rtol = df['RTOL'].values
        u = df['U'].values
        t = df[C.COL_TIME].values
        t0, t1 = t.min(), t.max()
        warm_end = t0 + np.timedelta64(int(C.WARM_MONTHS * 30), 'D')
        val_start = t1 - np.timedelta64(int(C.VAL_MONTHS * 30), 'D')
        elig = df['LONG'].values & df['VALID'].values & (t >= warm_end)
        fitm = elig & (t < val_start)
        valm = elig & (t >= val_start)
        if verbose:
            print(f'[fit] 적합 {int(fitm.sum())} / 캘리브 {int(valm.sum())} 건')
        if fitm.sum() < 2000 or valm.sum() < 1000:
            raise ValueError('간만진행 표본 부족. 이력 기간을 늘리십시오.')

        # 적합/캘리브 대상 행만 정규화 (전체 행렬을 float64 로 확장하지 않는다)
        Xf = np.nan_to_num(V[fitm].astype(np.float64) / rtol[fitm][:, None], nan=0.0, posinf=0.0, neginf=0.0)
        Xv = np.nan_to_num(V[valm].astype(np.float64) / rtol[valm][:, None], nan=0.0, posinf=0.0, neginf=0.0)
        del V
        gc.collect()
        yf = u[fitm]

        def f(w):
            d = Xf @ w
            a = np.abs(yf - d)
            return (np.where(a <= 1, .5 * a * a, a - .5).mean() + C.RIDGE_L2 * (w @ w),
                    (Xf.T @ np.where(a <= 1, d - yf, np.sign(d - yf))) / len(yf) + 2 * C.RIDGE_L2 * w)
        w = minimize(f, np.zeros(Xf.shape[1]), jac=True, method='L-BFGS-B',
                     bounds=[(0, 3)] * Xf.shape[1],
                     options=dict(maxiter=C.RIDGE_MAXITER)).x

        # 전역 스케일 (VAL 기준)
        yv = u[valm]

        def obj(uu, d, lam=C.FIT_LAMBDA):
            p0 = np.abs(uu) <= 1.0
            p1 = np.abs(uu - d) <= 1.0
            return (~p0 & p1).sum() - lam * (p0 & ~p1).sum()
        alpha = max((obj(yv, a * (Xv @ w)), a) for a in np.arange(0.3, 1.85, 0.05))[1]
        w = w * alpha
        self.weights = w

        # 캘리브레이션
        dv = np.clip(Xv @ w, -C.CLAMP_RTOL, C.CLAMP_RTOL)
        bins = np.asarray(C.CAL_BINS)
        b = np.clip(np.digitize(np.abs(dv), bins) - 1, 0, len(bins) - 2)
        sc = np.ones(len(bins) - 1)
        for k in range(len(sc)):
            m = b == k
            if m.sum() < 150:
                sc[k] = 0.0
                continue
            sc[k] = max((obj(yv[m], s * dv[m]), s) for s in np.arange(0, 1.85, 0.05))[1]
        self.cal_scale = sc
        if verbose:
            nz = int((w > 1e-4).sum())
            print(f'[fit] 비영 신호 {nz}/{len(w)} · alpha={alpha:.2f} · 캘리브 {np.round(sc,2)}')
            p0 = (np.abs(yv) <= 1).mean()
            a_ = sc[b] * dv
            p1 = (np.abs(yv - a_) <= 1).mean()
            print(f'[fit] 캘리브구간 자체평가 합격률 {p0*100:.2f}% -> {p1*100:.2f}% ({(p1-p0)*100:+.3f}%p)')
        return self

    # ---------- 보정량 산출 ----------
    def replay(self, df, mask=None):
        """
        인과적 재현 — 과거 이력에 보정을 걸었다면 어땠을지 계산한다.
        시간순으로 query -> update 를 반복하므로 각 Lot 은 자기 시점까지의
        정보만 본다(운영과 동일). 최종 상태로 과거를 조회하면 미래 정보가
        섞여 결과가 왜곡되므로 반드시 이 함수를 쓸 것.

        df   : derive() 결과 (시간순 정렬됨)
        mask : 보정 대상 행. 기본 = 간만진행 & 유효
        반환 : 상대 보정량 Δ (n,)  — 대상 외 행은 0
        """
        m = (df['LONG'].values & df['VALID'].values) if mask is None \
            else np.asarray(mask, bool)
        delta = np.zeros(len(df))
        if not m.any():
            return delta
        V = build_signals(df, self.axes, keep_mask=m)
        if V.shape[1] != len(self.weights):
            raise ValueError(f'신호 수 {V.shape[1]} != 가중치 수 {len(self.weights)}. '
                             'config.HALF_LIVES/AXES 가 학습 시점과 다릅니다.')
        rtol = df['RTOL'].values[m]
        dn = np.nan_to_num(V.astype(np.float64) / rtol[:, None], nan=0.0, posinf=0.0, neginf=0.0) @ self.weights
        dn = np.clip(dn, -C.CLAMP_RTOL, C.CLAMP_RTOL)
        b = np.clip(np.digitize(np.abs(dn), np.asarray(C.CAL_BINS)) - 1,
                    0, len(self.cal_scale) - 1)
        delta[m] = self.cal_scale[b] * dn * rtol
        return delta

    def _delta_from_signal(self, vals, rtol):
        dn = float(np.nan_to_num(vals / rtol, nan=0.0, posinf=0.0, neginf=0.0) @ self.weights)
        dn = float(np.clip(dn, -C.CLAMP_RTOL, C.CLAMP_RTOL))
        bins = np.asarray(C.CAL_BINS)
        b = int(np.clip(np.digitize(abs(dn), bins) - 1, 0, len(self.cal_scale) - 1))
        return self.cal_scale[b] * dn * rtol          # 상대 보정량 Δ

    def predict_row(self, row, t, gap_days):
        """신규 Lot 1건 보정. 반환: (보정노광량, Δ, 적용여부)"""
        proc = float(row['PROC_EXPOSE'])
        tol = float(row['MAIN_CONSTANT']) * float(row['CD_TARGET'])
        if tol <= 0 or proc <= 0:
            return proc, 0.0, False
        if not (np.isfinite(gap_days) and gap_days >= C.GAP_THRESHOLD):
            return proc, 0.0, False                    # 간만진행 아님 → 미적용
        rtol = tol / proc
        d = self._delta_from_signal(self.state.query(row, t), rtol)
        q = C.QUANT.get(row.get('EXPOSE_TYPE'), C.QUANT_DEFAULT)
        return round(proc * (1 + d) / q) * q, d, True

    # ---------- 영속화 ----------
    def save(self, dirpath):
        os.makedirs(dirpath, exist_ok=True)
        json.dump(dict(axes=self.axes, weights=list(self.weights),
                       cal_scale=list(self.cal_scale),
                       sig=self.state.sig),
                  open(f'{dirpath}/model.json', 'w'), indent=1)
        self.state.save(f'{dirpath}/state.npz')
        print(f'[save] {dirpath}')

    @classmethod
    def load(cls, dirpath):
        m = json.load(open(f'{dirpath}/model.json'))
        st = RuleState.load(f'{dirpath}/state.npz')
        return cls(axes=m['axes'], weights=m['weights'], cal_scale=m['cal_scale'], state=st)
