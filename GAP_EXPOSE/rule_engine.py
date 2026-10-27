"""
PHOTO R2R Normal APC EXPOSE 보정 — RULE 엔진 (v3)

구조
  RuleState  : 축별 EMA 상태 저장소. update(관측) / query(조회) 를 분리해
               "읽고 나서 갱신"하는 인과 순서를 강제한다.
  V3State    : v3 추가 상태 — 레벨 drift(그룹 수준 D, KEY 기준선 B), L3 자기이력 EMA,
               KEY 직전 이력(IDEAL / APC 원값 / 시각).
  RuleEngine : 상태 + 가중치 + 캘리브레이션을 묶은 보정기.
               fit(이력) 으로 학습, replay(이력) 로 인과적 재현, delta_for(요청) 로 실시간 보정.

v3 신호 = [144 EMA 신호 / RTOL] + [추가 22개 / RTOL]
  추가: DRIFT_<그룹>_<반감기> 14개, K_LASTIDEAL, K_PROCCHG, L3_21, L3_180

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
    # 일 배치 번호 (DAY_BOUNDARY_HOUR 기준). 같은 DAY 는 전일 배치까지의 상태만 본다
    df['DAY'] = day_index(df['_T'].values)
    with np.errstate(divide='ignore', invalid='ignore'):
        df['Y_LOG'] = np.where(df['VALID'] & (df['IDEAL_EXPOSE'] > 0),
                               np.log(df['IDEAL_EXPOSE'].clip(lower=1e-9)), np.nan)
    return df


def day_index(t_days):
    """일(day) 단위 시각 -> 일 배치 번호. 경계 = 매일 DAY_BOUNDARY_HOUR 시."""
    return np.floor(np.asarray(t_days, float) - C.DAY_BOUNDARY_HOUR / 24.0).astype(np.int64)


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

    def save(self, path, extra=None):
        d = {'__axes__': json.dumps(self.axes)}
        if extra:
            d.update(extra)
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
def build_signals(df, axes, keep_mask=None, return_state=False, daily=False):
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
        C.HL_STALE, C.SHRINK_M, C.CLIP, C.ROBUST_K,
        day=df['DAY'].values if daily else None)
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
# v3 추가 상태 (레벨 drift / L3 자기이력 / KEY 직전 이력)
# ============================================================
def extra_names():
    """추가 신호 이름 (가중치 순서와 동일)"""
    names = [f'DRIFT_{g}_{h:g}' for g in C.LEVEL_GROUPS for h in C.LEVEL_HLS]
    names += ['K_LASTIDEAL', 'K_PROCCHG']
    names += [f'L3_{h:g}' for h in C.L3_HLS]
    return names


def _join(df, cols):
    k = df[cols[0]].astype(str)
    for c in cols[1:]:
        k = k.str.cat(df[c].astype(str), sep='\x01')
    return k


def _level_specs():
    """(그룹이름, 컬럼, 반감기) — 추종기 J 개"""
    return [(g, cols, h) for g, cols in C.LEVEL_GROUPS.items() for h in C.LEVEL_HLS]


class V3State:
    """
    gidx : '{j}\\x02{그룹키}' -> 레벨 그룹 인덱스      D, TG (NG,)
    kidx : KEY 문자열 -> 인덱스                        B, KD, HK (NKEY, J) / KL_IDEAL, KL_APC, KL_T (NKEY,)
    lidx : L3 키 -> 인덱스                              S3, W3, L3 (NL, len(L3_HLS))
    """

    def __init__(self):
        J = len(_level_specs())
        nl = len(C.L3_HLS)
        self.gidx, self.kidx, self.lidx = {}, {}, {}
        self.D = np.zeros(0); self.TG = np.zeros(0)
        self.B = np.zeros((0, J)); self.KD = np.zeros((0, J)); self.HK = np.zeros((0, J), bool)
        self.KL_IDEAL = np.zeros(0); self.KL_APC = np.zeros(0); self.KL_T = np.zeros(0)
        self.S3 = np.zeros((0, nl)); self.W3 = np.zeros((0, nl)); self.L3 = np.zeros((0, nl))

    # ---------- 인덱스 ----------
    @staticmethod
    def _grow(a, n, fill):
        if a.shape[0] >= n:
            return a
        new = max(n, int(a.shape[0] * 1.5) + 256)
        out = np.full((new,) + a.shape[1:], fill, dtype=a.dtype)
        out[:a.shape[0]] = a
        return out

    def _gi(self, j, key, create):
        k = f'{j}\x02{key}'
        i = self.gidx.get(k)
        if i is None and create:
            i = len(self.gidx); self.gidx[k] = i
            self.D = self._grow(self.D, i + 1, 0.0); self.TG = self._grow(self.TG, i + 1, -1e18)
        return -1 if i is None else i

    def _ki(self, key, create):
        i = self.kidx.get(key)
        if i is None and create:
            i = len(self.kidx); self.kidx[key] = i
            for nm, fill in (('B', 0.0), ('KD', 0.0), ('HK', False),
                             ('KL_IDEAL', np.nan), ('KL_APC', np.nan), ('KL_T', np.nan)):
                setattr(self, nm, self._grow(getattr(self, nm), i + 1, fill))
        return -1 if i is None else i

    def _li(self, key, create):
        i = self.lidx.get(key)
        if i is None and create:
            i = len(self.lidx); self.lidx[key] = i
            self.S3 = self._grow(self.S3, i + 1, 0.0); self.W3 = self._grow(self.W3, i + 1, 0.0)
            self.L3 = self._grow(self.L3, i + 1, -1e18)
        return -1 if i is None else i

    # ---------- 조회 (요청 1건) ----------
    def query(self, row, proc):
        """추가 신호(상대값, RTOL 정규화 전). 상태 없으면 0."""
        key = '\x01'.join(str(row.get(c)) for c in C.KEY_COLS)
        ki = self._ki(key, False)
        out = []
        for j, (g, cols, h) in enumerate(_level_specs()):
            gi = self._gi(j, '\x01'.join(str(row.get(c)) for c in cols), False)
            out.append(self.D[gi] - self.KD[ki, j] if (ki >= 0 and gi >= 0 and self.HK[ki, j]) else 0.0)
        if ki >= 0 and np.isfinite(self.KL_IDEAL[ki]) and self.KL_IDEAL[ki] > 0:
            out.append(np.log(self.KL_IDEAL[ki] / proc))
        else:
            out.append(0.0)
        if ki >= 0 and np.isfinite(self.KL_APC[ki]) and self.KL_APC[ki] > 0:
            out.append(np.log(proc / self.KL_APC[ki]))
        else:
            out.append(0.0)
        li = self._li('\x01'.join(str(row.get(c)) for c in C.L3_COLS), False)
        for j in range(len(C.L3_HLS)):
            if li >= 0 and self.L3[li, j] > -1e17 and self.S3[li, j] > 0:
                out.append(np.log(self.S3[li, j] / proc))
            else:
                out.append(0.0)
        return np.asarray(out, float)

    # ---------- 갱신 (일 배치, 시간순 1행) ----------
    def update(self, row, t, ideal, apc, src):
        """ideal: 사용값 기준 IDEAL(=APC 무관 물리량), apc: Normal APC 원값, src: 상태갱신 대상(FIX 아님)"""
        good = bool(np.isfinite(ideal) and ideal > 0)
        src = bool(src and good)
        key = '\x01'.join(str(row.get(c)) for c in C.KEY_COLS)
        k = self._ki(key, True)
        y = np.log(ideal) if good else 0.0
        for j, (g, cols, h) in enumerate(_level_specs()):
            gi = self._gi(j, '\x01'.join(str(row.get(c)) for c in cols), True)
            if src:
                if self.HK[k, j]:
                    res = float(np.clip(y - self.B[k, j] - self.D[gi], -C.LEVEL_CLIP, C.LEVEL_CLIP))
                    a = 0.0
                    if self.TG[gi] > -1e17:
                        a = 1.0 - np.exp(-LN2 * max(t - self.TG[gi], 0.0) / h)
                    a = max(a, C.LEVEL_MIN_ALPHA)
                    self.D[gi] += a * res
                    self.TG[gi] = t
                    self.B[k, j] += C.LEVEL_KEY_ALPHA * (y - self.B[k, j] - self.D[gi])
                else:
                    self.B[k, j] = y - self.D[gi]
                    self.HK[k, j] = True
                    if self.TG[gi] < -1e17:
                        self.TG[gi] = t
            self.KD[k, j] = self.D[gi]
        if not good:
            return
        if src:
            li = self._li('\x01'.join(str(row.get(c)) for c in C.L3_COLS), True)
            for j, h in enumerate(C.L3_HLS):
                if self.L3[li, j] > -1e17:
                    w = self.W3[li, j] * np.exp(-LN2 / h * max(t - self.L3[li, j], 0.0))
                    self.S3[li, j] = (self.S3[li, j] * w + ideal) / (w + 1.0)
                    self.W3[li, j] = w + 1.0
                else:
                    self.S3[li, j] = ideal; self.W3[li, j] = 1.0
                self.L3[li, j] = t
        self.KL_IDEAL[k] = ideal
        self.KL_APC[k] = apc
        self.KL_T[k] = t

    # ---------- 저장 ----------
    def to_npz_dict(self):
        d = {}
        for nm, idx in (('g', self.gidx), ('k', self.kidx), ('l', self.lidx)):
            d[f'v3__{nm}keys'] = np.array(list(idx.keys()), dtype=object)
        ng, nk, nl = len(self.gidx), len(self.kidx), len(self.lidx)
        for nm, n in (('D', ng), ('TG', ng), ('B', nk), ('KD', nk), ('HK', nk), ('KL_IDEAL', nk),
                      ('KL_APC', nk), ('KL_T', nk), ('S3', nl), ('W3', nl), ('L3', nl)):
            d[f'v3__{nm}'] = getattr(self, nm)[:n]
        return d

    @classmethod
    def from_npz(cls, z):
        o = cls()
        if 'v3__kkeys' not in z.files:
            return None
        for nm, attr in (('g', 'gidx'), ('k', 'kidx'), ('l', 'lidx')):
            setattr(o, attr, {str(k): i for i, k in enumerate(z[f'v3__{nm}keys'])})
        for nm in ('D', 'TG', 'B', 'KD', 'HK', 'KL_IDEAL', 'KL_APC', 'KL_T', 'S3', 'W3', 'L3'):
            setattr(o, nm, np.array(z[f'v3__{nm}']))
        return o


# ============================================================
# β 배율 상태 (v3.1) — 세그먼트별 "보정량 대비 실제 오차" 비율
# ============================================================
def eq_target(eq):
    """PROC_EQ 값(스칼라 또는 Series)이 적용 대상 장비인가 (config.APPLY_EQ_PREFIX)"""
    pf = tuple(C.APPLY_EQ_PREFIX or ())
    if isinstance(eq, pd.Series):
        return eq.astype(str).str.startswith(pf).values if pf else np.ones(len(eq), bool)
    return True if not pf else str(eq).startswith(pf)


def beta_update_mask(df):
    """β 갱신 대상: 유효 & FIX 아님 & SC_IN_AI 아님 & U 유한 (평가 기준과 동일)"""
    m = df['VALID'].values & np.isfinite(df['U'].values)
    m &= ~df['CD_METHOD'].astype(str).isin(C.SRC_EXCLUDE_CD_METHOD).values
    if 'CD_LOGIC' in df.columns and C.BETA_EXCLUDE_CD_LOGIC:
        m &= ~df['CD_LOGIC'].astype(str).isin(C.BETA_EXCLUDE_CD_LOGIC).values
    return m


class BetaState:
    """
    계층 l 마다  idx[l] : 세그먼트 키 -> 인덱스,  A[l], B[l] : 지수가중 합,  T[l] : 마지막 갱신 DAY
    조회 β = 계층 순서대로 (A·f + K·β_상위)/(B·f + K),  f = exp(-λ·(DAY - T))
    """

    def __init__(self, levels=None, hl=None, K=None, lo=None, hi=None):
        self.levels = [list(c) for c in (levels or C.BETA_LEVELS)]
        self.hl = float(C.BETA_HL if hl is None else hl)
        self.K = float(C.BETA_K if K is None else K)
        self.lo = float(C.BETA_MIN if lo is None else lo)
        self.hi = float(C.BETA_MAX if hi is None else hi)
        L = len(self.levels)
        self.idx = [dict() for _ in range(L)]
        self.A = [np.zeros(0) for _ in range(L)]
        self.B = [np.zeros(0) for _ in range(L)]
        self.T = [np.zeros(0) for _ in range(L)]

    @property
    def lam(self):
        return np.log(2.0) / self.hl

    def _k(self, l, key, create):
        i = self.idx[l].get(key)
        if i is None and create:
            i = len(self.idx[l]); self.idx[l][key] = i
            self.A[l] = V3State._grow(self.A[l], i + 1, 0.0)
            self.B[l] = V3State._grow(self.B[l], i + 1, 0.0)
            self.T[l] = V3State._grow(self.T[l], i + 1, -1e18)
        return -1 if i is None else i

    def _key(self, row, l):
        return '\x01'.join(str(row.get(c)) for c in self.levels[l])

    def query(self, row, day):
        b = 1.0
        for l in range(len(self.levels)):
            k = self._k(l, self._key(row, l), False)
            a = bb = 0.0
            if k >= 0 and self.T[l][k] > -1e17:
                f = np.exp(-self.lam * (day - self.T[l][k]))
                a, bb = self.A[l][k] * f, self.B[l][k] * f
            b = (a + self.K * b) / (bb + self.K)
        return float(min(max(b, self.lo), self.hi))

    def update(self, row, day, s, u):
        u = float(np.clip(u, -C.BETA_UCLIP, C.BETA_UCLIP))
        for l in range(len(self.levels)):
            k = self._k(l, self._key(row, l), True)
            if self.T[l][k] > -1e17:
                f = np.exp(-self.lam * (day - self.T[l][k]))
                self.A[l][k] *= f; self.B[l][k] *= f
            self.A[l][k] += s * u
            self.B[l][k] += s * s
            self.T[l][k] = day

    def params(self):
        return dict(levels=self.levels, hl=self.hl, K=self.K, lo=self.lo, hi=self.hi)

    def to_npz_dict(self):
        d = {'beta__params': json.dumps(self.params())}
        for l in range(len(self.levels)):
            n = len(self.idx[l])
            d[f'beta__{l}__keys'] = np.array(list(self.idx[l].keys()), dtype=object)
            d[f'beta__{l}__A'] = self.A[l][:n]
            d[f'beta__{l}__B'] = self.B[l][:n]
            d[f'beta__{l}__T'] = self.T[l][:n]
        return d

    @classmethod
    def from_npz(cls, z):
        if 'beta__params' not in z.files:
            return None
        p = json.loads(str(z['beta__params']))
        o = cls(p['levels'], p['hl'], p['K'], p['lo'], p['hi'])
        for l in range(len(o.levels)):
            keys = z[f'beta__{l}__keys']
            o.idx[l] = {str(k): i for i, k in enumerate(keys)}
            o.A[l] = np.array(z[f'beta__{l}__A'], float)
            o.B[l] = np.array(z[f'beta__{l}__B'], float)
            o.T[l] = np.array(z[f'beta__{l}__T'], float)
        return o


def beta_replay(sub, s, upd):
    """
    일괄 β 재현 (인과적, 일 배치 의미).
    sub : 보정 대상 행만 (시간순), 필요 컬럼 = BETA_LEVELS 컬럼, DAY, U
    s   : 원 보정량 (u 단위), upd : β 갱신 대상 여부
    반환: (β (len(sub),), 최종 BetaState)
    """
    from fast import _run_beta
    st = BetaState()
    L = len(st.levels)
    n = len(sub)
    KID = np.full((n, L), -1, np.int64)
    uniq = []
    for l, cols in enumerate(st.levels):
        codes, u = pd.factorize(_join(sub, cols), sort=False)
        KID[:, l] = codes
        uniq.append(u)
    nkmax = max(1, max(len(u) for u in uniq))
    U = np.clip(np.nan_to_num(sub['U'].values.astype(np.float64)), -C.BETA_UCLIP, C.BETA_UCLIP)
    beta, A, B, TT = _run_beta(sub['DAY'].values.astype(np.float64), KID,
                               np.asarray(s, np.float64), U,
                               np.ones(n, np.bool_), np.asarray(upd, np.bool_),
                               nkmax, st.lam, st.K, st.lo, st.hi)
    for l in range(L):
        nk = len(uniq[l])
        st.idx[l] = {str(k): i for i, k in enumerate(uniq[l])}
        st.A[l], st.B[l], st.T[l] = A[l, :nk].copy(), B[l, :nk].copy(), TT[l, :nk].copy()
    return beta, st


def build_extra(df, keep, return_state=False):
    """
    추가 신호 일괄 생성 (일 배치 의미, 인과성 보장).
    반환: E (n_keep, 22) 상대값(정규화 전), [V3State]
    """
    import fast
    keep = np.asarray(keep, bool)
    n_keep = int(keep.sum())
    kpos = (np.cumsum(keep) - 1).astype(np.int64)
    T = df['_T'].values.astype(np.float64)
    DAY = df['DAY'].values.astype(np.int64)
    valid = df['VALID'].values & np.isfinite(df['Y_LOG'].values)
    src = df['SRC'].values & valid
    proc = df['PROC_EXPOSE'].values.astype(np.float64)
    apc = (df['APC_EXPOSE'] if 'APC_EXPOSE' in df.columns else df['PROC_EXPOSE']).values.astype(np.float64)
    ideal = df['IDEAL_EXPOSE'].values.astype(np.float64)

    key = _join(df, C.KEY_COLS)
    kid, kuniq = pd.factorize(key, sort=False)
    specs = _level_specs()
    GID = np.zeros((len(specs), len(df)), np.int64)
    gkeys = []
    off = 0
    for j, (g, cols, h) in enumerate(specs):
        c_, u_ = pd.factorize(_join(df, cols), sort=False)
        GID[j] = c_ + off
        gkeys += [f'{j}\x02{x}' for x in u_]
        off += len(u_)
    # 레벨: 유효 행만 참여 (무효 행은 KEY/그룹에 영향 없음)
    Vlv, D, TG, B, KD, HK = fast._run_level(
        T, DAY, np.nan_to_num(df['Y_LOG'].values), src, keep & np.ones(len(df), bool), kpos,
        GID, kid.astype(np.int64), off, len(kuniq), np.array([h for _, _, h in specs], float),
        float(C.LEVEL_CLIP), float(C.LEVEL_KEY_ALPHA), float(C.LEVEL_MIN_ALPHA), n_keep)
    # 무효 행의 KD 기록은 커널에서 일어나므로, 무효 행은 SRC=False 로 이미 갱신 제외됨
    lid, luniq = pd.factorize(_join(df, C.L3_COLS), sort=False)
    Vl3, S3, W3, L3 = fast._run_l3(T, DAY, np.nan_to_num(ideal), src, keep, kpos,
                                   lid.astype(np.int64), len(luniq), np.asarray(C.L3_HLS, float), n_keep)
    # KEY 직전 이력: DAY < 현재 DAY 인 마지막 유효 행
    h = pd.DataFrame({'k': kid, 'DAY': DAY, 'I': ideal, 'A': apc, 'T': T})[valid]
    last = h.groupby(['k', 'DAY'], sort=False).last().reset_index().sort_values('DAY')
    q = pd.DataFrame({'k': kid[keep], 'DAY': DAY[keep], 'pos': np.arange(n_keep)}).sort_values('DAY')
    m = pd.merge_asof(q, last, on='DAY', by='k', allow_exact_matches=False).sort_values('pos')
    pk = proc[keep]
    with np.errstate(divide='ignore', invalid='ignore'):
        k_lastideal = np.log(m['I'].values / pk)
        k_procchg = np.log(pk / m['A'].values)
        l3 = np.log(Vl3 / pk[:, None])
    E = np.column_stack([Vlv, k_lastideal, k_procchg, l3])
    E = np.nan_to_num(E, nan=0.0, posinf=0.0, neginf=0.0)
    if not return_state:
        return E
    st = V3State()
    st.gidx = {k: i for i, k in enumerate(gkeys)}
    st.D, st.TG = D, TG
    st.kidx = {str(k): i for i, k in enumerate(kuniq)}
    st.B, st.KD, st.HK = B, KD, HK
    fin = h.groupby('k', sort=False).last()
    st.KL_IDEAL = np.full(len(kuniq), np.nan); st.KL_APC = np.full(len(kuniq), np.nan)
    st.KL_T = np.full(len(kuniq), np.nan)
    st.KL_IDEAL[fin.index.values] = fin['I'].values
    st.KL_APC[fin.index.values] = fin['A'].values
    st.KL_T[fin.index.values] = fin['T'].values
    st.lidx = {str(k): i for i, k in enumerate(luniq)}
    st.S3, st.W3, st.L3 = S3, W3, L3
    return E, st


def assemble(V, E, rtol):
    """신호 정규화: [V/RTOL, clip(E/RTOL)] -> clip(FEAT_CLIP)"""
    rt = np.asarray(rtol, float)
    with np.errstate(divide='ignore', invalid='ignore'):
        Xv = np.asarray(V, np.float64) / rt[:, None]
        Xe = np.clip(np.asarray(E, np.float64) / rt[:, None], -C.EXTRA_CLIP, C.EXTRA_CLIP)
    X = np.nan_to_num(np.hstack([Xv, Xe]), nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(X, -C.FEAT_CLIP, C.FEAT_CLIP)


# ============================================================
# 가중치 적합 (v3)
# ============================================================
def _nn_ridge(X, y, lam, n_pos):
    """앞 n_pos 개 계수는 >=0, 나머지는 부호 자유. L2 = lam*n."""
    from scipy.optimize import nnls
    p = X.shape[1]
    free = list(range(n_pos, p))
    X2 = np.hstack([X, -X[:, free]])
    P = X2.shape[1]
    A = np.vstack([X2, np.sqrt(lam * len(y)) * np.eye(P)])
    b = np.r_[y, np.zeros(P)]
    w, _ = nnls(A, b, maxiter=20000)
    out = w[:p].copy()
    out[free] -= w[p:]
    return out


def _pass_opt(X, y, w0):
    """
    |y-Xw| <= FIT_PASS_BAND 비율의 시그모이드 근사를 최대화 = 그 기준의 순증(구제-손실) 최대화.
    FIT_PASS_BAND = 2.0 이면 "불량(>2%) 구제 − 불량 발생" 을 직접 최적화한다.
    """
    from scipy.optimize import minimize
    n = len(y)
    tau, lam = C.PASS_TAU, C.PASS_LAM
    band = float(getattr(C, 'FIT_PASS_BAND', 1.0))

    def f(w):
        r = y - X @ w
        a = np.sqrt(r * r + 1e-4)
        p = 1.0 / (1.0 + np.exp(-(band - a) / tau))
        dp = p * (1 - p) / tau
        return (-p.mean() + lam * np.sum((w - w0) ** 2) / len(w),
                -(X.T @ (dp * r / a)) / n + 2 * lam * (w - w0) / len(w))
    return minimize(f, w0, jac=True, method='L-BFGS-B',
                    options=dict(maxiter=C.PASS_MAXITER)).x


def _cal_obj(uu, d, lam):
    p0 = np.abs(uu) <= 1.0
    p1 = np.abs(uu - d) <= 1.0
    return (~p0 & p1).sum() - lam * (p0 & ~p1).sum()


def _fit_calibration(Xv, yv, w, clamp, verbose=True):
    """v2 와 같은 방식: 전역 배율 alpha + |dn| 구간별 스케일. 목적 = 구제 - FIT_LAMBDA*손실"""
    alpha = max((_cal_obj(yv, a * (Xv @ w), C.FIT_LAMBDA), a) for a in np.arange(0.3, 1.85, 0.05))[1]
    w = w * alpha
    dv = np.clip(Xv @ w, -clamp, clamp)
    bins = np.asarray(C.CAL_BINS)
    b = np.clip(np.digitize(np.abs(dv), bins) - 1, 0, len(bins) - 2)
    sc = np.ones(len(bins) - 1)
    for k in range(len(sc)):
        m = b == k
        if m.sum() < 150:
            sc[k] = 0.0
            continue
        sc[k] = max((_cal_obj(yv[m], s_ * dv[m], C.FIT_LAMBDA), s_) for s_ in np.arange(0, 1.85, 0.05))[1]
    if verbose:
        print(f'[fit] alpha={alpha:.2f} · 캘리브 {np.round(sc, 2)}')
    return w, sc


# ============================================================
# 보정기
# ============================================================
class RuleEngine:
    def __init__(self, axes=None, weights=None, cal_scale=None, state=None,
                 v3=None, version='v2', clamp=None, mode='rule', beta=None, apply_min_u=0.0):
        self.axes = axes
        self.weights = None if weights is None else np.asarray(weights, float)
        self.cal_scale = None if cal_scale is None else np.asarray(cal_scale, float)
        self.state = state
        self.v3 = v3
        self.version = version
        self.mode = mode
        self.clamp = C.CLAMP_RTOL if clamp is None else float(clamp)
        self.beta = beta                      # BetaState 또는 None(β=1)
        self.apply_min_u = float(apply_min_u or 0.0)

    @property
    def is_v3(self):
        return self.version == 'v3'

    @property
    def use_beta(self):
        return self.beta is not None

    # ---------- 학습 ----------
    def fit(self, df, verbose=True):
        if C.USE_V3:
            return self._fit_v3(df, verbose)
        return self._fit_v2(df, verbose)

    def _fit_v3(self, df, verbose=True):
        import gc
        df = derive(df)
        # 마지막 일 배치(DAY_BOUNDARY_HOUR 이후 ~ 데이터 끝)는 하루가 다 차지 않았으므로 제외한다.
        # 상태가 일 배치 경계에서 끝나야 다음 daily_update(경계 ~ 경계) 와 겹치거나 비지 않는다.
        last = df['DAY'].values.max()
        cut = df['DAY'].values < last
        if cut.sum() < len(df):
            b = pd.Timestamp(0) + pd.Timedelta(days=float(last)) + pd.Timedelta(hours=C.DAY_BOUNDARY_HOUR)
            if verbose:
                print(f'[fit] 마지막 일 배치({b:%Y-%m-%d %H:%M} 이후) {int((~cut).sum()):,}행 제외 — '
                      f'상태는 {b:%Y-%m-%d %H:%M} 까지. 다음 daily_update 는 이 시각부터 처리')
            df = df[cut].reset_index(drop=True)
        self.state_until = (pd.Timestamp(0) + pd.Timedelta(days=float(last))
                            + pd.Timedelta(hours=C.DAY_BOUNDARY_HOUR)).strftime('%Y%m%d%H%M%S')
        self.axes = active_axes(df)
        t = df[C.COL_TIME].values
        t0, t1 = t.min(), t.max()
        warm_end = t0 + np.timedelta64(int(C.WARM_MONTHS * 30), 'D')
        gap = df['GAP_D'].values
        rng = np.random.default_rng(0)
        hasgap = np.isfinite(gap)
        keep = df['VALID'].values & hasgap & (
            (gap >= C.FIT_GAP_FULL) | (rng.random(len(df)) < C.FIT_SHORT_SAMPLE))
        V, st = build_signals(df, self.axes, keep_mask=keep, return_state=True, daily=True)
        E, st3 = build_extra(df, keep, return_state=True)
        self.state, self.v3 = st, st3
        rtol = df['RTOL'].values[keep]
        X = assemble(V, E, rtol).astype(np.float64)
        del V, E
        gc.collect()
        u = df['U'].values[keep]
        tk = t[keep]
        fitok = df['SRC'].values[keep] & (tk >= warm_end) & np.isfinite(u)
        n_pos = len(self.axes) * len(C.HALF_LIVES)
        if C.FIT_MODE == 'pass':
            fm = fitok
            if verbose:
                print(f'[fit] v3/pass · 적합 {int(fm.sum()):,}건 (GAP>={C.FIT_GAP_FULL:g} 전부 + 미만 '
                      f'{C.FIT_SHORT_SAMPLE*100:.0f}% 표본)')
            w0 = _nn_ridge(X[fm], np.clip(u[fm], -C.Y_CLIP, C.Y_CLIP), C.NN_RIDGE_LAM, n_pos)
            w = _pass_opt(X[fm], u[fm], w0)
            self.weights, self.cal_scale, self.clamp = w, np.ones(len(C.CAL_BINS) - 1), C.CLAMP_PASS
            if verbose:
                bd = float(getattr(C, 'FIT_PASS_BAND', 1.0))
                p0 = (np.abs(u[fm]) <= bd).mean()
                p1 = (np.abs(u[fm] - np.clip(X[fm] @ w, -self.clamp, self.clamp)) <= bd).mean()
                print(f'[fit] 학습 기준 ±{bd:g}% · 학습구간 자체평가 {p0*100:.2f}% -> {p1*100:.2f}% ({(p1-p0)*100:+.3f}%p, in-sample)')
        else:
            val_start = t1 - np.timedelta64(int(C.VAL_MONTHS * 30), 'D')
            fm = fitok & (tk < val_start)
            vm = fitok & (tk >= val_start)
            if verbose:
                print(f'[fit] v3/ridge · 적합 {int(fm.sum()):,} / 캘리브 {int(vm.sum()):,}건')
            w = _nn_ridge(X[fm], np.clip(u[fm], -C.Y_CLIP, C.Y_CLIP), C.NN_RIDGE_LAM, n_pos)
            self.weights, self.cal_scale = _fit_calibration(X[vm], u[vm], w, C.CLAMP_RTOL, verbose)
            self.clamp = C.CLAMP_RTOL
        self.version, self.mode = 'v3', C.FIT_MODE
        if verbose:
            print(f'[fit] 신호 {len(self.weights)}개 (EMA {n_pos} + 추가 {len(self.weights)-n_pos}) · '
                  f'비영 {int((np.abs(self.weights) > 1e-6).sum())}')
        del X
        gc.collect()
        self._fit_apply(df, warm_end, verbose)
        return self

    def _fit_apply(self, df, warm_end, verbose=True):
        """
        학습 구간을 인과적으로 재현해 β 상태(학습 끝 시점)와 적용 임계값을 정한다.
          - β 상태: 운영이 학습 직후부터 이어서 쓰는 초기 상태
          - 임계값: 대상 Lot 의 |β·보정량| 상위 APPLY_RATIO 만 적용되도록 (APPLY_MIN_U 가 있으면 그 값)
        """
        self.beta, self.apply_min_u = None, 0.0
        m = self.default_mask(df)
        if not m.any():
            print('[fit] ⚠ 적용 대상 장비 행 없음 — β·임계값 생략 (APPLY_EQ_PREFIX 확인)')
            return
        info = self._replay_core(df, m, beta_on=C.BETA_ON)
        if C.BETA_ON:
            self.beta = info['beta_state']
        fu = np.abs(info['beta'] * info['base_u'])
        sub_t = df[C.COL_TIME].values[m]
        sel = info['upd'] & (sub_t >= warm_end) & (fu > 0)
        if C.APPLY_MIN_U is not None:
            self.apply_min_u = float(C.APPLY_MIN_U)
        elif sel.sum() >= 1000 and C.APPLY_RATIO < 1.0:
            self.apply_min_u = float(np.quantile(fu[sel], 1.0 - C.APPLY_RATIO))
        if verbose:
            b = info['beta'][sel] if sel.any() else np.array([1.0])
            print(f'[fit] 적용 장비 {C.APPLY_EQ_PREFIX or "전체"} · 대상 {int(m.sum()):,}건 · '
                  f'β {"ON" if C.BETA_ON else "OFF"} (중앙 {np.median(b):.2f}, 5~95% '
                  f'{np.percentile(b, 5):.2f}~{np.percentile(b, 95):.2f}) · '
                  f'적용 임계 |β·보정| >= {self.apply_min_u:.4f} CD% '
                  f'(학습구간 적용률 {(fu[sel] >= self.apply_min_u).mean()*100 if sel.any() else 0:.1f}%)')

    def _fit_v2(self, df, verbose=True):
        """v2: 간만진행만으로 Huber 릿지 + 캘리브레이션 (행 단위 재현)"""
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
            print(f'[fit] v2 · 적합 {int(fitm.sum())} / 캘리브 {int(valm.sum())} 건')
        if fitm.sum() < 2000 or valm.sum() < 1000:
            raise ValueError('간만진행 표본 부족. 이력 기간을 늘리십시오.')
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
        self.weights, self.cal_scale = _fit_calibration(Xv, u[valm], w, C.CLAMP_RTOL, verbose)
        self.version, self.mode, self.clamp = 'v2', 'rule', C.CLAMP_RTOL
        return self

    # ---------- 보정량 산출 ----------
    def _base_u(self, dn):
        """정규화 원시 보정량 dn -> clamp·캘리브 적용 보정량 (u = CD% 단위)"""
        dn = np.clip(dn, -self.clamp, self.clamp)
        b = np.clip(np.digitize(np.abs(dn), np.asarray(C.CAL_BINS)) - 1, 0, len(self.cal_scale) - 1)
        return self.cal_scale[b] * dn

    def _finish(self, dn, rtol, beta=1.0):
        """보정량 dn -> 상대 보정량 Δ (β 배율, 작은 보정 생략 포함)"""
        fu = np.asarray(beta) * self._base_u(dn)
        fu = np.where(np.abs(fu) >= self.apply_min_u, fu, 0.0) * C.APPLY_SCALE   # 판정은 배율 적용 전
        return fu * rtol

    def default_mask(self, df):
        """보정 대상: 유효 & 이력 있음 & GAP >= GAP_THRESHOLD & 적용 장비(APPLY_EQ_PREFIX)"""
        gap = df['GAP_D'].values
        return (df['VALID'].values & np.isfinite(gap) & (gap >= C.GAP_THRESHOLD)
                & eq_target(df['PROC_EQ']))

    def beta_mask(self, df):
        """β 학습 대상: 유효 & 이력 있음 & GAP >= BETA_GAP_MIN & 적용 장비 (적용 범위 GAP_THRESHOLD 와 별개)"""
        gap = df['GAP_D'].values
        return (df['VALID'].values & np.isfinite(gap) & (gap >= C.BETA_GAP_MIN)
                & eq_target(df['PROC_EQ']))

    def _replay_core(self, df, m, beta_on=None):
        """
        대상 행 m 에 대한 원시 보정량·β 계산. 반환 dict (대상 행 길이 배열)
        β 는 적용 범위(m)보다 넓은 beta_mask 행으로 학습할 수 있다
        (예: GAP>=30 에만 적용하되 β 는 같은 장비 전 GAP Lot 으로 추정).
        """
        beta_on = self.use_beta if beta_on is None else beta_on
        m = np.asarray(m, bool)
        U = m | self.beta_mask(df) if beta_on else m
        info = self._replay_rows(df, U, beta_on)
        sel = m[U]
        out = {k: (v[sel] if isinstance(v, np.ndarray) else v) for k, v in info.items()}
        return out

    def _replay_rows(self, df, m, beta_on):
        V = build_signals(df, self.axes, keep_mask=m, daily=self.is_v3)
        rtol = df['RTOL'].values[m]
        E = build_extra(df, m) if self.is_v3 else None
        nsig = V.shape[1] + (0 if E is None else E.shape[1])
        if nsig != len(self.weights):
            raise ValueError(f'신호 수 {nsig} != 가중치 수 {len(self.weights)}. '
                             'config.HALF_LIVES/AXES/LEVEL_GROUPS/L3_HLS 가 학습 시점과 다릅니다.')
        dn = np.empty(len(rtol))
        CH = 200000                                   # 메모리 절약: 구간별 정규화·가중합
        for s0 in range(0, len(rtol), CH):
            sl = slice(s0, s0 + CH)
            if self.is_v3:
                X = assemble(V[sl], E[sl], rtol[sl])
            else:
                X = np.nan_to_num(V[sl].astype(np.float64) / rtol[sl, None], nan=0.0, posinf=0.0, neginf=0.0)
            dn[sl] = X @ self.weights
        del V, E
        base = self._base_u(dn)
        sub = df.loc[m, list(dict.fromkeys(sum(C.BETA_LEVELS, []))) + ['DAY', 'U']]
        upd = (beta_update_mask(df) & self.beta_mask(df))[m]
        if beta_on:
            beta, bst = beta_replay(sub, base, upd)
        else:
            beta, bst = np.ones(len(base)), None
        return dict(dn=dn, base_u=base, beta=beta, rtol=rtol, upd=upd, beta_state=bst)

    def replay(self, df, mask=None, return_dn=False, return_info=False):
        """
        인과적 재현 — 과거 이력에 보정을 걸었다면 어땠을지 계산한다.
        v3: 운영과 같은 일 배치 의미(같은 DAY 는 전일까지의 상태). v2: 행 단위.
        β 상태도 입력 이력의 처음부터 새로 쌓는다(EMA 상태와 동일).
        df   : derive() 결과 (시간순 정렬됨)
        mask : 보정 대상 행. 기본 = 유효 & 이력 있음 & GAP >= GAP_THRESHOLD & 적용 장비
        반환 : 상대 보정량 Δ (n,) — 대상 외 행과 작은 보정(임계 미만)은 0
               return_dn   : (Δ, 정규화 원시 보정량 dn)
               return_info : (Δ, dict(DN, BETA, BASE_U, TARGET) 전체 행 길이)
        """
        m = self.default_mask(df) if mask is None else np.asarray(mask, bool)
        n = len(df)
        delta, dn_all = np.zeros(n), np.zeros(n)
        beta_all, base_all = np.ones(n), np.zeros(n)
        if m.any():
            info = self._replay_core(df, m)
            delta[m] = self._finish(info['dn'], info['rtol'], info['beta'])
            dn_all[m], beta_all[m], base_all[m] = info['dn'], info['beta'], info['base_u']
        if return_info:
            return delta, dict(DN=dn_all, BETA=beta_all, BASE_U=base_all, TARGET=m)
        return (delta, dn_all) if return_dn else delta

    def signals_for(self, row, t, proc, rtol):
        """요청 1건의 정규화 신호 벡터와 상태 보유 여부"""
        vals = self.state.query(row, t)
        if self.is_v3:
            ex = self.v3.query(row, proc)
            x = assemble(vals[None, :], ex[None, :], np.array([rtol]))[0]
            has = bool(np.any(vals) or np.any(ex))
        else:
            x = np.nan_to_num(vals / rtol, nan=0.0, posinf=0.0, neginf=0.0)
            has = bool(np.any(vals))
        return x, has

    def delta_for(self, row, t, proc, rtol):
        """
        요청 1건. 반환: (dn 원시, Δ 상대보정량, clamped, has_state, β, small)
          small = True 이면 |β·보정량| < 임계값이라 Δ = 0 (APC 원값 사용)
        """
        x, has = self.signals_for(row, t, proc, rtol)
        dn = float(x @ self.weights)
        b = self.beta.query(row, float(day_index(np.array([t]))[0])) if self.use_beta else 1.0
        fu = float(b * self._base_u(np.array([dn]))[0])
        small = abs(fu) < self.apply_min_u
        d = 0.0 if small else fu * C.APPLY_SCALE * rtol
        return dn, d, abs(dn) > self.clamp, has, b, small

    def predict_row(self, row, t, gap_days):
        """신규 Lot 1건 보정. 반환: (보정노광량, Δ, 적용여부)"""
        proc = float(row['PROC_EXPOSE'])
        tol = float(row['MAIN_CONSTANT']) * float(row['CD_TARGET'])
        if tol <= 0 or proc <= 0:
            return proc, 0.0, False
        if not (np.isfinite(gap_days) and gap_days >= C.GAP_THRESHOLD) or not eq_target(row.get('PROC_EQ')):
            return proc, 0.0, False
        _, d, _, _, _, small = self.delta_for(row, t, proc, tol / proc)
        if small or d == 0:
            return proc, 0.0, False
        q = C.QUANT.get(row.get('EXPOSE_TYPE'), C.QUANT_DEFAULT)
        return round(proc * (1 + d) / q) * q, d, True

    # ---------- 일 배치 갱신 ----------
    def update_rows(self, df):
        """
        일 배치 1회분 상태 갱신 (시간순). df 필요 컬럼:
          축 컬럼, _T, RRES(APC 원값 기준), IDEAL(사용값 기준), APC_EXPOSE, CD_METHOD
        반환: EMA 갱신 건수
        """
        src = ~df['CD_METHOD'].astype(str).str.upper().isin([m.upper() for m in C.SRC_EXCLUDE_CD_METHOD])
        KA = {a: _join(df, cc).values for a, cc in self.state.axes.items()}
        T, R = df['_T'].values, df['RRES'].values
        ideal = df['IDEAL'].values if 'IDEAL' in df.columns else df['IDEAL_EXPOSE'].values
        apc = df['APC_EXPOSE'].values if 'APC_EXPOSE' in df.columns else df['PROC_EXPOSE'].values
        recs = df.to_dict('records') if (self.is_v3 or self.use_beta) else None
        bq = self._beta_batch(df, recs, src, T, R, apc) if self.use_beta else []
        n = 0
        for i in range(len(df)):
            ok = bool(src.iat[i]) and np.isfinite(R[i])
            if ok:
                self.state.update_keys({a: KA[a][i] for a in KA}, float(T[i]), float(R[i]))
                n += 1
            if self.is_v3:
                iv = float(ideal[i]) if np.isfinite(R[i]) else np.nan   # 무효 행은 이력에 넣지 않음
                self.v3.update(recs[i], float(T[i]), iv, float(apc[i]), ok)
        for i, day, base, u in bq:                     # β 는 하루치 조회가 끝난 뒤 갱신(일 배치)
            self.beta.update(recs[i], day, base, u)
        return n

    def _beta_batch(self, df, recs, src, T, R, apc):
        """
        β 갱신 재료. 이 배치의 상태 갱신 전(= 요청 시점과 같은 상태)으로 원 보정량을 다시 계산한다.
        대상: 적용 장비 & 이력 있는 KEY & FIX/SC_IN_AI 아님 & 유효
        반환: [(행번호, DAY, 원 보정량 u단위, APC 원값 기준 u)]
        """
        logic = df['CD_LOGIC'].astype(str).values if 'CD_LOGIC' in df.columns else None
        mc = pd.to_numeric(df['MAIN_CONSTANT'], errors='coerce').values
        tgt = pd.to_numeric(df['CD_TARGET'], errors='coerce').values
        tmask = eq_target(df['PROC_EQ'].astype(str))
        last = {}
        out = []
        for i in range(len(df)):
            key = '\x01'.join(str(recs[i].get(c)) for c in C.KEY_COLS)
            if key in last:
                lt = last[key]
            elif self.v3 is not None and key in self.v3.kidx:
                lt = float(self.v3.KL_T[self.v3.kidx[key]])
                lt = lt if np.isfinite(lt) else -1e18
            else:
                lt = None
            last[key] = float(T[i])
            if lt is None or (float(T[i]) - lt) < C.BETA_GAP_MIN:      # 이력 없음 / β 학습 GAP 하한 미만
                continue
            if not (tmask[i] and bool(src.iat[i]) and np.isfinite(R[i])):
                continue
            if logic is not None and logic[i] in C.BETA_EXCLUDE_CD_LOGIC:
                continue
            tol = mc[i] * tgt[i]
            if not (tol > 0 and apc[i] > 0):
                continue
            rtol = tol / apc[i]
            x, _ = self.signals_for(recs[i], float(T[i]), float(apc[i]), rtol)
            base = float(self._base_u(np.array([float(x @ self.weights)]))[0])
            out.append((i, float(day_index(np.array([T[i]]))[0]), base, float(R[i] / rtol)))
        return out

    # ---------- 영속화 ----------
    def save(self, dirpath):
        os.makedirs(dirpath, exist_ok=True)
        names = list(self.state.sig) + (extra_names() if self.is_v3 else [])
        json.dump(dict(version=self.version, mode=self.mode, clamp=self.clamp,
                       axes=self.axes, weights=list(self.weights),
                       cal_scale=list(self.cal_scale), sig=names,
                       gap_threshold=C.GAP_THRESHOLD,
                       level_groups=C.LEVEL_GROUPS if self.is_v3 else None,
                       level_hls=C.LEVEL_HLS if self.is_v3 else None,
                       l3_hls=C.L3_HLS if self.is_v3 else None,
                       apply_eq_prefix=C.APPLY_EQ_PREFIX, apply_ratio=C.APPLY_RATIO,
                       apply_min_u=self.apply_min_u, apply_scale_at_fit=C.APPLY_SCALE, fit_pass_band=getattr(C, 'FIT_PASS_BAND', 1.0),
                       state_until=getattr(self, 'state_until', None),
                       beta=self.beta.params() if self.use_beta else None),
                  open(f'{dirpath}/model.json', 'w'), indent=1, ensure_ascii=False)
        extra = self.v3.to_npz_dict() if self.is_v3 else {}
        if self.use_beta:
            extra.update(self.beta.to_npz_dict())
        self.state.save(f'{dirpath}/state.npz', extra=extra or None)
        print(f'[save] {dirpath} ({self.version}/{self.mode})')

    @classmethod
    def load(cls, dirpath):
        m = json.load(open(f'{dirpath}/model.json'))
        st = RuleState.load(f'{dirpath}/state.npz')
        ver = m.get('version', 'v2')
        v3 = None
        z = np.load(f'{dirpath}/state.npz', allow_pickle=True)
        beta = BetaState.from_npz(z) if m.get('beta') else None
        if m.get('apply_eq_prefix', C.APPLY_EQ_PREFIX) != C.APPLY_EQ_PREFIX:
            print(f"[load] ⚠ 학습 시 적용 장비 {m.get('apply_eq_prefix')} != 현재 config {C.APPLY_EQ_PREFIX}. "
                  '임계값·β 는 학습 시 장비 기준입니다.')
        if ver == 'v3':
            v3 = V3State.from_npz(z)
            if v3 is None:
                raise ValueError('v3 모델인데 state.npz 에 v3 상태가 없습니다. fit.py 로 다시 학습하십시오.')
            if m.get('level_groups') and m['level_groups'] != C.LEVEL_GROUPS:
                raise ValueError('config.LEVEL_GROUPS 가 학습 시점과 다릅니다.')
        eng = cls(axes=m['axes'], weights=m['weights'], cal_scale=m['cal_scale'], state=st,
                  v3=v3, version=ver, clamp=m.get('clamp'), mode=m.get('mode', 'rule'),
                  beta=beta, apply_min_u=m.get('apply_min_u', 0.0))
        eng.state_until = m.get('state_until')
        return eng
