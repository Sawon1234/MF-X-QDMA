"""MF-X-QDMA: multifractal quantile detrending moving-average cross-correlation
analysis (Paper II), together with the reference estimators MF-X-DFA [Zhou 2008]
and MF-X-DMA [Jiang & Zhou 2011], in 1D and 2D.

Features beyond methods.py:
  * different quantile levels for the two series (lam_x, lam_y);
  * two conventions for q-th order cross moments:
      'abs'    : {<|F2_nu|^{q/2}>}^{1/q}                [Zhou 2008; Jiang & Zhou 2011]
      'signed' : {<sign(F2_nu)|F2_nu|^{q/2}>}^{1/q}      [Oswiecimka et al. 2014]
  * scale-resolved detrended cross-correlation coefficients
      rho(n)   = <F2xy> / sqrt(<F2xx><F2yy>)            (DCCA/DMCA coefficient)
      rho_q(n) = S_q^xy / sqrt(S_q^xx S_q^yy)           [Kwapien et al. 2015]
"""
import numpy as np
from .methods import (_profile, _win_bounds, _moving_quantiles_multi, _fit, EPS)


# ----------------------------------------------------------------- residuals
def residuals_1d(y, n, theta, lams):
    """Residuals of a profile for the moving mean (key None) and moving
    lam-quantiles (keys lam).  Returns {key: eps}, aligned on a common index."""
    N = len(y)
    nb, nf = _win_bounds(n, theta)
    lo, hi = nb, N - nf
    if hi <= lo:
        return None
    out = {}
    win = np.lib.stride_tricks.sliding_window_view(y, n)
    out[None] = y[lo:hi] - win.mean(axis=1)[: hi - lo]
    lq = [l for l in lams if l is not None]
    if lq:
        got = _moving_quantiles_multi(y, n, theta, lq)
        qd, lo_i, hi_i = got
        for l in lq:
            out[l] = y[lo_i:hi_i] - qd[l]
    return out


def residuals_dfa(y, n, order=1):
    N = len(y); Ns = N // n
    if Ns < 4:
        return None
    seg = y[: Ns * n].reshape(Ns, n)
    t = np.arange(n); V = np.vander(t, order + 1)
    c, *_ = np.linalg.lstsq(V, seg.T, rcond=None)
    return (seg.T - V @ c).T                       # (Ns, n)


# ----------------------------------------------------------------- moments
def _abs_moments(F2, q_list):
    a = np.abs(np.asarray(F2, float))
    a = a[np.isfinite(a) & (a > EPS)]
    if len(a) < 4:
        return None
    ln = 0.5 * np.log(a)
    out = np.empty(len(q_list))
    for i, q in enumerate(q_list):
        if abs(q) < 1e-12:
            out[i] = np.exp(ln.mean())
        else:
            m = (q * ln).max()
            out[i] = np.exp((m + np.log(np.exp(q * ln - m).mean())) / q)
    return out


def _signed_moments(F2, q_list):
    """Oswiecimka et al. (2014): S_q = < sign(F2) |F2|^{q/2} >; F_q = S_q^{1/q}
    when S_q > 0, NaN otherwise.  q = 0 is undefined in this convention."""
    F2 = np.asarray(F2, float)
    F2 = F2[np.isfinite(F2) & (np.abs(F2) > EPS)]
    if len(F2) < 4:
        return None
    s, a = np.sign(F2), np.abs(F2)
    out = np.full(len(q_list), np.nan)
    for i, q in enumerate(q_list):
        if abs(q) < 1e-12:
            continue
        Sq = np.mean(s * a ** (q / 2.0))
        if Sq > 0:
            out[i] = Sq ** (1.0 / q)
    return out


# ----------------------------------------------------------------- 1D
def mfx1d(x, y, q_list, scales, method='qdma', theta=0.0, lams=((0.5, 0.5),),
          demean=True, moments='abs', order=1, coef=True, center=None):
    """method: 'dfa', 'dma', or 'qdma'.  For 'qdma', `lams` is a list of
    (lam_x, lam_y) pairs evaluated in one pass.  Returns {key: result}, where
    key is 'dfa', 'dma' or the (lam_x, lam_y) tuple.  Each result holds the
    _fit() output for F_xy plus, if coef=True, rho(n) and the auto F_q.

    center: subtract the segment mean of each residual before forming
    products (segment covariance).  Default: True for quantile residuals in
    the signal convention (demean=True), where lam != 1/2 leaves a common
    offset that would otherwise create spurious cross-correlation; False for
    measures, whose residuals are box masses (Paper I, Prop. 1), and for the
    published MF-X-DFA/MF-X-DMA definitions."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    N = min(len(x), len(y))
    Yx, Yy = _profile(x[:N], demean), _profile(y[:N], demean)
    scales = np.asarray(scales)
    keys = ['dfa'] if method == 'dfa' else (['dma'] if method == 'dma' else list(lams))
    mom = _abs_moments if moments == 'abs' else _signed_moments
    Fxy = {k: np.full((len(q_list), len(scales)), np.nan) for k in keys}
    Fxx = {k: np.full((len(q_list), len(scales)), np.nan) for k in keys}
    Fyy = {k: np.full((len(q_list), len(scales)), np.nan) for k in keys}
    rho = {k: np.full(len(scales), np.nan) for k in keys}
    ulx = sorted({l[0] for l in lams}) if method == 'qdma' else []
    uly = sorted({l[1] for l in lams}) if method == 'qdma' else []
    for si, n in enumerate(scales):
        if method == 'dfa':
            rx, ry = residuals_dfa(Yx, n, order), residuals_dfa(Yy, n, order)
            if rx is None:
                continue
            seg = {'dfa': (rx, ry)}
        else:
            ex = residuals_1d(Yx, n, theta, ulx)
            ey = residuals_1d(Yy, n, theta, uly)
            if ex is None or ey is None:
                continue
            seg = {}
            for k in keys:
                a = ex[None] if k == 'dma' else ex[k[0]]
                b = ey[None] if k == 'dma' else ey[k[1]]
                L = min(len(a), len(b)); Nn = L // n
                if Nn < 4:
                    continue
                seg[k] = (a[: Nn * n].reshape(Nn, n), b[: Nn * n].reshape(Nn, n))
        for k, (sx, sy) in seg.items():
            c = (demean and method == 'qdma') if center is None else center
            if c:
                sx = sx - sx.mean(axis=1, keepdims=True)
                sy = sy - sy.mean(axis=1, keepdims=True)
            f2xy = (sx * sy).mean(axis=1)
            f2xx = (sx * sx).mean(axis=1)
            f2yy = (sy * sy).mean(axis=1)
            m = mom(f2xy, q_list)
            if m is not None:
                Fxy[k][:, si] = m
            if coef:
                mx, my = _abs_moments(f2xx, q_list), _abs_moments(f2yy, q_list)
                if mx is not None and my is not None:
                    Fxx[k][:, si], Fyy[k][:, si] = mx, my
                den = np.sqrt(f2xx.mean() * f2yy.mean())
                rho[k][si] = f2xy.mean() / den if den > 0 else np.nan
    out = {}
    for k in keys:
        r = _fit(Fxy[k], scales, q_list)
        if coef:
            r['rho_n'] = rho[k]
            r['hxx'] = _fit(Fxx[k], scales, q_list)['hq']
            r['hyy'] = _fit(Fyy[k], scales, q_list)['hq']
        out[k] = r
    return out


# ----------------------------------------------------------------- 2D
def residual_2d(Z, n, theta=0.0, lam=None):
    """Residual field eps(i1,i2) of 2D-MFDMA (lam=None) or 2D-MFQDMA,
    identical to methods.mfdma_2d (Paper I, Sec. II C)."""
    N1, N2 = Z.shape
    nb, nf = _win_bounds(n, theta)
    R1, R2 = N1 - n + 1, N2 - n + 1
    if N1 - nf <= nb or N2 - nf <= nb or R1 < 2 * n or R2 < 2 * n:
        return None
    shift = int(np.floor((n - 1) * theta))
    P = np.zeros((N1 + 1, N2 + 1))
    P[1:, 1:] = np.cumsum(np.cumsum(Z, axis=0), axis=1)
    Qbox = P[n:, n:] - P[:-n, n:] - P[n:, :-n] + P[:-n, :-n]
    if lam is None:
        ramp = np.arange(n, 0, -1, dtype=float)
        tmp = np.apply_along_axis(lambda v: np.convolve(v, ramp[::-1], mode='valid'), 0, Z)
        trend = np.apply_along_axis(lambda v: np.convolve(v, ramp[::-1], mode='valid'), 1, tmp) / (n * n)
    else:
        trend = np.empty((R1, R2))
        k = int(np.floor(lam * (n * n - 1))); k2 = min(k + 1, n * n - 1)
        fr = lam * (n * n - 1) - k
        budget = max(1, int(2.0e7 // max(1, R2 * n * n)))
        for r0 in range(0, R1, budget):
            r1 = min(R1, r0 + budget)
            W = np.lib.stride_tricks.sliding_window_view(P[r0:r1 + n, :], (n + 1, n + 1))
            Zp = W[:, :, 1:, 1:] - W[:, :, :1, 1:] - W[:, :, 1:, :1] + W[:, :, :1, :1]
            part = np.partition(Zp.reshape(r1 - r0, R2, -1), [k, k2], axis=2)
            trend[r0:r1] = part[:, :, k] * (1.0 - fr) + part[:, :, k2] * fr
    if shift > 0:
        if trend.shape[0] <= shift or trend.shape[1] <= shift:
            return None
        return Qbox[:-shift, :-shift] - trend[shift:, shift:]
    return Qbox - trend


def mfx2d(X, Y, q_list, scales, theta=0.0, lam=None, demean=False, moments='abs',
          center=None):
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    if demean:
        X = X - X.mean(); Y = Y - Y.mean()
    scales = np.asarray(scales)
    mom = _abs_moments if moments == 'abs' else _signed_moments
    F = np.full((len(q_list), len(scales)), np.nan)
    for si, n in enumerate(scales):
        ex, ey = residual_2d(X, n, theta, lam), residual_2d(Y, n, theta, lam)
        if ex is None or ey is None:
            continue
        M1, M2 = ex.shape; B1, B2 = M1 // n, M2 // n
        if B1 < 2 or B2 < 2:
            continue
        bx = ex[:B1 * n, :B2 * n].reshape(B1, n, B2, n)
        by = ey[:B1 * n, :B2 * n].reshape(B1, n, B2, n)
        c = (demean and lam is not None) if center is None else center
        if c:
            bx = bx - bx.mean(axis=(1, 3), keepdims=True)
            by = by - by.mean(axis=(1, 3), keepdims=True)
        f2 = (bx * by).mean(axis=(1, 3)).ravel()
        m = mom(f2, q_list)
        if m is not None:
            F[:, si] = m
    return _fit(F, scales, q_list, Df=2.0)


# ----------------------------------------------------------------- theory
def binomial_pair_exact(q, px, py):
    """Exact cross exponent of two binomial measures with a common split
    pattern, from the joint partition function sum_boxes (mu_x mu_y)^{q/2}:
        tau_xy(q) = -log2[ (px py)^{q/2} + ((1-px)(1-py))^{q/2} ],
        h_xy(q)   = [tau_xy(q) + 1]/q,  h_xy(0) = -ln(px py (1-px)(1-py))/(4 ln 2)."""
    q = np.asarray(q, float); A = px * py; B = (1 - px) * (1 - py)
    out = np.empty_like(q); s = np.abs(q) < 1e-9
    out[~s] = (1.0 - np.log2(A ** (q[~s] / 2) + B ** (q[~s] / 2))) / q[~s]
    out[s] = -(np.log(A) + np.log(B)) / (4 * np.log(2))
    return out


def binomial_pair_average(q, px, py):
    """The relation h_xy = (h_xx + h_yy)/2 used in Refs. [Zhou 2008, Jiang & Zhou 2011]."""
    from .generators import binomial_H_theory
    return 0.5 * (binomial_H_theory(q, px) + binomial_H_theory(q, py))


def cascade2d_pair_tau_exact(q, px, py):
    q = np.asarray(q, float)[..., None]
    a = np.asarray(px, float) * np.asarray(py, float)
    return -np.log2((a ** (q / 2)).sum(axis=-1))


def cascade2d_pair_tau_average(q, px, py):
    from .theory_2d import cascade2d_tau
    return 0.5 * (cascade2d_tau(q, px) + cascade2d_tau(q, py))


# ----------------------------------------------------------------- coefficient suite
SUITE = ('dma', 'q50', 'mid25', 'q25c')


def _suite_residuals(Y, n, theta=0.0):
    r = residuals_1d(Y, n, theta, [0.25, 0.5, 0.75])
    if r is None:
        return None
    return {'dma': r[None], 'q50': r[0.5], 'mid25': 0.5 * (r[0.25] + r[0.75]),
            'q25c': r[0.25]}


def coef_suite(x, y, scales, q_pos=(1.0, 2.0, 3.0, 4.0), theta=0.0):
    """Detrended cross-correlation coefficients and positive-q cross exponents
    for four trend functionals:
      dma   : moving average (DMCA / MF-X-DMA)
      q50   : moving median, uncentered (no offset for lam = 1/2)
      mid25 : midhinge (Q_0.25 + Q_0.75)/2, uncentered (offsets cancel, Prop. 2 of Paper I)
      q25c  : lam = 0.25 with segment centering (Prop. 3)
    Returns per functional: rho(n) [q=2 mean coefficient], rho1(n) [q=1
    coefficient of Kwapien et al.], and h_xy(q) for q in q_pos."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    N = min(len(x), len(y))
    Yx, Yy = _profile(x[:N], True), _profile(y[:N], True)
    scales = np.asarray(scales); qp = np.asarray(q_pos, float)
    rho = {k: np.full(len(scales), np.nan) for k in SUITE}
    rho1 = {k: np.full(len(scales), np.nan) for k in SUITE}
    F = {k: np.full((len(qp), len(scales)), np.nan) for k in SUITE}
    for si, n in enumerate(scales):
        rx, ry = _suite_residuals(Yx, n, theta), _suite_residuals(Yy, n, theta)
        if rx is None or ry is None:
            continue
        for k in SUITE:
            a, b = rx[k], ry[k]
            L = min(len(a), len(b)); Nn = L // n
            if Nn < 4:
                continue
            A = a[:Nn * n].reshape(Nn, n); B = b[:Nn * n].reshape(Nn, n)
            if k == 'q25c':
                A = A - A.mean(1, keepdims=True); B = B - B.mean(1, keepdims=True)
            fxy, fxx, fyy = (A * B).mean(1), (A * A).mean(1), (B * B).mean(1)
            rho[k][si] = fxy.mean() / np.sqrt(fxx.mean() * fyy.mean())
            S = lambda f: np.mean(np.sign(f) * np.sqrt(np.abs(f)))
            rho1[k][si] = S(fxy) / np.sqrt(S(fxx) * S(fyy))
            m = _abs_moments(fxy, qp)
            if m is not None:
                F[k][:, si] = m
    out = {}
    for k in SUITE:
        fit = _fit(F[k], scales, qp)
        out[k] = dict(rho=rho[k], rho1=rho1[k], hq=fit['hq'], r2=fit['r2'])
    return out
