"""Synthetic benchmarks for Paper II (MF-X-QDMA).

B1  binomial-measure pairs: exact joint-partition theory vs the average rule
B2  2D cascade pairs (exact vs average rule)
B3  bivariate fractional Gaussian noise (corpus, 75 pairs)
B4  two-component ARFIMA (corpus, d1 = d2)
C1  contaminated binomial pair: x only / both independent / both common
C2  contaminated BFBM: spikes and bad-tick pairs, x only / both; rho(n) bias
"""
import sys, os, json, time
import numpy as np
from .generators import binomial_pair, cascade_2d, bfbm, contaminate
from .methods_cross import (mfx1d, mfx2d, binomial_pair_exact, binomial_pair_average,
                           cascade2d_pair_tau_exact, cascade2d_pair_tau_average)

from . import paths
ROOT = str(paths.REPO)
OUT = f'{ROOT}/results/cross_bench.json'
Q = np.arange(-4, 4.5, 0.5)
SM = np.unique(np.logspace(np.log10(16), np.log10(2048), 14).astype(int))
SD = np.unique(np.logspace(np.log10(64), np.log10(8192), 14).astype(int))
LAMS = [(0.1, 0.1), (0.25, 0.25), (0.5, 0.5), (0.75, 0.75), (0.9, 0.9), (0.1, 0.9), (0.25, 0.75)]


def rmse(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    b = np.broadcast_to(b, a.shape)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2))) if m.sum() else np.nan


def all_methods(x, y, demean, lams=LAMS, sm=SM, sd=SD, thetas=True):
    out = {'MFXDFA': mfx1d(x, y, Q, sd, 'dfa', demean=demean)['dfa'],
           'MFXDMA_th0': mfx1d(x, y, Q, sm, 'dma', 0.0, demean=demean)['dma']}
    if thetas:
        out['MFXDMA_th0.5'] = mfx1d(x, y, Q, sm, 'dma', 0.5, demean=demean)['dma']
        out['MFXDMA_th1'] = mfx1d(x, y, Q, sm, 'dma', 1.0, demean=demean)['dma']
        out['MFXQDMA_th0.5_0.5,0.5'] = mfx1d(x, y, Q, sm, 'qdma', 0.5, [(0.5, 0.5)], demean=demean)[(0.5, 0.5)]
        out['MFXQDMA_th1_0.5,0.5'] = mfx1d(x, y, Q, sm, 'qdma', 1.0, [(0.5, 0.5)], demean=demean)[(0.5, 0.5)]
    for k, v in mfx1d(x, y, Q, sm, 'qdma', 0.0, lams, demean=demean).items():
        out[f'MFXQDMA_th0_{k[0]},{k[1]}'] = v
    return out


def B1():
    rows = []
    pairs = [(0.3, 0.3), (0.25, 0.35), (0.3, 0.4), (0.2, 0.3), (0.2, 0.45),
             (0.15, 0.4), (0.1, 0.3), (0.1, 0.45)]
    for px, py in pairs:
        x, y = binomial_pair(px, py, 16)
        ex, av = binomial_pair_exact(Q, px, py), binomial_pair_average(Q, px, py)
        for m, r in all_methods(x, y, demean=False).items():
            rows.append(dict(px=px, py=py, method=m, rmse_exact=rmse(r['hq'], ex),
                             rmse_avg=rmse(r['hq'], av), hq=r['hq'].tolist(),
                             r2=float(np.nanmean(r['r2'])),
                             gap=float(np.max(np.abs(ex - av)))))
        print('  B1', px, py, flush=True)
    return rows


def B2():
    P = {'A': (0.05, 0.15, 0.25, 0.55), 'B': (0.10, 0.20, 0.30, 0.40),
         'C': (0.02, 0.18, 0.30, 0.50)}
    s = np.unique(np.logspace(np.log10(6), np.log10(32), 8).astype(int))
    rows = []
    for a, b in (('A', 'B'), ('A', 'C'), ('B', 'C')):
        X, Y = cascade_2d(P[a], 8), cascade_2d(P[b], 8)
        te, ta = cascade2d_pair_tau_exact(Q, P[a], P[b]), cascade2d_pair_tau_average(Q, P[a], P[b])
        for lam in (None, 0.25, 0.5, 0.75):
            r = mfx2d(X, Y, Q, s, 0.0, lam)
            rows.append(dict(pair=a + b, lam=lam, rmse_exact=rmse(r['tau_q'], te),
                             rmse_avg=rmse(r['tau_q'], ta), tau=r['tau_q'].tolist(),
                             gap=float(np.max(np.abs(te - ta)))))
        print('  B2', a, b, flush=True)
    return rows


def _corpus(group):
    m = json.load(open(f'{ROOT}/MANIFEST.json'))['groups'][group]['items']
    return m


def B3():
    rows = []
    sm = np.unique(np.logspace(np.log10(16), np.log10(1024), 13).astype(int))
    sd = np.unique(np.logspace(np.log10(32), np.log10(4096), 13).astype(int))
    for it in _corpus('03_bfbm'):
        xy = np.load(f"{ROOT}/{it['file']}")
        res = all_methods(xy[0], xy[1], True, lams=[(0.1, 0.1), (0.25, 0.25), (0.5, 0.5)],
                          sm=sm, sd=sd, thetas=False)
        res['MFXDMA_th0.5'] = mfx1d(xy[0], xy[1], Q, sm, 'dma', 0.5)['dma']
        for m, r in res.items():
            rows.append(dict(Hxx=it['Hxx'], Hyy=it['Hyy'], rho=it['rho'], seed=it['seed'],
                             method=m, rmse=rmse(r['hq'], it['Hxy_theory']),
                             rho_n=r['rho_n'].tolist()))
    print('  B3 done', flush=True)
    return rows


def B4():
    rows = []
    sm = np.unique(np.logspace(np.log10(16), np.log10(1024), 13).astype(int))
    sd = np.unique(np.logspace(np.log10(32), np.log10(4096), 13).astype(int))
    for it in _corpus('04_arfima2'):
        if it['d1'] != it['d2']:
            continue
        H = 0.5 + it['d1']
        xy = np.load(f"{ROOT}/{it['file']}")
        res = all_methods(xy[0], xy[1], True, lams=[(0.1, 0.1), (0.25, 0.25), (0.5, 0.5)],
                          sm=sm, sd=sd, thetas=False)
        res['MFXDMA_th0.5'] = mfx1d(xy[0], xy[1], Q, sm, 'dma', 0.5)['dma']
        res['MFXDMA_th1'] = mfx1d(xy[0], xy[1], Q, sm, 'dma', 1.0)['dma']
        for m, r in res.items():
            rows.append(dict(d=it['d1'], W=it['W'], seed=it['seed'], method=m,
                             rmse=rmse(r['hq'], H)))
    print('  B4 done', flush=True)
    return rows


def _contam_pair(x, y, mode, frac, kind, mag, rep):
    if mode == 'x':
        return contaminate(x, frac, mag, kind, seed=900 + rep)[0], y
    if mode == 'both_indep':
        return (contaminate(x, frac, mag, kind, seed=900 + rep)[0],
                contaminate(y, frac, mag, kind, seed=1900 + rep)[0])
    # common positions (same seed -> same indices), independent signs
    xc, idx = contaminate(x, frac, mag, kind, seed=900 + rep)
    yc = y.copy()
    rng = np.random.RandomState(2900 + rep)
    yc[idx] = rng.choice([-1.0, 1.0], len(idx)) * mag * y.std()
    return xc, yc


def C1(n_rep=5):
    px, py = 0.3, 0.4
    x, y = binomial_pair(px, py, 16)
    ex = binomial_pair_exact(Q, px, py)
    lams = [(0.1, 0.1), (0.25, 0.25), (0.5, 0.5), (0.9, 0.9)]
    rows = []
    for mode in ('x', 'both_indep', 'both_common'):
        for frac in (0.0, 0.01, 0.02, 0.05, 0.10):
            for rep in range(1 if frac == 0 else n_rep):
                xc, yc = (x, y) if frac == 0 else _contam_pair(x, y, mode, frac, 'outlier', 20.0, rep)
                res = all_methods(xc, yc, False, lams=lams, thetas=False)
                for m, r in res.items():
                    rows.append(dict(mode=mode, frac=frac, rep=rep, method=m,
                                     rmse=rmse(r['hq'], ex)))
            print('  C1', mode, frac, flush=True)
    return rows


def _badticks(x, frac, kappa, rng):
    x = x.copy(); s = x.std()
    m = max(1, int(round(frac * len(x))))
    idx = rng.choice(np.arange(1, len(x) - 1), m, replace=False)
    sg = rng.choice([-1.0, 1.0], m)
    x[idx] += sg * kappa * s; x[idx + 1] -= sg * kappa * s
    return x


def _spikes(x, frac, kappa, rng):
    x = x.copy(); s = x.std()
    m = max(1, int(round(frac * len(x))))
    idx = rng.choice(len(x), m, replace=False)
    x[idx] += rng.choice([-1.0, 1.0], m) * kappa * s
    return x


def C2(n_rep=5):
    rows = []
    sm = np.unique(np.logspace(np.log10(16), np.log10(1024), 13).astype(int))
    sd = np.unique(np.logspace(np.log10(32), np.log10(4096), 13).astype(int))
    lams = [(0.1, 0.1), (0.25, 0.25), (0.5, 0.5), (0.9, 0.9)]
    for (Hx, Hy, rho) in ((0.6, 0.6, 0.6), (0.4, 0.8, 0.6)):
        for sd_ in range(2):
            x, y = bfbm(2 ** 15, Hx, Hy, rho, seed=3000 + sd_)
            Hxy = 0.5 * (Hx + Hy)
            base = all_methods(x, y, True, lams=lams, sm=sm, sd=sd, thetas=False)
            for m, r in base.items():
                rows.append(dict(Hx=Hx, Hy=Hy, rho=rho, real=sd_, kind='clean', mode='-',
                                 frac=0.0, rep=0, method=m, rmse=rmse(r['hq'], Hxy),
                                 rho_n=r['rho_n'].tolist(), drho=0.0, dev=0.0))
            for kind in ('spike', 'badtick'):
                for mode in ('x', 'both'):
                    for frac in (0.005, 0.01, 0.02, 0.05):
                        for rep in range(n_rep):
                            rng = np.random.RandomState(7000 + rep + 100 * sd_)
                            f = _spikes if kind == 'spike' else _badticks
                            xc = f(x, frac, 10.0, rng)
                            yc = f(y, frac, 10.0, rng) if mode == 'both' else y
                            res = all_methods(xc, yc, True, lams=lams, sm=sm, sd=sd, thetas=False)
                            for m, r in res.items():
                                rows.append(dict(Hx=Hx, Hy=Hy, rho=rho, real=sd_, kind=kind,
                                                 mode=mode, frac=frac, rep=rep, method=m,
                                                 rmse=rmse(r['hq'], Hxy),
                                                 dev=rmse(r['hq'], base[m]['hq']),
                                                 drho=float(np.nanmean(np.abs(
                                                     r['rho_n'] - base[m]['rho_n']))),
                                                 rho_n=r['rho_n'].tolist()))
                print('  C2', Hx, Hy, sd_, kind, flush=True)
    return rows



def C3(n_rep=3):
    """BFBM with spikes / bad ticks: coefficient suite (uncentered median,
    midhinge, centred lam=0.25, moving mean), q>0 cross exponents and rho(n)."""
    from .methods_cross import coef_suite, SUITE
    sm = np.unique(np.logspace(np.log10(10), np.log10(1000), 13).astype(int))
    qp = (1.0, 2.0, 3.0, 4.0)
    rows = []
    for (Hx, Hy, rho) in ((0.6, 0.6, 0.6), (0.4, 0.8, 0.6)):
        for sd_ in range(2):
            x, y = bfbm(2 ** 15, Hx, Hy, rho, seed=3000 + sd_)
            Hxy = 0.5 * (Hx + Hy)
            base = coef_suite(x, y, sm, qp)
            for k in SUITE:
                rows.append(dict(Hx=Hx, Hy=Hy, rho=rho, real=sd_, kind='clean', mode='-', frac=0.0,
                                 rep=0, method=k, rmse=rmse(base[k]['hq'], Hxy),
                                 rhohat=float(np.nanmean(base[k]['rho'])),
                                 rhohat1=float(np.nanmean(base[k]['rho1'])), dev=0.0, drho=0.0, drho1=0.0))
            for kind in ('spike', 'badtick'):
                for mode in ('x', 'both'):
                    for frac in (0.005, 0.01, 0.02, 0.05):
                        for rep in range(n_rep):
                            rng = np.random.RandomState(7000 + rep + 100 * sd_)
                            f = _spikes if kind == 'spike' else _badticks
                            xc = f(x, frac, 10.0, rng)
                            yc = f(y, frac, 10.0, rng) if mode == 'both' else y
                            r = coef_suite(xc, yc, sm, qp)
                            for k in SUITE:
                                rows.append(dict(Hx=Hx, Hy=Hy, rho=rho, real=sd_, kind=kind, mode=mode,
                                                 frac=frac, rep=rep, method=k,
                                                 rmse=rmse(r[k]['hq'], Hxy),
                                                 dev=rmse(r[k]['hq'], base[k]['hq']),
                                                 drho=float(np.nanmean(np.abs(r[k]['rho'] - base[k]['rho']))),
                                                 drho1=float(np.nanmean(np.abs(r[k]['rho1'] - base[k]['rho1'])))))
                print('  C3', Hx, Hy, sd_, kind, flush=True)
    return rows


if __name__ == '__main__':
    parts = sys.argv[1:] or ['B1', 'B2', 'B3', 'B4', 'C1', 'C2']
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    t0 = time.time()
    for p in parts:
        res[p] = globals()[p]()
        json.dump(res, open(OUT, 'w'))
        print(f'{p} saved {time.time() - t0:.0f}s', flush=True)
