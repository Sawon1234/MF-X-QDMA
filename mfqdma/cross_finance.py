"""Financial cross-correlation study for Paper II (MF-X-QDMA).

Pairs of daily increments from qrmdata (stock indices, FX, commodities, VIX,
US zero-coupon yields, Bitcoin) and 29 DJIA stock-index pairs.

AB  cross spectra h_xy(q) vs (h_xx+h_yy)/2, rho(n), surrogates
C   influence of common extreme days (leave-k-out)
D   injected data errors in one or both series
E   29 DJIA stock / DJIA index pairs
F   rolling detrended cross-correlation rho(n=20) with and without bad ticks
"""
import sys, os, json, time
import numpy as np
from .rdata_reader import read_rda, to_series
from .finance import load_prices, LABEL, RAW
from .methods_cross import mfx1d

from . import paths
OUT = str(paths.RESULTS / 'cross_finance.json')
Q = np.arange(-4, 4.5, 0.5)
LAMS = [(0.1, 0.1), (0.25, 0.25), (0.5, 0.5)]
LABEL = dict(LABEL, VIX='VIX', UST2='UST 2y', UST10='UST 10y')
PAIRS = [('SP500', 'DJ'), ('DJ', 'NASDAQ'), ('SP500', 'FTSE'), ('DAX', 'CAC'),
         ('FTSE', 'DAX'), ('NIKKEI', 'HSI'), ('HSI', 'SSEC'), ('SSEC', 'CSI'),
         ('EUR_USD', 'GBP_USD'), ('EUR_USD', 'JPY_USD'), ('GOLD', 'EUR_USD'),
         ('GOLD', 'OIL_Brent'), ('SP500', 'VIX'), ('UST2', 'UST10'),
         ('SP500', 'UST10'), ('BTC', 'SP500')]


# ------------------------------------------------------------------ data
def increments(name):
    paths.ensure_qrmdata()
    """dates, increments: log returns for prices, log changes for VIX,
    first differences (percentage points) for yields."""
    if name in ('UST2', 'UST10'):
        d, x, c = to_series(read_rda(f'{RAW}/ZCB_USD.rda')['ZCB_USD'])
        v = x[:, c.index('2y' if name == 'UST2' else '10y')]
        ok = np.isfinite(v)
        d, v = d[ok], v[ok]
        return d[1:], np.diff(v)
    if name == 'VIX':
        d, x, c = to_series(read_rda(f'{RAW}/VIX.rda')['VIX'])
        p = x[:, 0]; ok = np.isfinite(p) & (p > 0)
        d, p = d[ok], p[ok]
        return d[1:], np.diff(np.log(p))
    d, p = load_prices(name)
    return d[1:], np.diff(np.log(p))


def aligned(a, b):
    paths.ensure_qrmdata()
    """Common dates of two price series; increments recomputed on the
    intersection so that both refer to the same holding periods."""
    def lev(name):
        if name in ('UST2', 'UST10'):
            d, x, c = to_series(read_rda(f'{RAW}/ZCB_USD.rda')['ZCB_USD'])
            v = x[:, c.index('2y' if name == 'UST2' else '10y')]
            return d, v, 'diff'
        if name == 'VIX':
            d, x, c = to_series(read_rda(f'{RAW}/VIX.rda')['VIX'])
            return d, x[:, 0], 'log'
        d, p = load_prices(name)
        return d, p, 'log'
    da, va, ta = lev(a); db, vb, tb = lev(b)
    common, ia, ib = np.intersect1d(da, db, return_indices=True)
    va, vb = va[ia], vb[ib]
    ok = np.isfinite(va) & np.isfinite(vb) & ((va > 0) | (ta == 'diff')) & ((vb > 0) | (tb == 'diff'))
    common, va, vb = common[ok], va[ok], vb[ok]
    fa = np.diff(np.log(va)) if ta == 'log' else np.diff(va)
    fb = np.diff(np.log(vb)) if tb == 'log' else np.diff(vb)
    return common[1:], fa, fb


def dj_pairs(start='2000-01-01', end='2015-12-31'):
    paths.ensure_qrmdata()
    d, x, c = to_series(read_rda(f'{RAW}/DJ_const.rda')['DJ_const'])
    di, xi, _ = to_series(read_rda(f'{RAW}/DJ.rda')['DJ'])
    common, ia, ib = np.intersect1d(d, di, return_indices=True)
    m = (common >= np.datetime64(start)) & (common <= np.datetime64(end))
    X = x[ia][m]; idx = xi[ib, 0][m]
    out = {}
    for j, t in enumerate(c):
        p = X[:, j]
        if np.isfinite(p).all() and (p > 0).all():
            out[t] = (np.diff(np.log(p)), np.diff(np.log(idx)))
    return out


# ------------------------------------------------------------------ estimation
def scales(N):
    hi = min(500, N // 10)
    sm = np.unique(np.logspace(1, np.log10(hi), 14).astype(int))
    sd = np.unique(np.logspace(np.log10(20), np.log10(2 * hi), 14).astype(int))
    return sm, sd


def width(a):
    a = np.asarray(a)[np.isfinite(a)]
    return float(a.max() - a.min()) if len(a) > 2 else np.nan


def analyse(x, y, full=False):
    sm, sd = scales(len(x))
    res = {'MFXDFA': mfx1d(x, y, Q, sd, 'dfa')['dfa'],
           'MFXDMA': mfx1d(x, y, Q, sm, 'dma', 0.0)['dma']}
    for k, v in mfx1d(x, y, Q, sm, 'qdma', 0.0, LAMS).items():
        res[f'MFXQDMA_{k[0]}'] = v
    out = {}
    for m, r in res.items():
        o = dict(hq=r['hq'].tolist(), hxx=r['hxx'].tolist(), hyy=r['hyy'].tolist(),
                 dalpha=width(r['alpha']), rho_n=r['rho_n'].tolist(),
                 rho=float(np.nanmean(r['rho_n'])), r2=float(np.nanmean(r['r2'])),
                 h2=float(r['hq'][np.argmin(np.abs(Q - 2))]))
        if full:
            o.update(alpha=r['alpha'].tolist(), f_alpha=r['f_alpha'].tolist(),
                     Fq=r['Fq'].tolist(), scales=r['scales'].tolist())
        out[m] = o
    return out


def dev(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2))) if m.sum() else np.nan


def compare(a, base):
    return {m: dict(dev_h=dev(a[m]['hq'], base[m]['hq']),
                    d_rho=float(np.nanmean(np.abs(np.asarray(a[m]['rho_n']) - np.asarray(base[m]['rho_n'])))),
                    d_dalpha=a[m]['dalpha'] - base[m]['dalpha']) for m in a}


# ------------------------------------------------------------------ errors
def inject(x, kind, level, rng, kappa=10.0):
    x = x.copy(); s = x.std()
    if kind == 'noise':
        return x + np.diff(rng.randn(len(x) + 1) * level * s)
    m = max(1, int(round(level * len(x))))
    idx = rng.choice(np.arange(1, len(x) - 1), m, replace=False)
    sg = rng.choice([-1.0, 1.0], m)
    x[idx] += sg * kappa * s
    if kind == 'badtick':
        x[idx + 1] -= sg * kappa * s
    return x


SCEN = [('badtick', 'x'), ('badtick', 'both'), ('outlier', 'x'), ('noise', 'both')]
LEVELS = {'badtick': (0.001, 0.0025, 0.005, 0.01), 'outlier': (0.001, 0.0025, 0.005, 0.01),
          'noise': (0.1, 0.25, 0.5, 1.0)}


def error_study(x, y, base, n_rep, seed, levels=None):
    rows = []
    rng = np.random.RandomState(seed)
    for kind, mode in SCEN:
        for lev in (levels or {}).get(kind, LEVELS[kind]):
            for rep in range(n_rep):
                xc = inject(x, kind, lev, rng)
                yc = inject(y, kind, lev, rng) if mode == 'both' else y
                for m, v in compare(analyse(xc, yc), base).items():
                    rows.append(dict(kind=kind, mode=mode, level=lev, rep=rep, method=m, **v))
    return rows


# ------------------------------------------------------------------ parts
def part_AB():
    out = {}
    for a, b in PAIRS:
        d, x, y = aligned(a, b)
        rng = np.random.RandomState(11)
        orig = analyse(x, y, full=True)
        absr = analyse(np.abs(x - x.mean()), np.abs(y - y.mean()))
        js, ind, ph = [], [], []
        for _ in range(5):
            p = rng.permutation(len(x)); js.append(analyse(x[p], y[p]))
            ind.append(analyse(x[rng.permutation(len(x))], y[rng.permutation(len(y))]))
            F1, F2 = np.fft.rfft(x - x.mean()), np.fft.rfft(y - y.mean())
            th = rng.uniform(0, 2 * np.pi, len(F1)); th[0] = 0
            if len(x) % 2 == 0:
                th[-1] = 0
            ph.append(analyse(np.fft.irfft(F1 * np.exp(1j * th), n=len(x)),
                              np.fft.irfft(F2 * np.exp(1j * th), n=len(y))))
        surr = {m: dict(js_dalpha=float(np.mean([s[m]['dalpha'] for s in js])),
                        ph_dalpha=float(np.mean([s[m]['dalpha'] for s in ph])),
                        ind_rho=float(np.mean([abs(s[m]['rho']) for s in ind])),
                        ind_rho_max=float(np.max([np.nanmax(np.abs(s[m]['rho_n'])) for s in ind])),
                        js_rho=float(np.mean([s[m]['rho'] for s in js])))
                for m in orig}
        out[f'{a}|{b}'] = dict(a=a, b=b, N=len(x), start=str(d[0]), end=str(d[-1]),
                               pearson=float(np.corrcoef(x, y)[0, 1]),
                               orig=orig, abs=absr, surr=surr)
        print(f'  AB {a}|{b} N={len(x)} r={np.corrcoef(x, y)[0, 1]:.2f}', flush=True)
    return out


def part_C():
    out = {}
    for a, b in PAIRS:
        d, x, y = aligned(a, b)
        base = analyse(x, y)
        zx = np.abs(x - np.median(x)) / (1.4826 * np.median(np.abs(x - np.median(x))))
        zy = np.abs(y - np.median(y)) / (1.4826 * np.median(np.abs(y - np.median(y))))
        order = np.argsort(-np.maximum(zx, zy))
        rec = {}
        for k in (1, 5, 10, 20):
            keep = np.ones(len(x), bool); keep[order[:k]] = False
            rec[k] = compare(analyse(x[keep], y[keep]), base)
        out[f'{a}|{b}'] = dict(top=[str(d[i]) for i in order[:3]], rec=rec)
        print(f'  C {a}|{b}', flush=True)
    return out


def part_D(n_rep=3):
    out = {}
    for i, (a, b) in enumerate(PAIRS):
        d, x, y = aligned(a, b)
        out[f'{a}|{b}'] = error_study(x, y, analyse(x, y), n_rep, 300 + i)
        print(f'  D {a}|{b}', flush=True)
    return out


def part_E(n_rep=3):
    out = {}
    lv = {'badtick': (0.005,), 'outlier': (0.005,), 'noise': (0.5,)}
    for i, (t, (x, y)) in enumerate(dj_pairs().items()):
        out[t] = dict(N=len(x), pearson=float(np.corrcoef(x, y)[0, 1]),
                      rows=error_study(x, y, analyse(x, y), n_rep, 800 + i, lv))
        print(f'  E {t}', flush=True)
    return out


def part_F(win=1000, step=60, nref=20):
    out = {}
    for a, b in (('SP500', 'UST10'), ('DAX', 'CAC')):
        d, x, y = aligned(a, b)
        rng = np.random.RandomState(5)
        xc = inject(x, 'badtick', 0.005, rng)          # errors in ONE series
        sm = np.unique(np.logspace(1, np.log10(100), 8).astype(int))
        j = int(np.argmin(np.abs(sm - nref)))
        rec = dict(t=[], clean={}, dirty={}, n=int(sm[j]))
        for s in range(0, len(x) - win + 1, step):
            rec['t'].append(str(d[s + win - 1]))
            for tag, xx in (('clean', x), ('dirty', xc)):
                r = {'MFXDMA': mfx1d(xx[s:s + win], y[s:s + win], Q[[12]], sm, 'dma')['dma']}
                for k, v in mfx1d(xx[s:s + win], y[s:s + win], Q[[12]], sm, 'qdma', 0.0,
                                  [(0.25, 0.25), (0.5, 0.5)]).items():
                    r[f'MFXQDMA_{k[0]}'] = v
                for m, v in r.items():
                    rec[tag].setdefault(m, []).append(float(v['rho_n'][j]))
            # Pearson of raw returns for reference
            rec.setdefault('pearson_clean', []).append(float(np.corrcoef(x[s:s + win], y[s:s + win])[0, 1]))
            rec.setdefault('pearson_dirty', []).append(float(np.corrcoef(xc[s:s + win], y[s:s + win])[0, 1]))
        out[f'{a}|{b}'] = rec
        print(f'  F {a}|{b}', flush=True)
    return out


# ================================================================== part G/H
from .methods_cross import coef_suite, SUITE


def suite(x, y):
    sm, _ = scales(len(x))
    s = coef_suite(x, y, sm)
    out = {k: dict(rho=v['rho'].tolist(), rho1=v['rho1'].tolist(), hq=v['hq'].tolist())
           for k, v in s.items()}
    out['pearson'] = float(np.corrcoef(x, y)[0, 1])
    return out


def suite_dev(a, b):
    o = {'pearson': abs(a['pearson'] - b['pearson'])}
    for k in SUITE:
        o[f'{k}_rho'] = float(np.nanmean(np.abs(np.array(a[k]['rho']) - np.array(b[k]['rho']))))
        o[f'{k}_rho1'] = float(np.nanmean(np.abs(np.array(a[k]['rho1']) - np.array(b[k]['rho1']))))
        o[f'{k}_h'] = dev(a[k]['hq'], b[k]['hq'])
    return o


GSCEN = [('badtick', 'x', 10.0), ('badtick', 'x', 50.0), ('badtick', 'both', 10.0),
         ('badtick', 'both', 50.0), ('outlier', 'x', 10.0), ('noise', 'both', None)]


def g_study(x, y, n_rep, seed, level=0.005, eta=0.5):
    base = suite(x, y)
    rows = []
    rng = np.random.RandomState(seed)
    for kind, mode, kap in GSCEN:
        for rep in range(n_rep):
            lev = eta if kind == 'noise' else level
            xc = inject(x, kind, lev, rng, kappa=kap or 10.0)
            yc = inject(y, kind, lev, rng, kappa=kap or 10.0) if mode == 'both' else y
            rows.append(dict(kind=kind, mode=mode, kappa=kap, rep=rep, **suite_dev(suite(xc, yc), base)))
    return base, rows


def part_G(n_rep=3):
    out = {}
    for i, (a, b) in enumerate(PAIRS):
        d, x, y = aligned(a, b)
        base, rows = g_study(x, y, n_rep, 600 + i)
        out[f'{a}|{b}'] = dict(base=base, rows=rows)
        print(f'  G {a}|{b}', flush=True)
    for i, (t, (x, y)) in enumerate(dj_pairs().items()):
        base, rows = g_study(x, y, n_rep, 900 + i)
        out[f'DJ:{t}'] = dict(base=base, rows=rows)
        print(f'  G DJ:{t}', flush=True)
    return out


def part_H(win=1000, step=60):
    out = {}
    for a, b in (('SP500', 'UST10'), ('DAX', 'CAC')):
        d, x, y = aligned(a, b)
        rng = np.random.RandomState(5)
        xc = inject(x, 'badtick', 0.005, rng, kappa=50.0)
        sm = np.unique(np.logspace(1, np.log10(100), 8).astype(int))
        j = int(np.argmin(np.abs(sm - 20)))
        rec = dict(t=[], n=int(sm[j]), clean={}, dirty={})
        for s in range(0, len(x) - win + 1, step):
            rec['t'].append(str(d[s + win - 1]))
            for tag, xx in (('clean', x), ('dirty', xc)):
                cs = coef_suite(xx[s:s + win], y[s:s + win], sm)
                vals = {'pearson': float(np.corrcoef(xx[s:s + win], y[s:s + win])[0, 1])}
                for k in SUITE:
                    vals[f'{k}_rho'] = float(cs[k]['rho'][j]); vals[f'{k}_rho1'] = float(cs[k]['rho1'][j])
                for kk, vv in vals.items():
                    rec[tag].setdefault(kk, []).append(vv)
        out[f'{a}|{b}'] = rec
        print(f'  H {a}|{b}', flush=True)
    return out


if __name__ == '__main__':
    which = sys.argv[1:] or ['AB', 'C', 'F', 'D', 'E']
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    t0 = time.time()
    for w in which:
        res[w] = globals()['part_' + w]()
        json.dump(res, open(OUT, 'w'))
        print(f'{w} saved {time.time() - t0:.0f}s', flush=True)


