"""Experiments of Paper II. Each function recomputes one file in results/.

The notebook MFQDMA_PaperII.ipynb calls these functions when the corresponding
RECOMPUTE flag is set. Run times are rough estimates for a free Colab runtime. Seeds are fixed, so the
stored results are reproduced.
"""
import json
import os
import time

import numpy as np

from . import paths

RES = paths.RESULTS


def run_pair_corpus():
    """BFGN and two-component ARFIMA pairs of Table IV / Fig. 5 (same seeds as the
    paper). Writes datasets/03_bfbm, datasets/04_arfima2 and MANIFEST.json (~2 min)."""
    from .generators import bfbm, bfbm_hxy_theory, arfima_two_component
    root = paths.REPO
    man = {'groups': {}}

    def save(group, name, arr, meta):
        d = root / 'datasets' / group
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / f'{name}.npy', arr)
        meta = dict(meta, file=f'datasets/{group}/{name}.npy', shape=list(arr.shape))
        man['groups'][group]['items'].append(meta)

    man['groups']['03_bfbm'] = {'source': 'Jiang & Zhou, PRE 84, 016106, Sec. III A',
                                'analytic': 'H_xy = (H_xx+H_yy)/2', 'items': []}
    for (hx, hy) in [(0.1, 0.5), (0.3, 0.7), (0.8, 0.8), (0.2, 0.2), (0.4, 0.9)]:
        for rho in (0.1, 0.3, 0.5, 0.7, 0.9):
            for sd in range(3):
                x, y = bfbm(2 ** 15, hx, hy, rho, seed=1000 + sd)
                save('03_bfbm', f'bfbm_H{int(hx*10)}{int(hy*10)}_r{int(rho*10)}_s{sd}', np.vstack([x, y]),
                     {'Hxx': hx, 'Hyy': hy, 'rho': rho, 'seed': sd, 'convention': 'signal', 'pair': True,
                      'Hxy_theory': bfbm_hxy_theory(hx, hy)})
    man['groups']['04_arfima2'] = {'source': 'Jiang & Zhou, PRE 84, 016106, Sec. III B',
                                   'analytic': 'H_xx=H_yy=0.5+d when d1=d2=d', 'items': []}
    for d1, d2 in [(0.1, 0.4), (0.4, 0.4), (0.2, 0.2), (0.3, 0.3), (0.1, 0.1)]:
        for W in (0.5, 0.75, 1.0):
            for sd in range(2):
                x, y = arfima_two_component(2 ** 15, d1, d2, W, seed=2000 + sd)
                save('04_arfima2', f'arfima_d{int(d1*10)}{int(d2*10)}_W{int(W*100)}_s{sd}', np.vstack([x, y]),
                     {'d1': d1, 'd2': d2, 'W': W, 'seed': sd, 'convention': 'signal', 'pair': True,
                      'H_theory': (0.5 + d1) if d1 == d2 else None})
    for d1, d2 in [(0.1, 0.4), (0.3, 0.3)]:
        for sd in range(2):
            x, y = arfima_two_component(2 ** 15, d1, d2, 1.0, seed=2100 + sd, shared_noise=True)
            save('04_arfima2', f'arfima_shared_d{int(d1*10)}{int(d2*10)}_s{sd}', np.vstack([x, y]),
                 {'d1': d1, 'd2': d2, 'W': 1.0, 'shared_noise': True, 'seed': sd, 'convention': 'signal',
                  'pair': True, 'H_theory': (0.5 + d1) if d1 == d2 else None})
    json.dump(man, open(root / 'MANIFEST.json', 'w'), indent=1)
    print('wrote', sum(len(g['items']) for g in man['groups'].values()), 'pairs', flush=True)


def run_cross_bench(parts=('B1', 'B2', 'B3', 'B4', 'C1', 'C2', 'C3')):
    """Synthetic benchmarks: binomial pairs (B1), 2D cascade pairs (B2), BFGN (B3)
    and ARFIMA (B4) pairs, contaminated binomial pair (C1), contaminated BFGN
    pairs (C2, C3). Writes results/cross_bench.json (~1.5 h in total)."""
    from . import cross_bench as CB
    if any(p in parts for p in ('B3', 'B4')) and not (paths.REPO / 'MANIFEST.json').exists():
        run_pair_corpus()
    out = RES / 'cross_bench.json'
    res = json.load(open(out)) if out.exists() else {}
    t0 = time.time()
    for p in parts:
        res[p] = getattr(CB, p)()
        json.dump(res, open(out, 'w'))
        print(f'{p} saved {time.time() - t0:.0f}s', flush=True)


def run_cross_offset():
    """Offset-induced spurious cross-correlation (Proposition 4): rho(n) of BFGN
    pairs with known rho, quantile residuals with and without centering.
    Writes results/cross_offset.json (Table I, Fig. 2; ~3 min)."""
    from .generators import bfbm
    from .methods_cross import mfx1d
    Q = np.array([2.0])
    SM = np.unique(np.logspace(np.log10(10), np.log10(500), 10).astype(int))
    L = [(0.1, 0.1), (0.25, 0.25), (0.5, 0.5), (0.75, 0.75), (0.9, 0.9), (0.1, 0.9)]
    rows = []
    for H in (0.4, 0.7):
        for rho in (0.0, 0.3, 0.6, 0.9):
            for sd in range(6):
                x, y = bfbm(2 ** 14, H, H, rho, seed=4000 + sd)
                d = mfx1d(x, y, Q, SM, 'dma')['dma']
                rows.append(dict(H=H, rho=rho, seed=sd, method='MFXDMA', center=False,
                                 rho_n=d['rho_n'].tolist()))
                for cen in (True, False):
                    for k, v in mfx1d(x, y, Q, SM, 'qdma', 0.0, L, center=cen).items():
                        rows.append(dict(H=H, rho=rho, seed=sd, method=f'MFXQDMA_{k[0]},{k[1]}',
                                         center=cen, rho_n=v['rho_n'].tolist()))
            print(f'H={H} rho={rho} done', flush=True)
    json.dump(dict(scales=SM.tolist(), rows=rows), open(RES / 'cross_offset.json', 'w'))
    print('saved results/cross_offset.json', flush=True)


def run_cross_finance(parts=('AB', 'G', 'H')):
    """16 financial pairs (AB), sensitivity of the coefficients to injected errors
    for 16 pairs + 29 DJIA stock-index pairs (G), rolling correlations (H).
    Writes results/cross_finance.json (~15 min; clones qrmdata on first use)."""
    from . import cross_finance as CF
    paths.ensure_qrmdata()
    out = RES / 'cross_finance.json'
    res = json.load(open(out)) if out.exists() else {}
    t0 = time.time()
    for w in parts:
        res[w] = getattr(CF, 'part_' + w)()
        json.dump(res, open(out, 'w'))
        print(f'{w} saved {time.time() - t0:.0f}s', flush=True)
