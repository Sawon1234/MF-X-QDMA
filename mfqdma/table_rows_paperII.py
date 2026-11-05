"""LaTeX table rows of Paper II, generated from the stored results (no hand transcription)."""
import json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from . import paths


def build():
    """Return {key: LaTeX body rows} for all data tables, computed from results/."""
    R = str(paths.RESULTS)
    B = json.load(open(f'{R}/cross_bench.json'))
    F = json.load(open(f'{R}/cross_finance.json'))
    O = json.load(open(f'{R}/cross_offset.json'))
    T = {}



    def f3(v):
        return f'{v:.3f}'


    # ---------------------------------------------------------------- binomial pairs
    B1 = pd.DataFrame(B['B1'])
    rows = []
    for (px, py), g in B1.groupby(['px', 'py'], sort=False):
        g = g.set_index('method')
        cells = [f"{g.gap.iloc[0]:.3f}",
                 f3(g.loc['MFXDMA_th0', 'rmse_avg']), f3(g.loc['MFXDMA_th0', 'rmse_exact']),
                 f3(g.loc['MFXDFA', 'rmse_exact']),
                 f3(g.loc['MFXQDMA_th0_0.25,0.25', 'rmse_exact']), f3(g.loc['MFXQDMA_th0_0.5,0.5', 'rmse_exact']),
                 f3(g.loc['MFXQDMA_th0_0.1,0.9', 'rmse_exact'])]
        best = min(float(c) for c in cells[2:])
        cells = cells[:2] + [f'\\textbf{{{c}}}' if abs(float(c) - best) < 5e-4 and float(c) == best else c for c in cells[2:]]
        rows.append(f'({px}, {py}) & ' + ' & '.join(cells) + ' \\\\')
    m = B1.groupby('method')[['rmse_exact', 'rmse_avg']].mean()
    rows.append('\\hline Mean & --- & ' + ' & '.join([
        f3(m.loc['MFXDMA_th0', 'rmse_avg']), f3(m.loc['MFXDMA_th0', 'rmse_exact']), f3(m.loc['MFXDFA', 'rmse_exact']),
        f3(m.loc['MFXQDMA_th0_0.25,0.25', 'rmse_exact']), f3(m.loc['MFXQDMA_th0_0.5,0.5', 'rmse_exact']),
        '\\textbf{' + f3(m.loc['MFXQDMA_th0_0.1,0.9', 'rmse_exact']) + '}']) + ' \\\\')
    T['B1'] = '\n'.join(rows)
    T['B1_means'] = m.sort_values('rmse_exact')

    # ---------------------------------------------------------------- 2D pairs
    B2 = pd.DataFrame(B['B2'])
    rows = []
    for pr, g in B2.groupby('pair'):
        cells = []
        for lam in (None, 0.25, 0.5, 0.75):
            r = g[g.lam.isna()] if lam is None else g[g.lam == lam]
            cells.append((r.rmse_avg.iloc[0], r.rmse_exact.iloc[0]))
        best = min(c[1] for c in cells)
        txt = [f'{a:.3f}/' + (f'\\textbf{{{e:.3f}}}' if e == best else f'{e:.3f}') for a, e in cells]
        rows.append(f'{pr[0]}--{pr[1]} & {g.gap.iloc[0]:.3f} & ' + ' & '.join(txt) + ' \\\\')
    T['B2'] = '\n'.join(rows)

    # ---------------------------------------------------------------- monofractal pairs
    B3 = pd.DataFrame(B['B3']); B4 = pd.DataFrame(B['B4'])
    meths = [('MFXDFA', 'MF-X-DFA'), ('MFXDMA_th0', 'MF-X-DMA, $\\theta=0$'), ('MFXDMA_th0.5', 'MF-X-DMA, $\\theta=1/2$'),
             ('MFXDMA_th1', 'MF-X-DMA, $\\theta=1$'), ('MFXQDMA_th0_0.1,0.1', 'MF-X-QDMA, $\\lambda=0.1$'),
             ('MFXQDMA_th0_0.25,0.25', 'MF-X-QDMA, $\\lambda=0.25$'), ('MFXQDMA_th0_0.5,0.5', 'MF-X-QDMA, $\\lambda=0.5$')]
    rows = []
    b3m = {m: B3[B3.method == m].rmse for m, _ in meths}
    b4m = {m: B4[B4.method == m].rmse for m, _ in meths}
    best3 = min(v.mean() for v in b3m.values() if len(v)); best4 = min(v.mean() for v in b4m.values() if len(v))
    for m, lb in meths:
        c3 = f'${b3m[m].mean():.3f}\\pm{b3m[m].std():.3f}$' if len(b3m[m]) else '---'
        c4 = f'${b4m[m].mean():.3f}\\pm{b4m[m].std():.3f}$' if len(b4m[m]) else '---'
        if len(b3m[m]) and b3m[m].mean() == best3: c3 = '$\\bm{' + c3[1:-1] + '}$'
        if len(b4m[m]) and b4m[m].mean() == best4: c4 = '$\\bm{' + c4[1:-1] + '}$'
        rows.append(f'{lb} & {c3} & {c4} \\\\')
    T['MONO'] = '\n'.join(rows)

    # ---------------------------------------------------------------- contaminated binomial pair
    C1 = pd.DataFrame(B['C1'])
    rows = []
    for m, lb in [('MFXDFA', 'MF-X-DFA'), ('MFXDMA_th0', 'MF-X-DMA'), ('MFXQDMA_th0_0.1,0.1', '$\\lambda=0.1$'),
                  ('MFXQDMA_th0_0.25,0.25', '$\\lambda=0.25$'), ('MFXQDMA_th0_0.5,0.5', '$\\lambda=0.5$'),
                  ('MFXQDMA_th0_0.9,0.9', '$\\lambda=0.9$')]:
        cells = []
        for mode in ('x', 'both_indep', 'both_common'):
            for fr in (0.02, 0.05, 0.10):
                cells.append(C1[(C1['mode'] == mode) & (C1.frac == fr) & (C1.method == m)].rmse.mean())
        rows.append((lb, cells))
    best = [min(r[1][j] for r in rows) for j in range(9)]
    T['C1'] = '\n'.join(lb + ' & ' + ' & '.join((f'\\textbf{{{v:.3f}}}' if v == best[j] else f'{v:.3f}') for j, v in enumerate(c)) + ' \\\\'
                        for lb, c in rows)
    T['C1_clean'] = C1[C1.frac == 0].groupby('method').rmse.mean()

    # ---------------------------------------------------------------- offset table
    D = pd.DataFrame(O['rows']); D['rh'] = D.rho_n.apply(lambda v: np.nanmean(v))
    rows = []
    for (m, cen), lb in [(('MFXDMA', False), 'Moving mean (DMCA)'),
                         (('MFXQDMA_0.5,0.5', False), '$\\lambda=1/2$'),
                         (('MFXQDMA_0.25,0.25', False), '$\\lambda=0.25$, raw'),
                         (('MFXQDMA_0.25,0.25', True), '$\\lambda=0.25$, centred'),
                         (('MFXQDMA_0.1,0.1', False), '$\\lambda=0.1$, raw'),
                         (('MFXQDMA_0.1,0.1', True), '$\\lambda=0.1$, centred'),
                         (('MFXQDMA_0.1,0.9', False), '$(0.1,0.9)$, raw'),
                         (('MFXQDMA_0.1,0.9', True), '$(0.1,0.9)$, centred')]:
        cells = []
        for H in (0.4, 0.7):
            for r in (0.0, 0.3, 0.6, 0.9):
                S = D[(D.method == m) & (D.center == cen) & (D.H == H) & (D.rho == r)].rh
                cells.append(f'{S.mean():+.2f}')
        rows.append(lb + ' & ' + ' & '.join(cells) + ' \\\\')
    T['OFF'] = '\n'.join(rows)

    # ---------------------------------------------------------------- finance pairs
    AB = F['AB']
    LAB = {'SP500': 'S\\&P 500', 'DJ': 'DJIA', 'NASDAQ': 'NDX', 'FTSE': 'FTSE', 'DAX': 'DAX', 'CAC': 'CAC',
           'NIKKEI': 'Nikkei', 'HSI': 'HSI', 'SSEC': 'SSEC', 'CSI': 'CSI 300', 'EUR_USD': 'EUR', 'GBP_USD': 'GBP',
           'JPY_USD': 'JPY', 'GOLD': 'Gold', 'OIL_Brent': 'Brent', 'VIX': 'VIX', 'UST2': 'UST 2y',
           'UST10': 'UST 10y', 'BTC': 'Bitcoin'}
    G = F['G']
    rows = []
    for p, v in AB.items():
        a, b = p.split('|')
        g = G[p]['base']
        o, ab = v['orig'], v['abs']
        rows.append(f"{LAB[a]}--{LAB[b]} & {v['start'][:4]}--{v['end'][2:4]} & {v['N']} & {v['pearson']:+.2f} & "
                    f"{np.nanmean(g['dma']['rho']):+.2f} & {np.nanmean(g['q50']['rho']):+.2f} & {np.nanmean(g['q50']['rho1']):+.2f} & "
                    f"{o['MFXDMA']['h2']:.2f} & {o['MFXQDMA_0.5']['h2']:.2f} & {ab['MFXDMA']['h2']:.2f} & {ab['MFXQDMA_0.5']['h2']:.2f} \\\\")
    T['FIN'] = '\n'.join(rows)

    # ---------------------------------------------------------------- finance robustness
    rows = []
    for p, v in G.items():
        for r in v['rows']:
            r = dict(r); r['pair'] = p; r['set'] = 'DJ' if p.startswith('DJ:') else 'mkt'; rows.append(r)
    GD = pd.DataFrame(rows); GD['kappa'] = GD.kappa.fillna(0)
    scen = [('badtick', 'x', 10.0, 'Bad ticks in $x$, $10\\sigma$'), ('badtick', 'x', 50.0, 'Bad ticks in $x$, $50\\sigma$'),
            ('badtick', 'both', 10.0, 'Bad ticks in both, $10\\sigma$'), ('badtick', 'both', 50.0, 'Bad ticks in both, $50\\sigma$'),
            ('outlier', 'x', 10.0, 'Return outliers in $x$'), ('noise', 'both', 0.0, 'Microstructure noise')]
    keys = ['pearson', 'dma_rho', 'q50_rho', 'mid25_rho', 'q25c_rho', 'dma_rho1', 'q50_rho1']
    rows = []
    stats = {}
    for kind, mode, kap, lb in scen:
        for st in ('mkt', 'DJ'):
            S = GD[(GD.kind == kind) & (GD['mode'] == mode) & (GD.kappa == kap) & (GD.set == st)].groupby('pair').mean(numeric_only=True)
            vals = [S[k].median() for k in keys]
            best = min(vals[1:])
            cells = [f'{1e3 * v:.0f}' if k == 'pearson' else (f'\\textbf{{{1e3 * v:.1f}}}' if v == best else f'{1e3 * v:.1f}')
                     for k, v in zip(keys, vals)]
            w = int((S['q50_rho1'] < S['dma_rho']).sum()); n = len(S)
            p = wilcoxon(S['q50_rho1'], S['dma_rho']).pvalue
            w2 = int((S['q50_rho'] < S['dma_rho']).sum()); p2 = wilcoxon(S['q50_rho'], S['dma_rho']).pvalue
            stats[(kind, mode, kap, st)] = dict(w=w, n=n, p=p, w2=w2, p2=p2,
                                                r1=float(np.median(S['q50_rho1'] / S['dma_rho'])),
                                                r2=float(np.median(S['q50_rho'] / S['dma_rho'])),
                                                hq50=float(np.median(S['q50_h'] / S['dma_h'])),
                                                hw=int((S['q50_h'] < S['dma_h']).sum()),
                                                hp=wilcoxon(S['q50_h'], S['dma_h']).pvalue)
            rows.append(f"{lb if st == 'mkt' else ''} & {'16 pairs' if st == 'mkt' else '29 stocks'} & " + ' & '.join(cells)
                        + f' & {w}/{n} \\\\')
    T['G'] = '\n'.join(rows)
    T['G_stats'] = stats


    return T
