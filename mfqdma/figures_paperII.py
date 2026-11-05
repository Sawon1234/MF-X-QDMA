"""Figures of Paper II (MF-X-QDMA, cross-correlation). Each function writes
one manuscript figure to figures_out/ from the stored results in results/."""
import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .plotting import (RES, OUT, W1, W2, Q, C_DFA, C_DMA, C_MID, C_PEA, TCOL, FIGNAME,
                       lab, logticks, save, J)
from .methods_cross import (binomial_pair_exact, binomial_pair_average, mfx1d,
                            cascade2d_pair_tau_exact, cascade2d_pair_tau_average)

# ------------------------------------------------------------------ Fig 1
def fig1():
    from . import cross_finance as cf
    from .methods import _profile
    from .methods_cross import residuals_1d
    d, x, y = cf.aligned('DAX', 'CAC')
    Yx, Yy = _profile(x), _profile(y)
    n = 60; sl = slice(4200, 4700)
    rx, ry = residuals_1d(Yx, n, 0.0, [0.25, 0.5]), residuals_1d(Yy, n, 0.0, [0.25, 0.5])
    nb = n - 1
    win_x = np.lib.stride_tricks.sliding_window_view(Yx, n)
    win_y = np.lib.stride_tricks.sliding_window_view(Yy, n)
    t = d.astype('datetime64[D]').astype(float) / 365.25 + 1970
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.0))
    ii = np.arange(len(Yx))[sl]
    for Y, w, c, nm in ((Yx, win_x, 'k', 'DAX'), (Yy, win_y, '0.45', 'CAC')):
        ax[0].plot(t[ii], Y[ii], color=c, lw=0.9, label=f'{nm} profile')
        ax[0].plot(t[ii], np.median(w[sl.start - nb:sl.stop - nb], axis=1), color=TCOL[0.5],
                   lw=0.9, ls='--')
    ax[0].plot([], [], '--', color=TCOL[0.5], label=r'moving median')
    ax[0].set(xlabel='year', ylabel='$Y(i)$')
    ax[0].legend(frameon=False, fontsize=5.5)
    lab(ax[0], '(a)')
    ex, ey = rx[0.5][sl.start - nb:sl.stop - nb], ry[0.5][sl.start - nb:sl.stop - nb]
    ax[1].plot(t[ii], ex, color='k', lw=0.6, label=r'$\epsilon_x$')
    ax[1].plot(t[ii], ey, color='0.55', lw=0.6, label=r'$\epsilon_y$')
    ax[1].fill_between(t[ii], 0, ex * ey / np.std(ex), color=TCOL[0.25], alpha=0.5,
                       label=r'$\epsilon_x\epsilon_y$ (scaled)')
    ax[1].axhline(0, color='k', lw=0.4)
    ax[1].set(xlabel='year', ylabel='residual')
    ax[1].legend(frameon=False, fontsize=5.5, ncol=3, loc='upper center')
    lab(ax[1], r'(b) $\lambda=1/2$, $n=60$')
    sm, _ = cf.scales(len(x))
    from .methods_cross import coef_suite
    cs = coef_suite(x, y, sm)
    for k, c, lb in (('dma', C_DMA, 'moving mean'), ('q50', TCOL[0.5], r'median ($\lambda=1/2$)'),
                     ('mid25', C_MID, 'midhinge'), ('q25c', TCOL[0.25], r'$\lambda=0.25$, centred')):
        ax[2].semilogx(sm, cs[k]['rho'], 'o-', ms=2.4, color=c, label=lb)
    ax[2].axhline(np.corrcoef(x, y)[0, 1], color='k', ls=':', lw=0.8, label='Pearson $r$')
    ax[2].set(xlabel='$n$ (days)', ylabel=r'$\rho(n)$')
    ax[2].legend(frameon=False, fontsize=5.3)
    lab(ax[2], '(c) DAX--CAC')
    fig.tight_layout(w_pad=0.5)
    save(fig, 'figII01_schematic')


# ------------------------------------------------------------------ Fig 2
def fig2():
    d = J('cross_offset')
    D = pd.DataFrame(d['rows']); D['rhohat'] = D.rho_n.apply(lambda v: np.nanmean(v))
    sm = np.array(d['scales'])
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.0))
    S = D[(D.H == 0.7) & (D.rho == 0.0)]
    for m, cen, c, ls, lb in (('MFXDMA', False, C_DMA, '-', 'moving mean'),
                              ('MFXQDMA_0.1,0.1', False, TCOL[0.1], '--', r'$\lambda=0.1$, raw'),
                              ('MFXQDMA_0.1,0.1', True, TCOL[0.1], '-', r'$\lambda=0.1$, centred'),
                              ('MFXQDMA_0.25,0.25', False, TCOL[0.25], '--', r'$\lambda=0.25$, raw'),
                              ('MFXQDMA_0.1,0.9', False, TCOL[0.9], '--', r'$(0.1,0.9)$, raw'),
                              ('MFXQDMA_0.1,0.9', True, TCOL[0.9], '-', r'$(0.1,0.9)$, centred')):
        g = np.array(S[(S.method == m) & (S.center == cen)].rho_n.tolist())
        ax[0].semilogx(sm, g.mean(0), ls, color=c, lw=1.0, label=lb)
    ax[0].axhline(0, color='k', lw=0.5)
    ax[0].set(xlabel='$n$', ylabel=r'$\hat\rho(n)$', ylim=(-0.6, 0.6))
    ax[0].legend(frameon=False, fontsize=5.0, ncol=2, loc='upper center')
    lab(ax[0], r'(a) independent pair, $\rho=0$')
    rhos = sorted(D.rho.unique())
    for (m, cen), c, mk, lb in (((('MFXDMA', False)), C_DMA, 'o', 'moving mean'),
                                (('MFXQDMA_0.5,0.5', False), TCOL[0.5], 's', r'$\lambda=1/2$'),
                                (('MFXQDMA_0.25,0.25', True), TCOL[0.25], 's', r'$\lambda=0.25$ c.'),
                                (('MFXQDMA_0.25,0.25', False), TCOL[0.25], 'x', r'$\lambda=0.25$ raw'),
                                (('MFXQDMA_0.1,0.1', False), TCOL[0.1], 'x', r'$\lambda=0.1$ raw')):
        v = [D[(D.method == m) & (D.center == cen) & (D.rho == r) & (D.H == 0.7)].rhohat.mean() for r in rhos]
        ax[1].plot(rhos, np.array(v) - np.array(rhos), mk + '-', color=c, ms=3, label=lb,
                   mfc='none' if mk == 'o' else c)
    ax[1].axhline(0, color='k', lw=0.5)
    ax[1].set(xlabel=r'true $\rho$', ylabel=r'bias $\langle\hat\rho\rangle-\rho$')
    ax[1].legend(frameon=False, fontsize=5.2)
    lab(ax[1], r'(b) $H=0.7$')
    for H, ls in ((0.4, '-'), (0.7, '--')):
        for (m, cen), c in ((('MFXQDMA_0.1,0.1', True), TCOL[0.1]), (('MFXQDMA_0.25,0.25', True), TCOL[0.25]),
                            (('MFXQDMA_0.5,0.5', False), TCOL[0.5]), (('MFXDMA', False), C_DMA)):
            v = [D[(D.method == m) & (D.center == cen) & (D.rho == r) & (D.H == H)].rhohat.mean() for r in rhos]
            ax[2].plot(rhos, np.array(v) / np.where(np.array(rhos) == 0, np.nan, rhos), ls, color=c, lw=1.0)
    ax[2].plot([], [], 'k-', label='$H=0.4$'); ax[2].plot([], [], 'k--', label='$H=0.7$')
    ax[2].set(xlabel=r'true $\rho$', ylabel=r'$\langle\hat\rho\rangle/\rho$', ylim=(0.85, 1.05))
    ax[2].legend(frameon=False, fontsize=5.5)
    lab(ax[2], '(c) attenuation')
    fig.tight_layout(w_pad=0.5)
    save(fig, 'figII02_offset')


# ------------------------------------------------------------------ Fig 3
def fig3():
    B = pd.DataFrame(J('cross_bench')['B1'])
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.05))
    px, py = 0.2, 0.45
    qq = np.linspace(-4, 4, 161)
    ax[0].plot(qq, binomial_pair_exact(qq, px, py), 'k-', lw=1.5, label='exact joint partition')
    ax[0].plot(qq, binomial_pair_average(qq, px, py), 'k:', lw=1.2, label=r'$(h_{xx}+h_{yy})/2$')
    S = B[(B.px == px) & (B.py == py)].set_index('method')
    for m, c, mk, lb in (('MFXDFA', C_DFA, 'v', 'MF-X-DFA'), ('MFXDMA_th0', C_DMA, 'o', 'MF-X-DMA'),
                         ('MFXQDMA_th0_0.25,0.25', TCOL[0.25], 's', r'$\lambda=0.25$'),
                         ('MFXQDMA_th0_0.1,0.9', C_MID, 'D', r'$(\lambda_x,\lambda_y)=(0.1,0.9)$')):
        ax[0].plot(Q, S.loc[m, 'hq'], mk, color=c, ms=2.8, mfc='none' if mk in 'ov' else c, label=lb)
    ax[0].set(xlabel='$q$', ylabel=r'$h_{xy}(q)$')
    ax[0].legend(frameon=False, fontsize=5.2)
    lab(ax[0], r'(a) $p_x=0.2$, $p_y=0.45$')
    G = B.groupby(['px', 'py']).first().gap
    for m, c, mk, lb in (('MFXDMA_th0', C_DMA, 'o', 'MF-X-DMA'),
                         ('MFXQDMA_th0_0.25,0.25', TCOL[0.25], 's', r'$\lambda=0.25$'),
                         ('MFXQDMA_th0_0.1,0.9', C_MID, 'D', r'$(0.1,0.9)$')):
        S = B[B.method == m].set_index(['px', 'py'])
        ax[1].plot(G.values, S.loc[G.index, 'rmse_avg'], mk, color=c, mfc='none', ms=3.5)
        ax[1].plot(G.values, S.loc[G.index, 'rmse_exact'], mk, color=c, ms=3.5, label=lb)
    ax[1].plot([], [], 'ko', mfc='none', label='vs average rule'); ax[1].plot([], [], 'ko', label='vs exact')
    ax[1].set(xlabel=r'$\max_q|h^{\rm exact}_{xy}-h^{\rm avg}_{xy}|$', ylabel='RMSE of $h_{xy}(q)$',
              yscale='log')
    ax[1].legend(frameon=False, fontsize=5.0, ncol=2)
    lab(ax[1], '(b) eight pairs')
    meths = ['MFXDFA', 'MFXDMA_th0.5', 'MFXDMA_th1', 'MFXDMA_th0', 'MFXQDMA_th0_0.5,0.5',
             'MFXQDMA_th0_0.25,0.25', 'MFXQDMA_th0_0.25,0.75', 'MFXQDMA_th0_0.1,0.9']
    labels = ['X-DFA', r'X-DMA $\theta=\frac{1}{2}$', r'X-DMA $\theta=1$', r'X-DMA $\theta=0$',
              r'$\lambda=0.5$', r'$\lambda=0.25$', r'$(0.25,0.75)$', r'$(0.1,0.9)$']
    v = [B[B.method == m].rmse_exact.mean() for m in meths]
    cols = [C_DFA, C_DMA, C_DMA, C_DMA, TCOL[0.5], TCOL[0.25], C_MID, C_MID]
    ax[2].barh(range(len(meths)), v, color=cols)
    ax[2].set_yticks(range(len(meths))); ax[2].set_yticklabels(labels, fontsize=6)
    ax[2].set(xlabel='mean RMSE vs exact')
    lab(ax[2], '(c)')
    fig.tight_layout(w_pad=0.5)
    save(fig, 'figII03_binomial_pairs')


# ------------------------------------------------------------------ Fig 4
def fig4():
    B = pd.DataFrame(J('cross_bench')['B2'])
    P = {'A': (0.05, 0.15, 0.25, 0.55), 'B': (0.10, 0.20, 0.30, 0.40), 'C': (0.02, 0.18, 0.30, 0.50)}
    fig, ax = plt.subplots(1, 2, figsize=(W1 * 1.3, 1.9))
    pr = 'BC'
    te = cascade2d_pair_tau_exact(Q, P['B'], P['C']); ta = cascade2d_pair_tau_average(Q, P['B'], P['C'])
    S = B[B.pair == pr]
    ax[0].axhline(0, color='k', lw=0.5)
    ax[0].plot(Q, ta - te, 'k:', lw=1.2, label='average rule')
    for lam, c, mk, lb in ((None, C_DMA, 'o', '2D-MF-X-DMA'), (0.25, TCOL[0.25], 's', r'$\lambda=0.25$'),
                           (0.5, TCOL[0.5], 's', r'$\lambda=0.5$')):
        r = S[S.lam.isna()] if lam is None else S[S.lam == lam]
        ax[0].plot(Q, np.array(r.iloc[0]['tau']) - te, mk + '-', color=c, ms=2.5,
                   mfc='none' if mk == 'o' else c, label=lb)
    ax[0].set(xlabel='$q$', ylabel=r'$\tau_{xy}(q)-\tau^{\rm exact}_{xy}(q)$')
    ax[0].legend(frameon=False, fontsize=5)
    lab(ax[0], '(a) pair B--C')
    x = np.arange(3); w = 0.2
    for j, (lam, c, lb) in enumerate(((None, C_DMA, 'mean'), (0.25, TCOL[0.25], r'$\lambda=0.25$'),
                                      (0.5, TCOL[0.5], r'$\lambda=0.5$'), (0.75, TCOL[0.75], r'$\lambda=0.75$'))):
        v = [(B[(B.pair == p) & (B.lam.isna())] if lam is None else B[(B.pair == p) & (B.lam == lam)]).rmse_exact.iloc[0]
             for p in ('AB', 'AC', 'BC')]
        ax[1].bar(x + (j - 1.5) * w, v, w, color=c, label=lb)
    ax[1].set_xticks(x); ax[1].set_xticklabels(['A--B', 'A--C', 'B--C'])
    ax[1].set(ylabel=r'RMSE of $\tau_{xy}(q)$')
    ax[1].legend(frameon=False, fontsize=5)
    lab(ax[1], r'(b) $256^2$')
    fig.tight_layout(w_pad=0.5)
    save(fig, 'figII04_2d_pairs')


# ------------------------------------------------------------------ Fig 5
def fig5():
    Jb = J('cross_bench')
    B3 = pd.DataFrame(Jb['B3']); B4 = pd.DataFrame(Jb['B4'])
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.0))
    meths = ['MFXDFA', 'MFXDMA_th0', 'MFXDMA_th0.5', 'MFXQDMA_th0_0.1,0.1', 'MFXQDMA_th0_0.25,0.25',
             'MFXQDMA_th0_0.5,0.5']
    lb = ['X-DFA', r'X-DMA $\theta=0$', r'X-DMA $\theta=\frac{1}{2}$', r'$\lambda=0.1$', r'$\lambda=0.25$', r'$\lambda=0.5$']
    cols = [C_DFA, C_DMA, '#8fa0e0', TCOL[0.1], TCOL[0.25], TCOL[0.5]]
    for a, D, tt in ((ax[0], B3, '(a) BFBM, 75 pairs'), (ax[1], B4, '(b) 2-comp. ARFIMA, 26')):
        bp = a.boxplot([D[D.method == m].rmse.values for m in meths], patch_artist=True,
                       widths=0.6, showfliers=False)
        for p, c in zip(bp['boxes'], cols):
            p.set_facecolor(c); p.set_alpha(0.85); p.set_linewidth(0.5)
        for mm in bp['medians']: mm.set_color('k')
        a.set_xticks(range(1, len(meths) + 1)); a.set_xticklabels(lb, rotation=60, fontsize=5.5)
        lab(a, tt)
    ax[0].set_ylabel(r'RMSE of $h_{xy}(q)$')
    S = B3[B3.Hxx == B3.Hyy].copy(); S['rh'] = S.rho_n.apply(lambda v: np.nanmean(v))
    for m, c, mk in zip(meths, cols, ['v', 'o', 'D', 's', 's', 's']):
        g = S[S.method == m].groupby('rho').rh.mean()
        ax[2].plot(g.index, g.values - g.index, mk + '-', color=c, ms=2.6)
    ax[2].axhline(0, color='k', lw=0.5)
    ax[2].set(xlabel=r'BFBM $\rho$ ($H_{xx}=H_{yy}$)', ylabel=r'$\langle\hat\rho\rangle-\rho$')
    lab(ax[2], '(c) coefficient bias')
    fig.tight_layout(w_pad=0.5)
    save(fig, 'figII05_bfbm_arfima')


# ------------------------------------------------------------------ Fig 6
def fig6():
    Jb = J('cross_bench')
    C1 = pd.DataFrame(Jb['C1'])
    C2 = pd.DataFrame(Jb['C2'])
    fig, ax = plt.subplots(1, 4, figsize=(W2, 2.0))
    meths = [('MFXDFA', C_DFA, 'v', 'X-DFA'), ('MFXDMA_th0', C_DMA, 'o', 'X-DMA'),
             ('MFXQDMA_th0_0.1,0.1', TCOL[0.1], 's', r'$\lambda=0.1$'),
             ('MFXQDMA_th0_0.25,0.25', TCOL[0.25], 's', r'$\lambda=0.25$'),
             ('MFXQDMA_th0_0.5,0.5', TCOL[0.5], 's', r'$\lambda=0.5$'),
             ('MFXQDMA_th0_0.9,0.9', TCOL[0.9], 's', r'$\lambda=0.9$')]
    for a, mode, tt in zip(ax[:3], ('x', 'both_indep', 'both_common'),
                           ('(a) outliers in $x$ only', '(b) both, independent', '(c) both, common days')):
        S = C1[C1['mode'] == mode]
        for m, c, mk, lb in meths:
            g = S[S.method == m].groupby('frac').rmse.mean()
            a.plot(100 * g.index, g.values, mk + '-', color=c, ms=2.5, mfc='none' if mk in 'ov' else c, label=lb)
        a.set(xlabel=r'$\varepsilon$ (%)', yscale='log')
        lab(a, tt)
    ax[0].set_ylabel(r'RMSE of $h_{xy}(q)$')
    ax[0].legend(frameon=False, fontsize=5, ncol=2, loc='lower right')
    C3 = pd.DataFrame(Jb['C3'])
    S = C3[(C3.kind != 'clean') & (C3['mode'] == 'x')]
    for k, col, mk, lb in (('dma', C_DMA, 'o', 'DMCA'), ('q50', TCOL[0.5], 's', r'$\lambda=\frac{1}{2}$'),
                           ('q25c', TCOL[0.25], 's', r'$\lambda=0.25$ c.')):
        for kind, ls in (('badtick', '-'), ('spike', '--')):
            g = S[(S.method == k) & (S.kind == kind)].groupby('frac').drho.mean()
            ax[3].plot(100 * g.index, g.values, mk + ls, color=col, ms=2.3, mfc='none' if mk == 'o' else col,
                       label=lb if kind == 'badtick' else None)
    g = S[(S.method == 'q50') & (S.kind == 'badtick')].groupby('frac').drho1.mean()
    ax[3].plot(100 * g.index, g.values, 'x-', color=TCOL[0.9], ms=3, label=r'$\lambda=\frac{1}{2}$, $q=1$')
    g = S[(S.method == 'q50') & (S.kind == 'spike')].groupby('frac').drho1.mean()
    ax[3].plot(100 * g.index, g.values, 'x--', color=TCOL[0.9], ms=3)
    ax[3].set(xlabel=r'$\varepsilon$ (%)', ylabel=r'mean $|\delta\rho(n)|$', yscale='log', xscale='log')
    logticks(ax[3], [0.5, 1, 2, 5])
    ax[3].legend(frameon=False, fontsize=4.8, loc='center right')
    lab(ax[3], r'(d) BFGN, errors in $x$')
    fig.tight_layout(w_pad=0.35)
    save(fig, 'figII06_contamination')


# ------------------------------------------------------------------ Fig 7  finance spectra
def fig7():
    AB = J('cross_finance')['AB']
    fig, ax = plt.subplots(1, 4, figsize=(W2, 2.0))
    o = AB['DJ|NASDAQ']['orig']['MFXQDMA_0.5']
    s = np.array(o['scales']); Fq = np.array(o['Fq'])
    for qi, qv in enumerate(Q):
        if qv in (-4, -2, 0, 2, 4):
            c = plt.cm.viridis((qv + 4) / 8.5); sh = 10 ** (0.6 * qv / 2)
            ok = np.isfinite(Fq[qi])
            ax[0].loglog(s[ok], Fq[qi][ok] * sh, 'o', color=c, ms=2.3, label=f'$q={int(qv)}$')
            ax[0].loglog(s[ok], np.exp(np.polyval(np.polyfit(np.log(s[ok]), np.log(Fq[qi][ok]), 1),
                                                  np.log(s[ok]))) * sh, '-', color=c, lw=0.7)
    ax[0].set(xlabel='$n$ (days)', ylabel=r'$F_{xy,q}(n)$ (shifted)')
    ax[0].legend(frameon=False, fontsize=4.8, loc='upper left')
    lab(ax[0], r'(a) DJIA--NDX, $\lambda=\frac{1}{2}$')
    for pair, c in (('DJ|NASDAQ', 'k'), ('SP500|VIX', '0.5')):
        for m, ls in (('MFXDMA', '--'), ('MFXQDMA_0.5', '-')):
            oo = AB[pair]['orig'][m]
            qq = Q >= 0
            ax[1].plot(Q[qq], np.array(oo['hq'])[qq], ls, color=c, lw=1.0)
            ax[1].plot(Q[qq], 0.5 * (np.array(oo['hxx']) + np.array(oo['hyy']))[qq], ls, color=c, lw=0.5, alpha=0.6)
    ax[1].plot([], [], 'k-', label='DJIA--NDX'); ax[1].plot([], [], '-', color='0.5', label='S&P--VIX')
    ax[1].plot([], [], 'k--', lw=0.8, label='MF-X-DMA'); ax[1].plot([], [], 'k-', lw=0.8, label=r'$\lambda=\frac{1}{2}$')
    ax[1].set(xlabel='$q$', ylabel=r'$h_{xy}(q)$; thin: $(h_{xx}+h_{yy})/2$')
    ax[1].legend(frameon=False, fontsize=5)
    lab(ax[1], '(b) returns')
    for pair, c in (('SP500|FTSE', '#d62728'), ('NIKKEI|HSI', '#9467bd'), ('DAX|CAC', '#1f77b4'),
                    ('EUR_USD|GBP_USD', '#2ca02c'), ('SP500|VIX', '#7f7f7f'), ('UST2|UST10', '#ff7f0e')):
        o = AB[pair]['orig']['MFXQDMA_0.5']
        ax[2].semilogx(o['scales'], o['rho_n'], '-', color=c, lw=1.0, label=pair.replace('|', '--').replace('_USD', ''))
        ax[2].plot([o['scales'][0]], [AB[pair]['pearson']], '<', color=c, ms=3)
    ax[2].axhline(0, color='k', lw=0.4)
    ax[2].set(xlabel='$n$ (days)', ylabel=r'$\rho_{1/2}(n)$', ylim=(-1.0, 1.25))
    ax[2].legend(frameon=False, fontsize=4.4, ncol=2, loc='center', bbox_to_anchor=(0.55, 0.22))
    lab(ax[2], '(c) scale dependence')
    pairs = list(AB.keys())
    hr = [AB[p]['orig']['MFXQDMA_0.5']['h2'] for p in pairs]
    ha = [AB[p]['abs']['MFXQDMA_0.5']['h2'] for p in pairs]
    ax[3].plot(range(len(pairs)), hr, 's', color=TCOL[0.5], ms=3, label='returns')
    ax[3].plot(range(len(pairs)), ha, 'o', color='k', ms=3, mfc='none', label='$|$returns$|$')
    ax[3].axhline(0.5, color='k', lw=0.4, ls=':')
    ax[3].set_xticks(range(len(pairs)))
    ax[3].set_xticklabels([p.replace('_USD', '').replace('SP500', 'SPX').replace('NASDAQ', 'NDX')
                           .replace('NIKKEI', 'N225').replace('OIL_Brent', 'Brent').replace('UST', 'T')
                           .replace('|', '-') for p in pairs], rotation=90, fontsize=4.6)
    ax[3].set(ylabel=r'$h_{xy}(2)$')
    ax[3].legend(frameon=False, fontsize=5)
    lab(ax[3], r'(d) 16 pairs, $\lambda=\frac{1}{2}$')
    fig.tight_layout(w_pad=0.35)
    save(fig, 'figII07_fin_spectra')


# ------------------------------------------------------------------ Fig 8  finance robustness
SUITE_LAB = {'pearson': ('Pearson $r$', C_PEA, '^'), 'dma_rho': ('DMCA $\\rho$', C_DMA, 'o'),
             'q50_rho': (r'$\rho_{1/2}$', TCOL[0.5], 's'), 'mid25_rho': (r'$\rho_{\rm mid}$', C_MID, 'D'),
             'q25c_rho': (r'$\rho_{0.25}$ c.', TCOL[0.25], 's'),
             'dma_rho1': (r'DMCA $\rho_{q=1}$', C_DMA, 'x'), 'q50_rho1': (r'$\rho_{1/2,q=1}$', TCOL[0.5], 'x')}


def _G():
    G = J('cross_finance')['G']
    rows = []
    for p, v in G.items():
        for r in v['rows']:
            r = dict(r); r['pair'] = p; r['set'] = 'DJ' if p.startswith('DJ:') else 'mkt'; rows.append(r)
    return pd.DataFrame(rows)


def fig8():
    D = _G()
    keys = ['pearson', 'dma_rho', 'q50_rho', 'mid25_rho', 'q25c_rho', 'dma_rho1', 'q50_rho1']
    scen = [('badtick', 'x', 10.0, r'bad ticks in $x$, $10\sigma$'), ('badtick', 'x', 50.0, r'bad ticks in $x$, $50\sigma$'),
            ('badtick', 'both', 50.0, r'both, $50\sigma$'), ('noise', 'both', None, r'noise, $\eta=0.5$')]
    fig, ax = plt.subplots(1, 4, figsize=(W2, 2.1), sharey=False)
    for a, (kind, mode, kap, tt) in zip(ax, scen):
        S = D[(D.kind == kind) & (D['mode'] == mode)]
        S = S[S.kappa.isna()] if kap is None else S[S.kappa == kap]
        g = S.groupby('pair')[keys].mean()
        bp = a.boxplot([g[k].values for k in keys], patch_artist=True, widths=0.6, showfliers=False)
        for p, k in zip(bp['boxes'], keys):
            p.set_facecolor(SUITE_LAB[k][1]); p.set_alpha(0.7); p.set_linewidth(0.5)
        for mm in bp['medians']: mm.set_color('k')
        a.set_xticks(range(1, len(keys) + 1))
        a.set_xticklabels(['Pearson', 'DMCA', 'median', 'midhinge', r'$\lambda$=.25 c.', 'DMCA $q$=1',
                           'median $q$=1'], fontsize=5.2, rotation=60, ha='right')
        a.set(yscale='log')
        lab(a, f'({"abcd"[scen.index((kind, mode, kap, tt))]}) {tt}')
    ax[0].set_ylabel(r'mean $|\delta\rho|$ over scales')
    fig.tight_layout(w_pad=0.3)
    save(fig, 'figII08_fin_coeff_robust')


def fig9():
    H = J('cross_finance')['H']
    D = _G()
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.05), gridspec_kw=dict(width_ratios=[1.4, 1.4, 1]))
    for a, pair, tt in ((ax[0], 'SP500|UST10', '(a) S&P 500--UST 10y'), (ax[1], 'DAX|CAC', '(b) DAX--CAC')):
        r = H[pair]
        t = np.array([np.datetime64(x).astype('datetime64[D]').astype(float) / 365.25 + 1970 for x in r['t']])
        a.plot(t, r['clean']['pearson'], '-', color='0.6', lw=0.8, label='Pearson, clean')
        a.plot(t, r['dirty']['pearson'], ':', color='k', lw=0.9, label=r'Pearson, $50\sigma$ ticks')
        a.plot(t, r['clean']['q50_rho1'], '-', color=TCOL[0.5], lw=1.0, label=r'$\rho_{1/2,q=1}$, clean')
        a.plot(t, r['dirty']['q50_rho1'], '--', color=TCOL[0.9], lw=1.0, label=r'$\rho_{1/2,q=1}$, ticks')
        a.axhline(0, color='k', lw=0.4)
        a.set(xlabel='end of 4-yr window', ylabel=rf"correlation ($n={r['n']}$ d)")
        lab(a, tt)
    ax[0].legend(frameon=False, fontsize=4.9, loc='upper left')
    for nm, fun in (('h', lambda k: f'{k}_h'),):
        S = D[(D.kind == 'badtick') & (D['mode'] == 'x') & (D.kappa == 50.0)].groupby('pair').mean(numeric_only=True)
        keys = ['dma', 'q50', 'mid25', 'q25c']
        bp = ax[2].boxplot([S[f'{k}_h'].values for k in keys], patch_artist=True, widths=0.6, showfliers=False)
        for p, k, c in zip(bp['boxes'], keys, (C_DMA, TCOL[0.5], C_MID, TCOL[0.25])):
            p.set_facecolor(c); p.set_alpha(0.7); p.set_linewidth(0.5)
        for mm in bp['medians']: mm.set_color('k')
        ax[2].set_xticks(range(1, 5)); ax[2].set_xticklabels(['DMA', r'$\frac{1}{2}$', 'mid', '.25c'], fontsize=6)
        ax[2].set(ylabel=r'$D_h$, $q\in[1,4]$', yscale='log')
        lab(ax[2], r'(c) $h_{xy}$, $50\sigma$ ticks in $x$')
    fig.tight_layout(w_pad=0.4)
    save(fig, 'figII09_fin_rolling')




# ------------------------------------------------------------ manuscript order
FIGURES = [
    (1, 'Fig01_schematic', 'figII01_schematic', fig1),
    (2, 'Fig02_offset_bias', 'figII02_offset', fig2),
    (3, 'Fig03_binomial_pairs', 'figII03_binomial_pairs', fig3),
    (4, 'Fig04_2d_cascade_pairs', 'figII04_2d_pairs', fig4),
    (5, 'Fig05_bfgn_arfima', 'figII05_bfbm_arfima', fig5),
    (6, 'Fig06_contamination', 'figII06_contamination', fig6),
    (7, 'Fig07_finance_spectra', 'figII07_fin_spectra', fig7),
    (8, 'Fig08_finance_coefficient_robustness', 'figII08_fin_coeff_robust', fig8),
    (9, 'Fig09_finance_rolling', 'figII09_fin_rolling', fig9),
]
for _num, _out, _internal, _fn in FIGURES:
    FIGNAME[_internal] = _out


def make_figure(number):
    """Draw manuscript figure `number` (1-9) and save it to figures_out/."""
    import time
    for num, out, _, fn in FIGURES:
        if num == number:
            t0 = time.time()
            fn()
            print(f'Fig. {num} ({out}) done in {time.time() - t0:.0f} s', flush=True)
            return out
    raise ValueError(f'no figure {number}')
