"""Data tables of Paper II as DataFrames, built from the LaTeX rows that
table_rows_paperII.build() computes from results/. The LaTeX body rows are
also written to tables_out/<name>.tex (they are the rows printed in the paper).
Table VIII (guidelines) contains no computed numbers."""
import re

import pandas as pd

from . import paths
from .table_rows_paperII import build

OUT = paths.REPO / 'tables_out'
OUT.mkdir(exist_ok=True)

TABLES = {
    'I': ('OFF', 'table01_offset',
          ['Trend'] + [f'H={h} ρ={r}' for h in (0.4, 0.7) for r in (0, 0.3, 0.6, 0.9)]),
    'II': ('B1', 'table02_binomial_pairs',
           ['(p_x, p_y)', 'Gap', 'X-DMA vs average', 'X-DMA vs exact', 'X-DFA vs exact',
            'λ=0.25', 'λ=0.5', '(λx,λy)=(0.1,0.9)']),
    'III': ('B2', 'table03_2d_pairs', ['Pair', 'Gap', 'Mean', 'λ=0.25', 'λ=0.5', 'λ=0.75']),
    'IV': ('MONO', 'table04_monofractal_pairs', ['Method', 'BFGN (75)', 'ARFIMA (26)']),
    'V': ('C1', 'table05_contaminated_pair',
          ['Method'] + [f'{m} {e}%' for m in ('x only', 'both indep.', 'both common') for e in (2, 5, 10)]),
    'VI': ('FIN', 'table06_financial_pairs',
           ['Pair', 'Period', 'N', 'r', 'ρ', 'ρ_1/2', 'ρ_1/2,q=1', 'h_xy(2) X-DMA', 'h_xy(2) λ=1/2',
            'h_xy(2) |r| X-DMA', 'h_xy(2) |r| λ=1/2']),
    'VII': ('G', 'table07_coefficient_robustness',
            ['Error', 'Pairs', 'r', 'DMCA ρ', 'ρ_1/2', 'ρ_mid', 'ρ_0.25 c.', 'q=1 DMA', 'q=1 median', 'wins']),
}

_T = {}


def _plain(cell):
    """LaTeX cell -> readable text (bold, math and macros stripped)."""
    c = cell.strip()
    c = re.sub(r'\\(textbf|bm)\{(.*)\}', r'\2', c)
    c = c.replace('\\hline', '').replace('$', '').replace('\\&', '&').replace('\\pm', ' ± ')
    c = c.replace('\\lambda', 'λ').replace('\\theta', 'θ').replace('\\sigma', 'σ').replace('---', '—').replace('--', '–')
    c = re.sub(r'\\textbf\{([^}]*)\}', r'\1', c)
    return c.replace('{', '').replace('}', '').strip()


def table(num, refresh=False):
    """Return Table `num` ('I'...'VII') of Paper II as a DataFrame."""
    if refresh or not _T:
        _T.clear(); _T.update(build())
    key, fname, cols = TABLES[num]
    rows = _T[key]
    (OUT / f'{fname}.tex').write_text(rows + '\n')
    data = [[_plain(c) for c in r.rstrip().rstrip('\\').split(' & ')] for r in rows.split('\n')]
    return pd.DataFrame(data, columns=cols).set_index(cols[0])
