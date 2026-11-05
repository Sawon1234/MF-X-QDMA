"""Repository paths and data helpers (no hard-coded absolute paths)."""
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / 'results'
FIGURES = REPO / 'figures_out'          # figures produced by the notebooks
CACHE = REPO / 'cache'                  # cached intermediate computations
DATA = REPO / 'data'
QRM_REPO = DATA / 'qrmdata'             # git clone of github.com/cran/qrmdata
QRM_DATA = QRM_REPO / 'data'
BONN = DATA / 'bonn'                    # Bonn EEG: folders Z, O, N, F, S with .txt files

for _d in (FIGURES, CACHE, DATA, RESULTS):
    _d.mkdir(parents=True, exist_ok=True)


def ensure_qrmdata():
    """Clone the CRAN mirror of qrmdata (Hofert, Hornik & McNeil) on first use.
    The .rda files are read with mfqdma.rdata_reader, a pure-data parser."""
    if not (QRM_DATA / 'SP500.rda').exists():
        print('cloning https://github.com/cran/qrmdata ...', flush=True)
        subprocess.run(['git', 'clone', '--depth', '1',
                        'https://github.com/cran/qrmdata', str(QRM_REPO)], check=True)
    return QRM_DATA


def ensure_bonn():
    if not (BONN / 'Z').is_dir():
        raise FileNotFoundError(
            'Bonn EEG data not found. Download the five sets (Z, O, N, F, S) from\n'
            'https://www.ukbonn.de/epileptologie/arbeitsgruppen/ag-lehnertz-neurophysik/downloads/\n'
            f'and extract them so that {BONN}/Z/*.txt ... {BONN}/S/*.txt exist.')
    return BONN
