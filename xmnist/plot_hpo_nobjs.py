import numpy as np
np.set_printoptions(precision=4)
from itertools import product
import sys
import os
from pathlib import Path
import warnings
warnings.simplefilter('ignore')

import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import pickle
import argparse
import os


import logging
logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger('PLOT-HPO-NOBJS')
logger.setLevel(logging.DEBUG)



DPI=100
LW=1.0
ALPHA=0.3
WIDTH=3
HEIGHT=8
MS=2
USEMAX = False
TITLEFONT=7
TITLEPARAMS=3


opt_file = 'opt_stats_seen_tasks.csv'
del_file = 'opt_deltas.cvs'
uns_file = 'objs_unseen_tasks.csv'

PATH = './hpo-lin-reg-hd'
MINVAL, MAXVAL, STEP, STEP1 = 0.0, 2.0, 0.2, 0.04
LOGX, LOGY = False, True
XTEXT, YTEXT = 1000, 0.48
filter_list = [
]


parser = argparse.ArgumentParser()
parser.add_argument('--dpi', '-D', help='Image DPI', type=int, default=DPI)
parser.add_argument('--lw', '-L', help='Line width', type=int, default=LW)
parser.add_argument(
    '--alpha', '-A', help='Task line transperancy', type=float, default=ALPHA
)
parser.add_argument('--width', '-W', help='Per-image width', type=int, default=WIDTH)
parser.add_argument('--height', '-H', help='Per-image height', type=int, default=HEIGHT)
parser.add_argument('--msize', '-M', help='Marker size', type=int, default=MS)
parser.add_argument('--tfsize', '-f', help='Title font size', type=int, default=TITLEFONT)
parser.add_argument(
    '--path', '-P', help='Path to expt result directories', type=str, required=True
)
parser.add_argument(
    '--minyval', '-y', help='Minimum y-axis value', type=float, default=MINVAL
)
parser.add_argument(
    '--maxyval', '-Y', help='Maximum y-axis value', type=float, default=MAXVAL
)
parser.add_argument(
    '--ymajor', '-s', help='y-axis major step size', type=float, default=STEP
)
parser.add_argument(
    '--yminor', '-S', help='y-axis minor step size', type=float, default=STEP1
)
parser.add_argument('--xtext', '-t', help='x-val for text', default=XTEXT, type=float)
parser.add_argument('--ytext', '-T', help='y-val for text', default=YTEXT, type=float)
parser.add_argument('--xlog', help='Log scale for x-axis', action='store_true')
parser.add_argument('--ylog', help='Log scale for y-axis', action='store_true')
parser.add_argument(
    '--filter_list', '-F', help='comma-separated list of filtering terms', type=str,
    default=''
)
parser.add_argument('--include_lr', help='Incorporate LR in grad', action='store_true')
parser.add_argument('--small_range', help='Cover full range of nobjs or 4x ones', action='store_true')
parser.add_argument('--bnd', help='Plot theoretical bound', action='store_true')

args = parser.parse_args()

PATH = args.path
assert os.path.isdir(PATH)

if args.filter_list != '':
    filter_list = args.filter_list.split(',')
    logger.info(f'Filtering files based on:\n{filter_list}')

DPI=args.dpi
LW=args.lw
ALPHA=args.alpha
WIDTH=args.width
HEIGHT=args.height
MS=args.msize
USEMAX = False
TITLEFONT=args.tfsize
del_file = 'opt_deltas.csv'

MINVAL, MAXVAL, STEP, STEP1 = args.minyval, args.maxyval, args.ymajor, args.yminor
LOGX, LOGY = args.xlog, args.ylog
XTEXT, YTEXT = args.xtext, args.ytext
major_yticks = np.arange(MINVAL, MAXVAL+STEP, step=STEP)
minor_yticks = np.arange(MINVAL, MAXVAL+STEP1, step=STEP1)
configs = sorted([c.path for c in os.scandir(PATH) if c.is_dir()])
logger.info(f'Found {len(configs)} configs in {PATH}')
if len(filter_list) > 0:
    configs = [
        c for c in configs
        if all([f in c for f in filter_list])
    ]
logger.info(f'Filtered to {len(configs)} configs in {PATH}')
colors = [
    'g', 'y', 'r', 'm', 'k', 'b', 'c'
] if not args.small_range else [
    'g', 'r', 'c',
]
nobjs = [
    4, 9, 16, 25, 36, 49, 64
] if not args.small_range else [
    4, 16, 64
]
cdict = {f'n={n}': c for n, c in zip(nobjs, colors)}
METRIC = 'delta'
VAR = 'ALL'
XCOL = 'oiter'
agg_reps = {n: [] for n in cdict.keys()}
agg_reps['X'] = None
for c in tqdm(configs):
    minmax = ('M:True' in c)
    assert minmax
    matches = [n for n in nobjs if (f'a:{n}_' in c) and (f'A:0_' in c)]
    assert len(matches) <= 1, (f'Matches found for {c}:\n{matches}')
    if len(matches) == 0:
        logger.info(f'Skipping config {c}')
        continue
    nmatch = matches[0]
    nkey = f'n={nmatch}'
    logger.info(f'{nkey} matches in config {c}')
    # handle stats for seen tasks
    df = pd.read_csv(os.path.join(c, del_file))
    logger.debug(f'Read in {del_file} of size {df.shape}')
    for t, tdf in df.groupby(['task']):
        if t != 'ALL':
            continue
        if agg_reps['X'] is None:
            agg_reps['X'] = tdf['oiter'].values
        agg_reps[nkey] += [
            tdf['delta'].values/tdf['lr'].values
            if args.include_lr else tdf['delta'].values
        ] 

fig, ax = plt.subplots(
    1, 1, figsize=(WIDTH, HEIGHT), squeeze=True,
)
for k, v in agg_reps.items():
    if k == 'X':
        continue
    n = int(k.replace('n=', ''))
    XVALS = np.array(agg_reps['X'])
    curves = np.array(v)
    mid = np.percentile(curves, 50, axis=0)
    hi = np.percentile(curves, 75, axis=0)
    lo = np.percentile(curves, 25, axis=0)
    logger.info(f"[{k} -> {n}] Full {curves.shape}, Aggs: {mid.shape}, X: {XVALS.shape}")
    ax.plot(
        XVALS, mid, c=cdict[k], ls='-', linewidth=0.5*LW, **{'alpha': ALPHA}
    )
    ax.fill_between(
        XVALS, lo, hi, **{'alpha': ALPHA/3}, color=cdict[k], label=k,
    )
    bnd = np.array([np.max(mid) * np.sqrt(n) * np.power(float(t), -0.4) for t in XVALS])
    if args.bnd:
        ax.plot(XVALS, bnd, c=cdict[k], ls=':', linewidth=LW)
        ax.text(
            0.8 * XVALS[-1],
            bnd[-1]*1.1,
            f'O({np.sqrt(n).astype(int)}' + r'$K^{-2/5}$' + ')',
            fontsize=TITLEFONT
        )

if LOGX:
    ax.set_xscale('log')
if LOGY:
    ax.set_yscale('log', base=2)
major_yticks = np.power(2, np.arange(1, 7, step=1))
ax.set_yticks(major_yticks)
ax.grid(axis='both', which='major', alpha=0.2)
ax.legend(
    loc='upper right',
    ncol=1 if args.small_range else 2,
    fontsize=1.4*TITLEFONT,
    # title="# obj pairs",
)
ax.set_xlabel(
    'Iterations ' + r'$K$',
    fontsize=2*TITLEFONT
)
ax.set_ylabel(
    'Gradient norm: ' + r'$\| \nabla_x \|_2$',
    fontsize=2*TITLEFONT
)
plt.tight_layout()
prefix = os.path.join(PATH, f"nobjs_aggregate_results")
fig.savefig(f'{prefix}.png', dpi=DPI)
