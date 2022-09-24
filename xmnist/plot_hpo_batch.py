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
logger = logging.getLogger('PLOT-HPO-BATCH')
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
parser.add_argument('--agg_interval', help='Aggregate per method curves', action='store_true')
parser.add_argument('--small_range', help='Cover full range of nobjs or 4x ones', action='store_true')

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
opt_file = 'opt_stats_seen_tasks.csv'

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
colors = ['r', 'g', 'y', 'c', 'k'] if not args.small_range else ['r', 'g', 'k']
bsizes = [8, 16, 32, 64, 128] if not args.small_range else [8, 32, 128]
cdict = {f'b={b}': c for b, c in zip(bsizes, colors)}
METRIC = 'f-te-obj'
agg_reps = {b: [] for b in cdict.keys()}
agg_reps['X'] = None
for c in tqdm(configs):
    minmax = ('M:True' in c)
    assert minmax
    matches = [b for b in bsizes if (f'B:{b}_' in c) and (f'b:{b}_' in c)]
    assert len(matches) <= 1, (f'Matches found for {c}:\n{matches}')
    if len(matches) == 0:
        logger.info(f'Skipping config {c}')
        continue
    bmatch = matches[0]
    bkey = f'b={bmatch}'
    logger.info(f'{bkey} matches in config {c}')
    # handle stats for seen tasks
    df = pd.read_csv(os.path.join(c, opt_file))
    logger.debug(f'Read in {opt_file} of size {df.shape}')
    for b, bdf in df.groupby(['batch']):
        if b: continue
        per_task_ys = []
        per_task_xs = []
        for t, tdf in bdf.groupby(['task']):
            xvals = tdf['oiter'].values
            yvals = tdf[METRIC].values
            per_task_ys += [yvals]
            per_task_xs += [xvals]
        task_max_ys = np.max(np.array(per_task_ys), axis=0)
        assert len(task_max_ys) == len(per_task_ys[0]), (
            f'Task max shape: {task_max_ys.shape}'
        )
        assert np.sum(np.std(np.array(per_task_xs), axis=0)) == 0.0, (
            f'task x STDs: {np.std(np.array(per_task_xs), axis=0)}'
        )
        if agg_reps['X'] is None:
            agg_reps['X'] = xvals
        agg_reps[bkey] += [task_max_ys]


fig, ax = plt.subplots(
    1, 1, figsize=(WIDTH, HEIGHT), squeeze=True,
)
for k, v in agg_reps.items():
    if k == 'X':
        continue
    XVALS = np.array(agg_reps['X'])
    curves = np.array(v)
    mid = np.percentile(curves, 50, axis=0)
    hi = np.percentile(curves, 75, axis=0)
    lo = np.percentile(curves, 25, axis=0)
    logger.info(f"[{k}] Full {curves.shape}, Aggs: {mid.shape}, X: {XVALS.shape}")
    ax.plot(
        XVALS, mid, c=cdict[k], ls='-', linewidth=LW,
    )
    ax.fill_between(
        XVALS, lo, hi, **{'alpha': ALPHA}, color=cdict[k], label=k,
    )
if LOGX:
    ax.set_xscale('log')
if LOGY:
    ax.set_yscale('log', base=2)
    yticks = [-1.6, -1.4, -1.2, -1.0, -0.8, -0.6] # np.arange(-1.6, -0.8, step=0.4)
    major_yticks = np.power(2, yticks)
    ax.set_yticks(major_yticks)
    ylabels = [r'$2^{-1.6}$', r'$2^{-1.4}$', r'$2^{-1.2}$', r'$2^{-1.0}$', r'$2^{-0.8}$', r'$2^{-0.6}$']
    ax.set_yticklabels(ylabels)
ax.grid(axis='both', which='major', alpha=0.5)
# ax.grid(axis='y', which='minor', alpha=0.1)
ax.legend(
    loc='upper right',
    ncol=1 if args.small_range else 2,
    fontsize=1.5*TITLEFONT,
    title='Batch size $b$',
)
ax.set_xlabel(
    'Iterations ' + r'$K$',
    fontsize=2*TITLEFONT
)
ax.set_ylabel(
    # r'$\max_{i \in [n]}$' + ' ' + r'$f_i(x, y_i^*(x))$',
    'Worst-case gen. loss',
    fontsize=2*TITLEFONT
)
plt.tight_layout()
prefix = os.path.join(PATH, f"batchwise_aggregate_results")
fig.savefig(f'{prefix}.png', dpi=DPI)
