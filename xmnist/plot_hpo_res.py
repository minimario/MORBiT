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
logger = logging.getLogger('PLOT-HPO-RES')
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

# PATH = './svm-hpo-res'
# PATH = './svm-rep-hpo'
# PATH = './hpo-v2'
# PATH = './hpo-lin-hd'
PATH = './hpo-lin-reg-hd'
# PATH = './hpo-hd-klr'
MINVAL, MAXVAL, STEP, STEP1 = 0.0, 2.0, 0.2, 0.04
LOGX, LOGY = False, True
XTEXT, YTEXT = 1000, 0.48
filter_list = [
    # 'd:FashionMNIST',
    #'d:Letter',
    #'T:4_',
    #'D:100_',
    #'I:4_',
    #'z:100',
    #'E:1_',
    #'O:50000_',
    #'z:20_',
    #'N:100_',
    #'X:1000',
]


parser = argparse.ArgumentParser()
parser.add_argument('--dpi', '-D', help='Image DPI', type=int, default=DPI)
parser.add_argument('--lw', '-L', help='Line width', type=int, default=LW)
parser.add_argument(
    '--alpha', '-A', help='Task line transperancy', type=int, default=ALPHA
)
parser.add_argument('--width', '-W', help='Per-image width', type=int, default=WIDTH)
parser.add_argument('--height', '-H', help='Per-image height', type=int, default=HEIGHT)
parser.add_argument('--msize', '-M', help='Marker size', type=int, default=MS)
parser.add_argument('--tfsize', '-f', help='Title font size', type=int, default=TITLEFONT)
parser.add_argument(
    '--tpwidth', '-w', help='Number of expt params per line', type=int, default=TITLEPARAMS
)
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
parser.add_argument(
    '--skip_unseen',
    help='Skip plotting of unseen task test error',
    action='store_true'
)
parser.add_argument('--agg', help='Aggregate per method curves', action='store_true')


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
TITLEPARAMS=args.tpwidth
PAGG = args.agg

opt_file = 'opt_stats_seen_tasks.csv'
del_file = 'opt_deltas.cvs'
uns_file = 'objs_unseen_tasks.csv'

# PATH = './svm-hpo-res'
# PATH = './svm-rep-hpo'
# PATH = './hpo-v2'
# PATH = './hpo-lin-hd'
# PATH = './hpo-lin-reg-hd'
# PATH = './hpo-hd-klr'
MINVAL, MAXVAL, STEP, STEP1 = args.minyval, args.maxyval, args.ymajor, args.yminor
LOGX, LOGY = args.xlog, args.ylog
XTEXT, YTEXT = args.xtext, args.ytext


def plot_test_error(df, color, ax, ylab, label_suffix='', pagg=False, mm=None, reps_dict=None):
    per_task_ys = []
    per_task_xs = []
    for t, tdf in df.groupby(['task']):
        xvals = tdf['oiter'].values
        yvals = np.minimum.accumulate(tdf[ylab].values)
        logger.debug(f' ... task {t} ... X: {xvals.shape} ... Y: {yvals.shape}')
        per_task_ys += [yvals]
        per_task_xs += [xvals]
        if not pagg:
            ax.plot(
                xvals, yvals, c=color, ls=':', linewidth=LW,
                **{'alpha': ALPHA/4}, label='_nolegend_'
            )
    task_mean_ys = np.mean(np.array(per_task_ys), axis=0)
    per_task_bests = np.min(np.array(per_task_ys), axis=1)
    task_max_ys = np.max(np.array(per_task_ys), axis=0)
    assert len(task_max_ys) == len(per_task_ys[0]), f'Task max shape: {task_max_ys.shape}'
    assert len(task_mean_ys) == len(per_task_ys[0]), f'Task mean shape: {task_mean_ys.shape}'
    assert np.sum(np.std(np.array(per_task_xs), axis=0)) == 0.0, (
        f'task x STDs: {np.std(np.array(per_task_xs), axis=0)}'
    )
    if pagg:
        assert mm is not None
        assert isinstance(mm, bool)
        assert mm == True or mm == False
        reps_dict['U-MEAN-f-te'][mm] += [task_mean_ys]
        reps_dict['U-MAX-f-te'][mm] += [task_max_ys]
    else:
        ax.plot(
            xvals, task_mean_ys, c=color, ls='-',
            linewidth=LW, label=f'U-MEAN-f-te{label_suffix}',
            **{'alpha': ALPHA}
        )
        ax.plot(
            xvals, task_max_ys, c=color, ls='-',
            linewidth=1.5*LW, label=f'U-MAX-f-te{label_suffix}',
        )
        ax.text(
            0.6 * xvals[-1], 1.1 * task_max_ys[-1],
            f'te:{task_max_ys[-1]:.3f} ({np.min(task_max_ys):.3f})',
            fontsize=TITLEFONT,
            c=color,
        )
        logger.info(f'Unseen {label_suffix}:\n{per_task_bests}')

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
# print(configs)
nminmax = np.sum([('M:True' in c) for c in configs])
navg = np.sum([('M:False' in c) for c in configs])
logger.info(f' -- Minmax: {nminmax}, Avg: {navg}')
assert nminmax + navg == len(configs)
ncols = max(nminmax, navg, 2)
nrows = 2
fig, axs = plt.subplots(
    nrows, ncols, figsize=(WIDTH*ncols, HEIGHT), sharex=True, sharey=True,
    #squeeze=True,
) if not PAGG else (None, None)
cidxs = [0, 0]
colors = {
    'f-va-obj': 'r',
    'f-te-obj': 'b',
}
colors1 = ['g', 'y', 'k', 'c']

fig_agg, axs_agg = None, None
cagg = {True: 'r', False: 'b'}
agg_reps = {
    'X': None,
    'S-MEAN-f-va': {True: [], False: []},
    'S-MAX-f-va': {True: [], False: []},
    'S-MEAN-f-te': {True: [], False: []},
    'S-MAX-f-te': {True: [], False: []},
    'U-MEAN-f-te': {True: [], False: []},
    'U-MAX-f-te': {True: [], False: []},
} if PAGG else None
if PAGG:
    agg_cols = [[
        'S-MEAN-f-va-ci',
        'S-MEAN-f-va-iqr',
        'S-MAX-f-va-ci',
        'S-MAX-f-va-iqr',
    ],[
        'S-MEAN-f-te-ci',
        'S-MEAN-f-te-iqr',
        'S-MAX-f-te-ci',
        'S-MAX-f-te-iqr',
    ],[
        'U-MEAN-f-te-ci',
        'U-MEAN-f-te-iqr',
        'U-MAX-f-te-ci',
        'U-MAX-f-te-iqr',
    ]]
    fig_agg, axs_agg = plt.subplots(
        3, 4, figsize=(WIDTH*3, HEIGHT), sharex=True, sharey=True,
        #squeeze=True,
    )

METRICS = list(colors.keys())
for c in tqdm(configs):
    minmax = ('M:True' in c)
    ridx = int(minmax)
    cidx = cidxs[ridx]
    ax = axs[ridx, cidx] if not PAGG else None
    cstr = c.split('/')[-1]
    cparams = cstr.split('_')
    title = ''
    sidx = 0
    while sidx < len(cparams):
        eidx = min(sidx + TITLEPARAMS, len(cparams))
        title += (', '.join(cparams[sidx:eidx]))
        sidx += TITLEPARAMS
        if sidx < len(cparams):
            title += '\n'
    logger.debug(f'Plot title:\n{title}')
    logger.info(f"Config: {c.replace('_', ' ')}")

    # handle stats for seen tasks
    df = pd.read_csv(os.path.join(c, opt_file))
    logger.debug(f'Read in {opt_file} of size {df.shape}')
    for b, bdf in df.groupby(['batch']):
        if b: continue
        for m in METRICS:
            per_task_ys = []
            per_task_xs = []
            for t, tdf in bdf.groupby(['task']):
                xvals = tdf['oiter'].values
                yvals = tdf[m].values
                logger.debug(f'... task {t} ... X: {xvals.shape} ... Y: {yvals.shape}')
                per_task_ys += [yvals]
                per_task_xs += [xvals]
                if not PAGG:
                    ax.plot(
                        xvals, yvals, c=colors[m], ls=':', linewidth=LW,
                        **{'alpha': ALPHA/4}, label='_nolegend_'
                    )
            task_mean_ys = np.mean(np.array(per_task_ys), axis=0)
            per_task_bests = np.min(np.array(per_task_ys), axis=1)
            task_max_ys = np.max(np.array(per_task_ys), axis=0)
            assert len(task_max_ys) == len(per_task_ys[0]), (
                f'Task max shape: {task_max_ys.shape}'
            )
            assert len(task_mean_ys) == len(per_task_ys[0]), (
                f'Task mean shape: {task_mean_ys.shape}'
            )
            assert np.sum(np.std(np.array(per_task_xs), axis=0)) == 0.0, (
                f'task x STDs: {np.std(np.array(per_task_xs), axis=0)}'
            )
            if PAGG:
                if agg_reps['X'] is None:
                    agg_reps['X'] = xvals
                agg_reps['S-MEAN-' + m.replace('-obj', '')][minmax] += [task_mean_ys]
                agg_reps['S-MAX-' + m.replace('-obj', '')][minmax] += [task_max_ys]
            else:
                ax.plot(
                    xvals, task_mean_ys, c=colors[m], ls='-', linewidth=LW,
                    label='S-MEAN-' + m,
                    **{'alpha': ALPHA}
                )
                ax.plot(
                    xvals, task_max_ys, c=colors[m], ls='-', linewidth=1.5*LW,
                    label='S-MAX-' + m,
                )
                ax.text(
                    0.6 * xvals[-1], 1.1 * task_max_ys[-1],
                    (
                        f"{m.replace('f-', '').replace('-obj', '')}:{task_max_ys[-1]:.3f}"
                        + f' ({np.min(task_max_ys):.3f})'
                    ),
                    fontsize=TITLEFONT,
                    c=colors[m],
                )
            logger.info(f'SEEN-{m}\n{per_task_bests}')
    # handle stats for unseen tasks
    df = pd.read_csv(os.path.join(c, uns_file))
    logger.debug(f'Read in {uns_file} of size {df.shape}')
    color_idx = 0
    if len(df) > 0 and not args.skip_unseen:
        if 'ntrain' in list(df):
            for ntrain, ndf in df.groupby(['ntrain']):
                unseen_color = colors1[color_idx]
                logger.debug(f'... train size: {ntrain} ')
                plot_test_error(
                    ndf, unseen_color, ax, 'f-obj', f'-{ntrain}',
                    pagg=PAGG, mm=minmax, reps_dict=agg_reps
                )
                color_idx += 1
        else:
            plot_test_error(
                df, 'g', ax, 'f-obj',
                pagg=PAGG, mm=minmax, reps_dict=agg_reps
            )
    if not PAGG:
        ## ax.set_yticks(major_yticks)
        ## ax.set_yticks(minor_yticks, minor=True)
        if LOGX:
            ax.set_xscale('log')
        if LOGY:
            ax.set_yscale('log', base=2)
        ax.grid(axis='both', which='major', alpha=0.2)
        ax.grid(axis='y', which='minor', alpha=0.1)
        if cidx == 0 and ridx == 0:
            ax.legend(loc='lower left', ncol=4, bbox_to_anchor=(0, 1.07), fontsize=TITLEFONT)
        ax.set_title('MINMAX' if minmax else 'MINAVG')
        ax.text(XTEXT, YTEXT, title, fontsize=TITLEFONT)
        cidxs[ridx] += 1
        # break
if PAGG:
    for k, v in agg_reps.items():
        if k == 'X':
            continue
        for kk, vv in v.items():
            assert isinstance(kk, bool)
            nvv = np.array(vv).shape
            logger.info(f"{k} -- {'MINMAX' if kk else 'MINAVG'} -- {nvv}")
    XVALS = np.array(agg_reps['X'])
    for ridx, rnames in enumerate(agg_cols):
        for cidx, pname in enumerate(rnames):
            # find right key in agg_reps
            mkeys = [k for k in agg_reps.keys() if (k != 'X') and (k in pname)]
            assert len(mkeys) == 1, (f'Keys found {mkeys} for {pname}')
            k = mkeys[0]
            CI = (pname.replace(f'{k}-', '') == 'ci')
            logger.info(f"Plot name: {pname}, key found: {k}, {'CI' if CI else 'IQR'}")
            ax_agg = axs_agg[ridx, cidx]
            ax_agg.set_title(pname)
            for minmax, reps in agg_reps[k].items():
                ccol = cagg[minmax]
                curves = np.array(reps)
                hi, mid, lo = None, None, None
                if CI:
                    mid = np.mean(curves, axis=0)
                    std = np.std(curves, axis=0)
                    hi = mid + std
                    lo = np.clip(mid - std, 0, 100)
                else:
                    mid = np.percentile(curves, 50, axis=0)
                    hi = np.percentile(curves, 75, axis=0)
                    lo = np.percentile(curves, 25, axis=0)
                logger.info(f"- {'MINMAX' if minmax else 'MINAVG'}: Full {curves.shape}, Aggs: {mid.shape}")
                ax_agg.plot(
                    XVALS, mid, c=ccol, ls='-', linewidth=1.5*LW,
                    label='MINMAX' if minmax else 'MINAVG',
                )
                ax_agg.fill_between(
                    XVALS, lo, hi, **{'alpha': ALPHA}, color=ccol,
                    label='MINMAX' if minmax else 'MINAVG',
                )
            if LOGX:
                ax_agg.set_xscale('log')
            if LOGY:
                ax_agg.set_yscale('log', base=2)
            ax_agg.grid(axis='both', which='major', alpha=0.2)
            ax_agg.grid(axis='y', which='minor', alpha=0.1)
            if cidx == 0 and ridx == 0:
                ax_agg.legend(
                    ## loc='lower left',
                    ## bbox_to_anchor=(0, 1.07),
                    ## ncol=2,
                    loc='upper right',
                    fontsize=TITLEFONT
                )
    prefix = os.path.join(PATH, 'aggregate_results')
    fig_agg.savefig(f'{prefix}_opt.png', dpi=DPI)
else:
    # axs[0, 0].set_ylabel('Outer objective')
    # axs[1, -(ncols//2)].set_xlabel('Outer iterations')
    # fig.supxlabel('# Outer iterations')
    # plt.tight_layout()
    prefix = os.path.join(PATH, 'all_results')
    fig.savefig(f'{prefix}_opt.png', dpi=DPI)
