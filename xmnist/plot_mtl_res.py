from itertools import product
import sys
import os
from pathlib import Path
import warnings
warnings.simplefilter('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import pickle

import logging
logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger('PLOT-MTL-RES')
logger.setLevel(logging.INFO)


DPI=100
LW=1.0
ALPHA=0.2
WIDTH=3
HEIGHT=8
MS=2
USEMAX = False
TITLEFONT=7
TITLEPARAMS=3


def plot_test_error(df, color, ax, ylab, label_suffix=''):
    per_task_ys = []
    per_task_xs = []
    for t, tdf in df.groupby(['task']):
        xvals = tdf['oiter'].values
        yvals = np.minimum.accumulate(tdf[ylab].values)
        logger.debug(f' ... task {t} ... X: {xvals.shape} ... Y: {yvals.shape}')
        per_task_ys += [yvals]
        per_task_xs += [xvals]
        ax.plot(
            xvals, yvals, c=color, ls=':', linewidth=LW,
            **{'alpha': ALPHA}, label='_nolegend_'
        )
    task_mean_ys = np.mean(np.array(per_task_ys), axis=0)
    task_max_ys = np.max(np.array(per_task_ys), axis=0)
    assert len(task_max_ys) == len(per_task_ys[0]), f'Task max shape: {task_max_ys.shape}'
    assert len(task_mean_ys) == len(per_task_ys[0]), f'Task mean shape: {task_mean_ys.shape}'
    assert np.sum(np.std(np.array(per_task_xs), axis=0)) == 0.0, (
        f'task x STDs: {np.std(np.array(per_task_xs), axis=0)}'
    )
    ax.plot(xvals, task_mean_ys, c=color, ls='--', linewidth=LW, label=f'U-MEAN-f-te{label_suffix}')
    ax.plot(xvals, task_max_ys, c=color, ls='-', linewidth=LW, label=f'U-MAX-f-te{label_suffix}')




opt_file = 'opt_stats_seen_tasks.csv'
del_file = 'opt_deltas.cvs'
uns_file = 'objs_unseen_tasks.csv'

PATH = './mtl-res'
configs = sorted([c.path for c in os.scandir(PATH) if c.is_dir()])
logger.info(f'Found {len(configs)} configs in {PATH}')
filter_list = [
    'T:10',
]
if len(filter_list) > 0:
    configs = [
        c for c in configs
        if all([f in c for f in filter_list])
    ]
logger.info(f'Filtered to {len(configs)} configs in {PATH}')
nminmax = np.sum([('M:True' in c) for c in configs])
navg = np.sum([('M:False' in c) for c in configs])
logger.info(f' -- Minmax: {nminmax}, Avg: {navg}')
assert nminmax + navg == len(configs)
ncols = max(nminmax, navg, 2)
nrows = 2
fig, axs = plt.subplots(
    nrows, ncols, figsize=(WIDTH*ncols, HEIGHT), sharex=True, sharey=True,
    squeeze=True,
)
cidxs = [0, 0]
colors = {
    'f-va-obj': 'r',
    'f-te-obj': 'b',
}
colors1 = ['g', 'y', 'k']

METRICS = list(colors.keys())


for c in tqdm(configs):
    minmax = ('M:True' in c)
    ridx = int(minmax)
    cidx = cidxs[ridx]
    ax = axs[ridx, cidx]
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
                ax.plot(
                    xvals, yvals, c=colors[m], ls=':', linewidth=LW,
                    **{'alpha': ALPHA}, label='_nolegend_'
                )
            task_mean_ys = np.mean(np.array(per_task_ys), axis=0)
            task_max_ys = np.max(np.array(per_task_ys), axis=0)
            assert len(task_max_ys) == len(per_task_ys[0]), f'Task max shape: {task_max_ys.shape}'
            assert len(task_mean_ys) == len(per_task_ys[0]), f'Task mean shape: {task_mean_ys.shape}'
            assert np.sum(np.std(np.array(per_task_xs), axis=0)) == 0.0, (
                f'task x STDs: {np.std(np.array(per_task_xs), axis=0)}'
            )
            ax.plot(xvals, task_mean_ys, c=colors[m], ls='--', linewidth=LW, label='S-MEAN-' + m)
            ax.plot(xvals, task_max_ys, c=colors[m], ls='-', linewidth=LW, label='S-MAX-' + m)
    # handle stats for unseen tasks
    df = pd.read_csv(os.path.join(c, uns_file))
    logger.debug(f'Read in {uns_file} of size {df.shape}')
    color_idx = 0
    if 'ntrain' in list(df):
        for ntrain, ndf in df.groupby(['ntrain']):
            unseen_color = colors1[color_idx]
            logger.debug(f'... train size: {ntrain} ')
            plot_test_error(ndf, unseen_color, ax, 'f-obj', f'-{ntrain}')
            color_idx += 1
    else:
        plot_test_error(df, 'g', ax, 'f-obj')
    ax.grid(axis='both')
    if cidx == 0 and ridx == 1:
        ax.legend(loc='upper left', ncol=2, bbox_to_anchor=(0, -0.1), fontsize=TITLEFONT)
    ax.set_title((r'$\bf{MINMAX}$' if minmax else r'$\bf{AVG}$') + 'OBJ')
    ax.text(250, 0.6, title, fontsize=TITLEFONT)
    cidxs[ridx] += 1
    # break
axs[0, 0].set_ylabel('Outer objective')
axs[1, -(ncols//2)].set_xlabel('Outer iterations')
prefix = os.path.join(PATH, 'all_results')
fig.savefig(f'{prefix}_opt.png', dpi=DPI)
