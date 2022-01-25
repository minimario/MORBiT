import numpy as np
np.set_printoptions(precision=4)
import torch
torch.set_printoptions(precision=4)
from torch import nn
import pandas as pd
pd.set_option('display.precision', 4)

import argparse
import os
import sys
import random

from utils import get_data, get_task_data, wnorm

import logging
logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger('MTL-BASE')
logger.setLevel(logging.INFO)


def run_base_opt(
        TASK,
        TASK_DATA,
        RNG,
        IN_DIM,
        INT_DIM,
        NCLASSES,
        BATCH_SIZE=32,
        BATCH_SIZE_OUT=128,
        LRATE=0.001,
        LRATE_OUT=0.0001,
        LR_DECAY=0.8,
        IN_ITER=1,
        OUT_ITER=1000,
        NLIN=False,
        INREG=0.001,
        OUTREG=0.0001,
        INNORM=2,
        OUTNORM=2,
        FULL_STATS_PER_ITER=10,
        TOL=1e-7,
):

    loss = nn.CrossEntropyLoss()

    # OUTER LEVEL VARS
    outer_var = nn.Linear(IN_DIM, INT_DIM, bias=False)
    out_opt = torch.optim.SGD(
        outer_var.parameters(),
        lr=LRATE_OUT,
        momentum=0.9,
        weight_decay=1e-4,
    )
    out_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        out_opt, 'min', factor=LR_DECAY, verbose=True,
        patience=20,
    )

    # INNER LEVEL VARS
    inner_var = nn.Linear(INT_DIM, NCLASSES)
    topt = torch.optim.SGD(
        inner_var.parameters(),
        lr=LRATE,
        momentum=0.9,
        weight_decay=1e-4,
    )
    tsched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        topt, 'min', factor=LR_DECAY, verbose=True,
        patience=20,
    )

    # Functions used by all tasks and level -- not optimized
    in_softmax = nn.Softmax(dim=1)
    out_nlin = nn.ReLU()

    # Setting up stats to be collected

    # statistics for tasks seen during optimization
    all_stats_col = [
        'oiter', # outer iter
        'batch', # batch stats or stats on full data
        'task', # task ID
        'g-loss', # inner level loss (full or batch)
        'g-obj', # inner level obj (full or batch)
        'f-va-loss', # outer level loss (full or batch)
        'f-va-obj', # outer level obj (full or batch)
        'f-te-loss', # outer level loss on unseen data (full); when batch is NA
        'f-te-obj', # outer level obj on unseen data (full); when batch is NA
    ]
    all_stats = []

    # statistics for change in variables during optimization for seen tasks
    delta_stats_col = [
        'oiter', # outer iter
        'task', # task ID for inner_var or ALL for outer_var or SIMP for simplex_vars
        'delta', # change in variables
        'lr', # current learning rate for variable
    ]
    delta_stats = []

    bout_loss = np.inf
    bout_obj = np.inf
    bl_iter = 0
    bo_iter = 0

    prefix = f'{TASK}'
    t = TASK
    tdata = TASK_DATA
    tw = inner_var
    for oi in range(OUT_ITER):
        logger.info(f'Starting outer loop {oi+1}/{OUT_ITER} ...')
        batch_stats = ((oi+1) % FULL_STATS_PER_ITER != 0)
        out_opt.zero_grad()
        outer_loss = 0.
        out_acc = 0.
        in_loss = 0.
        in_obj = 0.
        in_delta = 0.
        outer_old = outer_var.weight.clone()
        ppr = f'{prefix}->[{oi+1}/{OUT_ITER}]'
        tsize = tdata['train-size']
        X, y = tdata['train']
        total_loss = 0.
        total_obj = 0.
        inner_old = tw.weight.clone()
        for ii in range(IN_ITER):
            topt.zero_grad()
            bidxs = [RNG.randint(0, tsize) for _ in range(BATCH_SIZE)]
            bX = X[bidxs]
            by = y[bidxs]
            int_rep = out_nlin(outer_var(bX)) if NLIN else outer_var(bX)
            out = tw(int_rep)
            probs = in_softmax(out)
            btloss = loss(probs, by)
            total_loss += btloss.item()
            if INREG > 0.0:
                btloss += (INREG * wnorm(tw, INNORM))
            btloss.backward(retain_graph=True)
            topt.step()
            total_obj += btloss.item()
        total_loss /= IN_ITER
        total_obj /= IN_ITER
        in_loss = total_loss
        in_obj = total_obj
        # Compute tasks specific outer level loss on batch of val data
        vX, vy = tdata['val']
        vsize = tdata['val-size']
        bidxs = [RNG.randint(0, vsize) for _ in range(BATCH_SIZE_OUT)]
        bX = vX[bidxs]
        by = vy[bidxs]
        out = in_softmax(tw(out_nlin(outer_var(bX)))) if NLIN else in_softmax(tw(outer_var(bX)))
        toloss = loss(out, by)
        toobj = toloss.item()
        if OUTREG > 0.0:
            toobj += (OUTREG * wnorm(outer_var, OUTNORM)).item()
        # Computing outer level accuracy on batch
        _, preds = torch.max(out, 1)
        out_acc = (by == preds).sum() / bX.size(0)
        # collect opt stats
        all_stats += [(
            oi+1, True, t,
            total_loss, total_obj, # g-stuff
            toloss.item(), toobj, # f-val-stuff
            np.nan, np.nan #f-te-stuff
        )]
        outer_loss = toloss
        # collect delta stats
        current_lr = topt.param_groups[0]['lr']
        with torch.no_grad():
            in_delta = torch.linalg.norm(tw.weight - inner_old)
        delta_stats += [(oi+1, f'{t}-IN', in_delta.item(), current_lr)]
        # print some stats
        if not batch_stats:
            logger.info(f'{ppr} IN LOSSES: {np.array([in_loss])}')
            logger.info(f'{ppr} IN OBJS: {np.array([in_obj])}')
            logger.info(f'{ppr} IN-Deltas: {np.array([in_delta])}')
            logger.info(f'{ppr} OUT LOSS: {np.array([outer_loss.item()])}')
            logger.info(f'{ppr} OUT ACCS: {np.array([out_acc])}')
        logger.info(
            f'{ppr} OUTER LOSS: {outer_loss.item():.8f} '
            f'(best: {bout_loss:.8f} ({bl_iter}/{OUT_ITER}))'
        )
        if outer_loss.item() < bout_loss:
            logger.info(f'{ppr} OUT: {np.array([outer_loss.item()])}')
            bout_loss = outer_loss.item()
            bl_iter = oi + 1

        # Add regularization and take opt step
        if OUTREG > 0.0:
            outer_loss += (OUTREG * wnorm(outer_var, OUTNORM))
        if not batch_stats:
            logger.info(
                f'{ppr} OUTER LOSS + REG: {outer_loss:.8f} '
                f'(best: {bout_obj:.8f} ({bo_iter}/{OUT_ITER}))'
            )
        if outer_loss.item() < bout_obj:
            bout_obj = outer_loss.item()
            bo_iter = oi + 1
        outer_loss.backward(retain_graph=True)
        out_opt.step()
        # save delta stats
        with torch.no_grad():
            out_delta = torch.linalg.norm(outer_var.weight - outer_old)
        curr_lr = out_opt.param_groups[0]['lr']
        logger.info(f'[{ppr}] OUT Delta: {out_delta.item():.6f}')
        delta_stats += [(oi+1, f'{t}-OUT', out_delta.item(), curr_lr)]

        converged = (in_delta < TOL) and (out_delta < TOL)
        if (not batch_stats) or (converged):
            # compute stats on full data
            voobjs, toobjs = [], []
            logger.info(f'{ppr} Task {t} full-stats comp ....')
            X, y = tdata['train']
            vX, vy = tdata['val']
            tX, ty = tdata['test']
            with torch.no_grad():
                train_out = in_softmax(tw(out_nlin(outer_var(X)))) if NLIN else in_softmax(tw(outer_var(X)))
                train_loss = loss(train_out, y)
                train_obj = train_loss.item()
                if INREG > 0.0:
                    train_obj += (INREG * wnorm(tw, INNORM)).item()
                val_out = in_softmax(tw(out_nlin(outer_var(vX)))) if NLIN else in_softmax(tw(outer_var(vX)))
                val_loss = loss(val_out, vy)
                val_obj = val_loss.item()
                if OUTREG > 0.0:
                    val_obj += (OUTREG * wnorm(outer_var, OUTNORM)).item()
                test_out = in_softmax(tw(out_nlin(outer_var(tX)))) if NLIN else in_softmax(tw(outer_var(tX)))
                test_loss = loss(test_out, ty)
                test_obj = test_loss.item()
                if OUTREG > 0.0:
                    test_obj += (OUTREG * wnorm(outer_var, OUTNORM)).item()
                all_stats += [(
                    oi+1, False, t,
                    train_loss.item(), train_obj, # g-stuff
                    val_loss.item(), val_obj, # f-val-stuff
                    test_loss.item(), test_obj, # f-te-stuff
                )]
                voobjs += [val_obj]
                toobjs += [test_obj]
            # invoking lr scheduler for inner level optimization
            tsched.step(test_obj)
            logger.info(f'[{ppr}] Full outer stats:')
            logger.info(f'[{ppr}] val: {np.array(voobjs)}')
            logger.info(f'[{ppr}] test: {np.array(toobjs)}')
            all_test_objs = np.sum(toobjs)
            # invoking lr scheduler for outer level optimization
            out_sched.step(all_test_objs)
        if converged:
            logger.warning(
                f"[{ppr}] Exiting optimization with sum(IN-DELTAs): {in_delta:.8f}, "
                f"OUT-DELTA: {out_delta:.8f}"
            )
            break

    logger.info(
        f'Best loss: {bout_loss:.5f} ({bl_iter}/{OUT_ITER}), Best obj: {bout_obj:.5f} ({bo_iter}/{OUT_ITER})'
    )
    all_stats_df = pd.DataFrame(all_stats, columns=all_stats_col)
    delta_stats_df = pd.DataFrame(delta_stats, columns=delta_stats_col)
    return all_stats_df, delta_stats_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', choices=['MNIST', 'FashionMNIST'], help='Data set to use')
    parser.add_argument(
        '--path_to_data', '-Z', type=str, default='/home/pari/data/torchvision/', help='Path to load/save data'
    )
    parser.add_argument('--int_dim', '-D', type=int, default=32, help='Latent space dimensionality')
    parser.add_argument(
        '--in_lrate', '-L', type=float, default=0.001, help='Initial learning rate for inner level'
    )
    parser.add_argument(
        '--out_lrate', '-l', type=float, default=0.0001, help='Initial learning rate for outer level'
    )
    parser.add_argument(
        '--lr_decay', '-y', type=float, default=0.5, help='Factor to reduce learning rate by'
    )
    parser.add_argument('--inner_loop', '-I', type=int, default=1, help='Number of inner level iterations')
    parser.add_argument('--max_outer_loop', '-O', type=int, default=100, help='Max outer level iters')
    parser.add_argument('--inner_reg', '-R', type=float, default=0.001, help='Reg. for inner level')
    parser.add_argument('--outer_reg', '-r', type=float, default=0.0001, help='Reg. for outer level')
    parser.add_argument('--inner_p', '-P', type=int, default=2, help='Norm order for Reg. for inner level')
    parser.add_argument('--outer_p', '-p', type=int, default=2, help='Norm order for Reg. for outer level')
    parser.add_argument('--inner_batch_size', '-B', type=int, default=32, help='Batch size for inner level')
    parser.add_argument('--outer_batch_size', '-b', type=int, default=128, help='Batch size for outer level')
    parser.add_argument('--nonlinear', '-N', action='store_true', help='Nonlinear version')
    parser.add_argument('--random_seed', '-S', type=int, default=5489, help='Random seed for RNG')
    parser.add_argument(
        '--full_stats_per_iter', '-F', type=int, default=10, help='Save full stats every this iters'
    )
    parser.add_argument('--tolerance', '-x', type=float, default=1e-7, help='Tolerance of optimization')
    parser.add_argument('--output_dir', '-U', type=str, default='', help='Directory to save results in')

    ostrings = [
        s.replace('-', '')
        for s in list(parser.__dict__['_option_string_actions'].keys())
    ]
    assert len(ostrings) % 2 == 0
    odict = {
        ostrings[2*i]: ostrings[2*i + 1]
        for i in range(len(ostrings) // 2)
        if ostrings[2*i] not in [
                'path_to_data',
                'output_dir',
                'help', 'h'
        ]
    }
    args = parser.parse_args()
    expt_tag = '_'.join([
        f'{odict[k]}:{args.__dict__[k]}'
        for k in sorted(args.__dict__.keys())
        if k in odict
    ])
    logger.info(f'Experiment tag: {expt_tag}')

    assert os.path.isdir(args.path_to_data)
    assert os.path.isdir(args.output_dir) or args.output_dir == ''
    assert args.int_dim > 10
    assert args.in_lrate > 0.
    assert args.out_lrate > 0.
    assert 0. < args.lr_decay < 1.
    assert args.inner_loop > 1
    assert args.max_outer_loop > 1
    assert args.inner_p in [1, 2] and args.outer_p in [1, 2]
    assert args.inner_batch_size > 1 and args.outer_batch_size > 1
    assert args.full_stats_per_iter > 1
    assert args.tolerance > 0.

    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    full_data = get_data(args.data, args.path_to_data)
    input_dim = full_data['train'].data.shape
    all_labels = np.unique(full_data['train'].targets.numpy())
    logger.info(f"Full data with {input_dim} features and {len(all_labels)} classes")
    logger.info(f"- Classes: {all_labels}")
    all_tasks = [
        (all_labels[i], all_labels[j])
        for i in range(len(all_labels)) for j in range(i + 1, len(all_labels))
    ]
    logger.info(f'Total of {len(all_tasks)} binary classification tasks')
    NCLASSES = 2
    logger.info(f"Performing independent optimization with the following tasks:")
    logger.info(f"- Tasks:\n{all_tasks}")
    task_data = [get_task_data(full_data, t, val=True) for t in all_tasks]
    orig_dim = task_data[0]['train'][0].shape[1]
    logger.info(f"Starting with original input dim: {orig_dim} ...")

    output_dir = os.path.join(args.output_dir, expt_tag)
    if args.output_dir != '':
        assert not os.path.exists(output_dir)

    all_astats, all_dstats = [], []
    for t, tdata in zip(all_tasks, task_data):
        X, _ = tdata['train']
        max_X = torch.max(X).item()
        X /= max_X
        tX, _ = tdata['test']
        tX /= max_X
        if 'val' in tdata:
            vX, _ = tdata['val']
            vX /= max_X

        RNG = np.random.RandomState(args.random_seed)
        torch.manual_seed(args.random_seed)
        astats, dstats = run_base_opt(
            TASK=t,
            TASK_DATA=tdata,
            RNG=RNG,
            IN_DIM=orig_dim,
            INT_DIM=args.int_dim,
            NCLASSES=NCLASSES,
            BATCH_SIZE=args.inner_batch_size,
            BATCH_SIZE_OUT=args.outer_batch_size,
            LRATE=args.in_lrate,
            LRATE_OUT=args.out_lrate,
            LR_DECAY=args.lr_decay,
            IN_ITER=args.inner_loop,
            OUT_ITER=args.max_outer_loop,
            NLIN=args.nonlinear,
            INREG=args.inner_reg,
            OUTREG=args.outer_reg,
            INNORM=args.inner_p,
            OUTNORM=args.outer_p,
            FULL_STATS_PER_ITER=args.full_stats_per_iter,
            TOL=args.tolerance,
        )
        all_astats += [astats]
        all_dstats += [dstats]

    astats = pd.concat(all_astats)
    dstats = pd.concat(all_dstats)

    logger.info(f"Results for tasks: {astats['task'].unique()}")
    print('All stats:')
    print(astats.head(10))
    print(astats.tail(10))
    print('Delta stats:')
    print(dstats.head(10))
    print(dstats.tail(10))

    tlist, vlist = [], []
    for t, tdf in astats.groupby(['task']):
        best_val_obj = np.min(tdf['f-va-obj'])
        best_val_loss = np.min(tdf['f-va-loss'])
        best_test_obj = np.min(tdf['f-te-obj'])
        best_test_loss = np.min(tdf['f-te-loss'])
        logger.info(f'Task: {t}')
        logger.info(f'- Validation: {best_val_loss:.4f} ({best_val_obj:.4f})')
        logger.info(f'- Test      : {best_test_loss:.4f} ({best_test_obj:.4f})')
        tlist += [t]
        vlist += [best_val_obj]

    hard_tasks = np.argsort(vlist)[-3:]
    logger.info(f'Hard tasks: (avg: {np.mean(vlist):.4f} +- {np.std(vlist):.4f})\n ALL:{np.array(vlist)}')
    for hi in hard_tasks:
        logger.info(f' - {tlist[hi]}: {vlist[hi]:.4f}')

    if args.output_dir != '':
        os.mkdir(output_dir)
        logger.info(f'Saving results in {output_dir} ...')
        dfs = [astats, dstats]
        fnames = [
            'opt_stats_seen_tasks.csv',
            'opt_deltas.csv',
        ]
        for d, f in zip(dfs, fnames):
            logger.info(f'- saving {f}')
            d.to_csv(os.path.join(output_dir, f), header=True, index=False)
