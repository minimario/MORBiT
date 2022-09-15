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

from utils import get_data, get_task_data, simplex_proj_inplace, wnorm, args2tag

import logging
logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger('LIN-REG-HD-HPO')
logger.setLevel(logging.INFO)

def reg(C, w, logspace=True):
    if logspace:
        return torch.dot(torch.exp(C), torch.square(w.view(-1)))
    else:
        return torch.dot(C, torch.square(w.view(-1)))

def run_hpo(
        TASKS,
        TASK_DATA,
        TTASKS,
        TTASK_DATA,
        RNG,
        IN_DIM,
        BATCH_SIZE=32,
        BATCH_SIZE_OUT=128,
        LRATE=0.001,
        LRATE_OUT=0.001,
        LRATE_SIMP=0.1,
        LR_DECAY=0.8,
        LR_PATIENCE=20,
        IN_ITER=1,
        OUT_ITER=1000,
        MINMAX=False,
        FULL_STATS_PER_ITER=10,
        TOL=1e-7,
        DELTA=0.1,
        LOGC=True,
        INIT_C=0.0,
        NOSLRS=False,
):

    # OUTER LEVEL VARS
    C = INIT_C * torch.ones(IN_DIM)
    C.requires_grad = True
    out_opt = torch.optim.SGD(
        [C],
        lr=LRATE_OUT,
        momentum=0.9,
    )
    out_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        out_opt, 'min', factor=LR_DECAY, verbose=True,
        patience=LR_PATIENCE,
    )

    # INNER LEVEL VARS
    ntasks = len(TASKS)
    inner_vars = [nn.Linear(IN_DIM, 1) for _ in TASKS]
    in_opts, in_scheds = [], []
    for iv in inner_vars:
        in_opt = torch.optim.SGD(
            iv.parameters(),
            lr=LRATE,
            momentum=0.9,
        )
        in_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            in_opt, 'min', factor=LR_DECAY, verbose=True,
            patience=LR_PATIENCE,
        )
        in_opts += [in_opt]
        in_scheds += [in_sched]

    # OPTIONAL: simplex vars
    simplex_vars, simplex_opt, simplex_sched = None, None, None
    simplex_vars = torch.Tensor([1./ntasks for i in range(ntasks)])
    if MINMAX:
        simplex_vars.requires_grad = True
        simplex_opt = torch.optim.SGD(
            [simplex_vars],
            lr=LRATE_SIMP,
            momentum=0.9,
        )
        simplex_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            simplex_opt, 'min', factor=LR_DECAY, verbose=True,
            patience=LR_PATIENCE,
        )


    # set up variables and optimizers for unseen tasks
    nttasks = len(TTASKS)
    t_inner_vars = [nn.Linear(IN_DIM, 1) for _ in TTASKS]
    t_in_opts, t_in_scheds = [], []
    for iv in t_inner_vars:
        in_opt = torch.optim.SGD(
            iv.parameters(),
            lr=LRATE,
            momentum=0.9,
        )
        in_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            in_opt, 'min', factor=LR_DECAY, verbose=True,
            patience=LR_PATIENCE,
        )
        t_in_opts += [in_opt]
        t_in_scheds += [in_sched]

    # Functions used by all tasks and level -- not optimized
    loss = nn.MSELoss()

    # Setting up stats to be collected

    # statistics for tasks seen during optimization
    all_stats_col = [
        'oiter', # outer iter
        'batch', # batch stats or stats on full data
        'task', # task ID
        'g-loss', # inner level loss (full or batch)
        'g-obj', # inner level obj (full or batch)
        'f-va-obj', # outer level obj (full or batch)
        'f-te-obj', # outer level obj on unseen data (full); when batch is NA
    ]
    all_stats = []

    # statistics for change in variables during optimization for seen tasks
    delta_stats_col = [
        'oiter', # outer iter
        # task ID for inner_var or ALL for outer_var or SIMP for simplex_vars
        'task',
        'delta', # change in variables
        'lr', # current learning rate for variable
        # weight for current task; when outer_var or simplex_vars is NA
        'lambda',
    ]
    delta_stats = []

    # statistics for tasks not seen during optimization
    tall_stats_col = [
        'oiter', # outer iter
        'task', # task ID
        'g-loss', # inner level loss at convergence (full)
        'g-obj', # inner level obj at convergence (full)
        'f-obj', # outer level obj at convergence (full)
    ]
    tall_stats = []

    bout_obj = np.inf
    bo_iter = 0
    bout_full_obj = [np.inf]*2
    bfo_iter = [0]*2

    for oi in range(OUT_ITER):
        logger.debug(f'Starting outer loop {oi+1}/{OUT_ITER} ...')
        ppr = f'{oi+1}/{OUT_ITER}'
        batch_stats = (
            ((oi+1) % FULL_STATS_PER_ITER != 0)
            or ((oi+1) == OUT_ITER)
        )
        out_opt.zero_grad()
        if MINMAX:
            simplex_opt.zero_grad()
            old_simplex_vars = simplex_vars.clone().detach()
        out_losses = []
        in_losses = []
        in_objs = []
        in_deltas = []
        outer_loss = 0.0
        outer_old = C.clone().detach()
        # Looping over all the tasks
        for t, tdata, tw, topt, tlambda in zip(
                TASKS, TASK_DATA, inner_vars, in_opts, simplex_vars,
        ):
            logger.debug(f'[{ppr}] Task {t} ....')
            tsize = tdata['train-size']
            X, y = tdata['train']
            total_loss = 0.
            total_obj = 0.
            inner_old = tw.weight.clone().detach()
            for ii in range(IN_ITER):
                topt.zero_grad()
                logger.debug(f'[{ppr}] [{t}] ({ii+1}/{IN_ITER}) Train size: {tsize}'),
                bidxs = [np.random.randint(0, tsize) for _ in range(BATCH_SIZE)]
                bX = X[bidxs]
                by = y[bidxs]
                preds = tw(bX)
                btloss = loss(by, preds)
                # tracking loss
                total_loss += btloss.item()
                btloss += reg(C, tw.weight, logspace=LOGC)
                btloss.backward(retain_graph=True)
                topt.step()
                # tracking obj
                total_obj += btloss.item()
            total_loss /= IN_ITER
            total_obj /= IN_ITER
            in_losses += [total_loss]
            in_objs += [total_obj]
            # Compute tasks specific outer level loss on batch of val data
            vX, vy = tdata['val']
            vsize = tdata['val-size']
            bidxs = [RNG.randint(0, vsize) for _ in range(BATCH_SIZE_OUT)]
            bX = vX[bidxs]
            by = vy[bidxs]
            preds = tw(bX)
            toloss = loss(by, preds)
            toobj = toloss.item()
            # collect opt stats
            all_stats += [(
                oi+1, True, t,
                total_loss, total_obj, # g-stuff
                toobj, # f-val-stuff
                np.nan #f-te-stuff
            )]
            outer_loss += (tlambda * toloss)
            out_losses += [toloss.item()]
            # collect delta stats
            current_lr = topt.param_groups[0]['lr']
            with torch.no_grad():
                in_delta = torch.linalg.norm(tw.weight - inner_old)
            in_deltas += [in_delta.item()]
            delta_stats += [
                (oi+1, t, in_delta.item(), current_lr, tlambda.item())
            ]
        # print some stats
        if not batch_stats:
            logger.debug(f'[{ppr}] IN LOSSES: {np.array(in_losses)}')
            logger.debug(f'[{ppr}] IN OBJS: {np.array(in_objs)}')
            logger.debug(f'[{ppr}] IN-Deltas: {np.array(in_deltas)}')
            logger.debug(f'[{ppr}] OUT LOSS: {np.array(out_losses)}')
        logger.debug(
            f'[{ppr}] OUTER BATCH LOSS: {outer_loss.item():.8f} '
            f'(best: {bout_obj:.8f} ({bo_iter}/{OUT_ITER}))'
        )
        if outer_loss.item() < bout_obj:
            bout_obj = outer_loss.item()
            bo_iter = oi + 1
        # take optimization step
        outer_loss.backward(retain_graph=True)
        term = 'logC' if LOGC else 'C'
        with torch.no_grad():
            if not batch_stats:
                logger.info(
                    f'[{ppr}] {term}: '
                    f'({torch.min(C):.2f},{torch.mean(C):.2f},'
                    f'{torch.max(C):.2f})'
                    f';  g-{term}: ('
                    f'{torch.min(C.grad):.3f}, '
                    f'{torch.mean(C.grad):.3f}, '
                    f'{torch.max(C.grad):.3f})'
                )
        out_opt.step()
        if not LOGC:
            with torch.no_grad():
                C.clamp_(min=0.0)
        # save delta stats
        with torch.no_grad():
            out_delta = torch.linalg.norm(C - outer_old)
        curr_lr = out_opt.param_groups[0]['lr']
        delta_stats += [(oi+1, 'ALL', out_delta.item(), curr_lr, np.nan)]
        if not batch_stats:
            logger.info(
                f'[{ppr}] Delta({term}): {out_delta:.6f}'
                f'; Current learning rate: {curr_lr:.6f}'
            )

        # Update simplex lambda if minmax
        logger.debug(f'[{ppr}] Lambdas: {simplex_vars}')
        if MINMAX:
            # negate gradient for gradient ascent
            simplex_vars.grad *= -1.
            logger.debug(
                f'[{ppr}] - g-Lambda: '
                f'{simplex_vars.grad.clone().detach().numpy()}'
            )
            simplex_opt.step()
            logger.debug(f'[{ppr}] U-Lambda: {simplex_vars}')
            simplex_proj_inplace(simplex_vars, z=1-DELTA)
            with torch.no_grad():
                simplex_vars += (DELTA/ntasks)
            if not batch_stats:
                logger.info(
                    f'[{ppr}] Sim-proj. U-Lambdas: '
                    f'{simplex_vars.clone().detach().numpy()}'
                )
            curr_lr = simplex_opt.param_groups[0]['lr']
            with torch.no_grad():
                lambda_delta = torch.linalg.norm(
                    simplex_vars - old_simplex_vars
                )
            if not batch_stats:
                logger.info(
                    f'[{ppr}] - lambda delta: {lambda_delta.item():.6f}, '
                    f'current LR: {curr_lr:.6f}'
                )
            delta_stats += [
                (oi+1, 'SIMP', lambda_delta.item(), curr_lr, np.nan)
            ]
            # update out_delta to include lambda delta
            out_delta += lambda_delta.item()
            assert simplex_vars.requires_grad

        in_delta_sum = np.sum(in_deltas)
        converged = (in_delta_sum < TOL) and (out_delta < TOL)
        if (not batch_stats) or (converged):
            # compute stats on full data
            voobjs, toobjs = [], []
            for t, tdata, tw, tsched in zip(
                    TASKS, TASK_DATA, inner_vars, in_scheds
            ):
                logger.info(f'[{ppr}] Task {t} full-stats comp ....')
                X, y = tdata['train']
                vX, vy = tdata['val']
                tX, ty = tdata['test']
                with torch.no_grad():
                    train_out = tw(X)
                    train_loss = loss(y, train_out)
                    train_obj = (
                        train_loss.item()
                        + reg(C, tw.weight, logspace=LOGC).item()
                    )
                    val_out = tw(vX)
                    val_loss = loss(vy, val_out)
                    test_out = tw(tX)
                    test_loss = loss(ty, test_out)
                    all_stats += [(
                        oi+1, False, t,
                        train_loss.item(), train_obj, # g-stuff
                        val_loss.item(), # f-val-stuff
                        test_loss.item(), # f-te-stuff
                    )]
                    voobjs += [val_loss.item()]
                    toobjs += [test_loss.item()]
                # invoking lr scheduler for inner level optimization
                tsched.step(test_loss)
            with torch.no_grad():
                all_test_objs = torch.sum(
                    simplex_vars * torch.Tensor(toobjs)
                ).item()
            logger.info(f'[{ppr}] Full outer stats:')
            logger.info(f'[{ppr}] val: {np.array(voobjs)}')
            logger.info(
                f'[{ppr}] test: {np.array(toobjs)} (w-sum: {all_test_objs:.4f})'
            )
            outer_objs = [
                np.mean(np.array(voobjs)),
                np.max(np.array(voobjs)),
            ]
            for i in [0, 1]:
                if outer_objs[i] < bout_full_obj[i]:
                    bout_full_obj[i] = outer_objs[i]
                    bfo_iter[i] = oi + 1
            logger.info(f'[{ppr}] OUTER FULL LOSS:')
            for i, s in zip([0, 1], ['mean', 'max']):
                logger.info(
                    f'[{ppr}]    - {s}: {outer_objs[i]:.8f} '
                    f'(best: {bout_full_obj[i]:.8f} ({bfo_iter[i]}/{OUT_ITER}))'
                )
            # invoking lr scheduler for outer level optimization
            out_sched.step(all_test_objs)
            if MINMAX and not NOSLRS:
                simplex_sched.step(all_test_objs)

            # compute opt & stats for unseen tasks
            dC = C.clone().detach()
            for tt, ttdata, ttw, ttin_opt, ttin_sched in zip(
                    TTASKS, TTASK_DATA, t_inner_vars, t_in_opts, t_in_scheds
            ):
                logger.info(f'[{ppr}] Unseen task {tt} full-stats comp ....')
                ntrain = ttdata['train-size']
                BSIZE = BATCH_SIZE
                X, y = ttdata['train']
                logger.debug(f"- Training unseen task {tt} with data of size {X.shape}")
                # Optimize inner var for unseen new task for fixed outer var
                epoch_loss = 0.0
                total_in_steps_left = FULL_STATS_PER_ITER * IN_ITER
                while total_in_steps_left > 0:
                    tidxs = np.arange(ntrain)
                    RNG.shuffle(tidxs)
                    logger.debug(
                        f'[{ppr}] [UNSEEN {tt}] '
                        f'Train size: {ntrain}'
                        f' ({total_in_steps_left} steps left)'
                    )
                    old_ttw = ttw.weight.clone().detach()
                    start_idx = 0
                    epoch_loss = 0.0
                    steps_in_epoch = 0
                    while start_idx < ntrain:
                        ttin_opt.zero_grad()
                        bidxs = tidxs[start_idx: min(start_idx + BSIZE, ntrain)]
                        bX = X[bidxs]
                        by = y[bidxs]
                        preds = ttw(bX)
                        btloss = loss(by, preds)
                        btloss += reg(dC, ttw.weight, logspace=LOGC)
                        btloss.backward(retain_graph=True)
                        ttin_opt.step()
                        start_idx += BSIZE
                        epoch_loss += btloss.item()
                        steps_in_epoch += 1
                        total_in_steps_left -= 1
                        if total_in_steps_left == 0:
                            break
                    epoch_loss /= steps_in_epoch
                    t_in_delta = torch.linalg.norm(ttw.weight - old_ttw).item()
                    if in_delta < TOL:
                        logger.info(
                            f'[{ppr}] [UNSEEN {tt}] '
                            f'Train size: {ntrain} '
                            f'exiting opt with delta: {t_in_delta:.8f}'
                        )
                        break
                logger.info(
                    f'[{ppr}] [UNSEEN {tt}] Train size: {ntrain} '
                    f' Concluded with epoch loss: {epoch_loss:.6f}'
                )
                # evaluate full train/test loss/obj
                tX, ty = ttdata['test']
                ntest = ttdata['test-size']
                with torch.no_grad():
                    # computing full loss + obj on train set
                    preds = ttw(X)
                    train_loss = loss(y, preds).item()
                    train_obj = train_loss + reg(dC, ttw.weight, logspace=LOGC).item()
                    # computing full obj on test set
                    preds = ttw(tX)
                    test_loss = loss(ty, preds).item()
                    ttin_sched.step(test_loss)
                    tall_stats += [(
                        oi+1, tt,
                        train_loss, train_obj, # g-stuff
                        test_loss, # f-te-stuff
                    )]

        if converged:
            logger.warning(
                f"[{ppr}] Exiting optimization with sum(IN-DELTAs): {in_delta_sum:.8f}, "
                f"OUT-DELTA: {out_delta:.8f}"
            )
            break
    logger.info(
        f'Best obj: {bout_obj:.5f} ({bo_iter}/{OUT_ITER})'
    )
    for i, s in zip([0, 1], ['mean', 'max']):
        logger.info(
            f' - Best {s} obj: {bout_full_obj[i]:.8f} '
            f'({bfo_iter[i]}/{OUT_ITER}))'
        )
    all_stats_df = pd.DataFrame(all_stats, columns=all_stats_col)
    delta_stats_df = pd.DataFrame(delta_stats, columns=delta_stats_col)
    tall_stats_df = pd.DataFrame(tall_stats, columns=tall_stats_col)

    return all_stats_df, delta_stats_df, tall_stats_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data', '-d', choices=['MNIST', 'FashionMNIST'], help='Data set to use'
    )
    parser.add_argument(
        '--path_to_data', '-Z', type=str,
        default='/home/pari/data/torchvision/',
        help='Path to load data'
    )
    parser.add_argument(
        '--nobjs', '-a', type=int, default=10, help='Number of objectives for optimization'
    )
    parser.add_argument(
        '--ntobjs', '-A', type=int, default=0, help='Number of unseen objectives'
    )
    parser.add_argument('--nnz_ratio', '-n', type=float, default=0.4, help='NNZ ratio')
    parser.add_argument(
        '--ex_nnz_ratio', '-e', type=float, default=0.05, help='Extra NNZ ratio'
    )
    parser.add_argument(
        '--n_hard_tasks', '-T', type=int, default=2, help='Number of hard tasks'
    )
    parser.add_argument(
        '--in_lrate', '-L', type=float, default=0.001,
        help='Initial learning rate for inner level'
    )
    parser.add_argument(
        '--out_lrate', '-l', type=float, default=0.0001,
        help='Initial learning rate for outer level'
    )
    parser.add_argument(
        '--simplex_lrate', '-Y', type=float, default=0.01,
        help='Initial learning rate for outer level'
    )
    parser.add_argument(
        '--lr_decay', '-y', type=float, default=0.5,
        help='Factor to reduce learning rate by'
    )
    parser.add_argument(
        '--lr_patience', '-z', type=int, default=20, help='Patience for LR scheduler'
    )
    parser.add_argument(
        '--inner_loop', '-I', type=int, default=1,
        help='Number of inner level iterations'
    )
    parser.add_argument(
        '--max_outer_loop', '-O', type=int, default=100,
        help='Max outer level iters'
    )
    parser.add_argument(
        '--inner_batch_size', '-B', type=int, default=32,
        help='Batch size for inner level'
    )
    parser.add_argument(
        '--outer_batch_size', '-b', type=int, default=128,
        help='Batch size for outer level'
    )
    parser.add_argument(
        '--minmax', '-M', action='store_true', help='Minmax version'
    )
    parser.add_argument(
        '--random_seed', '-S', type=int, default=5489, help='Random seed for RNG'
    )
    parser.add_argument(
        '--full_stats_per_iter', '-F', type=int, default=10,
        help='Save full stats every this iters'
    )
    parser.add_argument(
        '--tolerance', '-x', type=float, default=1e-7,
        help='Tolerance of optimization'
    )
    parser.add_argument(
        '--output_dir', '-U', type=str, default='',
        help='Directory to save results in'
    )
    parser.add_argument(
        '--delta', '-D', type=float, default=0.0,
        help='Minimum weight spread across all tasks'
    )
    parser.add_argument(
        '--init_c', '-C', type=float, default=0.0,
        help='Initial value for the regularization penalty'
    )
    parser.add_argument(
        '--logspace', '-c', action='store_true', help='Search in log scale'
    )
    parser.add_argument(
        '--noise_std_scale', '-N', type=float, default=0.0,
        help='Scaling factor for gaussian noise variance'
    )
    parser.add_argument(
        '--no_simplex_scheduler', '-s', action='store_true', help='No simplex LR scheduler'
    )
    parser.add_argument(
        '--train_val_size', '-X', type=int, default=0,
        help='Training and validation set sizes for each task'
    )

    args = parser.parse_args()
    expt_tag = args2tag(parser, args)
    logger.info(f'Experiment tag: {expt_tag}')

    assert os.path.isdir(args.path_to_data)
    assert os.path.isdir(args.output_dir) or args.output_dir == ''
    assert args.nobjs > 1
    assert args.ntobjs >= 0
    assert args.in_lrate > 0.
    assert args.out_lrate > 0.
    assert args.simplex_lrate > 0.
    assert 0. < args.lr_decay < 1.
    assert args.lr_patience > 0
    assert args.inner_loop >= 1
    assert args.max_outer_loop > 1
    assert args.inner_batch_size > 1 and args.outer_batch_size > 1
    assert args.full_stats_per_iter > 1
    assert args.tolerance > 0.
    assert args.n_hard_tasks >= 0
    assert args.nnz_ratio > 0.0
    assert args.ex_nnz_ratio > 0.0 or args.n_hard_tasks == 0
    assert 0.3 > args.delta >=0.0
    assert args.logspace or args.init_c >= 0.
    assert args.noise_std_scale >= 0.0
    assert args.train_val_size >= 0

    RNG = np.random.RandomState(args.random_seed)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    full_data = get_data(args.data, args.path_to_data)
    input_dim = full_data['train'].data.shape
    max_X = torch.max(full_data['train'].data).item()
    max_X1 = torch.max(full_data['test'].data).item()
    logger.info(f'Training set max: {max_X:.2f}, test set max: {max_X1}')
    all_labels = np.unique(full_data['train'].targets.numpy())
    logger.info(f"Full data with {input_dim} features and {len(all_labels)} classes")
    logger.info(f"- Classes: {all_labels}")
    all_tasks = [
        (all_labels[i], all_labels[j])
        for i in range(len(all_labels)) for j in range(i + 1, len(all_labels))
    ]
    logger.info(f'Total of {len(all_tasks)} binary classification tasks')
    # RANDOM CONFIG
    tidxs = np.arange(len(all_tasks))
    RNG.shuffle(tidxs)
    tasks = [all_tasks[tidxs[i]] for i in range(args.nobjs)]
    ttasks = [all_tasks[tidxs[i]] for i in range(args.nobjs, args.nobjs+args.ntobjs)]
    logger.info(
        f"Performing {'minmax' if args.minmax else 'average'} optimization"
        f" with the following tasks:"
    )
    logger.info(f"- Tasks: {tasks}")
    logger.info(f"To be evaluated with the following tasks:")
    logger.info(f"- Tasks: {ttasks}")
    task_data = [get_task_data(
        full_data, t, val=True, train_val_size=args.train_val_size
    ) for t in tasks]
    ttask_data = [get_task_data(
        full_data, tt, val=False, train_val_size=args.train_val_size
    ) for tt in ttasks]
    # scale the data for better scaling
    for td in task_data + ttask_data:
        X, _ = td['train']
        X /= max_X
        tX, _ = td['test']
        tX /= max_X
        if 'val' in td:
            vX, _ = td['val']
            vX /= max_X
    orig_dim = task_data[0]['train'][0].shape[1]
    logger.info(f"Starting with original input dim: {orig_dim} ...")

    for t, td in zip(tasks + ttasks, task_data + ttask_data):
        X, _ = td['train']
        max_trX = torch.max(X).item()
        tX, _ = td['test']
        max_teX = torch.max(tX).item()
        logger.info(f'Task {t} -- max X: {max_trX:.4f}, max tX: {max_teX:.4f}')
        if 'val' in td:
            vX, _ = td['val']
            max_vaX = torch.max(vX).item()
            logger.info(f'Task {t} -- max vX: {max_vaX:.4f}')
    output_dir = os.path.join(args.output_dir, expt_tag)
    if args.output_dir != '':
        assert not os.path.exists(output_dir)

    idxs = np.arange(orig_dim)
    np.random.shuffle(idxs)
    nnz = int(args.nnz_ratio * orig_dim)
    ennz = int(args.ex_nnz_ratio * orig_dim)
    nnz_idxs = idxs[:nnz]
    ennz_idxs = [
        idxs[nnz+(i*ennz):nnz+((i+1)*ennz)]
        for i in range(args.n_hard_tasks)
    ]
    logger.info(f'NZ idxs:\n{nnz_idxs}')
    logger.info(f'Extra NZ idxs:\n{ennz_idxs}')
    noise_var = 0.05 / np.sqrt(nnz)
    ntasks = len(tasks)
    nttasks = len(ttasks)
    task_weights = []
    ttask_weights = []
    for nt, tw in zip([ntasks, nttasks], [task_weights, ttask_weights]):
        logger.info(f'Generating true sparse models with {nnz} NNZs for {nt} tasks')
        logger.info(f'-- with {args.n_hard_tasks} tasks with extra {ennz} NNs')
        for i in range(nt):
            w = np.zeros(orig_dim)
            v = (
                0.1 * np.random.randint(1, 10, nnz)
                + np.random.normal(0, noise_var, nnz)
            )
            logger.info(f'Model weights:\n{v}')
            for ii, vv in zip(nnz_idxs, v):
                w[ii] = vv
            if i >= (nt - args.n_hard_tasks):
                v = (
                    0.1 * np.random.randint(1, 10, ennz)
                    + np.random.normal(0, noise_var, ennz)
                )
                logger.info(v)
                for ii, vv in zip(ennz_idxs[i - (nt - args.n_hard_tasks)], v):
                    w[ii] = vv
            tw += [w]

    for t, tw, td in zip(
            tasks + ttasks,
            task_weights + ttask_weights,
            task_data + ttask_data
    ):
        wtensor = torch.Tensor(tw)
        wtensor /= torch.linalg.norm(wtensor)
        logger.info(f'Weight tensor shape: {wtensor.shape}')
        for s in ['train', 'test', 'val']:
            if s in td:
                X, y = td[s]
                logger.info(f'Task: {t}, set: {s}, data: {X.shape}, {y.shape}')
                yreg = torch.matmul(X, wtensor).view(-1, 1)
                logger.info(
                    f'--> y-reg-stats: '
                    f' {yreg.shape}, {yreg.min():.3f}, {yreg.mean():.3f}, {yreg.max():.3f}'
                )
                if args.noise_std_scale > 0.0:
                    # std = args.noise_std_scale * yreg.mean().item()
                    std = args.noise_std_scale
                    noise = torch.normal(0, std, size=yreg.shape)
                    logger.info(f'Adding noise (std: {std:.5f}) of size: {noise.shape}')
                    yreg += noise
                td[s] = (X, yreg)
                
    astats, dstats, tastats = run_hpo(
        TASKS=tasks,
        TASK_DATA=task_data,
        TTASKS=ttasks,
        TTASK_DATA=ttask_data,
        RNG=RNG,
        IN_DIM=orig_dim,
        BATCH_SIZE=args.inner_batch_size,
        BATCH_SIZE_OUT=args.outer_batch_size,
        LRATE=args.in_lrate,
        LRATE_OUT=args.out_lrate,
        LRATE_SIMP=args.simplex_lrate,
        LR_DECAY=args.lr_decay,
        LR_PATIENCE=args.lr_patience,
        IN_ITER=args.inner_loop,
        OUT_ITER=args.max_outer_loop,
        MINMAX=args.minmax,
        FULL_STATS_PER_ITER=args.full_stats_per_iter,
        TOL=args.tolerance,
        DELTA=args.delta,
        LOGC=args.logspace,
        INIT_C=args.init_c,
        NOSLRS=args.no_simplex_scheduler,
    )

    print('All stats: ', astats.shape)
    print(astats.head(10))
    print(astats.tail(10))
    print('Delta stats: ', dstats.shape)
    print(dstats.head(10))
    print(dstats.tail(10))
    print('Unseen all stats', tastats.shape)
    print(tastats.head(10))
    print(tastats.tail(10))

    if args.output_dir != '':
        os.mkdir(output_dir)
        logger.info(f'Saving results in {output_dir} ...')
        dfs = [astats, dstats, tastats]
        fnames = [
            'opt_stats_seen_tasks.csv',
            'opt_deltas.csv',
            'objs_unseen_tasks.csv'
        ]
        for d, f in zip(dfs, fnames):
            logger.info(f'- saving {f}')
            d.to_csv(os.path.join(output_dir, f), header=True, index=False)
