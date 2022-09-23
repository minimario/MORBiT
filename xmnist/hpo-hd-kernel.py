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
logger = logging.getLogger('KRR-HD-HPO')
logger.setLevel(logging.INFO)

def reg(C, w, logspace=True):
    if logspace:
        # return torch.dot(torch.exp(C), torch.square(w.view(-1)))
        return torch.dot(torch.exp(C), torch.sum(torch.square(w), 0))
    else:
        # return torch.dot(C, torch.square(w.view(-1)))
        return torch.dot(C, torch.sum(torch.square(w), 0))


def RFF(G, Wrand, feats, logspace=True):
    pre_int_rep = torch.mm(
        torch.exp(G) * feats, Wrand
    ) if logspace else torch.mm(
        G * feats, Wrand
    )
    return torch.concat((
        torch.cos(pre_int_rep), torch.sin(pre_int_rep)
    ), dim=1)


def blogloss(true_labels, logits):
    in_softmax = nn.Softmax(dim=1)
    loss = nn.CrossEntropyLoss()
    return loss(in_softmax(logits), true_labels)
    ## sign_labels = 2 * true_labels - 1
    ## per_point_loss = torch.log(1 + torch.exp(- sign_labels * logits))
    ## return torch.mean(per_point_loss)


def run_hpo(
        TASKS,
        TASK_DATA,
        TTASKS,
        TTASK_DATA,
        RNG,
        IN_DIM,
        INT_DIM,
        NCLASSES=2,
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
        LOGCG=True,
        INITC=0.0,
        INITG=1.0,
        NOSLRS=False,
        LAMBDA_PENALTY=0.0,
):
    assert NCLASSES == 2
    # OUTER LEVEL VARS
    C = INITC * torch.ones(2 * INT_DIM)
    C.requires_grad = True
    G = INITG * torch.ones(IN_DIM)
    G.requires_grad = True
    out_opt = torch.optim.SGD(
        [C, G],
        lr=LRATE_OUT,
        momentum=0.9,
    )
    out_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        out_opt, 'min', factor=LR_DECAY, verbose=True,
        patience=LR_PATIENCE,
    )

    # INNER LEVEL VARS
    ntasks = len(TASKS)
    inner_vars = [nn.Linear(2 * INT_DIM, 2) for _ in TASKS]
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

    # Functions used by all tasks and level -- not optimized
    RPM = torch.normal(0, 1, size=(IN_DIM, INT_DIM))
    RPM.requires_grad = False

    # set up variables and optimizers for unseen tasks
    nttasks = len(TTASKS)
    t_inner_vars = [nn.Linear(2 * INT_DIM, 2) for _ in TTASKS]
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

    best_objs = {
        'batc': (np.inf, -1), # batch upper obj
        'fvme': (np.inf, -1), # full upper obj mean
        'fvma': (np.inf, -1), # full upper obj max
    }

    for oi in range(OUT_ITER):
        logger.debug(f'Starting outer loop {oi+1}/{OUT_ITER} ...')
        ppr = f'{oi+1}/{OUT_ITER}'
        batch_stats = ((oi+1) % FULL_STATS_PER_ITER != 0)
        out_opt.zero_grad()
        if MINMAX:
            simplex_opt.zero_grad()
            old_simplex_vars = simplex_vars.clone().detach()
            old_simplex_vars.requires_grad = False
        out_losses = []
        in_losses = []
        in_objs = []
        in_deltas = []
        outer_loss = 0.0
        outer_old = [C.clone().detach(), G.clone().detach()]
        outer_old[0].requires_grad = False
        outer_old[1].requires_grad = False
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
            inner_old.requires_grad = False
            for ii in range(IN_ITER):
                topt.zero_grad()
                logger.debug(
                    f'[{ppr}] [{t}] ({ii+1}/{IN_ITER}) Train size: {tsize}'
                )
                bidxs = [np.random.randint(0, tsize) for _ in range(BATCH_SIZE)]
                bX = X[bidxs]
                by = y[bidxs]
                preds = tw(RFF(G, RPM, bX, logspace=LOGCG))
                btloss = blogloss(by, preds)
                # tracking loss
                total_loss += btloss.item()
                btloss += reg(C, tw.weight, logspace=LOGCG)
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
            logits = tw(RFF(G, RPM, bX, logspace=LOGCG))
            toloss = blogloss(by, logits)
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
            f"(best: {best_objs['batc'][0]:.8f} ({best_objs['batc'][1]}/{OUT_ITER}))"
        )
        if outer_loss.item() < best_objs['batc'][0]:
            best_objs['batc'] = (outer_loss.item(),  oi + 1)
        # take optimization step
        outer_loss.backward(retain_graph=True)
        out_opt.step()
        term = ['logC', 'logG'] if LOGCG else ['C', 'G']
        # clamp to 0 when optimizing in original space
        if not LOGCG:
            with torch.no_grad():
                C.clamp_(min=0.0)
                G.clamp_(min=0.0)
        with torch.no_grad():
            curr_lr = out_opt.param_groups[0]['lr']
            out_delta = 0
            for t, v, ov in zip(term, [C, G], outer_old):
                od = torch.linalg.norm(v - ov)
                if not batch_stats:
                    logger.info(
                        f'[{ppr}] {t} {list(v.shape)}: '
                        f'({torch.min(v):.2f}, {torch.mean(v):.2f},'
                        f' {torch.max(v):.2f})'
                        f'; g-{t}: ('
                        f'{torch.min(v.grad):.2f}, '
                        f'{torch.mean(v.grad):.2f}, '
                        f'{torch.max(v.grad):.2f})'
                        f'; D-{t}: {od:.4f} '
                    )
                    logger.info(f'[{ppr}] Current learning rate: {curr_lr:.6f}')
                out_delta += od
            # save delta stats
            delta_stats += [(oi+1, 'ALL', out_delta.item(), curr_lr, np.nan)]

        # Update simplex lambda if minmax
        logger.debug(f'[{ppr}] Lambdas: {simplex_vars}')
        if MINMAX:
            # negate gradient for gradient ascent
            simplex_vars.grad *= -1.
            if LAMBDA_PENALTY > 0.:
                simplex_vars.grad += (2 * LAMBDA_PENALTY * (old_simplex_vars - 1./ntasks))
            logger.debug(
                f'[{ppr}] - g-Lambda: '
                f'{simplex_vars.grad.clone().detach().numpy()}'
            )
            simplex_opt.step()
            logger.debug(f'[{ppr}] - U-Lambdas: {simplex_vars}')
            simplex_proj_inplace(simplex_vars, z = 1-DELTA)
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
                    train_out = tw(RFF(G, RPM, X, logspace=LOGCG))
                    train_loss = blogloss(y, train_out)
                    train_obj = train_loss.item()
                    train_obj += reg(C, tw.weight, logspace=LOGCG).item()
                    val_out = tw(RFF(G, RPM, vX, logspace=LOGCG))
                    val_loss = blogloss(vy, val_out)
                    test_out = tw(RFF(G, RPM, tX, logspace=LOGCG))
                    test_loss = blogloss(ty, test_out)
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
            for i, k in zip([0, 1], ['fvme', 'fvma']):
                if outer_objs[i] < best_objs[k][0]:
                    best_objs[k] = (outer_objs[i], oi + 1)
            logger.info(f'[{ppr}] OUTER FULL LOSS:')
            for i, s in zip([0, 1], ['fvme', 'fvma']):
                logger.info(
                    f'[{ppr}]    - {s}: {outer_objs[i]:.8f} '
                    f'(best: {best_objs[s][0]:.8f} ({best_objs[s][1]}/{OUT_ITER}))'
                )
            # invoking lr scheduler for outer level optimization
            out_sched.step(all_test_objs)
            if MINMAX and not NOSLRS:
                simplex_sched.step(all_test_objs)

            # compute opt & stats for unseen tasks
            dC, dG = C.clone().detach(), G.clone().detach()
            dC.requires_grad = False
            dG.requires_grad = False
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
                        preds = ttw(RFF(dG, RPM, bX, logspace=LOGCG))
                        btloss = blogloss(by, preds)
                        btloss += reg(dC, ttw.weight, logspace=LOGCG)
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
                    preds = ttw(RFF(dG, RPM, X, logspace=LOGCG))
                    train_loss = blogloss(y, preds).item()
                    train_obj = train_loss + reg(dC, ttw.weight, logspace=LOGCG).item()
                    # computing full obj on test set
                    preds = ttw(RFF(dG, RPM, tX, logspace=LOGCG))
                    test_loss = blogloss(ty, preds).item()
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
        f'Best objective:\n'
        f"- batch       : {best_objs['batc'][0]:.5f} ({best_objs['batc'][1]}/{OUT_ITER})\n"
        f"- out-obj mean: {best_objs['fvme'][0]:.5f} ({best_objs['fvme'][1]}/{OUT_ITER})\n"
        f"- out-obj max : {best_objs['fvma'][0]:.5f} ({best_objs['fvma'][1]}/{OUT_ITER})\n"
    )
    all_stats_df = pd.DataFrame(all_stats, columns=all_stats_col)
    delta_stats_df = pd.DataFrame(delta_stats, columns=delta_stats_col)
    tall_stats_df = pd.DataFrame(tall_stats, columns=tall_stats_col)

    return all_stats_df, delta_stats_df, tall_stats_df




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data', '-d', choices=['Letter'],
        help='Data set to use'
    )
    parser.add_argument(
        '--nobjs', '-a', type=int, default=10, help='Number of objectives for optimization'
    )
    parser.add_argument(
        '--ntobjs', '-A', type=int, default=0, help='Number of unseen objectives'
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
        '--inner_loop', '-I', type=int, default=1, help='Number of inner level iterations'
    )
    parser.add_argument(
        '--max_outer_loop', '-O', type=int, default=100, help='Max outer level iters'
    )
    parser.add_argument(
        '--inner_batch_size', '-B', type=int, default=32,
        help='Batch size for inner level'
    )
    parser.add_argument(
        '--outer_batch_size', '-b', type=int, default=128,
        help='Batch size for outer level'
    )
    parser.add_argument('--minmax', '-M', action='store_true', help='Minmax version')
    parser.add_argument(
        '--random_seed', '-S', type=int, default=5489, help='Random seed for RNG'
    )
    parser.add_argument(
        '--random_seed_for_tasks', '-t', type=int, default=0, help='Random seed for task sampler'
    )
    parser.add_argument(
        '--random_seed_for_task_data', '-T', type=int, default=0,
        help='Random seed for task data sampler'
    )
    parser.add_argument(
        '--full_stats_per_iter', '-F', type=int, default=10,
        help='Save full stats every this iters'
    )
    parser.add_argument(
        '--tolerance', '-x', type=float, default=1e-7,
        help='Tolerance of optimization')
    ## parser.add_argument(
    ##     '--tobj_max_epochs', '-E', type=int, default=100,
    ##     help='Max epochs for test tasks'
    ## )
    parser.add_argument(
        '--output_dir', '-U', type=str, default='', help='Directory to save results in'
    )
    parser.add_argument(
        '--delta', '-D', type=float, default=0.0,
        help='Minimum weight spread across all tasks'
    )
    parser.add_argument(
        '--initc', '-C', type=float, default=0.0,
        help='Initial value for the regularization penalty'
    )
    parser.add_argument(
        '--initg', '-G', type=float, default=1e-3,
        help='Initial value for the per-feature scaling'
    )
    parser.add_argument(
        '--logspace', '-c', action='store_true', help='Search in log scale'
    )
    parser.add_argument('--int_dim', '-N', type=int, default=32, help='Number of RFFs')
    parser.add_argument(
        '--no_simplex_scheduler', '-s', action='store_true', help='No simplex LR scheduler'
    )
    parser.add_argument(
        '--train_val_size', '-X', type=int, default=0,
        help='Training and validation set sizes for each task'
    )
    parser.add_argument(
        '--lambda_penalty', '-e', type=float, default=0.,
        help='Penalty on the simplex vars to diverge from uniform distribution',
    )

    args = parser.parse_args()
    expt_tag = args2tag(parser, args)
    logger.info(f'Experiment tag: {expt_tag}')

    # assert os.path.isdir(args.path_to_data)
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
    # assert 1 <= args.tobj_max_epochs <= 10
    assert 0.3 > args.delta >=0.0
    assert args.logspace or (args.initc >= 0. and args.initg > 0.)
    assert args.int_dim > 10
    assert args.train_val_size >= 0
    assert args.lambda_penalty >= 0.



    SEED1 = args.random_seed if (
        (args.random_seed_for_task_data == 0) or
        (args.random_seed_for_task_data == args.random_seed)
    ) else args.random_seed_for_task_data

    RNG = np.random.RandomState(SEED1)
    torch.manual_seed(SEED1)
    np.random.seed(SEED1)
    random.seed(SEED1)

    tRNG = (
        np.random.RandomState(args.random_seed)
        if args.random_seed_for_tasks == 0
        else np.random.RandomState(args.random_seed_for_tasks)
    ) if args.random_seed_for_tasks != args.random_seed else RNG

    full_data = get_data(args.data, '')
    input_dim = full_data['train'].data.shape
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
    tRNG.shuffle(tidxs)
    tasks = [all_tasks[tidxs[i]] for i in range(args.nobjs)]
    ttasks = [all_tasks[tidxs[i]] for i in range(args.nobjs, args.nobjs+args.ntobjs)]
    ## # FIXED TASKS
    ## tasks = [
    ##     # EASY
    ##     ## (10, 12), # m,k
    ##     ## (0, 1),   # a,b
    ##     ## (14, 21), # o,v
    ##     ## (4, 20),  # e,u
    ##     ## (16, 22), # q,w
    ##     ## (13, 19), # n,t
    ##     ## (8, 15),  # i,p
    ##     ## (8, 12),  # i,m
    ##     ## (8, 22),  # i,w
    ##     ## (5, 13),  # f,n
    ##     (5, 21),  # f,v
    ##     (12, 21), # m,v
    ##     (12, 15), # m,p
    ##     (9, 22),  # j,w
    ##     (9, 13),  # j,n
    ##     # HARD
    ##     ## (21, 22), # v,w
    ##     ## (12, 13), # m,n
    ##     ## (8, 9),   # i,j
    ##     ## (5, 15),  # f,p
    ##     ## (18, 23), # s,x
    ##     ## (0, 20),  # a,u
    ##     ## (11, 19), # l,t
    ##     (1, 3),   # b,d
    ##     (7, 10),  # h,k
    ##     (4, 11),  # e,l
    ## ]
    ## ttasks = [
    ##     # EASY
    ##     (1, 2),   # b,c
    ##     (4, 21),  # e,v
    ##     (0, 12),  # a,k
    ##     # HARD
    ##     (18, 25), # s,z
    ##     (6, 16),  # g,q
    ##     (1, 17),  # b,r
    ##     (23, 25), # x,z
    ##     (2, 11),  # c,l
    ##     (2, 4),   # c,e
    ##     (6, 14),  # g,o
    ## ]
    logger.info(
        f"Performing {'minmax' if args.minmax else 'average'} optimization"
        f" with the following tasks:"
    )
    logger.info(f"- Tasks: {tasks}")
    logger.info(f"To be evaluated with the following tasks:")
    logger.info(f"- Tasks: {ttasks}")
    NCLASSES=2
    task_data = [get_task_data(
        full_data, t, val=True, train_val_size=args.train_val_size
    ) for t in tasks]
    ttask_data = [get_task_data(
        full_data, tt, val=False, train_val_size=args.train_val_size
    ) for tt in ttasks]
    orig_dim = task_data[0]['train'][0].shape[1]
    logger.info(f"Starting with original input dim: {orig_dim} ...")

    for t, td in zip(tasks + ttasks, task_data + ttask_data):
        X, _ = td['train']
        min_X, max_X = torch.min(X).item(), torch.max(X).item()
        tX, _ = td['test']
        min_tX, max_tX = torch.min(tX).item(), torch.max(tX).item()
        logger.info(
            f'Task {t} -- X ({min_X:.4f}, {max_X:.4f}), tX ({min_tX:.4f},{max_tX:.4f})'
        )
        if 'val' in td:
            vX, _ = td['val']
            min_vX, max_vX = torch.min(vX).item(), torch.max(vX).item()
            logger.info(f'Task {t} -- vX ({min_vX:.4f}, {max_vX:.4f})')


    SEED2 = args.random_seed
    if SEED2 != SEED1:
        logger.info(f'Updating RNG seed from {SEED1} --> {SEED2}')
        RNG = np.random.RandomState(SEED2)
        torch.manual_seed(SEED2)
        np.random.seed(SEED2)
        random.seed(SEED2)
    else:
        logger.info(f'Keeping RNG seed as {SEED1}')


    output_dir = os.path.join(args.output_dir, expt_tag)
    if args.output_dir != '':
        assert not os.path.exists(output_dir)

    astats, dstats, tastats = run_hpo(
        TASKS=tasks,
        TASK_DATA=task_data,
        TTASKS=ttasks,
        TTASK_DATA=ttask_data,
        RNG=RNG,
        IN_DIM=orig_dim,
        INT_DIM=args.int_dim,
        NCLASSES=NCLASSES,
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
        # TOBJ_MAX_EPOCHS=args.tobj_max_epochs,
        DELTA=args.delta,
        LOGCG=args.logspace,
        INITC=args.initc,
        INITG=args.initg,
        NOSLRS=args.no_simplex_scheduler,
        LAMBDA_PENALTY=args.lambda_penalty,
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
