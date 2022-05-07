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
logger = logging.getLogger('SVM-RFF-HPO')
logger.setLevel(logging.INFO)

## IN_DIMS = [in_dim, in_dim, in_dim]
## INT_DIMS = [10000, 10000, 10000]
## BATCH_SIZE = 64
## LRATE = 1.
## IN_ITER = 10000
## OUT_ITER = 1
## LOGLOSS = True

def RFF(lG, W, f, feats):
    pre_int_rep = torch.exp(lG) * torch.matmul(feats, W)
    return f *  torch.concat((
        torch.cos(pre_int_rep),
        torch.sin(pre_int_rep)
    ), dim=1)


def RFF2(lG, W, f, feats):
    return f * torch.cos(torch.exp(lG) * torch.matmul(feats, W))


def run_hpo(
        TASKS,
        TASK_DATA,
        TTASKS,
        TTASK_DATA,
        RNG,
        IN_DIM,
        INT_DIM,
        NCLASSES,
        BATCH_SIZE=32,
        BATCH_SIZE_OUT=128,
        LRATE=0.001,
        LRATE_OUT=0.001,
        LRATE_SIMP=0.1,
        LR_DECAY=0.8,
        LR_PATIENCE=20,
        IN_ITER=1,
        OUT_ITER=1000,
        INNORM=2,
        MINMAX=False,
        FULL_STATS_PER_ITER=10,
        TOL=1e-7,
        TOBJ_MAX_EPOCHS=10,
        PRECOMPUTE_RPS=False,
):

    # OUTER LEVEL VARS
    logC = torch.tensor(-1.0)
    logC.requires_grad = True
    logG = torch.tensor(0.0)
    logG.requires_grad = True
    out_opt = torch.optim.SGD(
        # [
        #     {'params': logC, 'lr': LRATE_OUT,},
        #     {'params': logG, 'lr': 0.1 * LRATE_OUT,},
        # ],
        [logC, logG],
        lr=LRATE_OUT,
        momentum=0.9,
        weight_decay=1e-4,
    )
    out_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        out_opt, 'min', factor=LR_DECAY, verbose=True,
        patience=LR_PATIENCE,
    )
    # out_sched = torch.optim.lr_scheduler.StepLR(
    #     out_opt, step_size=100, gamma=LR_DECAY, verbose=False
    # )

    # # PER TASK INTERMEDIATE DIM
    # TDIMS = [(INT_DIM if RNG.randint(10) % 2 == 0 else INT_DIM//2) for _ in tasks]
    # logger.info(f'Using intermediate dimensionalities: {TDIMS}')

    # INNER LEVEL VARS
    ntasks = len(TASKS)
    inner_vars = [nn.Linear(INT_DIM, NCLASSES) for _ in TASKS]
    # inner_vars = [nn.Linear(D, NCLASSES) for D in TDIMS]
    in_opts, in_scheds = [], []
    for iv in inner_vars:
        in_opt = torch.optim.SGD(
            iv.parameters(),
            lr=LRATE,
            momentum=0.9,
            weight_decay=1e-4,
        )
        # in_sched = torch.optim.lr_scheduler.StepLR(
        #     in_opt, step_size=100, gamma=LR_DECAY, verbose=False
        # )
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
            # lr=(LRATE_OUT / np.sqrt(ntasks)),
            momentum=0.9,
            weight_decay=1e-4,
        )
        # simplex_sched = torch.optim.lr_scheduler.StepLR(
        #     simplex_opt, step_size=30, gamma=LR_DECAY, verbose=False
        # )
        simplex_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            simplex_opt, 'min', factor=LR_DECAY, verbose=True,
            patience=LR_PATIENCE,
        )

    # Functions used by all tasks and level -- not optimized
    in_softmax = nn.Softmax(dim=1)
    loss = nn.CrossEntropyLoss()

    # PER-TASK RFF PROJECTION MATRIX
    PROJS = []
    for t, td in zip(TASKS, TASK_DATA):
        W = (1/ np.sqrt(IN_DIM)) * torch.normal(0., 1., size=(IN_DIM, INT_DIM))
        W.requires_grad = False
        if PRECOMPUTE_RPS:
            logger.info(f'Precomputing random projections for task {t} ...')
            X, y = td['train']
            td['train'] = (torch.matmul(X, W), y)
            vX, vy = td['val']
            td['val'] = (torch.matmul(vX, W), vy)
            tX, ty = td['test']
            td['test'] = (torch.matmul(tX, W), ty)
            PROJS += [None]
        else:
            PROJS += [W]
    TPROJS = []
    for tt, ttd in zip(TTASKS, TTASK_DATA):
        W = (1/ np.sqrt(IN_DIM)) * torch.normal(0., 1., size=(IN_DIM, INT_DIM))
        W.requires_grad = False
        if PRECOMPUTE_RPS:
            logger.info(f'Precomputing random projections for test task {tt} ...')
            X, y = ttd['train']
            ttd['train'] = (torch.matmul(X, W), y)
            tX, ty = ttd['test']
            ttd['test'] = (torch.matmul(tX, W), ty)
            TPROJS += [None]
        else:
            TPROJS += [W]

    # for D in TDIMS:
    #     W = (1/ np.sqrt(IN_DIM)) * torch.normal(0., 1., size=(IN_DIM, D))
    #     W.requires_grad = False
    #     PROJS += [W]

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
        'task', # task ID for inner_var or ALL for outer_var or SIMP for simplex_vars
        'delta', # change in variables
        'lr', # current learning rate for variable
        'lambda', # weight for current task; when outer_var or simplex_vars is NA
        'logC',
        'logG',
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
    # factor = (1.0 / np.sqrt(INT_DIM))
    factor = 1.0

    for oi in range(OUT_ITER):
        logger.info(f'Starting outer loop {oi+1}/{OUT_ITER} ...')
        ppr = f'{oi+1}/{OUT_ITER}'
        batch_stats = ((oi+1) % FULL_STATS_PER_ITER != 0)
        out_opt.zero_grad()
        if MINMAX:
            simplex_opt.zero_grad()
            old_simplex_vars = simplex_vars.clone().detach()
        out_losses = []
        out_accs = []
        in_losses = []
        in_objs = []
        in_deltas = []
        outer_loss = 0.0
        outer_old = [logC.item(), logG.item()]
        # Looping over all the tasks
        for t, tdata, tw, topt, tsch, tlambda, proj in zip(
                TASKS, TASK_DATA, inner_vars, in_opts, in_scheds, simplex_vars, PROJS,
        ):
            assert PRECOMPUTE_RPS or proj is not None
            logger.debug(f'[{ppr}] Task {t} ....')
            tsize = tdata['train-size']
            X, y = tdata['train']
            total_loss = 0.
            total_obj = 0.
            inner_old = tw.weight.clone()
            for ii in range(IN_ITER):
                topt.zero_grad()
                logger.debug(f'[{ppr}] [{t}] ({ii+1}/{IN_ITER}) Train size: {tsize}'),
                bidxs = [np.random.randint(0, tsize) for _ in range(BATCH_SIZE)]
                bX = X[bidxs]
                by = y[bidxs]
                probs = in_softmax(tw(
                    factor * torch.cos(torch.exp(logG) * bX)
                )) if PRECOMPUTE_RPS else in_softmax(tw(
                    RFF2(logG, proj, factor, bX)
                ))
                btloss = loss(probs, by)
                # tracking loss
                total_loss += btloss.item()
                btloss += (torch.exp(logC) * 0.5 * wnorm(tw, INNORM))
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
            probs = in_softmax(tw(
                factor * torch.cos(torch.exp(logG) * bX)
            )) if PRECOMPUTE_RPS else in_softmax(tw(
                RFF2(logG, proj, factor, bX)
            ))
            toloss = loss(probs, by)
            toobj = toloss.item()
            # Computing outer level accuracy on batch
            _, preds = torch.max(probs, 1)
            out_accs += [(by == preds).sum() / bX.size(0)]
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
            # current_lr = tsch.get_last_lr()[0] # <=== NOTE: StepLR
            current_lr = topt.param_groups[0]['lr'] # <== NOTE: ReduceLROnPlateau
            with torch.no_grad():
                in_delta = torch.linalg.norm(tw.weight - inner_old)
            in_deltas += [in_delta.item()]
            delta_stats += [(oi+1, t, in_delta.item(), current_lr, tlambda.item(), logC.item(), logG.item())]
        # print some stats
        if not batch_stats:
            logger.info(f'[{ppr}] IN LOSSES: {np.array(in_losses)}')
            logger.info(f'[{ppr}] IN OBJS: {np.array(in_objs)}')
            logger.info(f'[{ppr}] IN-Deltas: {np.array(in_deltas)}')
            logger.info(f'[{ppr}] OUT LOSS: {np.array(out_losses)}')
            logger.info(f'[{ppr}] OUT ACCS: {np.array(out_accs)}')
        logger.info(
            f'[{ppr}] OUTER LOSS: {outer_loss.item():.8f} '
            f'(best: {bout_obj:.8f} ({bo_iter}/{OUT_ITER}))'
        )
        if outer_loss.item() < bout_obj:
            bout_obj = outer_loss.item()
            bo_iter = oi + 1
        # take optimization step
        outer_loss.backward(retain_graph=True)
        logger.info(
            f'[{ppr}] logC: {logC.item():.4f}, grad-logC: {logC.grad.item():.4f}; '
            f'logG: {logG.item():.4f}, grad-logG: {logG.grad.item():.4f}'
        )
        out_opt.step()
        # save delta stats
        with torch.no_grad():
            delta_logC = abs(logC.item() - outer_old[0])
            delta_logG = abs(logG.item() - outer_old[1])
        logger.info(
            f'[{ppr}] logC: {logC.item():.6f}, Delta(logC): {delta_logC:.6f}; '
            f'logG: {logG.item():.6f}, Delta(logG): {delta_logG:.6f}'
        )
        # curr_lr = out_sched.get_last_lr()[0] # <== NOTE: StepLR
        curr_lr = out_opt.param_groups[0]['lr'] # <== NOTE: ReduceLROnPlateau
        delta_stats += [(oi+1, 'logC', delta_logC, curr_lr, np.nan, logC.item(), logG.item())]
        delta_stats += [(oi+1, 'logG', delta_logG, curr_lr, np.nan, logC.item(), logG.item())]

        # Update simplex lambda if minmax
        logger.debug(f'[{ppr}] Lambdas: {simplex_vars}')
        out_delta = delta_logC + delta_logG
        if MINMAX:
            # negate gradient for gradient ascent
            simplex_vars.grad *= -1.
            logger.debug(f'[{ppr}] - Lambda grads: {simplex_vars.grad}')
            simplex_opt.step()
            logger.debug(f'[{ppr}] - Updated Lambdas: {simplex_vars}')
            simplex_proj_inplace(simplex_vars)
            logger.info(f'[{ppr}] - Simplex projected updated Lambdas: {simplex_vars}')
            # curr_lr = simplex_sched.get_last_lr()[0] # <== NOTE: StepLR
            curr_lr = simplex_opt.param_groups[0]['lr'] # <== NOTE: ReduceLROnPlateau
            with torch.no_grad():
                lambda_delta = torch.linalg.norm(simplex_vars - old_simplex_vars)
            logger.info(f'[{ppr}] - lambda delta: {lambda_delta.item():.6f}')
            delta_stats += [(oi+1, 'SIMPLE', lambda_delta.item(), curr_lr, np.nan, logC.item(), logG.item())]
            # update out_delta to include lambda delta
            out_delta += lambda_delta.item()
            assert simplex_vars.requires_grad

        in_delta_sum = np.sum(in_deltas)
        converged = (in_delta_sum < TOL) and (out_delta < TOL)
        if (not batch_stats) or (converged):
            INREG = torch.exp(logC).item()
            # compute stats on full data
            voobjs, toobjs = [], []
            for t, tdata, tw, proj, tsched in zip(TASKS, TASK_DATA, inner_vars, PROJS, in_scheds):
                assert PRECOMPUTE_RPS or proj is not None
                logger.info(f'[{ppr}] Task {t} full-stats comp ....')
                X, y = tdata['train']
                vX, vy = tdata['val']
                tX, ty = tdata['test']
                with torch.no_grad():
                    train_out = in_softmax(tw(
                        factor * torch.cos(torch.exp(logG) * X)
                    )) if PRECOMPUTE_RPS else in_softmax(tw(
                        RFF2(logG, proj, factor, X)
                    ))
                    train_loss = loss(train_out, y)
                    train_obj = train_loss.item()
                    train_obj += (INREG * 0.5 * wnorm(tw, INNORM)).item()
                    val_out = in_softmax(tw(
                        factor * torch.cos(torch.exp(logG) * vX)
                    )) if PRECOMPUTE_RPS else in_softmax(tw(
                        RFF2(logG, proj, factor, vX)
                    ))
                    val_loss = loss(val_out, vy)
                    test_out = in_softmax(tw(
                        factor * torch.cos(torch.exp(logG) * tX)
                    )) if PRECOMPUTE_RPS else in_softmax(tw(
                        RFF2(logG, proj, factor, tX)
                    ))
                    test_loss = loss(test_out, ty)
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
            logger.info(f'[{ppr}] test: {np.array(toobjs)} (sum: {all_test_objs:.4f})')
            # invoking lr scheduler for outer level optimization
            out_sched.step(all_test_objs)
            if MINMAX:
                simplex_sched.step(all_test_objs)

            # compute opt & stats for unseen tasks
            for tt, ttdata, tproj in zip(TTASKS, TTASK_DATA, TPROJS):
                assert PRECOMPUTE_RPS or tproj is not None
                logger.info(f'[{ppr}] Unseen task {tt} full-stats comp ....')
                ntrain = ttdata['train-size']
                BSIZE = 8
                # INNER LEVEL TEST TASKS
                iv = nn.Linear(INT_DIM, NCLASSES)
                tin_opt = torch.optim.SGD(
                    iv.parameters(),
                    lr=LRATE,
                    momentum=0.9,
                    weight_decay=1e-4,
                )
                # NOTE: Avoid LR scheduling for unseen tasks; NEED to keep number of epochs small
                # tin_sched = torch.optim.lr_scheduler.StepLR(
                #     tin_opt, step_size=30, gamma=LR_DECAY, verbose=False
                # )
                X, y = ttdata['train']
                logger.debug(f"- Training unseen task {tt} with data of size {X.shape}")
                # Optimize inner var for unseen new task for fixed outer var
                epoch_loss = 0.0
                for eidx in range(TOBJ_MAX_EPOCHS):
                    tidxs = np.arange(ntrain)
                    RNG.shuffle(tidxs)
                    logger.debug(
                        f'[{ppr}] [UNSEEN {tt}] ({eidx+1}/{TOBJ_MAX_EPOCHS}) Train size: {ntrain}'
                    )
                    old_iv = iv.weight.clone()
                    start_idx = 0
                    epoch_loss = 0.0
                    steps_in_epoch = 0
                    while start_idx < ntrain:
                        tin_opt.zero_grad()
                        bidxs = tidxs[start_idx: min(start_idx + BSIZE, ntrain)]
                        bX = X[bidxs]
                        by = y[bidxs]
                        with torch.no_grad():
                            rffX = factor * torch.cos(
                                torch.exp(logG) * bX
                            ) if PRECOMPUTE_RPS else RFF2(
                                logG, tproj, factor, bX
                            )
                        probs = in_softmax(iv(rffX))
                        btloss = loss(probs, by)
                        btloss += (INREG * 0.5 * wnorm(iv, INNORM))
                        btloss.backward(retain_graph=True)
                        tin_opt.step()
                        start_idx += BSIZE
                        epoch_loss += btloss.item()
                        steps_in_epoch += 1
                    epoch_loss /= steps_in_epoch
                    # tin_sched.step()
                    in_delta = torch.linalg.norm(iv.weight - old_iv).item()
                    if in_delta < TOL:
                        logger.info(
                            f'[{ppr}] [UNSEEN {tt}] ({eidx+1}/{TOBJ_MAX_EPOCHS})'
                            f'Train size: {ntrain} '
                            f'exiting opt with delta: {in_delta:.8f}'
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
                    start_idx = 0
                    steps_in_epoch = 0
                    train_loss = 0.0
                    while start_idx < ntrain:
                        bidxs = np.arange(start_idx, min(start_idx + BATCH_SIZE_OUT, ntrain))
                        bX = X[bidxs]
                        by = y[bidxs]
                        out = in_softmax(tw(
                            factor * torch.cos(torch.exp(logG) * bX)
                        )) if PRECOMPUTE_RPS else in_softmax(iv(
                            RFF2(logG, W, factor, bX)
                        ))
                        train_loss += loss(out, by)
                        steps_in_epoch += 1
                        start_idx += BATCH_SIZE_OUT
                    train_loss /= steps_in_epoch
                    train_obj = train_loss.item()
                    train_obj += (INREG * 0.5 * wnorm(iv, INNORM)).item()
                    # computing full obj on test set
                    start_idx = 0
                    steps_in_epoch = 0
                    test_loss = 0.0
                    while start_idx < ntest:
                        bidxs = np.arange(start_idx, min(start_idx + BATCH_SIZE_OUT, ntest))
                        bX = tX[bidxs]
                        by = ty[bidxs]
                        out = in_softmax(tw(
                            factor * torch.cos(torch.exp(logG) * bX)
                        )) if PRECOMPUTE_RPS else in_softmax(iv(
                            RFF2(logG, W, factor, bX)
                        ))
                        test_loss += loss(out, by)
                        steps_in_epoch += 1
                        start_idx += BATCH_SIZE_OUT
                    test_loss /= steps_in_epoch
                    tall_stats += [(
                        oi+1, tt,
                        train_loss.item(), train_obj, # g-stuff
                        test_loss.item(), # f-te-stuff
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
    all_stats_df = pd.DataFrame(all_stats, columns=all_stats_col)
    delta_stats_df = pd.DataFrame(delta_stats, columns=delta_stats_col)
    tall_stats_df = pd.DataFrame(tall_stats, columns=tall_stats_col)

    return all_stats_df, delta_stats_df, tall_stats_df




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', choices=['MNIST', 'FashionMNIST'], help='Data set to use')
    parser.add_argument(
        '--path_to_data', '-Z', type=str, default='/home/pari/data/torchvision/', help='Path to load/save data'
    )
    parser.add_argument('--nobjs', '-t', type=int, default=5, help='Number of objectives for optimization')
    parser.add_argument('--ntobjs', '-T', type=int, default=0, help='Number of unseen objectives')
    parser.add_argument('--int_dim', '-D', type=int, default=32, help='Number of RFFs')
    parser.add_argument(
        '--in_lrate', '-L', type=float, default=0.001, help='Initial learning rate for inner level'
    )
    parser.add_argument(
        '--out_lrate', '-l', type=float, default=0.0001, help='Initial learning rate for outer level'
    )
    parser.add_argument(
        '--simplex_lrate', '-Y', type=float, default=0.01, help='Initial learning rate for outer level'
    )
    parser.add_argument(
        '--lr_decay', '-y', type=float, default=0.5, help='Factor to reduce learning rate by'
    )
    parser.add_argument(
        '--lr_patience', '-z', type=int, default=20, help='Patience for LR scheduler'
    )
    parser.add_argument('--inner_loop', '-I', type=int, default=1, help='Number of inner level iterations')
    parser.add_argument('--max_outer_loop', '-O', type=int, default=100, help='Max outer level iters')
    parser.add_argument('--inner_p', '-P', type=int, default=2, help='Norm order for Reg. for inner level')
    parser.add_argument('--inner_batch_size', '-B', type=int, default=32, help='Batch size for inner level')
    parser.add_argument('--outer_batch_size', '-b', type=int, default=128, help='Batch size for outer level')
    parser.add_argument('--minmax', '-M', action='store_true', help='Minmax version')
    parser.add_argument('--random_seed', '-S', type=int, default=5489, help='Random seed for RNG')
    parser.add_argument(
        '--full_stats_per_iter', '-F', type=int, default=10, help='Save full stats every this iters'
    )
    parser.add_argument('--tolerance', '-x', type=float, default=1e-7, help='Tolerance of optimization')
    parser.add_argument('--tobj_max_epochs', '-E', type=int, default=100, help='Max epochs for test tasks')
    parser.add_argument('--output_dir', '-U', type=str, default='', help='Directory to save results in')
    parser.add_argument(
        '--precompute_random_projections', '-R', action='store_true',
        help='Whether to precompute the random projections for the RFF'
    )

    args = parser.parse_args()
    expt_tag = args2tag(parser, args)
    logger.info(f'Experiment tag: {expt_tag}')

    assert os.path.isdir(args.path_to_data)
    assert os.path.isdir(args.output_dir) or args.output_dir == ''
    assert args.nobjs > 1
    assert args.ntobjs >= 0
    assert args.int_dim > 10
    assert args.in_lrate > 0.
    assert args.out_lrate > 0.
    assert args.simplex_lrate > 0.
    assert 0. < args.lr_decay < 1.
    assert args.lr_patience > 0
    assert args.inner_loop > 1
    assert args.max_outer_loop > 1
    assert args.inner_p in [1, 2]
    assert args.inner_batch_size > 1 and args.outer_batch_size > 1
    assert args.full_stats_per_iter > 1
    assert args.tolerance > 0.
    assert 1 <= args.tobj_max_epochs <= 10

    RNG = np.random.RandomState(args.random_seed)
    torch.manual_seed(args.random_seed)
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
    # RANDOM CONFIG
    # tidxs = np.arange(len(all_tasks))
    # RNG.shuffle(tidxs)
    # tasks = [all_tasks[tidxs[i]] for i in range(args.nobjs)]
    # ttasks = [all_tasks[tidxs[i]] for i in range(args.nobjs, args.nobjs+args.ntobjs)]
    # Config #1: D = 100
    # d:FashionMNIST_F:5_L:0.01_B:32_I:32_D:100_y:0.8_O:1000_l:0.01_b:128_S:54833779_x:1e-07
    # Config #2: D = 1000
    # d:FashionMNIST_F:5_L:0.01_B:32_I:32_D:1000_y:0.8_O:1000_l:0.01_b:128_S:54833779_x:1e-07
    tasks = [
        (0, 9), # EASY
        (1, 7), # EASY
        (1, 9), # EASY
        (2, 4), # HARD
        (2, 6), # HARD
        (2, 7), # EASY
        (3, 7), # EASY
        (3, 9), # EASY
        (4, 7), # EASY
        (4, 9), # EASY
    ]
    ttasks = [
        (0, 6), # HARD
        (0, 7), # EASY
        (1, 5), # EASY
        (2, 9), # EASY
        (4, 6), # HARD
        (6, 7), # EASY
        (6, 9), # EASY
    ]
    NCLASSES = 2
    logger.info(f"Performing {'minmax' if args.minmax else 'average'} optimization with the following tasks:")
    logger.info(f"- Tasks: {tasks}")
    logger.info(f"To be evaluated with the following tasks:")
    logger.info(f"- Tasks: {ttasks}")
    task_data = [get_task_data(full_data, t, val=True) for t in tasks]
    ttask_data = [get_task_data(full_data, tt, val=False) for tt in ttasks]
    # normalize the data for better scaling for RFF
    for td in task_data + ttask_data:
        X, _ = td['train']
        max_X = torch.max(X).item()
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
        max_X = torch.max(X).item()
        tX, _ = td['test']
        max_tX = torch.max(tX).item()
        logger.info(f'Task {t} -- max X: {max_X:.4f}, max tX: {max_tX:.4f}')
        if 'val' in td:
            vX, _ = td['val']
            max_vX = torch.max(vX).item()
            logger.info(f'Task {t} -- max vX: {max_vX:.4f}')
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
        INNORM=args.inner_p,
        MINMAX=args.minmax,
        FULL_STATS_PER_ITER=args.full_stats_per_iter,
        TOL=args.tolerance,
        TOBJ_MAX_EPOCHS=args.tobj_max_epochs,
        PRECOMPUTE_RPS=args.precompute_random_projections,
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
