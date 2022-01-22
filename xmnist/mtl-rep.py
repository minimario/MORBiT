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

from utils import get_data, get_task_data, simplex_proj_inplace, wnorm

import logging
logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger('MTL-REP')
logger.setLevel(logging.INFO)

# NCLASSES = 2
# tasks = [
#     (0, 4),
#     (3, 8),
#     (4, 7),
#     (5, 8),
#     (1, 7),
#     (4, 9),
# ]
# test_tasks = [
#     (2, 6),
#     (0, 5),
#     (3, 5),
#     (7, 9),
#     (0, 1),
# ]
#
# IN_DIM = in_dim
# INT_DIM = 32
# BATCH_SIZE = 32
# LRATE = 0.001
# IN_ITER = 1
# OUT_ITER = 1000
# NLIN = True
# INREG = 0.00 if NLIN else 0.001
# OUTREG = 0.00 if NLIN else 0.0001

def run_opt(
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
        LRATE_OUT=0.0001,
        LRATE_SIMP=0.1,
        LR_DECAY=0.8,
        IN_ITER=1,
        OUT_ITER=1000,
        NLIN=False,
        INREG=0.001,
        OUTREG=0.0001,
        INNORM=2,
        OUTNORM=2,
        MINMAX=False,
        FULL_STATS_PER_ITER=10,
        TOL=1e-7,
        TOBJ_MAX_EPOCHS=1000,
        
):

    loss = nn.CrossEntropyLoss()

    # OUTER LEVEL TASKS
    outer_var = nn.Linear(IN_DIM, INT_DIM, bias=False)
    out_opt = torch.optim.SGD(
        outer_var.parameters(),
        lr=LRATE_OUT,
        # momentum=0.9,
        # weight_decay=1e-4,
    )
    out_sched = torch.optim.lr_scheduler.StepLR(
        out_opt, step_size=30, gamma=LR_DECAY, verbose=False
    )

    # INNER LEVEL TASKS
    ntasks = len(TASKS)
    inner_vars = [nn.Linear(INT_DIM, NCLASSES) for _ in TASKS]
    in_opts, in_scheds = [], []
    for iv in inner_vars:
        in_opt = torch.optim.SGD(
            iv.parameters(),
            lr=LRATE,
            # momentum=0.9,
            # weight_decay=1e-4,
        )
        in_sched = torch.optim.lr_scheduler.StepLR(
            in_opt, step_size=30, gamma=LR_DECAY, verbose=False
        )
        in_opts += [in_opt]
        in_scheds += [in_sched]

    # OPTIONAL: simplex vars
    simplex_vars, simplex_out, simplex_sched = None, None, None
    simplex_vars = torch.Tensor([1./ntasks for i in range(ntasks)])
    if MINMAX:
        simplex_vars.requires_grad = True
        simplex_opt = torch.optim.SGD(
            [simplex_vars],
            lr=LRATE_SIMP,
            # lr=(LRATE_OUT / np.sqrt(ntasks)),
        )
        simplex_sched = torch.optim.lr_scheduler.StepLR(
            simplex_opt, step_size=30, gamma=LR_DECAY, verbose=False
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
        'task', # task ID for inner_var or ALL for outer_var
        'delta', # change in variables
    ]
    delta_stats = []

    # statistics for tasks not seen during optimization
    tall_stats_col = [
        'oiter', # outer iter
        'task', # task ID
        'ntrain', # number of training examples for unseen tasks
        'g-loss', # inner level loss at convergence (full)
        'g-obj', # inner level obj at convergence (full)
        'f-loss', # outer level loss on unseen data at convergence (full)
        'f-obj', # outer level obj at convergence (full)
    ]
    tall_stats = []

    bout_loss = np.inf
    bout_obj = np.inf
    bl_iter = 0
    bo_iter = 0

    for oi in range(OUT_ITER):
        logger.info(f'Starting outer loop {oi+1}/{OUT_ITER} ...')
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
        outer_old = outer_var.weight.clone()
        # Looping over all the tasks
        for t, tdata, tw, topt, tsch, tlambda in zip(
                TASKS, TASK_DATA, inner_vars, in_opts, in_scheds, simplex_vars
        ):
            logger.debug(f'[{oi+1}/{OUT_ITER}] Task {t} ....')
            tsize = tdata['train-size']
            X, y = tdata['train']
            total_loss = 0.
            total_obj = 0.
            inner_old = tw.weight.clone()
            for ii in range(IN_ITER):
                topt.zero_grad()
                logger.debug(f'[{oi+1}/{OUT_ITER}] [{t}] ({ii+1}/{IN_ITER}) Train size: {tsize}')
                bidxs = [RNG.randint(0, tsize) for _ in range(BATCH_SIZE)]
                logger.debug(f'     Batch: {bidxs}')
                bX = X[bidxs]
                by = y[bidxs]
                logger.debug(f'     {bX.shape}')
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
                # break
            # if not batch_stats:
            tsch.step()
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
            out = in_softmax(tw(out_nlin(outer_var(bX)))) if NLIN else in_softmax(tw(outer_var(bX)))
            toloss = loss(out, by)
            toobj = toloss.item()
            if OUTREG > 0.0:
                toobj += (OUTREG * wnorm(outer_var, OUTNORM)).item()
            # Computing outer level accuracy on batch
            _, preds = torch.max(out, 1)
            out_accs += [(by == preds).sum() / bX.size(0)]
            # collect opt stats
            all_stats += [(
                oi+1, True, t,
                total_loss, total_obj, # g-stuff
                toloss.item(), toobj, # f-val-stuff
                np.nan, np.nan #f-te-stuff
            )]
            outer_loss += (tlambda * toloss)
            out_losses += [toloss.item()]
            # collect delta stats
            in_delta = torch.linalg.norm(tw.weight - inner_old)
            in_deltas += [in_delta.item()]
            delta_stats += [(oi+1, t, in_delta.item())]

        # print some stats
        if not batch_stats:
            logger.info(f'[{oi+1}/{OUT_ITER}] IN LOSSES: {np.array(in_losses)}')
            logger.info(f'[{oi+1}/{OUT_ITER}] IN OBJS: {np.array(in_objs)}')
            logger.info(f'[{oi+1}/{OUT_ITER}] IN-Deltas: {np.array(in_deltas)}')
            logger.info(f'[{oi+1}/{OUT_ITER}] OUT LOSS: {np.array(out_losses)}')
            logger.info(f'[{oi+1}/{OUT_ITER}] OUT ACCS: {np.array(out_accs)}')
        logger.info(
            f'[{oi+1}/{OUT_ITER}] OUTER LOSS: {outer_loss.item():.8f} '
            f'(best: {bout_loss:.8f} ({bl_iter}/{OUT_ITER}))'
        )
        if outer_loss.item() < bout_loss:
            logger.info(f'OUT: {np.array(out_losses)}')
            bout_loss = outer_loss.item()
            bl_iter = oi + 1

        # Add regularization and take opt step
        if OUTREG > 0.0:
            outer_loss += (OUTREG * wnorm(outer_var, OUTNORM))
        logger.info(
            f'[{oi+1}/{OUT_ITER}] OUTER LOSS + REG: {outer_loss:.8f} '
            f'(best: {bout_obj:.8f} ({bo_iter}/{OUT_ITER}))'
        )
        if outer_loss.item() < bout_obj:
            bout_obj = outer_loss.item()
            bo_iter = oi + 1
        outer_loss.backward(retain_graph=True)
        out_opt.step()
        out_delta = torch.linalg.norm(outer_var.weight - outer_old)
        logger.info(f'[{oi+1}/{OUT_ITER}] OUT Delta: {out_delta.item():.6f}')
        # if not batch_stats:
        out_sched.step()
        logger.info(f'[{oi+1}/{OUT_ITER}] Lambdas: {simplex_vars}')
        lambda_delta = 0.
        if MINMAX:
            # negate gradient for gradient ascent
            simplex_vars.grad *= -1.
            logger.debug(f'[{oi+1}/{OUT_ITER}] - Lambda grads: {simplex_vars.grad}')
            simplex_opt.step()
            logger.info(f'[{oi+1}/{OUT_ITER}] - Updated Lambdas: {simplex_vars}')
            # if not batch_stats:
            simplex_sched.step()
            simplex_proj_inplace(simplex_vars)
            logger.info(f'[{oi+1}/{OUT_ITER}] - Simplex projected updated Lambdas: {simplex_vars}')
            lambda_delta = torch.linalg.norm(simplex_vars - old_simplex_vars)
            logger.info(f'[{oi+1}/{OUT_ITER}] - lambda delta: {lambda_delta.item():.6f}')
            simplex_vars.requires_grad = True
        delta_stats += [(oi+1, 'ALL', out_delta.item())]
        in_delta_sum = np.sum(in_deltas)
        converged = (in_delta_sum < TOL) and (out_delta < TOL)
        if (not batch_stats) or (converged):
            # compute stats on full data
            voobjs, toobjs = [], []
            for t, tdata, tw in zip(TASKS, TASK_DATA, inner_vars):
                logger.info(f'[{oi+1}/{OUT_ITER}] Task {t} full-stats comp ....')
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
            logger.info(f'[{oi+1}/{OUT_ITER} Full outer stats:')
            logger.info(f'[{oi+1}/{OUT_ITER} val: {np.array(voobjs)}')
            logger.info(f'[{oi+1}/{OUT_ITER} test: {np.array(toobjs)}')
            # compute opt & stats for unseen tasks
            for tt, ttdata in zip(TTASKS, TTASK_DATA):
                logger.info(f'[{oi+1}/{OUT_ITER}] Unseen task {tt} full-stats comp ....')
                max_ntrain = ttdata['train-size']
                # loop over different training set sizes
                for ntrain in [32, 128]:
                    if ntrain <= BATCH_SIZE or ntrain > max_ntrain:
                        continue
                    # INNER LEVEL TEST TASKS
                    iv = nn.Linear(INT_DIM, NCLASSES)
                    tin_opt = torch.optim.SGD(
                        iv.parameters(),
                        lr=LRATE,
                        # momentum=0.9,
                        # weight_decay=1e-4,
                    )
                    tin_sched = torch.optim.lr_scheduler.StepLR(
                        tin_opt, step_size=30, gamma=LR_DECAY, verbose=False
                    )
                    X, y = ttdata['train']
                    tidxs = np.arange(max_ntrain)
                    RNG.shuffle(tidxs)
                    X1, y1 = X[tidxs[:ntrain]], y[tidxs[:ntrain]]
                    logger.info(f"- Training unseen task {tt} with data of size {X1.shape}")
                    with torch.no_grad():
                        rX1 = out_nlin(outer_var(X1)) if NLIN else outer_var(X1)
                    # Optimize inner var for unseen new task for fixed outer var
                    epoch_loss = 0.0
                    for eidx in range(TOBJ_MAX_EPOCHS):
                        start_idx = 0
                        logger.debug(
                            f'[{oi+1}/{OUT_ITER}] [UNSEEN {tt}] ({eidx+1}/{TOBJ_MAX_EPOCHS}) Train size: {ntrain}'
                        )
                        old_iv = iv.weight.clone()
                        epoch_loss = 0.0
                        steps_in_epoch = 0
                        while start_idx < ntrain:
                            tin_opt.zero_grad()
                            bidxs = np.arange(start_idx, min(start_idx + BATCH_SIZE, ntrain))
                            logger.debug(f'     Batch: {bidxs}')
                            bX = rX1[bidxs]
                            by = y1[bidxs]
                            logger.debug(f'     {bX.shape}')
                            probs = in_softmax(iv(bX))
                            btloss = loss(probs, by)
                            if INREG > 0.0:
                                btloss += (INREG * wnorm(iv, INNORM))
                            btloss.backward(retain_graph=True)
                            tin_opt.step()
                            start_idx += BATCH_SIZE
                            epoch_loss += btloss.item()
                            steps_in_epoch += 1
                        epoch_loss /= steps_in_epoch
                        tin_sched.step()
                        in_delta = torch.linalg.norm(iv.weight - old_iv)
                        if in_delta < TOL:
                            logger.info(
                                f'[{oi+1}/{OUT_ITER}] [UNSEEN {tt}] ({eidx+1}/{TOBJ_MAX_EPOCHS})'
                                f'Train size: {ntrain} '
                                f'exiting opt with delta: {in_delta:.8f}'
                            )
                            break
                    logger.info(
                        f'[{oi+1}/{OUT_ITER}] [UNSEEN {tt}] Train size: {ntrain} '
                        f' Concluded with epoch loss: {epoch_loss:.6f}'
                    )
                    # evaluate full train/test loss/obj
                    tX, ty = ttdata['test']
                    with torch.no_grad():
                        train_out = in_softmax(iv(rX1))
                        train_loss = loss(train_out, y1)
                        train_obj = train_loss.item()
                        if INREG > 0.0:
                            train_obj += (INREG * wnorm(iv, INNORM)).item()
                        test_out = in_softmax(
                            iv(out_nlin(outer_var(tX)))
                        ) if NLIN else in_softmax(
                            iv(outer_var(tX))
                        )
                        test_loss = loss(test_out, ty)
                        test_obj = test_loss.item()
                        if OUTREG > 0.0:
                            test_obj += (OUTREG * wnorm(outer_var, OUTNORM)).item()
                        tall_stats += [(
                            oi+1, tt, ntrain,
                            train_loss.item(), train_obj, # g-stuff
                            test_loss.item(), test_obj, # f-te-stuff
                        )]

        if converged:
            logger.warning(
                f"[{oi+1}/{OUT_ITER}] Exiting optimization with sum(IN-DELTAs): {in_delta_sum:.8f}, "
                f"OUT-DELTA: {out_delta:.8f}"
            )
            break

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
    parser.add_argument('--int_dim', '-D', type=int, default=32, help='Latent space dimensionality')
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
    parser.add_argument('--inner_loop', '-I', type=int, default=1, help='Number of inner level iterations')
    parser.add_argument('--max_outer_loop', '-O', type=int, default=100, help='Max outer level iters')
    parser.add_argument('--inner_reg', '-R', type=float, default=0.001, help='Reg. for inner level')
    parser.add_argument('--outer_reg', '-r', type=float, default=0.0001, help='Reg. for outer level')
    parser.add_argument('--inner_p', '-P', type=int, default=2, help='Norm order for Reg. for inner level')
    parser.add_argument('--outer_p', '-p', type=int, default=2, help='Norm order for Reg. for outer level')
    parser.add_argument('--inner_batch_size', '-B', type=int, default=32, help='Batch size for inner level')
    parser.add_argument('--outer_batch_size', '-b', type=int, default=128, help='Batch size for outer level')
    parser.add_argument('--nonlinear', '-N', action='store_true', help='Nonlinear version')
    parser.add_argument('--minmax', '-M', action='store_true', help='Minmax version')
    parser.add_argument('--random_seed', '-S', type=int, default=5489, help='Random seed for RNG')
    parser.add_argument(
        '--full_stats_per_iter', '-F', type=int, default=10, help='Save full stats every this iters'
    )
    parser.add_argument('--tolerance', '-x', type=float, default=1e-7, help='Tolerance of optimization')
    parser.add_argument('--tobj_max_epochs', '-E', type=int, default=100, help='Max epochs for test tasks')
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
        f'{odict[k]}-{args.__dict__[k]}'
        for k in sorted(args.__dict__.keys())
        if k in odict
    ])
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
    assert args.inner_loop > 1
    assert args.max_outer_loop > 1
    assert args.inner_p in [1, 2] and args.outer_p in [1, 2]
    assert args.inner_batch_size > 1 and args.outer_batch_size > 1
    assert args.full_stats_per_iter > 1
    assert args.tolerance > 0.
    assert args.tobj_max_epochs > 1

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
    tidxs = np.arange(len(all_tasks))
    RNG.shuffle(tidxs)
    tasks = [all_tasks[tidxs[i]] for i in range(args.nobjs)]
    ttasks = [all_tasks[tidxs[i]] for i in range(args.nobjs, args.nobjs+args.ntobjs)]
    NCLASSES = 2
    logger.info(f"Performing {'minmax' if args.minmax else 'average'} optimization with the following tasks:")
    logger.info(f"- Tasks: {tasks}")
    logger.info(f"To be evaluated with the following tasks:")
    logger.info(f"- Tasks: {ttasks}")
    task_data = [get_task_data(full_data, t, val=True) for t in tasks]
    ttask_data = [get_task_data(full_data, tt, val=False) for tt in ttasks]
    orig_dim = task_data[0]['train'][0].shape[1]
    logger.info(f"Starting with original input dim: {orig_dim} ...")

    output_dir = os.path.join(args.output_dir, expt_tag)
    if args.output_dir != '':
        assert not os.path.exists(output_dir)

    astats, dstats, tastats = run_opt(
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
        IN_ITER=args.inner_loop,
        OUT_ITER=args.max_outer_loop,
        NLIN=args.nonlinear,
        INREG=args.inner_reg,
        OUTREG=args.outer_reg,
        INNORM=args.inner_p,
        OUTNORM=args.outer_p,
        MINMAX=args.minmax,
        FULL_STATS_PER_ITER=args.full_stats_per_iter,
        TOL=args.tolerance,
        TOBJ_MAX_EPOCHS=args.tobj_max_epochs,
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
