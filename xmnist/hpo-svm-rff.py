import numpy as np
np.set_printoptions(precision=4)
import torch
torch.set_printoptions(precision=4)
from torch import nn
from torchvision import transforms
from torchvision.datasets import MNIST

import sys
import logging
logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger()
logger.setLevel(logging.INFO)


mnist_train = MNIST(
    root='~/data/torchvision/',
    train=True,
    download=False,
    transform=transforms.ToTensor()
)
mnist_test = MNIST(
    root='~/data/torchvision/',
    train=False,
    download=False,
    transform=transforms.ToTensor()
)

NCLASSES = 2
tasks = [
    (0, 5),
    (4, 9),
    (2, 6),
]

full_data = {'train': mnist_train, 'val': mnist_test}
task_data = []

for c1, c2 in tasks:
    tdict = {}
    for k, v in full_data.items():
        idxs = (v.targets == c1) | (v.targets == c2)
        task_X = torch.flatten(v.data[idxs], start_dim=1).float() / 255.
        task_y = v.targets[idxs]
        c1_idxs = task_y == c1
        c2_idxs = task_y == c2
        task_y[c1_idxs] = 0
        task_y[c2_idxs] = 1
        logger.info(f'Set {k}, size: {task_X.shape}')
        tdict[k] = (task_X, task_y)
    task_data += [tdict]

in_dim = None
for t, td in zip(tasks, task_data):
    td1 = {}
    for k, (X, y) in td.items():
        if in_dim is None:
            in_dim = X.shape[1]
        logger.info(f'Task: {t}, set: {k}, sizes: ({X.shape}, {y.shape})')
        logger.info(f'Task: {t}, set: {k}, labels: {np.unique(y.numpy())}')
        td1[f'{k}-size'] = X.shape[0]
    for k, v in td1.items():
        td[k] = v

logger.info(f'Data dimensionality: {in_dim}')

IN_DIMS = [in_dim, in_dim, in_dim]
INT_DIMS = [10000, 10000, 10000]
BATCH_SIZE = 64
LRATE = 1.
IN_ITER = 10000
OUT_ITER = 1
LOGLOSS = True

def wnorm(w, p: int):
    assert p in [1, 2]
    if p == 2:
        # l2 norm
        return torch.square(torch.linalg.norm(w.weight))
    elif p == 1:
        # l1 norm
        return torch.linalg.norm(torch.linalg.norm(w.weight, ord=1, dim=1), ord=1)
    else:
        raise Exception(f'Unknown p={p} for norm')


inner_vars = [nn.Linear(2 * idim, NCLASSES) for _, idim in zip(tasks, INT_DIMS)]
in_opts, in_scheds = [], []
for iv in inner_vars:
    in_opt = torch.optim.SGD(
        iv.parameters(),
        lr=LRATE,
        momentum=0.9,
        weight_decay=1e-4,
    )
    in_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        in_opt, 'min', factor=0.5, patience=20, cooldown=5, verbose=True,
    )
    in_opts += [in_opt]
    in_scheds += [in_sched]

in_sms = [nn.Softmax(dim=1) for _ in tasks] if LOGLOSS else [None] * len(tasks)

loss = nn.CrossEntropyLoss(reduction='mean')


logC = torch.tensor(1.0)
logC.requires_grad = True
opt_logC = torch.optim.SGD(
    [logC],
    lr=0.0001,
    momentum=0.9,
    weight_decay=1e-4,
)
sched_logC = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt_logC, 'min', factor=0.5, patience=30, cooldown=5, verbose=True,
)
logG = torch.tensor(1.0)
logG.requires_grad = True
opt_logG = torch.optim.SGD(
    [logG],
    lr=0.0001,
    momentum=0.9,
    weight_decay=1e-4,
)
sched_logG = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt_logG, 'min', factor=0.5, patience=30, cooldown=5, verbose=True,
)

PROJS = [
    (1/ np.sqrt(in_d)) * torch.normal(0., 1., size=(in_d, int_d))
    for _, in_d, int_d in zip(tasks, IN_DIMS, INT_DIMS)
]
for proj in PROJS:
    proj.requires_grad = False

current_best_outer = np.inf
current_best_total = np.inf
best_iter = 0
best_iter_total = 0

for oi in range(OUT_ITER):
    assert logG.requires_grad and logC.requires_grad
    logger.info(f'Starting outer loop {oi+1}/{OUT_ITER} ...')
    opt_logC.zero_grad()
    opt_logG.zero_grad()
    out_losses = []
    out_accs = []
    in_losses = []
    in_deltas = []
    outer_loss = 0.0
    outer_old = [logC.item(), logG.item()]
    for t, tdata, tw, tsmax, topt, tsch, proj, int_d in zip(
            tasks, task_data, inner_vars, in_sms, in_opts, in_scheds, PROJS, INT_DIMS,
    ):
        logger.debug(f'[{oi+1}/{OUT_ITER}] Task {t} ....')
        tsize = tdata['train-size']
        X, y = tdata['train']
        total_loss = 0.
        inner_old = tw.weight.clone()
        factor = (1.0 / np.sqrt(int_d))
        for ii in range(IN_ITER):
            topt.zero_grad()
            logger.debug(f'[{oi+1}/{OUT_ITER}] [{t}] ({ii+1}/{IN_ITER}) Train size: {tsize}'),
            bidxs = [np.random.randint(0, tsize) for _ in range(BATCH_SIZE)]
            logger.debug(f'     Batch: {bidxs}')
            bX = X[bidxs]
            by = y[bidxs]
            logger.debug(f'     Batch size: {bX.shape}')
            logger.debug(f'     Proj size : {proj.shape}')
            pre_int_rep = torch.exp(logG) * torch.matmul(bX, proj)
            int_rep = factor *  torch.concat((
                torch.cos(pre_int_rep),
                torch.sin(pre_int_rep)
            ), dim=1)
            logger.debug(f'     Intermediate dimensionality: {int_rep.shape}')
            out = tw(int_rep)
            probs = tsmax(out)
            btloss = loss(probs, by)
            btloss += (torch.exp(logC) * 0.5 * wnorm(tw, 2))
            btloss.backward(retain_graph=True)
            topt.step()
            total_loss += btloss
            # break
            if (ii + 1) % 100 == 0:
                tsch.step(total_loss)
                logger.info(f'{ii+1}/{IN_ITER} Total cum loss: {total_loss:.4f}')
                total_loss = 0.0
        in_losses += [total_loss]
        vX, vy = tdata['val']
        # TODO: Should the outer loss be on the full validation set or a batch of the validation set?
        pre_int_rep = torch.exp(logG) * torch.matmul(vX, proj)
        int_rep = factor * torch.concat((
            torch.cos(pre_int_rep),
            torch.sin(pre_int_rep)
        ), dim=1)
        out = tsmax(tw(int_rep))
        _, preds = torch.max(out, 1)
        out_accs += [(vy == preds).sum() / vX.size(0)]
        toloss = loss(out, vy)
        outer_loss += toloss
        out_losses += [toloss]
        in_deltas += [torch.linalg.norm(tw.weight - inner_old)]
        break
    if (oi+1)  % 25 == 0 or oi + 1 == OUT_ITER:
        logger.info(f'IN: {np.array(in_losses)}')
        logger.info(f'IN-Deltas: {np.array(in_deltas)}')
        logger.info(f'OUT: {np.array(out_losses)}')
        logger.info(f'OUT ACCS: {np.array(out_accs)}')
    logger.info(
        f'[{oi+1}/{OUT_ITER}] OUTER LOSS: {outer_loss:.8f} '
        f'(best: {current_best_outer:.8f} ({best_iter}/{OUT_ITER}))'
    )
    if outer_loss.item() < current_best_outer:
        logger.info(f'OUT: {np.array(out_losses)}')
        current_best_outer = outer_loss.item()
        best_iter = oi + 1
    outer_loss.backward(retain_graph=True)
    opt_logC.step()
    opt_logG.step()
    # if (oi + 1) % 10:
    #     sched_logC.step(outer_loss)
    #     sched_logG.step(outer_loss)
    logger.info(
        f'[{oi+1}/{OUT_ITER}] OUT logC: {logC.item():.6f}, '
        f'Delta(logC): {abs(logC.item() - outer_old[0]):.6f}'
    )
    logger.info(
        f'[{oi+1}/{OUT_ITER}] OUT logG: {logG.item():.6f}, '
        f'Delta(logG): {abs(logG.item() - outer_old[1]):.6f}'
    )
    # break

