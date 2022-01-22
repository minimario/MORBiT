import numpy as np
np.set_printoptions(precision=4)
import torch
torch.set_printoptions(precision=4)
from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST

import sys
import os
import logging
logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger('UTILS')
logger.setLevel(logging.INFO)


def pull_data(dfunc, path_to_data):

    ret = []
    for train in [True, False]:
        try:
            logger.info(
                f"Attempting to load {'train' if train else 'test'} data from {path_to_data} ..."
            )
            d = dfunc(
                root=path_to_data,
                train=train,
                download=False,
                transform=transforms.ToTensor()
            )
        except Exception as e:
            logger.info(
                f"Can't load {'train' if train else 'test'} data from {path_to_data} ..."
                f" .. downloading to {path_to_data}."
            )
            d = dfunc(
                root=path_to_data,
                train=train,
                download=True,
                transform=transforms.ToTensor()
            )
        ret += [d]
    assert len(ret) == 2
    return {'train': ret[0], 'test': ret[1]}


def get_data(dname, path_to_data):
    assert dname in ['MNIST', 'FashionMNIST']
    assert os.path.isdir(path_to_data)
    if dname == 'MNIST':
        return pull_data(MNIST, path_to_data)
    elif dname == 'FashionMNIST':
        return pull_data(FashionMNIST, path_to_data)


def get_task_data(ddict, task, val):
    assert 'train' in ddict.keys() and 'test' in ddict.keys()
    assert len(task) == 2
    c1, c2 = task
    tdict = {}
    in_dim = None
    for k, v in ddict.items():
        idxs = (v.targets == c1) | (v.targets == c2)
        task_X = torch.flatten(v.data[idxs], start_dim=1).float()
        task_y = v.targets[idxs]
        c1_idxs = task_y == c1
        c2_idxs = task_y == c2
        task_y[c1_idxs] = 0
        task_y[c2_idxs] = 1
        if in_dim is None:
            in_dim = task_X.shape[1]
            tdict['dim'] = in_dim
        if k == 'test' and val:
            n = task_X.shape[0]
            sidxs = np.arange(n)
            np.random.shuffle(sidxs)
            tidxs = sidxs[:n//2]
            vidxs = sidxs[n//2:]
            task_vX, task_vy = task_X[vidxs], task_y[vidxs]
            task_X, task_y = task_X[tidxs], task_y[tidxs]
            logger.info(f'Set validation, size: {task_vX.shape}')
            tdict['val'] = (task_vX, task_vy)
            tdict['val-size'] = task_vX.shape[0]
        logger.info(f'Set {k}, size: {task_X.shape}')
        tdict[f'{k}-size'] = task_X.shape[0]
        tdict[k] = (task_X, task_y)
    return tdict


def simplex_proj_inplace(v, z=1):
    with torch.no_grad():
        shape = v.shape[0]
        if shape == 1:
            v[0] = z
            return
        mu = torch.sort(v)[0]
        mu = torch.flip(mu, dims=(0,))
        cum_sum = torch.cumsum(mu, dim=0)
        j = torch.arange(1, shape + 1, dtype=mu.dtype)
        rho = torch.sum(mu * j - cum_sum + z > 0.0) - 1
        max_nn = cum_sum[rho]
        theta = (max_nn - z) / (rho.type(max_nn.dtype) + 1)
        v -= theta
        v.clamp_(min=0.0)


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


if __name__ == '__main__':
    logger.info(f'Running data fetching tools ...')
    dpath = '/home/pari/data/torchvision/'
    for dname in ['MNIST', 'FashionMNIST']:
        ddict = get_data(dname, dpath)
        for k, v in ddict.items():
            logger.info(f"Data: {dname}+{k}")
            logger.info(f'- Features: {v.data.shape}')
            logger.info(f'- Labels: {v.targets.shape}')
            logger.info(f'  - {np.unique(v.targets.numpy())}')
        task = (4, 7)
        td = get_task_data(ddict, task, dname=='MNIST')
        for k, v in td.items():
            if isinstance(v, tuple):
                X, y = v
                logger.info(f'Task: {task}, set: {k}, sizes: ({X.shape}, {y.shape})')
                logger.info(f'Task: {task}, set: {k}, labels: {np.unique(y.numpy(), return_counts=True)}')
                continue
            logger.info(f'Task: {task}, {k}: {v}')
    logger.info(f'Evaluating simplex projection ...')
    for v in [
            torch.Tensor([0.2, 0.4, 0.3]),
            torch.Tensor([0.7, 0.4, 0.3]),
            torch.Tensor([-0.1, 1.4, 0.5]),
            torch.Tensor([0.2])
    ]:
        logger.info(f'In: {v}')
        simplex_proj_inplace(v, z=1)
        logger.info(f'Out: {v}')
