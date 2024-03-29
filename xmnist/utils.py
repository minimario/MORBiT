import numpy as np
np.set_printoptions(precision=4)
import torch
torch.set_printoptions(precision=4)
from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST, CIFAR100
from openml.datasets import get_dataset
from sklearn.model_selection import train_test_split


import sys
import os
import logging
logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger('UTILS')
logger.setLevel(logging.INFO)


def args2tag(parser, args):
    logger.info(f'Generating experiment tag ...')
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
    expt_tag = '_'.join([
        f'{odict[k]}:{args.__dict__[k]}'
        for k in sorted(args.__dict__.keys())
        if k in odict
    ])
    logger.info(f'... experiment tag generated')
    return expt_tag

def pull_data(dfunc, path_to_data, totensor=True):

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
                transform=(transforms.ToTensor() if totensor else None),
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
                transform=transforms.ToTensor() if totensor else None,
            )
        ret += [d]
    assert len(ret) == 2
    return {'train': ret[0], 'test': ret[1]}


def get_data(dname, path_to_data, totensor=True):
    assert dname in ['MNIST', 'FashionMNIST', 'CIFAR100', 'Letter']
    assert os.path.isdir(path_to_data) or dname in ['Letter']
    if dname == 'MNIST':
        return pull_data(MNIST, path_to_data, totensor=totensor)
    elif dname == 'FashionMNIST':
        return pull_data(FashionMNIST, path_to_data, totensor=totensor)
    elif dname == 'CIFAR100':
        return pull_data(CIFAR100, path_to_data, totensor=totensor)
    elif dname == 'Letter':
        d = get_dataset(6)
        X, y, C, N = d.get_data(target=d.default_target_attribute, dataset_format='array')
        assert not any(C)
        assert len(np.unique(y)) == 26
        assert X.shape == (20000, 16)
        X1, X2, y1, y2 = train_test_split(X, y, test_size=0.4, stratify=y)
        class Data:
            data: torch.Tensor
            targets: torch.Tensor
        
        TRAIN = Data()
        TRAIN.data = torch.tensor(X1)
        TRAIN.targets = torch.tensor(y1, dtype=torch.long)
        TEST = Data()
        TEST.data = torch.tensor(X2)
        TEST.targets = torch.tensor(y2, dtype=torch.long)
        return {
            'train': TRAIN,
            'test': TEST,
        }
        



def get_task_data(ddict, task, val, train_val_size=0):
    assert 'train' in ddict.keys() and 'test' in ddict.keys()
    assert len(task) == 2
    c1, c2 = task
    tdict = {}
    in_dim = None
    for k, v in ddict.items():
        idxs = (v.targets == c1) | (v.targets == c2)
        task_X = torch.flatten(v.data[idxs], start_dim=1).float()
        task_y = v.targets[idxs]
        if train_val_size > 0 and k == 'train':
            X1, X2, y1, y2 = train_test_split(
                task_X, task_y, train_size=train_val_size, stratify=task_y
            )
            task_X, task_y = X1, y1
        c1_idxs = task_y == c1
        c2_idxs = task_y == c2
        task_y[c1_idxs] = 0
        task_y[c2_idxs] = 1
        if in_dim is None:
            in_dim = task_X.shape[1]
            tdict['dim'] = in_dim
        if k == 'test' and val:
            ts = train_val_size if train_val_size > 0 else 0.5
            X1, X2, y1, y2 = train_test_split(
                task_X, task_y, test_size=ts, stratify=task_y
            )
            logger.info(f'Set validation, size: {X2.shape}')
            tdict['val'] = (X2, y2)
            tdict['val-size'] = X2.shape[0]
            task_X, task_y = X1, y1
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
    for dname in ['MNIST', 'Letter']:
        ddict = get_data(dname, dpath)
        for k, v in ddict.items():
            logger.info(f"Data: {dname}+{k}")
            logger.info(f'- Features: {v.data.shape}')
            logger.info(f'- Labels: {v.targets.shape}')
            logger.info(f'  - {np.unique(v.targets.numpy())}')
        task = (4, 7)
        td = get_task_data(
            ddict, task, False
        ) if dname == 'MNIST' else get_task_data(
            ddict, task, True, train_val_size=20
        )
        for k, v in td.items():
            if isinstance(v, tuple):
                X, y = v
                logger.info(f'Task: {task}, set: {k}, sizes: ({X.shape}, {y.shape})')
                logger.info(
                    f'Task: {task}, set: {k}, '
                    f'labels: {np.unique(y.numpy(), return_counts=True)}'
                )
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
        assert np.isclose(v.sum().item(), 1.0)
        logger.info(f'Out: {v}')
    
    for v in [
            torch.Tensor([0.2, 0.4, 0.3]),
            torch.Tensor([0.7, 0.4, 0.3]),
            torch.Tensor([-0.1, 1.4, 0.5]),
            torch.Tensor([0.2])
    ]:
        logger.info(f'In: {v}')
        simplex_proj_inplace(v, z=0.8)
        assert np.isclose(v.sum().item(), 0.8)
        logger.info(f'Out: {v}')
