import math

import torch
from torch.distributions import Laplace
from torch_geometric.data import Data

from datasets import load_dataset

available_mechanisms = {'bit', 'lap', 'pws'}


def laplace(data, eps):
    scale = torch.ones_like(data.x) * (data.delta / eps)
    x_priv = Laplace(data.x, scale).sample()
    data.x = data.priv_mask * x_priv + ~data.priv_mask * data.x
    return data


def one_bit(data, eps):
    exp = math.exp(eps)
    p = (data.x - data.alpha) / data.delta
    p[:, (data.delta == 0)] = 0
    p = p * (exp - 1) / (exp + 1) + 1 / (exp + 1)
    b = torch.bernoulli(p)
    # noinspection PyTypeChecker
    x_priv = ((b * (exp + 1) - 1) * data.delta) / (exp - 1) + data.alpha
    data.x = data.priv_mask * x_priv + ~data.priv_mask * data.x
    return data


def piecewise(data, eps):
    # normalize x between -1,1
    t = (data.x - data.alpha) / data.delta
    t[:, (data.delta == 0)] = 0
    t = 2 * t - 1

    # piecewise mechanism's variables
    P = (math.exp(eps) - math.exp(eps / 2)) / (2 * math.exp(eps / 2) + 2)
    C = (math.exp(eps / 2) + 1) / (math.exp(eps / 2) - 1)
    L = t * (C + 1) / 2 - (C - 1) / 2
    R = L + C - 1

    # thresholds for random sampling
    threshold_left = P * (L + C) / math.exp(eps)
    threshold_right = threshold_left + P * (R - L)

    # masks for piecewise random sampling
    x = torch.rand_like(t)
    mask_left = x < threshold_left
    mask_middle = (threshold_left < x) & (x < threshold_right)
    mask_right = threshold_right < x

    # random sampling
    t = mask_left * (torch.rand_like(t)*(L+C)-C)
    t += mask_middle * (torch.rand_like(t)*(R-L)+L)
    t += mask_right * (torch.rand_like(t)*(C-R)+R)

    # unbiased data
    x_priv = data.delta * (t + 1) / 2 + data.alpha
    data.x = data.priv_mask * x_priv + ~data.priv_mask * data.x
    return data


# noinspection PyTypeChecker
def privatize(data, pnr, pfr, eps, method='bit'):
    if pnr > 0 and pfr > 0:
        data = Data(**dict(data()))  # copy data to avoid changing the original
        mask = torch.zeros_like(data.x, dtype=torch.bool)
        n_rows = int(pnr * mask.size(0))
        n_cols = int(pfr * mask.size(1))
        priv_rows = torch.randperm(mask.size(0))[:n_rows]
        priv_cols = torch.randperm(mask.size(1))[:n_cols]
        mask[priv_rows.unsqueeze(1), priv_cols] = True
        data.priv_mask = mask
        alpha = data.x.min(dim=0)[0]
        beta = data.x.max(dim=0)[0]
        data.alpha = alpha
        data.delta = beta - alpha
        # data.delta[data.delta == 0] = 1e-7  # avoid division by zero

        if method == 'bit':
            data = one_bit(data, eps)
        elif method == 'lap':
            data = laplace(data, eps)
        elif method == 'pws':
            data = piecewise(data, eps)
        else:
            raise NotImplementedError

    data.priv_mask = None
    return data


if __name__ == '__main__':
    ds = privatize(load_dataset('cora'), pnr=1, pfr=1, eps=1, method='pws')
    print(ds.x.min(), ds.x.max())
