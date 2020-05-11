import torch
import numpy as np
from torch.distributions import Laplace
from torch_geometric.data import Data

from datasets import load_dataset

available_mechanisms = {'bit', 'lap', 'pws', 'hyb'}


def laplace(data, eps):
    scale = torch.ones_like(data.x) * (data.delta / eps)
    x_priv = Laplace(data.x, scale).sample()
    return x_priv


def one_bit(data, eps):
    exp = np.exp(eps)
    p = (data.x - data.alpha) / data.delta
    p[:, (data.delta == 0)] = 0
    p = p * (exp - 1) / (exp + 1) + 1 / (exp + 1)
    b = torch.bernoulli(p)
    x_priv = ((b * (exp + 1) - 1) * data.delta) / (exp - 1) + data.alpha
    return x_priv


def piecewise(data, eps):
    # normalize x between -1,1
    t = (data.x - data.alpha) / data.delta
    t[:, (data.delta == 0)] = 0
    t = 2 * t - 1

    # piecewise mechanism's variables
    P = (np.exp(eps) - np.exp(eps / 2)) / (2 * np.exp(eps / 2) + 2)
    C = (np.exp(eps / 2) + 1) / (np.exp(eps / 2) - 1)
    L = t * (C + 1) / 2 - (C - 1) / 2
    R = L + C - 1

    # thresholds for random sampling
    threshold_left = P * (L + C) / np.exp(eps)
    threshold_right = threshold_left + P * (R - L)

    # masks for piecewise random sampling
    x = torch.rand_like(t)
    mask_left = x < threshold_left
    mask_middle = (threshold_left < x) & (x < threshold_right)
    mask_right = threshold_right < x

    # random sampling
    t = mask_left * (torch.rand_like(t) * (L + C) - C)
    t += mask_middle * (torch.rand_like(t) * (R - L) + L)
    t += mask_right * (torch.rand_like(t) * (C - R) + R)

    # unbiased data
    x_priv = data.delta * (t + 1) / 2 + data.alpha
    return x_priv


def hybrid(data, eps):
    eps_star = np.log(
        (-5 + 2 * (6353 - 405 * np.sqrt(241)) ** (1 / 3) + 2 * (6353 + 405 * np.sqrt(241)) ** (1 / 3)) / 27
    )
    alpha = torch.zeros_like(data.x) + (eps > eps_star) * (1 - np.exp(-eps / 2))
    mask = torch.bernoulli(alpha).bool()
    x_priv = mask * piecewise(data, eps) + (~mask) * one_bit(data, eps)
    return x_priv


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
            x_priv = one_bit(data, eps)
        elif method == 'lap':
            x_priv = laplace(data, eps)
        elif method == 'pws':
            x_priv = piecewise(data, eps)
        elif method == 'hyb':
            x_priv = hybrid(data, eps)
        else:
            raise NotImplementedError

    # noinspection PyUnboundLocalVariable
    data.x = data.priv_mask * x_priv + ~data.priv_mask * data.x
    data.priv_mask = None
    return data


if __name__ == '__main__':
    ds = privatize(load_dataset('cora'), pnr=1, pfr=1, eps=1, method='pws')
    print(ds.x.min(), ds.x.max())
