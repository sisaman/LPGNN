import torch
import numpy as np
from torch.distributions import Laplace
from torch_geometric.data import Data

available_mechanisms = {'bit', 'lap', 'pws', 'hyb', 'pws-m'}


def laplace(data, delta, eps):
    scale = torch.ones_like(data.x) * (delta / eps)
    x_priv = Laplace(data.x, scale).sample()
    return x_priv


def one_bit(data, alpha, delta, eps):
    exp = np.exp(eps)
    p = (data.x - alpha) / delta
    p[:, (delta == 0)] = 0
    p = p * (exp - 1) / (exp + 1) + 1 / (exp + 1)
    b = torch.bernoulli(p)
    x_priv = ((b * (exp + 1) - 1) * delta) / (exp - 1) + alpha
    return x_priv


def piecewise(data, alpha, delta, eps):
    # normalize x between -1,1
    t = (data.x - alpha) / delta
    t[:, (delta == 0)] = 0
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
    x_priv = delta * (t + 1) / 2 + alpha
    return x_priv


def piecewise_multi(data, alpha, delta, eps):
    n = data.num_nodes
    d = data.num_features
    # k = int(max(1, min(d, np.floor(eps / 2.5))))
    k = 100
    sample = torch.cat([torch.randperm(d)[:k] for _ in range(n)]).view(n, k).to(data.x.device)
    mask = torch.zeros_like(data.x, dtype=torch.bool)
    mask.scatter_(dim=1, index=sample, value=True)
    x_priv = piecewise(data, alpha, delta, eps/k)
    x_priv = mask * x_priv * d / k
    return x_priv


def hybrid(data, alpha, delta, eps):
    eps_star = np.log(
        (-5 + 2 * (6353 - 405 * np.sqrt(241)) ** (1 / 3) + 2 * (6353 + 405 * np.sqrt(241)) ** (1 / 3)) / 27
    )
    a = torch.zeros_like(data.x) + (eps > eps_star) * (1 - np.exp(-eps / 2))
    mask = torch.bernoulli(a).bool()
    x_priv = mask * piecewise(data, alpha, delta, eps) + (~mask) * one_bit(data, alpha, delta, eps)
    return x_priv


def privatize(data, method, pfr=0, eps=1):
    # copy data to avoid changing the original
    data = Data(**dict(data()))

    # indicate private features randomly
    n_priv_features = int(pfr * data.num_features)
    priv_features = torch.randperm(data.num_features)[:n_priv_features]
    priv_mask = torch.zeros_like(data.x, dtype=torch.bool)
    priv_mask[:, priv_features] = True

    # set alpha and delta
    alpha = data.x.min(dim=0)[0]
    beta = data.x.max(dim=0)[0]
    delta = beta - alpha

    if method == 'raw':
        # just trim to non-private features
        data.x = data.x[~priv_mask].view(data.num_nodes, -1)
        return data
    elif method == 'lap':
        x_priv = laplace(data, delta, eps)
    elif method == 'bit':
        x_priv = one_bit(data, alpha, delta, eps)
    elif method == 'pws':
        x_priv = piecewise(data, alpha, delta, eps)
    elif method == 'pws-m':
        x_priv = piecewise_multi(data, alpha, delta, eps)
    elif method == 'hyb':
        x_priv = hybrid(data, alpha, delta, eps)
    else:
        raise NotImplementedError

    data.x = priv_mask * x_priv + ~priv_mask * data.x
    return data
