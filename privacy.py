import torch
import numpy as np
from torch_geometric.data import Data


class Mechanism:
    def __init__(self, x, eps, alpha=None, delta=None):
        self.x = x
        self.eps = eps
        self.alpha = alpha
        self.delta = delta

        # set alpha and delta
        if self.alpha is None:
            self.alpha = x.min(dim=0)[0]
        if self.delta is None:
            beta = x.max(dim=0)[0]
            self.delta = beta - self.alpha

    def transform(self):
        raise NotImplementedError


class Laplace(Mechanism):
    def transform(self):
        scale = torch.ones_like(self.x) * (self.delta / self.eps)
        x_priv = torch.distributions.Laplace(self.x, scale).sample()
        return x_priv


class PrivGraphConv(Mechanism):
    def transform(self):
        exp = np.exp(self.eps)
        p = (self.x - self.alpha) / self.delta
        p[:, (self.delta == 0)] = 0
        p = p * (exp - 1) / (exp + 1) + 1 / (exp + 1)
        b = torch.bernoulli(p)
        x_priv = ((b * (exp + 1) - 1) * self.delta) / (exp - 1) + self.alpha
        return x_priv


class Piecewise(Mechanism):
    def transform(self):
        # normalize x between -1,1
        t = (self.x - self.alpha) / self.delta
        t[:, (self.delta == 0)] = 0
        t = 2 * t - 1

        # piecewise mechanism's variables
        P = (np.exp(self.eps) - np.exp(self.eps / 2)) / (2 * np.exp(self.eps / 2) + 2)
        C = (np.exp(self.eps / 2) + 1) / (np.exp(self.eps / 2) - 1)
        L = t * (C + 1) / 2 - (C - 1) / 2
        R = L + C - 1

        # thresholds for random sampling
        threshold_left = P * (L + C) / np.exp(self.eps)
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
        x_priv = self.delta * (t + 1) / 2 + self.alpha
        return x_priv


class PiecewiseMulti(Mechanism):
    def transform(self):
        n = self.x.size(0)
        d = self.x.size(1)
        # k = int(max(1, min(d, np.floor(eps / 2.5))))
        k = 100
        sample = torch.cat([torch.randperm(d)[:k] for _ in range(n)]).view(n, k).to(self.x.device)
        mask = torch.zeros_like(self.x, dtype=torch.bool)
        mask.scatter_(dim=1, index=sample, value=True)
        x_priv = Piecewise(x=self.x, eps=self.eps / k, alpha=self.alpha, delta=self.delta).transform()
        x_priv = mask * x_priv * d / k
        return x_priv


class Hybrid(Mechanism):
    def transform(self):
        eps_star = np.log(
            (-5 + 2 * (6353 - 405 * np.sqrt(241)) ** (1 / 3) + 2 * (6353 + 405 * np.sqrt(241)) ** (1 / 3)) / 27
        )
        a = torch.zeros_like(self.x) + (self.eps > eps_star) * (1 - np.exp(-self.eps / 2))
        mask = torch.bernoulli(a).bool()
        pm = Piecewise(x=self.x, eps=self.eps, alpha=self.alpha, delta=self.delta)
        pgc = PrivGraphConv(x=self.x, eps=self.eps, alpha=self.alpha, delta=self.delta)
        x_priv = mask * pm.transform() + (~mask) * pgc.transform()
        return x_priv


available_mechanisms = {
    'pgc': PrivGraphConv,
    'pm': Piecewise,
    'lm': Laplace
}

extra_mechanisms = {
    'hm': Hybrid,
    'mpm': PiecewiseMulti
}


def get_available_mechanisms():
    return list(available_mechanisms.keys())


def privatize(data, method, eps=1, pfr=1):
    # copy data to avoid changing the original
    data = Data(**dict(data()))

    # indicate private features randomly
    n_priv_features = int(pfr * data.num_features)
    priv_features = torch.randperm(data.num_features)[:n_priv_features]
    priv_mask = torch.zeros_like(data.x, dtype=torch.bool)
    priv_mask[:, priv_features] = True

    mechanisms = {**available_mechanisms, **extra_mechanisms}

    if method == 'raw':
        # just trim to non-private features
        data.x = data.x[~priv_mask].view(data.num_nodes, -1)
    else:
        x_priv = mechanisms[method](x=data.x, eps=eps).transform()
        data.x = priv_mask * x_priv + ~priv_mask * data.x

    return data
