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
        y_star = torch.distributions.Laplace(self.x, scale).sample()
        return y_star


class PrivGraphConv(Mechanism):
    def transform(self):
        exp = np.exp(self.eps)
        p = (self.x - self.alpha) / self.delta
        p[:, (self.delta == 0)] = 0
        p = p * (exp - 1) / (exp + 1) + 1 / (exp + 1)
        y = torch.bernoulli(p)
        y_star = ((y * (exp + 1) - 1) * self.delta) / (exp - 1) + self.alpha
        return y_star


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

        # unbias data
        y_star = self.delta * (t + 1) / 2 + self.alpha
        return y_star


class PiecewiseMulti(Mechanism):
    def transform(self):
        n = self.x.size(0)
        d = self.x.size(1)
        # k = int(max(1, min(d, np.floor(eps / 2.5))))
        k = 100
        sample = torch.cat([torch.randperm(d)[:k] for _ in range(n)]).view(n, k).to(self.x.device)
        mask = torch.zeros_like(self.x, dtype=torch.bool)
        mask.scatter_(dim=1, index=sample, value=True)
        y_star = Piecewise(x=self.x, eps=self.eps / k, alpha=self.alpha, delta=self.delta).transform()
        y_star = mask * y_star * d / k
        return y_star


class Hybrid(Mechanism):
    def transform(self):
        eps_star = np.log(
            (-5 + 2 * (6353 - 405 * np.sqrt(241)) ** (1 / 3) + 2 * (6353 + 405 * np.sqrt(241)) ** (1 / 3)) / 27
        )
        a = torch.zeros_like(self.x) + (self.eps > eps_star) * (1 - np.exp(-self.eps / 2))
        mask = torch.bernoulli(a).bool()
        pm = Piecewise(x=self.x, eps=self.eps, alpha=self.alpha, delta=self.delta)
        pgc = PrivGraphConv(x=self.x, eps=self.eps, alpha=self.alpha, delta=self.delta)
        y_star = mask * pm.transform() + (~mask) * pgc.transform()
        return y_star


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


def privatize(data, method, eps):
    mechanisms = {**available_mechanisms, **extra_mechanisms}

    # copy data to avoid changing the original
    data = Data(**dict(data()))

    if method in mechanisms:
        data.x = mechanisms[method](x=data.x, eps=eps).transform()

    return data
