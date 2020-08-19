import torch
import numpy as np
from torch_geometric.data import Data


class Mechanism:
    def __init__(self, eps, alpha=None, delta=None):
        self.eps = eps
        self.alpha = alpha
        self.delta = delta

    def fit(self, x):
        # set alpha and delta
        if self.alpha is None:
            self.alpha = x.min(dim=0)[0]
        if self.delta is None:
            beta = x.max(dim=0)[0]
            self.delta = beta - self.alpha

    def transform(self, x):
        raise NotImplementedError

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)


class Laplace(Mechanism):
    def transform(self, x):
        scale = torch.ones_like(x) * (self.delta / self.eps)
        y_star = torch.distributions.Laplace(x, scale).sample()
        return y_star


class PrivGraphConv(Mechanism):
    def transform(self, x):
        exp = np.exp(self.eps)
        p = (x - self.alpha) / self.delta
        p[:, (self.delta == 0)] = 0
        p = p * (exp - 1) / (exp + 1) + 1 / (exp + 1)
        y = torch.bernoulli(p)
        y_star = ((y * (exp + 1) - 1) * self.delta) / (exp - 1) + self.alpha
        return y_star


class Piecewise(Mechanism):
    def transform(self, x):
        # normalize x between -1,1
        t = (x - self.alpha) / self.delta
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


available_mechanisms = {
    'pgc': PrivGraphConv,
    'pm': Piecewise,
    'lm': Laplace
}


def get_available_mechanisms():
    return list(available_mechanisms.keys())


def privatize(data, method, eps):
    # copy data to avoid changing the original
    data = Data(**dict(data()))

    if method in available_mechanisms:
        data.x = available_mechanisms[method](eps=eps).fit_transform(data.x)

    return data
