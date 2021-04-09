import math
import torch
import torch.nn.functional as F


class Mechanism:
    def __init__(self, eps, input_range, **kwargs):
        self.eps = eps
        self.alpha, self.beta = input_range

    def __call__(self, x):
        raise NotImplementedError


class Laplace(Mechanism):
    def __call__(self, x):
        d = x.size(1)
        sensitivity = (self.beta - self.alpha) * d
        scale = torch.ones_like(x) * (sensitivity / self.eps)
        out = torch.distributions.Laplace(x, scale).sample()
        # out = torch.clip(out, min=self.alpha, max=self.beta)
        return out


class MultiBit(Mechanism):
    def __init__(self, *args, m='best', **kwargs):
        super().__init__(*args, **kwargs)
        self.m = m

    def __call__(self, x):
        n, d = x.size()
        if self.m == 'best':
            m = int(max(1, min(d, math.floor(self.eps / 2.18))))
        elif self.m == 'max':
            m = d
        else:
            m = self.m

        # sample features for perturbation
        BigS = torch.rand_like(x).topk(m, dim=1).indices
        s = torch.zeros_like(x, dtype=torch.bool).scatter(1, BigS, True)
        del BigS

        # perturb sampled features
        em = math.exp(self.eps / m)
        p = (x - self.alpha) / (self.beta - self.alpha)
        p = (p * (em - 1) + 1) / (em + 1)
        t = torch.bernoulli(p)
        x_star = s * (2 * t - 1)
        del p, t, s

        # unbiase the result
        x_prime = d * (self.beta - self.alpha) / (2 * m)
        x_prime = x_prime * (em + 1) * x_star / (em - 1)
        x_prime = x_prime + (self.alpha + self.beta) / 2
        return x_prime


class OneBit(MultiBit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, m='max', **kwargs)


class RandomizedResopnse:
    def __init__(self, eps, d):
        self.d = d
        self.q = 1.0 / (math.exp(eps) + self.d - 1)
        self.p = self.q * math.exp(eps)

    def __call__(self, y):
        pr = y * self.p + (1 - y) * self.q
        out = torch.multinomial(pr, num_samples=1)
        return F.one_hot(out.squeeze(), num_classes=self.d)


supported_feature_mechanisms = {
    'mbm': MultiBit,
    '1bm': OneBit,
    'lpm': Laplace,
}
