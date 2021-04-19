import math
import torch
import torch.nn.functional as F
from scipy.special import erf


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


class RandomBit(OneBit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, x):
        w = torch.rand(x.size(1), 1, device=x.device)
        w /= w.sum(dim=0)
        x = torch.matmul(x, w)
        return super().__call__(x)


class Gaussian(Mechanism):
    def __init__(self, *args, delta=1e-10, **kwargs):
        super().__init__(*args, **kwargs)
        self.delta = delta
        self.sigma = None
        self.sensitivity = None

    def __call__(self, x):
        len_interval = self.beta - self.alpha
        if torch.is_tensor(len_interval) and len(len_interval) > 1:
            self.sensitivity = torch.norm(len_interval, p=2)
        else:
            d = x.size(1)
            self.sensitivity = len_interval * math.sqrt(d)

        self.sigma = self.calibrate_gaussian_mechanism()
        out = torch.normal(mean=x, std=self.sigma)
        # out = torch.clip(out, min=self.alpha, max=self.beta)
        return out

    def calibrate_gaussian_mechanism(self):
        return self.sensitivity * math.sqrt(2 * math.log(1.25 / self.delta)) / self.eps


class AnalyticGaussian(Gaussian):
    def calibrate_gaussian_mechanism(self, tol=1.e-12):
        """ Calibrate a Gaussian perturbation for differential privacy
        using the analytic Gaussian mechanism of [Balle and Wang, ICML'18]
        Arguments:
        tol : error tolerance for binary search (tol > 0)
        Output:
        sigma : standard deviation of Gaussian noise needed to achieve (epsilon,delta)-DP under global sensitivity GS
        """
        delta_thr = self._case_a(0.0)
        if self.delta == delta_thr:
            alpha = 1.0
        else:
            if self.delta > delta_thr:
                predicate_stop_DT = lambda s: self._case_a(s) >= self.delta
                function_s_to_delta = lambda s: self._case_a(s)
                predicate_left_BS = lambda s: function_s_to_delta(s) > self.delta
                function_s_to_alpha = lambda s: math.sqrt(1.0 + s / 2.0) - math.sqrt(s / 2.0)
            else:
                predicate_stop_DT = lambda s: self._case_b(s) <= self.delta
                function_s_to_delta = lambda s: self._case_b(s)
                predicate_left_BS = lambda s: function_s_to_delta(s) < self.delta
                function_s_to_alpha = lambda s: math.sqrt(1.0 + s / 2.0) + math.sqrt(s / 2.0)
            predicate_stop_BS = lambda s: abs(function_s_to_delta(s) - self.delta) <= tol
            s_inf, s_sup = self._doubling_trick(predicate_stop_DT, 0.0, 1.0)
            s_final = self._binary_search(predicate_stop_BS, predicate_left_BS, s_inf, s_sup)
            alpha = function_s_to_alpha(s_final)
        sigma = alpha * self.sensitivity / math.sqrt(2.0 * self.eps)
        return sigma

    @staticmethod
    def _phi(t):
        return 0.5 * (1.0 + erf(t / math.sqrt(2.0)))

    def _case_a(self, s):
        return self._phi(math.sqrt(self.eps * s)) - math.exp(self.eps) * self._phi(-math.sqrt(self.eps * (s + 2.0)))

    def _case_b(self, s):
        return self._phi(-math.sqrt(self.eps * s)) - math.exp(self.eps) * self._phi(-math.sqrt(self.eps * (s + 2.0)))

    @staticmethod
    def _doubling_trick(predicate_stop, s_inf, s_sup):
        while not predicate_stop(s_sup):
            s_inf = s_sup
            s_sup = 2.0 * s_inf
        return s_inf, s_sup

    @staticmethod
    def _binary_search(predicate_stop, predicate_left, s_inf, s_sup):
        s_mid = s_inf + (s_sup - s_inf) / 2.0
        while not predicate_stop(s_mid):
            if predicate_left(s_mid):
                s_sup = s_mid
            else:
                s_inf = s_mid
            s_mid = s_inf + (s_sup - s_inf) / 2.0
        return s_mid

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
    'agm': AnalyticGaussian,
    'rbm': RandomBit
}
