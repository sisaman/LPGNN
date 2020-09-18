import math
import torch
from scipy.special import erf


class Mechanism:
    def __init__(self, eps, delta=.001, alpha=None, sensitivity=None, **kwargs):
        self.eps = eps
        self.delta = delta
        self.alpha = alpha
        self.sensitivity = sensitivity

    def fit(self, x):
        # set alpha and delta
        if self.alpha is None:
            self.alpha = x.min(dim=0)[0]
        if self.sensitivity is None:
            beta = x.max(dim=0)[0]
            self.sensitivity = beta - self.alpha

    def transform(self, x):
        raise NotImplementedError

    def __call__(self, x):
        self.fit(x)
        return self.transform(x)


class Gaussian(Mechanism):
    def __init__(self, *args, **kwargs):
        super(Gaussian, self).__init__(*args, **kwargs)
        self.sigma = None

    def fit(self, x):
        super().fit(x)
        if torch.is_tensor(self.sensitivity) and len(self.sensitivity) > 1:
            self.sensitivity = torch.norm(self.sensitivity, p=2)
        else:
            d = x.size(1)
            self.sensitivity = self.sensitivity * math.sqrt(d)

        self.sigma = self.calibrate_analytic_gaussian_mechanism()

    def transform(self, x):
        return torch.normal(mean=x, std=self.sensitivity)

    @staticmethod
    def _phi(t):
        return 0.5 * (1.0 + erf(float(t) / math.sqrt(2.0)))

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

    def calibrate_analytic_gaussian_mechanism(self, tol=1.e-12):
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


class MultiBit(Mechanism):
    def transform(self, x):
        n, d = x.size()
        k = int(max(1, min(d, math.floor(self.eps / 2.18))))
        ek = math.exp(self.eps / k)
        p = (x - self.alpha) / self.sensitivity
        p = (p * (ek - 1) + 1) / (ek + 1)
        xs = torch.bernoulli(p) * 2 - 1
        sample = torch.cat([torch.randperm(d, device=x.device)[:k] for _ in range(n)]).view(n, k)
        mask = torch.zeros_like(x, dtype=torch.bool)
        mask.scatter_(dim=1, index=sample, value=True)
        xs = mask * xs
        ys = xs * (d*self.sensitivity) / (2 * k)
        ys = ys * (ek + 1) / (ek - 1)
        ys = ys + self.alpha + self.sensitivity / 2
        return ys


available_mechanisms = {
    'gm': Gaussian,
    'mbm': MultiBit,
}


def get_available_mechanisms():
    return list(available_mechanisms.keys())
