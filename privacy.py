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


class Laplace(Mechanism):
    def transform(self, x):
        scale = torch.ones_like(x) * (self.sensitivity / self.eps)
        y_star = torch.distributions.Laplace(x, scale).sample()
        return y_star


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


class PrivGraphConv(Mechanism):
    def transform(self, x):
        exp = math.exp(self.eps)
        p = (x - self.alpha) / self.sensitivity
        p = p * (exp - 1) / (exp + 1) + 1 / (exp + 1)
        y = torch.bernoulli(p)
        y_star = ((y * (exp + 1) - 1) * self.sensitivity) / (exp - 1) + self.alpha
        return y_star


class Piecewise(Mechanism):
    def transform(self, x):
        # normalize x between -1,1
        t = (x - self.alpha) / self.sensitivity
        t = 2 * t - 1

        # piecewise mechanism's variables
        P = (math.exp(self.eps) - math.exp(self.eps / 2)) / (2 * math.exp(self.eps / 2) + 2)
        C = (math.exp(self.eps / 2) + 1) / (math.exp(self.eps / 2) - 1)
        L = t * (C + 1) / 2 - (C - 1) / 2
        R = L + C - 1

        # thresholds for random sampling
        threshold_left = P * (L + C) / math.exp(self.eps)
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
        y_star = self.sensitivity * (t + 1) / 2 + self.alpha
        return y_star


class MultiDimPiecewise(Piecewise):
    def transform(self, x):
        n, d = x.size()
        k = int(max(1, min(d, math.floor(self.eps / 2.5))))
        sample = torch.cat([torch.randperm(d, device=x.device)[:k] for _ in range(n)]).view(n, k)
        mask = torch.zeros_like(x, dtype=torch.bool)
        mask.scatter_(dim=1, index=sample, value=True)
        self.eps /= k
        y = super().transform(x)
        z = mask * y * d / k
        return z


class SpecialPiecewise(Piecewise):
    def __init__(self, dv=None, **kwargs):
        super(SpecialPiecewise, self).__init__(**kwargs)
        self.dv = dv / dv.max()

    def transform(self, x):
        n, d = x.size()
        c = 100
        eps_total = self.eps
        Z = torch.empty(0, device=x.device)
        for i in range(n):
            k = max(int(self.dv[i] * c), 1)
            self.eps = eps_total / k
            y = super().transform(x[i, :])
            choice = torch.randperm(d, device=x.device)[:k]
            mask = torch.zeros_like(y, dtype=torch.bool)
            mask[choice] = True
            z = mask * y * d / k
            Z = torch.cat([Z, z])

        self.eps = eps_total
        Z = Z.view(n, d)
        return Z


class MultiDimDuchi(Mechanism):
    def transform(self, x):
        # normalize x between -1,1
        t = (x - self.alpha) / self.sensitivity
        t = 2 * t - 1

        n, d = t.size()
        if d % 2 == 0:
            C = (2 ** (d - 1)) / math.comb(d - 1, d // 2) + (math.comb(d, d // 2) / math.comb(d - 1, d // 2)) * 0.5
        else:
            C = (2 ** (d - 1)) / math.comb(d - 1, (d - 1) // 2)

        B = C * (math.exp(self.eps) + 1) / (math.exp(self.eps) - 1)
        V = torch.bernoulli(t * 0.5 + 0.5)
        V = V * 2 - 1  # rescale between -1 and 1
        u = torch.bernoulli(torch.ones(n, device=x.device) * (math.exp(self.eps) / (math.exp(self.eps) + 1)))
        u = u * 2 - 1

        T = torch.empty(0, device=x.device)
        for i in range(n):
            v = V[i, :]
            while True:
                t = torch.randint_like(v, low=0, high=2) * 2 - 1
                t *= B
                if t.dot(v) * u[i] >= 0:
                    break
            T = torch.cat([T, t])
        T = T.view(n, d)

        # unbias data
        y_star = self.sensitivity * (T + 1) / 2 + self.alpha
        return y_star


class MultiOneBit(Mechanism):
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


class Wang(Mechanism):
    def transform(self, x):
        # normalize x between -1,1
        t = (x - self.alpha) / self.sensitivity
        t = 2 * t - 1

        p = t * (math.exp(self.eps) + 2 * self.delta - 1) / (2 * (math.exp(self.eps)+ 1)) + 0.5
        u = torch.bernoulli(p) * 2 - 1
        t = u * (math.exp(self.eps) + 1) / (math.exp(self.eps) + 2 * self.delta - 1)

        # unbias data
        y_star = self.sensitivity * (t + 1) / 2 + self.alpha
        return y_star


class MultiDimWang(Wang):
    def transform(self, x):
        n, d = x.size()
        k = int(max(1, min(d, math.floor(self.eps / 2.17))))
        sample = torch.cat([torch.randperm(d, device=x.device)[:k] for _ in range(n)]).view(n, k)
        mask = torch.zeros_like(x, dtype=torch.bool)
        mask.scatter_(dim=1, index=sample, value=True)
        self.eps /= k
        y = super().transform(x)
        z = mask * y * d / k
        return z


available_mechanisms = {
    'pgc': PrivGraphConv,
    'pm': Piecewise,
    'lm': Laplace,
    'wm': Wang,
    'gm': Gaussian,
    'mdm': MultiDimDuchi,
    'mpm': MultiDimPiecewise,
    'spm': SpecialPiecewise,
    'mwm': MultiDimWang,
    'mob': MultiOneBit
}


def get_available_mechanisms():
    return list(available_mechanisms.keys())


class Privatize:
    def __init__(self, method, eps, **kwargs):
        self.method = method
        self.eps = eps
        self.kwargs = kwargs

    def __call__(self, data):
        if self.method == 'raw':
            if hasattr(data, 'x_raw'):
                data.x = data.x_raw  # bring back x_raw
        else:
            if not hasattr(data, 'x_raw'):
                data.x_raw = data.x  # save original x to x_raw
            data.x = available_mechanisms[self.method](eps=self.eps, **self.kwargs)(data.x_raw)
        return data


if __name__ == '__main__':
    # from datasets import GraphDataset
    # dataset = GraphDataset('cora')
    # x = dataset.get_data().x
    # mdm = MultiDimDuchi(eps=1)
    # t = mdm(x)
    pass
