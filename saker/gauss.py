import numpy as np

TAILCUT = 9.0


class Counter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.normals = 0
        self.dgauss = 0
        self.stages = 0
        self.ringmul = 0
        self.ringdiv = 0
        self.fftbutterfly = 0

    def snapshot(self):
        return dict(normals=self.normals, dgauss=self.dgauss, stages=self.stages,
                    ringmul=self.ringmul, ringdiv=self.ringdiv,
                    fftbutterfly=self.fftbutterfly)


CTR = Counter()


class GaussianZ:
    def __init__(self, rng=None):
        self.rng = rng if rng is not None else np.random.default_rng()

    def continuous(self, m):
        CTR.normals += m
        return self.rng.standard_normal(m)

    def discrete_fixed(self, centers, r):
        centers = np.asarray(centers, dtype=float)
        m = centers.size
        CTR.dgauss += m
        w = int(np.ceil(TAILCUT * r)) + 1
        base = np.floor(centers).astype(np.int64)
        frac = centers - base
        offs = np.arange(-w, w + 2, dtype=np.int64)
        d = offs[None, :] - frac[:, None]
        logp = -(d * d) / (2.0 * r * r)
        logp -= logp.max(axis=1, keepdims=True)
        p = np.exp(logp)
        p /= p.sum(axis=1, keepdims=True)
        cdf = np.cumsum(p, axis=1)
        u = self.rng.random(m)
        idx = (cdf < u[:, None]).sum(axis=1)
        idx = np.minimum(idx, offs.size - 1)
        return base + offs[idx]

    def discrete_var(self, centers, sigmas):
        centers = np.asarray(centers, dtype=float)
        sigmas = np.asarray(sigmas, dtype=float)
        m = centers.size
        CTR.dgauss += m
        w = int(np.ceil(TAILCUT * float(sigmas.max()))) + 1
        base = np.floor(centers).astype(np.int64)
        frac = centers - base
        offs = np.arange(-w, w + 2, dtype=np.int64)
        d = offs[None, :] - frac[:, None]
        logp = -(d * d) / (2.0 * (sigmas[:, None] ** 2))
        logp -= logp.max(axis=1, keepdims=True)
        p = np.exp(logp)
        p /= p.sum(axis=1, keepdims=True)
        cdf = np.cumsum(p, axis=1)
        u = self.rng.random(m)
        idx = (cdf < u[:, None]).sum(axis=1)
        idx = np.minimum(idx, offs.size - 1)
        return base + offs[idx]



def eta_eps(dim, eps):
    return (1.0 / np.pi) * np.sqrt(np.log(2.0 * dim * (1.0 + 1.0 / eps)) / 2.0)
