import numpy as np

ETA = {512: 1.2778, 1024: 1.2983}
TAU = 1.1
SALTLEN = 40


class ParamSet:
    def __init__(self, name, n, q, family, alpha, ell=None, m=None, w=None,
                 pct=25, nu_hint=None, published=None):
        self.name = name
        self.n = n
        self.q = q
        self.family = family
        self.alpha = alpha
        self.ell = ell
        self.m = m
        self.w = w
        self.pct = pct
        self.nu_hint = nu_hint
        self.published = published
        self.eta = ETA[n]
        self.sigma = alpha * self.eta * np.sqrt(q)
        self.beta = TAU * self.sigma * np.sqrt(2.0 * n)

    def radii(self):
        return np.sqrt(self.q * (self.m - self.w)), np.sqrt(self.q * (self.m + self.w))

    def pk_bytes(self):
        return int(np.ceil(self.n * np.ceil(np.log2(self.q)) / 8.0)) + 1

    def __repr__(self):
        return "%s(n=%d,q=%d,%s,alpha=%.4f,ell=%s)" % (
            self.name, self.n, self.q, self.family, self.alpha, str(self.ell))


SAKER_512 = ParamSet("SAKER-512", 512, 12289, "folded", 1.03, ell=5, m=1.00, w=0.20)
SAKER_1024 = ParamSet("SAKER-1024", 1024, 12289, "folded", 1.22, ell=6, m=1.45, w=0.20)
SAKER_1024C = ParamSet("SAKER-1024c", 1024, 12289, "folded", 1.03, ell=6, m=1.00, w=0.20)

FALCON_512 = ParamSet("Falcon-512", 512, 12289, "ffo", 1.17, pct=11, published=123)
FALCON_1024 = ParamSet("Falcon-1024", 1024, 12289, "ffo", 1.17, pct=11, published=284)
ANTRAG_512 = ParamSet("Antrag-512", 512, 12289, "hybrid", 1.15, m=1.0, w=None, published=124)
ANTRAG_1024 = ParamSet("Antrag-1024", 1024, 12289, "hybrid", 1.23, m=1.0, w=None, published=264)
MITAKA_512 = ParamSet("Mitaka-512", 512, 12289, "hybrid", 2.04, nu_hint=1.17, published=102)
MITAKA_1024 = ParamSet("Mitaka-1024", 1024, 12289, "hybrid", 2.33, nu_hint=1.17, published=233)

ALL = [FALCON_512, MITAKA_512, ANTRAG_512, SAKER_512,
       FALCON_1024, MITAKA_1024, ANTRAG_1024, SAKER_1024, SAKER_1024C]
