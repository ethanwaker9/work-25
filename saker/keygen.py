import numpy as np

from .ntru import ntru_solve, check_ntru, public_key
from .quality import alpha_fold, alpha_gpv, alpha_hybrid, sigma_embeddings
from .ring import fft, ifft, roots

_CONJ = {}


def conj_index(n):
    if n in _CONJ:
        return _CONJ[n]
    w = roots(n)
    key = {}
    for i, z in enumerate(w):
        key[(round(z.real, 9), round(z.imag, 9))] = i
    idx = np.empty(n, dtype=int)
    for i, z in enumerate(w):
        idx[i] = key[(round(z.real, 9), round(-z.imag, 9))]
    reps = []
    seen = set()
    for i in range(n):
        if i in seen:
            continue
        reps.append(i)
        seen.add(i)
        seen.add(int(idx[i]))
    _CONJ[n] = (idx, np.array(reps, dtype=int))
    return _CONJ[n]


class NTRUKey:
    def __init__(self, n, q, f, g, F, G):
        self.n = n
        self.q = q
        self.f = np.array([int(x) for x in f], dtype=object)
        self.g = np.array([int(x) for x in g], dtype=object)
        self.F = np.array([int(x) for x in F], dtype=object)
        self.G = np.array([int(x) for x in G], dtype=object)
        self.h = public_key(self.f, self.g, q)
        ff = fft(np.array([float(x) for x in self.f]))
        gf = fft(np.array([float(x) for x in self.g]))
        Ff = fft(np.array([float(x) for x in self.F]))
        Gf = fft(np.array([float(x) for x in self.G]))
        self.f_fft = ff
        self.g_fft = gf
        self.F_fft = Ff
        self.G_fft = Gf
        self.b1 = (gf, -ff)
        self.b2 = (Gf, -Ff)
        self.Sigma = (np.abs(ff) ** 2 + np.abs(gf) ** 2).real

    def alpha(self, ell):
        return alpha_fold(self.Sigma, self.q, ell)

    def alpha_gpv(self):
        return alpha_gpv(self.Sigma, self.q)

    def s1(self):
        T = (np.abs(self.f_fft) ** 2 + np.abs(self.g_fft) ** 2
             + np.abs(self.F_fft) ** 2 + np.abs(self.G_fft) ** 2).real
        disc = np.sqrt(np.maximum(T ** 2 - 4.0 * self.q ** 2, 0.0))
        return float(np.sqrt(0.5 * (T + disc).max()))

    def alpha_peikert(self):
        return self.s1() / np.sqrt(self.q)

    def norm_fg(self):
        return float(np.sqrt(sum(float(x) ** 2 for x in self.f)
                             + sum(float(x) ** 2 for x in self.g)))


def try_build(n, q, f, g):
    fo = np.array([int(x) for x in f], dtype=object)
    go = np.array([int(x) for x in g], dtype=object)
    if public_key(fo, go, q) is None:
        return None
    res = ntru_solve(fo, go, q)
    if res is None:
        return None
    F, G = res
    if not check_ntru(fo, go, F, G, q):
        return None
    return NTRUKey(n, q, fo, go, F, G)


def gaussian_candidate(n, q, rng, sig=None):
    if sig is None:
        sig = 1.17 * np.sqrt(q / (2.0 * n))
    f = np.rint(rng.normal(0, sig, n)).astype(np.int64)
    g = np.rint(rng.normal(0, sig, n)).astype(np.int64)
    return f, g


def annulus_candidate(n, q, rng, r, R):
    idx, reps = conj_index(n)
    m = len(reps)
    u = rng.uniform(r * r, R * R, m)
    rho = np.sqrt(u)
    th = rng.uniform(0, np.pi / 2, m)
    x = rho * np.cos(th)
    y = rho * np.sin(th)
    w1 = rng.uniform(0, 2 * np.pi, m)
    w2 = rng.uniform(0, 2 * np.pi, m)
    zf = np.zeros(n, dtype=complex)
    zg = np.zeros(n, dtype=complex)
    zf[reps] = x * np.exp(1j * w1)
    zg[reps] = y * np.exp(1j * w2)
    zf[idx[reps]] = np.conj(zf[reps])
    zg[idx[reps]] = np.conj(zg[reps])
    f = np.rint(ifft(zf)).astype(np.int64)
    g = np.rint(ifft(zg)).astype(np.int64)
    return f, g


def annulus_radii(q, alpha, xi):
    hi = alpha * np.sqrt(q)
    lo = np.sqrt(q) / alpha
    r = 0.5 * (1 - xi) * hi + 0.5 * (1 + xi) * lo
    R = 0.5 * (1 + xi) * hi + 0.5 * (1 - xi) * lo
    return r, R




def saker_keygen(n, q, rng, ell, alpha_target, r, R, max_tries=200000):
    tries = 0
    while tries < max_tries:
        tries += 1
        f, g = annulus_candidate(n, q, rng, r, R)
        S = sigma_embeddings(f, g)
        if alpha_fold(S, q, ell) > alpha_target:
            continue
        k = try_build(n, q, f, g)
        if k is not None:
            return k, tries
    return None, tries


