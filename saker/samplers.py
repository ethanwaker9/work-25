import numpy as np

from .gauss import CTR, GaussianZ
from .ring import (fft, ifft, split_fft, merge_fft, anticirculant)


class Node:
    __slots__ = ("l", "c0", "c1", "u", "d", "leaf", "dim")

    def __init__(self):
        self.l = None
        self.c0 = None
        self.c1 = None
        self.u = None
        self.d = None
        self.leaf = False
        self.dim = 0


def module_ldl(key):
    gf = key.b1[0]
    mff = key.b1[1]
    Gf = key.b2[0]
    mFf = key.b2[1]
    g00 = gf * np.conj(gf) + mff * np.conj(mff)
    g01 = gf * np.conj(Gf) + mff * np.conj(mFf)
    g11 = Gf * np.conj(Gf) + mFf * np.conj(mFf)
    d1 = g00.real
    l = np.conj(g01) / d1
    d2 = (g11 - g01 * np.conj(g01) / d1).real
    return d1, l, d2


def build_subtree(d, depth, sigma, r, leaf_peikert):
    node = Node()
    node.dim = len(d)
    if depth == 0 or len(d) == 1:
        node.leaf = True
        node.d = d
        if leaf_peikert:
            v = sigma * sigma / d - r * r
            if np.min(v) < -1e-9 * sigma * sigma:
                raise ValueError("perturbation covariance not positive")
            node.u = np.sqrt(np.maximum(v, 0.0))
        else:
            node.u = sigma / np.sqrt(d)
        return node
    dc = d.astype(complex)
    d0, d1 = split_fft(dc)
    CTR.ringdiv += len(d0)
    node.l = d1 / d0
    child0 = d0.real
    child1 = (d0 - d1 * np.conj(d1) / d0).real
    node.c0 = build_subtree(child0, depth - 1, sigma, r, leaf_peikert)
    node.c1 = build_subtree(child1, depth - 1, sigma, r, leaf_peikert)
    return node


def subtree_words(node):
    if node.leaf:
        return node.dim
    return node.dim // 2 + subtree_words(node.c0) + subtree_words(node.c1)


class FoldedSampler:
    name = "folded"

    def __init__(self, key, sigma, r, depth, rng=None, leaf_peikert=True, lazy=False):
        self.key = key
        self.n = key.n
        self.sigma = sigma
        self.r = r
        self.depth = depth
        self.leaf_peikert = leaf_peikert
        self.lazy = lazy
        self.gz = GaussianZ(rng)
        self.d1, self.lroot, self.d2 = module_ldl(key)
        if not lazy:
            self.t0 = build_subtree(self.d1, depth, sigma, r, leaf_peikert)
            self.t1 = build_subtree(self.d2, depth, sigma, r, leaf_peikert)
        else:
            self.t0 = None
            self.t1 = None

    def key_words(self):
        base = 4 * self.n + self.n
        if self.lazy:
            path = 0
            m = self.n
            for _ in range(self.depth):
                path += m
                m //= 2
            return base + 2 * m + path
        return base + subtree_words(self.t0) + subtree_words(self.t1)

    def _leaf(self, t, node):
        m = len(t)
        CTR.stages += 1
        if self.leaf_peikert:
            w = self.gz.continuous(m)
            if m > 1:
                c = ifft(t - fft(w) * node.u)
            else:
                c = np.array([t[0].real - w[0] * node.u[0]])
            z = self.gz.discrete_fixed(c, self.r)
        else:
            tc = ifft(t) if m > 1 else np.array([t[0].real])
            z = self.gz.discrete_var(tc, node.u)
        z = z.astype(float)
        return fft(z) if m > 1 else np.array([complex(z[0])])

    def _rec(self, t, node, d=None, depth=None):
        if self.lazy:
            if depth == 0 or len(t) == 1:
                nd = Node()
                nd.dim = len(d)
                nd.leaf = True
                if self.leaf_peikert:
                    v = self.sigma * self.sigma / d - self.r * self.r
                    nd.u = np.sqrt(np.maximum(v, 0.0))
                else:
                    nd.u = self.sigma / np.sqrt(d)
                return self._leaf(t, nd)
            dc = d.astype(complex)
            d0, d1 = split_fft(dc)
            CTR.ringdiv += len(d0)
            l = d1 / d0
            ch0 = d0.real
            ch1 = (d0 - d1 * np.conj(d1) / d0).real
            t0, t1 = split_fft(t)
            z1 = self._rec(t1, None, ch1, depth - 1)
            t0p = t0 + (t1 - z1) * l
            CTR.ringmul += len(t0)
            z0 = self._rec(t0p, None, ch0, depth - 1)
            return merge_fft(z0, z1)
        if node.leaf:
            return self._leaf(t, node)
        t0, t1 = split_fft(t)
        z1 = self._rec(t1, node.c1)
        t0p = t0 + (t1 - z1) * node.l
        CTR.ringmul += len(t0)
        z0 = self._rec(t0p, node.c0)
        return merge_fft(z0, z1)

    def sample(self, t):
        t0, t1 = t
        if self.lazy:
            z1 = self._rec(t1, None, self.d2, self.depth)
        else:
            z1 = self._rec(t1, self.t1)
        t0p = t0 + (t1 - z1) * self.lroot
        CTR.ringmul += len(t0)
        if self.lazy:
            z0 = self._rec(t0p, None, self.d1, self.depth)
        else:
            z0 = self._rec(t0p, self.t0)
        return z0, z1


class HybridSampler(FoldedSampler):
    name = "hybrid"

    def __init__(self, key, sigma, r, rng=None):
        FoldedSampler.__init__(self, key, sigma, r, 0, rng, True, False)


class FFOSampler(FoldedSampler):
    name = "ffo"

    def __init__(self, key, sigma, r, rng=None):
        FoldedSampler.__init__(self, key, sigma, r, int(np.log2(key.n)), rng, False, False)


class PeikertSampler:
    name = "peikert"

    def __init__(self, key, sigma, r, rng=None):
        self.key = key
        self.n = key.n
        self.sigma = sigma
        self.r = r
        self.gz = GaussianZ(rng)
        n = key.n
        B = np.empty((n, 2, 2), dtype=complex)
        B[:, 0, 0] = key.b1[0]
        B[:, 0, 1] = key.b1[1]
        B[:, 1, 0] = key.b2[0]
        B[:, 1, 1] = key.b2[1]
        self.B = B
        Gm = B @ np.conj(np.transpose(B, (0, 2, 1)))
        inv = np.conj(np.linalg.inv(Gm))
        Sp = (sigma * sigma) * inv - (r * r) * np.eye(2)[None, :, :]
        w, V = np.linalg.eigh(Sp)
        if np.min(w) < -1e-9 * sigma * sigma:
            raise ValueError("peikert perturbation not positive")
        w = np.maximum(w, 0.0)
        self.A = V @ (np.sqrt(w)[:, :, None] * np.conj(np.transpose(V, (0, 2, 1))))
        self.Binv = np.linalg.inv(B)

    def key_words(self):
        return 4 * self.n + 4 * self.n + 4 * self.n

    def sample(self, t):
        n = self.n
        CTR.stages += 1
        w0 = fft(self.gz.continuous(n))
        w1 = fft(self.gz.continuous(n))
        p0 = self.A[:, 0, 0] * w0 + self.A[:, 0, 1] * w1
        p1 = self.A[:, 1, 0] * w0 + self.A[:, 1, 1] * w1
        c0 = ifft(t[0] - p0)
        c1 = ifft(t[1] - p1)
        z0 = self.gz.discrete_fixed(c0, self.r).astype(float)
        z1 = self.gz.discrete_fixed(c1, self.r).astype(float)
        return fft(z0), fft(z1)


class KleinSampler:
    name = "klein"

    def __init__(self, key, sigma, r, rng=None):
        self.key = key
        self.n = key.n
        self.sigma = sigma
        self.gz = GaussianZ(rng)
        n = key.n
        Mg = anticirculant(np.array([float(x) for x in key.g]))
        Mf = anticirculant(np.array([-float(x) for x in key.f]))
        MG = anticirculant(np.array([float(x) for x in key.G]))
        MF = anticirculant(np.array([-float(x) for x in key.F]))
        self.Bmat = np.block([[Mg, Mf], [MG, MF]])
        m = 2 * n
        Q = np.zeros((m, m))
        nrm = np.zeros(m)
        for i in range(m):
            v = self.Bmat[i].copy()
            if i > 0:
                coef = (Q[:i] @ self.Bmat[i]) / nrm[:i]
                v -= coef @ Q[:i]
            Q[i] = v
            nrm[i] = v @ v
        self.Q = Q
        self.nrm = nrm
        self.gsnorm = np.sqrt(nrm)

    def key_words(self):
        m = 2 * self.n
        return m * m + m * m + m

    def sample(self, t):
        n = self.n
        t0 = ifft(t[0])
        t1 = ifft(t[1])
        c = t0 @ self.Bmat[:n] + t1 @ self.Bmat[n:]
        m = 2 * n
        z = np.zeros(m)
        for i in range(m - 1, -1, -1):
            CTR.stages += 1
            d = (c @ self.Q[i]) / self.nrm[i]
            si = self.sigma / self.gsnorm[i]
            zi = self.gz.discrete_var(np.array([d]), np.array([si]))[0]
            z[i] = zi
            c = c - zi * self.Bmat[i]
        return fft(z[:n]), fft(z[n:])
