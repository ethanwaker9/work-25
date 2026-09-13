import hashlib

import numpy as np

from .gauss import CTR
from .ring import fft, ifft, ntt_mul_mod
from .samplers import (FFOSampler, FoldedSampler, HybridSampler, KleinSampler,
                       PeikertSampler)


def hash_to_point(msg, salt, n, q):
    out = np.zeros(n, dtype=np.int64)
    ctr = 0
    i = 0
    k = (1 << 16) // q * q
    while i < n:
        h = hashlib.shake_256(salt + msg + ctr.to_bytes(4, "big")).digest(2 * n + 64)
        for j in range(0, len(h) - 1, 2):
            v = (h[j] << 8) | h[j + 1]
            if v < k:
                out[i] = v % q
                i += 1
                if i == n:
                    break
        ctr += 1
    return out


class Scheme:
    def __init__(self, key, sampler, sigma, beta):
        self.key = key
        self.sampler = sampler
        self.sigma = sigma
        self.beta = beta
        self.n = key.n
        self.q = key.q

    def target(self, c):
        q = float(self.q)
        cf = fft(c.astype(float))
        t0 = -cf * self.key.F_fft / q
        t1 = cf * self.key.f_fft / q
        return t0, t1

    def sign_raw(self, c):
        t0, t1 = self.target(c)
        z0, z1 = self.sampler.sample((t0, t1))
        d0 = t0 - z0
        d1 = t1 - z1
        s0 = d0 * self.key.b1[0] + d1 * self.key.b2[0]
        s1 = d0 * self.key.b1[1] + d1 * self.key.b2[1]
        return np.rint(ifft(s0)).astype(np.int64), np.rint(ifft(s1)).astype(np.int64)

    def sign(self, msg, rng, maxtries=64):
        for _ in range(maxtries):
            salt = rng.integers(0, 256, 40, dtype=np.int64).astype(np.uint8).tobytes()
            c = hash_to_point(msg, salt, self.n, self.q)
            s1, s2 = self.sign_raw(c)
            nrm = float(np.dot(s1, s1) + np.dot(s2, s2))
            if nrm <= self.beta ** 2:
                return salt, s2, s1
        raise RuntimeError("signing failed")

    def verify(self, msg, salt, s2):
        c = hash_to_point(msg, salt, self.n, self.q)
        hs = ntt_mul_mod(np.asarray(s2, dtype=np.int64) % self.q, self.key.h, self.q)
        s1 = (c - hs) % self.q
        s1 = np.where(s1 > self.q // 2, s1 - self.q, s1)
        nrm = float(np.dot(s1, s1) + np.dot(s2.astype(np.int64), s2.astype(np.int64)))
        return nrm <= self.beta ** 2


def make_sampler(kind, key, sigma, r, rng, depth=None, lazy=False):
    if kind == "hybrid":
        return HybridSampler(key, sigma, r, rng)
    if kind == "ffo":
        return FFOSampler(key, sigma, r, rng)
    if kind == "peikert":
        return PeikertSampler(key, sigma, r, rng)
    if kind == "klein":
        return KleinSampler(key, sigma, r, rng)
    if kind == "folded":
        return FoldedSampler(key, sigma, r, depth, rng, True, lazy)
    raise ValueError(kind)
