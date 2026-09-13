import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from saker.ring import fft, ifft, merge_fft, roots, split_fft


def test_fft_matches_evaluation():
    rng = np.random.default_rng(0)
    for n in (4, 16, 64):
        f = rng.standard_normal(n)
        w = roots(n)
        direct = np.array([sum(f[k] * z ** k for k in range(n)) for z in w])
        assert np.max(np.abs(fft(f) - direct)) < 1e-9


def test_roundtrip():
    rng = np.random.default_rng(1)
    for n in (2, 8, 128, 512):
        f = rng.standard_normal(n)
        assert np.max(np.abs(ifft(fft(f)) - f)) < 1e-9


def test_split_merge():
    rng = np.random.default_rng(2)
    for n in (8, 64, 256):
        a = fft(rng.standard_normal(n))
        a0, a1 = split_fft(a)
        assert np.max(np.abs(merge_fft(a0, a1) - a)) < 1e-9


def test_split_is_parity():
    rng = np.random.default_rng(3)
    n = 64
    f = rng.standard_normal(n)
    a0, a1 = split_fft(fft(f))
    assert np.max(np.abs(ifft(a0) - f[0::2])) < 1e-9
    assert np.max(np.abs(ifft(a1) - f[1::2])) < 1e-9
