import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from saker.keygen import annulus_candidate, gaussian_candidate, try_build
from saker.ntru import check_ntru
from saker.ring import ntt_mul_mod


def test_ntru_equation_and_public_key():
    rng = np.random.default_rng(5)
    q = 12289
    for n in (32, 64, 128):
        key = None
        while key is None:
            f, g = gaussian_candidate(n, q, rng)
            key = try_build(n, q, f, g)
        assert check_ntru(key.f, key.g, key.F, key.G, q)
        lhs = ntt_mul_mod(np.array([int(x) % q for x in key.f]), key.h, q)
        assert np.all(lhs % q == np.array([int(x) % q for x in key.g]) % q)


def test_annulus_keys():
    rng = np.random.default_rng(6)
    q, n = 12289, 64
    key = None
    while key is None:
        f, g = annulus_candidate(n, q, rng, np.sqrt(0.8 * q), np.sqrt(1.2 * q))
        key = try_build(n, q, f, g)
    assert check_ntru(key.f, key.g, key.F, key.G, q)
