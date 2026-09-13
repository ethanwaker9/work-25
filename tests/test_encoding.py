import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from saker.encoding import (entropy_bound_bytes, gaussian_table,
                            golomb_rice_decode, golomb_rice_encode,
                            range_decode, range_encode)


def test_golomb_roundtrip():
    rng = np.random.default_rng(31)
    v = np.rint(rng.normal(0, 160, 512)).astype(np.int64)
    bits = golomb_rice_encode(v, 7)
    out, _ = golomb_rice_decode(bits, len(v), 7)
    assert np.array_equal(out, v)


def test_arithmetic_roundtrip_and_rate():
    rng = np.random.default_rng(32)
    sigma, n = 150.0, 512
    tb = gaussian_table(sigma)
    xs = tb[0]
    p = np.exp(-(xs.astype(float) ** 2) / (2 * sigma * sigma))
    p /= p.sum()
    lens = []
    for _ in range(5):
        v = rng.choice(xs, size=n, p=p)
        bits = range_encode(v, tb)
        assert np.array_equal(range_decode(bits, n, tb), v)
        lens.append(len(bits) / 8.0)
    assert np.mean(lens) - entropy_bound_bytes(sigma, n) < 4.0
