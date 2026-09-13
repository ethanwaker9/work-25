import sys
import time

import numpy as np

from saker.encoding import gaussian_table, range_decode, range_encode
from saker.keygen import saker_keygen
from saker.params import SAKER_512, SALTLEN
from saker.scheme import Scheme, make_sampler


def main(n=256):
    ps = SAKER_512
    q = ps.q
    rng = np.random.default_rng(2024)
    r, R = np.sqrt(q * (ps.m - ps.w)), np.sqrt(q * (ps.m + ps.w))
    t0 = time.perf_counter()
    key, tries = saker_keygen(n, q, rng, ps.ell, ps.alpha, r, R)
    print("keygen: n=%d  candidates=%d  time=%.2fs" % (n, tries, time.perf_counter() - t0))
    print("        alpha_0=%.4f  alpha_%d=%.4f  alpha_GPV=%.4f  nu=%.4f"
          % (key.alpha(0), ps.ell, key.alpha(ps.ell), key.alpha_gpv(),
             key.norm_fg() / np.sqrt(q)))
    sigma = ps.alpha * ps.eta * np.sqrt(q)
    beta = 1.1 * sigma * np.sqrt(2.0 * n)
    smp = make_sampler("folded", key, sigma, ps.eta, rng, ps.ell, lazy=True)
    sch = Scheme(key, smp, sigma, beta)
    print("        signing state = %d words" % smp.key_words())
    table = gaussian_table(sigma)
    t0 = time.perf_counter()
    ok = 0
    total = 0
    for i in range(20):
        msg = b"demonstration message %d" % i
        salt, s2, s1 = sch.sign(msg, rng)
        ok += bool(sch.verify(msg, salt, s2))
        bits = range_encode(s2, table)
        total += int(np.ceil(len(bits) / 8.0)) + SALTLEN
        assert np.array_equal(range_decode(bits, n, table), s2)
    print("sign/verify: %d/20 valid, %.2f ms per signature, %.1f bytes on average"
          % (ok, 1e3 * (time.perf_counter() - t0) / 20, total / 20.0))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 256)
