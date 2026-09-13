import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from saker.keygen import annulus_candidate, try_build
from saker.quality import alpha_fold, sigma_embeddings
from saker.scheme import Scheme, make_sampler


def build_key(n=64, q=12289, seed=21):
    rng = np.random.default_rng(seed)
    while True:
        f, g = annulus_candidate(n, q, rng, np.sqrt(0.8 * q), np.sqrt(1.2 * q))
        if alpha_fold(sigma_embeddings(f, g), q, 3) > 1.10:
            continue
        key = try_build(n, q, f, g)
        if key is not None:
            return key, rng


def test_samplers_verify_and_match_width():
    key, rng = build_key()
    q, n, eta = key.q, key.n, 1.2778
    for kind, depth in [("hybrid", None), ("folded", 2), ("folded", 4),
                        ("peikert", None), ("klein", None), ("ffo", None)]:
        a = (key.alpha_gpv() if kind in ("klein", "ffo")
             else key.alpha_peikert() if kind == "peikert"
             else key.alpha(0 if depth is None else depth)) * 1.0001
        sigma = a * eta * np.sqrt(q)
        smp = make_sampler(kind, key, sigma, eta, rng, depth)
        sch = Scheme(key, smp, sigma, 1.3 * sigma * np.sqrt(2.0 * n))
        acc = []
        for i in range(60):
            msg = b"unit-%d" % i
            salt, s2, s1 = sch.sign(msg, rng)
            assert sch.verify(msg, salt, s2)
            acc.append(np.concatenate([s1, s2]))
        emp = np.sqrt((np.array(acc, dtype=float) ** 2).mean())
        assert 0.93 < emp / sigma < 1.07


def test_lazy_matches_eager_memory_and_quality():
    key, rng = build_key()
    q, n, eta = key.q, key.n, 1.2778
    a = key.alpha(3) * 1.0001
    sigma = a * eta * np.sqrt(q)
    eager = make_sampler("folded", key, sigma, eta, rng, 3, False)
    lazy = make_sampler("folded", key, sigma, eta, rng, 3, True)
    assert lazy.key_words() < eager.key_words()
    sch = Scheme(key, lazy, sigma, 1.3 * sigma * np.sqrt(2.0 * n))
    for i in range(20):
        salt, s2, s1 = sch.sign(b"lazy-%d" % i, rng)
        assert sch.verify(b"lazy-%d" % i, salt, s2)
