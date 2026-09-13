import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from saker.keygen import gaussian_candidate, try_build
from saker.quality import (alpha_fold, alpha_gpv, alpha_hybrid, brev_order,
                           fold_leaves, sigma_embeddings)
from saker.ring import anticirculant


def tower_profile(key):
    n = key.n
    Mg = anticirculant(np.array([float(x) for x in key.g]))
    Mf = anticirculant(np.array([-float(x) for x in key.f]))
    MG = anticirculant(np.array([float(x) for x in key.G]))
    MF = anticirculant(np.array([-float(x) for x in key.F]))
    B = np.block([[Mg, Mf], [MG, MF]])
    o = brev_order(n)
    B = B[np.concatenate([o, n + o])]
    m = 2 * n
    Q = np.zeros((m, m))
    nrm = np.zeros(m)
    out = np.zeros(m)
    for i in range(m):
        v = B[i].copy()
        if i > 0:
            v -= ((Q[:i] @ B[i]) / nrm[:i]) @ Q[:i]
        Q[i] = v
        nrm[i] = v @ v
        out[i] = np.sqrt(nrm[i])
    return out


def test_folding_equals_gram_schmidt():
    rng = np.random.default_rng(11)
    q = 12289
    for n in (32, 64):
        key = None
        while key is None:
            f, g = gaussian_candidate(n, q, rng)
            key = try_build(n, q, f, g)
        S = sigma_embeddings(key.f, key.g)
        pred = np.concatenate([np.sqrt(fold_leaves(S)),
                               np.sqrt(fold_leaves((q * q) / S))])
        prof = tower_profile(key)
        assert np.max(np.abs(prof - pred)) / np.max(prof) < 1e-12


def test_alpha_interpolates():
    rng = np.random.default_rng(12)
    q, n = 12289, 128
    f, g = gaussian_candidate(n, q, rng)
    S = sigma_embeddings(f, g)
    vals = [alpha_fold(S, q, l) for l in range(int(np.log2(n)) + 1)]
    assert abs(vals[0] - alpha_hybrid(S, q)) < 1e-12
    assert abs(vals[-1] - alpha_gpv(S, q)) < 1e-12
    assert all(vals[i] >= vals[i + 1] - 1e-12 for i in range(len(vals) - 1))
    assert vals[-1] >= 1.0
