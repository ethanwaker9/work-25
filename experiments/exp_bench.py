import time

import numpy as np

from common import save
from saker.gauss import CTR
from saker.keygen import annulus_candidate, try_build
from saker.params import ETA
from saker.quality import alpha_fold, alpha_gpv, sigma_embeddings
from saker.ring import OPS, reset_ops
from saker.scheme import Scheme, make_sampler

CONFIGS = [
    ("Klein-GPV", "klein", None, False),
    ("Peikert", "peikert", None, False),
    ("hybrid", "folded", 0, False),
    ("folded-2", "folded", 2, False),
    ("folded-4", "folded", 4, False),
    ("folded-5", "folded", 5, False),
    ("folded-6", "folded", 6, False),
    ("folded-5-lazy", "folded", 5, True),
    ("folded-full", "folded", None, False),
    ("folded-full-lazy", "folded", None, True),
    ("FFO", "ffo", None, False),
]


def make_key(n, q, rng, m=1.0, w=0.2):
    r, R = np.sqrt(q * (m - w)), np.sqrt(q * (m + w))
    while True:
        f, g = annulus_candidate(n, q, rng, r, R)
        S = sigma_embeddings(f, g)
        if alpha_fold(S, q, 5) > 1.04:
            continue
        key = try_build(n, q, f, g)
        if key is not None:
            return key


def quality_for(key, kind, depth):
    if kind == "klein" or kind == "ffo":
        return key.alpha_gpv()
    if kind == "peikert":
        return key.alpha_peikert()
    return key.alpha(depth)


def bench(n, q, nsig=40, seed=11, include_klein=True):
    rng = np.random.default_rng(seed)
    key = make_key(n, q, rng)
    eta = ETA[n]
    rows = []
    for label, kind, depth, lazy in CONFIGS:
        if kind == "klein" and not include_klein:
            continue
        if label.startswith("folded-full"):
            depth = int(np.log2(n))
        a = quality_for(key, kind, depth) * 1.0001
        sigma = a * eta * np.sqrt(q)
        t0 = time.perf_counter()
        smp = make_sampler(kind, key, sigma, eta, rng, depth, lazy)
        tprep = time.perf_counter() - t0
        sch = Scheme(key, smp, sigma, 1.2 * sigma * np.sqrt(2.0 * n))
        cs = [np.array(rng.integers(0, q, n)) for _ in range(nsig)]
        sch.sign_raw(cs[0])
        CTR.reset()
        reset_ops()
        t0 = time.perf_counter()
        for c in cs:
            sch.sign_raw(c)
        tsig = (time.perf_counter() - t0) / nsig
        row = dict(label=label, n=n, alpha=float(a), sigma=float(sigma),
                   prep_ms=1e3 * tprep, sign_ms=1e3 * tsig,
                   words=int(smp.key_words()),
                   stages=CTR.stages / nsig, normals=CTR.normals / nsig,
                   dgauss=CTR.dgauss / nsig, bfly=OPS["bfly"] / nsig,
                   ringmul=CTR.ringmul / nsig, ringdiv=CTR.ringdiv / nsig)
        rows.append(row)
        print("%-14s n=%4d alpha=%.4f prep=%8.2fms sign=%7.3fms words=%7d "
              "stages=%6.0f normals=%6.0f dgauss=%6.0f bfly=%8.0f"
              % (label, n, a, row["prep_ms"], row["sign_ms"], row["words"],
                 row["stages"], row["normals"], row["dgauss"], row["bfly"]))
    return rows


if __name__ == "__main__":
    out = {}
    for n in (512, 1024):
        out[str(n)] = bench(n, 12289, include_klein=(n <= 512))
    save("bench", out)
