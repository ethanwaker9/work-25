import time

import numpy as np

from common import save
from saker.keygen import (annulus_candidate, annulus_radii, gaussian_candidate,
                          try_build)
from saker.params import (ANTRAG_1024, ANTRAG_512, FALCON_1024, FALCON_512,
                          SAKER_1024, SAKER_1024C, SAKER_512)
from saker.quality import (alpha_fold, alpha_gpv, alpha_hybrid,
                           sigma_embeddings)


def build(ps, rng, keys=3, cap=40000):
    n, q = ps.n, ps.q
    if ps.family == "ffo":
        gen = lambda: gaussian_candidate(n, q, rng)
        qual = lambda S: alpha_gpv(S, q)
    elif ps.family == "hybrid":
        r, R = annulus_radii(q, ps.alpha, 0.4)
        gen = lambda: annulus_candidate(n, q, rng, r, R)
        qual = lambda S: alpha_hybrid(S, q)
    else:
        r, R = ps.radii()
        gen = lambda: annulus_candidate(n, q, rng, r, R)
        qual = lambda S: alpha_fold(S, q, ps.ell)
    tries = []
    tcand = []
    tsolve = []
    got = 0
    while got < keys:
        cnt = 0
        t0 = time.perf_counter()
        while True:
            cnt += 1
            f, g = gen()
            S = sigma_embeddings(f, g)
            if qual(S) <= ps.alpha:
                break
            if cnt > cap:
                return None
        tcand.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        key = try_build(n, q, f, g)
        if key is None:
            continue
        tsolve.append(time.perf_counter() - t0)
        tries.append(cnt)
        got += 1
    return dict(name=ps.name, n=n, alpha=ps.alpha,
                M=float(np.mean(tries)), cand_ms=1e3 * float(np.mean(tcand)),
                solve_ms=1e3 * float(np.mean(tsolve)),
                total_ms=1e3 * float(np.mean(tcand) + np.mean(tsolve)))


def run(seed=17):
    rng = np.random.default_rng(seed)
    rows = []
    for ps in [FALCON_512, ANTRAG_512, SAKER_512, FALCON_1024, ANTRAG_1024,
               SAKER_1024, SAKER_1024C]:
        r = build(ps, rng)
        if r is None:
            continue
        rows.append(r)
        print("%-12s M=%6.2f  candidates=%8.1f ms  NTRUSolve=%9.1f ms  total=%9.1f ms"
              % (r["name"], r["M"], r["cand_ms"], r["solve_ms"], r["total_ms"]))
    return rows


if __name__ == "__main__":
    save("keygen", run())
