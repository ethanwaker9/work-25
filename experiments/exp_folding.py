import numpy as np

from common import save
from saker.keygen import annulus_candidate, gaussian_candidate, try_build
from saker.quality import (alpha_fold, alpha_gpv, brev_order, fold_leaves,
                           fold_nodes, gs_profile_exact, sigma_embeddings)
from saker.ring import anticirculant


def tower_gs(key):
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
    prof = np.empty(m)
    nrm = np.zeros(m)
    for i in range(m):
        v = B[i].copy()
        if i > 0:
            coef = (Q[:i] @ B[i]) / nrm[:i]
            v -= coef @ Q[:i]
        Q[i] = v
        nrm[i] = v @ v
        prof[i] = np.sqrt(nrm[i])
    return prof


def run(dims=(32, 64, 128, 256), trials=4, q=12289, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for n in dims:
        errs = []
        for _ in range(trials):
            key = None
            while key is None:
                f, g = annulus_candidate(n, q, rng, np.sqrt(0.8 * q), np.sqrt(1.2 * q))
                key = try_build(n, q, f, g)
            S = sigma_embeddings(key.f, key.g)
            pred = np.concatenate([np.sqrt(fold_leaves(S)),
                                   np.sqrt(fold_leaves((q * q) / S))])
            prof = tower_gs(key)
            errs.append(float(np.max(np.abs(prof - pred)) / np.max(prof)))
        rows.append(dict(n=n, max_rel_err=float(np.max(errs)),
                         med_rel_err=float(np.median(errs))))
        print("n=%4d  max relative deviation %.3e" % (n, rows[-1]["max_rel_err"]))
    return rows


def interpolation(n=512, q=12289, seed=7, trials=64):
    rng = np.random.default_rng(seed)
    out = {}
    fams = {
        "gaussian": lambda: gaussian_candidate(n, q, rng),
        "annulus-wide": lambda: annulus_candidate(n, q, rng, np.sqrt(0.7 * q), np.sqrt(1.3 * q)),
        "annulus-narrow": lambda: annulus_candidate(n, q, rng, np.sqrt(0.8 * q), np.sqrt(1.2 * q)),
    }
    for name, gen in fams.items():
        acc = []
        for _ in range(trials):
            f, g = gen()
            S = sigma_embeddings(f, g)
            acc.append([alpha_fold(S, q, l) for l in range(int(np.log2(n)) + 1)])
        acc = np.array(acc)
        out[name] = dict(median=list(np.median(acc, axis=0)),
                         p25=list(np.percentile(acc, 25, axis=0)),
                         p75=list(np.percentile(acc, 75, axis=0)))
        print("%-16s alpha_ell median:" % name,
              " ".join("%.4f" % v for v in np.median(acc, axis=0)))
    return out


if __name__ == "__main__":
    res = dict(validation=run(), interpolation=interpolation())
    save("folding", res)
