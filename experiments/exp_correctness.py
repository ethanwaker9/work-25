import numpy as np

from common import save
from saker.keygen import annulus_candidate, try_build
from saker.params import ETA
from saker.quality import alpha_fold, sigma_embeddings
from saker.scheme import Scheme, make_sampler

CONFIGS = [("Klein-GPV", "klein", None), ("Peikert", "peikert", None),
           ("hybrid", "folded", 0), ("folded-3", "folded", 3),
           ("folded-5", "folded", 5), ("folded-full", "folded", -1),
           ("FFO", "ffo", None)]


def make_key(n, q, rng):
    r, R = np.sqrt(0.8 * q), np.sqrt(1.2 * q)
    while True:
        f, g = annulus_candidate(n, q, rng, r, R)
        if alpha_fold(sigma_embeddings(f, g), q, 5) > 1.05:
            continue
        k = try_build(n, q, f, g)
        if k is not None:
            return k


def run(n=256, q=12289, nsig=400, seed=13):
    rng = np.random.default_rng(seed)
    key = make_key(n, q, rng)
    eta = ETA[512]
    rows = []
    for label, kind, depth in CONFIGS:
        d = int(np.log2(n)) if depth == -1 else depth
        a = (key.alpha_gpv() if kind in ("klein", "ffo")
             else key.alpha_peikert() if kind == "peikert" else key.alpha(d))
        a *= 1.0001
        sigma = a * eta * np.sqrt(q)
        smp = make_sampler(kind, key, sigma, eta, rng, d)
        sch = Scheme(key, smp, sigma, 1.3 * sigma * np.sqrt(2.0 * n))
        M = []
        ok = 0
        for i in range(nsig):
            msg = b"correctness-%d" % i
            salt, s2, s1 = sch.sign(msg, rng)
            ok += bool(sch.verify(msg, salt, s2))
            M.append(np.concatenate([s1, s2]))
        M = np.array(M, dtype=float)
        emp = float(np.sqrt((M ** 2).mean()))
        C = (M.T @ M) / M.shape[0]
        dia = np.diag(C)
        offd = float(np.abs(C - np.diag(dia)).max() / sigma ** 2)
        noise = float(np.sqrt(2.0 / M.shape[0]))
        rows.append(dict(label=label, n=n, alpha=float(a), sigma=float(sigma),
                         emp_sigma=emp, ratio=emp / sigma, verified=ok,
                         trials=nsig, offdiag=offd, noise_level=noise,
                         diag_spread=float(dia.std() / dia.mean())))
        print("%-12s alpha=%.4f sigma=%7.2f  emp/sigma=%.4f  verified=%d/%d  "
              "max offdiag=%.3f (noise %.3f)"
              % (label, a, sigma, emp / sigma, ok, nsig, offd, noise))
    return rows


if __name__ == "__main__":
    save("correctness", run())
