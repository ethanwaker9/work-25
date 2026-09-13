import numpy as np

from common import save
from saker.keygen import annulus_candidate, annulus_radii, gaussian_candidate
from saker.params import (ANTRAG_1024, ANTRAG_512, ETA, FALCON_1024,
                          FALCON_512, MITAKA_1024, MITAKA_512, SAKER_1024,
                          SAKER_1024C, SAKER_512, TAU)
from saker.quality import (alpha_fold, alpha_gpv, alpha_hybrid,
                           sigma_embeddings)
from saker.security import core_svp, evaluate, key_stats
from saker.ring import fft


def candidate_family(ps, rng):
    n, q = ps.n, ps.q
    if ps.family == "ffo":
        return lambda: gaussian_candidate(n, q, rng)
    if ps.family == "hybrid":
        r, R = annulus_radii(q, ps.alpha, 0.4)
        return lambda: annulus_candidate(n, q, rng, r, R)
    r, R = ps.radii()
    return lambda: annulus_candidate(n, q, rng, r, R)


def quality_family(ps):
    if ps.family == "ffo":
        return lambda S: alpha_gpv(S, ps.q)
    if ps.family == "hybrid":
        return lambda S: alpha_hybrid(S, ps.q)
    return lambda S: alpha_fold(S, ps.q, ps.ell)


def measure(ps, rng, trials=600):
    gen = candidate_family(ps, rng)
    qual = quality_family(ps)
    A, NU, X, Y = [], [], [], []
    for _ in range(trials):
        f, g = gen()
        S = sigma_embeddings(f, g)
        A.append(qual(S))
        NU.append(np.sqrt(S.mean() / ps.q))
        ff = fft(np.asarray(f, dtype=float))
        gg = fft(np.asarray(g, dtype=float))
        X.append(np.mean(np.abs(ff) ** 4 + np.abs(gg) ** 4) / (2.0 * ps.q ** 2))
        Y.append(np.var(S) / ps.q ** 2)
    A = np.array(A)
    sel = A <= ps.alpha
    if sel.sum() < 5:
        sel = A <= np.percentile(A, 25)
    return dict(acc=float(sel.mean()), nu=float(np.median(np.array(NU)[sel])),
                x=float(np.median(np.array(X)[sel])), y=float(np.median(np.array(Y)[sel])),
                alpha_med=float(np.median(A)))


def run(seed=99):
    rng = np.random.default_rng(seed)
    rows = []
    for ps in [FALCON_512, MITAKA_512, ANTRAG_512, SAKER_512,
               FALCON_1024, MITAKA_1024, ANTRAG_1024, SAKER_1024, SAKER_1024C]:
        if ps.nu_hint is not None:
            st = dict(acc=float("nan"), nu=ps.nu_hint, x=0.3, y=0.5, alpha_med=ps.alpha)
        else:
            st = measure(ps, rng)
        ev = evaluate(ps.n, ps.q, ps.alpha, ps.eta, st["nu"] * np.sqrt(ps.q),
                      st["x"], st["y"], TAU)
        evq = evaluate(ps.n, ps.q, ps.alpha, ps.eta, st["nu"] * np.sqrt(ps.q),
                       st["x"], st["y"], TAU, quantum=True)
        row = dict(name=ps.name, n=ps.n, q=ps.q, family=ps.family, ell=ps.ell,
                   alpha=ps.alpha, sigma=float(ps.sigma), beta=float(ps.beta),
                   nu=st["nu"], acc=st["acc"], x=st["x"], y=st["y"],
                   forgery=ev["forgery"], key=ev["key"],
                   subfield=ev["subfield"], bits=ev["bits"], bits_q=evq["bits"],
                   b_forgery=ev["b_forgery"], b_key=ev["b_key"],
                   pk=ps.pk_bytes(), published=ps.published)
        rows.append(row)
        print("%-12s alpha=%.4f nu=%.4f acc=%.3f y=%.4f  frg=%3d key=%3d sub=%s -> %3d C / %3d Q"
              % (ps.name, ps.alpha, st["nu"], st["acc"], st["y"], ev["forgery"],
                 ev["key"], str(ev["subfield"]), ev["bits"], evq["bits"]))
    return rows


def frontier(n=512, q=12289, seed=5, trials=200):
    rng = np.random.default_rng(seed)
    eta = ETA[n]
    out = {}
    for fam, ell in [("hybrid", 0), ("folded", 3), ("folded", 5), ("gaussian", None)]:
        pts = []
        for m in np.arange(0.9, 1.75, 0.05):
            if fam == "gaussian":
                sig = np.sqrt(m * q / (2.0 * n))
                gen = lambda: gaussian_candidate(n, q, rng, sig)
                qf = lambda S: alpha_gpv(S, q)
            else:
                r, R = np.sqrt(q * max(m - 0.2, 0.05)), np.sqrt(q * (m + 0.2))
                gen = lambda: annulus_candidate(n, q, rng, r, R)
                qf = (lambda S: alpha_hybrid(S, q)) if fam == "hybrid" else (lambda S: alpha_fold(S, q, ell))
            A, NU = [], []
            for _ in range(trials):
                f, g = gen()
                S = sigma_embeddings(f, g)
                A.append(qf(S))
                NU.append(np.sqrt(S.mean() / q))
            A = np.array(A)
            NU = np.array(NU)
            thr = float(np.percentile(A, 25))
            nu = float(np.median(NU[A <= thr]))
            ev = evaluate(n, q, thr, eta, nu * np.sqrt(q), 0.3, 0.03, TAU)
            h = n * (0.5 * np.log2(2 * np.pi * np.e) + np.log2(ev["sigma"])) / 8.0 + 40
            pts.append(dict(m=float(m), nu=nu, alpha=thr, bits=ev["bits"],
                            forgery=ev["forgery"], key=ev["key"], sig=float(h)))
        key = fam if ell is None else "%s-%d" % (fam, ell)
        out[key] = pts
        print(key, " ".join("(%d,%.0f)" % (p["bits"], p["sig"]) for p in pts))
    return out


if __name__ == "__main__":
    save("params", dict(table=run(), frontier=frontier()))
