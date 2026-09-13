import numpy as np

from common import save
from saker.encoding import (entropy_bound_bytes, falcon_sig_bytes,
                            gaussian_table, range_decode, range_encode)
from saker.params import (ANTRAG_1024, ANTRAG_512, FALCON_1024, FALCON_512,
                          MITAKA_1024, MITAKA_512, SAKER_1024, SAKER_1024C,
                          SAKER_512, SALTLEN)


def sample_gaussian_vec(n, sigma, rng, tailcut=13.0):
    w = int(np.ceil(tailcut * sigma))
    xs = np.arange(-w, w + 1)
    p = np.exp(-(xs.astype(float) ** 2) / (2.0 * sigma * sigma))
    p /= p.sum()
    return rng.choice(xs, size=n, p=p)


def best_k(sigma):
    return max(1, int(np.round(np.log2(sigma))) - 1)


def run(sets, trials=64, seed=5):
    rng = np.random.default_rng(seed)
    rows = []
    for ps in sets:
        table = gaussian_table(ps.sigma)
        gr = []
        rc = []
        ok = True
        for _ in range(trials):
            v = sample_gaussian_vec(ps.n, ps.sigma, rng)
            gr.append(falcon_sig_bytes(v, best_k(ps.sigma)))
            data = range_encode(v, table)
            rc.append(int(np.ceil(len(data) / 8.0)))
            if not np.array_equal(range_decode(data, ps.n, table), v):
                ok = False
        row = dict(name=ps.name, n=ps.n, alpha=ps.alpha, sigma=float(ps.sigma),
                   entropy=float(entropy_bound_bytes(ps.sigma, ps.n)) + SALTLEN,
                   golomb=float(np.mean(gr)) + SALTLEN,
                   golomb_max=int(np.max(gr)) + SALTLEN,
                   rangecoder=float(np.mean(rc)) + SALTLEN,
                   rangecoder_max=int(np.max(rc)) + SALTLEN,
                   pk=ps.pk_bytes(), roundtrip=ok)
        rows.append(row)
        print("%-12s sigma=%7.2f  entropy=%7.1f  golomb=%7.1f (max %d)  "
              "range=%7.1f (max %d)  pk=%d  roundtrip=%s"
              % (ps.name, ps.sigma, row["entropy"], row["golomb"],
                 row["golomb_max"], row["rangecoder"], row["rangecoder_max"],
                 row["pk"], ok))
    return rows


if __name__ == "__main__":
    sets = [FALCON_512, MITAKA_512, ANTRAG_512, SAKER_512,
            FALCON_1024, MITAKA_1024, ANTRAG_1024, SAKER_1024, SAKER_1024C]
    save("sizes", run(sets))
