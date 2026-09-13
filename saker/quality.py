import numpy as np

from .ring import fft, ifft, split_fft, anticirculant


def sigma_embeddings(f, g):
    ff = fft(np.array([float(x) for x in f]))
    gg = fft(np.array([float(x) for x in g]))
    return (np.abs(ff) ** 2 + np.abs(gg) ** 2).real


def block_means(vals, ell):
    b = 1 << ell
    m = len(vals) // b
    return vals.reshape(m, b).mean(axis=1)


def alpha_fold(sigma_emb, q, ell):
    a = block_means(sigma_emb, ell) / q
    b = block_means((q * q) / sigma_emb, ell) / q
    return float(np.sqrt(max(a.max(), b.max())))


def alpha_hybrid(sigma_emb, q):
    return alpha_fold(sigma_emb, q, 0)


def alpha_gpv(sigma_emb, q, n=None):
    ell = int(np.log2(len(sigma_emb)))
    return alpha_fold(sigma_emb, q, ell)


def alpha_peikert(sigma_emb, tfg, q):
    disc = np.sqrt(np.maximum(tfg ** 2 - 4.0 * q * q, 0.0))
    s1sq = 0.5 * (tfg + disc)
    return float(np.sqrt(s1sq.max() / q))


def fold_nodes(vals, ell):
    nodes = [np.asarray(vals, dtype=float)]
    for _ in range(ell):
        nxt = []
        for v in nodes:
            a = v[0::2]
            b = v[1::2]
            A = 0.5 * (a + b)
            nxt.append(A)
            nxt.append(a * b / A)
        nodes = nxt
    return nodes


def fold_leaves(vals):
    ell = int(np.log2(len(vals)))
    nodes = fold_nodes(vals, ell)
    return np.array([x[0] for x in nodes])


def gs_profile_exact(f, g, F, G):
    n = len(f)
    Mg = anticirculant(np.array([float(x) for x in g]))
    Mf = anticirculant(np.array([-float(x) for x in f]))
    MG = anticirculant(np.array([float(x) for x in G]))
    MF = anticirculant(np.array([-float(x) for x in F]))
    B = np.block([[Mg, Mf], [MG, MF]])
    m = 2 * n
    prof = np.empty(m)
    Q = np.zeros((m, m))
    for i in range(m):
        v = B[i].copy()
        for j in range(i):
            nj = Q[j] @ Q[j]
            if nj > 0:
                v -= (B[i] @ Q[j]) / nj * Q[j]
        Q[i] = v
        prof[i] = np.linalg.norm(v)
    return prof


def brev(i, b):
    return int(format(i, "0%db" % b)[::-1], 2)


def brev_order(n):
    b = int(np.log2(n))
    return np.array([brev(i, b) for i in range(n)])
