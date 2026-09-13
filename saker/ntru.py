import numpy as np

from .ring import fft, ifft, field_norm, lift, negacyclic, poly_inv_mod_q


def bitsize(a):
    val = abs(int(a))
    res = 0
    while val:
        res += 8
        val >>= 8
    return res


def maxbitsize(v):
    return max(53, max(bitsize(x) for x in v))


def galois_conj(a):
    return np.array([((-1) ** i) * int(a[i]) for i in range(len(a))], dtype=object)


def xgcd(a, b):
    old_r, r = int(a), int(b)
    old_s, s = 1, 0
    old_t, t = 0, 1
    while r != 0:
        quo = old_r // r
        old_r, r = r, old_r - quo * r
        old_s, s = s, old_s - quo * s
        old_t, t = t, old_t - quo * t
    return old_r, old_s, old_t


def shift_right(v, k):
    if k <= 0:
        return np.array([int(x) for x in v], dtype=object)
    return np.array([int(x) >> k for x in v], dtype=object)


def to_float(v):
    return np.array([float(x) for x in v], dtype=float)


def reduce_FG(f, g, F, G):
    n = len(f)
    size = max(maxbitsize(f), maxbitsize(g))
    fa = to_float(shift_right(f, size - 53))
    ga = to_float(shift_right(g, size - 53))
    fa_fft = fft(fa)
    ga_fft = fft(ga)
    den = fa_fft * np.conj(fa_fft) + ga_fft * np.conj(ga_fft)
    F = np.array([int(x) for x in F], dtype=object)
    G = np.array([int(x) for x in G], dtype=object)
    while True:
        Size = max(maxbitsize(F), maxbitsize(G))
        if Size < size:
            break
        Fa = to_float(shift_right(F, Size - 53))
        Ga = to_float(shift_right(G, Size - 53))
        num = fft(Fa) * np.conj(fa_fft) + fft(Ga) * np.conj(ga_fft)
        k = ifft(num / den)
        kk = np.array([int(round(x)) for x in k], dtype=object)
        if not kk.any():
            break
        fk = negacyclic(f, kk)
        gk = negacyclic(g, kk)
        sh = Size - size
        for i in range(n):
            F[i] = int(F[i]) - (int(fk[i]) << sh)
            G[i] = int(G[i]) - (int(gk[i]) << sh)
    return F, G


def ntru_solve(f, g, q):
    n = len(f)
    if n == 1:
        f0, g0 = int(f[0]), int(g[0])
        d, u, v = xgcd(f0, g0)
        if d != 1:
            return None
        return (np.array([-q * v], dtype=object), np.array([q * u], dtype=object))
    fp = field_norm(f)
    gp = field_norm(g)
    res = ntru_solve(fp, gp, q)
    if res is None:
        return None
    Fp, Gp = res
    F = negacyclic(lift(Fp), galois_conj(g))
    G = negacyclic(lift(Gp), galois_conj(f))
    F, G = reduce_FG(np.array([int(x) for x in f], dtype=object),
                     np.array([int(x) for x in g], dtype=object), F, G)
    return F, G


def check_ntru(f, g, F, G, q):
    n = len(f)
    lhs = negacyclic(np.array([int(x) for x in f], dtype=object),
                     np.array([int(x) for x in G], dtype=object))
    rhs = negacyclic(np.array([int(x) for x in g], dtype=object),
                     np.array([int(x) for x in F], dtype=object))
    diff = np.array([int(lhs[i]) - int(rhs[i]) for i in range(n)], dtype=object)
    target = np.zeros(n, dtype=object)
    target[0] = q
    return all(diff[i] == target[i] for i in range(n))


def public_key(f, g, q):
    n = len(f)
    finv = poly_inv_mod_q(np.array([int(x) % q for x in f], dtype=np.int64), q)
    if finv is None:
        return None
    from .ring import ntt_mul_mod
    h = ntt_mul_mod(np.array([int(x) % q for x in g], dtype=np.int64), finv, q)
    return np.array(h, dtype=np.int64) % q
