import numpy as np

CLASSICAL = 0.292
QUANTUM = 0.257


_DELTA = {}


def delta_bkz(b):
    b = int(b)
    if b in _DELTA:
        return _DELTA[b]
    x = float(b)
    v = (((np.pi * x) ** (1.0 / x)) * x / (2.0 * np.pi * np.e)) ** (1.0 / (2.0 * (x - 1.0)))
    _DELTA[b] = v
    return v


def core_svp(b, quantum=False):
    return int(np.floor((QUANTUM if quantum else CLASSICAL) * b))


def forgery_blocksize(n, q, beta, bmax=4000):
    dim = 2 * n
    lt = np.log(beta / np.sqrt(q))
    lo, hi = 50, bmax
    while lo < hi:
        mid = (lo + hi) // 2
        if dim * np.log(delta_bkz(mid)) <= lt:
            hi = mid
        else:
            lo = mid + 1
    return lo


def key_recovery_blocksize(n, q, normfg, bmax=4000):
    A = (normfg ** 2) / q
    lo, hi = 50, bmax
    while lo < hi:
        mid = (lo + hi) // 2
        rhs = np.log(8.0 * n / (3.0 * mid)) + (4 * (mid - n) + 2) * np.log(delta_bkz(mid))
        if np.log(A) <= rhs:
            hi = mid
        else:
            lo = mid + 1
    return lo


def subfield_blocksize(n, q, x, y):
    eta = (y / x) ** (1.0 / 3.0)
    tgt = 3.0 * q * q * (x * x * y) ** (1.0 / 3.0)
    vol2 = 2.0 * q * np.sqrt(eta ** 2 + 2.0 / eta)
    for b in range(50, n + 1):
        lhs = (b / float(n)) * tgt
        rhs = (4.0 / 3.0) * vol2 * delta_bkz(b) ** (-4 * (n - b) + 2)
        if lhs <= rhs:
            return b
    return None


def key_stats(key):
    q = key.q
    f = key.f_fft
    g = key.g_fft
    S = (np.abs(f) ** 2 + np.abs(g) ** 2).real
    ffs = np.abs(f) ** 2
    ggs = np.abs(g) ** 2
    x = float(np.mean(ffs ** 2 + ggs ** 2) / (2.0 * q * q))
    y = float(np.var(S) / (q * q))
    return x, y


def sig_bound(sigma, n, tau=1.1):
    return tau * sigma * np.sqrt(2.0 * n)


def evaluate(n, q, alpha, eta, normfg, x, y, tau=1.1, quantum=False):
    sigma = alpha * eta * np.sqrt(q)
    beta = sig_bound(sigma, n, tau)
    bf = forgery_blocksize(n, q, beta)
    bk = key_recovery_blocksize(n, q, normfg)
    bs = subfield_blocksize(n, q, x, y) if y > 0 else None
    sub = core_svp(bs, quantum) if bs is not None else None
    bits = min(core_svp(bf, quantum), core_svp(bk, quantum))
    if sub is not None:
        bits = min(bits, sub)
    return dict(sigma=sigma, beta=beta,
                forgery=core_svp(bf, quantum), key=core_svp(bk, quantum),
                subfield=sub, b_forgery=bf, b_key=bk, b_sub=bs, bits=bits)


def signature_entropy_bits(sigma, n):
    return n * (0.5 * np.log2(2.0 * np.pi * np.e) + np.log2(sigma))


