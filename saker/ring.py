import numpy as np

_ROOTS = {}

OPS = {"bfly": 0, "cmul": 0, "cdiv": 0}


def reset_ops():
    for k in OPS:
        OPS[k] = 0


def roots(n):
    if n in _ROOTS:
        return _ROOTS[n]
    if n == 1:
        out = np.array([1.0 + 0.0j])
    elif n == 2:
        out = np.array([1.0j, -1.0j])
    else:
        prev = roots(n // 2)
        out = np.empty(n, dtype=complex)
        ang = np.angle(prev) / 2.0
        half = np.exp(1.0j * ang)
        out[0::2] = half
        out[1::2] = -half
    _ROOTS[n] = out
    return out


_BREV = {}


def brev_perm(n):
    if n in _BREV:
        return _BREV[n]
    b = int(np.log2(n))
    p = np.array([int(format(i, "0%db" % b)[::-1], 2) for i in range(n)]) if b > 0 else np.array([0])
    _BREV[n] = p
    return p


def fft(f):
    f = np.asarray(f, dtype=float)
    n = len(f)
    if n == 1:
        return np.array([complex(f[0])])
    S = f[brev_perm(n)].astype(complex).reshape(n, 1)
    m = 1
    while m < n:
        w = roots(2 * m)[0::2]
        a0 = S[0::2]
        a1 = S[1::2] * w[None, :]
        out = np.empty((S.shape[0] // 2, 2 * m), dtype=complex)
        out[:, 0::2] = a0 + a1
        out[:, 1::2] = a0 - a1
        S = out
        OPS["bfly"] += n // 2
        m *= 2
    return S[0]


def ifft(a):
    a = np.asarray(a, dtype=complex)
    n = len(a)
    if n == 1:
        return np.array([a[0].real])
    S = a.reshape(1, n)
    m = n
    while m > 1:
        w = roots(m)[0::2]
        e = S[:, 0::2]
        o = S[:, 1::2]
        a0 = 0.5 * (e + o)
        a1 = 0.5 * (e - o) / w[None, :]
        out = np.empty((S.shape[0] * 2, m // 2), dtype=complex)
        out[0::2] = a0
        out[1::2] = a1
        S = out
        OPS["bfly"] += n // 2
        m //= 2
    res = np.empty(n, dtype=float)
    res[brev_perm(n)] = S[:, 0].real
    return res


def split_fft(a):
    n = len(a)
    w = roots(n)[0::2]
    a0 = 0.5 * (a[0::2] + a[1::2])
    a1 = 0.5 * (a[0::2] - a[1::2]) / w
    OPS["bfly"] += n // 2
    return a0, a1


def merge_fft(a0, a1):
    n = 2 * len(a0)
    w = roots(n)[0::2]
    out = np.empty(n, dtype=complex)
    out[0::2] = a0 + w * a1
    out[1::2] = a0 - w * a1
    OPS["bfly"] += n // 2
    return out


def split_coef(f):
    return np.array(f[0::2]), np.array(f[1::2])








def negacyclic(a, b):
    n = len(a)
    c = np.convolve(np.asarray(a, dtype=object), np.asarray(b, dtype=object))
    out = np.zeros(n, dtype=object)
    for i in range(n):
        out[i] += c[i]
    for i in range(n, len(c)):
        out[i - n] -= c[i]
    return out




def field_norm(f):
    f0, f1 = split_coef(f)
    n2 = len(f0)
    t0 = negacyclic(f0, f0)
    t1 = negacyclic(f1, f1)
    out = np.zeros(n2, dtype=object)
    for i in range(n2):
        out[i] = t0[i]
    shifted = np.zeros(n2, dtype=object)
    shifted[0] = -t1[n2 - 1]
    for i in range(1, n2):
        shifted[i] = t1[i - 1]
    for i in range(n2):
        out[i] = out[i] - shifted[i]
    return out


def lift(f):
    n = len(f)
    out = np.zeros(2 * n, dtype=object)
    out[0::2] = f
    return out


def anticirculant(a):
    n = len(a)
    a = np.asarray(a, dtype=float)
    M = np.empty((n, n))
    for i in range(n):
        M[i, i:] = a[: n - i]
        if i > 0:
            M[i, :i] = -a[n - i:]
    return M


def ntt_mul_mod(a, b, q):
    n = len(a)
    c = np.zeros(n, dtype=np.int64)
    aa = np.asarray(a, dtype=np.int64) % q
    bb = np.asarray(b, dtype=np.int64) % q
    full = np.convolve(aa, bb)
    for i in range(n):
        c[i] = (full[i] - (full[i + n] if i + n < len(full) else 0)) % q
    return c


def poly_inv_mod_q(f, q):
    n = len(f)
    p = [0] * (n + 1)
    p[0] = 1
    p[n] = 1
    a = [int(x) % q for x in f]
    while len(a) > 0 and a[-1] == 0:
        a.pop()
    if not a:
        return None
    r0, r1 = p[:], a[:]
    s0, s1 = [0], [1]
    while any(r1):
        qt, rem = _polydivmod(r0, r1, q)
        r0, r1 = r1, rem
        s0, s1 = s1, _polysub(s0, _polymul(qt, s1, q), q)
    if len(_trim(r0)) != 1:
        return None
    c = pow(r0[0] % q, q - 2, q)
    inv = [(c * x) % q for x in s0]
    inv = inv[:n] + [0] * max(0, n - len(inv))
    return np.array(inv[:n], dtype=np.int64)


def _trim(a):
    a = a[:]
    while len(a) > 1 and a[-1] == 0:
        a.pop()
    return a


def _polymul(a, b, q):
    if not a or not b:
        return [0]
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                out[i + j] = (out[i + j] + x * y) % q
    return out


def _polysub(a, b, q):
    m = max(len(a), len(b))
    out = [0] * m
    for i in range(m):
        u = a[i] if i < len(a) else 0
        v = b[i] if i < len(b) else 0
        out[i] = (u - v) % q
    return _trim(out)


def _polydivmod(a, b, q):
    a = _trim(a[:])
    b = _trim(b[:])
    if len(b) == 1 and b[0] == 0:
        raise ZeroDivisionError
    out = [0] * max(1, len(a) - len(b) + 1)
    binv = pow(b[-1] % q, q - 2, q)
    while len(a) >= len(b) and any(a):
        d = len(a) - len(b)
        c = (a[-1] * binv) % q
        out[d] = c
        for i in range(len(b)):
            a[i + d] = (a[i + d] - c * b[i]) % q
        a = _trim(a)
        if len(a) < len(b):
            break
    return _trim(out), _trim(a)
