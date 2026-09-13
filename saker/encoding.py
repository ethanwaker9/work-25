import numpy as np


def golomb_rice_encode(coeffs, k=7):
    bits = []
    for c in coeffs:
        c = int(c)
        s = 1 if c < 0 else 0
        a = abs(c)
        bits.append(s)
        lo = a & ((1 << k) - 1)
        for i in range(k - 1, -1, -1):
            bits.append((lo >> i) & 1)
        hi = a >> k
        bits.extend([0] * hi)
        bits.append(1)
    return bits


def golomb_rice_decode(bits, n, k=7):
    out = []
    p = 0
    for _ in range(n):
        s = bits[p]
        p += 1
        lo = 0
        for _ in range(k):
            lo = (lo << 1) | bits[p]
            p += 1
        hi = 0
        while bits[p] == 0:
            hi += 1
            p += 1
        p += 1
        a = (hi << k) | lo
        out.append(-a if s else a)
    return np.array(out, dtype=np.int64), p


def gaussian_table(sigma, prec=24, tailcut=13.0):
    w = int(np.ceil(tailcut * sigma))
    xs = np.arange(-w, w + 1)
    p = np.exp(-(xs.astype(float) ** 2) / (2.0 * sigma * sigma))
    p /= p.sum()
    tot = 1 << prec
    f = np.maximum(np.floor(p * tot).astype(np.int64), 1)
    d = tot - int(f.sum())
    order = np.argsort(-p)
    i = 0
    while d != 0:
        j = order[i % len(order)]
        step = 1 if d > 0 else -1
        if f[j] + step >= 1:
            f[j] += step
            d -= step
        i += 1
    cum = np.concatenate([[0], np.cumsum(f)])
    return xs, f, cum, tot, w


TOP = 1 << 63
HALF = 1 << 62
QTR = 1 << 61
TQTR = 3 * (1 << 61)


class ACEncoder:
    def __init__(self):
        self.low = 0
        self.high = TOP - 1
        self.pending = 0
        self.bits = []

    def _out(self, b):
        self.bits.append(b)
        while self.pending > 0:
            self.bits.append(1 - b)
            self.pending -= 1

    def encode(self, lo, hi, tot):
        rng = self.high - self.low + 1
        self.high = self.low + (rng * hi) // tot - 1
        self.low = self.low + (rng * lo) // tot
        while True:
            if self.high < HALF:
                self._out(0)
            elif self.low >= HALF:
                self._out(1)
                self.low -= HALF
                self.high -= HALF
            elif self.low >= QTR and self.high < TQTR:
                self.pending += 1
                self.low -= QTR
                self.high -= QTR
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1

    def finish(self):
        self.pending += 1
        self._out(0 if self.low < QTR else 1)
        return self.bits


class ACDecoder:
    def __init__(self, bits):
        self.bits = bits
        self.pos = 0
        self.low = 0
        self.high = TOP - 1
        self.value = 0
        for _ in range(63):
            self.value = (self.value << 1) | self._bit()

    def _bit(self):
        if self.pos < len(self.bits):
            b = self.bits[self.pos]
            self.pos += 1
            return b
        return 0

    def target(self, tot):
        rng = self.high - self.low + 1
        return ((self.value - self.low + 1) * tot - 1) // rng

    def update(self, lo, hi, tot):
        rng = self.high - self.low + 1
        self.high = self.low + (rng * hi) // tot - 1
        self.low = self.low + (rng * lo) // tot
        while True:
            if self.high < HALF:
                pass
            elif self.low >= HALF:
                self.low -= HALF
                self.high -= HALF
                self.value -= HALF
            elif self.low >= QTR and self.high < TQTR:
                self.low -= QTR
                self.high -= QTR
                self.value -= QTR
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1
            self.value = (self.value << 1) | self._bit()


def range_encode(coeffs, table):
    xs, f, cum, tot, w = table
    enc = ACEncoder()
    for c in coeffs:
        i = int(c) + w
        enc.encode(int(cum[i]), int(cum[i + 1]), tot)
    return enc.finish()


def range_decode(bits, n, table):
    xs, f, cum, tot, w = table
    dec = ACDecoder(bits)
    out = np.empty(n, dtype=np.int64)
    for j in range(n):
        t = dec.target(tot)
        i = int(np.searchsorted(cum, t, side="right") - 1)
        dec.update(int(cum[i]), int(cum[i + 1]), tot)
        out[j] = i - w
    return out


def falcon_sig_bytes(coeffs, k=7):
    return int(np.ceil(len(golomb_rice_encode(coeffs, k)) / 8.0))


def entropy_bound_bytes(sigma, n):
    h = 0.5 * np.log2(2.0 * np.pi * np.e) + np.log2(sigma)
    return n * h / 8.0
