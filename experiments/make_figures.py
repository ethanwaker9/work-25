import os
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import FIGURES, ensure_dirs, load
from saker.keygen import annulus_candidate, gaussian_candidate, try_build
from saker.quality import (brev_order, fold_leaves, sigma_embeddings)
from saker.ring import anticirculant

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "figure.autolayout": True,
})


def finish(fig, name):
    ensure_dirs()
    eps = os.path.join(FIGURES, name + ".eps")
    fig.savefig(eps, format="eps")
    plt.close(fig)
    subprocess.run(["epstopdf", eps, "--outfile=" + eps[:-4] + ".pdf"], check=True)
    print("wrote", name)


def fig_alpha_ell():
    d = load("folding")["interpolation"]
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    styles = {"gaussian": ("o-", "discrete Gaussian keys"),
              "annulus-wide": ("s--", r"annular keys, $w=0.3$"),
              "annulus-narrow": ("^-.", r"annular keys, $w=0.2$")}
    for k, (st, lab) in styles.items():
        y = np.array(d[k]["median"])
        ax.plot(np.arange(len(y)), y, st, label=lab, color="k",
                markerfacecolor="none" if k != "gaussian" else "k")
    ax.axhline(1.17, ls=":", color="0.4")
    ax.text(0.15, 1.19, r"Falcon $1.17$", fontsize=7, color="0.3")
    ax.axhline(1.0, ls="-", color="0.7", lw=0.8)
    ax.text(4.0, 1.012, r"optimum $\alpha=1$", fontsize=7, color="0.4")
    ax.set_yscale("log")
    ax.set_yticks([1.0, 1.2, 1.5, 2.0, 3.0, 4.0])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel(r"folding depth $\ell$")
    ax.set_ylabel(r"quality $\alpha_\ell$")
    ax.legend(loc="upper right", fontsize=6.5)
    ax.grid(True, which="both", ls=":", lw=0.4)
    finish(fig, "alpha_ell")


def fig_gsprofile(n=256, q=12289, seed=3):
    rng = np.random.default_rng(seed)
    key = None
    while key is None:
        f, g = gaussian_candidate(n, q, rng)
        key = try_build(n, q, f, g)
    S = sigma_embeddings(key.f, key.g)
    pred = np.concatenate([np.sqrt(fold_leaves(S)), np.sqrt(fold_leaves((q * q) / S))])
    Mg = anticirculant(np.array([float(x) for x in key.g]))
    Mf = anticirculant(np.array([-float(x) for x in key.f]))
    MG = anticirculant(np.array([float(x) for x in key.G]))
    MF = anticirculant(np.array([-float(x) for x in key.F]))
    B = np.block([[Mg, Mf], [MG, MF]])
    o = brev_order(n)
    B = B[np.concatenate([o, n + o])]
    m = 2 * n
    Q = np.zeros((m, m))
    nrm = np.zeros(m)
    prof = np.zeros(m)
    for i in range(m):
        v = B[i].copy()
        if i > 0:
            v -= ((Q[:i] @ B[i]) / nrm[:i]) @ Q[:i]
        Q[i] = v
        nrm[i] = v @ v
        prof[i] = np.sqrt(nrm[i])
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5))
    ax = axes[0]
    ax.plot(prof / np.sqrt(q), "k-", label="Gram\u2013Schmidt norms")
    ax.plot(pred / np.sqrt(q), "r--", label="folding leaves")
    ax.set_xlabel("index in tower order")
    ax.set_ylabel(r"$\|\tilde b_i\|/\sqrt{q}$")
    ax.legend(loc="upper right")
    ax.grid(True, ls=":", lw=0.4)
    ax = axes[1]
    rel = np.abs(prof - pred) / prof
    ax.semilogy(np.maximum(rel, 1e-17), "k.", ms=2)
    ax.set_xlabel("index in tower order")
    ax.set_ylabel("relative deviation")
    ax.set_ylim(1e-17, 1e-13)
    ax.grid(True, which="both", ls=":", lw=0.4)
    finish(fig, "gsprofile")


def _panel(ax, rows, field, ylabel, logy=True):
    fam = sorted([r for r in rows if r["label"].startswith("folded")
                  and "lazy" not in r["label"]] +
                 [r for r in rows if r["label"] == "hybrid"],
                 key=lambda r: r["alpha"], reverse=True)
    ax.plot([r["alpha"] for r in fam], [r[field] for r in fam], "ko-",
            markerfacecolor="w", label=r"folded, depth $\ell=0,2,4,5,6,\log_2 n$")
    lz = sorted([r for r in rows if "lazy" in r["label"]], key=lambda r: r["alpha"])
    if lz:
        ax.plot([r["alpha"] for r in lz], [r[field] for r in lz], "kD", ms=5,
                label="folded, lazy tree")
    ffo = [r for r in rows if r["label"] == "FFO"][0]
    ax.plot(ffo["alpha"], ffo[field], "ks", ms=7, label="FFO (Falcon)")
    hyb = [r for r in rows if r["label"] == "hybrid"][0]
    ax.plot(hyb["alpha"], hyb[field], "k^", ms=7, label="hybrid (Mitaka, Antrag)")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(r"quality $\alpha$")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", ls=":", lw=0.4)


def fig_bench():
    d = load("bench")
    for n in ("512", "1024"):
        rows = d[n]
        fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8))
        _panel(axes[0], rows, "sign_ms", "signing time (ms)")
        _panel(axes[1], rows, "words", "stored words")
        kl = [r for r in rows if r["label"] == "Klein-GPV"]
        if kl:
            axes[0].annotate("Klein\u2013GPV: %.1f ms" % kl[0]["sign_ms"],
                             (0.40, 0.94), xycoords="axes fraction", fontsize=7)
            axes[1].annotate("Klein\u2013GPV: %d words" % kl[0]["words"],
                             (0.40, 0.94), xycoords="axes fraction", fontsize=7)
        axes[0].legend(loc="lower left", fontsize=6.5)
        axes[1].set_ylim(2e3, 3e4)
        fig.suptitle(r"$n=%s$" % n, fontsize=9)
        finish(fig, "bench_%s" % n)


def fig_pareto():
    d = load("params")["frontier"]
    styles = {"gaussian": ("^:", "Gaussian keys"),
              "hybrid-0": ("v--", r"annular keys, $\ell=0$"),
              "folded-3": ("s-.", r"annular keys, $\ell=3$"),
              "folded-5": ("o-", r"annular keys, $\ell=5$")}
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    nus = np.linspace(0.93, 1.35, 50)
    ax.plot(nus, nus, "-", color="0.6", lw=1.0)
    ax.text(1.20, 1.165, r"bound $\alpha=\nu$", fontsize=7, color="0.4", rotation=30)
    for k, (st, lab) in styles.items():
        pts = d[k]
        ax.plot([p["nu"] for p in pts], [p["alpha"] for p in pts], st, color="k",
                markerfacecolor="w", label=lab)
    ax.set_xlabel(r"$\nu=\|(f,g)\|/\sqrt{q}$")
    ax.set_ylabel(r"attained quality $\alpha$")
    ax.set_xlim(0.93, 1.35)
    ax.set_ylim(0.95, 1.55)
    ax.grid(True, ls=":", lw=0.4)
    ax.legend(loc="lower right", fontsize=6.5)
    finish(fig, "bound")
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    for k, (st, lab) in styles.items():
        pts = d[k]
        ax.plot([p["nu"] for p in pts], [p["bits"] for p in pts], st, color="k",
                markerfacecolor="w", label=lab)
        b = max(pts, key=lambda p: p["bits"])
        ax.plot([b["nu"]], [b["bits"]], "k*", ms=10)
    ax.set_xlabel(r"$\nu=\|(f,g)\|/\sqrt{q}$")
    ax.set_ylabel("classical CoreSVP security (bits)")
    ax.set_xlim(0.93, 1.35)
    ax.grid(True, ls=":", lw=0.4)
    ax.legend(loc="lower right", fontsize=6.5)
    finish(fig, "secbits")


def fig_spread(n=512, q=12289, seed=21):
    rng = np.random.default_rng(seed)
    f, g = gaussian_candidate(n, q, rng)
    Sg = sigma_embeddings(f, g) / q
    f, g = annulus_candidate(n, q, rng, np.sqrt(0.8 * q), np.sqrt(1.2 * q))
    Sa = sigma_embeddings(f, g) / q
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    bins = np.linspace(0, 5, 60)
    ax.hist(Sg, bins=bins, histtype="step", color="k", ls="-",
            label="discrete Gaussian keys", density=True)
    ax.hist(Sa, bins=bins, histtype="step", color="k", ls="--",
            label="annular keys", density=True)
    ax.axvline(1.0, color="0.6", lw=0.8)
    ax.set_xlabel(r"$\Sigma(\zeta)/q$")
    ax.set_ylabel("density")
    ax.set_xlim(0, 4)
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, ls=":", lw=0.4)
    finish(fig, "spread")


def fig_stages():
    d = load("bench")
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    for n, st in [("512", "o-"), ("1024", "s--")]:
        rows = [r for r in d[n] if (r["label"].startswith("folded") and "lazy" not in r["label"])
                or r["label"] == "hybrid"]
        rows = sorted(rows, key=lambda r: r["stages"])
        ax.plot([r["stages"] for r in rows], [r["alpha"] for r in rows], st,
                color="k", markerfacecolor="w", label=r"$n=%s$" % n)
        ffo = [r for r in d[n] if r["label"] == "FFO"][0]
        ax.plot(ffo["stages"], ffo["alpha"], "k*", ms=9)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("sequential sampling stages")
    ax.set_ylabel(r"quality $\alpha_\ell$")
    ax.grid(True, which="both", ls=":", lw=0.4)
    ax.legend(fontsize=7)
    finish(fig, "stages")


def fig_sizes():
    rows = load("sizes")
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))
    for ax, n in zip(axes, (512, 1024)):
        sel = [r for r in rows if r["n"] == n]
        names = [r["name"] for r in sel]
        x = np.arange(len(names))
        ax.bar(x - 0.25, [r["entropy"] for r in sel], 0.25, color="0.9",
               edgecolor="k", label="entropy bound")
        ax.bar(x, [r["rangecoder"] for r in sel], 0.25, color="0.55",
               edgecolor="k", label="arithmetic coder")
        ax.bar(x + 0.25, [r["golomb"] for r in sel], 0.25, color="0.2",
               edgecolor="k", label="Golomb\u2013Rice")
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("-%d" % n, "") for m in names],
                           rotation=20, ha="right", fontsize=7)
        lo = min(r["entropy"] for r in sel)
        hi = max(r["golomb"] for r in sel)
        ax.set_ylim(lo - 0.12 * (hi - lo), hi + 0.16 * (hi - lo))
        ax.set_ylabel("bytes")
        ax.set_title(r"$n=%d$" % n, fontsize=9)
        ax.grid(True, axis="y", ls=":", lw=0.4)
        ax.legend(fontsize=6.5, loc="upper left", ncol=1)
    finish(fig, "sizes")


if __name__ == "__main__":
    fig_alpha_ell()
    fig_gsprofile()
    fig_bench()
    fig_pareto()
    fig_spread()
    fig_stages()
    fig_sizes()
