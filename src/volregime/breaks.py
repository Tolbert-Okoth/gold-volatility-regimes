"""Exploratory structural breaks: exact least-squares segmentation of a mean (dynamic programming), BIC selection,
and a residual moving-block bootstrap for break-date stability.

This is the Bai-Perron segmentation principle (global SSE minimisation with a minimum segment length) with BIC
selection. It is NOT Bai-Perron sup-F inference: the dates are exploratory change points without p-values. On
persistent series (e.g. volatility) a constant-mean model tends to over-segment; read the bootstrap stability.
"""
import numpy as np
import pandas as pd


def segment_dp(y, max_breaks=4, min_size=18):
    """Exact optimal segmentation for m = 0..max_breaks. Returns {m: dict(sse, breaks, bic)}; breaks are start indices."""
    y = np.asarray(y, float)
    n = len(y)
    cs = np.r_[0, np.cumsum(y)]
    cs2 = np.r_[0, np.cumsum(y * y)]

    def cost(i, j):
        return (cs2[j] - cs2[i]) - (cs[j] - cs[i]) ** 2 / (j - i)

    F = np.full((max_breaks + 1, n + 1), np.inf)
    B = np.zeros((max_breaks + 1, n + 1), int)
    for j in range(min_size, n + 1):
        F[0, j] = cost(0, j)
    for m in range(1, max_breaks + 1):
        for j in range((m + 1) * min_size, n + 1):
            best, arg = np.inf, -1
            for i in range(m * min_size, j - min_size + 1):
                v = F[m - 1, i] + cost(i, j)
                if v < best:
                    best, arg = v, i
            F[m, j], B[m, j] = best, arg
    out = {}
    for m in range(max_breaks + 1):
        if not np.isfinite(F[m, n]):
            continue
        br, j = [], n
        for k in range(m, 0, -1):
            j = B[k, j]
            br.append(j)
        sse = float(F[m, n])
        out[m] = dict(sse=sse, breaks=sorted(br), bic=float(n * np.log(sse / n) + (2 * m + 1) * np.log(n)))
    return out


def select(y, **kw):
    """BIC-best number of breaks and the full table."""
    r = segment_dp(y, **kw)
    m = min(r, key=lambda k: r[k]["bic"])
    return m, r


def break_stability(y, breaks, n_boot=300, block=12, window=6, seed=42, **kw):
    """Share of bootstrap draws with a break within +-window of each break, and the distribution of the selected m."""
    y = np.asarray(y, float)
    rng = np.random.default_rng(seed)
    n = len(y)
    seg = np.r_[0, breaks, n]
    fit = np.empty(n)
    for a, b in zip(seg[:-1], seg[1:]):
        fit[a:b] = y[a:b].mean()
    e = y - fit
    hits = np.zeros(len(breaks))
    mcount = {}
    for _ in range(n_boot):
        st = rng.integers(0, n - block + 1, int(np.ceil(n / block)))
        eb = np.concatenate([e[s:s + block] for s in st])[:n]
        m, r = select(fit + eb, **kw)
        mcount[m] = mcount.get(m, 0) + 1
        for k, b0 in enumerate(breaks):
            hits[k] += any(abs(b1 - b0) <= window for b1 in r[m]["breaks"])
    return (hits / n_boot).tolist(), {int(k): v / n_boot for k, v in sorted(mcount.items())}


def segments(y, labels, breaks):
    """Table of segments: first/last label, length and mean."""
    y = np.asarray(y, float)
    labels = list(labels)
    seg = np.r_[0, breaks, len(y)]
    return pd.DataFrame([dict(first=labels[a], last=labels[b - 1], n=b - a, mean=float(y[a:b].mean()))
                         for a, b in zip(seg[:-1], seg[1:])])
