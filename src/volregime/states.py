"""Recurrent states with a pre-registered verdict: K-means on standardised descriptors, leave-one-group-out
stability, separation (silhouette) and run/transition statistics.

The point is to ask "is there really a regime taxonomy here?" with rules fixed BEFORE looking, instead of
assuming one because a clustering algorithm always returns K clusters.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

RULES = dict(min_holdout_agreement=0.75, min_silhouette=0.25, min_group_months=12)


@dataclass
class StateResult:
    labels: np.ndarray
    centroids: np.ndarray
    silhouette: float
    holdout: pd.DataFrame
    transitions: pd.DataFrame
    runs: pd.DataFrame
    verdict: dict = field(default_factory=dict)


def _kmeans(X, k, seed):
    from sklearn.cluster import KMeans

    return KMeans(n_clusters=k, random_state=seed, n_init=20).fit(X)


def _align(c_ref, c_new):
    """Map new labels to reference labels by Hungarian matching of centroids."""
    from scipy.optimize import linear_sum_assignment

    D = ((c_ref[:, None, :] - c_new[None, :, :]) ** 2).sum(-1)
    r, c = linear_sum_assignment(D)
    mp = np.empty(len(c_ref), int)
    mp[c] = r
    return mp


def run_lengths(labels):
    out, cur, k = [], labels[0], 1
    for x in labels[1:]:
        if x == cur:
            k += 1
        else:
            out.append((cur, k))
            cur, k = x, 1
    out.append((cur, k))
    return out


def fit_states(X, groups, k=3, seed=42, order_by=0, rules=RULES):
    """X: (n, d) descriptor matrix in time order; groups: era label per row (for leave-one-group-out).
    States are numbered by the centroid of column `order_by` (e.g. volatility) for readability only."""
    from sklearn.metrics import adjusted_rand_score, silhouette_score

    X = np.asarray(X, float)
    groups = np.asarray(groups)
    mu, sd = X.mean(0), X.std(0)
    Z = (X - mu) / sd
    km = _kmeans(Z, k, seed)
    order = np.argsort(km.cluster_centers_[:, order_by])
    remap = np.empty(k, int)
    remap[order] = np.arange(k)
    lab = remap[km.labels_]
    cent = km.cluster_centers_[order]
    sil = float(silhouette_score(Z, lab))

    rows = []
    for g in pd.unique(groups):
        tr, te = groups != g, groups == g
        m2, s2 = X[tr].mean(0), X[tr].std(0)
        km2 = _kmeans((X[tr] - m2) / s2, k, seed)
        mp = _align(cent, (km2.cluster_centers_ * s2 + m2 - mu) / sd)
        held = mp[km2.predict((X[te] - m2) / s2)]
        train = mp[km2.labels_]
        rows.append(dict(group=g, months=int(te.sum()), holdout_agreement=float((held == lab[te]).mean()),
                         train_agreement=float((train == lab[tr]).mean()), train_ari=float(adjusted_rand_score(lab[tr], train))))
    H = pd.DataFrame(rows)
    T = pd.crosstab(pd.Series(lab[:-1], name="from"), pd.Series(lab[1:], name="to"))
    rl = pd.DataFrame(run_lengths(list(lab)), columns=["state", "run"])
    runs = rl.groupby("state").run.agg(["median", "max", "count"])
    big = H[H.months >= rules["min_group_months"]]
    stable = bool((big.holdout_agreement >= rules["min_holdout_agreement"]).all())
    separated = sil >= rules["min_silhouette"]
    verdict = dict(stable=stable, separated=separated,
                   taxonomy="SUPPORTED" if stable and separated else "NOT_SUPPORTED",
                   reason=("" if separated else f"clusters overlap (silhouette {sil:.2f} < {rules['min_silhouette']})"))
    return StateResult(lab, cent * sd + mu, sil, H, T, runs, verdict)


def block_bootstrap_diff(a, b, n_boot=2000, block=3, seed=42):
    """95 % month-block bootstrap CI of mean(b) - mean(a)."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)

    def rs(x):
        n = len(x)
        st = rng.integers(0, max(n - block + 1, 1), int(np.ceil(n / block)))
        return np.concatenate([x[s:s + block] for s in st])[:n].mean()

    d = np.array([rs(b) - rs(a) for _ in range(n_boot)])
    return float(b.mean() - a.mean()), (float(np.quantile(d, .025)), float(np.quantile(d, .975)))
