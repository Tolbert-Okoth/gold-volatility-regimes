"""Reproduce the gold regime atlas (2009-2026) from the shipped monthly descriptors: breaks, states, figures.

    python examples/gold_atlas.py          (~1 minute; the break bootstrap dominates)
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from volregime import DESCRIPTORS, break_stability, fit_states, load_gold_atlas, segments, select  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs"
ERAS = [("2009-2012", 2009, 2012), ("2013-2016", 2013, 2016), ("2017-2020", 2017, 2020), ("2021-2024", 2021, 2024),
        ("2025-2026", 2025, 2026)]

M = load_gold_atlas()
E = M[M.eligible].reset_index(drop=True)
E["era"] = E.year.map(lambda y: next(n for n, a, b in ERAS if a <= y <= b))
t = pd.PeriodIndex(E.mon, freq="M").to_timestamp()
source_change = pd.Timestamp("2023-06-01")
summary = {}

# ---------------- structural breaks
series = {"ln vol": np.log(E.vol.to_numpy()), "ac1": E.ac1.to_numpy(), "ny_share": E.ny_share.to_numpy()}
fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
for ax, (name, y) in zip(axes, series.items()):
    m, r = select(y, max_breaks=4, min_size=18)
    br = r[m]["breaks"]
    stab, mdist = break_stability(y, br, n_boot=300, max_breaks=4, min_size=18)
    seg = segments(y, E.mon, br)
    summary[name] = dict(breaks=[E.mon[b] for b in br], stability=[round(s, 2) for s in stab],
                         selected_m_distribution=mdist, bic={k: round(v["bic"], 1) for k, v in r.items()},
                         segments=seg.round(4).to_dict("records"))
    ax.plot(t, y, lw=1, color="#4C72B0")
    for _, s in seg.iterrows():
        a, b = pd.Period(s["first"], "M").to_timestamp(), pd.Period(s["last"], "M").to_timestamp()
        ax.hlines(s["mean"], a, b, color="#C44E52", lw=2.5)
    for b, st in zip(br, stab):
        ax.axvline(t[b], color="#C44E52", lw=0.8, ls=":")
        ax.text(t[b], ax.get_ylim()[1], f" {E.mon[b]}\n stability {st:.2f}", fontsize=7, va="top", color="#C44E52")
    ax.axvline(source_change, color="k", lw=0.8, ls="--")
    ax.set_ylabel(name)
    ax.grid(alpha=.25)
axes[0].set_title("XAUUSD monthly descriptors with BIC change points (red) and data-source change (dashed)")
fig.tight_layout()
fig.savefig(OUT / "breaks.png", dpi=140)

# ---------------- recurrent states
S = fit_states(E[DESCRIPTORS].to_numpy(), E.era.to_numpy(), k=3, order_by=0)
E["state"] = S.labels
summary["states"] = dict(silhouette=round(S.silhouette, 3), verdict=S.verdict,
                         holdout=S.holdout.round(3).to_dict("records"),
                         months_by_era=pd.crosstab(E.state, E.era).to_dict(),
                         profiles=E.groupby("state")[DESCRIPTORS].median().round(4).to_dict("index"))
fig, ax = plt.subplots(figsize=(11, 2.6))
ax.scatter(t, S.labels, c=S.labels, cmap="viridis", marker="s", s=22)
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(["S0", "S1", "S2 (high vol)"])
ax.axvline(source_change, color="k", lw=0.8, ls="--")
ax.set_title(f"K-means monthly states: silhouette {S.silhouette:.2f} -> taxonomy {S.verdict['taxonomy']}")
fig.tight_layout()
fig.savefig(OUT / "states.png", dpi=140)

# ---------------- percent vs dollar volatility (price-level effect)
y = M[M.eligible].groupby("year").vol.mean()
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.bar(y.index, y.values, color="#4C72B0")
ax.set_ylabel("volatility, % per day")
ax.set_title("Gold intraday volatility in % terms by year (2025-26 is high, but not unprecedented in %)")
ax.grid(axis="y", alpha=.25)
fig.tight_layout()
fig.savefig(OUT / "vol_by_year.png", dpi=140)

json.dump(summary, open(OUT / "atlas_results.json", "w"), indent=1, default=str)
for k in series:
    print(f"{k:9s} breaks {summary[k]['breaks']} stability {summary[k]['stability']}")
print("states: silhouette", summary["states"]["silhouette"], summary["states"]["verdict"])
print(S.holdout.round(2).to_string(index=False))
