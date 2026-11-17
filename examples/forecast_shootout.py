"""Why does plain GARCH lose to a scaled ATR on intraday data? A synthetic demonstration of the mechanism.

The simulated 5-minute market has (1) a strong time-of-day volatility cycle (quiet Asia, busy London/New York) and
(2) slowly drifting volatility levels with occasional shocks, the two features of real intraday gold.
Plain GARCH on raw returns has no notion of time-of-day: it reads the daily cycle as persistence and ends up
(nearly) integrated, alpha + beta ~ 1.

    python examples/forecast_shootout.py      (~1-2 minutes)
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from volregime import compare, forecasts  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs"
rng = np.random.default_rng(7)

days = pd.bdate_range("2023-01-02", periods=2 * 260)
bars = 288
slot = np.arange(bars) * 5 / 60                         # hour of day
diurnal = 0.5 + 0.9 * np.exp(-((slot - 8.5) / 1.6) ** 2) + 1.3 * np.exp(-((slot - 14.0) / 1.8) ** 2)
level = np.exp(np.cumsum(rng.normal(0, 0.06, len(days))))            # slow day-to-day drift in volatility
level *= np.where(rng.random(len(days)) < 0.03, 2.5, 1.0)            # occasional shock days
sub = 5                                                              # 1-minute sub-steps for highs and lows
sig = (level[:, None] * diurnal[None, :]).ravel() * 3.0 / np.sqrt(sub)          # bp per sub-step
steps = rng.standard_t(5, (len(sig), sub)) * sig[:, None] / np.sqrt(5 / 3)
path = 2000 * np.exp(np.cumsum(steps.ravel()) / 1e4).reshape(len(sig), sub)
prev = np.r_[2000.0, path[:-1, -1]]
idx = (days.values[:, None] + (np.arange(bars) * 5).astype("timedelta64[m]")).ravel()
B = pd.DataFrame(dict(open=prev, high=np.maximum(prev, path.max(1)), low=np.minimum(prev, path.min(1)),
                      close=path[:, -1]), index=pd.DatetimeIndex(idx))

F, y, valid, P = forecasts(B, bars_per_day=bars)
eval_mask = B.index >= pd.Timestamp("2024-01-01")
table, n = compare(F, y, valid, mask=eval_mask)
print(f"evaluation bars: {n:,}")
print(table.round(4).to_string(index=False))
print("median GARCH persistence:", P.groupby("model").persistence.median().round(4).to_dict())
table.round(4).to_csv(OUT / "shootout_synthetic.csv", index=False)

fig, ax = plt.subplots(figsize=(7.5, 3.8))
order = table.model.tolist()
ax.barh(order, table.qlike, color=["#55A868" if m in ("atr", "diurnal") else "#C44E52" if m in ("garch", "gjr") else "#8C8C8C" for m in order])
ax.invert_yaxis()
ax.set_xlabel("QLIKE loss, one step ahead (lower is better)")
ax.set_title("Synthetic intraday market with a daily volatility cycle")
for i, (q, lo) in enumerate(zip(table.qlike, table.ci_low)):
    ax.text(q, i, f" {q:.3f}", va="center", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "shootout_synthetic.png", dpi=140)
