"""Out-of-sample intraday volatility forecasts and their honest comparison.

Forecasters (all causal: the forecast made at bar t uses data through t):
    atr      scaled ATR(14)^2 (Wilder smoothing), scale re-fitted per quarter on the trailing year
    ewma     RiskMetrics EWMA of r^2, lambda 0.94
    day      mean r^2 over the trailing `bars_per_day` bars
    diurnal  mean r^2 at the same time-of-day slot over the previous 20 days
    garch / gjr   GARCH(1,1) / GJR-GARCH(1,1,1), zero mean, normal; re-fitted per quarter on the trailing year
                  (needs the optional `arch` package)

Loss: QLIKE = y/f - log(y/f) - 1 (robust to noise in the realised-variance proxy; lower is better).
Uncertainty: day-block bootstrap of the mean QLIKE difference.
"""
import numpy as np
import pandas as pd


def returns_bp(bars, max_gap="60min"):
    """Log returns in basis points from OHLC bars; a return across a gap > max_gap is set to 0 and flagged."""
    c = bars.close.to_numpy(float)
    gap = np.r_[True, np.diff(bars.index.asi8) > pd.Timedelta(max_gap).value]
    r = np.r_[0.0, np.diff(np.log(c)) * 1e4]
    r[gap] = 0.0
    return r, gap


def atr_variance(bars, n=14):
    h, l, c = (bars[k].to_numpy(float) for k in ("high", "low", "close"))
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    atr = pd.Series(tr).ewm(alpha=1 / n, adjust=False).mean().to_numpy()
    return (atr / c * 1e4) ** 2


def ewma(r2, lam=0.94):
    out = np.empty(len(r2))
    s = r2[: min(len(r2), 2000)].mean()
    for t in range(len(r2)):
        s = lam * s + (1 - lam) * r2[t]
        out[t] = s
    return out


def _garch_recursion(r, om, al, ga, be, s0):
    out = np.empty(len(r))
    s = s0
    for t in range(len(r)):
        s = om + (al + (ga if r[t] < 0 else 0.0)) * r[t] * r[t] + be * s
        out[t] = s
    return out


def forecasts(bars, models=("atr", "ewma", "day", "diurnal", "garch", "gjr"), bars_per_day=288, train_days=365):
    """One-step-ahead variance forecasts (bp^2) for every bar, plus the target y = r^2 of the next bar.
    Returns (DataFrame of forecasts, target array, valid mask, fitted GARCH persistence per quarter)."""
    r, gap = returns_bp(bars)
    r2 = r ** 2
    n = len(r)
    t = bars.index
    F = {}
    if "ewma" in models:
        F["ewma"] = ewma(r2)
    if "day" in models:
        F["day"] = pd.Series(r2).rolling(bars_per_day, min_periods=int(bars_per_day * .7)).mean().to_numpy()
    if "diurnal" in models:
        slot = (t - t.normalize()).total_seconds().astype(int)
        d = pd.DataFrame({"r2": r2, "slot": slot}).groupby("slot").r2.transform(
            lambda x: x.rolling(20, min_periods=10).mean().shift(1)).to_numpy()
        F["diurnal"] = np.r_[d[1:], np.nan]
    xatr = atr_variance(bars)
    want_g = [m for m in ("garch", "gjr") if m in models]
    for m in ["atr"] * ("atr" in models) + want_g:
        F[m] = np.full(n, np.nan)
    y = np.r_[r2[1:], np.nan]
    valid = np.r_[~gap[1:], False]
    persistence = []
    for q in pd.period_range(t[0], t[-1], freq="Q"):
        a = np.searchsorted(t, q.start_time - pd.Timedelta(days=train_days))
        b = np.searchsorted(t, q.start_time)
        e = np.searchsorted(t, (q + 1).start_time)
        if b - a < 50 * bars_per_day or e <= b:
            continue
        tr = slice(a, b)
        if "atr" in models:
            ok = valid[tr] & np.isfinite(xatr[tr]) & (xatr[tr] > 0)
            F["atr"][b:e] = np.mean(y[tr][ok] / xatr[tr][ok]) * xatr[b:e]
        if want_g:
            from arch import arch_model

            for m in want_g:
                p = arch_model(r[tr], mean="Zero", vol="GARCH", p=1, o=int(m == "gjr"), q=1, dist="normal",
                               rescale=False).fit(disp="off", show_warning=False).params
                om, al, be = float(p["omega"]), float(p["alpha[1]"]), float(p["beta[1]"])
                ga = float(p.get("gamma[1]", 0.0))
                F[m][b:e] = _garch_recursion(r[a:e], om, al, ga, be, float(r[tr].var()))[b - a:]
                persistence.append(dict(quarter=str(q), model=m, persistence=al + be + ga / 2))
    return pd.DataFrame(F, index=t), y, valid, pd.DataFrame(persistence)


def qlike(y, f):
    return y / f - np.log(y / f) - 1


def compare(F, y, valid, mask=None, n_boot=1000, block_days=10, seed=1):
    """QLIKE per model on a common sample, Mincer-Zarnowitz slope, and a bootstrap CI of each model vs the best one."""
    ok = valid & np.isfinite(y) & np.all(np.isfinite(F.to_numpy()) & (F.to_numpy() > 0), axis=1)
    if mask is not None:
        ok &= mask
    yy = y[ok]
    yy = np.maximum(yy, 0.01 * np.median(yy[yy > 0]))
    days = F.index[ok].normalize().asi8
    u, inv = np.unique(days, return_inverse=True)
    rng = np.random.default_rng(seed)
    st = rng.integers(0, len(u), (n_boot, -(-len(u) // block_days)))
    idx = ((st[:, :, None] + np.arange(block_days)) % len(u)).reshape(n_boot, -1)[:, :len(u)]
    W = np.stack([np.bincount(b, minlength=len(u)) for b in idx]).astype(float)
    nb = W @ np.bincount(inv, minlength=len(u))
    Q = {k: qlike(yy, F[k].to_numpy()[ok]) for k in F}
    best = min(Q, key=lambda k: Q[k].mean())
    rows = []
    for k, q in Q.items():
        f = F[k].to_numpy()[ok]
        slope = np.linalg.lstsq(np.c_[np.ones(len(f)), f], y[ok], rcond=None)[0][1]
        d = q - Q[best]
        db = (W @ np.bincount(inv, weights=d, minlength=len(u))) / nb
        rows.append(dict(model=k, qlike=float(q.mean()), mz_slope=float(slope),
                         vs_best=float(d.mean()), ci_low=float(np.quantile(db, .025)), ci_high=float(np.quantile(db, .975))))
    return pd.DataFrame(rows).sort_values("qlike").reset_index(drop=True), int(ok.sum())
