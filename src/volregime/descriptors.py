"""Intraday market descriptors from 1-minute closes: per-session realised variance and five monthly descriptors.

Input: a 1-minute close series indexed by bar OPEN time in timezone-naive New York time.
Sessions run 17:00 -> 17:00 New York (the metals/FX trading day) and are labelled by the date they end on.

Monthly descriptors (all computed on valid sessions only):
    vol       sqrt(median daily RV of M5 log returns), % per day
    ac1       lag-1 autocorrelation of within-session M5 returns (pairs never cross a gap or a session)
    er12      median efficiency ratio |12-bar net move| / sum |12 M5 returns|, windows inside one session
    ny_share  median daily share of RV falling in New York 08:00-17:00
    tail      p99 / median of |M5 return|
"""
import numpy as np
import pandas as pd

FIVE = pd.Timedelta(minutes=5)
DESCRIPTORS = ["vol", "ac1", "er12", "ny_share", "tail"]


def m5_returns(close, session_start_hour=17, ny_window=(8, 17)):
    """M5 log returns on a regular 5-minute grid. A boundary price at T is the close of the M1 bar that opened at T-1min.
    A return is NaN unless both endpoints exist and lie in the same session (no return spans a gap or a session break)."""
    c = close[~close.index.duplicated()].sort_index()
    p = pd.Series(c.to_numpy(float), index=c.index + pd.Timedelta(minutes=1))
    p = p[(p.index.minute % 5 == 0) & (p > 0)]
    g = pd.date_range(p.index[0].floor("5min"), p.index[-1], freq="5min")
    p = p.reindex(g)
    r = np.log(p).diff()
    shift = pd.Timedelta(hours=24 - session_start_hour)
    start = g - FIVE
    sess = (start + shift).normalize()
    same = sess == (g - pd.Timedelta(microseconds=1) + shift).normalize()
    r = r.where(same)
    hour = start.hour + start.minute / 60
    ab = r.abs().rolling(12, min_periods=12).sum()
    net = np.log(p) - np.log(p.shift(12))
    sess12 = pd.Series(sess, index=g).shift(11)
    er = (net.abs() / ab).where((ab > 0) & (sess12.to_numpy() == sess))
    return pd.DataFrame(dict(r=r.to_numpy(), sess=sess, ny=(hour >= ny_window[0]) & (hour < ny_window[1]),
                             er=er.to_numpy()), index=g)


def session_table(R, min_returns=220, min_ny_returns=90):
    """Per-session counts and realised variance. A session is valid with >= min_returns M5 returns (of 288)."""
    v = R[R.r.notna()]
    g = v.groupby("sess")
    r2 = v.r ** 2
    S = pd.DataFrame(dict(n=g.r.size(), n_ny=g.ny.sum(), rv=r2.groupby(v.sess).sum(),
                          rv_ny=r2[v.ny].groupby(v.sess[v.ny]).sum()))
    S["rv_ny"] = S.rv_ny.fillna(0.0)
    S["valid"] = (S.n >= min_returns) & (S.n_ny >= min_ny_returns) & (S.rv > 0)
    return S


def monthly(R, S, min_sessions=15):
    """Monthly descriptor table. `eligible` = enough valid sessions and every descriptor defined."""
    vs = S[S.valid].index
    v = R[R.r.notna() & R.sess.isin(vs)].copy()
    v["mon"] = v.sess.dt.to_period("M")
    v["r_prev"] = v.r.shift(1)
    v.loc[(v.sess != v.sess.shift(1)) | (v.index.to_series().diff() != FIVE).to_numpy(), "r_prev"] = np.nan
    ers = R[R.er.notna() & R.sess.isin(vs)]
    er_m = ers.er.groupby(ers.sess.dt.to_period("M")).median()
    rows = []
    for mon, x in v.groupby("mon"):
        ss = S.loc[S.valid & (S.index.to_period("M") == mon)]
        pr = x.dropna(subset=["r_prev"])
        a = x.r.abs()
        med = float(a.median())
        rows.append(dict(mon=str(mon), sessions=len(ss), vol=float(np.sqrt(ss.rv.median())) * 100,
                         ac1=float(np.corrcoef(pr.r_prev, pr.r)[0, 1]) if len(pr) > 50 else np.nan,
                         er12=float(er_m.get(mon, np.nan)), ny_share=float((ss.rv_ny / ss.rv).median()),
                         tail=float(a.quantile(0.99) / med) if med > 0 else np.nan, n_ret=len(x)))
    M = pd.DataFrame(rows)
    M["year"] = M.mon.str[:4].astype(int)
    M["eligible"] = (M.sessions >= min_sessions) & M[DESCRIPTORS].notna().all(axis=1)
    return M


def monthly_descriptors(close, **kw):
    """One call: 1-minute closes (New York time) -> monthly descriptor table."""
    R = m5_returns(close, **{k: v for k, v in kw.items() if k in ("session_start_hour", "ny_window")})
    return monthly(R, session_table(R))
