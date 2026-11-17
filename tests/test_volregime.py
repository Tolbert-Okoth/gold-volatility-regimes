import importlib.util
import unittest

import numpy as np
import pandas as pd

from volregime import (
    DESCRIPTORS, compare, fit_states, forecasts, load_gold_atlas, m5_returns, qlike, segment_dp, select,
)

HAS_ARCH = importlib.util.find_spec("arch") is not None


def synthetic_bars(days=160, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.bdate_range("2024-01-01", periods=days)
    idx = (d.values[:, None] + (np.arange(288) * 5).astype("timedelta64[m]")).ravel()
    c = 2000 * np.exp(np.cumsum(rng.normal(0, 3, len(idx))) / 1e4)
    o = np.r_[2000, c[:-1]]
    return pd.DataFrame(dict(open=o, high=np.maximum(o, c) + .2, low=np.minimum(o, c) - .2, close=c), index=pd.DatetimeIndex(idx))


class BreaksTest(unittest.TestCase):
    def test_planted_break_found_and_dp_exact(self):
        rng = np.random.default_rng(0)
        y = np.r_[rng.normal(0, 1, 60), rng.normal(3, 1, 90)]
        m, r = select(y)
        self.assertEqual(m, 1)
        self.assertLessEqual(abs(r[1]["breaks"][0] - 60), 2)
        brute = min(range(18, 133), key=lambda b: ((y[:b] - y[:b].mean()) ** 2).sum() + ((y[b:] - y[b:].mean()) ** 2).sum())
        self.assertEqual(r[1]["breaks"][0], brute)

    def test_min_segment_respected(self):
        y = np.random.default_rng(1).normal(size=150)
        br = segment_dp(y, max_breaks=4, min_size=18)[4]["breaks"]
        self.assertTrue(all(np.diff([0] + br + [150]) >= 18))


class DescriptorTest(unittest.TestCase):
    def test_no_return_across_gap_or_session(self):
        idx = pd.date_range("2020-01-06 00:00", "2020-01-07 23:59", freq="1min")
        c = pd.Series(1500 + np.arange(len(idx)) * .01, index=idx)
        c = c.drop(c.index[(c.index >= "2020-01-06 10:00") & (c.index < "2020-01-06 10:30")])
        R = m5_returns(c)
        self.assertTrue(np.isnan(R.r.loc["2020-01-06 10:05"]))
        self.assertTrue(np.isfinite(R.r.loc["2020-01-06 10:40"]))
        self.assertAlmostEqual(R.r.loc["2020-01-06 09:00"], np.log(c.loc["2020-01-06 08:59"] / c.loc["2020-01-06 08:54"]))
        # the interval 16:55-17:00 closes session 2020-01-06; 17:00-17:05 opens session 2020-01-07
        self.assertEqual(R.sess.loc["2020-01-06 17:00"], pd.Timestamp("2020-01-06"))
        self.assertEqual(R.sess.loc["2020-01-06 17:05"], pd.Timestamp("2020-01-07"))
        # er12 windows never span the session boundary
        self.assertTrue(np.isnan(R.er.loc["2020-01-06 17:30"]))
        self.assertTrue(np.isfinite(R.er.loc["2020-01-06 18:00"]))


class StatesTest(unittest.TestCase):
    def test_clear_clusters_supported_overlap_not(self):
        rng = np.random.default_rng(3)
        groups = np.repeat(["a", "b", "c", "d"], 30)
        centers = np.array([[0, 0], [6, 0], [0, 6]])
        X = centers[np.tile(np.arange(3), 40)] + rng.normal(0, .5, (120, 2))
        self.assertEqual(fit_states(X, groups).verdict["taxonomy"], "SUPPORTED")
        self.assertEqual(fit_states(rng.normal(size=(120, 2)), groups).verdict["taxonomy"], "NOT_SUPPORTED")

    def test_shipped_gold_atlas_verdict(self):
        E = load_gold_atlas().query("eligible")
        self.assertEqual(len(E), 206)
        eras = pd.cut(E.year, [2008, 2012, 2016, 2020, 2024, 2026]).astype(str).to_numpy()
        r = fit_states(E[DESCRIPTORS].to_numpy(), eras)
        self.assertAlmostEqual(r.silhouette, 0.20, delta=0.01)
        self.assertEqual(r.verdict["taxonomy"], "NOT_SUPPORTED")


class ForecastTest(unittest.TestCase):
    def test_qlike_minimum_at_truth(self):
        self.assertEqual(qlike(np.array([2.0]), np.array([2.0]))[0], 0.0)
        self.assertGreater(qlike(np.array([2.0]), np.array([1.0]))[0], 0)

    def test_no_lookahead(self):
        b = synthetic_bars()
        cut = b.index[len(b) * 3 // 4]
        F1, *_ = forecasts(b, models=("atr", "ewma", "day", "diurnal"))
        b2 = b.copy()
        b2.loc[b2.index > cut, ["open", "high", "low", "close"]] *= 1.5     # rewrite the future
        F2, *_ = forecasts(b2, models=("atr", "ewma", "day", "diurnal"))
        past = F1.index < cut
        pd.testing.assert_frame_equal(F1[past], F2[past])

    def test_compare_runs(self):
        F, y, valid, _ = forecasts(synthetic_bars(), models=("atr", "ewma", "day"))
        t, n = compare(F, y, valid, n_boot=200)
        self.assertGreater(n, 1000)
        self.assertEqual(t.vs_best.iloc[0], 0.0)

    @unittest.skipUnless(HAS_ARCH, "arch not installed")
    def test_garch_runs(self):
        F, y, valid, P = forecasts(synthetic_bars(days=400), models=("garch",))
        self.assertTrue(np.isfinite(F.garch).any())
        self.assertTrue((P.persistence > 0).all())


if __name__ == "__main__":
    unittest.main()
