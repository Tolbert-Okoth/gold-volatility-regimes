"""Volatility regimes, honestly: descriptors, structural breaks, recurrent states and forecaster comparisons."""
from importlib.resources import files

import pandas as pd

from volregime.breaks import break_stability, segment_dp, segments, select
from volregime.descriptors import DESCRIPTORS, m5_returns, monthly, monthly_descriptors, session_table
from volregime.forecast import compare, forecasts, qlike
from volregime.states import block_bootstrap_diff, fit_states

__version__ = "0.1.0"


def load_gold_atlas():
    """Monthly descriptors for XAUUSD 2009-03 .. 2026-10 (derived statistics only; see README for data sources)."""
    return pd.read_csv(files("volregime") / "data" / "xauusd_monthly_descriptors.csv")


__all__ = ["DESCRIPTORS", "block_bootstrap_diff", "break_stability", "compare", "fit_states", "forecasts",
           "load_gold_atlas", "m5_returns", "monthly", "monthly_descriptors", "qlike", "segment_dp", "segments",
           "select", "session_table"]
