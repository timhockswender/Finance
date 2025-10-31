import numpy as np
import pandas as pd
from typing import Optional


def vwap(
    df: pd.DataFrame,
    window: int,
    price_col: str = "close",
    vol_col: str = "volume",
    min_periods: Optional[int] = None,
) -> pd.Series:
    """
    Rolling-window VWAP using close price and volume.

    Parameters
    - df: DataFrame containing price and volume columns.
    - window: lookback in bars (int > 0)
    - price_col: name of price column to use (default "close")
    - vol_col: name of volume column to use (default "volume")
    - min_periods: min periods for rolling sums. If None, defaults to window.

    Returns
    - pd.Series named 'vwap' aligned to df.index (NaN where insufficient data or zero-volume).
    """
    if not isinstance(window, int) or window <= 0:
        raise ValueError("window must be a positive integer")

    if price_col not in df:
        raise KeyError(f"price_col '{price_col}' not found in DataFrame")
    if vol_col not in df:
        raise KeyError(f"vol_col '{vol_col}' not found in DataFrame")

    price = df[price_col].astype(float)
    vol = df[vol_col].astype(float)

    pv = price * vol

    if min_periods is None:
        min_periods = window

    num = pv.rolling(window=window, min_periods=min_periods).sum()
    den = vol.rolling(window=window, min_periods=min_periods).sum()

    # safe division: where den == 0 or NaN -> result is NaN
    with np.errstate(divide="ignore", invalid="ignore"):
        v = num / den

    v = v.where(den != 0, np.nan)
    v.name = "vwap"
    return v
