import numpy as np
import pandas as pd
from typing import Tuple, Optional

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    # Wilder smoothing (EMA with alpha=1/period)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)  # neutral for early values
    return rsi

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # Wilder smoothing
    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    return atr

def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    # Directional Movement
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    tr = pd.concat([
        (high - low).abs(),
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    # Wilder smoothing
    tr_smooth = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    plus_dm_smooth = plus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm_smooth / tr_smooth)
    minus_di = 100 * (minus_dm_smooth / tr_smooth)
    dx = 100 * ( (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) )
    adx_series = dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    adx_series = adx_series.fillna(0)
    return adx_series

def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    tp = (high + low + close) / 3.0
    ma = tp.rolling(window=period, min_periods=1).mean()
    md = tp.rolling(window=period, min_periods=1).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci_series = (tp - ma) / (0.015 * md.replace(0, np.nan))
    return cci_series.fillna(0)

def wave_trend(close: pd.Series, high: pd.Series, low: pd.Series, channel_length: int = 10, average_length: int = 21, signal_sma: int = 4) -> Tuple[pd.Series, pd.Series]:
    # Typical WaveTrend implementation
    tp = (high + low + close) / 3.0
    esa = tp.ewm(span=channel_length, adjust=False).mean()
    de = (tp - esa).abs().ewm(span=channel_length, adjust=False).mean()
    ci = (tp - esa) / (0.015 * de.replace(0, np.nan))
    tci = ci.ewm(span=average_length, adjust=False).mean()
    wt1 = tci
    wt2 = wt1.rolling(window=signal_sma, min_periods=1).mean()
    return wt1.fillna(0), wt2.fillna(0)

# --- New indicators: MACD, OBV, MFI, VWAP ---

def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Returns (macd_line, signal_line, histogram)
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line.fillna(0), signal_line.fillna(0), hist.fillna(0)

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    On-Balance Volume: cumulative signed volume using close price direction.
    """
    direction = np.sign(close.diff().fillna(0))
    signed_vol = direction * volume.fillna(0)
    return signed_vol.cumsum()

def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
    """
    Money Flow Index: returns series between 0..100 (50 neutral)
    """
    tp = (high + low + close) / 3.0
    money_flow = tp * volume.fillna(0)
    # positive/negative money flow
    pos_flow = money_flow.where(tp > tp.shift(1), 0.0)
    neg_flow = money_flow.where(tp < tp.shift(1), 0.0)
    pos_mf = pos_flow.rolling(window=period, min_periods=1).sum()
    neg_mf = neg_flow.rolling(window=period, min_periods=1).sum()
    mfr = pos_mf / neg_mf.replace(0, np.nan)
    mfi_series = 100 - (100 / (1 + mfr))
    return mfi_series.fillna(50)

def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: Optional[int] = None) -> pd.Series:
    """
    VWAP: if window is None returns cumulative VWAP, otherwise rolling VWAP with given window.
    """
    tp = (high + low + close) / 3.0
    pv = tp * volume.fillna(0)
    if window is None:
        denom = volume.fillna(0).cumsum().replace(0, np.nan)
        return (pv.cumsum() / denom).fillna(method='ffill').fillna(tp)
    else:
        denom = volume.rolling(window=window, min_periods=1).sum().replace(0, np.nan)
        return (pv.rolling(window=window, min_periods=1).sum() / denom).fillna(method='ffill').fillna(tp)