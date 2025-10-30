"""
Build a composite feature index. Updated to optionally include volume indicators (OBV, MFI) and use a normalization strategy.
"""
import numpy as np
import pandas as pd
from .indicators import rsi, wave_trend, cci, adx, obv, mfi

def _rolling_zscore(series: pd.Series, window: int = 252, clip: float = 3.0) -> pd.Series:
    """Compute rolling z-score and clip to [-clip, clip], then scale to [-1,1].

    This helps normalize indicators with non-stationary ranges (e.g., OBV, CCI).
    """
    roll_mean = series.rolling(window=window, min_periods=1).mean()
    roll_std = series.rolling(window=window, min_periods=1).std().replace(0, np.nan)
    z = (series - roll_mean) / roll_std
    z = z.fillna(0).clip(-clip, clip) / clip
    return z

def _minmax_scale(series: pd.Series, lower=-1.0, upper=1.0, clip=True) -> pd.Series:
    s = series.copy().astype(float)
    mn = s.min()
    mx = s.max()
    if pd.isna(mn) or pd.isna(mx) or mn == mx:
        scaled = pd.Series(0.0, index=s.index)
    else:
        scaled = (s - mn) / (mx - mn)
        scaled = scaled * (upper - lower) + lower
    if clip:
        scaled = scaled.clip(lower, upper)
    return scaled

def _signed_rsi_signal(rsi_series: pd.Series) -> pd.Series:
    return (rsi_series - 50) / 50.0

def build_feature_index(df: pd.DataFrame,
                        window: int = 252,
                        weights: dict = None,
                        normalize: str = 'rolling_zscore') -> pd.Series:
    """
    Build a composite feature index from indicators.

    Parameters:
    - df: DataFrame with columns: ['open','high','low','close','volume'] (volume optional)
    - window: rolling window used for normalization
    - weights: dict with keys: ['rsi','rsi9','wt','wt_signal','cci','adx','obv','mfi']
    - normalize: one of ['rolling_zscore','minmax']

    Returns:
    - pd.Series feature_index
    """
    if weights is None:
        weights = {"rsi": 1.0, "rsi9": 1.0, "wt": 1.0, "wt_signal": 0.5, "cci": 0.7, "adx": 0.5, "obv": 0.4, "mfi": 0.6}

    close = df['close']
    high = df['high']
    low = df['low']

    rsi14 = rsi(close, period=14)
    rsi9 = rsi(close, period=9)
    wt1, wt2 = wave_trend(close=close, high=high, low=low)
    cci_series = cci(high=high, low=low, close=close, period=20)
    adx_series = adx(high=high, low=low, close=close, period=14)

    # Volume indicators (if available)
    obv_series = None
    mfi_series = None
    if 'volume' in df.columns:
        try:
            obv_series = obv(close, df['volume'])
            mfi_series = mfi(high, low, close, df['volume'])
        except Exception:
            obv_series = None
            mfi_series = None

    # Create raw signals
    sig_rsi = _signed_rsi_signal(rsi14).fillna(0)
    sig_rsi9 = _signed_rsi_signal(rsi9).fillna(0)

    wt_hist = (wt1 - wt2).fillna(0)

    # choose normalization
    if normalize == 'minmax':
        sig_wt = _minmax_scale(wt_hist)
        sig_cci = _minmax_scale(cci_series)
        # ADX is strength only - map 0..100 to 0..1 then sign with rsi9+wt
        direction = np.sign(sig_rsi9 + sig_wt)
        sig_adx = _minmax_scale(adx_series * direction / 100.0)
        sig_obv = _minmax_scale(obv_series) if obv_series is not None else pd.Series(0.0, index=df.index)
        sig_mfi = _minmax_scale((mfi_series - 50) / 50.0) if mfi_series is not None else pd.Series(0.0, index=df.index)
    else:  # rolling_zscore
        sig_wt = _rolling_zscore(wt_hist, window=window)
        sig_cci = _rolling_zscore(cci_series, window=window)
        direction = np.sign(sig_rsi9 + sig_wt)
        sig_adx = _rolling_zscore(adx_series * direction / 100.0, window=window)
        sig_obv = _rolling_zscore(obv_series.fillna(0), window=window) if obv_series is not None else pd.Series(0.0, index=df.index)
        # MFI is 0..100; convert to signed scale around 50 then zscore
        sig_mfi = _rolling_zscore(((mfi_series.fillna(50) - 50) / 50.0).fillna(0), window=window) if mfi_series is not None else pd.Series(0.0, index=df.index)

    # Build components dataframe
    components = pd.DataFrame(index=df.index)
    components['rsi'] = sig_rsi * weights.get('rsi', 1.0)
    components['rsi9'] = sig_rsi9 * weights.get('rsi9', 1.0)
    components['wt'] = sig_wt * weights.get('wt', 1.0)
    components['wt_signal'] = ((wt2 - wt1).fillna(0)) * weights.get('wt_signal', 0.5)
    components['cci'] = sig_cci * weights.get('cci', 0.7)
    components['adx'] = sig_adx * weights.get('adx', 0.5)
    components['obv'] = sig_obv * weights.get('obv', 0.4)
    components['mfi'] = sig_mfi * weights.get('mfi', 0.6)

    raw_score = components.sum(axis=1)
    feature_index = np.tanh(raw_score)
    return pd.Series(feature_index, index=df.index, name='feature_index')
