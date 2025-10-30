import os
from typing import Optional

import pandas as pd

def download_ticker_csv(symbol: str,
                        start: Optional[str] = None,
                        end: Optional[str] = None,
                        interval: str = "1d",
                        source: str = "yfinance",
                        to_path: Optional[str] = None) -> pd.DataFrame:
    """Download OHLCV data for a ticker and return a DataFrame and optionally save to CSV.

    Parameters
    - symbol: ticker symbol (e.g., 'AAPL')
    - start, end: date strings (YYYY-MM-DD) or None for defaults
    - interval: data interval (e.g., '1d', '1h')
    - source: 'yfinance' (default). Other sources not implemented.
    - to_path: if provided, save the CSV to this path and return the path in addition to df.

    Returns
    - pandas.DataFrame with index as DatetimeIndex and columns: open, high, low, close, volume
    """
    if source != "yfinance":
        raise ValueError("Only source='yfinance' is supported currently")

    try:
        import yfinance as yf
    except Exception as e:
        raise ImportError("yfinance is required for download_ticker_csv. Install with 'pip install yfinance'") from e

    ticker = yf.Ticker(symbol)
    hist = ticker.history(start=start, end=end, interval=interval, auto_adjust=False)
    if hist is None or hist.empty:
        raise RuntimeError(f"No data returned for {symbol} {start}..{end} interval={interval}")

    # Ensure columns exist and are named lower-case
    hist = hist.rename(columns={c: c.lower() for c in hist.columns})
    # Required columns: Open, High, Low, Close, Volume
    required = ['open', 'high', 'low', 'close', 'volume']
    for col in required:
        if col not in hist.columns:
            # try common alternatives
            alt = col.capitalize()
            if alt in hist.columns:
                hist[col] = hist[alt]
            else:
                hist[col] = pd.NA

    df = hist[['open', 'high', 'low', 'close', 'volume']].copy()
    df.index.name = 'date'

    if to_path:
        os.makedirs(os.path.dirname(to_path) or '.', exist_ok=True)
        out_df = df.reset_index()
        out_df.to_csv(to_path, index=False)
    return df
