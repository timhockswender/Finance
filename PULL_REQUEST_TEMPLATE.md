# Summary

This PR adds a rolling-window Volume-Weighted Average Price (VWAP) feature that uses the close price and volume. It includes the implementation, a registry entry so the feature can be discovered by existing feature-loading code, and unit tests that cover normal and zero-volume edge cases.

# What I changed
- Added features/technical_indicators.py implementing vwap(df, window, price_col="close", vol_col="volume", min_periods=None)
- Added features/registry.py with FEATURES['vwap_rolling'] (merge if you already have a registry)
- Added tests/test_vwap.py with two unit tests

# How to test locally
1) pip install pandas numpy pytest
2) pytest -q tests/test_vwap.py

# Notes
- min_periods defaults to window; denominator==0 -> NaN
- If you already have an existing registry file, merge the single entry instead of overwriting.