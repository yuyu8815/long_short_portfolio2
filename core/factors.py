# core/factors.py
import numpy as np
import pandas as pd

# ========================
# Cross-sectional functions
# ========================
def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise z-score normalization (safe version)."""
    mu = df.mean(axis=1)
    sd = df.std(axis=1, ddof=0).replace(0, np.nan)
    out = (df.sub(mu, axis=0)).div(sd, axis=0)
    return out.fillna(0)


def rank_cs(df: pd.DataFrame) -> pd.DataFrame:
    """Rank each row (cross-sectionally) into 0~1."""
    return df.rank(axis=1, method="average", pct=True)

def winsorize(df: pd.DataFrame, std: float = 4.0) -> pd.DataFrame:
    """Clip each row to mean ± std*σ."""
    mu = df.mean(axis=1)
    sigma = df.std(axis=1, ddof=0)
    lo = mu - std * sigma
    hi = mu + std * sigma
    return df.clip(lower=lo, upper=hi, axis=0)

# ========================
# Time-series functions
# ========================
def ts_delay(df: pd.DataFrame, d: int = 1) -> pd.DataFrame:
    return df.shift(d)

def ts_delta(df: pd.DataFrame, d: int = 1) -> pd.DataFrame:
    return df - df.shift(d)

def ts_mean(df: pd.DataFrame, d: int = 5) -> pd.DataFrame:
    return df.rolling(d, min_periods=d).mean()

def ts_std(df: pd.DataFrame, d: int = 5) -> pd.DataFrame:
    return df.rolling(d, min_periods=d).std(ddof=0)

def ts_zscore(df: pd.DataFrame, d: int = 5) -> pd.DataFrame:
    mu = ts_mean(df, d)
    sd = ts_std(df, d)
    return (df - mu) / sd.replace(0, np.nan)

def ts_max(df: pd.DataFrame, d: int = 5) -> pd.DataFrame:
    return df.rolling(d, min_periods=d).max()

def ts_min(df: pd.DataFrame, d: int = 5) -> pd.DataFrame:
    return df.rolling(d, min_periods=d).min()

# ========================
# Alphalens helper
# ========================
def to_factor_df(wide: pd.DataFrame) -> pd.DataFrame:
    """Convert wide (date×asset) to MultiIndex 'factor' format."""
    s = wide.stack().rename("factor").dropna()
    s.index = s.index.set_names(["date", "asset"])
    s.index = pd.MultiIndex.from_arrays([
        pd.to_datetime(s.index.get_level_values(0)).tz_localize(None),
        s.index.get_level_values(1)
    ], names=["date", "asset"])
    return s.to_frame()
def ln(df: pd.DataFrame) -> pd.DataFrame:
    """Natural log."""
    return np.log(df)

def log(df: pd.DataFrame, base: float = 10) -> pd.DataFrame:
    """Logarithm with custom base."""
    return np.log(df) / np.log(base)

def abs_(df: pd.DataFrame) -> pd.DataFrame:
    """Absolute value."""
    return df.abs()

def sign(df: pd.DataFrame) -> pd.DataFrame:
    """Sign function (+1, 0, -1)."""
    return np.sign(df)

def square(df: pd.DataFrame) -> pd.DataFrame:
    """Square."""
    return df ** 2

def sqrt(df: pd.DataFrame) -> pd.DataFrame:
    """Square root."""
    return np.sqrt(df.clip(lower=0))
def corr_ts(df1: pd.DataFrame, df2: pd.DataFrame, d: int = 10) -> pd.DataFrame:
    """
    Time-series correlation over rolling window d, per asset.
    """
    aligned = df1.align(df2, join="inner")
    x, y = aligned
    return x.rolling(d, min_periods=d).corr(y)
def ts_rank(df: pd.DataFrame, d: int = 10) -> pd.DataFrame:
    """
    Time-series rank over rolling window d, per asset.
    For each column (asset), rank the latest value within its past d values.

    Returns values between 0 and 1.
    """
    return df.rolling(d, min_periods=d).apply(
        lambda x: x.rank().iloc[-1] / len(x), raw=False
    )
def where(cond: pd.DataFrame, x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    """Element-wise conditional selection."""
    return pd.DataFrame(np.where(cond, x, y), index=x.index, columns=x.columns)

def greater(x: pd.DataFrame, y) -> pd.DataFrame:
    """Element-wise greater-than comparison (x > y)."""
    return x > y

def less(x: pd.DataFrame, y) -> pd.DataFrame:
    """Element-wise less-than comparison (x < y)."""
    return x < y

def mul(x, y) -> pd.DataFrame:
    """Element-wise multiplication."""
    return x * y
def ts_sum(df: pd.DataFrame, d: int = 3) -> pd.DataFrame:
    """Rolling sum over window d, per asset."""
    return df.rolling(d, min_periods=d).sum()
# --- math & basic ops ---
def returns(df: pd.DataFrame, n: int = 1) -> pd.DataFrame:
    return df.pct_change(n)

def decay_linear(df: pd.DataFrame, d: int = 10) -> pd.DataFrame:
    """
    Rolling linear-decay WMA over window d (weights d, d-1, ..., 1, normalized).
    Per asset (column-wise).
    """
    w = np.arange(1, d + 1)  # 1..d
    def _wma(x):
        if np.isnan(x).any():
            return np.nan
        return np.dot(x, w) / w.sum()
    return df.rolling(d, min_periods=d).apply(_wma, raw=True)

def ts_sum(df: pd.DataFrame, d: int = 3) -> pd.DataFrame:
    return df.rolling(d, min_periods=d).sum()

def scale_cs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectional scale: x / sum(|x|) per date.
    Safe for all-zeros rows.
    """
    denom = df.abs().sum(axis=1).replace(0, np.nan)
    out = df.div(denom, axis=0)
    return out.fillna(0)

def adv(df_amount: pd.DataFrame, d: int = 20) -> pd.DataFrame:
    """Average Dollar Volume over d days."""
    return df_amount.rolling(d, min_periods=d).mean()

def signed_power(df: pd.DataFrame, a: float = 2.0) -> pd.DataFrame:
    """SignedPower: sign(x) * |x|**a"""
    return np.sign(df) * (df.abs() ** a)
def ts_argmax(df: pd.DataFrame, d: int = 5) -> pd.DataFrame:
    """
    Time-series argmax position in last d days, per asset.
    Returns integer in [1..d], where 1 = most recent day is max.
    """
    return df.rolling(d, min_periods=d).apply(lambda x: np.argmax(x[::-1]) + 1, raw=True)
def indneutralize(df: pd.DataFrame, sector_map: pd.Series) -> pd.DataFrame:
    """
    Cross-sectional demeaning within each sector per date.
    sector_map: pd.Series, index=asset, value=sector_code
    """
    sector_map = sector_map.reindex(df.columns)
    return df - df.groupby(sector_map, axis=1).transform("mean")
def round_window(x: float) -> int:
    """Round halves up (e.g., 3.92795→4)."""
    return int(np.floor(x + 0.5))
