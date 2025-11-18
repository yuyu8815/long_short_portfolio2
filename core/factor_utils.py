import pandas as pd
import numpy as np

def demean_and_standardize(factor_df: pd.DataFrame, clip: float = 5.0) -> pd.DataFrame:
    """
    Cross-sectionally demean and standardize factor values each day.
    """
    mean_ = factor_df.mean(axis=1)
    std_  = factor_df.std(axis=1).replace(0, np.nan)
    z = factor_df.sub(mean_, axis=0).div(std_, axis=0)
    if clip is not None:
        z = z.clip(-clip, clip)
    return z.fillna(0.0)

def rank_factor(factor_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert factor values into within-day ranks scaled to [-0.5, 0.5].
    """
    ranks = factor_df.rank(axis=1, pct=True)
    return ranks - 0.5
