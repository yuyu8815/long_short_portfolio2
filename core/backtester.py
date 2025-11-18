import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_capital_backtest(
    factor: pd.DataFrame,
    open_px: pd.DataFrame,
    initial_capital: float = 10_000_000,
    long_frac: float = 0.3,
    short_frac: float = 0.3,
    cost_rate: float = 0.00145,
    plot: bool = True,
):
    """
    Capital-based long–short backtest:
    - Start with initial_capital (e.g. 10 million TWD)
    - Each day use factor(t-1) to allocate weights for day t
    - Trade at open, hold until next open
    - Equal-weight long & short, dollar-neutral
    """

    factor = factor.sort_index()
    open_px = open_px.reindex(factor.index).sort_index()

    # open-to-open returns
    r_oo = open_px.shift(-1).div(open_px).sub(1.0)
    r_oo = r_oo.iloc[:-1]
    sig = factor.shift(1).iloc[:-1]  # use previous day's factor

    # rank & select
    ranks = sig.rank(axis=1, pct=True)
    long_mask = ranks >= 1 - long_frac
    short_mask = ranks <= short_frac

    # equal-weight weights (sum |w| = 1)
    w = pd.DataFrame(0.0, index=sig.index, columns=sig.columns)
    w[long_mask] = 1.0
    w[short_mask] = -1.0
    w = w.div(w.abs().sum(axis=1), axis=0).fillna(0.0)

    # --- 模擬資金變化 ---
    capital = initial_capital
    capital_series = []
    daily_pnl = []
    turnover_series = []

    prev_w = pd.Series(0, index=w.columns)
    for date in w.index:
        weight = w.loc[date]
        ret = r_oo.loc[date]

        # 當天組合報酬
        daily_ret = (weight * ret).sum()

        # 估算換倉成本（依權重變化）
        turnover = (weight - prev_w).abs().sum()
        cost = capital * cost_rate * turnover

        pnl = capital * daily_ret - cost
        capital += pnl

        capital_series.append(capital)
        daily_pnl.append(pnl)
        turnover_series.append(turnover)
        prev_w = weight

    capital_series = pd.Series(capital_series, index=w.index, name="Capital")
    daily_pnl = pd.Series(daily_pnl, index=w.index, name="DailyPnL")
    turnover_series = pd.Series(turnover_series, index=w.index, name="Turnover")

    # --- 統計 ---
    ret_pct = daily_pnl / capital_series.shift(1)
    ann_ret = (capital_series.iloc[-1] / initial_capital) ** (252 / len(capital_series)) - 1
    ann_vol = ret_pct.std() * np.sqrt(252)
    sharpe = ret_pct.mean() / ret_pct.std() * np.sqrt(252)
    maxdd = (capital_series / capital_series.cummax() - 1).min()

    stats = {
        "FinalCapital": capital_series.iloc[-1],
        "AnnReturn": ann_ret,
        "AnnVol": ann_vol,
        "Sharpe": sharpe,
        "MaxDD": maxdd,
    }

    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(capital_series.index, capital_series.values, label="Capital Curve")
        plt.title(f"Capital-based Long–Short Portfolio (Start={initial_capital:,.0f})")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value")
        plt.grid(True)
        plt.legend()
        plt.show()

        print("Performance Stats:")
        for k, v in stats.items():
            if "Capital" in k:
                print(f"{k}: {v:,.0f}")
            else:
                print(f"{k}: {v:.4f}")

    return {
        "capital": capital_series,
        "daily_pnl": daily_pnl,
        "turnover": turnover_series,
        "stats": stats,
    }
