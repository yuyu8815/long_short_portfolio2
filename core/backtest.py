# core/backtest.py
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Literal, Union
from dataclasses import dataclass

# Optional plotting (kept same style you used)
try:
    import plotly.graph_objects as go
except Exception:
    go = None

# ----------------------------
# Small utils
# ----------------------------
def _safe_std(x: pd.Series) -> float:
    s = x.std()
    return float(s) if np.isfinite(s) and s != 0 else 0.0

def _align_two(a: Union[pd.Series, pd.DataFrame], b: Union[pd.Series, pd.DataFrame]):
    a1, b1 = a.align(b, join="inner", axis=0)
    return a1, b1

def _l1_normalize_rows(df: pd.DataFrame) -> pd.DataFrame:
    denom = df.abs().sum(axis=1).replace(0, np.nan)
    out = df.div(denom, axis=0)
    return out.fillna(0)

def _cap_weights(df: pd.DataFrame, cap: float) -> pd.DataFrame:
    if cap is None:
        return df
    return df.clip(lower=-cap, upper=cap)

# ----------------------------
# Public API
# ----------------------------
@dataclass
class PerfReport:
    summary: pd.DataFrame
    returns_by_period: pd.Series
    weights: Optional[pd.DataFrame] = None

def add_sector_column(df: pd.DataFrame, df_twse: pd.DataFrame) -> pd.DataFrame:
    """(Optional helper) Merge sector into long df by stock_id."""
    out = df.merge(df_twse[["stock_id", "sector"]], on="stock_id", how="left")
    return out

def indneutralize_wide(wide: pd.DataFrame, sector_map: pd.Series) -> pd.DataFrame:
    """Cross-sectional de-mean within sector per date (wide date×asset)."""
    sm = sector_map.reindex(wide.columns)
    return wide - wide.groupby(sm, axis=1).transform("mean")

def preprocess_factor(
    factor: pd.DataFrame,
    method: Literal["none", "winsor+zscore", "rank"] = "winsor+zscore",
    winsor_std: float = 4.0,
) -> pd.DataFrame:
    """Row-wise factor cleaning."""
    if method == "none":
        return factor
    if method == "rank":
        return factor.rank(axis=1, method="average", pct=True).sub(0.5, axis=0)  # center around 0
    # winsor + zscore
    mu = factor.mean(axis=1)
    sd = factor.std(axis=1, ddof=0)
    lo = mu - winsor_std * sd
    hi = mu + winsor_std * sd
    clipped = factor.clip(lower=lo, upper=hi, axis=0)
    mu2 = clipped.mean(axis=1)
    sd2 = clipped.std(axis=1, ddof=0).replace(0, np.nan)
    z = (clipped.sub(mu2, axis=0)).div(sd2, axis=0).fillna(0)
    return z

def factor_to_weights(
    factor: pd.DataFrame,
    sector_map: Optional[pd.Series] = None,
    neutralize: bool = False,
    l1_normalize: bool = True,
    leverage: float = 1.0,
    cap: Optional[float] = None,
) -> pd.DataFrame:
    """Convert cleaned factor (wide) into portfolio weights per row (sum |w| = 1 by default)."""
    f = factor.copy()
    if neutralize and sector_map is not None:
        f = indneutralize_wide(f, sector_map)
    if l1_normalize:
        w = _l1_normalize_rows(f)
    else:
        w = f
    if cap is not None:
        w = _cap_weights(w, cap)
        if l1_normalize:
            w = _l1_normalize_rows(w)
    if leverage and leverage != 1.0:
        w = w * float(leverage)
    return w

def compute_turnover(weights: pd.DataFrame) -> float:
    """Average per-period dollar turnover (sum of |Δw|, averaged over time)."""
    dw = weights.diff().abs().sum(axis=1)
    return float(dw.mean())

def backtest_core(
    factor: pd.DataFrame,                 # wide (date×asset), raw factor
    exp_ret: pd.DataFrame,                # wide (date×asset), expected/realized returns per period
    period_of_year: int,                  # 252 for daily, 52 for weekly, 12 for monthly
    benchmark: Optional[pd.Series] = None,# Series (date), same period returns
    fee: float = 0.0004,                  # per-unit L1 cost on |Δw|
    exec_lag: int = 0,                    # 0: use same-period factor, 1: trade next bar
    sector_map: Optional[pd.Series] = None,
    neutralize: bool = False,
    preprocess: Literal["none", "winsor+zscore", "rank"] = "winsor+zscore",
    leverage: float = 1.0,
    cap: Optional[float] = None,
    return_weights: bool = False,
    plot: bool = True,
) -> PerfReport:
    """
    End-to-end backtest:
    1) clean factor → 2) shift by exec_lag → 3) weights → 4) pnl minus fees → 5) summary + chart.
    """
    # 0) align
    factor, exp_ret = _align_two(factor, exp_ret)
    factor = factor.sort_index()
    exp_ret = exp_ret.sort_index()

    # 1) preprocess (row-wise)
    f_clean = preprocess_factor(factor, method=preprocess)

    # 2) exec lag
    if exec_lag:
        f_clean = f_clean.shift(exec_lag)

    # 3) weights
    w = factor_to_weights(
        f_clean,
        sector_map=sector_map,
        neutralize=neutralize,
        l1_normalize=True,
        leverage=leverage,
        cap=cap,
    )

    # 4) fees
    dw = w.diff().abs()
    fee_by_period = dw.sum(axis=1) * float(fee)

    # 5) portfolio returns
    # exp_ret is realized return per asset for the same period
    pnl_gross = (w * exp_ret).sum(axis=1)
    pnl_net = (pnl_gross - fee_by_period).dropna()

    # 6) trim both benchmark and pnl to common range
    if benchmark is not None:
        benchmark = pd.Series(benchmark).sort_index()
        pnl_net, benchmark = _align_two(pnl_net, benchmark)

    # 7) summary
    summary = _performance_report(pnl_net, benchmark, period_of_year)
    # 8) chart
    if plot and go is not None:
        _plot_cumret(pnl_net, benchmark)
    return PerfReport(summary=summary, returns_by_period=pnl_net, weights=w if return_weights else None)


def _sortino(x: pd.Series, period_of_year: int) -> float:
    """Annualized Sortino Ratio = mean / downside_std * sqrt(periods_per_year)."""
    downside = x[x < 0]
    if downside.empty:
        return np.nan
    downside_std = downside.std(ddof=0)
    if downside_std == 0 or not np.isfinite(downside_std):
        return np.nan
    return float(x.mean() / downside_std * np.sqrt(period_of_year))

def _cumprod_total(x: pd.Series) -> float:
    return float((1.0 + x).cumprod().iloc[-1] - 1.0)

def _cumsum_total(x: pd.Series) -> float:
    return float(x.sum())

def _sharpe(x: pd.Series, period_of_year: int) -> float:
    sd = _safe_std(x)
    if sd == 0:
        return 0.0
    return float(x.mean() / sd * np.sqrt(period_of_year))

def _vol(x: pd.Series, period_of_year: int) -> float:
    return float(_safe_std(x) * np.sqrt(period_of_year))

def _annual_ret(x: pd.Series, period_of_year: int) -> float:
    comp = (1.0 + x).cumprod()
    n = len(comp)
    if n <= 1:
        return 0.0
    total = float(comp.iloc[-1] - 1.0)
    return float((1.0 + total) ** (period_of_year / n) - 1.0)

def _mdd(x: pd.Series) -> float:
    comp = (1.0 + x).cumprod()
    dd = comp / comp.cummax() - 1.0
    return float(abs(dd.min()))

def _performance_report(returns_by_period: pd.Series,
                        benchmark: Optional[pd.Series],
                        period_of_year: int) -> pd.DataFrame:
    rbp = returns_by_period.dropna()
    if benchmark is not None:
        bench = benchmark.dropna()
        s_names = ["Performance", "Benchmark"]
        items = [
            [
                f"{_cumprod_total(rbp)*100:.2f} %",
                f"{_cumprod_total(bench)*100:.2f} %",
            ],
            [
                f"{_cumsum_total(rbp)*100:.2f} %",
                f"{_cumsum_total(bench)*100:.2f} %",
            ],
            [
                f"{_sharpe(rbp, period_of_year):.2f}",
                f"{_sharpe(bench, period_of_year):.2f}",
            ],
            [
                f"{_sortino(rbp, period_of_year):.2f}",
                f"{_sortino(bench, period_of_year):.2f}",
            ],
            [
                f"{_annual_ret(rbp, period_of_year)*100:.2f} %",
                f"{_annual_ret(bench, period_of_year)*100:.2f} %",
            ],
            [
                f"{_mdd(rbp)*100:.2f} %",
                f"{_mdd(bench)*100:.2f} %",
            ],
            [
                f"{_vol(rbp, period_of_year)*100:.2f} %",
                f"{_vol(bench, period_of_year)*100:.2f} %",
            ],
            [
                f"{rbp.std()*100:.2f} %",
                f"{bench.std()*100:.2f} %",
            ],
        ]
        df = pd.DataFrame(
            items,
            columns=s_names,
            index=[
                "Cumprod Total Returns",
                "Cumsum Total Returns",
                "Sharpe Ratio",
                "Sortino Ratio",     # <── 新增這行
                "Annualized Ret",
                "Max Drawdown",
                "Volatility",
                "STD",
            ],
        )
        return df

    else:
        s_names = ["Performance"]
        items = [
            [f"{_cumprod_total(rbp)*100:.2f} %"],
            [f"{_cumsum_total(rbp)*100:.2f} %"],
            [f"{_sharpe(rbp, period_of_year):.2f}"],
            [f"{_annual_ret(rbp, period_of_year)*100:.2f} %"],
            [f"{_mdd(rbp)*100:.2f} %"],
            [f"{_vol(rbp, period_of_year)*100:.2f} %"],
            [f"{rbp.std()*100:.2f} %"],
        ]
        df = pd.DataFrame(items,
                          columns=s_names,
                          index=[
                              "Cumprod Total Returns",
                              "Cumsum Total Returns",
                              "Sharpe Ratio",
                              "Annualized Ret",
                              "Max Drawdown",
                              "Volatility",
                              "STD",
                          ])
        return df

def _plot_cumret(rbp: pd.Series, benchmark: Optional[pd.Series]):
    comp = (1.0 + rbp).cumprod() - 1.0
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=comp.index, y=comp, mode="lines", name="Cumulative Returns"))
    if benchmark is not None:
        b = (1.0 + benchmark).cumprod() - 1.0
        fig.add_trace(go.Scatter(x=b.index, y=b, mode="lines", name="Benchmark"))
    fig.update_layout(title="Cumulative Returns",
                      xaxis_title="Date", yaxis_title="Cumulative Return")
    fig.show()
