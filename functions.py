import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from fredapi import Fred



def zscore(series, window=60):
    """
    Calculate the z-score of a pandas Series.
    
    Parameters:
    series (pd.Series): The input data series.
    
    Returns:
    pd.Series: The z-score of the input series.
    """
    return (series - series.rolling(window).mean()) / series.rolling(window).std()


def momentum(series, fast_window=60, slow_window=120, signal_window=20, zscore_window=60):
    """
    Calculate the momentum of a pandas Series.
    
    Parameters:
    series (pd.Series): The input data series.
    
    Returns:
    pd.Series: The z-score momentum of the input series.
    """

    series = series.copy()

    fast_ema = series.ewm(span=fast_window, adjust=False).mean()
    slow_ema = series.ewm(span=slow_window, adjust=False).mean()

    macd = fast_ema - slow_ema

    histogram = macd - macd.ewm(span=signal_window, adjust=False).mean()


    return zscore(histogram, window=zscore_window)

def run_single_factor_pca(df, factor_name, n_components=1):

    df = df.copy()

    yoy_change = df.pct_change(12).dropna()
    yoy_change = zscore(yoy_change, window=60).dropna()
    yoy_change = yoy_change.add_suffix("_yoy_zs")

    roc = momentum(
        df,
        fast_window=3,
        slow_window=12,
        signal_window=6,
        zscore_window=60
    ).dropna()
    roc = roc.add_suffix("_roc")

    features = pd.concat([yoy_change, roc], axis=1).dropna()

    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(X)

    pc_names = [f"{factor_name}_PC{i+1}" for i in range(n_components)]

    factors = pd.DataFrame(
        pcs,
        index=features.index,
        columns=pc_names
    )

    loadings = pd.DataFrame(
        pca.components_.T,
        index=features.columns,
        columns=pc_names
    )

    return factors.dropna()


def lead_lag_correlation(factor, target, max_lag=12, threshold=0.25):
    
    corr = pd.Series({
        lag: factor.shift(lag).corr(target)
        for lag in range(max_lag + 1)
    })

    return corr.where(corr.abs() >= threshold).dropna()


def lead_lag_correlation_df(factors, target, max_lag=12):

    """
    Compute lead-lag correlations between multiple factors and a target series.

    Parameters
    ----------
    factors : pd.DataFrame
        Data frame containing the factors to be analyzed.
    target : pd.Series
        The target series to compute correlations against.
    max_lag : int, optional
        The maximum lag to consider for correlation. Default is 12.

    Returns
    -------
    pd.DataFrame
        Data frame containing lead-lag correlations for each factor.
    """
    return pd.DataFrame(
        {
            factor: lead_lag_correlation(factors[factor], target, max_lag=max_lag)
            for factor in factors.columns
        }
    )

def convert_to_float(df):
        """
        Convert integers into floats

        """
        int_cols = df.select_dtypes(include=["integer"]).columns

        df[int_cols] = df[int_cols].astype(float)

        return df


def parse_dates_monthly(df):
    df.index = (
        pd.to_datetime(df.index.astype(str), format="%Y%m")
        + pd.offsets.MonthEnd(0)
    )
    return df


def consecutive_true(condition):
    condition = condition.fillna(False).astype(bool)

    groups = condition.ne(condition.shift()).cumsum()

    return (
        condition.astype(int)
        .groupby(groups)
        .cumsum()
        .where(condition, 0)
    )

def consecutive_positive(series):

    return consecutive_true(series.diff() > 0)


def to_quarter_end(df):
    df = df.copy()
    df.index = pd.to_datetime(df.index).to_period("Q").to_timestamp("Q")
    return df.sort_index()


def consecutive_true_days(condition: pd.Series) -> pd.Series:
    """
    Counts consecutive days where condition is True.
    Resets to 0 when condition is False.
    """
    condition = condition.fillna(False).astype(bool)
    groups = (~condition).cumsum()
    return condition.groupby(groups).cumcount() + 1


def days_since_event(event: pd.Series) -> pd.Series:
    """
    Counts days since event was last True.
    Event day = 0.
    """
    event = event.fillna(False).astype(bool)

    idx = np.arange(len(event))
    last_event_idx = pd.Series(
        np.where(event, idx, np.nan),
        index=event.index
    ).ffill()

    return pd.Series(idx, index=event.index) - last_event_idx



def avoid_future_leakage(series: pd.Series, offset_months: int = 6) -> pd.Series:
    """
    Shift a series forward by a specified number of months to avoid future leakage.
    """
    series = series.copy()
    series.index = series.index + pd.DateOffset(months=offset_months) - pd.DateOffset(days=1)

    return series




def consecutive_true_days(condition: pd.Series) -> pd.Series:
    condition = condition.fillna(False).astype(bool)
    groups = (~condition).cumsum()
    counts = condition.groupby(groups).cumcount() + 1
    return counts.where(condition, 0)


def days_since_event(event: pd.Series) -> pd.Series:
    event = event.fillna(False).astype(bool)

    idx = np.arange(len(event))
    last_event_idx = pd.Series(
        np.where(event, idx, np.nan),
        index=event.index
    ).ffill()

    return pd.Series(idx, index=event.index) - last_event_idx


def rolling_percentile(s: pd.Series, window: int = 756) -> pd.Series:
    return s.rolling(window).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1],
        raw=False
    )


def ratio_features(a: pd.Series, b: pd.Series, name: str, window: int = 756) -> pd.DataFrame:
    ratio = a / b
    out = pd.DataFrame(index=ratio.index)

    out[f"{name}_ratio"] = ratio
    out[f"{name}_mom_6m"] = ratio.pct_change(126)
    out[f"{name}_zscore_3y"] = zscore(ratio, window)

    return out



def get_rolling_sortino_ratio(returns_df, rf_series, window=252, periods_per_year=252):
    
    df = returns_df.join(rf_series.rename('rf')).dropna()
    
    excess = df[returns_df.columns].sub(df['rf'], axis=0)
    
    rolling_mean = excess.rolling(window).mean() * periods_per_year
    
    downside = excess.clip(upper=0)
    rolling_downside_std = downside.rolling(window).std(ddof=1) * np.sqrt(periods_per_year)
    
    rolling_sortino = rolling_mean / rolling_downside_std
    
    return rolling_sortino

def simple_sortino_ratio(returns, window=252, periods_per_year=252):
    
    
    rolling_mean = returns.rolling(window).mean() * periods_per_year
    
    downside = returns.clip(upper=0)
    rolling_downside_std = downside.rolling(window).std(ddof=1) * np.sqrt(periods_per_year)
    
    rolling_sortino = rolling_mean / rolling_downside_std
    
    return rolling_sortino


def rolling_tail_ratio(df, window=252, upper_q=0.90, lower_q=0.10):
    upper = df.rolling(window).quantile(upper_q)
    lower = df.rolling(window).quantile(lower_q).abs()
    tail_ratio = upper / lower
    zs_tail = (tail_ratio - tail_ratio.rolling(756).mean()) / tail_ratio.rolling(756).std()
    return zs_tail


def rolling_score(returns_df, window=252, periods_per_year=252):
    sortino = simple_sortino_ratio(returns_df, window=window, periods_per_year=periods_per_year)
    tail = rolling_tail_ratio(returns_df, window)
    score = (sortino * 0.7 + tail * 0.3)
    return score

def calculate_rolling_score(x, rf):
    x = x.sort_index()

    if isinstance(x.index, pd.MultiIndex):
        # Replace "Date" with the actual datetime level name if necessary
        dates = x.index.get_level_values("Date")
    else:
        dates = x.index

    returns = x.pct_change()

    rf_aligned = (
        rf.sort_index()
        .reindex(pd.DatetimeIndex(dates), method="ffill")
    )

    # rolling_score should receive two Series with identical indexes
    rf_aligned.index = returns.index

    score = rolling_score(
        returns_df=returns,
        rf_series=rf_aligned,
        window=252,
    )

    return score.reindex(x.index)
