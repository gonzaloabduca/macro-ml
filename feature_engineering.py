
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin    

from functions import lead_lag_correlation_df
from macro import compute_housing_pca

from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)




def best_lag_table(corr_df: pd.DataFrame, min_values: int = 2) -> pd.DataFrame:
    
    rows = []

    for col in corr_df.columns:
        corr = pd.to_numeric(corr_df[col], errors="coerce").dropna()

        if len(corr) < min_values:
            continue

        raw_peak_lag = corr.idxmax()
        raw_peak_corr = corr.max()

        abs_peak_lag = corr.abs().idxmax()
        abs_peak_corr = corr.loc[abs_peak_lag]

        rows.append({
            "factor": col,
            "best_positive_lag": raw_peak_lag,
            "best_positive_corr": raw_peak_corr,
            "best_absolute_lag": abs_peak_lag,
            "best_absolute_corr": abs_peak_corr,
        })

    return pd.DataFrame(rows)

def create_best_lagged_features(
    df: pd.DataFrame,
    target: pd.Series,
    max_lag: int = 12,
    use_absolute: bool = True,
    drop_zero_lags: bool = True
) -> pd.DataFrame:
    """
    Creates one best-lagged feature per column based on lead-lag correlation.

    Parameters
    ----------
    drop_zero_lags : bool
        If True, ignores features whose best lag is 0.
    """

    corr_df = lead_lag_correlation_df(df, target, max_lag=max_lag)
    lag_table = best_lag_table(corr_df)

    lag_col = "best_absolute_lag" if use_absolute else "best_positive_lag"

    if drop_zero_lags:
        lag_table = lag_table[lag_table[lag_col] > 1]

    lagged_features = pd.DataFrame(index=df.index)

    for _, row in lag_table.iterrows():
        factor = row["factor"]
        best_lag = int(row[lag_col])

        lagged_features[f"{factor}_lag{best_lag}"] = df[factor].shift(best_lag)

    return lagged_features.dropna()




def compute_pls_factors(
    X: pd.DataFrame,
    y: pd.Series,
    n_components: int = 2,
    prefix: str = "pls"
) -> tuple[pd.DataFrame, PLSRegression, StandardScaler, StandardScaler]:
    """
    Computes supervised PLS factors.

    Unlike PCA, PLS uses y when creating the factors.
    """

    data = pd.concat([X, y.rename("target")], axis=1).dropna()

    X_clean = data[X.columns]
    y_clean = data["target"]

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_scaled = x_scaler.fit_transform(X_clean)
    y_scaled = y_scaler.fit_transform(y_clean.to_frame())

    pls = PLSRegression(n_components=n_components)
    pls.fit(X_scaled, y_scaled)

    factors = pls.x_scores_

    factors_df = pd.DataFrame(
        factors,
        index=X_clean.index,
        columns=[f"{prefix}_PLS{i+1}" for i in range(n_components)]
    )

    return factors_df, pls, x_scaler, y_scaler


def compute_pls_factors_for_horizon(
    X: pd.DataFrame,
    y: pd.Series,
    horizon: int = 2,
    n_components: int = 2,
    prefix: str = "pls"
):
    """
    Computes PLS factors where X_t is trained to predict y_{t+horizon}.
    """

    y_forward = y.shift(-horizon).rename("target")

    data = pd.concat([X, y_forward], axis=1).dropna()

    X_clean = data[X.columns]
    y_clean = data["target"]

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_scaled = x_scaler.fit_transform(X_clean)
    y_scaled = y_scaler.fit_transform(y_clean.to_frame())

    pls = PLSRegression(n_components=n_components)
    pls.fit(X_scaled, y_scaled)

    factors = pls.x_scores_

    factors_df = pd.DataFrame(
        factors,
        index=X_clean.index,
        columns=[f"{prefix}_PLS{i+1}" for i in range(n_components)]
    )

    return factors_df, pls, x_scaler, y_scaler



def xgboost_feature_importance(X: pd.DataFrame, y: pd.Series, n_estimators: int = 100) -> pd.DataFrame:
    """
    Computes feature importance using XGBoost.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable.
    n_estimators : int, optional
        Number of trees in the ensemble, by default 100.

    Returns
    -------
    pd.DataFrame
        DataFrame containing features and their corresponding importance scores.
    """

    import xgboost as xgb


    X = create_best_lagged_features(X, y, max_lag=12, use_absolute=True)

    data = pd.concat([X, y.rename("target")], axis=1).dropna()

    X = data.drop(columns="target")
    y = data["target"]

    model = xgb.XGBRegressor(n_estimators=n_estimators)
    model.fit(X, y)

    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False)

    return importance_df.reset_index(drop=True)





class BestLagSelector(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        max_lag=12,
        use_absolute=True,
        drop_zero_lags=True,
        min_values=2,
        suffix=True,
    ):
        self.max_lag = max_lag
        self.use_absolute = use_absolute
        self.drop_zero_lags = drop_zero_lags
        self.min_values = min_values
        self.suffix = suffix

    def fit(self, X, y):
        X = pd.DataFrame(X).copy()
        y = pd.Series(y).copy()

        corr_df = lead_lag_correlation_df(
            X,
            y,
            max_lag=self.max_lag
        )

        lag_table = best_lag_table(
            corr_df,
            min_values=self.min_values
        )

        lag_col = "best_absolute_lag" if self.use_absolute else "best_positive_lag"

        if self.drop_zero_lags:
            lag_table = lag_table[lag_table[lag_col] != 0]

        self.lag_table_ = lag_table.copy()
        self.lag_col_ = lag_col
        self.input_features_ = X.columns.tolist()

        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()

        lagged_features = {}

        for _, row in self.lag_table_.iterrows():
            factor = row["factor"]
            lag = int(row[self.lag_col_])

            if factor not in X.columns:
                continue

            if self.suffix:
                new_name = f"{factor}_lag{lag}"
            else:
                new_name = factor

            lagged_features[new_name] = X[factor].shift(lag)

        return pd.DataFrame(
            lagged_features,
            index=X.index
        )
    


class AllLagFeatureGenerator(BaseEstimator, TransformerMixin):
    def __init__(self, max_lag=12, include_lag0=True):
        self.max_lag = max_lag
        self.include_lag0 = include_lag0

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        self.input_features_ = X.columns.tolist()
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()

        start_lag = 0 if self.include_lag0 else 1
        lagged_features = {}

        for col in self.input_features_:
            if col not in X.columns:
                continue

            for lag in range(start_lag, self.max_lag + 1):
                lagged_features[f"{col}_lag{lag}"] = X[col].shift(lag)

        return pd.DataFrame(lagged_features, index=X.index)
    
def build_gdp_dataset(
    macro,
    money,
    market_features,
    prior_gdp,
    current_gdp,
):
    X_gdp = pd.concat(
        [
            macro,
            money,
            market_features,
            prior_gdp,
        ],
        axis=1,
        sort=False,
    ).ffill().dropna()

    dataset = pd.concat(
        [
            current_gdp["gdp_yoy_current"].rename("target"),
            X_gdp,
        ],
        axis=1,
        sort=False,
    ).dropna()

    X = dataset.drop(columns="target")
    y = dataset["target"]

    return X, y


def calculate_xgb_shap_importance(
    X,
    y,
    model_params=None,
):
    """
    Fit XGBoost and calculate mean absolute SHAP importance.
    """

    dataset = pd.concat(
        [
            y.rename("target"),
            X,
        ],
        axis=1,
        sort=False,
    ).dropna()

    y_clean = dataset["target"].astype(float)
    X_clean = dataset.drop(columns="target").astype(float)

    params = {
        "objective": "reg:squarederror",
        "n_estimators": 500,
        "learning_rate": 0.03,
        "max_depth": 2,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.5,
        "reg_lambda": 3,
        "random_state": 42,
    }

    if model_params is not None:
        params.update(model_params)

    model = XGBRegressor(**params)
    model.fit(X_clean, y_clean)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_clean)

    importance = pd.DataFrame({
        "feature": X_clean.columns,
        "importance": np.abs(shap_values).mean(axis=0),
    })

    importance["importance_pct"] = (
        importance["importance"]
        / importance["importance"].sum()
    )

    importance = (
        importance
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "model": model,
        "importance": importance,
        "shap_values": shap_values,
        "X": X_clean,
        "y": y_clean,
    }




def build_forward_gdp_dataset(
    current_gdp: pd.DataFrame,
    macro: pd.DataFrame,
    money: pd.DataFrame,
    market_features: pd.DataFrame,
    prior_gdp: pd.DataFrame,
    horizon: int = 1,
):
    """
    Build a dataset for forecasting GDP growth several quarters ahead.

    Parameters
    ----------
    current_gdp : pd.DataFrame
        Must contain 'gdp_yoy_current'.

    macro, money, market_features, prior_gdp : pd.DataFrame
        Predictor blocks indexed by date.

    horizon : int, default 1
        Forecast horizon in quarters.

    Returns
    -------
    X : pd.DataFrame
        Features available at time t.

    y : pd.Series
        GDP YoY growth at time t + horizon.
    """

    if horizon < 1:
        raise ValueError("horizon must be at least 1 quarter.")

    X = pd.concat(
        [
            macro,
            money,
            market_features,
            prior_gdp,
        ],
        axis=1,
        sort=False,
    )

    X = (
        X
        .sort_index()
        .ffill()
    )

    # Convert monthly features to quarterly observations
    X = X.resample("QE").last()

    gdp = (
        current_gdp["gdp_yoy_current"]
        .sort_index()
        .resample("QE")
        .last()
    )

    # At date t, predict GDP at t + horizon
    y = gdp.shift(-horizon).rename(
        f"gdp_yoy_forward_{horizon}q"
    )

    dataset = pd.concat(
        [y, X],
        axis=1,
        sort=False,
    ).dropna()

    y = dataset[y.name].astype(float)
    X = dataset.drop(columns=y.name)

    X = X.select_dtypes(include=[np.number]).astype(float)

    # Remove duplicate column names if present
    X = X.loc[:, ~X.columns.duplicated()]

    return X, y





def rolling_performance_stats(
    returns_df: pd.DataFrame,
    window: int = 252,
    annualization_factor: int = 252,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """
    Calculate rolling performance statistics for daily industry returns.

    Returns
    -------
    pd.DataFrame
        MultiIndex rows: date, industry.
        Columns:
            annualized_return
            rolling_sharpe
            rolling_sortino
            volatility_percentile
    """

    if min_periods is None:
        min_periods = window

    returns = (
        returns_df
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .rename_axis(index="date", columns="industry")
        .stack()
        .rename("return")
        .reorder_levels(["industry", "date"])
        .sort_index()
    )

    grouped = returns.groupby(level="industry", group_keys=False)

    rolling_mean = (
        grouped
        .rolling(window=window, min_periods=min_periods)
        .mean()
        .droplevel(0)
    )

    rolling_volatility = (
        grouped
        .rolling(window=window, min_periods=min_periods)
        .std(ddof=1)
        .droplevel(0)
    )

    annualized_return = (
        grouped
        .rolling(window=window, min_periods=min_periods)
        .apply(
            lambda x: (
                np.prod(1 + x)
                ** (annualization_factor / x.count())
                - 1
            ),
            raw=False,
        )
        .droplevel(0)
    )

    rolling_sharpe = (
        rolling_mean
        .div(rolling_volatility)
        .mul(np.sqrt(annualization_factor))
    )

    downside_squared = returns.clip(upper=0).pow(2)

    downside_deviation = (
        downside_squared
        .groupby(level="industry", group_keys=False)
        .rolling(window=window, min_periods=min_periods)
        .mean()
        .pow(0.5)
        .droplevel(0)
    )

    rolling_sortino = (
        rolling_mean
        .div(downside_deviation)
        .mul(np.sqrt(annualization_factor))
    )

    annualized_volatility = (
        rolling_volatility * np.sqrt(annualization_factor)
    )

    volatility_percentile = (
        annualized_volatility
        .groupby(level="industry", group_keys=False)
        .transform(
            lambda x: x.expanding(min_periods=1).rank(pct=True)
        )
    )

    stats = pd.concat(
        {
            "annualized_return": annualized_return,
            "rolling_sharpe": rolling_sharpe,
            "rolling_sortino": rolling_sortino,
            "volatility_percentile": volatility_percentile,
        },
        axis=1,
    )

    stats.index = stats.index.reorder_levels(["date", "industry"])
    stats = stats.sort_index()

    return stats


def get_hmm_states(returns_df: pd.DataFrame, n_states: int = 4) -> pd.DataFrame:
    """
    Fit a Hidden Markov Model to macroeconomic features and return the most likely state for each date.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Macroeconomic features with dates as index and features as columns.
    n_states : int, optional
        Number of hidden states in the HMM, by default 4.

    Returns
    -------
    pd.DataFrame
        DataFrame with dates as index and a single column 'state' indicating the most likely state for each date.
    """

    from hmmlearn import hmm

    scaler = StandardScaler()

    # Prepare the data for HMM
    features = scaler.fit_transform(returns_df.dropna().values)

    # Fit the HMM model
    model = hmm.GaussianHMM(n_components=n_states, covariance_type="full", n_iter=1000)
    model.fit(features)

    # Predict the hidden states
    hidden_states = model.predict(features)

    # Create a DataFrame for the states
    states_df = pd.DataFrame(hidden_states, index=returns_df.dropna().index, columns=["state"])

    return states_df


import re

def create_fixed_lagged_features(
    data: pd.DataFrame,
    feature_names: list[str],
    prefix: str = "",
) -> pd.DataFrame:
    """
    Recreate model features from saved feature names.

    Example
    -------
    inflation__energy_PC1_lag8
    becomes:
        data["energy_PC1"].shift(8)
    """

    output = pd.DataFrame(index=data.index)

    for full_feature_name in feature_names:
        feature_name = full_feature_name

        if prefix and feature_name.startswith(prefix):
            feature_name = feature_name[len(prefix):]

        match = re.fullmatch(
            r"(.+)_lag(\d+)",
            feature_name,
        )

        if match is None:
            continue

        source_column = match.group(1)
        lag = int(match.group(2))

        if source_column not in data.columns:
            raise ValueError(
                f"Source column '{source_column}' required for "
                f"'{full_feature_name}' is missing."
            )

        output[full_feature_name] = (
            data[source_column].shift(lag)
        )

    return output