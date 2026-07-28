import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from functions import *
from feature_engineering import *
from macro import *
from macro_plots import *


# ============================================================
# 1. SETTINGS
# ============================================================

ZSCORE_WINDOW = 60
MAX_LAG = 18
RIDGE_ALPHA = 10.0
MIN_TRAIN_SIZE = 40
TOP_N_FEATURES = 30


# ============================================================
# 2. LOAD GDP TARGET
# ============================================================

regime = pd.read_pickle("data/macro_quadrants.pkl").copy()

regime["gdp_yoy_diff_z"] = zscore(
    regime["gdp_yoy"].diff(),
    window=80,
)

regime["cpi_yoy_diff_z"] = zscore(
    regime["cpi_yoy"].diff(),
    window=80,
)

gdp_target = (
    regime["gdp_yoy_z"]
    .rename("target_gdp_z")
)

print("GDP target range:")
print(gdp_target.dropna().index.min())
print(gdp_target.dropna().index.max())


# ============================================================
# 3. LOAD MONTHLY MACRO DATA
# ============================================================

macro_raw = pd.read_pickle("data/raw_macro_data.pkl").copy()
money_raw = pd.read_pickle("data/raw_money_market_data.pkl").copy()

macro_raw = macro_raw.sort_index()
money_raw = money_raw.sort_index()

macro_raw.index = pd.DatetimeIndex(macro_raw.index)
money_raw.index = pd.DatetimeIndex(money_raw.index)


# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================

macro_z = (
    zscore(
        macro_raw,
        window=ZSCORE_WINDOW,
    )
    .add_suffix("_z")
)

macro_momentum = (
    momentum(
        macro_raw,
        fast_window=3,
        slow_window=12,
        signal_window=6,
    )
    .add_suffix("_momentum")
)

macro_features = pd.concat(
    [
        macro_z,
        macro_momentum,
    ],
    axis=1,
).replace(
    [np.inf, -np.inf],
    np.nan,
)


money_z = (
    zscore(
        money_raw,
        window=ZSCORE_WINDOW,
    )
    .add_suffix("_z")
)

money_momentum = (
    momentum(
        money_raw,
        fast_window=3,
        slow_window=12,
        signal_window=6,
    )
    .add_suffix("_momentum")
)

money_features = pd.concat(
    [
        money_z,
        money_momentum,
    ],
    axis=1,
).replace(
    [np.inf, -np.inf],
    np.nan,
)


# ============================================================
# 5. SELECT BEST LAG FOR EACH MONTHLY FEATURE
# ============================================================

macro_lagged_features = create_best_lagged_features(
    macro_features,
    gdp_target,
    max_lag=MAX_LAG,
).add_prefix("macro__")

money_lagged_features = create_best_lagged_features(
    money_features,
    gdp_target,
    max_lag=MAX_LAG,
).add_prefix("money__")


# ============================================================
# 6. BUILD MONTHLY FEATURE MATRIX
# ============================================================

macro_monthly = pd.concat(
    [
        macro_lagged_features,
        money_lagged_features,
    ],
    axis=1,
).sort_index()

macro_monthly = macro_monthly.replace(
    [np.inf, -np.inf],
    np.nan,
)

if not macro_monthly.columns.is_unique:
    duplicate_columns = (
        macro_monthly.columns[
            macro_monthly.columns.duplicated(
                keep=False
            )
        ]
        .unique()
        .tolist()
    )

    raise ValueError(
        f"Duplicate monthly columns found: "
        f"{duplicate_columns}"
    )

print(
    "Monthly feature matrix:",
    macro_monthly.shape,
)


# ============================================================
# 7. CREATE QUARTERLY TRAINING FEATURES
# ============================================================

reduced_features = [
 'money__10y_yield_z_lag3',
 'macro__building_permits_z_lag11',
 'macro__pce_z_lag3',
 'macro__personal_disposable_income_z_lag5',
 'macro__manu_employment_z_lag9',
 'macro__industrial_production_z_lag3',
 'macro__capacity_utilization_z_lag4',
 'macro__ism_manufacturing_index_z_lag9',
 'macro__spy_eps_z_lag4',
 'macro__manu_production_z_lag9',
 'macro__manu_inventories_z_lag6',
 'macro__manufacturing_hours_z_lag4',
 'macro__manu_prices_z_lag7',
 'macro__personal_disposable_income_momentum_lag5',
 'macro__pce_momentum_lag10',
 'macro__manu_deliveries_z_lag8',
 'macro__payrolls_momentum_lag9',
 'money__financial_conditions_index_z_lag12',
 'money__M2_money_supply_momentum_lag6']

macro_q = (
    macro_monthly
    .resample("QE")
    .last()
)

X_q = macro_q[reduced_features].copy()

# Previous-quarter GDP information
X_q["gdp_z_lag1"] = (
    regime["gdp_yoy_z"]
    .shift(1)
    .reindex(X_q.index)
)

X_q["gdp_z_diff_lag1"] = (
    regime["gdp_yoy_z"]
    .diff()
    .shift(1)
    .reindex(X_q.index)
)

X_q = X_q.replace(
    [np.inf, -np.inf],
    np.nan,
)

if not X_q.columns.is_unique:
    duplicate_columns = (
        X_q.columns[
            X_q.columns.duplicated(
                keep=False
            )
        ]
        .unique()
        .tolist()
    )

    raise ValueError(
        f"Duplicate quarterly columns found: "
        f"{duplicate_columns}"
    )

model_features = X_q.columns.tolist()

print(
    "Quarterly feature matrix:",
    X_q.shape,
)

print(
    "Number of model features:",
    len(model_features),
)


# ============================================================
# 8. JOIN CURRENT-QUARTER GDP TARGET
# ============================================================

model_frame = X_q.join(
    gdp_target,
    how="left",
)

target_count = (
    model_frame.columns
    == "target_gdp_z"
).sum()

if target_count != 1:
    raise ValueError(
        f"Expected exactly one target column, "
        f"found {target_count}"
    )

model_data = model_frame.dropna(
    subset=model_features
    + ["target_gdp_z"]
).copy()

print(
    "Training dataset:",
    model_data.shape,
)

print(
    "Training period:",
    model_data.index.min(),
    "to",
    model_data.index.max(),
)


# ============================================================
# 9. EXPANDING-WINDOW NOWCAST BACKTEST
# ============================================================

predictions = []
baseline_predictions = []
actuals = []
dates = []

for i in range(
    MIN_TRAIN_SIZE,
    len(model_data),
):
    train = model_data.iloc[:i]
    test = model_data.iloc[[i]]

    model = Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "ridge",
                Ridge(
                    alpha=RIDGE_ALPHA,
                ),
            ),
        ]
    )

    model.fit(
        train[model_features],
        train["target_gdp_z"],
    )

    prediction = model.predict(
        test[model_features]
    )[0]

    # Persistence baseline:
    # current-quarter GDP equals previous-quarter GDP
    baseline = test[
        "gdp_z_lag1"
    ].iloc[0]

    actual = test[
        "target_gdp_z"
    ].iloc[0]

    predictions.append(prediction)
    baseline_predictions.append(baseline)
    actuals.append(actual)
    dates.append(test.index[0])


results = pd.DataFrame(
    {
        "actual": actuals,
        "prediction": predictions,
        "baseline": baseline_predictions,
    },
    index=pd.DatetimeIndex(dates),
)

results.index.name = "date"


# ============================================================
# 10. PERFORMANCE METRICS
# ============================================================

actual_change = (
    results["actual"]
    - results["baseline"]
)

predicted_change = (
    results["prediction"]
    - results["baseline"]
)

metrics = pd.Series(
    {
        "Model MAE": mean_absolute_error(
            results["actual"],
            results["prediction"],
        ),
        "Baseline MAE": mean_absolute_error(
            results["actual"],
            results["baseline"],
        ),
        "Model RMSE": np.sqrt(
            mean_squared_error(
                results["actual"],
                results["prediction"],
            )
        ),
        "Baseline RMSE": np.sqrt(
            mean_squared_error(
                results["actual"],
                results["baseline"],
            )
        ),
        "Model R2": r2_score(
            results["actual"],
            results["prediction"],
        ),
        "Baseline R2": r2_score(
            results["actual"],
            results["baseline"],
        ),
        "Model correlation": (
            results["actual"]
            .corr(
                results["prediction"]
            )
        ),
        "Baseline correlation": (
            results["actual"]
            .corr(
                results["baseline"]
            )
        ),
        "Directional accuracy": (
            np.sign(actual_change)
            == np.sign(predicted_change)
        ).mean(),
    }
)

print("\nBacktest metrics:")
print(metrics)

print(
    "\nDirectional accuracy:",
    f"{metrics['Directional accuracy']:.2%}",
)


# ============================================================
# 11. FIT FINAL MODEL
# ============================================================

final_model = Pipeline(
    [
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "ridge",
            Ridge(
                alpha=RIDGE_ALPHA,
            ),
        ),
    ]
)

final_model.fit(
    model_data[model_features],
    model_data["target_gdp_z"],
)

import joblib

latest_gdp_mean = (
    regime["gdp_yoy"]
    .rolling(ZSCORE_WINDOW)
    .mean()
    .iloc[-1]
)

latest_gdp_std = (
    regime["gdp_yoy"]
    .rolling(ZSCORE_WINDOW)
    .std()
    .iloc[-1]
)

artifacts = {
    "model": final_model,
    "features": model_features,
    "gdp_mean": latest_gdp_mean,
    "gdp_std": latest_gdp_std,
    "zscore_window": ZSCORE_WINDOW,
}

joblib.dump(
    artifacts,
    "ml_models/gdp_nowcast_artifacts.pkl",
)


# ============================================================
# 12. CURRENT-QUARTER GDP NOWCAST
# ============================================================

available_quarterly_features = (
    X_q
    .dropna(
        subset=model_features
    )
)

latest_quarterly_features = (
    available_quarterly_features
    .iloc[[-1]]
)

current_quarter_gdp_z_nowcast = (
    final_model.predict(
        latest_quarterly_features[
            model_features
        ]
    )[0]
)

nowcast_quarter = (
    latest_quarterly_features.index[0]
)

latest_known_gdp_quarter = (
    model_data.index.max()
)

print(
    "\nLatest known GDP quarter:",
    latest_known_gdp_quarter,
)

print(
    "Nowcast feature quarter:",
    nowcast_quarter,
)

print(
    "Current-quarter GDP z-score nowcast:",
    current_quarter_gdp_z_nowcast,
)


# ============================================================
# 13. ADD PREVIOUS GDP FEATURES TO MONTHLY MATRIX
# ============================================================

# Convert quarterly lagged GDP features to monthly frequency
gdp_features_monthly = (
    X_q[
        [
            "gdp_z_lag1",
            "gdp_z_diff_lag1",
        ]
    ]
    .resample("ME")
    .ffill()
)

# Normalize monthly index
macro_monthly.index = (
    pd.DatetimeIndex(
        macro_monthly.index
    )
    .to_period("M")
    .to_timestamp("M")
)

gdp_features_monthly.index = (
    pd.DatetimeIndex(
        gdp_features_monthly.index
    )
    .to_period("M")
    .to_timestamp("M")
)

# Remove GDP columns first in case this cell
# is executed multiple times
macro_monthly = macro_monthly.drop(
    columns=[
        "gdp_z_lag1",
        "gdp_z_diff_lag1",
    ],
    errors="ignore",
)

macro_monthly = (
    macro_monthly
    .join(
        gdp_features_monthly,
        how="left",
    )
    .ffill()
    .replace(
        [np.inf, -np.inf],
        np.nan,
    )
)

if not macro_monthly.columns.is_unique:
    raise ValueError(
        "Duplicate columns found in the "
        "monthly feature matrix."
    )


# ============================================================
# 14. MONTHLY GDP PROXY
# ============================================================

missing_monthly_features = [
    feature
    for feature in model_features
    if feature
    not in macro_monthly.columns
]

if missing_monthly_features:
    raise ValueError(
        "Monthly feature matrix is missing "
        f"these features: "
        f"{missing_monthly_features}"
    )

monthly_features = (
    macro_monthly[
        model_features
    ]
    .dropna()
)

monthly_gdp_proxy_z = pd.Series(
    final_model.predict(
        monthly_features
    ),
    index=monthly_features.index,
    name="monthly_gdp_proxy_z",
)

print(
    "\nMonthly proxy range:",
    monthly_gdp_proxy_z.index.min(),
    "to",
    monthly_gdp_proxy_z.index.max(),
)


# ============================================================
# 15. CONVERT MONTHLY Z-SCORE PROXY TO GDP YOY
# ============================================================

# Rolling quarterly mean and standard deviation used
# to reverse the GDP z-score
gdp_rolling_mean_q = (
    regime["gdp_yoy"]
    .rolling(ZSCORE_WINDOW)
    .mean()
)

gdp_rolling_std_q = (
    regime["gdp_yoy"]
    .rolling(ZSCORE_WINDOW)
    .std()
)

# Convert rolling GDP statistics to monthly frequency
gdp_rolling_mean_m = (
    gdp_rolling_mean_q
    .resample("ME")
    .ffill()
    .reindex(
        monthly_gdp_proxy_z.index,
        method="ffill",
    )
)

gdp_rolling_std_m = (
    gdp_rolling_std_q
    .resample("ME")
    .ffill()
    .reindex(
        monthly_gdp_proxy_z.index,
        method="ffill",
    )
)

monthly_gdp_proxy_yoy = (
    monthly_gdp_proxy_z
    * gdp_rolling_std_m
    + gdp_rolling_mean_m
)

monthly_gdp_proxy_yoy.name = (
    "monthly_gdp_proxy_yoy"
)

latest_proxy_z = (
    monthly_gdp_proxy_z.iloc[-1]
)

latest_gdp_yoy_nowcast = (
    monthly_gdp_proxy_yoy.iloc[-1]
)

latest_proxy_date = (
    monthly_gdp_proxy_yoy.index[-1]
)

print(
    "\nLatest monthly proxy date:",
    latest_proxy_date,
)

print(
    "Latest monthly GDP z-score nowcast:",
    latest_proxy_z,
)

print(
    "Latest monthly GDP YoY nowcast:",
    latest_gdp_yoy_nowcast,
)


# ============================================================
# 16. RIDGE FEATURE IMPORTANCE
# ============================================================

scaler = final_model.named_steps[
    "scaler"
]

ridge = final_model.named_steps[
    "ridge"
]

fitted_features = list(
    scaler.feature_names_in_
)

coefficients = np.asarray(
    ridge.coef_
).ravel()

print(
    "\nFitted feature count:",
    len(fitted_features),
)

print(
    "Coefficient count:",
    len(coefficients),
)

if len(fitted_features) != len(coefficients):
    raise ValueError(
        f"Feature mismatch: "
        f"{len(fitted_features)} names and "
        f"{len(coefficients)} coefficients."
    )

feature_importance = pd.DataFrame(
    {
        "feature": fitted_features,
        "coefficient": coefficients,
    }
)

feature_importance[
    "abs_importance"
] = (
    feature_importance[
        "coefficient"
    ].abs()
)

total_importance = (
    feature_importance[
        "abs_importance"
    ].sum()
)

feature_importance[
    "importance_pct"
] = (
    feature_importance[
        "abs_importance"
    ]
    / total_importance
    * 100
)

feature_importance = (
    feature_importance
    .sort_values(
        "abs_importance",
        ascending=False,
    )
    .reset_index(drop=True)
)

print(
    "\nTop GDP nowcast features:"
)

print(
    feature_importance[
        [
            "feature",
            "coefficient",
            "abs_importance",
            "importance_pct",
        ]
    ].head(TOP_N_FEATURES)
)


# ============================================================
# 17. CURRENT MONTHLY CONTRIBUTIONS
# ============================================================

latest_monthly_row = (
    monthly_features.iloc[[-1]]
)

latest_scaled_values = (
    scaler.transform(
        latest_monthly_row
    )[0]
)

latest_contributions = (
    latest_scaled_values
    * coefficients
)

current_contributions = pd.DataFrame(
    {
        "feature": fitted_features,
        "scaled_value": latest_scaled_values,
        "coefficient": coefficients,
        "contribution": latest_contributions,
    }
)

current_contributions[
    "abs_contribution"
] = (
    current_contributions[
        "contribution"
    ].abs()
)

current_contributions = (
    current_contributions
    .sort_values(
        "abs_contribution",
        ascending=False,
    )
    .reset_index(drop=True)
)

print(
    "\nLargest current GDP nowcast contributions:"
)

print(
    current_contributions[
        [
            "feature",
            "scaled_value",
            "coefficient",
            "contribution",
        ]
    ].head(TOP_N_FEATURES)
)

manual_latest_prediction = (
    latest_contributions.sum()
    + ridge.intercept_
)

pipeline_latest_prediction = (
    final_model.predict(
        latest_monthly_row
    )[0]
)

print(
    "\nManual prediction:",
    manual_latest_prediction,
)

print(
    "Pipeline prediction:",
    pipeline_latest_prediction,
)

assert np.isclose(
    manual_latest_prediction,
    pipeline_latest_prediction,
), (
    "Manual contribution calculation "
    "does not match pipeline prediction."
)


# ============================================================
# 18. OPTIONAL PLOTS
# ============================================================

plt.figure(
    figsize=(12, 6)
)

plt.plot(
    results.index,
    results["actual"],
    label="Actual GDP z-score",
)

plt.plot(
    results.index,
    results["prediction"],
    label="Model nowcast",
)

plt.plot(
    results.index,
    results["baseline"],
    label="Persistence baseline",
)

plt.title(
    "GDP Nowcast: Actual vs Model"
)

plt.legend()
plt.tight_layout()
plt.show()


top_features_plot = (
    feature_importance
    .head(20)
    .sort_values(
        "coefficient"
    )
)

plt.figure(
    figsize=(10, 8)
)

plt.barh(
    top_features_plot["feature"],
    top_features_plot["coefficient"],
)

plt.axvline(
    0,
    linewidth=1,
)

plt.title(
    "Top Standardized Ridge Coefficients"
)

plt.xlabel(
    "Ridge coefficient"
)

plt.tight_layout()
plt.show()


top_contributions_plot = (
    current_contributions
    .head(20)
    .sort_values(
        "contribution"
    )
)

plt.figure(
    figsize=(10, 8)
)

plt.barh(
    top_contributions_plot["feature"],
    top_contributions_plot[
        "contribution"
    ],
)

plt.axvline(
    0,
    linewidth=1,
)

plt.title(
    "Current Monthly GDP Nowcast Contributions"
)

plt.xlabel(
    "Contribution to GDP z-score nowcast"
)

plt.tight_layout()
plt.show()



# ============================================================
# 19. Refit for predictions
# ============================================================

best_features = (
    current_contributions
    .sort_values(
        "abs_contribution",
        ascending=False,
    )
    .head(20)["feature"]
    .tolist()
)

print(best_features)


reduced_model = Pipeline(
    [
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "ridge",
            Ridge(
                alpha=RIDGE_ALPHA,
            ),
        ),
    ]
)

reduced_model.fit(
    model_data[best_features],
    model_data["target_gdp_z"],
)

monthly_best_features = (
    macro_monthly[best_features]
    .replace(
        [np.inf, -np.inf],
        np.nan,
    )
    .dropna()
)

monthly_gdp_proxy_z = pd.Series(
    reduced_model.predict(
        monthly_best_features
    ),
    index=monthly_best_features.index,
    name="monthly_gdp_proxy_z",
)

scaler = reduced_model.named_steps["scaler"]
ridge = reduced_model.named_steps["ridge"]

monthly_scaled = scaler.transform(
    monthly_best_features
)

monthly_gdp_proxy_manual = pd.Series(
    monthly_scaled @ ridge.coef_
    + ridge.intercept_,
    index=monthly_best_features.index,
    name="monthly_gdp_proxy_z_manual",
)

assert np.allclose(
    monthly_gdp_proxy_z,
    monthly_gdp_proxy_manual,
)

gdp_mean_q = (
    regime["gdp_yoy"]
    .rolling(60)
    .mean()
)

gdp_std_q = (
    regime["gdp_yoy"]
    .rolling(60)
    .std()
)

gdp_mean_monthly = (
    gdp_mean_q
    .resample("ME")
    .ffill()
    .reindex(
        monthly_gdp_proxy_z.index,
        method="ffill",
    )
)

gdp_std_monthly = (
    gdp_std_q
    .resample("ME")
    .ffill()
    .reindex(
        monthly_gdp_proxy_z.index,
        method="ffill",
    )
)

monthly_gdp_proxy_yoy = (
    monthly_gdp_proxy_z
    * gdp_std_monthly
    + gdp_mean_monthly
)

monthly_gdp_proxy_yoy.name = (
    "monthly_gdp_proxy_yoy"
)


monthly_nowcast = pd.concat(
    [
        monthly_gdp_proxy_z,
        monthly_gdp_proxy_yoy,
    ],
    axis=1,
)

monthly_nowcast = monthly_nowcast.dropna()
monthly_nowcast.to_pickle("ml_models/monthly_gdp_nowcast.pkl")

