import joblib
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

from feature_engineering import create_best_lagged_features

from functions import *


# ============================================================
# 1. SETTINGS
# ============================================================

RIDGE_ALPHA = 10.0
MIN_TRAIN_SIZE = 60
MAX_LAG = 12
TOP_N_FEATURES = 15


# ============================================================
# 2. LOAD DATA
# ============================================================

inflation_data = (
    pd.read_pickle("data/inflation_factors.pkl")
    .sort_index()
    .copy()
)

inflation_data.index = pd.DatetimeIndex(
    inflation_data.index
)

inflation_data = inflation_data.replace(
    [np.inf, -np.inf],
    np.nan,
)

if "cpi_yoy" not in inflation_data.columns:
    raise ValueError(
        "'cpi_yoy' is missing from inflation_factors.pkl"
    )

money_data = (
    pd.read_pickle("data/raw_money_market_data.pkl")
    .sort_index()
    .copy()
).drop('cpi', axis=1)

money_z = zscore(money_data, window=60).add_suffix("_z").dropna()
money_momentum = momentum(
    money_data,
    fast_window=3,
    slow_window=12,
    signal_window=6,
    zscore_window=60
).add_suffix("_roc").dropna()

macro_data = pd.read_pickle("data/raw_macro_data.pkl").sort_index().copy()
pce_yoy = macro_data['pce'].pct_change(12).rename('pce_yoy')
pce_yoy_z = zscore(pce_yoy, window=60).rename('pce_yoy_z').dropna()
pce_yoy_momentum = momentum(
    pce_yoy,
    fast_window=3,
    slow_window=12,
    signal_window=6,
    zscore_window=60
).rename('pce_yoy_roc').dropna()

manu_prices = (
        macro_data["manu_prices"]
        .rename("manu_prices")
    )

manu_prices_z = (
    zscore(
        manu_prices,
        window=60,
    )
    .rename("manu_prices_z")
)

manu_prices_momentum = (
    momentum(
        manu_prices,
        fast_window=3,
        slow_window=12,
        signal_window=6,
        zscore_window=60,
    )
    .rename("manu_prices_roc")
)

inflation_data = pd.concat(
    [inflation_data, money_z, money_momentum, pce_yoy_z, pce_yoy_momentum, manu_prices_z, manu_prices_momentum],
    axis=1,
).dropna()

# ============================================================
# 3. CREATE CPI TARGET
# ============================================================

cpi_yoy = inflation_data["cpi_yoy"].copy()

# Full-sample z-score, matching your current approach
cpi_mean = cpi_yoy.mean()
cpi_std = cpi_yoy.std()

cpi_z = (
    (cpi_yoy - cpi_mean)
    / cpi_std
).rename("target_cpi_z")

# Remove the target from predictors
inflation_features = inflation_data.drop(
    columns=["cpi_yoy"]
)


# ============================================================
# 4. CREATE BEST-LAG FEATURES
# ============================================================

inflation_best_lag = create_best_lagged_features(
    inflation_features,
    cpi_z,
    max_lag=MAX_LAG,
)

inflation_best_lag = (
    inflation_best_lag
    .add_prefix("inflation__")
    .sort_index()
    .replace([np.inf, -np.inf], np.nan)
)

if not inflation_best_lag.columns.is_unique:
    duplicate_columns = (
        inflation_best_lag.columns[
            inflation_best_lag.columns.duplicated(
                keep=False
            )
        ]
        .unique()
        .tolist()
    )

    raise ValueError(
        f"Duplicate feature names: {duplicate_columns}"
    )

print("Selected lagged features:")
print(inflation_best_lag.columns.tolist())


# ============================================================
# 5. ADD CPI PERSISTENCE FEATURES
# ============================================================

X_monthly = inflation_best_lag.copy()

# Previous month's CPI YoY z-score
X_monthly["cpi_z_lag2"] = cpi_z.shift(2)

# Previous month's change in CPI z-score
X_monthly["cpi_z_diff_lag2"] = (
    cpi_z.diff().shift(2)
)

model_features = X_monthly.columns.tolist()

print(
    "\nMonthly feature matrix:",
    X_monthly.shape,
)

print(
    "Number of model features:",
    len(model_features),
)


# ============================================================
# 6. JOIN CURRENT-MONTH CPI TARGET
# ============================================================

model_frame = X_monthly.join(
    cpi_z,
    how="left",
)

model_data = model_frame.dropna(
    subset=model_features + ["target_cpi_z"]
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
# 7. EXPANDING-WINDOW CPI NOWCAST
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
        train["target_cpi_z"],
    )

    prediction = model.predict(
        test[model_features]
    )[0]

    # Persistence baseline:
    # current CPI z-score equals previous month's CPI z-score
    baseline = test["cpi_z_lag2"].iloc[0]

    actual = test["target_cpi_z"].iloc[0]

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
# 8. METRICS
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
        "Model correlation": results["actual"].corr(
            results["prediction"]
        ),
        "Baseline correlation": results["actual"].corr(
            results["baseline"]
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
# 9. FIT FINAL CPI MODEL
# ============================================================

final_cpi_model = Pipeline(
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

final_cpi_model.fit(
    model_data[model_features],
    model_data["target_cpi_z"],
)


# ============================================================
# 10. GENERATE LATEST CPI NOWCAST
# ============================================================

available_features = X_monthly.dropna(
    subset=model_features
)

latest_features = available_features.iloc[[-1]]

latest_cpi_z_nowcast = final_cpi_model.predict(
    latest_features[model_features]
)[0]

latest_nowcast_date = latest_features.index[0]

latest_cpi_yoy_nowcast = (
    latest_cpi_z_nowcast * cpi_std
    + cpi_mean
)

print(
    "\nLatest CPI nowcast date:",
    latest_nowcast_date,
)

print(
    "Latest CPI z-score nowcast:",
    latest_cpi_z_nowcast,
)

print(
    "Latest CPI YoY nowcast:",
    latest_cpi_yoy_nowcast,
)

if abs(latest_cpi_yoy_nowcast) < 1:
    print(
        "Latest CPI YoY nowcast formatted:",
        f"{latest_cpi_yoy_nowcast * 100:.2f}%",
    )
else:
    print(
        "Latest CPI YoY nowcast formatted:",
        f"{latest_cpi_yoy_nowcast:.2f}%",
    )


# ============================================================
# 11. FEATURE IMPORTANCE
# ============================================================

scaler = final_cpi_model.named_steps["scaler"]
ridge = final_cpi_model.named_steps["ridge"]

fitted_features = list(
    scaler.feature_names_in_
)

coefficients = np.asarray(
    ridge.coef_
).ravel()

if len(fitted_features) != len(coefficients):
    raise ValueError(
        f"{len(fitted_features)} feature names but "
        f"{len(coefficients)} coefficients."
    )

feature_importance = pd.DataFrame(
    {
        "feature": fitted_features,
        "coefficient": coefficients,
    }
)

feature_importance["abs_importance"] = (
    feature_importance["coefficient"].abs()
)

feature_importance["importance_pct"] = (
    feature_importance["abs_importance"]
    / feature_importance["abs_importance"].sum()
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

print("\nTop CPI nowcast features:")
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
# 12. LATEST FEATURE CONTRIBUTIONS
# ============================================================

latest_scaled_values = scaler.transform(
    latest_features[model_features]
)[0]

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

current_contributions["abs_contribution"] = (
    current_contributions["contribution"].abs()
)

current_contributions = (
    current_contributions
    .sort_values(
        "abs_contribution",
        ascending=False,
    )
    .reset_index(drop=True)
)

print("\nLargest current CPI contributions:")
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

manual_prediction = (
    current_contributions["contribution"].sum()
    + ridge.intercept_
)

pipeline_prediction = final_cpi_model.predict(
    latest_features[model_features]
)[0]

print(
    "\nManual prediction:",
    manual_prediction,
)

print(
    "Pipeline prediction:",
    pipeline_prediction,
)

assert np.isclose(
    manual_prediction,
    pipeline_prediction,
)


# ============================================================
# 13. SAVE MODEL ARTIFACT
# ============================================================

cpi_artifact = {
    "model": final_cpi_model,
    "features": model_features,
    "target_name": "cpi_yoy",
    "target_mean": cpi_mean,
    "target_std": cpi_std,
    "ridge_alpha": RIDGE_ALPHA,
    "max_lag": MAX_LAG,
    "feature_importance": feature_importance,
}

joblib.dump(
    cpi_artifact,
    "ml_models/cpi_nowcast_artifact.pkl",
)

print(
    "\nSaved model to:"
    " ml_models/cpi_nowcast_artifact.pkl"
)


# ============================================================
# 14. PLOTS
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    results.index,
    results["actual"],
    label="Actual CPI z-score",
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
    "CPI Nowcast: Actual vs Model"
)

plt.legend()
plt.tight_layout()
plt.show()


top_features_plot = (
    feature_importance
    .head(TOP_N_FEATURES)
    .sort_values("coefficient")
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features_plot["feature"],
    top_features_plot["coefficient"],
)

plt.axvline(0, linewidth=1)

plt.title(
    "Top Standardized CPI Ridge Coefficients"
)

plt.xlabel(
    "Ridge coefficient"
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
    model_data["target_cpi_z"],
)

monthly_best_features = (
    model_data[best_features]
    .replace(
        [np.inf, -np.inf],
        np.nan,
    )
    .dropna()
)

monthly_cpi_proxy_z = pd.Series(
    reduced_model.predict(
        monthly_best_features
    ),
    index=monthly_best_features.index,
    name="monthly_cpi_proxy_z",
)

scaler = reduced_model.named_steps["scaler"]
ridge = reduced_model.named_steps["ridge"]

monthly_scaled = scaler.transform(
    monthly_best_features
)

monthly_cpi_proxy_manual = pd.Series(
    monthly_scaled @ ridge.coef_
    + ridge.intercept_,
    index=monthly_best_features.index,
    name="monthly_cpi_proxy_z_manual",
)

assert np.allclose(
    monthly_cpi_proxy_z,
    monthly_cpi_proxy_manual,
)

cpi_mean_q = (
    cpi_yoy
    .rolling(60)
    .mean()
)

cpi_std_q = (
    cpi_yoy
    .rolling(60)
    .std()
)

cpi_mean_monthly = (
    cpi_mean_q
    .resample("ME")
    .ffill()
    .reindex(
        monthly_cpi_proxy_z.index,
        method="ffill",
    )
)

cpi_std_monthly = (
    cpi_std_q
    .resample("ME")
    .ffill()
    .reindex(
        monthly_cpi_proxy_z.index,
        method="ffill",
    )
)

monthly_cpi_proxy_yoy = (
    monthly_cpi_proxy_z
    * cpi_std_monthly
    + cpi_mean_monthly
)

monthly_cpi_proxy_yoy.name = (
    "monthly_cpi_proxy_yoy"
)


monthly_nowcast = pd.concat(
    [
        monthly_cpi_proxy_z,
        monthly_cpi_proxy_yoy,
    ],
    axis=1,
)
results

monthly_nowcast = monthly_nowcast.dropna()
monthly_nowcast.to_pickle("ml_models/monthly_cpi_nowcast.pkl")