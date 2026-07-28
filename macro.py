from matplotlib.pylab import beta, where
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from fredapi import Fred
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from functions import *
from feature_engineering import *
from ffloader import FamaFrenchLoader

fred = Fred(api_key='a2fb338b4ef6e2dcb7c667c21b2d1c4e')  # Replace with your actual FRED API key

def get_housing_data():


    df = pd.DataFrame({
        "mortgage_rate": fred.get_series("MORTGAGE30US").resample("ME").last().ffill() / 100,
        "nominal_wage": avoid_future_leakage(fred.get_series("LES1252881500Q"), offset_months=6).interpolate() * 4,
        "home_price": avoid_future_leakage(fred.get_series("MSPUS"), offset_months=6).interpolate(),
        "building_permits": avoid_future_leakage(fred.get_series("PERMIT"), offset_months=2).interpolate(),
        "housing_starts": avoid_future_leakage(fred.get_series("HOUST"), offset_months=2).interpolate(),
        "house_supply": avoid_future_leakage(fred.get_series("MSACSR"), offset_months=2).interpolate(),
        # "shiller_index": fred.get_series("CSUSHPINSA").dropna(),
        "cpi": avoid_future_leakage(fred.get_series("CPIAUCSL"), offset_months=2).interpolate(),
    })

    df = df.resample("ME").last().ffill()

    r = df["mortgage_rate"] / 12
    n = 30 * 12
    P = 0.8 * df["home_price"]

    df["monthly_payment"] = (
        P * (r * (1 + r) ** n)
        / ((1 + r) ** n - 1)
    )

    df["mortgage_burden"] = df["monthly_payment"] / df["nominal_wage"]
    df["home_price_to_wage"] = df["home_price"] / df["nominal_wage"]

    df["cpi_yoy"] = df["cpi"].pct_change(12)
    df["real_mortgage_rate"] = df["mortgage_rate"] - df["cpi_yoy"]

    # df["shiller_yoy"] = df["shiller_index"].pct_change(12)
    # df["shiller_accel"] = df["shiller_yoy"].diff(3)

    df["building_permits_yoy"] = df["building_permits"].pct_change(12)
    df["housing_starts_yoy"] = df["housing_starts"].pct_change(12)
    df["housing_supply_yoy"] = df["house_supply"].pct_change(12)

    df["mortgage_rate_change_12m"] = df["mortgage_rate"].diff(12)

    return df[
        [
            "mortgage_rate",
            "real_mortgage_rate",
            "mortgage_rate_change_12m",
            "monthly_payment",
            "mortgage_burden",
            "home_price_to_wage",
            "building_permits",
            "building_permits_yoy",
            "housing_starts",
            "housing_starts_yoy",
            "house_supply",
            "housing_supply_yoy",
            # "shiller_index",
            # "shiller_yoy",
            # "shiller_accel",
        ]
    ]



def compute_housing_features():

    df = get_housing_data()

    features = pd.DataFrame(index=df.index)

    features["permits_minus_starts_z"] = (
        zscore(df["building_permits"], window=60)
        - zscore(df["housing_starts"], window=60)
    )

    features["affordability_z"] = zscore(df["mortgage_burden"], window=60)

    zs_df = zscore(df, window=60).add_suffix("_z")

    momentum_df = momentum(
        df.add_suffix("_roc"),
        fast_window=3,
        slow_window=12,
        signal_window=6,
        zscore_window=60
    )

    accel_df = df.diff(3).add_suffix("_3m_accel")

    trend_df = pd.concat(
        {
            f"{col}_6m_mean": df[col].rolling(6).mean()
            for col in df.columns
        },
        axis=1
    )

    vol_df = pd.concat(
        {
            f"{col}_12m_vol": df[col].pct_change().rolling(12).std()
            for col in df.columns
        },
        axis=1
    )

    housing_features = pd.concat(
        [
            df,
            features,
            zs_df,
            momentum_df,
            accel_df,
            trend_df,
            vol_df,
        ],
        axis=1
    )

    return housing_features.dropna()



def compute_housing_pca():
    """
    Compute latent housing factors using Principal Component Analysis (PCA).

    Returns
    -------
    pd.DataFrame
        Housing latent factors:
            - housing_cycle
            - financing_conditions
            - housing_momentum
    """

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    
    df = compute_housing_features()
    
    scaler = StandardScaler()
    X = scaler.fit_transform(df)

    pca = PCA(n_components=0.75)
    pcs = pca.fit_transform(X)

    # ------------------------------------------------------------------
    # Flip PC1 so higher values = stronger housing market
    # ------------------------------------------------------------------

    pcs[:, 0] *= -1
    pca.components_[0] *= -1

    # ------------------------------------------------------------------
    # Keep first three interpretable factors
    # ------------------------------------------------------------------

    factors = pd.DataFrame(
        pcs[:, :3],
        index=df.index,
        columns=[
            "housing_cycle_pca",
            "financing_conditions_pca",
            "housing_momentum_pca",
        ],
    )

    return factors.dropna()


def get_commodities_prices():
    """
    Fetch commodities prices from FRED and local CSV files,
    resample to monthly frequency, and drop missing values.

    Returns
    -------
    pd.DataFrame
        Combined commodities prices.
    """

    # ------------------------------------------------------------------
    # Load commodities prices from FRED
    # ------------------------------------------------------------------


    df = pd.DataFrame({
        "crude_oil": fred.get_series("DCOILWTICO"),
        "brent_oil": fred.get_series("DCOILBRENTEU"),
        "heating_oil": fred.get_series("DHOILNYH"),
        "gold":  pd.read_csv("macro_data/commodities/gold_prices.csv", index_col=0, parse_dates=True).squeeze().rename("gold"),
        "copper": pd.read_csv("macro_data/commodities/copper_prices.csv", index_col=0, parse_dates=True).squeeze().rename("copper"),
        "lumber": pd.read_csv("macro_data/commodities/lumber_prices.csv", index_col=0, parse_dates=True).squeeze().rename("lumber"),
        "wheat": pd.read_csv("macro_data/commodities/wheat_prices.csv", index_col=0, parse_dates=True).squeeze().rename("wheat"),
        "corn": pd.read_csv("macro_data/commodities/corn_prices.csv", index_col=0, parse_dates=True).squeeze().rename("corn"),
        "coffee": pd.read_csv("macro_data/commodities/coffee_prices.csv", index_col=0, parse_dates=True).squeeze().rename("coffee"),
        "cotton": pd.read_csv("macro_data/commodities/cotton_prices.csv", index_col=0, parse_dates=True).squeeze().rename("cotton"),
        "commodity_index": yf.download('^SPGSCI', start='1900-01-01')['Close'].squeeze(),
        })

    df = (
        df
        .resample("ME")
        .last()
        .dropna()
    )

    return df


def compute_inflation_factors():

    df = get_commodities_prices()

    returns_12m = df.pct_change(12)
    returns_3m = df.pct_change(3)

    energy_cols = ["crude_oil", "brent_oil", "heating_oil"]
    food_cols = ["wheat", "corn", "coffee", "cotton"]

    energy_factors = run_single_factor_pca(
        df[energy_cols],
        factor_name="energy",
        n_components=1
    )

    food_factors = run_single_factor_pca(
        df[food_cols],
        factor_name="food",
        n_components=1
    )

    commodity_factors = run_single_factor_pca(
        df[["commodity_index"]],
        factor_name="commodity",
        n_components=1
    )

    breadth = pd.DataFrame(index=df.index)

    breadth["energy_breadth_12m"] = (returns_12m[energy_cols] > 0).mean(axis=1)
    breadth["food_breadth_12m"] = (returns_12m[food_cols] > 0).mean(axis=1)
    breadth["commodity_breadth_12m"] = (returns_12m > 0).mean(axis=1)

    breadth["energy_accel"] = returns_3m[energy_cols].mean(axis=1).diff(3)
    breadth["food_accel"] = returns_3m[food_cols].mean(axis=1).diff(3)

    breadth["commodity_vol_12m"] = df.pct_change().rolling(12).std().mean(axis=1)

    inflation_factors = pd.concat(
        [
            energy_factors,
            food_factors,
            commodity_factors,
            breadth,
            copper_to_gold_ratio().rename("copper_to_gold_ratio"),
        ],
        axis=1
    ).dropna()

    return inflation_factors


def copper_to_gold_ratio():
    """
    Calculate the copper to gold ratio using price data.

    Returns
    -------
    pd.Series
        Copper to gold ratio.
    """

    copper = (pd.read_csv("macro_data/commodities/copper_prices.csv", index_col=0, parse_dates=True)
              .resample("ME").last().dropna().squeeze()).pct_change(12).dropna()
    gold = (pd.read_csv("macro_data/commodities/gold_prices.csv", index_col=0, parse_dates=True)
            .resample("ME").last().dropna().squeeze()).pct_change(12).dropna()
    
    ratio = zscore(copper, window=60) - zscore(gold, window=60)
    
    return ratio.resample("ME").last().dropna()


def get_raw_macroeconomic_data():
    """
    Fetch raw monthly macroeconomic data.
    No feature engineering here.
    """

    ism = (
        pd.read_csv("macro_data/leading_macro/ism_data.csv", index_col=0, parse_dates=True)
        .resample("ME")
        .last()
    )

    spx_eps = (
        pd.read_csv("macro_data/markets/spx_eps.csv", index_col=0, parse_dates=True)
        .resample("ME")
        .last()
        .ffill()
        .squeeze()
        .rename("spy_eps")
    )

    df = pd.DataFrame({
        # Liquidity / money
        # "retail_money_market_funds": fred.get_series("WRMFNS").resample("ME").last().ffill().squeeze(),

        # Housing / construction
        "building_permits": avoid_future_leakage(fred.get_series("PERMIT"), offset_months=2),
        
        # Consumption
        "pce": avoid_future_leakage(fred.get_series("PCE"), offset_months=2),
        # "retail_sales": fred.get_series("RSAFS"),
        "personal_disposable_income": avoid_future_leakage(fred.get_series("DSPI"), offset_months=2),
        # "auto_sales": avoid_future_leakage(fred.get_series("TOTALSA"), offset_months=1),
        "consumer_sentiment": avoid_future_leakage(fred.get_series("UMCSENT"), offset_months=2),

        # Labor
        "payrolls": avoid_future_leakage(fred.get_series("PAYEMS"), offset_months=1),
        "manufacturing_hours": avoid_future_leakage(fred.get_series("AWHMAN"), offset_months=1),
        # "initial_jobless_claims": fred.get_series("ICSA").resample("ME").last().ffill().squeeze(),

        # Production / business cycle
        "industrial_production": avoid_future_leakage(fred.get_series("INDPRO"), offset_months=2),
        "capacity_utilization": avoid_future_leakage(fred.get_series("TCU"), offset_months=2),
        # "inventory_cycle": fred.get_series("ISRATIO"),
        # "manufacturers_new_orders": fred.get_series("AMTMNO"),
        # "business_inventories": fred.get_series("BUSINV").dropna(),

        # Financial / market
        "financial_conditions_index": fred.get_series("NFCI").resample("ME").last().ffill().squeeze(),
        "spy_eps": spx_eps,
        "copper_to_gold_ratio": copper_to_gold_ratio(),
    })

    df = (
        df
        .interpolate()
        .resample("ME")
        .last()
        .ffill()
    )

    df = pd.concat([df, ism], axis=1)

    return df.dropna()

def get_raw_macroeconomic_features():
    """
    Fetch raw monthly macroeconomic data.
    Little feature engineering here.
    """

    building = pd.DataFrame()

    building.loc[:, "building_permits"] = zscore(avoid_future_leakage(fred.get_series("PERMIT"), offset_months=2), window=60)
    building.loc[:, "building_permits_yoy"] = zscore(building.loc[:, "building_permits"].pct_change(12), window=60)
    building.loc[:, "building_permits_yoy_roc"] = momentum(building.loc[:, "building_permits"], fast_window=3, slow_window=12, signal_window=6, zscore_window=60)
    building.loc[:, "building_permits_rank"] = building.loc[:, "building_permits"].rank(pct=True)

    consumption = pd.DataFrame()

    consumption.loc[:, "consumer_sentiment"] = avoid_future_leakage(fred.get_series("UMCSENT"), offset_months=2).interpolate()
    consumption.loc[:, "consumer_sentiment_roc"] = momentum(consumption.loc[:, "consumer_sentiment"], fast_window=3, slow_window=12, signal_window=6, zscore_window=60)
    consumption.loc[:, "consumer_sentiment_rank"] = consumption.loc[:, "consumer_sentiment"].rank(pct=True)

    ism = (
        pd.read_csv("macro_data/leading_macro/ism_data.csv", index_col=0, parse_dates=True)
        .resample("ME")
        .last()
    )

    spx_eps = (
        pd.read_csv("macro_data/markets/spx_eps.csv", index_col=0, parse_dates=True)
        .resample("ME")
        .last()
        .ffill()
        .squeeze()
        .rename("spy_eps")
    )

    df = pd.DataFrame({
        # Liquidity / money
        # "retail_money_market_funds": fred.get_series("WRMFNS").resample("ME").last().ffill().squeeze(),

        # Housing / construction
        "building_permits": avoid_future_leakage(fred.get_series("PERMIT"), offset_months=2),
        
        # Consumption
        "pce": avoid_future_leakage(fred.get_series("PCE"), offset_months=2),
        # "retail_sales": fred.get_series("RSAFS"),
        "personal_disposable_income": avoid_future_leakage(fred.get_series("DSPI"), offset_months=2),
        # "auto_sales": avoid_future_leakage(fred.get_series("TOTALSA"), offset_months=1),
        "consumer_sentiment": avoid_future_leakage(fred.get_series("UMCSENT"), offset_months=2),

        # Labor
        "payrolls": avoid_future_leakage(fred.get_series("PAYEMS"), offset_months=1),
        "manufacturing_hours": avoid_future_leakage(fred.get_series("AWHMAN"), offset_months=1),
        # "initial_jobless_claims": fred.get_series("ICSA").resample("ME").last().ffill().squeeze(),

        # Production / business cycle
        "industrial_production": avoid_future_leakage(fred.get_series("INDPRO"), offset_months=2),
        "capacity_utilization": avoid_future_leakage(fred.get_series("TCU"), offset_months=2),
        # "inventory_cycle": fred.get_series("ISRATIO"),
        # "manufacturers_new_orders": fred.get_series("AMTMNO"),
        # "business_inventories": fred.get_series("BUSINV").dropna(),

        # Financial / market
        "financial_conditions_index": fred.get_series("NFCI").resample("ME").last().ffill().squeeze(),
        "spy_eps": spx_eps,
        "copper_to_gold_ratio": copper_to_gold_ratio(),
    })

    df = (
        df
        .interpolate()
        .resample("ME")
        .last()
        .ffill()
    )

    df = pd.concat([df, ism], axis=1)

    return df.dropna()



def engineer_macroeconomic_features(
    df,
    z_window=60,
    dynamic_threshold_window=None,
    min_threshold_periods=24,
):
    """
    Transform raw monthly macroeconomic data into ML-ready features.

    Parameters
    ----------
    df : pd.DataFrame
        Raw macroeconomic dataset with a DatetimeIndex.

    z_window : int, default=60
        Rolling window used to calculate z-scores.

    dynamic_threshold_window : int or None, default=None
        Window used for dynamic thresholds.

        If None, thresholds are calculated using an expanding historical
        median. If an integer is provided, a rolling median is used.

    min_threshold_periods : int, default=24
        Minimum number of historical observations required before calculating
        a dynamic threshold.

    Returns
    -------
    pd.DataFrame
        Engineered macroeconomic features.
    """

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df must have a DatetimeIndex.")

    df = df.sort_index().copy()

    features = pd.DataFrame(index=df.index)

    # --------------------------------------------------
    # 1. YoY growth features
    # --------------------------------------------------

    yoy_cols = [
        "building_permits",
        "pce",
        "retail_sales",
        "personal_disposable_income",
        "payrolls",
        "industrial_production",
        "manufacturers_new_orders",
        "business_inventories",
        "spy_eps",
    ]

    for col in yoy_cols:
        if col not in df.columns:
            continue

        series = pd.to_numeric(df[col], errors="coerce")

        features[f"{col}_yoy"] = (
            series
            .pct_change(
                periods=12,
                fill_method=None,
            )
            .replace([np.inf, -np.inf], np.nan)
        )

    # --------------------------------------------------
    # 2. Level / ratio features
    # --------------------------------------------------

    level_cols = [
        "mortgage_burden",
        "consumer_sentiment",
        "manufacturing_hours",
        "capacity_utilization",
        "inventory_cycle",
        "financial_conditions_index",
        "copper_to_gold_ratio",
    ]

    for col in level_cols:
        if col in df.columns:
            features[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

    # Add ISM columns as level variables
    ism_cols = [
        col
        for col in df.columns
        if "ism" in col.lower()
    ]

    for col in ism_cols:
        features[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    # --------------------------------------------------
    # 3. Rolling z-scores
    # --------------------------------------------------

    z_features = (
        zscore(
            features,
            window=z_window,
        )
        .add_suffix("_z")
    )

    # --------------------------------------------------
    # 4. Momentum features
    # --------------------------------------------------

    momentum_features = momentum(
        features,
        fast_window=3,
        slow_window=12,
        signal_window=6,
        zscore_window=z_window,
    )

    # Rename only after momentum has been calculated.
    momentum_features = momentum_features.add_suffix("_momentum")

    macro_features = pd.concat(
        [
            features,
            z_features,
            momentum_features,
        ],
        axis=1,
    )

    # --------------------------------------------------
    # 5. Threshold persistence features
    # --------------------------------------------------

    # Fixed thresholds with a genuine economic meaning.
    fixed_thresholds = {
        "capacity_utilization": 80.0,
    }

    # Financial conditions can use zero only if the index was explicitly
    # constructed so that:
    #   > 0 = tighter than normal
    #   < 0 = easier than normal
    zero_centered_features = {
        "financial_conditions_index",
    }

    # Variables with no universal economic threshold.
    dynamic_threshold_features = {
        "consumer_sentiment",
        "mortgage_burden",
        "manufacturing_hours",
        "inventory_cycle",
        "copper_to_gold_ratio",
    }

    # ISM has a structural expansion/contraction threshold of 50.
    for col in features.columns:
        if "ism" in col.lower():
            fixed_thresholds[col] = 50.0

    def historical_threshold(series):
        """
        Calculate a threshold using historical information only.

        The threshold is shifted by one month so that the observation at time t
        is compared with a threshold estimated through t-1.
        """

        if dynamic_threshold_window is None:
            threshold = (
                series
                .expanding(
                    min_periods=min_threshold_periods,
                )
                .median()
            )
        else:
            threshold = (
                series
                .rolling(
                    window=dynamic_threshold_window,
                    min_periods=min_threshold_periods,
                )
                .median()
            )

        return threshold.shift(1)

    threshold_series = {}

    # Fixed thresholds
    for col, threshold in fixed_thresholds.items():
        if col in features.columns:
            threshold_series[col] = pd.Series(
                threshold,
                index=features.index,
                dtype=float,
            )

    # Zero-centered thresholds
    for col in zero_centered_features:
        if col in features.columns:
            threshold_series[col] = pd.Series(
                0.0,
                index=features.index,
                dtype=float,
            )

    # Dynamic historical thresholds
    for col in dynamic_threshold_features:
        if col in features.columns:
            threshold_series[col] = historical_threshold(
                features[col]
            )

    # --------------------------------------------------
    # 6. Consecutive threshold and trend features
    # --------------------------------------------------

    for col, threshold in threshold_series.items():

        valid = (
            features[col].notna()
            & threshold.notna()
        )

        above = (features[col] > threshold).where(valid, False)
        below = (features[col] < threshold).where(valid, False)

        difference = features[col].diff()

        uptrend = (difference > 0).where(difference.notna(), False)
        downtrend = (difference < 0).where(difference.notna(), False)

        macro_features[f"{col}_months_above"] = (
            consecutive_true(above)
        )

        macro_features[f"{col}_months_below"] = (
            consecutive_true(below)
        )

        macro_features[f"{col}_uptrend"] = (
            consecutive_true(uptrend)
        )

        macro_features[f"{col}_downtrend"] = (
            consecutive_true(downtrend)
        )

        # Optional but useful for debugging and interpretation.
        macro_features[f"{col}_threshold"] = threshold

    return (
        macro_features
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        )

def get_macroeconomic_data():
    """
    Fetch raw macro data and return engineered ML-ready macro features.
    """

    raw_macro = get_raw_macroeconomic_data()

    macro_features = engineer_macroeconomic_features(
        raw_macro,
        z_window=60
    )

    return macro_features.dropna()


def compute_yield_curve_data():
    """
    Fetch yield curve data from FRED, resample to monthly frequency, and drop missing values.

    """
    df = pd.DataFrame({
        "fed_funds_rate": fred.get_series("FEDFUNDS"),
        "1_mo_yield": fred.get_series("DGS1MO"),
        "3_mo_yield": fred.get_series("DGS3MO"),
        "6_mo_yield": fred.get_series("DGS6MO"),
        "1_yr_yield": fred.get_series("DGS1"),
        "2_yr_yield": fred.get_series("DGS2"),
        "3_yr_yield": fred.get_series("DGS3"),
        "5_yr_yield": fred.get_series("DGS5"),
        "7_yr_yield": fred.get_series("DGS7"),
        "10_yr_yield": fred.get_series("DGS10"),
        "20_yr_yield": fred.get_series("DGS20"),
        "30_yr_yield": fred.get_series("DGS30"),
    })

    df = (
        df
        .interpolate()
        .resample("ME")
        .last()
        .dropna()
    )

    return df


def get_money_market_data():

    raw = pd.DataFrame({
        "10-2_yr_yield_spread": fred.get_series("T10Y2Y"),
        # "10-3_mo_yield_spread": fred.get_series("T10Y3M"),
        # "10y-baa_spread": fred.get_series("BAA10Y"),
        "10y_yield": fred.get_series("DGS10"),
        # "2y_yield": fred.get_series("DGS2"),
        # "3m_yield": fred.get_series("DGS3MO"),
        "fed_funds_rate": fred.get_series("FEDFUNDS"),
        "loans": fred.get_series("BUSLOANS"),
        "M2_money_supply": fred.get_series("M2SL"),
        "financial_conditions_index": fred.get_series("NFCI"),
        # "retail_money_market_funds": fred.get_series("WRMFNS"),
        "cpi": fred.get_series("CPIAUCSL"),
    })

    raw = raw.resample("ME").last().ffill().dropna()

    return raw


def compute_money_markets_factors():

    raw = pd.DataFrame({
        "10-2_yr_yield_spread": fred.get_series("T10Y2Y"),
        # "10-3_mo_yield_spread": fred.get_series("T10Y3M"),
        # "10y-baa_spread": fred.get_series("BAA10Y"),
        "10y_yield": fred.get_series("DGS10"),
        # "2y_yield": fred.get_series("DGS2"),
        # "3m_yield": fred.get_series("DGS3MO"),
        "fed_funds_rate": fred.get_series("FEDFUNDS"),
        "loans": fred.get_series("BUSLOANS"),
        "M2_money_supply": fred.get_series("M2SL"),
        "financial_conditions_index": fred.get_series("NFCI"),
        # "retail_money_market_funds": fred.get_series("WRMFNS"),
        "cpi": fred.get_series("CPIAUCSL"),
    })

    raw = raw.resample("ME").last().ffill().dropna()

    df = pd.DataFrame(index=raw.index)

    # df["yield_curve_slope"] = raw["10y_yield"] - raw["3m_yield"]
    # df["yield_curve_10_2"] = raw["10y_yield"] - raw["2y_yield"]
    # df["credit_spread"] = raw["10y-baa_spread"]

    # df["credit_spread_change_6m"] = raw["10y-baa_spread"].diff(6)
    df["financial_conditions_change_6m"] = raw["financial_conditions_index"].diff(6)

    df["loans_yoy"] = raw["loans"].pct_change(12)
    df["loans_accel"] = df["loans_yoy"].diff(3)

    df["m2_yoy"] = raw["M2_money_supply"].pct_change(12)
    df["m2_accel"] = df["m2_yoy"].diff(3)

    # df["money_market_funds_yoy"] = raw["retail_money_market_funds"].pct_change(12)
    # df["money_market_funds_accel"] = df["money_market_funds_yoy"].diff(3)

    df["cpi_yoy"] = raw["cpi"].pct_change(12)
    df["real_10y_yield"] = raw["10y_yield"] / 100 - df["cpi_yoy"]
    df["real_fed_funds"] = raw["fed_funds_rate"] / 100 - df["cpi_yoy"]

    z_df = zscore(df, window=60).add_suffix("_z")

    momentum_df = momentum(
        df.add_suffix("_roc"),
        fast_window=3,
        slow_window=12,
        signal_window=6,
        zscore_window=60
    )

    money_features = pd.concat([df, z_df, momentum_df], axis=1).dropna()

    return money_features


def get_market_factors():


    # small aggressive - big conservative
    # small growth - big value
    # small high beta - big low beta

    loader = FamaFrenchLoader()

    ff5 = (1+loader.load("ff5")).cumprod()
    earnings_factors = (1 + loader.load("ep_6")).cumprod()
    book_market = (1+loader.load("bm_6")).cumprod()
    investments = (1+loader.load("inv6")).cumprod()
    beta = (1+ loader.load("beta")).cumprod()
    momo = (1+loader.load("momentum")).cumprod()


    zs_book_market = zscore((book_market.pct_change(12)), window=36).dropna()
    zs_investments = zscore(investments.pct_change(12), window=36).dropna()
    zs_beta = zscore(beta.pct_change(12), window=36).dropna()
    zs_earnings = zscore(earnings_factors.pct_change(12), window=36).dropna()
    zs_momo = zscore(momo.pct_change(12), window=36).dropna()
    
    
    factors = pd.DataFrame({
        "book_market_SMB": zs_book_market["SMALL LoBM"] - zs_book_market["BIG HiBM"],
        "investment_SMB": zs_investments["SMALL HiINV"] - zs_investments["BIG LoINV"],
        "beta_SMB": zs_beta["SMALL HiBETA"] - zs_beta["BIG LoBETA"],
        "earnings_SMB": zs_earnings["SMALL LoEP"] - zs_earnings["BIG HiEP"],
        "momo": zs_momo["Mom"],})

    return factors.dropna()



def get_industry_returns():

    """
    Fetch industry returns from Fama-French data and calculate z-score momentum for each industry.

    Returns
    -------
    pd.DataFrame
        Z-score momentum of industry returns.
    """

    loader = FamaFrenchLoader()

    ff49 = loader.load("ff49_daily", freq="daily", force_download=True)

    ff49 = ff49.mask(ff49 < -0.9, np.nan).dropna()

    ff49 = ff49.drop(columns=["Other"], errors="ignore")

    return ff49


def calculate_sector_rotation(ff49, windows=(63, 126, 252), z_window=756):
    
    def make_total_return_index(returns):
        return (1 + returns / 100).cumprod()

    def basket_index(ff49, industries, min_available=0.7):
        missing = [x for x in industries if x not in ff49.columns]
        if missing:
            raise ValueError(f"Missing industries in ff49: {missing}")

        tri = make_total_return_index(ff49[industries])

        min_count = max(1, int(len(industries) * min_available))

        return tri.mean(axis=1, skipna=True).where(
            tri.count(axis=1) >= min_count
        )

    cyclical = [
        "Steel", "Mach", "ElcEq", "Autos", "Aero", "Ships",
        "Cnstr", "BldMt", "FabPr", "Rubbr", "Txtls", "Whlsl",
        "Trans", "Oil", "Coal", "Mines", "Hardw", "Chips",
        "Rtail", "Meals"
    ]

    defensive = [
        "Food", "Soda", "Beer", "Smoke", "Hshld",
        "Drugs", "MedEq", "Hlth", "Util", "Telcm"
    ]

    industrials = [
        "Steel", "Mach", "ElcEq", "Aero", "Ships",
        "Cnstr", "BldMt", "FabPr", "Trans"
    ]

    consumer_discretionary = [
        "Autos", "Toys", "Fun", "Clths", "Rtail", "Meals"
    ]

    staples = ["Food", "Soda", "Beer", "Smoke", "Hshld"]
    housing = ["Cnstr", "BldMt", "RlEst"]
    semis_hardware = ["Chips", "Hardw", "LabEq"]
    commodities = ["Oil", "Coal", "Mines", "Steel"]
    healthcare_defensive = ["Drugs", "MedEq", "Hlth"]

    ratios = pd.DataFrame(index=ff49.index)

    ratios["cyc_def"] = basket_index(ff49, cyclical) / basket_index(ff49, defensive)
    ratios["industrials_def"] = basket_index(ff49, industrials) / basket_index(ff49, defensive)
    ratios["disc_staples"] = basket_index(ff49, consumer_discretionary) / basket_index(ff49, staples)
    ratios["housing_util"] = basket_index(ff49, housing) / basket_index(ff49, ["Util"])
    ratios["semis_util"] = basket_index(ff49, semis_hardware) / basket_index(ff49, ["Util"])
    ratios["commodities_def"] = basket_index(ff49, commodities) / basket_index(ff49, defensive)
    ratios["banks_util"] = basket_index(ff49, ["Banks"]) / basket_index(ff49, ["Util"])
    ratios["trans_util"] = basket_index(ff49, ["Trans"]) / basket_index(ff49, ["Util"])
    ratios["software_util"] = basket_index(ff49, ["Softw"]) / basket_index(ff49, ["Util"])
    ratios["healthcare_market"] = basket_index(ff49, healthcare_defensive) / basket_index(ff49, cyclical + defensive)

    features = pd.DataFrame(index=ratios.index)

    for col in ratios.columns:
        x = ratios[col]

        for w in windows:
            features[f"{col}_{w}d_mom"] = x.pct_change(w)

        features[f"{col}_ma252_ratio"] = x / x.rolling(252).mean() - 1

        features[f"{col}_zscore"] = (
            x - x.rolling(z_window).mean()
        ) / x.rolling(z_window).std()

        features[f"{col}_mom_accel"] = x.pct_change(63) - x.pct_change(252)

    all_features = pd.concat([features, ratios.add_suffix("_ratio")], axis=1)

    return all_features

def get_macro_quadrants():

    """
    Quad 1: GDP up, CPI down
    Quad 2: GDP up, CPI up
    Quad 3: GDP down, CPI up
    Quad 4: GDP down, CPI down

    """
    gdp = avoid_future_leakage(fred.get_series("GDP"), offset_months=6).resample("QE").last().dropna()
    cpi = avoid_future_leakage(fred.get_series("CPIAUCSL"), offset_months=2).resample("QE").last().dropna()

    gdp_qoq = gdp.pct_change().dropna()
    cpi_qoq = cpi.pct_change().dropna()
    gdp_yoy = gdp.pct_change(4).dropna()
    cpi_yoy = cpi.pct_change(4).dropna()

    economy = pd.concat({
        "gdp_qoq": gdp_qoq,
        "cpi_qoq": cpi_qoq,
        "gdp_yoy": gdp_yoy,
        "cpi_yoy": cpi_yoy,
    }, axis=1).dropna()

    # z-scores
    economy["gdp_yoy_z"] = zscore(economy["gdp_yoy"], window=80)
    economy["cpi_yoy_z"] = zscore(economy["cpi_yoy"], window=80)
    economy["gdp_qoq_z"] = zscore(economy["gdp_qoq"], window=80)
    economy["cpi_qoq_z"] = zscore(economy["cpi_qoq"], window=80)

    # regimes
    economy["regime_yoy"] = np.select(
        [
            (economy["gdp_yoy"].diff() > 0) & (economy["cpi_yoy"].diff() < 0),
            (economy["gdp_yoy"].diff() > 0) & (economy["cpi_yoy"].diff() > 0),
            (economy["gdp_yoy"].diff() < 0) & (economy["cpi_yoy"].diff() > 0),
            (economy["gdp_yoy"].diff() < 0) & (economy["cpi_yoy"].diff() < 0),
        ],
        [1, 2, 3, 4],
        default=np.nan
    )

    economy["regime_qoq"] = np.select(
        [
            (economy["gdp_qoq_z"] > 0) & (economy["cpi_qoq_z"] < 0),
            (economy["gdp_qoq_z"] > 0) & (economy["cpi_qoq_z"] > 0),
            (economy["gdp_qoq_z"] < 0) & (economy["cpi_qoq_z"] > 0),
            (economy["gdp_qoq_z"] < 0) & (economy["cpi_qoq_z"] < 0),
        ],
        [1, 2, 3, 4],
        default=np.nan
    )

    # strength
    economy["quad_strength_yoy"] = np.hypot(
        economy["gdp_yoy"].diff(),
        economy["cpi_yoy"].diff()
    )

    economy["quad_strength_qoq"] = np.hypot(
        economy["gdp_qoq"].diff(),
        economy["cpi_qoq"].diff()
    )

    economy["regime_yoy_duration"] = (
        economy["regime_yoy"]
        .ne(economy["regime_yoy"].shift())
        .cumsum()
    )

    economy["regime_yoy_duration"] = (
        economy
        .groupby("regime_yoy_duration")
        .cumcount() + 1
    )

    economy["regime_qoq_duration"] = (
        economy["regime_qoq"]
        .ne(economy["regime_qoq"].shift())
        .cumsum()
    )

    economy["regime_qoq_duration"] = (
        economy
        .groupby("regime_qoq_duration")
        .cumcount() + 1
    )

    economy = economy.dropna()

    return economy



def get_macroeconomic_features():

    """
    Quad 1: GDP up, CPI down  -> Goldilocks
    Quad 2: GDP up, CPI up    -> Reflation
    Quad 3: GDP down, CPI up  -> Stagflation
    Quad 4: GDP down, CPI down -> Deflation
    """
    gdp = avoid_future_leakage(fred.get_series("GDP"), offset_months=6).resample("QE").last().dropna()
    cpi = avoid_future_leakage(fred.get_series("CPIAUCSL"), offset_months=2).resample("QE").last().dropna()
    
    gdp_qoq = gdp.pct_change()
    cpi_qoq = cpi.pct_change()
    gdp_yoy = gdp.pct_change(4)
    cpi_yoy = cpi.pct_change(4)

    economy = pd.concat({
        "gdp_qoq": gdp_qoq,
        "cpi_qoq": cpi_qoq,
        "gdp_yoy": gdp_yoy,
        "cpi_yoy": cpi_yoy,
    }, axis=1).dropna()

    # --------------------------------------------------
    # 1. Levels / growth z-scores
    # --------------------------------------------------
    economy["gdp_yoy_z"] = zscore(economy["gdp_yoy"], window=80)
    economy["cpi_yoy_z"] = zscore(economy["cpi_yoy"], window=80)
    economy["gdp_qoq_z"] = zscore(economy["gdp_qoq"], window=80)
    economy["cpi_qoq_z"] = zscore(economy["cpi_qoq"], window=80)

    # --------------------------------------------------
    # 2. Acceleration
    # --------------------------------------------------
    economy["gdp_yoy_accel"] = economy["gdp_yoy"].diff()
    economy["cpi_yoy_accel"] = economy["cpi_yoy"].diff()

    economy["gdp_qoq_accel"] = economy["gdp_qoq"].diff()
    economy["cpi_qoq_accel"] = economy["cpi_qoq"].diff()

    # --------------------------------------------------
    # 3. Standardized acceleration
    # Better than using raw acceleration directly
    # --------------------------------------------------
    economy["gdp_yoy_accel_z"] = zscore(economy["gdp_yoy_accel"], window=80)
    economy["cpi_yoy_accel_z"] = zscore(economy["cpi_yoy_accel"], window=80)

    economy["gdp_qoq_accel_z"] = zscore(economy["gdp_qoq_accel"], window=80)
    economy["cpi_qoq_accel_z"] = zscore(economy["cpi_qoq_accel"], window=80)

    # --------------------------------------------------
    # 4. Regimes based on z-scored growth levels
    # --------------------------------------------------
    economy["regime_yoy"] = np.select(
        [
            (economy["gdp_yoy_z"] > 0) & (economy["cpi_yoy_z"] < 0),
            (economy["gdp_yoy_z"] > 0) & (economy["cpi_yoy_z"] > 0),
            (economy["gdp_yoy_z"] < 0) & (economy["cpi_yoy_z"] > 0),
            (economy["gdp_yoy_z"] < 0) & (economy["cpi_yoy_z"] < 0),
        ],
        [1, 2, 3, 4],
        default=np.nan
    )

    economy["regime_qoq"] = np.select(
        [
            (economy["gdp_qoq_z"] > 0) & (economy["cpi_qoq_z"] < 0),
            (economy["gdp_qoq_z"] > 0) & (economy["cpi_qoq_z"] > 0),
            (economy["gdp_qoq_z"] < 0) & (economy["cpi_qoq_z"] > 0),
            (economy["gdp_qoq_z"] < 0) & (economy["cpi_qoq_z"] < 0),
        ],
        [1, 2, 3, 4],
        default=np.nan
    )

    
    # --------------------------------------------------
    # 7. Strength / magnitude of macro acceleration
    # --------------------------------------------------
    economy["strength_yoy"] = np.hypot(
        economy["gdp_yoy_accel_z"],
        economy["cpi_yoy_accel_z"]
    )

    economy["strength_qoq"] = np.hypot(
        economy["gdp_qoq_accel_z"],
        economy["cpi_qoq_accel_z"]
    )

    # --------------------------------------------------
    # 11. Regime transition flags
    # --------------------------------------------------
    economy["regime_yoy_changed"] = (
        economy["regime_yoy"] != economy["regime_yoy"].shift(1)
    )

    economy["regime_qoq_changed"] = (
        economy["regime_qoq"] != economy["regime_qoq"].shift(1)
    )

    economy["quad_yoy_changed"] = (
        economy["quads_yoy"] != economy["quads_yoy"].shift(1)
    )

    economy["quad_qoq_changed"] = (
        economy["quads_qoq"] != economy["quads_qoq"].shift(1)
    )

    # --------------------------------------------------
    # 12. Months/quarters in current regime
    # --------------------------------------------------
    economy["regime_yoy_duration"] = (
        economy["regime_yoy"]
        .ne(economy["regime_yoy"].shift())
        .cumsum()
    )

    economy["regime_yoy_duration"] = (
        economy
        .groupby("regime_yoy_duration")
        .cumcount() + 1
    )

    economy["regime_qoq_duration"] = (
        economy["regime_qoq"]
        .ne(economy["regime_qoq"].shift())
        .cumsum()
    )

    economy["regime_qoq_duration"] = (
        economy
        .groupby("regime_qoq_duration")
        .cumcount() + 1
    )

    economy = economy.dropna()

    return economy


def get_market_state(start: str = "1980-01-01") -> pd.DataFrame:
    """
    Build daily market-state features and resample them to month-end.

    Feature groups:
    - Trend
    - Breadth
    - Volatility
    - Credit / risk appetite
    - Cross-asset leadership
    - Persistence / market regime duration
    """

    # ============================================================
    # 1. Download market data
    # ============================================================

    tickers = [
        "^GSPC", "^NDX", "^DJI", "^VIX"
    ]

    prices = yf.download(
        tickers,
        start=start,
        auto_adjust=True,
        progress=False
    )["Close"]

    prices = prices.dropna(how="all")

    spx = prices["^GSPC"].dropna()
    ndx = prices["^NDX"].dropna()
    dow = prices["^DJI"].dropna()
    vix = prices["^VIX"].dropna()

    # Treasury yields
    dgs10 = fred.get_series("DGS10").ffill() / 100
    
    # ============================================================
    # 2. Core return / volatility calculations
    # ============================================================

    spx_ret = spx.pct_change()
    ndx_ret = ndx.pct_change()
    dow_ret = dow.pct_change()

    spx_vol = spx_ret.rolling(21).std() * np.sqrt(252) * 100
    ndx_vol = ndx_ret.rolling(21).std() * np.sqrt(252) * 100
    dow_vol = dow_ret.rolling(21).std() * np.sqrt(252) * 100

    dgs10_vol = dgs10.diff().rolling(21).std() * np.sqrt(252) * 100

    # ============================================================
    # 3. Trend features
    # ============================================================

    df = pd.DataFrame(index=spx.index)

    df["spx_ret_1m"] = spx.pct_change(21)
    df["spx_ret_3m"] = spx.pct_change(63)
    df["spx_ret_6m"] = spx.pct_change(126)
    df["spx_ret_12m"] = spx.pct_change(252)

    df["spx_above_50dma"] = (spx > spx.rolling(50).mean()).astype(int)
    df["spx_above_200dma"] = (spx > spx.rolling(200).mean()).astype(int)

    df["spx_distance_50dma"] = spx / spx.rolling(50).mean() - 1
    df["spx_distance_200dma"] = spx / spx.rolling(200).mean() - 1

    df["spx_50dma_vs_200dma"] = (
        spx.rolling(50).mean() / spx.rolling(200).mean() - 1
    )

    # ============================================================
    # 4. Drawdown / correction features
    # ============================================================

    running_peak = spx.cummax()
    drawdown = spx / running_peak - 1

    df["spx_drawdown"] = drawdown

    correction_start = (
        (drawdown <= -0.10) &
        (drawdown.shift(1) > -0.10)
    )

    df["spx_days_since_10pct_correction"] = days_since_event(correction_start)

    below_high = spx < running_peak
    df["spx_drawdown_duration"] = consecutive_true_days(below_high)

    # ============================================================
    # 5. Volatility features
    # ============================================================

    df["vix"] = vix
    df["spx_realized_vol_21d"] = spx_vol
    df["ndx_realized_vol_21d"] = ndx_vol
    df["dow_realized_vol_21d"] = dow_vol
    df["dgs10_vol_21d"] = dgs10_vol

    df["vix_minus_spx_realized_vol"] = vix - spx_vol
    df["vix_to_spx_realized_vol"] = vix / spx_vol

    df["spx_realized_vol_percentile_3y"] = rolling_percentile(spx_vol, 756)
    df["vix_percentile_3y"] = rolling_percentile(vix, 756)

    df["vix_days_below_20"] = consecutive_true_days(vix < 20)
    df["vix_days_above_30"] = consecutive_true_days(vix > 30)

    # ============================================================
    # 6. Market persistence features
    # ============================================================

    df["spx_days_above_200dma"] = consecutive_true_days(
        spx > spx.rolling(200).mean()
    )

    df["spx_days_below_200dma"] = consecutive_true_days(
        spx < spx.rolling(200).mean()
    )

    # ============================================================
    # 7. Cross-asset / leadership features
    # ============================================================

    # NDX vs SPX momentum leadership
    df["ndx_vs_spx_mom_6m_z"] = (
        zscore(ndx.pct_change(126), 756)
        - zscore(spx.pct_change(126), 756)
    )

    df["spx_vol_vs_dgs10_vol_z"] = (
        zscore(spx_vol, 756)
        - zscore(dgs10_vol, 756)
    )


    # ============================================================
    # 8. Industry breadth from Fama-French 49 industries
    # ============================================================

    ff49 = get_industry_returns()

    # If your Fama-French returns are in percent, use /100.
    # If they are already decimals, remove /100.
    ff49_index = (1 + ff49 / 100).cumprod()

    ff49_ma200 = ff49_index.rolling(200).mean()

    df["ff49_pct_above_200dma"] = (
        (ff49_index > ff49_ma200)
        .mean(axis=1)
        * 100
    )

    df["ff49_pct_positive_6m_mom"] = (
        (ff49_index.pct_change(126) > 0)
        .mean(axis=1)
        * 100
    )

    df["ff49_median_6m_mom"] = ff49_index.pct_change(126).median(axis=1)

    df["ff49_cross_sectional_dispersion_21d"] = (
        ff49.pct_change()
        .rolling(21)
        .std()
        .mean(axis=1)
    )

    # ============================================================
    # 9. Clean and resample to month-end
    # ============================================================

    df = df.sort_index()

    market_state_monthly = (
        df
        .resample("ME")
        .last()
        .ffill()
    )

    return market_state_monthly.dropna()
    

def get_bond_etf_data():
    """
    Fetch bond ETF data from Yahoo Finance, resample to monthly frequency, and drop missing values.

    Returns
    -------
    pd.DataFrame
        Combined bond ETF data.
    """

    bond_data = pd.DataFrame({
        "AGG": yf.download('AGG', start='1995-01-01', auto_adjust=True)['Close'].squeeze(),
        "TLT": yf.download('TLT', start='1995-01-01', auto_adjust=True)['Close'].squeeze(),
        "SHY": yf.download('SHY', start='1995-01-01', auto_adjust=True)['Close'].squeeze(),
        "IEF": yf.download('IEF', start='1995-01-01', auto_adjust=True)['Close'].squeeze(),
        "BND": yf.download('BND', start='1995-01-01', auto_adjust=True)['Close'].squeeze(),
        "LQD": yf.download('LQD', start='1995-01-01', auto_adjust=True)['Close'].squeeze(),
        "HYG": yf.download('HYG', start='1995-01-01', auto_adjust=True)['Close'].squeeze(),
        "PFF": yf.download('PFF', start='1995-01-01', auto_adjust=True)['Close'].squeeze(),
        "STIP": yf.download('STIP', start='1995-01-01', auto_adjust=True)['Close'].squeeze(),
        "TIP": yf.download('TIP', start='1995-01-01', auto_adjust=True)['Close'].squeeze(),
    })

    return bond_data

def get_factors_etfs():

    """
    Fetch factor ETF data from Yahoo Finance
    
    """

    factors_etfs = pd.DataFrame({
        "R1000 Growth IWF": yf.download('IWF', start='1995-01-01', auto_adjust=True)['Close'].squeeze(),
        "R1000 Value IWD": yf.download('IWD', start='1995-01-01', auto_adjust=True)['Close'].squeeze(),
        "EW-SPX RSP": yf.download('RSP', start='1995-01-01', auto_adjust=True)['Close'].squeeze(),
        "High-Beta SPHB": yf.download('SPHB', start='1995-01-01', auto_adjust=True)['Close'].squeeze(),
        "Low Vol SPLV": yf.download('SPLV', start='1995-01-01', auto_adjust=True)['Close'].squeeze(),
        "Momentum MTUM": yf.download('MTUM', start='1995-01-01', auto_adjust=True)['Close'].squeeze(),
        "Quality QUAL": yf.download('QUAL', start='1995-01-01', auto_adjust=True)['Close'].squeeze(),
        "Dividend SDY": yf.download('SDY', start='1995-01-01', auto_adjust=True)['Close'].squeeze(),
        "SPY 100": yf.download('OEF', start='1995-01-01', auto_adjust=True)['Close'].squeeze(),
        "Mid-Cap MDY": yf.download('MDY', start='1995-01-01', auto_adjust=True)['Close'].squeeze(),
        "Small-Cap IWM": yf.download('IWM', start='1995-01-01', auto_adjust=True)['Close'].squeeze(),
    })

    return factors_etfs

def get_sector_etfs():

    """
    Fetch sector ETF data from Yahoo Finance
    
    """

    sector_etfs = pd.DataFrame({
        "Energy XLE": yf.download('XLE', start='1995-01-01', auto_adjust=True)['Close'].squeeze(),
        "Financials XLF": yf.download('XLF', start='1995-01-01', auto_adjust=True)['Close'].squeeze(),
        "Tech XLK": yf.download('XLK', start='1995-01-01', auto_adjust=True)['Close'].squeeze(),
        "Health XLV": yf.download('XLV', start='1995-01-01', auto_adjust=True)['Close'].squeeze(),
        "Industrials XLI": yf.download('XLI', start='1995-01-01', auto_adjust=True)['Close'].squeeze(),
        "Consumer Staples XLP": yf.download('XLP', start='1995-01-01', auto_adjust=True)['Close'].squeeze(),
        "Real Estate XLRE": yf.download('XLRE', start='1995-01-01', auto_adjust=True)['Close'].squeeze(),
        "Utilities XLU": yf.download('XLU', start='1995-01-01', auto_adjust=True)['Close'].squeeze(),
        "Materials XLB": yf.download('XLB', start='1995-01-01', auto_adjust=True)['Close'].squeeze(),
        "Comm XLC": yf.download('XLC', start='1995-01-01', auto_adjust=True)['Close'].squeeze(),
        "S&P 500 SPY": yf.download('SPY', start='1995-01-01', auto_adjust=True)['Close'].squeeze(),
    })

    return sector_etfs

def get_performance_by_quad(returns):

    """
    Calculate the median industry performance for each economic regime (quad).

    """

    curr_regime = get_macroeconomic_features().resample("ME").last().ffill()

    returns = get_industry_returns().sort_index()

    rf = pd.Series(
        0.0,
        index=returns.index,
        name="rf"
    )

    # Metric calculated over each 63-day window
    sortino_63d = get_rolling_sortino_ratio(
        returns,
        rf_series=rf,
        window=63
    )

    tail_63d = rolling_tail_ratio(
        returns,
        window=63
    )

    # Assign the metric for t+1:t+63 to date t
    forward_sortino_3m = sortino_63d.shift(-63)
    forward_tail_3m = tail_63d.shift(-63)

    forward_risk_3m = (
        0.7 * forward_sortino_3m
        + 0.3 * forward_tail_3m
    )

    # Rank industries relative to each other at each date
    forward_risk_rank_3m = (
        forward_risk_3m
        .add_suffix("_forward_3m_risk_rank")
    )

    quad_daily = (
        curr_regime["quads_yoy_curr"]
        .reindex(returns.index)
        .ffill()
        .rename("quad")
    )

    quads = (
        forward_risk_rank_3m
        .join(quad_daily)
        .dropna()
    )

    quad_median_ranks = (
        quads
        .groupby("quad")
        .median()
        .T
    )

    return quad_median_ranks


def get_macroeconomic_factors_for_ml():

    """
    Fetch and process macroeconomic factors for modeling.
    """
    macro_data = get_macroeconomic_data()
    housing_data = compute_housing_pca()
    money_market_factors = compute_money_markets_factors()
    inflation_data = compute_inflation_factors()

    regime = get_macroeconomic_features().resample("ME").last().ffill()

    curr_regime = regime.add_suffix("_curr")
    prev_regime = regime.shift(1).add_suffix("_prev")

    industry_features = calculate_sector_rotation(get_industry_returns())
    market_state = get_market_state()

    dfs = {
        "macro_data": macro_data,
        "housing_data": housing_data,
        "money_market_factors": money_market_factors,
        "inflation": inflation_data,
        "curr_regime": curr_regime,
        "prev_regime": prev_regime,
        "industry_features": industry_features,
        "market_state": market_state,
    }

    dfs = {
        name: df.resample("ME").last().ffill()
        for name, df in dfs.items()
    }

    macro_factors = pd.concat(dfs.values(), axis=1, sort=True)

    macro_factors = macro_factors.loc["1991-04-30":]

    macro_factors = macro_factors.dropna(axis=1, thresh=int(len(macro_factors) * 0.95))

    macro_factors = macro_factors.dropna()

    return macro_factors


def get_benchmark():
    """
    Fetch benchmark returns for the S&P 500 index.

    Returns
    -------
    pd.DataFrame
        DataFrame containing daily returns of the S&P 500 index.
    """

    benchmark_prices = yf.download(
        "^GSPC",
        start="1900-01-01",
        auto_adjust=True,
        progress=False
    )
    
    price = benchmark_prices["Close"].squeeze()
    volume = benchmark_prices["Volume"].squeeze().replace(0, np.nan).ffill().dropna()
    relative_volume = volume / volume.rolling(63).mean()
    st_volatility = (price.pct_change().rolling(21).std() * np.sqrt(252) * 100).dropna()
    lt_volatility = (price.pct_change().rolling(63).std() * np.sqrt(252) * 100).dropna()

    vol_ratio = (st_volatility / lt_volatility).dropna()
    yoy_returns = price.pct_change(252).dropna()

    benchmark = pd.DataFrame({
        "spx_price": price,
        "spx_volume": volume,
        "spx_relative_volume": relative_volume,
        "spx_st_volatility": st_volatility,
        "spx_lt_volatility": lt_volatility,
        "spx_vol_ratio": vol_ratio,
        "spx_yoy_returns": yoy_returns
    })

    return benchmark.dropna()





def walk_forward_xgb(
    X: pd.DataFrame,
    y: pd.Series,
    min_train_size: int = 80,
    model_params: dict | None = None,
):
    """
    Expanding-window, one-observation-at-a-time GDP forecasting.
    """

    dataset = pd.concat(
        [y.rename("target"), X],
        axis=1,
        sort=False,
    ).dropna()

    y = dataset["target"]
    X = dataset.drop(columns="target")

    default_params = {
        "objective": "reg:squarederror",
        "n_estimators": 500,
        "learning_rate": 0.03,
        "max_depth": 2,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.5,
        "reg_lambda": 3.0,
        "random_state": 42,
        "n_jobs": -1,
    }

    if model_params:
        default_params.update(model_params)

    results = []

    for test_position in range(min_train_size, len(X)):
        X_train = X.iloc[:test_position]
        y_train = y.iloc[:test_position]

        X_test = X.iloc[[test_position]]
        y_test = y.iloc[test_position]

        model = XGBRegressor(**default_params)
        model.fit(X_train, y_train)

        prediction = model.predict(X_test)[0]

        results.append({
            "feature_date": X_test.index[0],
            "actual": y_test,
            "prediction": prediction,
        })

    results = pd.DataFrame(results).set_index("feature_date")

    results["error"] = (
        results["actual"] - results["prediction"]
    )

    return results


def fit_and_predict_next_gdp(
    X: pd.DataFrame,
    y: pd.Series,
    latest_features: pd.DataFrame | None = None,
):
    """
    Fit the final model and generate the next-quarter GDP forecast.
    """

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        learning_rate=0.03,
        max_depth=2,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=3.0,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X, y)

    if latest_features is None:
        latest_features = X.iloc[[-1]]

    latest_features = latest_features.reindex(
        columns=X.columns
    )

    forecast = model.predict(latest_features)[0]

    forecast_origin = latest_features.index[-1]
    forecast_period = (
        forecast_origin + pd.offsets.QuarterEnd(1)
    )

    return {
        "model": model,
        "forecast": forecast,
        "forecast_origin": forecast_origin,
        "forecast_period": forecast_period,
    }