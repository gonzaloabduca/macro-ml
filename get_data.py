import pandas as pd
import numpy as np
from feature_engineering import *
from macro import *
from fredapi import Fred
from functions import *
import yfinance as yf



fred = Fred(api_key='a2fb338b4ef6e2dcb7c667c21b2d1c4e')

quads = get_macro_quadrants()
quads.to_pickle("data/macro_quadrants.pkl")

macro = get_raw_macroeconomic_data()
macro.to_pickle("data/raw_macro_data.pkl")

money_market = get_money_market_data()
money_market.to_pickle("data/raw_money_market_data.pkl")

inflation_factors = compute_inflation_factors()

cpi = avoid_future_leakage(fred.get_series("CPIAUCSL"), offset_months=2).interpolate().pct_change(12).dropna().rename("cpi_yoy").resample("ME").last()

previous_cpi = zscore(cpi.shift(2).rename("previous_cpi_yoy_z"), window=60)
cpi_1y_mean = zscore(cpi.shift(2).rolling(12).mean().rename("cpi_1y_mean_z"), window=60)

inflation_ml = pd.concat([inflation_factors, cpi, previous_cpi, cpi_1y_mean], axis=1).dropna()

inflation_ml.to_pickle("data/inflation_factors.pkl")

industry_returns = get_industry_returns()
industry_returns.to_pickle("data/industry_returns.pkl")

benchmark = yf.download('^GSPC', start='1900-01-01')['Close'].squeeze().pct_change().dropna().rename("sp500")
benchmark.to_pickle("data/sp500_returns.pkl")
