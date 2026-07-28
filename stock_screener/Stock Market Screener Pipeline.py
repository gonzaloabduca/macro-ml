import pandas as pd 
import numpy as np
import yfinance as yf
import time
import requests
import re


def stock_screener(tickers, verbose=True):
    
    pd.options.mode.copy_on_write = True

    all_stats = []

    for ticker in tickers:

        try:

            if verbose:
                print(f"Processing ticker: {ticker}")

            # Add delay to avoid 429 error
            time.sleep(0.5)

            # Fetch stock data for the ticker
            stock = yf.Ticker(ticker)
            
            def safe_round(value, ndigits=2):
                return round(value, ndigits) if isinstance(value, (int, float)) else np.nan

            # Get company info
            info = stock.info
            company_name = info.get('shortName', 'N/A')
            sector = info.get('sector', 'N/A')
            industry = info.get('industry', 'N/A')
            country = info.get('country', 'N/A')
            current_price = info.get('currentPrice', np.nan)
            current_eps = info.get('trailingEps', np.nan)
            forward_eps = info.get('forwardEps', np.nan)
            trailing_pe = info.get('trailingPE', np.nan)
            forward_pe = info.get('forwardPE', np.nan)
            trailing_peg = info.get('trailingPegRatio', np.nan)
            current_eg = info.get('earningsGrowth', np.nan)
            current_rg = info.get('revenueGrowth', np.nan)
            current_roa = info.get('returnOnAssets', np.nan)
            current_roe = info.get('returnOnEquity', np.nan)
            gross_margins = info.get('grossMargins', np.nan)
            price_sales = info.get('priceToSalesTrailing12Months')
            price_book = info.get('priceToBook')

            analyts_recommendation = info.get('recommendationMean', np.nan)
            number_analysts = info.get('numberOfAnalystOpinions')
            institutions_hold = info.get('heldPercentInstitutions', np.nan)
            insiders_hold = info.get('heldPercentInsiders', np.nan)
            short_ratio = info.get('shortRatio', np.nan)
            short_float = info.get('shortPercentOfFloat', np.nan)
            market_cap = info.get('marketCap', np.nan)

            income = stock.quarterly_income_stmt.T.fillna(np.nan).infer_objects(copy=False)
            balance = stock.quarterly_balance_sheet.T.fillna(np.nan).infer_objects(copy=False)
            cash = stock.quarterly_cashflow.T.fillna(np.nan).infer_objects(copy=False)

            # Combine financial data
            financials = pd.concat([income, balance, cash], axis=1).sort_index().ffill()

            inc = stock.quarterly_incomestmt
            cf  = stock.quarterly_cashflow
            bs  = stock.quarterly_balance_sheet
            info = stock.info

            try:
                rd = inc.loc["Research And Development"].iloc[0]
            except:
                rd = np.nan
            try:
                rev = inc.loc["Total Revenue"].iloc[0]
            except:
                rev = np.nan

            # R&D ratios
            rd_rev = rd / rev if rev and rev != 0 else np.nan
            mktcap = info.get("marketCap", np.nan)
            rd_mcap = rd / mktcap if mktcap else np.nan

            # Cash burn (using OCF)
            try:
                ocf = cf.loc["Total Cash From Operating Activities"].iloc[0]
            except:
                ocf = np.nan
            burn_rate = -ocf if ocf < 0 else 0

            # Runway
            try:
                casheq = bs.loc["Cash"].iloc[0]
            except:
                casheq = np.nan
        
            runway_quarters = casheq / burn_rate if burn_rate > 0 else np.inf

            ebit = stock.quarterly_income_stmt.loc['EBIT'].iloc[0] if 'EBIT' in stock.quarterly_income_stmt.index else np.nan
            ev = stock.info.get('enterpriseValue', np.nan)
            ebit_yield = ebit/ev
            fcf_yield = stock.info.get('freeCashflow', np.nan) / stock.info.get('enterpriseValue', np.nan)

            performance_metrics = pd.DataFrame(index=financials.index)

            # Growth Metrics
            revenue = financials.get('Total Revenue', np.nan)
            performance_metrics['Revenue Growth'] = revenue.pct_change(periods=4) if isinstance(revenue, pd.Series) else np.nan

            gross_income = financials.get('Gross Profit', np.nan)
            performance_metrics['Gross Income Growth'] = gross_income.pct_change(periods=4) if isinstance(gross_income, pd.Series) else np.nan

            operating_income = financials.get('Operating Income', np.nan)
            performance_metrics['Operating Income Growth'] = operating_income.pct_change(periods=4) if isinstance(operating_income, pd.Series) else np.nan

            net_income = financials.get('Net Income', np.nan)
            performance_metrics['Net Income Growth'] = net_income.pct_change(periods=4) if isinstance(net_income, pd.Series) else np.nan

            # Capital Structure Metrics
            assets = financials.get('Total Assets', np.nan)
            equity = financials.get('Stockholders Equity', np.nan)

            if isinstance(net_income, pd.Series) and isinstance(assets, pd.Series):
                performance_metrics['Return on Assets'] = net_income.rolling(4).sum() / assets.rolling(4).mean()
            else:
                performance_metrics['Return on Assets'] = np.nan

            if isinstance(net_income, pd.Series) and isinstance(equity, pd.Series):
                performance_metrics['Return on Equity'] = net_income.rolling(4).sum() / equity.rolling(4).mean()
            else:
                performance_metrics['Return on Equity'] = np.nan

            # Financial Health Metrics
            interest_expense = financials.get('Interest Expense', np.nan)
            performance_metrics['Interest Coverage Ratio'] = (
                operating_income / interest_expense if isinstance(operating_income, pd.Series) and isinstance(interest_expense, pd.Series) else np.nan
            )

            net_debt = financials.get('Total Liabilities Net Minority Interest', np.nan) - financials.get('Cash And Cash Equivalents', np.nan)
            ebitda = financials.get('EBITDA', np.nan)

            if isinstance(net_debt, pd.Series) and isinstance(ebitda, pd.Series):
                performance_metrics['Net Debt to EBITDA'] = ebitda.rolling(4).sum() / net_debt
            else:
                performance_metrics['Net Debt to EBITDA'] = np.nan

            # Liquidity Metrics
            current_assets = financials.get('Current Assets', np.nan)
            current_liabilities = financials.get('Current Liabilities', np.nan)
            inventory = financials.get('Inventory', np.nan)

            performance_metrics['Current Ratio'] = (
                current_assets / current_liabilities if isinstance(current_assets, pd.Series) and isinstance(current_liabilities, pd.Series) else np.nan
            )
            performance_metrics['Quick Ratio'] = (
                (current_assets - inventory) / current_liabilities if isinstance(current_assets, pd.Series) and isinstance(inventory, pd.Series) and isinstance(current_liabilities, pd.Series) else np.nan
            )

            # Operating Cash Flow Ratio
            operating_cash_flow = financials.get('Operating Cash Flow', np.nan)

            if isinstance(operating_cash_flow, pd.Series) and isinstance(revenue, pd.Series):
                performance_metrics['Operating Cash Flow Ratio'] = operating_cash_flow.rolling(4).sum() / revenue.rolling(4).sum()
            else:
                performance_metrics['Operating Cash Flow Ratio'] = np.nan

            performance_metrics['R&D / Sales Ratio'] = rd_rev
            performance_metrics['R&D / Market Cap Ratio'] = rd_mcap
            performance_metrics['Runaway Quarters'] = runway_quarters

            eps_expectation = (forward_eps - current_eps) / abs(current_eps)

            latest_metrics = performance_metrics.iloc[-1].to_dict()
            latest_metrics.update({
                'Ticker': ticker,
                'Company Name': company_name,
                'Sector': sector,
                'Industry': industry,
                'Country': country,
                'Current Price' : current_price,
                'Market Cap in Millions (USD)': safe_round(market_cap * 1e-6, 2),
                'Current EPS': current_eps,
                'Forward EPS': forward_eps,
                'Current P/E': safe_round(trailing_pe, 2),
                'Forward P/E': safe_round(forward_pe, 2),
                'Current PEG Ratio' : safe_round(trailing_peg, 2),
                'Current Earnings Growth' : safe_round(current_eg, 2),
                'EPS Growth Expectation' : safe_round(eps_expectation, 2),
                'Current Revenue Growth' : safe_round(current_rg, 2),
                'Gross Margin' : safe_round(gross_margins, 2),
                'P/S Ratio': safe_round(price_sales, 2),
                'P/B Ratio' : safe_round(price_book, 2),
                'Current ROA': safe_round(current_roa, 2),
                'Current ROE' : safe_round(current_roe, 2),
                'EBIT Yield' : safe_round(ebit_yield * 100,2),
                'FCF Yield' : safe_round(fcf_yield * 100,2),
                'Analyst Recommendation' : analyts_recommendation,
                'Number of Analysts' : number_analysts,
                'Insiders Holdings' : insiders_hold,
                'Institutional Holdings' : institutions_hold,
                'Short Ratio' : short_ratio,
                '(%)Float Short' : short_float
            })

            # Append metrics to the list
            all_stats.append(latest_metrics)

        except Exception as e:
                if verbose:
                    print(f"Error processing ticker {ticker}: {e}")
                continue
        
    watchlist = pd.DataFrame(all_stats)

    first_columns = ['Ticker', 
                    'Company Name',
                    'Sector',
                    'Industry',
                    'Country',
                    'Current Price',
                    'Market Cap in Millions (USD)',
                    'Current EPS',
                    'Forward EPS',
                    'Current P/E',
                    'Forward P/E',
                    'Current PEG Ratio',
                    'Current Earnings Growth',
                    'EPS Growth Expectation',
                    'Current Revenue Growth',
                    'Gross Margin',
                    'P/S Ratio',
                    'P/B Ratio',
                    'Current ROA',
                    'Current ROE',
                    'EBIT Yield',
                    'FCF Yield',
                    'Analyst Recommendation',
                    'Number of Analysts',
                    'Insiders Holdings',
                    'Institutional Holdings',
                    'Short Ratio',
                    '(%)Float Short'
                    ]
    

    columns_order =  first_columns + [col for col in watchlist.columns if col not in first_columns]
    watchlist = watchlist[columns_order]

    return watchlist


def pain_trade(df: pd.DataFrame) -> pd.Series:

    d = df.copy()

    cols = ["Days to Earnings", "Short Ratio",
            "(%)Float Short", "Analyst Recommendation "]
    for c in cols:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    allowed_sizes = {"Mid Cap", "Small Cap", "Micro Cap", 'Nano Cap'}
    mask = d["Company Size"].isin(allowed_sizes) if "Company Size" in d.columns else pd.Series(True, index=d.index)

    comps = pd.DataFrame({
        "ShortRatio":          d.loc[mask, "Short Ratio"].rank(pct=True),
        "daystoearnings":     d.loc[mask, "Days to Earnings"].rank(pct=True),
        "Reco": d.loc[mask, "Analyst Recommendation"].rank(pct=True),
        "FloatShort": d.loc[mask, "(%)Float Short"].rank(pct=True)
    })

    comps = comps.fillna(0)
    
    weights = pd.Series({"ShortRatio":3.0, "FloatShort":2.0, "daystoearnings":1.5, "Reco":1.5})

    # weighted average across available (non-NaN) components
    w = weights.reindex(comps.columns)
    weighted = comps.mul(w, axis=1)
    denom_w = comps.notna().mul(w, axis=1).sum(axis=1)
    score = weighted.sum(axis=1, skipna=True) / denom_w

    # --- min–max scale to 0–10 across the scored subset ---
    smin = score.min()
    smax = score.max()
    if pd.isna(smin) or pd.isna(smax) or smax == smin:
        score_scaled = pd.Series(np.nan, index=score.index)  # avoid 0/0
    else:
        score_scaled = (score - smin) / (smax - smin) * 10.0

    # align back to full df: non-allowed sizes remain NaN
    return score_scaled.reindex(df.index)


def growth_score(df: pd.DataFrame) -> pd.Series:

    d = df.copy()

    # coerce to numeric to ensure .rank works
    cols = ["Current EPS","Forward EPS","Current P/E","Forward P/E",
            "Current Revenue Growth",
            "Gross Margin","Operating Cash Flow Ratio"]
    for c in cols:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    denom = d["Current EPS"].abs().replace(0, np.nan)
    d["EPS_Gap"] = (d["Forward EPS"] - d["Current EPS"]) / denom
    d["PE_Spread"] = d["Current P/E"] - d["Forward P/E"]

    allowed_sizes = {"Mid Cap", "Small Cap", "Micro Cap"}
    mask = d["Company Size"].isin(allowed_sizes) if "Company Size" in d.columns else pd.Series(True, index=d.index)

    comps = pd.DataFrame({
        "PE":          d.loc[mask, "Current P/E"].rank(pct=True),
        "EG1" : d.loc[mask, "Current Earnings Growth"].rank(pct=True),
        "EG2":     d.loc[mask, "EPS_Gap"].rank(pct=True),
        "RG1":     d.loc[mask, "Current Revenue Growth"].rank(pct=True),
        "GrossMargin": d.loc[mask, "Gross Margin"].rank(pct=True),
        "OCF_Ratio":   d.loc[mask, "Operating Cash Flow Ratio"].rank(pct=True),
    })

    comps = comps.fillna(0)

    weights = pd.Series({"PE":2.0, "EG2":3.0, "EG1":2.0, "RG1":2.0, "GrossMargin":1.5, "OCF_Ratio":1.5})

    # weighted average across available (non-NaN) components
    w = weights.reindex(comps.columns)
    weighted = comps.mul(w, axis=1)
    denom_w = comps.notna().mul(w, axis=1).sum(axis=1)
    score = weighted.sum(axis=1, skipna=True) / denom_w

    # --- min–max scale to 0–10 across the scored subset ---
    smin = score.min()
    smax = score.max()
    if pd.isna(smin) or pd.isna(smax) or smax == smin:
        score_scaled = pd.Series(np.nan, index=score.index)  # avoid 0/0
    else:
        score_scaled = (score - smin) / (smax - smin) * 10.0

    # align back to full df: non-allowed sizes remain NaN
    return score_scaled.reindex(df.index)


def value_score(df: pd.DataFrame) -> pd.Series:

    d = df.copy()

    # coerce to numeric to ensure .rank works
    cols = ["EBIT Yield", "FCF Yield",
            "5 Yr. Hist. EPS Growth",
            "5 Yr Historical Sales Growth",
            "Return on Equity",
            "Interest Coverage Ratio",
            "Current Ratio",
            "Insiders Holdings",
            "Operating Cash Flow Ratio"]
    
    for c in cols:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    comps = pd.DataFrame({
        "ebitYield":        d["EBIT Yield"].rank(pct=True),
        "fcfYield":         d["FCF Yield"].rank(pct=True),
        "5yEpsGrowth":      d["5 Yr. Hist. EPS Growth"].rank(pct=True),
        "5yRevGrowth":      d["5 Yr Historical Sales Growth"].rank(pct=True),
        "interestCoverage": d["Interest Coverage Ratio"].rank(pct=True),
        "currentRatio":     d["Current Ratio"].rank(pct=True),
        "insidersHold":     d["Insiders Holdings"].rank(pct=True),
    }, index=d.index)

    comps = comps.fillna(0)

    weights = pd.Series({"ebitYield":2.0, "fcfYield":3.0, "5yEpsGrowth":1.5,
                         "5yRevGrowth":2.0, "interestCoverage":1.5,
                         "currentRatio":1.5, "insidersHold":1})

    # weighted average across available (non-NaN) components
    w = weights.reindex(comps.columns)
    weighted = comps.mul(w, axis=1)
    denom_w = comps.notna().mul(w, axis=1).sum(axis=1)
    score = weighted.sum(axis=1, skipna=True) / denom_w

    # --- min–max scale to 0–10 across the scored subset ---
    smin = score.min()
    smax = score.max()
    if pd.isna(smin) or pd.isna(smax) or smax == smin:
        score_scaled = pd.Series(np.nan, index=score.index)  # avoid 0/0
    else:
        score_scaled = (score - smin) / (smax - smin) * 10.0

    # align back to full df: non-allowed sizes remain NaN
    return score_scaled.reindex(df.index)

def efficency_score(df: pd.DataFrame) -> pd.Series:

    d = df.copy()

    # coerce to numeric to ensure .rank works
    cols = ["Revenue Growth", "Operating Income Growth",
            "ROA Growth",
            "Return on Equity",
            "Quick Ratio",
            ]
        
    for c in cols:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    d['OL'] =  d['Operating Income Growth'].replace(0, np.nan) / d['Revenue Growth'].replace(0, np.nan)

    comps = pd.DataFrame({
        "ol":        d["OL"].rank(pct=True),
        "roa":         d["ROA Growth"].rank(pct=True),
        "roe":      d["Return on Equity"].rank(pct=True),
        "CuRatio" : d["Quick Ratio"].rank(pct=True)
    }, index=d.index)

    comps = comps.fillna(0)

    weights = pd.Series({"ol":4.0, "roa":3.0, "roe":2,
                         "CuRatio":1.5})

    # weighted average across available (non-NaN) components
    w = weights.reindex(comps.columns)
    weighted = comps.mul(w, axis=1)
    denom_w = comps.notna().mul(w, axis=1).sum(axis=1)
    score = weighted.sum(axis=1, skipna=True) / denom_w

    # --- min–max scale to 0–10 across the scored subset ---
    smin = score.min()
    smax = score.max()
    if pd.isna(smin) or pd.isna(smax) or smax == smin:
        score_scaled = pd.Series(np.nan, index=score.index)  # avoid 0/0
    else:
        score_scaled = (score - smin) / (smax - smin) * 10.0

    # align back to full df: non-allowed sizes remain NaN
    return score_scaled.reindex(df.index)


def promise_score(df: pd.DataFrame) -> pd.Series:

    d = df.copy()

    # coerce to numeric to ensure .rank works
    cols = ["EPS Growth Expectation",
            "Current Revenue Growth",
            "R&D / Sales Ratio",
            "R&D / Market Cap Ratio",
            'Runaway Quarters'
            ]
    for c in cols:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    allowed_sizes = {"Mid Cap", "Small Cap", "Micro Cap"}

    filter = (d['Current EPS'] < 0) & d['Company Size'].isin(allowed_sizes)
    mask = filter if "Current EPS" in d.columns else pd.Series(True, index=d.index)

    comps = pd.DataFrame({
        "rdMcap":      d.loc[mask, "R&D / Market Cap Ratio"].rank(pct=True),
        "rdSales":      d.loc[mask, "R&D / Sales Ratio"].rank(pct=True),
        "EPS_Gap":     d.loc[mask, "EPS Growth Expectation"].rank(pct=True),
        "fwdEPS":     d.loc[mask, "Forward EPS"].rank(pct=True),
        "RevGrowth":   d.loc[mask, "Current Revenue Growth"].rank(pct=True)
    })

    comps = comps.fillna(0)

    weights = pd.Series({"fwdEPS":2.0, "RevGrowth":2.0, "EPS_Gap":4.0, "rdMcap":3.0, "rdSales": 1.5})

    # weighted average across available (non-NaN) components
    w = weights.reindex(comps.columns)
    weighted = comps.mul(w, axis=1)
    denom_w = comps.notna().mul(w, axis=1).sum(axis=1)
    score = weighted.sum(axis=1, skipna=True) / denom_w

    # --- min–max scale to 0–10 across the scored subset ---
    smin = score.min()
    smax = score.max()
    if pd.isna(smin) or pd.isna(smax) or smax == smin:
        score_scaled = pd.Series(np.nan, index=score.index)  # avoid 0/0
    else:
        score_scaled = (score - smin) / (smax - smin) * 10.0

    # align back to full df: non-allowed sizes remain NaN
    return score_scaled.reindex(df.index)

def fragility_score(df: pd.DataFrame) -> pd.Series:

    d = df.copy()

    # coerce to numeric to ensure .rank works
    cols = ["Insiders Holdings", "Forward P/E",
            "Interest Coverage Ratio", "Net Debt to EBITDA", 'Current Ratio'
            ]
    
    for c in cols:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    comps = pd.DataFrame({
        "IntCov":      d["Interest Coverage Ratio"].rank(pct=True, ascending=False),
        "DebtEbitda":  d["Net Debt to EBITDA"].rank(pct=True, ascending=True),
        "CurrentRatio": d["Current Ratio"].rank(pct=True, ascending=False),
        "InsHoldings": d["Insiders Holdings"].rank(pct=True, ascending=False)
    })


    comps = comps.fillna(0)

    weights = pd.Series({"IntCov":3.0, "GrossMargin":3.0, "DebtEbitda":1.5, "OCF_Ratio":2,
                         "CurrentRatio": 2, "InsHoldings": 1})

    # weighted average across available (non-NaN) components
    w = weights.reindex(comps.columns)
    weighted = comps.mul(w, axis=1)
    denom_w = comps.notna().mul(w, axis=1).sum(axis=1)
    score = weighted.sum(axis=1, skipna=True) / denom_w

    # --- min–max scale to 0–10 across the scored subset ---
    smin = score.min()
    smax = score.max()
    if pd.isna(smin) or pd.isna(smax) or smax == smin:
        score_scaled = pd.Series(np.nan, index=score.index)  # avoid 0/0
    else:
        score_scaled = (score - smin) / (smax - smin) * 10.0

    # align back to full df: non-allowed sizes remain NaN
    return score_scaled.reindex(df.index)


def ex_growth_score(df: pd.DataFrame) -> pd.Series:

    d = df.copy()

    # coerce to numeric to ensure .rank works
    cols = ["Current Earnings Growth","Net Debt to EBITDA","Current P/E","Forward P/E",
            "Gross Margin","Operating Cash Flow Ratio"]
    for c in cols:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    d["PE_Spread"] = d["Current P/E"] - d["Forward P/E"]

    allowed_sizes = {"Large Cap", "Mega Cap"}
    mask = d["Company Size"].isin(allowed_sizes) if "Company Size" in d.columns else pd.Series(True, index=d.index)

    comps = pd.DataFrame({
        "PE":          d.loc[mask, "Current P/E"].rank(pct=True, ascending=False),
        "PE_Spread":          d.loc[mask, "PE_Spread"].rank(pct=True, ascending=False),
        "GrossMargin": d.loc[mask, "Gross Margin"].rank(pct=True, ascending=False),
        "OCF_Ratio":   d.loc[mask, "Operating Cash Flow Ratio"].rank(pct=True, ascending=False)
    })

    comps = comps.fillna(0)

    weights = pd.Series({"PE":3.0, "PE_Spread":2.0, "GrossMargin":2, "OCF_Ratio":1.5})

    # weighted average across available (non-NaN) components
    w = weights.reindex(comps.columns)
    weighted = comps.mul(w, axis=1)
    denom_w = comps.notna().mul(w, axis=1).sum(axis=1)
    score = weighted.sum(axis=1, skipna=True) / denom_w

    # --- min–max scale to 0–10 across the scored subset ---
    smin = score.min()
    smax = score.max()
    if pd.isna(smin) or pd.isna(smax) or smax == smin:
        score_scaled = pd.Series(np.nan, index=score.index)  # avoid 0/0
    else:
        score_scaled = (score - smin) / (smax - smin) * 10.0

    # align back to full df: non-allowed sizes remain NaN
    return score_scaled.reindex(df.index)

import random

#Data Preprocessing
random_sample =True
raw_screener = pd.read_excel("C:/Users/User/Desktop/Data Projects/Trading/Tickers/all_usa.xlsx")
tickers = raw_screener['Ticker'].to_list()
tickers = random.sample(tickers, 15) if random_sample == True else tickers

raw_screener['Analyst Net Sentiment'] = raw_screener['% Rating Strong Buy or Buy'] - raw_screener['% Rating Strong Sell or Sell']
raw_screener['Upgrades - Downgrades'] = raw_screener['# Rating Upgrades'] - raw_screener['# Rating Downgrades ']
raw_screener['Next EPS Report Date'] = pd.to_datetime(raw_screener['Next EPS Report Date  (yyyymmdd)'], format='%Y%m%d')
raw_screener['Days to Earnings'] = (raw_screener['Next EPS Report Date'] - pd.Timestamp.today()).dt.days
raw_screener['ROA Growth'] = (raw_screener['Current ROA (TTM)'] - raw_screener['ROA (5 Yr Avg)']) / raw_screener['ROA (5 Yr Avg)'].abs()

keep_columns = raw_screener[['Ticker', 'Days to Earnings', 'Next EPS Report Date', 'Analyst Net Sentiment',
                              'Upgrades - Downgrades', '5 Yr. Hist. EPS Growth', '5 Yr Historical Sales Growth',
                              'ROA Growth']]

#Screener Excecution
us_stock_market = stock_screener(tickers)

mkt_cap_col = us_stock_market['Market Cap in Millions (USD)']

bins = [-np.inf, 50, 300, 2000, 10000, 200000, np.inf]  # millions USD
labels = ['Nano Cap', 'Micro Cap', 'Small Cap', 'Mid Cap', 'Large Cap', 'Mega Cap']

us_stock_market['Company Size'] = pd.cut(mkt_cap_col, bins=bins, labels=labels)

us_market_watchlist = pd.merge(us_stock_market, keep_columns, how='left', on='Ticker')

gics_to_naics = {
    # --- Healthcare & Pharma ---
    "Diagnostics & Research": "Health Care & Social Assistance",
    "Biotechnology": "Chemical Products",
    "Drug Manufacturers - Specialty & Generic": "Chemical Products",
    "Drug Manufacturers - General": "Chemical Products",
    "Medical Devices": "Health Care & Social Assistance",
    "Medical Care Facilities": "Health Care & Social Assistance",
    "Health Information Services": "Professional, Scientific & Technical Services",
    "Medical Distribution": "Wholesale Trade",
    "Medical Instruments & Supplies": "Health Care & Social Assistance",
    "Healthcare Plans": "Health Care & Social Assistance",
    "Pharmaceutical Retailers": "Retail Trade",

    # --- Metals & Mining ---
    "Aluminum": "Primary Metals",
    "Gold": "Mining",
    "Silver": "Mining",
    "Copper": "Primary Metals",
    "Steel": "Primary Metals",
    "Coking Coal": "Mining",
    "Thermal Coal": "Mining",
    "Uranium": "Mining",
    "Other Precious Metals & Mining": "Mining",
    "Other Industrial Metals & Mining": "Mining",
    "Metal Fabrication": "Fabricated Metal Products",

    # --- Energy / Petroleum ---
    "Oil & Gas E&P": "Petroleum & Coal Products",
    "Oil & Gas Equipment & Services": "Petroleum & Coal Products",
    "Oil & Gas Drilling": "Petroleum & Coal Products",
    "Oil & Gas Refining & Marketing": "Petroleum & Coal Products",
    "Oil & Gas Integrated": "Petroleum & Coal Products",
    "Oil & Gas Midstream": "Petroleum & Coal Products",
    "Solar": "Electrical Equipment, Appliances & Components",

    # --- Financials ---
    "Insurance - Life": "Finance & Insurance",
    "Insurance - Diversified": "Finance & Insurance",
    "Insurance - Property & Casualty": "Finance & Insurance",
    "Insurance - Specialty": "Finance & Insurance",
    "Insurance Brokers": "Finance & Insurance",
    "Insurance - Reinsurance": "Finance & Insurance",
    "Banks - Regional": "Finance & Insurance",
    "Banks - Diversified": "Finance & Insurance",
    "Credit Services": "Finance & Insurance",
    "Capital Markets": "Finance & Insurance",
    "Asset Management": "Finance & Insurance",
    "Financial Conglomerates": "Finance & Insurance",
    "Mortgage Finance": "Finance & Insurance",
    "Financial Data & Stock Exchanges": "Finance & Insurance",

    # --- Real Estate ---
    "REIT - Diversified": "Real Estate, Rental & Leasing",
    "REIT - Mortgage": "Real Estate, Rental & Leasing",
    "REIT - Retail": "Real Estate, Rental & Leasing",
    "REIT - Healthcare Facilities": "Real Estate, Rental & Leasing",
    "REIT - Residential": "Real Estate, Rental & Leasing",
    "REIT - Specialty": "Real Estate, Rental & Leasing",
    "REIT - Hotel & Motel": "Real Estate, Rental & Leasing",
    "REIT - Office": "Real Estate, Rental & Leasing",
    "REIT - Industrial": "Real Estate, Rental & Leasing",
    "Real Estate - Development": "Real Estate, Rental & Leasing",
    "Real Estate - Diversified": "Real Estate, Rental & Leasing",
    "Real Estate Services": "Real Estate, Rental & Leasing",

    # --- Industrials / Construction ---
    "Building Products & Equipment": "Construction",
    "Engineering & Construction": "Construction",
    "Building Materials": "Nonmetallic Mineral Products",
    "Residential Construction": "Construction",
    "Lumber & Wood Production": "Wood Products",
    "Specialty Industrial Machinery": "Machinery",
    "Industrial Distribution": "Wholesale Trade",
    "Farm & Heavy Construction Machinery": "Machinery",
    "Pollution & Treatment Controls": "Miscellaneous Manufacturing",
    "Infrastructure Operations": "Construction",
    "Conglomerates": "Management of Companies & Support Services",

    # --- Technology ---
    "Communication Equipment": "Computer & Electronic Products",
    "Semiconductors": "Computer & Electronic Products",
    "Semiconductor Equipment & Materials": "Computer & Electronic Products",
    "Consumer Electronics": "Computer & Electronic Products",
    "Software - Infrastructure": "Computer & Electronic Products",
    "Software - Application": "Computer & Electronic Products",
    "Computer Hardware": "Computer & Electronic Products",
    "Scientific & Technical Instruments": "Computer & Electronic Products",
    "Electronic Components": "Computer & Electronic Products",
    "Electronics & Computer Distribution": "Wholesale Trade",
    "Information Technology Services": "Professional, Scientific & Technical Services",

    # --- Transportation ---
    "Airlines": "Transportation & Warehousing",
    "Airports & Air Services": "Transportation & Warehousing",
    "Marine Shipping": "Transportation & Warehousing",
    "Railroads": "Transportation & Warehousing",
    "Trucking": "Transportation & Warehousing",
    "Integrated Freight & Logistics": "Transportation & Warehousing",
    "Auto Manufacturers": "Transportation Equipment",
    "Auto Parts": "Transportation Equipment",
    "Recreational Vehicles": "Transportation Equipment",
    "Auto & Truck Dealerships": "Retail Trade",

    # --- Consumer / Retail ---
    "Grocery Stores": "Retail Trade",
    "Department Stores": "Retail Trade",
    "Discount Stores": "Retail Trade",
    "Specialty Retail": "Retail Trade",
    "Apparel Retail": "Retail Trade",
    "Internet Retail": "Retail Trade",
    "Restaurants": "Accommodation & Food Services",
    "Lodging": "Accommodation & Food Services",
    "Resorts & Casinos": "Accommodation & Food Services",
    "Leisure": "Arts, Entertainment & Recreation",
    "Travel Services": "Accommodation & Food Services",
    "Luxury Goods": "Apparel, Leather & Allied Products",
    "Footwear & Accessories": "Apparel, Leather & Allied Products",
    "Apparel Manufacturing": "Apparel, Leather & Allied Products",
    "Textile Manufacturing": "Textile Mills",

    # --- Food / Beverages / Tobacco ---
    "Packaged Foods": "Food, Beverage & Tobacco Products",
    "Food Distribution": "Wholesale Trade",
    "Beverages - Brewers": "Food, Beverage & Tobacco Products",
    "Beverages - Non-Alcoholic": "Food, Beverage & Tobacco Products",
    "Beverages - Wineries & Distilleries": "Food, Beverage & Tobacco Products",
    "Tobacco": "Food, Beverage & Tobacco Products",
    "Confectioners": "Food, Beverage & Tobacco Products",

    # --- Utilities / Energy ---
    "Utilities - Regulated Electric": "Utilities",
    "Utilities - Diversified": "Utilities",
    "Utilities - Regulated Water": "Utilities",
    "Utilities - Regulated Gas": "Utilities",
    "Utilities - Renewable": "Utilities",
    "Utilities - Independent Power Producers": "Utilities",

    # --- Media / Information ---
    "Entertainment": "Arts, Entertainment & Recreation",
    "Electronic Gaming & Multimedia": "Arts, Entertainment & Recreation",
    "Broadcasting": "Information",
    "Publishing": "Information",
    "Internet Content & Information": "Information",
    "Advertising Agencies": "Professional, Scientific & Technical Services",
    "Specialty Business Services": "Professional, Scientific & Technical Services",
    "Consulting Services": "Professional, Scientific & Technical Services",
    "Staffing & Employment Services": "Management of Companies & Support Services",
    "Education & Training Services": "Educational Services",

    # --- Miscellaneous ---
    "Business Equipment & Supplies": "Miscellaneous Manufacturing",
    "Personal Services": "Other Services",
    "Rental & Leasing Services": "Real Estate, Rental & Leasing",
    "Security & Protection Services": "Other Services",
    "Waste Management": "Other Services",
    "Gambling": "Arts, Entertainment & Recreation",
    "Shell Companies": "Management of Companies & Support Services",
    "Tools & Accessories": "Fabricated Metal Products",
    "Furnishings, Fixtures & Appliances": "Furniture & Related Products",
    "Household & Personal Products": "Miscellaneous Manufacturing",
    "Paper & Paper Products": "Paper Products",
    "Packaging & Containers": "Plastics & Rubber Products",
}

us_market_watchlist["NAICS Industry"] = (
    us_market_watchlist["Industry"].map(gics_to_naics)
)


# ============================================================
# 1. PARSE FF49 FROM HUMAN-READABLE TEXT
# ============================================================

def parse_ff49(path):
    """
    Reads the giant Fama-French 49 human-readable SIC mapping and
    converts it into a clean DataFrame with:
    - sic_from
    - sic_to
    - industry
    """
    with open(path, encoding="utf-8") as f:
        text = f.read()

    lines = text.split("\n")

    current_industry = None
    rows = []

    # Header example: "37 Chips Electronic Equipment"
    industry_header_pattern = re.compile(r"^\s*\d+\s+([A-Za-z0-9]+)")

    # SIC range example: "3670-3679 Electronic components..."
    sic_range_pattern = re.compile(r"^\s*(\d{4})-(\d{4})")

    for line in lines:
        # Detect FF49 industry header
        header = industry_header_pattern.match(line)
        if header:
            current_industry = header.group(1)  # e.g. "Chips"
            continue

        # Detect SIC range under that header
        match = sic_range_pattern.match(line)
        if match and current_industry:
            sic_from = int(match.group(1))
            sic_to = int(match.group(2))
            rows.append((sic_from, sic_to, current_industry))

    df = pd.DataFrame(rows, columns=["sic_from", "sic_to", "industry"])
    return df


# Load FF49 Table Once
FF49_TABLE = parse_ff49(
    r"C:/Users/User/Desktop/Data Projects/Python Scripts/Siccodes49.txt"
)


def ff49_from_sic(sic):
    """
    Returns FF49 industry for a SIC code.
    """
    if sic is None:
        return None
    sic = int(sic)

    match = FF49_TABLE[(FF49_TABLE.sic_from <= sic) & (FF49_TABLE.sic_to >= sic)]
    if len(match):
        return match.iloc[0]["industry"]
    return None


# ============================================================
# 2. GET SEC INFO (SIC, NAICS)
# ============================================================

def get_sec_metadata(ticker, email="research@example.com"):
    # Ticker → CIK mapping from SEC
    url = "https://www.sec.gov/files/company_tickers.json"
    headers = {"User-Agent": email}
    mapping = requests.get(url, headers=headers).json()

    ticker = ticker.upper().strip()

    cik = None
    for entry in mapping.values():
        if entry["ticker"] == ticker:
            cik = str(entry["cik_str"]).zfill(10)
            break

    if cik is None:
        raise ValueError(f"Ticker {ticker} not found in SEC database")

    # SEC submissions endpoint
    submissions = f"https://data.sec.gov/submissions/CIK{cik}.json"
    data = requests.get(submissions, headers=headers).json()

    return {
        "CIK": cik,
        "Company Name": data.get("name"),
        "SIC": data.get("sic"),
        "SIC Description": data.get("sicDescription"),
        }


# ============================================================
# 3. GICS via YFinance
# ============================================================

def get_gics(ticker):
    info = yf.Ticker(ticker).info
    return {
        "GICS Sector": info.get("sector"),
        "GICS Industry": info.get("industry"),
    }


# ============================================================
# 4. MASTER FUNCTION — FULL CLASSIFICATION
# ============================================================

def get_full_classification(ticker, email="research@example.com"):
    """
    Returns full classification:
    - SEC: SIC, NAICS
    - GICS: sector + industry
    - FF49: academic industry
    """
    s = get_sec_metadata(ticker, email=email)
    g = get_gics(ticker)

    # Add FF49 classification
    s["FF49 Industry"] = ff49_from_sic(s["SIC"])

    # Merge SEC + GICS
    result = {**{"Ticker": ticker}, **s, **g}

    return result


# ============================================================
# TEST
# ============================================================

# ============================================================
# 5. BATCH VERSION — RETURNS CLEAN DATAFRAME
# ============================================================

def classify_tickers(ticker_list, email="research@example.com"):
    """
    Takes a list of tickers and returns a DataFrame with:
    Ticker, CIK, SIC, FF49, NAICS, GICS Sector/Industry
    """
    records = []

    for t in ticker_list:
        try:
            info = get_full_classification(t, email=email)
            records.append(info)
        except Exception as e:
            records.append({
                "Ticker": t,
                "CIK": None,
                "Company Name": None,
                "SIC": None,
                "SIC Description": None,
                "FF49 Industry": None,
                "GICS Sector": None,
                "GICS Industry": None,
                "Error": str(e)
            })

    return pd.DataFrame(records)

df = classify_tickers(tickers)

df = df[['Ticker', 'FF49 Industry']]

us_market_watchlist = pd.merge(us_market_watchlist, df, on='Ticker', how='left')

us_market = us_market_watchlist

us_market["Growth Score"] = (
    us_market.groupby("NAICS Industry", group_keys=False)
             .apply(growth_score)
)

us_market["Efficiency Score"] = (
    us_market.groupby("NAICS Industry", group_keys=False)
             .apply(efficency_score)
)

us_market["Pain Score"] = pain_trade(us_market)

us_market["Venture Score"] = (
    us_market.groupby("NAICS Industry", group_keys=False)
             .apply(promise_score)
)   

us_market["Fragility Score"] = (
    us_market.groupby("NAICS Industry", group_keys=False)
             .apply(fragility_score)
)

us_market["Ex-Growth Score"] = (
    us_market.groupby("NAICS Industry", group_keys=False)
             .apply(ex_growth_score)
)

us_market["Value Score"] = value_score(us_market)

market_regimes = pd.read_csv('market_regimes.csv', index_col=0)
industries_regimes = pd.read_csv('Stock Selection/industries_regimes.csv', index_col=0)

current_regime = str(market_regimes.iloc[-1].unique()[0])

best_industries = list((industries_regimes[(industries_regimes[current_regime] > np.percentile(industries_regimes[current_regime], 70)) | (industries_regimes[current_regime]>0.8)]).sort_values(by=current_regime, ascending=False).index)
worst_industries = list(industries_regimes[(industries_regimes[current_regime] < np.percentile(industries_regimes[current_regime], 30)) | (industries_regimes[current_regime]<0.5)].index)

us_market[us_market['FF49 Industry'].isin(best_industries)]
us_market[us_market['FF49 Industry'].isin(worst_industries)]

# us_market.to_excel(
#     r"C:/Users/User/Desktop/Data Projects/Python Scripts/Stock Screener"
#     rf"\us_stock_market_watchlist2_{pd.Timestamp.today().strftime('%Y-%m')}.xlsx"
# )

