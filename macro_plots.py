import datetime
from datetime import timedelta
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from functions import *
from portopt import *

import streamlit as st

import yfinance as yf

from matplotlib.colors import LinearSegmentedColormap

def plot_macro_quadrants(
    df: pd.DataFrame,
    gdp_col: str = "gdp_yoy_diff",
    cpi_col: str = "cpi_yoy_diff",
    n_quarters: int = 10,
    title: str = "United States Macro Quadrants",
) -> go.Figure:
    """
    Plot the trajectory of GDP YoY acceleration versus CPI YoY acceleration.

    Horizontal axis
        CPI YoY first difference.

    Vertical axis
        GDP YoY first difference.

    Notes
    -----
    The chart uses symmetric x and y ranges, so all four quadrants have
    identical dimensions.
    
    """

    # gdp = pd.read_pickle("ml_models/monthly_gdp_nowcast.pkl")['monthly_gdp_proxy_yoy'].rename("gdp_yoy")
    # cpi = pd.read_pickle("ml_models/monthly_cpi_nowcast.pkl")['monthly_cpi_proxy_yoy'].rename("cpi_yoy")

    # df = pd.DataFrame({
    #     "gdp": gdp,
    #     "cpi": cpi,
    #     "gdp_diff": gdp.diff().rename("gdp_yoy_diff"),
    #     "cpi_diff": cpi.diff().rename("cpi_yoy_diff"),

    # })

    # gdp_col = "gdp_diff"
    # cpi_col = "cpi_diff"

    # n_quarters = 6
    
    required_columns = {gdp_col, cpi_col}
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    plot_df = (
        df[[gdp_col, cpi_col]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .sort_index()
        .tail(n_quarters)
        .copy()
    )

    if plot_df.empty:
        raise ValueError("No valid observations available to plot.")

    if isinstance(plot_df.index, pd.PeriodIndex):
        plot_df["label"] = plot_df.index.astype(str)

    elif isinstance(plot_df.index, pd.DatetimeIndex):
        plot_df["label"] = (
            plot_df.index
            .to_period("Q")
            .astype(str)
        )

    else:
        plot_df["label"] = plot_df.index.astype(str)

    # Assumes values are stored as decimal changes.
    # Example: 0.0025 = 25 basis points.
    plot_df["gdp_bps"] = plot_df[gdp_col] * 10_000
    plot_df["cpi_bps"] = plot_df[cpi_col] * 10_000

    # Use the same absolute limit for both axes.
    max_abs_value = max(
        plot_df["gdp_bps"].abs().max(),
        plot_df["cpi_bps"].abs().max(),
    )

    if max_abs_value == 0:
        max_abs_value = 10

    axis_padding = max(max_abs_value * 0.15, 10)
    axis_limit = max_abs_value + axis_padding

    x_range = [-axis_limit, axis_limit]
    y_range = [-axis_limit, axis_limit]

    latest = plot_df.iloc[-1]

    fig = go.Figure()

    quadrant_colors = {
        "q1": "rgba(0, 200, 120, 0.12)",
        "q2": "rgba(140, 200, 0, 0.12)",
        "q3": "rgba(255, 140, 0, 0.12)",
        "q4": "rgba(220, 50, 70, 0.12)",
    }

    # Quadrant backgrounds.
    fig.add_shape(
        type="rect",
        x0=x_range[0],
        x1=0,
        y0=0,
        y1=y_range[1],
        fillcolor=quadrant_colors["q1"],
        line_width=0,
        layer="below",
    )

    fig.add_shape(
        type="rect",
        x0=0,
        x1=x_range[1],
        y0=0,
        y1=y_range[1],
        fillcolor=quadrant_colors["q2"],
        line_width=0,
        layer="below",
    )

    fig.add_shape(
        type="rect",
        x0=0,
        x1=x_range[1],
        y0=y_range[0],
        y1=0,
        fillcolor=quadrant_colors["q3"],
        line_width=0,
        layer="below",
    )

    fig.add_shape(
        type="rect",
        x0=x_range[0],
        x1=0,
        y0=y_range[0],
        y1=0,
        fillcolor=quadrant_colors["q4"],
        line_width=0,
        layer="below",
    )

    # Historical trajectory.
    fig.add_trace(
        go.Scatter(
            x=plot_df["cpi_bps"],
            y=plot_df["gdp_bps"],
            mode="lines+markers+text",
            text=plot_df["label"],
            textposition="top center",
            customdata=plot_df[
                [gdp_col, cpi_col]
            ].to_numpy(),
            line=dict(
                width=3,
                color="#31BEFF",
            ),
            marker=dict(
                size=11,
                color="#E0A80E",
                line=dict(
                    width=1.5,
                    color="#D7DCFF",
                ),
            ),
            textfont=dict(
                size=11,
                color="#D9E1F2",
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "CPI YoY acceleration: %{customdata[1]:.2%}<br>"
                "GDP YoY acceleration: %{customdata[0]:.2%}<br>"
                "CPI change: %{x:.0f} bps<br>"
                "GDP change: %{y:.0f} bps"
                "<extra></extra>"
            ),
            name="Macro trajectory",
        )
    )

    # Highlight latest quarter.
    fig.add_trace(
        go.Scatter(
            x=[latest["cpi_bps"]],
            y=[latest["gdp_bps"]],
            mode="markers",
            marker=dict(
                size=21,
                symbol="circle-open",
                color="#FF6B4A",
                line=dict(
                    width=4,
                    color="#FF6B4A",
                ),
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # Zero axes.
    fig.add_vline(
        x=0,
        line_width=1.5,
        line_color="rgba(255,255,255,0.85)",
    )

    fig.add_hline(
        y=0,
        line_width=1.5,
        line_color="rgba(255,255,255,0.85)",
    )

    annotations = [
        {
            "x": -axis_limit * 0.65,
            "y": axis_limit * 0.72,
            "text": (
                "<b>Quad 1: Goldilocks</b><br>"
                "Growth accelerating<br>"
                "Inflation decelerating"
            ),
            "color": "#4DDBA0",
        },
        {
            "x": axis_limit * 0.65,
            "y": axis_limit * 0.72,
            "text": (
                "<b>Quad 2: Reflation</b><br>"
                "Growth accelerating<br>"
                "Inflation accelerating"
            ),
            "color": "#B9E769",
        },
        {
            "x": axis_limit * 0.65,
            "y": -axis_limit * 0.72,
            "text": (
                "<b>Quad 3: Stagflation</b><br>"
                "Growth decelerating<br>"
                "Inflation accelerating"
            ),
            "color": "#FFB45F",
        },
        {
            "x": -axis_limit * 0.65,
            "y": -axis_limit * 0.72,
            "text": (
                "<b>Quad 4: Deflation</b><br>"
                "Growth decelerating<br>"
                "Inflation decelerating"
            ),
            "color": "#FF7F91",
        },
    ]

    for annotation in annotations:
        fig.add_annotation(
            x=annotation["x"],
            y=annotation["y"],
            text=annotation["text"],
            showarrow=False,
            align="center",
            opacity=0.9,
            font=dict(
                size=12,
                color=annotation["color"],
            ),
        )

    fig.update_layout(
        title={
            "text": (
                f"<b>{title}</b><br>"
                "<sup>"
                "x-axis: first difference of YoY CPI; "
                "y-axis: first difference of YoY real GDP"
                "</sup>"
            ),
            "x": 0.5,
            "xanchor": "center",
        },
        template="plotly_dark",
        height=760,
        hovermode="closest",
        showlegend=False,
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        margin=dict(
            l=90,
            r=60,
            t=110,
            b=85,
        ),
        font=dict(
            color="#D9E1F2",
        ),
    )

    fig.update_xaxes(
        title="CPI YoY acceleration",
        range=x_range,
        ticksuffix=" bps",
        zeroline=False,
        showgrid=True,
        gridcolor="rgba(255,255,255,0.10)",
        linecolor="rgba(255,255,255,0.30)",
        mirror=False,
        constrain="domain",
    )

    fig.update_yaxes(
        title="GDP YoY acceleration",
        range=y_range,
        ticksuffix=" bps",
        zeroline=False,
        showgrid=True,
        gridcolor="rgba(255,255,255,0.10)",
        linecolor="rgba(255,255,255,0.30)",
        scaleanchor="x",
        scaleratio=1,
        constrain="domain",
    )

    return fig


from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots


def plot_macro_indicator(
    level: pd.Series,
    level_z: pd.Series,
    yoy_z: pd.Series,
    title: str | None = None,
    level_label: str | None = None,
    yoy_label: str = "YoY Z-Score",
    height: int = 750,
    z_range: tuple[float, float] = (-2.5, 2.5),
) -> go.Figure:
    """
    Plot a macroeconomic indicator with:

    1. A line chart whose segment colors depend on the indicator's z-score:
       red = low, yellow = neutral, green = high.
    2. A lower bar chart showing the YoY z-score:
       green above zero and red below zero.

    Parameters
    ----------
    level
        Original macroeconomic indicator.
    level_z
        Z-score of the indicator level.
    yoy_z
        Z-score of the indicator's year-over-year change.
    title
        Figure title.
    level_label
        Label used for the main chart's y-axis.
    yoy_label
        Label used for the lower chart's y-axis.
    height
        Figure height in pixels.
    z_range
        Minimum and maximum z-scores used to normalize line colors.

    Returns
    -------
    plotly.graph_objects.Figure
    """

    level = level.rename(level.name or "Indicator")
    level_z = level_z.rename(level_z.name or "Level Z-Score")
    yoy_z = yoy_z.rename(yoy_z.name or "YoY Z-Score")

    data = pd.concat(
        [level, level_z, yoy_z],
        axis=1,
    ).sort_index()

    data.columns = ["level", "level_z", "yoy_z"]

    main_data = data[["level", "level_z"]].dropna()
    bar_data = data[["yoy_z"]].dropna()

    if main_data.empty:
        raise ValueError("No valid observations are available for the main chart.")

    if bar_data.empty:
        raise ValueError("No valid observations are available for the YoY bar chart.")

    figure_title = title or level.name
    level_axis_label = level_label or level.name

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.70, 0.30],
    )

    z_min, z_max = z_range

    # Add one trace per line segment because Plotly does not support
    # changing the color continuously within a standard line trace.
    for i in range(len(main_data) - 1):
        current = main_data.iloc[i]
        following = main_data.iloc[i + 1]

        segment_z = np.nanmean(
            [current["level_z"], following["level_z"]]
        )

        normalized_z = np.clip(
            (segment_z - z_min) / (z_max - z_min),
            0,
            1,
        )

        segment_color = sample_colorscale(
            "RdYlGn",
            [normalized_z],
        )[0]

        fig.add_trace(
            go.Scatter(
                x=[
                    main_data.index[i],
                    main_data.index[i + 1],
                ],
                y=[
                    current["level"],
                    following["level"],
                ],
                mode="lines",
                line={
                    "color": segment_color,
                    "width": 3,
                },
                customdata=np.array(
                    [
                        [current["level_z"]],
                        [following["level_z"]],
                    ]
                ),
                hovertemplate=(
                    "%{x|%b %Y}<br>"
                    f"{level_axis_label}: %{{y:,.2f}}<br>"
                    "Level Z-Score: %{customdata[0]:+.2f}"
                    "<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    # Invisible markers provide consistent hover information at every point.
    fig.add_trace(
        go.Scatter(
            x=main_data.index,
            y=main_data["level"],
            mode="markers",
            marker={
                "size": 8,
                "color": "rgba(0,0,0,0)",
            },
            customdata=main_data["level_z"],
            hovertemplate=(
                "%{x|%b %Y}<br>"
                f"{level_axis_label}: %{{y:,.2f}}<br>"
                "Level Z-Score: %{customdata:+.2f}"
                "<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    bar_colors = np.where(
        bar_data["yoy_z"] >= 0,
        "#39FF14",  # lime green
        "#FF3131",  # red
    )

    fig.add_trace(
        go.Bar(
            x=bar_data.index,
            y=bar_data["yoy_z"],
            marker={
                "color": bar_colors,
                "line": {"width": 0},
            },
            hovertemplate=(
                "%{x|%b %Y}<br>"
                f"{yoy_label}: %{{y:+.2f}}"
                "<extra></extra>"
            ),
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    fig.add_hline(
        y=0,
        line_width=1,
        line_color="rgba(255,255,255,0.55)",
        row=2,
        col=1,
    )

    fig.update_layout(
        title={
            "text": figure_title,
            "x": 0.02,
            "xanchor": "left",
        },
        template="plotly_dark",
        height=height,
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        margin={
            "l": 70,
            "r": 30,
            "t": 80,
            "b": 50,
        },
        hovermode="x unified",
        bargap=0.08,
    )

    fig.update_yaxes(
        title_text=level_axis_label,
        gridcolor="rgba(255,255,255,0.10)",
        zeroline=False,
        row=1,
        col=1,
    )

    fig.update_yaxes(
        title_text=yoy_label,
        gridcolor="rgba(255,255,255,0.10)",
        zeroline=False,
        row=2,
        col=1,
    )

    fig.update_xaxes(
        showgrid=False,
        rangeslider_visible=False,
        row=1,
        col=1,
    )

    fig.update_xaxes(
        title_text="Date",
        showgrid=False,
        rangeslider_visible=False,
        row=2,
        col=1,
    )

    return fig

def plot_portfolio_weights(
    weights: pd.Series,
    title: str = "Optimized Portfolio Weights",
) -> go.Figure:

    plot_weights = (
        weights
        .dropna()
        .sort_values(ascending=True)
        .copy()
    )

    max_abs_weight = plot_weights.abs().max()
    axis_limit = max_abs_weight * 1.20

    fig = go.Figure()

    # Short-exposure background.
    fig.add_vrect(
        x0=-axis_limit,
        x1=0,
        fillcolor="rgba(220, 50, 70, 0.18)",
        line_width=0,
        layer="below",
    )

    # Long-exposure background.
    fig.add_vrect(
        x0=0,
        x1=axis_limit,
        fillcolor="rgba(0, 200, 120, 0.18)",
        line_width=0,
        layer="below",
    )

    # Cyan portfolio bars.
    fig.add_trace(
        go.Bar(
            x=plot_weights.values,
            y=plot_weights.index,
            orientation="h",
            marker=dict(
                color="#31BEFF",
                line=dict(
                    color="#D7F3FF",
                    width=1,
                ),
            ),
            text=[
                f"{weight:+.1%}"
                for weight in plot_weights.values
            ],
            textposition="outside",
            cliponaxis=False,
            customdata=np.abs(plot_weights.values),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Weight: %{x:+.2%}<br>"
                "Absolute exposure: %{customdata:.2%}"
                "<extra></extra>"
            ),
        )
    )

    # Central zero line.
    fig.add_vline(
        x=0,
        line_width=2,
        line_color="rgba(255,255,255,0.90)",
    )

    # Exposure labels.
    fig.add_annotation(
        x=-axis_limit * 0.70,
        y=1.08,
        xref="x",
        yref="paper",
        text="<b>SHORT EXPOSURE</b>",
        showarrow=False,
        font=dict(
            size=13,
            color="#FF7F91",
        ),
    )

    fig.add_annotation(
        x=axis_limit * 0.70,
        y=1.08,
        xref="x",
        yref="paper",
        text="<b>LONG EXPOSURE</b>",
        showarrow=False,
        font=dict(
            size=13,
            color="#4DDBA0",
        ),
    )

    fig.update_layout(
        title={
            "text": f"<b>{title}</b>",
            "x": 0.5,
            "xanchor": "center",
        },
        template="plotly_dark",
        height=550,
        showlegend=False,
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        margin=dict(
            l=80,
            r=90,
            t=100,
            b=65,
        ),
        bargap=0.25,
        font=dict(
            color="#D9E1F2",
        ),
    )

    fig.update_xaxes(
        title="Portfolio weight",
        range=[-axis_limit, axis_limit],
        tickformat=".0%",
        zeroline=False,
        showgrid=True,
        gridcolor="rgba(255,255,255,0.10)",
    )

    fig.update_yaxes(
        title=None,
        showgrid=False,
        automargin=True,
    )

    return fig



def get_macro_nowcast_quadrants():

    """
    Quad 1: GDP up, CPI down
    Quad 2: GDP up, CPI up
    Quad 3: GDP down, CPI up
    Quad 4: GDP down, CPI down

    """
    gdp_yoy = pd.read_pickle("ml_models/monthly_gdp_nowcast.pkl")['monthly_gdp_proxy_yoy'].rename("gdp_yoy")
    cpi_yoy = pd.read_pickle("ml_models/monthly_cpi_nowcast.pkl")['monthly_cpi_proxy_yoy'].rename("cpi_yoy")


    economy = pd.concat({
        "gdp_yoy": gdp_yoy,
        "cpi_yoy": cpi_yoy,
    }, axis=1).dropna()

    # z-scores
    economy["gdp_yoy_z"] = zscore(economy["gdp_yoy"], window=80)
    economy["cpi_yoy_z"] = zscore(economy["cpi_yoy"], window=80)
    
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


    # strength
    economy["quad_strength_yoy"] = np.hypot(
        economy["gdp_yoy"].diff(),
        economy["cpi_yoy"].diff()
    )

    economy["quad_strength_yoy_percentile"] = economy["quad_strength_yoy"].rank(pct=True)

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

    
    # economy["regime_qoq_duration"] = (
    #     economy["regime_qoq"]
    #     .ne(economy["regime_qoq"].shift())
    #     .cumsum()
    # )

    # economy["regime_qoq_duration"] = (
    #     economy
    #     .groupby("regime_qoq_duration")
    #     .cumcount() + 1
    # )

    economy = economy.dropna()

    return economy



gdp = pd.read_pickle("ml_models/monthly_gdp_nowcast.pkl")['monthly_gdp_proxy_yoy'].rename("gdp_yoy")
cpi = pd.read_pickle("ml_models/monthly_cpi_nowcast.pkl")['monthly_cpi_proxy_yoy'].rename("cpi_yoy")

quad_data = pd.DataFrame({
    "gdp": gdp,
    "cpi": cpi,
    "gdp_diff": gdp.diff().rename("gdp_yoy_diff"),
    "cpi_diff": cpi.diff().rename("cpi_yoy_diff"),

}).resample("QE").last().dropna()

plot_macro_quadrants(
    df=quad_data.dropna(),
    gdp_col="gdp_diff",
    cpi_col="cpi_diff",
    n_quarters=5,
)


macro_quadrants = get_macro_nowcast_quadrants()

industry_returns = pd.read_pickle("data/industry_returns.pkl")

daily_rf = pd.read_pickle("data/raw_money_market_data.pkl")['10y_yield'].rename("3m_tbill").resample("D").last().ffill() / 100 / 252

industry_sortino = get_rolling_sortino_ratio(
    returns_df=industry_returns,
    rf_series=daily_rf.reindex(industry_returns.index, method='ffill'),
    window=252)

industry_tail_ratio = rolling_tail_ratio(
    df=industry_returns,
    window=252)

industry_score = industry_sortino * 0.7 - industry_tail_ratio * 0.3
industry_vol_pct = industry_returns.rolling(252).std().rank(pct=True)

industry_score['quadrant'] = macro_quadrants['regime_yoy'].reindex(industry_sortino.index, method='ffill')

num_assets = 5

ind_by_quad = {1: {"Best Industries in QUAD 1": industry_score.groupby('quadrant').median().T[1.0].sort_values(ascending=False).head(num_assets),
                    "Worst Industries in QUAD 1": industry_score.groupby('quadrant').median().T[1.0].sort_values(ascending=True).head(num_assets)
                    },
                2: {"Best Industries in QUAD 2": industry_score.groupby('quadrant').median().T[2.0].sort_values(ascending=False).head(num_assets),
                    "Worst Industries in QUAD 2": industry_score.groupby('quadrant').median().T[2.0].sort_values(ascending=True).head(num_assets)
                },
                3: {"Best Industries in QUAD 3": industry_score.groupby('quadrant').median().T[3.0].sort_values(ascending=False).head(num_assets),
                    "Worst Industries in QUAD 3": industry_score.groupby('quadrant').median().T[3.0].sort_values(ascending=True).head(num_assets)
                },
                4: {"Best Industries in QUAD 4": industry_score.groupby('quadrant').median().T[4.0].sort_values(ascending=False).head(num_assets),
                    "Worst Industries in QUAD 4": industry_score.groupby('quadrant').median().T[4.0].sort_values(ascending=True).head(num_assets)
                }
                }

current_quad = int((macro_quadrants['regime_yoy'].resample("QE").last().tail(1).iloc[0]))

long_assets = ind_by_quad[current_quad][f"Best Industries in QUAD {current_quad}"].index.tolist()
short_assets = ind_by_quad[current_quad][f"Worst Industries in QUAD {current_quad}"].index.tolist()

benchmark = pd.read_pickle("data/sp500_returns.pkl")
benchmark_vol = benchmark.rolling(252).std().rank(pct=True, ascending=False)

nL = len(long_assets)
nS = len(short_assets)

breadth_signal = (nL - nS) / (nL + nS)

regime_bias = {0:  0.25,
                1: -0.25,
                2:  0.25,
                3: -0.10
                }

net_target = np.clip(1.5 * breadth_signal + regime_bias[current_quad], -1.0, 1.0)

gross_target = dynamic_gross_target(
                                    spx_returns=benchmark,   
                                    start_date=datetime.datetime.now() - datetime.timedelta(days=252),
                                    vol_window=63,
                                    target_vol=0.3,
                                    base_gross=2,
                                    min_gross=0.75,
                                    max_gross=3
                                    )
                        


opt_portfolio = (
    convex_downside_risk_budgeting_optimizer(
        returns_df=industry_returns,
        long_list=long_assets,
        short_list=short_assets,
        gross_target=3.0,
        net_target=0.25,
        max_position=0.5,
        min_long_position=0.05,
        min_short_position=0.04,
        l2_penalty=1e-3,
    )
)

weights = opt_portfolio["weights"]

#####

long_weights_norm = (
    weights.reindex(long_assets)
    .fillna(0)
    .clip(lower=0)
)

long_weights_norm /= long_weights_norm.sum()

short_weights_norm = (
    weights.reindex(short_assets)
    .fillna(0)
    .abs()
)

short_weights_norm /= short_weights_norm.sum()

cross_corr = industry_returns[long_assets].corrwith(
    industry_returns[short_assets]
)

corr_matrix = industry_returns[
    long_assets + short_assets
].corr()

cross_corr_matrix = corr_matrix.loc[
    long_assets,
    short_assets,
]

weight_matrix = np.outer(
    long_weights_norm.values,
    short_weights_norm.values,
)

weighted_average_cross_corr = np.sum(
    cross_corr_matrix.values * weight_matrix
)


####

long_returns = industry_returns[long_assets].dot(weights[long_assets])
short_returns = industry_returns[short_assets].dot(weights[short_assets])

portfolio_returns = long_returns + short_returns

port_volatility = (long_returns + short_returns).rolling(252).std().tail(1).iloc[0] * np.sqrt(252)

port_covariance = industry_returns[long_assets + short_assets].cov().dot(weights[long_assets + short_assets])
port_variance = port_covariance.dot(weights[long_assets + short_assets])

long_short_correlation = long_returns.cov(short_returns)

long_weights = weights.reindex(long_assets).fillna(0.0)
short_weights = weights.reindex(short_assets).fillna(0.0)

# Normalize within each side so each book sums to 100% in absolute terms
long_weights_normalized = (
    long_weights / long_weights.sum()
)

short_weights_normalized = (
    short_weights
    / short_weights.sum()
)

long_book_returns = (
    industry_returns[long_weights_normalized.index]
    @ long_weights_normalized
)

# Return of the underlying assets in the short book
short_underlying_returns = (
    industry_returns[short_weights_normalized.index]
    @ short_weights_normalized
)

long_short_correlation = long_book_returns.corr(
    short_underlying_returns
)


weights = opt_portfolio["weights"]

long_weights = (
    weights.reindex(long_assets)
    .fillna(0.0)
)

short_weights_abs = (
    weights.reindex(short_assets)
    .fillna(0.0)
    .abs()
)

long_weights_norm = (
    long_weights / long_weights.sum()
)

short_weights_norm = (
    short_weights_abs
    / short_weights_abs.sum()
)

long_returns = (
    industry_returns[long_assets]
    @ long_weights_norm
)

short_underlying_returns = (
    industry_returns[short_assets]
    @ short_weights_norm
)

aligned = pd.concat(
    [
        long_returns.rename("long"),
        short_underlying_returns.rename("short"),
        benchmark.rename("market"),
    ],
    axis=1,
).dropna()

long_beta = (
    aligned["long"].cov(aligned["market"])
    / aligned["market"].var()
)

short_underlying_beta = (
    aligned["short"].cov(aligned["market"])
    / aligned["market"].var()
)

port_beta = float((long_beta * opt_portfolio['long_target']) - (short_underlying_beta * opt_portfolio['short_target']))


# Dashboart Title

st.set_page_config(layout="wide")

st.markdown(
    """
    <style>
    .section-divider {
        border: none;
        border-top: 2px solid #31BEFF;
        margin: 1.2rem 0 1.4rem 0;
        opacity: 0.85;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Macroeconomic Regime-Based Industry Portfolio Optimization")

st.markdown(
    """
     The goal for this project is to deploy an ML algorithm that can predict macroeconomic regimes
     and use this information to construct a portfolio of industry ETFs that is optimized for the current 
     macroeconomic environment. The portfolio will be rebalanced on a quarterly basis, and the weights will be 
     determined by the predicted macroeconomic regime using ***CVXPY optimization*** techniques.

     ***This is not financial advice. This is a research project and should not be used for investment purposes.***
    """
)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

mcol1, mcol2, mcol3, mcol4 = st.columns(4)

quad_names = {
    1: "Goldilocks",
    2: "Reflation",
    3: "Stagflation",
    4: "Deflation",
}

quad_name = quad_names.get(current_quad, "Unknown")

mcol1.metric("Current Macro Regime", f"QUAD {current_quad}")
mcol2.metric("Macro Regime Name", quad_name)
mcol3.metric("GDP expected growth YoY", f"{gdp.iloc[-1]:.2%}")
mcol4.metric("CPI expected growth YoY", f"{cpi.iloc[-1]:.2%}")

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

top_cols = st.columns(2)


with top_cols[0]:
    st.subheader("Current Macro Regime")
    st.plotly_chart(
        plot_macro_quadrants(
            df=quad_data.dropna(),
            gdp_col="gdp_diff",
            cpi_col="cpi_diff",
            n_quarters=5,
        ),
        use_container_width=True,
    )

with top_cols[1]:
    st.subheader("Best Industries in Current Macro Regime")
    st.plotly_chart(
        plot_portfolio_weights(
            weights=weights,
            title="Optimized Portfolio Weights",
        ),
        use_container_width=True,
    )

    port_metrics_cols = st.columns(4)

    port_metrics_cols[0].metric("Portfolio Volatility (Annualized)", f"{port_volatility:.2%}")
    port_metrics_cols[1].metric("Portfolio Beta", f"{port_beta:.2f}")
    port_metrics_cols[2].metric("Gross Exposure", f"{weights.abs().sum():.2f}")
    port_metrics_cols[3].metric("Net Exposure", f"{weights.sum():.2f}")


currency_data = pd.read_pickle("data/currency_prices.pkl").interpolate()
currency_returns= currency_data.pct_change().dropna()
currency_score = get_rolling_sortino_ratio(currency_returns, rf_series=daily_rf.reindex(currency_returns.index, method='ffill'), window=252, periods_per_year=252)
currency_score['quadrants'] = macro_quadrants['regime_yoy'].reindex(currency_score.index, method='ffill')

bonds_data = pd.read_pickle("data/bond_etfs.pkl")
bond_returns = bonds_data.pct_change().dropna()
bond_score = get_rolling_sortino_ratio(bond_returns, rf_series=daily_rf.reindex(bond_returns.index, method='ffill'), window=252, periods_per_year=252)
bond_score['quadrants'] = macro_quadrants['regime_yoy'].reindex(bond_score.index, method='ffill')

industry_data = pd.read_pickle("data/industry_returns.pkl")
industry_returns = industry_data.pct_change().dropna()
industry_score = get_rolling_sortino_ratio(industry_returns, rf_series=daily_rf.reindex(industry_returns.index, method='ffill'), window=252, periods_per_year=252)
industry_score['quadrants'] = macro_quadrants['regime_yoy'].reindex(industry_score.index, method='ffill')

factors_data = pd.read_pickle("data/factors_etfs.pkl")
factors_returns = factors_data.pct_change().dropna()
factors_score = get_rolling_sortino_ratio(factors_returns, rf_series=daily_rf.reindex(factors_returns.index, method='ffill'), window=252, periods_per_year=252)
factors_score['quadrants'] = macro_quadrants['regime_yoy'].reindex(factors_score.index, method='ffill')

current_quad = int((macro_quadrants['regime_yoy'].reindex(factors_score.index, method='ffill').tail(1).iloc[0]))

best_regimes = pd.DataFrame({
    "Best Currencies": currency_score.groupby('quadrants').median().T[current_quad].sort_values(ascending=False).head(5).index.tolist(),
    "Best Bonds": bond_score.groupby('quadrants').median().T[current_quad].sort_values(ascending=False).head(5).index.tolist(),
    "Best Industries": long_assets,
    "Best Factors": factors_score.groupby('quadrants').median().T[current_quad].sort_values(ascending=False).head(5).index.tolist(),
})

worst_regimes = pd.DataFrame({
    "Worst Currencies": currency_score.groupby('quadrants').median().T[current_quad].sort_values(ascending=True).head(5).index.tolist(),
    "Worst Bonds": bond_score.groupby('quadrants').median().T[current_quad].sort_values(ascending=True).head(5).index.tolist(),
    "Worst Industries": short_assets,
    "Worst Factors": factors_score.groupby('quadrants').median().T[current_quad].sort_values(ascending=True).head(5).index.tolist(),
})

regime_cols = st.columns(2)

with regime_cols[0]:
    st.subheader("Best Performing Assets in Current Macro Regime")
    st.dataframe(
        best_regimes,
        use_container_width=True,
        hide_index=True
    )

with regime_cols[1]:
    st.subheader("Worst Performing Assets in Current Macro Regime")
    st.dataframe(
        worst_regimes,
        use_container_width=True,
        hide_index=True
    )


sector_data = pd.read_pickle("data/sector_etfs.pkl")

spy = yf.download('SPY', start='1900-01-01')['Close'].squeeze().rename("spy")

relative_strength = sector_data.div(spy, axis=0)
factor_data = pd.read_pickle("data/factors_etfs.pkl")

def get_dashboard(sector_data: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare sector ETF data for dashboard display.
    """
    rf = pd.read_pickle("data/daily_rf.pkl")
    sector_data = sector_data.stack().to_frame(name="Last Price").round(2)
    sector_data.index.names = ["Date", "Sector ETF"]
    sector_data['Current %Change'] = sector_data.groupby("Sector ETF")["Last Price"].pct_change().round(4)
    sector_data["MoM %Change"] = sector_data.groupby("Sector ETF")["Last Price"].pct_change(21).round(4)
    sector_data["3-month %Change"] = sector_data.groupby("Sector ETF")["Last Price"].pct_change(63).round(4)
    sector_data["YoY %Change"] = sector_data.groupby("Sector ETF")["Last Price"].pct_change(252).round(4)
    sector_data['RAR'] = (
        sector_data.groupby("Sector ETF")["Last Price"]
        .apply(lambda x: rolling_score(x.pct_change().dropna(), window=252).rank(pct=True))
        .round(4)
    ).droplevel(0)

    return sector_data.groupby("Sector ETF").tail(1).droplevel(0)

stock_screener = pd.read_excel("stock_screener/us_stock_market_watchlist2 2025-11.xlsx")

long_ideas = stock_screener.loc[stock_screener['FF49 Industry'].isin(long_assets)].copy()

short_ideas = stock_screener.loc[stock_screener['FF49 Industry'].isin(short_assets)].copy()

sector_dash = get_dashboard(sector_data)
relative_dash = get_dashboard(relative_strength)
factors_dash = get_dashboard(factor_data)

dash_cols = st.columns(2)

risk_cmap = LinearSegmentedColormap.from_list(
    "risk_scale",
    [
        "#8B0000",  # lowest: dark red
        "#FF3B30",  # red
        "#FF8C00",  # orange
        "#FFD700",  # yellow
        "#006400",  # dark green
        "#32CD32",  # highest: lime green
    ],
)

performance_columns = [
    "Current %Change",
    "MoM %Change",
    "3-month %Change",
    "YoY %Change",
    "RAR",
]

styled_sector_dash = (
    sector_dash.style
    .format({
        "Last Price": "${:,.2f}",
        "Current %Change": "{:+.2%}",
        "MoM %Change": "{:+.2%}",
        "3-month %Change": "{:+.2%}",
        "YoY %Change": "{:+.2%}",
        "RAR": "{:.1%}",
    })
    .background_gradient(
        cmap=risk_cmap,
        subset=performance_columns,
        axis=0,
    )
    .set_properties(
        subset=performance_columns,
        **{
            "color": "white",
            "font-weight": "600",
        },
    )
)

with dash_cols[0]:
    st.subheader("Sector ETFs Dashboard")

    st.dataframe(
        styled_sector_dash,
        use_container_width=True,
    )

with dash_cols[1]:
    st.subheader("Factors ETFs Dashboard")

    styled_factors_dash = (
        factors_dash.style
        .format({
            "Last Price": "${:,.2f}",
            "Current %Change": "{:+.2%}",
            "MoM %Change": "{:+.2%}",
            "3-month %Change": "{:+.2%}",
            "YoY %Change": "{:+.2%}",
            "RAR": "{:.1%}",
        })
        .background_gradient(
            cmap=risk_cmap,
            subset=performance_columns,
            axis=0,
            text_color_threshold=0.55,
        )
    )

    st.dataframe(
        styled_factors_dash,
        use_container_width=True,
    )

macro_cols = st.columns(2)

macro_raw_data = pd.read_pickle("data/raw_macro_data.pkl")

building_permits = (
    macro_raw_data["building_permits"]
    .rename("Building Permits")
)

building_permits_z = (
    (
        building_permits - building_permits.mean()
    )
    / building_permits.std()
).rename("Building Permits Z-Score")

building_permits_yoy = (
    building_permits
    .pct_change(12)
    .rename("Building Permits YoY %Change")
)

building_permits_yoy_z = (
    (
        building_permits_yoy - building_permits_yoy.mean()
    )
    / building_permits_yoy.std()
).rename("Building Permits YoY Z-Score")


ism_manufacturing = (
    macro_raw_data["ism_manufacturing_index"]
    .rename("ISM Manufacturing Index")
)

ism_manufacturing_z = (
    (
        ism_manufacturing - ism_manufacturing.mean()
    )
    / ism_manufacturing.std()
).rename("ISM Manufacturing Index Z-Score")

ism_manufacturing_momentum = (
    momentum(ism_manufacturing, fast_window=3,
             slow_window=6,
             signal_window=3,
             zscore_window=120)
)

building_permits_fig = plot_macro_indicator(
    level=building_permits.iloc[-360:],
    level_z=building_permits_z.iloc[-360:],
    yoy_z=building_permits_yoy_z.iloc[-360:],
    title="US Building Permits",
    level_label="Building Permits",
    yoy_label="Building Permits YoY Z-Score",
)

ism_manufacturing_fig = plot_macro_indicator(
    level=ism_manufacturing.iloc[-360:],
    level_z=ism_manufacturing_z.iloc[-360:],
    yoy_z=ism_manufacturing_momentum.iloc[-360:],
    title="ISM Manufacturing Index",
    level_label="ISM Manufacturing Index",
    yoy_label="ISM Manufacturing Momentum",
)

with macro_cols[0]:
    st.subheader("Key Macro Drivers")
    st.plotly_chart(
        building_permits_fig,
        use_container_width=True,
    )

with macro_cols[1]:
    st.subheader("      ")
    st.plotly_chart(
        ism_manufacturing_fig,
        use_container_width=True,
    )

b_macro_cols = st.columns(2)

consumer_sentiment = macro_raw_data['consumer_sentiment'].rolling(3).mean().rename("Consumer Sentiment Index")
consumer_sentiment_z = (consumer_sentiment - consumer_sentiment.mean()) / consumer_sentiment.std()
consumer_sentiment_momentum = momentum(consumer_sentiment, fast_window=3, slow_window=6, signal_window=3, zscore_window=120)

consumer_fig = plot_macro_indicator(
    level=consumer_sentiment.iloc[-360:],
    level_z=consumer_sentiment_z.iloc[-360:],
    yoy_z=consumer_sentiment_momentum.iloc[-360:],
    title="Consumer Sentiment Index",
    level_label="Consumer Sentiment Index",
    yoy_label="Consumer Sentiment Momentum",
)

financial_conditions = macro_raw_data['financial_conditions_index'].rolling(3).mean().rename("Financial Conditions Index")
financial_conditions_z = (zscore(financial_conditions, window=120)) * -1
financial_conditions_momentum = momentum(financial_conditions, fast_window=3, slow_window=6, signal_window=3, zscore_window=120)

with b_macro_cols[0]:

    st.plotly_chart(
        consumer_fig,
        use_container_width=True,
    )

with b_macro_cols[1]:

    st.plotly_chart(
        plot_macro_indicator(
            level=financial_conditions.iloc[-260:],
            level_z=financial_conditions_z.iloc[-260:],
            yoy_z=financial_conditions_momentum.iloc[-260:],
            title="Financial Conditions Index",
            level_label="Financial Conditions Index",
            yoy_label="Financial Conditions Momentum",
        ),
        use_container_width=True,
    )
