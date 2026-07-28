import pandas as pd
import cvxpy as cp
import numpy as np


def compute_downside_covariance(
    returns_df: pd.DataFrame,
    mar: float = 0.0,
    annualize: bool = False,
    periods_per_year: int = 252,
    shrink_diag: float = 1e-6,
) -> pd.DataFrame:
    """
    Compute a downside covariance matrix using returns below MAR.
    """

    if returns_df is None or returns_df.empty:
        raise ValueError("returns_df is empty")

    returns = (
        returns_df
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .dropna(how="any")
    )

    if returns.empty:
        raise ValueError(
            "No valid observations remain after cleaning returns"
        )

    # Downside deviations relative to MAR
    downside = returns.sub(mar).clip(upper=0.0)

    sigma_down = downside.cov()

    if annualize:
        sigma_down = (
            sigma_down * periods_per_year
        )

    # Convert to an explicitly writable NumPy array
    sigma_values = sigma_down.to_numpy(
        dtype=float,
        copy=True,
    )

    # Symmetrize
    sigma_values = 0.5 * (
        sigma_values + sigma_values.T
    )

    # Add diagonal regularization without touching
    # a potentially read-only DataFrame view
    sigma_values += (
        np.eye(
            sigma_values.shape[0],
            dtype=float,
        )
        * shrink_diag
    )

    return pd.DataFrame(
        sigma_values,
        index=sigma_down.index,
        columns=sigma_down.columns,
    )

def convex_downside_risk_budgeting_optimizer(
    returns_df: pd.DataFrame,
    long_list: list[str],
    short_list: list[str],
    gross_target: float = 2.0,
    net_target: float = 0.0,
    max_position: float = 0.20,
    min_long_position: float = 0.0,
    min_short_position: float = 0.0,
    mar: float = 0.0,
    annualize_cov: bool = False,
    periods_per_year: int = 252,
    l2_penalty: float = 0.0,
    turnover_penalty: float = 0.0,
    prev_weights: pd.Series | None = None,
    solver=cp.SCS,
) -> dict:
    """
    Convex long/short minimum downside-variance optimizer.

    min_long_position
        Minimum weight allocated to each long asset.
        Example: 0.05 means every long must be at least +5%.

    min_short_position
        Minimum absolute weight allocated to each short asset.
        Example: 0.05 means every short must be at least -5%.
    """

    if returns_df is None or returns_df.empty:
        return {
            "status": "empty_returns",
            "weights": None,
            "gross": None,
            "net": None,
            "downside_variance": None,
            "long_target": None,
            "short_target": None,
        }

    # --------------------------------------------------------
    # Universe
    # --------------------------------------------------------

    long_assets = [
        asset
        for asset in long_list
        if asset in returns_df.columns
    ]

    short_assets = [
        asset
        for asset in short_list
        if asset in returns_df.columns
    ]

    overlap = set(long_assets).intersection(short_assets)

    if overlap:
        long_assets = [
            asset
            for asset in long_assets
            if asset not in overlap
        ]

        short_assets = [
            asset
            for asset in short_assets
            if asset not in overlap
        ]

    n_long = len(long_assets)
    n_short = len(short_assets)

    if n_long == 0 or n_short == 0:
        return {
            "status": "invalid_universe",
            "weights": None,
            "gross": None,
            "net": None,
            "downside_variance": None,
            "long_target": None,
            "short_target": None,
        }

    # --------------------------------------------------------
    # Validate position limits
    # --------------------------------------------------------

    if max_position <= 0:
        raise ValueError(
            "max_position must be greater than zero."
        )

    if min_long_position < 0:
        raise ValueError(
            "min_long_position cannot be negative."
        )

    if min_short_position < 0:
        raise ValueError(
            "min_short_position cannot be negative."
        )

    if min_long_position > max_position:
        raise ValueError(
            "min_long_position cannot exceed max_position."
        )

    if min_short_position > max_position:
        raise ValueError(
            "min_short_position cannot exceed max_position."
        )

    # --------------------------------------------------------
    # Exposure targets
    # --------------------------------------------------------

    long_target = 0.5 * (
        gross_target + net_target
    )

    short_target = 0.5 * (
        gross_target - net_target
    )

    if long_target < 0 or short_target < 0:
        return {
            "status": "invalid_targets",
            "weights": None,
            "gross": None,
            "net": None,
            "downside_variance": None,
            "long_target": long_target,
            "short_target": short_target,
        }

    # Feasible exposure intervals
    minimum_long_exposure = (
        n_long * min_long_position
    )

    maximum_long_exposure = (
        n_long * max_position
    )

    minimum_short_exposure = (
        n_short * min_short_position
    )

    maximum_short_exposure = (
        n_short * max_position
    )

    if not (
        minimum_long_exposure
        <= long_target
        <= maximum_long_exposure
    ):
        return {
            "status": "infeasible_long_target",
            "weights": None,
            "gross": None,
            "net": None,
            "downside_variance": None,
            "long_target": long_target,
            "short_target": short_target,
            "minimum_long_exposure": minimum_long_exposure,
            "maximum_long_exposure": maximum_long_exposure,
        }

    if not (
        minimum_short_exposure
        <= short_target
        <= maximum_short_exposure
    ):
        return {
            "status": "infeasible_short_target",
            "weights": None,
            "gross": None,
            "net": None,
            "downside_variance": None,
            "long_target": long_target,
            "short_target": short_target,
            "minimum_short_exposure": minimum_short_exposure,
            "maximum_short_exposure": maximum_short_exposure,
        }

    # --------------------------------------------------------
    # Returns and covariance
    # --------------------------------------------------------

    selected_assets = (
        long_assets + short_assets
    )

    aligned_returns = (
        returns_df[selected_assets]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna(how="any")
    )

    if aligned_returns.empty:
        return {
            "status": "no_data_after_alignment",
            "weights": None,
            "gross": None,
            "net": None,
            "downside_variance": None,
            "long_target": long_target,
            "short_target": short_target,
        }

    sigma_down = compute_downside_covariance(
        aligned_returns,
        mar=mar,
        annualize=annualize_cov,
        periods_per_year=periods_per_year,
        shrink_diag=1e-6,
    )

    Sigma = sigma_down.to_numpy(
        dtype=float,
        copy=True,
    )

    Sigma = 0.5 * (
        Sigma + Sigma.T
    )

    minimum_eigenvalue = (
        np.linalg.eigvalsh(Sigma).min()
    )

    if minimum_eigenvalue < 0:
        Sigma += np.eye(len(selected_assets)) * (
            abs(minimum_eigenvalue) + 1e-8
        )

    # --------------------------------------------------------
    # Decision variables
    # --------------------------------------------------------

    wL = cp.Variable(
        n_long,
        nonneg=True,
    )

    wS = cp.Variable(
        n_short,
        nonneg=True,
    )

    signed_weights = cp.hstack(
        [
            wL,
            -wS,
        ]
    )

    # --------------------------------------------------------
    # Constraints
    # --------------------------------------------------------

    constraints = [
        wL >= min_long_position,
        wL <= max_position,
        wS >= min_short_position,
        wS <= max_position,
        cp.sum(wL) == long_target,
        cp.sum(wS) == short_target,
    ]

    # --------------------------------------------------------
    # Objective
    # --------------------------------------------------------

    objective_terms = [
        cp.quad_form(
            signed_weights,
            cp.psd_wrap(Sigma),
        )
    ]

    if l2_penalty > 0:
        objective_terms.append(
            l2_penalty
            * cp.sum_squares(
                signed_weights
            )
        )

    if (
        turnover_penalty > 0
        and prev_weights is not None
    ):
        previous = (
            prev_weights
            .reindex(selected_assets)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )

        objective_terms.append(
            turnover_penalty
            * cp.sum_squares(
                signed_weights - previous
            )
        )

    objective = cp.Minimize(
        cp.sum(objective_terms)
    )

    problem = cp.Problem(
        objective,
        constraints,
    )

    problem.solve(
        solver=solver,
    )

    if (
        wL.value is None
        or wS.value is None
    ):
        return {
            "status": problem.status,
            "weights": None,
            "gross": None,
            "net": None,
            "downside_variance": None,
            "long_target": long_target,
            "short_target": short_target,
        }

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    weights = pd.Series(
        np.concatenate(
            [
                wL.value,
                -wS.value,
            ]
        ),
        index=selected_assets,
        name="weight",
    )

    weights = weights.where(
        weights.abs() > 1e-6,
        0.0,
    )

    downside_variance = float(
        weights.to_numpy()
        @ Sigma
        @ weights.to_numpy()
    )

    weights_pct = (
        weights
        .mul(100)
        .round(2)
        .rename("weight_pct")
    )

    return {
        "status": problem.status,
        "weights": weights,
        "weights_pct": weights_pct,
        "gross": float(
            weights.abs().sum()
        ),
        "net": float(
            weights.sum()
        ),
        "gross_pct": round(
            float(weights.abs().sum()) * 100,
            2,
        ),
        "net_pct": round(
            float(weights.sum()) * 100,
            2,
        ),
        "downside_variance": downside_variance,
        "long_target": float(long_target),
        "short_target": float(short_target),
        "long_target_pct": round(
            float(long_target) * 100,
            2,
        ),
        "short_target_pct": round(
            float(short_target) * 100,
            2,
        ),
        "min_long_position": float(
            min_long_position
        ),
        "min_short_position": float(
            min_short_position
        ),
        "max_position": float(
            max_position
        ),
        "long_assets": long_assets,
        "short_assets": short_assets,
        "selected_assets": selected_assets,
        "downside_covariance": sigma_down,
    }

def dynamic_gross_target(
    spx_returns,
    start_date,
    vol_window=63,
    target_vol=0.18,
    base_gross=2.0,
    min_gross=0.75,
    max_gross=2.5
):
    """
    Gross exposure scales inversely with SPX realized volatility.

    Higher realized vol  -> lower gross
    Lower realized vol   -> higher gross
    """

    realized_vol = (
        spx_returns
        .loc[:start_date]
        .dropna()
        .tail(vol_window)
        .std()
        * np.sqrt(252)
    )

    if pd.isna(realized_vol) or realized_vol == 0:
        return base_gross, np.nan

    gross = base_gross * (target_vol / realized_vol)
    gross = np.clip(gross, min_gross, max_gross)

    return gross
