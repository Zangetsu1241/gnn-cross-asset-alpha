"""Chronological synthetic panels, strict market data, and causal graph snapshots.

A sample at decision close t uses only prices through t and predicts the simple
return from close t to close t+1. Training membership depends on target date.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class GraphDataset:
    features: np.ndarray  # (samples, assets, features), train-normalized
    graphs: np.ndarray    # (samples, assets, assets), signed L1 row normalization
    targets: np.ndarray   # (samples, assets), next close-to-close simple returns
    decision_dates: pd.DatetimeIndex
    target_dates: pd.DatetimeIndex
    splits: dict
    asset_names: list
    scaler_mean: np.ndarray
    scaler_std: np.ndarray
    raw_features: np.ndarray
    raw_returns: np.ndarray
    target_positions: np.ndarray


def synthetic_prices(days=800, assets=8, seed=42):
    """Regime-switching correlated factor market with weak lagged sector spillover.

    Assets are divided into two sectors, sharing a common market shock and
    an AR(1) sector factor; this does not ensure GNN outperformance.
    """
    if days < 180 or assets < 4 or assets > 64:
        raise ValueError("Need days >= 180 and assets in [4, 64]")
    rng = np.random.default_rng(seed)
    sector_id = np.arange(assets) % 2
    market = 0.
    sector_factor = np.zeros(2)
    previous = np.zeros(assets)
    returns = np.empty((days, assets))
    state = 0
    beta = rng.uniform(.65, 1.2, assets)
    idiosyncratic = rng.uniform(.0035, .007, assets)
    for t in range(days):
        if rng.random() < .016:
            state = 1 - state
        market = .24 * market + rng.normal(scale=.006 if state == 0 else .017)
        sector_factor = .36 * sector_factor + rng.normal(
            scale=.004 if state == 0 else .010, size=2)
        drift = (.00023 if state == 0 else -.00035)
        spillover = np.array([previous[sector_id == s].mean() for s in range(2)])
        returns[t] = (drift + beta * market + sector_factor[sector_id]
                      + .08 * spillover[sector_id]
                      + rng.normal(scale=idiosyncratic * (1 + state), size=assets))
        previous = returns[t].copy()
    close = 100 * np.exp(np.cumsum(returns, axis=0))
    dates = pd.bdate_range("2018-01-01", periods=days)
    return pd.DataFrame(close, index=dates, columns=[f"asset_{i+1:02d}" for i in range(assets)])


def validate_prices(prices):
    if not isinstance(prices, pd.DataFrame) or prices.shape[1] < 4 or len(prices) < 180:
        raise ValueError("Need 180+ daily observations and 4+ assets")
    if prices.columns.duplicated().any() or prices.index.isna().any():
        raise ValueError("Duplicate asset columns or missing dates")
    if prices.index.duplicated().any() or not prices.index.is_monotonic_increasing:
        raise ValueError("Dates must be unique and strictly increasing")
    try:
        matrix = prices.to_numpy(dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError("Prices must be numeric") from exc
    if not np.isfinite(matrix).all() or np.any(matrix <= 0):
        raise ValueError("All adjusted closing prices must be positive and finite")
    return matrix


def load_prices(csv_path):
    frame = pd.read_csv(csv_path)
    if "date" not in frame.columns or len(frame.columns) < 5:
        raise ValueError("CSV requires date and at least 4 adjusted close columns")
    dates = pd.to_datetime(frame.pop("date"), utc=True, errors="raise")
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("Dates must be nonmissing, strictly ascending and unique")
    numeric = frame.apply(pd.to_numeric, errors="raise")
    numeric.index = pd.DatetimeIndex(dates)
    validate_prices(numeric)
    return numeric


def make_graph(returns_window, neighbors=3):
    """Signed, symmetrized top-|correlation| graph with L1 row normalization.

    Neighbor selection uses a *single historical trailing window*. Signs and
    magnitudes of correlations are retained. Self loops are always present.
    """
    x = np.asarray(returns_window, dtype=float)
    if x.ndim != 2 or x.shape[0] < 10 or x.shape[1] < 4 or not np.isfinite(x).all():
        raise ValueError("Need finite returns window of shape (>=10, >=4)")
    n = x.shape[1]
    if not isinstance(neighbors, (int, np.integer)) or isinstance(neighbors, bool) or not 1 <= neighbors < n:
        raise ValueError("neighbors must be integer in [1, n_assets-1]")
    corr = np.nan_to_num(np.corrcoef(x, rowvar=False), nan=0., posinf=0., neginf=0.)
    corr = np.clip(corr, -1., 1.)
    np.fill_diagonal(corr, 0.)
    # Stable tie breaking avoids non-reproducible sorting of zero correlations.
    picked = np.zeros((n, n), dtype=bool)
    for i in range(n):
        candidates = np.argsort(-np.abs(corr[i]), kind="stable")
        candidates = candidates[candidates != i][:neighbors]
        picked[i, candidates] = True
    mask = picked | picked.T
    weighted = np.where(mask, corr, 0.)
    np.fill_diagonal(weighted, 1.)
    weighted /= np.abs(weighted).sum(axis=1, keepdims=True)
    return weighted.astype("float32")


def _causal_features(raw_returns, t):
    """All statistics are defined using observed return rows with indices <=t."""
    recent = raw_returns[t-19:t+1]
    momentum5 = raw_returns[t-4:t+1].sum(axis=0)
    momentum20 = recent.sum(axis=0)
    vol20 = recent.std(axis=0, ddof=1)
    relative = raw_returns[t] - raw_returns[t].mean()
    return np.stack([raw_returns[t], momentum5, momentum20, vol20, relative], axis=-1)


def build_dataset(prices, graph_window=40, neighbors=3,
                  train_fraction=.6, val_fraction=.2):
    arr = validate_prices(prices)
    n_days, n_assets = arr.shape
    if (graph_window < 20 or not isinstance(graph_window, int) or
        not .45 <= train_fraction <= .75 or not .1 <= val_fraction <= .3 or
        train_fraction + val_fraction >= .9):
        raise ValueError("Invalid graph_window or split fractions")
    if neighbors >= n_assets or neighbors < 1:
        raise ValueError("Invalid number of graph neighbors")
    r = np.empty_like(arr)
    r[0] = np.nan
    r[1:] = arr[1:] / arr[:-1] - 1
    start = max(20, graph_window)
    times = np.arange(start, n_days-1)
    if len(times) < 60:
        raise ValueError("Need enough windows and target rows")
    x = np.stack([_causal_features(r, int(t)) for t in times])
    adj = np.stack([make_graph(r[t-graph_window+1:t+1], neighbors) for t in times])
    targets = np.stack([r[t+1] for t in times]).astype("float32")
    target_positions = times + 1
    train_end = int(n_days * train_fraction)
    val_end = int(n_days * (train_fraction + val_fraction))
    splits = {
        "train": np.flatnonzero(target_positions < train_end),
        "validation": np.flatnonzero((target_positions >= train_end) & (target_positions < val_end)),
        "test": np.flatnonzero(target_positions >= val_end),
    }
    if any(len(v) < 20 for v in splits.values()):
        raise ValueError("One of the chronological splits has <20 dates")
    train_x = x[splits["train"]]
    mean = train_x.mean(axis=(0,1))
    std = train_x.std(axis=(0,1))
    std = np.where(std < 1e-10, 1., std)
    normalized = ((x-mean)/std).astype("float32")
    return GraphDataset(
        features=normalized, graphs=adj, targets=targets,
        decision_dates=prices.index[times], target_dates=prices.index[times+1],
        splits=splits, asset_names=list(prices.columns), scaler_mean=mean,
        scaler_std=std, raw_features=x.astype("float32"),
        raw_returns=r, target_positions=target_positions,
    )
