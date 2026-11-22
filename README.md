# Graph Neural Networks for Cross-Asset Alpha Prediction

Research-grade, reproducible Python benchmark comparing **GCN**, **graph attention (GAT)**, **node-wise MLP**, **Ridge**, and **momentum** for predicting next-day **cross-sectional equity returns** using strictly causal correlation graphs.

**Research question:** Does graph message passing provide incremental predictive value over a model using the exact same per-asset features without edges? The benchmark is designed to permit a negative answer.

## Features

- A synthetic eight-asset, two-sector regime-switching factor market with weak lagged sector spillovers, or user-supplied split-adjusted closing prices.
- Causal **rolling top-|correlation| signed graph**. At decision close `t`, edge weights use only returns through `t`, with symmetric neighbor selection and L1 row normalization. A new graph is created at every decision date; no full-sample correlation matrix is used.
- Per-asset features: last return, five- and twenty-day return momentum, twenty-day volatility, and same-day market-relative return. Training-only standardization.
- Compare zero forecast (prediction benchmark only), twenty-day momentum, pooled Ridge, shared node MLP, signed two-layer GCN, and signed masked-attention GAT. GNNs are implemented directly in PyTorch without PyTorch Geometric binary dependencies.
- Chronological 60% train, 20% validation, 20% test splits based on **target dates**. Model parameters fit on training only; validation loss chooses neural checkpoints; no selection based on test results.
- Next-day MSE, MAE, cross-sectional daily Spearman rank IC, directional accuracy, model parameter counts, and a dollar-neutral, 2x-gross long-short **research** backtest with turnover costs.
- Automated numerical, causal-data, model-gradient, portfolio-accounting, reproducibility, and integration tests, plus GitHub Actions.

## Setup

Use Python 3.10+.

```bash
python -m pip install -e '.[test]'
python -m pytest -q
gnn-alpha --days 700 --assets 8 --epochs 16 --output results
```

Small CPU smoke experiment:

```bash
gnn-alpha --days 260 --assets 8 --epochs 2 --hidden 12 --graph-window 30 --output results-smoke
```

To supply **point-in-time adjusted daily closing prices**, prepare a CSV with `date` plus at least four asset price columns. Include delisted assets and handle corporate actions appropriately in serious studies.

```csv
date,AAA,BBB,CCC,DDD
2023-01-03,100.0,105.0,91.0,120.0
2023-01-04,100.2,104.5,90.8,121.0
```

Then run:

```bash
gnn-alpha --csv adjusted_panel.csv --graph-window 40 --neighbors 3 --output market-results
```

## Outputs

| File | Contents |
|---|---|
| `metrics.json` | Chronological split sizes, test/validation forecasting and strategy statistics, parameter counts, epochs |
| `predictions.csv` | Per-asset target date, realized next-day simple return, forecast, and model |
| `portfolio_returns.csv` | Date, model, gross/net returns, turnover, cumulative wealth |
| `test_mse.png` | Held-out forecasting MSE comparison |
| `test_equity.png` | Illustrative held-out long-short cumulative wealth |
| `graph_snapshot.png` | Graph using the **last training** sample, not the full series |

## Research methodology and leakage audit

1. At decision close `t`, observe adjusted prices through `t` and calculate features using historical returns only.
2. Estimate top-m correlation neighbors using the last `graph_window` observations ending at `t`; zero unsupported edges, keep negative correlations, include self loops.
3. Predict simple return from close `t` to close `t+1`; each target is associated with its **future target date** for the split.
4. Fit feature standardization and Ridge on training examples only. Fit neural weights on training examples; use validation for early stopping.
5. Freeze the models and evaluate test dates once. Forecasts rank assets; the top `k` are bought and bottom `k` sold short, with +1 and -1 aggregate weights (gross 2x).
6. Deduct `cost_bps / 10000 * sum(|w_t-w_(t-1)|)` and calculate portfolio growth using **simple** daily returns.

Train/validation/test are chronological **but** consecutive input feature windows can overlap at split boundaries; historical inputs from earlier periods are valid information and do not make future labels available. This is a single holdout research demonstration, not repeated walk-forward refitting or significance-tested production alpha.

### Research limitations

**Same-close execution is not implementable after observing the closing price.** Close-`t` features predict close-`t+1` returns but execution must occur at a later executable price, such as next open, or via a properly timed market-on-close submission. Thus reported close-to-close strategies are **research upper bounds**, not live P&L. The toy cost model omits securities borrowing costs, financing, spread dynamics, impact and partial fills. Synthetic markets and sector construction are assumed. Results do not establish alpha. Real market studies require survival-bias-free universes, rebalancing constraints, realistic costs, repeated chronological testing, hyperparameter control, and uncertainty intervals. Correlation does not imply an economic exposure that will persist.

## Model design

- **Shared node MLP:** one network predicts each asset independently from features.
- **GCN:** two signed message-passing graph convolutions and a linear node head.
- **GAT:** two layers of attention masked to historically estimated graph edges, weighted by historical absolute edge strengths, with the sign used in message aggregation.

The only intended architectural difference between MLP and GNNs is the ability to aggregate historically linked asset information. Different architectures have different parameter counts, reported in `metrics.json`.

## License

MIT. See `LICENSE`.
