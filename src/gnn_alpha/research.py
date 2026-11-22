"""Comparable chronological out-of-sample cross-sectional forecasting experiment."""
from dataclasses import dataclass
from copy import deepcopy
import random
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr
from .data import build_dataset
from .models import NEURAL_MODELS


@dataclass(frozen=True)
class ResearchConfig:
    graph_window: int = 40
    neighbors: int = 3
    train_fraction: float = .60
    val_fraction: float = .20
    epochs: int = 16
    patience: int = 6
    hidden: int = 32
    batch_size: int = 64
    learning_rate: float = .002
    ridge_alpha: float = 10.
    long_short: int = 2
    cost_bps: float = 5.
    seed: int = 42
    torch_threads: int = 1


def cross_sectional_rank_ic(y_true, y_pred):
    """Per-date Spearman rank correlation, excluding constant vectors."""
    truth, pred = np.asarray(y_true,float), np.asarray(y_pred,float)
    if truth.shape != pred.shape or truth.ndim != 2 or not np.isfinite(truth).all() or not np.isfinite(pred).all():
        raise ValueError("Expected finite same-shape date-by-asset arrays")
    values = []
    for observed, forecast in zip(truth, pred):
        if np.ptp(observed) <= 1e-12 or np.ptp(forecast) <= 1e-12:
            continue
        corr = spearmanr(observed, forecast).statistic
        if np.isfinite(corr):
            values.append(float(corr))
    return float(np.mean(values)) if values else None


def long_short_weights(predictions, k=2):
    """Dollar neutral, gross exposure 2, 1/k short and long per selection."""
    p = np.asarray(predictions, float)
    if p.ndim != 2 or not np.isfinite(p).all() or not isinstance(k,int) or k < 1 or 2*k > p.shape[1]:
        raise ValueError("Need finite predictions with at least 2*k assets")
    weights = np.zeros_like(p)
    for t in range(len(p)):
        if np.ptp(p[t]) < 1e-12:
            continue  # No signal => no position; do not exploit arbitrary tie breaking.
        order = np.argsort(p[t], kind="stable")
        weights[t, order[:k]] = -1/k
        weights[t, order[-k:]] = 1/k
    return weights


def simulate_strategy(predictions, realized_returns, cost_bps=5., k=2):
    """Each signal sets gross-two holdings at decision close t for return t+1.

    Execution at the known close after it is observed is not achievable; these
    are research signals, not executable trading profits.
    """
    realized = np.asarray(realized_returns, dtype=float)
    if (realized.shape != np.asarray(predictions).shape or not np.isfinite(realized).all() or
        np.any(realized <= -1) or not np.isfinite(cost_bps) or cost_bps < 0):
        raise ValueError("Misaligned returns or invalid cost_bps")
    weights = long_short_weights(predictions,k)
    previous = np.vstack([np.zeros((1,weights.shape[1])),weights[:-1]])
    turnover = np.abs(weights-previous).sum(axis=1)
    gross = np.sum(weights * realized, axis=1)
    net = gross - turnover*cost_bps/10000
    if np.any(net <= -1):
        raise ValueError("Net strategy return <= -100%")
    wealth = np.cumprod(1+net)
    peaks = np.maximum.accumulate(np.r_[1.,wealth])[1:]
    sd = float(net.std(ddof=1)) if len(net)>1 else 0.
    stats = {"total_return":float(wealth[-1]-1),
             "annualized_return":float(wealth[-1]**(252/len(net))-1),
             "annualized_volatility":float(sd*np.sqrt(252)),
             "sharpe":float(net.mean()/sd*np.sqrt(252)) if sd>0 else 0.,
             "max_drawdown":float(np.min(wealth/peaks-1)),
             "turnover_units":float(turnover.sum()),
             "mean_gross_exposure":float(np.mean(np.abs(weights).sum(axis=1)))}
    return stats, pd.DataFrame({"gross_return":gross,"net_return":net,
                                "turnover":turnover,"wealth":wealth}), weights


def _train_network(name, dataset, cfg, model_seed):
    torch.manual_seed(model_seed)
    model = NEURAL_MODELS[name](dataset.features.shape[-1], cfg.hidden)
    optimizer = torch.optim.AdamW(model.parameters(),lr=cfg.learning_rate,weight_decay=1e-4)
    x = torch.from_numpy(dataset.features)
    a = torch.from_numpy(dataset.graphs)
    target = torch.from_numpy(dataset.targets * 100.)
    train = dataset.splits["train"]
    val = dataset.splits["validation"]
    rng = np.random.default_rng(model_seed)
    best_loss = np.inf
    best_state = deepcopy(model.state_dict())
    best_epoch = 0
    stale = 0
    for epoch in range(cfg.epochs):
        model.train()
        shuffled = rng.permutation(train)
        for batch in np.array_split(shuffled, max(1,int(np.ceil(len(shuffled)/cfg.batch_size)))):
            pred = model(x[batch], a[batch])
            loss = torch.nn.functional.mse_loss(pred,target[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(torch.nn.functional.mse_loss(model(x[val],a[val]),target[val]).item())
        if val_loss < best_loss - 1e-8:
            best_loss = val_loss
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch+1
            stale = 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = np.concatenate([model(x_batch,a_batch).numpy() for x_batch,a_batch in
                               zip(x.split(128),a.split(128))],axis=0)/100.
    return pred, {"parameter_count":sum(p.numel() for p in model.parameters()),
                  "best_epoch":best_epoch,"best_val_mse_percent2":best_loss}


def _accuracy(y, pred):
    return {"mse":float(mean_squared_error(y.ravel(),pred.ravel())),
            "mae":float(mean_absolute_error(y.ravel(),pred.ravel())),
            "rank_ic":cross_sectional_rank_ic(y,pred),
            "directional_accuracy":float(np.mean((y>0)==(pred>0)))}


def run_research(prices, cfg=ResearchConfig()):
    """Fit all models using training only; validation for neural early stopping.

    No hyperparameters or models are chosen using test outcomes.
    """
    if (cfg.epochs<1 or cfg.patience<1 or cfg.hidden<4 or cfg.batch_size<1 or
        cfg.learning_rate<=0 or cfg.ridge_alpha<0 or cfg.cost_bps<0 or cfg.torch_threads<1):
        raise ValueError("Invalid research configuration")
    data = build_dataset(prices,cfg.graph_window,cfg.neighbors,cfg.train_fraction,cfg.val_fraction)
    n_assets = data.features.shape[1]
    if not 1 <= cfg.long_short <= n_assets//2:
        raise ValueError("long_short must satisfy 1 <= k <= n_assets//2")
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(cfg.torch_threads)
    x = data.features
    train = data.splits["train"]
    ridge = Ridge(alpha=cfg.ridge_alpha).fit(x[train].reshape(-1,x.shape[-1]),
                                             data.targets[train].reshape(-1))
    baseline = ridge.predict(x.reshape(-1,x.shape[-1])).reshape(len(x),n_assets)
    predictions = {
        "Zero forecast":np.zeros_like(data.targets),
        "Momentum-20":data.raw_features[:,:,2]/20.,
        "Ridge":baseline,
    }
    details = {"Zero forecast":{"parameter_count":0},
               "Momentum-20":{"parameter_count":0},
               "Ridge":{"parameter_count":int(x.shape[-1]+1)}}
    for j,name in enumerate(NEURAL_MODELS):
        predictions[name],details[name] = _train_network(name,data,cfg,cfg.seed+11*j)
    metrics = {}
    prediction_rows = []
    equity_rows = []
    for name,pred in predictions.items():
        metrics[name] = {"details":details[name]}
        for split in ("validation","test"):
            ids = data.splits[split]
            y = data.targets[ids]
            p = pred[ids]
            report = _accuracy(y,p)
            if name == "Zero forecast":
                report["portfolio"] = None
            else:
                strategy, equity, weights = simulate_strategy(p,y,cfg.cost_bps,cfg.long_short)
                report["portfolio"] = strategy
                for date,row in zip(data.target_dates[ids],equity.itertuples(index=False)):
                    equity_rows.append({"date":date,"split":split,"model":name,
                                       "gross_return":row.gross_return,"net_return":row.net_return,
                                       "turnover":row.turnover,"wealth":row.wealth})
            metrics[name][split] = report
            for i in ids:
                for asset,jasset in zip(data.asset_names,range(n_assets)):
                    prediction_rows.append({"signal_date":data.decision_dates[i],
                                            "target_date":data.target_dates[i],"split":split,
                                            "model":name,"asset":asset,
                                            "actual_next_return":float(data.targets[i,jasset]),
                                            "predicted_next_return":float(pred[i,jasset])})
    report={"splits":{k:len(v) for k,v in data.splits.items()},
            "assets":data.asset_names,"graph_window":cfg.graph_window,
            "neighbors":cfg.neighbors,"train_last_target_date":str(data.target_dates[train[-1]]),
            "validation_last_target_date":str(data.target_dates[data.splits['validation'][-1]]),
            "metrics":metrics}
    return report,pd.DataFrame(prediction_rows),pd.DataFrame(equity_rows),data
