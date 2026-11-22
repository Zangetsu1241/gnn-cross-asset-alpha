"""Reproducible GNN cross-asset alpha experiment CLI and research artifacts."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from .data import synthetic_prices, load_prices
from .research import ResearchConfig, run_research


def main(argv=None):
    parser = argparse.ArgumentParser(description="Causal correlation-graph alpha forecasting study")
    parser.add_argument("--csv", help="Date plus 4+ point-in-time adjusted close columns")
    parser.add_argument("--days", type=int, default=700)
    parser.add_argument("--assets", type=int, default=8)
    parser.add_argument("--graph-window", type=int, default=40)
    parser.add_argument("--neighbors", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--long-short", type=int, default=2)
    parser.add_argument("--cost-bps", type=float, default=5.)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results")
    args = parser.parse_args(argv)
    prices = load_prices(args.csv) if args.csv else synthetic_prices(args.days,args.assets,args.seed)
    cfg = ResearchConfig(graph_window=args.graph_window,neighbors=args.neighbors,
                         epochs=args.epochs,patience=args.patience,hidden=args.hidden,
                         batch_size=args.batch_size,long_short=args.long_short,
                         cost_bps=args.cost_bps,seed=args.seed)
    report, predictions, equity, dataset = run_research(prices,cfg)
    out = Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    predictions.to_csv(out/"predictions.csv",index=False)
    equity.to_csv(out/"portfolio_returns.csv",index=False)
    payload = {
        "source":"user_adjusted_closes" if args.csv else "synthetic_regime_switching_factor_market",
        "configuration":vars(args),"research":report,
        "limitations":"Synthetic data is not trading evidence. Decision-close signals target the next close-to-close return; same-close execution after observing close is unrealistic. Borrow, financing, fill, spread and impact are not fully modeled.",
    }
    (out/"metrics.json").write_text(json.dumps(payload,indent=2,allow_nan=False))
    fig,ax = plt.subplots(figsize=(10,4.5))
    labels = list(report["metrics"])
    mse = [report["metrics"][model]["test"]["mse"] for model in labels]
    ax.bar(labels,mse)
    ax.set(title="Held-out next-day return prediction error",ylabel="MSE of simple returns")
    ax.tick_params(axis="x",rotation=25)
    fig.tight_layout();fig.savefig(out/"test_mse.png",dpi=160);plt.close(fig)

    fig,ax = plt.subplots(figsize=(10,5.5))
    for model, group in equity[equity.split=="test"].groupby("model"):
        ax.plot(group.date,group.wealth,label=model)
    ax.set(title="Long-short strategy: held-out test (illustrative)",ylabel="Wealth from 1")
    ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/"test_equity.png",dpi=160);plt.close(fig)

    sample_idx = dataset.splits["train"][-1]
    graph = dataset.graphs[sample_idx]
    fig,ax=plt.subplots(figsize=(6.5,5.5))
    im=ax.imshow(graph,vmin=-1,vmax=1,cmap="coolwarm")
    ax.set(title="Causal correlation graph (last training sample)")
    ax.set_xticks(np.arange(len(dataset.asset_names)),labels=dataset.asset_names,rotation=45,ha="right")
    ax.set_yticks(np.arange(len(dataset.asset_names)),labels=dataset.asset_names)
    fig.colorbar(im,ax=ax,label="Signed, row-normalized edge weight")
    fig.tight_layout();fig.savefig(out/"graph_snapshot.png",dpi=160);plt.close(fig)

    compact={m:{"mse":r["test"]["mse"],"rank_ic":r["test"]["rank_ic"],
                "portfolio_total_return":None if r["test"]["portfolio"] is None else r["test"]["portfolio"]["total_return"]}
             for m,r in report["metrics"].items()}
    print(json.dumps({"splits":report["splits"],"test":compact},indent=2))
    return payload


if __name__=="__main__":
    main()
