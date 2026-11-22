import json
import numpy as np
import pandas as pd
import pytest
import torch
from gnn_alpha.data import synthetic_prices,load_prices,make_graph,build_dataset
from gnn_alpha.models import SharedMLP,GraphConvNet,GraphAttentionNet
from gnn_alpha.research import (ResearchConfig,run_research,cross_sectional_rank_ic,
                                long_short_weights,simulate_strategy)
from gnn_alpha.cli import main


def quick_data(n=260):
    return synthetic_prices(n,assets=6,seed=7)


def test_synthetic_panel_reproducible():
    a=synthetic_prices(200,6,3)
    b=synthetic_prices(200,6,3)
    pd.testing.assert_frame_equal(a,b)
    assert (a>0).all().all()


def test_invalid_csv_dates_and_prices(tmp_path):
    f=tmp_path/'data.csv'
    f.write_text('date,a,b,c,d\n2020-01-01,1,1,1,1\n2020-01-01,2,2,2,2\n')
    with pytest.raises(ValueError,match='dates|Dates'):
        load_prices(f)
    with pytest.raises(ValueError):
        build_dataset(quick_data().assign(asset_01=0))
    with pytest.raises(ValueError):synthetic_prices(20)


def test_graph_normalization_and_self_loops():
    x=np.random.default_rng(5).normal(size=(45,6))
    a=make_graph(x,neighbors=2)
    assert a.shape==(6,6)
    assert np.all(np.diag(a)>0)
    assert np.allclose(np.abs(a).sum(axis=1),1,atol=1e-6)
    assert np.isfinite(a).all()


def test_signed_negative_correlation_edge():
    rng=np.random.default_rng(2)
    x=rng.normal(size=(60,6))
    x[:,1]=-x[:,0]
    a=make_graph(x,neighbors=2)
    assert a[0,1]<0 and a[1,0]<0


def test_graph_prefix_causality():
    x=np.random.default_rng(1).normal(size=(80,6))
    y=x.copy()
    y[50:]+=300.
    assert np.array_equal(make_graph(x[:50],2),make_graph(y[:50],2))


def test_dataset_targets_and_chronology():
    prices=quick_data()
    d=build_dataset(prices,graph_window=30,neighbors=2)
    assert max(d.splits['train'])<min(d.splits['validation'])
    assert max(d.splits['validation'])<min(d.splits['test'])
    assert all(d.decision_dates < d.target_dates)
    for i in [0,len(d.targets)//2,len(d.targets)-1]:
        t=prices.index.get_loc(d.target_dates[i])
        expected=(prices.iloc[t]/prices.iloc[t-1]-1).to_numpy()
        assert np.allclose(d.targets[i],expected,atol=1e-7)


def test_train_scaler_ignores_future_prices():
    p=quick_data()
    q=p.copy()
    q.iloc[210:]*=np.array([2,.6,1.3,1.8,.9,1.1])
    a=build_dataset(p,30,2)
    b=build_dataset(q,30,2)
    assert np.array_equal(a.scaler_mean,b.scaler_mean)
    assert np.allclose(a.scaler_std,b.scaler_std,rtol=1e-12,atol=1e-15)
    assert np.array_equal(a.features[a.splits['train']],b.features[b.splits['train']])
    assert np.array_equal(a.graphs[a.splits['train']],b.graphs[b.splits['train']])


def test_test_suffix_does_not_change_earlier_input_graphs():
    p=quick_data()
    q=p.copy();q.iloc[225:]+=10000.
    a=build_dataset(p,30,2)
    b=build_dataset(q,30,2)
    prefix=a.target_positions<225
    assert np.array_equal(a.features[prefix],b.features[prefix])
    assert np.array_equal(a.graphs[prefix],b.graphs[prefix])


@pytest.mark.parametrize('model',[SharedMLP,GraphConvNet,GraphAttentionNet])
def test_model_shapes_and_gradients(model):
    torch.manual_seed(0)
    m=model(features=5,hidden=8)
    x=torch.randn(4,6,5)
    a=torch.eye(6).repeat(4,1,1)
    pred=m(x,a)
    assert pred.shape==(4,6) and torch.isfinite(pred).all()
    pred.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


def test_graph_models_react_to_neighbor_edges():
    torch.manual_seed(1)
    x=torch.randn(3,6,5)
    a=torch.eye(6).repeat(3,1,1)
    b=a.clone();b[:,0,1]=.5;b[:,0,0]=.5
    for cls in [GraphConvNet,GraphAttentionNet]:
        model=cls(hidden=8)
        assert not torch.allclose(model(x,a),model(x,b))


def test_rank_ic_for_perfect_and_inverse_prediction():
    y=np.array([[1.,2.,3.,4.],[3.,-1.,2.,9.]])
    assert np.isclose(cross_sectional_rank_ic(y,y),1)
    assert np.isclose(cross_sectional_rank_ic(y,-y),-1)
    assert cross_sectional_rank_ic(y,np.zeros_like(y)) is None


def test_long_short_neutrality_and_all_ties():
    p=np.array([[1.,2.,3.,4.,5.,6.],[0.,0.,0.,0.,0.,0.]])
    w=long_short_weights(p,k=2)
    assert np.allclose(w.sum(axis=1),0.)
    assert np.allclose(np.abs(w[0]).sum(),2)
    assert np.allclose(w[1],0)


def test_strategy_costs_and_wealth_alignment():
    p=np.array([[1,2,3,4],[1,2,3,4],[4,3,2,1]],float)
    y=np.tile([.01,.01,.01,.01],(3,1))
    stats,eq,w=simulate_strategy(p,y,cost_bps=10,k=1)
    assert np.allclose(eq.gross_return,0.)
    assert np.allclose(eq.turnover,[2,0,4])
    assert np.allclose(eq.net_return,-eq.turnover*.001)
    assert np.isclose(stats['total_return'],np.prod(1+eq.net_return)-1)


def test_strategy_rejects_invalid_inputs():
    with pytest.raises(ValueError):long_short_weights(np.zeros((3,4)),k=3)
    with pytest.raises(ValueError):simulate_strategy(np.ones((4,4)),np.zeros((4,4)),cost_bps=-1)
    with pytest.raises(ValueError):simulate_strategy(np.ones((4,4)),np.zeros((2,4)))


def test_complete_experiment_and_metrics():
    cfg=ResearchConfig(graph_window=30,neighbors=2,epochs=2,patience=2,hidden=8,
                       batch_size=80,long_short=2,seed=1)
    report,pred,equity,data=run_research(quick_data(),cfg)
    assert set(report['metrics'])=={'Zero forecast','Momentum-20','Ridge','Node MLP','GCN','GAT'}
    assert len(pred)==(len(data.splits['validation'])+len(data.splits['test']))*6*6
    assert equity.model.nunique()==5
    assert (pred.signal_date<pred.target_date).all()
    assert all(np.isfinite(result['test']['mse']) for result in report['metrics'].values())


def test_deterministic_repeated_experiment():
    cfg=ResearchConfig(graph_window=30,neighbors=2,epochs=1,patience=1,hidden=8,seed=21)
    p=quick_data(210)
    first,_,_,_=run_research(p,cfg)
    second,_,_,_=run_research(p,cfg)
    for name in first['metrics']:
        assert np.isclose(first['metrics'][name]['test']['mse'],second['metrics'][name]['test']['mse'])


def test_bad_config():
    with pytest.raises(ValueError):run_research(quick_data(),ResearchConfig(epochs=0))
    with pytest.raises(ValueError):build_dataset(quick_data(),graph_window=10)


def test_cli_outputs(tmp_path):
    report=main(['--days','220','--assets','6','--epochs','1','--hidden','8',
                 '--graph-window','30','--neighbors','2','--output',str(tmp_path)])
    assert len(report['research']['metrics'])==6
    for file in ['metrics.json','predictions.csv','portfolio_returns.csv','test_mse.png',
                 'test_equity.png','graph_snapshot.png']:
        assert (tmp_path/file).exists()
    output=json.loads((tmp_path/'metrics.json').read_text())
    assert output['source']=='synthetic_regime_switching_factor_market'
