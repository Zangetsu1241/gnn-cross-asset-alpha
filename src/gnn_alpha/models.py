"""Dependency-light message passing GNNs: signed graph convolution and attention.

Use only PyTorch so the benchmark runs without compiled PyG extensions.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F


class SharedMLP(nn.Module):
    """Same node-wise forecasting network for each asset, without graph edges."""
    def __init__(self, features=5, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(features, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x, adjacency=None):
        return self.net(x).squeeze(-1)


class SignedGraphConv(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.self_proj = nn.Linear(in_features, out_features)
        self.neighbor_proj = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x, adjacency):
        return self.self_proj(x) + self.neighbor_proj(torch.bmm(adjacency, x))


class GraphConvNet(nn.Module):
    def __init__(self, features=5, hidden=32):
        super().__init__()
        self.layer1 = SignedGraphConv(features, hidden)
        self.layer2 = SignedGraphConv(hidden, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x, adjacency):
        h = F.relu(self.layer1(x, adjacency))
        h = F.relu(self.layer2(h, adjacency))
        return self.out(h).squeeze(-1)


class SignedGraphAttention(nn.Module):
    """Attention restricted to historical correlation-graph neighbors.

    Attention weights are positive, then each message is multiplied by the sign
    of its historical correlation. Absolute graph edge weight enters as a log
    attention prior. Zero edges are masked out.
    """
    def __init__(self, in_features, hidden):
        super().__init__()
        self.q = nn.Linear(in_features, hidden, bias=False)
        self.k = nn.Linear(in_features, hidden, bias=False)
        self.v = nn.Linear(in_features, hidden, bias=False)
        self.self_proj = nn.Linear(in_features, hidden)
        self.hidden = hidden

    def forward(self, x, adjacency):
        query, key, value = self.q(x), self.k(x), self.v(x)
        similarity = torch.bmm(query, key.transpose(1,2)) / math.sqrt(self.hidden)
        active = adjacency.abs() > 0
        similarity = similarity + torch.log(adjacency.abs().clamp_min(1e-10))
        similarity = similarity.masked_fill(~active, -1e9)
        probs = similarity.softmax(dim=-1)
        signed = probs * adjacency.sign()
        return self.self_proj(x) + torch.bmm(signed, value)


class GraphAttentionNet(nn.Module):
    def __init__(self, features=5, hidden=32):
        super().__init__()
        self.layer1 = SignedGraphAttention(features, hidden)
        self.layer2 = SignedGraphAttention(hidden, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x, adjacency):
        h = F.relu(self.layer1(x, adjacency))
        h = F.relu(self.layer2(h, adjacency))
        return self.out(h).squeeze(-1)


NEURAL_MODELS = {"Node MLP": SharedMLP, "GCN": GraphConvNet, "GAT": GraphAttentionNet}
