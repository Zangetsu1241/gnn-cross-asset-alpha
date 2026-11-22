"""Leakage-aware cross-sectional alpha research with dynamic financial graphs."""
from .data import synthetic_prices, load_prices, build_dataset, make_graph
from .research import ResearchConfig, run_research

__all__ = ["synthetic_prices", "load_prices", "build_dataset", "make_graph", "ResearchConfig", "run_research"]
