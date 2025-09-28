"""Tiling strategies for matrix decomposition."""

from .trivial import TrivialTilingStrategy
from .agent_grid_search import AgentGridSearchStrategy

try:
    from .grid import GridTilingStrategy
    __all__ = [
        "TrivialTilingStrategy",
        "GridTilingStrategy",
        "AgentGridSearchStrategy",
    ]
except ImportError:
    __all__ = [
        "TrivialTilingStrategy",
        "AgentGridSearchStrategy",
    ]
