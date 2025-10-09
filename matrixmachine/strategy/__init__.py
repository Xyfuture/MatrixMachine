"""Tiling strategies for matrix decomposition."""

from .trivial import TrivialTilingStrategy
from .agent_grid_search import AgentGridSearchStrategy
from .code_search import CodeSearchStrategy

try:
    from .grid import GridTilingStrategy
    __all__ = [
        "TrivialTilingStrategy",
        "GridTilingStrategy",
        "AgentGridSearchStrategy",
        "CodeSearchStrategy",
    ]
except ImportError:
    __all__ = [
        "TrivialTilingStrategy",
        "AgentGridSearchStrategy",
        "CodeSearchStrategy",
    ]
