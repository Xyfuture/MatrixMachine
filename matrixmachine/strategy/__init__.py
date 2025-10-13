"""Tiling strategies for matrix decomposition."""

from .trivial import TrivialTilingStrategy
from .agent_grid_search import AgentGridSearchStrategy
from .h2llm_mapping import H2LLMTilingStrategy, H2LLMDataCentricStrategy

try:
    from .grid import GridTilingStrategy
    __all__ = [
        "TrivialTilingStrategy",
        "GridTilingStrategy",
        "AgentGridSearchStrategy",
        "H2LLMTilingStrategy",
        "H2LLMDataCentricStrategy",
    ]
except ImportError:
    __all__ = [
        "TrivialTilingStrategy",
        "AgentGridSearchStrategy",
        "H2LLMTilingStrategy",
        "H2LLMDataCentricStrategy",
    ]
