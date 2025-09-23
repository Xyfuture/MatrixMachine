"""Tiling strategies for matrix decomposition."""

from .trivial import TrivialTilingStrategy
from .grid import GridTilingStrategy

__all__ = [
    "TrivialTilingStrategy",
    "GridTilingStrategy",
]
