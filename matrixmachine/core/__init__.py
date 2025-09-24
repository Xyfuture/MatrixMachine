"""Core modules for MatrixMachine."""

from .description import MatrixShape, Tile, ComputeDie, Chip, Mapping, ComputeDieSpec, ChipSpec, TileAssignmentInput
from .sim_engine import SimComputeDie, SimChip, simulate
from .utils import MappingResult, calculate_compute_utilization

__all__ = [
    # description module
    "MatrixShape",
    "Tile",
    "ComputeDie",
    "ComputeDieSpec",
    "Chip",
    "ChipSpec",
    "Mapping",
    "TileAssignmentInput",
    # sim_engine module
    "SimComputeDie",
    "SimChip",
    "simulate",
    # utils module
    "MappingResult",
    "calculate_compute_utilization",
]