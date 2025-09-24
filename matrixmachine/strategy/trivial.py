from dataclasses import dataclass
from math import ceil, sqrt
from typing import List, Optional, Tuple

from ..core.description import Chip, Mapping, MatrixShape


@dataclass
class TrivialTilingStrategy:
    """Grid-based tiling that assigns submatrices to dies in round-robin order."""

    def create_mapping(
        self,
        matrix_shape: MatrixShape,
        chip: Chip,
        grid_rows: int,
        grid_cols: int,
    ) -> Mapping:
        if grid_rows <= 0 or grid_cols <= 0:
            raise ValueError("grid_rows and grid_cols must be positive integers")
        if matrix_shape.rows <= 0 or matrix_shape.cols <= 0:
            raise ValueError("matrix dimensions must be positive")
        if grid_rows > matrix_shape.rows or grid_cols > matrix_shape.cols:
            raise ValueError("grid dimensions cannot exceed matrix dimensions")
        if not chip.compute_dies:
            raise ValueError("Chip must have at least one compute die")

        row_bounds = self._split_dimension(matrix_shape.rows, grid_rows)
        col_bounds = self._split_dimension(matrix_shape.cols, grid_cols)
        if not row_bounds or not col_bounds:
            raise ValueError("Grid configuration produced no tiles")

        die_ids = sorted(chip.compute_dies.keys())
        mapping = Mapping(matrix=matrix_shape, chip=chip)

        tile_index = 0
        for row0, row1 in row_bounds:
            for col0, col1 in col_bounds:
                die_id = die_ids[tile_index % len(die_ids)]
                mapping.add_tile(die_id, row0, row1, col0, col1, batch0=0, batch1=matrix_shape.batch_size)
                tile_index += 1

        mapping.check_all()
        return mapping

    def create_balanced_mapping(
        self,
        matrix_shape: MatrixShape,
        chip: Chip,
        max_tile_area: Optional[int] = None,
    ) -> Mapping:
        if not chip.compute_dies:
            raise ValueError("Chip must have at least one compute die")

        num_dies = len(chip.compute_dies)
        matrix_volume = matrix_shape.volume()

        if max_tile_area is None:
            target_tiles = max(1, num_dies)
        else:
            target_tiles = max(num_dies, ceil(matrix_volume / max_tile_area))

        grid_rows, grid_cols = self._derive_grid(matrix_shape, target_tiles)
        return self.create_mapping(matrix_shape, chip, grid_rows, grid_cols)

    def _split_dimension(self, total: int, parts: int) -> List[Tuple[int, int]]:
        if parts > total:
            raise ValueError("grid dimension parts cannot exceed matrix dimension")
        base = total // parts
        extra = total % parts
        bounds: List[Tuple[int, int]] = []
        start = 0
        for idx in range(parts):
            length = base + (1 if idx < extra else 0)
            end = start + length
            bounds.append((start, end))
            start = end
        return bounds

    def _derive_grid(self, matrix_shape: MatrixShape, tiles: int) -> Tuple[int, int]:
        # Aim for a near-square grid without exceeding matrix dimensions.
        side = max(1, ceil(sqrt(tiles)))
        grid_rows = min(matrix_shape.rows, side)
        grid_cols = min(matrix_shape.cols, ceil(tiles / grid_rows))
        if grid_cols == 0:
            grid_cols = 1
        return grid_rows, grid_cols
