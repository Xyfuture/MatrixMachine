from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from ..core.description import Chip, Mapping, MatrixShape
from ..core.sim_engine import simulate
from ..core.utils import MappingResult

logger = logging.getLogger("agent_grid_search")


@dataclass(frozen=True)
class SubMatrixSpec:
    """A contiguous sub-matrix region within the original matrix.

    Defined by element-space offsets and size. Used to recursively solve the
    remaining (tail) region after full round-robin assignment.
    """

    row0: int
    row1: int
    col0: int
    col1: int
    batch0: int = 0
    batch1: int = 1

    def to_matrix_shape(self) -> MatrixShape:
        return MatrixShape(
            rows=self.row1 - self.row0,
            cols=self.col1 - self.col0,
            batch_size=self.batch1 - self.batch0
        )


@dataclass
class AgentGridSearchStrategy:
    """Advanced recursive DSE strategy to find optimal Mapping via grid splits.

    This strategy combines the best features from both GPT and Codex implementations:

    Key improvements and fixes:
    - Geometric precision: Correctly handles tail regions as contiguous rectangles
    - Progress guarantee: Prevents infinite recursion when dies >= tiles
    - Complete round-robin: Only performs full rounds, handles remainder geometrically
    - Robust memoization: Caches results by (matrix_shape, available_dies)
    - Comprehensive validation: Full error handling and mapping verification
    - Exact coordinate mapping: Proper offset handling for sub-matrices
    - Iteration limit: Limits recursion depth with round-robin fallback

    Algorithm flow:
    1. Try all split configurations (num_split_row × num_split_col)
    2. Create tiles in row-major order from grid splits
    3. Distribute tiles using round-robin (complete rounds only)
    4. Handle remaining tiles by identifying exact geometric tail regions
    5. Recursively solve tail regions and combine with main mapping (up to max_iterations)
    6. If max iterations reached, fallback to round-robin distribution
    7. Return the configuration with minimum latency
    """

    num_split_row_candidates: List[int] = field(default_factory=lambda: list(range(1, 9)))
    num_split_col_candidates: List[int] = field(default_factory=lambda: list(range(1, 9)))
    memo: Dict[str, MappingResult] = field(default_factory=dict)
    max_iterations: int = field(default=2)

    # ---------------
    # Public API
    # ---------------
    def find_optimal_mapping(
        self,
        matrix_shape: MatrixShape,
        chip: Chip,
        available_dies: Optional[Set[str]] = None,
        current_iteration: int = 0,
    ) -> Optional[MappingResult]:
        """Find the optimal mapping for the given matrix and chip configuration.

        Args:
            matrix_shape: The matrix dimensions to optimize for
            chip: Hardware chip configuration with compute dies
            available_dies: Set of die IDs to use (defaults to all dies)
            current_iteration: Current recursion depth (internal parameter)

        Returns:
            MappingResult with optimal mapping and latency, or None if no valid mapping found
        """
        if available_dies is None:
            available_dies = set(chip.compute_dies.keys())
        if not available_dies:
            return None

        # Debug output for recursion
        indent = "  " * current_iteration
        logger.info(f"{indent}[递归 {current_iteration}] 开始处理矩阵: {matrix_shape.rows}x{matrix_shape.cols}x{matrix_shape.batch_size}")
        logger.info(f"{indent}  可用计算单元: {sorted(list(available_dies))}")
        logger.info(f"{indent}  计算单元数量: {len(available_dies)}")

        # Check memoization cache
        key = self._memo_key(matrix_shape, available_dies)
        if key in self.memo:
            logger.info(f"{indent}  从缓存中获取结果")
            return self.memo[key]

        best: Optional[MappingResult] = None
        best_latency = float("inf")

        # Try all split configurations
        split_count = 0
        for num_r in self.num_split_row_candidates:
            for num_c in self.num_split_col_candidates:
                # Skip invalid configurations
                if num_r <= 0 or num_c <= 0:
                    continue
                if num_r > matrix_shape.rows or num_c > matrix_shape.cols:
                    continue

                split_count += 1
                logger.info(f"{indent}  尝试分割配置 {split_count}: {num_r}x{num_c} (行x列)")

                try:
                    candidate = self._evaluate_split_configuration(
                        matrix_shape, chip, available_dies, num_r, num_c, current_iteration
                    )
                except Exception:
                    logger.warning(f"{indent}    配置失败")
                    candidate = None

                if candidate and candidate.latency < best_latency:
                    logger.info(f"{indent}    找到更好的配置，延迟: {candidate.latency}")
                    best = candidate
                    best_latency = candidate.latency
                elif candidate:
                    logger.info(f"{indent}    配置有效，延迟: {candidate.latency}")
                else:
                    logger.debug(f"{indent}    配置无效")

        # Cache the best result
        if best:
            self.memo[key] = best
            logger.info(f"{indent}[递归 {current_iteration}] 完成，最佳延迟: {best.latency}")
        else:
            logger.warning(f"{indent}[递归 {current_iteration}] 完成，未找到有效配置")

        return best

    # ---------------
    # Core evaluation logic
    # ---------------
    def _evaluate_split_configuration(
        self,
        matrix: MatrixShape,
        chip: Chip,
        available_dies: Set[str],
        num_split_row: int,
        num_split_col: int,
        current_iteration: int,
    ) -> Optional[MappingResult]:
        """Evaluate a specific split configuration and return the mapping result."""

        indent = "  " * current_iteration

        # Compute split boundaries with ceiling on edge tiles
        row_bounds = self._calculate_split_boundaries(matrix.rows, num_split_row)
        col_bounds = self._calculate_split_boundaries(matrix.cols, num_split_col)
        batch_bounds = [(0, matrix.batch_size)]

        # Create tiles in row-major order (critical for correct tail geometry)
        tiles: List[Tuple[int, int, int, int, int, int]] = []
        for r0, r1 in row_bounds:
            for c0, c1 in col_bounds:
                for b0, b1 in batch_bounds:
                    tiles.append((r0, r1, c0, c1, b0, b1))

        die_ids = sorted(list(available_dies))
        n_tiles = len(tiles)
        n_dies = len(die_ids)

        logger.info(f"{indent}    生成了 {n_tiles} 个 tiles，{n_dies} 个计算单元")

        if n_dies == 0:
            return None

        # Progress guarantee: if we have enough dies, assign all tiles
        if n_tiles <= n_dies:
            logger.info(f"{indent}    Tiles数量 <= 计算单元数量，直接分配")
            assignments: Dict[str, List[Tuple[int, int, int, int, int, int]]] = {d: [] for d in die_ids}
            for i, tile in enumerate(tiles):
                assignments[die_ids[i % n_dies]].append(tile)

            main_mapping = self._create_mapping_from_assignments(matrix, chip, assignments)
            try:
                main_mapping.check_all()
                latency = simulate(chip, main_mapping, save_trace=False)
                return MappingResult(mapping=main_mapping, latency=latency)
            except Exception:
                return None

        # Round-robin distribution: complete rounds only
        tiles_per_die = n_tiles // n_dies
        assigned_count = tiles_per_die * n_dies

        logger.info(f"{indent}    轮询分配: 每个计算单元 {tiles_per_die} 个 tiles，总共分配 {assigned_count} 个")

        main_assignments: Dict[str, List[Tuple[int, int, int, int, int, int]]] = {d: [] for d in die_ids}
        for i in range(assigned_count):
            main_assignments[die_ids[i % n_dies]].append(tiles[i])

        remaining_tiles = tiles[assigned_count:]
        main_mapping = self._create_mapping_from_assignments(matrix, chip, main_assignments)

        # If no remaining tiles, we're done
        if not remaining_tiles:
            logger.info(f"{indent}    没有剩余 tiles，完成分配")
            try:
                main_mapping.check_all()
                latency = simulate(chip, main_mapping, save_trace=False)
                return MappingResult(mapping=main_mapping, latency=latency)
            except Exception:
                return None

        logger.info(f"{indent}    剩余 {len(remaining_tiles)} 个 tiles 需要递归处理")

        # Check iteration limit before recursion
        if current_iteration >= self.max_iterations:
            logger.warning(f"{indent}    达到最大递归深度限制 ({self.max_iterations})，使用轮询分配回退策略")
            # Fallback: Use round-robin distribution for remaining tiles
            return self._round_robin_fallback(matrix, chip, main_assignments, remaining_tiles)

        # Handle tail tiles by constructing exact geometric sub-regions
        sub_specs = self._construct_tail_subregions(row_bounds, col_bounds, remaining_tiles)

        logger.info(f"{indent}    构建了 {len(sub_specs)} 个尾部子区域")

        # Recursively solve each subregion
        sub_mappings: List[Mapping] = []
        for i, spec in enumerate(sub_specs):
            logger.info(f"{indent}    处理子区域 {i+1}: [{spec.row0}:{spec.row1}, {spec.col0}:{spec.col1}, {spec.batch0}:{spec.batch1}]")
            sub_shape = spec.to_matrix_shape()
            sub_result = self.find_optimal_mapping(sub_shape, chip, available_dies, current_iteration + 1)
            if not sub_result:
                logger.warning(f"{indent}    子区域 {i+1} 无解")
                return None
            # Map sub-region coordinates back to original matrix space
            offset_mapping = self._apply_coordinate_offset(sub_result.mapping, spec)
            sub_mappings.append(offset_mapping)

        # Combine main mapping with sub-mappings
        combined_mapping = self._combine_mappings(main_mapping, sub_mappings)
        try:
            combined_mapping.check_all()
            latency = simulate(chip, combined_mapping, save_trace=False)
            return MappingResult(mapping=combined_mapping, latency=latency)
        except Exception:
            return None

    # ---------------
    # Geometry and tiling helpers
    # ---------------
    def _calculate_split_boundaries(self, dimension: int, num_splits: int) -> List[Tuple[int, int]]:
        """Calculate split boundaries with ceiling operation for edge tiles."""
        if num_splits <= 0:
            return [(0, dimension)]

        base_size = dimension // num_splits
        remainder = dimension % num_splits

        boundaries = []
        start = 0

        for i in range(num_splits):
            # Distribute remainder to first `remainder` splits (ceiling operation)
            size = base_size + (1 if i < remainder else 0)
            end = start + size
            boundaries.append((start, end))
            start = end

        return boundaries

    def _create_mapping_from_assignments(
        self,
        matrix: MatrixShape,
        chip: Chip,
        assignments: Dict[str, List[Tuple[int, int, int, int, int, int]]],
    ) -> Mapping:
        """Create a mapping object from tile assignments."""
        mapping = Mapping(matrix=matrix, chip=chip)
        for die_id, tile_list in assignments.items():
            for r0, r1, c0, c1, b0, b1 in tile_list:
                mapping.add_tile(die_id, r0, r1, c0, c1, b0, b1)
        return mapping

    def _construct_tail_subregions(
        self,
        row_bounds: List[Tuple[int, int]],
        col_bounds: List[Tuple[int, int]],
        remaining_tiles: List[Tuple[int, int, int, int, int, int]],
    ) -> List[SubMatrixSpec]:
        """Construct at most two rectangular subregions that exactly cover the tail.

        The tail in row-major order always consists of:
        - Zero or more complete bottom rows, and
        - An optional right-edge suffix of the row just above them.

        This geometric precision is critical for correctness.
        """
        if not remaining_tiles:
            return []

        # Grid dimensions
        g_rows = len(row_bounds)
        g_cols = len(col_bounds)

        # Analyze tail structure
        num_tail = len(remaining_tiles)
        full_rows = num_tail // g_cols
        suffix_cols = num_tail % g_cols

        specs: List[SubMatrixSpec] = []

        def grid_rect_to_spec(r0_idx: int, r1_idx_excl: int, c0_idx: int, c1_idx_excl: int) -> Optional[SubMatrixSpec]:
            """Convert grid indices to matrix coordinates."""
            if r0_idx >= r1_idx_excl or c0_idx >= c1_idx_excl:
                return None
            row0 = row_bounds[r0_idx][0]
            row1 = row_bounds[r1_idx_excl - 1][1]
            col0 = col_bounds[c0_idx][0]
            col1 = col_bounds[c1_idx_excl - 1][1]
            return SubMatrixSpec(row0=row0, row1=row1, col0=col0, col1=col1)

        # Case 1: Only partial row suffix
        if full_rows == 0 and suffix_cols > 0:
            r_idx = g_rows - 1
            c0_idx = g_cols - suffix_cols
            c1_idx = g_cols
            spec = grid_rect_to_spec(r_idx, r_idx + 1, c0_idx, c1_idx)
            if spec:
                specs.append(spec)
            return specs

        # Case 2: Only complete bottom rows
        if suffix_cols == 0 and full_rows > 0:
            r0_idx = g_rows - full_rows
            r1_idx = g_rows
            spec = grid_rect_to_spec(r0_idx, r1_idx, 0, g_cols)
            if spec:
                specs.append(spec)
            return specs

        # Case 3: Partial row + complete rows below
        if suffix_cols > 0 and full_rows > 0:
            # Top partial row
            top_r_idx = g_rows - full_rows - 1
            c0_idx = g_cols - suffix_cols
            c1_idx = g_cols
            top_spec = grid_rect_to_spec(top_r_idx, top_r_idx + 1, c0_idx, c1_idx)
            if top_spec:
                specs.append(top_spec)

            # Bottom complete rows
            bot_r0_idx = g_rows - full_rows
            bot_r1_idx = g_rows
            bottom_spec = grid_rect_to_spec(bot_r0_idx, bot_r1_idx, 0, g_cols)
            if bottom_spec:
                specs.append(bottom_spec)

        return specs

    def _apply_coordinate_offset(self, mapping: Mapping, spec: SubMatrixSpec) -> Mapping:
        """Apply coordinate offset to map sub-region tiles to original matrix space."""
        offset_mapping = Mapping(matrix=mapping.matrix, chip=mapping.chip)
        for die_id, tiles in mapping.placement.items():
            for tile in tiles:
                offset_mapping.add_tile(
                    die_id,
                    spec.row0 + tile.row0,
                    spec.row0 + tile.row1,
                    spec.col0 + tile.col0,
                    spec.col0 + tile.col1,
                    spec.batch0 + tile.batch0,
                    spec.batch0 + tile.batch1,
                )
        return offset_mapping

    def _combine_mappings(self, main_mapping: Mapping, sub_mappings: List[Mapping]) -> Mapping:
        """Combine main mapping with sub-region mappings."""
        combined = Mapping(matrix=main_mapping.matrix, chip=main_mapping.chip)

        # Add tiles from main mapping
        for die_id, tiles in main_mapping.placement.items():
            for tile in tiles:
                combined.add_tile(
                    die_id, tile.row0, tile.row1, tile.col0, tile.col1,
                    tile.batch0, tile.batch1
                )

        # Add tiles from sub-mappings
        for sub_mapping in sub_mappings:
            for die_id, tiles in sub_mapping.placement.items():
                for tile in tiles:
                    combined.add_tile(
                        die_id, tile.row0, tile.row1, tile.col0, tile.col1,
                        tile.batch0, tile.batch1
                    )

        return combined

    # ---------------
    # Memoization
    # ---------------
    def _memo_key(self, matrix: MatrixShape, available_dies: Set[str]) -> str:
        """Create a memoization key for the subproblem."""
        dies_key = ",".join(sorted(available_dies))
        return f"{matrix.rows}x{matrix.cols}x{matrix.batch_size}|{dies_key}"

    def _round_robin_fallback(
        self,
        matrix: MatrixShape,
        chip: Chip,
        main_assignments: Dict[str, List[Tuple[int, int, int, int, int, int]]],
        remaining_tiles: List[Tuple[int, int, int, int, int, int]]
    ) -> Optional[MappingResult]:
        """Fallback strategy: distribute remaining tiles using round-robin."""
        die_ids = sorted(list(main_assignments.keys()))

        # Add remaining tiles using round-robin
        for i, tile in enumerate(remaining_tiles):
            die_id = die_ids[i % len(die_ids)]
            main_assignments[die_id].append(tile)

        # Create and validate the mapping
        combined_mapping = self._create_mapping_from_assignments(matrix, chip, main_assignments)
        try:
            combined_mapping.check_all()
            latency = simulate(chip, combined_mapping, save_trace=False)
            return MappingResult(mapping=combined_mapping, latency=latency)
        except Exception:
            return None