from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from ..core.description import Chip, Mapping, MatrixShape
from ..core.sim_engine import simulate
from ..core.utils import MappingResult


@dataclass
class CodeSearchStrategy:
    """Heuristic DSE strategy focused on higher compute utilization.

    Key ideas:
    - Try a compact set of row/col grid splits (includes stripes like 1xD and Dx1)
    - Predict per-tile critical latency using the max of input/compute/memory/output
    - Assign tiles to dies using LPT (longest-processing-time first) load balancing
      to minimize makespan versus naive round-robin
    - Validate and simulate each candidate, picking the lowest-latency mapping

    This tends to reduce the critical-die time and thus improve overall
    compute utilization compared to simple round-robin grid assignment.
    """

    # Include 1 to allow stripe splits; keep others aligned with batch_test_matrices.py
    num_split_row_candidates: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 6, 8])
    num_split_col_candidates: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 6, 8])

    def find_optimal_mapping(
        self,
        matrix_shape: MatrixShape,
        chip: Chip,
        available_dies: Optional[Set[str]] = None,
    ) -> Optional[MappingResult]:
        if available_dies is None:
            available_dies = set(chip.compute_dies.keys())
        if not available_dies:
            return None

        best: Optional[MappingResult] = None
        best_latency = float("inf")

        for num_r in self.num_split_row_candidates:
            if num_r <= 0 or num_r > matrix_shape.rows:
                continue

            row_bounds = self._calculate_split_boundaries(matrix_shape.rows, num_r)

            for num_c in self.num_split_col_candidates:
                if num_c <= 0 or num_c > matrix_shape.cols:
                    continue

                col_bounds = self._calculate_split_boundaries(matrix_shape.cols, num_c)

                # Build tiles (row-major)
                tiles: List[Tuple[int, int, int, int, int, int]] = []
                for r0, r1 in row_bounds:
                    for c0, c1 in col_bounds:
                        tiles.append((r0, r1, c0, c1, 0, matrix_shape.batch_size))

                # Evaluate this split with LPT scheduling
                result = self._evaluate_with_lpt(matrix_shape, chip, available_dies, tiles)
                if result and result.latency < best_latency:
                    best = result
                    best_latency = result.latency

        return best

    # ----------
    # Core evaluation using LPT load balancing
    # ----------
    def _evaluate_with_lpt(
        self,
        matrix: MatrixShape,
        chip: Chip,
        available_dies: Set[str],
        tiles: List[Tuple[int, int, int, int, int, int]],
    ) -> Optional[MappingResult]:
        if not tiles:
            return None

        die_ids = sorted(list(available_dies))
        if not die_ids:
            return None

        # Pre-compute per-tile cost using critical stage latency
        # Use the first die spec (homogeneous dies per ChipSpec)
        sample_die = chip.compute_dies[die_ids[0]]
        comp_gops = sample_die.compute_power * 10**3
        mem_gops = sample_die.memory_bandwidth * 10**3
        in_bw = sample_die.input_bandwidth
        out_bw = sample_die.output_bandwidth

        def tile_cost(t: Tuple[int, int, int, int, int, int]) -> float:
            r0, r1, c0, c1, b0, b1 = t
            rows = r1 - r0
            cols = c1 - c0
            batches = b1 - b0
            # Stage latencies per tile (cycles)
            tin = (batches * rows) / in_bw if in_bw > 0 else float("inf")
            tout = (batches * cols) / out_bw if out_bw > 0 else float("inf")
            tcomp = (batches * rows * cols) / comp_gops if comp_gops > 0 else float("inf")
            tmem = (rows * cols) / mem_gops if mem_gops > 0 else float("inf")
            return max(tin, tout, tcomp, tmem)

        # Sort tiles by descending cost (LPT)
        tiles_sorted = sorted(tiles, key=tile_cost, reverse=True)

        # Min-heap of (accumulated_cost, die_id)
        heap: List[Tuple[float, str]] = [(0.0, d) for d in die_ids]
        heapq.heapify(heap)

        assignments: Dict[str, List[Tuple[int, int, int, int, int, int]]] = {d: [] for d in die_ids}
        acc_costs: Dict[str, float] = {d: 0.0 for d in die_ids}

        for t in tiles_sorted:
            cost = tile_cost(t)
            cur_cost, die_id = heapq.heappop(heap)
            assignments[die_id].append(t)
            new_cost = cur_cost + cost
            acc_costs[die_id] = new_cost
            heapq.heappush(heap, (new_cost, die_id))

        # Build mapping in the assigned order per die
        mapping = Mapping(matrix=matrix, chip=chip)
        for die_id, tile_list in assignments.items():
            for r0, r1, c0, c1, b0, b1 in tile_list:
                mapping.add_tile(die_id, r0, r1, c0, c1, b0, b1)

        try:
            mapping.check_all()
        except Exception:
            return None

        # Simulate and return result
        try:
            latency = simulate(chip, mapping, save_trace=False)
            return MappingResult(mapping=mapping, latency=latency)
        except Exception:
            return None

    # ----------
    # Helpers
    # ----------
    def _calculate_split_boundaries(self, dimension: int, num_splits: int) -> List[Tuple[int, int]]:
        if num_splits <= 0:
            return [(0, dimension)]
        base = dimension // num_splits
        remainder = dimension % num_splits
        bounds: List[Tuple[int, int]] = []
        start = 0
        for i in range(num_splits):
            size = base + (1 if i < remainder else 0)
            end = start + size
            bounds.append((start, end))
            start = end
        return bounds

