from __future__ import annotations

"""Recursive grid mapping and simple DSE for MatrixMachine.

This strategy explores grid (tile-size) candidates, performs a two-stage
tiling-and-assignment, evaluates each mapping via the simulation backend, and
returns the fastest mapping.

Algorithm sketch (matches the user's request in Chinese):

- Prepare a pool of grid sizes (tile_height, tile_width).
- For each first-round grid size, split the matrix into as many full tiles as
  possible and assign them to compute dies in round-robin order.
- Compute the leftover area (right strip, bottom strip, bottom-right corner).
  For the second (and final) round, try to find grid sizes that tile each
  leftover rectangle exactly and whose tile count is divisible by the number of
  dies; otherwise, fall back to slicing that rectangle into `die_count` vertical
  or horizontal strips exactly once.
- Enumerate the combinations of second-round options across the leftover
  rectangles, build candidate mappings, "soak time" via the simulator, and pick
  the one with the shortest simulated time.

Notes
- The simulator processes each die's tiles sequentially in FIFO order, so we
  append second-round tiles after first-round tiles to reflect the two rounds.
- The pool defaults are intentionally small to keep search time modest; callers
  can supply their own `grid_pool` for broader DSE.
"""

from dataclasses import dataclass
from itertools import product
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from ..description import Chip, Mapping, MatrixShape, Tile
from ..sim_engine import simulate


GridSize = Tuple[int, int]  # (tile_height, tile_width)
Rect = Tuple[int, int, int, int]  # row0, row1, col0, col1 (half-open)


@dataclass
class GridTilingStrategy:
    """Two-round recursive grid mapping with a DSE loop.

    Parameters
    - grid_pool: Optional list of `(tile_h, tile_w)` candidates. If omitted,
      a compact default pool is derived from the matrix shape during DSE.
    - stage2_max_per_rect: Cap on how many exact-fit grid options to try per
      leftover rectangle to control combinatorics.
    - time_evaluator: Optional callable `(chip, mapping) -> int` that returns a
      cycle count. Defaults to `simulate` from `sim_engine`.
    """

    grid_pool: Optional[List[GridSize]] = None
    stage2_max_per_rect: int = 3
    time_evaluator: Optional[Callable[[Chip, Mapping], int]] = None

    # --------------------
    # Public API
    # --------------------

    def dse_best_mapping(
        self,
        matrix: MatrixShape,
        chip: Chip,
        grid_pool: Optional[Sequence[GridSize]] = None,
    ) -> Tuple[Mapping, int, Dict[str, int]]:
        """Run DSE, evaluate candidates, and return the best mapping.

        Returns
        - mapping: Best mapping under the selected time evaluator
        - cycles: Simulated cycles for that mapping
        - stats:  Small dict with counters (e.g., candidates evaluated)
        """

        if matrix.rows <= 0 or matrix.cols <= 0:
            raise ValueError("matrix dimensions must be positive")
        if not chip.compute_dies:
            raise ValueError("Chip must have at least one compute die")

        pool = list(grid_pool or self.grid_pool or self._default_grid_pool(matrix))
        if not pool:
            raise ValueError("grid_pool is empty; provide candidates or rely on defaults")

        candidates_evaluated = 0
        best_cycles: Optional[int] = None
        best_mapping: Optional[Mapping] = None

        # Enumerate first-round grid choices
        for tile_h, tile_w in pool:
            if tile_h <= 0 or tile_w <= 0:
                continue
            if tile_h > matrix.rows or tile_w > matrix.cols:
                continue

            mappings = self._enumerate_two_stage_mappings(matrix, chip, (tile_h, tile_w), pool)
            for mapping in mappings:
                # Evaluate mapping runtime ("泡一下时间")
                evaluator = self.time_evaluator or simulate
                cycles = evaluator(chip, mapping)
                candidates_evaluated += 1
                if best_cycles is None or cycles < best_cycles:
                    best_cycles = cycles
                    best_mapping = mapping

        if best_mapping is None or best_cycles is None:
            raise RuntimeError("DSE failed to produce any candidate mappings")

        stats = {"candidates": candidates_evaluated}
        return best_mapping, best_cycles, stats

    # --------------------
    # Core mapping enumeration
    # --------------------

    def _enumerate_two_stage_mappings(
        self, matrix: MatrixShape, chip: Chip, stage1_tile: GridSize, pool: Sequence[GridSize]
    ) -> Iterable[Mapping]:
        """Yield all second-stage combinations given a first-stage tile size.

        Stage 1: pack as many full tiles of size `stage1_tile` as possible.
        Stage 2: for each leftover rectangle, try exact-fit grid options from
                 `pool` (filtered) or fall back to vertical/horizontal strips.
        """

        die_ids = sorted(chip.compute_dies.keys())
        die_count = len(die_ids)

        th1, tw1 = stage1_tile
        r_full = matrix.rows // th1
        c_full = matrix.cols // tw1

        # Nothing to place if no full tiles; skip this stage-1 choice.
        if r_full == 0 or c_full == 0:
            return []

        # Stage-1 tiles (full tiles only)
        stage1_tiles: List[Tile] = []
        for r in range(r_full):
            for c in range(c_full):
                row0 = r * th1
                row1 = row0 + th1
                col0 = c * tw1
                col1 = col0 + tw1
                stage1_tiles.append(Tile.create(row0, row1, col0, col1, prefix="s1"))

        # Leftover rectangles: right strip, bottom strip, bottom-right corner
        H1 = r_full * th1
        W1 = c_full * tw1
        leftovers: List[Tuple[str, Rect]] = []
        if W1 < matrix.cols and H1 > 0:
            leftovers.append(("right", (0, H1, W1, matrix.cols)))
        if H1 < matrix.rows and W1 > 0:
            leftovers.append(("bottom", (H1, matrix.rows, 0, W1)))
        if H1 < matrix.rows and W1 < matrix.cols:
            leftovers.append(("corner", (H1, matrix.rows, W1, matrix.cols)))

        # Enumerate second-stage options per leftover rectangle
        per_rect_options: List[List[Tuple[str, object]]] = []
        # Each option item is (kind, payload) where kind in {"grid", "vsplit", "hsplit"}
        # - grid payload: (tile_h, tile_w)
        # - vsplit/hsplit payload: None (implies die_count slices)
        for _, rect in leftovers:
            r0, r1, c0, c1 = rect
            h = r1 - r0
            w = c1 - c0
            opts: List[Tuple[str, object]] = []

            # Grid options that tile exactly and evenly across dies
            exacts: List[GridSize] = []
            for th2, tw2 in pool:
                if th2 <= 0 or tw2 <= 0:
                    continue
                if h % th2 != 0 or w % tw2 != 0:
                    continue
                tcount = (h // th2) * (w // tw2)
                if tcount % die_count == 0:
                    exacts.append((th2, tw2))

            # Prefer smaller tiles first to improve load balance; cap list size
            exacts.sort(key=lambda x: (x[0] * x[1], x[0], x[1]))
            for item in exacts[: max(1, self.stage2_max_per_rect)]:
                opts.append(("grid", item))

            # If no exact grid options, add fallbacks (vertical/horizontal split)
            if not opts:
                opts.append(("vsplit", None))
                opts.append(("hsplit", None))

            per_rect_options.append(opts)

        # If there are no leftovers at all, yield a single mapping with stage-1 only.
        if not per_rect_options:
            mapping = Mapping(matrix=matrix, chip=chip)
            # Round-robin assign stage-1 tiles
            for idx, tile in enumerate(stage1_tiles):
                die_id = die_ids[idx % die_count]
                mapping._register_tile(die_id, tile)
            mapping.check_all()
            yield mapping
            return

        # Build all combinations across rectangles
        for combo in product(*per_rect_options):
            # Prepare a fresh mapping for this combination
            mapping = Mapping(matrix=matrix, chip=chip)

            # Stage 1 assignment
            rr = 0
            for tile in stage1_tiles:
                die_id = die_ids[rr % die_count]
                mapping._register_tile(die_id, tile)
                rr += 1

            # Stage 2: expand each rectangle according to chosen option
            for (rect_label, rect), (kind, payload) in zip(leftovers, combo):
                r0, r1, c0, c1 = rect
                rect_tiles: List[Tile] = []
                if kind == "grid":
                    th2, tw2 = payload  # type: ignore[assignment]
                    for rr_idx in range((r1 - r0) // th2):
                        for cc_idx in range((c1 - c0) // tw2):
                            t_r0 = r0 + rr_idx * th2
                            t_r1 = t_r0 + th2
                            t_c0 = c0 + cc_idx * tw2
                            t_c1 = t_c0 + tw2
                            rect_tiles.append(
                                Tile.create(t_r0, t_r1, t_c0, t_c1, prefix=f"s2-{rect_label}")
                            )
                elif kind == "vsplit":
                    # Split into die_count vertical strips (full height)
                    widths = _balanced_splits(c1 - c0, die_count)
                    cstart = c0
                    for wsplit in widths:
                        if wsplit <= 0:
                            continue
                        rect_tiles.append(
                            Tile.create(r0, r1, cstart, cstart + wsplit, prefix=f"s2v-{rect_label}")
                        )
                        cstart += wsplit
                elif kind == "hsplit":
                    # Split into die_count horizontal strips (full width)
                    heights = _balanced_splits(r1 - r0, die_count)
                    rstart = r0
                    for hsplit in heights:
                        if hsplit <= 0:
                            continue
                        rect_tiles.append(
                            Tile.create(rstart, rstart + hsplit, c0, c1, prefix=f"s2h-{rect_label}")
                        )
                        rstart += hsplit
                else:
                    raise AssertionError(f"Unknown second-stage option kind: {kind}")

                # Assign the rectangle's tiles round-robin, continuing from rr
                for tile in rect_tiles:
                    die_id = die_ids[rr % die_count]
                    mapping._register_tile(die_id, tile)
                    rr += 1

            # Validate full coverage and yield
            mapping.check_all()
            yield mapping

    # --------------------
    # Helpers
    # --------------------

    def _default_grid_pool(self, matrix: MatrixShape) -> List[GridSize]:
        """Construct a compact, useful pool of tile sizes.

        The pool is based on simple divisors and powers-of-two like sizes. It
        includes pairs up to a moderate limit to avoid explosion.
        """

        rows, cols = matrix.rows, matrix.cols

        def divisors(n: int) -> List[int]:
            ds: List[int] = []
            for k in range(1, min(n, 16) + 1):
                # Prefer larger tiles by stepwise division
                val = max(1, n // k)
                if val not in ds:
                    ds.append(val)
            # Powers of two up to n
            p = 1
            while p <= n:
                if p not in ds:
                    ds.append(p)
                p *= 2
            ds.sort()
            return ds

        r_cands = [x for x in divisors(rows) if 1 <= x <= rows]
        c_cands = [x for x in divisors(cols) if 1 <= x <= cols]

        pool: List[GridSize] = []
        for rh in r_cands:
            for cw in c_cands:
                pool.append((rh, cw))

        # Deduplicate while preserving order
        seen = set()
        uniq: List[GridSize] = []
        for item in pool:
            if item not in seen:
                seen.add(item)
                uniq.append(item)
        # A light cap to avoid very large searches
        return uniq[:64]


def _balanced_splits(total: int, parts: int) -> List[int]:
    """Split an integer into `parts` non-negative integers that sum to total.

    This keeps sizes within +-1 of each other, similar to how the trivial grid
    strategy splits dimensions. Zero-length segments can occur if `parts > total`.
    """

    if parts <= 0:
        raise ValueError("parts must be positive")
    base = total // parts
    extra = total % parts
    splits = [base + (1 if i < extra else 0) for i in range(parts)]
    return splits
