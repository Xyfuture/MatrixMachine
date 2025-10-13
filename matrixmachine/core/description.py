from __future__ import annotations

"""Core domain model for MatrixMachine simulations.

The module is organised around three layers:

1. Matrix primitives (`MatrixShape`, `Tile`) describe how data is partitioned.
2. Hardware specifications (`ComputeDieSpec`, `ChipSpec`) capture immutable
   configuration that can be instantiated into runtime objects.
3. Runtime entities (`ComputeDie`, `Chip`, `Mapping`) encapsulate mutable state
   such as live die metadata or tile ownership.

Keeping the configuration separate from instances makes it easier to reason
about how a chip should look (spec) versus how it currently behaves (runtime
objects). Builders provided on the runtime classes help translate specs into
instantiated objects with minimal boilerplate.
"""

from dataclasses import dataclass, field
from typing import Callable, ClassVar, Dict, Iterable, List, Optional, Tuple
from enum import Enum


# ---------------------------------------------------------------------------
# Matrix primitives
# ---------------------------------------------------------------------------


class DataType(Enum):
    """Supported data types for matrix computations."""

    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"
    FP8 = "fp8"

    @property
    def bytes_per_element(self) -> float:
        """Return the number of bytes per element for this data type."""
        return {
            DataType.FP16: 2.0,
            DataType.INT8: 1.0,
            DataType.INT4: 0.5,  # 4 bits = 0.5 bytes
            DataType.FP8: 1.0,
        }[self]


@dataclass(frozen=True)
class DataFormat:
    """Data format configuration for matrix computations."""

    input_dtype: DataType = DataType.FP16
    output_dtype: DataType = DataType.FP16
    weight_dtype: DataType = DataType.FP16

    def total_bytes_per_element(self) -> float:
        """Return the total bytes per element considering all data types.

        For a matrix multiplication, this represents the combined data footprint
        of input, weight, and output for a single computation.
        """
        return (
            self.input_dtype.bytes_per_element +
            self.output_dtype.bytes_per_element +
            self.weight_dtype.bytes_per_element
        )

    def __str__(self) -> str:
        return f"DataFormat(input={self.input_dtype.value}, output={self.output_dtype.value}, weight={self.weight_dtype.value})"


@dataclass(frozen=True)
class MatrixShape:
    """Matrix shape with convenience helpers for GEMV-like operations."""

    rows: int
    cols: int
    batch_size: int = 1
    data_format: DataFormat = DataFormat()

    @property
    def area(self) -> int:
        """Total number of elements in the matrix."""
        return self.rows * self.cols

    @property
    def volume(self) -> int:
        """Total number of elements including batch dimension."""
        return self.rows * self.cols * self.batch_size

    def to_tuple(self) -> Tuple[int, int]:
        return self.rows, self.cols

    def to_tuple_with_batch(self) -> Tuple[int, int, int]:
        return self.rows, self.cols, self.batch_size

    def contains(self, row: int, col: int, batch: int = 0) -> bool:
        return (0 <= row < self.rows and
                0 <= col < self.cols and
                0 <= batch < self.batch_size)

    def fits(self, tile: "Tile") -> bool:
        """Check whether the tile lies entirely within the matrix."""
        return (
            0 <= tile.row0 < tile.row1 <= self.rows
            and 0 <= tile.col0 < tile.col1 <= self.cols
            and 0 <= tile.batch0 < tile.batch1 <= self.batch_size
        )

    def to_bytes(self) -> float:
        """Calculate total bytes required for the matrix including all data types."""
        return self.volume * self.data_format.total_bytes_per_element()

    def __str__(self) -> str:
        if self.batch_size == 1:
            return f"MatrixShape({self.rows}×{self.cols}, {self.data_format})"
        else:
            return f"MatrixShape({self.rows}×{self.cols}×{self.batch_size}, {self.data_format})"


@dataclass(frozen=True)
class Tile:
    """Half-open rectangular region [row0, row1) x [col0, col1) x [batch0, batch1)."""

    tile_id: str
    row0: int
    row1: int
    col0: int
    col1: int
    batch0: int = 0
    batch1: int = 1
    data_format: DataFormat = DataFormat()

    _id_counter: ClassVar[int] = 1

    @classmethod
    def create(
        cls,
        row0: int,
        row1: int,
        col0: int,
        col1: int,
        batch0: int = 0,
        batch1: int = 1,
        *,
        prefix: str = "tile",
        data_format: Optional["DataFormat"] = None,
    ) -> "Tile":
        """Create a new tile with an auto-generated identifier."""

        cls._validate_bounds(row0, row1, col0, col1, batch0, batch1)
        tile_id = f"{prefix}_{cls._id_counter}"
        cls._id_counter += 1
        return cls(
            tile_id=tile_id,
            row0=row0,
            row1=row1,
            col0=col0,
            col1=col1,
            batch0=batch0,
            batch1=batch1,
            data_format=data_format or DataFormat()
        )

    @staticmethod
    def _validate_bounds(row0: int, row1: int, col0: int, col1: int, batch0: int = 0, batch1: int = 1) -> None:
        if not (row0 < row1 and col0 < col1 and batch0 < batch1):
            raise ValueError(
                "Tile bounds must satisfy row0 < row1, col0 < col1, and batch0 < batch1: "
                f"({row0}, {row1}, {col0}, {col1}, {batch0}, {batch1})"
            )

    @property
    def rows(self) -> int:
        return self.row1 - self.row0

    @property
    def cols(self) -> int:
        return self.col1 - self.col0

    @property
    def batches(self) -> int:
        return self.batch1 - self.batch0

    @property
    def shape(self) -> Tuple[int, int]:
        return self.rows, self.cols

    @property
    def shape_with_batch(self) -> Tuple[int, int, int]:
        return self.rows, self.cols, self.batches

    @property
    def area(self) -> int:
        return self.rows * self.cols

    @property
    def volume(self) -> int:
        return self.rows * self.cols * self.batches

    def intersects(self, other: "Tile") -> bool:
        """Whether tiles intersect (half-open interval)."""
        row_overlap = not (self.row1 <= other.row0 or other.row1 <= self.row0)
        col_overlap = not (self.col1 <= other.col0 or other.col1 <= self.col0)
        batch_overlap = not (self.batch1 <= other.batch0 or other.batch1 <= self.batch0)
        return row_overlap and col_overlap and batch_overlap

    def intersection(self, other: "Tile") -> Optional["Tile"]:
        """Return the intersection region as a new tile (None if disjoint)."""
        if not self.intersects(other):
            return None
        r0, r1 = max(self.row0, other.row0), min(self.row1, other.row1)
        c0, c1 = max(self.col0, other.col0), min(self.col1, other.col1)
        b0, b1 = max(self.batch0, other.batch0), min(self.batch1, other.batch1)
        return Tile(
            tile_id=f"({self.tile_id})&({other.tile_id})",
            row0=r0,
            row1=r1,
            col0=c0,
            col1=c1,
            batch0=b0,
            batch1=b1,
            data_format=self.data_format
        )

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return self.row0, self.row1, self.col0, self.col1

    def as_tuple_with_batch(self) -> Tuple[int, int, int, int, int, int]:
        return self.row0, self.row1, self.col0, self.col1, self.batch0, self.batch1

    def to_bytes(self) -> float:
        """Calculate total bytes required for the tile including all data types."""
        return self.volume * self.data_format.total_bytes_per_element()

    def __str__(self) -> str:
        if self.batch0 == 0 and self.batch1 == 1:
            return f"Tile({self.tile_id}:[{self.row0}:{self.row1},{self.col0}:{self.col1}])"
        else:
            return f"Tile({self.tile_id}:[{self.row0}:{self.row1},{self.col0}:{self.col1},{self.batch0}:{self.batch1}])"


# ---------------------------------------------------------------------------
# Hardware specification layer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComputeDieSpec:
    """Immutable specification for a compute die.

    Args:
        compute_power: Compute power in TFLOPS
        input_bandwidth: Input bandwidth in GB/s (optional if shared_bandwidth is set)
        output_bandwidth: Output bandwidth in GB/s (optional if shared_bandwidth is set)
        memory_bandwidth: Memory bandwidth in TB/s
        shared_bandwidth: Shared I/O bandwidth in GB/s (mutually exclusive with input/output bandwidth)
    """

    compute_power: float  # TFLOPS
    input_bandwidth: Optional[float] = None  # GB/s
    output_bandwidth: Optional[float] = None  # GB/s
    memory_bandwidth: float = 0.0  # TB/s
    shared_bandwidth: Optional[float] = None  # GB/s

    def __post_init__(self) -> None:
        if self.compute_power <= 0:
            raise ValueError("compute_power must be positive")
        if self.memory_bandwidth <= 0:
            raise ValueError("memory_bandwidth must be positive")

        # Validate bandwidth configuration
        has_separate = self.input_bandwidth is not None or self.output_bandwidth is not None
        has_shared = self.shared_bandwidth is not None

        if has_separate and has_shared:
            raise ValueError(
                "Cannot specify both separate bandwidths (input_bandwidth/output_bandwidth) "
                "and shared_bandwidth at the same time"
            )

        if not has_separate and not has_shared:
            raise ValueError(
                "Must specify either separate bandwidths (input_bandwidth and output_bandwidth) "
                "or shared_bandwidth"
            )

        if has_separate:
            if self.input_bandwidth is None or self.input_bandwidth <= 0:
                raise ValueError("input_bandwidth must be positive when using separate bandwidths")
            if self.output_bandwidth is None or self.output_bandwidth <= 0:
                raise ValueError("output_bandwidth must be positive when using separate bandwidths")

        if has_shared:
            if self.shared_bandwidth is None or self.shared_bandwidth <= 0:
                raise ValueError("shared_bandwidth must be positive when using shared bandwidth mode")

    def get_input_bandwidth(self) -> float:
        """Get the effective input bandwidth.

        Returns shared_bandwidth if in shared mode, otherwise returns input_bandwidth.
        """
        if self.shared_bandwidth is not None:
            return self.shared_bandwidth
        assert self.input_bandwidth is not None
        return self.input_bandwidth

    def get_output_bandwidth(self) -> float:
        """Get the effective output bandwidth.

        Returns shared_bandwidth if in shared mode, otherwise returns output_bandwidth.
        """
        if self.shared_bandwidth is not None:
            return self.shared_bandwidth
        assert self.output_bandwidth is not None
        return self.output_bandwidth

    def scale(self, factor: float) -> "ComputeDieSpec":
        """Return a scaled copy (useful for quick what-if experiments)."""
        if factor <= 0:
            raise ValueError("scale factor must be positive")

        if self.shared_bandwidth is not None:
            return ComputeDieSpec(
                compute_power=self.compute_power * factor,
                memory_bandwidth=self.memory_bandwidth * factor,
                shared_bandwidth=self.shared_bandwidth * factor,
            )
        else:
            return ComputeDieSpec(
                compute_power=self.compute_power * factor,
                input_bandwidth=self.input_bandwidth * factor if self.input_bandwidth else None,
                output_bandwidth=self.output_bandwidth * factor if self.output_bandwidth else None,
                memory_bandwidth=self.memory_bandwidth * factor,
            )

    def __str__(self) -> str:
        if self.shared_bandwidth is not None:
            return (f"ComputeDieSpec(compute={self.compute_power}TFLOPS, "
                    f"shared_bandwidth={self.shared_bandwidth}GB/s, "
                    f"memory={self.memory_bandwidth}TB/s)")
        else:
            return (f"ComputeDieSpec(compute={self.compute_power}TFLOPS, "
                    f"input={self.input_bandwidth}GB/s, output={self.output_bandwidth}GB/s, "
                    f"memory={self.memory_bandwidth}TB/s)")


@dataclass(frozen=True)
class ChipSpec:
    """Immutable blueprint for a chip composed of homogeneous dies."""

    die_count: int
    die_spec: ComputeDieSpec

    def __post_init__(self) -> None:
        if self.die_count <= 0:
            raise ValueError("die_count must be positive")

    def __str__(self) -> str:
        return f"ChipSpec({self.die_count} dies, {self.die_spec})"

# Backwards compatibility aliases (prefer the *Spec* names going forward).
ComputeDieConfig = ComputeDieSpec
ChipConfig = ChipSpec


# ---------------------------------------------------------------------------
# Runtime hardware entities
# ---------------------------------------------------------------------------


@dataclass
class ComputeDie:
    """Runtime representation of a compute die."""

    die_id: str
    spec: ComputeDieSpec
    meta: Dict[str, str] = field(default_factory=dict)

  
    @property
    def compute_power(self) -> float:
        return self.spec.compute_power

    @property
    def input_bandwidth(self) -> float:
        return self.spec.get_input_bandwidth()

    @property
    def output_bandwidth(self) -> float:
        return self.spec.get_output_bandwidth()

    @property
    def memory_bandwidth(self) -> float:
        return self.spec.memory_bandwidth

    def clone(self, *, die_id: Optional[str] = None, meta: Optional[Dict[str, str]] = None) -> "ComputeDie":
        """Create a shallow copy, optionally overriding identifier or metadata."""
        return ComputeDie(
            die_id=die_id or self.die_id,
            spec=self.spec,
            meta=dict(self.meta if meta is None else meta),
        )

    def __str__(self) -> str:
        meta_str = f", meta={self.meta}" if self.meta else ""
        return f"ComputeDie({self.die_id}, {self.spec}{meta_str})"


@dataclass
class Chip:
    """Runtime chip composed of instantiated compute dies."""

    spec: ChipSpec
    compute_dies: Dict[str, ComputeDie] = field(default_factory=dict)

    @classmethod
    def create_from_spec(
        cls,
        spec: ChipSpec,
        *,
        id_prefix: str = "die",
        meta_factory: Optional[Callable[[int], Dict[str, str]]] = None,
    ) -> "Chip":
        """Create a Chip instance populated with compute dies from the spec."""

        compute_dies: Dict[str, ComputeDie] = {}
        for idx in range(spec.die_count):
            die_id = f"{id_prefix}_{idx}"
            meta = meta_factory(idx) if meta_factory else {}
            compute_dies[die_id] = ComputeDie(die_id=die_id, spec=spec.die_spec, meta=meta)
        return cls(spec=spec, compute_dies=compute_dies)

    def add_die(self, die: ComputeDie) -> None:
        self.compute_dies[die.die_id] = die

    def remove_die(self, die_id: str) -> None:
        self.compute_dies.pop(die_id, None)

    def get_die(self, die_id: str) -> Optional[ComputeDie]:
        return self.compute_dies.get(die_id)

    @property
    def total_compute_power(self) -> float:
        return sum(die.compute_power for die in self.compute_dies.values())

    @property
    def total_compute_power_gops(self) -> float:
        return self.total_compute_power * 10**3  # Convert TOPS to GOPS

    @property
    def total_input_bandwidth(self) -> float:
        return sum(die.input_bandwidth for die in self.compute_dies.values())

    @property
    def total_output_bandwidth(self) -> float:
        return sum(die.output_bandwidth for die in self.compute_dies.values())

    @property
    def total_bandwidth(self) -> float:
        return self.total_input_bandwidth + self.total_output_bandwidth

    @property
    def total_memory_bandwidth(self) -> float:
        return sum(die.memory_bandwidth for die in self.compute_dies.values())

    def __str__(self) -> str:
        die_count = len(self.compute_dies)
        total_compute = self.total_compute_power
        return f"Chip({die_count} dies, {total_compute}TFLOPS total compute)"


# ---------------------------------------------------------------------------
# Mapping and validation
# ---------------------------------------------------------------------------

TileAssignmentInput = Tuple[str, int, int, int, int, int, int]


@dataclass
class Mapping:
    """Bidirectional mapping between tiles and compute dies."""

    matrix: MatrixShape
    chip: Chip
    tiles: Dict[str, Tile] = field(default_factory=dict)
    placement: Dict[str, List[Tile]] = field(default_factory=dict)
    reverse_placement: Dict[str, ComputeDie] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for die_id in self.chip.compute_dies:
            self.placement.setdefault(die_id, [])

    @classmethod
    def from_tile_data(
        cls,
        matrix: MatrixShape,
        chip: Chip,
        tile_data: Iterable[TileAssignmentInput],
    ) -> "Mapping":
        mapping = cls(matrix=matrix, chip=chip)
        mapping.build(tile_data)
        mapping.check_all()
        return mapping

    def add_tile(
        self,
        die_id: str,
        row0: int,
        row1: int,
        col0: int,
        col1: int,
        batch0: int = 0,
        batch1: int = 1,
        *,
        tile_id: Optional[str] = None,
    ) -> Tile:
        Tile._validate_bounds(row0, row1, col0, col1, batch0, batch1)
        tile = (
            Tile(
                tile_id=tile_id,
                row0=row0,
                row1=row1,
                col0=col0,
                col1=col1,
                batch0=batch0,
                batch1=batch1,
                data_format=self.matrix.data_format
            )
            if tile_id is not None
            else Tile.create(
                row0=row0,
                row1=row1,
                col0=col0,
                col1=col1,
                batch0=batch0,
                batch1=batch1,
                data_format=self.matrix.data_format
            )
        )
        self._register_tile(die_id, tile)
        return tile

    def build(self, tile_data: Iterable[TileAssignmentInput]) -> None:
        for die_id, row0, row1, col0, col1, batch0, batch1 in tile_data:
            self.add_tile(die_id, row0, row1, col0, col1, batch0, batch1)

    def _register_tile(self, die_id: str, tile: Tile) -> None:
        if die_id not in self.chip.compute_dies:
            raise ValueError(f"Unknown compute die: {die_id}")

        self.tiles[tile.tile_id] = tile
        self.placement.setdefault(die_id, []).append(tile)
        self.reverse_placement[tile.tile_id] = self.chip.compute_dies[die_id]

    # ----------
    # Validation helpers
    # ----------

    def check_all(self) -> None:
        """Execute full validation; raises ValueError on failure."""
        self._check_tiles_exist_in_placement()
        self._check_bounds()
        self._check_overlap_free()
        self._check_full_coverage()

    def _check_tiles_exist_in_placement(self) -> None:
        seen = set()
        for die, tiles in self.placement.items():
            for tile in tiles:
                if tile.tile_id not in self.tiles:
                    raise ValueError(
                        f"Placement referenced non-existent tile: {tile.tile_id} @ die={die}"
                    )
                if tile.tile_id in seen:
                    raise ValueError(f"Tile appears on multiple dies: {tile.tile_id}")
                seen.add(tile.tile_id)
        if seen != set(self.tiles.keys()):
            missing = set(self.tiles.keys()) - seen
            raise ValueError(f"Unassigned tiles exist: {sorted(missing)}")

    def _check_bounds(self) -> None:
        for tile in self.tiles.values():
            if not self.matrix.fits(tile):
                raise ValueError(
                    "Tile out of bounds or illegal range: "
                    f"{tile.tile_id} -> ({tile.row0}:{tile.row1}, {tile.col0}:{tile.col1}, {tile.batch0}:{tile.batch1})"
                )

    def _check_overlap_free(self) -> None:
        items = list(self.tiles.values())
        n = len(items)
        for i in range(n):
            for j in range(i + 1, n):
                if items[i].intersects(items[j]):
                    inter = items[i].intersection(items[j])
                    if inter is not None:
                        raise ValueError(
                            f"Tile overlap: {items[i].tile_id} intersects with {items[j].tile_id} at "
                            f"[{inter.row0}:{inter.row1}, {inter.col0}:{inter.col1}, {inter.batch0}:{inter.batch1}]"
                        )

    def _check_full_coverage(self) -> None:
        total = sum(t.volume for t in self.tiles.values())
        if total != self.matrix.volume:
            raise ValueError(
                "Coverage volume not equal to matrix volume: "
                f"tiles={total}, matrix={self.matrix.volume}, may have holes or out of bounds"
            )

    # ----------
    # Convenience statistics
    # ----------

    def die_loads(self) -> Dict[str, int]:
        return {die: len(tiles) for die, tiles in self.placement.items()}

    def die_areas(self) -> Dict[str, int]:
        return {
            die: sum(tile.area for tile in tiles)
            for die, tiles in self.placement.items()
        }

    def die_volumes(self) -> Dict[str, int]:
        return {
            die: sum(tile.volume for tile in tiles)
            for die, tiles in self.placement.items()
        }

    def tiles_of_die(self, die_id: str) -> List[Tile]:
        return list(self.placement.get(die_id, []))

    def display(self) -> None:
        """Display mapping summary with matrix size and tile assignments per die."""
        print(f"Matrix: {self.matrix.rows}×{self.matrix.cols}×{self.matrix.batch_size}")
        for die_id in sorted(self.placement.keys()):
            tiles = self.placement[die_id]
            if tiles:
                tile_info = []
                for tile in tiles:
                    tile_info.append(f"{tile.rows}×{tile.cols}×{tile.batches}")
                print(f"  {die_id}: {', '.join(tile_info)}")
            else:
                print(f"  {die_id}: (empty)")

    # ----------
    # Bidirectional mapping checks
    # ----------

    def check_bidirectional_mapping(self) -> bool:
        for die_id, tiles in self.placement.items():
            for tile in tiles:
                compute_die = self.reverse_placement.get(tile.tile_id)
                if compute_die is None or compute_die.die_id != die_id:
                    return False

        for tile_id, compute_die in self.reverse_placement.items():
            if tile_id not in self.tiles:
                return False
            assigned_tiles = self.placement.get(compute_die.die_id, [])
            if not any(tile.tile_id == tile_id for tile in assigned_tiles):
                return False

        for die_id, tiles in self.placement.items():
            for tile in tiles:
                if tile.tile_id not in self.reverse_placement:
                    return False

        return True

    def __str__(self) -> str:
        tile_count = len(self.tiles)
        die_count = len([die for die, tiles in self.placement.items() if tiles])
        return f"Mapping({tile_count} tiles on {die_count} active dies, matrix={self.matrix})"
