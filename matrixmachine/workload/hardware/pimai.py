#!/usr/bin/env python3
"""
PiMai Hardware Configuration

Hardware specifications based on PiMai architecture:
- Single Die:
  - Internal bandwidth: 102 GB/s (推测是 8-bank 全开的 LPDDR5)
  - External bandwidth: 12.8 GB/s (推测是 6400MT/s 的 LPDDR5)
  - Compute power: 8 TOPS = 4 TMACS
  - Bandwidth scale: ~8x (internal vs external)
  - Compute scale: ~300x (考虑 4 倍的 shift, 可以达到 1200)
"""

from matrixmachine.core.description import ComputeDieSpec, ChipSpec, Chip


def create_pimai_chip(die_count: int = 4) -> Chip:
    """
    Create a PiMai chip with configurable number of compute dies.

    Args:
        die_count: Number of compute dies in the chip (default: 4)

    Returns:
        Chip: Configured PiMai chip instance

    Example:
        >>> chip = create_pimai_chip(die_count=8)
        >>> print(f"Total compute power: {chip.total_compute_power} TFLOPS")
    """
    # Create ComputeDie specification based on PiMai architecture
    # Each die specifications:
    # - compute_power: 4.0 TFLOPS (8 TOPS = 4 TMACS)
    # - memory_bandwidth: 102 GB/s = 0.102 TB/s (LPDDR5 8-bank)
    # - shared_bandwidth: 12.8 GB/s (shared I/O bandwidth)
    die_spec = ComputeDieSpec(
        compute_power=4.0,           # TFLOPS (8 TOPS = 4 TMACS)
        shared_bandwidth=12.8,       # GB/s (shared I/O bandwidth)
        memory_bandwidth=0.102       # TB/s (102 GB/s)
    )

    # Create chip with specified number of compute dies
    chip_spec = ChipSpec(die_count=die_count, die_spec=die_spec)
    chip = Chip.create_from_spec(chip_spec)

    return chip


def print_pimai_chip_info(chip: Chip):
    """
    Print detailed information about PiMai chip configuration.

    Args:
        chip: PiMai chip instance to display information for
    """
    die_spec = chip.compute_dies[0].spec  # Get spec from first die

    print("=== PiMai Hardware Configuration ===")
    print(f"Number of compute dies: {len(chip.compute_dies)}")
    print()
    print("Per Die Specifications:")
    print(f"  - Compute power: {die_spec.compute_power} TFLOPS (8 TOPS = 4 TMACS)")
    print(f"  - Memory bandwidth: {die_spec.memory_bandwidth} TB/s ({die_spec.memory_bandwidth * 1000} GB/s)")
    print(f"  - Shared I/O bandwidth: {die_spec.shared_bandwidth} GB/s")
    print(f"  - Bandwidth scale: {die_spec.memory_bandwidth * 1000 / die_spec.shared_bandwidth:.1f}x (internal/external)")
    print()
    print("Total Chip Specifications:")
    print(f"  - Total compute power: {chip.total_compute_power} TFLOPS")
    print(f"  - Total memory bandwidth: {die_spec.memory_bandwidth * len(chip.compute_dies)} TB/s")
    print(f"  - Total shared I/O bandwidth: {die_spec.shared_bandwidth} GB/s")
    print()


if __name__ == "__main__":
    # Example usage with different die counts
    print("Example 1: PiMai chip with 4 compute dies")
    print("=" * 50)
    chip_4 = create_pimai_chip(die_count=4)
    print_pimai_chip_info(chip_4)

    print("\nExample 2: PiMai chip with 8 compute dies")
    print("=" * 50)
    chip_8 = create_pimai_chip(die_count=8)
    print_pimai_chip_info(chip_8)

    print("\nExample 3: PiMai chip with 16 compute dies")
    print("=" * 50)
    chip_16 = create_pimai_chip(die_count=16)
    print_pimai_chip_info(chip_16)