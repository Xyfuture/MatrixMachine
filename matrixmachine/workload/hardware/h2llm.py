#!/usr/bin/env python3
"""
H2 LLM Hardware Configuration

Hardware specifications based on H2 architecture:
- Single Die (one channel):
  - Internal bandwidth: 25.6 GB/s * 16 = 409.6 GB/s
  - External bandwidth: 6400MT/s * 16bit = 12.5 GB/s
  - Compute power: 16*8*16*0.6G = 1228.6 G = 1.2 TMACS (FP16)
  - Bandwidth scale: ~40x (internal vs external)
"""

from matrixmachine.core.description import ComputeDieSpec, ChipSpec, Chip


def create_h2llm_chip(die_count: int = 4) -> Chip:
    """
    Create an H2 LLM chip with configurable number of compute dies.

    Args:
        die_count: Number of compute dies in the chip (default: 4)

    Returns:
        Chip: Configured H2 LLM chip instance

    Example:
        >>> chip = create_h2llm_chip(die_count=8)
        >>> print(f"Total compute power: {chip.total_compute_power} TFLOPS")
    """
    # Create ComputeDie specification based on H2 architecture
    # Each die specifications:
    # - compute_power: 1.2 TFLOPS (1.2 TMACS for FP16)
    # - memory_bandwidth: 409.6 GB/s = 0.4096 TB/s
    # - shared_bandwidth: 12.5 GB/s (shared I/O bandwidth)
    die_spec = ComputeDieSpec(
        compute_power=1.2*4,           # TFLOPS
        shared_bandwidth=12.5,       # GB/s (shared I/O bandwidth)
        memory_bandwidth=0.4096      # TB/s (409.6 GB/s)
    )

    # Create chip with specified number of compute dies
    chip_spec = ChipSpec(die_count=die_count, die_spec=die_spec)
    chip = Chip.create_from_spec(chip_spec)

    return chip


def print_h2llm_chip_info(chip: Chip):
    """
    Print detailed information about H2 LLM chip configuration.

    Args:
        chip: H2 LLM chip instance to display information for
    """
    # Get spec from first die (die_ids are strings like "die_0")
    first_die = next(iter(chip.compute_dies.values()))
    die_spec = first_die.spec

    print("=== H2 LLM Hardware Configuration ===")
    print(f"Number of compute dies: {len(chip.compute_dies)}")
    print()
    print("Per Die Specifications:")
    print(f"  - Compute power: {die_spec.compute_power} TFLOPS")
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
    print("Example 1: H2 LLM chip with 4 compute dies")
    print("=" * 50)
    chip_4 = create_h2llm_chip(die_count=4)
    print_h2llm_chip_info(chip_4)

    print("\nExample 2: H2 LLM chip with 8 compute dies")
    print("=" * 50)
    chip_8 = create_h2llm_chip(die_count=8)
    print_h2llm_chip_info(chip_8)

    print("\nExample 3: H2 LLM chip with 16 compute dies")
    print("=" * 50)
    chip_16 = create_h2llm_chip(die_count=16)
    print_h2llm_chip_info(chip_16)