from matrixmachine.description import MatrixShape, ComputeDieSpec, ChipSpec, Chip
from matrixmachine.strategy.grid import GridTilingStrategy
from matrixmachine.sim_engine import SimChip


def main() -> None:
    # Keep matrix and hardware config identical to main.py
    matrix_shape = MatrixShape(rows=4096, cols=4096)

    die_spec = ComputeDieSpec(
        compute_power=128,  # 128 TFLOPS
        input_bandwidth=128,  # 128 GB/s
        output_bandwidth=128,  # 128 GB/s
    )

    chip_spec = ChipSpec(die_count=8, die_spec=die_spec)
    chip = Chip.create_from_spec(chip_spec)

    print("Matrix:", matrix_shape.rows, "x", matrix_shape.cols)
    print("Chip:")
    print("  dies:", len(chip.compute_dies))
    print("  total compute:", chip.total_compute_power, "TFLOPS")
    print("  total bandwidth:", chip.total_bandwidth, "GB/s")
    print()

    # Provide a compact grid pool to bound search time on large matrices
    grid_pool = [
        (512, 512),
        (256, 512),
        (512, 256),
        (256, 256),
        (128, 256),
        (256, 128),
        (128, 128),
    ]

    strategy = GridTilingStrategy(stage2_max_per_rect=2)
    best_mapping, best_cycles, stats = strategy.dse_best_mapping(
        matrix_shape, chip, grid_pool=grid_pool
    )

    print("DSE finished:")
    print("  candidates:", stats.get("candidates", 0))
    print("  best cycles:", best_cycles)
    print("  tiles:", len(best_mapping.tiles))
    print("  valid bimap:", best_mapping.check_bidirectional_mapping())
    print()

    print("Tile distribution (per die):")
    for die_id, tiles in best_mapping.placement.items():
        area = sum(t.area for t in tiles)
        print(f"  {die_id}: {len(tiles)} tiles, area={area}")
        # Print each tile with coordinates and size
        for t in tiles:
            print(
                f"    {t.tile_id}: [r:{t.row0}:{t.row1}, c:{t.col0}:{t.col1}] "
                f"size={t.rows}x{t.cols} area={t.area}"
            )

        # Also print a compact summary by tile size
        size_hist = {}
        for t in tiles:
            key = (t.rows, t.cols)
            size_hist[key] = size_hist.get(key, 0) + 1
        print("    size histogram:")
        for (rh, cw), cnt in sorted(size_hist.items()):
            print(f"      {rh}x{cw}: {cnt}")

    # Save Perfetto trace for the best candidate
    trace_path = "trace.json"
    sim_chip = SimChip(chip)
    sim_chip.set_mapping(best_mapping)
    sim_chip.run_sim()
    sim_chip.save_trace_file(trace_path)
    print(f"Saved trace to {trace_path}")


if __name__ == "__main__":
    main()
