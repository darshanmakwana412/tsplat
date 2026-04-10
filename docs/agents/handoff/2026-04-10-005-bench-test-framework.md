# tsplat benchmark and regression test framework

Date: 2026-04-10
Session focus: building a testing and benchmarking framework to support aggressive CPU optimization of the Gaussian splatting forward pass.

## What was accomplished

- Created `src/lib.rs` to re-export core modules (`camera`, `framebuffer`, `rasterize`, `sh`, `splat`) so that integration tests and benchmarks can import them as a library crate.
- Added `criterion` as a dev-dependency in `Cargo.toml` with a `[[bench]]` section for `forward_pass`.
- Created `benches/forward_pass.rs` with five Criterion benchmark groups: `project_200k`, `sort_by_depth_200k`, `composite_200k`, `render_halfblocks`, and `full_pipeline_200k`. All use fixed camera parameters captured from the HUD screenshot of the garden scene.
- Created `tests/regression.rs` with three integration tests:
  - `forward_pass_framebuffer_regression`: saves a binary reference framebuffer on first run, then compares pixel-by-pixel with tolerance (hard fail at >0.01 diff per float component) on subsequent runs.
  - `forward_pass_deterministic`: verifies two identical pipeline runs produce byte-identical output.
  - `forward_pass_ansi_output_nonempty`: sanity checks that the ANSI half-block output is non-trivial and well-formed.
- Created `src/bin/bench_forward.rs`, a standalone benchmark binary that runs N frames with fixed camera parameters and prints a per-stage timing table with mean, min, max, p50, p95, p99 in microseconds, plus a percentage breakdown. Accepts CLI flags for `--frames`, `--max-splats`, `--width`, `--height`, and `--warmup`.
- Updated `.gitignore` to exclude `tests/reference/` (machine-specific binary reference data) and `target/criterion/` (benchmark artifacts).
- Verified everything builds cleanly, all three regression tests pass, Criterion benchmarks complete, and the standalone bench binary produces correct output.

## Files changed

| File | Nature of change |
|------|-----------------|
| `Cargo.toml` | Added `criterion` dev-dependency and `[[bench]]` section |
| `src/lib.rs` | New file, re-exports core modules for external test/bench access |
| `benches/forward_pass.rs` | New file, Criterion benchmark suite for all pipeline stages |
| `tests/regression.rs` | New file, correctness regression tests with reference framebuffer |
| `src/bin/bench_forward.rs` | New file, standalone timing stats binary |
| `.gitignore` | Added `tests/reference/` and `target/criterion/` |

Files not changed: `main.rs`, `splat.rs`, `camera.rs`, `rasterize.rs`, `framebuffer.rs`, `sh.rs`, `hud.rs`.

## Key decisions

1. **Fixed camera parameters from the HUD screenshot.** All benchmarks and tests use the same camera state: yaw=-0.080, pitch=+0.353, radius=52.161, fov_y=55 deg, target=(4.14, -11.97, -38.58). This ensures reproducibility across runs and sessions. The terminal resolution is fixed at 120x80 pixels (120 cols, 40 rows with half-block doubling).

2. **Binary reference file rather than a checksum.** The regression test stores the full framebuffer as raw little-endian f32 values rather than just a hash. This allows per-pixel diff reporting when a regression is detected, making it much easier to diagnose what changed. The reference file is about 38KB (120*80*4 floats * 4 bytes).

3. **Tolerance-based comparison, not exact match.** The regression test allows up to 0.01 absolute difference per float component to accommodate minor floating-point drift across compiler versions or LLVM optimization changes. Exact bitwise equality is tested separately in the determinism test (same binary, same inputs).

4. **Three-tier benchmarking approach.** Criterion gives statistically rigorous micro-benchmarks with automatic regression detection between runs. The standalone binary gives a quick, human-readable summary with percentile stats. The regression test catches correctness bugs. Each serves a different purpose during optimization work.

5. **lib.rs alongside main.rs.** Rust requires a `lib.rs` for integration tests and benchmarks to import project types. The `main.rs` continues to use `mod` declarations for its own modules, while `lib.rs` re-exports the same modules. The `hud` module is not re-exported since it is only used by the interactive viewer.

6. **Reference file is not committed.** The binary reference is machine-specific (floating point results can vary across CPU architectures and compiler versions). It is generated on first test run and gitignored. Each developer/CI machine generates its own baseline.

## Baseline performance numbers

Measured on the development machine with 200k splats at 120x80 resolution:

| Stage | Mean | % of frame |
|-------|------|-----------|
| project | 3.6ms | 10.0% |
| sort | 6.7ms | 18.6% |
| composite | 25.2ms | 70.1% |
| halfblocks | 0.5ms | 1.3% |
| Total | 36.0ms | ~28 FPS |

Composite is the dominant bottleneck at 70% of frame time. Sort is second at 19%. Project is relatively cheap. Halfblock rendering is negligible.

## Important context for future sessions

### Running the benchmarks

```
# Quick stats overview (standalone binary)
cargo run --release --bin bench_forward
cargo run --release --bin bench_forward -- --frames 200 --max-splats 500000

# Criterion (statistical, detects regressions between runs)
cargo bench --bench forward_pass

# Regression test (correctness)
cargo test --release --test regression

# Regenerate reference after intentional changes
rm tests/reference/garden_200k.bin && cargo test --release --test regression
```

### Scene data location

The garden scene PLY is at `data/garden/point_cloud.ply` (1.4GB). All benchmarks and tests look there first, then fall back to `~/datasets/3dgs/garden.ply`.

### Optimization targets

The user's stated goal is aggressive CPU optimization of the forward pass using multithreading, ILP, and SIMD. Based on the baseline numbers, the priority order should be:

1. Composite (70% of frame time, currently single-threaded, iterates per-splat then per-pixel within bbox)
2. Sort (19%, currently `sort_unstable_by` with `partial_cmp`)
3. Project (10%, already parallelized with rayon, but the per-splat math could benefit from SIMD)
4. Halfblocks (1.3%, not worth optimizing yet)

### Criterion HTML reports

After running `cargo bench`, Criterion generates HTML reports in `target/criterion/`. Open `target/criterion/report/index.html` in a browser to see performance history with violin plots and regression/improvement detection.

### Repository state

- Branch: `main`
- All changes since the initial commits are in the working tree (not yet committed).
- `LichtFeld-Studio/` is a reference-only C++ submodule. Do not modify it.
