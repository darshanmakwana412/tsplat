# tsplat forward-pass optimization

Date: 2026-04-10
Session focus: aggressive CPU optimization of the Gaussian splatting forward pass using multithreading, ILP, SIMD-friendly code, and algorithmic improvements.

## What was accomplished

Seven commits were pushed to `main`, each tested against the regression suite before landing. The cumulative result is a 2.7x speedup of the full pipeline at 200k splats / 120x80 resolution: from 28 FPS (36ms/frame) to 76 FPS (13ms/frame).

### Commit-by-commit summary

1. **Parallel tiled composite + HUD thread control.** Split the framebuffer into 16-row horizontal tiles processed in parallel via rayon. Added a `composite_parallel()` function alongside the original single-threaded `composite()`. Added a "threads" control to the HUD under a new "Perf" section so the user can adjust thread count at runtime (0 = all cores, default = 4). Composite: 25ms to 14ms.

2. **Branchless integer key sort.** Replaced `sort_unstable_by` with `partial_cmp` by `sort_unstable_by_key` using `f32::to_bits()`. Since all depths are positive, the u32 bit pattern preserves float ordering and eliminates NaN-check branches.

3. **ILP decomposition + fast_exp + target-cpu=native.** Decomposed the 2D Gaussian evaluation into row-constant terms (dy^2, cross product) hoisted outside the inner pixel loop. Replaced libm `exp()` with the Schraudolph bit-trick approximation (~1-2% error, imperceptible at terminal resolution). Added `.cargo/config.toml` with `target-cpu=native` to enable AVX2/FMA autovectorization. Composite: 14ms to 10ms.

4. **Cached thread pool + fast fb clear + LUT halfblocks.** Cached the rayon thread pool across frames (rebuild only when HUD thread count changes). Replaced the per-pixel framebuffer clear loop with `write_bytes` memset. Rewrote `render_halfblocks` to push raw bytes with a pre-computed decimal lookup table instead of per-cell `write!` formatting. Halfblocks: 540us to 140us.

5. **Sparse Jacobian projection + shared thread pool.** Exploited the Jacobian's sparse structure to compute the 2D covariance with ~20 scalar multiplies instead of two full 3x3 matrix multiplies (~54 multiplies). Inlined the 2x2 inverse. Pre-computed `1/zc` once. Routed projection through the same cached thread pool so the HUD thread count controls both stages.

6. **Row-level early-out in composite.** Pre-compute `ln(alpha_threshold / opacity)` per splat. If the dy^2 term alone falls below this cutoff, skip the entire inner pixel loop for that row. Composite: 8ms to 6ms.

7. **ScratchBuffers + stack histograms + single-pass histogram.** Added a `ScratchBuffers` struct to reuse the radix sort auxiliary buffer across frames. Replaced heap-allocated histogram Vecs with stack-allocated `[u32; 65536]` arrays. Compute both lo and hi histograms in a single pass over the keys. Sort: 3.6ms to 3.4ms.

Plus a standalone commit replacing the comparison sort with a 2-pass 16-bit radix sort, which was the single biggest sort improvement (7ms to 3.6ms).

## Files changed

| File | Nature of change |
|------|-----------------|
| `src/rasterize.rs` | New `composite_parallel`, `build_thread_pool`, `ScratchBuffers`, `fast_exp`, `composite_splat`. Rewrote `project` with sparse Jacobian, thread pool param. Rewrote `sort_by_depth` as 2-pass radix sort with scratch buffers. |
| `src/main.rs` | Uses `composite_parallel` with cached thread pool. Fast memset fb clear. Passes `ScratchBuffers` to sort. |
| `src/hud.rs` | Added `NumThreads` HUD item under "Perf" section. `num_threads` field on `HudState` (default 4). |
| `src/framebuffer.rs` | Rewrote `render_halfblocks` with raw byte buffer and `BYTE_STRINGS` LUT. |
| `src/bin/bench_forward.rs` | Updated to use `composite_parallel`, `build_thread_pool`, `ScratchBuffers`. |
| `tests/regression.rs` | Updated to pass `pool` and `ScratchBuffers` to changed APIs. |
| `benches/forward_pass.rs` | Updated to pass `pool` and `ScratchBuffers` to changed APIs. |
| `.cargo/config.toml` | New file. Sets `rustflags = ["-C", "target-cpu=native"]` for x86_64. |
| `Cargo.toml` | No changes (rayon was already a dependency). |

Files not changed: `splat.rs`, `camera.rs`, `sh.rs`, `lib.rs`.

## Key decisions

1. **4 threads by default, not all cores.** The user explicitly asked for 3-4 threads. The HUD defaults to 4 and allows runtime adjustment down to 0 (all cores). This avoids saturating the machine during interactive use.

2. **Horizontal tile parallelism, not per-splat parallelism for composite.** Each tile owns a disjoint framebuffer slice, so there are no data races and no atomic operations. The alternative (per-splat parallel with atomics) would have heavy contention on overlapping pixels.

3. **fast_exp (Schraudolph) instead of libm exp.** The 1-2% relative error is invisible at terminal resolution (8-bit color). This is the single biggest win in the inner loop since `exp()` was the dominant instruction.

4. **2-pass 16-bit radix sort instead of 4-pass 8-bit.** Two passes halve the number of data traversals. The 65536-entry histograms (256KB each on the stack) fit comfortably in L2 cache. At 200k elements this is consistently 1.9x faster than Rust's `sort_unstable_by_key`.

5. **Stack-allocated histograms.** The radix sort histograms are `[u32; 65536]` on the stack rather than heap-allocated Vecs. This avoids allocator overhead and is cache-friendlier since the stack is always warm.

6. **Kept single-threaded `composite` for regression tests.** The regression tests use the original single-threaded `composite` to ensure deterministic output for pixel-by-pixel comparison. The parallel version produces identical output (verified explicitly) but its test determinism depends on tile boundaries.

7. **`.cargo/config.toml` for target-cpu=native.** This enables AVX2/FMA autovectorization without requiring nightly Rust or explicit SIMD intrinsics. The tradeoff is that binaries are not portable across CPU generations, which is fine for a developer tool.

8. **Row-level early-out uses ln() per splat, not per row.** The `ln(alpha_threshold / opacity)` cutoff is computed once per splat (outside the row loop). This is a single transcendental per splat rather than a branch-heavy check per pixel.

## Baseline vs final performance numbers

Measured on the development machine with 200k splats at 120x80 resolution, 4 threads:

| Stage | Before (ms) | After (ms) | Speedup |
|-------|------------|------------|---------|
| project | 3.6 | 3.9 | ~same (thread pool overhead vs sparse math, wash) |
| sort | 6.7 | 3.4 | 2.0x |
| composite | 25.2 | 5.7 | 4.4x |
| halfblocks | 0.5 | 0.15 | 3.3x |
| **Total** | **36.0** | **13.2** | **2.7x** |
| **FPS** | **28** | **76** | **2.7x** |

## Important context for future sessions

### API changes

The following function signatures changed and callers must be updated:

- `project()` now takes an additional `pool: &Option<rayon::ThreadPool>` parameter.
- `sort_by_depth()` now takes `scratch: &mut ScratchBuffers`.
- `composite_parallel()` takes `pool: &Option<rayon::ThreadPool>` instead of `num_threads: usize`.
- The original `composite()` still exists for tests but is unused in the main binary.

### Where the time goes now

Composite is still the largest stage at 43% but it is 4.4x faster than baseline. Project and sort are now roughly equal at ~29% and ~26%. Further gains would come from:

1. **Tighter bboxes.** The current 3-sigma bbox is conservative. A 2.5-sigma bbox would reduce pixel work by ~30% with minimal visual impact at terminal resolution.
2. **Tile-based splat binning.** Instead of iterating all splats per tile, pre-bin splats into tiles so each tile only touches its own splats. This would reduce the O(splats * tiles) scan to O(splats + tile_work).
3. **SIMD composite inner loop.** The current code relies on autovectorization. Explicit AVX2 intrinsics for the 4-pixel-wide Gaussian evaluation and alpha blend could squeeze more out.
4. **Parallel radix sort.** The histogram computation could be parallelized across chunks, then merged. At 200k elements the serial histogram is already fast (~1ms) so the win may be modest.

### Regression test reference

The reference file `tests/reference/garden_200k.bin` was regenerated after the fast_exp change. It encodes the approximate-exp output. If someone reverts to libm exp, the reference must be regenerated:

```
rm tests/reference/garden_200k.bin && cargo test --release --test regression
```

### Running the benchmarks

Same as before:

```
cargo run --release --bin bench_forward
cargo run --release --bin bench_forward -- --frames 200 --max-splats 500000
cargo bench --bench forward_pass
cargo test --release --test regression
```

### Repository state

- Branch: `main`
- All changes committed and pushed.
- Seven new commits on top of the bench/test framework commit (aa2e9cd).
