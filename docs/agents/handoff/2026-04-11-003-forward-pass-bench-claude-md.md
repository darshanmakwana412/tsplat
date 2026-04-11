# Forward pass bench fix, tiled row early-out, CLAUDE.md refresh, Opus follow-up

Date: 2026-04-11
Session focus: Close the loop on a prior Opus 4.6 session that ran out of
context after implementing a large forward-pass refactor. This session
reconciled what was already in the tree, fixed two concrete gaps, refreshed
top-level agent documentation to match the codebase, and recorded context for
future work.

## What was accomplished

- **`src/bin/bench_forward.rs`:** The binary called `composite_parallel` but
  did not import it, so `cargo build --release` failed when building all
  targets. Added the missing import, removed unused `bin_splats` and
  `composite_tiled` imports, and dropped the unused `Timings.bin` field (it was
  never populated).

- **`src/rasterize.rs`:** In `composite_splat_region`, the row-level early-out
  compared `row_base` (Gaussian exponent at `dx = 0` only) against
  `ln(alpha_threshold / opacity)`. Along a scanline the exponent is a concave
  quadratic in `dx`, so its maximum can sit off center. Skipping rows using only
  `row_base` could drop visible contributions and diverge from the serial
  `composite_splat` path. The check now uses the vertex peak
  `row_peak = row_base + 0.5 * row_slope * row_slope / a` with `a` the `(0,0)`
  element of the inverse 2D covariance, and skips only when
  `row_peak < ln(alpha_threshold / opacity)`.

- **`CLAUDE.md`:** Rewritten to describe the current product: half-block and
  Kitty backends, dirty-frame rendering, HUD surface, full raster pipeline
  (`RenderParams`, scratch buffers, radix and optional parallel sort, tile
  binning, parallel composite), benchmarks and regression tests, accurate
  `Cargo.toml` dependencies (no `ply-rs`), and a revised "follow-ups" section
  instead of claiming tile radix and Kitty are still out of scope.

- **This handoff** under `docs/agents/handoff/2026-04-11-003-*.md`.

**Already present before this session** (from earlier work in the same
working tree, summarized here so one commit can land a coherent story):
`src/rasterize.rs` tile bins and `composite_tiled`, FlashGS-style opacity-aware
bbox, `fast_exp`, `sort_by_depth` radix and `sort_by_depth_parallel`,
`src/splat.rs` deterministic `random_scene` and `Rng`, `tests/regression.rs`
synthetic pipeline tests, and related wiring in `src/main.rs`.

## Files changed

| File | Nature of change |
|------|------------------|
| `src/bin/bench_forward.rs` | Import `composite_parallel`; remove dead `bin` timing slot. |
| `src/rasterize.rs` | Correct tiled composite row early-out using peak along `dx`. |
| `CLAUDE.md` | Full refresh to current architecture and features. |
| `src/main.rs` | Prior session: viewer loop, HUD, dirty frames, display (if touched in tree). |
| `src/splat.rs` | Prior session: synthetic scene generator for tests and bench. |
| `tests/regression.rs` | Prior session: synthetic regression vs serial reference. |
| `docs/agents/handoff/2026-04-11-003-forward-pass-bench-claude-md.md` | This note. |

## Key decisions

1. **Row skip uses the analytic peak along `dx`, not `row_base` alone.** Keeps
   the optimization conservative relative to the serial reference and avoids a
   subtle correctness bug on elongated Gaussians.

2. **Single documentation source for agents.** `CLAUDE.md` now states that
   `docs/plan/plan.md` is historical MVP scope while handoffs carry incremental
   rationale. Reduces confusion where the plan still lists items as deferred
   that the code already implements.

3. **Do not commit `opus_chat.jsonl` by default.** Large JSONL export of a
   prior chat; left untracked unless the project explicitly wants it versioned.

## Important context for future sessions

- **Verify builds with** `cargo build --release` **(all targets)** so
  `bench_forward` is included, not only the default binary.

- **Regression:** `cargo test --release --test regression` exercises
  determinism, sanity, ANSI shape, and serial vs `composite_parallel`
  equivalence with a loose per-pixel tolerance (tile order and FP reordering).

- **`composite` and `composite_splat`** may still be marked `allow(dead_code)`
  in places; the regression suite uses the serial `composite` path as the
  reference.

- **Handoff sequence for 2026-04-11:** `001` dirty-frame and Kitty, `002`
  camera and HUD speeds, **`003` this file.**

## Repository state

After this session, run `cargo build --release` and
`cargo test --release --test regression` before merging or tagging.
