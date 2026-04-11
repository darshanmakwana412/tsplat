# tsplat

tsplat is a terminal-based 3D Gaussian Splatting renderer written in Rust. It rasterizes on the CPU into an RGB framebuffer, then draws to the terminal either with **Unicode half-block** characters (24-bit truecolor SGR) or, when available, the **Kitty graphics protocol** for higher per-cell resolution. At ~120×80 pixels (half-block) the cost is dominated by **depth ordering and compositing**, so the forward pass has been heavily optimized: Jacobian projection with ILP-friendly math, **FlashGS-style opacity-aware 2D bboxes**, **16-bit radix depth sort**, optional **parallel float sort** for very large splat counts, **2D tile binning**, and **parallel per-tile compositing** with a fast approximate `exp` on the Gaussian falloff.

## Build and run

```sh
# Build (release is required for interactive FPS)
cargo build --release

# Run against an INRIA 3DGS `.ply` scene
cargo run --release -- path/to/scene.ply

# Non-interactive smoke test: loads, prints count, exits
cargo run --release -- path/to/scene.ply --dump-stats

# Override the default 200k splat cap (0 or --no-cap = load everything)
cargo run --release -- path/to/scene.ply --max-splats 500000
cargo run --release -- path/to/scene.ply --no-cap

# If the scene looks ~50% transparent, opacity is not a logit on this file:
cargo run --release -- path/to/scene.ply --raw-opacity
```

### Synthetic benchmark binary (no `.ply` required)

```sh
cargo run --release --bin bench_forward
cargo run --release --bin bench_forward -- --threads 1,2,4,8 --splats 200000
cargo run --release --bin bench_forward -- --ply path/to/scene.ply
```

### Tests

```sh
# Regression suite: deterministic synthetic scene, serial vs parallel equivalence
cargo test --release --test regression
```

### Criterion (optional; needs a garden `.ply` on disk)

```sh
cargo bench --bench forward_pass
```

Expects `data/garden/point_cloud.ply` or `~/datasets/3dgs/garden.ply` (see `benches/forward_pass.rs`).

## Display backends

- **`Backend::HalfBlock`** (default): `framebuffer::render_halfblocks` packs two vertical pixels per cell using U+2580 and truecolor fg/bg.
- **`Backend::Kitty`**: probes Kitty graphics support at startup (`display::probe_kitty_support`); sends RGB as Kitty image fragments with a **negative z-index** so the splat layer stays under normal text (FPS strip and HUD). Switchable from the HUD when the terminal reports support.

**`Display`** (`display.rs`) owns backend choice, terminal size, **per-cell pixel size** (query + defaults), **pixel density** multiplier for Kitty/hires framebuffer sizing, text-clear / resize coordination, and routes `render()` to the appropriate encoder.

## Controls

- **Camera:** `WASD` pans in the camera frame (step scales with orbit radius × HUD move speed). `J` / `K` and **Left** / **Right** yaw; `H` / `L` and **Up** / **Down** pitch. `+` / `-` and mouse wheel zoom.
- **`Tab`:** toggles the HUD. While the HUD is open, **arrows** move the highlighted row and adjust values; **PgUp** / **PgDn** move the cursor; **`,`** / **`.`** (and `<` / `>` where emitted) nudge values without arrows.
- **Quit:** `q`, `Esc` (closes HUD first if open), `Ctrl-C`.

Reference test scene is the INRIA 3DGS garden, often at `~/datasets/3dgs/garden.ply`. It is not tracked; `.ply` is in `.gitignore`.

## Crate layout

```
tsplat/
  Cargo.toml                 # glam, crossterm, rayon, clap, anyhow; libc (unix); criterion for benches
  benches/
    forward_pass.rs          # Criterion: project/sort/composite stages (requires .ply)
  docs/
    plan/plan.md             # Original weekend MVP plan (historical scope doc)
    agents/handoff/          # Per-session handoff notes — read the newest before big changes
  LichtFeld-Studio/          # C++ reference submodule — do not modify
  src/
    lib.rs                   # Library surface: camera, framebuffer, rasterize, sh, splat (used by tests/benches)
    main.rs                  # Binary: CLI, terminal guard, event loop, HUD, Display wiring
    splat.rs                 # Splat, PLY loader, uniform downsample, deterministic synthetic `random_scene`
    camera.rs                # OrbitCamera: view, intrinsics, orbit / pan / zoom
    rasterize.rs             # project, depth sort, tile binning, serial + parallel composite
    framebuffer.rs           # Half-block ANSI encoding
    display.rs               # Backend probe, Kitty + half-block output paths
    hud.rs                   # Tab HUD: splat cap, threads, render params, backend, density, camera speeds
    sh.rs                    # SH band-0 to RGB
  src/bin/
    bench_forward.rs         # Thread-scaling forward-pass stats (synthetic or `--ply`)
  tests/
    regression.rs            # Synthetic scene determinism + parallel vs serial framebuffer match
```

## Architecture

The **interactive** render path in `main.rs` only recomputes the framebuffer when something changes (**dirty-frame**): camera input, resize, HUD-driven reload, backend/density changes, etc. Each redraw:

1. **Zero** the `(Vec3, f32)` framebuffer (premultiplied-style accumulation: RGB sum and transmittance-related alpha channel per pixel).
2. **`rasterize::project`** — parallel over splats (rayon; optional dedicated `ThreadPool` from HUD). Builds `Projected` with screen position, depth `zc`, inverse 2D covariance, **axis-aligned bbox** clipped to the image. Bbox uses **opacity-aware extent** (FlashGS-style: effective radius from `2 ln(opacity / α_threshold)` capped by `extend_sigma`, combined with per-axis √Σᵢᵢ), **Jacobian structured** 2D covariance (same sign convention as before: matched flips for screen-up), and tunable **`RenderParams`** (`eps2d`, `alpha_threshold`, `extend_sigma`, `saturation`).
3. **`rasterize::sort_by_depth_parallel`** — for large inputs with a thread pool, may use **parallel unstable sort** on `depth.to_bits()`; otherwise **`sort_by_depth`**: **2-pass 16-bit radix** sort on float bits using reusable **`ScratchBuffers.sort_aux`** (no per-frame alloc once grown).
4. **`rasterize::composite_parallel`** — **`bin_splats`** builds **16×16 tile** index lists (counts + prefix sum + scatter, preserves front-to-back order per tile), then **`composite_tiled`** runs **rayon over tiles**; each tile only iterates splats overlapping that tile, writing through a **disjoint** pixel rect (raw-pointer writes documented in code). Uses a **fast approximate `exp`** for Gaussian alpha. Serial **`composite`** (same math, no tiles) remains for **deterministic regression** reference.
5. **`Display::render`** — half-block string and/or Kitty graphics into the frame buffer string; FPS and **`HudState::render`** append overlays.
6. **One `write_all` + `flush`** per emitted frame.

The event loop uses **`event::poll(Duration::from_millis(33))`** when idle so the process does not busy-spin waiting for keys, while still draining bursts quickly.

### Modules (concise)

- **`splat.rs`:** `Splat { pos, scale, rot, color, opacity }` — all linearized at load. `load_ply`, `downsample_uniform`. **`random_scene(n, seed, bounds)`** + **`Rng`**: deterministic synthetic scenes for tests and `bench_forward`.
- **`camera.rs`:** `OrbitCamera` — `view()`, `intrinsics()` (`fx == fy`), `orbit`, `pan`, `zoom`, `resize`.
- **`rasterize.rs`:** **`RenderParams`**, **`ScratchBuffers`**, **`Projected`**, **`TileBins`**, **`build_thread_pool`**, **`project`**, **`sort_by_depth`**, **`sort_by_depth_parallel`**, **`composite`**, **`composite_parallel`**, **`composite_tiled`**, **`bin_splats`**. Row-level skips in the tiled splat inner loop use the **peak along dx** of the 1D quadratic (not only `dx = 0`) so early-outs stay conservative vs serial.
- **`framebuffer.rs`:** Half-block ANSI: precomputed byte tables, cursor-home prefix; primary SGR escape writer for that backend.
- **`display.rs`:** Backend enum, Kitty probe, cell size, framebuffer dimensions, **`render` / `flush` / `overlay_string`**, resize and text-clear helpers used when switching Kitty/HUD.
- **`hud.rs`:** **`HudState`**: max splats, sigmoid reload, **thread count**, backend, pixel density, FOV, **translate / rotation / zoom** speeds, live **`RenderParams`** sliders. Emits **`HudAction`** for `main` to reload PLY, resize FB, sync FOV, etc.
- **`main.rs`:** **`TerminalGuard`** (raw + alt screen + mouse capture, restore on `Drop`). CLI with clap. Wires camera, HUD, display, thread pool, scratch buffers.

## Reference code

`LichtFeld-Studio/` is a C++ 3DGS implementation included as a git submodule for reference. Useful anchor points when something looks wrong:

- `src/io/formats/ply.cpp` and `src/io/formats/ply.hpp`: INRIA `.ply` field layout, including log-space scales.
- `src/rendering/rasterizer/gsplat_fwd/ProjectionUT3DGSFused.cu`: reference projection (Unscented Transform there; this repo uses the Jacobian / 3DGS-paper style).
- `src/rendering/rasterizer/gsplat_fwd/Cameras.cuh`: 3D covariance construction.
- `src/rendering/rasterizer/gsplat_fwd/Utils.cuh`: `eps2d` low-pass dilation (default `0.3` here, tunable in HUD).
- `src/rendering/rasterizer/gsplat_fwd/Common.h`: `ALPHA_THRESHOLD` (default `1/255` here via `RenderParams`).

Do not modify anything inside `LichtFeld-Studio/`.

## Plans vs reality

`docs/plan/plan.md` describes the **original MVP** and what was explicitly deferred then. The **current codebase implements several former follow-ups** (tile-based composite scheduling, radix depth sort, parallel compositing, Kitty backend, synthetic scenes for CI-style tests). Treat **`docs/agents/handoff/`** as the running lab notebook for **why** recent choices were made; treat **`plan.md`** as historical product intent unless someone updates it deliberately.

## Conventions and gotchas

- **One write per frame.** Keep terminal output batched; avoid per-cell flushes.
- **Decode once at load time.** No `exp`/sigmoid/SH in the per-splat raster hot path beyond what is already baked into `Splat` / `Projected`.
- **INRIA quaternion order is wxyz.** `Quat::from_xyzw(rx, ry, rz, rw)` after reordering from PLY.
- **View space is right-handed** with the camera looking down **`-z`**. Use **`zc = -p_view.z`** for depth and the Jacobian.
- **Terminal cell aspect.** Half-block path assumes roughly square pixels (`fx == fy`). Kitty path uses queried cell pixel size.
- **Restore the terminal on panic.** Always go through **`TerminalGuard`** for raw mode.
- **Sort stability** does not matter; parallel and radix paths use **unstable** ordering. **Parallel vs serial composite** can differ at the **1e-5–1e-2** level on some pixels due to **tile order** and **FP reordering**; regression tests allow a **loose tolerance** on RGB/alpha while enforcing broad equivalence.
- **glam, not nalgebra.**

## Reasonable follow-ups (not implemented)

SIMD (`wide` / `std::simd`), **spherical harmonics beyond band 0**, **Sixel** or other bitmap protocols, **RLE / compressed ANSI**, **bundled sample `.ply`**, **`cargo install` / packaging**, **asciinema** tooling. Any of these would be new scope rather than “finishing the MVP.”
