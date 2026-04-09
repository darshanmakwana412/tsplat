# tsplat MVP initial build

Date: 2026-04-10
Session focus: scaffolding the tsplat crate and implementing the weekend MVP described in `docs/plan/plan.md`.

## What was accomplished

Scaffolded the Rust binary crate and implemented every module listed in the plan. `cargo build --release` is clean with zero warnings. The binary compiles, accepts CLI flags, and is ready for a first run against a real `.ply` scene. No test scene has been fetched yet, so the MVP has not yet been visually verified.

New files:

- `Cargo.toml`: dependencies pinned per plan (`glam 0.29`, `crossterm 0.28`, `rayon 1.10`, `ply-rs 0.1`, `clap 4`, `anyhow 1`). Release profile uses `lto = "thin"` and `codegen-units = 1`.
- `src/sh.rs`: `SH_C0` constant and `sh_band0_to_rgb` helper.
- `src/splat.rs`: `Splat` struct plus `load_ply` and `downsample_uniform`. All per-splat decoding (exp of log scales, wxyz to xyzw quaternion reorder, SH band-0 to RGB, sigmoid of opacity logit) happens once at load time, never per frame.
- `src/camera.rs`: `OrbitCamera` with `view()`, `intrinsics()` returning `(fx, fy, cx, cy)`, and `orbit`/`zoom`/`resize` controls.
- `src/rasterize.rs`: `project` (rayon parallel, Jacobian-based EWA projection, 3 sigma bbox, 0.3 eps2d dilation, znear/zfar culling), `sort_by_depth` using `sort_unstable_by`, and `composite` doing front-to-back alpha accumulation with a 0.999 saturation early-out.
- `src/framebuffer.rs`: `render_halfblocks` builds one `String` for the whole frame using the U+2580 upper half block with truecolor fg/bg SGR sequences. Uses cursor home (`\x1b[H`) at frame start, never clears.
- `src/main.rs`: clap CLI, `TerminalGuard` RAII type that enters alt screen plus raw mode and restores on `Drop` (including on panic), non-blocking event loop with `hjkl`, arrows, `+`/`-`, `q`/`Esc`/`Ctrl-C`, resize handling that reallocates the framebuffer, FPS overlay written to the top right corner of the already-built frame string, single `write_all` per frame.
- `.gitignore`: appended `*.ply`.

CLI surface:

```
tsplat <PLY>
    --max-splats <N>   (default 200000)
    --no-cap           (load everything)
    --raw-opacity      (skip sigmoid; use if scene is hazy)
    --dump-stats       (load, print count, exit without rendering)
```

## Key decisions

1. **Y axis handling in the projection**: the plan's Jacobian assumes a convention where world-up maps to screen-up. I made this explicit by using `v = -fy * yv / zc + cy` for the v coordinate and flipping the sign of the second row of the Jacobian to match (`[0, -fy/zc, -fy*yv/zc^2]`). The resulting `Cov2D = J Cov3D_view J^T` is still a valid symmetric positive semidefinite matrix, and the two sign flips cancel for every covariance term. This is a best guess at the correct orientation. If the garden scene renders upside down, remove both sign flips together (the `-fy` on row 2 of `J` and the `-` in front of the v expression in `project`). Do not flip only one, that breaks the covariance.

2. **Depth convention**: RH view space with camera looking down `-z`. Work in `zc = -p_view.z` (positive in front). Culling uses `p_view.z > -znear || p_view.z < -zfar`. Depth sort key is `zc`, smaller first (front to back).

3. **`zfar` made load bearing**: the plan's `OrbitCamera` struct has `zfar`, but nothing in the rest of the plan uses it. I wired it into the far plane cull in `project` rather than suppress the dead code warning with an attribute. It is cheap and correct.

4. **No `unsafe` in the composite loop**: I briefly used `get_unchecked_mut` in the per pixel hot loop, then reverted to safe indexing per plan guidance of "ship first, profile second". The bbox is pre-clipped to framebuffer bounds so the bounds check is predictable.

5. **Frame emission strategy**: single `String` assembled with `write!`, one `stdout().lock().write_all(...)` plus `flush` per frame. Cursor home at start, SGR reset at end of each row and end of frame. Never clears the screen. FPS overlay is appended to the frame string after the main content so it rides the same flush.

6. **Terminal restoration**: `TerminalGuard` holds no state and its `Drop` impl calls `cursor::Show`, `LeaveAlternateScreen`, and `disable_raw_mode`. This guarantees restoration on panic as well as normal exit. The guard is bound to a `_guard` local in `main`.

7. **Scene auto framing**: on startup, `main.rs` computes a world space bounding box from the loaded splats and sets `camera.target` to the box center and `camera.radius` to `2.5 * half_diagonal`. This avoids a first-run experience where the default `radius = 5.0` puts the camera inside the cloud.

8. **Explicitly deferred and not implemented**: tile rasterization, SIMD, higher SH bands, Kitty or Sixel, RLE ANSI compression, radix sort, parallel composite, bundled sample scene, PPM dump path. All listed as out of scope in the plan.

## Important context for future sessions

### Test scene is not yet fetched

Step 0 of the plan, the INRIA 3DGS pretrained garden scene at `~/datasets/3dgs/garden.ply`, has not been downloaded. The `~/datasets/3dgs/` directory does not exist on this machine as of this session. The MVP has therefore never been run against real data, so none of the visual verification steps in the plan have been performed. First action for a future session should be fetching that file and running:

```
cargo run --release -- ~/datasets/3dgs/garden.ply --dump-stats
cargo run --release -- ~/datasets/3dgs/garden.ply
```

### Things to look for on first visual run

The plan and the code both flag these as the likely first bugs to debug after getting something on screen:

1. Scene is vertically mirrored: caused by the Y sign convention above. Fix by removing both sign flips together.
2. Scene is ~50% transparent: the opacity sigmoid is wrong for this file. Run with `--raw-opacity`.
3. Scene silhouette is rotated in a weird way: the quaternion component order is wrong. INRIA stores `rot_0, rot_1, rot_2, rot_3` as `w, x, y, z`; I pass `Quat::from_xyzw(rx, ry, rz, rw)`, which is correct per the plan, but this is the class of bug to check first.
4. Performance floor: plan suggests 200k splats should feel interactive at 120x40 on a modern laptop. If not, sort is the first suspect.

### Repository state

- Branch: `main`.
- The working tree has several untracked and uncommitted items from this session plus pre-existing state from before it: new `src/`, new `Cargo.toml`, new `Cargo.lock`, modified `.gitignore`, and pre-existing `.gitmodules` plus the `LichtFeld-Studio` submodule plus `CLAUDE.md` modifications. Nothing has been committed in this session.
- `LichtFeld-Studio/` is a reference-only C++ submodule. Do not modify it. The plan calls out specific files inside it (for example `src/io/formats/ply.cpp`, `src/rendering/rasterizer/gsplat_fwd/Cameras.cuh`) as ground truth for the INRIA field layout and the reference covariance math.
- `docs/plan/plan.md` is the source of truth for scope. Anything past the explicit "out of scope" list in that file is a follow up, not an MVP gap.

### Non obvious implementation notes

- `glam::Mat2` and `glam::Mat3` are column major with public `x_axis`, `y_axis`, `z_axis` fields. The Jacobian is assembled via `Mat3::from_cols` with a zero third row so the ordinary 3x3 multiply gives the right 2D covariance in the top left 2x2 block.
- `ply-rs` 0.1.3 uses `linked_hash_map::LinkedHashMap` for `DefaultElement`. `get(key)` works with `&str`. All INRIA splat properties come back as `Property::Float`.
- The composite loop zeroes the framebuffer at the start of each frame via `for c in fb.iter_mut() { *c = (Vec3::ZERO, 0.0); }`. If that shows up in profiles, a `fill` call or a SIMD memset would be the obvious replacement.
- The FPS overlay uses a direct `\x1b[1;{col}H` cursor jump rather than a second write. It lives in the same frame string, so there is still exactly one `write_all` per frame.
