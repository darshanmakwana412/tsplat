# tsplat

tsplat is a terminal-based 3D Gaussian Splatting renderer written in Rust. It rasterizes on the CPU and draws directly into the terminal using half-block characters with 24-bit truecolor. The premise is that a 120x40 terminal with half blocks is only ~9600 pixels, so CPU rasterization that would be laughable at HD becomes real-time at terminal resolution, and the true bottleneck is the depth sort, not shading.

## Build and run

```sh
# Build (release is required for interactive FPS)
cargo build --release

# Run against an INRIA 3DGS .ply scene
cargo run --release -- path/to/scene.ply

# Non-interactive smoke test: loads, prints count, exits
cargo run --release -- path/to/scene.ply --dump-stats

# Override the default 200k splat cap (0 or --no-cap = load everything)
cargo run --release -- path/to/scene.ply --max-splats 500000
cargo run --release -- path/to/scene.ply --no-cap

# If the scene looks ~50% transparent, opacity is not a logit on this file:
cargo run --release -- path/to/scene.ply --raw-opacity
```

Controls inside the viewer: `WASD` pans in the camera frame; `J`/`K` and Left/Right yaw; `H`/`L` and Up/Down pitch; `+`/`-` and mouse wheel zoom. `Tab` toggles the HUD (with HUD open, arrows move the row and adjust values). `q`/`Esc`/`Ctrl-C` to quit.

Reference test scene is the INRIA 3DGS garden scene, expected at `~/datasets/3dgs/garden.ply`. It is not tracked in the repo and `.ply` is in `.gitignore`.

## Crate layout

```
tsplat/
  Cargo.toml                 # glam, crossterm, rayon, ply-rs, clap, anyhow
  docs/
    plan/plan.md             # MVP implementation plan (source of truth for scope)
    agents/handoff/          # Per-session handoff notes for future agents
  LichtFeld-Studio/          # C++ reference submodule, reference only, do not modify
  src/
    main.rs                  # CLI, event loop, terminal guard, FPS overlay
    splat.rs                 # Splat struct, PLY loader, uniform downsample
    camera.rs                # OrbitCamera: view matrix, intrinsics, controls
    rasterize.rs             # project(), sort_by_depth(), composite()
    framebuffer.rs           # RGB buffer to half-block ANSI string
    sh.rs                    # SH_C0 constant and band-0 to RGB
```

## Architecture

The render pipeline is one pass per frame, wired together in `main.rs`:

1. `rasterize::project(&splats, &camera)` projects every splat in parallel with rayon. For each splat it builds the 3D covariance from `R * S * S^T * R^T`, rotates it into view space, applies the Jacobian of the pinhole projection to get the 2D image-plane covariance, adds a `0.3` low-pass dilation to the diagonal, computes a 3 sigma axis-aligned bbox, and clips the bbox to the framebuffer. View space is right-handed with the camera looking down `-z`; everything downstream uses `zc = -p_view.z` so depth is positive in front.
2. `rasterize::sort_by_depth(&mut projected)` sorts by `zc` ascending (front-to-back) using `sort_unstable_by`.
3. `rasterize::composite(&projected, &mut fb, w, h)` walks each projected splat's bbox in front-to-back order and does per-pixel alpha accumulation into a `(Vec3, f32)` framebuffer, with an early-out at `accum_alpha >= 0.999`.
4. `framebuffer::render_halfblocks(&fb, w, h, &mut out)` converts the RGB framebuffer into a single `String` of half-block escapes. Each terminal cell holds two stacked pixels using the U+2580 character with top pixel as fg and bottom pixel as bg.
5. `main` appends the FPS overlay to the same string and emits the whole frame with one `stdout().lock().write_all(...)` plus `flush`. Cursor is homed with `\x1b[H` at the top of the frame; the screen is never cleared.

### Modules

- `splat.rs`: `Splat { pos, scale, rot, color, opacity }`. `load_ply` does all decoding at load time: `exp()` on log-space scales, wxyz to xyzw quaternion reorder for `glam::Quat::from_xyzw`, SH band-0 to RGB via `sh::sh_band0_to_rgb`, sigmoid on the raw opacity logit. Nothing in this struct needs per-frame work. `downsample_uniform` is a stride-based subsample.
- `camera.rs`: `OrbitCamera` with `target`, `yaw`, `pitch`, `radius`, `fov_y`, pixel `width`/`height`, `znear`, `zfar`. `view()` is `Mat4::look_at_rh`. `intrinsics()` returns `(fx, fy, cx, cy)` with `fx == fy` because half-block pixels are approximately square.
- `rasterize.rs`: projection plus composite. The Jacobian has a sign flip on its second row (and a matching `-` in the v projection formula) so that world-up maps to screen-up on the y-down framebuffer. If the scene ever renders upside-down, remove both sign flips together; never just one, that breaks the covariance.
- `framebuffer.rs`: the only place that writes SGR escapes. One `String`, truecolor fg plus bg in a single SGR sequence per cell, explicit `\r\n` plus SGR reset between rows in raw mode.
- `main.rs`: `TerminalGuard` is an RAII type that enters alt-screen plus raw mode on construction and restores on `Drop`, so a panic cannot leave the shell broken. The event loop polls with `Duration::from_millis(0)` so rendering is never blocked on input. On startup the camera is auto-framed from the scene bounding box.

## Reference code

`LichtFeld-Studio/` is a C++ 3DGS implementation included as a git submodule for reference. Useful anchor points when something looks wrong:

- `src/io/formats/ply.cpp` and `src/io/formats/ply.hpp`: INRIA .ply field layout, including the fact that `scale_*` is log-space.
- `src/rendering/rasterizer/gsplat_fwd/ProjectionUT3DGSFused.cu`: reference projection. Note that LichtFeld uses an Unscented Transform rather than the Jacobian approach we use here. The plan intentionally picks the simpler Jacobian form from the original 3DGS paper.
- `src/rendering/rasterizer/gsplat_fwd/Cameras.cuh`: 3D covariance construction.
- `src/rendering/rasterizer/gsplat_fwd/Utils.cuh`: `eps2d = 0.3` low-pass dilation constant.
- `src/rendering/rasterizer/gsplat_fwd/Common.h`: `ALPHA_THRESHOLD = 1/255`.

Do not modify anything inside `LichtFeld-Studio/`.

## Source of truth for scope

`docs/plan/plan.md` is the authoritative implementation plan for the weekend MVP. Everything past that file's explicit "out of scope" list (tile rasterization, SIMD, higher SH bands, Kitty or Sixel, RLE ANSI compression, radix sort depth, parallel composite, bundled sample scene) is a follow-up, not an MVP gap.

`docs/agents/handoff/` contains per-session handoff notes. Read the most recent one before picking up work.

## Conventions and gotchas

- **One write per frame.** Flushing stdout per cell is the real latency killer. Always append to the frame string and emit the whole thing in one `write_all`.
- **Decode once at load time.** `exp` on scales, sigmoid on opacity, SH to RGB, quaternion normalize. None of these belong in the per-frame path.
- **INRIA quaternion order is wxyz.** `rot_0` is w and `rot_1..3` are xyz. `glam::Quat::from_xyzw` wants xyzw, so the call is `Quat::from_xyzw(rx, ry, rz, rw)`. Getting this wrong produces silently-rotated splats.
- **View space is right-handed.** Points in front of the camera have `p_view.z < 0`. Use `zc = -p_view.z` for the Jacobian and the depth sort key.
- **Terminal cell aspect.** Half-block pixels are roughly square because a cell is ~2:1 tall and holds 2 pixels vertically. Treat `fx == fy`.
- **Restore the terminal on panic.** Raw-mode setup is behind a `Drop` guard in `main.rs`. Do not bypass it.
- **Sort stability does not matter here.** Use `sort_unstable_by`. This is the hot loop.
- **glam, not nalgebra.** Graphics math is meaningfully faster in glam.

## Out of scope right now

Tile rasterization, SIMD (`wide` or nightly `std::simd`), SH bands beyond band 0, Kitty or Sixel graphics backends, RLE ANSI compression, radix-sort depth, per-pixel parallel composite, bundled sample scene, `cargo install` distribution, asciinema recording. Every one of these is a reasonable follow-up, but none of them belong in the MVP.
