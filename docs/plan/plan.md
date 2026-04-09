# tsplat MVP — Concrete Implementation Plan

## Context

`tsplat` is a Rust tool that renders 3D Gaussian Splat scenes directly in a terminal using CPU rasterization. The premise in `CLAUDE.md` is that a 120×40 terminal with half-blocks is only ~9600 pixels — so naive CPU rasterization that would be laughable at HD becomes real-time at terminal resolution, and the true bottleneck is depth sort, not shading.

The project currently has only `CLAUDE.md` and the `LichtFeld-Studio` submodule (as a C++ reference). No Rust code exists yet. This plan is for the **weekend MVP**: load a `.ply`, project, sort, composite per pixel, draw half-blocks, orbit camera, FPS counter. Nothing more. Tile rasterization, SIMD, higher SH bands, and Kitty/Sixel are explicitly **out of scope**.

Outcome: `cargo run --release -- scene.ply` should open an orbit-able rendering of a 3DGS scene in the terminal.

---

## Crate layout

```
tsplat/
├── Cargo.toml
├── src/
│   ├── main.rs          # CLI, event loop, terminal setup/teardown
│   ├── splat.rs         # Splat struct + .ply loader + downsample
│   ├── camera.rs        # OrbitCamera: view/proj matrices, intrinsics, controls
│   ├── rasterize.rs     # project(), per-pixel front-to-back composite
│   ├── framebuffer.rs   # RGB buffer → half-block ANSI string
│   └── sh.rs            # SH_C0 constant + band-0 → RGB
└── assets/              # (optional later) bundled sample .ply
```

Binary name: `tsplat`.

### `Cargo.toml` dependencies

| crate       | version | why                                                           |
|-------------|---------|---------------------------------------------------------------|
| `glam`      | `0.29`  | vec/mat math, faster than nalgebra for graphics (per CLAUDE) |
| `crossterm` | `0.28`  | raw-mode input, cursor, alt-screen, color — no ratatui        |
| `rayon`     | `1.10`  | parallel per-splat projection (tile rasterization is later)   |
| `ply-rs`    | `0.1`   | .ply parsing; may need to patch for INRIA field names         |
| `clap`      | `4`     | `--max-splats`, positional `<ply>` arg; `derive` feature      |
| `anyhow`    | `1`     | error plumbing in `main`                                      |

Profile: set `[profile.release] lto = "thin"` and `codegen-units = 1` — this is a hot-loop workload and the LTO wins are real.

---

## Data model

```rust
// src/splat.rs
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Splat {
    pub pos:     Vec3,   // world space
    pub scale:   Vec3,   // already exp'd (linear scale)
    pub rot:     Quat,   // already normalized, wxyz per INRIA
    pub color:   Vec3,   // already SH_C0-converted to [0,1] rgb
    pub opacity: f32,    // already sigmoid'd to [0,1]
}
```

**Decode once at load time**, not per frame. The `.ply` stores log-scales and pre-sigmoid opacity and SH DC coefficients — doing that on every frame is wasted work.

---

## Implementation steps

### 0. Fetch test scene (one-off, before any Rust work)
- Target: the trained Mip-NeRF 360 **garden** scene from the INRIA 3DGS pretrained models release (`https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip` — bundles all pretrained scenes as a single ~14GB zip; verify the URL before committing to the download).
- Extract only `garden/point_cloud/iteration_30000/point_cloud.ply` (≈1M splats, ~250MB).
- Store outside the repo (e.g. `~/datasets/3dgs/garden.ply`) — do not commit it, do not add to the submodule. Add `*.ply` to `.gitignore` as a safety net.
- This scene is the reference input for every verification step below.

### 1. Cargo project + skeleton (~30 min)
- `cargo init --bin`, drop in deps, create empty modules.
- `main.rs`: parse CLI (`clap`), print "loaded N splats" stub, exit clean.

### 2. PLY loader (`splat.rs`) (~1–2 hrs)
- Use `ply-rs` to read header + binary body.
- Expected INRIA field names (confirmed in LichtFeld at `LichtFeld-Studio/src/io/formats/ply.cpp:338-370`):
  - `x, y, z`
  - `f_dc_0, f_dc_1, f_dc_2` (SH band 0; higher bands `f_rest_*` are **ignored**)
  - `scale_0, scale_1, scale_2` (**log-space** per `LichtFeld-Studio/src/io/formats/ply.hpp:15`)
  - `rot_0, rot_1, rot_2, rot_3` (**wxyz**, normalized — per `LichtFeld-Studio/src/rendering/rasterizer/gsplat_fwd/ProjectionUT3DGSFused.cu:86`)
  - `opacity`
- **Decoding (at load time)**:
  - `scale = vec3(exp(scale_0), exp(scale_1), exp(scale_2))`
  - `rot = quat(rot_1, rot_2, rot_3, rot_0).normalize()` (note: glam `Quat::from_xyzw` takes xyzw, so feed `(rot_1, rot_2, rot_3, rot_0)`)
  - `color = (0.5 + SH_C0 * f_dc).clamp(0, 1)` with `SH_C0 = 0.28209479177387814`
  - `opacity = sigmoid(opacity_raw)` — INRIA stores it as a logit. (LichtFeld's report claims raw, but that's the LichtFeld convention; vanilla INRIA `.ply` is pre-sigmoid. **Verify with a visual sanity check on the first test scene** — if the whole thing looks ~50% transparent, LichtFeld is right and remove the sigmoid.)
- **Downsample**: `--max-splats N` uniformly strides the vector (`step = len / N`). Default `N = 200_000` when the flag is omitted (safe first-run on any scene; CLAUDE.md confirms uniform subsample is fine for terminal resolution). Accept `--max-splats 0` or `--no-cap` as an escape hatch to load everything.

### 3. Orbit camera (`camera.rs`) (~1 hr)
```rust
pub struct OrbitCamera {
    pub target: Vec3,
    pub yaw: f32, pub pitch: f32, pub radius: f32,
    pub fov_y: f32,          // radians
    pub width: u32, pub height: u32,  // pixel buffer dims (not cells)
    pub znear: f32, pub zfar: f32,
}
```
- `position()` = target + spherical(yaw, pitch, radius).
- `view()` = `Mat4::look_at_rh(pos, target, Vec3::Y)`.
- Intrinsics: `fy = 0.5 * height / tan(fov_y/2)`, `fx = fy` (half-block pixels are ~square because terminal cells are ~2:1 tall and each cell holds 2 vertically stacked pixels).
- Controls (wired in `main.rs` event loop):
  - `h/Left`, `l/Right`: `yaw ± Δ`
  - `k/Up`, `j/Down`: `pitch ± Δ` (clamp to ±89°)
  - `+`/`-`: `radius *= 0.9 / 1.1`
  - `q`/`Esc`: quit
- On terminal resize, recompute `width = cols` and `height = rows * 2`.

### 4. Projection + per-pixel composite (`rasterize.rs`) (~3–4 hrs, the meat)

Standard 3DGS EWA-splatting forward pipeline (LichtFeld uses a fancier Unscented Transform in `ProjectionUT3DGSFused.cu`, but the Jacobian approach from the original 3DGS paper is strictly simpler and well-documented — we use that).

**Per-splat projection** (parallel over splats with `rayon::par_iter`):

```rust
struct Projected {
    screen: Vec2,     // pixel coords
    depth: f32,       // view-space z (positive = in front)
    cov2d_inv: Mat2,  // inverted 2D covariance for Gaussian eval
    bbox: (i32, i32, i32, i32), // inclusive (x0, y0, x1, y1)
    color: Vec3,
    opacity: f32,
}
```

1. **View-transform center**: `p_view = view * vec4(p_world, 1.0)`; skip if `p_view.z > -znear` (RH: in front of camera is `-z`).
2. **Build 3D covariance**: `R = Mat3::from_quat(rot)`, `S = Mat3::from_diagonal(scale)`, `M = R * S`, `Cov3D = M * M.transpose()`. (Matches LichtFeld `Cameras.cuh:1232-1237`.)
3. **View-space covariance**: `Cov3D_view = W * Cov3D * W.transpose()` where `W = Mat3::from_mat4(view)` (rotation part only).
4. **Jacobian at splat center** (pinhole, using view-space `(x, y, z)` with z<0 in RH — take `zc = -p_view.z` for math):
   ```
   J = [ fx/zc     0      -fx*xc/(zc*zc) ]
       [  0      fy/zc   -fy*yc/(zc*zc) ]
   ```
5. **2D covariance**: `Cov2D = J * Cov3D_view * J.transpose()`.
6. **Low-pass dilation**: `Cov2D[(0,0)] += 0.3; Cov2D[(1,1)] += 0.3;` (matches `LichtFeld-Studio/src/rendering/rasterizer/gsplat_fwd/Utils.cuh:172-178`, `eps2d` parameter).
7. **Invert + radius**: `det = Cov2D.determinant()`; skip if `det <= 0`. Eigenvalues `b = 0.5*(a+d)`, `λ₁ = b + sqrt(max(0.01, b²-det))`, `radius = ceil(3.0 * sqrt(λ₁))` (LichtFeld uses `extend = 3.33` at `ProjectionUT3DGSFused.cu:212-240`; 3σ is standard). Store `cov2d_inv = Cov2D.inverse()`.
8. **Screen center**: project `p_view` with perspective → NDC → pixel `(sx, sy)`. Compute `bbox = (sx±radius, sy±radius)` clipped to framebuffer.
9. Skip if bbox is empty (fully off-screen).

**Depth sort**: `projected.sort_unstable_by(|a, b| a.depth.partial_cmp(&b.depth).unwrap())`. Front-to-back (smaller view-space `zc` = closer). CLAUDE explicitly says `sort_unstable_by`, this is the hot loop, radix sort is a later optimization. LichtFeld confirms view-space z is the sort key (`ProjectionUT3DGSFused.cu:276`).

**Per-pixel composite**: for each projected splat in front-to-back order, for each pixel in its bbox:
```rust
let d = vec2(px - screen.x, py - screen.y);
let power = -0.5 * d.dot(cov2d_inv * d);
if power > 0.0 { continue; } // outside ellipse
let alpha = (opacity * power.exp()).min(0.999);
if alpha < 1.0/255.0 { continue; }   // LichtFeld ALPHA_THRESHOLD, Common.h:94
let t = 1.0 - fb[px, py].accum_alpha;
fb[px, py].rgb    += t * alpha * color;
fb[px, py].accum  += t * alpha;
// (optional: early-out per pixel when accum > 0.999)
```
Formula source: standard 3DGS `alpha = opacity * exp(-0.5 * d^T · Σ⁻¹ · d)`. LichtFeld's `RasterizeToPixelsFromWorld3DGSFwd.cu:273-275` uses a ray-splat variant which we deliberately don't need for a plain projection rasterizer.

Framebuffer is `Vec<(Vec3, f32)>` sized `width * height`. Zero it at the start of each frame.

**Parallelism note**: Projection is embarrassingly parallel (rayon over splats). Per-pixel composite is **not** parallel-safe over splats (front-to-back dependency). Keep composite single-threaded for MVP. Ship first, profile second.

### 5. Half-block framebuffer output (`framebuffer.rs`) (~1 hr)
- Pixel buffer is `width × height` where `height = 2 * terminal_rows` (each row holds two vertically stacked pixels via the `▀` UPPER HALF BLOCK U+2580).
- For each cell `(col, row)`:
  - `top = fb[col, 2*row]`, `bot = fb[col, 2*row + 1]`
  - Emit `\x1b[38;2;{r_top};{g_top};{b_top}m\x1b[48;2;{r_bot};{g_bot};{b_bot}m▀`
- **Build one `String` for the whole frame**, then a single `stdout.write_all` + `flush` — CLAUDE explicitly flags per-cell flushing as the real latency killer.
- Reset SGR (`\x1b[0m`) at end of each row + end of frame.
- Position cursor with `\x1b[H` (home) at the start of each frame rather than clearing — clearing causes flicker.
- RLE is **not** MVP (~100KB/frame at 120×40 is fine per CLAUDE).

### 6. Event loop + FPS (`main.rs`) (~1 hr)
- Enter alt screen + raw mode + hide cursor with `crossterm::terminal` + `execute!`.
- Register `ctrlc`-style teardown (or use a `Drop` guard struct) so we always restore the terminal on panic/quit.
- Main loop:
  1. Poll `crossterm::event::poll(Duration::from_millis(0))` — non-blocking.
  2. Drain events, update camera.
  3. Detect resize → reallocate framebuffer.
  4. `clear_framebuffer()`.
  5. `project_splats(&splats, &camera)` (rayon).
  6. `sort_by_depth()`.
  7. `composite()`.
  8. `render_halfblocks()` → single write to stdout.
  9. Compute FPS from `Instant::now()` delta; draw as overlay in top-right corner of the already-built frame string (just write over the relevant cells before flushing).

---

## Critical files to be modified/created

All new:
- `/home/darshan/Projects/tsplat/Cargo.toml`
- `/home/darshan/Projects/tsplat/src/main.rs`
- `/home/darshan/Projects/tsplat/src/splat.rs`
- `/home/darshan/Projects/tsplat/src/camera.rs`
- `/home/darshan/Projects/tsplat/src/rasterize.rs`
- `/home/darshan/Projects/tsplat/src/framebuffer.rs`
- `/home/darshan/Projects/tsplat/src/sh.rs`

Reference-only (do not modify): `LichtFeld-Studio/src/io/formats/ply.cpp`, `LichtFeld-Studio/src/rendering/rasterizer/gsplat_fwd/Cameras.cuh`, `LichtFeld-Studio/src/rendering/rasterizer/gsplat_fwd/Utils.cuh`, `LichtFeld-Studio/src/rendering/rasterizer/gsplat_fwd/SphericalHarmonicsCUDA.cu`.

---

## Gotchas (explicit, all in one place)

1. **Opacity convention**: INRIA `.ply` stores pre-sigmoid logit; LichtFeld's loader treats it as raw. Apply `sigmoid`, then visually verify on first load. Toggle if the scene looks hazy.
2. **Quaternion order**: INRIA `.ply` uses `rot_0..3 = wxyz`; `glam::Quat::from_xyzw` takes `xyzw`. Don't get this wrong — it produces silently-rotated splats.
3. **Log-space scales**: `scale = exp(scale_i)` at load time, not per frame.
4. **Handedness**: glam `look_at_rh` gives view-space with camera looking down `-z`. The Jacobian above assumes we take `zc = -p_view.z` (positive in front). Easy to flip a sign and get inverted depth.
5. **Single stdout flush per frame**: one `String`, one `write_all`, no per-cell writes.
6. **Sort stability**: `sort_unstable_by` is intentional — this is the hot loop.
7. **Terminal cell aspect**: half-block pixels are approximately square because a cell is ~2:1 tall and holds 2 pixels vertically. Treat `fx == fy`. If it looks stretched, add a per-terminal aspect tweak later.
8. **Restore terminal on panic**: wrap raw-mode setup in a Drop guard, otherwise a crash leaves the user's shell broken.

---

## Verification

Reference scene for all steps: `~/datasets/3dgs/garden.ply` (fetched in step 0).

1. **Build**: `cargo build --release` — clean build, no warnings about unused mut or dead code.
2. **Load smoke test**: `cargo run --release -- ~/datasets/3dgs/garden.ply --max-splats 5000` prints the loaded count and exits cleanly (use a `--dump-stats` flag that skips the render loop).
3. **Single-frame visual**: temporarily dump the framebuffer to a PPM (`P6` binary) and open it — ground truth without fighting terminal escape codes. Garden is a good choice here: well-lit, distinctive silhouette (the table + potted plant), easy to recognize if projection is right. Remove once the terminal path works.
4. **Interactive**: `cargo run --release -- ~/datasets/3dgs/garden.ply` — terminal enters alt-screen, garden is visible, `hjkl` orbits, `+/-` zooms, `q` quits and restores the terminal cleanly. FPS counter visible top-right.
5. **Stress**: default cap (`200_000`) should feel interactive (≥15 FPS) at 120×40 on a modern laptop. Then `--max-splats 0` (full ~1M) to find the floor. If below 5 FPS at full res, profile before adding tiles — per CLAUDE, the sort is the suspected bottleneck.
6. **Edge cases**: resize the terminal mid-run (no crash), very small terminal (20×10), point camera away from scene (empty frame, no NaNs).
7. **Reference cross-check**: if LichtFeld-Studio's viewer is compiled locally, load garden there too and eyeball that colors and rough silhouette match. Catches sign/order bugs early.

---

## Out of scope for this plan (explicitly deferred)

Tile rasterization, SIMD (`wide` / `std::simd`), SH bands > 0, Kitty/Sixel backends, RLE ANSI compression, radix-sort depth, per-pixel parallel composite, bundled sample scene, `cargo install` distribution story, asciinema recording. Each is a nice post-MVP follow-up but lengthens the weekend.
