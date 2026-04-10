# tsplat HUD control panel

Date: 2026-04-10
Session focus: adding an interactive Heads-Up Display (HUD) overlay to the terminal viewer, allowing runtime parameter tuning without restarting.

## What was accomplished

- Created a new `src/hud.rs` module (~210 lines) implementing a toggle-able control panel rendered as ANSI escape code overlays in the upper-left corner of the terminal.
- Extracted the four compile-time constants in `src/rasterize.rs` (`EPS2D`, `ALPHA_THRESHOLD`, `EXTEND_SIGMA`, `SATURATION`) into a runtime-tunable `RenderParams` struct. Updated `project()` and `composite()` signatures to accept `&RenderParams`.
- Rewired the event loop in `src/main.rs` to support two input modes: normal (camera) and HUD (parameter adjustment). Added a reload path for parameters that require re-reading the PLY file.
- Fixed a camera-reset bug in the reload path: changing splat count or sigmoid no longer resets the camera to the auto-framed initial position.
- Verified `cargo build --release` passes with zero warnings.

## Files changed

| File | Nature of change |
|------|-----------------|
| `src/hud.rs` | New file. `HudState`, `HudItem` enum, `HudAction` enum, key handling, ANSI overlay rendering. |
| `src/rasterize.rs` | Added `RenderParams` struct with `Default` impl. Removed 4 `const` declarations. Changed `project()` and `composite()` to accept `&RenderParams`. |
| `src/main.rs` | Added `mod hud`, `HudState` initialization, restructured key dispatch for HUD vs camera mode, added reload path, added HUD overlay render call. |

Files not changed: `camera.rs`, `framebuffer.rs`, `splat.rs`, `sh.rs`.

## Key decisions

1. **`RenderParams` struct passed by reference, not thread-local.** The struct is 4 floats (16 bytes), trivially `Copy`, and `&RenderParams` is `Send + Sync` so rayon's `par_iter` closure in `project()` captures it without any synchronization. This was chosen over `thread_local!` or global statics because it keeps the API explicit and testable.

2. **Tab toggles HUD, Esc closes HUD or quits.** When the HUD is visible, Esc closes it. When hidden, Esc quits. `q` always quits regardless of HUD state. This avoids modal confusion where the user cannot exit.

3. **Vim keys always control camera.** `h/j/k/l` orbit the camera regardless of whether the HUD is visible. Arrow keys are context-dependent: they adjust HUD parameters when the HUD is open, and orbit the camera when it is closed. This means the user is never locked out of camera control while tuning parameters.

4. **Synchronous reload for splat count and sigmoid changes.** No background thread. At the default 200k cap the reload is imperceptible. At full 5.8M splats it takes about 0.4s, which is acceptable with the "Loading..." indicator. A brief red-on-white "Loading..." label is written to stdout before the reload begins.

5. **Camera position preserved across reloads.** The initial implementation re-framed the camera on every reload (reset target and radius from scene bounds). This was removed after the user reported the bug. Only the initial load auto-frames; subsequent reloads keep the camera exactly where the user left it.

6. **HUD rendering uses the same overlay pattern as the FPS counter.** Cursor-addressed ANSI escape sequences appended to the frame string after `render_halfblocks()`, before the single `write_all` flush. No changes to `framebuffer.rs` were needed.

## Important context for future sessions

### HUD controls

The HUD exposes 10 parameters across 3 groups:

- **Splats**: `max_splats` (multiply/divide by 2, range 1k to 10M), `sigmoid` (toggle on/off). Both trigger a PLY reload.
- **Camera**: `fov_y` (plus/minus 5 degrees, range 10 to 150), `orbit spd` (plus/minus 0.01), `pitch spd` (plus/minus 0.01), `zoom spd` (plus/minus 0.05). All real-time.
- **Render**: `eps2d` (plus/minus 0.05), `alpha thr` (multiply/divide by 2), `sigma ext` (plus/minus 0.5), `saturation` (plus/minus 0.001). All real-time.

### Interactive viewer has not been visually verified in this session

All work was done via `cargo build --release`. The interactive viewer was not run against a real scene in this session. The visual correctness checks from the original plan (upside-down rendering, hazy opacity, quaternion order) remain open from session 001. The HUD overlay positioning, color scheme, and item layout should be verified visually on first run.

### Known issue: zoom factor inversion

The zoom controls were changed from hardcoded `0.9`/`1.1` to `1.0 - zoom_step`/`1.0 + zoom_step`. At the default `zoom_step = 0.1` this produces the same `0.9`/`1.1` values. However, if a user increases `zoom_step` above 1.0 (which the current bounds of [0.05, 0.50] prevent), the zoom-in factor would go negative. The bounds clamp prevents this, but it is worth being aware of if the bounds are ever relaxed.

### Repository state

- Branch: `main`.
- Modified files relative to last commit: `src/main.rs`, `src/rasterize.rs`. New untracked file: `src/hud.rs`. Previously modified files from earlier sessions (`Cargo.toml`, `Cargo.lock`, `src/splat.rs`) remain uncommitted.
- Nothing has been committed in this session or any previous session. All changes since the initial commit are still in the working tree.
- `LichtFeld-Studio/` is a reference-only C++ submodule. Do not modify it.

### CLI surface (unchanged from previous sessions)

```
tsplat <PLY>
    --max-splats <N>   (default 200000; load at most N splats)
    --no-cap           (load everything; equivalent to --max-splats 0)
    --raw-opacity      (skip sigmoid on opacity; use if scene is hazy)
    --dump-stats       (load, print count, exit without rendering)
```

### Runtime controls

```
Tab         Toggle HUD overlay
Esc         Close HUD (if open) or quit (if closed)
q           Quit (always)
Ctrl-C      Quit (always)
h/j/k/l     Orbit camera (always, even with HUD open)
Arrows      Orbit camera (HUD closed) or navigate/adjust HUD (HUD open)
+/-         Zoom in/out (always)
```
