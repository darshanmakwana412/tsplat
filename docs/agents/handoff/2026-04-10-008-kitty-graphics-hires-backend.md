# tsplat hi-res rendering via Kitty graphics protocol

Date: 2026-04-10
Session focus: Add high-resolution rendering support using the Kitty graphics protocol, with auto-detection, HUD toggle, and pixel density control.

## What was accomplished

- Created `src/display.rs` — the new rendering backend abstraction containing:
  - `Backend` enum (`HalfBlock` | `Kitty`) with a name accessor
  - `CellSize` struct (pixel dimensions of a single terminal cell)
  - `Display` struct that owns the frame output buffer, backend state, cell size, and pixel density
  - Auto-detection: `probe_kitty_support()` sends a 1×1 query image (`a=q`) plus DA1 fence and reads back the response to determine if the terminal supports the Kitty graphics protocol
  - Cell pixel size discovery: tries CSI 16t first, falls back to `crossterm::terminal::window_size()` ioctl, defaults to 1×2 (half-block assumption)
  - `Display::framebuffer_size()` computes pixel dimensions: HalfBlock uses cols×(rows*2), Kitty uses (cols*cell_w*density)×(rows*cell_h*density)
  - `Display::render()` dispatches to the appropriate backend
  - `Display::overlay_string()` returns `&mut String` for cursor-addressed ANSI overlays (FPS, HUD) that work on top of both backends
  - Kitty backend: converts the `(Vec3, f32)` framebuffer to raw RGB bytes, base64-encodes inline (no external crate), transmits via chunked APC escape sequences (4096-byte chunks, `m=1`/`m=0` protocol), with `c=cols,r=rows` for terminal-native scaling, `q=2` to suppress responses, and `i=id,p=1` for flicker-free placement replacement
  - `Display::kitty_cleanup()` deletes placed images on exit or backend switch

- Updated `src/hud.rs`:
  - Added new "Display" group with two items: `backend` (toggle) and `px density` (slider)
  - `HudAction::BackendChanged` and `HudAction::DensityChanged` for the new controls
  - Backend toggle only allows switching to Kitty if it was detected at startup
  - Pixel density steps by 5% between 10% and 100%, shows "n/a" when in HalfBlock mode
  - Added "Display Info" read-only section showing resolution, cell pixel size, terminal dimensions, and detected backend
  - Introduced `DisplayInfo` snapshot struct to avoid borrow conflicts between `display.overlay_string()` and HUD render
  - `HudState::new()` now takes `&Display` to initialize backend/density state from detection results

- Updated `src/main.rs`:
  - Registered `mod display`
  - `Display::new()` called after `TerminalGuard::new()` (raw mode required for probing)
  - Framebuffer dimensions derived from `display.framebuffer_size()` instead of hardcoded `cols × rows*2`
  - `needs_fb_resize` flag for backend/density/terminal changes — reallocates framebuffer and calls `camera.resize()`
  - `display.render()` + `display.overlay_string()` + `display.flush()` replaces the old `framebuffer::render_halfblocks` + direct stdout path
  - `display.kitty_cleanup()` on all exit paths (q, Esc, Ctrl-C)
  - The old `framebuffer::render_halfblocks` is still used internally by the HalfBlock backend path

## Files changed

| File | Nature of change |
|------|------------------|
| `src/display.rs` | **New.** Backend enum, Display struct, Kitty protocol, detection, base64 encoder. |
| `src/hud.rs` | Added Display group (backend toggle, px density slider), DisplayInfo struct, Display Info section. |
| `src/main.rs` | Wired Display into render loop, framebuffer sizing, exit cleanup. |

## Key decisions

1. **No external crate for base64 or Kitty.** The base64 encoder is ~30 lines inline. The Kitty protocol is simple enough (APC header + chunked base64 + ST) that pulling in a crate would add more complexity than it removes. No new dependencies added.

2. **Pixel density only affects Kitty backend.** HalfBlock is always 1 pixel per column × 2 pixels per row — there's no meaningful way to vary its density. When the HUD shows "px density" in HalfBlock mode it displays "n/a".

3. **Cell size discovery cascade.** CSI 16t is the most accurate but not universally supported. The ioctl fallback via `crossterm::terminal::window_size()` works on Kitty/WezTerm/Ghostty but returns zeros on many others. The final fallback is 1×2 which just gives HalfBlock resolution even in Kitty mode — safe but not ideal.

4. **Flicker-free updates via placement ID.** The Kitty command uses `i=1000,p=1` so each frame replaces the previous placement atomically. Images are deleted on backend switch and on exit to avoid stale artifacts.

5. **DisplayInfo snapshot pattern.** The HUD render needs to read Display state but the overlay string needs `&mut Display`. Rather than refactoring into a more complex architecture, a small `DisplayInfo` struct captures the needed values before the mutable borrow.

## Important context for future sessions

### How to test

- **Ubuntu default terminal (GNOME Terminal):** Does not support Kitty graphics protocol. The probe will return false and the app falls back to HalfBlock automatically. No change from before.
- **Kitty / Ghostty / WezTerm:** Should detect Kitty graphics support. The app starts in hi-res mode. Open HUD (Tab) to see the resolution, toggle backend, or adjust pixel density.
- **If detection fails but terminal does support Kitty:** This would mean the probe response is getting swallowed. The terminal might need a config tweak. As a future enhancement, a `--backend kitty` CLI flag could force the backend.

### Performance implications

At full density on a 120×40 Kitty terminal with ~10×20 pixel cells, the framebuffer is 1200×800 = 960,000 pixels — 100× more than HalfBlock's 120×80. This will significantly reduce FPS. The pixel density slider (10%-100%) lets users trade resolution for framerate. At 50% density (600×400) the rasterizer workload is 25× HalfBlock — still much heavier but more tractable.

### What was NOT changed

- `framebuffer.rs` is untouched — the HalfBlock backend calls it as before.
- `rasterize.rs`, `camera.rs`, `splat.rs`, `sh.rs` are untouched — the rasterizer is resolution-agnostic.
- No CLI flags were added for backend selection (could be a follow-up).
