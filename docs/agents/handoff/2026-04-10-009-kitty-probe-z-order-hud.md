# tsplat Kitty detection fix and graphics z-order for HUD

Date: 2026-04-10
Session focus: Fix Kitty hi-res mode falsely falling back to HalfBlock, and stop FPS or HUD blinking when the Kitty graphics backend is active.

## What was accomplished

- **Root cause of false HalfBlock in Kitty:** `probe_kitty_support()` and `query_cell_size()` waited for input using `crossterm::event::poll()` while reading with `stdin.lock().read()`. Poll drives crossterm's tty reader, which feeds bytes into crossterm's escape parser. Kitty graphics replies (`ESC _ G ...` APC) are not crossterm events; the parser treats them as garbage and drops them. The manual buffer never saw `OK`, the probe timed out, and the app always selected HalfBlock even in Kitty.

- **Detection path rewrite (Unix):** Added `[target.'cfg(unix)'.dependencies] libc = "0.2"` in `Cargo.toml`. In `src/display.rs`, probe and CSI `16t` cell-size query now open `/dev/tty` for reading, use `libc::poll` for readiness, and `read()` directly. Crossterm is not used on that code path, so APC responses are not consumed by the event parser.

- **`query_cell_size` robustness:** Removed the early `return None` when the first `DA1` (`ESC [ ? ... c`) appeared in the buffer, because a successful graphics probe can leave the prior `ESC [ c` reply in the queue and that was mistaken for "CSI `16t` unsupported". The loop now waits for a parseable `ESC [ 6 ; height ; width t` within the deadline. After a successful parse, one short polled read drains the matching DA1 for the query's own `ESC [ c`. Read buffer for the query was increased to 256 bytes.

- **`contains_kitty_ok`:** Still matches `ESC _ Gi=31;OK ESC \`. Also accepts string terminator `0x9c` instead of `ESC \`, and a loose substring `i=31;OK` for minor format differences.

- **Non-Unix:** `probe_kitty_support()` returns false; `query_cell_size()` returns `None` (same practical behavior as before for Windows-only builds).

- **HUD or FPS blinking in Kitty:** Default stacking draws the full-screen `a=T` image above the text layer, so each frame refresh briefly covered cursor-addressed overlays. The Kitty spec uses negative `z` to draw under text; for cells with explicit SGR backgrounds (HUD uses `40m`, `107m`, etc.) `z` must be below `INT32_MIN / 2`. Added `KITTY_IMG_Z_UNDER_UI = -1_073_741_825` and pass `z=` in the first chunked graphics escape in `render_kitty`.

## Files changed

| File | Nature of change |
|------|------------------|
| `Cargo.toml` | Unix-only `libc` dependency for `poll` on tty reads during detection. |
| `src/display.rs` | Tty poll helpers, probe and `query_cell_size` without crossterm poll, DA1 or buffer tweaks, `contains_kitty_ok` variants, Kitty `z=` constant and metadata on `a=T`. |

## Key decisions

1. **Libc on Unix only.** Avoid pulling `libc` on Windows targets; Kitty graphics probing is not wired there anyway.

2. **Read `/dev/tty` for probe or query, not only stdin.** Aligns with needing a real tty when stdin might differ from the controlling terminal in odd setups; matches the spirit of crossterm's `tty_fd()`.

3. **Do not use crossterm for bytes during Kitty probe or cell-size query.** Any future code that reads graphics protocol replies from the tty must not go through `crossterm::event::poll` first on the same stream.

4. **Z value for under UI.** `-1_073_741_825` follows Kitty's rule for drawing under cells with non-default ANSI backgrounds so the HUD panel and FPS bar stay stable, not only plain default-background cells.

5. **Ruled out for this session:** Switching to transmit-then-place (`a=p` only) or Unicode placeholder placements to fix flicker; a single `z=` change was enough. Forcing Kitty via CLI flag remains a possible follow-up from handoff 008.

## Important context for future sessions

### Symptoms that led here

- In Kitty, scene looked half-block, often "blank" until a key such as an arrow (camera move made splats visible on a dark first frame). That was HalfBlock mode plus a black-looking first frame, not a broken Kitty image path.

### Testing

- Run `cargo build --release` and `cargo run --release -- <scene.ply>` inside Kitty. HUD (Tab): backend should show `kitty`, cell size not `1 x 2` when ioctl and CSI `16t` succeed.

### Pre-existing notes

- Handoff `2026-04-10-008-kitty-graphics-hires-backend.md` describes the original Display module and HUD wiring; this session fixes detection and overlay stacking on top of that work.

### Branch or remote

- Confirm `git status` and `origin` before assuming branch name; this handoff was written with workspace on `main` tracking `origin/main` unless the clone differs.
