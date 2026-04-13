# tsplat auto orbit for screen capture (O key)

Date: 2026-04-13
Session focus: add a hands free orbit around the current view target so a
GitHub teaser or asciinema style recording can show the viewer without holding
keys. Toggle with `o` or `O`. Orbit is time based yaw only, in the same
direction as `k` and Right arrow positive yaw.

## What was accomplished

- **`src/main.rs`**: boolean `auto_orbit` toggled by `KeyCode::Char('o')` and
  `'O'`. Each loop iteration computes `dt` from wall clock since the previous
  iteration, capped at `0.25` seconds to avoid large jumps after stalls.
- **Orbit step**: `camera.orbit(omega * dt, 0.0)` with `omega` proportional to
  `hud.rotation_speed`. Two module level constants map the HUD default
  `rotation_speed` of `0.035` to `0.5` rad/s:
  `AUTO_ORBIT_RAD_PER_SEC_AT_DEFAULT_ROT_SPEED` and `AUTO_ORBIT_ROT_SPEED_REF`.
  Raising the HUD rotation speed row speeds up auto orbit in proportion.
- **Idle polling**: when `auto_orbit` is true, `event::poll` uses `16` ms
  instead of `33` ms so the loop wakes often enough for smoother motion when
  the machine keeps up with redraw cost.
- **FPS strip**: when orbit is active the overlay text becomes
  `FPS … ORBIT ` so recordings show the mode.
- **Toggle edge case**: on toggle, `last_wallclock` resets to `Instant::now()`
  so the first frame after enabling does not accumulate a huge `dt`.
- **Verification**: `cargo build --release` and
  `cargo test --release --test regression` both succeeded after the change.

## Files changed

| File | Nature of change |
|------|------------------|
| `src/main.rs` | Constants for rad/s mapping, `auto_orbit` and `last_wallclock`, per loop `dt` and conditional `orbit`, shorter poll when orbiting, `o` / `O` key arm, FPS string branch. |

No edits to `camera.rs`, `hud.rs`, or `CLAUDE.md` in this session.

## Key decisions

1. **Yaw only, no auto pitch.** Teaser clips usually want a level horizon; full
   spherical sweep would need extra UX and risks odd framing.

2. **Reuse HUD rotation speed instead of a fourth HUD row.** Keeps the panel
   small and lets one slider tune both key tap orbit steps and continuous orbit
   rate. The reference mapping is documented in constants next to the loop.

3. **Positive yaw matches existing `k` / Right.** Consistent with current
   keybindings; reversing would confuse users who orbit manually then enable
   auto.

4. **`o` works regardless of HUD visibility.** Same pattern as WASD: no branch
   on `hud.visible` for this key. The HUD does not bind `o`.

5. **Explicitly not done:** no CLI flag for initial auto orbit, no separate
   rad/s field in the HUD, no change to dirty frame policy beyond setting
   `needs_redraw` every frame while orbit is on (expected cost while recording).

## Important context for future sessions

- **Camera model:** still `OrbitCamera` with `target`, `yaw`, `pitch`, `radius`.
  Auto orbit only calls the existing `orbit` API; it does not move `target`, so
  if the user panned off center the orbit circles whatever target is current.

- **Performance:** with auto orbit on, every poll timeout path still runs a
   full raster pass because `needs_redraw` stays true. That is intentional for
   video; turning orbit off returns to normal dirty frame behavior.

- **Docs:** workspace `CLAUDE.md` was not updated in this session. If the
   public controls list should mention `o`, add one line there in a follow up.

- **Repository state:** confirm with `git status` before picking up unrelated
  work; this handoff does not assume a committed state.
