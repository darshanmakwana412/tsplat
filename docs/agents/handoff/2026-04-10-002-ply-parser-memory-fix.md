# tsplat PLY parser memory fix

Date: 2026-04-10
Session focus: diagnosing and fixing an OOM/hang when loading the garden scene, and verifying the binary compiles and loads correctly.

## What was accomplished

- Replaced the `ply-rs`-based PLY loader in `src/splat.rs` with a hand-rolled binary PLY parser.
- Removed `ply-rs` from `Cargo.toml` (no longer a dependency).
- Moved the `max_splats` cap parameter into `load_ply` itself, so subsampling happens during the read rather than after full load.
- Removed `downsample_uniform` from `splat.rs` (no longer needed) and cleaned up the import in `main.rs`.
- Updated the startup log line in `main.rs` to show the active cap.
- Verified `cargo build --release` is clean with zero warnings.
- Verified `--dump-stats` loads 200k splats (default cap) instantly.
- Verified `--dump-stats --no-cap` loads all 5,834,784 splats in ~0.4s.

## Key decisions

1. **Stream one record at a time, subsample during load.** The old approach loaded all vertices into `Vec<DefaultElement>` (a `LinkedHashMap<String, Property>` per vertex) and subsampled afterwards. For the garden scene (~5.8M vertices, 62 properties each), that materialised ~360M HashMap entries and several GB of heap overhead before a single splat was decoded. The new parser reads one fixed-size vertex record at a time, skips records that fall outside the uniform subsample stride, and decodes only the fields it needs. Peak allocation at the default 200k cap is ~10MB.

2. **Binary-only, little-endian.** All INRIA 3DGS PLY files use `format binary_little_endian 1.0`. The parser bails with a clear error if it encounters any other format. ASCII PLY support is not needed and was not added.

3. **All properties assumed to be `float32`.** The INRIA format uses only `property float` entries. The parser counts properties and multiplies by 4 to get the per-vertex stride. If a future PLY uses `property double` or `property int`, the parser will miscompute offsets and should be extended then.

4. **`downsample_uniform` removed entirely.** There is no post-load subsampling path anymore. The cap is enforced inside `load_ply`. This means `--max-splats` and `--no-cap` are the only controls, which is sufficient for the MVP.

## Important context for future sessions

### Scene file location

The garden scene is at `data/garden/point_cloud.ply` inside the repo directory (not tracked by git; `*.ply` is in `.gitignore`). It has 5,834,784 splats. The default cap of 200k loads in well under a second. The full scene loads in ~0.4s.

### Interactive viewer not yet visually verified

The binary loads and exits cleanly under `--dump-stats`. The interactive render loop (`cargo run --release -- data/garden/point_cloud.ply`) has not been run in a terminal session yet, so the visual correctness checks from the original plan remain open:

1. Scene may render upside-down -- caused by the Y-sign convention in `rasterize.rs`. Fix by removing both sign flips together (the `-fy` on row 2 of `J` and the `-` in the `sy` expression). Never flip just one.
2. Scene may look uniformly hazy (~50% transparent) -- run with `--raw-opacity` to skip the sigmoid.
3. Quaternion component order mismatch -- INRIA stores `rot_0..3` as `w,x,y,z`; the loader calls `Quat::from_xyzw(rx, ry, rz, rw)`, which is correct, but verify visually.

### CLI surface

```
tsplat <PLY>
    --max-splats <N>   (default 200000; load at most N splats)
    --no-cap           (load everything; equivalent to --max-splats 0)
    --raw-opacity      (skip sigmoid on opacity; use if scene is hazy)
    --dump-stats       (load, print count, exit without rendering)
```

### Repository state

- Branch: `main`.
- Modified files relative to last commit: `Cargo.toml`, `Cargo.lock`, `src/main.rs`, `src/splat.rs`. Deleted: `data/3dgs/.gitkeep` (pre-existing). Nothing has been committed in this session or the previous one.
- `LichtFeld-Studio/` is a reference-only C++ submodule. Do not modify it.
