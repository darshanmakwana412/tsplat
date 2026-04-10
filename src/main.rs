use std::io::{Write, stdout};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::Result;
use clap::Parser as ClapParser;
use crossterm::{
    cursor,
    event::{self, EnableMouseCapture, DisableMouseCapture, Event, KeyCode, KeyModifiers, MouseEventKind},
    execute, terminal,
};
use glam::Vec3;

mod camera;
mod framebuffer;
mod hud;
mod rasterize;
mod sh;
mod splat;

use camera::OrbitCamera;
use hud::{HudAction, HudState};
use rasterize::{composite_parallel, project, sort_by_depth};
use splat::{Splat, load_ply};

#[derive(ClapParser, Debug)]
#[command(
    name = "tsplat",
    about = "Terminal 3D Gaussian Splatting renderer (CPU, half-block)"
)]
struct Args {
    /// Path to an INRIA 3DGS `.ply` scene.
    ply: PathBuf,

    /// Maximum number of splats to load. Set to 0 (or pass --no-cap) to
    /// load everything.
    #[arg(long, default_value_t = 200_000)]
    max_splats: usize,

    /// Load every splat, regardless of --max-splats.
    #[arg(long, default_value_t = false)]
    no_cap: bool,

    /// Treat opacity as already in [0, 1] instead of applying sigmoid.
    /// Useful if the scene looks hazy on first load.
    #[arg(long, default_value_t = false)]
    raw_opacity: bool,

    /// Load, print stats, exit — don't enter the render loop.
    #[arg(long, default_value_t = false)]
    dump_stats: bool,
}

/// RAII guard that enters alt-screen + raw mode on construction and restores
/// the terminal on drop (including on panic).
struct TerminalGuard;

impl TerminalGuard {
    fn new() -> Result<Self> {
        terminal::enable_raw_mode()?;
        execute!(stdout(), terminal::EnterAlternateScreen, cursor::Hide, EnableMouseCapture)?;
        Ok(Self)
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = execute!(stdout(), DisableMouseCapture, cursor::Show, terminal::LeaveAlternateScreen);
        let _ = terminal::disable_raw_mode();
    }
}

fn scene_bounds(splats: &[Splat]) -> (Vec3, f32) {
    if splats.is_empty() {
        return (Vec3::ZERO, 1.0);
    }
    let mut mn = Vec3::splat(f32::INFINITY);
    let mut mx = Vec3::splat(f32::NEG_INFINITY);
    for s in splats {
        mn = mn.min(s.pos);
        mx = mx.max(s.pos);
    }
    let center = (mn + mx) * 0.5;
    let radius = (mx - mn).length() * 0.5;
    (center, radius.max(1.0))
}

fn main() -> Result<()> {
    let args = Args::parse();

    let cap = if args.no_cap { 0 } else { args.max_splats };
    eprintln!(
        "loading {} (cap: {}) ...",
        args.ply.display(),
        if cap == 0 { "none".into() } else { cap.to_string() }
    );
    let mut splats = load_ply(&args.ply, !args.raw_opacity, cap)?;
    eprintln!("loaded {} splats", splats.len());

    if args.dump_stats {
        return Ok(());
    }

    // Guard runs first so a panic restores the terminal.
    let _guard = TerminalGuard::new()?;

    let (cols, rows) = terminal::size()?;
    let mut width = cols as u32;
    let mut height = rows as u32 * 2;

    let mut camera = OrbitCamera::new(width, height);
    let (center, radius) = scene_bounds(&splats);
    camera.target = center;
    camera.radius = radius * 2.5;

    let mut hud = HudState::new(cap, !args.raw_opacity, camera.fov_y);

    let mut fb: Vec<(Vec3, f32)> = vec![(Vec3::ZERO, 0.0); (width * height) as usize];
    let mut out = String::with_capacity(256 * 1024);

    let mut frames_since_report = 0u32;
    let mut last_fps_report = Instant::now();
    let mut fps = 0.0_f32;
    let mut needs_reload = false;

    loop {
        // ---- drain input (non-blocking) ----
        while event::poll(Duration::from_millis(0))? {
            match event::read()? {
                Event::Key(k) => match k.code {
                    KeyCode::Char('q') => return Ok(()),
                    KeyCode::Esc => {
                        if hud.visible {
                            hud.toggle();
                        } else {
                            return Ok(());
                        }
                    }
                    KeyCode::Char('c') if k.modifiers.contains(KeyModifiers::CONTROL) => {
                        return Ok(());
                    }
                    KeyCode::Tab => hud.toggle(),
                    // Vim keys always control camera
                    KeyCode::Char('h') => camera.orbit(-hud.yaw_step, 0.0),
                    KeyCode::Char('l') => camera.orbit(hud.yaw_step, 0.0),
                    KeyCode::Char('k') => camera.orbit(0.0, hud.pitch_step),
                    KeyCode::Char('j') => camera.orbit(0.0, -hud.pitch_step),
                    KeyCode::Char('+') | KeyCode::Char('=') => camera.zoom(1.0 - hud.zoom_step),
                    KeyCode::Char('-') | KeyCode::Char('_') => camera.zoom(1.0 + hud.zoom_step),
                    // WASD: spatial pan (proportional to current radius)
                    KeyCode::Char('w') | KeyCode::Char('W') => {
                        let step = camera.radius * 0.05;
                        camera.pan(0.0, step);
                    }
                    KeyCode::Char('s') | KeyCode::Char('S') => {
                        let step = camera.radius * 0.05;
                        camera.pan(0.0, -step);
                    }
                    KeyCode::Char('a') | KeyCode::Char('A') => {
                        let step = camera.radius * 0.05;
                        camera.pan(-step, 0.0);
                    }
                    KeyCode::Char('d') | KeyCode::Char('D') => {
                        let step = camera.radius * 0.05;
                        camera.pan(step, 0.0);
                    }
                    // Arrow keys: HUD when visible, camera when hidden
                    KeyCode::Up | KeyCode::Down | KeyCode::Left | KeyCode::Right => {
                        if hud.visible {
                            let action = hud.handle_key(k.code);
                            match action {
                                HudAction::ReloadSplats => needs_reload = true,
                                HudAction::ValueChanged => {
                                    camera.fov_y = hud.fov_y_deg.to_radians();
                                }
                                HudAction::None => {}
                            }
                        } else {
                            match k.code {
                                KeyCode::Left => camera.orbit(-hud.yaw_step, 0.0),
                                KeyCode::Right => camera.orbit(hud.yaw_step, 0.0),
                                KeyCode::Up => camera.orbit(0.0, hud.pitch_step),
                                KeyCode::Down => camera.orbit(0.0, -hud.pitch_step),
                                _ => {}
                            }
                        }
                    }
                    _ => {}
                },
                Event::Resize(new_cols, new_rows) => {
                    width = new_cols as u32;
                    height = new_rows as u32 * 2;
                    camera.resize(width, height);
                    fb = vec![(Vec3::ZERO, 0.0); (width * height) as usize];
                }
                Event::Mouse(me) => match me.kind {
                    MouseEventKind::ScrollUp => camera.zoom(1.0 - hud.zoom_step),
                    MouseEventKind::ScrollDown => camera.zoom(1.0 + hud.zoom_step),
                    _ => {}
                },
                _ => {}
            }
        }

        // ---- reload splats if requested ----
        if needs_reload {
            {
                let mut so = stdout().lock();
                so.write_all(b"\x1b[1;1H\x1b[97;41m Loading... \x1b[0m")?;
                so.flush()?;
            }
            splats = load_ply(&args.ply, hud.apply_sigmoid, hud.max_splats)?;
            needs_reload = false;
        }

        // ---- render ----
        for c in fb.iter_mut() {
            *c = (Vec3::ZERO, 0.0);
        }
        let render_params = &hud.render_params;
        let mut projected = project(&splats, &camera, render_params);
        sort_by_depth(&mut projected);
        composite_parallel(&projected, &mut fb, width, height, render_params, hud.num_threads);
        framebuffer::render_halfblocks(&fb, width, height, &mut out);

        // ---- FPS overlay ----
        frames_since_report += 1;
        let now = Instant::now();
        let elapsed = now.duration_since(last_fps_report).as_secs_f32();
        if elapsed >= 0.5 {
            fps = frames_since_report as f32 / elapsed;
            frames_since_report = 0;
            last_fps_report = now;
        }
        let fps_str = format!(" FPS {:5.1} ", fps);
        let col = (width as usize)
            .saturating_sub(fps_str.chars().count())
            .max(0)
            + 1; // 1-indexed
        use std::fmt::Write as _;
        let _ = write!(out, "\x1b[1;{}H\x1b[97;40m{}\x1b[0m", col, fps_str);

        // ---- HUD overlay ----
        hud.render(&camera, &mut out);

        // ---- single flush ----
        let mut so = stdout().lock();
        so.write_all(out.as_bytes())?;
        so.flush()?;
    }
}
