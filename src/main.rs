use std::io::{Write, stdout};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::Result;
use clap::Parser as ClapParser;
use crossterm::{
    cursor,
    event::{
        self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyModifiers, MouseEventKind,
    },
    execute, terminal,
};
use glam::Vec3;

mod camera;
mod display;
mod framebuffer;
mod hud;
mod rasterize;
mod sh;
mod splat;

use camera::OrbitCamera;
use display::{Backend, Display};
use hud::{DisplayInfo, HudAction, HudState};
use rasterize::{
    ScratchBuffers, build_thread_pool, composite_parallel, project, sort_by_depth_parallel,
};
use splat::{Splat, load_ply};

const AUTO_ORBIT_RAD_PER_SEC_AT_DEFAULT_ROT_SPEED: f32 = 0.5;
const AUTO_ORBIT_ROT_SPEED_REF: f32 = 0.035;

#[derive(ClapParser, Debug)]
#[command(
    name = "tsplat",
    about = "Terminal 3D Gaussian Splatting (CPU, half-block)"
)]
struct Args {
    ply: PathBuf,
    #[arg(long, default_value_t = 200_000)]
    max_splats: usize,
    #[arg(long, default_value_t = false)]
    no_cap: bool,
    #[arg(long, default_value_t = false)]
    raw_opacity: bool,
    #[arg(long, default_value_t = false)]
    dump_stats: bool,
}

struct TerminalGuard;

impl TerminalGuard {
    fn new() -> Result<Self> {
        terminal::enable_raw_mode()?;
        execute!(
            stdout(),
            terminal::EnterAlternateScreen,
            cursor::Hide,
            EnableMouseCapture
        )?;
        Ok(Self)
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = execute!(
            stdout(),
            DisableMouseCapture,
            cursor::Show,
            terminal::LeaveAlternateScreen
        );
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

fn apply_hud_key_action(
    action: HudAction,
    hud: &HudState,
    camera: &mut OrbitCamera,
    display: &mut Display,
    needs_reload: &mut bool,
    needs_fb_resize: &mut bool,
) {
    match action {
        HudAction::ReloadSplats => *needs_reload = true,
        HudAction::ValueChanged => {
            camera.fov_y = hud.fov_y_deg.to_radians();
        }
        HudAction::BackendChanged => {
            display.backend = hud.backend;
            display.kitty_cleanup();
            display.queue_text_clear();
            *needs_fb_resize = true;
        }
        HudAction::DensityChanged => {
            display.pixel_density = hud.pixel_density;
            display.queue_text_clear();
            *needs_fb_resize = true;
        }
        HudAction::None => {}
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    let cap = if args.no_cap { 0 } else { args.max_splats };
    eprintln!(
        "loading {} (cap: {}) ...",
        args.ply.display(),
        if cap == 0 {
            "none".into()
        } else {
            cap.to_string()
        }
    );
    let (mut splats, total_splats) = load_ply(&args.ply, !args.raw_opacity, cap)?;
    eprintln!("loaded {} / {} splats", splats.len(), total_splats);

    if args.dump_stats {
        return Ok(());
    }

    let _guard = TerminalGuard::new()?;

    let (cols, rows) = terminal::size()?;
    let cols = cols as u32;
    let rows = rows as u32;

    let mut display = Display::new(cols, rows);

    let (mut width, mut height) = display.framebuffer_size();

    let mut camera = OrbitCamera::new(width, height);
    let (center, radius) = scene_bounds(&splats);
    camera.target = center;
    camera.radius = radius * 2.5;

    let mut hud = HudState::new(cap, total_splats, !args.raw_opacity, camera.fov_y, &display);

    let mut fb: Vec<(Vec3, f32)> = vec![(Vec3::ZERO, 0.0); (width * height) as usize];

    let mut frames_since_report = 0u32;
    let mut last_fps_report = Instant::now();
    let mut fps = 0.0_f32;
    let mut needs_reload = false;
    let mut needs_fb_resize = false;
    let mut needs_redraw = true;
    let mut thread_pool = build_thread_pool(hud.num_threads);
    let mut last_num_threads = hud.num_threads;
    let mut scratch = ScratchBuffers::new();
    let mut auto_orbit = false;
    let mut last_wallclock = Instant::now();

    loop {
        if needs_fb_resize {
            let (new_w, new_h) = display.framebuffer_size();
            width = new_w;
            height = new_h;
            camera.resize(width, height);
            fb = vec![(Vec3::ZERO, 0.0); (width * height) as usize];
            needs_fb_resize = false;
        }

        if needs_reload {
            {
                let mut so = stdout().lock();
                so.write_all(b"\x1b[1;1H\x1b[97;41m Loading... \x1b[0m")?;
                so.flush()?;
            }
            let (new_splats, _total) = load_ply(&args.ply, hud.apply_sigmoid, hud.max_splats)?;
            splats = new_splats;
            needs_reload = false;
            display.queue_text_clear();
            needs_redraw = true;
        }

        let now = Instant::now();
        let mut dt = now.duration_since(last_wallclock).as_secs_f32();
        last_wallclock = now;
        dt = dt.min(0.25);
        if auto_orbit {
            let omega = hud.rotation_speed
                * (AUTO_ORBIT_RAD_PER_SEC_AT_DEFAULT_ROT_SPEED / AUTO_ORBIT_ROT_SPEED_REF);
            camera.orbit(omega * dt, 0.0);
            needs_redraw = true;
        }

        if needs_redraw {
            unsafe {
                std::ptr::write_bytes(fb.as_mut_ptr(), 0, fb.len());
            }
            if hud.num_threads != last_num_threads {
                thread_pool = build_thread_pool(hud.num_threads);
                last_num_threads = hud.num_threads;
            }

            let render_params = &hud.render_params;
            let mut projected = project(&splats, &camera, render_params, &thread_pool);
            sort_by_depth_parallel(&mut projected, &mut scratch, &thread_pool);
            composite_parallel(
                &projected,
                &mut fb,
                width,
                height,
                render_params,
                &mut scratch,
                &thread_pool,
            );

            display.render(&fb, width, height);

            frames_since_report += 1;
            let now = Instant::now();
            let elapsed = now.duration_since(last_fps_report).as_secs_f32();
            if elapsed >= 0.5 {
                fps = frames_since_report as f32 / elapsed;
                frames_since_report = 0;
                last_fps_report = now;
            }
            let fps_str = if auto_orbit {
                format!(" FPS {:5.1} ORBIT ", fps)
            } else {
                format!(" FPS {:5.1} ", fps)
            };
            let col = (display.cols as usize)
                .saturating_sub(fps_str.chars().count())
                .max(0)
                + 1;
            {
                let di = DisplayInfo::from_display(&display);
                let out = display.overlay_string();
                use std::fmt::Write as _;
                let _ = write!(out, "\x1b[1;{}H\x1b[97;40m{}\x1b[0m", col, fps_str);

                hud.render(&camera, &di, out);
            }

            display.flush()?;
            needs_redraw = false;
        }

        let poll_wait = if auto_orbit {
            Duration::from_millis(16)
        } else {
            Duration::from_millis(33)
        };
        if !event::poll(poll_wait)? {
            continue;
        }

        while event::poll(Duration::from_millis(0))? {
            match event::read()? {
                Event::Key(k) => match k.code {
                    KeyCode::Char('q') => {
                        display.kitty_cleanup();
                        return Ok(());
                    }
                    KeyCode::Esc => {
                        if hud.visible {
                            hud.toggle();
                            if display.backend == Backend::Kitty {
                                display.queue_text_clear();
                            }
                            needs_redraw = true;
                        } else {
                            display.kitty_cleanup();
                            return Ok(());
                        }
                    }
                    KeyCode::Char('c') if k.modifiers.contains(KeyModifiers::CONTROL) => {
                        display.kitty_cleanup();
                        return Ok(());
                    }
                    KeyCode::Tab => {
                        let was_visible = hud.visible;
                        hud.toggle();
                        if was_visible && !hud.visible && display.backend == Backend::Kitty {
                            display.queue_text_clear();
                        }
                        needs_redraw = true;
                    }
                    KeyCode::Char('h') | KeyCode::Char('H') => {
                        camera.orbit(0.0, hud.rotation_speed);
                        needs_redraw = true;
                    }
                    KeyCode::Char('l') | KeyCode::Char('L') => {
                        camera.orbit(0.0, -hud.rotation_speed);
                        needs_redraw = true;
                    }
                    KeyCode::Char('j') | KeyCode::Char('J') => {
                        camera.orbit(-hud.rotation_speed, 0.0);
                        needs_redraw = true;
                    }
                    KeyCode::Char('k') | KeyCode::Char('K') => {
                        camera.orbit(hud.rotation_speed, 0.0);
                        needs_redraw = true;
                    }
                    KeyCode::Char('+') | KeyCode::Char('=') => {
                        camera.zoom(1.0 - hud.zoom_step);
                        needs_redraw = true;
                    }
                    KeyCode::Char('-') | KeyCode::Char('_') => {
                        camera.zoom(1.0 + hud.zoom_step);
                        needs_redraw = true;
                    }
                    KeyCode::Char('w') | KeyCode::Char('W') => {
                        let step = camera.radius * hud.translate_speed;
                        camera.pan(0.0, step);
                        needs_redraw = true;
                    }
                    KeyCode::Char('s') | KeyCode::Char('S') => {
                        let step = camera.radius * hud.translate_speed;
                        camera.pan(0.0, -step);
                        needs_redraw = true;
                    }
                    KeyCode::Char('a') | KeyCode::Char('A') => {
                        let step = camera.radius * hud.translate_speed;
                        camera.pan(-step, 0.0);
                        needs_redraw = true;
                    }
                    KeyCode::Char('d') | KeyCode::Char('D') => {
                        let step = camera.radius * hud.translate_speed;
                        camera.pan(step, 0.0);
                        needs_redraw = true;
                    }
                    KeyCode::Char('o') | KeyCode::Char('O') => {
                        auto_orbit = !auto_orbit;
                        last_wallclock = Instant::now();
                        needs_redraw = true;
                    }
                    KeyCode::Up | KeyCode::Down | KeyCode::Left | KeyCode::Right => {
                        if hud.visible {
                            let action = hud.handle_key(k.code);
                            apply_hud_key_action(
                                action,
                                &hud,
                                &mut camera,
                                &mut display,
                                &mut needs_reload,
                                &mut needs_fb_resize,
                            );
                            needs_redraw = true;
                        } else {
                            match k.code {
                                KeyCode::Left => camera.orbit(-hud.rotation_speed, 0.0),
                                KeyCode::Right => camera.orbit(hud.rotation_speed, 0.0),
                                KeyCode::Up => camera.orbit(0.0, hud.rotation_speed),
                                KeyCode::Down => camera.orbit(0.0, -hud.rotation_speed),
                                _ => {}
                            }
                            needs_redraw = true;
                        }
                    }
                    KeyCode::PageUp
                    | KeyCode::PageDown
                    | KeyCode::Char(',')
                    | KeyCode::Char('.')
                    | KeyCode::Char('<')
                    | KeyCode::Char('>') => {
                        if hud.visible {
                            let action = hud.handle_key(k.code);
                            apply_hud_key_action(
                                action,
                                &hud,
                                &mut camera,
                                &mut display,
                                &mut needs_reload,
                                &mut needs_fb_resize,
                            );
                            needs_redraw = true;
                        }
                    }
                    _ => {}
                },
                Event::Resize(new_cols, new_rows) => {
                    display.resize(new_cols as u32, new_rows as u32);
                    display.queue_text_clear();
                    needs_fb_resize = true;
                    needs_redraw = true;
                }
                Event::Mouse(me) => match me.kind {
                    MouseEventKind::ScrollUp => {
                        camera.zoom(1.0 - hud.zoom_step);
                        needs_redraw = true;
                    }
                    MouseEventKind::ScrollDown => {
                        camera.zoom(1.0 + hud.zoom_step);
                        needs_redraw = true;
                    }
                    _ => {}
                },
                _ => {}
            }
        }
    }
}
