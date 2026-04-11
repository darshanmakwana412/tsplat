use std::fmt::Write;

use crossterm::event::KeyCode;

use crate::camera::OrbitCamera;
use crate::display::{Backend, CellSize, Display};
use crate::rasterize::RenderParams;

/// Snapshot of Display state needed for HUD rendering (avoids borrow conflicts).
pub struct DisplayInfo {
    pub fb_w: u32,
    pub fb_h: u32,
    pub cell_size: CellSize,
    pub cols: u32,
    pub rows: u32,
    pub detected_backend: Backend,
}

impl DisplayInfo {
    pub fn from_display(d: &Display) -> Self {
        let (fb_w, fb_h) = d.framebuffer_size();
        Self {
            fb_w,
            fb_h,
            cell_size: d.cell_size,
            cols: d.cols,
            rows: d.rows,
            detected_backend: d.detected_backend,
        }
    }
}

/// What happened after the HUD processed a key.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum HudAction {
    /// Nothing changed.
    None,
    /// A real-time parameter changed (camera or render). No reload needed.
    ValueChanged,
    /// A parameter that requires a PLY reload changed (max_splats or sigmoid).
    ReloadSplats,
    /// The rendering backend was switched.
    BackendChanged,
    /// Pixel density changed (needs framebuffer resize).
    DensityChanged,
}

/// Identity of each tunable item in the HUD.
#[derive(Clone, Copy)]
enum HudItem {
    MaxSplats,
    Sigmoid,
    NumThreads,
    RenderBackend,
    PixelDensity,
    FovY,
    TranslateSpeed,
    RotationSpeed,
    ZoomStep,
    Eps2d,
    AlphaThreshold,
    ExtendSigma,
    Saturation,
}

/// An entry in the flat display list.
struct Entry {
    group: Option<&'static str>,
    item: HudItem,
    label: &'static str,
}

const ITEMS: &[Entry] = &[
    Entry { group: Some("Splats"),   item: HudItem::MaxSplats,       label: "max_splats" },
    Entry { group: None,             item: HudItem::Sigmoid,         label: "sigmoid" },
    Entry { group: Some("Display"),  item: HudItem::RenderBackend,   label: "backend" },
    Entry { group: None,             item: HudItem::PixelDensity,    label: "px density" },
    Entry { group: Some("Perf"),     item: HudItem::NumThreads,      label: "threads" },
    Entry { group: Some("Camera"),   item: HudItem::FovY,            label: "fov_y" },
    Entry { group: None,             item: HudItem::TranslateSpeed,  label: "move spd" },
    Entry { group: None,             item: HudItem::RotationSpeed,   label: "rot spd" },
    Entry { group: None,             item: HudItem::ZoomStep,        label: "zoom spd" },
    Entry { group: Some("Render"),   item: HudItem::Eps2d,           label: "eps2d" },
    Entry { group: None,             item: HudItem::AlphaThreshold,  label: "alpha thr" },
    Entry { group: None,             item: HudItem::ExtendSigma,     label: "sigma ext" },
    Entry { group: None,             item: HudItem::Saturation,      label: "saturation" },
];

/// Width of the HUD panel in characters.
const HUD_WIDTH: usize = 32;

pub struct HudState {
    pub visible: bool,
    cursor: usize,

    // -- Reload parameters --
    pub max_splats: usize,
    /// Total vertex count declared in the PLY header. Used as the ceiling for
    /// the MaxSplats slider so the user can actually reach "all of it" instead
    /// of a magic 10M cap.
    pub total_splats: usize,
    pub apply_sigmoid: bool,

    // -- Performance parameters (real-time) --
    pub num_threads: usize,

    // -- Display parameters --
    pub backend: Backend,
    pub detected_backend: Backend,
    pub pixel_density: f32,

    // -- Camera parameters (real-time) --
    pub fov_y_deg: f32,
    /// Pan distance per key as a fraction of orbit radius (WASD).
    pub translate_speed: f32,
    /// Radians per key for yaw and pitch (HJKL / arrows).
    pub rotation_speed: f32,
    /// Multiplicative zoom factor per +/- or scroll tick (see `OrbitCamera::zoom`).
    pub zoom_step: f32,

    // -- Render parameters (real-time) --
    pub render_params: RenderParams,
}

impl HudState {
    pub fn new(
        max_splats: usize,
        total_splats: usize,
        apply_sigmoid: bool,
        fov_y_rad: f32,
        display: &Display,
    ) -> Self {
        // `max_splats == 0` (CLI --no-cap) means "load everything" which we
        // represent internally as "equal to the scene total" so the HUD slider
        // has a concrete value to show and clamp to.
        let max_splats = if max_splats == 0 || max_splats > total_splats {
            total_splats
        } else {
            max_splats
        };
        Self {
            visible: false,
            cursor: 0,
            max_splats,
            total_splats,
            apply_sigmoid,
            num_threads: 4,
            backend: display.backend,
            detected_backend: display.detected_backend,
            pixel_density: display.pixel_density,
            fov_y_deg: fov_y_rad.to_degrees(),
            translate_speed: 0.02,
            rotation_speed: 0.035,
            zoom_step: 0.06,
            render_params: RenderParams::default(),
        }
    }

    pub fn toggle(&mut self) {
        self.visible = !self.visible;
    }

    /// Process HUD navigation while the panel is visible: Up/Down move the
    /// selection, Left/Right adjust the focused value. PgUp/PgDn and `,` /
    /// `.` are aliases for the same actions.
    pub fn handle_key(&mut self, code: KeyCode) -> HudAction {
        match code {
            KeyCode::Up | KeyCode::PageUp => {
                if self.cursor == 0 {
                    self.cursor = ITEMS.len() - 1;
                } else {
                    self.cursor -= 1;
                }
                HudAction::None
            }
            KeyCode::Down | KeyCode::PageDown => {
                self.cursor = (self.cursor + 1) % ITEMS.len();
                HudAction::None
            }
            KeyCode::Left | KeyCode::Char(',') | KeyCode::Char('<') => self.adjust(-1),
            KeyCode::Right | KeyCode::Char('.') | KeyCode::Char('>') => self.adjust(1),
            _ => HudAction::None,
        }
    }

    fn adjust(&mut self, dir: i8) -> HudAction {
        let item = ITEMS[self.cursor].item;
        match item {
            HudItem::MaxSplats => {
                let new_val = if dir > 0 {
                    // Increase (right / `.`) at or above the scene total snaps to the
                    // full count (covers the "one more tap to go all the
                    // way" case when 2x would overshoot).
                    (self.max_splats.saturating_mul(2)).min(self.total_splats.max(1_000))
                } else {
                    (self.max_splats / 2).max(1_000)
                };
                if new_val == self.max_splats {
                    return HudAction::None;
                }
                self.max_splats = new_val;
                HudAction::ReloadSplats
            }
            HudItem::Sigmoid => {
                self.apply_sigmoid = !self.apply_sigmoid;
                HudAction::ReloadSplats
            }
            HudItem::NumThreads => {
                let max_cpus = std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(16);
                if dir > 0 {
                    self.num_threads = (self.num_threads + 1).min(max_cpus);
                } else {
                    self.num_threads = self.num_threads.saturating_sub(1);
                }
                HudAction::ValueChanged
            }
            HudItem::RenderBackend => {
                // Toggle between available backends.
                self.backend = match self.backend {
                    Backend::HalfBlock => {
                        if self.detected_backend == Backend::Kitty {
                            Backend::Kitty
                        } else {
                            Backend::HalfBlock // can't switch if not supported
                        }
                    }
                    Backend::Kitty => Backend::HalfBlock,
                };
                HudAction::BackendChanged
            }
            HudItem::PixelDensity => {
                // Step by 0.05 between 0.10 and 1.0.
                self.pixel_density = (self.pixel_density + dir as f32 * 0.05).clamp(0.10, 1.0);
                // Round to avoid float drift.
                self.pixel_density = (self.pixel_density * 20.0).round() / 20.0;
                HudAction::DensityChanged
            }
            HudItem::FovY => {
                self.fov_y_deg = (self.fov_y_deg + dir as f32 * 5.0).clamp(10.0, 150.0);
                HudAction::ValueChanged
            }
            HudItem::TranslateSpeed => {
                self.translate_speed =
                    (self.translate_speed + dir as f32 * 0.005).clamp(0.01, 0.25);
                HudAction::ValueChanged
            }
            HudItem::RotationSpeed => {
                self.rotation_speed =
                    (self.rotation_speed + dir as f32 * 0.01).clamp(0.01, 0.50);
                HudAction::ValueChanged
            }
            HudItem::ZoomStep => {
                self.zoom_step = (self.zoom_step + dir as f32 * 0.05).clamp(0.05, 0.50);
                HudAction::ValueChanged
            }
            HudItem::Eps2d => {
                self.render_params.eps2d =
                    (self.render_params.eps2d + dir as f32 * 0.05).clamp(0.0, 2.0);
                HudAction::ValueChanged
            }
            HudItem::AlphaThreshold => {
                self.render_params.alpha_threshold = if dir > 0 {
                    (self.render_params.alpha_threshold * 2.0).min(0.1)
                } else {
                    (self.render_params.alpha_threshold * 0.5).max(1.0 / 1020.0)
                };
                HudAction::ValueChanged
            }
            HudItem::ExtendSigma => {
                self.render_params.extend_sigma =
                    (self.render_params.extend_sigma + dir as f32 * 0.5).clamp(1.0, 6.0);
                HudAction::ValueChanged
            }
            HudItem::Saturation => {
                self.render_params.saturation =
                    (self.render_params.saturation + dir as f32 * 0.001).clamp(0.9, 1.0);
                HudAction::ValueChanged
            }
        }
    }

    fn format_value(&self, item: HudItem) -> String {
        match item {
            HudItem::MaxSplats => {
                let label = if self.max_splats >= 1_000_000 {
                    format!("{:.1}M", self.max_splats as f32 / 1_000_000.0)
                } else {
                    format!("{}k", self.max_splats / 1_000)
                };
                if self.total_splats > 0 && self.max_splats >= self.total_splats {
                    format!("all/{label}")
                } else {
                    label
                }
            }
            HudItem::Sigmoid => {
                if self.apply_sigmoid { "ON".into() } else { "OFF".into() }
            }
            HudItem::NumThreads => {
                if self.num_threads == 0 { "all".into() } else { self.num_threads.to_string() }
            }
            HudItem::RenderBackend => {
                let name = self.backend.name();
                if self.detected_backend == Backend::Kitty {
                    name.to_string()
                } else {
                    format!("{} (only)", name)
                }
            }
            HudItem::PixelDensity => {
                if self.backend == Backend::Kitty {
                    format!("{:.0}%", self.pixel_density * 100.0)
                } else {
                    "n/a".into()
                }
            }
            HudItem::FovY => format!("{:.0}", self.fov_y_deg),
            HudItem::TranslateSpeed => format!("{:.3}", self.translate_speed),
            HudItem::RotationSpeed => format!("{:.2}", self.rotation_speed),
            HudItem::ZoomStep => format!("{:.2}", self.zoom_step),
            HudItem::Eps2d => format!("{:.2}", self.render_params.eps2d),
            HudItem::AlphaThreshold => format!("{:.4}", self.render_params.alpha_threshold),
            HudItem::ExtendSigma => format!("{:.1}", self.render_params.extend_sigma),
            HudItem::Saturation => format!("{:.3}", self.render_params.saturation),
        }
    }

    /// Append cursor-addressed ANSI escape lines to `out`.
    pub fn render(&self, camera: &OrbitCamera, di: &DisplayInfo, out: &mut String) {
        if !self.visible {
            return;
        }

        let mut row: usize = 1;

        // Title
        write_hud_line(out, row, "\x1b[1;97;40m", " [HUD] Tab to close");
        row += 1;

        for (i, entry) in ITEMS.iter().enumerate() {
            if let Some(group) = entry.group {
                let header = format!(" -- {} ", group);
                let padded = format!("{:-<width$}", header, width = HUD_WIDTH);
                write_hud_line(out, row, "\x1b[90;40m", &padded);
                row += 1;
            }

            let value = self.format_value(entry.item);
            let marker = if i == self.cursor { ">" } else { " " };
            let content = format!(" {} {:<12} {:>8}", marker, entry.label, value);
            let sgr = if i == self.cursor {
                "\x1b[30;107m"
            } else {
                "\x1b[97;40m"
            };
            write_hud_line(out, row, sgr, &content);
            row += 1;
        }

        // Display info (read-only)
        let header = format!("{:-<width$}", " -- Display Info ", width = HUD_WIDTH);
        write_hud_line(out, row, "\x1b[90;40m", &header);
        row += 1;

        let info_rows: &[(&str, String)] = &[
            ("resolution", format!("{} x {}", di.fb_w, di.fb_h)),
            ("cell size",  format!("{} x {} px", di.cell_size.w, di.cell_size.h)),
            ("terminal",   format!("{} x {} cells", di.cols, di.rows)),
            ("detected",   di.detected_backend.name().to_string()),
        ];
        for (label, value) in info_rows {
            let content = format!("   {:<12} {:>8}", label, value);
            write_hud_line(out, row, "\x1b[97;40m", &content);
            row += 1;
        }

        // Camera state (read-only)
        let header = format!("{:-<width$}", " -- Camera State ", width = HUD_WIDTH);
        write_hud_line(out, row, "\x1b[90;40m", &header);
        row += 1;

        let pos = camera.position();
        let cam_rows: &[(&str, String)] = &[
            ("yaw",    format!("{:>+8.3} rad", camera.yaw)),
            ("pitch",  format!("{:>+8.3} rad", camera.pitch)),
            ("radius", format!("{:>8.3}", camera.radius)),
            ("fov_y",  format!("{:>7.1} deg", camera.fov_y.to_degrees())),
            ("pos x",  format!("{:>+8.2}", pos.x)),
            ("pos y",  format!("{:>+8.2}", pos.y)),
            ("pos z",  format!("{:>+8.2}", pos.z)),
            ("tgt x",  format!("{:>+8.2}", camera.target.x)),
            ("tgt y",  format!("{:>+8.2}", camera.target.y)),
            ("tgt z",  format!("{:>+8.2}", camera.target.z)),
        ];
        for (label, value) in cam_rows {
            let content = format!("   {:<12} {:>8}", label, value);
            write_hud_line(out, row, "\x1b[97;40m", &content);
            row += 1;
        }
    }
}

/// Write a single HUD line at the given terminal row, padded to `HUD_WIDTH`.
fn write_hud_line(out: &mut String, row: usize, sgr: &str, text: &str) {
    let display: String = if text.len() >= HUD_WIDTH {
        text[..HUD_WIDTH].to_string()
    } else {
        format!("{:<width$}", text, width = HUD_WIDTH)
    };
    let _ = write!(out, "\x1b[{};1H{}{}\x1b[0m", row, sgr, display);
}
