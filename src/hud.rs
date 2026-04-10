use std::fmt::Write;

use crossterm::event::KeyCode;

use crate::camera::OrbitCamera;
use crate::rasterize::RenderParams;

/// What happened after the HUD processed a key.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum HudAction {
    /// Nothing changed.
    None,
    /// A real-time parameter changed (camera or render). No reload needed.
    ValueChanged,
    /// A parameter that requires a PLY reload changed (max_splats or sigmoid).
    ReloadSplats,
}

/// Identity of each tunable item in the HUD.
#[derive(Clone, Copy)]
enum HudItem {
    MaxSplats,
    Sigmoid,
    FovY,
    YawStep,
    PitchStep,
    ZoomStep,
    Eps2d,
    AlphaThreshold,
    ExtendSigma,
    Saturation,
}

/// An entry in the flat display list. `group` is `Some("header")` for the
/// first item in a group (renders a header row above it), `None` otherwise.
struct Entry {
    group: Option<&'static str>,
    item: HudItem,
    label: &'static str,
}

const ITEMS: &[Entry] = &[
    Entry { group: Some("Splats"),  item: HudItem::MaxSplats,       label: "max_splats" },
    Entry { group: None,            item: HudItem::Sigmoid,         label: "sigmoid" },
    Entry { group: Some("Camera"),  item: HudItem::FovY,            label: "fov_y" },
    Entry { group: None,            item: HudItem::YawStep,         label: "orbit spd" },
    Entry { group: None,            item: HudItem::PitchStep,       label: "pitch spd" },
    Entry { group: None,            item: HudItem::ZoomStep,        label: "zoom spd" },
    Entry { group: Some("Render"),  item: HudItem::Eps2d,           label: "eps2d" },
    Entry { group: None,            item: HudItem::AlphaThreshold,  label: "alpha thr" },
    Entry { group: None,            item: HudItem::ExtendSigma,     label: "sigma ext" },
    Entry { group: None,            item: HudItem::Saturation,      label: "saturation" },
];

/// Width of the HUD panel in characters.
const HUD_WIDTH: usize = 32;

pub struct HudState {
    pub visible: bool,
    cursor: usize,

    // -- Reload parameters --
    pub max_splats: usize,
    pub apply_sigmoid: bool,

    // -- Camera parameters (real-time) --
    pub fov_y_deg: f32,
    pub yaw_step: f32,
    pub pitch_step: f32,
    pub zoom_step: f32,

    // -- Render parameters (real-time) --
    pub render_params: RenderParams,
}

impl HudState {
    pub fn new(max_splats: usize, apply_sigmoid: bool, fov_y_rad: f32) -> Self {
        Self {
            visible: false,
            cursor: 0,
            max_splats,
            apply_sigmoid,
            fov_y_deg: fov_y_rad.to_degrees(),
            yaw_step: 0.08,
            pitch_step: 0.06,
            zoom_step: 0.1,
            render_params: RenderParams::default(),
        }
    }

    pub fn toggle(&mut self) {
        self.visible = !self.visible;
    }

    /// Process an arrow key while the HUD is visible.
    /// Returns what kind of change happened (if any).
    pub fn handle_key(&mut self, code: KeyCode) -> HudAction {
        match code {
            KeyCode::Up => {
                if self.cursor == 0 {
                    self.cursor = ITEMS.len() - 1;
                } else {
                    self.cursor -= 1;
                }
                HudAction::None
            }
            KeyCode::Down => {
                self.cursor = (self.cursor + 1) % ITEMS.len();
                HudAction::None
            }
            KeyCode::Left => self.adjust(-1),
            KeyCode::Right => self.adjust(1),
            _ => HudAction::None,
        }
    }

    fn adjust(&mut self, dir: i8) -> HudAction {
        let item = ITEMS[self.cursor].item;
        match item {
            HudItem::MaxSplats => {
                self.max_splats = if dir > 0 {
                    (self.max_splats * 2).min(10_000_000)
                } else {
                    (self.max_splats / 2).max(1_000)
                };
                HudAction::ReloadSplats
            }
            HudItem::Sigmoid => {
                self.apply_sigmoid = !self.apply_sigmoid;
                HudAction::ReloadSplats
            }
            HudItem::FovY => {
                self.fov_y_deg = (self.fov_y_deg + dir as f32 * 5.0).clamp(10.0, 150.0);
                HudAction::ValueChanged
            }
            HudItem::YawStep => {
                self.yaw_step = (self.yaw_step + dir as f32 * 0.01).clamp(0.01, 0.50);
                HudAction::ValueChanged
            }
            HudItem::PitchStep => {
                self.pitch_step = (self.pitch_step + dir as f32 * 0.01).clamp(0.01, 0.50);
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
                if self.max_splats >= 1_000_000 {
                    format!("{}M", self.max_splats / 1_000_000)
                } else {
                    format!("{}k", self.max_splats / 1_000)
                }
            }
            HudItem::Sigmoid => {
                if self.apply_sigmoid { "ON".into() } else { "OFF".into() }
            }
            HudItem::FovY => format!("{:.0}", self.fov_y_deg),
            HudItem::YawStep => format!("{:.2}", self.yaw_step),
            HudItem::PitchStep => format!("{:.2}", self.pitch_step),
            HudItem::ZoomStep => format!("{:.2}", self.zoom_step),
            HudItem::Eps2d => format!("{:.2}", self.render_params.eps2d),
            HudItem::AlphaThreshold => format!("{:.4}", self.render_params.alpha_threshold),
            HudItem::ExtendSigma => format!("{:.1}", self.render_params.extend_sigma),
            HudItem::Saturation => format!("{:.3}", self.render_params.saturation),
        }
    }

    /// Append cursor-addressed ANSI escape lines to `out`.
    pub fn render(&self, camera: &OrbitCamera, out: &mut String) {
        if !self.visible {
            return;
        }

        let mut row: usize = 1; // 1-indexed terminal rows

        // Title
        write_hud_line(out, row, "\x1b[1;97;40m", " [HUD] Tab to close");
        row += 1;

        for (i, entry) in ITEMS.iter().enumerate() {
            // Group header
            if let Some(group) = entry.group {
                let header = format!(" -- {} ", group);
                let padded = format!("{:-<width$}", header, width = HUD_WIDTH);
                write_hud_line(out, row, "\x1b[90;40m", &padded);
                row += 1;
            }

            // Item row
            let value = self.format_value(entry.item);
            let marker = if i == self.cursor { ">" } else { " " };
            let content = format!(" {} {:<12} {:>8}", marker, entry.label, value);
            let sgr = if i == self.cursor {
                "\x1b[30;107m" // black on bright white (selected)
            } else {
                "\x1b[97;40m" // white on black
            };
            write_hud_line(out, row, sgr, &content);
            row += 1;
        }

        // Camera state (read-only)
        let header = format!("{:-<width$}", " -- Camera State ", width = HUD_WIDTH);
        write_hud_line(out, row, "\x1b[90;40m", &header);
        row += 1;

        let pos = camera.position();
        let rows: &[(&str, String)] = &[
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
        for (label, value) in rows {
            let content = format!("   {:<12} {:>8}", label, value);
            write_hud_line(out, row, "\x1b[97;40m", &content);
            row += 1;
        }
    }
}

/// Write a single HUD line at the given terminal row, padded to `HUD_WIDTH`.
fn write_hud_line(out: &mut String, row: usize, sgr: &str, text: &str) {
    // Truncate or pad to fixed width
    let display: String = if text.len() >= HUD_WIDTH {
        text[..HUD_WIDTH].to_string()
    } else {
        format!("{:<width$}", text, width = HUD_WIDTH)
    };
    let _ = write!(out, "\x1b[{};1H{}{}\x1b[0m", row, sgr, display);
}
