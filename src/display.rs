use glam::Vec3;
use std::io::{self, Read, Write};
use std::time::Duration;

use crate::framebuffer;

// ── Backend enum ────────────────────────────────────────────────────────────

/// Which rendering backend we're using to get pixels on screen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// Unicode half-block characters with 24-bit SGR color.
    /// Each terminal cell = 1 column wide × 2 pixels tall.
    HalfBlock,
    /// Kitty graphics protocol: raw RGB pixel data sent via APC escape.
    /// Each terminal cell = cell_w × cell_h real pixels.
    Kitty,
}

impl Backend {
    pub fn name(self) -> &'static str {
        match self {
            Backend::HalfBlock => "halfblock",
            Backend::Kitty => "kitty",
        }
    }
}

// ── Cell pixel size ─────────────────────────────────────────────────────────

/// Pixel dimensions of a single terminal cell.
#[derive(Clone, Copy, Debug)]
pub struct CellSize {
    pub w: u32,
    pub h: u32,
}

impl Default for CellSize {
    fn default() -> Self {
        // Conservative default: half-block assumption (1×2).
        Self { w: 1, h: 2 }
    }
}

// ── Detection ───────────────────────────────────────────────────────────────

/// Probe whether the terminal supports the Kitty graphics protocol.
/// Must be called while the terminal is in raw mode.
pub fn probe_kitty_support() -> bool {
    let mut stdout = io::stdout().lock();

    // Send graphics query (a=q) with a 1×1 RGB pixel, plus DA1 as a fence.
    let probe = b"\x1b_Gi=31,s=1,v=1,a=q,t=d,f=24;AAAA\x1b\\\x1b[c";
    if stdout.write_all(probe).is_err() || stdout.flush().is_err() {
        return false;
    }

    let mut buf = [0u8; 256];
    let mut filled = 0usize;
    let deadline = std::time::Instant::now() + Duration::from_millis(500);

    let stdin = io::stdin();
    let mut handle = stdin.lock();

    while std::time::Instant::now() < deadline && filled < buf.len() {
        if crossterm::event::poll(Duration::from_millis(50)).unwrap_or(false) {
            match handle.read(&mut buf[filled..]) {
                Ok(0) => break,
                Ok(n) => {
                    filled += n;
                    let so_far = &buf[..filled];
                    if contains_kitty_ok(so_far) {
                        return true;
                    }
                    if contains_da1_response(so_far) {
                        return false;
                    }
                }
                Err(_) => break,
            }
        }
    }

    false
}

fn contains_kitty_ok(buf: &[u8]) -> bool {
    let pattern = b"\x1b_Gi=31;OK\x1b\\";
    buf.windows(pattern.len()).any(|w| w == pattern)
}

fn contains_da1_response(buf: &[u8]) -> bool {
    for i in 0..buf.len().saturating_sub(3) {
        if buf[i] == 0x1b && buf[i + 1] == b'[' && buf[i + 2] == b'?' {
            for j in (i + 3)..buf.len() {
                if buf[j] == b'c' {
                    return true;
                }
            }
        }
    }
    false
}

/// Query terminal cell pixel size via CSI 16 t.
pub fn query_cell_size() -> Option<CellSize> {
    let mut stdout = io::stdout().lock();

    // CSI 16 t → response: CSI 6 ; height ; width t
    let query = b"\x1b[16t\x1b[c";
    if stdout.write_all(query).is_err() || stdout.flush().is_err() {
        return None;
    }

    let mut buf = [0u8; 128];
    let mut filled = 0usize;
    let deadline = std::time::Instant::now() + Duration::from_millis(300);

    let stdin = io::stdin();
    let mut handle = stdin.lock();

    while std::time::Instant::now() < deadline && filled < buf.len() {
        if crossterm::event::poll(Duration::from_millis(50)).unwrap_or(false) {
            match handle.read(&mut buf[filled..]) {
                Ok(0) => break,
                Ok(n) => {
                    filled += n;
                    if let Some(cs) = parse_cell_size_response(&buf[..filled]) {
                        return Some(cs);
                    }
                    if contains_da1_response(&buf[..filled]) {
                        return None;
                    }
                }
                Err(_) => break,
            }
        }
    }

    None
}

fn parse_cell_size_response(buf: &[u8]) -> Option<CellSize> {
    // Look for \x1b [ 6 ; H ; W t
    for i in 0..buf.len().saturating_sub(6) {
        if buf[i] == 0x1b && buf[i + 1] == b'[' && buf[i + 2] == b'6' && buf[i + 3] == b';' {
            let rest = &buf[i + 4..];
            if let Some(t_pos) = rest.iter().position(|&b| b == b't') {
                let params = &rest[..t_pos];
                let s = std::str::from_utf8(params).ok()?;
                let mut parts = s.split(';');
                let h: u32 = parts.next()?.parse().ok()?;
                let w: u32 = parts.next()?.parse().ok()?;
                if w > 0 && h > 0 {
                    return Some(CellSize { w, h });
                }
            }
        }
    }
    None
}

/// Try to get cell pixel size from the ioctl-based window_size().
pub fn cell_size_from_ioctl(_cols: u32, _rows: u32) -> Option<CellSize> {
    let ws = crossterm::terminal::window_size().ok()?;
    let pw = ws.width as u32;   // total pixel width
    let ph = ws.height as u32;  // total pixel height
    let cols = ws.columns as u32;
    let rows = ws.rows as u32;
    if pw == 0 || ph == 0 || cols == 0 || rows == 0 {
        return None;
    }
    Some(CellSize {
        w: pw / cols,
        h: ph / rows,
    })
}

// ── Display ─────────────────────────────────────────────────────────────────

/// Manages the rendering backend, resolution, and output buffer.
///
/// The render pipeline is:
///   1. `display.framebuffer_size()` → tells the rasterizer how many pixels
///   2. Rasterizer fills the framebuffer
///   3. `display.render(fb, w, h)` → converts pixels to the terminal format
///   4. Caller appends ANSI overlays (FPS, HUD) via `display.overlay_string()`
///   5. `display.flush()` → single write_all to stdout
pub struct Display {
    pub backend: Backend,
    pub detected_backend: Backend,
    pub cols: u32,
    pub rows: u32,
    pub cell_size: CellSize,
    /// Pixel density multiplier (0.25 .. 1.0). 1.0 = full native resolution.
    /// Only affects Kitty backend. HalfBlock always uses 1 col × 2 rows.
    pub pixel_density: f32,
    /// The frame output buffer. Contains the pixel data (halfblock escapes or
    /// kitty APC sequences). ANSI overlays (FPS, HUD) are appended after render().
    frame: String,
    /// Kitty image id, incremented to avoid stale image artifacts.
    kitty_img_id: u32,
}

impl Display {
    /// Create a new Display. Call after TerminalGuard::new() so raw mode is active.
    pub fn new(cols: u32, rows: u32) -> Self {
        let kitty_supported = probe_kitty_support();

        let cell_size = query_cell_size()
            .or_else(|| cell_size_from_ioctl(cols, rows))
            .unwrap_or_default();

        let detected = if kitty_supported {
            Backend::Kitty
        } else {
            Backend::HalfBlock
        };

        Self {
            backend: detected,
            detected_backend: detected,
            cols,
            rows,
            cell_size,
            pixel_density: 1.0,
            frame: String::with_capacity(512 * 1024),
            kitty_img_id: 1000,
        }
    }

    /// Pixel dimensions of the framebuffer for the current backend + density.
    pub fn framebuffer_size(&self) -> (u32, u32) {
        match self.backend {
            Backend::HalfBlock => {
                (self.cols, self.rows * 2)
            }
            Backend::Kitty => {
                let w = ((self.cols * self.cell_size.w) as f32 * self.pixel_density) as u32;
                let h = ((self.rows * self.cell_size.h) as f32 * self.pixel_density) as u32;
                (w.max(1), h.max(1))
            }
        }
    }

    /// Handle a terminal resize.
    pub fn resize(&mut self, cols: u32, rows: u32) {
        self.cols = cols;
        self.rows = rows;
        if let Some(cs) = cell_size_from_ioctl(cols, rows) {
            self.cell_size = cs;
        }
    }

    /// Render the framebuffer into the internal frame buffer.
    /// After this call, use `overlay_string()` to append HUD/FPS text,
    /// then call `flush()`.
    pub fn render(&mut self, fb: &[(Vec3, f32)], width: u32, height: u32) {
        match self.backend {
            Backend::HalfBlock => {
                framebuffer::render_halfblocks(fb, width, height, &mut self.frame);
            }
            Backend::Kitty => {
                self.render_kitty(fb, width, height);
            }
        }
    }

    /// Returns a mutable reference to the frame string so the caller can
    /// append cursor-addressed ANSI overlays (FPS counter, HUD panel).
    /// These will be rendered on top of both backends since they use
    /// absolute cursor positioning.
    pub fn overlay_string(&mut self) -> &mut String {
        &mut self.frame
    }

    /// Write the complete frame to stdout in one shot.
    pub fn flush(&self) -> io::Result<()> {
        let mut stdout = io::stdout().lock();
        stdout.write_all(self.frame.as_bytes())?;
        stdout.flush()
    }

    /// Clean up kitty state (delete images) before switching backends or exiting.
    pub fn kitty_cleanup(&mut self) {
        if self.backend == Backend::Kitty || self.detected_backend == Backend::Kitty {
            let mut stdout = io::stdout().lock();
            // Delete all images placed by us.
            let _ = stdout.write_all(
                format!("\x1b_Ga=d,d=I,i={}\x1b\\", self.kitty_img_id).as_bytes(),
            );
            let _ = stdout.flush();
        }
    }

    // ── Kitty rendering ─────────────────────────────────────────────────

    fn render_kitty(&mut self, fb: &[(Vec3, f32)], width: u32, height: u32) {
        self.frame.clear();

        let pixel_count = (width * height) as usize;

        // Build raw RGB bytes.
        let mut rgb = Vec::with_capacity(pixel_count * 3);
        for &(color, _) in &fb[..pixel_count] {
            rgb.push((color.x * 255.0).clamp(0.0, 255.0) as u8);
            rgb.push((color.y * 255.0).clamp(0.0, 255.0) as u8);
            rgb.push((color.z * 255.0).clamp(0.0, 255.0) as u8);
        }

        // Home cursor so the image starts at top-left.
        self.frame.push_str("\x1b[H");

        // Base64 encode.
        let mut b64 = Vec::with_capacity(rgb.len() * 4 / 3 + 4);
        base64_encode_into(&rgb, &mut b64);

        // Transmit in chunks.
        const CHUNK: usize = 4096;
        let total = (b64.len() + CHUNK - 1) / CHUNK;

        for (ci, chunk) in b64.chunks(CHUNK).enumerate() {
            let is_last = ci == total - 1;
            let m = if is_last { 0 } else { 1 };

            if ci == 0 {
                // First chunk with full metadata.
                // a=T transmit+display, f=24 RGB, t=d direct,
                // s/v pixel dims, c/r cell dims for scaling,
                // i=id p=1 for flicker-free replacement, q=2 suppress response.
                use std::fmt::Write;
                let _ = write!(
                    self.frame,
                    "\x1b_Ga=T,f=24,t=d,s={},v={},c={},r={},i={},p=1,q=2,m={};",
                    width, height, self.cols, self.rows, self.kitty_img_id, m,
                );
            } else {
                use std::fmt::Write;
                let _ = write!(self.frame, "\x1b_Gm={};", m);
            }
            // Append base64 chunk (it's ASCII, safe to push as str).
            // SAFETY: base64 output is always valid ASCII.
            unsafe {
                self.frame
                    .as_mut_vec()
                    .extend_from_slice(chunk);
            }
            self.frame.push_str("\x1b\\");
        }
    }
}

// ── Base64 encoder ──────────────────────────────────────────────────────────

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn base64_encode_into(data: &[u8], out: &mut Vec<u8>) {
    let mut i = 0;
    let len = data.len();
    while i + 2 < len {
        let b0 = data[i] as u32;
        let b1 = data[i + 1] as u32;
        let b2 = data[i + 2] as u32;
        let triple = (b0 << 16) | (b1 << 8) | b2;
        out.push(B64[((triple >> 18) & 0x3F) as usize]);
        out.push(B64[((triple >> 12) & 0x3F) as usize]);
        out.push(B64[((triple >> 6) & 0x3F) as usize]);
        out.push(B64[(triple & 0x3F) as usize]);
        i += 3;
    }
    let remaining = len - i;
    if remaining == 1 {
        let b0 = data[i] as u32;
        out.push(B64[(b0 >> 2) as usize]);
        out.push(B64[((b0 & 0x03) << 4) as usize]);
        out.push(b'=');
        out.push(b'=');
    } else if remaining == 2 {
        let b0 = data[i] as u32;
        let b1 = data[i + 1] as u32;
        out.push(B64[(b0 >> 2) as usize]);
        out.push(B64[(((b0 & 0x03) << 4) | (b1 >> 4)) as usize]);
        out.push(B64[((b1 & 0x0F) << 2) as usize]);
        out.push(b'=');
    }
}
