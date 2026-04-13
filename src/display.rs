use glam::Vec3;
use std::io::{self, Read, Write};
use std::time::Duration;

use crate::framebuffer;

const KITTY_IMG_Z_UNDER_UI: i32 = -1_073_741_825;

#[cfg(unix)]
fn open_tty_for_read() -> io::Result<std::fs::File> {
    std::fs::OpenOptions::new().read(true).open("/dev/tty")
}

#[cfg(unix)]
fn tty_poll_readable(fd: std::os::unix::io::RawFd, timeout_ms: i32) -> io::Result<bool> {
    let mut pfd = libc::pollfd {
        fd,
        events: libc::POLLIN,
        revents: 0,
    };
    let n = unsafe { libc::poll(&mut pfd, 1, timeout_ms) };
    if n < 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(n > 0 && (pfd.revents & libc::POLLIN) != 0)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    HalfBlock,
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

#[derive(Clone, Copy, Debug)]
pub struct CellSize {
    pub w: u32,
    pub h: u32,
}

impl Default for CellSize {
    fn default() -> Self {
        Self { w: 1, h: 2 }
    }
}

pub fn probe_kitty_support() -> bool {
    #[cfg(not(unix))]
    {
        return false;
    }

    #[cfg(unix)]
    {
        let mut tty_in = match open_tty_for_read() {
            Ok(f) => f,
            Err(_) => return false,
        };
        use std::os::unix::io::AsRawFd;
        let tty_fd = tty_in.as_raw_fd();

        let mut stdout = io::stdout().lock();

        let probe = b"\x1b_Gi=31,s=1,v=1,a=q,t=d,f=24;AAAA\x1b\\\x1b[c";
        if stdout.write_all(probe).is_err() || stdout.flush().is_err() {
            return false;
        }

        let mut buf = [0u8; 256];
        let mut filled = 0usize;
        let deadline = std::time::Instant::now() + Duration::from_millis(500);

        while std::time::Instant::now() < deadline && filled < buf.len() {
            if !tty_poll_readable(tty_fd, 50).unwrap_or(false) {
                continue;
            }
            match tty_in.read(&mut buf[filled..]) {
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

        false
    }
}

fn contains_kitty_ok(buf: &[u8]) -> bool {
    let st_backslash = b"\x1b_Gi=31;OK\x1b\\";
    let st_string_term = b"\x1b_Gi=31;OK\x9c";
    buf.windows(st_backslash.len()).any(|w| w == st_backslash)
        || buf
            .windows(st_string_term.len())
            .any(|w| w == st_string_term)
        || contains_subseq(buf, b"i=31;OK")
}

fn contains_subseq(haystack: &[u8], needle: &[u8]) -> bool {
    haystack
        .windows(needle.len())
        .any(|window| window == needle)
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

pub fn query_cell_size() -> Option<CellSize> {
    #[cfg(not(unix))]
    {
        return None;
    }

    #[cfg(unix)]
    {
        let mut tty_in = open_tty_for_read().ok()?;
        use std::os::unix::io::AsRawFd;
        let tty_fd = tty_in.as_raw_fd();

        let mut stdout = io::stdout().lock();

        let query = b"\x1b[16t\x1b[c";
        if stdout.write_all(query).is_err() || stdout.flush().is_err() {
            return None;
        }

        let mut buf = [0u8; 256];
        let mut filled = 0usize;
        let deadline = std::time::Instant::now() + Duration::from_millis(300);

        while std::time::Instant::now() < deadline && filled < buf.len() {
            if !tty_poll_readable(tty_fd, 50).unwrap_or(false) {
                continue;
            }
            match tty_in.read(&mut buf[filled..]) {
                Ok(0) => break,
                Ok(n) => {
                    filled += n;
                    if let Some(cs) = parse_cell_size_response(&buf[..filled]) {
                        let mut drain = [0u8; 256];
                        if tty_poll_readable(tty_fd, 20).unwrap_or(false) {
                            let _ = tty_in.read(&mut drain);
                        }
                        return Some(cs);
                    }
                }
                Err(_) => break,
            }
        }

        None
    }
}

fn parse_cell_size_response(buf: &[u8]) -> Option<CellSize> {
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

pub fn cell_size_from_ioctl(_cols: u32, _rows: u32) -> Option<CellSize> {
    let ws = crossterm::terminal::window_size().ok()?;
    let pw = ws.width as u32;
    let ph = ws.height as u32;
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

pub struct Display {
    pub backend: Backend,
    pub detected_backend: Backend,
    pub cols: u32,
    pub rows: u32,
    pub cell_size: CellSize,
    pub pixel_density: f32,
    frame: String,
    kitty_img_id: u32,
    pending_text_clear: bool,
}

impl Display {
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
            pending_text_clear: false,
        }
    }

    pub fn framebuffer_size(&self) -> (u32, u32) {
        match self.backend {
            Backend::HalfBlock => (self.cols, self.rows * 2),
            Backend::Kitty => {
                let w = ((self.cols * self.cell_size.w) as f32 * self.pixel_density) as u32;
                let h = ((self.rows * self.cell_size.h) as f32 * self.pixel_density) as u32;
                (w.max(1), h.max(1))
            }
        }
    }

    pub fn resize(&mut self, cols: u32, rows: u32) {
        self.cols = cols;
        self.rows = rows;
        if let Some(cs) = cell_size_from_ioctl(cols, rows) {
            self.cell_size = cs;
        }
    }

    pub fn render(&mut self, fb: &[(Vec3, f32)], width: u32, height: u32) {
        match self.backend {
            Backend::HalfBlock => {
                framebuffer::render_halfblocks(fb, width, height, &mut self.frame);
            }
            Backend::Kitty => {
                self.render_kitty(fb, width, height);
            }
        }
        if self.pending_text_clear {
            self.frame.insert_str(0, "\x1b[2J");
            self.pending_text_clear = false;
        }
    }

    pub fn overlay_string(&mut self) -> &mut String {
        &mut self.frame
    }

    pub fn queue_text_clear(&mut self) {
        self.pending_text_clear = true;
    }

    pub fn flush(&self) -> io::Result<()> {
        let mut stdout = io::stdout().lock();
        stdout.write_all(self.frame.as_bytes())?;
        stdout.flush()
    }

    pub fn kitty_cleanup(&mut self) {
        if self.backend == Backend::Kitty || self.detected_backend == Backend::Kitty {
            let mut stdout = io::stdout().lock();
            let _ =
                stdout.write_all(format!("\x1b_Ga=d,d=I,i={}\x1b\\", self.kitty_img_id).as_bytes());
            let _ = stdout.flush();
        }
    }

    fn render_kitty(&mut self, fb: &[(Vec3, f32)], width: u32, height: u32) {
        self.frame.clear();

        let pixel_count = (width * height) as usize;

        let mut rgb = Vec::with_capacity(pixel_count * 3);
        for &(color, _) in &fb[..pixel_count] {
            rgb.push((color.x * 255.0).clamp(0.0, 255.0) as u8);
            rgb.push((color.y * 255.0).clamp(0.0, 255.0) as u8);
            rgb.push((color.z * 255.0).clamp(0.0, 255.0) as u8);
        }

        self.frame.push_str("\x1b[H");

        let mut b64 = Vec::with_capacity(rgb.len() * 4 / 3 + 4);
        base64_encode_into(&rgb, &mut b64);

        const CHUNK: usize = 4096;
        let total = (b64.len() + CHUNK - 1) / CHUNK;

        for (ci, chunk) in b64.chunks(CHUNK).enumerate() {
            let is_last = ci == total - 1;
            let m = if is_last { 0 } else { 1 };

            if ci == 0 {
                use std::fmt::Write;
                let _ = write!(
                    self.frame,
                    "\x1b_Ga=T,f=24,t=d,s={},v={},c={},r={},i={},p=1,q=2,C=1,z={},m={};",
                    width, height, self.cols, self.rows, self.kitty_img_id, KITTY_IMG_Z_UNDER_UI, m,
                );
            } else {
                use std::fmt::Write;
                let _ = write!(self.frame, "\x1b_Gm={};", m);
            }
            unsafe {
                self.frame.as_mut_vec().extend_from_slice(chunk);
            }
            self.frame.push_str("\x1b\\");
        }
    }
}

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
