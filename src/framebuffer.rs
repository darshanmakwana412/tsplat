use glam::Vec3;
use std::fmt::Write;

/// Render the RGB framebuffer into `out` as ANSI 24-bit half-block escapes.
///
/// Each terminal cell packs two vertically stacked pixels using the
/// U+2580 UPPER HALF BLOCK character, with the top pixel as foreground and
/// the bottom pixel as background. The whole frame is appended to `out` so
/// the caller can emit it in a single `write_all`.
///
/// `width` must equal the number of terminal columns and `height` must be
/// `2 * terminal_rows`.
pub fn render_halfblocks(fb: &[(Vec3, f32)], width: u32, height: u32, out: &mut String) {
    out.clear();
    // Home the cursor rather than clearing — clearing flickers.
    out.push_str("\x1b[H");

    let w = width as usize;
    let rows = (height / 2) as usize;

    for row in 0..rows {
        if row > 0 {
            out.push_str("\x1b[0m\r\n");
        }
        let top_base = 2 * row * w;
        let bot_base = (2 * row + 1) * w;
        for col in 0..w {
            let (tr, tg, tb) = to_u8(fb[top_base + col].0);
            let (br, bg, bb) = to_u8(fb[bot_base + col].0);
            // Truecolor fg + bg in a single SGR sequence, then half-block.
            write!(
                out,
                "\x1b[38;2;{};{};{};48;2;{};{};{}m\u{2580}",
                tr, tg, tb, br, bg, bb
            )
            .unwrap();
        }
    }
    out.push_str("\x1b[0m");
}

#[inline]
fn to_u8(c: Vec3) -> (u8, u8, u8) {
    let r = (c.x * 255.0).clamp(0.0, 255.0) as u8;
    let g = (c.y * 255.0).clamp(0.0, 255.0) as u8;
    let b = (c.z * 255.0).clamp(0.0, 255.0) as u8;
    (r, g, b)
}
