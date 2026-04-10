use glam::Vec3;

/// Pre-computed lookup table: decimal ASCII for 0..=255 as 1-3 bytes.
/// Avoids per-byte itoa formatting in the hot loop.
static BYTE_STRINGS: [[u8; 3]; 256] = {
    let mut table = [[0u8; 3]; 256];
    let mut i: u16 = 0;
    while i < 256 {
        let v = i as u8;
        if v >= 100 {
            table[i as usize] = [b'0' + v / 100, b'0' + (v / 10) % 10, b'0' + v % 10];
        } else if v >= 10 {
            table[i as usize] = [0, b'0' + v / 10, b'0' + v % 10];
        } else {
            table[i as usize] = [0, 0, b'0' + v];
        }
        i += 1;
    }
    table
};

/// Push the decimal ASCII for a u8 value into the byte buffer.
#[inline(always)]
fn push_u8_decimal(buf: &mut Vec<u8>, v: u8) {
    let entry = &BYTE_STRINGS[v as usize];
    if v >= 100 {
        buf.push(entry[0]);
        buf.push(entry[1]);
        buf.push(entry[2]);
    } else if v >= 10 {
        buf.push(entry[1]);
        buf.push(entry[2]);
    } else {
        buf.push(entry[2]);
    }
}

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

    let w = width as usize;
    let rows = (height / 2) as usize;

    // Work directly with the underlying byte buffer for speed.
    // SAFETY: we only push valid UTF-8 sequences (ASCII + the known U+2580).
    let buf = unsafe { out.as_mut_vec() };
    buf.reserve(rows * w * 28 + 64); // ~28 bytes per cell estimate

    // Home cursor: \x1b[H
    buf.extend_from_slice(b"\x1b[H");

    for row in 0..rows {
        if row > 0 {
            buf.extend_from_slice(b"\x1b[0m\r\n");
        }
        let top_base = 2 * row * w;
        let bot_base = (2 * row + 1) * w;
        for col in 0..w {
            let (tr, tg, tb) = to_u8(fb[top_base + col].0);
            let (br, bg, bb) = to_u8(fb[bot_base + col].0);

            // \x1b[38;2;R;G;B;48;2;R;G;Bm▀
            buf.extend_from_slice(b"\x1b[38;2;");
            push_u8_decimal(buf, tr); buf.push(b';');
            push_u8_decimal(buf, tg); buf.push(b';');
            push_u8_decimal(buf, tb);
            buf.extend_from_slice(b";48;2;");
            push_u8_decimal(buf, br); buf.push(b';');
            push_u8_decimal(buf, bg); buf.push(b';');
            push_u8_decimal(buf, bb);
            buf.push(b'm');
            // U+2580 UPPER HALF BLOCK = UTF-8: E2 96 80
            buf.extend_from_slice(b"\xe2\x96\x80");
        }
    }
    buf.extend_from_slice(b"\x1b[0m");
}

#[inline]
fn to_u8(c: Vec3) -> (u8, u8, u8) {
    let r = (c.x * 255.0).clamp(0.0, 255.0) as u8;
    let g = (c.y * 255.0).clamp(0.0, 255.0) as u8;
    let b = (c.z * 255.0).clamp(0.0, 255.0) as u8;
    (r, g, b)
}
