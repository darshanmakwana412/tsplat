use glam::Vec3;

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

pub fn render_halfblocks(fb: &[(Vec3, f32)], width: u32, height: u32, out: &mut String) {
    out.clear();

    let w = width as usize;
    let rows = (height / 2) as usize;

    let buf = unsafe { out.as_mut_vec() };
    buf.reserve(rows * w * 28 + 64);

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

            buf.extend_from_slice(b"\x1b[38;2;");
            push_u8_decimal(buf, tr);
            buf.push(b';');
            push_u8_decimal(buf, tg);
            buf.push(b';');
            push_u8_decimal(buf, tb);
            buf.extend_from_slice(b";48;2;");
            push_u8_decimal(buf, br);
            buf.push(b';');
            push_u8_decimal(buf, bg);
            buf.push(b';');
            push_u8_decimal(buf, bb);
            buf.push(b'm');
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
