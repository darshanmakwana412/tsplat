use anyhow::{Context, Result, anyhow, bail};
use glam::{Quat, Vec3};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use crate::sh::sh_band0_to_rgb;

/// One 3D Gaussian, fully decoded and ready to render.
///
/// All values are already in their linear form — log-space scales have been
/// exponentiated, the quaternion normalised, SH band-0 converted to RGB, and
/// the raw opacity logit has been sigmoid'd.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Splat {
    pub pos: Vec3,
    pub scale: Vec3,
    pub rot: Quat,
    pub color: Vec3,
    pub opacity: f32,
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Load an INRIA 3DGS `.ply` file into at most `max_splats` `Splat`s.
///
/// Subsampling is done during the read — we never hold more than `max_splats`
/// decoded records in memory at once.  `max_splats == 0` means "load all".
///
/// `apply_sigmoid_opacity`: `true` for vanilla INRIA files (opacity stored as
/// a pre-sigmoid logit).  Pass `false` if the scene looks uniformly hazy.
pub fn load_ply(path: &Path, apply_sigmoid_opacity: bool, max_splats: usize) -> Result<Vec<Splat>> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    // Large read buffer — the binary body can be hundreds of MB.
    let mut reader = BufReader::with_capacity(4 * 1024 * 1024, file);

    // ---- Parse ASCII header -----------------------------------------------
    // The header is a sequence of ASCII lines terminated by "end_header\n".
    let mut vertex_count: usize = 0;
    let mut prop_names: Vec<String> = Vec::new();
    let mut in_vertex = false;

    loop {
        let mut line = String::new();
        reader
            .read_line(&mut line)
            .context("reading PLY header")?;
        if line.is_empty() {
            bail!("unexpected EOF inside PLY header");
        }
        let t = line.trim_end();

        if t == "end_header" {
            break;
        }
        if t.starts_with("format") {
            // We only handle binary_little_endian.
            if !t.contains("binary_little_endian") {
                bail!("only binary_little_endian PLY is supported (got: {t})");
            }
        } else if t.starts_with("element vertex") {
            in_vertex = true;
            vertex_count = t
                .split_whitespace()
                .last()
                .and_then(|s| s.parse().ok())
                .with_context(|| format!("cannot parse vertex count from: {t}"))?;
        } else if t.starts_with("element") {
            // Some other element (e.g. face) — stop collecting properties.
            in_vertex = false;
        } else if in_vertex && t.starts_with("property float") {
            let name = t
                .split_whitespace()
                .nth(2)
                .ok_or_else(|| anyhow!("property line has no name: {t}"))?
                .to_string();
            prop_names.push(name);
        }
        // Ignore "property double", "property int", comments, obj_info, etc.
        // INRIA 3DGS PLY files use only float32 for all vertex properties.
    }

    if vertex_count == 0 {
        bail!("PLY has no vertex element or vertex count is 0");
    }

    // Each property is a 4-byte float32 → fixed stride.
    let stride = prop_names.len() * 4;

    // Build name → byte-offset map.
    let offsets: HashMap<&str, usize> = prop_names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_str(), i * 4))
        .collect();

    let get = |key: &str| -> Result<usize> {
        offsets
            .get(key)
            .copied()
            .with_context(|| format!("PLY is missing required property '{key}'"))
    };

    let off_x  = get("x")?;
    let off_y  = get("y")?;
    let off_z  = get("z")?;
    let off_f0 = get("f_dc_0")?;
    let off_f1 = get("f_dc_1")?;
    let off_f2 = get("f_dc_2")?;
    let off_s0 = get("scale_0")?;
    let off_s1 = get("scale_1")?;
    let off_s2 = get("scale_2")?;
    let off_rw = get("rot_0")?; // INRIA: rot_0 = w component
    let off_rx = get("rot_1")?;
    let off_ry = get("rot_2")?;
    let off_rz = get("rot_3")?;
    let off_op = get("opacity")?;

    // ---- Compute subsampling stride ---------------------------------------
    let (step, capacity) = if max_splats == 0 || vertex_count <= max_splats {
        (1usize, vertex_count)
    } else {
        (vertex_count / max_splats, max_splats)
    };

    let mut splats = Vec::with_capacity(capacity);

    // ---- Stream the binary body ------------------------------------------
    // Read one fixed-size vertex record at a time; decode only the fields we
    // need; skip records that fall outside the uniform subsample stride.
    let mut record = vec![0u8; stride];

    for i in 0..vertex_count {
        reader
            .read_exact(&mut record)
            .with_context(|| format!("reading vertex {i}"))?;

        if i % step != 0 {
            continue;
        }
        if max_splats > 0 && splats.len() >= max_splats {
            break;
        }

        let f = |off: usize| f32::from_le_bytes(record[off..off + 4].try_into().unwrap());

        let pos   = Vec3::new(f(off_x), f(off_y), f(off_z));
        let scale = Vec3::new(f(off_s0).exp(), f(off_s1).exp(), f(off_s2).exp());
        // INRIA quaternion order is wxyz; glam Quat::from_xyzw wants xyzw.
        let rot   = Quat::from_xyzw(f(off_rx), f(off_ry), f(off_rz), f(off_rw)).normalize();
        let color = sh_band0_to_rgb(Vec3::new(f(off_f0), f(off_f1), f(off_f2)));
        let opacity = if apply_sigmoid_opacity {
            sigmoid(f(off_op))
        } else {
            f(off_op).clamp(0.0, 1.0)
        };

        splats.push(Splat { pos, scale, rot, color, opacity });
    }

    Ok(splats)
}
