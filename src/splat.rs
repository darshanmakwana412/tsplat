use anyhow::{Context, Result, anyhow};
use glam::{Quat, Vec3};
use ply_rs::parser::Parser;
use ply_rs::ply::{DefaultElement, Property};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use crate::sh::sh_band0_to_rgb;

/// One 3D Gaussian, fully decoded and ready to render.
///
/// All values are in their "real" (linear) form — log-space scales have been
/// exponentiated, the quaternion has been normalized, the band-0 SH DC has
/// been converted to RGB, and the raw opacity logit has been sigmoid'd.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Splat {
    pub pos: Vec3,
    pub scale: Vec3,
    pub rot: Quat,
    pub color: Vec3,
    pub opacity: f32,
}

fn get_f32(el: &DefaultElement, key: &str) -> Result<f32> {
    match el.get(key) {
        Some(Property::Float(v)) => Ok(*v),
        Some(Property::Double(v)) => Ok(*v as f32),
        Some(other) => Err(anyhow!("property {key} has unexpected type {other:?}")),
        None => Err(anyhow!("missing property {key}")),
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Load an INRIA 3DGS `.ply` file into `Vec<Splat>`.
///
/// `apply_sigmoid_opacity` should be `true` for vanilla INRIA `.ply` files
/// (opacity is stored as a pre-sigmoid logit). If the scene looks hazy /
/// ~50% transparent, flip this to `false`.
pub fn load_ply(path: &Path, apply_sigmoid_opacity: bool) -> Result<Vec<Splat>> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut reader = BufReader::new(file);
    let parser = Parser::<DefaultElement>::new();
    let ply = parser
        .read_ply(&mut reader)
        .with_context(|| format!("parsing {}", path.display()))?;

    let vertices = ply
        .payload
        .get("vertex")
        .ok_or_else(|| anyhow!("no 'vertex' element in {}", path.display()))?;

    let mut splats = Vec::with_capacity(vertices.len());
    for v in vertices {
        let x = get_f32(v, "x")?;
        let y = get_f32(v, "y")?;
        let z = get_f32(v, "z")?;

        let f0 = get_f32(v, "f_dc_0")?;
        let f1 = get_f32(v, "f_dc_1")?;
        let f2 = get_f32(v, "f_dc_2")?;

        // INRIA stores log-space scales.
        let s0 = get_f32(v, "scale_0")?;
        let s1 = get_f32(v, "scale_1")?;
        let s2 = get_f32(v, "scale_2")?;

        // INRIA stores rot in wxyz order. glam Quat::from_xyzw takes xyzw.
        let rw = get_f32(v, "rot_0")?;
        let rx = get_f32(v, "rot_1")?;
        let ry = get_f32(v, "rot_2")?;
        let rz = get_f32(v, "rot_3")?;

        let op_raw = get_f32(v, "opacity")?;

        let pos = Vec3::new(x, y, z);
        let scale = Vec3::new(s0.exp(), s1.exp(), s2.exp());
        let rot = Quat::from_xyzw(rx, ry, rz, rw).normalize();
        let color = sh_band0_to_rgb(Vec3::new(f0, f1, f2));
        let opacity = if apply_sigmoid_opacity {
            sigmoid(op_raw)
        } else {
            op_raw.clamp(0.0, 1.0)
        };

        splats.push(Splat {
            pos,
            scale,
            rot,
            color,
            opacity,
        });
    }
    Ok(splats)
}

/// Uniformly subsample a splat vector down to at most `max` entries by
/// striding. `max == 0` means "don't cap".
pub fn downsample_uniform(splats: Vec<Splat>, max: usize) -> Vec<Splat> {
    if max == 0 || splats.len() <= max {
        return splats;
    }
    let step = (splats.len() / max).max(1);
    splats.into_iter().step_by(step).take(max).collect()
}
