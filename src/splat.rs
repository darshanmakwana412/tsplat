use anyhow::{Context, Result, anyhow, bail};
use glam::{Quat, Vec3};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use crate::sh::sh_band0_to_rgb;

/// Tiny deterministic PRNG (splitmix64 → f32 in [0,1)).
/// We roll our own so synthetic scene generation stays dependency-free and
/// byte-identical across machines.
#[allow(dead_code)]
#[derive(Clone, Copy)]
pub struct Rng(u64);

#[allow(dead_code)]
impl Rng {
    pub fn new(seed: u64) -> Self {
        // Avoid the 0 fixed point — splitmix64 is fine with any nonzero state.
        Self(seed.wrapping_add(0x9E37_79B9_7F4A_7C15))
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform f32 in [0, 1).
    #[inline]
    pub fn f32(&mut self) -> f32 {
        // 24 random bits → f32 mantissa. This keeps the distribution exact.
        ((self.next_u64() >> 40) as u32 as f32) * (1.0 / (1u32 << 24) as f32)
    }

    /// Uniform f32 in [lo, hi).
    #[inline]
    pub fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.f32()
    }

    /// Box–Muller standard normal.
    #[inline]
    pub fn normal(&mut self) -> f32 {
        // Guard against log(0) without branching on the hot path.
        let u1 = self.f32().max(f32::MIN_POSITIVE);
        let u2 = self.f32();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}

/// Generate `n` random 3D Gaussians filling a cube of half-extent `bounds`,
/// using `seed` for determinism. Distribution is chosen to roughly match real
/// scenes: log-normal scales (most splats tiny, a few large), biased-low
/// opacity (most splats are faint), random colors and orientations.
///
/// Used by benchmarks and regression tests so we no longer depend on a real
/// .ply scene sitting on disk.
#[allow(dead_code)]
pub fn random_scene(n: usize, seed: u64, bounds: f32) -> Vec<Splat> {
    let mut rng = Rng::new(seed);
    let mut splats = Vec::with_capacity(n);
    for _ in 0..n {
        let pos = Vec3::new(
            rng.range(-bounds, bounds),
            rng.range(-bounds, bounds),
            rng.range(-bounds, bounds),
        );
        // Log-normal scales: exp(N(-3, 0.7)) ≈ most in [0.02, 0.15], tail to ~0.4.
        let scale = Vec3::new(
            (-3.0 + 0.7 * rng.normal()).exp(),
            (-3.0 + 0.7 * rng.normal()).exp(),
            (-3.0 + 0.7 * rng.normal()).exp(),
        );
        // Random unit quaternion via Marsaglia's method.
        let (x, y, z, w) = {
            let mut u1;
            let mut u2;
            let mut s1;
            loop {
                u1 = rng.range(-1.0, 1.0);
                u2 = rng.range(-1.0, 1.0);
                s1 = u1 * u1 + u2 * u2;
                if s1 < 1.0 { break; }
            }
            let mut u3;
            let mut u4;
            let mut s2;
            loop {
                u3 = rng.range(-1.0, 1.0);
                u4 = rng.range(-1.0, 1.0);
                s2 = u3 * u3 + u4 * u4;
                if s2 < 1.0 { break; }
            }
            let f = ((1.0 - s1) / s2).sqrt();
            (u1, u2, u3 * f, u4 * f)
        };
        let rot = Quat::from_xyzw(x, y, z, w).normalize();
        let color = Vec3::new(rng.f32(), rng.f32(), rng.f32());
        // Biased-low opacity: square of a uniform pushes density toward 0.
        // Matches FlashGS Fig. 6 observation that most real gaussians are faint.
        let u = rng.f32();
        let opacity = (u * u).clamp(0.01, 0.99);
        splats.push(Splat { pos, scale, rot, color, opacity });
    }
    splats
}

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
///
/// Returns `(splats, total_vertex_count)` where `total_vertex_count` is the
/// full count declared in the PLY header (used by the HUD to cap the
/// `max_splats` slider to the scene's real size rather than a magic ceiling).
pub fn load_ply(
    path: &Path,
    apply_sigmoid_opacity: bool,
    max_splats: usize,
) -> Result<(Vec<Splat>, usize)> {
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

    Ok((splats, vertex_count))
}
