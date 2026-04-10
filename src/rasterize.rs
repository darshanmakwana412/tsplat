use glam::{Mat2, Mat3, Vec2, Vec3, Vec4};
use rayon::prelude::*;

use crate::camera::OrbitCamera;
use crate::splat::Splat;

/// Reusable per-frame scratch buffers to avoid heap allocation on every frame.
pub struct ScratchBuffers {
    /// Aux buffer for radix sort ping-pong.
    pub sort_aux: Vec<Projected>,
}

impl ScratchBuffers {
    pub fn new() -> Self {
        Self {
            sort_aux: Vec::new(),
        }
    }
}

/// Fast approximate exp(x) for x in [-87, 0]. Uses the Schraudolph trick:
/// reinterpret a scaled+biased float as an IEEE 754 bit pattern.
/// Accuracy: ~1-2% relative error, more than sufficient for Gaussian alpha.
#[inline(always)]
fn fast_exp(x: f32) -> f32 {
    // Clamp to avoid underflow/overflow in the integer cast.
    let x = x.max(-87.0);
    // Magic: 2^23 / ln(2) ≈ 12102203.16, bias = 127 * 2^23 = 1065353216
    let v = (12102203.0f32 * x + 1065353216.0) as i32;
    f32::from_bits(v as u32)
}

/// Per-splat intermediate produced during projection. Everything a pixel loop
/// needs is baked in here — there's no reason to revisit the original `Splat`
/// struct during composite.
#[derive(Clone, Copy)]
pub struct Projected {
    pub screen: Vec2,
    pub depth: f32,
    pub cov2d_inv: Mat2,
    pub bbox: [i32; 4], // inclusive: x0, y0, x1, y1
    pub color: Vec3,
    pub opacity: f32,
}

/// Runtime-tunable rendering parameters. Defaults match the original constants.
#[derive(Clone, Copy, Debug)]
pub struct RenderParams {
    /// Low-pass dilation on the 2D covariance diagonal (LichtFeld `eps2d`).
    pub eps2d: f32,
    /// Minimum effective alpha to bother compositing (LichtFeld `ALPHA_THRESHOLD`).
    pub alpha_threshold: f32,
    /// Bbox extent in stddevs (3σ ≈ 99.7% of Gaussian mass).
    pub extend_sigma: f32,
    /// Accumulated alpha at which a pixel is considered opaque.
    pub saturation: f32,
}

impl Default for RenderParams {
    fn default() -> Self {
        Self {
            eps2d: 0.3,
            alpha_threshold: 1.0 / 255.0,
            extend_sigma: 3.0,
            saturation: 0.999,
        }
    }
}

/// Project every splat. Parallel over splats (embarrassingly parallel).
/// Pass a pre-built thread pool to control parallelism, or None for rayon's
/// global pool.
pub fn project(
    splats: &[Splat],
    camera: &OrbitCamera,
    params: &RenderParams,
    pool: &Option<rayon::ThreadPool>,
) -> Vec<Projected> {
    let view = camera.view();
    let w_mat = Mat3::from_mat4(view);
    let (fx, fy, cx, cy) = camera.intrinsics();
    let w_i = camera.width as i32;
    let h_i = camera.height as i32;
    let znear = camera.znear;
    let zfar = camera.zfar;
    let w_mat_t = w_mat.transpose();
    let eps2d = params.eps2d;
    let extend_sigma = params.extend_sigma;

    let do_project = || {
        splats
            .par_iter()
            .filter_map(|s| {
                // ---- View-transform center ----
                let p_view4 = view * Vec4::new(s.pos.x, s.pos.y, s.pos.z, 1.0);
                let p_view = Vec3::new(p_view4.x, p_view4.y, p_view4.z);
                if p_view.z > -znear || p_view.z < -zfar {
                    return None;
                }
                let zc = -p_view.z;
                let xv = p_view.x;
                let yv = p_view.y;

                // ---- 3D covariance: Σ = R S Sᵀ Rᵀ = (RS)(RS)ᵀ ----
                let r_mat = Mat3::from_quat(s.rot);
                let s_mat = Mat3::from_diagonal(s.scale);
                let m = r_mat * s_mat;
                let cov3d = m * m.transpose();

                // ---- Rotate covariance into view space ----
                let cov3d_view = w_mat * cov3d * w_mat_t;

                // ---- Compute 2D covariance directly (ILP: exploit J's sparse structure) ----
                // J has the form:
                //   [fx/zc,   0,      fx*xv/zc²]
                //   [0,       fy/zc,  fy*yv/zc²]
                //   [0,       0,      0         ]
                //
                // We only need the top-left 2x2 of J * cov3d_view * J^T.
                // Let C = cov3d_view, with elements c00..c22.
                let c = &cov3d_view;
                let inv_zc = 1.0 / zc;
                let inv_zc2 = inv_zc * inv_zc;

                let j00 = fx * inv_zc;
                let j02 = fx * xv * inv_zc2;
                let j11 = fy * inv_zc;
                let j12 = fy * yv * inv_zc2;

                // Row 0 of J * C: [j00*c00 + j02*c20, j00*c01 + j02*c21, j00*c02 + j02*c22]
                let t0x = j00 * c.x_axis.x + j02 * c.z_axis.x;
                let t0y = j00 * c.y_axis.x + j02 * c.z_axis.y; // c01 = c.y_axis.x (column-major)
                let t0z = j00 * c.x_axis.z + j02 * c.z_axis.z;

                // Row 1 of J * C: [j11*c10 + j12*c20, j11*c11 + j12*c21, j11*c12 + j12*c22]
                let t1y = j11 * c.y_axis.y + j12 * c.z_axis.y;
                let t1z = j11 * c.y_axis.z + j12 * c.z_axis.z;

                // 2D cov = (J*C) * J^T, top-left 2x2:
                // cov2d[0][0] = row0 dot col0_of_J^T = t0x*j00 + t0z*j02
                // cov2d[0][1] = row0 dot col1_of_J^T = t0y*j11 + t0z*j12  (= cov2d[1][0])
                // cov2d[1][1] = row1 dot col1_of_J^T = t1y*j11 + t1z*j12 (note: t1x*0 = 0)
                let cov2d_00 = t0x * j00 + t0z * j02 + eps2d;
                let cov2d_01 = t0y * j11 + t0z * j12; // off-diagonal, no dilation
                let cov2d_11 = t1y * j11 + t1z * j12 + eps2d;

                let det = cov2d_00 * cov2d_11 - cov2d_01 * cov2d_01;
                if det <= 0.0 {
                    return None;
                }

                // Largest eigenvalue → 3σ bbox radius.
                let mid = 0.5 * (cov2d_00 + cov2d_11);
                let lambda1 = mid + (mid * mid - det).max(0.01).sqrt();
                let radius_f = extend_sigma * lambda1.sqrt();
                if !radius_f.is_finite() || radius_f < 1.0 {
                    return None;
                }
                let radius = radius_f.ceil() as i32;

                // Invert 2x2 directly (avoid Mat2::inverse overhead).
                let inv_det = 1.0 / det;
                let cov2d_inv = Mat2::from_cols(
                    Vec2::new(cov2d_11 * inv_det, -cov2d_01 * inv_det),
                    Vec2::new(-cov2d_01 * inv_det, cov2d_00 * inv_det),
                );

                // ---- Project center to pixel coords ----
                let sx = fx * xv * inv_zc + cx;
                let sy = fy * yv * inv_zc + cy;

                let x0 = (sx - radius as f32).floor() as i32;
                let y0 = (sy - radius as f32).floor() as i32;
                let x1 = (sx + radius as f32).ceil() as i32;
                let y1 = (sy + radius as f32).ceil() as i32;

                // Clip to framebuffer.
                let x0 = x0.max(0);
                let y0 = y0.max(0);
                let x1 = x1.min(w_i - 1);
                let y1 = y1.min(h_i - 1);
                if x0 > x1 || y0 > y1 {
                    return None;
                }

                Some(Projected {
                    screen: Vec2::new(sx, sy),
                    depth: zc,
                    cov2d_inv,
                    bbox: [x0, y0, x1, y1],
                    color: s.color,
                    opacity: s.opacity,
                })
            })
            .collect()
    };

    match pool.as_ref() {
        Some(p) => p.install(do_project),
        None => do_project(),
    }
}

/// Sort front-to-back using a 2-pass 16-bit radix sort on bitcast u32 depth
/// keys. Reuses the scratch buffer's aux array across frames to avoid
/// per-frame allocation. Stack-allocated histograms avoid heap allocation.
pub fn sort_by_depth(projected: &mut [Projected], scratch: &mut ScratchBuffers) {
    let n = projected.len();
    if n <= 1 {
        return;
    }

    // Reuse the aux buffer from scratch.
    scratch.sort_aux.clear();
    scratch.sort_aux.reserve(n.saturating_sub(scratch.sort_aux.capacity()));
    unsafe { scratch.sort_aux.set_len(n); }
    let aux = &mut scratch.sort_aux;

    // Both histograms can be computed in a single pass over the keys.
    // Stack-allocated to avoid heap alloc.
    let mut counts_lo = [0u32; 65536];
    let mut counts_hi = [0u32; 65536];
    for p in projected.iter() {
        let k = p.depth.to_bits();
        counts_lo[(k & 0xFFFF) as usize] += 1;
        counts_hi[(k >> 16) as usize] += 1;
    }

    // Pass 1: sort by low 16 bits (projected -> aux).
    {
        let mut offsets = [0u32; 65536];
        let mut sum = 0u32;
        for i in 0..65536 {
            offsets[i] = sum;
            sum += counts_lo[i];
        }
        for p in projected.iter() {
            let bucket = (p.depth.to_bits() & 0xFFFF) as usize;
            let pos = offsets[bucket] as usize;
            offsets[bucket] += 1;
            aux[pos] = *p;
        }
    }

    // Pass 2: sort by high 16 bits (aux -> projected).
    {
        let mut offsets = [0u32; 65536];
        let mut sum = 0u32;
        for i in 0..65536 {
            offsets[i] = sum;
            sum += counts_hi[i];
        }
        for p in aux.iter() {
            let bucket = (p.depth.to_bits() >> 16) as usize;
            let pos = offsets[bucket] as usize;
            offsets[bucket] += 1;
            projected[pos] = *p;
        }
    }
}

/// Front-to-back alpha composite into the RGB framebuffer (single-threaded).
///
/// `fb` is a packed `(rgb, accum_alpha)` buffer of length `width * height`,
/// assumed to be zeroed at the start of each frame.
pub fn composite(projected: &[Projected], fb: &mut [(Vec3, f32)], width: u32, _height: u32, params: &RenderParams) {
    let w = width as usize;
    for p in projected {
        composite_splat(p, fb, w, params);
    }
}

/// Composite a single projected splat into the framebuffer.
/// Hoists row-constant Gaussian terms outside the inner pixel loop (ILP).
///
/// The 2D Gaussian exponent for pixel (px, py) is:
///   power = -0.5 * [dx dy] * [[a b],[b d]] * [dx dy]^T
///         = -0.5 * (a*dx² + 2*b*dx*dy + d*dy²)
///
/// For a fixed row (py), dy is constant, so we precompute:
///   row_base  = -0.5 * d * dy²
///   row_slope = -0.5 * 2 * b * dy = -b * dy
///   dx_coeff  = -0.5 * a
///
/// Then: power = dx_coeff * dx² + row_slope * dx + row_base
#[inline]
fn composite_splat(p: &Projected, fb: &mut [(Vec3, f32)], w: usize, params: &RenderParams) {
    let [x0, y0, x1, y1] = p.bbox;

    // Extract the inverse covariance matrix elements.
    let a = p.cov2d_inv.x_axis.x; // (0,0)
    let b = p.cov2d_inv.x_axis.y; // (0,1) = (1,0) since symmetric
    let d = p.cov2d_inv.y_axis.y; // (1,1)

    // Coefficients for the decomposed quadratic.
    let dx_coeff = -0.5 * a;
    let dy_coeff = -0.5 * d;
    let cross_coeff = -b; // -0.5 * 2 * b

    let saturation = params.saturation;
    let alpha_threshold = params.alpha_threshold;
    let opacity = p.opacity;
    let color = p.color;
    let sx = p.screen.x;
    let sy = p.screen.y;

    for py in y0..=y1 {
        let dy = py as f32 - sy;
        let row_base = dy_coeff * dy * dy;
        let row_slope = cross_coeff * dy;
        let row_offset = py as usize * w;

        for px in x0..=x1 {
            let idx = row_offset + px as usize;
            let cell = &mut fb[idx];
            if cell.1 >= saturation {
                continue;
            }
            let dx = px as f32 - sx;
            let power = dx_coeff * dx * dx + row_slope * dx + row_base;
            if power > 0.0 {
                continue;
            }
            let alpha = (opacity * fast_exp(power)).min(0.999);
            if alpha < alpha_threshold {
                continue;
            }
            let t = 1.0 - cell.1;
            let contrib = t * alpha;
            cell.0 += contrib * color;
            cell.1 += contrib;
        }
    }
}

/// Height in pixels of each horizontal tile for parallel composite.
const TILE_HEIGHT: usize = 16;

/// Build a rayon thread pool with the given number of threads.
/// Returns `None` for 0 (use rayon global pool).
pub fn build_thread_pool(num_threads: usize) -> Option<rayon::ThreadPool> {
    if num_threads > 0 {
        Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build()
                .expect("failed to build rayon thread pool"),
        )
    } else {
        None
    }
}

/// Front-to-back alpha composite into the RGB framebuffer, parallelised over
/// horizontal tiles. Each tile owns a disjoint slice of the framebuffer so
/// there are no data races. Pass a pre-built thread pool to avoid per-frame
/// allocation overhead (use `build_thread_pool`).
pub fn composite_parallel(
    projected: &[Projected],
    fb: &mut [(Vec3, f32)],
    width: u32,
    height: u32,
    params: &RenderParams,
    pool: &Option<rayon::ThreadPool>,
) {
    let w = width as usize;
    let h = height as usize;

    let do_composite = |fb: &mut [(Vec3, f32)]| {
        // Split fb into tile-sized chunks and process in parallel.
        let tile_stride = TILE_HEIGHT * w;
        fb.par_chunks_mut(tile_stride)
            .enumerate()
            .for_each(|(tile_idx, tile_fb)| {
                let tile_y0 = (tile_idx * TILE_HEIGHT) as i32;
                let tile_y1 = (tile_y0 + tile_fb.len() as i32 / w as i32 - 1).min(h as i32 - 1);

                for p in projected {
                    let [x0, y0, x1, y1] = p.bbox;
                    // Skip splats that don't overlap this tile.
                    if y1 < tile_y0 || y0 > tile_y1 {
                        continue;
                    }
                    // Clamp to tile bounds.
                    let py_start = y0.max(tile_y0);
                    let py_end = y1.min(tile_y1);

                    // ILP: hoist row-invariant terms out of the inner loop.
                    let a = p.cov2d_inv.x_axis.x;
                    let b = p.cov2d_inv.x_axis.y;
                    let d = p.cov2d_inv.y_axis.y;
                    let dx_coeff = -0.5 * a;
                    let dy_coeff = -0.5 * d;
                    let cross_coeff = -b;
                    let saturation = params.saturation;
                    let alpha_threshold = params.alpha_threshold;
                    let opacity = p.opacity;
                    let color = p.color;

                    // Pre-compute the row_base threshold: if the dy² term alone
                    // makes alpha negligible, skip the entire row.
                    // opacity * exp(row_base) < alpha_threshold
                    // row_base < ln(alpha_threshold / opacity)
                    let row_base_cutoff = (alpha_threshold / opacity).ln();

                    for py in py_start..=py_end {
                        let dy = py as f32 - p.screen.y;
                        let row_base = dy_coeff * dy * dy;

                        // Early-out: if the Gaussian's dy² term alone is too
                        // small, no pixel on this row can contribute.
                        if row_base < row_base_cutoff {
                            continue;
                        }

                        let row_slope = cross_coeff * dy;
                        let local_row = (py - tile_y0) as usize * w;

                        for px in x0..=x1 {
                            let idx = local_row + px as usize;
                            let cell = &mut tile_fb[idx];
                            if cell.1 >= saturation {
                                continue;
                            }
                            let dx = px as f32 - p.screen.x;
                            let power = dx_coeff * dx * dx + row_slope * dx + row_base;
                            if power > 0.0 {
                                continue;
                            }
                            let alpha = (opacity * fast_exp(power)).min(0.999);
                            if alpha < alpha_threshold {
                                continue;
                            }
                            let t = 1.0 - cell.1;
                            let contrib = t * alpha;
                            cell.0 += contrib * color;
                            cell.1 += contrib;
                        }
                    }
                }
            });
    };

    match pool.as_ref() {
        Some(p) => p.install(|| do_composite(fb)),
        None => do_composite(fb),
    }
}
