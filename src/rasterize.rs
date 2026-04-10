use glam::{Mat2, Mat3, Vec2, Vec3, Vec4};
use rayon::prelude::*;

use crate::camera::OrbitCamera;
use crate::splat::Splat;

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
pub fn project(splats: &[Splat], camera: &OrbitCamera, params: &RenderParams) -> Vec<Projected> {
    let view = camera.view();
    let w_mat = Mat3::from_mat4(view);
    let (fx, fy, cx, cy) = camera.intrinsics();
    let w_i = camera.width as i32;
    let h_i = camera.height as i32;
    let znear = camera.znear;
    let zfar = camera.zfar;

    splats
        .par_iter()
        .filter_map(|s| {
            // ---- View-transform center ----
            // RH view: camera looks down -z, so a point in front has z < 0.
            let p_view4 = view * Vec4::new(s.pos.x, s.pos.y, s.pos.z, 1.0);
            let p_view = Vec3::new(p_view4.x, p_view4.y, p_view4.z);
            if p_view.z > -znear || p_view.z < -zfar {
                return None;
            }
            // Work with positive depth (zc > 0 means "in front").
            let zc = -p_view.z;
            let xv = p_view.x;
            let yv = p_view.y;

            // ---- 3D covariance: Σ = R S Sᵀ Rᵀ = (RS)(RS)ᵀ ----
            let r_mat = Mat3::from_quat(s.rot);
            let s_mat = Mat3::from_diagonal(s.scale);
            let m = r_mat * s_mat;
            let cov3d = m * m.transpose();

            // ---- Rotate covariance into view space ----
            let cov3d_view = w_mat * cov3d * w_mat.transpose();

            // ---- Jacobian of pinhole projection at (xv, yv, zv) ----
            // Projection (y-down framebuffer, RH view space with zc = -zv):
            //   u = fx * xv / zc + cx
            //   v = fy * yv / zc + cy
            //
            // ∂u/∂xv = fx/zc
            // ∂u/∂zv = fx * xv / zc²
            // ∂v/∂yv = fy/zc
            // ∂v/∂zv = fy * yv / zc²
            //
            // Pad to 3x3 (third row zero) so glam's Mat3 multiplication works.
            let zc2 = zc * zc;
            let j = Mat3::from_cols(
                Vec3::new(fx / zc, 0.0, 0.0),
                Vec3::new(0.0, fy / zc, 0.0),
                Vec3::new(fx * xv / zc2, fy * yv / zc2, 0.0),
            );

            let jcov = j * cov3d_view * j.transpose();

            // Top-left 2x2 is the 2D image-plane covariance; the rest is 0
            // because the third row of J was zero.
            let mut cov2d = Mat2::from_cols(
                Vec2::new(jcov.x_axis.x, jcov.x_axis.y),
                Vec2::new(jcov.y_axis.x, jcov.y_axis.y),
            );
            // Low-pass dilation on the diagonal.
            cov2d.x_axis.x += params.eps2d;
            cov2d.y_axis.y += params.eps2d;

            let det = cov2d.determinant();
            if det <= 0.0 {
                return None;
            }

            // Largest eigenvalue → 3σ bbox radius.
            let a = cov2d.x_axis.x;
            let d = cov2d.y_axis.y;
            let b = 0.5 * (a + d);
            let lambda1 = b + (b * b - det).max(0.01).sqrt();
            let radius_f = params.extend_sigma * lambda1.sqrt();
            if !radius_f.is_finite() || radius_f < 1.0 {
                return None;
            }
            let radius = radius_f.ceil() as i32;

            let cov2d_inv = cov2d.inverse();

            // ---- Project center to pixel coords ----
            let sx = fx * xv / zc + cx;
            let sy = fy * yv / zc + cy;

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
}

/// Sort front-to-back (smaller view-space depth = closer to camera).
///
/// Uses `sort_unstable_by_key` with bitcast u32 keys. Since all depths are
/// positive (zc > 0), f32-to-u32 bitcast preserves ordering, and integer
/// comparison is branchless (no NaN checks, no `partial_cmp` overhead).
pub fn sort_by_depth(projected: &mut [Projected]) {
    projected.sort_unstable_by_key(|p| p.depth.to_bits());
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

/// Front-to-back alpha composite into the RGB framebuffer, parallelised over
/// horizontal tiles. Each tile owns a disjoint slice of the framebuffer so
/// there are no data races. `num_threads` controls the rayon thread pool size
/// (0 means use rayon's default, which is all logical cores).
pub fn composite_parallel(
    projected: &[Projected],
    fb: &mut [(Vec3, f32)],
    width: u32,
    height: u32,
    params: &RenderParams,
    num_threads: usize,
) {
    let w = width as usize;
    let h = height as usize;
    // Build a custom thread pool if num_threads > 0, otherwise use global pool.
    let pool = if num_threads > 0 {
        Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build()
                .expect("failed to build rayon thread pool"),
        )
    } else {
        None
    };

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

                    for py in py_start..=py_end {
                        let dy = py as f32 - p.screen.y;
                        let row_base = dy_coeff * dy * dy;
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

    match pool {
        Some(ref p) => p.install(|| do_composite(fb)),
        None => do_composite(fb),
    }
}
