use glam::{Mat2, Mat3, Vec2, Vec3, Vec4};
use rayon::prelude::*;

use crate::camera::OrbitCamera;
use crate::splat::Splat;

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
/// `sort_unstable_by` — this is the hot loop, stability doesn't matter.
pub fn sort_by_depth(projected: &mut [Projected]) {
    projected.sort_unstable_by(|a, b| {
        a.depth
            .partial_cmp(&b.depth)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

/// Front-to-back alpha composite into the RGB framebuffer (single-threaded).
///
/// `fb` is a packed `(rgb, accum_alpha)` buffer of length `width * height`,
/// assumed to be zeroed at the start of each frame.
pub fn composite(projected: &[Projected], fb: &mut [(Vec3, f32)], width: u32, _height: u32, params: &RenderParams) {
    let w = width as usize;
    for p in projected {
        let [x0, y0, x1, y1] = p.bbox;
        for py in y0..=y1 {
            let row = py as usize * w;
            for px in x0..=x1 {
                let idx = row + px as usize;
                let cell = &mut fb[idx];
                if cell.1 >= params.saturation {
                    continue;
                }
                let d = Vec2::new(px as f32 - p.screen.x, py as f32 - p.screen.y);
                let power = -0.5 * d.dot(p.cov2d_inv * d);
                if power > 0.0 {
                    continue;
                }
                let alpha = (p.opacity * power.exp()).min(0.999);
                if alpha < params.alpha_threshold {
                    continue;
                }
                let t = 1.0 - cell.1;
                let contrib = t * alpha;
                cell.0 += contrib * p.color;
                cell.1 += contrib;
            }
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

                    for py in py_start..=py_end {
                        let local_row = (py - tile_y0) as usize * w;
                        for px in x0..=x1 {
                            let idx = local_row + px as usize;
                            let cell = &mut tile_fb[idx];
                            if cell.1 >= params.saturation {
                                continue;
                            }
                            let d = Vec2::new(px as f32 - p.screen.x, py as f32 - p.screen.y);
                            let power = -0.5 * d.dot(p.cov2d_inv * d);
                            if power > 0.0 {
                                continue;
                            }
                            let alpha = (p.opacity * power.exp()).min(0.999);
                            if alpha < params.alpha_threshold {
                                continue;
                            }
                            let t = 1.0 - cell.1;
                            let contrib = t * alpha;
                            cell.0 += contrib * p.color;
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
