use glam::{Mat2, Mat3, Vec2, Vec3, Vec4};
use rayon::prelude::*;

use crate::camera::OrbitCamera;
use crate::splat::Splat;

pub struct ScratchBuffers {
    pub sort_aux: Vec<Projected>,
    pub tiles: TileBins,
}

impl ScratchBuffers {
    pub fn new() -> Self {
        Self {
            sort_aux: Vec::new(),
            tiles: TileBins::new(),
        }
    }
}

impl Default for ScratchBuffers {
    fn default() -> Self {
        Self::new()
    }
}

pub const TILE_W: i32 = 16;
pub const TILE_H: i32 = 16;

pub struct TileBins {
    pub num_tiles_x: i32,
    pub num_tiles_y: i32,
    pub offsets: Vec<u32>,
    pub splat_indices: Vec<u32>,
    cursor: Vec<u32>,
}

impl TileBins {
    pub fn new() -> Self {
        Self {
            num_tiles_x: 0,
            num_tiles_y: 0,
            offsets: Vec::new(),
            splat_indices: Vec::new(),
            cursor: Vec::new(),
        }
    }

    #[inline]
    pub fn num_tiles(&self) -> usize {
        (self.num_tiles_x * self.num_tiles_y) as usize
    }
}

impl Default for TileBins {
    fn default() -> Self {
        Self::new()
    }
}

pub fn bin_splats(projected: &[Projected], width: u32, height: u32, bins: &mut TileBins) {
    let num_tiles_x = ((width as i32) + TILE_W - 1) / TILE_W;
    let num_tiles_y = ((height as i32) + TILE_H - 1) / TILE_H;
    let num_tiles = (num_tiles_x * num_tiles_y) as usize;
    bins.num_tiles_x = num_tiles_x;
    bins.num_tiles_y = num_tiles_y;

    bins.offsets.clear();
    bins.offsets.resize(num_tiles + 1, 0);

    for p in projected {
        let [x0, y0, x1, y1] = p.bbox;
        let tx0 = (x0 / TILE_W).max(0);
        let ty0 = (y0 / TILE_H).max(0);
        let tx1 = (x1 / TILE_W).min(num_tiles_x - 1);
        let ty1 = (y1 / TILE_H).min(num_tiles_y - 1);
        if tx0 > tx1 || ty0 > ty1 {
            continue;
        }
        for ty in ty0..=ty1 {
            let row = (ty * num_tiles_x) as usize;
            for tx in tx0..=tx1 {
                bins.offsets[row + tx as usize + 1] += 1;
            }
        }
    }

    for i in 1..=num_tiles {
        bins.offsets[i] += bins.offsets[i - 1];
    }
    let total = bins.offsets[num_tiles] as usize;

    bins.splat_indices.clear();
    bins.splat_indices.resize(total, 0);

    bins.cursor.clear();
    bins.cursor.extend_from_slice(&bins.offsets[..num_tiles]);

    for (idx, p) in projected.iter().enumerate() {
        let [x0, y0, x1, y1] = p.bbox;
        let tx0 = (x0 / TILE_W).max(0);
        let ty0 = (y0 / TILE_H).max(0);
        let tx1 = (x1 / TILE_W).min(num_tiles_x - 1);
        let ty1 = (y1 / TILE_H).min(num_tiles_y - 1);
        if tx0 > tx1 || ty0 > ty1 {
            continue;
        }
        for ty in ty0..=ty1 {
            let row = (ty * num_tiles_x) as usize;
            for tx in tx0..=tx1 {
                let tile = row + tx as usize;
                let pos = bins.cursor[tile] as usize;
                bins.splat_indices[pos] = idx as u32;
                bins.cursor[tile] += 1;
            }
        }
    }
}

#[inline(always)]
fn fast_exp(x: f32) -> f32 {
    let x = x.max(-87.0);
    let v = (12102203.0f32 * x + 1065353216.0) as i32;
    f32::from_bits(v as u32)
}

#[derive(Clone, Copy)]
pub struct Projected {
    pub screen: Vec2,
    pub depth: f32,
    pub cov2d_inv: Mat2,
    pub bbox: [i32; 4],
    pub color: Vec3,
    pub opacity: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct RenderParams {
    pub eps2d: f32,
    pub alpha_threshold: f32,
    pub extend_sigma: f32,
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
    let max_k2 = params.extend_sigma * params.extend_sigma;
    let alpha_threshold = params.alpha_threshold;

    let do_project = || {
        splats
            .par_iter()
            .filter_map(|s| {
                let p_view4 = view * Vec4::new(s.pos.x, s.pos.y, s.pos.z, 1.0);
                let p_view = Vec3::new(p_view4.x, p_view4.y, p_view4.z);
                if p_view.z > -znear || p_view.z < -zfar {
                    return None;
                }
                let zc = -p_view.z;
                let xv = p_view.x;
                let yv = p_view.y;

                let r_mat = Mat3::from_quat(s.rot);
                let s_mat = Mat3::from_diagonal(s.scale);
                let m = r_mat * s_mat;
                let cov3d = m * m.transpose();

                let cov3d_view = w_mat * cov3d * w_mat_t;

                let c = &cov3d_view;
                let inv_zc = 1.0 / zc;
                let inv_zc2 = inv_zc * inv_zc;

                let j00 = fx * inv_zc;
                let j02 = fx * xv * inv_zc2;
                let j11 = fy * inv_zc;
                let j12 = fy * yv * inv_zc2;

                let t0x = j00 * c.x_axis.x + j02 * c.z_axis.x;
                let t0y = j00 * c.y_axis.x + j02 * c.z_axis.y;
                let t0z = j00 * c.x_axis.z + j02 * c.z_axis.z;

                let t1y = j11 * c.y_axis.y + j12 * c.z_axis.y;
                let t1z = j11 * c.y_axis.z + j12 * c.z_axis.z;

                let cov2d_00 = t0x * j00 + t0z * j02 + eps2d;
                let cov2d_01 = t0y * j11 + t0z * j12;
                let cov2d_11 = t1y * j11 + t1z * j12 + eps2d;

                let det = cov2d_00 * cov2d_11 - cov2d_01 * cov2d_01;
                if det <= 0.0 {
                    return None;
                }

                if s.opacity <= alpha_threshold {
                    return None;
                }
                let k2 = (2.0 * (s.opacity / alpha_threshold).ln()).min(max_k2);
                if k2.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
                    return None;
                }

                let rx_f = (k2 * cov2d_00).sqrt();
                let ry_f = (k2 * cov2d_11).sqrt();
                if !rx_f.is_finite() || !ry_f.is_finite() || rx_f < 1.0 || ry_f < 1.0 {
                    return None;
                }

                let inv_det = 1.0 / det;
                let cov2d_inv = Mat2::from_cols(
                    Vec2::new(cov2d_11 * inv_det, -cov2d_01 * inv_det),
                    Vec2::new(-cov2d_01 * inv_det, cov2d_00 * inv_det),
                );

                let sx = fx * xv * inv_zc + cx;
                let sy = fy * yv * inv_zc + cy;

                let x0 = (sx - rx_f).floor() as i32;
                let y0 = (sy - ry_f).floor() as i32;
                let x1 = (sx + rx_f).ceil() as i32;
                let y1 = (sy + ry_f).ceil() as i32;

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

pub fn sort_by_depth_parallel(
    projected: &mut [Projected],
    scratch: &mut ScratchBuffers,
    pool: &Option<rayon::ThreadPool>,
) {
    if projected.len() < 50_000 || pool.is_none() {
        sort_by_depth(projected, scratch);
        return;
    }
    pool.as_ref().unwrap().install(|| {
        projected.par_sort_unstable_by_key(|p| p.depth.to_bits());
    });
}

pub fn sort_by_depth(projected: &mut [Projected], scratch: &mut ScratchBuffers) {
    let n = projected.len();
    if n <= 1 {
        return;
    }

    scratch.sort_aux.clear();
    scratch
        .sort_aux
        .reserve(n.saturating_sub(scratch.sort_aux.capacity()));
    // Radix passes overwrite every element; avoid zeroing for allocation cost.
    #[allow(clippy::uninit_vec)]
    unsafe {
        scratch.sort_aux.set_len(n);
    }
    let aux = &mut scratch.sort_aux;

    let mut counts_lo = [0u32; 65536];
    let mut counts_hi = [0u32; 65536];
    for p in projected.iter() {
        let k = p.depth.to_bits();
        counts_lo[(k & 0xFFFF) as usize] += 1;
        counts_hi[(k >> 16) as usize] += 1;
    }

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

#[allow(dead_code)]
pub fn composite(
    projected: &[Projected],
    fb: &mut [(Vec3, f32)],
    width: u32,
    _height: u32,
    params: &RenderParams,
) {
    let w = width as usize;
    for p in projected {
        composite_splat(p, fb, w, params);
    }
}

#[inline]
#[allow(dead_code)]
fn composite_splat(p: &Projected, fb: &mut [(Vec3, f32)], w: usize, params: &RenderParams) {
    let [x0, y0, x1, y1] = p.bbox;

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

#[derive(Clone, Copy)]
struct FbPtr(*mut (Vec3, f32));
unsafe impl Send for FbPtr {}
unsafe impl Sync for FbPtr {}

pub fn composite_parallel(
    projected: &[Projected],
    fb: &mut [(Vec3, f32)],
    width: u32,
    height: u32,
    params: &RenderParams,
    scratch: &mut ScratchBuffers,
    pool: &Option<rayon::ThreadPool>,
) {
    bin_splats(projected, width, height, &mut scratch.tiles);
    composite_tiled(projected, &scratch.tiles, fb, width, height, params, pool);
}

pub fn composite_tiled(
    projected: &[Projected],
    bins: &TileBins,
    fb: &mut [(Vec3, f32)],
    width: u32,
    height: u32,
    params: &RenderParams,
    pool: &Option<rayon::ThreadPool>,
) {
    let w = width as usize;
    let h_i = height as i32;
    let w_i = width as i32;
    let num_tiles_x = bins.num_tiles_x;
    let num_tiles = bins.num_tiles();
    let fb_ptr = FbPtr(fb.as_mut_ptr());

    let do_composite = || {
        (0..num_tiles).into_par_iter().for_each(move |tile_idx| {
            let fbp = fb_ptr;
            let tile_x = (tile_idx as i32) % num_tiles_x;
            let tile_y = (tile_idx as i32) / num_tiles_x;
            let px0 = tile_x * TILE_W;
            let py0 = tile_y * TILE_H;
            let px1 = (px0 + TILE_W - 1).min(w_i - 1);
            let py1 = (py0 + TILE_H - 1).min(h_i - 1);
            if px0 > px1 || py0 > py1 {
                return;
            }

            let start = bins.offsets[tile_idx] as usize;
            let end = bins.offsets[tile_idx + 1] as usize;
            if start == end {
                return;
            }
            let splat_ids = &bins.splat_indices[start..end];

            for &sid in splat_ids {
                let p = unsafe { projected.get_unchecked(sid as usize) };
                let [bx0, by0, bx1, by1] = p.bbox;
                let x0 = bx0.max(px0);
                let y0 = by0.max(py0);
                let x1 = bx1.min(px1);
                let y1 = by1.min(py1);
                if x0 > x1 || y0 > y1 {
                    continue;
                }
                unsafe {
                    composite_splat_region(p, fbp.0, w, x0, y0, x1, y1, params);
                }
            }
        });
    };

    match pool.as_ref() {
        Some(p) => p.install(do_composite),
        None => do_composite(),
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn composite_splat_region(
    p: &Projected,
    fb_ptr: *mut (Vec3, f32),
    w: usize,
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
    params: &RenderParams,
) {
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
    let sx = p.screen.x;
    let sy = p.screen.y;

    let row_peak_cutoff = (alpha_threshold / opacity).ln();
    let inv_a = 1.0 / a;

    for py in y0..=y1 {
        let dy = py as f32 - sy;
        let row_base = dy_coeff * dy * dy;
        let row_slope = cross_coeff * dy;
        let row_peak = row_base + 0.5 * row_slope * row_slope * inv_a;
        if row_peak < row_peak_cutoff {
            continue;
        }
        let row_offset = py as usize * w;

        for px in x0..=x1 {
            let idx = row_offset + px as usize;
            let cell = unsafe { &mut *fb_ptr.add(idx) };
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
