use glam::{Mat2, Mat3, Vec2, Vec3, Vec4};
use rayon::prelude::*;

use crate::camera::OrbitCamera;
use crate::splat::Splat;

/// Reusable per-frame scratch buffers to avoid heap allocation on every frame.
pub struct ScratchBuffers {
    /// Aux buffer for radix sort ping-pong.
    pub sort_aux: Vec<Projected>,
    /// Tile binning storage (reused across frames).
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

/// Pixel dimensions of a composite tile. 16×16 gives ~35 tiles at 120×80 and
/// ~75 tiles at 240×160, which is enough parallel work for 8–16 threads even
/// on the small terminal.
pub const TILE_W: i32 = 16;
pub const TILE_H: i32 = 16;

/// Per-tile splat list, scatter-built once per frame after the global depth
/// sort. `offsets[i]..offsets[i+1]` is the index range into `splat_indices`
/// for tile `i`; indices are ordered front-to-back because they were inserted
/// in the same order as the sorted `projected` slice.
pub struct TileBins {
    pub num_tiles_x: i32,
    pub num_tiles_y: i32,
    pub offsets: Vec<u32>,
    pub splat_indices: Vec<u32>,
    /// Per-tile cursor used during scatter; reused to avoid reallocation.
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

/// Scatter projected splats into per-tile index lists. Two passes:
/// 1. For each splat, bump the per-tile count for every tile its bbox touches.
/// 2. Prefix-sum → offsets. Second pass scatters splat indices using a cursor
///    vec so each tile's bucket ends up depth-sorted (input order is preserved).
pub fn bin_splats(
    projected: &[Projected],
    width: u32,
    height: u32,
    bins: &mut TileBins,
) {
    let num_tiles_x = ((width as i32) + TILE_W - 1) / TILE_W;
    let num_tiles_y = ((height as i32) + TILE_H - 1) / TILE_H;
    let num_tiles = (num_tiles_x * num_tiles_y) as usize;
    bins.num_tiles_x = num_tiles_x;
    bins.num_tiles_y = num_tiles_y;

    // offsets is used first as a count array, then prefix-summed in place.
    bins.offsets.clear();
    bins.offsets.resize(num_tiles + 1, 0);

    // Pass 1: count tile touches per splat.
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
                // offsets[i+1] eventually becomes the cumulative count up to and
                // including tile i, so bumping index `row + tx + 1` here means
                // a plain in-place prefix sum lands offsets[i] on the *start*
                // of bucket i.
                bins.offsets[row + tx as usize + 1] += 1;
            }
        }
    }

    // Prefix sum: offsets[i] = start of bucket i.
    for i in 1..=num_tiles {
        bins.offsets[i] += bins.offsets[i - 1];
    }
    let total = bins.offsets[num_tiles] as usize;

    bins.splat_indices.clear();
    bins.splat_indices.resize(total, 0);

    // Cursor starts at each bucket's begin and advances as we scatter.
    bins.cursor.clear();
    bins.cursor.extend_from_slice(&bins.offsets[..num_tiles]);

    // Pass 2: scatter splat indices into their tile buckets, preserving the
    // front-to-back order of `projected` inside each bucket.
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
    // FlashGS Eq 3: the classic 3σ rule becomes `k² ≤ max_k2` where
    // `max_k2 = extend_sigma²`. Opacity-aware cutoff clamps k² below this when
    // the gaussian is faint enough that the per-pixel alpha hits the threshold
    // sooner. Precomputed once per frame.
    let max_k2 = params.extend_sigma * params.extend_sigma;
    let alpha_threshold = params.alpha_threshold;

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

                // FlashGS Eq 3: opacity-aware cutoff. A pixel is worth writing
                // while `alpha = opacity * exp(-0.5 * dᵀΣ⁻¹d) >= τ`, which
                // rearranges to `dᵀΣ⁻¹d <= k²` with `k² = 2 ln(opacity/τ)`.
                // Clamp at the classic 3σ so near-opaque splats don't balloon.
                if s.opacity <= alpha_threshold {
                    return None;
                }
                let k2 = (2.0 * (s.opacity / alpha_threshold).ln()).min(max_k2);
                if !(k2 > 0.0) {
                    return None;
                }

                // Tight per-axis ellipse AABB: `|dx|_max = k·√Σ₀₀`,
                // `|dy|_max = k·√Σ₁₁`. This is much smaller than the old
                // circumscribed-circle bbox (`k·√λ_max` on both axes) whenever
                // the projected ellipse is elongated — which is the common
                // case for grazing-angle splats.
                let rx_f = (k2 * cov2d_00).sqrt();
                let ry_f = (k2 * cov2d_11).sqrt();
                if !rx_f.is_finite() || !ry_f.is_finite() || rx_f < 1.0 || ry_f < 1.0 {
                    return None;
                }

                // Invert 2x2 directly (avoid Mat2::inverse overhead).
                let inv_det = 1.0 / det;
                let cov2d_inv = Mat2::from_cols(
                    Vec2::new(cov2d_11 * inv_det, -cov2d_01 * inv_det),
                    Vec2::new(-cov2d_01 * inv_det, cov2d_00 * inv_det),
                );

                // ---- Project center to pixel coords ----
                let sx = fx * xv * inv_zc + cx;
                let sy = fy * yv * inv_zc + cy;

                let x0 = (sx - rx_f).floor() as i32;
                let y0 = (sy - ry_f).floor() as i32;
                let x1 = (sx + rx_f).ceil() as i32;
                let y1 = (sy + ry_f).ceil() as i32;

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

/// Parallel front-to-back sort. Falls back to the serial radix sort for
/// single-threaded callers; otherwise uses rayon's parallel pattern-defeating
/// quicksort keyed on bitcast u32 depth. At 200k splats the parallel path is
/// measurably faster than the serial radix sort once you have 4+ threads.
pub fn sort_by_depth_parallel(
    projected: &mut [Projected],
    scratch: &mut ScratchBuffers,
    pool: &Option<rayon::ThreadPool>,
) {
    if projected.len() < 50_000 || pool.is_none() {
        // Thread-pool overhead isn't worth it for small inputs.
        sort_by_depth(projected, scratch);
        return;
    }
    // Depth is positive in front, so raw bit pattern preserves float order
    // and we can sort on `to_bits()` as a plain u32 key.
    pool.as_ref().unwrap().install(|| {
        projected.par_sort_unstable_by_key(|p| p.depth.to_bits());
    });
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
/// Used by the regression tests as a deterministic reference; the interactive
/// viewer uses `composite_parallel`.
///
/// `fb` is a packed `(rgb, accum_alpha)` buffer of length `width * height`,
/// assumed to be zeroed at the start of each frame.
#[allow(dead_code)]
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
#[allow(dead_code)]
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

/// Send+Sync wrapper around a raw framebuffer pointer. Each composite tile
/// writes to a *disjoint* pixel rect (guaranteed by 2D tile geometry), so we
/// escape Rust's aliasing rules with a bare pointer and a `for_each` that
/// never lets two threads hit the same cell.
#[derive(Clone, Copy)]
struct FbPtr(*mut (Vec3, f32));
unsafe impl Send for FbPtr {}
unsafe impl Sync for FbPtr {}

/// Convenience wrapper: bin + parallel composite in one call. Most callers
/// should use this; the bench splits them to time each stage independently.
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

/// Front-to-back alpha composite into the RGB framebuffer, parallelised over
/// 2D pixel tiles using a pre-built tile binning. Each tile iterates only the
/// splats whose bbox actually touches it, rather than scanning the whole
/// projected list — this changes composite from `O(tiles · splats)` work
/// back down to `O(visible pixel work)`. Raw-pointer writes give every tile
/// its own disjoint pixel range without violating `&mut` aliasing.
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
            // Force-capture the FbPtr wrapper (not just the inner raw pointer)
            // so the compiler sees our Send/Sync impls. Without this binding,
            // Rust 2021 disjoint capture takes only `fb_ptr.0` which is a bare
            // `*mut` and fails the closure's Send+Sync bounds.
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
                // SAFETY: (x0..=x1, y0..=y1) ⊂ this tile's exclusive pixel
                // rect; no other rayon task writes here.
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

/// Composite a single splat into a tile-local pixel rectangle via raw-pointer
/// writes. Shared by the tiled parallel compositor; the same ILP hoists as
/// `composite_splat` are preserved here.
#[inline]
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

    // Row-level early-out: peak of the 1D quadratic in dx is at the vertex of
    // power(dx)=dx_coeff*dx²+row_slope*dx+row_base (dx_coeff<0). Using row_base
    // alone would miss off-center peaks and could skip visible rows.
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
