use glam::Vec3;

use tsplat::camera::OrbitCamera;
use tsplat::framebuffer::render_halfblocks;
use tsplat::rasterize::{
    RenderParams, ScratchBuffers, build_thread_pool, composite, composite_parallel, project,
    sort_by_depth,
};
use tsplat::splat::{Splat, random_scene};

const WIDTH: u32 = 120;
const HEIGHT: u32 = 80;
const SPLATS: usize = 50_000;
const SEED: u64 = 0xC0FFEE;

fn scene() -> Vec<Splat> {
    random_scene(SPLATS, SEED, 20.0)
}

fn bench_camera() -> OrbitCamera {
    let mut cam = OrbitCamera::new(WIDTH, HEIGHT);
    cam.target = Vec3::ZERO;
    cam.yaw = 0.6;
    cam.pitch = 0.35;
    cam.radius = 55.0;
    cam.fov_y = 55.0_f32.to_radians();
    cam
}

fn fresh_fb() -> Vec<(Vec3, f32)> {
    vec![(Vec3::ZERO, 0.0f32); (WIDTH * HEIGHT) as usize]
}

fn run_pipeline_serial() -> (Vec<(Vec3, f32)>, String) {
    let splats = scene();
    let camera = bench_camera();
    let params = RenderParams::default();
    let pool = None;

    let mut fb = fresh_fb();
    let mut scratch = ScratchBuffers::new();
    let mut projected = project(&splats, &camera, &params, &pool);
    sort_by_depth(&mut projected, &mut scratch);
    composite(&projected, &mut fb, WIDTH, HEIGHT, &params);

    let mut out = String::with_capacity(256 * 1024);
    render_halfblocks(&fb, WIDTH, HEIGHT, &mut out);
    (fb, out)
}

fn run_pipeline_parallel(threads: usize) -> Vec<(Vec3, f32)> {
    let splats = scene();
    let camera = bench_camera();
    let params = RenderParams::default();
    let pool = build_thread_pool(threads);

    let mut fb = fresh_fb();
    let mut scratch = ScratchBuffers::new();
    let mut projected = project(&splats, &camera, &params, &pool);
    sort_by_depth(&mut projected, &mut scratch);
    composite_parallel(
        &projected,
        &mut fb,
        WIDTH,
        HEIGHT,
        &params,
        &mut scratch,
        &pool,
    );
    fb
}

fn touched_pixels(fb: &[(Vec3, f32)]) -> usize {
    fb.iter().filter(|(_, a)| *a > 0.01).count()
}

#[test]
fn forward_pass_deterministic() {
    let (fb1, _) = run_pipeline_serial();
    let (fb2, _) = run_pipeline_serial();
    assert_eq!(
        fb1.len(),
        fb2.len(),
        "framebuffers differ in length across runs"
    );
    for (i, (a, b)) in fb1.iter().zip(fb2.iter()).enumerate() {
        assert_eq!(
            a.0.to_array(),
            b.0.to_array(),
            "rgb drift at pixel {i} across identical runs"
        );
        assert_eq!(a.1, b.1, "alpha drift at pixel {i} across identical runs");
    }
}

#[test]
fn forward_pass_parallel_matches_serial() {
    let (serial_fb, _) = run_pipeline_serial();
    let parallel_fb = run_pipeline_parallel(4);

    assert_eq!(serial_fb.len(), parallel_fb.len());

    let mut max_rgb_diff: f32 = 0.0;
    let mut max_alpha_diff: f32 = 0.0;
    for (i, (s, p)) in serial_fb.iter().zip(parallel_fb.iter()).enumerate() {
        let drgb = (s.0 - p.0).abs();
        let dmax = drgb.max_element();
        let da = (s.1 - p.1).abs();
        if dmax > max_rgb_diff {
            max_rgb_diff = dmax;
        }
        if da > max_alpha_diff {
            max_alpha_diff = da;
        }
        assert!(
            dmax < 0.01,
            "pixel {i} rgb mismatch: serial={:?} parallel={:?} diff={}",
            s.0,
            p.0,
            dmax
        );
        assert!(
            da < 0.01,
            "pixel {i} alpha mismatch: serial={} parallel={} diff={}",
            s.1,
            p.1,
            da
        );
    }
    eprintln!(
        "parallel/serial equivalence: max_rgb_diff={:.2e} max_alpha_diff={:.2e}",
        max_rgb_diff, max_alpha_diff,
    );
}

#[test]
fn forward_pass_sanity() {
    let (fb, _) = run_pipeline_serial();

    let hits = touched_pixels(&fb);
    let total = fb.len();
    assert!(
        hits * 20 >= total,
        "only {hits}/{total} pixels touched; scene/camera likely broken"
    );

    for (i, (rgb, a)) in fb.iter().enumerate() {
        assert!(
            rgb.x.is_finite() && rgb.y.is_finite() && rgb.z.is_finite(),
            "nan/inf rgb at {i}"
        );
        assert!(a.is_finite(), "nan/inf alpha at {i}");
        assert!(
            *a >= 0.0 && *a <= 1.0 + 1e-5,
            "alpha out of range at {i}: {a}"
        );
        assert!(
            rgb.x >= -1e-5 && rgb.y >= -1e-5 && rgb.z >= -1e-5,
            "negative rgb at {i}: {rgb:?}"
        );
    }
}

#[test]
fn forward_pass_ansi_output_nonempty() {
    let (_fb, ansi) = run_pipeline_serial();
    assert!(
        ansi.len() > 1000,
        "ANSI output is suspiciously small ({} bytes)",
        ansi.len()
    );
    assert!(ansi.starts_with("\x1b[H"), "expected cursor-home prefix");
    assert!(ansi.contains('\u{2580}'), "expected half-block character");
}
