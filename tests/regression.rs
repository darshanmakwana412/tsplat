//! Regression test for the forward-pass pipeline.
//!
//! Loads the garden scene with fixed camera parameters, runs the full pipeline,
//! and compares the framebuffer output against a saved reference. This catches
//! correctness regressions when optimizing the rendering kernels.
//!
//! On the first run (or when the reference is missing), the test saves the
//! reference and passes. Subsequent runs compare against it.
//!
//! To regenerate the reference after an intentional change:
//!   rm tests/reference/garden_200k.bin && cargo test --release --test regression
//!
//! Run with: cargo test --release --test regression

use std::fs;
use std::path::{Path, PathBuf};

use glam::Vec3;

use tsplat::camera::OrbitCamera;
use tsplat::framebuffer::render_halfblocks;
use tsplat::rasterize::{RenderParams, ScratchBuffers, composite, project, sort_by_depth};
use tsplat::splat::load_ply;

const WIDTH: u32 = 120;
const HEIGHT: u32 = 80;
const MAX_SPLATS: usize = 200_000;
const REF_DIR: &str = "tests/reference";

fn scene_path() -> PathBuf {
    let local = PathBuf::from("data/garden/point_cloud.ply");
    if local.exists() {
        return local;
    }
    let home = PathBuf::from(std::env::var("HOME").unwrap_or_default())
        .join("datasets/3dgs/garden.ply");
    if home.exists() {
        return home;
    }
    panic!(
        "No garden scene found. Expected at data/garden/point_cloud.ply or ~/datasets/3dgs/garden.ply"
    );
}

fn bench_camera() -> OrbitCamera {
    let mut cam = OrbitCamera::new(WIDTH, HEIGHT);
    cam.yaw = -0.080;
    cam.pitch = 0.353;
    cam.radius = 52.161;
    cam.fov_y = 55.0_f32.to_radians();
    cam.target = Vec3::new(4.14, -11.97, -38.58);
    cam
}

/// Serialize framebuffer to bytes: each pixel is 4 floats (r, g, b, alpha) as little-endian f32.
fn fb_to_bytes(fb: &[(Vec3, f32)]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(fb.len() * 16);
    for (rgb, a) in fb {
        bytes.extend_from_slice(&rgb.x.to_le_bytes());
        bytes.extend_from_slice(&rgb.y.to_le_bytes());
        bytes.extend_from_slice(&rgb.z.to_le_bytes());
        bytes.extend_from_slice(&a.to_le_bytes());
    }
    bytes
}

fn run_pipeline() -> (Vec<(Vec3, f32)>, String) {
    let (splats, _total) = load_ply(&scene_path(), true, MAX_SPLATS).expect("failed to load scene");
    let camera = bench_camera();
    let params = RenderParams::default();
    let pool = None; // use rayon global pool for tests

    let mut fb = vec![(Vec3::ZERO, 0.0f32); (WIDTH * HEIGHT) as usize];
    let mut scratch = ScratchBuffers::new();
    let mut projected = project(&splats, &camera, &params, &pool);
    sort_by_depth(&mut projected, &mut scratch);
    composite(&projected, &mut fb, WIDTH, HEIGHT, &params);

    let mut out = String::with_capacity(256 * 1024);
    render_halfblocks(&fb, WIDTH, HEIGHT, &mut out);

    (fb, out)
}

#[test]
fn forward_pass_framebuffer_regression() {
    let ref_path = Path::new(REF_DIR).join("garden_200k.bin");

    let (fb, _ansi_out) = run_pipeline();
    let current_bytes = fb_to_bytes(&fb);

    if !ref_path.exists() {
        // First run — save reference
        fs::create_dir_all(REF_DIR).expect("failed to create reference dir");
        fs::write(&ref_path, &current_bytes).expect("failed to write reference");
        eprintln!(
            "Reference saved to {} ({} bytes). Re-run to verify.",
            ref_path.display(),
            current_bytes.len()
        );
        return;
    }

    let ref_bytes = fs::read(&ref_path).expect("failed to read reference");
    assert_eq!(
        ref_bytes.len(),
        current_bytes.len(),
        "Reference size mismatch: expected {} bytes, got {}. \
         Delete {} and re-run to regenerate.",
        ref_bytes.len(),
        current_bytes.len(),
        ref_path.display()
    );

    // Compare with tolerance — floating point may differ slightly across
    // compiler versions / LLVM optimizations, but should be very close.
    let ref_floats = ref_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect::<Vec<_>>();
    let cur_floats = current_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect::<Vec<_>>();

    let mut max_diff: f32 = 0.0;
    let mut sum_diff: f64 = 0.0;
    let mut diff_count: usize = 0;

    for (i, (r, c)) in ref_floats.iter().zip(cur_floats.iter()).enumerate() {
        let diff = (r - c).abs();
        if diff > 1e-6 {
            diff_count += 1;
            sum_diff += diff as f64;
            if diff > max_diff {
                max_diff = diff;
            }
        }
        // Hard fail on large deviations
        assert!(
            diff < 0.01,
            "Pixel component {} differs by {:.6} (ref={:.6}, cur={:.6}). \
             This indicates a correctness regression. \
             If the change is intentional, delete {} and re-run.",
            i,
            diff,
            r,
            c,
            ref_path.display()
        );
    }

    if diff_count > 0 {
        let mean_diff = sum_diff / diff_count as f64;
        eprintln!(
            "Regression test passed with minor float drift: {} components differ, \
             max_diff={:.8}, mean_diff={:.8}",
            diff_count, max_diff, mean_diff
        );
    } else {
        eprintln!("Regression test passed: exact match with reference.");
    }
}

#[test]
fn forward_pass_ansi_output_nonempty() {
    let (_fb, ansi_out) = run_pipeline();
    assert!(
        ansi_out.len() > 1000,
        "ANSI output is suspiciously small ({} bytes), expected >1000 for a 120x40 frame",
        ansi_out.len()
    );
    // Verify it starts with cursor home
    assert!(
        ansi_out.starts_with("\x1b[H"),
        "ANSI output should start with cursor home escape"
    );
    // Verify it contains half-block characters
    assert!(
        ansi_out.contains('\u{2580}'),
        "ANSI output should contain half-block characters"
    );
}

#[test]
fn forward_pass_deterministic() {
    // Run the pipeline twice and verify identical output
    let (fb1, _) = run_pipeline();
    let (fb2, _) = run_pipeline();

    let bytes1 = fb_to_bytes(&fb1);
    let bytes2 = fb_to_bytes(&fb2);

    assert_eq!(
        bytes1, bytes2,
        "Two runs of the same pipeline with the same inputs produced different results. \
         The pipeline must be deterministic for regression testing to work."
    );
}
