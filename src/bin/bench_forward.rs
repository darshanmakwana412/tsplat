//! Standalone forward-pass benchmark with detailed per-stage statistics.
//!
//! Runs the full pipeline N times with fixed camera parameters from the HUD
//! screenshot and reports timing stats for each stage.
//!
//! Usage:
//!   cargo run --release --bin bench_forward
//!   cargo run --release --bin bench_forward -- --frames 200 --max-splats 500000
//!   cargo run --release --bin bench_forward -- --width 240 --height 160

use std::path::PathBuf;
use std::time::Instant;

use glam::Vec3;

use tsplat::camera::OrbitCamera;
use tsplat::framebuffer::render_halfblocks;
use tsplat::rasterize::{RenderParams, build_thread_pool, composite_parallel, project, sort_by_depth};
use tsplat::splat::load_ply;

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

struct BenchArgs {
    frames: usize,
    max_splats: usize,
    width: u32,
    height: u32,
    warmup: usize,
}

impl Default for BenchArgs {
    fn default() -> Self {
        Self {
            frames: 100,
            max_splats: 200_000,
            width: 120,
            height: 80,
            warmup: 5,
        }
    }
}

fn parse_args() -> BenchArgs {
    let mut args = BenchArgs::default();
    let raw: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < raw.len() {
        match raw[i].as_str() {
            "--frames" | "-n" => {
                i += 1;
                args.frames = raw[i].parse().expect("invalid --frames");
            }
            "--max-splats" => {
                i += 1;
                args.max_splats = raw[i].parse().expect("invalid --max-splats");
            }
            "--width" => {
                i += 1;
                args.width = raw[i].parse().expect("invalid --width");
            }
            "--height" => {
                i += 1;
                args.height = raw[i].parse().expect("invalid --height");
            }
            "--warmup" => {
                i += 1;
                args.warmup = raw[i].parse().expect("invalid --warmup");
            }
            "--help" | "-h" => {
                eprintln!("Usage: bench_forward [OPTIONS]");
                eprintln!("  --frames N       Number of frames to benchmark (default: 100)");
                eprintln!("  --max-splats N   Splat cap (default: 200000)");
                eprintln!("  --width W        Framebuffer width in pixels (default: 120)");
                eprintln!("  --height H       Framebuffer height in pixels (default: 80)");
                eprintln!("  --warmup N       Warmup frames (default: 5)");
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {other}. Use --help for usage.");
                std::process::exit(1);
            }
        }
        i += 1;
    }
    args
}

#[derive(Default)]
struct Timings {
    project_us: Vec<f64>,
    sort_us: Vec<f64>,
    composite_us: Vec<f64>,
    halfblocks_us: Vec<f64>,
    total_us: Vec<f64>,
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (p / 100.0 * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn stats_line(label: &str, values: &mut Vec<f64>) {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let min = values[0];
    let max = values[values.len() - 1];
    let p50 = percentile(values, 50.0);
    let p95 = percentile(values, 95.0);
    let p99 = percentile(values, 99.0);

    println!(
        "  {:<16} {:>10.1} {:>10.1} {:>10.1} {:>10.1} {:>10.1} {:>10.1}",
        label, mean, min, max, p50, p95, p99
    );
}

fn main() {
    let args = parse_args();

    eprintln!("Loading scene...");
    let splats = load_ply(&scene_path(), true, args.max_splats).expect("failed to load scene");
    eprintln!("Loaded {} splats", splats.len());

    let mut camera = OrbitCamera::new(args.width, args.height);
    camera.yaw = -0.080;
    camera.pitch = 0.353;
    camera.radius = 52.161;
    camera.fov_y = 55.0_f32.to_radians();
    camera.target = Vec3::new(4.14, -11.97, -38.58);

    let params = RenderParams::default();
    let fb_size = (args.width * args.height) as usize;
    let mut fb = vec![(Vec3::ZERO, 0.0f32); fb_size];
    let mut out = String::with_capacity(256 * 1024);
    let pool = build_thread_pool(4);

    println!();
    println!("=== tsplat forward-pass benchmark ===");
    println!("  splats:     {}", splats.len());
    println!("  resolution: {}x{} ({}x{} terminal cells)",
        args.width, args.height, args.width, args.height / 2);
    println!("  frames:     {} (+{} warmup)", args.frames, args.warmup);
    println!();

    // Warmup
    for _ in 0..args.warmup {
        // Fast zero-fill via memset.
        unsafe { std::ptr::write_bytes(fb.as_mut_ptr(), 0, fb.len()); }
        let mut projected = project(&splats, &camera, &params);
        sort_by_depth(&mut projected);
        composite_parallel(&projected, &mut fb, args.width, args.height, &params, &pool);
        render_halfblocks(&fb, args.width, args.height, &mut out);
    }

    let mut timings = Timings::default();

    let bench_start = Instant::now();

    for frame in 0..args.frames {
        // Clear
        // Fast zero-fill via memset.
        unsafe { std::ptr::write_bytes(fb.as_mut_ptr(), 0, fb.len()); }

        let frame_start = Instant::now();

        // Project
        let t0 = Instant::now();
        let mut projected = project(&splats, &camera, &params);
        let t1 = Instant::now();

        // Sort
        let t2 = Instant::now();
        sort_by_depth(&mut projected);
        let t3 = Instant::now();

        // Composite
        let t4 = Instant::now();
        composite_parallel(&projected, &mut fb, args.width, args.height, &params, &pool);
        let t5 = Instant::now();

        // Halfblocks
        let t6 = Instant::now();
        render_halfblocks(&fb, args.width, args.height, &mut out);
        let t7 = Instant::now();

        let frame_end = Instant::now();

        timings.project_us.push(t1.duration_since(t0).as_micros() as f64);
        timings.sort_us.push(t3.duration_since(t2).as_micros() as f64);
        timings.composite_us.push(t5.duration_since(t4).as_micros() as f64);
        timings.halfblocks_us.push(t7.duration_since(t6).as_micros() as f64);
        timings.total_us.push(frame_end.duration_since(frame_start).as_micros() as f64);

        if (frame + 1) % 25 == 0 {
            eprint!("\r  frame {}/{}", frame + 1, args.frames);
        }
    }
    eprintln!();

    let wall_secs = bench_start.elapsed().as_secs_f64();
    let effective_fps = args.frames as f64 / wall_secs;

    println!("  {:<16} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "stage", "mean(us)", "min(us)", "max(us)", "p50(us)", "p95(us)", "p99(us)");
    println!("  {}", "-".repeat(82));
    stats_line("project", &mut timings.project_us);
    stats_line("sort", &mut timings.sort_us);
    stats_line("composite", &mut timings.composite_us);
    stats_line("halfblocks", &mut timings.halfblocks_us);
    println!("  {}", "-".repeat(82));
    stats_line("TOTAL", &mut timings.total_us);
    println!();
    println!("  wall time:      {:.2}s", wall_secs);
    println!("  effective FPS:  {:.1}", effective_fps);
    println!("  projected vis:  {} (of {} input)",
        timings.project_us.len(), splats.len());

    // Print breakdown percentages
    let total_mean = timings.total_us.iter().sum::<f64>() / timings.total_us.len() as f64;
    let proj_mean = timings.project_us.iter().sum::<f64>() / timings.project_us.len() as f64;
    let sort_mean = timings.sort_us.iter().sum::<f64>() / timings.sort_us.len() as f64;
    let comp_mean = timings.composite_us.iter().sum::<f64>() / timings.composite_us.len() as f64;
    let half_mean = timings.halfblocks_us.iter().sum::<f64>() / timings.halfblocks_us.len() as f64;

    println!();
    println!("  --- time breakdown (% of mean total) ---");
    println!("  project:    {:5.1}%", 100.0 * proj_mean / total_mean);
    println!("  sort:       {:5.1}%", 100.0 * sort_mean / total_mean);
    println!("  composite:  {:5.1}%", 100.0 * comp_mean / total_mean);
    println!("  halfblocks: {:5.1}%", 100.0 * half_mean / total_mean);
    println!();
}
