use std::path::PathBuf;
use std::time::Instant;

use glam::Vec3;

use tsplat::camera::OrbitCamera;
use tsplat::framebuffer::render_halfblocks;
use tsplat::rasterize::{
    RenderParams, ScratchBuffers, build_thread_pool, composite_parallel, project,
    sort_by_depth_parallel,
};
use tsplat::splat::{Splat, load_ply, random_scene};

struct BenchArgs {
    frames: usize,
    warmup: usize,
    splats: usize,
    width: u32,
    height: u32,
    threads: Vec<usize>,
    seed: u64,
    ply: Option<PathBuf>,
}

impl Default for BenchArgs {
    fn default() -> Self {
        Self {
            frames: 120,
            warmup: 10,
            splats: 200_000,
            width: 120,
            height: 80,
            threads: vec![1, 2, 4, 8],
            seed: 0xC0FFEE,
            ply: None,
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
            "--warmup" => {
                i += 1;
                args.warmup = raw[i].parse().expect("invalid --warmup");
            }
            "--splats" | "--max-splats" => {
                i += 1;
                args.splats = raw[i].parse().expect("invalid --splats");
            }
            "--width" => {
                i += 1;
                args.width = raw[i].parse().expect("invalid --width");
            }
            "--height" => {
                i += 1;
                args.height = raw[i].parse().expect("invalid --height");
            }
            "--seed" => {
                i += 1;
                args.seed = raw[i].parse().expect("invalid --seed");
            }
            "--threads" => {
                i += 1;
                args.threads = raw[i]
                    .split(',')
                    .map(|s| s.trim().parse().expect("invalid thread count"))
                    .collect();
            }
            "--ply" => {
                i += 1;
                args.ply = Some(PathBuf::from(&raw[i]));
            }
            "--help" | "-h" => {
                eprintln!("Usage: bench_forward [OPTIONS]");
                eprintln!("  --frames N       Frames per thread setting (default: 120)");
                eprintln!("  --warmup N       Warmup frames (default: 10)");
                eprintln!("  --splats N       Number of synthetic gaussians (default: 200000)");
                eprintln!("  --width W        Framebuffer width in pixels (default: 120)");
                eprintln!("  --height H       Framebuffer height in pixels (default: 80)");
                eprintln!("  --seed S         Synthetic scene PRNG seed (default: 0xC0FFEE)");
                eprintln!("  --threads LIST   Comma-separated thread sweep (default: 1,2,4,8)");
                eprintln!("  --ply PATH       Use a real .ply scene instead of synthetic");
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

fn bench_camera(width: u32, height: u32) -> OrbitCamera {
    let mut cam = OrbitCamera::new(width, height);
    cam.target = Vec3::ZERO;
    cam.yaw = 0.6;
    cam.pitch = 0.35;
    cam.radius = 55.0;
    cam.fov_y = 55.0_f32.to_radians();
    cam
}

fn load_scene(args: &BenchArgs) -> Vec<Splat> {
    if let Some(path) = &args.ply {
        let (splats, _) = load_ply(path, true, args.splats).expect("failed to load .ply");
        splats
    } else {
        random_scene(args.splats, args.seed, 20.0)
    }
}

#[derive(Default)]
struct Timings {
    project: Vec<f64>,
    sort: Vec<f64>,
    composite: Vec<f64>,
    halfblocks: Vec<f64>,
    total: Vec<f64>,
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (p / 100.0 * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

fn print_stats(label: &str, values: &mut Vec<f64>) {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let m = mean(values);
    let min = values[0];
    let max = values[values.len() - 1];
    let p50 = percentile(values, 50.0);
    let p95 = percentile(values, 95.0);
    let p99 = percentile(values, 99.0);
    println!(
        "  {:<12} {:>10.1} {:>10.1} {:>10.1} {:>10.1} {:>10.1} {:>10.1}",
        label, m, min, max, p50, p95, p99,
    );
}

struct RunResult {
    threads: usize,
    frames: usize,
    total_mean_us: f64,
    project_mean_us: f64,
    sort_mean_us: f64,
    composite_mean_us: f64,
    halfblocks_mean_us: f64,
    fps: f64,
}

fn run_one(threads: usize, splats: &[Splat], args: &BenchArgs) -> RunResult {
    let camera = bench_camera(args.width, args.height);
    let params = RenderParams::default();
    let fb_size = (args.width * args.height) as usize;
    let mut fb = vec![(Vec3::ZERO, 0.0f32); fb_size];
    let mut out = String::with_capacity(256 * 1024);
    let pool = build_thread_pool(threads);
    let mut scratch = ScratchBuffers::new();

    for _ in 0..args.warmup {
        unsafe {
            std::ptr::write_bytes(fb.as_mut_ptr(), 0, fb.len());
        }
        let mut projected = project(splats, &camera, &params, &pool);
        sort_by_depth_parallel(&mut projected, &mut scratch, &pool);
        composite_parallel(
            &projected,
            &mut fb,
            args.width,
            args.height,
            &params,
            &mut scratch,
            &pool,
        );
        render_halfblocks(&fb, args.width, args.height, &mut out);
    }

    let mut t = Timings::default();
    let wall_start = Instant::now();

    for _ in 0..args.frames {
        unsafe {
            std::ptr::write_bytes(fb.as_mut_ptr(), 0, fb.len());
        }
        let f0 = Instant::now();

        let t0 = Instant::now();
        let mut projected = project(splats, &camera, &params, &pool);
        let t1 = Instant::now();

        sort_by_depth_parallel(&mut projected, &mut scratch, &pool);
        let t2 = Instant::now();

        composite_parallel(
            &projected,
            &mut fb,
            args.width,
            args.height,
            &params,
            &mut scratch,
            &pool,
        );
        let t3 = Instant::now();

        render_halfblocks(&fb, args.width, args.height, &mut out);
        let t4 = Instant::now();

        t.project.push((t1 - t0).as_micros() as f64);
        t.sort.push((t2 - t1).as_micros() as f64);
        t.composite.push((t3 - t2).as_micros() as f64);
        t.halfblocks.push((t4 - t3).as_micros() as f64);
        t.total.push((t4 - f0).as_micros() as f64);
    }

    let wall = wall_start.elapsed().as_secs_f64();

    println!();
    println!("=== threads = {} ===", threads);
    println!(
        "  {:<12} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "stage", "mean(us)", "min(us)", "max(us)", "p50(us)", "p95(us)", "p99(us)",
    );
    println!("  {}", "-".repeat(78));
    print_stats("project", &mut t.project);
    print_stats("sort", &mut t.sort);
    print_stats("composite", &mut t.composite);
    print_stats("halfblocks", &mut t.halfblocks);
    println!("  {}", "-".repeat(78));
    print_stats("TOTAL", &mut t.total);
    println!("  wall={:.2}s  FPS={:.1}", wall, args.frames as f64 / wall);

    RunResult {
        threads,
        frames: args.frames,
        total_mean_us: mean(&t.total),
        project_mean_us: mean(&t.project),
        sort_mean_us: mean(&t.sort),
        composite_mean_us: mean(&t.composite),
        halfblocks_mean_us: mean(&t.halfblocks),
        fps: args.frames as f64 / wall,
    }
}

fn main() {
    let args = parse_args();
    let splats = load_scene(&args);

    println!();
    println!("=== tsplat forward-pass benchmark ===");
    match &args.ply {
        Some(p) => println!("  scene:      {} (real)", p.display()),
        None => println!("  scene:      synthetic (seed=0x{:X})", args.seed),
    }
    println!("  splats:     {}", splats.len());
    println!(
        "  resolution: {}x{} ({}x{} terminal cells)",
        args.width,
        args.height,
        args.width,
        args.height / 2,
    );
    println!(
        "  frames:     {} per config (+{} warmup)",
        args.frames, args.warmup,
    );
    println!("  threads:    {:?}", args.threads);

    let results: Vec<RunResult> = args
        .threads
        .iter()
        .map(|&t| run_one(t, &splats, &args))
        .collect();

    println!();
    println!("=== thread scaling summary ===");
    println!(
        "  {:>7} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "threads", "fps", "total_us", "proj", "sort", "comp", "halfbl",
    );
    println!("  {}", "-".repeat(72));
    let baseline_fps = results.first().map(|r| r.fps).unwrap_or(1.0);
    for r in &results {
        println!(
            "  {:>7} {:>10.1} {:>10.1} {:>10.1} {:>10.1} {:>10.1} {:>10.1}  ({:.2}x)",
            r.threads,
            r.fps,
            r.total_mean_us,
            r.project_mean_us,
            r.sort_mean_us,
            r.composite_mean_us,
            r.halfblocks_mean_us,
            r.fps / baseline_fps,
        );
    }
    println!();
    let _ = (results.first().map(|r| r.frames), baseline_fps);
}
