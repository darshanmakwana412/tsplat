use std::path::PathBuf;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use glam::Vec3;

use tsplat::camera::OrbitCamera;
use tsplat::framebuffer::render_halfblocks;
use tsplat::rasterize::{RenderParams, ScratchBuffers, composite, project, sort_by_depth};
use tsplat::splat::load_ply;

const BENCH_WIDTH: u32 = 120;
const BENCH_HEIGHT: u32 = 80;

fn scene_path() -> PathBuf {
    let local = PathBuf::from("data/garden/point_cloud.ply");
    if local.exists() {
        return local;
    }
    let home =
        PathBuf::from(std::env::var("HOME").unwrap_or_default()).join("datasets/3dgs/garden.ply");
    if home.exists() {
        return home;
    }
    panic!(
        "No garden scene found. Expected at data/garden/point_cloud.ply or ~/datasets/3dgs/garden.ply"
    );
}

fn bench_camera() -> OrbitCamera {
    let mut cam = OrbitCamera::new(BENCH_WIDTH, BENCH_HEIGHT);
    cam.yaw = -0.080;
    cam.pitch = 0.353;
    cam.radius = 52.161;
    cam.fov_y = 55.0_f32.to_radians();
    cam.target = Vec3::new(4.14, -11.97, -38.58);
    cam
}

fn bench_project(c: &mut Criterion) {
    let (splats, _total) = load_ply(&scene_path(), true, 200_000).expect("failed to load scene");
    let camera = bench_camera();
    let params = RenderParams::default();
    let pool = None;

    c.bench_function("project_200k", |b| {
        b.iter(|| {
            let projected = project(
                black_box(&splats),
                black_box(&camera),
                black_box(&params),
                black_box(&pool),
            );
            black_box(projected);
        });
    });
}

fn bench_sort(c: &mut Criterion) {
    let (splats, _total) = load_ply(&scene_path(), true, 200_000).expect("failed to load scene");
    let camera = bench_camera();
    let params = RenderParams::default();
    let pool = None;
    let base_projected = project(&splats, &camera, &params, &pool);

    let mut scratch = ScratchBuffers::new();
    c.bench_function("sort_by_depth_200k", |b| {
        b.iter_batched(
            || base_projected.clone(),
            |mut projected| {
                sort_by_depth(black_box(&mut projected), black_box(&mut scratch));
                black_box(projected);
            },
            criterion::BatchSize::LargeInput,
        );
    });
}

fn bench_composite(c: &mut Criterion) {
    let (splats, _total) = load_ply(&scene_path(), true, 200_000).expect("failed to load scene");
    let camera = bench_camera();
    let params = RenderParams::default();
    let pool = None;
    let mut projected = project(&splats, &camera, &params, &pool);
    let mut scratch = ScratchBuffers::new();
    sort_by_depth(&mut projected, &mut scratch);

    c.bench_function("composite_200k", |b| {
        let mut fb = vec![(Vec3::ZERO, 0.0f32); (BENCH_WIDTH * BENCH_HEIGHT) as usize];
        b.iter(|| {
            for c in fb.iter_mut() {
                *c = (Vec3::ZERO, 0.0);
            }
            composite(
                black_box(&projected),
                black_box(&mut fb),
                BENCH_WIDTH,
                BENCH_HEIGHT,
                black_box(&params),
            );
            black_box(&fb);
        });
    });
}

fn bench_halfblocks(c: &mut Criterion) {
    let (splats, _total) = load_ply(&scene_path(), true, 200_000).expect("failed to load scene");
    let camera = bench_camera();
    let params = RenderParams::default();
    let pool = None;
    let mut scratch = ScratchBuffers::new();
    let mut projected = project(&splats, &camera, &params, &pool);
    sort_by_depth(&mut projected, &mut scratch);
    let mut fb = vec![(Vec3::ZERO, 0.0f32); (BENCH_WIDTH * BENCH_HEIGHT) as usize];
    composite(&projected, &mut fb, BENCH_WIDTH, BENCH_HEIGHT, &params);

    c.bench_function("render_halfblocks", |b| {
        let mut out = String::with_capacity(256 * 1024);
        b.iter(|| {
            render_halfblocks(
                black_box(&fb),
                BENCH_WIDTH,
                BENCH_HEIGHT,
                black_box(&mut out),
            );
            black_box(&out);
        });
    });
}

fn bench_full_pipeline(c: &mut Criterion) {
    let (splats, _total) = load_ply(&scene_path(), true, 200_000).expect("failed to load scene");
    let camera = bench_camera();
    let params = RenderParams::default();
    let pool = None;

    let mut scratch = ScratchBuffers::new();
    c.bench_function("full_pipeline_200k", |b| {
        let mut fb = vec![(Vec3::ZERO, 0.0f32); (BENCH_WIDTH * BENCH_HEIGHT) as usize];
        let mut out = String::with_capacity(256 * 1024);
        b.iter(|| {
            for c in fb.iter_mut() {
                *c = (Vec3::ZERO, 0.0);
            }
            let mut projected = project(&splats, &camera, &params, &pool);
            sort_by_depth(&mut projected, &mut scratch);
            composite(&projected, &mut fb, BENCH_WIDTH, BENCH_HEIGHT, &params);
            render_halfblocks(&fb, BENCH_WIDTH, BENCH_HEIGHT, &mut out);
            black_box(&out);
        });
    });
}

criterion_group!(
    benches,
    bench_project,
    bench_sort,
    bench_composite,
    bench_halfblocks,
    bench_full_pipeline,
);
criterion_main!(benches);
