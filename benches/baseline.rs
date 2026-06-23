//! Module 3 baseline — time the naive `SimpleOps` backend BEFORE any optimization.
//!
//! Run with `cargo bench` (criterion auto-uses the optimized `bench` profile —
//! never benchmark a debug build, it's 10–100x slower and meaningless).
//! Numbers land in `target/criterion/`, with an HTML report at
//! `target/criterion/report/index.html`. Criterion saves each run, so after you
//! write `FastOps` and rerun, it prints the % change against this baseline.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rand::Rng;

use minitorch_rs::tensor_data::TensorData;
use minitorch_rs::tensor_ops::{SimpleOps, TensorOps};

/// A tensor of `shape` filled with uniform random f64 in [0, 1).
fn random(shape: Vec<usize>) -> TensorData {
    let n: usize = shape.iter().product();
    let mut rng = rand::rng();
    let data = (0..n).map(|_| rng.random::<f64>()).collect();
    TensorData::new(data, shape)
}

// matmul: compute-bound. Throughput is set to the multiply-add count (~n^3) so
// criterion reports it as elem/s — a proxy for FLOP/s that should climb as the
// optimizations land. Sizes kept small because the naive backend is slow.
fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul");
    // The naive backend is slow, and 1024^3 ≈ 1.5 s per call; 10 samples (vs
    // criterion's default 100) keeps the big size to ~15 s instead of minutes.
    group.sample_size(10);
    for n in [128usize, 256, 512, 1024] {
        let a = random(vec![n, n]);
        let b = random(vec![n, n]);
        group.throughput(Throughput::Elements((n * n * n) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bch, _| {
            bch.iter(|| SimpleOps::matmul(black_box(&a), black_box(&b)));
        });
    }
    group.finish();
}

// map / zip / reduce: bandwidth-bound. Throughput is the element count, so the
// reported elem/s is the memory-throughput story. Bigger sizes are fine here
// because the work is O(n), not O(n^3).
fn bench_map(c: &mut Criterion) {
    let mut group = c.benchmark_group("map");
    for n in [512usize, 1024, 2048] {
        let a = random(vec![n, n]);
        group.throughput(Throughput::Elements((n * n) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bch, _| {
            bch.iter(|| SimpleOps::map(black_box(&a), |x| x * 2.0));
        });
    }
    group.finish();
}

fn bench_zip(c: &mut Criterion) {
    let mut group = c.benchmark_group("zip");
    for n in [512usize, 1024, 2048] {
        let a = random(vec![n, n]);
        let b = random(vec![n, n]);
        group.throughput(Throughput::Elements((n * n) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bch, _| {
            bch.iter(|| SimpleOps::zip(black_box(&a), black_box(&b), |x, y| x + y));
        });
    }
    group.finish();
}

fn bench_reduce(c: &mut Criterion) {
    let mut group = c.benchmark_group("reduce");
    for n in [512usize, 1024, 2048] {
        let a = random(vec![n, n]);
        group.throughput(Throughput::Elements((n * n) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bch, _| {
            bch.iter(|| SimpleOps::reduce(black_box(&a), |x, y| x + y, 0.0, 0, false));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_matmul, bench_map, bench_zip, bench_reduce);
criterion_main!(benches);
