//! Module 3 — compare `SimpleOps` (naive) vs `FastOps` (Rayon-parallel today,
//! SIMD pending) on the same ops, side by side.
//!
//! Run with `cargo bench` (criterion auto-uses the optimized `bench` profile —
//! never benchmark a debug build, it's 10–100x slower and meaningless).
//! Each group registers both backends as `simple/<n>` and `fast/<n>`, so one
//! run plots them together. Numbers land in `target/criterion/`, HTML report at
//! `target/criterion/report/index.html`. Criterion also saves each run, so a
//! later optimization (Rayon, SIMD) prints its % change against this one.
//!
//! Run length is governed by the knobs below — bump them for longer runs and
//! tighter confidence intervals. They're set per group (matmul vs elementwise)
//! because matmul's big size is multiple seconds per call while the elementwise
//! ops are milliseconds; one global setting can't suit both. NOTE: per-group
//! settings take precedence over `--measurement-time` / `--sample-size` on the
//! CLI, so treat these constants as the source of truth.

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rand::Rng;

use minitorch_rs::fast_ops::FastOps;
use minitorch_rs::tensor_data::TensorData;
use minitorch_rs::tensor_ops::{SimpleOps, TensorOps};

// --- run-length knobs (bump for longer / tighter runs) ---
// matmul: the big size is multiple seconds per call, so few samples but a long
// measurement window — enough for those samples to finish without criterion's
// "unable to complete N samples" warning. (matmul CIs are already very tight, so
// raising these buys little; lower MATMUL_MEASURE if you want a quicker loop.)
const MATMUL_SAMPLES: usize = 10;
const MATMUL_MEASURE: Duration = Duration::from_secs(35);
// elementwise: cheap, so the default 100 samples are affordable; the longer
// warm-up + window let all 100 complete at 2048 and damp run-to-run outliers
// (this is where waiting actually tightens the numbers).
const ELEM_SAMPLES: usize = 100;
const ELEM_WARMUP: Duration = Duration::from_secs(5);
const ELEM_MEASURE: Duration = Duration::from_secs(15);

/// A tensor of `shape` filled with uniform random f64 in [0, 1).
fn random(shape: Vec<usize>) -> TensorData {
    let n: usize = shape.iter().product();
    let mut rng = rand::rng();
    let data = (0..n).map(|_| rng.random::<f64>()).collect();
    TensorData::new(data, shape)
}

// matmul: compute-bound. Throughput = multiply-add count (~n^3), reported as
// elem/s — a proxy for FLOP/s that should climb as the optimizations land.
fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul");
    group.sample_size(MATMUL_SAMPLES);
    group.measurement_time(MATMUL_MEASURE);
    for n in [128usize, 256, 512, 1024] {
        let a = random(vec![n, n]);
        let b = random(vec![n, n]);
        group.throughput(Throughput::Elements((n * n * n) as u64));
        group.bench_with_input(BenchmarkId::new("simple", n), &n, |bch, _| {
            bch.iter(|| SimpleOps::matmul(black_box(&a), black_box(&b)));
        });
        group.bench_with_input(BenchmarkId::new("fast", n), &n, |bch, _| {
            bch.iter(|| FastOps::matmul(black_box(&a), black_box(&b)));
        });
    }
    group.finish();
}

// map / zip / reduce: bandwidth-bound. Throughput = element count, so the
// reported elem/s is the memory-throughput story.
fn bench_map(c: &mut Criterion) {
    let mut group = c.benchmark_group("map");
    group.sample_size(ELEM_SAMPLES);
    group.warm_up_time(ELEM_WARMUP);
    group.measurement_time(ELEM_MEASURE);
    for n in [512usize, 1024, 2048] {
        let a = random(vec![n, n]);
        group.throughput(Throughput::Elements((n * n) as u64));
        group.bench_with_input(BenchmarkId::new("simple", n), &n, |bch, _| {
            bch.iter(|| SimpleOps::map(black_box(&a), |x| x * 2.0));
        });
        group.bench_with_input(BenchmarkId::new("fast", n), &n, |bch, _| {
            bch.iter(|| FastOps::map(black_box(&a), |x| x * 2.0));
        });
    }
    group.finish();
}

fn bench_zip(c: &mut Criterion) {
    let mut group = c.benchmark_group("zip");
    group.sample_size(ELEM_SAMPLES);
    group.warm_up_time(ELEM_WARMUP);
    group.measurement_time(ELEM_MEASURE);
    for n in [512usize, 1024, 2048] {
        let a = random(vec![n, n]);
        let b = random(vec![n, n]);
        group.throughput(Throughput::Elements((n * n) as u64));
        group.bench_with_input(BenchmarkId::new("simple", n), &n, |bch, _| {
            bch.iter(|| SimpleOps::zip(black_box(&a), black_box(&b), |x, y| x + y));
        });
        group.bench_with_input(BenchmarkId::new("fast", n), &n, |bch, _| {
            bch.iter(|| FastOps::zip(black_box(&a), black_box(&b), |x, y| x + y));
        });
    }
    group.finish();
}

fn bench_reduce(c: &mut Criterion) {
    let mut group = c.benchmark_group("reduce");
    group.sample_size(ELEM_SAMPLES);
    group.warm_up_time(ELEM_WARMUP);
    group.measurement_time(ELEM_MEASURE);
    for n in [512usize, 1024, 2048] {
        let a = random(vec![n, n]);
        group.throughput(Throughput::Elements((n * n) as u64));
        group.bench_with_input(BenchmarkId::new("simple", n), &n, |bch, _| {
            bch.iter(|| SimpleOps::reduce(black_box(&a), |x, y| x + y, 0.0, 0, false));
        });
        group.bench_with_input(BenchmarkId::new("fast", n), &n, |bch, _| {
            bch.iter(|| FastOps::reduce(black_box(&a), |x, y| x + y, 0.0, 0, false));
        });
    }
    group.finish();
}

// zip with broadcasting: a packed [n,n] matrix + a [n] row — the canonical
// `activations + bias` shape. Unlike `bench_zip` (both operands packed → fast
// slice path), one operand here is broadcast/strided, the common NN case the
// fast/slow split handles least well. Tracking it separately makes the cost of
// the broadcast path — and any optimization of it — visible.
fn bench_zip_broadcast(c: &mut Criterion) {
    let mut group = c.benchmark_group("zip_broadcast");
    group.sample_size(ELEM_SAMPLES);
    group.warm_up_time(ELEM_WARMUP);
    group.measurement_time(ELEM_MEASURE);
    for n in [512usize, 1024, 2048] {
        let a = random(vec![n, n]);
        let row = random(vec![n]);
        group.throughput(Throughput::Elements((n * n) as u64));
        group.bench_with_input(BenchmarkId::new("simple", n), &n, |bch, _| {
            bch.iter(|| SimpleOps::zip(black_box(&a), black_box(&row), |x, y| x + y));
        });
        group.bench_with_input(BenchmarkId::new("fast", n), &n, |bch, _| {
            bch.iter(|| FastOps::zip(black_box(&a), black_box(&row), |x, y| x + y));
        });
    }
    group.finish();
}

// reduce along the LAST (unit-stride) axis — the cache-friendly counterpart to
// `bench_reduce`, which reduces axis 0 (stride n: one cache line per read). Same
// total work; the gap between the two groups isolates memory-access cost from
// arithmetic. This is also the access pattern a vectorized reduce can actually
// accelerate, so it's the honest baseline for the upcoming SIMD work.
fn bench_reduce_contiguous(c: &mut Criterion) {
    let mut group = c.benchmark_group("reduce_contig_axis");
    group.sample_size(ELEM_SAMPLES);
    group.warm_up_time(ELEM_WARMUP);
    group.measurement_time(ELEM_MEASURE);
    for n in [512usize, 1024, 2048] {
        let a = random(vec![n, n]);
        let last = 1usize; // [n, n] → last axis index
        group.throughput(Throughput::Elements((n * n) as u64));
        group.bench_with_input(BenchmarkId::new("simple", n), &n, |bch, _| {
            bch.iter(|| SimpleOps::reduce(black_box(&a), |x, y| x + y, 0.0, last, false));
        });
        group.bench_with_input(BenchmarkId::new("fast", n), &n, |bch, _| {
            bch.iter(|| FastOps::reduce(black_box(&a), |x, y| x + y, 0.0, last, false));
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_matmul,
    bench_map,
    bench_zip,
    bench_zip_broadcast,
    bench_reduce,
    bench_reduce_contiguous
);
criterion_main!(benches);
