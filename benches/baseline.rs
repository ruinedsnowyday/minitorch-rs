//! Module 3/7 — compare three tiers on the same op, side by side:
//!   `simple`      — SimpleOps, naive single-thread (index-based).
//!   `fast_scalar` — FastOps, Rayon-parallel but a scalar closure per element.
//!   `fast_simd`   — FastOps, Rayon + std::simd via map_op / zip_op.
//! The simple→fast_scalar gap is the parallelism win; fast_scalar→fast_simd is
//! the vectorization win in isolation.
//!
//! Run with `cargo bench` (criterion auto-uses the optimized `bench` profile —
//! never benchmark a debug build, it's 10–100x slower and meaningless).
//! Numbers land in `target/criterion/`, HTML report at
//! `target/criterion/report/index.html`. Criterion also saves each run, so a
//! later optimization prints its % change against this one.
//!
//! Run length is governed by the knobs below — bump them for longer runs and
//! tighter confidence intervals. They're set per group (matmul vs elementwise)
//! because matmul's big size is multiple seconds per call while the elementwise
//! ops are milliseconds; one global setting can't suit both. NOTE: per-group
//! settings take precedence over `--measurement-time` / `--sample-size` on the
//! CLI, so treat these constants as the source of truth.

use std::hint::black_box;
use std::time::Duration;

use criterion::{
    BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
};
use rand::Rng;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use minitorch_rs::fast_ops::FastOps;
use minitorch_rs::operators;
use minitorch_rs::simd_ops::{LANES, SimdAdd, SimdExp, SimdReLU};
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

// Elementwise map across the three tiers. We bench TWO ops on purpose, because
// the SIMD payoff hinges on the compute:memory ratio:
//   - relu (cheap, one max): bandwidth-bound. The arithmetic was never the
//     bottleneck, so SIMD has little to win — the honest "SIMD can't rescue a
//     memory-bound op" baseline. Watch fast_scalar≈fast_simd here.
//   - exp (expensive, a transcendental): compute-bound per element, so a
//     vectorized exp is where SIMD should actually pull ahead.
// Throughput = element count, so reported elem/s reads as memory throughput.
fn bench_map_relu(c: &mut Criterion) {
    let mut group = c.benchmark_group("map_relu");
    group.sample_size(ELEM_SAMPLES);
    group.warm_up_time(ELEM_WARMUP);
    group.measurement_time(ELEM_MEASURE);
    for n in [512usize, 1024, 2048] {
        let a = random(vec![n, n]);
        group.throughput(Throughput::Elements((n * n) as u64));
        group.bench_with_input(BenchmarkId::new("simple", n), &n, |bch, _| {
            bch.iter(|| SimpleOps::map(black_box(&a), operators::relu));
        });
        group.bench_with_input(BenchmarkId::new("fast_scalar", n), &n, |bch, _| {
            bch.iter(|| FastOps::map(black_box(&a), operators::relu));
        });
        group.bench_with_input(BenchmarkId::new("fast_simd", n), &n, |bch, _| {
            bch.iter(|| FastOps::map_op::<SimdReLU, LANES>(black_box(&a)));
        });
    }
    group.finish();
}

fn bench_map_exp(c: &mut Criterion) {
    let mut group = c.benchmark_group("map_exp");
    group.sample_size(ELEM_SAMPLES);
    group.warm_up_time(ELEM_WARMUP);
    group.measurement_time(ELEM_MEASURE);
    for n in [512usize, 1024, 2048] {
        let a = random(vec![n, n]);
        group.throughput(Throughput::Elements((n * n) as u64));
        group.bench_with_input(BenchmarkId::new("simple", n), &n, |bch, _| {
            bch.iter(|| SimpleOps::map(black_box(&a), operators::exp));
        });
        group.bench_with_input(BenchmarkId::new("fast_scalar", n), &n, |bch, _| {
            bch.iter(|| FastOps::map(black_box(&a), operators::exp));
        });
        group.bench_with_input(BenchmarkId::new("fast_simd", n), &n, |bch, _| {
            bch.iter(|| FastOps::map_op::<SimdExp, LANES>(black_box(&a)));
        });
    }
    group.finish();
}

// zip: both operands packed → the fast slice path. Three tiers, add as the op
// (cheap, so this is the bandwidth-bound counterpart to map_relu).
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
            bch.iter(|| SimpleOps::zip(black_box(&a), black_box(&b), operators::add));
        });
        group.bench_with_input(BenchmarkId::new("fast_scalar", n), &n, |bch, _| {
            bch.iter(|| FastOps::zip(black_box(&a), black_box(&b), operators::add));
        });
        group.bench_with_input(BenchmarkId::new("fast_simd", n), &n, |bch, _| {
            bch.iter(|| {
                FastOps::zip_op::<SimdAdd, LANES>(black_box(&a), black_box(&b))
            });
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

// reduce along axis 0 (stride n: one cache line per read).
fn bench_reduce(c: &mut Criterion) {
    let mut group = c.benchmark_group("reduce");
    group.sample_size(ELEM_SAMPLES);
    group.warm_up_time(ELEM_WARMUP);
    group.measurement_time(ELEM_MEASURE);
    for n in [512usize, 1024, 2048] {
        let a = random(vec![n, n]);
        group.throughput(Throughput::Elements((n * n) as u64));
        group.bench_with_input(BenchmarkId::new("simple", n), &n, |bch, _| {
            bch.iter(|| {
                SimpleOps::reduce(black_box(&a), |x, y| x + y, 0.0, 0, false)
            });
        });
        group.bench_with_input(BenchmarkId::new("fast", n), &n, |bch, _| {
            bch.iter(|| FastOps::reduce(black_box(&a), |x, y| x + y, 0.0, 0, false));
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
            bch.iter(|| {
                SimpleOps::reduce(black_box(&a), |x, y| x + y, 0.0, last, false)
            });
        });
        group.bench_with_input(BenchmarkId::new("fast", n), &n, |bch, _| {
            bch.iter(|| {
                FastOps::reduce(black_box(&a), |x, y| x + y, 0.0, last, false)
            });
        });
    }
    group.finish();
}

// Diagnostic floor: how much of the map/zip time is just *producing the output
// buffer* — allocate + first-touch page-fault + one write pass — with NO input
// read and NO real compute? Compare these to fast_scalar / fast_simd above: if
// the ops land near this floor, buffer production is the wall and SIMD (which
// only speeds compute) cannot move them.
//
// Two variants on purpose:
//   `seq` — `vec![1.0; size]`: one thread faults every page. We fill a NON-zero
//           value because `vec![0.0; n]` can lower to calloc and skip the very
//           faulting we want to measure.
//   `par` — parallel collect, faulting across threads exactly like the ops do.
//           On a many-core box parallel first-touch can serialize on the kernel's
//           per-mm lock, so `par` is the fair comparison and `seq` vs `par`
//           exposes that contention directly.
// (Both reuse the freed block across iterations, same as the op benches, so the
// allocator is "warm" in all of them — the ratio stays apples-to-apples.)
fn bench_alloc_floor(c: &mut Criterion) {
    let mut group = c.benchmark_group("alloc_floor");
    group.sample_size(ELEM_SAMPLES);
    group.warm_up_time(ELEM_WARMUP);
    group.measurement_time(ELEM_MEASURE);
    for n in [512usize, 1024, 2048] {
        let size = n * n;
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("seq", n), &n, |bch, _| {
            bch.iter(|| black_box(vec![black_box(1.0f64); size]));
        });
        group.bench_with_input(BenchmarkId::new("par", n), &n, |bch, _| {
            bch.iter(|| {
                let v: Vec<f64> =
                    (0..size).into_par_iter().map(|_| black_box(1.0f64)).collect();
                black_box(v)
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_matmul,
    bench_map_relu,
    bench_map_exp,
    bench_zip,
    bench_zip_broadcast,
    bench_reduce,
    bench_reduce_contiguous,
    bench_alloc_floor
);
criterion_main!(benches);
