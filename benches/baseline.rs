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
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
    IntoParallelRefMutIterator, ParallelIterator,
};
use rayon::slice::{ParallelSlice, ParallelSliceMut};

use minitorch_rs::fast_ops::{FastOps, matmul2d_tiled, matmul2d_tiled_ik};
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

// Module 7 — does cache blocking flip matmul from memory-bound to compute-bound?
// Sweeps a SINGLE function, `matmul2d_tiled::<T>`, over tile sizes, so the only
// variable is the blocking: same raw-slice access, same single thread, same ikj
// inner order. That isolates blocking from every other confound.
//   `notile`   — T = 4096 > every n here, so there is exactly ONE tile spanning
//                the whole matrix: a plain raw-slice ikj with NO cache blocking,
//                sharing tiled's exact code path. The honest naive baseline — B
//                gets re-streamed from L2/L3 on every pass.
//   `tiled_T{16,32,48,64}` — block for cache. On the w9 (Sapphire Rapids) L1d =
//                48 KiB = 6144 f64; the "3 T×T tiles co-resident in L1" model caps
//                useful T at ~45, so this sweep brackets it: 16/32 fit L1, 48 is
//                marginal, 64 spills to L2. The optimum T and the drop past it ARE
//                the cache hierarchy showing through.
//   `tiledik_T{16,32,48,64}` — the controlled experiment: block i and k but
//                stream the FULL j row (no jj loop), so the inner loop is length
//                n again and auto-vectorizes like `notile`, while i/k blocking
//                still gives B-reuse. If the 3-D `tiled` loss was inner-loop
//                fragmentation (not blocking per se), this row should run at
//                ~`notile` speed AND be FLAT across T — T no longer sets the
//                inner-loop length. Flat-and-fast here vs tiled's climb-with-T is
//                the smoking gun.
//   `simple`   — SimpleOps::matmul: naive AND index-based (offset_at per element).
//                Reference tier; (simple − notile) is the indexing-vs-slice cost,
//                (notile − tiled) is pure blocking.
// Single-thread throughout — this measures BLOCKING alone; Rayon + explicit SIMD
// come next, layered on the winning T. NOTE: under `-C target-cpu=native` LLVM
// auto-vectorizes the contiguous inner j-loop in BOTH notile and tiled, so they
// share the same inner vectorization and the gap is blocking only. Throughput =
// n^3 mul-adds (FLOP/s proxy), same as `bench_matmul`, so numbers are directly
// comparable to that group.
fn bench_matmul_blocked(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_blocked");
    group.sample_size(MATMUL_SAMPLES);
    group.measurement_time(MATMUL_MEASURE);
    for n in [256usize, 512, 1024] {
        let a = random(vec![n, n]);
        let b = random(vec![n, n]);
        group.throughput(Throughput::Elements((n * n * n) as u64));
        group.bench_with_input(BenchmarkId::new("simple", n), &n, |bch, _| {
            bch.iter(|| SimpleOps::matmul(black_box(&a), black_box(&b)));
        });
        group.bench_with_input(BenchmarkId::new("notile", n), &n, |bch, _| {
            bch.iter(|| matmul2d_tiled::<4096>(black_box(&a), black_box(&b)));
        });
        group.bench_with_input(BenchmarkId::new("tiled_T16", n), &n, |bch, _| {
            bch.iter(|| matmul2d_tiled::<16>(black_box(&a), black_box(&b)));
        });
        group.bench_with_input(BenchmarkId::new("tiled_T32", n), &n, |bch, _| {
            bch.iter(|| matmul2d_tiled::<32>(black_box(&a), black_box(&b)));
        });
        group.bench_with_input(BenchmarkId::new("tiled_T48", n), &n, |bch, _| {
            bch.iter(|| matmul2d_tiled::<48>(black_box(&a), black_box(&b)));
        });
        group.bench_with_input(BenchmarkId::new("tiled_T64", n), &n, |bch, _| {
            bch.iter(|| matmul2d_tiled::<64>(black_box(&a), black_box(&b)));
        });
        group.bench_with_input(BenchmarkId::new("tiledik_T16", n), &n, |bch, _| {
            bch.iter(|| matmul2d_tiled_ik::<16>(black_box(&a), black_box(&b)));
        });
        group.bench_with_input(BenchmarkId::new("tiledik_T32", n), &n, |bch, _| {
            bch.iter(|| matmul2d_tiled_ik::<32>(black_box(&a), black_box(&b)));
        });
        group.bench_with_input(BenchmarkId::new("tiledik_T48", n), &n, |bch, _| {
            bch.iter(|| matmul2d_tiled_ik::<48>(black_box(&a), black_box(&b)));
        });
        group.bench_with_input(BenchmarkId::new("tiledik_T64", n), &n, |bch, _| {
            bch.iter(|| matmul2d_tiled_ik::<64>(black_box(&a), black_box(&b)));
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

// The decisive companion to `bench_alloc_floor`. That diagnostic showed the
// map/zip benches above are dominated by *producing the output Vec* (allocate +
// first-touch fault), which blinds them to SIMD. This group lifts that confound
// by pre-allocating ONE output buffer per size and reusing it across iterations
// (faulted during warm-up), then writing into it via the `_into` ops. Four
// tiers per op decompose the cost cleanly:
//   `*_alloc`          — status-quo op (`map_op`/`zip_op`): allocates a fresh Vec
//                        every call. The floor-confounded number.
//   `*_simd_into`      — `*_op_into` into the reused buffer: explicit `Simd<f64,N>`
//                        over `par_chunks_exact`, NO allocation.
//   `*_scalar_chunked` — same reused buffer and the SAME `par_chunks_exact(N)`
//                        structure as simd_into, but a scalar inner loop instead
//                        of `Op::simd` (lets LLVM auto-vectorize if it can).
//   `*_scalar_into`    — same reused buffer, element-wise `par_iter` scalar closure.
// This ladder splits the win into its causes:
//   (alloc − simd_into)            = allocation's share
//   (scalar_into − scalar_chunked) = iteration shape (chunked vs element-wise)
//   (scalar_chunked − simd_into)   = the LANES themselves — explicit SIMD vs what
//                                    LLVM auto-vectorizes from the same structure.
// If scalar_chunked ≈ simd_into, the hand-written std::simd bought nothing.
// `add` carries one extra tier, `add_scalar_flat`: same chunk-8 structure but a
// flat-indexed scalar inner (LLVM's best shot) — the tiebreaker for whether add's
// lane edge is real or just `scalar_chunked`'s nested zip auto-vectorizing poorly.
fn bench_reuse_buffer(c: &mut Criterion) {
    let mut group = c.benchmark_group("reuse_buffer");
    group.sample_size(ELEM_SAMPLES);
    group.warm_up_time(ELEM_WARMUP);
    group.measurement_time(ELEM_MEASURE);
    for n in [512usize, 1024, 2048] {
        let size = n * n;
        let a = random(vec![n, n]);
        let b = random(vec![n, n]);
        // a, b are packed (offset 0, contiguous) so storage == logical data.
        let a_data: &[f64] = &a.storage;
        let b_data: &[f64] = &b.storage;
        // Allocated ONCE, reused (and faulted in warm-up) across every iter.
        let mut out = vec![0.0f64; size];
        group.throughput(Throughput::Elements(size as u64));

        // --- map: relu (cheap → bandwidth-bound) ---
        group.bench_with_input(BenchmarkId::new("relu_alloc", n), &n, |bch, _| {
            bch.iter(|| FastOps::map_op::<SimdReLU, LANES>(black_box(&a)));
        });
        group.bench_with_input(BenchmarkId::new("relu_simd_into", n), &n, |bch, _| {
            bch.iter(|| {
                FastOps::map_op_into::<SimdReLU, LANES>(black_box(&a), &mut out);
                black_box(&mut out);
            });
        });
        group.bench_with_input(BenchmarkId::new("relu_scalar_into", n), &n, |bch, _| {
            bch.iter(|| {
                out.par_iter_mut()
                    .zip(a_data.par_iter())
                    .for_each(|(o, &x)| *o = operators::relu(x));
                black_box(&mut out);
            });
        });
        // scalar twin of map_op_into's packed body: same par_chunks_exact(LANES)
        // structure as simd_into, scalar inner loop instead of Op::simd. (Bench
        // sizes are multiples of LANES, so chunks_exact has no remainder.)
        group.bench_with_input(BenchmarkId::new("relu_scalar_chunked", n), &n, |bch, _| {
            bch.iter(|| {
                out.par_chunks_exact_mut(LANES)
                    .zip(a_data.par_chunks_exact(LANES))
                    .for_each(|(dst, src)| {
                        dst.iter_mut()
                            .zip(src)
                            .for_each(|(o, &x)| *o = operators::relu(x));
                    });
                black_box(&mut out);
            });
        });

        // --- zip: add (cheap → bandwidth-bound) ---
        group.bench_with_input(BenchmarkId::new("add_alloc", n), &n, |bch, _| {
            bch.iter(|| {
                FastOps::zip_op::<SimdAdd, LANES>(black_box(&a), black_box(&b))
            });
        });
        group.bench_with_input(BenchmarkId::new("add_simd_into", n), &n, |bch, _| {
            bch.iter(|| {
                FastOps::zip_op_into::<SimdAdd, LANES>(
                    black_box(&a),
                    black_box(&b),
                    &mut out,
                );
                black_box(&mut out);
            });
        });
        group.bench_with_input(BenchmarkId::new("add_scalar_into", n), &n, |bch, _| {
            bch.iter(|| {
                out.par_iter_mut()
                    .zip(a_data.par_iter())
                    .zip(b_data.par_iter())
                    .for_each(|((o, &x), &y)| *o = operators::add(x, y));
                black_box(&mut out);
            });
        });
        group.bench_with_input(BenchmarkId::new("add_scalar_chunked", n), &n, |bch, _| {
            bch.iter(|| {
                out.par_chunks_exact_mut(LANES)
                    .zip(a_data.par_chunks_exact(LANES).zip(b_data.par_chunks_exact(LANES)))
                    .for_each(|(dst, (a_src, b_src))| {
                        dst.iter_mut()
                            .zip(a_src.iter().zip(b_src))
                            .for_each(|(o, (&x, &y))| *o = operators::add(x, y));
                    });
                black_box(&mut out);
            });
        });
        // Tiebreaker for add's lane edge: SAME chunk-8 structure as simd_into, but
        // the scalar inner written the way LLVM auto-vectorizes best — all three
        // operands sliced to equal length (kills bounds checks) + a flat index
        // loop (no nested-iterator state, unlike `scalar_chunked`'s 3-slice zip).
        // If this matches simd_into, the nested zip was the handicap, not the lanes.
        group.bench_with_input(BenchmarkId::new("add_scalar_flat", n), &n, |bch, _| {
            bch.iter(|| {
                out.par_chunks_exact_mut(LANES)
                    .zip(a_data.par_chunks_exact(LANES).zip(b_data.par_chunks_exact(LANES)))
                    .for_each(|(dst, (a_src, b_src))| {
                        let dst = &mut dst[..LANES];
                        let a_src = &a_src[..LANES];
                        let b_src = &b_src[..LANES];
                        #[allow(clippy::needless_range_loop)]
                        for i in 0..LANES {
                            dst[i] = operators::add(a_src[i], b_src[i]);
                        }
                    });
                black_box(&mut out);
            });
        });
    }
    group.finish();
}

// Is `reduce` bandwidth-bound like map/zip — so hand-SIMD is pointless here too?
// Reduce reads the whole n*n input and writes only an n-element output, so its
// traffic is ~one read per element (≈8 B/elem) and its output alloc is negligible
// (no allocation confound, unlike map/zip). So we report `Throughput::Bytes` =
// input bytes read, and criterion prints achieved READ bandwidth directly. Compare
// against the memory wall the (bandwidth-bound) map/zip ops hit at 2048 in
// `reuse_buffer`: relu ≈ 60 GiB/s, add ≈ 87 GiB/s (their Melem/s × bytes-moved/elem).
// If the contiguous-axis reduce lands in that 60–90 GiB/s ballpark, reduce is
// bandwidth-bound and a SIMD / multi-accumulator rewrite can't beat it. (Read-only
// reduce may even EXCEED relu's number — no write-allocate traffic — which is still
// "memory-bound", just a higher achievable ceiling for a pure-read stream.)
//
// Two axes, since the access pattern sets achieved bandwidth:
//   `contig_axis`  — reduce the LAST axis: each fiber walks a contiguous row
//                    (unit stride), full cache-line use → best case, the honest
//                    bandwidth number to compare against the ceiling.
//   `strided_axis` — reduce axis 0: each fiber walks a column (stride n), the
//                    cache-hostile case. A large gap below `contig_axis` is wasted
//                    cache lines, NOT a compute deficit — SIMD wouldn't fix it; that
//                    gap is a layout lesson, not a vectorization opportunity.
fn bench_reduce_bandwidth(c: &mut Criterion) {
    let mut group = c.benchmark_group("reduce_bandwidth");
    group.sample_size(ELEM_SAMPLES);
    group.warm_up_time(ELEM_WARMUP);
    group.measurement_time(ELEM_MEASURE);
    for n in [512usize, 1024, 2048] {
        let a = random(vec![n, n]);
        // Input read dominates; the n-element output is negligible traffic.
        group.throughput(Throughput::Bytes((n * n * 8) as u64));
        group.bench_with_input(BenchmarkId::new("contig_axis", n), &n, |bch, _| {
            bch.iter(|| FastOps::reduce(black_box(&a), operators::add, 0.0, 1, false));
        });
        group.bench_with_input(BenchmarkId::new("strided_axis", n), &n, |bch, _| {
            bch.iter(|| FastOps::reduce(black_box(&a), operators::add, 0.0, 0, false));
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_matmul,
    bench_matmul_blocked,
    bench_map_relu,
    bench_map_exp,
    bench_zip,
    bench_zip_broadcast,
    bench_reduce,
    bench_reduce_contiguous,
    bench_reduce_bandwidth,
    bench_alloc_floor,
    bench_reuse_buffer
);
criterion_main!(benches);
