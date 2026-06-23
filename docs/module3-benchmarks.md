# Module 3 benchmarks

Tracking the same ops across backends as Module 3 progresses. **A speedup
comparison is only valid within one machine.**

Topology (settled):
- **CPU columns** — naive → Rayon → Rayon+SIMD — all on the **Xeon w9-3495X**
  (full AVX-512, 56 cores). One machine, so the CPU progression is a valid
  apples-to-apples speedup.
- **GPU columns** — CUDA (and later the wgpu retrofit) — on **Colab** (A100/H100).
- The **CPU → GPU step is a device crossover**, not a same-machine speedup. Label
  it as such; it compares "best CPU effort" vs "GPU", which is the point.

> Note: every row in a given table is one machine, so cross-size comparisons are
> valid. Absolute throughput is CPU-specific; what carries across the project is
> the *relative* speedup each backend buys on its own machine.

## Baseline — naive `SimpleOps` (f64, single-threaded)

- Machine: **remote box** — Intel Xeon w9-3495X (Sapphire Rapids, 56C/112T,
  x86_64); full AVX-512 + AMX. Baseline is scalar `SimpleOps`, so `target-cpu`
  doesn't affect it yet.
- Tool: `cargo bench` (criterion 0.8) · `benches/baseline.rs`
- Date: 2026-06-23

**matmul** (throughput = multiply-adds/s ≈ n³):

| n    | time     | throughput   |
|------|----------|--------------|
| 128  | 5.97 ms  | 351 Melem/s  |
| 256  | 47.1 ms  | 356 Melem/s  |
| 512  | 377 ms   | 356 Melem/s  |
| 1024 | 2.996 s  | 358 Melem/s  |

**map** (throughput = elem/s):

| n    | time     | throughput   |
|------|----------|--------------|
| 512  | 3.82 ms  | 68.7 Melem/s |
| 1024 | 15.3 ms  | 68.3 Melem/s |
| 2048 | 71.7 ms  | 58.5 Melem/s |

**zip**:

| n    | time     | throughput   |
|------|----------|--------------|
| 512  | 4.95 ms  | 53.0 Melem/s |
| 1024 | 19.7 ms  | 53.2 Melem/s |
| 2048 | 90.5 ms  | 46.3 Melem/s |

**reduce** (along dim 0):

| n    | time     | throughput   |
|------|----------|--------------|
| 512  | 1.49 ms  | 176 Melem/s  |
| 1024 | 7.31 ms  | 143 Melem/s  |
| 2048 | 32.2 ms  | 130 Melem/s  |

### Reading the baseline

- **matmul throughput is flat** (~356 Melem/s across a 16× size range) → fully
  overhead-bound by the per-access `get()` bookkeeping; memory/cache effects are
  hidden behind it. ~356 Melem multiply-adds/s ≈ 0.7 GFLOP/s — ~1–2% of one
  core's potential, so the headroom is enormous.
- **map / zip flat then dip at 2048** (~68→58, ~53→46) → a 2048² f64 tensor is
  33.5 MB and spills cache to DRAM; smaller sizes stay cache-resident.
- **reduce is faster but degrades monotonically** (176→143→130) → it reuses one
  index buffer (no per-element alloc, unlike map/zip), so it's lean enough that
  *memory locality shows through*. Reducing along **dim 0** strides memory by `n`,
  so locality worsens as the working set grows past cache.
- Cross-cutting: the leaner an op's per-element overhead, the more the memory
  hierarchy is visible in its scaling. matmul (heaviest overhead) is flattest;
  reduce (leanest) is the most size-sensitive.

## After Rayon

_TBD_

## After Rayon + SIMD

_TBD_

## CUDA

_TBD_
