# Module 3 benchmarks

Tracking the same ops across backends as Module 3 progresses. **A speedup
comparison is only valid within one machine.**

Topology (settled):
- **CPU columns** — naive → serial fast path → Rayon → Rayon+SIMD — all on the
  **Xeon w9-3495X** (full AVX-512, 56 cores). One machine, so the CPU progression
  is a valid apples-to-apples speedup.
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

## After contiguous fast path (serial `FastOps`)

- Machine: same Xeon w9-3495X as the baseline. Still **single-threaded** — no
  Rayon, no explicit SIMD yet. `target-cpu=native` is on, so the compiler
  auto-vectorizes the dependency-free loops (map/zip, matmul's hoisted AXPY); it
  can *not* auto-vectorize the reductions (f64 non-associativity blocks
  reassociation).
- Tool: `cargo bench` (criterion 0.8) · `benches/baseline.rs`, comparing
  `simple/<n>` vs `fast/<n>` in one run. The `simple` column reproduced the
  recorded baseline within noise, confirming same-machine comparability.
- Date: 2026-06-24
- `FastOps` keeps the same `TensorOps` contract; the only change is a contiguous
  fast path (slice-iterate packed tensors) plus `offset_at` for the strided path
  — no per-element `Vec` alloc, no `flat_index` per access. Speedups below are
  `fast` vs `simple` from the same run.

**matmul** (≈ n³ multiply-adds):

| n    | time    | throughput   | speedup |
|------|---------|--------------|---------|
| 128  | 549 µs  | 3.81 Gelem/s | 10.9×   |
| 256  | 3.16 ms | 5.30 Gelem/s | 14.9×   |
| 512  | 38.0 ms | 3.54 Gelem/s | 10.1×   |
| 1024 | 330 ms  | 3.25 Gelem/s |  9.1×   |

**map**:

| n    | time    | throughput   | speedup |
|------|---------|--------------|---------|
| 512  | 155 µs  | 1.69 Gelem/s | 25.3×   |
| 1024 | 620 µs  | 1.69 Gelem/s | 25.7×   |
| 2048 | 10.8 ms | 388 Melem/s  |  6.9×   |

**zip**:

| n    | time    | throughput   | speedup |
|------|---------|--------------|---------|
| 512  | 227 µs  | 1.15 Gelem/s | 21.3×   |
| 1024 | 913 µs  | 1.15 Gelem/s | 21.3×   |
| 2048 | 12.6 ms | 334 Melem/s  |  7.1×   |

**reduce, dim 0** (strided fiber — walks columns, stride n):

| n    | time    | throughput  | speedup |
|------|---------|-------------|---------|
| 512  | 769 µs  | 341 Melem/s | 1.9×    |
| 1024 | 5.73 ms | 183 Melem/s | 1.3×    |
| 2048 | 23.0 ms | 182 Melem/s | 1.4×    |

**reduce, dim 1** (contiguous fiber — stride 1):

| n    | time    | throughput  | speedup |
|------|---------|-------------|---------|
| 512  | 714 µs  | 367 Melem/s | 1.1×    |
| 1024 | 2.89 ms | 363 Melem/s | 1.1×    |
| 2048 | 11.6 ms | 361 Melem/s | 1.1×    |

### Reading the fast path

Each op's speedup is a fingerprint of *what its bottleneck actually was* — and
only the overhead-bound parts got the big win:

- **matmul: ~10× and now compute-bound.** Killing the per-access `get()` unlocked
  the hoisted AXPY inner loop, which auto-vectorizes under `target-cpu=native` →
  3–5 Gelem/s (~7–11 GFLOP/s). Notice it's no longer perfectly flat: it *peaks at
  256* (5.3 Gelem/s, all three matrices fit L2) and tapers by 1024 as the reused
  operand spills cache. The overhead-bound baseline hid this; with overhead gone,
  even matmul shows a mild cache profile.
- **map / zip: ~20–25× while cache-resident, collapsing to ~7× at 2048.** At
  512/1024 the data fits cache and the contiguous loop vectorizes → 1.1–1.7
  Gelem/s. At 2048 (33.5 MB) it spills to DRAM and goes **bandwidth-bound**
  (388 / 334 Melem/s); removing overhead can't push past the memory ceiling. The
  speedup shrank because the bottleneck *moved* — overhead (fixed) → bandwidth
  (not). This is the cache cliff the baseline predicted, now sharper.
- **reduce: small win, two distinct causes.**
  - *dim 0* is a strided column walk (stride n): cache-missing, memory-bound, and
    it degrades with size (341→182) exactly like the baseline. The fast path
    removed overhead but not the access pattern.
  - *dim 1* is a contiguous fiber → cache-friendly and **flat across size**
    (~363 Melem/s) — yet still only ~1.1× over `SimpleOps`, because the ceiling is
    now the **loop-carried f64 sum dependency**. It's non-associative, so the
    compiler won't reassociate it into parallel adds; both backends bottleneck on
    the same scalar add-chain, and overhead was never the cap. Breaking it needs
    explicit multiple-accumulator SIMD (trading bit-exactness for throughput).
- Cross-cutting setup for the next rungs: matmul (compute-bound) should scale
  near-linearly with Rayon; map/zip's 2048 cliff is a *bandwidth* question (more
  cores = more memory channels — does it lift?); reduce is the inversion — it's
  SIMD-hard (dependency within a fiber) but Rayon-easy (fibers are independent
  across output cells), so Rayon may give it the win SIMD can't.

## After Rayon (CPU parallel) — map, zip, reduce measured

- Machine: same Xeon w9-3495X (56C/112T). Rayon over the **independent axis**
  (output ordinal for map/zip, output cells for reduce); each task's
  fold/accumulation order is preserved, so it stays **bit-exact** with
  `SimpleOps` (oracle tests still `assert_eq!`). No SIMD yet.
- Δ below is **parallel vs the serial fast path** (previous section) — the Rayon
  delta alone, from criterion's saved-run `change`. Date: 2026-06-24.
- **No serial/parallel threshold yet**, so small inputs pay rayon's entry cost
  and can *regress*.
- **matmul** still serial (parallelize next).

**map** (parallel `par_iter`, `f = |x| x*2`):

| n    | serial fast      | parallel         | Δ vs serial   |
|------|------------------|------------------|---------------|
| 512  | 155 µs (1.69 G)  | 623 µs (420 M)   | **4× slower** |
| 1024 | 620 µs (1.69 G)  | 853 µs (1.23 G)  | 1.4× slower   |
| 2048 | 10.8 ms (388 M)  | 4.58 ms (917 M)  | **2.36× faster** |

**zip** (parallel `par_iter().zip()`, `f = +`):

| n    | serial fast      | parallel         | Δ vs serial   |
|------|------------------|------------------|---------------|
| 512  | 227 µs (1.15 G)  | 629 µs (417 M)   | **2.8× slower** |
| 1024 | 913 µs (1.15 G)  | 857 µs (1.22 G)  | ~break-even   |
| 2048 | 12.6 ms (334 M)  | 4.54 ms (925 M)  | **2.8× faster** |

**reduce, dim 1** (parallel over output cells, `f = +`):

| n    | serial fast      | parallel          | Δ vs serial   |
|------|------------------|-------------------|---------------|
| 512  | 714 µs (367 M)   | 142 µs (1.83 G)   | **5.0× faster** |
| 1024 | 2.89 ms (363 M)  | 190 µs (5.52 G)   | **15× faster** |
| 2048 | 11.6 ms (361 M)  | 257 µs (16.3 G)   | **45× faster** |

### Reading the Rayon results

- **map regresses small, wins sublinearly large.** `|x| x*2` is one flop/element,
  so rayon's pool-dispatch + parallel `collect` swamps it below the crossover
  (4× *slower* at 512, 1.4× slower at 1024). At 2048 it finally wins — but only
  **2.36× on 56 cores**. That sublinearity is the bandwidth-bound signature:
  917 Melem/s ≈ 15 GB/s, far under the Xeon's ~200 GB/s DRAM peak.
- **zip confirms it from a second angle.** zip mirrors map's shape (~2.8× slower
  at 512, break-even at 1024, ~2.8× at 2048) and lands at the *same* ~920 Melem/s
  parallel ceiling — despite reading **two** inputs (1.5× map's read traffic). If
  input-read bandwidth were the cap, zip would be slower than map; it isn't. So
  the ceiling is the **shared 33.5 MB output write + allocation**, identical for
  both ops. Two independent ops hitting the same wall → it's the output, not the
  reads.
- **reduce scales near-linearly — up to 45× at 2048** (~16 Gelem/s ≈ **130 GB/s**,
  approaching DRAM peak). Throughput *climbs* with size (1.8 → 5.5 → 16 Gelem/s)
  as rayon's fixed overhead amortizes over more work — the opposite of map's wall.
- **The contrast pins map's real bottleneck.** Both ops stream the same 33.5 MB
  input. The difference: reduce **writes almost nothing** (one value per cell) and
  allocates a tiny output; map **writes a full 33.5 MB output and allocates it
  fresh every call**. reduce recruits ~130 GB/s in parallel; map stalls at
  ~15 GB/s — a **9× gap**, far more than the ~2× a read-vs-read+write difference
  could explain. So map's parallel ceiling is the **per-call output write +
  allocation / first-touch page-faulting**, *not* read bandwidth. reduce proves
  the memory subsystem delivers ~130 GB/s once you don't pay that cost. (A
  profiler — `perf`/VTune — would confirm the alloc/page-fault time; the contrast
  is already strong evidence.)
- **Both regress on small inputs** → the fix is a serial/parallel threshold
  (op-cost-dependent; the trivial `x*2` / `+` here is the *worst* case to
  amortize, so real ops like `sigmoid` cross over much sooner). Deferred.
- Setup for matmul: it's **compute-bound** (O(n) data reuse, no full-size output
  churned per element), so predict near-linear scaling like reduce — not map's
  bandwidth-capped 2.36×.

## After Rayon + SIMD

_TBD_

## CUDA

_TBD_
