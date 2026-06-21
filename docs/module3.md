# Module 3 — Efficiency

Rayon-parallel CPU, `std::simd` vectorization, and a CUDA backend via
cudarc + NVRTC. Same `TensorOps` trait, implemented twice more.

- Rich version: [module3.html](./module3.html)
- Upstream lesson: <https://minitorch.github.io/module3/module3/>
- Prior module: [module2.html](./module2.html)

## Contents

- [How to read this doc](#how-to-read-this-doc)
- [Why this module exists](#why-this-module-exists)
- [Task 3.1 — Parallel + vectorized CPU map / zip / reduce](#task-31--parallel--vectorized-cpu-map--zip--reduce)
- [Task 3.2 — Parallel fused matmul (CPU)](#task-32--parallel-fused-matmul-cpu)
- [Task 3.3 — GPU map / zip / reduce (cudarc + NVRTC)](#task-33--gpu-map--zip--reduce-cudarc--nvrtc)
- [Task 3.4 — CUDA tiled matmul](#task-34--cuda-tiled-matmul)
- [Task 3.5 — Wiring, verification, benchmarks](#task-35--wiring-verification-benchmarks)
- [CUDA / Rayon / SIMD cheat sheet](#cuda--rayon--simd-cheat-sheet)
- [What's next](#whats-next)
- [Whiteboard sanity-check](#whiteboard-sanity-check)

## How to read this doc

Module 3 is one trait (`TensorOps`) implemented three more ways: Rayon +
SIMD (`FastOps`), CUDA (`CudaOps`). The wgpu cross-platform retrofit is
Module 3.5; not here.

The no-implementation contract holds. Stubs use `unimplemented!()`. Idiom
snippets use unrelated data — they show API surface, not the kernel. You
write every body, in every backend.

For prose, figures, and side-by-side "before / after" framing against
`SimpleOps`, open [module3.html](./module3.html). This file is the desk
reference.

## Why this module exists

`SimpleOps::zip` at `src/tensor_ops.rs:34-47` is hundreds of times slower
than the silicon can sustain: a `Vec<usize>` materialized per element, a
stride dot-product per `get`, a closure call. Two things change at once.

**Cores.** A laptop has 8-16; a server 64+. Rayon turns the output loop
into fork-join over a thread pool.

**Lanes.** Each core has 128/256/512-bit vector units doing 2-8 f64 ops
per instruction. SIMD turns each thread's inner loop into wide register
ops. The axes compose: 16 cores × 8 lanes = 128-way per node.

GPUs go further. Matmul wins because arithmetic intensity is high (every
byte loaded gets reused n times). Element-wise ops often lose to CPU —
launch + memcpy overhead dwarfs the work. Tensor cores (mixed-precision
matmul units on Volta and later) sit one more rung up; out of scope here.
Measure before you celebrate.

## Task 3.1 — Parallel + vectorized CPU map / zip / reduce

Parallelism (across cores) and vectorization (within a core) are
**independent axes**. Hoping the compiler auto-vectorizes the trait
closures fails reliably: closures + strided access + reductions confuse
`LoopVectorize`. You write both layers explicitly.

### Sub-3.1.a — Rayon parallelism

Concepts: data parallelism, `par_iter`, fork-join, work-stealing. Map and
zip are embarrassingly parallel over output indices. Reduce is not — a
shared accumulator is a write race. Pattern: per-thread fold, then serial
combine. Rayon exposes this as `fold(...).reduce(...)`.

Stubs in `src/fast_ops.rs`:

```rust
pub struct FastOps;

impl TensorOps for FastOps {
    fn map(input: &TensorData, f: impl Fn(f64) -> f64) -> TensorData { unimplemented!() }
    fn zip(a: &TensorData, b: &TensorData, f: impl Fn(f64, f64) -> f64) -> TensorData { unimplemented!() }
    fn reduce(input: &TensorData, f: impl Fn(f64, f64) -> f64, init: f64, dim: usize, keep_dims: bool) -> TensorData { unimplemented!() }
    fn matmul(a: &TensorData, b: &TensorData) -> TensorData { unimplemented!() }
}
```

Rust note: passing the closure across threads requires `Sync` (and maybe
`Send` — let the compiler say which). Because the trait method takes
`impl Fn(...)`, that bound goes on the `TensorOps` *trait declaration* in
`src/tensor_ops.rs`, where every backend shares it — `SimpleOps` picks it
up too (harmless; its closures are already `Sync`). Add `+ Sync` to the
`FastOps` impl alone and you get E0276, "impl has stricter requirements
than trait."

### Sub-3.1.b — SIMD vectorization

`std::simd` is nightly (`#![feature(portable_simd)]`). Flagged as an
advanced Rust feature per CLAUDE.md. Stable alternatives are `pulp`
(macro-driven runtime dispatch) and `wide` (fixed-width). We pick
`std::simd` — the lane-wise `Simd<f64, N>` interface reads closest to the
instruction level.

| Arch        | Lanes (f64) | Instruction family    |
| ----------- | ----------- | --------------------- |
| ARM NEON    | 2           | `fmla v0.2d, ...`     |
| x86 SSE2    | 2           | `addpd`               |
| x86 AVX2    | 4           | `vaddpd ymm`          |
| x86 AVX-512 | 8           | `vaddpd zmm`          |

The trait's `Fn(f64) -> f64` is scalar. SIMD wants `Fn(Simd<f64, N>) ->
Simd<f64, N>` — you have to lift. One decomposition (not the only one) is
a per-chunk inner helper parameterized by lane count; stub it, leave the
body:

```rust
fn simd_map_inner<const N: usize>(
    chunk: &[f64],
    out: &mut [f64],
    f: impl Fn(std::simd::Simd<f64, N>) -> std::simd::Simd<f64, N>,
) where std::simd::LaneCount<N>: std::simd::SupportedLaneCount {
    unimplemented!()
}
```

Tail handling (`len % N != 0`): masked load, scalar sweep, or chunk pad.
Pick one, document it.

### Tests to write

- **Cross-backend equivalence.** Non-contiguous transposed tensor (shape
  `[3, 4]` permuted to `[4, 3]`), apply `map(|x| x * 2.0 + 1.0)` under
  both `SimpleOps` and `FastOps`, assert element-wise equal.
- **Reduce on each axis.** Reduce a `[3, 4]` along dim 0 and dim 1 with
  `(+, 0.0)`, both keep_dims and not; equivalence to `SimpleOps`.
- **Broadcasting zip.** `[4, 3] * [3]` (stride-0 broadcast); equivalence.
- **SIMD bit-exact.** A pure-arithmetic map (`|x| x * 2.0 + 1.0`) on a
  fixed-seed `[1024]` tensor — scalar and SIMD must agree to the last bit,
  because an elementwise op does no cross-lane reduction. Don't use sigmoid
  here: a vectorized `exp` is a polynomial approximation, not bit-identical
  to libm's `f64::exp`, so it would only pass if you auto-lift the scalar
  `exp` per lane — the path that defeats the point.
- **SIMD approximate.** Sum-reduce on `[10_000]` — SIMD reorders the
  sum lane-wise; assert within `f64::EPSILON * size * 8`.

## Task 3.2 — Parallel fused matmul (CPU)

Don't compose matmul from `reduce(zip(broadcast(a), broadcast(b)))` —
that allocates an `(m, k, n)` intermediate. Write a dedicated kernel that
fuses the reduction.

Loop ordering matters for cache. The naive `i, p, j` in `matmul_2d`
(`src/tensor_ops.rs:220-225`) is friendlier than `i, j, p`: with `j`
innermost, `b[p, j]` walks row-major (stride 1, contiguous) and so does
the output `c[i, j]`; `i, j, p` would instead walk `b` down a column
(stride `n`) and thrash the cache. Parallelize the outer batch + `i` axes
(disjoint output rows = no contention). SIMD pays off by vectorizing the
output columns `j`, not the `k` reduction: broadcast `a[i, p]`, stream the
contiguous `b[p, j..]`, accumulate into `c[i, j..]` with one FMA
(`vfmadd231pd` on AVX, `fmla` on NEON). Vectorizing `k` instead forces a
horizontal `reduce_sum` per output cell — the slow shape. Reference
reading: BLIS micro-kernel, OpenBLAS `dgemm_kernel_*.S`. Don't copy them;
read them.

Stub:

```rust
impl TensorOps for FastOps {
    fn matmul(a: &TensorData, b: &TensorData) -> TensorData { unimplemented!() }
}
```

Hand-checkable fixture: `a = [[1, 2, 3], [4, 5, 6]]` (shape `[2, 3]`),
`b = [[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 1, 1]]` (shape `[3, 4]`).
Expected: `[[4, 5, 4, 5], [10, 11, 10, 11]]`. Trace on paper before
trusting the kernel.

## Task 3.3 — GPU map / zip / reduce (cudarc + NVRTC)

Iteration loop: edit `.cu` files locally, run on Colab Pro+ (A100/H100).
Rust host compiles on macOS; CUDA tests gate behind `feature = "cuda"`.

### CUDA C primer

CUDA C is C++ plus a few keywords. A kernel is `__global__` and runs once
per thread; thread coordinates come from built-in variables. Idiom on
unrelated data (incrementing a counter array):

```cuda
__global__ void bump(int *xs, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) xs[i] = xs[i] + 1;
}
```

`i` is the global thread index from block-coord + thread-coord; the
bounds check guards the tail. Host launches with `bump<<<grid, block>>>(xs, n);`.
Same shape for every element-wise kernel.

### cudarc wrappers (AI-free-speed boilerplate)

`src/cuda_ops/mod.rs` stubs:

```rust
pub struct CudaOps;

impl TensorOps for CudaOps {
    fn map(input: &TensorData, f: impl Fn(f64) -> f64) -> TensorData { unimplemented!() }
    fn zip(a: &TensorData, b: &TensorData, f: impl Fn(f64, f64) -> f64) -> TensorData { unimplemented!() }
    fn reduce(input: &TensorData, f: impl Fn(f64, f64) -> f64, init: f64, dim: usize, keep_dims: bool) -> TensorData { unimplemented!() }
    fn matmul(a: &TensorData, b: &TensorData) -> TensorData { unimplemented!() }
}

#[repr(C)]
pub struct KernelArgs { /* shape, strides, storage_offset, size */ }

pub fn launch_map(/* device, kernel_name, args, grid, block */) -> Result<(), CudaError> {
    unimplemented!()
}
```

The trait closure is a problem: it lives on host Rust, can't be shipped
into a kernel. Workaround: host picks a named kernel (`map_sigmoid`,
`zip_add`, ...) and dispatches. The "closure" lives in the host's
choice, not in the kernel body.

### Reduce — cadence only

One output, many threads, all wanting to add into it: a naive `out += x`
per thread serializes on the write. The fix uses the one thing a block
has that a CPU core doesn't — `blockDim.x` threads sharing fast
`__shared__` memory and a barrier. Stage the block's slice into a
`__shared__` array, then combine in parallel so the work halves each
round: log-depth instead of linear. Cross-block combine is a second
kernel, or an `atomicAdd` on the final write.

That's the shape. The decisions are yours, and they're where the lesson
is: which threads stay live each round and how a thread finds the element
it combines with (the obvious modulo answer is also the slow one — that's
the whole subject of the Harris paper), where the barrier goes and what
races without it, and the tail when `blockDim.x` isn't a power of two.
Read Mark Harris, "Optimizing Parallel Reduction in CUDA," *after* you
have a working version one to compare against — it walks seven versions
of this kernel, each fixing the last's bottleneck.

## Task 3.4 — CUDA tiled matmul

The whole point. Naive matmul on GPU is bandwidth-bound — every multiply
loads from global. Tiled matmul caches tiles of A and B in `__shared__`
and reuses them across `TILE` outputs.

### The 4-step rhythm (per tile)

1. Each thread loads one element of A's tile and one of B's tile from
   global into the block's `__shared__` buffers.
2. `__syncthreads()` — wait for all loads. Without this, step 3 reads
   garbage from slots peers haven't written.
3. Each thread multiply-accumulates across the `TILE` shared values for
   its `(row, col)` output slot, into a register.
4. `__syncthreads()` — wait for all reads to finish before any thread
   overwrites the tile with the next iteration's data.

The cadence is what the kernel does; the indexing is what you write.

### Stub

```cuda
__global__ void matmul_tiled(const float *A, const float *B, float *C,
                             int M, int N, int K) {
    __shared__ float a_tile[TILE][TILE];
    __shared__ float b_tile[TILE][TILE];
    // your indexing, your loop, your accumulator
}
```

### Hardware concepts (one paragraph each, no more)

- **Warp.** 32 threads in lock-step on one SM. Branch divergence (some
  lanes if, others else) serializes the warp.
- **Memory coalescing.** 32 consecutive threads reading 32 consecutive
  addresses = one transaction. Strided / misaligned = many; a same-address
  broadcast (all lanes → one address) still coalesces to one.
- **Bank conflicts.** `__shared__` is 32 banks (`addr % 32`). Multiple
  lanes hitting the same bank with different addresses serializes.
  Padding shared arrays by one column is the classic workaround.
- **Occupancy.** Active warps per SM ÷ max warps per SM. Higher =
  better stall-hiding. Limited by registers, shared mem, block size.
  Tune after the kernel works.

Profile with Nsight Compute on Colab once the kernel works. AI-free-speed
exploration; not a graded task.

## Task 3.5 — Wiring, verification, benchmarks

Backend dispatch — two shapes:

```rust
pub type Backend = FastOps;  // type alias — change one line, rebuild

pub fn train<B: TensorOps>(/* ... */) { unimplemented!() }
```

Recommend the type-alias for Module 3. Going generic over `B: TensorOps`
adds monomorphization cost and trait-bound clutter for no gain until
Module 8 (multi-device). For CUDA, gate behind `#[cfg(feature = "cuda")]`
— cudarc needs CUDA Toolkit at build time and that's not on macOS by
default.

Run the existing suite (`tests/test_tensor_ops.rs`, `tests/test_tensor.rs`,
`tests/test_backward.rs`) against each backend. Criterion harness compares
cold: naive CPU → Rayon → Rayon + SIMD → CUDA. **No baseline speedup
numbers belong in the doc** — you measure, you record, the gap is the
lesson.

### Did SIMD actually emit? — verification toolbox

Rust gives you no compile-time guarantee `std::simd` lowered to vector
instructions. Four ways to check:

- **cargo-show-asm.** Dump emitted asm per function:
  ```bash
  cargo install cargo-show-asm
  cargo asm --rust --release 'minitorch_rs::fast_ops::*'
  ```
  Look for `vfmadd231pd` / `vaddpd` (AVX) or `fmla v0.2d, v1.2d, v2.2d`
  (NEON). If you see `vaddsd` (scalar double), it didn't vectorize.

- **LLVM optimization remarks.** Reasons why a loop did or didn't
  vectorize:
  ```bash
  RUSTFLAGS="-C llvm-args=-pass-remarks=loop-vectorize -C llvm-args=-pass-remarks-missed=loop-vectorize" cargo build --release
  ```

- **godbolt.org (Compiler Explorer).** Single-function check: paste, set
  `-C opt-level=3 -C target-cpu=skylake-avx512`, read asm pane.

- **Criterion delta.** SIMD path should benchmark 2-8× the scalar path
  on element-wise. If not, something is wrong.

RUSTFLAGS reference. These `target-cpu` names are x86-only and apply on
x86 hosts (Colab, Intel): the default x86_64 baseline is SSE2, so you opt
into AVX2 / AVX-512 explicitly. On your Apple Silicon Mac, NEON is already
on by default — use `apple-m1` or `native` to confirm it. A `target-cpu`
rustc doesn't recognize for the host is silently ignored (it warns, then
builds), so a green build is *not* proof the wider instructions emitted.

```bash
RUSTFLAGS="-C target-cpu=haswell" cargo build --release          # AVX2  (x86)
RUSTFLAGS="-C target-cpu=skylake-avx512" cargo build --release   # AVX-512 (x86)
RUSTFLAGS="-C target-cpu=apple-m1" cargo build --release         # NEON  (Apple Silicon)
RUSTFLAGS="-C target-cpu=native" cargo build --release           # whatever this box has
```

On Colab: `grep -o 'avx[0-9a-z]*' /proc/cpuinfo | sort -u` shows what the
instance supports.

## CUDA / Rayon / SIMD cheat sheet

| Concept              | CUDA                       | Rayon                       | `std::simd`                  |
| -------------------- | -------------------------- | --------------------------- | ---------------------------- |
| Work unit            | thread                     | task on worker              | lane in a vector             |
| Work group           | thread block               | `rayon::scope` / par chunk  | one `Simd<f64, N>`           |
| Total job            | grid                       | the `par_iter` source       | the chunk loop               |
| Index of self        | `blockIdx + threadIdx`     | iterator element            | lane index `0..N`            |
| Sync within group    | `__syncthreads()`          | fold-then-reduce join       | no equivalent (in-register)  |
| Shared fast memory   | `__shared__` per block     | thread-local accumulator    | the SIMD register itself     |
| Reduce idiom         | tree reduce + atomics      | `.fold(...).reduce(...)`    | `Simd::reduce_sum()`         |
| Failure mode         | bank conflict / divergence | data race / contention      | lane mask / no emit          |
| Width on this hw     | 32 (warp)                  | n cores                     | 2 / 4 / 8 (NEON/AVX2/AVX-512)|

## What's next

Module 3.5 — wgpu retrofit. Port `CudaOps` kernels to WGSL so the backend
runs on Apple Silicon, Intel iGPUs, anything WebGPU touches. Lesson:
"what does WGSL abstract that CUDA doesn't, and at what cost?"

## Whiteboard sanity-check

Close this doc. Answer:

- Why doesn't `par_iter` work directly for reduce? Sketch fold-then-
  reduce in two lines of prose.
- What's the role of the **second** `__syncthreads()` in tiled matmul?
  (Not the first.)
- Which thread reads which element of A in a tile load, and why does
  that pattern matter for coalescing?
- In a `__shared__`-memory tree reduce, which threads stay live on each
  round, and how does a thread find its partner element? Why does the
  naive answer hurt warp efficiency?
- Name two reasons compiler auto-vectorization fails on the trait's
  closure-driven map.
- What's the bit-exactness contract between scalar and SIMD on (a) a
  sigmoid map and (b) a sum-reduce? Why different?
- For 2×3 by 3×4 matmul, smallest fixture that catches a transposed-
  indexing bug?
- Running `cargo asm` on `FastOps::map`: what instruction proves SIMD
  emitted, and what proves it didn't?
