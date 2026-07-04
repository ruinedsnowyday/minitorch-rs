//! High-performance GEMM: a BLIS-style register microkernel fed by two
//! interchangeable outer drivers, so we can benchmark **cache-aware iterative
//! tiling** against **cache-oblivious recursion** with the *same* leaf kernel.
//!
//! Three layers (build them bottom-up, in this order):
//!
//!   1. `microkernel` — computes a tiny `MR×NR` block of C with the accumulators
//!      pinned in **registers** across the whole k-strip. This is the only piece
//!      tuned to the CPU (register file + vector width). It is the lesson.
//!   2. `pack_a` / `pack_b` — copy A/B panels into contiguous, kernel-order
//!      buffers so every microkernel load is unit-stride and cache-line dense.
//!      Packing is half the speed; don't skip it.
//!   3. The **drivers** — `matmul_iterative` (BLIS 5-loops, cache-AWARE, tuned
//!      `MC/KC/NC`) and `matmul_recursive` (Frigo cache-OBLIVIOUS: split the
//!      largest dim, recurse, bottom out into a packed microkernel base case).
//!      Both call the same packed macrokernel; only the *walk* differs.
//!
//! The comparison this module exists to run (see `benches/baseline.rs::bench_gemm`):
//! iterative tuned blocking vs. oblivious recursion, same kernel, several n.
//! Expected: recursion lands within a modest constant of tuned BLIS (packing +
//! TLB + the K-reduction cost it the top spot) while winning on zero-tuning
//! portability. Ref: Frigo, Leiserson, Prokop, Ramachandran, "Cache-Oblivious
//! Algorithms", ACM TALG 8(1), 2012 (DOI 10.1145/2071379.2071383).
//!
//! SCAFFOLD STATUS: every compute body below is `todo!()` — the kernel, packing,
//! and both drivers are yours to write (core Module-7 material). The signatures
//! are a *suggested* contract; if you choose a different packed layout, change
//! the kernel and both pack functions together.
#![allow(unused)] // SCAFFOLD ONLY — delete this once the stubs are implemented;
// it silences dead-code / unused-variable warnings for the not-yet-wired pieces.

use crate::tensor_data::TensorData;

// --- Register-tile dimensions (the microkernel's C block) ---
// The `MR*NR/LANES` accumulator vectors PLUS a few for A-splats / B-loads must
// fit the register file (AVX-512 = 32 zmm). 8×8 → 8 accumulator vectors, roomy.
// VERIFY with `cargo asm` that these accumulators stay in registers — if they
// spill to the stack, the whole point is lost. Tune empirically.
pub const MR: usize = 8;
pub const NR: usize = 8;
pub const LANES: usize = 8; // f64 SIMD lanes; NR must be a multiple of this.

// --- Cache-block dimensions for the ITERATIVE driver only ---
// Sized so the packed A panel (MC×KC) is L2-resident, the packed B panel
// (KC×NC) is L3-resident, and the active B micro-panel (KC×NR) is L1-resident.
// The RECURSIVE driver needs NONE of these — that's the point of obliviousness.
const MC: usize = 256;
const KC: usize = 256;
const NC: usize = 4096;

/// The register microkernel: `C[i][j] += Σ_p Ap[p][i] · Bp[p][j]` over a strip
/// of `kc` contraction steps, for `i∈0..MR`, `j∈0..NR`.
///
/// Contract (BLIS-conventional packed layout — change here + in `pack_*` together
/// if you pick another):
/// - `ap`: `kc*MR` f64, `ap[p*MR + i] == A[i][p]` — the MR values for step `p`
///   are contiguous (so one k-step loads a dense MR-strip of A).
/// - `bp`: `kc*NR` f64, `bp[p*NR + j] == B[p][j]` — the NR values for step `p`
///   are contiguous (so one k-step loads a dense NR-vector of B).
/// - `c` / `ldc`: element `(i,j)` of the destination block lives at `c[i*ldc + j]`.
///   The kernel **accumulates** (`+=`), so the driver must zero C first and the
///   K-loop accumulates successive KC-panels into the same block.
///
/// Implement scalar first (get indexing right vs. the oracle), then vectorize:
/// hold `MR` accumulator vectors of width `NR`; per step load one B-vector,
/// splat each A value, `mul_add` into the accumulators (MR independent FMA chains
/// = the ILP that fills the pipeline). Needs `use std::simd::{Simd, StdFloat};`.
fn microkernel(kc: usize, ap: &[f64], bp: &[f64], c: &mut [f64], ldc: usize) {
    todo!("Stage 1–2: scalar register-tile accumulation, then std::simd")
}

/// Pack an `mc×kc` block of A (row-major, `a[i*lda + p]`) into the microkernel's
/// `ap` layout: `MR`-strips contiguous per k-step. Handles ragged tails by
/// zero-padding to full `MR` rows. Returns/fills a reusable buffer.
// `&mut Vec` (not `&mut [f64]`) is intentional: this is a growable scratch buffer
// reused across panels via clear()+extend, so we avoid a realloc per panel.
#[allow(clippy::ptr_arg)]
fn pack_a(a: &[f64], lda: usize, mc: usize, kc: usize, out: &mut Vec<f64>) {
    todo!("Stage 3: copy A block into MR-strip-contiguous packed order")
}

/// Pack a `kc×nc` block of B (row-major, `b[p*ldb + j]`) into the microkernel's
/// `bp` layout: `NR`-vectors contiguous per k-step. Zero-pads ragged tails to NR.
#[allow(clippy::ptr_arg)] // growable reuse buffer, same as pack_a
fn pack_b(b: &[f64], ldb: usize, kc: usize, nc: usize, out: &mut Vec<f64>) {
    todo!("Stage 3: copy B block into NR-vector-contiguous packed order")
}

/// Cache-AWARE iterative GEMM — the BLIS "5 loops around the microkernel".
/// Loop order (outer→inner): jc(NC) → pc(KC, pack B) → ic(MC, pack A) →
/// jr(NR) → ir(MR) → microkernel. Zero C, accumulate over the pc (K) loop.
/// Parallelize by partitioning the jc and/or ic loops across threads (static).
///
/// Precondition (start here, add ragged-edge handling last): both operands 2-D
/// and packed; contraction dims agree. Returns `[m, n]`.
pub fn matmul_iterative(a: &TensorData, b: &TensorData) -> TensorData {
    todo!("Stage 4: the 5 loops + packing + microkernel; then Rayon over jc/ic")
}

/// Cache-OBLIVIOUS recursive GEMM (Frigo et al.). Split the **largest** of
/// (m, n, k) in half and recurse; bottom out when the subproblem is small enough
/// to amortize call overhead into a packed microkernel base case (NOT sized to a
/// specific cache — the recursion handles every cache level for you).
///
/// Parallelism: M-splits and N-splits are independent output blocks → run with
/// `rayon::join`. K-splits are a **reduction** into the same C block → keep them
/// serial (or accumulate into a temporary), else you race. Tall-cache assumption
/// (Z = Ω(L²)) and split-largest-dim are what make the miss bound hold.
pub fn matmul_recursive(a: &TensorData, b: &TensorData) -> TensorData {
    todo!("Stage 5: split-largest-dim recursion, packed microkernel base case, \
           rayon::join on M/N splits, serial K")
}
