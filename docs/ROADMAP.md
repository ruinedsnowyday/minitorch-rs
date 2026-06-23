# Roadmap

This document tracks what's been built, what's planned, and where the
scope reaches beyond Sasha Rush's original five-module minitorch.

The original minitorch (`docs/minitorch-original/`) ends at module 4 — a
CNN trained on MNIST. This project follows that path for parity, makes CUDA
the Module 3 GPU backend with a wgpu cross-platform retrofit as Module 3.5
(both specified in `CLAUDE.md`), and then extends into
territory minitorch doesn't cover: modern optimizers, attention,
quantization, and packaging. Phase 1 hits parity; phase 2 modernizes;
phase 3 is aspirational and explicitly optional.

Each module names a deliverable (the visible artifact when it's done) and
the **learn-yourself surface** vs the **AI-free-speed surface** so the
discipline established in CLAUDE.md transfers without restating it.

---

## Status today

| Module | Topic | State |
|---|---|---|
| 0 | Fundamentals (operators, Module tree) | ✅ done |
| 1 | Scalar autodiff (tape, backprop, chain rule) | ✅ done |
| 2 | Tensors + tensor autodiff (forward + backward, all ops) | ✅ done |
| 3 | Efficiency: Rayon + SIMD CPU, CUDA GPU backends | ⬜ next |
| 3.5 | wgpu cross-platform retrofit (port CUDA kernels to WGSL) | ⬜ next, after 3 |
| 3.9 | Precision control (parameterize the scalar type, f32/f64) | ⬜ next, after 3.5 |
| 4 | NN layers + train a CNN on MNIST | ⬜ planned |
| 5+ | Scope expansions — see phase 2 | 🔮 proposed |

Source of truth for "done": `git log`. Source of truth for "what to
implement yourself": `CLAUDE.md`'s ten-item core list.

---

## Phase 1 — Reach minitorch parity

### Module 3 — Efficiency (Rayon + SIMD CPU, CUDA GPU)

**Deliverable.** A `FastOps` backend (Rayon parallelism + `std::simd`
vectorization) and a `CudaOps` backend (cudarc + CUDA C kernels via NVRTC)
that implement the same `TensorOps` trait as `SimpleOps`. Backend choice via
type alias or trait generics. All existing tests pass against any backend
(CUDA tests gated behind `feature = "cuda"`, run on Colab). A benchmark suite
(`cargo bench`, criterion) shows speedup across naive → Rayon → Rayon + SIMD
→ CUDA.

**Learn-yourself.** The parallel + vectorized CPU map/zip/reduce/matmul. The
SIMD lift. The CUDA map/zip/reduce kernels and the tiled matmul. The
trait-generic backend abstraction.

**AI-free-speed.** cudarc device init / NVRTC plumbing / kernel launch
wrappers. SIMD-emission verification tooling. Criterion harness. CI for the
CPU backends (CUDA CI is Colab-only).

**Prerequisite.** Module 2 (done). The `TensorOps` trait already in place.
The student has Colab Pro+ (A100/H100) for the CUDA work.

---

### Module 3.5 — wgpu cross-platform retrofit (side quest)

Already specified in CLAUDE.md. After the CUDA backend works, port the
kernels to WGSL compute shaders behind a wgpu `GpuOps` backend, so the
framework also runs on non-NVIDIA hardware (the student's Mac, any GPU).
Extends the benchmark to naive CPU → Rayon → Rayon + SIMD → CUDA → wgpu.

**Learn-yourself.** The WGSL kernels (map/zip/reduce/tiled matmul). What a
portable API exposes vs CUDA — workgroups vs blocks, `workgroupBarrier()` vs
`__syncthreads()`, no warp/bank-conflict control.

**AI-free-speed.** wgpu device/adapter/queue/pipeline setup. Buffer
management.

**Prerequisite.** Module 3 (need the CUDA kernels to port and to compare
against).

---

### Module 3.9 — Precision control (interlude)

Cross-cutting infra, not a numbered curriculum module: parameterize the
scalar type (`TensorData<T>`, a custom `Scalar` trait) so training can run
`f32` while tests stay `f64`. Sits at the 3.5 → 4 seam — the GPU backend
motivates it (tensor cores are ≤`f32`; `f64` is penalized on every NVIDIA
card), Module 4 training consumes it. Full write-up:
[module3.9-precision.md](module3.9-precision.md).

**Learn-yourself.** The `Scalar` trait surface (derive it from what
`operators.rs` / `tensor_autodiff.rs` actually call). Threading `<T>` through
every core module. The test split (grad-checks pinned to `f64`, `f32`-vs-`f64`
agreement at looser tolerance).

**AI-free-speed.** The mechanical signature churn once the trait is fixed.

**Scope fence.** `f32` + `f64` only. `bf16`/`f16` is Module 5 (mixed
precision); `int8` is Module 7.

**Prerequisite.** Module 3 (the GPU backend is what makes `f64` cost real
performance).

---

### Module 4 — Networks

**Deliverable.** `conv1d`, `conv2d`, max-pool, softmax (numerically
stable), dropout. A LeNet-style CNN trained to >95% on MNIST. A small NLP
sentiment classifier (per original minitorch). Loss/optim built on top of
existing `Tensor::backward` and a basic SGD step.

**Learn-yourself.** The conv forward (im2col or direct), conv backward,
pooling backward (argmax routing), the categorical-cross-entropy fused
softmax+log+NLL trick. The training loop that ties everything together.

**AI-free-speed.** MNIST loader (the IDX file format parser), batching
machinery, progress logging, run-to-run plot generation.

**Prerequisite.** Modules 2 and 3.

---

## Phase 2 — Beyond minitorch (modernization)

These are not in the original five-module sequence but flow naturally from
the foundation. Each is self-contained — pick any subset.

### Module 5 — Modern optimizers and training infrastructure

minitorch only ships SGD. Real training needs more.

**Deliverable.** SGD with momentum, Adam, AdamW. Learning-rate schedulers
(cosine, linear warmup). Gradient clipping. Checkpoint save/load via
safetensors (or a simple bincode format). Mixed precision flag (CPU
no-op; meaningful when paired with module 3's GPU backend).

**Learn-yourself.** The optimizer math (momentum buffers, bias-corrected
moment estimates for Adam). The interaction between gradient clipping
and accumulation. The checkpoint format design — what gets saved, how
shapes are validated on load.

**AI-free-speed.** Serialization glue, file I/O, format detection.

---

### Module 6 — Attention and a small language model

The single biggest gap between "I built a CNN" and "I understand modern
ML" is the attention mechanism. minitorch predates the LLM era; this
module closes it.

**Deliverable.** Token embeddings, positional encodings (sinusoidal +
learned options). Scaled dot-product attention. Multi-head attention.
A transformer block (attention + MLP + layer norm + residuals). A tiny
GPT (~10M params) trained on Tiny Shakespeare. Generation via greedy +
top-k sampling.

**Learn-yourself.** Why attention's gradient flows the way it does (it's
just matmul + softmax — both already in the framework). The masking trick
for causal attention. Layer norm's forward and backward. Why the residual
connection matters for gradient flow.

**AI-free-speed.** Tokenizer (BPE is overkill; character-level or a
pretrained tokenizer crate is fine). Text data loading. Sampling /
generation loop.

**Prerequisite.** Modules 2 (matmul, softmax), 4 (loss machinery), 5
(Adam — SGD won't train a transformer well).

---

### Module 7 — Quantization and inference efficiency

**Deliverable.** Int8 weight quantization (post-training, per-channel).
A quantized matmul kernel (one each for CPU and GPU). A tiny benchmark
showing model size shrink + inference speedup with bounded accuracy loss
on MNIST or the module-6 LM. Optionally: a custom fused kernel via
CubeCL or Rust GPU.

**Learn-yourself.** The quantize/dequantize math. Why per-channel beats
per-tensor. The int8 matmul rules (accumulator dtype, scale arithmetic).
Bank-conflict-free kernel design.

**AI-free-speed.** Calibration data plumbing, accuracy-vs-size reporting,
the bookkeeping that pairs weights with their scales.

**Prerequisite.** Modules 3 (GPU backend exists) and 4 (a trained model
to quantize).

---

## Phase 3 — Aspirational (explicit stretch)

The honest framing on these: each is a real project on its own. Pick at
most one as a finale; the others can stay roadmap items forever and
that's fine.

### Module 8 — Multi-device training

Synchronous data-parallel training across multiple wgpu adapters (or
multi-CPU-process for clarity without GPU dependency). AllReduce
implemented as a compute shader. A demo showing linear-ish speedup on a
2-device matmul or training step. This is where the trait-generic backend
abstraction earns its keep.

### Module 9 — A served model

Load a checkpoint, expose inference over HTTP with `axum` or `actix`,
batch requests, stream tokens for the LM. The "you can put this in front
of a user" exit.

### Module 10 — Comparative profiling

Run the same workload across `SimpleOps`, `FastOps`, `GpuOps`, `CudaOps`,
Burn, and candle. Identify where the framework loses to and beats
production options. Honest writeup; this is what makes the project a
genuine portfolio piece rather than a tutorial completion.

---

## Cross-cutting infrastructure

Worth doing whenever it pays for itself, not gated to any module:

- **Property-based tests** (`proptest`) for tensor invariants — broadcast
  shape, stride math, grad-check equivalences across backends.
- **`cargo bench` discipline** (criterion) for every backend op once
  there's more than one backend to compare.
- **CI matrix** — at minimum `cargo test` on Linux+macOS for the CPU
  backends; a CUDA test path needs a GPU runner (Colab) once module 3 lands,
  and a wgpu path arrives with module 3.5.
- **A `docs/learning-log.md`** capturing concepts mastered per module —
  the artifact that turns this from "code that exists" into "evidence the
  learning happened." `CLAUDE.md` calls this out; it doesn't exist yet.

---

## How to read this roadmap

This is a living document. Things shift. The phase numbers are ordering,
not a contract — if module 6 (attention + LM) sounds more motivating than
module 4 (CNN on MNIST), you can swap them; the prerequisites are honest
about what actually blocks what.

Two rules to keep the discipline:

1. **A module is "done" when its deliverable runs, not when its code
   compiles.** A CNN that trains to 95% on MNIST is done. A trait that
   compiles but has no benchmark proving the backend works is not.

2. **Don't start the next module before the current one's deliverable
   exists.** The temptation to write the transformer before finishing the
   CNN is real; both modules teach more if you finish in order.

The CLAUDE.md ten-item core list still governs what's AI-implementable
within each module. This roadmap names *what* to build; CLAUDE.md governs
*how* to build it.
