# Module 3.9 — Precision control (the 3.5 → 4 interlude)

Parameterize the scalar type so training can run `f32` while tests stay `f64`.

Cross-doc: [module3.md](module3.md) (efficiency) · [ROADMAP.md](ROADMAP.md) ·
[CLAUDE.md](../CLAUDE.md) (the no-implementation contract still holds — this
doc maps the refactor; you write it).

## Contents

- [Why this is its own step](#why-this-is-its-own-step)
- [Where it sits, and why here](#where-it-sits-and-why-here)
- [Decision 1 — compile-time generic vs runtime dtype](#decision-1--compile-time-generic-vs-runtime-dtype)
- [Decision 2 — custom `Scalar` trait vs `num_traits::Float`](#decision-2--custom-scalar-trait-vs-num_traitsfloat)
- [Where `<T>` threads](#where-t-threads)
- [The literal tax](#the-literal-tax)
- [Test strategy — the actual reason this is a type parameter](#test-strategy--the-actual-reason-this-is-a-type-parameter)
- [Scope: in and out](#scope-in-and-out)
- [Your inventory exercise](#your-inventory-exercise)
- [Definition of done](#definition-of-done)

## Why this is its own step

Through Module 3 everything is `f64`: `Rc<Vec<f64>>` storage, `Fn(f64) -> f64`
closures, `f64` gradients. That was the right call — finite-difference
grad-checks are clean in `f64`, so precision loss never masks a backprop bug.

But `f64` is not where ML lives, and the cost is concrete on your hardware:

- On an A100/H100, `f64` runs at ~½ the `f32` rate, and **tensor cores don't
  do `f64` at all** — they're `tf32`/`bf16`/`f16` only. Staying `f64` leaves
  the bulk of the GPU's FLOPs untouched.
- If Colab ever hands you a T4/L4 instead of an A100, `f64` drops to ~1/32 of
  `f32` — a cliff, not a slope.
- It ties back to the bandwidth-bound ops from Module 3: `f64 → f32` halves
  the bytes per element, so it roughly **doubles** every memory-bound op
  (GEMV, elementwise) for free, before any compute change.

So the scalar type has to become a parameter. This is cross-cutting
infrastructure (like the proptest and CI items in the ROADMAP), not a
numbered curriculum module — but it has a sharp trigger, which is what lets
us place it.

## Where it sits, and why here

The **3.5 → 4 seam**, as one focused PR — not folded into Module 4.

- **Module 3 / 3.5 motivates it.** The GPU backend is the first place `f64`
  actively costs you (tensor cores, the `f64` throughput penalty).
- **Module 4 consumes it.** Nobody trains a CNN in `f64`; you want `f32` to be
  the *default* the moment the training loop exists.
- **Doing it earlier** (inside Module 3) is premature — the `<T>` bound would
  smear across every signature while you're learning Rayon/CUDA, for zero
  conceptual payoff there.
- **Doing it inside Module 4** muddies an already-large module (conv + pool +
  softmax + training loop) with a project-wide refactor.
- It must **precede Module 5's mixed precision**, which is a harder, different
  thing (`bf16` compute + `f32` master weights) built *on top of* a working
  dtype parameter.

## Decision 1 — compile-time generic vs runtime dtype

| | Compile-time `TensorData<T>` | Runtime `Dtype` enum + byte storage |
|---|---|---|
| Model | dfdx / burn / candle | PyTorch / numpy |
| Dispatch | monomorphized, zero runtime cost | match on dtype everywhere |
| Safety | compile-time type checking | runtime checks |
| Flexibility | one dtype per tensor, fixed at compile | mixed dtypes at runtime |
| Rust lesson | trait bounds, generics, monomorphization | dynamic dispatch, type erasure |

**Recommendation: compile-time generic.** It's a genuine Rust-generics
exercise (a CLAUDE.md learning goal), it has no dispatch overhead, and you
don't need runtime-mixed dtypes for anything on the roadmap before M7. The
runtime-enum approach is a bigger machine for flexibility you won't use yet.

## Decision 2 — custom `Scalar` trait vs `num_traits::Float`

`num_traits::Float` is off-the-shelf and gives you `exp/ln/sqrt/powf` plus
`T::zero()/one()`. Tempting. But your roadmap ends in `bf16` (M5) and `int8`
(M7), and `num_traits::Float` is **not cleanly implemented for `half::bf16` /
`half::f16`**. A hand-rolled trait is the foundation those need — it's exactly
why burn defines its own `Element`/`Float` and candle its `WithDType` rather
than leaning on `num_traits`.

**Recommendation: a custom `Scalar` trait.** Defining it is also the lesson —
it forces you to inventory what operations the framework actually demands of a
number. The shape (signature only — the full method set is your inventory
exercise below):

```rust
// what the framework needs of a scalar. NOT the full list — you derive that.
pub trait Scalar: Copy + Add<Output = Self> + Mul<Output = Self> + /* Sub, Div, PartialOrd … */ {
    fn zero() -> Self;
    fn one() -> Self;
    fn exp(self) -> Self;          // … plus whatever operators.rs actually calls
}
```

## Where `<T>` threads

This is the whole reason to map it before committing: the bound touches
*every* core module. Each `f64` becomes `T`.

| Surface (file) | `f64` today → generic |
|---|---|
| `tensor_data.rs` | `storage: Rc<Vec<f64>>` → `Rc<Vec<T>>`; `get -> T`; `zeros`/`ones` need `T::zero()`/`T::one()` |
| `tensor_ops.rs` | trait becomes `TensorOps<T: Scalar>`; closures `Fn(T) -> T`; `reduce`'s `init: T`; the parallel/GPU backends add `T: Scalar + Send + Sync` (same bound that bit you in review) |
| `operators.rs` | `sigmoid<T: Scalar>(x: T) -> T` etc., calling into the `Scalar` surface (`exp`, `one`, …); derivatives too |
| `tensor_autodiff.rs` | `TensorGraph<T>`; nodes hold `TensorData<T>`; every backward rule expressed in `T` |
| `tensor.rs` | `Tensor<T>`; `item() -> T`; all op methods + operator overloads become `impl<T: Scalar>` |
| `tests/` | grad-checks **stay pinned to `Tensor<f64>`**; add `f32`-vs-`f64` agreement tests at looser tolerance |

Signature shape for the two anchors:

```rust
pub struct TensorData<T: Scalar> { storage: Rc<Vec<T>>, shape: Vec<usize>, strides: Vec<usize>, offset: usize }

pub trait TensorOps<T: Scalar> {
    fn map(input: &TensorData<T>, f: impl Fn(T) -> T + Sync) -> TensorData<T>;
    fn reduce(input: &TensorData<T>, f: impl Fn(T, T) -> T + Sync, init: T, /* … */) -> TensorData<T>;
    // zip, matmul …
}
```

## The literal tax (mostly dissolvable)

The friction people remember: in generic code, numeric literals stop being
literals.

- `0.0` → `T::zero()`
- `1.0` → `T::one()`
- arbitrary constants: `0.5` → `T::from(0.5).unwrap()` — peppered through every
  operator and backward rule.

But the tax is mostly an illusion if you design `Scalar` right. Almost every
literal in core ops is from a tiny fixed set — `0`, `1`, `0.5`, `2`, maybe
`ln 2` and `epsilon`. Make those **associated consts** on the trait. Then each
literal is written *once per concrete type*, in the `impl` block where the type
is known and a float literal is just a float literal — no tax — and generic
code never writes a literal at all:

```rust
trait Number: Copy {
    const HALF: Self;                         // compile-time constant, per type
}
impl Number for f64 { const HALF: f64 = 0.5; } // literal is free here — concrete type
impl Number for f32 { const HALF: f32 = 0.5; }

fn blend<T: Number + Mul<Output = T>>(x: T) -> T { x * T::HALF }  // no literal in generic code
```

The tax isn't fought — it's *moved* to the one place literals are free. This is
also **faster on the hot path**: `T::from(0.5).unwrap()` is a runtime
conversion plus a branch the optimizer only *usually* hoists out of a loop; an
associated const is materialized at compile time, strictly better in the exact
`map`/`reduce`/matmul kernels Module 3 optimized.

For the rare genuinely-arbitrary constant, bind it once instead of at every
site: `let c = T::from(0.123).unwrap();` at the top of the function (out of the
inner loop) — the nalgebra/ndarray idiom.

**Can a macro fight it?** Only the last 5%. A `macro_rules!` like
`s!(0.123)` → `T::from(0.123).unwrap()` shortens typing but doesn't remove the
runtime conversion, and it's clunky in generics (it must take the type, or
assume a conventionally-named `T` is in scope — fragile). A proc macro that
rewrites literals is the heavy hammer (advanced, opaque, a build dependency) —
don't reach there; it still loses to a compile-time const for the recurring
values. Design around the trait's consts; treat a macro as optional polish on
the leftovers, not the strategy.

This is also why the custom `Scalar` trait beats `num_traits::Float` here:
`num_traits` gives you `zero()`/`one()` but not *your own* named consts, and it
won't give you clean `bf16`/`f16` consts in Module 5 — `impl Scalar for bf16 {
const HALF: bf16 = … }` is yours to define.

The remaining placement argument still holds: even with the consts approach,
parameterizing every signature is churn you don't want to pay in Module 3 for
no `f32` payoff. That asymmetry is the whole case for the 3.5 → 4 seam.

## Test strategy — the actual reason this is a type parameter

If you only ever wanted `f32`, you'd change the typedef and move on. You make
it a *parameter* so both precisions coexist:

- Keep the finite-difference grad-check helper instantiated at **`f64`**.
  `f32` finite differences are too noisy — the perturbation `h` and the
  rounding error collide, and a correct gradient can fail the check. Your
  correctness net stays `f64`.
- Add a **cross-precision agreement test**: build the same small op (or tiny
  net) as `Tensor<f32>` and `Tensor<f64>`, run forward + backward, assert the
  results agree within `f32` tolerance (~`1e-5` relative, not bit-exact —
  `f32` reassociation under SIMD/parallel reduce forbids bit-exactness, same
  lesson as Module 3.1).
- Run the existing `TensorOps` suite against `Tensor<f32>` too; most exact
  assertions become approximate ones at `f32` tolerance.

## Scope: in and out

**In:** `f32` and `f64` — the two IEEE floats, one `Scalar` trait, the
parameterization across all core modules, the test split above.

**Out (later modules):**
- `f16` / `bf16` — not "just another float." They need `f32` accumulation,
  the `half` crate for CPU, overflow/underflow care. Belongs with **Module 5**
  (mixed precision) and tensor cores.
- `int8` quantization — **Module 7**. Different math entirely (scales,
  zero-points, accumulator dtype).
- Mixed precision (compute in low, master weights in `f32`) — **Module 5**,
  built on top of this parameter.

Keep this interlude to the two floats. Resisting scope creep here is part of
the exercise.

## Your inventory exercise

Don't accept a `Scalar` trait handed to you — derive its surface:

1. `grep` `operators.rs` and `tensor_autodiff.rs` for every operation applied
   to an `f64`: arithmetic, `.exp()`, `.ln()`, `.max()`, comparisons, every
   literal constant.
2. That set *is* your `Scalar` trait's method list. Anything the framework
   never calls doesn't belong in the bound.
3. Decide which come free from std traits (`Add`, `Mul`, `PartialOrd`, `Copy`)
   and which you declare explicitly (`zero`, `one`, `exp`, `from`).

The trait you arrive at is a precise statement of what your framework assumes
a "number" can do. That's the artifact worth having.

## Definition of done

- [ ] `Scalar` trait defined, implemented for `f32` and `f64`.
- [ ] All core modules parameterized; project compiles for both
      `Tensor<f32>` and `Tensor<f64>`.
- [ ] `cargo clippy -- -D warnings` clean.
- [ ] Existing `TensorOps` / tensor / backward suites pass against `f64`
      (exact) and `f32` (approximate).
- [ ] Grad-check helper still runs at `f64` and passes.
- [ ] One cross-precision agreement test (`f32` vs `f64`) passes.
- [ ] `FastOps` / `CudaOps` still satisfy their bounds with `T: Scalar + Send
      + Sync`.

When this is done, Module 4 trains in `f32` by default, and your tests still
prove correctness in `f64`. That dual track is the payoff.
