# Module 1 — Autodiff

Original: https://minitorch.github.io/module1/module1/

This module builds the first version of automatic differentiation using
only scalar values and functions. It covers the core mechanics:
numerical derivatives, forward pass through a computation graph, the
chain rule, backpropagation, and finally training a small model. This
is the heart of what makes an ML framework an ML framework.

No external autodiff crates. Everything is hand-written.


## Task 1.1: Numerical Derivatives (`src/autodiff.rs`)

Implement a `central_difference` function for computing numerical
approximations of derivatives. This won't be part of the main autodiff
engine — it exists for **testing**: you'll use it to verify that your
analytical derivatives (Task 1.4) are correct.

**Function to implement:**

```rust
/// Computes an approximation to the derivative of `f` with respect
/// to argument `arg` using central differences.
///
/// central_difference(f, &[x0, ..., xn], arg, epsilon)
///   = (f(x0, ..., x_arg + eps, ..., xn) - f(x0, ..., x_arg - eps, ..., xn))
///     / (2 * epsilon)
fn central_difference(
    f: impl Fn(&[f64]) -> f64,
    vals: &[f64],
    arg: usize,
    epsilon: f64,
) -> f64
```

**Rust notes:**
- The Python version uses `*vals` (variadic). In Rust, a slice
  `&[f64]` is the natural replacement.
- You'll need to create modified copies of `vals` with `vals[arg] ±
  epsilon`. This is a good first use of `.to_vec()` to get an owned
  copy of a slice.
- This function is only used in tests, so you could put it behind
  `#[cfg(test)]`, but it's also reasonable to keep it public for
  debugging purposes.


## Task 1.2: Scalar Forward Pass (`src/scalar.rs`)

Implement a `Scalar` type that wraps an `f64` and tracks computation
history for autodiff. Each `Scalar` operation records itself in a
computation graph so that gradients can flow backward later.

**What to implement:**

Define `ScalarFunction` variants (or trait implementations) with
`forward` methods for:

- `mul`
- `inv`
- `neg`
- `sigmoid`
- `relu`
- `exp`
- `lt`
- `eq`

Then wire up the `Scalar` type to support these operations (plus
derived ones):

- `add` (a + b)
- `sub` (a - b)
- `neg` (-a)
- `mul` (a * b)
- `lt` (a < b)
- `gt` (a > b)
- `log`
- `exp`
- `sigmoid`
- `relu`

Each operation should return a new `Scalar` that remembers its parents
and which operation produced it.

**Design choices (think about these before coding):**

The Python minitorch represents `ScalarFunction` as a class hierarchy
with `forward`/`backward` static methods and a `Context` object for
saving values needed during backward. In Rust, you have options:

1. **Enum-based** — a single `ScalarOp` enum with variants for each
   operation. `forward` and `backward` are match arms. Simple,
   everything in one place, but the enum grows with every new op.

2. **Trait-based** — a `ScalarFunction` trait with `forward` and
   `backward` methods, one struct per operation. More extensible,
   closer to the Python design.

3. **Closure-based** — store the backward function as a boxed closure
   captured at forward time. Flexible, but harder to debug.

**Recommendation:** start with option 1 (enum). It's the most
transparent for learning and matches Rust's strength with enums +
pattern matching. You can always refactor to traits later.

**Rust notes:**
- The "Context" pattern from Python (saving values for backward) maps
  naturally to storing data inside enum variants or alongside the
  graph node. Think about what each backward function needs and store
  exactly that.
- For the computation graph, use the arena/tape pattern from CLAUDE.md:
  `Vec<ScalarNode>` with `usize` indices. Each node stores its value,
  the operation that produced it, and indices to its parents.
- Implementing `std::ops::Add`, `std::ops::Mul`, `std::ops::Neg`, etc.
  for `Scalar` will give you natural operator syntax (`a + b` instead
  of `Scalar::add(a, b)`). This is a good exercise in Rust's operator
  overloading traits.
- `lt` and `eq` are tricky: they produce boolean-ish values but need
  to return `Scalar` for the graph to work. The Python version returns
  a Scalar with value 1.0 or 0.0. These have zero gradients
  (constants w.r.t. differentiation).


## Task 1.3: Chain Rule (`src/scalar.rs` or `src/autodiff.rs`)

Implement the chain rule for `Scalar` functions of arbitrary arguments.
Given a node and the derivative of the output with respect to that node
(`d_output`), this should:

1. Call the node's operation's `backward` method with `d_output` and
   any saved context to get local derivatives
2. Pair each local derivative with the corresponding input variable
3. Filter out constants (inputs that don't need gradients)

**Function signature (conceptual):**

```rust
/// Given d_output (the upstream gradient), compute and return
/// pairs of (input_variable, local_gradient) for each non-constant
/// input to this node's operation.
fn chain_rule(&self, d_output: f64) -> Vec<(usize, f64)>
// Returns (parent_index, gradient) pairs
```

**Rust notes:**
- This is where the "what does backward need?" question from Task 1.2
  becomes concrete. For `mul(a, b)`, backward needs the saved values
  of `a` and `b`. For `relu(a)`, backward needs to know whether `a`
  was positive.
- Think carefully about which inputs are "constants" (leaf values with
  no history) vs. "variables" (results of prior computation). Only
  variables need gradient propagation.
- This is a good place to use iterators with `.filter()` and `.map()`.


## Task 1.4: Backpropagation (`src/autodiff.rs`)

Implement topological sort and backpropagation over the computation
graph. These two functions are the engine of reverse-mode autodiff.

**Functions to implement:**

```rust
/// Computes topological order of the computation graph.
/// Starting from `variable` (the output node), returns all
/// non-constant nodes in topological order (dependencies before
/// dependents, or equivalently, reversed post-order DFS).
fn topological_sort(variable: usize, graph: &[ScalarNode]) -> Vec<usize>

/// Runs backpropagation on the computation graph.
/// `variable` is the output node, `deriv` is its derivative (usually 1.0).
/// Walks the graph in reverse topological order, accumulating
/// gradients into each leaf node via the chain rule.
fn backpropagate(variable: usize, deriv: f64, graph: &mut [ScalarNode])
```

Also add `backward` methods to each of your `ScalarFunction` /
`ScalarOp` variants:

| Operation | backward(d_output, saved) |
|-----------|---------------------------|
| `mul(a,b)` | `(d * b, d * a)` |
| `inv(a)` | `(-d / (a * a),)` |
| `neg(a)` | `(-d,)` |
| `sigmoid(a)` | `(d * sig * (1 - sig),)` where `sig = sigmoid(a)` |
| `relu(a)` | `(d * (1.0 if a > 0 else 0.0),)` |
| `exp(a)` | `(d * exp(a),)` |
| `log(a)` | `(d / a,)` |
| `lt`, `eq` | `(0.0, 0.0)` — no gradient through comparisons |

**Rust notes:**
- Topological sort is DFS with a visited set, collecting nodes in
  post-order then reversing. A `HashSet<usize>` for visited and a
  `Vec<usize>` for the result is straightforward.
- For backpropagate: walk the topo-sorted list in reverse. For each
  node, call `chain_rule` with the accumulated gradient, then add each
  returned gradient to the corresponding parent's accumulator.
- Key subtlety: a node that feeds into multiple operations must
  **accumulate** (add) gradients from all consumers, not replace.
  Initialize all gradient accumulators to 0.0 and use `+=`.
- The arena pattern makes this clean: `graph[i]` gives you the node,
  and parent indices give you direct access to parent nodes.
- Consider using a `Vec<f64>` of the same length as `graph` for
  gradient accumulators, indexed by node index.
- Use your `central_difference` from Task 1.1 to test: for any
  function, the analytical gradient (backprop) should be close to the
  numerical gradient (central difference). This is called a
  **gradient check** and is the standard way to verify autodiff.


## Task 1.5: Training

With autodiff working, train a scalar-based neural network on the
datasets from Module 0 (Simple, Diag, Split, Xor).

**What to implement:**

- A `Linear` module: `forward(x) = w * x + b` where `w` and `b` are
  parameters (using your Module system from Module 0)
- A small `Network` module: two `Linear` layers with a nonlinearity
  (sigmoid or relu) in between
- A training loop:
  1. Forward pass: compute predictions for all points
  2. Loss: simple loss function (e.g., binary cross-entropy or hinge)
  3. Backward pass: call `backward()` on the loss
  4. Update: SGD — `param -= learning_rate * param.grad`
  5. Zero gradients, repeat

**Suggested config to start:**

```
dataset: Simple
points: 50
hidden_size: 2
learning_rate: 0.5
```

Then work up to harder datasets (Xor needs `hidden_size >= 10`).

**Rust notes:**
- This is where Module 0's `Module` trait and this module's `Scalar`
  autodiff come together. Your `Linear` struct implements `Module` and
  its parameters are `Scalar` values tracked in the computation graph.
- The training loop will look different from Python because of
  ownership. Each forward pass creates a new computation graph (new
  arena). After backward, you extract gradients, update parameters,
  and drop the graph.
- Start with the simplest thing that works. Don't worry about
  batching, momentum, or fancy optimizers. Plain SGD on individual
  points is fine for these toy datasets.
- Print loss every N epochs to verify convergence.


## When you're done

You should have:
- [ ] `central_difference` implemented and tested
- [ ] `Scalar` type with forward pass tracking computation history
- [ ] `ScalarFunction`/`ScalarOp` for all listed operations
- [ ] Chain rule correctly computing local derivatives
- [ ] Topological sort over the computation graph
- [ ] `backpropagate` accumulating gradients through the graph
- [ ] Gradient checks passing (analytical vs. numerical)
- [ ] `Linear` and `Network` modules using `Scalar` parameters
- [ ] Successful training on Simple dataset
- [ ] Successful training on Xor dataset (stretch)

**Rust concepts you should now be comfortable with:**
enums with data, pattern matching, arena allocation (`Vec<T>` + `usize`
indices), `HashSet`, DFS/recursion with mutable state, operator
overloading (`std::ops` traits), closures that capture state, the
difference between owned and borrowed data in graph structures,
`Box<dyn Fn>` vs `impl Fn`.
