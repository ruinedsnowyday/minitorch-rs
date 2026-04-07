# Module 0 — Fundamentals

Original: https://minitorch.github.io/module0/module0/

This module is about getting comfortable with the project, the testing
approach, and two foundational pieces that everything else builds on:
math operators and the Module/Parameter tree. In Rust, this is also
where you learn the basics — functions, traits, enums, closures,
iterators, and testing.

No external ML crates. Everything is hand-written.


## Task 0.1: Operators (`src/operators.rs`)

Implement these as plain functions `fn(f64) -> f64` or
`fn(f64, f64) -> f64`. No structs yet, no traits yet.

**Functions to implement:**

```
mul(a, b)        add(a, b)        neg(a)           id(a)
lt(a, b) -> bool eq(a, b) -> bool max(a, b)        is_close(a, b) -> bool
sigmoid(a)       relu(a)          log(a)           exp(a)          inv(a)
log_back(a, d)   inv_back(a, d)   relu_back(a, d)
```

The `_back` functions compute the derivative of the corresponding
function multiplied by a second argument `d` (the upstream gradient).
You'll use these in Module 1 for backpropagation.

**Rust notes:**
- `f64` has methods like `.ln()`, `.exp()`, `.max()` built in
- For `sigmoid`: use the two-branch form (one for positive, one for
  negative inputs) to avoid overflow. This is a good first encounter
  with Rust's `if` as an expression.
- `is_close` means `(a - b).abs() < 1e-2`
- `id` is a reserved-ish word in some contexts but works fine as a
  function name in Rust


## Task 0.2: Property testing (`tests/test_operators.rs`)

Minitorch uses Hypothesis (Python property testing). The Rust
equivalent is `proptest`. Write property-based tests that verify
mathematical invariants, not just spot checks.

**Tests to write (at minimum):**
- `sigmoid` output is always in (0, 1)
- `sigmoid(x) + sigmoid(-x) ≈ 1.0` (symmetry property)
- `relu(x) >= 0` for all x
- `relu(x) == x` when `x > 0`
- Additive identity: `add(x, 0.0) ≈ x`
- Multiplicative identity: `mul(x, 1.0) ≈ x`
- Commutativity of `add` and `mul`
- `log(exp(x)) ≈ x`
- `exp(log(x)) ≈ x` for `x > 0`

**Rust notes:**
- Add `proptest` to `[dev-dependencies]` in Cargo.toml
- Basic pattern:
  ```rust
  use proptest::prelude::*;

  proptest! {
      #[test]
      fn sigmoid_bounds(x in -100.0f64..100.0) {
          let s = sigmoid(x);
          prop_assert!(s > 0.0 && s < 1.0);
      }
  }
  ```
- This is a good time to learn about Rust's testing conventions:
  `#[cfg(test)]`, `mod tests`, `cargo test`


## Task 0.3: Higher-order functions (`src/operators.rs`)

In Python minitorch, `map`, `zipWith`, and `reduce` are functions
that return functions (closures). In Rust, the equivalent uses
closures and generics with `Fn` trait bounds.

**Functions to implement:**

```rust
// Takes a function, returns a new function that applies it
// to every element of a slice
fn map<F>(f: F) -> impl Fn(&[f64]) -> Vec<f64>
where F: Fn(f64) -> f64

// Takes a function, returns a new function that applies it
// pairwise to two slices
fn zip_with<F>(f: F) -> impl Fn(&[f64], &[f64]) -> Vec<f64>
where F: Fn(f64, f64) -> f64

// Takes a function and initial value, returns a new function
// that folds a slice into a single value
fn reduce<F>(f: F, init: f64) -> impl Fn(&[f64]) -> f64
where F: Fn(f64, f64) -> f64
```

**Then build these using the above:**
- `neg_list(ls)` — negate all elements (use `map`)
- `add_lists(ls1, ls2)` — pairwise add (use `zip_with`)
- `sum(ls)` — sum all elements (use `reduce`)
- `prod(ls)` — product of all elements (use `reduce`)

**Rust notes:**
- This task is your first real encounter with Rust closures and the
  `Fn`/`FnMut`/`FnOnce` trait hierarchy. The key insight: `impl Fn`
  in the return type means "returns some closure" — the caller
  doesn't know the concrete type.
- You could also implement these using iterators (`.iter().map()`,
  `.iter().zip()`, `.iter().fold()`) — the standard Rust way. Worth
  doing BOTH: first the explicit higher-order function versions to
  match minitorch's pedagogy, then refactoring to idiomatic iterator
  style to see the equivalence.
- The Python versions return generators/lists. In Rust, use `Vec<f64>`
  for now. Later modules will need to generalize this.


## Task 0.4: Module tree (`src/module.rs`)

This is the most interesting Rust design challenge in Module 0.
Minitorch's `Module` uses Python's `__setattr__` magic to
auto-detect when a user assigns a `Parameter` or sub-`Module` as
an attribute, and sorts them into internal dictionaries. Rust has
no runtime attribute interception.

**What to implement:**
- `Module` struct with `training: bool` mode
- `Parameter<T>` wrapper that marks a value as learnable
- `train()` / `eval()` — set mode recursively on all descendants
- `named_parameters()` — collect all parameters with dotted path names
  (e.g. `"layer1.weight"`, `"layer1.bias"`, `"layer2.weight"`)
- `parameters()` — collect just the parameter values

**Design choices (think about these before coding):**

Minitorch uses Python's dynamic attribute system. In Rust, you have
several options:

1. **Manual registration** — each Module explicitly returns its
   children and parameters via trait methods. Most transparent.
   ```rust
   trait Module {
       fn children(&self) -> Vec<(&str, &dyn Module)>;
       fn parameters(&self) -> Vec<(&str, &Parameter<f64>)>;
       fn training(&self) -> bool;
       fn set_training(&mut self, mode: bool);
   }
   ```

2. **Derive macro** — write a proc macro that auto-generates the
   trait impl by inspecting struct fields. This is what Burn does.
   Elegant but proc macros are an advanced Rust topic.

3. **HashMap-based** — store children and params in
   `HashMap<String, ...>` like Python does. Loses type safety.

**Recommendation:** start with option 1 (manual registration). It's
verbose but you understand every line. If it bothers you later, option
2 is a great "Rust advanced topics" side quest.

**Rust notes:**
- `named_parameters` requires recursion over a tree. This is a good
  exercise in borrowing: you're returning references into a tree you
  don't own. Consider returning owned `Vec<(String, ...)>` to start.
- `train()`/`eval()` need `&mut self` and must recurse. This is your
  first encounter with recursive mutable borrowing.
- The Python version uses inheritance (`class MyModule(Module)`).
  In Rust, this becomes trait implementation (`impl Module for MyNet`).


## Task 0.5: Visualization

The original uses Streamlit for interactive 2D dataset visualization.
Skip the visualization framework but DO implement the datasets
(`src/datasets.rs`): Simple, Diag, Split, Xor. These are just
functions that generate `Vec<(f64, f64, bool)>` — 2D points with
labels. You'll need them for training in later modules.

**Optional stretch:** use the `plotters` crate to render datasets
to PNG. Nice to have, not required.


## When you're done

You should have:
- [ ] All operators implemented and passing tests
- [ ] Property-based tests via proptest for all operators
- [ ] `map`, `zip_with`, `reduce` working with closures
- [ ] Composite functions (`neg_list`, `add_lists`, `sum`, `prod`)
- [ ] `Module` trait with `train`/`eval`/`named_parameters`
- [ ] At least one concrete Module struct implementing the trait
- [ ] Dataset generators (Simple, Diag, Split, Xor)

**Rust concepts you should now be comfortable with:**
functions, `f64` methods, `if` expressions, `Vec`, slices, `#[test]`,
proptest basics, closures, `Fn`/`FnMut`/`FnOnce`, `impl Trait` return
types, struct definition, trait definition and implementation,
`&self` vs `&mut self`, recursive traversal, `String` vs `&str`.
