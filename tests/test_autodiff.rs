use minitorch_rs::autodiff::{central_difference, ScalarGraph, ScalarOp};
use minitorch_rs::operators;
use proptest::prelude::*;

/// Helper: assert two f64s are close (within 1e-2).
fn assert_close(a: f64, b: f64) {
    assert!(
        operators::is_close(a, b),
        "assert_close failed: {a} vs {b} (diff = {})",
        (a - b).abs()
    );
}

/// Strategy for small floats that avoid extreme values.
fn small_floats() -> impl Strategy<Value = f64> {
    -100.0f64..100.0
}

// ===================================================================
// Task 1.1 — Central difference
// ===================================================================

#[test]
fn test_central_diff_id() {
    let d = central_difference(|v| operators::id(v[0]), &[5.0], 0, 1e-6);
    assert_close(d, 1.0);
}

#[test]
fn test_central_diff_add() {
    let d = central_difference(|v| operators::add(v[0], v[1]), &[5.0, 10.0], 0, 1e-6);
    assert_close(d, 1.0);
}

#[test]
fn test_central_diff_mul_arg0() {
    let d = central_difference(|v| operators::mul(v[0], v[1]), &[5.0, 10.0], 0, 1e-6);
    assert_close(d, 10.0);
}

#[test]
fn test_central_diff_mul_arg1() {
    let d = central_difference(|v| operators::mul(v[0], v[1]), &[5.0, 10.0], 1, 1e-6);
    assert_close(d, 5.0);
}

#[test]
fn test_central_diff_exp() {
    let d = central_difference(|v| operators::exp(v[0]), &[2.0], 0, 1e-6);
    assert_close(d, operators::exp(2.0));
}

proptest! {
    #[test]
    fn test_central_diff_log(x in 0.1f64..100.0) {
        let d = central_difference(|v| operators::log(v[0]), &[x], 0, 1e-6);
        assert_close(d, 1.0 / x);
    }

    #[test]
    fn test_central_diff_sigmoid(x in small_floats()) {
        let d = central_difference(|v| operators::sigmoid(v[0]), &[x], 0, 1e-6);
        let s = operators::sigmoid(x);
        assert_close(d, s * (1.0 - s));
    }

    #[test]
    fn test_central_diff_relu(x in small_floats()) {
        let d = central_difference(|v| operators::relu(v[0]), &[x], 0, 1e-6);
        if x > 1e-4 {
            assert_close(d, 1.0);
        } else if x < -1e-4 {
            assert_close(d, 0.0);
        }
        // skip near zero — derivative is discontinuous
    }
}

// ===================================================================
// Task 1.2 — Forward pass: ScalarGraph operations produce correct values
// ===================================================================

/// Helper: build a graph with two leaves, apply a binary op, return output value.
fn eval_binary(a: f64, b: f64, op: ScalarOp) -> f64 {
    let mut g = ScalarGraph::new();
    let x = g.add_leaf(a);
    let y = g.add_leaf(b);
    let z = g.apply(op, vec![x, y]);
    g.get_node(z).out
}

/// Helper: build a graph with one leaf, apply a unary op, return output value.
fn eval_unary(a: f64, op: ScalarOp) -> f64 {
    let mut g = ScalarGraph::new();
    let x = g.add_leaf(a);
    let z = g.apply(op, vec![x]);
    g.get_node(z).out
}

proptest! {
    #[test]
    fn test_forward_add(a in small_floats(), b in small_floats()) {
        assert_close(eval_binary(a, b, ScalarOp::Add(a, b)), a + b);
    }

    #[test]
    fn test_forward_mul(a in small_floats(), b in small_floats()) {
        assert_close(eval_binary(a, b, ScalarOp::Mul(a, b)), a * b);
    }

    #[test]
    fn test_forward_neg(a in small_floats()) {
        assert_close(eval_unary(a, ScalarOp::Neg(a)), -a);
    }

    #[test]
    fn test_forward_inv(a in 0.01f64..100.0) {
        assert_close(eval_unary(a, ScalarOp::Inv(a)), 1.0 / a);
    }

    #[test]
    fn test_forward_sigmoid(a in small_floats()) {
        assert_close(eval_unary(a, ScalarOp::Sigmoid(a)), operators::sigmoid(a));
    }

    #[test]
    fn test_forward_relu(a in small_floats()) {
        assert_close(eval_unary(a, ScalarOp::ReLU(a)), operators::relu(a));
    }

    #[test]
    fn test_forward_exp(a in -10.0f64..10.0) {
        assert_close(eval_unary(a, ScalarOp::Exp(a)), operators::exp(a));
    }

    #[test]
    fn test_forward_log(a in 0.01f64..100.0) {
        assert_close(eval_unary(a, ScalarOp::Log(a)), operators::log(a));
    }

    #[test]
    fn test_forward_lt(a in small_floats(), b in small_floats()) {
        assert_close(eval_binary(a, b, ScalarOp::Lt(a, b)), operators::lt(a, b));
    }

    #[test]
    fn test_forward_eq(a in small_floats(), b in small_floats()) {
        assert_close(eval_binary(a, b, ScalarOp::Eq(a, b)), operators::eq(a, b));
    }
}

// ===================================================================
// Task 1.4 — Backpropagation: gradient checks
// Each test builds a small graph, runs backprop, and compares the
// analytical gradient against central_difference.
// ===================================================================

/// Helper: gradient check for a unary function.
/// Builds graph: x -> op -> output, backprops, compares grad(x) to central diff.
fn grad_check_unary(val: f64, make_op: impl Fn(f64) -> ScalarOp, f: impl Fn(&[f64]) -> f64) {
    let mut g = ScalarGraph::new();
    let x = g.add_leaf(val);
    let z = g.apply(make_op(val), vec![x]);
    g.backpropagate(z, 1.0);

    let analytical = g.get_node(x).gradient;
    let numerical = central_difference(&f, &[val], 0, 1e-5);
    assert!(
        (analytical - numerical).abs() < 1e-2,
        "grad check failed: analytical={analytical}, numerical={numerical}, val={val}"
    );
}

/// Helper: gradient check for a binary function.
/// Checks gradient w.r.t. both arguments.
fn grad_check_binary(
    a: f64,
    b: f64,
    make_op: impl Fn(f64, f64) -> ScalarOp,
    f: impl Fn(&[f64]) -> f64,
) {
    // Check grad w.r.t. first arg
    let mut g = ScalarGraph::new();
    let x = g.add_leaf(a);
    let y = g.add_leaf(b);
    let z = g.apply(make_op(a, b), vec![x, y]);
    g.backpropagate(z, 1.0);

    let grad_a = g.get_node(x).gradient;
    let grad_b = g.get_node(y).gradient;

    let num_a = central_difference(&f, &[a, b], 0, 1e-5);
    let num_b = central_difference(&f, &[a, b], 1, 1e-5);

    assert!(
        (grad_a - num_a).abs() < 1e-2,
        "grad check (arg 0) failed: analytical={grad_a}, numerical={num_a}, a={a}, b={b}"
    );
    assert!(
        (grad_b - num_b).abs() < 1e-2,
        "grad check (arg 1) failed: analytical={grad_b}, numerical={num_b}, a={a}, b={b}"
    );
}

proptest! {
    #[test]
    fn test_grad_add(a in small_floats(), b in small_floats()) {
        grad_check_binary(a, b, ScalarOp::Add, |v| v[0] + v[1]);
    }

    #[test]
    fn test_grad_mul(a in small_floats(), b in small_floats()) {
        grad_check_binary(a, b, ScalarOp::Mul, |v| v[0] * v[1]);
    }

    #[test]
    fn test_grad_neg(a in small_floats()) {
        grad_check_unary(a, ScalarOp::Neg, |v| -v[0]);
    }

    #[test]
    fn test_grad_inv(a in 0.1f64..100.0) {
        grad_check_unary(a, ScalarOp::Inv, |v| 1.0 / v[0]);
    }

    #[test]
    fn test_grad_sigmoid(a in small_floats()) {
        grad_check_unary(a, ScalarOp::Sigmoid, |v| operators::sigmoid(v[0]));
    }

    #[test]
    fn test_grad_relu(a in 0.1f64..100.0) {
        // Only test positive region; derivative at 0 is not well-defined
        grad_check_unary(a, ScalarOp::ReLU, |v| operators::relu(v[0]));
    }

    #[test]
    fn test_grad_exp(a in -10.0f64..10.0) {
        grad_check_unary(a, ScalarOp::Exp, |v| operators::exp(v[0]));
    }

    #[test]
    fn test_grad_log(a in 0.1f64..100.0) {
        grad_check_unary(a, ScalarOp::Log, |v| operators::log(v[0]));
    }
}

// ===================================================================
// Compound expression gradient checks — tests that backprop
// correctly accumulates through multiple operations.
// ===================================================================

#[test]
fn test_grad_compound_mul_add() {
    // f(x, y) = x * y + x
    // df/dx = y + 1, df/dy = x
    let (a, b) = (3.0, 5.0);
    let mut g = ScalarGraph::new();
    let x = g.add_leaf(a);
    let y = g.add_leaf(b);
    let xy = g.apply(ScalarOp::Mul(a, b), vec![x, y]);
    let out = g.apply(ScalarOp::Add(a * b, a), vec![xy, x]);
    g.backpropagate(out, 1.0);

    assert_close(g.get_node(x).gradient, b + 1.0); // df/dx = y + 1
    assert_close(g.get_node(y).gradient, a);        // df/dy = x
}

#[test]
fn test_grad_diamond() {
    // f(x) = x * x  (same node feeds both inputs)
    // df/dx = 2x
    let a = 4.0;
    let mut g = ScalarGraph::new();
    let x = g.add_leaf(a);
    let z = g.apply(ScalarOp::Mul(a, a), vec![x, x]);
    g.backpropagate(z, 1.0);

    assert_close(g.get_node(x).gradient, 2.0 * a);
}

#[test]
fn test_grad_chain() {
    // f(x) = sigmoid(relu(x)) for x > 0
    let a = 2.0;
    let mut g = ScalarGraph::new();
    let x = g.add_leaf(a);
    let r = g.apply(ScalarOp::ReLU(a), vec![x]);
    let relu_val = g.get_node(r).out;
    let s = g.apply(ScalarOp::Sigmoid(relu_val), vec![r]);
    g.backpropagate(s, 1.0);

    let numerical = central_difference(
        |v| operators::sigmoid(operators::relu(v[0])),
        &[a],
        0,
        1e-5,
    );
    assert!(
        (g.get_node(x).gradient - numerical).abs() < 1e-2,
        "chain grad check failed: analytical={}, numerical={numerical}",
        g.get_node(x).gradient
    );
}
