use minitorch_rs::autodiff::{ScalarGraph, ScalarOp, central_difference};
use minitorch_rs::loss::{bce, mse};

const EPS: f64 = 1e-9;

// operators::log and operators::log_back BOTH add a 1e-6 nudge for numerical
// stability: log(a) = (a + 1e-6).ln() and log_back uses 1/(a + 1e-6). So any
// "expected" value that flows through log — forward OR gradient — drifts from
// the textbook formula by ~1e-6. Compare those against textbook values at this
// looser tolerance, not the tight EPS. (The autodiff-vs-central-difference
// check is what actually proves backward correctness; the textbook-value
// asserts are a secondary sanity check and must tolerate the nudge.)
const LOG_TOL: f64 = 1e-5;

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < EPS
}

fn close_with(a: f64, b: f64, eps: f64) -> bool {
    (a - b).abs() < eps
}

// MSE forward value: (pred - label)^2
#[test]
fn test_mse_forward_value() {
    let mut graph = ScalarGraph::new();
    let pred_id = graph.add_leaf(2.0);
    let loss_id = mse(&mut graph, pred_id, 3.0);
    // (2 - 3)^2 = 1
    assert!(close(graph.get_node(loss_id).out, 1.0));
}

// At the minimum (pred == label) the gradient should be exactly zero.
// This also exercises the same-node Mul trick: Mul(0, 0) backward produces
// [0 * d, 0 * d] = [0, 0], which must accumulate to 0 on diff, not double.
#[test]
fn test_mse_zero_gradient_at_minimum() {
    let mut graph = ScalarGraph::new();
    let pred_id = graph.add_leaf(3.0);
    let loss_id = mse(&mut graph, pred_id, 3.0);
    assert!(close(graph.get_node(loss_id).out, 0.0));

    graph.backpropagate(loss_id, 1.0);
    assert!(close(graph.get_node(pred_id).gradient, 0.0));
}

// Compare the autodiff gradient against a central-difference perturbation
// of the same forward function. If they agree, the chain of ops in mse()
// is wired up correctly.
#[test]
fn test_mse_gradient_matches_central_difference() {
    let label = 3.0;

    // autodiff path
    let mut graph = ScalarGraph::new();
    let pred_id = graph.add_leaf(2.0);
    let loss_id = mse(&mut graph, pred_id, label);
    graph.backpropagate(loss_id, 1.0);
    let autodiff_grad = graph.get_node(pred_id).gradient;

    // numerical path: treat mse as a function of pred
    let f = |params: &[f64]| -> f64 {
        let mut g = ScalarGraph::new();
        let p = g.add_leaf(params[0]);
        let l = mse(&mut g, p, label);
        g.get_node(l).out
    };
    let numerical_grad = central_difference(f, &[2.0], 0, 1e-6);

    assert!((autodiff_grad - numerical_grad).abs() < 1e-5);
    // analytical sanity: d/dp [(p-3)^2] at p=2 is 2*(2-3) = -2
    assert!(close(autodiff_grad, -2.0));
}

// Gradients must reach the inputs that produced pred, not just pred itself.
// Setup: pred = a * b, then mse(pred, label).
//   d(loss)/d(pred) = 2*(pred - label)
//   d(loss)/d(a)    = d(loss)/d(pred) * b
//   d(loss)/d(b)    = d(loss)/d(pred) * a
#[test]
fn test_mse_gradient_chains_through_predecessors() {
    let label = 10.0;
    let a_val = 2.0;
    let b_val = 3.0;

    let mut graph = ScalarGraph::new();
    let a = graph.add_leaf(a_val);
    let b = graph.add_leaf(b_val);
    let pred = graph.apply(ScalarOp::Mul(a_val, b_val), vec![a, b]);
    let loss_id = mse(&mut graph, pred, label);
    graph.backpropagate(loss_id, 1.0);

    // pred = 6, diff = -4, loss = 16
    assert!(close(graph.get_node(loss_id).out, 16.0));
    // d(loss)/d(pred) = 2 * (6 - 10) = -8
    assert!(close(graph.get_node(pred).gradient, -8.0));
    // d(loss)/d(a) = -8 * b = -24
    assert!(close(graph.get_node(a).gradient, -24.0));
    // d(loss)/d(b) = -8 * a = -16
    assert!(close(graph.get_node(b).gradient, -16.0));
}

// ===================================================================
// BCE: loss = -[t * log(p) + (1 - t) * log(1 - p)]
// ===================================================================

// At p=0.5, t=1: loss = -log(0.5) = log(2) ≈ 0.6931472
#[test]
fn test_bce_forward_value() {
    let mut graph = ScalarGraph::new();
    let pred_id = graph.add_leaf(0.5);
    let loss_id = bce(&mut graph, pred_id, 1.0);
    assert!(close_with(
        graph.get_node(loss_id).out,
        (2.0_f64).ln(),
        LOG_TOL
    ));
}

// BCE is symmetric under (p, t) ↔ (1-p, 1-t):
//   loss(0.7, 1) == loss(0.3, 0)
// because swapping flips which term contributes but produces the same number.
#[test]
fn test_bce_symmetry_under_flip() {
    let mut g1 = ScalarGraph::new();
    let p1 = g1.add_leaf(0.7);
    let l1 = bce(&mut g1, p1, 1.0);
    let loss1 = g1.get_node(l1).out;

    let mut g2 = ScalarGraph::new();
    let p2 = g2.add_leaf(0.3);
    let l2 = bce(&mut g2, p2, 0.0);
    let loss2 = g2.get_node(l2).out;

    // symmetry itself is exact: same nudged log used on both sides
    assert!(close(loss1, loss2));
    // textbook value uses un-nudged log, so loosen for this check
    assert!(close_with(loss1, -(0.7_f64).ln(), LOG_TOL));
}

// Analytical gradient: d(loss)/d(p) = (p - t) / (p * (1 - p))
// At p=0.6, t=1: (0.6 - 1) / (0.6 * 0.4) = -0.4 / 0.24 ≈ -1.6667
#[test]
fn test_bce_gradient_matches_central_difference() {
    let label = 1.0;

    // autodiff path
    let mut graph = ScalarGraph::new();
    let pred_id = graph.add_leaf(0.6);
    let loss_id = bce(&mut graph, pred_id, label);
    graph.backpropagate(loss_id, 1.0);
    let autodiff_grad = graph.get_node(pred_id).gradient;

    // numerical path
    let f = |params: &[f64]| -> f64 {
        let mut g = ScalarGraph::new();
        let p = g.add_leaf(params[0]);
        let l = bce(&mut g, p, label);
        g.get_node(l).out
    };
    let numerical_grad = central_difference(f, &[0.6], 0, 1e-6);

    assert!((autodiff_grad - numerical_grad).abs() < 1e-5);
    // analytical sanity: (0.6 - 1) / (0.6 * 0.4) = -5/3, drifted by the
    // log_back nudge so checked at LOG_TOL rather than tight EPS.
    assert!((autodiff_grad - (-5.0 / 3.0)).abs() < LOG_TOL);
}

// Gradients flow back through predecessors of pred.
// Setup: pred = a * b with a=0.5, b=0.8 → pred=0.4, label=1.
//   d(loss)/d(pred) = (0.4 - 1) / (0.4 * 0.6) = -2.5
//   d(loss)/d(a)    = -2.5 * b = -2.0
//   d(loss)/d(b)    = -2.5 * a = -1.25
#[test]
fn test_bce_gradient_chains_through_predecessors() {
    let label = 1.0;
    let a_val = 0.5;
    let b_val = 0.8;

    let mut graph = ScalarGraph::new();
    let a = graph.add_leaf(a_val);
    let b = graph.add_leaf(b_val);
    let pred = graph.apply(ScalarOp::Mul(a_val, b_val), vec![a, b]);
    let loss_id = bce(&mut graph, pred, label);
    graph.backpropagate(loss_id, 1.0);

    // forward sanity: pred = 0.4, loss = -log(0.4)
    assert!(close(graph.get_node(pred).out, 0.4));
    assert!(close_with(
        graph.get_node(loss_id).out,
        -(0.4_f64).ln(),
        LOG_TOL
    ));

    // gradients — textbook values, drifted by the log_back nudge, so
    // checked at LOG_TOL rather than tight EPS.
    assert!(close_with(graph.get_node(pred).gradient, -2.5, LOG_TOL));
    assert!(close_with(graph.get_node(a).gradient, -2.0, LOG_TOL));
    assert!(close_with(graph.get_node(b).gradient, -1.25, LOG_TOL));
}
