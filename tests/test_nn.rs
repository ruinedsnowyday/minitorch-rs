use minitorch_rs::autodiff::{ScalarGraph, ScalarOp, central_difference};
use minitorch_rs::module::Module;
use minitorch_rs::nn::Linear;

const EPS: f64 = 1e-9;

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < EPS
}

// Round-trip test for Linear: forward → backward → step.
//
// Handcrafted Linear(in=2, out=2) so the whole pipeline is checkable
// against pen-and-paper arithmetic.
//
//   W = [[0.1, 0.2],
//        [0.3, 0.4]]
//   b = [0.5, 0.6]
//   x = [3.0, 5.0]
//
// Forward:
//   y0 = 0.1*3 + 0.2*5 + 0.5 = 1.8
//   y1 = 0.3*3 + 0.4*5 + 0.6 = 3.5
//
// Backward from y0 with d=1.0 (NOT y1):
//   dy0/dW[0][0] = x0 = 3.0
//   dy0/dW[0][1] = x1 = 5.0
//   dy0/db[0]    = 1.0
//   row 1 is unreachable from y0 → grads stay 0
//
// Step with lr=1.0:
//   W[0][0]: 0.1 - 3.0 = -2.9
//   W[0][1]: 0.2 - 5.0 = -4.8
//   b[0]:    0.5 - 1.0 = -0.5
//   W[1] and b[1]: unchanged
#[test]
fn test_linear_forward_backward_step_roundtrip() {
    let mut model = Linear::new(2, 2);
    // overwrite random Xavier init with the fixed values from the comment above
    model.weights = vec![vec![0.1, 0.2], vec![0.3, 0.4]];
    model.biases = vec![0.5, 0.6];

    let mut graph = ScalarGraph::new();
    let x_ids: Vec<_> = [3.0_f64, 5.0].iter().map(|v| graph.add_leaf(*v)).collect();

    // --- forward ---
    let y_ids = model.forward(&mut graph, x_ids);
    assert_eq!(y_ids.len(), 2);
    assert!(close(graph.get_node(y_ids[0]).out, 1.8));
    assert!(close(graph.get_node(y_ids[1]).out, 3.5));

    // --- backward from y0 only ---
    graph.backpropagate(y_ids[0], 1.0);

    let weight_ids = model.weight_ids.as_ref().unwrap();
    let bias_ids = model.bias_ids.as_ref().unwrap();

    // row 0: gradients == inputs
    assert!(close(graph.get_node(weight_ids[0][0]).gradient, 3.0));
    assert!(close(graph.get_node(weight_ids[0][1]).gradient, 5.0));
    assert!(close(graph.get_node(bias_ids[0]).gradient, 1.0));

    // row 1: untouched, since y1 was not part of the backward
    assert_eq!(graph.get_node(weight_ids[1][0]).gradient, 0.0);
    assert_eq!(graph.get_node(weight_ids[1][1]).gradient, 0.0);
    assert_eq!(graph.get_node(bias_ids[1]).gradient, 0.0);

    // --- step ---
    model.step(&graph, 1.0);

    assert!(close(model.weights[0][0], -2.9));
    assert!(close(model.weights[0][1], -4.8));
    assert!(close(model.biases[0], -0.5));
    assert_eq!(model.weights[1][0], 0.3);
    assert_eq!(model.weights[1][1], 0.4);
    assert_eq!(model.biases[1], 0.6);
}

// Backward from a sum of outputs: gradients reach BOTH rows, and the input
// gradient accumulates contributions from every output it feeds.
//
//   loss = y0 + y1
//   d(loss)/dW[i][j] = x[j]               (same for every row i)
//   d(loss)/db[i]    = 1
//   d(loss)/dx[j]    = W[0][j] + W[1][j]  (accumulation across rows)
#[test]
fn test_linear_backward_through_sum() {
    let mut model = Linear::new(2, 2);
    model.weights = vec![vec![0.1, 0.2], vec![0.3, 0.4]];
    model.biases = vec![0.5, 0.6];

    let mut graph = ScalarGraph::new();
    let x_ids: Vec<_> = [3.0_f64, 5.0].iter().map(|v| graph.add_leaf(*v)).collect();
    let x0_id = x_ids[0];
    let x1_id = x_ids[1];

    let y_ids = model.forward(&mut graph, x_ids);

    let loss_id = graph.apply(
        ScalarOp::Add(graph.get_node(y_ids[0]).out, graph.get_node(y_ids[1]).out),
        vec![y_ids[0], y_ids[1]],
    );

    graph.backpropagate(loss_id, 1.0);

    let weight_ids = model.weight_ids.as_ref().unwrap();
    let bias_ids = model.bias_ids.as_ref().unwrap();

    // every weight grad equals the corresponding input value
    assert!(close(graph.get_node(weight_ids[0][0]).gradient, 3.0));
    assert!(close(graph.get_node(weight_ids[0][1]).gradient, 5.0));
    assert!(close(graph.get_node(weight_ids[1][0]).gradient, 3.0));
    assert!(close(graph.get_node(weight_ids[1][1]).gradient, 5.0));

    // bias grads are 1.0 because d(y_i)/d(b_i) = 1
    assert!(close(graph.get_node(bias_ids[0]).gradient, 1.0));
    assert!(close(graph.get_node(bias_ids[1]).gradient, 1.0));

    // input x_j picks up W[0][j] + W[1][j] via gradient accumulation
    //   x0: 0.1 + 0.3 = 0.4
    //   x1: 0.2 + 0.4 = 0.6
    assert!(close(graph.get_node(x0_id).gradient, 0.4));
    assert!(close(graph.get_node(x1_id).gradient, 0.6));
}

// Numerical check on the autodiff gradient: perturb W[0][0] symmetrically
// and confirm that (forward(w+h) - forward(w-h)) / 2h matches the gradient
// produced by backprop. If the chain rule in autodiff is wrong, this fails.
#[test]
fn test_linear_gradient_matches_central_difference() {
    // autodiff path
    let mut model = Linear::new(2, 1);
    model.weights = vec![vec![0.1, 0.2]];
    model.biases = vec![0.5];

    let mut graph = ScalarGraph::new();
    let x_ids: Vec<_> = [3.0_f64, 5.0].iter().map(|v| graph.add_leaf(*v)).collect();
    let y_ids = model.forward(&mut graph, x_ids);
    graph.backpropagate(y_ids[0], 1.0);
    let autodiff_grad = graph
        .get_node(model.weight_ids.as_ref().unwrap()[0][0])
        .gradient;

    // numerical path: treat the whole forward pass as a function of W[0][0]
    let f = |params: &[f64]| -> f64 {
        let mut m = Linear::new(2, 1);
        m.weights = vec![vec![params[0], 0.2]];
        m.biases = vec![0.5];
        let mut g = ScalarGraph::new();
        let xs: Vec<_> = [3.0_f64, 5.0].iter().map(|v| g.add_leaf(*v)).collect();
        let ys = m.forward(&mut g, xs);
        g.get_node(ys[0]).out
    };
    let numerical_grad = central_difference(f, &[0.1], 0, 1e-6);

    // central difference is O(h^2) accurate; 1e-5 is comfortable
    assert!((autodiff_grad - numerical_grad).abs() < 1e-5);
}

// Calling step before any forward should leave the parameters untouched
// rather than panic — the Option<Vec<NodeId>>s start as None.
#[test]
fn test_linear_step_without_forward_is_noop() {
    let mut model = Linear::new(2, 3);
    let original_w = model.weights.clone();
    let original_b = model.biases.clone();

    let graph = ScalarGraph::new();
    model.step(&graph, 1.0);

    assert_eq!(model.weights, original_w);
    assert_eq!(model.biases, original_b);
}

// Passing the wrong number of input ids should trigger the assert_eq! in forward.
#[test]
#[should_panic]
fn test_linear_forward_panics_on_wrong_input_dim() {
    let mut model = Linear::new(2, 1);
    let mut graph = ScalarGraph::new();
    // Linear expects 2 inputs; we pass 1
    let x_ids: Vec<_> = [3.0_f64].iter().map(|v| graph.add_leaf(*v)).collect();
    model.forward(&mut graph, x_ids);
}

// named_parameters should produce one entry per scalar weight and bias,
// with names following the "w_{row},{col}" and "b_{row}" convention.
#[test]
fn test_linear_named_parameters() {
    let model = Linear::new(2, 3);
    let named = model.named_parameters();

    // 3 outputs × 2 inputs = 6 weights, plus 3 biases
    assert_eq!(named.len(), 9);

    let names: Vec<&str> = named.iter().map(|(n, _)| n.as_str()).collect();
    for row in 0..3 {
        assert!(names.contains(&format!("w_{row},0").as_str()));
        assert!(names.contains(&format!("w_{row},1").as_str()));
        assert!(names.contains(&format!("b_{row}").as_str()));
    }
}
