use std::cell::RefCell;
use std::rc::Rc;

use minitorch_rs::operators;
use minitorch_rs::tensor::{History, Tensor};
use minitorch_rs::tensor_autodiff::{TensorGraph, TensorOp};
use minitorch_rs::tensor_data::TensorData;
use minitorch_rs::tensor_ops::{SimpleOps, TensorOps};

// ===================================================================
// Test infrastructure
// ===================================================================

/// Shorthand for TensorData::new.
fn td(values: Vec<f64>, shape: Vec<usize>) -> TensorData {
    TensorData::new(values, shape)
}

/// Read a TensorData's logical values in row-major order, respecting strides.
fn values(t: &TensorData) -> Vec<f64> {
    t.iter_indices().map(|idx| t.get(&idx)).collect()
}

/// Element-wise comparison with tolerance.
fn assert_data_close(actual: &TensorData, expected: &TensorData, tol: f64) {
    assert_eq!(
        actual.shape, expected.shape,
        "shape mismatch: got {:?}, expected {:?}",
        actual.shape, expected.shape
    );
    let actuals = values(actual);
    let expecteds = values(expected);
    for (i, (a, e)) in actuals.iter().zip(expecteds.iter()).enumerate() {
        let diff = (a - e).abs();
        assert!(
            diff < tol,
            "values differ at flat index {}: got {}, expected {} (|diff| = {} >= tol {})",
            i,
            a,
            e,
            diff,
            tol
        );
    }
}

/// Central-difference numerical gradient. `f` consumes a candidate input and
/// returns the scalar value of the function at that input.
fn central_difference<F>(f: F, x: &TensorData, h: f64) -> TensorData
where
    F: Fn(&TensorData) -> f64,
{
    let mut grad_values = Vec::with_capacity(x.size());
    for idx in x.iter_indices() {
        let orig = x.get(&idx);
        let mut x_plus = x.clone();
        x_plus.set(&idx, orig + h);
        let f_plus = f(&x_plus);
        let mut x_minus = x.clone();
        x_minus.set(&idx, orig - h);
        let f_minus = f(&x_minus);
        grad_values.push((f_plus - f_minus) / (2.0 * h));
    }
    TensorData::new(grad_values, x.shape.clone())
}

/// Reduce a Tensor to a 1-element tensor by summing each dim once. We use
/// keep_dims=true in `sum`, so the final shape is all 1s and `.item()` works.
fn sum_all(mut t: Tensor) -> Tensor {
    for d in 0..t.ndim() {
        t = t.sum(d);
    }
    t
}

/// Grad-check a unary function f: Tensor -> Tensor by comparing backward's
/// analytical gradient against central differences of sum_all(f(x)).
fn grad_check_unary<F>(x_data: TensorData, f: F, h: f64, tol: f64)
where
    F: Fn(&Tensor) -> Tensor,
{
    // analytical
    let x = Tensor::from_data(x_data.clone()).requires_grad();
    let y = sum_all(f(&x));
    y.backward();
    let analytical = x.grad().expect("expected x to have a gradient");

    // numerical
    let numerical = central_difference(
        |x_d| {
            let x_t = Tensor::from_data(x_d.clone());
            sum_all(f(&x_t)).item()
        },
        &x_data,
        h,
    );

    assert_data_close(&analytical, &numerical, tol);
}

/// Construct two leaf Tensors that share a single autograd graph, so binary
/// ops won't trip the cross-graph panic. Test-only helper — production code
/// uses the (None, Some) registration path for the second operand.
fn requires_grad_pair(a_data: TensorData, b_data: TensorData) -> (Tensor, Tensor) {
    let mut graph = TensorGraph::new();
    let a_rc = Rc::new(a_data);
    let b_rc = Rc::new(b_data);
    let a_id = graph.add_leaf(Rc::clone(&a_rc));
    let b_id = graph.add_leaf(Rc::clone(&b_rc));
    let graph_rc = Rc::new(RefCell::new(graph));
    let a_t = Tensor {
        data: a_rc,
        history: Some(History {
            graph: Rc::clone(&graph_rc),
            node_id: a_id,
        }),
    };
    let b_t = Tensor {
        data: b_rc,
        history: Some(History {
            graph: graph_rc,
            node_id: b_id,
        }),
    };
    (a_t, b_t)
}

/// Grad-check a binary function w.r.t. BOTH inputs simultaneously.
fn grad_check_binary<F>(
    a_data: TensorData,
    b_data: TensorData,
    f: F,
    h: f64,
    tol: f64,
) where
    F: Fn(&Tensor, &Tensor) -> Tensor,
{
    let (a, b) = requires_grad_pair(a_data.clone(), b_data.clone());
    let y = sum_all(f(&a, &b));
    y.backward();
    let a_analytical = a.grad().expect("expected a to have a gradient");
    let b_analytical = b.grad().expect("expected b to have a gradient");

    let a_numerical = central_difference(
        |a_d| {
            let a_t = Tensor::from_data(a_d.clone());
            let b_t = Tensor::from_data(b_data.clone());
            sum_all(f(&a_t, &b_t)).item()
        },
        &a_data,
        h,
    );
    let b_numerical = central_difference(
        |b_d| {
            let a_t = Tensor::from_data(a_data.clone());
            let b_t = Tensor::from_data(b_d.clone());
            sum_all(f(&a_t, &b_t)).item()
        },
        &b_data,
        h,
    );

    assert_data_close(&a_analytical, &a_numerical, tol);
    assert_data_close(&b_analytical, &b_numerical, tol);
}

// ===================================================================
// Per-op grad checks: unary ops with Tensor methods
// ===================================================================

#[test]
fn test_grad_neg() {
    grad_check_unary(
        td(vec![1.0, -2.0, 3.0, -4.0], vec![2, 2]),
        |x| -x,
        1e-4,
        1e-5,
    );
}

#[test]
fn test_grad_sigmoid() {
    // moderate magnitudes — far from saturation
    grad_check_unary(
        td(vec![0.5, -0.3, 1.2, -1.0, 0.0, 0.7], vec![2, 3]),
        |x| x.sigmoid(),
        1e-4,
        1e-5,
    );
}

#[test]
fn test_grad_relu_all_positive() {
    grad_check_unary(
        td(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
        |x| x.relu(),
        1e-4,
        1e-5,
    );
}

#[test]
fn test_grad_relu_mixed() {
    // avoid x=0 (ReLU non-differentiable at the kink)
    grad_check_unary(
        td(vec![1.5, -0.5, 2.0, -1.5, 0.8, -2.0], vec![2, 3]),
        |x| x.relu(),
        1e-4,
        1e-5,
    );
}

#[test]
fn test_grad_log() {
    // strictly positive inputs
    grad_check_unary(
        td(vec![1.0, 2.0, 0.5, 1.5, 3.0, 0.7], vec![2, 3]),
        |x| x.log(),
        1e-4,
        1e-4,
    );
}

#[test]
fn test_grad_chained_unaries() {
    // log(sigmoid(x)) — covers chain rule across two non-trivial ops
    grad_check_unary(
        td(vec![0.5, -0.3, 1.0, -1.2], vec![2, 2]),
        |x| x.sigmoid().log(),
        1e-4,
        1e-4,
    );
}

// ===================================================================
// Per-op grad checks: binary ops + broadcasting
// ===================================================================

#[test]
fn test_grad_add_same_shape() {
    grad_check_binary(
        td(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
        td(vec![0.5, -0.5, 1.5, -1.5], vec![2, 2]),
        |a, b| a + b,
        1e-4,
        1e-5,
    );
}

#[test]
fn test_grad_add_broadcast_row() {
    // [2,3] + [3]: row broadcast — b's gradient is the column-sum of upstream
    grad_check_binary(
        td(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]),
        td(vec![0.5, -0.5, 1.5], vec![3]),
        |a, b| a + b,
        1e-4,
        1e-5,
    );
}

#[test]
fn test_grad_add_broadcast_column() {
    // [2,3] + [2,1]: column broadcast — b's gradient is row-sums kept as [2,1]
    grad_check_binary(
        td(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]),
        td(vec![0.5, -0.5], vec![2, 1]),
        |a, b| a + b,
        1e-4,
        1e-5,
    );
}

#[test]
fn test_grad_mul_same_shape() {
    grad_check_binary(
        td(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
        td(vec![0.5, -0.5, 1.5, -1.5], vec![2, 2]),
        |a, b| a * b,
        1e-4,
        1e-5,
    );
}

#[test]
fn test_grad_mul_broadcast() {
    // forces zip(d, b_broadcast) then reduce-back-to-b for b's grad
    grad_check_binary(
        td(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]),
        td(vec![0.5, -0.5, 1.5], vec![3]),
        |a, b| a * b,
        1e-4,
        1e-5,
    );
}

#[test]
fn test_grad_mul_then_add() {
    // (a * b) + a — a is used twice, so its gradient must accumulate
    grad_check_binary(
        td(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
        td(vec![0.5, -0.5, 1.5, -1.5], vec![2, 2]),
        |a, b| &(a * b) + a,
        1e-4,
        1e-5,
    );
}

// ===================================================================
// Per-op grad check: Sum
// ===================================================================

#[test]
fn test_grad_sum_dim_0() {
    grad_check_unary(
        td(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]),
        |x| x.sum(0),
        1e-4,
        1e-5,
    );
}

#[test]
fn test_grad_sum_dim_1() {
    grad_check_unary(
        td(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]),
        |x| x.sum(1),
        1e-4,
        1e-5,
    );
}

#[test]
fn test_grad_sum_3d() {
    grad_check_unary(
        td(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            vec![2, 3, 2],
        ),
        |x| x.sum(1),
        1e-4,
        1e-5,
    );
}

#[test]
fn test_grad_repeated_sum() {
    // sum chain — exercises backward through two Sum nodes
    grad_check_unary(
        td(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]),
        |x| x.sum(0).sum(1),
        1e-4,
        1e-5,
    );
}

// ===================================================================
// Per-op grad checks via Tensor wrappers: View, Permute, MatMul,
// Broadcast, Exp. Catches wrapper-to-op wiring regressions.
// ===================================================================

#[test]
fn test_grad_view_through_tensor() {
    grad_check_unary(
        td(vec![0.5, -0.3, 1.0, -1.2, 0.7, 0.2], vec![2, 3]),
        |x| x.sigmoid().view(&[6]),
        1e-4,
        1e-4,
    );
}

#[test]
fn test_grad_permute_through_tensor() {
    grad_check_unary(
        td(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]),
        |x| x.permute(&[1, 0]),
        1e-4,
        1e-5,
    );
}

#[test]
fn test_grad_permute_3d_through_tensor() {
    let storage: Vec<f64> = (1..=24).map(|i| i as f64 * 0.1).collect();
    grad_check_unary(
        td(storage, vec![2, 3, 4]),
        |x| x.permute(&[2, 0, 1]),
        1e-4,
        1e-5,
    );
}

#[test]
fn test_grad_matmul_through_tensor() {
    grad_check_binary(
        td(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]),
        td(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            vec![3, 4],
        ),
        |a, b| a.matmul(b),
        1e-4,
        1e-4,
    );
}

#[test]
fn test_grad_matmul_batched_through_tensor() {
    let a_storage: Vec<f64> = (1..=12).map(|i| i as f64 * 0.1).collect();
    let b_storage: Vec<f64> = (1..=18).map(|i| i as f64 * 0.1).collect();
    grad_check_binary(
        td(a_storage, vec![2, 2, 3]),
        td(b_storage, vec![2, 3, 3]),
        |a, b| a.matmul(b),
        1e-4,
        1e-4,
    );
}

#[test]
fn test_grad_broadcast_through_tensor() {
    grad_check_unary(
        td(vec![1.0, 2.0, 3.0], vec![3]),
        |x| x.broadcast_to(&[2, 3]),
        1e-4,
        1e-5,
    );
}

#[test]
fn test_grad_broadcast_then_mul() {
    // broadcast + a downstream op — covers backward through the wrapper
    // and into another op
    grad_check_unary(
        td(vec![1.0, 2.0, 3.0], vec![3]),
        |x| x.broadcast_to(&[2, 3]).sigmoid(),
        1e-4,
        1e-4,
    );
}

#[test]
fn test_grad_exp_through_tensor() {
    grad_check_unary(
        td(vec![0.0, 0.5, -0.5, 1.0], vec![2, 2]),
        |x| x.exp(),
        1e-4,
        1e-4,
    );
}

// ===================================================================
// Direct-graph tests: backward arms exercised without going through
// the Tensor wrapper. Retained for any future op whose wrapper lags
// behind compute_backward, and for tighter assertions on backward
// outputs than grad-check tolerances allow.
// ===================================================================

/// Build a graph with a single leaf, apply `op` producing `out` with one
/// parent, seed gradient on the resulting node, run backward, and return
/// the leaf's gradient.
fn run_unary_backward(
    leaf_data: TensorData,
    op: TensorOp,
    out_data: TensorData,
    upstream: TensorData,
) -> TensorData {
    let mut graph = TensorGraph::new();
    let leaf_id = graph.add_leaf(Rc::new(leaf_data));
    let out_id = graph.apply(op, Rc::new(out_data), vec![leaf_id]);
    graph.nodes[out_id.0].gradient = Some(upstream);
    graph.backpropagate(out_id);
    graph.nodes[leaf_id.0]
        .gradient
        .clone()
        .expect("leaf should have gradient")
}

/// Same but for a binary op with two parents.
fn run_binary_backward(
    a_data: TensorData,
    b_data: TensorData,
    op: TensorOp,
    out_data: TensorData,
    upstream: TensorData,
) -> (TensorData, TensorData) {
    let mut graph = TensorGraph::new();
    let a_id = graph.add_leaf(Rc::new(a_data));
    let b_id = graph.add_leaf(Rc::new(b_data));
    let out_id = graph.apply(op, Rc::new(out_data), vec![a_id, b_id]);
    graph.nodes[out_id.0].gradient = Some(upstream);
    graph.backpropagate(out_id);
    let a_grad = graph.nodes[a_id.0]
        .gradient
        .clone()
        .expect("a should have gradient");
    let b_grad = graph.nodes[b_id.0]
        .gradient
        .clone()
        .expect("b should have gradient");
    (a_grad, b_grad)
}

#[test]
fn test_backward_exp() {
    let x = td(vec![0.0, 1.0, -0.5, 2.0], vec![2, 2]);
    let y = SimpleOps::map(&x, operators::exp);
    let upstream = td(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
    let grad = run_unary_backward(x.clone(), TensorOp::Exp, y.clone(), upstream);
    // d/dx exp(x) = exp(x), with upstream=1 so grad == y
    assert_data_close(&grad, &y, 1e-10);
}

#[test]
fn test_backward_view_simple() {
    // forward View: shape [2,3] -> [6]; backward sends the gradient back
    // to [2,3] in row-major order.
    let x = td(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let y = td(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![6]);
    let upstream = td(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], vec![6]);
    let grad = run_unary_backward(
        x,
        TensorOp::View {
            orig_shape: vec![2, 3],
        },
        y,
        upstream,
    );
    assert_data_close(
        &grad,
        &td(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], vec![2, 3]),
        1e-10,
    );
}

#[test]
fn test_backward_view_from_strided_upstream() {
    // Upstream gradient is a strided view (not row-major contiguous). The
    // contiguous() inside the View arm must materialize values in logical
    // order before reshaping.
    let x = td(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0], vec![2, 3]);
    let y = td(vec![0.0; 6], vec![6]);
    // Build a strided upstream by permuting a [3,2] tensor — same 6 values,
    // different layout. After permute, logical order is column-major of the
    // original.
    let upstream_pre = td(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
    let upstream = upstream_pre.permute(&[1, 0]); // shape [2, 3] strided
    // flatten to [6] by hand to match the View output shape
    let upstream_flat = TensorData::new(values(&upstream), vec![6]);
    let grad = run_unary_backward(
        x,
        TensorOp::View {
            orig_shape: vec![2, 3],
        },
        y,
        upstream_flat.clone(),
    );
    // backward should produce shape [2,3] with the same row-major values
    assert_eq!(grad.shape, vec![2, 3]);
    assert_eq!(values(&grad), values(&upstream_flat));
}

#[test]
fn test_backward_permute() {
    // forward Permute: shape [2,3] with axes [1,0] -> [3,2]
    let x = td(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let y = x.permute(&[1, 0]);
    // upstream has y's shape [3,2]; backward should permute it back to [2,3]
    let upstream = td(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], vec![3, 2]);
    let grad = run_unary_backward(
        x,
        TensorOp::Permute { axes: vec![1, 0] },
        y,
        upstream.clone(),
    );
    // Inverse of [1,0] is [1,0]. So grad = upstream permuted by [1,0].
    let expected = upstream.permute(&[1, 0]);
    assert_eq!(grad.shape, expected.shape);
    assert_eq!(values(&grad), values(&expected));
}

#[test]
fn test_backward_permute_3d() {
    // axes [2,0,1] on shape [2,3,4] -> output shape [4,2,3]
    // inverse should be [1,2,0]
    let shape = vec![2, 3, 4];
    let n = 2 * 3 * 4;
    let storage: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let x = td(storage.clone(), shape.clone());
    let y = x.permute(&[2, 0, 1]);
    let upstream_storage: Vec<f64> = (0..n).map(|i| (i as f64) * 0.5).collect();
    let upstream = td(upstream_storage, vec![4, 2, 3]);
    let grad = run_unary_backward(
        x,
        TensorOp::Permute {
            axes: vec![2, 0, 1],
        },
        y,
        upstream.clone(),
    );
    assert_eq!(grad.shape, shape);
    // Sanity: permuting grad with axes [2,0,1] should recover upstream's values.
    let round_trip = grad.permute(&[2, 0, 1]);
    assert_eq!(values(&round_trip), values(&upstream));
}

#[test]
fn test_backward_matmul_2d() {
    // C = A @ B; verify dC/dA = upstream @ B^T and dC/dB = A^T @ upstream
    let a = td(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = td(
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        vec![3, 4],
    );
    let c = SimpleOps::matmul(&a, &b);
    assert_eq!(c.shape, vec![2, 4]);
    let upstream = td(vec![1.0, 0.5, -1.0, 2.0, 0.3, 1.5, -0.5, 1.0], vec![2, 4]);
    let (a_grad, b_grad) = run_binary_backward(
        a.clone(),
        b.clone(),
        TensorOp::MatMul,
        c,
        upstream.clone(),
    );
    // Expected: a_grad = upstream @ B^T, b_grad = A^T @ upstream
    let b_t = b.permute(&[1, 0]);
    let a_t = a.permute(&[1, 0]);
    let expected_a_grad = SimpleOps::matmul(&upstream, &b_t);
    let expected_b_grad = SimpleOps::matmul(&a_t, &upstream);
    assert_data_close(&a_grad, &expected_a_grad, 1e-10);
    assert_data_close(&b_grad, &expected_b_grad, 1e-10);
}

#[test]
fn test_backward_matmul_batched() {
    // C[2,2,3] = A[2,2,4] @ B[2,4,3]; both batched on dim 0
    let a_storage: Vec<f64> = (1..=16).map(|i| i as f64).collect();
    let b_storage: Vec<f64> = (1..=24).map(|i| (i as f64) * 0.5).collect();
    let a = td(a_storage, vec![2, 2, 4]);
    let b = td(b_storage, vec![2, 4, 3]);
    let c = SimpleOps::matmul(&a, &b);
    assert_eq!(c.shape, vec![2, 2, 3]);
    let upstream_storage: Vec<f64> = (1..=12).map(|i| (i as f64) * 0.1).collect();
    let upstream = td(upstream_storage, vec![2, 2, 3]);
    let (a_grad, b_grad) = run_binary_backward(
        a.clone(),
        b.clone(),
        TensorOp::MatMul,
        c,
        upstream.clone(),
    );
    // Expected: a_grad has a's shape, b_grad has b's shape.
    // Compute reference via the same formula.
    let b_t = b.permute(&[0, 2, 1]); // transpose last two of [2,4,3] -> [2,3,4]
    let a_t = a.permute(&[0, 2, 1]); // [2,2,4] -> [2,4,2]
    let expected_a_grad = SimpleOps::matmul(&upstream, &b_t);
    let expected_b_grad = SimpleOps::matmul(&a_t, &upstream);
    assert_data_close(&a_grad, &expected_a_grad, 1e-10);
    assert_data_close(&b_grad, &expected_b_grad, 1e-10);
}

#[test]
fn test_backward_broadcast() {
    // forward Broadcast: shape [3] -> [2,3]; backward reduces back to [3]
    // by summing the broadcast dim (axis 0).
    let x = td(vec![1.0, 2.0, 3.0], vec![3]);
    let y = x.broadcast_to(&[2, 3]);
    let upstream = td(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0], vec![2, 3]);
    let grad = run_unary_backward(
        x,
        TensorOp::Broadcast {
            orig_shape: vec![3],
        },
        y,
        upstream,
    );
    // Each element of [3] is broadcast across 2 rows, so each grad = 2.
    assert_data_close(&grad, &td(vec![2.0, 2.0, 2.0], vec![3]), 1e-10);
}

// ===================================================================
// Plumbing tests
// ===================================================================

// backward() is a no-op when the Tensor has no history — should not panic.
#[test]
fn test_backward_no_history_is_noop() {
    let t = Tensor::from_data(td(vec![1.0, 2.0], vec![2]));
    t.backward(); // must not panic
    assert!(t.grad().is_none());
}

// grad() returns None on a requires_grad tensor before backward is called.
#[test]
fn test_grad_none_before_backward() {
    let t = Tensor::from_data(td(vec![1.0, 2.0], vec![2])).requires_grad();
    assert!(t.grad().is_none());
}

// After backward on a single leaf chain (summed to scalar), the leaf's
// gradient should be ones — seeded gradient passes through identity ops.
#[test]
fn test_backward_seeds_with_ones() {
    let x = Tensor::from_data(td(vec![1.0, 2.0, 3.0], vec![3])).requires_grad();
    // identity-like graph: x → x + 0 (so there's at least one op), summed
    // to a scalar so backward's scalar-output contract holds.
    let zeros = Tensor::from_data(td(vec![0.0, 0.0, 0.0], vec![3]));
    let y = sum_all(&x + &zeros);
    y.backward();
    let g = x.grad().expect("grad should exist");
    assert_data_close(&g, &td(vec![1.0, 1.0, 1.0], vec![3]), 1e-10);
}

// Calling backward twice on the same tensor must produce the same gradient
// (clear_gradients prevents accumulation across calls).
#[test]
fn test_backward_idempotent_after_clear() {
    let x =
        Tensor::from_data(td(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])).requires_grad();
    let y = sum_all(x.sigmoid());
    y.backward();
    let g1 = x.grad().expect("grad");
    y.backward();
    let g2 = x.grad().expect("grad");
    assert_data_close(&g1, &g2, 1e-10);
}

// Diamond reuse: x is used twice in y = x * x. dy/dx = 2x. Tests gradient
// accumulation INSIDE a single backward pass (the legitimate accumulation).
#[test]
fn test_backward_diamond_accumulates() {
    let x_data = td(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let x = Tensor::from_data(x_data.clone()).requires_grad();
    let y = sum_all(&x * &x);
    y.backward();
    let g = x.grad().expect("grad");
    // d/dx (x*x) = 2x
    let expected = td(vec![2.0, 4.0, 6.0, 8.0], vec![2, 2]);
    assert_data_close(&g, &expected, 1e-10);
}

// Triangle reuse: x → 2x → x*(2x). dy/dx = 4x. Cross-checks the same
// accumulation pattern through a different topology.
#[test]
fn test_backward_triangle_via_grad_check() {
    grad_check_unary(
        td(vec![0.5, 1.5, -0.5, 2.0], vec![2, 2]),
        |x| &(x + x) * x,
        1e-4,
        1e-5,
    );
}

// clear_gradients on an intermediate node: build a graph, run backward,
// confirm intermediates have grads, call clear, confirm they're None.
#[test]
fn test_clear_gradients_wipes_intermediates() {
    let x = Tensor::from_data(td(vec![1.0, 2.0], vec![2])).requires_grad();
    let y = x.sigmoid(); // intermediate node
    let z = sum_all(y.clone());
    z.backward();
    {
        let graph = x.history.as_ref().unwrap().graph.borrow();
        // every node should have Some(gradient) right now
        for node in &graph.nodes {
            assert!(
                node.gradient.is_some(),
                "expected all nodes to have grads after backward"
            );
        }
    }
    x.history
        .as_ref()
        .unwrap()
        .graph
        .borrow_mut()
        .clear_gradients();
    {
        let graph = x.history.as_ref().unwrap().graph.borrow();
        for node in &graph.nodes {
            assert!(node.gradient.is_none(), "expected all nodes to be cleared");
        }
    }
}

// Leaf compute_backward should return an empty vec — terminates the chain.
#[test]
fn test_compute_backward_leaf_returns_empty() {
    let mut g = TensorGraph::new();
    let leaf = g.add_leaf(Rc::new(td(vec![1.0, 2.0], vec![2])));
    let result = g.compute_backward(leaf);
    assert!(result.is_empty());
}

// ===================================================================
// Adversarial tests — catch contracts the happy-path grad-check suite
// doesn't exercise.
// ===================================================================

// Backward must reject non-scalar outputs. Without the scalar assert,
// calling .backward() on a non-scalar tensor silently computes
// d(sum(output))/dx — a quiet, plausible-looking-but-wrong answer.
// PyTorch raises here for the same reason.
#[test]
#[should_panic(expected = "scalar")]
fn test_backward_panics_on_non_scalar_output() {
    let x = Tensor::from_data(td(vec![1.0, 2.0, 3.0], vec![3])).requires_grad();
    let y = &x + &x; // shape [3], not scalar
    y.backward();
}

// 1-element-multi-dim tensors (e.g. [1,1,1]) ARE valid scalars by size().
// This must not trip the scalar assert — sum_all produces exactly this shape.
#[test]
fn test_backward_accepts_1element_multidim_output() {
    let x =
        Tensor::from_data(td(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])).requires_grad();
    let y = sum_all(x.sigmoid()); // shape ends up [1, 1], size 1
    assert_eq!(y.size(), 1);
    y.backward(); // must not panic
    assert!(x.grad().is_some());
}

// Backward through View whose forward came from a sigmoid (which produces
// contiguous storage) and whose backward used to bypass invariants — now
// must produce a parent_gradient whose storage is tight. This catches the
// combined map+View+contiguous regression as an end-to-end Tensor-API test.
#[test]
fn test_backward_view_through_sigmoid_produces_tight_grad() {
    let x = Tensor::from_data(td(vec![0.5, -0.3, 1.0, -1.2, 0.7, 0.2], vec![2, 3]))
        .requires_grad();
    let y = sum_all(x.sigmoid().view(&[6]));
    y.backward();
    let g = x.grad().expect("grad should exist");
    // Critical: shape matches x, storage tight (no leftover from the
    // intermediate sigmoid/view tensors).
    assert_eq!(g.shape, vec![2, 3]);
    assert_eq!(g.storage.len(), 6, "grad storage must be tight to size");
}
