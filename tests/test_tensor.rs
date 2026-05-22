use std::rc::Rc;

use minitorch_rs::tensor::{Tensor};
use minitorch_rs::tensor_autodiff::{TensorGraph, TensorNodeId, TensorOp};
use minitorch_rs::tensor_data::TensorData;

// ===================================================================
// TensorGraph::new
// ===================================================================

#[test]
fn test_graph_new_is_empty() {
    let g = TensorGraph::new();
    assert!(g.nodes.is_empty());
}

// ===================================================================
// TensorGraph::add_leaf
// ===================================================================

#[test]
fn test_add_leaf_first_id_is_zero() {
    let mut g = TensorGraph::new();
    let data = Rc::new(TensorData::zeros(vec![2, 3]));
    let id = g.add_leaf(data);
    assert_eq!(id.0, 0);
    assert_eq!(g.nodes.len(), 1);
}

#[test]
fn test_add_leaf_ids_are_sequential() {
    let mut g = TensorGraph::new();
    let data = Rc::new(TensorData::zeros(vec![2, 3]));
    let id0 = g.add_leaf(Rc::clone(&data));
    let id1 = g.add_leaf(Rc::clone(&data));
    let id2 = g.add_leaf(data);
    assert_eq!(id0.0, 0);
    assert_eq!(id1.0, 1);
    assert_eq!(id2.0, 2);
}

#[test]
fn test_add_leaf_stores_correct_fields() {
    let mut g = TensorGraph::new();
    let data = Rc::new(TensorData::ones(vec![3]));
    let id = g.add_leaf(Rc::clone(&data));
    let node = &g.nodes[id.0];
    assert!(matches!(node.op, TensorOp::Leaf));
    assert!(node.parents.is_empty());
    assert!(node.gradient.is_none());
    // out shares the storage with the data we passed in
    assert!(Rc::ptr_eq(&node.out, &data));
}

// ===================================================================
// TensorGraph::apply
// ===================================================================

#[test]
fn test_apply_returns_sequential_id_after_leaves() {
    let mut g = TensorGraph::new();
    let a = g.add_leaf(Rc::new(TensorData::zeros(vec![2])));
    let b = g.add_leaf(Rc::new(TensorData::zeros(vec![2])));
    let out_data = Rc::new(TensorData::zeros(vec![2]));
    let c = g.apply(TensorOp::Add, out_data, vec![a, b]);
    assert_eq!(c.0, 2);
    assert_eq!(g.nodes.len(), 3);
}

#[test]
fn test_apply_stores_op_and_parents() {
    let mut g = TensorGraph::new();
    let a = g.add_leaf(Rc::new(TensorData::zeros(vec![2])));
    let b = g.add_leaf(Rc::new(TensorData::zeros(vec![2])));
    let out_data = Rc::new(TensorData::zeros(vec![2]));
    let c = g.apply(TensorOp::Mul, Rc::clone(&out_data), vec![a, b]);

    let node = &g.nodes[c.0];
    assert!(matches!(node.op, TensorOp::Mul));
    assert_eq!(node.parents, vec![a, b]);
    assert!(node.gradient.is_none());
    assert!(Rc::ptr_eq(&node.out, &out_data));
}

#[test]
fn test_apply_preserves_variant_payload() {
    // Variants like Sum carry metadata that backward needs — verify it round-trips.
    let mut g = TensorGraph::new();
    let a = g.add_leaf(Rc::new(TensorData::zeros(vec![2, 3])));
    let out_data = Rc::new(TensorData::zeros(vec![3]));
    let s = g.apply(
        TensorOp::Sum { dim: 0, orig_shape: vec![2, 3] },
        out_data,
        vec![a],
    );
    match &g.nodes[s.0].op {
        TensorOp::Sum { dim, orig_shape } => {
            assert_eq!(*dim, 0);
            assert_eq!(orig_shape, &vec![2, 3]);
        }
        _ => panic!("expected Sum variant"),
    }
}

// ===================================================================
// TensorGraph::topological_sort
// ===================================================================

#[test]
fn test_topo_sort_single_leaf() {
    let mut g = TensorGraph::new();
    let a = g.add_leaf(Rc::new(TensorData::zeros(vec![1])));
    let order = g.topological_sort(a);
    assert_eq!(order, vec![a]);
}

// Linear chain: a → b → c (each parent appears before its child in the order).
#[test]
fn test_topo_sort_linear_chain() {
    let mut g = TensorGraph::new();
    let a = g.add_leaf(Rc::new(TensorData::zeros(vec![1])));
    let b = g.apply(TensorOp::Neg, Rc::new(TensorData::zeros(vec![1])), vec![a]);
    let c = g.apply(TensorOp::Neg, Rc::new(TensorData::zeros(vec![1])), vec![b]);
    let order = g.topological_sort(c);
    // parents appear before children
    assert_eq!(order, vec![a, b, c]);
}

// Diamond: a → b, a → c, (b, c) → d. `a` must be visited only once.
//      a
//     / \
//    b   c
//     \ /
//      d
#[test]
fn test_topo_sort_diamond_visits_shared_parent_once() {
    let mut g = TensorGraph::new();
    let a = g.add_leaf(Rc::new(TensorData::zeros(vec![1])));
    let b = g.apply(TensorOp::Neg, Rc::new(TensorData::zeros(vec![1])), vec![a]);
    let c = g.apply(TensorOp::Neg, Rc::new(TensorData::zeros(vec![1])), vec![a]);
    let d = g.apply(
        TensorOp::Add,
        Rc::new(TensorData::zeros(vec![1])),
        vec![b, c],
    );
    let order = g.topological_sort(d);
    assert_eq!(order.len(), 4);
    // every node appears exactly once
    let mut sorted_ids: Vec<usize> = order.iter().map(|id| id.0).collect();
    sorted_ids.sort();
    assert_eq!(sorted_ids, vec![0, 1, 2, 3]);
    // parents before children: a before b/c; b and c before d
    let pos = |id: TensorNodeId| order.iter().position(|x| *x == id).unwrap();
    assert!(pos(a) < pos(b));
    assert!(pos(a) < pos(c));
    assert!(pos(b) < pos(d));
    assert!(pos(c) < pos(d));
}

// ===================================================================
// Tensor::from_data
// ===================================================================

#[test]
fn test_from_data_has_no_history() {
    let t = Tensor::from_data(TensorData::zeros(vec![2, 3]));
    assert!(t.history.is_none());
}

#[test]
fn test_from_data_preserves_shape() {
    let t = Tensor::from_data(TensorData::zeros(vec![2, 3, 4]));
    assert_eq!(t.shape(), &[2, 3, 4]);
    assert_eq!(t.ndim(), 3);
    assert_eq!(t.size(), 24);
}

// ===================================================================
// Tensor::requires_grad
// ===================================================================

#[test]
fn test_requires_grad_creates_history_with_leaf() {
    let t = Tensor::from_data(TensorData::zeros(vec![2])).requires_grad();
    let hist = t.history.as_ref().expect("history should be Some");
    // graph has exactly one node, the leaf
    let graph = hist.graph.borrow();
    assert_eq!(graph.nodes.len(), 1);
    assert_eq!(hist.node_id.0, 0);
    assert!(matches!(graph.nodes[0].op, TensorOp::Leaf));
}

#[test]
fn test_requires_grad_preserves_data() {
    let t = Tensor::from_data(TensorData::new(vec![1.0, 2.0, 3.0], vec![3]));
    let original_ptr = Rc::as_ptr(&t.data);
    let t = t.requires_grad();
    // data Rc still points at the same storage; no copy happened
    assert_eq!(Rc::as_ptr(&t.data), original_ptr);
    assert_eq!(t.shape(), &[3]);
}

#[test]
#[should_panic]
fn test_requires_grad_panics_on_non_leaf() {
    let t = Tensor::from_data(TensorData::zeros(vec![2])).requires_grad();
    // calling it twice should trip the assert
    let _ = t.requires_grad();
}

// ===================================================================
// Tensor::item
// ===================================================================

#[test]
fn test_item_on_zero_dim_tensor() {
    let t = Tensor::from_data(TensorData::new(vec![42.5], vec![]));
    assert_eq!(t.item(), 42.5);
}

#[test]
fn test_item_on_single_element_multi_dim() {
    let t = Tensor::from_data(TensorData::new(vec![7.0], vec![1, 1, 1]));
    assert_eq!(t.item(), 7.0);
}

// Verify item() reads through storage_offset rather than just storage[0].
// Build a TensorData whose single logical element lives at storage[5].
#[test]
fn test_item_respects_storage_offset() {
    let raw = TensorData {
        storage: Rc::new(vec![0.0, 1.0, 2.0, 3.0, 4.0, 99.0, 6.0]),
        storage_offset: 5,
        shape: vec![],
        strides: vec![],
    };
    let t = Tensor::from_data(raw);
    assert_eq!(t.item(), 99.0);
}

#[test]
#[should_panic]
fn test_item_panics_on_multi_element_tensor() {
    let t = Tensor::from_data(TensorData::zeros(vec![2, 3]));
    let _ = t.item();
}

// ===================================================================
// Tensor: Clone
// ===================================================================

// Cloning a Tensor must share the underlying Rc<TensorData> — no data copy.
#[test]
fn test_clone_shares_data_rc() {
    let t = Tensor::from_data(TensorData::ones(vec![2, 3]));
    let t2 = t.clone();
    assert!(Rc::ptr_eq(&t.data, &t2.data));
}

// Cloning a Tensor with history must share the same graph Rc — both clones
// see future mutations to the graph.
#[test]
fn test_clone_shares_history_graph() {
    let t = Tensor::from_data(TensorData::zeros(vec![2])).requires_grad();
    let t2 = t.clone();
    let h1 = t.history.as_ref().unwrap();
    let h2 = t2.history.as_ref().unwrap();
    assert!(Rc::ptr_eq(&h1.graph, &h2.graph));
    assert_eq!(h1.node_id, h2.node_id);
}
