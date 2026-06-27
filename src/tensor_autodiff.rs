use std::{collections::HashSet, rc::Rc};

use crate::{
    Backend, operators,
    simd_ops::{
        LANES, SimdAdd, SimdLogBack, SimdMul, SimdNeg, SimdReLUBack, SimdSigmoidBack,
    },
    tensor_data::{TensorData, contiguous_strides},
    tensor_ops::TensorOps,
};

#[derive(Clone, Copy, Eq, PartialEq, Debug, Hash)]
pub struct TensorNodeId(pub usize);

/// Struct for storing all nodes of a one computational graph with tensor operations
pub struct TensorGraph {
    pub nodes: Vec<TensorNode>,
}

/// Node in computational graph representing a tensor operation
pub struct TensorNode {
    pub op: TensorOp,
    pub out: Rc<TensorData>,
    pub gradient: Option<TensorData>,
    pub parents: Vec<TensorNodeId>,
}

/// Enumeration of existing operations on tensors
pub enum TensorOp {
    Leaf,
    Add,
    Mul,
    Neg,
    Sigmoid,
    ReLU,
    Log,
    Exp,
    Sum { orig_shape: Vec<usize> },
    View { orig_shape: Vec<usize> },
    Permute { axes: Vec<usize> },
    MatMul,
    Broadcast { orig_shape: Vec<usize> },
}

impl TensorGraph {
    /// Creates an empty tensor computational graph
    pub fn new() -> Self {
        TensorGraph { nodes: vec![] }
    }

    /// Adds a leaf to the tensor graph, i.e. a node in computational graph that just carries data
    /// without depending on any other node in the graph
    pub fn add_leaf(&mut self, data: Rc<TensorData>) -> TensorNodeId {
        let out = TensorNodeId(self.nodes.len());
        let leaf = TensorNode {
            op: TensorOp::Leaf,
            out: data,
            gradient: None,
            parents: vec![],
        };
        self.nodes.push(leaf);
        out
    }

    /// Adds a node to the computational that depends on existing nodes in the graph,
    /// operations performed on them, and the result of this operation
    pub fn apply(
        &mut self,
        op: TensorOp,
        out: Rc<TensorData>,
        parents: Vec<TensorNodeId>,
    ) -> TensorNodeId {
        let id = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            op,
            out,
            gradient: None,
            parents,
        });
        id
    }

    /// Returns a topological sort on the computational graph by doing DFS
    pub fn topological_sort(&self, root: TensorNodeId) -> Vec<TensorNodeId> {
        let mut order: Vec<TensorNodeId> = Vec::with_capacity(self.nodes.len());
        let mut visited: HashSet<TensorNodeId> =
            HashSet::with_capacity(self.nodes.len());
        fn visit(
            idx: TensorNodeId,
            visited: &mut HashSet<TensorNodeId>,
            order: &mut Vec<TensorNodeId>,
            nodes: &[TensorNode],
        ) {
            if visited.contains(&idx) {
                return;
            };
            visited.insert(idx);
            for parent in &nodes[idx.0].parents {
                visit(*parent, visited, order, nodes);
            }
            order.push(idx);
        }
        visit(root, &mut visited, &mut order, &self.nodes);
        order
    }

    /// for given id of the node, computes parent id-parent gradient pairs
    pub fn compute_backward(
        &self,
        id: TensorNodeId,
    ) -> Vec<(TensorNodeId, TensorData)> {
        let node = &self.nodes[id.0];
        if matches!(&node.op, TensorOp::Leaf) {
            return vec![];
        }
        let gradient = node
            .gradient
            .as_ref()
            .expect("gradient must be set before calling compute_backward");
        match &node.op {
            TensorOp::Leaf => unreachable!(),
            TensorOp::Add => {
                let left_parent_id = node.parents[0];
                let left_parent_shape = &self.nodes[left_parent_id.0].out.shape;
                let right_parent_id = node.parents[1];
                let right_parent_shape = &self.nodes[right_parent_id.0].out.shape;
                vec![
                    (
                        left_parent_id,
                        maybe_reduce_broadcast(gradient, left_parent_shape),
                    ),
                    (
                        right_parent_id,
                        maybe_reduce_broadcast(gradient, right_parent_shape),
                    ),
                ]
            }
            TensorOp::Mul => {
                let left_parent_id = node.parents[0];
                let left_parent = &self.nodes[left_parent_id.0];
                let right_parent_id = node.parents[1];
                let right_parent = &self.nodes[right_parent_id.0];
                let left_gradient =
                    Backend::zip_op::<SimdMul, LANES>(gradient, &right_parent.out);
                let left_gradient_reduced =
                    maybe_reduce_broadcast(&left_gradient, &left_parent.out.shape);
                let right_gradient =
                    Backend::zip_op::<SimdMul, LANES>(gradient, &left_parent.out);
                let right_gradient_reduced =
                    maybe_reduce_broadcast(&right_gradient, &right_parent.out.shape);
                vec![
                    (left_parent_id, left_gradient_reduced),
                    (right_parent_id, right_gradient_reduced),
                ]
            }
            TensorOp::Neg => {
                let parent_id = node.parents[0];
                let parent_gradient = Backend::map_op::<SimdNeg, LANES>(gradient);
                vec![(parent_id, parent_gradient)]
            }
            TensorOp::Sigmoid => {
                let parent_id = node.parents[0];
                let parent_gradient =
                    Backend::zip_op::<SimdSigmoidBack, LANES>(&node.out, gradient);
                vec![(parent_id, parent_gradient)]
            }
            TensorOp::ReLU => {
                let parent_id = node.parents[0];
                let parent = &self.nodes[parent_id.0];
                let parent_gradient =
                    Backend::zip_op::<SimdReLUBack, LANES>(&parent.out, gradient);
                vec![(parent_id, parent_gradient)]
            }
            TensorOp::Log => {
                let parent_id = node.parents[0];
                let parent = &self.nodes[parent_id.0];
                let parent_gradient =
                    Backend::zip_op::<SimdLogBack, LANES>(&parent.out, gradient);
                vec![(parent_id, parent_gradient)]
            }
            TensorOp::Exp => {
                let parent_id = node.parents[0];
                let parent_gradient =
                    Backend::zip_op::<SimdMul, LANES>(&node.out, gradient);
                vec![(parent_id, parent_gradient)]
            }
            TensorOp::Sum { orig_shape } => {
                let parent_id = node.parents[0];
                vec![(parent_id, gradient.broadcast_to(orig_shape))]
            }
            TensorOp::View { orig_shape } => {
                let parent_id = node.parents[0];
                let parent_gradient = TensorData {
                    storage: Rc::clone(&gradient.contiguous().storage),
                    storage_offset: 0,
                    shape: orig_shape.clone(),
                    strides: contiguous_strides(orig_shape),
                };
                vec![(parent_id, parent_gradient)]
            }
            TensorOp::Permute { axes } => {
                let parent_id = node.parents[0];
                let mut inverse_permutation = vec![0; axes.len()];
                for i in 0..axes.len() {
                    inverse_permutation[axes[i]] = i
                }
                vec![(parent_id, gradient.permute(&inverse_permutation))]
            }
            TensorOp::MatMul => {
                let left_parent_id = node.parents[0];
                let left_parent = &self.nodes[left_parent_id.0];
                let right_parent_id = node.parents[1];
                let right_parent = &self.nodes[right_parent_id.0];
                let left_gradient =
                    Backend::matmul(gradient, &right_parent.out.permute_last_two());
                let left_gradient_reduced =
                    maybe_reduce_broadcast(&left_gradient, &left_parent.out.shape);
                let right_gradient =
                    Backend::matmul(&left_parent.out.permute_last_two(), gradient);
                let right_gradient_reduced =
                    maybe_reduce_broadcast(&right_gradient, &right_parent.out.shape);
                vec![
                    (left_parent_id, left_gradient_reduced),
                    (right_parent_id, right_gradient_reduced),
                ]
            }
            TensorOp::Broadcast { orig_shape } => {
                let parent_id = node.parents[0];
                let parent_gradient = maybe_reduce_broadcast(gradient, orig_shape);
                vec![(parent_id, parent_gradient)]
            }
        }
    }

    /// For a given tensor operation graph and id of the root, propagates the gradient
    /// from the root to all other nodes in the graph
    pub fn backpropagate(&mut self, root: TensorNodeId) {
        let order = self.topological_sort(root).into_iter().rev();
        for id in order {
            let grads = self.compute_backward(id);
            for (parent_id, new_grad) in grads {
                let parent = &mut self.nodes[parent_id.0];
                parent.gradient = match parent.gradient.take() {
                    Some(old_grad) => {
                        Some(Backend::zip_op::<SimdAdd, LANES>(&old_grad, &new_grad))
                    }
                    None => Some(new_grad),
                };
            }
        }
    }

    /// Clears all gradients in a given tensor computational graph
    pub fn clear_gradients(&mut self) {
        for node in &mut self.nodes {
            node.gradient = None;
        }
    }
}

/// For a given tensor data and target shape, performs reductions on the tensor until
/// target shape is reached. If shapes match, returns the clone of the input tensor
pub fn maybe_reduce_broadcast(
    grad: &TensorData,
    target_shape: &[usize],
) -> TensorData {
    let mut result: TensorData = grad.clone();
    if result.shape == target_shape {
        return result;
    }
    while result.ndim() > target_shape.len() {
        result = Backend::reduce(&result, operators::add, 0., 0, false);
    }
    let result_shape = result.shape.clone();
    for (i, (&target_size, result_size)) in
        target_shape.iter().zip(result_shape).enumerate()
    {
        if target_size == 1 && result_size > 1 {
            result = Backend::reduce(&result, operators::add, 0., i, true);
        }
    }
    assert_eq!(
        result.shape, target_shape,
        "expected the result shape to be equal to the input target shape"
    );
    result
}

impl Default for TensorGraph {
    fn default() -> Self {
        Self::new()
    }
}
