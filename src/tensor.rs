use std::{
    cell::RefCell,
    ops::{Add, Mul, Neg},
    rc::Rc,
};

use crate::{
    operators,
    tensor_autodiff::{TensorGraph, TensorNodeId, TensorOp},
    tensor_data::{TensorData, contiguous_strides},
    tensor_ops::{SimpleOps, TensorOps},
};

/// Tensor struct holding the reference to the actual data as well as optionally
/// the computational history of the tensor
#[derive(Clone)]
pub struct Tensor {
    pub data: Rc<TensorData>,
    pub history: Option<History>,
}

/// History struct holding a reference to the graph and id of the corresponding node
/// in this graph
#[derive(Clone)]
pub struct History {
    pub graph: Rc<RefCell<TensorGraph>>,
    pub node_id: TensorNodeId,
}

impl Tensor {
    /// Creates a tensor from provided tensor data without assigning it to a
    /// computational graph
    pub fn from_data(data: TensorData) -> Self {
        Tensor {
            data: Rc::new(data),
            history: None,
        }
    }

    /// Turns input tensor into a new tensor that actually has history with the input
    /// tensor being a leaf of the graph. Panics if tensor is already in a
    /// computational graph
    pub fn requires_grad(self) -> Self {
        // attaches new graph + leaf
        assert!(
            self.history.is_none(),
            "requires_grad can only be called on a leaf tensor (one without history)"
        );
        let mut graph = TensorGraph::new();
        let node_id = graph.add_leaf(self.data.clone());
        let history = History {
            graph: Rc::new(RefCell::new(graph)),
            node_id,
        };
        Tensor {
            data: self.data,
            history: Some(history),
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.data.shape
    }

    pub fn ndim(&self) -> usize {
        self.data.ndim()
    }

    pub fn size(&self) -> usize {
        self.data.size()
    }

    /// returns an only item in 0-d/1-element tensor. Panics in other cases
    pub fn item(&self) -> f64 {
        // only for 0-d / 1-element tensors
        assert!(
            self.ndim() == 0 || self.size() == 1,
            "expected the tensor to be either 0-dimensional or containing 1 element, got tensor with size {:?}",
            self.size()
        );
        self.data.storage[self.data.storage_offset]
    }

    /// Tries to return the id of the tensor in a given computational graph. If
    /// tensor has history in the input graph, returns the node id from the history.
    /// If graphs in tensor history and input graph don't match, panics. Finally, if
    /// tensor doesn't have history, adds tensor data to the graph as a leaf and
    /// returns the new id
    pub fn node_id_in(&self, graph: &Rc<RefCell<TensorGraph>>) -> TensorNodeId {
        match &self.history {
            Some(h) if Rc::ptr_eq(&h.graph, graph) => h.node_id,
            Some(_) => panic!("operands belong to different autograd graphs"),
            None => graph.borrow_mut().add_leaf(Rc::clone(&self.data)),
        }
    }

    pub fn relu(&self) -> Tensor {
        make_unary_op(self, TensorOp::ReLU, |in_data| {
            SimpleOps::map(in_data, operators::relu)
        })
    }

    pub fn sigmoid(&self) -> Tensor {
        make_unary_op(self, TensorOp::Sigmoid, |in_data| {
            SimpleOps::map(in_data, operators::sigmoid)
        })
    }

    pub fn log(&self) -> Tensor {
        make_unary_op(self, TensorOp::Log, |in_data| {
            SimpleOps::map(in_data, operators::log)
        })
    }

    pub fn sum(&self, dim: usize) -> Tensor {
        make_unary_op(
            self,
            TensorOp::Sum {
                dim,
                orig_shape: self.shape().to_vec(),
            },
            |in_data| SimpleOps::reduce(in_data, operators::add, 0., dim, true),
        )
    }

    pub fn exp(&self) -> Tensor {
        make_unary_op(self, TensorOp::Exp, |in_data| {
            SimpleOps::map(in_data, operators::exp)
        })
    }

    /// Creates a view of a tensor with a given shape. Panics if the tensor is not stored
    /// contiguously
    pub fn view(&self, shape: &[usize]) -> Tensor {
        assert!(
            self.size() == shape.iter().product(),
            "target shape's size must equal the tensor's size"
        );
        assert!(
            contiguous_strides(self.shape()) == self.data.strides,
            "expected the data to be stored contiguously in tensor for view"
        );
        make_unary_op(
            self,
            TensorOp::View {
                orig_shape: self.shape().to_vec(),
            },
            |in_data| TensorData {
                storage: in_data.storage.clone(),
                storage_offset: in_data.storage_offset,
                shape: shape.to_vec(),
                strides: contiguous_strides(shape),
            },
        )
    }

    /// Permutes the axes of the tensor, producing a new tensor. Updates computational
    /// graph if present
    pub fn permute(&self, axes: &[usize]) -> Tensor {
        make_unary_op(
            self,
            TensorOp::Permute {
                axes: axes.to_vec(),
            },
            |in_data| in_data.permute(axes),
        )
    }

    /// Performs matrix multiplication between two tensors. Updates computational
    /// if present
    pub fn matmul(&self, right: &Tensor) -> Tensor {
        make_binary_op_ref(self, right, TensorOp::MatMul, |l, r| {
            SimpleOps::matmul(l, r)
        })
    }

    /// Broadcasts tensor to a given shape. Updates computational graph if present
    pub fn broadcast_to(&self, target_shape: &[usize]) -> Tensor {
        make_unary_op(
            self,
            TensorOp::Broadcast {
                orig_shape: self.data.shape.to_vec(),
            },
            |in_data| in_data.broadcast_to(target_shape),
        )
    }

    /// Seeds the gradient of the tensor with a tensor of ones and starts the
    /// backpropagation if the tensor has history
    pub fn backward(&self) {
        if let Some(h) = &self.history {
            let mut graph = h.graph.borrow_mut();
            graph.clear_gradients();
            let node = &mut graph.nodes[h.node_id.0];
            node.gradient = Some(TensorData::ones(node.out.shape.clone()));
            graph.backpropagate(h.node_id);
        }
    }

    /// Returns the gradient accumulated at the given tensor
    pub fn grad(&self) -> Option<TensorData> {
        match &self.history {
            Some(History { graph, node_id }) => {
                graph.borrow().nodes[node_id.0].gradient.clone()
            }
            None => None,
        }
    }
}

/// Performs a unary operation on the tensor, given the type of operation and a
/// function that maps tensor data reference to new tensor data. If input has history,
/// updates computational graph correspondingly
fn make_unary_op(
    input: &Tensor,
    op: TensorOp,
    compute: impl FnOnce(&TensorData) -> TensorData,
) -> Tensor {
    // compute output data
    let out = compute(&input.data);
    let out_rc = Rc::new(out);

    let history = input.history.as_ref().map(|h| {
        let graph = Rc::clone(&h.graph);
        let node_id = graph
            .borrow_mut()
            .apply(op, out_rc.clone(), vec![h.node_id]);
        History { graph, node_id }
    });

    Tensor {
        data: out_rc,
        history,
    }
}

/// Performs a binary tensor operation on two tensors. Performs compute, picks
/// a common computational graph (if it can) and updates it with new operation
fn make_binary_op_ref(
    a: &Tensor,
    b: &Tensor,
    op: TensorOp,
    compute: impl FnOnce(&TensorData, &TensorData) -> TensorData,
) -> Tensor {
    // compute output data
    let out = compute(&a.data, &b.data);
    let out_rc = Rc::new(out);

    let history = match (&a.history, &b.history) {
        (None, None) => None,
        _ => {
            let graph = pick_shared_graph(a, b);
            let a_id = a.node_id_in(&graph);
            let b_id = b.node_id_in(&graph);
            let node_id =
                graph
                    .borrow_mut()
                    .apply(op, out_rc.clone(), vec![a_id, b_id]);
            Some(History { graph, node_id })
        }
    };

    Tensor {
        data: out_rc,
        history,
    }
}

/// For two given tensors, returns a common computational graph, panicing if two
/// tensors belong to different computational graph
fn pick_shared_graph(a: &Tensor, b: &Tensor) -> Rc<RefCell<TensorGraph>> {
    match (&a.history, &b.history) {
        // Both in the same graph — use it.
        (Some(ha), Some(hb)) if Rc::ptr_eq(&ha.graph, &hb.graph) => {
            Rc::clone(&ha.graph)
        }

        // Only one has history — use that operand's graph.
        (Some(ha), None) => Rc::clone(&ha.graph),
        (None, Some(hb)) => Rc::clone(&hb.graph),

        // Both have history but different graphs - same panic story.
        (Some(_), Some(_)) => panic!("operands belong to different autograd graphs"),

        // (None, None) is short-circuited above the call site, so unreachable here.
        (None, None) => unreachable!("caller checks for (None, None) before calling"),
    }
}

impl Add<&Tensor> for &Tensor {
    type Output = Tensor;

    fn add(self, rhs: &Tensor) -> Self::Output {
        make_binary_op_ref(self, rhs, TensorOp::Add, |d1, d2| {
            SimpleOps::zip(d1, d2, operators::add)
        })
    }
}

impl Add<Tensor> for Tensor {
    type Output = Tensor;

    fn add(self, rhs: Tensor) -> Self::Output {
        (&self).add(&rhs)
    }
}

impl Mul<&Tensor> for &Tensor {
    type Output = Tensor;

    fn mul(self, rhs: &Tensor) -> Self::Output {
        make_binary_op_ref(self, rhs, TensorOp::Mul, |d1, d2| {
            SimpleOps::zip(d1, d2, operators::mul)
        })
    }
}

impl Mul<Tensor> for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: Tensor) -> Self::Output {
        (&self).mul(&rhs)
    }
}

impl Neg for &Tensor {
    type Output = Tensor;

    fn neg(self) -> Self::Output {
        make_unary_op(self, TensorOp::Neg, |d| SimpleOps::map(d, operators::neg))
    }
}

impl Neg for Tensor {
    type Output = Tensor;

    fn neg(self) -> Self::Output {
        (&self).neg()
    }
}
