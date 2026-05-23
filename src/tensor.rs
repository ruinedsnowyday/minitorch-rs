use std::{
    cell::RefCell,
    ops::{Add, Mul, Neg},
    rc::Rc,
};

use crate::{
    operators,
    tensor_autodiff::{TensorGraph, TensorNodeId, TensorOp},
    tensor_data::TensorData,
    tensor_ops::{SimpleOps, TensorOps},
};

#[derive(Clone)]
pub struct Tensor {
    pub data: Rc<TensorData>,
    pub history: Option<History>,
}

#[derive(Clone)]
pub struct History {
    pub graph: Rc<RefCell<TensorGraph>>,
    pub node_id: TensorNodeId,
}

impl Tensor {
    pub fn from_data(data: TensorData) -> Self {
        Tensor {
            data: Rc::new(data),
            history: None,
        }
    }

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
}

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
            let graph = pick_shared_graph(&a, &b);
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

fn pick_shared_graph(a: &Tensor, b: &Tensor) -> Rc<RefCell<TensorGraph>> {
    match (&a.history, &b.history) {
        // Both in the same graph — use it.
        (Some(ha), Some(hb)) if Rc::ptr_eq(&ha.graph, &hb.graph) => {
            Rc::clone(&ha.graph)
        }

        // Only one has history — use that operand's graph.
        (Some(ha), None) => Rc::clone(&ha.graph),
        (None, Some(hb)) => Rc::clone(&hb.graph),

        // Both have history but different graphs — same panic story.
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
