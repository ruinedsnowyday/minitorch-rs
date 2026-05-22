use std::{cell::RefCell, rc::Rc};

use crate::{
    tensor_autodiff::{TensorGraph, TensorNodeId},
    tensor_data::TensorData,
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
}
