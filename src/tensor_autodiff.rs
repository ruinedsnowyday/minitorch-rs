use std::{collections::HashSet, rc::Rc};

use crate::tensor_data::TensorData;

#[derive(Clone, Copy, Eq, PartialEq, Debug, Hash)]
pub struct TensorNodeId(pub usize);

pub struct TensorGraph {
    pub nodes: Vec<TensorNode>,
}

pub struct TensorNode {
    pub op: TensorOp,
    pub out: Rc<TensorData>,
    pub gradient: Option<TensorData>,
    pub parents: Vec<TensorNodeId>,
}

pub enum TensorOp {
    Leaf,
    Add,
    Mul,
    Neg,
    Sigmoid,
    ReLU,
    Log,
    Exp,
    Sum { dim: usize, orig_shape: Vec<usize> },
    View { orig_shape: Vec<usize> },
    Permute { axes: Vec<usize> },
    MatMul,
    Broadcast { orig_shape: Vec<usize> },
}

impl TensorGraph {
    pub fn new() -> Self {
        TensorGraph { nodes: vec![] }
    }

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
}

impl Default for TensorGraph {
    fn default() -> Self {
        Self::new()
    }
}
