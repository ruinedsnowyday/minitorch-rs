use std::collections::HashSet;

use crate::operators;

pub enum ScalarOp {
    Add(f64, f64),
    Id(f64),
    Mul(f64, f64),
    Inv(f64),
    Neg(f64),
    Sigmoid(f64),
    ReLU(f64),
    Exp(f64),
    Log(f64),
    Lt(f64, f64),
    Eq(f64, f64),
}

impl ScalarOp {
    pub fn forward(&self) -> f64 {
        match self {
            ScalarOp::Mul(a, b) => operators::mul(*a, *b),
            ScalarOp::Inv(a) => operators::inv(*a),
            ScalarOp::Neg(a) => operators::neg(*a),
            ScalarOp::Sigmoid(a) => operators::sigmoid(*a),
            ScalarOp::ReLU(a) => operators::relu(*a),
            ScalarOp::Exp(a) => operators::exp(*a),
            ScalarOp::Lt(a, b) => operators::lt(*a, *b),
            ScalarOp::Eq(a, b) => operators::eq(*a, *b),
            ScalarOp::Add(a, b) => operators::add(*a, *b),
            ScalarOp::Log(a) => operators::log(*a),
            ScalarOp::Id(a) => operators::id(*a),
        }
    }
    pub fn backward(&self, out: f64, d: f64) -> Vec<f64> {
        match self {
            ScalarOp::Add(_, _) => vec![d, d],
            ScalarOp::Mul(a, b) => vec![*b * d, *a * d],
            ScalarOp::Inv(_) => vec![-out.powi(2) * d],
            ScalarOp::Neg(_) => vec![-d],
            ScalarOp::Sigmoid(_) => vec![out * (1. - out) * d],
            ScalarOp::ReLU(a) => vec![operators::relu_back(*a, d)],
            ScalarOp::Exp(_) => vec![out * d],
            ScalarOp::Log(a) => vec![operators::log_back(*a, d)],
            ScalarOp::Lt(_, _) => vec![0., 0.],
            ScalarOp::Eq(_, _) => vec![0., 0.],
            ScalarOp::Id(_) => vec![d],
        }
    }
}

#[derive(Clone, Copy, Eq, PartialEq, Debug, Hash)]
pub struct NodeId(pub usize);

pub struct ScalarNode {
    pub op: ScalarOp,
    pub out: f64,
    pub gradient: f64,
    pub parents: Vec<NodeId>,
}

impl ScalarNode {
    /// Given d_output (the upstream gradient), compute and return pairs of
    /// (input_variable, local_gradient) for each input to this node's operation.
    pub fn chain_rule(&self, d: f64) -> Vec<(NodeId, f64)> {
        let grads = self.op.backward(self.out, d);
        self.parents
            .iter()
            .zip(grads)
            .map(|(idx, grad)| (*idx, grad))
            .collect()
    }
}

pub struct ScalarGraph {
    nodes: Vec<ScalarNode>,
}

impl ScalarGraph {
    pub fn new() -> Self {
        ScalarGraph { nodes: vec![] }
    }

    pub fn add_leaf(&mut self, val: f64) -> NodeId {
        let out = NodeId(self.nodes.len());
        let leaf = ScalarNode {
            op: ScalarOp::Id(val),
            out: val,
            gradient: 0.,
            parents: vec![],
        };
        self.nodes.push(leaf);
        out
    }

    pub fn apply(&mut self, op: ScalarOp, input_nodes: Vec<NodeId>) -> NodeId {
        let out_node = NodeId(self.nodes.len());
        let out = op.forward();
        let new_node = ScalarNode {
            op,
            out,
            gradient: 0.,
            parents: input_nodes,
        };
        self.nodes.push(new_node);
        out_node
    }

    pub fn get_node(&self, idx: NodeId) -> &ScalarNode {
        &self.nodes[idx.0]
    }

    pub fn get_node_mut(&mut self, idx: NodeId) -> &mut ScalarNode {
        &mut self.nodes[idx.0]
    }

    pub fn topological_sort(&self, out_node: NodeId) -> Vec<NodeId> {
        let mut order: Vec<NodeId> = Vec::with_capacity(self.nodes.len());
        let mut visited: HashSet<NodeId> = HashSet::with_capacity(self.nodes.len());
        fn visit(
            idx: NodeId,
            visited: &mut HashSet<NodeId>,
            order: &mut Vec<NodeId>,
            nodes: &[ScalarNode],
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
        visit(out_node, &mut visited, &mut order, &self.nodes);
        order
    }

    pub fn backpropagate(&mut self, out_node: NodeId, d: f64) {
        let order = self.topological_sort(out_node).into_iter().rev();
        self.get_node_mut(out_node).gradient = d;
        for id in order {
            let current_grad = self.get_node(id).gradient;
            for (parent, grad) in self.get_node(id).chain_rule(current_grad) {
                self.get_node_mut(parent).gradient += grad;
            }
        }
    }
}

impl Default for ScalarGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Computes an approximation to the derivative of `f` with respect
/// to argument `arg` using central differences.
///
/// central_difference(f, &[x0, ..., xn], arg, epsilon)
///   = (f(x0, ..., x_arg + eps, ..., xn) - f(x0, ..., x_arg - eps, ..., xn))
///     / (2 * epsilon)
pub fn central_difference(
    f: impl Fn(&[f64]) -> f64,
    vals: &[f64],
    arg: usize,
    epsilon: f64,
) -> f64 {
    let mut mutable_copy: Vec<f64> = vals.to_vec();
    mutable_copy[arg] += epsilon;
    let f1 = f(&mutable_copy);
    mutable_copy[arg] -= 2. * epsilon;
    let f2 = f(&mutable_copy);
    (f1 - f2) / (2. * epsilon)
}
