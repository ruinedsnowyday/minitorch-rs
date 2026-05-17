use crate::autodiff::{NodeId, ScalarGraph, ScalarOp};
use crate::module::Module;
use crate::nn::Linear;

pub struct Network {
    pub l1: Linear,
    pub l2: Linear,
    pub l3: Linear,
    training: bool,
}

fn apply_elementwise(
    f: impl Fn(f64) -> ScalarOp,
    graph: &mut ScalarGraph,
    input_ids: Vec<NodeId>,
) -> Vec<NodeId> {
    input_ids
        .into_iter()
        .map(|id| graph.apply(f(graph.get_node(id).out), vec![id]))
        .collect()
}

impl Network {
    pub fn new(input_dim: usize, hidden: usize, output_dim: usize) -> Self {
        Network {
            l1: Linear::new(input_dim, hidden),
            l2: Linear::new(hidden, hidden),
            l3: Linear::new(hidden, output_dim),
            training: false,
        }
    }

    pub fn forward(
        &mut self,
        graph: &mut ScalarGraph,
        input_ids: Vec<NodeId>,
    ) -> Vec<NodeId> {
        let l1_out = self.l1.forward(graph, input_ids);
        let l1_act = apply_elementwise(ScalarOp::ReLU, graph, l1_out);
        let l2_out = self.l2.forward(graph, l1_act);
        let l2_act = apply_elementwise(ScalarOp::ReLU, graph, l2_out);
        let l3_out = self.l3.forward(graph, l2_act);
        apply_elementwise(ScalarOp::Sigmoid, graph, l3_out)
    }
}

impl Module for Network {
    fn training(&self) -> bool {
        self.training
    }

    fn children(&self) -> Vec<(&str, &dyn Module)> {
        vec![("l1", &self.l1), ("l2", &self.l2), ("l3", &self.l3)]
    }

    fn children_mut(&mut self) -> Vec<(&str, &mut dyn Module)> {
        vec![
            ("l1", &mut self.l1),
            ("l2", &mut self.l2),
            ("l3", &mut self.l3),
        ]
    }

    fn parameters(&self) -> Vec<(String, &f64)> {
        vec![]
    }

    fn set_train(&mut self) {
        self.training = true;
    }

    fn set_eval(&mut self) {
        self.training = false;
    }
}
