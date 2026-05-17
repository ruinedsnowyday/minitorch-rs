use rand::Rng;

use crate::autodiff::{NodeId, ScalarGraph, ScalarOp};
use crate::module::Module;

pub struct Linear {
    pub weights: Vec<Vec<f64>>,
    pub biases: Vec<f64>,
    in_size: usize,
    out_size: usize,
    training: bool,
    pub weight_ids: Option<Vec<Vec<NodeId>>>,
    pub bias_ids: Option<Vec<NodeId>>,
}

impl Linear {
    pub fn new(in_size: usize, out_size: usize) -> Self {
        let mut rng = rand::rng();
        // Xavier uniform: range = sqrt(6 / (fan_in + fan_out))
        let bound = (6.0 / (in_size + out_size) as f64).sqrt();
        let weights: Vec<Vec<f64>> = (0..out_size)
            .map(|_| {
                (0..in_size)
                    .map(|_| rng.random_range(-bound..bound))
                    .collect()
            })
            .collect();
        let bias: Vec<f64> = (0..out_size)
            .map(|_| rng.random_range(-bound..bound))
            .collect();
        Linear {
            weights,
            biases: bias,
            training: false,
            in_size,
            out_size,
            weight_ids: None,
            bias_ids: None,
        }
    }

    /// calling forward multiple times before calling backward overwrites the previous
    /// gradients, be careful with that
    pub fn forward(
        &mut self,
        graph: &mut ScalarGraph,
        in_ids: Vec<NodeId>,
    ) -> Vec<NodeId> {
        assert_eq!(in_ids.len(), self.in_size);

        let weight_ids: Vec<Vec<NodeId>> = self
            .weights
            .iter()
            .map(|row| row.iter().map(|w| graph.add_leaf(*w)).collect())
            .collect();

        let bias_ids: Vec<NodeId> =
            self.biases.iter().map(|b| graph.add_leaf(*b)).collect();

        // process each output
        let mut output_ids: Vec<NodeId> =
            (0..self.biases.len()).map(|_| NodeId(0)).collect();
        for i in 0..self.biases.len() {
            let mut acc_id = graph.apply(
                ScalarOp::Mul(graph.get_node(in_ids[0]).out, self.weights[i][0]),
                vec![in_ids[0], weight_ids[i][0]],
            );
            for (j, weight) in self.weights[i].iter().enumerate().skip(1) {
                let prod = graph.apply(
                    ScalarOp::Mul(graph.get_node(in_ids[j]).out, *weight),
                    vec![in_ids[j], weight_ids[i][j]],
                );
                acc_id = graph.apply(
                    ScalarOp::Add(
                        graph.get_node(acc_id).out,
                        graph.get_node(prod).out,
                    ),
                    vec![acc_id, prod],
                );
            }
            acc_id = graph.apply(
                ScalarOp::Add(graph.get_node(acc_id).out, self.biases[i]),
                vec![acc_id, bias_ids[i]],
            );
            output_ids[i] = acc_id;
        }
        self.weight_ids = Some(weight_ids);
        self.bias_ids = Some(bias_ids);
        output_ids
    }
}

impl Module for Linear {
    fn training(&self) -> bool {
        self.training
    }

    fn children(&self) -> Vec<(&str, &dyn Module)> {
        vec![]
    }

    fn children_mut(&mut self) -> Vec<(&str, &mut dyn Module)> {
        vec![]
    }

    fn parameters(&self) -> Vec<(String, &f64)> {
        let mut out: Vec<(String, &f64)> = Vec::new();
        for row in 0..self.out_size {
            for col in 0..self.in_size {
                out.push((format!("w_{row},{col}"), &self.weights[row][col]));
            }
            out.push((format!("b_{row}"), &self.biases[row]));
        }
        out
    }

    fn set_train(&mut self) {
        self.training = true;
    }

    fn set_eval(&mut self) {
        self.training = false;
    }

    fn step(&mut self, graph: &ScalarGraph, lr: f64) {
        if let (Some(weight_ids), Some(bias_ids)) = (&self.weight_ids, &self.bias_ids)
        {
            let mut grad: f64;
            for i in 0..self.out_size {
                for j in 0..self.in_size {
                    grad = graph.get_node(weight_ids[i][j]).gradient;
                    self.weights[i][j] -= grad * lr;
                }
                grad = graph.get_node(bias_ids[i]).gradient;
                self.biases[i] -= grad * lr;
            }
        }
    }
}
