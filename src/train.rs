use crate::{
    autodiff::{NodeId, ScalarGraph},
    datasets::Graph,
    module::Module,
    xor_network::Network,
};
use rand::seq::SliceRandom;

pub struct TrainingResult {
    pub loss_history: Vec<f64>,
    pub acc_history: Vec<f64>,
}

pub fn train(
    network: &mut Network,
    dataset: &Graph,
    epochs: usize,
    lr: f64,
    loss_fn: impl Fn(&mut ScalarGraph, NodeId, f64) -> NodeId,
) -> TrainingResult {
    network.train();
    let mut rng = rand::rng();
    let mut result = TrainingResult {
        loss_history: vec![],
        acc_history: vec![],
    };
    for _ in 0..epochs {
        let mut total_loss = 0.;
        let mut correct: usize = 0;
        let mut indices: Vec<usize> = (0..dataset.n).collect();
        indices.shuffle(&mut rng);
        for i in indices {
            let mut graph = ScalarGraph::new();
            let (x0, x1) = dataset.x[i];
            let label = dataset.y[i];
            let in_ids = vec![graph.add_leaf(x0), graph.add_leaf(x1)];
            let y_ids = network.forward(&mut graph, in_ids);
            let pred_id = y_ids[0];
            let loss_id = loss_fn(&mut graph, pred_id, label as f64);
            total_loss += graph.get_node(loss_id).out;
            let pred_class = if graph.get_node(pred_id).out > 0.5 {
                1
            } else {
                0
            };
            if pred_class == label {
                correct += 1
            };
            graph.backpropagate(loss_id, 1.);
            network.step(&graph, lr);
        }
        result.loss_history.push(total_loss / dataset.n as f64);
        result.acc_history.push(correct as f64 / dataset.n as f64);
    }
    result
}
