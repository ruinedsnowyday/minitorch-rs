use crate::autodiff::{NodeId, ScalarGraph, ScalarOp};

pub fn mse(graph: &mut ScalarGraph, pred_id: NodeId, label: f64) -> NodeId {
    let target_id = graph.add_leaf(label);
    let neg_target_id = graph.apply(ScalarOp::Neg(label), vec![target_id]);
    let pred_val_id = graph.get_node(pred_id).out;
    let neg_target_val = graph.get_node(neg_target_id).out;
    let diff_id = graph.apply(
        ScalarOp::Add(pred_val_id, neg_target_val),
        vec![pred_id, neg_target_id],
    );
    let diff_val = graph.get_node(diff_id).out;
    graph.apply(ScalarOp::Mul(diff_val, diff_val), vec![diff_id, diff_id])
}

/// Numerically unstable when pred is near 0 or 1 due to log(p) and log(1-p).
/// Production frameworks fuse sigmoid+BCE on logits to avoid this.
pub fn bce(graph: &mut ScalarGraph, pred_id: NodeId, label: f64) -> NodeId {
    let target_id = graph.add_leaf(label);
    let neg_target_id = graph.apply(ScalarOp::Neg(label), vec![target_id]);
    let neg_target_val = graph.get_node(neg_target_id).out;
    let one_id = graph.add_leaf(1.);
    let one_val = graph.get_node(one_id).out;
    let pred_val = graph.get_node(pred_id).out;
    let neg_pred_id = graph.apply(ScalarOp::Neg(pred_val), vec![pred_id]);
    let neg_pred_val = graph.get_node(neg_pred_id).out;
    let one_m_p_id = graph.apply(
        ScalarOp::Add(one_val, neg_pred_val),
        vec![one_id, neg_pred_id],
    );
    let one_m_p_val = graph.get_node(one_m_p_id).out;
    let one_m_t_id = graph.apply(
        ScalarOp::Add(one_val, neg_target_val),
        vec![one_id, neg_target_id],
    );
    let one_m_t_val = graph.get_node(one_m_t_id).out;
    let log_p_id = graph.apply(ScalarOp::Log(pred_val), vec![pred_id]);
    let log_p_val = graph.get_node(log_p_id).out;
    let log_one_m_p_id = graph.apply(ScalarOp::Log(one_m_p_val), vec![one_m_p_id]);
    let log_one_m_p_val = graph.get_node(log_one_m_p_id).out;
    let t_log_p_id =
        graph.apply(ScalarOp::Mul(label, log_p_val), vec![target_id, log_p_id]);
    let t_log_p_val = graph.get_node(t_log_p_id).out;
    let second_id = graph.apply(
        ScalarOp::Mul(one_m_t_val, log_one_m_p_val),
        vec![one_m_t_id, log_one_m_p_id],
    );
    let second_val = graph.get_node(second_id).out;
    let sum_id = graph.apply(
        ScalarOp::Add(t_log_p_val, second_val),
        vec![t_log_p_id, second_id],
    );
    let sum_val = graph.get_node(sum_id).out;
    graph.apply(ScalarOp::Neg(sum_val), vec![sum_id])
}
