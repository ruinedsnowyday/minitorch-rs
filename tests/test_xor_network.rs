use minitorch_rs::autodiff::ScalarGraph;
use minitorch_rs::module::Module;
use minitorch_rs::xor_network::Network;

// Forward returns one value per output neuron, and the sigmoid head bounds
// it strictly inside (0, 1). Sanity check that the chain compiles and runs.
#[test]
fn test_network_forward_output_in_sigmoid_range() {
    let mut net = Network::new(2, 4, 1);
    let mut graph = ScalarGraph::new();
    let x_ids: Vec<_> = [0.3_f64, 0.7].iter().map(|v| graph.add_leaf(*v)).collect();

    let y_ids = net.forward(&mut graph, x_ids);

    assert_eq!(y_ids.len(), 1);
    let y = graph.get_node(y_ids[0]).out;
    assert!(y.is_finite(), "output should be finite, got {y}");
    assert!(
        y > 0.0 && y < 1.0,
        "sigmoid output should be in (0, 1), got {y}"
    );
}

// The default Module::step recurses via children_mut. After a forward and
// backward through the whole network, calling step must update weights in
// every Linear, not just the last one. This is the load-bearing test for
// the trait recursion.
//
// (Probabilistic: relies on Xavier init not landing every Linear in a
// fully-dead-ReLU regime for this single input. Extremely reliable in
// practice; if it ever flakes, replace with explicit deterministic weights.)
#[test]
fn test_network_step_recursion_updates_all_layers() {
    let mut net = Network::new(2, 4, 1);
    let l1_before = net.l1.weights.clone();
    let l2_before = net.l2.weights.clone();
    let l3_before = net.l3.weights.clone();

    let mut graph = ScalarGraph::new();
    let x_ids: Vec<_> = [0.3_f64, 0.7].iter().map(|v| graph.add_leaf(*v)).collect();
    let y_ids = net.forward(&mut graph, x_ids);

    // backward from the output as if it were the loss
    graph.backpropagate(y_ids[0], 1.0);

    // calls the default Module::step, which recurses through children_mut
    net.step(&graph, 0.1);

    assert_ne!(net.l1.weights, l1_before, "l1.weights should have changed");
    assert_ne!(net.l2.weights, l2_before, "l2.weights should have changed");
    assert_ne!(net.l3.weights, l3_before, "l3.weights should have changed");
}

// named_parameters walks children with dotted prefixes. For Network(2, 4, 1):
//   l1: 2*4 weights + 4 biases = 12
//   l2: 4*4 weights + 4 biases = 20
//   l3: 4*1 weights + 1 bias   = 5
//   total                      = 37
#[test]
fn test_network_named_parameters_has_dotted_names() {
    let net = Network::new(2, 4, 1);
    let named = net.named_parameters();

    assert_eq!(named.len(), 37);

    let names: Vec<&str> = named.iter().map(|(n, _)| n.as_str()).collect();
    assert!(names.contains(&"l1.w_0,0"));
    assert!(names.contains(&"l1.b_3"));
    assert!(names.contains(&"l2.w_3,3"));
    assert!(names.contains(&"l3.w_0,3"));
    assert!(names.contains(&"l3.b_0"));
}

// train()/eval() on the Network must propagate to every child via
// children_mut, not just flip the flag on Network itself.
#[test]
fn test_network_train_eval_recursion() {
    let mut net = Network::new(2, 4, 1);
    // Linear::new and Network::new both default to training=false
    assert!(!net.training());
    assert!(!net.l1.training());

    net.train();
    assert!(net.training());
    assert!(net.l1.training());
    assert!(net.l2.training());
    assert!(net.l3.training());

    net.eval();
    assert!(!net.training());
    assert!(!net.l1.training());
    assert!(!net.l2.training());
    assert!(!net.l3.training());
}
