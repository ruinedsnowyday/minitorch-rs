use minitorch_rs::datasets::simple;
use minitorch_rs::loss::bce;
use minitorch_rs::train::train;
use minitorch_rs::xor_network::Network;

// The weakest possible "did the loop learn" check: after some epochs,
// mean loss must be lower than at the start. If this fails, either
// the gradient flow is broken or step is updating in the wrong direction.
#[test]
fn test_train_loss_decreases_on_simple() {
    let dataset = simple(30);
    let mut net = Network::new(2, 4, 1);

    let result = train(&mut net, &dataset, 30, 0.1, bce);

    let first = *result.loss_history.first().unwrap();
    let last = *result.loss_history.last().unwrap();
    assert!(
        last < first,
        "loss should decrease over training: first={first}, last={last}"
    );
}

// `simple` is linearly separable on x0 (label = 1 iff x0 < 0.5), so even
// a single sigmoid neuron could solve it. A 3-layer Network with hidden=4
// should comfortably exceed 0.9 accuracy in a few hundred epochs.
//
// Threshold is intentionally loose (0.85) to absorb RNG variance from
// Xavier init and the per-epoch shuffle, since the loop uses the global
// thread-local RNG which we can't seed here.
#[test]
fn test_train_converges_on_simple() {
    let dataset = simple(50);
    let mut net = Network::new(2, 4, 1);

    let result = train(&mut net, &dataset, 200, 0.1, bce);

    let final_acc = *result.acc_history.last().unwrap();
    assert!(
        final_acc > 0.85,
        "expected accuracy > 0.85 on simple after 200 epochs, got {final_acc}"
    );
}

// TrainingResult vec lengths must equal `epochs`. Trivial check, catches
// off-by-one errors in the epoch loop.
#[test]
fn test_train_result_shape() {
    let dataset = simple(10);
    let mut net = Network::new(2, 4, 1);
    let epochs = 5;

    let result = train(&mut net, &dataset, epochs, 0.1, bce);

    assert_eq!(result.loss_history.len(), epochs);
    assert_eq!(result.acc_history.len(), epochs);
}
