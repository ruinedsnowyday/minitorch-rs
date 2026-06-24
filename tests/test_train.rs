use minitorch_rs::datasets::{simple, simple_with};
use minitorch_rs::loss::bce;
use minitorch_rs::train::{train, train_with};
use minitorch_rs::xor_network::Network;
use rand::SeedableRng;
use rand::rngs::StdRng;

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

// `simple` is linearly separable on x0 (label = 1 iff x0 < 0.5), so the
// network should comfortably exceed 0.85 accuracy.
//
// Fully reproducible: every randomness source — dataset, weight init, and the
// per-epoch shuffle — is driven by ONE seeded `StdRng`, so this test is
// deterministic and cannot flake. We assert over a few fixed seeds (a
// deterministic robustness check, not one lucky draw); each was verified to
// reach ~0.98 accuracy well before epoch 100. If a change breaks training,
// several fail together. (Without seeding, random ReLU init has a ~2% chance of
// a dead-layer collapse even at 1000 epochs — that nondeterminism is the
// flakiness this kills.)
#[test]
fn test_train_converges_on_simple() {
    for seed in [0u64, 1, 2] {
        let mut rng = StdRng::seed_from_u64(seed);
        let dataset = simple_with(50, &mut rng);
        let mut net = Network::new_with(2, 4, 1, &mut rng);

        let result = train_with(&mut net, &dataset, 200, 0.1, bce, &mut rng);

        let final_acc = *result.acc_history.last().unwrap();
        assert!(
            final_acc > 0.85,
            "seed {seed}: expected accuracy > 0.85 after 200 epochs, got {final_acc}"
        );
    }
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
