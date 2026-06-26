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

// `simple` is linearly separable on x0 (label = 1 iff x0 < 0.5), so a healthy
// network comfortably exceeds 0.85 accuracy.
//
// Fully reproducible: every randomness source — dataset, weight init, per-epoch
// shuffle — is driven by ONE seeded `StdRng` per trial, so the test is
// deterministic and cannot flake.
//
// We sweep a *batch* of fixed seeds rather than one lucky draw. A single (or
// three) fixed seeds let a regression that tanks most inits but happens to
// spare the chosen seeds pass forever; sweeping many seeds restores that
// breadth. The catch is the known ~2% dead-layer-collapse rate of random ReLU
// init, so we don't demand that *every* seed converge — we allow up to
// COLLAPSE_BUDGET of them to miss. That tolerance is wide enough to never flake
// on an unlucky init, but a real regression breaks far more than two of ten and
// trips the assert. (A broad, mild degradation that drops everything below 0.85
// also trips it, since the converged count collapses to ~0.)
#[test]
fn test_train_converges_on_simple() {
    const SEEDS: u64 = 10;
    const COLLAPSE_BUDGET: usize = 2;

    let accs: Vec<f64> = (0..SEEDS)
        .map(|seed| {
            let mut rng = StdRng::seed_from_u64(seed);
            let dataset = simple_with(50, &mut rng);
            let mut net = Network::new_with(2, 4, 1, &mut rng);
            let result = train_with(&mut net, &dataset, 200, 0.1, bce, &mut rng);
            *result.acc_history.last().unwrap()
        })
        .collect();

    let converged = accs.iter().filter(|&&a| a > 0.85).count();
    assert!(
        converged >= SEEDS as usize - COLLAPSE_BUDGET,
        "expected >= {} of {SEEDS} seeds to reach >0.85 accuracy, got {converged}; per-seed: {accs:?}",
        SEEDS as usize - COLLAPSE_BUDGET
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
