// Train a scalar Network on the `simple` dataset.
//   label = 1 iff x0 < 0.5
// Linearly separable on x0 alone — a single sigmoid neuron could solve it.
// Run with: cargo run --release --example train_simple

use minitorch_rs::datasets::simple;
use minitorch_rs::loss::bce;
use minitorch_rs::train::train;
use minitorch_rs::xor_network::Network;

fn main() {
    let n = 100;
    let hidden = 4;
    let epochs = 100;
    let lr = 0.1;

    let dataset = simple(n);
    let mut net = Network::new(2, hidden, 1);

    println!("training on `simple`: n={n}, hidden={hidden}, lr={lr}, epochs={epochs}");
    let result = train(&mut net, &dataset, epochs, lr, bce);

    let report_every = (epochs / 10).max(1);
    for (i, (loss, acc)) in result
        .loss_history
        .iter()
        .zip(result.acc_history.iter())
        .enumerate()
    {
        if (i + 1) % report_every == 0 || i == 0 {
            println!("epoch {:4} | loss {:.4} | acc {:.3}", i + 1, loss, acc);
        }
    }

    println!(
        "\nfinal: loss={:.4}, acc={:.3}",
        result.loss_history.last().unwrap(),
        result.acc_history.last().unwrap()
    );
}
