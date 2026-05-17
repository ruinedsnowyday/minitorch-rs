// Train a scalar Network on the `split` dataset.
//   label = 1 iff x0 not in [0.2, 0.8]   (two boundaries on x0)
// NOT linearly separable — needs the hidden layer to carve out two decision
// regions. Wider hidden layer + more epochs than the linear datasets.
// Run with: cargo run --release --example train_split

use minitorch_rs::datasets::split;
use minitorch_rs::loss::bce;
use minitorch_rs::train::train;
use minitorch_rs::xor_network::Network;

fn main() {
    let n = 100;
    let hidden = 8;
    let epochs = 300;
    let lr = 0.1;

    let dataset = split(n);
    let mut net = Network::new(2, hidden, 1);

    println!("training on `split`: n={n}, hidden={hidden}, lr={lr}, epochs={epochs}");
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
