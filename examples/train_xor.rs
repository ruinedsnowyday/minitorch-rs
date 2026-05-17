// Train a scalar Network on the `xor` dataset.
//   label = 1 iff (x0 < 0.5) XOR (x1 < 0.5)
// The classic non-linearly-separable problem. CANNOT be solved by a single
// linear layer; the hidden layer + ReLU nonlinearity is what makes it work.
// If `xor` converges, the whole module 1 stack is provably correct.
//
// Heads up: occasionally lands in a bad Xavier init and stalls. Just rerun.
// Run with: cargo run --release --example train_xor

use minitorch_rs::datasets::xor;
use minitorch_rs::loss::bce;
use minitorch_rs::train::train;
use minitorch_rs::xor_network::Network;

fn main() {
    let n = 100;
    let hidden = 8;
    let epochs = 500;
    let lr = 0.1;

    let dataset = xor(n);
    let mut net = Network::new(2, hidden, 1);

    println!("training on `xor`: n={n}, hidden={hidden}, lr={lr}, epochs={epochs}");
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
