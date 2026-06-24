const EPS: f64 = 1e-6;

/// leaky ReLU negative slope constant
const ALPHA: f64 = 1e-2;

pub fn mul(a: f64, b: f64) -> f64 {
    a * b
}

pub fn add(a: f64, b: f64) -> f64 {
    a + b
}

pub fn neg(a: f64) -> f64 {
    -a
}

pub fn id(a: f64) -> f64 {
    a
}

pub fn lt(a: f64, b: f64) -> f64 {
    if a < b { 1. } else { 0. }
}

pub fn eq(a: f64, b: f64) -> f64 {
    if a == b { 1. } else { 0. }
}

pub fn max(a: f64, b: f64) -> f64 {
    a.max(b)
}

pub fn is_close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-2
}

pub fn sigmoid(a: f64) -> f64 {
    if a > 0. {
        1. / (1. + (-a).exp())
    } else {
        a.exp() / (1. + a.exp())
    }
}

pub fn relu(a: f64) -> f64 {
    a.max(0.)
}

pub fn leaky_relu(a: f64) -> f64 {
    if a > 0. { a } else { a * ALPHA }
}

pub fn log(a: f64) -> f64 {
    (a + EPS).ln()
}

pub fn exp(a: f64) -> f64 {
    a.exp()
}

pub fn inv(a: f64) -> f64 {
    a.recip()
}

pub fn log_back(a: f64, d: f64) -> f64 {
    (a + EPS).recip() * d
}

pub fn inv_back(a: f64, d: f64) -> f64 {
    -a.recip().powi(2) * d
}

pub fn relu_back(a: f64, d: f64) -> f64 {
    if a > 0. { d } else { 0. }
}

pub fn leaky_relu_back(a: f64, d: f64) -> f64 {
    if a > 0. { d } else { ALPHA * d }
}

pub fn sigmoid_back(post_a: f64, d: f64) -> f64 {
    post_a * (1.0 - post_a) * d
}

pub fn map<F>(f: F) -> impl Fn(&[f64]) -> Vec<f64>
where
    F: Fn(f64) -> f64,
{
    move |a: &[f64]| a.iter().map(|&b| f(b)).collect()
}

pub fn zip_with<F>(f: F) -> impl Fn(&[f64], &[f64]) -> Vec<f64>
where
    F: Fn(f64, f64) -> f64,
{
    move |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(&a, &b)| f(a, b)).collect()
}

pub fn reduce<F>(f: F, init: f64) -> impl Fn(&[f64]) -> f64
where
    F: Fn(f64, f64) -> f64,
{
    move |a: &[f64]| a.iter().fold(init, |a, &b| f(a, b))
}

pub fn neg_list(ls: &[f64]) -> Vec<f64> {
    let func = map(neg);
    func(ls)
}

pub fn add_lists(ls1: &[f64], ls2: &[f64]) -> Vec<f64> {
    let func = zip_with(add);
    func(ls1, ls2)
}

pub fn sum(ls: &[f64]) -> f64 {
    let func = reduce(add, 0.);
    func(ls)
}

pub fn prod(ls: &[f64]) -> f64 {
    let func = reduce(mul, 1.);
    func(ls)
}
