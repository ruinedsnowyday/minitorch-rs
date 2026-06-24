use rand::Rng;

pub struct Graph {
    pub n: usize,
    pub x: Vec<(f64, f64)>,
    pub y: Vec<i32>,
}

pub fn make_pts(n: usize) -> Vec<(f64, f64)> {
    make_pts_with(n, &mut rand::rng())
}

/// Seeded variant: pass a fixed-seed RNG (e.g. `StdRng::seed_from_u64(..)`) for
/// reproducible data in tests. `make_pts` delegates here with the global RNG.
pub fn make_pts_with(n: usize, rng: &mut impl Rng) -> Vec<(f64, f64)> {
    (0..n).map(|_| (rng.random(), rng.random())).collect()
}

pub fn simple(n: usize) -> Graph {
    simple_with(n, &mut rand::rng())
}

pub fn simple_with(n: usize, rng: &mut impl Rng) -> Graph {
    let x = make_pts_with(n, rng);
    let y: Vec<i32> = x
        .iter()
        .map(|&(x1, _)| if x1 < 0.5 { 1 } else { 0 })
        .collect();
    Graph { n, x, y }
}

pub fn diag(n: usize) -> Graph {
    let x = make_pts(n);
    let y: Vec<i32> = x
        .iter()
        .map(|&(x1, x2)| if x1 + x2 < 1.0 { 1 } else { 0 })
        .collect();
    Graph { n, x, y }
}

pub fn split(n: usize) -> Graph {
    let x = make_pts(n);
    let y: Vec<i32> = x
        .iter()
        .map(|(x1, _)| if !(0.2..0.8).contains(x1) { 1 } else { 0 })
        .collect();
    Graph { n, x, y }
}

pub fn xor(n: usize) -> Graph {
    let x = make_pts(n);
    let y: Vec<i32> = x
        .iter()
        .map(|&(x1, x2)| if (x1 < 0.5) != (x2 < 0.5) { 1 } else { 0 })
        .collect();
    Graph { n, x, y }
}
