use rand::Rng;

pub struct Graph {
    pub n: usize,
    pub x: Vec<(f64, f64)>,
    pub y: Vec<i32>,
}

pub fn make_pts(n: usize) -> Vec<(f64, f64)> {
    let mut rng = rand::rng();
    (0..n).map(|_| (rng.random(), rng.random())).collect()
}

pub fn simple(n: usize) -> Graph {
    let x = make_pts(n);
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
