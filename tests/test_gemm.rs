//! Equivalence tests for the microkernel-based GEMM drivers (`src/gemm.rs`).
//! Both drivers must agree with `SimpleOps::matmul` (the trusted oracle) on every
//! (m, k, n), especially NON-SQUARE shapes — when m==k==n every row-major stride
//! coincides and a packing / indexing bug passes on a square input.
//!
//! These are BLACK-BOX tests: they only compare `driver(a,b)` to the oracle, so
//! they say nothing about *how* you pack or tile — implement freely behind them.
//! Every test is `#[ignore]`d until its driver exists (the stubs `todo!()`-panic);
//! delete the `#[ignore]` line as you finish each stage, TDD-style.
//!
//! Suggested first (unit) test to add yourself once the microkernel compiles:
//! hand-build a tiny packed `ap`/`bp` with a known MR×NR product and assert the
//! kernel accumulates it correctly — that pins the packed-layout contract before
//! any driver exists.

use minitorch_rs::gemm::{matmul_iterative, matmul_recursive};
use minitorch_rs::tensor_data::TensorData;
use minitorch_rs::tensor_ops::{SimpleOps, TensorOps};
use proptest::prelude::*;

// 0,1,2,…(+start): small integers keep f64 products/sums bit-exact regardless of
// the summation order a tiled/recursive kernel uses, so storage compare is exact.
fn seq(shape: Vec<usize>, start: usize) -> TensorData {
    let n: usize = shape.iter().product();
    TensorData::new((0..n).map(|i| (start + i) as f64).collect(), shape)
}

// Distinct-dims shapes (the ones that expose stride/packing bugs) + a couple of
// squares (the case a bug sneaks past) + sizes that straddle the register tile
// (MR=NR=8) and leave ragged edges (10, 17) once you implement edge handling.
const SHAPES: [(usize, usize, usize); 8] = [
    (2, 3, 4),
    (5, 2, 7),
    (1, 4, 3),
    (7, 5, 3),
    (6, 6, 6),
    (8, 8, 8),
    (10, 9, 17),
    (17, 16, 10),
];

fn assert_driver_matches_oracle(
    driver: fn(&TensorData, &TensorData) -> TensorData,
    a: &TensorData,
    b: &TensorData,
) {
    let got = driver(a, b);
    let oracle = SimpleOps::matmul(a, b);
    assert_eq!(
        got.shape, oracle.shape,
        "shape mismatch: a={:?} b={:?}",
        a.shape, b.shape
    );
    assert_eq!(
        &*got.storage, &*oracle.storage,
        "storage mismatch: a={:?} b={:?}",
        a.shape, b.shape
    );
}

#[test]
#[ignore = "implement matmul_iterative (src/gemm.rs) first, then drop this line"]
fn test_iterative_matches_oracle() {
    for (m, k, n) in SHAPES {
        let a = seq(vec![m, k], 0);
        let b = seq(vec![k, n], 1);
        assert_driver_matches_oracle(matmul_iterative, &a, &b);
    }
}

#[test]
#[ignore = "implement matmul_recursive (src/gemm.rs) first, then drop this line"]
fn test_recursive_matches_oracle() {
    for (m, k, n) in SHAPES {
        let a = seq(vec![m, k], 0);
        let b = seq(vec![k, n], 1);
        assert_driver_matches_oracle(matmul_recursive, &a, &b);
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    #[ignore = "implement matmul_iterative first, then drop this line"]
    fn prop_iterative_matches_oracle(m in 1usize..=40, k in 1usize..=40, n in 1usize..=40) {
        let a = seq(vec![m, k], 0);
        let b = seq(vec![k, n], 1);
        let got = matmul_iterative(&a, &b);
        let oracle = SimpleOps::matmul(&a, &b);
        prop_assert_eq!(&got.shape, &oracle.shape);
        prop_assert_eq!(&*got.storage, &*oracle.storage);
    }

    #[test]
    #[ignore = "implement matmul_recursive first, then drop this line"]
    fn prop_recursive_matches_oracle(m in 1usize..=40, k in 1usize..=40, n in 1usize..=40) {
        let a = seq(vec![m, k], 0);
        let b = seq(vec![k, n], 1);
        let got = matmul_recursive(&a, &b);
        let oracle = SimpleOps::matmul(&a, &b);
        prop_assert_eq!(&got.shape, &oracle.shape);
        prop_assert_eq!(&*got.storage, &*oracle.storage);
    }
}
