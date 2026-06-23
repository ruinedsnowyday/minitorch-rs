//! Cross-backend equivalence: `FastOps` must agree with `SimpleOps`
//! element-for-element, on every layout kind. `SimpleOps` is the trusted
//! oracle (already covered by `test_tensor_ops.rs`); these tests catch the
//! bugs that only surface on non-contiguous inputs — a wrong stride, an
//! off-by-one in `offset_at`, taking the fast path when the tensor isn't
//! actually packed.

use std::rc::Rc;

use minitorch_rs::fast_ops::FastOps;
use minitorch_rs::tensor_data::TensorData;
use minitorch_rs::tensor_ops::{SimpleOps, TensorOps};
use proptest::prelude::*;

// `map` output is always canonical (row-major, offset 0, tight buffer), so
// comparing the full storage Vec + shape is an exact equivalence check. The
// op uses small-integer inputs and simple arithmetic, so f64 equality is
// bit-exact here — no epsilon needed.
fn assert_map_matches_oracle(t: &TensorData) {
    let f = |x: f64| x * 2.0 + 1.0;
    let fast = FastOps::map(t, f);
    let simple = SimpleOps::map(t, f);
    assert_eq!(
        fast.shape, simple.shape,
        "shape mismatch for input shape={:?} strides={:?} offset={}",
        t.shape, t.strides, t.storage_offset
    );
    assert_eq!(
        &*fast.storage, &*simple.storage,
        "storage mismatch for input shape={:?} strides={:?} offset={}",
        t.shape, t.strides, t.storage_offset
    );
}

#[test]
fn test_map_contiguous_1d_2d_3d() {
    assert_map_matches_oracle(&TensorData::new(
        (0..5).map(|i| i as f64).collect(),
        vec![5],
    ));
    assert_map_matches_oracle(&TensorData::new(
        (0..6).map(|i| i as f64).collect(),
        vec![2, 3],
    ));
    assert_map_matches_oracle(&TensorData::new(
        (0..24).map(|i| i as f64).collect(),
        vec![2, 3, 4],
    ));
}

// Transposed view: non-contiguous strides force the slow `offset_at` path.
// This is the case the fast-only path would silently get wrong.
#[test]
fn test_map_transposed() {
    let t = TensorData::new((0..6).map(|i| i as f64).collect(), vec![2, 3]);
    assert_map_matches_oracle(&t.permute(&[1, 0]));
}

#[test]
fn test_map_3d_permutations() {
    let t = TensorData::new((0..24).map(|i| i as f64).collect(), vec![2, 3, 4]);
    assert_map_matches_oracle(&t.permute(&[2, 0, 1]));
    assert_map_matches_oracle(&t.permute(&[2, 1, 0]));
}

// Broadcast view: stride-0 dims. Not packed, so slow path; offsets repeat.
#[test]
fn test_map_broadcast() {
    let row = TensorData::new(vec![1.0, 2.0, 3.0], vec![3]);
    assert_map_matches_oracle(&row.broadcast_to(&[4, 3]));
}

// Packed but rooted at a non-zero offset: fast path must slice from
// storage_offset, NOT from 0. A bug here reads the wrong elements.
#[test]
fn test_map_packed_offset_view() {
    let t = TensorData {
        storage: Rc::new((0..12).map(|i| i as f64).collect()),
        storage_offset: 3,
        shape: vec![2, 3],
        strides: vec![3, 1],
    };
    assert_map_matches_oracle(&t);
}

// Strided AND offset (non-packed view starting partway into the buffer):
// exercises offset_at + storage_offset together on the slow path.
#[test]
fn test_map_strided_offset_view() {
    let base = TensorData {
        storage: Rc::new((0..12).map(|i| i as f64).collect()),
        storage_offset: 2,
        shape: vec![2, 3],
        strides: vec![3, 1],
    };
    assert_map_matches_oracle(&base.permute(&[1, 0]));
}

#[test]
fn test_map_scalar() {
    let t = TensorData::new(vec![7.0], vec![]);
    assert_map_matches_oracle(&t);
}

proptest! {
    // Any small shape, contiguous and axis-reversed, must match SimpleOps.
    #[test]
    fn prop_map_matches_simpleops(
        shape in proptest::collection::vec(1usize..=6, 1..=4)
    ) {
        let n: usize = shape.iter().product();
        let storage: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let t = TensorData::new(storage, shape.clone());

        let f = |x: f64| x * 2.0 + 1.0;
        let fast = FastOps::map(&t, f);
        let simple = SimpleOps::map(&t, f);
        prop_assert_eq!(&*fast.storage, &*simple.storage);

        let axes: Vec<usize> = (0..shape.len()).rev().collect();
        let rev = t.permute(&axes);
        let fast_rev = FastOps::map(&rev, f);
        let simple_rev = SimpleOps::map(&rev, f);
        prop_assert_eq!(&*fast_rev.storage, &*simple_rev.storage);
    }
}

// ===================================================================
// zip — cross-backend equivalence
// ===================================================================

// Asymmetric f (x*2 - y) so an operand swap or a wrong a/b pairing can't pass
// by accident — a symmetric op like + would hide it.
fn assert_zip_matches_oracle(a: &TensorData, b: &TensorData) {
    let f = |x: f64, y: f64| x * 2.0 - y;
    let fast = FastOps::zip(a, b, f);
    let simple = SimpleOps::zip(a, b, f);
    assert_eq!(
        fast.shape, simple.shape,
        "shape mismatch: a={:?} b={:?}",
        a.shape, b.shape
    );
    assert_eq!(
        &*fast.storage, &*simple.storage,
        "storage mismatch: a(shape={:?} strides={:?} off={}) b(shape={:?} strides={:?} off={})",
        a.shape, a.strides, a.storage_offset, b.shape, b.strides, b.storage_offset
    );
}

// Hand-computed anchor (independent of the oracle): matrix + row bias.
// row [10,20,30] adds across each row of the 2x3 matrix.
#[test]
fn test_zip_broadcast_bias_hand_computed() {
    let a = TensorData::new(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]);
    let b = TensorData::new(vec![10., 20., 30.], vec![3]);
    let out = FastOps::zip(&a, &b, |x, y| x + y);
    assert_eq!(out.shape, vec![2, 3]);
    assert_eq!(&*out.storage, &vec![11., 22., 33., 14., 25., 36.]);
}

// Both packed, same shape → fast slice-zip path.
#[test]
fn test_zip_both_contiguous_same_shape() {
    let a = TensorData::new((0..6).map(|i| i as f64).collect(), vec![2, 3]);
    let b = TensorData::new((10..16).map(|i| i as f64).collect(), vec![2, 3]);
    assert_zip_matches_oracle(&a, &b);
}

#[test]
fn test_zip_both_contiguous_3d() {
    let a = TensorData::new((0..24).map(|i| i as f64).collect(), vec![2, 3, 4]);
    let b = TensorData::new((100..124).map(|i| i as f64).collect(), vec![2, 3, 4]);
    assert_zip_matches_oracle(&a, &b);
}

// Matrix + trailing-dim row: a packed, b broadcast (slow path). Plus the
// swap, to prove the asymmetric f survives reordering correctly.
#[test]
fn test_zip_matrix_plus_row_and_swap() {
    let a = TensorData::new((0..12).map(|i| i as f64).collect(), vec![4, 3]);
    let row = TensorData::new(vec![1., 2., 3.], vec![3]);
    assert_zip_matches_oracle(&a, &row);
    assert_zip_matches_oracle(&row, &a);
}

// Column [4,1] + row [1,3] → both broadcast to [4,3]; neither view packed.
#[test]
fn test_zip_outer_broadcast_both_sides() {
    let col = TensorData::new(vec![1., 2., 3., 4.], vec![4, 1]);
    let row = TensorData::new(vec![10., 20., 30.], vec![1, 3]);
    assert_zip_matches_oracle(&col, &row);
}

// Scalar (0-d) operand broadcasts to the other's shape with all-zero strides.
#[test]
fn test_zip_scalar_operand_and_swap() {
    let a = TensorData::new((0..6).map(|i| i as f64).collect(), vec![2, 3]);
    let s = TensorData::new(vec![100.], vec![]);
    assert_zip_matches_oracle(&a, &s);
    assert_zip_matches_oracle(&s, &a);
}

// Transposed operand: same shape as the other but strided → slow path with a
// genuine (non-broadcast) stride permutation.
#[test]
fn test_zip_transposed_operand_and_swap() {
    let a = TensorData::new((0..6).map(|i| i as f64).collect(), vec![2, 3]);
    let b = TensorData::new((10..16).map(|i| i as f64).collect(), vec![3, 2])
        .permute(&[1, 0]); // shape [2,3], strides [1,3], not packed
    assert_zip_matches_oracle(&a, &b);
    assert_zip_matches_oracle(&b, &a);
}

#[test]
fn test_zip_both_transposed() {
    let a = TensorData::new((0..6).map(|i| i as f64).collect(), vec![3, 2])
        .permute(&[1, 0]);
    let b = TensorData::new((10..16).map(|i| i as f64).collect(), vec![3, 2])
        .permute(&[1, 0]);
    assert_zip_matches_oracle(&a, &b);
}

// Both packed but rooted at non-zero offsets → fast path must slice each from
// its own storage_offset.
#[test]
fn test_zip_packed_offset_views() {
    let a = TensorData {
        storage: Rc::new((0..10).map(|i| i as f64).collect()),
        storage_offset: 2,
        shape: vec![2, 3],
        strides: vec![3, 1],
    };
    let b = TensorData {
        storage: Rc::new((100..110).map(|i| i as f64).collect()),
        storage_offset: 1,
        shape: vec![2, 3],
        strides: vec![3, 1],
    };
    assert_zip_matches_oracle(&a, &b);
}

proptest! {
    // Same-shape contiguous operands of any small shape.
    #[test]
    fn prop_zip_same_shape(shape in proptest::collection::vec(1usize..=5, 1..=3)) {
        let n: usize = shape.iter().product();
        let a = TensorData::new((0..n).map(|i| i as f64).collect(), shape.clone());
        let b = TensorData::new((0..n).map(|i| (i as f64) * 3.0).collect(), shape.clone());
        let f = |x: f64, y: f64| x * 2.0 - y;
        let fast = FastOps::zip(&a, &b, f);
        let simple = SimpleOps::zip(&a, &b, f);
        prop_assert_eq!(&*fast.storage, &*simple.storage);
    }

    // Matrix + trailing-dim row broadcast, both orderings.
    #[test]
    fn prop_zip_row_broadcast(rows in 1usize..=5, cols in 1usize..=5) {
        let mat = TensorData::new(
            (0..rows * cols).map(|i| i as f64).collect(),
            vec![rows, cols],
        );
        let row = TensorData::new(
            (0..cols).map(|i| (i as f64) + 0.5).collect(),
            vec![cols],
        );
        let f = |x: f64, y: f64| x - y;

        let fast = FastOps::zip(&mat, &row, f);
        let simple = SimpleOps::zip(&mat, &row, f);
        prop_assert_eq!(&*fast.storage, &*simple.storage);

        let fast_sw = FastOps::zip(&row, &mat, f);
        let simple_sw = SimpleOps::zip(&row, &mat, f);
        prop_assert_eq!(&*fast_sw.storage, &*simple_sw.storage);
    }
}
