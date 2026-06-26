#![feature(portable_simd)]

//! Cross-backend equivalence: `FastOps` must agree with `SimpleOps`
//! element-for-element, on every layout kind. `SimpleOps` is the trusted
//! oracle (already covered by `test_tensor_ops.rs`); these tests catch the
//! bugs that only surface on non-contiguous inputs — a wrong stride, an
//! off-by-one in `offset_at`, taking the fast path when the tensor isn't
//! actually packed.

use std::rc::Rc;
use std::simd::Simd;

use minitorch_rs::fast_ops::FastOps;
use minitorch_rs::operators;
use minitorch_rs::simd_ops::{
    BinaryOp, LANES, SimdAdd, SimdEq, SimdExp, SimdId, SimdInv, SimdInvBack,
    SimdLeakyReLU, SimdLeakyReLUBack, SimdLog, SimdLogBack, SimdLt, SimdMax, SimdMul,
    SimdNeg, SimdReLU, SimdReLUBack, SimdSigmoid, SimdSigmoidBack, UnaryOp,
};
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

proptest! {
    // One operand non-packed (transposed) zipped against a contiguous one of
    // the same shape, both orderings. The existing zip proptests only feed
    // both-contiguous operands or a clean row-broadcast; this is the case the
    // upcoming "index the packed side linearly, offset_at only the strided
    // side" zip optimization must keep correct — a regression there shows here.
    #[test]
    fn prop_zip_mixed_layout(rows in 1usize..=5, cols in 1usize..=5) {
        // contiguous [rows, cols]
        let a = TensorData::new(
            (0..rows * cols).map(|i| i as f64).collect(),
            vec![rows, cols],
        );
        // also [rows, cols], but strided: built [cols, rows] then transposed,
        // so its strides are [1, rows] — not packed (except degenerate 1-dims).
        let b = TensorData::new(
            (0..rows * cols).map(|i| (i as f64) * 1.5 + 1.0).collect(),
            vec![cols, rows],
        )
        .permute(&[1, 0]);
        let f = |x: f64, y: f64| x * 2.0 - y;

        let fast = FastOps::zip(&a, &b, f);
        let simple = SimpleOps::zip(&a, &b, f);
        prop_assert_eq!(&fast.shape, &simple.shape);
        prop_assert_eq!(&*fast.storage, &*simple.storage);

        let fast_sw = FastOps::zip(&b, &a, f);
        let simple_sw = SimpleOps::zip(&b, &a, f);
        prop_assert_eq!(&*fast_sw.storage, &*simple_sw.storage);
    }
}

// ===================================================================
// reduce — cross-backend equivalence
// ===================================================================

// Order-sensitive fold (acc*2 + x): the result depends on the order the fiber
// is visited, so a reversed inner walk would diverge from SimpleOps and fail.
// A plain sum would mask that.
fn assert_reduce_matches_oracle(t: &TensorData, dim: usize, keep_dims: bool) {
    let f = |acc: f64, x: f64| acc * 2.0 + x;
    let fast = FastOps::reduce(t, f, 0.0, dim, keep_dims);
    let simple = SimpleOps::reduce(t, f, 0.0, dim, keep_dims);
    assert_eq!(
        fast.shape, simple.shape,
        "shape mismatch: shape={:?} dim={} keep={}",
        t.shape, dim, keep_dims
    );
    assert_eq!(
        &*fast.storage, &*simple.storage,
        "storage mismatch: shape={:?} strides={:?} off={} dim={} keep={}",
        t.shape, t.strides, t.storage_offset, dim, keep_dims
    );
}

// Hand-computed anchor (independent of the oracle): sums along each axis.
#[test]
fn test_reduce_sum_hand_computed() {
    let t = TensorData::new(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]);
    // dim 0 (down columns): [1+4, 2+5, 3+6]
    let d0 = FastOps::reduce(&t, |a, x| a + x, 0.0, 0, false);
    assert_eq!(d0.shape, vec![3]);
    assert_eq!(&*d0.storage, &vec![5., 7., 9.]);
    // dim 1 (across rows): [1+2+3, 4+5+6]
    let d1 = FastOps::reduce(&t, |a, x| a + x, 0.0, 1, false);
    assert_eq!(d1.shape, vec![2]);
    assert_eq!(&*d1.storage, &vec![6., 15.]);
    // keep_dims keeps the reduced axis as length 1
    let d1k = FastOps::reduce(&t, |a, x| a + x, 0.0, 1, true);
    assert_eq!(d1k.shape, vec![2, 1]);
    assert_eq!(&*d1k.storage, &vec![6., 15.]);
}

#[test]
fn test_reduce_2d_all_dims_keepdims() {
    let t = TensorData::new((0..6).map(|i| i as f64).collect(), vec![2, 3]);
    for dim in 0..2 {
        for keep in [false, true] {
            assert_reduce_matches_oracle(&t, dim, keep);
        }
    }
}

#[test]
fn test_reduce_3d_all_dims_keepdims() {
    let t = TensorData::new((0..24).map(|i| i as f64).collect(), vec![2, 3, 4]);
    for dim in 0..3 {
        for keep in [false, true] {
            assert_reduce_matches_oracle(&t, dim, keep);
        }
    }
}

// Strided input: a transposed view, reduced along each axis.
#[test]
fn test_reduce_transposed_input() {
    let t = TensorData::new((0..6).map(|i| i as f64).collect(), vec![3, 2])
        .permute(&[1, 0]); // shape [2,3], strides [1,3]
    for dim in 0..2 {
        for keep in [false, true] {
            assert_reduce_matches_oracle(&t, dim, keep);
        }
    }
}

// Offset view: storage_offset must flow into the fiber base.
#[test]
fn test_reduce_offset_view() {
    let t = TensorData {
        storage: Rc::new((0..10).map(|i| i as f64).collect()),
        storage_offset: 2,
        shape: vec![2, 3],
        strides: vec![3, 1],
    };
    for dim in 0..2 {
        assert_reduce_matches_oracle(&t, dim, false);
    }
}

#[test]
fn test_reduce_1d_to_scalar() {
    let t = TensorData::new(vec![1., 2., 3., 4.], vec![4]);
    assert_reduce_matches_oracle(&t, 0, false);
    assert_reduce_matches_oracle(&t, 0, true);
}

proptest! {
    // Any small shape, every axis, both keep_dims settings.
    #[test]
    fn prop_reduce_matches_simpleops(
        shape in proptest::collection::vec(1usize..=5, 1..=3),
        keep in any::<bool>(),
    ) {
        let n: usize = shape.iter().product();
        let t = TensorData::new((0..n).map(|i| i as f64).collect(), shape.clone());
        let f = |acc: f64, x: f64| acc * 2.0 + x;
        for dim in 0..shape.len() {
            let fast = FastOps::reduce(&t, f, 0.0, dim, keep);
            let simple = SimpleOps::reduce(&t, f, 0.0, dim, keep);
            prop_assert_eq!(&fast.shape, &simple.shape);
            prop_assert_eq!(&*fast.storage, &*simple.storage);
        }
    }
}

// --- reducing over a broadcast (stride-0) view ---
//
// The one reduce-input layout the tests above never build, and exactly what the
// autodiff backward path constructs (gradients reduced over broadcast-expanded
// dims). With a stride-0 axis, `data[base + k*stride]` reads the SAME element
// dim_size times; a silent bug here surfaces as wrong gradients, not a crash.

// Reduce ALONG the broadcast axis: row [1,2,3] expanded to [4,3], summed down
// the expanded dim → every element counted 4×.
#[test]
fn test_reduce_along_broadcast_axis_hand_computed() {
    let view = TensorData::new(vec![1., 2., 3.], vec![3]).broadcast_to(&[4, 3]);
    let out = FastOps::reduce(&view, |a, x| a + x, 0.0, 0, false);
    assert_eq!(out.shape, vec![3]);
    assert_eq!(&*out.storage, &vec![4., 8., 12.]);
}

// Reduce a REAL axis of a broadcast view: the stride-0 lands in a *kept* dim
// (skipped_strides), so every output fiber starts at the same base. row [1,2,3]
// → [4,3], summed across each row → 1+2+3 = 6, four identical rows.
#[test]
fn test_reduce_real_axis_of_broadcast_view_hand_computed() {
    let view = TensorData::new(vec![1., 2., 3.], vec![3]).broadcast_to(&[4, 3]);
    let out = FastOps::reduce(&view, |a, x| a + x, 0.0, 1, false);
    assert_eq!(out.shape, vec![4]);
    assert_eq!(&*out.storage, &vec![6., 6., 6., 6.]);
}

// Same two layouts, oracle-checked with the order-sensitive fold and both
// keep_dims settings, plus a column broadcast for the other stride-0 position.
#[test]
fn test_reduce_broadcast_view_matches_oracle() {
    let row = TensorData::new(vec![2., 5., 7.], vec![3]).broadcast_to(&[4, 3]);
    let col = TensorData::new(vec![3., 6.], vec![2, 1]).broadcast_to(&[2, 5]);
    for t in [&row, &col] {
        for dim in 0..2 {
            for keep in [false, true] {
                assert_reduce_matches_oracle(t, dim, keep);
            }
        }
    }
}

proptest! {
    // Broadcast a size-1 axis up, then reduce along every axis — sweeps both
    // "reduce the broadcast axis" and "reduce a real axis of a broadcast view"
    // (the stride-0 reduce path) across many shapes.
    #[test]
    fn prop_reduce_broadcast_view(
        base in 1usize..=5,
        expand in 2usize..=5,
        keep in any::<bool>(),
    ) {
        // [1, base] broadcast to [expand, base]: dim 0 is stride-0.
        let view = TensorData::new(
            (0..base).map(|i| (i as f64) + 0.5).collect(),
            vec![1, base],
        )
        .broadcast_to(&[expand, base]);
        let f = |acc: f64, x: f64| acc * 2.0 + x;
        for dim in 0..2 {
            let fast = FastOps::reduce(&view, f, 0.0, dim, keep);
            let simple = SimpleOps::reduce(&view, f, 0.0, dim, keep);
            prop_assert_eq!(&fast.shape, &simple.shape);
            prop_assert_eq!(&*fast.storage, &*simple.storage);
        }
    }
}

// ===================================================================
// matmul — cross-backend equivalence
// ===================================================================

// A `shape`-sized tensor filled with 0,1,2,... (+start). Small integers keep
// products/sums exact in f64, so storage comparison is bit-exact.
fn seq(shape: Vec<usize>, start: usize) -> TensorData {
    let n: usize = shape.iter().product();
    TensorData::new((0..n).map(|i| (start + i) as f64).collect(), shape)
}

fn assert_matmul_matches_oracle(a: &TensorData, b: &TensorData) {
    let fast = FastOps::matmul(a, b);
    let simple = SimpleOps::matmul(a, b);
    assert_eq!(
        fast.shape, simple.shape,
        "shape mismatch: a={:?} b={:?}",
        a.shape, b.shape
    );
    assert_eq!(
        &*fast.storage, &*simple.storage,
        "storage mismatch: a(shape={:?} strides={:?}) b(shape={:?} strides={:?})",
        a.shape, a.strides, b.shape, b.strides
    );
}

// Hand-computed anchor: [[1,2,3],[4,5,6]] · [[1,2],[3,4],[5,6]].
#[test]
fn test_matmul_2d_hand_computed() {
    let a = TensorData::new(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]);
    let b = TensorData::new(vec![1., 2., 3., 4., 5., 6.], vec![3, 2]);
    let c = FastOps::matmul(&a, &b);
    assert_eq!(c.shape, vec![2, 2]);
    // row0 = [1·1+2·3+3·5, 1·2+2·4+3·6] = [22, 28]
    // row1 = [4·1+5·3+6·5, 4·2+5·4+6·6] = [49, 64]
    assert_eq!(&*c.storage, &vec![22., 28., 49., 64.]);
}

#[test]
fn test_matmul_2d_nonsquare() {
    assert_matmul_matches_oracle(&seq(vec![2, 3], 0), &seq(vec![3, 4], 1));
    assert_matmul_matches_oracle(&seq(vec![5, 2], 0), &seq(vec![2, 7], 3));
}

#[test]
fn test_matmul_batched() {
    assert_matmul_matches_oracle(&seq(vec![4, 2, 3], 0), &seq(vec![4, 3, 2], 1));
}

// Batch-dim broadcasting: a 2-D operand reused across the other's batch, and
// an explicit size-1 batch dim.
#[test]
fn test_matmul_broadcast_batch() {
    assert_matmul_matches_oracle(&seq(vec![2, 3], 0), &seq(vec![4, 3, 2], 1));
    assert_matmul_matches_oracle(&seq(vec![4, 2, 3], 0), &seq(vec![3, 2], 1));
    assert_matmul_matches_oracle(&seq(vec![1, 2, 3], 0), &seq(vec![4, 3, 2], 1));
}

// Transposed (non-packed) operands → matmul_2d's contiguous() fallback. The
// b-transposed case is exactly what the old `a.is_packed()` typo got wrong.
#[test]
fn test_matmul_transposed_operands() {
    let a = seq(vec![2, 3], 0);
    let b_t = seq(vec![2, 3], 1).permute(&[1, 0]); // [3,2], strided
    assert_matmul_matches_oracle(&a, &b_t);

    let a_t = seq(vec![3, 2], 0).permute(&[1, 0]); // [2,3], strided
    assert_matmul_matches_oracle(&a_t, &seq(vec![3, 4], 1));

    let a_t2 = seq(vec![3, 2], 0).permute(&[1, 0]); // [2,3]
    let b_t2 = seq(vec![2, 3], 5).permute(&[1, 0]); // [3,2]
    assert_matmul_matches_oracle(&a_t2, &b_t2);
}

proptest! {
    #[test]
    fn prop_matmul_matches_simpleops(m in 1usize..=5, k in 1usize..=5, n in 1usize..=5) {
        let a = seq(vec![m, k], 0);
        let b = seq(vec![k, n], 1);
        let fast = FastOps::matmul(&a, &b);
        let simple = SimpleOps::matmul(&a, &b);
        prop_assert_eq!(&fast.shape, &simple.shape);
        prop_assert_eq!(&*fast.storage, &*simple.storage);
    }
}

// ===================================================================
// batched_matmul — gaps the matmul tests above don't reach
//
// FastOps::matmul now routes to batched_matmul, so the tests above already
// cover it — but every batched case there has m == n per matrix, and none
// feeds the batched path a strided operand. These target precisely that.
// ===================================================================

// All of m, k, n distinct, batched. The flattened decode is `batch = g / m,
// i = g % m`: with m == n (every batched test above) a `g / n` typo passes
// silently; m != n is what exposes it. This is the headline gap.
#[test]
fn test_matmul_batched_all_distinct() {
    assert_matmul_matches_oracle(&seq(vec![4, 2, 3], 0), &seq(vec![4, 3, 5], 1));
    assert_matmul_matches_oracle(&seq(vec![3, 6, 2], 0), &seq(vec![3, 2, 4], 5));
}

// More than one batch dim: n_batches is a product of several axes, so this
// exercises broadcast_strides / out_batch_shape over multi-axis batches while
// staying on the packed fast path. m, k, n distinct.
#[test]
fn test_matmul_multi_batch_dims() {
    assert_matmul_matches_oracle(
        &seq(vec![2, 3, 2, 4], 0),
        &seq(vec![2, 3, 4, 5], 1),
    );
}

// Both operands broadcast, on *different* batch axes (a: [3,1,..], b: [1,4,..]
// → out batch [3,4]). Both a_view and b_view get stride-0 dims, so both fall
// to batched_matmul's contiguous() materialization. m, k, n distinct.
#[test]
fn test_matmul_dual_side_batch_broadcast() {
    assert_matmul_matches_oracle(
        &seq(vec![3, 1, 2, 3], 0),
        &seq(vec![1, 4, 3, 5], 1),
    );
}

// Strided batched operand: the permute makes the view non-packed, forcing
// batched_matmul's own contiguous() path (distinct from matmul_2d's, which the
// 2-D transposed tests hit). Done on each side. m, k, n distinct.
#[test]
fn test_matmul_batched_strided_operand() {
    let a = seq(vec![4, 3, 2], 0).permute(&[0, 2, 1]); // [4,2,3], not packed
    assert_matmul_matches_oracle(&a, &seq(vec![4, 3, 5], 1));

    let b = seq(vec![4, 5, 3], 0).permute(&[0, 2, 1]); // [4,3,5], not packed
    assert_matmul_matches_oracle(&seq(vec![4, 2, 3], 7), &b);
}

// Hand-computed batched anchor, independent of the oracle: two stacked 2x2
// matmuls (the second b is 2·I, so its result is the input doubled). Guards
// batch-stacking order without trusting SimpleOps to stack the same way.
#[test]
fn test_matmul_batched_hand_computed() {
    // batch 0: [[1,2],[3,4]] · [[1,0],[0,1]] = [[1,2],[3,4]]
    // batch 1: [[5,6],[7,8]] · [[2,0],[0,2]] = [[10,12],[14,16]]
    let a = TensorData::new(vec![1., 2., 3., 4., 5., 6., 7., 8.], vec![2, 2, 2]);
    let b = TensorData::new(vec![1., 0., 0., 1., 2., 0., 0., 2.], vec![2, 2, 2]);
    let c = FastOps::matmul(&a, &b);
    assert_eq!(c.shape, vec![2, 2, 2]);
    assert_eq!(&*c.storage, &vec![1., 2., 3., 4., 10., 12., 14., 16.]);
}

// ===================================================================
// map_op — SIMD-vectorized unary map, cross-backend equivalence
//
// FastOps::map_op::<SimdReLU, N> must equal the scalar oracle
// SimpleOps::map(_, operators::relu) element-for-element. The failure modes
// here are specific to the chunk/tail SIMD structure:
//   * the scalar TAIL (the trailing len % N elements) being skipped or left
//     un-transformed — provoked by lengths that aren't multiples of N whose
//     tail elements are negative, so relu MUST change them (a raw passthrough
//     can't pass);
//   * block REORDERING from a non-order-preserving parallel collect — provoked
//     by distinct per-index values, so any permutation shows as a mismatch;
//   * taking the contiguous SIMD path on a non-packed view — provoked by the
//     transposed/broadcast cases, which must fall back to the scalar branch.
// relu here is exact in f64 (max with 0, no rounding), so storage comparison
// is bit-exact — no epsilon.
// ===================================================================

// Alternating signs with distinct magnitudes (1, -2, 3, -4, …). Both signs
// appear densely, so whatever the tail size, the tail holds positive values a
// zeroed/forgotten tail would wreck AND negatives a raw-passthrough tail would
// wreck. Distinct magnitudes also make any block reordering visible.
fn signed_seq(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let m = (i + 1) as f64;
            if i % 2 == 0 { m } else { -m }
        })
        .collect()
}

fn assert_relu_matches_oracle<const N: usize>(t: &TensorData) {
    let fast = FastOps::map_op::<SimdReLU, N>(t);
    let oracle = SimpleOps::map(t, operators::relu);
    assert_eq!(
        fast.shape, oracle.shape,
        "shape mismatch: N={} in(shape={:?} strides={:?} off={})",
        N, t.shape, t.strides, t.storage_offset
    );
    assert_eq!(
        &*fast.storage, &*oracle.storage,
        "storage mismatch: N={} in(shape={:?} strides={:?} off={})",
        N, t.shape, t.strides, t.storage_offset
    );
}

// Hand-computed anchor (independent of the oracle). Length 5 at N=4 → one SIMD
// chunk [-2,3,-1,4] then a 1-element tail [5]. The tail value is POSITIVE, so
// it pins the tail down two ways at once: a tail left un-relu'd keeps a wrong
// value, and a tail left as a 0-fill reads 0 instead of 5. The chunk carries
// negatives, so a passthrough there fails too.
#[test]
fn test_map_op_relu_tail_hand_computed() {
    let t = TensorData::new(vec![-2., 3., -1., 4., 5.], vec![5]);
    let out = FastOps::map_op::<SimdReLU, 4>(&t);
    assert_eq!(out.shape, vec![5]);
    assert_eq!(&*out.storage, &vec![0., 3., 0., 4., 5.]);
}

// Every tail size for N=4 and N=8 (lengths 1..=20 cover leftovers 0,1,2,3,…),
// each length's tail negative.
#[test]
fn test_map_op_relu_contiguous_various_lengths() {
    for n in 1..=20 {
        let t = TensorData::new(signed_seq(n), vec![n]);
        assert_relu_matches_oracle::<4>(&t);
        assert_relu_matches_oracle::<8>(&t);
    }
}

// Transposed view: non-contiguous strides → scalar fallback, not the SIMD
// slice path (which assumes contiguity).
#[test]
fn test_map_op_relu_transposed() {
    let t = TensorData::new(signed_seq(6), vec![2, 3]);
    assert_relu_matches_oracle::<4>(&t.permute(&[1, 0]));
}

// Broadcast view: stride-0 dim, not packed → scalar fallback.
#[test]
fn test_map_op_relu_broadcast() {
    let row = TensorData::new(vec![2., -1., -3.], vec![3]);
    assert_relu_matches_oracle::<4>(&row.broadcast_to(&[4, 3]));
}

// Packed but rooted at a non-zero offset: the SIMD path must slice from
// storage_offset, not 0 (the two leading 9s are decoys). Length 7 leaves a
// 3-element negative tail under N=4.
#[test]
fn test_map_op_relu_packed_offset_view() {
    let t = TensorData {
        storage: Rc::new(vec![9., 9., 3., 2., 1., 0., -1., -2., -3.]),
        storage_offset: 2,
        shape: vec![7],
        strides: vec![1],
    };
    assert_relu_matches_oracle::<4>(&t);
}

// 0-d scalar: size 1, so the SIMD slice is empty and the lone element is all
// tail — the degenerate end of the chunk/tail split.
#[test]
fn test_map_op_relu_scalar() {
    assert_relu_matches_oracle::<4>(&TensorData::new(vec![-5.], vec![]));
    assert_relu_matches_oracle::<4>(&TensorData::new(vec![5.], vec![]));
}

proptest! {
    // Sweep length over every tail size for three lane widths. Decreasing
    // signed values: the tail is negative (tail-bug bait) and every index is
    // distinct (reorder-bug bait).
    #[test]
    fn prop_map_op_relu_contiguous(n in 1usize..=128) {
        let t = TensorData::new(signed_seq(n), vec![n]);
        assert_relu_matches_oracle::<2>(&t);
        assert_relu_matches_oracle::<4>(&t);
        assert_relu_matches_oracle::<8>(&t);
    }

    // Transposed views of any small 2-D shape → the scalar fallback across many
    // stride patterns.
    #[test]
    fn prop_map_op_relu_transposed(rows in 1usize..=6, cols in 1usize..=6) {
        let t = TensorData::new(signed_seq(rows * cols), vec![rows, cols]);
        assert_relu_matches_oracle::<4>(&t.permute(&[1, 0]));
    }
}

// ===================================================================
// zip_op — SIMD-vectorized binary zip, cross-backend equivalence
//
// FastOps::zip_op::<Op, N> must equal the scalar oracle
// SimpleOps::zip(_, _, f). The SIMD path only fires when BOTH operands are
// packed (same shape, no broadcasting); any broadcast or strided operand drops
// to the scalar fallback (FastOps::zip with Op::scalar). So the cases below
// split into "both packed → SIMD (true,true) arm" and "broadcast/strided →
// fallback", both orderings.
//
// The op is asymmetric (2a - b): a swapped a/b pairing in the SIMD arm — which
// a symmetric add/mul would hide — surfaces as a mismatch. Inputs are small
// integers/halves, so 2a - b is exact in f64 and storage comparison is exact.
// ===================================================================

fn axmy(a: f64, b: f64) -> f64 {
    2.0 * a - b
}

// Asymmetric test fixture op, mirroring `axmy` in scalar and SIMD form.
struct AxmyOp;
impl BinaryOp for AxmyOp {
    fn scalar(a: f64, b: f64) -> f64 {
        axmy(a, b)
    }
    fn simd<const N: usize>(a: Simd<f64, N>, b: Simd<f64, N>) -> Simd<f64, N> {
        Simd::splat(2.0) * a - b
    }
}

fn assert_zip_op_matches_oracle<const N: usize>(a: &TensorData, b: &TensorData) {
    let fast = FastOps::zip_op::<AxmyOp, N>(a, b);
    let oracle = SimpleOps::zip(a, b, axmy);
    assert_eq!(
        fast.shape, oracle.shape,
        "shape mismatch: N={} a={:?} b={:?}",
        N, a.shape, b.shape
    );
    assert_eq!(
        &*fast.storage, &*oracle.storage,
        "storage mismatch: N={} a(shape={:?} strides={:?} off={}) b(shape={:?} strides={:?} off={})",
        N, a.shape, a.strides, a.storage_offset, b.shape, b.strides, b.storage_offset
    );
}

// Hand-computed anchor (independent of the oracle). Both packed [5] at N=4 → one
// SIMD chunk + a 1-element tail. 2a - b = [2-10, 4-20, 6-30, 8-40, 10-50].
#[test]
fn test_zip_op_hand_computed() {
    let a = TensorData::new(vec![1., 2., 3., 4., 5.], vec![5]);
    let b = TensorData::new(vec![10., 20., 30., 40., 50.], vec![5]);
    let out = FastOps::zip_op::<AxmyOp, 4>(&a, &b);
    assert_eq!(out.shape, vec![5]);
    assert_eq!(&*out.storage, &vec![-8., -16., -24., -32., -40.]);
}

// A real framework op (SimdAdd) through zip_op, not just the synthetic fixture.
#[test]
fn test_zip_op_real_add() {
    let a = TensorData::new(vec![1., 2., 3., 4., 5.], vec![5]);
    let b = TensorData::new(vec![10., 20., 30., 40., 50.], vec![5]);
    let out = FastOps::zip_op::<SimdAdd, 4>(&a, &b);
    assert_eq!(&*out.storage, &vec![11., 22., 33., 44., 55.]);
}

// Both packed, same shape → the SIMD (true,true) arm. Sweep lengths so every
// tail size for N=4 and N=8 is hit.
#[test]
fn test_zip_op_contiguous_various_lengths() {
    for n in 1..=20 {
        let a = TensorData::new(signed_seq(n), vec![n]);
        let b =
            TensorData::new((0..n).map(|i| i as f64 * 0.5 - 3.0).collect(), vec![n]);
        assert_zip_op_matches_oracle::<4>(&a, &b);
        assert_zip_op_matches_oracle::<8>(&a, &b);
    }
}

// Broadcast bias: [4,3] + [3]. The row broadcasts → its view isn't packed →
// fallback; the swap exercises the other ordering.
#[test]
fn test_zip_op_broadcast_bias_and_swap() {
    let a = TensorData::new(signed_seq(12), vec![4, 3]);
    let row = TensorData::new(vec![1., -2., 3.], vec![3]);
    assert_zip_op_matches_oracle::<4>(&a, &row);
    assert_zip_op_matches_oracle::<4>(&row, &a);
}

// One transposed (strided) operand vs a contiguous one, both orderings; then
// both strided.
#[test]
fn test_zip_op_strided_operands() {
    let a = TensorData::new(signed_seq(6), vec![2, 3]);
    let b = TensorData::new((10..16).map(|i| i as f64).collect(), vec![3, 2])
        .permute(&[1, 0]); // [2,3], strided
    assert_zip_op_matches_oracle::<4>(&a, &b);
    assert_zip_op_matches_oracle::<4>(&b, &a);

    let a_t = TensorData::new(signed_seq(6), vec![3, 2]).permute(&[1, 0]);
    assert_zip_op_matches_oracle::<4>(&a_t, &b);
}

proptest! {
    // Same-shape contiguous (the SIMD arm) over every tail size, three widths.
    #[test]
    fn prop_zip_op_same_shape(n in 1usize..=128) {
        let a = TensorData::new(signed_seq(n), vec![n]);
        let b = TensorData::new((0..n).map(|i| i as f64 * 1.5 - 2.0).collect(), vec![n]);
        assert_zip_op_matches_oracle::<2>(&a, &b);
        assert_zip_op_matches_oracle::<4>(&a, &b);
        assert_zip_op_matches_oracle::<8>(&a, &b);
    }

    // Row broadcast (fallback path), both orderings, across many shapes.
    #[test]
    fn prop_zip_op_row_broadcast(rows in 1usize..=5, cols in 1usize..=5) {
        let mat = TensorData::new(signed_seq(rows * cols), vec![rows, cols]);
        let row = TensorData::new((0..cols).map(|i| i as f64 - 1.5).collect(), vec![cols]);
        assert_zip_op_matches_oracle::<4>(&mat, &row);
        assert_zip_op_matches_oracle::<4>(&row, &mat);
    }
}

// ===================================================================
// All ops × scalar oracle — correctness coverage for every Simd* op
//
// Every Op::scalar delegates to the trusted operators::X, so for ANY op the
// oracle is SimpleOps::{map,zip}(_, Op::scalar). Inputs are contiguous (so the
// SIMD path runs, not the scalar fallback) and respect each op's domain.
// Comparison is:
//   * exact  — arithmetic / compare / select ops (SIMD and scalar run the same
//     IEEE ops, so they agree bit-for-bit)
//   * approx — ops through exp / ln / recip, where StdFloat diverges from scalar
//     by a few ULP; relative tolerance 1e-9 with an absolute 1e-9 floor.
// Lengths 1..=20 sweep every tail size for LANES=8; layout coverage (strided /
// broadcast / offset) lives in the relu/add sections above.
// ===================================================================

// distinct "gradient" / second operand for the binary ops; straddles zero.
fn grad_seq(n: usize) -> Vec<f64> {
    (0..n).map(|i| i as f64 * 0.5 - 2.0).collect()
}

// strictly positive, for the log-domain ops.
fn positive_seq(n: usize) -> Vec<f64> {
    (0..n).map(|i| (i + 1) as f64 * 0.5).collect()
}

fn check_map_exact<Op: UnaryOp, const N: usize>(input: &TensorData) {
    let fast = FastOps::map_op::<Op, N>(input);
    let oracle = SimpleOps::map(input, Op::scalar);
    assert_eq!(fast.shape, oracle.shape);
    assert_eq!(
        &*fast.storage,
        &*oracle.storage,
        "exact map mismatch, len={}",
        input.size()
    );
}

fn check_map_approx<Op: UnaryOp, const N: usize>(input: &TensorData) {
    let fast = FastOps::map_op::<Op, N>(input);
    let oracle = SimpleOps::map(input, Op::scalar);
    assert_eq!(fast.shape, oracle.shape);
    for (&f, &o) in fast.storage.iter().zip(oracle.storage.iter()) {
        assert!(
            (f - o).abs() <= 1e-9 * o.abs().max(1.0),
            "approx map mismatch: simd={f} scalar={o}, len={}",
            input.size()
        );
    }
}

fn check_zip_exact<Op: BinaryOp, const N: usize>(a: &TensorData, b: &TensorData) {
    let fast = FastOps::zip_op::<Op, N>(a, b);
    let oracle = SimpleOps::zip(a, b, Op::scalar);
    assert_eq!(fast.shape, oracle.shape);
    assert_eq!(
        &*fast.storage,
        &*oracle.storage,
        "exact zip mismatch, len={}",
        a.size()
    );
}

fn check_zip_approx<Op: BinaryOp, const N: usize>(a: &TensorData, b: &TensorData) {
    let fast = FastOps::zip_op::<Op, N>(a, b);
    let oracle = SimpleOps::zip(a, b, Op::scalar);
    assert_eq!(fast.shape, oracle.shape);
    for (&f, &o) in fast.storage.iter().zip(oracle.storage.iter()) {
        assert!(
            (f - o).abs() <= 1e-9 * o.abs().max(1.0),
            "approx zip mismatch: simd={f} scalar={o}, len={}",
            a.size()
        );
    }
}

#[test]
fn test_all_unary_ops_match_scalar() {
    for n in 1..=20 {
        let s = TensorData::new(signed_seq(n), vec![n]); // nonzero, straddles 0
        let p = TensorData::new(positive_seq(n), vec![n]); // > 0
        // exact: arithmetic / select
        check_map_exact::<SimdReLU, LANES>(&s);
        check_map_exact::<SimdNeg, LANES>(&s);
        check_map_exact::<SimdId, LANES>(&s);
        check_map_exact::<SimdLeakyReLU, LANES>(&s);
        // approx: through exp / recip (general domain)
        check_map_approx::<SimdSigmoid, LANES>(&s);
        check_map_approx::<SimdExp, LANES>(&s);
        check_map_approx::<SimdInv, LANES>(&s);
        // approx: through ln (positive domain)
        check_map_approx::<SimdLog, LANES>(&p);
    }
}

#[test]
fn test_all_binary_ops_match_scalar() {
    for n in 1..=20 {
        let s = TensorData::new(signed_seq(n), vec![n]); // nonzero, straddles 0
        let p = TensorData::new(positive_seq(n), vec![n]); // > 0
        let d = TensorData::new(grad_seq(n), vec![n]); // second operand / upstream grad
        // real binary ops, exact
        check_zip_exact::<SimdAdd, LANES>(&s, &d);
        check_zip_exact::<SimdMul, LANES>(&s, &d);
        check_zip_exact::<SimdMax, LANES>(&s, &d);
        check_zip_exact::<SimdLt, LANES>(&s, &d); // mixed true/false
        check_zip_exact::<SimdEq, LANES>(&s, &s); // all-equal → exercises the true branch
        // backward derivatives, exact (arithmetic / select)
        check_zip_exact::<SimdReLUBack, LANES>(&s, &d);
        check_zip_exact::<SimdLeakyReLUBack, LANES>(&s, &d);
        check_zip_exact::<SimdSigmoidBack, LANES>(&s, &d);
        // backward derivatives, approx (through recip)
        check_zip_approx::<SimdLogBack, LANES>(&p, &d); // value > 0
        check_zip_approx::<SimdInvBack, LANES>(&s, &d); // value ≠ 0
    }
}
