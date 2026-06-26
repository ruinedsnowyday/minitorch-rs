use minitorch_rs::tensor_data::TensorData;
use minitorch_rs::tensor_ops::{
    SimpleOps, TensorOps, broadcast_shape, broadcast_strides, matmul_2d,
};

const EPS: f64 = 1e-9;

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < EPS
}

// ===================================================================
// map
// ===================================================================

#[test]
fn test_map_doubles_values() {
    let t = TensorData::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let out = SimpleOps::map(&t, |x| x * 2.0);
    assert_eq!(&*out.storage, &vec![2.0, 4.0, 6.0, 8.0]);
    assert_eq!(out.shape, vec![2, 2]);
}

#[test]
fn test_map_preserves_shape_and_strides() {
    let t = TensorData::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let out = SimpleOps::map(&t, |x| x + 1.0);
    assert_eq!(out.shape, t.shape);
    assert_eq!(out.strides, t.strides);
    assert_eq!(out.size(), 6);
}

// map walks storage directly, so a permuted (non-contiguous) input still
// gets f applied to every element. The result shares the same view shape
// and strides, so logical reads through get() return the transformed values.
#[test]
fn test_map_on_permuted_input() {
    let t = TensorData::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let p = t.permute(&[1, 0]); // shape [3, 2], non-contiguous strides
    let out = SimpleOps::map(&p, |x| x * 10.0);
    // logical reads via get must produce f(p.get(idx)) == 10 * p.get(idx)
    for i in 0..3 {
        for j in 0..2 {
            assert!(close(out.get(&[i, j]), p.get(&[i, j]) * 10.0));
        }
    }
}

// ===================================================================
// zip — element-wise with broadcasting
// ===================================================================

// Same shapes: zip is just elementwise.
#[test]
fn test_zip_same_shapes() {
    let a = TensorData::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = TensorData::new(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]);
    let c = SimpleOps::zip(&a, &b, |x, y| x + y);
    assert_eq!(c.shape, vec![2, 2]);
    assert_eq!(&*c.storage, &vec![11.0, 22.0, 33.0, 44.0]);
}

// Broadcasting [3] across [4, 3]: each row of the (4, 3) result picks up
// the same (10, 20, 30) values from `a`, added to b's row.
#[test]
fn test_zip_broadcast_1d_against_2d() {
    let a = TensorData::new(vec![10.0, 20.0, 30.0], vec![3]);
    let b = TensorData::new(
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        vec![4, 3],
    );
    let c = SimpleOps::zip(&a, &b, |x, y| x + y);
    assert_eq!(c.shape, vec![4, 3]);
    assert_eq!(
        &*c.storage,
        &vec![
            11.0, 22.0, 33.0, 14.0, 25.0, 36.0, 17.0, 28.0, 39.0, 20.0, 31.0, 42.0
        ]
    );
}

// Both operands have a size-1 dim — broadcast in opposite directions.
//   a: [4, 1] → [4, 3] (broadcast col)
//   b: [1, 3] → [4, 3] (broadcast row)
// Result: outer product structure.
#[test]
fn test_zip_broadcast_both_have_size_1_dim() {
    let a = TensorData::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]);
    let b = TensorData::new(vec![10.0, 20.0, 30.0], vec![1, 3]);
    let c = SimpleOps::zip(&a, &b, |x, y| x * y);
    assert_eq!(c.shape, vec![4, 3]);
    // c[i, j] = a[i, 0] * b[0, j]
    let expected = vec![
        10.0, 20.0, 30.0, 20.0, 40.0, 60.0, 30.0, 60.0, 90.0, 40.0, 80.0, 120.0,
    ];
    assert_eq!(&*c.storage, &expected);
}

// Scalar-like ([1]) broadcasting against a higher-rank tensor.
#[test]
fn test_zip_broadcast_scalar_like() {
    let a = TensorData::new(vec![5.0], vec![1]);
    let b = TensorData::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let c = SimpleOps::zip(&a, &b, |x, y| x + y);
    assert_eq!(c.shape, vec![2, 2]);
    assert_eq!(&*c.storage, &vec![6.0, 7.0, 8.0, 9.0]);
}

#[test]
#[should_panic]
fn test_zip_panics_on_incompatible_shapes() {
    let a = TensorData::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
    let b = TensorData::new(vec![10.0, 20.0, 30.0], vec![3]);
    let _ = SimpleOps::zip(&a, &b, |x, y| x + y);
}

// ===================================================================
// reduce
// ===================================================================

// Sum along dim 0 of a (2, 3) tensor — collapses rows, output is [3].
//   storage: 1 2 3
//            4 5 6
//   dim-0 sum: [5, 7, 9]
#[test]
fn test_reduce_sum_dim_0() {
    let t = TensorData::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let out = SimpleOps::reduce(&t, |a, b| a + b, 0.0, 0, false);
    assert_eq!(out.shape, vec![3]);
    assert_eq!(&*out.storage, &vec![5.0, 7.0, 9.0]);
}

// Sum along dim 1 — collapses cols, output is [2].
//   row sums: [6, 15]
#[test]
fn test_reduce_sum_dim_1() {
    let t = TensorData::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let out = SimpleOps::reduce(&t, |a, b| a + b, 0.0, 1, false);
    assert_eq!(out.shape, vec![2]);
    assert_eq!(&*out.storage, &vec![6.0, 15.0]);
}

// Reduce with a different binary op + identity: product along dim 0.
//   col products: [1*4, 2*5, 3*6] = [4, 10, 18]
#[test]
fn test_reduce_product_dim_0() {
    let t = TensorData::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let out = SimpleOps::reduce(&t, |a, b| a * b, 1.0, 0, false);
    assert_eq!(out.shape, vec![3]);
    assert_eq!(&*out.storage, &vec![4.0, 10.0, 18.0]);
}

// 3D tensor reduction along middle dim.
//   shape [2, 3, 2], reduce dim 1 → output shape [2, 2]
//   Each output cell sums 3 values across the middle dim.
#[test]
fn test_reduce_3d_middle_dim() {
    // values laid out for shape [2, 3, 2]:
    //   batch 0:  (1, 2), (3, 4), (5, 6)   → col sums (9, 12)
    //   batch 1:  (7, 8), (9,10), (11,12) → col sums (27, 30)
    let t = TensorData::new(
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        vec![2, 3, 2],
    );
    let out = SimpleOps::reduce(&t, |a, b| a + b, 0.0, 1, false);
    assert_eq!(out.shape, vec![2, 2]);
    assert_eq!(&*out.storage, &vec![9.0, 12.0, 27.0, 30.0]);
}

#[test]
#[should_panic]
fn test_reduce_panics_on_dim_out_of_range() {
    let t = TensorData::new(vec![1.0, 2.0], vec![2]);
    // dim 5 doesn't exist on a 1D tensor
    let _ = SimpleOps::reduce(&t, |a, b| a + b, 0.0, 5, false);
}

// ---- keep_dims = true ----

// Same data as test_reduce_sum_dim_0, but the reduced dim is kept as 1.
//   shape [2, 3] reduced along dim 0 with keep_dims → shape [1, 3]
//   values unchanged: [5, 7, 9]
#[test]
fn test_reduce_keepdims_sum_dim_0() {
    let t = TensorData::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let out = SimpleOps::reduce(&t, |a, b| a + b, 0.0, 0, true);
    assert_eq!(out.shape, vec![1, 3]);
    assert_eq!(&*out.storage, &vec![5.0, 7.0, 9.0]);
}

// Reduce dim 1 with keep_dims — output rank matches input rank.
//   shape [2, 3] → [2, 1], values [6, 15]
#[test]
fn test_reduce_keepdims_sum_dim_1() {
    let t = TensorData::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let out = SimpleOps::reduce(&t, |a, b| a + b, 0.0, 1, true);
    assert_eq!(out.shape, vec![2, 1]);
    assert_eq!(&*out.storage, &vec![6.0, 15.0]);
}

// 3D middle-dim reduction with keep_dims preserves rank.
//   shape [2, 3, 2] → [2, 1, 2], same values as test_reduce_3d_middle_dim
#[test]
fn test_reduce_keepdims_3d_middle_dim() {
    let t = TensorData::new(
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        vec![2, 3, 2],
    );
    let out = SimpleOps::reduce(&t, |a, b| a + b, 0.0, 1, true);
    assert_eq!(out.shape, vec![2, 1, 2]);
    assert_eq!(&*out.storage, &vec![9.0, 12.0, 27.0, 30.0]);
}

// Reducing the only dim of a 1D tensor with keep_dims → shape [1].
// Without keep_dims this would give shape [].
#[test]
fn test_reduce_keepdims_rank_one_to_size_one() {
    let t = TensorData::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
    let out = SimpleOps::reduce(&t, |a, b| a + b, 0.0, 0, true);
    assert_eq!(out.shape, vec![1]);
    assert_eq!(&*out.storage, &vec![10.0]);
}

// Reducing the last dim of a 2D tensor with keep_dims — covers the
// "dim is the last axis" path (different splice arithmetic than dim=0).
#[test]
fn test_reduce_keepdims_last_dim_of_2d() {
    let t = TensorData::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let out = SimpleOps::reduce(&t, |a, b| a + b, 0.0, 1, true);
    assert_eq!(out.shape, vec![2, 1]);
    assert_eq!(&*out.storage, &vec![6.0, 15.0]);
}

// ===================================================================
// broadcast_shape
// ===================================================================

#[test]
fn test_broadcast_shape_1d_against_2d() {
    assert_eq!(broadcast_shape(&[3], &[4, 3]), vec![4, 3]);
    assert_eq!(broadcast_shape(&[4, 3], &[3]), vec![4, 3]);
}

#[test]
fn test_broadcast_shape_both_have_unit_dims() {
    // [4, 1] vs [1, 3] → [4, 3]
    assert_eq!(broadcast_shape(&[4, 1], &[1, 3]), vec![4, 3]);
}

#[test]
fn test_broadcast_shape_equal_shapes() {
    assert_eq!(broadcast_shape(&[2, 3], &[2, 3]), vec![2, 3]);
}

#[test]
fn test_broadcast_shape_scalar_like() {
    assert_eq!(broadcast_shape(&[1], &[5, 4, 3]), vec![5, 4, 3]);
}

#[test]
#[should_panic]
fn test_broadcast_shape_panics_on_mismatch() {
    // 4 vs 3 with neither being 1 — incompatible
    let _ = broadcast_shape(&[4], &[3]);
}

#[test]
#[should_panic]
fn test_broadcast_shape_panics_on_2d_mismatch() {
    let _ = broadcast_shape(&[2, 4], &[3, 4]);
}

// ===================================================================
// broadcast_strides
// ===================================================================

// orig [3] strides [1] → target [4, 3]: leading dim is new (stride 0),
// trailing dim matches and keeps stride 1.
#[test]
fn test_broadcast_strides_1d_to_2d() {
    assert_eq!(broadcast_strides(&[3], &[1], &[4, 3]), vec![0, 1]);
}

// orig [4, 1] strides [1, 1] → target [4, 3]:
//   dim 0 matches (4 → 4), keep stride 1
//   dim 1 size-1 broadcast (1 → 3), stride 0
#[test]
fn test_broadcast_strides_unit_inner_dim() {
    assert_eq!(broadcast_strides(&[4, 1], &[1, 1], &[4, 3]), vec![1, 0]);
}

// orig [1, 3] strides [3, 1] → target [4, 3]:
//   dim 0 size-1 broadcast (1 → 4), stride 0
//   dim 1 matches (3 → 3), keep stride 1
// Note: the original "stride 3" on dim 0 is irrelevant once that dim is broadcast.
#[test]
fn test_broadcast_strides_unit_outer_dim() {
    assert_eq!(broadcast_strides(&[1, 3], &[3, 1], &[4, 3]), vec![0, 1]);
}

// Same shape — no broadcasting at all, strides pass through unchanged.
#[test]
fn test_broadcast_strides_equal_shapes() {
    assert_eq!(broadcast_strides(&[2, 3], &[3, 1], &[2, 3]), vec![3, 1]);
}

#[test]
#[should_panic]
fn test_broadcast_strides_panics_on_mismatch() {
    // orig dim is 4, target dim is 3, neither is 1 — not broadcastable
    let _ = broadcast_strides(&[4], &[1], &[3]);
}

// ===================================================================
// matmul_2d — direct 2D kernel
// ===================================================================

// A: [[1, 2, 3],     B: [[7,  8,  9, 10],
//     [4, 5, 6]]         [11, 12, 13, 14],
//                        [15, 16, 17, 18]]
// C[0] = (1*7+2*11+3*15, 1*8+2*12+3*16, 1*9+2*13+3*17, 1*10+2*14+3*18)
//      = (74, 80, 86, 92)
// C[1] = (4*7+5*11+6*15, 4*8+5*12+6*16, 4*9+5*13+6*17, 4*10+5*14+6*18)
//      = (173, 188, 203, 218)
#[test]
fn test_matmul_2d_basic_2x3_times_3x4() {
    let a = TensorData::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = TensorData::new(
        vec![
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
        ],
        vec![3, 4],
    );
    let c = matmul_2d(&a, &b);
    assert_eq!(c.shape, vec![2, 4]);
    assert_eq!(
        &*c.storage,
        &vec![74.0, 80.0, 86.0, 92.0, 173.0, 188.0, 203.0, 218.0]
    );
}

// I @ A == A for the identity matrix.
#[test]
fn test_matmul_2d_identity() {
    let i = TensorData::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
    let a = TensorData::new(vec![3.0, 5.0, 7.0, 9.0], vec![2, 2]);
    let c = matmul_2d(&i, &a);
    assert_eq!(c.shape, vec![2, 2]);
    assert_eq!(&*c.storage, &vec![3.0, 5.0, 7.0, 9.0]);
}

// 2x3 @ 3x1 → 2x1 (effectively matrix-vector but expressed as 2D matmul).
#[test]
fn test_matmul_2d_matrix_times_column_vector() {
    let a = TensorData::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let v = TensorData::new(vec![10.0, 20.0, 30.0], vec![3, 1]);
    let c = matmul_2d(&a, &v);
    assert_eq!(c.shape, vec![2, 1]);
    // [1*10+2*20+3*30, 4*10+5*20+6*30] = [140, 320]
    assert_eq!(&*c.storage, &vec![140.0, 320.0]);
}

// Non-contiguous input: build a 3x2 contiguous tensor, transpose via permute
// to get a 2x3 non-contiguous view, then matmul. matmul_2d uses .get(), so it
// must work correctly through stride math even when storage isn't row-major.
//
// Storage [1,2,3,4,5,6] reshape (3, 2):  [[1,2], [3,4], [5,6]]
// Transposed view (2, 3):                [[1,3,5], [2,4,6]]
// Multiply by [[10], [20], [30]] (3, 1):
//   [1*10+3*20+5*30, 2*10+4*20+6*30] = [220, 280]
#[test]
fn test_matmul_2d_handles_non_contiguous_input() {
    let a_contig = TensorData::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
    let a_t = a_contig.permute(&[1, 0]); // shape [2, 3], non-contiguous
    let v = TensorData::new(vec![10.0, 20.0, 30.0], vec![3, 1]);
    let c = matmul_2d(&a_t, &v);
    assert_eq!(c.shape, vec![2, 1]);
    assert_eq!(&*c.storage, &vec![220.0, 280.0]);
}

#[test]
#[should_panic]
fn test_matmul_2d_panics_on_inner_dim_mismatch() {
    let a = TensorData::new(vec![1.0; 6], vec![2, 3]);
    let b = TensorData::new(vec![1.0; 20], vec![4, 5]); // inner dim 3 vs 4
    let _ = matmul_2d(&a, &b);
}

#[test]
#[should_panic]
fn test_matmul_2d_panics_on_non_2d_input() {
    let a = TensorData::new(vec![1.0; 24], vec![2, 3, 4]); // 3D, not 2D
    let b = TensorData::new(vec![1.0; 8], vec![4, 2]);
    let _ = matmul_2d(&a, &b);
}

// ===================================================================
// SimpleOps::matmul — batched dispatch
// ===================================================================

// 2D inputs route through the batched wrapper with batch shape [], producing
// the same result as calling matmul_2d directly. Sanity check the dispatch.
#[test]
fn test_simpleops_matmul_2d_dispatch() {
    let a = TensorData::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = TensorData::new(
        vec![
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
        ],
        vec![3, 4],
    );
    let c = SimpleOps::matmul(&a, &b);
    assert_eq!(c.shape, vec![2, 4]);
    assert_eq!(
        &*c.storage,
        &vec![74.0, 80.0, 86.0, 92.0, 173.0, 188.0, 203.0, 218.0]
    );
}

// Batched: shape [B, M, K] × [B, K, N] → [B, M, N], no broadcasting needed.
// B=2, M=K=N=2.
//   batch 0:  [[1,2],[3,4]] @ [[5,6],[7,8]]
//           = [[5+14, 6+16], [15+28, 18+32]]
//           = [[19, 22], [43, 50]]
//   batch 1:  [[9,10],[11,12]] @ [[13,14],[15,16]]
//           = [[117+150, 126+160], [143+180, 154+192]]
//           = [[267, 286], [323, 346]]
#[test]
fn test_simpleops_matmul_batched_same_batch_shape() {
    let a = TensorData::new(
        vec![1.0, 2.0, 3.0, 4.0, 9.0, 10.0, 11.0, 12.0],
        vec![2, 2, 2],
    );
    let b = TensorData::new(
        vec![5.0, 6.0, 7.0, 8.0, 13.0, 14.0, 15.0, 16.0],
        vec![2, 2, 2],
    );
    let c = SimpleOps::matmul(&a, &b);
    assert_eq!(c.shape, vec![2, 2, 2]);
    assert_eq!(
        &*c.storage,
        &vec![19.0, 22.0, 43.0, 50.0, 267.0, 286.0, 323.0, 346.0]
    );
}

// Broadcasting on the batch dim: [B, M, K] × [K, N] → [B, M, N].
// b has no batch dim; it gets broadcast across both batches of a.
//   a = [[[1,2,3], [4,5,6]],
//        [[7,8,9], [10,11,12]]]                     shape [2, 2, 3]
//   b = [[1,0], [0,1], [1,1]]                       shape [3, 2]
//
//   batch 0: [[1,2,3],[4,5,6]] @ b
//          = [[1*1+2*0+3*1, 1*0+2*1+3*1], [4+6, 5+6]]
//          = [[4, 5], [10, 11]]
//   batch 1: [[7,8,9],[10,11,12]] @ b
//          = [[7+9, 8+9], [10+12, 11+12]]
//          = [[16, 17], [22, 23]]
#[test]
fn test_simpleops_matmul_broadcast_batch() {
    let a = TensorData::new(
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        vec![2, 2, 3],
    );
    let b = TensorData::new(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![3, 2]);
    let c = SimpleOps::matmul(&a, &b);
    assert_eq!(c.shape, vec![2, 2, 2]);
    assert_eq!(
        &*c.storage,
        &vec![4.0, 5.0, 10.0, 11.0, 16.0, 17.0, 22.0, 23.0]
    );
}

#[test]
#[should_panic]
fn test_simpleops_matmul_panics_on_ndim_less_than_2() {
    let a = TensorData::new(vec![1.0, 2.0, 3.0], vec![3]); // 1D vector
    let b = TensorData::new(vec![1.0, 2.0, 3.0], vec![3]);
    let _ = SimpleOps::matmul(&a, &b);
}

#[test]
#[should_panic]
fn test_simpleops_matmul_panics_on_inner_dim_mismatch() {
    let a = TensorData::new(vec![1.0; 6], vec![2, 3]);
    let b = TensorData::new(vec![1.0; 20], vec![4, 5]); // 3 vs 4
    let _ = SimpleOps::matmul(&a, &b);
}

// ===================================================================
// Adversarial tests — designed to fail on incorrect implementations
// that happen to produce the right .get() values but violate
// structural invariants (storage tightness, aliasing, etc.)
// ===================================================================

// A correct map must produce a fresh row-major Vec of length product(shape),
// regardless of the input's layout. The pre-fix impl iterated the input's
// underlying Vec, so an oversized backing storage (e.g. from a slice) would
// propagate verbatim — values via .get() looked right but storage.len() was
// wrong, which is a footgun for any future set()/in-place op.
#[test]
fn test_simpleops_map_on_offset_input_produces_tight_storage() {
    use std::rc::Rc;
    // Logical view is shape [2,2] starting at storage offset 2 of a 7-element
    // backing Vec. Logical values: [1, 2, 3, 4].
    let raw = TensorData {
        storage: Rc::new(vec![999., 999., 1., 2., 3., 4., 999.]),
        storage_offset: 2,
        shape: vec![2, 2],
        strides: vec![2, 1],
    };
    let out = SimpleOps::map(&raw, |x| x + 10.0);

    // Logical values must reflect the input view (not the backing buffer).
    let collected: Vec<f64> = out.iter_indices().map(|i| out.get(&i)).collect();
    assert_eq!(collected, vec![11., 12., 13., 14.]);
    // Structural: storage tight, offset reset, strides contiguous.
    assert_eq!(
        out.storage.len(),
        4,
        "storage must be tight to logical size"
    );
    assert_eq!(out.storage_offset, 0);
    assert_eq!(out.strides, vec![2, 1]);
}

// Broadcasting produces stride-0 views where one storage cell aliases many
// logical positions. A correct map must materialize distinct cells; the
// pre-fix impl preserved the aliasing.
#[test]
fn test_simpleops_map_on_broadcast_view_materializes_distinct_cells() {
    let scalar = TensorData::new(vec![5.0], vec![1]);
    let broadcast = scalar.broadcast_to(&[3]); // stride [0], storage.len() = 1
    let out = SimpleOps::map(&broadcast, |x| x * 2.0);

    assert_eq!(out.shape, vec![3]);
    let collected: Vec<f64> = out.iter_indices().map(|i| out.get(&i)).collect();
    assert_eq!(collected, vec![10., 10., 10.]);
    // The critical assertion — three distinct storage cells, not one alias.
    assert_eq!(
        out.storage.len(),
        3,
        "broadcast map output must not preserve stride-0 aliasing"
    );
    assert_eq!(out.strides, vec![1]);
}
