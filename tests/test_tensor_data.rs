use std::rc::Rc;

use minitorch_rs::tensor_data::{contiguous_strides, TensorData};

// ===================================================================
// contiguous_strides
// ===================================================================

// Row-major rule: stride[i] = product of shape[i+1..]
// Innermost dim always has stride 1; each outer stride scales the inner.
#[test]
fn test_contiguous_strides_2d() {
    assert_eq!(contiguous_strides(&[3, 4]), vec![4, 1]);
}

#[test]
fn test_contiguous_strides_3d() {
    assert_eq!(contiguous_strides(&[2, 3, 4]), vec![12, 4, 1]);
}

#[test]
fn test_contiguous_strides_1d() {
    assert_eq!(contiguous_strides(&[5]), vec![1]);
}

// Empty shape = 0-d tensor (a scalar). No strides.
#[test]
fn test_contiguous_strides_empty() {
    assert_eq!(contiguous_strides(&[]), vec![] as Vec<usize>);
}

// All-ones shape: every stride is 1 because every shape[i+1..] product is 1.
// Worth testing because it's the degenerate case where row-major and
// column-major coincide.
#[test]
fn test_contiguous_strides_all_ones() {
    assert_eq!(contiguous_strides(&[1, 1, 1]), vec![1, 1, 1]);
}

// A "1" in the middle of the shape — stride should still come out right
// because the formula doesn't care about the value of the inner dims,
// only their product.
#[test]
fn test_contiguous_strides_with_unit_dim() {
    // shape=[2, 1, 3]
    //   stride[0] = shape[1] * shape[2] = 1 * 3 = 3
    //   stride[1] = shape[2]            = 3
    //   stride[2] = 1
    assert_eq!(contiguous_strides(&[2, 1, 3]), vec![3, 3, 1]);
}

// ===================================================================
// TensorData::new
// ===================================================================

#[test]
fn test_new_2d_fields() {
    let storage = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let t = TensorData::new(storage.clone(), vec![2, 3]);
    assert_eq!(&*t.storage, &storage);
    assert_eq!(t.shape, vec![2, 3]);
    assert_eq!(t.strides, vec![3, 1]);
}

#[test]
fn test_new_3d_fields() {
    let storage: Vec<f64> = (0..24).map(|i| i as f64).collect();
    let t = TensorData::new(storage.clone(), vec![2, 3, 4]);
    assert_eq!(t.storage.len(), 24);
    assert_eq!(t.shape, vec![2, 3, 4]);
    assert_eq!(t.strides, vec![12, 4, 1]);
}

#[test]
fn test_new_strides_match_contiguous() {
    let shape = vec![3, 5, 2];
    let storage = vec![0.0; 30];
    let t = TensorData::new(storage, shape.clone());
    assert_eq!(t.strides, contiguous_strides(&shape));
}

// Empty shape represents a 0-d scalar tensor.
// The empty product is 1, so storage must have exactly one element.
#[test]
fn test_new_zero_dim_tensor() {
    let t = TensorData::new(vec![42.0], vec![]);
    assert_eq!(&*t.storage, &vec![42.0]);
    assert_eq!(t.shape, vec![] as Vec<usize>);
    assert_eq!(t.strides, vec![] as Vec<usize>);
}

// Storage length must equal the product of shape dims, or we get
// out-of-bounds the first time any op tries to index.
#[test]
#[should_panic]
fn test_new_panics_on_storage_shape_mismatch() {
    // shape product = 6, but storage has 5 elements
    TensorData::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![2, 3]);
}

#[test]
#[should_panic]
fn test_new_panics_on_zero_dim_with_wrong_storage() {
    // 0-d tensor needs exactly 1 element, got 0
    TensorData::new(vec![], vec![]);
}

// ===================================================================
// zeros / ones
// ===================================================================

#[test]
fn test_zeros_2d() {
    let t = TensorData::zeros(vec![2, 3]);
    assert_eq!(t.shape, vec![2, 3]);
    assert_eq!(t.strides, vec![3, 1]);
    assert_eq!(&*t.storage, &vec![0.0; 6]);
}

#[test]
fn test_ones_3d() {
    let t = TensorData::ones(vec![2, 2, 2]);
    assert_eq!(t.shape, vec![2, 2, 2]);
    assert_eq!(&*t.storage, &vec![1.0; 8]);
}

#[test]
fn test_zeros_empty_shape_is_single_zero() {
    let t = TensorData::zeros(vec![]);
    assert_eq!(&*t.storage, &vec![0.0]);
}

// ===================================================================
// flat_index, get, set
// ===================================================================

// For a (2, 3) tensor with strides=[3, 1]:
//   T[0,0] → 0
//   T[0,2] → 2
//   T[1,0] → 3
//   T[1,2] → 5
#[test]
fn test_flat_index_2d() {
    let t = TensorData::zeros(vec![2, 3]);
    assert_eq!(t.flat_index(&[0, 0]), 0);
    assert_eq!(t.flat_index(&[0, 2]), 2);
    assert_eq!(t.flat_index(&[1, 0]), 3);
    assert_eq!(t.flat_index(&[1, 2]), 5);
}

// For (2, 3, 4) tensor with strides=[12, 4, 1]:
//   T[1, 2, 3] → 1*12 + 2*4 + 3*1 = 23
//   T[0, 0, 0] → 0
//   T[1, 0, 0] → 12
#[test]
fn test_flat_index_3d() {
    let t = TensorData::zeros(vec![2, 3, 4]);
    assert_eq!(t.flat_index(&[0, 0, 0]), 0);
    assert_eq!(t.flat_index(&[1, 0, 0]), 12);
    assert_eq!(t.flat_index(&[0, 2, 3]), 11);
    assert_eq!(t.flat_index(&[1, 2, 3]), 23);
}

#[test]
fn test_get_reads_correct_element() {
    let t = TensorData::new(
        vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        vec![2, 3],
    );
    assert_eq!(t.get(&[0, 0]), 10.0);
    assert_eq!(t.get(&[0, 2]), 30.0);
    assert_eq!(t.get(&[1, 0]), 40.0);
    assert_eq!(t.get(&[1, 2]), 60.0);
}

#[test]
fn test_set_then_get_roundtrip() {
    let mut t = TensorData::zeros(vec![2, 3]);
    t.set(&[1, 2], 7.5);
    assert_eq!(t.get(&[1, 2]), 7.5);
    // other elements untouched
    assert_eq!(t.get(&[0, 0]), 0.0);
    assert_eq!(t.get(&[1, 0]), 0.0);
}

// COW guarantee: if storage is shared (e.g. via permute), mutating one
// view via `set` must NOT visibly mutate the other. Rc::make_mut should
// clone the inner Vec before writing.
#[test]
fn test_set_cow_does_not_affect_shared_view() {
    let mut a = TensorData::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = a.permute(&[1, 0]);
    // shared storage right after permute
    assert!(Rc::ptr_eq(&a.storage, &b.storage));

    a.set(&[0, 0], 99.0);

    // after set, a's storage has been cloned by Rc::make_mut
    assert!(!Rc::ptr_eq(&a.storage, &b.storage));
    // a sees the new value, b sees the original
    assert_eq!(a.storage[0], 99.0);
    assert_eq!(b.storage[0], 1.0);
}

// ===================================================================
// size, ndim
// ===================================================================

#[test]
fn test_size_returns_logical_element_count() {
    let t = TensorData::zeros(vec![2, 3, 4]);
    assert_eq!(t.size(), 24);
}

#[test]
fn test_size_zero_dim_is_one() {
    let t = TensorData::new(vec![42.0], vec![]);
    assert_eq!(t.size(), 1);
}

#[test]
fn test_ndim() {
    assert_eq!(TensorData::zeros(vec![5]).ndim(),       1);
    assert_eq!(TensorData::zeros(vec![3, 4]).ndim(),    2);
    assert_eq!(TensorData::zeros(vec![2, 3, 4]).ndim(), 3);
    assert_eq!(TensorData::zeros(vec![]).ndim(),        0);
}

// ===================================================================
// permute
// ===================================================================

// Identity permute returns shape + strides unchanged.
#[test]
fn test_permute_identity() {
    let t = TensorData::zeros(vec![2, 3, 4]);
    let p = t.permute(&[0, 1, 2]);
    assert_eq!(p.shape, vec![2, 3, 4]);
    assert_eq!(p.strides, vec![12, 4, 1]);
}

// 2D transpose: shape [3, 4] → [4, 3], strides [4, 1] → [1, 4].
#[test]
fn test_permute_2d_transpose() {
    let t = TensorData::zeros(vec![3, 4]);
    let p = t.permute(&[1, 0]);
    assert_eq!(p.shape, vec![4, 3]);
    assert_eq!(p.strides, vec![1, 4]);
}

// 3D permute: axes=[2, 0, 1].
//   new_shape[i] = old_shape[axes[i]]
//   new_shape[0] = old_shape[2] = 4
//   new_shape[1] = old_shape[0] = 2
//   new_shape[2] = old_shape[1] = 3
// Same rule for strides.
#[test]
fn test_permute_3d_rotation() {
    let t = TensorData::zeros(vec![2, 3, 4]); // strides [12, 4, 1]
    let p = t.permute(&[2, 0, 1]);
    assert_eq!(p.shape,   vec![4, 2, 3]);
    assert_eq!(p.strides, vec![1, 12, 4]);
}

// permute must NOT copy storage — it's the whole point of strided views.
#[test]
fn test_permute_shares_storage() {
    let t = TensorData::new(
        (0..12).map(|i| i as f64).collect(),
        vec![3, 4],
    );
    let p = t.permute(&[1, 0]);
    assert!(Rc::ptr_eq(&t.storage, &p.storage));
}

// Storage sharing means the transposed view sees the same numbers
// through different indexing. T[i, j] in the original equals
// T_transposed[j, i].
#[test]
fn test_permute_view_consistent_with_original() {
    let t = TensorData::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
    );
    let p = t.permute(&[1, 0]); // shape [3, 2]
    for i in 0..2 {
        for j in 0..3 {
            assert_eq!(t.get(&[i, j]), p.get(&[j, i]));
        }
    }
}

// ===================================================================
// iter_indices
// ===================================================================

// Row-major lexicographic order: outermost dim varies slowest.
#[test]
fn test_iter_indices_2d() {
    let t = TensorData::zeros(vec![2, 3]);
    let indices: Vec<Vec<usize>> = t.iter_indices().collect();
    assert_eq!(
        indices,
        vec![
            vec![0, 0], vec![0, 1], vec![0, 2],
            vec![1, 0], vec![1, 1], vec![1, 2],
        ]
    );
}

#[test]
fn test_iter_indices_3d_count() {
    let t = TensorData::zeros(vec![2, 3, 4]);
    assert_eq!(t.iter_indices().count(), 24);
}

#[test]
fn test_iter_indices_3d_first_and_last() {
    let t = TensorData::zeros(vec![2, 3, 4]);
    let indices: Vec<Vec<usize>> = t.iter_indices().collect();
    assert_eq!(indices.first().unwrap(), &vec![0, 0, 0]);
    assert_eq!(indices.last().unwrap(),  &vec![1, 2, 3]);
}

#[test]
fn test_iter_indices_1d() {
    let t = TensorData::zeros(vec![4]);
    let indices: Vec<Vec<usize>> = t.iter_indices().collect();
    assert_eq!(indices, vec![vec![0], vec![1], vec![2], vec![3]]);
}

// 0-d tensor has logical size 1, so iter_indices yields exactly one
// empty Vec (the "index into nothing" that selects the single element).
#[test]
fn test_iter_indices_zero_dim() {
    let t = TensorData::new(vec![42.0], vec![]);
    let indices: Vec<Vec<usize>> = t.iter_indices().collect();
    assert_eq!(indices, vec![vec![] as Vec<usize>]);
}
