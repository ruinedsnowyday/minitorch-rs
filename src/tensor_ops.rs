use crate::tensor_data::{TensorData, iter_indices_of_shape};
use std::rc::Rc;

pub trait TensorOps {
    /// apply unary operation element-wise
    fn map(input: &TensorData, f: impl Fn(f64) -> f64) -> TensorData;

    /// apply binary operation element-wise with broadcasting
    fn zip(a: &TensorData, b: &TensorData, f: impl Fn(f64, f64) -> f64)
    -> TensorData;

    /// reduce along a single dimension with a binary operation and identity element
    fn reduce(
        input: &TensorData,
        f: impl Fn(f64, f64) -> f64,
        init: f64,
        dim: usize,
    ) -> TensorData;

    fn matmul(a: &TensorData, b: &TensorData) -> TensorData;
}

pub struct SimpleOps;

impl TensorOps for SimpleOps {
    fn map(input: &TensorData, f: impl Fn(f64) -> f64) -> TensorData {
        TensorData {
            storage: Rc::new(input.storage.iter().map(|&x| f(x)).collect()),
            storage_offset: input.storage_offset,
            shape: input.shape.clone(),
            strides: input.strides.clone(),
        }
    }

    fn zip(
        a: &TensorData,
        b: &TensorData,
        f: impl Fn(f64, f64) -> f64,
    ) -> TensorData {
        let out_shape = broadcast_shape(&a.shape, &b.shape);
        let a_view = a.broadcast_to(&out_shape);
        let b_view = b.broadcast_to(&out_shape);
        let mut storage: Vec<f64> = Vec::with_capacity(out_shape.iter().product());
        for idx_vec in a_view.iter_indices() {
            storage.push(f(a_view.get(&idx_vec), b_view.get(&idx_vec)));
        }
        TensorData::new(storage, out_shape)
    }

    fn reduce(
        input: &TensorData,
        f: impl Fn(f64, f64) -> f64,
        init: f64,
        dim: usize,
    ) -> TensorData {
        assert!(
            dim < input.shape.len(),
            "Can't reduce along axis {} for tensor with dimensions {:?}",
            dim,
            &input.shape
        );
        let target_shape: Vec<usize> = input
            .shape
            .iter()
            .enumerate()
            .filter_map(|(idx, &len)| if idx != dim { Some(len) } else { None })
            .collect();
        let mut out_storage = Vec::with_capacity(target_shape.iter().product());
        for out_idx in iter_indices_of_shape(&target_shape) {
            let mut val = init;
            let mut idx: Vec<usize> = Vec::with_capacity(input.shape.len());
            idx.extend_from_slice(&out_idx[..dim]);
            idx.push(0);
            idx.extend_from_slice(&out_idx[dim..]);
            for i in 0..input.shape[dim] {
                idx[dim] = i;
                val = f(val, input.get(&idx));
            }
            out_storage.push(val);
        }
        TensorData::new(out_storage, target_shape)
    }

    fn matmul(a: &TensorData, b: &TensorData) -> TensorData {
        assert!(
            a.ndim() >= 2 && b.ndim() >= 2,
            "matmul: both operands need >= 2 dims, got shapes {:?} and {:?}",
            a.shape,
            b.shape
        );
        let a_batch_shape = &a.shape[..a.ndim() - 2];
        let b_batch_shape = &b.shape[..b.ndim() - 2];
        let out_batch_shape = broadcast_shape(a_batch_shape, b_batch_shape);
        let (m, n) = (a.shape[a.ndim() - 2], b.shape[b.ndim() - 1]);
        let k = a.shape[a.ndim() - 1];
        assert_eq!(a.shape[a.ndim() - 1], b.shape[b.ndim() - 2]);
        let mut out_shape = out_batch_shape.clone();
        out_shape.extend_from_slice(&[m, n]);

        let a_batch_strides = broadcast_strides(
            a_batch_shape,
            &a.strides[..a.ndim() - 2],
            &out_batch_shape,
        );
        let b_batch_strides = broadcast_strides(
            b_batch_shape,
            &b.strides[..b.ndim() - 2],
            &out_batch_shape,
        );

        let mut out_storage: Vec<f64> =
            Vec::with_capacity(out_shape.iter().product());
        for batch_idx in iter_indices_of_shape(&out_batch_shape) {
            let a_off: usize = a.storage_offset
                + a_batch_strides
                    .iter()
                    .zip(&batch_idx)
                    .map(|(s, i)| s * i)
                    .sum::<usize>();
            let b_off: usize = b.storage_offset
                + b_batch_strides
                    .iter()
                    .zip(&batch_idx)
                    .map(|(s, i)| s * i)
                    .sum::<usize>();

            // build 2D views that share storage with a and b
            let a_2d = TensorData {
                storage: Rc::clone(&a.storage),
                storage_offset: a_off,
                shape: vec![m, k],
                strides: vec![a.strides[a.ndim() - 2], a.strides[a.ndim() - 1]],
            };
            let b_2d = TensorData {
                storage: Rc::clone(&b.storage),
                storage_offset: b_off,
                shape: vec![k, n],
                strides: vec![b.strides[b.ndim() - 2], b.strides[b.ndim() - 1]],
            };

            let mat = matmul_2d(&a_2d, &b_2d);
            out_storage.extend_from_slice(&mat.storage);
        }
        TensorData::new(out_storage, out_shape)
    }
}

pub fn broadcast_shape(shape_a: &[usize], shape_b: &[usize]) -> Vec<usize> {
    let len = shape_a.len().max(shape_b.len());
    let mut ext_a = vec![1; len];
    let mut ext_b = vec![1; len];
    for (i, dim) in (0..len).rev().zip(shape_a.iter().rev()) {
        ext_a[i] = *dim;
    }
    for (i, dim) in (0..len).rev().zip(shape_b.iter().rev()) {
        ext_b[i] = *dim;
    }
    let mut shape_out = vec![1; len];
    for (i, (dim_a, dim_b)) in ext_a.iter().zip(ext_b.iter()).enumerate() {
        if (dim_a != dim_b) && (*dim_a != 1) && (*dim_b != 1) {
            panic!("Can't broadcast tensors with shapes {shape_a:?} and {shape_b:?}");
        }
        shape_out[i] = *dim_a.max(dim_b);
    }
    shape_out
}

pub fn broadcast_strides(
    orig_shape: &[usize],
    orig_strides: &[usize],
    target_shape: &[usize],
) -> Vec<usize> {
    assert!(
        target_shape.len() >= orig_shape.len(),
        "Not broadcastable: {orig_shape:?} -> {target_shape:?}"
    );
    let offset = target_shape.len() - orig_shape.len();
    let mut out = vec![0usize; target_shape.len()];
    for j in offset..target_shape.len() {
        let i = j - offset;
        if orig_shape[i] == target_shape[j] {
            out[j] = orig_strides[i];
        } else if orig_shape[i] == 1 {
            out[j] = 0;
        } else {
            panic!("Not broadcastable: {orig_shape:?} -> {target_shape:?}");
        }
    }
    out
}

/// matrix multiplication for [M, L] and [L, N] tensors exclusively
pub fn matmul_2d(a: &TensorData, b: &TensorData) -> TensorData {
    assert!(
        a.ndim() == 2 && b.ndim() == 2,
        "expected input tensors to be 2D, got shapes {:?} and {:?}",
        a.shape,
        b.shape
    );
    assert_eq!(
        a.shape[1], b.shape[0],
        "can't multiply matrices with shapes {:?} and {:?}",
        a.shape, b.shape
    );
    let m = a.shape[0];
    let (k, n) = (b.shape[0], b.shape[1]);
    let mut out_storage: Vec<f64> = vec![0.; m * n];
    for i in 0..m {
        for p in 0..k {
            for j in 0..n {
                out_storage[i * n + j] += a.get(&[i, p]) * b.get(&[p, j]);
            }
        }
    }
    TensorData::new(out_storage, vec![m, n])
}
