use std::rc::Rc;

use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::{
    tensor_data::{TensorData, offset_at},
    tensor_ops::{TensorOps, broadcast_shape, broadcast_strides},
};

/// Tensor operations optimized for multi-core execution, SIMD, and limited allocation
/// per memory access
pub struct FastOps;

impl TensorOps for FastOps {
    fn map(input: &TensorData, f: impl Fn(f64) -> f64 + Send + Sync) -> TensorData {
        let offset = input.storage_offset;
        let size = input.size();
        let storage = if input.is_packed() {
            input.storage[offset..offset + size]
                .par_iter()
                .map(|&x| f(x))
                .collect()
        } else {
            let data: &[f64] = &input.storage[..];
            let offset = input.storage_offset;
            let shape: &[usize] = &input.shape[..];
            let strides: &[usize] = &input.strides[..];
            (0..size)
                .into_par_iter()
                .map(|idx| f(data[offset_at(idx, offset, shape, strides)]))
                .collect()
        };
        TensorData::new(storage, input.shape.clone())
    }

    fn zip(
        a: &TensorData,
        b: &TensorData,
        f: impl Fn(f64, f64) -> f64,
    ) -> TensorData {
        let out_shape = broadcast_shape(&a.shape, &b.shape);
        let a_view = a.broadcast_to(&out_shape);
        let size = a_view.size();
        let a_offset = a_view.storage_offset;
        let b_view = b.broadcast_to(&out_shape);
        let b_offset = b_view.storage_offset;
        let storage = if a_view.is_packed() && b_view.is_packed() {
            a_view.storage[a_offset..a_offset + size]
                .iter()
                .zip(&b_view.storage[b_offset..b_offset + size])
                .map(|(&a_val, &b_val)| f(a_val, b_val))
                .collect()
        } else {
            (0..size)
                .map(|p| {
                    f(
                        a_view.storage[a_view.offset_at(p)],
                        b_view.storage[b_view.offset_at(p)],
                    )
                })
                .collect()
        };
        TensorData::new(storage, out_shape)
    }

    fn reduce(
        input: &TensorData,
        f: impl Fn(f64, f64) -> f64,
        init: f64,
        dim: usize,
        keep_dims: bool,
    ) -> TensorData {
        assert!(
            dim < input.shape.len(),
            "Can't reduce along axis {} for tensor with dimensions {:?}",
            dim,
            &input.shape
        );
        let mut target_shape: Vec<usize> = input
            .shape
            .iter()
            .enumerate()
            .map(|(idx, &len)| if idx != dim { len } else { 1 })
            .collect();
        let target_size = target_shape.iter().product();
        let offset = input.storage_offset;
        let mut out_storage = Vec::with_capacity(target_size);
        let stride = input.strides[dim];
        let dim_size = input.shape[dim];
        let mut skipped_shape = input.shape.clone();
        skipped_shape.remove(dim);
        let mut skipped_strides = input.strides.clone();
        skipped_strides.remove(dim);
        for p_out in 0..target_size {
            let base = offset_at(p_out, offset, &skipped_shape, &skipped_strides);
            let mut val = init;
            for k in 0..dim_size {
                val = f(val, input.storage[base + k * stride]);
            }
            out_storage.push(val);
        }
        if !keep_dims {
            target_shape = skipped_shape;
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
        let out_size = out_shape.iter().product();
        let n_batches = out_batch_shape.iter().product();

        let mut out_storage: Vec<f64> = Vec::with_capacity(out_size);
        for batch in 0..n_batches {
            let a_off = offset_at(
                batch,
                a.storage_offset,
                &out_batch_shape,
                &a_batch_strides,
            );
            let b_off = offset_at(
                batch,
                b.storage_offset,
                &out_batch_shape,
                &b_batch_strides,
            );

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

/// FastOps implementation of [M, K] x [K, N] matrix multiplication
fn matmul_2d(a: &TensorData, b: &TensorData) -> TensorData {
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
    let a_holder: TensorData;
    let b_holder: TensorData;
    let a_storage = if a.is_packed() {
        &a.storage[a.storage_offset..a.storage_offset + m * k]
    } else {
        a_holder = a.contiguous();
        &a_holder.storage[..]
    };
    let b_storage = if b.is_packed() {
        &b.storage[b.storage_offset..b.storage_offset + k * n]
    } else {
        b_holder = b.contiguous();
        &b_holder.storage[..]
    };
    let mut out_storage: Vec<f64> = vec![0.; m * n];
    for i in 0..m {
        for p in 0..k {
            let a_ip = a_storage[i * k + p];
            for j in 0..n {
                out_storage[i * n + j] += a_ip * b_storage[p * n + j];
            }
        }
    }
    TensorData::new(out_storage, vec![m, n])
}
