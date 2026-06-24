use crate::{
    tensor_data::{TensorData, iter_indices_of_shape, offset_at},
    tensor_ops::{TensorOps, broadcast_shape},
};

/// Tensor operations optimized for multi-core execution, SIMD, and limited allocation
/// per memory access
pub struct FastOps;

impl TensorOps for FastOps {
    fn map(input: &TensorData, f: impl Fn(f64) -> f64) -> TensorData {
        let offset = input.storage_offset;
        let size = input.size();
        let storage = if input.is_packed() {
            input.storage[offset..offset + size]
                .iter()
                .map(|&x| f(x))
                .collect()
        } else {
            (0..size)
                .map(|p| f(input.storage[input.offset_at(p)]))
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
        todo!()
    }
}
