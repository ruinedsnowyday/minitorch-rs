use std::simd::Simd;

use rayon::{
    iter::{
        IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
        ParallelIterator,
    },
    slice::{ParallelSlice, ParallelSliceMut},
};

use crate::{
    simd_ops::{BinaryOp, UnaryOp},
    tensor_data::{TensorData, offset_at},
    tensor_ops::{TensorOps, broadcast_shape},
};

/// Tensor operations optimized for multi-core execution, SIMD, and limited allocation
/// per memory access
pub struct FastOps;

impl TensorOps for FastOps {
    fn map_op<Op: UnaryOp, const N: usize>(input: &TensorData) -> TensorData {
        let offset = input.storage_offset;
        let size = input.size();
        let tail = size % N;
        let body = size - tail;
        let storage = if input.is_packed() {
            let simd_slice = &input.storage[offset..offset + body];
            let tail_slice = &input.storage[offset + body..offset + size];

            let mut out: Vec<f64> = Vec::with_capacity(size);
            out.spare_capacity_mut()
                .par_chunks_exact_mut(N)
                .zip(simd_slice.par_chunks_exact(N))
                .for_each(|(dst, src)| {
                    dst.write_copy_of_slice(
                        &Op::simd(Simd::<f64, N>::from_slice(src)).to_array(),
                    );
                });
            // SAFETY: write into uninitialized memory. safe, since
            // 1. simd_slice has size of (size / N) * N (size / N chunks of size N)
            // and par_chunks_exact_mut(N) over spare capacity gives size / N chunks
            // of size N as well.
            // 2. f64 is not a niche type: any combination of 64 bits yields a valid
            //    f64 and we never read slots before writing to them
            unsafe {
                out.set_len(body);
            }
            out.spare_capacity_mut()
                .iter_mut()
                .zip(tail_slice)
                .for_each(|(dst, &src)| {
                    dst.write(Op::scalar(src));
                });
            // SAFETY: write into uninitialized memory. safe, since
            // 1. Similarly, tail slice has size % N elements and spare capacity has
            //    size - (size / N) * N = size % N elements disjoint from the written
            //    ones. Thus, no uninitialized but safely accessible memory is left
            //    in the out vector after two writes and two changes of length
            // 2. f64 is not a niche type: any combination of 64 bits yields a valid
            //    f64 and we never read slots before writing to them
            unsafe {
                out.set_len(size);
            };
            out
        } else {
            let data: &[f64] = &input.storage[..];
            let offset = input.storage_offset;
            let shape: &[usize] = &input.shape[..];
            let strides: &[usize] = &input.strides[..];
            (0..size)
                .into_par_iter()
                .map(|idx| Op::scalar(data[offset_at(idx, offset, shape, strides)]))
                .collect()
        };
        TensorData::new(storage, input.shape.clone())
    }

    fn zip_op<Op: BinaryOp, const N: usize>(
        a: &TensorData,
        b: &TensorData,
    ) -> TensorData {
        let out_shape = broadcast_shape(&a.shape, &b.shape);
        let a_view = a.broadcast_to(&out_shape);
        let size = a_view.size();
        let a_offset = a_view.storage_offset;
        let b_view = b.broadcast_to(&out_shape);
        let b_offset = b_view.storage_offset;
        let a_data: &[f64];
        let b_data: &[f64];
        let tail = size % N;
        let body = size - tail;
        match (a_view.is_packed(), b_view.is_packed()) {
            (true, true) => {
                a_data = &a_view.storage[a_offset..a_offset + size];
                b_data = &b_view.storage[b_offset..b_offset + size];
                let (a_body, a_tail) = a_data.split_at(body);
                let (b_body, b_tail) = b_data.split_at(body);
                let mut out: Vec<f64> = Vec::with_capacity(size);
                out.spare_capacity_mut()
                    .par_chunks_exact_mut(N)
                    .zip(a_body.par_chunks_exact(N).zip(b_body.par_chunks_exact(N)))
                    .for_each(|(dst, (a_src, b_src))| {
                        dst.write_copy_of_slice(
                            &Op::simd(
                                Simd::<f64, N>::from_slice(a_src),
                                Simd::<f64, N>::from_slice(b_src),
                            )
                            .to_array(),
                        );
                    });
                // SAFETY: write into uninitialized memory. safe, since
                // 1. both a_body and b_body have size of (size / N) * N (size
                //    / N chunks of size N) and par_chunks_exact_mut(N) over spare
                //    capacity gives at least size / N chunks of size N (since
                //    allocation can give more capacity than asked for)
                // 2. f64 is not a niche type: any combination of 64 bits yields
                //    a valid f64 and we never read slots before writing to them
                unsafe {
                    out.set_len(body);
                }
                out.spare_capacity_mut()
                    .iter_mut()
                    .zip(a_tail.iter().zip(b_tail))
                    .for_each(|(dst, (&a_src, &b_src))| {
                        dst.write(Op::scalar(a_src, b_src));
                    });
                // SAFETY: write into uninitialized memory. safe, since
                // 1. Similarly, tail slices have size % N elements and spare capacity
                //    has at least  size - (size / N) * N = size % N elements disjoint
                //    from the written ones. Thus, no uninitialized but safely
                //    accessible memory is left in the out vector after two writes and
                //    two changes of length
                // 2. f64 is not a niche type: any combination of 64 bits yields
                //    a valid f64 and we never read slots before writing to them
                unsafe {
                    out.set_len(size);
                };
                TensorData::new(out, out_shape)
            }
            _ => FastOps::zip(a, b, Op::scalar),
        }
    }

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
        f: impl Fn(f64, f64) -> f64 + Send + Sync,
    ) -> TensorData {
        let out_shape = broadcast_shape(&a.shape, &b.shape);
        let a_view = a.broadcast_to(&out_shape);
        let size = a_view.size();
        let a_offset = a_view.storage_offset;
        let b_view = b.broadcast_to(&out_shape);
        let b_offset = b_view.storage_offset;
        let a_data: &[f64];
        let b_data: &[f64];
        let storage = match (a_view.is_packed(), b_view.is_packed()) {
            (true, true) => {
                a_data = &a_view.storage[a_offset..a_offset + size];
                b_data = &b_view.storage[b_offset..b_offset + size];
                a_data
                    .par_iter()
                    .zip(b_data)
                    .map(|(&a_val, &b_val)| f(a_val, b_val))
                    .collect()
            }
            (true, false) => {
                a_data = &a_view.storage[a_offset..a_offset + size];
                b_data = &b_view.storage[..];
                let b_shape: &[usize] = &b_view.shape[..];
                let b_strides: &[usize] = &b_view.strides[..];
                (0..size)
                    .into_par_iter()
                    .map(|p| {
                        f(
                            a_data[p],
                            b_data[offset_at(p, b_offset, b_shape, b_strides)],
                        )
                    })
                    .collect()
            }
            (false, true) => {
                b_data = &b_view.storage[b_offset..b_offset + size];
                a_data = &a_view.storage[..];
                let a_shape: &[usize] = &a_view.shape[..];
                let a_strides: &[usize] = &a_view.strides[..];
                (0..size)
                    .into_par_iter()
                    .map(|p| {
                        f(
                            a_data[offset_at(p, a_offset, a_shape, a_strides)],
                            b_data[p],
                        )
                    })
                    .collect()
            }
            (false, false) => {
                a_data = &a_view.storage[..];
                b_data = &b_view.storage[..];
                let a_shape: &[usize] = &a_view.shape[..];
                let b_shape: &[usize] = &b_view.shape[..];
                let a_strides: &[usize] = &a_view.strides[..];
                let b_strides: &[usize] = &b_view.strides[..];
                (0..size)
                    .into_par_iter()
                    .map(|p| {
                        f(
                            a_data[offset_at(p, a_offset, a_shape, a_strides)],
                            b_data[offset_at(p, b_offset, b_shape, b_strides)],
                        )
                    })
                    .collect()
            }
        };
        TensorData::new(storage, out_shape)
    }

    fn reduce(
        input: &TensorData,
        f: impl Fn(f64, f64) -> f64 + Send + Sync,
        init: f64,
        dim: usize,
        keep_dims: bool,
    ) -> TensorData {
        assert!(
            dim < input.shape.len(),
            "Can't reduce along axis {} for tensor with dimensions {:?}",
            dim,
            input.shape
        );
        let mut target_shape: Vec<usize> = input
            .shape
            .iter()
            .enumerate()
            .map(|(idx, &len)| if idx != dim { len } else { 1 })
            .collect();
        let target_size = target_shape.iter().product();
        let offset = input.storage_offset;
        let stride = input.strides[dim];
        let dim_size = input.shape[dim];
        let mut skipped_shape = input.shape.clone();
        skipped_shape.remove(dim);
        let mut skipped_strides = input.strides.clone();
        skipped_strides.remove(dim);
        let data: &[f64] = &input.storage[..];
        let out_storage = (0..target_size)
            .into_par_iter()
            .map(|p_out| {
                let base = offset_at(p_out, offset, &skipped_shape, &skipped_strides);
                let mut val = init;
                for k in 0..dim_size {
                    val = f(val, data[base + k * stride]);
                }
                val
            })
            .collect();
        if !keep_dims {
            target_shape = skipped_shape;
        }
        TensorData::new(out_storage, target_shape)
    }

    fn matmul(a: &TensorData, b: &TensorData) -> TensorData {
        batched_matmul(a, b)
    }
}

/// Performs batch-row-parallel matrix multiplication between two tensors with shapes
/// [.., M, K] and [.., K, N] where batch shapes are broadcastable. Allocated tensors
/// contiguously before matmul, thus using additional memory in strided cases
pub fn batched_matmul(a: &TensorData, b: &TensorData) -> TensorData {
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
    let mut a_broadcasted_shape = out_batch_shape.clone();
    a_broadcasted_shape.extend_from_slice(&[m, k]);
    let mut b_broadcasted_shape = out_batch_shape.clone();
    b_broadcasted_shape.extend_from_slice(&[k, n]);
    let a_view = a.broadcast_to(&a_broadcasted_shape);
    let b_view = b.broadcast_to(&b_broadcasted_shape);

    let a_holder: TensorData;
    let b_holder: TensorData;
    let a_storage = if a_view.is_packed() {
        &a_view.storage[a_view.storage_offset..a_view.storage_offset + a_view.size()]
    } else {
        a_holder = a_view.contiguous();
        &a_holder.storage[..]
    };
    let b_storage = if b_view.is_packed() {
        &b_view.storage[b_view.storage_offset..b_view.storage_offset + b_view.size()]
    } else {
        b_holder = b_view.contiguous();
        &b_holder.storage[..]
    };
    let out_size = out_shape.iter().product();
    let mut out_storage = vec![0.; out_size];

    out_storage
        .par_chunks_mut(n)
        .enumerate()
        .for_each(|(g, out_row)| {
            let batch = g / m;
            let i = g % m;
            for p in 0..k {
                let a_ip = a_storage[m * k * batch + i * k + p];
                for j in 0..n {
                    out_row[j] += a_ip * b_storage[k * n * batch + p * n + j];
                }
            }
        });
    TensorData::new(out_storage, out_shape)
}
