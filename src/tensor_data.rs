use std::rc::Rc;

use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::tensor_ops::broadcast_strides;

#[derive(Clone)]
pub struct TensorData {
    pub storage: Rc<Vec<f64>>,
    pub storage_offset: usize,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
}

impl TensorData {
    /// requires the product of all elements in shape to be equal to the length of
    /// storage
    pub fn new(storage: Vec<f64>, shape: Vec<usize>) -> Self {
        let prod = shape.iter().product();
        assert_eq!(storage.len(), prod);
        let strides = contiguous_strides(&shape);
        TensorData {
            storage: Rc::new(storage),
            storage_offset: 0,
            shape,
            strides,
        }
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let prod = shape.iter().product();
        Self::new(vec![0.; prod], shape)
    }

    pub fn ones(shape: Vec<usize>) -> Self {
        let prod = shape.iter().product();
        Self::new(vec![1.; prod], shape)
    }

    pub fn flat_index(&self, idx: &[usize]) -> usize {
        self.storage_offset
            + self
                .strides
                .iter()
                .zip(idx.iter())
                .map(|(&x, &y)| x * y)
                .sum::<usize>()
    }

    pub fn get(&self, idx: &[usize]) -> f64 {
        self.storage[self.flat_index(idx)]
    }

    pub fn set(&mut self, idx: &[usize], val: f64) {
        let flat_idx = self.flat_index(idx);
        Rc::make_mut(&mut self.storage)[flat_idx] = val;
    }

    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn permute(&self, axes: &[usize]) -> Self {
        assert_eq!(axes.len(), self.ndim());
        let new_shape: Vec<usize> = axes.iter().map(|&a| self.shape[a]).collect();
        let new_strides: Vec<usize> = axes.iter().map(|&a| self.strides[a]).collect();
        Self {
            storage: Rc::clone(&self.storage),
            storage_offset: self.storage_offset,
            shape: new_shape,
            strides: new_strides,
        }
    }

    pub fn permute_last_two(&self) -> Self {
        let n = self.ndim();
        let mut axes: Vec<usize> = (0..n).collect();
        axes.swap(n - 1, n - 2);
        self.permute(&axes)
    }

    pub fn iter_indices(&self) -> impl Iterator<Item = Vec<usize>> {
        iter_indices_of_shape(&self.shape)
    }

    pub fn broadcast_to(&self, target_shape: &[usize]) -> TensorData {
        let strides = broadcast_strides(&self.shape, &self.strides, target_shape);
        TensorData {
            storage: self.storage.clone(),
            storage_offset: self.storage_offset,
            shape: target_shape.to_vec(),
            strides,
        }
    }

    /// Constructs a 2D view of the trailing matrix dims rooted at `base_offset`.
    /// Used by batched matmul and any op that views per-batch matrices.
    pub fn view_last_two(&self, base_offset: usize) -> Self {
        assert!(
            self.ndim() >= 2,
            "to construct a 2D view, number of dimensions should be at least 2, got tensor with shape {:?}",
            self.shape
        );
        let m = self.shape[self.ndim() - 2];
        let k = self.shape[self.ndim() - 1];
        TensorData {
            storage: Rc::clone(&self.storage),
            storage_offset: base_offset,
            shape: vec![m, k],
            strides: self.strides[self.ndim() - 2..].to_vec(),
        }
    }

    /// Checks if the tensor is stored contiguously, starts from the beginning of the
    /// allocation, and the allocation is fully owned by the tensor
    pub fn is_contiguous(&self) -> bool {
        self.strides == contiguous_strides(&self.shape)
            && self.storage_offset == 0
            && self.storage.len() == self.size()
    }

    /// Checks if the strides of the tensor are the same as contiguous strides, i.e.
    /// it is just stored contiguously in a row-major order
    pub fn is_packed(&self) -> bool {
        self.strides == contiguous_strides(&self.shape)
    }

    /// Produces a new TensorData with storage being a contiguous array.
    /// Short-circuits if the array is already contiguous
    pub fn contiguous(&self) -> Self {
        if self.is_contiguous() {
            return self.clone();
        }
        let size = self.size();
        let contiguous_storage = (0..size)
            .into_iter()
            .map(|idx| self.storage[self.offset_at(idx)])
            .collect();

        Self::new(contiguous_storage, self.shape.clone())
    }

    /// Calculates the offset of the element in storage given the ordinal. Used as a
    /// slow path in cases when the tensor is stored non-contiguously. Panics if
    /// one of the shape dimensions is 0
    pub fn offset_at(&self, p: usize) -> usize {
        offset_at(p, self.storage_offset, &self.shape, &self.strides)
    }
}

pub fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut out: Vec<usize> = vec![1; shape.len()];
    for (i, dim) in shape.iter().enumerate().skip(1).rev() {
        out[i - 1] = out[i] * dim;
    }
    out
}

/// Given the shape, strides, and storage offset, computes the offset of the given
/// ordinal in arbitrary storage. Panics if one of the shape dimensions is 0
pub fn offset_at(
    p: usize,
    offset: usize,
    shape: &[usize],
    strides: &[usize],
) -> usize {
    let mut out: usize = 0;
    let mut rem = p;
    for (&stride, &shape) in strides.iter().zip(shape).rev() {
        out += (rem % shape) * stride;
        rem /= shape;
    }
    out + offset
}

/// Iterates logical indices in row-major (lexicographic) order — the
/// same order in which a contiguous tensor stores them. So pushing
/// values into a Vec<f64> in this order produces a correctly-laid-out
/// contiguous storage for the given shape.
pub fn iter_indices_of_shape(shape: &[usize]) -> impl Iterator<Item = Vec<usize>> {
    let count = shape.iter().product();
    let ndim = shape.len();
    let strides = contiguous_strides(shape);
    (0..count).map(move |x| {
        let mut out = vec![0; ndim];
        let mut rem = x;
        for (i, &stride) in strides.iter().enumerate() {
            out[i] = rem / stride;
            rem %= stride;
        }
        out
    })
}
