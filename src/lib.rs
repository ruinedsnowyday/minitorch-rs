// Opt the whole crate into the unstable `portable_simd` feature (`std::simd`),
// used by the SIMD backend in fast_ops.rs. `#![...]` is an *inner* attribute: it
// applies to the item it's written *inside* — here the crate root — vs `#[...]`
// (outer) which applies to the next item below it. Requires nightly; see
// rust-toolchain.toml.
#![feature(portable_simd)]

pub mod autodiff;
pub mod datasets;
pub mod fast_ops;
pub mod loss;
pub mod module;
pub mod nn;
pub mod operators;
pub mod simd_ops;
pub mod tensor;
pub mod tensor_autodiff;
pub mod tensor_data;
pub mod tensor_ops;
pub mod train;
pub mod xor_network;

pub type Backend = crate::fast_ops::FastOps;
