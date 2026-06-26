use crate::operators;
use std::{
    ops::{Add, Mul, Neg},
    simd::{Simd, num::SimdFloat},
};

pub trait UnaryOp {
    fn scalar(a: f64) -> f64;
    fn simd<const N: usize>(a: Simd<f64, N>) -> Simd<f64, N>;
}

pub trait BinaryOp {
    fn scalar(a: f64, b: f64) -> f64;
    fn simd<const N: usize>(a: Simd<f64, N>, b: Simd<f64, N>) -> Simd<f64, N>;
}

pub struct SimdReLU;

impl UnaryOp for SimdReLU {
    fn scalar(a: f64) -> f64 {
        operators::relu(a)
    }

    fn simd<const N: usize>(a: Simd<f64, N>) -> Simd<f64, N> {
        a.simd_max(Simd::splat(0.))
    }
}

pub struct SimdNeg;

impl UnaryOp for SimdNeg {
    fn scalar(a: f64) -> f64 {
        operators::neg(a)
    }

    fn simd<const N: usize>(a: Simd<f64, N>) -> Simd<f64, N> {
        a.neg()
    }
}

pub struct SimdId;

impl UnaryOp for SimdId {
    fn scalar(a: f64) -> f64 {
        operators::id(a)
    }

    fn simd<const N: usize>(a: Simd<f64, N>) -> Simd<f64, N> {
        a
    }
}

pub struct SimdMul;

impl BinaryOp for SimdMul {
    fn scalar(a: f64, b: f64) -> f64 {
        a * b
    }

    fn simd<const N: usize>(a: Simd<f64, N>, b: Simd<f64, N>) -> Simd<f64, N> {
        a.mul(b)
    }
}

pub struct SimdAdd;

impl BinaryOp for SimdAdd {
    fn scalar(a: f64, b: f64) -> f64 {
        a + b
    }

    fn simd<const N: usize>(a: Simd<f64, N>, b: Simd<f64, N>) -> Simd<f64, N> {
        a.add(b)
    }
}
