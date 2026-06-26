pub const LANES: usize = 8;

use crate::operators::{self, ALPHA, EPS};
use std::{
    ops::{Add, Div, Mul, Neg, Sub},
    simd::{
        Select, Simd, StdFloat,
        cmp::{SimdPartialEq, SimdPartialOrd},
        num::SimdFloat,
    },
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

pub struct SimdSigmoid;

impl UnaryOp for SimdSigmoid {
    fn scalar(a: f64) -> f64 {
        operators::sigmoid(a)
    }

    fn simd<const N: usize>(a: Simd<f64, N>) -> Simd<f64, N> {
        let positive_mask = a.simd_gt(Simd::splat(0.));
        let pos_branch = Simd::splat(1.).div(Simd::splat(1.).add(a.neg().exp()));
        let neg_branch = a.exp().div(Simd::splat(1.).add(a.exp()));
        positive_mask.select(pos_branch, neg_branch)
    }
}

pub struct SimdLeakyReLU;

impl UnaryOp for SimdLeakyReLU {
    fn scalar(a: f64) -> f64 {
        operators::leaky_relu(a)
    }

    fn simd<const N: usize>(a: Simd<f64, N>) -> Simd<f64, N> {
        let positive_mask = a.simd_gt(Simd::splat(0.));
        positive_mask.select(a, a.mul(Simd::splat(ALPHA)))
    }
}

pub struct SimdLog;

impl UnaryOp for SimdLog {
    fn scalar(a: f64) -> f64 {
        operators::log(a)
    }

    fn simd<const N: usize>(a: Simd<f64, N>) -> Simd<f64, N> {
        a.add(Simd::splat(EPS)).ln()
    }
}

pub struct SimdExp;

impl UnaryOp for SimdExp {
    fn scalar(a: f64) -> f64 {
        operators::exp(a)
    }

    fn simd<const N: usize>(a: Simd<f64, N>) -> Simd<f64, N> {
        a.exp()
    }
}

pub struct SimdInv;

impl UnaryOp for SimdInv {
    fn scalar(a: f64) -> f64 {
        operators::inv(a)
    }

    fn simd<const N: usize>(a: Simd<f64, N>) -> Simd<f64, N> {
        a.recip()
    }
}

pub struct SimdMul;

impl BinaryOp for SimdMul {
    fn scalar(a: f64, b: f64) -> f64 {
        operators::mul(a, b)
    }

    fn simd<const N: usize>(a: Simd<f64, N>, b: Simd<f64, N>) -> Simd<f64, N> {
        a.mul(b)
    }
}

pub struct SimdAdd;

impl BinaryOp for SimdAdd {
    fn scalar(a: f64, b: f64) -> f64 {
        operators::add(a, b)
    }

    fn simd<const N: usize>(a: Simd<f64, N>, b: Simd<f64, N>) -> Simd<f64, N> {
        a.add(b)
    }
}

pub struct SimdLt;

impl BinaryOp for SimdLt {
    fn scalar(a: f64, b: f64) -> f64 {
        operators::lt(a, b)
    }

    fn simd<const N: usize>(a: Simd<f64, N>, b: Simd<f64, N>) -> Simd<f64, N> {
        a.simd_lt(b).select(Simd::splat(1.), Simd::splat(0.))
    }
}

pub struct SimdEq;

impl BinaryOp for SimdEq {
    fn scalar(a: f64, b: f64) -> f64 {
        operators::eq(a, b)
    }

    fn simd<const N: usize>(a: Simd<f64, N>, b: Simd<f64, N>) -> Simd<f64, N> {
        a.simd_eq(b).select(Simd::splat(1.), Simd::splat(0.))
    }
}

pub struct SimdMax;

impl BinaryOp for SimdMax {
    fn scalar(a: f64, b: f64) -> f64 {
        operators::max(a, b)
    }

    fn simd<const N: usize>(a: Simd<f64, N>, b: Simd<f64, N>) -> Simd<f64, N> {
        a.simd_max(b)
    }
}

pub struct SimdLogBack;

impl BinaryOp for SimdLogBack {
    fn scalar(a: f64, d: f64) -> f64 {
        operators::log_back(a, d)
    }

    fn simd<const N: usize>(a: Simd<f64, N>, d: Simd<f64, N>) -> Simd<f64, N> {
        a.add(Simd::splat(EPS)).recip().mul(d)
    }
}

pub struct SimdInvBack;

impl BinaryOp for SimdInvBack {
    fn scalar(a: f64, d: f64) -> f64 {
        operators::inv_back(a, d)
    }

    fn simd<const N: usize>(a: Simd<f64, N>, d: Simd<f64, N>) -> Simd<f64, N> {
        let recip = a.recip();
        recip.mul(recip).neg().mul(d)
    }
}

pub struct SimdReLUBack;

impl BinaryOp for SimdReLUBack {
    fn scalar(a: f64, d: f64) -> f64 {
        operators::relu_back(a, d)
    }

    fn simd<const N: usize>(a: Simd<f64, N>, d: Simd<f64, N>) -> Simd<f64, N> {
        let positive_mask = a.simd_gt(Simd::splat(0.));
        positive_mask.select(d, Simd::splat(0.))
    }
}

pub struct SimdLeakyReLUBack;

impl BinaryOp for SimdLeakyReLUBack {
    fn scalar(a: f64, d: f64) -> f64 {
        operators::leaky_relu_back(a, d)
    }

    fn simd<const N: usize>(a: Simd<f64, N>, d: Simd<f64, N>) -> Simd<f64, N> {
        let positive_mask = a.simd_gt(Simd::splat(0.));
        positive_mask.select(d, d.mul(Simd::splat(ALPHA)))
    }
}

pub struct SimdSigmoidBack;

impl BinaryOp for SimdSigmoidBack {
    fn scalar(out: f64, d: f64) -> f64 {
        operators::sigmoid_back(out, d)
    }

    fn simd<const N: usize>(out: Simd<f64, N>, d: Simd<f64, N>) -> Simd<f64, N> {
        Simd::splat(1.).sub(out).mul(out).mul(d)
    }
}
