//! Simdgroup matrices:
//! `air.simdgroup_matrix_8x8_{load,store,init_diag,init_filled,multiply_accumulate}` and
//! `air.simdgroup_matrix_16x16x16_[widening_]multiply_accumulate`.
//!
//! An 8x8 matrix is a `<64 x T>` value, a 16x16x16 operand an `<8 x T>` per-thread fragment.
//! 8-bit float formats travel in `i8` lanes; the name spells the format. The 8x8 load and store
//! take a pointer and three `<2 x i64>` operands, the 16x16x16 products an `i1` after each of the
//! two factors; those positions are typed operands whose meaning lowering owns.

use super::sig::{Operand, Signature};
use super::ty::{Elem, NumTy, Sign};
use crate::ir::AddrSpace;

/// A matrix element format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MatElem {
    /// `f32`: `float` cells.
    F32,
    /// `f16`: `half` cells.
    F16,
    /// `bf16`: `bfloat` cells.
    BF16,
    /// 8-bit floats (carried in `i8` lanes).
    F8E4M3,
    /// `f8e4m3fn`: an 8-bit float with 4 exponent and 3 mantissa bits, finite, in `i8` lanes.
    F8E4M3Fn,
    /// `f8e5m2`: an 8-bit float with 5 exponent and 2 mantissa bits, in `i8` lanes.
    F8E5M2,
    /// 8-bit integers of a 16x16x16 widening product.
    I8(Sign),
    /// The 32-bit integer accumulator of a widening product.
    I32,
}

impl MatElem {
    /// The IR element that carries this format.
    pub const fn carrier(self) -> Elem {
        match self {
            MatElem::F32 => Elem::F32,
            MatElem::F16 => Elem::F16,
            MatElem::BF16 => Elem::BF16,
            MatElem::F8E4M3 | MatElem::F8E4M3Fn | MatElem::F8E5M2 | MatElem::I8(_) => Elem::I8,
            MatElem::I32 => Elem::I32,
        }
    }

    /// A float format token (`f32`, `f16`, `bf16`, `f8e4m3`, `f8e4m3fn`, `f8e5m2`).
    fn parse_float(token: &str) -> Option<MatElem> {
        Some(match token {
            "f32" => MatElem::F32,
            "f16" => MatElem::F16,
            "bf16" => MatElem::BF16,
            "f8e4m3" => MatElem::F8E4M3,
            "f8e4m3fn" => MatElem::F8E4M3Fn,
            "f8e5m2" => MatElem::F8E5M2,
            _ => return None,
        })
    }

    /// `v<lanes><float format>`.
    fn parse_vector(token: &str, lanes: u32) -> Option<MatElem> {
        let rest = token.strip_prefix(&format!("v{lanes}"))?;
        MatElem::parse_float(rest)
    }
}

/// A decoded simdgroup-matrix operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SimdMatrix {
    /// `<64 x T> (ptr, <2 x i64>, <2 x i64>, <2 x i64>)`.
    Load8x8 {
        /// The element format: `F16`, `F32`, or `BF16`.
        elem: MatElem,
        /// The pointer's address space: device or threadgroup.
        space: AddrSpace,
    },
    /// `void (<64 x T>, ptr, <2 x i64>, <2 x i64>, <2 x i64>)`.
    Store8x8 {
        /// The element format: `F16`, `F32`, or `BF16`.
        elem: MatElem,
        /// The pointer's address space: device or threadgroup.
        space: AddrSpace,
    },
    /// The AIR 2.7 form of [`SimdMatrix::Load8x8`], the same name declared
    /// `<64 x T> (ptr, i64, <2 x i64>, i1)`: the elements per row, the origin, and whether the
    /// matrix is read transposed (`xcrun metal` 32023.921 writes it for macOS 15 and earlier).
    /// Canonicalization rewrites it into the AIR 2.8 form.
    Load8x8Legacy {
        /// The element format: `F16`, `F32`, or `BF16`.
        elem: MatElem,
        /// The pointer's address space: device or threadgroup.
        space: AddrSpace,
    },
    /// The AIR 2.7 form of [`SimdMatrix::Store8x8`]:
    /// `void (<64 x T>, ptr, i64, <2 x i64>, i1)`.
    Store8x8Legacy {
        /// The element format: `F16`, `F32`, or `BF16`.
        elem: MatElem,
        /// The pointer's address space: device or threadgroup.
        space: AddrSpace,
    },
    /// `<64 x T> (T)`: a matrix with the value on its diagonal.
    InitDiag8x8 {
        /// The element format: `F16`, `F32`, or `BF16`.
        elem: MatElem,
    },
    /// `<64 x T> (T)`: a matrix with the value in every cell (`make_filled_simdgroup_matrix`).
    InitFilled8x8 {
        /// The element format: `F16`, `F32`, or `BF16`.
        elem: MatElem,
    },
    /// `<64 x R> (<64 x A>, <64 x B>, <64 x C>)`: `R = A * B + C`.
    MultiplyAccumulate8x8 {
        /// The format of `R`: `F16` or `F32`.
        result: MatElem,
        /// The format of `A`.
        lhs: MatElem,
        /// The format of `B`.
        rhs: MatElem,
        /// The format of `C`.
        acc: MatElem,
    },
    /// `<8 x R> (<8 x A>, i1, <8 x B>, i1, <8 x R>)`, `R` = `f32` (float) or `i32` (widening
    /// integer product).
    MultiplyAccumulate16 {
        /// The format of `R`, the result and the accumulator: `F32` or `I32`.
        result: MatElem,
        /// The format of `A`: a float format, or `I8` with its sign.
        lhs: MatElem,
        /// The format of `B`: a float format, or `I8` with its sign.
        rhs: MatElem,
    },
}

impl SimdMatrix {
    /// The AIR 2.7 operand form the same name may be declared with (a block load's or store's).
    pub(super) fn alternative(self) -> Option<SimdMatrix> {
        match self {
            SimdMatrix::Load8x8 { elem, space } => Some(SimdMatrix::Load8x8Legacy { elem, space }),
            SimdMatrix::Store8x8 { elem, space } => {
                Some(SimdMatrix::Store8x8Legacy { elem, space })
            }
            _ => None,
        }
    }

    /// The AIR 2.8 form of an AIR 2.7 block load or store.
    pub fn modern(self) -> Option<SimdMatrix> {
        match self {
            SimdMatrix::Load8x8Legacy { elem, space } => Some(SimdMatrix::Load8x8 { elem, space }),
            SimdMatrix::Store8x8Legacy { elem, space } => {
                Some(SimdMatrix::Store8x8 { elem, space })
            }
            _ => None,
        }
    }

    /// Parse the part of a name after `air.`.
    pub(super) fn parse(rest: &str) -> Option<SimdMatrix> {
        if let Some(r) = rest.strip_prefix("simdgroup_matrix_8x8_") {
            let t: Vec<&str> = r.split('.').collect();
            return match t[..] {
                [kind @ ("load" | "store"), m, p] => {
                    let elem = MatElem::parse_vector(m, 64)?;
                    matches!(elem, MatElem::F16 | MatElem::F32 | MatElem::BF16).then_some(())?;
                    // `p<space><pointee>`: the pointee must be the element (only the space is typed
                    // in IR).
                    let digits = p.strip_prefix('p')?;
                    let n = digits.bytes().take_while(u8::is_ascii_digit).count();
                    let (space, pointee) = digits.split_at_checked(n)?;
                    let space = match space {
                        "1" => AddrSpace::Device,
                        "3" => AddrSpace::Threadgroup,
                        _ => return None,
                    };
                    (MatElem::parse_float(pointee)? == elem).then_some(())?;
                    Some(if kind == "load" {
                        SimdMatrix::Load8x8 { elem, space }
                    } else {
                        SimdMatrix::Store8x8 { elem, space }
                    })
                }
                [kind @ ("init_diag" | "init_filled"), m, s] => {
                    let elem = MatElem::parse_vector(m, 64)?;
                    matches!(elem, MatElem::F16 | MatElem::F32 | MatElem::BF16).then_some(())?;
                    (MatElem::parse_float(s)? == elem).then_some(())?;
                    Some(if kind == "init_diag" {
                        SimdMatrix::InitDiag8x8 { elem }
                    } else {
                        SimdMatrix::InitFilled8x8 { elem }
                    })
                }
                ["multiply_accumulate", r, a, b, c] => {
                    let result = MatElem::parse_vector(r, 64)?;
                    matches!(result, MatElem::F16 | MatElem::F32).then_some(())?;
                    Some(SimdMatrix::MultiplyAccumulate8x8 {
                        result,
                        lhs: MatElem::parse_vector(a, 64)?,
                        rhs: MatElem::parse_vector(b, 64)?,
                        acc: MatElem::parse_vector(c, 64)?,
                    })
                }
                _ => None,
            };
        }
        let r = rest.strip_prefix("simdgroup_matrix_16x16x16_")?;
        let t: Vec<&str> = r.split('.').collect();
        match t[..] {
            ["multiply_accumulate", "f", "f", "v8f32", a, b, "v8f32"] => {
                Some(SimdMatrix::MultiplyAccumulate16 {
                    result: MatElem::F32,
                    lhs: MatElem::parse_vector(a, 8)?,
                    rhs: MatElem::parse_vector(b, 8)?,
                })
            }
            ["widening_multiply_accumulate", sa, sb, "v8i32", "v8i8", "v8i8", "v8i32"] => {
                Some(SimdMatrix::MultiplyAccumulate16 {
                    result: MatElem::I32,
                    lhs: MatElem::I8(Sign::parse(sa)?),
                    rhs: MatElem::I8(Sign::parse(sb)?),
                })
            }
            _ => None,
        }
    }

    /// The typed signature.
    pub fn signature(&self) -> Signature {
        let m = |e: MatElem, lanes| {
            Operand::Num(NumTy {
                elem: e.carrier(),
                lanes,
            })
        };
        let v2i64 = Operand::Num(NumTy {
            elem: Elem::I64,
            lanes: 2,
        });
        let flag = Operand::Num(NumTy::scalar(Elem::I1));
        match *self {
            SimdMatrix::Load8x8 { elem, space } => {
                Signature::new(m(elem, 64), &[Operand::Ptr(space), v2i64, v2i64, v2i64])
            }
            SimdMatrix::Store8x8 { elem, space } => Signature::new(
                Operand::Void,
                &[m(elem, 64), Operand::Ptr(space), v2i64, v2i64, v2i64],
            ),
            SimdMatrix::Load8x8Legacy { elem, space } => {
                let i64s = Operand::Num(NumTy::scalar(Elem::I64));
                Signature::new(m(elem, 64), &[Operand::Ptr(space), i64s, v2i64, flag])
            }
            SimdMatrix::Store8x8Legacy { elem, space } => {
                let i64s = Operand::Num(NumTy::scalar(Elem::I64));
                Signature::new(
                    Operand::Void,
                    &[m(elem, 64), Operand::Ptr(space), i64s, v2i64, flag],
                )
            }
            SimdMatrix::InitDiag8x8 { elem } | SimdMatrix::InitFilled8x8 { elem } => {
                Signature::new(m(elem, 64), &[m(elem, 1)])
            }
            SimdMatrix::MultiplyAccumulate8x8 {
                result,
                lhs,
                rhs,
                acc,
            } => Signature::new(m(result, 64), &[m(lhs, 64), m(rhs, 64), m(acc, 64)]),
            SimdMatrix::MultiplyAccumulate16 { result, lhs, rhs } => Signature::new(
                m(result, 8),
                &[m(lhs, 8), flag, m(rhs, 8), flag, m(result, 8)],
            ),
        }
    }
}
