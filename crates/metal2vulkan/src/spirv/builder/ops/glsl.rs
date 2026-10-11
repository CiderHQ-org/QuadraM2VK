//! `GLSL.std.450` extended instructions.
//!
//! [`FunctionBuilder::glsl`] checks each instruction's operand contract from the `GLSL.std.450`
//! specification: float-only operations take 16- or 32-bit floats, and 64-bit floats only where the
//! specification permits them (not the trigonometric, exponential, logarithmic, `Pow`, `Atan2`,
//! `Radians`, or `Degrees` families); `FindILsb`/`FindSMsb`/`FindUMsb` take 32-bit integers;
//! operands that share a type must be the same type. Errors are [`BuildError::ExtOperand`].

use crate::spirv::builder::function::FunctionBuilder;
use crate::spirv::builder::ops::{index, Num};
use crate::spirv::error::{BResult, BuildError, OperandRule};
use crate::spirv::ids::{TypeId, Value};
use crate::spirv::ir::{Contraction, GlslOp, Inst, StorageClass, Type};

impl FunctionBuilder<'_> {
    /// A `GLSL.std.450` instruction `op` on `args`, with its result type derived from the
    /// operands (e.g. `Length` gives the component type, `ModfStruct` a `struct { T, T }`,
    /// the packs a `u32`, the interpolations the pointee of their `Input` pointer).
    pub fn glsl(&mut self, op: GlslOp, args: &[Value]) -> BResult<Value> {
        self.glsl_with(op, args, Contraction::Allowed)
    }

    /// [`glsl`](Self::glsl) with contraction `contract`. Only `Fma` may be
    /// [`Contraction::Forbidden`], which the serializer decorates `NoContraction`: a device then
    /// keeps the instruction as written, never rewriting it into an operation it may fuse with
    /// another.
    pub fn glsl_with(
        &mut self,
        op: GlslOp,
        args: &[Value],
        contract: Contraction,
    ) -> BResult<Value> {
        let e = |rule| BuildError::ExtOperand { op, rule };
        if contract == Contraction::Forbidden && op != GlslOp::Fma {
            return Err(e(OperandRule::NoContraction));
        }
        let ty = match op {
            GlslOp::Round
            | GlslOp::RoundEven
            | GlslOp::Trunc
            | GlslOp::FAbs
            | GlslOp::FSign
            | GlslOp::Floor
            | GlslOp::Ceil
            | GlslOp::Fract
            | GlslOp::Sqrt
            | GlslOp::InverseSqrt
            | GlslOp::Normalize => self.glsl_same(op, args, 1, Num::is_float, true)?.ty,
            GlslOp::Radians
            | GlslOp::Degrees
            | GlslOp::Sin
            | GlslOp::Cos
            | GlslOp::Tan
            | GlslOp::Asin
            | GlslOp::Acos
            | GlslOp::Atan
            | GlslOp::Sinh
            | GlslOp::Cosh
            | GlslOp::Tanh
            | GlslOp::Asinh
            | GlslOp::Acosh
            | GlslOp::Atanh
            | GlslOp::Exp
            | GlslOp::Log
            | GlslOp::Exp2
            | GlslOp::Log2 => self.glsl_same(op, args, 1, Num::is_float, false)?.ty,
            GlslOp::Atan2 | GlslOp::Pow => self.glsl_same(op, args, 2, Num::is_float, false)?.ty,
            GlslOp::FMin
            | GlslOp::FMax
            | GlslOp::NMin
            | GlslOp::NMax
            | GlslOp::Step
            | GlslOp::Reflect => self.glsl_same(op, args, 2, Num::is_float, true)?.ty,
            GlslOp::FClamp
            | GlslOp::NClamp
            | GlslOp::FMix
            | GlslOp::SmoothStep
            | GlslOp::Fma
            | GlslOp::FaceForward => self.glsl_same(op, args, 3, Num::is_float, true)?.ty,
            GlslOp::SAbs | GlslOp::SSign => self.glsl_same(op, args, 1, Num::is_int, true)?.ty,
            GlslOp::FindILsb | GlslOp::FindSMsb | GlslOp::FindUMsb => {
                let x = self.glsl_same(op, args, 1, Num::is_int, true)?;
                if x.width != 32 {
                    return Err(e(OperandRule::Width(0)));
                }
                x.ty
            }
            GlslOp::UMin | GlslOp::UMax | GlslOp::SMin | GlslOp::SMax => {
                self.glsl_same(op, args, 2, Num::is_int, true)?.ty
            }
            GlslOp::UClamp | GlslOp::SClamp => self.glsl_same(op, args, 3, Num::is_int, true)?.ty,
            GlslOp::Length => self.glsl_same(op, args, 1, Num::is_float, true)?.elem,
            GlslOp::Distance => self.glsl_same(op, args, 2, Num::is_float, true)?.elem,
            GlslOp::Cross => {
                let x = self.glsl_same(op, args, 2, Num::is_float, true)?;
                if x.n != 3 {
                    return Err(e(OperandRule::Category(0)));
                }
                x.ty
            }
            GlslOp::Refract => {
                let [i, n, eta] = args else {
                    return Err(e(OperandRule::Arity));
                };
                let x = self.glsl_same(op, &[*i, *n], 2, Num::is_float, true)?;
                if self.value_ty(*eta)? != x.elem {
                    return Err(e(OperandRule::Mismatch(2)));
                }
                x.ty
            }
            GlslOp::Ldexp => {
                let [x, exp] = args else {
                    return Err(e(OperandRule::Arity));
                };
                let x = self.glsl_float(op, 0, *x, true)?;
                match self.num(*exp)? {
                    Some(n) if n.is_int() && n.width == 32 && n.n == x.n => {}
                    _ => return Err(e(OperandRule::Mismatch(1))),
                }
                x.ty
            }
            GlslOp::ModfStruct | GlslOp::FrexpStruct => {
                let [a] = args else {
                    return Err(e(OperandRule::Arity));
                };
                let x = self.glsl_float(op, 0, *a, true)?;
                let second = if op == GlslOp::ModfStruct {
                    x.ty
                } else {
                    let i32_ty = self.module().ty(Type::Int {
                        width: 32,
                        signed: true,
                    })?;
                    self.vec_of(i32_ty, x.n)?
                };
                self.module().ty(Type::Struct {
                    members: [x.ty, second].into(),
                    layout: None,
                })?
            }
            GlslOp::Determinant | GlslOp::MatrixInverse => {
                let [m] = args else {
                    return Err(e(OperandRule::Arity));
                };
                let ty = self.value_ty(*m)?;
                let (col, cols) = match self.mb().types().get(ty)? {
                    Type::Matrix { col, cols } => (*col, *cols),
                    _ => return Err(e(OperandRule::Category(0))),
                };
                let c = self.num_of(col)?.ok_or(e(OperandRule::Category(0)))?;
                if c.n != cols {
                    return Err(e(OperandRule::Category(0)));
                }
                if op == GlslOp::Determinant {
                    c.elem
                } else {
                    ty
                }
            }
            GlslOp::PackSnorm4x8
            | GlslOp::PackUnorm4x8
            | GlslOp::PackSnorm2x16
            | GlslOp::PackUnorm2x16
            | GlslOp::PackHalf2x16 => {
                let n = if matches!(op, GlslOp::PackSnorm4x8 | GlslOp::PackUnorm4x8) {
                    4
                } else {
                    2
                };
                let [v] = args else {
                    return Err(e(OperandRule::Arity));
                };
                match self.num(*v)? {
                    Some(x) if x.is_float() && x.width == 32 && x.n == n => {}
                    _ => return Err(e(OperandRule::Category(0))),
                }
                self.uint_ty(32)?
            }
            GlslOp::UnpackSnorm2x16
            | GlslOp::UnpackUnorm2x16
            | GlslOp::UnpackHalf2x16
            | GlslOp::UnpackSnorm4x8
            | GlslOp::UnpackUnorm4x8 => {
                let n = if matches!(op, GlslOp::UnpackSnorm4x8 | GlslOp::UnpackUnorm4x8) {
                    4
                } else {
                    2
                };
                let [v] = args else {
                    return Err(e(OperandRule::Arity));
                };
                match self.num(*v)? {
                    Some(x) if x.is_int() && x.width == 32 && x.is_scalar() => {}
                    _ => return Err(e(OperandRule::Category(0))),
                }
                let f = self.f32_ty()?;
                self.vec_of(f, n)?
            }
            GlslOp::InterpolateAtCentroid
            | GlslOp::InterpolateAtSample
            | GlslOp::InterpolateAtOffset => {
                let want = if op == GlslOp::InterpolateAtCentroid {
                    1
                } else {
                    2
                };
                if args.len() != want {
                    return Err(e(OperandRule::Arity));
                }
                let (ptr, rest) = (args.first().copied(), args.get(1).copied());
                let ptr = ptr.ok_or(e(OperandRule::Arity))?;
                let pointee = self.interpolant(op, ptr)?;
                if let Some(v) = rest {
                    let ok = match (op, self.num(v)?) {
                        (GlslOp::InterpolateAtSample, Some(n)) => {
                            n.is_int() && n.width == 32 && n.is_scalar()
                        }
                        (_, Some(n)) => n.is_float() && n.width == 32 && n.n == 2,
                        _ => false,
                    };
                    if !ok {
                        return Err(e(OperandRule::Category(1)));
                    }
                }
                pointee
            }
        };
        self.emit(
            Inst::Glsl {
                op,
                args: args.into(),
                contract,
            },
            ty,
        )
    }

    /// Operand `i` must be a float scalar or vector of width 16 or 32 (or 64 when `f64`).
    fn glsl_float(&self, op: GlslOp, i: u8, v: Value, f64: bool) -> BResult<Num> {
        let e = |rule| BuildError::ExtOperand { op, rule };
        let x = self
            .num(v)?
            .filter(Num::is_float)
            .ok_or(e(OperandRule::Category(i)))?;
        if !(matches!(x.width, 16 | 32) || (f64 && x.width == 64)) {
            return Err(e(OperandRule::Width(i)));
        }
        Ok(x)
    }

    /// `n` operands of one type satisfying `class` (float widths per [`Self::glsl_float`]).
    fn glsl_same(
        &self,
        op: GlslOp,
        args: &[Value],
        n: usize,
        class: fn(&Num) -> bool,
        f64: bool,
    ) -> BResult<Num> {
        let e = |rule| BuildError::ExtOperand { op, rule };
        if args.len() != n {
            return Err(e(OperandRule::Arity));
        }
        let first = *args.first().ok_or(e(OperandRule::Arity))?;
        let x = self
            .num(first)?
            .filter(class)
            .ok_or(e(OperandRule::Category(0)))?;
        if x.is_float() {
            self.glsl_float(op, 0, first, f64)?;
        }
        for (i, a) in args.iter().enumerate().skip(1) {
            if self.value_ty(*a)? != x.ty {
                return Err(e(OperandRule::Mismatch(index(i))));
            }
        }
        Ok(x)
    }

    /// The pointee of an interpolant: a pointer into `Input` storage to a 32-bit float scalar or
    /// vector.
    fn interpolant(&self, op: GlslOp, ptr: Value) -> BResult<TypeId> {
        let e = |rule| BuildError::ExtOperand { op, rule };
        match self.mb().types().get(self.value_ty(ptr)?)? {
            Type::Pointer {
                class: StorageClass::Input,
                pointee,
            } => match self.num_of(*pointee)? {
                Some(n) if n.is_float() && n.width == 32 => Ok(*pointee),
                _ => Err(e(OperandRule::Category(0))),
            },
            _ => Err(e(OperandRule::Category(0))),
        }
    }
}
