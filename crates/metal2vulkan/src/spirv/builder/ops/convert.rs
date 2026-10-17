//! Conversion constructors: numeric conversions, pointer/integer conversions, and bitcasts.

use crate::spirv::builder::function::FunctionBuilder;
use crate::spirv::builder::ops::err;
use crate::spirv::error::{BResult, OperandRule};
use crate::spirv::ids::{TypeId, Value};
use crate::spirv::ir::{Contraction, ConvOp, Inst, StorageClass, Type};

/// A bitcast endpoint: a numeric (non-`bool`) scalar or vector, or a `PhysicalStorageBuffer`
/// pointer, with its total bit width.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Cast {
    Num { bits: u32, int: bool, n: u8 },
    Ptr,
}

impl FunctionBuilder<'_> {
    /// A conversion of `a` to type `to`. The target type is the only input a conversion cannot
    /// derive; it is checked against the opcode:
    ///
    /// - `ConvertFToU`/`ConvertFToS`: float to unsigned (resp. any) integer of the same component
    ///   count; `ConvertSToF`/`ConvertUToF`: integer to float.
    /// - `UConvert` (unsigned result), `SConvert`, `FConvert`: same class and component count, a
    ///   different width.
    /// - `ConvertPtrToU`: `PhysicalStorageBuffer` pointer to a 64-bit integer scalar;
    ///   `ConvertUToPtr`: 64-bit integer scalar to such a pointer.
    /// - `Bitcast`: numeric scalars/vectors or `PhysicalStorageBuffer` pointers of equal total
    ///   width; a pointer converts only to another such pointer, a 64-bit integer, or a vector of
    ///   two 32-bit integers.
    pub fn convert(&mut self, op: ConvOp, a: Value, to: TypeId) -> BResult<Value> {
        self.convert_with(op, a, to, Contraction::Allowed)
    }

    /// An `FConvert` of `a` to type `to` decorated `NoContraction`: a conversion the device must
    /// perform as written. Vulkan assumes `AllowTransform` on a float operation not so decorated,
    /// under which a driver may drop a conversion to 16 bits that another widens back (Mesa's NIR
    /// folds `f2f32(f2f16(x))` to `x`).
    pub fn convert_no_contraction(&mut self, a: Value, to: TypeId) -> BResult<Value> {
        self.convert_with(ConvOp::FConvert, a, to, Contraction::Forbidden)
    }

    fn convert_with(
        &mut self,
        op: ConvOp,
        a: Value,
        to: TypeId,
        contract: Contraction,
    ) -> BResult<Value> {
        let o = op.to_spirv();
        let result = err(o, OperandRule::ResultType);
        match op {
            ConvOp::ConvertFToU
            | ConvOp::ConvertFToS
            | ConvOp::ConvertSToF
            | ConvOp::ConvertUToF
            | ConvOp::UConvert
            | ConvOp::SConvert
            | ConvOp::FConvert => {
                let from_float = matches!(
                    op,
                    ConvOp::ConvertFToU | ConvOp::ConvertFToS | ConvOp::FConvert
                );
                let x = self.want(
                    o,
                    0,
                    a,
                    |n| {
                        if from_float {
                            n.is_float()
                        } else {
                            n.is_int()
                        }
                    },
                )?;
                let t = self.num_of(to)?.ok_or(result.clone())?;
                let ok = t.n == x.n
                    && match op {
                        ConvOp::ConvertFToU => t.is_uint(),
                        ConvOp::ConvertFToS => t.is_int(),
                        ConvOp::ConvertSToF | ConvOp::ConvertUToF => t.is_float(),
                        ConvOp::UConvert => t.is_uint() && t.width != x.width,
                        ConvOp::SConvert => t.is_int() && t.width != x.width,
                        ConvOp::FConvert => t.is_float() && t.width != x.width,
                        _ => false,
                    };
                if !ok {
                    return Err(result);
                }
            }
            ConvOp::ConvertUToPtr => {
                let x = self.want(o, 0, a, |n| n.is_int() && n.is_scalar())?;
                if x.width != 64 {
                    return Err(err(o, OperandRule::Width(0)));
                }
                if !self.is_psb_ptr(to)? {
                    return Err(result);
                }
            }
            ConvOp::Bitcast => {
                let from = self
                    .cast_of(self.value_ty(a)?)?
                    .ok_or(err(o, OperandRule::Category(0)))?;
                let into = self.cast_of(to)?.ok_or(result.clone())?;
                let ok = match (from, into) {
                    (Cast::Ptr, Cast::Ptr) => true,
                    (Cast::Ptr, Cast::Num { bits, int, n })
                    | (Cast::Num { bits, int, n }, Cast::Ptr) => int && bits == 64 && n <= 2,
                    (Cast::Num { bits: x, .. }, Cast::Num { bits: y, .. }) => x == y,
                };
                if !ok {
                    return Err(result);
                }
            }
        }
        self.emit(Inst::Convert { op, a, contract }, to)
    }

    fn is_psb_ptr(&self, ty: TypeId) -> BResult<bool> {
        Ok(matches!(
            self.mb().types().get(ty)?,
            Type::Pointer {
                class: StorageClass::PhysicalStorageBuffer,
                ..
            }
        ))
    }

    fn cast_of(&self, ty: TypeId) -> BResult<Option<Cast>> {
        if self.is_psb_ptr(ty)? {
            return Ok(Some(Cast::Ptr));
        }
        Ok(match self.num_of(ty)? {
            Some(x) if !x.is_bool() => Some(Cast::Num {
                bits: u32::from(x.width) * u32::from(x.n),
                int: x.is_int(),
                n: x.n,
            }),
            _ => None,
        })
    }
}
