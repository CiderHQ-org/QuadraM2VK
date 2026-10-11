//! Ray queries (`SPV_KHR_ray_query`): a query variable, its initialization over an acceleration
//! structure, traversal (`OpRayQueryProceedKHR`), committing a triangle candidate, and the values
//! of the committed or candidate intersection.
//!
//! The query is a `Function`- or `Private`-storage `OpTypeRayQueryKHR` variable, which the type
//! rules keep out of loads, stores, and aggregates; every instruction here takes its pointer.

use crate::spirv::builder::function::FunctionBuilder;
use crate::spirv::builder::ops::err;
use crate::spirv::error::{BResult, OperandRule};
use crate::spirv::ids::{TypeId, Value};
use crate::spirv::ir::{Inst, RayQueryInst, RayQueryValue, StorageClass, Type};

/// The operands of `OpRayQueryInitializeKHR` after the query: operand positions 1 to 7.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RayInit {
    /// An `OpTypeAccelerationStructureKHR` value.
    pub accel: Value,
    /// `u32` ray flags.
    pub flags: Value,
    /// `u32` cull mask (its low 8 bits are compared with each instance's mask).
    pub cull_mask: Value,
    /// `vec3<f32>`.
    pub origin: Value,
    /// `f32`.
    pub tmin: Value,
    /// `vec3<f32>`.
    pub direction: Value,
    /// `f32`.
    pub tmax: Value,
}

impl FunctionBuilder<'_> {
    /// A fresh `Function` variable holding a ray query; returns its pointer.
    pub fn ray_query(&mut self) -> BResult<Value> {
        let ty = self.module().ty(Type::RayQuery)?;
        let var = self.local_var(ty, None)?;
        self.var_ptr(var)
    }

    /// Operand `i` (`query`) must point at a ray query.
    fn query_operand(&self, op: crate::spv::Op, i: u8, query: Value) -> BResult<()> {
        let (class, pointee, _) = self
            .pointer(op, query)
            .map_err(|_| err(op, OperandRule::Category(i)))?;
        if !matches!(class, StorageClass::Function | StorageClass::Private) {
            return Err(err(op, OperandRule::Class(class)));
        }
        match self.mb().types().get(pointee)? {
            Type::RayQuery => Ok(()),
            _ => Err(err(op, OperandRule::Category(i))),
        }
    }

    /// `OpRayQueryInitializeKHR`: start traversing `ray.accel` (an acceleration structure value)
    /// along `ray.origin + t * ray.direction`, `ray.tmin <= t <= ray.tmax` (`vec3<f32>` and
    /// `f32`), with the `u32` ray flags and cull mask.
    pub fn ray_query_initialize(&mut self, query: Value, ray: RayInit) -> BResult<()> {
        const OP: crate::spv::Op = crate::spv::Op::RayQueryInitializeKHR;
        self.query_operand(OP, 0, query)?;
        if !matches!(
            self.mb().types().get(self.value_ty(ray.accel)?)?,
            Type::AccelerationStructure
        ) {
            return Err(err(OP, OperandRule::Category(1)));
        }
        let word = |n: &super::Num| n.is_int() && n.is_scalar() && n.width == 32;
        self.want(OP, 2, ray.flags, word)?;
        self.want(OP, 3, ray.cull_mask, word)?;
        let vec3 = |n: &super::Num| n.is_float() && n.width == 32 && n.n == 3;
        let f32s = |n: &super::Num| n.is_float() && n.width == 32 && n.is_scalar();
        self.want(OP, 4, ray.origin, vec3)?;
        self.want(OP, 5, ray.tmin, f32s)?;
        self.want(OP, 6, ray.direction, vec3)?;
        self.want(OP, 7, ray.tmax, f32s)?;
        self.emit_void(Inst::RayQuery(RayQueryInst::Initialize {
            query,
            accel: ray.accel,
            flags: ray.flags,
            cull_mask: ray.cull_mask,
            origin: ray.origin,
            tmin: ray.tmin,
            direction: ray.direction,
            tmax: ray.tmax,
        }))
    }

    /// `OpRayQueryProceedKHR`: `bool`, whether traversal stopped at a candidate the shader must
    /// decide on (`false` once traversal is complete).
    pub fn ray_query_proceed(&mut self, query: Value) -> BResult<Value> {
        const OP: crate::spv::Op = crate::spv::Op::RayQueryProceedKHR;
        self.query_operand(OP, 0, query)?;
        let ty = self.bool_ty(1)?;
        self.emit(Inst::RayQuery(RayQueryInst::Proceed { query }), ty)
    }

    /// `OpRayQueryConfirmIntersectionKHR`: commit the current candidate, which must be a triangle.
    pub fn ray_query_confirm(&mut self, query: Value) -> BResult<()> {
        const OP: crate::spv::Op = crate::spv::Op::RayQueryConfirmIntersectionKHR;
        self.query_operand(OP, 0, query)?;
        self.emit_void(Inst::RayQuery(RayQueryInst::Confirm { query }))
    }

    /// `OpRayQueryGetIntersection*KHR`: `value` of the committed (`committed`) or the candidate
    /// intersection, in the opcode's result type ([`RayQueryValue`]).
    pub fn ray_query_get(
        &mut self,
        query: Value,
        value: RayQueryValue,
        committed: bool,
    ) -> BResult<Value> {
        self.query_operand(value.to_spirv(), 0, query)?;
        let ty = self.ray_value_ty(value)?;
        let which = self.u32_const(u32::from(committed))?;
        self.emit(
            Inst::RayQuery(RayQueryInst::Get {
                query,
                value,
                committed,
                which,
            }),
            ty,
        )
    }

    /// The result type of `value`.
    fn ray_value_ty(&mut self, value: RayQueryValue) -> BResult<TypeId> {
        Ok(match value {
            RayQueryValue::Type
            | RayQueryValue::InstanceCustomIndex
            | RayQueryValue::InstanceId
            | RayQueryValue::GeometryIndex
            | RayQueryValue::PrimitiveIndex
            | RayQueryValue::InstanceShaderBindingTableRecordOffset => self.uint_ty(32)?,
            RayQueryValue::T => self.f32_ty()?,
            RayQueryValue::Barycentrics => {
                let f = self.f32_ty()?;
                self.vec_of(f, 2)?
            }
            RayQueryValue::FrontFace => self.bool_ty(1)?,
            RayQueryValue::ObjectToWorld | RayQueryValue::WorldToObject => {
                let f = self.f32_ty()?;
                let col = self.vec_of(f, 3)?;
                self.module().ty(Type::Matrix { col, cols: 4 })?
            }
        })
    }
}
