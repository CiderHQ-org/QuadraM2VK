//! `OpUndef`, helper-invocation query, and fragment-interlock critical sections.

use crate::spirv::builder::function::FunctionBuilder;
use crate::spirv::builder::ops::err;
use crate::spirv::error::{BResult, OperandRule};
use crate::spirv::ids::{TypeId, Value};
use crate::spirv::ir::{Inst, Shape, Type};

impl FunctionBuilder<'_> {
    /// `OpUndef` of a data type of fixed size (no opaque value or logical pointer).
    pub fn undef(&mut self, ty: TypeId) -> BResult<Value> {
        let f = *self.mb().types().facts(ty)?;
        if !matches!(f.shape, Shape::Plain | Shape::Implicit | Shape::Explicit) || f.runtime_sized {
            return Err(err(crate::spv::Op::Undef, OperandRule::ResultType));
        }
        let v = self.emit(Inst::Undef, ty)?;
        self.module().undefs.insert(v);
        Ok(v)
    }

    /// Whether `v` is an `OpUndef` result.
    pub fn is_undef(&self, v: Value) -> bool {
        self.mb().undefs.contains(&v)
    }

    /// `OpIsHelperInvocationEXT` (fragment only, `SPV_EXT_demote_to_helper_invocation`): whether
    /// the invocation is a helper, a demoted one included.
    pub fn is_helper_invocation(&mut self) -> BResult<Value> {
        let bool_ = self.module().ty(Type::Bool)?;
        self.emit(Inst::IsHelperInvocation, bool_)
    }

    /// `OpBeginInvocationInterlockEXT` (fragment with an interlock mode only).
    pub fn begin_interlock(&mut self) -> BResult<()> {
        self.emit_void(Inst::BeginInterlock)
    }

    /// `OpEndInvocationInterlockEXT` (fragment with an interlock mode only).
    pub fn end_interlock(&mut self) -> BResult<()> {
        self.emit_void(Inst::EndInterlock)
    }
}
