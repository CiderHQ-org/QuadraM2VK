//! A geometry entry's instructions: emitting a vertex and ending the output primitive.

use crate::spirv::builder::function::FunctionBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ir::Inst;

impl FunctionBuilder<'_> {
    /// `OpEmitVertex`: emit a vertex of the current output values (a geometry entry only, checked
    /// at finalize). Every output is undefined after it until it is stored again.
    pub fn emit_vertex(&mut self) -> BResult<()> {
        self.emit_void(Inst::EmitVertex)
    }

    /// `OpEndPrimitive`: end the output primitive (a geometry entry only, checked at finalize).
    pub fn end_primitive(&mut self) -> BResult<()> {
        self.emit_void(Inst::EndPrimitive)
    }
}
