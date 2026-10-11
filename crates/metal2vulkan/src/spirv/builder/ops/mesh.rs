//! A mesh pipeline's instructions (`SPV_EXT_mesh_shader`): a mesh entry's output counts and a task
//! entry's launch of its mesh workgroups.

use crate::spirv::builder::function::FunctionBuilder;
use crate::spirv::error::{BResult, BuildError, GlobalRule};
use crate::spirv::ids::{GlobalId, Value};
use crate::spirv::ir::{Inst, StorageClass, Terminator};

impl FunctionBuilder<'_> {
    /// `OpSetMeshOutputsEXT`: the vertices and primitives this mesh workgroup outputs, each a
    /// 32-bit unsigned integer (a mesh entry only, checked at finalize).
    pub fn set_mesh_outputs(&mut self, vertices: Value, primitives: Value) -> BResult<()> {
        let op = crate::spv::Op::SetMeshOutputsEXT;
        for (i, v) in [(0, vertices), (1, primitives)] {
            self.want(op, i, v, |n| n.is_uint() && n.is_scalar() && n.width == 32)?;
        }
        self.emit_void(Inst::SetMeshOutputs {
            vertices,
            primitives,
        })
    }

    /// Terminate the open block with `OpEmitMeshTasksEXT`: launch `x * y * z` mesh workgroups,
    /// each a 32-bit unsigned integer, which read `payload`, a `TaskPayloadWorkgroupEXT`
    /// variable (a task entry's entry function only, checked at finalize).
    pub fn emit_mesh_tasks(
        &mut self,
        x: Value,
        y: Value,
        z: Value,
        payload: Option<GlobalId>,
    ) -> BResult<()> {
        let op = crate::spv::Op::EmitMeshTasksEXT;
        for (i, v) in [(0, x), (1, y), (2, z)] {
            self.want(op, i, v, |n| n.is_uint() && n.is_scalar() && n.width == 32)?;
        }
        let payload = match payload {
            Some(g) => {
                let global = self.module().global(g)?;
                if global.class != StorageClass::TaskPayloadWorkgroupEXT {
                    return Err(BuildError::Global(GlobalRule::Payload));
                }
                Some(global.value)
            }
            None => None,
        };
        self.terminate(None, Terminator::EmitMeshTasks { x, y, z, payload })
    }
}
