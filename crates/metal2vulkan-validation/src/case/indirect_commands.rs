//! Descriptor literals shared by native indirect-command creation and record replay.

use metal2vulkan::bind::indirect_commands::{
    Action, Bindings, BufferBinding, BufferStage, CommandRecord, Inheritance, StateSource,
    BINDING_COUNT, COMMAND_BUFFER_HEADER_BYTES, COMMAND_BYTES,
};
use serde::{Deserialize, Serialize};

/// An immutable authored pipeline template exposed through an argument-buffer handle.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PipelineResource {
    pub buffer_binding: u32,
    pub field_offset: u32,
    pub token: u64,
    /// Exact source identity also locates the template's authored-case shard.
    pub air_sha256: String,
    pub case_id: String,
}

impl PipelineResource {
    /// Resolve an immutable template without visiting unrelated case or source shards.
    /// This verifies manifest identity; executable checking is still required.
    pub fn resolve_template(
        &self,
        store: &crate::store::CorpusStore,
    ) -> Result<super::AuthoredCase, String> {
        resolve_template(store, &self.air_sha256, &self.case_id)
    }
}

/// The command buffers of `resources` that execute, in the order both executors run them: by
/// `(buffer_binding, field_offset)`, the order a case's identity sorts its command buffers in, so a
/// listing order the identity does not record never changes what runs.
pub fn executed_in_key_order(resources: &[CommandBufferResource]) -> Vec<&CommandBufferResource> {
    let mut executed = resources
        .iter()
        .filter(|resource| resource.execute.is_some())
        .collect::<Vec<_>>();
    executed.sort_by_key(|resource| (resource.buffer_binding, resource.field_offset));
    executed
}

/// The authored case `case_id` of AIR `air_sha256`, a pipeline template, read from its own
/// case shard only.
pub fn resolve_template(
    store: &crate::store::CorpusStore,
    air_sha256: &str,
    case_id: &str,
) -> Result<super::AuthoredCase, String> {
    store
        .find_cases_for_air(air_sha256)?
        .into_iter()
        .find(|case| case.case_id == case_id)
        .ok_or_else(|| format!("pipeline template {case_id} does not exist for AIR {air_sha256}"))
}

/// What a command buffer allows, fixed when it is made (Metal's
/// `MTLIndirectCommandBufferDescriptor`, as far as the record format needs it): a compute
/// buffer's dispatch kinds, kernel-buffer and threadgroup-memory binding counts and dynamic
/// strides, or a render buffer's draw kinds, vertex, fragment, object and mesh binding counts and
/// dynamic strides. The case's own schema: [`product`](Self::product) is the record format's
/// descriptor, which initializes, decodes and guards the records.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum CommandDescriptor {
    Compute {
        dispatch: ComputeDispatch,
        max_buffer_bind_count: u32,
        max_threadgroup_memory_bind_count: u32,
        support_dynamic_attribute_stride: bool,
    },
    Render {
        /// The draws the buffer allows (Metal's `Draw`, `DrawIndexed`, `DrawPatches` and
        /// `DrawIndexedPatches` command types). Absent, draws of primitives alone, the one kind
        /// previously, so the cases written before it keep their identities.
        #[serde(default, skip_serializing_if = "RenderDraws::is_primitives_alone")]
        draws: RenderDraws,
        max_vertex_buffer_bind_count: u32,
        max_fragment_buffer_bind_count: u32,
        /// The object buffers a command binds (`maxObjectBufferBindCount`), which a mesh draw's
        /// object function reads. Absent, none, the one count previously, so the cases written
        /// before it keep their identities.
        #[serde(default, skip_serializing_if = "is_zero")]
        max_object_buffer_bind_count: u32,
        /// The mesh buffers a command binds (`maxMeshBufferBindCount`), as the object buffers.
        #[serde(default, skip_serializing_if = "is_zero")]
        max_mesh_buffer_bind_count: u32,
        support_dynamic_attribute_stride: bool,
    },
}

const fn is_zero(value: &u32) -> bool {
    *value == 0
}

/// The draws a render descriptor allows: Metal's `Draw` (draws of primitives), `DrawIndexed`
/// (indexed draws), `DrawPatches` (patch draws), `DrawIndexedPatches` (indexed patch draws),
/// `DrawMeshThreadgroups` and `DrawMeshThreads` (mesh draws of a grid of threadgroups or of
/// threads, which only a kernel encodes in a case), at least one of them. The default, and what
/// an absent field means, is draws of primitives alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RenderDraws {
    pub primitives: bool,
    pub indexed: bool,
    pub patches: bool,
    /// Indexed patch draws, which only a kernel encodes in a case. Absent, none, the one value
    /// previously, so the cases written before it keep their identities.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub indexed_patches: bool,
    /// Absent, none, as indexed patch draws.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub mesh_threadgroups: bool,
    /// Absent, none, as indexed patch draws.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub mesh_threads: bool,
}

impl Default for RenderDraws {
    fn default() -> RenderDraws {
        RenderDraws {
            primitives: true,
            indexed: false,
            patches: false,
            indexed_patches: false,
            mesh_threadgroups: false,
            mesh_threads: false,
        }
    }
}

impl RenderDraws {
    /// Whether these are draws of primitives alone, the default an absent field means.
    pub fn is_primitives_alone(&self) -> bool {
        *self == RenderDraws::default()
    }

    /// The record format's draws of the same kinds.
    pub fn product(self) -> metal2vulkan::bind::indirect_commands::RenderDraws {
        metal2vulkan::bind::indirect_commands::RenderDraws {
            primitives: self.primitives,
            indexed: self.indexed,
            patches: self.patches,
            indexed_patches: self.indexed_patches,
            mesh_threadgroups: self.mesh_threadgroups,
            mesh_threads: self.mesh_threads,
        }
    }

    /// Whether these allow a mesh draw of either grid form.
    pub fn mesh(self) -> bool {
        self.mesh_threadgroups || self.mesh_threads
    }

    /// Whether a host command's `action` is a draw these allow (or none).
    fn allow(self, action: Option<HostAction>) -> bool {
        match action {
            None => true,
            Some(HostAction::DrawPrimitives { .. }) => self.primitives,
            Some(HostAction::DrawIndexedPrimitives { .. }) => self.indexed,
            Some(HostAction::DrawPatches { .. }) => self.patches,
            Some(HostAction::DispatchThreadgroups { .. } | HostAction::DispatchThreads { .. }) => {
                false
            }
        }
    }
}

/// The dispatches a compute descriptor allows: Metal's `ConcurrentDispatch` (threadgroups),
/// `ConcurrentDispatchThreads` (threads), or both.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComputeDispatch {
    Threadgroups,
    Threads,
    Both,
}

impl CommandDescriptor {
    /// The record format's descriptor of the same allowances.
    pub fn product(&self) -> metal2vulkan::bind::indirect_commands::CommandDescriptor {
        use metal2vulkan::bind::indirect_commands::{CommandDescriptor as D, ComputeDispatch as C};
        match *self {
            Self::Compute {
                dispatch,
                max_buffer_bind_count,
                max_threadgroup_memory_bind_count,
                support_dynamic_attribute_stride,
            } => D::Compute {
                dispatch: match dispatch {
                    ComputeDispatch::Threadgroups => C::Threadgroups,
                    ComputeDispatch::Threads => C::Threads,
                    ComputeDispatch::Both => C::Both,
                },
                max_buffer_bind_count,
                max_threadgroup_memory_bind_count,
                support_dynamic_attribute_stride,
            },
            Self::Render {
                draws,
                max_vertex_buffer_bind_count,
                max_fragment_buffer_bind_count,
                max_object_buffer_bind_count,
                max_mesh_buffer_bind_count,
                support_dynamic_attribute_stride,
            } => D::Render {
                draws: draws.product(),
                max_vertex_buffer_bind_count,
                max_fragment_buffer_bind_count,
                max_object_buffer_bind_count,
                max_mesh_buffer_bind_count,
                support_dynamic_attribute_stride,
            },
        }
    }
}

/// One command-buffer handle stored in an argument-buffer field.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CommandBufferResource {
    pub buffer_binding: u32,
    pub field_offset: u32,
    pub capacity: u32,
    /// Half-open command indices, independent of the storage capacity.
    pub execution_range: [u32; 2],
    pub inherit_pipeline_state: bool,
    pub inherit_buffers: bool,
    pub descriptor: CommandDescriptor,
    /// Commands the host encodes before the producer runs, in index order: the state of each, which
    /// the producer's writes then change.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub commands: Vec<HostCommand>,
    /// Execute `execution_range` after the producer completes, in a second encoder of the
    /// descriptor's family with this parent state: a compute encoder for compute commands, a render
    /// encoder for render commands. Absent, the commands are produced and checked, never executed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execute: Option<Execution>,
}

/// The most commands one execution may run: each is a dispatch or a draw both executors replay one
/// at a time.
pub const MAX_EXECUTED_COMMANDS: u32 = 64;

/// The alignment of a host-encoded buffer offset: Metal's for a constant buffer, so any template
/// parameter may read it.
pub const COMMAND_OFFSET_ALIGNMENT: u32 = 256;

/// A pipeline template the host registers by its token without placing it in an argument-buffer
/// field: an execution's parent pipeline, a compute template for compute commands and a render
/// template for render commands.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PipelineTemplate {
    pub token: u64,
    pub air_sha256: String,
    pub case_id: String,
}

/// One buffer binding the host sets: the case's Metal buffer `buffer` at byte `offset`, at buffer
/// index `index` of the stage the list names. An `attribute_stride` is a host command's kernel or
/// vertex buffer bound by the stride form (`setKernelBuffer:offset:attributeStride:atIndex:`,
/// `setVertexBuffer:offset:attributeStride:atIndex:`), which a descriptor supporting dynamic
/// attribute strides admits; a stage-in layout of `MTLBufferLayoutStrideDynamic` reads its buffer
/// at it. Without one the binding is the plain form.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CommandBinding {
    pub index: u32,
    pub buffer: u32,
    #[serde(default)]
    pub offset: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attribute_stride: Option<u32>,
}

/// A stage-in region (`setStageInRegion:`): its origin and size, the region a kernel's
/// `[[stage_in]]` fetch, `[[grid_origin]]` and `[[grid_size]]` read; a host compute command's
/// ([`HostCommand::stage_in_region`]) or a case's dispatch's (`Dispatch::stage_in_region`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HostStageInRegion {
    pub origin: [u32; 3],
    pub size: [u32; 3],
}

/// A threadgroup-memory length the host sets at index `index`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThreadgroupLength {
    pub index: u32,
    pub length: u32,
}

/// The action a host-encoded command records: a compute command's dispatch or a render command's
/// draw.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum HostAction {
    /// `concurrentDispatchThreadgroups:threadsPerThreadgroup:`.
    DispatchThreadgroups {
        grid: [u32; 3],
        threads_per_threadgroup: [u32; 3],
    },
    /// `concurrentDispatchThreads:threadsPerThreadgroup:`.
    DispatchThreads {
        grid: [u32; 3],
        threads_per_threadgroup: [u32; 3],
    },
    /// `drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:`.
    DrawPrimitives {
        primitive: crate::case::Primitive,
        vertex_start: u32,
        vertex_count: u32,
        instance_count: u32,
        base_instance: u32,
    },
    /// `drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:instanceCount:
    /// baseVertex:baseInstance:`: `index_count` indices of `index_type` from byte
    /// `index_buffer_offset` of the case's buffer `index_buffer` (by its Metal index), each plus
    /// `base_vertex`.
    DrawIndexedPrimitives {
        primitive: crate::case::Primitive,
        index_count: u32,
        index_type: crate::case::IndexType,
        index_buffer: u32,
        index_buffer_offset: u32,
        instance_count: u32,
        base_vertex: i32,
        base_instance: u32,
    },
    /// `drawPatches:patchStart:patchCount:patchIndexBuffer:patchIndexBufferOffset:instanceCount:
    /// baseInstance:tessellationFactorBuffer:tessellationFactorBufferOffset:
    /// tessellationFactorBufferInstanceStride:`: `patch_count` patches from `patch_start`, each
    /// patch's index the `uint` at its position in the case's buffer `patch_index_buffer` from byte
    /// `patch_index_buffer_offset` where the draw names one, `instance_count` times from
    /// `base_instance`, the factors of the patch at position `p` at byte `factor_buffer_offset`
    /// plus `p` factor structures of the case's buffer `factor_buffer`, `factor_instance_stride`
    /// bytes apart per instance where the pipeline's step function reads per instance. The host's
    /// command has no factor scale: 1.
    DrawPatches {
        control_points: u32,
        patch_start: u32,
        patch_count: u32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        patch_index_buffer: Option<u32>,
        patch_index_buffer_offset: u32,
        instance_count: u32,
        base_instance: u32,
        factor_buffer: u32,
        factor_buffer_offset: u32,
        factor_instance_stride: u32,
    },
}

/// One command the host encodes at `index`: each field set is what the host's
/// `MTLIndirectComputeCommand` or `MTLIndirectRenderCommand` setter writes, and an absent one is
/// left as reset. `buffers` and `threadgroup_memory_lengths` are a compute command's,
/// `vertex_buffers` and `fragment_buffers` a render command's.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HostCommand {
    pub index: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub action: Option<HostAction>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub barrier: bool,
    /// A pipeline token of the case: a pipeline state's or an execution's parent pipeline's.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pipeline: Option<u64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub buffers: Vec<CommandBinding>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub threadgroup_memory_lengths: Vec<ThreadgroupLength>,
    /// A compute command's stage-in region.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stage_in_region: Option<HostStageInRegion>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub vertex_buffers: Vec<CommandBinding>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fragment_buffers: Vec<CommandBinding>,
}

/// The parent encoder's state of an execution: its pipeline, exactly when the commands inherit it,
/// and its buffers, only when they inherit buffers: kernel buffers for compute commands, vertex and
/// fragment buffers for render commands, and object and mesh buffers for the render commands of a
/// buffer that allows mesh draws.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Execution {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pipeline: Option<PipelineTemplate>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub buffers: Vec<CommandBinding>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub vertex_buffers: Vec<CommandBinding>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fragment_buffers: Vec<CommandBinding>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub object_buffers: Vec<CommandBinding>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub mesh_buffers: Vec<CommandBinding>,
}

/// The Metal primitive type of an authored primitive, as a draw's record holds it
/// (`Action::DrawPrimitives::primitive_type`): the one mapping, which the Metal oracle's draws and
/// the engine candidate's input assembly read too.
pub fn metal_primitive_type(
    primitive: crate::case::Primitive,
) -> metal2vulkan::metal::PrimitiveType {
    use crate::case::Primitive as P;
    use metal2vulkan::metal::PrimitiveType as M;
    match primitive {
        P::Point => M::Point,
        P::Line => M::Line,
        P::LineStrip => M::LineStrip,
        P::Triangle => M::Triangle,
        P::TriangleStrip => M::TriangleStrip,
    }
}

/// The Metal index type of an authored one (`MTLIndexType`), as an indexed draw's record holds it
/// (`Action::DrawIndexedPrimitives::index_type`).
pub fn metal_index_type(index_type: crate::case::IndexType) -> metal2vulkan::metal::IndexType {
    use crate::case::IndexType as I;
    use metal2vulkan::metal::IndexType as M;
    match index_type {
        I::Uint16 => M::UInt16,
        I::Uint32 => M::UInt32,
    }
}

/// The authored primitive of a record's Metal primitive type, `None` for a value that names none: a
/// draw the GPU encoded with such a type fails its replay.
pub fn authored_primitive(primitive_type: u32) -> Option<crate::case::Primitive> {
    use crate::case::Primitive as P;
    use metal2vulkan::metal::PrimitiveType as M;
    Some(match M::try_from(primitive_type).ok()? {
        M::Point => P::Point,
        M::Line => P::Line,
        M::LineStrip => P::LineStrip,
        M::Triangle => P::Triangle,
        M::TriangleStrip => P::TriangleStrip,
    })
}

impl HostAction {
    /// The action the host contract records, an indexed draw's index buffer and a patch draw's
    /// patch index and factor buffers at the address `address` gives each case buffer plus its
    /// offset (0 for no patch index buffer), a patch draw's factor scale 1.
    pub fn action(self, address: &impl Fn(u32) -> Option<u64>) -> Result<Action, String> {
        Ok(match self {
            Self::DispatchThreadgroups {
                grid,
                threads_per_threadgroup,
            } => Action::DispatchThreadgroups {
                grid,
                local_size: threads_per_threadgroup,
            },
            Self::DispatchThreads {
                grid,
                threads_per_threadgroup,
            } => Action::DispatchThreads {
                grid,
                local_size: threads_per_threadgroup,
            },
            Self::DrawPrimitives {
                primitive,
                vertex_start,
                vertex_count,
                instance_count,
                base_instance,
            } => Action::DrawPrimitives {
                primitive_type: metal_primitive_type(primitive).as_raw(),
                vertex_start,
                vertex_count,
                instance_count,
                base_instance,
            },
            Self::DrawIndexedPrimitives {
                primitive,
                index_count,
                index_type,
                index_buffer,
                index_buffer_offset,
                instance_count,
                base_vertex,
                base_instance,
            } => Action::DrawIndexedPrimitives {
                primitive_type: metal_primitive_type(primitive).as_raw(),
                index_count,
                index_type: metal_index_type(index_type).as_raw(),
                index_buffer: address(index_buffer)
                    .ok_or_else(|| format!("index buffer {index_buffer} has no address"))?
                    + u64::from(index_buffer_offset),
                instance_count,
                base_vertex,
                base_instance,
            },
            Self::DrawPatches {
                control_points,
                patch_start,
                patch_count,
                patch_index_buffer,
                patch_index_buffer_offset,
                instance_count,
                base_instance,
                factor_buffer,
                factor_buffer_offset,
                factor_instance_stride,
            } => {
                let at = |buffer: u32, offset: u32, what: &str| {
                    address(buffer)
                        .map(|address| address + u64::from(offset))
                        .ok_or_else(|| format!("{what} buffer {buffer} has no address"))
                };
                Action::DrawPatches {
                    control_points,
                    patch_start,
                    patch_count,
                    patch_index_buffer: patch_index_buffer
                        .map(|buffer| at(buffer, patch_index_buffer_offset, "patch index"))
                        .transpose()?
                        .unwrap_or(0),
                    instance_count,
                    base_instance,
                    factor_buffer: at(factor_buffer, factor_buffer_offset, "factor")?,
                    factor_instance_stride,
                    factor_scale: 1.0f32.to_bits(),
                }
            }
        })
    }

    /// Check a render command's draw against what Metal's validation stops, and an indexed
    /// draw's index buffer and a patch draw's patch index and factor buffers against the case's
    /// buffers (`lengths`, by Metal index): a draw of no instances, an indexed draw of no
    /// indices, a buffer that is not a case buffer, an index buffer offset that is not a multiple
    /// of the index size, indices past the buffer's end, a patch index buffer offset, factor
    /// offset or instance stride that is not a multiple of 4, and patch indices past their
    /// buffer's end (the runtime refuses each at its setter too). Which factors a patch draw reads
    /// depends on its pipeline's patch type, which the checker reads
    /// (`check::check_render_replay_templates`).
    fn check_draw(
        self,
        lengths: &std::collections::BTreeMap<u32, usize>,
        what: &str,
    ) -> Result<(), String> {
        match self {
            Self::DrawPrimitives {
                instance_count: 0, ..
            } => Err(format!("{what}: a draw of no instances")),
            Self::DrawIndexedPrimitives {
                index_count,
                index_type,
                index_buffer,
                index_buffer_offset,
                instance_count,
                ..
            } => {
                if index_count == 0 || instance_count == 0 {
                    return Err(format!(
                        "{what}: an indexed draw of no indices or instances"
                    ));
                }
                let length = lengths.get(&index_buffer).ok_or_else(|| {
                    format!("{what}: index buffer {index_buffer} is not a case buffer")
                })?;
                let size = u64::from(index_type.size());
                let offset = u64::from(index_buffer_offset);
                if offset % size != 0 || offset + u64::from(index_count) * size > *length as u64 {
                    return Err(format!(
                        "{what}: {index_count} indices from offset {offset} are not aligned \
                         inside index buffer {index_buffer}'s {length} bytes"
                    ));
                }
                Ok(())
            }
            Self::DrawPatches {
                patch_start,
                patch_count,
                patch_index_buffer,
                patch_index_buffer_offset,
                instance_count,
                factor_buffer,
                factor_buffer_offset,
                factor_instance_stride,
                ..
            } => {
                if instance_count == 0 {
                    return Err(format!("{what}: a patch draw of no instances"));
                }
                if let Some(buffer) = patch_index_buffer {
                    let length = lengths.get(&buffer).ok_or_else(|| {
                        format!("{what}: patch index buffer {buffer} is not a case buffer")
                    })?;
                    let offset = u64::from(patch_index_buffer_offset);
                    let indices = u64::from(patch_start) + u64::from(patch_count);
                    if offset % 4 != 0 || offset + 4 * indices > *length as u64 {
                        return Err(format!(
                            "{what}: {indices} patch indices from offset {offset} are not \
                             aligned inside patch index buffer {buffer}'s {length} bytes"
                        ));
                    }
                }
                if !lengths.contains_key(&factor_buffer) {
                    return Err(format!(
                        "{what}: factor buffer {factor_buffer} is not a case buffer"
                    ));
                }
                if factor_buffer_offset % 4 != 0 || factor_instance_stride % 4 != 0 {
                    return Err(format!(
                        "{what}: a factor offset or instance stride that is not a multiple of 4"
                    ));
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }
}

/// Check `bindings` (sorted by index, each below `count`, each a case buffer of `lengths` at an
/// aligned offset inside it) for `what`.
fn check_bindings(
    bindings: &[CommandBinding],
    count: u32,
    strides: bool,
    lengths: &std::collections::BTreeMap<u32, usize>,
    what: &str,
) -> Result<(), String> {
    let mut previous = None;
    for binding in bindings {
        if binding.attribute_stride.is_some() && !strides {
            return Err(format!(
                "{what}: buffer index {} binds an attribute stride, which this stage or \
                 descriptor does not admit",
                binding.index
            ));
        }
        if previous.is_some_and(|index| binding.index <= index) {
            return Err(format!("{what}: buffer indices must be sorted and unique"));
        }
        previous = Some(binding.index);
        if binding.index >= count {
            return Err(format!(
                "{what}: buffer index {} is not below {count}",
                binding.index
            ));
        }
        let length = lengths
            .get(&binding.buffer)
            .ok_or_else(|| format!("{what}: buffer {} is not a case buffer", binding.buffer))?;
        if binding.offset % COMMAND_OFFSET_ALIGNMENT != 0 || binding.offset as usize >= *length {
            return Err(format!(
                "{what}: offset {} of buffer {} is not a multiple of {COMMAND_OFFSET_ALIGNMENT} \
                 inside its {length} bytes",
                binding.offset, binding.buffer
            ));
        }
    }
    Ok(())
}

/// The `stage` buffer bindings of `bindings` with each address `address` gives its buffer.
fn bound(
    stage: BufferStage,
    bindings: &[CommandBinding],
    address: &impl Fn(u32) -> Option<u64>,
    into: &mut Bindings,
) -> Result<(), String> {
    for binding in bindings {
        let base = address(binding.buffer)
            .ok_or_else(|| format!("buffer {} has no address", binding.buffer))?;
        let set = BufferBinding {
            address: base + u64::from(binding.offset),
            attribute_stride: binding.attribute_stride.map(u64::from),
        };
        // A record's address below 2^32 is a nil buffer's offset: a case buffer there would bind
        // nothing.
        if !set.names_buffer() {
            return Err(format!(
                "buffer {} lies at {:#x}, below the addresses a record names a buffer at",
                binding.buffer, set.address
            ));
        }
        into.buffers_mut(stage)[binding.index as usize] = Some(set);
    }
    Ok(())
}

impl CommandBufferResource {
    /// Whether its commands are render commands, which run a render template, rather than compute
    /// commands, which run a compute template.
    pub fn is_render(&self) -> bool {
        match self.descriptor {
            CommandDescriptor::Render { .. } => true,
            CommandDescriptor::Compute { .. } => false,
        }
    }

    pub fn inheritance(&self) -> Inheritance {
        Inheritance {
            pipeline: if self.inherit_pipeline_state {
                StateSource::Parent
            } else {
                StateSource::Command
            },
            buffers: if self.inherit_buffers {
                StateSource::Parent
            } else {
                StateSource::Command
            },
        }
    }

    /// Initialize caller-owned candidate storage only after validating the authored contract.
    pub fn initialize_records(&self, bytes: &mut [u8]) -> Result<(), String> {
        self.record_storage_bytes()?;
        metal2vulkan::bind::indirect_commands::CommandBuffer::initialize(
            bytes,
            self.capacity,
            &self.descriptor.product(),
            self.inheritance(),
        )
        .map_err(|error| error.to_string())
    }

    /// Validate the completed producer buffer and all selected descriptors before exposing a command.
    /// Resource registration and pipeline/device checks are still required before submission.
    pub fn validated_records<'a>(
        &self,
        bytes: &'a [u8],
    ) -> Result<impl ExactSizeIterator<Item = CommandRecord> + 'a, String> {
        if bytes.len() != self.record_storage_bytes()? {
            return Err("indirect record storage does not match authored capacity".into());
        }
        let buffer = metal2vulkan::bind::indirect_commands::CommandBuffer::decode(
            bytes,
            &self.descriptor.product(),
            self.inheritance(),
        )
        .map_err(|error| error.to_string())?;
        let [start, end] = self.execution_range;
        buffer
            .records(start..end)
            .map_err(|error| error.to_string())
    }

    /// Check what the host encodes and executes against the descriptor, the case's buffers
    /// (`lengths`, by Metal index), and its pipeline tokens (`tokens`): each command inside the
    /// capacity, in index order, setting only what the descriptor's family admits (a compute
    /// command's dispatch, kernel buffers, and threadgroup lengths; a render command's draw and
    /// vertex and fragment buffers, and no barrier: Metal's render command has one, which neither
    /// executor encodes from the host yet) and nothing it inherits; an execution's parent pipeline
    /// exactly when the commands inherit it, its buffers (of the family's stages, object and mesh
    /// buffers where the descriptor allows mesh draws) only when they inherit buffers, and at most
    /// [`MAX_EXECUTED_COMMANDS`] commands. Which family each pipeline token's
    /// template is of, the checker decides (`check::resolve_pipeline_templates`).
    pub fn validate_host_state(
        &self,
        lengths: &std::collections::BTreeMap<u32, usize>,
        tokens: &std::collections::BTreeSet<u64>,
    ) -> Result<(), String> {
        if self.commands.is_empty() && self.execute.is_none() {
            return Ok(());
        }
        let mut previous = None;
        for command in &self.commands {
            let what = format!("host command {}", command.index);
            if previous.is_some_and(|index| command.index <= index) {
                return Err("host commands must be sorted by a unique index".into());
            }
            previous = Some(command.index);
            if command.index >= self.capacity {
                return Err(format!("{what} is not below the capacity"));
            }
            if let Some(token) = command.pipeline {
                if self.inherit_pipeline_state {
                    return Err(format!("{what} sets the pipeline it inherits"));
                }
                if !tokens.contains(&token) {
                    return Err(format!("{what}: pipeline token {token} is not the case's"));
                }
            }
            if self.inherit_buffers
                && !(command.buffers.is_empty()
                    && command.vertex_buffers.is_empty()
                    && command.fragment_buffers.is_empty())
            {
                return Err(format!("{what} sets the buffers it inherits"));
            }
            match self.descriptor {
                CommandDescriptor::Compute {
                    dispatch,
                    max_buffer_bind_count,
                    max_threadgroup_memory_bind_count,
                    support_dynamic_attribute_stride,
                } => {
                    match (command.action, dispatch) {
                        (None, _)
                        | (
                            Some(HostAction::DispatchThreadgroups { .. }),
                            ComputeDispatch::Threadgroups | ComputeDispatch::Both,
                        )
                        | (
                            Some(HostAction::DispatchThreads { .. }),
                            ComputeDispatch::Threads | ComputeDispatch::Both,
                        ) => {}
                        _ => return Err(format!("{what}: the descriptor admits no such dispatch")),
                    }
                    if !command.vertex_buffers.is_empty() || !command.fragment_buffers.is_empty() {
                        return Err(format!(
                            "{what}: a compute command binds no vertex or fragment buffer"
                        ));
                    }
                    check_bindings(
                        &command.buffers,
                        max_buffer_bind_count,
                        support_dynamic_attribute_stride,
                        lengths,
                        &what,
                    )?;
                    let mut previous = None;
                    for memory in &command.threadgroup_memory_lengths {
                        if previous.is_some_and(|index| memory.index <= index)
                            || memory.index >= max_threadgroup_memory_bind_count
                            || memory.length % 16 != 0
                        {
                            return Err(format!(
                                "{what}: threadgroup lengths must be sorted, unique, below \
                                 {max_threadgroup_memory_bind_count}, and multiples of 16"
                            ));
                        }
                        previous = Some(memory.index);
                    }
                }
                CommandDescriptor::Render {
                    draws,
                    max_vertex_buffer_bind_count,
                    max_fragment_buffer_bind_count,
                    support_dynamic_attribute_stride,
                    ..
                } => {
                    if command.stage_in_region.is_some() {
                        return Err(format!("{what}: a render command has no stage-in region"));
                    }
                    if matches!(
                        command.action,
                        Some(
                            HostAction::DispatchThreadgroups { .. }
                                | HostAction::DispatchThreads { .. }
                        )
                    ) {
                        return Err(format!("{what}: a render command draws, never dispatches"));
                    }
                    if !draws.allow(command.action) {
                        return Err(format!("{what}: the descriptor admits no such draw"));
                    }
                    if let Some(action) = command.action {
                        action.check_draw(lengths, &what)?;
                    }
                    if command.barrier && !draws.mesh() {
                        // Metal's validation stops it on a buffer without mesh draws ("setBarrier
                        // is not enabled on this indirect command buffer"), as the runtime refuses
                        // it.
                        return Err(format!(
                            "{what}: a host-encoded render command's barrier needs mesh draws"
                        ));
                    }
                    if command.barrier {
                        // Neither executor encodes one from the host yet.
                        return Err(format!(
                            "{what}: a host-encoded render command's barrier on a buffer of mesh \
                             draws is not modelled"
                        ));
                    }
                    if !command.buffers.is_empty() || !command.threadgroup_memory_lengths.is_empty()
                    {
                        return Err(format!(
                            "{what}: a render command binds no kernel buffer or threadgroup \
                             length"
                        ));
                    }
                    check_bindings(
                        &command.vertex_buffers,
                        max_vertex_buffer_bind_count,
                        support_dynamic_attribute_stride,
                        lengths,
                        &format!("{what}'s vertex buffers"),
                    )?;
                    check_bindings(
                        &command.fragment_buffers,
                        max_fragment_buffer_bind_count,
                        false,
                        lengths,
                        &format!("{what}'s fragment buffers"),
                    )?;
                }
            }
        }
        if let Some(execution) = &self.execute {
            let [start, end] = self.execution_range;
            let count = end
                .checked_sub(start)
                .ok_or("an execution's range is reversed")?;
            if count > MAX_EXECUTED_COMMANDS {
                return Err(format!(
                    "an execution runs at most {MAX_EXECUTED_COMMANDS} commands"
                ));
            }
            match (&execution.pipeline, self.inherit_pipeline_state) {
                (Some(template), true) if template.token != 0 => {}
                (None, false) => {}
                _ => {
                    return Err(
                        "an execution has a nonzero parent pipeline exactly when its commands \
                         inherit it"
                            .into(),
                    )
                }
            }
            if !self.inherit_buffers
                && !(execution.buffers.is_empty()
                    && execution.vertex_buffers.is_empty()
                    && execution.fragment_buffers.is_empty()
                    && execution.object_buffers.is_empty()
                    && execution.mesh_buffers.is_empty())
            {
                return Err("an execution's parent buffers need inherited buffers".into());
            }
            let (render, mesh) = match self.descriptor {
                CommandDescriptor::Render { draws, .. } => (true, draws.mesh()),
                CommandDescriptor::Compute { .. } => (false, false),
            };
            for (bindings, what, admitted) in [
                (
                    &execution.buffers,
                    "the parent encoder's kernel buffers",
                    !render,
                ),
                (
                    &execution.vertex_buffers,
                    "the parent encoder's vertex buffers",
                    render,
                ),
                (
                    &execution.fragment_buffers,
                    "the parent encoder's fragment buffers",
                    render,
                ),
                (
                    &execution.object_buffers,
                    "the parent encoder's object buffers",
                    mesh,
                ),
                (
                    &execution.mesh_buffers,
                    "the parent encoder's mesh buffers",
                    mesh,
                ),
            ] {
                if !admitted && !bindings.is_empty() {
                    return Err(format!(
                        "{what} are no stage of the commands the execution runs"
                    ));
                }
                // The parent encoder's attribute strides on inherited buffers are not measured.
                check_bindings(bindings, BINDING_COUNT as u32, false, lengths, what)?;
            }
        }
        Ok(())
    }

    /// Write the records the host encodes into the initialized record storage `bytes` (at least
    /// [`Self::record_storage_bytes`] long), each buffer at the address `address` gives its case
    /// buffer plus its offset.
    pub fn encode_host_records(
        &self,
        bytes: &mut [u8],
        address: impl Fn(u32) -> Option<u64>,
    ) -> Result<(), String> {
        let length = self.record_storage_bytes()?;
        let bytes = bytes
            .get_mut(..length)
            .ok_or("the record storage is shorter than the authored capacity")?;
        for (index, record) in self.host_records(address)? {
            let at = COMMAND_BUFFER_HEADER_BYTES + index as usize * COMMAND_BYTES;
            let slot = bytes
                .get_mut(at..at + COMMAND_BYTES)
                .ok_or("a host command is past its record storage")?;
            record
                .encode_into(slot)
                .map_err(|error| format!("host command {index}: {error}"))?;
        }
        Ok(())
    }

    /// The records the host encodes, by index, each buffer at the address `address` gives its case
    /// buffer plus its offset.
    fn host_records(
        &self,
        address: impl Fn(u32) -> Option<u64>,
    ) -> Result<Vec<(u32, CommandRecord)>, String> {
        self.commands
            .iter()
            .map(|command| {
                let mut record = CommandRecord {
                    action: command
                        .action
                        .map_or(Ok(Action::Empty), |action| action.action(&address))?,
                    barrier: command.barrier,
                    ..CommandRecord::default()
                };
                record.bindings.pipeline = command.pipeline.and_then(std::num::NonZeroU64::new);
                bound(
                    BufferStage::Kernel,
                    &command.buffers,
                    &address,
                    &mut record.bindings,
                )?;
                bound(
                    BufferStage::Vertex,
                    &command.vertex_buffers,
                    &address,
                    &mut record.bindings,
                )?;
                bound(
                    BufferStage::Fragment,
                    &command.fragment_buffers,
                    &address,
                    &mut record.bindings,
                )?;
                for memory in &command.threadgroup_memory_lengths {
                    record.threadgroup_memory_lengths[memory.index as usize] = memory.length;
                }
                if let Some(HostStageInRegion { origin, size }) = command.stage_in_region {
                    record.stage_in_region =
                        [origin[0], origin[1], origin[2], size[0], size[1], size[2]];
                }
                Ok((command.index, record))
            })
            .collect()
    }

    /// The parent encoder's state of the execution, each buffer at the address `address` gives its
    /// case buffer plus its offset; `None` when the commands are not executed.
    pub fn parent_bindings(
        &self,
        address: impl Fn(u32) -> Option<u64>,
    ) -> Result<Option<Bindings>, String> {
        let Some(execution) = &self.execute else {
            return Ok(None);
        };
        let mut parent = Bindings::default();
        parent.pipeline = execution
            .pipeline
            .as_ref()
            .and_then(|template| std::num::NonZeroU64::new(template.token));
        bound(
            BufferStage::Kernel,
            &execution.buffers,
            &address,
            &mut parent,
        )?;
        bound(
            BufferStage::Vertex,
            &execution.vertex_buffers,
            &address,
            &mut parent,
        )?;
        bound(
            BufferStage::Fragment,
            &execution.fragment_buffers,
            &address,
            &mut parent,
        )?;
        bound(
            BufferStage::Object,
            &execution.object_buffers,
            &address,
            &mut parent,
        )?;
        bound(
            BufferStage::Mesh,
            &execution.mesh_buffers,
            &address,
            &mut parent,
        )?;
        Ok(Some(parent))
    }

    /// Validate the literal and return its exact candidate allocation size without allocating.
    pub fn record_storage_bytes(&self) -> Result<usize, String> {
        self.descriptor.product().validate()?;
        let [start, end] = self.execution_range;
        if start > end || end > self.capacity {
            return Err("indirect execution range exceeds command capacity or is reversed".into());
        }
        self.field_offset
            .checked_add(8)
            .ok_or("indirect command-buffer handle field overflows its owner offset")?;
        usize::try_from(self.capacity)
            .ok()
            .and_then(|capacity| {
                capacity.checked_mul(metal2vulkan::bind::indirect_commands::COMMAND_BYTES)
            })
            .and_then(|bytes| {
                bytes
                    .checked_add(metal2vulkan::bind::indirect_commands::COMMAND_BUFFER_HEADER_BYTES)
            })
            .ok_or_else(|| "indirect command-buffer record storage size overflows".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use metal2vulkan::bind::indirect_commands::{Action, BufferStage};

    /// A buffer slot set at `address` (below 2^32 a nil buffer's offset), at `attribute_stride`.
    fn set(address: u64, attribute_stride: Option<u64>) -> Option<BufferBinding> {
        Some(BufferBinding {
            address,
            attribute_stride,
        })
    }

    #[test]
    fn descriptor_separates_command_families_and_checks_binding_boundaries() {
        let descriptor = CommandDescriptor::Compute {
            dispatch: ComputeDispatch::Threadgroups,
            max_buffer_bind_count: 2,
            max_threadgroup_memory_bind_count: 1,
            support_dynamic_attribute_stride: false,
        };
        let mut record = CommandRecord {
            action: Action::DispatchThreadgroups {
                grid: [1; 3],
                local_size: [1; 3],
            },
            ..CommandRecord::default()
        };
        record.bindings.buffers_mut(BufferStage::Kernel)[1] = set(64, None);
        record.threadgroup_memory_lengths[0] = 16;
        descriptor.product().validate_record(&record).unwrap();
        record.bindings.buffers_mut(BufferStage::Kernel)[2] = set(64, None);
        assert!(descriptor
            .product()
            .validate_record(&record)
            .unwrap_err()
            .contains("binding 2"));
        record.bindings.buffers_mut(BufferStage::Kernel)[2] = None;
        record.threadgroup_memory_lengths[1] = 16;
        assert!(descriptor
            .product()
            .validate_record(&record)
            .unwrap_err()
            .contains("threadgroup"));
        record.threadgroup_memory_lengths[1] = 0;
        // A compute buffer's records hold a dispatch of either kind, as Metal runs a kernel's; a
        // draw is the other family's.
        record.action = Action::DispatchThreads {
            grid: [1; 3],
            local_size: [1; 3],
        };
        descriptor.product().validate_record(&record).unwrap();
        record.action = Action::DrawPrimitives {
            primitive_type: 3,
            vertex_start: 0,
            vertex_count: 3,
            instance_count: 1,
            base_instance: 0,
        };
        assert!(descriptor
            .product()
            .validate_record(&record)
            .unwrap_err()
            .contains("action"));
        assert!(serde_json::from_str::<CommandDescriptor>(
            r#"{"kind":"compute","dispatch":"both","max_buffer_bind_count":1,"max_threadgroup_memory_bind_count":0,"support_dynamic_attribute_stride":true,"max_vertex_buffer_bind_count":1}"#
        ).is_err());
    }
    #[test]
    fn render_descriptor_checks_stage_state_and_dynamic_stride() {
        let mut record = CommandRecord {
            action: Action::DrawPrimitives {
                primitive_type: 3,
                vertex_start: 0,
                vertex_count: 3,
                instance_count: 1,
                base_instance: 0,
            },
            ..CommandRecord::default()
        };
        record.bindings.buffers_mut(BufferStage::Vertex)[0] = set(64, Some(16));
        for enabled in [false, true] {
            let descriptor = CommandDescriptor::Render {
                draws: RenderDraws::default(),
                max_vertex_buffer_bind_count: 1,
                max_fragment_buffer_bind_count: 1,
                max_object_buffer_bind_count: 0,
                max_mesh_buffer_bind_count: 0,
                support_dynamic_attribute_stride: enabled,
            };
            assert_eq!(
                descriptor.product().validate_record(&record).is_ok(),
                enabled
            );
        }
        let descriptor = CommandDescriptor::Render {
            draws: RenderDraws::default(),
            max_vertex_buffer_bind_count: 1,
            max_fragment_buffer_bind_count: 1,
            max_object_buffer_bind_count: 0,
            max_mesh_buffer_bind_count: 0,
            support_dynamic_attribute_stride: true,
        };
        record.bindings.buffers_mut(BufferStage::Fragment)[0] = set(0, Some(16));
        assert!(descriptor.product().validate_record(&record).is_err());
        record.bindings.buffers_mut(BufferStage::Fragment)[0] = None;
        for stage in [BufferStage::Kernel, BufferStage::Object] {
            record.bindings.buffers_mut(stage)[0] = set(64, None);
            assert!(descriptor.product().validate_record(&record).is_err());
            record.bindings.buffers_mut(stage)[0] = None;
        }
        record.action = Action::DispatchThreads {
            grid: [1; 3],
            local_size: [1; 3],
        };
        assert!(descriptor.product().validate_record(&record).is_err());
    }

    #[test]
    fn descriptor_limits_reject_oversized_counts_before_indexing() {
        for descriptor in [
            CommandDescriptor::Compute {
                dispatch: ComputeDispatch::Both,
                max_buffer_bind_count: u32::MAX,
                max_threadgroup_memory_bind_count: 0,
                support_dynamic_attribute_stride: false,
            },
            CommandDescriptor::Compute {
                dispatch: ComputeDispatch::Both,
                max_buffer_bind_count: 0,
                max_threadgroup_memory_bind_count: u32::MAX,
                support_dynamic_attribute_stride: false,
            },
            CommandDescriptor::Render {
                draws: RenderDraws::default(),
                max_vertex_buffer_bind_count: u32::MAX,
                max_fragment_buffer_bind_count: 0,
                max_object_buffer_bind_count: 0,
                max_mesh_buffer_bind_count: 0,
                support_dynamic_attribute_stride: false,
            },
            CommandDescriptor::Render {
                draws: RenderDraws::default(),
                max_vertex_buffer_bind_count: 0,
                max_fragment_buffer_bind_count: u32::MAX,
                max_object_buffer_bind_count: 0,
                max_mesh_buffer_bind_count: 0,
                support_dynamic_attribute_stride: false,
            },
        ] {
            assert!(descriptor
                .product()
                .validate_record(&CommandRecord::default())
                .is_err());
        }
    }
    #[test]
    fn command_resource_checks_range_and_exact_record_allocation_size() {
        use metal2vulkan::bind::indirect_commands::{COMMAND_BUFFER_HEADER_BYTES, COMMAND_BYTES};
        let mut resource = CommandBufferResource {
            buffer_binding: 0,
            field_offset: 16,
            capacity: 3,
            execution_range: [1, 3],
            inherit_pipeline_state: false,
            inherit_buffers: true,
            descriptor: CommandDescriptor::Compute {
                dispatch: ComputeDispatch::Both,
                max_buffer_bind_count: 0,
                max_threadgroup_memory_bind_count: 0,
                support_dynamic_attribute_stride: false,
            },
            commands: vec![],
            execute: None,
        };
        assert_eq!(
            resource.record_storage_bytes().unwrap(),
            COMMAND_BUFFER_HEADER_BYTES + 3 * COMMAND_BYTES
        );
        for range in [[0, 0], [3, 3], [0, 3]] {
            resource.execution_range = range;
            assert!(resource.record_storage_bytes().is_ok());
        }
        for range in [[2, 1], [0, 4], [4, 4], [0, u32::MAX]] {
            resource.execution_range = range;
            assert!(resource.record_storage_bytes().is_err());
        }
        resource.execution_range = [0, 0];
        resource.field_offset = u32::MAX;
        assert!(resource.record_storage_bytes().is_err());
        resource.field_offset = 0;
        resource.capacity = 0;
        assert_eq!(
            resource.record_storage_bytes().unwrap(),
            COMMAND_BUFFER_HEADER_BYTES
        );
        let value = serde_json::to_value(&resource).unwrap();
        assert_eq!(
            serde_json::from_value::<CommandBufferResource>(value.clone()).unwrap(),
            resource
        );
        let mut missing = value;
        missing.as_object_mut().unwrap().remove("inherit_buffers");
        assert!(serde_json::from_value::<CommandBufferResource>(missing).is_err());
    }
    #[test]
    fn completed_records_are_checked_before_any_command_is_exposed() {
        use metal2vulkan::bind::indirect_commands::{COMMAND_BUFFER_HEADER_BYTES, COMMAND_BYTES};
        let resource = CommandBufferResource {
            buffer_binding: 0,
            field_offset: 0,
            capacity: 2,
            execution_range: [0, 2],
            inherit_pipeline_state: false,
            inherit_buffers: false,
            descriptor: CommandDescriptor::Compute {
                dispatch: ComputeDispatch::Both,
                max_buffer_bind_count: 0,
                max_threadgroup_memory_bind_count: 0,
                support_dynamic_attribute_stride: false,
            },
            commands: vec![],
            execute: None,
        };
        let mut bytes = vec![0; resource.record_storage_bytes().unwrap()];
        resource.initialize_records(&mut bytes).unwrap();
        assert_eq!(resource.validated_records(&bytes).unwrap().len(), 2);
        let draw = CommandRecord {
            action: Action::DrawPrimitives {
                primitive_type: 3,
                vertex_start: 0,
                vertex_count: 3,
                instance_count: 1,
                base_instance: 0,
            },
            ..CommandRecord::default()
        };
        draw.encode_into(&mut bytes[COMMAND_BUFFER_HEADER_BYTES + COMMAND_BYTES..])
            .unwrap();
        assert!(
            resource.validated_records(&bytes).is_err(),
            "late invalid command rejects the whole selection"
        );
        assert!(resource
            .validated_records(&bytes[..COMMAND_BUFFER_HEADER_BYTES + COMMAND_BYTES])
            .is_err());
        resource.initialize_records(&mut bytes).unwrap();
        bytes[4..8].copy_from_slice(&1u32.to_le_bytes());
        assert!(
            resource.validated_records(&bytes).is_err(),
            "producer faults cannot expose commands"
        );
    }
    #[cfg(target_os = "macos")]
    #[test]
    fn native_descriptor_preserves_literal_settings_without_gpu_submission() {
        use objc2_metal::MTLIndirectCommandType;
        let mut resource = CommandBufferResource {
            buffer_binding: 0,
            field_offset: 0,
            capacity: 1,
            execution_range: [0, 1],
            inherit_pipeline_state: true,
            inherit_buffers: false,
            descriptor: CommandDescriptor::Compute {
                dispatch: ComputeDispatch::Both,
                max_buffer_bind_count: 2,
                max_threadgroup_memory_bind_count: 3,
                support_dynamic_attribute_stride: true,
            },
            commands: vec![],
            execute: None,
        };
        let descriptor = crate::metal::indirect_command_descriptor(&resource).unwrap();
        assert_eq!(
            descriptor.commandTypes(),
            MTLIndirectCommandType::ConcurrentDispatch
                | MTLIndirectCommandType::ConcurrentDispatchThreads
        );
        assert!(descriptor.inheritPipelineState());
        assert!(!descriptor.inheritBuffers());
        assert_eq!(descriptor.maxKernelBufferBindCount(), 2);
        assert_eq!(descriptor.maxKernelThreadgroupMemoryBindCount(), 3);
        assert_eq!(descriptor.maxVertexBufferBindCount(), 0);
        assert!(descriptor.supportDynamicAttributeStride());
        resource.descriptor = CommandDescriptor::Render {
            draws: RenderDraws::default(),
            max_vertex_buffer_bind_count: 4,
            max_fragment_buffer_bind_count: 5,
            max_object_buffer_bind_count: 0,
            max_mesh_buffer_bind_count: 0,
            support_dynamic_attribute_stride: false,
        };
        let descriptor = crate::metal::indirect_command_descriptor(&resource).unwrap();
        assert_eq!(descriptor.commandTypes(), MTLIndirectCommandType::Draw);
        assert_eq!(descriptor.maxVertexBufferBindCount(), 4);
        assert_eq!(descriptor.maxFragmentBufferBindCount(), 5);
        assert_eq!(descriptor.maxKernelBufferBindCount(), 0);
        assert_eq!(descriptor.maxKernelThreadgroupMemoryBindCount(), 0);
        assert!(!descriptor.supportDynamicAttributeStride());
    }
    #[test]
    fn inherited_state_rejects_command_owned_values_independently() {
        use metal2vulkan::bind::indirect_commands::COMMAND_BUFFER_HEADER_BYTES;
        let mut resource = CommandBufferResource {
            buffer_binding: 0,
            field_offset: 0,
            capacity: 1,
            execution_range: [0, 1],
            inherit_pipeline_state: true,
            inherit_buffers: false,
            descriptor: CommandDescriptor::Compute {
                dispatch: ComputeDispatch::Both,
                max_buffer_bind_count: 1,
                max_threadgroup_memory_bind_count: 1,
                support_dynamic_attribute_stride: false,
            },
            commands: vec![],
            execute: None,
        };
        let mut bytes = vec![0; resource.record_storage_bytes().unwrap()];
        resource.initialize_records(&mut bytes).unwrap();
        let mut record = CommandRecord::default();
        record.bindings.pipeline = std::num::NonZeroU64::new(1);
        record
            .encode_into(&mut bytes[COMMAND_BUFFER_HEADER_BYTES..])
            .unwrap();
        assert!(resource.validated_records(&bytes).is_err());
        resource.inherit_pipeline_state = false;
        resource.inherit_buffers = true;
        assert!(
            resource.validated_records(&bytes).is_err(),
            "policy changes require reinitialization"
        );
        resource.initialize_records(&mut bytes).unwrap();
        record
            .encode_into(&mut bytes[COMMAND_BUFFER_HEADER_BYTES..])
            .unwrap();
        assert!(resource.validated_records(&bytes).is_ok());
        record.bindings.buffers_mut(BufferStage::Kernel)[0] = set(64, None);
        record
            .encode_into(&mut bytes[COMMAND_BUFFER_HEADER_BYTES..])
            .unwrap();
        assert!(resource.validated_records(&bytes).is_err());
        record.bindings.buffers_mut(BufferStage::Kernel)[0] = None;
        record.threadgroup_memory_lengths[0] = 16;
        record
            .encode_into(&mut bytes[COMMAND_BUFFER_HEADER_BYTES..])
            .unwrap();
        assert!(resource.validated_records(&bytes).is_ok());
    }
    #[test]
    fn invalid_initialization_preserves_caller_storage() {
        let mut resource = CommandBufferResource {
            buffer_binding: 0,
            field_offset: 0,
            capacity: 1,
            execution_range: [0, 1],
            inherit_pipeline_state: false,
            inherit_buffers: false,
            descriptor: CommandDescriptor::Compute {
                dispatch: ComputeDispatch::Both,
                max_buffer_bind_count: 0,
                max_threadgroup_memory_bind_count: 0,
                support_dynamic_attribute_stride: false,
            },
            commands: vec![],
            execute: None,
        };
        let mut bytes = vec![0xa5; resource.record_storage_bytes().unwrap()];
        resource.execution_range = [0, 2];
        assert!(resource.initialize_records(&mut bytes).is_err());
        assert!(bytes.iter().all(|&byte| byte == 0xa5));
        resource.execution_range = [0, 1];
        let short = bytes.len() - 1;
        assert!(resource.initialize_records(&mut bytes[..short]).is_err());
        assert!(bytes.iter().all(|&byte| byte == 0xa5));
        resource.initialize_records(&mut bytes).unwrap();
        assert_eq!(resource.validated_records(&bytes).unwrap().len(), 1);
    }
}
