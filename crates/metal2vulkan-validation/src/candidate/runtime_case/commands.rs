//! A case's indirect command buffers and pipeline states through the runtime, made and run as the
//! Metal oracle makes and runs them (`metal::platform::indirect_commands`): a kernel's
//! (`runtime_case::encoded_case`) and a draw's (`runtime_case::encoded_render`).
//!
//! Each authored command buffer is an `IndirectCommandBuffer` of the oracle's descriptor (compute
//! commands' types from the authored dispatch kinds and their kernel buffer and
//! threadgroup-length counts; render commands' `DRAW` and their vertex and fragment buffer
//! counts; the authored inheritance), of the authored capacity, `Shared` where the host encodes
//! commands and `Private` otherwise. Its argument-buffer field holds its `gpu_resource_id`, as the
//! oracle's argument encoder writes it (`setIndirectCommandBuffer:atIndex:`), and a pipeline
//! state's field the `gpu_resource_id` of the template's state, compute or render, made with
//! `support_indirect_command_buffers` ([`CaseCommands::words`], [`TemplateState`]). The host's
//! commands are encoded through their setters before the producer runs
//! (`encode_host_commands`); the producer's encoder makes each command buffer resident for
//! reading and writing. Once the producer completed, each command buffer's commands are read back
//! as the runtime decodes them ([`CaseCommands::check_commands`], the host contract's check of what
//! the producer wrote: a `Private` one through a blit copy into a `Shared` twin the producer's
//! command buffer makes), and each command buffer the case executes runs in key order, each in a
//! command buffer of its own committed after the one before it on the producer's queue
//! ([`encoded_execution`]): compute commands in a compute encoder, render commands in a render
//! encoder of the oracle's pass (its one attachment a 1x1 `R8Unorm` sink, neither loaded nor
//! stored), with the parent's pipeline state and buffers, every case buffer and argument-buffer
//! buffer resident for reading and writing, and `execute_commands_in_buffer` over the execution
//! range.

use super::CaseError;
use crate::case::indirect_commands::{
    metal_index_type, metal_primitive_type, CommandBufferResource, CommandDescriptor,
    ComputeDispatch, HostAction, HostCommand,
};
use metal2vulkan::metal::{
    IndirectCommandType, PixelFormat, RenderStages, ResourceUsage, TextureUsage,
};
use metal2vulkan_runtime::{
    Buffer, CommandBuffer, CommandQueue, ComputePipelineState, Device, IndirectCommandBuffer,
    IndirectCommandBufferDescriptor, LoadAction, Region, RenderPassDescriptor, RenderPipelineState,
    ResourceId, ResourceOptions, ResourceRef, Size, StoreAction, TextureDescriptor,
};
use std::collections::BTreeMap;

/// The state of a pipeline template, made with `support_indirect_command_buffers`: a kernel's
/// compute state, or a vertex or fragment case's render state.
#[derive(Clone)]
pub enum TemplateState {
    Compute(ComputePipelineState),
    Render(RenderPipelineState),
}

impl TemplateState {
    /// What Metal's argument encoder writes at a pipeline state field for it.
    pub fn gpu_resource_id(&self) -> ResourceId {
        match self {
            TemplateState::Compute(state) => state.gpu_resource_id(),
            TemplateState::Render(state) => state.gpu_resource_id(),
        }
    }
}

/// A case's indirect command buffers and the pipeline states its argument buffers hold, each by
/// `(owner Metal index, field offset)`: a kernel's, or one stage's of a draw.
#[derive(Default)]
pub struct CaseCommands {
    /// Each command buffer.
    pub buffers: BTreeMap<(u32, u32), IndirectCommandBuffer>,
    /// The `Shared` twin of each `Private` command buffer, which the producer's command buffer
    /// copies it into after its dispatch or draw, so its commands read back
    /// ([`CaseCommands::check_commands`]).
    pub twins: BTreeMap<(u32, u32), IndirectCommandBuffer>,
    /// Each pipeline state an argument buffer holds: its token's template's state.
    pub states: BTreeMap<(u32, u32), TemplateState>,
}

impl CaseCommands {
    /// `command_buffers` on `device`, as the oracle makes them, each with a twin where it is
    /// `Private`; no pipeline state (a caller sets [`CaseCommands::states`] from the templates it
    /// made, `pipeline_states`).
    pub fn of(
        device: &Device,
        command_buffers: &BTreeMap<(u32, u32), CommandBufferResource>,
    ) -> Result<CaseCommands, CaseError> {
        let mut commands = CaseCommands::default();
        for (&key, resource) in command_buffers {
            let descriptor = descriptor(resource);
            let made = |options| {
                device
                    .new_indirect_command_buffer(&descriptor, resource.capacity.into(), options)
                    .map_err(CaseError::runtime(format!(
                        "command buffer {key:?}: new_indirect_command_buffer"
                    )))
            };
            let options = if resource.commands.is_empty() {
                commands.twins.insert(key, made(ResourceOptions::shared())?);
                ResourceOptions::private()
            } else {
                ResourceOptions::shared()
            };
            commands.buffers.insert(key, made(options)?);
        }
        Ok(commands)
    }

    /// The word each command buffer's and pipeline state's owner holds at its field, as Metal's
    /// argument encoder writes it: its `gpu_resource_id`.
    pub fn words(&self) -> impl Iterator<Item = ((u32, u32), u64)> + '_ {
        let buffers = self
            .buffers
            .iter()
            .map(|(&key, icb)| (key, icb.gpu_resource_id().0));
        let states = self
            .states
            .iter()
            .map(|(&key, state)| (key, state.gpu_resource_id().0));
        buffers.chain(states)
    }

    /// Each command buffer, which the producer's encoder makes resident for reading and writing.
    pub fn resources(&self) -> impl Iterator<Item = ResourceRef<'_>> {
        self.buffers
            .values()
            .map(IndirectCommandBuffer::as_resource)
    }

    /// The command buffer at `key`.
    fn buffer(&self, key: (u32, u32)) -> Result<&IndirectCommandBuffer, CaseError> {
        self.buffers
            .get(&key)
            .ok_or_else(|| format!("command buffer {key:?} was not made").into())
    }

    /// Read back, once the producer completed, the commands of each command buffer as the runtime
    /// decodes them (a `Private` one's from its twin): the host contract's check of what the
    /// producer wrote, as the oracle's guard and the executor's decode check it.
    pub fn check_commands(&self) -> Result<(), CaseError> {
        for (key, icb) in &self.buffers {
            let readable = self.twins.get(key).unwrap_or(icb);
            readable
                .commands(0..readable.size())
                .map_err(CaseError::runtime(format!(
                    "command buffer {key:?}: commands"
                )))?;
        }
        Ok(())
    }
}

/// The pipeline state each of `objects`' fields (`(owner Metal index, field offset)` to token)
/// holds: the state `templates` makes for its token, made with `support_indirect_command_buffers`.
pub fn pipeline_states(
    objects: &BTreeMap<(u32, u32), u64>,
    templates: &BTreeMap<u64, TemplateState>,
) -> Result<BTreeMap<(u32, u32), TemplateState>, CaseError> {
    objects
        .iter()
        .map(|(&key, token)| {
            let state = templates
                .get(token)
                .ok_or_else(|| format!("pipeline token {token} has no template"))?;
            Ok((key, state.clone()))
        })
        .collect()
}

/// The runtime's descriptor of `resource`, as the oracle states its own
/// (`metal::indirect_command_descriptor`): for compute commands no vertex or fragment buffers,
/// the authored kernel counts, its command types the authored dispatch kinds; for render commands
/// the authored draw kinds (`DRAW`, `DRAW_INDEXED`, `DRAW_PATCHES`, `DRAW_INDEXED_PATCHES`,
/// `DRAW_MESH_THREADGROUPS`, `DRAW_MESH_THREADS`) and vertex, fragment, object and mesh buffer
/// counts, no kernel buffers; and the authored inheritance.
fn descriptor(resource: &CommandBufferResource) -> IndirectCommandBufferDescriptor {
    let mut descriptor = IndirectCommandBufferDescriptor::new();
    match resource.descriptor {
        CommandDescriptor::Compute {
            dispatch,
            max_buffer_bind_count,
            max_threadgroup_memory_bind_count,
            support_dynamic_attribute_stride,
        } => {
            descriptor.command_types = match dispatch {
                ComputeDispatch::Threadgroups => IndirectCommandType::CONCURRENT_DISPATCH,
                ComputeDispatch::Threads => IndirectCommandType::CONCURRENT_DISPATCH_THREADS,
                ComputeDispatch::Both => IndirectCommandType::CONCURRENT_DISPATCH
                    .union(IndirectCommandType::CONCURRENT_DISPATCH_THREADS),
            };
            descriptor.max_kernel_buffer_bind_count = max_buffer_bind_count;
            descriptor.max_kernel_threadgroup_memory_bind_count = max_threadgroup_memory_bind_count;
            descriptor.support_dynamic_attribute_stride = support_dynamic_attribute_stride;
        }
        CommandDescriptor::Render {
            draws,
            max_vertex_buffer_bind_count,
            max_fragment_buffer_bind_count,
            max_object_buffer_bind_count,
            max_mesh_buffer_bind_count,
            support_dynamic_attribute_stride,
        } => {
            descriptor.command_types = [
                (draws.primitives, IndirectCommandType::DRAW),
                (draws.indexed, IndirectCommandType::DRAW_INDEXED),
                (draws.patches, IndirectCommandType::DRAW_PATCHES),
                (
                    draws.indexed_patches,
                    IndirectCommandType::DRAW_INDEXED_PATCHES,
                ),
                (
                    draws.mesh_threadgroups,
                    IndirectCommandType::DRAW_MESH_THREADGROUPS,
                ),
                (draws.mesh_threads, IndirectCommandType::DRAW_MESH_THREADS),
            ]
            .into_iter()
            .filter(|&(allowed, _)| allowed)
            .fold(IndirectCommandType::empty(), |types, (_, kind)| {
                types.union(kind)
            });
            descriptor.max_vertex_buffer_bind_count = max_vertex_buffer_bind_count;
            descriptor.max_fragment_buffer_bind_count = max_fragment_buffer_bind_count;
            descriptor.max_object_buffer_bind_count = max_object_buffer_bind_count;
            descriptor.max_mesh_buffer_bind_count = max_mesh_buffer_bind_count;
            descriptor.support_dynamic_attribute_stride = support_dynamic_attribute_stride;
        }
    }
    descriptor.inherit_pipeline_state = resource.inherit_pipeline_state;
    descriptor.inherit_buffers = resource.inherit_buffers;
    descriptor
}

/// The stages a render encoder makes the case's resources resident in for `resource`'s commands, as
/// the oracle does: the vertex and fragment stages, and the object and mesh stages too where its
/// descriptor allows mesh draws.
fn render_stages(resource: &CommandBufferResource) -> RenderStages {
    let mesh = match resource.descriptor {
        CommandDescriptor::Render { draws, .. } => draws.mesh(),
        CommandDescriptor::Compute { .. } => false,
    };
    let stages = RenderStages::VERTEX | RenderStages::FRAGMENT;
    if mesh {
        stages | RenderStages::OBJECT | RenderStages::MESH
    } else {
        stages
    }
}

/// The size of a grid or threadgroup.
fn size([x, y, z]: [u32; 3]) -> Size {
    Size::new(x.into(), y.into(), z.into())
}

/// The case's buffer `index` among `buffers`, which `what` binds.
fn case_buffer<'a>(
    buffers: &'a BTreeMap<u32, Buffer>,
    index: u32,
    what: &str,
) -> Result<&'a Buffer, CaseError> {
    buffers
        .get(&index)
        .ok_or_else(|| format!("{what} buffer {index} is not bound").into())
}

/// The compute template state of `token` among `templates`.
fn compute_template(
    templates: &BTreeMap<u64, TemplateState>,
    token: u64,
) -> Result<&ComputePipelineState, CaseError> {
    match templates.get(&token) {
        Some(TemplateState::Compute(state)) => Ok(state),
        _ => Err(format!("pipeline token {token} has no compute template").into()),
    }
}

/// The render template state of `token` among `templates`.
fn render_template(
    templates: &BTreeMap<u64, TemplateState>,
    token: u64,
) -> Result<&RenderPipelineState, CaseError> {
    match templates.get(&token) {
        Some(TemplateState::Render(state)) => Ok(state),
        _ => Err(format!("pipeline token {token} has no render template").into()),
    }
}

/// Encode each host command of `command_buffers` (made as `commands`) through its setters, before
/// the producer runs, as the oracle encodes them (`encode_host_commands`): its pipeline state (its
/// token's template among `templates`), each buffer the case's buffer at its offset (among
/// `buffers`), a compute command's threadgroup lengths, its dispatch or draw, and a compute
/// command's barrier (the oracle models no render command's).
pub fn encode_host_commands(
    command_buffers: &BTreeMap<(u32, u32), CommandBufferResource>,
    commands: &CaseCommands,
    buffers: &BTreeMap<u32, Buffer>,
    templates: &BTreeMap<u64, TemplateState>,
) -> Result<(), CaseError> {
    for (&key, resource) in command_buffers {
        let icb = commands.buffer(key)?;
        for host in &resource.commands {
            let label = format!("command buffer {key:?} command {}", host.index);
            if resource.is_render() {
                encode_render_command(icb, host, &label, buffers, templates)?;
            } else {
                encode_compute_command(icb, host, &label, buffers, templates)?;
            }
        }
    }
    Ok(())
}

/// Encode host command `host` of compute command buffer `icb` ([`encode_host_commands`]).
fn encode_compute_command(
    icb: &IndirectCommandBuffer,
    host: &HostCommand,
    label: &str,
    buffers: &BTreeMap<u32, Buffer>,
    templates: &BTreeMap<u64, TemplateState>,
) -> Result<(), CaseError> {
    let runtime = |call: &str| CaseError::runtime(format!("{label}: {call}"));
    let command = icb
        .indirect_compute_command_at(host.index.into())
        .map_err(runtime("indirect_compute_command_at"))?;
    if let Some(token) = host.pipeline {
        command
            .set_compute_pipeline_state(compute_template(templates, token)?)
            .map_err(runtime("set_compute_pipeline_state"))?;
    }
    for binding in &host.buffers {
        let buffer = case_buffer(buffers, binding.buffer, "a host command's")?;
        match binding.attribute_stride {
            Some(stride) => command
                .set_kernel_buffer_with_stride(
                    Some(buffer),
                    binding.offset.into(),
                    stride.into(),
                    binding.index,
                )
                .map_err(runtime("set_kernel_buffer_with_stride"))?,
            None => command
                .set_kernel_buffer(Some(buffer), binding.offset.into(), binding.index)
                .map_err(runtime("set_kernel_buffer"))?,
        }
    }
    if let Some(region) = host.stage_in_region {
        let [x, y, z] = region.origin.map(u64::from);
        let [width, height, depth] = region.size.map(u64::from);
        command
            .set_stage_in_region(Region::new_3d(x, y, z, width, height, depth))
            .map_err(runtime("set_stage_in_region"))?;
    }
    for memory in &host.threadgroup_memory_lengths {
        command
            .set_threadgroup_memory_length(memory.length, memory.index)
            .map_err(runtime("set_threadgroup_memory_length"))?;
    }
    match host.action {
        Some(HostAction::DispatchThreadgroups {
            grid,
            threads_per_threadgroup,
        }) => command
            .concurrent_dispatch_threadgroups(size(grid), size(threads_per_threadgroup))
            .map_err(runtime("concurrent_dispatch_threadgroups"))?,
        Some(HostAction::DispatchThreads {
            grid,
            threads_per_threadgroup,
        }) => command
            .concurrent_dispatch_threads(size(grid), size(threads_per_threadgroup))
            .map_err(runtime("concurrent_dispatch_threads"))?,
        Some(
            HostAction::DrawPrimitives { .. }
            | HostAction::DrawIndexedPrimitives { .. }
            | HostAction::DrawPatches { .. },
        ) => return Err(format!("{label}: a compute command draws nothing").into()),
        None => {}
    }
    if host.barrier {
        command.set_barrier().map_err(runtime("set_barrier"))?;
    }
    Ok(())
}

/// Encode host command `host` of render command buffer `icb` ([`encode_host_commands`]): its
/// render pipeline state, vertex and fragment buffers, and draw (of primitives, indexed from a
/// case buffer at its offset, or of patches from case buffers at their offsets), as the oracle's
/// `indirectRenderCommandAtIndex:` setters do.
fn encode_render_command(
    icb: &IndirectCommandBuffer,
    host: &HostCommand,
    label: &str,
    buffers: &BTreeMap<u32, Buffer>,
    templates: &BTreeMap<u64, TemplateState>,
) -> Result<(), CaseError> {
    let runtime = |call: &str| CaseError::runtime(format!("{label}: {call}"));
    let command = icb
        .indirect_render_command_at(host.index.into())
        .map_err(runtime("indirect_render_command_at"))?;
    if let Some(token) = host.pipeline {
        command
            .set_render_pipeline_state(render_template(templates, token)?)
            .map_err(runtime("set_render_pipeline_state"))?;
    }
    for binding in &host.vertex_buffers {
        let buffer = case_buffer(buffers, binding.buffer, "a host command's")?;
        match binding.attribute_stride {
            Some(stride) => command
                .set_vertex_buffer_with_stride(
                    Some(buffer),
                    binding.offset.into(),
                    stride.into(),
                    binding.index,
                )
                .map_err(runtime("set_vertex_buffer_with_stride"))?,
            None => command
                .set_vertex_buffer(Some(buffer), binding.offset.into(), binding.index)
                .map_err(runtime("set_vertex_buffer"))?,
        }
    }
    for binding in &host.fragment_buffers {
        command
            .set_fragment_buffer(
                Some(case_buffer(buffers, binding.buffer, "a host command's")?),
                binding.offset.into(),
                binding.index,
            )
            .map_err(runtime("set_fragment_buffer"))?;
    }
    match host.action {
        Some(HostAction::DrawPrimitives {
            primitive,
            vertex_start,
            vertex_count,
            instance_count,
            base_instance,
        }) => command
            .draw_primitives(
                metal_primitive_type(primitive),
                vertex_start.into(),
                vertex_count.into(),
                instance_count.into(),
                base_instance.into(),
            )
            .map_err(runtime("draw_primitives"))?,
        Some(HostAction::DrawIndexedPrimitives {
            primitive,
            index_count,
            index_type,
            index_buffer,
            index_buffer_offset,
            instance_count,
            base_vertex,
            base_instance,
        }) => command
            .draw_indexed_primitives(
                metal_primitive_type(primitive),
                index_count.into(),
                metal_index_type(index_type),
                case_buffer(buffers, index_buffer, "an indexed draw's index")?,
                index_buffer_offset.into(),
                instance_count.into(),
                base_vertex.into(),
                base_instance.into(),
            )
            .map_err(runtime("draw_indexed_primitives"))?,
        Some(HostAction::DrawPatches {
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
        }) => command
            .draw_patches(
                control_points.into(),
                patch_start.into(),
                patch_count.into(),
                patch_index_buffer
                    .map(|index| case_buffer(buffers, index, "a patch draw's patch index"))
                    .transpose()?,
                patch_index_buffer_offset.into(),
                instance_count.into(),
                base_instance.into(),
                case_buffer(buffers, factor_buffer, "a patch draw's factor")?,
                factor_buffer_offset.into(),
                factor_instance_stride.into(),
            )
            .map_err(runtime("draw_patches"))?,
        Some(_) => return Err(format!("{label}: a render command dispatches nothing").into()),
        None => {}
    }
    if host.barrier {
        return Err(
            format!("{label}: a host-encoded render command's barrier is not modeled").into(),
        );
    }
    Ok(())
}

/// Encode on `command_buffer`, after the producer's dispatch or draw, the copy of each `Private`
/// command buffer of `commands` into its `Shared` twin, so its commands read back once the
/// producer completed ([`CaseCommands::check_commands`]).
pub fn encode_twin_copies(
    command_buffer: &CommandBuffer,
    commands: &CaseCommands,
) -> Result<(), CaseError> {
    if commands.twins.is_empty() {
        return Ok(());
    }
    let mut blit = command_buffer
        .blit_command_encoder()
        .map_err(CaseError::runtime("blit_command_encoder"))?;
    for (&key, twin) in &commands.twins {
        let icb = commands.buffer(key)?;
        blit.copy_indirect_command_buffer(icb, 0..icb.size(), twin, 0)
            .map_err(CaseError::runtime(format!(
                "command buffer {key:?}: copy_indirect_command_buffer"
            )))?;
    }
    blit.end_encoding()
        .map_err(CaseError::runtime("end_encoding"))
}

/// The case's command buffer at `key` that it executes (`resource`, whose `execute` is set, one of
/// `commands`), encoded on a command buffer of `queue`, the producer's, and not committed, as the
/// oracle runs it (`execute_host_commands`): a compute encoder for compute commands, and for render
/// commands a render encoder of the oracle's pass, whose one attachment is a 1x1 `R8Unorm` sink
/// neither loaded nor stored (`make_rasterization_sink`: the render templates draw nothing into
/// it); the parent's pipeline state (its token's template among `templates`) and buffers (among
/// `buffers`, the case's by Metal index), `resident` (every buffer of the case and every
/// argument-buffer buffer) resident for reading and writing, in the vertex and fragment stages of
/// a render encoder, and `execute_commands_in_buffer` over the execution range. Committed after
/// the producer's command buffer, it runs once the producer completed, as the oracle's waits order
/// it: the queue runs its command buffers in commit order, and commands the producer writes run
/// once the work before them completed.
pub fn encoded_execution(
    queue: &CommandQueue,
    (buffers, resident): (&BTreeMap<u32, Buffer>, &[ResourceRef<'_>]),
    commands: &CaseCommands,
    (key, resource): ((u32, u32), &CommandBufferResource),
    templates: &BTreeMap<u64, TemplateState>,
) -> Result<CommandBuffer, CaseError> {
    let execution = resource
        .execute
        .as_ref()
        .ok_or_else(|| format!("command buffer {key:?} is not executed"))?;
    let icb = commands.buffer(key)?;
    let label = format!("command buffer {key:?}'s execution");
    let runtime = |call: &str| CaseError::runtime(format!("{label}: {call}"));
    let parent = |binding: &crate::case::indirect_commands::CommandBinding| {
        case_buffer(buffers, binding.buffer, "the parent's")
    };
    let command_buffer = queue.command_buffer().map_err(runtime("command_buffer"))?;
    let [start, end] = resource.execution_range;
    let range = u64::from(start)..u64::from(end);
    let usage = ResourceUsage::READ | ResourceUsage::WRITE;
    if resource.is_render() {
        let mut sink = TextureDescriptor::texture_2d(PixelFormat::R8Unorm, 1, 1, false);
        sink.usage = TextureUsage::RENDER_TARGET;
        sink.resource_options = ResourceOptions::shared();
        let sink = queue
            .device()
            .new_texture(&sink)
            .map_err(runtime("new_texture"))?;
        let mut pass = RenderPassDescriptor::new();
        let attachment = &mut pass.color_attachments[0];
        attachment.texture = Some(sink);
        attachment.load_action = LoadAction::DontCare;
        attachment.store_action = StoreAction::DontCare;
        let mut encoder = command_buffer
            .render_command_encoder(&pass)
            .map_err(runtime("render_command_encoder"))?;
        if let Some(parent) = &execution.pipeline {
            encoder
                .set_render_pipeline_state(render_template(templates, parent.token)?)
                .map_err(runtime("set_render_pipeline_state"))?;
        }
        for binding in &execution.vertex_buffers {
            encoder
                .set_vertex_buffer(Some(parent(binding)?), binding.offset.into(), binding.index)
                .map_err(runtime("set_vertex_buffer"))?;
        }
        for binding in &execution.fragment_buffers {
            encoder
                .set_fragment_buffer(Some(parent(binding)?), binding.offset.into(), binding.index)
                .map_err(runtime("set_fragment_buffer"))?;
        }
        for binding in &execution.object_buffers {
            encoder
                .set_object_buffer(Some(parent(binding)?), binding.offset.into(), binding.index)
                .map_err(runtime("set_object_buffer"))?;
        }
        for binding in &execution.mesh_buffers {
            encoder
                .set_mesh_buffer(Some(parent(binding)?), binding.offset.into(), binding.index)
                .map_err(runtime("set_mesh_buffer"))?;
        }
        encoder
            .use_resources(resident, usage, render_stages(resource))
            .map_err(runtime("use_resources"))?;
        encoder
            .execute_commands_in_buffer(icb, range)
            .map_err(runtime("execute_commands_in_buffer"))?;
        encoder.end_encoding().map_err(runtime("end_encoding"))?;
    } else {
        let mut encoder = command_buffer
            .compute_command_encoder()
            .map_err(runtime("compute_command_encoder"))?;
        if let Some(parent) = &execution.pipeline {
            encoder
                .set_compute_pipeline_state(compute_template(templates, parent.token)?)
                .map_err(runtime("set_compute_pipeline_state"))?;
        }
        for binding in &execution.buffers {
            encoder
                .set_buffer(Some(parent(binding)?), binding.offset.into(), binding.index)
                .map_err(runtime("set_buffer"))?;
        }
        encoder
            .use_resources(resident, usage)
            .map_err(runtime("use_resources"))?;
        encoder
            .execute_commands_in_buffer(icb, range)
            .map_err(runtime("execute_commands_in_buffer"))?;
        encoder.end_encoding().map_err(runtime("end_encoding"))?;
    }
    Ok(command_buffer)
}
