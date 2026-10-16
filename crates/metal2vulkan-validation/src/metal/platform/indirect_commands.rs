//! Native indirect-command resource preflight, ownership, and argument encoding.

use super::pipelines::NativePipeline;
use super::{argument_encoder, argument_member, Buffer};
use crate::case::AuthoredCase;
use crate::metal::indirect_command_descriptor;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLArgumentEncoder, MTLBuffer, MTLComputePipelineState, MTLDevice, MTLFunction,
    MTLIndirectCommandBuffer, MTLIndirectComputeCommand, MTLIndirectRenderCommand,
    MTLRenderPipelineState, MTLResourceOptions,
};
use std::collections::BTreeMap;

pub(super) type IndirectCommands = (
    (u32, u32),
    Retained<ProtocolObject<dyn MTLIndirectCommandBuffer>>,
);
/// Own registrations and their native resources through producer completion.
pub(super) struct NativeCommandResources {
    pub(super) commands: Vec<IndirectCommands>,
    pub(super) policy: Option<(crate::metal::command_guard::NativeCommandPolicies, Buffer)>,
}

impl NativeCommandResources {
    /// Call only after producer completion, before observing or replaying its commands.
    pub(super) fn validate_readback(
        &self,
    ) -> Result<(), metal2vulkan::bind::indirect_commands::CommandBufferError> {
        if let Some((expected, buffer)) = &self.policy {
            // Shared storage is host-readable, and completion ends GPU writes.
            let bytes = unsafe {
                std::slice::from_raw_parts(buffer.contents().as_ptr().cast::<u8>(), buffer.length())
            };
            expected.validate_readback(bytes)?;
        }
        Ok(())
    }
}

/// Finish descriptor validation before any resource allocation can begin.
pub(super) fn allocate_command_resources<T>(
    resources: &[crate::case::indirect_commands::CommandBufferResource],
    mut allocate: impl FnMut(
        &crate::case::indirect_commands::CommandBufferResource,
        Retained<objc2_metal::MTLIndirectCommandBufferDescriptor>,
    ) -> Result<T, String>,
) -> Result<Vec<T>, String> {
    let descriptors = resources
        .iter()
        .map(indirect_command_descriptor)
        .collect::<Result<Vec<_>, _>>()?;
    resources
        .iter()
        .zip(descriptors)
        .map(|(resource, descriptor)| allocate(resource, descriptor))
        .collect()
}

pub(super) fn make_indirect_commands(
    device: &ProtocolObject<dyn MTLDevice>,
    case: &AuthoredCase,
) -> Result<NativeCommandResources, String> {
    let commands = allocate_command_resources(&case.command_buffers, |resource, descriptor| {
        // The count is a validated u32 and fits NSUInteger on supported macOS hosts.
        // Native storage is opaque; the host-record stride is not its allocation size.
        // The host encodes commands into CPU-visible storage only; a buffer the host leaves alone
        // stays private, as before.
        let storage = if resource.commands.is_empty() {
            MTLResourceOptions::StorageModePrivate
        } else {
            MTLResourceOptions::StorageModeShared
        };
        let buffer = unsafe {
            device.newIndirectCommandBufferWithDescriptor_maxCommandCount_options(
                &descriptor,
                resource.capacity as usize,
                storage,
            )
        }
        .ok_or_else(|| {
            format!(
                "create Metal indirect command buffer {}+{}",
                resource.buffer_binding, resource.field_offset
            )
        })?;
        if buffer.size() != resource.capacity as usize {
            return Err(
                "Metal indirect command-buffer capacity differs from the authored count".into(),
            );
        }
        Ok(((resource.buffer_binding, resource.field_offset), buffer))
    })?;
    let policy = if commands.is_empty() {
        None
    } else {
        let registrations = commands
            .iter()
            .zip(&case.command_buffers)
            .map(|((_, command), resource)| {
                let id = std::num::NonZeroU64::new(command.gpuResourceID().to_raw())
                    .ok_or("Metal command buffer returned a zero resource ID")?;
                crate::metal::command_guard::NativeCommandPolicy::new(
                    id,
                    &resource.descriptor.product(),
                    resource.capacity,
                    resource.inheritance(),
                )
                .map_err(|error| error.to_string())
            })
            .collect::<Result<Vec<_>, String>>()?;
        let expected = crate::metal::command_guard::NativeCommandPolicies::new(registrations)
            .map_err(|error| error.to_string())?;
        let initial = expected.bytes();
        let buffer = device
            .newBufferWithLength_options(initial.len(), MTLResourceOptions::StorageModeShared)
            .ok_or("create Metal command policy table")?;
        unsafe {
            std::slice::from_raw_parts_mut(buffer.contents().as_ptr().cast::<u8>(), initial.len())
                .copy_from_slice(initial);
        }
        Some((expected, buffer))
    };
    Ok(NativeCommandResources { commands, policy })
}

pub(super) fn encode_indirect_commands(
    function: &ProtocolObject<dyn MTLFunction>,
    buffers: &[(u32, Buffer)],
    commands: &[IndirectCommands],
    interface: &crate::check::interface::AirInterface,
) -> Result<(), String> {
    for ((binding, offset), commands) in commands {
        let (index, ()) = argument_member(
            interface,
            *binding,
            *offset,
            "indirect command buffer",
            |role| matches!(role, metal2vulkan_air::abi::ParamRole::CommandBuffer(_)).then_some(()),
        )?;
        let encoder = argument_encoder(function, buffers, *binding)?;
        // The AIR layout identifies the native argument index. Both resources remain alive
        // through producer completion; encoding happens before the command buffer is submitted.
        unsafe { encoder.setIndirectCommandBuffer_atIndex(Some(&**commands), index as usize) };
    }
    Ok(())
}

/// The compiled template of each pipeline token of `case`: its pipeline states' and its executions'
/// parent pipelines', each a template of `pipelines` (by case id), of either family.
fn token_pipelines<'a>(
    case: &AuthoredCase,
    pipelines: &'a BTreeMap<String, NativePipeline>,
) -> Result<BTreeMap<u64, &'a NativePipeline>, String> {
    let parents = case.command_buffers.iter().filter_map(|resource| {
        let template = resource.execute.as_ref()?.pipeline.as_ref()?;
        Some((template.token, &template.case_id))
    });
    case.pipeline_states
        .iter()
        .map(|resource| (resource.token, &resource.case_id))
        .chain(parents)
        .map(|(token, case_id)| {
            pipelines
                .get(case_id)
                .map(|pipeline| (token, pipeline))
                .ok_or_else(|| format!("missing compiled pipeline template {case_id}"))
        })
        .collect()
}

/// The compute template of `token`.
fn compute_pipeline<'a>(
    tokens: &BTreeMap<u64, &'a NativePipeline>,
    token: u64,
) -> Result<&'a ProtocolObject<dyn MTLComputePipelineState>, String> {
    match tokens.get(&token) {
        Some(NativePipeline::Compute(pipeline)) => Ok(pipeline),
        _ => Err(format!("pipeline token {token} has no compute template")),
    }
}

/// The render template of `token`.
fn render_pipeline<'a>(
    tokens: &BTreeMap<u64, &'a NativePipeline>,
    token: u64,
) -> Result<&'a ProtocolObject<dyn MTLRenderPipelineState>, String> {
    match tokens.get(&token) {
        Some(NativePipeline::Render(pipeline)) => Ok(pipeline),
        _ => Err(format!("pipeline token {token} has no render template")),
    }
}

/// The case's buffer `index` (each binding names one; the literal checked each).
fn case_buffer<'a>(
    buffers: &'a [(u32, Buffer)],
    index: u32,
    what: &str,
) -> Result<&'a Buffer, String> {
    buffers
        .iter()
        .find(|(binding, _)| *binding == index)
        .map(|(_, buffer)| buffer)
        .ok_or_else(|| format!("{what} buffer {index} is not bound"))
}

/// Encode each command buffer's host commands from the CPU, before the producer runs: each field
/// the literal sets through its `MTLIndirectComputeCommand` or `MTLIndirectRenderCommand` setter, a
/// buffer as the case's buffer at its offset, a pipeline as the compiled template of its token.
pub(super) fn encode_host_commands(
    case: &AuthoredCase,
    commands: &[IndirectCommands],
    buffers: &[(u32, Buffer)],
    pipelines: &BTreeMap<String, NativePipeline>,
) -> Result<(), String> {
    use crate::case::indirect_commands::CommandDescriptor;
    use crate::case::indirect_commands::HostAction;
    if case
        .command_buffers
        .iter()
        .all(|resource| resource.commands.is_empty())
    {
        return Ok(());
    }
    let tokens = token_pipelines(case, pipelines)?;
    for resource in &case.command_buffers {
        let key = (resource.buffer_binding, resource.field_offset);
        let native = &commands
            .iter()
            .find(|(owner, _)| *owner == key)
            .ok_or_else(|| format!("command buffer {key:?} was not created"))?
            .1;
        for host in &resource.commands {
            // The literal checked each index against the capacity, each binding's offset inside
            // its buffer and index below the descriptor's count, and each field against the
            // descriptor's family.
            match resource.descriptor {
                CommandDescriptor::Compute { .. } => {
                    let command =
                        unsafe { native.indirectComputeCommandAtIndex(host.index as usize) };
                    if let Some(token) = host.pipeline {
                        command.setComputePipelineState(compute_pipeline(&tokens, token)?);
                    }
                    for binding in &host.buffers {
                        let buffer = case_buffer(buffers, binding.buffer, "host command")?;
                        match binding.attribute_stride {
                            Some(stride) => unsafe {
                                command.setKernelBuffer_offset_attributeStride_atIndex(
                                    buffer,
                                    binding.offset as usize,
                                    stride as usize,
                                    binding.index as usize,
                                )
                            },
                            None => unsafe {
                                command.setKernelBuffer_offset_atIndex(
                                    buffer,
                                    binding.offset as usize,
                                    binding.index as usize,
                                )
                            },
                        }
                    }
                    if let Some(region) = host.stage_in_region {
                        let [x, y, z] = region.origin.map(|axis| axis as usize);
                        command.setStageInRegion(objc2_metal::MTLRegion {
                            origin: objc2_metal::MTLOrigin { x, y, z },
                            size: super::mtl_size(region.size),
                        });
                    }
                    for memory in &host.threadgroup_memory_lengths {
                        unsafe {
                            command.setThreadgroupMemoryLength_atIndex(
                                memory.length as usize,
                                memory.index as usize,
                            )
                        };
                    }
                    match host.action {
                        Some(HostAction::DispatchThreadgroups {
                            grid,
                            threads_per_threadgroup,
                        }) => command.concurrentDispatchThreadgroups_threadsPerThreadgroup(
                            super::mtl_size(grid),
                            super::mtl_size(threads_per_threadgroup),
                        ),
                        Some(HostAction::DispatchThreads {
                            grid,
                            threads_per_threadgroup,
                        }) => command.concurrentDispatchThreads_threadsPerThreadgroup(
                            super::mtl_size(grid),
                            super::mtl_size(threads_per_threadgroup),
                        ),
                        Some(
                            HostAction::DrawPrimitives { .. }
                            | HostAction::DrawIndexedPrimitives { .. }
                            | HostAction::DrawPatches { .. },
                        ) => return Err("a compute command draws nothing".into()),
                        None => {}
                    }
                    if host.barrier {
                        command.setBarrier();
                    }
                }
                CommandDescriptor::Render { .. } => {
                    let command =
                        unsafe { native.indirectRenderCommandAtIndex(host.index as usize) };
                    if let Some(token) = host.pipeline {
                        command.setRenderPipelineState(render_pipeline(&tokens, token)?);
                    }
                    for binding in &host.vertex_buffers {
                        let buffer = case_buffer(buffers, binding.buffer, "host command")?;
                        match binding.attribute_stride {
                            Some(stride) => unsafe {
                                command.setVertexBuffer_offset_attributeStride_atIndex(
                                    buffer,
                                    binding.offset as usize,
                                    stride as usize,
                                    binding.index as usize,
                                )
                            },
                            None => unsafe {
                                command.setVertexBuffer_offset_atIndex(
                                    buffer,
                                    binding.offset as usize,
                                    binding.index as usize,
                                )
                            },
                        }
                    }
                    for binding in &host.fragment_buffers {
                        let buffer = case_buffer(buffers, binding.buffer, "host command")?;
                        unsafe {
                            command.setFragmentBuffer_offset_atIndex(
                                buffer,
                                binding.offset as usize,
                                binding.index as usize,
                            )
                        };
                    }
                    match host.action {
                        Some(HostAction::DrawPrimitives {
                            primitive,
                            vertex_start,
                            vertex_count,
                            instance_count,
                            base_instance,
                        }) => unsafe {
                            command
                                .drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
                                    super::metal_primitive(primitive),
                                    vertex_start as usize,
                                    vertex_count as usize,
                                    instance_count as usize,
                                    base_instance as usize,
                                )
                        },
                        Some(HostAction::DrawIndexedPrimitives {
                            primitive,
                            index_count,
                            index_type,
                            index_buffer,
                            index_buffer_offset,
                            instance_count,
                            base_vertex,
                            base_instance,
                        }) => {
                            // The literal checked the buffer, the offset's alignment and the
                            // indices inside it, and the counts.
                            let buffer = case_buffer(buffers, index_buffer, "index")?;
                            unsafe {
                                command.drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset_instanceCount_baseVertex_baseInstance(
                                    super::metal_primitive(primitive),
                                    index_count as usize,
                                    match index_type {
                                        crate::case::IndexType::Uint16 => {
                                            objc2_metal::MTLIndexType::UInt16
                                        }
                                        crate::case::IndexType::Uint32 => {
                                            objc2_metal::MTLIndexType::UInt32
                                        }
                                    },
                                    buffer,
                                    index_buffer_offset as usize,
                                    instance_count as usize,
                                    base_vertex as isize,
                                    base_instance as usize,
                                )
                            }
                        }
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
                        }) => {
                            // The literal checked the buffers, the offsets' and the stride's
                            // alignment, the patch indices and the factors inside their buffers,
                            // and the counts.
                            let patch_indices = patch_index_buffer
                                .map(|index| case_buffer(buffers, index, "patch index"))
                                .transpose()?;
                            let factors = case_buffer(buffers, factor_buffer, "factor")?;
                            unsafe {
                                command.drawPatches_patchStart_patchCount_patchIndexBuffer_patchIndexBufferOffset_instanceCount_baseInstance_tessellationFactorBuffer_tessellationFactorBufferOffset_tessellationFactorBufferInstanceStride(
                                    control_points as usize,
                                    patch_start as usize,
                                    patch_count as usize,
                                    patch_indices.map(|buffer| &**buffer),
                                    patch_index_buffer_offset as usize,
                                    instance_count as usize,
                                    base_instance as usize,
                                    factors,
                                    factor_buffer_offset as usize,
                                    factor_instance_stride as usize,
                                )
                            }
                        }
                        Some(_) => return Err("a render command dispatches nothing".into()),
                        None => {}
                    }
                    if host.barrier {
                        return Err("a host-encoded render command's barrier is not modeled".into());
                    }
                }
            }
        }
    }
    Ok(())
}

/// Execute each command buffer the case executes, after the producer completed and its policy
/// readback passed, in key order
/// ([`executed_in_key_order`](crate::case::indirect_commands::executed_in_key_order)), each in
/// its own command buffer, waited for before the next: compute commands in a compute encoder,
/// render commands in a render encoder of a pass whose one attachment is the oracle's own 1x1
/// `R8Unorm` sink (`make_rasterization_sink`, shared storage; the checker admits only
/// rasterization-disabled vertex and mesh templates, whose pipelines name that attachment and draw
/// nothing into it). The encoder takes the parent pipeline and buffers the execution states, makes
/// every buffer of the case and every argument-buffer buffer resident for reading and writing
/// (`useResource:usage:`, for the vertex and fragment stages of a render encoder, and the object
/// and mesh stages too where its descriptor allows mesh draws), and calls
/// `executeCommandsInBuffer:withRange:` over the execution range.
pub(super) fn execute_host_commands(
    device: &ProtocolObject<dyn MTLDevice>,
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
    case: &AuthoredCase,
    commands: &[IndirectCommands],
    buffers: &[(u32, Buffer)],
    argument_buffer_buffers: &[super::argument_buffers::ArgumentBufferBuffer],
    pipelines: &BTreeMap<String, NativePipeline>,
) -> Result<(), String> {
    use crate::case::indirect_commands::CommandDescriptor;
    use objc2_metal::{
        MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
        MTLLoadAction, MTLRenderCommandEncoder, MTLRenderPassDescriptor, MTLRenderStages,
        MTLResourceUsage, MTLStoreAction,
    };
    if case
        .command_buffers
        .iter()
        .all(|resource| resource.execute.is_none())
    {
        return Ok(());
    }
    let tokens = token_pipelines(case, pipelines)?;
    let resident = || {
        buffers
            .iter()
            .map(|(_, buffer)| buffer)
            .chain(argument_buffer_buffers.iter().map(|(_, buffer)| buffer))
    };
    let usage = MTLResourceUsage::Read | MTLResourceUsage::Write;
    // In key order, as the case's identity and the candidate's replay order them.
    for resource in crate::case::indirect_commands::executed_in_key_order(&case.command_buffers) {
        let Some(execution) = &resource.execute else {
            continue;
        };
        let key = (resource.buffer_binding, resource.field_offset);
        let native = &commands
            .iter()
            .find(|(owner, _)| *owner == key)
            .ok_or_else(|| format!("command buffer {key:?} was not created"))?
            .1;
        let [start, end] = resource.execution_range;
        // The literal checked the range inside the capacity.
        let range = objc2_foundation::NSRange::new(start as usize, (end - start) as usize);
        let command_buffer = queue
            .commandBuffer()
            .ok_or_else(|| "MTLCommandQueue::commandBuffer returned nil".to_string())?;
        match resource.descriptor {
            CommandDescriptor::Compute { .. } => {
                let encoder = command_buffer
                    .computeCommandEncoder()
                    .ok_or_else(|| "computeCommandEncoder returned nil".to_string())?;
                if let Some(template) = &execution.pipeline {
                    encoder.setComputePipelineState(compute_pipeline(&tokens, template.token)?);
                }
                for binding in &execution.buffers {
                    let buffer = case_buffer(buffers, binding.buffer, "parent")?;
                    unsafe {
                        encoder.setBuffer_offset_atIndex(
                            Some(buffer),
                            binding.offset as usize,
                            binding.index as usize,
                        )
                    };
                }
                for buffer in resident() {
                    encoder.useResource_usage(ProtocolObject::from_ref(&**buffer), usage);
                }
                unsafe { encoder.executeCommandsInBuffer_withRange(native, range) };
                encoder.endEncoding();
            }
            CommandDescriptor::Render { draws, .. } => {
                let sink = super::attachments::make_rasterization_sink(device)?;
                let pass = MTLRenderPassDescriptor::renderPassDescriptor();
                let attachment = unsafe { pass.colorAttachments().objectAtIndexedSubscript(0) };
                attachment.setTexture(Some(&*sink));
                attachment.setLoadAction(MTLLoadAction::DontCare);
                attachment.setStoreAction(MTLStoreAction::DontCare);
                let encoder = command_buffer
                    .renderCommandEncoderWithDescriptor(&pass)
                    .ok_or_else(|| "renderCommandEncoderWithDescriptor returned nil".to_string())?;
                if let Some(template) = &execution.pipeline {
                    encoder.setRenderPipelineState(render_pipeline(&tokens, template.token)?);
                }
                for binding in &execution.vertex_buffers {
                    let buffer = case_buffer(buffers, binding.buffer, "parent")?;
                    unsafe {
                        encoder.setVertexBuffer_offset_atIndex(
                            Some(buffer),
                            binding.offset as usize,
                            binding.index as usize,
                        )
                    };
                }
                for binding in &execution.fragment_buffers {
                    let buffer = case_buffer(buffers, binding.buffer, "parent")?;
                    unsafe {
                        encoder.setFragmentBuffer_offset_atIndex(
                            Some(buffer),
                            binding.offset as usize,
                            binding.index as usize,
                        )
                    };
                }
                for binding in &execution.object_buffers {
                    let buffer = case_buffer(buffers, binding.buffer, "parent")?;
                    unsafe {
                        encoder.setObjectBuffer_offset_atIndex(
                            Some(buffer),
                            binding.offset as usize,
                            binding.index as usize,
                        )
                    };
                }
                for binding in &execution.mesh_buffers {
                    let buffer = case_buffer(buffers, binding.buffer, "parent")?;
                    unsafe {
                        encoder.setMeshBuffer_offset_atIndex(
                            Some(buffer),
                            binding.offset as usize,
                            binding.index as usize,
                        )
                    };
                }
                let stages = if draws.mesh() {
                    MTLRenderStages::Vertex
                        | MTLRenderStages::Fragment
                        | MTLRenderStages::Object
                        | MTLRenderStages::Mesh
                } else {
                    MTLRenderStages::Vertex | MTLRenderStages::Fragment
                };
                for buffer in resident() {
                    encoder.useResource_usage_stages(
                        ProtocolObject::from_ref(&**buffer),
                        usage,
                        stages,
                    );
                }
                unsafe { encoder.executeCommandsInBuffer_withRange(native, range) };
                encoder.endEncoding();
            }
        }
        crate::gpu::wait::metal::commit_and_wait(&command_buffer, "Metal indirect execution")?;
    }
    Ok(())
}
