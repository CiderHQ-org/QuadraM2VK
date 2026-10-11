//! A tile kernel's dispatch through the Metal-like runtime, as the Metal oracle dispatches it
//! (`metal::platform::tile`): the one copy the runtime lane's tile census
//! (`runtime_lane::tile_dispatches`) and the engine executor's runtime path (`runtime_run`)
//! share.
//!
//! A tile pipeline state of the oracle's descriptor ([`tile_descriptor`]: the tile function, the
//! functions it links, the pass's attachments' formats, one sample), the pass's attachments, each
//! a `Shared` 2D render target of its authored format, extent and contents that loads and stores,
//! in a pass of the case's area whose tile is the case's imageblock and whose
//! `imageblockSampleLength` is the state's, as the oracle asks it; the case's buffers, textures,
//! sampler states and visible-function tables set on the encoder's tile stage, one
//! `dispatch_threads_per_tile` of the case's threads per tile, and the output read back after the
//! command buffer completed. Its argument buffers are made as a compute dispatch's are
//! (`runtime_case::bound_buffers`), their buffers and textures resident for reading and writing
//! in the tile stage, as the oracle's `useResource:usage:stages:` makes them; its tables are made
//! by the tile state for the tile stage ([`tile_tables`]).
//!
//! The oracle draws a triangle over the pass first, whose fragment writes no color, so that
//! Metal's imageblock holds every pixel (`ImplicitImageblockCoverage::FullSingleSample`); it
//! changes no pixel, so the runtime path draws none. The oracle's threadgroup memory lengths are
//! not set: the compile layer refuses a tile function's threadgroup memory, so the state's creation
//! refuses a case whose function reads one. A tile dispatch's command buffers and the pipeline
//! states its argument buffers hold are refused as a [`CaseError::Case`] naming "a later runtime
//! step"; the executor's route sends no such case here (`runtime_run::kernel_prerequisites`).

use super::{
    bound_buffers, tables_of, texture_with, CaseError, CaseResources, CaseTables, Encoded,
    LinkedSet,
};
use crate::candidate::engine::{ComputeCase, TilePass};
use metal2vulkan::bind::Dispatch;
use metal2vulkan::metal::{RenderStages, ResourceUsage, TextureUsage};
use metal2vulkan::pipeline::{
    LinkedFunctions, Stage as ShaderStage, StageSource, TileRenderPipelineDescriptor,
};
use metal2vulkan_runtime::{
    Buffer, Device, Function, LoadAction, RenderPassDescriptor, RenderPipelineState, ResourceRef,
    Size, StoreAction, Texture, VisibleFunctionTable,
};
use std::collections::BTreeMap;

/// The descriptor of a tile kernel's pipeline state over `pass`, as the Metal oracle states it:
/// its tile `function` (made with the case's constant values) linking `linked`, each attachment's
/// pixel format at its color index, one sample, and `threadgroupSizeMatchesTileSize` off. The
/// oracle's `requiredThreadsPerThreadgroup` sizes Metal's pipeline for the case's threads; the
/// runtime's tile pipeline runs every measured threadgroup up to 32 by 32, so the descriptor states
/// none.
pub fn tile_descriptor(
    function: &Function,
    linked: LinkedFunctions,
    pass: &TilePass,
) -> TileRenderPipelineDescriptor {
    let mut descriptor = TileRenderPipelineDescriptor::new(StageSource::Function(
        function.function_descriptor().with_linked(linked),
    ));
    for (index, format) in pass.formats() {
        if let Some(slot) = descriptor.color_attachment_formats.get_mut(index as usize) {
            *slot = Some(format);
        }
    }
    descriptor
}

/// The visible-function tables of tile kernel `case` for its tile state `pipeline`, as
/// `runtime_case::visible_function_tables` makes a kernel's: each made by the state for the tile
/// stage and each entry the handle the state gives the function it holds among `linked` for that
/// stage, as the oracle makes them (`newVisibleFunctionTableWithDescriptor:stage:`); the bytes
/// keyed by the kernel stage, as the executor keys a tile kernel's.
pub fn tile_tables(
    pipeline: &RenderPipelineState,
    case: &ComputeCase,
    linked: &LinkedSet,
) -> Result<CaseTables, CaseError> {
    tables_of(
        (ShaderStage::Kernel, &case.function_tables),
        linked,
        |descriptor| pipeline.new_visible_function_table(descriptor, RenderStages::TILE),
        |function| pipeline.function_handle(function, RenderStages::TILE),
    )
}

/// Encode tile kernel `case`'s dispatch on `device` with `pipeline` (the tile pipeline state of
/// its entry, [`tile_descriptor`]), its visible-function `tables` ([`tile_tables`]), and the
/// textures' `usage` and `argument_usage` ([`CaseResources::of`]): its resources holding their
/// bytes, its pass's attachments holding theirs, one command buffer with the fills of its
/// multisampled textures, then one render encoder of the pass and its one tile dispatch (the
/// module's rule), ended. [`Encoded::commit`] runs it.
pub fn encoded_tile(
    device: &Device,
    case: &ComputeCase,
    usage: &BTreeMap<u32, TextureUsage>,
    argument_usage: &BTreeMap<(u32, u32), TextureUsage>,
    pipeline: &RenderPipelineState,
    tables: BTreeMap<u32, VisibleFunctionTable>,
) -> Result<Encoded, CaseError> {
    let pass = case.tile.as_ref().ok_or("the case runs in no tile pass")?;
    if !case.command_buffers.is_empty() || !case.objects.is_empty() {
        return Err(CaseError::Case(
            "a tile dispatch's command buffers and pipeline states: a later runtime step"
                .to_owned(),
        ));
    }
    let Dispatch::Threadgroups {
        threads_per_group, ..
    } = case.dispatch
    else {
        return Err(format!("a tile pass's dispatch {:?}", case.dispatch).into());
    };
    let [tile_width, tile_height] = case.imageblock.ok_or("a tile pass without a tile size")?;
    let mut resources = CaseResources::of(device, case, usage, argument_usage)?;
    resources.tables = tables;
    let buffers = bound_buffers(device, case, &resources)?;
    let attachments = pass
        .attachments
        .iter()
        .map(|(&index, literal)| {
            Ok((
                index,
                texture_with(device, literal, TextureUsage::RENDER_TARGET)?,
            ))
        })
        .collect::<Result<BTreeMap<_, _>, CaseError>>()?;
    let mut descriptor = RenderPassDescriptor::new();
    for (&index, texture) in &attachments {
        let attachment = descriptor
            .color_attachments
            .get_mut(index as usize)
            .ok_or_else(|| format!("a tile pass's attachment {index}"))?;
        attachment.texture = Some(texture.clone());
        attachment.load_action = LoadAction::Load;
        attachment.store_action = StoreAction::Store;
    }
    descriptor.tile_width = tile_width.into();
    descriptor.tile_height = tile_height.into();
    descriptor.imageblock_sample_length = pipeline
        .imageblock_sample_length()
        .map_err(CaseError::runtime("imageblock_sample_length"))?;
    descriptor.render_target_width = pass.area[0].into();
    descriptor.render_target_height = pass.area[1].into();
    let queue = device
        .new_command_queue()
        .map_err(CaseError::runtime("new_command_queue"))?;
    let command_buffer = queue
        .command_buffer()
        .map_err(CaseError::runtime("command_buffer"))?;
    for fill in &resources.fills {
        fill.encode(&command_buffer)?;
    }
    let filled = command_buffer.executions().len();
    let mut encoder = command_buffer
        .render_command_encoder(&descriptor)
        .map_err(CaseError::runtime("render_command_encoder"))?;
    encoder
        .set_render_pipeline_state(pipeline)
        .map_err(CaseError::runtime("set_render_pipeline_state"))?;
    for (&index, buffer) in &buffers {
        encoder
            .set_tile_buffer(Some(buffer), 0, index)
            .map_err(CaseError::runtime(format!("set_tile_buffer {index}")))?;
    }
    for (&index, table) in &resources.tables {
        encoder
            .set_tile_visible_function_table(Some(table), index)
            .map_err(CaseError::runtime(format!(
                "set_tile_visible_function_table {index}"
            )))?;
    }
    // The argument buffers' members, resident for reading and writing in the tile stage, as the
    // oracle's `useResource:usage:stages:` makes them.
    let members: Vec<ResourceRef<'_>> = resources
        .argument_buffers
        .values()
        .map(Buffer::as_resource)
        .chain(
            resources
                .argument_textures
                .values()
                .map(Texture::as_resource),
        )
        .collect();
    if !members.is_empty() {
        encoder
            .use_resources(
                &members,
                ResourceUsage::READ | ResourceUsage::WRITE,
                RenderStages::TILE,
            )
            .map_err(CaseError::runtime("use_resources"))?;
    }
    for (&index, texture) in &resources.textures {
        encoder
            .set_tile_texture(Some(texture), index)
            .map_err(CaseError::runtime(format!("set_tile_texture {index}")))?;
    }
    for (&index, sampler) in &resources.samplers {
        encoder
            .set_tile_sampler_state(Some(sampler), index)
            .map_err(CaseError::runtime(format!(
                "set_tile_sampler_state {index}"
            )))?;
    }
    let [width, height, depth] = threads_per_group.map(u64::from);
    encoder
        .dispatch_threads_per_tile(Size::new(width, height, depth))
        .map_err(CaseError::runtime("dispatch_threads_per_tile"))?;
    encoder
        .end_encoding()
        .map_err(CaseError::runtime("end_encoding"))?;
    // Each ended encoder's executions join the command buffer's in encoding order.
    let executions = command_buffer.executions().split_off(filled);
    Ok(Encoded {
        command_buffer,
        buffers,
        resources,
        attachments,
        executions,
    })
}
