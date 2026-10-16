//! A render encoder's log, which a rebind encodes it again from (`replay.rs`, the compute and
//! blit encoders' alike): the pass's descriptor as the program gave it, and each public call with
//! its arguments owned, in order, before the call runs.
//!
//! - **Encoded again** by opening the pass again from the descriptor
//!   ([`RenderCommandEncoder::replaying`]: no refusal of a committed command buffer, the residency
//!   the encoders ended before it left), running each call again, and ending the pass as
//!   `end_encoding` ends it, but into the ended encoder's place
//!   ([`RenderCommandEncoder::finish_replay`]).
//! - **Kept** as the compute encoder's log is (`EncoderCore::end_with_replay`): where a command
//!   named a buffer over host pages, and the pass left no segment for later (no indirect command
//!   buffer's execution, no draw whose pipeline was still being made). A pass whose draws' tile
//!   reach its end captures (`capture.rs`) keeps none: the capture's tally lies in the command
//!   buffer, where an encoding again would leave the first one's beside its own.
//! - **Per-stage calls** share a variant, with the [`Table`] they set.

use super::RenderCommandEncoder;
use crate::amplification::VertexAmplificationViewMapping;
use crate::buffer::Buffer;
use crate::command_buffer::CommandBuffer;
use crate::encoder::Replayed;
use crate::error::Error;
use crate::fence::Fence;
use crate::function_table::VisibleFunctionTable;
use crate::heap::Heap;
use crate::indirect_command_buffer::IndirectCommandBuffer;
use crate::render::ResolveTarget;
use crate::render_pass::RenderPassDescriptor;
use crate::render_values::{ScissorRect, StoreAction, StoreActionOptions, Viewport};
use crate::replay::{CallLog, Held, Resource, Span, SPARE_LOG_BYTES};
use crate::residency::Residency;
use crate::sampler::SamplerState;
use crate::values::{Region, Size};
use crate::visibility::VisibilityResultMode;
use metal2vulkan::metal::{
    BarrierScope, CullMode, DepthClipMode, IndexType, PrimitiveType, RenderStages, ResourceUsage,
    TriangleFillMode, Winding,
};
use std::ops::Range;

/// A render encoder's log: the descriptor its pass was opened with, and its calls.
pub(crate) struct RenderLog {
    descriptor: Box<RenderPassDescriptor>,
    calls: CallLog<RenderCall>,
    /// The arguments of its `draw_indexed_primitives` calls, in order
    /// (`RenderCall::DrawIndexedPrimitives`): in one list, not a box each.
    indexed: Vec<DrawIndexed>,
}

/// A cleared render log's lists, their room kept for a next pass's log (`SpareLogs`).
pub(crate) struct RenderRoom {
    calls: CallLog<RenderCall>,
    indexed: Vec<DrawIndexed>,
}

impl RenderLog {
    /// The log of a pass `descriptor` opens, before any call: in `room` where one is kept.
    pub(crate) fn new(descriptor: &RenderPassDescriptor, room: Option<RenderRoom>) -> RenderLog {
        let (calls, indexed) = match room {
            Some(RenderRoom { calls, indexed }) => (calls, indexed),
            None => (CallLog::new(), Vec::new()),
        };
        RenderLog {
            descriptor: Box::new(descriptor.clone()),
            calls,
            indexed,
        }
    }

    /// Its lists emptied for a next pass's log, its descriptor, calls and handles let go of:
    /// `None` where their room is more than `SPARE_LOG_BYTES` (`CallLog::cleared`).
    pub(crate) fn cleared(self: Box<RenderLog>) -> Option<RenderRoom> {
        let RenderLog {
            descriptor,
            calls,
            mut indexed,
        } = *self;
        drop(descriptor);
        if indexed.capacity() * std::mem::size_of::<DrawIndexed>() > SPARE_LOG_BYTES {
            return None;
        }
        indexed.clear();
        Some(RenderRoom {
            calls: calls.cleared()?,
            indexed,
        })
    }

    /// The bytes it holds.
    pub(crate) fn bytes(&self) -> usize {
        std::mem::size_of::<RenderPassDescriptor>()
            + self.calls.bytes()
            + std::mem::size_of_val(&*self.indexed)
    }

    /// The descriptor its pass was opened with.
    pub(crate) fn descriptor(&self) -> &RenderPassDescriptor {
        &self.descriptor
    }

    /// Run its calls again on `encoder`, a later color group's of its pass
    /// (`render/color_groups.rs`). A call refused there was refused in the first group's
    /// encoder, whose answer the program had, or is a divergence the later group's encoder notes
    /// (`RenderCommandEncoder::group_outcome`): either way the calls after it run.
    pub(crate) fn replay_into(&self, encoder: &mut RenderCommandEncoder) {
        for call in &self.calls.calls {
            let _ = render(encoder, (&self.calls, &self.indexed), call);
        }
    }

    /// Encode its encoder again into `command_buffer`, starting with `residency` (`Replay::run`).
    pub(crate) fn run(
        &self,
        command_buffer: &CommandBuffer,
        residency: Residency,
    ) -> Result<(Replayed, Option<Residency>), Error> {
        let mut encoder =
            RenderCommandEncoder::replaying(command_buffer, &self.descriptor, residency)?;
        for call in &self.calls.calls {
            render(&mut encoder, (&self.calls, &self.indexed), call)?;
        }
        encoder.finish_replay()
    }
}

/// The stage whose argument table a call sets.
#[derive(Clone, Copy)]
pub(crate) enum Table {
    Vertex,
    Fragment,
    Object,
    Mesh,
    Tile,
}

/// One public call of a render encoder, its arguments owned (the method of each variant's name;
/// a per-stage method's with its [`Table`]).
pub(crate) enum RenderCall {
    PipelineState(Held),
    Buffer(Table, Option<Held>, u64, u32),
    Buffers(Table, Span, Span, u32),
    BufferOffset(Table, u64, u32),
    Bytes(Table, Span, u32),
    Texture(Table, Option<Held>, u32),
    Textures(Table, Span, u32),
    Sampler(Table, Option<Held>, u32),
    Samplers(Table, Box<Samplers>),
    FunctionTable(Table, Option<VisibleFunctionTable>, u32),
    FunctionTables(Table, Box<[Option<VisibleFunctionTable>]>, u32),
    VertexBufferWithStride(Option<Held>, u64, u64, u32),
    VertexBuffersWithStrides(Box<(Span, Span, Span, u32)>),
    VertexBufferOffsetWithStride(u64, u64, u32),
    VertexBytesWithStride(Span, u64, u32),
    ColorStoreAction(StoreAction, u32),
    DepthStoreAction(StoreAction),
    StencilStoreAction(StoreAction),
    ColorStoreActionOptions(StoreActionOptions, u32),
    DepthStoreActionOptions(StoreActionOptions),
    StencilStoreActionOptions(StoreActionOptions),
    Viewport(Box<Viewport>),
    Viewports(Box<[Viewport]>),
    ScissorRect(ScissorRect),
    ScissorRects(Box<[ScissorRect]>),
    CullMode(CullMode),
    FrontFacingWinding(Winding),
    DepthStencilState(Option<Held>),
    StencilReferenceValue(u32),
    StencilReferenceValues(u32, u32),
    DepthBias(f32, f32, f32),
    DepthClipMode(DepthClipMode),
    TriangleFillMode(TriangleFillMode),
    BlendColor([f32; 4]),
    TessellationFactorBuffer(Option<Buffer>, u64, u64),
    TessellationFactorScale(f32),
    VertexAmplificationCount(u64, Option<Box<[VertexAmplificationViewMapping]>>),
    DrawPrimitives(PrimitiveType, u32, u32, u32, u32),
    /// Its arguments' place in the log's list of them (`RenderLog::indexed`).
    DrawIndexedPrimitives(u32),
    DrawPrimitivesIndirect(PrimitiveType, Buffer, u64),
    DrawIndexedPrimitivesIndirect(Box<DrawIndexedIndirect>),
    DrawPatches(Box<DrawPatches>),
    DrawPatchesIndirect(Box<DrawPatchesIndirect>),
    DrawIndexedPatches(Box<DrawIndexedPatches>),
    DrawIndexedPatchesIndirect(Box<DrawIndexedPatchesIndirect>),
    VisibilityResultMode(VisibilityResultMode, u64),
    UseResource(Resource, ResourceUsage, RenderStages),
    UseResources(Box<[Resource]>, ResourceUsage, RenderStages),
    UseHeap(Heap, RenderStages),
    UseHeaps(Box<[Heap]>, RenderStages),
    Execute(IndirectCommandBuffer, Range<u64>),
    ExecuteIndirect(IndirectCommandBuffer, Buffer, u64),
    ColorResolveTarget(Option<Box<ResolveTarget>>, u32),
    DepthResolveTarget(Option<Box<ResolveTarget>>),
    StencilResolveTarget(Option<Box<ResolveTarget>>),
    MeshThreadgroups(Box<(Size, Size, Size)>),
    MeshThreads(Box<(Size, Size, Size)>),
    MeshThreadgroupsIndirect(Box<(Buffer, u64, Size, Size)>),
    PrimitiveRestartEnabled(bool, u64),
    ThreadsPerTile(Box<Size>),
    ThreadsPerTileInRegion(Box<(Size, Region, Option<u32>)>),
    ThreadgroupMemoryLength(u64, u64, u32),
    ObjectThreadgroupMemoryLength(u32, u32),
    UpdateFence(Fence, RenderStages),
    WaitForFence(Fence, RenderStages),
    BarrierWithScope(BarrierScope, RenderStages, RenderStages),
    BarrierWithResources(Box<[Resource]>, RenderStages, RenderStages),
    AlphaTestReferenceValue(f32),
    LineWidth(f32),
    PointSize(f32),
    ProvokingVertexMode(u64),
    ViewportTransformEnabled(bool),
    VertexAmplificationMode(u64, u64),
    DepthCleared,
    StencilCleared,
}

impl RenderCall {
    /// The bytes it holds in boxes of its own, besides its size.
    pub(super) fn extra(&self) -> usize {
        use std::mem::{size_of, size_of_val};
        use RenderCall as C;
        match self {
            C::Samplers(_, call) => {
                size_of::<Samplers>()
                    + size_of_val(&*call.samplers)
                    + call.clamps.as_ref().map_or(0, |(mins, maxes)| {
                        size_of_val(&**mins) + size_of_val(&**maxes)
                    })
            }
            C::FunctionTables(_, tables, _) => size_of_val(&**tables),
            C::VertexBuffersWithStrides(call) => size_of_val(&**call),
            C::Viewport(viewport) => size_of_val(&**viewport),
            C::Viewports(viewports) => size_of_val(&**viewports),
            C::ScissorRects(rects) => size_of_val(&**rects),
            C::VertexAmplificationCount(_, mappings) => mappings
                .as_ref()
                .map_or(0, |mappings| size_of_val(&**mappings)),
            C::DrawIndexedPrimitivesIndirect(call) => size_of_val(&**call),
            C::DrawPatches(call) => size_of_val(&**call),
            C::DrawPatchesIndirect(call) => size_of_val(&**call),
            C::DrawIndexedPatches(call) => size_of_val(&**call),
            C::DrawIndexedPatchesIndirect(call) => size_of_val(&**call),
            C::UseResources(resources, _, _) | C::BarrierWithResources(resources, _, _) => {
                size_of_val(&**resources)
            }
            C::UseHeaps(heaps, _) => size_of_val(&**heaps),
            C::ColorResolveTarget(target, _)
            | C::DepthResolveTarget(target)
            | C::StencilResolveTarget(target) => target.as_ref().map_or(0, |t| size_of_val(&**t)),
            C::MeshThreadgroups(call) | C::MeshThreads(call) => size_of_val(&**call),
            C::MeshThreadgroupsIndirect(call) => size_of_val(&**call),
            C::ThreadsPerTile(threads) => size_of_val(&**threads),
            C::ThreadsPerTileInRegion(call) => size_of_val(&**call),
            _ => 0,
        }
    }
}

/// A `set_*_sampler_states` call's arguments.
pub(crate) struct Samplers {
    pub(super) samplers: Box<[Option<SamplerState>]>,
    pub(super) clamps: Option<(Box<[f32]>, Box<[f32]>)>,
    pub(super) first: u32,
}

impl Samplers {
    /// `samplers`, `clamps` and `first`, owned.
    pub(super) fn of(
        samplers: &[Option<&SamplerState>],
        clamps: Option<(&[f32], &[f32])>,
        first: u32,
    ) -> Box<Samplers> {
        Box::new(Samplers {
            samplers: samplers.iter().map(|s| s.cloned()).collect(),
            clamps: clamps.map(|(mins, maxes)| (mins.into(), maxes.into())),
            first,
        })
    }
}

/// `draw_indexed_primitives`'s arguments.
pub(crate) struct DrawIndexed {
    pub(super) primitive_type: PrimitiveType,
    pub(super) index_count: u32,
    pub(super) index_type: IndexType,
    pub(super) index_buffer: Held,
    pub(super) index_buffer_offset: u64,
    pub(super) instance_count: u32,
    pub(super) base_vertex: i32,
    pub(super) base_instance: u32,
}

/// `draw_indexed_primitives_indirect`'s arguments.
pub(crate) struct DrawIndexedIndirect {
    pub(super) primitive_type: PrimitiveType,
    pub(super) index_type: IndexType,
    pub(super) index_buffer: Buffer,
    pub(super) index_buffer_offset: u64,
    pub(super) indirect_buffer: Buffer,
    pub(super) indirect_buffer_offset: u64,
}

/// `draw_patches`'s arguments.
pub(crate) struct DrawPatches {
    pub(super) control_points: u32,
    pub(super) patch_start: u32,
    pub(super) patch_count: u32,
    pub(super) patch_index_buffer: Option<Buffer>,
    pub(super) patch_index_buffer_offset: u64,
    pub(super) instance_count: u32,
    pub(super) base_instance: u32,
}

/// `draw_patches_indirect`'s arguments.
pub(crate) struct DrawPatchesIndirect {
    pub(super) control_points: u32,
    pub(super) patch_index_buffer: Option<Buffer>,
    pub(super) patch_index_buffer_offset: u64,
    pub(super) indirect_buffer: Buffer,
    pub(super) indirect_buffer_offset: u64,
}

/// `draw_indexed_patches`'s arguments.
pub(crate) struct DrawIndexedPatches {
    pub(super) control_points: u32,
    pub(super) patch_start: u32,
    pub(super) patch_count: u32,
    pub(super) patch_index_buffer: Option<Buffer>,
    pub(super) patch_index_buffer_offset: u64,
    pub(super) control_point_index_buffer: Buffer,
    pub(super) control_point_index_buffer_offset: u64,
    pub(super) instance_count: u32,
    pub(super) base_instance: u32,
}

/// `draw_indexed_patches_indirect`'s arguments.
pub(crate) struct DrawIndexedPatchesIndirect {
    pub(super) control_points: u32,
    pub(super) patch_index_buffer: Option<Buffer>,
    pub(super) patch_index_buffer_offset: u64,
    pub(super) control_point_index_buffer: Buffer,
    pub(super) control_point_index_buffer_offset: u64,
    pub(super) indirect_buffer: Buffer,
    pub(super) indirect_buffer_offset: u64,
}

impl RenderCommandEncoder {
    /// Log the call `call` makes of the log, before it runs, where the encoder logs (the module
    /// docs).
    pub(crate) fn log(&mut self, call: impl FnOnce(&mut CallLog<RenderCall>) -> RenderCall) {
        if let Some(log) = &mut self.log {
            let call = call(&mut log.calls);
            let extra = call.extra();
            log.calls.push(call, extra);
        }
    }

    /// Log a `draw_indexed_primitives` call, whose arguments `draw` makes of the log, before it
    /// runs, where the encoder logs: its arguments go into the log's list of them.
    pub(crate) fn log_indexed(
        &mut self,
        draw: impl FnOnce(&mut CallLog<RenderCall>) -> DrawIndexed,
    ) {
        if let Some(log) = &mut self.log {
            let draw = draw(&mut log.calls);
            // Fewer than the log's calls, each of 24 bytes or more.
            let at = log.indexed.len() as u32;
            log.indexed.push(draw);
            let call = RenderCall::DrawIndexedPrimitives(at);
            log.calls.push(call, std::mem::size_of::<DrawIndexed>());
        }
    }
}

/// Run `call` again on `encoder`, its slices in `log`'s arenas and an indexed draw's arguments in
/// `indexed`.
fn render(
    encoder: &mut RenderCommandEncoder,
    (log, indexed): (&CallLog<RenderCall>, &[DrawIndexed]),
    call: &RenderCall,
) -> Result<(), Error> {
    use RenderCall as C;
    use Table as T;
    match call {
        C::PipelineState(state) => encoder.set_render_pipeline_state(log.render_state(*state)),
        C::Buffer(table, buffer, offset, index) => {
            let buffer = buffer.map(|held| log.buffer(held));
            match table {
                T::Vertex => encoder.set_vertex_buffer(buffer, *offset, *index),
                T::Fragment => encoder.set_fragment_buffer(buffer, *offset, *index),
                T::Object => encoder.set_object_buffer(buffer, *offset, *index),
                T::Mesh => encoder.set_mesh_buffer(buffer, *offset, *index),
                T::Tile => encoder.set_tile_buffer(buffer, *offset, *index),
            }
        }
        C::Buffers(table, buffers, offsets, first) => {
            let (buffers, offsets) = (log.buffers_of(*buffers), log.words_of(*offsets));
            match table {
                T::Vertex => encoder.set_vertex_buffers(&buffers, offsets, *first),
                T::Fragment => encoder.set_fragment_buffers(&buffers, offsets, *first),
                T::Object => encoder.set_object_buffers(&buffers, offsets, *first),
                T::Mesh => encoder.set_mesh_buffers(&buffers, offsets, *first),
                T::Tile => encoder.set_tile_buffers(&buffers, offsets, *first),
            }
        }
        C::BufferOffset(table, offset, index) => match table {
            T::Vertex => encoder.set_vertex_buffer_offset(*offset, *index),
            T::Fragment => encoder.set_fragment_buffer_offset(*offset, *index),
            T::Object => encoder.set_object_buffer_offset(*offset, *index),
            T::Mesh => encoder.set_mesh_buffer_offset(*offset, *index),
            T::Tile => encoder.set_tile_buffer_offset(*offset, *index),
        },
        C::Bytes(table, bytes, index) => {
            let bytes = log.bytes_of(*bytes);
            match table {
                T::Vertex => encoder.set_vertex_bytes(bytes, *index),
                T::Fragment => encoder.set_fragment_bytes(bytes, *index),
                T::Object => encoder.set_object_bytes(bytes, *index),
                T::Mesh => encoder.set_mesh_bytes(bytes, *index),
                T::Tile => encoder.set_tile_bytes(bytes, *index),
            }
        }
        C::Texture(table, texture, index) => {
            let texture = texture.map(|held| log.texture(held));
            match table {
                T::Vertex => encoder.set_vertex_texture(texture, *index),
                T::Fragment => encoder.set_fragment_texture(texture, *index),
                T::Object => encoder.set_object_texture(texture, *index),
                T::Mesh => encoder.set_mesh_texture(texture, *index),
                T::Tile => encoder.set_tile_texture(texture, *index),
            }
        }
        C::Textures(table, textures, first) => {
            let textures = log.textures_of(*textures);
            match table {
                T::Vertex => encoder.set_vertex_textures(&textures, *first),
                T::Fragment => encoder.set_fragment_textures(&textures, *first),
                T::Object => encoder.set_object_textures(&textures, *first),
                T::Mesh => encoder.set_mesh_textures(&textures, *first),
                T::Tile => encoder.set_tile_textures(&textures, *first),
            }
        }
        C::Sampler(table, sampler, index) => {
            let sampler = sampler.map(|held| log.sampler(held));
            match table {
                T::Vertex => encoder.set_vertex_sampler_state(sampler, *index),
                T::Fragment => encoder.set_fragment_sampler_state(sampler, *index),
                T::Object => encoder.set_object_sampler_state(sampler, *index),
                T::Mesh => encoder.set_mesh_sampler_state(sampler, *index),
                T::Tile => encoder.set_tile_sampler_state(sampler, *index),
            }
        }
        C::Samplers(table, call) => {
            let samplers: Vec<_> = call.samplers.iter().map(Option::as_ref).collect();
            let clamps = call
                .clamps
                .as_ref()
                .map(|(mins, maxes)| (&mins[..], &maxes[..]));
            let first = call.first;
            match table {
                T::Vertex => encoder.set_vertex_sampler_states(&samplers, clamps, first),
                T::Fragment => encoder.set_fragment_sampler_states(&samplers, clamps, first),
                T::Object => encoder.set_object_sampler_states(&samplers, clamps, first),
                T::Mesh => encoder.set_mesh_sampler_states(&samplers, clamps, first),
                T::Tile => encoder.set_tile_sampler_states(&samplers, clamps, first),
            }
        }
        C::FunctionTable(table, function_table, index) => {
            let function_table = function_table.as_ref();
            match table {
                T::Vertex => encoder.set_vertex_visible_function_table(function_table, *index),
                T::Fragment => encoder.set_fragment_visible_function_table(function_table, *index),
                T::Tile => encoder.set_tile_visible_function_table(function_table, *index),
                T::Object | T::Mesh => Err(crate::error::internal(
                    "a logged visible function table of an object or mesh stage",
                )),
            }
        }
        C::FunctionTables(table, tables, first) => {
            let tables: Vec<_> = tables.iter().map(Option::as_ref).collect();
            match table {
                T::Vertex => encoder.set_vertex_visible_function_tables(&tables, *first),
                T::Fragment => encoder.set_fragment_visible_function_tables(&tables, *first),
                T::Tile => encoder.set_tile_visible_function_tables(&tables, *first),
                T::Object | T::Mesh => Err(crate::error::internal(
                    "logged visible function tables of an object or mesh stage",
                )),
            }
        }
        C::VertexBufferWithStride(buffer, offset, stride, index) => {
            let buffer = buffer.map(|held| log.buffer(held));
            encoder.set_vertex_buffer_with_stride(buffer, *offset, *stride, *index)
        }
        C::VertexBuffersWithStrides(call) => {
            let (buffers, offsets, strides, first) = **call;
            let buffers = log.buffers_of(buffers);
            let (offsets, strides) = (log.words_of(offsets), log.words_of(strides));
            encoder.set_vertex_buffers_with_strides(&buffers, offsets, strides, first)
        }
        C::VertexBufferOffsetWithStride(offset, stride, index) => {
            encoder.set_vertex_buffer_offset_with_stride(*offset, *stride, *index)
        }
        C::VertexBytesWithStride(bytes, stride, index) => {
            encoder.set_vertex_bytes_with_stride(log.bytes_of(*bytes), *stride, *index)
        }
        C::ColorStoreAction(action, index) => encoder.set_color_store_action(*action, *index),
        C::DepthStoreAction(action) => encoder.set_depth_store_action(*action),
        C::StencilStoreAction(action) => encoder.set_stencil_store_action(*action),
        C::ColorStoreActionOptions(options, index) => {
            encoder.set_color_store_action_options(*options, *index)
        }
        C::DepthStoreActionOptions(options) => encoder.set_depth_store_action_options(*options),
        C::StencilStoreActionOptions(options) => encoder.set_stencil_store_action_options(*options),
        C::Viewport(viewport) => encoder.set_viewport(**viewport),
        C::Viewports(viewports) => encoder.set_viewports(viewports),
        C::ScissorRect(rect) => encoder.set_scissor_rect(*rect),
        C::ScissorRects(rects) => encoder.set_scissor_rects(rects),
        C::CullMode(mode) => {
            encoder.set_cull_mode(*mode);
            Ok(())
        }
        C::FrontFacingWinding(winding) => {
            encoder.set_front_facing_winding(*winding);
            Ok(())
        }
        C::DepthStencilState(state) => {
            encoder.set_depth_stencil_state(state.map(|held| log.depth_stencil(held)))
        }
        C::StencilReferenceValue(value) => {
            encoder.set_stencil_reference_value(*value);
            Ok(())
        }
        C::StencilReferenceValues(front, back) => {
            encoder.set_stencil_reference_values(*front, *back);
            Ok(())
        }
        C::DepthBias(bias, slope, clamp) => encoder.set_depth_bias(*bias, *slope, *clamp),
        C::DepthClipMode(mode) => encoder.set_depth_clip_mode(*mode),
        C::TriangleFillMode(mode) => encoder.set_triangle_fill_mode(*mode),
        C::BlendColor([red, green, blue, alpha]) => {
            encoder.set_blend_color(*red, *green, *blue, *alpha);
            Ok(())
        }
        C::TessellationFactorBuffer(buffer, offset, stride) => {
            encoder.set_tessellation_factor_buffer(buffer.as_ref(), *offset, *stride)
        }
        C::TessellationFactorScale(scale) => {
            encoder.set_tessellation_factor_scale(*scale);
            Ok(())
        }
        C::VertexAmplificationCount(count, mappings) => {
            encoder.set_vertex_amplification_count(*count, mappings.as_deref())
        }
        C::DrawPrimitives(primitive, start, count, instances, base) => {
            encoder.draw_primitives(*primitive, *start, *count, *instances, *base)
        }
        C::DrawIndexedPrimitives(at) => {
            let call = indexed
                .get(*at as usize)
                .ok_or_else(|| crate::error::internal("a logged indexed draw past its list"))?;
            encoder.draw_indexed_primitives(
                call.primitive_type,
                call.index_count,
                call.index_type,
                log.buffer(call.index_buffer),
                call.index_buffer_offset,
                call.instance_count,
                call.base_vertex,
                call.base_instance,
            )
        }
        C::DrawPrimitivesIndirect(primitive, buffer, offset) => {
            encoder.draw_primitives_indirect(*primitive, buffer, *offset)
        }
        C::DrawIndexedPrimitivesIndirect(call) => encoder.draw_indexed_primitives_indirect(
            call.primitive_type,
            call.index_type,
            &call.index_buffer,
            call.index_buffer_offset,
            &call.indirect_buffer,
            call.indirect_buffer_offset,
        ),
        C::DrawPatches(call) => encoder.draw_patches(
            call.control_points,
            call.patch_start,
            call.patch_count,
            call.patch_index_buffer.as_ref(),
            call.patch_index_buffer_offset,
            call.instance_count,
            call.base_instance,
        ),
        C::DrawPatchesIndirect(call) => encoder.draw_patches_indirect(
            call.control_points,
            call.patch_index_buffer.as_ref(),
            call.patch_index_buffer_offset,
            &call.indirect_buffer,
            call.indirect_buffer_offset,
        ),
        C::DrawIndexedPatches(call) => encoder.draw_indexed_patches(
            call.control_points,
            call.patch_start,
            call.patch_count,
            call.patch_index_buffer.as_ref(),
            call.patch_index_buffer_offset,
            &call.control_point_index_buffer,
            call.control_point_index_buffer_offset,
            call.instance_count,
            call.base_instance,
        ),
        C::DrawIndexedPatchesIndirect(call) => encoder.draw_indexed_patches_indirect(
            call.control_points,
            call.patch_index_buffer.as_ref(),
            call.patch_index_buffer_offset,
            &call.control_point_index_buffer,
            call.control_point_index_buffer_offset,
            &call.indirect_buffer,
            call.indirect_buffer_offset,
        ),
        C::VisibilityResultMode(mode, offset) => encoder.set_visibility_result_mode(*mode, *offset),
        C::UseResource(resource, usage, stages) => {
            encoder.use_resource(resource.as_ref(), *usage, *stages)
        }
        C::UseResources(resources, usage, stages) => {
            let resources: Vec<_> = resources.iter().map(Resource::as_ref).collect();
            encoder.use_resources(&resources, *usage, *stages)
        }
        C::UseHeap(heap, stages) => encoder.use_heap(heap, *stages),
        C::UseHeaps(heaps, stages) => {
            let heaps: Vec<&Heap> = heaps.iter().collect();
            encoder.use_heaps(&heaps, *stages)
        }
        C::Execute(icb, range) => encoder.execute_commands_in_buffer(icb, range.clone()),
        C::ExecuteIndirect(icb, buffer, offset) => {
            encoder.execute_commands_in_buffer_indirect(icb, buffer, *offset)
        }
        C::ColorResolveTarget(target, index) => {
            encoder.set_color_resolve_target(target.as_deref().cloned(), *index)
        }
        C::DepthResolveTarget(target) => {
            encoder.set_depth_resolve_target(target.as_deref().cloned())
        }
        C::StencilResolveTarget(target) => {
            encoder.set_stencil_resolve_target(target.as_deref().cloned())
        }
        C::MeshThreadgroups(call) => {
            let (groups, object, mesh) = **call;
            encoder.draw_mesh_threadgroups(groups, object, mesh)
        }
        C::MeshThreads(call) => {
            let (threads, object, mesh) = **call;
            encoder.draw_mesh_threads(threads, object, mesh)
        }
        C::MeshThreadgroupsIndirect(call) => {
            let (buffer, offset, object, mesh) = &**call;
            encoder.draw_mesh_threadgroups_indirect(buffer, *offset, *object, *mesh)
        }
        C::PrimitiveRestartEnabled(enabled, index) => {
            encoder.set_primitive_restart_enabled(*enabled, *index);
            Ok(())
        }
        C::ThreadsPerTile(threads) => encoder.dispatch_threads_per_tile(**threads),
        C::ThreadsPerTileInRegion(call) => {
            let (threads, region, index) = **call;
            encoder.dispatch_threads_per_tile_in_region(threads, region, index)
        }
        C::ThreadgroupMemoryLength(length, offset, index) => {
            encoder.set_threadgroup_memory_length(*length, *offset, *index)
        }
        C::ObjectThreadgroupMemoryLength(length, index) => {
            encoder.set_object_threadgroup_memory_length(*length, *index)
        }
        C::UpdateFence(fence, stages) => encoder.update_fence(fence, *stages),
        C::WaitForFence(fence, stages) => encoder.wait_for_fence(fence, *stages),
        C::BarrierWithScope(scope, after, before) => {
            encoder.memory_barrier_with_scope(*scope, *after, *before);
            Ok(())
        }
        C::BarrierWithResources(resources, after, before) => {
            let resources: Vec<_> = resources.iter().map(Resource::as_ref).collect();
            encoder.memory_barrier_with_resources(&resources, *after, *before)
        }
        C::AlphaTestReferenceValue(value) => encoder.set_alpha_test_reference_value(*value),
        C::LineWidth(width) => encoder.set_line_width(*width),
        C::PointSize(size) => encoder.set_point_size(*size),
        C::ProvokingVertexMode(mode) => encoder.set_provoking_vertex_mode(*mode),
        C::ViewportTransformEnabled(enabled) => encoder.set_viewport_transform_enabled(*enabled),
        C::VertexAmplificationMode(mode, value) => {
            encoder.set_vertex_amplification_mode(*mode, *value)
        }
        C::DepthCleared => encoder.set_depth_cleared(),
        C::StencilCleared => encoder.set_stencil_cleared(),
    }
}
