//! Encoders encoded again at a rebind: Metal resolves what a command buffer's commands name as it
//! first submits it, so a command buffer committed before a rebind of a buffer one of its commands
//! named, and first submitted after it, runs that command over the buffer's new pages (the
//! consumer's rebind probe on the M3 Max, table 2 cases 1, 2 and 5). The runtime records each
//! command into its `VkCommandBuffer` as it is encoded, with the `VkBuffer` the buffer lay in then;
//! so a compute, blit or render encoder logs its calls while the device has a buffer over host
//! pages, and the rebind encodes the encoders that named the buffer again from their logs before
//! the command buffer's first submission (`command_buffer/reencode.rs`).
//!
//! - **The log.** Each public call of a compute, blit or render encoder, with its arguments owned (a
//!   handle's clone, bytes copied into the log's own arena), in order, before the call runs:
//!   encoding it again re-runs each call, so the commands come out as the program encoded them,
//!   over where their buffers lie now. A call refused at its first encoding is refused again,
//!   which leaves the encoder as it was encoded first ([`Replay::run`] fails, and the encoder is
//!   not encoded again). The log starts at the encoder's open where the device has a buffer over
//!   host pages (`Rebinds::any`); else nothing is logged. It costs one push per
//!   call; a slice of buffers, offsets, textures or bytes goes into one of the log's arenas, so
//!   the common calls allocate nothing of their own.
//! - **What is kept.** The command buffer keeps an ended encoder's log only where a command of
//!   it named a buffer over host pages and the encoder can be encoded again: it left no segment
//!   for later (an indirect command buffer's execution, an indirect stage-in region), and fails
//!   nothing as it commits. An encoder of a command buffer that used command predication keeps
//!   none (a command encoded again after the commit would run outside its block): it fails
//!   `CommandBufferError::EncodedUseRebound` at a rebind as before, as does any other encoder
//!   (a resource state or acceleration structure encoder's), which logs nothing. (Changed:) a
//!   render encoder logs as these do, with its pass's descriptor (`render/replay.rs`).
//! - **Its memory.** No fixed cap (the coordinator's ruling of 2026-10-11): each entry holds its
//!   call's own arguments, so a log grows with its encoder's recording, as Metal's command buffer
//!   holds what the same calls encode. The logs go at the command buffer's first submission, after
//!   which no rebind can matter, without a lock: the scheduler drops them as it drops what a
//!   submission spent (`Submission::logs`).
//! - **Notices.** A call encoded again announces no kind its first encoding announced: each kind
//!   at most once (`Device::degrade`), as the program's call already met it.

use super::buffer::Buffer;
use super::command_buffer::CommandBuffer;
use super::encoder::Replayed;
use super::error::Error;
use super::fence::Fence;
use super::function_table::VisibleFunctionTable;
use super::heap::Heap;
use super::indirect_command_buffer::IndirectCommandBuffer;
use super::pipeline::ComputePipelineState;
use super::render::RenderLog;
use super::render_values::ClearColor;
use super::residency::Residency;
use super::resource::ResourceRef;
use super::sampler::SamplerState;
use super::texture::Texture;
use super::used_resources::EncoderKind;
use super::values::{Origin, Region, Size};
use super::{BlitCommandEncoder, ComputeCommandEncoder};
use crate::copy_plan::BlitOption;
use metal2vulkan::metal::{BarrierScope, DispatchType, PixelFormat, ResourceUsage};
use std::cell::Cell;
use std::ops::Range;

/// A run of elements in one of a log's arenas.
#[derive(Clone, Copy)]
pub(super) struct Span {
    start: usize,
    len: usize,
}

impl Span {
    fn range(self) -> Range<usize> {
        self.start..self.start + self.len
    }
}

/// A resource a call named, owned.
#[derive(Clone)]
pub(super) enum Resource {
    Buffer(Buffer),
    Texture(Texture),
    IndirectCommandBuffer(IndirectCommandBuffer),
}

impl Resource {
    fn of(resource: ResourceRef<'_>) -> Resource {
        match resource {
            ResourceRef::Buffer(buffer) => Resource::Buffer(buffer.clone()),
            ResourceRef::Texture(texture) => Resource::Texture(texture.clone()),
            ResourceRef::IndirectCommandBuffer(icb) => Resource::IndirectCommandBuffer(icb.clone()),
        }
    }

    pub(crate) fn as_ref(&self) -> ResourceRef<'_> {
        match self {
            Resource::Buffer(buffer) => ResourceRef::Buffer(buffer),
            Resource::Texture(texture) => ResourceRef::Texture(texture),
            Resource::IndirectCommandBuffer(icb) => ResourceRef::IndirectCommandBuffer(icb),
        }
    }

    /// `resources`, owned.
    pub(super) fn all(resources: &[ResourceRef<'_>]) -> Box<[Resource]> {
        resources
            .iter()
            .map(|&resource| Resource::of(resource))
            .collect()
    }

    /// One of `resource`, owned.
    pub(super) fn one(resource: ResourceRef<'_>) -> Resource {
        Resource::of(resource)
    }
}

/// An encoder's calls, in order, with the arenas their slices lie in.
pub(super) struct CallLog<C> {
    pub(crate) calls: Vec<C>,
    bytes: Vec<u8>,
    buffers: Vec<Option<Buffer>>,
    words: Vec<u64>,
    textures: Vec<Option<Texture>>,
    /// What it holds, counted as it grows: each call's size, and its arenas' and boxes'.
    size: usize,
    /// Calls of the encoder's own public calls run now, which the call that makes them logged
    /// (`BlitCommandEncoder::unlogged`).
    nested: usize,
}

impl<C> CallLog<C> {
    /// An empty log, which allocates nothing until its first call.
    pub(super) fn new() -> CallLog<C> {
        CallLog {
            calls: Vec::new(),
            bytes: Vec::new(),
            buffers: Vec::new(),
            words: Vec::new(),
            textures: Vec::new(),
            size: 0,
            nested: 0,
        }
    }

    /// Whether the encoder runs a public call of its own now, which logs nothing.
    pub(super) fn muted(&self) -> bool {
        self.nested > 0
    }

    /// Enter (`true`) or leave a public call of the encoder's own, which logs nothing.
    pub(super) fn mute(&mut self, enter: bool) {
        match enter {
            true => self.nested += 1,
            false => self.nested -= 1,
        }
    }

    /// Log `call`, which holds `extra` bytes of its own besides its size (a box's).
    pub(super) fn push(&mut self, call: C, extra: usize) {
        self.calls.push(call);
        self.size += std::mem::size_of::<C>() + extra;
    }

    /// The bytes it holds.
    pub(super) fn bytes(&self) -> usize {
        self.size
    }

    /// `bytes`, copied into the log's arena.
    pub(super) fn byte_span(&mut self, bytes: &[u8]) -> Span {
        self.size += bytes.len();
        span(&mut self.bytes, bytes.iter().copied())
    }

    /// `buffers`, held in the log's arena.
    pub(super) fn buffer_span(&mut self, buffers: &[Option<&Buffer>]) -> Span {
        self.size += std::mem::size_of_val(buffers);
        span(
            &mut self.buffers,
            buffers.iter().map(|buffer| buffer.cloned()),
        )
    }

    /// `words`, copied into the log's arena.
    pub(super) fn word_span(&mut self, words: &[u64]) -> Span {
        self.size += std::mem::size_of_val(words);
        span(&mut self.words, words.iter().copied())
    }

    /// `textures`, held in the log's arena.
    pub(super) fn texture_span(&mut self, textures: &[Option<&Texture>]) -> Span {
        self.size += std::mem::size_of_val(textures);
        span(
            &mut self.textures,
            textures.iter().map(|texture| texture.cloned()),
        )
    }

    pub(crate) fn bytes_of(&self, span: Span) -> &[u8] {
        &self.bytes[span.range()]
    }

    pub(crate) fn buffers_of(&self, span: Span) -> Vec<Option<&Buffer>> {
        self.buffers[span.range()]
            .iter()
            .map(Option::as_ref)
            .collect()
    }

    pub(crate) fn words_of(&self, span: Span) -> &[u64] {
        &self.words[span.range()]
    }

    pub(crate) fn textures_of(&self, span: Span) -> Vec<Option<&Texture>> {
        self.textures[span.range()]
            .iter()
            .map(Option::as_ref)
            .collect()
    }
}

/// Append `items` to `arena`: their span.
fn span<T>(arena: &mut Vec<T>, items: impl ExactSizeIterator<Item = T>) -> Span {
    let start = arena.len();
    let len = items.len();
    arena.extend(items);
    Span { start, len }
}

/// One public call of a compute encoder, its arguments owned (the method of each variant's name).
pub(super) enum ComputeCall {
    PipelineState(ComputePipelineState),
    Buffer(Option<Buffer>, u64, u32),
    Buffers(Span, Span, u32),
    BufferOffset(u64, u32),
    BufferWithStride(Option<Buffer>, u64, u64, u32),
    BuffersWithStrides(Span, Span, Span, u32),
    BufferOffsetWithStride(u64, u64, u32),
    Bytes(Span, u32),
    BytesWithStride(Span, u64, u32),
    FunctionTable(Option<VisibleFunctionTable>, u32),
    FunctionTables(Box<[Option<VisibleFunctionTable>]>, u32),
    Texture(Option<Texture>, u32),
    Textures(Span, u32),
    Sampler(Option<SamplerState>, u32),
    Samplers(Box<[Option<SamplerState>]>, u32),
    SamplerWithLodClamps(Option<SamplerState>, f32, f32, u32),
    SamplersWithLodClamps(Box<SamplersWithLodClamps>),
    ThreadgroupMemoryLength(u32, u32),
    ImageblockSize(u64, u64),
    StageInRegion(Region),
    StageInRegionIndirect(Buffer, u64),
    Threadgroups(Size, Size),
    Threads(Size, Size),
    ThreadgroupsIndirect(Buffer, u64, Size),
    ThreadsIndirect(Buffer, u64),
    BarrierWithScope(BarrierScope),
    BarrierWithResources(Box<[Resource]>),
    UseResource(Resource, ResourceUsage),
    UseResources(Box<[Resource]>, ResourceUsage),
    UseHeap(Heap),
    UseHeaps(Box<[Heap]>),
    Execute(IndirectCommandBuffer, Range<u64>),
    ExecuteIndirect(IndirectCommandBuffer, Buffer, u64),
    UpdateFence(Fence),
    WaitForFence(Fence),
}

impl ComputeCall {
    /// The bytes it holds in boxes of its own, besides its size.
    pub(super) fn extra(&self) -> usize {
        match self {
            ComputeCall::FunctionTables(tables, _) => std::mem::size_of_val(&**tables),
            ComputeCall::Samplers(samplers, _) => std::mem::size_of_val(&**samplers),
            ComputeCall::SamplersWithLodClamps(call) => {
                std::mem::size_of::<SamplersWithLodClamps>()
                    + std::mem::size_of_val(&*call.samplers)
                    + std::mem::size_of_val(&*call.mins)
                    + std::mem::size_of_val(&*call.maxes)
            }
            ComputeCall::BarrierWithResources(resources)
            | ComputeCall::UseResources(resources, _) => std::mem::size_of_val(&**resources),
            ComputeCall::UseHeaps(heaps) => std::mem::size_of_val(&**heaps),
            _ => 0,
        }
    }
}

/// `set_sampler_states_with_lod_clamps`'s arguments.
pub(super) struct SamplersWithLodClamps {
    pub(super) samplers: Box<[Option<SamplerState>]>,
    pub(super) mins: Box<[f32]>,
    pub(super) maxes: Box<[f32]>,
    pub(super) first: u32,
}

/// One public call of a blit encoder, its arguments owned (the method of each variant's name).
pub(super) enum BlitCall {
    CopyBufferToBuffer(Buffer, u64, Buffer, u64, u64),
    FillBuffer(Buffer, Range<u64>, u8),
    FillBufferPattern4(Buffer, Range<u64>, u32),
    SynchronizeResource(Resource),
    FillTexture(Box<(Texture, u64, u64, Region)>, Span),
    FillTextureColor(Box<(Texture, u64, u64, Region, ClearColor, Option<PixelFormat>)>),
    CopyBufferToTexture(Box<BufferToTexture>),
    CopyTextureToBuffer(Box<TextureToBuffer>),
    CopyTextureToTexture(Box<TextureToTexture>),
    CopyTextureSlices(Box<(Texture, u64, u64, Texture, u64, u64, u64, u64)>),
    CopyTexture(Texture, Texture),
    GenerateMipmaps(Texture),
    SynchronizeTexture(Texture, u64, u64),
    OptimizeForGpu(Texture),
    OptimizeForGpuSliceLevel(Texture, u64, u64),
    OptimizeForCpu(Texture),
    OptimizeForCpuSliceLevel(Texture, u64, u64),
    CopyIndirectCommandBuffer(
        Box<(
            IndirectCommandBuffer,
            Range<u64>,
            IndirectCommandBuffer,
            u64,
        )>,
    ),
    ResetCommandsInBuffer(IndirectCommandBuffer, Range<u64>),
    OptimizeIndirectCommandBuffer(IndirectCommandBuffer, Range<u64>),
    UpdateFence(Fence),
    WaitForFence(Fence),
    InvalidateCompressedTexture(Texture),
    InvalidateCompressedTextureSlice(Texture, u64, u64),
}

impl BlitCall {
    /// The bytes it holds in boxes of its own, besides its size.
    pub(super) fn extra(&self) -> usize {
        match self {
            BlitCall::FillTexture(call, _) => std::mem::size_of_val(&**call),
            BlitCall::FillTextureColor(call) => std::mem::size_of_val(&**call),
            BlitCall::CopyBufferToTexture(call) => std::mem::size_of_val(&**call),
            BlitCall::CopyTextureToBuffer(call) => std::mem::size_of_val(&**call),
            BlitCall::CopyTextureToTexture(call) => std::mem::size_of_val(&**call),
            BlitCall::CopyTextureSlices(call) => std::mem::size_of_val(&**call),
            BlitCall::CopyIndirectCommandBuffer(call) => std::mem::size_of_val(&**call),
            _ => 0,
        }
    }
}

/// `copy_from_buffer_to_texture`'s arguments.
pub(super) struct BufferToTexture {
    pub(super) source: Buffer,
    pub(super) source_offset: u64,
    pub(super) bytes_per_row: u64,
    pub(super) bytes_per_image: u64,
    pub(super) size: Size,
    pub(super) destination: Texture,
    pub(super) slice: u64,
    pub(super) level: u64,
    pub(super) origin: Origin,
    pub(super) options: BlitOption,
}

/// `copy_from_texture_to_buffer`'s arguments.
pub(super) struct TextureToBuffer {
    pub(super) source: Texture,
    pub(super) slice: u64,
    pub(super) level: u64,
    pub(super) origin: Origin,
    pub(super) size: Size,
    pub(super) destination: Buffer,
    pub(super) destination_offset: u64,
    pub(super) bytes_per_row: u64,
    pub(super) bytes_per_image: u64,
    pub(super) options: BlitOption,
}

/// `copy_from_texture_to_texture`'s arguments.
pub(super) struct TextureToTexture {
    pub(super) source: Texture,
    pub(super) source_slice: u64,
    pub(super) source_level: u64,
    pub(super) source_origin: Origin,
    pub(super) source_size: Size,
    pub(super) destination: Texture,
    pub(super) destination_slice: u64,
    pub(super) destination_level: u64,
    pub(super) destination_origin: Origin,
}

/// An ended encoder's log, which encodes it again (the module docs).
pub(super) enum Replay {
    Compute {
        dispatch_type: DispatchType,
        log: CallLog<ComputeCall>,
    },
    Blit(CallLog<BlitCall>),
    Render(Box<RenderLog>),
}

impl Replay {
    /// The bytes its log holds.
    pub(super) fn bytes(&self) -> usize {
        match self {
            Replay::Compute { log, .. } => log.bytes(),
            Replay::Blit(log) => log.bytes(),
            Replay::Render(log) => log.bytes(),
        }
    }

    /// The kind of its encoder.
    pub(super) fn kind(&self) -> EncoderKind {
        match self {
            Replay::Compute { .. } => EncoderKind::Compute,
            Replay::Blit(_) => EncoderKind::Blit,
            Replay::Render(_) => EncoderKind::Render,
        }
    }

    /// Encode its encoder again into `command_buffer`, committed and not yet first submitted,
    /// starting with `residency` as the encoder started (`CommandBuffer::inherited_residency`,
    /// of the encoders ended before it): what it recorded, and (a compute encoder's) what it
    /// made resident. Its calls announce no kind announced before (the module docs). Refused as
    /// a call is refused, or where it records otherwise than its first encoding did
    /// (`EncoderCore::finish_replay`).
    pub(super) fn run(
        &self,
        command_buffer: &CommandBuffer,
        residency: Residency,
    ) -> Result<(Replayed, Option<Residency>), Error> {
        replaying(|| match self {
            Replay::Compute { dispatch_type, log } => {
                let mut encoder =
                    ComputeCommandEncoder::replaying(command_buffer, *dispatch_type, residency)?;
                for call in &log.calls {
                    compute(&mut encoder, log, call)?;
                }
                encoder.finish_replay()
            }
            Replay::Blit(log) => {
                let mut encoder = BlitCommandEncoder::replaying(command_buffer)?;
                for call in &log.calls {
                    blit(&mut encoder, log, call)?;
                }
                Ok((encoder.finish_replay()?, None))
            }
            Replay::Render(log) => log.run(command_buffer, residency),
        })
    }
}

thread_local! {
    /// The encoders this thread is encoding again now ([`replaying`]).
    static REPLAYING: Cell<usize> = const { Cell::new(0) };
}

/// Run `f`, which encodes an encoder again, marked ([`is_replaying`]).
fn replaying<T>(f: impl FnOnce() -> T) -> T {
    /// Ends the mark as `f` returns or unwinds.
    struct Mark;
    impl Drop for Mark {
        fn drop(&mut self) {
            REPLAYING.with(|count| count.set(count.get() - 1));
        }
    }
    REPLAYING.with(|count| count.set(count.get() + 1));
    let _mark = Mark;
    f()
}

/// Whether this thread is encoding an encoder again now: its notices were announced as the
/// program encoded it (`Device::degrade`).
pub(crate) fn is_replaying() -> bool {
    REPLAYING.with(Cell::get) > 0
}

/// Run `call` again on `encoder`, its slices in `log`'s arenas.
fn compute(
    encoder: &mut ComputeCommandEncoder,
    log: &CallLog<ComputeCall>,
    call: &ComputeCall,
) -> Result<(), Error> {
    use ComputeCall as C;
    match call {
        C::PipelineState(state) => encoder.set_compute_pipeline_state(state),
        C::Buffer(buffer, offset, index) => encoder.set_buffer(buffer.as_ref(), *offset, *index),
        C::Buffers(buffers, offsets, first) => {
            let buffers = log.buffers_of(*buffers);
            encoder.set_buffers(&buffers, log.words_of(*offsets), *first)
        }
        C::BufferOffset(offset, index) => encoder.set_buffer_offset(*offset, *index),
        C::BufferWithStride(buffer, offset, stride, index) => {
            encoder.set_buffer_with_stride(buffer.as_ref(), *offset, *stride, *index)
        }
        C::BuffersWithStrides(buffers, offsets, strides, first) => {
            let buffers = log.buffers_of(*buffers);
            let (offsets, strides) = (log.words_of(*offsets), log.words_of(*strides));
            encoder.set_buffers_with_strides(&buffers, offsets, strides, *first)
        }
        C::BufferOffsetWithStride(offset, stride, index) => {
            encoder.set_buffer_offset_with_stride(*offset, *stride, *index)
        }
        C::Bytes(bytes, index) => encoder.set_bytes(log.bytes_of(*bytes), *index),
        C::BytesWithStride(bytes, stride, index) => {
            encoder.set_bytes_with_stride(log.bytes_of(*bytes), *stride, *index)
        }
        C::FunctionTable(table, index) => {
            encoder.set_visible_function_table(table.as_ref(), *index)
        }
        C::FunctionTables(tables, first) => {
            let tables: Vec<_> = tables.iter().map(Option::as_ref).collect();
            encoder.set_visible_function_tables(&tables, *first)
        }
        C::Texture(texture, index) => encoder.set_texture(texture.as_ref(), *index),
        C::Textures(textures, first) => encoder.set_textures(&log.textures_of(*textures), *first),
        C::Sampler(sampler, index) => encoder.set_sampler_state(sampler.as_ref(), *index),
        C::Samplers(samplers, first) => {
            let samplers: Vec<_> = samplers.iter().map(Option::as_ref).collect();
            encoder.set_sampler_states(&samplers, *first)
        }
        C::SamplerWithLodClamps(sampler, min, max, index) => {
            encoder.set_sampler_state_with_lod_clamps(sampler.as_ref(), *min, *max, *index)
        }
        C::SamplersWithLodClamps(call) => {
            let samplers: Vec<_> = call.samplers.iter().map(Option::as_ref).collect();
            encoder.set_sampler_states_with_lod_clamps(
                &samplers,
                &call.mins,
                &call.maxes,
                call.first,
            )
        }
        C::ThreadgroupMemoryLength(length, index) => {
            encoder.set_threadgroup_memory_length(*length, *index)
        }
        C::ImageblockSize(width, height) => encoder.set_imageblock_size(*width, *height),
        C::StageInRegion(region) => encoder.set_stage_in_region(*region),
        C::StageInRegionIndirect(buffer, offset) => {
            encoder.set_stage_in_region_indirect(buffer, *offset)
        }
        C::Threadgroups(groups, threads) => encoder.dispatch_threadgroups(*groups, *threads),
        C::Threads(threads, per_group) => encoder.dispatch_threads(*threads, *per_group),
        C::ThreadgroupsIndirect(buffer, offset, threads) => {
            encoder.dispatch_threadgroups_indirect(buffer, *offset, *threads)
        }
        C::ThreadsIndirect(buffer, offset) => encoder.dispatch_threads_indirect(buffer, *offset),
        C::BarrierWithScope(scope) => {
            encoder.memory_barrier_with_scope(*scope);
            Ok(())
        }
        C::BarrierWithResources(resources) => {
            let resources: Vec<_> = resources.iter().map(Resource::as_ref).collect();
            encoder.memory_barrier_with_resources(&resources)
        }
        C::UseResource(resource, usage) => encoder.use_resource(resource.as_ref(), *usage),
        C::UseResources(resources, usage) => {
            let resources: Vec<_> = resources.iter().map(Resource::as_ref).collect();
            encoder.use_resources(&resources, *usage)
        }
        C::UseHeap(heap) => encoder.use_heap(heap),
        C::UseHeaps(heaps) => {
            let heaps: Vec<&Heap> = heaps.iter().collect();
            encoder.use_heaps(&heaps)
        }
        C::Execute(icb, range) => encoder.execute_commands_in_buffer(icb, range.clone()),
        C::ExecuteIndirect(icb, buffer, offset) => {
            encoder.execute_commands_in_buffer_indirect(icb, buffer, *offset)
        }
        C::UpdateFence(fence) => encoder.update_fence(fence),
        C::WaitForFence(fence) => encoder.wait_for_fence(fence),
    }
}

/// Run `call` again on `encoder`, its slices in `log`'s arenas.
fn blit(
    encoder: &mut BlitCommandEncoder,
    log: &CallLog<BlitCall>,
    call: &BlitCall,
) -> Result<(), Error> {
    use BlitCall as B;
    match call {
        B::CopyBufferToBuffer(source, source_offset, destination, destination_offset, size) => {
            encoder.copy_from_buffer_to_buffer(
                source,
                *source_offset,
                destination,
                *destination_offset,
                *size,
            )
        }
        B::FillBuffer(buffer, range, value) => encoder.fill_buffer(buffer, range.clone(), *value),
        B::FillBufferPattern4(buffer, range, pattern) => {
            encoder.fill_buffer_pattern4(buffer, range.clone(), *pattern)
        }
        B::SynchronizeResource(resource) => encoder.synchronize_resource(resource.as_ref()),
        B::FillTexture(call, pattern) => {
            let (texture, level, slice, region) = &**call;
            encoder.fill_texture(texture, *level, *slice, *region, log.bytes_of(*pattern))
        }
        B::FillTextureColor(call) => {
            let (texture, level, slice, region, color, format) = &**call;
            encoder.fill_texture_color(texture, *level, *slice, *region, *color, *format)
        }
        B::CopyBufferToTexture(call) => encoder.copy_from_buffer_to_texture(
            &call.source,
            call.source_offset,
            call.bytes_per_row,
            call.bytes_per_image,
            call.size,
            &call.destination,
            call.slice,
            call.level,
            call.origin,
            call.options,
        ),
        B::CopyTextureToBuffer(call) => encoder.copy_from_texture_to_buffer(
            &call.source,
            call.slice,
            call.level,
            call.origin,
            call.size,
            &call.destination,
            call.destination_offset,
            call.bytes_per_row,
            call.bytes_per_image,
            call.options,
        ),
        B::CopyTextureToTexture(call) => encoder.copy_from_texture_to_texture(
            &call.source,
            call.source_slice,
            call.source_level,
            call.source_origin,
            call.source_size,
            &call.destination,
            call.destination_slice,
            call.destination_level,
            call.destination_origin,
        ),
        B::CopyTextureSlices(call) => {
            let (source, source_slice, source_level, destination, slice, level, slices, levels) =
                &**call;
            encoder.copy_texture_slices(
                source,
                *source_slice,
                *source_level,
                destination,
                *slice,
                *level,
                *slices,
                *levels,
            )
        }
        B::CopyTexture(source, destination) => encoder.copy_texture(source, destination),
        B::GenerateMipmaps(texture) => encoder.generate_mipmaps(texture),
        B::SynchronizeTexture(texture, slice, level) => {
            encoder.synchronize_texture(texture, *slice, *level)
        }
        B::OptimizeForGpu(texture) => encoder.optimize_contents_for_gpu_access(texture),
        B::OptimizeForGpuSliceLevel(texture, slice, level) => {
            encoder.optimize_contents_for_gpu_access_slice_level(texture, *slice, *level)
        }
        B::OptimizeForCpu(texture) => encoder.optimize_contents_for_cpu_access(texture),
        B::OptimizeForCpuSliceLevel(texture, slice, level) => {
            encoder.optimize_contents_for_cpu_access_slice_level(texture, *slice, *level)
        }
        B::CopyIndirectCommandBuffer(call) => {
            let (source, range, destination, index) = &**call;
            encoder.copy_indirect_command_buffer(source, range.clone(), destination, *index)
        }
        B::ResetCommandsInBuffer(icb, range) => {
            encoder.reset_commands_in_buffer(icb, range.clone())
        }
        B::OptimizeIndirectCommandBuffer(icb, range) => {
            encoder.optimize_indirect_command_buffer(icb, range.clone())
        }
        B::UpdateFence(fence) => encoder.update_fence(fence),
        B::WaitForFence(fence) => encoder.wait_for_fence(fence),
        B::InvalidateCompressedTexture(texture) => encoder.invalidate_compressed_texture(texture),
        B::InvalidateCompressedTextureSlice(texture, slice, level) => {
            encoder.invalidate_compressed_texture_slice(texture, *slice, *level)
        }
    }
}
