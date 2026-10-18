//! A deferred command buffer's log: each command its recording made, with its arguments copied at
//! the call, and the replay of them into a `VkCommandBuffer` (`recorder`).
//!
//! - **Copied at the call.** Every argument a command reads through a pointer (regions, barriers,
//!   viewports, clear values, push constants, an update's bytes, a pushed set's descriptors, a
//!   begin's inheritance) is copied into one of the log's arenas as the command is called, as a
//!   driver reads them at the call: the program's and the runtime's memory may change at once.
//!   Handles are copied as they are; every object a command names stays live until its command
//!   buffer completes (the runtime's contract for a recording), so past the replay.
//! - **Extension structures.** A structure's chain holds only what the runtime chains there (a
//!   depth bias's representation); a chain of anything else is a defect of the runtime's, which
//!   ends the process at the call: the command would otherwise be replayed without it.
//! - **Its memory** is kept for the next log ([`Log::clear`]): the arenas keep their room. What a
//!   replay builds of them to call with (a pushed set's writes, an execution's secondaries) is
//!   built in its thread's room ([`SCRATCH`]), so a replay allocates nothing once that holds
//!   enough.

use super::handles;
use super::replays::{real_of, Replays};
use super::thunks::{real, Command};
use crate::exit_guard::{self, Call};
use crate::generated_commands::GeneratedCommandsInfo;
use ash::vk::{self, Handle};
use std::cell::RefCell;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// What a replay builds of its log's arenas to call with.
struct Scratch {
    /// A pushed set's writes, pointing into the log's arenas while their call lasts.
    writes: Vec<vk::WriteDescriptorSet<'static>>,
    /// An execution's secondaries, replayed.
    secondaries: Vec<vk::CommandBuffer>,
}

thread_local! {
    /// The room replays on this thread build in ([`Scratch`]), kept for the next.
    static SCRATCH: RefCell<Scratch> = const {
        RefCell::new(Scratch {
            writes: Vec::new(),
            secondaries: Vec::new(),
        })
    };
}

/// A run of elements in one of a log's arenas.
#[derive(Clone, Copy, Default)]
struct Span {
    start: u32,
    len: u32,
}

/// An arena of `T`s that crosses threads: its elements' pointers (an `ash` structure's `p_next`)
/// are null, checked as each goes in.
struct Arena<T>(Vec<T>);

// SAFETY: every element's pointers are null (the type's docs): the elements are plain values.
unsafe impl<T> Send for Arena<T> {}
// SAFETY: as above.
unsafe impl<T> Sync for Arena<T> {}

impl<T: Copy> Arena<T> {
    const fn new() -> Arena<T> {
        Arena(Vec::new())
    }

    /// `count` elements from `from`, copied in.
    ///
    /// # Safety
    ///
    /// `from` points at `count` readable elements, or `count` is 0.
    unsafe fn push(&mut self, from: *const T, count: u32) -> Span {
        let start = self.0.len() as u32;
        if count > 0 && !from.is_null() {
            // SAFETY: the caller's.
            self.0
                .extend_from_slice(unsafe { std::slice::from_raw_parts(from, count as usize) });
        }
        Span {
            start,
            len: self.0.len() as u32 - start,
        }
    }

    fn one(&mut self, value: T) -> Span {
        let start = self.0.len() as u32;
        self.0.push(value);
        Span { start, len: 1 }
    }

    fn get(&self, span: Span) -> &[T] {
        &self.0[span.start as usize..(span.start + span.len) as usize]
    }

    fn get_ptr(&self, span: Span) -> *const T {
        match span.len {
            0 => std::ptr::null(),
            _ => self.get(span).as_ptr(),
        }
    }
}

/// End the process for a structure chain the log does not copy (the module docs).
#[cold]
fn unchained(what: &str) -> ! {
    panic!("a deferred command's {what} chains a structure the recorder does not copy");
}

/// A pushed set's write, its descriptors in the log's arenas.
#[derive(Clone, Copy)]
struct Write {
    binding: u32,
    element: u32,
    count: u32,
    kind: vk::DescriptorType,
    images: Span,
    buffers: Span,
    views: Span,
}

/// A secondary command buffer's inheritance, copied.
#[derive(Clone, Copy)]
struct Inheritance {
    render_pass: vk::RenderPass,
    subpass: u32,
    framebuffer: vk::Framebuffer,
    occlusion: vk::Bool32,
    query_flags: vk::QueryControlFlags,
    statistics: vk::QueryPipelineStatisticFlags,
}

/// One command, its arrays in the log's arenas.
#[derive(Clone, Copy)]
enum Logged {
    BindPipeline(vk::PipelineBindPoint, vk::Pipeline),
    SetViewport(u32, Span),
    SetScissor(u32, Span),
    SetDepthBias(f32, f32, f32),
    SetBlendConstants([f32; 4]),
    SetStencilCompareMask(vk::StencilFaceFlags, u32),
    SetStencilWriteMask(vk::StencilFaceFlags, u32),
    SetStencilReference(vk::StencilFaceFlags, u32),
    BindDescriptorSets {
        point: vk::PipelineBindPoint,
        layout: vk::PipelineLayout,
        first: u32,
        sets: Span,
        offsets: Span,
    },
    BindIndexBuffer(vk::Buffer, vk::DeviceSize, vk::IndexType),
    BindVertexBuffers {
        first: u32,
        buffers: Span,
        offsets: Span,
    },
    Draw(u32, u32, u32, u32),
    DrawIndexed(u32, u32, u32, i32, u32),
    DrawIndirect(vk::Buffer, vk::DeviceSize, u32, u32),
    DrawIndexedIndirect(vk::Buffer, vk::DeviceSize, u32, u32),
    Dispatch(u32, u32, u32),
    DispatchIndirect(vk::Buffer, vk::DeviceSize),
    CopyBuffer(vk::Buffer, vk::Buffer, Span),
    CopyImage {
        src: vk::Image,
        src_layout: vk::ImageLayout,
        dst: vk::Image,
        dst_layout: vk::ImageLayout,
        regions: Span,
    },
    BlitImage {
        src: vk::Image,
        src_layout: vk::ImageLayout,
        dst: vk::Image,
        dst_layout: vk::ImageLayout,
        regions: Span,
        filter: vk::Filter,
    },
    CopyBufferToImage {
        src: vk::Buffer,
        dst: vk::Image,
        layout: vk::ImageLayout,
        regions: Span,
    },
    CopyImageToBuffer {
        src: vk::Image,
        layout: vk::ImageLayout,
        dst: vk::Buffer,
        regions: Span,
    },
    UpdateBuffer(vk::Buffer, vk::DeviceSize, Span),
    FillBuffer(vk::Buffer, vk::DeviceSize, vk::DeviceSize, u32),
    ClearColorImage(vk::Image, vk::ImageLayout, vk::ClearColorValue, Span),
    ClearDepthStencilImage(vk::Image, vk::ImageLayout, vk::ClearDepthStencilValue, Span),
    ClearAttachments(Span, Span),
    PipelineBarrier {
        src: vk::PipelineStageFlags,
        dst: vk::PipelineStageFlags,
        flags: vk::DependencyFlags,
        memory: Span,
        buffers: Span,
        images: Span,
    },
    BeginQuery(vk::QueryPool, u32, vk::QueryControlFlags),
    EndQuery(vk::QueryPool, u32),
    ResetQueryPool(vk::QueryPool, u32, u32),
    WriteTimestamp(vk::PipelineStageFlags, vk::QueryPool, u32),
    CopyQueryPoolResults {
        pool: vk::QueryPool,
        first: u32,
        count: u32,
        dst: vk::Buffer,
        offset: vk::DeviceSize,
        stride: vk::DeviceSize,
        flags: vk::QueryResultFlags,
    },
    PushConstants {
        layout: vk::PipelineLayout,
        stages: vk::ShaderStageFlags,
        offset: u32,
        bytes: Span,
    },
    BeginRenderPass {
        render_pass: vk::RenderPass,
        framebuffer: vk::Framebuffer,
        area: vk::Rect2D,
        clears: Span,
        contents: vk::SubpassContents,
    },
    EndRenderPass,
    ExecuteCommands(Span),
    SetCullMode(vk::CullModeFlags),
    SetFrontFace(vk::FrontFace),
    SetPrimitiveTopology(vk::PrimitiveTopology),
    BindVertexBuffers2 {
        first: u32,
        buffers: Span,
        offsets: Span,
        sizes: Option<Span>,
        strides: Option<Span>,
    },
    SetDepthTestEnable(vk::Bool32),
    SetDepthWriteEnable(vk::Bool32),
    SetDepthCompareOp(vk::CompareOp),
    SetStencilTestEnable(vk::Bool32),
    SetStencilOp(
        vk::StencilFaceFlags,
        vk::StencilOp,
        vk::StencilOp,
        vk::StencilOp,
        vk::CompareOp,
    ),
    SetPrimitiveRestartEnable(vk::Bool32),
    PushDescriptorSet {
        point: vk::PipelineBindPoint,
        layout: vk::PipelineLayout,
        set: u32,
        writes: Span,
    },
    SetColorWriteMask(u32, Span),
    SetDepthBias2 {
        factors: [f32; 3],
        representation: Option<(vk::DepthBiasRepresentationEXT, vk::Bool32)>,
    },
    BindIndexBuffer2(vk::Buffer, vk::DeviceSize, vk::DeviceSize, vk::IndexType),
    DrawMeshTasks(u32, u32, u32),
    DrawMeshTasksIndirect(vk::Buffer, vk::DeviceSize, u32, u32),
    /// Preprocessed, and the info's place in its arena.
    ExecuteGeneratedCommands(vk::Bool32, Span),
    /// The info's place, and the state command buffer where it is another than this one.
    PreprocessGeneratedCommands(Span, Option<vk::CommandBuffer>),
}

/// Where a log stands.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Stage {
    /// Made, not begun.
    Initial,
    Recording,
    Ended,
}

/// One deferred command buffer's log (the module docs).
pub(super) struct Log {
    /// Its handle's entry, and its command buffer's replays, while its handle is live.
    index: u32,
    replays: Option<Arc<Replays>>,
    /// The command buffer its replay recorded, once replayed (0 before).
    real: AtomicU64,
    pub(super) stage: Stage,
    pub(super) level: vk::CommandBufferLevel,
    /// The begin's usage flags and a secondary's inheritance.
    flags: vk::CommandBufferUsageFlags,
    inheritance: Option<Inheritance>,
    ops: Vec<Logged>,
    bytes: Arena<u8>,
    words: Arena<u64>,
    u32s: Arena<u32>,
    buffers: Arena<vk::Buffer>,
    sets: Arena<vk::DescriptorSet>,
    viewports: Arena<vk::Viewport>,
    rects: Arena<vk::Rect2D>,
    buffer_copies: Arena<vk::BufferCopy>,
    image_copies: Arena<vk::ImageCopy>,
    blits: Arena<vk::ImageBlit>,
    buffer_images: Arena<vk::BufferImageCopy>,
    ranges: Arena<vk::ImageSubresourceRange>,
    clear_attachments: Arena<vk::ClearAttachment>,
    clear_rects: Arena<vk::ClearRect>,
    clear_values: Arena<vk::ClearValue>,
    memory_barriers: Arena<vk::MemoryBarrier<'static>>,
    buffer_barriers: Arena<vk::BufferMemoryBarrier<'static>>,
    image_barriers: Arena<vk::ImageMemoryBarrier<'static>>,
    command_buffers: Arena<vk::CommandBuffer>,
    writes: Arena<Write>,
    image_infos: Arena<vk::DescriptorImageInfo>,
    buffer_infos: Arena<vk::DescriptorBufferInfo>,
    views: Arena<vk::BufferView>,
    masks: Arena<vk::ColorComponentFlags>,
    generated: Arena<GeneratedCommandsInfo>,
}

impl Log {
    /// An empty log of `level`.
    pub(super) fn new(level: vk::CommandBufferLevel) -> Log {
        Log {
            index: 0,
            replays: None,
            real: AtomicU64::new(0),
            stage: Stage::Initial,
            level,
            flags: vk::CommandBufferUsageFlags::empty(),
            inheritance: None,
            ops: Vec::new(),
            bytes: Arena::new(),
            words: Arena::new(),
            u32s: Arena::new(),
            buffers: Arena::new(),
            sets: Arena::new(),
            viewports: Arena::new(),
            rects: Arena::new(),
            buffer_copies: Arena::new(),
            image_copies: Arena::new(),
            blits: Arena::new(),
            buffer_images: Arena::new(),
            ranges: Arena::new(),
            clear_attachments: Arena::new(),
            clear_rects: Arena::new(),
            clear_values: Arena::new(),
            memory_barriers: Arena::new(),
            buffer_barriers: Arena::new(),
            image_barriers: Arena::new(),
            command_buffers: Arena::new(),
            writes: Arena::new(),
            image_infos: Arena::new(),
            buffer_infos: Arena::new(),
            views: Arena::new(),
            masks: Arena::new(),
            generated: Arena::new(),
        }
    }

    /// Empty it for a new command buffer of `level`, keeping its arenas' room.
    pub(super) fn clear(&mut self, level: vk::CommandBufferLevel) {
        self.stage = Stage::Initial;
        self.level = level;
        self.flags = vk::CommandBufferUsageFlags::empty();
        self.inheritance = None;
        self.ops.clear();
        self.bytes.0.clear();
        self.words.0.clear();
        self.u32s.0.clear();
        self.buffers.0.clear();
        self.sets.0.clear();
        self.viewports.0.clear();
        self.rects.0.clear();
        self.buffer_copies.0.clear();
        self.image_copies.0.clear();
        self.blits.0.clear();
        self.buffer_images.0.clear();
        self.ranges.0.clear();
        self.clear_attachments.0.clear();
        self.clear_rects.0.clear();
        self.clear_values.0.clear();
        self.memory_barriers.0.clear();
        self.buffer_barriers.0.clear();
        self.image_barriers.0.clear();
        self.command_buffers.0.clear();
        self.writes.0.clear();
        self.image_infos.0.clear();
        self.buffer_infos.0.clear();
        self.views.0.clear();
        self.masks.0.clear();
        self.generated.0.clear();
    }

    /// It is handle `index`'s, of a command buffer's `replays`.
    pub(super) fn own(&mut self, index: u32, replays: Arc<Replays>) {
        self.index = index;
        self.replays = Some(replays);
        self.real.store(0, Ordering::Relaxed);
    }

    /// Its handle was let go of.
    pub(super) fn disown(&mut self) {
        self.replays = None;
    }

    /// Its handle's entry.
    pub(super) fn index(&self) -> u32 {
        self.index
    }

    /// Whether its replay may record into a primary begun ahead (`spares.rs`): a primary begun as
    /// that one is, `ONE_TIME_SUBMIT` alone.
    pub(super) fn takes_begun(&self) -> bool {
        self.level == vk::CommandBufferLevel::PRIMARY
            && self.flags == vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT
            && self.inheritance.is_none()
    }

    /// Its command buffer's replays, while its handle is live.
    pub(super) fn replays(&self) -> Option<&Arc<Replays>> {
        self.replays.as_ref()
    }

    /// The command buffer its replay recorded, once replayed.
    pub(super) fn real(&self) -> Option<vk::CommandBuffer> {
        real_of(&self.real)
    }

    /// Its replay recorded `real`, ended.
    pub(super) fn replayed(&self, real: vk::CommandBuffer) {
        self.real.store(real.as_raw(), Ordering::Release);
    }

    /// Room for the end a commit may fold into this log, its command buffer's tail
    /// ([`reserve_end`](super::reserve_end)): two commands and one memory barrier.
    pub(super) fn reserve_end(&mut self) {
        self.ops.reserve(2);
        self.memory_barriers.0.reserve(1);
    }

    fn op(&mut self, op: Logged) {
        debug_assert_eq!(
            self.stage,
            Stage::Recording,
            "a command outside a recording"
        );
        self.ops.push(op);
    }
}

/// The logging of each routed command: the same parameters as its entry point's after the
/// command buffer (`thunks.rs`), each copying what it reads through a pointer.
///
/// # Safety (each)
///
/// The command's valid usage: its pointers point at as many elements as its counts say.
impl Log {
    pub(super) unsafe fn begin_command_buffer(
        &mut self,
        info: *const vk::CommandBufferBeginInfo<'_>,
    ) -> vk::Result {
        // SAFETY: the caller's.
        let info = unsafe { &*info };
        if !info.p_next.is_null() {
            unchained("begin");
        }
        let level = self.level;
        self.clear(level);
        self.flags = info.flags;
        if level == vk::CommandBufferLevel::SECONDARY && !info.p_inheritance_info.is_null() {
            // SAFETY: the caller's.
            let inherited = unsafe { &*info.p_inheritance_info };
            if !inherited.p_next.is_null() {
                unchained("inheritance");
            }
            self.inheritance = Some(Inheritance {
                render_pass: inherited.render_pass,
                subpass: inherited.subpass,
                framebuffer: inherited.framebuffer,
                occlusion: inherited.occlusion_query_enable,
                query_flags: inherited.query_flags,
                statistics: inherited.pipeline_statistics,
            });
        }
        self.stage = Stage::Recording;
        vk::Result::SUCCESS
    }

    /// Its end: its replays take it once the call's borrow of it ended (`handles::record`).
    pub(super) unsafe fn end_command_buffer(&mut self) -> vk::Result {
        debug_assert_eq!(self.stage, Stage::Recording, "an end outside a recording");
        self.stage = Stage::Ended;
        vk::Result::SUCCESS
    }

    pub(super) unsafe fn cmd_bind_pipeline(
        &mut self,
        point: vk::PipelineBindPoint,
        pipeline: vk::Pipeline,
    ) {
        self.op(Logged::BindPipeline(point, pipeline));
    }

    pub(super) unsafe fn cmd_set_viewport(
        &mut self,
        first: u32,
        count: u32,
        viewports: *const vk::Viewport,
    ) {
        // SAFETY: the caller's.
        let span = unsafe { self.viewports.push(viewports, count) };
        self.op(Logged::SetViewport(first, span));
    }

    pub(super) unsafe fn cmd_set_scissor(
        &mut self,
        first: u32,
        count: u32,
        rects: *const vk::Rect2D,
    ) {
        // SAFETY: the caller's.
        let span = unsafe { self.rects.push(rects, count) };
        self.op(Logged::SetScissor(first, span));
    }

    pub(super) unsafe fn cmd_set_depth_bias(&mut self, constant: f32, clamp: f32, slope: f32) {
        self.op(Logged::SetDepthBias(constant, clamp, slope));
    }

    pub(super) unsafe fn cmd_set_blend_constants(&mut self, constants: *const [f32; 4]) {
        // SAFETY: the caller's.
        self.op(Logged::SetBlendConstants(unsafe { *constants }));
    }

    pub(super) unsafe fn cmd_set_stencil_compare_mask(
        &mut self,
        faces: vk::StencilFaceFlags,
        mask: u32,
    ) {
        self.op(Logged::SetStencilCompareMask(faces, mask));
    }

    pub(super) unsafe fn cmd_set_stencil_write_mask(
        &mut self,
        faces: vk::StencilFaceFlags,
        mask: u32,
    ) {
        self.op(Logged::SetStencilWriteMask(faces, mask));
    }

    pub(super) unsafe fn cmd_set_stencil_reference(
        &mut self,
        faces: vk::StencilFaceFlags,
        reference: u32,
    ) {
        self.op(Logged::SetStencilReference(faces, reference));
    }

    pub(super) unsafe fn cmd_bind_descriptor_sets(
        &mut self,
        point: vk::PipelineBindPoint,
        layout: vk::PipelineLayout,
        first: u32,
        count: u32,
        sets: *const vk::DescriptorSet,
        offset_count: u32,
        offsets: *const u32,
    ) {
        // SAFETY: the caller's.
        let (sets, offsets) = unsafe {
            (
                self.sets.push(sets, count),
                self.u32s.push(offsets, offset_count),
            )
        };
        self.op(Logged::BindDescriptorSets {
            point,
            layout,
            first,
            sets,
            offsets,
        });
    }

    pub(super) unsafe fn cmd_bind_index_buffer(
        &mut self,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        index_type: vk::IndexType,
    ) {
        self.op(Logged::BindIndexBuffer(buffer, offset, index_type));
    }

    pub(super) unsafe fn cmd_bind_vertex_buffers(
        &mut self,
        first: u32,
        count: u32,
        buffers: *const vk::Buffer,
        offsets: *const vk::DeviceSize,
    ) {
        // SAFETY: the caller's.
        let (buffers, offsets) = unsafe {
            (
                self.buffers.push(buffers, count),
                self.words.push(offsets, count),
            )
        };
        self.op(Logged::BindVertexBuffers {
            first,
            buffers,
            offsets,
        });
    }

    pub(super) unsafe fn cmd_draw(&mut self, vertices: u32, instances: u32, first: u32, base: u32) {
        self.op(Logged::Draw(vertices, instances, first, base));
    }

    pub(super) unsafe fn cmd_draw_indexed(
        &mut self,
        indices: u32,
        instances: u32,
        first: u32,
        offset: i32,
        base: u32,
    ) {
        self.op(Logged::DrawIndexed(indices, instances, first, offset, base));
    }

    pub(super) unsafe fn cmd_draw_indirect(
        &mut self,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        count: u32,
        stride: u32,
    ) {
        self.op(Logged::DrawIndirect(buffer, offset, count, stride));
    }

    pub(super) unsafe fn cmd_draw_indexed_indirect(
        &mut self,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        count: u32,
        stride: u32,
    ) {
        self.op(Logged::DrawIndexedIndirect(buffer, offset, count, stride));
    }

    pub(super) unsafe fn cmd_dispatch(&mut self, x: u32, y: u32, z: u32) {
        self.op(Logged::Dispatch(x, y, z));
    }

    pub(super) unsafe fn cmd_dispatch_indirect(
        &mut self,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
    ) {
        self.op(Logged::DispatchIndirect(buffer, offset));
    }

    pub(super) unsafe fn cmd_copy_buffer(
        &mut self,
        src: vk::Buffer,
        dst: vk::Buffer,
        count: u32,
        regions: *const vk::BufferCopy,
    ) {
        // SAFETY: the caller's.
        let regions = unsafe { self.buffer_copies.push(regions, count) };
        self.op(Logged::CopyBuffer(src, dst, regions));
    }

    pub(super) unsafe fn cmd_copy_image(
        &mut self,
        src: vk::Image,
        src_layout: vk::ImageLayout,
        dst: vk::Image,
        dst_layout: vk::ImageLayout,
        count: u32,
        regions: *const vk::ImageCopy,
    ) {
        // SAFETY: the caller's.
        let regions = unsafe { self.image_copies.push(regions, count) };
        self.op(Logged::CopyImage {
            src,
            src_layout,
            dst,
            dst_layout,
            regions,
        });
    }

    pub(super) unsafe fn cmd_blit_image(
        &mut self,
        src: vk::Image,
        src_layout: vk::ImageLayout,
        dst: vk::Image,
        dst_layout: vk::ImageLayout,
        count: u32,
        regions: *const vk::ImageBlit,
        filter: vk::Filter,
    ) {
        // SAFETY: the caller's.
        let regions = unsafe { self.blits.push(regions, count) };
        self.op(Logged::BlitImage {
            src,
            src_layout,
            dst,
            dst_layout,
            regions,
            filter,
        });
    }

    pub(super) unsafe fn cmd_copy_buffer_to_image(
        &mut self,
        src: vk::Buffer,
        dst: vk::Image,
        layout: vk::ImageLayout,
        count: u32,
        regions: *const vk::BufferImageCopy,
    ) {
        // SAFETY: the caller's.
        let regions = unsafe { self.buffer_images.push(regions, count) };
        self.op(Logged::CopyBufferToImage {
            src,
            dst,
            layout,
            regions,
        });
    }

    pub(super) unsafe fn cmd_copy_image_to_buffer(
        &mut self,
        src: vk::Image,
        layout: vk::ImageLayout,
        dst: vk::Buffer,
        count: u32,
        regions: *const vk::BufferImageCopy,
    ) {
        // SAFETY: the caller's.
        let regions = unsafe { self.buffer_images.push(regions, count) };
        self.op(Logged::CopyImageToBuffer {
            src,
            layout,
            dst,
            regions,
        });
    }

    pub(super) unsafe fn cmd_update_buffer(
        &mut self,
        dst: vk::Buffer,
        offset: vk::DeviceSize,
        size: vk::DeviceSize,
        data: *const c_void,
    ) {
        // SAFETY: the caller's: `size` bytes at `data` (at most 65536, Vulkan's bound).
        let bytes = unsafe { self.bytes.push(data.cast(), size as u32) };
        self.op(Logged::UpdateBuffer(dst, offset, bytes));
    }

    pub(super) unsafe fn cmd_fill_buffer(
        &mut self,
        dst: vk::Buffer,
        offset: vk::DeviceSize,
        size: vk::DeviceSize,
        data: u32,
    ) {
        self.op(Logged::FillBuffer(dst, offset, size, data));
    }

    pub(super) unsafe fn cmd_clear_color_image(
        &mut self,
        image: vk::Image,
        layout: vk::ImageLayout,
        color: *const vk::ClearColorValue,
        count: u32,
        ranges: *const vk::ImageSubresourceRange,
    ) {
        // SAFETY: the caller's.
        let (color, ranges) = unsafe { (*color, self.ranges.push(ranges, count)) };
        self.op(Logged::ClearColorImage(image, layout, color, ranges));
    }

    pub(super) unsafe fn cmd_clear_depth_stencil_image(
        &mut self,
        image: vk::Image,
        layout: vk::ImageLayout,
        value: *const vk::ClearDepthStencilValue,
        count: u32,
        ranges: *const vk::ImageSubresourceRange,
    ) {
        // SAFETY: the caller's.
        let (value, ranges) = unsafe { (*value, self.ranges.push(ranges, count)) };
        self.op(Logged::ClearDepthStencilImage(image, layout, value, ranges));
    }

    pub(super) unsafe fn cmd_clear_attachments(
        &mut self,
        count: u32,
        attachments: *const vk::ClearAttachment,
        rect_count: u32,
        rects: *const vk::ClearRect,
    ) {
        // SAFETY: the caller's.
        let (attachments, rects) = unsafe {
            (
                self.clear_attachments.push(attachments, count),
                self.clear_rects.push(rects, rect_count),
            )
        };
        self.op(Logged::ClearAttachments(attachments, rects));
    }

    pub(super) unsafe fn cmd_pipeline_barrier(
        &mut self,
        src: vk::PipelineStageFlags,
        dst: vk::PipelineStageFlags,
        flags: vk::DependencyFlags,
        memory_count: u32,
        memory: *const vk::MemoryBarrier<'_>,
        buffer_count: u32,
        buffers: *const vk::BufferMemoryBarrier<'_>,
        image_count: u32,
        images: *const vk::ImageMemoryBarrier<'_>,
    ) {
        // SAFETY: the caller's; each structure is copied whole, its chain checked empty.
        let (memory, buffers, images) = unsafe {
            (
                self.memory_barriers.push(memory.cast(), memory_count),
                self.buffer_barriers.push(buffers.cast(), buffer_count),
                self.image_barriers.push(images.cast(), image_count),
            )
        };
        let chained = self
            .memory_barriers
            .get(memory)
            .iter()
            .any(|b| !b.p_next.is_null())
            || self
                .buffer_barriers
                .get(buffers)
                .iter()
                .any(|b| !b.p_next.is_null())
            || self
                .image_barriers
                .get(images)
                .iter()
                .any(|b| !b.p_next.is_null());
        if chained {
            unchained("barrier");
        }
        self.op(Logged::PipelineBarrier {
            src,
            dst,
            flags,
            memory,
            buffers,
            images,
        });
    }

    pub(super) unsafe fn cmd_begin_query(
        &mut self,
        pool: vk::QueryPool,
        query: u32,
        flags: vk::QueryControlFlags,
    ) {
        self.op(Logged::BeginQuery(pool, query, flags));
    }

    pub(super) unsafe fn cmd_end_query(&mut self, pool: vk::QueryPool, query: u32) {
        self.op(Logged::EndQuery(pool, query));
    }

    pub(super) unsafe fn cmd_reset_query_pool(
        &mut self,
        pool: vk::QueryPool,
        first: u32,
        count: u32,
    ) {
        self.op(Logged::ResetQueryPool(pool, first, count));
    }

    pub(super) unsafe fn cmd_write_timestamp(
        &mut self,
        stage: vk::PipelineStageFlags,
        pool: vk::QueryPool,
        query: u32,
    ) {
        self.op(Logged::WriteTimestamp(stage, pool, query));
    }

    pub(super) unsafe fn cmd_copy_query_pool_results(
        &mut self,
        pool: vk::QueryPool,
        first: u32,
        count: u32,
        dst: vk::Buffer,
        offset: vk::DeviceSize,
        stride: vk::DeviceSize,
        flags: vk::QueryResultFlags,
    ) {
        self.op(Logged::CopyQueryPoolResults {
            pool,
            first,
            count,
            dst,
            offset,
            stride,
            flags,
        });
    }

    pub(super) unsafe fn cmd_push_constants(
        &mut self,
        layout: vk::PipelineLayout,
        stages: vk::ShaderStageFlags,
        offset: u32,
        size: u32,
        values: *const c_void,
    ) {
        // SAFETY: the caller's.
        let bytes = unsafe { self.bytes.push(values.cast(), size) };
        self.op(Logged::PushConstants {
            layout,
            stages,
            offset,
            bytes,
        });
    }

    pub(super) unsafe fn cmd_begin_render_pass(
        &mut self,
        begin: *const vk::RenderPassBeginInfo<'_>,
        contents: vk::SubpassContents,
    ) {
        // SAFETY: the caller's.
        let begin = unsafe { &*begin };
        if !begin.p_next.is_null() {
            unchained("render pass begin");
        }
        // SAFETY: the caller's.
        let clears = unsafe {
            self.clear_values
                .push(begin.p_clear_values, begin.clear_value_count)
        };
        self.op(Logged::BeginRenderPass {
            render_pass: begin.render_pass,
            framebuffer: begin.framebuffer,
            area: begin.render_area,
            clears,
            contents,
        });
    }

    pub(super) unsafe fn cmd_end_render_pass(&mut self) {
        self.op(Logged::EndRenderPass);
    }

    pub(super) unsafe fn cmd_execute_commands(
        &mut self,
        count: u32,
        buffers: *const vk::CommandBuffer,
    ) {
        // SAFETY: the caller's.
        let buffers = unsafe { self.command_buffers.push(buffers, count) };
        self.op(Logged::ExecuteCommands(buffers));
    }

    pub(super) unsafe fn cmd_set_cull_mode(&mut self, mode: vk::CullModeFlags) {
        self.op(Logged::SetCullMode(mode));
    }

    pub(super) unsafe fn cmd_set_front_face(&mut self, face: vk::FrontFace) {
        self.op(Logged::SetFrontFace(face));
    }

    pub(super) unsafe fn cmd_set_primitive_topology(&mut self, topology: vk::PrimitiveTopology) {
        self.op(Logged::SetPrimitiveTopology(topology));
    }

    pub(super) unsafe fn cmd_bind_vertex_buffers2(
        &mut self,
        first: u32,
        count: u32,
        buffers: *const vk::Buffer,
        offsets: *const vk::DeviceSize,
        sizes: *const vk::DeviceSize,
        strides: *const vk::DeviceSize,
    ) {
        // SAFETY: the caller's; a null array is none.
        let op = unsafe {
            Logged::BindVertexBuffers2 {
                first,
                buffers: self.buffers.push(buffers, count),
                offsets: self.words.push(offsets, count),
                sizes: (!sizes.is_null()).then(|| self.words.push(sizes, count)),
                strides: (!strides.is_null()).then(|| self.words.push(strides, count)),
            }
        };
        self.op(op);
    }

    pub(super) unsafe fn cmd_set_depth_test_enable(&mut self, enable: vk::Bool32) {
        self.op(Logged::SetDepthTestEnable(enable));
    }

    pub(super) unsafe fn cmd_set_depth_write_enable(&mut self, enable: vk::Bool32) {
        self.op(Logged::SetDepthWriteEnable(enable));
    }

    pub(super) unsafe fn cmd_set_depth_compare_op(&mut self, op: vk::CompareOp) {
        self.op(Logged::SetDepthCompareOp(op));
    }

    pub(super) unsafe fn cmd_set_stencil_test_enable(&mut self, enable: vk::Bool32) {
        self.op(Logged::SetStencilTestEnable(enable));
    }

    pub(super) unsafe fn cmd_set_stencil_op(
        &mut self,
        faces: vk::StencilFaceFlags,
        fail: vk::StencilOp,
        pass: vk::StencilOp,
        depth_fail: vk::StencilOp,
        compare: vk::CompareOp,
    ) {
        self.op(Logged::SetStencilOp(faces, fail, pass, depth_fail, compare));
    }

    pub(super) unsafe fn cmd_set_primitive_restart_enable(&mut self, enable: vk::Bool32) {
        self.op(Logged::SetPrimitiveRestartEnable(enable));
    }

    pub(super) unsafe fn cmd_push_descriptor_set(
        &mut self,
        point: vk::PipelineBindPoint,
        layout: vk::PipelineLayout,
        set: u32,
        count: u32,
        writes: *const vk::WriteDescriptorSet<'_>,
    ) {
        let start = self.writes.0.len() as u32;
        for at in 0..count as usize {
            // SAFETY: the caller's: `count` writes, each with as many descriptors of its kind as
            // it counts.
            let write = unsafe { &*writes.add(at) };
            if !write.p_next.is_null() {
                unchained("pushed write");
            }
            let n = write.descriptor_count;
            // SAFETY: as above; the arrays a kind does not read may be anything, so only the
            // kind's own is read.
            let (images, buffers, views) = unsafe {
                match write.descriptor_type {
                    vk::DescriptorType::SAMPLER
                    | vk::DescriptorType::COMBINED_IMAGE_SAMPLER
                    | vk::DescriptorType::SAMPLED_IMAGE
                    | vk::DescriptorType::STORAGE_IMAGE
                    | vk::DescriptorType::INPUT_ATTACHMENT => (
                        self.image_infos.push(write.p_image_info, n),
                        Span::default(),
                        Span::default(),
                    ),
                    vk::DescriptorType::UNIFORM_TEXEL_BUFFER
                    | vk::DescriptorType::STORAGE_TEXEL_BUFFER => (
                        Span::default(),
                        Span::default(),
                        self.views.push(write.p_texel_buffer_view, n),
                    ),
                    vk::DescriptorType::UNIFORM_BUFFER
                    | vk::DescriptorType::STORAGE_BUFFER
                    | vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC
                    | vk::DescriptorType::STORAGE_BUFFER_DYNAMIC => (
                        Span::default(),
                        self.buffer_infos.push(write.p_buffer_info, n),
                        Span::default(),
                    ),
                    _ => unchained("pushed write of another descriptor type"),
                }
            };
            self.writes.one(Write {
                binding: write.dst_binding,
                element: write.dst_array_element,
                count: n,
                kind: write.descriptor_type,
                images,
                buffers,
                views,
            });
        }
        let writes = Span { start, len: count };
        self.op(Logged::PushDescriptorSet {
            point,
            layout,
            set,
            writes,
        });
    }

    pub(super) unsafe fn cmd_set_color_write_mask(
        &mut self,
        first: u32,
        count: u32,
        masks: *const vk::ColorComponentFlags,
    ) {
        // SAFETY: the caller's.
        let masks = unsafe { self.masks.push(masks, count) };
        self.op(Logged::SetColorWriteMask(first, masks));
    }

    pub(super) unsafe fn cmd_set_depth_bias2(&mut self, info: *const vk::DepthBiasInfoEXT<'_>) {
        // SAFETY: the caller's.
        let info = unsafe { &*info };
        let representation = match info.p_next.is_null() {
            true => None,
            false => {
                // SAFETY: the caller's: a chained structure starts with its type.
                let chained =
                    unsafe { &*info.p_next.cast::<vk::DepthBiasRepresentationInfoEXT<'_>>() };
                if chained.s_type != vk::StructureType::DEPTH_BIAS_REPRESENTATION_INFO_EXT
                    || !chained.p_next.is_null()
                {
                    unchained("depth bias");
                }
                Some((chained.depth_bias_representation, chained.depth_bias_exact))
            }
        };
        self.op(Logged::SetDepthBias2 {
            factors: [
                info.depth_bias_constant_factor,
                info.depth_bias_clamp,
                info.depth_bias_slope_factor,
            ],
            representation,
        });
    }

    pub(super) unsafe fn cmd_bind_index_buffer2(
        &mut self,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        size: vk::DeviceSize,
        index_type: vk::IndexType,
    ) {
        self.op(Logged::BindIndexBuffer2(buffer, offset, size, index_type));
    }

    pub(super) unsafe fn cmd_draw_mesh_tasks(&mut self, x: u32, y: u32, z: u32) {
        self.op(Logged::DrawMeshTasks(x, y, z));
    }

    pub(super) unsafe fn cmd_draw_mesh_tasks_indirect(
        &mut self,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        count: u32,
        stride: u32,
    ) {
        self.op(Logged::DrawMeshTasksIndirect(buffer, offset, count, stride));
    }

    /// The info, copied, its chain checked empty.
    ///
    /// # Safety
    ///
    /// `info` points at a valid info.
    unsafe fn generated_info(&mut self, info: *const GeneratedCommandsInfo) -> Span {
        // SAFETY: the caller's.
        let span = unsafe { self.generated.push(info, 1) };
        if self.generated.get(span).iter().any(|i| !i.p_next.is_null()) {
            unchained("generated commands");
        }
        span
    }

    pub(super) unsafe fn cmd_execute_generated_commands(
        &mut self,
        preprocessed: vk::Bool32,
        info: *const GeneratedCommandsInfo,
    ) {
        // SAFETY: the caller's.
        let info = unsafe { self.generated_info(info) };
        self.op(Logged::ExecuteGeneratedCommands(preprocessed, info));
    }

    pub(super) unsafe fn cmd_preprocess_generated_commands(
        &mut self,
        info: *const GeneratedCommandsInfo,
        state: vk::CommandBuffer,
    ) {
        // SAFETY: the caller's.
        let info = unsafe { self.generated_info(info) };
        let state = (state != handles::handle(self.index)).then_some(state);
        self.op(Logged::PreprocessGeneratedCommands(info, state));
    }
}

/// What a replay names in place of each deferred command buffer it executes: its real one, once
/// replayed; `None` where it was not.
pub(super) type Resolve<'a> = &'a dyn Fn(vk::CommandBuffer) -> Option<vk::CommandBuffer>;

impl Log {
    /// The begin of its replay into `real`, of slot `slot`'s device, with its secondaries' real
    /// command buffers from `resolve`, each command it calls counted in `counts` at its index
    /// (`CommandsReplayed`).
    ///
    /// # Safety
    ///
    /// `real` is a command buffer of the slot's device in the initial or executable state, only
    /// this call records it or any command buffer of its pool, and every object the log names is
    /// live.
    pub(super) unsafe fn replay(
        &self,
        slot: usize,
        real_cb: vk::CommandBuffer,
        begun: bool,
        resolve: Resolve<'_>,
        counts: &[AtomicU64; Command::COUNT],
    ) -> Result<(), vk::Result> {
        let inheritance = self.inheritance.map(|i| {
            vk::CommandBufferInheritanceInfo::default()
                .render_pass(i.render_pass)
                .subpass(i.subpass)
                .framebuffer(i.framebuffer)
                .occlusion_query_enable(i.occlusion != vk::FALSE)
                .query_flags(i.query_flags)
                .pipeline_statistics(i.statistics)
        });
        let mut begin = vk::CommandBufferBeginInfo::default().flags(self.flags);
        if let Some(inheritance) = &inheritance {
            begin = begin.inheritance_info(inheritance);
        }
        let cb = real_cb;
        macro_rules! call {
            ($command:ident, $pfn:ty, $($arg:expr),*) => {{
                // SAFETY: the slot's device's command of this name, of this signature
                // (`thunks::real`); the arguments are the logged call's, their arrays in the
                // log's arenas.
                let f: $pfn = unsafe { real(slot, Command::$command) };
                exit_guard::called(Call::$command);
                counts[Command::$command as usize].fetch_add(1, Ordering::Relaxed);
                unsafe { f(cb, $($arg),*) }
            }};
        }
        if !begun {
            let began = call!(BeginCommandBuffer, vk::PFN_vkBeginCommandBuffer, &begin);
            if began != vk::Result::SUCCESS {
                return Err(began);
            }
        }
        let recorded = SCRATCH.with_borrow_mut(|scratch| -> Result<(), vk::Result> {
            for op in &self.ops {
                match *op {
                    Logged::BindPipeline(point, pipeline) => {
                        call!(CmdBindPipeline, vk::PFN_vkCmdBindPipeline, point, pipeline)
                    }
                    Logged::SetViewport(first, span) => call!(
                        CmdSetViewport,
                        vk::PFN_vkCmdSetViewport,
                        first,
                        span.len,
                        self.viewports.get_ptr(span)
                    ),
                    Logged::SetScissor(first, span) => call!(
                        CmdSetScissor,
                        vk::PFN_vkCmdSetScissor,
                        first,
                        span.len,
                        self.rects.get_ptr(span)
                    ),
                    Logged::SetDepthBias(a, b, c) => {
                        call!(CmdSetDepthBias, vk::PFN_vkCmdSetDepthBias, a, b, c)
                    }
                    Logged::SetBlendConstants(constants) => call!(
                        CmdSetBlendConstants,
                        vk::PFN_vkCmdSetBlendConstants,
                        &constants
                    ),
                    Logged::SetStencilCompareMask(faces, mask) => call!(
                        CmdSetStencilCompareMask,
                        vk::PFN_vkCmdSetStencilCompareMask,
                        faces,
                        mask
                    ),
                    Logged::SetStencilWriteMask(faces, mask) => call!(
                        CmdSetStencilWriteMask,
                        vk::PFN_vkCmdSetStencilWriteMask,
                        faces,
                        mask
                    ),
                    Logged::SetStencilReference(faces, reference) => call!(
                        CmdSetStencilReference,
                        vk::PFN_vkCmdSetStencilReference,
                        faces,
                        reference
                    ),
                    Logged::BindDescriptorSets {
                        point,
                        layout,
                        first,
                        sets,
                        offsets,
                    } => call!(
                        CmdBindDescriptorSets,
                        vk::PFN_vkCmdBindDescriptorSets,
                        point,
                        layout,
                        first,
                        sets.len,
                        self.sets.get_ptr(sets),
                        offsets.len,
                        self.u32s.get_ptr(offsets)
                    ),
                    Logged::BindIndexBuffer(buffer, offset, index_type) => call!(
                        CmdBindIndexBuffer,
                        vk::PFN_vkCmdBindIndexBuffer,
                        buffer,
                        offset,
                        index_type
                    ),
                    Logged::BindVertexBuffers {
                        first,
                        buffers,
                        offsets,
                    } => call!(
                        CmdBindVertexBuffers,
                        vk::PFN_vkCmdBindVertexBuffers,
                        first,
                        buffers.len,
                        self.buffers.get_ptr(buffers),
                        self.words.get_ptr(offsets)
                    ),
                    Logged::Draw(a, b, c, d) => call!(CmdDraw, vk::PFN_vkCmdDraw, a, b, c, d),
                    Logged::DrawIndexed(a, b, c, d, e) => {
                        call!(CmdDrawIndexed, vk::PFN_vkCmdDrawIndexed, a, b, c, d, e)
                    }
                    Logged::DrawIndirect(buffer, offset, count, stride) => call!(
                        CmdDrawIndirect,
                        vk::PFN_vkCmdDrawIndirect,
                        buffer,
                        offset,
                        count,
                        stride
                    ),
                    Logged::DrawIndexedIndirect(buffer, offset, count, stride) => call!(
                        CmdDrawIndexedIndirect,
                        vk::PFN_vkCmdDrawIndexedIndirect,
                        buffer,
                        offset,
                        count,
                        stride
                    ),
                    Logged::Dispatch(x, y, z) => call!(CmdDispatch, vk::PFN_vkCmdDispatch, x, y, z),
                    Logged::DispatchIndirect(buffer, offset) => call!(
                        CmdDispatchIndirect,
                        vk::PFN_vkCmdDispatchIndirect,
                        buffer,
                        offset
                    ),
                    Logged::CopyBuffer(src, dst, regions) => call!(
                        CmdCopyBuffer,
                        vk::PFN_vkCmdCopyBuffer,
                        src,
                        dst,
                        regions.len,
                        self.buffer_copies.get_ptr(regions)
                    ),
                    Logged::CopyImage {
                        src,
                        src_layout,
                        dst,
                        dst_layout,
                        regions,
                    } => call!(
                        CmdCopyImage,
                        vk::PFN_vkCmdCopyImage,
                        src,
                        src_layout,
                        dst,
                        dst_layout,
                        regions.len,
                        self.image_copies.get_ptr(regions)
                    ),
                    Logged::BlitImage {
                        src,
                        src_layout,
                        dst,
                        dst_layout,
                        regions,
                        filter,
                    } => call!(
                        CmdBlitImage,
                        vk::PFN_vkCmdBlitImage,
                        src,
                        src_layout,
                        dst,
                        dst_layout,
                        regions.len,
                        self.blits.get_ptr(regions),
                        filter
                    ),
                    Logged::CopyBufferToImage {
                        src,
                        dst,
                        layout,
                        regions,
                    } => call!(
                        CmdCopyBufferToImage,
                        vk::PFN_vkCmdCopyBufferToImage,
                        src,
                        dst,
                        layout,
                        regions.len,
                        self.buffer_images.get_ptr(regions)
                    ),
                    Logged::CopyImageToBuffer {
                        src,
                        layout,
                        dst,
                        regions,
                    } => call!(
                        CmdCopyImageToBuffer,
                        vk::PFN_vkCmdCopyImageToBuffer,
                        src,
                        layout,
                        dst,
                        regions.len,
                        self.buffer_images.get_ptr(regions)
                    ),
                    Logged::UpdateBuffer(dst, offset, bytes) => call!(
                        CmdUpdateBuffer,
                        vk::PFN_vkCmdUpdateBuffer,
                        dst,
                        offset,
                        u64::from(bytes.len),
                        self.bytes.get_ptr(bytes).cast()
                    ),
                    Logged::FillBuffer(dst, offset, size, data) => call!(
                        CmdFillBuffer,
                        vk::PFN_vkCmdFillBuffer,
                        dst,
                        offset,
                        size,
                        data
                    ),
                    Logged::ClearColorImage(image, layout, color, ranges) => call!(
                        CmdClearColorImage,
                        vk::PFN_vkCmdClearColorImage,
                        image,
                        layout,
                        &color,
                        ranges.len,
                        self.ranges.get_ptr(ranges)
                    ),
                    Logged::ClearDepthStencilImage(image, layout, value, ranges) => call!(
                        CmdClearDepthStencilImage,
                        vk::PFN_vkCmdClearDepthStencilImage,
                        image,
                        layout,
                        &value,
                        ranges.len,
                        self.ranges.get_ptr(ranges)
                    ),
                    Logged::ClearAttachments(attachments, rects) => call!(
                        CmdClearAttachments,
                        vk::PFN_vkCmdClearAttachments,
                        attachments.len,
                        self.clear_attachments.get_ptr(attachments),
                        rects.len,
                        self.clear_rects.get_ptr(rects)
                    ),
                    Logged::PipelineBarrier {
                        src,
                        dst,
                        flags,
                        memory,
                        buffers,
                        images,
                    } => call!(
                        CmdPipelineBarrier,
                        vk::PFN_vkCmdPipelineBarrier,
                        src,
                        dst,
                        flags,
                        memory.len,
                        self.memory_barriers.get_ptr(memory),
                        buffers.len,
                        self.buffer_barriers.get_ptr(buffers),
                        images.len,
                        self.image_barriers.get_ptr(images)
                    ),
                    Logged::BeginQuery(pool, query, flags) => {
                        call!(CmdBeginQuery, vk::PFN_vkCmdBeginQuery, pool, query, flags)
                    }
                    Logged::EndQuery(pool, query) => {
                        call!(CmdEndQuery, vk::PFN_vkCmdEndQuery, pool, query)
                    }
                    Logged::ResetQueryPool(pool, first, count) => call!(
                        CmdResetQueryPool,
                        vk::PFN_vkCmdResetQueryPool,
                        pool,
                        first,
                        count
                    ),
                    Logged::WriteTimestamp(stage, pool, query) => call!(
                        CmdWriteTimestamp,
                        vk::PFN_vkCmdWriteTimestamp,
                        stage,
                        pool,
                        query
                    ),
                    Logged::CopyQueryPoolResults {
                        pool,
                        first,
                        count,
                        dst,
                        offset,
                        stride,
                        flags,
                    } => call!(
                        CmdCopyQueryPoolResults,
                        vk::PFN_vkCmdCopyQueryPoolResults,
                        pool,
                        first,
                        count,
                        dst,
                        offset,
                        stride,
                        flags
                    ),
                    Logged::PushConstants {
                        layout,
                        stages,
                        offset,
                        bytes,
                    } => call!(
                        CmdPushConstants,
                        vk::PFN_vkCmdPushConstants,
                        layout,
                        stages,
                        offset,
                        bytes.len,
                        self.bytes.get_ptr(bytes).cast()
                    ),
                    Logged::BeginRenderPass {
                        render_pass,
                        framebuffer,
                        area,
                        clears,
                        contents,
                    } => {
                        let begin = vk::RenderPassBeginInfo {
                            render_pass,
                            framebuffer,
                            render_area: area,
                            clear_value_count: clears.len,
                            p_clear_values: self.clear_values.get_ptr(clears),
                            ..Default::default()
                        };
                        call!(
                            CmdBeginRenderPass,
                            vk::PFN_vkCmdBeginRenderPass,
                            &begin,
                            contents
                        )
                    }
                    Logged::EndRenderPass => call!(CmdEndRenderPass, vk::PFN_vkCmdEndRenderPass,),
                    Logged::ExecuteCommands(buffers) => {
                        let resolved = &mut scratch.secondaries;
                        resolved.clear();
                        for &secondary in self.command_buffers.get(buffers) {
                            resolved.push(resolve(secondary).ok_or(vk::Result::ERROR_UNKNOWN)?);
                        }
                        call!(
                            CmdExecuteCommands,
                            vk::PFN_vkCmdExecuteCommands,
                            resolved.len() as u32,
                            resolved.as_ptr()
                        )
                    }
                    Logged::SetCullMode(mode) => {
                        call!(CmdSetCullMode, vk::PFN_vkCmdSetCullMode, mode)
                    }
                    Logged::SetFrontFace(face) => {
                        call!(CmdSetFrontFace, vk::PFN_vkCmdSetFrontFace, face)
                    }
                    Logged::SetPrimitiveTopology(topology) => call!(
                        CmdSetPrimitiveTopology,
                        vk::PFN_vkCmdSetPrimitiveTopology,
                        topology
                    ),
                    Logged::BindVertexBuffers2 {
                        first,
                        buffers,
                        offsets,
                        sizes,
                        strides,
                    } => call!(
                        CmdBindVertexBuffers2,
                        vk::PFN_vkCmdBindVertexBuffers2,
                        first,
                        buffers.len,
                        self.buffers.get_ptr(buffers),
                        self.words.get_ptr(offsets),
                        sizes.map_or(std::ptr::null(), |s| self.words.get_ptr(s)),
                        strides.map_or(std::ptr::null(), |s| self.words.get_ptr(s))
                    ),
                    Logged::SetDepthTestEnable(enable) => call!(
                        CmdSetDepthTestEnable,
                        vk::PFN_vkCmdSetDepthTestEnable,
                        enable
                    ),
                    Logged::SetDepthWriteEnable(enable) => call!(
                        CmdSetDepthWriteEnable,
                        vk::PFN_vkCmdSetDepthWriteEnable,
                        enable
                    ),
                    Logged::SetDepthCompareOp(op) => {
                        call!(CmdSetDepthCompareOp, vk::PFN_vkCmdSetDepthCompareOp, op)
                    }
                    Logged::SetStencilTestEnable(enable) => call!(
                        CmdSetStencilTestEnable,
                        vk::PFN_vkCmdSetStencilTestEnable,
                        enable
                    ),
                    Logged::SetStencilOp(faces, fail, pass, depth_fail, compare) => call!(
                        CmdSetStencilOp,
                        vk::PFN_vkCmdSetStencilOp,
                        faces,
                        fail,
                        pass,
                        depth_fail,
                        compare
                    ),
                    Logged::SetPrimitiveRestartEnable(enable) => call!(
                        CmdSetPrimitiveRestartEnable,
                        vk::PFN_vkCmdSetPrimitiveRestartEnable,
                        enable
                    ),
                    Logged::PushDescriptorSet {
                        point,
                        layout,
                        set,
                        writes,
                    } => {
                        let built = &mut scratch.writes;
                        built.clear();
                        built.extend(self.writes.get(writes).iter().map(|w| {
                            vk::WriteDescriptorSet {
                                dst_binding: w.binding,
                                dst_array_element: w.element,
                                descriptor_count: w.count,
                                descriptor_type: w.kind,
                                p_image_info: self.image_infos.get_ptr(w.images),
                                p_buffer_info: self.buffer_infos.get_ptr(w.buffers),
                                p_texel_buffer_view: self.views.get_ptr(w.views),
                                ..Default::default()
                            }
                        }));
                        call!(
                            CmdPushDescriptorSet,
                            vk::PFN_vkCmdPushDescriptorSetKHR,
                            point,
                            layout,
                            set,
                            built.len() as u32,
                            built.as_ptr()
                        );
                        // No pointer into the log outlives its call.
                        built.clear();
                    }
                    Logged::SetColorWriteMask(first, masks) => call!(
                        CmdSetColorWriteMask,
                        vk::PFN_vkCmdSetColorWriteMaskEXT,
                        first,
                        masks.len,
                        self.masks.get_ptr(masks)
                    ),
                    Logged::SetDepthBias2 {
                        factors,
                        representation,
                    } => {
                        let mut chained = representation.map(|(representation, exact)| {
                            vk::DepthBiasRepresentationInfoEXT::default()
                                .depth_bias_representation(representation)
                                .depth_bias_exact(exact != vk::FALSE)
                        });
                        let mut info = vk::DepthBiasInfoEXT::default()
                            .depth_bias_constant_factor(factors[0])
                            .depth_bias_clamp(factors[1])
                            .depth_bias_slope_factor(factors[2]);
                        if let Some(chained) = &mut chained {
                            info = info.push_next(chained);
                        }
                        call!(CmdSetDepthBias2, vk::PFN_vkCmdSetDepthBias2EXT, &info)
                    }
                    Logged::BindIndexBuffer2(buffer, offset, size, index_type) => call!(
                        CmdBindIndexBuffer2,
                        vk::PFN_vkCmdBindIndexBuffer2KHR,
                        buffer,
                        offset,
                        size,
                        index_type
                    ),
                    Logged::DrawMeshTasks(x, y, z) => {
                        call!(CmdDrawMeshTasks, vk::PFN_vkCmdDrawMeshTasksEXT, x, y, z)
                    }
                    Logged::DrawMeshTasksIndirect(buffer, offset, count, stride) => call!(
                        CmdDrawMeshTasksIndirect,
                        vk::PFN_vkCmdDrawMeshTasksIndirectEXT,
                        buffer,
                        offset,
                        count,
                        stride
                    ),
                    Logged::ExecuteGeneratedCommands(preprocessed, info) => call!(
                        CmdExecuteGeneratedCommands,
                        unsafe extern "system" fn(
                            vk::CommandBuffer,
                            vk::Bool32,
                            *const GeneratedCommandsInfo,
                        ),
                        preprocessed,
                        self.generated.get_ptr(info)
                    ),
                    Logged::PreprocessGeneratedCommands(info, state) => {
                        let state = match state {
                            None => cb,
                            Some(other) => resolve(other).unwrap_or(other),
                        };
                        call!(
                            CmdPreprocessGeneratedCommands,
                            unsafe extern "system" fn(
                                vk::CommandBuffer,
                                *const GeneratedCommandsInfo,
                                vk::CommandBuffer,
                            ),
                            self.generated.get_ptr(info),
                            state
                        )
                    }
                }
            }
            Ok(())
        });
        recorded?;
        let ended = call!(EndCommandBuffer, vk::PFN_vkEndCommandBuffer,);
        match ended {
            vk::Result::SUCCESS => Ok(()),
            error => Err(error),
        }
    }
}
