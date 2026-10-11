//! The Metal API the sequences are run through: Metal's calls a sequence makes, in the runtime's
//! vocabulary, which Metal's side maps onto `objc2-metal` and the runtime's side passes on. Its
//! own trait beside the harness's [`Api`](super::super::api::Api), whose sources every other
//! scenario's records depend on: this one, the sequences' Metal side and their runner are the
//! sources the sequences' records depend on instead.

use super::super::api::{Answer, Completion};
use super::program::{Load, ViewSpec};
use metal2vulkan::metal::PixelFormat;
use metal2vulkan_runtime::{ResourceOptions, TextureDescriptor};
use std::ops::Range;
use std::ptr::NonNull;

/// Metal's API, as the sequences use it. Each implementation is one device.
pub(crate) trait SeqApi {
    type Heap;
    type Buffer;
    type Texture;
    type Kernel;
    type Draw;
    type Queue;
    type CommandBuffer;
    type Event;
    type Fence;
    type Blit;
    type Compute;
    type Render;

    /// `newBufferWithLength:options:`.
    fn buffer(&self, length: u64, options: ResourceOptions) -> Answer<Self::Buffer>;
    /// `newBufferWithBytesNoCopy:length:options:deallocator:` with no deallocator.
    ///
    /// # Safety
    ///
    /// The `length` bytes at `bytes` stay mapped, readable and writable, until the buffer and
    /// every command buffer that used it are gone.
    unsafe fn no_copy_buffer(
        &self,
        bytes: NonNull<u8>,
        length: u64,
        options: ResourceOptions,
    ) -> Answer<Self::Buffer>;
    /// `newHeapWithDescriptor:` of a placement heap of `size` bytes and `options`.
    fn heap(&self, size: u64, options: ResourceOptions) -> Answer<Self::Heap>;
    /// `newBufferWithLength:options:offset:` of a placement heap.
    fn heap_buffer(
        &self,
        heap: &Self::Heap,
        length: u64,
        options: ResourceOptions,
        offset: u64,
    ) -> Answer<Self::Buffer>;
    /// `newTextureWithDescriptor:offset:` of a placement heap.
    fn heap_texture(
        &self,
        heap: &Self::Heap,
        descriptor: &TextureDescriptor,
        offset: u64,
    ) -> Answer<Self::Texture>;
    /// Write `bytes` at the start of a `Shared` buffer's `contents`.
    fn write(&self, buffer: &Self::Buffer, bytes: &[u8]);
    /// A `Shared` buffer's `contents`, every byte.
    fn read(&self, buffer: &Self::Buffer) -> Vec<u8>;
    /// `newTextureWithDescriptor:`.
    fn texture(&self, descriptor: &TextureDescriptor) -> Answer<Self::Texture>;
    /// The view `view` describes of `texture` (its texture is `texture`), made the way its
    /// form names: `newTextureViewWithPixelFormat:`, or with `textureType:levels:slices:` (a 2D
    /// view of level 0 and the view's slice), and `swizzle:`.
    fn view(&self, texture: &Self::Texture, view: &ViewSpec) -> Answer<Self::Texture>;
    /// The compute pipeline state of the fixture's kernel `entry`.
    fn kernel(&self, entry: &str) -> Answer<Self::Kernel>;
    /// The render pipeline state of the fixture's `seq_rect` and `seq_color`, one colour
    /// attachment of `format`.
    fn draw_state(&self, format: PixelFormat) -> Answer<Self::Draw>;
    /// `newCommandQueue`.
    fn queue(&self) -> Answer<Self::Queue>;
    /// `commandBuffer`.
    fn command_buffer(&self, queue: &Self::Queue) -> Answer<Self::CommandBuffer>;
    /// `newSharedEvent`.
    fn event(&self) -> Answer<Self::Event>;
    /// `newFence`.
    fn fence(&self) -> Answer<Self::Fence>;
    /// `encodeSignalEvent:value:`.
    fn signal(&self, cb: &Self::CommandBuffer, event: &Self::Event, value: u64) -> Answer<()>;
    /// `encodeWaitForEvent:value:`.
    fn wait(&self, cb: &Self::CommandBuffer, event: &Self::Event, value: u64) -> Answer<()>;

    /// `blitCommandEncoder`.
    fn blit(&self, cb: &Self::CommandBuffer) -> Answer<Self::Blit>;
    /// `fillBuffer:range:value:`.
    fn fill(
        &self,
        blit: &mut Self::Blit,
        buffer: &Self::Buffer,
        range: Range<u64>,
        value: u8,
    ) -> Answer<()>;
    /// `copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:`.
    fn copy(
        &self,
        blit: &mut Self::Blit,
        from: (&Self::Buffer, u64),
        to: (&Self::Buffer, u64),
        size: u64,
    ) -> Answer<()>;
    /// Whole rows `rows` of slice `texture.1` of a texture of `rows_of.0` texels a row from a
    /// buffer's bytes at an offset, `rows_of.1` bytes a row (`copyFromBuffer:...toTexture:...`).
    fn buffer_to_texture(
        &self,
        blit: &mut Self::Blit,
        from: (&Self::Buffer, u64),
        texture: (&Self::Texture, u32),
        rows: Range<u32>,
        rows_of: (u32, u64),
    ) -> Answer<()>;
    /// Whole rows `rows` of slice `texture.1` of a texture of `rows_of.0` texels a row into a
    /// buffer's bytes at an offset, `rows_of.1` bytes a row (`copyFromTexture:...toBuffer:...`).
    fn texture_to_buffer(
        &self,
        blit: &mut Self::Blit,
        texture: (&Self::Texture, u32),
        rows: Range<u32>,
        rows_of: (u32, u64),
        to: (&Self::Buffer, u64),
    ) -> Answer<()>;
    /// `updateFence:` on a blit encoder.
    fn blit_update_fence(&self, blit: &mut Self::Blit, fence: &Self::Fence) -> Answer<()>;
    /// `waitForFence:` on a blit encoder.
    fn blit_wait_fence(&self, blit: &mut Self::Blit, fence: &Self::Fence) -> Answer<()>;
    /// `endEncoding` of a blit encoder.
    fn end_blit(&self, blit: Self::Blit) -> Answer<()>;

    /// `computeCommandEncoderWithDispatchType:`, concurrent or serial.
    fn compute(&self, cb: &Self::CommandBuffer, concurrent: bool) -> Answer<Self::Compute>;
    /// `setComputePipelineState:`, buffer 0 at `a` and buffer 1 at `b`, and one thread:
    /// `dispatchThreads:threadsPerThreadgroup:` of one, or with `indirect`,
    /// `dispatchThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerThreadgroup:` of
    /// one-thread groups.
    fn dispatch(
        &self,
        compute: &mut Self::Compute,
        kernel: &Self::Kernel,
        a: (&Self::Buffer, u64),
        b: (&Self::Buffer, u64),
        indirect: Option<(&Self::Buffer, u64)>,
    ) -> Answer<()>;
    /// `setComputePipelineState:`, texture 0 `texture` and buffer 0 at `words`, and
    /// `dispatchThreads:threadsPerThreadgroup:` of `rows_of.0` by `rows_of.1` threads, a row of
    /// them a threadgroup.
    fn texel_dispatch(
        &self,
        compute: &mut Self::Compute,
        kernel: &Self::Kernel,
        texture: &Self::Texture,
        words: (&Self::Buffer, u64),
        rows_of: (u32, u32),
    ) -> Answer<()>;
    /// `memoryBarrierWithScope:` of buffers, and of textures with `textures`.
    fn barrier(&self, compute: &mut Self::Compute, textures: bool) -> Answer<()>;
    /// `updateFence:` on a compute encoder.
    fn compute_update_fence(&self, compute: &mut Self::Compute, fence: &Self::Fence) -> Answer<()>;
    /// `waitForFence:` on a compute encoder.
    fn compute_wait_fence(&self, compute: &mut Self::Compute, fence: &Self::Fence) -> Answer<()>;
    /// `endEncoding` of a compute encoder.
    fn end_compute(&self, compute: Self::Compute) -> Answer<()>;

    /// `renderCommandEncoderWithDescriptor:` of a pass whose colour attachment 0 is slice
    /// `target.1` of `target.0`, loaded as `load`, stored.
    fn render(
        &self,
        cb: &Self::CommandBuffer,
        target: (&Self::Texture, u32),
        load: Load,
    ) -> Answer<Self::Render>;
    /// `setRenderPipelineState:`, vertex buffer 0 at `rects`, fragment buffer 0 at `colors`, and
    /// `drawPrimitives:` of triangles: `counts.0` vertices of `counts.1` instances from 0, or
    /// with `indirect`, `drawPrimitives:indirectBuffer:indirectBufferOffset:`.
    fn draw(
        &self,
        render: &mut Self::Render,
        state: &Self::Draw,
        rects: (&Self::Buffer, u64),
        colors: (&Self::Buffer, u64),
        counts: (u32, u32),
        indirect: Option<(&Self::Buffer, u64)>,
    ) -> Answer<()>;
    /// `updateFence:afterStages:` of the fragment stage.
    fn render_update_fence(&self, render: &mut Self::Render, fence: &Self::Fence) -> Answer<()>;
    /// `waitForFence:beforeStages:` of the vertex stage.
    fn render_wait_fence(&self, render: &mut Self::Render, fence: &Self::Fence) -> Answer<()>;
    /// `endEncoding` of a render encoder.
    fn end_render(&self, render: Self::Render) -> Answer<()>;

    /// `commit`.
    fn commit(&self, cb: &Self::CommandBuffer) -> Answer<()>;
    /// Wait for the command buffer to complete, then its status.
    fn completion(&self, cb: Self::CommandBuffer) -> Result<Completion, String>;
}
