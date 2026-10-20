//! The Metal calls a workload makes, as one trait both sides implement: native Metal
//! (`native.rs`) and the runtime (`runtime.rs`). Each method is one Metal call, or the few calls a
//! descriptor takes, so a workload written once against it makes the same calls on either side.

use std::path::Path;

/// The pixel formats the workloads use.
#[derive(Clone, Copy, Debug)]
pub enum Format {
    Bgra8,
    Rgba8,
    Rgba16Float,
    /// A depth attachment's (`texture` with `target`).
    Depth32Float,
}

/// One side of the A/B: a device, its queue, and Metal's calls on them.
pub trait Gpu {
    type Library;
    type RenderPipeline;
    type ComputePipeline;
    type Buffer;
    type Texture;
    type Sampler;
    type CommandBuffer;
    type RenderEncoder;
    type ComputeEncoder;
    type Heap;
    type DepthStencil;

    /// The side and the device's name.
    fn label(&self) -> String;

    /// `newLibraryWithURL:` (native) or `newLibraryWithData:` of the file's bytes (the runtime).
    fn library(&self, metallib: &Path) -> Self::Library;
    /// `newRenderPipelineStateWithDescriptor:error:` of two functions and one color attachment.
    fn render_pipeline(
        &self,
        library: &Self::Library,
        vertex: &str,
        fragment: &str,
        format: Format,
        blend: bool,
    ) -> Self::RenderPipeline;
    /// As `render_pipeline` into `BGRA8Unorm`, with a `Depth32Float` depth attachment, blending
    /// source over destination by the source's alpha where `blend`.
    fn depth_pipeline(
        &self,
        library: &Self::Library,
        vertex: &str,
        fragment: &str,
        blend: bool,
    ) -> Self::RenderPipeline;
    /// `newDepthStencilStateWithDescriptor:`: a depth compare of `Less` with writes, or of
    /// `LessEqual` without.
    fn depth_stencil(&self, writes: bool) -> Self::DepthStencil;
    /// `newComputePipelineStateWithFunction:error:`.
    fn compute_pipeline(&self, library: &Self::Library, kernel: &str) -> Self::ComputePipeline;

    /// `newBufferWithLength:options:` of `Shared` storage.
    fn buffer(&self, length: usize) -> Self::Buffer;
    /// `newBufferWithBytes:length:options:` of `Shared` storage.
    fn buffer_with(&self, bytes: &[u8]) -> Self::Buffer;
    /// `newBufferWithBytesNoCopy:length:options:deallocator:` of `Shared` storage, no
    /// deallocator.
    ///
    /// # Safety
    ///
    /// `pointer` is page-aligned, `length` (a multiple of the page size) bytes of read-write
    /// memory that outlive the buffer.
    unsafe fn buffer_no_copy(&self, pointer: *mut u8, length: usize) -> Self::Buffer;
    /// `newHeapWithDescriptor:` of a `Placement` heap of `Shared` storage, `size` bytes.
    fn heap(&self, size: usize) -> Self::Heap;
    /// The runtime's heap of the same over `size` bytes of host pages at `pointer`
    /// (`new_heap_with_bytes_no_copy`); none natively.
    ///
    /// # Safety
    ///
    /// `pointer` is page-aligned, `size` (a multiple of the page size) bytes of read-write memory
    /// that outlive the heap and every texture placed in it.
    unsafe fn heap_no_copy(&self, pointer: *mut u8, size: usize) -> Option<Self::Heap>;
    /// `newTextureWithDescriptor:offset:` of `heap`: a 2D `BGRA8Unorm` texture of one level that
    /// passes render to and the shaders read, at `offset`, whose contents Metal does not optimize
    /// for the GPU (`allowGPUOptimizedContents` off), so its bytes in the heap are its texels.
    fn heap_texture(
        &self,
        heap: &Self::Heap,
        width: usize,
        height: usize,
        offset: usize,
    ) -> Self::Texture;
    /// A 2D texture of one level: a render target of `Private` storage, or a `Shared` one the
    /// shaders read.
    fn texture(&self, format: Format, width: usize, height: usize, target: bool) -> Self::Texture;
    /// `replaceRegion:mipmapLevel:withBytes:bytesPerRow:` over the whole level 0.
    fn replace_region(
        &self,
        texture: &Self::Texture,
        size: (usize, usize),
        bytes: &[u8],
        row: usize,
    );
    /// `getBytes:bytesPerRow:fromRegion:mipmapLevel:` of the whole level 0.
    fn get_bytes(
        &self,
        texture: &Self::Texture,
        size: (usize, usize),
        bytes: &mut [u8],
        row: usize,
    );
    /// The first `bytes.len()` bytes of a `Shared` buffer's `contents`, no GPU work writing them.
    fn read_buffer(&self, buffer: &Self::Buffer, bytes: &mut [u8]);
    /// `newSamplerStateWithDescriptor:`, nearest or linear.
    fn sampler(&self, linear: bool) -> Self::Sampler;

    /// `commandBuffer`.
    fn command_buffer(&self) -> Self::CommandBuffer;
    /// `renderCommandEncoderWithDescriptor:` of one color attachment, cleared or loaded, stored.
    fn render_encoder(
        &self,
        command_buffer: &Self::CommandBuffer,
        target: &Self::Texture,
        clear: bool,
    ) -> Self::RenderEncoder;
    /// `renderCommandEncoderWithDescriptor:` of one color attachment and a depth attachment,
    /// both cleared, the color stored and the depth not.
    fn depth_render_encoder(
        &self,
        command_buffer: &Self::CommandBuffer,
        target: &Self::Texture,
        depth: &Self::Texture,
    ) -> Self::RenderEncoder;
    fn set_render_pipeline(&self, encoder: &mut Self::RenderEncoder, state: &Self::RenderPipeline);
    fn set_depth_stencil(&self, encoder: &mut Self::RenderEncoder, state: &Self::DepthStencil);
    /// `setCullMode:` of `Back`, or of `None`.
    fn set_cull_back(&self, encoder: &mut Self::RenderEncoder, back: bool);
    /// `setFrontFacingWinding:` of `CounterClockwise`.
    fn set_counter_clockwise(&self, encoder: &mut Self::RenderEncoder);
    /// `setVertexBytes:length:atIndex:`.
    fn set_vertex_bytes(&self, encoder: &mut Self::RenderEncoder, bytes: &[u8], index: usize);
    fn set_vertex_buffer(
        &self,
        encoder: &mut Self::RenderEncoder,
        buffer: &Self::Buffer,
        offset: usize,
        index: usize,
    );
    fn set_fragment_buffer(
        &self,
        encoder: &mut Self::RenderEncoder,
        buffer: &Self::Buffer,
        offset: usize,
        index: usize,
    );
    fn set_fragment_texture(
        &self,
        encoder: &mut Self::RenderEncoder,
        texture: &Self::Texture,
        index: usize,
    );
    fn set_fragment_sampler(
        &self,
        encoder: &mut Self::RenderEncoder,
        sampler: &Self::Sampler,
        index: usize,
    );
    /// `drawPrimitives:vertexStart:vertexCount:` of a triangle strip.
    fn draw_strip(&self, encoder: &mut Self::RenderEncoder, vertices: usize);
    /// `drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:` of triangles
    /// and 16-bit indices from the buffer's start.
    fn draw_indexed(&self, encoder: &mut Self::RenderEncoder, count: usize, indices: &Self::Buffer);
    fn end_render(&self, encoder: Self::RenderEncoder);

    /// `computeCommandEncoder`.
    fn compute_encoder(&self, command_buffer: &Self::CommandBuffer) -> Self::ComputeEncoder;
    fn set_compute_pipeline(
        &self,
        encoder: &mut Self::ComputeEncoder,
        state: &Self::ComputePipeline,
    );
    fn set_buffer(
        &self,
        encoder: &mut Self::ComputeEncoder,
        buffer: &Self::Buffer,
        offset: usize,
        index: usize,
    );
    fn set_texture(
        &self,
        encoder: &mut Self::ComputeEncoder,
        texture: &Self::Texture,
        index: usize,
    );
    /// `dispatchThreadgroups:threadsPerThreadgroup:` of a 1D grid.
    fn dispatch(&self, encoder: &mut Self::ComputeEncoder, groups: usize, threads: usize);
    fn end_compute(&self, encoder: Self::ComputeEncoder);

    /// A blit encoder of one `fillBuffer:range:value:` over the buffer's first `length` bytes.
    fn fill(&self, command_buffer: &Self::CommandBuffer, buffer: &Self::Buffer, length: usize);

    fn commit(&self, command_buffer: &Self::CommandBuffer);
    /// `waitUntilCompleted`, then a check that it completed without error.
    fn wait(&self, command_buffer: &Self::CommandBuffer);
    /// `addCompletedHandler:`.
    fn on_completed(&self, command_buffer: &Self::CommandBuffer, handler: Box<dyn FnOnce() + Send>);
    /// `GPUEndTime - GPUStartTime`, in seconds, where the side measures it on the GPU.
    fn gpu_seconds(&self, command_buffer: &Self::CommandBuffer) -> Option<f64>;
    /// `kernelStartTime`, `kernelEndTime`, `GPUStartTime` and `GPUEndTime`, in seconds of the
    /// host clock [`crate::host_seconds`] reads.
    fn times(&self, command_buffer: &Self::CommandBuffer) -> [f64; 4];

    /// Run `f` inside an autorelease pool (native), as a Metal program's frame runs.
    fn pool<R>(&self, f: impl FnOnce() -> R) -> R;
}
