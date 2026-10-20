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
    fn set_render_pipeline(&self, encoder: &mut Self::RenderEncoder, state: &Self::RenderPipeline);
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
