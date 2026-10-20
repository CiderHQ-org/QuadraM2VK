//! The native side: Metal's calls on Metal itself (objc2-metal), on the system default device.

use crate::gpu::{Format, Gpu};
use block2::RcBlock;
use objc2::rc::{autoreleasepool, Retained};
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSString, NSURL};
use objc2_metal::{
    MTLBlendFactor, MTLBlitCommandEncoder, MTLBuffer, MTLClearColor, MTLCommandBuffer,
    MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue, MTLCompareFunction,
    MTLComputeCommandEncoder, MTLComputePipelineDescriptor, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLCullMode, MTLDepthStencilDescriptor, MTLDepthStencilState,
    MTLDevice, MTLDispatchType, MTLHazardTrackingMode, MTLHeap, MTLHeapDescriptor, MTLHeapType,
    MTLIndexType, MTLIndirectCommandBuffer, MTLIndirectCommandBufferDescriptor,
    MTLIndirectCommandType, MTLLibrary, MTLLoadAction, MTLOrigin, MTLPipelineOption,
    MTLPixelFormat, MTLPrimitiveType, MTLRegion, MTLRenderCommandEncoder, MTLRenderPassDescriptor,
    MTLRenderPipelineDescriptor, MTLRenderPipelineState, MTLResourceOptions, MTLResourceUsage,
    MTLSamplerDescriptor, MTLSamplerMinMagFilter, MTLSamplerState, MTLSize, MTLStorageMode,
    MTLStoreAction, MTLTexture, MTLTextureDescriptor, MTLTextureUsage, MTLWinding,
};
use std::ffi::c_void;
use std::path::Path;
use std::ptr::NonNull;
use std::sync::Mutex;

type Object<T> = Retained<ProtocolObject<T>>;

pub struct Native {
    device: Object<dyn MTLDevice>,
    queue: Object<dyn MTLCommandQueue>,
}

impl Native {
    pub fn open() -> Native {
        let device = MTLCreateSystemDefaultDevice().expect("MTLCreateSystemDefaultDevice");
        let queue = device.newCommandQueue().expect("newCommandQueue");
        Native { device, queue }
    }
}

fn pixel_format(format: Format) -> MTLPixelFormat {
    match format {
        Format::Bgra8 => MTLPixelFormat::BGRA8Unorm,
        Format::Rgba8 => MTLPixelFormat::RGBA8Unorm,
        Format::Rgba16Float => MTLPixelFormat::RGBA16Float,
        Format::Depth32Float => MTLPixelFormat::Depth32Float,
    }
}

fn region(size: (usize, usize)) -> MTLRegion {
    MTLRegion {
        origin: MTLOrigin { x: 0, y: 0, z: 0 },
        size: MTLSize {
            width: size.0,
            height: size.1,
            depth: 1,
        },
    }
}

fn function(
    library: &ProtocolObject<dyn MTLLibrary>,
    name: &str,
) -> Object<dyn objc2_metal::MTLFunction> {
    library
        .newFunctionWithName(&NSString::from_str(name))
        .unwrap_or_else(|| panic!("newFunctionWithName {name}"))
}

impl Gpu for Native {
    type Library = Object<dyn MTLLibrary>;
    type RenderPipeline = Object<dyn MTLRenderPipelineState>;
    type ComputePipeline = Object<dyn MTLComputePipelineState>;
    type Buffer = Object<dyn MTLBuffer>;
    type Texture = Object<dyn MTLTexture>;
    type Sampler = Object<dyn MTLSamplerState>;
    type CommandBuffer = Object<dyn MTLCommandBuffer>;
    type RenderEncoder = Object<dyn MTLRenderCommandEncoder>;
    type ComputeEncoder = Object<dyn MTLComputeCommandEncoder>;
    type Heap = Object<dyn MTLHeap>;
    type DepthStencil = Object<dyn MTLDepthStencilState>;
    type IndirectCommands = Object<dyn MTLIndirectCommandBuffer>;

    fn label(&self) -> String {
        format!("native Metal on {}", self.device.name())
    }

    fn library(&self, metallib: &Path) -> Self::Library {
        let path = NSString::from_str(metallib.to_str().expect("a UTF-8 path"));
        self.device
            .newLibraryWithURL_error(&NSURL::fileURLWithPath(&path))
            .expect("newLibraryWithURL")
    }

    fn render_pipeline(
        &self,
        library: &Self::Library,
        vertex: &str,
        fragment: &str,
        format: Format,
        blend: bool,
    ) -> Self::RenderPipeline {
        let descriptor = MTLRenderPipelineDescriptor::new();
        descriptor.setVertexFunction(Some(&function(library, vertex)));
        descriptor.setFragmentFunction(Some(&function(library, fragment)));
        // SAFETY: attachment 0 exists.
        let color = unsafe { descriptor.colorAttachments().objectAtIndexedSubscript(0) };
        color.setPixelFormat(pixel_format(format));
        color.setBlendingEnabled(blend);
        self.device
            .newRenderPipelineStateWithDescriptor_error(&descriptor)
            .expect("newRenderPipelineStateWithDescriptor")
    }

    fn depth_pipeline(
        &self,
        library: &Self::Library,
        vertex: &str,
        fragment: &str,
        blend: bool,
    ) -> Self::RenderPipeline {
        let descriptor = MTLRenderPipelineDescriptor::new();
        descriptor.setVertexFunction(Some(&function(library, vertex)));
        descriptor.setFragmentFunction(Some(&function(library, fragment)));
        // SAFETY: attachment 0 exists.
        let color = unsafe { descriptor.colorAttachments().objectAtIndexedSubscript(0) };
        color.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
        if blend {
            color.setBlendingEnabled(true);
            color.setSourceRGBBlendFactor(MTLBlendFactor::SourceAlpha);
            color.setSourceAlphaBlendFactor(MTLBlendFactor::One);
            color.setDestinationRGBBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
            color.setDestinationAlphaBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
        }
        descriptor.setDepthAttachmentPixelFormat(MTLPixelFormat::Depth32Float);
        self.device
            .newRenderPipelineStateWithDescriptor_error(&descriptor)
            .expect("newRenderPipelineStateWithDescriptor")
    }

    fn depth_stencil(&self, writes: bool) -> Self::DepthStencil {
        let descriptor = MTLDepthStencilDescriptor::new();
        descriptor.setDepthCompareFunction(if writes {
            MTLCompareFunction::Less
        } else {
            MTLCompareFunction::LessEqual
        });
        descriptor.setDepthWriteEnabled(writes);
        self.device
            .newDepthStencilStateWithDescriptor(&descriptor)
            .expect("newDepthStencilStateWithDescriptor")
    }

    fn compute_pipeline(&self, library: &Self::Library, kernel: &str) -> Self::ComputePipeline {
        self.device
            .newComputePipelineStateWithFunction_error(&function(library, kernel))
            .expect("newComputePipelineStateWithFunction")
    }

    fn indirect_compute_pipeline(
        &self,
        library: &Self::Library,
        kernel: &str,
    ) -> Self::ComputePipeline {
        let descriptor = MTLComputePipelineDescriptor::new();
        descriptor.setComputeFunction(Some(&function(library, kernel)));
        descriptor.setSupportIndirectCommandBuffers(true);
        self.device
            .newComputePipelineStateWithDescriptor_options_reflection_error(
                &descriptor,
                MTLPipelineOption::None,
                None,
            )
            .expect("newComputePipelineStateWithDescriptor")
    }

    fn indirect_commands(&self, count: usize) -> Self::IndirectCommands {
        let descriptor = MTLIndirectCommandBufferDescriptor::new();
        descriptor.setCommandTypes(MTLIndirectCommandType::ConcurrentDispatch);
        descriptor.setInheritPipelineState(false);
        descriptor.setInheritBuffers(false);
        descriptor.setMaxKernelBufferBindCount(1);
        // SAFETY: `count` is within Metal's most commands (16,384), which the workload keeps.
        unsafe {
            self.device
                .newIndirectCommandBufferWithDescriptor_maxCommandCount_options(
                    &descriptor,
                    count,
                    MTLResourceOptions::StorageModePrivate,
                )
        }
        .expect("newIndirectCommandBufferWithDescriptor")
    }

    fn commands_id(&self, commands: &Self::IndirectCommands) -> u64 {
        commands.gpuResourceID().to_raw()
    }

    fn pipeline_id(&self, state: &Self::ComputePipeline) -> u64 {
        state.gpuResourceID().to_raw()
    }

    fn buffer(&self, length: usize) -> Self::Buffer {
        self.device
            .newBufferWithLength_options(length, MTLResourceOptions::StorageModeShared)
            .expect("newBufferWithLength")
    }

    fn buffer_with(&self, bytes: &[u8]) -> Self::Buffer {
        let pointer = NonNull::new(bytes.as_ptr() as *mut c_void).expect("bytes");
        // SAFETY: `bytes` holds `bytes.len()` bytes, which Metal copies.
        unsafe {
            self.device.newBufferWithBytes_length_options(
                pointer,
                bytes.len(),
                MTLResourceOptions::StorageModeShared,
            )
        }
        .expect("newBufferWithBytes")
    }

    unsafe fn buffer_no_copy(&self, pointer: *mut u8, length: usize) -> Self::Buffer {
        let pointer = NonNull::new(pointer as *mut c_void).expect("a pointer");
        // SAFETY: the caller's contract.
        unsafe {
            self.device
                .newBufferWithBytesNoCopy_length_options_deallocator(
                    pointer,
                    length,
                    MTLResourceOptions::StorageModeShared,
                    None,
                )
        }
        .expect("newBufferWithBytesNoCopy")
    }

    fn heap(&self, size: usize) -> Self::Heap {
        let descriptor = MTLHeapDescriptor::new();
        descriptor.setType(MTLHeapType::Placement);
        descriptor.setStorageMode(MTLStorageMode::Shared);
        descriptor.setSize(size);
        self.device
            .newHeapWithDescriptor(&descriptor)
            .expect("newHeapWithDescriptor")
    }

    unsafe fn heap_no_copy(&self, _pointer: *mut u8, _size: usize) -> Option<Self::Heap> {
        None
    }

    fn heap_texture(
        &self,
        heap: &Self::Heap,
        width: usize,
        height: usize,
        offset: usize,
    ) -> Self::Texture {
        // SAFETY: plain sizes.
        let descriptor = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                MTLPixelFormat::BGRA8Unorm,
                width,
                height,
                false,
            )
        };
        descriptor.setUsage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
        descriptor.setStorageMode(MTLStorageMode::Shared);
        descriptor.setAllowGPUOptimizedContents(false);
        // SAFETY: the offset lies inside the heap, aligned for the texture (the workload's).
        unsafe { heap.newTextureWithDescriptor_offset(&descriptor, offset) }
            .expect("newTextureWithDescriptor:offset:")
    }

    fn buffer_texture(
        &self,
        buffer: &Self::Buffer,
        (width, height): (usize, usize),
        offset: usize,
        bytes_per_row: usize,
    ) -> Self::Texture {
        // SAFETY: plain sizes.
        let descriptor = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                MTLPixelFormat::BGRA8Unorm,
                width,
                height,
                false,
            )
        };
        descriptor.setUsage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
        descriptor.setStorageMode(MTLStorageMode::Shared);
        buffer
            .newTextureWithDescriptor_offset_bytesPerRow(&descriptor, offset, bytes_per_row)
            .expect("newTextureWithDescriptor:offset:bytesPerRow:")
    }

    fn texture(&self, format: Format, width: usize, height: usize, target: bool) -> Self::Texture {
        // SAFETY: plain sizes.
        let descriptor = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                pixel_format(format),
                width,
                height,
                false,
            )
        };
        if target {
            descriptor.setUsage(MTLTextureUsage::RenderTarget);
            descriptor.setStorageMode(MTLStorageMode::Private);
        } else {
            descriptor.setUsage(MTLTextureUsage::ShaderRead);
            descriptor.setStorageMode(MTLStorageMode::Shared);
        }
        self.device
            .newTextureWithDescriptor(&descriptor)
            .expect("newTextureWithDescriptor")
    }

    fn tracked_heap(&self, size: usize) -> Self::Heap {
        let descriptor = MTLHeapDescriptor::new();
        descriptor.setType(MTLHeapType::Placement);
        descriptor.setStorageMode(MTLStorageMode::Shared);
        descriptor.setHazardTrackingMode(MTLHazardTrackingMode::Tracked);
        descriptor.setSize(size);
        self.device
            .newHeapWithDescriptor(&descriptor)
            .expect("newHeapWithDescriptor")
    }

    fn heap_render_texture(
        &self,
        heap: &Self::Heap,
        format: Format,
        side: usize,
        offset: usize,
    ) -> Self::Texture {
        // SAFETY: plain sizes.
        let descriptor = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                pixel_format(format),
                side,
                side,
                false,
            )
        };
        descriptor.setUsage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
        descriptor.setStorageMode(MTLStorageMode::Shared);
        // SAFETY: the workload places each texture within the heap at its alignment.
        unsafe { heap.newTextureWithDescriptor_offset(&descriptor, offset) }
            .expect("newTextureWithDescriptor:offset:")
    }

    fn replace_region(
        &self,
        texture: &Self::Texture,
        size: (usize, usize),
        bytes: &[u8],
        row: usize,
    ) {
        let pointer = NonNull::new(bytes.as_ptr() as *mut c_void).expect("bytes");
        // SAFETY: `bytes` holds the region's rows at `row` bytes each.
        unsafe {
            texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow(region(size), 0, pointer, row)
        };
    }

    fn get_bytes(
        &self,
        texture: &Self::Texture,
        size: (usize, usize),
        bytes: &mut [u8],
        row: usize,
    ) {
        let pointer = NonNull::new(bytes.as_mut_ptr() as *mut c_void).expect("bytes");
        // SAFETY: `bytes` has room for the region's rows at `row` bytes each.
        unsafe {
            texture.getBytes_bytesPerRow_fromRegion_mipmapLevel(pointer, row, region(size), 0)
        };
    }

    fn read_buffer(&self, buffer: &Self::Buffer, bytes: &mut [u8]) {
        assert!(bytes.len() <= buffer.length(), "a read within the buffer");
        // SAFETY: a `Shared` buffer's `contents` holds its `length` bytes, which no GPU work
        // writes now (the caller's contract), and `bytes` is as long as the copy.
        unsafe {
            std::ptr::copy_nonoverlapping(
                buffer.contents().as_ptr() as *const u8,
                bytes.as_mut_ptr(),
                bytes.len(),
            )
        };
    }

    fn write_buffer(&self, buffer: &Self::Buffer, value: u8) {
        // SAFETY: a `Shared` buffer's `contents` holds its `length` bytes, which no GPU work
        // uses now (the caller's contract).
        unsafe {
            std::ptr::write_bytes(
                buffer.contents().as_ptr() as *mut u8,
                value,
                buffer.length(),
            )
        };
    }

    fn sampler(&self, linear: bool) -> Self::Sampler {
        let descriptor = MTLSamplerDescriptor::new();
        if linear {
            descriptor.setMinFilter(MTLSamplerMinMagFilter::Linear);
            descriptor.setMagFilter(MTLSamplerMinMagFilter::Linear);
        }
        self.device
            .newSamplerStateWithDescriptor(&descriptor)
            .expect("newSamplerStateWithDescriptor")
    }

    fn command_buffer(&self) -> Self::CommandBuffer {
        self.queue.commandBuffer().expect("commandBuffer")
    }

    fn render_encoder(
        &self,
        command_buffer: &Self::CommandBuffer,
        target: &Self::Texture,
        clear: bool,
    ) -> Self::RenderEncoder {
        let pass = MTLRenderPassDescriptor::renderPassDescriptor();
        // SAFETY: attachment 0 exists.
        let color = unsafe { pass.colorAttachments().objectAtIndexedSubscript(0) };
        color.setTexture(Some(target));
        color.setLoadAction(if clear {
            MTLLoadAction::Clear
        } else {
            MTLLoadAction::Load
        });
        color.setStoreAction(MTLStoreAction::Store);
        color.setClearColor(MTLClearColor {
            red: 0.0,
            green: 0.0,
            blue: 0.0,
            alpha: 1.0,
        });
        command_buffer
            .renderCommandEncoderWithDescriptor(&pass)
            .expect("renderCommandEncoderWithDescriptor")
    }

    fn depth_render_encoder(
        &self,
        command_buffer: &Self::CommandBuffer,
        target: &Self::Texture,
        depth: &Self::Texture,
    ) -> Self::RenderEncoder {
        let pass = MTLRenderPassDescriptor::renderPassDescriptor();
        // SAFETY: attachment 0 exists.
        let color = unsafe { pass.colorAttachments().objectAtIndexedSubscript(0) };
        color.setTexture(Some(target));
        color.setLoadAction(MTLLoadAction::Clear);
        color.setStoreAction(MTLStoreAction::Store);
        color.setClearColor(MTLClearColor {
            red: 0.1,
            green: 0.1,
            blue: 0.12,
            alpha: 1.0,
        });
        let attachment = pass.depthAttachment();
        attachment.setTexture(Some(depth));
        attachment.setLoadAction(MTLLoadAction::Clear);
        attachment.setStoreAction(MTLStoreAction::DontCare);
        attachment.setClearDepth(1.0);
        command_buffer
            .renderCommandEncoderWithDescriptor(&pass)
            .expect("renderCommandEncoderWithDescriptor")
    }

    fn set_render_pipeline(&self, encoder: &mut Self::RenderEncoder, state: &Self::RenderPipeline) {
        encoder.setRenderPipelineState(state);
    }

    fn set_depth_stencil(&self, encoder: &mut Self::RenderEncoder, state: &Self::DepthStencil) {
        encoder.setDepthStencilState(Some(state));
    }

    fn set_cull_back(&self, encoder: &mut Self::RenderEncoder, back: bool) {
        encoder.setCullMode(if back {
            MTLCullMode::Back
        } else {
            MTLCullMode::None
        });
    }

    fn set_counter_clockwise(&self, encoder: &mut Self::RenderEncoder) {
        encoder.setFrontFacingWinding(MTLWinding::CounterClockwise);
    }

    fn set_vertex_bytes(&self, encoder: &mut Self::RenderEncoder, bytes: &[u8], index: usize) {
        let pointer = NonNull::new(bytes.as_ptr() as *mut c_void).expect("bytes");
        // SAFETY: `bytes` holds `bytes.len()` bytes, which Metal copies; the index is in Metal's
        // range.
        unsafe { encoder.setVertexBytes_length_atIndex(pointer, bytes.len(), index) };
    }

    fn set_vertex_buffer(
        &self,
        encoder: &mut Self::RenderEncoder,
        buffer: &Self::Buffer,
        offset: usize,
        index: usize,
    ) {
        // SAFETY: the offset lies in the buffer and the index is in Metal's range.
        unsafe { encoder.setVertexBuffer_offset_atIndex(Some(buffer), offset, index) };
    }

    fn set_fragment_buffer(
        &self,
        encoder: &mut Self::RenderEncoder,
        buffer: &Self::Buffer,
        offset: usize,
        index: usize,
    ) {
        // SAFETY: as `set_vertex_buffer`.
        unsafe { encoder.setFragmentBuffer_offset_atIndex(Some(buffer), offset, index) };
    }

    fn set_fragment_texture(
        &self,
        encoder: &mut Self::RenderEncoder,
        texture: &Self::Texture,
        index: usize,
    ) {
        // SAFETY: the index is in Metal's range.
        unsafe { encoder.setFragmentTexture_atIndex(Some(texture), index) };
    }

    fn set_fragment_sampler(
        &self,
        encoder: &mut Self::RenderEncoder,
        sampler: &Self::Sampler,
        index: usize,
    ) {
        // SAFETY: the index is in Metal's range.
        unsafe { encoder.setFragmentSamplerState_atIndex(Some(sampler), index) };
    }

    fn draw_strip(&self, encoder: &mut Self::RenderEncoder, vertices: usize) {
        // SAFETY: the bound state reads only its bound buffers.
        unsafe {
            encoder.drawPrimitives_vertexStart_vertexCount(
                MTLPrimitiveType::TriangleStrip,
                0,
                vertices,
            )
        };
    }

    fn draw_indexed(
        &self,
        encoder: &mut Self::RenderEncoder,
        count: usize,
        indices: &Self::Buffer,
    ) {
        // SAFETY: the indices lie in the buffer, and the bound state reads only its bound buffers.
        unsafe {
            encoder.drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset(
                MTLPrimitiveType::Triangle,
                count,
                MTLIndexType::UInt16,
                indices,
                0,
            )
        };
    }

    fn end_render(&self, encoder: Self::RenderEncoder) {
        encoder.endEncoding();
    }

    fn compute_encoder(&self, command_buffer: &Self::CommandBuffer) -> Self::ComputeEncoder {
        command_buffer
            .computeCommandEncoder()
            .expect("computeCommandEncoder")
    }

    fn concurrent_compute_encoder(
        &self,
        command_buffer: &Self::CommandBuffer,
    ) -> Self::ComputeEncoder {
        command_buffer
            .computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent)
            .expect("computeCommandEncoderWithDispatchType")
    }

    fn set_compute_pipeline(
        &self,
        encoder: &mut Self::ComputeEncoder,
        state: &Self::ComputePipeline,
    ) {
        encoder.setComputePipelineState(state);
    }

    fn set_buffer(
        &self,
        encoder: &mut Self::ComputeEncoder,
        buffer: &Self::Buffer,
        offset: usize,
        index: usize,
    ) {
        // SAFETY: as `set_vertex_buffer`.
        unsafe { encoder.setBuffer_offset_atIndex(Some(buffer), offset, index) };
    }

    fn set_texture(
        &self,
        encoder: &mut Self::ComputeEncoder,
        texture: &Self::Texture,
        index: usize,
    ) {
        // SAFETY: the index is in Metal's range.
        unsafe { encoder.setTexture_atIndex(Some(texture), index) };
    }

    fn dispatch(&self, encoder: &mut Self::ComputeEncoder, groups: usize, threads: usize) {
        let size = |width| MTLSize {
            width,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(size(groups), size(threads));
    }

    fn use_commands(&self, encoder: &mut Self::ComputeEncoder, commands: &Self::IndirectCommands) {
        encoder.useResource_usage(
            ProtocolObject::from_ref(&**commands),
            MTLResourceUsage::Write,
        );
    }

    fn use_buffer(&self, encoder: &mut Self::ComputeEncoder, buffer: &Self::Buffer) {
        encoder.useResource_usage(
            ProtocolObject::from_ref(&**buffer),
            MTLResourceUsage::Read | MTLResourceUsage::Write,
        );
    }

    fn execute_commands(
        &self,
        encoder: &mut Self::ComputeEncoder,
        commands: &Self::IndirectCommands,
        count: usize,
    ) {
        let range = objc2_foundation::NSRange {
            location: 0,
            length: count,
        };
        // SAFETY: the encoder's command buffer holds `commands` until it completes, and the range
        // lies within its commands.
        unsafe { encoder.executeCommandsInBuffer_withRange(commands, range) };
    }

    fn end_compute(&self, encoder: Self::ComputeEncoder) {
        encoder.endEncoding();
    }

    fn fill(&self, command_buffer: &Self::CommandBuffer, buffer: &Self::Buffer, length: usize) {
        let blit = command_buffer
            .blitCommandEncoder()
            .expect("blitCommandEncoder");
        blit.fillBuffer_range_value(
            buffer,
            objc2_foundation::NSRange {
                location: 0,
                length,
            },
            0x5a,
        );
        blit.endEncoding();
    }

    fn upload(
        &self,
        command_buffer: &Self::CommandBuffer,
        source: &Self::Buffer,
        textures: &[&Self::Texture],
        (width, height): (usize, usize),
        bytes_per_row: usize,
    ) {
        let blit = command_buffer
            .blitCommandEncoder()
            .expect("blitCommandEncoder");
        for texture in textures {
            // SAFETY: the region lies in the buffer and in the texture (the workload's).
            unsafe {
                blit.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                    source,
                    0,
                    bytes_per_row,
                    bytes_per_row * height,
                    MTLSize { width, height, depth: 1 },
                    texture,
                    0,
                    0,
                    MTLOrigin { x: 0, y: 0, z: 0 },
                )
            };
        }
        blit.endEncoding();
    }

    fn commit(&self, command_buffer: &Self::CommandBuffer) {
        command_buffer.commit();
    }

    fn wait(&self, command_buffer: &Self::CommandBuffer) {
        command_buffer.waitUntilCompleted();
        let status = command_buffer.status();
        assert!(
            status == MTLCommandBufferStatus::Completed,
            "the command buffer ended {status:?}: {:?}",
            command_buffer.error()
        );
    }

    fn on_completed(
        &self,
        command_buffer: &Self::CommandBuffer,
        handler: Box<dyn FnOnce() + Send>,
    ) {
        let handler = Mutex::new(Some(handler));
        let block = RcBlock::new(move |_: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
            if let Some(handler) = handler.lock().unwrap().take() {
                handler();
            }
        });
        // SAFETY: the block is valid for the call, and Metal copies it before returning; it is
        // added before the commit.
        unsafe { command_buffer.addCompletedHandler(RcBlock::as_ptr(&block)) };
    }

    fn gpu_seconds(&self, command_buffer: &Self::CommandBuffer) -> Option<f64> {
        Some(command_buffer.GPUEndTime() - command_buffer.GPUStartTime())
    }

    fn times(&self, command_buffer: &Self::CommandBuffer) -> [f64; 4] {
        [
            command_buffer.kernelStartTime(),
            command_buffer.kernelEndTime(),
            command_buffer.GPUStartTime(),
            command_buffer.GPUEndTime(),
        ]
    }

    fn pool<R>(&self, f: impl FnOnce() -> R) -> R {
        autoreleasepool(|_| f())
    }
}
