//! Metal's side of the sequences: [`SeqApi`] on this Mac's system `MTLDevice`, through
//! `objc2-metal`, the fixture's library made from its typed AIR text (`metal-as`) as the
//! oracle's public fixtures are (`metal::shader_air`). A call that answers nil is
//! [`Refusal::Failed`]; a call Metal's API validation stops ends the recorder's child process.

use super::super::api::{Answer, Completion, Refused};
use super::super::record::Refusal;
use super::api::SeqApi;
use super::program::{Load, ViewForm, ViewSpec};
use super::run::{pixel_format, swizzle};
use super::FIXTURE;
use crate::gpu::compile;
use crate::gpu::wait::metal::{settle_watched, watch, Watch};
use crate::metal::{load_library, shader_air, system_device};
use metal2vulkan::metal::{PixelFormat, TextureType};
use metal2vulkan_runtime::{ResourceOptions, TextureDescriptor};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSRange, NSString};
use objc2_metal::{
    MTLBarrierScope, MTLBlitCommandEncoder, MTLBlitOption, MTLBuffer, MTLClearColor,
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice, MTLDispatchType, MTLFence, MTLHeap, MTLHeapDescriptor,
    MTLHeapType, MTLLibrary, MTLLoadAction, MTLOrigin, MTLPixelFormat, MTLPrimitiveType,
    MTLRenderCommandEncoder, MTLRenderPassDescriptor, MTLRenderPipelineDescriptor,
    MTLRenderPipelineState, MTLRenderStages, MTLResourceOptions, MTLSharedEvent, MTLSize,
    MTLStoreAction, MTLTexture, MTLTextureDescriptor, MTLTextureSwizzle, MTLTextureSwizzleChannels,
    MTLTextureType, MTLTextureUsage,
};
use std::cell::RefCell;
use std::ops::Range;
use std::ptr::NonNull;

type Object<P> = Retained<ProtocolObject<P>>;

/// This Mac's system device and the fixture's library.
pub(crate) struct MetalSeq {
    device: Object<dyn MTLDevice>,
    library: Object<dyn MTLLibrary>,
}

impl MetalSeq {
    pub(crate) fn open() -> Result<MetalSeq, String> {
        let device = system_device()?;
        let label = format!("public/engine/{FIXTURE}.ll");
        let row = crate::source::public_sources()?
            .into_iter()
            .find(|row| row.label == label)
            .ok_or_else(|| format!("no public source {label}"))?;
        let air = shader_air(&row)?;
        let library = load_library(&device, &air.bitcode, None)?;
        Ok(MetalSeq { device, library })
    }

    fn function(&self, entry: &str) -> Answer<Object<dyn objc2_metal::MTLFunction>> {
        self.library
            .newFunctionWithName(&NSString::from_str(entry))
            .ok_or_else(|| nil("newFunctionWithName:"))
    }
}

/// A command buffer and, once committed, the handler its completion is waited on through.
pub(crate) struct SeqCommandBuffer {
    raw: Object<dyn MTLCommandBuffer>,
    watch: RefCell<Option<Watch>>,
}

/// What a call that answered nil records.
fn nil(call: &str) -> Refused {
    Refused {
        refusal: Refusal::Failed,
        detail: format!("{call} answered nil"),
    }
}

fn failed(detail: String) -> Refused {
    Refused {
        refusal: Refusal::Failed,
        detail,
    }
}

fn to_usize(value: u64) -> usize {
    usize::try_from(value).expect("a 64-bit host")
}

fn resource_options(options: ResourceOptions) -> MTLResourceOptions {
    MTLResourceOptions(to_usize(options.raw()))
}

fn mtl_pixel_format(format: PixelFormat) -> MTLPixelFormat {
    MTLPixelFormat(to_usize(u64::from(format.as_raw())))
}

fn mtl_texture_type(texture_type: TextureType) -> MTLTextureType {
    MTLTextureType(to_usize(u64::from(texture_type.as_raw())))
}

fn one() -> MTLSize {
    MTLSize {
        width: 1,
        height: 1,
        depth: 1,
    }
}

/// The copy's texels: whole rows of a 2D texture.
fn rows_region(rows: &Range<u32>, width: u32) -> (MTLOrigin, MTLSize) {
    (
        MTLOrigin {
            x: 0,
            y: rows.start as usize,
            z: 0,
        },
        MTLSize {
            width: width as usize,
            height: (rows.end - rows.start) as usize,
            depth: 1,
        },
    )
}

/// Metal's descriptor of `d`.
fn mtl_texture_descriptor(d: &TextureDescriptor) -> objc2::rc::Retained<MTLTextureDescriptor> {
    let descriptor = MTLTextureDescriptor::new();
    descriptor.setTextureType(mtl_texture_type(d.texture_type));
    descriptor.setPixelFormat(mtl_pixel_format(d.pixel_format));
    // SAFETY: the extents, counts and options are the descriptor's own, which Metal's API
    // validation checks.
    unsafe {
        descriptor.setWidth(to_usize(d.width));
        descriptor.setHeight(to_usize(d.height));
        descriptor.setDepth(to_usize(d.depth));
        descriptor.setMipmapLevelCount(to_usize(d.mipmap_level_count));
        descriptor.setSampleCount(to_usize(d.sample_count));
        descriptor.setArrayLength(to_usize(d.array_length));
        descriptor.setResourceOptions(resource_options(d.resource_options));
    }
    descriptor.setUsage(MTLTextureUsage(to_usize(u64::from(d.usage.as_raw()))));
    descriptor
}

impl SeqApi for MetalSeq {
    type Heap = Object<dyn MTLHeap>;
    type Buffer = Object<dyn MTLBuffer>;
    type Texture = Object<dyn MTLTexture>;
    type Kernel = Object<dyn MTLComputePipelineState>;
    type Draw = Object<dyn MTLRenderPipelineState>;
    type Queue = Object<dyn MTLCommandQueue>;
    type CommandBuffer = SeqCommandBuffer;
    type Event = Object<dyn MTLSharedEvent>;
    type Fence = Object<dyn MTLFence>;
    type Blit = Object<dyn MTLBlitCommandEncoder>;
    type Compute = Object<dyn MTLComputeCommandEncoder>;
    type Render = Object<dyn MTLRenderCommandEncoder>;

    fn buffer(&self, length: u64, options: ResourceOptions) -> Answer<Self::Buffer> {
        self.device
            .newBufferWithLength_options(to_usize(length), resource_options(options))
            .ok_or_else(|| nil("newBufferWithLength:options:"))
    }

    unsafe fn no_copy_buffer(
        &self,
        bytes: NonNull<u8>,
        length: u64,
        options: ResourceOptions,
    ) -> Answer<Self::Buffer> {
        // SAFETY: the caller's contract keeps the bytes mapped while the buffer lives; no
        // deallocator.
        unsafe {
            self.device
                .newBufferWithBytesNoCopy_length_options_deallocator(
                    bytes.cast(),
                    to_usize(length),
                    resource_options(options),
                    None,
                )
        }
        .ok_or_else(|| nil("newBufferWithBytesNoCopy:length:options:deallocator:"))
    }

    fn heap(&self, size: u64, options: ResourceOptions) -> Answer<Self::Heap> {
        let descriptor = MTLHeapDescriptor::new();
        descriptor.setType(MTLHeapType::Placement);
        descriptor.setSize(to_usize(size));
        descriptor.setResourceOptions(resource_options(options));
        self.device
            .newHeapWithDescriptor(&descriptor)
            .ok_or_else(|| nil("newHeapWithDescriptor:"))
    }

    fn heap_buffer(
        &self,
        heap: &Self::Heap,
        length: u64,
        options: ResourceOptions,
        offset: u64,
    ) -> Answer<Self::Buffer> {
        // SAFETY: Metal's API validation checks the length, options and offset against the heap.
        unsafe {
            heap.newBufferWithLength_options_offset(
                to_usize(length),
                resource_options(options),
                to_usize(offset),
            )
        }
        .ok_or_else(|| nil("newBufferWithLength:options:offset:"))
    }

    fn heap_texture(
        &self,
        heap: &Self::Heap,
        descriptor: &TextureDescriptor,
        offset: u64,
    ) -> Answer<Self::Texture> {
        let descriptor = mtl_texture_descriptor(descriptor);
        // SAFETY: Metal's API validation checks the descriptor and offset against the heap.
        unsafe { heap.newTextureWithDescriptor_offset(&descriptor, to_usize(offset)) }
            .ok_or_else(|| nil("newTextureWithDescriptor:offset:"))
    }

    fn view(&self, texture: &Self::Texture, view: &ViewSpec) -> Answer<Self::Texture> {
        let format = mtl_pixel_format(pixel_format(view.format));
        let two_d = mtl_texture_type(TextureType::Type2D);
        let levels = NSRange::new(0, 1);
        let slices = NSRange::new(view.slice as usize, 1);
        let made = match view.form {
            ViewForm::Whole => texture.newTextureViewWithPixelFormat(format),
            // SAFETY: the format, type, levels and slices are the sequence's own, which Metal's
            // API validation checks against the texture.
            ViewForm::Slices => unsafe {
                texture.newTextureViewWithPixelFormat_textureType_levels_slices(
                    format, two_d, levels, slices,
                )
            },
            ViewForm::Swizzled(channels) => {
                let channels = swizzle(channels);
                let mtl = |channel: metal2vulkan::metal::TextureSwizzle| {
                    MTLTextureSwizzle(u8::try_from(channel.as_raw()).expect("a swizzle"))
                };
                let channels = MTLTextureSwizzleChannels {
                    red: mtl(channels.red),
                    green: mtl(channels.green),
                    blue: mtl(channels.blue),
                    alpha: mtl(channels.alpha),
                };
                // SAFETY: as above.
                unsafe {
                    texture.newTextureViewWithPixelFormat_textureType_levels_slices_swizzle(
                        format, two_d, levels, slices, channels,
                    )
                }
            }
        };
        made.ok_or_else(|| nil("newTextureViewWithPixelFormat:"))
    }

    fn write(&self, buffer: &Self::Buffer, bytes: &[u8]) {
        assert!(bytes.len() <= buffer.length());
        // SAFETY: a Shared buffer's contents cover its length, which holds `bytes`; no command
        // buffer uses the buffer while the runner writes it.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                buffer.contents().as_ptr().cast::<u8>(),
                bytes.len(),
            );
        }
    }

    fn read(&self, buffer: &Self::Buffer) -> Vec<u8> {
        // SAFETY: a Shared buffer's contents cover its length; the command buffers that wrote it
        // completed.
        unsafe {
            std::slice::from_raw_parts(buffer.contents().as_ptr().cast::<u8>(), buffer.length())
        }
        .to_vec()
    }

    fn texture(&self, d: &TextureDescriptor) -> Answer<Self::Texture> {
        self.device
            .newTextureWithDescriptor(&mtl_texture_descriptor(d))
            .ok_or_else(|| nil("newTextureWithDescriptor:"))
    }

    fn kernel(&self, entry: &str) -> Answer<Self::Kernel> {
        let function = self.function(entry)?;
        compile::bounded("sequence compute pipeline creation", || {
            self.device
                .newComputePipelineStateWithFunction_error(&function)
        })
        .map_err(|error| failed(error.to_string()))
    }

    fn draw_state(&self, format: PixelFormat) -> Answer<Self::Draw> {
        let vertex = self.function("seq_rect")?;
        let fragment = self.function("seq_color")?;
        let descriptor = MTLRenderPipelineDescriptor::new();
        descriptor.setVertexFunction(Some(&*vertex));
        descriptor.setFragmentFunction(Some(&*fragment));
        // SAFETY: colour attachment 0 of a render pipeline descriptor exists.
        let attachment = unsafe { descriptor.colorAttachments().objectAtIndexedSubscript(0) };
        attachment.setPixelFormat(mtl_pixel_format(format));
        compile::bounded("sequence render pipeline creation", || {
            self.device
                .newRenderPipelineStateWithDescriptor_error(&descriptor)
        })
        .map_err(|error| failed(error.to_string()))
    }

    fn queue(&self) -> Answer<Self::Queue> {
        self.device
            .newCommandQueue()
            .ok_or_else(|| nil("newCommandQueue"))
    }

    fn command_buffer(&self, queue: &Self::Queue) -> Answer<SeqCommandBuffer> {
        let raw = queue.commandBuffer().ok_or_else(|| nil("commandBuffer"))?;
        Ok(SeqCommandBuffer {
            raw,
            watch: RefCell::new(None),
        })
    }

    fn event(&self) -> Answer<Self::Event> {
        self.device
            .newSharedEvent()
            .ok_or_else(|| nil("newSharedEvent"))
    }

    fn fence(&self) -> Answer<Self::Fence> {
        self.device.newFence().ok_or_else(|| nil("newFence"))
    }

    fn signal(&self, cb: &SeqCommandBuffer, event: &Self::Event, value: u64) -> Answer<()> {
        cb.raw
            .encodeSignalEvent_value(ProtocolObject::from_ref(&**event), value);
        Ok(())
    }

    fn wait(&self, cb: &SeqCommandBuffer, event: &Self::Event, value: u64) -> Answer<()> {
        cb.raw
            .encodeWaitForEvent_value(ProtocolObject::from_ref(&**event), value);
        Ok(())
    }

    fn blit(&self, cb: &SeqCommandBuffer) -> Answer<Self::Blit> {
        cb.raw
            .blitCommandEncoder()
            .ok_or_else(|| nil("blitCommandEncoder"))
    }

    fn fill(
        &self,
        blit: &mut Self::Blit,
        buffer: &Self::Buffer,
        range: Range<u64>,
        value: u8,
    ) -> Answer<()> {
        let length = range.end.saturating_sub(range.start);
        blit.fillBuffer_range_value(
            buffer,
            NSRange::new(to_usize(range.start), to_usize(length)),
            value,
        );
        Ok(())
    }

    fn copy(
        &self,
        blit: &mut Self::Blit,
        (from, from_offset): (&Self::Buffer, u64),
        (to, to_offset): (&Self::Buffer, u64),
        size: u64,
    ) -> Answer<()> {
        // SAFETY: Metal's API validation checks the offsets and size against the buffers.
        unsafe {
            blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                from,
                to_usize(from_offset),
                to,
                to_usize(to_offset),
                to_usize(size),
            );
        }
        Ok(())
    }

    fn buffer_to_texture(
        &self,
        blit: &mut Self::Blit,
        (from, offset): (&Self::Buffer, u64),
        (texture, slice): (&Self::Texture, u32),
        rows: Range<u32>,
        (width, row_bytes): (u32, u64),
    ) -> Answer<()> {
        let (origin, size) = rows_region(&rows, width);
        // SAFETY: Metal's API validation checks the layout and region against the buffer and
        // the texture.
        unsafe {
            blit.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin_options(
                from,
                to_usize(offset),
                to_usize(row_bytes),
                to_usize(row_bytes) * size.height,
                size,
                texture,
                slice as usize,
                0,
                origin,
                MTLBlitOption::None,
            );
        }
        Ok(())
    }

    fn texture_to_buffer(
        &self,
        blit: &mut Self::Blit,
        (texture, slice): (&Self::Texture, u32),
        rows: Range<u32>,
        (width, row_bytes): (u32, u64),
        (to, offset): (&Self::Buffer, u64),
    ) -> Answer<()> {
        let (origin, size) = rows_region(&rows, width);
        // SAFETY: as `buffer_to_texture`.
        unsafe {
            blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage_options(
                texture,
                slice as usize,
                0,
                origin,
                size,
                to,
                to_usize(offset),
                to_usize(row_bytes),
                to_usize(row_bytes) * size.height,
                MTLBlitOption::None,
            );
        }
        Ok(())
    }

    fn blit_update_fence(&self, blit: &mut Self::Blit, fence: &Self::Fence) -> Answer<()> {
        blit.updateFence(fence);
        Ok(())
    }

    fn blit_wait_fence(&self, blit: &mut Self::Blit, fence: &Self::Fence) -> Answer<()> {
        blit.waitForFence(fence);
        Ok(())
    }

    fn end_blit(&self, blit: Self::Blit) -> Answer<()> {
        blit.endEncoding();
        Ok(())
    }

    fn compute(&self, cb: &SeqCommandBuffer, concurrent: bool) -> Answer<Self::Compute> {
        let dispatch_type = if concurrent {
            MTLDispatchType::Concurrent
        } else {
            MTLDispatchType::Serial
        };
        cb.raw
            .computeCommandEncoderWithDispatchType(dispatch_type)
            .ok_or_else(|| nil("computeCommandEncoderWithDispatchType:"))
    }

    fn dispatch(
        &self,
        compute: &mut Self::Compute,
        kernel: &Self::Kernel,
        (a, a_offset): (&Self::Buffer, u64),
        (b, b_offset): (&Self::Buffer, u64),
        indirect: Option<(&Self::Buffer, u64)>,
    ) -> Answer<()> {
        compute.setComputePipelineState(kernel);
        // SAFETY: live buffers at offsets inside them; Metal's API validation checks the
        // offsets, the indices and the indirect arguments' place.
        unsafe {
            compute.setBuffer_offset_atIndex(Some(a), to_usize(a_offset), 0);
            compute.setBuffer_offset_atIndex(Some(b), to_usize(b_offset), 1);
            match indirect {
                Some((arguments, offset)) => compute
                    .dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup(
                        arguments,
                        to_usize(offset),
                        one(),
                    ),
                None => compute.dispatchThreads_threadsPerThreadgroup(one(), one()),
            }
        }
        Ok(())
    }

    fn texel_dispatch(
        &self,
        compute: &mut Self::Compute,
        kernel: &Self::Kernel,
        texture: &Self::Texture,
        (words, offset): (&Self::Buffer, u64),
        (width, rows): (u32, u32),
    ) -> Answer<()> {
        compute.setComputePipelineState(kernel);
        // SAFETY: a live texture and buffer at an offset inside it; Metal's API validation
        // checks the indices, the offset and the grid.
        unsafe {
            compute.setTexture_atIndex(Some(texture), 0);
            compute.setBuffer_offset_atIndex(Some(words), to_usize(offset), 0);
            compute.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: width as usize,
                    height: rows as usize,
                    depth: 1,
                },
                MTLSize {
                    width: width as usize,
                    height: 1,
                    depth: 1,
                },
            );
        }
        Ok(())
    }

    fn barrier(&self, compute: &mut Self::Compute, textures: bool) -> Answer<()> {
        let scope = if textures {
            MTLBarrierScope::Buffers | MTLBarrierScope::Textures
        } else {
            MTLBarrierScope::Buffers
        };
        compute.memoryBarrierWithScope(scope);
        Ok(())
    }

    fn compute_update_fence(&self, compute: &mut Self::Compute, fence: &Self::Fence) -> Answer<()> {
        compute.updateFence(fence);
        Ok(())
    }

    fn compute_wait_fence(&self, compute: &mut Self::Compute, fence: &Self::Fence) -> Answer<()> {
        compute.waitForFence(fence);
        Ok(())
    }

    fn end_compute(&self, compute: Self::Compute) -> Answer<()> {
        compute.endEncoding();
        Ok(())
    }

    fn render(
        &self,
        cb: &SeqCommandBuffer,
        (target, slice): (&Self::Texture, u32),
        load: Load,
    ) -> Answer<Self::Render> {
        let pass = MTLRenderPassDescriptor::renderPassDescriptor();
        // SAFETY: colour attachment 0 of a render pass descriptor exists.
        let attachment = unsafe { pass.colorAttachments().objectAtIndexedSubscript(0) };
        attachment.setTexture(Some(&**target));
        attachment.setSlice(slice as usize);
        match load {
            Load::Clear(color) => {
                let unorm = |byte: u8| f64::from(byte) / 255.0;
                attachment.setLoadAction(MTLLoadAction::Clear);
                attachment.setClearColor(MTLClearColor {
                    red: unorm(color[0]),
                    green: unorm(color[1]),
                    blue: unorm(color[2]),
                    alpha: unorm(color[3]),
                });
            }
            Load::Load => attachment.setLoadAction(MTLLoadAction::Load),
        }
        attachment.setStoreAction(MTLStoreAction::Store);
        cb.raw
            .renderCommandEncoderWithDescriptor(&pass)
            .ok_or_else(|| nil("renderCommandEncoderWithDescriptor:"))
    }

    fn draw(
        &self,
        render: &mut Self::Render,
        state: &Self::Draw,
        (rects, rects_offset): (&Self::Buffer, u64),
        (colors, colors_offset): (&Self::Buffer, u64),
        (vertices, instances): (u32, u32),
        indirect: Option<(&Self::Buffer, u64)>,
    ) -> Answer<()> {
        render.setRenderPipelineState(state);
        // SAFETY: live buffers at offsets inside them; Metal's API validation checks the
        // offsets, the indices, the counts and the indirect arguments' place.
        unsafe {
            render.setVertexBuffer_offset_atIndex(Some(rects), to_usize(rects_offset), 0);
            render.setFragmentBuffer_offset_atIndex(Some(colors), to_usize(colors_offset), 0);
            match indirect {
                Some((arguments, offset)) => render
                    .drawPrimitives_indirectBuffer_indirectBufferOffset(
                        MTLPrimitiveType::Triangle,
                        arguments,
                        to_usize(offset),
                    ),
                None => render.drawPrimitives_vertexStart_vertexCount_instanceCount(
                    MTLPrimitiveType::Triangle,
                    0,
                    vertices as usize,
                    instances as usize,
                ),
            }
        }
        Ok(())
    }

    fn render_update_fence(&self, render: &mut Self::Render, fence: &Self::Fence) -> Answer<()> {
        render.updateFence_afterStages(fence, MTLRenderStages::Fragment);
        Ok(())
    }

    fn render_wait_fence(&self, render: &mut Self::Render, fence: &Self::Fence) -> Answer<()> {
        render.waitForFence_beforeStages(fence, MTLRenderStages::Vertex);
        Ok(())
    }

    fn end_render(&self, render: Self::Render) -> Answer<()> {
        render.endEncoding();
        Ok(())
    }

    fn commit(&self, cb: &SeqCommandBuffer) -> Answer<()> {
        *cb.watch.borrow_mut() = Some(watch(&cb.raw));
        cb.raw.commit();
        Ok(())
    }

    fn completion(&self, cb: SeqCommandBuffer) -> Result<Completion, String> {
        let watched = cb
            .watch
            .into_inner()
            .ok_or("a command buffer waited for before its commit")?;
        settle_watched(&cb.raw, watched, "a sequence's command buffer")?;
        Ok(Completion {
            status: "completed".to_owned(),
            times: None,
        })
    }
}
