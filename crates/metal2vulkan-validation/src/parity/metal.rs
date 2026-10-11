//! Metal's side of the parity harness: [`Api`] on this Mac's system `MTLDevice`, through
//! `objc2-metal`. Each value of the runtime's vocabulary maps to Metal's own raw value
//! (`PixelFormat::as_raw`, `ResourceOptions::raw`, ...), which they share.
//!
//! Metal answers nothing for a call its API validation stops: the recorder's child process ends
//! there (`parity::lane`). A call that answers nil is [`Refusal::Failed`].

use super::api::{Answer, Api, Completion, Fixture, Layout, Refused, Region, Times};
use super::record::Refusal;
use crate::gpu::compile;
use crate::gpu::wait::metal::{settle_watched, watch, Watch};
use crate::metal::{load_library, shader_air, system_device};
use metal2vulkan_runtime::{BlitOption, Origin, ResourceOptions, Size, TextureDescriptor};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSRange, NSString};
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBlitOption, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder,
    MTLCommandQueue, MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLFence,
    MTLLibrary, MTLOrigin, MTLPixelFormat, MTLResourceOptions, MTLSharedEvent, MTLSize, MTLTexture,
    MTLTextureDescriptor, MTLTextureSwizzle, MTLTextureSwizzleChannels, MTLTextureType,
    MTLTextureUsage,
};
use std::cell::RefCell;
use std::ops::Range;

type Object<P> = Retained<ProtocolObject<P>>;

/// This Mac's system device.
pub(crate) struct MetalApi {
    device: Object<dyn MTLDevice>,
}

impl MetalApi {
    pub(crate) fn open() -> Result<MetalApi, String> {
        Ok(MetalApi {
            device: system_device()?,
        })
    }
}

/// A command buffer and, once committed, the handler its completion is waited on through.
pub(crate) struct MetalCommandBuffer {
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

fn size(size: Size) -> MTLSize {
    MTLSize {
        width: to_usize(size.width),
        height: to_usize(size.height),
        depth: to_usize(size.depth),
    }
}

fn origin(origin: Origin) -> MTLOrigin {
    MTLOrigin {
        x: to_usize(origin.x),
        y: to_usize(origin.y),
        z: to_usize(origin.z),
    }
}

fn to_usize(value: u64) -> usize {
    usize::try_from(value).expect("a 64-bit host")
}

fn blit_option(options: BlitOption) -> MTLBlitOption {
    match options {
        BlitOption::None => MTLBlitOption::None,
        BlitOption::DepthFromDepthStencil => MTLBlitOption::DepthFromDepthStencil,
        BlitOption::StencilFromDepthStencil => MTLBlitOption::StencilFromDepthStencil,
        BlitOption::RowLinearPVRTC => MTLBlitOption::RowLinearPVRTC,
        other => panic!("no MTLBlitOption for {other:?}"),
    }
}

fn resource_options(options: ResourceOptions) -> MTLResourceOptions {
    MTLResourceOptions(to_usize(options.raw()))
}

impl Api for MetalApi {
    type Buffer = Object<dyn MTLBuffer>;
    type Texture = Object<dyn MTLTexture>;
    type Pipeline = Object<dyn MTLComputePipelineState>;
    type Queue = Object<dyn MTLCommandQueue>;
    type CommandBuffer = MetalCommandBuffer;
    type Event = Object<dyn MTLSharedEvent>;
    type Fence = Object<dyn MTLFence>;
    type Blit = Object<dyn MTLBlitCommandEncoder>;
    type Compute = Object<dyn MTLComputeCommandEncoder>;

    fn buffer(&self, length: u64, options: ResourceOptions) -> Answer<Self::Buffer> {
        self.device
            .newBufferWithLength_options(to_usize(length), resource_options(options))
            .ok_or_else(|| nil("newBufferWithLength:options:"))
    }

    fn write(&self, buffer: &Self::Buffer, bytes: &[u8]) {
        assert!(bytes.len() <= buffer.length());
        // SAFETY: a Shared buffer's contents cover its length, which holds `bytes`; no command
        // buffer uses the buffer while the scenario writes it.
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
        let descriptor = MTLTextureDescriptor::new();
        descriptor.setTextureType(MTLTextureType(to_usize(u64::from(d.texture_type.as_raw()))));
        descriptor.setPixelFormat(MTLPixelFormat(to_usize(u64::from(d.pixel_format.as_raw()))));
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
        descriptor.setAllowGPUOptimizedContents(d.allow_gpu_optimized_contents);
        let swizzle = |channel: metal2vulkan::metal::TextureSwizzle| {
            MTLTextureSwizzle(u8::try_from(channel.as_raw()).expect("a swizzle"))
        };
        descriptor.setSwizzle(MTLTextureSwizzleChannels {
            red: swizzle(d.swizzle.red),
            green: swizzle(d.swizzle.green),
            blue: swizzle(d.swizzle.blue),
            alpha: swizzle(d.swizzle.alpha),
        });
        self.device
            .newTextureWithDescriptor(&descriptor)
            .ok_or_else(|| nil("newTextureWithDescriptor:"))
    }

    fn pipeline(&self, fixture: &Fixture) -> Answer<Self::Pipeline> {
        let failed = |detail: String| Refused {
            refusal: Refusal::Failed,
            detail,
        };
        let air = shader_air(&fixture.row).map_err(failed)?;
        let library: Object<dyn MTLLibrary> =
            load_library(&self.device, &air.bitcode, None).map_err(failed)?;
        let function = library
            .newFunctionWithName(&NSString::from_str(&fixture.row.entry))
            .ok_or_else(|| nil("newFunctionWithName:"))?;
        compile::bounded("parity compute pipeline creation", || {
            self.device
                .newComputePipelineStateWithFunction_error(&function)
        })
        .map_err(|error| failed(error.to_string()))
    }

    fn queue(&self) -> Answer<Self::Queue> {
        self.device
            .newCommandQueue()
            .ok_or_else(|| nil("newCommandQueue"))
    }

    fn command_buffer(&self, queue: &Self::Queue) -> Answer<MetalCommandBuffer> {
        let raw = queue.commandBuffer().ok_or_else(|| nil("commandBuffer"))?;
        Ok(MetalCommandBuffer {
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

    fn signal(&self, cb: &MetalCommandBuffer, event: &Self::Event, value: u64) -> Answer<()> {
        cb.raw
            .encodeSignalEvent_value(ProtocolObject::from_ref(&**event), value);
        Ok(())
    }

    fn wait(&self, cb: &MetalCommandBuffer, event: &Self::Event, value: u64) -> Answer<()> {
        cb.raw
            .encodeWaitForEvent_value(ProtocolObject::from_ref(&**event), value);
        Ok(())
    }

    fn blit(&self, cb: &MetalCommandBuffer) -> Answer<Self::Blit> {
        cb.raw
            .blitCommandEncoder()
            .ok_or_else(|| nil("blitCommandEncoder"))
    }

    fn copy_buffer(
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

    fn copy_to_texture(
        &self,
        blit: &mut Self::Blit,
        (from, layout): (&Self::Buffer, Layout),
        (to, region): (&Self::Texture, Region),
        options: BlitOption,
    ) -> Answer<()> {
        // SAFETY: Metal's API validation checks the layout and region against the buffer and
        // the texture.
        unsafe {
            blit.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin_options(
                from,
                to_usize(layout.offset),
                to_usize(layout.bytes_per_row),
                to_usize(layout.bytes_per_image),
                size(region.size),
                to,
                to_usize(region.slice),
                to_usize(region.level),
                origin(region.origin),
                blit_option(options),
            );
        }
        Ok(())
    }

    fn copy_from_texture(
        &self,
        blit: &mut Self::Blit,
        (from, region): (&Self::Texture, Region),
        (to, layout): (&Self::Buffer, Layout),
        options: BlitOption,
    ) -> Answer<()> {
        // SAFETY: as `copy_to_texture`.
        unsafe {
            blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage_options(
                from,
                to_usize(region.slice),
                to_usize(region.level),
                origin(region.origin),
                size(region.size),
                to,
                to_usize(layout.offset),
                to_usize(layout.bytes_per_row),
                to_usize(layout.bytes_per_image),
                blit_option(options),
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

    fn compute(&self, cb: &MetalCommandBuffer) -> Answer<Self::Compute> {
        cb.raw
            .computeCommandEncoder()
            .ok_or_else(|| nil("computeCommandEncoder"))
    }

    fn dispatch(
        &self,
        compute: &mut Self::Compute,
        pipeline: &Self::Pipeline,
        buffers: &[(&Self::Buffer, u64)],
        threads: Size,
    ) -> Answer<()> {
        compute.setComputePipelineState(pipeline);
        for &(buffer, index) in buffers {
            // SAFETY: a live buffer at offset 0; Metal's API validation checks the index.
            unsafe { compute.setBuffer_offset_atIndex(Some(buffer), 0, to_usize(index)) };
        }
        compute.dispatchThreads_threadsPerThreadgroup(
            size(threads),
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
        );
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

    fn commit(&self, cb: &MetalCommandBuffer) -> Answer<()> {
        *cb.watch.borrow_mut() = Some(watch(&cb.raw));
        cb.raw.commit();
        Ok(())
    }

    fn completion(&self, cb: MetalCommandBuffer) -> Result<Completion, String> {
        let watched = cb
            .watch
            .into_inner()
            .ok_or("a command buffer waited for before its commit")?;
        settle_watched(&cb.raw, watched, "a parity scenario's command buffer")?;
        Ok(Completion {
            status: "completed".to_owned(),
            times: Some(times(&cb.raw)),
        })
    }

    fn times(&self, cb: &MetalCommandBuffer) -> Answer<Times> {
        Ok(times(&cb.raw))
    }

    fn host_time(&self) -> f64 {
        // `CLOCK_UPTIME_RAW` is `mach_absolute_time` in nanoseconds (clock_gettime(3)).
        let mut now = libc::timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        // SAFETY: `now` is a timespec for the call to fill.
        let read = unsafe { libc::clock_gettime(libc::CLOCK_UPTIME_RAW, &mut now) };
        assert_eq!(read, 0, "CLOCK_UPTIME_RAW");
        now.tv_sec as f64 + now.tv_nsec as f64 / 1e9
    }
}

/// `cb`'s four times now.
fn times(cb: &Object<dyn MTLCommandBuffer>) -> Times {
    Times {
        gpu_start: cb.GPUStartTime(),
        gpu_end: cb.GPUEndTime(),
        kernel_start: cb.kernelStartTime(),
        kernel_end: cb.kernelEndTime(),
    }
}
