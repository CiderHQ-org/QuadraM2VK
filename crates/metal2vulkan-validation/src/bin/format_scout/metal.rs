//! The format scout's Metal side (macOS): the same cells as the runtime side, driven through Metal
//! itself (objc2-metal), with the same texels, functions (the fixture's MSL, compiled by the
//! device) and readback, so a made cell's detail is comparable byte for byte with the runtime's.
//! Metal's refusals are a nil object or an `NSError`. This is plain Metal, the reference: the
//! parent removes Metal's debug and validation layers' variables from this side's children, and a
//! cell that ends the child (Metal's own assertion) is recorded as died with Metal's text from
//! stderr. A surface cell's texture is made over an IOSurface laid out as the runtime side's
//! buffer stands for one (`surface.rs`), filled with the same pattern.
//!
//! GPU safety is the crate's (`gpu.rs`): the device opens behind the hang tripwire check
//! (`metal::system_device`), every command buffer is committed and waited for under the bounded
//! wait's hang protocol (`gpu::wait::metal::commit_and_wait`: a cell that outlasts it writes the
//! tripwire and ends the child), and every compilation runs under the compile deadline
//! (`gpu::compile::bounded`).

use super::surface::{self, Plane, Surface, TwoPlanes};
use super::{
    begin_cell, class, compressed, digest, has_depth, has_stencil, layout, one_line, pattern,
    Blend, Class, Fail, Form, Op, CHROMA_OFFSET, SIDE, X420,
};
use core::ffi::c_void;
use core::ptr::NonNull;
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan_devtools::engine_dir;
use metal2vulkan_runtime::StorageMode;
use metal2vulkan_validation::gpu::compile;
use metal2vulkan_validation::gpu::wait::metal::commit_and_wait;
use metal2vulkan_validation::metal::system_device;
use objc2::msg_send;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSRange, NSString};
use objc2_metal::{
    MTLBlendFactor, MTLBlitCommandEncoder, MTLBlitOption, MTLBuffer, MTLClearColor,
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice, MTLHeap, MTLHeapDescriptor, MTLHeapType, MTLLibrary,
    MTLLoadAction, MTLOrigin, MTLPixelFormat, MTLPrimitiveType, MTLRegion, MTLRenderCommandEncoder,
    MTLRenderPassDescriptor, MTLRenderPipelineDescriptor, MTLRenderPipelineState,
    MTLResourceOptions, MTLSamplerDescriptor, MTLSamplerMinMagFilter, MTLSamplerState, MTLSize,
    MTLStorageMode, MTLStoreAction, MTLTexture, MTLTextureDescriptor, MTLTextureSwizzle,
    MTLTextureSwizzleChannels, MTLTextureType, MTLTextureUsage,
};
use std::collections::HashMap;
use std::io::Write;

type Object<T> = Retained<ProtocolObject<T>>;
type Done = Result<String, Fail>;

fn nil(what: &str) -> Fail {
    Fail::Refused(format!("nil: {what}"))
}

fn error(what: &str, e: &objc2_foundation::NSError) -> Fail {
    Fail::Refused(format!("{what}: {}", e.localizedDescription()))
}

fn size(w: u64, h: u64, d: u64) -> MTLSize {
    MTLSize {
        width: w as usize,
        height: h as usize,
        depth: d as usize,
    }
}

fn origin() -> MTLOrigin {
    MTLOrigin { x: 0, y: 0, z: 0 }
}

fn region(w: u64, h: u64) -> MTLRegion {
    MTLRegion {
        origin: origin(),
        size: size(w, h, 1),
    }
}

/// Refuses a render pass into a surface texture of a two-plane YCbCr format: on the M3 under
/// macOS 27.0.1, a pass into an IOSurface-backed `YCBCR10_420_2P_PACKED` texture ended with
/// `kIOGPUCommandBufferCallbackErrorHang`, so the scout never draws into one.
fn refuse_ycbcr_surface_pass(pf: PixelFormat) -> Result<(), Fail> {
    match metal2vulkan::metal::format::ycbcr_format(pf) {
        Some(_) => Err(Fail::Skip(
            "no render pass into a two-plane YCbCr surface (it hangs the GPU)".into(),
        )),
        None => Ok(()),
    }
}

fn storage_raw(storage: StorageMode) -> usize {
    match storage {
        StorageMode::Shared => 0,
        StorageMode::Managed => 1,
        StorageMode::Private => 2,
        _ => 3,
    }
}

fn blit_option(pf: PixelFormat) -> MTLBlitOption {
    if has_depth(pf) && has_stencil(pf) {
        MTLBlitOption::DepthFromDepthStencil
    } else if super::short(pf).starts_with("PVRTC") {
        MTLBlitOption::RowLinearPVRTC
    } else {
        MTLBlitOption::None
    }
}

struct Side {
    pf: PixelFormat,
    device: Object<dyn MTLDevice>,
    queue: Object<dyn MTLCommandQueue>,
    library: Object<dyn MTLLibrary>,
    kernels: HashMap<String, Object<dyn MTLComputePipelineState>>,
    samplers: HashMap<bool, Object<dyn MTLSamplerState>>,
}

impl Side {
    fn desc(&self, usage: TextureUsage, storage: StorageMode) -> Retained<MTLTextureDescriptor> {
        let d = MTLTextureDescriptor::new();
        d.setTextureType(MTLTextureType(TextureType::Type2D.as_raw() as usize));
        d.setPixelFormat(MTLPixelFormat(self.pf.as_raw() as usize));
        // SAFETY: plain sizes of a small two-dimensional texture.
        unsafe {
            d.setWidth(SIDE as usize);
            d.setHeight(SIDE as usize);
        }
        d.setUsage(MTLTextureUsage(usage.as_raw() as usize));
        d.setStorageMode(MTLStorageMode(storage_raw(storage)));
        d
    }

    fn texture(&self, d: &MTLTextureDescriptor) -> Result<Object<dyn MTLTexture>, Fail> {
        self.device
            .newTextureWithDescriptor(d)
            .ok_or_else(|| nil("newTextureWithDescriptor"))
    }

    fn buffer(&self, bytes: &[u8]) -> Result<Object<dyn MTLBuffer>, Fail> {
        let b = self
            .device
            .newBufferWithLength_options(bytes.len().max(16), MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| nil("newBufferWithLength"))?;
        // SAFETY: the buffer is shared and at least `bytes.len()` long; no command uses it yet.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                b.contents().as_ptr().cast::<u8>(),
                bytes.len(),
            );
        }
        Ok(b)
    }

    fn read(b: &ProtocolObject<dyn MTLBuffer>, length: u64) -> Vec<u8> {
        // SAFETY: the buffer is shared, at least `length` long, and its commands completed.
        unsafe {
            std::slice::from_raw_parts(b.contents().as_ptr().cast::<u8>(), length as usize).to_vec()
        }
    }

    fn commit(&self, cb: &ProtocolObject<dyn MTLCommandBuffer>) -> Result<(), Fail> {
        commit_and_wait(cb, "format-scout cell")
            .map_err(|e| Fail::Died(format!("command buffer: {e}")))
    }

    fn command_buffer(&self) -> Result<Object<dyn MTLCommandBuffer>, Fail> {
        self.queue
            .commandBuffer()
            .ok_or_else(|| nil("commandBuffer"))
    }

    fn upload(&self, t: &ProtocolObject<dyn MTLTexture>) -> Result<(), Fail> {
        let l = layout(self.pf).ok_or_else(|| Fail::Skip("no copy layout".into()))?;
        let src = self.buffer(&pattern(self.pf.as_raw(), l.row * l.rows))?;
        let cb = self.command_buffer()?;
        let blit = cb
            .blitCommandEncoder()
            .ok_or_else(|| nil("blitCommandEncoder"))?;
        // SAFETY: the buffer holds the rows the copy reads; the texture is 16 x 16.
        unsafe {
            blit.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin_options(
                &src,
                0,
                l.row as usize,
                (l.row * l.rows) as usize,
                size(SIDE, SIDE, 1),
                t,
                0,
                0,
                origin(),
                blit_option(self.pf),
            );
        }
        blit.endEncoding();
        self.commit(&cb)
    }

    fn read_back(
        &self,
        t: &ProtocolObject<dyn MTLTexture>,
        level: u64,
        side: u64,
    ) -> Result<Vec<u8>, Fail> {
        let l = layout(self.pf).ok_or_else(|| Fail::Skip("no copy layout".into()))?;
        let row = side.div_ceil(SIDE.div_ceil(l.across).max(1)).max(1) * l.unit;
        let rows = side.div_ceil(SIDE.div_ceil(l.down).max(1)).max(1);
        let dst = self.buffer(&vec![0xcd; (row * rows) as usize])?;
        let cb = self.command_buffer()?;
        let blit = cb
            .blitCommandEncoder()
            .ok_or_else(|| nil("blitCommandEncoder"))?;
        // SAFETY: the buffer holds the rows the copy writes.
        unsafe {
            blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage_options(
                t,
                0,
                level as usize,
                origin(),
                size(side, side, 1),
                &dst,
                0,
                row as usize,
                (row * rows) as usize,
                blit_option(self.pf),
            );
        }
        blit.endEncoding();
        self.commit(&cb)?;
        Ok(Self::read(&dst, row * rows))
    }

    fn digest_of(&mut self, t: &ProtocolObject<dyn MTLTexture>) -> Done {
        let bytes = self.read_back(t, 0, SIDE)?;
        Ok(format!("sha={}", digest(&bytes)))
    }

    fn kernel(&mut self, name: &str) -> Result<Object<dyn MTLComputePipelineState>, Fail> {
        if let Some(k) = self.kernels.get(name) {
            return Ok(k.clone());
        }
        let function = self
            .library
            .newFunctionWithName(&NSString::from_str(name))
            .ok_or_else(|| nil("newFunctionWithName"))?;
        let state = compile::bounded("format-scout kernel", || {
            self.device
                .newComputePipelineStateWithFunction_error(&function)
        })
        .map_err(|e| error("newComputePipelineStateWithFunction", &e))?;
        self.kernels.insert(name.to_owned(), state.clone());
        Ok(state)
    }

    fn sampler(&mut self, linear: bool) -> Result<Object<dyn MTLSamplerState>, Fail> {
        if let Some(s) = self.samplers.get(&linear) {
            return Ok(s.clone());
        }
        let d = MTLSamplerDescriptor::new();
        let filter = if linear {
            MTLSamplerMinMagFilter::Linear
        } else {
            MTLSamplerMinMagFilter::Nearest
        };
        d.setMinFilter(filter);
        d.setMagFilter(filter);
        let s = self
            .device
            .newSamplerStateWithDescriptor(&d)
            .ok_or_else(|| nil("newSamplerStateWithDescriptor"))?;
        self.samplers.insert(linear, s.clone());
        Ok(s)
    }

    fn kernel_read(
        &mut self,
        t: &ProtocolObject<dyn MTLTexture>,
        sampler: Option<bool>,
    ) -> Result<Vec<u8>, Fail> {
        let class = class(self.pf).ok_or_else(|| Fail::Skip("no read class".into()))?;
        let name = match sampler {
            Some(_) if class == Class::Int || class == Class::Uint => {
                return Err(Fail::Skip("an integer format is not sampled".into()));
            }
            Some(_) => format!("scout_sample_{}", class.suffix()),
            None => format!("scout_read_{}", class.suffix()),
        };
        let state = self.kernel(&name)?;
        let length = SIDE * SIDE * class.out_bytes();
        let out = self.buffer(&vec![0xab; length as usize])?;
        let sampler = sampler.map(|linear| self.sampler(linear)).transpose()?;
        let cb = self.command_buffer()?;
        let e = cb
            .computeCommandEncoder()
            .ok_or_else(|| nil("computeCommandEncoder"))?;
        e.setComputePipelineState(&state);
        // SAFETY: indices 0 of the kernel's texture, buffer and sampler.
        unsafe {
            e.setTexture_atIndex(Some(t), 0);
            e.setBuffer_offset_atIndex(Some(&*out), 0, 0);
            if let Some(s) = &sampler {
                e.setSamplerState_atIndex(Some(&**s), 0);
            }
        }
        e.dispatchThreads_threadsPerThreadgroup(size(SIDE, SIDE, 1), size(SIDE, SIDE, 1));
        e.endEncoding();
        self.commit(&cb)?;
        Ok(Self::read(&out, length))
    }

    fn kernel_store(&mut self, t: &ProtocolObject<dyn MTLTexture>, what: &str) -> Result<(), Fail> {
        let class = class(self.pf).ok_or_else(|| Fail::Skip("no write class".into()))?;
        if class == Class::Depth || has_stencil(self.pf) {
            return Err(Fail::Skip(
                "depth and stencil formats take no function writes".into(),
            ));
        }
        let state = self.kernel(&format!("scout_{what}_{}", class.suffix()))?;
        let cb = self.command_buffer()?;
        let e = cb
            .computeCommandEncoder()
            .ok_or_else(|| nil("computeCommandEncoder"))?;
        e.setComputePipelineState(&state);
        // SAFETY: index 0 of the kernel's texture.
        unsafe { e.setTexture_atIndex(Some(t), 0) };
        e.dispatchThreads_threadsPerThreadgroup(size(SIDE, SIDE, 1), size(SIDE, SIDE, 1));
        e.endEncoding();
        self.commit(&cb)
    }

    fn pipeline(
        &mut self,
        blend: Blend,
        samples: u32,
        depth_or_stencil: bool,
    ) -> Result<Object<dyn MTLRenderPipelineState>, Fail> {
        let pf = self.pf;
        let function = |name: &str| {
            self.library
                .newFunctionWithName(&NSString::from_str(name))
                .ok_or_else(|| nil("newFunctionWithName"))
        };
        let d = MTLRenderPipelineDescriptor::new();
        let vertex = function("scout_vertex")?;
        d.setVertexFunction(Some(&*vertex));
        d.setRasterSampleCount(samples as usize);
        let raw = MTLPixelFormat(pf.as_raw() as usize);
        if depth_or_stencil || has_depth(pf) || has_stencil(pf) {
            if has_depth(pf) || !has_stencil(pf) {
                d.setDepthAttachmentPixelFormat(raw);
            }
            if has_stencil(pf) {
                d.setStencilAttachmentPixelFormat(raw);
            }
        } else {
            let fragment = match class(pf) {
                Some(Class::Int) => "scout_fragment_i",
                Some(Class::Uint) => "scout_fragment_u",
                _ => "scout_fragment_f",
            };
            let fragment = function(fragment)?;
            d.setFragmentFunction(Some(&*fragment));
            // SAFETY: attachment 0 of eight.
            let color = unsafe { d.colorAttachments().objectAtIndexedSubscript(0) };
            color.setPixelFormat(raw);
            if let Some(factors) = blend.factors() {
                let [rgb_source, rgb_destination, alpha_source, alpha_destination] =
                    factors.map(|f| MTLBlendFactor(f.as_raw() as usize));
                color.setBlendingEnabled(true);
                color.setSourceRGBBlendFactor(rgb_source);
                color.setDestinationRGBBlendFactor(rgb_destination);
                color.setSourceAlphaBlendFactor(alpha_source);
                color.setDestinationAlphaBlendFactor(alpha_destination);
            }
        }
        compile::bounded("format-scout render pipeline", || {
            self.device.newRenderPipelineStateWithDescriptor_error(&d)
        })
        .map_err(|e| error("newRenderPipelineStateWithDescriptor", &e))
    }

    fn pass(
        &self,
        target: &ProtocolObject<dyn MTLTexture>,
        resolve: Option<&ProtocolObject<dyn MTLTexture>>,
        pipeline: Option<&ProtocolObject<dyn MTLRenderPipelineState>>,
    ) -> Result<(), Fail> {
        self.pass_with(target, resolve, pipeline, false)
    }

    /// The runtime side's `pass_with`: loaded where `load`, cleared elsewhere.
    fn pass_with(
        &self,
        target: &ProtocolObject<dyn MTLTexture>,
        resolve: Option<&ProtocolObject<dyn MTLTexture>>,
        pipeline: Option<&ProtocolObject<dyn MTLRenderPipelineState>>,
        load: bool,
    ) -> Result<(), Fail> {
        let pf = self.pf;
        let load_action = if load {
            MTLLoadAction::Load
        } else {
            MTLLoadAction::Clear
        };
        let pass = MTLRenderPassDescriptor::new();
        let store = if resolve.is_some() {
            MTLStoreAction::MultisampleResolve
        } else {
            MTLStoreAction::Store
        };
        if has_depth(pf) || has_stencil(pf) {
            if has_depth(pf) {
                let a = pass.depthAttachment();
                a.setTexture(Some(target));
                a.setResolveTexture(resolve);
                a.setLoadAction(MTLLoadAction::Clear);
                a.setStoreAction(store);
                a.setClearDepth(0.375);
            }
            if has_stencil(pf) {
                let a = pass.stencilAttachment();
                a.setTexture(Some(target));
                a.setResolveTexture(resolve);
                a.setLoadAction(MTLLoadAction::Clear);
                a.setStoreAction(store);
                a.setClearStencil(0x5a);
            }
        } else {
            // SAFETY: attachment 0 of eight.
            let a = unsafe { pass.colorAttachments().objectAtIndexedSubscript(0) };
            a.setTexture(Some(target));
            a.setResolveTexture(resolve);
            a.setLoadAction(load_action);
            a.setStoreAction(store);
            a.setClearColor(MTLClearColor {
                red: 0.125,
                green: 0.25,
                blue: 0.5,
                alpha: 0.75,
            });
        }
        let cb = self.command_buffer()?;
        let e = cb
            .renderCommandEncoderWithDescriptor(&pass)
            .ok_or_else(|| nil("renderCommandEncoderWithDescriptor"))?;
        if let Some(p) = pipeline {
            e.setRenderPipelineState(p);
            // SAFETY: three vertices the vertex function makes from their IDs.
            unsafe { e.drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::Triangle, 0, 3) };
        }
        e.endEncoding();
        self.commit(&cb)
    }

    fn target_usage() -> TextureUsage {
        TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ
    }

    /// The surface a surface cell's texture is made over (`surface.rs`), laid out as the runtime
    /// side's buffer stands for one, with planes where `planar`, filled with the runtime side's
    /// pattern.
    fn surface(&self, planar: bool) -> Result<Surface, Fail> {
        let pf = self.pf;
        let element = layout(pf).map_or((4, 1, 1), |l| (l.unit, SIDE / l.across, SIDE / l.down));
        let facts = if planar {
            super::surface_facts(pf)
        } else {
            None
        };
        let made = match (planar, facts) {
            (false, _) => Surface::plain(element),
            (true, None) => Surface::planar(
                &[Plane {
                    width: SIDE,
                    height: SIDE,
                    bytes_per_element: element.0,
                    element_width: element.1,
                    element_height: element.2,
                }],
                None,
            ),
            (true, Some(f)) => {
                let planes = surface::core_video_planes(f.pixel_format)
                    .or_else(|| surface::known_planes(f.pixel_format))
                    .ok_or_else(|| {
                        Fail::Skip(format!(
                            "no plane layout of OSType '{}'",
                            surface::four_cc(f.pixel_format)
                        ))
                    })?;
                Surface::planar(
                    &planes,
                    Some(TwoPlanes {
                        pixel_format: f.pixel_format,
                        chroma_offset: f.chroma_offset,
                    }),
                )
            }
        };
        let s = made.ok_or_else(|| Fail::Skip("IOSurfaceCreate made no surface".into()))?;
        s.fill(&pattern(pf.as_raw() ^ 0x5, surface::BYTES as u64))
            .ok_or_else(|| Fail::Skip("the surface did not lock".into()))?;
        Ok(s)
    }

    fn cell(&mut self, op: Op) -> Done {
        let pf = self.pf;
        let read = TextureUsage::SHADER_READ;
        match op {
            Op::Own(storage, ty) => {
                let usage = if storage == StorageMode::Memoryless {
                    TextureUsage::RENDER_TARGET
                } else {
                    read
                };
                let d = self.desc(usage, storage);
                d.setTextureType(MTLTextureType(ty.as_raw() as usize));
                // SAFETY: plain sizes and counts of a small texture.
                unsafe {
                    match ty {
                        TextureType::Type1D | TextureType::Type1DArray => d.setHeight(1),
                        TextureType::Type3D => d.setDepth(4),
                        TextureType::Type2DMultisample => {
                            d.setSampleCount(4);
                            d.setUsage(MTLTextureUsage(
                                TextureUsage::RENDER_TARGET.as_raw() as usize
                            ));
                        }
                        _ => {}
                    }
                    if matches!(
                        ty,
                        TextureType::Type1DArray
                            | TextureType::Type2DArray
                            | TextureType::CubeArray
                    ) {
                        d.setArrayLength(2);
                    }
                }
                self.texture(&d)?;
                Ok("made".into())
            }
            Op::Heap(placement) => {
                let h = MTLHeapDescriptor::new();
                h.setType(if placement {
                    MTLHeapType::Placement
                } else {
                    MTLHeapType::Automatic
                });
                h.setStorageMode(MTLStorageMode::Private);
                h.setSize(4 << 20);
                let heap = self
                    .device
                    .newHeapWithDescriptor(&h)
                    .ok_or_else(|| nil("newHeapWithDescriptor"))?;
                let d = self.desc(read, StorageMode::Private);
                let t = if placement {
                    // SAFETY: offset 0 of a 4 MiB heap.
                    unsafe { heap.newTextureWithDescriptor_offset(&d, 0) }
                } else {
                    heap.newTextureWithDescriptor(&d)
                }
                .ok_or_else(|| nil("the heap's newTextureWithDescriptor"))?;
                self.upload(&t)?;
                self.digest_of(&t)
            }
            Op::OverBuffer => {
                let l = layout(pf);
                let buffer = self.buffer(&pattern(pf.as_raw(), 1 << 16))?;
                let row = l.map_or(256, |l| l.row.max(256).next_multiple_of(256));
                let t = buffer
                    .newTextureWithDescriptor_offset_bytesPerRow(
                        &self.desc(read, StorageMode::Shared),
                        0,
                        row as usize,
                    )
                    .ok_or_else(|| nil("newTextureWithDescriptor:offset:bytesPerRow:"))?;
                self.digest_of(&t)
            }
            Op::OverSurface { planar, render } => {
                if render {
                    refuse_ycbcr_surface_pass(pf)?;
                }
                let s = self.surface(planar)?;
                let usage = if render { Self::target_usage() } else { read };
                let d = self.desc(usage, StorageMode::Shared);
                let device: &ProtocolObject<dyn MTLDevice> = &self.device;
                let descriptor: &MTLTextureDescriptor = &d;
                // SAFETY: `newTextureWithDescriptor:iosurface:plane:` of a live surface, whose
                // planes the surface states, at plane 0 (the runtime side's plane).
                let t: Option<Object<dyn MTLTexture>> = unsafe {
                    msg_send![device, newTextureWithDescriptor: descriptor, iosurface: s.as_ptr(), plane: 0usize]
                };
                let t = t.ok_or_else(|| nil("newTextureWithDescriptor:iosurface:plane:"))?;
                if render {
                    let p = self.pipeline(Blend::Off, 1, false)?;
                    self.pass(&t, None, Some(&*p))?;
                    let bytes = s
                        .bytes()
                        .ok_or_else(|| Fail::Skip("the surface did not lock".into()))?;
                    Ok(format!("drawn surface sha={}", digest(&bytes)))
                } else {
                    match self.kernel_read(&t, Some(false)) {
                        Ok(bytes) => Ok(format!("sampled sha={}", digest(&bytes))),
                        Err(Fail::Skip(_)) => {
                            let bytes = self.kernel_read(&t, None)?;
                            Ok(format!("read sha={}", digest(&bytes)))
                        }
                        Err(other) => Err(other),
                    }
                }
            }
            Op::ViewSame | Op::ViewSwizzle => {
                let t = self.texture(
                    &self.desc(read | TextureUsage::PIXEL_FORMAT_VIEW, StorageMode::Private),
                )?;
                self.upload(&t)?;
                let raw = MTLPixelFormat(pf.as_raw() as usize);
                let view = if matches!(op, Op::ViewSame) {
                    t.newTextureViewWithPixelFormat(raw)
                } else {
                    // SAFETY: level 0 and slice 0 of a one-level 2D texture.
                    unsafe {
                        t.newTextureViewWithPixelFormat_textureType_levels_slices_swizzle(
                            raw,
                            MTLTextureType(TextureType::Type2D.as_raw() as usize),
                            NSRange::new(0, 1),
                            NSRange::new(0, 1),
                            MTLTextureSwizzleChannels {
                                red: MTLTextureSwizzle::Blue,
                                green: MTLTextureSwizzle::Green,
                                blue: MTLTextureSwizzle::Red,
                                alpha: MTLTextureSwizzle::One,
                            },
                        )
                    }
                }
                .ok_or_else(|| nil("newTextureViewWithPixelFormat"))?;
                let bytes = self.kernel_read(&view, None)?;
                Ok(format!("read sha={}", digest(&bytes)))
            }
            Op::HostCopy(storage) => {
                let t = self.texture(&self.desc(read, storage))?;
                let l = layout(pf).ok_or_else(|| Fail::Skip("no copy layout".into()))?;
                let bytes = pattern(pf.as_raw(), l.row * l.rows);
                let mut back = vec![0xcdu8; bytes.len()];
                // SAFETY: both byte runs hold the 16 x 16 region at `l.row` bytes a row.
                unsafe {
                    t.replaceRegion_mipmapLevel_slice_withBytes_bytesPerRow_bytesPerImage(
                        region(SIDE, SIDE),
                        0,
                        0,
                        NonNull::new(bytes.as_ptr().cast_mut().cast::<c_void>()).expect("bytes"),
                        l.row as usize,
                        0,
                    );
                    t.getBytes_bytesPerRow_bytesPerImage_fromRegion_mipmapLevel_slice(
                        NonNull::new(back.as_mut_ptr().cast::<c_void>()).expect("bytes"),
                        l.row as usize,
                        0,
                        region(SIDE, SIDE),
                        0,
                        0,
                    );
                }
                let same = if back == bytes { "same" } else { "differs" };
                Ok(format!("roundtrip={same} sha={}", digest(&back)))
            }
            Op::BlitBuffer => {
                let t = self.texture(&self.desc(read, StorageMode::Private))?;
                self.upload(&t)?;
                let l = layout(pf).ok_or_else(|| Fail::Skip("no copy layout".into()))?;
                let back = self.read_back(&t, 0, SIDE)?;
                let same = if back == pattern(pf.as_raw(), l.row * l.rows) {
                    "same"
                } else {
                    "differs"
                };
                Ok(format!("roundtrip={same} sha={}", digest(&back)))
            }
            Op::BlitCopy => {
                let d = self.desc(read, StorageMode::Private);
                let (src, dst) = (self.texture(&d)?, self.texture(&d)?);
                self.upload(&src)?;
                let cb = self.command_buffer()?;
                let blit = cb
                    .blitCommandEncoder()
                    .ok_or_else(|| nil("blitCommandEncoder"))?;
                // SAFETY: two 16 x 16 textures of one format.
                unsafe {
                    blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                        &src,
                        0,
                        0,
                        origin(),
                        size(SIDE, SIDE, 1),
                        &dst,
                        0,
                        0,
                        origin(),
                    );
                }
                blit.endEncoding();
                self.commit(&cb)?;
                self.digest_of(&dst)
            }
            Op::Mipmaps => {
                let d = self.desc(read | TextureUsage::RENDER_TARGET, StorageMode::Private);
                // SAFETY: the full chain of a 16 x 16 texture.
                unsafe { d.setMipmapLevelCount(5) };
                let t = self.texture(&d)?;
                self.upload(&t)?;
                let cb = self.command_buffer()?;
                let blit = cb
                    .blitCommandEncoder()
                    .ok_or_else(|| nil("blitCommandEncoder"))?;
                blit.generateMipmapsForTexture(&t);
                blit.endEncoding();
                self.commit(&cb)?;
                let back = self.read_back(&t, 1, SIDE / 2)?;
                Ok(format!("level1 sha={}", digest(&back)))
            }
            Op::FillBytes | Op::FillColor => {
                let t = self.texture(&self.desc(read, StorageMode::Private))?;
                let (across, down) = layout(pf).map_or((SIDE, SIDE), |l| (l.across, l.down));
                let r = if compressed(pf) {
                    region(across, down)
                } else {
                    region(SIDE, SIDE)
                };
                let cb = self.command_buffer()?;
                let blit = cb
                    .blitCommandEncoder()
                    .ok_or_else(|| nil("blitCommandEncoder"))?;
                let bytes = pattern(pf.as_raw() ^ 0xf1, 16);
                let tex: &ProtocolObject<dyn MTLTexture> = &t;
                let enc: &ProtocolObject<dyn MTLBlitCommandEncoder> = &blit;
                // SAFETY: Metal's private fills (`fillTexture:level:slice:region:bytes:length:`
                // and `...:color:`), which every Apple GPU's blit encoder answers; the bytes
                // live through the call.
                unsafe {
                    if matches!(op, Op::FillBytes) {
                        let _: () = msg_send![enc, fillTexture: tex, level: 0usize, slice: 0usize, region: r, bytes: bytes.as_ptr().cast::<c_void>(), length: bytes.len()];
                    } else {
                        let color = MTLClearColor {
                            red: 0.25,
                            green: 0.5,
                            blue: 0.75,
                            alpha: 1.0,
                        };
                        let _: () = msg_send![enc, fillTexture: tex, level: 0usize, slice: 0usize, region: r, color: color];
                    }
                }
                blit.endEncoding();
                self.commit(&cb)?;
                self.digest_of(&t)
            }
            Op::Read => {
                let t = self.texture(&self.desc(read, StorageMode::Private))?;
                self.upload(&t)?;
                let bytes = self.kernel_read(&t, None)?;
                Ok(format!("read sha={}", digest(&bytes)))
            }
            Op::Write | Op::ReadWrite => {
                let t = self
                    .texture(&self.desc(read | TextureUsage::SHADER_WRITE, StorageMode::Private))?;
                if matches!(op, Op::ReadWrite) {
                    self.upload(&t)?;
                }
                let what = if matches!(op, Op::Write) {
                    "write"
                } else {
                    "read_write"
                };
                self.kernel_store(&t, what)?;
                self.digest_of(&t)
            }
            Op::Sample(linear) => {
                let t = self.texture(&self.desc(read, StorageMode::Private))?;
                self.upload(&t)?;
                let bytes = self.kernel_read(&t, Some(linear))?;
                Ok(format!("sampled sha={}", digest(&bytes)))
            }
            Op::Render { blend } => {
                let p = self.pipeline(Blend::source_over(blend), 1, false)?;
                let t = self.texture(&self.desc(Self::target_usage(), StorageMode::Private))?;
                self.pass(&t, None, Some(&*p))?;
                self.digest_of(&t)
            }
            Op::Clear => {
                let t = self.texture(&self.desc(Self::target_usage(), StorageMode::Private))?;
                self.pass(&t, None, None)?;
                self.digest_of(&t)
            }
            Op::Msaa { resolve } => {
                let p = self.pipeline(Blend::Off, 4, false)?;
                let d = self.desc(TextureUsage::RENDER_TARGET, StorageMode::Private);
                d.setTextureType(MTLTextureType(
                    TextureType::Type2DMultisample.as_raw() as usize
                ));
                // SAFETY: four samples a texel.
                unsafe { d.setSampleCount(4) };
                let t = self.texture(&d)?;
                if resolve {
                    let r = self.texture(&self.desc(Self::target_usage(), StorageMode::Private))?;
                    self.pass(&t, Some(&*r), Some(&*p))?;
                    self.digest_of(&r)
                } else {
                    self.pass(&t, None, Some(&*p))?;
                    Ok("drawn".into())
                }
            }
            Op::Attachment {
                blend,
                depth_or_stencil,
            } => {
                if depth_or_stencil {
                    self.pipeline(Blend::Off, 1, true)?;
                } else if has_depth(pf) || has_stencil(pf) {
                    return Err(Fail::Skip("a depth or stencil format's color cell".into()));
                } else {
                    self.pipeline(Blend::source_over(blend), 1, false)?;
                }
                Ok("made".into())
            }
            Op::Alias { texture } => {
                let h = MTLHeapDescriptor::new();
                h.setType(MTLHeapType::Placement);
                h.setStorageMode(MTLStorageMode::Private);
                h.setSize(4 << 20);
                let heap = self
                    .device
                    .newHeapWithDescriptor(&h)
                    .ok_or_else(|| nil("newHeapWithDescriptor"))?;
                let d = self.desc(read, StorageMode::Private);
                // SAFETY: offset 0 of a 4 MiB heap.
                let live = unsafe { heap.newTextureWithDescriptor_offset(&d, 0) }
                    .ok_or_else(|| nil("the heap's newTextureWithDescriptor:offset:"))?;
                let alias = if texture {
                    let a = MTLTextureDescriptor::new();
                    a.setPixelFormat(MTLPixelFormat(PixelFormat::R8Unorm.as_raw() as usize));
                    // SAFETY: a 16 x 16 texture at offset 0 of a 4 MiB heap.
                    let a = unsafe {
                        a.setWidth(SIDE as usize);
                        a.setHeight(SIDE as usize);
                        a.setUsage(MTLTextureUsage(read.as_raw() as usize));
                        a.setStorageMode(MTLStorageMode::Private);
                        heap.newTextureWithDescriptor_offset(&a, 0)
                    }
                    .ok_or_else(|| nil("the alias texture"))?;
                    format!("alias texture {}x{}", a.width(), a.height())
                } else {
                    // SAFETY: 4096 bytes at offset 0 of a 4 MiB heap.
                    let b = unsafe {
                        heap.newBufferWithLength_options_offset(
                            4096,
                            MTLResourceOptions::StorageModePrivate,
                            0,
                        )
                    }
                    .ok_or_else(|| nil("the alias buffer"))?;
                    format!("alias buffer {}", b.length())
                };
                self.upload(&live)?;
                let read = self.digest_of(&live)?;
                Ok(format!("{alias}; {read}"))
            }
            Op::OverX420 { plane } => {
                let ostype = u32::from_be_bytes(X420);
                let planes = surface::core_video_planes(ostype)
                    .or_else(|| surface::known_planes(ostype))
                    .ok_or_else(|| Fail::Skip("no plane layout of OSType 'x420'".into()))?;
                let s = Surface::planar(
                    &planes,
                    Some(TwoPlanes {
                        pixel_format: ostype,
                        chroma_offset: CHROMA_OFFSET,
                    }),
                )
                .ok_or_else(|| Fail::Skip("IOSurfaceCreate made no surface".into()))?;
                s.fill(&pattern(pf.as_raw() ^ 0x5, surface::BYTES as u64))
                    .ok_or_else(|| Fail::Skip("the surface did not lock".into()))?;
                let side = match plane {
                    0 => SIDE,
                    _ => SIDE / 2,
                };
                let d = self.desc(read, StorageMode::Shared);
                // SAFETY: plain sizes of a small two-dimensional texture.
                unsafe {
                    d.setWidth(side as usize);
                    d.setHeight(side as usize);
                }
                let device: &ProtocolObject<dyn MTLDevice> = &self.device;
                let descriptor: &MTLTextureDescriptor = &d;
                let index = plane as usize;
                // SAFETY: `newTextureWithDescriptor:iosurface:plane:` of a live two-plane
                // surface, at one of its planes.
                let t: Option<Object<dyn MTLTexture>> = unsafe {
                    msg_send![device, newTextureWithDescriptor: descriptor, iosurface: s.as_ptr(), plane: index]
                };
                let t = t.ok_or_else(|| nil("newTextureWithDescriptor:iosurface:plane:"))?;
                let bytes = self.kernel_read(&t, Some(false))?;
                Ok(format!("sampled sha={}", digest(&bytes)))
            }
            Op::BlendLoaded { form, blend } => {
                let p = self.pipeline(blend, 1, false)?;
                let usage = Self::target_usage();
                match form {
                    Form::Surface => {
                        refuse_ycbcr_surface_pass(pf)?;
                        let s = self.surface(true)?;
                        let d = self.desc(usage, StorageMode::Shared);
                        let device: &ProtocolObject<dyn MTLDevice> = &self.device;
                        let descriptor: &MTLTextureDescriptor = &d;
                        // SAFETY: `newTextureWithDescriptor:iosurface:plane:` of a live surface,
                        // whose planes the surface states, at plane 0 (the runtime side's plane).
                        let t: Option<Object<dyn MTLTexture>> = unsafe {
                            msg_send![device, newTextureWithDescriptor: descriptor, iosurface: s.as_ptr(), plane: 0usize]
                        };
                        let t =
                            t.ok_or_else(|| nil("newTextureWithDescriptor:iosurface:plane:"))?;
                        self.pass_with(&t, None, Some(&*p), true)?;
                        let bytes = s
                            .bytes()
                            .ok_or_else(|| Fail::Skip("the surface did not lock".into()))?;
                        Ok(format!("drawn surface sha={}", digest(&bytes)))
                    }
                    Form::Heap | Form::Own => {
                        let d = self.desc(usage, StorageMode::Private);
                        let t = if form == Form::Heap {
                            let h = MTLHeapDescriptor::new();
                            h.setType(MTLHeapType::Placement);
                            h.setStorageMode(MTLStorageMode::Private);
                            h.setSize(4 << 20);
                            let heap = self
                                .device
                                .newHeapWithDescriptor(&h)
                                .ok_or_else(|| nil("newHeapWithDescriptor"))?;
                            // SAFETY: offset 0 of a 4 MiB heap.
                            unsafe { heap.newTextureWithDescriptor_offset(&d, 0) }
                                .ok_or_else(|| nil("the heap's newTextureWithDescriptor:offset:"))?
                        } else {
                            self.texture(&d)?
                        };
                        self.upload(&t)?;
                        self.pass_with(&t, None, Some(&*p), true)?;
                        self.digest_of(&t)
                    }
                }
            }
        }
    }
}

/// The Metal side's child: the cells `indices` of `pf` on the system's default device.
pub(super) fn child(pf: PixelFormat, indices: &[usize]) -> Result<(), String> {
    let all = super::cells();
    let opened = (|| -> Result<Side, String> {
        let device = system_device()?;
        let queue = device.newCommandQueue().ok_or("no command queue")?;
        let source = std::fs::read_to_string(engine_dir().join("format_scout.metal"))
            .map_err(|e| format!("format_scout.metal: {e}"))?;
        let library = compile::bounded("format-scout library", || {
            device.newLibraryWithSource_options_error(&NSString::from_str(&source), None)
        })
        .map_err(|e| format!("the scout's library: {}", e.localizedDescription()))?;
        Ok(Side {
            pf,
            device,
            queue,
            library,
            kernels: HashMap::new(),
            samplers: HashMap::new(),
        })
    })();
    let mut side = match opened {
        Ok(side) => side,
        Err(e) => {
            println!("DEVICE\tfail\t{}", one_line(&e, 300));
            return Ok(());
        }
    };
    println!("DEVICE\tok");
    let mut out = std::io::stdout();
    for &i in indices {
        begin_cell(i);
        let _ = writeln!(out, "BEGIN\t{i}");
        let _ = out.flush();
        let op = all[i].op;
        let result = objc2::rc::autoreleasepool(|_| {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| side.cell(op)))
        });
        let (verdict, detail) = match result {
            Ok(Ok(detail)) => ("made", detail),
            Ok(Err(Fail::Refused(why))) => ("refused", why),
            Ok(Err(Fail::Died(why))) => ("died", why),
            Ok(Err(Fail::Skip(why))) => ("skip", why),
            Err(panic) => {
                let text = panic
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| panic.downcast_ref::<&str>().map(|s| (*s).to_owned()))
                    .unwrap_or_default();
                ("died", format!("panic: {text}"))
            }
        };
        let _ = writeln!(out, "CELL\t{i}\t{verdict}\t{}", one_line(&detail, 600));
        let _ = out.flush();
    }
    Ok(())
}
