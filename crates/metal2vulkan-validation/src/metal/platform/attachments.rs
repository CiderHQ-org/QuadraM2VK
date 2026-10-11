//! Render-attachment allocation and reflected depth/stencil state.

use super::commit_and_wait;
use super::textures::{
    initialize_multisample_texture, metal_pixel_format, upload_texture, Texture,
};
use crate::case::{TextureFormat, TextureType};
use crate::executor_contract::multisample::uniform_stencil;
use crate::literal::LiteralResources;
use core::ffi::c_void;
use core::ptr::NonNull;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLCompareFunction,
    MTLDepthStencilDescriptor, MTLDepthStencilState, MTLDevice, MTLLoadAction, MTLOrigin,
    MTLPixelFormat, MTLRegion, MTLRenderPassDescriptor, MTLSize, MTLStencilDescriptor,
    MTLStencilOperation, MTLStorageMode, MTLStoreAction, MTLTexture, MTLTextureDescriptor,
    MTLTextureType, MTLTextureUsage,
};

type DepthStencilState = Retained<ProtocolObject<dyn MTLDepthStencilState>>;

/// Each render target of `resources`: a one-sample target as a shared texture uploaded by
/// `replaceRegion` (a one-slice array in a `layered` pass), a multisampled one as
/// [`make_multisampled_attachment`].
pub(super) fn make_render_targets(
    device: &ProtocolObject<dyn MTLDevice>,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    resources: &LiteralResources,
    layered: bool,
) -> Result<Vec<(u32, Texture)>, String> {
    resources
        .render_targets
        .iter()
        .map(|resource| {
            if resource.samples > 1 {
                if layered {
                    return Err("a layered multisampled pass is not modelled".into());
                }
                let texture = make_multisampled_attachment(
                    device,
                    queue,
                    &format!("render target {}", resource.index),
                    resource.format,
                    resource.dimensions,
                    resource.samples,
                    &resource.bytes,
                )?;
                return Ok((resource.index, texture));
            }
            let descriptor = MTLTextureDescriptor::new();
            descriptor.setTextureType(if layered {
                MTLTextureType::Type2DArray
            } else {
                MTLTextureType::Type2D
            });
            descriptor.setPixelFormat(metal_pixel_format(resource.format));
            unsafe {
                descriptor.setWidth(resource.dimensions[0] as usize);
                descriptor.setHeight(resource.dimensions[1] as usize);
                if layered {
                    descriptor.setArrayLength(1);
                }
            }
            descriptor.setStorageMode(MTLStorageMode::Shared);
            descriptor.setUsage(MTLTextureUsage::RenderTarget);
            let texture = device
                .newTextureWithDescriptor(&descriptor)
                .ok_or_else(|| format!("create Metal render target {}", resource.index))?;
            upload_texture(
                &texture,
                &format!("render target {}", resource.index),
                if layered {
                    TextureType::D2Array
                } else {
                    TextureType::D2
                },
                [resource.dimensions[0], resource.dimensions[1], 1],
                1,
                resource.format,
                &resource.bytes,
            )?;
            Ok((
                resource.index,
                Texture {
                    object: texture,
                    _backing: None,
                    _heap: None,
                },
            ))
        })
        .collect()
}

pub(super) fn make_rasterization_sink(
    device: &ProtocolObject<dyn MTLDevice>,
) -> Result<Texture, String> {
    let descriptor = MTLTextureDescriptor::new();
    descriptor.setTextureType(MTLTextureType::Type2D);
    descriptor.setPixelFormat(MTLPixelFormat::R8Unorm);
    unsafe {
        descriptor.setWidth(1);
        descriptor.setHeight(1);
    }
    descriptor.setStorageMode(MTLStorageMode::Shared);
    descriptor.setUsage(MTLTextureUsage::RenderTarget);
    Ok(Texture {
        object: device
            .newTextureWithDescriptor(&descriptor)
            .ok_or_else(|| "create Metal rasterization-disabled sink".to_string())?,
        _backing: None,
        _heap: None,
    })
}

/// A multisampled attachment of `format`: a 2D multisample texture of `samples` samples, private
/// (Metal keeps no multisampled texture in shared memory), a render target that the per-sample
/// readback reads (`ShaderRead`), through a view of another format for a color attachment
/// (`PixelFormatView`), and filled sample by sample with `bytes` (each pixel's samples in order) by
/// the literal initializer before the case's pass loads it. A count the device does not rasterize
/// is refused before anything is made.
fn make_multisampled_attachment(
    device: &ProtocolObject<dyn MTLDevice>,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    label: &str,
    format: TextureFormat,
    [width, height]: [u32; 2],
    samples: u32,
    bytes: &[u8],
) -> Result<Texture, String> {
    if !device.supportsTextureSampleCount(samples as usize) {
        return Err(format!(
            "the Metal device does not rasterize {samples} samples ({label})"
        ));
    }
    let layout =
        crate::literal::texture_layout(TextureType::D2Multisample, [width, height, 1], samples)?;
    let descriptor = MTLTextureDescriptor::new();
    descriptor.setTextureType(MTLTextureType::Type2DMultisample);
    descriptor.setPixelFormat(metal_pixel_format(format));
    unsafe {
        descriptor.setWidth(width as usize);
        descriptor.setHeight(height as usize);
        descriptor.setSampleCount(samples as usize);
    }
    descriptor.setStorageMode(MTLStorageMode::Private);
    let view = if format.is_depth() {
        MTLTextureUsage::empty()
    } else {
        MTLTextureUsage::PixelFormatView
    };
    descriptor.setUsage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead | view);
    let texture = device
        .newTextureWithDescriptor(&descriptor)
        .ok_or_else(|| format!("create Metal multisampled {label}"))?;
    initialize_multisample_texture(device, queue, &texture, label, layout, format, bytes)?;
    Ok(Texture {
        object: texture,
        _backing: None,
        _heap: None,
    })
}

pub(super) struct DepthStencilTexture {
    pub(super) depth: Option<Texture>,
    pub(super) stencil: Option<Texture>,
}

/// A multisampled `Stencil8` attachment of `samples` samples whose samples all hold the one value
/// of `bytes` (`executor_contract::multisample`): private, a render target that the per-sample
/// readback reads (`ShaderRead`, as `texture2d_ms<uint>`), and cleared to that value by a pass of
/// its own (load action `Clear`, no draw) before the case's pass loads it.
fn make_multisampled_stencil(
    device: &ProtocolObject<dyn MTLDevice>,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    [width, height]: [u32; 2],
    samples: u32,
    bytes: &[u8],
) -> Result<Texture, String> {
    let value = uniform_stencil(bytes)?;
    if !device.supportsTextureSampleCount(samples as usize) {
        return Err(format!(
            "the Metal device does not rasterize {samples} samples (stencil attachment)"
        ));
    }
    let descriptor = MTLTextureDescriptor::new();
    descriptor.setTextureType(MTLTextureType::Type2DMultisample);
    descriptor.setPixelFormat(MTLPixelFormat::Stencil8);
    unsafe {
        descriptor.setWidth(width as usize);
        descriptor.setHeight(height as usize);
        descriptor.setSampleCount(samples as usize);
    }
    descriptor.setStorageMode(MTLStorageMode::Private);
    descriptor.setUsage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
    let texture = device
        .newTextureWithDescriptor(&descriptor)
        .ok_or("create Metal multisampled stencil attachment")?;
    let pass = MTLRenderPassDescriptor::new();
    let attachment = pass.stencilAttachment();
    attachment.setTexture(Some(&*texture));
    attachment.setLoadAction(MTLLoadAction::Clear);
    attachment.setClearStencil(u32::from(value));
    attachment.setStoreAction(MTLStoreAction::Store);
    let command = queue
        .commandBuffer()
        .ok_or("MTLCommandQueue::commandBuffer returned nil")?;
    command
        .renderCommandEncoderWithDescriptor(&pass)
        .ok_or("create the multisampled stencil clear encoder")?
        .endEncoding();
    commit_and_wait(&command, "Metal multisampled stencil clear")?;
    Ok(Texture {
        object: texture,
        _backing: None,
        _heap: None,
    })
}

/// The depth and stencil attachments of `resources`, each authored aspect its own texture
/// (Metal's separate `Depth32Float` and `Stencil8`): of one sample, shared and uploaded by
/// `replaceRegion`; of more, the depth aspect a [`make_multisampled_attachment`] and the stencil
/// aspect a [`make_multisampled_stencil`].
pub(super) fn make_depth_stencil(
    device: &ProtocolObject<dyn MTLDevice>,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    resources: &LiteralResources,
    layered: bool,
) -> Result<Option<DepthStencilTexture>, String> {
    let Some(resource) = &resources.depth_stencil else {
        return Ok(None);
    };
    if resource.samples > 1 {
        if layered {
            return Err("a layered multisampled pass is not modelled".into());
        }
        let depth = resource
            .depth
            .as_deref()
            .map(|depth| {
                make_multisampled_attachment(
                    device,
                    queue,
                    "depth attachment",
                    TextureFormat::Depth32Float,
                    resource.dimensions,
                    resource.samples,
                    depth,
                )
            })
            .transpose()?;
        let stencil = resource
            .stencil
            .as_deref()
            .map(|stencil| {
                make_multisampled_stencil(
                    device,
                    queue,
                    resource.dimensions,
                    resource.samples,
                    stencil,
                )
            })
            .transpose()?;
        if depth.is_none() && stencil.is_none() {
            return Err("depth/stencil attachment has no aspect bytes".into());
        }
        return Ok(Some(DepthStencilTexture { depth, stencil }));
    }
    let make_aspect = |format, bytes: &[u8], pixel_size| -> Result<Texture, String> {
        let descriptor = MTLTextureDescriptor::new();
        descriptor.setTextureType(if layered {
            MTLTextureType::Type2DArray
        } else {
            MTLTextureType::Type2D
        });
        descriptor.setPixelFormat(format);
        unsafe {
            descriptor.setWidth(resource.dimensions[0] as usize);
            descriptor.setHeight(resource.dimensions[1] as usize);
            if layered {
                descriptor.setArrayLength(1);
            }
        }
        descriptor.setStorageMode(MTLStorageMode::Shared);
        descriptor.setUsage(MTLTextureUsage::RenderTarget);
        let texture = device
            .newTextureWithDescriptor(&descriptor)
            .ok_or_else(|| "create Metal depth/stencil attachment".to_string())?;
        let pointer = NonNull::new(bytes.as_ptr().cast_mut().cast::<c_void>())
            .ok_or_else(|| "depth/stencil bytes pointer is null".to_string())?;
        let bytes_per_row = resource.dimensions[0] as usize * pixel_size;
        unsafe {
            texture.replaceRegion_mipmapLevel_slice_withBytes_bytesPerRow_bytesPerImage(
                MTLRegion {
                    origin: MTLOrigin { x: 0, y: 0, z: 0 },
                    size: MTLSize {
                        width: resource.dimensions[0] as usize,
                        height: resource.dimensions[1] as usize,
                        depth: 1,
                    },
                },
                0,
                0,
                pointer,
                bytes_per_row,
                bytes_per_row * resource.dimensions[1] as usize,
            );
        }
        Ok(Texture {
            object: texture,
            _backing: None,
            _heap: None,
        })
    };
    let depth = resource
        .depth
        .as_deref()
        .map(|bytes| make_aspect(MTLPixelFormat::Depth32Float, bytes, 4))
        .transpose()?;
    let stencil = resource
        .stencil
        .as_deref()
        .map(|bytes| make_aspect(MTLPixelFormat::Stencil8, bytes, 1))
        .transpose()?;
    if depth.is_none() && stencil.is_none() {
        return Err("depth/stencil attachment has no aspect bytes".into());
    }
    Ok(Some(DepthStencilTexture { depth, stencil }))
}

/// The `MTLDepthStencilState` of `state`'s descriptor ([`crate::executor_contract::DrawDepthStencil`]
/// of the case: its authored state, or the one its fragment function's AIR-layer interface
/// implies; the state the engine candidate sets too). A face whose stencil test is the identity
/// is left unset, as before an authored state existed (Metal's default).
pub(super) fn make_depth_stencil_state(
    device: &ProtocolObject<dyn MTLDevice>,
    state: &crate::executor_contract::DrawDepthStencil,
) -> Result<DepthStencilState, String> {
    let desc = &state.descriptor;
    let descriptor = MTLDepthStencilDescriptor::new();
    descriptor.setDepthWriteEnabled(desc.depth_write_enabled);
    descriptor
        .setDepthCompareFunction(MTLCompareFunction(desc.depth_compare_function.as_raw() as _));
    if state.tests_stencil() {
        let face = |s: &metal2vulkan::metal::state::StencilDescriptor| {
            let stencil = MTLStencilDescriptor::new();
            stencil.setStencilCompareFunction(MTLCompareFunction(
                s.stencil_compare_function.as_raw() as _,
            ));
            stencil.setStencilFailureOperation(MTLStencilOperation(
                s.stencil_failure_operation.as_raw() as _,
            ));
            stencil.setDepthFailureOperation(MTLStencilOperation(
                s.depth_failure_operation.as_raw() as _,
            ));
            stencil.setDepthStencilPassOperation(MTLStencilOperation(
                s.depth_stencil_pass_operation.as_raw() as _,
            ));
            stencil.setReadMask(s.read_mask);
            stencil.setWriteMask(s.write_mask);
            stencil
        };
        descriptor.setFrontFaceStencil(Some(&face(&desc.front_face_stencil)));
        descriptor.setBackFaceStencil(Some(&face(&desc.back_face_stencil)));
    }
    device
        .newDepthStencilStateWithDescriptor(&descriptor)
        .ok_or_else(|| "create Metal depth/stencil state".to_string())
}
