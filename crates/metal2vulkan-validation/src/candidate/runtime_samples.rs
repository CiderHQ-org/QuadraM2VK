//! A multisampled texture filled sample by sample through the Metal-like runtime, as the Metal
//! oracle fills one (`metal::platform::textures`'s multisample initializer): Metal writes no
//! multisampled texture from the CPU, so a render pass per slice draws the fullscreen triangle at
//! sample rate and each invocation writes its own sample's value from a buffer of every pixel's
//! samples in order. The functions are the owned fixture `rt6_samples`'s (the oracle's source with
//! the extent's width and the sample count read from a buffer, where the oracle writes them into
//! its source), one fragment function per format the oracle fills.
//!
//! A fill is encoded into the command buffer whose work reads the texture, ahead of that work
//! ([`SampleFill::encode`]): Metal runs a command buffer's passes in order, so nothing waits on
//! the fill alone and one wait covers it.
//!
//! A multisampled attachment is read back sample by sample the same way ([`encode`], as the
//! oracle's `metal::platform` `sample_readback` reads one): Metal copies no multisampled texture
//! to a buffer, so a dispatch of one thread per sample (threadgroups of one, the grid the pixels
//! by the samples) reads each sample and writes the texel's words at its place in the
//! sample-minor layout, with the fixture's kernels. A `Depth32Float` attachment is read as its
//! bits, a `Stencil8` one as its first component. A color one is read through an unsigned view
//! with its own components and bits (`color_view`), whose kernel packs them into the words the
//! oracle's view of the texel size gives: the bytes are the oracle's, and no device needs
//! Vulkan's `imageViewFormatReinterpretation` (MoltenVK withholds it). Nothing is resolved.

use super::engine::resources::pixel_format;
use super::runtime_case::{buffer_with, read_buffer, texture_descriptor, CaseError};
use crate::air_library::AirLibrary;
use crate::case::TextureFormat;
use crate::executor_contract::multisample::{readback_bytes, unpack_words, SampleRead};
use crate::literal::{LiteralTexture, TextureLayout};
use metal2vulkan::metal::state::DepthStencilDescriptor;
use metal2vulkan::metal::{CompareFunction, PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::FunctionDescriptor;
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    Buffer, CommandBuffer, DepthStencilState, Device, Library, LoadAction, PipelineOptions,
    RenderPassDescriptor, RenderPipelineState, ResourceOptions, Size, StoreAction, Texture,
};
use std::sync::OnceLock;

/// The owned fixture's opaque-pointer text.
const FIXTURE: &str = "rt6_samples.ll";

/// The fixture's vertex function: the oracle's fullscreen triangle.
const VERTEX: &str = "rt6_fill_vertex";

/// The fixture's library on `device`, its bitcode assembled once per process.
pub fn library(device: &Device) -> Result<Library, CaseError> {
    static BITCODE: OnceLock<Result<Vec<u8>, String>> = OnceLock::new();
    let bitcode = BITCODE
        .get_or_init(|| {
            llvm_assemble(&engine_dir().join(FIXTURE))
                .map_err(|error| format!("llvm-as of {FIXTURE}: {error:?}"))
        })
        .as_ref()
        .map_err(|error| CaseError::from(error.clone()))?;
    device
        .new_library_with_air(bitcode)
        .map_err(CaseError::helper(format!(
            "{FIXTURE}: new_library_with_air"
        )))
}

/// A buffer of the shape the fixture's functions read: the extent's `width` and the `samples` of
/// each pixel.
pub fn shape(device: &Device, width: u32, samples: u32) -> Result<Buffer, CaseError> {
    let mut bytes = width.to_le_bytes().to_vec();
    bytes.extend_from_slice(&samples.to_le_bytes());
    buffer_with(device, &bytes)
}

/// The fixture's fragment function that fills a texture of `format`, as the oracle's
/// initializer fills it: a sampled-only format and the 64-bit one are no multisampled texture's.
fn fill_function(format: TextureFormat) -> Result<&'static str, String> {
    Ok(match format {
        TextureFormat::R8Unorm => "rt6_fill_r8unorm",
        TextureFormat::Rgba8Unorm => "rt6_fill_rgba8unorm",
        TextureFormat::Rgba8Uint => "rt6_fill_rgba8uint",
        TextureFormat::Rgba8Sint => "rt6_fill_rgba8sint",
        TextureFormat::R16Float => "rt6_fill_r16float",
        TextureFormat::R16Uint => "rt6_fill_r16uint",
        TextureFormat::Rg16Float => "rt6_fill_rg16float",
        TextureFormat::Rg32Float => "rt6_fill_rg32float",
        TextureFormat::Rgba16Float => "rt6_fill_rgba16float",
        TextureFormat::Rgba16Uint => "rt6_fill_rgba16uint",
        TextureFormat::Rgba16Sint => "rt6_fill_rgba16sint",
        TextureFormat::R32Uint => "rt6_fill_r32uint",
        TextureFormat::R32Sint => "rt6_fill_r32sint",
        TextureFormat::R32Float => "rt6_fill_r32float",
        TextureFormat::Rgba32Uint => "rt6_fill_rgba32uint",
        TextureFormat::Rgba32Sint => "rt6_fill_rgba32sint",
        TextureFormat::Rgba32Float => "rt6_fill_rgba32float",
        TextureFormat::Depth32Float => "rt6_fill_depth32float",
        TextureFormat::R8Snorm
        | TextureFormat::Rgba8Snorm
        | TextureFormat::R16Unorm
        | TextureFormat::Rgba16Unorm
        | TextureFormat::R16Snorm
        | TextureFormat::Rg16Snorm
        | TextureFormat::Rgba16Snorm
        | TextureFormat::Rgba8UnormSrgb
        | TextureFormat::Bgra8Unorm
        | TextureFormat::Bgra8UnormSrgb
        | TextureFormat::R8UnormSrgb
        | TextureFormat::Rg8Unorm
        | TextureFormat::Rg8UnormSrgb
        | TextureFormat::Rg8Snorm
        | TextureFormat::Rg16Unorm
        | TextureFormat::A8Unorm
        | TextureFormat::B5g6r5Unorm
        | TextureFormat::A1bgr5Unorm
        | TextureFormat::Abgr4Unorm
        | TextureFormat::Bgr5a1Unorm
        | TextureFormat::Rgb10a2Unorm
        | TextureFormat::Bgr10a2Unorm
        | TextureFormat::Rg11b10Float
        | TextureFormat::Rgb9e5Float => {
            return Err(format!(
                "{format:?} is a sampled-only format, which no case multisamples"
            ))
        }
        TextureFormat::Rg32Uint => {
            return Err(format!(
                "{format:?} is the ulong texture format, which no case multisamples"
            ))
        }
    })
}

/// The fixture's function `name` on `device`.
fn entry(device: &Device, name: &str) -> Result<FunctionDescriptor, CaseError> {
    library(device)?
        .new_function_with_name(name)
        .map(|f| f.function_descriptor())
        .map_err(CaseError::helper(format!("{FIXTURE}: function {name}")))
}

/// The render pipeline [`SampleFill::new`] makes the state of its fill of a texture of `format`
/// and `samples` of, for a test that expects what the making announces.
pub fn fill_descriptor(
    device: &Device,
    format: TextureFormat,
    samples: u32,
) -> Result<RenderPipelineDescriptor, CaseError> {
    let function = fill_function(format)?;
    let mut descriptor =
        RenderPipelineDescriptor::new(StageSource::Function(entry(device, VERTEX)?));
    descriptor.fragment = Some(StageSource::Function(entry(device, function)?));
    if format == TextureFormat::Depth32Float {
        descriptor.depth_format = Some(PixelFormat::Depth32Float);
    } else {
        descriptor.color_attachments[0] =
            Some(ColorAttachmentDescriptor::new(pixel_format(format)));
    }
    descriptor.raster_sample_count = samples;
    Ok(descriptor)
}

/// The kernel [`encode`] reads a texture of `format` as `read` with, for a test that expects
/// what the making of its state announces.
pub fn read_function(
    device: &Device,
    format: PixelFormat,
    read: SampleRead,
) -> Result<FunctionDescriptor, CaseError> {
    let kernel = match read {
        SampleRead::Color(_) => color_view(format)?.1,
        SampleRead::Depth => "rt6_read_depth",
        SampleRead::Stencil => "rt6_read_stencil",
    };
    entry(device, kernel)
}

/// The view [`encode`] reads a color texture of `format` through: an unsigned format of its
/// components and bits, which is `format` itself for an unsigned one. A texture made without
/// `PixelFormatView` usage holds a view of another format apart (`shadowed-texture-view`).
pub fn read_view(format: PixelFormat) -> Result<PixelFormat, CaseError> {
    Ok(color_view(format)?.0)
}

/// The fill of one multisampled texture: its pipeline, the buffers it reads, and the slices it
/// writes ([`SampleFill::new`]).
pub struct SampleFill {
    texture: Texture,
    pipeline: RenderPipelineState,
    /// A depth texture's state: every sample written, whatever it held (`Always`).
    depth_state: Option<DepthStencilState>,
    values: Buffer,
    shape: Buffer,
    layers: u32,
    layer_bytes: u64,
}

impl SampleFill {
    /// The fill of `texture`, a multisampled render target of `format` and `layout`, with
    /// `bytes`: each slice's pixels, rows in order, each pixel's samples in order.
    pub fn new(
        device: &Device,
        texture: &Texture,
        format: TextureFormat,
        layout: TextureLayout,
        bytes: &[u8],
    ) -> Result<SampleFill, CaseError> {
        let function = fill_function(format)?;
        let depth = format == TextureFormat::Depth32Float;
        let descriptor = fill_descriptor(device, format, layout.sample_count)?;
        let pipeline = device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .map_err(CaseError::helper(format!(
                "{function}: new_render_pipeline_state"
            )))?;
        let depth_state = depth
            .then(|| {
                let mut state = DepthStencilDescriptor::default();
                state.depth_compare_function = CompareFunction::Always;
                state.depth_write_enabled = true;
                device
                    .new_depth_stencil_state(&state)
                    .map_err(CaseError::runtime("new_depth_stencil_state"))
            })
            .transpose()?;
        let layer_bytes = [
            layout.width,
            layout.height,
            layout.sample_count,
            format.bytes_per_pixel() as u32,
        ]
        .into_iter()
        .try_fold(1u64, |n, f| n.checked_mul(u64::from(f)))
        .ok_or("a multisampled texture's slice size overflows")?;
        let needed = layer_bytes * u64::from(layout.array_layers);
        if (bytes.len() as u64) < needed || needed == 0 {
            return Err(format!(
                "a multisampled texture supplies {} bytes but its extent needs {needed}",
                bytes.len()
            )
            .into());
        }
        Ok(SampleFill {
            texture: texture.clone(),
            pipeline,
            depth_state,
            values: buffer_with(device, bytes)?,
            shape: shape(device, layout.width, layout.sample_count)?,
            layers: layout.array_layers,
            layer_bytes,
        })
    }

    /// Encode the fill into `command_buffer`: per slice, a render pass that loads nothing and
    /// stores every sample, drawing the fullscreen triangle once.
    pub fn encode(&self, command_buffer: &CommandBuffer) -> Result<(), CaseError> {
        self.encode_into(command_buffer, None)
    }

    /// [`SampleFill::encode`] of a color texture whose passes also resolve each slice into the
    /// same slice of `resolve` (`StoreAndMultisampleResolve`): a draw's samples resolved by the
    /// store action of the pass that wrote them.
    pub fn encode_resolving(
        &self,
        command_buffer: &CommandBuffer,
        resolve: &Texture,
    ) -> Result<(), CaseError> {
        self.encode_into(command_buffer, Some(resolve))
    }

    /// The fill's passes, each resolving its color slice into `resolve` where given.
    fn encode_into(
        &self,
        command_buffer: &CommandBuffer,
        resolve: Option<&Texture>,
    ) -> Result<(), CaseError> {
        for layer in 0..self.layers {
            let mut pass = RenderPassDescriptor::new();
            let texture = Some(self.texture.clone());
            if self.depth_state.is_some() {
                let attachment = &mut pass.depth_attachment;
                attachment.texture = texture;
                attachment.slice = layer.into();
                attachment.load_action = LoadAction::DontCare;
                attachment.store_action = StoreAction::Store;
            } else {
                let attachment = &mut pass.color_attachments[0];
                attachment.texture = texture;
                attachment.slice = layer.into();
                attachment.load_action = LoadAction::DontCare;
                attachment.store_action = StoreAction::Store;
                if let Some(resolve) = resolve {
                    attachment.resolve_texture = Some(resolve.clone());
                    attachment.resolve_slice = layer.into();
                    attachment.store_action = StoreAction::StoreAndMultisampleResolve;
                }
            }
            let mut encoder =
                command_buffer
                    .render_command_encoder(&pass)
                    .map_err(CaseError::runtime(
                        "the sample fill's render_command_encoder",
                    ))?;
            let fill = CaseError::runtime;
            encoder
                .set_render_pipeline_state(&self.pipeline)
                .map_err(fill("the sample fill's set_render_pipeline_state"))?;
            if let Some(state) = &self.depth_state {
                encoder
                    .set_depth_stencil_state(state)
                    .map_err(fill("the sample fill's set_depth_stencil_state"))?;
            }
            encoder
                .set_fragment_buffer(Some(&self.values), u64::from(layer) * self.layer_bytes, 0)
                .map_err(fill("the sample fill's set_fragment_buffer"))?;
            encoder
                .set_fragment_buffer(Some(&self.shape), 0, 1)
                .map_err(fill("the sample fill's set_fragment_buffer"))?;
            encoder
                .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                .map_err(fill("the sample fill's draw_primitives"))?;
            encoder
                .end_encoding()
                .map_err(fill("the sample fill's end_encoding"))?;
        }
        Ok(())
    }
}

/// The multisampled texture of `literal` on `device`, made as the Metal oracle makes it: its
/// type, pixel format, extent and samples, `Private` (Metal keeps no multisampled texture in
/// shared memory), `usage` and `RenderTarget`; with the fill that writes the literal's bytes into
/// it, for the command buffer that reads it.
pub fn multisampled_texture_with(
    device: &Device,
    literal: &LiteralTexture,
    usage: TextureUsage,
) -> Result<(Texture, SampleFill), CaseError> {
    let layout = literal.layout()?;
    let mut descriptor = texture_descriptor(literal, usage | TextureUsage::RENDER_TARGET)?;
    descriptor.sample_count = layout.sample_count.into();
    descriptor.resource_options = ResourceOptions::private();
    let texture = device
        .new_texture(&descriptor)
        .map_err(CaseError::runtime(format!(
            "texture {}: new_texture",
            literal.binding
        )))?;
    let fill = SampleFill::new(device, &texture, literal.format, layout, &literal.bytes)?;
    Ok((texture, fill))
}

/// One attachment's readback, encoded: the words its dispatch writes, read once the command
/// buffer completed ([`SampleReadback::bytes`]).
pub struct SampleReadback {
    words: Buffer,
    read: SampleRead,
    extent: [u32; 2],
    samples: u32,
}

/// The unsigned view of a color attachment of `format` with the same components and bits, the
/// fixture's kernel that packs a texel of it into words, and how many words it writes. Only the
/// formats the fill writes ([`fill_function`]) are read.
fn color_view(format: PixelFormat) -> Result<(PixelFormat, &'static str, usize), String> {
    use PixelFormat as P;
    Ok(match format {
        P::R8Unorm => (P::R8Uint, "rt6_read_words1", 1),
        P::RGBA8Unorm | P::RGBA8Uint | P::RGBA8Sint => (P::RGBA8Uint, "rt6_read_bytes4", 1),
        P::R16Float | P::R16Uint => (P::R16Uint, "rt6_read_words1", 1),
        P::RG16Float => (P::RG16Uint, "rt6_read_halves2", 1),
        P::RGBA16Float | P::RGBA16Uint | P::RGBA16Sint => (P::RGBA16Uint, "rt6_read_halves4", 2),
        P::R32Float | P::R32Uint | P::R32Sint => (P::R32Uint, "rt6_read_words1", 1),
        P::RG32Float => (P::RG32Uint, "rt6_read_words2", 2),
        P::RGBA32Float | P::RGBA32Uint | P::RGBA32Sint => (P::RGBA32Uint, "rt6_read_words4", 4),
        other => {
            return Err(format!(
                "a multisampled {other:?} attachment, which nothing fills"
            ))
        }
    })
}

/// Encode into `command_buffer` the readback of every sample of `texture`, a multisampled
/// attachment of `extent` and `samples`, read as `read`.
pub fn encode(
    device: &Device,
    command_buffer: &CommandBuffer,
    texture: &Texture,
    read: SampleRead,
    extent: [u32; 2],
    samples: u32,
) -> Result<SampleReadback, CaseError> {
    let (source, kernel) = match read {
        SampleRead::Color(_) => {
            let (view, kernel, words) = color_view(texture.pixel_format())?;
            if words != read.words() {
                return Err(format!(
                    "{kernel} writes {words} words of a texel, the oracle's view {}",
                    read.words()
                )
                .into());
            }
            let view = texture
                .new_texture_view(view)
                .map_err(CaseError::runtime(format!("{kernel}: new_texture_view")))?;
            (view, kernel)
        }
        SampleRead::Depth => (texture.clone(), "rt6_read_depth"),
        SampleRead::Stencil => (texture.clone(), "rt6_read_stencil"),
    };
    encode_through(
        device,
        command_buffer,
        &source,
        kernel,
        read,
        extent,
        samples,
    )
}

/// Encode into `command_buffer` the readback of every sample of `source`, a multisampled texture
/// or a view of one, of `extent` and `samples`, by the fixture's kernel `name`, which writes the
/// words of `read`.
pub fn encode_through(
    device: &Device,
    command_buffer: &CommandBuffer,
    source: &Texture,
    name: &'static str,
    read: SampleRead,
    extent: [u32; 2],
    samples: u32,
) -> Result<SampleReadback, CaseError> {
    let function = library(device)?
        .new_function_with_name(name)
        .map_err(CaseError::helper(format!("function {name}")))?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function, &PipelineOptions::default())
        .map_err(CaseError::helper(format!(
            "{name}: new_compute_pipeline_state"
        )))?;
    let length = readback_bytes(extent, samples, read)? as u64;
    let words = device
        .new_buffer(length, ResourceOptions::shared())
        .map_err(CaseError::runtime(format!("{name}: new_buffer")))?;
    let shape = shape(device, extent[0], samples)?;
    let mut encoder = command_buffer
        .compute_command_encoder()
        .map_err(CaseError::runtime(format!(
            "{name}: compute_command_encoder"
        )))?;
    let call = |what: &str| CaseError::runtime(format!("{name}: {what}"));
    encoder
        .set_compute_pipeline_state(&pipeline)
        .map_err(call("set_compute_pipeline_state"))?;
    encoder
        .set_texture(Some(source), 0)
        .map_err(call("set_texture"))?;
    encoder
        .set_buffer(Some(&words), 0, 0)
        .map_err(call("set_buffer"))?;
    encoder
        .set_buffer(Some(&shape), 0, 1)
        .map_err(call("set_buffer"))?;
    // One thread per sample, one per threadgroup: no grid edge to guard.
    encoder
        .dispatch_threadgroups(
            Size::new(extent[0].into(), extent[1].into(), samples.into()),
            Size::new(1, 1, 1),
        )
        .map_err(call("dispatch_threadgroups"))?;
    encoder.end_encoding().map_err(call("end_encoding"))?;
    Ok(SampleReadback {
        words,
        read,
        extent,
        samples,
    })
}

impl SampleReadback {
    /// Every sample's bytes in the attachment layout, once the command buffer completed.
    pub fn bytes(&self) -> Result<Vec<u8>, CaseError> {
        let length = readback_bytes(self.extent, self.samples, self.read)? as u64;
        let words = read_buffer(&self.words, 0, length)?;
        Ok(unpack_words(&words, self.extent, self.samples, self.read)?)
    }
}
