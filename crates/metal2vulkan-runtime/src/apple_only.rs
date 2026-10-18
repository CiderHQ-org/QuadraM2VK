//! The Apple-only calls of the consumer's trait: each is accepted where it is exact on Vulkan, and
//! otherwise refused with [`AppleOnly`] naming the missing feature, and [`AppleOnly::gap`] saying
//! why: Metal's answer is not measured, or the runtime has not built it. None is a device's gap:
//! each is the runtime's own.
//!
//! Exact means one of:
//! - the call states Metal's default, which every runtime object already has: a line width of
//!   1, the viewport transform on, no protection options, provoking vertex mode 0
//!   (the first vertex, as Vulkan's default and every runtime draw: a Metal probe drew a flat
//!   varying from the first vertex without the call and after mode 0, from the last after mode 1);
//! - the call acts on state no runtime object has: Vulkan shows no texture compression, so a
//!   reinterpretation flush has nothing to flush, and after an invalidation Metal leaves the
//!   contents undefined, so keeping them is one of the outcomes Metal allows; and no pipeline the
//!   runtime builds has a fixed-function alpha test, so its reference value has no effect, as on
//!   Metal with the test off.
//!
//! - `setPointSize:`, which Metal measured makes a no-op: the point is the vertex function's
//!   `[[point_size]]`.
//!
//! Every other call is refused, and records nothing:
//! - nonzero protection options, whose effect on Metal is not measured (Vulkan's protected
//!   memory is the nearest form);
//! - a line width other than 1 (Apple's wide-line rule is unmeasured, where Vulkan has
//!   `wideLines`), and vertex amplification mode, and depth or stencil "cleared", whose meaning
//!   or raw values are unmeasured;
//! - a provoking vertex mode other than 0: SPI outside the public SDK, measured (mode 1 draws a
//!   flat varying from a primitive's last vertex) but not built (no `VK_EXT_provoking_vertex`
//!   path);
//! - turning the viewport transform off, which Vulkan has no switch for and whose meaning on
//!   Metal is not measured;
//! - a texture over an IOSurface made with `kIOSurfaceClientDirection`, a key in no header of the
//!   public macOS SDK, refused by the IOSurface plane's stand-in
//!   (`Buffer::new_texture_with_iosurface_plane`), which is no call of the table;
//! - the texture fills' cases whose answer is unmeasured (`blit_fill.rs`): a fill of a texture
//!   type, region, pattern or two-plane level the probes did not cover. A color fill packed in a
//!   format with no color packing, where Metal's bytes differ between processes, is built: one of
//!   Metal's answers, announced as `unpacked-fill-color`.
//!
//! [`AppleOnlyCall`] is the table, and every call of it is here: the command buffer's, the
//! compute, render and blit encoders', and the texture's protection options. A texture's
//! rotation is the runtime's (`TextureDescriptor::rotation`), and command predication, loops
//! included, is `predication.rs`'s.

use super::blit::BlitCommandEncoder;
use super::blit_texture::check_slice_level;
use super::command_buffer::CommandBuffer;
use super::compute::ComputeCommandEncoder;
use super::error::Error;
use super::render::RenderCall;
use super::render::RenderCommandEncoder;
use super::replay::BlitCall;
use super::texture::Texture;

/// An Apple-only feature the runtime refuses, each for the reason [`AppleOnly::gap`] gives (No
/// device lacks any of them; each is a gap of the runtime's own).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AppleOnly {
    /// Protected content: nonzero `setProtectionOptions:`.
    ProtectedContent,
    /// A line width other than 1 (`setLineWidth:`).
    LineWidth,
    /// `setProvokingVertexMode:` other than 0: SPI, outside the public SDK, and outside the
    /// runtime's scope (no `VK_EXT_provoking_vertex` path). Mode 1 draws a flat varying from a
    /// primitive's last vertex (a Metal probe, an M4).
    ProvokingVertexMode,
    /// The viewport transform turned off (`setViewportTransformEnabled:NO`).
    ViewportTransformDisabled,
    /// `setVertexAmplificationMode:value:`.
    VertexAmplificationMode,
    /// `setDepthCleared`.
    DepthCleared,
    /// `setStencilCleared`.
    StencilCleared,
    /// A texture over an IOSurface made with `kIOSurfaceClientDirection`
    /// (`newTextureWithDescriptor:iosurface:plane:` of such a surface): IOSurface SPI, a key
    /// IOSurface.framework exports and no header of the public macOS SDK declares, so the
    /// surface is the consumer's to build. The consumer reports that Metal drops a GPU write into
    /// a surface made with value 2 (read-only), without a fault; the runtime does not model the
    /// key's values.
    SurfaceClientDirection,
    /// A texture fill (`fillTexture:level:slice:region:bytes:length:` or a color form) whose
    /// answer on Apple's GPU is unmeasured: a multisample texture or a texture buffer, an empty
    /// pattern, a format outside the measured families (a color packed in a format the probes
    /// did not fill, such as the private 10-bit, PQ and HLG formats), and a two-plane
    /// extended-range texture whose fill Metal writes where the runtime holds no bytes (one of its
    /// own memory at a width that is not a multiple of 4, where Metal's blit rows move texels
    /// between rows; a heap's texture other than a one-level 2D one) or whose bytes Metal's blits
    /// read back no rule fits (one of its own of more than one level).
    /// 1D, 3D, cube and array textures and regions of several depth images are measured and built.
    FillUnmeasured,
}

/// Why the runtime refuses an [`AppleOnly`] feature. None is a device's gap: no measured Vulkan
/// device lacks what an exact form would need. Each is the runtime's own, and a measurement or a
/// build can lift it. A color fill whose bytes Metal leaves undefined, once refused as having no
/// one answer, is built: the runtime gives one of Metal's answers.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AppleOnlyGap {
    /// What Metal does on Apple's GPU is not measured, so the runtime does not guess it: the
    /// runtime's gap, which a measurement would let it build.
    Unmeasured,
    /// Metal's answer is measured, and Vulkan has a form for it, but the runtime has not built it.
    NotBuilt,
}

impl AppleOnly {
    /// Why the runtime refuses the feature ([`AppleOnlyGap`]).
    pub const fn gap(self) -> AppleOnlyGap {
        match self {
            Self::ProvokingVertexMode => AppleOnlyGap::NotBuilt,
            Self::ProtectedContent
            | Self::LineWidth
            | Self::ViewportTransformDisabled
            | Self::VertexAmplificationMode
            | Self::DepthCleared
            | Self::StencilCleared
            | Self::SurfaceClientDirection
            | Self::FillUnmeasured => AppleOnlyGap::Unmeasured,
        }
    }
}

/// Metal's `protectionOptions` of a resource made without any, and of every runtime texture.
pub const DEFAULT_PROTECTION_OPTIONS: u64 = 0;

/// One Apple-only call (class A) with the arguments its answer depends on: the table each
/// encoder's, descriptor's and command buffer's method of that call answers by.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AppleOnlyCall {
    /// `invalidateCompressedTexture:` (blit encoder).
    InvalidateCompressedTexture,
    /// `invalidateCompressedTexture:slice:level:` (blit encoder).
    InvalidateCompressedTextureSlice,
    /// `insertCompressedTextureReinterpretationFlush` (compute encoder).
    CompressedTextureReinterpretationFlush,
    /// `setProtectionOptions:` (command buffer).
    SetProtectionOptions(u64),
    /// `setAlphaTestReferenceValue:` (render encoder).
    SetAlphaTestReferenceValue(f32),
    /// `setLineWidth:` (render encoder).
    SetLineWidth(f32),
    /// `setPointSize:` (render encoder).
    SetPointSize(f32),
    /// `setProvokingVertexMode:` (render encoder).
    SetProvokingVertexMode(u64),
    /// `setViewportTransformEnabled:` (render encoder).
    SetViewportTransformEnabled(bool),
    /// `setVertexAmplificationMode:value:` (render encoder).
    SetVertexAmplificationMode {
        /// The mode given.
        mode: u64,
        /// The value given.
        value: u64,
    },
    /// `setDepthCleared` (render encoder).
    SetDepthCleared,
    /// `setStencilCleared` (render encoder).
    SetStencilCleared,
}

impl AppleOnlyCall {
    /// Metal's selector.
    pub fn selector(self) -> &'static str {
        match self {
            AppleOnlyCall::InvalidateCompressedTexture => "invalidateCompressedTexture:",
            AppleOnlyCall::InvalidateCompressedTextureSlice => {
                "invalidateCompressedTexture:slice:level:"
            }
            AppleOnlyCall::CompressedTextureReinterpretationFlush => {
                "insertCompressedTextureReinterpretationFlush"
            }
            AppleOnlyCall::SetProtectionOptions(_) => "setProtectionOptions:",
            AppleOnlyCall::SetAlphaTestReferenceValue(_) => "setAlphaTestReferenceValue:",
            AppleOnlyCall::SetLineWidth(_) => "setLineWidth:",
            AppleOnlyCall::SetPointSize(_) => "setPointSize:",
            AppleOnlyCall::SetProvokingVertexMode(_) => "setProvokingVertexMode:",
            AppleOnlyCall::SetViewportTransformEnabled(_) => "setViewportTransformEnabled:",
            AppleOnlyCall::SetVertexAmplificationMode { .. } => "setVertexAmplificationMode:value:",
            AppleOnlyCall::SetDepthCleared => "setDepthCleared",
            AppleOnlyCall::SetStencilCleared => "setStencilCleared",
        }
    }

    /// The runtime's answer: `Ok` where the call is exact on Vulkan (it states Metal's default,
    /// or acts on state no runtime object has, so it records nothing), else the refusal naming
    /// the missing feature.
    pub fn check(self) -> Result<(), AppleOnly> {
        match self {
            AppleOnlyCall::InvalidateCompressedTexture
            | AppleOnlyCall::InvalidateCompressedTextureSlice
            | AppleOnlyCall::CompressedTextureReinterpretationFlush
            | AppleOnlyCall::SetAlphaTestReferenceValue(_)
            | AppleOnlyCall::SetPointSize(_) => Ok(()),
            AppleOnlyCall::SetProtectionOptions(options) => default_or(
                options == DEFAULT_PROTECTION_OPTIONS,
                AppleOnly::ProtectedContent,
            ),
            AppleOnlyCall::SetLineWidth(width) => default_or(width == 1.0, AppleOnly::LineWidth),
            AppleOnlyCall::SetViewportTransformEnabled(enabled) => {
                default_or(enabled, AppleOnly::ViewportTransformDisabled)
            }
            AppleOnlyCall::SetProvokingVertexMode(mode) => {
                default_or(mode == 0, AppleOnly::ProvokingVertexMode)
            }
            AppleOnlyCall::SetVertexAmplificationMode { .. } => {
                Err(AppleOnly::VertexAmplificationMode)
            }
            AppleOnlyCall::SetDepthCleared => Err(AppleOnly::DepthCleared),
            AppleOnlyCall::SetStencilCleared => Err(AppleOnly::StencilCleared),
        }
    }
}

/// `Ok` where the call states Metal's default, else the refusal of `feature`.
fn default_or(is_default: bool, feature: AppleOnly) -> Result<(), AppleOnly> {
    if is_default {
        Ok(())
    } else {
        Err(feature)
    }
}

/// The runtime error of `call`'s answer.
fn answer(call: AppleOnlyCall) -> Result<(), Error> {
    call.check().map_err(Error::AppleOnly)
}

impl CommandBuffer {
    /// `setProtectionOptions:` (SPI): accepted at Metal's default, no protection options
    /// ([`DEFAULT_PROTECTION_OPTIONS`]); any other value is refused
    /// ([`AppleOnly::ProtectedContent`]): an SPI's feature is the consumer's to build.
    pub fn set_protection_options(&self, options: u64) -> Result<(), Error> {
        answer(AppleOnlyCall::SetProtectionOptions(options))
    }
}

impl Texture {
    /// `protectionOptions` (SPI): [`DEFAULT_PROTECTION_OPTIONS`], the only protection options a
    /// runtime texture has.
    pub fn protection_options(&self) -> u64 {
        DEFAULT_PROTECTION_OPTIONS
    }
}

impl BlitCommandEncoder {
    /// `invalidateCompressedTexture:` (SPI): a no-op. Vulkan shows no texture compression, and
    /// after an invalidation Metal leaves the contents undefined, so keeping them is an outcome
    /// Metal allows. Refused (`Invalid`, `ForeignObject`) for another device's texture, as every
    /// blit is.
    pub fn invalidate_compressed_texture(&mut self, texture: &Texture) -> Result<(), Error> {
        self.log(|_| BlitCall::InvalidateCompressedTexture(texture.clone()));
        self.core()
            .check_device("invalidate_compressed_texture", texture.device())?;
        answer(AppleOnlyCall::InvalidateCompressedTexture)
    }

    /// `invalidateCompressedTexture:slice:level:` (SPI): a no-op, as
    /// [`invalidate_compressed_texture`](Self::invalidate_compressed_texture) is, for one slice
    /// and level. Refused (`Invalid`) also for a slice or level the texture does not have, as the
    /// blit encoder's hints of a slice and level are.
    pub fn invalidate_compressed_texture_slice(
        &mut self,
        texture: &Texture,
        slice: u64,
        level: u64,
    ) -> Result<(), Error> {
        self.log(|_| BlitCall::InvalidateCompressedTextureSlice(texture.clone(), slice, level));
        const CALL: &str = "invalidate_compressed_texture_slice";
        self.core().check_device(CALL, texture.device())?;
        check_slice_level(CALL, texture, slice, level)?;
        answer(AppleOnlyCall::InvalidateCompressedTextureSlice)
    }
}

impl ComputeCommandEncoder {
    /// `insertCompressedTextureReinterpretationFlush` (SPI): a no-op. Vulkan shows no texture
    /// compression, so there is nothing to flush.
    pub fn insert_compressed_texture_reinterpretation_flush(&mut self) {}
}

impl RenderCommandEncoder {
    /// `setAlphaTestReferenceValue:` (SPI): a no-op. No pipeline the runtime builds has a
    /// fixed-function alpha test, so the reference has no effect, as on Metal with the test off.
    pub fn set_alpha_test_reference_value(&mut self, value: f32) -> Result<(), Error> {
        self.log(|_| RenderCall::AlphaTestReferenceValue(value));
        answer(AppleOnlyCall::SetAlphaTestReferenceValue(value))
    }

    /// `setLineWidth:` (SPI): accepted at Metal's default width of 1, which every line the
    /// runtime draws has; any other width is refused ([`AppleOnly::LineWidth`]): Apple's
    /// wide-line rule is unmeasured.
    pub fn set_line_width(&mut self, width: f32) -> Result<(), Error> {
        self.log(|_| RenderCall::LineWidth(width));
        answer(AppleOnlyCall::SetLineWidth(width))
    }

    /// `setPointSize:` (SPI): a no-op. On an M4 it changes nothing Metal draws at any size (0,
    /// 0.5, 1, 3, 5, 100, -2, NaN, infinity), whether the vertex function writes `[[point_size]]`
    /// or not and whether the pipeline's private `pointSizeOutputVS` is set.
    pub fn set_point_size(&mut self, size: f32) -> Result<(), Error> {
        self.log(|_| RenderCall::PointSize(size));
        answer(AppleOnlyCall::SetPointSize(size))
    }

    /// `setProvokingVertexMode:` (SPI): accepted at Metal's default, mode 0, the first vertex,
    /// from which every runtime draw takes a flat varying; any other mode is refused
    /// ([`AppleOnly::ProvokingVertexMode`]).
    pub fn set_provoking_vertex_mode(&mut self, mode: u64) -> Result<(), Error> {
        self.log(|_| RenderCall::ProvokingVertexMode(mode));
        answer(AppleOnlyCall::SetProvokingVertexMode(mode))
    }

    /// `setViewportTransformEnabled:` (SPI): accepted on, Metal's default and every runtime
    /// draw's; off is refused ([`AppleOnly::ViewportTransformDisabled`]): Vulkan has no form for
    /// it.
    pub fn set_viewport_transform_enabled(&mut self, enabled: bool) -> Result<(), Error> {
        self.log(|_| RenderCall::ViewportTransformEnabled(enabled));
        answer(AppleOnlyCall::SetViewportTransformEnabled(enabled))
    }

    /// `setVertexAmplificationMode:value:` (SPI): refused
    /// ([`AppleOnly::VertexAmplificationMode`]), its meaning unmeasured.
    pub fn set_vertex_amplification_mode(&mut self, mode: u64, value: u64) -> Result<(), Error> {
        self.log(|_| RenderCall::VertexAmplificationMode(mode, value));
        answer(AppleOnlyCall::SetVertexAmplificationMode { mode, value })
    }

    /// `setDepthCleared` (SPI): refused ([`AppleOnly::DepthCleared`]), its meaning unmeasured.
    pub fn set_depth_cleared(&mut self) -> Result<(), Error> {
        self.log(|_| RenderCall::DepthCleared);
        answer(AppleOnlyCall::SetDepthCleared)
    }

    /// `setStencilCleared` (SPI): refused ([`AppleOnly::StencilCleared`]), its meaning
    /// unmeasured.
    pub fn set_stencil_cleared(&mut self) -> Result<(), Error> {
        self.log(|_| RenderCall::StencilCleared);
        answer(AppleOnlyCall::SetStencilCleared)
    }
}
