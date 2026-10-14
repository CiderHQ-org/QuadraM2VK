//! `DevicePolicy`: the consumer's product choices that, together with a `DeviceDescription`,
//! decide a `DeviceProfile`.

use crate::device::profile::{FramebufferFetchPolicy, GuestIsolation, HandleDomain, PointerDomain};

/// Policy inputs of `Device::new`. Every field but `padded_reads` can change emitted code, so the
/// profile carries it (containment as the encoding it selects) and its fingerprint covers it;
/// `padded_reads` is the runtime's alone (how it makes a texture over a buffer), and no profile
/// carries it.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DevicePolicy {
    /// How pointer values resolve to resources (v1 ships `ResidentTable` only).
    pub pointer_domain: PointerDomain,
    /// How resource handles resolve (v1 ships `Table` only).
    pub handle_domain: HandleDomain,
    /// Whether accesses are contained to the resources the binder supplied.
    pub containment: ContainmentPolicy,
    /// Whether guest shaders are isolated from each other's resources.
    pub isolation: GuestIsolation,
    /// Framebuffer-fetch ordering without rasterization-order attachment access.
    pub framebuffer_fetch: FramebufferFetchPolicy,
    /// Whether an 8-bit unsigned normalized texture's linear sample may be the device's own
    /// filter on a device measured within one code of Metal; by default it is Metal's filter,
    /// emulated exactly.
    pub unorm8_linear_filter: Unorm8LinearFilterPolicy,
    /// Whether a device that does not sample through Metal's own sampler samples natively (the
    /// default) or emulates Metal's sampler exactly.
    pub sampling: SamplingPolicy,
    /// Whether the math library's functions (`sin`, `cos`, `exp`, `log`, `pow` and the rest), a
    /// precise division and Metal's fast reciprocal, and a fused multiply-add are the device's
    /// own instructions (native device maths, the default) or the translation's routines of
    /// Metal's answers.
    pub float_maths: FloatMathsPolicy,
    /// Whether a texel's `float` or `half` value is the device's conversion of its codes (native
    /// device sampling, the default off Apple's GPUs) or Metal's, converted exactly in the
    /// shader.
    pub texel_conversion: TexelConversionPolicy,
    /// Whether a `half` result computed in `float` rounds to `half` through the device's own
    /// conversion to 16 bits (or its `OpQuantizeToF16`) where neither is measured to be Metal's
    /// (native half rounding, the default), or by the translation's integer rounding, Metal's on
    /// every device.
    pub half_rounding: HalfRoundingPolicy,
    /// Whether a NaN result's bits, and the operand Metal's minimum and maximum functions
    /// (`fmin`, `fmax`, `min3`, `max3`, `clamp`, `saturate`) pick at a NaN or a signed zero, are
    /// the device's own (the default) or Metal's.
    pub nan_and_min_max: NanMinMaxPolicy,
    /// Whether Metal's `dot`, `mix` and `fast::sqrt` round in Metal's order (its fused chain of
    /// the products, its two fused multiply-adds, its `x · rsqrt(x)`) or are the device's single
    /// instructions (the default).
    pub rounding_order: RoundingOrderPolicy,
    /// Whether a `half4` color output of a module without fast math blends into a 16-bit float
    /// attachment in the device's blend (the default) or in Metal's `half` arithmetic, which the
    /// function computes in the shader.
    pub half_blend: HalfBlendPolicy,
    /// Whether a texture over a buffer whose rows (or slices) lie with bytes between them, and
    /// which only reads (its usage holds no `RenderTarget`, `ShaderWrite`, `ShaderAtomic` or block
    /// writes), is an image over the buffer's bytes at Metal's row pitch (the default) or the
    /// runtime's shadow of them, copied in before each use. The runtime's choice alone: no
    /// emitted code changes.
    pub padded_reads: PaddedReadsPolicy,
}

impl DevicePolicy {
    /// The untrusted-guest production policy: containment and isolation on,
    /// ordered framebuffer fetch required, Metal's 8-bit linear filter emulated exactly where the
    /// emulation runs, native device sampling, native device maths, the device's own texel
    /// conversions, native half rounding, the device's NaN and minimum and maximum, the device's
    /// rounding order, the device's blend of a `half` output into a 16-bit float attachment, and
    /// an image over a texture's padded rows where the texture only reads.
    pub const UNTRUSTED: DevicePolicy = DevicePolicy {
        pointer_domain: PointerDomain::ResidentTable,
        handle_domain: HandleDomain::Table,
        containment: ContainmentPolicy::Contained,
        isolation: GuestIsolation::On,
        framebuffer_fetch: FramebufferFetchPolicy::RequireOrdered,
        unorm8_linear_filter: Unorm8LinearFilterPolicy::Exact,
        sampling: SamplingPolicy::NativeDevice,
        float_maths: FloatMathsPolicy::Native,
        texel_conversion: TexelConversionPolicy::Device,
        half_rounding: HalfRoundingPolicy::Native,
        nan_and_min_max: NanMinMaxPolicy::Native,
        rounding_order: RoundingOrderPolicy::Native,
        half_blend: HalfBlendPolicy::Device,
        padded_reads: PaddedReadsPolicy::Alias,
    };

    /// This policy with Metal's answer wherever the choices offer it: its maths functions,
    /// divisions and fused multiply-adds ([`FloatMathsPolicy::Exact`]), its texel conversions
    /// ([`TexelConversionPolicy::Exact`]), its `half` rounding ([`HalfRoundingPolicy::Exact`]), its
    /// NaN and minimum and maximum ([`NanMinMaxPolicy::Exact`]), its rounding order
    /// ([`RoundingOrderPolicy::Exact`]), its 16-bit float blend of a `half` output
    /// ([`HalfBlendPolicy::Exact`]), and the bytes between a texture's rows that no first move of
    /// an image over them reaches ([`PaddedReadsPolicy::Shadow`]). Sampling and every other field
    /// are kept.
    pub const fn exact_answers(self) -> DevicePolicy {
        DevicePolicy {
            float_maths: FloatMathsPolicy::Exact,
            texel_conversion: TexelConversionPolicy::Exact,
            half_rounding: HalfRoundingPolicy::Exact,
            nan_and_min_max: NanMinMaxPolicy::Exact,
            rounding_order: RoundingOrderPolicy::Exact,
            half_blend: HalfBlendPolicy::Exact,
            padded_reads: PaddedReadsPolicy::Shadow,
            ..self
        }
    }
}

impl Default for DevicePolicy {
    /// [`DevicePolicy::UNTRUSTED`].
    fn default() -> DevicePolicy {
        DevicePolicy::UNTRUSTED
    }
}

/// The containment choice. The encoding (robustness or in-shader checks) is selected by
/// `Device::new` from the enabled robustness features; the policy only says whether to contain.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ContainmentPolicy {
    /// Every access stays inside the resources the binder supplied (the default).
    Contained,
    /// No containment. Only valid with `GuestIsolation::Off`; recorded in the fingerprint.
    UnsafeOff,
}

/// How an 8-bit unsigned normalized texture's linear sample is filtered: a 2D texture's or 2D
/// array's sample of a slot bound in `R8Unorm`, `RG8Unorm`, `RGBA8Unorm` or `BGRA8Unorm`, through a
/// constexpr sampler with normalized coordinates or a runtime sampler slot, without comparison or
/// offset.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Unorm8LinearFilterPolicy {
    /// Metal's filter, emulated exactly: the default (the owner's ruling of 2026-10-11; on the
    /// consumer's fragments the device's filter bought no GPU time and cost module size and
    /// making time).
    #[default]
    Exact,
    /// The device's own filter at the centre of Metal's 1/256 footprint step, on a device
    /// measured to filter the class within one code (1/255) of Metal
    /// ([`DriverDeviation::Unorm8LinearFilter`](crate::error::DriverDeviation::Unorm8LinearFilter)):
    /// such a sample records
    /// [`PrecisionNote::DeviceUnorm8LinearFilter`](crate::pipeline::PrecisionNote::DeviceUnorm8LinearFilter)
    /// and requires `VkProperty::Unorm8LinearFilter`. An unmeasured device keeps Metal's filter.
    DeviceWithinOneCode,
}

/// How a device that does not sample through Metal's own sampler (MoltenVK on Apple's GPU,
/// `DriverDeviation::MetalSampler`) answers a module's samples, gathers and comparisons: natively,
/// its own sampler's answer (the default: the owner's ruling of 2026-10-11, making time before
/// exactness), or Metal's sampler, emulated exactly. A device that samples through Metal's own
/// sampler answers as Metal under either.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SamplingPolicy {
    /// Native device sampling: each sample, gather and comparison goes to the device's sampler
    /// as on a device whose sampler is Metal's, with no emulation of Metal's filters, footprint,
    /// level selection, anisotropy or comparisons in the module. Such a module records
    /// [`PrecisionNote::DeviceSampling`](crate::pipeline::PrecisionNote::DeviceSampling), whose
    /// docs state the bound measured against Metal, and requires `VkProperty::DeviceSampling`.
    #[default]
    NativeDevice,
    /// Metal's sampler, emulated exactly where the device's differs: the module's size and the
    /// driver's making time grow with each sample.
    Exact,
}

/// How the `float` and `half` maths functions, divisions and fused multiply-adds Metal answers
/// beyond Vulkan's precision table are computed on a device whose functions are not Metal's own
/// (the owner's "Maths functions" of 2026-10-11, native by default). On a device whose functions
/// are Metal's (MoltenVK on an Apple GPU, `DriverDeviation::MetalFloatFunctions`), the device's
/// instructions are Metal's own whatever the policy. A NaN such a function answers is
/// [`NanMinMaxPolicy`]'s, `dot`, `mix` and `fast::sqrt` are [`RoundingOrderPolicy`]'s, and a `half`
/// result's rounding is [`HalfRoundingPolicy`]'s.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FloatMathsPolicy {
    /// Native device maths (the default): the math library's functions are the device's
    /// `GLSL.std.450` instructions (or Metal's compositions of them), a precise division and
    /// Metal's fast reciprocal the device's `OpFDiv`, and a fused multiply-add without
    /// `OpFmaKHR` the device's `Fma`, each within Vulkan's precision table rather than Metal's
    /// answer, and each recording its kind
    /// ([`PrecisionNote::NativeTranscendentals`](crate::pipeline::PrecisionNote::NativeTranscendentals),
    /// [`NativeDivision`](crate::pipeline::PrecisionNote::NativeDivision),
    /// [`NativeFma`](crate::pipeline::PrecisionNote::NativeFma)) with the bound in its docs.
    #[default]
    Native,
    /// The translation's routines, which answer as Metal does on every device: larger modules and
    /// longer pipeline creation.
    Exact,
}

/// How a texel's value reaches a read, sample or gather with a `float` or `half` result on a
/// device that does not sample through Metal's own texture unit (`VkProperty::MetalSampler`). On
/// such a device Metal's conversion of a texel is emulated by a routine that tells the bound
/// format's class apart at every access, and the class is data, so no driver folds it: on the
/// consumer's fragments those routines were most of the driver's making time (EV SRT-nm1: 296 calls
/// in `Xgc`, 311 in `A2Xghfc`).
///
/// The choice is made per access. A read's texel is this policy's. A sample's or gather's is too
/// where Metal's sampler is emulated ([`SamplingPolicy::Exact`]); under native device sampling
/// ([`SamplingPolicy::NativeDevice`], the module requiring `VkProperty::DeviceSampling`) it is the
/// device's under either policy: the device's sampler decodes its samples and gathers, and an exact
/// conversion of a sample's texel reads the runtime sampler's filter to tell one texel from a
/// filtered sum, which splits the pipeline by the bound sampler state. Measured on the RTX 5080
/// Laptop GPU, `Exact`'s reads cost the consumer's `Xgc` 3 to 5% of its making and 4 to 18% of its
/// GPU time per draw.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TexelConversionPolicy {
    /// Native device sampling, the default (the owner's ruling of 2026-10-11: making time before
    /// exactness off Apple's GPUs, with the bound recorded and announced): the device's `float`
    /// value of each texel, and its `half` rounded to nearest even. Such an access records
    /// [`PrecisionNote::DeviceTexelConversion`](crate::pipeline::PrecisionNote::DeviceTexelConversion)
    /// (`float`) or
    /// [`PrecisionNote::HalfTexelConversion`](crate::pipeline::PrecisionNote::HalfTexelConversion)
    /// (`half`), whose docs state the bound.
    #[default]
    Device,
    /// Metal's conversion, emulated exactly by the bound format's class, of every read, and of
    /// every sample and gather where Metal's sampler is emulated: what validation compares against
    /// Metal's answers.
    Exact,
}

/// How a `half` result the module computes in `float` (every `half` arithmetic but the device's
/// 16-bit arithmetic and `Fma` where they are measured to be Metal's) is rounded to `half` on a
/// device whose conversion to 16 bits and whose `OpQuantizeToF16` are not measured to round as
/// Metal's `half(float)` does (the owner's "tiny half-precision values flushed to zero on some
/// GPUs" of 2026-10-11, which the device's conversion may do besides its rounding, native by
/// default). Where either is measured to be Metal's (`DriverDeviation::MetalHalfConversion`,
/// MoltenVK on an Apple GPU; `DriverDeviation::MetalHalfQuantize`, MoltenVK, RADV and ANV where
/// measured) the module rounds through it under either policy.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum HalfRoundingPolicy {
    /// Native half rounding (the default): the device's conversion of the `float` to 16 bits
    /// and its widening where the profile has `shaderFloat16`, else the device's
    /// `OpQuantizeToF16` with a result below the smallest normal `half` rounded exactly. Such a
    /// module records
    /// [`PrecisionNote::NativeHalfConversion`](crate::pipeline::PrecisionNote::NativeHalfConversion)
    /// or
    /// [`PrecisionNote::NativeHalfQuantize`](crate::pipeline::PrecisionNote::NativeHalfQuantize),
    /// whose docs state the bound, and requires no measured fact.
    #[default]
    Native,
    /// Metal's rounding on every device: integer arithmetic on the `float`'s bits, about twenty
    /// instructions a rounding, where the native forms take three: larger modules and longer
    /// pipeline creation.
    Exact,
}

/// Whose NaN a `float` or `half` result answers, and which operand Metal's minimum and maximum
/// functions (`fmin`, `fmax`, `min3`, `max3`, `clamp`, and `saturate`, Metal's
/// `clamp(x, +0, 1)`) pick at a NaN, a signed zero or an `f32` denormal, on a device not measured
/// to answer Metal's (the owner's "the bit pattern of NaN values, and min and max when NaN or
/// signed zero is involved" of 2026-10-11). A device measured to answer Metal's canonical NaN
/// (`DriverDeviation::MetalCanonicalNans`) keeps it, and one measured to compute Metal's minimum
/// and maximum (`DriverDeviation::MetalMinMax`) or whose functions are Metal's own
/// (`DriverDeviation::MetalFloatFunctions`, MoltenVK on an Apple GPU) keeps Metal's picks, under
/// either setting.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum NanMinMaxPolicy {
    /// The device's own (the default): every NaN Metal answers is the device's (its sign and
    /// payload; no select of Metal's bits), and those functions are the device's `NMin`, `NMax`
    /// and `NClamp`. Such a module records
    /// [`PrecisionNote::NativeNanEncoding`](crate::pipeline::PrecisionNote::NativeNanEncoding)
    /// or [`NativeMinMax`](crate::pipeline::PrecisionNote::NativeMinMax), whose docs state the
    /// bound.
    #[default]
    Native,
    /// Metal's: the translation selects Metal's NaN where the device may answer another, and
    /// Metal's operand where the device's instructions may pick another. A `half` texel read,
    /// sampled or gathered keeps the NaN the device decoded through native half rounding too
    /// (`HalfRoundingPolicy::Native`), where the default takes the rounding's.
    Exact,
}

/// The order in which Metal's `dot`, `mix` and `fast::sqrt` on `float` and `half` round, on a
/// device whose functions are not Metal's own (the owner's "rounding order in dot products, blends
/// between two values (mix) and fast square root" of 2026-10-11). On a device whose functions are
/// Metal's (MoltenVK on an Apple GPU, `DriverDeviation::MetalFloatFunctions`) they are Metal's
/// under either setting.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum RoundingOrderPolicy {
    /// The device's single instructions (the default): `dot` the device's `OpDot`, `mix` its
    /// `FMix` (a `half` result summed in `float` and rounded once), and `fast::sqrt` its `Sqrt`.
    /// Such a module records
    /// [`PrecisionNote::NativeDot`](crate::pipeline::PrecisionNote::NativeDot),
    /// [`NativeMix`](crate::pipeline::PrecisionNote::NativeMix) or
    /// [`NativeTranscendentals`](crate::pipeline::PrecisionNote::NativeTranscendentals)
    /// (`fast::sqrt`, with `DeviceDenormals`), whose docs state the bound.
    #[default]
    Native,
    /// Metal's order: `dot` the fused chain of its products, `mix` `fma(t, y, fma(-t, x, x))`, and
    /// `fast::sqrt` `x · rsqrt(x)` with Metal's answers at a zero, a denormal and `+inf`.
    Exact,
}

/// How a `half4` color output of a module compiled without fast math
/// (`air.compile.fast_math_disable`, safe or relaxed math) blends into an `R16Float`, `RG16Float`
/// or `RGBA16Float` attachment. Metal blends it in `half` arithmetic: the `half` the function
/// returns, each factor (`1 - x` and the blend color's `half` included), each product and the sum
/// rounded to `half`, nearest even (EV FLT-hb2-m3); a device's own blend adds in its arithmetic and
/// rounds once.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum HalfBlendPolicy {
    /// The device's blend (the default): a blended channel may store a value apart from Metal's
    /// within the bound
    /// [`PrecisionNote::HalfOutputPrecision`](crate::pipeline::PrecisionNote::HalfOutputPrecision)
    /// records, which the pipeline then records.
    #[default]
    Device,
    /// Metal's blend, computed by the function in the interlocked storage form of the attachment
    /// (`lower::fragment::fetch`) on a device that orders framebuffer fetch through the pixel
    /// interlock: the attachment becomes a storage image the function reads and writes, so its
    /// image needs `STORAGE` usage (a pass renders a texture without it into a stand-in) and a
    /// pass splits where its draws change the attachment's role. Where that form is not exact
    /// for the pipeline (a device without the interlock, a depth or stencil test the write would
    /// precede, alpha to coverage, multiview, a format the device does not store, a blend of the
    /// second source), the device blends and the pipeline records the note as under
    /// [`HalfBlendPolicy::Device`].
    Exact,
}

/// How the runtime makes a texture over a buffer whose rows (or an array's slices) lie with bytes
/// between them that are not its texels (a guest browser's padded surfaces), and which only reads:
/// its usage, as Metal reports it (`Unknown` reads as `ShaderRead | ShaderWrite | RenderTarget |
/// PixelFormatView`), holds no `RenderTarget`, `ShaderWrite`, `ShaderAtomic` or block writes.
/// Metal never writes those bytes. A render target, or a texture a shader writes, is a shadow
/// under either value: a render pass's load and store operations may write back any byte of an
/// attachment's subresource.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PaddedReadsPolicy {
    /// An image over the buffer's bytes at Metal's row pitch, as a texture whose rows lie one
    /// after another has (the default): the device's own linear image, a `DRM_FORMAT_MOD_LINEAR`
    /// image at that pitch, or one over a dma-buf of the pages, holding the bytes between the
    /// rows in its subresource. After its first move every command uses it in `GENERAL`, and
    /// copies and shader reads touch its texels alone; the first move, a layout transition, "may
    /// perform read and write accesses on all memory bound to the image subresource range"
    /// (Vulkan's Image Layout Transitions), and from `UNDEFINED` may discard it. So the bytes
    /// between the rows stay the program's only as far as the driver's move keeps them, which
    /// the validation lane checks per driver (docs/FAST-PATHS.md, its known gaps).
    #[default]
    Alias,
    /// The runtime's shadow image of the texels, copied in from the buffer before each use: the
    /// copies address the texels alone, so no byte between the rows is ever read or written
    /// through an image.
    Shadow,
}
