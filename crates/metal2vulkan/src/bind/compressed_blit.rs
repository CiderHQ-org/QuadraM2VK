//! The texels Metal's GPU blits write where a texture it keeps losslessly compressed is a side of
//! the blit.
//!
//! Metal keeps a texture losslessly compressed where `allowGPUOptimizedContents` holds, the
//! shorter side of its level 0 is 16 texels or more, and its usage has neither `ShaderWrite` nor
//! `PixelFormatView` (its depth and stencil formats at any usage). A GPU blit that reads or
//! writes such a texture (buffer into it, it into a buffer, texture into texture either way)
//! moves its texels through the compression, which keeps only the bits a format's texel holds.
//! For three families the bytes the blit writes differ from the bytes it read (an Apple M4 under
//! macOS 27.0.1, plain: random bytes, every direction, every level, slice and texture type, heap
//! textures and views; and an Apple M3 Max under 27.0.1, alike; each rule below matched every
//! texel):
//!
//! - `BGRA10_XR` and `BGRA10_XR_sRGB` ([`CompressedBlit::ExtendedRangeFields`]): each 16-bit
//!   field's low six bits cleared, and the alpha field's code clamped to `[384, 894]`;
//! - `RG11B10Float` ([`CompressedBlit::PackedFloatNan`]): a NaN field (exponent all ones,
//!   mantissa nonzero) made the quiet NaN, its mantissa's top bit alone; infinities kept;
//! - `RGB9E5Float` ([`CompressedBlit::SharedExponent`]): the shared exponent lowered while the
//!   largest mantissa's top bit is clear, each mantissa shifted up as far (to exponent 0 at
//!   most): the same value, at the least exponent that holds it.
//!
//! The 70 other color, depth and stencil formats the census tried keep their bytes, compressed
//! or not, the two-plane and `BGR10_XR` extended-range formats among them, and so does every
//! texture of an IOSurface or over a buffer. CPU copies (`replaceRegion`, `getBytes`) keep the
//! bytes whatever the texture.
//!
//! A consumer runs the copy, then rewrites what it wrote: the written texels copied into a
//! scratch buffer of words, one dispatch of [`compressed_blit_module`] of the rule, and the
//! words copied back.
//!
//! **The module** (one per device and rule). One `GLCompute` entry `main` with local size
//! ([`CompressedBlitModule::LOCAL_SIZE`], 1, 1), no push constants, and one storage buffer of
//! 4-byte words in set 0 at [`CompressedBlitModule::WORDS_BINDING`]: whole texels, tight, from a
//! texel's first byte. Invocation `i` below the binding's words rewrites word `i` in place, the
//! odd word of an 8-byte texel where `i` is odd; one past them writes nothing. A consumer whose
//! words exceed `maxStorageBufferRange` or `maxComputeWorkGroupCount[0]` workgroups binds them
//! in bands of whole texels, one dispatch each: no word reads another.
//!
//! **Synchronization** (consumer): the copy into the words (`TRANSFER_WRITE`) →
//! `SHADER_STORAGE_READ` and `SHADER_STORAGE_WRITE` in `COMPUTE_SHADER` before the dispatch; its
//! writes → `TRANSFER_READ` before the words are copied back.

use std::sync::Arc;

use crate::budget::{Budget, Meter};
use crate::device::requirements::Requirements;
use crate::device::Device;
use crate::error::{Error, InternalError, InternalKind};
use crate::metal::PixelFormat;
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage, Symbol};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, TypeId, Value};
use crate::spirv::ir::{
    natural_layout, BinOp, ExecMode, ExecModel, IoDir, LayoutRule, MemAccess, StorageClass, Type,
    TypedModule,
};

/// How Metal's GPU blit rewrites a texel of a format whose blits through its lossless
/// compression change bytes (the module docs).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CompressedBlit {
    /// `BGRA10_XR` and `BGRA10_XR_sRGB`: 8-byte texels of four 16-bit fields (blue, green, red,
    /// alpha), each field's low six bits cleared and alpha's 10-bit code clamped to `[384, 894]`.
    ExtendedRangeFields,
    /// `RG11B10Float`: a NaN field made the quiet NaN (red and green `0x7e0`, blue `0x3f0`).
    PackedFloatNan,
    /// `RGB9E5Float`: the shared exponent at the least value that holds the texel's value.
    SharedExponent,
}

impl CompressedBlit {
    /// The rule of `pf`'s texels, or `None` for a format whose blits keep their bytes.
    pub fn of(pf: PixelFormat) -> Option<CompressedBlit> {
        Some(match pf {
            PixelFormat::BGRA10XR | PixelFormat::BGRA10XRSRGB => {
                CompressedBlit::ExtendedRangeFields
            }
            PixelFormat::RG11B10Float => CompressedBlit::PackedFloatNan,
            PixelFormat::RGB9E5Float => CompressedBlit::SharedExponent,
            _ => return None,
        })
    }

    /// The bytes of one texel: 8 for the extended-range fields, else 4.
    pub fn texel_bytes(self) -> u32 {
        match self {
            CompressedBlit::ExtendedRangeFields => 8,
            CompressedBlit::PackedFloatNan | CompressedBlit::SharedExponent => 4,
        }
    }

    /// The word Metal's blit writes for `word`, the `odd` word of its texel (an 8-byte texel's
    /// second; a 4-byte texel's words are all even): the kernel's rule, on the CPU.
    pub fn word(self, word: u32, odd: bool) -> u32 {
        match self {
            CompressedBlit::ExtendedRangeFields if odd => {
                let alpha = (word >> 22).clamp(ALPHA_CODES[0], ALPHA_CODES[1]);
                (word & FIELD_MASK & 0xffff) | alpha << 22
            }
            CompressedBlit::ExtendedRangeFields => word & FIELD_MASK,
            CompressedBlit::PackedFloatNan => {
                PACKED_FIELDS
                    .iter()
                    .fold(word, |w, &(shift, mantissa_bits)| {
                        let field = (w >> shift) & field_mask(mantissa_bits);
                        let nan = field >> mantissa_bits == 31
                            && field & mantissa_mask(mantissa_bits) != 0;
                        match nan {
                            true => {
                                (w & !(field_mask(mantissa_bits) << shift))
                                    | quiet(mantissa_bits) << shift
                            }
                            false => w,
                        }
                    })
            }
            CompressedBlit::SharedExponent => {
                let exponent = word >> 27;
                let largest = (0..3).map(|i| (word >> (9 * i)) & 0x1ff).max().unwrap_or(0);
                let shift = match largest {
                    0 => exponent,
                    m => (m.leading_zeros() - 23).min(exponent),
                };
                let mantissas = (0..3).fold(0, |w, i| {
                    w | (((word >> (9 * i)) & 0x1ff) << shift) << (9 * i)
                });
                (exponent - shift) << 27 | mantissas
            }
        }
    }
}

/// The 16-bit fields' code bits, two fields a word.
const FIELD_MASK: u32 = 0xffc0_ffc0;
/// The alpha codes Metal's blit keeps.
const ALPHA_CODES: [u32; 2] = [384, 894];
/// `RG11B10Float`'s fields: their shift and mantissa bits (5 exponent bits each).
const PACKED_FIELDS: [(u32, u32); 3] = [(0, 6), (11, 6), (22, 5)];

/// The bits of a packed float field of `mantissa_bits` mantissa bits.
const fn field_mask(mantissa_bits: u32) -> u32 {
    (1 << (mantissa_bits + 5)) - 1
}

/// The mantissa bits of such a field.
const fn mantissa_mask(mantissa_bits: u32) -> u32 {
    (1 << mantissa_bits) - 1
}

/// The quiet NaN of such a field: exponent all ones, the mantissa's top bit alone.
const fn quiet(mantissa_bits: u32) -> u32 {
    31 << mantissa_bits | 1 << (mantissa_bits - 1)
}

/// The crate-owned compressed blit module of one device and rule, validated like any output.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompressedBlitModule {
    /// SPIR-V 1.5 words, entry point `main`, its binding numbered as below in set 0.
    pub spirv: Arc<[u32]>,
    /// What the module needs from the device (within every Vulkan 1.2 device).
    pub requirements: Requirements,
}

impl CompressedBlitModule {
    /// Descriptor set of the binding.
    pub const SET: u32 = 0;
    /// The texels' words, a storage buffer the dispatch rewrites in place.
    pub const WORDS_BINDING: u32 = 0;
    /// Invocations of one workgroup, along x: one 4-byte word each.
    pub const LOCAL_SIZE: u32 = 64;
}

/// The binding symbol of the words in the typed module.
const WORDS: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 0,
};

/// A `Block { u32 words[]; }` storage buffer bound at `sym`.
fn words_buffer(b: &mut ModuleBuilder, sym: ResourceSlot, u32_: TypeId) -> BResult<GlobalId> {
    let rta = b.ty(Type::RuntimeArray {
        elem: u32_,
        stride: Some(4),
    })?;
    let layout = natural_layout(b.types(), LayoutRule::Std430, true, &[rta])?;
    let block = b.ty(Type::Struct {
        members: [rta].into(),
        layout: Some(layout),
    })?;
    b.resource(sym, StorageClass::StorageBuffer, block)
}

fn u32_value(f: &mut FunctionBuilder<'_>, v: u32) -> BResult<Value> {
    let c = f.u32_const(v)?;
    f.module().const_value(c)
}

/// `(word >> shift) & mask`.
fn bits(f: &mut FunctionBuilder<'_>, word: Value, shift: u32, mask: u32) -> BResult<Value> {
    let shift = u32_value(f, shift)?;
    let shifted = f.binary(BinOp::ShiftRightLogical, word, shift)?;
    let mask = u32_value(f, mask)?;
    f.binary(BinOp::BitwiseAnd, shifted, mask)
}

/// `rule`'s rewrite of `word`, the `odd` (a `bool`) word of its texel ([`CompressedBlit::word`]).
fn rewritten(
    f: &mut FunctionBuilder<'_>,
    rule: CompressedBlit,
    word: Value,
    odd: Value,
) -> BResult<Value> {
    match rule {
        CompressedBlit::ExtendedRangeFields => {
            super::unused_bits::extended_range_fields(f, word, odd)
        }
        CompressedBlit::PackedFloatNan => {
            let mut w = word;
            let zero = u32_value(f, 0)?;
            for (shift, mantissa_bits) in PACKED_FIELDS {
                let field = bits(f, w, shift, field_mask(mantissa_bits))?;
                let exponent = bits(f, field, mantissa_bits, 31)?;
                let mantissa = bits(f, field, 0, mantissa_mask(mantissa_bits))?;
                let k31 = u32_value(f, 31)?;
                let all_ones = f.binary(BinOp::IEqual, exponent, k31)?;
                let nonzero = f.binary(BinOp::INotEqual, mantissa, zero)?;
                let nan = f.binary(BinOp::LogicalAnd, all_ones, nonzero)?;
                let kept = u32_value(f, !(field_mask(mantissa_bits) << shift))?;
                let others = f.binary(BinOp::BitwiseAnd, w, kept)?;
                let quiet = u32_value(f, quiet(mantissa_bits) << shift)?;
                let made = f.binary(BinOp::BitwiseOr, others, quiet)?;
                w = f.select(nan, made, w)?;
            }
            Ok(w)
        }
        CompressedBlit::SharedExponent => {
            let zero = u32_value(f, 0)?;
            let exponent = bits(f, word, 27, 31)?;
            let mut mantissas = Vec::with_capacity(3);
            for i in 0..3 {
                mantissas.push(bits(f, word, 9 * i, 0x1ff)?);
            }
            let mut largest = mantissas[0];
            for &m in &mantissas[1..] {
                let above = f.binary(BinOp::UGreaterThan, m, largest)?;
                largest = f.select(above, m, largest)?;
            }
            // The largest mantissa's leading zeros among its 9 bits: how many of 2^0 .. 2^8 it
            // lies below (9 for 0, which takes the whole exponent).
            let one = u32_value(f, 1)?;
            let mut zeros = zero;
            for k in 0..=8 {
                let bound = u32_value(f, 1 << k)?;
                let below = f.binary(BinOp::ULessThan, largest, bound)?;
                let add = f.select(below, one, zero)?;
                zeros = f.binary(BinOp::IAdd, zeros, add)?;
            }
            let none = f.binary(BinOp::IEqual, largest, zero)?;
            let fewer = f.binary(BinOp::ULessThan, zeros, exponent)?;
            let shift = f.select(fewer, zeros, exponent)?;
            let shift = f.select(none, exponent, shift)?;
            let lowered = f.binary(BinOp::ISub, exponent, shift)?;
            let k27 = u32_value(f, 27)?;
            let mut out = f.binary(BinOp::ShiftLeftLogical, lowered, k27)?;
            for (i, m) in mantissas.into_iter().enumerate() {
                let up = f.binary(BinOp::ShiftLeftLogical, m, shift)?;
                let at = u32_value(f, 9 * i as u32)?;
                let placed = f.binary(BinOp::ShiftLeftLogical, up, at)?;
                out = f.binary(BinOp::BitwiseOr, out, placed)?;
            }
            Ok(out)
        }
    }
}

/// The module as a typed module, before serialization: the module [`compressed_blit_module`]
/// serializes, which the CPU interpreter runs, of `rule`. Its binding is the symbol `WORDS`.
fn typed_module(rule: CompressedBlit) -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
    let words = words_buffer(&mut b, WORDS, u32_)?;
    let invocation = b.builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let zero = u32_value(&mut f, 0)?;
    let invocation = f.module().global_value(invocation)?;
    let invocation = f.load(invocation, MemAccess::default())?;
    let i = f.composite_extract(invocation, &[0])?;
    let words = f.module().global_value(words)?;
    let count = f.array_length(words, 0)?;
    let inside = f.binary(BinOp::ULessThan, i, count)?;
    let rewrite: ArmFn<'_, ()> = Box::new(move |f, _| {
        let k1 = u32_value(f, 1)?;
        let odd = f.binary(BinOp::BitwiseAnd, i, k1)?;
        let odd = f.binary(BinOp::INotEqual, odd, zero)?;
        let p = f.access_chain(words, &[zero, i])?;
        let word = f.load(p, MemAccess::default())?;
        let made = rewritten(f, rule, word, odd)?;
        f.store(p, made, MemAccess::default())?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    f.if_(&mut (), inside, rewrite, None)?;
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize {
            x: CompressedBlitModule::LOCAL_SIZE,
            y: 1,
            z: 1,
        }],
    )?;
    b.finalize()
}

/// The fixed binding numbers of the module's symbols.
fn binding_of(sym: &Symbol) -> Option<u32> {
    match sym {
        Symbol::DescriptorSet(_) => Some(CompressedBlitModule::SET),
        Symbol::Binding(s) if *s == WORDS => Some(CompressedBlitModule::WORDS_BINDING),
        _ => None,
    }
}

/// The compressed blit module of `device` for `rule`: built with the typed builder, checked by
/// the owned verifier against the device's profile (requirements included), serialized, and
/// numbered with the fixed binding of [`CompressedBlitModule`].
pub fn compressed_blit_module(
    device: &Device,
    rule: CompressedBlit,
) -> Result<CompressedBlitModule, Error> {
    let profile = device.profile();
    let m = typed_module(rule)?;
    let mut meter = Meter::new(&Budget::default());
    crate::spirv::verify::verify(&m, profile, &mut meter)?;
    let r = crate::spirv::serialize::serialize(m)?;
    profile.check(&r.requirements).map_err(|c| {
        Error::Internal(InternalError::new(
            InternalKind::RequirementsExceedProfile,
            format!("the compressed blit kernel needs {c:?}"),
        ))
    })?;
    let words = crate::spirv::serialize::relocate(&r, &binding_of)?;
    Ok(CompressedBlitModule {
        spirv: words.into(),
        requirements: r.requirements,
    })
}

#[cfg(test)]
mod tests;
