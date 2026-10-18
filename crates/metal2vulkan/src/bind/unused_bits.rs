//! An extended-range or padded 10-bit texture's words as Metal's tile store leaves them.
//!
//! A render pass on an Apple GPU stores its attachments a 32 x 32 tile at a time. A pass that
//! loads a color attachment of an extended-range format (`BGR10_XR`, `BGRA10_XR`,
//! `RGB10A8_2P_XR10`, their `_sRGB` twins) or a padded 10-bit one (`R10Unorm_X6`,
//! `RG10Unorm_X12`, their `_sRGB` twins) and records a draw writes every texel of each tile it
//! stores back, where a channel a fragment writes takes the write; every other texel keeps its
//! codes, the bits they leave unused made canonical (on an Apple M4 and the consumer's Apple M3
//! Max):
//!
//! - `BGR10_XR`'s word: bits 30 and 31 set;
//! - `BGRA10_XR`'s four 16-bit fields: each field's low six bits cleared, and the alpha field's
//!   code clamped to `[384, 894]`;
//! - the two-plane formats' color word: bits 30 and 31 cleared (the alpha byte kept);
//! - a padded 16-bit word: its low six bits cleared, an `_sRGB` twin's code through Metal's table
//!   (`ExtendedRange::load_code`: 23 codes of `R10Unorm_X6_sRGB` move, 1021 of
//!   `RG10Unorm_X12_sRGB`);
//! - a bfloat format's 16-bit code (`R16Bfloat`, `RG16Bfloat`, `RGBA16Bfloat`): its value through
//!   `half` to nearest even and back to the nearest bfloat code, ties to even, a NaN `0x7fc0`
//!   (`ExtendedRange::load_code`; on an Apple M4 under macOS 27.0.1: 56,829 of the
//!   65,536 codes move, in every tile a draw reaches).
//!
//! The same store writes a single-sampled signed normalized or `RGB9E5Float` attachment's tiles
//! back through their values (an Apple M3 Max under macOS 27.0.1), which a consumer's exact tile
//! store takes (`TileStorePolicy::Exact`):
//!
//! - a signed normalized format's code (`R8Snorm`, `RG8Snorm`, `RGBA8Snorm`, `R16Snorm`,
//!   `RG16Snorm`, `RGBA16Snorm`): a channel's minimum code (-128, or -32768), which reads as -1.0
//!   as the code above it does, raised to that code (-127, -32767);
//! - an `RGB9E5Float` word whose largest mantissa is below 256 under an exponent above 0:
//!   normalized, its three mantissas shifted up one bit as its exponent steps down one, until the
//!   largest reaches 256 or the exponent 0 (the word Metal's encoding of its value gives).
//!
//! A consumer copies the region's texels, row after row, into a buffer of words, one dispatch of
//! [`unused_bits_module`] of the texture's form rewrites the words of the tiles a mask names, and
//! it copies them back before the pass: the pass's write masks, discards, tests and uncovered
//! texels then keep Metal's words.
//!
//! **The module** (one per device, format and [`TexelWords`]). One `GLCompute` entry `main` with
//! local size ([`UnusedBitsModule::LOCAL_SIZE`], 1, 1), no push constants, and three storage
//! buffers of 4-byte words in set 0:
//!
//! - [`WORDS_BINDING`](UnusedBitsModule::WORDS_BINDING): the texels, each row
//!   `row_words` words (its texels, then any padding to a word), the rows of each layer
//!   `layer_rows` apart, layer after layer, from the region's row `first_row`;
//! - [`TILES_BINDING`](UnusedBitsModule::TILES_BINDING): the tile mask, one bit a 32 x 32 tile
//!   (bit `t % 32` of word `t / 32`), tile `t = (layer * tile_rows + y / 32) * tiles_per_row +
//!   x / 32` of the texel at `(x, y)` of a layer, `tile_rows = ceil(layer_rows / 32)`;
//! - [`SHAPE_BINDING`](UnusedBitsModule::SHAPE_BINDING): `[row_words, first_row, layer_rows,
//!   tiles_per_row]` ([`UnusedBitsModule::shape`]).
//!
//! Invocation `i` below the texels' words rewrites word `i` where its texel's tile's bit is set
//! (a tile past the mask's words is not); one past them writes nothing. A consumer whose words
//! exceed `maxStorageBufferRange` or `maxComputeWorkGroupCount[0]` workgroups binds them in
//! bands of rows, each with its `first_row`, one dispatch each: no word reads another.
//!
//! **Synchronization** (consumer): the copy into the words and the mask's and shape's writes
//! (`TRANSFER_WRITE`) → `SHADER_STORAGE_READ` and `SHADER_STORAGE_WRITE` in `COMPUTE_SHADER`
//! before the dispatch; its writes → `TRANSFER_READ` before the words are copied back.

use std::sync::Arc;

use crate::budget::{Budget, Meter};
use crate::device::requirements::Requirements;
use crate::device::Device;
use crate::error::{Error, Feature, InternalError, InternalKind, Unsupported};
use crate::metal::format::ExtendedRange;
use crate::metal::PixelFormat;
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage, Symbol};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, TypeId, Value};
use crate::spirv::ir::{
    natural_layout, BinOp, ConstBits, Constant, ExecMode, ExecModel, GlslOp, IoDir, LayoutRule,
    MemAccess, StorageClass, Type, TypedModule,
};

/// The crate-owned unused bits module of one device, format and [`TexelWords`], validated like
/// any output: the words of the texels Metal's tile store rewrites (the module docs).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnusedBitsModule {
    /// SPIR-V 1.5 words, entry point `main`, its bindings numbered as below in set 0.
    pub spirv: Arc<[u32]>,
    /// What the module needs from the device (within every Vulkan 1.2 device).
    pub requirements: Requirements,
}

impl UnusedBitsModule {
    /// Descriptor set of the bindings.
    pub const SET: u32 = 0;
    /// The texels' words, a storage buffer the dispatch rewrites in place.
    pub const WORDS_BINDING: u32 = 0;
    /// The tile mask, a storage buffer of one bit a tile.
    pub const TILES_BINDING: u32 = 1;
    /// The shape ([`shape`](Self::shape)), a storage buffer of [`SHAPE_WORDS`](Self::SHAPE_WORDS)
    /// words.
    pub const SHAPE_BINDING: u32 = 2;
    /// Invocations of one workgroup, along x: one 4-byte word each.
    pub const LOCAL_SIZE: u32 = 64;
    /// The side of a tile, in texels.
    pub const TILE: u32 = 32;
    /// The shape's words.
    pub const SHAPE_WORDS: u32 = 4;

    /// The shape's bytes: each row's words, the region's row of the first word bound, each
    /// layer's rows, and the tiles of a layer's row of tiles (the module docs).
    pub fn shape(row_words: u32, first_row: u32, layer_rows: u32, tiles_per_row: u32) -> [u8; 16] {
        let mut bytes = [0; 16];
        for (at, word) in [row_words, first_row, layer_rows, tiles_per_row]
            .into_iter()
            .enumerate()
        {
            bytes[4 * at..4 * at + 4].copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }

    /// The bytes of one texel of `format` held as `words`: an extended-range, padded or bfloat
    /// format's raw texel (`ExtendedRange::raw_format`) or a two-plane format's color word
    /// alone, or a signed normalized or `RGB9E5Float` texel. `None` where `format` has no such
    /// words: another format, or [`TexelWords::ColorPlane`] of a format but a two-plane one.
    pub fn texel_bytes(format: PixelFormat, words: TexelWords) -> Option<u32> {
        let rule = rule(format, words)?;
        let Some(form) = ExtendedRange::of(format) else {
            return match format {
                PixelFormat::R8Snorm => Some(1),
                PixelFormat::RG8Snorm | PixelFormat::R16Snorm => Some(2),
                PixelFormat::RGBA16Snorm => Some(8),
                _ => Some(4),
            };
        };
        Some(match (rule, form) {
            (Rule::Padded(_), ExtendedRange::R10X6 | ExtendedRange::R10X6Srgb) => 2,
            (Rule::Bfloat, ExtendedRange::R16Bfloat) => 2,
            (Rule::Bfloat, ExtendedRange::Rgba16Bfloat) => 8,
            (Rule::Fields | Rule::PlaneTexel, _) => 8,
            _ => 4,
        })
    }
}

/// How a consumer holds a texture's texels in the words it binds.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TexelWords {
    /// The raw texels of `ExtendedRange::raw_format`: a two-plane format's color word and then its
    /// alpha byte in a word of its own.
    Raw,
    /// A two-plane format's color words alone, as Metal lays its plane 0 in memory (a texture over
    /// a surface's planes or a heap's bytes).
    ColorPlane,
}

/// What the kernel does to each word.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Rule {
    /// Bits 30 and 31 set: `BGR10_XR`'s word.
    OnesOnTop,
    /// Bits 30 and 31 cleared: a two-plane format's color word.
    ZerosOnTop,
    /// A two-plane format's raw texel: the even word's bits 30 and 31 cleared, the odd word (the
    /// alpha byte) kept.
    PlaneTexel,
    /// `BGRA10_XR`'s fields: each field's low six bits cleared, and the odd word's high field
    /// (alpha) code clamped to `[384, 894]`.
    Fields,
    /// A padded form's two 16-bit words: each one's low six bits cleared, or each one's code
    /// replaced by the table's word for it (an `_sRGB` twin's, [`rewritten_words`]).
    Padded(Option<Vec<u32>>),
    /// A bfloat form's two 16-bit codes: each one's value through `half` and back
    /// (`ExtendedRange::load_code`, [`bfloat_load`]).
    Bfloat,
    /// A signed normalized format's codes, each byte (each 16-bit half where `wide`): one at the
    /// minimum code (`0x80`, `0x8000`) raised to the code above it ([`snorm_minimum`]).
    SnormMinimum { wide: bool },
    /// An `RGB9E5Float` word, normalized ([`shared_exponent`]).
    SharedExponent,
}

/// The rule of `format`'s texels held as `words`, or `None` where it has none.
fn rule(format: PixelFormat, words: TexelWords) -> Option<Rule> {
    if let Some(form) = ExtendedRange::of(format) {
        return extended_rule(form, words);
    }
    Some(match (format, words) {
        (_, TexelWords::ColorPlane) => return None,
        (PixelFormat::R8Snorm | PixelFormat::RG8Snorm | PixelFormat::RGBA8Snorm, _) => {
            Rule::SnormMinimum { wide: false }
        }
        (PixelFormat::R16Snorm | PixelFormat::RG16Snorm | PixelFormat::RGBA16Snorm, _) => {
            Rule::SnormMinimum { wide: true }
        }
        (PixelFormat::RGB9E5Float, _) => Rule::SharedExponent,
        _ => return None,
    })
}

/// The rule of an extended-range, padded or bfloat `form`'s texels held as `words`, or `None`
/// where it has none.
fn extended_rule(form: ExtendedRange, words: TexelWords) -> Option<Rule> {
    Some(match (form, words) {
        (ExtendedRange::Bgr10 { .. }, TexelWords::Raw) => Rule::OnesOnTop,
        (ExtendedRange::Bgra10 { .. }, TexelWords::Raw) => Rule::Fields,
        (ExtendedRange::Planes { .. }, TexelWords::Raw) => Rule::PlaneTexel,
        (ExtendedRange::Planes { .. }, TexelWords::ColorPlane) => Rule::ZerosOnTop,
        (_, TexelWords::ColorPlane) => return None,
        _ if form.padded() => Rule::Padded(rewritten_words(form)),
        _ if form.bfloat() => Rule::Bfloat,
        _ => return None,
    })
}

/// The mask a padded linear form's word is ANDed with: the top ten bits of each of its two 16-bit
/// words. `BGRA10_XR`'s fields take it too.
const FIELDS_MASK: u32 = 0xffc0_ffc0;
/// The color word's code bits: bits 30 and 31 apart.
const CODES: u32 = 0x3fff_ffff;
/// `BGRA10_XR`'s alpha codes Metal's store keeps.
const ALPHA_CODES: [u32; 2] = [384, 894];

/// The words of the codes a loading pass leaves for each code `c` of `form`, `load_code(c) << 6`
/// (`ExtendedRange::load_code`), or `None` where every code stays (the linear forms: a mask
/// does) or `form` is not padded.
fn rewritten_words(form: ExtendedRange) -> Option<Vec<u32>> {
    let words: Vec<u32> = (0..1024u32)
        .map(|c| form.load_code(c).map(|m| m << 6))
        .collect::<Option<_>>()?;
    let moved = words.iter().zip(0u32..).any(|(&w, c)| w != c << 6);
    moved.then_some(words)
}

/// A `Private` `u32[1024]` holding `words`.
fn words_table(b: &mut ModuleBuilder, u32_: TypeId, words: &[u32]) -> BResult<GlobalId> {
    let len = b.constant(Constant::Scalar {
        ty: u32_,
        bits: ConstBits(words.len() as u64),
    })?;
    let arr = b.ty(Type::Array {
        elem: u32_,
        len,
        stride: None,
    })?;
    let mut parts = Vec::with_capacity(words.len());
    for &w in words {
        parts.push(b.constant(Constant::Scalar {
            ty: u32_,
            bits: ConstBits(u64::from(w)),
        })?);
    }
    let init = b.constant(Constant::Composite {
        ty: arr,
        parts: parts.into(),
    })?;
    b.plain_global(StorageClass::Private, arr, Some(init))
}

/// The binding symbols of the words, the mask and the shape in the typed module.
const WORDS: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 0,
};
const TILES: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 1,
};
const SHAPE: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 2,
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

/// `rule`'s rewrite of `word`, the `odd` (a `bool`) word of its texel; `table` is a padded
/// `_sRGB` twin's table of rewritten words ([`words_table`]).
fn rewritten(
    f: &mut FunctionBuilder<'_>,
    rule: &Rule,
    table: Option<GlobalId>,
    word: Value,
    odd: Value,
) -> BResult<Value> {
    Ok(match rule {
        Rule::OnesOnTop => {
            let top = u32_value(f, !CODES)?;
            f.binary(BinOp::BitwiseOr, word, top)?
        }
        Rule::ZerosOnTop => {
            let codes = u32_value(f, CODES)?;
            f.binary(BinOp::BitwiseAnd, word, codes)?
        }
        Rule::PlaneTexel => {
            let codes = u32_value(f, CODES)?;
            let color = f.binary(BinOp::BitwiseAnd, word, codes)?;
            f.select(odd, word, color)?
        }
        Rule::Fields => extended_range_fields(f, word, odd)?,
        Rule::Padded(_) => {
            let Some(table) = table else {
                let mask = u32_value(f, FIELDS_MASK)?;
                return f.binary(BinOp::BitwiseAnd, word, mask);
            };
            // Each 16-bit word's code indexes the table of its rewritten word.
            let (k6, k16) = (u32_value(f, 6)?, u32_value(f, 16)?);
            let low = u32_value(f, 0xffff)?;
            let table = f.module().global_value(table)?;
            let lo = f.binary(BinOp::BitwiseAnd, word, low)?;
            let lo = f.binary(BinOp::ShiftRightLogical, lo, k6)?;
            let hi = f.binary(BinOp::ShiftRightLogical, word, k16)?;
            let hi = f.binary(BinOp::ShiftRightLogical, hi, k6)?;
            let lo = f.access_chain(table, &[lo])?;
            let lo = f.load(lo, MemAccess::default())?;
            let hi = f.access_chain(table, &[hi])?;
            let hi = f.load(hi, MemAccess::default())?;
            let hi = f.binary(BinOp::ShiftLeftLogical, hi, k16)?;
            f.binary(BinOp::BitwiseOr, lo, hi)?
        }
        Rule::Bfloat => {
            let (k16, low) = (u32_value(f, 16)?, u32_value(f, 0xffff)?);
            let lo = f.binary(BinOp::BitwiseAnd, word, low)?;
            let lo = bfloat_load(f, lo)?;
            let hi = f.binary(BinOp::ShiftRightLogical, word, k16)?;
            let hi = bfloat_load(f, hi)?;
            let hi = f.binary(BinOp::ShiftLeftLogical, hi, k16)?;
            f.binary(BinOp::BitwiseOr, lo, hi)?
        }
        Rule::SnormMinimum { wide } => snorm_minimum(f, word, *wide)?,
        Rule::SharedExponent => shared_exponent(f, word)?,
    })
}

/// `word`'s signed normalized codes, each byte (each 16-bit half where `wide`) at the minimum
/// code (`0x80`, `0x8000`) raised by one to the code above it: no carry leaves its lane.
fn snorm_minimum(f: &mut FunctionBuilder<'_>, word: Value, wide: bool) -> BResult<Value> {
    let (bits, minimum) = match wide {
        true => (16, 0x8000),
        false => (8, 0x80),
    };
    let (zero, mask, minimum) = (
        u32_value(f, 0)?,
        u32_value(f, (1 << bits) - 1)?,
        u32_value(f, minimum)?,
    );
    let mut out = word;
    for shift in (0..32).step_by(bits) {
        let k = u32_value(f, shift)?;
        let lane = f.binary(BinOp::ShiftRightLogical, word, k)?;
        let lane = f.binary(BinOp::BitwiseAnd, lane, mask)?;
        let least = f.binary(BinOp::IEqual, lane, minimum)?;
        let one = u32_value(f, 1 << shift)?;
        let raise = f.select(least, one, zero)?;
        out = f.binary(BinOp::IAdd, out, raise)?;
    }
    Ok(out)
}

/// The `RGB9E5Float` word `word` normalized: its three 9-bit mantissas (red at bit 0, green at
/// 9, blue at 18) shifted up by `s` and its exponent (bits 27 to 31) lowered by `s`, `s` the
/// fewest shifts that bring the largest mantissa to 256 or more, at most the exponent (all of it
/// for three zero mantissas). A word whose largest mantissa is 256 or more, or whose exponent is
/// 0, stays.
fn shared_exponent(f: &mut FunctionBuilder<'_>, word: Value) -> BResult<Value> {
    let c = |f: &mut FunctionBuilder<'_>, v: u32| u32_value(f, v);
    let mantissa = c(f, 0x1ff)?;
    let (k9, k18, k27) = (c(f, 9)?, c(f, 18)?, c(f, 27)?);
    let r = f.binary(BinOp::BitwiseAnd, word, mantissa)?;
    let g = f.binary(BinOp::ShiftRightLogical, word, k9)?;
    let g = f.binary(BinOp::BitwiseAnd, g, mantissa)?;
    let b = f.binary(BinOp::ShiftRightLogical, word, k18)?;
    let b = f.binary(BinOp::BitwiseAnd, b, mantissa)?;
    let e = f.binary(BinOp::ShiftRightLogical, word, k27)?;
    let largest = f.glsl(GlslOp::UMax, &[g, b])?;
    let largest = f.glsl(GlslOp::UMax, &[r, largest])?;
    // The shifts that bring the largest to 256: `8 - msb` below it, none at or above it, and the
    // whole exponent for a zero.
    let msb = f.glsl(GlslOp::FindUMsb, &[largest])?;
    let (zero, k8, k31, k256) = (c(f, 0)?, c(f, 8)?, c(f, 31)?, c(f, 256)?);
    let need = f.binary(BinOp::ISub, k8, msb)?;
    let full = f.binary(BinOp::UGreaterThanEqual, largest, k256)?;
    let need = f.select(full, zero, need)?;
    let none = f.binary(BinOp::IEqual, largest, zero)?;
    let need = f.select(none, k31, need)?;
    let shift = f.glsl(GlslOp::UMin, &[need, e])?;
    let r = f.binary(BinOp::ShiftLeftLogical, r, shift)?;
    let g = f.binary(BinOp::ShiftLeftLogical, g, shift)?;
    let b = f.binary(BinOp::ShiftLeftLogical, b, shift)?;
    let e = f.binary(BinOp::ISub, e, shift)?;
    let g = f.binary(BinOp::ShiftLeftLogical, g, k9)?;
    let b = f.binary(BinOp::ShiftLeftLogical, b, k18)?;
    let e = f.binary(BinOp::ShiftLeftLogical, e, k27)?;
    let out = f.binary(BinOp::BitwiseOr, r, g)?;
    let out = f.binary(BinOp::BitwiseOr, out, b)?;
    f.binary(BinOp::BitwiseOr, out, e)
}

/// `x >> k` rounded to nearest, ties to even (`k` at most 31, `x` below `2^30`).
fn shift_nearest_even(f: &mut FunctionBuilder<'_>, x: Value, k: Value) -> BResult<Value> {
    let (zero, one) = (u32_value(f, 0)?, u32_value(f, 1)?);
    let q = f.binary(BinOp::ShiftRightLogical, x, k)?;
    let back = f.binary(BinOp::ShiftLeftLogical, q, k)?;
    let rest = f.binary(BinOp::ISub, x, back)?;
    // Twice the rest against `2^k`: past it rounds up, at it (a tie) up to an even quotient.
    let twice = f.binary(BinOp::ShiftLeftLogical, rest, one)?;
    let unit = f.binary(BinOp::ShiftLeftLogical, one, k)?;
    let past = f.binary(BinOp::UGreaterThan, twice, unit)?;
    let tie = f.binary(BinOp::IEqual, twice, unit)?;
    let odd = f.binary(BinOp::BitwiseAnd, q, one)?;
    let odd = f.binary(BinOp::INotEqual, odd, zero)?;
    let tie_up = f.binary(BinOp::LogicalAnd, tie, odd)?;
    let up = f.binary(BinOp::LogicalOr, past, tie_up)?;
    let up = f.select(up, one, zero)?;
    f.binary(BinOp::IAdd, q, up)
}

/// The bfloat code a loading pass leaves for the code `code` (a `u32` below `0x10000`;
/// `ExtendedRange::load_code`): a NaN `0x7fc0`; at least `65536` in magnitude the infinity of
/// its sign; a value `half` holds (at least `2^-14`) itself; a smaller one rounded to a multiple
/// of `2^-24` (nearest, ties to even), then to the bfloat code nearest that (ties to even), its
/// sign kept. Integer operations on the bits alone, so no float mode changes the answer.
fn bfloat_load(f: &mut FunctionBuilder<'_>, code: Value) -> BResult<Value> {
    let c = |f: &mut FunctionBuilder<'_>, v: u32| u32_value(f, v);
    let sign_bit = c(f, 0x8000)?;
    let sign = f.binary(BinOp::BitwiseAnd, code, sign_bit)?;
    let magnitude_bits = c(f, 0x7fff)?;
    let mag = f.binary(BinOp::BitwiseAnd, code, magnitude_bits)?;
    // The code's exponent field and its significand (the hidden bit set past the subnormals).
    let k7 = c(f, 7)?;
    let exponent = f.binary(BinOp::ShiftRightLogical, mag, k7)?;
    let fraction_bits = c(f, 0x7f)?;
    let fraction = f.binary(BinOp::BitwiseAnd, mag, fraction_bits)?;
    let (zero, one) = (c(f, 0)?, c(f, 1)?);
    let subnormal = f.binary(BinOp::IEqual, exponent, zero)?;
    let hidden = c(f, 0x80)?;
    let normal = f.binary(BinOp::BitwiseOr, fraction, hidden)?;
    let significand = f.select(subnormal, fraction, normal)?;
    let exponent = f.select(subnormal, one, exponent)?;
    // The value in units of `2^-24`, `significand * 2^(exponent - 110)`: four times the
    // significand shifted right by `112 - exponent`, rounded to nearest even.
    let top = c(f, 112)?;
    let capped = f.glsl(GlslOp::UMin, &[exponent, top])?;
    let k = f.binary(BinOp::ISub, top, capped)?;
    let k31 = c(f, 31)?;
    let k = f.glsl(GlslOp::UMin, &[k, k31])?;
    let k2 = c(f, 2)?;
    let quad = f.binary(BinOp::ShiftLeftLogical, significand, k2)?;
    let units = shift_nearest_even(f, quad, k)?;
    // Those units (at most `2^10`) rounded to eight significant bits, the bfloat's.
    let by_two = shift_nearest_even(f, units, one)?;
    let by_two = f.binary(BinOp::ShiftLeftLogical, by_two, one)?;
    let by_four = shift_nearest_even(f, units, k2)?;
    let by_four = f.binary(BinOp::ShiftLeftLogical, by_four, k2)?;
    let (k256, k512) = (c(f, 256)?, c(f, 512)?);
    let nine = f.binary(BinOp::UGreaterThanEqual, units, k256)?;
    let ten = f.binary(BinOp::UGreaterThanEqual, units, k512)?;
    let units = f.select(nine, by_two, units)?;
    let units = f.select(ten, by_four, units)?;
    // The `float` of `units * 2^-24`, its top 16 bits: the exponent field `msb + 103` and the
    // bits below the most significant one.
    let msb = f.glsl(GlslOp::FindUMsb, &[units])?;
    let bias = c(f, 103)?;
    let field = f.binary(BinOp::IAdd, msb, bias)?;
    let k23 = c(f, 23)?;
    let field = f.binary(BinOp::ShiftLeftLogical, field, k23)?;
    let up = f.binary(BinOp::ISub, k23, msb)?;
    let mantissa = f.binary(BinOp::ShiftLeftLogical, units, up)?;
    let mantissa_bits = c(f, 0x7f_ffff)?;
    let mantissa = f.binary(BinOp::BitwiseAnd, mantissa, mantissa_bits)?;
    let bits = f.binary(BinOp::BitwiseOr, field, mantissa)?;
    let k16 = c(f, 16)?;
    let small = f.binary(BinOp::ShiftRightLogical, bits, k16)?;
    let none = f.binary(BinOp::IEqual, units, zero)?;
    let small = f.select(none, zero, small)?;
    let small = f.binary(BinOp::BitwiseOr, small, sign)?;
    // The value `half` holds, the infinity past it, and a NaN.
    let least_normal = c(f, 0x3880)?;
    let held = f.binary(BinOp::UGreaterThanEqual, mag, least_normal)?;
    let out = f.select(held, code, small)?;
    let overflow = c(f, 0x4780)?;
    let past = f.binary(BinOp::UGreaterThanEqual, mag, overflow)?;
    let infinity = c(f, 0x7f80)?;
    let signed_infinity = f.binary(BinOp::BitwiseOr, infinity, sign)?;
    let out = f.select(past, signed_infinity, out)?;
    let nan = f.binary(BinOp::UGreaterThan, mag, infinity)?;
    let quiet = c(f, 0x7fc0)?;
    f.select(nan, quiet, out)
}

/// `BGRA10_XR`'s `word`, the `odd` (a `bool`) word of its texel, with each field's low six bits
/// cleared and (the odd word's high field) alpha's code clamped to `[384, 894]`: Metal's tile
/// store and its blits through a compressed texture alike (`compressed_blit.rs`).
pub(super) fn extended_range_fields(
    f: &mut FunctionBuilder<'_>,
    word: Value,
    odd: Value,
) -> BResult<Value> {
    let mask = u32_value(f, FIELDS_MASK)?;
    let even = f.binary(BinOp::BitwiseAnd, word, mask)?;
    // The odd word: red's field, then alpha's code clamped.
    let low_mask = u32_value(f, FIELDS_MASK & 0xffff)?;
    let low = f.binary(BinOp::BitwiseAnd, word, low_mask)?;
    let k22 = u32_value(f, 22)?;
    let code = f.binary(BinOp::ShiftRightLogical, word, k22)?;
    let [least, most] = ALPHA_CODES;
    let (least, most) = (u32_value(f, least)?, u32_value(f, most)?);
    let below = f.binary(BinOp::ULessThan, code, least)?;
    let code = f.select(below, least, code)?;
    let above = f.binary(BinOp::UGreaterThan, code, most)?;
    let code = f.select(above, most, code)?;
    let high = f.binary(BinOp::ShiftLeftLogical, code, k22)?;
    let odd_word = f.binary(BinOp::BitwiseOr, low, high)?;
    f.select(odd, odd_word, even)
}

/// The module as a typed module, before serialization: the module [`unused_bits_module`]
/// serializes, which the CPU interpreter runs, of `rule` over texels of `texel_bytes` bytes. Its
/// bindings are the symbols `WORDS`, `TILES` and `SHAPE`.
fn typed_module(rule: &Rule, texel_bytes: u32) -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
    let words = words_buffer(&mut b, WORDS, u32_)?;
    let tiles = words_buffer(&mut b, TILES, u32_)?;
    let shape = words_buffer(&mut b, SHAPE, u32_)?;
    let table = match rule {
        Rule::Padded(Some(t)) => Some(words_table(&mut b, u32_, t)?),
        _ => None,
    };
    let invocation = b.builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let zero = u32_value(&mut f, 0)?;
    let invocation = f.module().global_value(invocation)?;
    let invocation = f.load(invocation, MemAccess::default())?;
    let i = f.composite_extract(invocation, &[0])?;
    let words = f.module().global_value(words)?;
    let tiles = f.module().global_value(tiles)?;
    let shape = f.module().global_value(shape)?;
    let count = f.array_length(words, 0)?;
    let inside = f.binary(BinOp::ULessThan, i, count)?;
    let rule = rule.clone();
    let rewrite: ArmFn<'_, ()> = Box::new(move |f, _| {
        let shape_word = |f: &mut FunctionBuilder<'_>, at: u32| -> BResult<Value> {
            let at = u32_value(f, at)?;
            let p = f.access_chain(shape, &[zero, at])?;
            f.load(p, MemAccess::default())
        };
        let row_words = shape_word(f, 0)?;
        let first_row = shape_word(f, 1)?;
        let layer_rows = shape_word(f, 2)?;
        let tiles_per_row = shape_word(f, 3)?;
        let (k1, k5, k31) = (u32_value(f, 1)?, u32_value(f, 5)?, u32_value(f, 31)?);
        // The word's row of the region (across its layers), and its word in that row.
        let row = f.binary(BinOp::UDiv, i, row_words)?;
        let row = f.binary(BinOp::IAdd, first_row, row)?;
        let w = f.binary(BinOp::UMod, i, row_words)?;
        let layer = f.binary(BinOp::UDiv, row, layer_rows)?;
        let y = f.binary(BinOp::UMod, row, layer_rows)?;
        // The word's first texel: its byte in the row over the texel's bytes.
        let four = u32_value(f, 4)?;
        let texel = u32_value(f, texel_bytes)?;
        let byte = f.binary(BinOp::IMul, w, four)?;
        let x = f.binary(BinOp::UDiv, byte, texel)?;
        let tile_rows = f.binary(BinOp::IAdd, layer_rows, k31)?;
        let tile_rows = f.binary(BinOp::ShiftRightLogical, tile_rows, k5)?;
        let tile_y = f.binary(BinOp::ShiftRightLogical, y, k5)?;
        let tile_x = f.binary(BinOp::ShiftRightLogical, x, k5)?;
        let tile = f.binary(BinOp::IMul, layer, tile_rows)?;
        let tile = f.binary(BinOp::IAdd, tile, tile_y)?;
        let tile = f.binary(BinOp::IMul, tile, tiles_per_row)?;
        let tile = f.binary(BinOp::IAdd, tile, tile_x)?;
        let at = f.binary(BinOp::ShiftRightLogical, tile, k5)?;
        let bit = f.binary(BinOp::BitwiseAnd, tile, k31)?;
        let mask_words = f.array_length(tiles, 0)?;
        let named = f.binary(BinOp::ULessThan, at, mask_words)?;
        // A tile past the mask's words reads as unset (its word read from 0, which exists).
        let read_at = f.select(named, at, zero)?;
        let p = f.access_chain(tiles, &[zero, read_at])?;
        let mask = f.load(p, MemAccess::default())?;
        let mask = f.binary(BinOp::ShiftRightLogical, mask, bit)?;
        let mask = f.binary(BinOp::BitwiseAnd, mask, k1)?;
        let set = f.binary(BinOp::INotEqual, mask, zero)?;
        let set = f.binary(BinOp::LogicalAnd, set, named)?;
        let odd = f.binary(BinOp::BitwiseAnd, w, k1)?;
        let odd = f.binary(BinOp::INotEqual, odd, zero)?;
        let p = f.access_chain(words, &[zero, i])?;
        let word = f.load(p, MemAccess::default())?;
        let made = rewritten(f, &rule, table, word, odd)?;
        let kept = f.select(set, made, word)?;
        f.store(p, kept, MemAccess::default())?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    f.if_(&mut (), inside, rewrite, None)?;
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize {
            x: UnusedBitsModule::LOCAL_SIZE,
            y: 1,
            z: 1,
        }],
    )?;
    b.finalize()
}

/// The fixed binding numbers of the module's symbols.
fn binding_of(sym: &Symbol) -> Option<u32> {
    match sym {
        Symbol::DescriptorSet(_) => Some(UnusedBitsModule::SET),
        Symbol::Binding(s) if *s == WORDS => Some(UnusedBitsModule::WORDS_BINDING),
        Symbol::Binding(s) if *s == TILES => Some(UnusedBitsModule::TILES_BINDING),
        Symbol::Binding(s) if *s == SHAPE => Some(UnusedBitsModule::SHAPE_BINDING),
        _ => None,
    }
}

/// The unused bits module of `device` for textures of `format` held as `words`: built with the
/// typed builder, checked by the owned verifier against the device's profile (requirements
/// included), serialized, and numbered with the fixed bindings of [`UnusedBitsModule`]. A format
/// that has no such words ([`UnusedBitsModule::texel_bytes`] is `None`) is
/// `NotImplemented(Format)`.
pub fn unused_bits_module(
    device: &Device,
    format: PixelFormat,
    words: TexelWords,
) -> Result<UnusedBitsModule, Error> {
    let (Some(rule), Some(texel_bytes)) = (
        rule(format, words),
        UnusedBitsModule::texel_bytes(format, words),
    ) else {
        return Err(Unsupported::not_implemented(Feature::Format).into());
    };
    let profile = device.profile();
    let m = typed_module(&rule, texel_bytes)?;
    let mut meter = Meter::new(&Budget::default());
    crate::spirv::verify::verify(&m, profile, &mut meter)?;
    let r = crate::spirv::serialize::serialize(m)?;
    profile.check(&r.requirements).map_err(|c| {
        Error::Internal(InternalError::new(
            InternalKind::RequirementsExceedProfile,
            format!("the unused bits kernel needs {c:?}"),
        ))
    })?;
    let words = crate::spirv::serialize::relocate(&r, &binding_of)?;
    Ok(UnusedBitsModule {
        spirv: words.into(),
        requirements: r.requirements,
    })
}

#[cfg(test)]
mod tests;
