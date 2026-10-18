//! Metal's conversion of a texel to a `half` result, by the treatment of the bound pixel format.
//!
//! A read, gather, or sample with a `half` result of a texture slot reads the slot's
//! `SpecKey::TexelFormat` specialization (`metal::texel_format`: the bound format's
//! `BicubicTexels` bits where its conversion is measured, else 0), which the binder sets from the
//! bound format, and converts the image's `f32` texel (the device's, filtered where the access
//! filters) through one module routine, [`define`]: `fn(x: vec4, flags: u32, sample: bool,
//! channel: u32) -> vec4`, called with the slot's constant, so once a device specializes the
//! module and inlines the call, the other families' arms fold away. Measured on an Apple M3 Max (macOS 26.7; the
//! case `f8f81b81` and the probe cases):
//!
//! - **Reads and gathers** ([`HalfForm::Read`]): the texel's value correctly rounded to nearest
//!   even, whatever the texture type and access (`access::sample`, `access::read`,
//!   `access::read_write`); a 32-bit float texel overflows to the infinity and keeps half
//!   denormals.
//! - **Samples, nearest or linear** ([`HalfForm::Sample`]): a 32-bit float texel's (filtered)
//!   value rounded toward zero, saturating at `±65504`; a half-float texture's linear sum to
//!   nearest with ties away from zero; a normalized texel's value rounded onto its format's grid
//!   (`1/4080`, `1/2032`, `1/65535`, `1/32767`) and then to nearest even, which for a single
//!   texel is its read. The same in every form read: normalized coordinates and pixel ones, a
//!   constant offset, an explicit level, a bias, gradients, every address mode probed, and 1D,
//!   1D array, 2D array, 3D, and cube textures.
//!
//! **How the routine computes**. Every family rounds one `f32` value, and every rounding to nearest
//! even is one sequence shared by the families, so the routine is short where a device inlines it
//! at every call before it specializes (MoltenVK: SPIRV-Cross writes each function inlined):
//!
//! - A normalized or packed family's value is put onto its grid `D` first: the step
//!   `n = RoundEven(x · D)` (`texel_round::grid_step`), then `|n| / D` with the sign of `n` (`+0`
//!   for `n = 0`). On the 8-bit grids (`4080`, `2032`) and the packed code scales (1, 3, 15, 31,
//!   63, 255, 1023) that is the `f32` product `|n| · RN(1 / D)`, whose rounding to `half` is
//!   `|n| / D`'s for every step, whichever way the product rounds (both `f32` neighbours of the
//!   exact product round to that `half`; `the_reciprocal_product_rounds_like_the_quotient`). On
//!   the 16-bit grids (`65535`, `32767`) it is not, and the exact `half` comes from one integer
//!   multiplication (`texel_round::ones_ratio_to_half`).
//! - That value is rounded to nearest even once by integer arithmetic on its bits, in every
//!   rounding form: it is a normal `half`'s, or a `half` already. The texel itself, for the other
//!   families, is rounded to nearest even once by the module's `half` rounding
//!   (`narrow::round_texel`) where the device's is measured or controlled to be Metal's
//!   (`QuantizeForm::Device`, `Converted` and `Controlled`), else by the integer sequence
//!   (`texel_round::f32_to_half`, `QuantizeForm::Integer`), so every device answers alike. The
//!   grid's value is selected after that rounding, so where a device specializes a grid class, no
//!   rounding of the texel (on MoltenVK a `[[clang::optnone]]` call at every access) is left.
//! - A sample of a float or half-float texel rounds toward zero saturating or to nearest with ties
//!   away: from the device's rounding to nearest even where that is Metal's, by stepping to its
//!   neighbour where the directed mode differs (`texel_round::f32_to_half_from_even`), else by the
//!   integer sequence with its mode chosen at run time (`f32_to_half_by`).
//!
//! A format whose conversion is not measured (`TexelFormat::OTHER`) takes the exact rounding of
//! every other `half` texel (`narrow::round_texel`, to nearest, half subnormals and the decoded NaN
//! kept), which the module records (`PrecisionNote::HalfTexelConversion`).
//!
//! **Packed formats**. A packed format's class holds its `TexelFormat::PACKED` field: an access
//! that reads one texel (`TexelLanes::Texel` or `Gathered`) recovers each channel's code from the
//! device's value on its code scale and converts `c / (2^b - 1)` to `half` exactly
//! (`ratio_to_half`), and a small float's value rounds to nearest even (it is a `half` value
//! already, so exactly). A two-plane YCbCr format's one texel (`TexelLanes::Texel`: a read or a
//! sample of one texel; `PackedTexels::ycbcr`) and a 2D gather's lanes (`Gathered`) round its
//! value, a step of `1/2048` or `1/8192`, with ties away from zero (every channel of five formats
//! on the grid `8192`, and the gathers; a step of `1/2048` is a `half`). A value that may be
//! filtered (`TexelLanes::Filtered`: a sample the device filters, or a
//! cube gather's lanes, each one channel's) keeps the `OTHER` arm, even under a packed normalized
//! class's `EXACT` family (`TexelFormat::packed_exact`), whose grids are per channel; a small
//! float's takes the half-float family's. The emulated linear filter of these formats
//! (`lower::sampler::linear`) rounds its own result and does not come here.
//!
//! **On a device that samples through Metal's own texture unit**: where the device's `f32` texel is
//! Metal's `float` result of the same access (`VkProperty::MetalSampler`) and the module's rounding
//! to nearest even is Metal's (`QuantizeForm::Device`, `Converted` and `Controlled`), the routine
//! tells apart only the families whose `half` result is not that `float` result rounded to nearest
//! even ([`HalfFamilies::MetalSampler`]): a 32-bit float texel's sample (toward zero), the 16-bit
//! normalized grids (their `f32` value rounds twice: `65455 / 65535`, `65519 / 65535` and `32759 /
//! 32767`), and a YCbCr field's one texel on the grid `8192` (ties away). Every other family's
//! `half` is the `OTHER` arm's rounding of the device's texel there: an 8-bit normalized step `n /
//! D` on `1/255`, `1/4080`, `1/127` or `1/2032`, a packed code's `c / S`, a YCbCr step on `1/2048`
//! and a filter grid's step, rounded to `f32` and then to `half`, is the step's own `half` (every
//! step checked, `metal_float_steps_round_to_their_half`), and a half-float or small-float texel's
//! value is a `half` already. So the module reads none of their bits, and their classes specialize
//! alike with a slot bound to nothing (`spirv::spec_bits`). The 16-bit grids' exact `half` is there
//! one product of the step by a factor whose every product rounds as the quotient does
//! (`texel_round::ones_ratio_factor`), rounded by the module, all float arithmetic: Metal's
//! compiler, which MoltenVK hands the routine inlined at every access, takes far less time over it
//! than over the integer form (about 2 s over a function of 311 such accesses). Where the module's
//! rounding is the device's own (native half rounding) they keep the integer form.
//!
//! A sample's filtered value is the device's where the device filters it: on MoltenVK, which
//! samples through Metal's own sampler (`VkProperty::MetalSampler`), that is Metal's `float`
//! result, on the format's grid (normalized formats) or the sum rounded to `f32` (32-bit float
//! formats), so the conversion of it is Metal's `half` result exactly. Elsewhere a linear sample
//! the emulation reaches filters by fetches with Metal's footprint and weights
//! (`lower::sampler::linear`), and this routine converts only the samples it does not reach.

use crate::lower::arith::legal::{FloatFmt, Form, Held};
use crate::lower::arith::narrow;
use crate::lower::arith::ops::{elem_ty, fsplat, splat, ty};
use crate::lower::arith::texel_round;
use crate::metal::bicubic::BicubicTexels;
use crate::metal::texel_format::{PackedTexels, TexelFormat};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder, QuantizeForm};
use crate::spirv::error::{BResult, BuildError, OperandRule};
use crate::spirv::ids::{FuncId, TypeId, Value as SValue};
use crate::spirv::ir::{BinOp, ConstBits, Constant, ConvOp, GlslOp, Type as SType, UnOp};

/// Which of Metal's rules a texel access with a `half` result takes (module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HalfForm {
    /// A read or a gather: the texel's value to nearest even.
    Read,
    /// A sample, nearest or linear: a 32-bit float value toward zero saturating, a half-float
    /// sum to nearest with ties away from zero.
    Sample,
}

/// Which lanes of an access's value hold which channel of one texel: what a packed format's
/// conversion recovers codes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TexelLanes {
    /// One texel's channels, red to alpha: a read, or a sample whose sampler reads one texel of
    /// one level.
    Texel,
    /// Every lane channel `c` (0 red to 3 alpha) of its own texel: a 2D gather's.
    Gathered(u32),
    /// A value that may be filtered: a sample through a sampler that may filter, or a cube
    /// gather (its corner texels are made up).
    Filtered,
}

impl TexelLanes {
    /// The routine's `channel` argument: the gathered channel, 4 for a texel, 5 for a value that
    /// may be filtered.
    pub fn code(self) -> u32 {
        match self {
            TexelLanes::Gathered(c) => c.min(3),
            TexelLanes::Texel => TEXEL_LANES,
            TexelLanes::Filtered => FILTERED_LANES,
        }
    }

    /// The argument as a constant of `fb`'s module.
    pub fn value(self, fb: &mut FunctionBuilder<'_>) -> BResult<SValue> {
        let c = fb.u32_const(self.code())?;
        fb.module().const_value(c)
    }
}

/// The `channel` argument of one texel's lanes ([`TexelLanes::Texel`]).
pub const TEXEL_LANES: u32 = 4;

/// The `channel` argument of a value that may be filtered ([`TexelLanes::Filtered`]).
pub const FILTERED_LANES: u32 = 5;

/// An upper estimate of the routine's words, counted once per module that defines it, and the bound
/// on its size in every `QuantizeForm`, which a device that inlines it at each call compiles once
/// per call (`the_half_texel_routine_words_bound_its_emission`). 1600 since a YCbCr format's one
/// texel rounds with ties away (1553 words in the `Integer` form).
pub const ROUTINE_WORDS: u64 = 1600;

/// An upper estimate of one call's words: the call and its three constant arguments.
pub const CALL_WORDS: u32 = 16;

/// The census name of the routine.
pub const CENSUS_NAME: &str = "texture.half_texel";

fn fform() -> Form {
    Form {
        held: Held::Float(FloatFmt::F32),
        lanes: 4,
    }
}

/// The held form of a texel's four lanes as `half`s.
fn hform() -> Form {
    Form {
        held: Held::Float(FloatFmt::Half),
        lanes: 4,
    }
}

fn uform(lanes: u8) -> Form {
    Form {
        held: Held::Int {
            bits: 32,
            container: 32,
        },
        lanes,
    }
}

/// Which families the routine tells apart (module docs).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum HalfFamilies {
    /// Every measured family: the device's texel is its own value, filtered by its own filter.
    #[default]
    Every,
    /// A device whose texel is Metal's `float` result and whose rounding to nearest even is
    /// Metal's: only a 32-bit float sample, the 16-bit normalized grids and a YCbCr field's one
    /// texel on the grid `8192` (module docs). The routine reads its `channel` for that field
    /// alone.
    MetalSampler,
}

/// Define the routine `fn(x: vec4, flags: u32, sample: bool, channel: u32) -> vec4` in `mb`
/// (module docs; `channel` is [`TexelLanes::code`]), telling apart `families`. A module that
/// rounds by the integer sequence (`QuantizeForm::Integer`) tells apart every family.
pub fn define(mb: &mut ModuleBuilder, families: HalfFamilies) -> BResult<FuncId> {
    let families = match mb.quantize_form() {
        QuantizeForm::Integer => HalfFamilies::Every,
        QuantizeForm::Device | QuantizeForm::Converted | QuantizeForm::Controlled => families,
    };
    let f32_ = mb.ty(SType::Float { width: 32 })?;
    let u32_ = mb.ty(SType::Int {
        width: 32,
        signed: false,
    })?;
    let bool_ = mb.ty(SType::Bool)?;
    let vf = mb.ty(SType::Vector { elem: f32_, len: 4 })?;
    let vu = mb.ty(SType::Vector { elem: u32_, len: 4 })?;
    let mut fb = mb.routine(vf)?;
    let x = fb.param(vf)?;
    let flags = fb.param(u32_)?;
    let sample = fb.param(bool_)?;
    let channel = fb.param(u32_)?;
    let r = match families {
        HalfFamilies::Every => body(&mut fb, x, flags, sample, channel, vu)?,
        HalfFamilies::MetalSampler => metal_body(&mut fb, x, flags, sample, channel)?,
    };
    fb.ret(Some(r))?;
    fb.finish()
}

/// The constant of form `f` (four lanes) whose lanes hold `bits`.
fn lanes_of(fb: &mut FunctionBuilder<'_>, f: Form, bits: [u64; 4]) -> BResult<SValue> {
    let e = elem_ty(fb, f.held)?;
    let mut parts = Vec::with_capacity(4);
    for b in bits {
        parts.push(fb.module().constant(Constant::Scalar {
            ty: e,
            bits: ConstBits(b),
        })?);
    }
    let t = ty(fb, f)?;
    let c = fb.module().constant(Constant::Composite {
        ty: t,
        parts: parts.into(),
    })?;
    fb.module().const_value(c)
}

/// A per-channel table of a packed class ([`PackedField::lanes`]): the four lanes it holds for a
/// field's channel code scales, or for `None`, a field without scales.
type Table<'a> = &'a dyn Fn(&mut FunctionBuilder<'_>, Option<[u32; 4]>) -> BResult<SValue>;

/// A packed class's field (`TexelFormat::PACKED`) read once from a routine's `flags` and `channel`,
/// which selects each per-channel table the routine needs ([`PackedField::lanes`]).
struct PackedField {
    /// Each field with scales: its scales, and whether it is the field.
    fields: Vec<([u32; 4], SValue)>,
    /// Whether the field states coded channels (a field with scales).
    coded: SValue,
    /// Whether the field is `PackedTexels::SmallFloat`.
    small: SValue,
    /// Whether the field is a two-plane YCbCr format's (`PackedTexels::ycbcr`).
    ycbcr: SValue,
    /// Whether the lanes are a gather's, each its channel's (`channel` below 4).
    gathered: SValue,
    /// The gathered channel (`channel`, at most 3).
    channel: SValue,
}

impl PackedField {
    fn read(fb: &mut FunctionBuilder<'_>, flags: SValue, channel: SValue) -> BResult<PackedField> {
        let u1 = |fb: &mut FunctionBuilder<'_>, v: u32| splat(fb, uform(1), u64::from(v));
        let shift = u1(fb, TexelFormat::PACKED)?;
        let field = fb.binary(BinOp::ShiftRightLogical, flags, shift)?;
        let four_bits = u1(fb, 0xF)?;
        let field = fb.binary(BinOp::BitwiseAnd, field, four_bits)?;
        let mut coded = splat(
            fb,
            Form {
                held: Held::Bool,
                lanes: 1,
            },
            0,
        )?;
        let mut ycbcr = coded;
        let mut fields = Vec::with_capacity(PackedTexels::ALL.len());
        for p in PackedTexels::ALL {
            if !p.ycbcr() && p.scales().is_none() {
                continue;
            }
            let k = u1(fb, p as u32)?;
            let is = fb.binary(BinOp::IEqual, field, k)?;
            if p.ycbcr() {
                ycbcr = fb.binary(BinOp::LogicalOr, ycbcr, is)?;
            }
            let Some(s) = p.scales() else {
                continue;
            };
            coded = fb.binary(BinOp::LogicalOr, coded, is)?;
            fields.push((s, is));
        }
        let small = u1(fb, PackedTexels::SmallFloat as u32)?;
        let small = fb.binary(BinOp::IEqual, field, small)?;
        let three = u1(fb, 3)?;
        let gathered = fb.binary(BinOp::ULessThanEqual, channel, three)?;
        let channel = fb.glsl(GlslOp::UMin, &[channel, three])?;
        Ok(PackedField {
            fields,
            coded,
            small,
            ycbcr,
            gathered,
            channel,
        })
    }

    /// `table`'s lanes for the field (its `None` lanes for a field without scales), a gather's
    /// lanes each its channel's.
    fn lanes(&self, fb: &mut FunctionBuilder<'_>, table: Table<'_>) -> BResult<SValue> {
        let mut lanes = table(fb, None)?;
        for (s, is) in &self.fields {
            let these = table(fb, Some(*s))?;
            lanes = fb.select(*is, these, lanes)?;
        }
        // A gather's lanes are one channel's: its lane in every lane.
        let one = fb.vector_extract_dynamic(lanes, self.channel)?;
        let t = fb.value_ty(lanes)?;
        let one = fb.composite_construct(t, &[one; 4])?;
        fb.select(self.gathered, one, lanes)
    }
}

/// The code scales `S` of a packed class's channels (`TexelFormat::PACKED`) as `u32` lanes, a
/// gather's lanes each its channel's (`channel` below 4), 1 in every lane for any other field;
/// whether the field states coded channels (a field with scales); and whether it is
/// `PackedTexels::SmallFloat`. Shared with the `float` routine (`float_texel`).
pub(crate) fn packed_scales(
    fb: &mut FunctionBuilder<'_>,
    flags: SValue,
    channel: SValue,
) -> BResult<(SValue, SValue, SValue)> {
    let p = PackedField::read(fb, flags, channel)?;
    let scales = p.lanes(fb, &|fb, s| {
        lanes_of(fb, uform(4), s.unwrap_or([1; 4]).map(u64::from))
    })?;
    Ok((scales, p.coded, p.small))
}

/// The `f32` bits of `v` and of `RN(1 / v)`, the reciprocal correctly rounded to `f32`.
fn grid_bits(v: u32) -> (u64, u64) {
    let d = v as f32;
    (u64::from(d.to_bits()), u64::from((1.0 / d).to_bits()))
}

fn body(
    fb: &mut FunctionBuilder<'_>,
    x: SValue,
    flags: SValue,
    sample: SValue,
    channel: SValue,
    vu: TypeId,
) -> BResult<SValue> {
    let f = fform();
    let u = uform(4);
    let ft = ty(fb, f)?;
    let u1 = |fb: &mut FunctionBuilder<'_>, v: u32| splat(fb, uform(1), u64::from(v));
    let f1 = |fb: &mut FunctionBuilder<'_>, bits: u64| {
        let s = Form {
            held: Held::Float(FloatFmt::F32),
            lanes: 1,
        };
        splat(fb, s, bits)
    };
    let has = |fb: &mut FunctionBuilder<'_>, bit: u32| -> BResult<SValue> {
        let m = u1(fb, bit)?;
        let z = u1(fb, 0)?;
        let and = fb.binary(BinOp::BitwiseAnd, flags, m)?;
        fb.binary(BinOp::INotEqual, and, z)
    };
    let is_float = has(fb, BicubicTexels::LINEAR)?;
    let is_half = has(fb, BicubicTexels::HALF)?;
    let float_or_half = fb.binary(BinOp::LogicalOr, is_float, is_half)?;
    // Normalized channels: the value onto the format's grid `D`, then `n / D` to nearest even. A
    // packed normalized class (`TexelFormat::packed_exact`) sums on its field's grids, which a
    // value that may be filtered does not name per lane (a cube gather's lanes are one channel's):
    // it keeps the `OTHER` arm.
    let wide = has(fb, BicubicTexels::WIDE16)?;
    let signed = has(fb, BicubicTexels::SNORM)?;
    let exact = has(fb, BicubicTexels::EXACT)?;
    let packed = has(fb, TexelFormat::PACKED_MASK)?;
    let own_grid = fb.unary(UnOp::LogicalNot, packed)?;
    let norm_arm = fb.binary(BinOp::LogicalAnd, exact, own_grid)?;
    let (g8, r8) = grid_bits(TexelFormat::GRID8);
    let (g8s, r8s) = grid_bits(TexelFormat::GRID8S);
    let (g16, _) = grid_bits(TexelFormat::GRID16);
    let (g16s, _) = grid_bits(TexelFormat::GRID16S);
    let (g8, g8s, g16, g16s) = (f1(fb, g8)?, f1(fb, g8s)?, f1(fb, g16)?, f1(fb, g16s)?);
    let narrow_grid = fb.select(signed, g8s, g8)?;
    let broad = fb.select(signed, g16s, g16)?;
    let d = fb.select(wide, broad, narrow_grid)?;
    let d = fb.composite_construct(ft, &[d; 4])?;
    // The 16-bit grids' value is `ones_ratio_to_half`'s, not the product's.
    let (r8, r8s) = (f1(fb, r8)?, f1(fb, r8s)?);
    let rcp = fb.select(signed, r8s, r8)?;
    let rcp = fb.composite_construct(ft, &[rcp; 4])?;
    // A packed class's one texel: each channel's code recovered on its scale and its value `c / S`,
    // or a small float's value (a `half` already) to nearest even. A value that may be filtered
    // keeps the `OTHER` arm, but a small float's, the half-float family's.
    let scale: Table<'_> = &|fb, s| {
        let s = s.unwrap_or([1; 4]).map(|v| grid_bits(v).0);
        lanes_of(fb, f, s)
    };
    let reciprocal: Table<'_> = &|fb, s| {
        let s = s.unwrap_or([1; 4]).map(|v| grid_bits(v).1);
        lanes_of(fb, f, s)
    };
    let packed_field = PackedField::read(fb, flags, channel)?;
    let scales = packed_field.lanes(fb, scale)?;
    let reciprocals = packed_field.lanes(fb, reciprocal)?;
    let texel_lanes = u1(fb, TEXEL_LANES)?;
    let single = fb.binary(BinOp::ULessThanEqual, channel, texel_lanes)?;
    let coded = fb.binary(BinOp::LogicalAnd, packed_field.coded, single)?;
    let small = fb.binary(BinOp::LogicalAnd, packed_field.small, single)?;
    // A YCbCr format's one texel (a read or a sample of one texel) and a 2D gather's lanes, each
    // one texel's channel: its value to nearest with ties away from zero, as a half-float texel's
    // sample.
    let ycbcr = fb.binary(BinOp::LogicalAnd, packed_field.ycbcr, single)?;
    let d = fb.select(coded, scales, d)?;
    let rcp = fb.select(coded, reciprocals, rcp)?;
    // The step `n` on the grid, and `|n| / D` with the sign of `n` (`+0` for `n = 0`).
    let neg_d = fb.unary(UnOp::FNegate, d)?;
    let n = texel_round::grid_step(fb, f, x, d, neg_d)?;
    let z = splat(fb, u, 0)?;
    let negative = fb.binary(BinOp::SLessThan, n, z)?;
    let m = fb.glsl(GlslOp::SAbs, &[n])?;
    let mf = fb.convert(ConvOp::ConvertUToF, m, ft)?;
    let product = fb.binary_no_contraction(BinOp::FMul, mf, rcp)?;
    let (b15, b16) = (u1(fb, 15)?, u1(fb, 16)?);
    let b = fb.select(signed, b15, b16)?;
    let b = fb.composite_construct(vu, &[b; 4])?;
    let ones = texel_round::ones_ratio_to_half(fb, u, m, b)?;
    let sixteen = fb.binary(BinOp::LogicalAnd, norm_arm, wide)?;
    let value = fb.select(sixteen, ones, product)?;
    let bits = fb.convert(ConvOp::Bitcast, value, vu)?;
    // The step's `half`, to nearest even by integer arithmetic on the bits: a nonzero step's
    // product lies in `[2^-12, 1]`, within the normal halves, where the `half` keeps the top 11 of
    // the `f32`'s 24 significand bits, so adding `0xFFF` plus the lowest kept bit and clearing the
    // 13 bits below rounds it once; a carry raises the exponent, as it must. The 16-bit grids'
    // value is a `half` already (normal or denormal), whose 13 low bits are clear, so it is kept; a
    // zero step's `+0` too. No conversion: nothing for a device to fold, so the grid families need
    // no barrier (`narrow::round_texel`).
    let (k13, one) = (splat(fb, u, 13)?, splat(fb, u, 1)?);
    let lsb = fb.binary(BinOp::ShiftRightLogical, bits, k13)?;
    let lsb = fb.binary(BinOp::BitwiseAnd, lsb, one)?;
    let half_ulp = splat(fb, u, 0xFFF)?;
    let bias = fb.binary(BinOp::IAdd, half_ulp, lsb)?;
    let bits = fb.binary(BinOp::IAdd, bits, bias)?;
    let keep = splat(fb, u, 0xFFFF_E000)?;
    let bits = fb.binary(BinOp::BitwiseAnd, bits, keep)?;
    let sign = splat(fb, u, 0x8000_0000)?;
    let sign = fb.select(negative, sign, z)?;
    let bits = fb.binary(BinOp::BitwiseOr, bits, sign)?;
    let on_grid = fb.convert(ConvOp::Bitcast, bits, ft)?;
    let grid = fb.binary(BinOp::LogicalOr, coded, norm_arm)?;
    let own = fb.unary(UnOp::LogicalNot, float_or_half)?;
    let grid = fb.binary(BinOp::LogicalAnd, grid, own)?;
    // Every other family's texel to nearest even, once; a sample of a 32-bit float texel toward
    // zero saturating, of a half-float texel to nearest with ties away. The grid's `half` is
    // selected after it, so a device that specializes a grid class drops the rounding of `x`
    // whole.
    let rounded = match fb.module().quantize_form() {
        QuantizeForm::Device | QuantizeForm::Converted | QuantizeForm::Controlled => {
            // The module's rounding is Metal's here, and is every other family's, the `OTHER`
            // arm's among them, so a sample's directed rounding follows from `even`.
            let even = narrow::round_texel(fb, hform(), x)?;
            let sampled = texel_round::f32_to_half_from_even(fb, f, (x, even), is_float)?;
            let takes = fb.binary(BinOp::LogicalAnd, sample, float_or_half)?;
            let takes = fb.binary(BinOp::LogicalOr, takes, ycbcr)?;
            fb.select(takes, sampled, even)?
        }
        QuantizeForm::Integer => {
            let toward_zero = fb.binary(BinOp::LogicalAnd, sample, is_float)?;
            let away = fb.binary(BinOp::LogicalAnd, sample, is_half)?;
            let away = fb.binary(BinOp::LogicalOr, away, ycbcr)?;
            let rounded = texel_round::f32_to_half_by(fb, f, x, (toward_zero, away))?;
            // A format whose conversion is not measured: every other `half` texel's exact
            // rounding, noted.
            let other = narrow::round_texel(fb, hform(), x)?;
            let measured = fb.binary(BinOp::LogicalOr, float_or_half, small)?;
            let measured = fb.binary(BinOp::LogicalOr, measured, ycbcr)?;
            fb.select(measured, rounded, other)?
        }
    };
    fb.select(grid, on_grid, rounded)
}

/// [`body`] on a device whose texel is Metal's `float` result and whose rounding to nearest even
/// is Metal's ([`HalfFamilies::MetalSampler`]): the 16-bit normalized grids' exact `half`
/// ([`grid16_by_product`]), a 32-bit float sample toward zero and a YCbCr field's one texel on the
/// grid `8192` with ties away (`f32_to_half_from_even`), every other texel to nearest even
/// (`narrow::round_texel`). It reads the class's `LINEAR`, `SNORM` and `WIDE16` bits (a class
/// with `WIDE16` is `EXACT`'s, never packed) and the packed field's top bit
/// ([`TexelFormat::YCBCR_8192`]), and `channel` only where that bit is set.
fn metal_body(
    fb: &mut FunctionBuilder<'_>,
    x: SValue,
    flags: SValue,
    sample: SValue,
    channel: SValue,
) -> BResult<SValue> {
    let f = fform();
    let ft = ty(fb, f)?;
    let u1 = |fb: &mut FunctionBuilder<'_>, v: u32| splat(fb, uform(1), u64::from(v));
    let f1 = |fb: &mut FunctionBuilder<'_>, bits: u64| {
        let s = Form {
            held: Held::Float(FloatFmt::F32),
            lanes: 1,
        };
        splat(fb, s, bits)
    };
    let has = |fb: &mut FunctionBuilder<'_>, bit: u32| -> BResult<SValue> {
        let m = u1(fb, bit)?;
        let z = u1(fb, 0)?;
        let and = fb.binary(BinOp::BitwiseAnd, flags, m)?;
        fb.binary(BinOp::INotEqual, and, z)
    };
    let is_float = has(fb, BicubicTexels::LINEAR)?;
    let wide = has(fb, BicubicTexels::WIDE16)?;
    let signed = has(fb, BicubicTexels::SNORM)?;
    // The 16-bit grids: the step `n` on the grid, and the `half` of `n / D` (`+0` for `n = 0`).
    let (g16, _) = grid_bits(TexelFormat::GRID16);
    let (g16s, _) = grid_bits(TexelFormat::GRID16S);
    let (g16, g16s) = (f1(fb, g16)?, f1(fb, g16s)?);
    let d = fb.select(signed, g16s, g16)?;
    let d = fb.composite_construct(ft, &[d; 4])?;
    let neg_d = fb.unary(UnOp::FNegate, d)?;
    let on_grid = match fb.module().native_half() {
        false => grid16_by_product(fb, x, (d, neg_d), signed)?,
        true => grid16_by_integers(fb, x, (d, neg_d), signed)?,
    };
    // A YCbCr field's one texel on the grid `8192`: ties away from zero.
    let ycbcr = has(fb, TexelFormat::YCBCR_8192)?;
    let texel_lanes = u1(fb, TEXEL_LANES)?;
    let single = fb.binary(BinOp::ULessThanEqual, channel, texel_lanes)?;
    let ycbcr = fb.binary(BinOp::LogicalAnd, ycbcr, single)?;
    let even = narrow::round_texel(fb, hform(), x)?;
    let sampled = texel_round::f32_to_half_from_even(fb, f, (x, even), is_float)?;
    let takes = fb.binary(BinOp::LogicalAnd, sample, is_float)?;
    let takes = fb.binary(BinOp::LogicalOr, takes, ycbcr)?;
    let rounded = fb.select(takes, sampled, even)?;
    fb.select(wide, on_grid, rounded)
}

/// The `half` of a 16-bit normalized texel `x` (four `f32` lanes) on its grid `D` (`d` and
/// `neg_d`, `65535`, or `32767` where `signed`): its step `n` (`texel_round::grid_round`), and `n
/// / D` correctly rounded to `half` with the sign of `n`, `+0` for `n = 0`, as one product. The
/// magnitude `|n| · c`, for the grid's factor `c` (`texel_round::ones_ratio_factor`: every step's
/// product rounds to the `half` of its quotient, either way it rounds to `f32`), rounded to `half`
/// by the module's rounding ([`narrow::round`]: the device's conversion behind its barrier, or its
/// quantize), which a module whose rounding is measured to be Metal's takes as exact. The sign is
/// put back by a test of `n` below `-1/2`, so that a `-0` step gives `+0`. Every operation is a
/// float one: on MoltenVK, Metal's compiler takes about 2 s less over a function converting 311
/// such texels than over [`grid16_by_integers`]'s integer multiplication, leading-bit search and
/// shifts (the M3 Max, macOS 27.0.1, MoltenVK's MSL of a guest's 270,000-word fragment function).
fn grid16_by_product(
    fb: &mut FunctionBuilder<'_>,
    x: SValue,
    (d, neg_d): (SValue, SValue),
    signed: SValue,
) -> BResult<SValue> {
    let f = fform();
    let ft = ty(fb, f)?;
    let factor = |fb: &mut FunctionBuilder<'_>, b: u32| {
        let s = Form {
            held: Held::Float(FloatFmt::F32),
            lanes: 1,
        };
        splat(
            fb,
            s,
            u64::from(texel_round::ones_ratio_factor(b).to_bits()),
        )
    };
    let (c15, c16) = (factor(fb, 15)?, factor(fb, 16)?);
    let c = fb.select(signed, c15, c16)?;
    let c = fb.composite_construct(ft, &[c; 4])?;
    let n = texel_round::grid_round(fb, x, d, neg_d)?;
    let m = fb.glsl(GlslOp::FAbs, &[n])?;
    let p = fb.binary_no_contraction(BinOp::FMul, m, c)?;
    let below = fsplat(fb, f, -0.5)?;
    let negative = fb.binary(BinOp::FOrdLessThan, n, below)?;
    let minus = fb.unary(UnOp::FNegate, p)?;
    let v = fb.select(negative, minus, p)?;
    narrow::round(fb, hform(), v)
}

/// [`grid16_by_product`]'s `half` by integer arithmetic alone (`texel_round::ones_ratio_to_half`):
/// the 16-bit grids' value where the module's rounding is the device's own (native half rounding),
/// so that their exact `half` does not take it.
fn grid16_by_integers(
    fb: &mut FunctionBuilder<'_>,
    x: SValue,
    (d, neg_d): (SValue, SValue),
    signed: SValue,
) -> BResult<SValue> {
    let f = fform();
    let u = uform(4);
    let ft = ty(fb, f)?;
    let vu = ty(fb, u)?;
    let u1 = |fb: &mut FunctionBuilder<'_>, v: u32| splat(fb, uform(1), u64::from(v));
    let n = texel_round::grid_step(fb, f, x, d, neg_d)?;
    let z = splat(fb, u, 0)?;
    let negative = fb.binary(BinOp::SLessThan, n, z)?;
    let m = fb.glsl(GlslOp::SAbs, &[n])?;
    let (b15, b16) = (u1(fb, 15)?, u1(fb, 16)?);
    let b = fb.select(signed, b15, b16)?;
    let b = fb.composite_construct(vu, &[b; 4])?;
    let ones = texel_round::ones_ratio_to_half(fb, u, m, b)?;
    let bits = fb.convert(ConvOp::Bitcast, ones, vu)?;
    let sign = splat(fb, u, 0x8000_0000)?;
    let sign = fb.select(negative, sign, z)?;
    let bits = fb.binary(BinOp::BitwiseOr, bits, sign)?;
    fb.convert(ConvOp::Bitcast, bits, ft)
}

/// The held `half` texel of image texel `x` (a `vec4` of `f32`), converted by `routine` under the
/// slot's treatment constant `flags` (its `TexelFormat` specialization), the access's `form`, and
/// `lanes`, its [`TexelLanes::code`] (a constant, or the shader's choice where a runtime
/// sampler's word decides whether a sample reads one texel).
pub fn convert(
    fb: &mut FunctionBuilder<'_>,
    routine: FuncId,
    x: SValue,
    flags: SValue,
    (form, lanes): (HalfForm, SValue),
) -> BResult<SValue> {
    let b = fb.module().ty(SType::Bool)?;
    let c = fb.module().constant(Constant::Scalar {
        ty: b,
        bits: ConstBits(u64::from(form == HalfForm::Sample)),
    })?;
    let sample = fb.module().const_value(c)?;
    // Every call of the routine returns its texel.
    fb.call(routine, &[x, flags, sample, lanes])?
        .ok_or(BuildError::Operand {
            op: crate::spv::Op::FunctionCall,
            rule: OperandRule::Callee,
        })
}
