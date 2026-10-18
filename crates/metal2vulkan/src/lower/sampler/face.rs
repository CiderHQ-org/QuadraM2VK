//! The cube face and texel Metal reads along a direction.
//!
//! A cube sample, gather, or level-of-detail query reads the face of its direction's major axis,
//! the component of largest magnitude, at the face coordinates `s_c / |m_a|` and `t_c / |m_a|`.
//! Vulkan leaves two things there to the device that Metal decides its own way (Apple M4, macOS
//! 27.0.1, rgba32_float cubes of 8x8 and 5x5 faces, nearest samples; each run with and without
//! Metal's validation layer, byte-identical):
//!
//! - **The face at a tie.** Where two or three components share the largest magnitude, Vulkan's
//!   "Cube Map Face Selection" only recommends `z` over `y` over `x`; Metal picks `x` over `y`
//!   over `z`. Every one of 834 lanes, each pair and the triple of tied axes in every sign
//!   combination at magnitudes 1, 0.885, 0.5 and 3, read that face, and each control with one
//!   tied component an ulp larger read that component's face; so did ties at `2^-125`, `2^-124`,
//!   `2^-100`, `10^-30`, and `2^100`. Authored case `914ba505`'s lane 239, `(-1, -0.885, -1)`,
//!   read the `-z` face on five Vulkan devices where Metal reads `-x`.
//! - **The ends of the range.** Metal reads a subnormal component as zero when it selects the
//!   face, and a direction left zero, every component zero or subnormal in any signs (magnitudes
//!   `2^-149`, `2^-140`, `2^-127`, the largest subnormal), at face `+x`'s texel `(0, 0)`, where
//!   Vulkan divides by zero. It reads a face coordinate as zero where its component is below
//!   `2^-125` (a minor `2^-126`, `1.5 * 2^-126`, or a subnormal one beside a major `2^-125` or
//!   `2^-120` read the face's centre texel, and a tie at `2^-126` or the next float above it too)
//!   or where the major magnitude is above `2^126` (ties at `2^127` and at the largest float read
//!   the centre; `2^125` beside `2^126` reads exactly).
//!
//! `metal_ties` hands the device a direction that reads Metal's face on every device, and its
//! texel wherever the device rounds the face coordinate as Metal does (below), Apple's texture
//! unit (which MoltenVK samples through) included: each minor component Metal
//! reads as zero is `+0`; a direction left zero is `(1, 1 - 2^-24, 1 - 2^-24)`, which reads face
//! `+x` at texel `(0, 0)` on every face size below `2^24`; and each minor component tying the
//! major magnitude otherwise is one ulp smaller in magnitude (its bits less one), so no tie is
//! left for a device to break. At a tie at `2^-125` itself the shrunk component would fall below
//! `2^-125`, where Apple's texture unit reads it as zero (MoltenVK read the face's centre there),
//! so the major component grows one ulp in magnitude instead (its bits plus
//! one). On Metal's face a tied component is the edge coordinate `±1`, which becomes `±(1 - e)`
//! with `e <= 2^-23` either way: the texel the edge reads through a nearest sample on every face
//! size below `2^23`, and the footprint it reads through a gather or a linear filter on every size
//! below `2^15` (the position quantizes to 1/256 of a texel), above Metal's largest cube face,
//! 16,384. Every other component passes unchanged. A gather or linear sample at a zero direction,
//! like one aimed at any corner, reads a corner texel Vulkan leaves to the device
//! (`PrecisionNote::DeviceGatherFootprint`, `PrecisionNote::FloatSampleFilter`).
//!
//! **A huge major magnitude.** Devices misread a direction whose major magnitude is near the top
//! of the range (the probe lanes, `wrg_cube_ties` at 8x8 and 5x5): RADV (AMD Radeon 780M)
//! read `(2^126, 2^125, 0)` at the face's centre, where Metal reads `t = 1/4`, as a reciprocal of
//! AMD's doubled major axis (`v_cubema`) flushed to zero would; ANV (Intel Iris Xe) and NVIDIA
//! (RTX 5080 laptop) read face `+z` along every direction with a major magnitude of `2^127` or
//! more (`(2^127, 0, 0)` and `(f32::MAX, 0, 0)` among them), and the major face at `2^126`. So a
//! direction whose major magnitude is above `2^125` is scaled by a power of two, `1/2` up to
//! `2^126` and `1/8` past it, where every minor component is already `+0`: the device divides by
//! at most `2^125`, and no quotient moves, since every component scaled by `1/2` that Metal reads
//! (at least `2^-125`) stays normal.
//!
//! **The texel on the face** (`snapped`). Along `(1, 1 + 2^-23, 0.25)` (a tie's control) the
//! quotient `0.25 / (1 + 2^-23)` is `1/4 - 2^-25` and the face coordinate `(q + 1) / 2` just below
//! `0.625`. Metal reads texel 5 of an 8x8 face there, `0.625 * 8`, as RADV and lavapipe do; ANV and
//! NVIDIA read texel 4, as keeping the bits below would, on all 24 such lanes. Metal's position on
//! a face axis of size `n` (the four cube rounding probes, Apple M4 on macOS 27.0.1 and Apple M3
//! Max on macOS 26.7, byte-identical, each run with and without Metal's validation layer): with `r`
//! its texture unit's reciprocal of the major magnitude `|m_a|` and `s_c` the minor component as
//! the face orients it, `F = RN(1 + s_c * r)`, fused and rounded once with ties to even, and `X =
//! RN(F / 2 * n)`; a nearest sample reads `floor(X)`, at most `n - 1`, and a gather or linear
//! filter quantizes `X` as in 2D, `floor(256 X + 1/2)` (860 of 868 interior lanes; the other 8 are
//! not explained). Twelve other orders fit worse (the quotient first, an unfused sum, a fused
//! scale, an exact quotient, among others), and rounding the sum's ties away from zero misses 2,112
//! of 6,336 tie coordinates, toward zero 1,812 of 5,436, all of which ties to even reads. The
//! texture unit's `r` is not correctly rounded: it equals `RN(1 / |m_a|)` on 1,793 of 2,001 random
//! majors, and lies above it on 204 and below on 4, by one ulp or two (the lanes tell those apart
//! from `RN(1 / |m_a|)`, not from each other), each a major whose reciprocal lies within about 0.2
//! ulp of a rounding midpoint; the shader's own `1.0f / m_a` and `precise::divide` are `RN(1 /
//! |m_a|)` on all of them. It is Metal's fast reciprocal, whose model (`libm::Fun::Recip`) reads
//! Metal's texel on 1,992 of those 2,001 lanes, on the M3 Max and the M4 alike; no exact form of it
//! is known.
//!
//! **A repeating face axis** ([`FaceWrap`]). Metal reads every other address mode of a cube's
//! sampler as clamping, but `repeat` on a face axis whose size is not a power of two wraps that
//! axis's face coordinate `F / 2` of exactly 1 (a minor component equal to the major magnitude,
//! after the reciprocal) to 0, so `X = 0`: the face's other edge. The sampler's `s` address governs
//! the face's `s` axis and its `t` address the `t` axis; its `r` address governs neither, and a
//! size that is a power of two never wraps (our cube probes, Apple M4 on macOS 27.0.1: gathers,
//! linear samples, and nearest samples of identity cubes of 3, 5, and 6 texels wrap on every such
//! lane under `repeat` on both axes, on the `s` axis alone under `s_address::repeat`, and on the
//! `t` axis alone under `t_address::repeat`, and of 4 and 8 texels on none; `mirrored_repeat`,
//! `clamp_to_zero`, and a border read as clamping everywhere). A tied
//! minor component, which `metal_ties` shrinks by an ulp, still sums to `F = 2` (`1 + (1 - 2^-24)`
//! rounds to 2 with ties to even), so it wraps as Metal's tie does.
//!
//! So a nearest sample through a constexpr normalized sampler that reads the base level, on a
//! device without `VkProperty::MetalSampler`, hands the device the direction whose major component
//! is `±1` and whose minor components are the centres of the texels Metal's position names with
//! `r` Metal's reciprocal's model (`libm::Fun::Recip`, or the device's `1 / |m_a|` where that is
//! Metal's) and the fused multiply-add, wrapped on a repeating face axis ([`FaceWrap`]): each
//! `(2i + 1) / n - 1`, oriented as Vulkan's face table orients it, which every device reads at the
//! face coordinate `(i + 1/2) / n`, half a texel from either edge. Where the model misses Apple's
//! reciprocal (by one ulp, on about 0.3 % of significands) and `X` lies within about `n * 2^-23` of
//! a texel edge, Metal reads the next texel, which the module records
//! (`PrecisionNote::CubeFaceReciprocal`). A linear sample that Metal's filter answers takes the
//! same position (`position`) and reads its taps by fetches, across a face's edges through
//! `define_fold` (`linear::cube`), under the same note, and so does a cube gather (a depth cube's
//! comparison included), which reads the same four taps in gather order, through a constexpr
//! sampler or a runtime sampler slot, whose word's address modes wrap the face axes
//! (`Wraps::Word`). A nearest sample's comparison, a runtime or argument buffer's sampler, and
//! another level keep the device's position; on `VkProperty::MetalSampler` the sample is Metal's
//! own.
//!
//! A direction with an infinite or NaN component passes unchanged, and the device's answer
//! stands: no infinity becomes finite and no NaN moves. Metal's is measured and not reproduced:
//! it reads a face coordinate of `±inf / inf` at texel 0 and `(inf, 0, 0)` at the centre of `+x`;
//! its face is the first axis whose magnitude equals the largest non-NaN one (`z` where none
//! does), and a NaN face coordinate reads texel 0. A level of detail the device takes from the
//! direction's derivatives sees each broken tie as a face coordinate moved by at most `2^-24`
//! (`PrecisionNote::CubeGradientFootprint`).

use crate::error::Error;
use crate::lower::arith::fma::{self, Fma};
use crate::lower::arith::legal::{FloatFmt, Form, Held};
use crate::lower::arith::libm;
use crate::lower::emit::EmitCx;
use crate::lower::select::SelectCx;
use crate::metal::sampler::SamplerWord;
use crate::pipeline::artifact::PrecisionNote;
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::{BResult, BuildError, TypeRule};
use crate::spirv::ids::{FuncId, Value as SValue};
use crate::spirv::ir::{BinOp, ConstBits, Constant, ConvOp, GlslOp, Type as SType, UnOp};

use super::bicubic::TapAddress;

/// The estimated words of `metal_ties`: its 89 instructions and 11 constants, at most 6 words each.
pub const TIE_WORDS: u32 = 600;

/// `1 - 2^-24`, the largest float below 1: a zero direction's `y` and `z`.
const BELOW_ONE: u32 = 0x3f7f_ffff;

/// `2^-125`: Metal reads a face coordinate's component below it as zero.
const LEAST_READ: u32 = 0x0100_0000;

/// `2^126`: past a major magnitude above it Metal reads every face coordinate as zero.
const MOST_READ: u32 = 0x7e80_0000;

/// `2^125`: the largest major magnitude the device is handed. AMD's cube instructions take the
/// major axis doubled (`v_cubema`), whose reciprocal is subnormal, and flushed, past it.
const MOST_DIVIDED: u32 = 0x7e00_0000;

/// The cube direction `p` (three floats) as Metal reads it, handed to the device so that every
/// device reads Metal's face and texel. A direction with an infinite or NaN component passes
/// unchanged. Otherwise Metal's major axis is the first of `x`, `y`, `z` whose magnitude, a
/// subnormal one read as zero, is the largest; a direction left zero is handed as
/// `(1, 1 - 2^-24, 1 - 2^-24)` (face `+x`, texel `(0, 0)`); each minor component is `+0` where
/// its magnitude is below `2^-125` or the major magnitude above `2^126`, else one ulp smaller in
/// magnitude where it ties the major magnitude; but at a tie at `2^-125` the major component is
/// one ulp larger instead; every other component passes unchanged. Then a direction whose major
/// magnitude is above `2^125` is scaled by `1/2`, or by `1/8` where it is above `2^126`.
pub(super) fn metal_ties(fb: &mut FunctionBuilder<'_>, p: SValue) -> BResult<SValue> {
    let f32_ = fb.module().ty(SType::Float { width: 32 })?;
    let u32_ = fb.module().ty(SType::Int {
        width: 32,
        signed: false,
    })?;
    let triple = fb.module().ty(SType::Vector { elem: f32_, len: 3 })?;
    let f32_const = |fb: &mut FunctionBuilder<'_>, x: f32| -> BResult<SValue> {
        let c = fb.module().constant(Constant::Scalar {
            ty: f32_,
            bits: ConstBits(u64::from(x.to_bits())),
        })?;
        fb.module().const_value(c)
    };
    let least = f32_const(fb, f32::MIN_POSITIVE)?;
    let most = f32_const(fb, f32::MAX)?;
    let least_read = f32_const(fb, f32::from_bits(LEAST_READ))?;
    let most_read = f32_const(fb, f32::from_bits(MOST_READ))?;
    let most_divided = f32_const(fb, f32::from_bits(MOST_DIVIDED))?;
    let zero = f32_const(fb, 0.0)?;
    let a = fb.glsl(GlslOp::FAbs, &[p])?;
    let cx = fb.composite_extract(p, &[0])?;
    let cy = fb.composite_extract(p, &[1])?;
    let cz = fb.composite_extract(p, &[2])?;
    let ax = fb.composite_extract(a, &[0])?;
    let ay = fb.composite_extract(a, &[1])?;
    let az = fb.composite_extract(a, &[2])?;
    // Every component finite (an infinity or a NaN fails the test): the direction is changed at
    // all only then.
    let finite_x = fb.binary(BinOp::FOrdLessThanEqual, ax, most)?;
    let finite_y = fb.binary(BinOp::FOrdLessThanEqual, ay, most)?;
    let finite_z = fb.binary(BinOp::FOrdLessThanEqual, az, most)?;
    let finite_xy = fb.binary(BinOp::LogicalAnd, finite_x, finite_y)?;
    let regular = fb.binary(BinOp::LogicalAnd, finite_xy, finite_z)?;
    // The magnitude Metal selects the face by: a subnormal one as zero.
    let selecting = |fb: &mut FunctionBuilder<'_>, ak: SValue| -> BResult<SValue> {
        let subnormal = fb.binary(BinOp::FOrdLessThan, ak, least)?;
        fb.select(subnormal, zero, ak)
    };
    let mx = selecting(fb, ax)?;
    let my = selecting(fb, ay)?;
    let mz = selecting(fb, az)?;
    // Metal's major axis: `x` over `y` over `z` at equal magnitudes.
    let x_y = fb.binary(BinOp::FOrdGreaterThanEqual, mx, my)?;
    let x_z = fb.binary(BinOp::FOrdGreaterThanEqual, mx, mz)?;
    let major_x = fb.binary(BinOp::LogicalAnd, x_y, x_z)?;
    let y_z = fb.binary(BinOp::FOrdGreaterThanEqual, my, mz)?;
    let z_y = fb.binary(BinOp::FOrdGreaterThan, mz, my)?;
    let minor_x = fb.unary(UnOp::LogicalNot, major_x)?;
    let minor_y = fb.binary(BinOp::LogicalOr, major_x, z_y)?;
    let minor_z = fb.binary(BinOp::LogicalOr, major_x, y_z)?;
    let yz = fb.select(y_z, my, mz)?;
    let top = fb.select(major_x, mx, yz)?;
    // Past a major magnitude above `2^126` Metal reads every face coordinate as zero.
    let beyond = fb.binary(BinOp::FOrdGreaterThan, top, most_read)?;
    // Whether Metal reads a minor component of magnitude `ak` as zero.
    let zeroed = |fb: &mut FunctionBuilder<'_>, ak: SValue, minor: SValue| -> BResult<SValue> {
        let small = fb.binary(BinOp::FOrdLessThan, ak, least_read)?;
        let drop = fb.binary(BinOp::LogicalOr, small, beyond)?;
        let dropped = fb.binary(BinOp::LogicalAnd, minor, drop)?;
        fb.binary(BinOp::LogicalAnd, regular, dropped)
    };
    let zeroed_x = zeroed(fb, ax, minor_x)?;
    let zeroed_y = zeroed(fb, ay, minor_y)?;
    let zeroed_z = zeroed(fb, az, minor_z)?;
    // A minor component tying the major magnitude: `y` beside a major `x`, `z` beside either.
    let tie = |fb: &mut FunctionBuilder<'_>, ak: SValue, beside: SValue| -> BResult<SValue> {
        let equal = fb.binary(BinOp::FOrdEqual, ak, top)?;
        let tied = fb.binary(BinOp::LogicalAnd, beside, equal)?;
        fb.binary(BinOp::LogicalAnd, regular, tied)
    };
    let tie_y = tie(fb, ay, major_x)?;
    let tie_z = tie(fb, az, minor_z)?;
    // A tied component shrinks where its magnitude stays at least `2^-125`. At a tie at `2^-125`
    // itself it would fall below, where Apple's texture unit reads it as zero (MoltenVK samples
    // there), so the major component grows one ulp in magnitude instead.
    let above = fb.binary(BinOp::FOrdGreaterThan, top, least_read)?;
    let shrink_y = fb.binary(BinOp::LogicalAnd, tie_y, above)?;
    let shrink_z = fb.binary(BinOp::LogicalAnd, tie_z, above)?;
    let at_least = fb.binary(BinOp::FOrdEqual, top, least_read)?;
    let any_tie = fb.binary(BinOp::LogicalOr, tie_y, tie_z)?;
    let grow = fb.binary(BinOp::LogicalAnd, at_least, any_tie)?;
    let grow_x = fb.binary(BinOp::LogicalAnd, grow, major_x)?;
    let major_y = fb.binary(BinOp::LogicalAnd, minor_x, y_z)?;
    let grow_y = fb.binary(BinOp::LogicalAnd, grow, major_y)?;
    let one = fb.u32_const(1)?;
    let one = fb.module().const_value(one)?;
    // A normal float's bits less one are its magnitude one ulp smaller, and its bits plus one
    // (below the largest float) one ulp larger, its sign kept either way.
    let step = |fb: &mut FunctionBuilder<'_>, bits: SValue, op: BinOp| -> BResult<SValue> {
        let stepped = fb.binary(op, bits, one)?;
        fb.convert(ConvOp::Bitcast, stepped, f32_)
    };
    let bits_x = fb.convert(ConvOp::Bitcast, cx, u32_)?;
    let bits_y = fb.convert(ConvOp::Bitcast, cy, u32_)?;
    let bits_z = fb.convert(ConvOp::Bitcast, cz, u32_)?;
    let grown_x = step(fb, bits_x, BinOp::IAdd)?;
    let x = fb.select(grow_x, grown_x, cx)?;
    let x = fb.select(zeroed_x, zero, x)?;
    let grown_y = step(fb, bits_y, BinOp::IAdd)?;
    let shrunk_y = step(fb, bits_y, BinOp::ISub)?;
    let y = fb.select(grow_y, grown_y, cy)?;
    let y = fb.select(shrink_y, shrunk_y, y)?;
    let y = fb.select(zeroed_y, zero, y)?;
    let shrunk_z = step(fb, bits_z, BinOp::ISub)?;
    let z = fb.select(shrink_z, shrunk_z, cz)?;
    let z = fb.select(zeroed_z, zero, z)?;
    // A direction left zero reads face `+x`'s texel `(0, 0)`.
    let empty = fb.binary(BinOp::FOrdEqual, top, zero)?;
    let nothing = fb.binary(BinOp::LogicalAnd, regular, empty)?;
    let unit = f32_const(fb, 1.0)?;
    let below_one = f32_const(fb, f32::from_bits(BELOW_ONE))?;
    let x = fb.select(nothing, unit, x)?;
    let y = fb.select(nothing, below_one, y)?;
    let z = fb.select(nothing, below_one, z)?;
    // A major magnitude past `2^125` is scaled down by a power of two, `1/2` up to `2^126` and
    // `1/8` past it (every minor component is `+0` there), so that the device divides by at most
    // `2^125`; no quotient moves.
    let huge = fb.binary(BinOp::FOrdGreaterThan, top, most_divided)?;
    let huge = fb.binary(BinOp::LogicalAnd, regular, huge)?;
    let half = f32_const(fb, 0.5)?;
    let eighth = f32_const(fb, 0.125)?;
    let scale = fb.select(beyond, eighth, half)?;
    let scaled = |fb: &mut FunctionBuilder<'_>, c: SValue| -> BResult<SValue> {
        let product = fb.binary(BinOp::FMul, c, scale)?;
        fb.select(huge, product, c)
    };
    let x = scaled(fb, x)?;
    let y = scaled(fb, y)?;
    let z = scaled(fb, z)?;
    fb.composite_construct(triple, &[x, y, z])
}

/// The estimated words of `snapped` beside its reciprocal's use and its two fused multiply-adds'
/// (`Snap::words` adds those): its 70 instructions and 6 constants, at most 6 words each.
pub const SNAP_WORDS: u32 = 480;

/// The routines `snapped` computes Metal's face position with: Metal's reciprocal
/// (`libm::Fun::Recip`) by its function `recip`, or by the device's `1 / y` where that is Metal's
/// (`None`), and the correctly rounded fused multiply-add of `fma`, by its function `fma_routine`
/// where its route has one.
#[derive(Clone, Copy, Debug)]
pub struct SnapRoutines {
    pub recip: Option<FuncId>,
    pub fma: Fma,
    pub fma_routine: Option<FuncId>,
}

/// The `float` scalar form the routines take.
pub(super) const SCALAR: Form = Form {
    held: Held::Float(FloatFmt::F32),
    lanes: 1,
};

/// The routines Metal's face position takes on a profile ([`SnapRoutines`] before the module
/// declares them): Metal's reciprocal, which the arith family plans and declares with the math
/// library's routines (`libm::Fun::Recip`), and the fused multiply-add `fma`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FaceRoutines {
    pub fma: Fma,
}

impl FaceRoutines {
    /// The census words a position adds beside its own: the reciprocal's use (its routine's
    /// call) and the two fused multiply-adds'.
    pub(super) fn use_words(self) -> u32 {
        libm::use_words(1, 1) + 2 * self.fma.words()
    }

    /// The declared routines, as emission calls them.
    pub(super) fn declared(self, cx: &mut EmitCx<'_>) -> BResult<SnapRoutines> {
        let recip = match crate::lower::arith::lib(cx).device(libm::Fun::Recip) {
            true => None,
            false => match cx.decl.families.arith.libm.get(&libm::Fun::Recip) {
                Some(id) => Some(*id),
                None => return Err(cx.fail("a cube position whose reciprocal is not declared")),
            },
        };
        let fma_routine = match self.fma.routine() {
            Some(r) => match cx.decl.families.arith.fmas.get(&r) {
                Some(id) => Some(*id),
                None => {
                    return Err(cx.fail("a cube position whose fused multiply-add is not declared"))
                }
            },
            None => None,
        };
        Ok(SnapRoutines {
            recip,
            fma: self.fma,
            fma_routine,
        })
    }
}

/// Which face axes of a sample's cube wrap a face coordinate of exactly 1 to 0 where the face size
/// is not a power of two (module docs): those its constexpr sampler addresses by `repeat`, the
/// face's `s` axis by the sampler's `s` address and its `t` axis by its `t` address.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct FaceWrap {
    pub s: bool,
    pub t: bool,
}

impl FaceWrap {
    /// The wrap of a sampler whose `s` and `t` axes address their taps by `s` and `t` (`None`: an
    /// axis the sample does not address, which never wraps).
    pub(super) fn of(s: Option<TapAddress>, t: Option<TapAddress>) -> FaceWrap {
        FaceWrap {
            s: s == Some(TapAddress::Repeat),
            t: t == Some(TapAddress::Repeat),
        }
    }

    /// The census words the wrap adds to a position: the size's power-of-two test, and per wrapping
    /// axis its comparison, condition, subtraction, and select, with their constants.
    pub(super) fn words(self) -> u32 {
        match (self.s, self.t) {
            (false, false) => 0,
            (true, true) => WRAP_WORDS + 2 * WRAP_AXIS_WORDS,
            _ => WRAP_WORDS + WRAP_AXIS_WORDS,
        }
    }
}

/// The face axes a cube sample's sampler repeats: a constexpr sampler's ([`FaceWrap`]), or a
/// runtime sampler's, whose word's `s` and `t` address modes the function reads.
#[derive(Clone, Copy)]
pub(super) enum Wraps {
    Axes(FaceWrap),
    Word(SValue),
}

/// The census words [`Wraps::Word`] adds: both axes' wraps ([`FaceWrap::words`]) and each axis's
/// mode, extracted from the word and tested, with their constants.
pub(super) const WORD_WRAP_WORDS: u32 = WRAP_WORDS + 2 * WRAP_AXIS_WORDS + 64;

/// [`FaceWrap::words`]' power-of-two test of the face size: three instructions and two constants.
const WRAP_WORDS: u32 = 32;

/// [`FaceWrap::words`] per wrapping axis: four instructions and a constant.
const WRAP_AXIS_WORDS: u32 = 32;

/// Per face axis whether it wraps a face coordinate of 1 here: the axes `wrap` names (a runtime
/// sampler's, those whose word's mode is `Repeat`), where the face size `n` (a `u32`) is not a
/// power of two.
fn wrapping(
    fb: &mut FunctionBuilder<'_>,
    n: SValue,
    wrap: Wraps,
) -> BResult<(Option<SValue>, Option<SValue>)> {
    if let Wraps::Axes(FaceWrap { s: false, t: false }) = wrap {
        return Ok((None, None));
    }
    let u = |fb: &mut FunctionBuilder<'_>, v: u32| -> BResult<SValue> {
        let c = fb.u32_const(v)?;
        fb.module().const_value(c)
    };
    let (zero, one) = (u(fb, 0)?, u(fb, 1)?);
    let less = fb.binary(BinOp::ISub, n, one)?;
    let low = fb.binary(BinOp::BitwiseAnd, n, less)?;
    let odd = fb.binary(BinOp::INotEqual, low, zero)?;
    match wrap {
        Wraps::Axes(w) => Ok((w.s.then_some(odd), w.t.then_some(odd))),
        Wraps::Word(word) => {
            let modes = u(fb, 0b111)?;
            let repeat = u(fb, crate::metal::SamplerAddressMode::Repeat.as_raw())?;
            let mut axes = [odd; 2];
            for (axis, shift) in axes.iter_mut().zip(SamplerWord::ADDRESS_SHIFT) {
                let sh = u(fb, shift)?;
                let mode = fb.binary(BinOp::ShiftRightLogical, word, sh)?;
                let mode = fb.binary(BinOp::BitwiseAnd, mode, modes)?;
                let repeats = fb.binary(BinOp::IEqual, mode, repeat)?;
                *axis = fb.binary(BinOp::LogicalAnd, odd, repeats)?;
            }
            let [s, t] = axes;
            Ok((Some(s), Some(t)))
        }
    }
}

/// The routines of a sample that takes Metal's face position ([`snapped`], [`position`]): Metal's
/// reciprocal (`libm::Fun::Recip`) and the profile's correctly rounded fused multiply-add, whose
/// module-scope routines the arith family declares. The reciprocal's model is Apple's texture
/// unit's on most majors, not all (module docs), which the module records
/// (`PrecisionNote::CubeFaceReciprocal`).
pub(super) fn plan(cx: &mut SelectCx<'_>) -> Result<FaceRoutines, Error> {
    // The sum `1 + sc * r` ([`texel_x`]) is computed between the sample's image and its use, which
    // stay in one block, so it takes no branch: `OpFmaKHR` unchecked ([`Fma::unchecked`]), which is
    // Metal's here. Its exact value is zero or at least `2^-48` in magnitude: a product near `-1`
    // is an integer below `2^48` times a power of two of at least `2^-48`, so the sum is a
    // multiple of that power (or, for a power above 1, at least 1).
    let Some(fma) = Fma::on(cx.profile, SCALAR).map(Fma::unchecked) else {
        return Err(cx.invariant("a cube position without a float fused multiply-add"));
    };
    cx.families.arith.libm_uses.insert(libm::Fun::Recip);
    cx.families.arith.take_fma(fma);
    cx.precision.insert(PrecisionNote::CubeFaceReciprocal);
    // Native device maths: the reciprocal is the device's `1 / |m_a|`.
    if cx.profile.fields().float_functions == crate::device::profile::FloatFunctions::Native {
        cx.precision.insert(PrecisionNote::NativeDivision);
    }
    Ok(FaceRoutines { fma })
}

/// Metal's reciprocal of the major magnitude `ma` (a `float`): the call of its routine, or the
/// device's `1 / ma` (`one` is `1.0`) where that is Metal's ([`SnapRoutines::recip`]).
fn reciprocal(
    fb: &mut FunctionBuilder<'_>,
    one: SValue,
    ma: SValue,
    routines: SnapRoutines,
) -> BResult<SValue> {
    match routines.recip {
        Some(f) => fb
            .call(f, &[ma])?
            .ok_or(BuildError::Type(TypeRule::FunctionSignature)),
        None => fb.binary(BinOp::FDiv, one, ma),
    }
}

/// A cube direction `p` (three floats, `metal_ties`'s) handed to the device so that a nearest
/// sample reads the texel Metal reads on a face of size `n` (a `u32`, the base level's), on every
/// device (module docs): the major component is `±1`, and each minor component the centre of
/// Metal's texel on its face axis, `(2i + 1) / n - 1`, oriented as Vulkan's face table orients
/// it, so that the device's face coordinate is `(i + 1/2) / n`. Metal's texel `i` on an axis is
/// `floor(X)`, at most `n - 1`, of its position `X` ([`texel_x`]), wrapped on the face axes `wrap`
/// names. A direction with an infinite or NaN component passes unchanged.
pub(super) fn snapped(
    fb: &mut FunctionBuilder<'_>,
    p: SValue,
    n: SValue,
    (routines, wrap): (SnapRoutines, FaceWrap),
) -> BResult<SValue> {
    let f32_ = fb.module().ty(SType::Float { width: 32 })?;
    let triple = fb.module().ty(SType::Vector { elem: f32_, len: 3 })?;
    let zero = f32_value(fb, 0.0)?;
    let one = f32_value(fb, 1.0)?;
    let nf = fb.convert(ConvOp::ConvertUToF, n, f32_)?;
    let last = fb.binary(BinOp::FSub, nf, one)?;
    let fr = frame(fb, p)?;
    let (wrap_s, wrap_t) = wrapping(fb, n, Wraps::Axes(wrap))?;
    // Metal's reciprocal of `|ma|`, its texture unit's on most majors (module docs).
    let r = reciprocal(fb, one, fr.ma, routines)?;
    // The centre of Metal's texel on one face axis, as the minor component handed on.
    let centre = |fb: &mut FunctionBuilder<'_>,
                  (flip, c): (SValue, SValue),
                  wrap: Option<SValue>|
     -> BResult<SValue> {
        let sc = oriented(fb, flip, c)?;
        let x = texel_x(fb, sc, (r, nf), (routines, wrap))?;
        let i = fb.glsl(GlslOp::Floor, &[x])?;
        let i = fb.glsl(GlslOp::FClamp, &[i, zero, last])?;
        let twice = fb.binary(BinOp::FAdd, i, i)?;
        let odd = fb.binary(BinOp::FAdd, twice, one)?;
        let from = fb.binary(BinOp::FSub, odd, nf)?;
        let at = fb.binary(BinOp::FDiv, from, nf)?;
        oriented(fb, flip, at)
    };
    let s = centre(fb, (fr.flip_s, fr.c_s), wrap_s)?;
    let t = centre(fb, (fr.flip_t, fr.c_t), wrap_t)?;
    let minus = f32_value(fb, -1.0)?;
    let unit = fb.select(fr.negative, minus, one)?;
    // `x` is `sc` beside a major `y` or `z`; `y` is `tc` beside a major `x` or `z`; `z` is `sc`
    // beside a major `x` and `tc` beside a major `y`.
    let x = fb.select(fr.major_x, unit, s)?;
    let y = fb.select(fr.major_y, unit, t)?;
    let st = fb.select(fr.major_x, s, t)?;
    let z = fb.select(fr.major_z, unit, st)?;
    let x = fb.select(fr.regular, x, fr.cx)?;
    let y = fb.select(fr.regular, y, fr.cy)?;
    let z = fb.select(fr.regular, z, fr.cz)?;
    fb.composite_construct(triple, &[x, y, z])
}

/// Metal's face of a cube direction and its position on each face axis ([`position`]).
pub(super) struct FacePosition {
    /// The face, `2 a + m` for the major axis `a` (`x`, `y`, `z`) and `m` 1 where the major
    /// component is negative: Vulkan's face order (`+x`, `-x`, `+y`, `-y`, `+z`, `-z`), a `u32`.
    pub face: SValue,
    /// The position `X` on the face's `s` axis, then on its `t` axis, `f32`s in `[0, n]`.
    pub s: SValue,
    pub t: SValue,
}

/// Metal's face of cube direction `p` (three floats, `metal_ties`'s) and its position `X` on each
/// axis of a face of size `n` (a `u32`, the base level's): the face of the major axis and its
/// sign, and on each face axis `X = RN(RN(1 + sc * r) / 2 * n)` ([`texel_x`]), wrapped on the
/// face axes `wrap` names, which a linear filter and a gather quantize as in 2D
/// (`floor(256 X + 1/2)`). A direction with an infinite or NaN component gives an unspecified face
/// and position.
pub(super) fn position(
    fb: &mut FunctionBuilder<'_>,
    p: SValue,
    n: SValue,
    (routines, wrap): (SnapRoutines, Wraps),
) -> BResult<FacePosition> {
    let f32_ = fb.module().ty(SType::Float { width: 32 })?;
    let one = f32_value(fb, 1.0)?;
    let nf = fb.convert(ConvOp::ConvertUToF, n, f32_)?;
    let fr = frame(fb, p)?;
    let (wrap_s, wrap_t) = wrapping(fb, n, wrap)?;
    let r = reciprocal(fb, one, fr.ma, routines)?;
    let sc = oriented(fb, fr.flip_s, fr.c_s)?;
    let s = texel_x(fb, sc, (r, nf), (routines, wrap_s))?;
    let tc = oriented(fb, fr.flip_t, fr.c_t)?;
    let t = texel_x(fb, tc, (r, nf), (routines, wrap_t))?;
    let u = |fb: &mut FunctionBuilder<'_>, v: u32| -> BResult<SValue> {
        let c = fb.u32_const(v)?;
        fb.module().const_value(c)
    };
    let (zero, one, two, four) = (u(fb, 0)?, u(fb, 1)?, u(fb, 2)?, u(fb, 4)?);
    let yz = fb.select(fr.major_y, two, four)?;
    let axis = fb.select(fr.major_x, zero, yz)?;
    let sign = fb.select(fr.negative, one, zero)?;
    let face = fb.binary(BinOp::IAdd, axis, sign)?;
    Ok(FacePosition { face, s, t })
}

/// The frame of a cube direction's face (module docs): its components, which of them is the
/// major one and its sign and magnitude, the components that are the face's `s` and `t`
/// coordinates and whether Vulkan's face table negates each, and whether every component is
/// finite.
struct Frame {
    cx: SValue,
    cy: SValue,
    cz: SValue,
    major_x: SValue,
    major_y: SValue,
    major_z: SValue,
    negative: SValue,
    ma: SValue,
    c_s: SValue,
    c_t: SValue,
    flip_s: SValue,
    flip_t: SValue,
    regular: SValue,
}

/// The [`Frame`] of direction `p` (three floats, `metal_ties`'s, which leaves no tie among the
/// largest magnitudes).
fn frame(fb: &mut FunctionBuilder<'_>, p: SValue) -> BResult<Frame> {
    let most = f32_value(fb, f32::MAX)?;
    let zero = f32_value(fb, 0.0)?;
    let a = fb.glsl(GlslOp::FAbs, &[p])?;
    let cx = fb.composite_extract(p, &[0])?;
    let cy = fb.composite_extract(p, &[1])?;
    let cz = fb.composite_extract(p, &[2])?;
    let ax = fb.composite_extract(a, &[0])?;
    let ay = fb.composite_extract(a, &[1])?;
    let az = fb.composite_extract(a, &[2])?;
    let finite_x = fb.binary(BinOp::FOrdLessThanEqual, ax, most)?;
    let finite_y = fb.binary(BinOp::FOrdLessThanEqual, ay, most)?;
    let finite_z = fb.binary(BinOp::FOrdLessThanEqual, az, most)?;
    let finite_xy = fb.binary(BinOp::LogicalAnd, finite_x, finite_y)?;
    let regular = fb.binary(BinOp::LogicalAnd, finite_xy, finite_z)?;
    let x_y = fb.binary(BinOp::FOrdGreaterThanEqual, ax, ay)?;
    let x_z = fb.binary(BinOp::FOrdGreaterThanEqual, ax, az)?;
    let major_x = fb.binary(BinOp::LogicalAnd, x_y, x_z)?;
    let y_z = fb.binary(BinOp::FOrdGreaterThanEqual, ay, az)?;
    let not_x = fb.unary(UnOp::LogicalNot, major_x)?;
    let major_y = fb.binary(BinOp::LogicalAnd, not_x, y_z)?;
    let z_y = fb.binary(BinOp::FOrdGreaterThan, az, ay)?;
    let major_z = fb.binary(BinOp::LogicalAnd, not_x, z_y)?;
    let yz = fb.select(major_y, cy, cz)?;
    let m = fb.select(major_x, cx, yz)?;
    let ma = fb.glsl(GlslOp::FAbs, &[m])?;
    let negative = fb.binary(BinOp::FOrdLessThan, m, zero)?;
    let positive = fb.unary(UnOp::LogicalNot, negative)?;
    // Vulkan's face table: `sc` is `-z` (`+x`), `z` (`-x`), `x` (`±y`, `+z`), or `-x` (`-z`);
    // `tc` is `z` (`+y`), `-z` (`-y`), or `-y` (`±x`, `±z`). A flip is the component negated.
    let c_s = fb.select(major_x, cz, cx)?;
    let c_t = fb.select(major_y, cz, cy)?;
    let flip_x = fb.binary(BinOp::LogicalAnd, major_x, positive)?;
    let flip_z = fb.binary(BinOp::LogicalAnd, major_z, negative)?;
    let flip_s = fb.binary(BinOp::LogicalOr, flip_x, flip_z)?;
    let not_y = fb.unary(UnOp::LogicalNot, major_y)?;
    let flip_t = fb.binary(BinOp::LogicalOr, not_y, negative)?;
    Ok(Frame {
        cx,
        cy,
        cz,
        major_x,
        major_y,
        major_z,
        negative,
        ma,
        c_s,
        c_t,
        flip_s,
        flip_t,
        regular,
    })
}

/// `v`, negated where `flip`.
fn oriented(fb: &mut FunctionBuilder<'_>, flip: SValue, v: SValue) -> BResult<SValue> {
    let negated = fb.unary(UnOp::FNegate, v)?;
    fb.select(flip, negated, v)
}

/// Metal's position on a face axis of size `nf` (an `f32`) of the oriented minor component `sc`,
/// with `r` the reciprocal of the major magnitude: `X = RN(RN(1 + sc * r) / 2 * n)`, the sum
/// fused and each step rounded on its own (module docs). Where `wrap` holds (a repeating axis of a
/// size that is not a power of two), a face coordinate `F / 2` of 1 or more is first less 1,
/// exactly (it lies in `[1, 2)`).
fn texel_x(
    fb: &mut FunctionBuilder<'_>,
    sc: SValue,
    (r, nf): (SValue, SValue),
    (routines, wrap): (SnapRoutines, Option<SValue>),
) -> BResult<SValue> {
    let one = f32_value(fb, 1.0)?;
    let half = f32_value(fb, 0.5)?;
    let sum = fma::emit(fb, routines.fma, &[sc, r, one], routines.fma_routine)?;
    let s = fb.binary_no_contraction(BinOp::FMul, sum, half)?;
    let s = match wrap {
        Some(w) => {
            let past = fb.binary(BinOp::FOrdGreaterThanEqual, s, one)?;
            let past = fb.binary(BinOp::LogicalAnd, w, past)?;
            let less = fb.binary_no_contraction(BinOp::FSub, s, one)?;
            fb.select(past, less, s)?
        }
        None => s,
    };
    fb.binary_no_contraction(BinOp::FMul, s, nf)
}

/// Define the fold of a linear cube filter's tap: `fn(face: u32, ti: i32, tj: i32, n: i32, pick:
/// u32, base: u32) -> uvec3`, the 2D-array coordinate `(i, j, base + g)` of the texel that tap
/// `(ti, tj)` of face `face` (Vulkan's face order) reads on a cube of faces `n` texels wide whose
/// first layer is `base`. In the doubled integer frame where a face spans `[-n, n]`, the tap's
/// centre is the direction `D` of Vulkan's face table at `sc = 2 ti + 1 - n` and `tc = 2 tj + 1 -
/// n` with the major component `±n`. A tap inside the face reads its own texel. A tap past one edge
/// has one component past `±n` and reads the adjacent face's texel along the edge: `D` clamped into
/// `±(n - 1)` with that component set to its sign times `n`, whose major axis is the new face. A
/// tap past both edges, at the cube's corner, has two such components, and reads the texel at the
/// corner of the face of axis `pick` (0, 1, 2: `x`, `y`, `z`) the corner touches: `D` clamped with
/// that axis's component set to its sign times `n`. The texel `(i, j)` of a face-frame direction is
/// `((sc + n - 1) / 2, (tc + n - 1) / 2)` of its `sc` and `tc` (the inverse of the face table).
/// Every result lies on the cube.
pub(super) fn define_fold(mb: &mut ModuleBuilder) -> BResult<FuncId> {
    let u = mb.ty(SType::Int {
        width: 32,
        signed: false,
    })?;
    let i = mb.ty(SType::Int {
        width: 32,
        signed: true,
    })?;
    let uv3 = mb.ty(SType::Vector { elem: u, len: 3 })?;
    let mut fb = mb.routine(uv3)?;
    let face = fb.param(u)?;
    let ti = fb.param(i)?;
    let tj = fb.param(i)?;
    let n = fb.param(i)?;
    let pick = fb.param(u)?;
    let base = fb.param(u)?;
    let fb_ = &mut fb;
    let uc = |fb: &mut FunctionBuilder<'_>, v: u32| -> BResult<SValue> {
        let c = fb.u32_const(v)?;
        fb.module().const_value(c)
    };
    let ic = |fb: &mut FunctionBuilder<'_>, v: i32| -> BResult<SValue> {
        let c = fb.module().constant(Constant::Scalar {
            ty: i,
            bits: ConstBits(u64::from(v as u32)),
        })?;
        fb.module().const_value(c)
    };
    let (u0, u1, u2) = (uc(fb_, 0)?, uc(fb_, 1)?, uc(fb_, 2)?);
    let (i0, i1, minus) = (ic(fb_, 0)?, ic(fb_, 1)?, ic(fb_, -1)?);
    // The face's major axis and sign.
    let m = fb_.binary(BinOp::ShiftRightLogical, face, u1)?;
    let low = fb_.binary(BinOp::BitwiseAnd, face, u1)?;
    let negative = fb_.binary(BinOp::INotEqual, low, u0)?;
    let sign = fb_.select(negative, minus, i1)?;
    let is_x = fb_.binary(BinOp::IEqual, m, u0)?;
    let is_y = fb_.binary(BinOp::IEqual, m, u1)?;
    let is_z = fb_.binary(BinOp::IEqual, m, u2)?;
    // The tap's direction `D`: Vulkan's face table at `sc` and `tc`, the major component `±n`.
    let doubled = |fb: &mut FunctionBuilder<'_>, t: SValue| -> BResult<SValue> {
        let twice = fb.binary(BinOp::IAdd, t, t)?;
        let odd = fb.binary(BinOp::IAdd, twice, i1)?;
        fb.binary(BinOp::ISub, odd, n)
    };
    let sc = doubled(fb_, ti)?;
    let tc = doubled(fb_, tj)?;
    let major = fb_.binary(BinOp::IMul, sign, n)?;
    let signed_sc = fb_.binary(BinOp::IMul, sign, sc)?;
    let signed_tc = fb_.binary(BinOp::IMul, sign, tc)?;
    let minus_tc = fb_.unary(UnOp::SNegate, tc)?;
    let minus_signed_sc = fb_.unary(UnOp::SNegate, signed_sc)?;
    // `x`: `±n`, `sc` (`±y`), `sign · sc` (`±z`); `y`: `±n`, else `-tc`; `z`: `±n`,
    // `-sign · sc` (`±x`), `sign · tc` (`±y`).
    let x = fb_.select(is_y, sc, signed_sc)?;
    let x = fb_.select(is_x, major, x)?;
    let y = fb_.select(is_y, major, minus_tc)?;
    let z = fb_.select(is_x, minus_signed_sc, signed_tc)?;
    let z = fb_.select(is_z, major, z)?;
    // Per axis whether `D` lies past the face's edge there, `D` clamped into `±(n - 1)`, and the
    // component's sign times `n`.
    let last = fb_.binary(BinOp::ISub, n, i1)?;
    let minus_last = fb_.unary(UnOp::SNegate, last)?;
    let minus_n = fb_.unary(UnOp::SNegate, n)?;
    let axis = |fb_: &mut FunctionBuilder<'_>, d: SValue| -> BResult<[SValue; 3]> {
        let a = fb_.glsl(GlslOp::SAbs, &[d])?;
        let over = fb_.binary(BinOp::SGreaterThan, a, n)?;
        let inner = fb_.glsl(GlslOp::SClamp, &[d, minus_last, last])?;
        let below = fb_.binary(BinOp::SLessThan, d, i0)?;
        Ok([over, inner, fb_.select(below, minus_n, n)?])
    };
    let [over_x, inner_x, outer_x] = axis(fb_, x)?;
    let [over_y, inner_y, outer_y] = axis(fb_, y)?;
    let [over_z, inner_z, outer_z] = axis(fb_, z)?;
    // The axis set to `±n`: the one past an edge, the major one where none is, `pick` where two
    // are (the cube's corner).
    let a = fb_.select(over_z, u2, m)?;
    let a = fb_.select(over_y, u1, a)?;
    let a = fb_.select(over_x, u0, a)?;
    let xy = fb_.binary(BinOp::LogicalAnd, over_x, over_y)?;
    let xz = fb_.binary(BinOp::LogicalAnd, over_x, over_z)?;
    let yz = fb_.binary(BinOp::LogicalAnd, over_y, over_z)?;
    let corner = fb_.binary(BinOp::LogicalOr, xy, xz)?;
    let corner = fb_.binary(BinOp::LogicalOr, corner, yz)?;
    let a = fb_.select(corner, pick, a)?;
    let on_x = fb_.binary(BinOp::IEqual, a, u0)?;
    let on_y = fb_.binary(BinOp::IEqual, a, u1)?;
    let on_z = fb_.binary(BinOp::IEqual, a, u2)?;
    let cx = fb_.select(on_x, outer_x, inner_x)?;
    let cy = fb_.select(on_y, outer_y, inner_y)?;
    let cz = fb_.select(on_z, outer_z, inner_z)?;
    // The texel's face, `2 a` plus 1 where that component is negative, and its `sc` and `tc`:
    // `-sign · z` (`±x`), `x` (`±y`), `sign · x` (`±z`); `sign · z` (`±y`), else `-y`.
    let yz_ = fb_.select(on_y, y, z)?;
    let da = fb_.select(on_x, x, yz_)?;
    let below = fb_.binary(BinOp::SLessThan, da, i0)?;
    let twice = fb_.binary(BinOp::ShiftLeftLogical, a, u1)?;
    let low = fb_.select(below, u1, u0)?;
    let g = fb_.binary(BinOp::IAdd, twice, low)?;
    let sign = fb_.select(below, minus, i1)?;
    let signed_x = fb_.binary(BinOp::IMul, sign, cx)?;
    let signed_z = fb_.binary(BinOp::IMul, sign, cz)?;
    let minus_signed_z = fb_.unary(UnOp::SNegate, signed_z)?;
    let minus_y = fb_.unary(UnOp::SNegate, cy)?;
    let s = fb_.select(on_y, cx, signed_x)?;
    let s = fb_.select(on_x, minus_signed_z, s)?;
    let t = fb_.select(on_y, signed_z, minus_y)?;
    let index = |fb: &mut FunctionBuilder<'_>, c: SValue| -> BResult<SValue> {
        let shifted = fb.binary(BinOp::IAdd, c, last)?;
        let half = fb.binary(BinOp::ShiftRightArithmetic, shifted, i1)?;
        fb.convert(ConvOp::Bitcast, half, u)
    };
    let ii = index(fb_, s)?;
    let jj = index(fb_, t)?;
    let layer = fb_.binary(BinOp::IAdd, base, g)?;
    let at = fb_.composite_construct(uv3, &[ii, jj, layer])?;
    fb.ret(Some(at))?;
    fb.finish()
}

/// The `f32` constant `x`.
fn f32_value(fb: &mut FunctionBuilder<'_>, x: f32) -> BResult<SValue> {
    let f32_ = fb.module().ty(SType::Float { width: 32 })?;
    let c = fb.module().constant(Constant::Scalar {
        ty: f32_,
        bits: ConstBits(u64::from(x.to_bits())),
    })?;
    fb.module().const_value(c)
}

#[cfg(test)]
mod tests;
