//! Metal's precise division: the correctly rounded quotient on every profile.
//!
//! Metal's `float` division without `arcp` (a flagless `fdiv`, `precise::divide`, and the plain
//! operator under `-fno-fast-math`) answers the quotient rounded once to nearest even, with an
//! `f32` denormal operand read as a zero of its sign, a quotient below the smallest normal once
//! rounded to 24 bits (with an unbounded exponent) flushed to a zero of its sign, and
//! `0x7FC00000` for a NaN (a probe cases 914cc4d2 and 711cc190, every lane; A1 case 87d13612). The
//! flush follows the 24-bit rounding, not a rounding on the denormal grid: the quotient `(2^24 - 1)
//! · 2^-150`, which that grid rounds up to `2^-126`, is a zero (a probe case c10cba13, 64 lanes of
//! 512, every lane matched). Vulkan bounds `OpFDiv` at 2.5 ulp for a divisor in `[2^-126, 2^126]`
//! and not at all beyond, and devices use that latitude: RADV answers `3 / 3` with `0x3f7fffff` and
//! a divisor past `2^126` with a zero.
//!
//! Each profile calls one module-scope routine per lane count ([`Routine`], as the wide divisions
//! are), by one of two [`Path`]s:
//!
//! - [`Path::Integer`] (`quotient`), on every profile: the operands' significands `X` and `Y`
//!   (24-bit integers) and the device's `OpFDiv` of the operands moved to `[1, 2)`, within 2.5
//!   ulp there, give a 24-bit integer quotient `Q` within three of `F = floor(X · 2^s / Y)`
//!   (`s` = 23 or 24, so that `F` has 24 bits). The remainder `X · 2^s - Q · Y` is below `2^31`
//!   in magnitude, so its low 32 bits, which 32-bit integer arithmetic computes, are exactly it
//!   as a signed integer; three steps each way move `Q` to `F` and the remainder into `[0, Y)`.
//!   Twice the remainder against `Y` then rounds `F` to nearest even. The exponent is the
//!   operands' difference, with the rounding's carry; a result past the largest `float` is the
//!   infinity of its sign, and one below the smallest normal is a zero of its sign. Zeros,
//!   infinities, NaNs, and denormal operands are selected as Metal answers them.
//! - [`Path::Fused`] (`fused`), where the profile's `float` fused multiply-add is `OpFmaKHR`
//!   ([`fma::Route::Khr`]): Markstein's sequence on the operands' significands moved to `[1, 2)`,
//!   where no intermediate overflows or reaches a denormal, so the device's environment does not
//!   enter. The device's reciprocal of the divisor's (within 2.5 ulp) takes two Newton steps, `r +=
//!   r · (1 - y · r)`, each fused, which make it `RN(1/y)` except at an all-ones significand, where
//!   it is the even neighbor below and one ulp up corrects it (every significand, every estimate
//!   within 2.5 ulp: the `markstein_reciprocal_is_correctly_rounded` test). `q = x · r` is within
//!   two ulp of `x / y`; one fused step `q += r · (x - y · q)` brings it within one ulp, where `x -
//!   y · q` is exact, and Markstein's theorem (P. Markstein, IBM J. Res. Develop. 34(1), 1990)
//!   makes a second step the correctly rounded quotient. The quotient of the significands is in
//!   `(1/2, 2)`, so its rounding is the exact quotient's 24-bit rounding; its exponent moved by the
//!   operands' difference gives the result, with the overflow, the flush, and the special operands
//!   as `quotient` answers them. Every lane takes this one path: a divisor far from one costs what
//!   a divisor near it does (an earlier form ran the sequence on the operands themselves, in range
//!   only for `|y|` in `[2^-31, 2^33)`, and sent any other lane through `quotient` as well, 17
//!   times `OpFDiv`'s GPU time on RADV for a divisor near `2^40`).
//!
//! Metal divides `half` the same way under every flag set, `arcp` included: the quotient rounded
//! once from the exact one, `half` denormals kept (a probe cases 4c363561 under fast math and
//! 54f0fe8e without, every lane). The device's `float` `OpFDiv` rounded to `half` is not that
//! quotient by Vulkan's contract: within 2.5 ulp, it may cross a `half` rounding midpoint, from
//! which a quotient of two `half` values can be as little as two `float` ulp away (1 / (2047 /
//! 1024) is 2.001 ulp from one). So every `half` division calls a routine, and its `float`
//! quotient of the operands (held in `float`) is rounded once to `half`. On the integer path and
//! the fused path that quotient is correctly rounded, and a quotient of two `half` values is never
//! within half a `float` ulp of a `half` rounding midpoint without being one, so the two roundings
//! are one. Where the fused path's profiles hold every `half` value exactly (not native half
//! rounding, whose device compiler may hand over an unrounded operand), a `half` division takes a
//! third path:
//!
//! - [`Path::Halves`] (`halves`): one fused correction `q + r · (x - y · q)` of `q = x · r`, `r`
//!   the device's `1 / y`, on the operands themselves. That quotient is within `1/2 + 2^-18` ulp
//!   of the exact one and is it wherever the exact one is a `float`, which suffices: the exact
//!   quotient of two `half` values is a point of the 12-bit grid (every `half` value and every
//!   `half` rounding midpoint, the overflow threshold 65520 and the denormal ones included) or at
//!   least two `float` ulp from every such point, so both round to the same `half`. No step leaves
//!   the normal range (the divisor is in `[2^-24, 65504]`, the quotient in `[2^-40, 2^40]`), so
//!   the device's environment does not enter, and scaling an operand by a power of two scales
//!   every step exactly: the `one_fused_correction_rounds_every_half_quotient` test runs every
//!   pair of 11-bit significands with every estimate within 2.5 ulp. Zeros, infinities, and NaNs
//!   are answered as `fused` answers them, by integer tests that fast math keeps. On an Apple M3
//!   Max through MoltenVK, the pipeline of a fragment function with 164 such divisions is created
//!   in about 5.8 s instead of 6.5 s through `fused`.

use crate::air::ir::{BinOp as AirBin, FastMath};
use crate::device::profile::FloatFunctions;
use crate::device::DeviceProfile;
use crate::lower::arith::fma::{self, Route};
use crate::lower::arith::legal::{FloatFmt, Form, Held};
use crate::lower::arith::narrow::{self, u32_form};
use crate::lower::arith::ops::{fsplat, splat, ty};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::{BResult, BuildError, TypeRule};
use crate::spirv::ids::{FuncId, Value};
use crate::spirv::ir::{BinOp, ConvOp, Type, UnOp};

/// The census name of a precise division ([`super::ArithStrategy::PreciseDivide`]).
pub const PRECISE: &str = "arith.fdiv.precise";

/// Whether the float binary operator `op` with fast-math flags `fmf` on format `fmt` is Metal's
/// precise division, which the shared routine computes: an `fdiv` on `float` without `arcp`, or
/// on `half` under every flag set (the module's comment).
pub fn precise(op: AirBin, fmt: FloatFmt, fmf: FastMath) -> bool {
    op == AirBin::FDiv
        && match fmt {
            FloatFmt::F32 => !fmf.contains(FastMath::ARCP),
            FloatFmt::Half => true,
            FloatFmt::BFloat | FloatFmt::F64 => false,
        }
}

/// How a precise division routine computes its quotient.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Path {
    /// `quotient`: 32-bit integer arithmetic from the device's estimate. Every profile.
    Integer,
    /// `fused`, Markstein's sequence on the significands: profiles whose `float` fused
    /// multiply-add is `OpFmaKHR`.
    Fused,
    /// `halves`, one fused correction on `half` operands held in `float`: a `half` division on
    /// those profiles where every `half` value is held exactly ([`Routine::on`]).
    Halves,
}

impl Path {
    /// The path of a `float` division on profile `p`: [`Path::Fused`] where its `float` fused
    /// multiply-add is `OpFmaKHR` ([`fma::float_route`]), else [`Path::Integer`].
    pub fn on(p: &DeviceProfile) -> Path {
        match fma::float_route(p) {
            Route::Khr => Path::Fused,
            Route::Wide
            | Route::Soft
            | Route::HalfOdd
            | Route::F16
            | Route::F16Khr
            | Route::Native => Path::Integer,
        }
    }
}

/// The module-scope precise division routine on `float` lanes of `lanes` lanes by `path`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Routine {
    pub lanes: u8,
    pub path: Path,
}

impl Routine {
    /// The routine for a division of form `f` (`float` or `half` lanes) on profile `p`: the
    /// profile's [`Path`], or [`Path::Halves`] for `half` lanes on the fused path where the
    /// profile's `half` rounding is not the device's own (`narrow::native_rounding`), so that
    /// every operand is a `half` value.
    pub fn on(p: &DeviceProfile, f: Form) -> Routine {
        let path = match Path::on(p) {
            Path::Fused
                if f.float_fmt() == Some(FloatFmt::Half)
                    && narrow::native_rounding(p, narrow::quantize_form(p)).is_none() =>
            {
                Path::Halves
            }
            path => path,
        };
        Routine {
            lanes: f.lanes,
            path,
        }
    }

    /// An upper estimate of the routine's SPIR-V words, with its constants and types.
    pub fn words(self) -> u32 {
        let lanes = u32::from(self.lanes);
        match self.path {
            Path::Integer => 640 + 48 * lanes,
            Path::Fused => 420 + 40 * lanes,
            Path::Halves => 240 + 24 * lanes,
        }
    }
}

/// An upper estimate of the SPIR-V words per use of a precise division on form `f`: the call,
/// and a `half` quotient's rounding.
pub fn use_words(f: Form) -> u32 {
    24 + narrow::round_words(f)
}

/// Define routine `r`: a function of two `float` values of `r.lanes` lanes returning Metal's
/// precise quotient.
pub fn define(mb: &mut ModuleBuilder, r: Routine) -> BResult<FuncId> {
    define_with(mb, r, &device_division)
}

/// [`define`] with `rough` the device's quotient (a test moves it within Vulkan's bound).
pub(super) fn define_with(mb: &mut ModuleBuilder, r: Routine, rough: Rough<'_>) -> BResult<FuncId> {
    let f = f32_form(r.lanes);
    let e = mb.ty(Type::Float { width: 32 })?;
    let t = if r.lanes == 1 {
        e
    } else {
        mb.ty(Type::Vector {
            elem: e,
            len: r.lanes,
        })?
    };
    let mut fb = mb.routine(t)?;
    let (x, y) = (fb.param(t)?, fb.param(t)?);
    let q = match r.path {
        Path::Integer => quotient(&mut fb, f, x, y, rough)?,
        Path::Fused => fused(&mut fb, f, x, y, rough)?,
        Path::Halves => halves(&mut fb, f, x, y, rough)?,
    };
    fb.ret(Some(q))?;
    fb.finish()
}

/// Metal's precise quotient of `x` and `y` (form `f`, `float` or `half` lanes) through
/// `routine`, the form's lane count's [`define`].
pub fn emit(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    x: Value,
    y: Value,
    routine: FuncId,
) -> BResult<Value> {
    let q = fb
        .call(routine, &[x, y])?
        .ok_or(BuildError::Type(TypeRule::FunctionSignature))?;
    match f.float_fmt() {
        Some(FloatFmt::F32) => Ok(q),
        Some(FloatFmt::Half) => narrow::round(fb, f, q),
        _ => Err(BuildError::Type(TypeRule::FunctionSignature)),
    }
}

/// The census name of a precise division by native device maths ([`native`]).
pub const NATIVE: &str = "arith.fdiv.native";

/// An upper estimate of the SPIR-V words of one [`native`] division on form `f`: the divisor's
/// magnitude, its test, the scale's select, two products, the division, and a `half` result's
/// rounding.
pub fn native_words(f: Form) -> u32 {
    96 + 16 * u32::from(f.lanes)
}

/// Metal's precise division `x / y` on `float` or `half` lanes (form `f`) by native device maths
/// (`FloatFunctions::Native`): the device's `OpFDiv`, within Vulkan's 2.5 ulp for a divisor in
/// `[2^-126, 2^126]`, the operands of a divisor past `2^126` first scaled by `2^-2` into that range
/// (exactly, but where the dividend becomes a denormal, whose quotient by such a divisor Metal
/// flushes to zero anyway); a `half` quotient rounded to `half` (`PrecisionNote::NativeDivision`).
pub fn native(fb: &mut FunctionBuilder<'_>, f: Form, x: Value, y: Value) -> BResult<Value> {
    let ay = fb.glsl(crate::spirv::ir::GlslOp::FAbs, &[y])?;
    let limit = fsplat(fb, f, 2f64.powi(126))?;
    let past = fb.binary(BinOp::FOrdGreaterThan, ay, limit)?;
    let (quarter, one) = (fsplat(fb, f, 0.25)?, fsplat(fb, f, 1.0)?);
    let scale = fb.select(past, quarter, one)?;
    let x = fb.binary(BinOp::FMul, x, scale)?;
    let y = fb.binary(BinOp::FMul, y, scale)?;
    let q = fb.binary(BinOp::FDiv, x, y)?;
    match f.float_fmt() {
        Some(FloatFmt::F32) => Ok(q),
        Some(FloatFmt::Half) => narrow::round(fb, f, q),
        _ => Err(BuildError::Type(TypeRule::FunctionSignature)),
    }
}

/// The device's quotient of two `float` lanes, the estimate `quotient` (of two lanes in
/// `[1, 2)`), `fused` (of one and a divisor in its range), `halves` (of one and a `half` divisor),
/// and [`Reciprocal::Rounded`] correct.
pub(super) type Rough<'a> = &'a dyn Fn(&mut FunctionBuilder<'_>, Value, Value) -> BResult<Value>;

/// How Metal's fast reciprocal (`math::fast_reciprocal`, the reciprocal of an `arcp` division) is
/// rounded on a profile. Metal's own is within about 0.64 ulp of `1 / y` and not correctly rounded:
/// it differs from `RN(1 / y)` on 38 of a probe case 914cc4d2's 512 lanes, each near a rounding
/// midpoint, and no exact form of it is known. Off a device whose division is Metal's, either
/// reciprocal is therefore a near-equivalent (`PrecisionNote::DeviceArithmetic`); `RN(1 / y)` is
/// the nearer one, with the 38 lanes its residue.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Reciprocal {
    /// The device's `OpFDiv(1, y)`. On a profile whose float functions and division are
    /// Metal's (`FloatFunctions::Metal`) that is Metal's own: its fast reciprocal where Metal
    /// compiles the module with fast math, and `RN(1 / y)` where the module turns fast math off.
    /// Elsewhere it is within Vulkan's 2.5 ulp: profiles whose `float` fused multiply-add is not
    /// `OpFmaKHR`.
    Device,
    /// `RN(1 / y)`, the device's estimate refined by `OpFmaKHR` (`rounded_reciprocal_bits`):
    /// the other profiles, whose `float` fused multiply-add is `OpFmaKHR` ([`fma::float_route`]).
    Rounded,
    /// Native device maths (`FloatFunctions::Native`): Metal's fast division is the device's
    /// `OpFDiv(x, y)` itself, and its reciprocal the device's `OpFDiv(1, y)`, within Vulkan's
    /// 2.5 ulp (`PrecisionNote::NativeDivision`).
    Native,
}

impl Reciprocal {
    /// The reciprocal on profile `p`. The typed fact decides it, never the driver: where the
    /// profile's float functions are Metal's, the device's `OpFDiv(1, y)` is Metal's own
    /// reciprocal or `RN(1 / y)` by the module's math mode, so nothing nearer exists. Otherwise it
    /// is the reciprocal beside the profile's `float` fused multiply-add.
    pub fn on(p: &DeviceProfile) -> Reciprocal {
        match p.fields().float_functions {
            FloatFunctions::Metal => Reciprocal::Device,
            FloatFunctions::Device => Reciprocal::of(fma::float_route(p)),
            FloatFunctions::Native => Reciprocal::Native,
        }
    }

    /// The reciprocal beside a `float` fused multiply-add by `route`, on a profile whose float
    /// functions are the device's.
    pub fn of(route: Route) -> Reciprocal {
        match route {
            Route::Khr => Reciprocal::Rounded,
            Route::Wide
            | Route::Soft
            | Route::HalfOdd
            | Route::F16
            | Route::F16Khr
            | Route::Native => Reciprocal::Device,
        }
    }

    /// An upper estimate of the SPIR-V words the reciprocal adds to the device's division on
    /// form `f`: the high range's test and moves, `-y`, four fused multiply-adds, the all-ones
    /// significand's test and ulp, and their constants.
    pub fn words(self, f: Form) -> u32 {
        match self {
            Reciprocal::Device | Reciprocal::Native => 0,
            Reciprocal::Rounded => 128 + 8 * u32::from(f.lanes),
        }
    }
}

/// `RN(1 / y)` of `float` lanes `y` (form `f`), as their bits, from `r`, an estimate within 2.5
/// ulp of it: two Newton steps `r += r · (1 - y · r)`, each by `OpFmaKHR`, give `RN(1 / y)`
/// except at an all-ones significand, where they give the even neighbor below and one ulp up is
/// `RN(1 / y)` (every significand, every estimate within 2.5 ulp:
/// `markstein_reciprocal_is_correctly_rounded`). `one` is `1.0`, `ny` is `-y`, and `yb` holds
/// `y`'s bits. Scaling `y` by a power of two scales every step exactly while `1 / y` and the
/// estimate are normal numbers, so the result is exact for every such `y`.
pub(super) fn rounded_reciprocal_bits(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    one: Value,
    ny: Value,
    yb: Value,
    mut r: Value,
) -> BResult<Value> {
    let u = u32_form(f);
    let ut = ty(fb, u)?;
    for _ in 0..2 {
        let t = fb.fma(ny, r, one)?;
        r = fb.fma(t, r, r)?;
    }
    let mant_mask = splat(fb, u, 0x7F_FFFF)?;
    let mant = fb.binary(BinOp::BitwiseAnd, yb, mant_mask)?;
    let ones = fb.binary(BinOp::IEqual, mant, mant_mask)?;
    let (k0, k1) = (splat(fb, u, 0)?, splat(fb, u, 1)?);
    let bump = fb.select(ones, k1, k0)?;
    let rb = fb.convert(ConvOp::Bitcast, r, ut)?;
    fb.binary(BinOp::BitwiseOr, rb, bump)
}

/// `OpFDiv`: within 2.5 ulp for a divisor in `[2^-126, 2^126]` (the Vulkan precision table).
fn device_division(fb: &mut FunctionBuilder<'_>, x: Value, y: Value) -> BResult<Value> {
    fb.binary(BinOp::FDiv, x, y)
}

fn f32_form(lanes: u8) -> Form {
    Form {
        held: Held::Float(FloatFmt::F32),
        lanes,
    }
}

/// Markstein's correctly rounded quotient of `float` lanes `x` and `y` (form `f`) by
/// `OpFmaKHR`, on every lane: the significands moved to `[1, 2)`, inside the sequence's range
/// (`[2^-31, 2^33)`), divided from `rough`'s estimate of `1 / yn`, which must be within 2.5 ulp
/// of it, and the quotient's exponent moved by the operands' difference ([`assemble`], which also
/// answers zeros, infinities, and NaNs). The significands' quotient is in `(1/2, 2)`, so its
/// rounding is the exact quotient's 24-bit rounding with an unbounded exponent, which decides
/// Metal's flush and overflow.
fn fused(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    x: Value,
    y: Value,
    rough: Rough<'_>,
) -> BResult<Value> {
    let u = u32_form(f);
    let (ut, ft) = (ty(fb, u)?, ty(fb, f)?);
    let k = |fb: &mut FunctionBuilder<'_>, v: u64| splat(fb, u, v);
    let bin = |fb: &mut FunctionBuilder<'_>, op: BinOp, a: Value, b: Value| fb.binary(op, a, b);
    let p = Parts::of(fb, f, x, y)?;

    // The significands as floats in [1, 2).
    let one_bits = k(fb, 0x3F80_0000)?;
    let xn = bin(fb, BinOp::BitwiseOr, p.mx, one_bits)?;
    let xn = fb.convert(ConvOp::Bitcast, xn, ft)?;
    let ynb = bin(fb, BinOp::BitwiseOr, p.my, one_bits)?;
    let yn = fb.convert(ConvOp::Bitcast, ynb, ft)?;

    // RN(1 / yn): two fused Newton steps from the estimate, then the all-ones significand's ulp.
    let one = fsplat(fb, f, 1.0)?;
    let ny = fb.unary(UnOp::FNegate, yn)?;
    let r = rough(fb, one, yn)?;
    let rb = rounded_reciprocal_bits(fb, f, one, ny, ynb, r)?;
    let r = fb.convert(ConvOp::Bitcast, rb, ft)?;

    // q = RN(xn · r), then two fused steps; the second is Markstein's.
    let minus_zero = fsplat(fb, f, -0.0)?;
    let mut q = fb.fma(xn, r, minus_zero)?;
    for _ in 0..2 {
        let e = fb.fma(ny, q, xn)?;
        q = fb.fma(e, r, q)?;
    }

    // The quotient's biased exponent ex - ey + eq, and its significand field.
    let qb = fb.convert(ConvOp::Bitcast, q, ut)?;
    let (k23, exp_mask, mant_mask) = (k(fb, 23)?, k(fb, 0xFF)?, k(fb, 0x7F_FFFF)?);
    let eq = bin(fb, BinOp::ShiftRightLogical, qb, k23)?;
    let eq = bin(fb, BinOp::BitwiseAnd, eq, exp_mask)?;
    let e = bin(fb, BinOp::IAdd, p.ex, eq)?;
    let e = bin(fb, BinOp::ISub, e, p.ey)?;
    let mant = bin(fb, BinOp::BitwiseAnd, qb, mant_mask)?;
    assemble(fb, f, &p, e, mant)
}

/// The `float` quotient of `half` lanes `x` and `y` held in `float` (form `f`) by `OpFmaKHR`,
/// whose rounding to `half` is Metal's `half` quotient ([`Path::Halves`]): `q = x · r` from
/// `rough`'s estimate `r` of `1 / y`, within 2.5 ulp of it, and one fused step
/// `q + r · (x - y · q)`, on every lane whose operands are both finite and nonzero. With `ε` and
/// `δ` the estimate's and the product's relative errors (`|ε| <= 2.5 · 2^-23`, `|δ| <= 2^-24`),
/// `q` is within `|x / y| · 3 · 2^-23` of `x / y`; the fused residual `x - y · q` is rounded
/// once at most (by `η`, `|η| <= 2^-24`), so the step's exact sum is within that times
/// `|ε + η + εη|`, below `2^-18` ulp, and its one rounding gives the quotient within
/// `1/2 + 2^-18` ulp, and exactly where the quotient is a `float`. Zeros, infinities, and NaNs,
/// which a `half` operand reaches only as itself (no `half` value is an `f32` denormal), are
/// answered as `fused` answers them: a NaN for a NaN operand, `0 / 0`, or `∞ / ∞`; the infinity
/// of the quotient's sign for an infinite dividend or a zero divisor; else its zero.
fn halves(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    x: Value,
    y: Value,
    rough: Rough<'_>,
) -> BResult<Value> {
    let u = u32_form(f);
    let (ut, ft) = (ty(fb, u)?, ty(fb, f)?);
    let k = |fb: &mut FunctionBuilder<'_>, v: u64| splat(fb, u, v);
    let bin = |fb: &mut FunctionBuilder<'_>, op: BinOp, a: Value, b: Value| fb.binary(op, a, b);

    // q = x · r, then one fused step q + r · (x - y · q).
    let one = fsplat(fb, f, 1.0)?;
    let r = rough(fb, one, y)?;
    let q = bin(fb, BinOp::FMul, x, r)?;
    let ny = fb.unary(UnOp::FNegate, y)?;
    let e = fb.fma(ny, q, x)?;
    let q = fb.fma(e, r, q)?;
    let qb = fb.convert(ConvOp::Bitcast, q, ut)?;

    // A lane is regular where both magnitudes are in [1, 0x7F7FFFFF]: finite and nonzero.
    let xb = fb.convert(ConvOp::Bitcast, x, ut)?;
    let yb = fb.convert(ConvOp::Bitcast, y, ut)?;
    let (magnitude, sign_bit) = (k(fb, 0x7FFF_FFFF)?, k(fb, 0x8000_0000)?);
    let ax = bin(fb, BinOp::BitwiseAnd, xb, magnitude)?;
    let ay = bin(fb, BinOp::BitwiseAnd, yb, magnitude)?;
    let xy = bin(fb, BinOp::BitwiseXor, xb, yb)?;
    let sign = bin(fb, BinOp::BitwiseAnd, xy, sign_bit)?;
    let (k0, k1, largest) = (k(fb, 0)?, k(fb, 1)?, k(fb, 0x7F7F_FFFF)?);
    let xm = bin(fb, BinOp::ISub, ax, k1)?;
    let ym = bin(fb, BinOp::ISub, ay, k1)?;
    let xr = bin(fb, BinOp::ULessThan, xm, largest)?;
    let yr = bin(fb, BinOp::ULessThan, ym, largest)?;
    let regular = bin(fb, BinOp::LogicalAnd, xr, yr)?;

    // Elsewhere one operand is a zero, an infinity, or a NaN: equal magnitudes there are two
    // zeros, two infinities, or NaNs.
    let inf_bits = k(fb, 0x7F80_0000)?;
    let xnan = bin(fb, BinOp::UGreaterThan, ax, inf_bits)?;
    let ynan = bin(fb, BinOp::UGreaterThan, ay, inf_bits)?;
    let same = bin(fb, BinOp::IEqual, ax, ay)?;
    let nan = bin(fb, BinOp::LogicalOr, xnan, ynan)?;
    let nan = bin(fb, BinOp::LogicalOr, nan, same)?;
    let xi = bin(fb, BinOp::IEqual, ax, inf_bits)?;
    let yz = bin(fb, BinOp::IEqual, ay, k0)?;
    let to_inf = bin(fb, BinOp::LogicalOr, xi, yz)?;
    let inf = bin(fb, BinOp::BitwiseOr, sign, inf_bits)?;
    let special = fb.select(to_inf, inf, sign)?;
    let qnan = k(fb, 0x7FC0_0000)?;
    let special = fb.select(nan, qnan, special)?;
    let res = fb.select(regular, qb, special)?;
    fb.convert(ConvOp::Bitcast, res, ft)
}

/// The fields of two `float` operands both paths read: their bits, the quotient's sign bit, the
/// biased exponent fields, and the significand fields.
struct Parts {
    xb: Value,
    yb: Value,
    sign: Value,
    ex: Value,
    ey: Value,
    mx: Value,
    my: Value,
}

impl Parts {
    fn of(fb: &mut FunctionBuilder<'_>, f: Form, x: Value, y: Value) -> BResult<Parts> {
        let u = u32_form(f);
        let ut = ty(fb, u)?;
        let (sign_bit, exp_mask, mant_mask, k23) = (
            splat(fb, u, 0x8000_0000)?,
            splat(fb, u, 0xFF)?,
            splat(fb, u, 0x7F_FFFF)?,
            splat(fb, u, 23)?,
        );
        let xb = fb.convert(ConvOp::Bitcast, x, ut)?;
        let yb = fb.convert(ConvOp::Bitcast, y, ut)?;
        let xy = fb.binary(BinOp::BitwiseXor, xb, yb)?;
        let sign = fb.binary(BinOp::BitwiseAnd, xy, sign_bit)?;
        let ex = fb.binary(BinOp::ShiftRightLogical, xb, k23)?;
        let ex = fb.binary(BinOp::BitwiseAnd, ex, exp_mask)?;
        let ey = fb.binary(BinOp::ShiftRightLogical, yb, k23)?;
        let ey = fb.binary(BinOp::BitwiseAnd, ey, exp_mask)?;
        let mx = fb.binary(BinOp::BitwiseAnd, xb, mant_mask)?;
        let my = fb.binary(BinOp::BitwiseAnd, yb, mant_mask)?;
        Ok(Parts {
            xb,
            yb,
            sign,
            ex,
            ey,
            mx,
            my,
        })
    }
}

/// The quotient of the operands `p` whose 24-bit rounding has the biased exponent `e` (a signed
/// integer, unbounded) and the significand field `mant`: the infinity of its sign from 255 up,
/// the zero of its sign from 0 down (below the smallest normal once rounded to 24 bits,
/// c10cba13), and zeros (an `f32` denormal is one), infinities, and NaNs as Metal answers them.
fn assemble(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    p: &Parts,
    e: Value,
    mant: Value,
) -> BResult<Value> {
    let u = u32_form(f);
    let ft = ty(fb, f)?;
    let k = |fb: &mut FunctionBuilder<'_>, v: u64| splat(fb, u, v);
    let bin = |fb: &mut FunctionBuilder<'_>, op: BinOp, a: Value, b: Value| fb.binary(op, a, b);
    let (k0, k23) = (k(fb, 0)?, k(fb, 23)?);
    let field = bin(fb, BinOp::ShiftLeftLogical, e, k23)?;
    let normal = bin(fb, BinOp::BitwiseOr, p.sign, field)?;
    let normal = bin(fb, BinOp::BitwiseOr, normal, mant)?;
    let inf_bits = k(fb, 0x7F80_0000)?;
    let inf = bin(fb, BinOp::BitwiseOr, p.sign, inf_bits)?;
    let k255 = k(fb, 255)?;
    let over = bin(fb, BinOp::SGreaterThanEqual, e, k255)?;
    let under = bin(fb, BinOp::SLessThanEqual, e, k0)?;
    let res = fb.select(over, inf, normal)?;
    let res = fb.select(under, p.sign, res)?;

    // Zeros (an f32 denormal is one), infinities, and NaNs, as Metal answers them.
    let magnitude = k(fb, 0x7FFF_FFFF)?;
    let ax = bin(fb, BinOp::BitwiseAnd, p.xb, magnitude)?;
    let ay = bin(fb, BinOp::BitwiseAnd, p.yb, magnitude)?;
    let xz = bin(fb, BinOp::IEqual, p.ex, k0)?;
    let yz = bin(fb, BinOp::IEqual, p.ey, k0)?;
    let xi = bin(fb, BinOp::IEqual, ax, inf_bits)?;
    let yi = bin(fb, BinOp::IEqual, ay, inf_bits)?;
    let xnan = bin(fb, BinOp::UGreaterThan, ax, inf_bits)?;
    let ynan = bin(fb, BinOp::UGreaterThan, ay, inf_bits)?;
    let zz = bin(fb, BinOp::LogicalAnd, xz, yz)?;
    let ii = bin(fb, BinOp::LogicalAnd, xi, yi)?;
    let nan = bin(fb, BinOp::LogicalOr, xnan, ynan)?;
    let nan = bin(fb, BinOp::LogicalOr, nan, zz)?;
    let nan = bin(fb, BinOp::LogicalOr, nan, ii)?;
    let to_inf = bin(fb, BinOp::LogicalOr, xi, yz)?;
    let to_zero = bin(fb, BinOp::LogicalOr, xz, yi)?;
    let res = fb.select(to_zero, p.sign, res)?;
    let res = fb.select(to_inf, inf, res)?;
    let qnan = k(fb, 0x7FC0_0000)?;
    let res = fb.select(nan, qnan, res)?;
    fb.convert(ConvOp::Bitcast, res, ft)
}

/// Metal's precise quotient of `float` lanes `x` and `y` (form `f`), from `rough`'s estimate of
/// the quotient of their significands, which must be within 2.5 ulp of it.
fn quotient(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    x: Value,
    y: Value,
    rough: Rough<'_>,
) -> BResult<Value> {
    let u = u32_form(f);
    let (ut, ft) = (ty(fb, u)?, ty(fb, f)?);
    let k = |fb: &mut FunctionBuilder<'_>, v: u64| splat(fb, u, v);
    let bin = |fb: &mut FunctionBuilder<'_>, op: BinOp, a: Value, b: Value| fb.binary(op, a, b);
    let p = Parts::of(fb, f, x, y)?;
    let (mx, my) = (p.mx, p.my);
    let (mant_mask, hidden) = (k(fb, 0x7F_FFFF)?, k(fb, 0x80_0000)?);
    let (k0, k1, k23) = (k(fb, 0)?, k(fb, 1)?, k(fb, 23)?);

    // The significands, as integers in [2^23, 2^24) and as floats in [1, 2).
    let big_x = bin(fb, BinOp::BitwiseOr, mx, hidden)?;
    let big_y = bin(fb, BinOp::BitwiseOr, my, hidden)?;
    let one = k(fb, 0x3F80_0000)?;
    let xn = bin(fb, BinOp::BitwiseOr, mx, one)?;
    let xn = fb.convert(ConvOp::Bitcast, xn, ft)?;
    let yn = bin(fb, BinOp::BitwiseOr, my, one)?;
    let yn = fb.convert(ConvOp::Bitcast, yn, ft)?;

    // F = floor(X · 2^s / Y) has 24 bits: s = 23 where X >= Y, else 24.
    let ge = bin(fb, BinOp::UGreaterThanEqual, big_x, big_y)?;
    let s = {
        let k24 = k(fb, 24)?;
        fb.select(ge, k23, k24)?
    };
    let q0 = rough(fb, xn, yn)?;
    let scale = {
        let (a, b) = (fsplat(fb, f, 8_388_608.0)?, fsplat(fb, f, 16_777_216.0)?);
        fb.select(ge, a, b)?
    };
    // `q0 · 2^s` is exact; truncation adds less than one to the estimate's 2.5.
    let q0s = bin(fb, BinOp::FMul, q0, scale)?;
    let mut q = fb.convert(ConvOp::ConvertFToU, q0s, ut)?;
    // The remainder, below 2^31 in magnitude, exactly in the low 32 bits.
    let xs = bin(fb, BinOp::ShiftLeftLogical, big_x, s)?;
    let qy = bin(fb, BinOp::IMul, q, big_y)?;
    let mut r = bin(fb, BinOp::ISub, xs, qy)?;
    for _ in 0..3 {
        let neg = bin(fb, BinOp::SLessThan, r, k0)?;
        let r_up = bin(fb, BinOp::IAdd, r, big_y)?;
        let q_down = bin(fb, BinOp::ISub, q, k1)?;
        r = fb.select(neg, r_up, r)?;
        q = fb.select(neg, q_down, q)?;
    }
    for _ in 0..3 {
        let over = bin(fb, BinOp::SGreaterThanEqual, r, big_y)?;
        let r_down = bin(fb, BinOp::ISub, r, big_y)?;
        let q_up = bin(fb, BinOp::IAdd, q, k1)?;
        r = fb.select(over, r_down, r)?;
        q = fb.select(over, q_up, q)?;
    }

    // Round to nearest even: up past the midpoint, and at it when F is odd.
    let twice = bin(fb, BinOp::ShiftLeftLogical, r, k1)?;
    let past = bin(fb, BinOp::UGreaterThan, twice, big_y)?;
    let at = bin(fb, BinOp::IEqual, twice, big_y)?;
    let low = bin(fb, BinOp::BitwiseAnd, q, k1)?;
    let odd = bin(fb, BinOp::IEqual, low, k1)?;
    let tie_up = bin(fb, BinOp::LogicalAnd, at, odd)?;
    let up = bin(fb, BinOp::LogicalOr, past, tie_up)?;
    let inc = fb.select(up, k1, k0)?;
    let sig = bin(fb, BinOp::IAdd, q, inc)?;
    let carry = {
        let k24 = k(fb, 24)?;
        bin(fb, BinOp::ShiftRightLogical, sig, k24)?
    };
    let mant = bin(fb, BinOp::BitwiseAnd, sig, mant_mask)?;

    // The biased exponent ex - ey + 126 + [X >= Y] + carry, signed.
    let k126 = k(fb, 126)?;
    let e = bin(fb, BinOp::IAdd, p.ex, k126)?;
    let e = bin(fb, BinOp::ISub, e, p.ey)?;
    let gei = fb.select(ge, k1, k0)?;
    let e = bin(fb, BinOp::IAdd, e, gei)?;
    let e = bin(fb, BinOp::IAdd, e, carry)?;
    assemble(fb, f, &p, e, mant)
}
