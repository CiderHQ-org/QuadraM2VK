//! Metal's precise `sin` and `cos` of a `float`: Metal's own form, `t S(t)` of the folded
//! quarter-turn value `t` and its hardware's `sinc`, which follows Apple's runtime library; the
//! `sinc` through a model of the hardware's, read off directly.
//!
//! **Reduction** to quarter turns, `x 2/π = k + r`, by Metal's own reduction ([`quarter::turns`],
//! without the error term of its large operands). The quarter turns fold `r` into `t` in `[-1, 1]`
//! with `sin(x) = sin(π t / 2)`: `t = r` for an even `k` and `1 - |r|` for an odd one (`sin`),
//! `1 - |r|` and `-r` (`cos`), negated where `k`'s bit 1 is set.
//!
//! **The `sinc`.** Metal multiplies `t` by its hardware's `S(t)`, about `sin(π t / 2) / t`
//! within an ulp, and rounds the product. The hardware reads `|t|` as 23 fractional bits,
//! truncated (`m = floor(|t| 2^23)`; every word of the probe is constant on those cells), and is
//! a cubic in the offset `d` of `m` in each of 32 segments of `2^18` (its errors jump at the
//! segments' ends), with a residue of a few thousandths of an ulp that no cubic follows: the same
//! word on every run and on the M3 and the M4, so the hardware's own behaviour, unmodelled. The
//! model ([`SINC`]) is a cubic per segment by Horner's rule in integers, 12 bits below `S`'s ulp,
//! fitted to the hardware's words at every `m`: Metal's word at 8,377,984 of the 8,388,608 `m`
//! below 1 (99.873 %), and through it Metal's precise `sin` and `cos` at 99.86 % to 99.91 % of
//! the measured operands, where the correctly rounded value is at 45 % to 58 %.
//! `S(0)` is `RN(π/2)` and `S(1)` is 1, as the hardware answers.
//!
//! **Special operands**: below `2^-12` in magnitude or at it (a zero and a denormal included),
//! `sin(x)` is `x` and `cos(x)` is 1; an infinity or a NaN, Metal's NaN.
//!
//! **Metal's `half` `sin` and `cos`** ([`half_sin_cos`]) are its own form, which follows Apple's
//! runtime library: a shorter reduction (`2/π`'s first two parts, no table) with the same fold and
//! `sinc`, and `sin(x) = x` below `1/64`. Rounded to `half`, Metal's word at every `half`, the 6
//! (`sin`) and 2 (`cos`) where that is not the correctly rounded value
//! among them.

use crate::lower::arith::libm::atan::read;
use crate::lower::arith::libm::quarter::{self, Error};
use crate::lower::arith::libm::trig::Trig;
use crate::lower::arith::libm::{Fun, E};
use crate::spirv::builder::ModuleBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, Value};
use crate::spirv::ir::{BinOp, MemAccess, UnOp};

/// The model of the hardware's `sinc`, per segment `j` of `2^18` cells `m = floor(|t| 2^23)`: the
/// bits of `S` at the segment's start (`0x3f800000` plus the constant term's whole ulps), the
/// constant term's 12 bits below the ulp, and the cubic's coefficients of `(d / 2^18)^1..3` in
/// units of `2^-12` ulp (`i32` as `u32`).
pub(crate) const SINC: [[u32; 5]; 32] = [
    [
        0x3FC9_0FDA,
        0x0000_0794,
        0x0000_05DE,
        0xFEB5_3041,
        0x0000_17DB,
    ],
    [
        0x3FC8_FB2F,
        0x0000_0595,
        0xFD6A_B5BE,
        0xFEB5_70B1,
        0x0000_3F85,
    ],
    [
        0x3FC8_BD35,
        0x0000_0B8F,
        0xFAD6_5BFF,
        0xFEB6_280C,
        0x0000_67F5,
    ],
    [
        0x3FC8_5604,
        0x0000_078B,
        0xF843_E80D,
        0xFEB7_5BF3,
        0x0000_9004,
    ],
    [
        0x3FC7_C5C1,
        0x0000_0B97,
        0xF5B4_53AB,
        0xFEB9_08C7,
        0x0000_B782,
    ],
    [
        0x3FC7_0CA2,
        0x0000_0E8F,
        0xF328_9001,
        0xFEBB_27F8,
        0x0000_E00C,
    ],
    [
        0x3FC6_2AEC,
        0x0000_0792,
        0xF0A1_87C7,
        0xFEBD_C4AF,
        0x0001_077D,
    ],
    [
        0x3FC5_20F1,
        0x0000_0C93,
        0xEE20_2FCC,
        0xFEC0_D483,
        0x0001_2FAB,
    ],
    [
        0x3FC3_EF15,
        0x0000_0190,
        0xEBA5_69E6,
        0xFEC4_6840,
        0x0001_4FDA,
    ],
    [
        0x3FC2_95C7,
        0x0000_048F,
        0xE932_2DE1,
        0xFEC8_6042,
        0x0001_77E1,
    ],
    [
        0x3FC1_1587,
        0x0000_0A91,
        0xE6C7_5BC8,
        0xFECC_C892,
        0x0001_9FA1,
    ],
    [
        0x3FBF_6EE3,
        0x0000_0E91,
        0xE465_CDCF,
        0xFED1_AC7B,
        0x0001_BFB2,
    ],
    [
        0x3FBD_A277,
        0x0000_0892,
        0xE20E_6FA9,
        0xFED6_F105,
        0x0001_E742,
    ],
    [
        0x3FBB_B0EC,
        0x0000_0099,
        0xDFC2_0B40,
        0xFEDC_AE08,
        0x0002_069C,
    ],
    [
        0x3FB9_9AF7,
        0x0000_0F93,
        0xDD81_83BB,
        0xFEE2_C4B8,
        0x0002_2F83,
    ],
    [
        0x3FB7_615F,
        0x0000_0794,
        0xDB4D_9D79,
        0xFEE9_4D70,
        0x0002_4F05,
    ],
    [
        0x3FB5_04F3,
        0x0000_008E,
        0xD927_29D4,
        0xFEF0_3496,
        0x0002_6F90,
    ],
    [
        0x3FB2_868F,
        0x0000_0E94,
        0xD70E_E999,
        0xFEF7_7CEE,
        0x0002_8F71,
    ],
    [
        0x3FAF_E71F,
        0x0000_0493,
        0xD505_91C9,
        0xFEFF_3076,
        0x0002_A7BC,
    ],
    [
        0x3FAD_2795,
        0x0000_0F92,
        0xD30B_ED89,
        0xFF07_2D5B,
        0x0002_C70B,
    ],
    [
        0x3FAA_48F4,
        0x0000_008E,
        0xD122_9FBD,
        0xFF0F_88BB,
        0x0002_DF87,
    ],
    [
        0x3FA7_4C44,
        0x0000_088A,
        0xCF4A_561C,
        0xFF18_2BA7,
        0x0003_004C,
    ],
    [
        0x3FA4_329C,
        0x0000_0999,
        0xCD83_AD66,
        0xFF21_2976,
        0x0003_1712,
    ],
    [
        0x3FA0_FD1B,
        0x0000_0690,
        0xCBCF_45BC,
        0xFF2A_6CA0,
        0x0003_2FA1,
    ],
    [
        0x3F9D_ACE9,
        0x0000_0796,
        0xCA2D_AF77,
        0xFF33_F561,
        0x0003_4715,
    ],
    [
        0x3F9A_4338,
        0x0000_048E,
        0xC89F_6FA7,
        0xFF3D_CD00,
        0x0003_574F,
    ],
    [
        0x3F96_C141,
        0x0000_079B,
        0xC725_154A,
        0xFF47_D1AB,
        0x0003_6EF6,
    ],
    [
        0x3F93_2846,
        0x0000_0D95,
        0xC5BF_0970,
        0xFF52_1D7B,
        0x0003_7F02,
    ],
    [
        0x3F8F_7991,
        0x0000_049A,
        0xC46D_C522,
        0xFF5C_9E3C,
        0x0003_8E84,
    ],
    [
        0x3F8B_B670,
        0x0000_0697,
        0xC331_AD68,
        0xFF67_4D67,
        0x0003_9F20,
    ],
    [
        0x3F87_E03A,
        0x0000_008E,
        0xC20B_27CA,
        0xFF72_246B,
        0x0003_AFCE,
    ],
    [
        0x3F83_F849,
        0x0000_0B94,
        0xC0FA_7D5C,
        0xFF7D_2DAB,
        0x0003_B6E7,
    ],
];

/// An upper estimate of the SPIR-V words of [`SINC`] in a module: its 160 constants, the
/// composite, and the array, pointer, and variable.
pub(crate) const TABLE_WORDS: u32 = 160 * 4 + 163 + 24;

/// Declare [`SINC`] in the module.
pub(crate) fn declare(mb: &mut ModuleBuilder) -> BResult<GlobalId> {
    super::declare_words(mb, SINC.as_flattened())
}

/// The `float` bits at or below which `sin(x)` is `x` and `cos(x)` is 1 (`2^-12`).
const TINY: u32 = 0x3980_0000;

/// `floor(a d / 2^18)` of the `i32` `a` and the offset `d` (below `2^18`), as an `i32` (the
/// callers' values keep it in range): the 64-bit product's bits from 18 on.
fn scaled(e: &mut E<'_, '_>, a: Value, d: Value) -> BResult<Value> {
    let p = e.fb.binary(BinOp::SMulExtended, a, d)?;
    let lo = e.fb.composite_extract(p, &[0])?;
    let hi = e.fb.composite_extract(p, &[1])?;
    let (k18, k14) = (e.u(18)?, e.u(14)?);
    let lo = e.iop(BinOp::ShiftRightLogical, lo, k18)?;
    let hi = e.iop(BinOp::ShiftLeftLogical, hi, k14)?;
    e.iop(BinOp::BitwiseOr, hi, lo)
}

/// The model of the hardware's `sinc` of `|t| <= 1` (the module's doc), which Metal's `sinpi`,
/// `cospi` and `tanpi` take too (`pi`).
pub(super) fn sinc(e: &mut E<'_, '_>, at: Value) -> BResult<Value> {
    let cells = e.c(8_388_608.0)?;
    let scaled_t = e.mul(at, cells)?;
    let m = e.utrunc(scaled_t)?;
    let k18 = e.u(18)?;
    let j = e.iop(BinOp::ShiftRightLogical, m, k18)?;
    let last = e.u(31)?;
    let past = e.iop(BinOp::UGreaterThan, j, last)?;
    let j = e.sel(past, last, j)?;
    let mask = e.u(0x3_FFFF)?;
    let d = e.iop(BinOp::BitwiseAnd, m, mask)?;
    let five = e.u(5)?;
    let first = e.iop(BinOp::IMul, j, five)?;
    let table = e.sinc_table()?;
    let base = e.fb.module().global_value(table)?;
    let word = |e: &mut E<'_, '_>, i: u32| -> BResult<Value> {
        let i = e.u(i)?;
        let i = e.iop(BinOp::IAdd, first, i)?;
        let p = e.fb.access_chain(base, &[i])?;
        e.fb.load(p, MemAccess::default())
    };
    let (start, low, c1, c2, c3) = (
        word(e, 0)?,
        word(e, 1)?,
        word(e, 2)?,
        word(e, 3)?,
        word(e, 4)?,
    );
    let s = scaled(e, c3, d)?;
    let t2 = e.iop(BinOp::IAdd, c2, s)?;
    let s = scaled(e, t2, d)?;
    let t1 = e.iop(BinOp::IAdd, c1, s)?;
    let p = scaled(e, t1, d)?;
    let w = e.iop(BinOp::IAdd, low, p)?;
    let (half, k12) = (e.u(2048)?, e.u(12)?);
    let w = e.iop(BinOp::IAdd, w, half)?;
    let q = e.iop(BinOp::ShiftRightArithmetic, w, k12)?;
    let bits = e.iop(BinOp::IAdd, start, q)?;
    let v = e.float(bits)?;
    // `S(0) = RN(π/2)`, `S(1) = 1`.
    let zero = e.u(0)?;
    let at_zero = e.iop(BinOp::IEqual, m, zero)?;
    let half_pi = e.c(std::f32::consts::FRAC_PI_2)?;
    let v = e.sel(at_zero, half_pi, v)?;
    let one = e.c(1.0)?;
    e.sel(past, one, v)
}

/// Metal's precise `which` (`Sin` or `Cos`) of `x` (the module's doc).
pub(crate) fn sin_cos(e: &mut E<'_, '_>, which: Trig, x: Value) -> BResult<Value> {
    let fun = match which {
        Trig::Sin => Fun::Sin,
        Trig::Cos => Fun::Cos,
    };
    let x = read(e, fun, x)?;
    let (r, k) = quarter::turns(e, x, Error::Dropped)?;
    let (one_u, two_u, zero_u) = (e.u(1)?, e.u(2)?, e.u(0)?);
    let odd = e.iop(BinOp::BitwiseAnd, k, one_u)?;
    let odd = e.iop(BinOp::INotEqual, odd, zero_u)?;
    let flip = e.iop(BinOp::BitwiseAnd, k, two_u)?;
    let flip = e.iop(BinOp::INotEqual, flip, zero_u)?;
    let ar = e.abs(r)?;
    let one = e.c(1.0)?;
    let rest = e.sub(one, ar)?;
    let t = match which {
        Trig::Sin => e.sel(odd, rest, r)?,
        Trig::Cos => {
            let minus_r = e.fb.unary(UnOp::FNegate, r)?;
            e.sel(odd, minus_r, rest)?
        }
    };
    let minus_t = e.fb.unary(UnOp::FNegate, t)?;
    let t = e.sel(flip, minus_t, t)?;
    let at = e.abs(t)?;
    let s = sinc(e, at)?;
    let v = e.mul(t, s)?;
    // At or below 2^-12: `x` or 1; an infinity or a NaN: Metal's NaN.
    let ax = e.abs(x)?;
    let bits = e.bits(ax)?;
    let tiny = e.u(TINY + 1)?;
    let is_tiny = e.iop(BinOp::ULessThan, bits, tiny)?;
    let small = match which {
        Trig::Sin => x,
        Trig::Cos => one,
    };
    let v = e.sel(is_tiny, small, v)?;
    let inf = e.u(0x7F80_0000)?;
    let finite = e.iop(BinOp::ULessThan, bits, inf)?;
    let nan = e.metal_nan(None)?;
    e.sel(finite, v, nan)
}

/// The `float` bits below which Metal's `half` `sin(x)` is `x` (`1/64`).
const HALF_SIN_TINY: u32 = 0x3C80_0000;

/// Metal's `half` `sin` or `cos` (`which`) before its rounding to `half`, of a `half` `x` held as
/// `float`: its own form, which follows Apple's runtime library. The quarter turns `k` are the low
/// bits of `s = fma(x, c0, m)`, `m = 1.5 2^23` (`1.5 2^23 + 1` for `cos`, a quarter turn on), and
/// `r = fma(x, c1, fma(x, c0, m - s))` of `2/π`'s first two parts ([`quarter::QUARTER`]; the
/// `float` routine adds a third), folded and signed as the `float` routine's (`t = r` for an even
/// `k` and `1 - |r|` for an odd one, negated where `k`'s bit 1 is set), and `t S(t)` of the
/// hardware's `sinc` ([`sinc`]). `sin(x)` is `x` below `1/64` in magnitude; an infinity or a NaN
/// gives Metal's NaN.
pub(crate) fn half_sin_cos(e: &mut E<'_, '_>, which: Trig, x: Value) -> BResult<Value> {
    let ax = e.abs(x)?;
    let bits = e.bits(ax)?;
    let inf = e.u(0x7F80_0000)?;
    let finite = e.iop(BinOp::ULessThan, bits, inf)?;
    let zero = e.c(0.0)?;
    // An infinity or a NaN reduces 0 instead, so that no value selected away reaches the `sinc`'s
    // integer conversion.
    let xr = e.sel(finite, x, zero)?;
    let [c0, c1, _] = quarter::QUARTER.map(f32::from_bits);
    let magic = match which {
        Trig::Sin => 12_582_912.0,
        Trig::Cos => 12_582_913.0,
    };
    let (c0, c1, m) = (e.c(c0)?, e.c(c1)?, e.c(magic)?);
    let s = e.fused(xr, c0, m)?;
    let k = e.bits(s)?;
    let minus_k = e.sub(m, s)?;
    let r = e.fused(xr, c0, minus_k)?;
    let r = e.fused(xr, c1, r)?;
    let (one_u, two_u, zero_u) = (e.u(1)?, e.u(2)?, e.u(0)?);
    let odd = e.iop(BinOp::BitwiseAnd, k, one_u)?;
    let odd = e.iop(BinOp::INotEqual, odd, zero_u)?;
    let flip = e.iop(BinOp::BitwiseAnd, k, two_u)?;
    let flip = e.iop(BinOp::INotEqual, flip, zero_u)?;
    let ar = e.abs(r)?;
    let one = e.c(1.0)?;
    let rest = e.sub(one, ar)?;
    let t = e.sel(odd, rest, r)?;
    let minus_t = e.fb.unary(UnOp::FNegate, t)?;
    let t = e.sel(flip, minus_t, t)?;
    let at = e.abs(t)?;
    let st = sinc(e, at)?;
    let v = e.mul(t, st)?;
    let v = match which {
        Trig::Sin => {
            let tiny = e.u(HALF_SIN_TINY)?;
            let is_tiny = e.iop(BinOp::ULessThan, bits, tiny)?;
            e.sel(is_tiny, x, v)?
        }
        Trig::Cos => v,
    };
    let nan = e.metal_nan(None)?;
    e.sel(finite, v, nan)
}
