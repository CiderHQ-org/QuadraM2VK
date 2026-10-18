//! Metal's fast `rsqrt`: its hardware's inverse square root, the one Metal's fast `rsqrt` answers
//! and its `asin` and `acos` in both modes and precise `asinh` and `acosh` take inside.
//! [`metal_rsqrt`] is [`super::Fun::FastRsqrt`]'s routine, which those routines call
//! ([`E::fast_rsqrt`]). A standalone `fast::rsqrt` or `fast::sqrt` keeps the device's
//! `InverseSqrt` (a recorded near-equivalent within Apple's bound, as the fast division keeps its
//! `RN(1 / b)`), so a small shader does not grow by the routine.
//!
//! Read off Metal's words (on the M3 Max, on the M4, the same at every operand): `fast::rsqrt` of
//! every `float` of `[1, 4)` and of every 256th positive `float`. Metal's root of `y = m 4^k`, `m`
//! in `[1, 4)`, is its root of `m` scaled by `2^-k`, and its root of `m` is the exact `1 / sqrt(m)`
//! plus a smooth error per 32nd of each binade of `m` (up to about a third of a unit), rounded to
//! nearest: it differs from the correctly rounded root at 725,821 of the 16,777,216 `float` values
//! of `[1, 4)`. [`TABLE`] fits the smooth part, a quartic per 32nd by its five `f32` coefficients
//! (in units of `2^-24`), so the routine is `RN(1 / sqrt(m) + eps(m))`: Metal's word at 16,746,537
//! `float` values of `[1, 4)` (99.817 %) and 99.805 % of the sampled normal `float` values, one
//! unit off at the rest (an open near-equivalent).

use crate::lower::arith::libm::E;
use crate::spirv::builder::ModuleBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, Value};
use crate::spirv::ir::{BinOp, MemAccess};

/// Metal's inverse square root's error per 32nd of a binade of `m` in `[1, 4)`: row `(k >> 18) +
/// 32 b` of the fraction field `k` of `m` in binade `b` (0 for `[1, 2)`, 1 for `[2, 4)`), `eps =
/// c0 + t (c1 + t (c2 + t (c3 + t c4)))` at `t = (k mod 2^18) 2^-18`, in units of `2^-24` (`f32`
/// bits).
pub(crate) const TABLE: [[u32; 5]; 64] = [
    [
        0x39D9_A548,
        0x3F8A_DE5A,
        0xC0B1_2C77,
        0x4111_27AC,
        0xC093_2A42,
    ],
    [
        0x3C9F_B9B1,
        0x3F94_4437,
        0xC0A5_7EC3,
        0x40F0_C515,
        0xC061_7408,
    ],
    [
        0xBB7E_3C94,
        0x3F48_60CE,
        0xC082_BD8A,
        0x40CB_F450,
        0xC048_125A,
    ],
    [
        0xBD86_870F,
        0x3F7B_F8E7,
        0xC07F_2586,
        0x40BD_49D1,
        0xC030_5985,
    ],
    [
        0x3DB0_FD94,
        0x3EFE_9CB4,
        0xC043_5A08,
        0x409F_B6D1,
        0xC01C_D405,
    ],
    [
        0x3D87_842E,
        0x3E9B_0B75,
        0xC01B_ADA7,
        0x4085_4249,
        0xC006_BAA2,
    ],
    [
        0xBC42_ABB3,
        0x3F0A_9035,
        0xC022_AF86,
        0x4081_A0F1,
        0xC007_22A9,
    ],
    [
        0x3D4F_76AC,
        0xBC2C_CAC4,
        0xBFB6_6E80,
        0x4044_2648,
        0xBFDA_69E4,
    ],
    [
        0x3DBF_5C77,
        0x3E90_9F4B,
        0xC00B_059B,
        0x4055_E04C,
        0xBFC5_D510,
    ],
    [
        0xBB89_1985,
        0x3F48_1D44,
        0xC024_52B3,
        0x404D_A207,
        0xBFAE_8450,
    ],
    [
        0xBD8C_2ECD,
        0x3F56_F550,
        0xC00F_6C14,
        0x4030_ED89,
        0xBF9B_89DC,
    ],
    [
        0x3D9E_449A,
        0x3D09_F390,
        0xBF5F_38B3,
        0x3FF1_031B,
        0xBF8C_7C9D,
    ],
    [
        0x3CAD_CCF2,
        0x3F0D_1232,
        0xBFF4_CBEE,
        0x401B_3C98,
        0xBF70_B150,
    ],
    [
        0x3C5A_F741,
        0xBD58_057D,
        0xBEF8_F647,
        0x3FB9_B885,
        0xBF59_B990,
    ],
    [
        0xBD5E_B8EC,
        0x3ED1_FEBC,
        0xBFC1_8926,
        0x4003_5BFA,
        0xBF5F_544E,
    ],
    [
        0x3CB4_ADDD,
        0x3EA5_CF8A,
        0xBF91_EB2E,
        0x3FD6_A113,
        0xBF3B_4574,
    ],
    [
        0xBDCE_901B,
        0x3F23_F497,
        0xBFCD_672F,
        0x3FD4_7A87,
        0xBF23_103E,
    ],
    [
        0x3DA4_0977,
        0x3DBE_A274,
        0xBF1B_2170,
        0x3F83_EA4B,
        0xBF1E_DCE8,
    ],
    [
        0x3DD1_408A,
        0x3E52_B163,
        0xBF86_B137,
        0x3F88_9830,
        0xBE8D_6290,
    ],
    [
        0x3D25_83EE,
        0x3EAD_377D,
        0xBF8E_57D7,
        0x3FB3_7822,
        0xBF0E_A7A2,
    ],
    [
        0xBC77_EE3A,
        0x3F1A_CD9E,
        0xBFA3_2322,
        0x3FA1_CB2F,
        0xBEF0_7009,
    ],
    [
        0xBC91_EA98,
        0x3EF5_FA4B,
        0xBFB9_B80E,
        0x3FBB_70F2,
        0xBEF1_4999,
    ],
    [
        0x3B53_7F7F,
        0x3D95_3A78,
        0xBEB9_9178,
        0x3F0C_213C,
        0xBEB6_0FC5,
    ],
    [
        0x3CFC_3EDF,
        0x3E9E_2DA3,
        0xBF70_28A6,
        0x3F83_1ABD,
        0xBEB6_6013,
    ],
    [
        0x3D8D_8A3B,
        0xBE3E_3877,
        0xBD02_F8E1,
        0x3F03_2B5E,
        0xBE9C_588C,
    ],
    [
        0xBD8E_D5C8,
        0x3D63_077B,
        0x3D43_F066,
        0x3E7E_F632,
        0xBEA6_66F7,
    ],
    [
        0xBD30_7445,
        0x3F17_B577,
        0xBF94_7BC5,
        0x3F5F_5A11,
        0xBE5E_5587,
    ],
    [
        0x3D0E_4CB1,
        0xBD61_0726,
        0xBF0F_C52E,
        0x3F48_F2AB,
        0xBE83_CB68,
    ],
    [
        0xBD5D_AADD,
        0x3EC3_B4A5,
        0xBF47_09C8,
        0x3F33_C738,
        0xBE71_AAA3,
    ],
    [
        0x3C75_1580,
        0x3E74_B043,
        0xBF2B_0C91,
        0x3F48_2105,
        0xBE81_22FA,
    ],
    [
        0xBC39_4794,
        0x3E7E_80BE,
        0xBF63_5C71,
        0x3F5B_3CD4,
        0xBE71_22B1,
    ],
    [
        0x3DC0_F279,
        0xBE11_034E,
        0x3E62_AF3F,
        0x3AA5_DF59,
        0xBE67_3A63,
    ],
    [
        0x3D8E_B56D,
        0x3F5C_518A,
        0xC082_43AF,
        0x40C1_B049,
        0xC037_C871,
    ],
    [
        0x3D03_F11D,
        0x3F71_AA73,
        0xC083_9481,
        0x40BA_AAC8,
        0xC029_4C94,
    ],
    [
        0xBDA0_5A60,
        0x3F6C_7EF7,
        0xC04A_A125,
        0x4093_5268,
        0xC011_232A,
    ],
    [
        0x3C33_29F6,
        0x3EB0_BAE9,
        0xC005_3EE8,
        0x4068_BFE4,
        0xBFF6_1A57,
    ],
    [
        0xBC8D_18C0,
        0x3F5F_F259,
        0xC038_AAC5,
        0x4074_A6D6,
        0xBFD6_0B2B,
    ],
    [
        0xBBB2_6C38,
        0x3F1A_5B4A,
        0xC004_972B,
        0x403A_1508,
        0xBFB8_A1B1,
    ],
    [
        0xBC2B_AA4D,
        0x3F1C_6D18,
        0xC009_2718,
        0x4037_81AD,
        0xBFA5_625E,
    ],
    [
        0x3CEB_1182,
        0x3EFE_370E,
        0xBFFA_F7FF,
        0x4024_CC07,
        0xBF97_45E8,
    ],
    [
        0xBD3B_6143,
        0x3E9D_48E1,
        0xBF77_7400,
        0x3FE9_8B67,
        0xBF8B_5784,
    ],
    [
        0x3CEA_C677,
        0xB914_F100,
        0xBF13_12A4,
        0x3FBE_5ED0,
        0xBF70_87DF,
    ],
    [
        0xBB2E_8F75,
        0x3D81_0130,
        0xBF7C_6316,
        0x3FE7_CE27,
        0xBF6D_0765,
    ],
    [
        0x3DAC_F134,
        0x3DAA_880E,
        0xBF1B_BB32,
        0x3F99_BDE6,
        0xBF44_40F6,
    ],
    [
        0xBBEE_FAB9,
        0x3E1F_E89A,
        0xBF37_1A0E,
        0x3F91_C795,
        0xBF34_AA63,
    ],
    [
        0xBC2B_06C5,
        0x3D82_FBBD,
        0xBF29_3D90,
        0x3FAE_22AB,
        0xBF28_6AE4,
    ],
    [
        0xBD05_6313,
        0xBD5D_9DBB,
        0xBDAB_48AA,
        0x3F41_C2E6,
        0xBF0F_5A2A,
    ],
    [
        0x3CCE_9CD1,
        0x3E95_B1E9,
        0xBFA4_D4B5,
        0x3FBD_627C,
        0xBF08_0CAD,
    ],
    [
        0xBC9F_5063,
        0x3E82_49FA,
        0xBF78_CAE9,
        0x3F9C_4B5A,
        0xBECE_1F1D,
    ],
    [
        0x3DA3_18A3,
        0xBE1C_2771,
        0x3DF2_5CED,
        0x3EF8_2A55,
        0xBEE1_53F1,
    ],
    [
        0xBD12_110B,
        0x3E8C_08B8,
        0xBF1E_E1F5,
        0x3F3E_960E,
        0xBED0_2000,
    ],
    [
        0x3DA1_73DE,
        0x3E20_3934,
        0xBF65_D9F7,
        0x3F97_3811,
        0xBEBF_0D1F,
    ],
    [
        0xBDD4_90BB,
        0x3EB6_C5F2,
        0xBECF_DE2B,
        0x3E80_3EB2,
        0xBDC8_7A82,
    ],
    [
        0xBB67_0D8D,
        0x3DA1_62C8,
        0xBE1E_C4B3,
        0x3EC4_485D,
        0xBEA3_347E,
    ],
    [
        0xBC87_0346,
        0x3E1C_B7E6,
        0x3C8E_8608,
        0x3E31_3918,
        0xBE99_B067,
    ],
    [
        0x3CD4_D1FB,
        0x3F04_5C42,
        0xBFAC_7676,
        0x3F8C_C303,
        0xBE9F_5584,
    ],
    [
        0x3DDF_71C6,
        0x3E0B_0763,
        0xBF5A_5BC6,
        0x3F6F_1D1C,
        0xBE6B_49B9,
    ],
    [
        0x3DC3_F94C,
        0xBEDC_F690,
        0x3F03_9C3F,
        0x3D5B_6287,
        0xBE80_0515,
    ],
    [
        0x3DDB_FEA9,
        0xBE7B_98D5,
        0x3E87_D9ED,
        0x3B02_0E39,
        0xBE16_937E,
    ],
    [
        0xBCA1_5416,
        0x3E40_F303,
        0xBE50_234E,
        0x3E8A_044D,
        0xBE2E_B6D3,
    ],
    [
        0x3D81_9004,
        0x3E1E_6115,
        0xBF06_AD9E,
        0x3F1C_92E1,
        0xBE5A_C76E,
    ],
    [
        0xBD12_5845,
        0x3EB0_9C15,
        0xBF78_8C58,
        0x3F5F_B8B1,
        0xBE3D_DC2B,
    ],
    [
        0x3E1A_0DA6,
        0xBE44_2A42,
        0xBD09_0233,
        0x3E94_A18A,
        0xBE4F_645A,
    ],
    [
        0xBDDE_4A49,
        0x3EFC_1A0A,
        0xBEFD_384C,
        0xBD80_9F8E,
        0x3E5C_970A,
    ],
];

/// The coefficients per row of [`TABLE`].
const COEFFICIENTS: u32 = 5;

/// An upper estimate of the SPIR-V words of [`TABLE`] in a module: its 320 constants, the
/// composite, and the array, pointer, and variable.
pub(crate) const TABLE_WORDS: u32 = 320 * 4 + 323 + 24;

/// Declare [`TABLE`] in the module.
pub(crate) fn declare(mb: &mut ModuleBuilder) -> BResult<GlobalId> {
    super::declare_words(mb, TABLE.as_flattened())
}

/// Metal's inverse square root of `y`, for a positive normal `y`: `y = m 4^k` with `m` in `[1,
/// 4)` (`y`'s fraction field under an exponent field of 127 or 128), `r = RN(hi + RN(lo + eps(m)
/// 2^-24))` of `hi = RN(1 / sqrt(m))` (the module's correctly rounded `rsqrt`, [`E::rsqrt`]) and
/// `lo = RN(RN(hi d) / 2)` of `d = 1 - m hi^2` (formed from exact products, so `hi + lo` is the
/// exact root to a few units of `2^-48 hi`), then `r`'s exponent field moved by `-k`. Metal's
/// words at its special operands: a zero or denormal `y` gives `+inf`, `+inf`
/// gives `+0`, and a NaN gives a NaN (`y`; Metal's is `0x7fc00000`, its callers' NaN). Its
/// callers pass no negative operand, whose answer is its magnitude's.
pub(crate) fn metal_rsqrt(e: &mut E<'_, '_>, y: Value) -> BResult<Value> {
    let table = e.rsqrt_table()?;
    let b = e.bits(y)?;
    let (fraction, field, k23) = (e.u(0x007F_FFFF)?, e.u(0xFF)?, e.u(23)?);
    let k = e.iop(BinOp::BitwiseAnd, b, fraction)?;
    let n = e.iop(BinOp::ShiftRightLogical, b, k23)?;
    let n = e.iop(BinOp::BitwiseAnd, n, field)?;
    // `b`: 0 for an odd exponent field (an even power of 2), 1 for an even one.
    let one_u = e.u(1)?;
    let odd = e.iop(BinOp::BitwiseAnd, n, one_u)?;
    let binade = e.iop(BinOp::BitwiseXor, odd, one_u)?;
    let one_bits = e.u(0x3F80_0000)?;
    let shifted = e.iop(BinOp::ShiftLeftLogical, binade, k23)?;
    let mb = e.iop(BinOp::IAdd, one_bits, shifted)?;
    let mb = e.iop(BinOp::BitwiseOr, mb, k)?;
    let m = e.float(mb)?;
    let hi = e.rsqrt(m)?;
    // `d = 1 - m hi^2`: `hi^2 = p + pe` and `m p = q + qe` exactly, `1 - q` exact (Sterbenz),
    // and the rest far below `d`'s unit.
    let p = e.mul(hi, hi)?;
    let pe = e.exact_error(hi, hi, p)?;
    let q = e.mul(m, p)?;
    let qe = e.exact_error(m, p, q)?;
    let w = e.mul(m, pe)?;
    let one = e.c(1.0)?;
    let d1 = e.sub(one, q)?;
    let d2 = e.add(qe, w)?;
    let d = e.sub(d1, d2)?;
    let lo = e.mul(hi, d)?;
    let half = e.c(0.5)?;
    let lo = e.mul(lo, half)?;
    // `eps(m)`: row `(k >> 18) + 32 b`, at `t = (k mod 2^18) 2^-18`.
    let (low, k18, k5) = (e.u(0x3_FFFF)?, e.u(18)?, e.u(5)?);
    let row = e.iop(BinOp::ShiftRightLogical, k, k18)?;
    let upper = e.iop(BinOp::ShiftLeftLogical, binade, k5)?;
    let row = e.iop(BinOp::IAdd, row, upper)?;
    let t = e.iop(BinOp::BitwiseAnd, k, low)?;
    let t = e.ufloat(t)?;
    let scale = e.c(1.0 / 262_144.0)?;
    let t = e.mul(t, scale)?;
    let width = e.u(COEFFICIENTS)?;
    let first = e.iop(BinOp::IMul, row, width)?;
    let base = e.fb.module().global_value(table)?;
    let coefficient = |e: &mut E<'_, '_>, j: u32| -> BResult<Value> {
        let j = e.u(j)?;
        let i = e.iop(BinOp::IAdd, first, j)?;
        let p = e.fb.access_chain(base, &[i])?;
        let w = e.fb.load(p, MemAccess::default())?;
        e.float(w)
    };
    // Horner's rule in plain `float`, as the table was fitted.
    let mut eps = coefficient(e, COEFFICIENTS - 1)?;
    for j in (0..COEFFICIENTS - 1).rev() {
        let c = coefficient(e, j)?;
        let p = e.mul(eps, t)?;
        eps = e.add(p, c)?;
    }
    let unit = e.c(1.0 / 16_777_216.0)?;
    let eps = e.mul(eps, unit)?;
    let tail = e.add(lo, eps)?;
    let r = e.add(hi, tail)?;
    // `r` in `[1/2, 1]` scaled by `2^((127 + b - n) / 2)` of the exponent field `n` (an even
    // difference), through `r`'s exponent field.
    let rb = e.bits(r)?;
    let bias = e.u(0x7F)?;
    let s = e.iop(BinOp::IAdd, bias, binade)?;
    let s = e.iop(BinOp::ISub, s, n)?;
    let s = e.iop(BinOp::ShiftRightArithmetic, s, one_u)?;
    let s = e.iop(BinOp::ShiftLeftLogical, s, k23)?;
    let v = e.iop(BinOp::IAdd, rb, s)?;
    // Metal's specials: a zero exponent field gives `+inf`, and `+inf` gives `+0`.
    let zero = e.u(0)?;
    let infinity = e.u(0x7F80_0000)?;
    let tiny = e.iop(BinOp::IEqual, n, zero)?;
    let v = e.sel(tiny, infinity, v)?;
    let inf = e.iop(BinOp::IEqual, b, infinity)?;
    let v = e.sel(inf, zero, v)?;
    let r = e.float(v)?;
    let nan = e.is_nan(y)?;
    e.sel(nan, y, r)
}
