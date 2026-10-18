//! Metal's reciprocal: the `1 / y` of Metal's fast division (`arcp`), of its precise `sinh`,
//! `cosh`, and `tanh`, which compose their exponentials with it, and of its texture unit's cube
//! face coordinate. [`metal_rcp`] is [`super::Fun::Recip`]'s routine, which the other routines call
//! ([`E::recip`]) and any lowering may per lane.
//!
//! Read off every `float` of the M4 (a library compiled at run time with
//! `mathMode` fast): Metal's reciprocal of a normal `y` whose reciprocal is normal is the
//! reciprocal `r(m)` of its significand `m` in `[1, 2)` with the exponent negated and the sign
//! kept, and `r(m)` is `1 / m` plus a smooth error per 64th of `m` (up to about a fifth of a
//! unit) under a noise of a few thousandths of a unit, rounded to nearest: it differs from
//! `RN(1 / m)` at 868,174 of the 8,388,608 significands, by one unit. [`TABLE`] fits the smooth
//! part, a cubic per 64th by its four `f32` coefficients (scaled by `2^-24`, least squares to the
//! bounds Metal's words set, each constant term then moved to the most words), so the routine is
//! `RN(1 / m + eps(m))`: Metal's word at 8,361,659 of the significands (99.68 %), one unit off at
//! the other 26,949 (an open near-equivalent).

use crate::lower::arith::libm::E;
use crate::spirv::builder::ModuleBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, Value};
use crate::spirv::ir::{BinOp, MemAccess};

/// Metal's reciprocal error per 64th of the significand: row `k >> 17` of the fraction field `k`,
/// `eps = c0 + t (c1 + t (c2 + t c3))` at `t = (k mod 2^17) 2^-17`, in units of `2^-24` already
/// (`f32` bits).
pub(crate) const TABLE: [[u32; 4]; 64] = [
    [0xB214_8996, 0x3406_49CA, 0xB47F_97D9, 0x3405_171E],
    [0xB0FC_71E5, 0xB234_0DF9, 0x33D4_9DBE, 0xB3CB_D992],
    [0xB227_87AA, 0x33EE_63DE, 0xB460_3011, 0x33E9_D938],
    [0xB126_868E, 0xB1FF_9A1E, 0x33BB_75A6, 0xB3AF_2E82],
    [0xB1C0_FDC3, 0x33C5_3972, 0xB43D_B71E, 0x33C9_14CE],
    [0xAD07_6890, 0xB1D8_03F7, 0x33A5_F505, 0xB39E_28EF],
    [0xB186_F82F, 0x33B6_A66E, 0xB428_525F, 0x33AF_B697],
    [0x314F_E2CF, 0xB182_9ACB, 0x3382_BD4A, 0xB383_D414],
    [0xB122_9A6A, 0x3371_209A, 0xB3FE_AC88, 0x338D_A07C],
    [0xB0C1_A142, 0xAFDB_8FF3, 0x336F_237F, 0xB36B_FAB8],
    [0xB148_D921, 0x337C_2C6E, 0xB3E4_9D7D, 0x3372_9779],
    [0x316E_41C5, 0x3012_ACD3, 0x333F_B7C3, 0xB358_E276],
    [0xB146_641A, 0x338E_6B8F, 0xB3F5_7A65, 0x3376_FEFF],
    [0x319B_F8FC, 0xB186_7911, 0x332D_9920, 0xB325_8748],
    [0x30F3_C936, 0x331C_00BD, 0xB3A1_4826, 0x3335_92CA],
    [0x3168_FC4A, 0x301D_C599, 0x332D_5B3C, 0xB32E_9CC0],
    [0x313D_5D28, 0x3309_8103, 0xB398_5D03, 0x3335_619C],
    [0x3181_D0D7, 0x316E_2FCB, 0x3312_B202, 0xB311_F8F0],
    [0x31E2_DB1E, 0x3313_E265, 0xB39B_6842, 0x3329_8F7C],
    [0x31E0_7D3E, 0xB143_9D5C, 0x3301_6D0D, 0xB2F0_4595],
    [0xB11D_A12E, 0x3348_BAB3, 0xB3A2_0B07, 0x3322_6770],
    [0x31B4_B536, 0x30C7_DA56, 0x32CD_90FA, 0xB2A6_E469],
    [0x3158_AF01, 0x32EB_414F, 0xB370_6A3F, 0x32FD_B1D1],
    [0x313E_6377, 0xB178_465D, 0x32B7_57E1, 0xB2BE_AB31],
    [0x31B3_AC90, 0x32D5_BB9A, 0xB355_0129, 0x32E0_B357],
    [0x31BA_2713, 0xB104_C3F2, 0x328E_B66E, 0xB2A1_1332],
    [0x31F1_EEFA, 0x3313_BD9A, 0xB37B_41A6, 0x32D9_970A],
    [0x3212_AC4F, 0xB1CA_CFAD, 0x3298_A70A, 0xB289_75EC],
    [0x3187_B6E7, 0x32B4_D559, 0xB329_AFA5, 0x32B0_4F61],
    [0x31AD_C0DB, 0xB135_EFED, 0x327C_5986, 0xB295_6174],
    [0x31E6_ECF4, 0x32BC_32C4, 0xB346_FD7E, 0x32C9_137A],
    [0x31A1_C5B2, 0xB1B7_B003, 0x326B_FFDD, 0xB21C_A5E4],
    [0x323C_7597, 0x322F_AEDB, 0xB2F1_E3F3, 0x3284_68AF],
    [0x3206_5907, 0xB155_8FD5, 0x328C_F3C6, 0xB264_A7C7],
    [0x31FD_CCD7, 0x328D_B75E, 0xB31F_438F, 0x32A3_9033],
    [0x31AE_2EFF, 0xB199_5AED, 0x3282_C156, 0xB21B_92C1],
    [0x31DC_3D10, 0x3186_B786, 0xB299_E438, 0x3244_DA0E],
    [0x3154_B814, 0xB03C_D244, 0x321A_CA9C, 0xB25B_227F],
    [0x31C6_0728, 0x325D_AD26, 0xB301_B452, 0x328C_BAB8],
    [0x318B_2BBB, 0xB11F_FFED, 0x324B_06D1, 0xB1E8_8D5A],
    [0x31DE_C4B9, 0x31CF_C5AD, 0xB286_B87C, 0x3234_F881],
    [0x31E4_5B98, 0x314F_9841, 0x320D_867E, 0xB22E_97AF],
    [0x3202_72D1, 0x320D_B866, 0xB26E_6A3A, 0x3207_42D1],
    [0x321B_2903, 0x310B_7BAB, 0x3100_5DB6, 0xB226_E626],
    [0x3146_FFE0, 0x323A_C1C5, 0xB2A6_9A7C, 0x321E_E006],
    [0x3154_EC03, 0xB162_8C37, 0x315C_6263, 0xB206_52E8],
    [0x3123_BE15, 0x320A_437D, 0xB257_2532, 0x31FC_1527],
    [0x31AA_B487, 0x3108_9B7F, 0x31A3_6760, 0xB227_DF03],
    [0x321D_9B42, 0xB1A4_2AAC, 0xB1E7_145D, 0x31E2_6E9C],
    [0x3183_2E38, 0xB0CE_6720, 0x321B_10B6, 0xB211_A59F],
    [0x322F_7BEF, 0x3242_26E2, 0xB2A0_63E3, 0x3205_64DE],
    [0x3232_DA05, 0xB1BC_432D, 0x31B1_C95B, 0xB1D0_0858],
    [0x3186_F10B, 0x2FFC_19A4, 0xB1E3_9A97, 0x31C0_7093],
    [0x3151_D208, 0x313C_4297, 0x31EC_EC85, 0xB1BC_7069],
    [0x31F1_D8A4, 0xB11E_B11A, 0xB1AC_503C, 0x318F_E554],
    [0x3177_B355, 0xAF73_0117, 0x31CD_AA51, 0xB1B2_C583],
    [0x3189_0398, 0x321D_EF92, 0xB260_648A, 0x31D7_2FED],
    [0x31D2_5CF7, 0xAEC5_5547, 0x31A9_A37D, 0xB195_652F],
    [0x31E0_C597, 0x3138_231B, 0xB0EB_AEA1, 0x30EC_BC02],
    [0x3219_1B31, 0x3120_8F08, 0x312F_B150, 0xB20D_F155],
    [0x31BC_D310, 0x324F_E2D4, 0xB287_661A, 0x31D8_FF8A],
    [0x3208_CFBA, 0xB104_09DC, 0x3101_3BBB, 0xB11A_3291],
    [0x31BC_6932, 0x30D8_C0FA, 0xB207_224F, 0x31AF_AEEA],
    [0x317F_68FC, 0x31A4_1CE2, 0xB217_BC4F, 0x31FF_34D7],
];

/// The coefficients per row of [`TABLE`].
const COEFFICIENTS: u32 = 4;

/// An upper estimate of the SPIR-V words of [`TABLE`] in a module: its 256 constants, the
/// composite, and the array, pointer, and variable.
pub(crate) const TABLE_WORDS: u32 = 256 * 4 + 259 + 24;

/// Declare [`TABLE`] in the module.
pub(crate) fn declare(mb: &mut ModuleBuilder) -> BResult<GlobalId> {
    super::declare_words(mb, TABLE.as_flattened())
}

/// Metal's reciprocal of `y`. For a normal `y` up to `2^126` in magnitude, `r = RN(1 / m +
/// eps(m))` of its significand `m`, the sum formed from `r0 = RN(1 / m)` (the module's precise
/// division) and its exact residual `1 - m r0` as `r0 + ((1 - m r0) r0 + eps)`, each step's
/// rounding far below the unit `r` rounds to; then `r`'s exponent field moved by `y`'s, and `y`'s
/// sign (Metal's reciprocal of `-y` is the negation of its reciprocal of `y`). Metal's flushes,
/// as measured: a zero or denormal `y` gives the infinity of its sign, and a
/// `y` past `2^126` (an infinity included), whose reciprocal would be denormal, the zero of its
/// sign. A NaN gives a NaN (`y`; Metal's NaN bits are its callers').
pub(crate) fn metal_rcp(e: &mut E<'_, '_>, y: Value) -> BResult<Value> {
    let table = e.rcp_table()?;
    let b = e.bits(y)?;
    let (fraction, one_bits) = (e.u(0x007F_FFFF)?, e.u(0x3F80_0000)?);
    let k = e.iop(BinOp::BitwiseAnd, b, fraction)?;
    let mb = e.iop(BinOp::BitwiseOr, k, one_bits)?;
    let m = e.float(mb)?;
    let one = e.c(1.0)?;
    let r0 = e.div(one, m)?;
    // `1 - m r0` exactly: `1 - p` by Sterbenz's lemma, less the product's exact error.
    let p = e.mul(r0, m)?;
    let error = e.exact_error(r0, m, p)?;
    let residual = e.sub(one, p)?;
    let residual = e.sub(residual, error)?;
    let d = e.mul(residual, r0)?;
    // `eps(m)`: row `k >> 17`, at `t = (k mod 2^17) 2^-17`.
    let (low, k17) = (e.u(0x1_FFFF)?, e.u(17)?);
    let row = e.iop(BinOp::ShiftRightLogical, k, k17)?;
    let t = e.iop(BinOp::BitwiseAnd, k, low)?;
    let t = e.ufloat(t)?;
    let scale = e.c(1.0 / 131_072.0)?;
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
    // Horner's rule in plain `float`: `eps` is below `2^-25`, so its own rounding errors (about
    // `2^-49`) move the result only where the sum lies that close to a midpoint.
    let mut eps = coefficient(e, COEFFICIENTS - 1)?;
    for j in (0..COEFFICIENTS - 1).rev() {
        let c = coefficient(e, j)?;
        let p = e.mul(eps, t)?;
        eps = e.add(p, c)?;
    }
    let tail = e.add(d, eps)?;
    let r = e.add(r0, tail)?;
    // `r` in `[1/2, 1]`: `2^(127 - n)` of `y`'s exponent field `n` by the field, and `y`'s sign.
    let rb = e.bits(r)?;
    let (exponent, sign) = (e.u(0x7F80_0000)?, e.u(0x8000_0000)?);
    let n = e.iop(BinOp::BitwiseAnd, b, exponent)?;
    let shift = e.iop(BinOp::ISub, one_bits, n)?;
    let v = e.iop(BinOp::IAdd, rb, shift)?;
    let s = e.iop(BinOp::BitwiseAnd, b, sign)?;
    let v = e.iop(BinOp::BitwiseOr, v, s)?;
    // The flushes, by the magnitude's bits: a zero exponent field, and past `2^126`.
    let magnitude = e.u(0x7FFF_FFFF)?;
    let a = e.iop(BinOp::BitwiseAnd, b, magnitude)?;
    let zero = e.u(0)?;
    let tiny = e.iop(BinOp::IEqual, n, zero)?;
    let infinity = e.u(0x7F80_0000)?;
    let inf = e.iop(BinOp::BitwiseOr, s, infinity)?;
    let v = e.sel(tiny, inf, v)?;
    let limit = e.u(0x7E80_0000)?;
    let past = e.iop(BinOp::UGreaterThan, a, limit)?;
    let v = e.sel(past, s, v)?;
    let r = e.float(v)?;
    let nan = e.is_nan(y)?;
    e.sel(nan, y, r)
}

/// Metal's reciprocal of a normal `float` `y` whose reciprocal is normal, as the model computes it
/// ([`TABLE`]) and the routine forms it: `r0 + ((1 - m r0) r0 + eps)` of its significand `m`, `r0 =
/// RN(1 / m)`, the residual exact, `eps` by Horner's rule in plain `float`; then `y`'s exponent
/// negated and its sign.
#[cfg(test)]
pub(crate) fn model(y: f32) -> f32 {
    let b = y.to_bits();
    let k = b & 0x007F_FFFF;
    let m = f32::from_bits(k | 0x3F80_0000);
    let r0 = 1.0 / m;
    let residual = (1.0 - f64::from(m) * f64::from(r0)) as f32;
    let row = TABLE[(k >> 17) as usize].map(f32::from_bits);
    let t = (k & 0x1_FFFF) as f32 / 131_072.0;
    let eps = row.iter().rev().fold(0.0f32, |acc, c| acc * t + c);
    let r = r0 + (residual * r0 + eps);
    let shift = 0x3F80_0000u32.wrapping_sub(b & 0x7F80_0000);
    f32::from_bits(r.to_bits().wrapping_add(shift) | (b & 0x8000_0000))
}
