//! Metal's reduction of a `float` to quarter turns, which its precise `sin`, `cos` and `tan` share:
//! `x 2/π = k + r` with `k` an integer and `|r| <= 1/2` (about), in Metal's own arithmetic, so that
//! the reduced argument is Metal's word for word.
//!
//! Below `2^21` in magnitude, `k` is `x 2/π` rounded to an integer by adding and removing `2^23`
//! with `x`'s sign in a fused step, and `r` is three fused steps of `x` by `2/π` in three parts
//! ([`QUARTER`]). From `2^21` up, `x 2^-126` is multiplied by row `(|x| - 2^21) >> 24` of
//! [`QUARTER_TABLE`] (in bits; the bits of `2/π` that matter for that pair of binades, in three
//! `float`s `h`, `m`, `l`): `k0` is the first product rounded as above (itself from `2^23`),
//! `r0 = fma(h, x', -k0)`, `k1` is `fma(m, x', r0)` rounded half away from zero, `k = k0 + k1`,
//! and `r = fma(l, x', fma(m, x', r0 - k1))`, to which `tan` adds the rounding error of
//! `r0 - k1` ([`Error::Compensated`]) and `sin` and `cos` do not ([`Error::Dropped`]).
//!
//! Metal multiplies `x 2^-126` by the row; here `x 2^-102` by the row times `2^-24`, the same
//! products exactly (every factor stays normal), so that no factor of an emulated fused step
//! is too large to split. Both reductions run on every lane, each on an operand in its own range (0
//! from `2^21`, `2^21` below it or for an infinity or a NaN), so that no integer conversion of a
//! lane selected away leaves its range.

use crate::lower::arith::libm::E;
use crate::spirv::builder::ModuleBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, Value};
use crate::spirv::ir::{BinOp, MemAccess, UnOp};

/// `2/π` in three parts (`f32` bits, the largest first), the reduction's below `2^21`.
pub(crate) const QUARTER: [u32; 3] = [0x3F22_F983, 0x32DC_9C88, 0x25A9_4FE1];

/// Metal's reduction from `2^21` up: per pair of binades (row `(|x| bits - 0x4a000000) >> 24`),
/// the bits of `2/π` that matter there, as three `float`s that multiply `x 2^-126` (`f32` bits).
/// The last row is never read.
pub(crate) const QUARTER_TABLE: [[u32; 3]; 55] = [
    [0x7E22_F983, 0x71DC_9C88, 0x64A9_4FE1],
    [0x7E22_F983, 0x71DC_9C88, 0x64A9_4FE1],
    [0xFDBA_0CF9, 0xF08D_8DDF, 0xE42D_603E],
    [0xFCE8_33E5, 0x7064_E441, 0x63A5_3F85],
    [0x7B3E_60DC, 0xEED8_DDF5, 0xE256_03D9],
    [0xFA83_3E49, 0x6D9C_882A, 0x6127_F09D],
    [0xF7CF_9236, 0xEADD_F56B, 0xDB76_2A0C],
    [0xF7CF_9236, 0xEADD_F56B, 0xDB76_2A0C],
    [0xF7CF_9236, 0xEADD_F56B, 0xDB76_2A0C],
    [0x76C1_B727, 0x6988_2A54, 0xDB76_2A0C],
    [0xF5F9_2363, 0xE96F_AB58, 0xDB76_2A0C],
    [0x735B_9391, 0x64A9_4FE1, 0x57EA_FA3F],
    [0x735B_9391, 0x64A9_4FE1, 0x57EA_FA3F],
    [0xF211_B1BC, 0x64A9_4FE1, 0x57EA_FA3F],
    [0x71DC_9C88, 0x64A9_4FE1, 0x57EA_FA3F],
    [0xF08D_8DDF, 0xE42D_603E, 0x57EA_FA3F],
    [0xEED8_DDF5, 0xE256_03D9, 0x55AF_A3EA],
    [0xEED8_DDF5, 0xE256_03D9, 0x55AF_A3EA],
    [0x6D9C_882A, 0x6127_F09D, 0x54BE_8FAA],
    [0x6C64_4153, 0xDFC0_7B15, 0xD138_2B2D],
    [0xEADD_F56B, 0xDB76_2A0C, 0x4EFA_9A6F],
    [0xEADD_F56B, 0xDB76_2A0C, 0x4EFA_9A6F],
    [0x6988_2A54, 0xDB76_2A0C, 0x4EFA_9A6F],
    [0x6782_A540, 0xDB76_2A0C, 0x4EFA_9A6F],
    [0x6782_A540, 0xDB76_2A0C, 0x4EFA_9A6F],
    [0x64A9_4FE1, 0x57EA_FA3F, 0xCB32_C890],
    [0x64A9_4FE1, 0x57EA_FA3F, 0xCB32_C890],
    [0x64A9_4FE1, 0x57EA_FA3F, 0xCB32_C890],
    [0x63A5_3F85, 0xD628_2E0B, 0x4953_7703],
    [0x6294_FE14, 0xD628_2E0B, 0x4953_7703],
    [0x6127_F09D, 0x54BE_8FAA, 0xC832_23F2],
    [0xE0B0_1EC5, 0xD402_E0AD, 0x479B_B81B],
    [0xDFC0_7B15, 0xD138_2B2D, 0x44EE_06DB],
    [0x5EFE_13AC, 0xD138_2B2D, 0x44EE_06DB],
    [0xDB76_2A0C, 0x4EFA_9A6F, 0xC17C_9276],
    [0xDB76_2A0C, 0x4EFA_9A6F, 0xC17C_9276],
    [0xDB76_2A0C, 0x4EFA_9A6F, 0xC17C_9276],
    [0x591D_5F48, 0xCC2C_B224, 0x3E5B_6296],
    [0x591D_5F48, 0xCC2C_B224, 0x3E5B_6296],
    [0xD8C5_4170, 0xCC2C_B224, 0x3E5B_6296],
    [0x57EA_FA3F, 0xCB32_C890, 0x3E5B_6296],
    [0xD628_2E0B, 0x4953_7703, 0x3CDB_14AD],
    [0x55AF_A3EA, 0x4953_7703, 0x3CDB_14AD],
    [0x54BE_8FAA, 0xC832_23F2, 0xBB93_AD4D],
    [0x53FA_3EA7, 0xC748_8FC9, 0xBA1D_6A67],
    [0xD138_2B2D, 0x44EE_06DB, 0x3725_664F],
    [0xD138_2B2D, 0x44EE_06DB, 0x3725_664F],
    [0x508F_A9A7, 0xC30F_C927, 0xB6B5_3362],
    [0x4EFA_9A6F, 0xC17C_9276, 0x34AC_C9E2],
    [0x4EFA_9A6F, 0xC17C_9276, 0x34AC_C9E2],
    [0xCC2C_B224, 0x3E5B_6296, 0xB1CD_8779],
    [0xCC2C_B224, 0x3E5B_6296, 0xB1CD_8779],
    [0x4BA6_9BB8, 0x3E5B_6296, 0xB1CD_8779],
    [0x4A9A_6EE0, 0x3E5B_6296, 0xB1CD_8779],
    [0x0000_0000, 0x0000_0000, 0x0000_0000],
];

/// An upper estimate of the SPIR-V words of [`QUARTER_TABLE`] in a module: its 165 constants and
/// a zero, the composite (168 words with its three zeros, `declare_words`), and the array,
/// pointer, and variable.
pub(crate) const TABLE_WORDS: u32 = 166 * 4 + 171 + 24;

/// Declare [`QUARTER_TABLE`] in the module.
pub(crate) fn declare(mb: &mut ModuleBuilder) -> BResult<GlobalId> {
    super::declare_words(mb, QUARTER_TABLE.as_flattened())
}

/// The `float` bits from which the reduction reads [`QUARTER_TABLE`] (`2^21`).
const TABLE_FROM: u32 = 0x4A00_0000;

/// Whether the reduction from `2^21` up adds the rounding error of `r0 - k1` to `r`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Error {
    /// `tan`'s.
    Compensated,
    /// `sin`'s and `cos`'s.
    Dropped,
}

/// `c v` rounded to an integer-valued `float` (half to even): `fma(c, v, b) - b` of `b = 2^23`
/// with the sign of `like`, an approximation of `c v`.
fn rounded(e: &mut E<'_, '_>, c: Value, v: Value, like: Value) -> BResult<Value> {
    let big = e.c(8_388_608.0)?;
    let b = e.copysign(big, like)?;
    let s = e.fused(c, v, b)?;
    e.sub(s, b)
}

/// The `u32` of the integer-valued `float` `k` modulo `2^32` (its magnitude below `2^32`).
fn integer(e: &mut E<'_, '_>, k: Value) -> BResult<Value> {
    let ak = e.abs(k)?;
    let n = e.utrunc(ak)?;
    let zero = e.u(0)?;
    let minus_n = e.iop(BinOp::ISub, zero, n)?;
    let zero_f = e.c(0.0)?;
    let negative = e.lt(k, zero_f)?;
    e.sel(negative, minus_n, n)
}

/// Metal's reduction of the `float` `x` (the module's doc): `(r, k)`, `r` the reduced argument
/// in quarter turns and `k` the quarter turns as a `u32` modulo `2^32`. For an infinity or a NaN
/// both are some finite values, which the callers select away.
pub(crate) fn turns(e: &mut E<'_, '_>, x: Value, error: Error) -> BResult<(Value, Value)> {
    let ax = e.abs(x)?;
    let bits = e.bits(ax)?;
    let from = e.u(TABLE_FROM)?;
    let below = e.iop(BinOp::ULessThan, bits, from)?;
    let inf = e.u(0x7F80_0000)?;
    let finite = e.iop(BinOp::ULessThan, bits, inf)?;
    let zero = e.c(0.0)?;
    let start = e.c(2_097_152.0)?;
    let xn = e.sel(below, x, zero)?;
    let xf = e.sel(finite, x, start)?;
    let xf = e.sel(below, start, xf)?;
    // Below 2^21: three fused steps.
    let [c0, c1, c2] = QUARTER.map(f32::from_bits);
    let (c0, c1, c2) = (e.c(c0)?, e.c(c1)?, e.c(c2)?);
    let k = rounded(e, c0, xn, xn)?;
    let minus_k = e.fb.unary(UnOp::FNegate, k)?;
    let r = e.fused(c0, xn, minus_k)?;
    let r = e.fused(c1, xn, r)?;
    let near = e.fused(c2, xn, r)?;
    let near_k = integer(e, k)?;
    // From 2^21: the table's row for `x`'s binades.
    let axf = e.abs(xf)?;
    let fbits = e.bits(axf)?;
    let offset = e.iop(BinOp::ISub, fbits, from)?;
    let k24 = e.u(24)?;
    let row = e.iop(BinOp::ShiftRightLogical, offset, k24)?;
    let three = e.u(3)?;
    let first = e.iop(BinOp::IMul, row, three)?;
    let table = e.quarter_table()?;
    let base = e.fb.module().global_value(table)?;
    let part = |e: &mut E<'_, '_>, j: u32| -> BResult<Value> {
        let j = e.u(j)?;
        let i = e.iop(BinOp::IAdd, first, j)?;
        let p = e.fb.access_chain(base, &[i])?;
        let w = e.fb.load(p, MemAccess::default())?;
        e.float(w)
    };
    let down = e.c(1.0 / 16_777_216.0)?;
    let (h, m, l) = (part(e, 0)?, part(e, 1)?, part(e, 2)?);
    let (h, m, l) = (e.mul(h, down)?, e.mul(m, down)?, e.mul(l, down)?);
    let scale = e.c(f32::from_bits(0x0C80_0000))?;
    let xs = e.mul(xf, scale)?;
    let p = e.mul(xs, h)?;
    let k0 = rounded(e, h, xs, p)?;
    let ap = e.abs(p)?;
    let big = e.c(8_388_608.0)?;
    let integral = e.ge(ap, big)?;
    let k0 = e.sel(integral, p, k0)?;
    let minus_k0 = e.fb.unary(UnOp::FNegate, k0)?;
    let r0 = e.fused(h, xs, minus_k0)?;
    let r1 = e.fused(m, xs, r0)?;
    let half = e.c(0.5)?;
    let half = e.copysign(half, r1)?;
    let k1 = e.add(r1, half)?;
    let k1 = e.fb.glsl(crate::spirv::ir::GlslOp::Trunc, &[k1])?;
    let d = e.sub(r0, k1)?;
    let r2 = e.fused(m, xs, d)?;
    let r3 = e.fused(l, xs, r2)?;
    let far = match error {
        Error::Dropped => r3,
        Error::Compensated => {
            // `r0 - k1`'s rounding error.
            let e1 = e.sub(d, r0)?;
            let e2 = e.sub(d, e1)?;
            let e3 = e.sub(r0, e2)?;
            let minus_k1 = e.fb.unary(UnOp::FNegate, k1)?;
            let e4 = e.sub(minus_k1, e1)?;
            let err = e.add(e4, e3)?;
            e.add(r3, err)?
        }
    };
    let (n0, n1) = (integer(e, k0)?, integer(e, k1)?);
    let far_k = e.iop(BinOp::IAdd, n0, n1)?;
    let r = e.sel(below, near, far)?;
    let k = e.sel(below, near_k, far_k)?;
    Ok((r, k))
}
