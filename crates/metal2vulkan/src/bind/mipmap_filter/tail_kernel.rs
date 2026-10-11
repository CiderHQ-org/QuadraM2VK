//! The Apple M3 Max's tail in the kernel (`kernel.rs`; the module docs' "The M3 Max's tail"):
//! each destination texel's four taps from the level's tail tables, each field's four values
//! (the level above's kept values or its codes decoded), mixed along x and then y, encoded, and
//! kept for the next level where it reads them; every rounding exact in integers (`soft.rs`).
//! Each field's rule is fixed when the kernel is built, so it computes that rule's arithmetic
//! alone.

use super::kernel::{
    bin, bink, bits, c32, c64, keep_value, kept_at, param, paramk, place, prec, rec, select,
    texel as texel_words, xr_step, K,
};
use super::layout::{header, record, rule};
use super::soft::{self, Soft};
use crate::lower::arith::extended_range::table as extended_range;
use crate::lower::arith::srgb;
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::Value;
use crate::spirv::ir::{BinOp, ConvOp, GlslOp, MemAccess, UnOp};

/// One axis of a tail level: `(x0, x1, w)`, from the axis's table at word `table` and the
/// level above's `big` texels.
fn axis(
    f: &mut FunctionBuilder<'_>,
    k: &K<'_>,
    table: Value,
    coordinate: Value,
    big: Value,
) -> BResult<[Value; 3]> {
    let i = bin(f, BinOp::IAdd, table, coordinate)?;
    let w = param(f, k, i)?;
    let x0 = bink(f, BinOp::BitwiseAnd, w, 0xffff)?;
    let t = bink(f, BinOp::ShiftRightLogical, w, 16)?;
    let next = bink(f, BinOp::IAdd, x0, 1)?;
    let last = bink(f, BinOp::ISub, big, 1)?;
    let x1 = f.glsl(GlslOp::UMin, &[next, last])?;
    Ok([x0, x1, t])
}

/// The value the M3 Max reads for field `c`'s code in a source texel of `words` (by the field's
/// rule).
fn decode(f: &mut FunctionBuilder<'_>, k: &K<'_>, c: u32, words: &[Value]) -> BResult<Soft> {
    let offset = k.word(c, record::OFFSET);
    let code = bits(f, words, offset, k.word(c, record::MASK))?;
    // A code read as a ratio of its numerator `n` (normalized, sRGB and extended-range codes).
    let ratio = |f: &mut FunctionBuilder<'_>, n: Value| -> BResult<Soft> {
        let den = c32(f, k.word(c, record::DEN).max(1))?;
        let pr = prec(f, k, c)?;
        soft::ratio(f, k, n, den, pr)
    };
    match k.word(c, record::RULE) {
        rule::SRGB8 => {
            let b = bink(f, BinOp::BitwiseAnd, code, 0xff)?;
            let n = srgb::step(f, k.srgb, b)?;
            ratio(f, n)
        }
        rule::XR_COLOUR | rule::XR_ALPHA => {
            let n = bink(f, BinOp::ISub, code, 384)?;
            ratio(f, n)
        }
        rule::XR_SRGB => {
            let n = xr_step(f, k, code)?;
            ratio(f, n)
        }
        rule::FLOAT32 => soft::unpack_f32(f, k, code),
        rule::FLOAT16 => soft::unpack_f16(f, k, code),
        rule::UFLOAT => {
            let mb = k.word(c, record::MB);
            let ef = bink(f, BinOp::ShiftRightLogical, code, mb)?;
            let mf = bink(f, BinOp::BitwiseAnd, code, (1 << mb) - 1)?;
            let positive = soft::falsy(f)?;
            let format = (c32(f, mb)?, c32(f, 31)?, c32(f, mb + 15)?);
            soft::unpack(f, k, positive, ef, mf, format)
        }
        // `RGB9E5Float`'s mantissa and the texel's shared exponent.
        rule::RGB9E5 => {
            let es = bits(f, words, 27, 0x1f)?;
            let positive = soft::falsy(f)?;
            Ok(Soft {
                neg: positive,
                m: f.convert(ConvOp::UConvert, code, k.u64_)?,
                e: bink(f, BinOp::ISub, es, 24)?,
                inf: positive,
                nan: positive,
            })
        }
        // A normalized or 16-bit code.
        _ => {
            let v = match k.word(c, record::EXTEND) {
                0 => code,
                extend => {
                    let v = bink(f, BinOp::ShiftLeftLogical, code, extend)?;
                    bink(f, BinOp::ShiftRightArithmetic, v, extend)?
                }
            };
            let lo = rec(f, k, c, record::IN_LO)?;
            let hi = rec(f, k, c, record::IN_HI)?;
            let n = f.glsl(GlslOp::SClamp, &[v, lo, hi])?;
            ratio(f, n)
        }
    }
}

/// An unsigned float's code of `mb` mantissa bits for `v`: toward zero, the subnormals below
/// `2^-14`, the largest finite above `2^16`, a negative value as 0, its infinity and its NaN.
fn ufloat(f: &mut FunctionBuilder<'_>, k: &K<'_>, v: Soft, mb: u32) -> BResult<Value> {
    let zero = c32(f, 0)?;
    let zero64 = c64(f, k, 0)?;
    let cap = c32(f, 63)?;
    let b = soft::msb64(f, k, v.m)?;
    let x = bin(f, BinOp::IAdd, b, v.e)?;
    let least = c32(f, -14i32 as u32)?;
    let floor = f.glsl(GlslOp::SMax, &[x, least])?;
    let ulp = bink(f, BinOp::ISub, floor, mb)?;
    let drop = bin(f, BinOp::ISub, ulp, v.e)?;
    let dropping = bin(f, BinOp::SGreaterThan, drop, zero)?;
    let dr = f.glsl(GlslOp::UMin, &[drop, cap])?;
    let right = bin(f, BinOp::ShiftRightLogical, v.m, dr)?;
    let minus = bin(f, BinOp::ISub, zero, drop)?;
    let wl = f.glsl(GlslOp::UMin, &[minus, cap])?;
    let left = bin(f, BinOp::ShiftLeftLogical, v.m, wl)?;
    let kept = select(f, dropping, right, left)?;
    let kept = f.convert(ConvOp::UConvert, kept, k.u32_)?;
    let frac = bink(f, BinOp::BitwiseAnd, kept, (1 << mb) - 1)?;
    let biased = bink(f, BinOp::IAdd, x, 15)?;
    let biased = bink(f, BinOp::ShiftLeftLogical, biased, mb)?;
    let normal = bin(f, BinOp::BitwiseOr, biased, frac)?;
    let inf = c32(f, 31 << mb)?;
    let largest = c32(f, (31 << mb) - 1)?;
    let fifteen = c32(f, 15)?;
    let over = bin(f, BinOp::SGreaterThan, x, fifteen)?;
    let under = bin(f, BinOp::SLessThan, x, least)?;
    let code = select(f, under, kept, normal)?;
    let code = select(f, over, largest, code)?;
    let is_zero = bin(f, BinOp::IEqual, v.m, zero64)?;
    let gone = bin(f, BinOp::LogicalOr, is_zero, v.neg)?;
    let code = select(f, gone, zero, code)?;
    let inf_code = select(f, v.neg, zero, inf)?;
    let code = select(f, v.inf, inf_code, code)?;
    // The quiet bit, the mantissa's top.
    let nan = c32(f, (31 << mb) | (1 << mb >> 1))?;
    select(f, v.nan, nan, code)
}

/// The code the M3 Max stores for value `v` of field `c` (by the field's rule; `RGB9E5Float`'s
/// channels 0: its word is made from the three values).
fn encode(f: &mut FunctionBuilder<'_>, k: &K<'_>, c: u32, v: Soft) -> BResult<Value> {
    let zero = c32(f, 0)?;
    let zero64 = c64(f, k, 0)?;
    let id = k.word(c, record::RULE);
    let den = rec(f, k, c, record::DEN)?;
    match id {
        // sRGB bytes: v 4095 (the record's denominator) rounded, clamped, searched.
        rule::SRGB8 => {
            let t = soft::times_rne(f, k, v, den)?;
            let top = c32(f, 4095)?;
            let i = f.glsl(GlslOp::SClamp, &[t, zero, top])?;
            srgb::search(f, k.srgb, i)
        }
        // Extended-range sRGB: |v| 8192 rounded, searched, toward v's sign.
        rule::XR_SRGB => {
            let positive = soft::falsy(f)?;
            let magnitude = Soft { neg: positive, ..v };
            let a = soft::times_rne(f, k, magnitude, den)?;
            let cap = c32(f, 16384)?;
            let a = f.glsl(GlslOp::UMin, &[a, cap])?;
            let n = extended_range::search(f, k.xr, a)?;
            let (up, down) = (c32(f, 639)?, c32(f, 384)?);
            let above = f.glsl(GlslOp::UMin, &[n, up])?;
            let above = bin(f, BinOp::IAdd, down, above)?;
            let below = f.glsl(GlslOp::UMin, &[n, down])?;
            let below = bin(f, BinOp::ISub, down, below)?;
            let nonzero = bin(f, BinOp::INotEqual, v.m, zero64)?;
            let negative = bin(f, BinOp::LogicalAnd, v.neg, nonzero)?;
            select(f, negative, below, above)
        }
        // f32 bits, a value below the least normal flushed to its signed zero.
        rule::FLOAT32 => {
            let nonzero = bin(f, BinOp::INotEqual, v.m, zero64)?;
            let b = soft::msb64(f, k, v.m)?;
            let x = bin(f, BinOp::IAdd, b, v.e)?;
            let least = c32(f, -126i32 as u32)?;
            let small = bin(f, BinOp::SLessThan, x, least)?;
            let tiny = bin(f, BinOp::LogicalAnd, nonzero, small)?;
            let special = bin(f, BinOp::LogicalOr, v.inf, v.nan)?;
            let ordinary = f.unary(UnOp::LogicalNot, special)?;
            let tiny = bin(f, BinOp::LogicalAnd, tiny, ordinary)?;
            let sign = c32(f, 0x8000_0000)?;
            let signed_zero = select(f, v.neg, sign, zero)?;
            let packed = soft::pack_f32(f, k, v)?;
            select(f, tiny, signed_zero, packed)
        }
        rule::FLOAT16 => soft::pack_f16(f, k, v),
        rule::UFLOAT => ufloat(f, k, v, k.word(c, record::MB)),
        rule::RGB9E5 => Ok(zero),
        // The normalized, 16-bit and extended-range codes: v den rounded to nearest even,
        // clamped (an extended-range code about 384).
        _ => {
            let t = soft::times_rne(f, k, v, den)?;
            let t = if matches!(id, rule::XR_COLOUR | rule::XR_ALPHA) {
                bink(f, BinOp::IAdd, t, 384)?
            } else {
                t
            };
            let lo = rec(f, k, c, record::LO)?;
            let hi = rec(f, k, c, record::HI)?;
            f.glsl(GlslOp::SClamp, &[t, lo, hi])
        }
    }
}

/// What a tail field computes from: the four source texels (x0 y0, x1 y0, x0 y1, x1 y1) and
/// their words where the level reads the level above's codes, the weights along x and y, and
/// the destination texel.
struct Taps {
    index: [Value; 4],
    words: Option<[Vec<Value>; 4]>,
    tx: Value,
    ty: Value,
    /// Whether the level above is one texel wide, and one tall: an axis it reads alone.
    one_x: Value,
    one_y: Value,
    i: Value,
}

/// Field `c`'s placed bits, its word, and its value read as at least 0 `(m, e)` for
/// `RGB9E5Float`'s word (zeros for another format).
fn field(
    f: &mut FunctionBuilder<'_>,
    k: &K<'_>,
    c: u32,
    t: &Taps,
) -> BResult<(Value, u32, Value, Value)> {
    let zero = c32(f, 0)?;
    let zero64 = c64(f, k, 0)?;
    let mask = k.word(c, record::MASK);
    let offset = k.word(c, record::OFFSET);
    let (code, m, e) = if k.word(c, record::RULE) == rule::CONSTANT {
        // The constant; the value it keeps is 0, as every value it reads is.
        if k.spec.keep {
            keep_value(f, k, t.i, c, zero)?;
        }
        (c32(f, k.word(c, record::CONSTANT) & mask)?, zero64, zero)
    } else {
        let mut v = Vec::with_capacity(4);
        match &t.words {
            None => {
                for index in t.index {
                    let w = kept_at(f, index, c)?;
                    let p = f.access_chain(k.kept_in, &[zero, w])?;
                    let bits = f.load(p, MemAccess::default())?;
                    v.push(soft::unpack_f32(f, k, bits)?);
                }
            }
            Some(words) => {
                for w in words {
                    v.push(decode(f, k, c, w)?);
                }
            }
        }
        let pr = prec(f, k, c)?;
        // An axis of one texel reads it alone: no mix along it, so an infinity stays one.
        let r0 = soft::mix(f, k, v[0], v[1], t.tx, pr)?;
        let r0 = soft::choose(f, t.one_x, v[0], r0)?;
        let r1 = soft::mix(f, k, v[2], v[3], t.tx, pr)?;
        let r1 = soft::choose(f, t.one_x, v[2], r1)?;
        let value = soft::mix(f, k, r0, r1, t.ty, pr)?;
        let value = soft::choose(f, t.one_y, r0, value)?;
        let code = encode(f, k, c, value)?;
        let code = bink(f, BinOp::BitwiseAnd, code, mask)?;
        if k.spec.keep {
            let kept = soft::pack_f32(f, k, value)?;
            keep_value(f, k, t.i, c, kept)?;
        }
        let (m, e) = if k.spec.rgb9e5 {
            soft::nonnegative(f, k, value)?
        } else {
            (zero64, zero)
        };
        (code, m, e)
    };
    let placed = match offset % 32 {
        0 => code,
        shift => bink(f, BinOp::ShiftLeftLogical, code, shift)?,
    };
    Ok((placed, offset / 32, m, e))
}

/// `RGB9E5Float`'s word from its three channels' values read as at least 0, `(m, e)` each: the
/// shared exponent of the largest (at least -16 before the bias of 16, at most 31), each
/// mantissa floored, at most 511.
fn rgb9e5(f: &mut FunctionBuilder<'_>, k: &K<'_>, v: [(Value, Value); 3]) -> BResult<Value> {
    let zero = c32(f, 0)?;
    let zero64 = c64(f, k, 0)?;
    let sentinel = c32(f, 0xc000_0000)?;
    let mut x = sentinel;
    let mut any = bin(f, BinOp::INotEqual, zero, zero)?;
    for &(m, e) in &v {
        let b = soft::msb64(f, k, m)?;
        let xe = bin(f, BinOp::IAdd, b, e)?;
        let live = bin(f, BinOp::INotEqual, m, zero64)?;
        let xe = select(f, live, xe, sentinel)?;
        x = f.glsl(GlslOp::SMax, &[x, xe])?;
        any = bin(f, BinOp::LogicalOr, any, live)?;
    }
    let floor = c32(f, -16i32 as u32)?;
    let x = f.glsl(GlslOp::SMax, &[x, floor])?;
    let es = bink(f, BinOp::IAdd, x, 16)?;
    let top = c32(f, 31)?;
    let es = f.glsl(GlslOp::SMin, &[es, top])?;
    let mut word = bink(f, BinOp::ShiftLeftLogical, es, 27)?;
    let cap = c32(f, 63)?;
    let most = c64(f, k, 511)?;
    for (c, (m, e)) in v.into_iter().enumerate() {
        let sh = bin(f, BinOp::ISub, e, es)?;
        let sh = bink(f, BinOp::IAdd, sh, 24)?;
        let left = bin(f, BinOp::SGreaterThanEqual, sh, zero)?;
        let l = f.glsl(GlslOp::UMin, &[sh, cap])?;
        let minus = bin(f, BinOp::ISub, zero, sh)?;
        let r = f.glsl(GlslOp::UMin, &[minus, cap])?;
        let up = bin(f, BinOp::ShiftLeftLogical, m, l)?;
        let down = bin(f, BinOp::ShiftRightLogical, m, r)?;
        let mant = select(f, left, up, down)?;
        let mant = f.glsl(GlslOp::UMin, &[mant, most])?;
        let mant = f.convert(ConvOp::UConvert, mant, k.u32_)?;
        let mant = bink(f, BinOp::ShiftLeftLogical, mant, 9 * c as u32)?;
        word = bin(f, BinOp::BitwiseOr, word, mant)?;
    }
    select(f, any, word, zero)
}

/// Destination texel `i`'s words at a tail level, the values it keeps written where the level
/// keeps them, the level above's kept values read where it reads them, else its codes (each
/// source texel's words read once).
pub(super) fn texel(f: &mut FunctionBuilder<'_>, k: &K<'_>, i: Value) -> BResult<[Value; 4]> {
    let sx = paramk(f, k, header::EXTENT)?;
    let sy = paramk(f, k, header::EXTENT + 1)?;
    let bx = paramk(f, k, header::ABOVE)?;
    let by = paramk(f, k, header::ABOVE + 1)?;
    let tx = paramk(f, k, header::TAPS)?;
    let ty = paramk(f, k, header::TAPS + 1)?;
    let x = bin(f, BinOp::UMod, i, sx)?;
    let r = bin(f, BinOp::UDiv, i, sx)?;
    let y = bin(f, BinOp::UMod, r, sy)?;
    let layer = bin(f, BinOp::UDiv, r, sy)?;
    let [x0, x1, wx] = axis(f, k, tx, x, bx)?;
    let [y0, y1, wy] = axis(f, k, ty, y, by)?;
    let row = bin(f, BinOp::IMul, layer, by)?;
    let mut index = [i; 4];
    for (j, (xx, yy)) in [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]
        .into_iter()
        .enumerate()
    {
        let at = bin(f, BinOp::IAdd, row, yy)?;
        let at = bin(f, BinOp::IMul, at, bx)?;
        index[j] = bin(f, BinOp::IAdd, at, xx)?;
    }
    let words = if k.spec.read_kept {
        None
    } else {
        Some([
            texel_words(f, k, index[0])?,
            texel_words(f, k, index[1])?,
            texel_words(f, k, index[2])?,
            texel_words(f, k, index[3])?,
        ])
    };
    let taps = Taps {
        index,
        words,
        tx: wx,
        ty: wy,
        one_x: bink(f, BinOp::IEqual, bx, 1)?,
        one_y: bink(f, BinOp::IEqual, by, 1)?,
        i,
    };
    let zero = c32(f, 0)?;
    let mut words = [zero; 4];
    let zero64 = c64(f, k, 0)?;
    let mut channels = [(zero64, zero); 3];
    for c in 0..k.fields() {
        let (placed, word, m, e) = field(f, k, c, &taps)?;
        place(f, &mut words, placed, word)?;
        if let Some(ch) = channels.get_mut(c as usize) {
            *ch = (m, e);
        }
    }
    if k.spec.rgb9e5 {
        words[0] = rgb9e5(f, k, channels)?;
    }
    Ok(words)
}
