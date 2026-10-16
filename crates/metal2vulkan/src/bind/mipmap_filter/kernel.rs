//! The filter kernel ([`super::mipmap_filter_module`]) of one [`super::MipmapFilterKernel`]: one
//! invocation per destination texel, its taps from the level's tap tables (eight in 3D, four
//! where the level above is one texel deep), each source texel's words read once, each field
//! computed by its record's rule in integer arithmetic (the module docs); or, at a tail level,
//! the Apple M3 Max's arithmetic (`tail_kernel.rs`). The kernel is built with its format's
//! records and its level's step ([`Spec`]), so that the module holds only what its levels
//! compute: no branch on a format's rule or a level's step is left to the GPU.

use super::fetch::{self, Component, Recover};
use super::layout::{decode, encode, header, mode, record, rule, KEPT_WORDS, RECORD_WORDS};
use super::soft::{self, Prec, Soft};
use super::{DESTINATION, KEPT_IN, KEPT_OUT, PARAMS, SOURCE, SOURCE_IMAGE};
use crate::lower::arith::extended_range::table as extended_range;
use crate::lower::arith::srgb;
use crate::metal::format::extended_srgb;
use crate::pipeline::sym::ResourceSlot;
use crate::spirv::builder::constructs::{ArmEnd, ArmFn};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, TypeId, Value};
use crate::spirv::ir::{
    natural_layout, AtomicOp, BinOp, ConstBits, Constant, ConvOp, Dim, ExecMode, ExecModel, GlslOp,
    ImageOperands, ImageType, ImageUsage, IoDir, LayoutRule, MemAccess, Scope, Semantics,
    StorageClass, Type, TypedModule, UnOp,
};

/// Invocations of one workgroup, along x.
pub(super) const LOCAL_SIZE: u32 = 64;

/// What a kernel is built with (`MipmapFilterKernel::spec`): the format's texel bytes, its
/// fields' records (`layout::record`, their tie tables where the level's words hold them),
/// whether its word is `RGB9E5Float`'s; and its level's step: the M3 Max's tail (`tail`) or the
/// GPU's filter, the fields the M3 Max carries through a half converted (`half`, a sampler
/// level), the values the next level reads kept (`keep`), the level above's kept values read
/// instead of its codes (`read_kept`, a tail level), the depth taps taken (`depth`, a
/// sampler level whose level above is more than one texel deep), and the first level fetched
/// from its sampled view instead of read from the scratch buffer (`fetch`, the first later
/// level of a chain that reads its first level in place: `fetch.rs`).
pub(super) struct Spec {
    pub(super) bytes: u32,
    pub(super) records: Vec<[u32; RECORD_WORDS as usize]>,
    pub(super) rgb9e5: bool,
    pub(super) tail: bool,
    pub(super) half: bool,
    pub(super) keep: bool,
    pub(super) read_kept: bool,
    pub(super) depth: bool,
    pub(super) fetch: Option<Fetch>,
}

/// The first level's sampled view a kernel fetches its taps from (`fetch.rs`): the components a
/// fetch returns, and whether the view is a 3D one (a 3D texture's) or a 2D array of every layer.
#[derive(Clone, Copy)]
pub(super) struct Fetch {
    pub(super) components: [Option<Component>; 4],
    pub(super) volume: bool,
}

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

/// What every part of the kernel takes: the types, the buffers, the tables, and what the kernel
/// is built with.
#[derive(Clone, Copy)]
pub(super) struct K<'s> {
    pub(super) u32_: TypeId,
    pub(super) u64_: TypeId,
    /// The level above: its words in the scratch buffer, or, where the spec fetches
    /// ([`Spec::fetch`]), the first level's sampled view, loaded ([`texel`] reads only the
    /// first).
    pub(super) source: Value,
    pub(super) destination: Value,
    pub(super) params: Value,
    pub(super) kept_in: Value,
    pub(super) kept_out: Value,
    pub(super) srgb: GlobalId,
    pub(super) xr: GlobalId,
    pub(super) spec: &'s Spec,
}

impl K<'_> {
    /// Word `w` of field `c`'s record.
    pub(super) fn word(&self, c: u32, w: u32) -> u32 {
        self.spec.records[c as usize][w as usize]
    }

    /// The fields of the texel.
    pub(super) fn fields(&self) -> u32 {
        self.spec.records.len() as u32
    }
}

pub(super) fn c32(f: &mut FunctionBuilder<'_>, v: u32) -> BResult<Value> {
    let c = f.u32_const(v)?;
    f.module().const_value(c)
}

pub(super) fn c64(f: &mut FunctionBuilder<'_>, k: &K<'_>, v: u64) -> BResult<Value> {
    let c = f.module().constant(Constant::Scalar {
        ty: k.u64_,
        bits: ConstBits(v),
    })?;
    f.module().const_value(c)
}

pub(super) fn bin(f: &mut FunctionBuilder<'_>, op: BinOp, a: Value, b: Value) -> BResult<Value> {
    f.binary(op, a, b)
}

/// `a op v` for a `u32` constant `v`.
pub(super) fn bink(f: &mut FunctionBuilder<'_>, op: BinOp, a: Value, v: u32) -> BResult<Value> {
    let c = c32(f, v)?;
    f.binary(op, a, c)
}

/// Word `i` (a value) of the level's params.
pub(super) fn param(f: &mut FunctionBuilder<'_>, k: &K<'_>, i: Value) -> BResult<Value> {
    let zero = c32(f, 0)?;
    let p = f.access_chain(k.params, &[zero, i])?;
    f.load(p, MemAccess::default())
}

/// Word `i` (a constant) of the level's params.
pub(super) fn paramk(f: &mut FunctionBuilder<'_>, k: &K<'_>, i: u32) -> BResult<Value> {
    let i = c32(f, i)?;
    param(f, k, i)
}

/// Source texel `index`'s words, read once from the scratch buffer: its bytes from bit 0 of the
/// first (a texel narrower than a word shifted down from the word that holds all of it).
pub(super) fn texel(f: &mut FunctionBuilder<'_>, k: &K<'_>, index: Value) -> BResult<Vec<Value>> {
    let zero = c32(f, 0)?;
    let bytes = k.spec.bytes;
    if bytes < 4 {
        let byte = bink(f, BinOp::IMul, index, bytes)?;
        let word = bink(f, BinOp::ShiftRightLogical, byte, 2)?;
        let lane = bink(f, BinOp::BitwiseAnd, byte, 3)?;
        let lane = bink(f, BinOp::ShiftLeftLogical, lane, 3)?;
        let p = f.access_chain(k.source, &[zero, word])?;
        let w = f.load(p, MemAccess::default())?;
        return Ok(vec![bin(f, BinOp::ShiftRightLogical, w, lane)?]);
    }
    let per = bytes / 4;
    let first = bink(f, BinOp::IMul, index, per)?;
    let mut words = Vec::with_capacity(per as usize);
    for j in 0..per {
        let at = if j == 0 {
            first
        } else {
            bink(f, BinOp::IAdd, first, j)?
        };
        let p = f.access_chain(k.source, &[zero, at])?;
        words.push(f.load(p, MemAccess::default())?);
    }
    Ok(words)
}

/// The field at bit `offset` of a texel's `words` ([`texel`]), masked by `mask`: a field never
/// crosses a 32-bit word.
pub(super) fn bits(
    f: &mut FunctionBuilder<'_>,
    words: &[Value],
    offset: u32,
    mask: u32,
) -> BResult<Value> {
    let w = words[(offset / 32) as usize];
    let w = match offset % 32 {
        0 => w,
        shift => bink(f, BinOp::ShiftRightLogical, w, shift)?,
    };
    bink(f, BinOp::BitwiseAnd, w, mask)
}

pub(super) fn select(f: &mut FunctionBuilder<'_>, c: Value, t: Value, e: Value) -> BResult<Value> {
    f.select(c, t, e)
}

/// Field `c`'s record word `w`, a constant of the module.
pub(super) fn rec(f: &mut FunctionBuilder<'_>, k: &K<'_>, c: u32, w: u32) -> BResult<Value> {
    c32(f, k.word(c, w))
}

/// The precision of field `c`'s M3 Max arithmetic, from its record.
pub(super) fn prec(f: &mut FunctionBuilder<'_>, k: &K<'_>, c: u32) -> BResult<Prec> {
    Ok(Prec {
        bits: rec(f, k, c, record::BITS)?,
        least: rec(f, k, c, record::LEAST)?,
    })
}

/// The signed linear step of extended-range code `code`, about code 384, in `2^-13`
/// (`lower::arith::extended_range::table`).
pub(super) fn xr_step(f: &mut FunctionBuilder<'_>, k: &K<'_>, code: Value) -> BResult<Value> {
    let base = f.module().global_value(k.xr)?;
    let o = bink(f, BinOp::ISub, code, 384)?;
    let a = f.glsl(GlslOp::SAbs, &[o])?;
    let cap = c32(f, 639)?;
    let a = f.glsl(GlslOp::UMin, &[a, cap])?;
    let p = f.access_chain(base, &[a])?;
    let w = f.load(p, MemAccess::default())?;
    let d = bink(f, BinOp::BitwiseAnd, w, 0xffff)?;
    let zero = c32(f, 0)?;
    let minus = bin(f, BinOp::ISub, zero, d)?;
    let below = bin(f, BinOp::SLessThan, o, zero)?;
    select(f, below, minus, d)
}

/// `a` (a `u32`) rounded half away from zero to 11 significant bits.
fn half_away_11(f: &mut FunctionBuilder<'_>, a: Value) -> BResult<Value> {
    let one = c32(f, 1)?;
    let b = f.glsl(GlslOp::FindUMsb, &[a])?;
    let eleven = c32(f, 11)?;
    let wide = bin(f, BinOp::SGreaterThanEqual, b, eleven)?;
    let drop = bink(f, BinOp::ISub, b, 10)?;
    let most = c32(f, 31)?;
    let drop = f.glsl(GlslOp::UMin, &[drop, most])?;
    let drop = f.glsl(GlslOp::UMax, &[drop, one])?;
    let d1 = bin(f, BinOp::ISub, drop, one)?;
    let r = bin(f, BinOp::ShiftRightLogical, a, d1)?;
    let r = bin(f, BinOp::IAdd, r, one)?;
    let r = bin(f, BinOp::ShiftRightLogical, r, one)?;
    let r = bin(f, BinOp::ShiftLeftLogical, r, drop)?;
    select(f, wide, r, a)
}

/// The `f32` bits of half-float bits.
pub(super) fn half_to_f32(f: &mut FunctionBuilder<'_>, k: &K<'_>, h: Value) -> BResult<Value> {
    let v = soft::unpack_f16(f, k, h)?;
    soft::pack_f32(f, k, v)
}

/// A tap of a sampler level: its source texel's words ([`texel`]), its weight `n` (of `2^24`),
/// and at a 2D level its weight along x and y, `nxy` (of `2^16`: `n = 256 nxy`, the one depth
/// tap weighing 256).
struct Tap {
    words: Vec<Value>,
    n: Value,
    nxy: Option<Value>,
}

/// How the sum of an integer field of record `r` fits 32 bits at a 2D level, where it is exact
/// in 32 bits: `Some(true)`
/// in `i32`s, `Some(false)` in `u32`s (an unsigned field's terms floored one by one); `None`
/// where it needs 64.
///
/// A 2D level's taps weigh `n = 256 nxy`, each `nxy = wx wy` at most `2^16` and the four summing
/// to `2^16`; so with `m` the largest magnitude of the field's decoded values, `|sum nxy v|` and
/// each `|nxy v|` are at most `m 2^16`. A term floored by `PER_TERM = p`, `(n v) >> p`, is
/// `(nxy v) >> (p - 8)` where `p >= 8`, and `nxy v` itself where `p <= 8` (a multiple of `2^8`
/// of `n v`'s units already), its sum's rounding then `(S + ADD / 2^(8 - p)) >> (SHIFT - 8 + p)`
/// where `2^(8 - p)` divides `ADD` and `SHIFT >= 8 - p`. In `i32` where `m 2^16` and the added
/// constant fit one, in `u32` (shifted logically) where the values are at least 0 and a floored
/// term fits one.
pub(super) fn narrow(r: &[u32; RECORD_WORDS as usize]) -> Option<bool> {
    let word = |w: u32| r[w as usize];
    let (least, most) = match word(record::DECODE) {
        decode::SRGB => (0, i64::from(*srgb::SRGB_DECODE_STEPS.iter().max()?)),
        decode::XR_SRGB => {
            let m = i64::from(*extended_srgb::SRGB_DECODE_STEPS.iter().max()?);
            (-m, m)
        }
        _ => (
            i64::from(word(record::IN_LO) as i32),
            i64::from(word(record::IN_HI) as i32),
        ),
    };
    let widen = i64::from(word(record::WIDEN).max(1));
    let reach = (least.abs().max(most.abs()) * widen) << 16;
    let (add, _) = narrow_rounding(r)?;
    if reach + i64::from(add) <= i64::from(i32::MAX) {
        Some(true)
    } else {
        (word(record::PER_TERM) > 8 && least >= 0 && reach <= i64::from(u32::MAX)).then_some(false)
    }
}

/// A 2D level's sum rounding of terms `nxy v` floored by `PER_TERM - 8` (`narrow`): `(ADD,
/// SHIFT)` in those terms' units, `None` where they do not divide.
fn narrow_rounding(r: &[u32; RECORD_WORDS as usize]) -> Option<(u32, u32)> {
    let word = |w: u32| r[w as usize];
    let (add, shift, floor) = (
        word(record::ADD),
        word(record::SHIFT),
        word(record::PER_TERM),
    );
    if floor >= 8 {
        return Some((add, shift));
    }
    let up = 8 - floor;
    (add % (1 << up) == 0 && shift >= up).then(|| (add >> up, shift - up))
}

/// The code of grid sum `q` of field `c` (the `GRID` encode): its integer part, one more above
/// a half, and at an exact half the bit of the field's tie table. A widened field's (`m =
/// record::WIDEN` above 1, an unsigned field's, `q` at least 0) is the code nearest `q / (m
/// 2^k)`, its own code's tie table at a half.
fn grid(f: &mut FunctionBuilder<'_>, k: &K<'_>, c: u32, q: Value) -> BResult<Value> {
    let widen = k.word(c, record::WIDEN).max(1);
    let (code, part, half) = if widen == 1 {
        let kk = rec(f, k, c, record::K)?;
        let fraction = rec(f, k, c, record::FRACTION)?;
        let code = bin(f, BinOp::ShiftRightArithmetic, q, kk)?;
        let part = bin(f, BinOp::BitwiseAnd, q, fraction)?;
        (code, part, rec(f, k, c, record::HALF)?)
    } else {
        let unit = widen << k.word(c, record::K);
        let code = bink(f, BinOp::UDiv, q, unit)?;
        let part = bink(f, BinOp::UMod, q, unit)?;
        (code, part, c32(f, unit / 2)?)
    };
    let above = bin(f, BinOp::UGreaterThan, part, half)?;
    let tie = bin(f, BinOp::IEqual, part, half)?;
    let base = rec(f, k, c, record::TIE_BASE)?;
    let table = rec(f, k, c, record::TIE_AT)?;
    let j = bin(f, BinOp::ISub, code, base)?;
    let zero = c32(f, 0)?;
    let last = c32(f, k.word(c, record::TIE_LEN).wrapping_sub(1))?;
    let j = f.glsl(GlslOp::SClamp, &[j, zero, last])?;
    let word = bink(f, BinOp::ShiftRightLogical, j, 5)?;
    let word = bin(f, BinOp::IAdd, table, word)?;
    let word = param(f, k, word)?;
    let bit = bink(f, BinOp::BitwiseAnd, j, 31)?;
    let bit = bin(f, BinOp::ShiftRightLogical, word, bit)?;
    let bit = bink(f, BinOp::BitwiseAnd, bit, 1)?;
    let bit = bin(f, BinOp::INotEqual, bit, zero)?;
    let at_tie = bin(f, BinOp::LogicalAnd, tie, bit)?;
    let up = bin(f, BinOp::LogicalOr, above, at_tie)?;
    let one = c32(f, 1)?;
    let step = select(f, up, one, zero)?;
    let code = bin(f, BinOp::IAdd, code, step)?;
    let lo = rec(f, k, c, record::LO)?;
    let hi = rec(f, k, c, record::HI)?;
    f.glsl(GlslOp::SClamp, &[code, lo, hi])
}

/// The extended-range sRGB code of sum `q` whose magnitude, in steps of `2^-13`, is `a`: the code
/// whose encode range holds `a`, toward `q`'s sign about code 384.
fn xr_code(f: &mut FunctionBuilder<'_>, k: &K<'_>, a: Value, q: Value) -> BResult<Value> {
    let cap = c32(f, 16384)?;
    let a = f.glsl(GlslOp::UMin, &[a, cap])?;
    let m = extended_range::search(f, k.xr, a)?;
    let up = c32(f, 639)?;
    let down = c32(f, 384)?;
    let above = f.glsl(GlslOp::UMin, &[m, up])?;
    let above = bin(f, BinOp::IAdd, down, above)?;
    let below = f.glsl(GlslOp::UMin, &[m, down])?;
    let below = bin(f, BinOp::ISub, down, below)?;
    let zero = c32(f, 0)?;
    let positive = bin(f, BinOp::SGreaterThanEqual, q, zero)?;
    select(f, positive, above, below)
}

/// Integer field `c` (the `INTEGER` mode): the codes decoded, summed exactly or term by term,
/// rounded, and encoded, through a half where the level converts the fields the M3 Max carries
/// through one; the field's bits, and the `f32` bits of the value the level keeps where it
/// keeps them.
fn integer(
    f: &mut FunctionBuilder<'_>,
    k: &K<'_>,
    c: u32,
    taps: &[Tap],
) -> BResult<(Value, Option<Value>)> {
    let offset = k.word(c, record::OFFSET);
    let mask = k.word(c, record::MASK);
    let id = k.word(c, record::RULE);
    // The decoded codes, as `i32` bits.
    let mut values = Vec::with_capacity(taps.len());
    for tap in taps {
        let b = bits(f, &tap.words, offset, mask)?;
        values.push(match k.word(c, record::DECODE) {
            decode::SRGB => srgb::step(f, k.srgb, b)?,
            decode::XR_SRGB => xr_step(f, k, b)?,
            _ => {
                let v = match k.word(c, record::EXTEND) {
                    0 => b,
                    extend => {
                        let v = bink(f, BinOp::ShiftLeftLogical, b, extend)?;
                        bink(f, BinOp::ShiftRightArithmetic, v, extend)?
                    }
                };
                let lo = rec(f, k, c, record::IN_LO)?;
                let hi = rec(f, k, c, record::IN_HI)?;
                let v = f.glsl(GlslOp::SClamp, &[v, lo, hi])?;
                // A widened field's code as its wide code.
                match k.word(c, record::WIDEN) {
                    0 | 1 => v,
                    widen => bink(f, BinOp::IMul, v, widen)?,
                }
            }
        });
    }
    let floor = k.word(c, record::PER_TERM);
    let (add, shift) = (k.word(c, record::ADD), k.word(c, record::SHIFT));
    let record = &k.spec.records[c as usize];
    let flat: Option<Vec<Value>> = taps.iter().map(|tap| tap.nxy).collect();
    let q = match (flat, narrow(record), narrow_rounding(record)) {
        // A 2D level's sum in 32 bits (`narrow`).
        (Some(nxy), Some(signed), Some((add, shift))) => {
            let mut sum = c32(f, 0)?;
            for (v, n) in values.into_iter().zip(nxy) {
                let p = bin(f, BinOp::IMul, v, n)?;
                let t = match (floor.saturating_sub(8), signed) {
                    (0, _) => p,
                    (s, true) => bink(f, BinOp::ShiftRightArithmetic, p, s)?,
                    (s, false) => bink(f, BinOp::ShiftRightLogical, p, s)?,
                };
                sum = bin(f, BinOp::IAdd, sum, t)?;
            }
            let q = bink(f, BinOp::IAdd, sum, add)?;
            bink(f, BinOp::ShiftRightArithmetic, q, shift)?
        }
        _ => {
            let by = c32(f, floor)?;
            let mut sum = c64(f, k, 0)?;
            for (v, tap) in values.into_iter().zip(taps) {
                let v = f.convert(ConvOp::SConvert, v, k.u64_)?;
                let n = f.convert(ConvOp::UConvert, tap.n, k.u64_)?;
                let p = bin(f, BinOp::IMul, v, n)?;
                let t = if floor != 0 {
                    bin(f, BinOp::ShiftRightArithmetic, p, by)?
                } else {
                    p
                };
                sum = bin(f, BinOp::IAdd, sum, t)?;
            }
            let add = c64(f, k, u64::from(add))?;
            let shift = c32(f, shift)?;
            let q = bin(f, BinOp::IAdd, sum, add)?;
            let q = bin(f, BinOp::ShiftRightArithmetic, q, shift)?;
            f.convert(ConvOp::UConvert, q, k.u32_)?
        }
    };
    let carried = k.word(c, record::CARRIED) != 0 && k.spec.half;
    // An extended-range sRGB sum's magnitude, which the M3 Max rounds to a half first.
    let xr_i = if id == rule::XR_SRGB {
        let magnitude = f.glsl(GlslOp::SAbs, &[q])?;
        Some(if carried {
            half_away_11(f, magnitude)?
        } else {
            magnitude
        })
    } else {
        None
    };
    let zero = c32(f, 0)?;
    let lo = rec(f, k, c, record::LO)?;
    let hi = rec(f, k, c, record::HI)?;
    let v = match (k.word(c, record::ENCODE), xr_i) {
        (encode::SRGB, _) => {
            let top = c32(f, 4095)?;
            let i = f.glsl(GlslOp::SClamp, &[q, zero, top])?;
            srgb::search(f, k.srgb, i)?
        }
        (encode::CLAMP, _) => f.glsl(GlslOp::SClamp, &[q, lo, hi])?,
        (encode::XR_SRGB, Some(a)) => xr_code(f, k, a, q)?,
        _ => grid(f, k, c, q)?,
    };
    // The value the level keeps: `(n - KEPT_BIAS) / KEPT_UNIT` of the grid sum, the sRGB step,
    // or the 16-bit code; the M3 Max's code of it where it carries the field through a half.
    let use_half = carried && id != rule::XR_SRGB;
    if !use_half && !k.spec.keep {
        return Ok((bink(f, BinOp::BitwiseAnd, v, mask)?, None));
    }
    let n = match id {
        rule::FIXED16 => v,
        rule::SRGB8 => {
            let top = c32(f, 4095)?;
            f.glsl(GlslOp::SClamp, &[q, zero, top])?
        }
        _ => match k.word(c, record::KEPT_BIAS) {
            0 => q,
            bias => bink(f, BinOp::ISub, q, bias)?,
        },
    };
    let unit = c32(f, k.word(c, record::KEPT_UNIT).max(1))?;
    let pr = prec(f, k, c)?;
    let value = soft::ratio(f, k, n, unit, pr)?;
    let v = if use_half {
        let den = rec(f, k, c, record::DEN)?;
        let t = soft::times_rne(f, k, value, den)?;
        let t = if matches!(id, rule::XR_COLOUR | rule::XR_ALPHA) {
            bink(f, BinOp::IAdd, t, 384)?
        } else {
            t
        };
        f.glsl(GlslOp::SClamp, &[t, lo, hi])?
    } else {
        v
    };
    let kept = if k.spec.keep {
        let kept = match xr_i {
            Some(a) => Soft {
                neg: bin(f, BinOp::SLessThan, q, zero)?,
                m: f.convert(ConvOp::UConvert, a, k.u64_)?,
                e: c32(f, -13i32 as u32)?,
                inf: value.inf,
                nan: value.nan,
            },
            None => value,
        };
        Some(soft::pack_f32(f, k, kept)?)
    } else {
        None
    };
    Ok((bink(f, BinOp::BitwiseAnd, v, mask)?, kept))
}

/// Float field `c` (`FLOAT32` mode where `is32`, else `HALF`): the taps decoded, Metal's
/// accumulator, the rounding and the result's bits; `(field bits, half bits kept for
/// RGB9E5's word, f32 bits of the value the level keeps where it keeps them)`.
fn float(
    f: &mut FunctionBuilder<'_>,
    k: &K<'_>,
    c: u32,
    taps: &[Tap],
    is32: bool,
) -> BResult<(Value, Value, Option<Value>)> {
    let offset = k.word(c, record::OFFSET);
    let mask = k.word(c, record::MASK);
    let exp_offset = k.word(c, record::EXP_OFFSET);
    let exp_mask = k.word(c, record::EXP_MASK);
    let sign_mask = rec(f, k, c, record::SIGN)?;
    let mantissa = rec(f, k, c, record::MANTISSA)?;
    let hidden = rec(f, k, c, record::HIDDEN)?;
    let bias = rec(f, k, c, record::BIAS)?;
    let subnormal = rec(f, k, c, record::SUBNORMAL_EXP)?;
    let flush_in = k.word(c, record::FLUSH_IN) != 0;
    // The exponent of infinities and NaNs, where the format has them.
    let special = match k.word(c, record::SPECIAL) {
        u32::MAX => None,
        e => Some(c32(f, e)?),
    };
    let zero = c32(f, 0)?;
    let sentinel = c32(f, 0xc000_0000)?;
    let falsy = bin(f, BinOp::INotEqual, zero, zero)?;
    let (mut nan, mut pos, mut neg) = (falsy, falsy, falsy);
    // The largest exponent estimate of each depth slice (each four taps in turn).
    let mut slice_e = vec![sentinel; taps.len().div_ceil(4)];
    // Per tap: (negative, M, z, live, n, nxy).
    let mut terms = Vec::with_capacity(taps.len());
    for (at, tap) in taps.iter().enumerate() {
        let n = tap.n;
        let b = bits(f, &tap.words, offset, mask)?;
        let ef = bits(f, &tap.words, exp_offset, exp_mask)?;
        let mf = bin(f, BinOp::BitwiseAnd, b, mantissa)?;
        let s = bin(f, BinOp::BitwiseAnd, b, sign_mask)?;
        let negative = bin(f, BinOp::INotEqual, s, zero)?;
        let normal = bin(f, BinOp::INotEqual, ef, zero)?;
        let with_hidden = bin(f, BinOp::BitwiseOr, mf, hidden)?;
        let sub = if flush_in { zero } else { mf };
        let m = select(f, normal, with_hidden, sub)?;
        let is_special = match special {
            Some(special) => Some(bin(f, BinOp::IEqual, ef, special)?),
            None => None,
        };
        let m = match is_special {
            Some(is_special) => select(f, is_special, zero, m)?,
            None => m,
        };
        let zr = select(f, normal, ef, subnormal)?;
        let z = bin(f, BinOp::ISub, zr, bias)?;
        let weighed = bin(f, BinOp::INotEqual, n, zero)?;
        let nonzero = bin(f, BinOp::INotEqual, m, zero)?;
        let live = bin(f, BinOp::LogicalAnd, nonzero, weighed)?;
        let msb_m = f.glsl(GlslOp::FindUMsb, &[m])?;
        let msb_n = f.glsl(GlslOp::FindUMsb, &[n])?;
        let ei = bin(f, BinOp::IAdd, msb_m, z)?;
        let ei = bin(f, BinOp::IAdd, ei, msb_n)?;
        let ei = bink(f, BinOp::ISub, ei, 24)?;
        let ei = select(f, live, ei, sentinel)?;
        slice_e[at / 4] = f.glsl(GlslOp::SMax, &[slice_e[at / 4], ei])?;
        if let Some(is_special) = is_special {
            let counted = bin(f, BinOp::LogicalAnd, is_special, weighed)?;
            let quiet = bin(f, BinOp::INotEqual, mf, zero)?;
            let is_nan = bin(f, BinOp::LogicalAnd, counted, quiet)?;
            let inf = f.unary(UnOp::LogicalNot, quiet)?;
            let is_inf = bin(f, BinOp::LogicalAnd, counted, inf)?;
            let positive = f.unary(UnOp::LogicalNot, negative)?;
            let p_inf = bin(f, BinOp::LogicalAnd, is_inf, positive)?;
            let n_inf = bin(f, BinOp::LogicalAnd, is_inf, negative)?;
            nan = bin(f, BinOp::LogicalOr, nan, is_nan)?;
            pos = bin(f, BinOp::LogicalOr, pos, p_inf)?;
            neg = bin(f, BinOp::LogicalOr, neg, n_inf)?;
        }
        terms.push((negative, m, z, live, n, tap.nxy));
    }
    let precision = k.word(c, record::PRECISION);
    let fraction = c32(f, precision & 0xffff)?;
    let guard = c32(f, precision >> 16)?;
    let quantum = |f: &mut FunctionBuilder<'_>, e: Value| -> BResult<Value> {
        let sc = bin(f, BinOp::ISub, e, fraction)?;
        bin(f, BinOp::ISub, sc, guard)
    };
    let mut e = slice_e[0];
    for &es in &slice_e[1..] {
        e = f.glsl(GlslOp::SMax, &[e, es])?;
    }
    let sc = quantum(f, e)?;
    // A half's accumulator at a 2D level, in 32 bits: each live term `M n 2^(z - 24 - sc)` lies
    // below `2^(e_i + 2 - sc) <= 2^19` (`e_i <= e`, `sc = e - 17`), so the four terms and their
    // sum lie below `2^21`; with `n = 256 nxy`, `M nxy` (below `2^27`) shifted by 8 more is the
    // same truncation. An `f32`'s terms reach `2^40`, a 3D level's products `2^35`: 64 bits.
    let narrow = !is32 && terms.iter().all(|t| t.5.is_some());
    let (zero_w, one_w, cap) = if narrow {
        (zero, c32(f, 1)?, c32(f, 31)?)
    } else {
        (c64(f, k, 0)?, c64(f, k, 1)?, c32(f, 63)?)
    };
    // Each slice's terms truncated to its own quantum and summed; a 3D level's two slice sums
    // then truncated toward zero to the quantum of the largest estimate (`model::accumulate`).
    let mut sum = zero_w;
    for (slice, terms) in terms.chunks(4).enumerate() {
        let sc_s = if slice_e.len() == 1 {
            sc
        } else {
            quantum(f, slice_e[slice])?
        };
        let mut part = zero_w;
        for &(negative, m, z, live, n, nxy) in terms {
            let (p, sh) = match (narrow, nxy) {
                (true, Some(nxy)) => (bin(f, BinOp::IMul, m, nxy)?, bink(f, BinOp::ISub, z, 16)?),
                _ => {
                    let m = f.convert(ConvOp::UConvert, m, k.u64_)?;
                    let n = f.convert(ConvOp::UConvert, n, k.u64_)?;
                    (bin(f, BinOp::IMul, m, n)?, bink(f, BinOp::ISub, z, 24)?)
                }
            };
            let sh = bin(f, BinOp::ISub, sh, sc_s)?;
            let left = bin(f, BinOp::SGreaterThanEqual, sh, zero)?;
            let l = f.glsl(GlslOp::UMin, &[sh, cap])?;
            let l = select(f, left, l, zero)?;
            let minus = bin(f, BinOp::ISub, zero, sh)?;
            let r = f.glsl(GlslOp::UMin, &[minus, cap])?;
            let r = select(f, left, zero, r)?;
            let mag = bin(f, BinOp::ShiftLeftLogical, p, l)?;
            let mag = bin(f, BinOp::ShiftRightLogical, mag, r)?;
            let minus = bin(f, BinOp::ISub, zero_w, mag)?;
            let t = select(f, negative, minus, mag)?;
            let t = select(f, live, t, zero_w)?;
            part = bin(f, BinOp::IAdd, part, t)?;
        }
        let part = if slice_e.len() == 1 {
            part
        } else {
            // `part 2^sc_s` truncated toward zero to `2^sc` (`sc >= sc_s`; a slice of no live term
            // sums to 0 whatever its shift).
            let by = bin(f, BinOp::ISub, sc, sc_s)?;
            let by = f.glsl(GlslOp::UMin, &[by, cap])?;
            let negative = bin(f, BinOp::SLessThan, part, zero_w)?;
            let minus = bin(f, BinOp::ISub, zero_w, part)?;
            let mag = select(f, negative, minus, part)?;
            let mag = bin(f, BinOp::ShiftRightLogical, mag, by)?;
            let minus = bin(f, BinOp::ISub, zero_w, mag)?;
            select(f, negative, minus, mag)?
        };
        sum = bin(f, BinOp::IAdd, sum, part)?;
    }
    // The rounding, half away from zero, of `sum 2^sc`.
    let negative = bin(f, BinOp::SLessThan, sum, zero_w)?;
    let minus = bin(f, BinOp::ISub, zero_w, sum)?;
    let mag = select(f, negative, minus, sum)?;
    let is_zero = bin(f, BinOp::IEqual, mag, zero_w)?;
    let b = if narrow {
        f.glsl(GlslOp::FindUMsb, &[mag])?
    } else {
        soft::msb64(f, k, mag)?
    };
    let x = bin(f, BinOp::IAdd, b, sc)?;
    let emin = rec(f, k, c, record::EMIN)?;
    let ulp = f.glsl(GlslOp::SMax, &[x, emin])?;
    let ulp = bin(f, BinOp::ISub, ulp, fraction)?;
    let drop = bin(f, BinOp::ISub, ulp, sc)?;
    let dropping = bin(f, BinOp::SGreaterThan, drop, zero)?;
    let one = c32(f, 1)?;
    let d = select(f, dropping, drop, one)?;
    let d = f.glsl(GlslOp::UMin, &[d, cap])?;
    let kept = bin(f, BinOp::ShiftRightLogical, mag, d)?;
    let d1 = bink(f, BinOp::ISub, d, 1)?;
    let half = bin(f, BinOp::ShiftRightLogical, mag, d1)?;
    let half = bin(f, BinOp::BitwiseAnd, half, one_w)?;
    let rounded = bin(f, BinOp::IAdd, kept, half)?;
    let minus = bin(f, BinOp::ISub, zero, drop)?;
    let l = f.glsl(GlslOp::UMin, &[minus, cap])?;
    let l = select(f, dropping, zero, l)?;
    let widened = bin(f, BinOp::ShiftLeftLogical, mag, l)?;
    let r = select(f, dropping, rounded, widened)?;
    let r = if narrow {
        r
    } else {
        f.convert(ConvOp::UConvert, r, k.u32_)?
    };
    let two = c32(f, 2)?;
    let carried = bin(f, BinOp::ShiftLeftLogical, two, fraction)?;
    let carry = bin(f, BinOp::IEqual, r, carried)?;
    let r_half = bin(f, BinOp::ShiftRightLogical, r, one)?;
    let r = select(f, carry, r_half, r)?;
    let ulp_up = bin(f, BinOp::IAdd, ulp, one)?;
    let ulp = select(f, carry, ulp_up, ulp)?;
    let msb_r = f.glsl(GlslOp::FindUMsb, &[r])?;
    let x2 = bin(f, BinOp::IAdd, msb_r, ulp)?;
    let bits = if is32 {
        // f32 bits.
        let sign32 = c32(f, 0x8000_0000)?;
        let sign32 = select(f, negative, sign32, zero)?;
        let inf32 = bink(f, BinOp::BitwiseOr, sign32, 0x7f80_0000)?;
        let e32 = bink(f, BinOp::IAdd, x2, 127)?;
        let e32 = bink(f, BinOp::ShiftLeftLogical, e32, 23)?;
        let m32 = bink(f, BinOp::BitwiseAnd, r, 0x7f_ffff)?;
        let n32 = bin(f, BinOp::BitwiseOr, sign32, e32)?;
        let n32 = bin(f, BinOp::BitwiseOr, n32, m32)?;
        let k127 = c32(f, 127)?;
        let over = bin(f, BinOp::SGreaterThan, x2, k127)?;
        let f32_bits = select(f, over, inf32, n32)?;
        // Zero, or below the least normal before rounding where the format flushes it.
        let gone = if k.word(c, record::FLUSH_OUT) != 0 {
            let flushed = bin(f, BinOp::SLessThan, x, emin)?;
            bin(f, BinOp::LogicalOr, is_zero, flushed)?
        } else {
            is_zero
        };
        select(f, gone, zero, f32_bits)?
    } else {
        // f16 bits.
        let sign16 = c32(f, 0x8000)?;
        let sign16 = select(f, negative, sign16, zero)?;
        let inf16 = bink(f, BinOp::BitwiseOr, sign16, 0x7c00)?;
        let sub16 = bin(f, BinOp::BitwiseOr, sign16, r)?;
        let e16 = bink(f, BinOp::IAdd, x2, 15)?;
        let e16 = bink(f, BinOp::ShiftLeftLogical, e16, 10)?;
        let m16 = bink(f, BinOp::BitwiseAnd, r, 0x3ff)?;
        let n16 = bin(f, BinOp::BitwiseOr, sign16, e16)?;
        let n16 = bin(f, BinOp::BitwiseOr, n16, m16)?;
        let k15 = c32(f, 15)?;
        let km14 = c32(f, -14i32 as u32)?;
        let over16 = bin(f, BinOp::SGreaterThan, x2, k15)?;
        let under16 = bin(f, BinOp::SLessThan, x2, km14)?;
        let h = select(f, under16, sub16, n16)?;
        let h = select(f, over16, inf16, h)?;
        // Zero, or a sum that rounds to zero: +0, its sign dropped, as `model.rs`.
        let rounds_to_zero = bin(f, BinOp::IEqual, r, zero)?;
        select(f, rounds_to_zero, zero, h)?
    };
    // Specials.
    let bits = if special.is_some() {
        let (nan_bits, pinf_bits, ninf_bits) = if is32 {
            (0x7fc0_0000, 0x7f80_0000, 0xff80_0000)
        } else {
            (0x7e00, 0x7c00, 0xfc00)
        };
        let both = bin(f, BinOp::LogicalAnd, pos, neg)?;
        let nan = bin(f, BinOp::LogicalOr, nan, both)?;
        let ninf_bits = c32(f, ninf_bits)?;
        let bits = select(f, neg, ninf_bits, bits)?;
        let pinf_bits = c32(f, pinf_bits)?;
        let bits = select(f, pos, pinf_bits, bits)?;
        let nan_bits = c32(f, nan_bits)?;
        select(f, nan, nan_bits, bits)?
    } else {
        bits
    };
    // The field: the bits, shifted (RG11B10), or none (RGB9E5, whose word is made after).
    let (v, kept_half) = match k.word(c, record::ENCODE) {
        encode::HALF_SHIFTED => {
            let shift = k.word(c, record::K);
            (bink(f, BinOp::ShiftRightLogical, bits, shift)?, zero)
        }
        encode::HALF_RGB9E5 => (zero, bits),
        _ => (bits, zero),
    };
    let v = bink(f, BinOp::BitwiseAnd, v, mask)?;
    let value = match (k.spec.keep, is32) {
        (false, _) => None,
        (true, true) => Some(bits),
        (true, false) => Some(half_to_f32(f, k, bits)?),
    };
    Ok((v, kept_half, value))
}

/// Where texel `i`'s field `c` keeps its value, in the kept buffers.
pub(super) fn kept_at(f: &mut FunctionBuilder<'_>, i: Value, c: u32) -> BResult<Value> {
    let at = bink(f, BinOp::IMul, i, KEPT_WORDS)?;
    bink(f, BinOp::IAdd, at, c)
}

/// Write `bits` as texel `i`'s field `c`'s kept value.
pub(super) fn keep_value(
    f: &mut FunctionBuilder<'_>,
    k: &K<'_>,
    i: Value,
    c: u32,
    bits: Value,
) -> BResult<()> {
    let w = kept_at(f, i, c)?;
    let zero = c32(f, 0)?;
    let p = f.access_chain(k.kept_out, &[zero, w])?;
    f.store(p, bits, MemAccess::default())
}

/// Field `c` of destination texel `i` of a sampler level, from its `taps`: its bits placed in
/// their word, the word, and its half bits kept for `RGB9E5Float`'s word (else 0); its kept
/// value written where the level keeps them.
fn field(
    f: &mut FunctionBuilder<'_>,
    k: &K<'_>,
    c: u32,
    taps: &[Tap],
    i: Value,
) -> BResult<(Value, u32, Value)> {
    let zero = c32(f, 0)?;
    let m = k.word(c, record::MODE);
    let (v, half, kept) = match m {
        mode::INTEGER => {
            let (v, kept) = integer(f, k, c, taps)?;
            (v, zero, kept)
        }
        mode::FLOAT32 | mode::HALF => float(f, k, c, taps, m == mode::FLOAT32)?,
        // A constant field (its kept value 0).
        _ => {
            let v = k.word(c, record::CONSTANT) & k.word(c, record::MASK);
            (c32(f, v)?, zero, k.spec.keep.then_some(zero))
        }
    };
    if let Some(kept) = kept {
        keep_value(f, k, i, c, kept)?;
    }
    let offset = k.word(c, record::OFFSET);
    let placed = match offset % 32 {
        0 => v,
        shift => bink(f, BinOp::ShiftLeftLogical, v, shift)?,
    };
    Ok((placed, offset / 32, half))
}

/// `RGB9E5Float`'s word from its three channels' half bits (the model's `rgb9e5`).
fn rgb9e5(f: &mut FunctionBuilder<'_>, h: [Value; 3]) -> BResult<Value> {
    let zero = c32(f, 0)?;
    let mx = f.glsl(GlslOp::UMax, &[h[1], h[2]])?;
    let mx = f.glsl(GlslOp::UMax, &[h[0], mx])?;
    let is_zero = bin(f, BinOp::IEqual, mx, zero)?;
    // (M, z) of a non-negative half.
    let dec = |f: &mut FunctionBuilder<'_>, h: Value| -> BResult<(Value, Value)> {
        let zero = c32(f, 0)?;
        let e = bink(f, BinOp::ShiftRightLogical, h, 10)?;
        let m = bink(f, BinOp::BitwiseAnd, h, 0x3ff)?;
        let normal = bin(f, BinOp::INotEqual, e, zero)?;
        let with_hidden = bink(f, BinOp::BitwiseOr, m, 0x400)?;
        let m = select(f, normal, with_hidden, m)?;
        let ez = bink(f, BinOp::ISub, e, 25)?;
        let sub = c32(f, -24i32 as u32)?;
        let z = select(f, normal, ez, sub)?;
        Ok((m, z))
    };
    let (m, z) = dec(f, mx)?;
    let msb = f.glsl(GlslOp::FindUMsb, &[m])?;
    let ex = bin(f, BinOp::IAdd, msb, z)?;
    let floor = c32(f, -16i32 as u32)?;
    let ex = f.glsl(GlslOp::SMax, &[ex, floor])?;
    let es = bink(f, BinOp::IAdd, ex, 16)?;
    let top = c32(f, 31)?;
    let es = f.glsl(GlslOp::SMin, &[es, top])?;
    let mut word = bink(f, BinOp::ShiftLeftLogical, es, 27)?;
    let cap = c32(f, 31)?;
    let most = c32(f, 511)?;
    for (c, h) in h.into_iter().enumerate() {
        let (m, z) = dec(f, h)?;
        let sh = bin(f, BinOp::ISub, z, es)?;
        let sh = bink(f, BinOp::IAdd, sh, 24)?;
        let left = bin(f, BinOp::SGreaterThanEqual, sh, zero)?;
        let l = f.glsl(GlslOp::UMin, &[sh, cap])?;
        let l = select(f, left, l, zero)?;
        let minus = bin(f, BinOp::ISub, zero, sh)?;
        let r = f.glsl(GlslOp::UMin, &[minus, cap])?;
        let r = select(f, left, zero, r)?;
        let v = bin(f, BinOp::ShiftLeftLogical, m, l)?;
        let v = bin(f, BinOp::ShiftRightLogical, v, r)?;
        let v = f.glsl(GlslOp::UMin, &[v, most])?;
        let v = bink(f, BinOp::ShiftLeftLogical, v, 9 * c as u32)?;
        word = bin(f, BinOp::BitwiseOr, word, v)?;
    }
    select(f, is_zero, zero, word)
}

/// The tap of `coordinate` along one axis: `(x0, x1, w0, w1)` from the axis's table at word
/// `table` and the level above's `big` texels.
fn axis(
    f: &mut FunctionBuilder<'_>,
    k: &K<'_>,
    table: Value,
    coordinate: Value,
    big: Value,
) -> BResult<[Value; 4]> {
    let i = bin(f, BinOp::IAdd, table, coordinate)?;
    let w = param(f, k, i)?;
    let x0 = bink(f, BinOp::BitwiseAnd, w, 0xffff)?;
    let q = bink(f, BinOp::ShiftRightLogical, w, 16)?;
    let next = bink(f, BinOp::IAdd, x0, 1)?;
    let last = bink(f, BinOp::ISub, big, 1)?;
    let x1 = f.glsl(GlslOp::UMin, &[next, last])?;
    let k256 = c32(f, 256)?;
    let w0 = bin(f, BinOp::ISub, k256, q)?;
    Ok([x0, x1, w0, q])
}

/// The words of the first level's texel `at` (`x`, `y`, and the layer or slice) fetched from
/// its sampled view, as the scratch buffer would hold them (`fetch.rs`): each component's code
/// recovered and placed at its bit offset, a texel narrower than a word in the low bits of the
/// first.
fn fetched(
    f: &mut FunctionBuilder<'_>,
    k: &K<'_>,
    fetch: &Fetch,
    at: [Value; 3],
) -> BResult<Vec<Value>> {
    let uvec3 = f.module().ty(Type::Vector {
        elem: k.u32_,
        len: 3,
    })?;
    let coord = f.composite_construct(uvec3, &at)?;
    let ops = ImageOperands {
        lod: Some(c32(f, 0)?),
        ..ImageOperands::default()
    };
    let value = f.fetch(k.source, coord, &ops)?;
    let zero = c32(f, 0)?;
    let mut words = vec![zero; k.spec.bytes.div_ceil(4) as usize];
    for (lane, component) in fetch.components.iter().enumerate() {
        let Some(component) = *component else {
            continue;
        };
        let v = f.composite_extract(value, &[lane as u32])?;
        let code = recovered(f, k, component, v)?;
        let shifted = match component.offset % 32 {
            0 => code,
            shift => bink(f, BinOp::ShiftLeftLogical, code, shift)?,
        };
        let w = &mut words[(component.offset / 32) as usize];
        *w = bin(f, BinOp::BitwiseOr, *w, shifted)?;
    }
    Ok(words)
}

/// The code of a fetched component `v` (a `float`, or a `u32` where the view is an integer
/// one), recovered by its rule (`fetch.rs`), in its low `bits` bits.
fn recovered(
    f: &mut FunctionBuilder<'_>,
    k: &K<'_>,
    component: Component,
    v: Value,
) -> BResult<Value> {
    let bits = component.bits;
    let mask = if bits == 32 {
        u32::MAX
    } else {
        (1 << bits) - 1
    };
    let scaled = |f: &mut FunctionBuilder<'_>, scale: u32| -> BResult<Value> {
        let f32_ = f.module().ty(Type::Float { width: 32 })?;
        let c = f.module().constant(Constant::Scalar {
            ty: f32_,
            bits: ConstBits(u64::from((scale as f32).to_bits())),
        })?;
        let c = f.module().const_value(c)?;
        let product = f.binary(BinOp::FMul, v, c)?;
        f.glsl(GlslOp::RoundEven, &[product])
    };
    match component.recover {
        Recover::Unorm => {
            let r = scaled(f, mask)?;
            f.convert(ConvOp::ConvertFToU, r, k.u32_)
        }
        Recover::Snorm => {
            let r = scaled(f, mask >> 1)?;
            let i32_ = f.module().ty(Type::Int {
                width: 32,
                signed: true,
            })?;
            let r = f.convert(ConvOp::ConvertFToS, r, i32_)?;
            let r = f.convert(ConvOp::Bitcast, r, k.u32_)?;
            bink(f, BinOp::BitwiseAnd, r, mask)
        }
        Recover::Half => {
            let b = f.convert(ConvOp::Bitcast, v, k.u32_)?;
            half_bits(f, b)
        }
        Recover::Bits if bits == 32 => Ok(v),
        Recover::Bits => bink(f, BinOp::BitwiseAnd, v, mask),
    }
}

/// The half-float bits of `f32` bits `b` whose value is a half float, exactly: a normal or
/// subnormal half's value and sign, an infinity, and a NaN as a quiet NaN with the top ten bits
/// of its mantissa (`fetch.rs`). `b` holds a half's value: a smaller one reads as the zero of
/// its sign, a larger one as no value.
fn half_bits(f: &mut FunctionBuilder<'_>, b: Value) -> BResult<Value> {
    let sign = bink(f, BinOp::ShiftRightLogical, b, 16)?;
    let sign = bink(f, BinOp::BitwiseAnd, sign, 0x8000)?;
    let e = bink(f, BinOp::ShiftRightLogical, b, 23)?;
    let e = bink(f, BinOp::BitwiseAnd, e, 0xff)?;
    let m = bink(f, BinOp::BitwiseAnd, b, 0x7f_ffff)?;
    let top = bink(f, BinOp::ShiftRightLogical, m, 13)?;
    // A normal half: exponent `e - 112` (of 1 to 30).
    let ne = bink(f, BinOp::ISub, e, 112)?;
    let ne = bink(f, BinOp::ShiftLeftLogical, ne, 10)?;
    let normal = bin(f, BinOp::BitwiseOr, ne, top)?;
    // A subnormal half (`e` of 103 to 112): the significand shifted down by `126 - e`.
    let hidden = bink(f, BinOp::BitwiseOr, m, 0x80_0000)?;
    let shift = c32(f, 126)?;
    let shift = bin(f, BinOp::ISub, shift, e)?;
    let most = c32(f, 31)?;
    let shift = f.glsl(GlslOp::UMin, &[shift, most])?;
    let subnormal = bin(f, BinOp::ShiftRightLogical, hidden, shift)?;
    // An infinity, or a NaN kept quiet.
    let zero = c32(f, 0)?;
    let is_nan = bin(f, BinOp::INotEqual, m, zero)?;
    let quiet = c32(f, 0x7e00)?;
    let inf = c32(f, 0x7c00)?;
    let special = select(f, is_nan, quiet, inf)?;
    let special = bin(f, BinOp::BitwiseOr, special, top)?;
    let low = c32(f, 103)?;
    let least = c32(f, 113)?;
    let all = c32(f, 255)?;
    let h = bin(f, BinOp::UGreaterThanEqual, e, low)?;
    let h = select(f, h, subnormal, zero)?;
    let is_normal = bin(f, BinOp::UGreaterThanEqual, e, least)?;
    let h = select(f, is_normal, normal, h)?;
    let is_special = bin(f, BinOp::IEqual, e, all)?;
    let h = select(f, is_special, special, h)?;
    bin(f, BinOp::BitwiseOr, h, sign)
}

/// Destination texel `i`'s words at a sampler level: its taps from the level's tap tables
/// (eight in 3D; four where the level above is one texel deep, whose second depth taps weigh
/// 0 and are left out), each source texel's words read once, each field by its record's rule,
/// the values it keeps written where the level keeps them.
fn sampled(f: &mut FunctionBuilder<'_>, k: &K<'_>, i: Value) -> BResult<[Value; 4]> {
    let mut small = [i; 3];
    let mut big = [i; 3];
    let mut tables = [i; 3];
    let axes = if k.spec.depth { 3 } else { 2 };
    for a in 0..axes {
        small[a as usize] = paramk(f, k, header::EXTENT + a)?;
        big[a as usize] = paramk(f, k, header::ABOVE + a)?;
        tables[a as usize] = paramk(f, k, header::TAPS + a)?;
    }
    let x = bin(f, BinOp::UMod, i, small[0])?;
    let r = bin(f, BinOp::UDiv, i, small[0])?;
    let y = bin(f, BinOp::UMod, r, small[1])?;
    let r = bin(f, BinOp::UDiv, r, small[1])?;
    let ax = axis(f, k, tables[0], x, big[0])?;
    let ay = axis(f, k, tables[1], y, big[1])?;
    // The depth axis, where it has two taps: the slices, the layer's first slice, the taps.
    let depth = if k.spec.depth {
        let z = bin(f, BinOp::UMod, r, small[2])?;
        let layer = bin(f, BinOp::UDiv, r, small[2])?;
        let first = bin(f, BinOp::IMul, layer, big[2])?;
        Some((first, axis(f, k, tables[2], z, big[2])?))
    } else {
        None
    };
    let count = if k.spec.depth { 8 } else { 4 };
    let mut taps: Vec<Tap> = Vec::with_capacity(count);
    for t in 0..count {
        let (sx, wx) = if t & 1 == 1 {
            (ax[1], ax[3])
        } else {
            (ax[0], ax[2])
        };
        let (sy, wy) = if (t >> 1) & 1 == 1 {
            (ay[1], ay[3])
        } else {
            (ay[0], ay[2])
        };
        // The image: the layer's (`r`, one slice each), or the layer's depth tap's.
        let (image, n, nxy) = match depth {
            Some((first, az)) => {
                let (sz, wz) = if (t >> 2) & 1 == 1 {
                    (az[1], az[3])
                } else {
                    (az[0], az[2])
                };
                let image = bin(f, BinOp::IAdd, first, sz)?;
                // Split along y last (`model::tap_weight`): the far row's part of `wx wz` rounded
                // half down to `2^-16`, the near row's the rest.
                let nxz = bin(f, BinOp::IMul, wx, wz)?;
                let far = bin(f, BinOp::IMul, nxz, ay[3])?;
                let far = bink(f, BinOp::IAdd, far, 127)?;
                let far = bink(f, BinOp::ShiftRightLogical, far, 8)?;
                let far = bink(f, BinOp::ShiftLeftLogical, far, 8)?;
                let n = if (t >> 1) & 1 == 1 {
                    far
                } else {
                    let whole = bink(f, BinOp::ShiftLeftLogical, nxz, 8)?;
                    bin(f, BinOp::ISub, whole, far)?
                };
                (image, n, None)
            }
            // The one depth tap weighs 256.
            None => {
                let nxy = bin(f, BinOp::IMul, wx, wy)?;
                (r, bink(f, BinOp::ShiftLeftLogical, nxy, 8)?, Some(nxy))
            }
        };
        let words = match &k.spec.fetch {
            // The first level's view: a 2D array's layer or a 3D texture's slice is `image`.
            Some(fetch) => fetched(f, k, fetch, [sx, sy, image])?,
            None => {
                let index = bin(f, BinOp::IMul, image, big[1])?;
                let index = bin(f, BinOp::IAdd, index, sy)?;
                let index = bin(f, BinOp::IMul, index, big[0])?;
                let index = bin(f, BinOp::IAdd, index, sx)?;
                texel(f, k, index)?
            }
        };
        taps.push(Tap { words, n, nxy });
    }
    let zero = c32(f, 0)?;
    let mut words = [zero; 4];
    let mut halves = [zero; 3];
    for c in 0..k.fields() {
        let (placed, word, half) = field(f, k, c, &taps, i)?;
        place(f, &mut words, placed, word)?;
        if let Some(h) = halves.get_mut(c as usize) {
            *h = half;
        }
    }
    if k.spec.rgb9e5 {
        words[0] = rgb9e5(f, halves)?;
    }
    Ok(words)
}

/// `placed` ORed into word `word` of a texel's `words`.
pub(super) fn place(
    f: &mut FunctionBuilder<'_>,
    words: &mut [Value; 4],
    placed: Value,
    word: u32,
) -> BResult<()> {
    let w = &mut words[word as usize];
    *w = bin(f, BinOp::BitwiseOr, *w, placed)?;
    Ok(())
}

/// Destination texel `i`'s `words` written: a sub-word texel ORed into the zeroed level, a
/// wider one stored word by word.
fn store_texel(f: &mut FunctionBuilder<'_>, k: &K<'_>, i: Value, words: [Value; 4]) -> BResult<()> {
    let zero = c32(f, 0)?;
    let bytes = k.spec.bytes;
    if bytes < 4 {
        let byte = bink(f, BinOp::IMul, i, bytes)?;
        let word = bink(f, BinOp::ShiftRightLogical, byte, 2)?;
        let lane = bink(f, BinOp::BitwiseAnd, byte, 3)?;
        let lane = bink(f, BinOp::ShiftLeftLogical, lane, 3)?;
        let v = bin(f, BinOp::ShiftLeftLogical, words[0], lane)?;
        let p = f.access_chain(k.destination, &[zero, word])?;
        f.atomic(AtomicOp::Or, p, Scope::Device, Semantics::RELAXED, v)?;
        return Ok(());
    }
    let per = bytes / 4;
    let first = bink(f, BinOp::IMul, i, per)?;
    for (j, w) in words.into_iter().take(per as usize).enumerate() {
        let at = match j {
            0 => first,
            j => bink(f, BinOp::IAdd, first, j as u32)?,
        };
        let p = f.access_chain(k.destination, &[zero, at])?;
        f.store(p, w, MemAccess::default())?;
    }
    Ok(())
}

/// The filter module of `spec` as a typed module, before serialization: the module
/// [`super::mipmap_filter_module`] serializes, which the CPU interpreter runs. Its bindings are
/// the symbols `SOURCE` (or `SOURCE_IMAGE`, the first level's sampled view, where the spec
/// fetches), `DESTINATION`, `PARAMS`, `KEPT_IN` and `KEPT_OUT` (the module docs).
pub(super) fn typed_module(spec: &Spec) -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let u64_ = b.ty(Type::Int {
        width: 64,
        signed: false,
    })?;
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
    let source = match &spec.fetch {
        Some(view) => {
            let sampled_type = if fetch::integer(&view.components) {
                u32_
            } else {
                b.ty(Type::Float { width: 32 })?
            };
            let (dim, arrayed) = if view.volume {
                (Dim::D3, false)
            } else {
                (Dim::D2, true)
            };
            let ty = b.ty(Type::Image(ImageType {
                sampled_type,
                dim,
                depth: false,
                arrayed,
                multisampled: false,
                usage: ImageUsage::Sampled,
                format: crate::spv::ImageFormat::Unknown,
            }))?;
            b.resource(SOURCE_IMAGE, StorageClass::UniformConstant, ty)?
        }
        None => words_buffer(&mut b, SOURCE, u32_)?,
    };
    let destination = words_buffer(&mut b, DESTINATION, u32_)?;
    let params = words_buffer(&mut b, PARAMS, u32_)?;
    let kept_in = words_buffer(&mut b, KEPT_IN, u32_)?;
    let kept_out = words_buffer(&mut b, KEPT_OUT, u32_)?;
    let srgb_table = srgb::declare(&mut b)?;
    let xr_table = extended_range::declare(&mut b)?;
    let invocation = b.builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let source = f.module().global_value(source)?;
    // The sampled view is loaded once, where every tap's fetch reads it.
    let source = match spec.fetch {
        Some(_) => f.load(source, MemAccess::default())?,
        None => source,
    };
    let destination = f.module().global_value(destination)?;
    let params = f.module().global_value(params)?;
    let kept_in = f.module().global_value(kept_in)?;
    let kept_out = f.module().global_value(kept_out)?;
    let k = K {
        u32_,
        u64_,
        source,
        destination,
        params,
        kept_in,
        kept_out,
        srgb: srgb_table,
        xr: xr_table,
        spec,
    };
    let texels = paramk(&mut f, &k, header::TEXELS)?;
    let row = paramk(&mut f, &k, header::ROW)?;
    let invocation = f.module().global_value(invocation)?;
    let invocation = f.load(invocation, MemAccess::default())?;
    let gx = f.composite_extract(invocation, &[0])?;
    let gy = f.composite_extract(invocation, &[1])?;
    let i = bin(&mut f, BinOp::IMul, gy, row)?;
    let i = bin(&mut f, BinOp::IAdd, i, gx)?;
    let inside = bin(&mut f, BinOp::ULessThan, i, texels)?;
    let body: ArmFn<'_, ()> = Box::new(move |f, _| {
        let words = if spec.tail {
            super::tail_kernel::texel(f, &k, i)?
        } else {
            sampled(f, &k, i)?
        };
        store_texel(f, &k, i, words)?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    f.if_(&mut (), inside, body, None)?;
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize {
            x: LOCAL_SIZE,
            y: 1,
            z: 1,
        }],
    )?;
    b.finalize()
}
