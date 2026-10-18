//! A depth comparison's per-tap test, the one entry point every comparison the shader makes
//! itself takes: a cube's comparison gather (`cube::gather`) and a linear comparison
//! (`linear::sum`).
//!
//! **What Metal compares** (Apple M4, macOS 27.0.1; the probes for cube gathers, for 2D linear and
//! nearest comparisons and comparison gathers, each run with and without Metal's validation layer;
//! the Apple M3 Max, macOS 26.7, answers every lane of the four alike). Each tap is compared on its
//! own, the reference `r` against the tap's depth `x` as `r OP x` in `f32`, `1` where it passes and
//! `0` elsewhere; a NaN fails every ordered comparison and passes `not_equal`. A float format's
//! reference and texel are compared as given (NaN, denormals and values outside `[0, 1]` included);
//! a `Depth16Unorm` texture's as codes, the texel's its code and the reference's the code of its
//! exact product with 65535 ([`reference`]). The shader compares the values' integer keys
//! ([`key`]), not the floats: a device's float comparison may flush a denormal to zero, as x86
//! lavapipe's does (`runtime_lane::depth_compares` with float comparisons: 82 of 3,392
//! `Depth32Float` comparisons differed from Metal's, such as the least denormal reference passing
//! `less_equal` against a zero texel; with the keys, none), and Metal's does not. A tap outside a
//! border axis compares the border's depth as it would a texel's (zero, opaque black and opaque
//! white borders under `less` and `greater`, every lane of 2,048).
//!
//! **A linear comparison** (2,560 lanes clamping to the edge, 2,048 to zero, and 2,032 through a
//! runtime pixel sampler; `Depth16Unorm` alike) is an `r8Unorm` texture's linear sample of the
//! passes as codes 0 and 255 (`linear::sum` with [`PASS_FORMAT`]): the exact weighted sum of the
//! passes rounded once onto `1/4080`, ties up.
//!
//! `TapCompare::new` takes the sampler's compare function; a constexpr sampler's is known when
//! the function compiles. A runtime sampler's is its word's `COMPARE` field, the raw
//! `MTLCompareFunction`, whose bits 0, 1 and 2 say whether a reference below, equal to, and above
//! the texel passes (`never` 0, `less` 1, `equal` 2, `less_equal` 3, `greater` 4, `not_equal` 5,
//! `greater_equal` 6, `always` 7): `TapCompare::of_word` tests the three orders and takes the bits,
//! and where either value is a NaN passes exactly where bits 0 and 2 are both set (`not_equal` and
//! `always`), as the named functions do.

use crate::air::abi::CompareFunction;
use crate::lower::emit::EmitCx;
use crate::metal::bicubic::BicubicTexels;
use crate::metal::sampler::SamplerWord;
use crate::metal::PixelFormat;
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::Value as SValue;
use crate::spirv::ir::{BinOp, ConvOp, GlslOp, Type as SType, UnOp};

use super::super::bicubic::{scalar, value, Bits, Tys};

/// The format whose linear sum of codes 0 and 255 a linear comparison's passes are summed as
/// (module docs).
pub(in crate::lower::sampler) const PASS_FORMAT: PixelFormat = PixelFormat::R8Unorm;

/// The census words of a comparison's setup: its reference's code ([`reference`]) and key
/// ([`key`]), the format's test, and their constants.
pub(in crate::lower::sampler) const REFERENCE_WORDS: u32 = 240;

/// The census words of one tap's comparison: the depth's extraction, its code and key, the test and
/// the NaNs', the pass's select, and the tap's assembly.
pub(in crate::lower::sampler) const TAP_WORDS: u32 = 112;

/// The census words a runtime sampler's comparison adds to its setup: the word's three function
/// bits and the NaN's pass, with their constants.
pub(in crate::lower::sampler) const WORD_REFERENCE_WORDS: u32 = 64;

/// The census words a runtime sampler's comparison adds to each tap: the three orders' tests and
/// their selection by the function's bits.
pub(in crate::lower::sampler) const WORD_TAP_WORDS: u32 = 48;

/// A comparison's function (module docs): a constexpr sampler's, or a runtime sampler's word's
/// `COMPARE` field.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Comparison {
    Function(CompareFunction),
    Word,
}

impl Comparison {
    /// The census words of the comparison's setup and of `taps` taps' tests.
    pub(in crate::lower::sampler) fn words(self, taps: u32) -> u32 {
        let base = REFERENCE_WORDS + taps * TAP_WORDS;
        match self {
            Comparison::Function(_) => base,
            Comparison::Word => base + WORD_REFERENCE_WORDS + taps * WORD_TAP_WORDS,
        }
    }
}

/// The code steps of a normalized depth format, `Depth16Unorm`'s (the only one whose
/// `TexelFormat` is `BicubicTexels::EXACT`, `metal::texel_format`).
const DEPTH_CODES: f32 = 65535.0;

/// How a comparison tests each tap (module docs): its reference as compared and whether the
/// bound format compares codes.
pub(in crate::lower::sampler) struct TapCompare {
    test: Test,
    /// The key of the reference as compared ([`reference`], [`key`]).
    reference: SValue,
    /// The reference is not a NaN (a `bool`).
    ordered: SValue,
    /// The bound format is normalized (`Depth16Unorm`, a `bool`): its codes are compared.
    normalized: SValue,
}

/// A compare function's test of reference `r` against texel `x`.
#[derive(Clone, Copy)]
enum Test {
    /// `r OP x` on their keys ([`key`]), passing where either is a NaN iff `unordered`.
    Op {
        op: BinOp,
        unordered: bool,
    },
    Always,
    Never,
    /// A runtime sampler's function (module docs): `r < x`, `r == x` and `r > x` on their keys,
    /// each passing where its bit (`below`, `equal`, `above`, `bool`s) is set, and where either
    /// is a NaN `unordered` (a `bool`).
    Word {
        below: SValue,
        equal: SValue,
        above: SValue,
        unordered: SValue,
    },
}

impl TapCompare {
    /// The comparison of reference `dref` under `comparison`, a runtime sampler's by `word` (its
    /// `SamplerState` specialization's word, a `u32`), on a texture bound in the format whose
    /// treatment is `bits` ([`TapCompare::new`], [`TapCompare::of_word`]).
    pub(in crate::lower::sampler) fn of(
        cx: &mut EmitCx<'_>,
        fb: &mut FunctionBuilder<'_>,
        t: &Tys,
        (comparison, word): (Comparison, Option<SValue>),
        (dref, bits): (SValue, Bits),
    ) -> BResult<TapCompare> {
        match (comparison, word) {
            (Comparison::Function(f), _) => TapCompare::new(cx, fb, t, f, (dref, bits)),
            (Comparison::Word, Some(w)) => TapCompare::of_word(fb, t, w, (dref, bits)),
            (Comparison::Word, None) => {
                Err(cx.fail("a runtime sampler's comparison without its word"))
            }
        }
    }

    /// The comparison of reference `dref` under the function runtime sampler word `word` (a
    /// `u32`) states in its `COMPARE` field (module docs), on a texture bound in the format whose
    /// treatment is `bits`.
    pub(in crate::lower::sampler) fn of_word(
        fb: &mut FunctionBuilder<'_>,
        t: &Tys,
        word: SValue,
        (dref, bits): (SValue, Bits),
    ) -> BResult<TapCompare> {
        let field = super::super::bicubic::word_field(
            fb,
            t,
            word,
            SamplerWord::COMPARE_SHIFT,
            SamplerWord::COMPARE >> SamplerWord::COMPARE_SHIFT,
        )?;
        let zero = scalar(fb, t.u32_, 0)?;
        let zero = value(fb, zero)?;
        let mut set = [field; 3];
        for (k, b) in set.iter_mut().enumerate() {
            let bit = scalar(fb, t.u32_, 1 << k)?;
            let bit = value(fb, bit)?;
            let and = fb.binary(BinOp::BitwiseAnd, field, bit)?;
            *b = fb.binary(BinOp::INotEqual, and, zero)?;
        }
        let [below, equal, above] = set;
        let unordered = fb.binary(BinOp::LogicalAnd, below, above)?;
        let normalized = bits.test(fb, t, BicubicTexels::EXACT)?;
        let reference = reference(fb, dref, normalized)?;
        let (reference, ordered) = key(fb, reference)?;
        Ok(TapCompare {
            test: Test::Word {
                below,
                equal,
                above,
                unordered,
            },
            reference,
            ordered,
            normalized,
        })
    }

    /// The comparison of reference `dref` (an `f32`) under `function`, on a texture bound in the
    /// format whose treatment is `bits` (a slot's `TexelFormat` bits; `EXACT` is the normalized
    /// `Depth16Unorm`'s).
    pub(in crate::lower::sampler) fn new(
        cx: &mut EmitCx<'_>,
        fb: &mut FunctionBuilder<'_>,
        t: &Tys,
        function: CompareFunction,
        (dref, bits): (SValue, Bits),
    ) -> BResult<TapCompare> {
        let ordered = |op| Test::Op {
            op,
            unordered: false,
        };
        let test = match function {
            CompareFunction::Less => ordered(BinOp::SLessThan),
            CompareFunction::LessEqual => ordered(BinOp::SLessThanEqual),
            CompareFunction::Greater => ordered(BinOp::SGreaterThan),
            CompareFunction::GreaterEqual => ordered(BinOp::SGreaterThanEqual),
            CompareFunction::Equal => ordered(BinOp::IEqual),
            CompareFunction::NotEqual => Test::Op {
                op: BinOp::INotEqual,
                unordered: true,
            },
            CompareFunction::Always => Test::Always,
            CompareFunction::Never => Test::Never,
            CompareFunction::None => return Err(cx.fail("a comparison without its function")),
        };
        let normalized = bits.test(fb, t, BicubicTexels::EXACT)?;
        let reference = reference(fb, dref, normalized)?;
        let (reference, ordered) = key(fb, reference)?;
        Ok(TapCompare {
            test,
            reference,
            ordered,
            normalized,
        })
    }

    /// `1.0` where the reference passes against depth `x` (an `f32`: a fetched tap's, the
    /// border's, or a cube's made-up corner's), else `0.0`.
    pub(in crate::lower::sampler) fn pass(
        &self,
        fb: &mut FunctionBuilder<'_>,
        x: SValue,
    ) -> BResult<SValue> {
        let (one, zero) = (float(fb, 1.0)?, float(fb, 0.0)?);
        let (op, unordered) = match self.test {
            Test::Op { op, unordered } => (op, unordered),
            Test::Always => return Ok(one),
            Test::Never => return Ok(zero),
            Test::Word {
                below,
                equal,
                above,
                unordered,
            } => {
                let x = compared_texel(fb, x, self.normalized)?;
                let (x, ordered) = key(fb, x)?;
                let mut passes = [x; 3];
                for (p, (op, bit)) in passes.iter_mut().zip([
                    (BinOp::SLessThan, below),
                    (BinOp::IEqual, equal),
                    (BinOp::SGreaterThan, above),
                ]) {
                    let order = fb.binary(op, self.reference, x)?;
                    *p = fb.binary(BinOp::LogicalAnd, order, bit)?;
                }
                let [lt, eq, gt] = passes;
                let pass = fb.binary(BinOp::LogicalOr, lt, eq)?;
                let pass = fb.binary(BinOp::LogicalOr, pass, gt)?;
                let ordered = fb.binary(BinOp::LogicalAnd, self.ordered, ordered)?;
                let pass = fb.select(ordered, pass, unordered)?;
                return fb.select(pass, one, zero);
            }
        };
        let x = compared_texel(fb, x, self.normalized)?;
        let (x, ordered) = key(fb, x)?;
        let pass = fb.binary(op, self.reference, x)?;
        let ordered = fb.binary(BinOp::LogicalAnd, self.ordered, ordered)?;
        let pass = if unordered {
            let nan = fb.unary(UnOp::LogicalNot, ordered)?;
            fb.binary(BinOp::LogicalOr, pass, nan)?
        } else {
            fb.binary(BinOp::LogicalAnd, pass, ordered)?
        };
        fb.select(pass, one, zero)
    }
}

/// The `f32` constant `v`.
fn float(fb: &mut FunctionBuilder<'_>, v: f32) -> BResult<SValue> {
    let f32_ = fb.module().ty(SType::Float { width: 32 })?;
    let c = scalar(fb, f32_, u64::from(v.to_bits()))?;
    value(fb, c)
}

/// The value a comparison compares reference `r` as (the probes, Apple M4 on macOS 27.0.1): a float
/// format's as given, NaN and values outside `[0, 1]` included; where `normalized`
/// (`Depth16Unorm`), its code: `r` clamped into `[0, 1]` (`m`) and the exact product `m · 65535`
/// rounded to the nearest integer (at the one half-way reference, 0.5, up to 32768), a NaN kept.
/// The product is rounded once in `f32` to `k` and corrected: with `h = m · 65536` (exact), the
/// exact product lies above `k + 1/2` exactly where `h - (k + 1/2) > m`, and below `k -
/// 1/2` where `h - (k - 1/2) < m`, each difference exact where it decides (its operands within a
/// factor of two). Metal rounds the exact product: our probe's 96 references whose
/// `f32` product rounds to a half-way point between the other code and the exact product's code
/// each read the exact product's.
fn reference(fb: &mut FunctionBuilder<'_>, r: SValue, normalized: SValue) -> BResult<SValue> {
    let (zero, one, half) = (float(fb, 0.0)?, float(fb, 1.0)?, float(fb, 0.5)?);
    let (codes, wide) = (float(fb, DEPTH_CODES)?, float(fb, DEPTH_CODES + 1.0)?);
    let m = fb.glsl(GlslOp::FClamp, &[r, zero, one])?;
    let p = fb.binary_no_contraction(BinOp::FMul, m, codes)?;
    let k = fb.glsl(GlslOp::RoundEven, &[p])?;
    let h = fb.binary_no_contraction(BinOp::FMul, m, wide)?;
    let above = fb.binary_no_contraction(BinOp::FAdd, k, half)?;
    let above = fb.binary_no_contraction(BinOp::FSub, h, above)?;
    let up = fb.binary(BinOp::FOrdGreaterThan, above, m)?;
    let below = fb.binary_no_contraction(BinOp::FSub, k, half)?;
    let below = fb.binary_no_contraction(BinOp::FSub, h, below)?;
    let down = fb.binary(BinOp::FOrdLessThan, below, m)?;
    let next = fb.binary_no_contraction(BinOp::FAdd, k, one)?;
    let previous = fb.binary_no_contraction(BinOp::FSub, k, one)?;
    let k = fb.select(up, next, k)?;
    let k = fb.select(down, previous, k)?;
    let nan = fb.unary(UnOp::IsNan, r)?;
    let code = fb.select(nan, r, k)?;
    fb.select(normalized, code, r)
}

/// The key comparison orders `f32` value `v` by (an `i32`), and whether `v` is not a NaN (a
/// `bool`): `v`'s bits as a signed integer where its sign is clear, else its magnitude's bits
/// negated, so that `-0` and `+0` share a key and the keys of the other values order as the values
/// do, denormals included. No float operation reads `v` (module docs).
fn key(fb: &mut FunctionBuilder<'_>, v: SValue) -> BResult<(SValue, SValue)> {
    let i32_ = fb.module().ty(SType::Int {
        width: 32,
        signed: true,
    })?;
    let int = |fb: &mut FunctionBuilder<'_>, x: u32| -> BResult<SValue> {
        let c = scalar(fb, i32_, u64::from(x))?;
        value(fb, c)
    };
    let (zero, magnitudes, infinity) = (int(fb, 0)?, int(fb, 0x7fff_ffff)?, int(fb, 0x7f80_0000)?);
    let bits = fb.convert(ConvOp::Bitcast, v, i32_)?;
    let magnitude = fb.binary(BinOp::BitwiseAnd, bits, magnitudes)?;
    let negative = fb.binary(BinOp::SLessThan, bits, zero)?;
    let negated = fb.unary(UnOp::SNegate, magnitude)?;
    let key = fb.select(negative, negated, bits)?;
    let ordered = fb.binary(BinOp::SLessThanEqual, magnitude, infinity)?;
    Ok((key, ordered))
}

/// The value a comparison compares texel `x` (a fetched tap, the border's, or the made-up
/// corner) as: a float format's as fetched; where `normalized`, its code, `x · 65535` rounded to
/// nearest (exact: the fetch is the code over 65535 within far less than half a step).
fn compared_texel(fb: &mut FunctionBuilder<'_>, x: SValue, normalized: SValue) -> BResult<SValue> {
    let codes = float(fb, DEPTH_CODES)?;
    let c = fb.binary_no_contraction(BinOp::FMul, x, codes)?;
    let c = fb.glsl(GlslOp::RoundEven, &[c])?;
    fb.select(normalized, c, x)
}
