//! `lower::arith`: math, conversions, comparisons, and composites.
//!
//! `lower::select` routes every non-pointer arithmetic instruction here (binary operators,
//! `fneg`, numeric casts and bitcasts, `icmp`/`fcmp`, `select`, `freeze`, vector and aggregate
//! element operations) together with the `air` math, conversion, and packing intrinsics and the
//! `llvm` math intrinsics. [`select`] decides one [`ArithStrategy`] per instruction; [`emit`]
//! routes it to the emission core in [`ops`] (the LLVM instruction set) and [`math`] (the
//! intrinsics). `air.sincos` stores its cosine through the access family's store. Everything not
//! lowered yet is refused as `Unsupported{Lowering(Arith), NotImplemented}`: `air.pack` to the
//! packed float formats `rg11b10f` and `rgb9e5` (their unpacking is lowered), `i1` results of
//! `air.convert` from anything but a `float` or `half` (whose reading is Metal's), and the few
//! chunked-vector operations [`chunk`] lists. A vector of other than 2 to 4 lanes is held as a
//! composite of chunks and operated on per chunk ([`chunk`]).
//!
//! This family also owns the **lowered representation of non-pointer values** ([`legal`]):
//! [`lowered_type`] is what `select` demotes block-crossing values and phis to and what it requires
//! of terminator operands, [`constant`] is which constants lowering materializes, and
//! [`spirv_type`] and [`materialize`] are their emission. The representation depends on the AIR
//! type and `shaderInt64` only: `i1` is `bool`, other integers are held zero-extended in a 32- or
//! 64-bit unsigned container (so `i8`/`i16`/`i2`/`i40` need no narrow capability), `i33`..`i64` as
//! two `u32` words without `shaderInt64` ([`wide`]: every operation on them is
//! `ArithStrategy::Wide`, per chunk in a chunked vector; division unrolled without a loop), `float`
//! and `double` are native (`f64` needs `shaderFloat64`), and `half` and `bfloat` are held widened
//! in an `f32` that is always a value of the narrow format ([`narrow`]): each operation rounds its
//! `f32` result back to the format, so no profile needs `shaderFloat16`; where the device's 16-bit
//! `+ - *` is Metal's `half` arithmetic and the profile has `shaderFloat16`, a `half` `+ - *` is
//! that instruction, kept apart, between exact conversions instead ([`ArithStrategy::F16`]): where
//! it is measured to be, and where the 16-bit float controls the module then declares make it so
//! by the Vulkan spec ([`narrow::controlled_modes`]).
//!
//! Precision: a `half` rounding keeps half denormals, as Metal does; a `float` or `half` fused
//! multiply-add (`air.fma`, `llvm.fma`, and `llvm.fmuladd`, which Metal fuses) rounds once on every
//! profile ([`fma`]); operations whose Vulkan precision is looser than correct rounding add
//! `PrecisionNote::DeviceArithmetic` (`fdiv`, `frem`, `sqrt`, a `double` or `bfloat` `fma`, `fmod`,
//! `dot`) or `PrecisionNote::DeviceTranscendentals` (the transcendental functions) to the plan.
//! Float arithmetic without both of LLVM's `contract` and `reassoc` fast-math flags (and `mix`'s
//! multiply and add, unless its call carries both) is decorated `NoContraction`, so no device fuses
//! or regroups operations that LLVM keeps apart. A `contract` multiply is fused into the `contract`
//! additions that use it exactly where Metal fuses it, one correctly rounded fused multiply-add
//! ([`contract`]); an addition Metal keeps apart from such a multiply is `NoContraction`.
//! Arithmetic the flags let a compiler regroup with its neighbour, a pair whose fusion the probe
//! did not measure, and a `bfloat` or `double` `llvm.fmuladd`, add
//! `PrecisionNote::FastMathRewrites`: its value is any of those the flags allow, and Metal's
//! compiler may take another than the device's. Every other lowering is exact for every non-poison
//! input within the device's floating-point environment: signed zeros, infinities, and NaNs are
//! preserved by `SignedZeroInfNanPreserve` where an operation without `nnan`/`ninf`/`nsz` requires
//! it, or a check or a fused multiply-add routine answers Metal's zero of its sign
//! ([`ArithPlan::keep_zero_signs`]), and the profile supports it (else
//! `PrecisionNote::DeviceSignedZeroInfNan`), while denormal flushing and the rounding mode are the
//! device default (the rounding mode decides the inexact `FConvert` narrowings and integer-to-float
//! conversions, which Vulkan leaves to the device without `RoundingModeRTE`), but in an entry that
//! rounds to `half` under the float controls ([`narrow::controlled_modes`]), which declares
//! `RoundingModeRTE`, `DenormPreserve` and `SignedZeroInfNanPreserve` for 16 bits and the modes the
//! device's independence ties to them: `RoundingModeRTE` for 32 bits on a device whose rounding
//! modes are not independent, Metal's own rounding of `float` arithmetic, and for 64 bits the
//! rounding and denormals IEEE 754 gives `double`. `sign`,
//! `fract`, `saturate`, `median3`, and `pow` answer as Metal does at zeros, NaNs, infinities, and
//! `f32` denormals whatever the device's modes, and the normalized and sRGB `air.unpack`/`air.pack`
//! conversions are exact integer sequences and Metal's tables.
//!
//! A NaN result is Metal's ([`nan`]): `float` `+ - * /`, the fused multiply-add on `OpFmaKHR`, and
//! the functions whose NaN Metal canonicalizes answer `0x7FC00000`, and `half` `+`, `*`, fused
//! multiply-add, and a `float` narrowed to `half` the canonical `half` NaN `0x7E00`, which the
//! lowering selects where the result's bits are observed unless the profile's measured device fact
//! leaves it to the device (`NanEncoding::MetalCanonical`); the functions whose NaN carries an
//! operand's sign or bits answer it on every profile.

pub mod aggregate;
pub mod chunk;
pub mod color_write;
pub mod contract;
pub mod denormal;
pub mod divide;
pub mod exact_division;
pub mod extended_range;
pub mod fma;
pub mod legal;
pub mod libm;
pub mod math;
pub mod nan;
pub mod narrow;
pub mod ops;
pub mod output_precision;
pub(crate) mod reassoc;
pub(crate) mod regroup;
pub mod srgb;
pub mod texel_round;
pub mod wide;

#[cfg(test)]
mod tests;

use crate::air::intrinsics::{
    AirIntrinsic, Convert, LlvmFloatOp, LlvmIntOp, LlvmIntrinsic, MathFn, NumKind, Pack, PackDir,
    PackFormat, Sign as AirSign,
};
use crate::air::intrinsics::{Elem, Math};
use crate::air::ir::{
    BinOp as AirBin, BlockId, CastOp, Const, FastMath, FloatPredicate, Inst, InstId, InstKind,
    IntPredicate, Type, TypeId as AirTypeId, Value, ValueId,
};
use crate::air::Entry;
use crate::budget::Meter;
use crate::device::profile::{
    DriverWorkaround, FloatFunctions, MinMaxRoute, NanEncoding, RoundingOrderRoute,
};
use crate::device::{DeviceProfile, FloatWidth};
use crate::error::Error;
use crate::lower::access;
use crate::lower::emit::EmitCx;
use crate::lower::select::{FamilyPlans, Plan, SelectCx};
use crate::lower::strategy::FamilyStrategy;
use crate::pipeline::artifact::{EntryMath, PrecisionNote};
use crate::spirv::builder::{FunctionBuilder, MaskForm, ModuleBuilder, ProductForm, QuantizeForm};
use crate::spirv::error::{BResult, BuildError, TypeRule};
use crate::spirv::ids::{DemoteTy, FuncId, GlobalId, TypeId, Value as SValue};
use crate::spirv::ir::{Contraction, ExecMode, UnOp};
use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};

use legal::{check, demote_of, form_on, not_lowered};
pub use legal::{FloatFmt, Form, Held, LoweredConst};
use math::{ConvertKind, FloatFn, FloatPrecision, IntFn, Sign};
use ops::{BitcastPlan, CastKind};

/// The lowered representation of a non-pointer AIR value of type `ty` on profile `p`: a type every
/// lowered value of `ty` has, which is always demotable. A type this family does not lower yet is
/// `Unsupported{Lowering(Arith), NotImplemented}`; a type that needs a capability the profile lacks
/// is a typed refusal naming it.
///
/// On a profile without `shaderInt64` every 64-bit integer scalar or vector is held as two words
/// ([`legal::widen`]), each chunk of a chunked vector of them too.
pub fn lowered_type(e: &Entry, p: &DeviceProfile, ty: AirTypeId) -> Result<DemoteTy, Error> {
    let mut d = demote_of(e.module.types(), ty)?;
    if legal::holds_wide(p) {
        d = legal::widen(&d);
    }
    check(p, &d)?;
    Ok(d)
}

/// `v` as a constant lowering materializes, in its profile-independent representation (64-bit
/// integers native). It is `Some` only for constants of a type [`lowered_type`] lowers on some
/// profile; `select` checks the profile. Emission materializes [`constant_in`].
pub fn constant(e: &Entry, v: Value) -> Option<LoweredConst> {
    let Value::Const(c) = v else { return None };
    legal::lower_const(e, c)
}

/// [`constant`] in the representation of the plan's profile: 64-bit integers as two words where
/// the plan holds them so.
pub fn constant_in(e: &Entry, plan: &ArithPlan, v: Value) -> Option<LoweredConst> {
    let k = constant(e, v)?;
    Some(if plan.wide { legal::widen_const(k) } else { k })
}

/// [`constant`] in lowered representation `d`: the constant itself, or its wide form when `d` is
/// the wide representation of its type; `None` when neither is `d`.
pub fn constant_as(e: &Entry, v: Value, d: &DemoteTy) -> Option<LoweredConst> {
    let k = constant(e, v)?;
    if k.demote() == *d {
        return Some(k);
    }
    let w = legal::widen_const(k);
    (w.demote() == *d).then_some(w)
}

/// The SPIR-V type of lowered representation `ty`, or `None` for one this family does not
/// produce.
pub fn spirv_type(mb: &mut ModuleBuilder, ty: &DemoteTy) -> BResult<Option<TypeId>> {
    legal::spirv_type(mb, ty)
}

/// Materialize constant `k` in the open function, or `None` for one this family does not
/// produce.
pub fn materialize(fb: &mut FunctionBuilder<'_>, k: &LoweredConst) -> BResult<Option<SValue>> {
    legal::materialize(fb, k)
}

/// The arith strategies. Operands are the instruction's value operands (a call's arguments) in
/// order; each variant carries the forms it was selected for.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ArithStrategy {
    /// An integer binary operator on held integers.
    Int {
        op: AirBin,
        form: Form,
    },
    /// A binary operator on `i1` values.
    Bool {
        op: AirBin,
        form: Form,
    },
    /// A float binary operator on floats of `form`; `contract` is `Forbidden` unless the
    /// instruction carries the `contract` fast-math flag.
    Float {
        op: AirBin,
        form: Form,
        contract: Contraction,
    },
    /// A `half` `fadd`, `fsub` or `fmul` on lanes of `form` by the device's 16-bit arithmetic,
    /// Metal's own in the forms that keep it apart (measured, `DriverDeviation::MetalHalfArithmetic`,
    /// or by the Vulkan spec under the 16-bit float controls, [`ops::f16_binary`]): the operands
    /// narrowed to 16 bits, exactly, the instruction kept apart, and its result widened, exactly.
    F16 {
        op: AirBin,
        form: Form,
    },
    /// `fneg` on floats of `form`: the sign bit flipped on the bits where `bits`
    /// ([`math::sign_on_bits`]: `float`), else the device's `OpFNegate`.
    FNeg {
        form: Form,
        bits: bool,
    },
    IntCompare {
        pred: IntPredicate,
        form: Form,
    },
    BoolCompare {
        pred: IntPredicate,
    },
    FloatCompare {
        pred: FloatPredicate,
        form: Form,
    },
    Cast {
        kind: CastKind,
        from: Form,
        to: Form,
    },
    Bitcast {
        plan: BitcastPlan,
        from: Form,
        to: Form,
    },
    /// `OpSelect` (any lowered representation).
    Select,
    /// The first operand itself (`freeze` of a value, unsigned `abs`).
    Copy,
    /// The zero of the result's representation (`freeze` of `undef`/`poison`).
    Zero(DemoteTy),
    /// `OpUndef` of the result's representation (an element index known out of range).
    Undef(DemoteTy),
    ExtractConst(u32),
    /// `OpVectorExtractDynamic`, its index of form `index` made an integer
    /// ([`ops::element_index`]).
    ExtractDynamic {
        index: Form,
    },
    InsertConst(u32),
    /// `OpVectorInsertDynamic`, its index of form `index` made an integer
    /// ([`ops::element_index`]).
    InsertDynamic {
        index: Form,
    },
    /// Component indices into the concatenation of the two operands.
    Shuffle(Box<[u32]>),
    ExtractValue(Box<[u32]>),
    InsertValue(Box<[u32]>),
    /// `extractvalue` of a 2-4-lane `bool` vector member at `path`, which its aggregate holds as
    /// `u32` lanes of 0 and 1 (`legal::as_member`): the extract, then a comparison with 0.
    ExtractBools {
        path: Box<[u32]>,
        lanes: u8,
    },
    /// `insertvalue` of a 2-4-lane `bool` vector into such a member at `path`: a select of 1 and
    /// 0, then the insert.
    InsertBools {
        path: Box<[u32]>,
        lanes: u8,
    },
    IntFn {
        f: IntFn,
        form: Form,
    },
    /// A float function; `contract` (from the call's fast-math flags) governs the multiply and
    /// add of a sequence such as `mix`.
    FloatFn {
        f: FloatFn,
        form: Form,
        contract: Contraction,
    },
    /// A correctly rounded fused multiply-add on `float` or `half` lanes ([`fma`]): `OpFmaKHR`, a
    /// shared `double` or integer routine, or `half`'s exact `float` sum.
    Fma(fma::Fma),
    /// `air.fast_fmod` on `float`: Metal's `fma(-trunc(x / y), y, x)` of its fast quotient
    /// (`math::fast_fmod`), the fused multiply-add `fma` and the quotient's reciprocal `reciprocal`
    /// ([`divide::Reciprocal::on`]).
    FastFmod {
        fma: fma::Fma,
        reciprocal: divide::Reciprocal,
    },
    /// `air.mix` on `float` or `half`, with or without fast-math flags: Metal's
    /// `fma(t, y, fma(-t, x, x))` (`math::metal_mix`), both fused multiply-adds `fma`; where
    /// `flush`, the inner one's `float` result read as Metal reads a denormal.
    Mix {
        fma: fma::Fma,
        flush: bool,
    },
    /// `air.dot` on `float` or `half` vectors of form `form`: Metal's fused chain of the
    /// products (`math::metal_dot`), each fused multiply-add `fma`, on one lane; where `flush`, the
    /// chain's `float` values before the last read as Metal reads a denormal; where `tiny`, the
    /// first product Metal's where the device's rounding reaches `±2^-126`, by that check
    /// ([`denormal::tiny_product`]).
    Dot {
        fma: fma::Fma,
        form: Form,
        flush: bool,
        tiny: Option<denormal::TinyCheck>,
    },
    /// An `fdiv` carrying `arcp` on `float` lanes: Metal's fast division, `x` times its fast
    /// reciprocal of `y` with that reciprocal's two flushes, answered as Metal answers at
    /// denormals and NaNs (`math::fast_divide`), by a call of the shared `routine`; or, where the
    /// quotient is the device's ([`ArithPlan::device_quotient`]), its `OpFDiv(x, y)`.
    FastDivide {
        form: Form,
        routine: math::FastDivision,
    },
    /// An `fdiv` without `arcp` on `float` lanes, or any `fdiv` on `half` lanes: Metal's
    /// precise division, the correctly rounded quotient, by a call of the shared `routine`
    /// ([`divide`]).
    PreciseDivide {
        form: Form,
        routine: divide::Routine,
    },
    /// Metal's precise division on `float` or `half` lanes by native device maths
    /// (`FloatFunctions::Native`): the device's `OpFDiv`, its divisor kept in Vulkan's range
    /// ([`divide::native`]).
    NativeDivide {
        form: Form,
    },
    /// An `frem` on `float` lanes without every fast-math flag: Metal's `x - y * trunc(x / y)`
    /// of its precise quotient (the shared `routine`'s), the product and the difference each
    /// rounded on its own (`math::metal_frem`); the product read as Metal reads an `f32` denormal
    /// where `flush` (the operands and the result are [`ArithStrategy::Denormals`]').
    Remainder {
        form: Form,
        routine: divide::Routine,
        flush: bool,
    },
    /// A `contract` multiply (or its `fpext` to `float`) fused into every addition that uses it
    /// ([`contract`]), or an addition of a tree Metal regroups folded into a chain: no value of its
    /// own.
    Folded,
    /// A `contract` addition fused with its product ([`contract`]): `fma(a, b, c)` by `fma`'s
    /// route, with the product or the addend negated as `negate` says; in a chain, `c` is the
    /// addend of the last of the `inner` links, whose values are each the addend of the one before,
    /// the first's that of `fma(a, b, _)`.
    Fused {
        fma: fma::Fma,
        a: ValueId,
        b: ValueId,
        c: ValueId,
        negate: contract::Negate,
        /// The `f32` denormals of `a`, `b` and `c` (bits 0 to 2) and of the result the module
        /// flushes itself where `OpFmaKHR` may keep them ([`denormal::Flush`]).
        flush: denormal::Flush,
        /// A chain's fused multiply-adds inside the outer one, outermost first.
        inner: Box<[contract::FusedLink]>,
    },
    Convert {
        kind: ConvertKind,
        from: Form,
        to: Form,
    },
    /// `air.pack`/`air.unpack` of `format` between float lanes of form `floats` and a `u32`.
    Pack {
        dir: PackDir,
        format: PackFormat,
        floats: Form,
    },
    /// A `zext`/`sext` to a 64-bit integer that the access family consumes as a narrow GEP index
    /// (`access::index`): nothing is emitted, and no 64-bit value is formed.
    IndexExtension,
    /// `i64` arithmetic whose only uses are narrow GEP indices, held as its low word
    /// (`access::index`): a `u32` operation, so no 64-bit value is formed.
    IndexWord(access::index::IndexWord),
    /// A lane-wise operation on chunked vectors of `lanes` lanes: `parts` is the operation selected
    /// for each distinct chunk size, `wide` which operands are chunked, and `result` the result's
    /// representation.
    Chunked {
        lanes: u16,
        parts: Box<[(u8, ArithStrategy)]>,
        wide: Box<[bool]>,
        result: DemoteTy,
    },
    /// `extractelement` from a vector of shape `src`: lane `at`, or the lane a dynamic index of
    /// form `index` names.
    LaneExtract {
        src: chunk::Shape,
        at: Option<u32>,
        index: Option<Form>,
    },
    /// `insertelement` into a vector of shape `dst`, as [`ArithStrategy::LaneExtract`].
    LaneInsert {
        dst: chunk::Shape,
        at: Option<u32>,
        index: Option<Form>,
    },
    /// `shufflevector` with a chunked operand or result: result lane `i` is lane `mask[i]` of the
    /// operands' concatenation.
    LaneShuffle {
        a: chunk::Shape,
        b: chunk::Shape,
        out: chunk::Shape,
        mask: Box<[u32]>,
        /// The result is a simdgroup matrix's fragment (`MemoryPlan::is_fragment`): only lanes 0
        /// and 1 are moved, and the others are zero ([`chunk::shuffle_fragment`]).
        fragment: bool,
    },
    /// A bitcast with a chunked side, through 32-bit words.
    WordBitcast {
        from: chunk::Shape,
        to: chunk::Shape,
    },
    /// `air.sincos` on floats of `shape` (chunked per chunk): the sine is the result, and the
    /// cosine is stored through the pointer operand by the access family's store `store`, selected
    /// for a store of the result's type through that pointer.
    SinCos {
        shape: chunk::Shape,
        store: Box<access::AccessStrategy>,
        /// `air.fast_sincos` on `float`: [`FloatFn::FastSin`] and [`FloatFn::FastCos`], which
        /// flush past the last evaluated quadrant as Metal's `fast::sincos` does.
        fast: bool,
        /// Metal's NaN selected for the sine and for the cosine, where Metal answers one the
        /// device may not: the answers of the `sin` and `cos` the call computes, as `with_nan`
        /// selects them for those calls.
        nan: [Option<nan::NanAnswer>; 2],
    },
    /// `air.dot` of chunked vectors of `lanes` lanes of `elem` ([`chunk::dot`]).
    ChunkedDot {
        lanes: u16,
        elem: Form,
        contract: Contraction,
    },
    /// `air.dot` of chunked vectors of `lanes` lanes of `elem` (`float` or `half`) as Metal's fused
    /// chain over every lane ([`chunk::dot_chain`]), each fused multiply-add `fma`; where `flush`,
    /// the chain's `float` values before the last read as Metal reads a denormal, and the lanes of
    /// the operands `reads` names and the result, where it says so; where `tiny`, the first product
    /// Metal's where the device's rounding reaches `±2^-126`, by that check
    /// ([`denormal::tiny_product`]).
    ChunkedChain {
        lanes: u16,
        elem: Form,
        fma: fma::Fma,
        flush: bool,
        reads: denormal::Flush,
        tiny: Option<denormal::TinyCheck>,
    },
    /// A float result with Metal's NaN selected where it is a NaN ([`nan`]): the value of `of`, a
    /// result of `form`, then [`nan::select`] of `answer`. Named as `of`.
    Nan {
        of: Box<ArithStrategy>,
        answer: nan::NanAnswer,
        form: Form,
    },
    /// A `float` operation whose `f32` denormals the device may read otherwise than Metal, which
    /// reads a denormal operand as the zero of its sign and flushes a denormal result
    /// ([`denormal`]): the operands `flush` names, of form `form`, read by
    /// [`denormal::flush`], then `of`, then its result, of form `out`, flushed where `flush`
    /// says. Named as `of`.
    Denormals {
        of: Box<ArithStrategy>,
        form: Form,
        out: Form,
        flush: denormal::Flush,
    },
    /// A `float` multiply `of` (lanes of form `form`) whose device product may round up to
    /// `±2^-126` where Metal flushes the exact product, below it, to the zero of its sign
    /// ([`denormal::tiny_product`]): `of`'s product, then that zero where `check` finds the exact
    /// product below `2^-126`.
    TinyProduct {
        of: Box<ArithStrategy>,
        form: Form,
        check: denormal::TinyCheck,
    },
    /// An operation with a wide integer operand or result on a profile without `shaderInt64`,
    /// on the two words of each value ([`wide`]).
    Wide(wide::WideOp),
    /// An `insertvalue` or `extractvalue` whose result is an aggregate that holds a pointer: no
    /// SPIR-V value ([`aggregate`]).
    PointerAggregate,
    /// An `extractvalue` of a texture or sampler handle with one static candidate: that
    /// candidate's constant `u32` handle index ([`aggregate`]).
    Handle(u32),
    /// An `extractvalue` from an aggregate that holds a pointer, whose member one `insertvalue`
    /// supplied: that inserted value ([`aggregate::forwarded`]).
    Forward(ValueId),
    /// An `extractvalue` of a pointer member that a null or undefined pointer constant, or an
    /// undefined aggregate, supplied: the null pointer in the result's representation
    /// ([`aggregate`]).
    Null(DemoteTy),
    /// An `extractvalue` of member `path` of the member-wise phi `of` ([`aggregate::phi_type`]): a
    /// composite extract from the phi's value.
    Member {
        of: ValueId,
        path: Box<[u32]>,
        /// The lanes of a `bool` vector member, held as `u32` lanes (`legal::as_member`): the
        /// extract is compared with 0.
        bools: Option<u8>,
    },
}

impl FamilyStrategy for ArithStrategy {
    fn name(&self, _: &FamilyPlans) -> Cow<'static, str> {
        use ArithStrategy as S;
        // The NaN select is its operation's: the census names the operation.
        let mut s = self;
        while let S::Nan { of, .. } | S::Denormals { of, .. } = s {
            s = of;
        }
        Cow::Borrowed(match s {
            S::Int { form, .. } | S::IntCompare { form, .. } if form.legalized() => {
                "arith.int.legalized"
            }
            S::Int { .. } | S::IntCompare { .. } => "arith.int",
            S::Bool { .. } | S::BoolCompare { .. } => "arith.bool",
            S::Float { form, .. } if matches!(form.float_fmt(), Some(legal::FloatFmt::Half)) => {
                "arith.float.half"
            }
            S::Float { form, .. } if form.is_narrow() => "arith.float.bfloat",
            S::F16 { .. } => ops::F16,
            S::Float { .. } | S::FNeg { .. } | S::FloatCompare { .. } => "arith.float",
            S::FastDivide { .. } => FAST_DIVIDE,
            S::PreciseDivide { .. } => divide::PRECISE,
            S::NativeDivide { .. } => divide::NATIVE,
            S::Remainder { .. } => REMAINDER,
            S::Cast { .. } => "arith.cast",
            S::Bitcast {
                plan: BitcastPlan::Packed { .. },
                ..
            } => "arith.bitcast.packed",
            S::Bitcast {
                plan: BitcastPlan::Words { .. },
                ..
            } => "arith.bitcast.words",
            S::Bitcast {
                plan: BitcastPlan::Lanes,
                ..
            } => "arith.bitcast.lanes",
            S::Bitcast { .. } => "arith.bitcast",
            S::Select => "arith.select",
            S::Copy | S::Zero(_) | S::Undef(_) => "arith.copy",
            S::ExtractConst(_)
            | S::ExtractDynamic { .. }
            | S::InsertConst(_)
            | S::InsertDynamic { .. } => "arith.element",
            S::Shuffle(_) => "arith.shuffle",
            S::ExtractValue(_)
            | S::InsertValue(_)
            | S::ExtractBools { .. }
            | S::InsertBools { .. } => "arith.aggregate",
            S::IntFn { .. } => "arith.int_fn",
            S::FloatFn { form, .. } if form.is_narrow() => "arith.float_fn.narrow",
            S::FloatFn { f, .. } => match f.precision() {
                FloatPrecision::Exact => "arith.float_fn",
                FloatPrecision::DeviceArithmetic => "arith.float_fn.device_arithmetic",
                FloatPrecision::DeviceTranscendental => "arith.float_fn.device_transcendental",
            },
            S::Fma(f) | S::Fused { fma: f, .. } => f.name(),
            S::FastFmod { .. } => FAST_FMOD,
            S::Mix { .. } => MIX,
            S::Dot { .. } => DOT,
            S::Folded => FOLDED,
            S::Convert {
                kind: ConvertKind::ToInt { .. },
                ..
            } => "arith.convert.saturating",
            S::Convert {
                kind: ConvertKind::FromFp8(_),
                ..
            } => "arith.convert.fp8",
            S::Convert { .. } => "arith.convert",
            // The sRGB conversions are named apart: they read Metal's table.
            S::Pack {
                dir: PackDir::Pack,
                format: PackFormat::Unorm4x8Srgb,
                ..
            } => SRGB_PACK,
            S::Pack {
                format: PackFormat::Unorm4x8Srgb,
                ..
            } => SRGB_UNPACK,
            S::Pack {
                dir: PackDir::Pack, ..
            } => "arith.pack",
            S::Pack { .. } => "arith.unpack",
            S::IndexExtension => "arith.index_extension",
            S::IndexWord(_) => "arith.index_word",
            S::SinCos { .. } => "arith.sincos",
            S::ChunkedDot { .. } => "arith.chunked.dot",
            S::ChunkedChain { .. } => CHUNKED_DOT,
            S::Chunked { .. } => "arith.chunked",
            S::LaneExtract { .. } | S::LaneInsert { .. } => "arith.element.chunked",
            S::LaneShuffle { fragment: true, .. } => "arith.shuffle.fragment",
            S::LaneShuffle { .. } => "arith.shuffle.chunked",
            S::WordBitcast { .. } => "arith.bitcast.chunked",
            S::Wide(w) => w.name(),
            S::PointerAggregate => "arith.aggregate.pointer",
            S::Handle(_) => "arith.aggregate.handle",
            S::Forward(_) => "arith.aggregate.forward",
            S::Null(_) => "arith.aggregate.null",
            S::Member { .. } => "arith.aggregate.member",
            // Named as its operation: unwrapped above.
            S::Nan { .. } => "arith.nan",
            S::Denormals { .. } => "arith.denormals",
            S::TinyProduct { .. } => "arith.float.tiny",
        })
    }

    fn words(&self) -> u32 {
        // Generous per-use bounds: each instruction is at most 6 words plus its share of new
        // constants and types (a vector constant adds its lanes).
        use ArithStrategy as S;
        match self {
            S::IndexExtension | S::PointerAggregate | S::Folded => 0,
            S::Handle(_) | S::Forward(_) => 4,
            S::Member { path, .. } => {
                6u32.saturating_add(u32::try_from(path.len()).unwrap_or(u32::MAX))
            }
            // Two operand conversions (a shift pair or a select each) and the operation.
            S::IndexWord(_) => 32,
            S::Copy => 4,
            S::Zero(d) | S::Undef(d) | S::Null(d) => {
                8u32.saturating_add(8u32.saturating_mul(leaves(d)))
            }
            S::Int { form, .. } | S::IntCompare { form, .. } if form.legalized() => 64,
            S::Int { .. } | S::IntCompare { .. } => 16,
            S::Bool { .. } | S::BoolCompare { .. } | S::FloatCompare { .. } => 32,
            S::Float { form, .. } => 8 + narrow::round_words(*form),
            S::F16 { form, .. } => ops::f16_binary_words(*form),
            // The call; the routine's own words are the module's ([`routine_words`]).
            S::FastDivide { .. } => math::FAST_DIVISION_CALL_WORDS,
            S::PreciseDivide { form, .. } => divide::use_words(*form),
            S::NativeDivide { form } => divide::native_words(*form),
            // The quotient's call, its truncation, the product, the difference, and the product's
            // flush.
            S::Remainder { form, flush, .. } => {
                divide::use_words(*form)
                    + 16
                    + if *flush {
                        denormal::flush_words(*form)
                    } else {
                        0
                    }
            }
            // The bits' two conversions, the operation, and the mask's and its types' share.
            S::FNeg { form, bits: true } => 24 + 4 * u32::from(form.lanes),
            S::FNeg { .. } | S::Select => 8,
            S::Cast { from, to, .. } => 64 + conversion_words(*from, *to),
            S::Bitcast {
                plan: BitcastPlan::Packed { .. } | BitcastPlan::Words { .. } | BitcastPlan::Lanes,
                from,
                to,
            } => (32 * u32::from(from.lanes) + format_bits_words(*from))
                .saturating_add(32 * u32::from(to.lanes) + format_bits_words(*to))
                .saturating_add(32),
            S::Bitcast { .. } => 8,
            S::ExtractConst(_)
            | S::ExtractDynamic { .. }
            | S::InsertConst(_)
            | S::InsertDynamic { .. } => 8,
            S::Shuffle(c) | S::ExtractValue(c) | S::InsertValue(c) => {
                8u32.saturating_add(u32::try_from(c.len()).unwrap_or(u32::MAX))
            }
            // The access, the conversion, and its two (or one) splat constants.
            S::ExtractBools { path, .. } | S::InsertBools { path, .. } => {
                40u32.saturating_add(u32::try_from(path.len()).unwrap_or(u32::MAX))
            }
            S::IntFn { f, form } => f.words(form.bits()),
            S::FloatFn { f, form, .. } => f.words(*form),
            S::Fma(f) => f.words(),
            // The fused multiply-add, a negation, and the flushes, for the pair and each link.
            S::Fused {
                fma, flush, inner, ..
            } => inner
                .iter()
                .fold(fma.words() + 8 + flush.words(fma.form), |w, l| {
                    w.saturating_add(fma.words() + 8 + l.flush.words(fma.form))
                }),
            // The call of the fast division's routine, the truncation, its negation, and the
            // fused multiply-add.
            S::FastFmod { fma, .. } => 64 + fma.words() + math::FAST_DIVISION_CALL_WORDS,
            // The negation of `t` on its bits and the two fused multiply-adds.
            S::Mix { fma, flush } => {
                32 + 2 * fma.words()
                    + if *flush {
                        denormal::flush_words(fma.form)
                    } else {
                        0
                    }
            }
            S::Dot {
                fma,
                form,
                flush,
                tiny,
            } => math::metal_dot_words(*form, fma.words(), (*flush, *tiny)),
            S::Convert {
                kind: ConvertKind::FromFp8(_),
                ..
            } => 320,
            // The value (as `FromFp8`), then the saturating conversion from `f32`.
            S::Convert {
                kind: ConvertKind::Fp8ToInt { .. },
                from,
                to,
            } => 320 + 160 + conversion_words(from.with(Held::Float(FloatFmt::F32)), *to),
            S::Convert { from, to, .. } => 160 + conversion_words(*from, *to),
            S::Pack {
                dir,
                format,
                floats,
            } => {
                // Each sRGB color channel's search or step read; the table is the module's,
                // charged once ([`routine_words`]).
                let srgb = match (*format, *dir) {
                    (PackFormat::Unorm4x8Srgb, PackDir::Pack) => 3 * srgb::SEARCH_WORDS,
                    (PackFormat::Unorm4x8Srgb, PackDir::Unpack) => 3 * srgb::STEP_WORDS,
                    _ => 0,
                };
                // One vector sequence of the exact products or exact divisions
                // (`exact_division::div_exact`) with its per-lane constants, rounded once, and each
                // channel's extract and placement.
                let lanes = u32::from(floats.lanes);
                64 + srgb + 400 + 24 * lanes + narrow::round_words(*floats)
            }
            S::SinCos {
                shape,
                store,
                fast,
                nan,
            } => chunk::sin_cos_words(*shape, *fast, *nan).saturating_add(store.words()),
            S::ChunkedDot { lanes, elem, .. } => chunk::dot_words(*lanes, *elem),
            S::ChunkedChain {
                lanes,
                elem,
                fma,
                flush,
                reads,
                tiny,
            } => chunk::dot_chain_words(*lanes, *elem, fma.words(), (*flush, *tiny), *reads),
            S::Chunked {
                lanes, parts, wide, ..
            } => {
                let per = 4u32.saturating_mul(wide.len() as u32 + 1);
                legal::chunk_lanes(u32::from(*lanes))
                    .into_iter()
                    .map(|k| {
                        parts
                            .iter()
                            .find(|(n, _)| *n == k)
                            .map_or(0, |(_, s)| s.words())
                            .saturating_add(per)
                    })
                    .fold(16u32, u32::saturating_add)
            }
            S::LaneExtract { src, at, .. } => {
                chunk::move_words(*src, chunk::LaneMove::Extract, at.is_none())
            }
            S::LaneInsert { dst, at, .. } => {
                chunk::move_words(*dst, chunk::LaneMove::Insert, at.is_none())
            }
            S::LaneShuffle {
                out,
                fragment: true,
                ..
            } => chunk::fragment_shuffle_words(*out),
            S::LaneShuffle { out, .. } => chunk::move_words(*out, chunk::LaneMove::Shuffle, false),
            S::WordBitcast { from, to } => chunk::bitcast_words(*from, *to),
            S::Wide(w) => w.words(),
            S::Nan { of, answer, form } => of.words().saturating_add(answer.words(*form)),
            S::TinyProduct { of, form, check } => of
                .words()
                .saturating_add(denormal::tiny_product_words(*form, *check)),
            S::Denormals {
                of,
                form,
                out,
                flush,
            } => of.words().saturating_add(
                denormal::Flush {
                    result: false,
                    ..*flush
                }
                .words(*form)
                .saturating_add(if flush.result {
                    denormal::flush_words(*out)
                } else {
                    0
                }),
            ),
        }
    }

    /// An aggregate that holds a pointer has no value.
    fn yields_value(&self) -> bool {
        !matches!(
            self,
            ArithStrategy::PointerAggregate | ArithStrategy::Folded
        )
    }
}

/// The extra words a conversion from `from` to `to` spends on narrow floats: a final rounding (by
/// integer arithmetic from an integer to `half`), a round-to-odd step from `double` or into
/// `bfloat` (twice as long from a 64-bit integer).
fn conversion_words(from: Form, to: Form) -> u32 {
    if !to.is_narrow() {
        return 0;
    }
    if !from.is_float() && to.float_fmt() == Some(FloatFmt::Half) {
        return narrow::round_integer_words(to);
    }
    let odd = if from.float_fmt() == Some(FloatFmt::F64) || !from.is_float() {
        if from.container() == 64 && !from.is_float() {
            192
        } else {
            96
        }
    } else {
        0
    };
    odd + narrow::round_words(to)
}

/// The words `f` spends moving to or from its format bits: a narrow float vector converts once as a
/// vector.
fn format_bits_words(f: Form) -> u32 {
    if f.is_narrow() {
        192
    } else {
        0
    }
}

/// The scalar leaves of `d` (for word bounds).
fn leaves(d: &DemoteTy) -> u32 {
    match d {
        DemoteTy::Vector { len, .. } => u32::from(*len),
        DemoteTy::Composite(parts) => parts.iter().map(leaves).fold(0u32, u32::saturating_add),
        _ => 1,
    }
}

/// The arith family's part of the per-function plan (`select::FamilyPlans`): the entry's
/// floating-point environment.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ArithPlan {
    /// The float container widths whose operations must keep signed zeros, infinities, and NaNs:
    /// those LLVM requires it of (an operation without all of `nnan`, `ninf`, and `nsz`), and
    /// those of a check or a fused multiply-add routine that answers Metal's zero of its sign
    /// ([`ArithPlan::keep_zero_signs`]).
    pub preserve: BTreeSet<u8>,
    /// The widths whose `SignedZeroInfNanPreserve` execution mode the entry declares: those of
    /// `preserve` the profile supports.
    pub modes: BTreeSet<u8>,
    /// Whether 64-bit integers are held as two words (the profile lacks `shaderInt64`), so emission
    /// materializes their constants in that form ([`constant_in`]).
    pub wide: bool,
    /// The shared division routines the wide divisions call, each defined once by [`declare`].
    pub divides: BTreeSet<wide::Divide>,
    /// The shared fused multiply-add routines the plan's `float` fused multiply-adds call, each
    /// defined once by [`declare`].
    pub fmas: BTreeSet<fma::Routine>,
    /// The shared precise division routines the plan's `float` divisions call, each defined
    /// once by [`declare`].
    pub precise: BTreeSet<divide::Routine>,
    /// The shared fast division routines the plan's `arcp` divisions, `fast::tanh`,
    /// `fast::fmod`, and `fast::tan` call, each defined once by [`declare`].
    pub fast: BTreeSet<math::FastDivision>,
    /// The fast division routines the plan's `arcp` divisions call unless their quotient is the
    /// device's own ([`ArithPlan::device_quotient`]), recorded at selection; [`finish`] adds them
    /// to [`ArithPlan::fast`] where it is not.
    pub arcp: BTreeSet<math::FastDivision>,
    /// The value each forwarding `extractvalue` reads ([`aggregate::forwarded`]), and the
    /// member-wise phi each member extraction reads.
    pub forwards: BTreeMap<InstId, Value>,
    /// The composite representation of each member-wise phi: a `phi` of an aggregate that holds
    /// a pointer ([`aggregate::phi_type`]).
    pub phis: BTreeMap<InstId, DemoteTy>,
    /// How each member-wise phi's value on the edge from a block is assembled from its incoming
    /// aggregate's members ([`aggregate::incoming`]).
    pub incoming: BTreeMap<(InstId, BlockId), aggregate::Build>,
    /// The multiply-add pairs Metal fuses ([`contract`]).
    pub contract: contract::Plan,
    /// The `half` additions, subtractions, products and fused multiply-adds a fragment function
    /// returns as an output's lane that Metal computes at `float` precision
    /// ([`output_precision`]): each selected in `float` on its widened operands, unrounded to
    /// `half`.
    pub unrounded: BTreeSet<InstId>,
    /// The direction of the first `air.pack`/`air.unpack` of `unorm4x8_srgb`, when one reads
    /// Metal's sRGB table, which [`declare`] then declares once ([`srgb`]) and the census charges
    /// once, to that direction's strategy ([`routine_words`]).
    pub srgb: Option<PackDir>,
    /// The math-library routines the plan's `float` calls may take ([`libm::uses`]), recorded at
    /// selection; [`finish`] keeps in [`ArithPlan::libm`] those the device's instruction does not
    /// answer.
    pub libm_uses: BTreeSet<libm::Fun>,
    /// The math-library routines the module defines, each once ([`declare`]).
    pub libm: BTreeSet<libm::Fun>,
    /// The mode of the device's own `float` functions where the profile's are Metal's
    /// (`FloatFunctions::Metal`): the mode Metal compiles this entry in ([`libm::module_mode`],
    /// decided by [`finish`]). `None` on any other profile.
    pub metal_functions: Option<libm::Mode>,
    /// Whether the profile takes native device maths (`FloatFunctions::Native`), decided by
    /// [`finish`]: the math library's functions are the device's instructions ([`libm::Lib::native`])
    /// and an `arcp` division's quotient the device's ([`ArithPlan::device_quotient`]).
    pub native: bool,
    /// Whether a selected fused multiply-add takes native device maths' `Fma`
    /// ([`fma::Route::Native`]), whose note [`finish`] records.
    pub native_fma: bool,
    /// Whether the profile rounds `mix` and `fast::sqrt` in the device's order
    /// (`RoundingOrderRoute::Native`), decided by [`finish`]: they are its `FMix` and `Sqrt`
    /// ([`libm::Lib::with_native_order`]).
    pub native_order: bool,
    /// How the module emits a multiply the device must round on its own, decided by [`finish`] from
    /// the same fact as [`ArithPlan::metal_functions`]: a device whose `float` functions are
    /// Metal's reaches Metal through SPIRV-Cross, which loses a `-0` product of a `NoContraction`
    /// `OpFMul`, so it takes `Fma(a, b, -0.0)`; any other device takes the `OpFMul`, which has no
    /// `-0.0` addend for it to answer.
    pub product: ProductForm,
    /// How the module clears the bits above a narrow integer's width in a 32-bit container
    /// ([`ops::mask`]), decided by [`finish`] from the profile's driver workarounds:
    /// `OpBitFieldUExtract` on a driver whose compiler drops the mask after a signed clamp
    /// (`DriverWorkaround::NarrowMaskAfterSignedClamp`), else the mask.
    pub mask: MaskForm,
    /// The profile's `float` fused multiply-add is `OpFmaKHR` ([`fma::float_route`]), decided by
    /// [`finish`]: the trigonometric routines take a product's exact error as one `OpFmaKHR`.
    pub libm_fma: bool,
    /// The precise division routine the math-library routines call (`libm::Helpers`), on one
    /// `float` lane, when one of [`ArithPlan::libm`] divides; decided by [`finish`], which adds it
    /// to [`ArithPlan::precise`].
    pub libm_division: Option<divide::Routine>,
    /// Whether a fragment function writes a float to an sRGB attachment by Metal's encode
    /// ([`color_write`]), which reads the byte's linear value from the color table, and the
    /// buffer the table lives in on the profile, which [`declare`] then declares once
    /// ([`srgb::declare_color_table`]).
    pub color_table: Option<srgb::ColorTable>,
    /// Whether a function reads or writes a texture slot bound in an `_sRGB` extended-range
    /// format (`LoweringInputs::extended_range`), through Metal's tables, which [`declare`] then
    /// declares once ([`extended_range::declare`]).
    pub extended_range_srgb: bool,
    /// The padded `_sRGB` forms whose texels a function reads or writes
    /// ([`ArithPlan::converts`]), through Metal's tables: [`declare`] declares each form's once
    /// ([`extended_range::declare_padded`]).
    pub padded_srgb: BTreeSet<crate::metal::format::ExtendedRange>,
    /// Whether a `half` fused multiply-add takes the device's 16-bit `Fma` ([`fma::Route::F16`]),
    /// recorded at selection: the module then requires `VkProperty::MetalHalfFma`.
    pub f16_fma: bool,
    /// Whether a `float` operation leaves its `f32` denormals to the device because the device
    /// reads them as Metal does ([`denormal::device_flushes`]), recorded at selection: the module
    /// then requires `VkProperty::MetalFloatDenormals`.
    pub device_denormals: bool,
    /// How the module rounds an `f32` to `half` ([`narrow::round`]), decided by [`finish`] from
    /// the profile's measured driver deviations: the device's `OpQuantizeToF16` where it is
    /// measured to round as Metal does (`DriverDeviation::MetalHalfQuantize`; a module that
    /// quantizes then requires `VkProperty::MetalHalfQuantize`); where the device's conversion
    /// to 16 bits is measured to be Metal's and the profile has `shaderFloat16`, the conversion
    /// and its widening (`DriverDeviation::MetalHalfConversion`; a module that rounds so then
    /// requires `VkProperty::MetalHalfConversion`); where neither is measured, under native half
    /// rounding (the default), the same forms as the device's own rounding, noted
    /// (`NativeHalfConversion`, `NativeHalfQuantize`); else integer arithmetic, exact on every
    /// device.
    pub quantize: QuantizeForm,
    /// Whether [`ArithPlan::quantize`] is the device's own rounding (native half rounding,
    /// [`narrow::native_rounding`]), decided with it by [`finish`].
    pub native_half: bool,
    /// Whether a `half` texel keeps the NaN the device decoded under that native rounding too
    /// (`narrow::round_texel`): where the profile's NaN is the translation's to answer as Metal's
    /// (`NanEncoding::Device`, the policy's `NanMinMaxPolicy::Exact`), decided by [`finish`].
    pub texel_nans: bool,
    /// Whether a `half`'s NaN bits are left to the device's conversion where the module rounds
    /// through the conversion under the float controls ([`QuantizeForm::Controlled`]): a `half`
    /// read from its bits is the device's widening of them (`narrow::from_bits`), and a `half`
    /// texel takes the rounding's NaN (`narrow::round_texel`). Where the profile's NaN is the
    /// device's own (`NanEncoding::Native`, the policy's `NanMinMaxPolicy::Native`, the default),
    /// decided by [`finish`].
    pub native_half_nans: bool,
    /// Whether Metal's `fmin`, `fmax`, `min3`, `max3`, `clamp` or `saturate` on `float` or `half`
    /// takes the device's `NMin`, `NMax` or `NClamp`, measured to be Metal's
    /// (`DriverDeviation::MetalMinMax`), recorded at selection: the module then requires
    /// `VkProperty::MetalMinMax`, and every such call in it takes those instructions.
    pub device_min_max: bool,
    /// Whether those functions take the device's `NMin`, `NMax` or `NClamp` as native device
    /// maths ([`native_min_max`]), recorded at selection: the module records `NativeMinMax` and
    /// requires no property.
    pub native_min_max: bool,
    /// Whether a `half` `+`, `-` or `*` takes the device's 16-bit arithmetic
    /// ([`ArithStrategy::F16`]), measured to be Metal's (`DriverDeviation::MetalHalfArithmetic`),
    /// recorded at selection: the module then requires `VkProperty::MetalHalfArithmetic`.
    pub f16_arith: bool,
    /// The precision notes the plan's operations record unless their entry's quotient and
    /// functions are Metal's own ([`ArithPlan::device_quotient`]), recorded at selection
    /// ([`OpNotes::unless_metal_fast`]); [`finish`] records them where they are not.
    pub unless_metal_fast: BTreeSet<PrecisionNote>,
}

impl ArithPlan {
    /// Record that a function reads or writes the texels of a slot or attachment bound in
    /// `form`, which then convert through Metal's tables: an `_sRGB` extended-range form's
    /// ([`ArithPlan::extended_range_srgb`]) or a padded `_sRGB` form's
    /// ([`ArithPlan::padded_srgb`]).
    pub fn converts(&mut self, form: crate::metal::format::ExtendedRange) {
        self.extended_range_srgb |= form.srgb();
        if form.padded_srgb() {
            self.padded_srgb.insert(form);
        }
    }

    /// Record fused multiply-add `fma`, selected: its routine, which [`declare`] defines once
    /// (and whose widths [`finish`] keeps zero signs of), or the device's 16-bit `Fma`
    /// ([`ArithPlan::f16_fma`]).
    pub fn take_fma(&mut self, fma: fma::Fma) {
        if let Some(r) = fma.routine() {
            self.fmas.insert(r);
        }
        self.f16_fma |= fma.route == fma::Route::F16;
        self.native_fma |= fma.route == fma::Route::Native;
    }

    /// Record a check of a `float` product at `±2^-126` on lanes of form `f`
    /// ([`denormal::tiny_product`]), selected where `check` is one: it answers Metal's zero of
    /// the product's sign, so its width keeps zero signs ([`ArithPlan::keep_zero_signs`]).
    pub fn take_tiny_check(
        &mut self,
        check: Option<denormal::TinyCheck>,
        f: Form,
    ) -> Option<denormal::TinyCheck> {
        if check.is_some() {
            self.keep_zero_signs(f.container());
        }
        check
    }

    /// Record that a check or routine computed by the translation answers Metal's zero of its
    /// sign on floats of container width `width`: a fused multiply-add routine where the profile's
    /// functions answer as Metal's ([`fma::Routine::widths`]: in `double`, in integers, a `half`
    /// one, or a checked `OpFmaKHR`'s small results), a product flushed at `±2^-126`
    /// ([`denormal::tiny_product`]). Vulkan lets a device take a zero of either sign from any
    /// float operation of an entry without `SignedZeroInfNanPreserve` at its width (it assumes
    /// `NSZ` there: the Vulkan specification's "Precision and Operation of SPIR-V Instructions"),
    /// whatever the AIR's flags: NVIDIA's 615.71.09 on the RTX 3090 answered `-0` for a checked
    /// `fma(+0, 100, +0)` whose answer is `+0`, in a fast-math entry. So the width joins
    /// [`ArithPlan::preserve`], which [`finish`] declares where the profile supports it, else
    /// notes `PrecisionNote::DeviceSignedZeroInfNan`.
    pub fn keep_zero_signs(&mut self, width: u8) {
        self.preserve.insert(width);
    }

    /// Whether an `arcp` division's quotient is the device's `OpFDiv(x, y)`: where the profile's
    /// float functions and division are Metal's and Metal compiles the entry with fast math
    /// ([`ArithPlan::metal_functions`] is [`libm::Mode::Fast`]: the entry declares no
    /// `SignedZeroInfNanPreserve`). That instruction reaches Metal's compiler as the AIR's own fast
    /// `x / y`, which it compiles as it compiles Metal's, folds included. Where the entry declares
    /// the mode, Metal compiles `x / y` as the correctly rounded quotient, not as its fast
    /// division, so the quotient stays `x` times the device's `OpFDiv(1, y)`. The same fact decides
    /// which operations are Metal's own and record no note ([`OpNotes`]).
    pub fn device_quotient(&self) -> bool {
        // Native device maths takes the device's quotient too.
        metal_fast(self.metal_functions) || self.native
    }
}

/// The words of the module-scope routines the plan's strategies call, each with the census name
/// of the strategies that call it: counted once per module, not per use. The fast division's
/// routine is charged to [`FAST_DIVIDE`], whether an `arcp` division, `fast::tanh`, `fast::fmod`,
/// or `fast::tan` calls it.
pub fn routine_words(plan: &ArithPlan) -> impl Iterator<Item = (&'static str, u64)> + '_ {
    let table = plan.srgb.map(|dir| {
        let name = match dir {
            PackDir::Pack => SRGB_PACK,
            PackDir::Unpack => SRGB_UNPACK,
        };
        (name, u64::from(srgb::TABLE_WORDS))
    });
    // The extended-range sRGB table, once, and each padded `_sRGB` form's.
    let extended = plan.extended_range_srgb.then_some((
        extended_range::CENSUS_NAME,
        u64::from(extended_range::TABLE_WORDS),
    ));
    let padded = plan.padded_srgb.iter().map(|_| {
        (
            extended_range::PADDED_CENSUS_NAME,
            u64::from(extended_range::PADDED_TABLE_WORDS),
        )
    });
    plan.divides
        .iter()
        .map(|d| (wide::DIVIDE, u64::from(d.words())))
        .chain(plan.fmas.iter().map(|r| {
            (
                r.name(),
                u64::from(r.words_among(&plan.fmas, plan.quantize)),
            )
        }))
        .chain(
            plan.precise
                .iter()
                .map(|r| (divide::PRECISE, u64::from(r.words()))),
        )
        .chain(
            plan.fast
                .iter()
                .map(|r| (FAST_DIVIDE, u64::from(r.words()))),
        )
        .chain(plan.libm.iter().map(|r| (r.name(), u64::from(r.words()))))
        .chain(table)
        .chain(
            plan.color_table
                .map(|_| (COLOR_WRITE_SRGB, u64::from(srgb::COLOR_TABLE_WORDS))),
        )
        .chain(extended)
        .chain(padded)
}

/// The census names of the sRGB conversions, apart from the other normalized formats: they read
/// Metal's table, which the module declares once.
pub const SRGB_PACK: &str = "arith.pack.srgb";
/// The census name of `air.fast_fmod` on `float` ([`ArithStrategy::FastFmod`]).
pub const FAST_FMOD: &str = "arith.fmod.fast";
/// The census name of an `frem` on `float` without every fast-math flag
/// ([`ArithStrategy::Remainder`]).
pub const REMAINDER: &str = "arith.frem";
/// The census name of `air.mix` on `float` and `half` ([`ArithStrategy::Mix`]).
pub const MIX: &str = "arith.mix.fused";
/// The census name of `air.dot` on `float` and `half` as Metal's fused chain
/// ([`ArithStrategy::Dot`]).
pub const DOT: &str = "arith.dot.fused";
/// The census name of `air.dot` on chunked `float` and `half` vectors as Metal's fused chain
/// ([`ArithStrategy::ChunkedChain`]).
pub const CHUNKED_DOT: &str = "arith.chunked.dot.fused";
/// See [`SRGB_PACK`].
pub const SRGB_UNPACK: &str = "arith.unpack.srgb";
/// The census name of the color table a fragment function's sRGB color writes read
/// ([`ArithPlan::color_table`]).
pub const COLOR_WRITE_SRGB: &str = "arith.color_write.srgb";
/// The census name of Metal's fast division ([`ArithStrategy::FastDivide`]).
pub const FAST_DIVIDE: &str = "arith.fdiv.fast";
/// The census name of a multiply fused into the additions that use it
/// ([`ArithStrategy::Folded`]).
pub const FOLDED: &str = "arith.float.folded";

/// Complete the arith family's plan once every instruction and parameter is selected: declare
/// `SignedZeroInfNanPreserve` for each required width the profile supports (with those its fused
/// multiply-add routines compute in where its functions answer as Metal's), and note
/// `DeviceSignedZeroInfNan` for one it does not (the device's behavior is accepted and
/// reported); then, with the entry's execution modes known, keep the math-library routines whose
/// function the device's instruction does not answer: on a profile whose `float` functions are
/// Metal's, those of a call whose mode is not the one Metal compiles the entry in
/// ([`libm::module_mode`]); on any other profile, every one. Likewise keep the `arcp` divisions'
/// fast division routines unless their quotient is the device's ([`ArithPlan::device_quotient`]),
/// record the notes of the operations that are Metal's own in such an entry only where it is not
/// ([`ArithPlan::unless_metal_fast`]), and choose the form of a multiply the device must round on
/// its own by the same fact ([`ArithPlan::product`]), the form of a narrow integer's mask by the
/// driver's workarounds ([`ArithPlan::mask`]), whether the trigonometric routines take `OpFmaKHR`
/// ([`ArithPlan::libm_fma`]), and which fused multiply-add routines are defined per lane
/// ([`fma::complete`]).
pub fn finish(cx: &mut SelectCx<'_>) -> Result<(), Error> {
    cx.families.arith.wide = legal::holds_wide(cx.profile);
    cx.families.arith.quantize = narrow::quantize_form(cx.profile);
    cx.families.arith.native_half =
        narrow::native_rounding(cx.profile, cx.families.arith.quantize).is_some();
    // Where the NaN switch asks for Metal's NaN, a texel's NaN is its own through that rounding.
    cx.families.arith.texel_nans = nan::encoding(cx.profile) == NanEncoding::Device;
    // Under the float controls a `half`'s NaN is the device's conversion's where the profile
    // leaves NaN bits to the device.
    cx.families.arith.native_half_nans = cx.families.arith.quantize == QuantizeForm::Controlled
        && nan::encoding(cx.profile) == NanEncoding::Native;
    cx.families.arith.mask = if cx
        .profile
        .fields()
        .workarounds
        .contains(&DriverWorkaround::NarrowMaskAfterSignedClamp)
    {
        MaskForm::Extract
    } else {
        MaskForm::And
    };
    // Where the profile's functions answer as Metal's (`FloatMathsPolicy::Exact`), a fused
    // multiply-add routine answers Metal's zero of its sign: every width it computes in keeps
    // zero signs. Every family selected its fused multiply-adds by now.
    if cx.profile.fields().float_functions == FloatFunctions::Device {
        let arith = &mut cx.families.arith;
        let kept: BTreeSet<u8> = arith
            .fmas
            .iter()
            .flat_map(|r| r.widths())
            .copied()
            .collect();
        for w in kept {
            arith.keep_zero_signs(w);
        }
    }
    let fc = cx.profile.fields().float_controls;
    let widths: Vec<u8> = cx.families.arith.preserve.iter().copied().collect();
    for w in widths {
        let supported = match w {
            32 => fc.f32.signed_zero_inf_nan_preserve,
            64 => fc.f64.signed_zero_inf_nan_preserve,
            _ => false,
        };
        if supported {
            cx.families.arith.modes.insert(w);
        } else {
            cx.precision.insert(PrecisionNote::DeviceSignedZeroInfNan);
        }
    }
    cx.families.arith.libm_fma = fma::float_route(cx.profile) == fma::Route::Khr;
    let arith = &mut cx.families.arith;
    // Every family selected its fused multiply-adds by now (a cube position's at selection).
    fma::complete(&mut arith.fmas, arith.quantize);
    // The measured device fact, never the driver's name.
    arith.metal_functions = match cx.profile.fields().float_functions {
        FloatFunctions::Metal => Some(libm::module_mode(&arith.modes)),
        _ => None,
    };
    arith.native = cx.profile.fields().float_functions == FloatFunctions::Native;
    arith.native_order = cx.profile.fields().rounding_order == RoundingOrderRoute::Native;
    let native = arith.native;
    if arith.native_fma {
        cx.precision.insert(PrecisionNote::NativeFma);
    }
    let metal = arith.metal_functions;
    // The same fact decides the multiply's form.
    arith.product = match metal {
        Some(_) => ProductForm::NegativeZeroFma,
        None => ProductForm::Multiply,
    };
    arith.libm = arith
        .libm_uses
        .iter()
        .copied()
        .filter(|r| !metal.is_some_and(|m| r.device_answers(m)) && !(native && r.native()))
        .collect();
    // The routines' division is the module's precise one (`libm::Helpers`): the precise division
    // routine on one lane; and Metal's reciprocal's routine, its fast `rsqrt`'s, the precise
    // `rsqrt`'s (which the fast one calls), and the fused step's where a routine calls them.
    let needs = arith
        .libm
        .iter()
        .fold(libm::Needs::default(), |n, r| n.or(r.needs()));
    if needs.fast_inverse_root {
        arith.libm.insert(libm::Fun::FastRsqrt);
    }
    // The emulated fused step where the profile has no `OpFmaKHR`: one routine the others call.
    if needs.fused && !arith.libm_fma {
        arith.libm.insert(libm::Fun::Fused);
    }
    let needs = arith.libm.iter().fold(needs, |n, r| n.or(r.needs()));
    if needs.recip {
        arith.libm.insert(libm::Fun::Recip);
    }
    if needs.inverse_root {
        arith.libm.insert(libm::Fun::Rsqrt);
    }
    let needs = arith.libm.iter().fold(needs, |n, r| n.or(r.needs()));
    arith.libm_division = needs.division.then(|| divide::Routine {
        lanes: 1,
        path: divide::Path::on(cx.profile),
    });
    arith.precise.extend(arith.libm_division);
    // An `arcp` division calls the fast division's routine unless its quotient is the
    // device's, and an operation that is Metal's own in such an entry records its notes only
    // outside one.
    if !arith.device_quotient() {
        arith.fast.extend(arith.arcp.iter().copied());
    }
    if !metal_fast(arith.metal_functions) {
        cx.precision.extend(arith.unless_metal_fast.iter().copied());
    }
    Ok(())
}

/// The arith family's module-scope objects: the shared wide division routines (the
/// `GLSL.std.450` import is the serializer's, and constants are interned where they are used).
#[derive(Clone, Debug, Default)]
pub struct Declared {
    /// The function of each of the plan's [`ArithPlan::divides`].
    pub divides: BTreeMap<wide::Divide, FuncId>,
    /// The function of each of the plan's [`ArithPlan::fmas`].
    pub fmas: BTreeMap<fma::Routine, FuncId>,
    /// The function of each of the plan's [`ArithPlan::precise`].
    pub precise: BTreeMap<divide::Routine, FuncId>,
    /// The function of each of the plan's [`ArithPlan::fast`].
    pub fast: BTreeMap<math::FastDivision, FuncId>,
    /// The sRGB table, when [`ArithPlan::srgb`].
    pub srgb: Option<GlobalId>,
    /// The color table, when [`ArithPlan::color_table`].
    pub color_table: Option<GlobalId>,
    /// The extended-range sRGB table, when [`ArithPlan::extended_range_srgb`].
    pub extended_range_srgb: Option<GlobalId>,
    /// The table of each of the plan's [`ArithPlan::padded_srgb`].
    pub padded_srgb: BTreeMap<crate::metal::format::ExtendedRange, GlobalId>,
    /// The function of each of the plan's [`ArithPlan::libm`].
    pub libm: BTreeMap<libm::Fun, FuncId>,
}

/// Declare the arith family's module-scope objects and entry execution modes: one function per
/// shared division routine the plan calls, and one `SignedZeroInfNanPreserve` per width the plan
/// selected.
pub fn declare(
    mb: &mut ModuleBuilder,
    plan: &Plan,
    modes: &mut Vec<ExecMode>,
) -> BResult<Declared> {
    for w in &plan.families.arith.modes {
        modes.push(ExecMode::SignedZeroInfNanPreserve { width: *w });
    }
    let mut decl = Declared::default();
    for d in &plan.families.arith.divides {
        decl.divides.insert(*d, wide::define_divide(mb, *d)?);
    }
    let fmas = &plan.families.arith.fmas;
    for r in fmas {
        let id = fma::define_among(mb, *r, fmas, &decl.fmas)?;
        decl.fmas.insert(*r, id);
    }
    for r in &plan.families.arith.precise {
        decl.precise.insert(*r, divide::define(mb, *r)?);
    }
    for r in &plan.families.arith.fast {
        decl.fast.insert(*r, math::define_fast_division(mb, *r)?);
    }
    if plan.families.arith.srgb.is_some() {
        decl.srgb = Some(srgb::declare(mb)?);
    }
    if let Some(class) = plan.families.arith.color_table {
        decl.color_table = Some(srgb::declare_color_table(mb, plan.stage, class)?);
    }
    if plan.families.arith.extended_range_srgb {
        decl.extended_range_srgb = Some(extended_range::declare(mb)?);
    }
    for &form in &plan.families.arith.padded_srgb {
        if let Some(table) = extended_range::declare_padded(mb, form)? {
            decl.padded_srgb.insert(form, table);
        }
    }
    let arith = &plan.families.arith;
    let mut helpers = libm::Helpers {
        division: match arith.libm_division {
            Some(r) => Some(
                *decl
                    .precise
                    .get(&r)
                    .ok_or(BuildError::Type(TypeRule::FunctionSignature))?,
            ),
            None => None,
        },
        exp2: None,
        log2: None,
        rcp: None,
        recip: None,
        inverse_root: None,
        rsqrt: None,
        fast_inverse_root: None,
        fused: None,
        quarter: None,
        sinc: None,
        powr: None,
    };
    if arith.libm.iter().any(|r| r.needs().exp2) {
        helpers.exp2 = Some(libm::exp::declare(mb)?);
    }
    if arith.libm.iter().any(|r| r.needs().log2) {
        helpers.log2 = Some(libm::log::declare(mb)?);
    }
    if arith.libm.iter().any(|r| r.needs().rcp) {
        helpers.rcp = Some(libm::rcp::declare(mb)?);
    }
    if arith.libm.iter().any(|r| r.needs().rsqrt) {
        helpers.rsqrt = Some(libm::rsqrt::declare(mb)?);
    }
    if arith.libm.iter().any(|r| r.needs().quarter) {
        helpers.quarter = Some(libm::quarter::declare(mb)?);
    }
    if arith.libm.iter().any(|r| r.needs().sinc) {
        helpers.sinc = Some(libm::sincos::declare(mb)?);
    }
    if arith.libm.iter().any(|r| r.needs().powr) {
        helpers.powr = Some(libm::log::declare_powr(mb)?);
    }
    // The fused step's first: the routines with fused steps call it.
    if arith.libm.contains(&libm::Fun::Fused) {
        let id = libm::define(mb, libm::Fun::Fused, arith.libm_fma, helpers)?;
        decl.libm.insert(libm::Fun::Fused, id);
        helpers.fused = Some(id);
    }
    // Metal's reciprocal's next: `sinh`, `cosh`, and `tanh` call it.
    if arith.libm.contains(&libm::Fun::Recip) {
        let id = libm::define(mb, libm::Fun::Recip, arith.libm_fma, helpers)?;
        decl.libm.insert(libm::Fun::Recip, id);
        helpers.recip = Some(id);
    }
    // The precise `rsqrt`'s next: Metal's fast one calls it.
    if arith.libm.contains(&libm::Fun::Rsqrt) {
        let id = libm::define(mb, libm::Fun::Rsqrt, arith.libm_fma, helpers)?;
        decl.libm.insert(libm::Fun::Rsqrt, id);
        helpers.inverse_root = Some(id);
    }
    // Metal's fast `rsqrt`'s next: `asin`, `acos`, and both modes' `asinh` and `acosh` call it.
    if arith.libm.contains(&libm::Fun::FastRsqrt) {
        let id = libm::define(mb, libm::Fun::FastRsqrt, arith.libm_fma, helpers)?;
        decl.libm.insert(libm::Fun::FastRsqrt, id);
        helpers.fast_inverse_root = Some(id);
    }
    for r in arith.libm.iter().filter(|r| {
        !matches!(
            r,
            libm::Fun::Fused | libm::Fun::Recip | libm::Fun::Rsqrt | libm::Fun::FastRsqrt
        )
    }) {
        decl.libm
            .insert(*r, libm::define(mb, *r, arith.libm_fma, helpers)?);
    }
    Ok(decl)
}

/// Record that a float operation on form `f` with fast-math flags `fmf` needs signed zeros,
/// infinities, and NaNs preserved at `f`'s container width unless it carries all of `nnan`,
/// `ninf`, and `nsz` (LLVM semantics; a Vulkan device may otherwise ignore them). The simd
/// family's float calls record here too: the simdgroup and quad sums, prefix sums, minima, and
/// maxima, and the matrix and AGX multiply-adds, whose steps are `f32` `Fma`s.
pub(crate) fn fp_env(cx: &mut SelectCx<'_>, fmf: FastMath, f: Form) {
    let relaxed = FastMath::NNAN.union(FastMath::NINF).union(FastMath::NSZ);
    if f.is_float() && !fmf.contains(relaxed) {
        cx.families.arith.preserve.insert(f.container());
    }
}

// ---- Selection ----

/// The AIR type of `v`.
fn ty_of(cx: &SelectCx<'_>, v: Value) -> Result<AirTypeId, Error> {
    cx.value_ty(v)
        .ok_or_else(|| cx.invariant("an arith operand without a type"))
}

/// The form of `v` on the profile.
fn form(cx: &SelectCx<'_>, v: Value) -> Result<Form, Error> {
    form_on(cx.entry, cx.profile, ty_of(cx, v)?)
}

/// The form of the instruction's result.
fn result_form(cx: &SelectCx<'_>, inst: &Inst) -> Result<Form, Error> {
    form_on(cx.entry, cx.profile, inst.ty)
}

/// Every operand must be a value emission produces in its lowered representation: a constant
/// this family materializes, or a value of a type lowered on the profile (instruction results
/// are selected on their own; parameters by their family).
fn check_operands(cx: &SelectCx<'_>, operands: &[Value]) -> Result<(), Error> {
    for v in operands {
        if matches!(v, Value::Const(_)) && cx.constant(*v).is_none() {
            return Err(not_lowered());
        }
        cx.lowered_repr(*v, ty_of(cx, *v)?)?;
    }
    Ok(())
}

/// The value operands of a non-call instruction, in order.
fn operands_of(kind: &InstKind) -> Vec<Value> {
    let mut out = Vec::new();
    kind.for_each_value(|v| out.push(v));
    out
}

/// The integer constant `v`, if it is one.
fn const_int(cx: &SelectCx<'_>, v: Value) -> Option<u64> {
    let Value::Const(c) = v else { return None };
    match cx.entry.module.consts().get(c)?.value {
        Const::Int(b) => Some(b),
        _ => None,
    }
}

/// The lane count of vector value `v`.
fn lanes_of(cx: &SelectCx<'_>, v: Value) -> Result<u32, Error> {
    let t = ty_of(cx, v)?;
    match cx.entry.module.types().get(t) {
        Some(Type::Vector { len, .. }) => Ok(*len),
        _ => Err(cx.invariant("a vector operation on a non-vector")),
    }
}

/// The SPIR-V contraction of an operation with fast-math flags `fmf`. An undecorated SPIR-V
/// operation lets the device both fuse it and regroup it with its neighbours (`NoContraction`
/// forbids both), and MoltenVK's fast-math Metal compile does regroup. LLVM grants the two
/// separately, `contract` and `reassoc`, so only an operation that carries both stays undecorated;
/// any other is `NoContraction`, which keeps the rounding of each operation, one of the values its
/// flags allow.
fn contraction(fmf: FastMath) -> Contraction {
    if fmf.contains(FastMath::CONTRACT.union(FastMath::REASSOC)) {
        Contraction::Allowed
    } else {
        Contraction::Forbidden
    }
}

/// Note `PrecisionNote::FastMathRewrites` when float arithmetic `op` with flags `fmf` on
/// operands `a` and `b` may be regrouped with the operation defining one of them, as LLVM's
/// flags allow: `op` and the operand are additions, subtractions, or multiplies that both carry
/// `reassoc`. A negation between them is looked through: it is exact, and a compiler folds it into
/// either operation. So are lane selections, insertions and extractions: Metal's compiler splits
/// vectors into lanes before it regroups. Arithmetic that chains only through memory is not seen
/// here. A multiply's fusion into an addition (both carrying `contract`) is Metal's rule, which
/// [`contract`] follows and notes where it is not measured, and a multiply `at` whose lane-uniform
/// factors are paired is Metal's grouping ([`regroup::paired`]).
fn note_rewrites(
    cx: &mut SelectCx<'_>,
    at: Option<InstId>,
    (op, fmf): (AirBin, FastMath),
    a: Value,
    b: Value,
) {
    if !matches!(op, AirBin::FAdd | AirBin::FSub | AirBin::FMul) {
        return;
    }
    let rewritable = [a, b].into_iter().any(|v| {
        float_arith_def(cx, v, LOOK_THROUGH).is_some_and(|(_, def_fmf)| {
            fmf.contains(FastMath::REASSOC) && def_fmf.contains(FastMath::REASSOC)
        })
    });
    let ir = regroup::Ir {
        module: &cx.entry.module,
        func: cx.entry.func,
        body: cx.body,
    };
    if rewritable && !(op == AirBin::FMul && at.is_some_and(|at| regroup::paired(ir, at))) {
        cx.precision.insert(PrecisionNote::FastMathRewrites);
    }
}

/// How many negations, lane selections, insertions and extractions [`float_arith_def`] looks
/// through: a bound on its walk.
const LOOK_THROUGH: u32 = 8;

/// The float addition, subtraction, or multiply defining `v`, through negations and lane
/// selections, insertions and extractions, at most `depth` of them, with its fast-math flags: of
/// a selection of two vectors' lanes or an insertion, the first such operation either source
/// reaches.
fn float_arith_def(cx: &SelectCx<'_>, v: Value, depth: u32) -> Option<(AirBin, FastMath)> {
    let Value::Inst(id) = v else {
        return None;
    };
    let kind = &cx.body.insts.get(id)?.kind;
    if let InstKind::BinOp {
        op: op @ (AirBin::FAdd | AirBin::FSub | AirBin::FMul),
        fmf,
        ..
    } = kind
    {
        return Some((*op, *fmf));
    }
    let depth = depth.checked_sub(1)?;
    match kind {
        InstKind::FNeg { v: negated, .. } => float_arith_def(cx, *negated, depth),
        InstKind::ShuffleVector { a, b, .. } => {
            float_arith_def(cx, *a, depth).or_else(|| float_arith_def(cx, *b, depth))
        }
        InstKind::InsertElement { vec, elt, .. } => {
            float_arith_def(cx, *elt, depth).or_else(|| float_arith_def(cx, *vec, depth))
        }
        InstKind::ExtractElement { vec, .. } => float_arith_def(cx, *vec, depth),
        _ => None,
    }
}

fn note(cx: &mut SelectCx<'_>, p: FloatPrecision) {
    cx.precision.extend(precision_note(p));
}

/// The note that records precision class `p`, if it records one.
fn precision_note(p: FloatPrecision) -> Option<PrecisionNote> {
    match p {
        FloatPrecision::Exact => None,
        FloatPrecision::DeviceArithmetic => Some(PrecisionNote::DeviceArithmetic),
        FloatPrecision::DeviceTranscendental => Some(PrecisionNote::DeviceTranscendentals),
    }
}

/// Record that `float` operations on form `f` are left to the device's denormal handling (with
/// `denormals`) and its NaN encoding (with `nans`): Metal flushes `f32` denormal operands and
/// results and answers `0x7fc00000` for a NaN result, which no Vulkan device is required to do.
/// Other formats record nothing: a `half` is held widened (never an `f32` denormal) and rounded
/// by this translation, and Metal's `double` and `bfloat` environments are not measured.
fn device_float(cx: &mut SelectCx<'_>, f: Form, denormals: bool, nans: bool) {
    cx.precision.extend(device_float_notes(f, denormals, nans));
}

/// The notes [`device_float`] records.
fn device_float_notes(f: Form, denormals: bool, nans: bool) -> impl Iterator<Item = PrecisionNote> {
    let single = f.float_fmt() == Some(FloatFmt::F32);
    [
        (denormals, PrecisionNote::DeviceDenormals),
        (nans, PrecisionNote::DeviceNanEncoding),
    ]
    .into_iter()
    .filter_map(move |(left, note)| (single && left).then_some(note))
}

/// The precision notes a float operation's lowering records for the operation itself, by where they
/// hold: `always` in every entry, and `unless_metal_fast` in every entry but one whose quotient and
/// functions are Metal's own ([`ArithPlan::device_quotient`]: the profile's `float` functions are
/// Metal's, and Metal compiles the entry with fast math), where the operation is Metal's own and
/// its notes name no difference from Metal. Selection records the first ([`record`]) and holds the
/// second in [`ArithPlan::unless_metal_fast`] until [`finish`] knows the entry's math mode.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OpNotes {
    /// The notes the operation records in every entry.
    pub always: BTreeSet<PrecisionNote>,
    /// The notes it records unless the entry is Metal's fast one.
    pub unless_metal_fast: BTreeSet<PrecisionNote>,
}

impl OpNotes {
    /// `notes`, held in every entry.
    fn everywhere(notes: BTreeSet<PrecisionNote>) -> OpNotes {
        OpNotes {
            always: notes,
            unless_metal_fast: BTreeSet::new(),
        }
    }

    /// `notes`, held in every entry but Metal's fast one where `own`, else in every entry.
    fn unless_own(notes: BTreeSet<PrecisionNote>, own: bool) -> OpNotes {
        if own {
            OpNotes {
                always: BTreeSet::new(),
                unless_metal_fast: notes,
            }
        } else {
            OpNotes::everywhere(notes)
        }
    }

    /// Both sets of `other` added to this one's.
    fn with(mut self, other: OpNotes) -> OpNotes {
        self.always.extend(other.always);
        self.unless_metal_fast.extend(other.unless_metal_fast);
        self
    }

    /// The notes the operation records in an entry whose `float` functions are Metal's in mode
    /// `metal` (`None` on a profile whose functions are the device's), as [`finish`] decides
    /// them.
    pub fn in_entry(self, metal: Option<libm::Mode>) -> BTreeSet<PrecisionNote> {
        let mut notes = self.always;
        if !metal_fast(metal) {
            notes.extend(self.unless_metal_fast);
        }
        notes
    }
}

/// Whether an entry whose `float` functions are Metal's in mode `metal` (`None`: the device's) is
/// one Metal compiles with fast math: its `arcp` quotient and its functions are then Metal's own
/// ([`ArithPlan::device_quotient`], [`OpNotes`]).
fn metal_fast(metal: Option<libm::Mode>) -> bool {
    metal == Some(libm::Mode::Fast)
}

/// Record `notes`, an operation's: those that hold in every entry now, the others once [`finish`]
/// knows whether the entry is Metal's fast one.
fn record(cx: &mut SelectCx<'_>, notes: OpNotes) {
    cx.precision.extend(notes.always);
    cx.families
        .arith
        .unless_metal_fast
        .extend(notes.unless_metal_fast);
}

/// The precision notes the lowering of the float binary operator `op` with fast-math flags `fmf`
/// on operands of format `fmt` records for the operator itself on profile `p`: a division or
/// remainder is the device's arithmetic, and a `float` operator leaves denormals to the device
/// (Metal flushes `f32` denormals); a `float` remainder also leaves the NaN encoding to it, where
/// `+ - *` answer Metal's `0x7fc00000`. Metal's precise division ([`divide::precise`]: an `fdiv` on
/// `float` without `arcp`, or on `half`) records none: a shared routine computes Metal's quotient
/// apart from the device's division and its environment. Metal's fast division (an `fdiv` on
/// `float` with `arcp`) selects Metal's answers at denormals and NaNs itself (`math::fast_divide`),
/// so its reciprocal alone is the device's arithmetic; where the profile's `float` functions are
/// Metal's, the quotient of an entry Metal compiles with fast math is the device's `OpFDiv(x, y)`,
/// Metal's own fast division ([`ArithPlan::device_quotient`]), and the note holds only outside one.
/// Beside these, an addition, subtraction, or multiply whose flags let the device fuse or regroup
/// it with an operand's definition records `FastMathRewrites`, which the operands decide
/// (`note_rewrites`). `None` for an integer operator.
///
/// [`select`] records exactly these for such an operator (a `half` result's rounding is noted from
/// the emitted module); validation reads them to tell which operations the product computes as a
/// near-equivalent of Metal's, as [`PrecisionNote::of_float_operator`].
pub fn float_binop_notes(
    op: AirBin,
    fmt: FloatFmt,
    fmf: FastMath,
    p: &DeviceProfile,
) -> Option<OpNotes> {
    if !op.is_float() {
        return None;
    }
    // Native device maths divides by the device's `OpFDiv`, and leaves the NaN Metal answers to the
    // device.
    let native = p.fields().float_functions == FloatFunctions::Native;
    let native_nan = |answers: bool| {
        (answers && nan::encoding(p) == NanEncoding::Native)
            .then_some(PrecisionNote::NativeNanEncoding)
    };
    if divide::precise(op, fmt, fmf) {
        return Some(match native {
            true => OpNotes::everywhere(
                [Some(PrecisionNote::NativeDivision), native_nan(true)]
                    .into_iter()
                    .flatten()
                    .collect(),
            ),
            false => OpNotes::default(),
        });
    }
    // A `float` remainder is Metal's: with every flag its `fast::fmod`, with that function's notes;
    // otherwise exact, its NaN Metal's canonical one.
    if op == AirBin::FRem && fmt == FloatFmt::F32 {
        if !every_flag(fmf) {
            return Some(OpNotes::everywhere(native_nan(true).into_iter().collect()));
        }
        let f = Form {
            held: Held::Float(FloatFmt::F32),
            lanes: 1,
        };
        return float_call(FloatFn::Fmod, true, fmf, f, FloatFacts::on(p)).map(|l| l.notes);
    }
    if op == AirBin::FDiv && fmt == FloatFmt::F32 {
        let mut notes = BTreeSet::from([PrecisionNote::DeviceArithmetic]);
        if native {
            notes.insert(PrecisionNote::NativeDivision);
            notes.extend(native_nan(true));
        }
        return Some(OpNotes::unless_own(
            notes,
            p.fields().float_functions == FloatFunctions::Metal,
        ));
    }
    // A `float` `+ - *` answers Metal's NaN: the device's, where the profile states it is, else
    // selected ([`nan`]), and reads its denormals as Metal does: the device's, where the profile
    // states it does, else flushed ([`with_denormals`]). Metal's `half` and `bfloat` remainders are
    // not measured.
    let answer = nan::binary_answer(op, fmt);
    let nans = answer.is_none();
    let denormals = op == AirBin::FRem;
    let mut notes: BTreeSet<_> = device_float_notes(scalar_float(fmt), denormals, nans).collect();
    if matches!(op, AirBin::FDiv | AirBin::FRem) {
        notes.insert(PrecisionNote::DeviceArithmetic);
    }
    notes.extend(native_nan(answer.is_some()));
    Some(OpNotes::everywhere(notes))
}

/// Strategy `s` of instruction `at`, a result of form `form`, with Metal's NaN `answer` selected
/// ([`nan`]): where Metal answers one the lowering does not, the profile does not leave it to the
/// device, and the result's NaN bits may be observed. Under native device maths
/// ([`NanEncoding::Native`]) every answer is left to the device, noted `NativeNanEncoding`.
fn with_nan(
    cx: &mut SelectCx<'_>,
    at: InstId,
    s: ArithStrategy,
    answer: Option<nan::NanAnswer>,
    form: Form,
) -> ArithStrategy {
    note_native_nan(cx, answer);
    match answer {
        Some(answer)
            if !answer.left_to(nan::encoding(cx.profile)) && cx.nan_payload_observed(at) =>
        {
            ArithStrategy::Nan {
                of: Box::new(s),
                answer,
                form,
            }
        }
        _ => s,
    }
}

/// Record `NativeNanEncoding` where Metal answers a NaN (`answer`) that native device maths
/// leaves to the device ([`NanEncoding::Native`]).
fn note_native_nan(cx: &mut SelectCx<'_>, answer: Option<nan::NanAnswer>) {
    if answer.is_some() && nan::encoding(cx.profile) == NanEncoding::Native {
        cx.precision.insert(PrecisionNote::NativeNanEncoding);
    }
}

/// `s`, an operation on `float` lanes of form `form` whose result has form `out` and whose
/// operands are `operands`, with Metal's reading of its `f32` denormals, `reading`
/// ([`denormal`]). Where the device is measured to read them so (on the deviation, where the
/// profile takes it: [`denormal::device_flushes`]) `s` is left to it, and the module requires
/// the property; else `s` flushes them itself ([`ArithStrategy::Denormals`]), but for an operand
/// no denormal can reach ([`flushed`]). Another format is `s` itself: a `half` is held widened,
/// never an `f32` denormal, and Metal's `bfloat` and `double` environments are not measured.
fn with_denormals(
    cx: &mut SelectCx<'_>,
    s: ArithStrategy,
    (form, out): (Form, Form),
    operands: &[Value],
    reading: denormal::Reading,
) -> ArithStrategy {
    if form.float_fmt() != Some(FloatFmt::F32) {
        return s;
    }
    if reading.covered == denormal::Covered::Deviation && denormal::device_flushes(cx.profile) {
        cx.families.arith.device_denormals = true;
        return s;
    }
    let flush = denormal_flush(cx, operands, reading.operands, reading.result);
    if !flush.any() {
        return s;
    }
    ArithStrategy::Denormals {
        of: Box::new(s),
        form,
        out,
        flush,
    }
}

/// `s`, a `float` operation `op` on lanes of form `f`, with Metal's flush of a product whose exact
/// value lies below `2^-126` where the device's rounding may reach `±2^-126`
/// ([`ArithStrategy::TinyProduct`]): an `f32` multiply whose result can be that small (`result`,
/// [`denormal_result`]) on a profile that computes Metal's products
/// ([`denormal::checks_tiny_products`]). A sum or difference that small is exact, so a rounding
/// never reaches `2^-126` from below it.
fn tiny_product(
    cx: &mut SelectCx<'_>,
    s: ArithStrategy,
    op: AirBin,
    f: Form,
    result: bool,
) -> ArithStrategy {
    let check = denormal::tiny_check(cx.profile)
        .filter(|_| op == AirBin::FMul && result && f.float_fmt() == Some(FloatFmt::F32));
    let Some(check) = cx.families.arith.take_tiny_check(check, f) else {
        return s;
    };
    ArithStrategy::TinyProduct {
        of: Box::new(s),
        form: f,
        check,
    }
}

/// The flushes of a fused addition by `OpFmaKHR` on `float` lanes of form `f` of `operands`
/// (`a`, `b` and `c`, or a chain link's `a` and `b`, [`ArithStrategy::Fused`]): none where the
/// device reads its denormals as Metal does ([`denormal::device_flushes`]; the module then
/// requires the property), else each operand a denormal may reach and the result, as
/// [`with_denormals`].
fn fused_flush(cx: &mut SelectCx<'_>, f: Form, operands: &[Value]) -> denormal::Flush {
    if f.float_fmt() != Some(FloatFmt::F32) {
        return denormal::Flush::NONE;
    }
    if denormal::device_flushes(cx.profile) {
        cx.families.arith.device_denormals = true;
        return denormal::Flush::NONE;
    }
    let reads = u8::try_from((1u32 << operands.len().min(8)) - 1).unwrap_or(u8::MAX);
    denormal_flush(cx, operands, reads, true)
}

/// The flushes of an operation on `operands` that reads those bit `reads` names as Metal does,
/// and flushes its result where `result`: each such operand a denormal may reach
/// ([`flushed`]).
fn denormal_flush(
    cx: &SelectCx<'_>,
    operands: &[Value],
    reads: u8,
    result: bool,
) -> denormal::Flush {
    let mut read = 0u8;
    for (i, v) in operands.iter().enumerate().take(8) {
        if reads & (1 << i) != 0 && !flushed(cx, *v) {
            read |= 1 << i;
        }
    }
    denormal::Flush {
        operands: read,
        result,
    }
}

/// Whether the result of `float` `op` (`+ - *`) on operands `a` and `b`, each read as Metal
/// reads it (no denormal), can be a denormal: not where an operand is a constant whose every
/// lane is a zero, a denormal (read as one), an infinity, a NaN, or, for a sum or difference, of
/// magnitude at least `2^-102`, or, for a product, at least 1. A sum of `x` and such a `c` is a
/// zero or a multiple of `2^-126` (a sum that cancels to below `2^-126` needs `|x|` above
/// `2^-103`, whose unit in the last place is at least `2^-126`, as `c`'s is); a product of `x`
/// and such a `c` is a zero or at least `|x|`, which is normal.
fn denormal_result(cx: &SelectCx<'_>, op: AirBin, a: Value, b: Value) -> bool {
    let least = match op {
        AirBin::FAdd | AirBin::FSub => 2f32.powi(-102),
        AirBin::FMul => 1.0,
        _ => return true,
    };
    let bounds = |v: Value| {
        f32_lanes(cx, v).is_some_and(|lanes| {
            lanes.iter().all(|&bits| {
                let x = f32::from_bits(bits);
                bits & 0x7F80_0000 == 0 || !x.is_finite() || x.abs() >= least
            })
        })
    };
    !(bounds(a) || bounds(b))
}

/// The `f32` lanes of constant `v`, or `None` for another value.
fn f32_lanes(cx: &SelectCx<'_>, v: Value) -> Option<Vec<u32>> {
    use crate::spirv::ids::ScalarTy;
    match cx.constant(v)? {
        LoweredConst::Scalar {
            ty: ScalarTy::Float { bits: 32 },
            bits,
        } => u32::try_from(bits).ok().map(|b| vec![b]),
        LoweredConst::Vector {
            elem: ScalarTy::Float { bits: 32 },
            lanes,
        } => lanes.iter().map(|&b| u32::try_from(b).ok()).collect(),
        _ => None,
    }
}

/// How many definitions [`flushed`] reads for one operand: constant work per operand, so the
/// walk needs no meter. An operand whose walk runs out is flushed, which is exact, only larger.
const FLUSHED_BUDGET: u32 = 48;

/// Whether no `f32` lane of `v` can be a denormal, by its definition's structure: a constant
/// without one; the result of a `float` `+ - * /` or of a function whose denormal result is
/// flushed (`ldexp` on its operand's bits; `fma`, `llvm.fmuladd`: by the device where it is
/// measured to, else by [`with_denormals`]; the divisions' and fused multiply-add routines' by
/// Metal's own rule); an integer or a `half` converted to `float`; an integral or square-root
/// function's (never a denormal); Metal's `fract`, `sign` and `saturate`; and a sign move, lane
/// move, `select` or `phi` of such values, a `phi` on a cycle taken as one while its other incoming
/// values are checked (every value on the cycle is then such a value). A load, an argument or any
/// other value may be one, as may a value past the walk's budget ([`FLUSHED_BUDGET`]).
fn flushed(cx: &SelectCx<'_>, v: Value) -> bool {
    let mut walk = FlushedWalk {
        budget: FLUSHED_BUDGET,
        phis: Vec::new(),
    };
    walk.flushed(cx, v)
}

/// The state of one [`flushed`] walk: its remaining budget and the `phi`s it is inside.
struct FlushedWalk {
    budget: u32,
    phis: Vec<InstId>,
}

impl FlushedWalk {
    fn all(&mut self, cx: &SelectCx<'_>, vs: &[Value]) -> bool {
        vs.iter().all(|x| self.flushed(cx, *x))
    }

    fn flushed(&mut self, cx: &SelectCx<'_>, v: Value) -> bool {
        let Value::Inst(id) = v else {
            return cx.constant(v).is_some_and(|k| !denormal_constant(&k));
        };
        let Some(left) = self.budget.checked_sub(1) else {
            return false;
        };
        self.budget = left;
        let Some(inst) = cx.body.insts.get(id) else {
            return false;
        };
        match &inst.kind {
            InstKind::BinOp {
                op: AirBin::FAdd | AirBin::FSub | AirBin::FMul | AirBin::FDiv,
                ..
            } => true,
            InstKind::FNeg { v, .. } | InstKind::Freeze(v) => self.flushed(cx, *v),
            InstKind::Cast {
                op: CastOp::SIToFP | CastOp::UIToFP,
                ..
            } => true,
            InstKind::Cast {
                op: CastOp::FPExt,
                v,
            } => form(cx, *v).is_ok_and(|f| f.float_fmt() == Some(FloatFmt::Half)),
            InstKind::Select { t, f, .. } => self.all(cx, &[*t, *f]),
            InstKind::ExtractElement { vec, .. } => self.flushed(cx, *vec),
            InstKind::InsertElement { vec, elt, .. } => self.all(cx, &[*vec, *elt]),
            InstKind::ShuffleVector { a, b, .. } => self.all(cx, &[*a, *b]),
            InstKind::Phi { incoming, .. } => {
                if self.phis.contains(&id) {
                    return true;
                }
                self.phis.push(id);
                let all = incoming.iter().all(|(x, _)| self.flushed(cx, *x));
                self.phis.pop();
                all
            }
            InstKind::Call(call) => {
                let fun = match &call.callee {
                    crate::air::ir::Callee::Air {
                        op: AirIntrinsic::Math(m),
                        ..
                    } => air_math(m.f),
                    crate::air::ir::Callee::Air {
                        op: AirIntrinsic::Convert(c),
                        ..
                    } => {
                        return c.dst_kind == NumKind::Float
                            && (c.src_kind != NumKind::Float || c.src.elem == Elem::F16)
                    }
                    crate::air::ir::Callee::Llvm { op, .. } => llvm_math(op),
                    _ => return false,
                };
                use FloatFn as F;
                match fun {
                    Ok(Fun::Float(
                        F::Floor
                        | F::Ceil
                        | F::Trunc
                        | F::Round
                        | F::Rint
                        | F::Sqrt
                        | F::Rsqrt
                        | F::FastSqrt
                        | F::FastRsqrt
                        | F::Ldexp
                        | F::Fma
                        | F::MulAdd
                        | F::Fract
                        | F::Sign
                        | F::Saturate,
                    )) => true,
                    Ok(Fun::Float(F::Abs | F::CopySign)) => {
                        call.args.first().is_some_and(|x| self.flushed(cx, *x))
                    }
                    Ok(Fun::Float(F::Min | F::Max | F::Clamp | F::Min3 | F::Max3 | F::Median3)) => {
                        self.all(cx, &call.args)
                    }
                    _ => false,
                }
            }
            _ => false,
        }
    }
}

/// Whether constant `k` has an `f32` denormal lane.
fn denormal_constant(k: &LoweredConst) -> bool {
    use crate::spirv::ids::ScalarTy;
    let den = |b: u64| b & 0x7F80_0000 == 0 && b & 0x007F_FFFF != 0;
    match k {
        LoweredConst::Scalar {
            ty: ScalarTy::Float { bits: 32 },
            bits,
        } => den(*bits),
        LoweredConst::Vector {
            elem: ScalarTy::Float { bits: 32 },
            lanes,
        } => lanes.iter().any(|b| den(*b)),
        LoweredConst::Composite { parts, .. } => parts.iter().any(denormal_constant),
        _ => false,
    }
}

/// A scalar of float format `fmt`.
fn scalar_float(fmt: FloatFmt) -> Form {
    Form {
        held: Held::Float(fmt),
        lanes: 1,
    }
}

/// Select the strategy of instruction `inst` of this family, metering in `m` the walks the
/// access family takes for the store of `air.sincos`.
pub fn select(
    cx: &mut SelectCx<'_>,
    at: InstId,
    inst: &Inst,
    m: &mut Meter,
) -> Result<ArithStrategy, Error> {
    if let Some(s) = contract::select(cx, at, inst)? {
        return Ok(s);
    }
    if let InstKind::Cast {
        op: CastOp::ZExt | CastOp::SExt,
        ..
    } = &inst.kind
    {
        if access::index::is_index_extension(cx, at) {
            return Ok(ArithStrategy::IndexExtension);
        }
    }
    if let InstKind::BinOp { .. } = &inst.kind {
        if access::index::is_index_word(cx, at) {
            return access::index::select_word(cx, inst).map(ArithStrategy::IndexWord);
        }
    }
    if aggregate::involved(cx, inst) {
        return aggregate::select(cx, at, inst);
    }
    if chunk::involved(cx, inst) {
        return chunk::select(cx, at, inst, m);
    }
    if wide::involved(cx, inst) {
        return wide::select(cx, at, inst);
    }
    select_in(cx, at, inst, None, m)
}

/// The form of `v`; with `chunk`, a chunked vector's form is that of one chunk of `chunk` lanes.
fn form_in(cx: &SelectCx<'_>, v: Value, chunk: Option<u8>) -> Result<Form, Error> {
    let t = ty_of(cx, v)?;
    chunk_form(cx, t, chunk)
}

/// [`form_in`] of the instruction's result.
fn result_form_in(cx: &SelectCx<'_>, inst: &Inst, chunk: Option<u8>) -> Result<Form, Error> {
    chunk_form(cx, inst.ty, chunk)
}

fn chunk_form(cx: &SelectCx<'_>, t: AirTypeId, chunk: Option<u8>) -> Result<Form, Error> {
    if let (Some(k), Some((held, _))) = (chunk, legal::chunked(cx.entry.module.types(), t)) {
        let f = Form { held, lanes: k };
        legal::check_scalar(cx.profile, f.scalar())?;
        return Ok(f);
    }
    form_on(cx.entry, cx.profile, t)
}

/// Select `inst`; with `chunk`, for one chunk of `chunk` lanes of a lane-wise operation on
/// chunked vectors (its chunked operands and result take the chunk's form).
fn select_in(
    cx: &mut SelectCx<'_>,
    at: InstId,
    inst: &Inst,
    chunk: Option<u8>,
    m: &mut Meter,
) -> Result<ArithStrategy, Error> {
    use ArithStrategy as S;
    if let InstKind::Call(call) = &inst.kind {
        return select_call(cx, at, inst, call, chunk, m);
    }
    let mut ops = operands_of(&inst.kind);
    // A constant element index is a literal of the instruction, never a materialized value (an
    // `i64` index needs no `shaderInt64`).
    if let InstKind::ExtractElement { idx, .. } | InstKind::InsertElement { idx, .. } = &inst.kind {
        if matches!(idx, Value::Const(_)) {
            ops.retain(|v| v != idx);
        }
    }
    check_operands(cx, &ops)?;
    // The result is a value of a lowered type (or the operation is refused).
    let result = cx.lowered_repr(Value::Inst(at), inst.ty)?;
    Ok(match &inst.kind {
        InstKind::BinOp { op, a, b, fmf, .. } => {
            let f = form_in(cx, *a, chunk)?;
            // A `half` output's last operation Metal computes at `float` precision: its operands,
            // `half`s held widened, operated on as `float`s.
            let f = match cx.families.arith.unrounded.contains(&at) {
                true => f.with(Held::Float(FloatFmt::F32)),
                false => f,
            };
            match f.held {
                Held::Bool if !op.is_float() => S::Bool { op: *op, form: f },
                Held::Int { .. } if !op.is_float() => S::Int { op: *op, form: f },
                // Metal divides `float` without `arcp` correctly rounded, a denormal operand
                // read as a zero and a denormal quotient flushed (a probe cases 914cc4d2, 711cc190,
                // and c10cba13, every lane), and `half` under every flag set rounded once from the
                // exact quotient (4c363561, 54f0fe8e, every lane): a shared routine computes that
                // quotient on every profile, apart from the device's division and its environment
                // (a `half` one is its `float` quotient rounded once). LLVM's `frem` on `float` is
                // Metal's (measured on the M3 Max): with every fast-math flag its `fast::fmod`, as
                // `air.fast_fmod` is lowered; otherwise `x - y * trunc(x / y)` of its precise
                // quotient, each step rounded and read as Metal reads `f32` denormals, a NaN the
                // canonical one.
                Held::Float(FloatFmt::F32) if *op == AirBin::FRem => {
                    if every_flag(*fmf) {
                        return select_float_fn(cx, at, (FloatFn::Fmod, true), *fmf, f, &[*a, *b]);
                    }
                    fp_env(cx, *fmf, f);
                    let routine = divide::Routine::on(cx.profile, f);
                    cx.families.arith.precise.insert(routine);
                    let flush = !denormal::device_flushes(cx.profile);
                    cx.families.arith.device_denormals |= !flush;
                    let s = S::Remainder {
                        form: f,
                        routine,
                        flush,
                    };
                    let s =
                        with_denormals(cx, s, (f, f), &[*a, *b], denormal::Reading::binary(true));
                    with_nan(cx, at, s, Some(nan::NanAnswer::Canonical), f)
                }
                Held::Float(fmt) if divide::precise(*op, fmt, *fmf) => {
                    let notes = float_binop_notes(*op, fmt, *fmf, cx.profile);
                    record(cx, notes.unwrap_or_default());
                    note_rewrites(cx, Some(at), (*op, *fmf), *a, *b);
                    // Native device maths: the device's quotient. Its operands and result read as
                    // Metal reads `f32` denormals, and its NaN Metal's canonical one, as the
                    // routine answers them.
                    if cx.profile.fields().float_functions == FloatFunctions::Native {
                        fp_env(cx, *fmf, f);
                        let s = S::NativeDivide { form: f };
                        // No device is measured to read a quotient's denormals so.
                        let reading = denormal::Reading {
                            covered: denormal::Covered::Never,
                            operands: 0b11,
                            result: true,
                        };
                        let s = with_denormals(cx, s, (f, f), &[*a, *b], reading);
                        return Ok(with_nan(cx, at, s, Some(nan::NanAnswer::Canonical), f));
                    }
                    let routine = divide::Routine::on(cx.profile, f);
                    cx.families.arith.precise.insert(routine);
                    S::PreciseDivide { form: f, routine }
                }
                Held::Float(fmt) if op.is_float() => {
                    let notes = float_binop_notes(*op, fmt, *fmf, cx.profile);
                    record(cx, notes.unwrap_or_default());
                    // A tree's addition Metal regroups by the measured rule; its plan notes the
                    // values the tree reads.
                    if !cx.families.arith.contract.members.contains(&at) {
                        note_rewrites(cx, Some(at), (*op, *fmf), *a, *b);
                    }
                    fp_env(cx, *fmf, f);
                    // An addition Metal keeps apart from its `contract` multiply operand is
                    // never fused by the device either.
                    let contract = if cx.families.arith.contract.kept.contains(&at) {
                        Contraction::Forbidden
                    } else {
                        contraction(*fmf)
                    };
                    // Metal divides `float` under `arcp` as `x` times its fast reciprocal of `y`
                    // (a probe cases 914cc4d2 and 711cc190, every flag set measured, every lane),
                    // its reciprocal flushing past `2^126` and at a denormal `y`. That reciprocal
                    // is the profile's (`divide::Reciprocal::on`): the device's own where its float
                    // functions and division are Metal's, `RN(1 / y)` where it has `OpFmaKHR`, and
                    // the device's elsewhere. Where the first's module is compiled with fast math,
                    // the quotient is the device's `OpFDiv(x, y)` itself
                    // (`ArithPlan::device_quotient`, known once the entry's modes are, so the
                    // routine waits in `arcp`). Only that form is Metal's on every lane, and only
                    // in that mode, so the note stays. The routine answers the canonical NaN
                    // itself.
                    if *op == AirBin::FDiv
                        && fmf.contains(FastMath::ARCP)
                        && f.float_fmt() == Some(FloatFmt::F32)
                    {
                        let reciprocal = divide::Reciprocal::on(cx.profile);
                        let routine = math::FastDivision::new(f, reciprocal, contract);
                        cx.families.arith.arcp.insert(routine);
                        let s = S::FastDivide { form: f, routine };
                        // Native device maths' quotient is the device's `OpFDiv`: its NaN is the
                        // device's where the profile leaves NaNs to it, else Metal's canonical one,
                        // selected as the routine would answer it.
                        if cx.profile.fields().float_functions == FloatFunctions::Native {
                            with_nan(cx, at, s, Some(nan::NanAnswer::Canonical), f)
                        } else {
                            s
                        }
                    } else if let Some(basis) = f16_arithmetic(cx.profile, *op, f) {
                        // The device's own `half` arithmetic, Metal's in the forms that keep it
                        // apart, which the AIR's flags allow whatever they are. Only the measured
                        // basis is a property the module requires; the float controls are modes
                        // its entry declares.
                        cx.families.arith.f16_arith |= basis == F16Basis::Measured;
                        let s = S::F16 { op: *op, form: f };
                        with_nan(cx, at, s, nan::binary_answer(*op, fmt), f)
                    } else {
                        let s = S::Float {
                            op: *op,
                            form: f,
                            contract,
                        };
                        // Metal reads an `f32` denormal operand of `+ - *` as a zero and flushes
                        // a denormal result, a product's before rounding; its remainder is not
                        // measured.
                        let s = match op {
                            AirBin::FAdd | AirBin::FSub | AirBin::FMul => {
                                let result = denormal_result(cx, *op, *a, *b);
                                let s = tiny_product(cx, s, *op, f, result);
                                with_denormals(
                                    cx,
                                    s,
                                    (f, f),
                                    &[*a, *b],
                                    denormal::Reading::binary(result),
                                )
                            }
                            _ => s,
                        };
                        with_nan(cx, at, s, nan::binary_answer(*op, fmt), f)
                    }
                }
                _ => return Err(cx.invariant("a binary operator on mismatched operands")),
            }
        }
        InstKind::FNeg { v, fmf } => {
            let f = form_in(cx, *v, chunk)?;
            // A sign flip: it makes no NaN. Metal's `float` `fneg` flips the sign bit alone,
            // keeping a NaN's payload and a denormal, whatever the flags, so it moves on the bits;
            // elsewhere whether it keeps a denormal is the device's.
            let bits = math::sign_on_bits(f.float_fmt());
            device_float(cx, f, !bits, false);
            fp_env(cx, *fmf, f);
            // Metal's `half` `fneg` answers the canonical NaN.
            let nan = f.float_fmt().and_then(nan::fneg_answer);
            with_nan(cx, at, S::FNeg { form: f, bits }, nan, f)
        }
        InstKind::ICmp { pred, a, .. } => {
            let f = form_in(cx, *a, chunk)?;
            if f.is_bool() {
                S::BoolCompare { pred: *pred }
            } else {
                S::IntCompare {
                    pred: *pred,
                    form: f,
                }
            }
        }
        InstKind::FCmp { pred, a, b, fmf } => {
            let f = form_in(cx, *a, chunk)?;
            // Metal compares an `f32` denormal as a zero; the result is a `bool`, never a NaN.
            fp_env(cx, *fmf, f);
            let s = S::FloatCompare {
                pred: *pred,
                form: f,
            };
            with_denormals(cx, s, (f, f), &[*a, *b], denormal::Reading::binary(false))
        }
        InstKind::Cast { op, v } => select_cast(cx, at, *op, *v, inst, chunk)?,
        InstKind::Select { .. } => S::Select,
        InstKind::Freeze(v) => match cx.constant(*v) {
            Some(LoweredConst::Undef(_)) => S::Zero(result),
            _ => S::Copy,
        },
        // A constant index out of range (or `undef`) gives poison.
        InstKind::ExtractElement { vec, idx } => match (idx, const_int(cx, *idx)) {
            (Value::Const(_), Some(i)) if i < u64::from(lanes_of(cx, *vec)?) => {
                S::ExtractConst(u32::try_from(i).map_err(|_| not_lowered())?)
            }
            (Value::Const(_), _) => S::Undef(result),
            _ => S::ExtractDynamic {
                index: form_in(cx, *idx, None)?,
            },
        },
        InstKind::InsertElement { vec, idx, .. } => match (idx, const_int(cx, *idx)) {
            (Value::Const(_), Some(i)) if i < u64::from(lanes_of(cx, *vec)?) => {
                S::InsertConst(u32::try_from(i).map_err(|_| not_lowered())?)
            }
            (Value::Const(_), _) => S::Undef(result),
            _ => S::InsertDynamic {
                index: form_in(cx, *idx, None)?,
            },
        },
        InstKind::ShuffleVector { a, b, mask } => {
            let total = lanes_of(cx, *a)?.saturating_add(lanes_of(cx, *b)?);
            let mut comps = Vec::with_capacity(mask.len());
            for m in mask.iter() {
                // A poison lane may be any value: take component 0.
                let c = m.unwrap_or(0);
                if c >= total {
                    return Err(cx.invariant("a shuffle index beyond its operands"));
                }
                comps.push(c);
            }
            S::Shuffle(comps.into())
        }
        // A `bool` vector member is held as `u32` lanes.
        InstKind::ExtractValue { indices, .. } => {
            match legal::bool_lanes(cx.entry.module.types(), inst.ty) {
                Some(lanes) => S::ExtractBools {
                    path: indices.clone(),
                    lanes,
                },
                None => S::ExtractValue(indices.clone()),
            }
        }
        InstKind::InsertValue { val, indices, .. } => {
            match legal::bool_lanes(cx.entry.module.types(), ty_of(cx, *val)?) {
                Some(lanes) => S::InsertBools {
                    path: indices.clone(),
                    lanes,
                },
                None => S::InsertValue(indices.clone()),
            }
        }
        _ => return Err(not_lowered()),
    })
}

/// Select the cast `op` of `v`, instruction `at`; with `chunk`, for one chunk (as
/// [`select_in`]).
fn select_cast(
    cx: &mut SelectCx<'_>,
    at: InstId,
    op: CastOp,
    v: Value,
    inst: &Inst,
    chunk: Option<u8>,
) -> Result<ArithStrategy, Error> {
    let from = form_in(cx, v, chunk)?;
    let to = result_form_in(cx, inst, chunk)?;
    let kind = match op {
        CastOp::Trunc => CastKind::Trunc,
        CastOp::ZExt => CastKind::ZExt,
        CastOp::SExt => CastKind::SExt,
        CastOp::FPTrunc | CastOp::FPExt => {
            // Between `float` and `double` a denormal (and a NaN's payload) is the device's.
            if from.bits() == 64 || to.bits() == 64 {
                device_float(cx, from, true, true);
                device_float(cx, to, true, true);
            }
            let s = ArithStrategy::Cast {
                kind: CastKind::FpResize,
                from,
                to,
            };
            // A `float` narrowed to `half` answers Metal's canonical `half` NaN.
            let answer = match (from.float_fmt(), to.float_fmt()) {
                (Some(a), Some(b)) => nan::cast_answer(op, a, b),
                _ => None,
            };
            return Ok(with_nan(cx, at, s, answer, to));
        }
        CastOp::FPToUI => CastKind::FpToUi,
        CastOp::FPToSI => CastKind::FpToSi,
        CastOp::UIToFP => CastKind::UiToFp,
        CastOp::SIToFP => CastKind::SiToFp,
        CastOp::Bitcast if chunk.is_some() => {
            return Err(cx.invariant("a bitcast selected per chunk"))
        }
        CastOp::Bitcast => {
            let plan = ops::bitcast_plan(from, to).ok_or_else(not_lowered)?;
            if let BitcastPlan::Packed { total } = plan {
                // The packed integer is a legal type on this profile.
                legal::check_scalar(cx.profile, ops::packed_form(total).scalar())?;
            }
            return Ok(ArithStrategy::Bitcast { plan, from, to });
        }
        CastOp::PtrToInt | CastOp::IntToPtr | CastOp::AddrSpaceCast => {
            return Err(cx.invariant("a pointer cast routed to arith"))
        }
    };
    Ok(ArithStrategy::Cast { kind, from, to })
}

fn sign(s: AirSign) -> Sign {
    match s {
        AirSign::Signed => Sign::Signed,
        AirSign::Unsigned => Sign::Unsigned,
    }
}

/// The arith function of a decoded `air` math intrinsic, or a refusal.
enum Fun {
    Int(IntFn),
    Float(FloatFn),
    Copy,
    /// `sincos`, which also stores (see [`select_sincos`]).
    SinCos,
}

fn air_math(f: MathFn) -> Result<Fun, Error> {
    use FloatFn as F;
    use MathFn as M;
    Ok(match f {
        M::Fabs => Fun::Float(F::Abs),
        M::Sqrt => Fun::Float(F::Sqrt),
        M::Rsqrt => Fun::Float(F::Rsqrt),
        M::Exp => Fun::Float(F::Exp),
        M::Exp2 => Fun::Float(F::Exp2),
        M::Exp10 => Fun::Float(F::Exp10),
        M::Log => Fun::Float(F::Log),
        M::Log2 => Fun::Float(F::Log2),
        M::Log10 => Fun::Float(F::Log10),
        M::Sin => Fun::Float(F::Sin),
        M::Cos => Fun::Float(F::Cos),
        M::Tan => Fun::Float(F::Tan),
        M::Asin => Fun::Float(F::Asin),
        M::Acos => Fun::Float(F::Acos),
        M::Atan => Fun::Float(F::Atan),
        M::Sinh => Fun::Float(F::Sinh),
        M::Cosh => Fun::Float(F::Cosh),
        M::Tanh => Fun::Float(F::Tanh),
        M::Asinh => Fun::Float(F::Asinh),
        M::Acosh => Fun::Float(F::Acosh),
        M::Atanh => Fun::Float(F::Atanh),
        M::SinPi => Fun::Float(F::SinPi),
        M::CosPi => Fun::Float(F::CosPi),
        M::TanPi => Fun::Float(F::TanPi),
        M::Floor => Fun::Float(F::Floor),
        M::Ceil => Fun::Float(F::Ceil),
        M::Round => Fun::Float(F::Round),
        M::Rint => Fun::Float(F::Rint),
        M::Trunc => Fun::Float(F::Trunc),
        M::Fract => Fun::Float(F::Fract),
        M::FSign => Fun::Float(F::Sign),
        M::Saturate => Fun::Float(F::Saturate),
        M::FMin | M::Min => Fun::Float(F::Min),
        M::FMax | M::Max => Fun::Float(F::Max),
        M::Powr => Fun::Float(F::Powr),
        M::Fmod => Fun::Float(F::Fmod),
        M::Atan2 => Fun::Float(F::Atan2),
        M::Fma => Fun::Float(F::Fma),
        M::Mix => Fun::Float(F::Mix),
        M::Clamp => Fun::Float(F::Clamp),
        M::FMin3 => Fun::Float(F::Min3),
        M::FMax3 => Fun::Float(F::Max3),
        M::FMedian3 => Fun::Float(F::Median3),
        M::Ldexp => Fun::Float(F::Ldexp),
        M::Dot => Fun::Float(F::Dot),
        M::Abs(AirSign::Signed) => Fun::Int(IntFn::Abs),
        M::Abs(AirSign::Unsigned) => Fun::Copy,
        M::AbsDiff(s) => Fun::Int(IntFn::AbsDiff(sign(s))),
        M::IMin(s) => Fun::Int(IntFn::Min(sign(s))),
        M::IMax(s) => Fun::Int(IntFn::Max(sign(s))),
        M::IClamp(s) => Fun::Int(IntFn::Clamp(sign(s))),
        M::IMin3(s) => Fun::Int(IntFn::Min3(sign(s))),
        M::IMax3(s) => Fun::Int(IntFn::Max3(sign(s))),
        M::MulHi(s) => Fun::Int(IntFn::MulHi(sign(s))),
        M::MadHi(s) => Fun::Int(IntFn::MadHi(sign(s))),
        M::MadSat(s) => Fun::Int(IntFn::MadSat(sign(s))),
        M::Pow => Fun::Float(F::Pow),
        M::AddSat(s) => Fun::Int(IntFn::AddSat(sign(s))),
        M::SubSat(s) => Fun::Int(IntFn::SubSat(sign(s))),
        M::HAdd(s) => Fun::Int(IntFn::HAdd(sign(s))),
        M::RHAdd(s) => Fun::Int(IntFn::RHAdd(sign(s))),
        M::ExtractBits(s) => Fun::Int(IntFn::ExtractBits(sign(s))),
        M::InsertBits(_) => Fun::Int(IntFn::InsertBits),
        M::Popcount => Fun::Int(IntFn::Popcount),
        M::ReverseBits => Fun::Int(IntFn::ReverseBits),
        M::Bswap => Fun::Int(IntFn::Bswap),
        M::Rotate => Fun::Int(IntFn::Rotate),
        M::Clz { .. } => Fun::Int(IntFn::Clz),
        M::Ctz { .. } => Fun::Int(IntFn::Ctz),
        M::SinCos => Fun::SinCos,
        // Canonicalization replaces every runtime routine's call (`air::canon::routine`).
        M::Frexp | M::Ilogb => {
            return Err(Error::Internal(crate::error::InternalError::new(
                crate::error::InternalKind::Invariant,
                "a runtime routine's call survived canonicalization",
            )))
        }
    })
}

fn llvm_math(op: &LlvmIntrinsic) -> Result<Fun, Error> {
    Ok(match *op {
        LlvmIntrinsic::Float { op, .. } => Fun::Float(match op {
            LlvmFloatOp::FMulAdd => FloatFn::MulAdd,
            LlvmFloatOp::Fma => FloatFn::Fma,
            LlvmFloatOp::FAbs => FloatFn::Abs,
            LlvmFloatOp::MinNum => FloatFn::MinNum,
            LlvmFloatOp::MaxNum => FloatFn::MaxNum,
            // Apple's math library defines precise `air.exp2`, `air.log2`, and `air.sqrt` as
            // these calls, so they are those functions.
            LlvmFloatOp::Exp2 => FloatFn::Exp2,
            LlvmFloatOp::Log2 => FloatFn::Log2,
            LlvmFloatOp::Sqrt => FloatFn::Sqrt,
            LlvmFloatOp::CopySign => FloatFn::CopySign,
        }),
        LlvmIntrinsic::Int { op, .. } => Fun::Int(match op {
            LlvmIntOp::SMax => IntFn::Max(Sign::Signed),
            LlvmIntOp::SMin => IntFn::Min(Sign::Signed),
            LlvmIntOp::UMax => IntFn::Max(Sign::Unsigned),
            LlvmIntOp::UMin => IntFn::Min(Sign::Unsigned),
            LlvmIntOp::Fshl => IntFn::Fshl,
            LlvmIntOp::Fshr => IntFn::Fshr,
            LlvmIntOp::Bswap => IntFn::Bswap,
            LlvmIntOp::Ctpop => IntFn::Popcount,
            LlvmIntOp::UAddSat => IntFn::AddSat(Sign::Unsigned),
            LlvmIntOp::USubSat => IntFn::SubSat(Sign::Unsigned),
            LlvmIntOp::SAddSat => IntFn::AddSat(Sign::Signed),
            LlvmIntOp::SSubSat => IntFn::SubSat(Sign::Signed),
        }),
        LlvmIntrinsic::Abs { .. } => Fun::Int(IntFn::Abs),
        LlvmIntrinsic::Cttz { .. } => Fun::Int(IntFn::Ctz),
        LlvmIntrinsic::Ctlz { .. } => Fun::Int(IntFn::Clz),
        LlvmIntrinsic::Memset { .. }
        | LlvmIntrinsic::Memcpy { .. }
        | LlvmIntrinsic::Memmove { .. }
        | LlvmIntrinsic::Agx(_) => return Err(not_lowered()),
    })
}

/// The float function a call of `op` on form `f` computes, given whether its `air` symbol is the
/// `fast_` variant. The `fast_` variants on `float` that answer differently from the precise ones
/// are their own functions; the fast-math accuracy table is single precision, so a narrow `fast_`
/// call keeps the precise function:
///
/// - `air.fast_tanh` is the MSL fast-math formula ([`FloatFn::FastTanh`]), divided by the
///   profile's fast reciprocal (`reciprocal`);
/// - `air.fast_asinh` with every flag is MSL Table 8.2's formula, the canonical NaN at `-inf`
///   ([`FloatFn::FastAsinh`]);
/// - `air.fast_sinh`, `air.fast_cosh`, `air.fast_acosh`, and `air.fast_atanh` are MSL Table
///   8.2's formulas of Metal's fast `exp`, `log`, `sqrt`, and division ([`FloatFn::FastSinh`],
///   [`FloatFn::FastCosh`], [`FloatFn::FastAcosh`], [`FloatFn::FastAtanh`]), the last divided by
///   the profile's fast reciprocal (`reciprocal`), where the device's instruction is not Metal's
///   fast function;
/// - `air.fast_fract` is not bounded below one, and with every flag is the canonical NaN at an
///   infinity ([`FloatFn::FastFract`]);
/// - `air.fast_fmedian3` answers `b` for two NaN first operands ([`FloatFn::FastMedian3`]);
/// - `air.fast_pow` answers the canonical NaN for a NaN base to the exponent one
///   ([`FloatFn::FastPow`]);
/// - `air.fast_atan2` is `atan(y / x)` turned by `±π`, NaN at `0 / 0` and `inf / inf`
///   ([`FloatFn::FastAtan2`]);
/// - `air.fast_sin` and `air.fast_cos` answer `+0` past `float(π/2) · 2^22` and for an infinity
///   or NaN ([`FloatFn::FastSin`], [`FloatFn::FastCos`]);
/// - `air.fast_tan` is Metal's fast `tan` ([`FloatFn::FastTan`]), Apple's formula divided by the
///   profile's fast reciprocal (`reciprocal`) where the device's `Tan` is not it;
/// - `air.fast_asin`, `air.fast_acos`, and `air.fast_atan` are Metal's fast functions, which a
///   device that is Metal's own answers only in a fast-math module and which answer `fast::asin`
///   of a zero and of a denormal as `+0` and `fast::atan` of one as its signed zero
///   ([`FloatFn::FastAsin`], [`FloatFn::FastAcos`], [`FloatFn::FastAtan`]);
/// - `air.fast_rsqrt` is Metal's fast `rsqrt` ([`FloatFn::FastRsqrt`]), apart from the correctly
///   rounded precise one;
/// - `air.fast_sqrt`, `air.fast_exp`, `air.fast_exp10`, `air.fast_log`, `air.fast_log10`, and
///   `air.fast_powr` are Metal's compositions `x · rsqrt(x)`, `exp2(x · log2 e)`,
///   `exp2(x · log2 10)`, `log2(x) · ln 2`, `log2(x) · log10 2`, and `exp2(y · log2 x)`
///   ([`FloatFn::FastSqrt`], [`FloatFn::FastExp`], [`FloatFn::FastExp10`], [`FloatFn::FastLog`],
///   [`FloatFn::FastLog10`], [`FloatFn::FastPowr`]); `air.fast_pow`'s magnitude is the last of
///   `|x|`;
/// - `air.fast_log2` is Metal's `log2`, the precise one's routine with its table
///   ([`FloatFn::FastLog2`]), which the fast formulas' `log2` is too;
/// - `air.fast_sinpi`, `air.fast_cospi`, and `air.fast_tanpi` are Metal's precise functions,
///   which a device whose functions are Metal's answers by its `Sin` and `Cos` within Apple's
///   fast bound ([`FloatFn::FastSinPi`], [`FloatFn::FastCosPi`], [`FloatFn::FastTanPi`]).
///
/// `air.fast_fmod` on `float` is selected beside this, as [`ArithStrategy::FastFmod`].
///
/// `llvm.minnum` and `llvm.maxnum` on `float`, Metal's fast-math form of `a < b ? a : b` and
/// `a > b ? a : b` (with every flag), are Metal's `fmin` and `fmax` ([`FloatFn::Min`],
/// [`FloatFn::Max`]), with any flags; on the other formats they stay LLVM's (`FloatFn::MinNum`).
///
/// `air.fract` on `half` with every flag, Metal's fast-math compile of a `half` `fract` (it has
/// no `fast_` `half` function), answers the largest `half` below one for a NaN and an infinity
/// ([`FloatFn::FlaggedFract`]); a `fast_` `half` symbol, not measured, keeps the precise function
/// as the other narrow `fast_` calls do.
///
/// A call's fast-math flags `fmf` do not change the function: Metal's `sign`, for one, answers
/// `+0` for `-0` under fast math too (A1 case f9a12252). The exceptions measured are
/// `fast_atan2`'s quotient beyond a divisor of `2^126`, which a call without every flag takes
/// from a flushed reciprocal (probe cases 6802ab13 and c11eb8cb), and `fast_fract` and `fast_asinh`
/// at infinities, which only a call with every flag answers as a NaN (A1 case aed3faaa without
/// flags, our probe with them), and `half` `fract`'s NaN and infinities. The other flag
/// combinations are not measured; they take the answer of a call without flags.
fn refine_float(
    op: FloatFn,
    fast: bool,
    fmf: FastMath,
    f: Form,
    reciprocal: divide::Reciprocal,
) -> FloatFn {
    if f.float_fmt() == Some(FloatFmt::Half) && op == FloatFn::Fract && !fast && every_flag(fmf) {
        return FloatFn::FlaggedFract;
    }
    if f.float_fmt() != Some(FloatFmt::F32) {
        return op;
    }
    // `a < b ? a : b` and `a > b ? a : b` under fast math are `llvm.minnum` and `llvm.maxnum`
    // with every flag, which Metal answers as `fmin` and `fmax` (`nan_minmax_fast`, every lane);
    // without a flag Metal answers them the same, word for word (those two calls of
    // `nan_minmax_fast` with their flags removed, every pair of 20 patterns, NaNs and denormals
    // among them), and so with some flags (`nnan` on `llvm.minnum`, `nsz arcp` on `llvm.maxnum`).
    match op {
        FloatFn::MinNum => return FloatFn::Min,
        FloatFn::MaxNum => return FloatFn::Max,
        _ => {}
    }
    if !fast {
        return op;
    }
    match op {
        FloatFn::Tanh => FloatFn::FastTanh { reciprocal },
        FloatFn::Asinh if every_flag(fmf) => FloatFn::FastAsinh,
        FloatFn::Sinh => FloatFn::FastSinh,
        FloatFn::Cosh => FloatFn::FastCosh,
        FloatFn::Acosh => FloatFn::FastAcosh,
        FloatFn::Atanh => FloatFn::FastAtanh { reciprocal },
        FloatFn::Exp10 => FloatFn::FastExp10,
        FloatFn::Log10 => FloatFn::FastLog10,
        FloatFn::Fract => FloatFn::FastFract {
            infinite_nan: every_flag(fmf),
        },
        FloatFn::Median3 => FloatFn::FastMedian3,
        FloatFn::Pow => FloatFn::FastPow,
        FloatFn::Sin => FloatFn::FastSin,
        FloatFn::Cos => FloatFn::FastCos,
        FloatFn::Tan => FloatFn::FastTan { reciprocal },
        FloatFn::Asin => FloatFn::FastAsin,
        FloatFn::Acos => FloatFn::FastAcos,
        FloatFn::Atan => FloatFn::FastAtan,
        FloatFn::Sqrt => FloatFn::FastSqrt,
        FloatFn::Rsqrt => FloatFn::FastRsqrt,
        FloatFn::Exp => FloatFn::FastExp,
        FloatFn::Log => FloatFn::FastLog,
        FloatFn::Log2 => FloatFn::FastLog2,
        FloatFn::Powr => FloatFn::FastPowr,
        FloatFn::Atan2 => FloatFn::FastAtan2 {
            reciprocal_flushed: !every_flag(fmf),
        },
        FloatFn::SinPi => FloatFn::FastSinPi,
        FloatFn::CosPi => FloatFn::FastCosPi,
        FloatFn::TanPi => FloatFn::FastTanPi,
        op => op,
    }
}

/// Whether `fmf` holds every fast-math flag, as Metal's compiler gives each `float` operation
/// under fast math: the state the `fast_` functions are measured in.
fn every_flag(fmf: FastMath) -> bool {
    let every = FastMath::ALL
        .iter()
        .fold(FastMath::NONE, |all, (flag, _)| all.union(*flag));
    fmf.contains(every)
}

fn select_call(
    cx: &mut SelectCx<'_>,
    at: InstId,
    inst: &Inst,
    call: &crate::air::ir::Call,
    chunk: Option<u8>,
    m: &mut Meter,
) -> Result<ArithStrategy, Error> {
    use crate::air::intrinsics::Precision;
    use crate::air::ir::Callee;
    let mut fast = false;
    let fun = match &call.callee {
        Callee::Air {
            op: AirIntrinsic::Math(m),
            ..
        } => {
            fast = m.precision == Precision::Fast;
            air_math(m.f)?
        }
        Callee::Air {
            op: AirIntrinsic::Convert(c),
            ..
        } => return select_convert(cx, at, inst, call, c, chunk),
        Callee::Air {
            op: AirIntrinsic::Pack(p),
            ..
        } if chunk.is_none() => return select_pack(cx, inst, call, p),
        Callee::Llvm { op, .. } => llvm_math(op)?,
        _ => return Err(not_lowered()),
    };
    let arity = match &fun {
        Fun::Int(f) => f.arity(),
        Fun::Float(f) => f.arity(),
        Fun::Copy => 1,
        // A chunked `sincos` is selected whole ([`chunk::select`]), never per chunk.
        Fun::SinCos if chunk.is_some() => return Err(cx.invariant("air.sincos selected per chunk")),
        Fun::SinCos => return select_sincos(cx, at, inst, call, m),
    };
    let args = call
        .args
        .get(..arity)
        .ok_or_else(|| cx.invariant("too few intrinsic operands"))?;
    check_operands(cx, args)?;
    lowered_type(cx.entry, cx.profile, inst.ty)?;
    let first = *args
        .first()
        .ok_or_else(|| cx.invariant("an intrinsic without operands"))?;
    let f = form_in(cx, first, chunk)?;
    Ok(match fun {
        Fun::Copy => ArithStrategy::Copy,
        Fun::SinCos => return Err(cx.invariant("sincos selected as a plain function")),
        Fun::Int(op) => {
            let ok = match (f.held, op) {
                (Held::Int { bits, .. }, IntFn::Bswap) => bits % 16 == 0,
                (Held::Int { .. }, _) => true,
                _ => false,
            };
            if !ok {
                return Err(not_lowered());
            }
            ArithStrategy::IntFn { f: op, form: f }
        }
        Fun::Float(op) => select_float_fn(cx, at, (op, fast), call.fmf, f, args)?,
    })
}

/// The strategy of float function `op` (`fast` its `fast_` variant) with fast-math flags `fmf`
/// on operands `args` of form `f`, the result of instruction `at`: a math call's, or LLVM's
/// `frem`'s, which is `fmod`.
fn select_float_fn(
    cx: &mut SelectCx<'_>,
    at: InstId,
    (op, fast): (FloatFn, bool),
    fmf: FastMath,
    f: Form,
    args: &[Value],
) -> Result<ArithStrategy, Error> {
    let facts = FloatFacts::on(cx.profile);
    let lowered = float_call(op, fast, fmf, f, facts).ok_or_else(not_lowered)?;
    record(cx, lowered.notes);
    fp_env(cx, fmf, f);
    if let FloatLowering::Fn(op) = lowered.how {
        let uses = libm::uses(op, f);
        let arith = &mut cx.families.arith;
        arith.libm_uses.extend(uses.iter().copied());
        arith.device_min_max |= device_min_max(cx.profile, op, f);
        arith.native_min_max |= native_min_max(cx.profile, op, f);
    }
    if let FloatLowering::Fma(fma)
    | FloatLowering::FastFmod { fma, .. }
    | FloatLowering::Mix(fma)
    | FloatLowering::Dot(fma) = lowered.how
    {
        cx.families.arith.take_fma(fma);
    }
    // `fast::fmod`, `fast::tanh`, and `fast::tan` divide by the fast division's routine.
    if let Some(r) = fast_division_of(lowered.how, f) {
        cx.families.arith.fast.insert(r);
    }
    // `dot`'s result is a scalar.
    let result = match lowered.how {
        FloatLowering::Fn(FloatFn::Dot) | FloatLowering::Dot(_) => f.scalar_form(),
        _ => f,
    };
    let s = match lowered.how {
        FloatLowering::Fma(fma) => ArithStrategy::Fma(fma),
        FloatLowering::FastFmod { fma, reciprocal } => ArithStrategy::FastFmod { fma, reciprocal },
        // As `dot`'s chain: the inner fused multiply-add's result is flushed where `OpFmaKHR` may
        // keep a denormal ([`with_denormals`] flushes the operands and the result); the routines,
        // and `OpFmaKHR` checked ([`fma::Fma::checked`]), read and answer them as Metal does.
        FloatLowering::Mix(fma) => ArithStrategy::Mix {
            fma,
            flush: fma.route.device_float()
                && !fma.checked()
                && f.float_fmt() == Some(FloatFmt::F32)
                && !denormal::device_flushes(cx.profile),
        },
        // The chain's values are flushed where `OpFmaKHR` and the multiply may keep a
        // denormal ([`with_denormals`] flushes the operands and the result); the
        // routines read and answer them as Metal does.
        FloatLowering::Dot(fma) => {
            // The first product is a multiply: Metal's flush before its rounding.
            let tiny =
                denormal::tiny_check(cx.profile).filter(|_| f.float_fmt() == Some(FloatFmt::F32));
            ArithStrategy::Dot {
                fma,
                form: f,
                flush: fma.route.device_float()
                    && f.float_fmt() == Some(FloatFmt::F32)
                    && !denormal::device_flushes(cx.profile),
                tiny: cx.families.arith.take_tiny_check(tiny, f),
            }
        }
        FloatLowering::Fn(op) => ArithStrategy::FloatFn {
            f: op,
            form: f,
            contract: contraction(fmf),
        },
    };
    // The device's instruction reads `f32` denormals as Metal does only where the table
    // says so; `OpFmaKHR` is the device's, the routines are Metal's.
    // Native device maths answers `fast::atan2` by the device's instructions, its denormals the
    // device's (`PrecisionNote::DeviceDenormals`, [`function_notes`]); nothing is flushed there.
    let native_atan2 = |op: FloatFn| {
        matches!(op, FloatFn::FastAtan2 { .. })
            && facts.native
            && libm::uses(op, f).iter().any(|r| r.native())
    };
    // Where the profile's functions are Metal's, Metal's fast `sinpi`, `cospi` and `tanpi` are the
    // device's form ([`FloatFn::FastSinPi`], `math::pi_scaled`), whose first multiply and
    // difference read the operand: as Metal does on the deviation, else flushed. For a normal
    // operand no step of it is a denormal, and it answers none.
    let device_pi = |op: FloatFn| {
        facts.metal_functions
            && matches!(
                op,
                FloatFn::FastSinPi | FloatFn::FastCosPi | FloatFn::FastTanPi
            )
    };
    // `OpFmaKHR` checked ([`fma::Fma::checked`]) answers no denormal: only its operands are read.
    let checked = |fma: fma::Fma| {
        denormal::coverage(FloatFn::Fma).map(|r| denormal::Reading {
            result: r.result && !fma.checked(),
            ..r
        })
    };
    let reading = match lowered.how {
        FloatLowering::Fma(fma) if fma.route.device_float() => checked(fma),
        FloatLowering::Fn(op) if native_atan2(op) => None,
        FloatLowering::Fn(op) if device_pi(op) => Some(denormal::Reading {
            covered: denormal::Covered::Deviation,
            operands: 0b1,
            result: false,
        }),
        FloatLowering::Fn(op) => denormal::coverage(op),
        FloatLowering::Dot(_) => denormal::coverage(FloatFn::Dot),
        // `mix` by `OpFmaKHR` reads its operands and result as its fused multiply-adds do.
        FloatLowering::Mix(fma) if fma.route.device_float() => checked(fma),
        // `fast::fmod` by `OpFmaKHR` reads its two operands and its result as `fma` does, but under
        // native device maths, which leaves them to the device; its quotient's division reads them
        // as Metal's.
        FloatLowering::FastFmod { fma, .. } if fma.route.device_float() && !facts.native => {
            checked(fma).map(|r| denormal::Reading {
                operands: 0b11,
                ..r
            })
        }
        _ => None,
    };
    let s = match reading {
        Some(reading) => with_denormals(cx, s, (f, result), args, reading),
        None => s,
    };
    Ok(with_nan(cx, at, s, lowered.nan, result))
}

/// Whether Metal's minimum and maximum function `op` on lanes of form `f` takes the device's
/// `NMin`, `NMax` or `NClamp`: `fmin`, `fmax`, `min3`, `max3`, `clamp` and `saturate`
/// ([`min_max_family`]) on `float` and `half` ([`math::metal_picks`]) where profile `p` takes
/// `DriverDeviation::MetalMinMax`.
fn device_min_max(p: &DeviceProfile, op: FloatFn, f: Form) -> bool {
    use crate::error::DriverDeviation;
    min_max_family(op)
        && math::metal_picks(f.float_fmt())
        && p.takes_driver_deviation(DriverDeviation::MetalMinMax)
}

/// Whether float function `op` is one of Metal's minimum and maximum functions, which pick one of
/// their operands: `fmin`, `fmax`, `min3`, `max3`, `clamp`, and `saturate`, which is Metal's
/// `clamp(x, +0, 1)` at every operand (a NaN, either zero and an `f32` denormal answer `+0`, as
/// `fmax` of them and `+0` answers `+0`).
fn min_max_family(op: FloatFn) -> bool {
    matches!(
        op,
        FloatFn::Min
            | FloatFn::Max
            | FloatFn::Clamp
            | FloatFn::Min3
            | FloatFn::Max3
            | FloatFn::Saturate
    )
}

/// Whether Metal's minimum and maximum function `op` on lanes of form `f` takes the device's
/// `NMin`, `NMax` or `NClamp` as they are: one of [`device_min_max`]'s functions on a profile whose
/// picks are the device's (`MinMaxRoute::Native`: not measured to be Metal's, and the policy's
/// `NanMinMaxPolicy::Native`), noted `NativeMinMax`.
fn native_min_max(p: &DeviceProfile, op: FloatFn, f: Form) -> bool {
    min_max_family(op)
        && math::metal_picks(f.float_fmt())
        && p.fields().min_max == MinMaxRoute::Native
}

/// Why the device's 16-bit `+ - *` answers as Metal's `half` arithmetic ([`f16_arithmetic`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum F16Basis {
    /// Measured (`DriverDeviation::MetalHalfArithmetic`): the module requires
    /// `VkProperty::MetalHalfArithmetic`.
    Measured,
    /// The 16-bit float controls the module's `half` roundings take ([`QuantizeForm::Controlled`],
    /// [`narrow::controlled_modes`]), which its entry then declares ([`narrow::computes_in_16_bits`]):
    /// no fact.
    Controlled,
}

/// Whether the float binary operator `op` on lanes of form `f` takes the device's 16-bit
/// arithmetic ([`ArithStrategy::F16`]), and on what basis: a `half` `fadd`, `fsub` or `fmul`, on
/// one to four lanes (a chunk's own form), where profile `p` has `shaderFloat16` and takes
/// `DriverDeviation::MetalHalfArithmetic` ([`F16Basis::Measured`]) or rounds to `half` under the
/// 16-bit float controls ([`F16Basis::Controlled`]). Under those controls the instruction is Metal's
/// by the Vulkan spec ([`ops::f16_binary`]). A `half` `fdiv` keeps the shared precise routine:
/// Metal's fast and relaxed compiles answer the device's 16-bit quotient of widened operands as a
/// `float` one, it has no form that keeps it apart, and Vulkan's precision table
/// does not make a 16-bit quotient correctly rounded. A `half` `frem` is not measured.
fn f16_arithmetic(p: &DeviceProfile, op: AirBin, f: Form) -> Option<F16Basis> {
    use crate::error::DriverDeviation;
    let applies = matches!(op, AirBin::FAdd | AirBin::FSub | AirBin::FMul)
        && f.float_fmt() == Some(FloatFmt::Half)
        && (1..=4).contains(&f.lanes)
        && p.fields().leaf.float16;
    if !applies {
        None
    } else if p.takes_driver_deviation(DriverDeviation::MetalHalfArithmetic) {
        Some(F16Basis::Measured)
    } else if narrow::quantize_form(p) == QuantizeForm::Controlled {
        Some(F16Basis::Controlled)
    } else {
        None
    }
}

/// The fast division routine a float function call of form `f` lowered as `how` calls:
/// `fast::fmod`'s and `fast::tanh`'s quotients, `fast::tan`'s, and `fast::atanh`'s, their device
/// division `NoContraction`. A `fast::tan` or `fast::atanh` that the device's own instruction
/// answers calls none, and its unused routine leaves the module with the dead functions.
fn fast_division_of(how: FloatLowering, f: Form) -> Option<math::FastDivision> {
    let reciprocal = match how {
        FloatLowering::FastFmod { reciprocal, .. } => reciprocal,
        FloatLowering::Fn(
            FloatFn::FastTanh { reciprocal }
            | FloatFn::FastTan { reciprocal }
            | FloatFn::FastAtanh { reciprocal },
        ) => reciprocal,
        FloatLowering::Fma(_)
        | FloatLowering::Mix(_)
        | FloatLowering::Dot(_)
        | FloatLowering::Fn(_) => return None,
    };
    Some(math::FastDivision::new(
        f,
        reciprocal,
        Contraction::Forbidden,
    ))
}

/// How [`select_call`] lowers a float function call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FloatLowering {
    /// A fused multiply-add.
    Fma(fma::Fma),
    /// `air.fast_fmod` on `float`, its quotient by `reciprocal`.
    FastFmod {
        fma: fma::Fma,
        reciprocal: divide::Reciprocal,
    },
    /// `air.mix` on `float` or `half`.
    Mix(fma::Fma),
    /// `air.dot` on `float` or `half`, Metal's fused chain by the fused multiply-add on one lane.
    Dot(fma::Fma),
    /// The float function.
    Fn(FloatFn),
}

/// A float function call's lowering, the precision notes it records, and the NaN Metal answers
/// that its lowering does not.
struct FloatCall {
    how: FloatLowering,
    notes: OpNotes,
    nan: Option<nan::NanAnswer>,
}

/// The facts of a profile a float function call's lowering and notes read.
#[derive(Clone, Copy, Debug)]
struct FloatFacts {
    /// The fused multiply-add routes ([`fma::routes`]).
    routes: fma::Routes,
    /// The fast reciprocal ([`divide::Reciprocal::on`]).
    reciprocal: divide::Reciprocal,
    /// Whether the profile's `float` functions are Metal's (`FloatFunctions::Metal`).
    metal_functions: bool,
    /// Whether the profile takes native device maths (`FloatFunctions::Native`).
    native: bool,
    /// Whether the profile leaves Metal's NaN to the device ([`NanEncoding::Native`]).
    nan_native: bool,
    /// Whether Metal's minimum and maximum functions are the device's as they are
    /// (`MinMaxRoute::Native`, [`native_min_max`]).
    native_min_max: bool,
    /// Whether `dot`, `mix` and `fast::sqrt` round in the device's order
    /// (`RoundingOrderRoute::Native`, [`native_chain`]).
    native_order: bool,
}

impl FloatFacts {
    /// Profile `p`'s.
    fn on(p: &DeviceProfile) -> FloatFacts {
        FloatFacts {
            routes: fma::routes(p),
            reciprocal: divide::Reciprocal::on(p),
            metal_functions: p.fields().float_functions == FloatFunctions::Metal,
            native: p.fields().float_functions == FloatFunctions::Native,
            nan_native: nan::encoding(p) == NanEncoding::Native,
            native_min_max: p.fields().min_max == MinMaxRoute::Native,
            native_order: p.fields().rounding_order == RoundingOrderRoute::Native,
        }
    }
}

/// How much of a call of float function `op` (with the `fast_` prefix where `fast`) on form `f`
/// is Metal's own in an entry Metal compiles with fast math, on a device whose `float` functions
/// are Metal's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MetalOwn {
    /// None of it: the lowering takes a routine of the translation's (a model of Metal's
    /// function the device does not answer in a fast-math module), the device answers a routine
    /// by a near-equivalent ([`libm::Fun::device_near_equivalent`]), or no `float` or `half`
    /// function is called.
    No,
    /// Its function and its division: a `fast_` call that takes no routine (`fast::rsqrt`,
    /// `fast::sqrt`), whose device instruction the module compiles as Metal's fast function, and
    /// whose reciprocal is Metal's fast one ([`divide::Reciprocal::Device`]).
    Function,
    /// Wholly: every routine the call takes the device answers in a fast-math module as Metal's
    /// own function ([`libm::Fun::device_answers`]), whose every word, denormals and NaNs among
    /// them, is Metal's (`DriverDeviation::MetalFloatFunctions`); its division is by Metal's
    /// fast reciprocal.
    Wholly,
}

impl MetalOwn {
    /// The call of `op` (`fast_` where `fast`) on form `f`.
    fn of(op: FloatFn, fast: bool, f: Form) -> MetalOwn {
        if !matches!(f.float_fmt(), Some(FloatFmt::F32 | FloatFmt::Half)) {
            return MetalOwn::No;
        }
        let uses = libm::uses(op, f);
        let own = |r: &libm::Fun| r.device_answers(libm::Mode::Fast) && !r.device_near_equivalent();
        if !uses.iter().all(own) {
            MetalOwn::No
        } else if !uses.is_empty() {
            MetalOwn::Wholly
        } else if fast && f.float_fmt() == Some(FloatFmt::F32) {
            MetalOwn::Function
        } else {
            MetalOwn::No
        }
    }
}

/// The notes of a call of float function `op` (with the `fast_` prefix where `fast`) on form `f`
/// lowered as the function itself, on a profile whose `float` functions are Metal's where
/// `metal_functions`, the device's (native device maths) where `native`, and whose `fast::sqrt`
/// rounds in the device's order where `native_order`: the function's precision
/// ([`FloatFn::precision`]) and its division's (`DeviceArithmetic` for [`FloatFn::divides`]), and
/// its `f32` denormals' and NaNs' where they are left to the device (`denormals`, `nans`,
/// [`device_float_notes`]).
///
/// A function whose answer is the one right value, Metal's on every device
/// ([`libm::Fun::correctly_rounded`]: `sqrt`, `rsqrt`, and `fmod`, precise, and their `half`
/// calls), records no precision note. Where the profile's `float` functions are Metal's, a call
/// that is Metal's own in a fast-math entry ([`MetalOwn`]) records its function's and its
/// division's notes only outside one, and, where it is wholly Metal's own, its denormals' and
/// NaNs' too ([`OpNotes`]). Of its fast functions, `fast::atan2` is Apple's formula
/// `atan(y / x)` turned by `±π` (MSL Table 8.2) of Metal's own fast `atan` and fast reciprocal,
/// and answers Metal's word on every lane of case f6524207 through MoltenVK but two, whose
/// divisors are past `2^126`, where Table 8.2 states no accuracy for the quotient: within Apple's
/// bound, which the user's fast-math sign-off accepts ("fast-math within Apple's bound",
/// 2026-10-03), so it is Metal's own too. Its fast `sinpi`, `cospi`, and `tanpi`, the device's
/// near-equivalents, keep their notes, as does every call in an entry that keeps NaNs, and every
/// routine of the translation's.
fn function_notes(
    op: FloatFn,
    fast: bool,
    f: Form,
    (metal_functions, native, native_order): (bool, bool, bool),
    (denormals, nans): (bool, bool),
) -> OpNotes {
    let uses = libm::uses(op, f);
    // Native device maths answers by the device's instruction what a routine would, and the
    // device's order `fast::sqrt`: its precision is Vulkan's, and its `f32` denormals the device's.
    let native_uses =
        (native && uses.iter().any(|r| r.native())) || (native_order && op.native_instruction());
    let exact = !native_uses && !uses.is_empty() && uses.iter().all(|r| r.correctly_rounded());
    let mut function = BTreeSet::new();
    if native_uses {
        function.insert(PrecisionNote::NativeTranscendentals);
    } else if !exact {
        function.extend(precision_note(op.precision()));
    }
    if op.divides() {
        function.insert(PrecisionNote::DeviceArithmetic);
        if native {
            function.insert(PrecisionNote::NativeDivision);
        }
    }
    let denormals = denormals || native_uses;
    let environment: BTreeSet<_> = device_float_notes(f, denormals, nans).collect();
    let own = if metal_functions {
        MetalOwn::of(op, fast, f)
    } else {
        MetalOwn::No
    };
    OpNotes::unless_own(function, own != MetalOwn::No)
        .with(OpNotes::unless_own(environment, own == MetalOwn::Wholly))
}

/// Whether the device's instruction (`OpDot`, `FMix`) answers Metal's fused chain of `dot` or
/// `mix` with fused multiply-add `fma` on lanes of form `f`: on a profile that rounds in the
/// device's order (`RoundingOrderRoute::Native`), a `float` chain, and a `half` one whose fused
/// multiply-add is not the measured 16-bit route ([`fma::Route::F16`], Metal's own `half` `fma`,
/// which the chain keeps).
fn native_chain(facts: FloatFacts, fma: fma::Fma, f: Form) -> bool {
    facts.native_order
        && match f.float_fmt() {
            Some(FloatFmt::F32) => true,
            Some(FloatFmt::Half) => fma.route != fma::Route::F16,
            _ => false,
        }
}

/// The note of fused multiply-add `fma`: `NativeFma` where it takes native device maths' `Fma`
/// ([`fma::Route::Native`]), which [`finish`] records for every such route a family takes.
fn native_fma(fma: fma::Fma) -> OpNotes {
    match fma.route {
        fma::Route::Native => OpNotes::everywhere(BTreeSet::from([PrecisionNote::NativeFma])),
        _ => OpNotes::default(),
    }
}

/// The lowering of float function `op` called with the `fast_` prefix if `fast` and with
/// fast-math flags `fmf`, on operands of form `f`, on a profile with float facts `facts`
/// ([`FloatFacts`]); `None` where the form refuses the function (it is not a float, or a `double`
/// the function does not take). Where the profile leaves NaNs to the device a Metal NaN the call
/// answers is the device's, noted `NativeNanEncoding` (as [`with_nan`] records it), and where its
/// picks are the device's Metal's minimum and maximum functions are the device's
/// ([`native_min_max`]), noted `NativeMinMax`.
fn float_call(
    op: FloatFn,
    fast: bool,
    fmf: FastMath,
    f: Form,
    facts: FloatFacts,
) -> Option<FloatCall> {
    let mut call = float_call_of(op, fast, fmf, f, facts)?;
    let mut native = BTreeSet::new();
    if call.nan.is_some() && facts.nan_native {
        native.insert(PrecisionNote::NativeNanEncoding);
    }
    if let FloatLowering::Fn(op) = call.how {
        let picks = min_max_family(op) && math::metal_picks(f.float_fmt());
        if picks && facts.native_min_max {
            native.insert(PrecisionNote::NativeMinMax);
        }
    }
    call.notes = call.notes.with(OpNotes::everywhere(native));
    Some(call)
}

/// [`float_call`] before native device maths' NaN and minimum and maximum notes.
fn float_call_of(
    op: FloatFn,
    fast: bool,
    fmf: FastMath,
    f: Form,
    facts: FloatFacts,
) -> Option<FloatCall> {
    if !f.is_float() || (f.bits() == 64 && !op.allows_f64()) {
        return None;
    }
    let FloatFacts {
        routes,
        reciprocal,
        metal_functions,
        ..
    } = facts;
    let op = refine_float(op, fast, fmf, f, reciprocal);
    // `float` and `half` fused multiply-adds round once on every profile; a `bfloat` one corrects
    // its tie, and a `double` one is `GLSL.std.450 Fma` at the device's precision. `llvm.fmuladd`
    // on `float` and `half` is the fused one too: LLVM lets the implementation choose, and Metal
    // fuses it (a probe case f7850ba7, every lane), so its value is exact and needs no
    // `FastMathRewrites`.
    if matches!(op, FloatFn::Fma | FloatFn::MulAdd) {
        if let Some(fma) = fma::Fma::by(routes, f) {
            // `OpFmaKHR` is the device's at denormals and answers the device's NaN, so Metal's
            // reading of its denormals ([`with_denormals`]) and Metal's NaN are selected; the
            // `float` routines are Metal's at both. A `half` result's rounding answers the device's
            // `half` NaN on every route, so Metal's is selected where it is measured.
            let khr = fma.route.device_float();
            let nan = match f.float_fmt() {
                Some(FloatFmt::Half) => {
                    nan::function_answer(op, fast, every_flag(fmf), FloatFmt::Half)
                }
                _ => khr.then_some(nan::NanAnswer::Canonical),
            };
            return Some(FloatCall {
                how: FloatLowering::Fma(fma),
                notes: native_fma(fma),
                nan,
            });
        }
    }
    // Metal's `mix(x, y, t)` on `float` and `half` is `fma(t, y, fma(-t, x, x))`, with and without
    // fast-math flags, so both fused multiply-adds take the profile's route. The routines answer
    // Metal's `float` NaN; `OpFmaKHR` answers the device's, which is selected, and a `half`
    // result's as the `half` `fma`'s. Each reads an `f32` denormal as Metal's `fma` does: the
    // routines themselves, `OpFmaKHR` on the deviation or flushed ([`ArithStrategy::Mix`]'s
    // `flush`, [`with_denormals`]).
    if op == FloatFn::Mix {
        if let Some(fma) = fma::Fma::by(routes, f) {
            // The device's order takes its `FMix` where Metal's fused form would be emulated:
            // on `float`, and on `half` where its fused multiply-add is not the measured 16-bit
            // one, rounded once (`NativeMix`).
            if native_chain(facts, fma, f) {
                return Some(FloatCall {
                    how: FloatLowering::Fn(op),
                    notes: OpNotes::everywhere(
                        device_float_notes(f, true, false)
                            .chain([PrecisionNote::NativeMix])
                            .collect(),
                    ),
                    nan: f.float_fmt().and_then(|fmt| match fmt {
                        FloatFmt::Half => {
                            nan::function_answer(op, fast, every_flag(fmf), FloatFmt::Half)
                        }
                        _ => Some(nan::NanAnswer::Canonical),
                    }),
                });
            }
            let khr = fma.route.device_float();
            let nan = match f.float_fmt() {
                Some(FloatFmt::Half) => {
                    nan::function_answer(op, fast, every_flag(fmf), FloatFmt::Half)
                }
                _ => khr.then_some(nan::NanAnswer::Canonical),
            };
            return Some(FloatCall {
                how: FloatLowering::Mix(fma),
                notes: native_fma(fma),
                nan,
            });
        }
    }
    // Metal's `dot` on `float` and `half` is the products' fused chain, with and without
    // fast-math flags, so each lane after the first takes the profile's route on one lane: the
    // device's `OpDot` may sum otherwise (a tree, an unfused sum, or, under fast-math flags, `b *
    // (a + c)` for `dot(float2(a, b), float2(b, c))`). Where the profile's `float` functions are
    // Metal's, a `float` `OpDot` reaches Metal's own `dot`, the same chain; a `half` there would be
    // summed in its `float` container, so it takes the chain. Either is Metal's answer at denormals
    // and records no note; Metal's NaN is selected as for any function.
    if op == FloatFn::Dot {
        if let Some(fma) = fma::Fma::by(routes, f.scalar_form()) {
            let nan = f
                .float_fmt()
                .and_then(|fmt| nan::function_answer(op, fast, every_flag(fmf), fmt));
            // The device's order takes its `OpDot` where Metal's chain would be emulated: on
            // `float`, and on `half` where its fused multiply-add is not the measured 16-bit one,
            // rounded once (`NativeDot`).
            if native_chain(facts, fma, f.scalar_form()) {
                return Some(FloatCall {
                    how: FloatLowering::Fn(op),
                    notes: OpNotes::everywhere(BTreeSet::from([PrecisionNote::NativeDot])),
                    nan,
                });
            }
            let how = if metal_functions && f.float_fmt() == Some(FloatFmt::F32) {
                FloatLowering::Fn(op)
            } else {
                FloatLowering::Dot(fma)
            };
            return Some(FloatCall {
                how,
                notes: native_fma(fma),
                nan,
            });
        }
    }
    // `air.fast_fmod` on `float` is Metal's composition around a fused multiply-add; the fast-math
    // table is single precision, so a narrow call keeps `fmod`. Where the profile's `float`
    // functions are Metal's, its `fma` is Metal's own and its quotient's reciprocal Metal's fast
    // one in a fast-math entry ([`divide::Reciprocal::Device`]), so its notes hold only outside
    // one. Metal reads a denormal operand as the zero of its sign and answers no denormal, in both
    // modes: the routines read them so, and the device's fused multiply-add on the
    // deviation or flushed ([`with_denormals`]) where the profile asks Metal's functions; native
    // device maths leaves them to the device's (`DeviceDenormals`), unflushed.
    if fast && op == FloatFn::Fmod && f.float_fmt() == Some(FloatFmt::F32) {
        if let Some(fma) = fma::Fma::by(routes, f) {
            let khr = fma.route.device_float();
            let mut notes = BTreeSet::from([PrecisionNote::DeviceArithmetic]);
            if khr && facts.native {
                notes.extend(device_float_notes(f, true, false));
            }
            return Some(FloatCall {
                how: FloatLowering::FastFmod { fma, reciprocal },
                notes: OpNotes::unless_own(notes, metal_functions && khr).with(native_fma(fma)),
                nan: khr.then_some(nan::NanAnswer::Canonical),
            });
        }
    }
    // The NaN Metal answers, selected where the lowering does not answer it itself.
    let answer = f
        .float_fmt()
        .and_then(|fmt| nan::function_answer(op, fast, every_flag(fmf), fmt));
    // Native device maths' fast quotient is the device's, NaN included.
    let native_quotient = facts.native && matches!(op, FloatFn::FastTanh { .. });
    let metal_nans = f.float_fmt().is_some_and(|fmt| op.metal_nans(fmt)) && !native_quotient;
    let nan = answer.filter(|_| !metal_nans);
    // A function whose denormals Metal's table reads ([`denormal::coverage`]) reads them as
    // Metal does: the device's, where the profile states it does, else flushed
    // ([`with_denormals`]). So do Metal's fast `sinpi`, `cospi` and `tanpi` (a
    // denormal operand read as a zero, no denormal answer): the routines read the operand by
    // `denormal::operand`, and the device's form where the profile's functions are Metal's reads
    // it as `select_float_fn` says. Native device maths records the note ([`function_notes`]).
    let fast_pi = matches!(
        op,
        FloatFn::FastSinPi | FloatFn::FastCosPi | FloatFn::FastTanPi
    );
    let mut notes = function_notes(
        op,
        fast,
        f,
        (metal_functions, facts.native, facts.native_order),
        (
            !op.metal_denormals() && denormal::coverage(op).is_none() && !fast_pi,
            !metal_nans && answer.is_none(),
        ),
    );
    // A `bfloat` or `double` `llvm.fmuladd` is fused or not, as the implementation chooses.
    if op == FloatFn::MulAdd {
        notes.always.insert(PrecisionNote::FastMathRewrites);
    }
    Some(FloatCall {
        how: FloatLowering::Fn(op),
        notes,
        nan,
    })
}

/// The precision notes this family's lowering records for the `air` math call `call` with
/// fast-math flags `fmf` on profile `p`: exactly what [`select`] records for the call itself
/// (its operands' and result's own instructions record theirs; a `half` result's rounding is
/// noted from the emitted module: Metal's under `HalfRoundingPolicy::Exact` and where measured,
/// native half rounding's elsewhere). `None` where the family does not lower `call` as a float
/// function: an integer function, `sincos` (selected with its store), or a function its type
/// refuses.
///
/// Validation reads it, through the AIR symbol Metal compiles an MSL function to, to tell which
/// operations the product computes as a near-equivalent of Metal's, as
/// [`PrecisionNote::of_math_call`].
pub fn float_call_notes(call: Math, fmf: FastMath, p: &DeviceProfile) -> Option<OpNotes> {
    let Ok(Fun::Float(op)) = air_math(call.f) else {
        return None;
    };
    let held = Held::Float(match call.ty.elem {
        Elem::F16 => FloatFmt::Half,
        Elem::BF16 => FloatFmt::BFloat,
        Elem::F32 => FloatFmt::F32,
        _ => return None,
    });
    let f = Form {
        held,
        lanes: u8::try_from(call.ty.lanes).ok()?,
    };
    let fast = call.precision == crate::air::intrinsics::Precision::Fast;
    float_call(op, fast, fmf, f, FloatFacts::on(p)).map(|lowered| lowered.notes)
}

/// The mode of the `float` functions of an entry of math mode `math` on profile `p`
/// ([`ArithPlan::metal_functions`]): Metal's in that mode where the profile's are Metal's,
/// `None` elsewhere.
fn metal_functions_in(p: &DeviceProfile, math: EntryMath) -> Option<libm::Mode> {
    (p.fields().float_functions == FloatFunctions::Metal).then_some(match math {
        EntryMath::Fast => libm::Mode::Fast,
        EntryMath::Precise => libm::Mode::Precise,
    })
}

impl PrecisionNote {
    /// The precision notes a compile for profile `p` records for the AIR float operator `op`
    /// (`fadd`, `fsub`, `fmul`, `fdiv` or `frem`) on lanes of `width`, with fast-math flags
    /// `fmf`, in an entry of math mode `math`: what a module states for the operator itself. Its
    /// operands' instructions record their own, and a `half` result's rounding is noted from the
    /// emitted module. `None` for an integer operator. The precision contract a tool reads to tell
    /// which operations the crate computes as a near-equivalent of Metal's.
    pub fn of_float_operator(
        op: AirBin,
        width: FloatWidth,
        fmf: FastMath,
        p: &DeviceProfile,
        math: EntryMath,
    ) -> Option<BTreeSet<PrecisionNote>> {
        let fmt = match width {
            FloatWidth::F16 => FloatFmt::Half,
            FloatWidth::F32 => FloatFmt::F32,
            FloatWidth::F64 => FloatFmt::F64,
        };
        float_binop_notes(op, fmt, fmf, p).map(|notes| notes.in_entry(metal_functions_in(p, math)))
    }

    /// The precision notes a compile for profile `p` records for the AIR math call `call` (an
    /// `air.*` function, as `metal2vulkan_air::intrinsics::AirIntrinsic::parse` decodes its
    /// symbol) with fast-math flags `fmf`, in an entry of math mode `math`: the notes for the call
    /// itself. `None` where the crate does not lower `call` as a float function: an integer
    /// function, `sincos` (lowered with its store), or a type it refuses.
    pub fn of_math_call(
        call: Math,
        fmf: FastMath,
        p: &DeviceProfile,
        math: EntryMath,
    ) -> Option<BTreeSet<PrecisionNote>> {
        float_call_notes(call, fmf, p).map(|notes| notes.in_entry(metal_functions_in(p, math)))
    }
}

/// `air.sincos(x, ptr)`: the sine is the result and the cosine is stored through `ptr` (a thread
/// pointer to the result's type). The store is the access family's: it is selected as the store
/// of the call's own result through `ptr` (the store the call performs, with the type the
/// signature gives it and the type's ABI alignment, which a `thread T&` has), and emitted with
/// the cosine bound to the result while it runs.
pub(super) fn select_sincos(
    cx: &mut SelectCx<'_>,
    at: InstId,
    inst: &Inst,
    call: &crate::air::ir::Call,
    m: &mut Meter,
) -> Result<ArithStrategy, Error> {
    let (Some(&x), Some(&ptr)) = (call.args.first(), call.args.get(1)) else {
        return Err(cx.invariant("air.sincos without its operands"));
    };
    check_operands(cx, &[x])?;
    lowered_type(cx.entry, cx.profile, inst.ty)?;
    let shape = chunk::shape_of(cx, ty_of(cx, x)?)?;
    let f = shape.lane_form();
    if !f.is_float() || (f.bits() == 64 && !FloatFn::Sin.allows_f64()) {
        return Err(not_lowered());
    }
    // `fast::sincos` answers as `fast::sin` and `fast::cos` do, `+0` for both past the last
    // evaluated quadrant (a probe cases 914cc4d2 and 711cc190, every lane); the fast-math table is
    // single precision, so a narrow call keeps the precise functions.
    let symbol_fast = matches!(
        &call.callee,
        crate::air::ir::Callee::Air {
            op: AirIntrinsic::Math(math),
            ..
        } if math.precision == crate::air::intrinsics::Precision::Fast
    );
    let fast = symbol_fast && f.float_fmt() == Some(FloatFmt::F32);
    let (sin, cos) = chunk::sin_cos_fns(fast);
    // Each result takes the NaN its function answers as a call of its own (`float_call`'s rule):
    // Metal's `sincos` answers its `sin` and `cos`, word for word (cases 7b283404 and f6524207,
    // every pair), so where Metal's answer is measured it is selected; otherwise the device's NaN
    // is noted. The cosine is stored, so its bits are observed. Each records its notes as a call of
    // its own (`function_notes`).
    let metal_functions = cx.profile.fields().float_functions == FloatFunctions::Metal;
    let mut nans = [None; 2];
    for (slot, (op, stored)) in nans.iter_mut().zip([(sin, false), (cos, true)]) {
        let answer = f
            .float_fmt()
            .and_then(|fmt| nan::function_answer(op, symbol_fast, every_flag(call.fmf), fmt));
        let metal_nans = f.float_fmt().is_some_and(|fmt| op.metal_nans(fmt));
        let native = cx.profile.fields().float_functions == FloatFunctions::Native;
        let native_order = cx.profile.fields().rounding_order == RoundingOrderRoute::Native;
        let notes = function_notes(
            op,
            fast,
            f,
            (metal_functions, native, native_order),
            (!op.metal_denormals(), !metal_nans && answer.is_none()),
        );
        record(cx, notes);
        let observed = stored || cx.nan_payload_observed(at);
        note_native_nan(cx, answer.filter(|_| !metal_nans));
        *slot = answer.filter(|a| !metal_nans && !a.left_to(nan::encoding(cx.profile)) && observed);
        cx.families
            .arith
            .libm_uses
            .extend(libm::uses(op, f).iter().copied());
    }
    fp_env(cx, call.fmf, f);
    let align = cx
        .memory
        .layouts
        .abi_align(inst.ty)
        .and_then(crate::air::ir::Align::from_bytes);
    let store = Inst {
        ty: inst.ty,
        kind: InstKind::Store {
            val: Value::Inst(at),
            ptr,
            align,
            volatile: false,
            atomic: None,
        },
    };
    let store = access::select(cx, at, &store, m)?;
    Ok(ArithStrategy::SinCos {
        shape,
        store: Box::new(store),
        fast,
        nan: nans,
    })
}

fn select_pack(
    cx: &mut SelectCx<'_>,
    inst: &Inst,
    call: &crate::air::ir::Call,
    p: &Pack,
) -> Result<ArithStrategy, Error> {
    // The float lane count; packing to a packed float format is not lowered.
    let lanes = match (
        math::pack_layout(p.format),
        math::float_unpack_lanes(p.format),
    ) {
        (Some((widths, _)), _) => widths.len(),
        (None, Some(n)) if p.dir == PackDir::Unpack => usize::from(n),
        _ => return Err(not_lowered()),
    };
    let x = *call
        .args
        .first()
        .ok_or_else(|| cx.invariant("air.pack without an operand"))?;
    check_operands(cx, &[x])?;
    let (floats, packed) = match p.dir {
        PackDir::Pack => (form(cx, x)?, result_form(cx, inst)?),
        PackDir::Unpack => (result_form(cx, inst)?, form(cx, x)?),
    };
    let word = Form {
        held: Held::Int {
            bits: math::packed_bits(p.format),
            container: 32,
        },
        lanes: 1,
    };
    if !floats.is_float() || usize::from(floats.lanes) != lanes || packed != word {
        return Err(cx.invariant("an air.pack whose types disagree with its format"));
    }
    // Every conversion is exact: the sRGB ones read Metal's table.
    if p.format == PackFormat::Unorm4x8Srgb {
        cx.families.arith.srgb.get_or_insert(p.dir);
    }
    Ok(ArithStrategy::Pack {
        dir: p.dir,
        format: p.format,
        floats,
    })
}

fn select_convert(
    cx: &mut SelectCx<'_>,
    at: InstId,
    inst: &Inst,
    call: &crate::air::ir::Call,
    c: &Convert,
    chunk: Option<u8>,
) -> Result<ArithStrategy, Error> {
    let x = *call
        .args
        .first()
        .ok_or_else(|| cx.invariant("air.convert without an operand"))?;
    check_operands(cx, &[x])?;
    let from = form_in(cx, x, chunk)?;
    let to = result_form_in(cx, inst, chunk)?;
    // An integer to `bool` is C++'s `x != 0`.
    if to.is_bool() && c.src_kind != NumKind::Float {
        if !from.is_int() || from.lanes != to.lanes {
            return Err(not_lowered());
        }
        return Ok(ArithStrategy::Convert {
            kind: ConvertKind::IntToBool,
            from,
            to,
        });
    }
    // A `float` or `half` to `bool` is Metal's measured reading. No reading exists of a `bfloat`
    // (whose denormals are `f32` denormals) or `double` source.
    if to.is_bool() {
        let measured = matches!(from.float_fmt(), Some(FloatFmt::F32 | FloatFmt::Half));
        if c.src_kind != NumKind::Float || !measured || from.lanes != to.lanes {
            return Err(not_lowered());
        }
        return Ok(ArithStrategy::Convert {
            kind: ConvertKind::ToBool,
            from,
            to,
        });
    }
    if let Some(fmt) = c.src_fp8 {
        let fp8 = narrow::Fp8Layout::of(fmt);
        let to_int = c.dst_kind != NumKind::Float && to.is_int();
        if from.bits() != 8 || !(to.is_float() || to_int) || from.lanes != to.lanes {
            return Err(cx.invariant("an 8-bit float air.convert whose types disagree"));
        }
        let kind = if to_int {
            // The format's documented value, not read on any Metal device.
            cx.precision.insert(PrecisionNote::Fp8Value);
            ConvertKind::Fp8ToInt {
                fp8,
                signed: c.dst_kind == NumKind::Signed,
            }
        } else {
            ConvertKind::FromFp8(fp8)
        };
        return Ok(ArithStrategy::Convert { kind, from, to });
    }
    let kind = match (c.dst_kind, c.src_kind) {
        (NumKind::Float, NumKind::Float) => ConvertKind::Float,
        (NumKind::Float, s) => ConvertKind::ToFloat {
            signed: s == NumKind::Signed,
        },
        (d, NumKind::Float) => ConvertKind::ToInt {
            signed: d == NumKind::Signed,
        },
        (_, s) => ConvertKind::Int {
            signed: s == NumKind::Signed,
        },
    };
    let consistent = match kind {
        ConvertKind::Float => from.is_float() && to.is_float(),
        ConvertKind::ToFloat { .. } => !from.is_float() && to.is_float(),
        ConvertKind::ToInt { .. } => from.is_float() && to.is_int(),
        ConvertKind::Int { .. } => !from.is_float() && to.is_int(),
        ConvertKind::FromFp8(_)
        | ConvertKind::Fp8ToInt { .. }
        | ConvertKind::ToBool
        | ConvertKind::IntToBool => false,
    };
    if !consistent || from.lanes != to.lanes {
        return Err(cx.invariant("an air.convert whose kinds disagree with its types"));
    }
    let s = ArithStrategy::Convert { kind, from, to };
    // Metal's widening of a `half` NaN, as `fpext`'s.
    let nan = match (kind, from.float_fmt(), to.float_fmt()) {
        (ConvertKind::Float, Some(a), Some(b)) => nan::convert_answer(a, b),
        _ => None,
    };
    Ok(with_nan(cx, at, s, nan, to))
}

// ---- Emission ----

fn operand(cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>, v: Value) -> BResult<SValue> {
    match cx.plan.values.id(v) {
        Some(id) => cx.operand(fb, id),
        None => Err(cx.fail("an arith operand out of the value range")),
    }
}

/// The SPIR-V values of the first `n` operands (all when `None`) of instruction `inst`.
fn operand_values(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    inst: &Inst,
    n: Option<usize>,
) -> BResult<Vec<SValue>> {
    let vals: Vec<Value> = match &inst.kind {
        InstKind::Call(call) => call.args.to_vec(),
        kind => operands_of(kind),
    };
    let take = n.unwrap_or(vals.len()).min(vals.len());
    let mut out = Vec::with_capacity(take);
    for v in vals.into_iter().take(take) {
        out.push(operand(cx, fb, v)?);
    }
    Ok(out)
}

fn nth(cx: &mut EmitCx<'_>, xs: &[SValue], i: usize) -> BResult<SValue> {
    match xs.get(i) {
        Some(x) => Ok(*x),
        None => Err(cx.fail("an arith strategy with too few operands")),
    }
}

/// Emit strategy `s` of instruction `at` into the open block.
pub fn emit(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    at: InstId,
    s: &ArithStrategy,
) -> BResult<()> {
    use ArithStrategy as S;
    let body = cx.body;
    let Some(inst) = body.insts.get(at) else {
        return Err(cx.fail("an arith strategy names a missing instruction"));
    };
    let Some(result) = cx.plan.values.inst(at) else {
        return Err(cx.fail("an arith instruction out of the value range"));
    };
    let xs = operand_values(cx, fb, inst, arity(s))?;
    let x = match s {
        // Its only uses read its operand: it has no value of its own.
        S::IndexExtension => return Ok(()),
        // Its additions read its factors.
        S::Folded => return Ok(()),
        // No SPIR-V value: its extractions resolve through provenance.
        S::PointerAggregate => return Ok(()),
        S::Handle(k) => crate::memory::leaf::k32(fb, *k)?,
        S::Forward(v) => cx.operand(fb, *v)?,
        S::Null(d) => aggregate::null_value(cx, fb, d)?,
        S::Member { of, path, bools } => {
            let x = cx.operand(fb, *of)?;
            let x = fb.composite_extract(x, path)?;
            match bools {
                Some(lanes) => ops::member_to_bools(fb, x, *lanes)?,
                None => x,
            }
        }
        S::IndexWord(w) => access::index::emit_word(cx, fb, w)?,
        S::SinCos {
            shape,
            store,
            fast,
            nan,
        } => {
            let p = nth(cx, &xs, 0)?;
            let test = nan::NanTest::for_modes(&cx.plan.families.arith.modes);
            let lib = lib(cx);
            let (sin, cos) = chunk::sin_cos(fb, *shape, p, *fast, *nan, test, &lib)?;
            // The access family's store reads the stored value as the call's result.
            cx.define(result, cos);
            access::emit(cx, fb, at, store)?;
            sin
        }
        _ => value(cx, fb, s, &xs)?,
    };
    cx.define(result, x);
    Ok(())
}

/// The math-library routines of the entry being emitted.
pub(crate) fn lib<'a>(cx: &EmitCx<'a>) -> libm::Lib<'a> {
    let arith = &cx.plan.families.arith;
    let funcs = &cx.decl.families.arith.libm;
    let lib = if arith.native {
        libm::Lib::native(funcs)
    } else {
        libm::Lib::new(arith.metal_functions, funcs)
    };
    lib.with_min_max(arith.device_min_max || arith.native_min_max)
        .with_native_order(arith.native_order)
}

/// The function of the plan's fast division routine `r`.
fn fast_division(cx: &mut EmitCx<'_>, r: math::FastDivision) -> BResult<FuncId> {
    match cx.decl.families.arith.fast.get(&r) {
        Some(id) => Ok(*id),
        None => Err(cx.fail("a fast division whose routine is not declared")),
    }
}

/// The operands strategy `s` reads (a prefix of the instruction's; all when `None`).
fn arity(s: &ArithStrategy) -> Option<usize> {
    use ArithStrategy as S;
    match s {
        S::IntFn { f, .. } => Some(f.arity()),
        S::FloatFn { f, .. } => Some(f.arity()),
        // A constant element index is a literal, never materialized.
        S::Copy
        | S::Convert { .. }
        | S::Pack { .. }
        | S::ExtractConst(_)
        | S::SinCos { .. }
        | S::WordBitcast { .. }
        | S::LaneExtract { at: Some(_), .. } => Some(1),
        S::InsertConst(_)
        | S::LaneInsert { at: Some(_), .. }
        | S::ChunkedDot { .. }
        | S::ChunkedChain { .. }
        | S::Dot { .. } => Some(2),
        // An index word reads its operands itself (an index extension has no value).
        S::Zero(_)
        | S::Undef(_)
        | S::IndexExtension
        | S::IndexWord(_)
        | S::Folded
        | S::Fused { .. }
        | S::PointerAggregate
        | S::Handle(_)
        | S::Forward(_)
        | S::Null(_)
        | S::Member { .. } => Some(0),
        S::Chunked { parts, .. } => parts.first().and_then(|(_, p)| arity(p)),
        S::Wide(w) => w.arity(),
        S::Nan { of, .. } | S::Denormals { of, .. } | S::TinyProduct { of, .. } => arity(of),
        _ => None,
    }
}

/// The value of strategy `s` on operand values `xs` (every strategy but the ones [`emit`]
/// handles itself).
fn value(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    s: &ArithStrategy,
    xs: &[SValue],
) -> BResult<SValue> {
    use ArithStrategy as S;
    let a = |cx: &mut EmitCx<'_>, i| nth(cx, xs, i);
    Ok(match s {
        S::IndexExtension
        | S::IndexWord(_)
        | S::Folded
        | S::SinCos { .. }
        | S::PointerAggregate
        | S::Handle(_)
        | S::Forward(_)
        | S::Null(_)
        | S::Member { .. } => return Err(cx.fail("a strategy with no value of its own")),
        S::Chunked {
            lanes,
            parts,
            wide,
            result,
        } => chunk::emit_chunked(cx, fb, *lanes, parts, wide, result, xs)?,
        S::Nan { of, answer, form } => {
            let r = value(cx, fb, of, xs)?;
            let test = nan::NanTest::for_modes(&cx.plan.families.arith.modes);
            nan::select(fb, *form, *answer, test, r, xs.first().copied())?
        }
        S::Denormals {
            of,
            form,
            out,
            flush,
        } => {
            let mut ys = xs.to_vec();
            for (i, y) in ys.iter_mut().enumerate() {
                if flush.operand(i) {
                    *y = denormal::flush(fb, *form, *y)?;
                }
            }
            let r = value(cx, fb, of, &ys)?;
            if flush.result {
                denormal::flush(fb, *out, r)?
            } else {
                r
            }
        }
        S::TinyProduct { of, form, check } => {
            let r = value(cx, fb, of, xs)?;
            let ab = match xs {
                [a, b, ..] => [*a, *b],
                _ => return Err(cx.fail("a product without its operands")),
            };
            denormal::tiny_product(fb, *form, ab, r, *check)?
        }
        S::ChunkedDot {
            lanes,
            elem,
            contract,
        } => chunk::dot(fb, *lanes, *elem, *contract, a(cx, 0)?, a(cx, 1)?)?,
        S::ChunkedChain {
            lanes,
            elem,
            fma: f,
            flush,
            reads,
            tiny,
        } => {
            let routine = match f.routine() {
                Some(r) => match cx.decl.families.arith.fmas.get(&r) {
                    Some(id) => Some(*id),
                    None => return Err(cx.fail("a dot whose fused multiply-add is not declared")),
                },
                None => None,
            };
            let fma =
                |fb: &mut FunctionBuilder<'_>, args: &[SValue]| fma::emit(fb, *f, args, routine);
            let xy = [a(cx, 0)?, a(cx, 1)?];
            chunk::dot_chain(fb, *lanes, *elem, xy, &fma, (*flush, *tiny), *reads)?
        }
        S::LaneExtract { src, at, index } => chunk::extract(fb, *src, *at, *index, xs)?,
        S::LaneInsert { dst, at, index } => chunk::insert(fb, *dst, *at, *index, xs)?,
        S::LaneShuffle {
            a,
            b,
            out,
            mask,
            fragment: false,
        } => chunk::shuffle(fb, *a, *b, *out, mask, xs)?,
        S::LaneShuffle {
            a,
            b,
            out,
            mask,
            fragment: true,
        } => chunk::shuffle_fragment(fb, *a, *b, *out, mask, xs)?,
        S::WordBitcast { from, to } => {
            let p = nth(cx, xs, 0)?;
            chunk::bitcast(fb, *from, *to, p)?
        }
        S::Int { op, form } => {
            let (p, q) = (a(cx, 0)?, a(cx, 1)?);
            ops::int_binary(fb, *op, *form, p, q)?
        }
        S::Bool { op, form } => {
            let (p, q) = (a(cx, 0)?, a(cx, 1)?);
            ops::bool_binary(fb, *op, *form, p, q)?
        }
        S::Float { op, form, contract } => {
            let (p, q) = (a(cx, 0)?, a(cx, 1)?);
            ops::float_binary(fb, *op, *form, *contract, p, q)?
        }
        S::F16 { op, form } => {
            let (p, q) = (a(cx, 0)?, a(cx, 1)?);
            ops::f16_binary(fb, *op, form.lanes, p, q)?
        }
        S::FastDivide { routine, .. } => {
            let (p, q) = (a(cx, 0)?, a(cx, 1)?);
            if cx.plan.families.arith.device_quotient() {
                math::device_fast_division(fb, *routine, p, q)?
            } else {
                let id = fast_division(cx, *routine)?;
                math::call_fast_division(fb, id, p, q)?
            }
        }
        S::NativeDivide { form } => {
            let (p, q) = (a(cx, 0)?, a(cx, 1)?);
            divide::native(fb, *form, p, q)?
        }
        S::PreciseDivide { form, routine } => {
            let (p, q) = (a(cx, 0)?, a(cx, 1)?);
            let Some(id) = cx.decl.families.arith.precise.get(routine).copied() else {
                return Err(cx.fail("a precise division whose routine is not declared"));
            };
            divide::emit(fb, *form, p, q, id)?
        }
        S::Remainder {
            form,
            routine,
            flush,
        } => {
            let (x, y) = (a(cx, 0)?, a(cx, 1)?);
            let Some(id) = cx.decl.families.arith.precise.get(routine).copied() else {
                return Err(cx.fail("a remainder whose division routine is not declared"));
            };
            let divide = |fb: &mut FunctionBuilder<'_>, p: SValue, q: SValue| {
                divide::emit(fb, *form, p, q, id)
            };
            math::metal_frem(fb, *form, (x, y), &divide, *flush)?
        }
        S::FNeg { form, bits } => {
            let p = a(cx, 0)?;
            if *bits {
                math::sign_bits(fb, *form, p, math::SignBit::Flip)?
            } else {
                fb.unary(UnOp::FNegate, p)?
            }
        }
        S::IntCompare { pred, form } => {
            let (p, q) = (a(cx, 0)?, a(cx, 1)?);
            ops::int_compare(fb, *pred, *form, p, q)?
        }
        S::BoolCompare { pred } => {
            let (p, q) = (a(cx, 0)?, a(cx, 1)?);
            ops::bool_compare(fb, *pred, p, q)?
        }
        S::FloatCompare { pred, form } => {
            let (p, q) = (a(cx, 0)?, a(cx, 1)?);
            ops::float_compare(fb, *pred, *form, p, q)?
        }
        S::Cast { kind, from, to } => {
            let p = a(cx, 0)?;
            ops::cast(fb, *kind, *from, *to, p)?
        }
        S::Bitcast { plan, from, to } => {
            let p = a(cx, 0)?;
            ops::bitcast(fb, *plan, *from, *to, p)?
        }
        S::Select => {
            let (c, t, f) = (a(cx, 0)?, a(cx, 1)?, a(cx, 2)?);
            fb.select(c, t, f)?
        }
        S::Copy => a(cx, 0)?,
        S::Zero(d) => match legal::zero(d) {
            Some(k) => match legal::materialize(fb, &k)? {
                Some(x) => x,
                None => return Err(cx.fail("a zero arith does not materialize")),
            },
            None => return Err(cx.fail("a zero of a non-arith type")),
        },
        S::Undef(d) => match legal::spirv_type(fb.module(), d)? {
            Some(t) => fb.undef(t)?,
            None => return Err(cx.fail("an undef of a non-arith type")),
        },
        S::ExtractConst(i) => {
            let v = a(cx, 0)?;
            let y = fb.composite_extract(v, &[*i])?;
            // A store of a narrow lane takes its bits from its vector's.
            if narrow::may_be_narrow(fb, y)? {
                fb.note_lane(y, v, *i);
            }
            y
        }
        S::ExtractDynamic { index } => {
            let (v, i) = (a(cx, 0)?, a(cx, 1)?);
            let i = ops::element_index(fb, i, *index)?;
            fb.vector_extract_dynamic(v, i)?
        }
        S::InsertConst(i) => {
            let (v, e) = (a(cx, 0)?, a(cx, 1)?);
            // Narrow lanes' bits follow the insert.
            narrow::insert(fb, v, e, *i)?
        }
        S::InsertDynamic { index } => {
            let (v, e, i) = (a(cx, 0)?, a(cx, 1)?, a(cx, 2)?);
            let i = ops::element_index(fb, i, *index)?;
            fb.vector_insert_dynamic(v, e, i)?
        }
        S::Shuffle(c) => {
            let (p, q) = (a(cx, 0)?, a(cx, 1)?);
            fb.vector_shuffle(p, q, c)?
        }
        S::ExtractValue(idx) => {
            let v = a(cx, 0)?;
            fb.composite_extract(v, idx)?
        }
        S::InsertValue(idx) => {
            let (agg, v) = (a(cx, 0)?, a(cx, 1)?);
            fb.composite_insert(v, agg, idx)?
        }
        S::ExtractBools { path, lanes } => {
            let v = a(cx, 0)?;
            let x = fb.composite_extract(v, path)?;
            ops::member_to_bools(fb, x, *lanes)?
        }
        S::InsertBools { path, lanes } => {
            let (agg, v) = (a(cx, 0)?, a(cx, 1)?);
            let x = ops::bools_to_member(fb, v, *lanes)?;
            fb.composite_insert(x, agg, path)?
        }
        S::IntFn { f, form } => math::int_fn(fb, *f, *form, xs)?,
        S::Wide(w) => {
            let routine = match w.routine() {
                Some(d) => match cx.decl.families.arith.divides.get(&d) {
                    Some(f) => Some(*f),
                    None => return Err(cx.fail("a wide division whose routine is not declared")),
                },
                None => None,
            };
            wide::emit(fb, w, xs, routine)?
        }
        S::FloatFn {
            f: op @ FloatFn::FastTanh { reciprocal },
            form,
            ..
        } => {
            let routine = math::FastDivision::new(*form, *reciprocal, Contraction::Forbidden);
            let id = fast_division(cx, routine)?;
            let x = a(cx, 0)?;
            let divide = |fb: &mut FunctionBuilder<'_>, p: SValue, q: SValue| {
                math::call_fast_division(fb, id, p, q)
            };
            let lib = lib(cx);
            math::fast_tanh(fb, *op, *form, x, &lib, &divide)?
        }
        // Metal's formula divides by the fast division's routine.
        S::FloatFn {
            f: op @ FloatFn::FastAtanh { reciprocal },
            form,
            ..
        } => {
            let routine = math::FastDivision::new(*form, *reciprocal, Contraction::Forbidden);
            let id = fast_division(cx, routine)?;
            let x = a(cx, 0)?;
            let divide = |fb: &mut FunctionBuilder<'_>, p: SValue, q: SValue| {
                math::call_fast_division(fb, id, p, q)
            };
            let lib = lib(cx);
            math::fast_atanh(fb, *op, *form, x, &lib, &divide)?
        }
        // Apple's formula divides by the fast division's routine.
        S::FloatFn {
            f: FloatFn::FastTan { reciprocal },
            form,
            ..
        } => {
            let routine = math::FastDivision::new(*form, *reciprocal, Contraction::Forbidden);
            let id = fast_division(cx, routine)?;
            let x = a(cx, 0)?;
            let divide = |fb: &mut FunctionBuilder<'_>, p: SValue, q: SValue| {
                math::call_fast_division(fb, id, p, q)
            };
            let lib = lib(cx);
            math::fast_tan(fb, *form, x, &lib, &divide)?
        }
        S::FloatFn { f, form, contract } => {
            let test = nan::NanTest::for_modes(&cx.plan.families.arith.modes);
            let lib = lib(cx);
            math::float_fn(fb, *f, *form, *contract, test, xs, &lib)?
        }
        S::Fma(f) => {
            let routine = match f.routine() {
                Some(r) => match cx.decl.families.arith.fmas.get(&r) {
                    Some(id) => Some(*id),
                    None => {
                        return Err(cx.fail("a fused multiply-add whose routine is not declared"))
                    }
                },
                None => None,
            };
            fma::emit(fb, *f, xs, routine)?
        }
        S::Fused {
            fma,
            a,
            b,
            c,
            negate,
            flush,
            inner,
        } => {
            let routine = match fma.routine() {
                Some(r) => match cx.decl.families.arith.fmas.get(&r) {
                    Some(id) => Some(*id),
                    None => return Err(cx.fail("a fused addition whose routine is not declared")),
                },
                None => None,
            };
            let mut xyz = [
                cx.operand(fb, *a)?,
                cx.operand(fb, *b)?,
                cx.operand(fb, *c)?,
            ];
            for (i, v) in xyz.iter_mut().enumerate() {
                if flush.operand(i) {
                    *v = denormal::flush(fb, fma.form, *v)?;
                }
            }
            let [mut x, y, mut z] = xyz;
            match negate {
                contract::Negate::Neither => {}
                contract::Negate::Product => x = fb.unary(UnOp::FNegate, x)?,
                contract::Negate::Addend => z = fb.unary(UnOp::FNegate, z)?,
            }
            // A chain's links, innermost first, each one's value the next one's addend.
            for link in inner.iter().rev() {
                let mut ab = [cx.operand(fb, link.a)?, cx.operand(fb, link.b)?];
                for (i, v) in ab.iter_mut().enumerate() {
                    if link.flush.operand(i) {
                        *v = denormal::flush(fb, fma.form, *v)?;
                    }
                }
                let [mut p, q] = ab;
                if link.negate {
                    p = fb.unary(UnOp::FNegate, p)?;
                }
                z = fma::emit(fb, *fma, &[p, q, z], routine)?;
                if link.flush.result {
                    z = denormal::flush(fb, fma.form, z)?;
                }
            }
            let r = fma::emit(fb, *fma, &[x, y, z], routine)?;
            if flush.result {
                denormal::flush(fb, fma.form, r)?
            } else {
                r
            }
        }
        S::FastFmod { fma: f, reciprocal } => {
            let routine = match f.routine() {
                Some(r) => match cx.decl.families.arith.fmas.get(&r) {
                    Some(id) => Some(*id),
                    None => {
                        return Err(cx.fail("a fast fmod whose fused multiply-add is not declared"))
                    }
                },
                None => None,
            };
            let (x, y) = match xs {
                [x, y, ..] => (*x, *y),
                _ => return Err(cx.fail("a fast fmod without its operands")),
            };
            let fma =
                |fb: &mut FunctionBuilder<'_>, args: &[SValue]| fma::emit(fb, *f, args, routine);
            let division = math::FastDivision::new(f.form, *reciprocal, Contraction::Forbidden);
            let id = fast_division(cx, division)?;
            let divide = |fb: &mut FunctionBuilder<'_>, p: SValue, q: SValue| {
                math::call_fast_division(fb, id, p, q)
            };
            math::fast_fmod(fb, x, y, &divide, &fma)?
        }
        S::Mix { fma: f, flush } => {
            let routine = match f.routine() {
                Some(r) => match cx.decl.families.arith.fmas.get(&r) {
                    Some(id) => Some(*id),
                    None => return Err(cx.fail("a mix whose fused multiply-add is not declared")),
                },
                None => None,
            };
            let xyt = match xs {
                [x, y, t, ..] => [*x, *y, *t],
                _ => return Err(cx.fail("a mix without its operands")),
            };
            let fma =
                |fb: &mut FunctionBuilder<'_>, args: &[SValue]| fma::emit(fb, *f, args, routine);
            math::metal_mix(fb, f.form, xyt, &fma, *flush)?
        }
        S::Dot {
            fma: f,
            form,
            flush,
            tiny,
        } => {
            let routine = match f.routine() {
                Some(r) => match cx.decl.families.arith.fmas.get(&r) {
                    Some(id) => Some(*id),
                    None => return Err(cx.fail("a dot whose fused multiply-add is not declared")),
                },
                None => None,
            };
            let xy = match xs {
                [x, y, ..] => [*x, *y],
                _ => return Err(cx.fail("a dot without its operands")),
            };
            let fma =
                |fb: &mut FunctionBuilder<'_>, args: &[SValue]| fma::emit(fb, *f, args, routine);
            math::metal_dot(fb, *form, xy, &fma, (*flush, *tiny))?
        }
        S::Convert { kind, from, to } => {
            let p = a(cx, 0)?;
            // An 8-bit float source converts through `f32`.
            let width = from.float_fmt().map_or(32, FloatFmt::container);
            let test = nan::NanTest::for_width(&cx.plan.families.arith.modes, width);
            math::convert(fb, *kind, *from, *to, p, test)?
        }
        S::Pack {
            dir,
            format,
            floats,
        } => {
            let p = a(cx, 0)?;
            let table = cx.decl.families.arith.srgb;
            if *format == PackFormat::Unorm4x8Srgb && table.is_none() {
                return Err(cx.fail("an sRGB pack whose table is not declared"));
            }
            let test = nan::NanTest::for_modes(&cx.plan.families.arith.modes);
            match dir {
                PackDir::Pack => math::pack(fb, *format, *floats, p, table, test)?,
                PackDir::Unpack => math::unpack(fb, *format, *floats, p, table)?,
            }
        }
    })
}
