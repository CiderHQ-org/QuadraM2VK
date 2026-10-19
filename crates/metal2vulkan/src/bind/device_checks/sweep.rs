//! The device's sweeps of three deviations measured on every input of a 32-bit domain
//! (`DriverDeviation`'s `MetalHalfConversion`, `MetalMinMax` and `MetalHalfArithmetic`). One
//! module per deviation and math mode computes, on each input, the form translation writes where
//! the profile takes the deviation (the device's instruction) and the form it writes elsewhere
//! (Metal's on every device, which the validation lanes hold against Metal's own answers), and
//! counts the inputs where the two differ. Metal's answer is a count of 0 over all 2^32 inputs;
//! the comparison runs on the GPU, so nothing leaves it but the count and the least input that
//! differs. A module emits a multiply the device rounds on its own in the form translation writes
//! on the device (`arith::product_form`).
//!
//! **The module** ([`SweepKernel`]): one `GLCompute` entry `main` at local size
//! ([`CheckKernel::LOCAL_SIZE`](crate::bind::CheckKernel::LOCAL_SIZE), 1, 1), a `uvec2` push
//! constant `(base, listed)` at offset 0, and in set 0 two storage buffers of 4-byte words: the
//! listed inputs at binding
//! [`SweepKernel::LIST_BINDING`] (at least one word) and the result at
//! [`SweepKernel::RESULT_BINDING`] (two words, [`SweepKernel::RESULT_START`] before the first
//! dispatch). Invocation `x` takes input `list[x]` where `listed` is not 0 (and none at or past
//! `listed`), else `base + x`, wrapping. Where the forms differ on it, it adds 1 to result word 0 and
//! takes its minimum with word 1, atomically. A sweep runs the structured sample
//! ([`SweepKernel::sample`]) listed first, as an early out (inputs where a wrong form is most likely
//! to differ), then the domain in dispatches of [`SweepKernel::CHUNK`] inputs from `base` 0 up.
//!
//! Each input runs the domain's operations on one scalar and on a four-lane vector whose lanes take
//! the input and three bijections of it (signs flipped, halves exchanged, complemented), so every
//! lane of the vector form meets every input too. Translation writes the device's form in entries
//! with and without `SignedZeroInfNanPreserve` for 32 bits (an operation without every fast-math
//! flag declares it), so a sweep runs in each such math mode the device has ([`MathMode`]):
//!
//! - **The `half` conversion** ([`Sweep::HalfConversion`]): the input is an `f32`'s bits. The
//!   device's form is `QuantizeForm::Converted` (`x + -0.0` decorated `NoContraction`, converted to
//!   16 bits and widened); the exact form `QuantizeForm::Integer`, bit for bit, a NaN included:
//!   Metal converts every `float` NaN to `0x7E00` (`narrow`'s integer form), and where the profile
//!   takes the deviation a NaN's conversion is the device's, the NaN model selecting none before a
//!   narrowing (which answers the canonical NaN itself). Needs `shaderFloat16`.
//! - **The minimum and maximum** ([`Sweep::MinMax`]): the input is two halves (its high and low 16
//!   bits), widened to `float` as translation holds them, and three `float`s hashed from it. The
//!   device's form is `GLSL.std.450` `NMin`, `NMax` and `NClamp`; the exact form Metal's pick
//!   (`math::metal_min_max`), bit for bit: `fmin` and `fmax` of the halves and of two of the
//!   `float`s, and `clamp` with a third operand of each kind. Where the operands of an `fmin` or
//!   `fmax` tie (zeros of either sign, or `float`s that compare as a zero) or are both NaNs, either
//!   operand is Metal's answer: Metal's compiler takes the two as commutative, and one module
//!   compiled by Metal on Apple M3 Max answered each operand of such a pair in different calls
//!   ([`metal_answers`]). So the device's form answers Metal's where it is the exact form or the
//!   exact form with the operands of any of its `fmin`s and `fmax`s exchanged, bit for bit.
//! - **The `half` arithmetic** ([`Sweep::HalfArithmetic`]): the input is two halves, widened. The
//!   device's form is `ops::f16_binary` (the 16-bit `OpFAdd` and `OpFSub` decorated
//!   `NoContraction` and `Fma(a, b, -0.0)`, between exact conversions); the exact form
//!   `ops::float_binary` of the `half` form under `QuantizeForm::Integer` (the `float` result
//!   rounded to `half` by integer arithmetic), bit for bit, a NaN result included (Metal's
//!   `0x7E00`), its product in the device's multiply form (on MoltenVK `Fma(a, b, -0.0)`, since
//!   SPIRV-Cross writes an `OpFMul` decorated `NoContraction` as `fma(l, r, 0)`, which answers `+0`
//!   for a `-0` product). Needs `shaderFloat16`.

use std::sync::Arc;

use super::kernel::{finish, MathMode, LOCAL_SIZE};
use crate::air::ir::BinOp as AirBin;
use crate::device::requirements::Requirements;
use crate::device::Device;
use crate::error::Error;
use crate::float16::f16_bits_to_f32;
use crate::lower::arith::legal::{FloatFmt, Form, Held};
use crate::lower::arith::math::{metal_min_max, FloatFn};
use crate::lower::arith::nan::NanTest;
use crate::lower::arith::narrow;
use crate::lower::arith::ops::{f16_binary, float_binary, splat, ty};
use crate::lower::arith::product_form;
use crate::pipeline::sym::{PushField, ResourceKind, ResourceSlot, Stage};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder, ProductForm, QuantizeForm};
use crate::spirv::error::{BResult, BuildError, TypeRule};
use crate::spirv::ids::{GlobalId, TypeId, Value};
use crate::spirv::ir::{
    natural_layout, AtomicOp, BinOp, Contraction, ConvOp, ExecMode, ExecModel, GlslOp, IoDir,
    LayoutRule, MemAccess, Scope, Semantics, StorageClass, Type, TypedModule, UnOp,
};

/// A deviation the device sweeps over every input (the module docs).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Sweep {
    /// `DriverDeviation::MetalHalfConversion`: every `f32`'s conversion to `half`.
    HalfConversion,
    /// `DriverDeviation::MetalMinMax`: `fmin`, `fmax` and `clamp` of every pair of widened halves
    /// and of `float`s hashed from every word.
    MinMax,
    /// `DriverDeviation::MetalHalfArithmetic`: `+`, `-` and `*` of every pair of halves.
    HalfArithmetic,
}

impl Sweep {
    /// Every sweep.
    pub const ALL: [Sweep; 3] = [Sweep::HalfConversion, Sweep::MinMax, Sweep::HalfArithmetic];

    /// Why `device` cannot run the sweep: the `half` sweeps need `shaderFloat16`, without which
    /// translation writes neither of their device forms.
    pub fn unsupported(self, device: &Device) -> Option<&'static str> {
        match self {
            Sweep::MinMax => None,
            Sweep::HalfConversion | Sweep::HalfArithmetic => {
                match device.profile().fields().leaf.float16 {
                    true => None,
                    false => Some("the device has no shaderFloat16"),
                }
            }
        }
    }

    /// The sweep's module on `device` in math mode `mode`, and its sample. Errs where
    /// [`Sweep::unsupported`] names a reason, or `mode` is not among [`MathMode::of`]'s.
    pub fn kernel(self, device: &Device, mode: MathMode) -> Result<SweepKernel, Error> {
        let product = product_form(device.profile().fields().float_functions);
        let (m, what, sample) = match self {
            Sweep::HalfConversion => (
                sweep_module(&conversion_ops, mode, product)?,
                "half conversion sweep",
                conversion_sample(),
            ),
            Sweep::MinMax => (
                sweep_module(&min_max_ops, mode, product)?,
                "minimum and maximum sweep",
                pair_sample(),
            ),
            Sweep::HalfArithmetic => (
                sweep_module(&arithmetic_ops, mode, product)?,
                "half arithmetic sweep",
                pair_sample(),
            ),
        };
        let (spirv, requirements) = finish(device, m, what)?;
        Ok(SweepKernel {
            spirv,
            requirements,
            sample,
        })
    }
}

/// One sweep's module on one device in one math mode, and its sample (the module docs).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SweepKernel {
    /// SPIR-V 1.5 words, entry point `main`, bindings in set [`SweepKernel::SET`].
    pub spirv: Arc<[u32]>,
    /// What the module needs from the device (within its profile).
    pub requirements: Requirements,
    /// The structured sample's inputs, dispatched listed before the domain: the early out.
    pub sample: Vec<u32>,
}

impl SweepKernel {
    /// The descriptor set of both bindings.
    pub const SET: u32 = 0;
    /// The listed inputs: a storage buffer of at least one word.
    pub const LIST_BINDING: u32 = 0;
    /// The result: a storage buffer of two words, the inputs where the forms differ and the least
    /// of them.
    pub const RESULT_BINDING: u32 = 1;
    /// The result's words before the first dispatch.
    pub const RESULT_START: [u32; 2] = [0, u32::MAX];
    /// Bytes of the `COMPUTE` push-constant range at offset 0: `(base, listed)`, two `u32`s.
    pub const PUSH_BYTES: u32 = 8;
    /// The inputs of one dispatch of the domain, in `CHUNK / LOCAL_SIZE` workgroups (16,384,
    /// within every device's `maxComputeWorkGroupCount[0]`): a few milliseconds of an integrated
    /// GPU's time at most.
    pub const CHUNK: u32 = 1 << 20;
    /// The domain's dispatches: every 32-bit word in [`SweepKernel::CHUNK`]s.
    pub const CHUNKS: u32 = ((1u64 << 32) / Self::CHUNK as u64) as u32;

    /// The push bytes of the domain's dispatch `chunk` (0 to [`SweepKernel::CHUNKS`]).
    pub fn chunk_push(chunk: u32) -> [u8; 8] {
        push(chunk.wrapping_mul(Self::CHUNK), 0)
    }

    /// The workgroups of a dispatch of the domain.
    pub fn chunk_workgroups() -> u32 {
        Self::CHUNK / LOCAL_SIZE
    }

    /// The push bytes of the dispatch of the sample, listed.
    pub fn sample_push(&self) -> [u8; 8] {
        push(0, self.sample_len())
    }

    /// The workgroups of the dispatch of the sample.
    pub fn sample_workgroups(&self) -> u32 {
        self.sample_len().div_ceil(LOCAL_SIZE)
    }

    fn sample_len(&self) -> u32 {
        u32::try_from(self.sample.len()).unwrap_or(u32::MAX)
    }

    /// Whether a sweep's result words answer Metal's: no input differed.
    pub fn answers_metal(result: &[u32]) -> bool {
        result.first() == Some(&0)
    }

    /// How a sweep's result words differ from Metal's answer, `None` where they answer it: the
    /// inputs that differed among those dispatched, and the least of them (its word, as the
    /// sweep's domain reads it).
    pub fn difference(result: &[u32]) -> Option<String> {
        match result {
            [0, ..] => None,
            [count, least, ..] => Some(format!(
                "{count} inputs differ from the exact form, the least {least:#010x}"
            )),
            _ => Some(format!(
                "{} result words read where the sweep writes 2",
                result.len()
            )),
        }
    }
}

/// The push bytes `(base, listed)`.
fn push(base: u32, listed: u32) -> [u8; 8] {
    let mut bytes = [0; 8];
    bytes[..4].copy_from_slice(&base.to_le_bytes());
    bytes[4..].copy_from_slice(&listed.to_le_bytes());
    bytes
}

/// One operation's results, as `u32` bits of `float` lanes: the device's form's, the exact form's,
/// and Metal's other answers. Metal's answer has the device's result equal, in each lane, to the
/// exact one or to one of the others.
struct Pair {
    device: Value,
    exact: Value,
    /// Metal's answers besides the exact form's ([`metal_answers`]); none but the minimum and
    /// maximum's.
    also: Vec<Value>,
}

/// A domain's operations on input words `x` (`u32` lanes of the form given, one or four): each
/// operation's [`Pair`].
type Ops = dyn Fn(&mut FunctionBuilder<'_>, Form, Value) -> BResult<Vec<Pair>>;

/// `u32` lanes, `lanes` wide.
fn words(lanes: u8) -> Form {
    Form {
        held: Held::Int {
            bits: 32,
            container: 32,
        },
        lanes,
    }
}

/// `float` lanes, `lanes` wide.
fn floats(lanes: u8) -> Form {
    Form {
        held: Held::Float(FloatFmt::F32),
        lanes,
    }
}

/// `half` lanes held widened in `float`, `lanes` wide.
fn halves(lanes: u8) -> Form {
    Form {
        held: Held::Float(FloatFmt::Half),
        lanes,
    }
}

/// The lanes of the four-lane form of input word `x`: the input, its signs flipped (bits 31 and
/// 15), its halves exchanged, and its complement.
fn lanes_of(f: &mut FunctionBuilder<'_>, x: Value) -> BResult<[Value; 4]> {
    let w = words(1);
    let signs = splat(f, w, 0x8000_8000)?;
    let flipped = f.binary(BinOp::BitwiseXor, x, signs)?;
    let sixteen = splat(f, w, 16)?;
    let high = f.binary(BinOp::ShiftRightLogical, x, sixteen)?;
    let low = f.binary(BinOp::ShiftLeftLogical, x, sixteen)?;
    let exchanged = f.binary(BinOp::BitwiseOr, high, low)?;
    let ones = splat(f, w, u64::from(u32::MAX))?;
    let complement = f.binary(BinOp::BitwiseXor, x, ones)?;
    Ok([x, flipped, exchanged, complement])
}

/// A `Block { u32 words[]; }` storage buffer (std430) at buffer slot `index`.
fn words_buffer(b: &mut ModuleBuilder, index: u32, u32_: TypeId) -> BResult<GlobalId> {
    let rta = b.ty(Type::RuntimeArray {
        elem: u32_,
        stride: Some(4),
    })?;
    let layout = natural_layout(b.types(), LayoutRule::Std430, true, &[rta])?;
    let block = b.ty(Type::Struct {
        members: [rta].into(),
        layout: Some(layout),
    })?;
    let slot = ResourceSlot {
        stage: Stage::Kernel,
        kind: ResourceKind::Buffer,
        index,
    };
    b.resource(slot, StorageClass::StorageBuffer, block)
}

/// The sweep module around `ops` in math mode `mode`, its multiplies in form `product` (the module
/// docs).
fn sweep_module(ops: &Ops, mode: MathMode, product: ProductForm) -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    b.product_form(product);
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let uvec2 = b.ty(Type::Vector { elem: u32_, len: 2 })?;
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
    let shape = b.push_member(PushField::DispatchRegion, uvec2, 0)?;
    let push = b.push_global()?;
    let list = words_buffer(&mut b, SweepKernel::LIST_BINDING, u32_)?;
    let result = words_buffer(&mut b, SweepKernel::RESULT_BINDING, u32_)?;
    let invocation = b.builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let zero = splat(&mut f, words(1), 0)?;
    let one = splat(&mut f, words(1), 1)?;
    let invocation = f.module().global_value(invocation)?;
    let invocation = f.load(invocation, MemAccess::default())?;
    let x = f.composite_extract(invocation, &[0])?;
    let push = f.module().global_value(push)?;
    let member = splat(&mut f, words(1), u64::from(shape.index))?;
    let shape = f.access_chain(push, &[member])?;
    let shape = f.load(shape, MemAccess::default())?;
    let base = f.composite_extract(shape, &[0])?;
    let listed = f.composite_extract(shape, &[1])?;
    let list = f.module().global_value(list)?;
    let result = f.module().global_value(result)?;
    // The listed input, read in bounds whatever `listed` is (the list holds at least one word).
    let from_list = f.binary(BinOp::INotEqual, listed, zero)?;
    let last = f.binary(BinOp::ISub, listed, one)?;
    let clamped = f.glsl(GlslOp::UMin, &[x, last])?;
    let at = f.select(from_list, clamped, zero)?;
    let entry = f.access_chain(list, &[zero, at])?;
    let listed_input = f.load(entry, MemAccess::default())?;
    let counted = f.binary(BinOp::IAdd, base, x)?;
    let input = f.select(from_list, listed_input, counted)?;
    let past = f.binary(BinOp::UGreaterThanEqual, x, listed)?;
    let skipped = f.binary(BinOp::LogicalAnd, from_list, past)?;
    let taken = f.unary(UnOp::LogicalNot, skipped)?;
    let body: ArmFn<'_, ()> = Box::new(move |f, _| {
        let differs = differs(f, input, ops)?;
        let record: ArmFn<'_, ()> = Box::new(move |f, _| {
            let count = f.access_chain(result, &[zero, zero])?;
            f.atomic(
                AtomicOp::IAdd,
                count,
                Scope::Device,
                Semantics::RELAXED,
                one,
            )?;
            let least = f.access_chain(result, &[zero, one])?;
            f.atomic(
                AtomicOp::UMin,
                least,
                Scope::Device,
                Semantics::RELAXED,
                input,
            )?;
            Ok(ArmEnd::FallThrough(Vec::new()))
        });
        f.if_(&mut (), differs, record, None)?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    f.if_(&mut (), taken, body, None)?;
    f.ret(None)?;
    let main = f.finish()?;
    let mut modes = vec![ExecMode::LocalSize {
        x: LOCAL_SIZE,
        y: 1,
        z: 1,
    }];
    modes.extend(mode.exec_modes());
    b.entry(ExecModel::GLCompute, main, modes)?;
    b.finalize()
}

/// Whether any of `ops`' pairs on input word `input`, scalar or on the four lanes of
/// [`lanes_of`], differ.
fn differs(f: &mut FunctionBuilder<'_>, input: Value, ops: &Ops) -> BResult<Value> {
    let uvec4 = ty(f, words(4))?;
    let lanes = lanes_of(f, input)?;
    let packed = f.composite_construct(uvec4, &lanes)?;
    let mut any = splat(
        f,
        Form {
            held: Held::Bool,
            lanes: 1,
        },
        0,
    )?;
    for (lanes, x) in [(1, input), (4, packed)] {
        for pair in ops(f, words(lanes), x)? {
            let d = disagree(f, lanes, pair)?;
            any = f.binary(BinOp::LogicalOr, any, d)?;
        }
    }
    Ok(any)
}

/// Whether `pair`'s device result (`lanes` lanes) differs, in any lane, from the exact result and
/// from each of Metal's other answers.
fn disagree(f: &mut FunctionBuilder<'_>, lanes: u8, pair: Pair) -> BResult<Value> {
    let mut differ = f.binary(BinOp::INotEqual, pair.device, pair.exact)?;
    for other in pair.also {
        let d = f.binary(BinOp::INotEqual, pair.device, other)?;
        differ = f.binary(BinOp::LogicalAnd, differ, d)?;
    }
    match lanes {
        1 => Ok(differ),
        _ => f.unary(UnOp::Any, differ),
    }
}

/// `x`'s bits (`u32` lanes) as `float` lanes.
fn as_float(f: &mut FunctionBuilder<'_>, lanes: u8, x: Value) -> BResult<Value> {
    let t = ty(f, floats(lanes))?;
    f.convert(ConvOp::Bitcast, x, t)
}

/// `float` lanes `x`'s bits.
fn as_bits(f: &mut FunctionBuilder<'_>, lanes: u8, x: Value) -> BResult<Value> {
    let t = ty(f, words(lanes))?;
    f.convert(ConvOp::Bitcast, x, t)
}

/// The two halves of input words `x` (`u32` lanes), high then low, each widened exactly to `float`
/// as translation holds a `half` (`narrow::from_bits`).
fn halves_of(f: &mut FunctionBuilder<'_>, lanes: u8, x: Value) -> BResult<(Value, Value)> {
    let sixteen = splat(f, words(lanes), 16)?;
    let high = f.binary(BinOp::ShiftRightLogical, x, sixteen)?;
    Ok((
        narrow::from_bits(f, halves(lanes), high)?,
        narrow::from_bits(f, halves(lanes), x)?,
    ))
}

/// Run `body` with the module's quantize form `form`, then restore the one before.
fn with_quantize(
    f: &mut FunctionBuilder<'_>,
    form: QuantizeForm,
    body: impl FnOnce(&mut FunctionBuilder<'_>) -> BResult<Value>,
) -> BResult<Value> {
    let before = f.module().quantize_form();
    f.module().set_quantize_form(form);
    let r = body(f);
    f.module().set_quantize_form(before);
    r
}

// ---------------------------------------------------------------------------------------------
// The `half` conversion.

/// The conversion's operation on input words `x`: the device's conversion of the `f32` against
/// its rounding in integer arithmetic.
fn conversion_ops(f: &mut FunctionBuilder<'_>, w: Form, x: Value) -> BResult<Vec<Pair>> {
    let lanes = w.lanes;
    let v = as_float(f, lanes, x)?;
    let device = with_quantize(f, QuantizeForm::Converted, |f| {
        narrow::round(f, halves(lanes), v)
    })?;
    let exact = with_quantize(f, QuantizeForm::Integer, |f| {
        narrow::round(f, halves(lanes), v)
    })?;
    Ok(vec![Pair {
        device: as_bits(f, lanes, device)?,
        exact: as_bits(f, lanes, exact)?,
        also: Vec::new(),
    }])
}

/// The `f32`s the conversion's sample adds to the halves' grid: the infinities, the largest
/// finite, the smallest and largest denormals, and NaNs whose payload's top ten bits are 0 (which a
/// conversion that drops the low bits makes an infinity) or not.
const CONVERSION_SPECIALS: [u32; 10] = [
    0x7f80_0000,
    0x7f7f_ffff,
    0x0000_0001,
    0x007f_ffff,
    0x7f80_0001,
    0x7f80_1fff,
    0x7f80_2000,
    0x7fbf_ffff,
    0x7fc0_0000,
    0x7fff_ffff,
];

/// The conversion's sample: every finite half's value, the `f32` either side of it, every tie
/// between neighbouring halves (65520, the last, rounds to infinity) and the `f32` either side of
/// it, and [`CONVERSION_SPECIALS`], each of both signs.
fn conversion_sample() -> Vec<u32> {
    let mut out = Vec::new();
    for h in 0u16..0x7c00 {
        let a = f16_bits_to_f32(h);
        let b = f16_bits_to_f32(h + 1);
        let tie = match h {
            0x7bff => 65520.0f32,
            _ => ((f64::from(a) + f64::from(b)) / 2.0) as f32,
        };
        let (a, tie) = (a.to_bits(), tie.to_bits());
        for bits in [a.wrapping_sub(1), a, a + 1, tie - 1, tie, tie + 1] {
            out.push(bits);
            out.push(bits ^ 0x8000_0000);
        }
    }
    for bits in CONVERSION_SPECIALS {
        out.push(bits);
        out.push(bits ^ 0x8000_0000);
    }
    out.sort_unstable();
    out.dedup();
    out
}

// ---------------------------------------------------------------------------------------------
// The minimum and maximum.

/// The odd multipliers of the three `float`s hashed from an input word.
const HASH: [u32; 3] = [0x9e37_79b1, 0x85eb_ca77, 0xc2b2_ae3d];

/// The bits hashed from input words `x` (`u32` lanes) by multiplier `k`: `x * k` rotated left by
/// 13, a bijection of the words.
fn hashed(f: &mut FunctionBuilder<'_>, lanes: u8, x: Value, k: u32) -> BResult<Value> {
    let w = words(lanes);
    let k = splat(f, w, u64::from(k))?;
    let p = f.binary(BinOp::IMul, x, k)?;
    let thirteen = splat(f, w, 13)?;
    let nineteen = splat(f, w, 19)?;
    let l = f.binary(BinOp::ShiftLeftLogical, p, thirteen)?;
    let r = f.binary(BinOp::ShiftRightLogical, p, nineteen)?;
    f.binary(BinOp::BitwiseOr, l, r)
}

/// The minimum and maximum's operations on input words `x`: `fmin`, `fmax`, `clamp(a, b, c)` and
/// `clamp(c, a, b)` of its widened halves `(a, b)` and the half `c` of the low bits of a hash, and
/// the same of `float`s `(p, q, r)` hashed from it, each `NMin`, `NMax` or `NClamp` against Metal's
/// answers ([`metal_answers`]).
fn min_max_ops(f: &mut FunctionBuilder<'_>, w: Form, x: Value) -> BResult<Vec<Pair>> {
    let lanes = w.lanes;
    let (a, b) = halves_of(f, lanes, x)?;
    let [p, q, r] = [
        hashed(f, lanes, x, HASH[0])?,
        hashed(f, lanes, x, HASH[1])?,
        hashed(f, lanes, x, HASH[2])?,
    ];
    let c = narrow::from_bits(f, halves(lanes), r)?;
    let (p, q, r) = (
        as_float(f, lanes, p)?,
        as_float(f, lanes, q)?,
        as_float(f, lanes, r)?,
    );
    let mut out = Vec::new();
    for (form, [x, y, z]) in [(halves(lanes), [a, b, c]), (floats(lanes), [p, q, r])] {
        let cases: [(FloatFn, GlslOp, &[Value]); 4] = [
            (FloatFn::Min, GlslOp::NMin, &[x, y]),
            (FloatFn::Max, GlslOp::NMax, &[x, y]),
            (FloatFn::Clamp, GlslOp::NClamp, &[x, y, z]),
            (FloatFn::Clamp, GlslOp::NClamp, &[z, x, y]),
        ];
        for (op, device_op, args) in cases {
            let device = f.glsl(device_op, args)?;
            let device = as_bits(f, lanes, device)?;
            let (exact, also) = metal_answers(f, op, form, args)?;
            let exact = as_bits(f, lanes, exact)?;
            let also = also
                .into_iter()
                .map(|v| as_bits(f, lanes, v))
                .collect::<BResult<Vec<_>>>()?;
            out.push(Pair {
                device,
                exact,
                also,
            });
        }
    }
    Ok(out)
}

/// Metal's answers to `fmin`, `fmax` or `clamp` (`op`) of `args` on lanes of form `form`: its pick
/// ([`metal_min_max`]), and the others with the operands of each `fmin` and `fmax` exchanged
/// (`clamp(x, lo, hi)` is `fmin(fmax(x, lo), hi)`). Metal's compiler takes `fmin` and `fmax` as
/// commutative, so where the operands tie or are both NaNs either operand is its answer: on Apple
/// M3 Max (MoltenVK 1.4.1) one module of the sweep's `half` operands compiled by Metal answered
/// `fmin(-0, +0)`, `fmax(-0, +0)` and `clamp(-0, +0, c)` with `+0` in its scalar call and `-0` in a
/// vector lane whose operands it had exchanged, and `fmin` and `fmax` of two NaNs with the second's
/// bits in the one and the first's in the other. Elsewhere the orders answer alike.
fn metal_answers(
    f: &mut FunctionBuilder<'_>,
    op: FloatFn,
    form: Form,
    args: &[Value],
) -> BResult<(Value, Vec<Value>)> {
    let pick = |f: &mut FunctionBuilder<'_>, op: FloatFn, args: &[Value]| {
        metal_min_max(f, op, form, NanTest::Bits, args)
    };
    let exact = pick(f, op, args)?;
    let also = match (op, args) {
        (FloatFn::Min | FloatFn::Max, &[x, y]) => vec![pick(f, op, &[y, x])?],
        (FloatFn::Clamp, &[x, lo, hi]) => {
            let low = pick(f, FloatFn::Max, &[x, lo])?;
            let exchanged = pick(f, FloatFn::Max, &[lo, x])?;
            vec![
                pick(f, FloatFn::Min, &[hi, low])?,
                pick(f, FloatFn::Min, &[exchanged, hi])?,
                pick(f, FloatFn::Min, &[hi, exchanged])?,
            ]
        }
        // Not one of the sweep's minimum and maximum functions.
        _ => return Err(BuildError::Type(TypeRule::FunctionSignature)),
    };
    Ok((exact, also))
}

/// The halves of the pair sample, each also of the other sign: the zeros, the smallest and largest
/// denormals, the smallest normal and its neighbour, 1 and its neighbours, a third, the largest
/// finite, the infinities, signalling and quiet NaNs of several payloads.
const SPECIAL_HALVES: [u16; 16] = [
    0x0000, 0x0001, 0x03ff, 0x0400, 0x0401, 0x3bff, 0x3c00, 0x3c01, 0x3555, 0x7bff, 0x7c00, 0x7c01,
    0x7d55, 0x7e00, 0x7e01, 0x7fff,
];

/// The sample of the pair domains: every pair of [`SPECIAL_HALVES`] of either sign, as input words.
fn pair_sample() -> Vec<u32> {
    let halves: Vec<u32> = SPECIAL_HALVES
        .iter()
        .flat_map(|&h| [u32::from(h), u32::from(h | 0x8000)])
        .collect();
    let mut out: Vec<u32> = halves
        .iter()
        .flat_map(|&a| halves.iter().map(move |&b| (a << 16) | b))
        .collect();
    out.sort_unstable();
    out
}

// ---------------------------------------------------------------------------------------------
// The `half` arithmetic.

/// The `half` arithmetic's operations on input words `x`: `a + b`, `a - b` and `a * b` of its two
/// widened halves, the device's 16-bit instruction against the `float` result rounded to `half` in
/// integer arithmetic.
fn arithmetic_ops(f: &mut FunctionBuilder<'_>, w: Form, x: Value) -> BResult<Vec<Pair>> {
    let lanes = w.lanes;
    let (a, b) = halves_of(f, lanes, x)?;
    let mut out = Vec::new();
    for op in [AirBin::FAdd, AirBin::FSub, AirBin::FMul] {
        let device = f16_binary(f, op, lanes, a, b)?;
        let exact = with_quantize(f, QuantizeForm::Integer, |f| {
            float_binary(f, op, halves(lanes), Contraction::Forbidden, a, b)
        })?;
        out.push(Pair {
            device: as_bits(f, lanes, device)?,
            exact: as_bits(f, lanes, exact)?,
            also: Vec::new(),
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::{
        arithmetic_ops, conversion_ops, conversion_sample, disagree, halves, halves_of,
        metal_answers, min_max_ops, pair_sample, sweep_module, words, words_buffer, MathMode, Pair,
        Sweep, SweepKernel, CONVERSION_SPECIALS,
    };
    use super::{as_bits, narrow, splat, FloatFn};
    use crate::device::profile::FloatFunctions;
    use crate::device::{Device, DevicePolicy};
    use crate::float16::{f16_bits_to_f32, f32_to_f16_bits};
    use crate::lower::arith::product_form;
    use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage};
    use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
    use crate::spirv::error::BResult;
    use crate::spirv::ids::{GlobalId, Value};
    use crate::spirv::interp::{run, Buffer, Config, Resources};
    use crate::spirv::ir::{BinOp, ExecMode, ExecModel, IoDir, MemAccess, Type};
    use crate::spirv::testing::{assert_valid, committed_descriptions, committed_profiles};
    use crate::spv::{Decoration, Op};

    /// Each sweep module is valid in each math mode on every committed profile that has what it
    /// needs.
    #[test]
    fn the_sweep_modules_are_valid_on_the_committed_profiles() {
        let mut halves = 0;
        for (_, profile) in committed_profiles().expect("profiles") {
            let fields = profile.fields();
            let product = product_form(fields.float_functions);
            for mode in MathMode::of_profile(&profile) {
                let m = sweep_module(&min_max_ops, mode, product).expect("min max");
                assert_valid(m, &profile);
                if fields.leaf.float16 {
                    let m = sweep_module(&conversion_ops, mode, product).expect("conversion");
                    assert_valid(m, &profile);
                    let m = sweep_module(&arithmetic_ops, mode, product).expect("arithmetic");
                    assert_valid(m, &profile);
                    halves += 1;
                }
            }
        }
        assert!(halves > 0, "no committed profile has shaderFloat16");
    }

    /// The conversion's sample holds the edges of the half range and the NaNs, and the edges
    /// convert as Metal does.
    #[test]
    fn the_conversion_sample_holds_the_edges() {
        let sample = conversion_sample();
        let edges = [
            0x477f_f000u32,
            0x477f_efff,
            0x477f_f001,
            0x3300_0000,
            0xb300_0000,
        ];
        for bits in edges.into_iter().chain(CONVERSION_SPECIALS) {
            assert!(sample.binary_search(&bits).is_ok(), "{bits:#x}");
        }
        let converted = |x: u32| f16_bits_to_f32(f32_to_f16_bits(f32::from_bits(x))).to_bits();
        assert_eq!(converted(0x477f_f000), 0x7f80_0000);
        assert_eq!(converted(0x477f_efff), 0x477f_e000);
    }

    /// The pair sample is every pair of 32 halves; the domain is every word in its chunks.
    #[test]
    fn the_pair_sample_and_the_chunks_cover_their_domains() {
        assert_eq!(pair_sample().len(), 32 * 32);
        assert_eq!(
            u64::from(SweepKernel::CHUNKS) * u64::from(SweepKernel::CHUNK),
            1u64 << 32
        );
        assert_eq!(SweepKernel::chunk_workgroups(), 16_384);
        assert_eq!(SweepKernel::chunk_push(1), [0, 0, 0x10, 0, 0, 0, 0, 0]);
    }

    /// A result of no differing input answers Metal's and names no difference; one of some names
    /// their count and the least.
    #[test]
    fn a_sweep_result_names_its_difference() {
        assert!(SweepKernel::answers_metal(&SweepKernel::RESULT_START));
        assert_eq!(SweepKernel::difference(&SweepKernel::RESULT_START), None);
        let found = [3, 0x3c00_0001];
        assert!(!SweepKernel::answers_metal(&found));
        let how = SweepKernel::difference(&found).expect("a difference");
        assert!(how.starts_with("3 inputs differ"), "{how}");
        assert!(how.ends_with("0x3c000001"), "{how}");
    }

    /// The `half` arithmetic's exact form multiplies in the form translation writes on the
    /// device: where the device's `float` functions are Metal's (MoltenVK), no `OpFMul` decorated
    /// `NoContraction` reaches SPIRV-Cross, which writes one as `fma(l, r, 0)`, `+0` for a `-0`
    /// product (on Apple M3 Max such a module's sample differed on 88 inputs, the least
    /// `0x00008000`, `+0 · -0`); elsewhere the product is that `OpFMul`.
    #[test]
    fn the_half_arithmetic_sweep_multiplies_in_the_device_form() {
        let mut metal = 0;
        for (stem, desc) in committed_descriptions().expect("descriptions") {
            let device = Device::new(desc.clone(), DevicePolicy::default()).expect("device");
            if Sweep::HalfArithmetic.unsupported(&device).is_some() {
                continue;
            }
            let kernel = Sweep::HalfArithmetic
                .kernel(&device, MathMode::Relaxed)
                .unwrap_or_else(|e| panic!("{stem}: {e}"));
            let products = decorated_products(&kernel.spirv);
            match device.profile().fields().float_functions {
                FloatFunctions::Metal => {
                    assert_eq!(products, 0, "{stem}");
                    metal += 1;
                }
                FloatFunctions::Device | FloatFunctions::Native => {
                    assert!(products > 0, "{stem}");
                }
            }
        }
        assert!(
            metal > 0,
            "no committed profile with shaderFloat16 has Metal's functions"
        );
    }

    /// The `OpFMul`s of module `words` decorated `NoContraction`.
    fn decorated_products(words: &[u32]) -> usize {
        let mut decorated = HashSet::new();
        let mut products = Vec::new();
        let mut at = 5;
        while let Some(&head) = words.get(at) {
            let len = usize::try_from(head >> 16).unwrap_or(1).max(1);
            let inst = words.get(at..at + len).unwrap_or_default();
            let opcode = head & 0xffff;
            if opcode == Op::Decorate as u32
                && inst.get(2) == Some(&(Decoration::NoContraction as u32))
            {
                decorated.extend(inst.get(1).copied());
            }
            if opcode == Op::FMul as u32 {
                products.extend(inst.get(2).copied());
            }
            at += len;
        }
        products.iter().filter(|r| decorated.contains(*r)).count()
    }

    /// Metal's answers on Apple M3 Max (MoltenVK 1.4.1), recorded from one module of the sweep's
    /// `half` operands compiled by Metal: an input word (`a` its high half, `b` its low), the half
    /// `c`, and one call's `fmin(a, b)`, `fmax(a, b)` and `clamp(a, b, c)`. Each pair of rows is
    /// the module's scalar call, then a vector lane holding the same operands, which Metal's
    /// compiler had exchanged: a tie of zeros (`-0`, `+0`), and two NaNs (`0x7d55`, `0x7c01`).
    const RECORDED: [(u32, u32, [u32; 3]); 4] = [
        (0x8000_0000, 0x1000, [0x0000, 0x0000, 0x0000]),
        (0x8000_0000, 0x1000, [0x8000, 0x8000, 0x8000]),
        (0x7d55_7c01, 0xbe4b, [0x7c01, 0x7c01, 0xbe4b]),
        (0x7d55_7c01, 0xbe4b, [0x7d55, 0x7d55, 0xbe4b]),
    ];

    /// The minimum and maximum's comparison takes each of Metal's recorded answers ([`RECORDED`]),
    /// where the exact form alone takes only the scalar call's, and it takes no answer that is
    /// neither operand: a canonical NaN, and the second NaN quieted. On the interpreter, each row's
    /// three answers against the exact form alone, then against Metal's answers, 1 where they
    /// differ.
    #[test]
    fn the_minimum_and_maximum_take_metals_recorded_answers() {
        let rows: Vec<(u32, u32, [u32; 3])> = RECORDED
            .into_iter()
            .chain([(0x7d55_7c01, 0xbe4b, [0x7e00, 0x7e01, 0xbe4b])])
            .collect();
        let m = recorded_module().expect("module");
        let slot = |index| ResourceSlot {
            stage: Stage::Kernel,
            kind: ResourceKind::Buffer,
            index,
        };
        let input: Vec<u8> = rows
            .iter()
            .flat_map(|&(x, c, answers)| [x, c].into_iter().chain(answers))
            .flat_map(u32::to_le_bytes)
            .collect();
        let resources = Resources {
            buffers: vec![
                Buffer {
                    slot: Some(slot(0)),
                    address: None,
                    bytes: input,
                },
                Buffer {
                    slot: Some(slot(1)),
                    address: None,
                    bytes: vec![0; rows.len() * 6 * 4],
                },
            ],
            ..Resources::default()
        };
        let config = Config {
            workgroups: [u32::try_from(rows.len()).expect("rows"), 1, 1],
            ..Config::default()
        };
        let outcome = run(&m, &config, resources).expect("run");
        let read: Vec<u32> = outcome.buffers[1]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&w| u32::from_le_bytes(w))
            .collect();
        let want: [[u32; 6]; 5] = [
            [0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 0, 1, 1, 0],
        ];
        assert_eq!(read, want.concat());
    }

    /// A kernel of one invocation a row of words `(x, c, fmin, fmax, clamp)` in binding 0, writing
    /// six words a row to binding 1 ([`recorded_row`]).
    fn recorded_module() -> BResult<crate::spirv::ir::TypedModule> {
        let mut b = ModuleBuilder::new();
        let u32_ = b.ty(Type::Int {
            width: 32,
            signed: false,
        })?;
        let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
        let rows = words_buffer(&mut b, 0, u32_)?;
        let out = words_buffer(&mut b, 1, u32_)?;
        let id = b.builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)?;
        let void = b.ty(Type::Void)?;
        let mut f = b.function(void)?;
        recorded_row(&mut f, [rows, out, id])?;
        f.ret(None)?;
        let main = f.finish()?;
        let size = ExecMode::LocalSize { x: 1, y: 1, z: 1 };
        b.entry(ExecModel::GLCompute, main, vec![size])?;
        b.finalize()
    }

    /// Row `id.x`'s comparisons: each recorded answer of `fmin(a, b)`, `fmax(a, b)` and
    /// `clamp(a, b, c)` against the exact form alone (words 0 to 2), then against Metal's answers
    /// ([`metal_answers`], words 3 to 5), 1 where it differs.
    fn recorded_row(f: &mut FunctionBuilder<'_>, [rows, out, id]: [GlobalId; 3]) -> BResult<()> {
        let w = words(1);
        let id = f.module().global_value(id)?;
        let id = f.load(id, MemAccess::default())?;
        let i = f.composite_extract(id, &[0])?;
        let rows = f.module().global_value(rows)?;
        let out = f.module().global_value(out)?;
        let zero = splat(f, w, 0)?;
        let one = splat(f, w, 1)?;
        let at = |f: &mut FunctionBuilder<'_>, stride: u64, k: u64| -> BResult<Value> {
            let stride = splat(f, w, stride)?;
            let k = splat(f, w, k)?;
            let base = f.binary(BinOp::IMul, i, stride)?;
            f.binary(BinOp::IAdd, base, k)
        };
        let load = |f: &mut FunctionBuilder<'_>, k: u64| -> BResult<Value> {
            let at = at(f, 5, k)?;
            let p = f.access_chain(rows, &[zero, at])?;
            f.load(p, MemAccess::default())
        };
        let x = load(f, 0)?;
        let (a, b) = halves_of(f, 1, x)?;
        let c = load(f, 1)?;
        let c = narrow::from_bits(f, halves(1), c)?;
        let cases = [
            (FloatFn::Min, vec![a, b]),
            (FloatFn::Max, vec![a, b]),
            (FloatFn::Clamp, vec![a, b, c]),
        ];
        for (k, (op, args)) in (0u64..).zip(cases) {
            let answer = load(f, 2 + k)?;
            let answer = narrow::from_bits(f, halves(1), answer)?;
            let device = as_bits(f, 1, answer)?;
            let (exact, also) = metal_answers(f, op, halves(1), &args)?;
            let exact = as_bits(f, 1, exact)?;
            let also = also
                .into_iter()
                .map(|v| as_bits(f, 1, v))
                .collect::<BResult<Vec<_>>>()?;
            let alone = Pair {
                device,
                exact,
                also: Vec::new(),
            };
            let alone = disagree(f, 1, alone)?;
            let metal = disagree(
                f,
                1,
                Pair {
                    device,
                    exact,
                    also,
                },
            )?;
            for (word, differs) in [(k, alone), (3 + k, metal)] {
                let flag = f.select(differs, one, zero)?;
                let at = at(f, 6, word)?;
                let p = f.access_chain(out, &[zero, at])?;
                f.store(p, flag, MemAccess::default())?;
            }
        }
        Ok(())
    }
}
