//! The device's checks of two arithmetic deviations Metal's rule decides (`DriverDeviation`'s
//! `MetalCanonicalNans` and `MetalFloatDenormals`). Each kernel runs the device's own
//! instructions, the ones translation leaves to the device where the profile takes the deviation,
//! over rows of operands, and stores each result an operand row's mask selects (0 where it does
//! not): Metal's answer by its rule, computed here on the CPU ([`CheckKernel::expected`]). A row's
//! mask leaves out a result Vulkan leaves to the device whatever its arithmetic (a quotient by a
//! divisor outside `[2^-126, 2^126]`), and one the deviation does not claim (a result rounded to
//! the smallest normal before a device flushes it: Metal decides tininess before rounding, the
//! devices measured after, and translation answers that window itself,
//! `denormal::checks_tiny_products`).
//!
//! - **Canonical NaNs** ([`canonical_nans_check`]): `float` `+ - * /`, `fmod` (`OpFRem`),
//!   `GLSL.std.450` `Fma` and `Sqrt`, `OpFmaKHR` where the device has `shaderFmaFloat32`, and a
//!   sum's narrowing to `half` and widening back where it has `shaderFloat16`, over NaN operands
//!   of every class (quiet and signalling, both signs, several payloads) beside finite, zero and
//!   infinite ones, and over operands whose result is a NaN by IEEE 754. A NaN result is stored
//!   as its bits and any other result as 0: Metal answers the canonical NaN `0x7FC00000` for each
//!   (`lower::arith::nan`'s class `Canonical`, the narrowing's `0x7E00` widened).
//! - **`float` denormals** ([`float_denormals_check`]): `float` `+ - *`, `OpFmaKHR` (where the
//!   device has it), `OpDot`, `Ldexp`, `Sqrt`, `InverseSqrt`, `Trunc`, `RoundEven`,
//!   `OpFOrdLessThan` and `OpFOrdEqual`, over rows rich in denormals of both signs and the
//!   smallest normals: Metal reads an `f32` denormal operand as the zero of its sign and flushes a
//!   denormal result to the zero of its sign, rounding to nearest even otherwise. Results Vulkan
//!   does not round exactly (a square root of a normal, a dot product of two nonzero products) are
//!   left out.
//!
//! Both run in each math mode ([`MathMode`]) the device has, one kernel a mode.

use super::kernel::{finish, MathMode, LOCAL_SIZE};
use super::CheckKernel;
use crate::device::Device;
use crate::error::Error;
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, TypeId, Value};
use crate::spirv::ir::{
    natural_layout, BinOp, ConstBits, Constant, ConvOp, ExecMode, ExecModel, GlslOp, IoDir,
    LayoutRule, MemAccess, StorageClass, Type, TypedModule,
};

/// One operation of a kernel: its `u32` result from the row's operand words.
type Op<'a> = Box<dyn Fn(&mut FunctionBuilder<'_>, &Types, &[Value]) -> BResult<Value> + 'a>;

/// The scalar and vector types the operations take.
struct Types {
    u32_: TypeId,
    i32_: TypeId,
    f32_: TypeId,
    vec2: TypeId,
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

/// A `u32` constant as a value.
fn word(f: &mut FunctionBuilder<'_>, u32_: TypeId, bits: u32) -> BResult<Value> {
    let c = f.module().constant(Constant::Scalar {
        ty: u32_,
        bits: ConstBits(u64::from(bits)),
    })?;
    f.module().const_value(c)
}

/// The kernel over rows of `operands` words and a mask word, storing operation `k` of `ops`
/// where the mask's bit `k` is set and 0 elsewhere, in math mode `mode`: storage buffer 0 holds
/// the rows, buffer 1 receives `ops.len()` words a row.
fn rows_module(operands: u32, ops: &[Op<'_>], mode: MathMode) -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let i32_ = b.ty(Type::Int {
        width: 32,
        signed: true,
    })?;
    let f32_ = b.ty(Type::Float { width: 32 })?;
    let vec2 = b.ty(Type::Vector { elem: f32_, len: 2 })?;
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
    let input = words_buffer(&mut b, 0, u32_)?;
    let output = words_buffer(&mut b, 1, u32_)?;
    let invocation = b.builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let types = Types {
        u32_,
        i32_,
        f32_,
        vec2,
    };
    let zero = word(&mut f, u32_, 0)?;
    let invocation = f.module().global_value(invocation)?;
    let invocation = f.load(invocation, MemAccess::default())?;
    let i = f.composite_extract(invocation, &[0])?;
    let input = f.module().global_value(input)?;
    let output = f.module().global_value(output)?;
    let row = word(&mut f, u32_, operands + 1)?;
    let first = f.binary(BinOp::IMul, i, row)?;
    let mut words = Vec::with_capacity(operands as usize + 1);
    for k in 0..=operands {
        let k = word(&mut f, u32_, k)?;
        let at = f.binary(BinOp::IAdd, first, k)?;
        let p = f.access_chain(input, &[zero, at])?;
        words.push(f.load(p, MemAccess::default())?);
    }
    let mask = words[operands as usize];
    let count = word(&mut f, u32_, ops.len() as u32)?;
    let out = f.binary(BinOp::IMul, i, count)?;
    for (k, op) in ops.iter().enumerate() {
        let r = op(&mut f, &types, &words[..operands as usize])?;
        let bit = word(&mut f, u32_, 1 << k)?;
        let taken = f.binary(BinOp::BitwiseAnd, mask, bit)?;
        let taken = f.binary(BinOp::INotEqual, taken, zero)?;
        let r = f.select(taken, r, zero)?;
        let k = word(&mut f, u32_, k as u32)?;
        let at = f.binary(BinOp::IAdd, out, k)?;
        let p = f.access_chain(output, &[zero, at])?;
        f.store(p, r, MemAccess::default())?;
    }
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

/// The kernel of `rows` (each `operands` words and a mask word) and their `expected` results on
/// `device` in math mode `mode`, padded to whole workgroups with rows whose mask is 0.
fn rows_kernel(
    device: &Device,
    m: TypedModule,
    what: &str,
    mut rows: Vec<Vec<u32>>,
    mut expected: Vec<Vec<u32>>,
    operands: usize,
    ops: usize,
) -> Result<CheckKernel, Error> {
    let workgroups = (rows.len() as u32).div_ceil(LOCAL_SIZE).max(1);
    let padded = (workgroups * LOCAL_SIZE) as usize;
    rows.resize(padded, vec![0; operands + 1]);
    expected.resize(padded, vec![0; ops]);
    let outputs = vec![0; padded * ops];
    Ok(CheckKernel::new(
        finish(device, m, what)?,
        workgroups,
        vec![rows.concat(), outputs],
        1,
        expected.concat(),
    ))
}

/// `x`'s bits as a `float`.
fn float(f: &mut FunctionBuilder<'_>, t: &Types, x: Value) -> BResult<Value> {
    f.convert(ConvOp::Bitcast, x, t.f32_)
}

/// A `float`'s bits.
fn bits(f: &mut FunctionBuilder<'_>, t: &Types, x: Value) -> BResult<Value> {
    f.convert(ConvOp::Bitcast, x, t.u32_)
}

/// A `bool` as 1 or 0.
fn flag(f: &mut FunctionBuilder<'_>, t: &Types, x: Value) -> BResult<Value> {
    let one = word(f, t.u32_, 1)?;
    let zero = word(f, t.u32_, 0)?;
    f.select(x, one, zero)
}

/// The operands `k` of a row as `float`s.
fn floats<const N: usize>(
    f: &mut FunctionBuilder<'_>,
    t: &Types,
    w: &[Value],
) -> BResult<[Value; N]> {
    let mut out = [w[0]; N];
    for (k, o) in out.iter_mut().enumerate() {
        *o = float(f, t, w[k])?;
    }
    Ok(out)
}

/// A binary `float` instruction's result bits.
fn binary(op: BinOp) -> Op<'static> {
    Box::new(move |f, t, w| {
        let [a, b] = floats::<2>(f, t, w)?;
        let r = f.binary(op, a, b)?;
        bits(f, t, r)
    })
}

/// A `GLSL.std.450` instruction's result bits, of the row's first `arity` operands.
fn glsl(op: GlslOp, arity: usize) -> Op<'static> {
    Box::new(move |f, t, w| {
        let x = floats::<3>(f, t, w)?;
        let r = f.glsl(op, &x[..arity])?;
        bits(f, t, r)
    })
}

/// `OpFmaKHR(a, b, c)`'s result bits.
fn fma_khr() -> Op<'static> {
    Box::new(|f, t, w| {
        let [a, b, c] = floats::<3>(f, t, w)?;
        let r = f.fma(a, b, c)?;
        bits(f, t, r)
    })
}

/// A comparison of the first two operands as 1 or 0.
fn compare(op: BinOp) -> Op<'static> {
    Box::new(move |f, t, w| {
        let [a, b] = floats::<2>(f, t, w)?;
        let r = f.binary(op, a, b)?;
        flag(f, t, r)
    })
}

// ---------------------------------------------------------------------------------------------
// Canonical NaNs.

/// The NaN check's operations, in mask-bit order; the last two where the device has
/// `shaderFmaFloat32` and `shaderFloat16`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NanOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Fma,
    Sqrt,
    FmaKhr,
    Narrowed,
}

const NAN_OPS: [NanOp; 9] = [
    NanOp::Add,
    NanOp::Sub,
    NanOp::Mul,
    NanOp::Div,
    NanOp::Rem,
    NanOp::Fma,
    NanOp::Sqrt,
    NanOp::FmaKhr,
    NanOp::Narrowed,
];

/// The NaNs of the check: quiet and signalling, several payloads, each of both signs.
const NANS: [u32; 7] = [
    0x7fc0_0000,
    0x7fc0_0001,
    0x7fd5_5555,
    0x7fff_ffff,
    0x7f80_0001,
    0x7fa0_0000,
    0x7fbf_ffff,
];

/// The other operands beside a NaN: ones, a fraction, zeros, infinities, a large and a small
/// normal.
const OTHERS: [u32; 8] = [
    0x3f80_0000,
    0xc020_0000,
    0x0000_0000,
    0x8000_0000,
    0x7f80_0000,
    0xff80_0000,
    0x7f61_b1e6,
    0x0e2a_d4f1,
];

/// The rows without a NaN operand whose result is a NaN in some operation: `inf - inf`,
/// `inf + -inf`, `0 * inf`, `fma(0, inf, 1)`, `fma(inf, 1, -inf)`, `sqrt(-1)`, `sqrt(-inf)` and
/// `fmod(inf, 1)`.
const NAN_MAKING: [[u32; 3]; 6] = [
    [0x7f80_0000, 0x7f80_0000, 0x3f80_0000],
    [0x7f80_0000, 0xff80_0000, 0x3f80_0000],
    [0x0000_0000, 0x7f80_0000, 0x3f80_0000],
    [0x7f80_0000, 0x3f80_0000, 0xff80_0000],
    [0xbf80_0000, 0x3f80_0000, 0x3f80_0000],
    [0xff80_0000, 0x3f80_0000, 0x3f80_0000],
];

/// Whether the device's quotient or remainder by `b` is defined: `b` a NaN, or a normal within
/// `[2^-126, 2^126]` in magnitude (Vulkan's precision of `OpFDiv`, which `OpFRem`'s inherits).
fn divisor_defined(b: f32) -> bool {
    b.is_nan() || (b.abs() >= f32::MIN_POSITIVE && b.abs() <= 2f32.powi(126))
}

/// Whether operation `op` of row `(a, b, c)` is a NaN by IEEE 754, where the device's answer is
/// defined: `None` where it is not.
fn nan_result(op: NanOp, [a, b, c]: [f32; 3]) -> Option<bool> {
    let r = match op {
        NanOp::Add | NanOp::Narrowed => a + b,
        NanOp::Sub => a - b,
        NanOp::Mul => a * b,
        NanOp::Div | NanOp::Rem if !divisor_defined(b) => return None,
        NanOp::Div => a / b,
        NanOp::Rem => a % b,
        NanOp::Fma | NanOp::FmaKhr => a.mul_add(b, c),
        NanOp::Sqrt => a.sqrt(),
    };
    Some(r.is_nan())
}

/// The NaN check's rows: each NaN as each operand beside each other operand, NaNs together, and
/// the [`NAN_MAKING`] rows.
fn nan_rows() -> Vec<[u32; 3]> {
    let nans: Vec<u32> = NANS.iter().flat_map(|&n| [n, n | 0x8000_0000]).collect();
    let mut rows = Vec::new();
    for &n in &nans {
        for &x in &OTHERS {
            rows.extend([[n, x, x], [x, n, x], [x, x, n]]);
        }
    }
    for (k, &n) in nans.iter().enumerate() {
        rows.push([n, nans[(k + 1) % nans.len()], nans[(k + 5) % nans.len()]]);
    }
    rows.extend(NAN_MAKING);
    rows
}

/// Whether `device` has the NaN check's operation `op`.
fn nan_op_on(device: &Device, op: NanOp) -> bool {
    let leaf = &device.profile().fields().leaf;
    match op {
        NanOp::FmaKhr => leaf.fma32,
        NanOp::Narrowed => leaf.float16,
        _ => true,
    }
}

/// The NaN check's operation `op` in the kernel.
fn nan_op(op: NanOp) -> Op<'static> {
    match op {
        NanOp::Add => binary(BinOp::FAdd),
        NanOp::Sub => binary(BinOp::FSub),
        NanOp::Mul => binary(BinOp::FMul),
        NanOp::Div => binary(BinOp::FDiv),
        NanOp::Rem => binary(BinOp::FRem),
        NanOp::Fma => glsl(GlslOp::Fma, 3),
        NanOp::Sqrt => glsl(GlslOp::Sqrt, 1),
        NanOp::FmaKhr => fma_khr(),
        NanOp::Narrowed => Box::new(|f, t, w| {
            let [a, b] = floats::<2>(f, t, w)?;
            let s = f.binary_no_contraction(BinOp::FAdd, a, b)?;
            let f16_ = f.module().ty(Type::Float { width: 16 })?;
            let h = f.convert(ConvOp::FConvert, s, f16_)?;
            let r = f.convert(ConvOp::FConvert, h, t.f32_)?;
            bits(f, t, r)
        }),
    }
}

/// Every operation result's stored word: its bits where it is a NaN, else 0.
fn nan_or_zero(op: Op<'static>) -> Op<'static> {
    Box::new(move |f, t, w| {
        let r = op(f, t, w)?;
        let magnitude = word(f, t.u32_, 0x7fff_ffff)?;
        let infinity = word(f, t.u32_, 0x7f80_0000)?;
        let zero = word(f, t.u32_, 0)?;
        let m = f.binary(BinOp::BitwiseAnd, r, magnitude)?;
        let nan = f.binary(BinOp::UGreaterThan, m, infinity)?;
        f.select(nan, r, zero)
    })
}

/// The check of `DriverDeviation::MetalCanonicalNans` on `device` in math mode `mode` (the module
/// docs): storage buffer 0 holds four words a row (`a`, `b`, `c` and the mask), buffer 1 receives
/// nine a row.
pub fn canonical_nans_check(device: &Device, mode: MathMode) -> Result<CheckKernel, Error> {
    let ops: Vec<Op<'static>> = NAN_OPS
        .iter()
        .map(|&op| match nan_op_on(device, op) {
            true => nan_or_zero(nan_op(op)),
            false => {
                Box::new(|f: &mut FunctionBuilder<'_>, t: &Types, _: &[Value]| word(f, t.u32_, 0))
                    as Op<'static>
            }
        })
        .collect();
    let m = rows_module(3, &ops, mode)?;
    let mut rows = Vec::new();
    let mut expected = Vec::new();
    for row in nan_rows() {
        let values = row.map(f32::from_bits);
        let mut mask = 0;
        let mut words = Vec::new();
        for (k, &op) in NAN_OPS.iter().enumerate() {
            let nan = match nan_op_on(device, op) {
                true => nan_result(op, values),
                false => None,
            };
            match nan {
                Some(nan) => {
                    mask |= 1 << k;
                    words.push(if nan { 0x7fc0_0000 } else { 0 });
                }
                None => words.push(0),
            }
        }
        rows.push(vec![row[0], row[1], row[2], mask]);
        expected.push(words);
    }
    rows_kernel(
        device,
        m,
        "canonical NaN check",
        rows,
        expected,
        3,
        NAN_OPS.len(),
    )
}

// ---------------------------------------------------------------------------------------------
// `float` denormals.

/// The denormal check's operations, in mask-bit order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DenormalOp {
    Add,
    Sub,
    Mul,
    FmaKhr,
    Dot,
    Ldexp,
    Sqrt,
    InverseSqrt,
    Trunc,
    RoundEven,
    Less,
    Equal,
}

const DENORMAL_OPS: [DenormalOp; 12] = [
    DenormalOp::Add,
    DenormalOp::Sub,
    DenormalOp::Mul,
    DenormalOp::FmaKhr,
    DenormalOp::Dot,
    DenormalOp::Ldexp,
    DenormalOp::Sqrt,
    DenormalOp::InverseSqrt,
    DenormalOp::Trunc,
    DenormalOp::RoundEven,
    DenormalOp::Less,
    DenormalOp::Equal,
];

/// The denormal check's values, each of both signs: denormals (the smallest, the largest, and
/// between), the smallest normals, and normals whose products, sums and scalings fall below the
/// smallest normal or onto it.
const DENORMAL_VALUES: [u32; 16] = [
    0x0000_0000,
    0x0000_0001,
    0x0000_0002,
    0x0012_3456,
    0x0040_0000,
    0x007f_ffff,
    0x0080_0000,
    0x0080_0001,
    0x00ff_ffff,
    0x2000_0000,
    0x1fc0_0000,
    0x0d80_0000,
    0x3080_0000,
    0x3f80_0000,
    0x3fc0_0000,
    0x4040_0000,
];

/// The scalings `Ldexp` takes, by row.
const EXPONENTS: [i32; 5] = [0, -1, -24, -126, 3];

/// Whether `x` is an `f32` denormal.
fn denormal(x: f32) -> bool {
    x != 0.0 && x.abs() < f32::MIN_POSITIVE
}

/// `x` read as Metal reads an operand, or flushed as Metal flushes a result: a denormal as the
/// zero of its sign.
fn flushed(x: f32) -> f32 {
    match denormal(x) {
        true => 0.0f32.copysign(x),
        false => x,
    }
}

/// Whether a result whose exact value is `exact` (to the precision of an `f64`) may round to the
/// smallest normal from below it, where Metal, deciding tininess before rounding, flushes it and
/// the devices measured, rounding first, do not (the module docs): within `2^-149` below
/// `2^-126`, or on it.
fn tiny_window(exact: f64) -> bool {
    let smallest = f64::from(f32::MIN_POSITIVE);
    let m = exact.abs();
    m <= smallest && m >= smallest - 2f64.powi(-149)
}

/// Metal's result of operation `op` on row `(a, b, c)` with scaling `e`, as the stored word,
/// where the check compares it: `None` where Vulkan does not round the result exactly or it lies
/// in [`tiny_window`].
fn denormal_answer(op: DenormalOp, [a, b, c]: [f32; 3], e: i32) -> Option<u32> {
    let (x, y, z) = (flushed(a), flushed(b), flushed(c));
    let (wx, wy, wz) = (f64::from(x), f64::from(y), f64::from(z));
    let value = |r: f32| Some(flushed(r).to_bits());
    match op {
        DenormalOp::Add => value(x + y),
        DenormalOp::Sub => value(x - y),
        DenormalOp::Mul if tiny_window(wx * wy) => None,
        DenormalOp::Mul => value(x * y),
        DenormalOp::FmaKhr if tiny_window(wx.mul_add(wy, wz)) => None,
        DenormalOp::FmaKhr => value(x.mul_add(y, z)),
        // `(a, b) . (b, c)`: exact where the first product is a zero and the second a normal.
        DenormalOp::Dot => {
            let second = y * z;
            let exact = x == 0.0 && second != 0.0 && !denormal(second);
            match exact && !tiny_window(wy * wz) {
                true => value(second),
                false => None,
            }
        }
        DenormalOp::Ldexp => {
            let exact = wx * 2f64.powi(e);
            match tiny_window(exact) {
                true => None,
                false => value(exact as f32),
            }
        }
        DenormalOp::Sqrt if x == 0.0 => value(x),
        DenormalOp::InverseSqrt if x == 0.0 && x.is_sign_positive() => value(f32::INFINITY),
        DenormalOp::Sqrt | DenormalOp::InverseSqrt => None,
        DenormalOp::Trunc => value(x.trunc()),
        DenormalOp::RoundEven => value(x.round_ties_even()),
        DenormalOp::Less => Some(u32::from(x < y)),
        DenormalOp::Equal => Some(u32::from(x == y)),
    }
}

/// The denormal check's operation `op` in the kernel (`Ldexp`'s scaling the row's fourth word).
fn denormal_op(op: DenormalOp) -> Op<'static> {
    match op {
        DenormalOp::Add => binary(BinOp::FAdd),
        DenormalOp::Sub => binary(BinOp::FSub),
        DenormalOp::Mul => binary(BinOp::FMul),
        DenormalOp::FmaKhr => fma_khr(),
        DenormalOp::Dot => Box::new(|f, t, w| {
            let [a, b, c] = floats::<3>(f, t, w)?;
            let u = f.composite_construct(t.vec2, &[a, b])?;
            let v = f.composite_construct(t.vec2, &[b, c])?;
            let r = f.binary(BinOp::Dot, u, v)?;
            bits(f, t, r)
        }),
        DenormalOp::Ldexp => Box::new(|f, t, w| {
            let a = float(f, t, w[0])?;
            let e = f.convert(ConvOp::Bitcast, w[3], t.i32_)?;
            let r = f.glsl(GlslOp::Ldexp, &[a, e])?;
            bits(f, t, r)
        }),
        DenormalOp::Sqrt => glsl(GlslOp::Sqrt, 1),
        DenormalOp::InverseSqrt => glsl(GlslOp::InverseSqrt, 1),
        DenormalOp::Trunc => glsl(GlslOp::Trunc, 1),
        DenormalOp::RoundEven => glsl(GlslOp::RoundEven, 1),
        DenormalOp::Less => compare(BinOp::FOrdLessThan),
        DenormalOp::Equal => compare(BinOp::FOrdEqual),
    }
}

/// The denormal check's rows: every pair of [`DENORMAL_VALUES`] of either sign, the third operand
/// and the scaling taken in turn.
fn denormal_rows() -> Vec<([u32; 3], i32)> {
    let values: Vec<u32> = DENORMAL_VALUES
        .iter()
        .flat_map(|&v| [v, v | 0x8000_0000])
        .collect();
    let mut rows = Vec::new();
    for (i, &a) in values.iter().enumerate() {
        for (j, &b) in values.iter().enumerate() {
            let c = values[(i * 7 + j * 3) % values.len()];
            let e = EXPONENTS[(i + j) % EXPONENTS.len()];
            rows.push(([a, b, c], e));
        }
    }
    rows
}

/// The check of `DriverDeviation::MetalFloatDenormals` on `device` in math mode `mode` (the module
/// docs): storage buffer 0 holds five words a row (`a`, `b`, `c`, `Ldexp`'s scaling and the mask),
/// buffer 1 receives twelve a row.
pub fn float_denormals_check(device: &Device, mode: MathMode) -> Result<CheckKernel, Error> {
    let fma = device.profile().fields().leaf.fma32;
    let on = |op: DenormalOp| op != DenormalOp::FmaKhr || fma;
    let ops: Vec<Op<'static>> = DENORMAL_OPS
        .iter()
        .map(|&op| match on(op) {
            true => denormal_op(op),
            false => {
                Box::new(|f: &mut FunctionBuilder<'_>, t: &Types, _: &[Value]| word(f, t.u32_, 0))
                    as Op<'static>
            }
        })
        .collect();
    let m = rows_module(4, &ops, mode)?;
    let mut rows = Vec::new();
    let mut expected = Vec::new();
    for (row, e) in denormal_rows() {
        let values = row.map(f32::from_bits);
        let mut mask = 0;
        let mut words = Vec::new();
        for (k, &op) in DENORMAL_OPS.iter().enumerate() {
            match denormal_answer(op, values, e).filter(|_| on(op)) {
                Some(w) => {
                    mask |= 1 << k;
                    words.push(w);
                }
                None => words.push(0),
            }
        }
        rows.push(vec![row[0], row[1], row[2], e as u32, mask]);
        expected.push(words);
    }
    rows_kernel(
        device,
        m,
        "float denormal check",
        rows,
        expected,
        4,
        DENORMAL_OPS.len(),
    )
}

#[cfg(test)]
mod tests {
    use super::{
        denormal_answer, denormal_op, denormal_rows, nan_op, nan_or_zero, nan_result, nan_rows,
        rows_module, tiny_window, DenormalOp, NanOp, Op, DENORMAL_OPS, NAN_OPS,
    };
    use crate::bind::device_checks::kernel::MathMode;
    use crate::spirv::testing::{assert_valid, committed_profiles};

    /// The two kernels are valid in each math mode on every committed profile, each operation
    /// where the profile has it.
    #[test]
    fn the_arithmetic_check_kernels_are_valid_on_the_committed_profiles() {
        for (_, profile) in committed_profiles().expect("profiles") {
            let leaf = &profile.fields().leaf;
            let nan_ops: Vec<Op<'static>> = NAN_OPS
                .iter()
                .filter(|&&op| match op {
                    NanOp::FmaKhr => leaf.fma32,
                    NanOp::Narrowed => leaf.float16,
                    _ => true,
                })
                .map(|&op| nan_or_zero(nan_op(op)))
                .collect();
            let denormal_ops: Vec<Op<'static>> = DENORMAL_OPS
                .iter()
                .filter(|&&op| op != DenormalOp::FmaKhr || leaf.fma32)
                .map(|&op| denormal_op(op))
                .collect();
            for mode in MathMode::of_profile(&profile) {
                assert_valid(rows_module(3, &nan_ops, mode).expect("NaNs"), &profile);
                assert_valid(
                    rows_module(4, &denormal_ops, mode).expect("denormals"),
                    &profile,
                );
            }
        }
    }

    /// Every NaN row has a NaN result to compare, and the NaN-making rows make one without a NaN
    /// operand.
    #[test]
    fn every_nan_row_compares_a_nan() {
        for row in nan_rows() {
            let values = row.map(f32::from_bits);
            let nans = NAN_OPS
                .iter()
                .filter(|&&op| nan_result(op, values) == Some(true))
                .count();
            assert!(nans > 0, "{row:x?}");
        }
        let inf = f32::INFINITY;
        assert_eq!(nan_result(NanOp::Sub, [inf, inf, 1.0]), Some(true));
        assert_eq!(nan_result(NanOp::Div, [1.0, 0.0, 1.0]), None);
    }

    /// The denormal answers flush operands and results, and leave out the window below the
    /// smallest normal.
    #[test]
    fn the_denormal_answers_flush_and_skip_the_window() {
        let tiny = f32::from_bits(0x0000_0001);
        let smallest = f32::MIN_POSITIVE;
        assert_eq!(
            denormal_answer(DenormalOp::Add, [tiny, 0.0, 0.0], 0),
            Some(0)
        );
        assert_eq!(
            denormal_answer(DenormalOp::Add, [-tiny, 0.0, 0.0], 0),
            Some(0)
        );
        assert_eq!(
            denormal_answer(DenormalOp::Sub, [-tiny, 0.0, 0.0], 0),
            Some(0x8000_0000)
        );
        assert_eq!(
            denormal_answer(DenormalOp::Equal, [tiny, 0.0, 0.0], 0),
            Some(1)
        );
        // `(2 - 2^-23) * 2^-127` is `2^-126 - 2^-150`: in the window.
        let below = f32::from_bits(0x00ff_ffff);
        assert!(tiny_window(f64::from(below) * 0.5));
        assert_eq!(
            denormal_answer(DenormalOp::Ldexp, [below, 0.0, 0.0], -1),
            None
        );
        assert_eq!(
            denormal_answer(DenormalOp::Ldexp, [smallest, 0.0, 0.0], -1),
            Some(0)
        );
        assert_eq!(
            denormal_answer(DenormalOp::InverseSqrt, [tiny, 0.0, 0.0], 0),
            Some(0x7f80_0000)
        );
        let compared = denormal_rows()
            .iter()
            .filter(|(row, e)| {
                denormal_answer(DenormalOp::Mul, row.map(f32::from_bits), *e).is_some()
            })
            .count();
        assert!(compared > 900, "{compared}");
    }
}
