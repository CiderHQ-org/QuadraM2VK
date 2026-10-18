//! Interpreter differential tests of the arith emission core.
//!
//! Each test builds one compute kernel around a core function: invocation `i` reads its operands
//! from buffer 1 (each lane one `u32` word, two for a 64-bit lane), converts them to their held
//! form, runs the function, and writes the result words to buffer 0. The kernel passes L3
//! (`assert_valid`) on a committed profile with 64-bit and 16-bit arithmetic, then runs on the
//! CPU interpreter over every case whose reference result is defined (LLVM poison and undefined
//! behavior are skipped, so an interpreter trap is a real failure). References are written from
//! the LLVM and Metal definitions in plain Rust.

use super::divide::Reciprocal;
use super::legal::{f16_to_f64, f64_to_f16, FloatFmt, Form, Held};
use super::libm::Lib;
use super::math::{self, ConvertKind, FloatFn, IntFn, Sign};
use super::nan::NanTest;
use super::ops::{self, BitcastPlan, CastKind};
use crate::air::ir::{BinOp as AirBin, FloatPredicate, IntPredicate};
use crate::device::DeviceProfile;
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage};
use crate::spirv::builder::{FunctionBuilder, MaskForm, ModuleBuilder, ProductForm, QuantizeForm};
use crate::spirv::error::BResult;
use crate::spirv::ids::{FuncId, GlobalId, TypeId, Value};
use crate::spirv::interp::{run, Buffer, Config, Resources};
use crate::spirv::ir::{
    natural_layout, BinOp, ConstBits, Constant, Contraction, ConvOp, ExecMode, ExecModel, GlslOp,
    Inst, IoDir, LayoutRule, MemAccess, StorageClass, Type, TypedModule,
};
use crate::spirv::testing::{assert_valid, committed_profiles};

// ---- Forms ----

fn int(bits: u8) -> Form {
    Form {
        held: Held::Int {
            bits,
            container: if bits <= 32 { 32 } else { 64 },
        },
        lanes: 1,
    }
}

fn float(bits: u8) -> Form {
    Form {
        held: Held::Float(match bits {
            16 => FloatFmt::Half,
            32 => FloatFmt::F32,
            _ => FloatFmt::F64,
        }),
        lanes: 1,
    }
}

fn bfloat() -> Form {
    Form {
        held: Held::Float(FloatFmt::BFloat),
        lanes: 1,
    }
}

fn boolean() -> Form {
    Form {
        held: Held::Bool,
        lanes: 1,
    }
}

fn vec(f: Form, lanes: u8) -> Form {
    Form { lanes, ..f }
}

fn mask(bits: u8) -> u64 {
    ops::ones(bits)
}

fn sext(v: u64, bits: u8) -> i64 {
    let sh = 64 - u32::from(bits);
    ((v << sh) as i64) >> sh
}

/// Words per lane.
fn lane_words(f: Form) -> usize {
    if f.container() == 64 {
        2
    } else {
        1
    }
}

fn words(f: Form) -> usize {
    lane_words(f) * usize::from(f.lanes)
}

// ---- Kernel ----

fn slot(index: u32) -> ResourceSlot {
    ResourceSlot {
        stage: Stage::Kernel,
        kind: ResourceKind::Buffer,
        index,
    }
}

fn u32_ty(b: &mut ModuleBuilder) -> BResult<TypeId> {
    b.ty(Type::Int {
        width: 32,
        signed: false,
    })
}

fn buffer(b: &mut ModuleBuilder, index: u32) -> BResult<GlobalId> {
    let u32_ = u32_ty(b)?;
    let rta = b.ty(Type::RuntimeArray {
        elem: u32_,
        stride: Some(4),
    })?;
    let layout = natural_layout(b.types(), LayoutRule::Std430, true, &[rta])?;
    let block = b.ty(Type::Struct {
        members: [rta].into(),
        layout: Some(layout),
    })?;
    b.resource(slot(index), StorageClass::StorageBuffer, block)
}

fn k32(fb: &mut FunctionBuilder<'_>, v: u32) -> BResult<Value> {
    let c = fb.u32_const(v)?;
    fb.module().const_value(c)
}

fn word_ptr(fb: &mut FunctionBuilder<'_>, buf: GlobalId, at: Value, k: u32) -> BResult<Value> {
    let zero = k32(fb, 0)?;
    let off = k32(fb, k)?;
    let i = fb.binary(BinOp::IAdd, at, off)?;
    let base = fb.module().global_value(buf)?;
    fb.access_chain(base, &[zero, i])
}

fn const_of(fb: &mut FunctionBuilder<'_>, ty: TypeId, bits: u64) -> BResult<Value> {
    let c = fb.module().constant(Constant::Scalar {
        ty,
        bits: ConstBits(bits),
    })?;
    fb.module().const_value(c)
}

/// Load one operand of form `f` from word `at + k` on.
fn load(fb: &mut FunctionBuilder<'_>, f: Form, buf: GlobalId, at: Value, k: u32) -> BResult<Value> {
    if f.is_wide() {
        // Two words per lane, low first.
        let mut ws = Vec::new();
        for j in 0..2 * u32::from(f.lanes) {
            let p = word_ptr(fb, buf, at, k + j)?;
            ws.push(fb.load(p, MemAccess::default())?);
        }
        return super::wide::from_words(fb, f, &ws);
    }
    let lane = f.scalar_form();
    let et = ops::ty(fb, lane)?;
    let mut lanes = Vec::new();
    let mut k = k;
    for _ in 0..f.lanes {
        let p = word_ptr(fb, buf, at, k)?;
        let w = fb.load(p, MemAccess::default())?;
        let v = match lane.held {
            Held::Bool => {
                let z = k32(fb, 0)?;
                fb.binary(BinOp::INotEqual, w, z)?
            }
            Held::Int { container: 32, .. } => w,
            Held::Float(FloatFmt::F32) => fb.convert(ConvOp::Bitcast, w, et)?,
            // A narrow float lane is its format bits in the low half of the word, below the
            // junk `encode` puts above them, which `from_bits` ignores.
            Held::Float(FloatFmt::Half | FloatFmt::BFloat) => {
                super::narrow::from_bits(fb, lane, w)?
            }
            _ => {
                let p2 = word_ptr(fb, buf, at, k + 1)?;
                let hi = fb.load(p2, MemAccess::default())?;
                let u64_ = fb.module().ty(Type::Int {
                    width: 64,
                    signed: false,
                })?;
                let lo = fb.convert(ConvOp::UConvert, w, u64_)?;
                let hi = fb.convert(ConvOp::UConvert, hi, u64_)?;
                let sh = const_of(fb, u64_, 32)?;
                let hi = fb.binary(BinOp::ShiftLeftLogical, hi, sh)?;
                let x = fb.binary(BinOp::BitwiseOr, lo, hi)?;
                if lane.is_float() {
                    fb.convert(ConvOp::Bitcast, x, et)?
                } else {
                    x
                }
            }
        };
        lanes.push(v);
        k += lane_words(lane) as u32;
    }
    if f.lanes == 1 {
        return lanes
            .first()
            .copied()
            .ok_or(crate::spirv::error::BuildError::IdSpaceExhausted);
    }
    let t = ops::ty(fb, f)?;
    fb.composite_construct(t, &lanes)
}

/// Store `v` of form `f` from word `at` on.
fn store(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    buf: GlobalId,
    at: Value,
    v: Value,
    lanes: bool,
) -> BResult<()> {
    if f.is_wide() {
        let ws = super::wide::to_words(fb, f, v)?;
        for (j, w) in ws.into_iter().enumerate() {
            let p = word_ptr(fb, buf, at, j as u32)?;
            fb.store(p, w, MemAccess::default())?;
        }
        return Ok(());
    }
    let lane = f.scalar_form();
    let u32_ = u32_ty(fb.module())?;
    // A narrow vector encodes once, as the memory leaf's store does; with `lanes`, each lane is
    // stored alone, as a scalarized store is, taking its bits from the vector's or weighing one
    // encode of the vector against the lanes' own.
    let narrow_bits = if f.is_narrow() && f.lanes > 1 && !lanes {
        Some(super::narrow::to_bits(fb, f, v)?)
    } else {
        None
    };
    let mut k = 0u32;
    for i in 0..f.lanes {
        if let Some(bits) = narrow_bits {
            let w = fb.composite_extract(bits, &[u32::from(i)])?;
            let p = word_ptr(fb, buf, at, k)?;
            fb.store(p, w, MemAccess::default())?;
            k += 1;
            continue;
        }
        let x = if f.lanes == 1 {
            v
        } else {
            let x = fb.composite_extract(v, &[u32::from(i)])?;
            if f.is_narrow() {
                fb.note_lane(x, v, u32::from(i));
            }
            x
        };
        let ws: Vec<Value> = match lane.held {
            Held::Bool => {
                let (one, zero) = (k32(fb, 1)?, k32(fb, 0)?);
                vec![fb.select(x, one, zero)?]
            }
            Held::Int { container: 32, .. } => vec![x],
            Held::Float(FloatFmt::F32) => vec![fb.convert(ConvOp::Bitcast, x, u32_)?],
            Held::Float(FloatFmt::Half | FloatFmt::BFloat) => {
                vec![super::narrow::to_bits(fb, lane, x)?]
            }
            _ => {
                let u64_ = fb.module().ty(Type::Int {
                    width: 64,
                    signed: false,
                })?;
                let x = if lane.is_float() {
                    fb.convert(ConvOp::Bitcast, x, u64_)?
                } else {
                    x
                };
                let lo = fb.convert(ConvOp::UConvert, x, u32_)?;
                let sh = const_of(fb, u64_, 32)?;
                let hi = fb.binary(BinOp::ShiftRightLogical, x, sh)?;
                let hi = fb.convert(ConvOp::UConvert, hi, u32_)?;
                vec![lo, hi]
            }
        };
        for w in ws {
            let p = word_ptr(fb, buf, at, k)?;
            fb.store(p, w, MemAccess::default())?;
            k += 1;
        }
    }
    Ok(())
}

type Body<'a> = &'a dyn Fn(&mut FunctionBuilder<'_>, &[Value]) -> BResult<Value>;

fn kernel(ins: &[Form], out: Form, body: Body<'_>) -> BResult<TypedModule> {
    kernel_with(ins, out, body, false, false)
}

/// [`kernel`], with what nothing reads removed and the lowering's offers weighed, as
/// `emit_module` does when `dce`, and a narrow vector result stored lane by lane when `lanes`.
fn kernel_with(
    ins: &[Form],
    out: Form,
    body: Body<'_>,
    dce: bool,
    lanes: bool,
) -> BResult<TypedModule> {
    kernel_defs(
        ins,
        out,
        &|_| Ok(Vec::new()),
        &|fb, xs, _| body(fb, xs),
        dce,
        lanes,
    )
}

/// The functions a [`kernel_defs`] module defines before its entry.
type Defs<'a> = &'a dyn Fn(&mut ModuleBuilder) -> BResult<Vec<FuncId>>;
/// A [`kernel_defs`] body, given the functions [`Defs`] defined.
type BodyCalling<'a> = &'a dyn Fn(&mut FunctionBuilder<'_>, &[Value], &[FuncId]) -> BResult<Value>;

/// [`kernel_with`], whose module first defines the functions `defs` returns, which `body` may
/// call.
fn kernel_defs(
    ins: &[Form],
    out: Form,
    defs: Defs<'_>,
    body: BodyCalling<'_>,
    dce: bool,
    lanes: bool,
) -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let outb = buffer(&mut b, 0)?;
    let inb = buffer(&mut b, 1)?;
    let funcs = defs(&mut b)?;
    let void = b.ty(Type::Void)?;
    let mut fb = b.function(void)?;
    let u32_ = u32_ty(fb.module())?;
    let v3 = fb.module().ty(Type::Vector { elem: u32_, len: 3 })?;
    let g = fb
        .module()
        .builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, v3)?;
    let gp = fb.module().global_value(g)?;
    let gid = fb.load(gp, MemAccess::default())?;
    let i = fb.composite_extract(gid, &[0])?;
    let per_in: usize = ins.iter().map(|f| words(*f)).sum();
    let kin = k32(&mut fb, per_in as u32)?;
    let at_in = fb.binary(BinOp::IMul, i, kin)?;
    let kout = k32(&mut fb, words(out) as u32)?;
    let at_out = fb.binary(BinOp::IMul, i, kout)?;
    let mut args = Vec::new();
    let mut k = 0u32;
    for f in ins {
        args.push(load(&mut fb, *f, inb, at_in, k)?);
        k += words(*f) as u32;
    }
    let r = body(&mut fb, &args, &funcs)?;
    store(&mut fb, out, outb, at_out, r, lanes)?;
    fb.ret(None)?;
    let id = fb.finish()?;
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize { x: 1, y: 1, z: 1 }],
    )?;
    let offers = b.offers().to_vec();
    let mut m = b.finalize()?;
    if dce {
        let budget = crate::budget::Budget::default();
        let mut meter = crate::budget::Meter::new(&budget);
        if let Err(e) = crate::spirv::dead::remove_dead_keeping(&mut m, &offers, &[], &mut meter) {
            panic!("remove_dead_keeping: {e:?}");
        }
    }
    Ok(m)
}

/// A committed profile with 64-bit integers and floats and 16-bit floats (for L3).
fn wide_profile() -> DeviceProfile {
    let all = committed_profiles().expect("profiles");
    all.into_iter()
        .map(|(_, p)| p)
        .find(|p| {
            let l = &p.fields().leaf;
            l.int64 && l.float64 && l.float16
        })
        .expect("a committed profile with int64, float64, and float16")
}

/// The `half` roundings the blends' checks run under (`extended_range::blend::half_of`): the
/// integer sequence every device answers alike, and the device's conversion under the 16-bit float
/// controls, each checked against Metal's arithmetic.
const HALF_FORMS: [QuantizeForm; 2] = [QuantizeForm::Integer, QuantizeForm::Controlled];

/// One case: per operand, per lane, the held bits.
type Case = Vec<Vec<u64>>;

/// The bits `encode` puts above a narrow float lane's 16 format bits, which `narrow::from_bits`
/// must ignore.
const NARROW_JUNK: u32 = 0xA5C3_0000;

fn encode(f: Form, lanes: &[u64], out: &mut Vec<u32>) {
    let junk = match f.held {
        Held::Float(FloatFmt::Half | FloatFmt::BFloat) => NARROW_JUNK,
        _ => 0,
    };
    for b in lanes {
        out.push(*b as u32 | junk);
        if lane_words(f) == 2 {
            out.push((*b >> 32) as u32);
        }
    }
}

fn decode(f: Form, ws: &[u32]) -> Vec<u64> {
    let lw = lane_words(f);
    ws.chunks(lw)
        .map(|c| {
            let lo = u64::from(c.first().copied().unwrap_or(0));
            let hi = u64::from(c.get(1).copied().unwrap_or(0));
            lo | (hi << 32)
        })
        .collect()
}

fn is_nan(f: Form, b: u64) -> bool {
    match f.held {
        Held::Float(FloatFmt::Half) => (b & 0x7C00) == 0x7C00 && (b & 0x3FF) != 0,
        Held::Float(FloatFmt::BFloat) => (b & 0x7F80) == 0x7F80 && (b & 0x7F) != 0,
        Held::Float(FloatFmt::F32) => f32::from_bits(b as u32).is_nan(),
        Held::Float(FloatFmt::F64) => f64::from_bits(b).is_nan(),
        _ => false,
    }
}

fn is_zero(f: Form, b: u64) -> bool {
    f.is_float() && (b << 1) & mask(f.bits()) == 0
}

/// Run `body` over every case whose reference is defined and compare. `loose_zero` accepts
/// either signed zero (min/max of zeros may return either).
fn check(
    what: &str,
    ins: &[Form],
    out: Form,
    body: Body<'_>,
    cases: &[Case],
    reference: &dyn Fn(&Case) -> Option<Vec<u64>>,
    loose_zero: bool,
) {
    let accept = |g: u64, w: u64| {
        g == w
            || (is_nan(out, g) && is_nan(out, w))
            || (loose_zero && is_zero(out, g) && is_zero(out, w))
    };
    check_accept(what, ins, out, body, cases, reference, &accept);
}

/// [`check`] with the lane comparison `accept(got, want)`.
fn check_accept(
    what: &str,
    ins: &[Form],
    out: Form,
    body: Body<'_>,
    cases: &[Case],
    reference: &dyn Fn(&Case) -> Option<Vec<u64>>,
    accept: &dyn Fn(u64, u64) -> bool,
) {
    // Through the emitter's dead-code removal and offers, as the product runs it.
    let m =
        kernel_with(ins, out, body, true, false).unwrap_or_else(|e| panic!("{what}: build: {e:?}"));
    check_module(what, m, ins, out, cases, reference, accept);
}

/// Run module `m` (a [`kernel`] of `ins` and `out`) over every case whose reference is defined
/// and compare with `accept(got, want)`.
fn check_module(
    what: &str,
    m: TypedModule,
    ins: &[Form],
    out: Form,
    cases: &[Case],
    reference: &dyn Fn(&Case) -> Option<Vec<u64>>,
    accept: &dyn Fn(u64, u64) -> bool,
) {
    let mut want = Vec::new();
    let mut kept = Vec::new();
    for c in cases {
        let Some(r) = reference(c) else { continue };
        want.push(r);
        kept.push(c.clone());
    }
    assert!(!kept.is_empty(), "{what}: no defined case");
    let n = kept.len();
    let outputs = run_module(what, m, ins, out, &kept);
    let mut failures = Vec::new();
    for ((c, w), got) in kept.iter().zip(&want).zip(&outputs) {
        let same = got.len() == w.len() && got.iter().zip(w).all(|(g, w)| accept(*g, *w));
        if !same {
            failures.push(format!("{c:x?}: got {got:x?}, want {w:x?}"));
        }
    }
    assert!(
        failures.is_empty(),
        "{what}: {} of {n} cases differ:\n{}",
        failures.len(),
        failures
            .iter()
            .take(12)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}

/// The most cases one interpreter dispatch of [`run_module`] runs: the interpreter's step limit
/// (`Config::step_limit`) bounds a dispatch, so a routine of thousands of steps (a precise `tan`
/// through the integer division routine) runs its cases in several.
const CASES_PER_DISPATCH: usize = 1024;

/// Run module `m` (a [`kernel`] of `ins` and `out`, valid for L3) on the interpreter over
/// `cases`, one invocation each, in dispatches of at most [`CASES_PER_DISPATCH`], and return each
/// case's result lanes.
fn run_module(
    what: &str,
    m: TypedModule,
    ins: &[Form],
    out: Form,
    cases: &[Case],
) -> Vec<Vec<u64>> {
    assert_valid(m.clone(), &wide_profile());
    let mut lanes = Vec::with_capacity(cases.len());
    for part in cases.chunks(CASES_PER_DISPATCH) {
        lanes.extend(dispatch_module(what, &m, ins, out, part));
    }
    lanes
}

/// [`run_module`]'s dispatch of `cases`, one invocation each, after its validation.
fn dispatch_module(
    what: &str,
    m: &TypedModule,
    ins: &[Form],
    out: Form,
    cases: &[Case],
) -> Vec<Vec<u64>> {
    let mut input = Vec::new();
    for c in cases {
        for (f, lanes) in ins.iter().zip(c) {
            encode(*f, lanes, &mut input);
        }
    }
    let n = cases.len();
    let res = Resources {
        buffers: vec![
            Buffer {
                slot: Some(slot(0)),
                address: None,
                bytes: vec![0; n * words(out) * 4],
            },
            Buffer {
                slot: Some(slot(1)),
                address: None,
                bytes: input.iter().flat_map(|w| w.to_le_bytes()).collect(),
            },
            // The color table a kernel that rounds an sRGB channel declares, bound as the binder
            // binds it.
            Buffer {
                slot: Some(super::srgb::color_table_symbol(Stage::Kernel)),
                address: None,
                bytes: super::srgb::color_write_table().to_vec(),
            },
        ],
        push: Vec::new(),
        spec: Default::default(),
    };
    let cfg = Config {
        workgroups: [u32::try_from(n).expect("cases"), 1, 1],
        ..Config::default()
    };
    let outcome = run(m, &cfg, res).unwrap_or_else(|e| {
        let at = match &e {
            crate::spirv::interp::InterpError::Trap(t) => t.invocation.map(|i| i[0] as usize),
            _ => None,
        };
        panic!(
            "{what}: interpreter: {e} (case {:?})",
            at.and_then(|i| cases.get(i))
        )
    });
    let bytes = outcome.buffers.first().expect("output");
    let ws: Vec<u32> = bytes
        .chunks(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    (0..n)
        .map(|i| decode(out, &ws[i * words(out)..(i + 1) * words(out)]))
        .collect()
}

// ---- Parts ----

/// The items of `all` in part `part` of `of`: every `of`-th from the `part`-th, so each part takes
/// its share of every kind of item.
fn part_of<T>(all: Vec<T>, part: usize, of: usize) -> Vec<T> {
    all.into_iter().skip(part).step_by(of).collect()
}

/// One test per part of a grid (`$grid(part, of)`), so nextest runs the parts at once.
macro_rules! parts {
    ($grid:path, $of:expr; $($test:ident = $part:expr),* $(,)?) => {
        $(
            #[test]
            fn $test() {
                $grid($part, $of);
            }
        )*
    };
}

// ---- Inputs ----

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// Interesting and pseudo-random values of an `n`-bit integer.
fn int_values(bits: u8, rng: &mut Rng, random: usize) -> Vec<u64> {
    let m = mask(bits);
    let min = 1u64 << (bits - 1);
    let mut v = vec![
        0,
        1,
        2,
        3,
        7,
        m,
        m - 1,
        min,
        min - 1,
        min + 1,
        m >> 2,
        0x55 & m,
    ];
    for _ in 0..random {
        v.push(rng.next() & m);
    }
    v.sort_unstable();
    v.dedup();
    v
}

fn pairs(a: &[u64], b: &[u64]) -> Vec<Case> {
    a.iter()
        .flat_map(|x| b.iter().map(move |y| vec![vec![*x], vec![*y]]))
        .collect()
}

fn f32_values() -> Vec<u64> {
    [
        0.0f32,
        -0.0,
        1.0,
        -1.0,
        0.5,
        -0.5,
        1.5,
        -1.5,
        2.5,
        -2.5,
        3.7,
        -3.7,
        0.49999997,
        -0.49999997,
        1e10,
        -1e10,
        8388609.0,
        -8388609.0,
        127.5,
        -128.5,
        255.9,
        256.0,
        -129.0,
        32767.9,
        -32769.0,
        65535.5,
        2147483520.0,
        2147483648.0,
        -2147483648.0,
        -2147483904.0,
        4294967040.0,
        4294967296.0,
        9.223372e18,
        -9.223372e18,
        1.8446743e19,
        f32::MIN_POSITIVE,
        1e-40,
        -1e-40,
        f32::MAX,
        f32::MIN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ]
    .iter()
    .map(|x| u64::from(x.to_bits()))
    .collect()
}

fn f64_values() -> Vec<u64> {
    f32_values()
        .iter()
        .map(|b| f64::from(f32::from_bits(*b as u32)).to_bits())
        .chain(
            [0.1f64, -0.1, 9.223372036854776e18, 4503599627370497.0]
                .iter()
                .map(|x| x.to_bits()),
        )
        .collect()
}

fn f32v(b: u64) -> f32 {
    f32::from_bits(b as u32)
}

fn f64v(b: u64) -> f64 {
    f64::from_bits(b)
}

/// The value of float bits `b` of width `bits`, as f64.
fn fval(bits: u8, b: u64) -> f64 {
    match bits {
        32 => f64::from(f32v(b)),
        _ => f64v(b),
    }
}

fn fbits(bits: u8, x: f64) -> u64 {
    match bits {
        32 => u64::from((x as f32).to_bits()),
        _ => x.to_bits(),
    }
}

fn arg(c: &Case, i: usize) -> u64 {
    c[i][0]
}

// ---- Integer operators ----

/// LLVM integer binary operator semantics; `None` is poison or undefined behavior.
fn llvm_int(op: AirBin, bits: u8, a: u64, b: u64) -> Option<u64> {
    let m = mask(bits);
    let (sa, sb) = (sext(a, bits), sext(b, bits));
    Some(
        match op {
            AirBin::Add => a.wrapping_add(b),
            AirBin::Sub => a.wrapping_sub(b),
            AirBin::Mul => a.wrapping_mul(b),
            AirBin::UDiv => a.checked_div(b)?,
            AirBin::URem => a.checked_rem(b)?,
            AirBin::SDiv | AirBin::SRem => {
                if b == 0 || (sa == sext(1 << (bits - 1), bits) && sb == -1) {
                    return None;
                }
                (if op == AirBin::SDiv { sa / sb } else { sa % sb }) as u64
            }
            AirBin::Shl | AirBin::LShr | AirBin::AShr => {
                if b >= u64::from(bits) {
                    return None;
                }
                match op {
                    AirBin::Shl => a << b,
                    AirBin::LShr => a >> b,
                    _ => (sa >> b) as u64,
                }
            }
            AirBin::And => a & b,
            AirBin::Or => a | b,
            AirBin::Xor => a ^ b,
            _ => return None,
        } & m,
    )
}

const INT_OPS: [AirBin; 13] = [
    AirBin::Add,
    AirBin::Sub,
    AirBin::Mul,
    AirBin::UDiv,
    AirBin::SDiv,
    AirBin::URem,
    AirBin::SRem,
    AirBin::Shl,
    AirBin::LShr,
    AirBin::AShr,
    AirBin::And,
    AirBin::Or,
    AirBin::Xor,
];

/// Every widths class: legalized in 32 (odd, 8, 16), full 32, legalized in 64 (i40), full 64.
const WIDTHS: [u8; 6] = [5, 8, 16, 32, 40, 64];

#[test]
fn integer_operators_match_llvm_at_every_width() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    for bits in WIDTHS {
        let vals = int_values(bits, &mut rng, 6);
        let mut shifts = vals.clone();
        shifts.extend(0..u64::from(bits));
        let f = int(bits);
        for op in INT_OPS {
            let b: &[u64] = if matches!(op, AirBin::Shl | AirBin::LShr | AirBin::AShr) {
                &shifts
            } else {
                &vals
            };
            check(
                &format!("{op:?} i{bits}"),
                &[f, f],
                f,
                &|fb, x| ops::int_binary(fb, op, f, x[0], x[1]),
                &pairs(&vals, b),
                &|c| Some(vec![llvm_int(op, bits, arg(c, 0), arg(c, 1))?]),
                false,
            );
        }
    }
}

#[test]
fn integer_operators_on_vectors_work_per_lane() {
    let f = vec(int(8), 4);
    let cases: Vec<Case> = vec![
        vec![vec![0xFF, 0x80, 1, 7], vec![1, 0xFF, 0x80, 3]],
        vec![vec![0x7F, 0, 0x10, 0xFE], vec![0x81, 2, 0x10, 0x7F]],
        vec![vec![0x80, 0x7F, 0xFF, 1], vec![1, 7, 3, 0]],
    ];
    for op in [AirBin::Add, AirBin::Mul, AirBin::AShr, AirBin::SDiv] {
        check(
            &format!("{op:?} <4 x i8>"),
            &[f, f],
            f,
            &|fb, x| ops::int_binary(fb, op, f, x[0], x[1]),
            &cases,
            &|c| {
                (0..4)
                    .map(|l| llvm_int(op, 8, c[0][l], c[1][l]))
                    .collect::<Option<Vec<_>>>()
            },
            false,
        );
    }
}

fn llvm_icmp(pred: IntPredicate, bits: u8, a: u64, b: u64) -> bool {
    use IntPredicate as P;
    let (sa, sb) = (sext(a, bits), sext(b, bits));
    match pred {
        P::Eq => a == b,
        P::Ne => a != b,
        P::Ugt => a > b,
        P::Uge => a >= b,
        P::Ult => a < b,
        P::Ule => a <= b,
        P::Sgt => sa > sb,
        P::Sge => sa >= sb,
        P::Slt => sa < sb,
        P::Sle => sa <= sb,
    }
}

#[test]
fn integer_and_boolean_comparisons_match_llvm() {
    let mut rng = Rng(7);
    for bits in WIDTHS {
        let vals = int_values(bits, &mut rng, 4);
        let f = int(bits);
        for pred in IntPredicate::ALL {
            check(
                &format!("icmp {pred:?} i{bits}"),
                &[f, f],
                boolean(),
                &|fb, x| ops::int_compare(fb, pred, f, x[0], x[1]),
                &pairs(&vals, &vals),
                &|c| Some(vec![u64::from(llvm_icmp(pred, bits, arg(c, 0), arg(c, 1)))]),
                false,
            );
        }
    }
    let bools = [0u64, 1];
    for pred in IntPredicate::ALL {
        check(
            &format!("icmp {pred:?} i1"),
            &[boolean(), boolean()],
            boolean(),
            &|fb, x| ops::bool_compare(fb, pred, x[0], x[1]),
            &pairs(&bools, &bools),
            &|c| Some(vec![u64::from(llvm_icmp(pred, 1, arg(c, 0), arg(c, 1)))]),
            false,
        );
    }
    for op in INT_OPS {
        check(
            &format!("{op:?} i1"),
            &[boolean(), boolean()],
            boolean(),
            &|fb, x| ops::bool_binary(fb, op, boolean(), x[0], x[1]),
            &pairs(&bools, &bools),
            &|c| Some(vec![llvm_int(op, 1, arg(c, 0), arg(c, 1))?]),
            false,
        );
    }
}

// ---- Floats ----

fn llvm_fcmp(pred: FloatPredicate, a: f64, b: f64) -> bool {
    use FloatPredicate as P;
    let uno = a.is_nan() || b.is_nan();
    match pred {
        P::False => false,
        P::True => true,
        P::Ord => !uno,
        P::Uno => uno,
        P::Oeq => a == b,
        P::Ogt => a > b,
        P::Oge => a >= b,
        P::Olt => a < b,
        P::Ole => a <= b,
        P::One => !uno && a != b,
        P::Ueq => uno || a == b,
        P::Ugt => uno || a > b,
        P::Uge => uno || a >= b,
        P::Ult => uno || a < b,
        P::Ule => uno || a <= b,
        P::Une => uno || a != b,
    }
}

/// A multiply rounded on its own is IEEE's product bit for bit on every pair of the value sets,
/// `-0.0` operands, zero products of either sign, denormals, overflow, infinities, and NaNs
/// included, on `float` scalars and vectors and on `double`, in either of its forms (`OpFMul`, and
/// `Fma(a, b, -0.0)` where SPIRV-Cross renders the SPIR-V).
#[test]
fn a_multiply_without_contraction_is_the_product() {
    for (form, bits, vals, lanes) in [ProductForm::Multiply, ProductForm::NegativeZeroFma]
        .into_iter()
        .flat_map(|form| {
            [
                (form, 32u8, f32_values(), 1u8),
                (form, 32, f32_values(), 4),
                (form, 64, f64_values(), 1),
            ]
        })
    {
        let f = vec(float(bits), lanes);
        let cases: Vec<Case> = pairs(&vals, &vals)
            .into_iter()
            .map(|c| {
                c.into_iter()
                    .map(|v| vec![v[0]; usize::from(lanes)])
                    .collect()
            })
            .collect();
        check(
            &format!("fmul without contraction f{bits} x{lanes} as {form:?}"),
            &[f, f],
            f,
            &|fb, x| {
                fb.module().product_form(form);
                ops::float_binary(fb, AirBin::FMul, f, Contraction::Forbidden, x[0], x[1])
            },
            &cases,
            &|c| {
                let r = |l: usize| {
                    let (a, b) = (c[0][l], c[1][l]);
                    if bits == 32 {
                        u64::from((f32::from_bits(a as u32) * f32::from_bits(b as u32)).to_bits())
                    } else {
                        (f64::from_bits(a) * f64::from_bits(b)).to_bits()
                    }
                };
                Some((0..usize::from(lanes)).map(r).collect())
            },
            false,
        );
    }
}

#[test]
fn float_operators_and_comparisons_match_ieee() {
    for (bits, vals) in [(32u8, f32_values()), (64, f64_values())] {
        let f = float(bits);
        let small: Vec<u64> = vals.iter().copied().step_by(3).collect();
        for op in [
            AirBin::FAdd,
            AirBin::FSub,
            AirBin::FMul,
            AirBin::FDiv,
            AirBin::FRem,
        ] {
            check(
                &format!("{op:?} f{bits}"),
                &[f, f],
                f,
                &|fb, x| ops::float_binary(fb, op, f, Contraction::Forbidden, x[0], x[1]),
                &pairs(&small, &small),
                &|c| {
                    let (a, b) = (fval(bits, arg(c, 0)), fval(bits, arg(c, 1)));
                    let r = if bits == 32 {
                        let (a, b) = (a as f32, b as f32);
                        f64::from(match op {
                            AirBin::FAdd => a + b,
                            AirBin::FSub => a - b,
                            AirBin::FMul => a * b,
                            AirBin::FDiv => a / b,
                            _ => a % b,
                        })
                    } else {
                        match op {
                            AirBin::FAdd => a + b,
                            AirBin::FSub => a - b,
                            AirBin::FMul => a * b,
                            AirBin::FDiv => a / b,
                            _ => a % b,
                        }
                    };
                    Some(vec![fbits(bits, r)])
                },
                false,
            );
        }
        for pred in FloatPredicate::ALL {
            check(
                &format!("fcmp {pred:?} f{bits}"),
                &[f, f],
                boolean(),
                &|fb, x| ops::float_compare(fb, pred, f, x[0], x[1]),
                &pairs(&small, &small),
                &|c| {
                    let (a, b) = (fval(bits, arg(c, 0)), fval(bits, arg(c, 1)));
                    Some(vec![u64::from(llvm_fcmp(pred, a, b))])
                },
                false,
            );
        }
    }
}

// ---- Casts ----

#[test]
fn integer_casts_match_llvm() {
    let mut rng = Rng(99);
    let widths = [1u8, 5, 8, 16, 32, 40, 64];
    for from in widths {
        for to in widths {
            let ff = if from == 1 { boolean() } else { int(from) };
            let tf = if to == 1 { boolean() } else { int(to) };
            let vals = if from == 1 {
                vec![0, 1]
            } else {
                int_values(from, &mut rng, 4)
            };
            let cases: Vec<Case> = vals.iter().map(|v| vec![vec![*v]]).collect();
            let kinds: &[CastKind] = if to < from {
                &[CastKind::Trunc]
            } else if to > from {
                &[CastKind::ZExt, CastKind::SExt]
            } else {
                &[]
            };
            for k in kinds {
                check(
                    &format!("{k:?} i{from} to i{to}"),
                    &[ff],
                    tf,
                    &|fb, x| ops::cast(fb, *k, ff, tf, x[0]),
                    &cases,
                    &|c| {
                        let v = arg(c, 0);
                        Some(vec![match k {
                            CastKind::SExt => sext(v, from) as u64 & mask(to),
                            _ => v & mask(to),
                        }])
                    },
                    false,
                );
            }
        }
    }
}

#[test]
fn float_integer_casts_match_llvm() {
    let mut rng = Rng(5);
    for fbits_ in [32u8, 64] {
        let ff = float(fbits_);
        let fvals = if fbits_ == 32 {
            f32_values()
        } else {
            f64_values()
        };
        for ibits in [1u8, 5, 8, 16, 32, 40, 64] {
            let inf = if ibits == 1 { boolean() } else { int(ibits) };
            let fcases: Vec<Case> = fvals.iter().map(|v| vec![vec![*v]]).collect();
            for (k, signed) in [(CastKind::FpToUi, false), (CastKind::FpToSi, true)] {
                check(
                    &format!("{k:?} f{fbits_} to i{ibits}"),
                    &[ff],
                    inf,
                    &|fb, x| ops::cast(fb, k, ff, inf, x[0]),
                    &fcases,
                    &|c| {
                        let t = fval(fbits_, arg(c, 0)).trunc();
                        let n = i32::from(ibits);
                        let (lo, hi) = if signed {
                            (-(2f64.powi(n - 1)), 2f64.powi(n - 1))
                        } else {
                            (0.0, 2f64.powi(n))
                        };
                        // Out of range (and NaN) is poison.
                        if !(t >= lo && t < hi) {
                            return None;
                        }
                        Some(vec![(t as i128 as u64) & mask(ibits)])
                    },
                    false,
                );
            }
            let ivals = if ibits == 1 {
                vec![0, 1]
            } else {
                int_values(ibits, &mut rng, 4)
            };
            let icases: Vec<Case> = ivals.iter().map(|v| vec![vec![*v]]).collect();
            for (k, signed) in [(CastKind::UiToFp, false), (CastKind::SiToFp, true)] {
                check(
                    &format!("{k:?} i{ibits} to f{fbits_}"),
                    &[inf],
                    ff,
                    &|fb, x| ops::cast(fb, k, inf, ff, x[0]),
                    &icases,
                    &|c| {
                        let v = arg(c, 0);
                        let x = if signed {
                            let s = if ibits == 1 {
                                -(v as i64)
                            } else {
                                sext(v, ibits)
                            };
                            if fbits_ == 32 {
                                f64::from(s as f32)
                            } else {
                                s as f64
                            }
                        } else if fbits_ == 32 {
                            f64::from(v as f32)
                        } else {
                            v as f64
                        };
                        Some(vec![fbits(fbits_, x)])
                    },
                    false,
                );
            }
        }
    }
    // fptrunc / fpext.
    let (f32f, f64f) = (float(32), float(64));
    let wide: Vec<Case> = f64_values().iter().map(|v| vec![vec![*v]]).collect();
    check(
        "fptrunc f64 to f32",
        &[f64f],
        f32f,
        &|fb, x| ops::cast(fb, CastKind::FpResize, f64f, f32f, x[0]),
        &wide,
        &|c| Some(vec![u64::from((f64v(arg(c, 0)) as f32).to_bits())]),
        false,
    );
    let narrow: Vec<Case> = f32_values().iter().map(|v| vec![vec![*v]]).collect();
    check(
        "fpext f32 to f64",
        &[f32f],
        f64f,
        &|fb, x| ops::cast(fb, CastKind::FpResize, f32f, f64f, x[0]),
        &narrow,
        &|c| Some(vec![f64::from(f32v(arg(c, 0))).to_bits()]),
        false,
    );
}

#[test]
fn bitcasts_pack_and_unpack_lanes_low_first() {
    let cases: Vec<(Form, Form)> = vec![
        (vec(int(8), 4), int(32)),
        (int(32), vec(int(8), 4)),
        (vec(int(16), 2), int(32)),
        (vec(int(16), 4), int(64)),
        (int(64), vec(int(16), 4)),
        (vec(int(8), 2), int(16)),
        (int(16), vec(int(8), 2)),
        (vec(int(32), 2), int(64)),
        (int(64), vec(float(32), 2)),
        (vec(float(32), 2), vec(int(16), 4)),
        (float(32), int(32)),
        (int(64), float(64)),
        (vec(int(8), 4), float(32)),
        // Narrow float lanes move through their exact format bits.
        (float(16), int(16)),
        (int(16), float(16)),
        (bfloat(), int(16)),
        (vec(float(16), 2), int(32)),
        (int(32), vec(float(16), 2)),
        (vec(float(16), 2), float(32)),
        (vec(float(16), 4), vec(float(32), 2)),
        (vec(float(32), 2), vec(float(16), 4)),
        (vec(float(16), 4), int(64)),
        (float(64), vec(float(16), 4)),
        (vec(bfloat(), 4), vec(int(16), 4)),
        (vec(float(16), 2), vec(bfloat(), 2)),
        (vec(int(8), 4), vec(int(16), 2)),
        (vec(int(16), 4), vec(int(32), 2)),
        // Three 16-bit lanes are no whole number of words: lane by lane.
        (vec(float(16), 3), vec(int(16), 3)),
        (vec(int(16), 3), vec(float(16), 3)),
        (vec(bfloat(), 3), vec(float(16), 3)),
    ];
    let mut rng = Rng(3);
    for (from, to) in cases {
        let plan = ops::bitcast_plan(from, to).expect("a plan");
        let lane_mask = mask(from.bits());
        let inputs: Vec<Case> = (0..16)
            .map(|_| {
                vec![(0..from.lanes)
                    .map(|_| {
                        let mut v = rng.next() & lane_mask;
                        // Keep float inputs away from NaN payloads.
                        if from.is_float() && is_nan(from, v) {
                            v = 0;
                        }
                        v
                    })
                    .collect()]
            })
            .collect();
        check(
            &format!("bitcast {from:?} to {to:?}"),
            &[from],
            to,
            &|fb, x| ops::bitcast(fb, plan, from, to, x[0]),
            &inputs,
            &|c| {
                let mut all: u128 = 0;
                for (i, l) in c[0].iter().enumerate() {
                    all |= u128::from(*l) << (i as u32 * u32::from(from.bits()));
                }
                let out: Vec<u64> = (0..to.lanes)
                    .map(|i| {
                        ((all >> (u32::from(i) * u32::from(to.bits()))) as u64) & mask(to.bits())
                    })
                    .collect();
                if to.is_float() && out.iter().any(|b| is_nan(to, *b)) {
                    return None;
                }
                Some(out)
            },
            false,
        );
        if !from.padded() && !to.padded() {
            assert_eq!(plan, BitcastPlan::Direct);
        }
        // Lanes of one count and width never pack into a 64-bit integer.
        if from.lanes == to.lanes && from.bits() == to.bits() {
            assert!(
                !matches!(plan, BitcastPlan::Packed { total } if total > 32),
                "{from:?} to {to:?}: {plan:?}"
            );
        }
    }
}

// ---- Integer functions ----

fn umax(bits: u8) -> i128 {
    i128::from(mask(bits))
}

fn smin(bits: u8) -> i128 {
    -(1i128 << (bits - 1))
}

fn smax(bits: u8) -> i128 {
    (1i128 << (bits - 1)) - 1
}

/// Metal / LLVM reference of an integer function; `None` is undefined.
fn int_ref(op: IntFn, bits: u8, a: &[u64]) -> Option<u64> {
    let m = mask(bits);
    let n = u32::from(bits);
    let s = |i: usize| i128::from(sext(a[i], bits));
    let u = |i: usize| i128::from(a[i]);
    let val = |sg: Sign, i: usize| if sg == Sign::Signed { s(i) } else { u(i) };
    let wrap = |x: i128| (x as u64) & m;
    // The high half of the double-width product (u128 for unsigned 64-bit operands).
    let hi = |sg: Sign| -> i128 {
        if sg == Sign::Signed {
            (s(0) * s(1)) >> n
        } else {
            ((u128::from(a[0]) * u128::from(a[1])) >> n) as i128
        }
    };
    Some(match op {
        IntFn::Abs => wrap(s(0).abs()),
        IntFn::AbsDiff(sg) => wrap((val(sg, 0) - val(sg, 1)).abs()),
        IntFn::Min(sg) => wrap(val(sg, 0).min(val(sg, 1))),
        IntFn::Max(sg) => wrap(val(sg, 0).max(val(sg, 1))),
        IntFn::Min3(sg) => wrap(val(sg, 0).min(val(sg, 1)).min(val(sg, 2))),
        IntFn::Max3(sg) => wrap(val(sg, 0).max(val(sg, 1)).max(val(sg, 2))),
        IntFn::Clamp(sg) => wrap(val(sg, 0).max(val(sg, 1)).min(val(sg, 2))),
        IntFn::MulHi(sg) => wrap(hi(sg)),
        IntFn::MadHi(sg) => wrap(hi(sg) + u(2)),
        IntFn::MadSat(Sign::Signed) => wrap((s(0) * s(1) + s(2)).clamp(smin(bits), smax(bits))),
        IntFn::MadSat(Sign::Unsigned) => {
            let r = u128::from(a[0]) * u128::from(a[1]) + u128::from(a[2]);
            (r.min(umax(bits) as u128)) as u64
        }
        IntFn::AddSat(sg) | IntFn::SubSat(sg) => {
            let r = if matches!(op, IntFn::AddSat(_)) {
                val(sg, 0) + val(sg, 1)
            } else {
                val(sg, 0) - val(sg, 1)
            };
            let (lo, hi) = if sg == Sign::Signed {
                (smin(bits), smax(bits))
            } else {
                (0, umax(bits))
            };
            wrap(r.clamp(lo, hi))
        }
        IntFn::HAdd(sg) => wrap((val(sg, 0) + val(sg, 1)) >> 1),
        IntFn::RHAdd(sg) => wrap((val(sg, 0) + val(sg, 1) + 1) >> 1),
        IntFn::ExtractBits(sg) => {
            let (off, cnt) = (a[1], a[2]);
            if off + cnt > u64::from(bits) {
                return None;
            }
            if cnt == 0 {
                return Some(0);
            }
            let field = (a[0] >> off) & mask(cnt as u8);
            if sg == Sign::Signed {
                (sext(field, cnt as u8) as u64) & m
            } else {
                field
            }
        }
        IntFn::InsertBits => {
            let (off, cnt) = (a[2], a[3]);
            if off + cnt > u64::from(bits) {
                return None;
            }
            if cnt == 0 {
                return Some(a[0]);
            }
            let fm = mask(cnt as u8) << off;
            (a[0] & !fm) | ((a[1] << off) & fm)
        }
        IntFn::Popcount => u64::from(a[0].count_ones()),
        IntFn::ReverseBits => a[0].reverse_bits() >> (64 - n),
        IntFn::Bswap => {
            let bytes = n / 8;
            (0..bytes).fold(0u64, |acc, i| {
                acc | (((a[0] >> (8 * i)) & 0xFF) << (8 * (bytes - 1 - i)))
            })
        }
        IntFn::Rotate => {
            let k = (a[1] % u64::from(n)) as u32;
            if k == 0 {
                a[0]
            } else {
                ((a[0] << k) | (a[0] >> (n - k))) & m
            }
        }
        IntFn::Fshl | IntFn::Fshr => {
            let k = (a[2] % u64::from(n)) as u32;
            let cat = (u128::from(a[0]) << n) | u128::from(a[1]);
            if op == IntFn::Fshl {
                ((cat << k) >> n) as u64 & m
            } else {
                (cat >> k) as u64 & m
            }
        }
        IntFn::Clz => {
            if a[0] == 0 {
                u64::from(n)
            } else {
                u64::from(a[0].leading_zeros() - (64 - n))
            }
        }
        IntFn::Ctz => {
            if a[0] == 0 {
                u64::from(n)
            } else {
                u64::from(a[0].trailing_zeros())
            }
        }
    })
}

#[test]
fn sixty_four_bit_bit_fields_work_per_lane() {
    let f = vec(int(64), 2);
    let (x, y) = (0xF0E1_D2C3_B4A5_9687u64, 0x0123_4567_89AB_CDEFu64);
    let mut ext = Vec::new();
    let mut ins = Vec::new();
    for (off, cnt) in [(0u64, 64u64), (60, 4), (4, 40), (63, 1), (17, 0)] {
        ext.push(vec![vec![x, y], vec![off], vec![cnt]]);
        ins.push(vec![vec![x, y], vec![y, x], vec![off], vec![cnt]]);
    }
    for s in [Sign::Signed, Sign::Unsigned] {
        let op = IntFn::ExtractBits(s);
        check(
            &format!("{op:?} <2 x i64>"),
            &[f, int(32), int(32)],
            f,
            &|fb, v| math::int_fn(fb, op, f, v),
            &ext,
            &|c| {
                (0..2)
                    .map(|l| int_ref(op, 64, &[c[0][l], c[1][0], c[2][0]]))
                    .collect::<Option<Vec<u64>>>()
            },
            false,
        );
    }
    check(
        "InsertBits <2 x i64>",
        &[f, f, int(32), int(32)],
        f,
        &|fb, v| math::int_fn(fb, IntFn::InsertBits, f, v),
        &ins,
        &|c| {
            (0..2)
                .map(|l| int_ref(IntFn::InsertBits, 64, &[c[0][l], c[1][l], c[2][0], c[3][0]]))
                .collect::<Option<Vec<u64>>>()
        },
        false,
    );
}

/// Every integer function answers its definition at every width: part `part` of `of` of the
/// functions at each width.
fn integer_functions(part: usize, of: usize) {
    let mut rng = Rng(0xDEAD_BEEF);
    let signs = [Sign::Signed, Sign::Unsigned];
    let mut fns: Vec<IntFn> = vec![
        IntFn::Abs,
        IntFn::InsertBits,
        IntFn::Popcount,
        IntFn::ReverseBits,
        IntFn::Bswap,
        IntFn::Rotate,
        IntFn::Clz,
        IntFn::Ctz,
        IntFn::Fshl,
        IntFn::Fshr,
    ];
    for s in signs {
        fns.extend([
            IntFn::AbsDiff(s),
            IntFn::Min(s),
            IntFn::Max(s),
            IntFn::Clamp(s),
            IntFn::Min3(s),
            IntFn::Max3(s),
            IntFn::MulHi(s),
            IntFn::MadHi(s),
            IntFn::MadSat(s),
            IntFn::AddSat(s),
            IntFn::SubSat(s),
            IntFn::HAdd(s),
            IntFn::RHAdd(s),
            IntFn::ExtractBits(s),
        ]);
    }
    let mut checks = Vec::new();
    for bits in [8u8, 12, 16, 24, 32, 40, 64] {
        let vals = int_values(bits, &mut rng, 5);
        for op in &fns {
            if *op != IntFn::Bswap || bits % 16 == 0 {
                checks.push((bits, *op, vals.clone()));
            }
        }
    }
    for (bits, op, vals) in part_of(checks, part, of) {
        let f = int(bits);
        let arity = op.arity();
        let cases: Vec<Case> = match op {
            IntFn::ExtractBits(_) => {
                let mut c = Vec::new();
                for v in &vals {
                    for (off, cnt) in [
                        (0, 0),
                        (0, 1),
                        (1, 3),
                        (0, bits),
                        (bits / 2, bits / 2),
                        (bits - 1, 1),
                        (bits, 0),
                    ] {
                        c.push(vec![vec![*v], vec![u64::from(off)], vec![u64::from(cnt)]]);
                    }
                }
                c
            }
            IntFn::InsertBits => {
                let mut c = Vec::new();
                for (i, v) in vals.iter().enumerate() {
                    let w = vals[(i * 7 + 3) % vals.len()];
                    for (off, cnt) in [(0, 0), (0, 1), (2, 3), (0, bits), (bits / 2, bits / 2)] {
                        c.push(vec![
                            vec![*v],
                            vec![w],
                            vec![u64::from(off)],
                            vec![u64::from(cnt)],
                        ]);
                    }
                }
                c
            }
            _ => {
                let mut c = Vec::new();
                for (i, x) in vals.iter().enumerate() {
                    for (j, y) in vals.iter().enumerate() {
                        let z = vals[(i * 3 + j * 5) % vals.len()];
                        c.push(
                            vec![vec![*x], vec![*y], vec![z]]
                                .into_iter()
                                .take(arity)
                                .collect(),
                        );
                    }
                }
                c
            }
        };
        // `ExtractBits`/`InsertBits` take `i32` offsets and counts.
        let ins: Vec<Form> = match op {
            IntFn::ExtractBits(_) => vec![f, int(32), int(32)],
            IntFn::InsertBits => vec![f, f, int(32), int(32)],
            _ => vec![f; arity],
        };
        // `clz`/`ctz`/`popcount` count into the operand's type.
        check(
            &format!("{op:?} i{bits}"),
            &ins,
            f,
            &|fb, x| math::int_fn(fb, op, f, x),
            &cases,
            &|c| {
                let a: Vec<u64> = c.iter().map(|l| l[0]).collect();
                Some(vec![int_ref(op, bits, &a)?])
            },
            false,
        );
    }
}

mod integer_functions_match_their_definitions {
    parts!(super::integer_functions, 10;
        part_1_of_10 = 0, part_2_of_10 = 1, part_3_of_10 = 2, part_4_of_10 = 3,
        part_5_of_10 = 4, part_6_of_10 = 5, part_7_of_10 = 6, part_8_of_10 = 7,
        part_9_of_10 = 8, part_10_of_10 = 9);
}

// ---- Float functions ----

/// A signed saturating add or subtract narrower than its container takes no signed min, max or
/// clamp: it clamps its operands' keys (bit N - 1 flipped) unsigned, so no signed clamp to the
/// width's range meets the mask that zero-extends the result, which the RTX 3090 (NVIDIA
/// 615.71.09) drops after a sum of two sign-extended operands. Each answers its definition lane by
/// lane on vectors at every edge of the width.
#[test]
fn signed_narrow_saturation_clamps_without_signed_min_or_max() {
    let fns = [IntFn::AddSat(Sign::Signed), IntFn::SubSat(Sign::Signed)];
    let mut rng = Rng(0x0D0D_0D0D);
    for bits in [8u8, 16, 24, 40] {
        let vals = int_values(bits, &mut rng, 3);
        for lanes in [1u8, 3] {
            let f = if lanes == 1 {
                int(bits)
            } else {
                vec(int(bits), lanes)
            };
            for op in fns {
                let what = format!("{op:?} i{bits} x{lanes}");
                let arity = op.arity();
                let m = kernel_with(
                    &vec![f; arity],
                    f,
                    &|fb, x| math::int_fn(fb, op, f, x),
                    true,
                    false,
                )
                .unwrap_or_else(|e| panic!("{what}: build: {e:?}"));
                let signed_order = m
                    .blocks()
                    .iter()
                    .flat_map(|b| b.insts.iter())
                    .filter(|n| {
                        matches!(
                            n.inst,
                            Inst::Glsl {
                                op: GlslOp::SMin | GlslOp::SMax | GlslOp::SClamp,
                                ..
                            }
                        )
                    })
                    .count();
                assert_eq!(signed_order, 0, "{what}: a signed min, max or clamp");
                // Each lane its own operands: the grid rotated per lane.
                let usize_lanes = usize::from(lanes);
                let mut cases: Vec<Case> = Vec::new();
                for (i, x) in vals.iter().enumerate() {
                    for (j, y) in vals.iter().enumerate() {
                        let pick =
                            |k: usize, l: usize| vals[(i * 3 + j * 5 + k * 7 + l) % vals.len()];
                        let lane = |first: u64, k: usize| {
                            (0..usize_lanes)
                                .map(|l| if l == 0 { first } else { pick(k, l) })
                                .collect::<Vec<u64>>()
                        };
                        let ops = [lane(*x, 0), lane(*y, 1), lane(pick(2, 0), 2)];
                        cases.push(ops.into_iter().take(arity).collect());
                    }
                }
                check(
                    &what,
                    &vec![f; arity],
                    f,
                    &|fb, x| math::int_fn(fb, op, f, x),
                    &cases,
                    &|c| {
                        (0..usize_lanes)
                            .map(|l| {
                                let a: Vec<u64> = c.iter().map(|o| o[l]).collect();
                                int_ref(op, bits, &a)
                            })
                            .collect::<Option<Vec<u64>>>()
                    },
                    false,
                );
            }
        }
    }
}

/// Under [`MaskForm::Extract`], the form a driver with
/// `DriverWorkaround::NarrowMaskAfterSignedClamp` takes (the RTX 3090), a narrow integer's mask
/// in a 32-bit container is `OpBitFieldUExtract(x, 0, N)` and never `OpBitwiseAnd`, and the
/// source-arithmetic shape whose mask that driver drops answers its definition at every edge:
/// two values sign-extended from `N` bits, added or subtracted, clamped signed to exactly `N`
/// bits' range (`SClamp`, and `SMin` then `SMax`), truncated to `N` bits. At 8 and 16 bits, scalar
/// and three lanes. Under [`MaskForm::And`] the same body keeps the mask and emits no extract.
#[test]
fn a_narrow_mask_after_a_signed_clamp_is_an_extract_under_the_workaround() {
    let mut rng = Rng(0x0D0D_0D0E);
    for bits in [8u8, 16] {
        let vals = int_values(bits, &mut rng, 3);
        for lanes in [1u8, 3] {
            let f = vec(int(bits), lanes);
            let wide = vec(int(32), lanes);
            for (sub, chained, form) in [false, true]
                .into_iter()
                .flat_map(|sub| [false, true].map(|chained| (sub, chained)))
                .flat_map(|(sub, chained)| {
                    [MaskForm::And, MaskForm::Extract].map(|form| (sub, chained, form))
                })
            {
                let what = format!("i{bits} x{lanes} sub {sub} chained {chained} {form:?}");
                let body = |fb: &mut FunctionBuilder<'_>, xs: &[Value]| -> BResult<Value> {
                    fb.module().set_mask_form(form);
                    let a = ops::cast(fb, CastKind::SExt, f, wide, xs[0])?;
                    let b = ops::cast(fb, CastKind::SExt, f, wide, xs[1])?;
                    let op = if sub { BinOp::ISub } else { BinOp::IAdd };
                    let s = fb.binary(op, a, b)?;
                    let lo = ops::splat(fb, wide, ops::ones(32) & !(ops::ones(bits) >> 1))?;
                    let hi = ops::splat(fb, wide, ops::ones(bits) >> 1)?;
                    let c = if chained {
                        let m = fb.glsl(GlslOp::SMin, &[s, hi])?;
                        fb.glsl(GlslOp::SMax, &[m, lo])?
                    } else {
                        fb.glsl(GlslOp::SClamp, &[s, lo, hi])?
                    };
                    ops::cast(fb, CastKind::Trunc, wide, f, c)
                };
                let m = kernel_with(&[f, f], f, &body, true, false)
                    .unwrap_or_else(|e| panic!("{what}: build: {e:?}"));
                let insts = || m.blocks().iter().flat_map(|b| b.insts.iter());
                let extracts = insts()
                    .filter(|n| matches!(n.inst, Inst::BitFieldExtract { signed: false, .. }))
                    .count();
                let ands = insts()
                    .filter(|n| {
                        matches!(
                            n.inst,
                            Inst::Binary {
                                op: BinOp::BitwiseAnd,
                                ..
                            }
                        )
                    })
                    .count();
                match form {
                    MaskForm::Extract => {
                        assert!(extracts > 0, "{what}: no extract");
                        assert_eq!(ands, 0, "{what}: a mask");
                    }
                    MaskForm::And => {
                        assert!(ands > 0, "{what}: no mask");
                        assert_eq!(extracts, 0, "{what}: an extract");
                    }
                }
                let usize_lanes = usize::from(lanes);
                let mut cases: Vec<Case> = Vec::new();
                for (i, x) in vals.iter().enumerate() {
                    for (j, y) in vals.iter().enumerate() {
                        let lane = |first: u64, k: usize| {
                            (0..usize_lanes)
                                .map(|l| {
                                    if l == 0 {
                                        first
                                    } else {
                                        vals[(i * 3 + j * 5 + k * 7 + l) % vals.len()]
                                    }
                                })
                                .collect::<Vec<u64>>()
                        };
                        cases.push(vec![lane(*x, 0), lane(*y, 1)]);
                    }
                }
                let limit = (1i64 << (bits - 1)) - 1;
                check(
                    &what,
                    &[f, f],
                    f,
                    &body,
                    &cases,
                    &|c| {
                        Some(
                            (0..usize_lanes)
                                .map(|l| {
                                    let (a, b) = (sext(c[0][l], bits), sext(c[1][l], bits));
                                    let r = if sub { a - b } else { a + b };
                                    (r.clamp(-limit - 1, limit) as u64) & mask(bits)
                                })
                                .collect(),
                        )
                    },
                    false,
                );
            }
        }
    }
}

/// Metal's `round`: halves away from zero, and on `float` and `half` (`positive_zero`) a zero
/// result is `+0` (`round(-0.3)` and `round(-0)` are `+0`).
fn metal_round(positive_zero: bool, x: f64) -> f64 {
    let r = x.round();
    if positive_zero && r == 0.0 {
        0.0
    } else {
        r
    }
}

/// Metal's `median3` of operand bits `b` (A1 case aed3faaa): `min(max(c, min(a, b)), max(a, b))`,
/// where `min(p, q)` is `p` for a NaN `q`, else `p < q ? p : q` (and `max` with `>`), comparing the
/// operands' values `val` (an `f32` denormal as zero) but returning their bits; `fast` answers `b`
/// for two NaN first operands.
fn median3_ref(b: [u64; 3], val: &dyn Fn(u64) -> f64, fast: bool) -> u64 {
    let pick = |p: u64, q: u64, less: bool| {
        let (vp, vq) = (val(p), val(q));
        if vq.is_nan() || (less && vp < vq) || (!less && vp > vq) {
            p
        } else {
            q
        }
    };
    let [a, b, c] = b;
    if fast && val(a).is_nan() && val(b).is_nan() {
        return b;
    }
    let lo = pick(a, b, true);
    let hi = pick(a, b, false);
    pick(pick(c, lo, false), hi, true)
}

/// Metal's `fmin`, `fmax`, `min3`, `max3`, or `clamp` (`op`) of operand bits `b`: `fmin(p, q)` is
/// `q` for a NaN `p`, else `p` for a NaN `q` or `p < q`, else `q` (and `fmax` with `>`), comparing
/// the operands' values `val` (an `f32` denormal as zero) but returning their bits; `min3` and
/// `max3` nest it, and `clamp(x, lo, hi)` is `fmin(fmax(x, lo), hi)`.
fn metal_min_max_ref(op: FloatFn, b: &[u64], val: &dyn Fn(u64) -> f64) -> u64 {
    let pick = |p: u64, q: u64, less: bool| {
        let (vp, vq) = (val(p), val(q));
        if vp.is_nan() {
            q
        } else if vq.is_nan() || (less && vp < vq) || (!less && vp > vq) {
            p
        } else {
            q
        }
    };
    match op {
        FloatFn::Min => pick(b[0], b[1], true),
        FloatFn::Max => pick(b[0], b[1], false),
        FloatFn::Min3 => pick(pick(b[0], b[1], true), b[2], true),
        FloatFn::Max3 => pick(pick(b[0], b[1], false), b[2], false),
        FloatFn::Clamp => pick(pick(b[0], b[1], false), b[2], true),
        _ => panic!("{op:?} is not a minimum or maximum"),
    }
}

/// An `f32` operand's value as Metal's `float` comparisons read it: a denormal as zero.
fn f32_compared(b: u64) -> f64 {
    if b & 0x7F80_0000 == 0 {
        0.0
    } else {
        f64::from(f32::from_bits(b as u32))
    }
}

fn float_ref(op: FloatFn, bits: u8, a: &[u64]) -> Option<u64> {
    let x = |i: usize| fval(bits, a[i]);
    // An `f32` zero or denormal, which Metal's `float` functions read as zero.
    let flushed = |i: usize| bits == 32 && a[i] & 0x7F80_0000 == 0;
    // Compute in the operand width.
    let r = |v: f64| Some(fbits(bits, v));
    let min = |p: f64, q: f64| p.min(q);
    let max = |p: f64, q: f64| p.max(q);
    match op {
        FloatFn::Abs => r(x(0).abs()),
        FloatFn::Floor => r(x(0).floor()),
        FloatFn::Ceil => r(x(0).ceil()),
        FloatFn::Trunc => r(x(0).trunc()),
        FloatFn::Round => r(metal_round(bits == 32, x(0))),
        FloatFn::Rint => r(x(0).round_ties_even()),
        FloatFn::Fract | FloatFn::FastFract { .. } => {
            if x(0).is_nan() {
                return Some(fbits(bits, f64::NAN));
            }
            // `fast::fract`'s `x - floor(x)` is NaN at an infinity under every fast-math flag.
            if x(0).is_infinite() && op == (FloatFn::FastFract { infinite_nan: true }) {
                return Some(fbits(bits, f64::NAN));
            }
            if x(0).is_infinite() || flushed(0) {
                return r(0.0);
            }
            let below = if bits == 32 {
                f64::from(f32::from_bits(0x3F7F_FFFF))
            } else {
                f64::from_bits(0x3FEF_FFFF_FFFF_FFFF)
            };
            let d = if bits == 32 {
                f64::from(f32v(a[0]) - f32v(a[0]).floor())
            } else {
                x(0) - x(0).floor()
            };
            r(if op == FloatFn::Fract {
                d.min(below)
            } else {
                d
            })
        }
        FloatFn::Sign => {
            let v = x(0);
            r(if flushed(0) || v.is_nan() || v == 0.0 {
                0.0
            } else if v > 0.0 {
                1.0
            } else {
                -1.0
            })
        }
        FloatFn::Saturate => {
            let v = x(0);
            r(if flushed(0) || v.is_nan() || v <= 0.0 {
                0.0
            } else {
                v.min(1.0)
            })
        }
        FloatFn::Min | FloatFn::Max | FloatFn::Clamp | FloatFn::Min3 | FloatFn::Max3
            if bits == 32 =>
        {
            Some(metal_min_max_ref(op, a, &f32_compared))
        }
        FloatFn::Min | FloatFn::MinNum => r(min(x(0), x(1))),
        FloatFn::Max | FloatFn::MaxNum => r(max(x(0), x(1))),
        FloatFn::Clamp => r(min(max(x(0), x(1)), x(2))),
        FloatFn::Min3 => r(min(min(x(0), x(1)), x(2))),
        FloatFn::Max3 => r(max(max(x(0), x(1)), x(2))),
        FloatFn::Median3 | FloatFn::FastMedian3 => Some(median3_ref(
            [a[0], a[1], a[2]],
            &|b| {
                if bits == 32 && b & 0x7F80_0000 == 0 {
                    0.0
                } else {
                    fval(bits, b)
                }
            },
            op == FloatFn::FastMedian3,
        )),
        FloatFn::Mix | FloatFn::MulAdd => {
            // Rounded per operation in the operand width.
            if bits == 32 {
                let (p, q, t) = (f32v(a[0]), f32v(a[1]), f32v(a[2]));
                let v = if op == FloatFn::Mix {
                    p * (1.0 - t) + q * t
                } else {
                    p * q + t
                };
                Some(u64::from(v.to_bits()))
            } else {
                let (p, q, t) = (x(0), x(1), x(2));
                r(if op == FloatFn::Mix {
                    p * (1.0 - t) + q * t
                } else {
                    p * q + t
                })
            }
        }
        FloatFn::Sqrt => {
            if bits == 32 {
                Some(u64::from(f32v(a[0]).sqrt().to_bits()))
            } else {
                r(x(0).sqrt())
            }
        }
        FloatFn::Fma => {
            if bits == 32 {
                Some(u64::from(
                    f32v(a[0]).mul_add(f32v(a[1]), f32v(a[2])).to_bits(),
                ))
            } else {
                r(x(0).mul_add(x(1), x(2)))
            }
        }
        _ => None,
    }
}

/// `llvm.copysign` is the first operand's bits with the second's sign bit, exactly: NaN payloads,
/// infinities, zeros, and denormals keep every other bit, on `float` and on the narrow formats held
/// widened (whose container's sign bit is the format's).
#[test]
fn copysign_moves_exactly_the_sign_bit() {
    let mut f32s = f32_values();
    f32s.extend([
        0x7FC0_0001,
        0xFFC0_0000,
        0x7F80_0001,
        0x8000_0001,
        0x0000_0001,
    ]);
    let mut halves: Vec<u64> = vec![0, 0x8000, 0x0001, 0x8001, 0x3C00, 0xBC00, 0x7BFF];
    halves.extend([0x7C00, 0xFC00, 0x7E00, 0x7C01, 0xFE01, 0x03FF]);
    let bfloats: Vec<u64> = vec![0, 0x8000, 0x0001, 0x3F80, 0xBF80, 0x7F80, 0xFF80, 0x7FC1];
    for (what, f, vals, sign) in [
        ("f32", float(32), f32s, 0x8000_0000u64),
        ("half", float(16), halves, 0x8000),
        ("bfloat", bfloat(), bfloats, 0x8000),
    ] {
        check_accept(
            &format!("copysign {what}"),
            &[f, f],
            f,
            &|fb, x| {
                math::float_fn(
                    fb,
                    FloatFn::CopySign,
                    f,
                    Contraction::Forbidden,
                    NanTest::IsNan,
                    x,
                    &Lib::instructions(),
                )
            },
            &pairs(&vals, &vals),
            &|c| Some(vec![(arg(c, 0) & !sign) | (arg(c, 1) & sign)]),
            &|g, w| g == w,
        );
    }
}

#[test]
fn exact_float_functions_match_metal() {
    let fns = [
        FloatFn::Abs,
        FloatFn::Floor,
        FloatFn::Ceil,
        FloatFn::Trunc,
        FloatFn::Round,
        FloatFn::Rint,
        FloatFn::Fract,
        FloatFn::FastFract {
            infinite_nan: false,
        },
        FloatFn::FastFract { infinite_nan: true },
        FloatFn::Sign,
        FloatFn::Saturate,
        FloatFn::Min,
        FloatFn::Max,
        FloatFn::MinNum,
        FloatFn::MaxNum,
        FloatFn::Clamp,
        FloatFn::Min3,
        FloatFn::Max3,
        FloatFn::Median3,
        FloatFn::FastMedian3,
        FloatFn::Mix,
        FloatFn::MulAdd,
        FloatFn::Sqrt,
        FloatFn::Fma,
    ];
    for (bits, vals) in [(32u8, f32_values()), (64, f64_values())] {
        let f = float(bits);
        let few: Vec<u64> = vals.iter().copied().step_by(4).collect();
        for op in fns {
            // A `float` `mix` is its own strategy, Metal's fused form (`tests::fused`).
            if op == FloatFn::Mix && bits == 32 {
                continue;
            }
            let arity = op.arity();
            let cases: Vec<Case> = if arity == 1 {
                vals.iter().map(|v| vec![vec![*v]]).collect()
            } else {
                let mut c = Vec::new();
                for (i, p) in few.iter().enumerate() {
                    for (j, q) in few.iter().enumerate() {
                        let t = few[(i + 2 * j) % few.len()];
                        c.push(
                            vec![vec![*p], vec![*q], vec![t]]
                                .into_iter()
                                .take(arity)
                                .collect(),
                        );
                    }
                }
                c
            };
            // The device's `NMin` and `NMax` may give either zero; Metal's (`float`) gives the
            // second operand.
            let loose = match op {
                FloatFn::MinNum | FloatFn::MaxNum => true,
                FloatFn::Min | FloatFn::Max | FloatFn::Clamp | FloatFn::Min3 | FloatFn::Max3 => {
                    bits == 64
                }
                _ => false,
            };
            check(
                &format!("{op:?} f{bits}"),
                &vec![f; arity],
                f,
                &|fb, x| {
                    math::float_fn(
                        fb,
                        op,
                        f,
                        Contraction::Forbidden,
                        NanTest::IsNan,
                        x,
                        &Lib::instructions(),
                    )
                },
                &cases,
                &|c| {
                    let a: Vec<u64> = c.iter().map(|l| l[0]).collect();
                    Some(vec![float_ref(op, bits, &a)?])
                },
                loose,
            );
        }
    }
}

// ---- Metal's NaN ----

/// The NaN operands of the NaN-model probes: quiet and signaling, of either sign, with and
/// without a payload, beside the canonical NaN, infinities, zeros, and ordinary values.
fn nan_operands() -> Vec<u64> {
    vec![
        0x7FC0_0000,
        0xFFC0_0000,
        0x7FC0_1234,
        0xFFC0_1234,
        0x7F80_0001,
        0xFF80_0001,
        0x7FFF_FFFF,
        0xFFFF_FFFF,
        0x7FA0_0000,
        0x7F80_0000,
        0xFF80_0000,
        0,
        0x8000_0000,
        0x3F80_0000,
        0xBF80_0000,
        0x0000_0001,
    ]
}

/// [`nan::select`] answers Metal's NaN bit for bit, by either test, on scalars and vectors: the
/// canonical NaN for every NaN result (`Canonical`, `CanonicalExact`); with a NaN operand's sign or `+` for a domain error
/// (`OperandSign`); `+` for a NaN operand and the operand's sign for a domain error
/// (`DomainSign`); the NaN operand itself, else the canonical NaN (`Operand`); and every other
/// result untouched. The kernel's first input is the operand, its second the result.
#[test]
fn nan_select_answers_metals_nan_exactly() {
    use super::nan::{self, NanAnswer};
    let f32_nan = |b: u64| f32::from_bits(b as u32).is_nan();
    let vals = nan_operands();
    let cases = pairs(&vals, &vals);
    for answer in [
        NanAnswer::Canonical,
        NanAnswer::CanonicalExact,
        NanAnswer::OperandSign,
        NanAnswer::DomainSign,
        NanAnswer::Operand,
    ] {
        for test in [NanTest::IsNan, NanTest::Bits] {
            for lanes in [1, 3] {
                let f = vec(float(32), lanes);
                let lane_cases: Vec<Case> = cases
                    .iter()
                    .map(|c| c.iter().map(|o| vec![o[0]; usize::from(lanes)]).collect())
                    .collect();
                check_accept(
                    &format!("{answer:?} {test:?} x{lanes}"),
                    &[f, f],
                    f,
                    &|fb, x| nan::select(fb, f, answer, test, x[1], Some(x[0])),
                    &lane_cases,
                    &|c| {
                        let (a, r) = (arg(c, 0), arg(c, 1));
                        let sign = a & 0x8000_0000;
                        let want = match answer {
                            NanAnswer::Operand if f32_nan(a) => a,
                            _ if !f32_nan(r) => r,
                            NanAnswer::Canonical
                            | NanAnswer::CanonicalExact
                            | NanAnswer::Operand => nan::CANONICAL,
                            NanAnswer::OperandSign if f32_nan(a) => sign | nan::CANONICAL,
                            NanAnswer::OperandSign => nan::CANONICAL,
                            NanAnswer::DomainSign if f32_nan(a) => nan::CANONICAL,
                            NanAnswer::DomainSign => sign | nan::CANONICAL,
                        };
                        Some(vec![want; usize::from(lanes)])
                    },
                    &|g, w| g == w,
                );
            }
        }
    }
}

/// Class B: on a `half` result, held widened, [`nan::select`] of the canonical answer gives Metal's
/// `half` NaN `0x7E00` for every NaN lane, quiet or signaling, of either sign and any payload, by
/// either test, and keeps every other lane, infinities and zeros included.
#[test]
fn nan_select_answers_metals_half_nan() {
    use super::nan::{self, NanAnswer};
    let halves: [u64; 16] = [
        0x7E00, 0xFE00, 0x7C01, 0xFC01, 0x7DFF, 0xFFFF, 0x7FFF, 0x7E55, 0x7C00, 0xFC00, 0, 0x8000,
        0x3C00, 0xBC00, 0x0001, 0x7BFF,
    ];
    for test in [NanTest::IsNan, NanTest::Bits] {
        for lanes in [1u8, 3] {
            let f = vec(float(16), lanes);
            let cases: Vec<Case> = halves
                .iter()
                .map(|&h| vec![vec![h; usize::from(lanes)]])
                .collect();
            check_accept(
                &format!("half {test:?} x{lanes}"),
                &[f],
                f,
                &|fb, x| nan::select(fb, f, NanAnswer::Canonical, test, x[0], None),
                &cases,
                &|c| {
                    let h = arg(c, 0);
                    let want = if is_nan(float(16), h) { 0x7E00 } else { h };
                    Some(vec![want; usize::from(lanes)])
                },
                &|g, w| g == w,
            );
        }
    }
}

/// Class F: Metal's `fmin`, `fmax`, `min3`, `max3`, and `clamp` on `float` and `half`
/// ([`math::metal_min_max`]) answer an operand bit for bit, by either NaN test, on scalars and
/// vectors: a NaN loses, two NaNs and a tie of zeros answer the second operand, and a `float`
/// denormal compares as zero but is returned itself (`fmin(2^-149, +0)` is `+0`, `fmin(+0,
/// 2^-149)` is `2^-149`), as `nan_minmax`, `nan_minmax_fast`, their zeros-and-denormals cases,
/// `nan_half`, and `nan_half_fast` read on every lane.
#[test]
fn min_max_answer_metals_operand_bit_for_bit() {
    let mut floats = nan_operands();
    floats.extend([
        0x8000_0001,
        0x007F_FFFF,
        0x0080_0000,
        0x8080_0000,
        0x4000_0000,
    ]);
    let halves: Vec<u64> = vec![
        0x7E00, 0xFE00, 0x7C01, 0xFC01, 0x7E55, 0x7C00, 0xFC00, 0, 0x8000, 0x3C00, 0xBC00, 0x0001,
        0x8001, 0x03FF, 0x0400,
    ];
    let half_compared = |b: u64| f16_to_f64(b as u16);
    let formats: [(u8, Vec<u64>, &dyn Fn(u64) -> f64); 2] =
        [(32, floats, &f32_compared), (16, halves, &half_compared)];
    for (bits, vals, val) in formats {
        let few: Vec<u64> = vals.iter().copied().step_by(2).collect();
        let triples: Vec<Case> = few
            .iter()
            .flat_map(|&a| {
                let few = &few;
                few.iter()
                    .flat_map(move |&b| few.iter().map(move |&c| vec![vec![a], vec![b], vec![c]]))
            })
            .collect();
        for op in [
            FloatFn::Min,
            FloatFn::Max,
            FloatFn::Min3,
            FloatFn::Max3,
            FloatFn::Clamp,
        ] {
            let arity = op.arity();
            let cases = if arity == 2 {
                pairs(&vals, &vals)
            } else {
                triples.clone()
            };
            for test in [NanTest::IsNan, NanTest::Bits] {
                for lanes in [1u8, 3] {
                    let f = vec(float(bits), lanes);
                    let n = usize::from(lanes);
                    let lane_cases: Vec<Case> = cases
                        .iter()
                        .map(|c| c.iter().map(|o| vec![o[0]; n]).collect())
                        .collect();
                    check_accept(
                        &format!("{op:?} f{bits} {test:?} x{lanes}"),
                        &vec![f; arity],
                        f,
                        &|fb, x| {
                            math::float_fn(
                                fb,
                                op,
                                f,
                                Contraction::Forbidden,
                                test,
                                x,
                                &Lib::instructions(),
                            )
                        },
                        &lane_cases,
                        &|c| {
                            let b: Vec<u64> = c.iter().map(|l| l[0]).collect();
                            Some(vec![metal_min_max_ref(op, &b, val); n])
                        },
                        &|g, w| g == w,
                    );
                }
            }
        }
    }
}

/// Class H: Metal's `half` `fmod` answers `+0` for a zero remainder of a nonzero dividend and the
/// dividend itself for a zero one (`fmod(-2, 0.5)` and `fmod(-65504, 65504)` are `+0`, `fmod(-0,
/// 1)` and `fmod(-0, inf)` are `-0`), as `nan_fmod_zero` and `nan_fmod_zero_fast` read on every
/// lane; every other remainder is C's, with the dividend's sign, which the interpreter's `OpFRem`
/// answers. Scalars and vectors, bit for bit but for a NaN's payload (the canonical NaN is selected
/// beside the function), and within the word estimate. A `float` remainder keeps the dividend's
/// sign (`fmod(-2, 0.5)` is `-0`), as Metal's does.
#[test]
fn half_fmod_answers_metals_zero_remainder() {
    let halves: Vec<u64> = vec![
        0, 0x8000, 0x3C00, 0xBC00, 0x4000, 0xC000, 0x4200, 0xC200, 0x3800, 0xB800, 0x3E00, 0xBE00,
        0x3A00, 0xBA00, 0x7BFF, 0xFBFF, 0x0001, 0x8001, 0x0002, 0x8002, 0x03FF, 0x7C00, 0xFC00,
        0x7E00,
    ];
    let reference = |c: &Case| {
        let (a, b) = (c[0][0], c[1][0]);
        let (x, y) = (f16_to_f64(a as u16), f16_to_f64(b as u16));
        let r = x % y;
        let bits = if r == 0.0 {
            if x == 0.0 {
                a
            } else {
                0
            }
        } else {
            u64::from(f64_to_f16(r))
        };
        Some(vec![bits; c[0].len()])
    };
    for lanes in [1u8, 3] {
        let f = vec(float(16), lanes);
        let n = usize::from(lanes);
        let cases: Vec<Case> = pairs(&halves, &halves)
            .iter()
            .map(|c| c.iter().map(|o| vec![o[0]; n]).collect())
            .collect();
        check(
            &format!("fmod f16 x{lanes}"),
            &[f, f],
            f,
            &|fb, x| {
                math::float_fn(
                    fb,
                    FloatFn::Fmod,
                    f,
                    Contraction::Forbidden,
                    NanTest::IsNan,
                    x,
                    &Lib::instructions(),
                )
            },
            &cases,
            &reference,
            false,
        );
    }
    let f = float(32);
    let cases: Vec<Case> = [(-2.0f32, 0.5f32), (-0.0, 1.0), (2.0, -0.5), (-3.0, 2.0)]
        .iter()
        .map(|(x, y)| vec![vec![u64::from(x.to_bits())], vec![u64::from(y.to_bits())]])
        .collect();
    check(
        "fmod f32",
        &[f, f],
        f,
        &|fb, x| {
            math::float_fn(
                fb,
                FloatFn::Fmod,
                f,
                Contraction::Forbidden,
                NanTest::IsNan,
                x,
                &Lib::instructions(),
            )
        },
        &cases,
        &|c| {
            let (x, y) = (
                f32::from_bits(c[0][0] as u32),
                f32::from_bits(c[1][0] as u32),
            );
            Some(vec![u64::from((x % y).to_bits())])
        },
        false,
    );
    let module_words = |f: Form, body: Body<'_>| {
        let m = kernel(&[f, f], f, body).expect("kernel");
        crate::spirv::serialize::serialize(m)
            .expect("words")
            .words
            .len()
    };
    for f in [float(16), vec(float(16), 4)] {
        let bare = module_words(f, &|_, x| Ok(x[0]));
        let used = module_words(f, &|fb, x| {
            math::float_fn(
                fb,
                FloatFn::Fmod,
                f,
                Contraction::Allowed,
                NanTest::IsNan,
                x,
                &Lib::instructions(),
            )
        });
        let (emitted, bound) = (used - bare, FloatFn::Fmod.words(f) as usize);
        assert!(emitted <= bound, "fmod {f:?}: {emitted} words over {bound}");
    }
}

/// Classes B, C, D, E, and K's measured states: Metal's `half` `+ - * /`, `air.fma.f16`, and
/// `llvm.fmuladd` answer the canonical `half` NaN with and without fast-math flags (`nan_narrow`,
/// `nan_narrow_fast`, `nan_half`, `nan_half_fast`), and so does its narrowing of a `float` to
/// `half` (`fptrunc`, and `air.convert` in both math modes: `nan_convert`, `nan_convert_fast`); its
/// widening of a `half` to `float` (`fpext`, and `air.convert` in both modes), and its `half`
/// `fneg` and `fabs` (`nan_half`, `nan_half_fast`), answer the canonical NaN, computed exactly on
/// the device; neither cast carries flags in either math mode. Its `half` functions answer the
/// canonical `half` NaN, but `asin` (the operand's sign at a domain error) and `round` (the NaN
/// operand), and those that answer a number or an operand. `float` `+ - * /` answer the canonical
/// NaN in both states, and `frem` in neither; a `float` `fneg` is class E's. Only the canonical NaN
/// of rounding arithmetic is left to a device whose arithmetic answers it.
#[test]
fn half_nan_answers_follow_the_measured_states() {
    use super::nan::{
        binary_answer, cast_answer, convert_answer, fneg_answer, function_answer, NanAnswer as A,
    };
    use crate::air::ir::CastOp;
    let (h, f) = (FloatFmt::Half, FloatFmt::F32);
    for op in [AirBin::FAdd, AirBin::FSub, AirBin::FMul, AirBin::FDiv] {
        assert_eq!(binary_answer(op, h), Some(A::Canonical), "{op:?}");
    }
    assert_eq!(binary_answer(AirBin::FRem, h), None);
    for op in [AirBin::FAdd, AirBin::FSub, AirBin::FMul, AirBin::FDiv] {
        assert_eq!(binary_answer(op, f), Some(A::Canonical), "{op:?}");
    }
    assert_eq!(binary_answer(AirBin::FRem, f), None);
    for fmt in [FloatFmt::BFloat, FloatFmt::F64] {
        assert_eq!(binary_answer(AirBin::FAdd, fmt), None, "{fmt:?}");
    }
    for flagged in [false, true] {
        assert_eq!(
            function_answer(FloatFn::Fma, false, flagged, h),
            Some(A::Canonical)
        );
        assert_eq!(
            function_answer(FloatFn::Abs, false, flagged, h),
            Some(A::CanonicalExact)
        );
        assert_eq!(function_answer(FloatFn::Abs, false, flagged, f), None);
        for op in [
            FloatFn::MulAdd,
            FloatFn::Mix,
            FloatFn::Exp,
            FloatFn::Ldexp,
            FloatFn::Atan,
            FloatFn::Fmod,
            FloatFn::Sqrt,
            FloatFn::Dot,
            FloatFn::SinPi,
            FloatFn::TanPi,
        ] {
            assert_eq!(
                function_answer(op, false, flagged, h),
                Some(A::Canonical),
                "{op:?}"
            );
        }
        assert_eq!(
            function_answer(FloatFn::Asin, false, flagged, h),
            Some(A::DomainSign)
        );
        assert_eq!(
            function_answer(FloatFn::Round, false, flagged, h),
            Some(A::Operand)
        );
        for op in [
            FloatFn::Fract,
            FloatFn::FlaggedFract,
            FloatFn::Sign,
            FloatFn::Min,
            FloatFn::Median3,
        ] {
            assert_eq!(function_answer(op, false, flagged, h), None, "{op:?}");
        }
        // Metal has no `fast_` `half` function.
        assert_eq!(function_answer(FloatFn::Exp, true, flagged, h), None);
    }
    assert_eq!(fneg_answer(h), Some(A::CanonicalExact));
    for fmt in [f, FloatFmt::BFloat, FloatFmt::F64] {
        assert_eq!(fneg_answer(fmt), None, "{fmt:?}");
    }
    assert_eq!(cast_answer(CastOp::FPTrunc, f, h), Some(A::Canonical));
    assert_eq!(cast_answer(CastOp::FPExt, h, f), Some(A::CanonicalExact));
    assert_eq!(convert_answer(h, f), Some(A::CanonicalExact));
    assert_eq!(convert_answer(f, h), Some(A::Canonical));
    for (from, to) in [
        (FloatFmt::BFloat, f),
        (f, FloatFmt::BFloat),
        (f, FloatFmt::F64),
        (FloatFmt::F64, h),
        (h, h),
    ] {
        assert_eq!(convert_answer(from, to), None, "{from:?} {to:?}");
    }
    for (op, from, to) in [
        (CastOp::FPExt, FloatFmt::BFloat, f),
        (CastOp::FPExt, f, FloatFmt::F64),
        (CastOp::FPTrunc, FloatFmt::F64, f),
        (CastOp::FPTrunc, FloatFmt::F64, h),
        (CastOp::FPTrunc, f, FloatFmt::BFloat),
    ] {
        assert_eq!(cast_answer(op, from, to), None, "{op:?} {from:?} {to:?}");
    }
    use crate::device::profile::NanEncoding;
    assert!(A::Canonical.left_to(NanEncoding::MetalCanonical));
    for a in [A::CanonicalExact, A::Operand, A::OperandSign, A::DomainSign] {
        assert!(!a.left_to(NanEncoding::MetalCanonical), "{a:?}");
    }
    for a in [A::Canonical, A::CanonicalExact] {
        assert!(!a.left_to(NanEncoding::Device), "{a:?}");
    }
    // Native device maths leaves every answer to the device.
    for a in [
        A::Canonical,
        A::CanonicalExact,
        A::Operand,
        A::OperandSign,
        A::DomainSign,
    ] {
        assert!(a.left_to(NanEncoding::Native), "{a:?}");
    }
}

/// The float facts of a profile whose `float` fused multiply-add is `OpFmaKHR`, checked by the
/// integer routine, whose `half` one is the routine, and whose `float` functions are the device's.
const KHR_FACTS: super::FloatFacts = super::FloatFacts {
    routes: super::fma::Routes {
        float: super::fma::Route::Khr,
        half: super::fma::Route::HalfOdd,
        tiny: Some(super::fma::Route::Soft),
    },
    reciprocal: super::divide::Reciprocal::Rounded,
    metal_functions: false,
    native: false,
    nan_native: false,
    native_min_max: false,
    native_order: false,
};

/// Class E: Metal's `float` `fneg` and `fabs` flip or clear the sign bit alone, a NaN's payload and
/// a denormal kept, with and without fast-math flags (`nan_signbits`, `nan_signbits_fast`), so they
/// move on the bits on `float` only: such a `fabs` answers Metal's NaN itself and selects none,
/// records no device note, and leaves the operand's bits observed, while a `half` `fabs` still
/// selects its canonical NaN (class D). The bit operations are exact at every NaN, zero, and
/// denormal.
#[test]
fn float_sign_operations_move_the_bits() {
    use super::math::{sign_on_bits, SignBit};
    use super::nan::NanAnswer as A;
    use crate::air::ir::FastMath;
    let (h, f) = (FloatFmt::Half, FloatFmt::F32);
    assert!(sign_on_bits(Some(f)));
    for fmt in [h, FloatFmt::BFloat, FloatFmt::F64] {
        assert!(!sign_on_bits(Some(fmt)), "{fmt:?}");
    }
    assert!(!sign_on_bits(None));
    assert!(FloatFn::Abs.metal_nans(f));
    assert!(!FloatFn::Abs.metal_nans(h));
    assert!(FloatFn::Abs.metal_denormals());
    let scalar = |fmt| Form {
        held: Held::Float(fmt),
        lanes: 1,
    };
    let every = FastMath::ALL
        .iter()
        .fold(FastMath::NONE, |a, (b, _)| a.union(*b));
    // `float` `fabs` is `air.fabs` or `air.fast_fabs`, either with or without flags; a `half` one
    // is `air.fabs.f16` (Metal has no `fast_` `half` function) with or without them.
    for fmf in [FastMath::NONE, every] {
        for fast in [false, true] {
            let call = super::float_call(FloatFn::Abs, fast, fmf, scalar(f), KHR_FACTS)
                .expect("float fabs");
            assert_eq!(call.nan, None, "fast {fast}, {fmf:?}");
            assert_eq!(
                call.notes,
                super::OpNotes::default(),
                "fast {fast}, {fmf:?}"
            );
        }
        let call =
            super::float_call(FloatFn::Abs, false, fmf, scalar(h), KHR_FACTS).expect("half fabs");
        assert_eq!(call.nan, Some(A::CanonicalExact), "{fmf:?}");
    }
    let mut vals = nan_operands();
    vals.extend([0x8000_0001, 0x807F_FFFF, 0x007F_FFFF, 0xFF7F_FFFF]);
    for lanes in [1u8, 3] {
        let form = Form {
            held: Held::Float(f),
            lanes,
        };
        let cases: Vec<Case> = (0..vals.len())
            .map(|i| {
                let v = (0..lanes).map(|l| vals[(i + 5 * usize::from(l)) % vals.len()]);
                vec![v.collect()]
            })
            .collect();
        for (op, apply) in [
            (SignBit::Flip, (|b: u64| b ^ 0x8000_0000) as fn(u64) -> u64),
            (SignBit::Clear, |b: u64| b & 0x7FFF_FFFF),
        ] {
            check_accept(
                &format!("{op:?} f32x{lanes}"),
                &[form],
                form,
                &|fb, x| math::sign_bits(fb, form, x[0], op),
                &cases,
                &|c| Some(c[0].iter().map(|b| apply(*b)).collect()),
                &|g, w| g == w,
            );
        }
    }
}

/// Metal's NaN answers are measured in two states, the precise functions without fast-math flags
/// and the `fast_` ones with every flag, and a third for the precise trigonometric functions with
/// every flag, which answer as without them: precise `asin`, `atan`, `tan`, `sin` and `cos` with
/// flags keep their precise answers. A call in another state has an answer only where both states
/// agree: `fast_exp`, `fast_round`, and `fast_fract` without flags (the canonical NaN, the operand,
/// and the canonical NaN of A1 case aed3faaa), `dot` with them, but neither `fast_tan`,
/// `fast_asin`, `fast_atan`, nor `fast_asinh` without flags, nor precise `pow` with them
/// (unmeasured: its `fast_` variant answers otherwise). `half` answers the precise functions in
/// both states (class K), and no `fast_` one; `bfloat` and `double` have none. The function a call
/// selects (`refine_float`, with the math-library branch's `fast_` variants) answers as the call
/// was measured.
#[test]
fn nan_answers_follow_the_measured_flag_states() {
    use super::nan::{function_answer, NanAnswer as A};
    use crate::air::ir::FastMath;
    let f32_ = FloatFmt::F32;
    let fract = |infinite_nan| FloatFn::FastFract { infinite_nan };
    for (op, fast, flagged, want) in [
        (FloatFn::Exp, false, false, Some(A::Canonical)),
        (FloatFn::FastExp, true, true, Some(A::Canonical)),
        (FloatFn::FastExp, true, false, Some(A::Canonical)),
        (FloatFn::Exp, false, true, Some(A::Canonical)),
        (FloatFn::Round, true, false, Some(A::Operand)),
        (FloatFn::Round, false, true, Some(A::Operand)),
        (fract(false), true, false, Some(A::Canonical)),
        (fract(true), true, true, Some(A::Canonical)),
        (FloatFn::Dot, false, true, Some(A::Canonical)),
        (FloatFn::Tan, true, true, Some(A::Operand)),
        (FloatFn::Tan, false, false, Some(A::Canonical)),
        (FloatFn::Tan, true, false, None),
        (FloatFn::Tan, false, true, Some(A::Canonical)),
        (FloatFn::Asin, true, true, Some(A::DomainSign)),
        (FloatFn::Asin, false, false, Some(A::OperandSign)),
        (FloatFn::Asin, true, false, None),
        (FloatFn::Asin, false, true, Some(A::OperandSign)),
        (FloatFn::Atan, true, true, Some(A::Canonical)),
        (FloatFn::Atan, true, false, None),
        (FloatFn::Atan, false, true, Some(A::OperandSign)),
        (FloatFn::FastAsinh, true, true, Some(A::Canonical)),
        (FloatFn::Asinh, true, false, None),
        (FloatFn::Asinh, false, true, Some(A::OperandSign)),
        (FloatFn::Atanh, false, true, Some(A::OperandSign)),
        (FloatFn::Sin, false, true, Some(A::Canonical)),
        (FloatFn::Cos, false, true, Some(A::Canonical)),
        (FloatFn::Atan2, false, true, Some(A::Canonical)),
        (FloatFn::Pow, false, true, None),
        (FloatFn::FastSin, true, true, None),
        (FloatFn::Sign, false, false, None),
        (FloatFn::SinPi, false, false, Some(A::Canonical)),
        (FloatFn::Rsqrt, true, true, Some(A::Canonical)),
        (FloatFn::Rsqrt, true, false, Some(A::Canonical)),
        (FloatFn::Acos, true, true, Some(A::Canonical)),
        (FloatFn::Acos, true, false, Some(A::Canonical)),
    ] {
        assert_eq!(
            function_answer(op, fast, flagged, f32_),
            want,
            "{op:?} fast {fast} flagged {flagged}"
        );
        // The function a call selects (`refine_float`) answers as the call was measured, so a
        // `fast_` refinement of its own (the `FastTan`, `FastAsin`, ...) keeps the answer.
        let all = FastMath::ALL
            .iter()
            .fold(FastMath::NONE, |every, (flag, _)| every.union(*flag));
        let fmf = if flagged { all } else { FastMath::NONE };
        let selected = super::refine_float(op, fast, fmf, float(32), Reciprocal::Device);
        assert_eq!(
            function_answer(selected, fast, flagged, f32_),
            want,
            "{op:?} as {selected:?} fast {fast} flagged {flagged}"
        );
        for other in [FloatFmt::BFloat, FloatFmt::F64] {
            assert_eq!(function_answer(op, fast, flagged, other), None, "{op:?}");
        }
        if fast {
            assert_eq!(
                function_answer(op, fast, flagged, FloatFmt::Half),
                None,
                "{op:?}"
            );
        }
    }
}

/// `powr(x, y)` is Metal's canonical NaN where Metal's `exp2(y * log2(x))` is a NaN for operands
/// that are not: `x < 0` (`-0` is not), `y = ±0` with `x = ±0` or `+inf`,
/// and `y = ±inf` with `x = 1`, an `f32` denormal read as a zero by the zero tests and, in
/// `fast::powr` only, by the sign test; every other lane keeps the device's result, which the
/// kernel's third input stands for, except that `fast::powr` answers IEEE's value where `y ·
/// log2(x)` is infinite (`math::infinite_product`'s rule, on the same classification of the
/// operands). `half`'s NaN lanes are the same rule's (`nan_half`, `nan_half_fast`, every lane,
/// class K).
#[test]
fn powr_answers_metals_nan_at_its_special_operands() {
    // The answer where `y · log2(x)` is infinite (`log2(x)` read as `-inf` at a zero), for
    // operands that are not NaNs nor in the NaN rule's lanes; `None` where it is finite.
    fn infinite(x: f64, y: f64, y_negative: bool) -> Option<f64> {
        let log_infinite = x == 0.0 || x.is_infinite();
        if !((log_infinite && y != 0.0) || (y.is_infinite() && x != 1.0)) {
            return None;
        }
        let log_negative = x == 0.0 || (x.is_finite() && x < 1.0);
        Some(if log_negative != y_negative {
            0.0
        } else {
            f64::INFINITY
        })
    }
    let h = float(16);
    let halves: Vec<u64> = vec![
        0, 0x8000, 0x3C00, 0xBC00, 0x4000, 0x3800, 0xB800, 0x7C00, 0xFC00, 0x7E00, 0x0001, 0x7BFF,
    ];
    let device = 0x3D00;
    let cases: Vec<Case> = pairs(&halves, &halves)
        .into_iter()
        .map(|mut c| {
            c.push(vec![device]);
            c
        })
        .collect();
    for fast in [false, true] {
        check_accept(
            &format!("half powr special operands fast={fast}"),
            &[h, h, h],
            h,
            &|fb, x| {
                if fast {
                    math::fast_powr_special(fb, h, x[0], x[1], x[2])
                } else {
                    math::powr_nan(fb, h, x[0], x[1], x[2])
                }
            },
            &cases,
            &|c| {
                let (x, y) = (f16_to_f64(arg(c, 0) as u16), f16_to_f64(arg(c, 1) as u16));
                let nan = x < 0.0
                    || (y == 0.0 && (x == 0.0 || x == f64::INFINITY))
                    || (y.is_infinite() && x == 1.0);
                if nan {
                    return Some(vec![0x7E00]);
                }
                let y_negative = arg(c, 1) & 0x8000 != 0;
                Some(vec![match infinite(x, y, y_negative) {
                    Some(v) if fast && !x.is_nan() && !y.is_nan() => {
                        if v == 0.0 {
                            0
                        } else {
                            0x7C00
                        }
                    }
                    _ => device,
                }])
            },
            &|g, w| g == w || (is_nan(h, g) && is_nan(h, w)),
        );
    }
    let f = float(32);
    let xs: Vec<u64> = [
        0.0f32,
        -0.0,
        1.0,
        -1.0,
        2.0,
        0.5,
        -0.5,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        -f32::NAN,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        // Denormals, which Metal reads as zeros.
        f32::from_bits(1),
        -f32::from_bits(0x007F_FFFF),
    ]
    .iter()
    .map(|v| u64::from(v.to_bits()))
    .collect();
    let device = u64::from(1.25f32.to_bits());
    let cases: Vec<Case> = pairs(&xs, &xs)
        .into_iter()
        .map(|mut c| {
            c.push(vec![device]);
            c
        })
        .collect();
    for fast in [false, true] {
        check_accept(
            &format!("powr special operands fast={fast}"),
            &[f, f, f],
            f,
            &|fb, x| {
                if fast {
                    math::fast_powr_special(fb, f, x[0], x[1], x[2])
                } else {
                    math::powr_nan(fb, f, x[0], x[1], x[2])
                }
            },
            &cases,
            &|c| {
                // A denormal is a zero to Metal's zero tests, and to `fast::powr`'s sign test;
                // the precise sign is the bit's, a NaN's included.
                let read = |v: f32| if v.is_subnormal() { 0.0 } else { v };
                let raw = f32v(arg(c, 0));
                let (x, y) = (read(raw), read(f32v(arg(c, 1))));
                let below = if fast {
                    x < 0.0
                } else {
                    raw.is_sign_negative() && raw != 0.0
                };
                let nan = below
                    || (y == 0.0 && (x == 0.0 || x == f32::INFINITY))
                    || (y.is_infinite() && x == 1.0);
                if nan {
                    return Some(vec![0x7FC0_0000]);
                }
                let y_negative = arg(c, 1) & 0x8000_0000 != 0;
                let product = infinite(f64::from(x), f64::from(y), y_negative);
                Some(vec![match product {
                    Some(v) if fast && !x.is_nan() && !y.is_nan() => {
                        u64::from((v as f32).to_bits())
                    }
                    _ => device,
                }])
            },
            &|g, w| g == w,
        );
    }
}

/// `fast::fract` answers one where `x - floor(x) = 1 + x` rounds to it, `x` in `[-2^-25, 0)` (A1
/// case aed3faaa: `-2^-126`, `-2^-25`, and `-1e-10` give `1.0`), and the module selects that one
/// itself: a device whose `x - floor(x)` is a fract bounded below one (RADV's `v_fract`, which
/// Mesa's NIR folds the subtraction into) still answers Metal's one.
#[test]
fn an_unbounded_fract_selects_one_where_one_plus_x_rounds_to_it() {
    let f = float(32);
    let below_tie = f32::from_bits(0xB300_0001);
    let xs = [
        -2f32.powi(-25),
        below_tie,
        -f32::MIN_POSITIVE,
        -1e-10,
        -2f32.powi(-24),
        -1e-40,
        -0.0,
    ];
    let want = |x: f32| -> u64 {
        let r = if x.to_bits() & 0x7F80_0000 == 0 {
            0.0
        } else if (-2f32.powi(-25)..0.0).contains(&x) {
            1.0
        } else {
            x - x.floor()
        };
        u64::from(r.to_bits())
    };
    assert_eq!(want(below_tie), 0x3F7F_FFFF);
    assert_eq!(want(-1e-10), 0x3F80_0000);
    let cases: Vec<Case> = xs
        .iter()
        .map(|x| vec![vec![u64::from(x.to_bits())]])
        .collect();
    check(
        "fast fract near minus zero",
        &[f],
        f,
        &|fb, x| {
            math::float_fn(
                fb,
                FloatFn::FastFract {
                    infinite_nan: false,
                },
                f,
                Contraction::Forbidden,
                NanTest::IsNan,
                x,
                &Lib::instructions(),
            )
        },
        &cases,
        &|c| Some(vec![want(f32::from_bits(c[0][0] as u32))]),
        false,
    );
    // The bound `-2^-25` is the module's own constant: the one does not come from the
    // subtraction.
    let has_bound = |op: FloatFn| {
        let m = kernel(&[f], f, &|fb, x| {
            math::float_fn(
                fb,
                op,
                f,
                Contraction::Forbidden,
                NanTest::IsNan,
                x,
                &Lib::instructions(),
            )
        })
        .expect("kernel");
        let words = crate::spirv::serialize::serialize(m).expect("words").words;
        const OP_CONSTANT: u32 = 43;
        let mut i = 5;
        let mut found = false;
        while i < words.len() {
            let (n, op) = ((words[i] >> 16) as usize, words[i] & 0xFFFF);
            if op == OP_CONSTANT && n == 4 && words.get(i + 3) == Some(&0xB300_0000) {
                found = true;
            }
            i += n.max(1);
        }
        found
    };
    assert!(has_bound(FloatFn::FastFract {
        infinite_nan: false
    }));
    assert!(!has_bound(FloatFn::Fract));
}

/// The C99 Annex F special cases of `pow` run on the interpreter against the host's `powf`. The
/// interpreter does not model `GLSL.std.450 Pow`, so the kernel takes the magnitude
/// `Pow(|x|, y)` as a third input (computed on the host) and runs only the selects around it.
/// Covers the NaN exponent, which must override the zero-base and `-1` results
/// (`pow(±0, NaN)` and `pow(-1, NaN)` are NaN) but not `pow(+1, NaN) = 1`.
#[test]
fn pow_special_cases_match_c99() {
    let f = float(32);
    let (inf, nan) = (f32::INFINITY, f32::NAN);
    let xs = [
        0.0f32, -0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, inf, -inf, nan,
    ];
    let ys = [
        0.0f32, -0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 0.5, -0.5, inf, -inf, nan,
    ];
    let mut cases: Vec<Case> = Vec::new();
    for x in xs {
        for y in ys {
            let m = x.abs().powf(y);
            cases.push(
                [x, y, m]
                    .iter()
                    .map(|v| vec![u64::from(v.to_bits())])
                    .collect(),
            );
        }
    }
    for precise in [true, false] {
        check(
            &format!("pow special cases f32 (precise {precise})"),
            &[f, f, f],
            f,
            &|fb, a| math::pow_special(fb, f, a[0], a[1], a[2], precise),
            &cases,
            &|c| {
                let r = f32v(arg(c, 0)).powf(f32v(arg(c, 1)));
                Some(vec![u64::from(r.to_bits())])
            },
            false,
        );
    }
}

/// The inputs of the authored case `pi-scaled-sine-and-cosine-are-exact-at-every-half-integer`
/// with Metal's `precise::sinpi` and `precise::cospi` of each (Apple M3 Max, ORACLE_ABI v32; the
/// seven lanes from 4 to -8388607.5, put where the case read irrational values, at v36), for the
/// arguments whose answer the reduction makes exact: the multiples of one half, the large arguments
/// (every one an even integer), and the non-finite ones.
const METAL_SINPI_COSPI: [(u32, u32, u32); 32] = [
    (0x0000_0000, 0x0000_0000, 0x3F80_0000), // 0
    (0x8000_0000, 0x0000_0000, 0x3F80_0000), // -0
    (0x3F00_0000, 0x3F80_0000, 0x0000_0000), // 0.5
    (0x3F80_0000, 0x8000_0000, 0xBF80_0000), // 1
    (0x3FC0_0000, 0xBF80_0000, 0x0000_0000), // 1.5
    (0x4000_0000, 0x0000_0000, 0x3F80_0000), // 2
    (0x4020_0000, 0x3F80_0000, 0x0000_0000), // 2.5
    (0x4040_0000, 0x8000_0000, 0xBF80_0000), // 3
    (0x4060_0000, 0xBF80_0000, 0x0000_0000), // 3.5
    (0xBF00_0000, 0xBF80_0000, 0x0000_0000), // -0.5
    (0xBF80_0000, 0x8000_0000, 0xBF80_0000), // -1
    (0xBFC0_0000, 0x3F80_0000, 0x0000_0000), // -1.5
    (0xC020_0000, 0xBF80_0000, 0x0000_0000), // -2.5
    (0xC040_0000, 0x8000_0000, 0xBF80_0000), // -3
    (0xC080_0000, 0x0000_0000, 0x3F80_0000), // -4
    (0x4080_0000, 0x0000_0000, 0x3F80_0000), // 4
    (0xC000_0000, 0x0000_0000, 0x3F80_0000), // -2
    (0x4090_0000, 0x3F80_0000, 0x0000_0000), // 4.5
    (0xC060_0000, 0x3F80_0000, 0x0000_0000), // -3.5
    (0xC128_0000, 0xBF80_0000, 0x0000_0000), // -10.5
    (0x4AFF_FFFF, 0xBF80_0000, 0x0000_0000), // 8388607.5, the largest float with a fraction
    (0xCAFF_FFFF, 0x3F80_0000, 0x0000_0000), // -8388607.5
    (0x4128_0000, 0x3F80_0000, 0x0000_0000), // 10.5
    (0x42C9_0000, 0x3F80_0000, 0x0000_0000), // 100.5
    (0x4B18_9680, 0x0000_0000, 0x3F80_0000), // 1e7
    (0xCB18_9680, 0x0000_0000, 0x3F80_0000), // -1e7
    (0x4B00_0000, 0x0000_0000, 0x3F80_0000), // 2^23
    (0x4B80_0000, 0x0000_0000, 0x3F80_0000), // 2^24
    (0x60AD_78EC, 0x0000_0000, 0x3F80_0000), // 1e20
    (0x7F80_0000, 0x7FC0_0000, 0x7FC0_0000), // +inf
    (0xFF80_0000, 0x7FC0_0000, 0x7FC0_0000), // -inf
    (0x7FC0_0000, 0x7FC0_0000, 0x7FC0_0000), // NaN
];

/// The inputs of the authored case `tanpi-has-poles-at-the-half-integers` with Metal's `tanpi`
/// of each (both `fast::` and `precise::` answer the same; M3 Max, ORACLE_ABI v32), the quarter
/// turns included: there `tanpi` is exactly `±1` whatever `Sin`/`Cos` round `π/4` to.
const METAL_TANPI: [(u32, u32); 12] = [
    (0x3F00_0000, 0x7F80_0000), // 0.5: +inf
    (0xBF00_0000, 0xFF80_0000), // -0.5: -inf
    (0x3FC0_0000, 0xFF80_0000), // 1.5: -inf
    (0x42C9_0000, 0x7F80_0000), // 100.5: +inf
    (0x4640_E600, 0xFF80_0000), // 12345.5: -inf
    (0x0000_0000, 0x0000_0000), // 0
    (0x8000_0000, 0x0000_0000), // -0: +0
    (0x3F80_0000, 0x0000_0000), // 1: +0
    (0xBF80_0000, 0x0000_0000), // -1: +0
    (0x3E80_0000, 0x3F80_0000), // 0.25: 1
    (0xBE80_0000, 0xBF80_0000), // -0.25: -1
    (0x7F80_0000, 0x7FC0_0000), // +inf: NaN (the case stores isnan)
];

/// What the stand-in for `Sin` and `Cos` is off by: inexact at 0, as Vulkan's `2^-11` absolute
/// bound allows a device to be.
const SIN_COS_SLOP: f32 = 1.0 / 1_048_576.0;

/// The reduction of [`math::pi_scaled`] in `f32` on the host, with the stand-in
/// `(a + e, (1 + a) - e)` ([`SIN_COS_SLOP`]) for the sine and cosine of the residue angle `a`,
/// replaced by the exact `(+0, 1)` on an axis; `tanpi` is `±1` at a quarter turn.
fn pi_scaled_model(which: math::PiScaled, x: f32) -> f32 {
    if !x.is_finite() {
        return f32::from_bits(0x7FC0_0000);
    }
    let reduced = x - (x * 0.5).round_ties_even() * 2.0;
    let q = (reduced * 2.0).round_ties_even();
    let r = reduced - q * 0.5;
    let a = std::f32::consts::PI * r;
    let (s, c) = if r == 0.0 {
        (0.0, 1.0)
    } else {
        (a + SIN_COS_SLOP, (1.0 + a) - SIN_COS_SLOP)
    };
    let pick = |arms: [f32; 4]| {
        if q == 0.0 {
            arms[0]
        } else if q == 1.0 {
            arms[1]
        } else if q == -1.0 {
            arms[2]
        } else {
            arms[3]
        }
    };
    let sin = pick([s, c, -c, -s]);
    let cos = pick([c, if r == 0.0 { 0.0 } else { -s }, s, -c]);
    match which {
        math::PiScaled::Sin => sin,
        math::PiScaled::Cos => cos,
        math::PiScaled::Tan if r.abs() == 0.25 => 1f32.copysign(sin / cos),
        math::PiScaled::Tan => sin / cos,
    }
}

/// Metal's `sinpi`/`cospi`/`tanpi` reduce their argument exactly (period 2) instead of scaling it
/// by `float(π)`. The interpreter does not model `Sin`/`Cos`, so the kernel runs the whole
/// reduction, quadrant selection, and finiteness guard around the stand-in `(a + e, (1 + a) - e)`
/// for the residue angle's sine and cosine. It is inexact at 0 (`Sin(0) = 2^-20`, `Cos(0) = 1 -
/// 2^-20`, within Vulkan's bound), so the exact zeros and poles must come from the lowering's axis
/// arms, not from the device, and it is distinct for every arm elsewhere, so at a quarter turn its
/// arms differ in magnitude, as a device's `Sin` and `Cos` of `π/4` may. At every argument of
/// [`METAL_SINPI_COSPI`] and [`METAL_TANPI`] the kernel gives Metal's observed answer bit for
/// bit (signed zeros, signed infinities, the quiet NaN, and `tanpi`'s `±1` at the quarter turns
/// included); on a grid of other arguments it gives the host model of the same reduction, which
/// checks the quadrant arms and the quarter turns' signs.
#[test]
fn pi_scaled_functions_reduce_exactly_to_metal_answers() {
    use math::PiScaled;
    let mut grid: Vec<f32> = (-48..=48).map(|k| k as f32 / 8.0).collect();
    grid.extend([
        0.1,
        1.7,
        -2.3,
        0.999_999_9,
        1.000_000_1,
        2.999_999_8,
        1e-8,
        0.3,
        0.7,
    ]);
    for which in [PiScaled::Sin, PiScaled::Cos, PiScaled::Tan] {
        let metal: Vec<(u32, u32)> = match which {
            PiScaled::Sin => METAL_SINPI_COSPI.iter().map(|r| (r.0, r.1)).collect(),
            PiScaled::Cos => METAL_SINPI_COSPI.iter().map(|r| (r.0, r.2)).collect(),
            PiScaled::Tan => METAL_TANPI.to_vec(),
        };
        // The host model agrees with Metal wherever Metal's answer is exact.
        for (x, want) in &metal {
            let got = pi_scaled_model(which, f32::from_bits(*x)).to_bits();
            assert_eq!(got, *want, "{which:?} model at {x:#010x}");
        }
        let mut cases: Vec<Case> = metal
            .iter()
            .map(|(x, _)| vec![vec![u64::from(*x)]])
            .collect();
        cases.extend(grid.iter().map(|x| vec![vec![u64::from(x.to_bits())]]));
        for f in [float(32), vec(float(32), 4)] {
            let lanes = usize::from(f.lanes);
            let cases: Vec<Case> = cases
                .chunks(lanes)
                .filter(|c| c.len() == lanes)
                .map(|c| vec![c.iter().map(|l| l[0][0]).collect()])
                .collect();
            check_accept(
                &format!("{which:?} {f:?}"),
                &[f],
                f,
                &|fb, x| {
                    math::pi_scaled(
                        fb,
                        f,
                        which,
                        x[0],
                        &|fb, r| {
                            // The device path's angle, `float(π) r`.
                            let pi = ops::fsplat(fb, f, std::f64::consts::PI)?;
                            let a = fb.binary(BinOp::FMul, pi, r)?;
                            let one = ops::fsplat(fb, f, 1.0)?;
                            let slop = ops::fsplat(fb, f, f64::from(SIN_COS_SLOP))?;
                            let sine = fb.binary(BinOp::FAdd, a, slop)?;
                            let shifted = fb.binary(BinOp::FAdd, one, a)?;
                            Ok((sine, fb.binary(BinOp::FSub, shifted, slop)?))
                        },
                        &|fb, num, den| fb.binary(BinOp::FDiv, num, den),
                    )
                },
                &cases,
                &|c| {
                    Some(
                        c[0].iter()
                            .map(|x| {
                                let x = *x as u32;
                                let r = metal.iter().find(|(m, _)| *m == x).map_or_else(
                                    || pi_scaled_model(which, f32::from_bits(x)).to_bits(),
                                    |(_, w)| *w,
                                );
                                u64::from(r)
                            })
                            .collect(),
                    )
                },
                &|g, w| g == w,
            );
        }
    }
}

/// [`FloatFn::words`] bounds what `sinpi`, `cospi`, and `tanpi` emit on every form: on `float` and
/// `half` the calls of Metal's routines (the routines are charged on their own), a fast call's on
/// `float` also where the device's functions are Metal's, which is the exact reduction around the
/// device's `Sin` and `Cos`, as on a `bfloat`, which no routine answers, the axis arms and
/// `tanpi`'s quarter turns included. The module's growth over an identity kernel (with the same
/// routines) is charged in full, its constants and the `GLSL.std.450` import included.
#[test]
fn pi_scaled_functions_stay_within_their_word_estimate() {
    use super::libm::{self, Fun, Mode};
    use std::collections::BTreeMap;
    let routines = [Mode::Precise, Mode::Fast]
        .map(|m| [Fun::SinPi(m), Fun::CosPi(m), Fun::TanPi(m)])
        .concat();
    let defs = |b: &mut ModuleBuilder| -> BResult<Vec<FuncId>> {
        routines
            .iter()
            .map(|fun| libm::define_alone(b, *fun, false))
            .collect()
    };
    let module_words = |f: Form, body: BodyCalling<'_>| {
        let m = kernel_defs(&[f], f, &defs, body, false, false).expect("kernel");
        crate::spirv::serialize::serialize(m)
            .expect("words")
            .words
            .len()
    };
    let precise = [FloatFn::SinPi, FloatFn::CosPi, FloatFn::TanPi];
    let fast = [FloatFn::FastSinPi, FloatFn::FastCosPi, FloatFn::FastTanPi];
    for f in [float(32), vec(float(32), 4), float(16), vec(float(16), 4)] {
        let bare = module_words(f, &|_, x, _| Ok(x[0]));
        // `refine_float` gives a `half` call no fast function.
        let calls = if f.float_fmt() == Some(FloatFmt::F32) {
            [precise, fast].concat()
        } else {
            precise.to_vec()
        };
        for g in calls {
            for metal in [None, Some(Mode::Fast), Some(Mode::Precise)] {
                let used = module_words(f, &|fb, x, funcs| {
                    let map: BTreeMap<Fun, FuncId> = routines
                        .iter()
                        .copied()
                        .zip(funcs.iter().copied())
                        .collect();
                    math::float_fn(
                        fb,
                        g,
                        f,
                        Contraction::Allowed,
                        NanTest::IsNan,
                        x,
                        &Lib::new(metal, &map),
                    )
                });
                let (emitted, bound) = (used - bare, g.words(f) as usize);
                assert!(
                    emitted <= bound,
                    "{g:?} {f:?} with Metal's functions {metal:?}: {emitted} words over {bound}"
                );
            }
        }
    }
    let module_words = |f: Form, body: Body<'_>| {
        let m = kernel(&[f], f, body).expect("kernel");
        crate::spirv::serialize::serialize(m)
            .expect("words")
            .words
            .len()
    };
    for f in [bfloat(), vec(bfloat(), 4)] {
        let bare = module_words(f, &|_, x| Ok(x[0]));
        for g in [FloatFn::SinPi, FloatFn::CosPi, FloatFn::TanPi] {
            let used = module_words(f, &|fb, x| {
                math::float_fn(
                    fb,
                    g,
                    f,
                    Contraction::Allowed,
                    NanTest::IsNan,
                    x,
                    &Lib::instructions(),
                )
            });
            let (emitted, bound) = (used - bare, g.words(f) as usize);
            assert!(
                emitted <= bound,
                "{g:?} {f:?}: {emitted} words over {bound}"
            );
        }
    }
}

/// `air.fast_tanh` on `float` is the MSL fast-math formula `(t - 1) / (t + 1)` of `t = exp(2x)`,
/// divided as Metal's fast division (`(t - 1) · (1 / (t + 1))`, the reciprocal flushed past
/// `2^126`). The interpreter does not model `Exp2`, so the kernel takes `t` as an input (`exp2(x *
/// 2 log2 e)` in `f32` on the host) and runs the quotient. Where the quotient does not depend on
/// the last place of `t` it gives the Metal observation of the authored case
/// `fast-tanh-through-all-three-of-its-regimes` (M3 Max, ORACLE_ABI v32): `x = 0` is 0, `x = 43`
/// is 1, `x = 44` (where `t + 1` is above `2^126` and the reciprocal flushes) is 0, `x = 44.4`
/// (where `t` overflows) is NaN, and `x = -100` (where `t` is 0) is -1.
#[test]
fn fast_tanh_is_the_fast_math_formula() {
    let f = float(32);
    let two_log2e = (2.0 * std::f64::consts::LOG2_E) as f32;
    let t_of = |x: f32| (x * two_log2e).exp2();
    // (x, Metal's fast::tanh(x)).
    let metal: [(f32, u32); 5] = [
        (0.0, 0x0000_0000),
        (43.0, 0x3F80_0000),
        (44.0, 0x0000_0000),
        (44.4, 0x7FC0_0000),
        (-100.0, 0xBF80_0000),
    ];
    // Metal's fast quotient: the numerator times the reciprocal, flushed past 2^126.
    let quotient = |t: f32| {
        let den = t + 1.0;
        let r = if den.abs() > 2f32.powi(126) {
            0.0
        } else {
            1.0 / den
        };
        (t - 1.0) * r
    };
    let mut xs: Vec<f32> = metal.iter().map(|m| m.0).collect();
    xs.extend([0.5, -3.75, 5.0, 1e-3, -1e-3, 20.0, -44.4, 43.7]);
    let cases: Vec<Case> = xs
        .iter()
        .map(|x| vec![vec![u64::from(t_of(*x).to_bits())]])
        .collect();
    check(
        "fast tanh quotient f32",
        &[f],
        f,
        &|fb, t| {
            let divide = |fb: &mut FunctionBuilder<'_>, x: Value, y: Value| {
                math::fast_divide(fb, f, x, y, Contraction::Forbidden, Reciprocal::Device)
            };
            math::tanh_of_exp(fb, f, &divide, t[0])
        },
        &cases,
        &|c| {
            let t = f32v(arg(c, 0));
            Some(vec![u64::from(quotient(t).to_bits())])
        },
        false,
    );
    check(
        "fast tanh quotient f32, the reciprocal rounded",
        &[f],
        f,
        &|fb, t| {
            let divide = |fb: &mut FunctionBuilder<'_>, x: Value, y: Value| {
                math::fast_divide(fb, f, x, y, Contraction::Forbidden, Reciprocal::Rounded)
            };
            math::tanh_of_exp(fb, f, &divide, t[0])
        },
        &cases,
        &|c| {
            let t = f32v(arg(c, 0));
            Some(vec![u64::from(quotient(t).to_bits())])
        },
        false,
    );
    for (x, want) in metal {
        let got = quotient(t_of(x));
        assert!(
            got.to_bits() == want || (got.is_nan() && f32::from_bits(want).is_nan()),
            "fast::tanh({x}) = {got} ({:#010x}), Metal {want:#010x}",
            got.to_bits()
        );
    }
    // x = 44: t + 1 > 2^126, where the reciprocal flushes.
    assert!(t_of(44.0) + 1.0 > 2f32.powi(126));
}

/// The function a call computes follows its `air` symbol's `fast_` variant, never a name: on
/// `float`, `fast_tanh` is the fast-math formula, `fast_fract` is unbounded, `fast_fmedian3`
/// answers the second of two NaNs, `fast_pow` the canonical NaN for a NaN base to the exponent one,
/// `fast_atan2` is the turned `atan` of the quotient, `fast_sin` and `fast_cos` flush past the last
/// evaluated quadrant, and `fast_sqrt`, `fast_exp`, `fast_log`, and `fast_powr` are Metal's
/// compositions; a narrow `fast_` call and the precise ones keep the precise function. The
/// fast-math flags change only `fast_atan2`'s quotient beyond `2^126` (a call with every flag
/// divides there, any other call flushes the reciprocal) and the infinities of `fast_fract` and
/// `fast_asinh` (a call with every flag answers a NaN; any other call is `fract`'s `+0` and the
/// precise `asinh`), and `llvm.minnum` and `llvm.maxnum` on `float`, which with every flag are
/// Metal's `fmin` and `fmax`.
#[test]
fn fast_variants_refine_float_functions() {
    use crate::air::ir::FastMath;
    let refine = |op: FloatFn, fast: bool, fmf: FastMath, f: Form| {
        super::refine_float(op, fast, fmf, f, Reciprocal::Rounded)
    };
    let every = FastMath::ALL
        .iter()
        .fold(FastMath::NONE, |all, (flag, _)| all.union(*flag));
    let some = FastMath::NNAN.union(FastMath::NINF).union(FastMath::NSZ);
    let f32_ = float(32);
    for fmf in [FastMath::NONE, some, every] {
        for (op, fast) in [
            (
                FloatFn::Tanh,
                FloatFn::FastTanh {
                    reciprocal: Reciprocal::Rounded,
                },
            ),
            (
                FloatFn::Asinh,
                if fmf == every {
                    FloatFn::FastAsinh
                } else {
                    FloatFn::Asinh
                },
            ),
            (
                FloatFn::Fract,
                FloatFn::FastFract {
                    infinite_nan: fmf == every,
                },
            ),
            (FloatFn::Median3, FloatFn::FastMedian3),
            (FloatFn::Pow, FloatFn::FastPow),
            (FloatFn::Sin, FloatFn::FastSin),
            (FloatFn::Cos, FloatFn::FastCos),
            (
                FloatFn::Tan,
                FloatFn::FastTan {
                    reciprocal: Reciprocal::Rounded,
                },
            ),
            (FloatFn::Asin, FloatFn::FastAsin),
            (FloatFn::Acos, FloatFn::FastAcos),
            (FloatFn::Atan, FloatFn::FastAtan),
            (FloatFn::Sqrt, FloatFn::FastSqrt),
            (FloatFn::Exp, FloatFn::FastExp),
            (FloatFn::Log, FloatFn::FastLog),
            (FloatFn::Log2, FloatFn::FastLog2),
            (FloatFn::Powr, FloatFn::FastPowr),
            (
                FloatFn::Atan2,
                FloatFn::FastAtan2 {
                    reciprocal_flushed: fmf != every,
                },
            ),
            (FloatFn::SinPi, FloatFn::FastSinPi),
            (FloatFn::CosPi, FloatFn::FastCosPi),
            (FloatFn::TanPi, FloatFn::FastTanPi),
        ] {
            assert_eq!(refine(op, true, fmf, f32_), fast);
            assert_eq!(refine(op, false, fmf, f32_), op);
            // A precise `half` `fract` with every flag answers `0x3BFF` at a NaN and an infinity; a
            // `fast_` `half` symbol is not measured and keeps `op`.
            let half = if op == FloatFn::Fract && fmf == every {
                FloatFn::FlaggedFract
            } else {
                op
            };
            assert_eq!(refine(op, true, fmf, float(16)), op);
            assert_eq!(refine(op, false, fmf, float(16)), half);
            assert_eq!(refine(op, true, fmf, bfloat()), op);
        }
        for op in [FloatFn::Sign, FloatFn::Saturate] {
            assert_eq!(refine(op, true, fmf, f32_), op);
        }
        // `llvm.minnum` and `llvm.maxnum` (never `fast_`), Metal's fast-math `a < b ? a : b` and
        // `a > b ? a : b`, are Metal's `fmin` and `fmax` on `float` with any flags; on the other
        // formats they keep LLVM's function.
        for (op, metal) in [
            (FloatFn::MinNum, FloatFn::Min),
            (FloatFn::MaxNum, FloatFn::Max),
        ] {
            assert_eq!(refine(op, false, fmf, f32_), metal);
            assert_eq!(refine(op, false, fmf, float(16)), op);
            assert_eq!(refine(op, false, fmf, bfloat()), op);
        }
    }
    // The fast division's reciprocal is the profile's.
    let device = super::refine_float(
        FloatFn::Tanh,
        true,
        FastMath::NONE,
        f32_,
        Reciprocal::Device,
    );
    assert_eq!(
        device,
        FloatFn::FastTanh {
            reciprocal: Reciprocal::Device
        }
    );
    assert!(device.divides() && !FloatFn::Tanh.divides());
    assert_eq!(
        super::refine_float(FloatFn::Tan, true, FastMath::NONE, f32_, Reciprocal::Device),
        FloatFn::FastTan {
            reciprocal: Reciprocal::Device
        }
    );
    for reciprocal_flushed in [false, true] {
        assert!(FloatFn::FastAtan2 { reciprocal_flushed }.divides());
    }
    assert!(!FloatFn::Atan2.divides());
}

/// The transcendental and device-precision functions build and pass L3 (the interpreter does not
/// model them), `fast::tanh` with its division routine; `ldexp` and `dot` run: part `part` of `of`
/// of the checks.
fn device_precision_float_functions(part: usize, of: usize) {
    let device = [
        FloatFn::Rsqrt,
        FloatFn::FastSqrt,
        FloatFn::Exp,
        FloatFn::FastExp,
        FloatFn::Exp2,
        FloatFn::Exp10,
        FloatFn::Log,
        FloatFn::FastLog,
        FloatFn::Log2,
        FloatFn::FastLog2,
        FloatFn::Log10,
        FloatFn::Sin,
        FloatFn::Cos,
        FloatFn::FastSin,
        FloatFn::FastCos,
        FloatFn::Tan,
        FloatFn::FastTan {
            reciprocal: Reciprocal::Device,
        },
        FloatFn::Asin,
        FloatFn::FastAsin,
        FloatFn::Acos,
        FloatFn::FastAcos,
        FloatFn::Atan,
        FloatFn::FastAtan,
        FloatFn::Sinh,
        FloatFn::Cosh,
        FloatFn::Tanh,
        FloatFn::Asinh,
        FloatFn::FastAsinh,
        FloatFn::Acosh,
        FloatFn::Atanh,
        FloatFn::SinPi,
        FloatFn::CosPi,
        FloatFn::TanPi,
        FloatFn::FastSinPi,
        FloatFn::FastCosPi,
        FloatFn::FastTanPi,
        FloatFn::Powr,
        FloatFn::FastPowr,
        FloatFn::Pow,
        FloatFn::FastPow,
        FloatFn::Atan2,
        FloatFn::FastAtan2 {
            reciprocal_flushed: false,
        },
        FloatFn::FastAtan2 {
            reciprocal_flushed: true,
        },
        FloatFn::Fmod,
        FloatFn::Fma,
    ];
    let mut checks: Vec<Box<dyn Fn()>> = Vec::new();
    for f in [vec(float(16), 3), vec(float(32), 3), vec(bfloat(), 3)] {
        for op in device {
            checks.push(Box::new(move || {
                let m = kernel(&vec![f; op.arity()], f, &|fb, x| {
                    math::float_fn(
                        fb,
                        op,
                        f,
                        Contraction::Forbidden,
                        NanTest::IsNan,
                        x,
                        &Lib::instructions(),
                    )
                })
                .unwrap_or_else(|e| panic!("{op:?}: {e:?}"));
                assert_valid(m, &wide_profile());
            }));
        }
    }
    for lanes in [1u8, 3] {
        let f = vec(float(32), lanes);
        for reciprocal in [Reciprocal::Device, Reciprocal::Rounded] {
            checks.push(Box::new(move || {
                let routine = math::FastDivision::new(f, reciprocal, Contraction::Forbidden);
                let m = kernel_defs(
                    &[f],
                    f,
                    &|b| Ok(vec![math::define_fast_division(b, routine)?]),
                    &|fb, x, funcs| {
                        let divide = |fb: &mut FunctionBuilder<'_>, p: Value, q: Value| {
                            math::call_fast_division(fb, funcs[0], p, q)
                        };
                        math::fast_tanh(
                            fb,
                            FloatFn::FastTanh { reciprocal },
                            f,
                            x[0],
                            &Lib::instructions(),
                            &divide,
                        )
                    },
                    false,
                    false,
                )
                .unwrap_or_else(|e| panic!("fast tanh {reciprocal:?}: {e:?}"));
                assert_valid(m, &wide_profile());
            }));
        }
    }
    checks.push(Box::new(|| {
        let f = float(32);
        let cases: Vec<Case> = [(1.5f32, 3u32), (-0.75, 0xFFFF_FFFE), (1.0, 127), (3.0, 0)]
            .iter()
            .map(|(x, e)| vec![vec![u64::from(x.to_bits())], vec![u64::from(*e)]])
            .collect();
        check(
            "ldexp f32",
            &[f, int(32)],
            f,
            &|fb, x| {
                math::float_fn(
                    fb,
                    FloatFn::Ldexp,
                    f,
                    Contraction::Forbidden,
                    NanTest::IsNan,
                    x,
                    &Lib::instructions(),
                )
            },
            &cases,
            &|c| {
                let e = arg(c, 1) as u32 as i32;
                Some(vec![u64::from((f32v(arg(c, 0)) * 2f32.powi(e)).to_bits())])
            },
            false,
        );
    }));
    // `ldexp` answers Metal's for every exponent: past GLSL's `[-126, 128]` and past the format's
    // range, a denormal operand read as zero, a result below `2^-126` flushed, a NaN kept bit
    // for bit (`math::ldexp_bits`); `half` rounds the exact product once.
    checks.push(Box::new(|| {
        let f = float(32);
        let bits: [u32; 12] = [
            0x3FC0_0000, // 1.5
            0xBFC0_0000, // -1.5
            0x7180_0000, // 2^100
            0x0080_0000, // 2^-126
            0x0000_0001, // the least denormal
            0x8000_0005, // a negative denormal
            0x7FC1_2345, // a quiet NaN with a payload
            0xFFA0_0001, // a negative signaling NaN
            0x7F80_0000, // +inf
            0x8000_0000, // -0
            0x7F7F_FFFF, // the greatest finite
            0x4040_0000, // 3.0
        ];
        let exps: [i32; 16] = [
            0,
            1,
            -1,
            127,
            128,
            129,
            -126,
            -127,
            -150,
            200,
            -200,
            300,
            -300,
            1000,
            i32::MAX,
            i32::MIN,
        ];
        let cases: Vec<Case> = bits
            .iter()
            .flat_map(|&x| {
                exps.iter()
                    .map(move |&e| vec![vec![u64::from(x)], vec![u64::from(e as u32)]])
            })
            .collect();
        // Metal's answer, from the exact product in `f64`.
        let metal = |x: u32, e: i32| -> u32 {
            let (sign, field) = (x & 0x8000_0000, (x >> 23) & 0xFF);
            match field {
                0 => sign,
                0xFF => x,
                _ => {
                    let v = f64::from(f32::from_bits(x)) * 2f64.powi(e.clamp(-1000, 1000));
                    match v.abs() {
                        a if a >= 2f64.powi(128) => sign | 0x7F80_0000,
                        a if a < 2f64.powi(-126) => sign,
                        _ => (v as f32).to_bits(),
                    }
                }
            }
        };
        check_accept(
            "ldexp f32, every exponent",
            &[f, int(32)],
            f,
            &|fb, x| {
                math::float_fn(
                    fb,
                    FloatFn::Ldexp,
                    f,
                    Contraction::Forbidden,
                    NanTest::IsNan,
                    x,
                    &Lib::instructions(),
                )
            },
            &cases,
            &|c| {
                let (x, e) = (arg(c, 0) as u32, arg(c, 1) as u32 as i32);
                Some(vec![u64::from(metal(x, e))])
            },
            &|g, w| g == w,
        );
    }));
    checks.push(Box::new(|| {
        let h = float(16);
        let halves: [f64; 7] = [
            1.5,
            -1.5,
            65504.0,
            2f64.powi(-24),
            -2f64.powi(-14),
            0.0,
            3.0,
        ];
        let exps: [i32; 12] = [0, 1, -1, 15, 16, -14, -25, -40, 40, 64, 1000, i32::MIN];
        let cases: Vec<Case> = halves
            .iter()
            .flat_map(|&x| {
                exps.iter()
                    .map(move |&e| vec![vec![u64::from(f64_to_f16(x))], vec![u64::from(e as u32)]])
            })
            .collect();
        check(
            "ldexp half, every exponent",
            &[h, int(32)],
            h,
            &|fb, x| {
                math::float_fn(
                    fb,
                    FloatFn::Ldexp,
                    h,
                    Contraction::Forbidden,
                    NanTest::IsNan,
                    x,
                    &Lib::instructions(),
                )
            },
            &cases,
            &|c| {
                let x = f16_to_f64(arg(c, 0) as u16);
                let e = (arg(c, 1) as u32 as i32).clamp(-1000, 1000);
                Some(vec![u64::from(f64_to_f16(x * 2f64.powi(e)))])
            },
            false,
        );
    }));
    checks.push(Box::new(|| {
        let f = float(32);
        let v = vec(f, 3);
        let dots: Vec<Case> = vec![vec![
            [1.0f32, 2.0, 3.0]
                .iter()
                .map(|x| u64::from(x.to_bits()))
                .collect(),
            [4.0f32, -5.0, 0.5]
                .iter()
                .map(|x| u64::from(x.to_bits()))
                .collect(),
        ]];
        check(
            "dot vec3",
            &[v, v],
            f,
            &|fb, x| {
                math::float_fn(
                    fb,
                    FloatFn::Dot,
                    v,
                    Contraction::Forbidden,
                    NanTest::IsNan,
                    x,
                    &Lib::instructions(),
                )
            },
            &dots,
            &|_| Some(vec![u64::from((4.0f32 - 10.0 + 1.5).to_bits())]),
            false,
        );
    }));
    for run in part_of(checks, part, of) {
        run();
    }
}

mod device_precision_float_functions_are_valid {
    parts!(super::device_precision_float_functions, 8;
        part_1_of_8 = 0, part_2_of_8 = 1, part_3_of_8 = 2, part_4_of_8 = 3,
        part_5_of_8 = 4, part_6_of_8 = 5, part_7_of_8 = 6, part_8_of_8 = 7);
}

// ---- air.convert ----

#[test]
fn saturating_float_to_integer_convert_matches_metal() {
    for (fb_, vals) in [(32u8, f32_values()), (64, f64_values())] {
        let ff = float(fb_);
        let cases: Vec<Case> = vals.iter().map(|v| vec![vec![*v]]).collect();
        for ibits in [5u8, 8, 16, 32, 40, 64] {
            let tf = int(ibits);
            for signed in [true, false] {
                check(
                    &format!(
                        "convert f{fb_} to {}{ibits}",
                        if signed { "s" } else { "u" }
                    ),
                    &[ff],
                    tf,
                    &|fb, x| {
                        math::convert(
                            fb,
                            ConvertKind::ToInt { signed },
                            ff,
                            tf,
                            x[0],
                            NanTest::IsNan,
                        )
                    },
                    &cases,
                    &|c| {
                        // Round toward zero, saturate at the bounds, NaN to 0 (Rust `as`).
                        let x = fval(fb_, arg(c, 0));
                        let v: i128 = if x.is_nan() {
                            0
                        } else if signed {
                            (x.trunc().clamp(smin(ibits) as f64, smax(ibits) as f64)) as i128
                        } else {
                            (x.trunc().clamp(0.0, umax(ibits) as f64)) as i128
                        };
                        let v = if signed {
                            v.clamp(smin(ibits), smax(ibits))
                        } else {
                            v.clamp(0, umax(ibits))
                        };
                        Some(vec![(v as u64) & mask(ibits)])
                    },
                    false,
                );
            }
        }
    }
}

/// Metal's saturating conversion of `x` to a `bits`-wide integer: truncated toward zero and
/// clamped to the destination's range, NaN to 0 (`as i128` truncates, and saturates far
/// outside every destination, exactly).
fn saturated(x: f64, bits: u8, signed: bool) -> u64 {
    let (lo, hi) = if signed {
        (smin(bits), smax(bits))
    } else {
        (0, umax(bits))
    };
    ((x as i128).clamp(lo, hi) as u64) & mask(bits)
}

/// A saturating conversion selects the maximum only where the clamp's upper bound does not truncate
/// to it, and no minimum (`-2^k` is a float of both containers); every source value around each
/// destination's bounds (`±2^k` and the floats beside them, `2^k ± 1`, the largest finite floats,
/// infinities, and NaN) still converts as Metal does, for `f32` and `f64` into every 8-, 16-, 32-,
/// and 64-bit destination; and every half pattern does too.
#[test]
fn saturating_converts_saturate_at_every_bound() {
    let reference = |fw: u8, ibits: u8, signed: bool| {
        move |c: &Case| Some(vec![saturated(fval(fw, arg(c, 0)), ibits, signed)])
    };
    for fw in [32u8, 64] {
        let ff = float(fw);
        let bits = |x: f64| -> u64 {
            if fw == 32 {
                u64::from((x as f32).to_bits())
            } else {
                x.to_bits()
            }
        };
        let next = |b: u64, up: bool| if up { b + 1 } else { b - 1 };
        for ibits in [8u8, 16, 32, 64] {
            let mut vals = vec![
                bits(f64::INFINITY),
                bits(f64::NEG_INFINITY),
                bits(f64::NAN),
                bits(f64::MAX),
                bits(-f64::MAX),
                bits(0.0),
                bits(-0.5),
            ];
            // NaNs whose payload is only the lowest bit, and a negative one: the wide bit test
            // reads the low word.
            vals.extend(if fw == 32 {
                [0x7F80_0001, 0xFFC0_0000]
            } else {
                [0x7FF0_0000_0000_0001, 0xFFF8_0000_0000_0000]
            });
            for k in [u32::from(ibits) - 1, u32::from(ibits)] {
                let p = 2f64.powi(k as i32);
                for x in [p, -p, p - 1.0, -p + 1.0, p + 1.0, -p - 1.0] {
                    let b = bits(x);
                    vals.extend([b, next(b, true), next(b, false)]);
                }
            }
            let cases: Vec<Case> = vals.iter().map(|v| vec![vec![*v]]).collect();
            // Either NaN test converts NaN to 0, in both containers.
            for (signed, test) in [true, false]
                .into_iter()
                .flat_map(|s| [(s, NanTest::IsNan), (s, NanTest::Bits)])
            {
                check(
                    &format!(
                        "bounds f{fw} to {}{ibits} ({test:?})",
                        if signed { "s" } else { "u" }
                    ),
                    &[ff],
                    int(ibits),
                    &|fb, x| {
                        math::convert(
                            fb,
                            ConvertKind::ToInt { signed },
                            ff,
                            int(ibits),
                            x[0],
                            test,
                        )
                    },
                    &cases,
                    &reference(fw, ibits, signed),
                    false,
                );
            }
        }
    }
    let half = float(16);
    let every: Vec<Case> = (0..=0xFFFFu64).map(|h| vec![vec![h]]).collect();
    for ibits in [8u8, 16, 32, 64] {
        for signed in [true, false] {
            check(
                &format!("every half to {}{ibits}", if signed { "s" } else { "u" }),
                &[half],
                int(ibits),
                &|fb, x| {
                    math::convert(
                        fb,
                        ConvertKind::ToInt { signed },
                        half,
                        int(ibits),
                        x[0],
                        NanTest::IsNan,
                    )
                },
                &every,
                &|c| Some(vec![saturated(f16_to_f64(arg(c, 0) as u16), ibits, signed)]),
                false,
            );
        }
    }
}

#[test]
fn integer_and_float_converts_follow_their_source_kind() {
    let mut rng = Rng(11);
    for (from, to) in [(8u8, 32u8), (32, 8), (16, 64), (64, 16), (5, 40), (32, 32)] {
        let (ff, tf) = (int(from), int(to));
        let cases: Vec<Case> = int_values(from, &mut rng, 4)
            .iter()
            .map(|v| vec![vec![*v]])
            .collect();
        for signed in [true, false] {
            check(
                &format!("convert i{from} to i{to} signed={signed}"),
                &[ff],
                tf,
                &|fb, x| {
                    math::convert(
                        fb,
                        ConvertKind::Int { signed },
                        ff,
                        tf,
                        x[0],
                        NanTest::IsNan,
                    )
                },
                &cases,
                &|c| {
                    let v = arg(c, 0);
                    let x = if signed { sext(v, from) as u64 } else { v };
                    Some(vec![x & mask(to)])
                },
                false,
            );
        }
    }
    let (f32f, i16f) = (float(32), int(16));
    let cases: Vec<Case> = [0u64, 1, 0x7FFF, 0x8000, 0xFFFF]
        .iter()
        .map(|v| vec![vec![*v]])
        .collect();
    for signed in [true, false] {
        check(
            &format!("convert i16 to f32 signed={signed}"),
            &[i16f],
            f32f,
            &|fb, x| {
                math::convert(
                    fb,
                    ConvertKind::ToFloat { signed },
                    i16f,
                    f32f,
                    x[0],
                    NanTest::IsNan,
                )
            },
            &cases,
            &|c| {
                let v = arg(c, 0);
                let x = if signed { sext(v, 16) as f32 } else { v as f32 };
                Some(vec![u64::from(x.to_bits())])
            },
            false,
        );
    }
}

// ---- Aggregate bounds ----

/// A small AIR type that expands to many scalar leaves is refused from its type alone, before any
/// representation is built: `[64 x [64 x [64 x [64 x [64 x i32]]]]]` would be about 1.07e9
/// leaves. Types within the total cap still lower, whatever their per-level width.
#[test]
fn aggregate_leaf_total_is_capped_before_expansion() {
    use super::legal::{demote_of, MAX_DEPTH, MAX_SCALARS};
    use crate::air::ir::{Type as AirType, TypeTable};
    use crate::budget::{Budget, Meter};
    use crate::error::{Error, Needs};
    use crate::spirv::ids::DemoteTy;

    let refused = |r: Result<DemoteTy, Error>| matches!(r, Err(Error::Unsupported(ref u)) if u.needs == Needs::NotImplemented);
    let mut meter = Meter::new(&Budget::default());
    let mut types = TypeTable::new();
    let i32t = types.intern(AirType::Int(32), &mut meter).unwrap();
    let mut wide = i32t;
    for _ in 0..5 {
        wide = types
            .intern(
                AirType::Array {
                    elem: wide,
                    len: 64,
                },
                &mut meter,
            )
            .unwrap();
    }
    let start = std::time::Instant::now();
    assert!(refused(demote_of(&types, wide)));
    // The refusal does no expansion; a generous bound keeps this robust on a loaded host.
    assert!(start.elapsed() < std::time::Duration::from_secs(1));

    // A struct of many narrow fields, each a nested wide array, is refused the same way.
    let a4 = types
        .intern(
            AirType::Array {
                elem: i32t,
                len: 64,
            },
            &mut meter,
        )
        .unwrap();
    let s = types
        .intern(
            AirType::Struct {
                fields: vec![a4; 64].into(),
                packed: false,
                name: None,
            },
            &mut meter,
        )
        .unwrap();
    assert!(refused(demote_of(&types, s)));

    // Exactly the cap lowers, one more leaf does not; vector lanes count singly.
    let at_cap = types
        .intern(
            AirType::Array {
                elem: i32t,
                len: MAX_SCALARS,
            },
            &mut meter,
        )
        .unwrap();
    assert!(demote_of(&types, at_cap).is_ok());
    let over = types
        .intern(
            AirType::Array {
                elem: i32t,
                len: MAX_SCALARS + 1,
            },
            &mut meter,
        )
        .unwrap();
    assert!(refused(demote_of(&types, over)));
    let v4 = types
        .intern(AirType::Vector { elem: i32t, len: 4 }, &mut meter)
        .unwrap();
    let v4s = types
        .intern(
            AirType::Array {
                elem: v4,
                len: MAX_SCALARS / 4 + 1,
            },
            &mut meter,
        )
        .unwrap();
    assert!(refused(demote_of(&types, v4s)));

    // Nesting deeper than MAX_DEPTH is refused even with one leaf.
    let mut deep = i32t;
    for _ in 0..=MAX_DEPTH {
        deep = types
            .intern(AirType::Array { elem: deep, len: 1 }, &mut meter)
            .unwrap();
    }
    assert!(refused(demote_of(&types, deep)));
}

mod atan2;
mod bicubic;
mod blend16;
mod blend8;
mod blend_packed;
mod color_write;
mod depth16;
mod divide;
mod fast;
mod float_texel;
mod fused;
mod half_texel;
mod libm;
mod libm_exp;
mod narrow;
mod norm_write;
mod round_texel;
mod snapshot_codes;
mod specials;
mod texel_round;
mod wide;

/// `air.convert` from an integer to `bool` is C++'s `x != 0`, lane by lane, at every width and
/// whatever the source's sign: the values that differ from the low bit (2, 0x80, the top bit) are
/// true.
#[test]
fn integer_to_bool_convert_is_nonzero() {
    for bits in [8u8, 16, 32, 64] {
        let top = 1u64 << (bits - 1);
        let vals = [0, 1, 2, 3, 0x80 & mask(bits), top, mask(bits), top | 2];
        for lanes in [1u8, 4] {
            let (ff, tf) = (vec(int(bits), lanes), vec(boolean(), lanes));
            let cases: Vec<Case> = vals
                .chunks(usize::from(lanes))
                .map(|c| vec![c.to_vec()])
                .collect();
            check(
                &format!("convert i{bits} x{lanes} to bool"),
                &[ff],
                tf,
                &|fb, x| math::convert(fb, ConvertKind::IntToBool, ff, tf, x[0], NanTest::IsNan),
                &cases,
                &|c| Some(c[0].iter().map(|v| u64::from(*v != 0)).collect()),
                false,
            );
        }
    }
}

/// `air.convert` from a `float` or `half` to `bool` is Metal's reading of A1's cases
/// `float-to-bool-at-every-special-value` (099ba79b) and its `-fno-fast-math` twin (9a824047),
/// each lane of which is one of these patterns: false exactly for the zeros and the `float`
/// denormals, true for everything else (NaNs, infinities, half denormals, values in (0, 1)).
#[test]
fn float_to_bool_convert_is_metals_reading() {
    let floats: [u32; 40] = [
        0x0000_0000,
        0x8000_0000,
        0x0000_0001,
        0x8000_0001,
        0x007F_FFFF,
        0x807F_FFFF,
        0x0080_0000,
        0x8080_0000,
        0x3E80_0000,
        0x3F00_0000,
        0x3F40_0000,
        0x3F7F_FFFF,
        0x3F80_0000,
        0x3FC0_0000,
        0x4000_0000,
        0x4040_0000,
        0xBF00_0000,
        0xBF80_0000,
        0xC000_0000,
        0xC040_0000,
        0x437F_0000,
        0x4380_0000,
        0x4F00_0000,
        0x4F80_0000,
        0x7149_F2CA,
        0x7F7F_FFFF,
        0xFF7F_FFFF,
        0xCF80_0000,
        0x7F80_0000,
        0xFF80_0000,
        0x7FC0_0000,
        0xFFC0_0000,
        0x7F80_0001,
        0xFF80_0001,
        0x7FFF_FFFF,
        0xFFFF_FFFF,
        0x3DCC_CCCD,
        0xBDCC_CCCD,
        0x3F80_0001,
        0xBF7F_FFFF,
    ];
    let halves: [u16; 40] = [
        0x0000, 0x8000, 0x0001, 0x8001, 0x03FF, 0x83FF, 0x0400, 0x8400, 0x3400, 0x3800, 0x3A00,
        0x3BFF, 0x3C00, 0x3E00, 0x4000, 0x4200, 0xB800, 0xBC00, 0xC000, 0xC200, 0x5BF8, 0x5C00,
        0x7800, 0x7BFF, 0x6800, 0x7BFE, 0xFBFF, 0xF800, 0x7C00, 0xFC00, 0x7E00, 0xFE00, 0x7C01,
        0xFC01, 0x7FFF, 0xFFFF, 0x2E66, 0xAE66, 0x3C01, 0xBBFF,
    ];
    // Metal: zeros and `float` denormals are false.
    let f32_true = |b: u64| b & 0x7F80_0000 != 0;
    let half_true = |b: u64| b & 0x7FFF != 0;
    for (from, vals, truth) in [
        (
            float(32),
            floats.iter().map(|b| u64::from(*b)).collect::<Vec<_>>(),
            &f32_true as &dyn Fn(u64) -> bool,
        ),
        (
            float(16),
            halves.iter().map(|b| u64::from(*b)).collect(),
            &half_true,
        ),
    ] {
        for lanes in [1u8, 4] {
            let (ff, tf) = (vec(from, lanes), vec(boolean(), lanes));
            let cases: Vec<Case> = vals
                .chunks(usize::from(lanes))
                .map(|c| vec![c.to_vec()])
                .collect();
            check(
                &format!("convert f{} x{lanes} to bool", from.bits()),
                &[ff],
                tf,
                &|fb, x| math::convert(fb, ConvertKind::ToBool, ff, tf, x[0], NanTest::IsNan),
                &cases,
                &|c| Some(c[0].iter().map(|b| u64::from(truth(*b))).collect()),
                false,
            );
        }
    }
}

/// Metal's `fmin`, `fmax`, `min3`, `max3` and `clamp` on `float` and `half` take the device's
/// `NMin`, `NMax` and `NClamp` exactly where the profile takes `DriverDeviation::MetalMinMax` (the
/// MoltenVK captures alone); `median3`, `llvm.minnum`'s own form, and `bfloat` and `double` lanes
/// never do.
#[test]
fn the_minimum_and_maximum_family_follows_the_profile() {
    use crate::spirv::testing::is_moltenvk_capture;
    let family = [
        FloatFn::Min,
        FloatFn::Max,
        FloatFn::Clamp,
        FloatFn::Min3,
        FloatFn::Max3,
        FloatFn::Saturate,
    ];
    let mut taken = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        let moltenvk = is_moltenvk_capture(&stem);
        for op in family {
            for f in [float(32), float(16), vec(float(32), 4), vec(float(16), 3)] {
                assert_eq!(
                    super::device_min_max(&p, op, f),
                    moltenvk,
                    "{stem}: {op:?} on {f:?}"
                );
            }
            for f in [bfloat(), float(64)] {
                assert!(!super::device_min_max(&p, op, f), "{stem}: {op:?} on {f:?}");
            }
        }
        for op in [FloatFn::Median3, FloatFn::MinNum, FloatFn::MaxNum] {
            assert!(!super::device_min_max(&p, op, float(32)), "{stem}: {op:?}");
        }
        taken += usize::from(moltenvk);
    }
    assert!(taken >= 2, "the MoltenVK captures take it");
}

/// Where the library says the device's `NMin`, `NMax` and `NClamp` are Metal's
/// (`Lib::with_min_max`), each of the family is those instructions alone, with no NaN test;
/// elsewhere it is `math::metal_min_max`'s picks, with NaN tests and no such instruction.
#[test]
fn the_minimum_and_maximum_family_takes_the_device_instructions_where_measured() {
    // SPIR-V `OpExtInst`, `OpIsNan`; GLSL.std.450 `NMin`, `NMax`, `NClamp`.
    const OP_EXT_INST: u32 = 12;
    const OP_IS_NAN: u32 = 156;
    const DEVICE: [u32; 3] = [79, 80, 81];
    let counts = |op: FloatFn, f: Form, device: bool| -> (usize, usize) {
        let m = kernel(&vec![f; op.arity()], f, &|fb, a| {
            math::float_fn(
                fb,
                op,
                f,
                Contraction::Forbidden,
                NanTest::IsNan,
                a,
                &Lib::instructions().with_min_max(device),
            )
        })
        .expect("kernel");
        let words = crate::spirv::serialize::serialize(m).expect("words").words;
        let (mut ext, mut nan) = (0, 0);
        let mut i = 5;
        while i < words.len() {
            let (n, code) = ((words[i] >> 16) as usize, words[i] & 0xFFFF);
            if code == OP_EXT_INST && words.get(i + 4).is_some_and(|g| DEVICE.contains(g)) {
                ext += 1;
            }
            if code == OP_IS_NAN {
                nan += 1;
            }
            i += n.max(1);
        }
        (ext, nan)
    };
    for op in [
        FloatFn::Min,
        FloatFn::Max,
        FloatFn::Clamp,
        FloatFn::Min3,
        FloatFn::Max3,
    ] {
        for f in [float(32), float(16), vec(float(32), 4)] {
            let (ext, nan) = counts(op, f, true);
            let want = if matches!(op, FloatFn::Min3 | FloatFn::Max3) {
                2
            } else {
                1
            };
            assert_eq!((ext, nan), (want, 0), "{op:?} on {f:?}, the device's");
            let (ext, nan) = counts(op, f, false);
            assert_eq!(ext, 0, "{op:?} on {f:?}, picked");
            assert!(nan > 0, "{op:?} on {f:?}, picked: no NaN test");
        }
    }
}
