//! Metal's extended-range conversions run through the interpreter: every code read, and the
//! writes around every rounding boundary, against the exact rule.

use super::blend::{blend, Arith, Atom, BlendMath, Node, Summand, Term, FLOAT_SOURCE_RANGE};
use super::blend_color::{
    constant_words, COLOR_HULL, PAST_DIRECT_BOUND, SRGB_DIRECT_BOUND, SRGB_DIRECT_TOP,
};
use super::{
    declare, declare_padded, decode, encode, encode_keeping, tile_read, tile_value, Blend,
    SOURCE_RANGE, SRGB_SOURCE_RANGE,
};
use crate::error::{Error, ExtendedRangeBlend, Feature, Needs};
use crate::lower::arith::{FloatFmt, Form, Held};
use crate::lower::stage_io::extended_color::bfloat_output;
use crate::metal::format::extended_srgb::{SRGB_DECODE_STEPS, SRGB_ENCODE_FIRST};
use crate::metal::format::padded_srgb::{
    R10X6_SRGB_DECODE_STEPS, R10X6_SRGB_ENCODE_FIRST, RG10X12_SRGB_WRITE_CODES,
};
use crate::metal::format::ExtendedRange;
use crate::metal::{
    BlendFactor, BlendOperation, ColorWriteMask, PixelFormat, TextureWriteRoundingMode,
};
use crate::pipeline::lower_inputs::{BlendEquation, ExtendedRangeColor};
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, TypeId, Value};
use crate::spirv::interp::{self, Buffer, Config, Resources};
use crate::spirv::ir::{
    natural_layout, BinOp, ConvOp, ExecMode, ExecModel, IoDir, LayoutRule, MemAccess, StorageClass,
    Type, TypedModule,
};
use crate::spirv::testing::sharded_tests;
use metal2vulkan_devtools::xr_blends;

const FORMS: [ExtendedRange; 4] = [
    ExtendedRange::Bgr10 { srgb: false },
    ExtendedRange::Bgr10 { srgb: true },
    ExtendedRange::Bgra10 { srgb: false },
    ExtendedRange::Bgra10 { srgb: true },
];

/// [`FORMS`] and the two-plane forms.
pub(crate) const EVERY_FORM: [ExtendedRange; 6] = [
    ExtendedRange::Bgr10 { srgb: false },
    ExtendedRange::Bgr10 { srgb: true },
    ExtendedRange::Bgra10 { srgb: false },
    ExtendedRange::Bgra10 { srgb: true },
    ExtendedRange::Planes { srgb: false },
    ExtendedRange::Planes { srgb: true },
];

/// What a test kernel computes of each invocation's words.
#[derive(Clone, Copy)]
enum Op {
    /// The decode of a raw texel.
    Decode,
    /// The decode of a raw texel to `half` values (as `f32` bits).
    DecodeHalf,
    /// The encode of a `vec4` of `f32` bits.
    Encode,
    /// The tile value of a raw texel.
    Tile,
    /// The blend under an equation of a write of the channels of a mask (`blend::blend`) of an
    /// output of a format ([`Blend::for_output`]) of a source (words 4 to 7, `f32` bits: `half`
    /// values but where the blend is a `float4` output's own model, which takes it unrounded)
    /// over the tile value of a raw texel (words 0 to 3), the blend color's rows words 8 to 15
    /// ([`constant_words`]), in a function of a math, the second source words 16 to 19 (`f32` bits
    /// of `half` values).
    Blend(BlendEquation, ColorWriteMask, FloatFmt, BlendMath),
    /// That blend's texel ([`encode_keeping`]), the channels the mask leaves out kept from the
    /// raw texel.
    BlendTexel(BlendEquation, ColorWriteMask, FloatFmt, BlendMath),
    /// Metal's read of a tile holding a `vec4` of `f32` bits ([`tile_read`]).
    TileRead,
    /// A render pass's store of a fragment function's output (`vec4` of `f32` bits, of the
    /// format, `half` values held as `f32`) to a bfloat attachment of a pipeline at
    /// `textureWriteRoundingMode` `NearestEven` where the flag is set
    /// (`stage_io::extended_color::bfloat_output`).
    Output(bool, FloatFmt),
}

impl Op {
    /// The words an invocation reads.
    fn words(self) -> u32 {
        match self {
            Op::Blend(..) | Op::BlendTexel(..) => 20,
            _ => 4,
        }
    }
}

fn slot(index: u32) -> ResourceSlot {
    ResourceSlot {
        stage: Stage::Kernel,
        kind: ResourceKind::Buffer,
        index,
    }
}

fn storage_buffer(b: &mut ModuleBuilder, index: u32, elem: TypeId) -> BResult<GlobalId> {
    let rta = b.ty(Type::RuntimeArray {
        elem,
        stride: Some(4),
    })?;
    let layout = natural_layout(b.types(), LayoutRule::Std430, true, &[rta])?;
    let block = b.ty(Type::Struct {
        members: [rta].into(),
        layout: Some(layout),
    })?;
    b.resource(slot(index), StorageClass::StorageBuffer, block)
}

/// Word `k` of invocation `i`'s `n` words in `buf`.
fn word(
    f: &mut FunctionBuilder<'_>,
    buf: GlobalId,
    (i, n): (Value, u32),
    k: u32,
) -> BResult<Value> {
    let n = f.u32_const(n)?;
    let n = f.module().const_value(n)?;
    let at = f.binary(BinOp::IMul, i, n)?;
    let k = f.u32_const(k)?;
    let k = f.module().const_value(k)?;
    let at = f.binary(BinOp::IAdd, at, k)?;
    let zero = f.u32_const(0)?;
    let zero = f.module().const_value(zero)?;
    let base = f.module().global_value(buf)?;
    f.access_chain(base, &[zero, at])
}

/// A kernel whose invocation `i` reads `op`'s words from buffer 0 and writes four to buffer 1.
fn kernel(form: ExtendedRange, op: Op) -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let f32_ = b.ty(Type::Float { width: 32 })?;
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
    let uvec4 = b.ty(Type::Vector { elem: u32_, len: 4 })?;
    let vec4 = b.ty(Type::Vector { elem: f32_, len: 4 })?;
    let input = storage_buffer(&mut b, 0, u32_)?;
    let output = storage_buffer(&mut b, 1, u32_)?;
    let table = match form.srgb() {
        true => Some(declare(&mut b)?),
        false => declare_padded(&mut b, form)?,
    };
    let id = b.builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let gid = f.module().global_value(id)?;
    let gid = f.load(gid, MemAccess::default())?;
    let i = f.composite_extract(gid, &[0])?;
    let mut lanes = Vec::with_capacity(8);
    for k in 0..op.words() {
        let at = word(&mut f, input, (i, op.words()), k)?;
        lanes.push(f.load(at, MemAccess::default())?);
    }
    let x = f.composite_construct(uvec4, &lanes[..4])?;
    let out = match op {
        Op::Encode => {
            let v = f.convert(ConvOp::Bitcast, x, vec4)?;
            encode(&mut f, form, v, table)?
        }
        Op::Decode => {
            let v = decode(&mut f, form, x, table, false)?;
            f.convert(ConvOp::Bitcast, v, uvec4)?
        }
        Op::DecodeHalf => {
            let v = decode(&mut f, form, x, table, true)?;
            f.convert(ConvOp::Bitcast, v, uvec4)?
        }
        Op::Tile => {
            let v = tile_value(&mut f, form, x, table)?;
            f.convert(ConvOp::Bitcast, v, uvec4)?
        }
        Op::TileRead => {
            let held = f.convert(ConvOp::Bitcast, x, vec4)?;
            let v = tile_read(&mut f, form, held)?;
            f.convert(ConvOp::Bitcast, v, uvec4)?
        }
        Op::Output(nearest_even, fmt) => {
            let v = f.convert(ConvOp::Bitcast, x, vec4)?;
            let rounding = match nearest_even {
                true => TextureWriteRoundingMode::NearestEven,
                false => TextureWriteRoundingMode::TowardZero,
            };
            let e = ExtendedRangeColor::new(form, false).with_rounding(rounding);
            let lanes = Form {
                held: Held::Float(fmt),
                lanes: 4,
            };
            bfloat_output(&mut f, e, lanes, v)?
        }
        Op::Blend(eq, mask, output, math) | Op::BlendTexel(eq, mask, output, math) => {
            let b =
                Blend::for_output(&eq, form, mask, (Some(output), math)).expect("a modelled blend");
            let s = f.composite_construct(uvec4, &lanes[4..8])?;
            let s = f.convert(ConvOp::Bitcast, s, vec4)?;
            let c = f.composite_construct(uvec4, &lanes[8..12])?;
            let m = f.composite_construct(uvec4, &lanes[12..16])?;
            let d = tile_value(&mut f, form, x, table)?;
            let s1 = match b.reads_second() {
                true => {
                    let s1 = f.composite_construct(uvec4, &lanes[16..20])?;
                    Some(f.convert(ConvOp::Bitcast, s1, vec4)?)
                }
                false => None,
            };
            let v = blend(&mut f, form, &b, (s, s1, d), Some([c, m]))?;
            match op {
                Op::BlendTexel(..) => {
                    let kept = [
                        ColorWriteMask::RED,
                        ColorWriteMask::GREEN,
                        ColorWriteMask::BLUE,
                        ColorWriteMask::ALPHA,
                    ]
                    .map(|c| !mask.contains(c));
                    encode_keeping(&mut f, form, v, table, (x, kept))?
                }
                _ => f.convert(ConvOp::Bitcast, v, uvec4)?,
            }
        }
    };
    for k in 0..4u32 {
        let c = f.composite_extract(out, &[k])?;
        let at = word(&mut f, output, (i, 4), k)?;
        f.store(at, c, MemAccess::default())?;
    }
    f.ret(None)?;
    let main = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        main,
        vec![ExecMode::LocalSize { x: 1, y: 1, z: 1 }],
    )?;
    b.finalize()
}

fn run<const N: usize>(form: ExtendedRange, op: Op, lanes: &[[u32; N]]) -> Vec<[u32; 4]> {
    assert_eq!(N as u32, op.words());
    let m = kernel(form, op).expect("kernel");
    let bytes: Vec<u8> = lanes
        .iter()
        .flatten()
        .flat_map(|w| w.to_le_bytes())
        .collect();
    let res = Resources {
        buffers: vec![
            Buffer {
                slot: Some(slot(0)),
                address: None,
                bytes,
            },
            Buffer {
                slot: Some(slot(1)),
                address: None,
                bytes: vec![0xAB; lanes.len() * 16],
            },
        ],
        ..Resources::default()
    };
    let cfg = Config {
        workgroups: [u32::try_from(lanes.len()).expect("lanes"), 1, 1],
        ..Config::default()
    };
    let (out, _) = interp::run_with_images(&m, &cfg, res, Vec::new()).expect("run");
    out.buffers[1]
        .as_chunks::<16>()
        .0
        .iter()
        .map(|c| {
            std::array::from_fn(|k| {
                u32::from_le_bytes(c[4 * k..4 * k + 4].try_into().expect("word"))
            })
        })
        .collect()
}

/// `n / 510` correctly rounded to `f32` (`|n| <= 639`): the `f64` quotient's neighbours compared
/// exactly (each `f32` times 510 is exact in `f64`).
fn ratio(n: i32) -> f32 {
    let q = f64::from(n) / 510.0;
    let y = q as f32;
    let candidates = [
        y,
        f32::from_bits(y.to_bits().wrapping_sub(1)),
        f32::from_bits(y.to_bits() + 1),
    ];
    let err = |c: f32| (f64::from(c) * 510.0 - f64::from(n)).abs();
    let best = candidates
        .into_iter()
        .filter(|c| c.is_finite())
        .min_by(|a, b| err(*a).partial_cmp(&err(*b)).expect("finite"))
        .expect("a candidate");
    if n == 0 {
        0.0
    } else {
        best
    }
}

/// Metal's value of code `c` of a color channel.
fn color_value(c: u32, srgb: bool) -> f32 {
    let n = c as i32 - 384;
    match srgb {
        false => ratio(n),
        true => {
            let s = f32::from(SRGB_DECODE_STEPS[n.unsigned_abs() as usize]) / 8192.0;
            if n < 0 {
                -s
            } else {
                s
            }
        }
    }
}

/// The raw texel of red, green, blue, and alpha codes `rgba` in `form` (a two-plane form's alpha
/// its byte), with `junk` in the bits Metal does not read.
pub(crate) fn raw(form: ExtendedRange, rgba: [u32; 4], junk: u32) -> [u32; 4] {
    match form {
        ExtendedRange::Bgr10 { .. } => [
            rgba[2] | rgba[1] << 10 | rgba[0] << 20 | (junk & 3) << 30,
            0,
            0,
            0,
        ],
        ExtendedRange::Planes { .. } => [
            rgba[2] | rgba[1] << 10 | rgba[0] << 20 | (junk & 3) << 30,
            (rgba[3] & 0xFF) | (junk & 0xFF) << 8,
            0,
            0,
        ],
        ExtendedRange::R10X6 | ExtendedRange::R10X6Srgb => [rgba[0] << 6 | (junk & 0x3F), 0, 0, 0],
        ExtendedRange::Rg10X12 | ExtendedRange::Rg10X12Srgb => [
            rgba[0] << 6 | (junk & 0x3F),
            rgba[1] << 6 | (junk >> 6 & 0x3F),
            0,
            0,
        ],
        _ => {
            let f = |c: u32| c << 6 | (junk & 0x3F);
            [f(rgba[2]), f(rgba[1]), f(rgba[0]), f(rgba[3])]
        }
    }
}

/// Every code of every channel reads Metal's value, bit for bit, whatever the bits Metal does not
/// read hold: `(c - 384) / 510` correctly rounded, an `_sRGB` color channel its table's step over
/// 8192, alpha clamped to `[0, 1]` (1 where the format has none).
#[test]
fn every_code_reads_metals_value() {
    for form in FORMS {
        let srgb = form.srgb();
        let lanes: Vec<[u32; 4]> = (0..1024u32)
            .map(|c| raw(form, [c, 1023 - c, (c * 7) % 1024, c], c * 13))
            .collect();
        let got = run(form, Op::Decode, &lanes);
        for (c, g) in (0..1024u32).zip(&got) {
            let want = [
                color_value(c, srgb),
                color_value(1023 - c, srgb),
                color_value((c * 7) % 1024, srgb),
                match form {
                    ExtendedRange::Bgr10 { .. } => 1.0,
                    _ => ratio(c.clamp(384, 894) as i32 - 384),
                },
            ];
            assert_eq!(*g, want.map(f32::to_bits), "{form:?} code {c}");
        }
    }
}

/// `x` rounded to the nearest integer, ties to even (`x` exact in `f64`).
fn round_even(x: f64) -> f64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
        r - x.signum()
    } else {
        r
    }
}

/// The code Metal writes for color value `v` (alpha where `alpha`).
fn code(v: f32, srgb: bool, alpha: bool) -> u32 {
    let v = if v.is_nan() { 0.0 } else { v };
    let v = if alpha { v.clamp(0.0, 1.0) } else { v };
    let neg = v < 0.0;
    if srgb && !alpha {
        let q = round_even(f64::from(v.abs().min(2.0)) * 8192.0) as u32;
        let n = (0..640)
            .rev()
            .find(|&n| u32::from(SRGB_ENCODE_FIRST[n]) <= q)
            .expect("offset 0") as u32;
        return if neg { 384 - n.min(384) } else { 384 + n };
    }
    let k = round_even(f64::from(v.abs().min(1.5)) * 510.0) as i32;
    (384 + if neg { -k } else { k }).clamp(0, 1023) as u32
}

/// A write stores Metal's code around every boundary: each code's value and its neighbours, the
/// `f32`s either side of each midpoint (each an exact tie is not, the exact product rounding to
/// nearest even), each `_sRGB` index's half-point and its neighbours, and the specials; bits Metal
/// does not read are written as it writes them.
#[test]
fn writes_round_at_metals_boundaries() {
    let specials = [
        0.0f32,
        -0.0,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        1e30,
        -1e30,
        -0.75,
        0.25,
        1.25,
        1.5,
        -1.5,
        2.0,
        -2.0,
        1e-40,
    ];
    for form in FORMS {
        let srgb = form.srgb();
        let mut xs: Vec<f32> = specials.to_vec();
        for c in 0..1023u32 {
            let (a, b) = (color_value(c, srgb), color_value(c + 1, srgb));
            let mid = ((f64::from(a) + f64::from(b)) / 2.0) as f32;
            xs.extend([
                a,
                mid,
                f32::from_bits(mid.to_bits() + 1),
                f32::from_bits(mid.to_bits().wrapping_sub(1)),
            ]);
        }
        if srgb {
            for first in SRGB_ENCODE_FIRST.iter().skip(1) {
                let h = (f32::from(*first) - 0.5) / 8192.0;
                for x in [h, -h] {
                    xs.extend([
                        x,
                        f32::from_bits(x.to_bits() + 1),
                        f32::from_bits(x.to_bits() - 1),
                    ]);
                }
            }
        }
        let lanes: Vec<[u32; 4]> = xs
            .iter()
            .map(|x| [x.to_bits(), (-x).to_bits(), x.to_bits(), x.to_bits()])
            .collect();
        let got = run(form, Op::Encode, &lanes);
        for (x, g) in xs.iter().zip(&got) {
            let rgba = [
                code(*x, srgb, false),
                code(-x, srgb, false),
                code(*x, srgb, false),
                code(*x, srgb, true),
            ];
            let want = match form {
                ExtendedRange::Bgr10 { .. } => {
                    [rgba[2] | rgba[1] << 10 | rgba[0] << 20 | 3 << 30, 0, 0, 0]
                }
                _ => raw(form, rgba, 0),
            };
            assert_eq!(*g, want, "{form:?} {x:e} ({:#010x})", x.to_bits());
        }
    }
}

/// Metal's own answers on an Apple M3 Max and an M4: writes at its measured ties and
/// thresholds, and reads of its measured values.
#[test]
fn metals_measured_answers_hold() {
    // Linear: -0.75, 0.25 and 1.25 are exact ties, written as codes 2, 512 and 1022; code 0
    // reads -384/510 (0xbf40c0c1), code 512 0.25098 (0x3e808081).
    let linear = ExtendedRange::Bgra10 { srgb: false };
    let w = run(
        linear,
        Op::Encode,
        &[[
            (-0.75f32).to_bits(),
            0.25f32.to_bits(),
            1.25f32.to_bits(),
            0,
        ]],
    );
    assert_eq!(w[0], [1022 << 6, 512 << 6, 2 << 6, 384 << 6]);
    let r = run(linear, Op::Decode, &[raw(linear, [0, 512, 1023, 1023], 0)]);
    assert_eq!(r[0], [0xbf40_c0c1, 0x3e80_8081, 0x3fa0_6060, 0x3f80_0000]);
    // sRGB: code 0 reads -0.527099609375 (0xbf06f000), code 1023 1.6749267578125 (0x3fd66400);
    // 0.56634521484375 is the least float writing code 781, -0.13214111328125 the least writing
    // code 181 (bisection thresholds).
    let srgb = ExtendedRange::Bgr10 { srgb: true };
    let r = run(srgb, Op::Decode, &[raw(srgb, [0, 1023, 384, 0], 0)]);
    assert_eq!(r[0][..3], [0xbf06_f000, 0x3fd6_6400, 0]);
    let below = |x: f32| f32::from_bits(x.to_bits() - 1);
    let w = run(
        srgb,
        Op::Encode,
        &[[
            0.566_345_2f32.to_bits(),
            below(0.566_345_2).to_bits(),
            (-0.132_141_11f32).to_bits(),
            0,
        ]],
    );
    let codes = [
        (w[0][0] >> 20) & 0x3FF,
        (w[0][0] >> 10) & 0x3FF,
        w[0][0] & 0x3FF,
    ];
    assert_eq!(codes, [781, 780, 181]);
}

/// `x` (exact in `f64`) rounded to `half`: to nearest, ties away from zero where `away`, else to
/// even; held as an `f32`. No value here reaches the infinities.
fn to_half(x: f64, away: bool) -> f32 {
    if x == 0.0 {
        return x as f32;
    }
    let e = ((x.abs().to_bits() >> 52) & 0x7FF) as i32 - 1023;
    // A `half`'s spacing at `x`: `2^(e - 10)`, at least the denormals' `2^-24`.
    let ulp = 2f64.powi(e.max(-14) - 10);
    let q = x / ulp;
    let r = if away { q.round() } else { q.round_ties_even() };
    let y = r * ulp;
    assert!(y.abs() < 65520.0, "{x} rounds past the largest half");
    y as f32
}

/// The `half` value of bits `h`, as an `f32` (a NaN quiet).
fn half(h: u16) -> f32 {
    crate::float16::f16_bits_to_f32(h)
}

/// Metal's tile value of a texel of `form` whose codes are `rgba` (a two-plane form's alpha its
/// byte), as the module docs of `tile_value` state the rule.
pub(crate) fn tile_model(form: ExtendedRange, rgba: [u32; 4]) -> [f32; 4] {
    let srgb = form.srgb();
    let by_code = !srgb || matches!(form, ExtendedRange::Bgra10 { .. });
    let (lo, hi) = ((-384.0f64 / 510.0) as f32, (638.0f64 / 510.0) as f32);
    let color = |c: u32| match by_code {
        true => color_value(c.min(1022), srgb),
        false => color_value(c, srgb).clamp(lo, hi),
    };
    let alpha = match form {
        ExtendedRange::Bgr10 { .. } => 1.0,
        ExtendedRange::Bgra10 { .. } => ratio(rgba[3].clamp(384, 894) as i32 - 384),
        ExtendedRange::Planes { .. } => (f64::from(rgba[3]) / 255.0) as f32,
        ExtendedRange::R10X6
        | ExtendedRange::Rg10X12
        | ExtendedRange::R10X6Srgb
        | ExtendedRange::Rg10X12Srgb => {
            panic!("{form:?}: a padded form's tile value is not measured")
        }
        ExtendedRange::R16Bfloat | ExtendedRange::Rg16Bfloat | ExtendedRange::Rgba16Bfloat => {
            panic!("{form:?}: a bfloat form's tile value is not measured")
        }
    };
    [color(rgba[0]), color(rgba[1]), color(rgba[2]), alpha].map(|v| to_half(f64::from(v), true))
}

/// Every code of every channel of the six forms reads Metal's tile value, whatever the bits Metal
/// does not read hold: a linear or `BGRA10_XR_sRGB` code above 1022 as 1022, the other `_sRGB`
/// forms' values clamped to the linear values of codes 0 and 1022, and each lane rounded to `half`,
/// ties away from zero. The ends are the blend's source ranges.
#[test]
fn every_code_reads_metals_tile_value() {
    for form in EVERY_FORM {
        let alpha = |c: u32| match form {
            ExtendedRange::Planes { .. } => c % 256,
            _ => c,
        };
        let codes = |c: u32| [c, 1023 - c, (c * 7) % 1024, alpha(c)];
        let lanes: Vec<[u32; 4]> = (0..1024u32).map(|c| raw(form, codes(c), c * 13)).collect();
        let got = run(form, Op::Tile, &lanes);
        for (c, g) in (0..1024u32).zip(&got) {
            let want = tile_model(form, codes(c));
            assert_eq!(*g, want.map(f32::to_bits), "{form:?} code {c}");
        }
        // The other `_sRGB` forms' tiles end inside their source range.
        let (lo, hi) = match form {
            ExtendedRange::Bgra10 { srgb: true } => SRGB_SOURCE_RANGE,
            _ if form.srgb() => continue,
            _ => SOURCE_RANGE,
        };
        // Codes 0, 1022 and 1023: the source range's ends.
        assert_eq!(f32::from_bits(got[0][0]), lo, "{form:?}");
        assert_eq!(f32::from_bits(got[0][1]), hi, "{form:?}");
        assert_eq!(f32::from_bits(got[1][1]), hi, "{form:?}");
    }
}

/// The blend `eq` (added, source factor zero or one and destination factor zero, one, or one minus
/// the source's alpha) of `half` source `s` over tile value `d` into `form`, as our probes measured
/// it: the source clamped, a NaN color to the range's low end and a NaN alpha
/// to 0, the factor `1 - a` rounded to `half`, nearest even, and `d * factor + s` exactly (at most
/// 50 significant bits, exact in `f64`), `s` 0 under source factor zero, rounded once to `half`,
/// nearest even.
fn d639_model(form: ExtendedRange, eq: &BlendEquation, s: [f32; 4], d: [f32; 4]) -> [f32; 4] {
    let (lo, hi) = match form {
        ExtendedRange::Bgra10 { srgb: true } => SRGB_SOURCE_RANGE,
        _ => SOURCE_RANGE,
    };
    let s: [f32; 4] = std::array::from_fn(|k| match (k, s[k].is_nan()) {
        (3, true) => 0.0,
        (3, false) => s[k].clamp(0.0, 1.0),
        (_, true) => lo,
        (_, false) => s[k].clamp(lo, hi),
    });
    let minus = to_half(1.0 - f64::from(s[3]), false);
    let factor = |f: BlendFactor| match f {
        BlendFactor::Zero => 0.0,
        BlendFactor::One => 1.0,
        BlendFactor::OneMinusSourceAlpha => minus,
        f => panic!("{f:?} is not a factor of these blends"),
    };
    std::array::from_fn(|k| {
        let (fs, fd) = match k {
            3 => (eq.source_alpha, eq.destination_alpha),
            _ => (eq.source_rgb, eq.destination_rgb),
        };
        let term = f64::from(factor(fs)) * f64::from(s[k]);
        to_half(f64::from(d[k]) * f64::from(factor(fd)) + term, false)
    })
}

/// SplitMix64.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u64) -> u32 {
        (self.next() % n) as u32
    }

    /// A random `half`'s bits: a quarter any pattern (NaNs, the infinities, denormals, and values
    /// far past every range), the rest of magnitude below 2 (exponent fields 0 to 15, denormals
    /// among them), of either sign.
    fn half_bits(&mut self) -> u16 {
        if self.below(4) == 0 {
            return self.below(1 << 16) as u16;
        }
        let sign = (self.below(2) << 15) as u16;
        sign | (self.below(16) << 10) as u16 | self.below(1 << 10) as u16
    }

    /// A random `float`: a quarter any pattern, the rest of magnitude below 2 (exponents down to
    /// `2^-30`, and the denormal `half`s' among them), of either sign.
    fn float(&mut self) -> f32 {
        if self.below(4) == 0 {
            return f32::from_bits(self.next() as u32);
        }
        let sign = self.below(2) << 31;
        let exponent = (127 - 30 + self.below(31)) << 23;
        f32::from_bits(sign | exponent | self.below(1 << 23))
    }
}

/// The blend of color factors and operation `color` and alpha's `alpha`.
const fn equation(
    color: (BlendFactor, BlendFactor, BlendOperation),
    alpha: (BlendFactor, BlendFactor, BlendOperation),
) -> BlendEquation {
    BlendEquation {
        source_rgb: color.0,
        destination_rgb: color.1,
        rgb_operation: color.2,
        source_alpha: alpha.0,
        destination_alpha: alpha.1,
        alpha_operation: alpha.2,
    }
}

/// The blends checked: our probe's three (source factor one, added, destination zero, one, or one
/// minus the source's alpha for color and alpha alike), two that mix them, and four of source
/// factor zero, the first the consumer's (color zero over one minus the source's
/// alpha).
const D639_BLENDS: [BlendEquation; 9] = {
    use BlendFactor::{One, OneMinusSourceAlpha as Msa, Zero};
    use BlendOperation::Add;
    [
        equation((One, Zero, Add), (One, Zero, Add)),
        equation((One, One, Add), (One, One, Add)),
        equation((One, Msa, Add), (One, Msa, Add)),
        equation((One, Msa, Add), (One, One, Add)),
        equation((One, Zero, Add), (One, Msa, Add)),
        equation((Zero, Msa, Add), (Zero, Msa, Add)),
        equation((Zero, Msa, Add), (One, Msa, Add)),
        equation((Zero, One, Add), (Zero, Zero, Add)),
        equation((One, Zero, Add), (Zero, Msa, Add)),
    ]
};

/// Every channel.
const ALL: ColorWriteMask = ColorWriteMask::ALL;

/// The three linear forms.
const LINEAR: [ExtendedRange; 3] = [
    ExtendedRange::Bgra10 { srgb: false },
    ExtendedRange::Bgr10 { srgb: false },
    ExtendedRange::Planes { srgb: false },
];

/// `n` random invocations' words of a blend into `form`: a random texel, a source of `output`
/// (NaNs, the infinities, denormals and values past the ranges among them, the specials first),
/// and blend color `color`'s rows; and each one's codes and source.
fn blend_lanes(
    r: &mut Rng,
    form: ExtendedRange,
    n: usize,
    color: [f32; 4],
    output: FloatFmt,
) -> (Vec<[u32; 20]>, Vec<([u32; 4], [f32; 4])>) {
    const SPECIAL: [u16; 16] = [
        0x7E00, 0xFE00, 0x7C00, 0xFC00, 0x0000, 0x8000, 0x0001, 0x8001, 0x3C00, 0xBC00, 0x3D01,
        0x3D02, 0xBA06, 0xBA07, 0x3EAD, 0x3EAE,
    ];
    // A `float4` output's: the specials, the float range's ends and their neighbours, `half`
    // midpoints (1 + 2^-11, its neighbours), and a denormal `float`.
    let (flo, fhi) = FLOAT_SOURCE_RANGE;
    let float_special: [f32; 16] = [
        f32::NAN,
        -f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        0.0,
        -0.0,
        f32::from_bits(1),
        -f32::from_bits(1),
        flo,
        flo.next_down(),
        fhi,
        fhi.next_up(),
        1.0 + 2f32.powi(-11),
        (1.0 + 2f32.powi(-11)).next_up(),
        (1.0 + 2f32.powi(-11)).next_down(),
        -0.5 - 2f32.powi(-12),
    ];
    let rows = constant_words(color);
    let mut lanes = Vec::with_capacity(n);
    let mut inputs = Vec::with_capacity(n);
    for i in 0..n {
        let codes = [
            r.below(1024),
            r.below(1024),
            r.below(1024),
            match form {
                ExtendedRange::Planes { .. } => r.below(256),
                _ => r.below(1024),
            },
        ];
        let texel = raw(form, codes, r.below(1 << 16));
        let s: [f32; 4] = std::array::from_fn(|k| match output {
            FloatFmt::F32 => match float_special.get(i * 4 + k) {
                Some(&x) => x,
                None => r.float(),
            },
            _ => half(match SPECIAL.get(i * 4 + k) {
                Some(&h) => h,
                None => r.half_bits(),
            }),
        });
        let mut words = [0u32; 20];
        words[..4].copy_from_slice(&texel);
        for (w, x) in words[4..8].iter_mut().zip(s) {
            *w = x.to_bits();
        }
        words[8..16].copy_from_slice(&rows);
        // A second source of `half` values, NaNs and values past the ranges among them.
        for w in words[16..].iter_mut() {
            *w = half(r.half_bits()).to_bits();
        }
        lanes.push(words);
        inputs.push((codes, s));
    }
    (lanes, inputs)
}

/// A blend of the of a `half` source over a texel of each of the six forms computes the measured
/// value over random texels and sources, the general model's arithmetic giving each, in each lane
/// the format stores (`BGR10_XR` stores no alpha). A zero's sign is not observable: the encode
/// stores the zero code for either.
#[test]
fn d639_blends_compute_metals_value() {
    let mut r = Rng(0x5852_5232);
    for form in EVERY_FORM {
        let stored = match form {
            ExtendedRange::Bgr10 { .. } => 3,
            _ => 4,
        };
        for eq in D639_BLENDS {
            let (lanes, inputs) = blend_lanes(&mut r, form, 512, [0.0; 4], FloatFmt::Half);
            let got = run(
                form,
                Op::Blend(eq, ALL, FloatFmt::Half, BlendMath::Fast),
                &lanes,
            );
            for (g, (codes, s)) in got.iter().zip(&inputs) {
                let d = tile_model(form, *codes);
                let want = d639_model(form, &eq, *s, d);
                let lanes = g.map(f32::from_bits);
                assert!(
                    lanes.iter().zip(want).take(stored).all(|(g, w)| *g == w),
                    "{form:?} {eq:?}: source {s:?} over codes {codes:?}: {lanes:?}, the model's \
                     {want:?}"
                );
            }
        }
    }
}

/// `x` (exact in `f64`) rounded to `half`, nearest even.
fn h(x: f64) -> f64 {
    f64::from(to_half(x, false))
}

/// One lane's values: the clamped source's lane and alpha, the destination's, the blend color's,
/// and one minus the blend color's (each a `half`); and a `float4` output's lane unrounded,
/// clamped to its own range (a `float`).
#[derive(Clone, Copy)]
struct LaneValues {
    s: f64,
    sa: f64,
    d: f64,
    da: f64,
    c: f64,
    ca: f64,
    mc: f64,
    mca: f64,
    sf: f64,
    s1: f64,
    s1a: f64,
}

/// The exact sum `a + b` of two `f64`s rounded to `float`, nearest even: the rounded `f64` sum,
/// whose error (its two-sum's low part) decides only where that sum is a `float` midpoint (a
/// midpoint is on the `f64` grid, so a sum off it lies on the exact sum's side of it).
fn rn32_sum(a: f64, b: f64) -> f64 {
    let s = a + b;
    let v = s - a;
    let e = (a - (s - v)) + (b - v);
    let r = s as f32;
    if e == 0.0 || f64::from(r) == s {
        return f64::from(r);
    }
    let (lo, hi) = match f64::from(r) < s {
        true => (r, r.next_up()),
        false => (r.next_down(), r),
    };
    if s != (f64::from(lo) + f64::from(hi)) / 2.0 {
        return f64::from(r);
    }
    f64::from(if e > 0.0 { hi } else { lo })
}

/// `x` (exact in `f64`) rounded to `float`, nearest even.
fn rn32(x: f64) -> f64 {
    f64::from(x as f32)
}

/// Whether term `t` reads a `float4` output's lane unrounded ([`Atom::Sf`]).
fn raw_term(t: Term) -> bool {
    t.0 == Atom::Sf || t.1 == Node::Atom(Atom::Sf)
}

/// `h(p + a)`: exactly of `half` values, through `float` first where a value is a `float`
/// (`raw`: the product's operands, the addend), as the shader's multiply-add rounds it.
fn fma_value((raw_product, raw_addend): (bool, bool), p: f64, a: f64) -> f64 {
    match raw_product || raw_addend {
        true => h(rn32_sum(p, a)),
        false => h(p + a),
    }
}

fn atom_value(v: &LaneValues, a: Atom) -> f64 {
    match a {
        Atom::S => v.s,
        Atom::Sa => v.sa,
        Atom::D => v.d,
        Atom::Da => v.da,
        Atom::C => v.c,
        Atom::Ca => v.ca,
        Atom::Sf => v.sf,
        Atom::S1 => v.s1,
        Atom::S1a => v.s1a,
    }
}

fn node_value(v: &LaneValues, n: Node) -> f64 {
    match n {
        Node::Zero => 0.0,
        Node::One => 1.0,
        Node::Atom(a) => atom_value(v, a),
        Node::OneMinus(Atom::C) => v.mc,
        Node::OneMinus(Atom::Ca) => v.mca,
        // A `float`'s `1 - x` rounds to `float` first.
        Node::OneMinus(Atom::Sf) => h(rn32_sum(1.0, -v.sf)),
        Node::OneMinus(a) => h(1.0 - atom_value(v, a)),
        Node::Saturate => v.sa.min(h(1.0 - v.da)),
    }
}

/// An inner sum's value: one summand as it is, more added in order, each sum rounded (through
/// `float` where it adds an unrounded lane); and whether it is that lane unrounded (one summand).
fn inner_value(v: &LaneValues, summands: &[Summand]) -> (f64, bool) {
    let mut acc: Option<(f64, bool)> = None;
    for &(c, k) in summands {
        let x = f64::from(c) * k.map_or(1.0, |n| node_value(v, n));
        let raw = k == Some(Node::Atom(Atom::Sf));
        acc = Some(match acc {
            None => (x, raw),
            Some((a, raw_a)) => (fma_value((false, raw_a || raw), a, x), false),
        });
    }
    acc.unwrap_or((0.0, false))
}

/// The value of arithmetic `a` of one lane, exactly in `f64` (every product here, of two `half`s
/// or of a `float` and a `half` or another `float`, fits 53 bits, and so does every sum of
/// `half`s; a sum with a `float` rounds to `float` through [`rn32_sum`]), each rounding where
/// the module docs place it: a `float4` output's products and sums with its unrounded lane round
/// to `float` first.
fn arith_value(v: &LaneValues, a: &Arith) -> f64 {
    let term = |(sign, (at, f)): (i8, Term)| f64::from(sign) * atom_value(v, at) * node_value(v, f);
    match a {
        Arith::MinMax { max, source } => {
            let s = atom_value(v, *source);
            let m = if *max { s.max(v.d) } else { s.min(v.d) };
            match source {
                Atom::Sf => h(m),
                _ => m,
            }
        }
        Arith::Zero => 0.0,
        Arith::Term { sign, term: t } => match t.1 {
            Node::One if t.0 == Atom::Sf => h(f64::from(*sign) * v.sf),
            Node::One => f64::from(*sign) * atom_value(v, t.0),
            _ => h(rn32(term((*sign, *t)))),
        },
        Arith::Fused { product, addend } => fma_value(
            (raw_term(product.1), addend.1 .0 == Atom::Sf),
            term(*product),
            term(*addend),
        ),
        Arith::Lerp { y, z, inner } => {
            let (i, raw_inner) = inner_value(v, inner);
            let sf = Node::Atom(Atom::Sf);
            fma_value(
                (*z == sf || raw_inner, *y == sf),
                node_value(v, *z) * i,
                node_value(v, *y),
            )
        }
        Arith::Common {
            z,
            inner,
            round_inner,
        } => {
            let (i, _) = inner_value(v, inner);
            let i = if *round_inner { h(i) } else { i };
            h(rn32(node_value(v, *z) * i))
        }
        Arith::Contracted { rounded, fused } => {
            let r = h(rn32(term(*rounded)));
            fma_value((raw_term(fused.1), false), term(*fused), r)
        }
        Arith::Kept { kept, rounded } => {
            let r = h(rn32(term(*rounded)));
            h(rn32_sum(rn32(term(*kept)), r))
        }
        Arith::Rounded {
            source,
            destination,
        } => {
            let product = |t: (i8, Term)| match t.1 .1 {
                Node::One => f64::from(t.0) * atom_value(v, t.1 .0),
                _ => h(term(t)),
            };
            h(product(*source) + product(*destination))
        }
    }
}

/// The blended lanes of `b` of source `s` (a `half4` output's, or a `float4` one's under its own
/// model, [`Blend::float`]) over tile value `d` into a linear form, blend color rows `rows`: each
/// class's arithmetic, `d`'s lane where the write stores none.
fn blend_value(b: &Blend, (s, s1): ([f32; 4], [f32; 4]), d: [f32; 4], rows: [u32; 8]) -> [f32; 4] {
    let (lo, hi) = SOURCE_RANGE;
    // The second source clamped as the source.
    let s1: [f64; 4] = std::array::from_fn(|k| {
        f64::from(match (k, s1[k].is_nan()) {
            (3, true) => 0.0,
            (3, false) => s1[k].clamp(0.0, 1.0),
            (_, true) => lo,
            (_, false) => s1[k].clamp(lo, hi),
        })
    });
    // A `float4` output's color lanes unrounded, clamped to its own range (a NaN to its low end),
    // and every other read of its `half`.
    let (flo, fhi) = FLOAT_SOURCE_RANGE;
    let sf: [f64; 4] = s.map(|x| f64::from(if x.is_nan() { flo } else { x.clamp(flo, fhi) }));
    let s = match b.float {
        true => s.map(|x| half(crate::float16::f32_to_f16_bits(x))),
        false => s,
    };
    let s: [f64; 4] = std::array::from_fn(|k| {
        f64::from(match (k, s[k].is_nan()) {
            (3, true) => 0.0,
            (3, false) => s[k].clamp(0.0, 1.0),
            (_, true) => lo,
            (_, false) => s[k].clamp(lo, hi),
        })
    });
    let d = d.map(f64::from);
    let c: [f64; 8] = rows.map(|w| f64::from(f32::from_bits(w)));
    // The first row's color clamped to the form's range, as the source's.
    let c: [f64; 8] = std::array::from_fn(|k| match k {
        0..=2 => c[k].clamp(f64::from(lo), f64::from(hi)),
        _ => c[k],
    });
    let minus = |k: usize| match b.math {
        BlendMath::Fast => c[4 + k],
        BlendMath::Rounded => h(1.0 - c[k]),
    };
    let lane = |k: usize| LaneValues {
        s: s[k],
        sa: s[3],
        d: d[k],
        da: d[3],
        c: c[k],
        ca: c[3],
        mc: minus(k),
        mca: minus(3),
        sf: sf[k],
        s1: s1[k],
        s1a: s1[3],
    };
    std::array::from_fn(|k| {
        let class = match k {
            3 => &b.alpha,
            _ => &b.color,
        };
        match class {
            Some(a) => arith_value(&lane(k), a) as f32,
            None => d[k] as f32,
        }
    })
}

/// The factors the model holds (every one but the second source's).
const FACTORS: [BlendFactor; 15] = [
    BlendFactor::Zero,
    BlendFactor::One,
    BlendFactor::SourceColor,
    BlendFactor::OneMinusSourceColor,
    BlendFactor::SourceAlpha,
    BlendFactor::OneMinusSourceAlpha,
    BlendFactor::DestinationColor,
    BlendFactor::OneMinusDestinationColor,
    BlendFactor::DestinationAlpha,
    BlendFactor::OneMinusDestinationAlpha,
    BlendFactor::SourceAlphaSaturated,
    BlendFactor::BlendColor,
    BlendFactor::OneMinusBlendColor,
    BlendFactor::BlendAlpha,
    BlendFactor::OneMinusBlendAlpha,
];

/// Every factor pair under each of the three operations that read them, then `Min` and `Max`.
fn keys() -> Vec<(BlendFactor, BlendFactor, BlendOperation)> {
    let mut out = Vec::new();
    for op in [
        BlendOperation::Add,
        BlendOperation::Subtract,
        BlendOperation::ReverseSubtract,
    ] {
        for fs in FACTORS {
            for fd in FACTORS {
                out.push((fs, fd, op));
            }
        }
    }
    out.push((BlendFactor::One, BlendFactor::One, BlendOperation::Min));
    out.push((BlendFactor::One, BlendFactor::One, BlendOperation::Max));
    out
}

/// The masks the general check rotates through: every channel, two color channels and alpha,
/// alpha alone, and two color channels alone.
const MASKS: [ColorWriteMask; 4] = [
    ColorWriteMask::ALL,
    ColorWriteMask::RED
        .union(ColorWriteMask::GREEN)
        .union(ColorWriteMask::ALPHA),
    ColorWriteMask::ALPHA,
    ColorWriteMask::RED.union(ColorWriteMask::BLUE),
];

/// The shader computes each blend's arithmetic (`blend_value`), on each linear form, for every
/// color key (each paired with an alpha key in turn) and every alpha key under color factors of one
/// minus the blend alpha, under several masks, over random texels, sources and blend colors.
#[test]
fn every_blend_computes_its_arithmetic() {
    let keys = keys();
    let mut configs = Vec::new();
    for (i, &color) in keys.iter().enumerate() {
        configs.push(equation(color, keys[(i * 31 + 7) % keys.len()]));
    }
    // Alpha's sums keep the color factors' one-minus-blend-alpha whole where it comes second.
    for (i, &alpha) in keys.iter().enumerate() {
        let shared = matches!(
            (alpha.0, alpha.1),
            (
                BlendFactor::OneMinusBlendColor | BlendFactor::OneMinusBlendAlpha,
                _
            ) | (
                _,
                BlendFactor::OneMinusBlendColor | BlendFactor::OneMinusBlendAlpha
            )
        );
        if shared {
            let other = FACTORS[i % FACTORS.len()];
            configs.push(equation(
                (other, BlendFactor::OneMinusBlendAlpha, BlendOperation::Add),
                alpha,
            ));
        }
    }
    let mut r = Rng(0x5852_4232);
    let mut built = [0; 3];
    let models = [
        (FloatFmt::Half, BlendMath::Fast),
        (FloatFmt::F32, BlendMath::Fast),
        (FloatFmt::Half, BlendMath::Rounded),
    ];
    for (i, eq) in configs.iter().enumerate() {
        let form = LINEAR[i % LINEAR.len()];
        let mask = MASKS[i % MASKS.len()];
        for (n, (output, math)) in models.into_iter().enumerate() {
            let Ok(b) = Blend::for_output(eq, form, mask, (Some(output), math)) else {
                continue;
            };
            built[n] += 1;
            arithmetic_holds(&mut r, &b, (eq, form, mask), (output, math));
        }
    }
    // Only the color channels' blend color difference is refused among them, for each model.
    for (n, built) in built.into_iter().enumerate() {
        assert!(
            built > configs.len() * 9 / 10,
            "output {n}: {built} of {}",
            configs.len()
        );
    }
    // Without fast math, the second source's factors in color and in alpha, under masks of one
    // color channel too: each is built.
    use BlendFactor as F;
    let mut dual = Vec::new();
    for f1 in [
        F::Source1Color,
        F::OneMinusSource1Color,
        F::Source1Alpha,
        F::OneMinusSource1Alpha,
    ] {
        for f in FACTORS {
            for op in [
                BlendOperation::Add,
                BlendOperation::Subtract,
                BlendOperation::ReverseSubtract,
            ] {
                dual.push((f1, f, op));
                dual.push((f, f1, op));
            }
        }
    }
    let one = [
        ColorWriteMask::RED,
        ColorWriteMask::GREEN.union(ColorWriteMask::ALPHA),
        ColorWriteMask::BLUE,
    ];
    let rounded = (FloatFmt::Half, BlendMath::Rounded);
    for (i, &color) in dual.iter().enumerate() {
        let eq = equation(color, dual[(i * 31 + 7) % dual.len()]);
        let form = LINEAR[i % LINEAR.len()];
        let mask = match i % 2 {
            0 => MASKS[(i / 2) % MASKS.len()],
            _ => one[(i / 2) % one.len()],
        };
        let b = Blend::for_output(&eq, form, mask, (Some(rounded.0), rounded.1))
            .unwrap_or_else(|e| panic!("{eq:?} {mask:?} without fast math: {e}"));
        arithmetic_holds(&mut r, &b, (&eq, form, mask), rounded);
    }
}

/// The shader's blend `b` of `eq` into `form` under `mask`, of an output of `output` in a function
/// of `math`, computes [`blend_value`] over random texels, sources, second sources and blend
/// colors.
fn arithmetic_holds(
    r: &mut Rng,
    b: &Blend,
    (eq, form, mask): (&BlendEquation, ExtendedRange, ColorWriteMask),
    (output, math): (FloatFmt, BlendMath),
) {
    let color = [0u8; 4].map(|_| half(r.half_bits()));
    let (lanes, inputs) = blend_lanes(r, form, 12, color, output);
    let got = run(form, Op::Blend(*eq, mask, output, math), &lanes);
    let rows = constant_words(color);
    for ((g, (codes, s)), words) in got.iter().zip(&inputs).zip(&lanes) {
        let d = tile_model(form, *codes);
        let s1: [f32; 4] = std::array::from_fn(|k| f32::from_bits(words[16 + k]));
        let want = blend_value(b, (*s, s1), d, rows);
        let lanes = g.map(f32::from_bits);
        assert!(
            lanes.iter().zip(want).all(|(g, w)| *g == w),
            "{form:?} {eq:?} {mask:?} {output:?} {math:?}: {b:?}, source {s:?} second {s1:?} \
             over codes {codes:?}, blend color {color:?}: {lanes:?}, the arithmetic's {want:?}"
        );
    }
}

/// The code a write of alpha value `v` stores in a two-plane texel's alpha byte.
fn alpha_byte(v: f32) -> u32 {
    let v = if v.is_nan() { 0.0 } else { v.clamp(0.0, 1.0) };
    round_even(f64::from(v) * 255.0) as u32
}

/// A blended write under a mask stores the blended channels' codes and keeps the others as Metal
/// keeps them: a color code as it is (1023 too), plane 0's and `BGR10_XR`'s top two bits as the
/// write sets them, a two-plane texel's alpha byte, and `BGRA10_XR`'s alpha clamped to
/// `[384, 894]`, its six low bits zero.
#[test]
fn masked_blends_keep_metals_codes() {
    let mut r = Rng(0x5852_4233);
    let premul = equation(
        (
            BlendFactor::SourceAlpha,
            BlendFactor::OneMinusSourceAlpha,
            BlendOperation::Add,
        ),
        (
            BlendFactor::One,
            BlendFactor::OneMinusSourceAlpha,
            BlendOperation::Add,
        ),
    );
    for form in LINEAR {
        for mask in MASKS.into_iter().skip(1) {
            let (lanes, inputs) = blend_lanes(&mut r, form, 256, [0.0; 4], FloatFmt::Half);
            let half = (FloatFmt::Half, BlendMath::Fast);
            let values = run(form, Op::Blend(premul, mask, half.0, half.1), &lanes);
            let texels = run(form, Op::BlendTexel(premul, mask, half.0, half.1), &lanes);
            for ((v, t), (codes, _)) in values.iter().zip(&texels).zip(&inputs) {
                let v = v.map(f32::from_bits);
                let channels = [
                    ColorWriteMask::RED,
                    ColorWriteMask::GREEN,
                    ColorWriteMask::BLUE,
                    ColorWriteMask::ALPHA,
                ];
                let rgba: [u32; 4] = std::array::from_fn(|k| {
                    let written = mask.contains(channels[k]);
                    match (k, written, form) {
                        (3, true, ExtendedRange::Planes { .. }) => alpha_byte(v[3]),
                        (3, true, _) => code(v[3], false, true),
                        (3, false, ExtendedRange::Bgra10 { .. }) => codes[3].clamp(384, 894),
                        (_, true, _) => code(v[k], false, false),
                        (_, false, _) => codes[k],
                    }
                });
                let top = match form {
                    ExtendedRange::Bgr10 { .. } => 3,
                    _ => 0,
                };
                let want = raw(form, rgba, top);
                assert_eq!(
                    *t, want,
                    "{form:?} {mask:?}: codes {codes:?}, blended {v:?}"
                );
            }
        }
    }
}

/// The gap `eq` (into `form`, the write storing `mask`) is refused with.
fn gap(
    eq: &BlendEquation,
    form: ExtendedRange,
    mask: ColorWriteMask,
) -> Option<ExtendedRangeBlend> {
    match Blend::of(eq, form, mask, BlendMath::Fast) {
        Err(Error::Unsupported(u)) => {
            assert_eq!(u.needs, Needs::NotImplemented);
            match u.feature {
                Feature::ExtendedRangeBlend(g) => Some(g),
                f => panic!("{eq:?}: refused as {f:?}"),
            }
        }
        Err(e) => panic!("{eq:?}: {e}"),
        Ok(_) => None,
    }
}

/// What the model does not hold is refused by name: a factor of the second source, the color
/// channels' blend color difference (but alpha's, and the sum), a mask of one color channel, an
/// `_sRGB` form's blend other than the or under a mask, a `float4` output other than under the
/// blends whose color lanes take one term alone, and another output; a placeholder factor or
/// operation is the descriptor's error.
#[test]
fn blends_outside_the_model_are_refused_by_name() {
    use BlendFactor as F;
    use BlendOperation as O;
    let linear = ExtendedRange::Planes { srgb: false };
    let srgb = ExtendedRange::Planes { srgb: true };
    let premul = equation(
        (F::One, F::OneMinusSourceAlpha, O::Add),
        (F::One, F::OneMinusSourceAlpha, O::Add),
    );
    assert_eq!(gap(&premul, linear, ALL), None);
    assert_eq!(gap(&premul, srgb, ALL), None);
    let dual = BlendEquation {
        destination_rgb: F::OneMinusSource1Color,
        ..premul
    };
    assert_eq!(
        gap(&dual, linear, ALL),
        Some(ExtendedRangeBlend::DualSource)
    );
    for op in [O::Subtract, O::ReverseSubtract] {
        for (fs, fd) in [
            (F::BlendColor, F::OneMinusBlendColor),
            (F::OneMinusBlendColor, F::BlendColor),
        ] {
            let color = BlendEquation {
                source_rgb: fs,
                destination_rgb: fd,
                rgb_operation: op,
                ..premul
            };
            assert_eq!(
                gap(&color, linear, ALL),
                Some(ExtendedRangeBlend::ConstantColorDifference)
            );
            // Alpha alone is written: the color lanes' pair is not computed.
            assert_eq!(gap(&color, linear, ColorWriteMask::ALPHA), None);
            let alpha = BlendEquation {
                source_alpha: fs,
                destination_alpha: fd,
                alpha_operation: op,
                ..premul
            };
            assert_eq!(gap(&alpha, linear, ALL), None);
        }
    }
    let sum = BlendEquation {
        source_rgb: F::BlendColor,
        destination_rgb: F::OneMinusBlendColor,
        ..premul
    };
    assert_eq!(gap(&sum, linear, ALL), None);
    let one = ColorWriteMask::GREEN.union(ColorWriteMask::ALPHA);
    assert_eq!(
        gap(&premul, linear, one),
        Some(ExtendedRangeBlend::OneColorChannel)
    );
    let over = BlendEquation {
        source_rgb: F::SourceAlpha,
        ..premul
    };
    assert_eq!(gap(&over, srgb, ALL), Some(ExtendedRangeBlend::SrgbForm));
    let rgb = ColorWriteMask::ALL.difference(ColorWriteMask::ALPHA);
    assert_eq!(gap(&premul, srgb, rgb), Some(ExtendedRangeBlend::SrgbForm));
    // A `float4` output: the blends whose color lanes take one term alone, unmasked.
    let takes = |eq: &BlendEquation, mask: ColorWriteMask, fmt: Option<FloatFmt>| match Blend::of(
        eq,
        linear,
        mask,
        BlendMath::Fast,
    )
    .expect("modelled")
    .takes(fmt)
    {
        Ok(()) => None,
        Err(Error::Unsupported(u)) => match u.feature {
            Feature::ExtendedRangeBlend(g) => Some(g),
            f => panic!("{f:?}"),
        },
        Err(e) => panic!("{e}"),
    };
    let mask_blend = BlendEquation {
        source_rgb: F::Zero,
        ..premul
    };
    assert_eq!(takes(&premul, ALL, Some(FloatFmt::Half)), None);
    assert_eq!(takes(&over, ALL, Some(FloatFmt::Half)), None);
    assert_eq!(takes(&mask_blend, ALL, Some(FloatFmt::F32)), None);
    assert_eq!(
        takes(&premul, ALL, Some(FloatFmt::F32)),
        Some(ExtendedRangeBlend::FloatOutput)
    );
    assert_eq!(
        takes(&mask_blend, rgb, Some(FloatFmt::F32)),
        Some(ExtendedRangeBlend::FloatOutput)
    );
    assert_eq!(takes(&premul, ALL, None), Some(ExtendedRangeBlend::Output));
    // `for_output`: a `float4` output into a linear form takes its own model, but a factor of the
    // second source in any of the four factors; into an `_sRGB` form, what `takes` answers.
    let output =
        |eq: &BlendEquation, form: ExtendedRange, mask, (output, math)| match Blend::for_output(
            eq,
            form,
            mask,
            (output, math),
        ) {
            Ok(b) => Ok(b.float),
            Err(Error::Unsupported(u)) => match u.feature {
                Feature::ExtendedRangeBlend(g) => Err(g),
                f => panic!("{f:?}"),
            },
            Err(e) => panic!("{e}"),
        };
    let float = |eq: &BlendEquation, form: ExtendedRange, mask: ColorWriteMask| {
        output(eq, form, mask, (Some(FloatFmt::F32), BlendMath::Fast))
    };
    assert_eq!(float(&premul, linear, ALL), Ok(true));
    assert_eq!(float(&over, linear, ALL), Ok(true));
    assert_eq!(float(&premul, linear, rgb), Ok(true));
    let dual_alpha = BlendEquation {
        destination_alpha: F::OneMinusSource1Alpha,
        ..premul
    };
    assert_eq!(
        float(&dual_alpha, linear, ALL),
        Err(ExtendedRangeBlend::DualSource)
    );
    assert_eq!(float(&mask_blend, srgb, ALL), Ok(false));
    assert_eq!(
        float(&premul, srgb, ALL),
        Err(ExtendedRangeBlend::FloatOutput)
    );
    // Without fast math a `float4` output is taken as a `half4` one (rounded to `half` first),
    // into an `_sRGB` form too; the second source's factors, a mask of one color channel, the blend
    // color difference, and an `_sRGB` form's every blend, masked too, are made.
    let rounded = |eq: &BlendEquation, form: ExtendedRange, mask, out| {
        output(eq, form, mask, (out, BlendMath::Rounded))
    };
    for out in [Some(FloatFmt::Half), Some(FloatFmt::F32)] {
        assert_eq!(rounded(&premul, linear, ALL, out), Ok(false));
        assert_eq!(rounded(&over, linear, rgb, out), Ok(false));
        assert_eq!(rounded(&dual, linear, ALL, out), Ok(false));
        assert_eq!(rounded(&dual_alpha, srgb, ALL, out), Ok(false));
        assert_eq!(rounded(&premul, linear, one, out), Ok(false));
        assert_eq!(rounded(&over, srgb, ALL, out), Ok(false));
        assert_eq!(rounded(&premul, srgb, rgb, out), Ok(false));
        for op in [O::Subtract, O::ReverseSubtract] {
            let difference = BlendEquation {
                source_rgb: F::BlendColor,
                destination_rgb: F::OneMinusBlendColor,
                rgb_operation: op,
                ..premul
            };
            assert_eq!(rounded(&difference, linear, ALL, out), Ok(false));
        }
        // Into every `_sRGB` form a color class reading the blend color, or its one minus, is
        // built (`BGRA10_XR_sRGB`'s past its bound too).
        let bgra_srgb = ExtendedRange::Bgra10 { srgb: true };
        let reads = |f: F| BlendEquation {
            source_rgb: f,
            destination_rgb: F::OneMinusSourceAlpha,
            ..premul
        };
        assert_eq!(
            rounded(&reads(F::BlendColor), bgra_srgb, ALL, out),
            Ok(false)
        );
        assert_eq!(
            rounded(&reads(F::OneMinusBlendColor), bgra_srgb, ALL, out),
            Ok(false)
        );
        assert_eq!(
            rounded(&reads(F::BlendAlpha), bgra_srgb, ALL, out),
            Ok(false)
        );
        for form in [srgb, ExtendedRange::Bgr10 { srgb: true }] {
            assert_eq!(rounded(&reads(F::BlendColor), form, ALL, out), Ok(false));
        }
    }
    assert_eq!(
        rounded(&premul, linear, ALL, None),
        Err(ExtendedRangeBlend::Output)
    );
    assert_eq!(rounded(&premul, srgb, ALL, Some(FloatFmt::F32)), Ok(false));
    // A placeholder is the descriptor's error.
    let placeholder = BlendEquation {
        alpha_operation: O::Unspecialized,
        ..premul
    };
    assert!(matches!(
        Blend::of(&placeholder, linear, ALL, BlendMath::Fast),
        Err(Error::InvalidDescriptor(_))
    ));
}

/// The forms the model gives the guests' common blends and one of each kind.
#[test]
fn the_model_picks_metals_forms() {
    use BlendFactor as F;
    use BlendOperation as O;
    let linear = ExtendedRange::Bgra10 { srgb: false };
    let of = |color, alpha| {
        let b = Blend::of(&equation(color, alpha), linear, ALL, BlendMath::Fast).expect("modelled");
        (b.color.expect("color"), b.alpha.expect("alpha"))
    };
    // Premultiplied over: one fused multiply-add per class.
    let (color, alpha) = of(
        (F::One, F::OneMinusSourceAlpha, O::Add),
        (F::One, F::OneMinusSourceAlpha, O::Add),
    );
    let fused = |s, sa, d| Arith::Fused {
        product: (1, (d, Node::OneMinus(sa))),
        addend: (1, (s, Node::One)),
    };
    assert_eq!(color, fused(Atom::S, Atom::Sa, Atom::D));
    assert_eq!(alpha, fused(Atom::Sa, Atom::Sa, Atom::Da));
    // Over: the color lanes' contraction (their `1 - sa` a splat), the source's product rounded
    // first at equal ranks; alpha's lerp `h(da + sa (sa - da))`.
    let (color, alpha) = of(
        (F::SourceAlpha, F::OneMinusSourceAlpha, O::Add),
        (F::SourceAlpha, F::OneMinusSourceAlpha, O::Add),
    );
    assert_eq!(
        color,
        Arith::Contracted {
            rounded: (1, (Atom::S, Node::Atom(Atom::Sa))),
            fused: (1, (Atom::D, Node::OneMinus(Atom::Sa))),
        }
    );
    assert_eq!(
        alpha,
        Arith::Lerp {
            y: Node::Atom(Atom::Da),
            z: Node::Atom(Atom::Sa),
            inner: vec![
                (1, Some(Node::Atom(Atom::Sa))),
                (-1, Some(Node::Atom(Atom::Da)))
            ],
        }
    );
    // A shared factor: `d (s - d)` of color, destination color both.
    let (color, _) = of(
        (F::DestinationColor, F::DestinationColor, O::Subtract),
        (F::Zero, F::One, O::Add),
    );
    assert_eq!(
        color,
        Arith::Common {
            z: Node::Atom(Atom::D),
            inner: vec![
                (1, Some(Node::Atom(Atom::S))),
                (-1, Some(Node::Atom(Atom::D)))
            ],
            round_inner: false,
        }
    );
    // `BGR10_XR`'s destination alpha is one: its factor folds to one, a fused multiply-add.
    let b = Blend::of(
        &equation(
            (F::SourceAlpha, F::DestinationAlpha, O::Add),
            (F::One, F::Zero, O::Add),
        ),
        ExtendedRange::Bgr10 { srgb: false },
        ALL,
        BlendMath::Fast,
    )
    .expect("modelled");
    assert_eq!(b.alpha, None);
    assert_eq!(
        b.color,
        Some(Arith::Fused {
            product: (1, (Atom::S, Node::Atom(Atom::Sa))),
            addend: (1, (Atom::D, Node::One)),
        })
    );
    assert!(!b.reads_constant());
    let b = Blend::of(
        &equation(
            (F::BlendColor, F::OneMinusBlendAlpha, O::Add),
            (F::One, F::Zero, O::Add),
        ),
        linear,
        ALL,
        BlendMath::Fast,
    )
    .expect("modelled");
    assert!(b.reads_constant());
    // Without fast math: premultiplied over adds the source to the destination's rounded
    // product; over rounds both products; a zero factor drops its term, `min` reads no factor.
    let rounded = |color, alpha, form| {
        let b = Blend::for_output(
            &equation(color, alpha),
            form,
            ALL,
            (Some(FloatFmt::Half), BlendMath::Rounded),
        )
        .expect("modelled");
        (b.color.expect("color"), b.alpha)
    };
    let terms = |s: (Atom, Node), d: (Atom, Node)| Arith::Rounded {
        source: (1, s),
        destination: (1, d),
    };
    let (color, alpha) = rounded(
        (F::One, F::OneMinusSourceAlpha, O::Add),
        (F::One, F::OneMinusSourceAlpha, O::Add),
        linear,
    );
    assert_eq!(
        color,
        terms((Atom::S, Node::One), (Atom::D, Node::OneMinus(Atom::Sa)))
    );
    assert_eq!(
        alpha,
        Some(terms(
            (Atom::Sa, Node::One),
            (Atom::Da, Node::OneMinus(Atom::Sa))
        ))
    );
    let (color, _) = rounded(
        (F::SourceAlpha, F::OneMinusSourceAlpha, O::Add),
        (F::Zero, F::One, O::ReverseSubtract),
        linear,
    );
    assert_eq!(
        color,
        terms(
            (Atom::S, Node::Atom(Atom::Sa)),
            (Atom::D, Node::OneMinus(Atom::Sa))
        )
    );
    let (color, alpha) = rounded(
        (
            F::DestinationAlpha,
            F::OneMinusDestinationAlpha,
            O::Subtract,
        ),
        (F::Zero, F::One, O::Min),
        ExtendedRange::Bgr10 { srgb: false },
    );
    assert_eq!(
        color,
        Arith::Term {
            sign: 1,
            term: (Atom::S, Node::One),
        }
    );
    assert_eq!(alpha, None);
    let (_, alpha) = rounded((F::One, F::One, O::Add), (F::Zero, F::One, O::Min), linear);
    assert_eq!(
        alpha,
        Some(Arith::MinMax {
            max: false,
            source: Atom::Sa,
        })
    );
}

/// The blend color's push words: the color clamped to the source range (a NaN to its low end) and
/// alpha to `[0, 1]` (a NaN to 0), each rounded to `half`, nearest even, then `h(1 - x)` of the
/// clamped color and `h(1 - h(a))` of alpha. A NaN of either sign, the infinities, and values
/// beyond `half`'s range clamp so. The first row's color is clamped to the
/// hull of the linear and `BGRA10_XR_sRGB` source ranges, which each blend clamps to its own range.
#[test]
fn the_blend_color_is_metals() {
    for (lo, hi) in [SOURCE_RANGE, SRGB_SOURCE_RANGE] {
        assert!(COLOR_HULL.0 <= lo && hi <= COLOR_HULL.1);
    }
    assert_eq!(COLOR_HULL.0, SOURCE_RANGE.0.min(SRGB_SOURCE_RANGE.0));
    assert_eq!(COLOR_HULL.1, SOURCE_RANGE.1.max(SRGB_SOURCE_RANGE.1));
    // A color channel at or past `BGRA10_XR_sRGB`'s direct bound is marked a `half` past the
    // hull's top; the `f32` below the bound is the hull's top rounded.
    let top = PAST_DIRECT_BOUND;
    assert_eq!(
        f64::from(PAST_DIRECT_BOUND) - f64::from(COLOR_HULL.1),
        1.0 / 1024.0
    );
    assert_eq!(
        f64::from(COLOR_HULL.1) - f64::from(SRGB_DIRECT_TOP),
        1.0 / 1024.0
    );
    let below = f32::from_bits(SRGB_DIRECT_BOUND.to_bits() - 1);
    let w = constant_words([SRGB_DIRECT_BOUND, below, SRGB_SOURCE_RANGE.1, 1.0]);
    assert_eq!(
        w[..3]
            .iter()
            .map(|&b| f32::from_bits(b))
            .collect::<Vec<_>>(),
        [PAST_DIRECT_BOUND, COLOR_HULL.1, PAST_DIRECT_BOUND]
    );
    assert_eq!(
        constant_words([0.3, 1.2, -0.4, 0.45]),
        [
            0x3e99_a000,
            0x3f99_a000,
            0xbecc_c000,
            0x3ee6_6000,
            0x3f33_4000,
            0xbe4c_c000,
            0x3fb3_4000,
            0x3f0c_c000
        ]
    );
    let (lo, hi) = SOURCE_RANGE;
    let w = constant_words([f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::NAN]);
    let v = w.map(f32::from_bits);
    assert_eq!(v, [lo, top, lo, 0.0, 1.0 - lo, 1.0 - hi, 1.0 - lo, 1.0]);
    let w = constant_words([-f32::NAN, 65520.0, -1e30, f32::INFINITY]);
    let v = w.map(f32::from_bits);
    assert_eq!(v, [lo, top, lo, 1.0, 1.0 - lo, 1.0 - hi, 1.0 - lo, 0.0]);
    let w = constant_words([1e30, -65520.0, -f32::NAN, -f32::NAN]);
    let v = w.map(f32::from_bits);
    assert_eq!(v, [top, lo, lo, 0.0, 1.0 - hi, 1.0 - lo, 1.0 - lo, 1.0]);
}

/// Without fast math, `BlendColor` into `BGRA10_XR_sRGB` reads a color channel at or past
/// [`SRGB_DIRECT_BOUND`] as [`SRGB_DIRECT_TOP`] and the `f32` below it as the form's top, where
/// `OneMinusBlendColor` reads one minus the top at both (every `f32` from
/// `1.6689000129699707` to `1.6689451932907104` under both factors), but beside `BlendColor` in
/// its class one minus `BlendColor`'s read (our probe over the same sweep, and its
/// configurations 17 and 40); every other form reads its own range's top under each. Through the
/// shader the binder's words feed, a `half` source of one over a texel of zeros.
#[test]
fn the_blend_color_reads_metals_bound() {
    use BlendFactor as F;
    use BlendOperation as O;
    let below = f32::from_bits(SRGB_DIRECT_BOUND.to_bits() - 1);
    let rows = constant_words([SRGB_DIRECT_BOUND, below, f32::INFINITY, 1.0]);
    for form in EVERY_FORM {
        let (_, hi) = super::source_range(form);
        let srgb = form == ExtendedRange::Bgra10 { srgb: true };
        let (read, minus_read) = match srgb {
            true => (SRGB_DIRECT_TOP, 1.0 - SRGB_DIRECT_TOP),
            false => (hi, 1.0 - hi),
        };
        for (factor, other, want) in [
            (F::BlendColor, F::Zero, [read, hi, read]),
            (F::OneMinusBlendColor, F::Zero, [1.0 - hi; 3]),
            (F::BlendColor, F::OneMinusBlendColor, [read, hi, read]),
            (
                F::OneMinusBlendColor,
                F::BlendColor,
                [minus_read, 1.0 - hi, minus_read],
            ),
        ] {
            let eq = equation((factor, other, O::Add), (F::One, F::Zero, O::Add));
            let mut words = [0u32; 20];
            words[..4].copy_from_slice(&raw(form, [384; 4], 0));
            words[4..8].copy_from_slice(&[1.0f32.to_bits(); 4]);
            words[8..16].copy_from_slice(&rows);
            let op = Op::Blend(eq, ALL, FloatFmt::Half, BlendMath::Rounded);
            let got = run(form, op, &[words]);
            assert_eq!(
                got[0][..3]
                    .iter()
                    .map(|&b| f32::from_bits(b))
                    .collect::<Vec<_>>(),
                want,
                "{form:?} {factor:?} {other:?}"
            );
        }
    }
}

/// A padded 10-bit form reads every 16-bit word as the code in its top 10 bits over 1023, correctly
/// rounded (a `half` result to nearest even), blue 0 and alpha 1, whatever the low 6 bits hold
/// (every word); and writes the code of the exact product of the value clamped to `[0, 1]` (a
/// NaN 0) with 1023, rounded to nearest, in its word's top 10 bits, the low 6 bits zero (a grid,
/// and three `f32`s about every midpoint, each lane).
#[test]
fn padded_forms_read_and_write_metals_codes() {
    let value = |w: u32| ((w >> 6) as f64 / 1023.0) as f32;
    for form in [ExtendedRange::R10X6, ExtendedRange::Rg10X12] {
        let two = form == ExtendedRange::Rg10X12;
        // Every code under four fillings of the low 6 bits, the other lanes' words other codes.
        let lanes: Vec<[u32; 4]> = (0..4096u32)
            .map(|i| {
                let w = (i & 1023) << 6 | [0, 0x3F, 0x15, 0x2A][(i >> 10) as usize];
                [w, w.wrapping_mul(40503) & 0xFFFF, 0xFFFF, 0xFFFF]
            })
            .collect();
        let got = run(form, Op::Decode, &lanes);
        let halves = run(form, Op::DecodeHalf, &lanes);
        for ((l, g), h) in lanes.iter().zip(&got).zip(&halves) {
            let green = if two { value(l[1]) } else { 0.0 };
            let want = [value(l[0]), green, 0.0, 1.0];
            assert_eq!(*g, want.map(f32::to_bits), "{form:?} words {l:x?}");
            let want = want.map(|v| to_half(f64::from(v), false));
            assert_eq!(*h, want.map(f32::to_bits), "{form:?} half, words {l:x?}");
        }
        // Writes: about every midpoint, the codes, and the specials.
        let mut values = vec![1.0f32];
        for k in 0..1023u32 {
            values.push((f64::from(k) / 1023.0) as f32);
            let mid = ((f64::from(k) + 0.5) / 1023.0) as f32;
            values.extend((0..5).map(|d| f32::from_bits(mid.to_bits() + d - 2)));
        }
        values.extend([
            f32::NAN,
            -f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            -0.0,
            -1.0,
            1.5,
            2.0,
            f32::from_bits(1),
            -f32::from_bits(1),
        ]);
        let code = |v: f32| -> u32 {
            match v.is_nan() {
                true => 0,
                false => (f64::from(v.clamp(0.0, 1.0)) * 1023.0).round_ties_even() as u32,
            }
        };
        let lanes: Vec<[u32; 4]> = (0..values.len())
            .map(|i| {
                let g = values[(i * 7919) % values.len()];
                [
                    values[i].to_bits(),
                    g.to_bits(),
                    0.5f32.to_bits(),
                    1f32.to_bits(),
                ]
            })
            .collect();
        let got = run(form, Op::Encode, &lanes);
        for (l, g) in lanes.iter().zip(&got) {
            let (r, gr) = (f32::from_bits(l[0]), f32::from_bits(l[1]));
            let green = if two { code(gr) << 6 } else { 0 };
            assert_eq!(*g, [code(r) << 6, green, 0, 0], "{form:?} values {r}, {gr}");
        }
    }
}

/// The code Metal's write of `v` stores to a padded `_sRGB` form's channel (the form docs): a NaN
/// 0, the value clamped to `[0, 1]`; `R10Unorm_X6_sRGB`'s the largest code whose first encode index
/// is at most the value rounded to `2^-13` (nearest even), `RG10Unorm_X12_sRGB`'s the table's code
/// of `RG10Unorm_X12`'s.
fn padded_srgb_code(form: ExtendedRange, v: f32) -> u32 {
    let v = match v.is_nan() {
        true => 0.0,
        false => f64::from(v.clamp(0.0, 1.0)),
    };
    match form {
        ExtendedRange::R10X6Srgb => {
            let i = (v * 8192.0).round_ties_even() as u16;
            let codes = R10X6_SRGB_ENCODE_FIRST.partition_point(|&f| f <= i);
            codes as u32 - 1
        }
        _ => u32::from(RG10X12_SRGB_WRITE_CODES[(v * 1023.0).round_ties_even() as usize]),
    }
}

/// `R10Unorm_X6_sRGB` reads every 16-bit word as its code's decode step over 8192 (a `half` result
/// with ties away from zero), `RG10Unorm_X12_sRGB` as `RG10Unorm_X12` does (`c / 1023`, a `half` to
/// nearest even), blue 0 and alpha 1, whatever the low 6 bits hold; and each writes Metal's code
/// through its table ([`padded_srgb_code`]) in its word's top 10 bits, the low 6 bits zero: values
/// about every threshold of `R10Unorm_X6_sRGB`'s codes and every midpoint of the linear codes, the
/// codes' values, and the specials.
#[test]
fn padded_srgb_forms_read_and_write_metals_codes() {
    let step = |w: u32| f32::from(R10X6_SRGB_DECODE_STEPS[(w >> 6) as usize]) / 8192.0;
    let linear = |w: u32| ((w >> 6) as f64 / 1023.0) as f32;
    // Values about every threshold and midpoint, the codes' values, and the specials.
    let mut values = vec![1.0f32];
    for k in 0..1024u32 {
        values.push(step(k << 6));
        values.push((f64::from(k) / 1023.0) as f32);
        let mid = ((f64::from(k) + 0.5) / 1023.0) as f32;
        values.extend((0..5).map(|d| f32::from_bits(mid.to_bits() + d - 2)));
        let first = f64::from(R10X6_SRGB_ENCODE_FIRST[k as usize]);
        if first > 0.0 {
            let edge = ((first - 0.5) / 8192.0) as f32;
            values.extend((0..5).map(|d| f32::from_bits(edge.to_bits() + d - 2)));
        }
    }
    values.extend([
        f32::NAN,
        -f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        -0.0,
        -1.0,
        1.5,
        2.0,
        f32::from_bits(1),
        -f32::from_bits(1),
    ]);
    for form in [ExtendedRange::R10X6Srgb, ExtendedRange::Rg10X12Srgb] {
        let two = form == ExtendedRange::Rg10X12Srgb;
        let (red, rounding_away) = match two {
            false => (step as fn(u32) -> f32, true),
            true => (linear as fn(u32) -> f32, false),
        };
        // Every code under four fillings of the low 6 bits, the other lanes' words other codes.
        let lanes: Vec<[u32; 4]> = (0..4096u32)
            .map(|i| {
                let w = (i & 1023) << 6 | [0, 0x3F, 0x15, 0x2A][(i >> 10) as usize];
                [w, w.wrapping_mul(40503) & 0xFFFF, 0xFFFF, 0xFFFF]
            })
            .collect();
        let got = run(form, Op::Decode, &lanes);
        let halves = run(form, Op::DecodeHalf, &lanes);
        for ((l, g), h) in lanes.iter().zip(&got).zip(&halves) {
            let green = if two { linear(l[1]) } else { 0.0 };
            let want = [red(l[0]), green, 0.0, 1.0];
            assert_eq!(*g, want.map(f32::to_bits), "{form:?} words {l:x?}");
            let want = [
                to_half(f64::from(want[0]), rounding_away),
                to_half(f64::from(green), false),
                0.0,
                1.0,
            ];
            assert_eq!(*h, want.map(f32::to_bits), "{form:?} half, words {l:x?}");
        }
        let lanes: Vec<[u32; 4]> = (0..values.len())
            .map(|i| {
                let g = values[(i * 7919) % values.len()];
                [
                    values[i].to_bits(),
                    g.to_bits(),
                    0.5f32.to_bits(),
                    1f32.to_bits(),
                ]
            })
            .collect();
        let got = run(form, Op::Encode, &lanes);
        for (l, g) in lanes.iter().zip(&got) {
            let (r, gr) = (f32::from_bits(l[0]), f32::from_bits(l[1]));
            let green = if two {
                padded_srgb_code(form, gr) << 6
            } else {
                0
            };
            let want = [padded_srgb_code(form, r) << 6, green, 0, 0];
            assert_eq!(*g, want, "{form:?} values {r}, {gr}");
        }
    }
    // The crate's clear texels and load codes agree with the write rule (the runtime's clear
    // writes `clear_texel`, its load pass `load_code`).
    for c in 0..1024u32 {
        let read = step(c << 6);
        let code = padded_srgb_code(ExtendedRange::R10X6Srgb, read);
        assert_eq!(
            ExtendedRange::R10X6Srgb.load_code(c),
            Some(code),
            "code {c}"
        );
        let read = linear(c << 6);
        let code = padded_srgb_code(ExtendedRange::Rg10X12Srgb, read);
        assert_eq!(
            ExtendedRange::Rg10X12Srgb.load_code(c),
            Some(code),
            "code {c}"
        );
    }
}

/// The texel words [`run`] reads of a table texel of `layout`: its plane-0 bytes `plane` and a
/// two-plane texel's alpha byte `alpha`.
fn table_texel(layout: xr_blends::Layout, plane: &[u8], alpha: Option<&u8>) -> [u32; 4] {
    let word = |k: usize| -> u32 {
        let b: [u8; 2] = std::array::from_fn(|i| plane.get(2 * k + i).copied().unwrap_or(0));
        u32::from(u16::from_le_bytes(b))
    };
    match layout {
        xr_blends::Layout::Bgra10 => std::array::from_fn(word),
        xr_blends::Layout::Bgr10 => [word(0) | word(1) << 16, 0, 0, 0],
        xr_blends::Layout::Planes => [
            word(0) | word(1) << 16,
            u32::from(alpha.copied().unwrap_or(0)),
            0,
            0,
        ],
    }
}

/// A texel [`run`] wrote, as the table stores it: plane 0's bytes and a two-plane texel's alpha
/// byte.
fn table_bytes(layout: xr_blends::Layout, t: [u32; 4], plane: &mut Vec<u8>, alpha: &mut Vec<u8>) {
    match layout {
        xr_blends::Layout::Bgra10 => {
            for w in t {
                plane.extend((w as u16).to_le_bytes());
            }
        }
        xr_blends::Layout::Bgr10 => plane.extend(t[0].to_le_bytes()),
        xr_blends::Layout::Planes => {
            plane.extend(t[0].to_le_bytes());
            alpha.push(t[1] as u8);
        }
    }
}

/// metal2vulkan's outcome for a table configuration of `eq` and `mask` into `pf` from an output
/// of `output` of a function of math `math`, as the storage form decides it
/// (`fragment/fetch.rs`): a write that stores none of the format's channels blends nothing;
/// another is made where [`Blend::for_output`] holds it, else refused with the gap it names.
fn table_outcome(
    pf: PixelFormat,
    eq: &BlendEquation,
    mask: ColorWriteMask,
    (output, math): (FloatFmt, BlendMath),
) -> (String, Option<ColorWriteMask>) {
    let form = ExtendedRange::of(pf).expect("an extended-range format");
    let channels = match form {
        ExtendedRange::Bgr10 { .. } => ALL.difference(ColorWriteMask::ALPHA),
        _ => ALL,
    };
    let written = channels.intersection(mask);
    if written.is_empty() {
        return ("made".into(), None);
    }
    match Blend::for_output(eq, form, written, (Some(output), math)) {
        Ok(_) => ("made".into(), Some(written)),
        Err(Error::Unsupported(u)) => match u.feature {
            Feature::ExtendedRangeBlend(g) => (format!("{g:?}"), None),
            f => panic!("{eq:?}: refused as {f:?}"),
        },
        Err(e) => panic!("{eq:?}: {e}"),
    }
}

/// The lane table (`metal2vulkan_devtools::xr_blends`; plain Metal on an Apple M4):
/// every row's outcome is the model's ([`table_outcome`]), and every blend the model makes stores
/// Metal's bytes, plane 0 and alpha, over the table's 256 texels and outputs through the
/// interpreter (a `float4` output under its own model unrounded, else rounded to `half` first, as
/// the storage form does). A write that stores none of the format's channels is the pass's, the
/// device lanes' to check.
fn the_blend_table_is_metals(shard: usize, shards: usize) {
    blend_table_is_metals(&xr_blends::table(), BlendMath::Fast, false, (shard, shards));
}

sharded_tests!(
    the_blend_table_is_metals = the_blend_table_is_metals(
        shard_0 => 0,
        shard_1 => 1,
        shard_2 => 2,
        shard_3 => 3,
    )
);

/// The lane table of the probe built without fast math (under safe math; under relaxed math gave
/// the same bytes) holds as the table does under [`BlendMath::Rounded`]: a `float4` output rounded
/// to `half` first and blended as a `half4` one, each term's product rounded before the sum.
fn the_blend_table_without_fast_math_is_metals(shard: usize, shards: usize) {
    blend_table_is_metals(
        &xr_blends::safe_table(),
        BlendMath::Rounded,
        false,
        (shard, shards),
    );
}

sharded_tests!(
    the_blend_table_without_fast_math_is_metals = the_blend_table_without_fast_math_is_metals(
        shard_0 => 0,
        shard_1 => 1,
        shard_2 => 2,
        shard_3 => 3,
        shard_4 => 4,
        shard_5 => 5,
        shard_6 => 6,
        shard_7 => 7,
    )
);

/// The lane table under fast math with the `float4` output computed (the source times a loaded
/// vector of ones): a blend under a key of the computed class's tables
/// (`extended_range::computed`), or into an `_sRGB` form, is announced, and every other made
/// blend stores Metal's bytes through the interpreter, the loaded output's model's.
#[test]
fn the_computed_blend_table_is_metals() {
    blend_table_is_metals(&xr_blends::computed_table(), BlendMath::Fast, true, (0, 1));
}

/// The second source's table without fast math (`xr_blends::dual_table`): every
/// configuration reads the function's second output, each is made, and its every blend stores
/// Metal's bytes through the interpreter, the second output rounded to `half` and clamped as the
/// first.
fn the_second_source_blend_table_without_fast_math_is_metals(shard: usize, shards: usize) {
    blend_table_is_metals(
        &xr_blends::dual_table(),
        BlendMath::Rounded,
        false,
        (shard, shards),
    );
}

sharded_tests!(
    the_second_source_blend_table_without_fast_math_is_metals =
        the_second_source_blend_table_without_fast_math_is_metals(
            shard_0 => 0,
            shard_1 => 1,
            shard_2 => 2,
            shard_3 => 3,
            shard_4 => 4,
            shard_5 => 5,
        )
);

/// Every row of table `t` is metal2vulkan's outcome under `math`, and every blend made stores
/// Metal's bytes through the interpreter ([`the_blend_table_is_metals`]); of a computed `float4`
/// output where `computed`. Of the rows whose index is `shard` modulo `shards` (a long table runs
/// as `shards` tests, [`sharded_tests!`]).
fn blend_table_is_metals(
    t: &xr_blends::Table,
    math: BlendMath,
    computed: bool,
    (shard, shards): (usize, usize),
) {
    let mut wrong = Vec::new();
    let mut checked = 0;
    for row in t.rows.iter().skip(shard).step_by(shards) {
        let c = t.configs.get(row.config).expect("a configuration");
        let factor = |k: usize| BlendFactor::try_from(c.equation[k] as u32).expect("a factor");
        let op = |k: usize| BlendOperation::try_from(c.equation[k] as u32).expect("an operation");
        let eq = equation((factor(0), factor(1), op(2)), (factor(3), factor(4), op(5)));
        let mask = ColorWriteMask::try_from(c.mask as u32).expect("a mask");
        let pf = PixelFormat::try_from(row.format).expect("a format");
        let output = if row.half {
            FloatFmt::Half
        } else {
            FloatFmt::F32
        };
        let (ours, written) = table_outcome(pf, &eq, mask, (output, math));
        // A computed `float4`'s blend Metal rounds otherwise than a loaded one's is announced.
        let form = ExtendedRange::of(pf).expect("an extended-range format");
        let (ours, written) = match written {
            Some(_) if computed && !super::computed::holds_as_loaded(&eq, form) => {
                ("announced".to_owned(), None)
            }
            _ => (ours, written),
        };
        let what = format!(
            "{} {} cfg {}",
            row.format,
            if row.half { "h" } else { "f" },
            row.config
        );
        if ours != row.ours {
            wrong.push(format!("{what}: ours {ours}, the table {}", row.ours));
        }
        let Some(written) = written else { continue };
        let layout = xr_blends::Layout::of(row.format).expect("a layout");
        let (plane, alpha) = xr_blends::texels(row.format);
        let sources = xr_blends::sources(row.half);
        // The second outputs, which the second source's configurations read (rounded to `half`).
        let seconds = xr_blends::second_sources(row.half);
        let rows = constant_words(c.color);
        let float =
            Blend::for_output(&eq, form, written, (Some(output), math)).is_ok_and(|b| b.float);
        let lanes: Vec<[u32; 20]> = (0..xr_blends::BAND)
            .map(|i| {
                let texel = plane.chunks(layout.bytes()).nth(i).unwrap_or_default();
                let mut words = [0u32; 20];
                words[..4].copy_from_slice(&table_texel(layout, texel, alpha.get(i)));
                let s: [f32; 4] = std::array::from_fn(|k| match row.half {
                    true => {
                        let b: [u8; 2] = std::array::from_fn(|j| sources[8 * i + 2 * k + j]);
                        half(u16::from_le_bytes(b))
                    }
                    // A `float4` output's own model reads it unrounded; the rounds it first.
                    false => {
                        let b: [u8; 4] = std::array::from_fn(|j| sources[16 * i + 4 * k + j]);
                        let v = f32::from_le_bytes(b);
                        match float {
                            true => v,
                            false => half(crate::float16::f32_to_f16_bits(v)),
                        }
                    }
                });
                for (w, x) in words[4..8].iter_mut().zip(s) {
                    *w = x.to_bits();
                }
                words[8..16].copy_from_slice(&rows);
                for (k, w) in words[16..].iter_mut().enumerate() {
                    let s1 = match row.half {
                        true => {
                            let b: [u8; 2] = std::array::from_fn(|j| seconds[8 * i + 2 * k + j]);
                            half(u16::from_le_bytes(b))
                        }
                        false => {
                            let b: [u8; 4] = std::array::from_fn(|j| seconds[16 * i + 4 * k + j]);
                            half(crate::float16::f32_to_f16_bits(f32::from_le_bytes(b)))
                        }
                    };
                    *w = s1.to_bits();
                }
                words
            })
            .collect();
        let out = run(form, Op::BlendTexel(eq, written, output, math), &lanes);
        let (mut got_plane, mut got_alpha) = (Vec::new(), Vec::new());
        for texel in out {
            table_bytes(layout, texel, &mut got_plane, &mut got_alpha);
        }
        let got_alpha = match got_alpha.is_empty() {
            true => 0,
            false => xr_blends::fnv(&got_alpha),
        };
        checked += 1;
        if (xr_blends::fnv(&got_plane), got_alpha) != (row.plane, row.alpha) {
            wrong.push(format!(
                "{what}: {eq:?} {mask:?} differs from Metal's bytes"
            ));
        }
    }
    assert!(checked > 0, "shard {shard} of {shards} ran no blend");
    assert!(
        wrong.is_empty(),
        "{} of the {} rows of shard {shard} of {shards} ({checked} blends run):\n{}",
        wrong.len(),
        t.rows.iter().skip(shard).step_by(shards).count(),
        wrong.join("\n")
    );
}

/// Metal's `float` of bfloat code `c`, as bits (`metal::format::ExtendedRange::R16Bfloat`): the
/// code shifted up 16, any NaN code the quiet NaN.
fn bfloat_read(c: u32) -> u32 {
    match c & 0x7FFF > 0x7F80 {
        true => 0x7FC0_0000,
        false => c << 16,
    }
}

/// Metal's code of a kernel's write of the `f32` of bits `b` to a bfloat texture: its top 16 bits,
/// a NaN `0x7fc0`.
fn bfloat_write(b: u32) -> u32 {
    match b & 0x7FFF_FFFF > 0x7F80_0000 {
        true => 0x7FC0,
        false => b >> 16,
    }
}

/// Every code of each bfloat form in each channel reads Metal's `float` and its `half` (rounded to
/// nearest even, overflows to the infinities), the channels a form lacks 0 and alpha 1 whatever
/// their words hold; a write stores Metal's codes: per code its bits and its tie's neighbours,
/// every eighth `half` widened, and the specials, the components a form lacks zero (an Apple M4:
/// every code, read and written, in each bfloat format).
#[test]
fn bfloat_forms_read_and_write_metals_codes() {
    let forms = [
        (ExtendedRange::R16Bfloat, 1usize, 4096u32),
        (ExtendedRange::Rg16Bfloat, 2, 4096),
        (ExtendedRange::Rgba16Bfloat, 4, 65536),
    ];
    let one = 1.0f32.to_bits();
    for (form, ch, n) in forms {
        let step = 65536 / n;
        // Every code (every `step`-th in the narrower forms) in each channel; the words a form
        // lacks hold junk.
        let lanes: Vec<[u32; 4]> = (0..n)
            .map(|i| {
                std::array::from_fn(|c| {
                    let code = (i * step + 16411 * c as u32 + c as u32) & 0xFFFF;
                    match c < ch {
                        true => code,
                        false => code | 0xABC0_0000,
                    }
                })
            })
            .collect();
        let floats = run(form, Op::Decode, &lanes);
        let halves = run(form, Op::DecodeHalf, &lanes);
        for ((l, f), h) in lanes.iter().zip(&floats).zip(&halves) {
            let want: [u32; 4] = std::array::from_fn(|c| match (c < ch, c) {
                (true, _) => bfloat_read(l[c]),
                (false, 3) => one,
                (false, _) => 0,
            });
            assert_eq!(*f, want, "{form:?} words {l:x?}");
            let half = |b: u32| {
                let h = crate::float16::f32_to_f16_bits(f32::from_bits(b));
                crate::float16::f16_bits_to_f32(h).to_bits()
            };
            assert_eq!(*h, want.map(half), "{form:?} half, words {l:x?}");
        }
        // Writes: per code its bits and the neighbours of its tie, every eighth `half`, and the
        // specials, `value k` in channel `k % 4` of invocation `k / 4`.
        let mut values: Vec<u32> = (0..65536u32)
            .step_by(16)
            .flat_map(|c| [0, 0x7FFF, 0x8000, 0x8001, 0xFFFF].map(|low| c << 16 | low))
            .collect();
        values.extend(
            (0..65536u32)
                .step_by(8)
                .map(|h| crate::float16::f16_bits_to_f32(h as u16).to_bits()),
        );
        values.extend([
            f32::NAN.to_bits(),
            (-f32::NAN).to_bits(),
            0x7F80_0001,
            0xFFFF_FFFF,
            f32::INFINITY.to_bits(),
            f32::NEG_INFINITY.to_bits(),
            1,
            0x8000_0001,
        ]);
        while !values.len().is_multiple_of(4) {
            values.push(0);
        }
        let lanes: Vec<[u32; 4]> = values
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| [c[0], c[1], c[2], c[3]])
            .collect();
        let got = run(form, Op::Encode, &lanes);
        for (l, g) in lanes.iter().zip(&got) {
            let want: [u32; 4] = std::array::from_fn(|c| match c < ch {
                true => bfloat_write(l[c]),
                false => 0,
            });
            assert_eq!(*g, want, "{form:?} values {l:x?}");
        }
    }
}

/// FNV-1a of Metal's codes (each little-endian, 2 bytes) of a fragment function's `float` outputs
/// to a bfloat attachment, at the default `textureWriteRoundingMode` and at `NearestEven`: per
/// code `c` the `float`s of bits `c << 16`, and `| 0x7fff`, `| 0x8000` and `| 0x8001`, in that
/// order; every form's codes are these (an Apple M4 under macOS 27.0.1, each
/// value into `R16Bfloat`, `RG16Bfloat` and `RGBA16Bfloat`).
const METAL_FLOAT_OUTPUTS: [(bool, u64); 2] = [
    (false, 0x4334_841e_f3f2_1ba5),
    (true, 0x5142_c964_386f_8c65),
];

/// FNV-1a of Metal's codes of a fragment function's `half` outputs to a bfloat attachment, every
/// `half` code in order, at either mode.
const METAL_HALF_OUTPUTS: u64 = 0x9eb0_0b77_54b2_35e5;

/// A fragment function's `float` and `half` outputs to a bfloat attachment store Metal's codes:
/// a `float` rounded to `half` toward zero (65504 at most) or, at `NearestEven`, to nearest even,
/// then the bfloat code nearest, ties to even, a NaN `0x7fc0`; every `float` around each code's
/// tie and every `half` code against Metal's digests, the narrower forms keeping their lanes, and
/// named answers.
#[test]
fn bfloat_attachments_store_metals_codes() {
    let digest = |got: &[[u32; 4]]| {
        assert!(got.iter().flatten().all(|&c| c <= 0xFFFF));
        let bytes: Vec<u8> = got
            .iter()
            .flatten()
            .flat_map(|&c| (c as u16).to_le_bytes())
            .collect();
        xr_blends::fnv(&bytes)
    };
    let floats: Vec<[u32; 4]> = (0..65536u32)
        .map(|c| [0, 0x7FFF, 0x8000, 0x8001].map(|low| c << 16 | low))
        .collect();
    let halves: Vec<[u32; 4]> = (0..16384u32)
        .map(|i| std::array::from_fn(|k| half(4 * i as u16 + k as u16).to_bits()))
        .collect();
    for (nearest_even, metal) in METAL_FLOAT_OUTPUTS {
        let form = ExtendedRange::Rgba16Bfloat;
        let got = run(form, Op::Output(nearest_even, FloatFmt::F32), &floats);
        assert_eq!(
            digest(&got),
            metal,
            "float outputs, nearest even {nearest_even}"
        );
        let got = run(form, Op::Output(nearest_even, FloatFmt::Half), &halves);
        assert_eq!(digest(&got), METAL_HALF_OUTPUTS, "half outputs");
        // The narrower forms keep their lanes of the same codes, the rest zero.
        let sample: Vec<[u32; 4]> = floats.iter().copied().step_by(61).collect();
        let wide = run(form, Op::Output(nearest_even, FloatFmt::F32), &sample);
        for (narrow, kept) in [
            (ExtendedRange::R16Bfloat, 1),
            (ExtendedRange::Rg16Bfloat, 2),
        ] {
            let got = run(narrow, Op::Output(nearest_even, FloatFmt::F32), &sample);
            for (g, w) in got.iter().zip(&wide) {
                let want: [u32; 4] = std::array::from_fn(|k| if k < kept { w[k] } else { 0 });
                assert_eq!(*g, want, "{narrow:?}");
            }
        }
    }
    // Named answers: `float` bits, the default mode's code and `NearestEven`'s.
    let named: [(u32, u32, u32); 12] = [
        (0x477F_F000, 0x4780, 0x7F80),
        (0xC77F_F000, 0xC780, 0xFF80),
        (0x477F_EFFF, 0x4780, 0x4780),
        (0x3F81_7FFF, 0x3F81, 0x3F82),
        (0x3F81_8000, 0x3F82, 0x3F82),
        (0x3F80_8000, 0x3F80, 0x3F80),
        (0x3300_0001, 0, 0x3380),
        (0x3300_0000, 0, 0),
        (0x8000_0000, 0x8000, 0x8000),
        (0xFF80_0000, 0xFF80, 0xFF80),
        (0x7FC0_0001, 0x7FC0, 0x7FC0),
        (0xFFC0_0000, 0x7FC0, 0x7FC0),
    ];
    let lanes: Vec<[u32; 4]> = named.iter().map(|&(b, ..)| [b; 4]).collect();
    for nearest_even in [false, true] {
        let got = run(
            ExtendedRange::Rgba16Bfloat,
            Op::Output(nearest_even, FloatFmt::F32),
            &lanes,
        );
        for (&(b, default, nearest), g) in named.iter().zip(&got) {
            let want = if nearest_even { nearest } else { default };
            assert_eq!(g[0], want, "{b:#x}, nearest even {nearest_even}");
        }
    }
}

/// A read of what the tile holds clamps as Metal's does (an Apple M4: unblended
/// `half` writes read back by a fetch): each color lane to the form's source range, a NaN to its
/// low end, the infinities and values past the range to its ends; alpha to `[0, 1]`, a NaN to 0;
/// a subnormal `half` kept; `BGR10_XR`'s alpha one.
#[test]
fn a_read_of_the_tile_clamps_as_metals_does() {
    let (n, inf) = (f32::NAN, f32::INFINITY);
    let subnormal = crate::float16::f16_bits_to_f32(7);
    let written: [[f32; 4]; 5] = [
        [n, -inf, inf, -0.5],
        [inf, n, -inf, n],
        [-inf, 3.0, n, 2.0],
        [10000.0, -3.0, 0.25, inf],
        [2.0, -1.0, subnormal, 0.5],
    ];
    for form in [
        ExtendedRange::Bgra10 { srgb: false },
        ExtendedRange::Bgra10 { srgb: true },
        ExtendedRange::Bgr10 { srgb: true },
        ExtendedRange::Planes { srgb: false },
    ] {
        let (lo, hi) = match form {
            ExtendedRange::Bgra10 { srgb: true } => SRGB_SOURCE_RANGE,
            _ => SOURCE_RANGE,
        };
        let a = |v: f32| match form {
            ExtendedRange::Bgr10 { .. } => 1.0,
            _ => v,
        };
        let want: [[f32; 4]; 5] = [
            [lo, lo, hi, a(0.0)],
            [hi, lo, lo, a(0.0)],
            [lo, hi, lo, a(1.0)],
            [hi, lo, 0.25, a(1.0)],
            [hi, lo, subnormal, a(0.5)],
        ];
        let lanes: Vec<[u32; 4]> = written.iter().map(|w| w.map(f32::to_bits)).collect();
        let got = run(form, Op::TileRead, &lanes);
        for ((w, g), want) in written.iter().zip(&got).zip(&want) {
            assert_eq!(*g, want.map(f32::to_bits), "{form:?}: {w:?}");
        }
    }
}
