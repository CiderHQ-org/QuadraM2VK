//! The color resolve modules: Metal's measured tables as the module docs state them, valid on
//! every committed profile (L3, `G-val`), needing nothing of a device, and exact on the CPU
//! interpreter: the fragment stage's arithmetic, run in a kernel over every pair of 8-bit codes and
//! over random and edge texels, writes the float the attachment stores as Metal's code.

use super::*;
use crate::device::{DeviceDescription, DevicePolicy};
use crate::spirv::interp::{run, Buffer, Config, Order, Outcome, Resources};
use crate::spirv::testing::{assert_valid, committed_profiles};
use std::path::PathBuf;

/// Every format the module docs list.
const CLOSED: [Format; 35] = [
    Format::R8_UNORM,
    Format::A8_UNORM,
    Format::R8G8_UNORM,
    Format::R8G8B8A8_UNORM,
    Format::B8G8R8A8_UNORM,
    Format::R8_SRGB,
    Format::R8G8_SRGB,
    Format::R8G8B8A8_SRGB,
    Format::B8G8R8A8_SRGB,
    Format::R8_SNORM,
    Format::R8G8_SNORM,
    Format::R8G8B8A8_SNORM,
    Format::R16_UNORM,
    Format::R16G16_UNORM,
    Format::R16G16B16A16_UNORM,
    Format::R16_SNORM,
    Format::R16G16_SNORM,
    Format::R16G16B16A16_SNORM,
    Format::A2B10G10R10_UNORM_PACK32,
    Format::A2R10G10B10_UNORM_PACK32,
    Format::R32_SFLOAT,
    Format::R32G32_SFLOAT,
    Format::R32G32B32A32_SFLOAT,
    Format::R16_SFLOAT,
    Format::R16G16_SFLOAT,
    Format::R16G16B16A16_SFLOAT,
    Format::R5G6B5_UNORM_PACK16,
    Format::B5G6R5_UNORM_PACK16,
    Format::R5G5B5A1_UNORM_PACK16,
    Format::B5G5R5A1_UNORM_PACK16,
    Format::A1R5G5B5_UNORM_PACK16,
    Format::R4G4B4A4_UNORM_PACK16,
    Format::B4G4R4A4_UNORM_PACK16,
    Format::B10G11R11_UFLOAT_PACK32,
    Format::E5B9G9R9_UFLOAT_PACK32,
];

/// Every format the module docs list, and a few they leave out.
#[test]
fn the_formats_are_the_ones_metals_rules_close() {
    for format in CLOSED {
        let resolve = ColorResolveFormat::of(format).expect("closed");
        assert_eq!(resolve.format(), format);
        // Each channel lies inside the texel, apart from every other.
        let size = format.block().expect("block").size * 8;
        let mut used = 0u128;
        for channel in resolve.channels().iter().flatten() {
            let bits = u32::from(channel.rule.bits());
            let offset = u32::from(channel.offset);
            assert!(offset + bits <= size, "{format:?}");
            let mask = ((1u128 << bits) - 1) << offset;
            assert_eq!(used & mask, 0, "{format:?}");
            used |= mask;
            assert_eq!(
                offset / 32,
                (offset + bits - 1) / 32,
                "{format:?}: one word"
            );
        }
    }
    for format in [
        Format::A4R4G4B4_UNORM_PACK16,
        Format::A2B10G10R10_SNORM_PACK32,
        Format::R8_UINT,
        Format::R32_UINT,
        Format::D32_SFLOAT,
    ] {
        assert_eq!(ColorResolveFormat::of(format), None, "{format:?}");
    }
}

/// The sRGB tables are what the module docs say: the 12-bit linear value of each code rounded,
/// and Metal's encode, ascending, the exact encode rounded half up but at the four entries
/// measured one code higher.
#[test]
fn the_srgb_tables_are_metals_measured_ones() {
    for (code, &linear) in SRGB_RESOLVE_LINEAR.iter().enumerate() {
        assert_eq!(
            f64::from(linear),
            (srgb_decode(code as u32) * 4095.0).round(),
            "{code}"
        );
    }
    assert!(SRGB_RESOLVE_THRESHOLDS.windows(2).all(|w| w[0] < w[1]));
    let encode = |linear: u16| SRGB_RESOLVE_THRESHOLDS.partition_point(|&t| t <= linear);
    let mut higher = Vec::new();
    for linear in 0..4096u16 {
        let l = f64::from(linear) / 4095.0;
        let exact = if l <= 0.003_130_8 {
            l * 12.92
        } else {
            1.055 * l.powf(1.0 / 2.4) - 0.055
        };
        let rounded = (exact * 255.0 + 0.5).floor() as usize;
        match encode(linear) {
            code if code == rounded => {}
            code if code == rounded + 1 => higher.push(linear),
            code => panic!("{linear}: {code} against the exact encode's {rounded}"),
        }
    }
    assert_eq!(higher, [804, 1371, 2121, 2569]);
}

/// The sources a test reads `format` from: the input attachment, in its own format and, where that
/// is another, its raw twin's, and the sample planes in `copy`.
fn sources(format: ColorResolveFormat, copy: PlanesCopy) -> Vec<ResolveSource> {
    let mut sources = vec![ResolveSource::InputAttachment];
    if format.raw_format() != format.format() {
        sources.push(ResolveSource::RawInputAttachment);
    }
    sources.push(ResolveSource::Planes(copy));
    sources
}

/// The resolves a test runs of `format`: its result written in its own format and, where its raw
/// twin is another, through the twin ([`ColorResolveFormat::written_raw`]), each from every source
/// ([`sources`]).
fn cases(format: ColorResolveFormat, copy: PlanesCopy) -> Vec<(ColorResolveFormat, ResolveSource)> {
    let mut written = vec![format];
    if format.written_raw() != format {
        written.push(format.written_raw());
    }
    written
        .into_iter()
        .flat_map(|format| sources(format, copy).into_iter().map(move |s| (format, s)))
        .collect()
}

/// The requirements of `m` as serialized.
fn needs(m: TypedModule) -> Requirements {
    crate::spirv::serialize::serialize(m)
        .expect("serialize")
        .requirements
}

/// A format of each rule, and `A8_UNORM` (its planes held in its own format), each with its sample
/// planes' copy format.
fn representatives() -> Vec<(ColorResolveFormat, PlanesCopy)> {
    [
        Format::R8G8B8A8_UNORM,
        Format::B8G8R8A8_SRGB,
        Format::R16G16_SNORM,
        Format::A2B10G10R10_UNORM_PACK32,
        Format::R32G32B32A32_SFLOAT,
        Format::R16G16B16A16_SFLOAT,
        Format::R5G6B5_UNORM_PACK16,
        Format::A1R5G5B5_UNORM_PACK16,
        Format::B10G11R11_UFLOAT_PACK32,
        Format::E5B9G9R9_UFLOAT_PACK32,
        Format::A8_UNORM,
    ]
    .into_iter()
    .map(|format| {
        (
            ColorResolveFormat::of(format).expect("closed"),
            PlanesCopy::for_planes(format).expect("copy"),
        )
    })
    .collect()
}

/// The fragment stage needs nothing of a device beyond Vulkan 1.2's shaders, at every sample count
/// its rules hold at and from every source.
#[test]
fn the_fragment_stage_needs_nothing_of_a_device() {
    for (format, copy) in representatives() {
        for samples in [2, 4, 8].into_iter().filter(|&n| format.resolves(n)) {
            for (format, source) in cases(format, copy) {
                let r = needs(fragment_typed_module(format, samples, source).expect("module"));
                assert!(
                    r.features.is_empty(),
                    "{format:?} x{samples} {source:?}: {r:?}"
                );
            }
        }
    }
}

/// Every committed profile validates each representative's modules.
#[test]
fn the_modules_are_valid_on_every_committed_profile() {
    for (name, profile) in committed_profiles().expect("profiles") {
        for (format, copy) in representatives() {
            for (format, source) in cases(format, copy) {
                for samples in [2, 4] {
                    let m = fragment_typed_module(format, samples, source).expect("module");
                    let _ = assert_valid(m, &profile);
                }
            }
            let _ = name;
        }
    }
}

/// On a committed device the modules come numbered with the fixed binding, and a sample count
/// that is no resolve's is refused.
#[test]
fn the_modules_come_numbered_and_refuse_another_count() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles/lavapipe.json");
    let text = std::fs::read_to_string(&path).expect("read");
    let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    let device = Device::new(desc, DevicePolicy::default()).expect("device");
    let (format, copy) = representatives()[0];
    for (format, source) in cases(format, copy) {
        let modules =
            color_resolve_modules(&device, format, 4, source, Feature::Format).expect("modules");
        assert!(!modules.vertex.is_empty() && !modules.fragment.is_empty());
    }
    for samples in [0, 1, 3, 128] {
        let refused = color_resolve_modules(
            &device,
            format,
            samples,
            ResolveSource::InputAttachment,
            Feature::Format,
        );
        assert!(matches!(refused, Err(Error::Internal(_))), "{samples}");
    }
}

// ---- The arithmetic on the interpreter ----

/// The kernel's buffers: each invocation's samples (four words a texel), and its four output
/// words.
const SAMPLES_IN: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 0,
};
const RESOLVED_OUT: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 1,
};

/// A kernel running the fragment stage's arithmetic: invocation `i` reads its `samples` texels as
/// four words each from `SAMPLES_IN` (the planes' bits for [`ResolveSource::Planes`] of a bits
/// copy, the attachment's floats' bits for [`ResolveSource::InputAttachment`] and for planes held
/// in `A8_UNORM`, its raw twin's for [`ResolveSource::RawInputAttachment`]), and writes the output
/// floats' bits to `RESOLVED_OUT`.
fn arithmetic_module(
    format: ColorResolveFormat,
    samples: u32,
    source: ResolveSource,
) -> TypedModule {
    let mut b = ModuleBuilder::new();
    let t = Types::new(&mut b).expect("types");
    let srgb = format
        .srgb()
        .then(|| SrgbTables::declare(&mut b, t, format, source))
        .transpose()
        .expect("tables");
    let words = super::super::visibility_resolve::words_buffer;
    let input = words(&mut b, SAMPLES_IN, t.u32_).expect("in");
    let output = words(&mut b, RESOLVED_OUT, t.u32_).expect("out");
    let uvec3 = b
        .ty(Type::Vector {
            elem: t.u32_,
            len: 3,
        })
        .expect("uvec3");
    let uvec4 = b
        .ty(Type::Vector {
            elem: t.u32_,
            len: 4,
        })
        .expect("uvec4");
    let invocation = b
        .builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)
        .expect("id");
    let void = b.ty(Type::Void).expect("void");
    let mut f = b.function(void).expect("function");
    let build = |f: &mut FunctionBuilder<'_>| -> BResult<()> {
        let zero = uint(f, t, 0)?;
        let invocation = f.module().global_value(invocation)?;
        let invocation = f.load(invocation, MemAccess::default())?;
        let i = f.composite_extract(invocation, &[0])?;
        let input = f.module().global_value(input)?;
        let output = f.module().global_value(output)?;
        let per = uint(f, t, samples * 4)?;
        let base = f.binary(BinOp::IMul, i, per)?;
        let mut texels = Vec::new();
        for s in 0..samples {
            let mut lanes = Vec::new();
            for w in 0..4 {
                let k = uint(f, t, s * 4 + w)?;
                let at = f.binary(BinOp::IAdd, base, k)?;
                let p = f.access_chain(input, &[zero, at])?;
                lanes.push(f.load(p, MemAccess::default())?);
            }
            let texel = f.composite_construct(uvec4, &lanes)?;
            texels.push(match source.reads_bits() {
                true => texel,
                false => f.convert(ConvOp::Bitcast, texel, t.vec4)?,
            });
        }
        let four = uint(f, t, 4)?;
        let out_base = f.binary(BinOp::IMul, i, four)?;
        let lanes = resolve_texel(f, t, srgb, format, &texels, source)?;
        for (component, channel) in format.channels.iter().enumerate() {
            if channel.is_none() {
                continue;
            }
            let bits = f.convert(ConvOp::Bitcast, lanes[component], t.u32_)?;
            let c = uint(f, t, component as u32)?;
            let at = f.binary(BinOp::IAdd, out_base, c)?;
            let p = f.access_chain(output, &[zero, at])?;
            f.store(p, bits, MemAccess::default())?;
        }
        Ok(())
    };
    build(&mut f).expect("body");
    f.ret(None).expect("ret");
    let id = f.finish().expect("finish");
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize { x: 1, y: 1, z: 1 }],
    )
    .expect("entry");
    b.finalize().expect("finalize")
}

/// SplitMix64.
struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// Metal's resolve of one channel's codes (the module docs), written independently of the
/// modules: a `u32` code, or the float's bits.
fn metal(rule: ResolveRule, codes: &[u32]) -> u32 {
    let n = codes.len() as i64;
    match rule {
        ResolveRule::Unorm { .. } => {
            let sum: i64 = codes.iter().map(|&c| i64::from(c)).sum();
            ((sum + n / 2) / n) as u32
        }
        ResolveRule::Snorm { bits } => {
            let half = 1i64 << (bits - 1);
            let sum: i64 = codes
                .iter()
                .map(|&c| {
                    let v = i64::from(c);
                    let v = if v >= half { v - 2 * half } else { v };
                    v.max(-(half - 1))
                })
                .sum();
            ((2 * sum + n).div_euclid(2 * n) as u32) & ((1u32 << bits) - 1)
        }
        ResolveRule::Srgb => {
            let sum: i64 = codes
                .iter()
                .map(|&c| i64::from(SRGB_RESOLVE_LINEAR[c as usize]))
                .sum();
            let linear = ((sum + n / 2) / n) as u16;
            SRGB_RESOLVE_THRESHOLDS.partition_point(|&t| t <= linear) as u32
        }
        ResolveRule::Float32 => {
            let flush = |v: f32| {
                if v.is_subnormal() {
                    0.0f32.copysign(v)
                } else {
                    v
                }
            };
            let mut sum = flush(f32::from_bits(codes[0]));
            for &c in &codes[1..] {
                sum = flush(sum + flush(f32::from_bits(c)));
            }
            let resolved = flush(sum * (1.0 / n as f32));
            if resolved.is_nan() {
                0x7fc0_0000
            } else {
                resolved.to_bits()
            }
        }
        ResolveRule::HalfUnorm { bits } => half_unorm_model(bits, codes),
        ResolveRule::Float16 => half_rule(codes),
        ResolveRule::SmallFloat { mantissa } => small_rule(mantissa, codes),
        ResolveRule::SharedExponent => panic!("a shared exponent resolves by texel"),
    }
}

/// The exponent of a nonzero finite `f64` (its leading bit's), from its bits.
fn exponent(x: f64) -> i32 {
    let field = ((x.abs().to_bits() >> 52) & 0x7ff) as i32;
    assert!(field != 0 && field != 0x7ff, "{x}");
    field - 1023
}

/// 2^k, for a `k` of a normal `f64`.
fn pow2(k: i32) -> f64 {
    f64::from_bits(((k + 1023) as u64) << 52)
}

/// Metal's resolve of a unorm channel of `bits` bits read through a `half`, from the definition
/// the module docs give (not from the deficits mask): each code `c` of `m` becomes the `half`
/// nearest `c / m` (ties to even), times `m` rounded toward zero to a `half`; the average of those
/// rounds half up. In `f64`, where `c / m` lies far from every tie of the `half` grid (`m` is odd)
/// and each other step is exact.
fn half_unorm_model(bits: u8, codes: &[u32]) -> u32 {
    let m = f64::from((1u32 << bits) - 1);
    let sum: f64 = codes
        .iter()
        .map(|&c| {
            let v = f64::from(c) / m;
            if v == 0.0 {
                return 0.0;
            }
            let step = pow2(exponent(v) - 10);
            let x = (v / step).round_ties_even() * step;
            let product = x * m;
            let step = pow2(exponent(product) - 10);
            (product / step).floor() * step
        })
        .sum();
    (sum / codes.len() as f64 + 0.5).floor() as u32
}

/// The exponent of `v` units of 2^-28 (its leading bit's), -100 for zero.
fn fine_exponent(v: i64) -> i32 {
    if v == 0 {
        return -100;
    }
    63 - v.unsigned_abs().leading_zeros() as i32 - 28
}

/// `v` units of 2^-28 floored to the multiple of 2^(e - 14).
fn floored(v: i64, e: i32) -> i64 {
    let step = 1i64 << (e + 14);
    v.div_euclid(step) * step
}

/// Metal's 2- and 4-sample half resolve (the module docs) in exact integers, units of 2^-28: each
/// half's value (a NaN 0, an infinity 2^16 of its sign) and exponent (at least -14); at 2 the two
/// floored to the larger exponent's grid and summed; at 4 samples 0 and 1 so summed into `A`,
/// whose exponent is its leading bit's (one lower for a negative power of two) but at most the
/// larger of theirs, and `A` and samples 2 and 3 floored to the largest exponent's grid and
/// summed; the average rounded to the `half` grid with a tie toward +∞. The half's bits.
fn half_rule(codes: &[u32]) -> u32 {
    let fine: Vec<(i64, i32)> = codes
        .iter()
        .map(|&h| {
            let v = half_units(h) * 16;
            (v, fine_exponent(v).max(-14))
        })
        .collect();
    let total = match *fine.as_slice() {
        [(a, ea), (b, eb)] => {
            let e = ea.max(eb);
            floored(a, e) + floored(b, e)
        }
        [(a, ea), (b, eb), (c, ec), (d, ed)] => {
            let pair = ea.max(eb);
            let first = floored(a, pair) + floored(b, pair);
            let mut e_first = fine_exponent(first);
            if first < 0 && first.unsigned_abs().is_power_of_two() {
                e_first -= 1;
            }
            let e = e_first.min(pair).max(ec).max(ed);
            floored(first, e) + floored(c, e) + floored(d, e)
        }
        _ => panic!(
            "a half's rule holds at 2 and 4 samples, not {}",
            codes.len()
        ),
    };
    let n = codes.len() as i64;
    // The grid's step in units of 2^-28: 2^-24 below 2^-13 (the subnormals and the first
    // binade), else the binade's.
    let mut step = 16i64;
    while total.abs() >= n * (step << 11) {
        step <<= 1;
    }
    let q = (2 * total + n * step).div_euclid(2 * n * step);
    half_of_units(q * step / 16)
}

/// A small float's bits of `mantissa` bits for the value `v` it holds exactly (+∞ too).
fn small_bits(v: f64, mantissa: u8) -> u32 {
    let m = i32::from(mantissa);
    if v.is_infinite() {
        assert!(v > 0.0);
        return 31 << m;
    }
    if v == 0.0 {
        return 0;
    }
    let e = exponent(v);
    if e < -14 {
        let units = v / pow2(-14 - m);
        assert_eq!(units.fract(), 0.0, "{v} is no small float");
        return units as u32;
    }
    let units = v / pow2(e - m);
    assert_eq!(units.fract(), 0.0, "{v} is no small float");
    (((e + 15) as u32) << m) | (units as u32 - (1 << m))
}

/// Metal's resolve of a small float of `mantissa` bits (the module docs): the samples' half
/// resolve (a small float's bits are a half's leading bits), an infinity infinite, else that half
/// rounded to nearest even onto the small float's grid and held to its largest finite value. The
/// small float's bits.
fn small_rule(mantissa: u8, codes: &[u32]) -> u32 {
    let up = 10 - u32::from(mantissa);
    let halves: Vec<u32> = codes.iter().map(|&c| c << up).collect();
    let h = half_rule(&halves);
    if h == 0x7c00 {
        return 31 << mantissa;
    }
    let v = f64::from(half_float(h));
    assert!(v >= 0.0, "{h:#x}");
    let rounded = if v == 0.0 {
        0.0
    } else {
        let step = pow2(exponent(v).max(-14) - i32::from(mantissa));
        (v / step).round_ties_even() * step
    };
    let largest = (2.0 - pow2(-i32::from(mantissa))) * 32768.0;
    small_bits(rounded.min(largest), mantissa)
}

/// The three channels (R, G, B) of a shared-exponent texel `word`.
fn shared_values(word: u32) -> [f64; 3] {
    let scale = pow2((word >> 27) as i32 - 24);
    [0, 9, 18].map(|at| f64::from((word >> at) & 0x1ff) * scale)
}

/// `EXT_texture_shared_exponent`'s encode of `channels`, as the values the texel holds.
fn shared_encode(channels: [f64; 3]) -> [f64; 3] {
    let channels = channels.map(|v| v.clamp(0.0, 65408.0));
    let most = channels[0].max(channels[1]).max(channels[2]);
    let mut e = if most > 0.0 {
        exponent(most).max(-16)
    } else {
        -16
    } + 16;
    if (most / pow2(e - 24) + 0.5).floor() == 512.0 {
        e += 1;
    }
    let step = pow2(e - 24);
    channels.map(|v| (v / step + 0.5).floor() * step)
}

/// Metal's resolve of shared-exponent texels `words` (the module docs): each channel's half
/// resolve, then the encode. The values the resolved texel holds.
fn shared_rule(words: &[u32]) -> [f64; 3] {
    let values: Vec<[f64; 3]> = words.iter().map(|&w| shared_values(w)).collect();
    let resolved = [0, 1, 2].map(|c| {
        let halves: Vec<u32> = values.iter().map(|v| float_half(v[c] as f32)).collect();
        f64::from(half_float(half_rule(&halves)))
    });
    shared_encode(resolved)
}

/// A half's value in units of 2^-24, exactly: a NaN 0, an infinity 2^16 of its sign.
fn half_units(h: u32) -> i64 {
    let sign = if h & 0x8000 != 0 { -1 } else { 1 };
    let e = (h >> 10) & 0x1f;
    let m = i64::from(h & 0x3ff);
    sign * match (e, m) {
        (31, 0) => 1 << 40,
        (31, _) => 0,
        (0, _) => m,
        _ => (1024 + m) << (e - 1),
    }
}

/// Metal's 2-sample half resolve in exact integers (the module docs): the average of the units
/// rounded to the `half` grid with a tie toward +∞, 2^16 and more an infinity, a zero +0; the
/// half's bits.
fn half_average(codes: &[u32]) -> u32 {
    let n = codes.len() as i64;
    let sum: i64 = codes.iter().map(|&c| half_units(c)).sum();
    // The grid's step in units: 1 below 2^10 units (the subnormals and the first binade), else
    // the binade's.
    let mut step = 1i64;
    while sum.abs() >= n * (step << 11) {
        step <<= 1;
    }
    // floor(sum / (n · step) + 1/2), in integers.
    let q = (2 * sum + n * step).div_euclid(2 * n * step);
    half_of_units(q * step)
}

/// The half's bits of `v` units of 2^-24 on the `half` grid: 2^16 and more an infinity, a zero +0.
fn half_of_units(v: i64) -> u32 {
    let sign = if v < 0 { 0x8000 } else { 0 };
    let a = v.unsigned_abs();
    if a >= 1 << 40 {
        return sign | 0x7c00;
    }
    if a == 0 {
        return 0;
    }
    if a < 1024 {
        return sign | a as u32;
    }
    let e = 64 - a.leading_zeros() - 11;
    let m = (a >> e) as u32 - 1024;
    sign | ((e + 1) << 10) | m
}

/// The `float` a half's bits are.
fn half_float(h: u32) -> f32 {
    let e = (h >> 10) & 0x1f;
    let negative = h & 0x8000 != 0;
    let v = match (e, h & 0x3ff) {
        (31, 0) => f32::INFINITY,
        (31, _) => f32::NAN,
        _ => half_units(h) as f32 * (2.0f32).powi(-24),
    };
    if negative && v == 0.0 {
        -0.0
    } else if negative && e == 31 {
        -v.abs()
    } else {
        v
    }
}

/// The half bits of a `float` a half represents exactly (an infinity too).
fn float_half(x: f32) -> u32 {
    let sign = if x.is_sign_negative() { 0x8000 } else { 0 };
    if x.is_infinite() {
        return sign | 0x7c00;
    }
    let units = f64::from(x.abs()) * f64::from(1u32 << 24);
    assert_eq!(units.fract(), 0.0, "{x} is no half");
    let a = units as u64;
    if a < 1024 {
        return sign | a as u32;
    }
    let e = 64 - a.leading_zeros() - 11;
    assert_eq!(a & ((1 << e) - 1), 0, "{x} is no half");
    sign | ((e + 1) << 10) | ((a >> e) as u32 - 1024)
}

/// The float an attachment of `rule` reads as `code`.
fn read_as(rule: ResolveRule, code: u32) -> f32 {
    match rule {
        ResolveRule::Unorm { bits } | ResolveRule::HalfUnorm { bits } => {
            (f64::from(code) / ((1u64 << bits) - 1) as f64) as f32
        }
        ResolveRule::Snorm { bits } => {
            let half = 1i64 << (bits - 1);
            let v = i64::from(code);
            let v = if v >= half { v - 2 * half } else { v };
            (v.max(-(half - 1)) as f64 / (half - 1) as f64) as f32
        }
        ResolveRule::Srgb => srgb_decode(code) as f32,
        ResolveRule::Float32 => f32::from_bits(code),
        ResolveRule::Float16 => half_float(code),
        ResolveRule::SmallFloat { mantissa } => half_float(code << (10 - u32::from(mantissa))),
        other => panic!("no per-channel read of {other:?}"),
    }
}

/// The code an attachment of `rule` stores for the output float `x`, converting to nearest.
fn stored_as(rule: ResolveRule, x: f32) -> u32 {
    let x = f64::from(x);
    match rule {
        ResolveRule::Unorm { bits } | ResolveRule::HalfUnorm { bits } => {
            (x * ((1u64 << bits) - 1) as f64).round() as u32
        }
        ResolveRule::Snorm { bits } => {
            let v = (x * ((1i64 << (bits - 1)) - 1) as f64).round() as i64;
            (v as u32) & ((1u32 << bits) - 1)
        }
        ResolveRule::Srgb => (0..256u32)
            .min_by(|&a, &b| {
                let da = (srgb_decode(a) - x).abs();
                let db = (srgb_decode(b) - x).abs();
                da.total_cmp(&db)
            })
            .expect("a code"),
        ResolveRule::Float32 => (x as f32).to_bits(),
        ResolveRule::Float16 => float_half(x as f32),
        ResolveRule::SmallFloat { mantissa } => small_bits(x, mantissa),
        other => panic!("no per-channel store of {other:?}"),
    }
}

/// Half bits that reach every case of the half rule: any, specials, the largest, subnormals and
/// near neighbours of the previous draw (ties).
fn half_bits(r: &mut SplitMix) -> u32 {
    let pick = r.next();
    let sign = ((pick >> 8) & 1) as u32;
    let any = (r.next() % 0x7c00) as u32;
    let bits = match pick % 10 {
        0 => [0x7c00, 0x7e00, 0x7c01, 0x7fff][(r.next() % 4) as usize],
        1 => 0x7bff - (r.next() % 4) as u32,
        2 => (r.next() % 0x800) as u32,
        3 => 0,
        _ => any,
    };
    (sign << 15) | bits
}

/// Float bits that reach every rule.
fn float_bits(r: &mut SplitMix) -> u32 {
    let pick = r.next();
    let sign = ((pick >> 8) & 1) as u32;
    let mantissa = (r.next() & 0x7f_ffff) as u32;
    let bits = match pick % 12 {
        0 => mantissa.max(1),
        1 => 0,
        2 => 0x7f80_0000,
        3 => 0x7f80_0000 | mantissa.max(1),
        4 => 0x7f7f_ffff - (r.next() % 4) as u32,
        5 => 0x0080_0000 + (r.next() % 64) as u32,
        6 => (1 << 23) | mantissa,
        _ => (r.next() % 0x7f80_0000) as u32,
    };
    (sign << 31) | bits
}

/// The kernel's input words of `texels` (each sample's codes by component, one texel a row) of
/// `format` as read from `source`.
fn sample_words(
    format: ColorResolveFormat,
    source: ResolveSource,
    texels: &[Vec<[u32; 4]>],
) -> Vec<u8> {
    let mut input = Vec::new();
    for texel in texels {
        for sample in texel {
            let mut words = [0u32; 4];
            for (component, channel) in format.channels.iter().enumerate() {
                let Some(channel) = channel else { continue };
                let code = sample[component];
                match source {
                    _ if source.reads_bits() => {
                        let offset = u32::from(channel.offset);
                        words[(offset / 32) as usize] |= code << (offset % 32);
                    }
                    // The raw twin reads an sRGB code as an 8-bit unorm one.
                    ResolveSource::RawInputAttachment if channel.rule == ResolveRule::Srgb => {
                        words[component] = read_as(ResolveRule::Unorm { bits: 8 }, code).to_bits()
                    }
                    _ => words[component] = read_as(channel.rule, code).to_bits(),
                }
            }
            input.extend(words.iter().flat_map(|w| w.to_le_bytes()));
        }
    }
    input
}

/// Run `format`'s arithmetic at `samples` from `source` over the kernel's `input` words of
/// `texels` texels.
fn dispatch(
    format: ColorResolveFormat,
    samples: usize,
    source: ResolveSource,
    input: Vec<u8>,
    texels: usize,
) -> Outcome {
    let m = arithmetic_module(format, samples as u32, source);
    let res = Resources {
        buffers: vec![
            Buffer {
                slot: Some(SAMPLES_IN),
                address: None,
                bytes: input,
            },
            Buffer {
                slot: Some(RESOLVED_OUT),
                address: None,
                bytes: vec![0; texels * 16],
            },
        ],
        push: Vec::new(),
        spec: Default::default(),
    };
    let cfg = Config {
        order: Order::Forward,
        workgroups: [texels as u32, 1, 1],
        ..Config::default()
    };
    run(&m, &cfg, res).expect("run")
}

/// Run `format`'s arithmetic at `samples` from `source` over `texels` (each sample's codes by
/// component, one texel a row), and check every channel against [`metal`].
fn check(
    format: ColorResolveFormat,
    samples: usize,
    source: ResolveSource,
    texels: &[Vec<[u32; 4]>],
) {
    let input = sample_words(format, source, texels);
    let out = dispatch(format, samples, source, input, texels.len())
        .buffers
        .remove(1);
    for (i, texel) in texels.iter().enumerate() {
        for (component, channel) in format.channels.iter().enumerate() {
            let Some(channel) = channel else { continue };
            let at = (i * 4 + component) * 4;
            let word = u32::from_le_bytes(out[at..at + 4].try_into().expect("word"));
            let codes: Vec<u32> = texel.iter().map(|s| s[component]).collect();
            let want = metal(channel.rule, &codes);
            // Written through the raw twin, an sRGB channel's float is stored as an 8-bit unorm
            // one.
            let stored = match channel.rule {
                ResolveRule::Srgb if format.written_raw => ResolveRule::Unorm { bits: 8 },
                rule => rule,
            };
            let got = stored_as(stored, f32::from_bits(word));
            assert_eq!(
                got,
                want,
                "{:?} written as {:?} x{samples} {source:?} component {component}: {codes:x?} \
                 wrote {word:#x}",
                format.format(),
                format.written_format()
            );
        }
    }
}

/// The codes of `rule` a test draws: random, the rule's edges, and for floats every class.
fn draw(rule: ResolveRule, r: &mut SplitMix) -> u32 {
    let bits = u32::from(rule.bits());
    match rule {
        ResolveRule::Float32 => float_bits(r),
        ResolveRule::Float16 => half_bits(r),
        ResolveRule::SmallFloat { mantissa } => small_float_bits(r, mantissa),
        _ => {
            let mask = if bits == 32 {
                u32::MAX
            } else {
                (1u32 << bits) - 1
            };
            if r.next().is_multiple_of(4) {
                let half = 1u32 << (bits - 1);
                let edges = [0, 1, half - 1, half, half + 1, mask - 1, mask];
                edges[(r.next() % edges.len() as u64) as usize] & mask
            } else {
                (r.next() as u32) & mask
            }
        }
    }
}

/// Small-float bits of `mantissa` bits that reach every case of the small-float rule: any, the
/// infinity, NaNs, the largest finite values, subnormals and zero.
fn small_float_bits(r: &mut SplitMix, mantissa: u8) -> u32 {
    let m = u32::from(mantissa);
    let inf = 31 << m;
    match r.next() % 10 {
        0 => inf,
        1 => inf | ((r.next() as u32) & ((1 << m) - 1)).max(1),
        2 => inf - 1 - (r.next() % 4) as u32,
        3 => (r.next() as u32) & ((1 << m) - 1),
        4 => 0,
        _ => (r.next() % u64::from(inf)) as u32,
    }
}

/// Every pair of 8-bit codes of `format`'s one channel at 2 samples, through every source
/// ([`sources`]).
fn every_pair_of_8_bit_codes(format: Format) {
    let resolve = ColorResolveFormat::of(format).expect("closed");
    let copy = PlanesCopy::for_planes(format).expect("copy");
    let component = resolve
        .channels()
        .iter()
        .position(Option::is_some)
        .expect("a channel");
    let sample = |code: u32| {
        let mut codes = [0; 4];
        codes[component] = code;
        codes
    };
    let texels: Vec<Vec<[u32; 4]>> = (0..65536u32)
        .map(|p| vec![sample(p & 0xff), sample(p >> 8)])
        .collect();
    for (resolve, source) in cases(resolve, copy) {
        check(resolve, 2, source, &texels);
    }
}

#[test]
fn every_pair_of_unorm_codes_resolves_as_metal() {
    every_pair_of_8_bit_codes(Format::R8_UNORM);
}

/// `A8_UNORM`'s alpha, from its input attachment and from sample planes held in its own format
/// (`PlanesCopy::Alpha8`).
#[test]
fn every_pair_of_alpha_codes_resolves_as_metal() {
    every_pair_of_8_bit_codes(Format::A8_UNORM);
}

#[test]
fn every_pair_of_srgb_codes_resolves_as_metal() {
    every_pair_of_8_bit_codes(Format::R8_SRGB);
}

#[test]
fn every_pair_of_snorm_codes_resolves_as_metal() {
    every_pair_of_8_bit_codes(Format::R8_SNORM);
}

/// Random and edge texels of every closed format at 2, 4 and 8 samples (those its rules hold at),
/// through every source ([`sources`]).
#[test]
fn random_texels_of_every_format_resolve_as_metal() {
    let mut r = SplitMix(0x0052_504c);
    for format in [
        Format::R8G8_UNORM,
        Format::B8G8R8A8_SRGB,
        Format::R8G8B8A8_SNORM,
        Format::R16G16B16A16_UNORM,
        Format::R16G16_SNORM,
        Format::A2R10G10B10_UNORM_PACK32,
        Format::R32G32B32A32_SFLOAT,
        Format::R5G6B5_UNORM_PACK16,
        Format::A1R5G5B5_UNORM_PACK16,
        Format::B4G4R4A4_UNORM_PACK16,
        Format::R16G16B16A16_SFLOAT,
        Format::B10G11R11_UFLOAT_PACK32,
    ] {
        let resolve = ColorResolveFormat::of(format).expect("closed");
        let copy = PlanesCopy::for_format(format).expect("copy");
        for samples in [2usize, 4, 8]
            .into_iter()
            .filter(|&n| resolve.resolves(n as u32))
        {
            let texels: Vec<Vec<[u32; 4]>> = (0..512)
                .map(|_| {
                    (0..samples)
                        .map(|_| {
                            let mut codes = [0u32; 4];
                            for (component, channel) in resolve.channels().iter().enumerate() {
                                if let Some(channel) = channel {
                                    codes[component] = draw(channel.rule, &mut r);
                                }
                            }
                            codes
                        })
                        .collect()
                })
                .collect();
            for (resolve, source) in cases(resolve, copy) {
                check(resolve, samples, source, &texels);
            }
        }
    }
}

/// Metal's resolves of 2-sample `RGBA16Float` channels measured on an M3 Max under macOS 26.7 (our
/// own probe `resrule.swift`): two samples' bits and the resolved bits, six of
/// each of the probe's categories, and ties of both signs.
const MEASURED_HALF_PAIRS: [[u32; 3]; 64] = [
    [0xa307, 0x88b6, 0x9f1a],
    [0x2f06, 0x9b28, 0x2acd],
    [0x962d, 0xca0f, 0xc60f],
    [0xf855, 0x009c, 0xf455],
    [0xd336, 0x63f6, 0x5f83],
    [0x40ad, 0x4482, 0x42d9],
    [0xa8bc, 0xa8bd, 0xa8bc],
    [0xd93c, 0xd93d, 0xd93c],
    [0xa3e2, 0xa3e3, 0xa3e2],
    [0xe08f, 0xe091, 0xe090],
    [0x5d77, 0x5d77, 0x5d77],
    [0x42e9, 0x42e9, 0x42e9],
    [0x2c6d, 0x2c6d, 0x2c6d],
    [0x9892, 0x988a, 0x988e],
    [0xda47, 0xda3f, 0xda43],
    [0x13cf, 0x13d0, 0x13d0],
    [0x63b3, 0x63b0, 0x63b2],
    [0xa57e, 0xa578, 0xa57b],
    [0x5fe9, 0x0134, 0x5be9],
    [0x5b39, 0x53d3, 0x5897],
    [0xe3a1, 0xd371, 0xe00c],
    [0x791e, 0x04af, 0x751e],
    [0x6a7b, 0x8e33, 0x667b],
    [0x6f09, 0x8586, 0x6b09],
    [0x8606, 0x04ed, 0x808c],
    [0x07c1, 0x85be, 0x0102],
    [0x01f0, 0x86bb, 0x8265],
    [0x0427, 0x0705, 0x0596],
    [0x0313, 0x830e, 0x0003],
    [0x86e3, 0x8545, 0x8614],
    [0x00c9, 0x80c5, 0x0002],
    [0x2901, 0xa903, 0x8200],
    [0x1ee4, 0x9ee8, 0x8080],
    [0x1afd, 0x9b00, 0x8030],
    [0x2bb4, 0xabb4, 0x0000],
    [0x3c18, 0xbc17, 0x1000],
    [0x7c01, 0xcd5d, 0xc95d],
    [0xf6b9, 0x8f01, 0xf2b9],
    [0x6198, 0x7d00, 0x5d98],
    [0x235a, 0x1c25, 0x20b6],
    [0xb9f2, 0x7fff, 0xb5f2],
    [0x9384, 0x7c00, 0x7800],
    [0x7800, 0xfc00, 0xf400],
    [0x7800, 0x7a00, 0x7900],
    [0x7a00, 0x7bfe, 0x7aff],
    [0x7800, 0x7bff, 0x7a00],
    [0x7c00, 0x7a00, 0x7b00],
    [0xfc00, 0x7c00, 0x0000],
    [0x8002, 0x8000, 0x8001],
    [0x0000, 0x0001, 0x0001],
    [0x0002, 0x8001, 0x0001],
    [0x8000, 0x0001, 0x0001],
    [0x0002, 0x0002, 0x0002],
    [0x8002, 0x0002, 0x0000],
    [0xeea7, 0xeea4, 0xeea5],
    [0x8001, 0x8002, 0x8001],
    [0x87ec, 0x87eb, 0x87eb],
    [0x98ed, 0x98ee, 0x98ed],
    [0x5580, 0x5579, 0x557d],
    [0x7800, 0x7bfd, 0x79ff],
    [0x4f07, 0x4f06, 0x4f07],
    [0x4782, 0x4783, 0x4783],
    [0xd993, 0xd998, 0xd995],
    [0x827c, 0x862f, 0x8455],
];

/// The 16-bit float, `RG11B10Float` and `RGB9E5Float` formats' rules hold at 2 and 4 samples,
/// where their modules are made; at 8 they are refused, as Metal measures no such count.
#[test]
fn the_float_rules_hold_at_2_and_4_samples() {
    let floats = [
        Format::R16_SFLOAT,
        Format::R16G16_SFLOAT,
        Format::R16G16B16A16_SFLOAT,
        Format::B10G11R11_UFLOAT_PACK32,
        Format::E5B9G9R9_UFLOAT_PACK32,
    ];
    for format in floats {
        let resolve = ColorResolveFormat::of(format).expect("closed");
        assert!(resolve.resolves(2) && resolve.resolves(4), "{format:?}");
        for samples in [8, 16] {
            assert!(!resolve.resolves(samples), "{format:?} x{samples}");
        }
    }
    for format in [Format::R8G8B8A8_UNORM, Format::R5G6B5_UNORM_PACK16] {
        let resolve = ColorResolveFormat::of(format).expect("closed");
        assert!([2, 4, 8]
            .into_iter()
            .all(|samples| resolve.resolves(samples)));
    }
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles/lavapipe.json");
    let text = std::fs::read_to_string(&path).expect("read");
    let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    let device = Device::new(desc, DevicePolicy::default()).expect("device");
    for format in floats {
        let resolve = ColorResolveFormat::of(format).expect("closed");
        let copy = PlanesCopy::for_format(format).expect("copy");
        for source in [ResolveSource::InputAttachment, ResolveSource::Planes(copy)] {
            for samples in [2, 4] {
                color_resolve_modules(&device, resolve, samples, source, Feature::Format)
                    .expect("modules");
            }
            let refused = color_resolve_modules(&device, resolve, 8, source, Feature::Format);
            assert!(matches!(refused, Err(Error::Internal(_))), "{source:?}");
        }
    }
}

/// The independent model gives Metal's measured answers, and the modules give the model's: the
/// measured pairs, random and special pairs of every half format, and every half beside zero, its
/// neighbour, an infinity and the smallest negative half, at 2 samples through both sources. At 2
/// samples the rule is the exact average rounded with a tie toward +∞, as macOS 26.7 measured.
#[test]
fn half_pairs_resolve_as_metal() {
    for [a, b, want] in MEASURED_HALF_PAIRS {
        assert_eq!(half_average(&[a, b]), want, "{a:#06x} {b:#06x}");
        assert_eq!(half_rule(&[a, b]), want, "{a:#06x} {b:#06x}");
    }
    let mut r = SplitMix(0x0048_5032);
    for _ in 0..100_000 {
        let pair = [half_bits(&mut r), half_bits(&mut r)];
        assert_eq!(half_rule(&pair), half_average(&pair), "{pair:x?}");
    }
    let mut r = SplitMix(0x0052_4553);
    for format in [
        Format::R16_SFLOAT,
        Format::R16G16_SFLOAT,
        Format::R16G16B16A16_SFLOAT,
    ] {
        let resolve = ColorResolveFormat::of(format).expect("closed");
        let copy = PlanesCopy::for_format(format).expect("copy");
        let mut texels: Vec<Vec<[u32; 4]>> = MEASURED_HALF_PAIRS
            .iter()
            .map(|&[a, b, _]| vec![[a, b, a, b], [b, a, b, a]])
            .collect();
        texels.extend((0..2048).map(|_| {
            (0..2)
                .map(|_| {
                    let mut codes = [0u32; 4];
                    for (component, channel) in resolve.channels().iter().enumerate() {
                        if let Some(channel) = channel {
                            codes[component] = draw(channel.rule, &mut r);
                        }
                    }
                    codes
                })
                .collect()
        }));
        for source in [ResolveSource::InputAttachment, ResolveSource::Planes(copy)] {
            // In runs the interpreter's step limit holds.
            for part in texels.chunks(4096) {
                check(resolve, 2, source, part);
            }
        }
    }
    let resolve = ColorResolveFormat::of(Format::R16G16B16A16_SFLOAT).expect("closed");
    let copy = PlanesCopy::for_format(Format::R16G16B16A16_SFLOAT).expect("copy");
    // Each half beside zero, its neighbour (a tie), an infinity and the smallest negative.
    let texels: Vec<Vec<[u32; 4]>> = (0..0x10000u32)
        .map(|h| vec![[h, h, h, h], [0x0000, h ^ 1, 0x7c00, 0x8001]])
        .collect();
    for source in [ResolveSource::InputAttachment, ResolveSource::Planes(copy)] {
        for part in texels.chunks(4096) {
            check(resolve, 2, source, part);
        }
    }
}

// ---- The rules the M3 Max under macOS 27.0.1 closed ----

/// Metal's resolves of packed 16-bit unorm channels measured on an M3 Max under macOS 27.0.1 (our
/// own probe `resolve.swift`, `B5G6R5Unorm` and `ABGR4Unorm`): the channel's bits, the sample
/// count, the samples' codes (0 past the count) and the resolved code, each one where the unorm
/// rule `(sum + n / 2) / n` is not Metal's.
const MEASURED_HALF_UNORM: [[u32; 7]; 32] = [
    [5, 2, 19, 22, 0, 0, 20],
    [5, 2, 29, 20, 0, 0, 24],
    [5, 2, 6, 17, 0, 0, 11],
    [5, 2, 25, 10, 0, 0, 17],
    [6, 2, 60, 9, 0, 0, 34],
    [6, 2, 7, 14, 0, 0, 10],
    [6, 2, 9, 40, 0, 0, 24],
    [6, 2, 27, 0, 0, 0, 13],
    [5, 4, 9, 9, 20, 20, 14],
    [5, 4, 20, 20, 15, 15, 17],
    [5, 4, 9, 8, 5, 4, 6],
    [5, 4, 30, 4, 21, 11, 16],
    [6, 4, 37, 37, 54, 54, 45],
    [6, 4, 26, 26, 17, 17, 21],
    [6, 4, 1, 1, 62, 62, 31],
    [6, 4, 17, 8, 56, 49, 32],
    [4, 2, 14, 5, 0, 0, 9],
    [4, 2, 8, 13, 0, 0, 10],
    [4, 2, 2, 11, 0, 0, 6],
    [4, 2, 9, 14, 0, 0, 11],
    [4, 2, 10, 15, 0, 0, 12],
    [4, 2, 15, 10, 0, 0, 12],
    [4, 2, 11, 12, 0, 0, 11],
    [4, 2, 7, 6, 0, 0, 6],
    [4, 4, 3, 3, 14, 14, 8],
    [4, 4, 4, 4, 15, 15, 9],
    [4, 4, 3, 4, 15, 0, 5],
    [4, 4, 8, 13, 9, 12, 10],
    [4, 4, 1, 1, 12, 12, 6],
    [4, 4, 1, 1, 10, 10, 5],
    [4, 4, 1, 1, 0, 0, 0],
    [4, 4, 2, 10, 0, 6, 4],
];

/// The deficits are where the `half` nearest a code's value lies below it, and are the measured
/// rule's: the masks of 1, 4, 5 and 6 bits.
#[test]
fn the_half_unorm_deficits_are_the_halves_below_each_code() {
    for (bits, mask) in [
        (1, 0),
        (4, 0x55fe),
        (5, 0x00ff_0f36),
        (6, 0x5555_5555_ffff_fffe),
    ] {
        assert_eq!(half_unorm_deficits(bits), mask, "{bits}");
        let m = f64::from((1u32 << bits) - 1);
        for c in 0..(1u32 << bits) {
            let v = f64::from(c) / m;
            let below = v > 0.0 && {
                let step = pow2(exponent(v) - 10);
                (v / step).round_ties_even() * step < v
            };
            assert_eq!((mask >> c) & 1 == 1, below, "{bits} bits, code {c}");
        }
    }
}

/// The model from the definition gives Metal's measured answers, which the unorm rule does not,
/// and the modules give the model's: the measured texels, every pair of codes of 4, 5 and 6 bits
/// at 2 samples, and random texels at 4 and 8, through both sources.
#[test]
fn packed_unorm_resolves_as_metal() {
    for [bits, n, c0, c1, c2, c3, want] in MEASURED_HALF_UNORM {
        let codes = &[c0, c1, c2, c3][..n as usize];
        let what = format!("{bits} bits x{n} {codes:?}");
        assert_eq!(half_unorm_model(bits as u8, codes), want, "{what}");
        let unorm = metal(ResolveRule::Unorm { bits: bits as u8 }, codes);
        assert_ne!(unorm, want, "{what}: the unorm rule is Metal's here");
    }
    for (format, components) in [
        (Format::R5G6B5_UNORM_PACK16, [0usize, 1]),
        (Format::R4G4B4A4_UNORM_PACK16, [0, 3]),
    ] {
        let resolve = ColorResolveFormat::of(format).expect("closed");
        let copy = PlanesCopy::for_format(format).expect("copy");
        // A measured row's samples in each component of its width; `None` where there is none.
        let place = |bits: u32, codes: &[u32]| -> Option<Vec<[u32; 4]>> {
            let at: Vec<usize> = (0..4)
                .filter(|&k| {
                    resolve.channels()[k].is_some_and(|c| u32::from(c.rule.bits()) == bits)
                })
                .collect();
            (!at.is_empty()).then(|| {
                codes
                    .iter()
                    .map(|&c| {
                        let mut sample = [0u32; 4];
                        for &k in &at {
                            sample[k] = c;
                        }
                        sample
                    })
                    .collect()
            })
        };
        let measured = |samples: u32| -> Vec<Vec<[u32; 4]>> {
            MEASURED_HALF_UNORM
                .iter()
                .filter(|row| row[1] == samples)
                .filter_map(|row| place(row[0], &row[2..2 + samples as usize]))
                .collect()
        };
        let mut texels = measured(2);
        // Every pair of codes, in the two components of the widths named.
        let [first, second] = components;
        let bits = |k: usize| u32::from(resolve.channels()[k].expect("channel").rule.bits());
        let (a, b) = (1u32 << bits(first), 1u32 << bits(second));
        for p in 0..(a * a).max(b * b) {
            let mut x = [0u32; 4];
            let mut y = [0u32; 4];
            x[first] = p % a;
            y[first] = (p / a) % a;
            x[second] = p % b;
            y[second] = (p / b) % b;
            texels.push(vec![x, y]);
        }
        for source in [ResolveSource::InputAttachment, ResolveSource::Planes(copy)] {
            for part in texels.chunks(4096) {
                check(resolve, 2, source, part);
            }
        }
        let mut r = SplitMix(0x0050_4b31);
        for samples in [4usize, 8] {
            let mut texels = measured(samples as u32);
            texels.extend((0..1024).map(|_| {
                (0..samples)
                    .map(|_| {
                        let mut codes = [0u32; 4];
                        for (k, channel) in resolve.channels().iter().enumerate() {
                            if let Some(channel) = channel {
                                codes[k] = draw(channel.rule, &mut r);
                            }
                        }
                        codes
                    })
                    .collect()
            }));
            for source in [ResolveSource::InputAttachment, ResolveSource::Planes(copy)] {
                check(resolve, samples, source, &texels);
            }
        }
    }
}

/// Metal's resolves of 4-sample `RGBA16Float` channels measured on an M3 Max under macOS 27.0.1
/// (our own probe `resolve.swift`): four samples' bits and the resolved bits. All but the last
/// four are channels the exact average, rounded as at 2 samples, does not give: with an infinity,
/// a NaN, a subnormal, and the rest.
const MEASURED_HALF_QUADS: [[u32; 5]; 32] = [
    [0x92ea, 0xfc00, 0x7bff, 0x8000, 0xc880],
    [0xcfa4, 0xfc00, 0x1a03, 0x7bfc, 0xd100],
    [0xfc00, 0xbb27, 0x7bff, 0xe8ff, 0xe112],
    [0x7c00, 0x89dc, 0x5010, 0xfc00, 0x4700],
    [0xfc00, 0x118c, 0xc389, 0x7bfc, 0xd020],
    [0x840b, 0x7c00, 0xc0f6, 0xfbfc, 0x4f80],
    [0x1e00, 0x7e00, 0x9dd4, 0x8406, 0x01be],
    [0x78a4, 0x7e00, 0x9180, 0xcbff, 0x70a3],
    [0xb1a5, 0x8536, 0x3219, 0x7e00, 0x1b35],
    [0xf8ee, 0xd1f8, 0x7e00, 0x91d9, 0xf0f0],
    [0xddbb, 0x7bfd, 0x82f0, 0x82ad, 0x73f1],
    [0xe5f5, 0x81a5, 0xbe41, 0xd03c, 0xde19],
    [0x03b7, 0x8815, 0xe768, 0x676a, 0x37c0],
    [0x6d27, 0x9e75, 0x0284, 0x4f86, 0x652e],
    [0x91ff, 0x039b, 0x3418, 0x109c, 0x2c17],
    [0x037d, 0x2e96, 0xbf7e, 0x06de, 0xb715],
    [0xac63, 0x4eaa, 0x23ca, 0x27f0, 0x46a8],
    [0x8000, 0xea06, 0x9fae, 0x3c1a, 0xe206],
    [0xdadb, 0xefd7, 0x6ecc, 0x9b9c, 0xdd08],
    [0x9624, 0xfbff, 0x63a2, 0x0e6f, 0xf3e1],
    [0xe771, 0x92b6, 0x3ea7, 0xfb68, 0xf3a4],
    [0x8aa2, 0xd949, 0x2c50, 0x8000, 0xd149],
    [0x7bfc, 0x2387, 0xfbfd, 0xb173, 0xc840],
    [0xb12c, 0x31e6, 0xa5d3, 0x8816, 0x82cc],
    [0x0478, 0x7bfe, 0xfbfe, 0x5a44, 0x5240],
    [0x7bfc, 0x3c00, 0xfbfe, 0x296c, 0xcc00],
    [0x6141, 0xfbfe, 0x75b5, 0x840d, 0xf10f],
    [0x7bfd, 0x0407, 0xfbfe, 0x8409, 0xc840],
    [0x0000, 0x840c, 0xe3d7, 0xd745, 0xdc60],
    [0xfbff, 0xef30, 0x7bfc, 0x2fb4, 0xe748],
    [0xbada, 0x1cf2, 0x1500, 0x2f0a, 0xb1ec],
    [0xce25, 0xc665, 0xda3e, 0x4f4e, 0xd24c],
];

/// The model gives Metal's measured answers at 4 samples, which the exact average does not, and
/// the modules give the model's: the measured quads, quads that carry `A` into a new binade or
/// make it a negative power of two, and random and special quads of every half format, through
/// both sources.
#[test]
fn half_quads_resolve_as_metal() {
    for (i, [a, b, c, d, want]) in MEASURED_HALF_QUADS.into_iter().enumerate() {
        let quad = [a, b, c, d];
        assert_eq!(half_rule(&quad), want, "{quad:x?}");
        let average = half_average(&quad);
        if i < MEASURED_HALF_QUADS.len() - 4 {
            assert_ne!(
                average, want,
                "{quad:x?}: the exact average is Metal's here"
            );
        } else {
            assert_eq!(average, want, "{quad:x?}");
        }
    }
    // A carry of samples 0 and 1 into 2^15 and a sum of -128 (a negative power of two), each
    // beside small samples.
    let shaped = [
        [0x7781, 0x7781, 0x3555, 0x1001],
        [0xd800, 0x0000, 0x2d71, 0x2eff],
        [0xfbff, 0x7bfc, 0x29c1, 0x2afd],
    ];
    let mut r = SplitMix(0x0048_5134);
    for format in [
        Format::R16_SFLOAT,
        Format::R16G16_SFLOAT,
        Format::R16G16B16A16_SFLOAT,
    ] {
        let resolve = ColorResolveFormat::of(format).expect("closed");
        let copy = PlanesCopy::for_format(format).expect("copy");
        let mut texels: Vec<Vec<[u32; 4]>> = MEASURED_HALF_QUADS
            .iter()
            .map(|row| row[..4].iter().map(|&h| [h, h, h, h]).collect())
            .collect();
        texels.extend(
            shaped
                .iter()
                .map(|quad| quad.iter().map(|&h| [h, h, h, h]).collect()),
        );
        texels.extend((0..2048).map(|_| {
            (0..4)
                .map(|_| {
                    let mut codes = [0u32; 4];
                    for (k, channel) in resolve.channels().iter().enumerate() {
                        if let Some(channel) = channel {
                            codes[k] = draw(channel.rule, &mut r);
                        }
                    }
                    codes
                })
                .collect()
        }));
        for source in [ResolveSource::InputAttachment, ResolveSource::Planes(copy)] {
            for part in texels.chunks(4096) {
                check(resolve, 4, source, part);
            }
        }
    }
}

/// Metal's resolves of `RG11B10Float` channels measured on an M3 Max under macOS 27.0.1 (our own
/// probe `resolve.swift`, its random draw and its corners): the mantissa's bits, the sample count,
/// the samples' bits (0 past the count) and the resolved bits, each one where the exact average
/// rounded to nearest even onto the channel's grid is not Metal's: a double rounding, or a result
/// past the largest finite value held to it.
const MEASURED_SMALL_FLOATS: [[u32; 7]; 36] = [
    [6, 2, 0x15f, 0x2de, 0x000, 0x000, 0x2a0],
    [6, 2, 0x298, 0x0c3, 0x000, 0x000, 0x258],
    [6, 2, 0x0b8, 0x299, 0x000, 0x000, 0x25a],
    [6, 2, 0x386, 0x26f, 0x000, 0x000, 0x34a],
    [5, 2, 0x07e, 0x14d, 0x000, 0x000, 0x12e],
    [5, 2, 0x217, 0x13e, 0x000, 0x000, 0x1f8],
    [5, 2, 0x11f, 0x1f1, 0x000, 0x000, 0x1d2],
    [5, 2, 0x07f, 0x141, 0x000, 0x000, 0x122],
    [6, 4, 0x2a9, 0x62a, 0x5e6, 0x336, 0x5ce],
    [6, 4, 0x4f6, 0x175, 0x516, 0x071, 0x4c8],
    [6, 4, 0x59a, 0x7e0, 0x115, 0x618, 0x5ae],
    [6, 4, 0x69c, 0x085, 0x60a, 0x37e, 0x62e],
    [5, 4, 0x110, 0x2e8, 0x09e, 0x268, 0x2aa],
    [5, 4, 0x198, 0x2ea, 0x32a, 0x03b, 0x2f4],
    [5, 4, 0x1a8, 0x256, 0x132, 0x144, 0x218],
    [5, 4, 0x3d5, 0x040, 0x279, 0x3cc, 0x3b0],
    [6, 2, 0x7bf, 0x7c0, 0x000, 0x000, 0x7bf],
    [6, 2, 0x7c0, 0x7bf, 0x000, 0x000, 0x7bf],
    [6, 2, 0x7bf, 0x7c0, 0x000, 0x000, 0x7bf],
    [6, 2, 0x7c0, 0x7bf, 0x000, 0x000, 0x7bf],
    [6, 2, 0x7bf, 0x7c0, 0x000, 0x000, 0x7bf],
    [5, 2, 0x3df, 0x3e0, 0x000, 0x000, 0x3df],
    [5, 2, 0x3e0, 0x3df, 0x000, 0x000, 0x3df],
    [5, 2, 0x3df, 0x3e0, 0x000, 0x000, 0x3df],
    [5, 2, 0x3df, 0x3e0, 0x000, 0x000, 0x3df],
    [5, 2, 0x3e0, 0x3df, 0x000, 0x000, 0x3df],
    [6, 4, 0x3c0, 0x000, 0x3c1, 0x03f, 0x380],
    [6, 4, 0x000, 0x7bf, 0x7be, 0x3c1, 0x77e],
    [6, 4, 0x040, 0x000, 0x7bd, 0x7c0, 0x77e],
    [6, 4, 0x780, 0x03f, 0x000, 0x7bd, 0x75e],
    [6, 4, 0x781, 0x3c0, 0x740, 0x780, 0x750],
    [5, 4, 0x3c1, 0x3d0, 0x1e0, 0x000, 0x3a8],
    [5, 4, 0x020, 0x01f, 0x3c0, 0x3c1, 0x3a0],
    [5, 4, 0x3dd, 0x01f, 0x3d0, 0x01f, 0x3b6],
    [5, 4, 0x01f, 0x3e0, 0x1e1, 0x3c1, 0x3b0],
    [5, 4, 0x1e1, 0x3d0, 0x1e0, 0x3c1, 0x3a8],
];

/// The model gives Metal's measured answers, and the modules give the model's: the measured
/// channels and random and special texels at 2 and 4 samples, through both sources.
#[test]
fn small_floats_resolve_as_metal() {
    for [mantissa, n, s0, s1, s2, s3, want] in MEASURED_SMALL_FLOATS {
        let codes = &[s0, s1, s2, s3][..n as usize];
        assert_eq!(
            small_rule(mantissa as u8, codes),
            want,
            "{mantissa} x{n} {codes:x?}"
        );
    }
    let format = Format::B10G11R11_UFLOAT_PACK32;
    let resolve = ColorResolveFormat::of(format).expect("closed");
    let copy = PlanesCopy::for_format(format).expect("copy");
    let mut r = SplitMix(0x0052_3131);
    for samples in [2usize, 4] {
        let mut texels: Vec<Vec<[u32; 4]>> = Vec::new();
        for [mantissa, n, s0, s1, s2, s3, _] in MEASURED_SMALL_FLOATS {
            if n as usize == samples {
                let k = if mantissa == 6 { 0 } else { 2 };
                texels.push(
                    [s0, s1, s2, s3][..samples]
                        .iter()
                        .map(|&v| {
                            let mut codes = [0u32; 4];
                            codes[k] = v;
                            codes
                        })
                        .collect(),
                );
            }
        }
        texels.extend((0..2048).map(|_| {
            (0..samples)
                .map(|_| {
                    let mut codes = [0u32; 4];
                    for (k, channel) in resolve.channels().iter().enumerate() {
                        if let Some(channel) = channel {
                            codes[k] = draw(channel.rule, &mut r);
                        }
                    }
                    codes
                })
                .collect()
        }));
        for source in [ResolveSource::InputAttachment, ResolveSource::Planes(copy)] {
            check(resolve, samples, source, &texels);
        }
    }
}

/// Metal's resolves of `RGB9E5Float` texels measured on an M3 Max under macOS 27.0.1 (our own
/// probe `resolve.swift`): the sample count, the samples' words (0 past the count) and the
/// resolved word.
const MEASURED_SHARED_EXPONENT: [[u32; 6]; 16] = [
    [
        2, 0x7b33c928, 0x942c5078, 0x00000000, 0x00000000, 0x8c94ca9d,
    ],
    [
        2, 0x85303381, 0xc9898f1c, 0x00000000, 0x00000000, 0xc18d8f1d,
    ],
    [
        2, 0xb97087dd, 0x0857c703, 0x00000000, 0x00000000, 0xb17087dd,
    ],
    [
        2, 0x3fa5cdb8, 0x3f728b8d, 0x00000000, 0x00000000, 0x3f8e2da3,
    ],
    [
        2, 0x1f2c9981, 0xc23f44d5, 0x00000000, 0x00000000, 0xba3f44d5,
    ],
    [
        2, 0xfc095c24, 0xf1b210cc, 0x00000000, 0x00000000, 0xf4e2648a,
    ],
    [
        2, 0xa11f1942, 0xc30eaafd, 0x00000000, 0x00000000, 0xbb22dd11,
    ],
    [
        2, 0x3188a954, 0x9d7de2fe, 0x00000000, 0x00000000, 0x957de2fe,
    ],
    [
        4, 0x9d8310bd, 0x45361262, 0xcecde86c, 0x2283e375, 0xbee5f46f,
    ],
    [
        4, 0x1ef914b5, 0xb08ea23a, 0x2d7586ba, 0x48359741, 0xa08ea23a,
    ],
    [
        4, 0x67bd78a0, 0x4cedf356, 0xa323edd6, 0xc47f0a67, 0xb4b34a84,
    ],
    [
        4, 0xe476232b, 0x88e5a132, 0xc47098ce, 0x2cb12d98, 0xd4be2d38,
    ],
    [
        4, 0x98680b84, 0xa5bc92b6, 0xcce3ded8, 0x1c6df956, 0xbd13e2e4,
    ],
    [
        4, 0x4a832090, 0xacabc292, 0x8e6041a6, 0x35a82a90, 0x9d13c6ac,
    ],
    [
        4, 0x839be263, 0x46ff82eb, 0x78531cc5, 0x36bcf71f, 0x79e6ba63,
    ],
    [
        4, 0xc7fcde2d, 0x8e175bce, 0xc16338ba, 0xe0f39825, 0xd18bda34,
    ],
];

/// Shared-exponent words that reach every case of the encode: random exponents and mantissas,
/// zero, the largest value, and a channel alone.
fn shared_word(r: &mut SplitMix) -> u32 {
    let e = (r.next() % 32) as u32;
    let mut m = [0u32; 3].map(|_| (r.next() & 0x1ff) as u32);
    match r.next() % 8 {
        0 => return 0,
        1 => return (31 << 27) | (0x1ff << 18) | (0x1ff << 9) | 0x1ff,
        2 => {
            m[1] = 0;
            m[2] = 0;
        }
        _ => {}
    }
    (e << 27) | (m[2] << 18) | (m[1] << 9) | m[0]
}

/// The kernel's input words of shared-exponent `texels` (each sample's word) as read from
/// `source`.
fn shared_words(source: ResolveSource, texels: &[Vec<u32>]) -> Vec<u8> {
    let mut input = Vec::new();
    for texel in texels {
        for &word in texel {
            let words = match source {
                ResolveSource::Planes(_) => [word, 0, 0, 0],
                _ => {
                    let v = shared_values(word);
                    [v[0], v[1], v[2], 0.0].map(|x| (x as f32).to_bits())
                }
            };
            input.extend(words.iter().flat_map(|w| w.to_le_bytes()));
        }
    }
    input
}

/// Run `RGB9E5Float`'s arithmetic at `samples` from `source` over `texels` (each sample's word),
/// and check each channel's output against [`shared_rule`].
fn check_shared(samples: usize, source: ResolveSource, texels: &[Vec<u32>]) {
    let format = ColorResolveFormat::of(Format::E5B9G9R9_UFLOAT_PACK32).expect("closed");
    let input = shared_words(source, texels);
    let out = dispatch(format, samples, source, input, texels.len())
        .buffers
        .remove(1);
    for (i, texel) in texels.iter().enumerate() {
        let want = shared_rule(texel);
        for (component, &w) in want.iter().enumerate() {
            let at = (i * 4 + component) * 4;
            let word = u32::from_le_bytes(out[at..at + 4].try_into().expect("word"));
            assert_eq!(
                f32::from_bits(word).to_bits(),
                (w as f32).to_bits(),
                "x{samples} {source:?} component {component}: {texel:x?}"
            );
        }
    }
}

/// The model gives Metal's measured answers, and the modules give the model's: the measured
/// texels and random and special texels at 2 and 4 samples, through both sources.
#[test]
fn shared_exponents_resolve_as_metal() {
    for [n, w0, w1, w2, w3, want] in MEASURED_SHARED_EXPONENT {
        let words = &[w0, w1, w2, w3][..n as usize];
        assert_eq!(shared_rule(words), shared_values(want), "x{n} {words:x?}");
    }
    let copy = PlanesCopy::for_format(Format::E5B9G9R9_UFLOAT_PACK32).expect("copy");
    let mut r = SplitMix(0x0052_3945);
    for samples in [2usize, 4] {
        let mut texels: Vec<Vec<u32>> = MEASURED_SHARED_EXPONENT
            .iter()
            .filter(|row| row[0] as usize == samples)
            .map(|row| row[1..1 + samples].to_vec())
            .collect();
        texels.extend((0..2048).map(|_| (0..samples).map(|_| shared_word(&mut r)).collect()));
        for source in [ResolveSource::InputAttachment, ResolveSource::Planes(copy)] {
            check_shared(samples, source, &texels);
        }
    }
}

// ---- Pixels whose samples are alike ----

/// Metal's encode of each 8-bit sRGB code's linear value is the code, so the sRGB rule gives back
/// the code a pixel's every sample holds (`resolve_alike`).
#[test]
fn the_srgb_encode_of_each_code_s_linear_value_is_the_code() {
    for (code, &linear) in SRGB_RESOLVE_LINEAR.iter().enumerate() {
        assert_eq!(
            SRGB_RESOLVE_THRESHOLDS.partition_point(|&t| t <= linear),
            code,
            "{linear}"
        );
    }
}

/// The codes of `rule` an alike pixel takes in [`alike_texels`]: every code of a channel of 11 bits
/// or fewer, else 2048 drawn ([`draw`]: random, the edges and every float class).
fn alike_codes(rule: ResolveRule, r: &mut SplitMix) -> Vec<u32> {
    match u32::from(rule.bits()) {
        bits @ ..=11 => (0..1 << bits).collect(),
        _ => (0..2048).map(|_| draw(rule, r)).collect(),
    }
}

/// Texels of `format` whose `samples` samples all hold the same codes: component `k` of texel `i`
/// the code `(i + 7k) mod len` of its channel's [`alike_codes`], over the longest list.
fn alike_texels(
    format: ColorResolveFormat,
    samples: usize,
    r: &mut SplitMix,
) -> Vec<Vec<[u32; 4]>> {
    let lists: Vec<Option<Vec<u32>>> = format
        .channels()
        .iter()
        .map(|c| c.map(|c| alike_codes(c.rule, r)))
        .collect();
    let count = lists.iter().flatten().map(Vec::len).max().unwrap_or(0);
    (0..count)
        .map(|i| {
            let mut codes = [0u32; 4];
            for (k, list) in lists.iter().enumerate() {
                if let Some(list) = list {
                    codes[k] = list[(i + 7 * k) % list.len()];
                }
            }
            vec![codes; samples]
        })
        .collect()
}

/// A pixel whose samples all hold the same codes resolves as the model of Metal's rules
/// ([`metal`]) says, which is the code (`resolve_alike`): every closed format but the shared
/// exponent's (below), at each of 2, 4 and 8 samples its rules hold at, through every source
/// ([`sources`]).
#[test]
fn alike_samples_resolve_as_metal() {
    let mut r = SplitMix(0x0041_4c4b);
    for format in CLOSED {
        if format == Format::E5B9G9R9_UFLOAT_PACK32 {
            continue;
        }
        let resolve = ColorResolveFormat::of(format).expect("closed");
        let copy = PlanesCopy::for_planes(format).expect("copy");
        for samples in [2usize, 4, 8]
            .into_iter()
            .filter(|&n| resolve.resolves(n as u32))
        {
            let texels = alike_texels(resolve, samples, &mut r);
            for (resolve, source) in cases(resolve, copy) {
                for chunk in texels.chunks(4096) {
                    check(resolve, samples, source, chunk);
                }
            }
        }
    }
}

/// Every 16-bit float, a NaN, an infinity and -0 among them, held by every sample of a pixel,
/// resolves as the model says (a NaN and -0 to +0), at 2 and 4 samples through both sources: the
/// four channels of `RGBA16Float` take the codes `4i` to `4i + 3`.
#[test]
fn every_alike_half_resolves_as_metal() {
    let format = Format::R16G16B16A16_SFLOAT;
    let resolve = ColorResolveFormat::of(format).expect("closed");
    let copy = PlanesCopy::for_format(format).expect("copy");
    for samples in [2usize, 4] {
        let texels: Vec<Vec<[u32; 4]>> = (0..1u32 << 14)
            .map(|i| vec![[4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3]; samples])
            .collect();
        for source in [ResolveSource::InputAttachment, ResolveSource::Planes(copy)] {
            for chunk in texels.chunks(4096) {
                check(resolve, samples, source, chunk);
            }
        }
    }
}

/// A shared-exponent texel held by every sample of a pixel resolves as the model says, at 2 and
/// 4 samples through both sources.
#[test]
fn alike_shared_exponents_resolve_as_metal() {
    let copy = PlanesCopy::for_format(Format::E5B9G9R9_UFLOAT_PACK32).expect("copy");
    let mut r = SplitMix(0x0041_4c45);
    for samples in [2usize, 4] {
        let texels: Vec<Vec<u32>> = (0..2048)
            .map(|_| vec![shared_word(&mut r); samples])
            .collect();
        for source in [ResolveSource::InputAttachment, ResolveSource::Planes(copy)] {
            check_shared(samples, source, &texels);
        }
    }
}

/// A pixel whose samples all hold the same bits skips the average: on the interpreter, 64 such
/// texels of each representative format at 4 samples take fewer lane-steps, through every
/// source, than the same texels with the last sample's first channel changed; a format that
/// always averages (32-bit floats, or unorm and snorm channels alone) takes as many.
#[test]
fn alike_samples_skip_the_average() {
    let mut r = SplitMix(0x0041_4c53);
    for (format, copy) in representatives() {
        for source in sources(format, copy) {
            let inputs = if format.shared_exponent() {
                let words: Vec<u32> = (0..64).map(|_| shared_word(&mut r)).collect();
                [false, true].map(|change| {
                    let texels: Vec<Vec<u32>> = words
                        .iter()
                        .map(|&w| vec![w, w, w, if change { w ^ 2 } else { w }])
                        .collect();
                    shared_words(source, &texels)
                })
            } else {
                let first = format
                    .channels()
                    .iter()
                    .position(Option::is_some)
                    .expect("a channel");
                let codes: Vec<[u32; 4]> = (0..64)
                    .map(|_| {
                        let mut codes = [0u32; 4];
                        for (k, channel) in format.channels().iter().enumerate() {
                            if let Some(channel) = channel {
                                codes[k] = draw(channel.rule, &mut r);
                            }
                        }
                        codes
                    })
                    .collect();
                [false, true].map(|change| {
                    let texels: Vec<Vec<[u32; 4]>> = codes
                        .iter()
                        .map(|&c| {
                            let mut last = c;
                            if change {
                                last[first] ^= 2;
                            }
                            vec![c, c, c, last]
                        })
                        .collect();
                    sample_words(format, source, &texels)
                })
            };
            let [alike, changed] = inputs.map(|input| dispatch(format, 4, source, input, 64).steps);
            let what = format!(
                "{:?} {source:?}: {alike} against {changed}",
                format.format()
            );
            let rules = || format.channels().into_iter().flatten().map(|c| c.rule);
            let compares = rules().all(ResolveRule::keeps_alike)
                && !rules().all(ResolveRule::averages_cheaply);
            if compares {
                assert!(alike < changed, "{what}");
            } else {
                assert_eq!(alike, changed, "{what}");
            }
        }
    }
}

/// The raw twin reads an sRGB format's codes with no table lookup: on the interpreter, 64 random
/// `BGRA8Unorm_sRGB` texels at 4 samples take fewer lane-steps from
/// [`ResolveSource::RawInputAttachment`] than from [`ResolveSource::InputAttachment`], each sample
/// alike and with the last sample's red changed.
#[test]
fn the_raw_twin_reads_srgb_codes_in_fewer_steps() {
    let format = ColorResolveFormat::of(Format::B8G8R8A8_SRGB).expect("closed");
    let mut r = SplitMix(0x0052_4157);
    let codes: Vec<[u32; 4]> = (0..64)
        .map(|_| [0; 4].map(|_: u32| (r.next() as u32) & 0xff))
        .collect();
    for change in [false, true] {
        let texels: Vec<Vec<[u32; 4]>> = codes
            .iter()
            .map(|&c| {
                let mut last = c;
                if change {
                    last[0] ^= 2;
                }
                vec![c, c, c, last]
            })
            .collect();
        let [read, raw] = [
            ResolveSource::InputAttachment,
            ResolveSource::RawInputAttachment,
        ]
        .map(|source| {
            let input = sample_words(format, source, &texels);
            dispatch(format, 4, source, input, texels.len()).steps
        });
        assert!(raw < read, "changed {change}: {raw} against {read}");
    }
}

/// Written through the raw twin, an sRGB format's codes go out with no table: the fragment module
/// of `BGRA8Unorm_sRGB` at 4 samples, read through the raw twin, holds no 256-entry table of
/// linear values written raw ([`ColorResolveFormat::written_raw`]), and so is at least that many
/// words smaller than the one written in the format. Lane-steps do not show the saving: the
/// lookup takes as many instructions as the conversion and scale in its place, where the GPU pays
/// for a dynamically indexed `Private` array (`runtime_lane::resolve_rounding`'s cost test).
#[test]
fn the_raw_twin_writes_srgb_codes_with_no_table() {
    let format = ColorResolveFormat::of(Format::B8G8R8A8_SRGB).expect("closed");
    let source = ResolveSource::RawInputAttachment;
    let [own, raw] = [format, format.written_raw()].map(|format| {
        let m = fragment_typed_module(format, 4, source).expect("module");
        crate::spirv::serialize::serialize(m)
            .expect("serialize")
            .words
            .len()
    });
    assert!(raw + 256 <= own, "{raw} words against {own}");
}

// ---- The sRGB codes by estimate and correction ----

/// A kernel whose invocation `i` reads `words` words from `SAMPLES_IN` at `i * words` and writes
/// what `body` makes of them (with the module's sRGB tables) to word `i` of `RESOLVED_OUT`.
fn word_kernel(
    words: u32,
    body: &dyn Fn(&mut FunctionBuilder<'_>, Types, SrgbTables, &[Value]) -> BResult<Value>,
) -> TypedModule {
    let mut b = ModuleBuilder::new();
    let t = Types::new(&mut b).expect("types");
    // Every table: an sRGB format's, read as floats and written in its own format.
    let srgb = ColorResolveFormat::of(Format::R8G8B8A8_SRGB).expect("closed");
    let tables =
        SrgbTables::declare(&mut b, t, srgb, ResolveSource::InputAttachment).expect("tables");
    let buffer = super::super::visibility_resolve::words_buffer;
    let input = buffer(&mut b, SAMPLES_IN, t.u32_).expect("in");
    let output = buffer(&mut b, RESOLVED_OUT, t.u32_).expect("out");
    let uvec3 = b
        .ty(Type::Vector {
            elem: t.u32_,
            len: 3,
        })
        .expect("uvec3");
    let invocation = b
        .builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)
        .expect("id");
    let void = b.ty(Type::Void).expect("void");
    let mut f = b.function(void).expect("function");
    let build = |f: &mut FunctionBuilder<'_>| -> BResult<()> {
        let zero = uint(f, t, 0)?;
        let invocation = f.module().global_value(invocation)?;
        let invocation = f.load(invocation, MemAccess::default())?;
        let i = f.composite_extract(invocation, &[0])?;
        let input = f.module().global_value(input)?;
        let output = f.module().global_value(output)?;
        let per = uint(f, t, words)?;
        let base = f.binary(BinOp::IMul, i, per)?;
        let mut read = Vec::new();
        for w in 0..words {
            let k = uint(f, t, w)?;
            let at = f.binary(BinOp::IAdd, base, k)?;
            let p = f.access_chain(input, &[zero, at])?;
            read.push(f.load(p, MemAccess::default())?);
        }
        let value = body(f, t, tables, &read)?;
        let p = f.access_chain(output, &[zero, i])?;
        f.store(p, value, MemAccess::default())
    };
    build(&mut f).expect("body");
    f.ret(None).expect("ret");
    let id = f.finish().expect("finish");
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize { x: 1, y: 1, z: 1 }],
    )
    .expect("entry");
    b.finalize().expect("finalize")
}

/// Run `m` (a [`word_kernel`]) over `rows`, each its words, and return each invocation's word.
fn run_words(m: &TypedModule, rows: &[Vec<u32>]) -> Vec<u32> {
    let input = rows
        .iter()
        .flatten()
        .flat_map(|w| w.to_le_bytes())
        .collect();
    let res = Resources {
        buffers: vec![
            Buffer {
                slot: Some(SAMPLES_IN),
                address: None,
                bytes: input,
            },
            Buffer {
                slot: Some(RESOLVED_OUT),
                address: None,
                bytes: vec![0; rows.len() * 4],
            },
        ],
        push: Vec::new(),
        spec: Default::default(),
    };
    let cfg = Config {
        order: Order::Forward,
        workgroups: [rows.len() as u32, 1, 1],
        ..Config::default()
    };
    let out = run(m, &cfg, res).expect("run").buffers.remove(1);
    out.chunks(4)
        .map(|w| u32::from_le_bytes(w.try_into().expect("word")))
        .collect()
}

/// The sRGB encode of the linear `x` in codes, `255 encode(x)` (IEC 61966-2-1), in `f64`.
fn srgb_encode_codes(x: f64) -> f64 {
    let encoded = if x <= 0.0031308 {
        12.92 * x
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    };
    255.0 * encoded
}

/// The midpoints the module's table holds: the `f32` midway between each code's linear value and
/// the next's.
fn srgb_midpoints() -> Vec<f32> {
    (0..255)
        .map(|k| ((srgb_decode(k) + srgb_decode(k + 1)) / 2.0) as f32)
        .collect()
}

/// The floats whose sRGB code a test asks: each code's linear value, and the lowest and highest
/// float of the code (the midpoint below it, and the float under the midpoint above it), all in
/// `[0, 1]`; each with its code.
fn srgb_floats() -> Vec<(f32, u32)> {
    let mid = srgb_midpoints();
    let mut out = Vec::new();
    for c in 0..256u32 {
        out.push((srgb_decode(c) as f32, c));
        if c > 0 {
            out.push((mid[c as usize - 1], c));
        }
        if c < 255 {
            out.push((f32::from_bits(mid[c as usize].to_bits() - 1), c));
        }
    }
    out
}

/// The estimate is within 0.12 codes of the sRGB encode over every float a test asks and every
/// 12-bit value's linear value, on the interpreter (its square root rounded correctly).
#[test]
fn the_srgb_estimate_is_within_0_12_codes_of_the_encode() {
    let m = word_kernel(1, &|f, t, _, w| {
        let x = f.convert(ConvOp::Bitcast, w[0], t.f32_)?;
        let e = srgb_estimate(f, t, x)?;
        f.convert(ConvOp::Bitcast, e, t.u32_)
    });
    let mut xs: Vec<f32> = srgb_floats().into_iter().map(|(x, _)| x).collect();
    xs.extend((0..4096u32).map(|l| l as f32 * (1.0f32 / 4095.0)));
    xs.extend((0..=4096u32).map(|k| k as f32 / 4096.0));
    let rows: Vec<Vec<u32>> = xs.iter().map(|x| vec![x.to_bits()]).collect();
    for (x, got) in xs.iter().zip(run_words(&m, &rows)) {
        let want = srgb_encode_codes(f64::from(*x));
        let error = (f64::from(f32::from_bits(got)) - want).abs();
        assert!(
            error < 0.12,
            "{x}: estimate {} against {want}",
            f32::from_bits(got)
        );
    }
}

/// The offsets [`the_srgb_codes_are_exact_from_any_estimate_near_the_encode`] adds to the
/// estimate: an estimate this far off still rounds to within one code, so the correction is exact
/// for any device whose square roots err by far less.
const ESTIMATE_OFFSETS: [f32; 5] = [-0.75, -0.4, 0.0, 0.4, 0.75];

/// The corrected rank is exact from any estimate up to 0.75 codes off the module's own: an input
/// attachment's float of every code, and its lowest and highest float, give the code (the count of
/// midpoints it reaches); and every 12-bit value gives Metal's encode of it (the count of
/// thresholds it reaches), on the interpreter.
#[test]
fn the_srgb_codes_are_exact_from_any_estimate_near_the_encode() {
    let floats = word_kernel(2, &|f, t, tables, w| {
        let x = f.convert(ConvOp::Bitcast, w[0], t.f32_)?;
        let offset = f.convert(ConvOp::Bitcast, w[1], t.f32_)?;
        let e = srgb_estimate(f, t, x)?;
        let e = f.binary(BinOp::FAdd, e, offset)?;
        let midpoints = tables.midpoints.expect("midpoints");
        corrected_rank(f, t, midpoints, x, e, BinOp::FOrdGreaterThanEqual)
    });
    let cases: Vec<(f32, u32, f32)> = srgb_floats()
        .into_iter()
        .flat_map(|(x, c)| ESTIMATE_OFFSETS.map(|o| (x, c, o)))
        .collect();
    let mid = srgb_midpoints();
    let rows: Vec<Vec<u32>> = cases
        .iter()
        .map(|&(x, _, o)| vec![x.to_bits(), o.to_bits()])
        .collect();
    for (&(x, c, o), got) in cases.iter().zip(run_words(&floats, &rows)) {
        assert_eq!(mid.partition_point(|&m| m <= x) as u32, c, "{x}");
        assert_eq!(got, c, "the float {x} with the estimate {o:+} off");
    }
    let values = word_kernel(2, &|f, t, tables, w| {
        let offset = f.convert(ConvOp::Bitcast, w[1], t.f32_)?;
        let real = f.convert(ConvOp::ConvertUToF, w[0], t.f32_)?;
        let unit = float(f, t, 1.0 / 4095.0)?;
        let linear = f.binary(BinOp::FMul, real, unit)?;
        let e = srgb_estimate(f, t, linear)?;
        let e = f.binary(BinOp::FAdd, e, offset)?;
        corrected_rank(f, t, tables.thresholds, w[0], e, BinOp::UGreaterThanEqual)
    });
    let cases: Vec<(u32, f32)> = (0..4096u32)
        .flat_map(|l| ESTIMATE_OFFSETS.map(|o| (l, o)))
        .collect();
    let rows: Vec<Vec<u32>> = cases.iter().map(|&(l, o)| vec![l, o.to_bits()]).collect();
    for (&(l, o), got) in cases.iter().zip(run_words(&values, &rows)) {
        let want = SRGB_RESOLVE_THRESHOLDS.partition_point(|&t| u32::from(t) <= l) as u32;
        assert_eq!(
            got, want,
            "the 12-bit value {l} with the estimate {o:+} off"
        );
    }
}
