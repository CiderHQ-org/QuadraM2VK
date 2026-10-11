//! The color resolve modules: Metal's measured tables as the module docs state them, valid on
//! every committed profile (L3, `G-val`), needing nothing of a device, and exact on the CPU
//! interpreter: the fragment stage's arithmetic, run in a kernel over every pair of 8-bit codes and
//! over random and edge texels, writes the float the attachment stores as Metal's code.

use super::*;
use crate::device::{DeviceDescription, DevicePolicy};
use crate::spirv::interp::{run, Buffer, Config, Order, Resources};
use crate::spirv::testing::{assert_valid, committed_profiles};
use std::path::PathBuf;

/// Every format the module docs list, and a few they leave out.
#[test]
fn the_formats_are_the_ones_metals_rules_close() {
    let closed = [
        Format::R8_UNORM,
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
    ];
    for format in closed {
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
        Format::B10G11R11_UFLOAT_PACK32,
        Format::E5B9G9R9_UFLOAT_PACK32,
        Format::B5G6R5_UNORM_PACK16,
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

/// The requirements of `m` as serialized.
fn needs(m: TypedModule) -> Requirements {
    crate::spirv::serialize::serialize(m)
        .expect("serialize")
        .requirements
}

/// A format of each rule, each with its sample planes' copy format.
fn representatives() -> Vec<(ColorResolveFormat, PlanesCopy)> {
    [
        Format::R8G8B8A8_UNORM,
        Format::B8G8R8A8_SRGB,
        Format::R16G16_SNORM,
        Format::A2B10G10R10_UNORM_PACK32,
        Format::R32G32B32A32_SFLOAT,
    ]
    .into_iter()
    .map(|format| {
        (
            ColorResolveFormat::of(format).expect("closed"),
            PlanesCopy::for_format(format).expect("copy"),
        )
    })
    .collect()
}

/// The fragment stage needs nothing of a device beyond Vulkan 1.2's shaders, at any sample count
/// and from either source.
#[test]
fn the_fragment_stage_needs_nothing_of_a_device() {
    for (format, copy) in representatives() {
        for samples in [2, 4, 8] {
            for source in [ResolveSource::InputAttachment, ResolveSource::Planes(copy)] {
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
            for source in [ResolveSource::InputAttachment, ResolveSource::Planes(copy)] {
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
    for source in [ResolveSource::InputAttachment, ResolveSource::Planes(copy)] {
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
/// four words each from `SAMPLES_IN` (the planes' bits for [`ResolveSource::Planes`], the
/// attachment's floats' bits for [`ResolveSource::InputAttachment`]), and writes the output
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
        .then(|| SrgbTables::declare(&mut b, t))
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
            texels.push(match source {
                ResolveSource::Planes(_) => texel,
                _ => f.convert(ConvOp::Bitcast, texel, t.vec4)?,
            });
        }
        let four = uint(f, t, 4)?;
        let out_base = f.binary(BinOp::IMul, i, four)?;
        for (component, channel) in format.channels.iter().enumerate() {
            let Some(channel) = *channel else { continue };
            let mut codes = Vec::new();
            for &texel in &texels {
                codes.push(code(f, t, srgb, channel, component as u32, texel, source)?);
            }
            let value = resolve(f, t, srgb, channel, &codes)?;
            let bits = f.convert(ConvOp::Bitcast, value, t.u32_)?;
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
        ResolveRule::Float16 => half_average(codes),
    }
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
    let v = q * step;
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
        ResolveRule::Unorm { bits } => (f64::from(code) / ((1u64 << bits) - 1) as f64) as f32,
        ResolveRule::Snorm { bits } => {
            let half = 1i64 << (bits - 1);
            let v = i64::from(code);
            let v = if v >= half { v - 2 * half } else { v };
            (v.max(-(half - 1)) as f64 / (half - 1) as f64) as f32
        }
        ResolveRule::Srgb => srgb_decode(code) as f32,
        ResolveRule::Float32 => f32::from_bits(code),
        ResolveRule::Float16 => half_float(code),
    }
}

/// The code an attachment of `rule` stores for the output float `x`, converting to nearest.
fn stored_as(rule: ResolveRule, x: f32) -> u32 {
    let x = f64::from(x);
    match rule {
        ResolveRule::Unorm { bits } => (x * ((1u64 << bits) - 1) as f64).round() as u32,
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

/// Run `format`'s arithmetic at `samples` from `source` over `texels` (each sample's codes by
/// component, one texel a row), and check every channel against [`metal`].
fn check(
    format: ColorResolveFormat,
    samples: usize,
    source: ResolveSource,
    texels: &[Vec<[u32; 4]>],
) {
    let m = arithmetic_module(format, samples as u32, source);
    let mut input = Vec::new();
    for texel in texels {
        for sample in texel {
            let mut words = [0u32; 4];
            for (component, channel) in format.channels.iter().enumerate() {
                let Some(channel) = channel else { continue };
                let code = sample[component];
                match source {
                    ResolveSource::Planes(_) => {
                        let offset = u32::from(channel.offset);
                        words[(offset / 32) as usize] |= code << (offset % 32);
                    }
                    _ => words[component] = read_as(channel.rule, code).to_bits(),
                }
            }
            input.extend(words.iter().flat_map(|w| w.to_le_bytes()));
        }
    }
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
                bytes: vec![0; texels.len() * 16],
            },
        ],
        push: Vec::new(),
        spec: Default::default(),
    };
    let cfg = Config {
        order: Order::Forward,
        workgroups: [texels.len() as u32, 1, 1],
        ..Config::default()
    };
    let out = run(&m, &cfg, res).expect("run").buffers.remove(1);
    for (i, texel) in texels.iter().enumerate() {
        for (component, channel) in format.channels.iter().enumerate() {
            let Some(channel) = channel else { continue };
            let at = (i * 4 + component) * 4;
            let word = u32::from_le_bytes(out[at..at + 4].try_into().expect("word"));
            let codes: Vec<u32> = texel.iter().map(|s| s[component]).collect();
            let want = metal(channel.rule, &codes);
            let got = stored_as(channel.rule, f32::from_bits(word));
            assert_eq!(
                got,
                want,
                "{:?} x{samples} {source:?} component {component}: {codes:x?} wrote {word:#x}",
                format.format()
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
        _ => {
            let mask = if bits == 32 {
                u32::MAX
            } else {
                (1u32 << bits) - 1
            };
            if r.next().is_multiple_of(4) {
                let half = 1u32 << (bits - 1);
                let edges = [0, 1, half - 1, half, half + 1, mask - 1, mask];
                edges[(r.next() % edges.len() as u64) as usize]
            } else {
                (r.next() as u32) & mask
            }
        }
    }
}

/// Every pair of 8-bit codes of `format` at 2 samples, through both sources.
fn every_pair_of_8_bit_codes(format: Format) {
    let resolve = ColorResolveFormat::of(format).expect("closed");
    let copy = PlanesCopy::for_format(format).expect("copy");
    let texels: Vec<Vec<[u32; 4]>> = (0..65536u32)
        .map(|p| vec![[p & 0xff, 0, 0, 0], [p >> 8, 0, 0, 0]])
        .collect();
    for source in [ResolveSource::InputAttachment, ResolveSource::Planes(copy)] {
        check(resolve, 2, source, &texels);
    }
}

#[test]
fn every_pair_of_unorm_codes_resolves_as_metal() {
    every_pair_of_8_bit_codes(Format::R8_UNORM);
}

#[test]
fn every_pair_of_srgb_codes_resolves_as_metal() {
    every_pair_of_8_bit_codes(Format::R8_SRGB);
}

#[test]
fn every_pair_of_snorm_codes_resolves_as_metal() {
    every_pair_of_8_bit_codes(Format::R8_SNORM);
}

/// Random and edge texels of every closed format at 2, 4 and 8 samples, through both sources.
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
    ] {
        let resolve = ColorResolveFormat::of(format).expect("closed");
        let copy = PlanesCopy::for_format(format).expect("copy");
        for samples in [2usize, 4, 8] {
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
            for source in [ResolveSource::InputAttachment, ResolveSource::Planes(copy)] {
                check(resolve, samples, source, &texels);
            }
        }
    }
}

/// Metal's resolves of 2-sample `RGBA16Float` channels measured on an M3 Max under macOS 26.7 (EV
/// `RES-hr1-m3`, our own probe `resrule.swift`): two samples' bits and the resolved bits, six of
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

/// A half format's rules hold at 2 samples only, where its modules are made; at 4 they are
/// refused, as no rule of Metal's is closed there.
#[test]
fn the_half_rule_holds_at_2_samples_only() {
    for format in [
        Format::R16_SFLOAT,
        Format::R16G16_SFLOAT,
        Format::R16G16B16A16_SFLOAT,
    ] {
        let resolve = ColorResolveFormat::of(format).expect("closed");
        assert!(resolve.resolves(2), "{format:?}");
        for samples in [4, 8] {
            assert!(!resolve.resolves(samples), "{format:?} x{samples}");
        }
    }
    let unorm = ColorResolveFormat::of(Format::R8G8B8A8_UNORM).expect("closed");
    assert!([2, 4, 8].into_iter().all(|samples| unorm.resolves(samples)));
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles/lavapipe.json");
    let text = std::fs::read_to_string(&path).expect("read");
    let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    let device = Device::new(desc, DevicePolicy::default()).expect("device");
    let half = ColorResolveFormat::of(Format::R16G16B16A16_SFLOAT).expect("closed");
    let copy = PlanesCopy::for_format(Format::R16G16B16A16_SFLOAT).expect("copy");
    for source in [ResolveSource::InputAttachment, ResolveSource::Planes(copy)] {
        color_resolve_modules(&device, half, 2, source, Feature::Format).expect("modules");
        let refused = color_resolve_modules(&device, half, 4, source, Feature::Format);
        assert!(matches!(refused, Err(Error::Internal(_))), "{source:?}");
    }
}

/// A half format's modules at 2 samples need nothing of a device and validate on every committed
/// profile.
#[test]
fn the_half_modules_are_valid_on_every_committed_profile() {
    let format = Format::R16G16_SFLOAT;
    let resolve = ColorResolveFormat::of(format).expect("closed");
    let copy = PlanesCopy::for_format(format).expect("copy");
    for source in [ResolveSource::InputAttachment, ResolveSource::Planes(copy)] {
        let r = needs(fragment_typed_module(resolve, 2, source).expect("module"));
        assert!(r.features.is_empty(), "{source:?}: {r:?}");
    }
    for (_, profile) in committed_profiles().expect("profiles") {
        for source in [ResolveSource::InputAttachment, ResolveSource::Planes(copy)] {
            let m = fragment_typed_module(resolve, 2, source).expect("module");
            let _ = assert_valid(m, &profile);
        }
    }
}

/// The independent model gives Metal's measured answers, and the modules give the model's: the
/// measured pairs, random and special pairs of every half format, and every half beside zero, its
/// neighbour, an infinity and the smallest negative half, at 2 samples through both sources.
#[test]
fn half_pairs_resolve_as_metal() {
    for [a, b, want] in MEASURED_HALF_PAIRS {
        assert_eq!(half_average(&[a, b]), want, "{a:#06x} {b:#06x}");
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
