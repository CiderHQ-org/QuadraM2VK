//! Metal's level rule against the probes' answers, and [`LevelSnap::emit`] on the interpreter
//! against [`LevelSnap::of`].

use super::LevelSnap;
use crate::air::abi::{
    AddressMode, BorderColor, CompareFunction, Coordinates, Filter, Half, MipFilter, Reduction,
    SamplerState,
};
use crate::float16::{f16_bits_to_f32, f32_to_f16_bits};
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, TypeId, Value};
use crate::spirv::interp::{self, Buffer, Config, Resources};
use crate::spirv::ir::{
    natural_layout, BinOp, ConvOp, ExecMode, ExecModel, IoDir, LayoutRule, MemAccess, StorageClass,
    Type, TypedModule,
};

/// A sampler's snap with its level clamp `[min, max]`.
fn snap(mip: MipFilter, (min, max): (f32, f32)) -> LevelSnap {
    LevelSnap {
        mip,
        min: min.to_bits(),
        max: max.to_bits(),
    }
}

/// No level clamp.
const OPEN: (f32, f32) = (0.0, f32::INFINITY);

/// The level a device's nearest mip filter reads at `λ` (Vulkan: `ceil(λ + 1/2) - 1`), in a
/// texture of `levels` levels.
fn nearest_level(lambda: f32, levels: f32) -> f32 {
    ((lambda + 0.5).ceil() - 1.0).clamp(0.0, levels - 1.0)
}

/// Our probe's `r32Float` 2D texture of five levels, level `L` holding `L`: rows of (level
/// `x`'s bits; the nearest-mip and linear-mip samples; and those of samplers clamping the level
/// into `[0.75, 2.5]`, linear then nearest), Apple M4 on macOS 27.0.1. A linear mip filter's
/// sample is `L + q`, a nearest one's the level it reads.
const MEASURED: [(u32, f32, f32, f32, f32); 23] = [
    (0x3f00_0000, 0.0, 0.5, 0.75, 1.0),
    (0x3f02_0000, 0.0, 0.5, 0.75, 1.0),
    (0x3f04_0000, 1.0, 0.515_625, 0.75, 1.0),
    (0x3fc1_0000, 1.0, 1.5, 1.5, 1.0),
    (0x3fc2_0000, 2.0, 1.515_625, 1.515_625, 2.0),
    (0x3f80_1000, 1.0, 1.0, 1.0, 1.0),
    (0x401f_8000, 2.0, 2.484_375, 2.484_375, 2.0),
    (0x3c80_0000, 0.0, 0.015_625, 0.75, 1.0),
    (0x3c7f_ffff, 0.0, 0.015_625, 0.75, 1.0),
    (0xbf00_0000, 0.0, 0.0, 0.75, 1.0),
    (0x4090_0000, 4.0, 4.0, 2.5, 2.0),
    (0x42c8_0000, 4.0, 4.0, 2.5, 2.0),
    (0x7f80_0000, 4.0, 4.0, 2.5, 2.0),
    (0x7fc0_0000, 0.0, 0.0, 0.75, 1.0),
    (0x3f03_dfff, 0.0, 0.5, 0.75, 1.0),
    (0x3f03_e000, 0.0, 0.5, 0.75, 1.0),
    (0x3f03_e001, 0.0, 0.5, 0.75, 1.0),
    (0x3f03_ffff, 1.0, 0.515_625, 0.75, 1.0),
    (0x3fc1_efff, 1.0, 1.5, 1.5, 1.0),
    (0x3fc1_f000, 2.0, 1.515_625, 1.515_625, 2.0),
    (0x3fc1_f001, 2.0, 1.515_625, 1.515_625, 2.0),
    (0x3fc1_ffff, 2.0, 1.515_625, 1.515_625, 2.0),
    (0x0000_0001, 0.0, 0.0, 0.75, 1.0),
];

/// Each measured level, handed to a device as [`LevelSnap::of`] makes it, reads Metal's level
/// under a nearest mip filter and blends Metal's weight under a linear one, without a level
/// clamp and within `[0.75, 2.5]`.
#[test]
fn the_snapped_level_reads_metals_measured_levels() {
    let levels = 5.0;
    let clamp = (0.75, 2.5);
    for (bits, nearest, linear, linear_clamped, nearest_clamped) in MEASURED {
        let x = f32::from_bits(bits);
        let n = snap(MipFilter::Nearest, OPEN).of(x);
        assert_eq!(
            nearest_level(n, levels),
            nearest,
            "nearest at {x:e} (λ {n:e})"
        );
        let l = snap(MipFilter::Linear, OPEN).of(x);
        assert_eq!(l.min(levels - 1.0), linear, "linear at {x:e}");
        let l = snap(MipFilter::Linear, clamp).of(x);
        assert_eq!(
            l.min(levels - 1.0),
            linear_clamped,
            "clamped linear at {x:e}"
        );
        let n = snap(MipFilter::Nearest, clamp).of(x);
        assert_eq!(
            nearest_level(n, levels),
            nearest_clamped,
            "clamped nearest at {x:e}"
        );
    }
}

/// A constexpr sampler of mip filter `mip` clamping the level into `[min, max]`.
fn state(mip: MipFilter, (min, max): (f32, f32)) -> SamplerState {
    SamplerState {
        address_s: AddressMode::ClampToEdge,
        address_t: AddressMode::ClampToEdge,
        address_r: AddressMode::ClampToEdge,
        mag_filter: Filter::Linear,
        min_filter: Filter::Linear,
        mip_filter: mip,
        coordinates: Coordinates::Normalized,
        compare: CompareFunction::None,
        max_anisotropy: 1,
        lod_min_clamp: Half(f32_to_f16_bits(min)),
        lod_max_clamp: Half(f32_to_f16_bits(max)),
        border_color: BorderColor::TransparentBlack,
        reduction: Reduction::WeightedAverage,
        lod_bias: Half(0),
    }
}

/// The snap reaches a level clamp only where the device's `minLod` never moves the level it hands
/// over: on the 1/64 grid under a linear mip filter, off `[1/2, 33/64)` under a nearest one, and
/// any without a mip filter; and no pixel, anisotropic, or biasing sampler.
#[test]
fn the_snap_reaches_clamps_the_device_keeps() {
    use MipFilter::{Linear, Nearest, None as Unmipped};
    let reached = |mip, clamp| LevelSnap::plan(&state(mip, clamp)).is_some();
    for (mip, min, want) in [
        (Linear, 0.0, true),
        (Linear, 0.75, true),
        (Linear, 2.0 + 5.0 / 64.0, true),
        (Linear, 0.3, false),
        (Linear, 1.0 + 1.0 / 128.0, false),
        (Nearest, 0.3, true),
        (Nearest, 0.75, true),
        (Nearest, 1.0 + 33.0 / 64.0, true),
        (Nearest, 0.5, false),
        (Nearest, 2.51, false),
        (Unmipped, 0.3, true),
    ] {
        assert_eq!(reached(mip, (min, 1000.0)), want, "{mip:?} from {min}");
    }
    assert!(!reached(Linear, (2.0, 1.0)), "an empty clamp");
    let mut s = state(Linear, OPEN);
    s.coordinates = Coordinates::Pixel;
    assert!(LevelSnap::plan(&s).is_none(), "a pixel sampler");
    let mut s = state(Linear, OPEN);
    s.max_anisotropy = 4;
    assert!(LevelSnap::plan(&s).is_none(), "an anisotropic sampler");
    let mut s = state(Linear, OPEN);
    s.lod_bias = Half(f32_to_f16_bits(0.5));
    assert!(LevelSnap::plan(&s).is_none(), "a biasing sampler");
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

/// Word `k` of invocation `i`'s record of `per` words in `buf`.
fn word(f: &mut FunctionBuilder<'_>, buf: GlobalId, i: Value, per: u32, k: u32) -> BResult<Value> {
    let per = f.u32_const(per)?;
    let per = f.module().const_value(per)?;
    let at = f.binary(BinOp::IMul, i, per)?;
    let k = f.u32_const(k)?;
    let k = f.module().const_value(k)?;
    let at = f.binary(BinOp::IAdd, at, k)?;
    let zero = f.u32_const(0)?;
    let zero = f.module().const_value(zero)?;
    let base = f.module().global_value(buf)?;
    f.access_chain(base, &[zero, at])
}

/// A kernel whose invocation `i` reads a level's bits from buffer 0 and writes the bits of each
/// of `snaps`' level to buffer 1.
fn kernel(snaps: &[LevelSnap]) -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let f32_ = b.ty(Type::Float { width: 32 })?;
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
    let input = storage_buffer(&mut b, 0, u32_)?;
    let output = storage_buffer(&mut b, 1, u32_)?;
    let id = b.builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let gid = f.module().global_value(id)?;
    let gid = f.load(gid, MemAccess::default())?;
    let i = f.composite_extract(gid, &[0])?;
    let at = word(&mut f, input, i, 1, 0)?;
    let xb = f.load(at, MemAccess::default())?;
    let x = f.convert(ConvOp::Bitcast, xb, f32_)?;
    let per = u32::try_from(snaps.len()).expect("snaps");
    for (k, s) in snaps.iter().enumerate() {
        let v = s.emit(&mut f, x)?;
        let bits = f.convert(ConvOp::Bitcast, v, u32_)?;
        let at = word(&mut f, output, i, per, k as u32)?;
        f.store(at, bits, MemAccess::default())?;
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

/// The levels the snap is checked at: the measured ones, every `1/256` from -2 to 18, each `half`
/// of `[1/4, 6)` with its tie to the next and the tie's `f32` neighbours, and the edges.
fn levels() -> Vec<u32> {
    let mut out: Vec<u32> = MEASURED.iter().map(|m| m.0).collect();
    out.extend((-512..=4608).map(|k| (k as f32 / 256.0).to_bits()));
    for h in 0x3400u16..0x4600 {
        let (v, next) = (f16_bits_to_f32(h), f16_bits_to_f32(h + 1));
        // Exact: two neighbouring halves' midpoint has 12 significant bits.
        let tie = ((v + next) / 2.0).to_bits();
        out.extend([v.to_bits(), tie - 1, tie, tie + 1]);
    }
    out.extend([
        0x7f80_0000,
        0xff80_0000,
        0x7fc0_0000,
        0x8000_0000,
        0x477f_e000,
        0x477f_f000,
        0x4780_0000,
    ]);
    out
}

/// [`LevelSnap::emit`] is [`LevelSnap::of`], bit for bit, under every mip filter, without and with
/// a level clamp, at every level of [`levels`].
#[test]
fn the_snap_computes_metals_level_in_the_function() {
    let snaps = [
        snap(MipFilter::Nearest, OPEN),
        snap(MipFilter::Linear, OPEN),
        snap(MipFilter::None, OPEN),
        snap(MipFilter::Nearest, (0.300_048_83, 2.699_218_8)),
        snap(MipFilter::Linear, (0.75, 2.5)),
    ];
    let m = kernel(&snaps).expect("kernel");
    let lanes = levels();
    let per = snaps.len();
    let bytes: Vec<u8> = lanes.iter().flat_map(|b| b.to_le_bytes()).collect();
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
                bytes: vec![0xAB; lanes.len() * per * 4],
            },
        ],
        ..Resources::default()
    };
    let cfg = Config {
        workgroups: [u32::try_from(lanes.len()).expect("lanes"), 1, 1],
        ..Config::default()
    };
    let (out, _) = interp::run_with_images(&m, &cfg, res, Vec::new()).expect("run");
    let mut wrong = Vec::new();
    for (i, &b) in lanes.iter().enumerate() {
        let x = f32::from_bits(b);
        for (k, s) in snaps.iter().enumerate() {
            let at = 4 * (i * per + k);
            let got = u32::from_le_bytes(out.buffers[1][at..at + 4].try_into().expect("word"));
            let want = s.of(x).to_bits();
            if got != want {
                wrong.push(format!(
                    "{s:?} at {x:e} ({b:#010x}): {got:#010x}, want {want:#010x}"
                ));
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{} differ:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

/// The word rule's level handed to the device, and the level before the sampler's clamp a
/// nearest mip filter reads, at level `x` through a runtime sampler of word `word`, as the host
/// computes them (`by_word`).
fn word_rule(word: u32, x: f32) -> (f32, f32) {
    use crate::metal::sampler::SamplerWord as W;
    let h = if x.is_nan() {
        0.0
    } else {
        f16_bits_to_f32(f32_to_f16_bits(x))
    };
    let h = h.clamp(0.0, 64.0);
    let l = h.floor();
    let q = ((h - l) * 64.0).floor() / 64.0;
    let nearest = l + if q > 0.5 { 1.0 } else { 0.0 };
    let rule = word & W::LEVEL_RULE;
    let both = W::MIN_LINEAR | W::MAG_LINEAR;
    let (n, li) = (1 << W::MIP_FILTER_SHIFT, 2 << W::MIP_FILTER_SHIFT);
    let lambda = if rule == n || rule == n | both {
        nearest
    } else if rule == li || rule == li | both {
        l + q
    } else {
        x
    };
    (lambda, nearest)
}

/// The level a runtime sampler's leveled nearest snap reads before the texture's levels clamp it
/// (`snap_level`), as the host computes it.
fn snap_level_of(word: u32, nearest: f32) -> u32 {
    use crate::metal::sampler::{SamplerWord as W, LAST_WORD_LEVEL};
    if word & W::MIP_FILTER == 0 {
        return 0;
    }
    let min = (word >> W::LEVEL_MIN_SHIFT) & 0xf;
    let max = LAST_WORD_LEVEL - ((word >> W::LEVEL_MAX_SHIFT) & 0xf);
    (nearest as u32).max(min).min(max)
}

/// Metal's level and weight at level `x` through a runtime sampler clamping it into `[lo, hi]`
/// (`f32`s) in a texture of `levels` levels: the rule with the clamp held as `f32`s, which our
/// probe reads on all 37,080 of its lanes (Apple M4, macOS 27.0.1). The nearest mip
/// filter's level, and the linear one's `L + q`.
fn metal_runtime(x: f32, (lo, hi): (f32, f32), levels: f32) -> (f32, f32) {
    let h = if x.is_nan() {
        0.0
    } else {
        f16_bits_to_f32(f32_to_f16_bits(x))
    };
    let v = h.max(lo).min(hi).max(0.0).min(levels - 1.0);
    let l = v.floor();
    let q = ((v - l) * 64.0).floor() / 64.0;
    (l + if q > 0.5 { 1.0 } else { 0.0 }, l + q)
}

/// Rows our probe measured through runtime samplers (Apple M4, macOS 27.0.1, an
/// `r32Float` texture of five levels, level `L` holding `L`): the clamp `[lo, hi]`'s bits, level
/// `x`'s bits, and the nearest and linear mip filters' samples (nearest filters). Bounds just
/// below `33/64` and `21/64` read as `f32`s, where `half`s would round them onto the step.
const MEASURED_RUNTIME: [(u32, u32, u32, f32, f32); 9] = [
    // `[0.515_624_94, 9]` (just below 33/64) at -1: level 0, weight 32/64.
    (0x3f03_ffff, 0x4110_0000, 0xbf80_0000, 0.0, 0.5),
    // `[33/64, 9]` at -1: level 1, weight 33/64.
    (0x3f04_0000, 0x4110_0000, 0xbf80_0000, 1.0, 0.515_625),
    // `[0.328_124_97, 9]` (just below 21/64) at 0: weight 20/64.
    (0x3ea7_ffff, 0x4110_0000, 0x0000_0000, 0.0, 0.3125),
    // `[0.75, 2.5]` at 3: level 2 and `2.5`.
    (0x3f40_0000, 0x4020_0000, 0x4040_0000, 2.0, 2.5),
    // `[0.3, 2.7]` at -1 and at 4.
    (0x3e99_999a, 0x402c_cccd, 0xbf80_0000, 0.0, 0.296_875),
    (0x3e99_999a, 0x402c_cccd, 0x4080_0000, 3.0, 2.687_5),
    // `[0, 2.4999]` at 3: level 2, weight 31/64.
    (0x0000_0000, 0x401f_fe5d, 0x4040_0000, 2.0, 2.484_375),
    // `[0, 1 + 33/64]` at 4.
    (0x0000_0000, 0x3fc2_0000, 0x4080_0000, 2.0, 1.515_625),
    // No clamp, at a NaN.
    (0x0000_0000, 0x7f7f_ffff, 0x7fc0_0000, 0.0, 0.0),
];

/// A runtime sampler's level handed to a device as the word rule makes it, through the `VkSampler`
/// the runtime makes of it (its clamp on Metal's grid, `metal::sampler::vk_sampler`), reads Metal's
/// level under a nearest mip filter and Metal's `L + q` under a linear one, at every level of
/// [`levels`] and through every clamp our probe measured; and Metal's rule with `f32`
/// clamps answers the probe's measured rows.
#[test]
fn a_runtime_samplers_word_level_reads_metals_level_through_its_grid_clamp() {
    use crate::metal::sampler::{level_on_grid, nearest_level, SamplerWord as W};
    for &(lo, hi, x, nearest, linear) in &MEASURED_RUNTIME {
        let (lo, hi, x) = (f32::from_bits(lo), f32::from_bits(hi), f32::from_bits(x));
        assert_eq!(
            metal_runtime(x, (lo, hi), 5.0),
            (nearest, linear),
            "[{lo:e}, {hi:e}] at {x:e}"
        );
    }
    let below = |x: f32| x.next_down();
    let clamps = [
        (0.0, f32::MAX),
        (0.75, 2.5),
        (0.3, 2.7),
        (0.5, 9.0),
        (below(33.0 / 64.0), 9.0),
        (33.0 / 64.0, 9.0),
        (0.507_812_5, 9.0),
        (below(21.0 / 64.0), 9.0),
        (1.5, 9.0),
        (1.0 + below(33.0 / 64.0), 9.0),
        (0.0, 1.5),
        (0.0, 1.0 + below(33.0 / 64.0)),
        (0.0, 1.0 + 33.0 / 64.0),
        (0.0, 2.4999),
        (0.0, 1.0 + below(21.0 / 64.0)),
        (2.2, 2.2),
        (6.0, 9.0),
        (0.0001, 9.0),
    ];
    let count = 5.0;
    let lanes = levels();
    let mut checked = 0;
    for (lo, hi) in clamps {
        for filters in [0, W::MIN_LINEAR | W::MAG_LINEAR] {
            for (mip, nearest_mip) in [(1u32, true), (2, false)] {
                let word = filters | (mip << W::MIP_FILTER_SHIFT);
                let (min_lod, max_lod) = if nearest_mip {
                    (nearest_level(lo) as f32, nearest_level(hi) as f32)
                } else {
                    (level_on_grid(lo), level_on_grid(hi))
                };
                for &b in &lanes {
                    let x = f32::from_bits(b);
                    let (lambda, _) = word_rule(word, x);
                    let handed = lambda.max(min_lod).min(max_lod);
                    let (metal_nearest, metal_linear) = metal_runtime(x, (lo, hi), count);
                    if nearest_mip {
                        assert_eq!(
                            nearest_level(handed.min(count - 1.0)) as f32,
                            metal_nearest,
                            "[{lo:e}, {hi:e}] word {word:#x} at {x:e}: handed {handed:e}"
                        );
                        assert_eq!(handed.fract(), 0.0, "a whole level: {handed:e}");
                    } else {
                        assert_eq!(
                            handed.min(count - 1.0),
                            metal_linear,
                            "[{lo:e}, {hi:e}] word {word:#x} at {x:e}"
                        );
                    }
                    checked += 1;
                }
            }
        }
    }
    assert!(checked > 100_000, "{checked}");
}

/// A kernel whose invocation `i` reads `(bits(x), word)` from buffer 0 and writes the bits of the
/// word rule's level and `snap_level`'s level of it to buffer 1.
fn word_kernel() -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let f32_ = b.ty(Type::Float { width: 32 })?;
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
    let input = storage_buffer(&mut b, 0, u32_)?;
    let output = storage_buffer(&mut b, 1, u32_)?;
    let id = b.builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let gid = f.module().global_value(id)?;
    let gid = f.load(gid, MemAccess::default())?;
    let i = f.composite_extract(gid, &[0])?;
    let at = word(&mut f, input, i, 2, 0)?;
    let xb = f.load(at, MemAccess::default())?;
    let x = f.convert(ConvOp::Bitcast, xb, f32_)?;
    let at = word(&mut f, input, i, 2, 1)?;
    let w = f.load(at, MemAccess::default())?;
    let (lambda, nearest) = super::by_word(&mut f, w, x)?;
    let level = super::snap_level(&mut f, w, nearest)?;
    let bits = f.convert(ConvOp::Bitcast, lambda, u32_)?;
    let at = word(&mut f, output, i, 2, 0)?;
    f.store(at, bits, MemAccess::default())?;
    let at = word(&mut f, output, i, 2, 1)?;
    f.store(at, level, MemAccess::default())?;
    f.ret(None)?;
    let main = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        main,
        vec![ExecMode::LocalSize { x: 1, y: 1, z: 1 }],
    )?;
    b.finalize()
}

/// `by_word` and `snap_level` in the function are the host's word rule, bit for bit, for words of
/// every mip filter, filters alike and not, coordinates, anisotropy and clamp levels, at every
/// level of [`levels`].
#[test]
fn the_word_rule_computes_metals_level_in_the_function() {
    use crate::metal::sampler::SamplerWord as W;
    let clamp = |min: u32, max: u32| {
        (min << W::LEVEL_MIN_SHIFT)
            | ((crate::metal::sampler::LAST_WORD_LEVEL - max) << W::LEVEL_MAX_SHIFT)
    };
    let mut words = Vec::new();
    for mip in 0..3u32 {
        for filters in [
            0,
            W::MIN_LINEAR,
            W::MAG_LINEAR,
            W::MIN_LINEAR | W::MAG_LINEAR,
        ] {
            for extra in [0, W::PIXEL_COORDINATES, W::ANISOTROPIC] {
                for (min, max) in [(0, 15), (1, 2), (3, 3)] {
                    words.push((mip << W::MIP_FILTER_SHIFT) | filters | extra | clamp(min, max));
                }
            }
        }
    }
    let xs = levels();
    let step = (xs.len() / 600).max(1);
    let xs: Vec<u32> = xs.into_iter().step_by(step).collect();
    let lanes: Vec<(u32, u32)> = words
        .iter()
        .flat_map(|&w| xs.iter().map(move |&x| (x, w)))
        .collect();
    let m = word_kernel().expect("kernel");
    let bytes: Vec<u8> = lanes
        .iter()
        .flat_map(|(x, w)| [x.to_le_bytes(), w.to_le_bytes()])
        .flatten()
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
                bytes: vec![0xAB; lanes.len() * 8],
            },
        ],
        ..Resources::default()
    };
    let cfg = Config {
        workgroups: [u32::try_from(lanes.len()).expect("lanes"), 1, 1],
        ..Config::default()
    };
    let (out, _) = interp::run_with_images(&m, &cfg, res, Vec::new()).expect("run");
    let mut wrong = Vec::new();
    for (i, &(b, w)) in lanes.iter().enumerate() {
        let x = f32::from_bits(b);
        let at = 8 * i;
        let got = |k: usize| {
            u32::from_le_bytes(
                out.buffers[1][at + 4 * k..at + 4 * k + 4]
                    .try_into()
                    .expect("word"),
            )
        };
        let (lambda, nearest) = word_rule(w, x);
        let want = (lambda.to_bits(), snap_level_of(w, nearest));
        if (got(0), got(1)) != want {
            wrong.push(format!(
                "word {w:#x} at {x:e} ({b:#010x}): ({:#010x}, {}), want ({:#010x}, {})",
                got(0),
                got(1),
                want.0,
                want.1
            ));
        }
    }
    assert!(
        wrong.is_empty(),
        "{} differ:\n{}",
        wrong.len(),
        wrong[..wrong.len().min(40)].join("\n")
    );
    assert!(lanes.len() > 50_000, "{}", lanes.len());
}
