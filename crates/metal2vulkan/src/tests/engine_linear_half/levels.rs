//! The level a depth comparison through a runtime sampler reads: where the sampler's word reads one
//! texel, the shader fetches it from the level the word reads at the sample's level or a kernel's
//! bias (`fetched::FetchLevel::Word`), on the CPU interpreter against Metal's rule, which our probe
//! measured through runtime and argument buffer's samplers: the level rounded to
//! `half` (a NaN read as 0), clamped into the sampler's level clamp as `f32`s and into the
//! texture's levels, and under a nearest mip filter `L + 1` where its fraction truncated onto
//! sixty-fourths passes one half; level 0 without a mip filter.

use super::*;
use crate::bind::SamplerDescriptor;
use crate::float16::{f16_bits_to_f32, f32_to_f16_bits};
use crate::metal::{CompareFunction, SamplerMipFilter};
use crate::pipeline::artifact::SpecializedNote;

/// How a kernel's `sample_compare` states its level: `level(x)`, a kernel's `bias(x)` (the
/// explicit level `x`), or neither (level 0).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Stated {
    Level,
    Bias,
    Neither,
}

/// A kernel comparing `depth2d<float>` slot 0 through runtime sampler slot 0 at buffer 0's lane
/// `tid` (a `float4`: the coordinate, the reference in `z`, and the level in `w`) at the level
/// `stated` names; buffer 1's lane `tid` is a `float4` whose `x` is the pass.
pub(super) fn runtime_level_kernel(stated: Stated) -> String {
    let (level, lod) = match stated {
        Stated::Level => ("true", "%lod"),
        Stated::Bias => ("false", "%lod"),
        Stated::Neither => ("false", "0.0"),
    };
    format!(
        r#"define void @k(ptr addrspace(1) %t, ptr addrspace(1) %c, ptr addrspace(1) %out, ptr addrspace(2) %s, i32 %tid) {{
  %i = zext i32 %tid to i64
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %c, i64 %i
  %c4 = load <4 x float>, ptr addrspace(1) %p, align 16
  %uv = shufflevector <4 x float> %c4, <4 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %ref = extractelement <4 x float> %c4, i64 2
  %lod = extractelement <4 x float> %c4, i64 3
  %r = call {{ float, i8 }} @air.sample_compare_depth_2d.f32(ptr addrspace(1) %t, ptr addrspace(2) %s, i32 1, <2 x float> %uv, float %ref, i1 true, <2 x i32> zeroinitializer, i1 {level}, float {lod}, float 0.0, i32 0)
  %v = extractvalue {{ float, i8 }} %r, 0
  %v4 = insertelement <4 x float> zeroinitializer, float %v, i64 0
  %q = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i64 %i
  store <4 x float> %v4, ptr addrspace(1) %q, align 16
  ret void
}}

declare {{ float, i8 }} @air.sample_compare_depth_2d.f32(ptr addrspace(1), ptr addrspace(2), i32, <2 x float>, float, i1, <2 x i32>, i1, float, float, i32)

!air.kernel = !{{!0}}
!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4, !5, !6, !7}}
!3 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"depth2d<float, sample>", !"air.arg_name", !"t"}}
!4 = !{{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"c"}}
!5 = !{{i32 2, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}}
!6 = !{{i32 3, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"s"}}
!7 = !{{i32 4, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}}
"#
    )
}

/// The level Metal reads at level `x` (`None`: level 0) through a sampler of nearest filters and
/// mip filter `mip` whose level clamp is `(min, max)` (`f32`s, a runtime sampler's), of a texture
/// of `levels` levels (module docs).
fn metal_level(x: Option<f32>, mip: SamplerMipFilter, (min, max): (f32, f32), levels: u32) -> u32 {
    if mip == SamplerMipFilter::NotMipmapped {
        return 0;
    }
    let x = match x {
        Some(x) if !x.is_nan() => f16_bits_to_f32(f32_to_f16_bits(x)),
        _ => 0.0,
    };
    let x = x.max(min).min(max).clamp(0.0, (levels - 1) as f32);
    let whole = x.floor();
    let q = ((x - whole) * 64.0).floor() / 64.0;
    whole as u32 + u32::from(q > 0.5)
}

/// Whether specialized note `n` holds where the runtime sampler's word is `word` and the texture
/// slot's `TexelFormat` is `class` (a `SamplerState` specialization is the word masked by its
/// fields).
fn holds(n: &SpecializedNote, word: u32, class: u32) -> bool {
    n.when.iter().any(|clause| {
        clause.all.iter().all(|t| {
            let v = match t.key {
                SpecKey::SamplerState { fields, .. } => word & fields,
                SpecKey::TexelFormat(_) => class,
                other => panic!("a test of {other:?}"),
            };
            ((v & t.mask) == t.bits) == t.equal
        })
    })
}

/// A runtime sampler's `sample_compare` with a level, a kernel's bias, or neither, fetches its
/// texel from the level its word reads where the word reads one texel, on every lane of the
/// interpreter: nearest filters under a nearest mip filter through level clamps of whole, half-way,
/// and sixty-fourth bounds (and an empty range of levels), and without a mip filter; levels below
/// 0, past the texture's three, at and around the half-way points, the infinities, and a NaN, on
/// `Depth32Float` and `Depth16Unorm`. The module reads the word's mip filter and level clamp, and
/// records `DeviceDepthComparison` only where the word filters.
#[test]
fn a_runtime_comparison_reads_the_level_its_word_reads() {
    let levels = [
        -1.0f32,
        0.0,
        0.25,
        0.5,
        0.5078125,
        0.515625,
        0.75,
        1.0,
        1.4,
        1.5,
        1.53125,
        2.0,
        2.6,
        7.0,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
    ];
    let clamps = [
        (0.0f32, f32::MAX),
        (0.75, 2.5),
        (1.515625, 1.75),
        (0.5, 0.5),
        (0.0, 0.0),
        (2.0, 100.0),
        (0.3, 1.49),
    ];
    let p = lavapipe();
    let (w, h) = (12i64, 8i64);
    for stated in [Stated::Level, Stated::Bias, Stated::Neither] {
        let name = format!("{stated:?}");
        let Some(bc) = assemble(&name, &runtime_level_kernel(stated)) else {
            return;
        };
        let a = compile(&bc, "k", &p).unwrap_or_else(|e| panic!("{name}: {e}"));
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{name}: {e}"));
        let fields = a
            .interface
            .read_keys()
            .into_iter()
            .find_map(|k| match k {
                SpecKey::SamplerState { fields, .. } => Some(fields),
                _ => None,
            })
            .unwrap_or_else(|| panic!("{name}: no sampler word"));
        let levels_read = SamplerWord::MIP_FILTER | SamplerWord::LEVEL_MIN | SamplerWord::LEVEL_MAX;
        assert_eq!(fields & levels_read, levels_read, "{name}");
        let notes: Vec<&SpecializedNote> = a
            .specialized
            .iter()
            .filter(|n| n.note == PrecisionNote::DeviceDepthComparison)
            .collect();
        let [note] = notes[..] else {
            panic!("{name}: the device comparison's note is not specialized: {notes:?}");
        };
        let (plan, m) = typed_module(&bc, "k", &p).expect("typed module");
        for (mip, clamp) in clamps
            .iter()
            .map(|&c| (SamplerMipFilter::Nearest, c))
            .chain([(SamplerMipFilter::NotMipmapped, (0.0, f32::MAX))])
        {
            let mut desc = SamplerDescriptor::new();
            desc.mip_filter = mip;
            (desc.lod_min_clamp, desc.lod_max_clamp) = clamp;
            desc.compare_function = CompareFunction::LessEqual;
            let word = SamplerWord::of(&desc).0;
            for (seed, pf) in DEPTHS.iter().enumerate() {
                let what = format!("{name} {mip:?} {clamp:?} {pf:?}");
                let class = texel_format(*pf).0;
                assert!(!holds(note, word, class), "{what}: noted");
                for linear in [SamplerWord::MIN_LINEAR, 2 << SamplerWord::MIP_FILTER_SHIFT] {
                    let filtered = word & !SamplerWord::MIP_FILTER | linear;
                    assert!(holds(note, filtered, class), "{what}: not noted");
                }
                let texs: Vec<Texels> = (0..3)
                    .map(|l| Texels::random(class, w >> l, h >> l, seed as u64 + 71 + l as u64))
                    .collect();
                // The comparison's binding and the one without comparison it fetches through hold
                // the same texels.
                let images: Vec<Image> = combined_slots(&m)
                    .iter()
                    .map(|&(slot, _)| {
                        let mut image = Image::new(slot, [w as u32, h as u32, 1], 1, 3, 1);
                        for (l, tex) in texs.iter().enumerate() {
                            for y in 0..tex.h as u32 {
                                for x in 0..tex.w as u32 {
                                    let at = image.index(l as u32, 0, 0, [x, y, 0]).expect("texel");
                                    let v = tex.texel(0, i64::from(x), i64::from(y));
                                    image.texels[at] = v.map(|c| tex.fetched(c).to_bits());
                                }
                            }
                        }
                        image
                    })
                    .collect();
                let lanes = compare_lanes(&texs[0], w, h, &levels);
                let bytes: Vec<u8> = lanes
                    .iter()
                    .flat_map(|(x, y, r, l)| {
                        [(x / w as f64) as f32, (y / h as f64) as f32, *r, *l].map(f32::to_le_bytes)
                    })
                    .flatten()
                    .collect();
                let res = resources(&plan, &[bytes, vec![0xab; 16 * lanes.len()]], class);
                let res = with_word(res, &a, Some(word));
                let out = run_images(&m, &res, &images, lanes.len() as u32, false);
                for (lane, ((x, y, r, l), got)) in lanes.iter().zip(&out).enumerate() {
                    let x_level = (stated != Stated::Neither).then_some(*l);
                    let level = metal_level(x_level, mip, clamp, 3) as usize;
                    let tex = &texs[level];
                    let (u, v) = (
                        f64::from((x / w as f64) as f32),
                        f64::from((y / h as f64) as f32),
                    );
                    let (lw, lh) = (tex.w as f64, tex.h as f64);
                    let (sx, sy) = (
                        seen(u * lw, lw, Address::Edge, false),
                        seen(v * lh, lh, Address::Edge, false),
                    );
                    let at = (sx.floor() as i64, sy.floor() as i64);
                    let want = tap_pass(tex, 2, *r, tap(tex, Address::Edge, 0, at, 0));
                    assert_eq!(
                        got[0], want,
                        "{what} lane {lane} ({x}, {y}) level {l} (Metal's {level}) reference {r}"
                    );
                }
            }
        }
    }
}

/// Metal's blend of two levels' passes `[p0, p1]` (each 0 or 1) by `1 - q` and `q`: the passes as
/// `r8Unorm` codes 0 and 255, their sum `255 (p0 (1 - q) + p1 q)` rounded once onto sixteenths of a
/// code, ties up, divided by 255.
fn blend([p0, p1]: [f32; 2], q: f32) -> f32 {
    let m = (q * 64.0) as u32;
    let s = p0 as u32 * (64 - m) + p1 as u32 * m;
    let n = (255 * s + 2) / 4;
    n as f32 / 4080.0
}

/// A `sample_compare` through a constexpr sampler of nearest filters under a linear mip filter, at
/// a dynamic level, blends two levels' comparisons in the shader where the slot's format is
/// measured, on every lane of the interpreter: with `L + q` the level Metal's rule reads (rounded
/// to `half`, clamped into the sampler's level clamp as `half`s and into the texture's three
/// levels, its fraction truncated onto sixty-fourths), the comparison of level `L`'s texel weighted
/// `1 - q` and level `L + 1`'s weighted `q` as Metal blends them ([`blend`]), through clamps of
/// none, of whole and half-way bounds, and of a sixty-fourth's fraction, on `Depth32Float` and
/// `Depth16Unorm`. The device's comparison stays in the arm a format of no measured class takes,
/// where the module notes it and the filter.
#[test]
fn a_linear_mip_filters_comparison_blends_metals_two_levels() {
    use crate::lower::sampler::level::LevelSnap;
    let levels = [
        -1.0f32,
        0.0,
        0.015625,
        0.25,
        0.5,
        0.5078125,
        0.75,
        0.99,
        1.0,
        1.4,
        1.53125,
        1.984375,
        2.0,
        2.6,
        7.0,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
    ];
    // (lod_min_clamp, lod_max_clamp) as halfs: none, whole and half-way bounds, a sixty-fourth's.
    let clamps: [(u16, u16); 4] = [
        (0x0000, 0x7bff),
        (0x3a00, 0x4100),
        (0x3c10, 0x3e00),
        (0x3e00, 0x3e00),
    ];
    let p = lavapipe();
    let (w, h) = (12i64, 8i64);
    for (min, max) in clamps {
        let words = sampler_words(0, 2, 1, false, 0) & !(0xf << 16) & !(0xffff_ffff << 24)
            | 2 << 16
            | u64::from(min) << 24
            | u64::from(max) << 40;
        let name = format!("clamp {min:#06x} {max:#06x}");
        let Some(bc) = assemble(&name, &level_compare_kernel(words)) else {
            return;
        };
        let a = compile(&bc, "k", &p).unwrap_or_else(|e| panic!("{name}: {e}"));
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{name}: {e}"));
        for note in [
            PrecisionNote::DeviceDepthComparison,
            PrecisionNote::FloatSampleFilter,
        ] {
            let n = a
                .specialized
                .iter()
                .find(|n| n.note == note)
                .unwrap_or_else(|| panic!("{name}: {note:?} is not specialized"));
            for pf in DEPTHS {
                assert!(
                    !holds(n, 0, texel_format(pf).0),
                    "{name}: {note:?} on {pf:?}"
                );
            }
            assert!(holds(n, 0, 0), "{name}: {note:?} on an unmeasured class");
        }
        let snap = LevelSnap {
            mip: crate::air::abi::MipFilter::Linear,
            min: f16_bits_to_f32(min).to_bits(),
            max: f16_bits_to_f32(max).to_bits(),
        };
        let (plan, m) = typed_module(&bc, "k", &p).expect("typed module");
        for (seed, pf) in DEPTHS.iter().enumerate() {
            let what = format!("{name} {pf:?}");
            let class = texel_format(*pf).0;
            let texs: Vec<Texels> = (0..3)
                .map(|l| Texels::random(class, w >> l, h >> l, seed as u64 + 83 + l as u64))
                .collect();
            let images: Vec<Image> = combined_slots(&m)
                .iter()
                .map(|&(slot, _)| {
                    let mut image = Image::new(slot, [w as u32, h as u32, 1], 1, 3, 1);
                    for (l, tex) in texs.iter().enumerate() {
                        for y in 0..tex.h as u32 {
                            for x in 0..tex.w as u32 {
                                let at = image.index(l as u32, 0, 0, [x, y, 0]).expect("texel");
                                let v = tex.texel(0, i64::from(x), i64::from(y));
                                image.texels[at] = v.map(|c| tex.fetched(c).to_bits());
                            }
                        }
                    }
                    image
                })
                .collect();
            let lanes = compare_lanes(&texs[0], w, h, &levels);
            let bytes: Vec<u8> = lanes
                .iter()
                .flat_map(|(x, y, r, l)| {
                    [(x / w as f64) as f32, (y / h as f64) as f32, *r, *l].map(f32::to_le_bytes)
                })
                .flatten()
                .collect();
            let res = resources(&plan, &[bytes, vec![0xab; 16 * lanes.len()]], class);
            let out = run_images(&m, &res, &images, lanes.len() as u32, false);
            for (lane, ((x, y, r, l), got)) in lanes.iter().zip(&out).enumerate() {
                let lambda = snap.of(*l).min(2.0);
                let whole = lambda.floor();
                let q = lambda - whole;
                let pass = |level: usize| {
                    let tex = &texs[level.min(2)];
                    let (u, v) = (
                        f64::from((x / w as f64) as f32),
                        f64::from((y / h as f64) as f32),
                    );
                    let (lw, lh) = (tex.w as f64, tex.h as f64);
                    let (sx, sy) = (
                        seen(u * lw, lw, Address::Edge, false),
                        seen(v * lh, lh, Address::Edge, false),
                    );
                    let at = (sx.floor() as i64, sy.floor() as i64);
                    f32::from_bits(tap_pass(tex, 2, *r, tap(tex, Address::Edge, 0, at, 0)))
                };
                let want = blend([pass(whole as usize), pass(whole as usize + 1)], q);
                assert_eq!(
                    got[0],
                    want.to_bits(),
                    "{what} lane {lane} ({x}, {y}) level {l} (Metal's {lambda}) reference {r}"
                );
            }
        }
    }
}
