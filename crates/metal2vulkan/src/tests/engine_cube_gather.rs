//! Metal's gather of a cube texture, emulated by fetches on a device that does not sample through
//! Metal's own sampler, on the CPU interpreter:
//!
//! - Metal's own answers: eight lanes of a probe (Apple M4, macOS 27.0.1; its `gs`
//!   run on the identity cube of 3x3 faces, `lanes-3.bin`), bit for bit: channel `w` through a
//!   nearest sampler clamping to the edge and through one repeating, and channel `x` through the
//!   first. Five lanes read across a face's edge and past both edges, and three wrap a face
//!   coordinate of 1.
//! - The rule: random texels of every format family on faces of 1, 2, 3, and 5 texels and a cube
//!   array's cubes, every channel, `float` and `half` results, bit for bit against the reference
//!   of the linear cube emulation's tests (`engine_linear_cube`): the four taps of the linear
//!   filter's footprint in gather order, a tap past both edges a normalized format's made-up texel
//!   and a float format's `x` face's, a `half` result the texel rounded to nearest even. Integer
//!   results, the tap past both edges the `x` face's texel. An 8-bit sRGB format through the
//!   slot's raw view under its variant: each tap's table step, the corner's made-up step.
//! - A repeating face axis wraps a face coordinate of 1 on faces of 3 and 5 texels: the sampler's
//!   `s` address the face's `s` axis and its `t` address the `t` axis, its `r` address neither.
//! - The selection and the interface: an emulated gather is `sampler.gather.cube`, records
//!   `CubeFaceReciprocal` (and a `float` or `half` one `DeviceGatherFootprint`, for a format whose
//!   corner rule is not measured), and declares the slot's 2D-array view of its faces; a device
//!   with `VkProperty::MetalSampler` keeps every gather.
//! - The word estimate bounds the emission, which validates, on every committed profile.
//!
//! `llvm-as` assembles the kernels; without it the tests skip visibly.

use crate::budget::SPIRV_WORDS;
use crate::device::requirements::{VkFeature, VkProperty};
use crate::library::{FunctionArtifact, FunctionConstantValues};
use crate::metal::texel_format::{texel_format, CubeCorner, TexelFormat};
use crate::metal::PixelFormat;
use crate::pipeline::compile::{compile_bitcode, plan_bitcode, EntryRef};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::sym::{ResourceKind, SpecKey};
use crate::pipeline::testing::{compile, link_and_validate, meter, typed_module_with};
use crate::pipeline::PrecisionNote;
use crate::spirv::testing::committed_exact_profiles;

use super::engine_linear_cube::{
    assemble, directions, edge_directions, footprint, images, inputs, lavapipe, m3, norm, resource,
    resources, run, sampler_words, Rng, FORMATS, TEXTURE,
};
use super::engine_linear_half::{to_half, Rounding, Stored, Texels};
use super::engine_srgb_raw::{f32_bits, half_bits, step};

/// A gather's result type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Texel {
    Float,
    Half,
    Int,
    Uint,
}

impl Texel {
    /// The MSL scalar, the AIR vector, the intrinsic's suffix, and the output's MSL type.
    fn air(self) -> (&'static str, &'static str, &'static str, &'static str) {
        match self {
            Texel::Float => ("float", "<4 x float>", "v4f32", "float4"),
            Texel::Half => ("half", "<4 x half>", "v4f16", "half4"),
            Texel::Int => ("int", "<4 x i32>", "s.v4i32", "int4"),
            Texel::Uint => ("uint", "<4 x i32>", "u.v4i32", "uint4"),
        }
    }

    fn half(self) -> bool {
        self == Texel::Half
    }

    /// The result's bytes per lane.
    fn bytes(self) -> usize {
        match self {
            Texel::Half => 8,
            _ => 16,
        }
    }

    fn integer(self) -> bool {
        matches!(self, Texel::Int | Texel::Uint)
    }
}

/// The words of a constexpr nearest sampler without a mip filter addressing the `s`, `t`, and `r`
/// axes by `s`, `t`, and `r` (0 zero, 1 edge, 2 repeat, 3 mirrored repeat).
fn axes(s: u64, t: u64, r: u64) -> u64 {
    sampler_words(0, 0, 0, 0) | s | t << 3 | r << 6
}

/// A nearest sampler clamping to the edge.
fn near() -> u64 {
    sampler_words(0, 0, 1, 0)
}

/// What a kernel gathers.
#[derive(Clone, Copy, Debug)]
enum Gather {
    /// Channel `component` of a color texture, with `texel` results.
    Color(Texel, u32),
    /// A depth texture.
    Depth,
    /// A depth texture's comparison, against the lane's `w` (an array's: 0.5).
    Compare,
}

/// A kernel gathering channel `component` of `texturecube` slot 0 (`texturecube_array` where
/// `array`) through constexpr sampler `words` along `float4` buffer 0's lane `tid` (its `xyz`; its
/// `w`'s bits the array index), storing the four taps (`texel`) to buffer 1.
fn kernel(words: u64, texel: Texel, array: bool, component: u32) -> String {
    kernel_of(words, Gather::Color(texel, component), array)
}

/// A kernel making gather `g` of slot 0, a cube (a cube array where `array`), through constexpr
/// sampler `words` along `float4` buffer 0's lane `tid` (its `xyz`; its `w`'s bits the array
/// index, or a non-array comparison's reference), storing the four taps to buffer 1.
fn kernel_of(words: u64, g: Gather, array: bool) -> String {
    let (texel, depth) = match g {
        Gather::Color(texel, _) => (texel, false),
        Gather::Depth | Gather::Compare => (Texel::Float, true),
    };
    let (scalar, vec4, suffix, out_ty) = texel.air();
    let size = texel.bytes();
    let (dim, layer, layer_ty) = match array {
        true => ("cube_array", "i32 %a, ", "i32, "),
        false => ("cube", "", ""),
    };
    let tex_ty = match depth {
        true => format!("depth{dim}<float, sample>"),
        false => format!("texture{dim}<{scalar}, sample>"),
    };
    let reference = match array {
        true => "float 5.000000e-01, ",
        false => "float %wf, ",
    };
    let (name, args, tys) = match g {
        Gather::Color(_, component) => (
            format!("gather_texture_{dim}.{suffix}"),
            format!("<3 x float> %d, {layer}i32 {component}, i32 0"),
            format!("<3 x float>, {layer_ty}i32, i32"),
        ),
        Gather::Depth => (
            format!("gather_depth_{dim}.v4f32"),
            format!("i32 1, <3 x float> %d, {layer}i32 0"),
            format!("i32, <3 x float>, {layer_ty}i32"),
        ),
        Gather::Compare => (
            format!("gather_compare_depth_{dim}.f32"),
            format!("i32 1, <3 x float> %d, {layer}{reference}i32 0"),
            format!("i32, <3 x float>, {layer_ty}float, i32"),
        ),
    };
    format!(
        r#"@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 {words}, i64 0], align 8

define void @k(ptr addrspace(1) %t, ptr addrspace(1) %c, ptr addrspace(1) %out, i32 %tid) {{
  %i = zext i32 %tid to i64
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %c, i64 %i
  %c4 = load <4 x float>, ptr addrspace(1) %p, align 16
  %d = shufflevector <4 x float> %c4, <4 x float> zeroinitializer, <3 x i32> <i32 0, i32 1, i32 2>
  %wf = extractelement <4 x float> %c4, i64 3
  %a = bitcast float %wf to i32
  %r = call {{ {vec4}, i8 }} @air.{name}(ptr addrspace(1) %t, ptr addrspace(2) @__air_sampler_state, {args})
  %v = extractvalue {{ {vec4}, i8 }} %r, 0
  %q = getelementptr inbounds {vec4}, ptr addrspace(1) %out, i64 %i
  store {vec4} %v, ptr addrspace(1) %q, align {size}
  ret void
}}

declare {{ {vec4}, i8 }} @air.{name}(ptr addrspace(1), ptr addrspace(2), {tys})

!air.kernel = !{{!0}}
!air.sampler_states = !{{!7}}
!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4, !5, !6}}
!3 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"{tex_ty}", !"air.arg_name", !"t"}}
!4 = !{{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"c"}}
!5 = !{{i32 2, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 {size}, !"air.arg_type_align_size", i32 {size}, !"air.arg_type_name", !"{out_ty}", !"air.arg_name", !"out"}}
!6 = !{{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}}
!7 = !{{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}}
"#
    )
}

// ---- The reference ----------------------------------------------------------------------------

/// Gather order as indices into a footprint's taps (`footprint`'s `(i, j)`, `(i + 1, j)`, `(i, j +
/// 1)`, `(i + 1, j + 1)`): `x` `(i, j + 1)`, `y` `(i + 1, j + 1)`, `z` `(i + 1, j)`, `w` `(i, j)`.
const ORDER: [usize; 4] = [2, 3, 1, 0];

/// The stored channel `c` Metal gathers at a tap reading texels `at` (`footprint`'s) of cube
/// texels `stored(face, i, j)` of format class `class`: the texel, or past both edges a normalized
/// format's made-up code `⌊(6 X + 5 Y + 5 Z + 8) / 16⌋` (codes read from `-scale` up) and any
/// other format's `x` face's texel.
fn gathered(
    class: u32,
    at: &[(usize, i64, i64)],
    stored: &dyn Fn(usize, i64, i64) -> [Stored; 4],
    c: usize,
) -> Stored {
    let v = |k: usize| stored(at[k].0, at[k].1, at[k].2)[c];
    match (at.len(), norm(class)) {
        (3, Some((scale, _))) => {
            let code = |k: usize| match v(k) {
                Stored::Code(x) => x.max(-scale),
                Stored::Value(_) => unreachable!("a float texel of a normalized format"),
            };
            Stored::Code((6 * code(0) + 5 * code(1) + 5 * code(2) + 8).div_euclid(16))
        }
        _ => v(0),
    }
}

/// An 8-bit sRGB format's gathered bits of channel `c` at a tap reading texels `at` of codes
/// `codes(face, i, j)`: a color channel's table step `k` as `k / 4095`, alpha's code as
/// `code / 255`, past both edges the made-up step or code, each correctly rounded (`half`).
fn srgb_gathered(
    at: &[(usize, i64, i64)],
    codes: &dyn Fn(usize, i64, i64) -> [u32; 4],
    (c, half): (usize, bool),
) -> u32 {
    let color = c < 3;
    let unit = |code: u32| match color {
        true => step(code),
        false => u64::from(code),
    };
    let v = |k: usize| unit(codes(at[k].0, at[k].1, at[k].2)[c]);
    let n = match at.len() {
        3 => (6 * v(0) + 5 * v(1) + 5 * v(2) + 8) >> 4,
        _ => v(0),
    };
    let grid = if color { 4095 } else { 255 };
    match half {
        true => u32::from(half_bits(n, grid)),
        false => f32_bits(n, grid),
    }
}

/// The bits of a gathered `f32` texel `v` as a `float` result, or rounded to nearest even as a
/// `half` one.
fn result_bits(v: f32, half: bool) -> u32 {
    match half {
        true => u32::from(to_half(f64::from(v), Rounding::NearestEven)),
        false => v.to_bits(),
    }
}

/// Every lane of random cubes of every format family (faces of 1 texel and of 2, 3, or 5; three
/// cubes of an array) gathers Metal's taps, every channel, with `texel` results (`float` or
/// `half`), through a nearest or a linear sampler alike; some lanes of each read a corner tap.
fn check_formats(texel: Texel) {
    let p = lavapipe();
    for array in [false, true] {
        for component in 0..4u32 {
            // A gather reads no filter: nearest and linear samplers alike.
            let words = sampler_words(u64::from(component % 2), 0, 1, 0);
            let name = format!("{texel:?} array={array} component={component}");
            let Some(bc) = assemble(&name, &kernel(words, texel, array, component)) else {
                return;
            };
            let (plan, m) = typed_module_with(&bc, "k", &p, &inputs(false))
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            let cubes = if array { 3 } else { 1 };
            for (seed, pf) in FORMATS.iter().enumerate() {
                let class = texel_format(*pf).0;
                let other = [2u32, 3, 5][(seed + component as usize) % 3];
                let sizes = if array { vec![other] } else { vec![1, other] };
                for n in sizes {
                    let mut rng = Rng(0x0c6a_7e23_0000
                        + u64::from(component) * 4096
                        + seed as u64 * 64
                        + u64::from(n));
                    let layers = 6 * cubes;
                    let tex =
                        Texels::random(class, i64::from(n), i64::from(n * layers), rng.next());
                    let stored = |l: u32, i: i64, j: i64| -> [Stored; 4] {
                        tex.t[((i64::from(l * n) + j) * i64::from(n) + i) as usize]
                    };
                    let imgs = images(&m, (n, layers), false, &|l, i, j| {
                        stored(l, i64::from(i), i64::from(j)).map(|s| tex.fetched(s).to_bits())
                    });
                    let lanes = directions(n, 64, cubes, &mut rng);
                    let res = resources(&plan, &lanes, texel.bytes(), class);
                    let out = run(&m, &res, &imgs, texel.half());
                    let mut corners = 0;
                    for (lane, (d, got)) in lanes.iter().zip(&out).enumerate() {
                        let dir = [d[0], d[1], d[2]].map(f32::from_bits);
                        let fp = footprint(dir, n, (false, false));
                        let base = 6 * if array { d[3].min(cubes - 1) } else { 0 };
                        let at = |g: usize, i: i64, j: i64| stored(base + g as u32, i, j);
                        corners += usize::from(fp.iter().any(|(_, t)| t.len() == 3));
                        let want: [u32; 4] = std::array::from_fn(|k| {
                            let s = gathered(class, &fp[ORDER[k]].1, &at, component as usize);
                            result_bits(tex.fetched(s), texel.half())
                        });
                        assert_eq!(
                            *got, want,
                            "{name} {pf:?} n={n} lane {lane} {dir:?} [{}]: {got:08x?} vs \
                             {want:08x?}",
                            d[3]
                        );
                    }
                    assert!(corners > 0, "{name} {pf:?} n={n}: no lane reads a corner");
                }
            }
        }
    }
}

/// `float` gathers of every format family are Metal's taps, bit for bit.
#[test]
fn float_cube_gathers_take_metals_taps() {
    check_formats(Texel::Float);
}

/// `half` gathers of every format family are Metal's taps rounded to nearest even, bit for bit.
#[test]
fn half_cube_gathers_take_metals_taps() {
    check_formats(Texel::Half);
}

/// A channel of a `CubeCorner` format as the device fetches it: a step `k / D` of its stored grid
/// `D` (`k` from `-D` where signed), a float value, or a missing channel's.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Chan {
    Step(i64, bool),
    Float,
    Missing,
}

/// The fetched channels of `pf`, a format of a `CubeCorner`: its stored grids, which divide the
/// corner's grids.
fn corner_channels(pf: PixelFormat) -> [Chan; 4] {
    use Chan::{Float, Missing, Step};
    use PixelFormat as P;
    let u = |bits: u32| Step((1 << bits) - 1, false);
    match pf {
        P::RG11B10Float | P::RGB9E5Float | P::BC6HRGBFloat | P::BC6HRGBUfloat => {
            [Float, Float, Float, Missing]
        }
        P::ASTC4x4HDR
        | P::ASTC5x4HDR
        | P::ASTC5x5HDR
        | P::ASTC6x5HDR
        | P::ASTC6x6HDR
        | P::ASTC8x5HDR
        | P::ASTC8x6HDR
        | P::ASTC8x8HDR
        | P::ASTC10x5HDR
        | P::ASTC10x6HDR
        | P::ASTC10x8HDR
        | P::ASTC10x10HDR
        | P::ASTC12x10HDR
        | P::ASTC12x12HDR => [Float; 4],
        P::ABGR4Unorm => [u(4); 4],
        P::BC1RGBA | P::BC2RGBA | P::BC3RGBA | P::BC7RGBAUnorm | P::ETC2RGB8A1 | P::EACRGBA8 => {
            [u(8); 4]
        }
        P::ETC2RGB8 => [u(8), u(8), u(8), Missing],
        P::RGB10A2Unorm | P::BGR10A2Unorm => [u(10), u(10), u(10), u(2)],
        P::A1BGR5Unorm | P::BGR5A1Unorm => [u(5), u(5), u(5), u(1)],
        P::B5G6R5Unorm => [u(5), u(6), u(5), Missing],
        P::EACR11Unorm => [u(11), Missing, Missing, Missing],
        P::EACRG11Unorm => [u(11), u(11), Missing, Missing],
        P::EACR11Snorm => [Step(1023, true), Missing, Missing, Missing],
        P::EACRG11Snorm => [Step(1023, true), Step(1023, true), Missing, Missing],
        P::BC4RUnorm => [Step(16320, false), Missing, Missing, Missing],
        P::BC5RGUnorm => [Step(16320, false), Step(16320, false), Missing, Missing],
        P::BC4RSnorm => [Step(8128, true), Missing, Missing, Missing],
        P::BC5RGSnorm => [Step(8128, true), Step(8128, true), Missing, Missing],
        _ => panic!("{pf:?} is not a corner format"),
    }
}

/// A cube gather of a format of a `CubeCorner` (its `TexelFormat::CORNER` field alone) is emulated,
/// its taps the fetched texels in gather order and the tap past both edges the `x` face's texel
/// (the float formats) or the texel made up of the three faces' codes on each channel's grid, `⌊(6
/// X + 5 Y + 5 Z + 8) / 16⌋` correctly rounded, bit for bit with `float` and `half` results, every
/// channel, on random texels of each format's stored grids (fetched as their steps `k / D`
/// correctly rounded) on faces of 2, 3, and 5 texels: one color format of each class and channel
/// layout, which the others share.
#[test]
fn a_cube_corner_class_gathers_its_corner() {
    let p = lavapipe();
    let mut formats: Vec<PixelFormat> = Vec::new();
    for &pf in PixelFormat::ALL {
        if CubeCorner::of(pf).is_none() || pf == PixelFormat::Depth32FloatStencil8 {
            continue;
        }
        // The corner's class: the field alone (a packed format's family and field, are no
        // corner's).
        let corner = |q: PixelFormat| texel_format(q).0 & TexelFormat::CORNER_MASK;
        let shared = |q: &PixelFormat| {
            corner(*q) == corner(pf) && corner_channels(*q) == corner_channels(pf)
        };
        if !formats.iter().any(shared) {
            formats.push(pf);
        }
    }
    assert_eq!(formats.len(), 16, "{formats:?}");
    for texel in [Texel::Float, Texel::Half] {
        for component in 0..4u32 {
            let words = sampler_words(u64::from(component % 2), 0, 1, 0);
            let name = format!("{texel:?} component={component}");
            let Some(bc) = assemble(&name, &kernel(words, texel, false, component)) else {
                return;
            };
            let (plan, m) = typed_module_with(&bc, "k", &p, &inputs(false))
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            for (seed, pf) in formats.iter().enumerate() {
                let Some(corner) = CubeCorner::of(*pf) else {
                    continue;
                };
                let class = texel_format(*pf).0;
                let chans = corner_channels(*pf);
                let c = component as usize;
                for n in [2u32, 3, 5] {
                    let mut rng = Rng(0x0c0e_4150_0000
                        + u64::from(component) * 4096
                        + seed as u64 * 64
                        + u64::from(n));
                    // Each texel's fetched channels, and its codes on their stored grids.
                    let texels: Vec<[(f32, i64); 4]> = (0..6 * n * n)
                        .map(|_| {
                            std::array::from_fn(|k| match chans[k] {
                                Chan::Step(d, signed) => {
                                    let lo = if signed { -d } else { 0 };
                                    let code = lo + rng.below((d - lo + 1) as u64) as i64;
                                    ((code as f64 / d as f64) as f32, code)
                                }
                                Chan::Float => {
                                    let m = rng.below(4096) as f64;
                                    ((m * 2f64.powi(-10)) as f32, 0)
                                }
                                Chan::Missing => (if k == 3 { 1.0 } else { 0.0 }, 0),
                            })
                        })
                        .collect();
                    let at = |l: usize, i: i64, j: i64| {
                        texels[((l as i64 * i64::from(n) + j) * i64::from(n) + i) as usize]
                    };
                    let imgs = images(&m, (n, 6), false, &|l, i, j| {
                        at(l as usize, i64::from(i), i64::from(j)).map(|(v, _)| v.to_bits())
                    });
                    let lanes = directions(n, 64, 1, &mut rng);
                    let out = run(
                        &m,
                        &resources(&plan, &lanes, texel.bytes(), class),
                        &imgs,
                        texel.half(),
                    );
                    let mut corners = 0;
                    for (lane, (d, got)) in lanes.iter().zip(&out).enumerate() {
                        let dir = [d[0], d[1], d[2]].map(f32::from_bits);
                        let fp = footprint(dir, n, (false, false));
                        corners += usize::from(fp.iter().any(|(_, t)| t.len() == 3));
                        let want: [u32; 4] = std::array::from_fn(|k| {
                            let tap = &fp[ORDER[k]].1;
                            let read = |q: usize| at(tap[q].0, tap[q].1, tap[q].2)[c];
                            let v = match (tap.len(), corner.grids(), chans[c]) {
                                (3, Some(grids), Chan::Step(stored, _)) => {
                                    // The codes on the corner's grid, which the stored grid
                                    // divides.
                                    let grid = i64::from(grids[c]);
                                    let code = |q: usize| read(q).1 * (grid / stored);
                                    let made = (6 * code(0) + 5 * code(1) + 5 * code(2) + 8)
                                        .div_euclid(16);
                                    (made as f64 / grid as f64) as f32
                                }
                                _ => read(0).0,
                            };
                            result_bits(v, texel.half())
                        });
                        assert_eq!(
                            *got, want,
                            "{name} {pf:?} n={n} lane {lane} {dir:?}: {got:08x?} vs {want:08x?}"
                        );
                    }
                    assert!(corners > 0, "{name} {pf:?} n={n}: no lane reads a corner");
                }
            }
        }
    }
}

/// Integer gathers (`int` and `uint`, a cube and a cube array, channels `x` and `w`) are the
/// fetched texels in gather order, the tap past both edges the `x` face's texel, on random words.
#[test]
fn integer_cube_gathers_take_metals_taps() {
    let p = lavapipe();
    for texel in [Texel::Int, Texel::Uint] {
        for array in [false, true] {
            for component in [0u32, 3] {
                let name = format!("{texel:?} array={array} component={component}");
                let Some(bc) = assemble(&name, &kernel(near(), texel, array, component)) else {
                    return;
                };
                let (plan, m) = typed_module_with(&bc, "k", &p, &inputs(false))
                    .unwrap_or_else(|e| panic!("{name}: {e}"));
                let cubes = if array { 2 } else { 1 };
                for n in [1u32, 2, 3, 5] {
                    let mut rng = Rng(0x01a7_e623_0000 + u64::from(component) * 64 + u64::from(n));
                    let layers = 6 * cubes;
                    let words: Vec<[u32; 4]> = (0..layers * n * n)
                        .map(|_| std::array::from_fn(|_| rng.next() as u32))
                        .collect();
                    let word = |l: u32, i: i64, j: i64| -> [u32; 4] {
                        words[((i64::from(l * n) + j) * i64::from(n) + i) as usize]
                    };
                    let imgs = images(&m, (n, layers), false, &|l, i, j| {
                        word(l, i64::from(i), i64::from(j))
                    });
                    let lanes = directions(n, 64, cubes, &mut rng);
                    let out = run(&m, &resources(&plan, &lanes, 16, 0), &imgs, false);
                    let mut corners = 0;
                    for (lane, (d, got)) in lanes.iter().zip(&out).enumerate() {
                        let dir = [d[0], d[1], d[2]].map(f32::from_bits);
                        let fp = footprint(dir, n, (false, false));
                        let base = 6 * if array { d[3].min(cubes - 1) } else { 0 };
                        corners += usize::from(fp.iter().any(|(_, t)| t.len() == 3));
                        let want: [u32; 4] = std::array::from_fn(|k| {
                            let (g, i, j) = fp[ORDER[k]].1[0];
                            word(base + g as u32, i, j)[component as usize]
                        });
                        assert_eq!(
                            *got, want,
                            "{name} n={n} lane {lane} {dir:?}: {got:08x?} vs {want:08x?}"
                        );
                    }
                    assert!(corners > 0, "{name} n={n}: no lane reads a corner");
                }
            }
        }
    }
}

/// An 8-bit sRGB format's gather (through the slot's raw view, under its variant) is each tap's
/// table step, the corner's made-up step, bit for bit with `half` and `float` results, every
/// channel, on random codes.
#[test]
fn a_raw_cube_gather_takes_metals_steps() {
    let p = lavapipe();
    let class = texel_format(PixelFormat::RGBA8UnormSRGB).0;
    for texel in [Texel::Half, Texel::Float] {
        for component in 0..4u32 {
            let name = format!("srgb {texel:?} component={component}");
            let Some(bc) = assemble(&name, &kernel(near(), texel, false, component)) else {
                return;
            };
            let (plan, m) = typed_module_with(&bc, "k", &p, &inputs(true))
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            assert!(
                !plan.precision.contains(&PrecisionNote::SrgbTexelDecode),
                "{name}"
            );
            for n in [1u32, 2, 3] {
                let mut rng = Rng(0x05ee_d023_0000 + u64::from(component) * 64 + u64::from(n));
                let codes: Vec<[u32; 4]> = (0..6 * n * n)
                    .map(|_| std::array::from_fn(|_| rng.below(256) as u32))
                    .collect();
                let code = |l: usize, i: i64, j: i64| {
                    codes[((l as i64 * i64::from(n) + j) * i64::from(n) + i) as usize]
                };
                let imgs = images(&m, (n, 6), true, &|l, i, j| {
                    code(l as usize, i64::from(i), i64::from(j))
                        .map(|c| f32_bits(u64::from(c), 255))
                });
                let lanes = directions(n, 96, 0, &mut rng);
                let res = resources(&plan, &lanes, texel.bytes(), class);
                let out = run(&m, &res, &imgs, texel.half());
                for (lane, (d, got)) in lanes.iter().zip(&out).enumerate() {
                    let dir = [d[0], d[1], d[2]].map(f32::from_bits);
                    let fp = footprint(dir, n, (false, false));
                    let want: [u32; 4] = std::array::from_fn(|k| {
                        srgb_gathered(&fp[ORDER[k]].1, &code, (component as usize, texel.half()))
                    });
                    assert_eq!(
                        *got, want,
                        "{name} n={n} lane {lane} {dir:?}: {got:08x?} vs {want:08x?}"
                    );
                }
            }
        }
    }
}

/// A gather through a sampler repeating on a face axis wraps a face coordinate of exactly 1 to 0 on
/// faces of 3 and 5 texels and stays on the last texel on faces of 4, the sampler's `s` address
/// governing the face's `s` axis and its `t` address the `t` axis, its `r` address neither: every
/// lane of [`edge_directions`] is the rule, bit for bit (each axis alone on faces of 3 texels), and
/// some lanes of each wrapping case on a size that is not a power of two gather other taps than a
/// clamping sampler's.
#[test]
fn a_repeating_face_axis_wraps_a_gather() {
    let p = lavapipe();
    let pf = PixelFormat::RGBA32Float;
    let class = texel_format(pf).0;
    let lanes = edge_directions();
    // (name, sampler, the face's `s` and `t` axes wrap)
    let cases = [
        ("repeat", axes(2, 2, 2), (true, true)),
        ("s repeat", axes(2, 1, 1), (true, false)),
        ("t repeat", axes(1, 2, 1), (false, true)),
        ("r repeat", axes(1, 1, 2), (false, false)),
    ];
    for (name, words, wrap) in cases {
        let Some(bc) = assemble(name, &kernel(words, Texel::Float, false, 3)) else {
            return;
        };
        let (plan, m) = typed_module_with(&bc, "k", &p, &inputs(false))
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let sizes: &[u32] = match wrap {
            (true, true) => &[3, 5, 4],
            _ => &[3],
        };
        for &n in sizes {
            let mut rng = Rng(0x0e0d_9e23_0000 + u64::from(n));
            let tex = Texels::random(class, i64::from(n), i64::from(6 * n), rng.next());
            let stored = |l: usize, i: i64, j: i64| -> [Stored; 4] {
                tex.t[((l as i64 * i64::from(n) + j) * i64::from(n) + i) as usize]
            };
            let imgs = images(&m, (n, 6), false, &|l, i, j| {
                stored(l as usize, i64::from(i), i64::from(j)).map(|s| tex.fetched(s).to_bits())
            });
            let out = run(&m, &resources(&plan, &lanes, 16, class), &imgs, false);
            let mut moved = 0;
            for (lane, (d, got)) in lanes.iter().zip(&out).enumerate() {
                let dir = [d[0], d[1], d[2]].map(f32::from_bits);
                let fp = footprint(dir, n, wrap);
                moved += usize::from(fp != footprint(dir, n, (false, false)));
                let want: [u32; 4] = std::array::from_fn(|k| {
                    tex.fetched(gathered(class, &fp[ORDER[k]].1, &stored, 3))
                        .to_bits()
                });
                assert_eq!(
                    *got, want,
                    "{name} n={n} lane {lane} {dir:?}: {got:08x?} vs {want:08x?}"
                );
            }
            match n.is_power_of_two() || wrap == (false, false) {
                true => assert_eq!(moved, 0, "{name} n={n}"),
                false => assert!(moved > 0, "{name} n={n}: no lane wraps"),
            }
        }
    }
}

// ---- Runtime samplers ---------------------------------------------------------------------------

/// [`kernel_of`]'s kernel of a cube (not an array) through runtime sampler slot 0, argument 3,
/// in place of a constexpr sampler.
fn runtime_kernel_of(g: Gather) -> String {
    let constexpr = kernel_of(0, g, false);
    let first = constexpr
        .find("define void")
        .expect("the kernel's definition");
    let sampler = "!7 = !{i32 3, !\"air.sampler\", !\"air.location_index\", i32 0, i32 1, \
                   !\"air.arg_type_name\", !\"sampler\", !\"air.arg_name\", !\"s\"}\n";
    // The sampler-state metadata first: it names the constexpr sampler's global too.
    constexpr[first..]
        .replace(
            "!7 = !{!\"air.sampler_state\", ptr addrspace(2) @__air_sampler_state}\n",
            sampler,
        )
        .replace(
            "ptr addrspace(1) %out, i32 %tid)",
            "ptr addrspace(1) %out, ptr addrspace(2) %s, i32 %tid)",
        )
        .replace(
            "ptr addrspace(2) @__air_sampler_state",
            "ptr addrspace(2) %s",
        )
        .replace("!air.sampler_states = !{!7}\n", "")
        .replace("!2 = !{!3, !4, !5, !6}", "!2 = !{!3, !4, !5, !7, !6}")
        .replace(
            "!6 = !{i32 3, !\"air.thread_position_in_grid\"",
            "!6 = !{i32 4, !\"air.thread_position_in_grid\"",
        )
}

/// The `SamplerState` specialization of compiled `a`: its key and the fields of the word the
/// function reads.
fn sampler_state(a: &FunctionArtifact, name: &str) -> (SpecKey, u32) {
    let state = a.interface.read_keys().into_iter().find_map(|k| match k {
        SpecKey::SamplerState { fields, .. } => Some((k, fields)),
        _ => None,
    });
    state.unwrap_or_else(|| panic!("{name}: no `SamplerState` specialization"))
}

/// A cube gather through a runtime sampler slot takes Metal's footprint by fetches
/// (`sampler.gather.cube`), its face axes wrapped by the bound sampler's word: a face axis whose
/// word's mode is `repeat` wraps a face coordinate of 1 on faces of 3 and 5 texels and not of 4,
/// each axis by its own mode, and `clamp_to_edge` and `mirrored_repeat` read as clamping, every
/// lane of [`edge_directions`] bit for bit on the interpreter.
#[test]
fn a_runtime_sampler_wraps_a_cube_gather_by_its_word() {
    use crate::metal::sampler::SamplerWord;
    use crate::metal::SamplerAddressMode as M;
    let p = lavapipe();
    let name = "runtime cube gather";
    let Some(bc) = assemble(name, &runtime_kernel_of(Gather::Color(Texel::Float, 3))) else {
        return;
    };
    let a = compile(&bc, "k", &p).unwrap_or_else(|e| panic!("{name}: {e}"));
    link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{name}: {e}"));
    assert!(emulated(&a, "sampler.gather"), "{name}: not emulated");
    let (key, fields) = sampler_state(&a, name);
    let address = SamplerWord::ADDRESS[0] | SamplerWord::ADDRESS[1];
    assert_eq!(
        fields & address,
        address,
        "{name}: the word's fields {fields:#x}"
    );
    let (plan, m) =
        typed_module_with(&bc, "k", &p, &inputs(false)).unwrap_or_else(|e| panic!("{name}: {e}"));
    let pf = PixelFormat::RGBA32Float;
    let class = texel_format(pf).0;
    let lanes = edge_directions();
    // (the `s` and `t` modes, faces' sizes)
    let cases: [((M, M), &[u32]); 5] = [
        ((M::Repeat, M::Repeat), &[3, 5, 4]),
        ((M::Repeat, M::ClampToEdge), &[3]),
        ((M::ClampToEdge, M::Repeat), &[3]),
        ((M::ClampToEdge, M::ClampToEdge), &[3]),
        ((M::MirrorRepeat, M::MirrorRepeat), &[3]),
    ];
    for ((ms, mt), sizes) in cases {
        let word = ms.as_raw() << SamplerWord::ADDRESS_SHIFT[0]
            | mt.as_raw() << SamplerWord::ADDRESS_SHIFT[1]
            | M::Repeat.as_raw() << SamplerWord::ADDRESS_SHIFT[2];
        let wrap = (ms == M::Repeat, mt == M::Repeat);
        for &n in sizes {
            let mut rng = Rng(0x0e0d_9e24_0000 + u64::from(n));
            let tex = Texels::random(class, i64::from(n), i64::from(6 * n), rng.next());
            let stored = |l: usize, i: i64, j: i64| -> [Stored; 4] {
                tex.t[((l as i64 * i64::from(n) + j) * i64::from(n) + i) as usize]
            };
            let imgs = images(&m, (n, 6), false, &|l, i, j| {
                stored(l as usize, i64::from(i), i64::from(j)).map(|s| tex.fetched(s).to_bits())
            });
            let mut res = resources(&plan, &lanes, 16, class);
            res.spec.insert(key, u64::from(word & fields));
            let out = run(&m, &res, &imgs, false);
            let mut moved = 0;
            for (lane, (d, got)) in lanes.iter().zip(&out).enumerate() {
                let dir = [d[0], d[1], d[2]].map(f32::from_bits);
                let fp = footprint(dir, n, wrap);
                moved += usize::from(fp != footprint(dir, n, (false, false)));
                let want: [u32; 4] = std::array::from_fn(|k| {
                    tex.fetched(gathered(class, &fp[ORDER[k]].1, &stored, 3))
                        .to_bits()
                });
                assert_eq!(
                    *got, want,
                    "{name} {ms:?} {mt:?} n={n} lane {lane} {dir:?}: {got:08x?} vs {want:08x?}"
                );
            }
            match n.is_power_of_two() || wrap == (false, false) {
                true => assert_eq!(moved, 0, "{name} {ms:?} {mt:?} n={n}"),
                false => assert!(moved > 0, "{name} {ms:?} {mt:?} n={n}: no lane wraps"),
            }
        }
    }
}

/// A depth cube's comparison gather through a runtime sampler slot compares each of Metal's taps
/// (`sampler.gather_compare.cube`) under the compare function the bound sampler's word states,
/// every `MTLCompareFunction`, on `Depth32Float` and `Depth16Unorm` faces of 1 and 3 texels with
/// random references, bit for bit on the interpreter.
#[test]
fn a_runtime_sampler_compares_a_cube_gather_by_its_word() {
    use crate::metal::sampler::SamplerWord;
    use crate::metal::CompareFunction as F;
    let p = lavapipe();
    let name = "runtime cube comparison gather";
    let Some(bc) = assemble(name, &runtime_kernel_of(Gather::Compare)) else {
        return;
    };
    let a = compile(&bc, "k", &p).unwrap_or_else(|e| panic!("{name}: {e}"));
    link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{name}: {e}"));
    assert!(
        emulated(&a, "sampler.gather_compare"),
        "{name}: not emulated"
    );
    let (key, fields) = sampler_state(&a, name);
    assert_eq!(
        fields & SamplerWord::COMPARE,
        SamplerWord::COMPARE,
        "{name}: the word's fields {fields:#x}"
    );
    let (plan, m) =
        typed_module_with(&bc, "k", &p, &inputs(false)).unwrap_or_else(|e| panic!("{name}: {e}"));
    // Each function and [`passes`]' code for it.
    let functions = [
        (F::Never, 8),
        (F::Less, 1),
        (F::Equal, 5),
        (F::LessEqual, 2),
        (F::Greater, 3),
        (F::NotEqual, 6),
        (F::GreaterEqual, 4),
        (F::Always, 7),
    ];
    for (f, op) in functions {
        let word = f.as_raw() << SamplerWord::COMPARE_SHIFT;
        for (seed, pf) in DEPTHS.iter().enumerate() {
            let class = texel_format(*pf).0;
            for n in [1u32, 3] {
                let mut rng = Rng(0x0c09_9a24_0000 + op * 4096 + seed as u64 * 64 + u64::from(n));
                let tex = Texels::random(
                    depth_values(*pf),
                    i64::from(n),
                    i64::from(6 * n),
                    rng.next(),
                );
                let stored = |l: u32, i: i64, j: i64| -> [Stored; 4] {
                    tex.t[((i64::from(l * n) + j) * i64::from(n) + i) as usize]
                };
                let imgs = images(&m, (n, 6), false, &|l, i, j| {
                    stored(l, i64::from(i), i64::from(j)).map(|s| tex.fetched(s).to_bits())
                });
                let mut lanes = directions(n, 64, 1, &mut rng);
                for lane in lanes.iter_mut() {
                    let r = match rng.below(2) {
                        0 => tex.fetched(tex.t[rng.below(tex.t.len() as u64) as usize][0]),
                        _ => -0.5 + 2.0 * (rng.below(1 << 20) as f32 / (1 << 20) as f32),
                    };
                    lane[3] = r.to_bits();
                }
                let mut res = resources(&plan, &lanes, 16, class);
                res.spec.insert(key, u64::from(word & fields));
                let out = run(&m, &res, &imgs, false);
                for (lane, (d, got)) in lanes.iter().zip(&out).enumerate() {
                    let dir = [d[0], d[1], d[2]].map(f32::from_bits);
                    let fp = footprint(dir, n, (false, false));
                    let r = f32::from_bits(d[3]);
                    let at = |g: usize, i: i64, j: i64| stored(g as u32, i, j);
                    let want: [u32; 4] = std::array::from_fn(|k| {
                        let s = gathered(class, &fp[ORDER[k]].1, &at, 0);
                        f32::from(u8::from(passes(op, r, s))).to_bits()
                    });
                    assert_eq!(
                        *got, want,
                        "{name} {f:?} {pf:?} n={n} lane {lane} {dir:?} reference {r:?}: \
                         {got:08x?} vs {want:08x?}"
                    );
                }
            }
        }
    }
}

// ---- Depth ------------------------------------------------------------------------------------

/// The depth formats whose cube gather is measured (`metal::texel_format`):
/// `Depth32Float_Stencil8`'s by its `CubeCorner`.
const DEPTHS: [PixelFormat; 3] = [
    PixelFormat::Depth32Float,
    PixelFormat::Depth16Unorm,
    PixelFormat::Depth32FloatStencil8,
];

/// The class whose random texels a depth format's tests store: its own, but
/// `Depth32Float_Stencil8`'s depth, which holds `Depth32Float`'s values.
fn depth_values(pf: PixelFormat) -> u32 {
    match pf {
        PixelFormat::Depth32FloatStencil8 => texel_format(PixelFormat::Depth32Float).0,
        _ => texel_format(pf).0,
    }
}

/// A nearest comparison sampler clamping to the edge with compare function `op` (the sampler
/// word's code: 1 `less`, 2 `less_equal`, 3 `greater`, 4 `greater_equal`, 5 `equal`, 6
/// `not_equal`, 7 `always`, 8 `never`).
fn compare_sampler(op: u64) -> u64 {
    near() & !(0xf << 16) | op << 16
}

/// Whether Metal's comparison `op` ([`compare_sampler`]'s code) of reference `r` passes against a
/// tap's stored texel `s`: a float format's texel against the reference as given, in `f32`; a
/// `Depth16Unorm` code against the reference's, `r` clamped into `[0, 1]` and its exact product
/// with 65535 rounded to nearest (0.5 up), a NaN failing every ordered comparison.
fn passes(op: u64, r: f32, s: Stored) -> bool {
    let (r, t) = match s {
        Stored::Code(c) => {
            let k = match r.is_nan() {
                true => f64::NAN,
                false => (f64::from(r.clamp(0.0, 1.0)) * 65535.0).round_ties_even(),
            };
            (k, c as f64)
        }
        Stored::Value(v) => (f64::from(r), f64::from(v)),
    };
    match op {
        1 => r < t,
        2 => r <= t,
        3 => r > t,
        4 => r >= t,
        5 => r == t,
        6 => r != t,
        7 => true,
        _ => false,
    }
}

/// A depth cube's gather (`Depth32Float` and `Depth16Unorm`, a cube and a cube array, faces of 1,
/// 2, 3, and 5 texels) is Metal's taps in gather order, the tap past both edges `Depth32Float`'s
/// `x` face texel and `Depth16Unorm`'s made-up texel, bit for bit on random texels.
#[test]
fn depth_cube_gathers_take_metals_taps() {
    let p = lavapipe();
    for array in [false, true] {
        let name = format!("depth array={array}");
        let Some(bc) = assemble(&name, &kernel_of(near(), Gather::Depth, array)) else {
            return;
        };
        let (plan, m) = typed_module_with(&bc, "k", &p, &inputs(false))
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let cubes = if array { 2 } else { 1 };
        for (seed, pf) in DEPTHS.iter().enumerate() {
            let class = texel_format(*pf).0;
            for n in [1u32, 2, 3, 5] {
                let mut rng = Rng(0x0de9_7423_0000 + seed as u64 * 64 + u64::from(n));
                let layers = 6 * cubes;
                let tex = Texels::random(
                    depth_values(*pf),
                    i64::from(n),
                    i64::from(n * layers),
                    rng.next(),
                );
                let stored = |l: u32, i: i64, j: i64| -> [Stored; 4] {
                    tex.t[((i64::from(l * n) + j) * i64::from(n) + i) as usize]
                };
                let imgs = images(&m, (n, layers), false, &|l, i, j| {
                    stored(l, i64::from(i), i64::from(j)).map(|s| tex.fetched(s).to_bits())
                });
                let lanes = directions(n, 64, cubes, &mut rng);
                let out = run(&m, &resources(&plan, &lanes, 16, class), &imgs, false);
                let mut corners = 0;
                for (lane, (d, got)) in lanes.iter().zip(&out).enumerate() {
                    let dir = [d[0], d[1], d[2]].map(f32::from_bits);
                    let fp = footprint(dir, n, (false, false));
                    let base = 6 * if array { d[3].min(cubes - 1) } else { 0 };
                    let at = |g: usize, i: i64, j: i64| stored(base + g as u32, i, j);
                    corners += usize::from(fp.iter().any(|(_, t)| t.len() == 3));
                    let want: [u32; 4] = std::array::from_fn(|k| {
                        tex.fetched(gathered(class, &fp[ORDER[k]].1, &at, 0))
                            .to_bits()
                    });
                    assert_eq!(
                        *got, want,
                        "{name} {pf:?} n={n} lane {lane} {dir:?}: {got:08x?} vs {want:08x?}"
                    );
                }
                assert!(corners > 0, "{name} {pf:?} n={n}: no lane reads a corner");
            }
        }
    }
}

/// A depth cube's comparison gather under every compare function is Metal's comparison of each tap
/// (the corner's by the format's rule) with the lane's reference, on random texels
/// (`Depth32Float`'s outside `[0, 1]` too) and references: special values (NaN, the infinities,
/// subnormals, `-0`, values beside 1), the texels' own values, references near a `Depth16Unorm`
/// code's half-way point, and random ones; and a cube array's against 0.5.
#[test]
fn depth_cube_comparison_gathers_take_metals_taps() {
    let p = lavapipe();
    let specials = [
        -1.0,
        -0.0,
        0.0,
        1.0,
        1.5,
        2.0,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(1),
        -f32::from_bits(1),
        0.5,
        -0.25,
        f32::from_bits(0x3f80_0001),
        f32::from_bits(0x3f7f_ffff),
    ];
    for array in [false, true] {
        for op in 1..=8u64 {
            let name = format!("compare {op} array={array}");
            let words = compare_sampler(op);
            let Some(bc) = assemble(&name, &kernel_of(words, Gather::Compare, array)) else {
                return;
            };
            let (plan, m) = typed_module_with(&bc, "k", &p, &inputs(false))
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            let cubes = if array { 2 } else { 1 };
            for (seed, pf) in DEPTHS.iter().enumerate() {
                let class = texel_format(*pf).0;
                for n in [1u32, 3] {
                    let mut rng =
                        Rng(0x0c09_9a23_0000 + op * 4096 + seed as u64 * 64 + u64::from(n));
                    let layers = 6 * cubes;
                    let tex = Texels::random(
                        depth_values(*pf),
                        i64::from(n),
                        i64::from(n * layers),
                        rng.next(),
                    );
                    let stored = |l: u32, i: i64, j: i64| -> [Stored; 4] {
                        tex.t[((i64::from(l * n) + j) * i64::from(n) + i) as usize]
                    };
                    let imgs = images(&m, (n, layers), false, &|l, i, j| {
                        stored(l, i64::from(i), i64::from(j)).map(|s| tex.fetched(s).to_bits())
                    });
                    let mut lanes = directions(n, 64, cubes, &mut rng);
                    if !array {
                        for (k, lane) in lanes.iter_mut().enumerate() {
                            let r = match rng.below(4) {
                                0 => specials[k % specials.len()],
                                1 => tex.fetched(tex.t[rng.below(tex.t.len() as u64) as usize][0]),
                                2 => -0.5 + 2.0 * (rng.below(1 << 20) as f32 / (1 << 20) as f32),
                                _ => {
                                    let half = (rng.below(65535) as f64 + 0.5) / 65535.0;
                                    let r = half as f32;
                                    f32::from_bits(
                                        (r.to_bits() as i64 + rng.below(5) as i64 - 2) as u32,
                                    )
                                }
                            };
                            lane[3] = r.to_bits();
                        }
                    }
                    let out = run(&m, &resources(&plan, &lanes, 16, class), &imgs, false);
                    for (lane, (d, got)) in lanes.iter().zip(&out).enumerate() {
                        let dir = [d[0], d[1], d[2]].map(f32::from_bits);
                        let fp = footprint(dir, n, (false, false));
                        let (base, r) = match array {
                            true => (6 * d[3].min(cubes - 1), 0.5),
                            false => (0, f32::from_bits(d[3])),
                        };
                        let at = |g: usize, i: i64, j: i64| stored(base + g as u32, i, j);
                        let want: [u32; 4] = std::array::from_fn(|k| {
                            let s = gathered(class, &fp[ORDER[k]].1, &at, 0);
                            let pass = passes(op, r, s);
                            f32::from(u8::from(pass)).to_bits()
                        });
                        assert_eq!(
                            *got, want,
                            "{name} {pf:?} n={n} lane {lane} {dir:?} reference {r:?}: \
                             {got:08x?} vs {want:08x?}"
                        );
                    }
                }
            }
        }
    }
}

// ---- Metal's own lanes --------------------------------------------------------------------------

/// Eight lanes of our probe's `gs` run on the identity cube of 3x3 faces, by their index in
/// `lanes-3.bin`: the direction's bits, and Metal's gathers there of channel `w` through a nearest
/// sampler clamping to the edge (`s_ne`) and one repeating (`s_nr`), and of channel `x` through
/// the first. Lanes 0 to 5 read across an edge and past both edges; lanes 5130, 5132, and 5135
/// wrap a face coordinate of 1 (a tie of `y` and `z`, a tie of all three, and a minor component
/// equal to the major).
const PROBE_LANES: [(u32, [u32; 3], [[f32; 4]; 3]); 8] = [
    (
        0,
        [0x3f49_0000, 0x3f80_0000, 0xbf74_7e6e],
        [
            [21.0, 3.0, 3.0, 46.0],
            [21.0, 3.0, 3.0, 46.0],
            [3.0, 1.0, 1.0, 6.0],
        ],
    ),
    (
        1,
        [0x3f4b_aaab, 0xbf31_0000, 0xbf80_0000],
        [
            [9.0, 36.0, 52.0, 9.0],
            [9.0, 36.0, 52.0, 9.0],
            [1.0, 4.0, 6.0, 1.0],
        ],
    ),
    (
        2,
        [0x3f80_0000, 0x3f73_0000, 0x3f0b_c625],
        [
            [1.0, 2.0, 24.0, 27.0],
            [1.0, 2.0, 24.0, 27.0],
            [1.0, 1.0, 3.0, 3.0],
        ],
    ),
    (
        3,
        [0xbf48_5555, 0x3e86_fd85, 0x3f80_0000],
        [
            [15.0, 40.0, 37.0, 12.0],
            [15.0, 40.0, 37.0, 12.0],
            [2.0, 5.0, 5.0, 2.0],
        ],
    ),
    (
        5,
        [0x3f67_0000, 0xbf80_0000, 0x3f4b_aaab],
        [
            [30.0, 7.0, 7.0, 45.0],
            [30.0, 7.0, 7.0, 45.0],
            [4.0, 1.0, 1.0, 5.0],
        ],
    ),
    (
        5130,
        [0x3e80_0000, 0x3f80_0000, 0x3f80_0000],
        [
            [38.0, 39.0, 27.0, 26.0],
            [20.0, 21.0, 46.0, 47.0],
            [5.0, 5.0, 3.0, 3.0],
        ],
    ),
    (
        5132,
        [0x3f80_0000, 0x3f80_0000, 0xbf80_0000],
        [
            [3.0, 46.0, 3.0, 21.0],
            [39.0, 1.0, 27.0, 1.0],
            [1.0, 6.0, 1.0, 3.0],
        ],
    ),
    (
        5135,
        [0x3f80_0000, 0x3e80_0000, 0xbf80_0000],
        [
            [6.0, 49.0, 46.0, 3.0],
            [42.0, 4.0, 1.0, 39.0],
            [1.0, 6.0, 6.0, 1.0],
        ],
    ),
];

/// The emulation answers Metal's own lanes ([`PROBE_LANES`]) on the probe's identity cube
/// (`rgba32Float`, face `f`'s texel `(i, j)` holding `(f + 1, i + 1, j + 1, 1 + 9 f
/// + 3 j + i)`), bit for bit.
#[test]
fn a_cube_gather_answers_metals_probe_lanes() {
    let p = lavapipe();
    let class = texel_format(PixelFormat::RGBA32Float).0;
    let lanes: Vec<[u32; 4]> = PROBE_LANES
        .iter()
        .map(|(_, d, _)| [d[0], d[1], d[2], 0])
        .collect();
    let identity = |l: u32, i: u32, j: u32| -> [u32; 4] {
        [l + 1, i + 1, j + 1, 1 + 9 * l + 3 * j + i].map(|v| (v as f32).to_bits())
    };
    // (name, sampler, channel, Metal's result by lane)
    let runs = [
        ("w edge", near(), 3, 0),
        ("w repeat", axes(2, 2, 2), 3, 1),
        ("x edge", near(), 0, 2),
    ];
    for (name, words, component, at) in runs {
        let Some(bc) = assemble(name, &kernel(words, Texel::Float, false, component)) else {
            return;
        };
        let (plan, m) = typed_module_with(&bc, "k", &p, &inputs(false))
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let imgs = images(&m, (3, 6), false, &identity);
        let out = run(&m, &resources(&plan, &lanes, 16, class), &imgs, false);
        for ((index, _, metal), got) in PROBE_LANES.iter().zip(&out) {
            let want = metal[at].map(f32::to_bits);
            assert_eq!(
                *got, want,
                "{name} probe lane {index}: {got:08x?} vs Metal's {want:08x?}"
            );
        }
    }
}

/// Our probe's references, in each face's order (`refs-cc.bin`): every lane reads its
/// face's centre, the face's four texels of a cube of 2x2 faces.
const CG3_REFS: [u32; 19] = [
    0xbf80_0000,
    0x8000_0000,
    0x0000_0000,
    0x3f80_0000,
    0x3fc0_0000,
    0x4000_0000,
    0x7fc0_0000,
    0x7f80_0000,
    0xff80_0000,
    0x0000_0001,
    0x8000_0001,
    0x3f00_0000,
    0xbe80_0000,
    0x447a_0000,
    0x4040_0000,
    0x3f80_0001,
    0x3f7f_ffff,
    0x3780_0080,
    0x3f7f_ff00,
];

/// Our probe's `Depth32Float` cube (`tex-depth32float.bin`, face after face, rows of the
/// face): texels outside `[0, 1]`, NaN, the infinities, `-0`, and subnormals.
const CG3_FLOATS: [u32; 24] = [
    0xbf80_0000,
    0x4000_0000,
    0x447a_0000,
    0x8000_0000,
    0x0000_0000,
    0x3f80_0000,
    0x7fc0_0000,
    0x7f80_0000,
    0xff80_0000,
    0x0000_0001,
    0x3f00_0000,
    0x3fc0_0000,
    0x8000_0001,
    0xbe80_0000,
    0x4040_0000,
    0x7149_f2ca,
    0x3f80_0000,
    0x0000_0000,
    0xbf80_0000,
    0x4000_0000,
    0x3f80_0001,
    0x3f7f_ffff,
    0x8000_0000,
    0x0000_0000,
];

/// Our probe's `Depth16Unorm` cube's codes (`tex-depth16unorm.bin`): 0, 1, 65534, 65535,
/// and the codes beside 0.5.
const CG3_CODES: [u64; 24] = [
    0, 65535, 0, 65535, 65535, 0, 1, 65534, 0, 0, 65535, 65535, 1, 65534, 32768, 32767, 65535,
    65535, 0, 1, 0, 32768, 65535, 65534,
];

/// Metal's comparison gathers on our probe's cubes: per format and compare code
/// ([`compare_sampler`]), per face, per reference of [`CG3_REFS`], the taps that pass (bit `k`
/// the gather-order tap `k`; `out-cc-<format>.bin`).
type Cg3 = (PixelFormat, u64, [[u8; 19]; 6]);

const CG3: [Cg3; 4] = [
    (
        PixelFormat::Depth32Float,
        2,
        [
            [15, 7, 7, 5, 5, 5, 0, 0, 15, 5, 7, 5, 7, 1, 1, 5, 5, 5, 5],
            [
                14, 14, 14, 6, 2, 2, 0, 2, 14, 6, 14, 6, 14, 2, 2, 2, 6, 6, 6,
            ],
            [7, 7, 7, 2, 2, 0, 0, 0, 15, 7, 7, 3, 7, 0, 0, 2, 2, 3, 2],
            [15, 3, 3, 3, 3, 3, 0, 0, 15, 3, 11, 3, 15, 2, 3, 3, 3, 3, 3],
            [
                15, 14, 14, 10, 2, 2, 0, 0, 15, 10, 14, 10, 14, 0, 0, 2, 10, 10, 10,
            ],
            [
                15, 15, 15, 8, 0, 0, 0, 0, 15, 12, 15, 12, 15, 0, 0, 8, 12, 12, 12,
            ],
        ],
    ),
    (
        PixelFormat::Depth32Float,
        6,
        [
            [
                7, 13, 13, 15, 15, 11, 15, 15, 15, 15, 15, 15, 15, 14, 15, 15, 15, 15, 15,
            ],
            [
                15, 7, 7, 11, 15, 15, 15, 13, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
            ],
            [
                15, 15, 15, 15, 13, 15, 15, 15, 7, 11, 15, 14, 15, 15, 15, 15, 15, 15, 15,
            ],
            [
                15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 7, 15, 11, 15, 14, 15, 15, 15, 15,
            ],
            [
                14, 11, 11, 7, 15, 13, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
            ],
            [
                15, 12, 12, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 7, 11, 15, 15,
            ],
        ],
    ),
    (
        PixelFormat::Depth16Unorm,
        2,
        [
            [
                15, 15, 15, 6, 6, 6, 0, 6, 15, 15, 15, 6, 15, 6, 6, 6, 6, 6, 6,
            ],
            [
                15, 15, 15, 8, 8, 8, 0, 8, 15, 15, 15, 10, 15, 8, 8, 8, 8, 11, 10,
            ],
            [
                15, 15, 15, 3, 3, 3, 0, 3, 15, 15, 15, 3, 15, 3, 3, 3, 3, 3, 3,
            ],
            [
                15, 15, 15, 0, 0, 0, 0, 0, 15, 15, 15, 5, 15, 0, 0, 0, 0, 15, 4,
            ],
            [
                15, 15, 15, 12, 12, 12, 0, 12, 15, 15, 15, 12, 15, 12, 12, 12, 12, 14, 12,
            ],
            [
                15, 15, 15, 1, 1, 1, 0, 1, 15, 15, 15, 7, 15, 1, 1, 1, 1, 7, 3,
            ],
        ],
    ),
    (
        PixelFormat::Depth16Unorm,
        6,
        [
            [6, 6, 6, 9, 9, 9, 15, 9, 6, 6, 6, 15, 6, 9, 9, 9, 9, 15, 15],
            [
                11, 11, 11, 7, 7, 7, 15, 7, 11, 11, 11, 15, 11, 7, 7, 7, 7, 14, 13,
            ],
            [
                3, 3, 3, 12, 12, 12, 15, 12, 3, 3, 3, 15, 3, 12, 12, 12, 12, 15, 15,
            ],
            [
                15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 14, 15, 15, 15, 15, 15, 7, 11,
            ],
            [
                14, 14, 14, 3, 3, 3, 15, 3, 14, 14, 14, 15, 14, 3, 3, 3, 3, 13, 15,
            ],
            [
                7, 7, 7, 14, 14, 14, 15, 14, 7, 7, 7, 11, 7, 14, 14, 14, 14, 15, 13,
            ],
        ],
    ),
];

/// The emulation answers Metal's own comparison gathers of our probe ([`CG3`]) under
/// `less_equal` and `not_equal`, bit for bit: `Depth32Float` texels outside `[0, 1]`, NaN, and the
/// infinities compared as held, and `Depth16Unorm`'s reference clamped and rounded to its code.
#[test]
fn a_cube_comparison_answers_metals_probe_lanes() {
    let p = lavapipe();
    let dirs = [
        [1.0f32, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ];
    let lanes: Vec<[u32; 4]> = dirs
        .iter()
        .flat_map(|d| CG3_REFS.map(|r| [d[0].to_bits(), d[1].to_bits(), d[2].to_bits(), r]))
        .collect();
    for (pf, op, metal) in CG3 {
        let name = format!("{pf:?} compare {op}");
        let Some(bc) = assemble(
            &name,
            &kernel_of(compare_sampler(op), Gather::Compare, false),
        ) else {
            return;
        };
        let (plan, m) = typed_module_with(&bc, "k", &p, &inputs(false))
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let class = texel_format(pf).0;
        let texel = |l: u32, i: u32, j: u32| -> [u32; 4] {
            let at = (4 * l + 2 * j + i) as usize;
            let d = match pf {
                PixelFormat::Depth16Unorm => f32_bits(CG3_CODES[at], 65535),
                _ => CG3_FLOATS[at],
            };
            [d, 0, 0, 1f32.to_bits()]
        };
        let imgs = images(&m, (2, 6), false, &texel);
        let out = run(&m, &resources(&plan, &lanes, 16, class), &imgs, false);
        for (k, got) in out.iter().enumerate() {
            let want: [u32; 4] = std::array::from_fn(|t| {
                let pass = metal[k / CG3_REFS.len()][k % CG3_REFS.len()] >> t & 1;
                f32::from(pass).to_bits()
            });
            assert_eq!(
                *got,
                want,
                "{name} face {} reference {:#010x}: {got:08x?} vs Metal's {want:08x?}",
                k / CG3_REFS.len(),
                CG3_REFS[k % CG3_REFS.len()]
            );
        }
    }
}

/// 24 of our probe's references (`refs-q.bin`): per reference `m` the code `k` beside
/// it, and the code Metal compared it as, the exact product `m · 65535` rounded, where rounding
/// its `f32` rounding (`k + 1/2` exactly) to even picks the other code.
const CG4: [(u64, u32, u64); 24] = [
    (9247, 0x3e10_7e90, 9247),
    (54674, 0x3f55_9356, 54675),
    (44434, 0x3f2d_932e, 44435),
    (30830, 0x3ef0_ddf1, 30831),
    (58165, 0x3f63_3663, 58165),
    (39539, 0x3f1a_741a, 39539),
    (12344, 0x3e40_e2c1, 12345),
    (63470, 0x3f77_ef78, 63471),
    (24497, 0x3ebf_63bf, 24497),
    (57772, 0x3f61_ad62, 57773),
    (47514, 0x3f39_9b3a, 47515),
    (62446, 0x3f73_ef74, 62447),
    (30899, 0x3ef1_67f1, 30899),
    (36012, 0x3f0c_ad0d, 36013),
    (49600, 0x3f41_c142, 49601),
    (21216, 0x3ea5_c1a6, 21217),
    (65248, 0x3f7e_e17f, 65249),
    (59417, 0x3f68_1a68, 59417),
    (40787, 0x3f1f_541f, 40787),
    (48076, 0x3f3b_cd3c, 48077),
    (3111, 0x3d42_78c2, 3111),
    (18979, 0x3e94_4794, 18979),
    (45306, 0x3f30_fb31, 45307),
    (56534, 0x3f5c_d75d, 56535),
];

/// A `Depth16Unorm` comparison compares the code of its reference's exact product with 65535, as
/// Metal does on [`CG4`]'s references: on cubes of 2x2 faces, each face holding
/// the codes `k` and `k + 1` on either side of a reference's half-way point, `equal` passes on the
/// taps holding Metal's code.
#[test]
fn a_depth16_reference_takes_its_exact_code() {
    let p = lavapipe();
    let name = "depth16 equal";
    let Some(bc) = assemble(name, &kernel_of(compare_sampler(5), Gather::Compare, false)) else {
        return;
    };
    let (plan, m) =
        typed_module_with(&bc, "k", &p, &inputs(false)).unwrap_or_else(|e| panic!("{name}: {e}"));
    let class = texel_format(PixelFormat::Depth16Unorm).0;
    let dirs = [
        [1.0f32, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ];
    for run6 in CG4.as_chunks::<6>().0.iter() {
        let lanes: Vec<[u32; 4]> = dirs
            .iter()
            .zip(run6)
            .map(|(d, (_, r, _))| [d[0].to_bits(), d[1].to_bits(), d[2].to_bits(), *r])
            .collect();
        // Face `l`'s texels: `k` in its first column, `k + 1` in its second.
        let texel = |l: u32, i: u32, _: u32| -> [u32; 4] {
            let code = run6[l as usize].0 + u64::from(i);
            [f32_bits(code, 65535), 0, 0, 1f32.to_bits()]
        };
        let imgs = images(&m, (2, 6), false, &texel);
        let out = run(&m, &resources(&plan, &lanes, 16, class), &imgs, false);
        for ((k, r, metal), got) in run6.iter().zip(&out) {
            // Gather order: `x` (0, 1) and `w` (0, 0) hold `k`, `y` and `z` `k + 1`.
            let held = [*k, k + 1, k + 1, *k];
            let want = held.map(|c| f32::from(u8::from(c == *metal)).to_bits());
            assert_eq!(
                *got, want,
                "{name} reference {r:#010x} beside {k}: {got:08x?} vs Metal's {want:08x?}"
            );
        }
    }
}

// ---- Selection, interface, and size -------------------------------------------------------------

/// Whether artifact `a` emulates its one gather, whose census name is `census` (`.cube` where it
/// is emulated).
fn emulated(a: &FunctionArtifact, census: &str) -> bool {
    let cube = a
        .strategies
        .entries
        .get(&format!("{census}.cube"))
        .map(|u| u.uses);
    let plain = a.strategies.entries.get(census).map(|u| u.uses);
    assert_eq!(
        cube.is_some(),
        plain.is_none(),
        "the census of {:?}",
        a.strategies
    );
    cube == Some(1)
}

/// On lavapipe a cube gather through a constexpr sampler is emulated whatever its filters, mip
/// filter, and address modes, a cube array's, an integer result's, a depth cube's, and a depth
/// cube's comparison too (`sampler.gather_compare.cube`), and records `CubeFaceReciprocal` (and for
/// a `float` or `half` result `DeviceGatherFootprint` and the slot's `TexelFormat` specialization),
/// declaring the slot's 2D-array view of its faces. On MoltenVK nothing is emulated, and every
/// gather requires `VkProperty::MetalSampler`.
#[test]
fn the_cube_gather_emulation_reaches_what_metal_measured() {
    let color = |texel, component| Gather::Color(texel, component);
    // (name, sampler, gather, array)
    let cases = [
        ("float", near(), color(Texel::Float, 1), false),
        (
            "half linear",
            sampler_words(1, 0, 1, 0),
            color(Texel::Half, 1),
            false,
        ),
        (
            "float mip",
            sampler_words(1, 2, 2, 0),
            color(Texel::Float, 1),
            false,
        ),
        (
            "half zero",
            sampler_words(0, 0, 0, 2),
            color(Texel::Half, 1),
            false,
        ),
        ("int", near(), color(Texel::Int, 1), false),
        ("uint array", near(), color(Texel::Uint, 1), true),
        ("float array", axes(2, 1, 3), color(Texel::Float, 1), true),
        ("depth", sampler_words(1, 0, 1, 0), Gather::Depth, false),
        ("depth array", near(), Gather::Depth, true),
        ("compare", compare_sampler(2), Gather::Compare, false),
        ("compare array", compare_sampler(6), Gather::Compare, true),
    ];
    let (lvp, m3) = (lavapipe(), m3());
    let faces = RuntimeFact::ImageViewShape { slot: TEXTURE };
    let format = RuntimeFact::TexelFormat { slot: TEXTURE };
    for (name, words, gather, array) in cases {
        let Some(bc) = assemble(name, &kernel_of(words, gather, array)) else {
            return;
        };
        let integer = matches!(gather, Gather::Color(t, _) if t.integer());
        let census = match gather {
            Gather::Compare => "sampler.gather_compare",
            _ => "sampler.gather",
        };
        let a = compile(&bc, "k", &lvp).unwrap_or_else(|e| panic!("{name} on lavapipe: {e}"));
        link_and_validate(&a, &lvp).unwrap_or_else(|e| panic!("{name} on lavapipe: {e}"));
        let what = format!("{name} on lavapipe");
        assert!(emulated(&a, census), "{what}");
        assert!(
            a.precision.contains(&PrecisionNote::CubeFaceReciprocal),
            "{what}"
        );
        assert_eq!(
            a.precision.contains(&PrecisionNote::DeviceGatherFootprint),
            !integer,
            "{what}"
        );
        assert!(a.facts.contains(&faces), "{what}");
        assert_eq!(a.facts.contains(&format), !integer, "{what}");
        assert!(
            !a.requirements
                .properties
                .contains(&VkProperty::MetalSampler),
            "{what}"
        );
        let (_, m) = typed_module_with(&bc, "k", &lvp, &inputs(false))
            .unwrap_or_else(|e| panic!("{what}: {e}"));
        let view = resource(&m, |k| *k == ResourceKind::Texture);
        assert!(view.is_some(), "{what}: the view of the faces");
        let b = compile(&bc, "k", &m3).unwrap_or_else(|e| panic!("{name} on moltenvk-m3: {e}"));
        link_and_validate(&b, &m3).unwrap_or_else(|e| panic!("{name} on moltenvk-m3: {e}"));
        let what = format!("{name} on moltenvk-m3");
        assert!(!emulated(&b, census), "{what}");
        assert!(
            !b.precision.contains(&PrecisionNote::CubeFaceReciprocal),
            "{what}"
        );
        assert!(!b.facts.contains(&faces), "{what}");
        assert!(
            b.requirements
                .properties
                .contains(&VkProperty::MetalSampler),
            "{what}"
        );
    }
}

/// The plan's word estimate bounds the emitted module of every cube gather form (`float`, `half`,
/// and integer results, a cube array, an sRGB variant's raw view, a repeating sampler's wrap, a
/// depth cube's gather and comparison), which validates, on every committed profile.
#[test]
fn the_cube_gather_estimate_bounds_its_emission() {
    let mut modules = Vec::new();
    let repeat = axes(2, 2, 2);
    for (name, words, texel, array, srgb, component) in [
        ("float", near(), Texel::Float, false, false, 0),
        ("half", near(), Texel::Half, false, false, 1),
        ("int", near(), Texel::Int, false, false, 2),
        ("uint array", near(), Texel::Uint, true, false, 3),
        ("float array", near(), Texel::Float, true, false, 3),
        ("raw half", near(), Texel::Half, false, true, 0),
        ("raw float array", near(), Texel::Float, true, true, 3),
        ("repeat half", repeat, Texel::Half, false, false, 2),
        (
            "repeat raw float array",
            repeat,
            Texel::Float,
            true,
            true,
            1,
        ),
    ] {
        let Some(bc) = assemble(name, &kernel(words, texel, array, component)) else {
            return;
        };
        modules.push((name, bc, srgb));
    }
    for (name, words, gather, array) in [
        ("depth", near(), Gather::Depth, false),
        ("depth array", repeat, Gather::Depth, true),
        ("compare", compare_sampler(1), Gather::Compare, false),
        ("compare array", compare_sampler(6), Gather::Compare, true),
        (
            "repeat compare",
            axes(2, 2, 2) & !(0xf << 16) | 4 << 16,
            Gather::Compare,
            false,
        ),
    ] {
        let Some(bc) = assemble(name, &kernel_of(words, gather, array)) else {
            return;
        };
        modules.push((name, bc, false));
    }
    let consts = FunctionConstantValues::defaults();
    for (name, bc, srgb) in &modules {
        for (stem, p) in committed_exact_profiles().expect("profiles") {
            let what = format!("{name} on {stem}");
            // A profile without cube arrays refuses a cube array's gather, whose combined binding
            // stays declared.
            let cube_arrays = p
                .fields()
                .enabled
                .features
                .contains(&VkFeature::ImageCubeArray);
            if name.contains("array") && !cube_arrays {
                continue;
            }
            let inputs = inputs(*srgb);
            let entry = EntryRef::Name("k");
            let plan = plan_bitcode(bc, entry, &consts, &p, &inputs, &mut meter())
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            let a = compile_bitcode(bc, entry, &consts, &p, &inputs, &mut meter(), SPIRV_WORDS)
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: {e}"));
            let words = a.module.words.len() as u64;
            eprintln!("{what}: estimate {} emitted {words}", plan.estimate.words);
            assert!(
                plan.estimate.words >= words,
                "{what}: estimate {} < emitted {words}",
                plan.estimate.words
            );
        }
    }
}
