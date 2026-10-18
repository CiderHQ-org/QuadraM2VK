//! Metal's linear filter of a cube texture, emulated by fetches on a device that does not sample
//! through Metal's own sampler, on the CPU interpreter:
//!
//! - Metal's own answers: seven lanes per format family of a probe (its `cl` run
//!   on 1x1 cubes, `lanes-cube-1.bin`; Apple M4 on macOS 27.0.1 and Apple M3 Max on macOS 26.7,
//!   byte-identical), bit for bit, with `float` and `half` results: 8-bit unorm and snorm,
//!   `R8Unorm`, 8-bit sRGB (through the raw view), 16-bit unorm and snorm, half-float, and 32-bit
//!   float. On a 1x1 cube each lane reads one tap on its face, two across its edges, and one past
//!   both edges, whose texel is the format's corner rule: a normalized format's made-up texel, a
//!   float format's `x` face's.
//! - The rule: random texels of every family on faces of 1, 2, 3, and 5 texels and a cube array's
//!   cubes, directions near the faces' edges and corners and at random, majors of 1 and scaled,
//!   `half` and `float` results, bit for bit against a reference that finds an edge tap's texel
//!   as the texel whose centre on the cube is nearest the tap's.
//! - The selection and the interface: an emulated sample records its filter note and
//!   `CubeFaceReciprocal` and declares the slot's 2D-array view of its faces
//!   (`RuntimeFact::ImageViewShape`) beside its combined binding; every address mode reaches it;
//!   a device with `VkProperty::MetalSampler` keeps every one.
//! - The word estimate bounds the emission, which validates, on every committed profile.
//!
//! `llvm-as` assembles the kernels; without it the tests skip visibly.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::budget::SPIRV_WORDS;
use crate::device::profile::DeviceProfile;
use crate::device::requirements::{VkFeature, VkProperty};
use crate::library::{FunctionArtifact, FunctionConstantValues};
use crate::lower::select::{LowerInputs, Plan};
use crate::metal::bicubic::BicubicTexels;
use crate::metal::texel_format::texel_format;
use crate::metal::PixelFormat;
use crate::pipeline::compile::{compile_bitcode, plan_bitcode, EntryRef};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::interface::PushLoc;
use crate::pipeline::sym::{
    Axis, PushField, ResourceKind, ResourceSlot, SlotClass, SlotRef, SpecKey, Stage,
};
use crate::pipeline::testing::EXACT_SAMPLING;
use crate::pipeline::testing::{
    compile, link_and_validate, llvm_assemble, meter, profile, typed_module_with, AssembleError,
};
use crate::pipeline::PrecisionNote;
use crate::spirv::interp::{run_with_images, Buffer, Config, Image, Order, Resources};
use crate::spirv::ir::{GlobalKind, TypedModule};
use crate::spirv::testing::committed_exact_profiles;

use super::engine_linear_half::{depth_passes, plane_sum, Stored, Tap, Texels};
use super::engine_srgb_raw::{f32_bits, half_bits, step};

const HEADER: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n";

pub(super) const TEXTURE: SlotRef = SlotRef {
    stage: Stage::Kernel,
    class: SlotClass::Texture,
    index: 0,
};

/// Bitcode of the module `body` (after the AIR header), or `None` without `llvm-as`.
pub(super) fn assemble(name: &str, body: &str) -> Option<Vec<u8>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "m2v-linear-cube-{}-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed),
        name.replace(' ', "-")
    ));
    std::fs::create_dir_all(&dir).expect("scratch");
    let ll = dir.join("k.ll");
    std::fs::write(&ll, format!("{HEADER}{body}")).expect("write");
    let r = llvm_assemble(&ll);
    let _ = std::fs::remove_dir_all(&dir);
    match r {
        Ok(x) => Some(x),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("{name}: {why}"),
    }
}

pub(super) fn lavapipe() -> DeviceProfile {
    profile("lavapipe", EXACT_SAMPLING).expect("profile")
}

pub(super) fn m3() -> DeviceProfile {
    profile("moltenvk-m3", EXACT_SAMPLING).expect("profile")
}

/// The words of a constexpr sampler (the `hlf_half_address` fixture's fields): minification and
/// magnification `filter` (0 nearest, 1 linear), mip filter `mip` (0 none, 2 linear), every axis
/// addressed by `address` (0 zero or border, 1 edge, 2 repeat, 3 mirrored repeat), and border
/// color code `border` (0 transparent black, 2 opaque white).
pub(super) fn sampler_words(filter: u64, mip: u64, address: u64, border: u64) -> u64 {
    const BASE: u64 = 0x007b_ff00_0008_0000;
    BASE | address
        | address << 3
        | address << 6
        | filter << 9
        | filter << 11
        | mip << 13
        | border << 56
}

/// A linear sampler clamping to the edge, without a mip filter: the probes'.
pub(super) fn edge() -> u64 {
    sampler_words(1, 0, 1, 0)
}

/// A kernel sampling `texturecube` slot 0 (`texturecube_array` where `array`) through constexpr
/// sampler `words` along `float4` buffer 0's lane `tid` (its `xyz`; its `w`'s bits the array
/// index), storing the `half4` (`float4` unless `half`) to buffer 1.
fn kernel(words: u64, half: bool, array: bool) -> String {
    let (texel, vec4, size, out_ty, suffix) = if half {
        ("half", "<4 x half>", 8, "half4", "v4f16")
    } else {
        ("float", "<4 x float>", 16, "float4", "v4f32")
    };
    let (dim, tex_ty, layer, layer_ty) = if array {
        (
            "cube_array",
            format!("texturecube_array<{texel}, sample>"),
            "i32 %a, ",
            "i32, ",
        )
    } else {
        ("cube", format!("texturecube<{texel}, sample>"), "", "")
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
  %r = call {{ {vec4}, i8 }} @air.sample_texture_{dim}.{suffix}(ptr addrspace(1) %t, ptr addrspace(2) @__air_sampler_state, <3 x float> %d, {layer}i1 false, float 0.0, float 0.0, i32 0)
  %v = extractvalue {{ {vec4}, i8 }} %r, 0
  %q = getelementptr inbounds {vec4}, ptr addrspace(1) %out, i64 %i
  store {vec4} %v, ptr addrspace(1) %q, align {size}
  ret void
}}

declare {{ {vec4}, i8 }} @air.sample_texture_{dim}.{suffix}(ptr addrspace(1), ptr addrspace(2), <3 x float>, {layer_ty}i1, float, float, i32)

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

/// The lowering inputs: the slot bound in an 8-bit sRGB format where `srgb` (its variant).
pub(super) fn inputs(srgb: bool) -> LowerInputs {
    let mut i = LowerInputs::new();
    if srgb {
        i.srgb_textures.insert(TEXTURE);
    }
    i
}

/// The module's resource of texture slot 0 whose kind `kind` accepts.
pub(super) fn resource(
    m: &TypedModule,
    kind: impl Fn(&ResourceKind) -> bool,
) -> Option<ResourceSlot> {
    m.globals().iter().find_map(|g| match g.kind {
        GlobalKind::Resource(s) if s.index == 0 && kind(&s.kind) => Some(s),
        _ => None,
    })
}

/// Resources of a dispatch of one-thread threadgroups over `lanes` (buffer 0, a direction and an
/// array index each), the result's `bytes` per lane (buffer 1), and the slot's `TexelFormat`
/// specialization `class`.
pub(super) fn resources(plan: &Plan, lanes: &[[u32; 4]], bytes: usize, class: u32) -> Resources {
    let mut spec = BTreeMap::new();
    for axis in [Axis::X, Axis::Y, Axis::Z] {
        spec.insert(SpecKey::LocalSize(axis), 1);
    }
    spec.insert(SpecKey::FlatLocalSize, 1);
    spec.insert(SpecKey::TexelFormat(TEXTURE), u64::from(class));
    let push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
    for f in &plan.push.fields {
        match (f.field, f.loc) {
            (PushField::SlotBindOffset(_) | PushField::DispatchRegion, PushLoc::Push { .. }) => {}
            // A bound value read as data, which the interpreter writes from `spec`.
            (PushField::Specialized(_), PushLoc::Push { .. }) => {}
            other => panic!("unexpected push field {other:?}"),
        }
    }
    let buffer = |index: u32, bytes: Vec<u8>| Buffer {
        slot: Some(ResourceSlot {
            stage: Stage::Kernel,
            kind: ResourceKind::Buffer,
            index,
        }),
        address: None,
        bytes,
    };
    let input = lanes
        .iter()
        .flatten()
        .flat_map(|w| w.to_le_bytes())
        .collect();
    Resources {
        buffers: vec![buffer(0, input), buffer(1, vec![0xab; bytes * lanes.len()])],
        push,
        spec,
    }
}

/// The run of `m` over `lanes` in both scheduling orders, which must agree: per lane the result's
/// four channels' bits (a `half`'s when `half`).
pub(super) fn run(m: &TypedModule, res: &Resources, images: &[Image], half: bool) -> Vec<[u32; 4]> {
    let lanes = res.buffers[0].bytes.len() / 16;
    let mut outs = Vec::new();
    for order in [Order::Forward, Order::Reverse] {
        let cfg = Config {
            workgroups: [lanes as u32, 1, 1],
            order,
            ..Config::default()
        };
        let (o, _) = run_with_images(m, &cfg, res.clone(), images.to_vec())
            .unwrap_or_else(|e| panic!("interpreter: {e:?}"));
        outs.push(o.buffers);
    }
    assert_eq!(outs[0], outs[1], "the scheduling orders disagree");
    let size = if half { 2 } else { 4 };
    outs[0][1]
        .chunks_exact(4 * size)
        .map(|c| {
            std::array::from_fn(|k| {
                let mut b = [0u8; 4];
                b[..size].copy_from_slice(&c[size * k..size * (k + 1)]);
                u32::from_le_bytes(b)
            })
        })
        .collect()
}

/// The images of every texture binding of slot 0, of faces `n` texels wide in `layers` layers (6
/// per cube): the slot's view of its faces (its raw view where `raw`) holds `texel(layer, i, j)`
/// (each channel's `f32` bits as a fetch reads them), and every other binding (the combined
/// binding's cube, the plain view beside a raw one) a NaN no emulated sample may read.
pub(super) fn images(
    m: &TypedModule,
    (n, layers): (u32, u32),
    raw: bool,
    texel: &dyn Fn(u32, u32, u32) -> [u32; 4],
) -> Vec<Image> {
    let faces = match raw {
        true => ResourceKind::RawTexture,
        false => ResourceKind::Texture,
    };
    assert!(
        resource(m, |k| *k == faces).is_some(),
        "no view of the slot's faces"
    );
    let bindings = m.globals().iter().filter_map(|g| match g.kind {
        GlobalKind::Resource(s)
            if s.index == 0
                && matches!(
                    s.kind,
                    ResourceKind::Texture
                        | ResourceKind::RawTexture
                        | ResourceKind::CombinedTextureSampler(_)
                ) =>
        {
            Some(s)
        }
        _ => None,
    });
    bindings
        .map(|slot| {
            let mut img = Image::new(slot, [n, n, 1], layers, 1, 1);
            if slot.kind != faces {
                img.texels.fill([0x7fc0_0001; 4]);
                return img;
            }
            for l in 0..layers {
                for j in 0..n {
                    for i in 0..n {
                        let at = img.index(0, l, 0, [i, j, 0]).expect("texel");
                        img.texels[at] = texel(l, i, j);
                    }
                }
            }
            img
        })
        .collect()
}

// ---- The reference ----------------------------------------------------------------------------

/// Metal's face of direction `d` (ties `x` over `y` over `z`; the lanes hold none), its minor
/// components as Vulkan's face table orients them, and the major magnitude.
fn select(d: [f32; 3]) -> (usize, f32, f32, f32) {
    let a = d.map(f32::abs);
    if a[0] >= a[1] && a[0] >= a[2] {
        match d[0] >= 0.0 {
            true => (0, -d[2], -d[1], a[0]),
            false => (1, d[2], -d[1], a[0]),
        }
    } else if a[1] >= a[2] {
        match d[1] >= 0.0 {
            true => (2, d[0], d[2], a[1]),
            false => (3, d[0], -d[2], a[1]),
        }
    } else {
        match d[2] >= 0.0 {
            true => (4, d[0], -d[1], a[2]),
            false => (5, -d[0], -d[1], a[2]),
        }
    }
}

/// Metal's base texel and 1/256 steps on a face axis of `n` texels of minor component `c` beside
/// major magnitude `ma`: `X = RN(RN(1 + c r) / 2 n)` with `r` Metal's reciprocal of `ma` (its
/// model, `libm::rcp::model`), the sum fused, then `q = floor(256 X + 1/2) - 128`; on a repeating
/// axis (`wraps`) of a size that is not a power of two, a face coordinate `RN(1 + c r) / 2` of 1 or
/// more less 1 first.
fn axis(c: f32, ma: f32, n: u32, wraps: bool) -> (i64, i64) {
    let r = crate::lower::arith::libm::rcp::model(ma);
    let f = c.mul_add(r, 1.0);
    let u = f * 0.5;
    let u = match wraps && !n.is_power_of_two() && u >= 1.0 {
        true => u - 1.0,
        false => u,
    };
    let x = u * n as f32;
    let q = (f64::from(x) * 256.0 + 0.5).floor() as i64 - 128;
    (q >> 8, q & 255)
}

/// The direction of tap `(i, j)` of face `face` in the doubled frame where a face spans `[-n, n]`
/// (Vulkan's face table at `sc = 2 i + 1 - n`, `tc = 2 j + 1 - n`, the major `±n`).
fn direction(face: usize, i: i64, j: i64, n: i64) -> [i64; 3] {
    let (s, t) = (2 * i + 1 - n, 2 * j + 1 - n);
    match face {
        0 => [n, -t, -s],
        1 => [-n, -t, s],
        2 => [s, n, t],
        3 => [s, -n, -t],
        4 => [s, -t, n],
        _ => [-s, -t, -n],
    }
}

/// The texels a tap reads: one (its own, or across one edge the texel whose centre is nearest the
/// tap's), or past both edges of its face the corner texels of the `x`, `y`, and `z` faces there.
fn texels(face: usize, (i, j): (i64, i64), n: i64) -> Vec<(usize, i64, i64)> {
    let d = direction(face, i, j, n);
    let mut all = Vec::new();
    for g in 0..6 {
        for b in 0..n {
            for a in 0..n {
                all.push((g, a, b, direction(g, a, b, n)));
            }
        }
    }
    let past = |v: i64| !(0..n).contains(&v);
    if past(i) && past(j) {
        return (0..3)
            .map(|pick| {
                let mut c = d.map(|v| v.signum() * (n - 1));
                c[pick] = d[pick].signum() * n;
                let t = all.iter().find(|t| t.3 == c).expect("a corner texel");
                (t.0, t.1, t.2)
            })
            .collect();
    }
    let distance = |c: &[i64; 3]| -> i64 { (0..3).map(|k| (c[k] - d[k]).pow(2)).sum() };
    let least = all.iter().map(|t| distance(&t.3)).min().expect("texels");
    let near: Vec<_> = all.iter().filter(|t| distance(&t.3) == least).collect();
    assert_eq!(near.len(), 1, "one nearest texel to {d:?}");
    vec![(near[0].0, near[0].1, near[0].2)]
}

/// The four taps of direction `d` on faces of `n` texels, each with its weight `wx · wy` in units
/// of `2^-16` and its texels ([`texels`]), the face's `s` and `t` axes wrapped where `wrap` says
/// ([`axis`]).
pub(super) type Footprint = Vec<(i64, Vec<(usize, i64, i64)>)>;

pub(super) fn footprint(d: [f32; 3], n: u32, wrap: (bool, bool)) -> Footprint {
    let (face, sc, tc, ma) = select(d);
    let (i, kx) = axis(sc, ma, n, wrap.0);
    let (j, ky) = axis(tc, ma, n, wrap.1);
    let n = i64::from(n);
    [
        (i, j, (256 - kx) * (256 - ky)),
        (i + 1, j, kx * (256 - ky)),
        (i, j + 1, (256 - kx) * ky),
        (i + 1, j + 1, kx * ky),
    ]
    .into_iter()
    .map(|(a, b, w)| (w, texels(face, (a, b), n)))
    .collect()
}

/// A normalized format's code scale and whether it is signed, by its texel class.
pub(super) fn norm(class: u32) -> Option<(i64, bool)> {
    let has = |b: u32| class & b != 0;
    if !has(BicubicTexels::EXACT) {
        return None;
    }
    let signed = has(BicubicTexels::SNORM);
    Some(match (has(BicubicTexels::WIDE16), signed) {
        (false, false) => (255, false),
        (false, true) => (127, true),
        (true, false) => (65535, false),
        (true, true) => (32767, true),
    })
}

/// Metal's result bits of channel `c` over footprint `fp` of a cube of stored texels `stored(layer,
/// i, j)` of format class `class` (`tex` its texels, whose class the sums read), the cube's first
/// layer `base`: the 2D sums ([`plane_sum`]) of the taps, a tap at the cube's corner a normalized
/// format's made-up code `⌊(6 X + 5 Y + 5 Z + 8) / 16⌋` (codes read from `-scale` up) and a float
/// format's `x` face's texel.
fn expected(
    tex: &Texels,
    class: u32,
    fp: &Footprint,
    stored: &dyn Fn(usize, i64, i64) -> [Stored; 4],
    (c, half): (usize, bool),
) -> u32 {
    let taps: Vec<(Tap, i64)> = fp
        .iter()
        .map(|(w, at)| {
            let v = |k: usize| stored(at[k].0, at[k].1, at[k].2)[c];
            let tap = match (at.len(), norm(class)) {
                (3, Some((scale, _))) => {
                    let code = |k: usize| match v(k) {
                        Stored::Code(x) => x.max(-scale),
                        Stored::Value(_) => unreachable!("a float texel of a normalized format"),
                    };
                    Stored::Code((6 * code(0) + 5 * code(1) + 5 * code(2) + 8).div_euclid(16))
                }
                _ => v(0),
            };
            (Tap::Stored(tap), *w)
        })
        .collect();
    plane_sum(tex, &taps, half)
}

/// An 8-bit sRGB format's result bits of channel `c` over footprint `fp` of codes `codes(layer, i,
/// j)`: a color channel's table steps (grid 4095), alpha's `16 · code` (grid 4080), a corner's
/// made-up step or code, each term truncated onto `2^-4`, then `⌊(T + 2^11) / 2^12⌋` divided by the
/// grid.
fn expected_srgb(
    fp: &Footprint,
    codes: &dyn Fn(usize, i64, i64) -> [u32; 4],
    (c, half): (usize, bool),
) -> u32 {
    let unit = |code: u32| -> i64 {
        match c < 3 {
            true => step(code) as i64,
            false => i64::from(code),
        }
    };
    let (scale, grid) = if c < 3 { (1, 4095) } else { (16, 4080) };
    let mut t = 0i64;
    for (w, at) in fp {
        let v = |k: usize| unit(codes(at[k].0, at[k].1, at[k].2)[c]);
        let g = match at.len() {
            3 => (6 * v(0) + 5 * v(1) + 5 * v(2) + 8) >> 4,
            _ => v(0),
        } * scale;
        t += (g * w) >> 4;
    }
    let n = ((t + (1 << 11)) >> 12).clamp(0, grid) as u64;
    match half {
        true => u32::from(half_bits(n, grid as u64)),
        false => f32_bits(n, grid as u64),
    }
}

/// A deterministic source: SplitMix64.
pub(super) struct Rng(pub(super) u64);

impl Rng {
    pub(super) fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    pub(super) fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// Directions on faces of `n` texels as the probes made them: per face axis a position within
/// half a texel of either edge or on a 1/256 step, or anywhere, moved up to two ulps, on a random
/// face, its major `±1`, or (one lane in two) the direction scaled by a factor in `[0.3, 3)`; the
/// `w` lane holds an array index in `[0, cubes]` (one past the last cube) or `-1`.
pub(super) fn directions(n: u32, count: usize, cubes: u32, rng: &mut Rng) -> Vec<[u32; 4]> {
    let n64 = f64::from(n);
    let position = |rng: &mut Rng| -> f32 {
        let x = match rng.below(4) {
            0 => (rng.below(128) as f64 + 0.5) / 256.0,
            1 => n64 - (rng.below(128) as f64 + 0.5) / 256.0,
            2 => (rng.below(u64::from(n)) as f64 + 0.5 + (rng.below(256) as f64 + 0.5) / 256.0)
                .min(n64 - 1.0 / 512.0),
            _ => n64 * (0.001 + 0.998 * (rng.below(1 << 20) as f64 / f64::from(1 << 20))),
        };
        let c0 = (2.0 * x / n64 - 1.0) as f32;
        let moved = f32::from_bits((c0.to_bits() as i64 + rng.below(5) as i64 - 2) as u32);
        if c0 == 0.0 || moved.abs() >= 1.0 {
            c0
        } else {
            moved
        }
    };
    (0..count)
        .map(|k| {
            let face = rng.below(6);
            let (s, t) = (position(rng), position(rng));
            let d = match face {
                0 => [1.0, -t, -s],
                1 => [-1.0, -t, s],
                2 => [s, 1.0, t],
                3 => [s, -1.0, -t],
                4 => [s, -t, 1.0],
                _ => [-s, -t, -1.0],
            };
            let m = match k % 2 {
                0 => 1.0,
                _ => 0.3 + 2.7 * (rng.below(1 << 20) as f32 / (1 << 20) as f32),
            };
            // Metal clamps the index as an unsigned integer: `-1` reads the last cube.
            let index = match rng.below(8) {
                0 => u32::MAX,
                _ => rng.below(u64::from(cubes) + 1) as u32,
            };
            [
                (d[0] * m).to_bits(),
                (d[1] * m).to_bits(),
                (d[2] * m).to_bits(),
                index,
            ]
        })
        .collect()
}

/// The formats whose stored texels [`Texels::random`] makes: every non-sRGB family.
pub(super) const FORMATS: [PixelFormat; 7] = [
    PixelFormat::RGBA8Unorm,
    PixelFormat::RGBA8Snorm,
    PixelFormat::R8Unorm,
    PixelFormat::RGBA16Unorm,
    PixelFormat::RGBA16Snorm,
    PixelFormat::RGBA16Float,
    PixelFormat::RGBA32Float,
];

/// Every lane of random cubes of every format family (faces of 1 texel and of 2, 3, or 5 by
/// format; three cubes of an array) is Metal's rule, bit for bit, with `half` and `float` results;
/// and some lanes of each format read a corner tap.
#[test]
fn linear_cube_samples_take_metals_filter() {
    let p = lavapipe();
    for half in [true, false] {
        for array in [false, true] {
            let name = format!("cube half={half} array={array}");
            let Some(bc) = assemble(&name, &kernel(edge(), half, array)) else {
                return;
            };
            let (plan, m) = typed_module_with(&bc, "k", &p, &inputs(false))
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            let cubes = if array { 3 } else { 1 };
            for (seed, pf) in FORMATS.iter().enumerate() {
                let class = texel_format(*pf).0;
                // Faces of one texel, where every lane reads past the edges, and one more size
                // per format.
                let other = [2u32, 3, 5][seed % 3];
                let sizes = if array { vec![other] } else { vec![1, other] };
                for n in sizes {
                    let mut rng = Rng(0x00c0_be22_0000 + seed as u64 * 64 + u64::from(n));
                    let layers = 6 * cubes;
                    let tex =
                        Texels::random(class, i64::from(n), i64::from(n * layers), rng.next());
                    let stored = |l: u32, i: i64, j: i64| -> [Stored; 4] {
                        tex.t[((i64::from(l * n) + j) * i64::from(n) + i) as usize]
                    };
                    let imgs = images(&m, (n, layers), false, &|l, i, j| {
                        stored(l, i64::from(i), i64::from(j)).map(|s| tex.fetched(s).to_bits())
                    });
                    let lanes = directions(n, 96, cubes, &mut rng);
                    let bytes = if half { 8 } else { 16 };
                    let out = run(&m, &resources(&plan, &lanes, bytes, class), &imgs, half);
                    let mut corners = 0;
                    for (lane, (d, got)) in lanes.iter().zip(&out).enumerate() {
                        let dir = [d[0], d[1], d[2]].map(f32::from_bits);
                        let fp = footprint(dir, n, (false, false));
                        let base = 6 * if array { d[3].min(cubes - 1) } else { 0 };
                        let at = |g: usize, i: i64, j: i64| stored(base + g as u32, i, j);
                        corners += usize::from(fp.iter().any(|(w, t)| *w > 0 && t.len() == 3));
                        for (c, got) in got.iter().enumerate() {
                            let want = expected(&tex, class, &fp, &at, (c, half));
                            assert_eq!(
                                *got, want,
                                "{name} {pf:?} n={n} lane {lane} {dir:?} [{}] channel {c}: \
                                 {got:#010x} vs {want:#010x}",
                                d[3]
                            );
                        }
                    }
                    assert!(corners > 0, "{name} {pf:?} n={n}: no lane reads a corner");
                }
            }
        }
    }
}

/// An 8-bit sRGB format's cube (through the slot's raw view, under its variant) is Metal's sum of
/// the table's steps, its corner tap the made-up step, bit for bit with `half` and `float`
/// results, on random codes in every channel.
#[test]
fn a_raw_cube_sums_the_steps_as_metal_does() {
    let p = lavapipe();
    let class = texel_format(PixelFormat::RGBA8UnormSRGB).0;
    for half in [true, false] {
        let name = format!("srgb cube half={half}");
        let Some(bc) = assemble(&name, &kernel(edge(), half, false)) else {
            return;
        };
        let (plan, m) = typed_module_with(&bc, "k", &p, &inputs(true))
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(
            !plan.precision.contains(&PrecisionNote::SrgbTexelDecode),
            "{name}"
        );
        for n in [1u32, 2, 3] {
            let mut rng = Rng(0x05ee_d022_0000 + u64::from(n));
            let codes: Vec<[u32; 4]> = (0..6 * n * n)
                .map(|_| std::array::from_fn(|_| rng.below(256) as u32))
                .collect();
            let code = |l: usize, i: i64, j: i64| {
                codes[((l as i64 * i64::from(n) + j) * i64::from(n) + i) as usize]
            };
            let imgs = images(&m, (n, 6), true, &|l, i, j| {
                code(l as usize, i64::from(i), i64::from(j)).map(|c| f32_bits(u64::from(c), 255))
            });
            let lanes = directions(n, 160, 0, &mut rng);
            let bytes = if half { 8 } else { 16 };
            let out = run(&m, &resources(&plan, &lanes, bytes, class), &imgs, half);
            for (lane, (d, got)) in lanes.iter().zip(&out).enumerate() {
                let dir = [d[0], d[1], d[2]].map(f32::from_bits);
                let fp = footprint(dir, n, (false, false));
                for (c, got) in got.iter().enumerate() {
                    let want = expected_srgb(&fp, &code, (c, half));
                    assert_eq!(
                        *got, want,
                        "{name} n={n} lane {lane} {dir:?} channel {c}: {got:#x} vs {want:#x}"
                    );
                }
            }
        }
    }
}

/// Directions whose face coordinate on either axis of every face is exactly `±1`, an ulp or two
/// inside, or half way, beside a major magnitude of 1, 3, or 0.75 (ties among them): where a
/// repeating face axis wraps.
pub(super) fn edge_directions() -> Vec<[u32; 4]> {
    let below = |k: u32| f32::from_bits(1.0f32.to_bits() - k);
    let minors = [-1.0, -below(1), -below(2), 0.5, below(2), below(1), 1.0];
    let mut out = Vec::new();
    for m in [1.0f32, 3.0, 0.75] {
        for face in 0..6 {
            for s in minors {
                for t in minors {
                    let d = match face {
                        0 => [1.0, -t, -s],
                        1 => [-1.0, -t, s],
                        2 => [s, 1.0, t],
                        3 => [s, -1.0, -t],
                        4 => [s, -t, 1.0],
                        _ => [-s, -t, -1.0],
                    };
                    let d = d.map(|c| (c * m).to_bits());
                    out.push([d[0], d[1], d[2], 0]);
                }
            }
        }
    }
    out
}

/// Through a linear sampler that repeats on every axis, a face coordinate of exactly 1 wraps to 0
/// on faces of 3 and 5 texels and stays on the last texel on faces of 4: every lane of
/// [`edge_directions`] is the rule, bit for bit, and some lanes of each size that is not a power of
/// two read another footprint than a clamping sampler's.
#[test]
fn a_repeating_face_axis_wraps_a_coordinate_of_one() {
    let p = lavapipe();
    let repeat = sampler_words(1, 0, 2, 0);
    let name = "cube repeat";
    let Some(bc) = assemble(name, &kernel(repeat, false, false)) else {
        return;
    };
    let (plan, m) =
        typed_module_with(&bc, "k", &p, &inputs(false)).unwrap_or_else(|e| panic!("{name}: {e}"));
    let pf = PixelFormat::RGBA32Float;
    let class = texel_format(pf).0;
    let lanes = edge_directions();
    for n in [3u32, 5, 4] {
        let mut rng = Rng(0x0e0d_9e22_0000 + u64::from(n));
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
            let fp = footprint(dir, n, (true, true));
            let clamped = footprint(dir, n, (false, false));
            moved += usize::from(fp != clamped);
            for (c, got) in got.iter().enumerate() {
                let want = expected(&tex, class, &fp, &stored, (c, false));
                assert_eq!(
                    *got, want,
                    "{name} n={n} lane {lane} {dir:?} channel {c}: {got:#010x} vs {want:#010x}"
                );
            }
        }
        match n.is_power_of_two() {
            true => assert_eq!(moved, 0, "{name} n={n}"),
            false => assert!(moved > 0, "{name} n={n}: no lane wraps"),
        }
    }
}

/// A kernel sampling `texturecube<float>` slot 0 through runtime sampler slot 0 along `float4`
/// buffer 0's lane `tid` (its `xyz`), storing the `float4` to buffer 1.
fn runtime_kernel() -> String {
    r#"define void @k(ptr addrspace(1) %t, ptr addrspace(1) %c, ptr addrspace(1) %out, ptr addrspace(2) %s, i32 %tid) {
  %i = zext i32 %tid to i64
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %c, i64 %i
  %c4 = load <4 x float>, ptr addrspace(1) %p, align 16
  %d = shufflevector <4 x float> %c4, <4 x float> zeroinitializer, <3 x i32> <i32 0, i32 1, i32 2>
  %r = call { <4 x float>, i8 } @air.sample_texture_cube.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %s, <3 x float> %d, i1 false, float 0.0, float 0.0, i32 0)
  %v = extractvalue { <4 x float>, i8 } %r, 0
  %q = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i64 %i
  store <4 x float> %v, ptr addrspace(1) %q, align 16
  ret void
}

declare { <4 x float>, i8 } @air.sample_texture_cube.v4f32(ptr addrspace(1), ptr addrspace(2), <3 x float>, i1, float, float, i32)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6, !7}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texturecube<float, sample>", !"air.arg_name", !"t"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"c"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}
!6 = !{i32 3, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"s"}
!7 = !{i32 4, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
"#
    .to_string()
}

/// A cube sample through a runtime sampler slot takes Metal's cube filter by the bound sampler's
/// word: a face axis whose word's mode is `repeat` wraps a face coordinate of 1 on a face of 3
/// texels and not of 4, and `clamp_to_edge` and `mirrored_repeat` read as clamping, every lane of
/// [`edge_directions`] bit for bit on the interpreter.
#[test]
fn a_runtime_sampler_wraps_a_cube_face_by_its_word() {
    use crate::metal::sampler::SamplerWord;
    use crate::metal::SamplerAddressMode as M;
    let p = lavapipe();
    let name = "runtime cube";
    let Some(bc) = assemble(name, &runtime_kernel()) else {
        return;
    };
    let a = compile(&bc, "k", &p).unwrap_or_else(|e| panic!("{name}: {e}"));
    link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{name}: {e}"));
    let state = a.interface.read_keys().into_iter().find_map(|k| match k {
        SpecKey::SamplerState { fields, .. } => Some((k, fields)),
        _ => None,
    });
    let Some((key, fields)) = state else {
        panic!("{name}: no `SamplerState` specialization");
    };
    let (plan, m) =
        typed_module_with(&bc, "k", &p, &inputs(false)).unwrap_or_else(|e| panic!("{name}: {e}"));
    let pf = PixelFormat::RGBA32Float;
    let class = texel_format(pf).0;
    let lanes = edge_directions();
    for (mode, wraps, n) in [
        (M::Repeat, true, 3u32),
        (M::Repeat, true, 4),
        (M::ClampToEdge, false, 5),
        (M::MirrorRepeat, false, 3),
    ] {
        let mut word = SamplerWord::LINEAR;
        for shift in SamplerWord::ADDRESS_SHIFT {
            word |= mode.as_raw() << shift;
        }
        {
            let mut rng = Rng(0x0e0d_9e22_0000 + u64::from(n));
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
            for (lane, (d, got)) in lanes.iter().zip(&out).enumerate() {
                let dir = [d[0], d[1], d[2]].map(f32::from_bits);
                let fp = footprint(dir, n, (wraps, wraps));
                for (c, got) in got.iter().enumerate() {
                    let want = expected(&tex, class, &fp, &stored, (c, false));
                    assert_eq!(
                        *got, want,
                        "{name} {mode:?} n={n} lane {lane} {dir:?} channel {c}: {got:#010x} vs \
                         {want:#010x}"
                    );
                }
            }
        }
    }
}

// ---- Depth comparisons ------------------------------------------------------------------------

/// A kernel comparing `depthcube` slot 0 (`depthcube_array` at cube 1 where `array`) through
/// constexpr sampler `words` along `float4` buffer 0's lane `tid` (its `xyz`) against its `w`,
/// storing the `float` result in the `x` of buffer 1's `float4`.
fn compare_kernel(words: u64, array: bool) -> String {
    let (dim, tex_ty, layer, layer_ty) = match array {
        true => (
            "cube_array",
            "depthcube_array<float, sample>",
            "i32 1, ",
            "i32, ",
        ),
        false => ("cube", "depthcube<float, sample>", "", ""),
    };
    format!(
        r#"@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 {words}, i64 0], align 8

define void @k(ptr addrspace(1) %t, ptr addrspace(1) %c, ptr addrspace(1) %out, i32 %tid) {{
  %i = zext i32 %tid to i64
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %c, i64 %i
  %c4 = load <4 x float>, ptr addrspace(1) %p, align 16
  %d = shufflevector <4 x float> %c4, <4 x float> zeroinitializer, <3 x i32> <i32 0, i32 1, i32 2>
  %ref = extractelement <4 x float> %c4, i64 3
  %r = call {{ float, i8 }} @air.sample_compare_depth_{dim}.f32(ptr addrspace(1) %t, ptr addrspace(2) @__air_sampler_state, i32 1, <3 x float> %d, {layer}float %ref, i1 true, float 0.0, float 0.0, i32 0)
  %v = extractvalue {{ float, i8 }} %r, 0
  %v4 = insertelement <4 x float> zeroinitializer, float %v, i64 0
  %q = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i64 %i
  store <4 x float> %v4, ptr addrspace(1) %q, align 16
  ret void
}}

declare {{ float, i8 }} @air.sample_compare_depth_{dim}.f32(ptr addrspace(1), ptr addrspace(2), i32, <3 x float>, {layer_ty}float, i1, float, float, i32)

!air.kernel = !{{!0}}
!air.sampler_states = !{{!7}}
!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4, !5, !6}}
!3 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"{tex_ty}", !"air.arg_name", !"t"}}
!4 = !{{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"c"}}
!5 = !{{i32 2, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}}
!6 = !{{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}}
!7 = !{{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}}
"#
    )
}

/// The depth formats whose cube comparison the emulation answers (`metal::texel_format`).
const DEPTHS: [PixelFormat; 2] = [PixelFormat::Depth32Float, PixelFormat::Depth16Unorm];

/// Metal's linear cube comparison over footprint `fp` of depths `stored(layer, i, j)` (channel 0 of
/// a cube of `Depth16Unorm` codes where `d16`, else `Depth32Float` values) against reference `r`
/// under AIR compare function code `f`: each tap's depth compared (a tap at the cube's corner
/// `Depth16Unorm`'s made-up code `⌊(6 X + 5 Y + 5 Z + 8) / 16⌋`, else the `x` face's texel), and
/// the passes summed as an `r8Unorm` texture's codes 0 and 255 (`r8` its texels, whose class the
/// sum reads).
fn compared(
    fp: &Footprint,
    stored: &dyn Fn(usize, i64, i64) -> Stored,
    (fetched, d16): (&dyn Fn(Stored) -> f32, bool),
    (f, r): (u64, f32),
    r8: &Texels,
) -> u32 {
    let taps: Vec<(Tap, i64)> = fp
        .iter()
        .map(|(w, at)| {
            let code = |s: Stored| match s {
                Stored::Code(c) => Some(c),
                Stored::Value(_) => None,
            };
            let depth = match (at.len(), d16) {
                (3, true) => {
                    let c = |k: usize| code(stored(at[k].0, at[k].1, at[k].2)).expect("a code");
                    let m = (6 * c(0) + 5 * c(1) + 5 * c(2) + 8).div_euclid(16);
                    ((m as f64 / 65535.0) as f32, Some(m))
                }
                _ => {
                    let s = stored(at[0].0, at[0].1, at[0].2);
                    (fetched(s), code(s))
                }
            };
            let pass = i64::from(depth_passes(f, r, depth));
            (Tap::Stored(Stored::Code(255 * pass)), *w)
        })
        .collect();
    plane_sum(r8, &taps, false)
}

/// A depth cube's linear comparison compares each tap, the corner's by the format's rule
/// (`Depth32Float`'s `x` face's texel, `Depth16Unorm`'s made-up code), and sums the passes as
/// `r8Unorm` codes 0 and 255 (3,072 directions of two cubes and a cube array,
/// every lane): every lane of random depth cubes (faces of 1 and 3 texels; a cube array's second
/// cube) under `less`, `greater_equal` and `not_equal`, references equal to a texel's depth and NaN
/// included, and some lanes reading a corner tap.
#[test]
fn linear_cube_comparisons_sum_their_passes() {
    let p = lavapipe();
    let r8 = Texels::random(texel_format(PixelFormat::R8Unorm).0, 1, 1, 0);
    let refs = [0.5f32, 0.0, 1.0, -0.25, f32::NAN, 0.75];
    for (f, array) in [(1, false), (4, false), (6, false), (1, true)] {
        let name = format!("cube compare {f} array={array}");
        let words = edge() & !(0xf << 16) | f << 16;
        let Some(bc) = assemble(&name, &compare_kernel(words, array)) else {
            return;
        };
        let a = compile(&bc, "k", &p).unwrap_or_else(|e| panic!("{name}: {e}"));
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{name}: {e}"));
        let (plan, m) = typed_module_with(&bc, "k", &p, &inputs(false))
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let cubes = if array { 2 } else { 1 };
        for (seed, pf) in DEPTHS.iter().enumerate() {
            let class = texel_format(*pf).0;
            let d16 = *pf == PixelFormat::Depth16Unorm;
            for n in [1u32, 3] {
                let mut rng = Rng(0x00c0_c0de_0000 + seed as u64 * 64 + u64::from(n) + f * 4096);
                let layers = 6 * cubes;
                let tex = Texels::random(class, i64::from(n), i64::from(n * layers), rng.next());
                let stored = |l: u32, i: i64, j: i64| -> Stored {
                    tex.t[((i64::from(l * n) + j) * i64::from(n) + i) as usize][0]
                };
                let imgs = images(&m, (n, layers), false, &|l, i, j| {
                    let v = tex.fetched(stored(l, i64::from(i), i64::from(j)));
                    [v.to_bits(), 0, 0, 1f32.to_bits()]
                });
                let lanes: Vec<[u32; 4]> = directions(n, 96, 0, &mut rng)
                    .into_iter()
                    .enumerate()
                    .map(|(k, d)| {
                        let r = match k % 3 {
                            0 => tex.fetched(tex.t[(7 * k) % tex.t.len()][0]),
                            _ => refs[k % refs.len()],
                        };
                        [d[0], d[1], d[2], r.to_bits()]
                    })
                    .collect();
                let out = run(&m, &resources(&plan, &lanes, 16, class), &imgs, false);
                let mut corners = 0;
                for (lane, (d, got)) in lanes.iter().zip(&out).enumerate() {
                    let dir = [d[0], d[1], d[2]].map(f32::from_bits);
                    let fp = footprint(dir, n, (false, false));
                    let base = if array { 6 } else { 0 };
                    let at = |g: usize, i: i64, j: i64| stored(base + g as u32, i, j);
                    corners += usize::from(fp.iter().any(|(w, t)| *w > 0 && t.len() == 3));
                    let fetched = |s: Stored| tex.fetched(s);
                    let r = f32::from_bits(d[3]);
                    let want = compared(&fp, &at, (&fetched, d16), (f, r), &r8);
                    assert_eq!(
                        got[0],
                        want,
                        "{name} {pf:?} n={n} lane {lane} {dir:?} reference {r}: {:?} vs {:?}",
                        f32::from_bits(got[0]),
                        f32::from_bits(want)
                    );
                }
                assert!(corners > 0, "{name} {pf:?} n={n}: no lane reads a corner");
            }
        }
    }
}

/// Our probe's 5x5 pattern cube (faces in order, rows of five): each texel's depth 0 or
/// 1 (`Depth32Float`), or code 0 or 65535 (`Depth16Unorm`).
const PX5_PATTERN: &str = "000010111100010001001100011010010101001011001011101000010000001011111111001000101000000010000111100101100000111111001101111011101011001101101110111001";

/// Lanes of our probe's `cc` run (Apple M4, macOS 27.0.1): the lane, its direction's
/// bits, and Metal's `less` comparison against 0.5 of [`PX5_PATTERN`] as `Depth32Float` and as
/// `Depth16Unorm`. The first ten differ between the formats (284 of 3,072 lanes do, all at the
/// corner tap); the rest read the same value from both.
const PX5_LANES: [(u32, [u32; 3], [u32; 2]); 16] = [
    (
        65,
        [0x4065_694a, 0xc071_6dc6, 0xc052_fc0a],
        [0x3f20_0000, 0x3f31_a1a2],
    ),
    (
        94,
        [0xc04d_bb16, 0xc050_13c2, 0xc06c_e404],
        [0x3f54_0404, 0x3f5c_9c9d],
    ),
    (
        119,
        [0xc025_c94a, 0xc031_0dbe, 0xc013_0704],
        [0x3f28_f8f9, 0x3f2f_6f6f],
    ),
    (
        175,
        [0x4035_b662, 0xc058_b412, 0xc044_ebdc],
        [0x3f67_0707, 0x3f6d_ddde],
    ),
    (
        198,
        [0x4006_6fcc, 0xc007_cfc0, 0xc022_861a],
        [0x3f6e_feff, 0x3f70_8081],
    ),
    (
        212,
        [0xc04f_09b2, 0xc065_6868, 0xc073_9272],
        [0x3f60_0000, 0x3f6b_5b5b],
    ),
    (
        2051,
        [0x4053_0af7, 0xc05c_411d, 0xc072_160e],
        [0x3f52_0202, 0x3f5e_8e8f],
    ),
    (
        2054,
        [0xbfaa_f637, 0xbfb4_f003, 0xbfb6_2560],
        [0x3f27_0707, 0x3f52_2222],
    ),
    (
        2056,
        [0xc034_b744, 0xc054_702c, 0xc038_494b],
        [0x3f60_0000, 0x3f65_6565],
    ),
    (
        2060,
        [0xbf9d_b6d7, 0xbf97_bc33, 0xbfb4_5647],
        [0x3f50_0000, 0x3f54_e4e5],
    ),
    (
        0,
        [0x3f9a_0f34, 0x3f1a_6150, 0x3ded_9d80],
        [0x3f3e_feff, 0x3f3e_feff],
    ),
    (
        1,
        [0x404c_3ebc, 0x403a_4578, 0x404d_673e],
        [0x3e0c_4c4c, 0x3e0c_4c4c],
    ),
    (
        4,
        [0x4072_fb9a, 0xbfe3_f9f0, 0xbfc1_f110],
        [0x3b50_d0d1, 0x3b50_d0d1],
    ),
    (
        5,
        [0xc071_289c, 0x3fe4_afb4, 0x4017_adfe],
        [0x3ed9_f9fa, 0x3ed9_f9fa],
    ),
    (
        6,
        [0x3e9d_fe30, 0xc071_da30, 0xbffd_17fc],
        [0x3e0f_cfd0, 0x3e0f_cfd0],
    ),
    (
        7,
        [0xc015_c014, 0xc02b_cc7a, 0xbe5d_9e80],
        [0x3e37_f7f8, 0x3e37_f7f8],
    ),
];

/// The emulation answers [`PX5_LANES`] as Metal did, bit for bit, on both formats: the
/// `Depth16Unorm` corner's made-up code, compared as a texel, is what tells the formats apart.
#[test]
fn a_cube_comparison_answers_metals_px5_lanes() {
    let p = lavapipe();
    let name = "px5 cube compare";
    let words = edge() & !(0xf << 16) | 1 << 16;
    let Some(bc) = assemble(name, &compare_kernel(words, false)) else {
        return;
    };
    let (plan, m) =
        typed_module_with(&bc, "k", &p, &inputs(false)).unwrap_or_else(|e| panic!("{name}: {e}"));
    let pattern: Vec<f32> = PX5_PATTERN
        .bytes()
        .map(|b| f32::from(u8::from(b == b'1')))
        .collect();
    assert_eq!(pattern.len(), 150);
    let imgs = images(&m, (5, 6), false, &|l, i, j| {
        let v = pattern[(l * 25 + j * 5 + i) as usize];
        [v.to_bits(), 0, 0, 1f32.to_bits()]
    });
    let lanes: Vec<[u32; 4]> = PX5_LANES
        .iter()
        .map(|(_, d, _)| [d[0], d[1], d[2], 0.5f32.to_bits()])
        .collect();
    for (k, pf) in DEPTHS.iter().enumerate() {
        let class = texel_format(*pf).0;
        let out = run(&m, &resources(&plan, &lanes, 16, class), &imgs, false);
        for ((lane, _, metal), got) in PX5_LANES.iter().zip(&out) {
            assert_eq!(
                got[0],
                metal[k],
                "{pf:?} lane {lane}: {:?} vs Metal's {:?}",
                f32::from_bits(got[0]),
                f32::from_bits(metal[k])
            );
        }
    }
}

// ---- Metal's own lanes --------------------------------------------------------------------------

/// The directions (bits) of seven lanes of our probe's `cl` run on 1x1 cubes, by their
/// index in `lanes-cube-1.bin`: six with a major of `±1`, one on each face, and one scaled.
const PROBE_LANES: [(u32, [u32; 3]); 7] = [
    (0, [0xbf16_af84, 0xbf7e_fffe, 0x3f80_0000]),
    (1, [0x3e73_662d, 0xbf80_0000, 0xbf35_0001]),
    (2, [0x3f80_0000, 0x3eaa_8d6b, 0x3df8_0000]),
    (4, [0xbde7_ffff, 0x3da3_0c36, 0xbf80_0000]),
    (6, [0xbf0b_0000, 0x3f80_0000, 0xbe74_0002]),
    (11, [0xbf80_0000, 0x3f70_ffff, 0xbed5_fffe]),
    (2048, [0xbfa5_864c, 0xbf8d_f4ca, 0x3fbf_0898]),
];

/// One format's run of the probe on a 1x1 cube: the six faces' texel bytes as uploaded
/// (`tex-<format>-cube-1.bin`, face after face), and Metal's `float` and `half` samples' bits at
/// [`PROBE_LANES`] (`out-<format>-cube-1.bin`), Apple M4 and M3 Max alike.
struct Probe {
    format: PixelFormat,
    texels: &'static [u8],
    float: [[u32; 4]; 7],
    half: [[u16; 4]; 7],
}

fn probes() -> Vec<Probe> {
    vec![
        Probe {
            format: PixelFormat::RGBA8Unorm,
            texels: &[
                0x02, 0xf8, 0xe9, 0xfe, 0x2d, 0xf8, 0x7b, 0x44, 0x5d, 0xd3, 0x41, 0x81, 0x38, 0xc0,
                0x55, 0x88, 0x42, 0x8b, 0x2d, 0xbc, 0x51, 0x51, 0x22, 0x64,
            ],
            float: [
                [0x3e68_2828, 0x3f36_a6a7, 0x3e99_d9da, 0x3f0f_bfc0],
                [0x3e6d_edee, 0x3f21_7171, 0x3ea4_4444, 0x3f07_8788],
                [0x3da0_a0a1, 0x3f6d_4d4d, 0x3f44_f4f5, 0x3f67_5757],
                [0x3e9f_5f5f, 0x3ebf_9fa0, 0x3e21_a1a2, 0x3ec7_4747],
                [0x3ea0_6060, 0x3f50_a0a1, 0x3e99_999a, 0x3ede_dedf],
                [0x3e8b_2b2b, 0x3f52_8283, 0x3eaf_0f0f, 0x3ec2_c2c3],
                [0x3e62_6262, 0x3f3d_cdce, 0x3ea3_8384, 0x3f07_1717],
            ],
            half: [
                [0x3341, 0x39b5, 0x34cf, 0x387e],
                [0x336f, 0x390c, 0x3522, 0x383c],
                [0x2d05, 0x3b6a, 0x3a28, 0x3b3b],
                [0x34fb, 0x35fd, 0x310d, 0x363a],
                [0x3503, 0x3a85, 0x34cd, 0x36f7],
                [0x3459, 0x3a94, 0x3578, 0x3616],
                [0x3313, 0x39ee, 0x351c, 0x3839],
            ],
        },
        Probe {
            format: PixelFormat::RGBA8Snorm,
            texels: &[
                0x89, 0x91, 0x79, 0xb3, 0xb6, 0x62, 0xb5, 0x63, 0x04, 0x3a, 0x5b, 0x4b, 0xae, 0x42,
                0x3c, 0x78, 0xfe, 0xc9, 0xea, 0x79, 0x62, 0x12, 0x2c, 0xc2,
            ],
            float: [
                [0xbec2_850a, 0x3e40_0000, 0xbac1_8306, 0x3f69_f3e8],
                [0xbe56_2c59, 0x3e89_d3a7, 0x3ef3_264d, 0x3eab_972e],
                [0xbf3b_b76f, 0xbf23_060c, 0x3f5b_366d, 0xbeaa_54a9],
                [0x3f2a_952a, 0x3e42_0408, 0x3e9c_f9f4, 0xbec0_4081],
                [0xbd51_a347, 0x3f01_62c6, 0x3eb5_ab57, 0x3f08_f1e4],
                [0xbe15_2a55, 0x3f0e_9d3a, 0x3d88_1020, 0x3f06_8d1a],
                [0xbebc_b973, 0x3e5d_3a75, 0xbe13_264d, 0x3f64_68d2],
            ],
            half: [
                [0xb614, 0x3200, 0x960c, 0x3b50],
                [0xb2b1, 0x344f, 0x3799, 0x355d],
                [0xb9de, 0xb918, 0x3ada, 0xb553],
                [0x3955, 0x3210, 0x34e8, 0xb602],
                [0xaa8d, 0x380b, 0x35ad, 0x3848],
                [0xb0a9, 0x3875, 0x2c41, 0x3834],
                [0xb5e6, 0x32ea, 0xb099, 0x3b23],
            ],
        },
        Probe {
            format: PixelFormat::R8Unorm,
            texels: &[0x48, 0x6b, 0xb4, 0xfe, 0x7a, 0xda],
            float: [
                [0x3f2b_fbfc, 0, 0, 0x3f80_0000],
                [0x3f62_9293, 0, 0, 0x3f80_0000],
                [0x3eb8_b8b9, 0, 0, 0x3f80_0000],
                [0x3f53_1313, 0, 0, 0x3f80_0000],
                [0x3f26_1616, 0, 0, 0x3f80_0000],
                [0x3f18_9899, 0, 0, 0x3f80_0000],
                [0x3f18_0808, 0, 0, 0x3f80_0000],
            ],
            half: [
                [0x3960, 0, 0, 0x3c00],
                [0x3b15, 0, 0, 0x3c00],
                [0x35c6, 0, 0, 0x3c00],
                [0x3a99, 0, 0, 0x3c00],
                [0x3931, 0, 0, 0x3c00],
                [0x38c5, 0, 0, 0x3c00],
                [0x38c0, 0, 0, 0x3c00],
            ],
        },
        Probe {
            format: PixelFormat::RGBA8UnormSRGB,
            texels: &[
                0x2b, 0xc5, 0xb0, 0xcc, 0x2e, 0x51, 0xba, 0xab, 0xe6, 0x08, 0x6b, 0xb8, 0x90, 0xed,
                0xc1, 0xdf, 0xa2, 0xf2, 0x41, 0x7e, 0x1f, 0xb9, 0xf7, 0xc2,
            ],
            float: [
                [0x3e85_c85d, 0x3f35_7b58, 0x3ea9_ca9d, 0x3f2e_2e2e],
                [0x3e2d_0ad1, 0x3f33_cb3d, 0x3f27_6a77, 0x3f54_a4a5],
                [0x3e29_4a95, 0x3ef8_af8b, 0x3ebc_6bc7, 0x3f45_8586],
                [0x3d33_0b31, 0x3ee3_6e37, 0x3f60_2e03, 0x3f41_1111],
                [0x3f06_b86c, 0x3d8c_08c1, 0x3e9d_89d9, 0x3f36_6666],
                [0x3eaa_4aa5, 0x3dda_8da9, 0x3ed4_2d43, 0x3f34_2424],
                [0x3e6a_0ea1, 0x3f1b_e9bf, 0x3ea5_ca5d, 0x3f26_e6e7],
            ],
            half: [
                [0x342e, 0x39ac, 0x354e, 0x3971],
                [0x3168, 0x399e, 0x393b, 0x3aa5],
                [0x314a, 0x37c5, 0x35e3, 0x3a2c],
                [0x2998, 0x371b, 0x3b01, 0x3a09],
                [0x3836, 0x2c60, 0x34ec, 0x39b3],
                [0x3552, 0x2ed4, 0x36a1, 0x39a1],
                [0x3350, 0x38df, 0x352e, 0x3937],
            ],
        },
        Probe {
            format: PixelFormat::RGBA16Unorm,
            texels: &[
                0xed, 0x82, 0xfa, 0x74, 0xc9, 0xf9, 0x61, 0x63, 0x2d, 0x6c, 0x36, 0xfa, 0x9c, 0xe7,
                0xe8, 0x5e, 0x82, 0x86, 0xb6, 0xbb, 0x59, 0x7b, 0x28, 0x60, 0x59, 0x42, 0x94, 0x71,
                0x69, 0xbe, 0x18, 0x65, 0x41, 0xf1, 0x69, 0x7d, 0x53, 0x28, 0x4a, 0x7a, 0x14, 0x77,
                0x0d, 0xe8, 0x08, 0xf5, 0x62, 0x03,
            ],
            float: [
                [0x3f11_1091, 0x3f11_f392, 0x3f0a_738a, 0x3ed8_b6d9],
                [0x3eb2_f0b3, 0x3f18_9c99, 0x3f55_add6, 0x3e8a_028a],
                [0x3f09_3989, 0x3f00_d181, 0x3f5a_8cdb, 0x3ec8_18c8],
                [0x3eee_12ee, 0x3f67_6de7, 0x3f6f_b4f0, 0x3d41_80c2],
                [0x3efc_eafd, 0x3f4f_8ed0, 0x3f22_02a2, 0x3ead_faae],
                [0x3ef0_96f1, 0x3f5e_89df, 0x3f3d_f8be, 0x3ea5_22a5],
                [0x3f17_6097, 0x3f23_e4a4, 0x3f0f_1f8f, 0x3ed7_4cd7],
            ],
            half: [
                [0x3889, 0x3890, 0x3854, 0x36c6],
                [0x3598, 0x38c5, 0x3aad, 0x3450],
                [0x384a, 0x3807, 0x3ad4, 0x3641],
                [0x3771, 0x3b3b, 0x3b7e, 0x2a0c],
                [0x37e7, 0x3a7c, 0x3910, 0x3570],
                [0x3785, 0x3af4, 0x39f0, 0x3529],
                [0x38bb, 0x391f, 0x3879, 0x36ba],
            ],
        },
        Probe {
            format: PixelFormat::RGBA16Snorm,
            texels: &[
                0x0a, 0xb8, 0x15, 0xab, 0xff, 0x7f, 0x1a, 0x84, 0x0c, 0xa7, 0x7c, 0x8f, 0x78, 0xad,
                0xff, 0x7f, 0x5d, 0x8a, 0xff, 0x7f, 0xe1, 0xb5, 0x01, 0x80, 0x0b, 0x57, 0xda, 0x70,
                0x25, 0x0b, 0xaf, 0xfc, 0xff, 0x7f, 0xdf, 0xeb, 0x00, 0x80, 0x01, 0x80, 0x01, 0x80,
                0x01, 0x80, 0xa2, 0x78, 0x00, 0x80,
            ],
            float: [
                [0x3f07_d710, 0x3dde_11bc, 0xbefe_bdfd, 0xbe56_d9ae],
                [0x3c91_4123, 0x3e00_e102, 0x3ee6_51cd, 0xbedc_3db8],
                [0xbf09_ed14, 0xbebc_f57a, 0x3f24_4d49, 0xbf79_8bf3],
                [0xbf7a_c1f6, 0xbf6b_09d6, 0x3f4b_7197, 0xbf62_bbc5],
                [0xbf5e_edbe, 0x3ead_e95c, 0xbee6_85cd, 0xbf00_1b00],
                [0xbf53_efa8, 0xbe0f_f920, 0xbeca_9595, 0xbdad_715b],
                [0x3eb4_5169, 0xbe02_1104, 0xbf19_3b32, 0xbda4_4149],
            ],
            half: [
                [0x383f, 0x2ef1, 0xb7f6, 0xb2b7],
                [0x248a, 0x3007, 0x3733, 0xb6e2],
                [0xb84f, 0xb5e8, 0x3922, 0xbbcc],
                [0xbbd6, 0xbb58, 0x3a5c, 0xbb16],
                [0xbaf7, 0x356f, 0xb734, 0xb801],
                [0xba9f, 0xb080, 0xb655, 0xad6c],
                [0x35a3, 0xb011, 0xb8ca, 0xad22],
            ],
        },
        Probe {
            format: PixelFormat::RGBA16Float,
            texels: &[
                0xcb, 0x61, 0x5f, 0x2d, 0xd6, 0x6f, 0xdb, 0xbf, 0x2e, 0x35, 0xf8, 0xa3, 0xd9, 0x03,
                0x53, 0xc7, 0x88, 0x44, 0xe3, 0x2a, 0x0b, 0xc4, 0xfe, 0x51, 0xca, 0xb1, 0xa2, 0xba,
                0xa4, 0xbf, 0x92, 0x0e, 0x29, 0x70, 0xc9, 0x80, 0xa3, 0x85, 0x89, 0x03, 0xfa, 0x6e,
                0xc4, 0xab, 0x42, 0x42, 0x3c, 0xa8,
            ],
            float: [
                [0x453d_c000, 0xbe97_4000, 0xbf2b_8000, 0xc009_6000],
                [0x4511_8000, 0xbef6_4000, 0x446b_0000, 0xbe76_0000],
                [0x447b_0000, 0x3d99_c000, 0x45c6_e000, 0x40c0_c000],
                [0x45ca_0000, 0xbd5c_6000, 0x402b_a000, 0x3fa6_e000],
                [0x4419_a000, 0x3ccf_6000, 0xc015_c000, 0x41e7_8000],
                [0x4444_e000, 0x3bac_6000, 0xbf94_4000, 0x4160_4000],
                [0x453d_c000, 0xbe39_6000, 0xbecd_a000, 0xc04b_4000],
            ],
            half: [
                [0x69ee, 0xb4ba, 0xb95c, 0xc04b],
                [0x688c, 0xb7b2, 0x6358, 0xb3b0],
                [0x63d8, 0x2cce, 0x6e37, 0x4606],
                [0x6e50, 0xaae3, 0x415d, 0x3d37],
                [0x60cd, 0x267b, 0xc0ae, 0x4f3c],
                [0x6227, 0x1d63, 0xbca2, 0x4b02],
                [0x69ee, 0xb1cb, 0xb66d, 0xc25a],
            ],
        },
        Probe {
            format: PixelFormat::RGBA32Float,
            texels: &[
                0xfd, 0x2f, 0x42, 0x8f, 0xe0, 0xdd, 0x3e, 0xcc, 0xde, 0x9e, 0x7b, 0x48, 0x25, 0x9d,
                0x16, 0x3e, 0xd1, 0xd5, 0xcd, 0xb9, 0x81, 0xf4, 0xb9, 0x49, 0xd2, 0x03, 0x47, 0x4b,
                0xf3, 0xba, 0x33, 0x3f, 0x95, 0xb3, 0x23, 0xba, 0x29, 0xf2, 0xb5, 0xc6, 0xd7, 0x30,
                0x48, 0x36, 0x7d, 0x82, 0xe7, 0xb9, 0x8d, 0x8f, 0xfd, 0xb7, 0x68, 0xa4, 0x69, 0xc1,
                0x31, 0x05, 0x48, 0xc5, 0xc7, 0x13, 0xe7, 0x40, 0x6a, 0x13, 0xf7, 0x3a, 0xd4, 0xdd,
                0xb5, 0xb6, 0xa8, 0xac, 0x57, 0x3e, 0xa3, 0x7a, 0x2a, 0xba, 0x69, 0xd0, 0x6f, 0xc1,
                0x04, 0x00, 0x1f, 0xb9, 0xe6, 0xcd, 0xee, 0x64, 0xa0, 0xb7, 0xb7, 0xb9,
            ],
            float: [
                [0x3a0f_1fca, 0x48d9_e9e3, 0x4a69_26f0, 0x402f_405f],
                [0xc096_8386, 0xcab2_f013, 0x6415_e122, 0x4084_07f4],
                [0xb6e5_bbfc, 0xcc17_64fe, 0x4847_90cf, 0x3dee_b3c2],
                [0xc158_f1c3, 0x47ac_a902, 0x64d8_07b7, 0x3d27_21d8],
                [0xbfa4_4b13, 0x48c1_2650, 0x6323_8961, 0x3e41_730b],
                [0xbfd3_1067, 0x493e_a6cb, 0x6352_1eab, 0x3eba_355b],
                [0x3a01_bc2a, 0x4921_41d7, 0x4aac_900f, 0x3fe9_3735],
            ],
            half: [
                [0x1078, 0x7bff, 0x7bff, 0x417a],
                [0xc4b4, 0xfbff, 0x7bff, 0x4420],
                [0x8072, 0xfbff, 0x7bff, 0x2f75],
                [0xcac7, 0x7bff, 0x7bff, 0x2939],
                [0xbd22, 0x7bff, 0x7bff, 0x320b],
                [0xbe98, 0x7bff, 0x7bff, 0x35d1],
                [0x100d, 0x7bff, 0x7bff, 0x3f49],
            ],
        },
    ]
}

/// The `f32` value of `half` bits `h`, exactly.
fn half_value(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let (e, m) = (i32::from(h >> 10 & 0x1f), f64::from(h & 0x3ff));
    let a = match e {
        0 => m * 2f64.powi(-24),
        31 if m == 0.0 => f64::INFINITY,
        31 => f64::NAN,
        _ => (1024.0 + m) * 2f64.powi(e - 25),
    };
    (sign * a) as f32
}

/// Texel `k`'s channels as a fetch of format `pf` reads its bytes `b` (`f32` bits; a raw view
/// reads an sRGB format's codes as UNORM): a code `c / scale` correctly rounded (`-1` for a signed
/// format's least code), a float as it is, a missing channel `0` (`1` for alpha).
fn fetched(pf: PixelFormat, b: &[u8], k: usize) -> [u32; 4] {
    let unsigned = |c: u64, scale: u64| f32_bits(c, scale);
    let signed = |c: i64, scale: i64| {
        let c = c.max(-scale);
        let m = f32::from_bits(f32_bits(c.unsigned_abs(), scale as u64));
        if c < 0 { -m } else { m }.to_bits()
    };
    match pf {
        PixelFormat::R8Unorm => [unsigned(u64::from(b[k]), 255), 0, 0, 1f32.to_bits()],
        PixelFormat::RGBA8Unorm | PixelFormat::RGBA8UnormSRGB => {
            std::array::from_fn(|c| unsigned(u64::from(b[4 * k + c]), 255))
        }
        PixelFormat::RGBA8Snorm => {
            std::array::from_fn(|c| signed(i64::from(b[4 * k + c] as i8), 127))
        }
        PixelFormat::RGBA16Unorm | PixelFormat::RGBA16Snorm | PixelFormat::RGBA16Float => {
            std::array::from_fn(|c| {
                let h = u16::from_le_bytes([b[8 * k + 2 * c], b[8 * k + 2 * c + 1]]);
                match pf {
                    PixelFormat::RGBA16Unorm => unsigned(u64::from(h), 65535),
                    PixelFormat::RGBA16Snorm => signed(i64::from(h as i16), 32767),
                    _ => half_value(h).to_bits(),
                }
            })
        }
        PixelFormat::RGBA32Float => std::array::from_fn(|c| {
            let at = 16 * k + 4 * c;
            u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
        }),
        other => unreachable!("no probe of {other:?}"),
    }
}

/// The emulation answers Metal's own lanes of every format family, bit for bit, with `float` and
/// `half` results (`PROBE_LANES`, `probes`): an 8-bit sRGB format's through the raw view, under its
/// variant.
#[test]
fn a_cube_sample_answers_metals_probe_lanes() {
    let p = lavapipe();
    let lanes: Vec<[u32; 4]> = PROBE_LANES
        .iter()
        .map(|(_, d)| [d[0], d[1], d[2], 0])
        .collect();
    for half in [false, true] {
        let name = format!("probe half={half}");
        let Some(bc) = assemble(&name, &kernel(edge(), half, false)) else {
            return;
        };
        for probe in probes() {
            let srgb = probe.format == PixelFormat::RGBA8UnormSRGB;
            let (plan, m) = typed_module_with(&bc, "k", &p, &inputs(srgb))
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            let class = texel_format(probe.format).0;
            let imgs = images(&m, (1, 6), srgb, &|l, _, _| {
                fetched(probe.format, probe.texels, l as usize)
            });
            let bytes = if half { 8 } else { 16 };
            let out = run(&m, &resources(&plan, &lanes, bytes, class), &imgs, half);
            for (k, got) in out.iter().enumerate() {
                let want = match half {
                    true => probe.half[k].map(u32::from),
                    false => probe.float[k],
                };
                assert_eq!(
                    *got, want,
                    "{name} {:?} probe lane {}: {got:08x?} vs Metal's {want:08x?}",
                    probe.format, PROBE_LANES[k].0
                );
            }
        }
    }
}

// ---- Selection, interface, and size -------------------------------------------------------------

/// Whether artifact `a` emulates its one sample (`sampler.linear` and the cube's routines).
fn emulated(a: &FunctionArtifact) -> bool {
    let linear = a.strategies.entries.get("sampler.linear").map(|u| u.uses);
    let routines = a.strategies.entries.contains_key("sampler.linear.cube");
    assert_eq!(
        linear.is_some(),
        routines,
        "the census of {:?}",
        a.strategies
    );
    linear == Some(1)
}

/// On lavapipe a linear cube sample through a constexpr sampler without a mip filter is emulated
/// whatever its address modes and border, a cube array's too, and records its filter note and
/// `CubeFaceReciprocal`, declaring the slot's 2D-array view of its faces beside its combined
/// binding; a linear mip filter or a nearest sampler keeps the device's sample. On MoltenVK nothing
/// is emulated, and every sample requires `VkProperty::MetalSampler`.
#[test]
fn the_cube_emulation_reaches_what_metal_measured() {
    // (name, sampler, half, array, emulated on lavapipe, filtered)
    let cases = [
        ("edge", edge(), true, false, true, true),
        ("repeat", sampler_words(1, 0, 2, 0), true, false, true, true),
        ("mirror", sampler_words(1, 0, 3, 0), true, false, true, true),
        ("zero", sampler_words(1, 0, 0, 0), true, false, true, true),
        ("white", sampler_words(1, 0, 0, 2), true, false, true, true),
        ("array", edge(), true, true, true, true),
        ("float", edge(), false, false, true, true),
        ("float array", edge(), false, true, true, true),
        ("mip", sampler_words(1, 2, 1, 0), true, false, false, true),
        (
            "nearest",
            sampler_words(0, 0, 1, 0),
            true,
            false,
            false,
            false,
        ),
    ];
    let (lvp, m3) = (lavapipe(), m3());
    for (name, words, half, array, reached, filtered) in cases {
        let Some(bc) = assemble(name, &kernel(words, half, array)) else {
            return;
        };
        let note = match half {
            true => PrecisionNote::HalfSampleFilter,
            false => PrecisionNote::FloatSampleFilter,
        };
        let faces = RuntimeFact::ImageViewShape { slot: TEXTURE };
        let a = compile(&bc, "k", &lvp).unwrap_or_else(|e| panic!("{name} on lavapipe: {e}"));
        link_and_validate(&a, &lvp).unwrap_or_else(|e| panic!("{name} on lavapipe: {e}"));
        assert_eq!(emulated(&a), reached, "{name} on lavapipe");
        assert_eq!(a.precision.contains(&note), filtered, "{name} on lavapipe");
        let reciprocal = a.precision.contains(&PrecisionNote::CubeFaceReciprocal);
        // A nearest cube sample is snapped onto Metal's texel, under the same note.
        assert_eq!(reciprocal, reached || !filtered, "{name} on lavapipe");
        assert_eq!(a.facts.contains(&faces), reached, "{name} on lavapipe");
        assert!(
            !a.requirements
                .properties
                .contains(&VkProperty::MetalSampler),
            "{name} on lavapipe"
        );
        let (_, m) = typed_module_with(&bc, "k", &lvp, &inputs(false))
            .unwrap_or_else(|e| panic!("{name} on lavapipe: {e}"));
        let view = resource(&m, |k| *k == ResourceKind::Texture);
        let combined = resource(&m, |k| matches!(k, ResourceKind::CombinedTextureSampler(_)));
        assert_eq!(view.is_some(), reached, "{name}: the view of the faces");
        assert!(combined.is_some(), "{name}: the combined binding");
        let b = compile(&bc, "k", &m3).unwrap_or_else(|e| panic!("{name} on moltenvk-m3: {e}"));
        link_and_validate(&b, &m3).unwrap_or_else(|e| panic!("{name} on moltenvk-m3: {e}"));
        assert!(!emulated(&b), "{name} on moltenvk-m3");
        assert!(!b.precision.contains(&note), "{name} on moltenvk-m3");
        assert!(
            !b.precision.contains(&PrecisionNote::CubeFaceReciprocal),
            "{name} on moltenvk-m3"
        );
        assert!(!b.facts.contains(&faces), "{name} on moltenvk-m3");
        // A nearest one is Metal's own too, where lavapipe snaps it.
        assert!(
            b.requirements
                .properties
                .contains(&VkProperty::MetalSampler),
            "{name} on moltenvk-m3"
        );
    }
}

/// A packed normalized format's cube sample is not measured, so its emulation is not reached there:
/// bound to a `B5G6R5Unorm` or `RGB10A2Unorm` cube, a linear cube sample's specialized pipeline on
/// lavapipe keeps its filter note, where an `RGBA8Unorm` or a small float one (the half-float
/// family's emulation) does not.
#[test]
fn a_packed_normalized_cube_keeps_the_devices_filter() {
    use super::engine_linear_half::{device_of, notes_bound, notes_pipeline};
    use crate::metal::TextureType;
    let Some(bc) = assemble("cube notes", &kernel(edge(), true, false)) else {
        return;
    };
    let lvp = device_of("lavapipe");
    let p = notes_pipeline(&lvp, &bc);
    for (pf, noted) in [
        (PixelFormat::B5G6R5Unorm, true),
        (PixelFormat::RGB10A2Unorm, true),
        (PixelFormat::RGBA8Unorm, false),
        (PixelFormat::RG11B10Float, false),
    ] {
        let notes = notes_bound(&p, &lvp, (TextureType::Cube, pf), None);
        assert_eq!(
            notes.contains(&PrecisionNote::HalfSampleFilter),
            noted,
            "{pf:?}: {notes:?}"
        );
    }
}

/// The plan's word estimate bounds the emitted module of every cube form (`half` and `float`, a
/// cube array, an sRGB variant's raw view, a repeating sampler's wrap), which validates, on every
/// committed profile.
#[test]
fn the_cube_estimate_bounds_its_emission() {
    let mut modules = Vec::new();
    let repeat = sampler_words(1, 0, 2, 0);
    for (name, words, half, array, srgb) in [
        ("half", edge(), true, false, false),
        ("float", edge(), false, false, false),
        ("half array", edge(), true, true, false),
        ("float array", edge(), false, true, false),
        ("raw half", edge(), true, false, true),
        ("raw float array", edge(), false, true, true),
        ("repeat half", repeat, true, false, false),
        ("repeat raw float array", repeat, false, true, true),
    ] {
        let Some(bc) = assemble(name, &kernel(words, half, array)) else {
            return;
        };
        modules.push((name, bc, srgb));
    }
    // Depth cubes' comparisons.
    let less = edge() & !(0xf << 16) | 1 << 16;
    for (name, array) in [("compare", false), ("compare array", true)] {
        let Some(bc) = assemble(name, &compare_kernel(less, array)) else {
            return;
        };
        modules.push((name, bc, false));
    }
    let consts = FunctionConstantValues::defaults();
    for (name, bc, srgb) in &modules {
        for (stem, p) in committed_exact_profiles().expect("profiles") {
            let what = format!("{name} on {stem}");
            // A profile without cube arrays refuses a cube array's sample, whose combined binding
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
