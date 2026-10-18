//! Metal's linear filter for a sample with a `half` or `float` result, emulated by fetches on a
//! device that does not sample through Metal's own sampler, run on the CPU interpreter against
//! Metal's rule itself, bit for bit, for an 8- or 16-bit normalized format and a 32-bit float or
//! half-float format, in 1D and 2D: the footprint's 1/256 fraction rounded half up, the taps'
//! weights `(256 - k)` and `k` per axis; the codes' exact sum rounded half up onto the format's
//! grid and then to nearest even, to `f32` or `half`; Metal's linear accumulator for a float format
//! (each term truncated toward zero onto a grid 38 bits, or 17 for a half-float format, below the
//! largest present term's tap and weight exponents, summed exactly), its sum rounded with ties away
//! from zero, a 32-bit float format's to `f32` (a denormal `+0`), and for a `half` result then
//! toward zero saturating, a half-float format's to `half` (a zero `+0`), widened for a `float`
//! result.
//!
//! A float texture's texels include `-0`, at the texel a centre sample reads alone among them.
//!
//! The selection tests pin which samples the emulation reaches, which the device filters under
//! the notes (`HalfSampleFilter`, and `FloatSampleFilter` for every filtered `float` sample, a
//! depth comparison's included), and that a device sampling through Metal's own sampler keeps
//! every one and requires `VkProperty::MetalSampler`.
//!
//! `llvm-as` assembles the kernels; without it the tests skip visibly.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::device::profile::DeviceProfile;
use crate::device::requirements::VkProperty;
use crate::device::DevicePolicy;
use crate::lower::select::Plan;
use crate::metal::bicubic::BicubicTexels;
use crate::metal::sampler::SamplerWord;
use crate::metal::texel_format::texel_format;
use crate::metal::{PixelFormat, SamplerAddressMode, SamplerBorderColor};
use crate::pipeline::interface::PushLoc;
use crate::pipeline::sym::{
    Axis, PushField, ResourceKind, ResourceSlot, SlotClass, SlotRef, SpecKey, Stage,
};
use crate::pipeline::testing::EXACT_SAMPLING;
use crate::pipeline::testing::{
    compile, link_and_validate, llvm_assemble, profile, typed_module, AssembleError,
};
use crate::pipeline::PrecisionNote;
use crate::spirv::interp::{run_with_images, Buffer, Config, Image, Order, Resources};
use crate::spirv::ir::{GlobalKind, TypedModule};

mod levels;
mod packed;
mod ycbcr;

const HEADER: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n";

/// Bitcode of the module `body` (after the AIR header), or `None` without `llvm-as`.
fn assemble(name: &str, body: &str) -> Option<Vec<u8>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "m2v-linear-half-{}-{}-{}",
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

fn lavapipe() -> DeviceProfile {
    profile("lavapipe", EXACT_SAMPLING).expect("profile")
}

fn m3() -> DeviceProfile {
    profile("moltenvk-m3", EXACT_SAMPLING).expect("profile")
}

/// The words of `constexpr sampler(filter::linear, ...)`: minification and magnification
/// `filter` (AIR code: 0 nearest, 1 linear), mip filter `mip` (0 none, 2 linear), every axis
/// addressed by `address` (0 zero or border, 1 edge, 2 repeat, 3 mirrored repeat), `pixel`
/// coordinates, and border color code `border` (0 transparent black, 2 opaque white), with the
/// fields Apple's compiler writes for the rest (the `hlf_half_address` fixture's sampler).
fn sampler_words(filter: u64, mip: u64, address: u64, pixel: bool, border: u64) -> u64 {
    const BASE: u64 = 0x007b_ff00_0008_0000;
    BASE | address
        | address << 3
        | address << 6
        | filter << 9
        | filter << 11
        | mip << 13
        | u64::from(pixel) << 15
        | border << 56
}

fn linear(address: u64, pixel: bool, border: u64) -> u64 {
    sampler_words(1, 0, address, pixel, border)
}

/// How a test kernel samples its `half` texture.
#[derive(Clone, Copy, Debug)]
enum Form {
    /// `texture2d<half>`: `t.sample(s, coord[tid], offset)` with constant `offset`.
    Plain { offset: (i32, i32) },
    /// `texture2d_array<half>`: layer `layer`.
    Array { layer: i32 },
    /// `texture1d<half>`: `coord[tid].x`.
    OneD,
    /// `texture2d<half>` with `min_lod_clamp(1)`.
    MinLodClamp,
    /// `texture2d<half>` with lane `tid`'s dynamic `int2` offset (buffer 2).
    DynamicOffset,
    /// `texture3d<half>` at `float4` buffer 0's lane `tid`'s `xyz`, with constant `offset`.
    ThreeD { offset: (i32, i32, i32) },
    /// `texture2d<float>`: a `float` result.
    Float,
}

/// A kernel sampling texture slot 0 through sampler `sampler` (a constexpr sampler's words, or
/// `None` for the runtime `[[sampler(0)]]`) at `float2` buffer 0's lane `tid`, storing the
/// `half4` (`float4` for [`Form::Float`]) to buffer 1.
fn kernel(sampler: Option<u64>, form: Form) -> String {
    kernel_of(sampler, form, matches!(form, Form::Float))
}

/// [`kernel`] with a `float` result when `float`, whatever the form.
fn kernel_of(sampler: Option<u64>, form: Form, float: bool) -> String {
    let half = !float;
    let (texel, vec4, size) = if half {
        ("half", "<4 x half>", 8)
    } else {
        ("float", "<4 x float>", 16)
    };
    let suffix = if half { "v4f16" } else { "v4f32" };
    let s = if sampler.is_some() {
        "@__air_sampler_state"
    } else {
        "%s"
    };
    let (tex_ty, name, args, call_args) = match form {
        Form::Plain { offset: (ox, oy) } => (
            format!("texture2d<{texel}, sample>"),
            "2d",
            "ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32",
            format!("<2 x float> %uv, i1 true, <2 x i32> <i32 {ox}, i32 {oy}>, i1 false, float 0.0, float 0.0, i32 0"),
        ),
        Form::Float => (
            format!("texture2d<{texel}, sample>"),
            "2d",
            "ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32",
            "<2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.0, float 0.0, i32 0".to_string(),
        ),
        Form::Array { layer } => (
            format!("texture2d_array<{texel}, sample>"),
            "2d_array",
            "ptr addrspace(1), ptr addrspace(2), <2 x float>, i32, i1, <2 x i32>, i1, float, float, i32",
            format!("<2 x float> %uv, i32 {layer}, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.0, float 0.0, i32 0"),
        ),
        Form::OneD => (
            format!("texture1d<{texel}, sample>"),
            "1d",
            "ptr addrspace(1), ptr addrspace(2), float, i1, i32, i1, float, float, i32",
            "float %u, i1 false, i32 0, i1 false, float 0.0, float 0.0, i32 0".to_string(),
        ),
        Form::MinLodClamp => (
            format!("texture2d<{texel}, sample>"),
            "2d",
            "ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32",
            "<2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.0, float 1.0, i32 0".to_string(),
        ),
        Form::DynamicOffset => (
            format!("texture2d<{texel}, sample>"),
            "2d",
            "ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32",
            "<2 x float> %uv, i1 true, <2 x i32> %o, i1 false, float 0.0, float 0.0, i32 0".to_string(),
        ),
        Form::ThreeD { offset: (ox, oy, oz) } => (
            format!("texture3d<{texel}, sample>"),
            "3d",
            "ptr addrspace(1), ptr addrspace(2), <3 x float>, i1, <3 x i32>, i1, float, float, i32",
            format!("<3 x float> %uvw, i1 true, <3 x i32> <i32 {ox}, i32 {oy}, i32 {oz}>, i1 false, float 0.0, float 0.0, i32 0"),
        ),
    };
    let dynamic = matches!(form, Form::DynamicOffset);
    let runtime = sampler.is_none();
    // Parameters after `out`: the offsets buffer, then the runtime sampler.
    let mut params = String::new();
    let mut mds = Vec::new();
    let mut next = 3;
    if dynamic {
        params.push_str(", ptr addrspace(1) %offs");
        mds.push(format!("!{{i32 {next}, !\"air.buffer\", !\"air.location_index\", i32 2, i32 1, !\"air.read\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 8, !\"air.arg_type_align_size\", i32 8, !\"air.arg_type_name\", !\"int2\", !\"air.arg_name\", !\"offs\"}}"));
        next += 1;
    }
    if runtime {
        params.push_str(", ptr addrspace(2) %s");
        mds.push(format!("!{{i32 {next}, !\"air.sampler\", !\"air.location_index\", i32 0, i32 1, !\"air.arg_type_name\", !\"sampler\", !\"air.arg_name\", !\"s\"}}"));
        next += 1;
    }
    mds.push(format!("!{{i32 {next}, !\"air.thread_position_in_grid\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"tid\"}}"));
    let first = 7;
    let arg_mds: Vec<String> = (0..mds.len()).map(|i| format!("!{}", first + i)).collect();
    let tail: String = mds
        .iter()
        .enumerate()
        .map(|(i, m)| format!("!{} = {m}\n", first + i))
        .collect();
    let load_offset = if dynamic {
        "  %po = getelementptr inbounds <2 x i32>, ptr addrspace(1) %offs, i64 %i\n  %o = load <2 x i32>, ptr addrspace(1) %po, align 8\n"
    } else {
        ""
    };
    let state = match sampler {
        Some(w) => format!(
            "@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 {w}, i64 0], align 8\n\n"
        ),
        None => String::new(),
    };
    let (states, state_md) = if sampler.is_some() {
        (
            "!air.sampler_states = !{!6}\n",
            "!6 = !{!\"air.sampler_state\", ptr addrspace(2) @__air_sampler_state}\n",
        )
    } else {
        ("", "")
    };
    let half_name = if half { "half4" } else { "float4" };
    // A 3D sample's coordinates are a `float4`'s `xyz`, every other's a `float2`.
    let (load_coord, coord_size, coord_name) = if matches!(form, Form::ThreeD { .. }) {
        (
            "  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %c, i64 %i\n  %c4 = load <4 x float>, ptr addrspace(1) %p, align 16\n  %uvw = shufflevector <4 x float> %c4, <4 x float> zeroinitializer, <3 x i32> <i32 0, i32 1, i32 2>\n",
            16,
            "float4",
        )
    } else {
        (
            "  %p = getelementptr inbounds <2 x float>, ptr addrspace(1) %c, i64 %i\n  %uv = load <2 x float>, ptr addrspace(1) %p, align 8\n  %u = extractelement <2 x float> %uv, i64 0\n",
            8,
            "float2",
        )
    };
    format!(
        r#"{state}define void @k(ptr addrspace(1) %t, ptr addrspace(1) %c, ptr addrspace(1) %out{params}, i32 %tid) {{
  %i = zext i32 %tid to i64
{load_coord}{load_offset}  %r = call {{ {vec4}, i8 }} @air.sample_texture_{name}.{suffix}(ptr addrspace(1) %t, ptr addrspace(2) {s}, {call_args})
  %v = extractvalue {{ {vec4}, i8 }} %r, 0
  %q = getelementptr inbounds {vec4}, ptr addrspace(1) %out, i64 %i
  store {vec4} %v, ptr addrspace(1) %q, align {size}
  ret void
}}

declare {{ {vec4}, i8 }} @air.sample_texture_{name}.{suffix}({args})

!air.kernel = !{{!0}}
{states}!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4, !5, {arg_mds}}}
!3 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"{tex_ty}", !"air.arg_name", !"t"}}
!4 = !{{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 {coord_size}, !"air.arg_type_align_size", i32 {coord_size}, !"air.arg_type_name", !"{coord_name}", !"air.arg_name", !"c"}}
!5 = !{{i32 2, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 {size}, !"air.arg_type_align_size", i32 {size}, !"air.arg_type_name", !"{half_name}", !"air.arg_name", !"out"}}
{state_md}{tail}"#,
        arg_mds = arg_mds.join(", "),
    )
}

/// The combined image-sampler bindings a module declares, and whether each compares.
fn combined_slots(m: &TypedModule) -> Vec<(ResourceSlot, bool)> {
    m.globals()
        .iter()
        .filter_map(|g| match g.kind {
            GlobalKind::Resource(
                s @ ResourceSlot {
                    kind: ResourceKind::CombinedTextureSampler(c),
                    ..
                },
            ) => Some((s, c.compare)),
            _ => None,
        })
        .collect()
}

/// The combined image-sampler binding a module declares for texture slot 0.
fn combined_slot(m: &TypedModule) -> ResourceSlot {
    m.globals()
        .iter()
        .find_map(|g| match g.kind {
            GlobalKind::Resource(
                s @ ResourceSlot {
                    kind: ResourceKind::CombinedTextureSampler(_),
                    ..
                },
            ) => Some(s),
            _ => None,
        })
        .expect("a combined binding")
}

const TEXTURE: SlotRef = SlotRef {
    stage: Stage::Kernel,
    class: SlotClass::Texture,
    index: 0,
};

/// Resources of a dispatch of one-thread threadgroups: `[[buffer(i)]]` holding `init[i]`, and
/// the texture slot's `TexelFormat` specialization `class`.
fn resources(plan: &Plan, init: &[Vec<u8>], class: u32) -> Resources {
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
    let buffers = init
        .iter()
        .enumerate()
        .map(|(i, b)| Buffer {
            slot: Some(ResourceSlot {
                stage: Stage::Kernel,
                kind: ResourceKind::Buffer,
                index: i as u32,
            }),
            address: None,
            bytes: b.clone(),
        })
        .collect();
    Resources {
        buffers,
        push,
        spec,
    }
}

/// `res` with each `SamplerState` specialization artifact `a` declares valued by runtime sampler
/// word `word`, masked by its fields; unchanged without a word.
fn with_word(
    mut res: Resources,
    a: &crate::library::FunctionArtifact,
    word: Option<u32>,
) -> Resources {
    let Some(word) = word else {
        return res;
    };
    let mut declared = 0;
    for key in a.interface.read_keys() {
        if let SpecKey::SamplerState { fields, .. } = key {
            res.spec.insert(key, u64::from(word & fields));
            declared += 1;
        }
    }
    assert_eq!(
        declared, 1,
        "the runtime sampler's `SamplerState` specialization"
    );
    res
}

/// The word of a runtime sampler with linear minification and magnification, no mip filter, address
/// mode `mode` on every axis, pixel coordinates when `pixel`, and border color `border`.
fn runtime_word(mode: SamplerAddressMode, pixel: bool, border: SamplerBorderColor) -> u32 {
    let mut w = SamplerWord::LINEAR | (border.as_raw() << SamplerWord::BORDER_SHIFT);
    if pixel {
        w |= SamplerWord::PIXEL_COORDINATES;
    }
    for shift in SamplerWord::ADDRESS_SHIFT {
        w |= mode.as_raw() << shift;
    }
    w
}

/// `m` over `groups` one-thread threadgroups with `image` bound, in both scheduling orders, which
/// must agree: buffer 1's `half4`s (`float4`s unless `half`), as bits.
fn run(m: &TypedModule, res: &Resources, image: &Image, groups: u32, half: bool) -> Vec<[u32; 4]> {
    run_images(m, res, std::slice::from_ref(image), groups, half)
}

/// [`run`] with `images` bound.
fn run_images(
    m: &TypedModule,
    res: &Resources,
    images: &[Image],
    groups: u32,
    half: bool,
) -> Vec<[u32; 4]> {
    let mut outs = Vec::new();
    for order in [Order::Forward, Order::Reverse] {
        let cfg = Config {
            workgroups: [groups, 1, 1],
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

/// A deterministic source: a 64-bit LCG.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 11
    }
}

/// The family of a format's treatment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Family {
    /// A normalized format: its code scale `2^b - 1` (`2^(b-1) - 1` signed), signedness, and grid
    /// factor (16 for 8 bits, else 1).
    Norm {
        scale: i64,
        signed: bool,
        g: i64,
    },
    Float32,
    Float16,
}

fn family(class: u32) -> Family {
    let has = |b: u32| class & b != 0;
    if has(BicubicTexels::LINEAR) {
        return Family::Float32;
    }
    if has(BicubicTexels::HALF) {
        return Family::Float16;
    }
    assert!(has(BicubicTexels::EXACT), "a measured class");
    let signed = has(BicubicTexels::SNORM);
    let wide = has(BicubicTexels::WIDE16);
    let scale = match (wide, signed) {
        (false, false) => 255,
        (false, true) => 127,
        (true, false) => 65535,
        (true, true) => 32767,
    };
    Family::Norm {
        scale,
        signed,
        g: if wide { 1 } else { 16 },
    }
}

/// A texel as the bound format stores it, per channel: a normalized code, or a float value (a
/// half-float format's a `half`).
#[derive(Clone, Copy, Debug)]
pub(super) enum Stored {
    Code(i64),
    Value(f32),
}

/// A texture's stored texels (one layer, `w` by `h`), with the format's missing channels as a
/// fetch reads them (0 for red, green, blue; 1 for alpha).
pub(super) struct Texels {
    class: u32,
    w: i64,
    h: i64,
    pub(super) t: Vec<[Stored; 4]>,
}

impl Texels {
    pub(super) fn random(class: u32, w: i64, h: i64, seed: u64) -> Texels {
        let mut rng = Lcg(seed);
        let fam = family(class);
        let present = |c: u32| class & (1 << (BicubicTexels::PRESENT + c)) != 0;
        let t = (0..w * h)
            .map(|_| {
                std::array::from_fn(|c| {
                    let missing = if c == 3 { 1.0 } else { 0.0 };
                    match fam {
                        Family::Norm { scale, signed, .. } if present(c as u32) => {
                            let lo = if signed { -scale - 1 } else { 0 };
                            Stored::Code(lo + (rng.next() % (scale - lo + 1) as u64) as i64)
                        }
                        Family::Norm { scale, .. } => Stored::Code(if c == 3 { scale } else { 0 }),
                        // Values with at most 12 significant bits in `[-2, 2)` (one in eight
                        // scaled by `2^-10`, one in sixteen `-0`), so every reference sum below
                        // is exact in `f64`.
                        Family::Float32 | Family::Float16 if present(c as u32) => {
                            let m = (rng.next() % 4096) as f64 - 2048.0;
                            let e = if rng.next() & 7 == 0 { -20 } else { -10 };
                            if rng.next() & 15 == 0 {
                                Stored::Value(-0.0)
                            } else {
                                Stored::Value((m * 2f64.powi(e)) as f32)
                            }
                        }
                        _ => Stored::Value(missing),
                    }
                })
            })
            .collect::<Vec<[Stored; 4]>>();
        // `-0` in the first three texels of rows 0 and 2 (a layer reads the texel as many places
        // on), which the samples at `(0.5, 2.5)` and `0.5` read alone (`coordinates`).
        let mut t = t;
        if matches!(fam, Family::Float32 | Family::Float16) {
            for row in [0, 2.min(h - 1)] {
                for x in 0..3 {
                    for (c, v) in t[(row * w + x) as usize].iter_mut().enumerate() {
                        if present(c as u32) {
                            *v = Stored::Value(-0.0);
                        }
                    }
                }
            }
        }
        Texels { class, w, h, t }
    }

    /// A stored channel's value as the device fetches it: `c / scale` correctly rounded to
    /// `f32` (`-1` for the least signed code), or the float itself.
    pub(super) fn fetched(&self, s: Stored) -> f32 {
        match (s, family(self.class)) {
            (Stored::Code(c), Family::Norm { scale, .. }) => {
                (c.max(-scale) as f64 / scale as f64) as f32
            }
            (Stored::Value(v), _) => v,
            (Stored::Code(_), _) => unreachable!("a code of a float format"),
        }
    }

    /// The interpreter image of these texels in slot `slot`, as `layers` copies (layer `k`'s
    /// codes or values replaced by layer 0's of the texel `k` places on, so a wrong layer shows).
    fn image(&self, slot: ResourceSlot, dims: u32, layers: u32) -> Image {
        let h = if dims == 1 { 1 } else { self.h as u32 };
        let mut image = Image::new(slot, [self.w as u32, h, 1], layers, 1, 1);
        for l in 0..layers {
            for y in 0..h {
                for x in 0..self.w as u32 {
                    let at = image.index(0, l, 0, [x, y, 0]).expect("texel");
                    let v = self.texel(l, x as i64, y as i64);
                    image.texels[at] = v.map(|c| self.fetched(c).to_bits());
                }
            }
        }
        image
    }

    /// Layer `l`'s stored texel at `(x, y)` within the level.
    fn texel(&self, l: u32, x: i64, y: i64) -> [Stored; 4] {
        let n = self.w * self.h;
        self.t[((y * self.w + x + i64::from(l)) % n) as usize]
    }
}

/// How a tap outside the level is addressed.
#[derive(Clone, Copy, Debug)]
enum Address {
    Edge,
    Repeat,
    Mirror,
    /// The border color's channels.
    Border([f64; 4]),
}

/// A tap's channel value: a stored channel, or the border's.
#[derive(Clone, Copy, Debug)]
pub(super) enum Tap {
    Stored(Stored),
    Border(f64),
}

/// The tap of layer `l` at `(i, j)` (`j` 0 for a 1D texture), channel `c`, under `address`.
fn tap(tex: &Texels, address: Address, l: u32, (i, j): (i64, i64), c: usize) -> Tap {
    let mirror = |v: i64, n: i64| {
        let m = if v >= 0 { v } else { -1 - v };
        let r = m % (2 * n);
        if r < n {
            r
        } else {
            2 * n - 1 - r
        }
    };
    let fix = |v: i64, n: i64| match address {
        Address::Edge | Address::Border(_) => v.clamp(0, n - 1),
        Address::Repeat => v.rem_euclid(n),
        Address::Mirror => mirror(v, n),
    };
    if let Address::Border(b) = address {
        if i < 0 || i >= tex.w || j < 0 || j >= tex.h {
            let present = tex.class & (1 << (BicubicTexels::PRESENT + c as u32)) != 0;
            let missing = if c == 3 { 1.0 } else { 0.0 };
            return Tap::Border(if present { b[c] } else { missing });
        }
    }
    Tap::Stored(tex.texel(l, fix(i, tex.w), fix(j, tex.h))[c])
}

/// Metal's split of axis coordinate `x` (texels): the base tap and the fraction's 1/256 steps.
fn split(x: f64) -> (i64, i64) {
    let q = ((x - 0.5) * 256.0 + 0.5).floor() as i64;
    (q >> 8, q & 255)
}

/// The `half` bits of `v`, which must be a `half` value (or an infinity).
fn half_bits_of(v: f64) -> u16 {
    let sign = if v.is_sign_negative() { 0x8000 } else { 0 };
    let a = v.abs();
    if a.is_infinite() {
        return sign | 0x7C00;
    }
    if a < 2f64.powi(-14) {
        let m = a / 2f64.powi(-24);
        assert_eq!(m.fract(), 0.0, "{v} is no half");
        return sign | m as u16;
    }
    let e = a.log2().floor() as i32;
    let e = if 2f64.powi(e) > a { e - 1 } else { e };
    let e = if 2f64.powi(e + 1) <= a { e + 1 } else { e };
    let m = a / 2f64.powi(e - 10);
    assert_eq!(m.fract(), 0.0, "{v} is no half");
    sign | ((e + 15) as u16) << 10 | (m as u16 - 1024)
}

/// The spacing of `half` values at magnitude `a`.
fn half_ulp(a: f64) -> f64 {
    if a < 2f64.powi(-14) {
        return 2f64.powi(-24);
    }
    let e = a.log2().floor() as i32;
    let e = if 2f64.powi(e) > a { e - 1 } else { e };
    let e = if 2f64.powi(e + 1) <= a { e + 1 } else { e };
    2f64.powi(e - 10)
}

#[derive(Clone, Copy)]
pub(super) enum Rounding {
    NearestEven,
    NearestAway,
    TowardZeroSaturating,
}

/// The `half` bits of exact value `v` (exact in `f64`) rounded by `mode`.
pub(super) fn to_half(v: f64, mode: Rounding) -> u16 {
    if v == 0.0 {
        return if v.is_sign_negative() { 0x8000 } else { 0 };
    }
    let a = v.abs();
    let q = half_ulp(a);
    let n = a / q;
    let (fl, fr) = (n.floor(), n - n.floor());
    let m = match mode {
        Rounding::TowardZeroSaturating => fl,
        Rounding::NearestAway => {
            if fr >= 0.5 {
                fl + 1.0
            } else {
                fl
            }
        }
        Rounding::NearestEven => {
            if fr > 0.5 || (fr == 0.5 && fl % 2.0 == 1.0) {
                fl + 1.0
            } else {
                fl
            }
        }
    };
    let r = m * q;
    let r = match mode {
        Rounding::TowardZeroSaturating => r.min(65504.0),
        _ if r > 65504.0 => f64::INFINITY,
        _ => r,
    };
    half_bits_of(r.copysign(v))
}

/// `x` (exact in `f64`) rounded to `f32` with ties away from zero, a denormal result `+0`.
fn f32_away(x: f64) -> f32 {
    let r = x as f32;
    let d = x - f64::from(r);
    let r = if d == 0.0 {
        r
    } else {
        let other = if d > 0.0 {
            f32::from_bits(if r >= 0.0 {
                r.to_bits() + 1
            } else {
                r.to_bits() - 1
            })
        } else {
            f32::from_bits(if r > 0.0 {
                r.to_bits() - 1
            } else {
                r.to_bits() + 1
            })
        };
        if (f64::from(other) - x).abs() == d.abs() && other.abs() > r.abs() {
            other
        } else {
            r
        }
    };
    if r != 0.0 && r.abs() < f32::MIN_POSITIVE {
        0.0
    } else {
        r
    }
}

/// The `f32` value of `half` bits `h`.
fn half_value(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let (e, m) = (i32::from(h >> 10 & 0x1F), f64::from(h & 0x3FF));
    let a = match e {
        0 => m * 2f64.powi(-24),
        31 if m == 0.0 => f64::INFINITY,
        31 => f64::NAN,
        _ => (1024.0 + m) * 2f64.powi(e - 25),
    };
    (sign * a) as f32
}

/// `n / d` (`|n| <= d`, `d < 2^16`) correctly rounded to `f32` (nearest even), in integers.
fn ratio_f32(n: i64, d: i64) -> f32 {
    if n == 0 {
        return 0.0;
    }
    let (a, d) = (u128::from(n.unsigned_abs()), d as u128);
    // `a / d` lies in `[2^e, 2^(e+1))`.
    let mut e = 0i32;
    while a << ((-e) as u32) < d {
        e -= 1;
    }
    let num = a << (23 - e) as u32;
    let (q, r) = (num / d, num % d);
    let up = 2 * r > d || (2 * r == d && q & 1 == 1);
    let m = (q + u128::from(up)) as f64 * 2f64.powi(e - 23);
    (m.copysign(n as f64)) as f32
}

/// A half-float format's `half` result `h`, `+0` where it is a zero of either sign (Metal's,
/// measured on the P2a probe).
fn unsigned_half(h: u16) -> u16 {
    if h & 0x7FFF == 0 {
        0
    } else {
        h
    }
}

/// The result bits of `half` bits `h`: themselves for a `half` result, else the value's `f32`.
fn widened(h: u16, half: bool) -> u32 {
    if half {
        u32::from(h)
    } else {
        half_value(h).to_bits()
    }
}

/// Metal's result for the taps `taps` with integer weights `w` (summing to `total`) of a format of
/// family `fam`, read exactly, as `half` bits when `half`, else `f32` bits: a normalized format's
/// codes summed exactly and rounded half up onto its grid, then to nearest even; a float format's
/// taps through Metal's linear accumulator ([`accumulated`]), a 32-bit float sum rounded to `f32`
/// with ties away from zero, then for a `half` result toward zero saturating, a half-float sum to
/// `half` with ties away from zero (a zero `+0`), which a `float` result widens.
fn metal(tex: &Texels, taps: &[(Tap, i64)], total: i64, half: bool) -> u32 {
    match family(tex.class) {
        Family::Norm { scale, signed, g } => {
            let d = g * scale;
            let s: i64 = taps
                .iter()
                .map(|(t, w)| {
                    let n = match t {
                        Tap::Stored(Stored::Code(c)) => c.max(&-scale) * g,
                        Tap::Border(b) => (b * d as f64) as i64,
                        Tap::Stored(Stored::Value(_)) => unreachable!("a float tap"),
                    };
                    n * w
                })
                .sum();
            let lo = if signed { -d } else { 0 };
            let n = (s + total / 2).div_euclid(total).clamp(lo, d);
            if n == 0 {
                0
            } else if half {
                u32::from(to_half(n as f64 / d as f64, Rounding::NearestEven))
            } else {
                ratio_f32(n, d).to_bits()
            }
        }
        fam => {
            let terms: Vec<(f64, f64)> = taps
                .iter()
                .map(|(t, w)| {
                    let v = match t {
                        Tap::Stored(Stored::Value(v)) => f64::from(*v),
                        Tap::Border(b) => *b,
                        Tap::Stored(Stored::Code(_)) => unreachable!("a code tap"),
                    };
                    (v, *w as f64 / total as f64)
                })
                .collect();
            let bits = if fam == Family::Float16 { 17 } else { 38 };
            // Zero is `+0`, whatever the taps' signs.
            let s = accumulated(&terms, bits);
            let s = if s == 0.0 { 0.0 } else { s };
            match (fam, half) {
                (Family::Float32, true) => u32::from(to_half(
                    f64::from(f32_away(s)),
                    Rounding::TowardZeroSaturating,
                )),
                (Family::Float32, false) => f32_away(s).to_bits(),
                _ => widened(unsigned_half(to_half(s, Rounding::NearestAway)), half),
            }
        }
    }
}

/// Metal's linear accumulator over the terms `terms` (each a tap's value and its weight, exact in
/// `f64`): each present term (a nonzero tap of nonzero weight) `v · w` truncated toward zero onto
/// the grid `2^(E - bits)`, `E` the largest present `e(v) + e(w)` (`e` the exponent), and the
/// truncated terms summed. Exact in `f64`: the texels carry at most 12 significant bits and the
/// weights 16, and each truncated term is an integer below `2^(bits + 2)` grid units.
fn accumulated(terms: &[(f64, f64)], bits: i32) -> f64 {
    let exp = |x: f64| ((x.abs().to_bits() >> 52) as i32) - 1023;
    let present: Vec<&(f64, f64)> = terms
        .iter()
        .filter(|(v, w)| *v != 0.0 && *w != 0.0)
        .collect();
    let Some(top) = present.iter().map(|(v, w)| exp(*v) + exp(*w)).max() else {
        return 0.0;
    };
    let unit = 2f64.powi(top - bits);
    present
        .iter()
        .map(|(v, w)| ((v * w) / unit).trunc() * unit)
        .sum()
}

/// How the 2D reference reads a 16-bit normalized format's sum: Metal's, or with one rule dropped,
/// to show a lane depends on it.
#[derive(Clone, Copy, PartialEq, Eq)]
enum PlaneRule {
    Metal,
    /// The exact sum, the terms untruncated (the earlier reading).
    Exact,
    /// An axis whose two taps are one texel keeps its fraction.
    Unmerged,
}

/// The expected bits of channel `c` of a sample at texel-space `(x, y)` (`y` ignored in 1D) of
/// layer `l`, constant offset `offset`, read by `rule` in 2D: a `half`'s when `half`, else an
/// `f32`'s. A 2D sample's axis whose two taps are one texel (or both outside a border axis)
/// weighs them `(256, 0)`, the taps weigh `wx · wy` in units of `2^-16`, and a normalized format's
/// terms are truncated onto its grid as a 3D sample's are ([`metal_terms`], a 16-bit format's onto
/// `2^-8` of its codes; an 8-bit format's `16 · code` terms lose nothing).
fn expected(
    tex: &Texels,
    address: Address,
    (x, y): (f64, f64),
    (one_d, half, rule): (bool, bool, PlaneRule),
    offset: (i32, i32),
    l: u32,
    c: usize,
) -> u32 {
    let (i, kx) = split(x);
    let i = i + i64::from(offset.0);
    if one_d {
        let t0 = tap(tex, address, l, (i, 0), c);
        let t1 = tap(tex, address, l, (i + 1, 0), c);
        return metal(tex, &[(t0, 256 - kx), (t1, kx)], 256, half);
    }
    let (j, ky) = split(y);
    let j = j + i64::from(offset.1);
    let t = [
        [
            tap(tex, address, l, (i, j), c),
            tap(tex, address, l, (i + 1, j), c),
        ],
        [
            tap(tex, address, l, (i, j + 1), c),
            tap(tex, address, l, (i + 1, j + 1), c),
        ],
    ];
    // A float format's sum merges such an axis too, through the accumulator.
    let merged = |i: i64, n: i64, k: i64| {
        let one = volume_index(i, n, address) == volume_index(i + 1, n, address);
        if one && rule == PlaneRule::Metal {
            0
        } else {
            k
        }
    };
    let (kx, ky) = (merged(i, tex.w, kx), merged(j, tex.h, ky));
    let w = |a: i64, b: i64| a * b;
    let taps = [
        (t[0][0], w(256 - kx, 256 - ky)),
        (t[0][1], w(kx, 256 - ky)),
        (t[1][0], w(256 - kx, ky)),
        (t[1][1], w(kx, ky)),
    ];
    let s: fn(i64) -> u32 = match rule {
        PlaneRule::Exact => |_| 0,
        _ => |g| if g == 16 { 4 } else { 8 },
    };
    metal_terms(tex, &taps, half, s)
}

/// Coordinates (texels) from outside one edge to outside the other on both axes, with fractions
/// on and off Metal's 1/256 grid (halfway between two steps included), and far outside the
/// level, where repeat and mirror wrap several periods. Four sit within a few ulps of a
/// footprint's step several periods out (at a width of 12): there the coordinate wrapped first and
/// the product of the unwrapped one read different taps, under repeat and mirror alike. One `y`
/// lies a ulp of its `f32` coordinate below a footprint's step one period below the level (at a
/// height of 8, a power of two): there the wrap rounded up reads the step's other side, under
/// repeat and mirror alike, and Metal's, exact, does not.
fn coordinates(w: i64, h: i64) -> Vec<(f64, f64)> {
    let xs = [
        -34.501954078674316,
        -21.501953601837158,
        64.49804306030273,
        71.49804496765137,
        -25.3,
        -2.25,
        -0.75,
        0.0,
        0.5,
        1.0 + 1.0 / 512.0,
        2.0 + 3.0 / 512.0,
        3.3,
        w as f64 / 2.0 + 0.123,
        w as f64 - 0.75,
        w as f64 + 0.25,
        w as f64 + 2.25,
        3.0 * w as f64 + 0.7,
    ];
    let ys = [
        -1.5,
        0.25,
        2.5,
        1.75,
        h as f64 / 2.0 + 0.6,
        h as f64 - 0.1,
        h as f64 + 1.5,
        -2.0 * h as f64 - 0.3,
        // `u = (4.5 - 1/512) / 8 - 1` less a ulp (`2^-25`): `u + 1` (or `u + 2`) needs a bit more
        // than an `f32` holds, and the position lies `2^-22` below the step `4.5 - 1/512` (or
        // `12.5 - 1/512` mirrored).
        -3.501953125 - 2f64.powi(-22),
    ];
    ys.iter()
        .flat_map(|y| xs.iter().map(move |x| (*x, *y)))
        .collect()
}

/// The exact real `x` (an `f64` holding it exactly) rounded to `f32` with ties away from zero.
fn ties_away(x: f64) -> f32 {
    let r = x as f32;
    let d = x - f64::from(r);
    if d == 0.0 || !r.is_finite() || r == 0.0 {
        return r;
    }
    let other = if (d > 0.0) == (r > 0.0) {
        f32::from_bits(r.to_bits() + 1)
    } else {
        f32::from_bits(r.to_bits() - 1)
    };
    if (f64::from(other) - x).abs() == d.abs() && other.abs() > r.abs() {
        other
    } else {
        r
    }
}

/// The texel-space coordinate a kernel reads for `x` (texels) under `scale` (the extent, or 1
/// for pixel coordinates, `pixel`) on an axis addressed by `address`: Metal's position, the `f32`
/// coordinate it is given wrapped into one period first on a repeating or mirrored axis (rounded
/// up, or down, which reads the exact wrap's footprints, where the extent is a power of two, item
/// 19), then times the extent rounded to `f32` with ties away from zero; a pixel coordinate as
/// given.
fn seen(x: f64, scale: f64, address: Address, pixel: bool) -> f64 {
    seen_rounded(x, scale, address, pixel, !(scale as u32).is_power_of_two())
}

/// [`seen`] with the wrap rounded up (`up`) or down on every extent.
fn seen_rounded(x: f64, scale: f64, address: Address, pixel: bool, up: bool) -> f64 {
    let u = (x / scale) as f32;
    if pixel {
        return f64::from(u);
    }
    let u = match address {
        Address::Repeat => wrap(u, u.floor(), up),
        Address::Mirror => {
            let fl = (u * 0.5).floor();
            wrap(u, fl + fl, up)
        }
        Address::Edge | Address::Border(_) => u,
    };
    f64::from(ties_away(f64::from(u) * scale))
}

/// `u - k` (`k` the integer start of `u`'s period) rounded toward positive infinity (`up`) or
/// negative infinity, as Metal's wrap rounds. Exact in `f64` but where `|u| < 2^-29`, where it lies
/// just below `-k` and rounds to `-k` (to nearest) either way, so `s` is a neighbour of the exact
/// difference, which `s + k` (exact in `f64`) against `u` places on one side of it or on it.
fn wrap(u: f32, k: f32, up: bool) -> f32 {
    let s = (f64::from(u) - f64::from(k)) as f32;
    let z = f64::from(s) + f64::from(k);
    if up && z < f64::from(u) {
        f32::from_bits(s.to_bits() + 1)
    } else if !up && z > f64::from(u) {
        f32::from_bits(s.to_bits() - 1)
    } else {
        s
    }
}

/// The texel-space coordinate the emulation read previously for `x` (texels) under `scale`: the
/// `f32` coordinate times the extent, rounded to `f32` to nearest even, unwrapped.
fn product(x: f64, scale: f64) -> f64 {
    f64::from(((x / scale) as f32) * scale as f32)
}

fn coord_bytes(points: &[(f64, f64)], scale: (f64, f64)) -> Vec<u8> {
    points
        .iter()
        .flat_map(|(x, y)| {
            [
                ((x / scale.0) as f32).to_le_bytes(),
                ((y / scale.1) as f32).to_le_bytes(),
            ]
        })
        .flatten()
        .collect()
}

const FORMATS: [PixelFormat; 8] = [
    PixelFormat::RGBA8Unorm,
    PixelFormat::R8Unorm,
    PixelFormat::RGBA8Snorm,
    PixelFormat::RGBA16Unorm,
    PixelFormat::RG16Snorm,
    PixelFormat::RGBA32Float,
    PixelFormat::R32Float,
    PixelFormat::RGBA16Float,
];

/// Run the kernel `bc` (sampling through an `address` sampler with `pixel` coordinates, in
/// `form`, a `half` result) on a `w` by `h` texture of every format of [`FORMATS`] on lavapipe,
/// against [`expected`].
fn check(name: &str, bc: &[u8], address: Address, pixel: bool, form: Form) {
    check_result(name, bc, address, pixel, form, true);
}

/// [`check`] of a kernel with a `half` result when `half`, else a `float` one; returns how many
/// channels of a 16-bit normalized format each dropped 2D rule changes (the exact sum, the unmerged
/// fractions).
fn check_result(
    name: &str,
    bc: &[u8],
    address: Address,
    pixel: bool,
    form: Form,
    half: bool,
) -> (usize, usize) {
    check_with(name, bc, (address, pixel, None), form, half)
}

/// [`check_result`] of a kernel sampling through runtime sampler slot 0 whose bound sampler's
/// word is `word` (its `SamplerState` specialization), or through its constexpr sampler.
fn check_with(
    name: &str,
    bc: &[u8],
    (address, pixel, word): (Address, bool, Option<u32>),
    form: Form,
    half: bool,
) -> (usize, usize) {
    check_sized(name, bc, (address, pixel, word), form, half, (12, 0)).0
}

/// The offset Metal moves an axis's taps by (`position::offset_dropped`): none on a repeating or
/// mirrored axis whose base-level size `n` is 3, 5, 6 or 7, else `o`, added before the wrap as
/// Vulkan adds it.
fn placed(o: i32, n: i64, address: Address) -> i32 {
    placed_under(o, n, address, false)
}

/// [`placed`] on an axis of a view whose parent's base-level size, which Metal's rule reads, is not
/// a power of two where the view's `n` is, where `parent` holds: none at `n` of 2 and 4 too.
fn placed_under(o: i32, n: i64, address: Address, parent: bool) -> i32 {
    let small = matches!(n, 3 | 5 | 6 | 7) || (parent && matches!(n, 2 | 4));
    if matches!(address, Address::Repeat | Address::Mirror) && small {
        0
    } else {
        o
    }
}

/// [`check_with`] on a texture `w` texels wide (and 8 high); also returns how many channels the
/// offset added on every axis (Vulkan's placement) changes. The texture's `OffsetBaseAxes`
/// specialization is `parent`, the axes on which it is a view whose parent's base-level size is not
/// a power of two where its own is.
fn check_sized(
    name: &str,
    bc: &[u8],
    (address, pixel, word): (Address, bool, Option<u32>),
    form: Form,
    half: bool,
    (w, parent): (i64, u64),
) -> ((usize, usize), usize) {
    let p = lavapipe();
    let a = compile(bc, "k", &p).expect("compile");
    let note = if half {
        PrecisionNote::HalfSampleFilter
    } else {
        PrecisionNote::FloatSampleFilter
    };
    assert!(a.precision.contains(&note), "{name}");
    assert!(
        !a.requirements
            .properties
            .contains(&VkProperty::MetalSampler),
        "{name}"
    );
    link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{name}: {e}"));
    let (plan, m) = typed_module(bc, "k", &p).expect("typed module");
    let h = 8i64;
    let one_d = matches!(form, Form::OneD);
    let points = coordinates(w, h);
    let scale = if pixel {
        (1.0, 1.0)
    } else {
        (w as f64, h as f64)
    };
    let (layers, layer) = match form {
        Form::Array { layer } => (3u32, layer.clamp(0, 2) as u32),
        _ => (1, 0),
    };
    let added = match form {
        Form::Plain { offset } => offset,
        _ => (0, 0),
    };
    // Metal's placement of the offset, by a view's parent's base.
    let offset = (
        placed_under(added.0, w, address, parent & 1 != 0),
        placed_under(added.1, h, address, parent & 2 != 0),
    );
    // The channels whose answer the exact sum changes, and those the unmerged fractions change,
    // and those the offset added on every axis changes.
    let mut told_rules = (0, 0);
    let mut told_added = 0;
    for (seed, pf) in FORMATS.iter().enumerate() {
        let class = texel_format(*pf).0;
        let tex = Texels::random(class, w, if one_d { 1 } else { h }, seed as u64 + 11);
        let image = tex.image(combined_slot(&m), if one_d { 1 } else { 2 }, layers);
        let size = if half { 8 } else { 16 };
        let res = resources(
            &plan,
            &[coord_bytes(&points, scale), vec![0xab; size * points.len()]],
            class,
        );
        let mut res = with_word(res, &a, word);
        if parent != 0 {
            let key = SpecKey::OffsetBaseAxes(TEXTURE);
            assert!(a.interface.reads_key(key), "{name}");
            res.spec.insert(key, parent);
        }
        let out = run(&m, &res, &image, points.len() as u32, half);
        // The lanes whose answer the position previously (the unwrapped product) changes, and those
        // the wrap rounded up on the power-of-two height changes.
        let (mut told, mut told_up) = (0, 0);
        for (lane, ((x, y), got)) in points.iter().zip(&out).enumerate() {
            let at = (
                seen(*x, scale.0, address, pixel),
                seen(*y, scale.1, address, pixel),
            );
            let before = (product(*x, scale.0), product(*y, scale.1));
            let up = (at.0, seen_rounded(*y, scale.1, address, pixel, true));
            let metal = (one_d, half, PlaneRule::Metal);
            let differ = |p: (f64, f64)| {
                (0..got.len()).any(|c| {
                    expected(&tex, address, p, metal, offset, layer, c)
                        != expected(&tex, address, at, metal, offset, layer, c)
                })
            };
            told += usize::from(!pixel && differ(before));
            told_up += usize::from(!pixel && !one_d && up != at && differ(up));
            for (c, got) in got.iter().enumerate() {
                let want = expected(&tex, address, at, metal, offset, layer, c);
                let other =
                    |rule| expected(&tex, address, at, (one_d, half, rule), offset, layer, c);
                told_rules.0 += usize::from(other(PlaneRule::Exact) != want);
                told_rules.1 += usize::from(other(PlaneRule::Unmerged) != want);
                told_added +=
                    usize::from(expected(&tex, address, at, metal, added, layer, c) != want);
                assert_eq!(
                    *got, want,
                    "{name} {pf:?} lane {lane} ({x}, {y}) channel {c}: {got:#010x} vs {want:#010x}"
                );
            }
        }
        // A repeating or mirrored sampler's lanes tell Metal's position from the unwrapped
        // product, so the emulation previously fails them ([`coordinates`] chooses such lanes for
        // the 12-texel width).
        let wraps = w == 12 && matches!(address, Address::Repeat | Address::Mirror);
        assert!(
            pixel || !wraps || told > 0,
            "{name} {pf:?}: no lane tells the wrapped position from the unwrapped product"
        );
        // And, in 2D, from the wrap rounded up on the power-of-two axis, so the emulation before
        // item 19 fails them.
        assert!(
            pixel || one_d || !wraps || told_up > 0,
            "{name} {pf:?}: no lane tells the exact wrap from the wrap rounded up"
        );
    }
    (told_rules, told_added)
}

/// Every address mode, normalized and pixel coordinates, and each format family on a 12x8
/// texture: every lane is Metal's, on the interpreter.
#[test]
fn linear_half_samples_take_metals_filter() {
    const WHITE: [f64; 4] = [1.0; 4];
    let samplers = [
        ("edge", 1, false, 0, Address::Edge),
        ("zero", 0, false, 0, Address::Border([0.0; 4])),
        ("white", 0, false, 2, Address::Border(WHITE)),
        ("repeat", 2, false, 0, Address::Repeat),
        ("mirror", 3, false, 0, Address::Mirror),
        ("pixel edge", 1, true, 0, Address::Edge),
        ("pixel mirror", 3, true, 0, Address::Mirror),
    ];
    for (name, code, pixel, border, address) in samplers {
        let form = Form::Plain { offset: (0, 0) };
        let Some(bc) = assemble(name, &kernel(Some(linear(code, pixel, border)), form)) else {
            return;
        };
        check(name, &bc, address, pixel, form);
    }
}

/// A 1D texture's two taps, a 2D array's layer (clamped into the array), and a constant offset
/// shifting the base taps, under edge and repeat addressing.
#[test]
fn a_1d_texture_an_array_layer_and_an_offset() {
    let cases = [
        ("1d edge", 1, Address::Edge, Form::OneD),
        ("1d mirror", 3, Address::Mirror, Form::OneD),
        ("layer 1", 1, Address::Edge, Form::Array { layer: 1 }),
        ("layer 9", 1, Address::Edge, Form::Array { layer: 9 }),
        ("offset", 1, Address::Edge, Form::Plain { offset: (2, -1) }),
        (
            "repeat offset",
            2,
            Address::Repeat,
            Form::Plain { offset: (-3, 5) },
        ),
    ];
    for (name, code, address, form) in cases {
        let Some(bc) = assemble(name, &kernel(Some(linear(code, false, 0)), form)) else {
            return;
        };
        check(name, &bc, address, false, form);
    }
}

/// A constant offset on a repeating or mirrored axis, on textures 3, 5, 6, 7 (where Metal drops
/// it), 4 and 9 (where it adds it before the wrap) texels wide and 8 high (added): every lane is
/// Metal's, on the interpreter, and where Metal drops it some channel tells its placement from the
/// offset added on every axis.
#[test]
fn an_offset_on_a_wrapped_axis_takes_metals_placement() {
    for (code, address) in [(2, Address::Repeat), (3, Address::Mirror)] {
        // Neither `s` offset is a multiple of a dropping size's period, so each moves the taps.
        for (ox, oy) in [(2, 5), (-1, -1)] {
            let form = Form::Plain { offset: (ox, oy) };
            for half in [true, false] {
                let name = format!("{address:?} offset ({ox}, {oy}) half {half}");
                let k = kernel_of(Some(linear(code, false, 0)), form, !half);
                let Some(bc) = assemble(&name, &k) else {
                    return;
                };
                for w in [3, 4, 5, 6, 7, 9] {
                    let name = format!("{name} {w}x8");
                    let at = (address, false, None);
                    let (_, told) = check_sized(&name, &bc, at, form, half, (w, 0));
                    let dropped = matches!(w, 3 | 5 | 6 | 7);
                    assert_eq!(
                        told > 0,
                        dropped,
                        "{name}: {told} channels tell Metal's placement from the added offset"
                    );
                }
            }
        }
    }
}

/// The residue: a view whose first level lies above its parent's base reads Metal's rule by its
/// parent's base-level size. Under the slot's `OffsetBaseAxes` bit on `s` (a parent whose width is
/// not a power of two), a constant offset on a repeating or mirrored axis 2 or 4 texels wide drops,
/// and 8 wide adds; every lane is Metal's, on the interpreter, and at 2 and 4 some channel tells
/// the drop from the offset added; without the bit, 2, 4 and 8 add it, as before. A runtime
/// sampler's word reads the same bit; a `float` result's filter the same placement.
#[test]
fn an_offset_on_a_view_s_wrapped_axis_takes_its_parent_s_placement() {
    let runtime = runtime_word(
        SamplerAddressMode::Repeat,
        false,
        SamplerBorderColor::TransparentBlack,
    );
    for (code, address, word, halves) in [
        (Some(2), Address::Repeat, None, &[true, false][..]),
        (Some(3), Address::Mirror, None, &[true][..]),
        (None, Address::Repeat, Some(runtime), &[true][..]),
    ] {
        let form = Form::Plain { offset: (-1, 5) };
        for &half in halves {
            let name = format!("{address:?} runtime {} half {half}", word.is_some());
            let k = kernel_of(code.map(|c| linear(c, false, 0)), form, !half);
            let Some(bc) = assemble(&name, &k) else {
                return;
            };
            for w in [2, 4, 8] {
                for parent in [0, 1] {
                    let name = format!("{name} {w}x8 parent {parent}");
                    let at = (address, false, word);
                    let (_, told) = check_sized(&name, &bc, at, form, half, (w, parent));
                    let dropped = parent == 1 && w != 8;
                    assert_eq!(
                        told > 0,
                        dropped,
                        "{name}: {told} channels tell Metal's placement from the added offset"
                    );
                }
            }
        }
    }
}

/// A sample whose texel offset the emulation places on a repeating or mirrored axis, constexpr or
/// runtime, reads Metal's base-level size, a view's parent's: a slot's through its `OffsetBaseAxes`
/// specialization, which the stage declares, recording no `ViewBaseTexelOffset`. No offset, an
/// offset on clamping axes only, and MoltenVK (Metal's own sampler) declare none, and nothing
/// records the note.
#[test]
fn a_wrapped_offset_reads_the_parent_base_specialization() {
    let (lvp, m3) = (lavapipe(), m3());
    let note = PrecisionNote::ViewBaseTexelOffset;
    let offset = Form::Plain { offset: (2, -1) };
    let plain = Form::Plain { offset: (0, 0) };
    let cases = [
        ("repeat offset", Some(linear(2, false, 0)), offset, true),
        ("mirror offset", Some(linear(3, false, 0)), offset, true),
        ("runtime offset", None, offset, true),
        (
            "3d repeat offset",
            Some(linear(2, false, 0)),
            Form::ThreeD { offset: (0, 1, 0) },
            true,
        ),
        ("edge offset", Some(linear(1, false, 0)), offset, false),
        ("repeat", Some(linear(2, false, 0)), plain, false),
    ];
    let declares = |a: &crate::library::FunctionArtifact| {
        a.interface
            .read_keys()
            .into_iter()
            .any(|k| matches!(k, SpecKey::OffsetBaseAxes(_)))
    };
    for (name, words, form, placed) in cases {
        let Some(bc) = assemble(name, &kernel_of(words, form, false)) else {
            return;
        };
        let a = compile(&bc, "k", &lvp).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(!a.precision.contains(&note), "{name} on lavapipe");
        assert_eq!(declares(&a), placed, "{name} on lavapipe");
        let b = compile(&bc, "k", &m3).unwrap_or_else(|e| panic!("{name} on moltenvk-m3: {e}"));
        assert!(!b.precision.contains(&note), "{name} on moltenvk-m3");
        assert!(!declares(&b), "{name} on moltenvk-m3");
    }
}

/// A `float` result through the same filter, under each address mode, normalized and pixel
/// coordinates, a constant offset, 1D, and an array layer, on each format family: Metal's rule (a
/// normalized format's `n / D` to `f32`, a 16-bit one's 2D terms truncated; a 32-bit float or
/// half-float format's accumulated sum, the half-float's `half` widened).
#[test]
fn linear_float_samples_take_metals_filter() {
    const WHITE: [f64; 4] = [1.0; 4];
    let plain = Form::Plain { offset: (0, 0) };
    let cases = [
        ("float edge", 1, false, 0, Address::Edge, plain),
        ("float zero", 0, false, 0, Address::Border([0.0; 4]), plain),
        ("float white", 0, false, 2, Address::Border(WHITE), plain),
        ("float mirror", 3, false, 0, Address::Mirror, plain),
        (
            "float repeat offset",
            2,
            false,
            0,
            Address::Repeat,
            Form::Plain { offset: (-3, 5) },
        ),
        ("float pixel edge", 1, true, 0, Address::Edge, plain),
        (
            "float 1d zero",
            0,
            false,
            0,
            Address::Border([0.0; 4]),
            Form::OneD,
        ),
        ("float 1d mirror", 3, false, 0, Address::Mirror, Form::OneD),
        (
            "float layer 9",
            1,
            false,
            0,
            Address::Edge,
            Form::Array { layer: 9 },
        ),
    ];
    let mut told = (0, 0);
    for (name, code, pixel, border, address, form) in cases {
        let k = kernel_of(Some(linear(code, pixel, border)), form, true);
        let Some(bc) = assemble(name, &k) else {
            return;
        };
        let (sum, merge) = check_result(name, &bc, address, pixel, form, false);
        told = (told.0 + sum, told.1 + merge);
    }
    // Channels of the 16-bit formats tell Metal's 2D sum from the exact one and from the
    // unmerged fractions, so the emulation before it fails them. A `half` result rounds most such
    // differences away.
    assert!(
        told.0 > 0,
        "no channel tells the truncated terms from the exact sum"
    );
    assert!(
        told.1 > 0,
        "no channel tells the merged fractions from the unmerged ones"
    );
}

/// The extent of the 3D test texture: the position (the coordinate wrapped into one period first,
/// rounded up) is Metal's on its `s` and `r` axes, and the exact wrap on its `t` axis, a power of
/// two (the 3D probes, per axis).
const VOLUME: (i64, i64, i64) = (5, 8, 3);

/// The texel of a 3D texture's axis of extent `n` at index `v` under `address`, or `None` outside
/// a border axis.
fn volume_index(v: i64, n: i64, address: Address) -> Option<i64> {
    match address {
        Address::Edge => Some(v.clamp(0, n - 1)),
        Address::Repeat => Some(v.rem_euclid(n)),
        Address::Mirror => {
            let m = if v >= 0 { v } else { -1 - v };
            let r = m % (2 * n);
            Some(if r < n { r } else { 2 * n - 1 - r })
        }
        Address::Border(_) => (0..n).contains(&v).then_some(v),
    }
}

/// How the 3D reference reads Metal's rule: Metal's own, or with one rule dropped, to show a lane
/// depends on it.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Rule {
    Metal,
    /// A tie rounds up on the `t` axis's other tap.
    OtherTie,
    /// An axis whose two taps are one texel keeps its fraction.
    Unmerged,
}

/// Metal's 2D result of the taps `taps` (each with its 16-bit weight) of these texels' format
/// ([`metal_terms`]): a normalized format's terms truncated onto `2^-s` of its grid, `s` 4 for an
/// 8-bit format's `16 · code` and 8 for a 16-bit one's code. A linear cube sample's sum.
pub(super) fn plane_sum(tex: &Texels, taps: &[(Tap, i64)], half: bool) -> u32 {
    metal_terms(tex, taps, half, |g| if g == 16 { 4 } else { 8 })
}

/// Metal's 3D result of the taps `taps` (each with its 16-bit weight): a normalized format's terms
/// truncated onto `2^-s` of its grid, `s` 4 for an 8-bit format's `16 · code` and 8 for a 16-bit
/// one's code ([`metal_terms`]).
fn metal_3d(tex: &Texels, taps: &[(Tap, i64)], half: bool) -> u32 {
    metal_terms(tex, taps, half, |g| if g == 16 { 4 } else { 8 })
}

/// Metal's 2D or 3D result of the taps `taps` (each with its 16-bit weight) of a format of family
/// `fam`, as `half` bits when `half`, else `f32` bits: a normalized format's terms `code · w` each
/// truncated onto `2^-s` of its grid toward negative infinity (`s` of the family's grid factor
/// `g`), summed, then `⌊(T + 2^(15 - s)) / 2^(16 - s)⌋` and to nearest even; a float format's
/// through Metal's linear accumulator ([`metal`]).
fn metal_terms(tex: &Texels, taps: &[(Tap, i64)], half: bool, s: fn(i64) -> u32) -> u32 {
    let Family::Norm { scale, signed, g } = family(tex.class) else {
        return metal(tex, taps, 1 << 16, half);
    };
    let d = g * scale;
    let s = s(g);
    let t: i64 = taps
        .iter()
        .map(|(t, w)| {
            let n = match t {
                Tap::Stored(Stored::Code(c)) => c.max(&-scale) * g,
                Tap::Border(b) => (b * d as f64) as i64,
                Tap::Stored(Stored::Value(_)) => unreachable!("a float tap"),
            };
            (n * w) >> s
        })
        .sum();
    let lo = if signed { -d } else { 0 };
    let n = ((t + (1 << (15 - s))) >> (16 - s)).clamp(lo, d);
    if n == 0 {
        0
    } else if half {
        u32::from(to_half(n as f64 / d as f64, Rounding::NearestEven))
    } else {
        ratio_f32(n, d).to_bits()
    }
}

/// The expected bits of channel `c` of a 3D sample at texel-space `at` with constant `offset`
/// on a texture of extent [`VOLUME`] (`tex` holds its slices' rows in turn), read by `rule`: per
/// axis the base tap and fraction ([`split`]), each tap addressed, an axis whose two taps are one
/// texel (or both outside a border axis) weighed `(256, 0)`, the eight taps' weights
/// `round(wx · wy · wz / 256)` with a tie up on the `t` axis's first tap in Metal's frame (its
/// second where a mirrored axis's taps run backwards), and [`metal_3d`]'s sum.
fn expected_3d(
    tex: &Texels,
    address: Address,
    at: (f64, f64, f64),
    (offset, half, rule): ((i32, i32, i32), bool, Rule),
    c: usize,
) -> u32 {
    let (w, h, d) = VOLUME;
    let axis = |x: f64, o: i32, n: i64| {
        let (i, k) = split(x);
        let i = i + i64::from(o);
        let taps = [volume_index(i, n, address), volume_index(i + 1, n, address)];
        let k = if taps[0] == taps[1] && rule != Rule::Unmerged {
            0
        } else {
            k
        };
        (taps, [256 - k, k])
    };
    let (xs, wx) = axis(at.0, offset.0, w);
    let (ys, wy) = axis(at.1, offset.1, h);
    let (zs, wz) = axis(at.2, offset.2, d);
    let flip = matches!(address, Address::Mirror) && ys[0] > ys[1];
    let first = usize::from(flip != (rule == Rule::OtherTie));
    let mut taps = Vec::with_capacity(8);
    for n in 0..8 {
        let (a, b, e) = (n & 1, (n >> 1) & 1, n >> 2);
        let p = wx[a] * wy[b] * wz[e];
        let (q, r) = (p >> 8, p & 255);
        let weight = q + i64::from(r > 128 || (r == 128 && b == first));
        let tap = match (xs[a], ys[b], zs[e], address) {
            (Some(x), Some(y), Some(z), _) => Tap::Stored(tex.texel(0, x, y + h * z)[c]),
            (_, _, _, Address::Border(color)) => {
                let present = tex.class & (1 << (BicubicTexels::PRESENT + c as u32)) != 0;
                let missing = if c == 3 { 1.0 } else { 0.0 };
                Tap::Border(if present { color[c] } else { missing })
            }
            _ => unreachable!("only a border axis leaves the level"),
        };
        taps.push((tap, weight));
    }
    metal_3d(tex, &taps, half)
}

/// Texel-space coordinates of a 3D sample: [`coordinates`]' `(x, y)` on the plane, each at a depth
/// in turn from outside one end of the `r` axis to outside the other (several periods out
/// included), and lanes that make the rules show: the `s` fraction at `1/2` with odd `t` and `r`
/// fractions (every weight a rounding tie), in both halves of a mirrored `t` axis's period.
fn volume_coordinates() -> Vec<(f64, f64, f64)> {
    let (w, h, d) = VOLUME;
    let zs = [
        -1.25,
        0.0,
        0.5 + 1.0 / 512.0,
        1.3,
        d as f64 / 2.0 + 0.127,
        d as f64 - 0.6,
        d as f64 + 0.75,
        -2.0 * d as f64 - 0.4,
    ];
    let mut out: Vec<(f64, f64, f64)> = coordinates(w, h)
        .into_iter()
        .enumerate()
        .map(|(n, (x, y))| (x, y, zs[(3 * n + 1) % zs.len()]))
        .collect();
    for y in [3.0 + 5.0 / 256.0, -3.0 + 5.0 / 256.0, 2.0 + 77.0 / 256.0] {
        for x in [2.0, 1.0, 4.0] {
            out.push((x, y, 1.0 + 3.0 / 256.0));
        }
    }
    out
}

/// Run the 3D kernel `bc` (through an `address` sampler, constant `offset`, a `half` result when
/// `half`) on a [`VOLUME`] texture of every format of [`FORMATS`] on the interpreter, against
/// [`expected_3d`]; returns how many lanes each dropped rule changes (the other tie, unmerged),
/// how many the wrap rounded up on the power-of-two `t` axis changes, and how many the offset
/// added on every axis (Vulkan's placement) changes.
fn check_3d(
    name: &str,
    bc: &[u8],
    address: Address,
    offset: (i32, i32, i32),
    half: bool,
) -> (usize, usize, usize, usize) {
    check_3d_with(name, bc, (address, None), offset, half)
}

/// [`check_3d`] of a kernel sampling through runtime sampler slot 0 whose bound sampler's word is
/// `word`, or through its constexpr sampler.
fn check_3d_with(
    name: &str,
    bc: &[u8],
    (address, word): (Address, Option<u32>),
    added: (i32, i32, i32),
    half: bool,
) -> (usize, usize, usize, usize) {
    let p = lavapipe();
    let a = compile(bc, "k", &p).expect("compile");
    let note = if half {
        PrecisionNote::HalfSampleFilter
    } else {
        PrecisionNote::FloatSampleFilter
    };
    assert!(a.precision.contains(&note), "{name}");
    assert!(
        a.strategies.entries.contains_key("sampler.linear.block"),
        "{name}: not emulated"
    );
    link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{name}: {e}"));
    let (plan, m) = typed_module(bc, "k", &p).expect("typed module");
    let (w, h, d) = VOLUME;
    // Metal's placement of the offset, per axis.
    let offset = (
        placed(added.0, w, address),
        placed(added.1, h, address),
        placed(added.2, d, address),
    );
    let points = volume_coordinates();
    let bytes: Vec<u8> = points
        .iter()
        .flat_map(|(x, y, z)| {
            [
                ((x / w as f64) as f32).to_le_bytes(),
                ((y / h as f64) as f32).to_le_bytes(),
                ((z / d as f64) as f32).to_le_bytes(),
                0f32.to_le_bytes(),
            ]
        })
        .flatten()
        .collect();
    let mut shown = (0, 0, 0, 0);
    for (seed, pf) in FORMATS.iter().enumerate() {
        let class = texel_format(*pf).0;
        let tex = Texels::random(class, w, h * d, seed as u64 + 23);
        let mut image = Image::new(combined_slot(&m), [w as u32, h as u32, d as u32], 1, 1, 1);
        for z in 0..d {
            for y in 0..h {
                for x in 0..w {
                    let at = image
                        .index(0, 0, 0, [x as u32, y as u32, z as u32])
                        .expect("texel");
                    let v = tex.texel(0, x, y + h * z);
                    image.texels[at] = v.map(|c| tex.fetched(c).to_bits());
                }
            }
        }
        let size = if half { 8 } else { 16 };
        let res = resources(
            &plan,
            &[bytes.clone(), vec![0xab; size * points.len()]],
            class,
        );
        let res = with_word(res, &a, word);
        let out = run(&m, &res, &image, points.len() as u32, half);
        for (lane, ((x, y, z), got)) in points.iter().zip(&out).enumerate() {
            let at = (
                seen(*x, w as f64, address, false),
                seen(*y, h as f64, address, false),
                seen(*z, d as f64, address, false),
            );
            let want =
                |c: usize, rule: Rule| expected_3d(&tex, address, at, (offset, half, rule), c);
            shown.0 += usize::from((0..4).any(|c| want(c, Rule::OtherTie) != want(c, Rule::Metal)));
            shown.1 += usize::from((0..4).any(|c| want(c, Rule::Unmerged) != want(c, Rule::Metal)));
            shown.3 += usize::from((0..4).any(|c| {
                expected_3d(&tex, address, at, (added, half, Rule::Metal), c)
                    != want(c, Rule::Metal)
            }));
            let up = (at.0, seen_rounded(*y, h as f64, address, false, true), at.2);
            shown.2 += usize::from(
                up != at
                    && (0..4).any(|c| {
                        expected_3d(&tex, address, up, (offset, half, Rule::Metal), c)
                            != want(c, Rule::Metal)
                    }),
            );
            for (c, got) in got.iter().enumerate() {
                let want = want(c, Rule::Metal);
                assert_eq!(
                    *got, want,
                    "{name} {pf:?} lane {lane} ({x}, {y}, {z}) channel {c}: {got:#010x} vs \
                     {want:#010x}"
                );
            }
        }
    }
    shown
}

/// A 3D linear sample under every address mode, a constant offset on clamping axes and on repeating
/// and mirrored ones ([`VOLUME`]'s `s` and `r` axes, 5 and 3 texels, drop it, and its `t` axis, 8,
/// adds it), each format family, and a `half` or `float` result: every lane is Metal's 3D rule, on
/// the interpreter. The lanes include ties of the 16-bit weights (in both frames of a mirrored
/// axis) and axes whose two taps are one texel, which the dropped rules answer differently, and
/// lanes where the offset added on every axis reads other taps.
#[test]
fn linear_3d_samples_take_metals_filter() {
    const WHITE: [f64; 4] = [1.0; 4];
    let cases = [
        ("3d edge", 1, 0, Address::Edge, (0, 0, 0), true),
        ("3d zero", 0, 0, Address::Border([0.0; 4]), (0, 0, 0), true),
        ("3d white", 0, 2, Address::Border(WHITE), (0, 0, 0), true),
        ("3d repeat", 2, 0, Address::Repeat, (0, 0, 0), true),
        ("3d mirror", 3, 0, Address::Mirror, (0, 0, 0), true),
        ("3d edge offset", 1, 0, Address::Edge, (1, -2, 1), true),
        ("3d repeat offset", 2, 0, Address::Repeat, (2, -3, 1), true),
        ("3d mirror offset", 3, 0, Address::Mirror, (-1, 1, 2), true),
        ("3d float edge", 1, 0, Address::Edge, (0, 0, 0), false),
        (
            "3d float repeat offset",
            2,
            0,
            Address::Repeat,
            (-4, 5, -2),
            false,
        ),
        ("3d float mirror", 3, 0, Address::Mirror, (0, 0, 0), false),
        (
            "3d float white offset",
            0,
            2,
            Address::Border(WHITE),
            (-1, 0, 2),
            false,
        ),
    ];
    for (name, code, border, address, offset, half) in cases {
        let k = kernel_of(
            Some(linear(code, false, border)),
            Form::ThreeD { offset },
            !half,
        );
        let Some(bc) = assemble(name, &k) else {
            return;
        };
        let (tie, merge, up, added) = check_3d(name, &bc, address, offset, half);
        assert!(tie > 0, "{name}: no lane shows the tie rule");
        if matches!(address, Address::Repeat | Address::Mirror) && offset != (0, 0, 0) {
            assert!(
                added > 0,
                "{name}: no lane shows Metal's placement of the offset"
            );
        }
        if matches!(address, Address::Edge | Address::Mirror) {
            assert!(merge > 0, "{name}: no lane shows the merged taps");
        }
        if matches!(address, Address::Repeat | Address::Mirror) {
            assert!(
                up > 0,
                "{name}: no lane shows the exact wrap on the power-of-two axis"
            );
        }
    }
}

/// A sample through a runtime sampler slot takes Metal's filter by the bound sampler's word, as a
/// constexpr sampler of the same state does: every lane is Metal's rule on the
/// interpreter under each address mode, a transparent, opaque-black, and white border (and
/// `clamp_to_zero` beside a white and an opaque-black border color, which it reads as
/// `clamp_to_border` does), pixel coordinates, 1D, a constant offset, 3D, and a `half` or `float`
/// result, with the word's state read only in the function.
#[test]
fn runtime_sampler_samples_take_metals_filter() {
    use SamplerAddressMode as M;
    use SamplerBorderColor as B;
    const WHITE: [f64; 4] = [1.0; 4];
    const BLACK: [f64; 4] = [0.0, 0.0, 0.0, 1.0];
    let plain = Form::Plain { offset: (0, 0) };
    let cases = [
        (
            "runtime edge",
            M::ClampToEdge,
            B::TransparentBlack,
            Address::Edge,
            false,
            plain,
        ),
        (
            "runtime zero",
            M::ClampToZero,
            B::OpaqueWhite,
            Address::Border(WHITE),
            false,
            plain,
        ),
        (
            "runtime zero black",
            M::ClampToZero,
            B::OpaqueBlack,
            Address::Border(BLACK),
            false,
            plain,
        ),
        (
            "runtime white",
            M::ClampToBorderColor,
            B::OpaqueWhite,
            Address::Border(WHITE),
            false,
            plain,
        ),
        (
            "runtime black",
            M::ClampToBorderColor,
            B::OpaqueBlack,
            Address::Border(BLACK),
            false,
            plain,
        ),
        (
            "runtime repeat",
            M::Repeat,
            B::TransparentBlack,
            Address::Repeat,
            false,
            plain,
        ),
        (
            "runtime mirror",
            M::MirrorRepeat,
            B::TransparentBlack,
            Address::Mirror,
            false,
            plain,
        ),
        (
            "runtime pixel zero",
            M::ClampToZero,
            B::TransparentBlack,
            Address::Border([0.0; 4]),
            true,
            plain,
        ),
        (
            "runtime 1d mirror",
            M::MirrorRepeat,
            B::TransparentBlack,
            Address::Mirror,
            false,
            Form::OneD,
        ),
        (
            "runtime repeat offset",
            M::Repeat,
            B::TransparentBlack,
            Address::Repeat,
            false,
            Form::Plain { offset: (-3, 5) },
        ),
    ];
    for (name, mode, border, address, pixel, form) in cases {
        let Some(bc) = assemble(name, &kernel(None, form)) else {
            return;
        };
        let word = runtime_word(mode, pixel, border);
        check_with(name, &bc, (address, pixel, Some(word)), form, true);
    }
    for (name, mode, border, address) in [
        (
            "runtime float mirror",
            M::MirrorRepeat,
            B::TransparentBlack,
            Address::Mirror,
        ),
        (
            "runtime float white",
            M::ClampToBorderColor,
            B::OpaqueWhite,
            Address::Border(WHITE),
        ),
    ] {
        let Some(bc) = assemble(name, &kernel_of(None, plain, true)) else {
            return;
        };
        let word = runtime_word(mode, false, border);
        check_with(name, &bc, (address, false, Some(word)), plain, false);
    }
    for (name, mode, border, address, offset, half) in [
        (
            "runtime 3d zero",
            M::ClampToZero,
            B::TransparentBlack,
            Address::Border([0.0; 4]),
            (0, 0, 0),
            true,
        ),
        (
            "runtime 3d mirror",
            M::MirrorRepeat,
            B::TransparentBlack,
            Address::Mirror,
            (0, 0, 0),
            true,
        ),
        (
            "runtime 3d repeat",
            M::Repeat,
            B::TransparentBlack,
            Address::Repeat,
            (0, 0, 0),
            false,
        ),
        (
            "runtime 3d white offset",
            M::ClampToBorderColor,
            B::OpaqueWhite,
            Address::Border(WHITE),
            (-1, 0, 2),
            false,
        ),
        (
            "runtime 3d repeat offset",
            M::Repeat,
            B::TransparentBlack,
            Address::Repeat,
            (1, -2, 2),
            true,
        ),
    ] {
        let k = kernel_of(None, Form::ThreeD { offset }, !half);
        let Some(bc) = assemble(name, &k) else {
            return;
        };
        let word = runtime_word(mode, false, border);
        check_3d_with(name, &bc, (address, Some(word)), offset, half);
    }
}

/// What the emulation reaches, what the device filters under the note, and what neither touches. On
/// lavapipe a linear sample of a 1D, 2D, or 3D slot is emulated, with a `half` or `float` result
/// (it calls the linear routines and fetches), through a constexpr sampler or a runtime one (whose
/// word selects the emulation in the function), a minimum or maximum reduction too (read as the
/// weighted average); the device filters a mip filter (a linear one with nearest minification and
/// magnification included), anisotropy, differing filters, `min_lod_clamp`, and a dynamic offset (a
/// 3D sample's offset on a repeating or mirrored axis is emulated). Every filtered sample is noted,
/// emulated or not (`PrecisionNote::HalfSampleFilter` for a `half` result,
/// `PrecisionNote::FloatSampleFilter` for a `float` one); a nearest sample carries no note. On
/// MoltenVK no sample is emulated, and every filtered sample of either result type requires
/// `VkProperty::MetalSampler`, as does every `half` sample, whose texel's rounding to `half` there
/// is Metal's only through Metal's own texture unit.
#[test]
fn the_emulation_reaches_what_metal_measured() {
    let edge = linear(1, false, 0);
    let plain = Form::Plain { offset: (0, 0) };
    // The census names an emulated sample (and counts its words once per call site, the routines
    // once per module).
    let emulated = |a: &crate::library::FunctionArtifact| {
        let linear = a.strategies.entries.get("sampler.linear").map(|u| u.uses);
        let routines = ["quad", "span", "block"].iter().any(|r| {
            a.strategies
                .entries
                .contains_key(&format!("sampler.linear.{r}"))
        });
        assert_eq!(
            linear.is_some(),
            routines,
            "the census of {:?}",
            a.strategies
        );
        linear == Some(1)
    };
    // The note of a filtered sample of `form`'s result type, and the other type's, which it never
    // records.
    let notes = |form: Form| {
        if matches!(form, Form::Float) {
            (
                PrecisionNote::FloatSampleFilter,
                PrecisionNote::HalfSampleFilter,
            )
        } else {
            (
                PrecisionNote::HalfSampleFilter,
                PrecisionNote::FloatSampleFilter,
            )
        }
    };
    let metal = |a: &crate::library::FunctionArtifact| {
        a.requirements
            .properties
            .contains(&VkProperty::MetalSampler)
    };
    // (name, sampler, form, emulated on lavapipe, filtered)
    let cases: Vec<(&str, Option<u64>, Form, bool, bool)> = vec![
        ("2d", Some(edge), plain, true, true),
        ("1d", Some(edge), Form::OneD, true, true),
        ("array", Some(edge), Form::Array { layer: 1 }, true, true),
        ("runtime", None, plain, true, true),
        (
            "mip",
            Some(sampler_words(1, 2, 1, false, 0)),
            plain,
            false,
            true,
        ),
        ("anisotropy", Some(edge | 1 << 20), plain, false, true),
        // A minimum reduction reads the weighted average, as on Apple's GPU.
        ("min reduction", Some(edge | 1 << 58), plain, true, true),
        ("max reduction", Some(edge | 2 << 58), plain, true, true),
        (
            "linear min nearest mag",
            Some(edge & !(1 << 9)),
            plain,
            false,
            true,
        ),
        (
            "3d",
            Some(edge),
            Form::ThreeD { offset: (0, 0, 0) },
            true,
            true,
        ),
        (
            "3d edge offset",
            Some(edge),
            Form::ThreeD { offset: (1, -2, 3) },
            true,
            true,
        ),
        (
            "3d repeat offset",
            Some(linear(2, false, 0)),
            Form::ThreeD { offset: (0, 1, 0) },
            true,
            true,
        ),
        (
            "3d mirror offset",
            Some(linear(3, false, 0)),
            Form::ThreeD { offset: (0, 0, -1) },
            true,
            true,
        ),
        (
            "3d repeat",
            Some(linear(2, false, 0)),
            Form::ThreeD { offset: (0, 0, 0) },
            true,
            true,
        ),
        ("min_lod_clamp", Some(edge), Form::MinLodClamp, false, true),
        (
            "dynamic offset",
            Some(edge),
            Form::DynamicOffset,
            false,
            true,
        ),
        (
            "nearest",
            Some(sampler_words(0, 0, 1, false, 0)),
            plain,
            false,
            false,
        ),
        (
            "nearest with a linear mip filter",
            Some(sampler_words(0, 2, 1, false, 0)),
            plain,
            false,
            true,
        ),
        ("float", Some(edge), Form::Float, true, true),
        ("float runtime", None, Form::Float, true, true),
        (
            "float nearest",
            Some(sampler_words(0, 0, 1, false, 0)),
            Form::Float,
            false,
            false,
        ),
        (
            "float nearest with a linear mip filter",
            Some(sampler_words(0, 2, 1, false, 0)),
            Form::Float,
            false,
            true,
        ),
    ];
    let (lvp, m3) = (lavapipe(), m3());
    for (name, sampler, form, reached, filtered) in cases {
        let Some(bc) = assemble(name, &kernel(sampler, form)) else {
            return;
        };
        let a = compile(&bc, "k", &lvp).unwrap_or_else(|e| panic!("{name} on lavapipe: {e}"));
        link_and_validate(&a, &lvp).unwrap_or_else(|e| panic!("{name} on lavapipe: {e}"));
        let (note, other) = notes(form);
        assert_eq!(emulated(&a), reached, "{name} on lavapipe");
        assert_eq!(a.precision.contains(&note), filtered, "{name} on lavapipe");
        assert!(!a.precision.contains(&other), "{name} on lavapipe");
        assert!(!metal(&a), "{name} on lavapipe");
        let b = compile(&bc, "k", &m3).unwrap_or_else(|e| panic!("{name} on moltenvk-m3: {e}"));
        link_and_validate(&b, &m3).unwrap_or_else(|e| panic!("{name} on moltenvk-m3: {e}"));
        assert!(!emulated(&b), "{name} on moltenvk-m3");
        assert!(!b.precision.contains(&note), "{name} on moltenvk-m3");
        assert!(!b.precision.contains(&other), "{name} on moltenvk-m3");
        let half = !matches!(form, Form::Float);
        assert_eq!(metal(&b), filtered || half, "{name} on moltenvk-m3");
    }
}

/// A kernel comparing `depth2d<float>` slot 0 at buffer 0's lane `tid` against `0.5` through a
/// constexpr sampler with comparison `less`, minification and magnification `filter` (0 nearest,
/// 1 linear), and mip filter `mip` (0 none, 2 linear), at level 0, storing the `float` result to
/// buffer 1.
fn compare_kernel(filter: u64, mip: u64) -> String {
    let w = sampler_words(filter, mip, 1, false, 0) & !(0xf << 16) | 1 << 16;
    format!(
        r#"@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 {w}, i64 0], align 8

define void @k(ptr addrspace(1) %d, ptr addrspace(1) %c, ptr addrspace(1) %out, i32 %tid) {{
  %i = zext i32 %tid to i64
  %p = getelementptr inbounds <2 x float>, ptr addrspace(1) %c, i64 %i
  %uv = load <2 x float>, ptr addrspace(1) %p, align 8
  %r = call {{ float, i8 }} @air.sample_compare_depth_2d.f32(ptr addrspace(1) %d, ptr addrspace(2) @__air_sampler_state, i32 1, <2 x float> %uv, float 0.5, i1 true, <2 x i32> zeroinitializer, i1 true, float 0.0, float 0.0, i32 0)
  %v = extractvalue {{ float, i8 }} %r, 0
  %q = getelementptr inbounds float, ptr addrspace(1) %out, i64 %i
  store float %v, ptr addrspace(1) %q, align 4
  ret void
}}

declare {{ float, i8 }} @air.sample_compare_depth_2d.f32(ptr addrspace(1), ptr addrspace(2), i32, <2 x float>, float, i1, <2 x i32>, i1, float, float, i32)

!air.kernel = !{{!0}}
!air.sampler_states = !{{!6}}
!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4, !5, !7}}
!3 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"depth2d<float, sample>", !"air.arg_name", !"d"}}
!4 = !{{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"float2", !"air.arg_name", !"c"}}
!5 = !{{i32 2, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}}
!6 = !{{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}}
!7 = !{{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}}
"#
    )
}

/// A depth texture's comparison through a linear sampler, or a nearest one with a linear mip
/// filter, is a filtered `float` sample: noted `PrecisionNote::FloatSampleFilter` on lavapipe
/// (where the linear one is emulated, and the other blends two levels' comparisons where the slot's
/// format is measured and keeps the device's filter elsewhere) and requiring
/// `VkProperty::MetalSampler` on MoltenVK; through a nearest sampler it is not noted, and requires
/// the property on MoltenVK too: the device's comparison is Metal's there, and elsewhere the shader
/// compares. Only the comparison the device may make on lavapipe, the nearest one with a linear mip
/// filter (where the format's class is not measured), notes `PrecisionNote::DeviceDepthComparison`:
/// the shader compares the others' fetched texels.
#[test]
fn a_depth_comparison_through_a_filter_is_a_filtered_float_sample() {
    let (lvp, m3) = (lavapipe(), m3());
    for (name, filter, mip, filtered, device) in [
        ("linear", 1, 0, true, false),
        ("nearest with a linear mip filter", 0, 2, true, true),
        ("nearest", 0, 0, false, false),
    ] {
        let Some(bc) = assemble(name, &compare_kernel(filter, mip)) else {
            return;
        };
        let a = compile(&bc, "k", &lvp).unwrap_or_else(|e| panic!("{name} on lavapipe: {e}"));
        link_and_validate(&a, &lvp).unwrap_or_else(|e| panic!("{name} on lavapipe: {e}"));
        assert_eq!(
            a.precision.contains(&PrecisionNote::FloatSampleFilter),
            filtered,
            "{name} on lavapipe"
        );
        assert!(!a.precision.contains(&PrecisionNote::HalfSampleFilter));
        assert_eq!(
            a.precision.contains(&PrecisionNote::DeviceDepthComparison),
            device,
            "{name} on lavapipe"
        );
        assert!(!a
            .requirements
            .properties
            .contains(&VkProperty::MetalSampler));
        let b = compile(&bc, "k", &m3).unwrap_or_else(|e| panic!("{name} on moltenvk-m3: {e}"));
        link_and_validate(&b, &m3).unwrap_or_else(|e| panic!("{name} on moltenvk-m3: {e}"));
        assert!(!b.precision.contains(&PrecisionNote::FloatSampleFilter));
        assert!(!b.precision.contains(&PrecisionNote::DeviceDepthComparison));
        assert!(
            b.requirements
                .properties
                .contains(&VkProperty::MetalSampler),
            "{name} on moltenvk-m3"
        );
    }
}

/// The depth formats whose linear sample and comparison Metal answers through their color class
/// (`metal::texel_format`).
const DEPTHS: [PixelFormat; 2] = [PixelFormat::Depth32Float, PixelFormat::Depth16Unorm];

/// The words of a linear constexpr sampler addressed by AIR code `address` with border code
/// `border`, pixel coordinates where `pixel`, under AIR compare function code `compare` (1
/// `less` to 7 `always`, 8 `never`).
fn depth_words(address: u64, pixel: bool, border: u64, compare: u64) -> u64 {
    linear(address, pixel, border) & !(0xf << 16) | compare << 16
}

/// The text of a depth kernel that names its sampler: the constexpr sampler of `words`, or
/// (`None`) runtime sampler slot 0, the parameter after `out`. Its fields are the global, the
/// parameter, the call's operand, the module's sampler-state list, the kernel's argument list,
/// the sampler's metadata node `!6`, and the thread position's argument index.
struct SamplerText {
    global: String,
    param: &'static str,
    operand: &'static str,
    states: &'static str,
    args: &'static str,
    node: &'static str,
    tid: u32,
}

fn sampler_text(words: Option<u64>) -> SamplerText {
    match words {
        Some(w) => SamplerText {
            global: format!(
                "@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 {w}, i64 0], align 8\n\n"
            ),
            param: "",
            operand: "@__air_sampler_state",
            states: "!air.sampler_states = !{!6}\n",
            args: "!3, !4, !5, !7",
            node: r#"!{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}"#,
            tid: 3,
        },
        None => SamplerText {
            global: String::new(),
            param: ", ptr addrspace(2) %s",
            operand: "%s",
            states: "",
            args: "!3, !4, !5, !6, !7",
            node: r#"!{i32 3, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"s"}"#,
            tid: 4,
        },
    }
}

/// A kernel reading `depth2d<float>` slot 0 (`depth2d_array<float>` at layer `layer`) through
/// the constexpr sampler of `words` (runtime sampler slot 0 for `None`, [`sampler_text`]) at level 0 at buffer 0's lane `tid` (a `float4`: the
/// coordinate, and the reference in `z`) with constant offset `offset`: a sample, or where
/// `compare` a comparison against the reference. Buffer 1's lane `tid` is a `float4` whose `x` is
/// the result.
fn depth_kernel(
    words: Option<u64>,
    compare: bool,
    layer: Option<i32>,
    (ox, oy): (i32, i32),
) -> String {
    let SamplerText {
        global,
        param,
        operand,
        states,
        args,
        node,
        tid,
    } = sampler_text(words);
    let (dim, tex_ty, layer_arg, layer_ty) = match layer {
        Some(l) => (
            "2d_array",
            "depth2d_array<float, sample>",
            format!("i32 {l}, "),
            "i32, ",
        ),
        None => ("2d", "depth2d<float, sample>", String::new(), ""),
    };
    let (name, ref_arg, ref_ty) = match compare {
        true => (
            format!("sample_compare_depth_{dim}"),
            "float %ref, ",
            "float, ",
        ),
        false => (format!("sample_depth_{dim}"), "", ""),
    };
    format!(
        r#"{global}define void @k(ptr addrspace(1) %t, ptr addrspace(1) %c, ptr addrspace(1) %out{param}, i32 %tid) {{
  %i = zext i32 %tid to i64
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %c, i64 %i
  %c4 = load <4 x float>, ptr addrspace(1) %p, align 16
  %uv = shufflevector <4 x float> %c4, <4 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %ref = extractelement <4 x float> %c4, i64 2
  %r = call {{ float, i8 }} @air.{name}.f32(ptr addrspace(1) %t, ptr addrspace(2) {operand}, i32 1, <2 x float> %uv, {layer_arg}{ref_arg}i1 true, <2 x i32> <i32 {ox}, i32 {oy}>, i1 true, float 0.0, float 0.0, i32 0)
  %v = extractvalue {{ float, i8 }} %r, 0
  %v4 = insertelement <4 x float> zeroinitializer, float %v, i64 0
  %q = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i64 %i
  store <4 x float> %v4, ptr addrspace(1) %q, align 16
  ret void
}}

declare {{ float, i8 }} @air.{name}.f32(ptr addrspace(1), ptr addrspace(2), i32, <2 x float>, {layer_ty}{ref_ty}i1, <2 x i32>, i1, float, float, i32)

!air.kernel = !{{!0}}
{states}!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{{args}}}
!3 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"{tex_ty}", !"air.arg_name", !"t"}}
!4 = !{{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"c"}}
!5 = !{{i32 2, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}}
!6 = {node}
!7 = !{{i32 {tid}, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}}
"#
    )
}

/// Whether reference `r` passes AIR compare function code `f` against depth `x`, as Metal
/// compares them: a `Depth16Unorm` texture's (`code`, the texel's code) as codes, the reference's
/// the code of its exact product with 65535 (clamped into `[0, 1]`, half up, a NaN kept); a
/// `Depth32Float` texture's as `f32`s. A NaN fails every ordered test and passes `not_equal`.
pub(super) fn depth_passes(f: u64, r: f32, (x, code): (f32, Option<i64>)) -> bool {
    let (r, x) = match code {
        Some(c) if !r.is_nan() => {
            let m = f64::from(r.clamp(0.0, 1.0));
            ((m * 65535.0 + 0.5).floor(), c as f64)
        }
        Some(c) => (f64::NAN, c as f64),
        None => (f64::from(r), f64::from(x)),
    };
    match f {
        1 => r < x,
        2 => r <= x,
        3 => r > x,
        4 => r >= x,
        5 => r == x,
        6 => r != x,
        7 => true,
        8 => false,
        _ => unreachable!("compare code {f}"),
    }
}

/// Runs depth kernel `bc` (through a sampler addressed by `address`, with `pixel` coordinates,
/// constant offset `offset`; on an array of `layers` read at `layer`) on lavapipe for each of
/// [`DEPTHS`] on a 12x8 texture at [`coordinates`]' lanes, each with a reference (a texel's
/// depth on every third lane), against Metal's rule: a sample is the color class's channel 0; a
/// comparison under AIR code `compare` is an `r8Unorm` linear sample of each tap's pass as codes 0
/// and 255, the border's depth compared as a texel's. A comparison declares a binding with
/// comparison, which a format whose comparison is measured never reads, and one without, through
/// which the emulation fetches its taps (`Linear::fetch`): the texels are bound through the latter
/// alone, the former holding zeros. A kernel sampling through runtime sampler slot 0 reads `word`
/// as the bound sampler's word ([`with_word`]).
fn check_depth(
    name: &str,
    bc: &[u8],
    (address, pixel): (Address, bool),
    (offset, layers, layer): ((i32, i32), u32, u32),
    (compare, word): (Option<u64>, Option<u32>),
) {
    let p = lavapipe();
    let a = compile(bc, "k", &p).unwrap_or_else(|e| panic!("{name}: {e}"));
    link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{name}: {e}"));
    assert!(a.precision.contains(&PrecisionNote::FloatSampleFilter));
    assert!(
        a.strategies.entries.contains_key("sampler.linear"),
        "{name}: {:?}",
        a.strategies
    );
    let (plan, m) = typed_module(bc, "k", &p).expect("typed module");
    let (w, h) = (12i64, 8i64);
    let points = coordinates(w, h);
    let scale = if pixel {
        (1.0, 1.0)
    } else {
        (w as f64, h as f64)
    };
    let refs = [0.0f32, 0.5, 1.0, -0.25, 0.75, f32::NAN, 1.5, 0.375];
    for (seed, pf) in DEPTHS.iter().enumerate() {
        let class = texel_format(*pf).0;
        let d16 = *pf == PixelFormat::Depth16Unorm;
        let tex = Texels::random(class, w, h, seed as u64 + 31);
        let slots = combined_slots(&m);
        let mut compares: Vec<bool> = slots.iter().map(|(_, c)| *c).collect();
        compares.sort();
        assert_eq!(
            compares,
            match compare {
                Some(_) => vec![false, true],
                None => vec![false],
            },
            "{name}: the combined bindings' comparisons"
        );
        let images: Vec<Image> = slots
            .iter()
            .map(|&(slot, compares)| match compares {
                true => Image::new(slot, [w as u32, h as u32, 1], layers, 1, 1),
                false => tex.image(slot, 2, layers),
            })
            .collect();
        let n = tex.t.len();
        let lanes: Vec<(f64, f64, f32)> = points
            .iter()
            .enumerate()
            .map(|(k, (x, y))| {
                let r = match k % 3 {
                    0 => tex.fetched(tex.t[(7 * k) % n][0]),
                    _ => refs[k % refs.len()],
                };
                (*x, *y, r)
            })
            .collect();
        let bytes: Vec<u8> = lanes
            .iter()
            .flat_map(|(x, y, r)| {
                [(x / scale.0) as f32, (y / scale.1) as f32, *r, 0.0].map(f32::to_le_bytes)
            })
            .flatten()
            .collect();
        let res = resources(&plan, &[bytes, vec![0xab; 16 * lanes.len()]], class);
        let res = with_word(res, &a, word);
        let out = run_images(&m, &res, &images, lanes.len() as u32, false);
        for (lane, ((x, y, r), got)) in lanes.iter().zip(&out).enumerate() {
            let at = (
                seen(*x, scale.0, address, pixel),
                seen(*y, scale.1, address, pixel),
            );
            let rule = (false, false, PlaneRule::Metal);
            let want = match compare {
                None => expected(&tex, address, at, rule, offset, layer, 0),
                Some(f) => {
                    let pass = |s: Stored| {
                        let code = match s {
                            Stored::Code(c) => Some(c),
                            Stored::Value(_) => None,
                        };
                        i64::from(depth_passes(f, *r, (tex.fetched(s), code)))
                    };
                    let t = tex
                        .t
                        .iter()
                        .map(|s| {
                            let p = Stored::Code(255 * pass(s[0]));
                            [p, Stored::Code(0), Stored::Code(0), Stored::Code(255)]
                        })
                        .collect();
                    let passes = Texels {
                        class: texel_format(PixelFormat::R8Unorm).0,
                        w,
                        h,
                        t,
                    };
                    let address = match address {
                        Address::Border(b) => {
                            let depth = match d16 {
                                true => Stored::Code((b[0] * 65535.0) as i64),
                                false => Stored::Value(b[0] as f32),
                            };
                            Address::Border([pass(depth) as f64, 0.0, 0.0, 1.0])
                        }
                        other => other,
                    };
                    expected(&passes, address, at, rule, offset, layer, 0)
                }
            };
            assert_eq!(
                got[0],
                want,
                "{name} {pf:?} lane {lane} ({x}, {y}) reference {r}: {:?} vs {:?}",
                f32::from_bits(got[0]),
                f32::from_bits(want)
            );
        }
    }
}

/// A depth texture's linear sample is its color class's (`Depth32Float`'s `R32Float`'s,
/// `Depth16Unorm`'s `R16Unorm`'s; 2,191 lanes), emulated on lavapipe and Metal's on every lane of
/// the interpreter: clamping to the edge, to zero, and to a white border, pixel coordinates, an
/// offset, and an array's layer.
#[test]
fn depth_linear_samples_take_their_color_class() {
    const WHITE: [f64; 4] = [1.0; 4];
    let cases = [
        ("edge", (1, false, 0), Address::Edge, (0, 0), None),
        (
            "zero",
            (0, false, 0),
            Address::Border([0.0; 4]),
            (0, 0),
            None,
        ),
        ("white", (0, false, 2), Address::Border(WHITE), (0, 0), None),
        ("pixel edge", (1, true, 0), Address::Edge, (0, 0), None),
        ("offset", (1, false, 0), Address::Edge, (2, -1), None),
        ("layer 9", (1, false, 0), Address::Edge, (0, 0), Some(9)),
    ];
    for (name, (code, pixel, border), address, offset, layer) in cases {
        let words = depth_words(code, pixel, border, 8);
        let Some(bc) = assemble(name, &depth_kernel(Some(words), false, layer, offset)) else {
            return;
        };
        let (layers, at) = layer.map_or((1, 0), |l| (3, l.clamp(0, 2) as u32));
        check_depth(
            name,
            &bc,
            (address, pixel),
            (offset, layers, at),
            (None, None),
        );
    }
}

/// A linear comparison is an `r8Unorm` linear sample of the taps' passes as codes 0 and 255
/// (our probes, 8,640 lanes; our probe compares the border's depth as a
/// texel's), on `Depth32Float` and `Depth16Unorm` (whose reference and texels compare as codes):
/// every compare function clamping to the edge, `less` and `greater` against zero and white
/// borders, `less` with an offset, and `less_equal` at an array's layer, on every lane of the
/// interpreter, references equal to a texel's depth and NaN included. (Apple's compiler rejects a
/// constexpr pixel sampler that compares.)
#[test]
fn linear_comparisons_sum_their_passes_as_r8unorm_codes() {
    const WHITE: [f64; 4] = [1.0; 4];
    let mut cases = Vec::new();
    for f in 1..=8 {
        cases.push((f, (1, false, 0), Address::Edge, (0, 0), None));
    }
    for f in [1, 3] {
        cases.push((f, (0, false, 0), Address::Border([0.0; 4]), (0, 0), None));
        cases.push((f, (0, false, 2), Address::Border(WHITE), (0, 0), None));
    }
    cases.push((1, (1, false, 0), Address::Edge, (2, -1), None));
    cases.push((2, (1, false, 0), Address::Edge, (0, 0), Some(9)));
    for (f, (code, pixel, border), address, offset, layer) in cases {
        let name = format!(
            "compare {f} address {code} border {border} pixel {pixel} {offset:?} {layer:?}"
        );
        let words = depth_words(code, pixel, border, f);
        let Some(bc) = assemble(&name, &depth_kernel(Some(words), true, layer, offset)) else {
            return;
        };
        let (layers, at) = layer.map_or((1, 0), |l| (3, l.clamp(0, 2) as u32));
        check_depth(
            &name,
            &bc,
            (address, pixel),
            (offset, layers, at),
            (Some(f), None),
        );
    }
}

/// `MTLCompareFunction`'s raw values (a runtime sampler word's `COMPARE` field) with the AIR codes
/// of the same functions ([`depth_passes`]).
const RUNTIME_FUNCTIONS: [(u32, u64); 8] = [
    (0, 8),
    (1, 1),
    (2, 5),
    (3, 2),
    (4, 3),
    (5, 6),
    (6, 4),
    (7, 7),
];

/// A linear comparison through a runtime sampler slot compares each tap under the function its
/// word's `COMPARE` field states, as a constexpr sampler's does under its own, on every lane of the
/// interpreter: every function, clamping to the edge, to zero (beside a transparent and a white
/// border color, which it reads) and to a white border, pixel coordinates, and an array's layer.
#[test]
fn runtime_linear_comparisons_compare_by_the_words_function() {
    use SamplerAddressMode as M;
    use SamplerBorderColor as B;
    let mut cases = Vec::new();
    for (raw, code) in RUNTIME_FUNCTIONS {
        cases.push((
            raw,
            code,
            (M::ClampToEdge, B::TransparentBlack),
            false,
            None,
        ));
    }
    cases.push((3, 2, (M::ClampToZero, B::TransparentBlack), false, None));
    cases.push((4, 3, (M::ClampToZero, B::OpaqueWhite), false, None));
    cases.push((1, 1, (M::ClampToBorderColor, B::OpaqueWhite), false, None));
    cases.push((5, 6, (M::ClampToEdge, B::TransparentBlack), true, None));
    cases.push((2, 5, (M::ClampToEdge, B::TransparentBlack), false, Some(9)));
    for (raw, code, (mode, border), pixel, layer) in cases {
        let name = format!("runtime compare {raw} {mode:?} {border:?} pixel {pixel} {layer:?}");
        let Some(bc) = assemble(&name, &depth_kernel(None, true, layer, (0, 0))) else {
            return;
        };
        let word = runtime_word(mode, pixel, border) | raw << SamplerWord::COMPARE_SHIFT;
        // `ClampToZero` reads the border color as `ClampToBorderColor` does.
        let address = match (mode, border) {
            (M::ClampToEdge, _) => Address::Edge,
            (_, B::TransparentBlack) => Address::Border([0.0; 4]),
            _ => Address::Border([1.0; 4]),
        };
        let (layers, at) = layer.map_or((1, 0), |l: i32| (3, l.clamp(0, 2) as u32));
        check_depth(
            &name,
            &bc,
            (address, pixel),
            ((0, 0), layers, at),
            (Some(code), Some(word)),
        );
    }
}

/// A runtime sampler's `sample_compare` and `gather_compare` of a 2D depth texture or array compare
/// in the shader by the word's function on every committed profile without
/// `VkProperty::MetalSampler`. A gather fetches its texels, addressed by the word, where the
/// slot's format is measured, and gathers them without comparison elsewhere (no `Dref`
/// instruction, a texture slot that no comparison reads); a sample branches on the word's filter
/// between the one texel, fetched where the word states no mip filter and sampled without
/// comparison through the slot's second binding without comparison otherwise, and the filtered
/// comparison, whose device arm keeps its `Dref` sample. Both read the word's compare function,
/// coordinates, address modes and border color, and a sample its filter; only the sample notes
/// `DeviceDepthComparison`. On a profile with the property the device's comparison is Metal's
/// own: the module keeps its `Dref` instructions and requires the property. Each validates. (The
/// interpreter models no sample or gather; the runtime lane's `depth_compares` runs these
/// comparisons on each device against Metal's measured rule.)
#[test]
fn runtime_nearest_comparisons_and_comparison_gathers_compare_in_the_shader() {
    let profiles = crate::spirv::testing::committed_exact_profiles().expect("profiles");
    for (gather, layer) in [
        (false, None),
        (true, None),
        (false, Some(1)),
        (true, Some(1)),
    ] {
        let name = format!("runtime gather {gather} layer {layer:?}");
        let text = match gather {
            true => gather_compare_kernel(None, layer),
            false => depth_kernel(None, true, layer, (0, 0)),
        };
        let Some(bc) = assemble(&name, &text) else {
            return;
        };
        for (stem, p) in &profiles {
            let what = format!("{name} on {stem}");
            let a = compile(&bc, "k", p).unwrap_or_else(|e| panic!("{what}: {e}"));
            link_and_validate(&a, p).unwrap_or_else(|e| panic!("{what}: {e}"));
            let metal = p
                .fields()
                .enabled
                .properties
                .contains(&VkProperty::MetalSampler);
            let ops = opcodes(&a.module.words);
            let dref = ops.iter().any(|o| DREF_OPS.contains(o));
            let required = a
                .requirements
                .properties
                .contains(&VkProperty::MetalSampler);
            let compares = a.interface.slots.values().any(|slot| {
                matches!(slot, crate::pipeline::interface::SlotInterface::Texture(t) if t.depth_compare)
            });
            let fields = a.interface.read_keys().into_iter().find_map(|k| match k {
                SpecKey::SamplerState { fields, .. } => Some(fields),
                _ => None,
            });
            // A sample's filtered arm may compare on the device; its nearest mip filter's texel is
            // the device's read, which `DeviceTexelDenormals` names; a gather's texels
            // are fetched.
            assert_eq!(
                a.precision.contains(&PrecisionNote::DeviceDepthComparison),
                !metal && !gather,
                "{what}"
            );
            if metal {
                assert_eq!((dref, required, compares), (true, true, true), "{what}");
                continue;
            }
            assert_eq!(
                (dref, required, compares),
                (!gather, false, !gather),
                "{what}"
            );
            let fields = fields.unwrap_or_else(|| panic!("{what}: no sampler word"));
            assert_eq!(
                fields & SamplerWord::COMPARE,
                SamplerWord::COMPARE,
                "{what}"
            );
            let filters = SamplerWord::MIN_LINEAR | SamplerWord::MIP_FILTER;
            assert_eq!(fields & filters == filters, !gather, "{what}");
            // The fetch's addressing by the word, and its `OpImageFetch`.
            let addressing = SamplerWord::PIXEL_COORDINATES
                | SamplerWord::ADDRESS[0]
                | SamplerWord::ADDRESS[1]
                | SamplerWord::BORDER;
            assert_eq!(fields & addressing, addressing, "{what}");
            assert!(ops.contains(&95), "{what}");
            // The one-texel arm's sample without comparison: `OpImageSampleExplicitLod`.
            assert_eq!(ops.contains(&88), !gather, "{what}");
        }
        // A sample's two bindings of the slot: the filtered arm's with comparison, and the
        // one-texel arm's without.
        let (_, m) = typed_module(&bc, "k", &lavapipe()).expect("typed module");
        let mut compares: Vec<bool> = combined_slots(&m).iter().map(|(_, c)| *c).collect();
        compares.sort();
        let want = match gather {
            true => vec![false],
            false => vec![false, true],
        };
        assert_eq!(compares, want, "{name}");
    }
}

/// The words of a constexpr sampler with nearest filters and no mip filter, every axis addressed
/// by `address` (AIR codes, as [`sampler_words`]) with border color code `border`, comparing
/// under AIR code `compare`.
fn nearest_depth_words(address: u64, border: u64, compare: u64) -> u64 {
    sampler_words(0, 0, address, false, border) & !(0xf << 16) | compare << 16
}

/// A kernel gathering `depth2d<float>` slot 0 (`depth2d_array<float>` at layer `layer`) with
/// comparison through the constexpr sampler of `words` (runtime sampler slot 0 for `None`,
/// [`sampler_text`]) at buffer 0's lane `tid` (a `float4`: the coordinate, and the reference in
/// `z`), the four passes to buffer 1's lane `tid`.
fn gather_compare_kernel(words: Option<u64>, layer: Option<i32>) -> String {
    let SamplerText {
        global,
        param,
        operand,
        states,
        args,
        node,
        tid,
    } = sampler_text(words);
    let (dim, tex_ty, layer_arg, layer_ty) = match layer {
        Some(l) => (
            "2d_array",
            "depth2d_array<float, sample>",
            format!("i32 {l}, "),
            "i32, ",
        ),
        None => ("2d", "depth2d<float, sample>", String::new(), ""),
    };
    format!(
        r#"{global}define void @k(ptr addrspace(1) %t, ptr addrspace(1) %c, ptr addrspace(1) %out{param}, i32 %tid) {{
  %i = zext i32 %tid to i64
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %c, i64 %i
  %c4 = load <4 x float>, ptr addrspace(1) %p, align 16
  %uv = shufflevector <4 x float> %c4, <4 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %ref = extractelement <4 x float> %c4, i64 2
  %r = call {{ <4 x float>, i8 }} @air.gather_compare_depth_{dim}.f32(ptr addrspace(1) %t, ptr addrspace(2) {operand}, i32 1, <2 x float> %uv, {layer_arg}float %ref, i1 true, <2 x i32> zeroinitializer, i32 0)
  %v = extractvalue {{ <4 x float>, i8 }} %r, 0
  %q = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i64 %i
  store <4 x float> %v, ptr addrspace(1) %q, align 16
  ret void
}}

declare {{ <4 x float>, i8 }} @air.gather_compare_depth_{dim}.f32(ptr addrspace(1), ptr addrspace(2), i32, <2 x float>, {layer_ty}float, i1, <2 x i32>, i32)

!air.kernel = !{{!0}}
{states}!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{{args}}}
!3 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"{tex_ty}", !"air.arg_name", !"t"}}
!4 = !{{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"c"}}
!5 = !{{i32 2, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}}
!6 = {node}
!7 = !{{i32 {tid}, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}}
"#
    )
}

/// The opcodes of a SPIR-V module's instructions.
fn opcodes(words: &[u32]) -> Vec<u16> {
    let mut out = Vec::new();
    let mut i = 5;
    while i < words.len() {
        out.push((words[i] & 0xffff) as u16);
        i += ((words[i] >> 16) as usize).max(1);
    }
    out
}

/// A nearest comparison through a constexpr sampler with a nearest mip filter and a
/// `min_lod_clamp`, whose texels the device reads (the exceptions: the shader does not fetch beside
/// the clamp), is the shader's own comparison on every committed profile without
/// `VkProperty::MetalSampler` (no `Dref` instruction): it records the device's read,
/// `PrecisionNote::DeviceTexelDenormals`, and not `PrecisionNote::DeviceDepthComparison`, which
/// names the device's comparisons alone. On a profile with the property the device's comparison is
/// Metal's own, and neither is recorded. Each validates. (Without a mip filter Metal reads level 0
/// whatever the clamp, so the shader fetches that texel itself and records neither.)
#[test]
fn a_comparison_of_texels_the_device_reads_notes_the_read_alone() {
    let name = "a nearest comparison with a min_lod_clamp";
    let words = sampler_words(0, 1, 1, false, 0) & !(0xf << 16) | 2 << 16;
    let text = depth_kernel(Some(words), true, None, (0, 0)).replace(
        "i1 true, float 0.0, float 0.0, i32 0)",
        "i1 false, float 0.0, float 1.0, i32 0)",
    );
    assert!(text.contains("float 1.0, i32 0)"), "{name}");
    let Some(bc) = assemble(name, &text) else {
        return;
    };
    let profiles = crate::spirv::testing::committed_exact_profiles().expect("profiles");
    for (stem, p) in &profiles {
        let what = format!("{name} on {stem}");
        let a = compile(&bc, "k", p).unwrap_or_else(|e| panic!("{what}: {e}"));
        link_and_validate(&a, p).unwrap_or_else(|e| panic!("{what}: {e}"));
        let metal = p
            .fields()
            .enabled
            .properties
            .contains(&VkProperty::MetalSampler);
        let dref = opcodes(&a.module.words)
            .iter()
            .any(|o| DREF_OPS.contains(o));
        assert_eq!(dref, metal, "{what}");
        assert!(
            !a.precision.contains(&PrecisionNote::DeviceDepthComparison),
            "{what}"
        );
        assert_eq!(
            a.precision.contains(&PrecisionNote::DeviceTexelDenormals),
            !metal,
            "{what}"
        );
    }
}

/// `OpImageSampleDrefImplicitLod`, `OpImageSampleDrefExplicitLod`, and `OpImageDrefGather`.
const DREF_OPS: [u16; 3] = [89, 90, 97];

/// The shader's own comparisons: a nearest `sample_compare` and a `gather_compare` of a 2D depth
/// texture or array through a constexpr sampler read their texels without comparison on every
/// committed profile without `VkProperty::MetalSampler`: no `Dref` instruction, a texture slot that
/// no comparison reads (so a `Depth32Float` texture held with a twin is read through its
/// `R32_SFLOAT` image), and no `RestrictedDepthRange`; every compare function, clamping to the
/// edge, repeating, mirroring, and to a white border, and at an array's layer. On a profile with
/// the property the device's comparison is Metal's own: the module keeps its `Dref` instruction and
/// requires the property. Each validates. (The interpreter models no sample or gather; the runtime
/// lane's `depth_compares` runs these comparisons on each device against Metal's measured rule.)
#[test]
fn nearest_comparisons_and_comparison_gathers_compare_in_the_shader() {
    let mut cases = Vec::new();
    for f in 1..=8 {
        cases.push((f, false, (1, 0), None));
        cases.push((f, true, (1, 0), None));
    }
    for gather in [false, true] {
        for (address, border) in [(0, 2), (2, 0), (3, 0)] {
            cases.push((2, gather, (address, border), None));
        }
        cases.push((5, gather, (1, 0), Some(1)));
    }
    let profiles = crate::spirv::testing::committed_exact_profiles().expect("profiles");
    for (f, gather, (address, border), layer) in cases {
        let name = format!("compare {f} gather {gather} address {address} layer {layer:?}");
        let words = nearest_depth_words(address, border, f);
        let text = match gather {
            true => gather_compare_kernel(Some(words), layer),
            false => depth_kernel(Some(words), true, layer, (0, 0)),
        };
        let Some(bc) = assemble(&name, &text) else {
            return;
        };
        for (stem, p) in &profiles {
            let what = format!("{name} on {stem}");
            let a = compile(&bc, "k", p).unwrap_or_else(|e| panic!("{what}: {e}"));
            link_and_validate(&a, p).unwrap_or_else(|e| panic!("{what}: {e}"));
            let metal = p
                .fields()
                .enabled
                .properties
                .contains(&VkProperty::MetalSampler);
            let dref = opcodes(&a.module.words)
                .iter()
                .any(|o| DREF_OPS.contains(o));
            let required = a
                .requirements
                .properties
                .contains(&VkProperty::MetalSampler);
            let compares = a.interface.slots.values().any(|slot| {
                matches!(slot, crate::pipeline::interface::SlotInterface::Texture(t) if t.depth_compare)
            });
            assert_eq!((dref, required, compares), (metal, metal, metal), "{what}");
            if !metal {
                assert!(
                    !a.precision.contains(&PrecisionNote::RestrictedDepthRange),
                    "{what}"
                );
            }
            // Its texels are fetched at a level the function knows.
            assert!(
                !a.precision.contains(&PrecisionNote::DeviceDepthComparison),
                "{what}"
            );
        }
    }
}

/// A kernel comparing `depth2d<float>` slot 0 through the constexpr sampler of `words` at buffer
/// 0's lane `tid` (a `float4`: the coordinate, the reference in `z`, and the level in `w`) at that
/// level; buffer 1's lane `tid` is a `float4` whose `x` is the pass.
fn level_compare_kernel(words: u64) -> String {
    format!(
        r#"@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 {words}, i64 0], align 8

define void @k(ptr addrspace(1) %t, ptr addrspace(1) %c, ptr addrspace(1) %out, i32 %tid) {{
  %i = zext i32 %tid to i64
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %c, i64 %i
  %c4 = load <4 x float>, ptr addrspace(1) %p, align 16
  %uv = shufflevector <4 x float> %c4, <4 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %ref = extractelement <4 x float> %c4, i64 2
  %lod = extractelement <4 x float> %c4, i64 3
  %r = call {{ float, i8 }} @air.sample_compare_depth_2d.f32(ptr addrspace(1) %t, ptr addrspace(2) @__air_sampler_state, i32 1, <2 x float> %uv, float %ref, i1 true, <2 x i32> zeroinitializer, i1 true, float %lod, float 0.0, i32 0)
  %v = extractvalue {{ float, i8 }} %r, 0
  %v4 = insertelement <4 x float> zeroinitializer, float %v, i64 0
  %q = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i64 %i
  store <4 x float> %v4, ptr addrspace(1) %q, align 16
  ret void
}}

declare {{ float, i8 }} @air.sample_compare_depth_2d.f32(ptr addrspace(1), ptr addrspace(2), i32, <2 x float>, float, i1, <2 x i32>, i1, float, float, i32)

!air.kernel = !{{!0}}
!air.sampler_states = !{{!6}}
!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4, !5, !7}}
!3 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"depth2d<float, sample>", !"air.arg_name", !"t"}}
!4 = !{{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"c"}}
!5 = !{{i32 2, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}}
!6 = !{{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}}
!7 = !{{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}}
"#
    )
}

/// The pass bits (`1.0` or `0.0`) of reference `r` against depth tap `x` of `tex` under AIR
/// compare function code `f` ([`depth_passes`]): a stored texel's, or the border's depth, which
/// Metal compares as a texel's.
fn tap_pass(tex: &Texels, f: u64, r: f32, x: Tap) -> u32 {
    let d16 = matches!(family(tex.class), Family::Norm { .. });
    let (depth, code) = match x {
        Tap::Stored(s @ Stored::Code(c)) => (tex.fetched(s), Some(c)),
        Tap::Stored(s @ Stored::Value(_)) => (tex.fetched(s), None),
        Tap::Border(b) => (b as f32, d16.then_some((b * 65535.0) as i64)),
    };
    match depth_passes(f, r, (depth, code)) {
        true => 1.0f32.to_bits(),
        false => 0.0f32.to_bits(),
    }
}

/// The lanes of a comparison check on a `w`x`h` texture `tex`: [`coordinates`]', each with a
/// reference (a texel's depth on every third lane), and level `levels[k % levels.len()]`.
fn compare_lanes(tex: &Texels, w: i64, h: i64, levels: &[f32]) -> Vec<(f64, f64, f32, f32)> {
    let refs = [0.0f32, 0.5, 1.0, -0.25, 0.75, f32::NAN, 1.5, 0.375];
    let n = tex.t.len();
    coordinates(w, h)
        .iter()
        .enumerate()
        .map(|(k, (x, y))| {
            let r = match k % 3 {
                0 => tex.fetched(tex.t[(7 * k) % n][0]),
                _ => refs[k % refs.len()],
            };
            (*x, *y, r, levels[k % levels.len()])
        })
        .collect()
}

/// Runs comparison kernel `bc` (a nearest `sample_compare` at level 0, or a `gather_compare`
/// where `gather`) through a sampler addressed by `address`, comparing under AIR code `f`, with
/// constant offset `offset`, on an array of `layers` read at `layer`, on lavapipe for each of
/// [`DEPTHS`] on a 12x8 texture at [`compare_lanes`], against Metal's rule: the texel at `floor` of
/// Metal's position, or the gather's four at its quantized position in gather order, each compared
/// as a texel, the border's depth included. The shader fetches them through the one binding the
/// comparison declares, which does not compare; the interpreter models no device sample or gather,
/// so every lane reads the fetches.
fn check_compared(
    name: &str,
    bc: &[u8],
    (address, gather, f): (Address, bool, u64),
    (offset, layers, layer): ((i32, i32), u32, u32),
) {
    let p = lavapipe();
    let a = compile(bc, "k", &p).unwrap_or_else(|e| panic!("{name}: {e}"));
    link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{name}: {e}"));
    let dref = opcodes(&a.module.words)
        .iter()
        .any(|o| DREF_OPS.contains(o));
    assert!(!dref, "{name}: a Dref instruction");
    let (plan, m) = typed_module(bc, "k", &p).expect("typed module");
    let (w, h) = (12i64, 8i64);
    for (seed, pf) in DEPTHS.iter().enumerate() {
        let class = texel_format(*pf).0;
        let tex = Texels::random(class, w, h, seed as u64 + 47);
        let slots = combined_slots(&m);
        let compares: Vec<bool> = slots.iter().map(|(_, c)| *c).collect();
        assert_eq!(compares, [false], "{name}: the combined bindings");
        let images: Vec<Image> = slots
            .iter()
            .map(|&(slot, _)| tex.image(slot, 2, layers))
            .collect();
        let lanes = compare_lanes(&tex, w, h, &[0.0]);
        let bytes: Vec<u8> = lanes
            .iter()
            .flat_map(|(x, y, r, _)| {
                [(x / w as f64) as f32, (y / h as f64) as f32, *r, 0.0].map(f32::to_le_bytes)
            })
            .flatten()
            .collect();
        let res = resources(&plan, &[bytes, vec![0xab; 16 * lanes.len()]], class);
        let out = run_images(&m, &res, &images, lanes.len() as u32, false);
        let (ox, oy) = (i64::from(offset.0), i64::from(offset.1));
        for (lane, ((x, y, r, _), got)) in lanes.iter().zip(&out).enumerate() {
            let (sx, sy) = (
                seen(*x, w as f64, address, false),
                seen(*y, h as f64, address, false),
            );
            let want: Vec<u32> = match gather {
                true => {
                    let (i, j) = (split(sx).0 + ox, split(sy).0 + oy);
                    [(i, j + 1), (i + 1, j + 1), (i + 1, j), (i, j)]
                        .iter()
                        .map(|&at| tap_pass(&tex, f, *r, tap(&tex, address, layer, at, 0)))
                        .collect()
                }
                false => {
                    let at = (sx.floor() as i64 + ox, sy.floor() as i64 + oy);
                    vec![tap_pass(&tex, f, *r, tap(&tex, address, layer, at, 0))]
                }
            };
            assert_eq!(
                got[..want.len()],
                want[..],
                "{name} {pf:?} lane {lane} ({x}, {y}) reference {r}"
            );
        }
    }
}

/// A nearest `sample_compare` and a `gather_compare` of a 2D depth texture or array through a
/// constexpr sampler fetch their texels on a device without `VkProperty::MetalSampler`, so a
/// denormal texel is compared as Metal compares it, and compare as Metal does on every lane of the
/// interpreter: every compare function clamping to the edge, repeating, mirroring, to zero and to a
/// white border, an offset (a sample's), and an array's layer, on `Depth32Float` and
/// `Depth16Unorm`, references equal to a texel's depth and NaN included.
#[test]
fn fetched_comparisons_take_metals_texels() {
    const WHITE: [f64; 4] = [1.0; 4];
    let mut cases = Vec::new();
    for gather in [false, true] {
        for f in 1..=8 {
            cases.push((gather, f, (1, 0), Address::Edge, (0, 0), None));
        }
        cases.push((gather, 2, (2, 0), Address::Repeat, (0, 0), None));
        cases.push((gather, 4, (3, 0), Address::Mirror, (0, 0), None));
        cases.push((gather, 1, (0, 0), Address::Border([0.0; 4]), (0, 0), None));
        cases.push((gather, 3, (0, 2), Address::Border(WHITE), (0, 0), None));
        cases.push((gather, 2, (1, 0), Address::Edge, (0, 0), Some(9)));
    }
    cases.push((false, 5, (1, 0), Address::Edge, (2, -1), None));
    cases.push((false, 2, (2, 0), Address::Repeat, (-3, 1), None));
    for (gather, f, (code, border), address, offset, layer) in cases {
        let name = format!(
            "gather {gather} compare {f} address {code} border {border} {offset:?} {layer:?}"
        );
        let words = nearest_depth_words(code, border, f);
        let text = match gather {
            true => gather_compare_kernel(Some(words), layer),
            false => depth_kernel(Some(words), true, layer, offset),
        };
        let Some(bc) = assemble(&name, &text) else {
            return;
        };
        let (layers, at) = layer.map_or((1, 0), |l| (3, l.clamp(0, 2) as u32));
        check_compared(&name, &bc, (address, gather, f), (offset, layers, at));
    }
}

/// A nearest comparison under a nearest mip filter at a dynamic level fetches its texel from the
/// level Metal reads (`L + 1` where the level's fraction on sixty-fourths passes one half), clamped
/// into the texture's three levels, at Metal's position on that level, on every lane of the
/// interpreter.
#[test]
fn a_fetched_comparison_reads_metals_level() {
    let words = sampler_words(0, 1, 1, false, 0) & !(0xf << 16) | 2 << 16;
    let Some(bc) = assemble("levels", &level_compare_kernel(words)) else {
        return;
    };
    let p = lavapipe();
    let a = compile(&bc, "k", &p).unwrap_or_else(|e| panic!("levels: {e}"));
    link_and_validate(&a, &p).unwrap_or_else(|e| panic!("levels: {e}"));
    let (plan, m) = typed_module(&bc, "k", &p).expect("typed module");
    let snap = crate::lower::sampler::level::LevelSnap {
        mip: crate::air::abi::MipFilter::Nearest,
        min: 0.0f32.to_bits(),
        max: 65504.0f32.to_bits(),
    };
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
        1.53125,
        2.0,
        2.6,
        7.0,
        f32::NAN,
        f32::INFINITY,
    ];
    let (w, h) = (12i64, 8i64);
    for (seed, pf) in DEPTHS.iter().enumerate() {
        let class = texel_format(*pf).0;
        let texs: Vec<Texels> = (0..3)
            .map(|l| Texels::random(class, w >> l, h >> l, seed as u64 + 61 + l as u64))
            .collect();
        let slot = combined_slot(&m);
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
        let lanes = compare_lanes(&texs[0], w, h, &levels);
        let bytes: Vec<u8> = lanes
            .iter()
            .flat_map(|(x, y, r, l)| {
                [(x / w as f64) as f32, (y / h as f64) as f32, *r, *l].map(f32::to_le_bytes)
            })
            .flatten()
            .collect();
        let res = resources(&plan, &[bytes, vec![0xab; 16 * lanes.len()]], class);
        let out = run_images(&m, &res, &[image], lanes.len() as u32, false);
        for (lane, ((x, y, r, l), got)) in lanes.iter().zip(&out).enumerate() {
            let level = (snap.of(*l) as usize).min(2);
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
                "{pf:?} lane {lane} ({x}, {y}) level {l} (Metal's {level}) reference {r}"
            );
        }
    }
}

/// The emulation is fetches, integer and float arithmetic, and a branch on a specialization
/// constant, which every Vulkan 1.2 device runs: a 2D, a 1D, and a 3D linear `half` sample
/// compile and pass L3 on every committed profile.
#[test]
fn the_emulation_validates_on_every_committed_profile() {
    let white = linear(0, false, 2);
    let forms = [
        ("2d", Form::Plain { offset: (1, -1) }),
        ("1d", Form::OneD),
        ("3d", Form::ThreeD { offset: (1, 0, -1) }),
    ];
    for (name, form) in forms {
        let Some(bc) = assemble(name, &kernel(Some(white), form)) else {
            return;
        };
        for (stem, p) in crate::spirv::testing::committed_exact_profiles().expect("profiles") {
            let a = compile(&bc, "k", &p).unwrap_or_else(|e| panic!("{name} on {stem}: {e}"));
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{name} on {stem}: {e}"));
        }
    }
}

/// The plan's word estimate bounds the emitted module of every linear sample form, with a `half` or
/// `float` result, through a constexpr sampler or a runtime one, of the comparisons whose texels
/// the shader fetches, and of the public linear `half` fixtures and the runtime-sampler fixture on
/// every committed profile.
#[test]
fn the_linear_estimate_bounds_its_emission() {
    use crate::budget::SPIRV_WORDS;
    use crate::library::FunctionConstantValues;
    use crate::lower::select::LowerInputs;
    use crate::pipeline::compile::{compile_bitcode, plan_bitcode, EntryRef};
    use crate::pipeline::testing::{engine_dir, meter};
    let mut modules: Vec<(String, Vec<u8>, String)> = Vec::new();
    for (name, code, form, float) in [
        ("2d white offset", 0, Form::Plain { offset: (1, -1) }, false),
        ("2d mirror", 3, Form::Plain { offset: (0, 0) }, false),
        ("array", 1, Form::Array { layer: 1 }, false),
        ("1d white", 0, Form::OneD, false),
        (
            "2d float white offset",
            0,
            Form::Plain { offset: (1, -1) },
            true,
        ),
        ("1d float white", 0, Form::OneD, true),
        (
            "3d white offset",
            0,
            Form::ThreeD { offset: (1, -1, 2) },
            false,
        ),
        ("3d mirror", 3, Form::ThreeD { offset: (0, 0, 0) }, false),
        (
            "3d float white offset",
            0,
            Form::ThreeD { offset: (1, -1, 2) },
            true,
        ),
    ] {
        let k = kernel_of(Some(linear(code, false, 2)), form, float);
        let Some(bc) = assemble(name, &k) else {
            return;
        };
        modules.push((name.to_string(), bc, "k".to_string()));
    }
    for (name, form, float) in [
        ("2d runtime offset", Form::Plain { offset: (1, -1) }, false),
        ("array runtime", Form::Array { layer: 1 }, false),
        ("1d runtime", Form::OneD, false),
        ("2d float runtime", Form::Plain { offset: (0, 0) }, true),
        (
            "3d runtime offset",
            Form::ThreeD { offset: (1, -1, 2) },
            false,
        ),
        ("3d float runtime", Form::ThreeD { offset: (0, 0, 0) }, true),
    ] {
        let Some(bc) = assemble(name, &kernel_of(None, form, float)) else {
            return;
        };
        modules.push((name.to_string(), bc, "k".to_string()));
    }
    // Depth samples and comparisons.
    for (name, compare, layer) in [
        ("depth white offset", false, None),
        ("depth array", false, Some(1)),
        ("comparison white offset", true, None),
        ("comparison array", true, Some(1)),
    ] {
        let words = depth_words(0, false, 2, 1);
        let Some(bc) = assemble(name, &depth_kernel(Some(words), compare, layer, (1, -1))) else {
            return;
        };
        modules.push((name.to_string(), bc, "k".to_string()));
    }
    // Comparisons whose texels the shader fetches.
    let level_words = sampler_words(0, 1, 3, false, 2) & !(0xf << 16) | 2 << 16;
    for (name, text) in [
        (
            "fetched gather white",
            gather_compare_kernel(Some(nearest_depth_words(0, 2, 2)), None),
        ),
        (
            "fetched gather array mirror",
            gather_compare_kernel(Some(nearest_depth_words(3, 0, 2)), Some(1)),
        ),
        (
            "fetched nearest white offset",
            depth_kernel(Some(nearest_depth_words(0, 2, 2)), true, None, (1, -1)),
        ),
        ("fetched nearest level", level_compare_kernel(level_words)),
        // A runtime sampler's comparisons.
        (
            "runtime comparison offset",
            depth_kernel(None, true, None, (1, -1)),
        ),
        (
            "runtime comparison array",
            depth_kernel(None, true, Some(1), (0, 0)),
        ),
        ("runtime gather", gather_compare_kernel(None, None)),
        // A runtime sampler's comparison at the level its word reads, and a blend of two levels'
        // comparisons under a linear mip filter.
        (
            "runtime comparison level",
            levels::runtime_level_kernel(levels::Stated::Level),
        ),
        (
            "blended comparison level",
            level_compare_kernel(sampler_words(0, 2, 1, false, 0) & !(0xf << 16) | 2 << 16),
        ),
    ] {
        let Some(bc) = assemble(name, &text) else {
            return;
        };
        modules.push((name.to_string(), bc, "k".to_string()));
    }
    let Ok(bc) = llvm_assemble(&engine_dir().join("msf_runtime_samplers.ll")) else {
        eprintln!("SKIPPED: llvm-as");
        return;
    };
    for entry in [
        "msf_slot2d_linear_edge",
        "msf_slot2d_linear_zero_pixel",
        "msf_slot2d_linear_repeat_half",
        "msf_slot3d_linear_mirror",
        "msf_slot1d_linear_mirror",
        "msf_slotcube_linear",
    ] {
        modules.push((entry.to_string(), bc.clone(), entry.to_string()));
    }
    for name in [
        "hlf_half_forms",
        "hlf_half_address",
        "hlf_half_dims",
        "hlf_float_linear",
    ] {
        let Ok(bc) = llvm_assemble(&engine_dir().join(format!("{name}.ll"))) else {
            eprintln!("SKIPPED: llvm-as");
            return;
        };
        modules.push((name.to_string(), bc, name.to_string()));
    }
    for (name, bc, entry) in &modules {
        for (stem, p) in crate::spirv::testing::committed_exact_profiles().expect("profiles") {
            let what = format!("{name} on {stem}");
            let (entry, consts) = (
                EntryRef::Name(entry.as_str()),
                FunctionConstantValues::defaults(),
            );
            let inputs = LowerInputs::new();
            let plan = plan_bitcode(bc, entry, &consts, &p, &inputs, &mut meter())
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            let a = compile_bitcode(bc, entry, &consts, &p, &inputs, &mut meter(), SPIRV_WORDS)
                .unwrap_or_else(|e| panic!("{what}: {e}"));
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

/// The device of committed profile `stem`.
pub(super) fn device_of(stem: &str) -> crate::device::Device {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(&path).expect("profile");
    let desc =
        crate::device::DeviceDescription::from_khronos_json(&text, None).expect("description");
    crate::device::Device::new(desc, EXACT_SAMPLING).expect("device")
}

/// The kernel of bitcode `bc` compiled into a compute pipeline on `dev`.
pub(super) fn notes_pipeline(
    dev: &crate::device::Device,
    bc: &[u8],
) -> crate::pipeline::ComputePipeline {
    use crate::pipeline::{ComputePipelineDescriptor, FunctionDescriptor, StageSource};
    let lib = crate::library::Library::from_air(bc).expect("library");
    let f = lib.function("k").expect("function");
    let compiled = dev
        .compile_function(
            &FunctionDescriptor::new(f),
            &crate::lower::select::LowerInputs::new(),
            &crate::budget::Budget::standard(),
        )
        .expect("compile");
    dev.compile_compute_pipeline(
        &ComputePipelineDescriptor::new(StageSource::Compiled(compiled)),
        &crate::budget::Budget::standard(),
    )
    .expect("pipeline")
}

/// The notes of pipeline `p` on `dev` planned for texture slot 0 bound as a `ty` texture in `pf`,
/// buffers 0 to 2 bound, and sampler slot 0 bound to `sampler` (or nothing): its stage's notes
/// under the plan's specialization, where a test of the format bound at texture slot 0
/// (`SpecKey::BoundTexelFormat`), which no pipeline's key carries, reads `pf`'s class
/// (`metal::format::bound_texel_format`), as a runtime reads it at each draw or dispatch.
pub(super) fn notes_bound(
    p: &crate::pipeline::ComputePipeline,
    dev: &crate::device::Device,
    texture: (crate::metal::TextureType, PixelFormat),
    sampler: Option<crate::bind::SamplerDescriptor>,
) -> Vec<PrecisionNote> {
    let pf = texture.1;
    let plan = plan_bound(p, dev, texture, sampler);
    p.stage
        .precision_notes_bound(&plan.specialization, &|k| match k {
            SpecKey::BoundTexelFormat(SlotRef {
                class: SlotClass::Texture,
                index: 0,
                ..
            }) => Some(u64::from(crate::metal::format::bound_texel_format(pf))),
            k => plan.specialized_value(k),
        })
}

/// The plan of pipeline `p` on `dev` for texture slot 0 bound as a `ty` texture in `pf`, buffers
/// 0 to 2 bound, and sampler slot 0 bound to `sampler` (or nothing).
fn plan_bound(
    p: &crate::pipeline::ComputePipeline,
    dev: &crate::device::Device,
    (ty, pf): (crate::metal::TextureType, PixelFormat),
    sampler: Option<crate::bind::SamplerDescriptor>,
) -> crate::bind::BindingPlan {
    let mut facts = crate::bind::TextureFacts::new(ty, pf);
    facts.usage = crate::metal::TextureUsage::SHADER_READ;
    plan_facts(p, dev, facts, sampler)
}

/// [`plan_bound`] with texture slot 0 bound to a texture of `facts`.
fn plan_facts(
    p: &crate::pipeline::ComputePipeline,
    dev: &crate::device::Device,
    facts: crate::bind::TextureFacts,
    sampler: Option<crate::bind::SamplerDescriptor>,
) -> crate::bind::BindingPlan {
    use crate::bind::{BindState, BoundBuffer, BoundSampler, BoundTexture, StageTable};
    let pf = facts.pixel_format;
    let mut t = StageTable::default();
    t.textures.insert(
        0,
        BoundTexture::Texture {
            facts,
            resource_id: None,
        },
    );
    for (slot, address) in [(0u32, 0x1_0000u64), (1, 0x2_0000), (2, 0x3_0000)] {
        t.buffers.insert(
            slot,
            BoundBuffer::Buffer {
                length: 4096,
                offset: 0,
                gpu_address: address,
                device_address: None,
                attribute_stride: None,
            },
        );
    }
    if let Some(d) = sampler {
        t.samplers.insert(0, BoundSampler::new(d));
    }
    let mut s = BindState::default();
    s.stages.insert(Stage::Kernel, t);
    p.plan(dev, &s.shape())
        .unwrap_or_else(|e| panic!("plan {pf:?}: {e}"))
}

/// A texture slot's format class is data the function reads from its push field, so no format
/// splits a pipeline: every format gives one specialization, through any sampler, on MoltenVK as on
/// lavapipe, and the binder pushes each format's class.
#[test]
fn formats_share_a_specialization_since_the_class_is_data() {
    use crate::metal::TextureType;
    let formats = [
        PixelFormat::RG8Unorm,
        PixelFormat::RGBA8Unorm,
        PixelFormat::A8Unorm,
        PixelFormat::RGBA16Float,
        PixelFormat::RGBA16Unorm,
    ];
    for (stem, sampler) in [
        ("moltenvk-m3", linear(1, false, 0)),
        ("lavapipe", linear(0, false, 2)),
    ] {
        let dev = device_of(stem);
        let Some(bc) = assemble(stem, &kernel(Some(sampler), Form::Plain { offset: (0, 0) }))
        else {
            return;
        };
        let p = notes_pipeline(&dev, &bc);
        assert!(
            !p.stage
                .specialization
                .constants
                .iter()
                .any(|c| matches!(c.key, SpecKey::TexelFormat(_))),
            "{stem}"
        );
        let plan = |pf| plan_bound(&p, &dev, (TextureType::Type2D, pf), None);
        let base = plan(formats[0]).specialization;
        for pf in formats {
            let plan = plan(pf);
            assert_eq!(plan.specialization, base, "{stem} {pf:?}");
            assert!(
                plan.specialized_value(SpecKey::TexelFormat(TEXTURE))
                    .is_some(),
                "{stem} {pf:?}"
            );
        }
    }
}

/// On MoltenVK, a device that samples through Metal's own texture unit and rounds to `half` as
/// Metal does, a `half` sample through a runtime sampler reads of the sampler word only what the
/// pixel-centre test reads, which the binder keys as 0 for a sampler of normalized coordinates; its
/// slot's class it reads as data: every measured format specializes alike through any sampler of
/// normalized coordinates, as through none, and through each pixel sampler; an unmeasured format
/// apart, through the slot's `UnmeasuredTexels` specialization, under which alone its `half`
/// conversion is noted. On lavapipe every format specializes alike too, and the samplers apart.
#[test]
fn metal_s_half_of_its_float_specializes_measured_formats_alike() {
    use crate::bind::SamplerDescriptor;
    use crate::metal::{SamplerMinMagFilter, SamplerMipFilter, TextureType};
    use PixelFormat as F;
    let m3 = device_of("moltenvk-m3");
    let Some(bc) = assemble(
        "metal sampler fold",
        &kernel(None, Form::Plain { offset: (0, 0) }),
    ) else {
        return;
    };
    let p = notes_pipeline(&m3, &bc);
    let constants = &p.stage.specialization.constants;
    // The slot's class is a push field, no constant.
    assert!(
        !constants
            .iter()
            .any(|c| matches!(c.key, SpecKey::TexelFormat(_))),
        "{constants:?}"
    );
    assert!(p
        .interface
        .push
        .iter()
        .flat_map(|pl| &pl.fields)
        .any(|f| matches!(f.field, PushField::Specialized(SpecKey::TexelFormat(_)))));
    // The pixel-centre test reads whether the word states a nearest pixel sampler, and nothing else
    // reads it: its fields are gated by the coordinates bit; nothing reads its mip filter or
    // anisotropy.
    let word = constants
        .iter()
        .find(|c| matches!(c.key, SpecKey::SamplerState { .. }))
        .expect("the sampler's word");
    let SpecKey::SamplerState { fields, .. } = word.key else {
        unreachable!("a sampler word")
    };
    assert_ne!(fields & SamplerWord::PIXEL_GATED, 0, "{word:?}");
    let read = fields & !(word.unread_bits as u32) & !SamplerWord::PIXEL_GATED;
    assert_eq!(read & !SamplerWord::PIXEL_NEAREST, 0, "{word:?}");
    assert!(
        constants
            .iter()
            .any(|c| matches!(c.key, SpecKey::UnmeasuredTexels(_))),
        "{constants:?}"
    );
    let nearest = SamplerDescriptor::new();
    let mut linear = nearest;
    linear.min_filter = SamplerMinMagFilter::Linear;
    linear.mag_filter = SamplerMinMagFilter::Linear;
    let mut mipped = linear;
    mipped.mip_filter = SamplerMipFilter::Linear;
    mipped.max_anisotropy = 4;
    let mut pixel = nearest;
    pixel.normalized_coordinates = false;
    let mut pixel_linear = linear;
    pixel_linear.normalized_coordinates = false;
    let spec = |pf, s| plan_bound(&p, &m3, (TextureType::Type2D, pf), Some(s)).specialization;
    // Every sampler of normalized coordinates keys as no sampler does; a pixel sampler apart,
    // a nearest one from a linear one.
    let base = spec(F::RGBA8Unorm, nearest);
    assert_eq!(spec(F::RGBA8Unorm, linear), base);
    assert_eq!(spec(F::RGBA8Unorm, mipped), base);
    assert_ne!(spec(F::RGBA8Unorm, pixel), base);
    assert_ne!(spec(F::RGBA8Unorm, pixel_linear), base);
    assert_ne!(
        spec(F::RGBA8Unorm, pixel_linear),
        spec(F::RGBA8Unorm, pixel)
    );
    let alike = [
        F::RGBA8Unorm,
        F::BGRA8Unorm,
        F::R8Unorm,
        F::A8Unorm,
        F::RGBA16Float,
        F::RGB10A2Unorm,
        F::B5G6R5Unorm,
        F::RG11B10Float,
        F::RGBA8Snorm,
        F::RG8Snorm,
        F::RGBA16Unorm,
        F::RGBA16Snorm,
        F::RGBA32Float,
    ];
    for pf in alike {
        for s in [nearest, linear, mipped] {
            assert_eq!(spec(pf, s), base, "{pf:?} {s:?}");
        }
        assert_eq!(spec(pf, pixel), spec(F::RGBA8Unorm, pixel), "{pf:?}");
    }
    assert_ne!(spec(F::BC7RGBAUnorm, nearest), base);
    let converts = |pf, s| {
        notes_bound(&p, &m3, (TextureType::Type2D, pf), Some(s))
            .contains(&PrecisionNote::HalfTexelConversion)
    };
    for s in [nearest, linear] {
        assert!(converts(F::BC7RGBAUnorm, s), "{s:?}");
        for pf in alike {
            assert!(!converts(pf, s), "{pf:?} {s:?}");
        }
    }
    // On lavapipe the emulated linear filter reads the word's filter for normalized coordinates
    // too: no gate, and the samplers stay apart; the formats, read as data, key alike.
    let lvp = device_of("lavapipe");
    let p = notes_pipeline(&lvp, &bc);
    assert!(p
        .stage
        .specialization
        .constants
        .iter()
        .all(|c| match c.key {
            SpecKey::SamplerState { fields, .. } => fields & SamplerWord::PIXEL_GATED == 0,
            _ => true,
        }));
    let spec = |pf, s| plan_bound(&p, &lvp, (TextureType::Type2D, pf), Some(s)).specialization;
    assert_eq!(spec(F::RGBA8Unorm, nearest), spec(F::RGBA16Float, nearest));
    assert_ne!(spec(F::RGBA8Unorm, nearest), spec(F::RGBA8Unorm, linear));
}

/// A texture access's notes are the specialized pipeline's. A `half` linear 2D sample through a
/// constexpr sampler on lavapipe records `HalfSampleFilter` and `HalfTexelConversion` only where
/// the bound format's class names no measured family (a compressed format), the arms its
/// specialization selects: none of them under an 8-bit, 16-bit, half-float, 32-bit float, or packed
/// format. In 3D a half-float, small-float, or 32-bit float format keeps the filter note (the
/// residue), a normalized or packed normalized one does not. Through a runtime sampler slot the
/// filter note holds where the bound sampler filters and its state is not the emulation's (in 2D a
/// mirror-clamped axis is) or its format is not measured, `PixelSamplerScaling` where it reads
/// pixel coordinates and that arm is the device's filter (a nearest pixel word reads its texel's
/// centre, a linear one the emulation where it reaches the format, both at Metal's position), and
/// `HalfTexelConversion` for a format of no family where it filters, and for one of neither family
/// nor packed field also where it does not (a packed format's one texel converts from its codes).
/// The module's notes are the union, every one of them conditional. On MoltenVK the filter is
/// Metal's own and takes no note at all.
#[test]
fn a_pipelines_texture_notes_follow_its_bound_format_and_sampler() {
    use crate::bind::SamplerDescriptor;
    use crate::metal::{SamplerMinMagFilter, TextureType};
    use PrecisionNote::{HalfSampleFilter, HalfTexelConversion, PixelSamplerScaling};
    let lvp = device_of("lavapipe");
    let plane = (TextureType::Type2D, PixelFormat::RGBA8Unorm);
    let measured = [
        PixelFormat::RGBA8Unorm,
        PixelFormat::RG16Snorm,
        PixelFormat::RGBA16Float,
        PixelFormat::RGBA32Float,
        PixelFormat::RGB10A2Unorm,
        PixelFormat::B5G6R5Unorm,
        PixelFormat::RG11B10Float,
    ];
    let conversion = [HalfSampleFilter, HalfTexelConversion];
    // A constexpr linear sampler.
    let Some(bc) = assemble(
        "notes plane",
        &kernel(Some(linear(1, false, 0)), Form::Plain { offset: (0, 0) }),
    ) else {
        return;
    };
    let p = notes_pipeline(&lvp, &bc);
    for note in conversion {
        assert!(p.stage.precision_notes.contains(&note), "{note:?}");
        assert!(
            p.stage.specialized_notes.iter().any(|s| s.note == note),
            "{note:?}: {:?}",
            p.stage.specialized_notes
        );
    }
    for pf in measured {
        let notes = notes_bound(&p, &lvp, (TextureType::Type2D, pf), None);
        for note in conversion {
            assert!(!notes.contains(&note), "{pf:?}: {notes:?}");
        }
    }
    let unmeasured = notes_bound(
        &p,
        &lvp,
        (TextureType::Type2D, PixelFormat::BC7RGBAUnorm),
        None,
    );
    for note in conversion {
        assert!(unmeasured.contains(&note), "{note:?}: {unmeasured:?}");
    }
    // In 3D the emulation's half-float and 32-bit float sums keep a residue.
    let Some(bc) = assemble(
        "notes solid",
        &kernel(
            Some(linear(1, false, 0)),
            Form::ThreeD { offset: (0, 0, 0) },
        ),
    ) else {
        return;
    };
    let solid = notes_pipeline(&lvp, &bc);
    for (pf, noted) in [
        (PixelFormat::RGBA8Unorm, false),
        (PixelFormat::RGBA16Unorm, false),
        (PixelFormat::RGB10A2Unorm, false),
        (PixelFormat::RGBA16Float, true),
        (PixelFormat::RGBA32Float, true),
        (PixelFormat::RG11B10Float, true),
    ] {
        let notes = notes_bound(&solid, &lvp, (TextureType::Type3D, pf), None);
        assert_eq!(notes.contains(&HalfSampleFilter), noted, "3D {pf:?}");
        assert!(!notes.contains(&HalfTexelConversion), "3D {pf:?}");
    }
    // A runtime sampler slot: the bound sampler's word decides.
    let Some(bc) = assemble(
        "notes runtime",
        &kernel(None, Form::Plain { offset: (0, 0) }),
    ) else {
        return;
    };
    let runtime = notes_pipeline(&lvp, &bc);
    for note in [HalfSampleFilter, PixelSamplerScaling] {
        assert!(
            runtime
                .stage
                .specialized_notes
                .iter()
                .any(|s| s.note == note),
            "{note:?}: {:?}",
            runtime.stage.specialized_notes
        );
    }
    let nearest = SamplerDescriptor::new();
    let mut filtered = SamplerDescriptor::new();
    filtered.min_filter = SamplerMinMagFilter::Linear;
    filtered.mag_filter = SamplerMinMagFilter::Linear;
    let mut mirrored = filtered;
    mirrored.s_address_mode = SamplerAddressMode::MirrorClampToEdge;
    let mut pixels = filtered;
    pixels.normalized_coordinates = false;
    let mut pixels_nearest = nearest;
    pixels_nearest.normalized_coordinates = false;
    let mut pixels_mixed = pixels;
    pixels_mixed.mag_filter = SamplerMinMagFilter::Nearest;
    for (name, sampler, pf, filter, scaled) in [
        ("nearest", nearest, PixelFormat::RGBA8Unorm, false, false),
        (
            "nearest packed",
            nearest,
            PixelFormat::RGB10A2Unorm,
            false,
            false,
        ),
        ("linear", filtered, PixelFormat::RGBA8Unorm, false, false),
        (
            "linear packed",
            filtered,
            PixelFormat::RGB10A2Unorm,
            false,
            false,
        ),
        (
            "nearest unmeasured",
            nearest,
            PixelFormat::BC7RGBAUnorm,
            false,
            false,
        ),
        (
            "linear unmeasured",
            filtered,
            PixelFormat::BC7RGBAUnorm,
            true,
            false,
        ),
        (
            "mirror-clamped",
            mirrored,
            PixelFormat::RGBA8Unorm,
            false,
            false,
        ),
        ("pixels", pixels, PixelFormat::RGBA8Unorm, false, false),
        (
            "pixels unmeasured",
            pixels,
            PixelFormat::BC7RGBAUnorm,
            true,
            true,
        ),
        (
            "pixels nearest",
            pixels_nearest,
            PixelFormat::RGBA8Unorm,
            false,
            false,
        ),
        (
            "pixels mixed",
            pixels_mixed,
            PixelFormat::RGBA8Unorm,
            true,
            true,
        ),
    ] {
        let notes = notes_bound(&runtime, &lvp, (TextureType::Type2D, pf), Some(sampler));
        assert_eq!(
            notes.contains(&HalfSampleFilter),
            filter,
            "{name}: {notes:?}"
        );
        assert_eq!(
            notes.contains(&PixelSamplerScaling),
            scaled,
            "{name}: {notes:?}"
        );
        // A packed format's one texel converts from its codes, and Metal's filter of it is
        // measured: the conversion note holds for a format of neither.
        assert_eq!(
            notes.contains(&HalfTexelConversion),
            pf == PixelFormat::BC7RGBAUnorm,
            "{name}: {notes:?}"
        );
    }
    // MoltenVK filters through Metal's own sampler: no filter note to decide.
    let Some(bc) = assemble(
        "notes plane m3",
        &kernel(Some(linear(1, false, 0)), Form::Plain { offset: (0, 0) }),
    ) else {
        return;
    };
    let m3 = device_of("moltenvk-m3");
    let q = notes_pipeline(&m3, &bc);
    assert!(!q.stage.precision_notes.contains(&HalfSampleFilter));
    let notes = notes_bound(&q, &m3, plane, None);
    assert!(
        !notes.contains(&HalfTexelConversion) && !notes.contains(&HalfSampleFilter),
        "{notes:?}"
    );
    // A runtime pixel sampler there: a nearest word reads its texel's centre and a linear one
    // snaps onto its footprint cell, both Metal's position; a word of mixed filters is the
    // device's.
    let Some(bc) = assemble(
        "notes runtime m3",
        &kernel(None, Form::Plain { offset: (0, 0) }),
    ) else {
        return;
    };
    let q = notes_pipeline(&m3, &bc);
    for (name, sampler, scaled) in [
        ("pixels", pixels, false),
        ("pixels nearest", pixels_nearest, false),
        ("pixels mixed", pixels_mixed, true),
        ("normalized", filtered, false),
    ] {
        let notes = notes_bound(&q, &m3, plane, Some(sampler));
        assert_eq!(
            notes.contains(&PixelSamplerScaling),
            scaled,
            "m3 {name}: {notes:?}"
        );
    }
}

/// An access that reads one texel converts its texel from its codes. Through a nearest constexpr
/// sampler on lavapipe a `half` sample of a packed format takes no `HalfTexelConversion` (its
/// class's packed arm), and neither does a normalized one; through a linear sampler neither does a
/// packed format (Metal's filter of it is emulated), and a compressed one keeps it. A `float` read
/// or nearest sample declares the slot's `TexelFormat` and the module's `float` routine where the
/// device's sampler is not Metal's own, and not on MoltenVK.
#[test]
fn a_texel_read_as_one_texel_converts_from_its_codes() {
    use crate::metal::TextureType;
    use crate::pipeline::sym::SpecKey;
    use PrecisionNote::HalfTexelConversion;
    let lvp = device_of("lavapipe");
    let m3 = device_of("moltenvk-m3");
    let nearest = sampler_words(0, 0, 1, false, 0);
    let plane = Form::Plain { offset: (0, 0) };
    let Some(bc) = assemble("one texel half", &kernel(Some(nearest), plane)) else {
        return;
    };
    let p = notes_pipeline(&lvp, &bc);
    for pf in [
        PixelFormat::RGB10A2Unorm,
        PixelFormat::B5G6R5Unorm,
        PixelFormat::RG11B10Float,
        PixelFormat::RGBA8Unorm,
    ] {
        let notes = notes_bound(&p, &lvp, (TextureType::Type2D, pf), None);
        assert!(
            !notes.contains(&HalfTexelConversion),
            "nearest {pf:?}: {notes:?}"
        );
    }
    let Some(bc) = assemble("filtered half", &kernel(Some(linear(1, false, 0)), plane)) else {
        return;
    };
    let p = notes_pipeline(&lvp, &bc);
    for (pf, noted) in [
        (PixelFormat::RGB10A2Unorm, false),
        (PixelFormat::RG11B10Float, false),
        (PixelFormat::BC7RGBAUnorm, true),
    ] {
        let notes = notes_bound(&p, &lvp, (TextureType::Type2D, pf), None);
        assert_eq!(
            notes.contains(&HalfTexelConversion),
            noted,
            "linear {pf:?}: {notes:?}"
        );
    }
    let Some(bc) = assemble("one texel float", &kernel_of(Some(nearest), plane, true)) else {
        return;
    };
    for (stem, dev, converts) in [("lavapipe", &lvp, true), ("moltenvk-m3", &m3, false)] {
        let p = notes_pipeline(dev, &bc);
        // The class, which the function reads from its push field.
        let class = p
            .interface
            .push
            .iter()
            .flat_map(|placement| &placement.fields)
            .any(|f| matches!(f.field, PushField::Specialized(SpecKey::TexelFormat(_))));
        let routine = p
            .stage
            .strategies
            .entries
            .contains_key(crate::lower::texture::float_texel::CENSUS_NAME);
        assert_eq!((class, routine), (converts, converts), "{stem}");
    }
}

/// Each function of SPIR-V module `words`: its result id, its function control, and the callees
/// of its `OpFunctionCall`s, with the result ids of those calls an `OpSelect` reads.
struct ModuleFunctions {
    controls: BTreeMap<u32, u32>,
    calls: BTreeMap<u32, Vec<u32>>,
    selected_calls: usize,
}

fn module_functions(words: &[u32]) -> ModuleFunctions {
    let (op_function, op_end, op_call, op_select) = (
        crate::spv::Op::Function as u32,
        crate::spv::Op::FunctionEnd as u32,
        crate::spv::Op::FunctionCall as u32,
        crate::spv::Op::Select as u32,
    );
    let mut out = ModuleFunctions {
        controls: BTreeMap::new(),
        calls: BTreeMap::new(),
        selected_calls: 0,
    };
    let mut current = None;
    let mut call_results = std::collections::BTreeSet::new();
    let mut i = 5;
    while i < words.len() {
        let (len, opcode) = ((words[i] >> 16) as usize, words[i] & 0xffff);
        assert!(len > 0 && i + len <= words.len(), "malformed module");
        let w = &words[i..i + len];
        if opcode == op_function {
            current = Some(w[2]);
            out.controls.insert(w[2], w[3]);
        } else if opcode == op_end {
            current = None;
        } else if opcode == op_call {
            let f = current.expect("a call outside a function");
            out.calls.entry(f).or_default().push(w[3]);
            call_results.insert(w[2]);
        } else if opcode == op_select && w[4..].iter().any(|v| call_results.contains(v)) {
            out.selected_calls += 1;
        }
        i += len;
    }
    out
}

impl ModuleFunctions {
    /// The entry point: the one function no other function calls.
    fn entry(&self, what: &str) -> u32 {
        let called: std::collections::BTreeSet<u32> =
            self.calls.values().flatten().copied().collect();
        let entries: Vec<u32> = self
            .controls
            .keys()
            .copied()
            .filter(|f| !called.contains(f))
            .collect();
        let [entry] = entries[..] else {
            panic!("{what}: one entry, {entries:?}");
        };
        entry
    }
}

/// A linear `half` sample through a runtime sampler, emulated on a device that does not sample
/// through Metal's own sampler, reaches its exact and accumulated sums from the two arms of a
/// branch on the class, so one of them runs: no `OpSelect` reads a call's result. The emulation is
/// its shape's sampling routine, which the entry calls; the routine calls the two sums, which are
/// declared `DontInline` on every such profile. The routine states `DontInline` where the driver is
/// measured to keep such a call a call (`DriverWorkaround::DontInlineKeepsCalls`, NVIDIA's
/// 615.71.09 on the RTX 3090 and the RTX 5080 Laptop GPU) and no control elsewhere: on Mesa's
/// drivers, which inline every function whatever it states, and on a GPU not measured (the desktop
/// RTX 5080's profile, `nvidia-rtx50-linux`, on the same release). The
/// texel conversion the device's sample calls, which its class specialization folds where it is
/// inlined, and the entry take no control.
#[test]
fn the_linear_sums_run_one_by_a_branch_and_stay_calls() {
    let Some(bc) = assemble("kept sums", &kernel(None, Form::Plain { offset: (0, 0) })) else {
        return;
    };
    let profiles = [
        ("rtx3090", rtx3090(), true),
        (
            "nvidia-rtx5080-laptop",
            profile("nvidia-rtx5080-laptop", EXACT_SAMPLING).expect("profile"),
            true,
        ),
        (
            "nvidia-rtx50-linux",
            profile("nvidia-rtx50-linux", EXACT_SAMPLING).expect("profile"),
            false,
        ),
        (
            "lavapipe",
            profile("lavapipe", EXACT_SAMPLING).expect("profile"),
            false,
        ),
        (
            "amd-rdna3-radv",
            profile("amd-rdna3-radv", EXACT_SAMPLING).expect("profile"),
            false,
        ),
    ];
    for (stem, p, keeps) in profiles {
        let a = compile(&bc, "k", &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        let fs = module_functions(&a.module.words);
        let entry = fs.entry(stem);
        let from_entry: std::collections::BTreeSet<u32> = fs
            .calls
            .get(&entry)
            .map(|c| c.iter().copied().collect())
            .unwrap_or_default();
        assert_eq!(
            from_entry.len(),
            2,
            "{stem}: the entry calls the emulation's routine and the conversion"
        );
        // The emulation's routine is the one the entry calls that calls the sums.
        let routines: Vec<u32> = from_entry
            .iter()
            .copied()
            .filter(|f| fs.calls.contains_key(f))
            .collect();
        let [routine] = routines[..] else {
            panic!("{stem}: one routine that calls, {routines:?}");
        };
        let sums: std::collections::BTreeSet<u32> = fs.calls[&routine].iter().copied().collect();
        assert_eq!(
            sums.len(),
            2,
            "{stem}: the routine calls the exact sum and the accumulated sum"
        );
        for f in &sums {
            assert_eq!(fs.controls.get(f), Some(&0x2), "{stem}: a sum stays a call");
        }
        assert_eq!(
            fs.controls.get(&routine),
            Some(&if keeps { 0x2 } else { 0 }),
            "{stem}: the sampling routine's control is the driver's fact"
        );
        for f in from_entry.iter().filter(|f| **f != routine) {
            assert_eq!(
                fs.controls.get(f),
                Some(&0),
                "{stem}: the conversion takes no control"
            );
        }
        assert_eq!(
            fs.controls.get(&entry),
            Some(&0),
            "{stem}: the entry takes no control"
        );
        assert_eq!(
            fs.selected_calls, 0,
            "{stem}: no select reads a sum's result"
        );
    }
}

/// The profile of the RTX 5080 Laptop GPU's committed description with the RTX 3090's `deviceID`:
/// NVIDIA's 615.71.09 on the RTX 3090, a GPU `DriverWorkaround::DontInlineKeepsCalls` is measured
/// on.
fn rtx3090() -> DeviceProfile {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles/nvidia-rtx5080-laptop.json");
    let text = std::fs::read_to_string(&path).expect("the RTX 5080 Laptop GPU's profile");
    let mut desc = crate::device::DeviceDescription::from_khronos_json(&text, None)
        .expect("the RTX 5080 Laptop GPU's description");
    desc.device_id = crate::device::driver_facts::NVIDIA_RTX_3090;
    let device = crate::device::Device::new(desc, EXACT_SAMPLING).expect("device");
    device.profile().clone()
}

/// [`kernel_of`] through the runtime `[[sampler(0)]]` with `n` samples of texture slot 0 at the
/// coordinate plus `k / 8` for sample `k`, their texels summed; where `offsets`, sample `k` at
/// the constant texel offset `(k % 4 + 1, -k)`, each other sample's.
fn sites_kernel(n: usize, float: bool, offsets: bool) -> String {
    let k = kernel_of(None, Form::Plain { offset: (0, 0) }, float);
    let vec4 = if float { "<4 x float>" } else { "<4 x half>" };
    let call = k
        .lines()
        .find(|l| l.starts_with("  %r = call"))
        .expect("the sample's call")
        .to_string();
    let extract = format!("  %v = extractvalue {{ {vec4}, i8 }} %r, 0");
    assert!(k.contains(&extract), "the sample's texel");
    let mut body = String::new();
    for i in 0..n {
        let d = i as f32 / 8.0;
        body.push_str(&format!(
            "  %uv{i} = fadd <2 x float> %uv, <float {d:?}, float {d:?}>\n"
        ));
        let (ox, oy) = match offsets {
            true => ((i % 4) as i32 + 1, -(i as i32)),
            false => (0, 0),
        };
        body.push_str(
            &call
                .replacen("%r =", &format!("%r{i} ="), 1)
                .replacen("<2 x float> %uv,", &format!("<2 x float> %uv{i},"), 1)
                .replacen(
                    "<2 x i32> <i32 0, i32 0>",
                    &format!("<2 x i32> <i32 {ox}, i32 {oy}>"),
                    1,
                ),
        );
        body.push('\n');
        body.push_str(&format!(
            "  %v{i} = extractvalue {{ {vec4}, i8 }} %r{i}, 0\n"
        ));
    }
    let mut sum = "%v0".to_string();
    for i in 1..n {
        body.push_str(&format!("  %s{i} = fadd {vec4} {sum}, %v{i}\n"));
        sum = format!("%s{i}");
    }
    body.push_str(&format!("  %v = fadd {vec4} {sum}, zeroinitializer"));
    k.replacen(&format!("{call}\n{extract}"), &body, 1)
}

/// The `OpImageFetch` instructions of each function of SPIR-V module `words`, by result id.
fn fetches_by_function(words: &[u32]) -> BTreeMap<u32, usize> {
    let (op_function, op_end, op_fetch) = (
        crate::spv::Op::Function as u32,
        crate::spv::Op::FunctionEnd as u32,
        crate::spv::Op::ImageFetch as u32,
    );
    let mut out = BTreeMap::new();
    let mut current = None;
    let mut i = 5;
    while i < words.len() {
        let (len, opcode) = ((words[i] >> 16) as usize, words[i] & 0xffff);
        assert!(len > 0 && i + len <= words.len(), "malformed module");
        if opcode == op_function {
            current = Some(words[i + 2]);
            out.insert(words[i + 2], 0);
        } else if opcode == op_end {
            current = None;
        } else if opcode == op_fetch {
            *out.entry(current.expect("a fetch outside a function"))
                .or_default() += 1;
        }
        i += len;
    }
    out
}

/// The emulation of one combined binding's sample shape is one sampling routine, which each of the
/// binding's sites calls. A kernel of `n` linear samples of one texture through one runtime
/// sampler, emulated on a device that does not sample through Metal's own sampler, holds one
/// emulation routine that the entry calls `n` times (a `float` result's nearest-exact fetch arm one
/// more routine, called `n` times too), and every fetch is the routines': the entry fetches
/// nothing. The census counts each routine once, as one sample's module counts it. Sites at
/// different constant texel offsets call one routine too, which takes the offset as its call's
/// argument.
#[test]
fn the_samples_of_one_binding_call_one_routine() {
    const N: usize = 5;
    for (float, offsets) in [(false, false), (true, false), (false, true), (true, true)] {
        let what = match (float, offsets) {
            (false, false) => "half",
            (true, false) => "float",
            (false, true) => "half at offsets",
            (true, true) => "float at offsets",
        };
        let Some(bc) = assemble(&format!("sites {what}"), &sites_kernel(N, float, offsets)) else {
            return;
        };
        let Some(one) = assemble(&format!("site {what}"), &sites_kernel(1, float, offsets)) else {
            return;
        };
        let profiles = [
            ("rtx3090", rtx3090()),
            (
                "lavapipe",
                profile("lavapipe", EXACT_SAMPLING).expect("profile"),
            ),
        ];
        for (stem, p) in profiles {
            let a = compile(&bc, "k", &p).unwrap_or_else(|e| panic!("{stem} {what}: {e}"));
            let fs = module_functions(&a.module.words);
            let entry = fs.entry(stem);
            let calls = fs.calls.get(&entry).cloned().unwrap_or_default();
            let fetches = fetches_by_function(&a.module.words);
            // The routines that fetch: the emulation's, and a `float` sample's fetch arm.
            let fetching: Vec<u32> = fetches
                .iter()
                .filter(|(f, n)| **n > 0 && **f != entry)
                .map(|(f, _)| *f)
                .collect();
            assert_eq!(
                fetching.len(),
                1 + usize::from(float),
                "{stem} {what}: the sampling routines, {fetching:?}"
            );
            for f in &fetching {
                assert_eq!(
                    calls.iter().filter(|c| *c == f).count(),
                    N,
                    "{stem} {what}: each site calls routine {f}"
                );
            }
            assert_eq!(
                fetches.get(&entry),
                Some(&0),
                "{stem} {what}: the sites fetch nothing themselves"
            );
            // The census counts each routine once.
            let b = compile(&one, "k", &p).unwrap_or_else(|e| panic!("{stem} {what}: {e}"));
            for name in ["sampler.linear.routine", "sampler.fetch.routine"] {
                assert_eq!(
                    a.strategies.entries.get(name).map(|e| e.words),
                    b.strategies.entries.get(name).map(|e| e.words),
                    "{stem} {what}: {name} counted once"
                );
            }
            assert!(
                a.strategies.entries.contains_key("sampler.linear.routine"),
                "{stem} {what}: the census names the routine"
            );
        }
    }
}

/// A kernel reading `texture2d<T, read>` slot 0 at its thread's position and storing the texel,
/// a `float4` or a `half4`, to buffer 0.
fn read_kernel_of(float: bool) -> String {
    let (texel, vec4, suffix, size) = match float {
        true => ("float", "<4 x float>", "v4f32", 16),
        false => ("half", "<4 x half>", "v4f16", 8),
    };
    format!(
        r#"define void @k(ptr addrspace(1) %t, ptr addrspace(1) %out, i32 %tid) {{
  %i = zext i32 %tid to i64
  %xy = insertelement <2 x i32> zeroinitializer, i32 %tid, i64 0
  %s = call ptr addrspace(2) @air.get_read_sampler()
  %r = call {{ {vec4}, i8 }} @air.read_texture_2d.{suffix}(ptr addrspace(1) %t, ptr addrspace(2) %s, <2 x i32> %xy, <2 x i32> zeroinitializer, i32 0, i32 1)
  %v = extractvalue {{ {vec4}, i8 }} %r, 0
  %q = getelementptr inbounds {vec4}, ptr addrspace(1) %out, i64 %i
  store {vec4} %v, ptr addrspace(1) %q, align {size}
  ret void
}}

declare ptr addrspace(2) @air.get_read_sampler()
declare {{ {vec4}, i8 }} @air.read_texture_2d.{suffix}(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)

!air.kernel = !{{!0}}
!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4, !5}}
!3 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<{texel}, read>", !"air.arg_name", !"t"}}
!4 = !{{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 {size}, !"air.arg_type_align_size", i32 {size}, !"air.arg_type_name", !"{texel}4", !"air.arg_name", !"out"}}
!5 = !{{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}}
"#
    )
}

/// Under the default policies, on a device that does not sample through Metal's own texture unit, a
/// texel's `float` or `half` value is the device's conversion (native device sampling): a sample's
/// and a read's module defines neither conversion routine and records `DeviceTexelConversion`
/// (`float`) or `HalfTexelConversion` (`half`), and the sample reads no field of a runtime
/// sampler's word for it, so a sampler of normalized coordinates keys no pipeline (what still reads
/// the word is a pixel sampler's, gated on its pixel bit). `TexelConversionPolicy::Exact` alone
/// makes a read Metal's conversion by the bound format's class through the routine, recording no
/// `DeviceTexelConversion`, and leaves a natively sampled texel the device's; under
/// `SamplingPolicy::Exact` the emulation converts a sample's by the class only where
/// `TexelConversionPolicy::Exact` says. On MoltenVK, which samples through Metal's own sampler, the
/// default keeps that device's form.
#[test]
fn texel_conversions_are_the_device_s_by_default() {
    use crate::device::{SamplingPolicy, TexelConversionPolicy};
    use crate::lower::texture::{float_texel, half};
    let policy = |sampling, texel_conversion| DevicePolicy {
        sampling,
        texel_conversion,
        ..DevicePolicy::default()
    };
    // (the policy, whether a sample's conversion is the routine's, whether a read's is)
    let policies = [
        (DevicePolicy::default(), false, false),
        (
            policy(SamplingPolicy::NativeDevice, TexelConversionPolicy::Exact),
            false,
            true,
        ),
        (
            policy(SamplingPolicy::Exact, TexelConversionPolicy::Device),
            false,
            false,
        ),
        (EXACT_SAMPLING, true, true),
    ];
    for float in [false, true] {
        let what = if float { "float" } else { "half" };
        let kernel = kernel_of(None, Form::Plain { offset: (0, 0) }, float);
        let Some(bc) = assemble(&format!("conversion {what}"), &kernel) else {
            return;
        };
        let Some(read) = assemble(&format!("read {what}"), &read_kernel_of(float)) else {
            return;
        };
        let (routine, note) = match float {
            true => (
                float_texel::CENSUS_NAME,
                PrecisionNote::DeviceTexelConversion,
            ),
            false => (half::CENSUS_NAME, PrecisionNote::HalfTexelConversion),
        };
        for stem in ["lavapipe", "nvidia-rtx5080-laptop", "amd-780m-radv"] {
            for (p, exact, exact_read) in policies {
                let what = format!("{stem} {what} {:?} {:?}", p.sampling, p.texel_conversion);
                let device = profile(stem, p).expect("profile");
                let a = compile(&bc, "k", &device).unwrap_or_else(|e| panic!("{what}: {e}"));
                assert_eq!(
                    a.strategies.entries.contains_key(routine),
                    exact,
                    "{what}: Metal's conversion is the routine's"
                );
                if !exact || float {
                    assert_eq!(a.precision.contains(&note), !exact, "{what}: {note:?}");
                }
                // A read: Metal's conversion where the policy alone says so.
                let r = compile(&read, "k", &device).unwrap_or_else(|e| panic!("{what} read: {e}"));
                link_and_validate(&r, &device).unwrap_or_else(|e| panic!("{what} read: L3: {e}"));
                assert_eq!(
                    r.strategies.entries.contains_key(routine),
                    exact_read,
                    "{what} read: Metal's conversion is the routine's"
                );
                if !exact_read || float {
                    assert_eq!(
                        r.precision.contains(&note),
                        !exact_read,
                        "{what} read: {note:?} in {:?}",
                        r.precision
                    );
                }
                if p.sampling != SamplingPolicy::NativeDevice {
                    continue;
                }
                for key in a.interface.read_keys() {
                    let SpecKey::SamplerState { fields, .. } = key else {
                        continue;
                    };
                    let read = match fields & SamplerWord::PIXEL_GATED {
                        0 => SamplerWord::PIXEL_COORDINATES,
                        _ => SamplerWord::PIXEL_NEAREST | SamplerWord::PIXEL_GATED,
                    };
                    assert_eq!(fields & !read, 0, "{what}: {key:?}");
                }
            }
        }
        let m = profile("moltenvk-m3", DevicePolicy::default()).expect("profile");
        let c = compile(&bc, "k", &m).unwrap_or_else(|e| panic!("moltenvk-m3 {what}: {e}"));
        assert!(
            !c.precision.contains(&PrecisionNote::DeviceTexelConversion),
            "moltenvk-m3 {what}: Metal's own sampler"
        );
    }
}

/// A kernel that reads `texture2d<half, sample>` slot 0 at its thread's position and samples it
/// through runtime sampler 0, storing both `half4`s to buffer 0.
const READ_AND_SAMPLE: &str = r#"define void @k(ptr addrspace(1) %t, ptr addrspace(1) %out, ptr addrspace(2) %s, i32 %tid) {
  %i = zext i32 %tid to i64
  %xy = insertelement <2 x i32> zeroinitializer, i32 %tid, i64 0
  %rs = call ptr addrspace(2) @air.get_read_sampler()
  %r = call { <4 x half>, i8 } @air.read_texture_2d.v4f16(ptr addrspace(1) %t, ptr addrspace(2) %rs, <2 x i32> %xy, <2 x i32> zeroinitializer, i32 0, i32 1)
  %rv = extractvalue { <4 x half>, i8 } %r, 0
  %u = uitofp i32 %tid to float
  %uv = insertelement <2 x float> <float poison, float 0.5>, float %u, i64 0
  %q = call { <4 x half>, i8 } @air.sample_texture_2d.v4f16(ptr addrspace(1) %t, ptr addrspace(2) %s, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.0, float 0.0, i32 0)
  %qv = extractvalue { <4 x half>, i8 } %q, 0
  %j = shl i64 %i, 1
  %pr = getelementptr inbounds <4 x half>, ptr addrspace(1) %out, i64 %j
  store <4 x half> %rv, ptr addrspace(1) %pr, align 8
  %k1 = or i64 %j, 1
  %pq = getelementptr inbounds <4 x half>, ptr addrspace(1) %out, i64 %k1
  store <4 x half> %qv, ptr addrspace(1) %pq, align 8
  ret void
}

declare ptr addrspace(2) @air.get_read_sampler()
declare { <4 x half>, i8 } @air.read_texture_2d.v4f16(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)
declare { <4 x half>, i8 } @air.sample_texture_2d.v4f16(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<half, sample>", !"air.arg_name", !"t"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"half4", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"s"}
!6 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
"#;

/// The `OpFunctionCall`s of a module.
fn calls(words: &[u32]) -> usize {
    const OP_FUNCTION_CALL: u32 = 57;
    let mut n = 0;
    let mut i = 5;
    while i < words.len() {
        n += usize::from(words[i] & 0xFFFF == OP_FUNCTION_CALL);
        i += ((words[i] >> 16) as usize).max(1);
    }
    n
}

/// A `half` texel the read converts by the class and the sample leaves to the device, in one
/// function of one slot: under `TexelConversionPolicy::Exact` with native device sampling, the slot
/// declares its class and the module the routine for the read, and the sample (whose texel the
/// device's sampler answers, `Sampling::half_texel`) calls nothing more than the read alone does,
/// records `HalfTexelConversion`, and the module passes L3. Under `SamplingPolicy::Exact` both
/// convert, the sample through the routine too.
#[test]
fn a_read_converts_by_the_class_beside_a_natively_sampled_texel() {
    use crate::device::{SamplingPolicy, TexelConversionPolicy};
    use crate::lower::texture::half;
    let (Some(mixed), Some(read)) = (
        assemble("read and sample", READ_AND_SAMPLE),
        assemble("read half", &read_kernel_of(false)),
    ) else {
        return;
    };
    let native = DevicePolicy {
        texel_conversion: TexelConversionPolicy::Exact,
        ..DevicePolicy::default()
    };
    for stem in ["lavapipe", "nvidia-rtx5080-laptop"] {
        let device = profile(stem, native).expect("profile");
        let a = compile(&mixed, "k", &device).unwrap_or_else(|e| panic!("{stem}: {e}"));
        link_and_validate(&a, &device).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        let r = compile(&read, "k", &device).unwrap_or_else(|e| panic!("{stem} read: {e}"));
        assert!(
            a.strategies.entries.contains_key(half::CENSUS_NAME),
            "{stem}"
        );
        assert!(
            a.precision.contains(&PrecisionNote::HalfTexelConversion),
            "{stem}: {:?}",
            a.precision
        );
        assert_eq!(
            calls(&a.module.words),
            calls(&r.module.words),
            "{stem}: the sample calls no conversion"
        );
        let exact = profile(stem, EXACT_SAMPLING).expect("profile");
        let e = compile(&mixed, "k", &exact).unwrap_or_else(|e| panic!("{stem} exact: {e}"));
        link_and_validate(&e, &exact).unwrap_or_else(|e| panic!("{stem} exact: L3: {e}"));
        assert!(
            calls(&e.module.words) > calls(&a.module.words),
            "{stem}: the emulated sample converts through the routine"
        );
        assert_eq!(
            SamplingPolicy::NativeDevice,
            native.sampling,
            "the default samples natively"
        );
    }
}

/// The residue: the binder sets a slot's `OffsetBaseAxes` from the bound texture's
/// `TextureFacts::offset_base_axes`: the axes of a view whose first level lies above its parent's
/// base level on which the parent's base-level size is not a power of two and the view's own is,
/// and some level of the view is 2 or 4 wide; 0 for a texture that is not a view, a view at the
/// parent's base, an axis whose own size is not a power of two either, and a view whose levels
/// stay above 4.
#[test]
fn the_binder_states_a_view_s_parent_base_axes() {
    use crate::metal::TextureType;
    let Some(bc) = assemble(
        "view parent base",
        &kernel_of(
            Some(linear(2, false, 0)),
            Form::Plain { offset: (2, -1) },
            false,
        ),
    ) else {
        return;
    };
    let lvp = device_of("lavapipe");
    let p = notes_pipeline(&lvp, &bc);
    let id = p
        .stage
        .specialization
        .constants
        .iter()
        .find(|c| c.key == SpecKey::OffsetBaseAxes(TEXTURE))
        .map(|c| c.id)
        .expect("the slot's OffsetBaseAxes");
    let cases = [
        // (parent size, level, view levels, axes)
        ([18, 16], 2, 3, 0b01),
        ([16, 18], 2, 3, 0b10),
        ([17, 9], 2, 1, 0b11),
        ([20, 12], 2, 3, 0b00),
        ([16, 16], 2, 3, 0b00),
        ([18, 16], 0, 5, 0b00),
        ([1025, 16], 1, 1, 0b00),
        ([1025, 16], 1, 10, 0b01),
    ];
    for ([w, h], level, levels, want) in cases {
        let mut facts =
            crate::bind::TextureFacts::new(TextureType::Type2D, PixelFormat::RGBA8Unorm);
        facts.usage = crate::metal::TextureUsage::SHADER_READ;
        facts.parent_relative_level = level;
        facts.parent_base_size = [w, h, 1];
        facts.mipmap_level_count = levels;
        assert_eq!(facts.offset_base_axes(), want, "{w}x{h} at {level}");
        let plan = plan_facts(&p, &lvp, facts, None);
        let e = plan
            .specialization
            .entries
            .iter()
            .find(|e| e.constant_id == id)
            .expect("specialized");
        let at = e.offset as usize;
        let value = u32::from_le_bytes(
            plan.specialization.data[at..at + 4]
                .try_into()
                .expect("a word"),
        );
        assert_eq!(
            value,
            u32::from(want),
            "{w}x{h} at {level} of {levels} levels"
        );
    }
    // A 3D view's depth, a 1D one's width; a cube's none.
    let mut facts = crate::bind::TextureFacts::new(TextureType::Type3D, PixelFormat::RGBA8Unorm);
    facts.parent_relative_level = 1;
    facts.parent_base_size = [16, 16, 9];
    facts.mipmap_level_count = 3;
    assert_eq!(facts.offset_base_axes(), 0b100);
    facts.texture_type = TextureType::Type1D;
    facts.parent_base_size = [9, 1, 1];
    assert_eq!(facts.offset_base_axes(), 0b1);
    facts.texture_type = TextureType::Cube;
    facts.parent_base_size = [9, 9, 1];
    assert_eq!(facts.offset_base_axes(), 0);
}

/// On a profile that takes `DriverDeviation::Unorm8LinearFilter` (the RTX 5080 Laptop GPU's
/// capture, NVIDIA's 615.71.09, under a policy of `Unorm8LinearFilterPolicy::DeviceWithinOneCode`;
/// the default policy takes it nowhere), a 2D texture's or 2D array's linear sample through
/// a constexpr sampler with normalized coordinates or a runtime sampler slot, with a `half` or
/// `float` result and no offset, hands the 8-bit unsigned normalized classes to the device's
/// filter: it records `DeviceUnorm8LinearFilter` under the slot's class (and a runtime slot's
/// word), requires `VkProperty::Unorm8LinearFilter`, and validates. A texel offset, pixel
/// coordinates, a 1D or 3D texture, every sample under the default policy, and every sample on
/// an unmeasured profile (lavapipe's capture, Mesa 26.2.3, below the measured 26.2.4) keep
/// Metal's filter emulated: no note, no requirement. The plan's word estimate bounds each.
#[test]
fn a_measured_device_filters_the_eight_bit_classes_itself() {
    let edge = linear(1, false, 0);
    let mirror = linear(3, false, 0);
    let plain = Form::Plain { offset: (0, 0) };
    // (name, sampler, form, `float`, native on a measured profile)
    let cases: Vec<(&str, Option<u64>, Form, bool, bool)> = vec![
        ("2d edge", Some(edge), plain, false, true),
        ("2d mirror float", Some(mirror), plain, true, true),
        ("array", Some(edge), Form::Array { layer: 1 }, false, true),
        ("runtime", None, plain, false, true),
        ("runtime float", None, plain, true, true),
        ("runtime array", None, Form::Array { layer: 1 }, true, true),
        (
            "offset",
            Some(edge),
            Form::Plain { offset: (1, 0) },
            false,
            false,
        ),
        (
            "runtime offset",
            None,
            Form::Plain { offset: (0, -1) },
            false,
            false,
        ),
        ("pixel", Some(linear(1, true, 0)), plain, false, false),
        ("1d", Some(edge), Form::OneD, false, false),
        (
            "3d",
            Some(edge),
            Form::ThreeD { offset: (0, 0, 0) },
            false,
            false,
        ),
    ];
    use crate::budget::SPIRV_WORDS;
    use crate::device::Unorm8LinearFilterPolicy;
    use crate::error::DriverDeviation as D;
    use crate::library::FunctionConstantValues;
    use crate::lower::select::LowerInputs;
    use crate::pipeline::compile::{compile_bitcode, plan_bitcode, EntryRef};
    use crate::pipeline::testing::meter;
    // Under Metal's sampler emulated exactly: native device sampling, the default, hands every
    // such sample to the device's sampler before the 8-bit filter's choice.
    let emulated = EXACT_SAMPLING;
    let accepting = DevicePolicy {
        unorm8_linear_filter: Unorm8LinearFilterPolicy::DeviceWithinOneCode,
        ..emulated
    };
    let measured = profile("nvidia-rtx5080-laptop", accepting).expect("profile");
    assert!(measured.takes_driver_deviation(D::Unorm8LinearFilter));
    let exact = profile("nvidia-rtx5080-laptop", emulated).expect("profile");
    assert!(!exact.takes_driver_deviation(D::Unorm8LinearFilter));
    let unmeasured = profile("lavapipe", accepting).expect("profile");
    assert!(!unmeasured.takes_driver_deviation(D::Unorm8LinearFilter));
    for (name, sampler, form, float, native) in cases {
        let Some(bc) = assemble(name, &kernel_of(sampler, form, float)) else {
            return;
        };
        for (stem, p, want) in [
            ("nvidia-rtx5080-laptop accepting", &measured, native),
            ("nvidia-rtx5080-laptop by default", &exact, false),
            ("lavapipe accepting", &unmeasured, false),
        ] {
            let what = format!("{name} on {stem}");
            let a = compile(&bc, "k", p).unwrap_or_else(|e| panic!("{what}: {e}"));
            link_and_validate(&a, p).unwrap_or_else(|e| panic!("{what}: {e}"));
            // The plan's word estimate bounds the native snap's site too: the estimate test's
            // committed profiles take the default policy.
            let (entry, consts, inputs) = (
                EntryRef::Name("k"),
                FunctionConstantValues::defaults(),
                LowerInputs::new(),
            );
            let plan = plan_bitcode(&bc, entry, &consts, p, &inputs, &mut meter())
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            let emitted =
                compile_bitcode(&bc, entry, &consts, p, &inputs, &mut meter(), SPIRV_WORDS)
                    .unwrap_or_else(|e| panic!("{what}: {e}"))
                    .module
                    .words
                    .len() as u64;
            assert!(
                plan.estimate.words >= emitted,
                "{what}: estimate {} < emitted {emitted}",
                plan.estimate.words
            );
            assert_eq!(
                a.precision
                    .contains(&PrecisionNote::DeviceUnorm8LinearFilter),
                want,
                "{what}"
            );
            assert_eq!(
                a.specialized
                    .iter()
                    .any(|n| n.note == PrecisionNote::DeviceUnorm8LinearFilter),
                want,
                "{what}: {:?}",
                a.specialized
            );
            assert_eq!(
                a.requirements
                    .properties
                    .contains(&VkProperty::Unorm8LinearFilter),
                want,
                "{what}"
            );
            // The emulation stays for every other class, and the filtered note with it.
            let filtered = match float {
                true => PrecisionNote::FloatSampleFilter,
                false => PrecisionNote::HalfSampleFilter,
            };
            assert!(a.precision.contains(&filtered), "{what}");
        }
    }
}

/// Native device sampling, the default policy, hands a linear sample through a constexpr sampler or
/// a runtime sampler slot, of a 2D texture, a 2D array or a 3D texture, to the device's sampler on
/// a device whose sampler is not Metal's (the RTX 5080 Laptop GPU's capture and lavapipe's): the
/// artifact requires `VkProperty::DeviceSampling` and records `PrecisionNote::DeviceSampling` and
/// no filter note, its module is smaller than the emulation's, it validates, and its module reads
/// none of the emulation's fields of a sampler word (`SpecKey::SamplerState`): a constexpr
/// sampler's module reads no word, and a runtime sampler's reads only what a device whose sampler
/// is Metal's reads (its texel conversion is the device's too), where the emulation's reads its
/// address modes and border too. Under `SamplingPolicy::Exact` Metal's filter stays emulated, with
/// its note, and nothing records the native sampling.
#[test]
fn native_device_sampling_hands_samples_to_the_device() {
    use crate::library::FunctionArtifact;
    let exact = EXACT_SAMPLING;
    let plain = Form::Plain { offset: (0, 0) };
    let cases: Vec<(&str, Option<u64>, Form, bool)> = vec![
        ("2d edge", Some(linear(1, false, 0)), plain, false),
        ("2d mirror float", Some(linear(3, false, 0)), plain, true),
        ("runtime", None, plain, false),
        ("runtime array float", None, Form::Array { layer: 1 }, true),
        (
            "3d",
            Some(linear(1, false, 0)),
            Form::ThreeD { offset: (0, 0, 0) },
            false,
        ),
    ];
    let sampler_keys = |a: &FunctionArtifact| {
        a.interface
            .read_keys()
            .into_iter()
            .filter(|k| matches!(k, SpecKey::SamplerState { .. }))
            .collect::<Vec<_>>()
    };
    let noted = |a: &FunctionArtifact, note: PrecisionNote| {
        a.precision.contains(&note) || a.specialized.iter().any(|s| s.note == note)
    };
    for (name, sampler, form, float) in cases {
        let Some(bc) = assemble(name, &kernel_of(sampler, form, float)) else {
            return;
        };
        let filtered = match float {
            true => PrecisionNote::FloatSampleFilter,
            false => PrecisionNote::HalfSampleFilter,
        };
        for stem in ["nvidia-rtx5080-laptop", "lavapipe"] {
            let what = format!("{name} on {stem}");
            let native = profile(stem, DevicePolicy::default()).expect("profile");
            let emulated = profile(stem, exact).expect("profile");
            let n = compile(&bc, "k", &native).unwrap_or_else(|e| panic!("{what}: {e}"));
            link_and_validate(&n, &native).unwrap_or_else(|e| panic!("{what}: {e}"));
            let e = compile(&bc, "k", &emulated).unwrap_or_else(|e| panic!("{what} exact: {e}"));
            link_and_validate(&e, &emulated).unwrap_or_else(|e| panic!("{what} exact: {e}"));
            assert!(
                n.requirements
                    .properties
                    .contains(&VkProperty::DeviceSampling),
                "{what}"
            );
            assert!(
                n.precision.contains(&PrecisionNote::DeviceSampling),
                "{what}"
            );
            assert!(!noted(&n, filtered), "{what}");
            // The emulation's address modes and border are read nowhere natively; what a native
            // module may still read of a runtime sampler's word is the texel conversion's
            // (`texture::OneTexel::Word`): its coordinates and filter.
            let conversion = SamplerWord::PIXEL_COORDINATES
                | SamplerWord::LINEAR_FILTER
                | SamplerWord::PIXEL_GATED;
            // Natively the conversion is the device's too: what remains is what a device whose
            // sampler is Metal's reads, a pixel sampler's coordinates and the footprint cells'
            // filter through its pixel tests alone.
            for key in sampler_keys(&n) {
                let SpecKey::SamplerState { fields, .. } = key else {
                    unreachable!()
                };
                let read = match fields & SamplerWord::PIXEL_GATED {
                    0 => SamplerWord::PIXEL_COORDINATES,
                    _ => SamplerWord::PIXEL_NEAREST | SamplerWord::PIXEL_GATED,
                };
                assert_eq!(fields & !read, 0, "{what}: {key:?}");
            }
            if sampler.is_none() {
                let reads = |a: &FunctionArtifact| {
                    sampler_keys(a).iter().fold(0, |m, k| match k {
                        SpecKey::SamplerState { fields, .. } => m | fields,
                        _ => m,
                    })
                };
                assert_ne!(reads(&e) & !conversion, 0, "{what} exact");
            } else {
                assert_eq!(sampler_keys(&n), Vec::new(), "{what}");
            }
            let words = |a: &FunctionArtifact| a.module.words.len();
            assert!(
                words(&n) < words(&e),
                "{what}: {} words natively, {} emulated",
                words(&n),
                words(&e)
            );
            assert!(
                !e.requirements
                    .properties
                    .contains(&VkProperty::DeviceSampling),
                "{what} exact"
            );
            assert!(!noted(&e, PrecisionNote::DeviceSampling), "{what} exact");
            assert!(noted(&e, filtered), "{what} exact");
        }
    }
}
