//! A plain gather's or nearest sample's `float` texels, fetched by the shader where Metal's
//! footprint and level are known (`lower::sampler:: fetched`), and
//! `PrecisionNote::DeviceTexelDenormals` where the device reads them.
//!
//! A device's sample or gather may read a denormal 32-bit float texel as zero (RADV, NVIDIA and
//! ANV); its `OpImageFetch` keeps it. The selection tests pin which samples and
//! gathers fetch on every committed profile, and which record the note. The interpreter runs the
//! fetches of an `R32Float` texture whose texels include denormals and signed zeros: every lane
//! is the texel Metal reads, bit for bit. (The interpreter models no device sample or gather, so
//! the other arm is not run.)
//!
//! `llvm-as` assembles the kernels; without it the tests skip visibly.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::bind::SamplerDescriptor;
use crate::device::requirements::VkProperty;
use crate::metal::sampler::SamplerWord;
use crate::metal::texel_format::texel_format;
use crate::metal::PixelFormat;
use crate::pipeline::sym::{Axis, ResourceKind, ResourceSlot, SpecKey, Stage};
use crate::pipeline::testing::EXACT_SAMPLING;
use crate::pipeline::testing::{
    compile, link_and_validate, llvm_assemble, profile, typed_module, AssembleError,
};
use crate::pipeline::PrecisionNote;
use crate::spirv::interp::{run_with_images, Buffer, Config, Image, Order, Resources};
use crate::spirv::ir::{GlobalKind, TypedModule};

const HEADER: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n";

/// Bitcode of the module `body` (after the AIR header), or `None` without `llvm-as`.
fn assemble(name: &str, body: &str) -> Option<Vec<u8>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "m2v-sampler-fetched-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed),
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

/// The words of a constexpr sampler: minification and magnification `filter` (AIR code: 0
/// nearest, 1 linear), mip filter `mip` (0 none, 1 nearest, 2 linear), every axis addressed by
/// `address` (0 zero or border, 1 edge, 2 repeat, 3 mirrored repeat), and border color code
/// `border`, with the fields Apple's compiler writes for the rest (as `engine_linear_half`'s).
fn sampler_words(filter: u64, mip: u64, address: u64, border: u64) -> u64 {
    const BASE: u64 = 0x007b_ff00_0008_0000;
    BASE | address
        | address << 3
        | address << 6
        | filter << 9
        | filter << 11
        | mip << 13
        | border << 56
}

/// A nearest, clamp-to-edge sampler without a mip filter.
const NEAREST: u64 = 0x007b_ff00_0008_0000 | 1 | 1 << 3 | 1 << 6;

/// How a test kernel reads texture slot 0 at buffer 0's lane `tid` (a `float4`: the coordinate
/// in `x`, `y` (and `z` for a 3D texture), and a level in `w`).
#[derive(Clone, Copy, Debug)]
enum Read {
    /// `texture2d<float>::gather(s, uv, int2(0), component)`.
    Gather { component: u32 },
    /// `texture2d<float>::sample(s, uv)`, at bias 0.
    Sample,
    /// `texture2d<float>::sample(s, uv, level(w))`.
    Level,
    /// `texture2d<float>::sample(s, uv, level(2))`.
    ConstLevel,
    /// `texture2d<float>::sample(s, uv, min_lod_clamp(1))`.
    MinLodClamp,
    /// `texture2d_array<float>::sample(s, uv, 1)`.
    Array,
    /// `texture1d<float>::sample(s, u)`.
    OneD,
    /// `texture3d<float>::sample(s, uvw)`.
    ThreeD,
    /// `depth2d<float>::sample(s, uv)`.
    Depth,
    /// `texture2d<half>::sample(s, uv)`.
    Half,
}

/// A kernel reading texture slot 0 through the constexpr sampler of `words` as `read` says,
/// storing the `float4` (a `half4` for [`Read::Half`]; a depth sample's in `x`) to buffer 1.
fn kernel(words: u64, read: Read) -> String {
    let uv = "%uv = shufflevector <4 x float> %c4, <4 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>";
    let s = "ptr addrspace(1) %t, ptr addrspace(2) @__air_sampler_state";
    let sample2 =
        "ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32";
    let (tex_ty, ty, call, decl, store_ty, size) = match read {
        Read::Gather { component } => (
            "texture2d<float, sample>",
            "<4 x float>",
            format!("{uv}\n  %r = call {{ <4 x float>, i8 }} @air.gather_texture_2d.v4f32({s}, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i32 {component}, i32 0)"),
            "declare { <4 x float>, i8 } @air.gather_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i32, i32)".to_string(),
            "<4 x float>",
            16,
        ),
        Read::Sample | Read::Level | Read::ConstLevel | Read::MinLodClamp => {
            let (level, lod, min) = match read {
                Read::Level => ("true", "%lod", "0.0"),
                Read::ConstLevel => ("true", "2.0", "0.0"),
                Read::MinLodClamp => ("false", "0.0", "1.0"),
                _ => ("false", "0.0", "0.0"),
            };
            (
                "texture2d<float, sample>",
                "<4 x float>",
                format!("{uv}\n  %r = call {{ <4 x float>, i8 }} @air.sample_texture_2d.v4f32({s}, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 {level}, float {lod}, float {min}, i32 0)"),
                format!("declare {{ <4 x float>, i8 }} @air.sample_texture_2d.v4f32({sample2})"),
                "<4 x float>",
                16,
            )
        }
        Read::Array => (
            "texture2d_array<float, sample>",
            "<4 x float>",
            format!("{uv}\n  %r = call {{ <4 x float>, i8 }} @air.sample_texture_2d_array.v4f32({s}, <2 x float> %uv, i32 1, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.0, float 0.0, i32 0)"),
            "declare { <4 x float>, i8 } @air.sample_texture_2d_array.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i32, i1, <2 x i32>, i1, float, float, i32)".to_string(),
            "<4 x float>",
            16,
        ),
        Read::OneD => (
            "texture1d<float, sample>",
            "<4 x float>",
            format!("%u = extractelement <4 x float> %c4, i64 0\n  %r = call {{ <4 x float>, i8 }} @air.sample_texture_1d.v4f32({s}, float %u, i1 false, i32 0, i1 false, float 0.0, float 0.0, i32 0)"),
            "declare { <4 x float>, i8 } @air.sample_texture_1d.v4f32(ptr addrspace(1), ptr addrspace(2), float, i1, i32, i1, float, float, i32)".to_string(),
            "<4 x float>",
            16,
        ),
        Read::ThreeD => (
            "texture3d<float, sample>",
            "<4 x float>",
            format!("%uvw = shufflevector <4 x float> %c4, <4 x float> zeroinitializer, <3 x i32> <i32 0, i32 1, i32 2>\n  %r = call {{ <4 x float>, i8 }} @air.sample_texture_3d.v4f32({s}, <3 x float> %uvw, i1 true, <3 x i32> zeroinitializer, i1 false, float 0.0, float 0.0, i32 0)"),
            "declare { <4 x float>, i8 } @air.sample_texture_3d.v4f32(ptr addrspace(1), ptr addrspace(2), <3 x float>, i1, <3 x i32>, i1, float, float, i32)".to_string(),
            "<4 x float>",
            16,
        ),
        Read::Depth => (
            "depth2d<float, sample>",
            "float",
            format!("{uv}\n  %r = call {{ float, i8 }} @air.sample_depth_2d.f32({s}, i32 1, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.0, float 0.0, i32 0)"),
            "declare { float, i8 } @air.sample_depth_2d.f32(ptr addrspace(1), ptr addrspace(2), i32, <2 x float>, i1, <2 x i32>, i1, float, float, i32)".to_string(),
            "<4 x float>",
            16,
        ),
        Read::Half => (
            "texture2d<half, sample>",
            "<4 x half>",
            format!("{uv}\n  %r = call {{ <4 x half>, i8 }} @air.sample_texture_2d.v4f16({s}, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.0, float 0.0, i32 0)"),
            "declare { <4 x half>, i8 } @air.sample_texture_2d.v4f16(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)".to_string(),
            "<4 x half>",
            8,
        ),
    };
    let value = match read {
        Read::Depth => "%d = extractvalue { float, i8 } %r, 0\n  %v = insertelement <4 x float> zeroinitializer, float %d, i64 0".to_string(),
        _ => format!("%v = extractvalue {{ {ty}, i8 }} %r, 0"),
    };
    let out_name = if size == 8 { "half4" } else { "float4" };
    format!(
        r#"@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 {words}, i64 0], align 8

define void @k(ptr addrspace(1) %t, ptr addrspace(1) %c, ptr addrspace(1) %out, i32 %tid) {{
  %i = zext i32 %tid to i64
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %c, i64 %i
  %c4 = load <4 x float>, ptr addrspace(1) %p, align 16
  %lod = extractelement <4 x float> %c4, i64 3
  {call}
  {value}
  %q = getelementptr inbounds {store_ty}, ptr addrspace(1) %out, i64 %i
  store {store_ty} %v, ptr addrspace(1) %q, align {size}
  ret void
}}

{decl}

!air.kernel = !{{!0}}
!air.sampler_states = !{{!6}}
!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4, !5, !7}}
!3 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"{tex_ty}", !"air.arg_name", !"t"}}
!4 = !{{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"c"}}
!5 = !{{i32 2, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 {size}, !"air.arg_type_align_size", i32 {size}, !"air.arg_type_name", !"{out_name}", !"air.arg_name", !"out"}}
!6 = !{{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}}
!7 = !{{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}}
"#
    )
}

/// [`kernel`]'s kernel reading texture slot 0 through runtime sampler slot 0 (parameter `s`)
/// instead of a constexpr sampler.
fn runtime_kernel(read: Read) -> String {
    let text = kernel(NEAREST, read);
    let mut out: String = text
        .lines()
        .filter(|l| !l.starts_with("@__air_sampler_state") && *l != "!air.sampler_states = !{!6}")
        .map(|l| format!("{l}\n"))
        .collect();
    for (from, to) in [
        (
            "!6 = !{!\"air.sampler_state\", ptr addrspace(2) @__air_sampler_state}",
            "!6 = !{i32 3, !\"air.sampler\", !\"air.location_index\", i32 0, i32 1, !\"air.arg_type_name\", !\"sampler\", !\"air.arg_name\", !\"s\"}",
        ),
        ("ptr addrspace(2) @__air_sampler_state", "ptr addrspace(2) %s"),
        (
            "ptr addrspace(1) %out, i32 %tid)",
            "ptr addrspace(1) %out, ptr addrspace(2) %s, i32 %tid)",
        ),
        ("!2 = !{!3, !4, !5, !7}", "!2 = !{!3, !4, !5, !6, !7}"),
        (
            "!7 = !{i32 3, !\"air.thread_position_in_grid\"",
            "!7 = !{i32 4, !\"air.thread_position_in_grid\"",
        ),
    ] {
        assert!(out.contains(from), "{read:?}: no {from}");
        out = out.replace(from, to);
    }
    out
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

/// `OpImageFetch`.
const FETCH: u16 = 95;

/// `OpImageSampleImplicitLod`, `OpImageSampleExplicitLod`, and `OpImageGather`.
const DEVICE_READS: [u16; 3] = [87, 88, 96];

/// A plain gather, and a sample through a nearest sampler at a level the function knows (no mip
/// filter, a nearest mip filter's constant or kernel level), of a 1D, 2D, 2D array or 3D texture
/// slot or a depth texture, through a constexpr sampler, fetches its texels behind the slot's
/// `TexelFormat` specialization on every committed profile without `VkProperty::MetalSampler`,
/// beside the device's read for the other formats, and records no `DeviceTexelDenormals`. On a
/// profile with the property Metal's own sampler reads them, with no fetch. Each validates.
#[test]
fn plain_gathers_and_nearest_samples_fetch_their_float_texels() {
    let mip_nearest = sampler_words(0, 1, 1, 0);
    let cases = [
        (NEAREST, Read::Gather { component: 0 }),
        (sampler_words(1, 2, 2, 0), Read::Gather { component: 2 }),
        (sampler_words(0, 0, 0, 2), Read::Gather { component: 1 }),
        (NEAREST, Read::Sample),
        (sampler_words(0, 0, 3, 0), Read::Sample),
        (mip_nearest, Read::Level),
        (mip_nearest, Read::ConstLevel),
        (NEAREST, Read::Array),
        (NEAREST, Read::OneD),
        (sampler_words(0, 0, 2, 0), Read::ThreeD),
        (NEAREST, Read::Depth),
    ];
    let profiles = crate::spirv::testing::committed_exact_profiles().expect("profiles");
    for (words, read) in cases {
        let name = format!("{read:?} through {words:#x}");
        let Some(bc) = assemble(&name, &kernel(words, read)) else {
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
            let fetches = ops.contains(&FETCH);
            let device = ops.iter().any(|o| DEVICE_READS.contains(o));
            let format = a
                .interface
                .read_keys()
                .into_iter()
                .any(|k| matches!(k, SpecKey::TexelFormat(_)));
            assert_eq!((fetches, device, format), (!metal, true, !metal), "{what}");
            assert!(
                !a.precision.contains(&PrecisionNote::DeviceTexelDenormals),
                "{what}"
            );
        }
    }
}

/// A `float` sample whose texels the device reads records `DeviceTexelDenormals` on every committed
/// profile without `VkProperty::MetalSampler`: a filtered sample no emulation reaches (a linear mip
/// filter), and a nearest one under a nearest mip filter with a `min_lod_clamp`, whose level the
/// function does not know. A linear sample the emulation reaches fetches its texels and records
/// none, and so does a `half` result, which holds no 32-bit denormal. A profile with the property
/// records none.
#[test]
fn device_reads_of_float_texels_record_the_note() {
    let cases = [
        (sampler_words(1, 2, 1, 0), Read::Sample, true),
        (sampler_words(0, 1, 1, 0), Read::MinLodClamp, true),
        (sampler_words(1, 0, 1, 0), Read::Sample, false),
        (NEAREST, Read::Half, false),
    ];
    let profiles = crate::spirv::testing::committed_exact_profiles().expect("profiles");
    for (words, read, noted) in cases {
        let name = format!("{read:?} through {words:#x}");
        let Some(bc) = assemble(&name, &kernel(words, read)) else {
            return;
        };
        for (stem, p) in &profiles {
            let what = format!("{name} on {stem}");
            let a = compile(&bc, "k", p).unwrap_or_else(|e| panic!("{what}: {e}"));
            let metal = p
                .fields()
                .enabled
                .properties
                .contains(&VkProperty::MetalSampler);
            assert_eq!(
                a.precision.contains(&PrecisionNote::DeviceTexelDenormals),
                noted && !metal,
                "{what}"
            );
        }
    }
}

/// The texels of a 4x4 `R32Float` texture: denormals of both signs, signed zeros, the smallest
/// normal, and ordinary values.
const TEXELS: [u32; 16] = [
    0x0000_0001,
    0x8000_0002,
    0x007f_ffff,
    0x807f_ffff,
    0x0040_0000,
    0x8000_0000,
    0x0000_0000,
    0x0080_0000,
    0x3f80_0000,
    0xbf00_0000,
    0x0012_3456,
    0x8065_4321,
    0x4120_0000,
    0x0000_0100,
    0x8000_0001,
    0x3e80_0000,
];

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

/// Buffer 1's `float4`s, as bits, after `bc` runs one thread per coordinate of `points` (buffer
/// 0's `float4`s) on lavapipe's profile with the [`TEXELS`] texture bound and its
/// `TexelFormat` specialization `R32Float`'s.
fn run(name: &str, bc: &[u8], points: &[[f32; 4]]) -> Vec<[u32; 4]> {
    let p = profile("lavapipe", EXACT_SAMPLING).expect("profile");
    let a = compile(bc, "k", &p).expect("compile");
    // The note holds nowhere, or (a runtime sampler's sample) only where the word filters.
    let note = PrecisionNote::DeviceTexelDenormals;
    assert!(
        !a.precision.contains(&note) || a.specialized.iter().any(|n| n.note == note),
        "{name}"
    );
    let (plan, m) = typed_module(bc, "k", &p).expect("typed module");
    let slot = combined_slot(&m);
    let mut image = Image::new(slot, [4, 4, 1], 1, 1, 1);
    for y in 0..4u32 {
        for x in 0..4u32 {
            let at = image.index(0, 0, 0, [x, y, 0]).expect("texel");
            image.texels[at] = [TEXELS[(4 * y + x) as usize], 0, 0, 1f32.to_bits()];
        }
    }
    let mut spec = BTreeMap::new();
    for axis in [Axis::X, Axis::Y, Axis::Z] {
        spec.insert(SpecKey::LocalSize(axis), 1);
    }
    spec.insert(SpecKey::FlatLocalSize, 1);
    let texture = a
        .interface
        .read_keys()
        .into_iter()
        .find_map(|k| match k {
            k @ SpecKey::TexelFormat(_) => Some(k),
            _ => None,
        })
        .expect("the texture slot's `TexelFormat` specialization");
    spec.insert(texture, u64::from(texel_format(PixelFormat::R32Float).0));
    // A runtime sampler slot's word: `MTLSamplerDescriptor`'s defaults (nearest filters, no mip
    // filter, clamped to the edge, normalized coordinates), masked by the fields the function
    // reads.
    for k in a.interface.read_keys() {
        if let SpecKey::SamplerState { fields, .. } = k {
            let word = SamplerWord::of(&SamplerDescriptor::new()).0 & fields;
            spec.insert(k, u64::from(word));
        }
    }
    let coords: Vec<u8> = points
        .iter()
        .flat_map(|c| c.iter().flat_map(|v| v.to_le_bytes()))
        .collect();
    let buffers = [coords, vec![0xab; 16 * points.len()]]
        .into_iter()
        .enumerate()
        .map(|(i, bytes)| Buffer {
            slot: Some(ResourceSlot {
                stage: Stage::Kernel,
                kind: ResourceKind::Buffer,
                index: i as u32,
            }),
            address: None,
            bytes,
        })
        .collect();
    let res = Resources {
        buffers,
        push: vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)],
        spec,
    };
    let mut outs = Vec::new();
    for order in [Order::Forward, Order::Reverse] {
        let cfg = Config {
            workgroups: [points.len() as u32, 1, 1],
            order,
            ..Config::default()
        };
        let (o, _) = run_with_images(&m, &cfg, res.clone(), vec![image.clone()])
            .unwrap_or_else(|e| panic!("{name}: interpreter: {e:?}"));
        outs.push(o.buffers);
    }
    assert_eq!(outs[0], outs[1], "{name}: the scheduling orders disagree");
    outs[0][1]
        .as_chunks::<16>()
        .0
        .iter()
        .map(|c| {
            std::array::from_fn(|k| {
                u32::from_le_bytes([c[4 * k], c[4 * k + 1], c[4 * k + 2], c[4 * k + 3]])
            })
        })
        .collect()
}

/// On the interpreter: a gather at each interior texel corner reads Metal's 2x2 footprint, `x` `(i,
/// j + 1)`, `y` `(i + 1, j + 1)`, `z` `(i + 1, j)` and `w` `(i, j)`, and a nearest sample at each
/// texel centre that texel, every denormal and signed zero kept, through the fetches the `R32Float`
/// specialization selects.
#[test]
fn fetched_float_texels_keep_their_denormals() {
    let texel = |x: u32, y: u32| TEXELS[(4 * y + x) as usize];
    let Some(gather) = assemble("gather", &kernel(NEAREST, Read::Gather { component: 0 })) else {
        return;
    };
    let corners: Vec<(u32, u32)> = (0..3).flat_map(|j| (0..3).map(move |i| (i, j))).collect();
    let points: Vec<[f32; 4]> = corners
        .iter()
        .map(|&(i, j)| [(i + 1) as f32 / 4.0, (j + 1) as f32 / 4.0, 0.0, 0.0])
        .collect();
    for ((i, j), got) in corners.iter().zip(run("gather", &gather, &points)) {
        let want = [
            texel(*i, j + 1),
            texel(i + 1, j + 1),
            texel(i + 1, *j),
            texel(*i, *j),
        ];
        assert_eq!(got, want, "gather at corner ({i}, {j})");
    }
    let Some(sample) = assemble("sample", &kernel(NEAREST, Read::Sample)) else {
        return;
    };
    let centres: Vec<(u32, u32)> = (0..4).flat_map(|y| (0..4).map(move |x| (x, y))).collect();
    let points: Vec<[f32; 4]> = centres
        .iter()
        .map(|&(x, y)| [(x as f32 + 0.5) / 4.0, (y as f32 + 0.5) / 4.0, 0.0, 0.0])
        .collect();
    for ((x, y), got) in centres.iter().zip(run("sample", &sample, &points)) {
        assert_eq!(
            got,
            [texel(*x, *y), 0, 0, 1f32.to_bits()],
            "sample at ({x}, {y})"
        );
    }
}

/// A plain gather, and a sample of a 1D, 2D, 2D array or 3D texture slot or a depth texture,
/// through a runtime sampler slot fetches its texels addressed by the slot's word
/// (`fetched::FetchAddress::Word`), behind the slot's `TexelFormat` specialization, on every
/// committed profile without `VkProperty::MetalSampler`, beside the device's read; the word
/// carries the coordinates, the plane axes' address modes and the border color, a sample's its
/// filter, and a stated level's its mip filter and level clamp. A sample fetches only where the
/// word states one texel (`fetched::word_reach`), so `DeviceTexelDenormals` holds only under the
/// specializations where the word filters (and, without a stated level, has a mip filter); a
/// gather's taps are fetched whatever the word's filters, so it records none. A profile with the
/// property reads them by Metal's own sampler. Each validates.
#[test]
fn runtime_sampler_gathers_and_nearest_samples_fetch_by_their_word() {
    let cases = [
        Read::Gather { component: 0 },
        Read::Gather { component: 3 },
        Read::Sample,
        Read::Level,
        Read::MinLodClamp,
        Read::Array,
        Read::OneD,
        Read::ThreeD,
        Read::Depth,
    ];
    let profiles = crate::spirv::testing::committed_exact_profiles().expect("profiles");
    for read in cases {
        let name = format!("{read:?} through a runtime sampler");
        let Some(bc) = assemble(&name, &runtime_kernel(read)) else {
            return;
        };
        let gather = matches!(read, Read::Gather { .. });
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
            let fetches = ops.contains(&FETCH);
            let device = ops.iter().any(|o| DEVICE_READS.contains(o));
            let format = a
                .interface
                .read_keys()
                .into_iter()
                .any(|k| matches!(k, SpecKey::TexelFormat(_)));
            assert_eq!((fetches, device, format), (!metal, true, !metal), "{what}");
            // A specialized note is among the module's notes.
            let noted = a.precision.contains(&PrecisionNote::DeviceTexelDenormals);
            let specialized = a
                .specialized
                .iter()
                .any(|n| n.note == PrecisionNote::DeviceTexelDenormals);
            assert_eq!(
                (noted, specialized),
                (!metal && !gather, !metal && !gather),
                "{what}"
            );
            if metal {
                continue;
            }
            let fields = a
                .interface
                .read_keys()
                .into_iter()
                .find_map(|k| match k {
                    SpecKey::SamplerState { fields, .. } => Some(fields),
                    _ => None,
                })
                .unwrap_or_else(|| panic!("{what}: no sampler word"));
            let mut want = SamplerWord::PIXEL_COORDINATES
                | SamplerWord::ADDRESS[0]
                | SamplerWord::ADDRESS[1]
                | SamplerWord::BORDER;
            if !gather {
                want |= SamplerWord::LINEAR_FILTER;
            }
            if matches!(read, Read::ThreeD) {
                want |= SamplerWord::ADDRESS[2];
            }
            if matches!(read, Read::Level) {
                want |= SamplerWord::LEVEL_MIN | SamplerWord::LEVEL_MAX;
            }
            assert_eq!(fields & want, want, "{what}: word fields {fields:#x}");
        }
    }
}

/// On the interpreter: through a runtime sampler slot whose word states `MTLSamplerDescriptor`'s
/// defaults (nearest, no mip filter, clamped to the edge), a gather at each interior texel corner
/// reads Metal's 2x2 footprint and a sample at each texel centre that texel, every denormal and
/// signed zero kept, through the fetches the word addresses.
#[test]
fn runtime_sampler_fetches_keep_their_denormals() {
    let texel = |x: u32, y: u32| TEXELS[(4 * y + x) as usize];
    let Some(gather) = assemble(
        "runtime gather",
        &runtime_kernel(Read::Gather { component: 0 }),
    ) else {
        return;
    };
    let corners: Vec<(u32, u32)> = (0..3).flat_map(|j| (0..3).map(move |i| (i, j))).collect();
    let points: Vec<[f32; 4]> = corners
        .iter()
        .map(|&(i, j)| [(i + 1) as f32 / 4.0, (j + 1) as f32 / 4.0, 0.0, 0.0])
        .collect();
    for ((i, j), got) in corners.iter().zip(run("runtime gather", &gather, &points)) {
        let want = [
            texel(*i, j + 1),
            texel(i + 1, j + 1),
            texel(i + 1, *j),
            texel(*i, *j),
        ];
        assert_eq!(got, want, "gather at corner ({i}, {j})");
    }
    let Some(sample) = assemble("runtime sample", &runtime_kernel(Read::Sample)) else {
        return;
    };
    let centres: Vec<(u32, u32)> = (0..4).flat_map(|y| (0..4).map(move |x| (x, y))).collect();
    let points: Vec<[f32; 4]> = centres
        .iter()
        .map(|&(x, y)| [(x as f32 + 0.5) / 4.0, (y as f32 + 0.5) / 4.0, 0.0, 0.0])
        .collect();
    for ((x, y), got) in centres.iter().zip(run("runtime sample", &sample, &points)) {
        assert_eq!(
            got,
            [texel(*x, *y), 0, 0, 1f32.to_bits()],
            "sample at ({x}, {y})"
        );
    }
}
