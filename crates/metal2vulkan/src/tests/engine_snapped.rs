//! The coordinates a device's gathers and repeating nearest samples read. On a device without
//! `VkProperty::MetalSampler`, a gather through a constexpr normalized sampler of a 2D texture or
//! array (with or without a comparison or an offset) hands the device a coordinate snapped onto
//! Metal's 2x2 block (`sampler.gather.snapped`, `sampler.gather_compare.snapped`), and a runtime
//! sampler's behind its word; every other gather (a cube's) keeps the device's footprint under
//! `PrecisionNote::DeviceGatherFootprint`; a nearest sample through a constexpr normalized sampler
//! with a repeating or mirrored axis that reads the base level snaps those axes
//! (`sampler.nearest.snapped`), and such a nearest sample of a cube its direction onto the centre
//! of Metal's texel (`sampler.cube.snapped`) under `PrecisionNote::CubeFaceReciprocal`. On a device
//! with the property all are the device's, Metal's own sampler, and the module requires it. A
//! constexpr pixel sampler's position is the coordinate as given, on every device: its gather snaps
//! from it, and its linear sample is emulated from it or, with the property, snapped onto its
//! footprint cell (`sampler.linear.snapped`), none of them noted `PixelSamplerScaling`; and a cube
//! reads it as a normalized sampler. Every module validates on every committed profile, and the
//! plan's word estimate bounds what it emits.
//!
//! The snapped coordinate's arithmetic is checked bit for bit on the interpreter by
//! `lower::sampler::position`'s and `lower::sampler::face`'s tests (the interpreter has no gather
//! or sample).
//!
//! `llvm-as` assembles the kernels; without it the tests skip visibly.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::device::profile::DeviceProfile;
use crate::device::requirements::VkProperty;
use crate::library::FunctionArtifact;
use crate::pipeline::artifact::PrecisionNote;
use crate::pipeline::testing::{
    compile, engine_dir, link_and_validate, llvm_assemble, AssembleError,
};

const HEADER: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n";

/// Bitcode of the module `body` (after the AIR header), or `None` without `llvm-as`.
fn assemble(name: &str, body: &str) -> Option<Vec<u8>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "m2v-snapped-{}-{}-{}",
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

/// AIR address mode codes.
const ZERO: u64 = 0;
const EDGE: u64 = 1;
const REPEAT: u64 = 2;
const MIRROR: u64 = 3;

/// The words of a constexpr sampler: minification and magnification `filter` (0 nearest, 1
/// linear), mip filter `mip` (0 none, 1 nearest, 2 linear), axes `s`, `t`, `r` addressed by `address`'s
/// codes, and `pixel` coordinates, with the fields Apple's compiler writes for the rest (the
/// `hlf_half_address` fixture's sampler).
fn sampler(filter: u64, mip: u64, address: [u64; 3], pixel: bool) -> u64 {
    const BASE: u64 = 0x007b_ff00_0008_0000;
    BASE | address[0]
        | address[1] << 3
        | address[2] << 6
        | filter << 9
        | filter << 11
        | mip << 13
        | u64::from(pixel) << 15
}

/// What a test kernel does with texture slot 0 at `float2` buffer 0's lane `tid`, storing the
/// `float4` to buffer 1.
#[derive(Clone, Copy, Debug)]
enum Op {
    /// `texture2d<float>::gather(s, uv, offset, component::y)`.
    Gather { offset: (i32, i32) },
    /// `texture2d_array<float>::gather(s, uv, 1, int2(0), component::x)`.
    GatherArray,
    /// `texture1d<float>::sample(s, uv.x)`.
    Sample1D,
    /// `texture2d<float>::sample(s, uv)`.
    Sample2D,
    /// `texture2d<float>::sample(s, uv, level(lod))` with a dynamic level (buffer 2).
    SampleLevel,
    /// `texture3d<float>::sample(s, float3(uv, 0.5))`.
    Sample3D,
    /// `texturecube<float>::sample(s, float3(uv, 0.5))`.
    SampleCube,
}

/// A kernel performing `op` through constexpr sampler `words` (`None`: the runtime
/// `[[sampler(0)]]`).
fn kernel(op: Op, words: Option<u64>) -> String {
    let s = if words.is_some() {
        "@__air_sampler_state"
    } else {
        "%s"
    };
    let (tex_ty, call, decl) = match op {
        Op::Gather { offset: (ox, oy) } => (
            "texture2d<float, sample>",
            format!("@air.gather_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) {s}, <2 x float> %uv, i1 true, <2 x i32> <i32 {ox}, i32 {oy}>, i32 1, i32 0)"),
            "@air.gather_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i32, i32)",
        ),
        Op::GatherArray => (
            "texture2d_array<float, sample>",
            format!("@air.gather_texture_2d_array.v4f32(ptr addrspace(1) %t, ptr addrspace(2) {s}, <2 x float> %uv, i32 1, i1 true, <2 x i32> zeroinitializer, i32 0, i32 0)"),
            "@air.gather_texture_2d_array.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i32, i1, <2 x i32>, i32, i32)",
        ),
        Op::Sample1D => (
            "texture1d<float, sample>",
            format!("@air.sample_texture_1d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) {s}, float %u, i1 false, i32 0, i1 false, float 0.0, float 0.0, i32 0)"),
            "@air.sample_texture_1d.v4f32(ptr addrspace(1), ptr addrspace(2), float, i1, i32, i1, float, float, i32)",
        ),
        Op::Sample2D => (
            "texture2d<float, sample>",
            format!("@air.sample_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) {s}, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.0, float 0.0, i32 0)"),
            "@air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)",
        ),
        Op::SampleLevel => (
            "texture2d<float, sample>",
            format!("@air.sample_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) {s}, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 true, float %lod, float 0.0, i32 0)"),
            "@air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)",
        ),
        Op::Sample3D => (
            "texture3d<float, sample>",
            format!("@air.sample_texture_3d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) {s}, <3 x float> %uvw, i1 true, <3 x i32> zeroinitializer, i1 false, float 0.0, float 0.0, i32 0)"),
            "@air.sample_texture_3d.v4f32(ptr addrspace(1), ptr addrspace(2), <3 x float>, i1, <3 x i32>, i1, float, float, i32)",
        ),
        Op::SampleCube => (
            "texturecube<float, sample>",
            format!("@air.sample_texture_cube.v4f32(ptr addrspace(1) %t, ptr addrspace(2) {s}, <3 x float> %uvw, i1 false, float 0.0, float 0.0, i32 0)"),
            "@air.sample_texture_cube.v4f32(ptr addrspace(1), ptr addrspace(2), <3 x float>, i1, float, float, i32)",
        ),
    };
    let level = matches!(op, Op::SampleLevel);
    let mut params = String::new();
    let mut mds = Vec::new();
    let mut next = 3;
    if level {
        params.push_str(", ptr addrspace(1) %lods");
        mds.push(format!("!{{i32 {next}, !\"air.buffer\", !\"air.location_index\", i32 2, i32 1, !\"air.read\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"float\", !\"air.arg_name\", !\"lods\"}}"));
        next += 1;
    }
    if words.is_none() {
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
    let load_lod = if level {
        "  %pl = getelementptr inbounds float, ptr addrspace(1) %lods, i64 %i\n  %lod = load float, ptr addrspace(1) %pl, align 4\n"
    } else {
        ""
    };
    let (state, states, state_md) = match words {
        Some(w) => (
            format!("@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 {w}, i64 0], align 8\n\n"),
            "!air.sampler_states = !{!6}\n",
            "!6 = !{!\"air.sampler_state\", ptr addrspace(2) @__air_sampler_state}\n",
        ),
        None => (String::new(), "", ""),
    };
    format!(
        r#"{state}define void @k(ptr addrspace(1) %t, ptr addrspace(1) %c, ptr addrspace(1) %out{params}, i32 %tid) {{
  %i = zext i32 %tid to i64
  %p = getelementptr inbounds <2 x float>, ptr addrspace(1) %c, i64 %i
  %uv = load <2 x float>, ptr addrspace(1) %p, align 8
  %u = extractelement <2 x float> %uv, i64 0
  %uvw = shufflevector <2 x float> %uv, <2 x float> <float 0.5, float 0.5>, <3 x i32> <i32 0, i32 1, i32 2>
{load_lod}  %r = call {{ <4 x float>, i8 }} {call}
  %v = extractvalue {{ <4 x float>, i8 }} %r, 0
  %q = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i64 %i
  store <4 x float> %v, ptr addrspace(1) %q, align 16
  ret void
}}

declare {{ <4 x float>, i8 }} {decl}

!air.kernel = !{{!0}}
{states}!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4, !5, {arg_mds}}}
!3 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"{tex_ty}", !"air.arg_name", !"t"}}
!4 = !{{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"float2", !"air.arg_name", !"c"}}
!5 = !{{i32 2, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}}
{state_md}{tail}"#,
        arg_mds = arg_mds.join(", "),
    )
}

fn uses(a: &FunctionArtifact, name: &str) -> u32 {
    a.strategies.entries.get(name).map_or(0, |u| u.uses)
}

fn metal_sampler(p: &DeviceProfile) -> bool {
    p.fields()
        .enabled
        .properties
        .contains(&VkProperty::MetalSampler)
}

/// How a kernel's one access compiles on a device without `VkProperty::MetalSampler`: its census
/// name, and which of the notes `NOTES` the module records.
struct Expect {
    name: &'static str,
    note: Option<PrecisionNote>,
}

/// The notes of an access the module leaves to the device's position, or snaps near Metal's.
const NOTES: [PrecisionNote; 2] = [
    PrecisionNote::DeviceGatherFootprint,
    PrecisionNote::CubeFaceReciprocal,
];

/// The kernels of the selection tests, with what each compiles to on a device without the
/// property; on one with it, each `*.snapped` or noted access is the device's (`plain`) and the
/// module requires the property.
fn cases() -> Vec<(&'static str, Op, Option<u64>, Expect, &'static str)> {
    let near = |a: [u64; 3]| Some(sampler(0, 0, a, false));
    let gather = Op::Gather { offset: (0, 0) };
    let snapped = |name| Expect { name, note: None };
    vec![
        (
            "gather edge",
            gather,
            near([EDGE; 3]),
            snapped("sampler.gather.snapped"),
            "sampler.gather",
        ),
        (
            "gather zero",
            gather,
            near([ZERO; 3]),
            snapped("sampler.gather.snapped"),
            "sampler.gather",
        ),
        (
            "gather repeat offset",
            Op::Gather { offset: (1, -1) },
            near([REPEAT; 3]),
            snapped("sampler.gather.snapped"),
            "sampler.gather",
        ),
        (
            "gather mirror",
            gather,
            near([MIRROR; 3]),
            snapped("sampler.gather.snapped"),
            "sampler.gather",
        ),
        (
            "gather linear sampler",
            gather,
            Some(sampler(1, 0, [EDGE; 3], false)),
            snapped("sampler.gather.snapped"),
            "sampler.gather",
        ),
        (
            "gather array",
            Op::GatherArray,
            near([EDGE; 3]),
            snapped("sampler.gather.snapped"),
            "sampler.gather",
        ),
        (
            // Snapped behind its word, under the operation's own name.
            "gather runtime sampler",
            gather,
            None,
            snapped("sampler.gather"),
            "sampler.gather",
        ),
        (
            "nearest repeat 2d",
            Op::Sample2D,
            near([REPEAT; 3]),
            snapped("sampler.nearest.snapped"),
            "sampler.sample",
        ),
        (
            "nearest mirror s, edge t",
            Op::Sample2D,
            near([MIRROR, EDGE, EDGE]),
            snapped("sampler.nearest.snapped"),
            "sampler.sample",
        ),
        (
            "nearest repeat 1d",
            Op::Sample1D,
            near([REPEAT; 3]),
            snapped("sampler.nearest.snapped"),
            "sampler.sample",
        ),
        (
            "nearest repeat r only 3d",
            Op::Sample3D,
            near([EDGE, EDGE, REPEAT]),
            snapped("sampler.nearest.snapped"),
            "sampler.sample",
        ),
        (
            "nearest repeat, nearest mip filter, level 0",
            Op::Sample2D,
            Some(sampler(0, 1, [REPEAT; 3], false)),
            snapped("sampler.nearest.snapped"),
            "sampler.sample",
        ),
        (
            "nearest repeat, linear mip filter, level 0",
            Op::Sample2D,
            Some(sampler(0, 2, [REPEAT; 3], false)),
            snapped("sampler.sample"),
            "sampler.sample",
        ),
        (
            "nearest edge",
            Op::Sample2D,
            near([EDGE; 3]),
            snapped("sampler.sample"),
            "sampler.sample",
        ),
        (
            "nearest repeat, nearest mip filter, dynamic level",
            Op::SampleLevel,
            Some(sampler(0, 1, [REPEAT; 3], false)),
            snapped("sampler.sample"),
            "sampler.sample",
        ),
        (
            // A runtime sampler's sample is planned for the linear emulation, which its word
            // selects; a nearest word selects the device's sample.
            "nearest repeat runtime sampler",
            Op::Sample2D,
            None,
            snapped("sampler.linear"),
            "sampler.sample",
        ),
        (
            "nearest cube",
            Op::SampleCube,
            near([EDGE; 3]),
            Expect {
                name: "sampler.cube.snapped",
                note: Some(PrecisionNote::CubeFaceReciprocal),
            },
            "sampler.sample",
        ),
        (
            "nearest cube, nearest mip filter, level 0",
            Op::SampleCube,
            Some(sampler(0, 1, [REPEAT; 3], false)),
            Expect {
                name: "sampler.cube.snapped",
                note: Some(PrecisionNote::CubeFaceReciprocal),
            },
            "sampler.sample",
        ),
        (
            "nearest cube, linear mip filter, level 0",
            Op::SampleCube,
            Some(sampler(0, 2, [EDGE; 3], false)),
            snapped("sampler.sample"),
            "sampler.sample",
        ),
        (
            // Emulated on Metal's position.
            "linear cube",
            Op::SampleCube,
            Some(sampler(1, 0, [EDGE; 3], false)),
            Expect {
                name: "sampler.linear",
                note: Some(PrecisionNote::CubeFaceReciprocal),
            },
            "sampler.sample",
        ),
        (
            "nearest cube runtime sampler",
            Op::SampleCube,
            None,
            Expect {
                name: "sampler.linear",
                note: Some(PrecisionNote::CubeFaceReciprocal),
            },
            "sampler.sample",
        ),
    ]
}

/// On every committed profile, each kernel compiles to its expected access and notes, and
/// validates; on a device with `VkProperty::MetalSampler` every gather and every snapped nearest
/// sample through a constexpr normalized sampler requires the property instead, as does a sample at
/// a dynamic level through one, whose level is Metal's there.
#[test]
fn gathers_and_repeating_nearest_samples_snap_where_metal_reads_them() {
    let profiles = crate::spirv::testing::committed_exact_profiles().expect("profiles");
    for (name, op, words, expect, plain) in cases() {
        let Some(bc) = assemble(name, &kernel(op, words)) else {
            return;
        };
        let snapped = expect.name.ends_with(".snapped");
        for (stem, p) in &profiles {
            let what = format!("{name} on {stem}");
            let a = compile(&bc, "k", p).unwrap_or_else(|e| panic!("{what}: {e}"));
            link_and_validate(&a, p).unwrap_or_else(|e| panic!("{what}: {e}"));
            let noted = |n: &PrecisionNote| a.precision.contains(n);
            let required = a
                .requirements
                .properties
                .contains(&VkProperty::MetalSampler);
            if metal_sampler(p) {
                assert_eq!(uses(&a, plain), 1, "{what}");
                assert!(!NOTES.iter().any(noted), "{what}");
                // A runtime sampler's or a pixel sampler's gather requires it too: every gather
                // is Metal's own there; and so does a sample that filters or may, a linear mip
                // filter's and a runtime sampler's included.
                let gathers = matches!(op, Op::Gather { .. } | Op::GatherArray);
                let filtered = words.is_none_or(|w| (w >> 9) & 0xf != 0 || (w >> 13) & 3 == 2);
                // A sample at a dynamic level through a constexpr normalized sampler reads Metal's
                // level, the device's own there.
                let leveled =
                    matches!(op, Op::SampleLevel) && words.is_some_and(|w| w >> 15 & 1 == 0);
                assert_eq!(
                    required,
                    snapped || gathers || filtered || leveled,
                    "{what}"
                );
            } else {
                assert_eq!(uses(&a, expect.name), 1, "{what}");
                for n in &NOTES {
                    assert_eq!(noted(n), expect.note == Some(*n), "{what}: {n:?}");
                }
                assert!(!required, "{what}");
            }
        }
    }
}

/// Over an owned fixture: `unk_gather`'s comparison gathers of a depth 2D texture and array through
/// a constexpr sampler snap; its five gathers of cubes (three color, two depth) and two comparison
/// gathers of cubes take Metal's footprint by fetches (`sampler.gather.cube` and
/// `sampler.gather_compare.cube`), and keep the device's under the note where the bound format's
/// rule is not measured.
#[test]
fn the_fixture_cube_gathers_take_metals_footprint() {
    let Ok(bc) = llvm_assemble(&engine_dir().join("unk_gather.ll")) else {
        eprintln!("SKIPPED: llvm-as");
        return;
    };
    for (stem, p) in crate::spirv::testing::committed_exact_profiles().expect("profiles") {
        let a = match compile(&bc, "unk_gather", &p) {
            Ok(a) => a,
            // A profile without cube arrays refuses the fixture.
            Err(e) => {
                eprintln!("{stem}: refused: {e}");
                continue;
            }
        };
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        let noted = a.precision.contains(&PrecisionNote::DeviceGatherFootprint);
        if metal_sampler(&p) {
            assert!(!noted, "{stem}");
            assert_eq!(uses(&a, "sampler.gather"), 5, "{stem}");
            assert_eq!(uses(&a, "sampler.gather_compare"), 4, "{stem}");
        } else {
            assert!(noted, "{stem}");
            assert_eq!(uses(&a, "sampler.gather"), 0, "{stem}");
            assert_eq!(uses(&a, "sampler.gather.cube"), 5, "{stem}");
            assert_eq!(uses(&a, "sampler.gather_compare"), 0, "{stem}");
            assert_eq!(uses(&a, "sampler.gather_compare.cube"), 2, "{stem}");
            assert_eq!(uses(&a, "sampler.gather_compare.snapped"), 2, "{stem}");
        }
    }
}

/// The kernels of the pixel sampler test: each through a constexpr pixel sampler, with its census
/// name on a device without `VkProperty::MetalSampler` and on one with it (`None`: a cube's,
/// which compiles as through the same sampler with normalized coordinates).
fn pixel_cases() -> Vec<(&'static str, Op, u64, Option<(&'static str, &'static str)>)> {
    let pixel = |filter, address| sampler(filter, 0, address, true);
    let snapped = Some(("sampler.gather.snapped", "sampler.gather.snapped"));
    let linear = Some(("sampler.linear", "sampler.linear.snapped"));
    vec![
        (
            "pixel gather edge",
            Op::Gather { offset: (0, 0) },
            pixel(0, [EDGE; 3]),
            snapped,
        ),
        (
            "pixel gather zero offset",
            Op::Gather { offset: (2, -3) },
            pixel(0, [ZERO; 3]),
            snapped,
        ),
        (
            "pixel gather linear sampler",
            Op::Gather { offset: (0, 0) },
            pixel(1, [EDGE; 3]),
            snapped,
        ),
        (
            "pixel gather array",
            Op::GatherArray,
            pixel(0, [ZERO, EDGE, EDGE]),
            snapped,
        ),
        ("pixel linear 1d", Op::Sample1D, pixel(1, [EDGE; 3]), linear),
        (
            "pixel linear 2d",
            Op::Sample2D,
            pixel(1, [ZERO, EDGE, EDGE]),
            linear,
        ),
        ("pixel linear 3d", Op::Sample3D, pixel(1, [EDGE; 3]), linear),
        (
            "pixel nearest cube",
            Op::SampleCube,
            pixel(0, [EDGE; 3]),
            None,
        ),
        (
            "pixel linear cube",
            Op::SampleCube,
            pixel(1, [EDGE, ZERO, EDGE]),
            None,
        ),
    ]
}

/// On every committed profile, a constexpr pixel sampler's gather and linear sample compile to
/// their census names (a linear one requiring `VkProperty::MetalSampler` where it snaps), validate,
/// and record neither `PixelSamplerScaling` nor `DeviceGatherFootprint`, but a linear sample's
/// `PixelSamplerScaling` where the bound format's class hands it to the device's filter at the
/// scaled coordinate (a specialized note); a cube through a pixel sampler compiles as through the
/// same sampler with normalized coordinates, to the same census, notes, and requirements.
#[test]
fn pixel_samplers_read_metals_position_on_every_device() {
    let profiles = crate::spirv::testing::committed_exact_profiles().expect("profiles");
    for (name, op, words, expect) in pixel_cases() {
        let Some(bc) = assemble(name, &kernel(op, Some(words))) else {
            return;
        };
        let normalized = match expect {
            Some(_) => None,
            None => assemble(name, &kernel(op, Some(words & !(1 << 15)))),
        };
        for (stem, p) in &profiles {
            let what = format!("{name} on {stem}");
            let a = compile(&bc, "k", p).unwrap_or_else(|e| panic!("{what}: {e}"));
            link_and_validate(&a, p).unwrap_or_else(|e| panic!("{what}: {e}"));
            let required = a
                .requirements
                .properties
                .contains(&VkProperty::MetalSampler);
            match (expect, &normalized) {
                (Some((without, with)), _) => {
                    let census = if metal_sampler(p) { with } else { without };
                    assert_eq!(uses(&a, census), 1, "{what}");
                    let gather = matches!(op, Op::Gather { .. } | Op::GatherArray);
                    for n in [
                        PrecisionNote::PixelSamplerScaling,
                        PrecisionNote::DeviceGatherFootprint,
                    ] {
                        let device_arm = n == PrecisionNote::PixelSamplerScaling
                            && !gather
                            && !metal_sampler(p)
                            && a.specialized.iter().any(|s| s.note == n);
                        assert!(!a.precision.contains(&n) || device_arm, "{what}: {n:?}");
                    }
                    assert_eq!(required, metal_sampler(p) && !gather, "{what}");
                }
                (None, Some(nbc)) => {
                    let b = compile(nbc, "k", p).unwrap_or_else(|e| panic!("{what}: {e}"));
                    assert_eq!(a.strategies.entries, b.strategies.entries, "{what}");
                    assert_eq!(a.precision, b.precision, "{what}");
                    assert_eq!(a.requirements, b.requirements, "{what}");
                }
                (None, None) => return,
            }
        }
    }
}

/// The fields of the word compiled `a`'s runtime sampler slot specializes.
fn sampler_fields(a: &FunctionArtifact) -> u32 {
    use crate::pipeline::sym::SpecKey;
    a.interface
        .read_keys()
        .into_iter()
        .find_map(|k| match k {
            SpecKey::SamplerState { fields, .. } => Some(fields),
            _ => None,
        })
        .expect("a `SamplerState` specialization")
}

/// On every committed profile, a runtime sampler's gather of a 2D texture or array snaps behind its
/// word (`position::SnapKind::WordGather`): the function reads the word's coordinates and its `s`
/// and `t` modes, and records neither `PixelSamplerScaling` nor `DeviceGatherFootprint`, every
/// word's gather reading Metal's block. On a device with `VkProperty::MetalSampler` it requires the
/// property (a normalized word's gather is Metal's own). A runtime sampler's 2D sample on a device
/// with the property reads whether the word states a linear pixel sampler
/// (`position::SnapKind::WordCell`; elsewhere the linear emulation reads it), and keeps
/// `PixelSamplerScaling`.
#[test]
fn a_runtime_samplers_gather_snaps_behind_its_word() {
    use crate::metal::sampler::SamplerWord;
    let profiles = crate::spirv::testing::committed_exact_profiles().expect("profiles");
    let gather = SamplerWord::PIXEL_COORDINATES | SamplerWord::ADDRESS[0] | SamplerWord::ADDRESS[1];
    let cell = SamplerWord::PIXEL_COORDINATES | SamplerWord::MIN_LINEAR | SamplerWord::MAG_LINEAR;
    for (name, op) in [
        ("runtime gather", Op::Gather { offset: (0, 0) }),
        ("runtime gather offset", Op::Gather { offset: (2, -1) }),
        ("runtime gather array", Op::GatherArray),
        ("runtime sample", Op::Sample2D),
    ] {
        let Some(bc) = assemble(name, &kernel(op, None)) else {
            return;
        };
        for (stem, p) in &profiles {
            let what = format!("{name} on {stem}");
            let a = compile(&bc, "k", p).unwrap_or_else(|e| panic!("{what}: {e}"));
            link_and_validate(&a, p).unwrap_or_else(|e| panic!("{what}: {e}"));
            let fields = sampler_fields(&a);
            let noted = |n: PrecisionNote| a.precision.contains(&n);
            let metal = metal_sampler(p);
            if matches!(op, Op::Sample2D) {
                // Elsewhere the linear emulation reads those fields and more.
                assert_eq!(fields & cell, cell, "{what}: {fields:#x}");
                assert!(noted(PrecisionNote::PixelSamplerScaling), "{what}");
                continue;
            }
            assert_eq!(fields & gather, gather, "{what}: {fields:#x}");
            assert!(!noted(PrecisionNote::PixelSamplerScaling), "{what}");
            assert!(!noted(PrecisionNote::DeviceGatherFootprint), "{what}");
            let required = a
                .requirements
                .properties
                .contains(&VkProperty::MetalSampler);
            assert_eq!(required, metal, "{what}");
        }
    }
}

/// The plan's word estimate bounds the emitted module of every kernel above on every committed
/// profile.
#[test]
fn the_snapped_estimate_bounds_its_emission() {
    use crate::budget::SPIRV_WORDS;
    use crate::library::FunctionConstantValues;
    use crate::lower::select::LowerInputs;
    use crate::pipeline::compile::{compile_bitcode, plan_bitcode, EntryRef};
    use crate::pipeline::testing::meter;
    let profiles = crate::spirv::testing::committed_exact_profiles().expect("profiles");
    let pixel = pixel_cases()
        .into_iter()
        .map(|(n, o, w, _)| (n, o, Some(w)));
    for (name, op, words) in cases()
        .into_iter()
        .map(|(n, o, w, _, _)| (n, o, w))
        .chain(pixel)
    {
        let Some(bc) = assemble(name, &kernel(op, words)) else {
            return;
        };
        for (stem, p) in &profiles {
            let what = format!("{name} on {stem}");
            let (entry, consts) = (EntryRef::Name("k"), FunctionConstantValues::defaults());
            let inputs = LowerInputs::new();
            let plan = plan_bitcode(&bc, entry, &consts, p, &inputs, &mut meter())
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            let a = compile_bitcode(&bc, entry, &consts, p, &inputs, &mut meter(), SPIRV_WORDS)
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            let emitted = a.module.words.len() as u64;
            eprintln!("{what}: estimate {} emitted {emitted}", plan.estimate.words);
            assert!(
                plan.estimate.words >= emitted,
                "{what}: estimate {} < emitted {emitted}",
                plan.estimate.words
            );
        }
    }
}
