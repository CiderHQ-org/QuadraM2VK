//! The sampling operations the decoder refused previously end to end: family B, over the owned
//! fixtures `unk_sample_grad`, `unk_gather`, and `unk_calculate_lod`, on every committed profile
//! that compiles them (and `moltenvk-m3` and `lavapipe` without robustness). Every module links and
//! passes L3 (`G-val`). The CPU interpreter does not model sampling, so the tests check the emitted
//! instructions:
//!
//! - gradient samples of 3D, cube, and cube array textures and of depth 2D, 2D array, cube, and
//!   cube array textures, with and without comparison, are explicit-LOD samples with the `Grad`
//!   operand (a comparison's with its reference, but a depth 2D or 2D array texture's through a
//!   nearest sampler on a device without `VkProperty::MetalSampler`, which the shader compares); a
//!   depth cube array's sample and a depth cube and cube array's comparison at level 0 are
//!   explicit-LOD samples with the `Lod` operand;
//! - gathers of cube and cube array textures read their component, depth gathers component 0, and
//!   comparison gathers are `OpImageDrefGather`, none with an offset (a cube takes none), except a
//!   depth 2D or 2D array texture's on a device without `VkProperty::MetalSampler`, which reads
//!   component 0 for the shader to compare;
//! - the clamped and unclamped LOD queries of 2D array, 3D, cube, and cube array textures and of
//!   depth 2D, 2D array, cube, and cube array textures are `OpImageQueryLod`;
//! - a cube's explicit gradients take Metal's footprint (`sampler.sample_cube_grad`, with
//!   `PrecisionNote::CubeGradientFootprint`), other gradients pass unchanged, and an anisotropic
//!   constexpr sampler's keep Metal's two face components.
//!
//! Previously each kernel was refused `UnknownIntrinsic`, so every test fails without it. A profile
//! without cube arrays (`vk12-floor`) refuses the fixtures with the typed device gap, which the
//! tests accept and count.
//!
//! `llvm-as` assembles the fixtures; without it the tests skip visibly.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::device::profile::{CubeGradientRoute, DeviceProfile};
use crate::device::requirements::{VkFeature, VkProperty};
use crate::device::{Device, DeviceDescription};
use crate::error::{Error, Needs};
use crate::library::FunctionArtifact;
use crate::pipeline::testing::EXACT_SAMPLING;
use crate::pipeline::testing::{
    compile, engine_dir, link_and_validate, llvm_assemble, AssembleError,
};
use crate::pipeline::PrecisionNote;
use crate::spirv::testing::committed_exact_profiles;

const OP_EXT_INST: u16 = 12;
const OP_CONSTANT: u16 = 43;
/// `Length` in `GLSL.std.450`.
const GLSL_LENGTH: u32 = 66;
const OP_IMAGE_SAMPLE_IMPLICIT_LOD: u16 = 87;
const OP_IMAGE_SAMPLE_EXPLICIT_LOD: u16 = 88;
const OP_IMAGE_SAMPLE_DREF_IMPLICIT_LOD: u16 = 89;
const OP_IMAGE_SAMPLE_DREF_EXPLICIT_LOD: u16 = 90;
const OP_IMAGE_GATHER: u16 = 96;
const OP_IMAGE_DREF_GATHER: u16 = 97;
const OP_IMAGE_QUERY_LOD: u16 = 105;
const LOD: u32 = 0x2;
const GRAD: u32 = 0x4;
const CONST_OFFSET: u32 = 0x8;
const OFFSET: u32 = 0x10;

fn fixture(name: &str) -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join(format!("{name}.ll"))) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {name}: {why}"),
    }
}

/// `text` (a whole module) assembled, through a scratch file removed afterwards.
fn assemble_text(name: &str, text: &str) -> Option<Vec<u8>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "m2v-sampler-unk-{}-{}-{name}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).expect("scratch");
    let ll = dir.join("k.ll");
    std::fs::write(&ll, text).expect("write");
    let r = llvm_assemble(&ll);
    let _ = std::fs::remove_dir_all(&dir);
    match r {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {name}: {why}"),
    }
}

/// `stem` with the robustness features removed.
fn unrobust(stem: &str) -> DeviceProfile {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    let mut desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    for f in [
        VkFeature::RobustImageAccess,
        VkFeature::RobustImageAccess2,
        VkFeature::RobustBufferAccess,
        VkFeature::RobustBufferAccess2,
    ] {
        desc.features.remove(&f);
    }
    Device::new(desc, EXACT_SAMPLING)
        .expect("device")
        .profile()
        .clone()
}

/// Every committed profile, and `moltenvk-m3` and `lavapipe` without robustness.
fn profiles() -> Vec<(String, DeviceProfile)> {
    let mut out = committed_exact_profiles().expect("profiles");
    for stem in ["moltenvk-m3", "lavapipe"] {
        out.push((format!("{stem} unrobust"), unrobust(stem)));
    }
    out
}

/// The artifact of `entry` on `p`, linked and past L3; `None` when `p` lacks a capability it
/// needs (a typed device gap). Any other refusal fails the test.
fn artifact(
    bc: &[u8],
    entry: &str,
    stem: &str,
    p: &DeviceProfile,
) -> Option<std::sync::Arc<FunctionArtifact>> {
    match compile(bc, entry, p) {
        Ok(a) => {
            link_and_validate(&a, p).unwrap_or_else(|e| panic!("{stem}: {entry}: L3: {e}"));
            Some(a)
        }
        Err(Error::Unsupported(u)) if matches!(u.needs, Needs::AnyOf(_)) => None,
        Err(e) => panic!("{stem}: {entry}: {e}"),
    }
}

/// The instructions of a SPIR-V module: opcode and its words.
fn insts(words: &[u32]) -> Vec<(u16, &[u32])> {
    let mut out = Vec::new();
    let mut i = 5;
    while i < words.len() {
        let n = ((words[i] >> 16) as usize).max(1);
        out.push((
            (words[i] & 0xFFFF) as u16,
            &words[i..(i + n).min(words.len())],
        ));
        i += n;
    }
    out
}

/// The image-operand mask of each instruction of opcode `op` whose mask is word `at` (0 when the
/// instruction has none).
fn masks(a: &FunctionArtifact, op: u16, at: usize) -> Vec<u32> {
    insts(&a.module.words)
        .into_iter()
        .filter(|(o, _)| *o == op)
        .map(|(_, w)| w.get(at).copied().unwrap_or(0))
        .collect()
}

fn count(a: &FunctionArtifact, op: u16) -> usize {
    masks(a, op, 0).len()
}

/// Each profile's artifact of `entry`, checked by `check`; returns how many compiled and how many
/// were refused for a device gap.
fn each(name: &str, entry: &str, check: impl Fn(&str, &FunctionArtifact)) -> Option<(u32, u32)> {
    let bc = fixture(name)?;
    let (mut compiled, mut gaps) = (0, 0);
    for (stem, p) in profiles() {
        match artifact(&bc, entry, &stem, &p) {
            Some(a) => {
                check(&stem, &a);
                compiled += 1;
            }
            None => gaps += 1,
        }
    }
    Some((compiled, gaps))
}

#[test]
fn gradient_and_level_samples_of_3d_cube_and_depth_textures() {
    let metal: Vec<String> = profiles()
        .into_iter()
        .filter(|(_, p)| {
            p.fields()
                .enabled
                .properties
                .contains(&VkProperty::MetalSampler)
        })
        .map(|(stem, _)| stem)
        .collect();
    let Some((compiled, _)) = each("unk_sample_grad", "unk_sample_grad", |stem, a| {
        // The depth 2D and 2D array gradient comparisons through the nearest sampler compare in
        // the shader off `VkProperty::MetalSampler`: the device reads their texel by a plain
        // gradient sample, as it would compare it, at the device's level.
        let compared = if metal.iter().any(|m| m == stem) {
            0
        } else {
            2
        };
        // A kernel samples nothing implicitly.
        for op in [
            OP_IMAGE_SAMPLE_IMPLICIT_LOD,
            OP_IMAGE_SAMPLE_DREF_IMPLICIT_LOD,
        ] {
            assert_eq!(count(a, op), 0, "{stem}: {op}");
        }
        let lod = |m: &u32| m & (LOD | GRAD);
        // Three color and four depth gradient samples, and the compared ones; the depth cube
        // array's at level 0, twice.
        let mut plain: Vec<u32> = masks(a, OP_IMAGE_SAMPLE_EXPLICIT_LOD, 5)
            .iter()
            .map(lod)
            .collect();
        plain.sort();
        let mut expected = vec![LOD, LOD];
        expected.extend(vec![GRAD; 7 + compared]);
        assert_eq!(plain, expected, "{stem}");
        // Four gradient comparisons, less the compared ones, and two at level 0.
        let mut dref: Vec<u32> = masks(a, OP_IMAGE_SAMPLE_DREF_EXPLICIT_LOD, 6)
            .iter()
            .map(lod)
            .collect();
        dref.sort();
        let mut expected = vec![LOD, LOD];
        expected.extend(vec![GRAD; 4 - compared]);
        assert_eq!(dref, expected, "{stem}");
    }) else {
        return;
    };
    assert!(compiled >= 4, "{compiled}");
}

#[test]
fn cube_depth_and_comparison_gathers() {
    let metal: Vec<String> = profiles()
        .into_iter()
        .filter(|(_, p)| {
            p.fields()
                .enabled
                .properties
                .contains(&VkProperty::MetalSampler)
        })
        .map(|(stem, _)| stem)
        .collect();
    let Some((compiled, _)) = each("unk_gather", "unk_gather", |stem, a| {
        // The 2D and 2D array comparison gathers the shader compares.
        let compared = if metal.iter().any(|m| m == stem) {
            0
        } else {
            2
        };
        let words = &a.module.words;
        // The component operand is a constant id: its value.
        let constant = |id: u32| {
            insts(words)
                .into_iter()
                .find(|(o, w)| *o == OP_CONSTANT && w.get(2) == Some(&id))
                .and_then(|(_, w)| w.get(3).copied())
                .unwrap_or_else(|| panic!("{stem}: component %{id} is no constant"))
        };
        let gathers: Vec<&[u32]> = insts(words)
            .into_iter()
            .filter(|(o, _)| *o == OP_IMAGE_GATHER)
            .map(|(_, w)| w)
            .collect();
        let mut components: Vec<u32> = gathers.iter().map(|w| constant(w[5])).collect();
        components.sort();
        // Cube x and w, cube array z, the two depth gathers' component 0, and the compared ones'.
        let mut expected = vec![0; 3 + compared];
        expected.extend([2, 3]);
        assert_eq!(components, expected, "{stem}");
        assert_eq!(count(a, OP_IMAGE_DREF_GATHER), 4 - compared, "{stem}");
        for (op, at) in [(OP_IMAGE_GATHER, 6), (OP_IMAGE_DREF_GATHER, 6)] {
            for m in masks(a, op, at) {
                assert_eq!(m & (CONST_OFFSET | OFFSET), 0, "{stem}: an offset on {op}");
            }
        }
    }) else {
        return;
    };
    assert!(compiled >= 4, "{compiled}");
}

#[test]
fn lod_queries_of_arrays_3d_and_cubes() {
    let Some((compiled, _)) = each("unk_calculate_lod", "unk_calculate_lod", |stem, a| {
        assert_eq!(count(a, OP_IMAGE_QUERY_LOD), 16, "{stem}");
    }) else {
        return;
    };
    assert!(compiled >= 4, "{compiled}");
}

fn uses(a: &FunctionArtifact, name: &str) -> u32 {
    a.strategies.entries.get(name).map_or(0, |u| u.uses)
}

/// The `GLSL.std.450` instructions numbered `number` (an `OpExtInst`'s word 4).
fn glsl(a: &FunctionArtifact, number: u32) -> usize {
    insts(&a.module.words)
        .into_iter()
        .filter(|(o, w)| *o == OP_EXT_INST && w.get(4) == Some(&number))
        .count()
}

/// Each of the fixture's six explicit cube gradient samples (color, depth, and comparison; cube and
/// cube array) takes Metal's footprint: its gradients are the counted components' length on one
/// tangent (a `GLSL.std.450` `Length` per gradient), its census name is `sampler.sample_cube_grad`,
/// and the module records `CubeGradientFootprint`. The 3D and depth 2D and 2D array gradient
/// samples pass their gradients unchanged (`sampler.sample_grad`), and a module whose gradients are
/// all 2D records no such note. The depth samples without comparison read through a sampler that
/// filters nearest with no mip filter, so on a device without `VkProperty::MetalSampler` their
/// directions are snapped onto Metal's texel instead (`sampler.cube.snapped`): the two gradient
/// samples and the two at level 0. The 2D fixture's gradient sample reads a runtime sampler, so on
/// a device without the property the linear emulation plans it (`sampler.linear`, its word
/// selecting the device's sample). On a driver that hands a compute pipeline's cube gradients to
/// Metal's `gradientcube` (`CubeGradientRoute::Metal`), the kernel hands Metal's own gradients,
/// exactly: no `Length`, and no note.
#[test]
fn explicit_cube_gradients_take_metals_footprint() {
    let handed = handed_metal();
    let Some((compiled, _)) = each("unk_sample_grad", "unk_sample_grad", |stem, a| {
        let snapped = match a
            .requirements
            .properties
            .contains(&VkProperty::MetalSampler)
        {
            true => 0,
            false => 2,
        };
        assert_eq!(uses(a, "sampler.sample_cube_grad"), 6 - snapped, "{stem}");
        assert_eq!(uses(a, "sampler.cube.snapped"), 2 * snapped, "{stem}");
        assert_eq!(uses(a, "sampler.sample_grad"), 5, "{stem}");
        let metal = handed.iter().any(|m| m == stem);
        assert_eq!(glsl(a, GLSL_LENGTH) >= 2, !metal, "{stem}");
        assert_eq!(
            a.precision.contains(&PrecisionNote::CubeGradientFootprint),
            !metal,
            "{stem}: {:?}",
            a.precision
        );
    }) else {
        return;
    };
    assert!(compiled >= 4, "{compiled}");
    assert!(!handed.is_empty(), "no profile hands Metal's gradients");
    let Some((compiled, _)) = each("a4_sample_grad", "a4_sample_grad", |stem, a| {
        assert_eq!(uses(a, "sampler.sample_cube_grad"), 0, "{stem}");
        let census = match a
            .requirements
            .properties
            .contains(&VkProperty::MetalSampler)
        {
            true => "sampler.sample_grad",
            false => "sampler.linear",
        };
        assert!(uses(a, census) > 0, "{stem}");
        assert_eq!(glsl(a, GLSL_LENGTH), 0, "{stem}");
        assert!(
            !a.precision.contains(&PrecisionNote::CubeGradientFootprint),
            "{stem}"
        );
    }) else {
        return;
    };
    assert!(compiled >= 4, "{compiled}");
}

/// An anisotropic constexpr sampler (`max_anisotropy(2)`, bits 20-23 of the state's first word) on
/// a cube's explicit gradients takes Metal's anisotropic footprint on every profile that compiles
/// the isotropic fixture and has `samplerAnisotropy`: each of the four samples is
/// `sampler.sample_cube_grad` with `CubeGradientFootprint` recorded, and its gradients keep Metal's
/// two face components (`emit::cube_footprint`) instead of the isotropic form's length on one
/// tangent, so the module has no `GLSL.std.450` `Length` where the isotropic one has one per
/// gradient. A profile without the feature refuses it with the typed device gap. On a driver that
/// hands a compute pipeline's cube gradients to Metal's `gradientcube`, the kernel hands Metal's
/// own through either sampler: no `Length` and no note.
#[test]
fn an_anisotropic_constexpr_sampler_takes_metals_cube_gradient_footprint() {
    let path = engine_dir().join("unk_cube_grad_lod.ll");
    let text = std::fs::read_to_string(&path).expect("fixture");
    let word: u64 = 34_901_797_601_019_977;
    let original = format!("[2 x i64] [i64 {word}, i64 0]");
    assert_eq!(
        text.matches(&original).count(),
        1,
        "the fixture's sampler state"
    );
    assert_eq!((word >> 20) & 0xf, 0, "the fixture's sampler is isotropic");
    let anisotropic = text.replace(
        &original,
        &format!("[2 x i64] [i64 {}, i64 0]", word | 1 << 20),
    );
    let (Some(iso), Some(aniso)) = (
        assemble_text("isotropic", &text),
        assemble_text("anisotropic", &anisotropic),
    ) else {
        return;
    };
    let mut compiled = 0;
    let handed = handed_metal();
    for (stem, p) in profiles() {
        let Some(i) = artifact(&iso, "unk_cube_grad_lod", &stem, &p) else {
            continue;
        };
        let metal = handed.contains(&stem);
        assert_eq!(glsl(&i, GLSL_LENGTH) >= 2, !metal, "{stem}");
        assert_eq!(
            i.precision.contains(&PrecisionNote::CubeGradientFootprint),
            !metal,
            "{stem}"
        );
        let Some(a) = artifact(&aniso, "unk_cube_grad_lod", &stem, &p) else {
            assert!(!p.fields().images.sampler_anisotropy, "{stem}");
            continue;
        };
        compiled += 1;
        assert_eq!(
            uses(&a, "sampler.sample_cube_grad"),
            uses(&i, "sampler.sample_cube_grad"),
            "{stem}"
        );
        assert!(uses(&a, "sampler.sample_cube_grad") > 0, "{stem}");
        assert_eq!(glsl(&a, GLSL_LENGTH), 0, "{stem}");
        assert_eq!(
            a.precision.contains(&PrecisionNote::CubeGradientFootprint),
            !metal,
            "{stem}: {:?}",
            a.precision
        );
    }
    assert!(compiled >= 4, "{compiled}");
}

/// An unmeasured route: `moltenvk-m3` without its measured compute cube-gradient route (as a
/// MoltenVK release no measurement covers) keeps the isotropic form for `unk_cube_grad_lod`'s
/// isotropic sampler (a `Length` per gradient, `CubeGradientFootprint`), and refuses the same
/// kernel through an anisotropic constexpr sampler with `Needs::UnmeasuredDeviation`, naming the
/// cube-gradient route; the measured description compiles both.
#[test]
fn an_unmeasured_cube_gradient_route_refuses_an_anisotropic_constexpr_sampler() {
    use crate::device::profile::CubeGradients;
    use crate::error::{DriverDeviation, Feature};
    let path = engine_dir().join("unk_cube_grad_lod.ll");
    let text = std::fs::read_to_string(&path).expect("fixture");
    let word: u64 = 34_901_797_601_019_977;
    let original = format!("[2 x i64] [i64 {word}, i64 0]");
    let anisotropic = text.replace(
        &original,
        &format!("[2 x i64] [i64 {}, i64 0]", word | 1 << 20),
    );
    assert_ne!(anisotropic, text, "the fixture's sampler state");
    let (Some(iso), Some(aniso)) = (
        assemble_text("isotropic", &text),
        assemble_text("anisotropic", &anisotropic),
    ) else {
        return;
    };
    let profile_text = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../metal2vulkan-validation/profiles/moltenvk-m3.json"),
    )
    .expect("profile");
    let measured = DeviceDescription::from_khronos_json(&profile_text, None).expect("description");
    let profile = |d: DeviceDescription, taken: bool| {
        let route = DriverDeviation::CubeGradientRoute;
        Device::with_deviation(d, EXACT_SAMPLING, route, taken)
            .expect("device")
            .profile()
            .clone()
    };
    let (measured, unmeasured) = (profile(measured.clone(), true), profile(measured, false));
    // The graphics route is its own deviation, still taken here.
    assert_eq!(
        unmeasured.fields().cube_gradients,
        CubeGradients {
            graphics: CubeGradientRoute::AppleSwizzled,
            compute: CubeGradientRoute::Unmeasured,
        }
    );
    let i = artifact(&iso, "unk_cube_grad_lod", "unmeasured", &unmeasured).expect("isotropic");
    assert!(glsl(&i, GLSL_LENGTH) >= 2);
    assert!(i.precision.contains(&PrecisionNote::CubeGradientFootprint));
    match compile(&aniso, "unk_cube_grad_lod", &unmeasured) {
        Err(Error::Unsupported(u)) => {
            assert_eq!(u.feature, Feature::SamplerAnisotropy);
            assert_eq!(
                u.needs,
                Needs::UnmeasuredDeviation(DriverDeviation::CubeGradientRoute)
            );
        }
        other => panic!("an unmeasured route: {:?}", other.map(|_| ())),
    }
    for bc in [&iso, &aniso] {
        let a = artifact(bc, "unk_cube_grad_lod", "measured", &measured).expect("measured");
        assert_eq!(glsl(&a, GLSL_LENGTH), 0);
    }
}

/// The graphics cube route is measured per GPU. On `moltenvk-m3` (the Apple M3 Max's `deviceID`) a
/// fragment function's explicit cube gradients through an anisotropic constexpr sampler take
/// SPIRV-Cross's fixup in the transposed form; on `moltenvk-m4` (the M4's), whose graphics route is
/// unmeasured (that form misses Metal's sample on 96 and 48 of 144 samples, EV RTL-lane2-mvk4), the
/// same fragment is refused naming `GraphicsCubeGradientRoute`, while the M4's kernel route stays
/// Metal's and the kernel of the same samples compiles. The cover vertex function compiles on both.
#[test]
fn the_graphics_cube_route_is_measured_on_the_m3_max_alone() {
    use crate::error::{DriverDeviation, Feature};
    let (Some(fragment), Some(kernel)) =
        (fixture("rtl_cube_grad_fragment"), fixture("ani_cube_grad"))
    else {
        return;
    };
    let all = committed_exact_profiles().expect("profiles");
    let of = |stem: &str| {
        all.iter()
            .find(|(s, _)| s == stem)
            .map(|(_, p)| p.clone())
            .expect(stem)
    };
    let (m3, m4) = (of("moltenvk-m3"), of("moltenvk-m4"));
    assert_eq!(
        (
            m3.fields().cube_gradients.graphics,
            m3.fields().cube_gradients.compute
        ),
        (CubeGradientRoute::AppleSwizzled, CubeGradientRoute::Metal)
    );
    assert_eq!(
        (
            m4.fields().cube_gradients.graphics,
            m4.fields().cube_gradients.compute
        ),
        (CubeGradientRoute::Unmeasured, CubeGradientRoute::Metal)
    );
    let a = artifact(&fragment, "rtl_cube_grad_fragment", "moltenvk-m3", &m3).expect("M3 Max");
    assert!(a.precision.contains(&PrecisionNote::CubeGradientFootprint));
    match compile(&fragment, "rtl_cube_grad_fragment", &m4) {
        Err(Error::Unsupported(u)) => {
            assert_eq!(u.feature, Feature::SamplerAnisotropy);
            assert_eq!(
                u.needs,
                Needs::UnmeasuredDeviation(DriverDeviation::GraphicsCubeGradientRoute)
            );
        }
        other => panic!("the M4's graphics route: {:?}", other.map(|_| ())),
    }
    for (stem, p) in [("moltenvk-m3", &m3), ("moltenvk-m4", &m4)] {
        artifact(&fragment, "rtl_cube_cover", stem, p).expect(stem);
        let k = artifact(&kernel, "ani_cube_grad", stem, p).expect(stem);
        assert!(
            !k.precision.contains(&PrecisionNote::CubeGradientFootprint),
            "{stem}"
        );
    }
}

/// A kernel sampling the cube at texture slot 0 with explicit gradients through the runtime
/// sampler at slot 0.
const RUNTIME_CUBE_GRAD: &str = r#"define void @runtime_cube_grad(ptr addrspace(1) %t, ptr addrspace(2) %s, ptr addrspace(1) %out, i32 %gid) {
  %v = call { <4 x float>, i8 } @air.sample_texture_cube_grad.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %s, <3 x float> <float 1.0, float 0.25, float -0.5>, <3 x float> <float 0.0, float 0x3F847AE140000000, float 0.0>, <3 x float> <float 0.0, float 0.0, float 0x3F847AE140000000>, float 0.0, i32 0)
  %x = extractvalue { <4 x float>, i8 } %v, 0
  %o = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i32 %gid
  store <4 x float> %x, ptr addrspace(1) %o, align 16
  ret void
}
declare { <4 x float>, i8 } @air.sample_texture_cube_grad.v4f32(ptr addrspace(1), ptr addrspace(2), <3 x float>, <3 x float>, <3 x float>, float, i32)
!air.kernel = !{!0}
!air.version = !{!11}
!air.language_version = !{!12}
!0 = !{ptr @runtime_cube_grad, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texturecube<float, sample>", !"air.arg_name", !"t"}
!4 = !{i32 1, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"s"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}
!6 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
!11 = !{i32 2, i32 8, i32 0}
!12 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

/// A cube's explicit gradients through a runtime sampler slot take their footprint by the slot's
/// word on every profile whose route is measured and not Metal's: the slot's `SamplerState`
/// specialization reads `SamplerWord::ANISOTROPIC`, the module computes the isotropic form (a
/// `Length` per gradient) beside the anisotropic one, records `CubeGradientFootprint`, and its
/// combined binding carries no cube-gradient bit, so the plan takes an anisotropic sampler there.
/// On a driver that hands a compute pipeline's cube gradients to Metal's `gradientcube`, the kernel
/// hands Metal's own: no word read, no `Length`, no note. On `moltenvk-m3` without its measured
/// route (unmeasured) the kernel keeps the isotropic form without reading the word, and its binding
/// carries the bit, which refuses an anisotropic sampler where it is bound
/// (`bind::plan::resources`' refusal test).
#[test]
fn a_runtime_samplers_cube_gradients_take_their_footprint_by_its_word() {
    use crate::metal::sampler::SamplerWord;
    use crate::pipeline::sym::{ResourceKind, SpecKey};
    let Some(bc) = assemble_text("runtime_cube_grad", RUNTIME_CUBE_GRAD) else {
        return;
    };
    let profile_text = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../metal2vulkan-validation/profiles/moltenvk-m3.json"),
    )
    .expect("profile");
    let unmeasured =
        DeviceDescription::from_khronos_json(&profile_text, None).expect("description");
    let route = crate::error::DriverDeviation::CubeGradientRoute;
    let unmeasured = Device::with_deviation(unmeasured, EXACT_SAMPLING, route, false)
        .expect("device")
        .profile()
        .clone();
    let mut all = profiles();
    all.push(("unmeasured".to_string(), unmeasured));
    let (mut by_word, mut handed) = (0, 0);
    for (stem, p) in all {
        let Some(a) = artifact(&bc, "runtime_cube_grad", &stem, &p) else {
            continue;
        };
        let route = p.fields().cube_gradients.compute;
        let fields = a.interface.read_keys().into_iter().find_map(|k| match k {
            SpecKey::SamplerState { fields, .. } => Some(fields),
            _ => None,
        });
        let reads = fields.is_some_and(|f| f & SamplerWord::ANISOTROPIC != 0);
        let bits: Vec<bool> = a
            .interface
            .descriptors
            .iter()
            .filter_map(|d| match d.slot.kind {
                ResourceKind::CombinedTextureSampler(c) => Some(c.cube_gradient),
                _ => None,
            })
            .collect();
        let noted = a.precision.contains(&PrecisionNote::CubeGradientFootprint);
        match route {
            CubeGradientRoute::Metal => {
                assert!(!reads, "{stem}: {fields:?}");
                assert_eq!(glsl(&a, GLSL_LENGTH), 0, "{stem}");
                assert!(!noted, "{stem}");
                assert_eq!(bits, vec![false], "{stem}");
                handed += 1;
            }
            CubeGradientRoute::Unmeasured => {
                assert!(!reads, "{stem}: {fields:?}");
                assert!(glsl(&a, GLSL_LENGTH) >= 2, "{stem}");
                assert!(noted, "{stem}");
                assert_eq!(bits, vec![true], "{stem}");
            }
            CubeGradientRoute::Vulkan | CubeGradientRoute::AppleSwizzled => {
                assert!(reads, "{stem}: {fields:?}");
                assert!(glsl(&a, GLSL_LENGTH) >= 2, "{stem}");
                assert!(noted, "{stem}");
                assert_eq!(bits, vec![false], "{stem}");
                by_word += 1;
            }
        }
    }
    assert!(by_word >= 4, "{by_word}");
    assert!(handed > 0, "no profile hands Metal's gradients");
}

/// The test profiles whose driver hands a compute pipeline's cube gradients to Metal's
/// `gradientcube` as given (`CubeGradientRoute::Metal`).
fn handed_metal() -> Vec<String> {
    profiles()
        .into_iter()
        .filter(|(_, p)| p.fields().cube_gradients.compute == CubeGradientRoute::Metal)
        .map(|(stem, _)| stem)
        .collect()
}
