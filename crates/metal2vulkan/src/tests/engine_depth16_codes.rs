//! A `Depth16Unorm` depth stored as Metal's code by the fragment stage
//! (`LoweringInputs::depth16_codes`, `lower::stage_io::depth16`), over the owned fixtures
//! `engine/rt6_render_state` (`rt6_full`, and `rt6_white`, which returns no depth) and
//! `engine/rfb_viewports` (`rfb_full`; `rfb_depth` returns a depth, `rfb_position` reads its
//! position's depth and returns none):
//!
//! - On a committed profile without `DriverDeviation::MetalDepth16Stores`, a single-sampled
//!   pipeline whose depth attachment is `Depth16Unorm` states `RuntimeFact::Depth16Codes`: biased
//!   where its fragment function returns no depth (it reads `PushField::DepthBias`), unbiased where
//!   it returns one; a pipeline without a fragment function gets a fragment stage of its own. A
//!   profile that takes the deviation, a pipeline of another depth format, a multisampled one and
//!   a function that reads its position's depth and returns none state nothing. Every module
//!   passes L3.
//! - The render pass's depth decides it, whatever the pipeline names: a draw against another
//!   depth than the pipeline's (`BindState::depth16_pass`) runs the variant of the pass's
//!   (`VariantKey::depth16_pass`), which states it (`RuntimeFact::Depth16Pass`) and plans itself,
//!   and pushes the draw's bias (`BindState::depth_bias`) where it writes the codes.
//! - A draw that biases the depth (`BindState::depth16_biased`) on a profile that takes the
//!   deviation but not `DriverDeviation::MetalDepth16BiasedStores` runs the variant whose fragment
//!   stage adds the bias (`VariantKey::depth16_biased`), which states it
//!   (`RuntimeFact::Depth16Biased`) and plans itself; a function that returns its depth or reads
//!   its position's, a multisampled pipeline and a profile of either other form run as compiled.
//! - A biased draw whose bias's side the consumer proved (`BindState::depth16_bias_raises`), where
//!   the fragment stage adds the bias itself (under that variant, or on a profile without the
//!   deviation), runs the variant whose stage declares the side (`VariantKey::depth16_bias_raises`,
//!   `RuntimeFact::Depth16BiasSide`): `DepthGreater` with `NMax` for a raising bias, `DepthLess`
//!   with `NMin` for a lowering one. A profile that stores biased depths as Metal's, and a function
//!   that returns its depth, run as compiled.
//!
//! What the device stores is the runtime lane's (`runtime_lane::unorm_depth_bias`, Metal probes).
//! `llvm-as` assembles the fixture; a missing tool skips visibly. No GPU is used.

use std::path::Path;

use crate::bind::{BindState, DrawCall};
use crate::budget::Budget;
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::error::{BindError, DriverDeviation};
use crate::library::Library;
use crate::metal::PixelFormat;
use crate::pipeline::artifact::RenderPipeline;
use crate::pipeline::descriptor::{
    ColorAttachmentDescriptor, FunctionDescriptor, RenderPipelineDescriptor, StageSource,
};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::interface::PushLoc;
use crate::pipeline::sym::{PushField, Stage};
use crate::pipeline::testing::{engine_dir, llvm_assemble, AssembleError};
use crate::spirv::testing::check_words_valid;
use crate::vk;

/// The library of `engine/<name>.ll`, or `None` (printed) where `llvm-as` is missing.
fn fixture(name: &str) -> Option<Library> {
    match llvm_assemble(&engine_dir().join(format!("{name}.ll"))) {
        Ok(b) => Some(Library::from_air(&b).expect("library")),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {name}.ll: {why}"),
    }
}

/// The two fixtures' libraries, or `None` (printed) where `llvm-as` is missing.
fn fixtures() -> Option<Fixtures> {
    Some(Fixtures {
        rt6: fixture("rt6_render_state")?,
        rfb: fixture("rfb_viewports")?,
    })
}

/// The libraries of `engine/rt6_render_state` and `engine/rfb_viewports`.
struct Fixtures {
    rt6: Library,
    rfb: Library,
}

/// A fragment stage of the test's pipelines.
#[derive(Clone, Copy, Debug)]
enum Fragment {
    /// `rt6_white` after `rt6_full`: returns no depth.
    White,
    /// `rfb_depth` after `rfb_full`: returns a depth.
    Depth,
    /// `rfb_position` after `rfb_full`: reads its position's depth and returns none.
    Position,
    /// `rt6_full` alone: no fragment function.
    Absent,
}

/// The device of the committed description `stem`, under the default policy.
fn device(stem: &str) -> Device {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    Device::new(desc, DevicePolicy::default()).expect("device")
}

/// The profiles this test compiles on: two that take the deviation, two that do not.
const STEMS: [&str; 4] = [
    "amd-780m-radv",
    "moltenvk-m3",
    "nvidia-rtx50-linux",
    "lavapipe",
];

/// A pipeline of `fragment` and its vertex function with one `RGBA32Float` color, a depth
/// attachment of `depth` and `samples` raster samples.
fn descriptor(
    libs: &Fixtures,
    fragment: Fragment,
    depth: PixelFormat,
    samples: u32,
) -> RenderPipelineDescriptor {
    let function = |lib: &Library, name| {
        StageSource::Function(FunctionDescriptor::new(lib.function(name).expect(name)))
    };
    let (vertex, fragment) = match fragment {
        Fragment::White => (
            function(&libs.rt6, "rt6_full"),
            Some((&libs.rt6, "rt6_white")),
        ),
        Fragment::Depth => (
            function(&libs.rfb, "rfb_full"),
            Some((&libs.rfb, "rfb_depth")),
        ),
        Fragment::Position => (
            function(&libs.rfb, "rfb_full"),
            Some((&libs.rfb, "rfb_position")),
        ),
        Fragment::Absent => (function(&libs.rt6, "rt6_full"), None),
    };
    let mut d = RenderPipelineDescriptor::new(vertex);
    d.fragment = fragment.map(|(lib, name)| function(lib, name));
    d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA32Float));
    d.depth_format = Some(depth);
    d.raster_sample_count = samples;
    d
}

/// Every stage of `p` passes L3 on `dev`.
fn validate(p: &RenderPipeline, dev: &Device, what: &str) {
    for m in &p.stages {
        check_words_valid(&m.spirv, &p.requirements, dev.profile())
            .unwrap_or_else(|e| panic!("{what} {:?}: L3: {e}", m.stage));
    }
}

/// How `p` stores a `Depth16Unorm` depth: its `RuntimeFact::Depth16Codes`, `Some(biased)`.
fn codes(p: &RenderPipeline) -> Option<bool> {
    p.dependencies.iter().find_map(|d| match d.fact {
        RuntimeFact::Depth16Codes { biased } => Some(biased),
        _ => None,
    })
}

/// The push offset of `p`'s fragment `PushField::DepthBias`, `None` without it.
fn bias_offset(p: &RenderPipeline, what: &str) -> Option<usize> {
    p.interface
        .push
        .iter()
        .filter(|pl| pl.stage == Stage::Fragment)
        .flat_map(|pl| pl.fields.iter())
        .find(|f| f.field == PushField::DepthBias)
        .map(|f| match f.loc {
            PushLoc::Push { offset } => {
                assert_eq!(f.size, 16, "{what}");
                offset as usize
            }
            other => panic!("{what}: {other:?}"),
        })
}

#[test]
fn a_depth16_pipeline_s_fragment_stage_writes_metal_s_codes_where_the_device_does_not() {
    let Some(libs) = fixtures() else {
        return;
    };
    let budget = Budget::standard();
    let d16 = PixelFormat::Depth16Unorm;
    let mut forms = [0usize; 2];
    for stem in STEMS {
        let dev = device(stem);
        let stores = dev
            .profile()
            .takes_driver_deviation(DriverDeviation::MetalDepth16Stores);
        forms[usize::from(stores)] += 1;
        let cases = [
            (Fragment::White, d16, 1, Some(true)),
            (Fragment::Depth, d16, 1, Some(false)),
            (Fragment::Absent, d16, 1, Some(true)),
            (Fragment::Position, d16, 1, None),
            (Fragment::White, PixelFormat::Depth32Float, 1, None),
            (Fragment::White, d16, 4, None),
            (Fragment::Absent, d16, 4, None),
        ];
        for (fragment, depth, samples, shader) in cases {
            let what = format!("{stem} {fragment:?} {depth:?} x{samples}");
            let d = descriptor(&libs, fragment, depth, samples);
            let p = dev
                .compile_render_pipeline(&d, &budget)
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            validate(&p, &dev, &what);
            let want = shader.filter(|_| !stores);
            assert_eq!(codes(&p), want, "{what}");
            assert_eq!(
                bias_offset(&p, &what).is_some(),
                want == Some(true),
                "{what}"
            );
            let fragments = p
                .stages
                .iter()
                .filter(|m| m.stage == vk::ShaderStage::FRAGMENT)
                .count();
            let fragmentless = matches!(fragment, Fragment::Absent);
            let synthesized = fragmentless && want.is_some();
            assert_eq!(
                fragments,
                usize::from(!fragmentless || synthesized),
                "{what}"
            );
        }
    }
    assert_eq!(forms, [2, 2], "the profiles of each form");
}

/// A draw against another depth than its pipeline names runs the pass's variant: a pipeline
/// naming `Depth16Unorm` drawn against a `float` depth writes no codes, one naming `Depth32Float`
/// (or none, with no fragment function) drawn against a `Depth16Unorm` depth writes them and
/// pushes the draw's bias; each variant states its pass and plans itself, and a draw against the
/// named depth, or against none, runs the pipeline as compiled.
#[test]
fn a_draw_against_another_depth_runs_the_variant_of_the_pass_s_depth() {
    let Some(libs) = fixtures() else {
        return;
    };
    let budget = Budget::standard();
    let triangle = DrawCall::Primitives {
        vertex_start: 0,
        vertex_count: 3,
        instance_count: 1,
        base_instance: 0,
    };
    let bias = [3.0f32 / 65535.0, 0.5, -0.25];
    let against = |pass: Option<bool>| BindState {
        depth16_pass: pass,
        depth_bias: bias,
        ..Default::default()
    };
    for stem in ["lavapipe", "nvidia-rtx50-linux"] {
        let dev = device(stem);
        assert!(
            !dev.profile()
                .takes_driver_deviation(DriverDeviation::MetalDepth16Stores),
            "{stem}"
        );
        let cases = [
            (Fragment::White, PixelFormat::Depth16Unorm, false),
            (Fragment::White, PixelFormat::Depth32Float, true),
            (Fragment::Absent, PixelFormat::Depth32Float, true),
        ];
        for (fragment, named, pass) in cases {
            let what = format!("{stem} {fragment:?} {named:?} against {pass}");
            let d = descriptor(&libs, fragment, named, 1);
            let base = dev
                .compile_render_pipeline(&d, &budget)
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            // Against the named depth, or none, as compiled.
            for same in [Some(!pass), None] {
                let state = against(same);
                let plan = base
                    .plan(&dev, &state.shape())
                    .unwrap_or_else(|e| panic!("{what}: {e:?}"));
                assert_eq!(plan.variant.depth16_pass, None, "{what} {same:?}");
                plan.instantiate_draw(&state, &triangle)
                    .unwrap_or_else(|e| panic!("{what} {same:?}: {e:?}"));
            }
            let state = against(Some(pass));
            let plan = base
                .plan(&dev, &state.shape())
                .unwrap_or_else(|e| panic!("{what}: {e:?}"));
            assert_eq!(plan.variant.depth16_pass, Some(pass), "{what}");
            assert_eq!(
                plan.instantiate_draw(&state, &triangle),
                Err(BindError::NeedsVariant {
                    variant: Box::new(plan.variant.clone())
                }),
                "{what}"
            );
            let key = plan.variant.clone();
            let v = dev
                .compile_render_variant(&d, &key, &budget)
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            validate(&v, &dev, &what);
            assert_ne!(v.key, base.key, "{what}");
            assert!(
                v.dependencies
                    .iter()
                    .any(|x| x.fact == RuntimeFact::Depth16Pass { depth16: pass }),
                "{what}"
            );
            assert_eq!(codes(&v), pass.then_some(true), "{what}");
            let plan = v
                .plan(&dev, &state.shape())
                .unwrap_or_else(|e| panic!("{what}: {e:?}"));
            assert_eq!(plan.variant, key, "{what}");
            let commands = plan
                .instantiate_draw(&state, &triangle)
                .unwrap_or_else(|e| panic!("{what}: {e:?}"));
            let Some(offset) = bias_offset(&v, &what) else {
                assert!(!pass, "{what}: no bias field");
                continue;
            };
            let mut pushed = vec![None; 256];
            for w in &commands.push_constants {
                for (i, b) in w.bytes.iter().enumerate() {
                    if let Some(slot) = pushed.get_mut(w.offset as usize + i) {
                        *slot = Some(*b);
                    }
                }
            }
            let bytes: Vec<Option<u8>> = bias
                .iter()
                .chain(&[0.0])
                .flat_map(|v| v.to_le_bytes())
                .map(Some)
                .collect();
            assert_eq!(pushed.get(offset..offset + 16), Some(&bytes[..]), "{what}");
        }
    }
}

/// A draw biasing a `Depth16Unorm` depth on the Iris Xe RPL-P's profile, which takes
/// `DriverDeviation::MetalDepth16Stores` but not `DriverDeviation::MetalDepth16BiasedStores`,
/// runs the variant whose fragment stage adds the bias: it states `RuntimeFact::Depth16Biased`,
/// writes the codes biased, pushes the draw's bias, and plans itself, where an unbiased draw runs
/// the pipeline as compiled. A function that returns its depth (which Metal does not bias) or
/// reads its position's, a multisampled pipeline, and the profiles that take both deviations (the
/// 780M's) or neither (the RTX 50's) run a biased draw as compiled.
#[test]
fn a_biased_draw_runs_the_biased_variant_where_the_device_stores_only_unbiased_codes() {
    let Some(libs) = fixtures() else {
        return;
    };
    let budget = Budget::standard();
    let d16 = PixelFormat::Depth16Unorm;
    let triangle = DrawCall::Primitives {
        vertex_start: 0,
        vertex_count: 3,
        instance_count: 1,
        base_instance: 0,
    };
    let bias = [3.0f32 / 65535.0, 0.5, -0.25];
    let drawn = |biased: bool| BindState {
        depth16_pass: Some(true),
        depth16_biased: biased,
        depth_bias: bias,
        ..Default::default()
    };
    let stem = "intel-irisxe-rplp-anv";
    let dev = device(stem);
    let profile = dev.profile();
    assert!(
        profile.takes_driver_deviation(DriverDeviation::MetalDepth16Stores)
            && !profile.takes_driver_deviation(DriverDeviation::MetalDepth16BiasedStores),
        "{stem}"
    );
    for fragment in [Fragment::White, Fragment::Absent] {
        let what = format!("{stem} {fragment:?}");
        let d = descriptor(&libs, fragment, d16, 1);
        let base = dev
            .compile_render_pipeline(&d, &budget)
            .unwrap_or_else(|e| panic!("{what}: {e}"));
        assert_eq!(codes(&base), None, "{what}");
        let unbiased = drawn(false);
        let plan = base
            .plan(&dev, &unbiased.shape())
            .unwrap_or_else(|e| panic!("{what}: {e:?}"));
        assert!(!plan.variant.depth16_biased, "{what} unbiased");
        plan.instantiate_draw(&unbiased, &triangle)
            .unwrap_or_else(|e| panic!("{what} unbiased: {e:?}"));
        let state = drawn(true);
        let plan = base
            .plan(&dev, &state.shape())
            .unwrap_or_else(|e| panic!("{what}: {e:?}"));
        assert!(plan.variant.depth16_biased, "{what}");
        assert_eq!(plan.variant.depth16_pass, None, "{what}");
        assert_eq!(
            plan.instantiate_draw(&state, &triangle),
            Err(BindError::NeedsVariant {
                variant: Box::new(plan.variant.clone())
            }),
            "{what}"
        );
        let key = plan.variant.clone();
        let v = dev
            .compile_render_variant(&d, &key, &budget)
            .unwrap_or_else(|e| panic!("{what}: {e}"));
        validate(&v, &dev, &what);
        assert_ne!(v.key, base.key, "{what}");
        assert!(
            v.dependencies
                .iter()
                .any(|x| x.fact == RuntimeFact::Depth16Biased),
            "{what}"
        );
        assert_eq!(codes(&v), Some(true), "{what}");
        assert!(bias_offset(&v, &what).is_some(), "{what}");
        let plan = v
            .plan(&dev, &state.shape())
            .unwrap_or_else(|e| panic!("{what}: {e:?}"));
        assert_eq!(plan.variant, key, "{what}");
        plan.instantiate_draw(&state, &triangle)
            .unwrap_or_else(|e| panic!("{what}: {e:?}"));
    }
    let compiled = [
        (stem, Fragment::Depth, 1),
        (stem, Fragment::Position, 1),
        (stem, Fragment::White, 4),
        ("amd-780m-radv", Fragment::White, 1),
        ("nvidia-rtx50-linux", Fragment::White, 1),
    ];
    for (stem, fragment, samples) in compiled {
        let what = format!("{stem} {fragment:?} x{samples}");
        let dev = device(stem);
        let d = descriptor(&libs, fragment, d16, samples);
        let base = dev
            .compile_render_pipeline(&d, &budget)
            .unwrap_or_else(|e| panic!("{what}: {e}"));
        let plan = base
            .plan(&dev, &drawn(true).shape())
            .unwrap_or_else(|e| panic!("{what}: {e:?}"));
        assert!(!plan.variant.depth16_biased, "{what}");
    }
}

/// The execution modes the module `words` declares, and the `GLSL.std.450` instructions it uses.
fn modes_and_glsl(words: &[u32]) -> (Vec<u32>, Vec<u32>) {
    const HEADER_WORDS: usize = 5;
    let (mut modes, mut glsl) = (Vec::new(), Vec::new());
    let mut at = HEADER_WORDS;
    while at < words.len() {
        let length = (words[at] >> 16) as usize;
        assert!(length > 0, "an instruction of zero words at {at}");
        match words[at] & 0xffff {
            op if op == crate::spv::Op::ExecutionMode as u32 => modes.push(words[at + 2]),
            op if op == crate::spv::Op::ExtInst as u32 => glsl.push(words[at + 4]),
            _ => {}
        }
        at += length;
    }
    (modes, glsl)
}

/// A biased draw of proven side runs the variant whose fragment stage declares it, where the stage
/// adds the bias itself: the Iris Xe RPL-P under `VariantKey::depth16_biased`, and the RTX 50
/// (no `DriverDeviation::MetalDepth16Stores`) for every draw. Each side's variant states
/// `RuntimeFact::Depth16BiasSide`, declares `DepthGreater` (raising) or `DepthLess` (lowering)
/// beside `DepthReplacing`, bounds the depth with `NMax` or `NMin`, passes L3 and plans itself; a
/// draw of unknown side keeps the stage that declares neither. The 780M, which stores biased
/// depths as Metal's, and a function that returns its depth do not plan it.
#[test]
fn a_biased_draw_of_proven_side_declares_it() {
    use crate::spv::{ExecutionMode as M, GlslStd450Op as G};
    let Some(libs) = fixtures() else {
        return;
    };
    let budget = Budget::standard();
    let d16 = PixelFormat::Depth16Unorm;
    let triangle = DrawCall::Primitives {
        vertex_start: 0,
        vertex_count: 3,
        instance_count: 1,
        base_instance: 0,
    };
    let drawn = |raises: Option<bool>| BindState {
        depth16_pass: Some(true),
        depth16_biased: true,
        depth16_bias_raises: raises,
        depth_bias: crate::metal::state::depth16_bias(2.0, 0.0, 0.0),
        ..Default::default()
    };
    for stem in ["intel-irisxe-rplp-anv", "nvidia-rtx50-linux"] {
        let dev = device(stem);
        for fragment in [Fragment::White, Fragment::Absent] {
            let what = format!("{stem} {fragment:?}");
            let d = descriptor(&libs, fragment, d16, 1);
            let base = dev
                .compile_render_pipeline(&d, &budget)
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            let plan = base
                .plan(&dev, &drawn(None).shape())
                .unwrap_or_else(|e| panic!("{what}: {e:?}"));
            assert_eq!(
                plan.variant.depth16_bias_raises, None,
                "{what} unknown side"
            );
            let unsided = dev
                .compile_render_variant(&d, &plan.variant, &budget)
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            for raises in [true, false] {
                let what = format!("{what} raises {raises}");
                let state = drawn(Some(raises));
                let plan = base
                    .plan(&dev, &state.shape())
                    .unwrap_or_else(|e| panic!("{what}: {e:?}"));
                assert_eq!(plan.variant.depth16_bias_raises, Some(raises), "{what}");
                let key = plan.variant.clone();
                let v = dev
                    .compile_render_variant(&d, &key, &budget)
                    .unwrap_or_else(|e| panic!("{what}: {e}"));
                validate(&v, &dev, &what);
                assert_ne!(v.key, unsided.key, "{what}");
                assert!(
                    v.dependencies
                        .iter()
                        .any(|x| x.fact == RuntimeFact::Depth16BiasSide { raises }),
                    "{what}"
                );
                assert_eq!(codes(&v), Some(true), "{what}");
                let fragment_words = |p: &RenderPipeline| {
                    let m = p
                        .stages
                        .iter()
                        .find(|m| m.stage == vk::ShaderStage::FRAGMENT)
                        .unwrap_or_else(|| panic!("{what}: no fragment stage"));
                    modes_and_glsl(&m.spirv)
                };
                let (mode, op, other_mode) = match raises {
                    true => (M::DepthGreater, G::NMax, M::DepthLess),
                    false => (M::DepthLess, G::NMin, M::DepthGreater),
                };
                let (modes, glsl) = fragment_words(&v);
                assert!(modes.contains(&(M::DepthReplacing as u32)), "{what}");
                assert!(modes.contains(&(mode as u32)), "{what}: {modes:?}");
                assert!(!modes.contains(&(other_mode as u32)), "{what}: {modes:?}");
                assert!(glsl.contains(&(op as u32)), "{what}: {glsl:?}");
                let (unsided_modes, unsided_glsl) = fragment_words(&unsided);
                for m in [M::DepthGreater, M::DepthLess] {
                    assert!(!unsided_modes.contains(&(m as u32)), "{what} unsided");
                }
                let bounds = |g: &[u32]| g.iter().filter(|&&x| x == op as u32).count();
                assert_eq!(bounds(&glsl), bounds(&unsided_glsl) + 1, "{what}");
                let plan = v
                    .plan(&dev, &state.shape())
                    .unwrap_or_else(|e| panic!("{what}: {e:?}"));
                assert_eq!(plan.variant, key, "{what}");
                plan.instantiate_draw(&state, &triangle)
                    .unwrap_or_else(|e| panic!("{what}: {e:?}"));
            }
        }
    }
    let compiled = [
        ("intel-irisxe-rplp-anv", Fragment::Depth),
        ("amd-780m-radv", Fragment::White),
    ];
    for (stem, fragment) in compiled {
        let what = format!("{stem} {fragment:?}");
        let dev = device(stem);
        let d = descriptor(&libs, fragment, d16, 1);
        let base = dev
            .compile_render_pipeline(&d, &budget)
            .unwrap_or_else(|e| panic!("{what}: {e}"));
        let plan = base
            .plan(&dev, &drawn(Some(true)).shape())
            .unwrap_or_else(|e| panic!("{what}: {e:?}"));
        assert_eq!(plan.variant.depth16_bias_raises, None, "{what}");
    }
}
