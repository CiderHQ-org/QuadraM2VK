//! Metal's private fixed-function render pipeline state (`PrivateRenderState`) over the owned
//! fixtures `engine/rt6_render_state` (Metal's fullscreen triangle and a white fragment
//! function) and `engine/a27_a4_atomic_load_only` (a kernel), on the `lavapipe` and
//! `moltenvk-m3` profiles at one and four samples:
//!
//! - Metal's defaults are the ones measured, and the properties that change nothing Metal draws
//!   keep the pipeline's key, modules and graphics state at any value.
//! - The sample mask is `GraphicsState::sample_mask`, its bits past the sample count dropped; it
//!   changes the key, never the modules, and only where it drops a sample; a pipeline without a
//!   fragment function ignores it, as Metal's does. A mask that drops one
//!   is refused where the fragment function could tell Metal's mask, applied after it runs, from
//!   Vulkan's (the owned fixtures `engine/sample_rate`, reading `[[sample_mask]]`, and
//!   `engine/multisample`, writing memory).
//! - Each property not built refuses a value other than Metal's default `NotImplemented`, naming
//!   it. A compute pipeline's `needs_custom_border_color_samplers`, which changes nothing, keeps
//!   its key.
//!
//! `llvm-as` assembles the fixtures; a missing tool skips visibly. No GPU is used.

use std::path::Path;

use crate::budget::Budget;
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::error::{Error, Feature, PrivateRenderField, RuntimeInput, Unsupported};
use crate::library::Library;
use crate::metal::PixelFormat;
use crate::pipeline::descriptor::{
    ColorAttachmentDescriptor, ComputePipelineDescriptor, FunctionDescriptor, PrivateRenderState,
    RenderPipelineDescriptor, StageSource,
};
use crate::pipeline::testing::{engine_dir, llvm_assemble, AssembleError};
use crate::pipeline::RenderPipeline;
use crate::spirv::testing::check_words_valid;

const STEMS: [&str; 2] = ["lavapipe", "moltenvk-m3"];

fn fixture(stem: &str) -> Option<Library> {
    match llvm_assemble(&engine_dir().join(format!("{stem}.ll"))) {
        Ok(b) => Some(Library::from_air(&b).expect("library")),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {stem}.ll: {why}"),
    }
}

fn device(stem: &str) -> Device {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    let description = DeviceDescription::from_khronos_json(&text, None).expect("description");
    Device::new(description, DevicePolicy::default()).expect("device")
}

/// The fullscreen triangle drawn white into one `RGBA8Unorm` attachment at `samples`.
fn render(lib: &Library, samples: u32) -> RenderPipelineDescriptor {
    let stage = |name: &str| {
        StageSource::Function(FunctionDescriptor::new(
            lib.function(name).expect("function"),
        ))
    };
    let mut d = RenderPipelineDescriptor::new(stage("rt6_full"));
    d.fragment = Some(stage("rt6_white"));
    d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    d.raster_sample_count = samples;
    d
}

fn compile(dev: &Device, d: &RenderPipelineDescriptor, what: &str) -> RenderPipeline {
    dev.compile_render_pipeline(d, &Budget::standard())
        .unwrap_or_else(|e| panic!("{what}: {e}"))
}

fn same_modules(a: &RenderPipeline, b: &RenderPipeline) -> bool {
    a.stages.len() == b.stages.len()
        && a.stages
            .iter()
            .zip(&b.stages)
            .all(|(x, y)| x.stage == y.stage && x.spirv == y.spirv)
}

fn refused(field: PrivateRenderField) -> Error {
    Unsupported::not_implemented(Feature::PrivateRenderState(field)).into()
}

#[test]
fn defaults_and_properties_that_change_nothing_keep_the_pipeline() {
    // Metal's defaults, as an M4 answers a new descriptor's.
    let s = PrivateRenderState::new();
    assert_eq!(s.sample_mask, u64::MAX);
    assert_eq!(s.sample_coverage, 1.0);
    assert_eq!(s.clip_distance_enable_mask, 0xff);
    assert_eq!(
        (s.alpha_test_function, s.logic_operation),
        (0, 0),
        "raw enum values"
    );
    assert_eq!(
        (
            s.vertex_depth_compare_clamp_mask,
            s.fragment_depth_compare_clamp_mask
        ),
        (0, 0)
    );
    assert!(![
        s.sample_coverage_invert,
        s.alpha_test_enabled,
        s.logic_operation_enabled,
        s.point_smooth_enabled,
        s.point_coord_lower_left,
        s.point_size_output_vs,
        s.two_side_enabled,
        s.depth_stencil_write_disabled,
        s.needs_custom_border_color_samplers,
    ]
    .contains(&true));
    assert_eq!(PrivateRenderState::default(), s);
    let Some(lib) = fixture("rt6_render_state") else {
        return;
    };
    for stem in STEMS {
        let dev = device(stem);
        for samples in [1, 4] {
            let what = format!("{stem} {samples}x");
            let base = compile(&dev, &render(&lib, samples), &what);
            assert_eq!(base.graphics.sample_mask, None, "{what}");
            let mut d = render(&lib, samples);
            let p = &mut d.private;
            p.sample_coverage_invert = true;
            p.alpha_test_function = 3;
            p.clip_distance_enable_mask = 0;
            p.point_coord_lower_left = true;
            p.point_size_output_vs = true;
            p.two_side_enabled = true;
            p.vertex_depth_compare_clamp_mask = u32::MAX;
            p.depth_stencil_write_disabled = true;
            p.needs_custom_border_color_samplers = true;
            // Every sample's bit set, and bits past the sample count cleared or set.
            p.sample_mask = ((1 << samples) - 1) | 0xf0_0000_0000;
            let kept = compile(&dev, &d, &what);
            assert_eq!(kept.key, base.key, "{what}");
            assert_eq!(kept.graphics, base.graphics, "{what}");
            assert!(same_modules(&kept, &base), "{what}");
        }
    }
}

#[test]
fn the_sample_mask_is_its_bits_within_the_sample_count() {
    let Some(lib) = fixture("rt6_render_state") else {
        return;
    };
    let cases: [(u32, u64, Option<u32>); 6] = [
        (4, 0x5, Some(0x5)),
        (4, 0, Some(0)),
        (4, 0xffff_fff5, Some(0x5)),
        (4, 0x1_0000_000f, None),
        (1, 0x2, Some(0)),
        (1, 0x3, None),
    ];
    for stem in STEMS {
        let dev = device(stem);
        for (samples, mask, effective) in cases {
            let what = format!("{stem} {samples}x {mask:#x}");
            let base = compile(&dev, &render(&lib, samples), &what);
            let mut d = render(&lib, samples);
            d.private.sample_mask = mask;
            let masked = compile(&dev, &d, &what);
            assert_eq!(masked.graphics.sample_mask, effective, "{what}");
            assert_eq!(masked.key == base.key, effective.is_none(), "{what}");
            assert!(same_modules(&masked, &base), "{what}");
            let mut graphics = masked.graphics.clone();
            graphics.sample_mask = None;
            assert_eq!(graphics, base.graphics, "{what}");
        }
        // Without a fragment function Metal ignores the mask.
        for samples in [1, 4] {
            let what = format!("{stem} {samples}x without a fragment function");
            let mut d = render(&lib, samples);
            d.fragment = None;
            d.depth_format = Some(PixelFormat::Depth32Float);
            let base = compile(&dev, &d, &what);
            d.private.sample_mask = 0;
            let masked = compile(&dev, &d, &what);
            assert_eq!(masked.graphics.sample_mask, None, "{what}");
            assert_eq!(masked.key, base.key, "{what}");
        }
    }
}

/// The storage classes of the variables `words` decorates `BuiltIn SampleMask` (1 `Input`, 3
/// `Output`), sorted.
fn sample_mask_variables(words: &[u32]) -> Vec<u32> {
    const DECORATE: u32 = 71;
    const VARIABLE: u32 = 59;
    const BUILT_IN: u32 = 11;
    const SAMPLE_MASK: u32 = 20;
    let (mut ids, mut classes) = (Vec::new(), Vec::new());
    let mut i = 5;
    while i < words.len() {
        let (op, n) = (words[i] & 0xffff, (words[i] >> 16) as usize);
        let w = &words[i..i + n.max(1)];
        if op == DECORATE && n >= 4 && w[2] == BUILT_IN && w[3] == SAMPLE_MASK {
            ids.push(w[1]);
        }
        if op == VARIABLE && n >= 4 && ids.contains(&w[2]) {
            classes.push(w[3]);
        }
        i += n.max(1);
    }
    classes.sort();
    classes
}

#[test]
fn a_function_that_could_tell_applies_the_mask_itself() {
    let (Some(lib), Some(rate), Some(multi), Some(early)) = (
        fixture("rt6_render_state"),
        fixture("sample_rate"),
        fixture("multisample"),
        fixture("cfb_depth_never"),
    ) else {
        return;
    };
    let stage = |lib: &Library, name: &str| {
        StageSource::Function(FunctionDescriptor::new(
            lib.function(name).expect("function"),
        ))
    };
    // Reads `[[sample_mask]]`: Metal's is the rasterized coverage, unmasked.
    let mut reads = RenderPipelineDescriptor::new(stage(&lib, "rt6_full"));
    reads.fragment = Some(stage(&rate, "sample_rate_coverage"));
    reads.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA32Uint));
    // Writes memory: Metal runs it for the masked samples too.
    let mut writes = RenderPipelineDescriptor::new(stage(&multi, "multisample_vertex"));
    writes.fragment = Some(stage(&multi, "multisample_counter"));
    writes.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA16Float));
    // Early fragment tests: Metal's mask keeps every sample's depth and stencil.
    let mut tests = RenderPipelineDescriptor::new(stage(&lib, "rt6_full"));
    tests.fragment = Some(stage(&early, "cfb_dn_red_early"));
    tests.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    tests.depth_format = Some(PixelFormat::Depth32Float);
    // (name, the function's own SampleMask variables).
    let cases = [
        ("reads", &reads, vec![1]),
        ("writes", &writes, vec![]),
        ("early tests", &tests, vec![]),
    ];
    for stem in STEMS {
        let dev = device(stem);
        for (name, d, own) in &cases {
            let what = format!("{stem} {name}");
            let base = compile(&dev, d, &what);
            assert_eq!(base.graphics.sample_mask, None, "{what}");
            // A mask keeping the pipeline's one sample changes nothing.
            let mut kept = (*d).clone();
            kept.private.sample_mask = 0x1;
            let kept = compile(&dev, &kept, &what);
            assert_eq!(kept.key, base.key, "{what}");
            assert!(same_modules(&kept, &base), "{what}");
            let mut masked = (*d).clone();
            masked.private.sample_mask = 0;
            let p = compile(&dev, &masked, &what);
            // No Vulkan mask: the fragment function writes its `SampleMask` output.
            assert_eq!(p.graphics.sample_mask, None, "{what}");
            let fragment = p
                .stages
                .iter()
                .find(|m| m.stage == crate::vk::ShaderStage::FRAGMENT)
                .expect("fragment");
            let mut want = own.clone();
            want.push(3);
            assert_eq!(sample_mask_variables(&fragment.spirv), want, "{what}");
            for m in &p.stages {
                check_words_valid(&m.spirv, &p.requirements, dev.profile())
                    .unwrap_or_else(|e| panic!("{what} {:?}: L3: {e}", m.stage));
            }
            // The keys a consumer looks the pipeline up by name the masked compile.
            let keys = dev
                .render_pipeline_key(&masked, &Budget::standard())
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            assert_eq!(keys.pipeline, p.key, "{what}");
            let inputs = keys.fragment_inputs.expect("fragment inputs");
            assert_eq!(inputs.sample_mask, Some(0), "{what}");
            // The function compiled without the mask, under the inputs the unmasked pipeline
            // gives it, is compiled for the pipeline again.
            let unmasked = dev
                .render_pipeline_key(d, &Budget::standard())
                .unwrap_or_else(|e| panic!("{what}: {e}"))
                .fragment_inputs
                .expect("fragment inputs");
            let Some(StageSource::Function(f)) = &d.fragment else {
                unreachable!("a function fragment stage")
            };
            let mut compiled = masked.clone();
            compiled.fragment = Some(StageSource::Compiled(
                dev.compile_function(f, &unmasked, &Budget::standard())
                    .unwrap_or_else(|e| panic!("{what}: {e}")),
            ));
            match dev.compile_render_pipeline(&compiled, &Budget::standard()) {
                Err(e) => assert_eq!(e, Error::NeedsInput(RuntimeInput::SampleMask), "{what}"),
                Ok(_) => panic!("{what}: compiled"),
            }
        }
    }
}

#[test]
fn each_property_not_built_refuses_a_value_other_than_metals_default() {
    use PrivateRenderField as F;
    let Some(lib) = fixture("rt6_render_state") else {
        return;
    };
    let cases: [(fn(&mut PrivateRenderState), PrivateRenderField); 6] = [
        (|p| p.sample_coverage = 0.5, F::SampleCoverage),
        (|p| p.sample_coverage = f32::NAN, F::SampleCoverage),
        (|p| p.alpha_test_enabled = true, F::AlphaTest),
        (|p| p.logic_operation_enabled = true, F::LogicOperation),
        (|p| p.point_smooth_enabled = true, F::PointSmooth),
        (
            |p| p.fragment_depth_compare_clamp_mask = 1,
            F::FragmentDepthCompareClamp,
        ),
    ];
    for stem in STEMS {
        let dev = device(stem);
        for (set, field) in cases {
            let mut d = render(&lib, 4);
            set(&mut d.private);
            match dev.compile_render_pipeline(&d, &Budget::standard()) {
                Err(e) => assert_eq!(e, refused(field), "{stem} {field:?}"),
                Ok(_) => panic!("{stem} {field:?}: compiled"),
            }
        }
    }
}

#[test]
fn the_compute_custom_border_color_flag_keeps_the_pipeline() {
    let Some(kernels) = fixture("a27_a4_atomic_load_only") else {
        return;
    };
    let f = kernels.function("a4_atomic_load_only").expect("kernel");
    let mut d = ComputePipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(f)));
    for stem in STEMS {
        let dev = device(stem);
        d.needs_custom_border_color_samplers = false;
        let base = dev
            .compile_compute_pipeline(&d, &Budget::standard())
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        d.needs_custom_border_color_samplers = true;
        let kept = dev
            .compile_compute_pipeline(&d, &Budget::standard())
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert_eq!(kept.key, base.key, "{stem}");
    }
}
