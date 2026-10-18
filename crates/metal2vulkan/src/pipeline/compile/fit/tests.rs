//! Tests of the compiler-word-bound fit, one per pipeline kind (render, mesh, tile, compute): past
//! the bound, the pipeline is its functions' compile on a device of the reduced policy, recording
//! the forms dropped (`RuntimeFact::ExactFormsOverCompilerLimit`) under its own compile's key and
//! profile fingerprint; within the bound, or on a device without one, it is the compile unchanged.
//! A pipeline linked from stages compiled ahead (the runtime's shape) is fitted from its
//! functions' descriptor and left as it is from its `Compiled` stages. The bound is pinned low in
//! code (`Device::fit_*_within`); no device runs, and no driver's compiler is driven.
//!
//! `llvm-as` assembles the owned fixtures; a missing tool skips a test visibly.

use std::collections::BTreeMap;
use std::path::Path;

use super::{Fitted, ReducedForms};
use crate::budget::{Budget, Meter};
use crate::cache::VariantKey;
use crate::device::policy::SamplingPolicy;
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::library::Library;
use crate::metal;
use crate::pipeline::artifact::{RenderPipeline, RuntimeDependency};
use crate::pipeline::descriptor::{
    ColorAttachmentDescriptor, ComputePipelineDescriptor, FunctionDescriptor,
    MeshPipelineDescriptor, RenderPipelineDescriptor, StageSource, TileRenderPipelineDescriptor,
};
use crate::pipeline::facts::{Mechanism, RuntimeFact};
use crate::pipeline::lower_inputs::LoweringInputs;
use crate::pipeline::sym::Stage;
use crate::pipeline::testing::{engine_dir, llvm_assemble, AssembleError};
use crate::vk::ShaderStage;

/// The owned fixture at `path` as a library, or `None` (skipped visibly) without `llvm-as`.
fn library(path: &Path) -> Option<Library> {
    match llvm_assemble(path) {
        Ok(bc) => Some(Library::from_air(&bc).expect("library")),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {path:?}: {why}"),
    }
}

/// Committed profile `stem` as the Khronos reader describes it.
fn description(stem: &str) -> DeviceDescription {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(&path).expect("profile");
    DeviceDescription::from_khronos_json(&text, None).expect("description")
}

/// Every exact form, Metal's sampler emulated among them.
fn exact() -> DevicePolicy {
    DevicePolicy {
        sampling: SamplingPolicy::Exact,
        ..DevicePolicy::default().exact_answers()
    }
}

/// Committed profile `stem` as a device under `policy`.
fn device(stem: &str, policy: DevicePolicy) -> Device {
    Device::new(description(stem), policy).expect("device")
}

/// The forms a fitted pipeline's `dependencies` record it dropped, in the fact's field order;
/// `None` where it records none.
fn dropped(dependencies: &[RuntimeDependency]) -> Option<[bool; 7]> {
    dependencies.iter().find_map(|d| match d.fact {
        RuntimeFact::ExactFormsOverCompilerLimit {
            sampling,
            float_maths,
            texel_conversion,
            half_rounding,
            nan_and_min_max,
            rounding_order,
            half_blend,
        } => Some([
            sampling,
            float_maths,
            texel_conversion,
            half_rounding,
            nan_and_min_max,
            rounding_order,
            half_blend,
        ]),
        _ => None,
    })
}

/// `reduced`, the compile of the fitted functions on a device of `exact()` with every
/// module-growing form native, as the fit of `compiled` (their compile under `exact()`) past its
/// bound gives it: recording that it dropped every exact form, named as `compiled` is.
fn fitted_as<P: Fitted>(reduced: P, compiled: &P) -> P {
    let fact = RuntimeFact::ExactFormsOverCompilerLimit {
        sampling: true,
        float_maths: true,
        texel_conversion: true,
        half_rounding: true,
        nan_and_min_max: true,
        rounding_order: true,
        half_blend: true,
    };
    let mut reduced = reduced;
    reduced
        .dependencies_mut()
        .push(RuntimeDependency::new(fact, Mechanism::None));
    reduced.named_as(compiled)
}

/// A render pipeline whose largest stage module passes a device's compiler word bound is
/// translated again with the device policy's exact forms dropped until it fits: native sampling
/// first, then every exact answer that grows a module native, recording which forms it dropped
/// (`RuntimeFact::ExactFormsOverCompilerLimit`) and keeping the compile's key and profile
/// fingerprint. A pipeline compiled from stages compiled ahead (the runtime's shape) is fitted from
/// its functions' descriptor ([`Device::fit_render_variant`]), and is left as it is from its
/// `Compiled` stages, which cannot be translated again. Uses the owned `runtime_examples` fixture
/// (a sampled `half` fragment), on a generic desktop profile whose sampler is not Metal's own.
#[test]
fn a_module_over_the_compiler_bound_drops_its_exact_forms_to_fit() {
    let path = metal2vulkan_devtools::fixtures_dir().join("examples/runtime_examples.ll");
    let Some(lib) = library(&path) else {
        return;
    };
    let vertex = lib.function("colored_vertex").expect("vertex");
    let fragment = lib.function("sampled_fragment").expect("fragment");
    let descriptor = || {
        let mut d = RenderPipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(
            vertex.clone(),
        )));
        d.fragment = Some(StageSource::Function(FunctionDescriptor::new(
            fragment.clone(),
        )));
        d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(
            metal::PixelFormat::RGBA8Unorm,
        ));
        d
    };
    let v = VariantKey::new();
    let b = Budget::standard();
    let stem = "lunarg-desktop-baseline-2023";
    let words = |policy: DevicePolicy| -> u32 {
        let pipe = device(stem, policy)
            .compile_render_variant(&descriptor(), &v, &b)
            .expect("compile");
        largest(&pipe)
    };
    fn largest(pipe: &RenderPipeline) -> u32 {
        pipe.largest_stage_words()
    }

    let w_exact = words(exact());
    let w_sampling_native = words(DevicePolicy::default().exact_answers());
    let w_fast = words(DevicePolicy::default());
    assert!(
        w_fast <= w_sampling_native && w_sampling_native <= w_exact,
        "fast {w_fast} <= sampling-native {w_sampling_native} <= exact {w_exact}"
    );

    let dev = device(stem, exact());
    let first = || {
        dev.compile_render_variant(&descriptor(), &v, &b)
            .expect("compile")
    };
    let fit = |bound: u32| {
        let first = first();
        let fitted = dev
            .fit_render_within(&descriptor(), &v, &b, first.clone(), Some(bound))
            .expect("fit");
        // The fitted pipeline is what the device's compile of the descriptor gives, as a cache
        // stores and finds it.
        assert_eq!(fitted.key, first.key, "the compile's key kept");
        assert_eq!(fitted.profile, first.profile, "the compile's profile kept");
        fitted
    };

    // A bound below every form drops every exact form the policy carries: the fast default.
    let pipe = fit(0);
    assert_eq!(largest(&pipe), w_fast, "reduced to the fast default");
    assert_eq!(
        dropped(&pipe.dependencies),
        Some([true; 7]),
        "every exact form dropped"
    );

    // Over the bound by exact sampling alone (where it grows the module): sampling is dropped
    // first, and nothing more, because native sampling already fits.
    if w_sampling_native < w_exact {
        let pipe = fit(w_sampling_native);
        assert_eq!(largest(&pipe), w_sampling_native);
        assert_eq!(
            dropped(&pipe.dependencies),
            Some([true, false, false, false, false, false, false]),
            "only sampling dropped"
        );
    }

    // Within the bound, or on a device without one: no fallback, no fact, the exact forms kept.
    for pipe in [
        fit(u32::MAX),
        dev.fit_render_within(&descriptor(), &v, &b, first(), None)
            .expect("no bound"),
        dev.fit_render_variant(&descriptor(), &v, &b, first())
            .expect("the device's own: none"),
    ] {
        assert_eq!(
            dropped(&pipe.dependencies),
            None,
            "no fallback within the bound"
        );
        assert_eq!(largest(&pipe), w_exact);
    }

    // The runtime's shape: each stage compiled ahead under the keys' inputs, linked from the
    // `Compiled` stages, then fitted from the functions' descriptor.
    let keys = dev.render_pipeline_key(&descriptor(), &b).expect("keys");
    let compiled = |source: &StageSource, inputs: &LoweringInputs| match source {
        StageSource::Function(f) => {
            StageSource::Compiled(dev.compile_function(f, inputs, &b).expect("stage"))
        }
        StageSource::Compiled(_) => unreachable!("the descriptor's stages are functions"),
    };
    let mut linked = descriptor();
    linked.vertex = compiled(&linked.vertex, keys.vertex_inputs.as_ref().expect("inputs"));
    linked.fragment = linked
        .fragment
        .as_ref()
        .map(|f| compiled(f, keys.fragment_inputs.as_ref().expect("inputs")));
    let ahead = dev
        .compile_render_variant(&linked, &v, &b)
        .expect("compiled stages");
    assert_eq!(ahead.key, keys.pipeline);
    assert_eq!(largest(&ahead), w_exact);
    // From the compiled stages alone nothing can be translated again: left as it is.
    let kept = dev
        .fit_render_within(&linked, &v, &b, ahead.clone(), Some(0))
        .expect("compiled stages kept");
    assert_eq!(kept, ahead);
    // From the functions' descriptor: fitted, under the compiled pipeline's key.
    let fitted = dev
        .fit_render_within(&descriptor(), &v, &b, ahead, Some(0))
        .expect("fitted from the functions");
    assert_eq!(fitted.key, keys.pipeline);
    assert_eq!(largest(&fitted), w_fast);
    assert_eq!(dropped(&fitted.dependencies), Some([true; 7]));
}

/// A mesh render pipeline is fitted as a vertex and fragment pipeline is: past the bound, its
/// object, mesh and fragment functions translated again with every exact form dropped (its
/// largest stage counted among all of them), recording the forms under its compile's key; within
/// the bound, on a device without one, or with a `Compiled` stage, the compile unchanged. Both
/// forms of a mesh pipeline: `TaskEXT` and `MeshEXT` stages on ANV's Arrow Lake profile, whose
/// own bound (400,000 words) the fixture fits, so its compile is the unfitted one; and on a
/// device without `VK_EXT_mesh_shader`, the compute stages and synthesized vertex stage that
/// emulate them. Uses the owned `mesh_draws` fixture.
#[test]
fn a_mesh_pipeline_over_the_compiler_bound_drops_its_exact_forms_to_fit() {
    let Some(lib) = library(&engine_dir().join("mesh_draws.ll")) else {
        return;
    };
    let function = |name: &str| FunctionDescriptor::new(lib.function(name).expect("function"));
    let descriptor = || {
        let mut d = MeshPipelineDescriptor::new(StageSource::Function(function("mesh_draws_mesh")));
        d.object = Some(StageSource::Function(function("mesh_draws_object")));
        d.fragment = Some(StageSource::Function(function("mesh_draws_fragment")));
        for color in &mut d.color_attachments[..2] {
            *color = Some(ColorAttachmentDescriptor::new(
                metal::PixelFormat::RGBA32Uint,
            ));
        }
        d
    };
    let (v, b) = (VariantKey::new(), Budget::standard());
    for (stem, bound, kinds) in [
        (
            "intel-arl-anv",
            Some(400_000),
            &[
                ShaderStage::TASK_EXT,
                ShaderStage::MESH_EXT,
                ShaderStage::FRAGMENT,
            ][..],
        ),
        (
            "lunarg-desktop-baseline-2023",
            None,
            &[
                ShaderStage::COMPUTE,
                ShaderStage::COMPUTE,
                ShaderStage::VERTEX,
                ShaderStage::FRAGMENT,
            ][..],
        ),
    ] {
        let dev = device(stem, exact());
        assert_eq!(dev.profile().module_word_bound(), bound, "{stem}");
        let compiled = dev.compile_mesh_variant(&descriptor(), &v, &b).expect(stem);
        let unfitted = dev
            .mesh_pipeline(&descriptor(), &v, &mut Meter::new(&b))
            .expect(stem);
        assert_eq!(compiled, unfitted, "{stem}: within the device's own bound");
        assert_eq!(dropped(&compiled.dependencies), None, "{stem}");
        let stages: Vec<ShaderStage> = compiled.stages.iter().map(|s| s.stage).collect();
        assert_eq!(stages, kinds, "{stem}");

        let fast = device(stem, ReducedForms::AllNative.of(exact()));
        let reduced = fast
            .mesh_pipeline(&descriptor(), &v, &mut Meter::new(&b))
            .expect(stem);
        let w_fast = device(stem, DevicePolicy::default())
            .compile_mesh_variant(&descriptor(), &v, &b)
            .expect(stem)
            .largest_stage_words();
        assert_eq!(
            reduced.largest_stage_words(),
            w_fast,
            "{stem}: the fast forms"
        );
        let w_exact = compiled.largest_stage_words();
        assert!(w_fast < w_exact, "{stem}: fast {w_fast} < exact {w_exact}");
        let fit = |d: &MeshPipelineDescriptor, bound| {
            dev.fit_mesh_within(d, &v, &b, compiled.clone(), bound)
                .expect(stem)
        };
        assert_eq!(
            fit(&descriptor(), Some(0)),
            fitted_as(reduced, &compiled),
            "{stem}: past the bound"
        );
        for bound in [Some(u32::MAX), None] {
            assert_eq!(
                fit(&descriptor(), bound),
                compiled,
                "{stem}: within {bound:?}"
            );
        }
        assert_eq!(
            dev.fit_mesh_variant(&descriptor(), &v, &b, compiled.clone())
                .expect(stem),
            compiled,
            "{stem}: the device's own bound"
        );
        // A `Compiled` stage cannot be translated again: left as it is, past the bound too.
        let object = dev
            .compile_function(
                &function("mesh_draws_object"),
                &v.lowering_inputs(Stage::Object),
                &b,
            )
            .expect(stem);
        let mut ahead = descriptor();
        ahead.object = Some(StageSource::Compiled(object));
        assert_eq!(fit(&ahead, Some(0)), compiled, "{stem}: a compiled stage");
    }
}

/// A tile render pipeline is fitted as a render pipeline is, its tile function translated again
/// under the inputs it was compiled under: past the bound, every exact form dropped, recorded
/// under the compile's key; within it, or on a device without one, the compile unchanged. In the
/// runtime's shape (the tile function compiled ahead under the descriptor's inputs and linked
/// from the `Compiled` stage) it is fitted from the function's descriptor and left as it is from
/// the `Compiled` one. On ANV's UHD 770 profile, whose own bound the fixture fits, and a device
/// without a bound. Uses the owned `ttg_tile_simd_blur` fixture (`half` arithmetic into an
/// `RGBA16Float` attachment), its threadgroup the tile's size.
#[test]
fn a_tile_pipeline_over_the_compiler_bound_drops_its_exact_forms_to_fit() {
    let Some(lib) = library(&engine_dir().join("ttg_tile_simd_blur.ll")) else {
        return;
    };
    let f = FunctionDescriptor::new(lib.function("ttg_tile_simd_blur").expect("function"));
    let format = metal::PixelFormat::RGBA16Float;
    let mut d = TileRenderPipelineDescriptor::new(StageSource::Function(f.clone()));
    d.color_attachment_formats[0] = Some(format);
    d.threadgroup_size_matches_tile_size = true;
    // The inputs `compile_tile_pipeline` compiles a `Function` tile function of `d` under.
    let inputs = LoweringInputs {
        tile_attachments: Some(BTreeMap::from([(0, format)])),
        threadgroup_size_matches_tile_size: true,
        ..LoweringInputs::new()
    };
    let b = Budget::standard();
    for (stem, bound) in [
        ("intel-uhd770-rpls-anv", Some(400_000)),
        ("lunarg-desktop-baseline-2023", None),
    ] {
        let dev = device(stem, exact());
        assert_eq!(dev.profile().module_word_bound(), bound, "{stem}");
        let compiled = dev.compile_tile_pipeline(&d, &b).expect(stem);
        let unfitted = dev
            .tile_pipeline(&d, &inputs, &mut Meter::new(&b))
            .expect(stem);
        assert_eq!(compiled, unfitted, "{stem}: within the device's own bound");
        assert_eq!(dropped(&compiled.dependencies), None, "{stem}");

        let fast = device(stem, ReducedForms::AllNative.of(exact()));
        let reduced = fast
            .tile_pipeline(&d, &inputs, &mut Meter::new(&b))
            .expect(stem);
        let w_fast = device(stem, DevicePolicy::default())
            .compile_tile_pipeline(&d, &b)
            .expect(stem)
            .largest_stage_words();
        assert_eq!(
            reduced.largest_stage_words(),
            w_fast,
            "{stem}: the fast forms"
        );
        let w_exact = compiled.largest_stage_words();
        assert!(w_fast < w_exact, "{stem}: fast {w_fast} < exact {w_exact}");
        let expected = fitted_as(reduced, &compiled);
        let fit = |d: &TileRenderPipelineDescriptor, pipeline, bound| {
            dev.fit_tile_within(d, &inputs, &b, pipeline, bound)
                .expect(stem)
        };
        assert_eq!(
            fit(&d, compiled.clone(), Some(0)),
            expected,
            "{stem}: past the bound"
        );
        for bound in [Some(u32::MAX), None] {
            assert_eq!(fit(&d, compiled.clone(), bound), compiled, "{stem}: within");
        }

        // The runtime's shape.
        let function = dev.compile_function(&f, &inputs, &b).expect(stem);
        let mut linked = d.clone();
        linked.tile = StageSource::Compiled(function);
        let ahead = dev.compile_tile_pipeline(&linked, &b).expect(stem);
        assert_eq!(ahead, compiled, "{stem}: linked from the compiled function");
        assert_eq!(
            dev.fit_tile_pipeline(&d, &inputs, &b, ahead.clone())
                .expect(stem),
            compiled,
            "{stem}: the device's own bound"
        );
        assert_eq!(
            fit(&linked, ahead.clone(), Some(0)),
            compiled,
            "{stem}: a compiled stage"
        );
        assert_eq!(
            fit(&d, ahead, Some(0)),
            expected,
            "{stem}: fitted from the function"
        );
    }
}

/// A compute pipeline is fitted as a render pipeline is, its kernel translated again under the
/// inputs it was compiled under, and one within the bound translates exactly as before the fit:
/// on ANV's UHD 770 profile, whose own bound the kernel fits, and on a device without a bound,
/// under every exact form and under the fast default, `compile_compute_pipeline` is the unfitted
/// compile, word for word. Past the bound every exact form is dropped; past it by exact sampling
/// alone, sampling alone. In the runtime's shape (the kernel compiled ahead and linked from the
/// `Compiled` stage) it is fitted from the function's descriptor and left as it is from the
/// `Compiled` one. Uses the owned `a2_kernel_lod_filter` fixture, a kernel that samples.
#[test]
fn a_compute_pipeline_over_the_compiler_bound_drops_its_exact_forms_to_fit() {
    let Some(lib) = library(&engine_dir().join("a2_kernel_lod_filter.ll")) else {
        return;
    };
    let f = FunctionDescriptor::new(lib.function("a2_kernel_lod_filter").expect("function"));
    let d = ComputePipelineDescriptor::new(StageSource::Function(f.clone()));
    // The inputs `compile_compute_pipeline` compiles a `Function` kernel of `d` under.
    let inputs = LoweringInputs::new();
    let b = Budget::standard();
    for (stem, bound) in [
        ("intel-uhd770-rpls-anv", Some(400_000)),
        ("lunarg-desktop-baseline-2023", None),
    ] {
        let unfitted = |dev: &Device| {
            dev.compute_pipeline(&d, &inputs, &mut Meter::new(&b))
                .expect(stem)
        };
        for policy in [exact(), DevicePolicy::default()] {
            let dev = device(stem, policy);
            assert_eq!(dev.profile().module_word_bound(), bound, "{stem}");
            let compiled = dev.compile_compute_pipeline(&d, &b).expect(stem);
            assert_eq!(
                compiled,
                unfitted(&dev),
                "{stem}: within the device's own bound, as before the fit"
            );
            assert_eq!(dropped(&compiled.dependencies), None, "{stem}");
        }

        let dev = device(stem, exact());
        let compiled = dev.compile_compute_pipeline(&d, &b).expect(stem);
        let w_exact = compiled.largest_stage_words();
        let sampled = unfitted(&device(stem, ReducedForms::NativeSampling.of(exact())));
        let w_sampling_native = sampled.largest_stage_words();
        let reduced = unfitted(&device(stem, ReducedForms::AllNative.of(exact())));
        let w_fast = device(stem, DevicePolicy::default())
            .compile_compute_pipeline(&d, &b)
            .expect(stem)
            .largest_stage_words();
        assert_eq!(
            reduced.largest_stage_words(),
            w_fast,
            "{stem}: the fast forms"
        );
        assert!(
            w_fast <= w_sampling_native && w_sampling_native < w_exact,
            "{stem}: fast {w_fast} <= sampling-native {w_sampling_native} < exact {w_exact}"
        );
        let expected = fitted_as(reduced, &compiled);
        let fit = |d: &ComputePipelineDescriptor, pipeline, bound| {
            dev.fit_compute_within(d, &inputs, &b, pipeline, bound)
                .expect(stem)
        };
        assert_eq!(
            fit(&d, compiled.clone(), Some(0)),
            expected,
            "{stem}: past the bound"
        );
        // Over the bound by exact sampling alone: sampling alone dropped.
        let pipe = fit(&d, compiled.clone(), Some(w_sampling_native));
        assert_eq!(pipe.stage, sampled.stage, "{stem}: native sampling");
        assert_eq!(
            dropped(&pipe.dependencies),
            Some([true, false, false, false, false, false, false]),
            "{stem}: only sampling dropped"
        );
        assert_eq!(
            (pipe.key, pipe.profile),
            (compiled.key, compiled.profile),
            "{stem}"
        );
        for bound in [Some(w_exact), Some(u32::MAX), None] {
            assert_eq!(fit(&d, compiled.clone(), bound), compiled, "{stem}: within");
        }

        // The runtime's shape.
        let function = dev.compile_function(&f, &inputs, &b).expect(stem);
        let linked = ComputePipelineDescriptor::new(StageSource::Compiled(function));
        let ahead = dev.compile_compute_pipeline(&linked, &b).expect(stem);
        assert_eq!(ahead, compiled, "{stem}: linked from the compiled function");
        assert_eq!(
            dev.fit_compute_pipeline(&d, &inputs, &b, ahead.clone())
                .expect(stem),
            compiled,
            "{stem}: the device's own bound"
        );
        assert_eq!(
            fit(&linked, ahead.clone(), Some(0)),
            compiled,
            "{stem}: a compiled stage"
        );
        assert_eq!(
            fit(&d, ahead, Some(0)),
            expected,
            "{stem}: fitted from the function"
        );
    }
}
