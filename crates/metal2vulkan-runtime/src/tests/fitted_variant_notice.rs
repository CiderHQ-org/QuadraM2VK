//! The exact forms a pipeline's translation dropped to fit the device's compiler word bound
//! (`RuntimeFact::ExactFormsOverCompilerLimit`) are announced from each variant's own pipeline:
//! `Device::degrade_pipeline`, which every variant's making runs (the base variant and every
//! other, of a render, mesh render, compute or tile render state alike), and `Kinds::of`, whose
//! kinds a bind's notices are made of. So a variant the fit reduced while its base was not
//! announces `exact-forms-over-compiler-limit`, once per device, naming the forms dropped and its
//! state's kind, and a bind of it on a device asking for the kind at every occurrence announces it
//! there; the base announces nothing of it.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The pipelines are the owned
//! `runtime_examples`, `mesh_draws` and `cfb_tile_norm` fixtures' (`llvm-as`), compiled for the
//! device, each fitted one carrying the fit's record as the translation adds it; nothing runs on
//! the GPU.

use crate::degradation::{CompiledPipeline, Kinds, Occurrence};
use crate::device::lock;
use crate::{Degradation, Device, DeviceKind, DeviceOptions, MessageSink};
use metal2vulkan::library::Library;
use metal2vulkan::metal::PixelFormat;
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, ComputePipeline, ComputePipelineDescriptor, FunctionDescriptor,
    Mechanism, MeshPipelineDescriptor, RenderPipeline, RenderPipelineDescriptor, RuntimeDependency,
    RuntimeFact, StageModule, StageSource, TileRenderPipelineDescriptor,
};
use metal2vulkan::Budget;
use std::collections::BTreeSet;
use std::path::Path;
use std::sync::{Arc, Mutex};

/// What the device's sink heard: each degrade notice's kind, object and reason.
type Heard = Arc<Mutex<Vec<(Degradation, String, String)>>>;

/// The device the cfg selects, asking for `each` at every occurrence, with a sink that keeps every
/// degrade notice.
fn listening(each: BTreeSet<Degradation>) -> (Device, Heard) {
    let heard: Heard = Arc::default();
    let kept = Arc::clone(&heard);
    let sink = MessageSink::new(move |message: &crate::DebugMessage| {
        if let Some(notice) = &message.degradation {
            lock(&kept).push((
                notice.degradation,
                notice.object.clone(),
                notice.reason.clone(),
            ));
        }
    });
    let options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        messages: Some(sink),
        announce_each: each,
        ..DeviceOptions::default()
    };
    (Device::system_default_with(&options).unwrap(), heard)
}

/// The owned fixture `name` (under the public fixtures) as a library.
fn library(name: &str) -> Library {
    let ll = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/fixtures/public")
        .join(name);
    let bitcode = metal2vulkan_devtools::llvm_assemble(&ll).unwrap();
    Library::from_air(&bitcode).unwrap()
}

/// Add to `dependencies` the fit's record of dropping exact sampling and exact float maths.
fn fit_into(dependencies: &mut Vec<RuntimeDependency>) {
    dependencies.push(RuntimeDependency::new(
        RuntimeFact::ExactFormsOverCompilerLimit {
            sampling: true,
            float_maths: true,
            texel_conversion: false,
            half_rounding: false,
            nan_and_min_max: false,
            rounding_order: false,
            half_blend: false,
        },
        Mechanism::None,
    ));
}

/// `pipeline` as the fit records it.
fn render_fit(pipeline: &RenderPipeline) -> RenderPipeline {
    let mut fit = pipeline.clone();
    fit_into(&mut fit.dependencies);
    fit
}

/// `pipeline` as the fit records it.
fn kernel_fit(pipeline: &ComputePipeline) -> ComputePipeline {
    let mut fit = pipeline.clone();
    fit_into(&mut fit.dependencies);
    fit
}

/// The fixture's render pipeline (`colored_vertex`, `colored_fragment`, one `RGBA8Unorm` target),
/// compiled for `device`, and the same pipeline as the fit records it.
fn pipelines(device: &Device) -> (RenderPipeline, RenderPipeline) {
    let library = library("examples/runtime_examples.ll");
    let function = |name| FunctionDescriptor::new(library.function(name).unwrap());
    let mut descriptor =
        RenderPipelineDescriptor::new(StageSource::Function(function("colored_vertex")));
    descriptor.fragment = Some(StageSource::Function(function("colored_fragment")));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    let base = device
        .compile_device()
        .compile_render_pipeline(&descriptor, &Budget::standard())
        .unwrap();
    let fit = render_fit(&base);
    (base, fit)
}

/// A mesh render pipeline (`mesh_draws`' object, mesh and fragment functions, two `RGBA32Uint`
/// targets), compiled for `device`: its `MeshEXT` stages, or the compute and vertex stages that
/// emulate them.
fn mesh_pipeline(device: &Device) -> RenderPipeline {
    let library = library("engine/mesh_draws.ll");
    let stage =
        |name| StageSource::Function(FunctionDescriptor::new(library.function(name).unwrap()));
    let mut descriptor = MeshPipelineDescriptor::new(stage("mesh_draws_mesh"));
    descriptor.object = Some(stage("mesh_draws_object"));
    descriptor.fragment = Some(stage("mesh_draws_fragment"));
    for color in &mut descriptor.color_attachments[..2] {
        *color = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA32Uint));
    }
    device
        .compile_device()
        .compile_mesh_pipeline(&descriptor, &Budget::standard())
        .unwrap()
}

/// A compute pipeline (`runtime_examples`' `scale_and_offset` kernel) and a tile render pipeline
/// (`cfb_tile_norm`'s write of an `RGBA8Unorm` attachment), compiled for `device`.
fn kernel_pipelines(device: &Device) -> (ComputePipeline, ComputePipeline) {
    let compile = device.compile_device();
    let examples = library("examples/runtime_examples.ll");
    let kernel = FunctionDescriptor::new(examples.function("scale_and_offset").unwrap());
    let compute = compile
        .compile_compute_pipeline(
            &ComputePipelineDescriptor::new(StageSource::Function(kernel)),
            &Budget::standard(),
        )
        .unwrap();
    let tiles = library("engine/cfb_tile_norm.ll");
    let tile = FunctionDescriptor::new(tiles.function("cfb_tn_write4").unwrap());
    let mut descriptor = TileRenderPipelineDescriptor::new(StageSource::Function(tile));
    descriptor.color_attachment_formats[0] = Some(PixelFormat::RGBA8Unorm);
    let tile = compile
        .compile_tile_pipeline(&descriptor, &Budget::standard())
        .unwrap();
    (compute, tile)
}

/// A render variant of `pipeline`: its stages and pipeline, as its making and binds announce them.
fn render_variant(pipeline: &RenderPipeline) -> (&[StageModule], CompiledPipeline<'_>) {
    (&pipeline.stages, CompiledPipeline::Render(pipeline))
}

/// A compute or tile variant of `pipeline`: its stage and pipeline, as its making and binds
/// announce them.
fn kernel_variant(pipeline: &ComputePipeline) -> (&[StageModule], CompiledPipeline<'_>) {
    (
        std::slice::from_ref(&pipeline.stage),
        CompiledPipeline::Kernel(pipeline),
    )
}

/// The notices of `kind` the sink heard, each's object and reason.
fn of_kind(heard: &Heard, kind: Degradation) -> Vec<(String, String)> {
    lock(heard)
        .iter()
        .filter(|(k, _, _)| *k == kind)
        .map(|(_, object, reason)| (object.clone(), reason.clone()))
        .collect()
}

#[test]
fn a_variant_fitted_while_its_base_was_not_still_announces() {
    let kind = Degradation::ExactFormsOverCompilerLimit;
    let (device, heard) = listening(BTreeSet::new());
    let (base, fit) = pipelines(&device);
    // The base variant's making: no form dropped, nothing of the kind.
    device.degrade_pipeline(CompiledPipeline::Render(&base), Occurrence::Made);
    assert!(of_kind(&heard, kind).is_empty(), "the base announced it");
    // Another variant's making, its pipeline fitted: announced, naming the forms dropped.
    device.degrade_pipeline(CompiledPipeline::Render(&fit), Occurrence::Made);
    let said = of_kind(&heard, kind);
    assert_eq!(said.len(), 1, "{said:?}");
    let (object, reason) = &said[0];
    assert_eq!(object, "a render pipeline state");
    assert!(
        reason.contains("exact sampling, exact float maths")
            && !reason.contains("exact half blend"),
        "{reason}"
    );
    // Once per device: a later fitted variant adds no second notice.
    device.degrade_pipeline(CompiledPipeline::Render(&fit), Occurrence::Made);
    assert_eq!(of_kind(&heard, kind).len(), 1);

    // A device asking for the kind at every occurrence: the fitted variant's kinds hold it, and each
    // bind's notices carry it, where the base's hold none of it; its making announces nothing.
    let (each, heard) = listening(BTreeSet::from([kind]));
    let (base, fit) = pipelines(&each);
    let (base, fit) = (render_variant(&base), render_variant(&fit));
    each.degrade_pipeline(fit.1, Occurrence::Made);
    assert!(of_kind(&heard, kind).is_empty(), "announced at the making");
    assert!(each.binds_each(&Kinds::of(fit.0, fit.1)));
    assert!(!each.binds_each(&Kinds::of(base.0, base.1)));
    for _ in 0..2 {
        let notices = each.bound_notices(fit, &[]);
        let kinds: Vec<Degradation> = notices.iter().map(|(k, _)| *k).collect();
        assert_eq!(kinds, [kind], "a bind of the fitted variant");
    }
    assert!(each
        .bound_notices(base, &[])
        .iter()
        .all(|(k, _)| *k != kind));
}

/// A mesh render, compute or tile render state's fitted variant announces the kind as a vertex
/// and fragment state's does, naming its own state's kind, read from its pipeline's structure:
/// at its making (a compute variant's too, which `pipeline/mod.rs` makes), and at each bind on a
/// device asking for the kind at every occurrence, where its unfitted base announces nothing of
/// it.
#[test]
fn every_kind_of_fitted_pipeline_announces_its_state_kind() {
    let kind = Degradation::ExactFormsOverCompilerLimit;
    // A compute variant's making: announced, as a compute pipeline state's.
    let (device, heard) = listening(BTreeSet::new());
    let (compute, _) = kernel_pipelines(&device);
    device.degrade_pipeline(CompiledPipeline::Kernel(&compute), Occurrence::Made);
    assert!(of_kind(&heard, kind).is_empty(), "the base announced it");
    let fit = kernel_fit(&compute);
    device.degrade_pipeline(CompiledPipeline::Kernel(&fit), Occurrence::Made);
    let said = of_kind(&heard, kind);
    assert_eq!(said.len(), 1, "{said:?}");
    assert_eq!(said[0].0, "a compute pipeline state");
    assert!(
        said[0].1.contains("exact sampling, exact float maths"),
        "{said:?}"
    );

    // Each kind's bind, on a device asking for the kind at every occurrence.
    let (each, _) = listening(BTreeSet::from([kind]));
    let (render, _) = pipelines(&each);
    let mesh = mesh_pipeline(&each);
    let (compute, tile) = kernel_pipelines(&each);
    let (render_fitted, mesh_fitted) = (render_fit(&render), render_fit(&mesh));
    let (compute_fitted, tile_fitted) = (kernel_fit(&compute), kernel_fit(&tile));
    let cases = [
        (
            "a render pipeline state",
            render_variant(&render),
            render_variant(&render_fitted),
        ),
        (
            "a mesh render pipeline state",
            render_variant(&mesh),
            render_variant(&mesh_fitted),
        ),
        (
            "a compute pipeline state",
            kernel_variant(&compute),
            kernel_variant(&compute_fitted),
        ),
        (
            "a tile render pipeline state",
            kernel_variant(&tile),
            kernel_variant(&tile_fitted),
        ),
    ];
    for (object, base, fit) in cases {
        assert!(each.binds_each(&Kinds::of(fit.0, fit.1)), "{object}");
        let notices = each.bound_notices(fit, &[]);
        let named: Vec<&str> = notices
            .iter()
            .filter(|(k, _)| *k == kind)
            .filter_map(|(_, message)| Some(message.degradation.as_ref()?.object.as_str()))
            .collect();
        assert_eq!(named, [object], "a bind of the fitted variant");
        assert!(
            each.bound_notices(base, &[])
                .iter()
                .all(|(k, _)| *k != kind),
            "{object}: its base"
        );
    }
}
