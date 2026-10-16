//! The exact forms a render pipeline's translation dropped to fit the device's compiler word bound
//! (`RuntimeFact::ExactFormsOverCompilerLimit`) are announced from each render variant's own
//! pipeline: `Device::degrade_render`, which every render variant's making runs (the base variant
//! and every other), and `Kinds::of`, whose kinds a bind's notices are made of. So a variant the
//! fit reduced while its base was not announces `exact-forms-over-compiler-limit`, once per device,
//! naming the forms dropped, and a bind of it on a device asking for the kind at every occurrence
//! announces it there; the base announces nothing of it.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The pipelines are the owned
//! `runtime_examples` fixture's (`llvm-as`), the fitted one carrying the fit's record as the
//! translation adds it.

use crate::degradation::{Kinds, Occurrence};
use crate::device::lock;
use crate::{Degradation, Device, DeviceKind, DeviceOptions, MessageSink};
use metal2vulkan::metal::PixelFormat;
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, FunctionDescriptor, Mechanism, RenderPipeline,
    RenderPipelineDescriptor, RuntimeDependency, RuntimeFact, StageSource,
};
use std::collections::BTreeSet;
use std::path::Path;
use std::sync::{Arc, Mutex};

/// What the device's sink heard: each degrade notice's kind and reason.
type Heard = Arc<Mutex<Vec<(Degradation, String)>>>;

/// The device the cfg selects, asking for `each` at every occurrence, with a sink that keeps every
/// degrade notice.
fn listening(each: BTreeSet<Degradation>) -> (Device, Heard) {
    let heard: Heard = Arc::default();
    let kept = Arc::clone(&heard);
    let sink = MessageSink::new(move |message: &crate::DebugMessage| {
        if let Some(notice) = &message.degradation {
            lock(&kept).push((notice.degradation, notice.reason.clone()));
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

/// The fixture's render pipeline (`colored_vertex`, `colored_fragment`, one `RGBA8Unorm` target),
/// compiled for `device`, and the same pipeline as the fit records it where it dropped exact
/// sampling and exact float maths.
fn pipelines(device: &Device) -> (RenderPipeline, RenderPipeline) {
    let ll = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/fixtures/public/examples/runtime_examples.ll");
    let bitcode = metal2vulkan_devtools::llvm_assemble(&ll).unwrap();
    let library = metal2vulkan::library::Library::from_air(&bitcode).unwrap();
    let function = |name| FunctionDescriptor::new(library.function(name).unwrap());
    let mut descriptor =
        RenderPipelineDescriptor::new(StageSource::Function(function("colored_vertex")));
    descriptor.fragment = Some(StageSource::Function(function("colored_fragment")));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    let base = device
        .compile_device()
        .compile_render_pipeline(&descriptor, &metal2vulkan::Budget::standard())
        .unwrap();
    let mut fitted = base.clone();
    fitted.dependencies.push(RuntimeDependency::new(
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
    (base, fitted)
}

/// The notices of `kind` the sink heard.
fn of_kind(heard: &Heard, kind: Degradation) -> Vec<String> {
    lock(heard)
        .iter()
        .filter(|(k, _)| *k == kind)
        .map(|(_, reason)| reason.clone())
        .collect()
}

#[test]
fn a_variant_fitted_while_its_base_was_not_still_announces() {
    let kind = Degradation::ExactFormsOverCompilerLimit;
    let (device, heard) = listening(BTreeSet::new());
    let (base, fitted) = pipelines(&device);
    // The base variant's making: no form dropped, nothing of the kind.
    device.degrade_render(&base, Occurrence::Made);
    assert!(of_kind(&heard, kind).is_empty(), "the base announced it");
    // Another variant's making, its pipeline fitted: announced, naming the forms dropped.
    device.degrade_render(&fitted, Occurrence::Made);
    let said = of_kind(&heard, kind);
    assert_eq!(said.len(), 1, "{said:?}");
    assert!(
        said[0].contains("exact sampling, exact float maths")
            && !said[0].contains("exact half blend"),
        "{}",
        said[0]
    );
    // Once per device: a later fitted variant adds no second notice.
    device.degrade_render(&fitted, Occurrence::Made);
    assert_eq!(of_kind(&heard, kind).len(), 1);

    // A device asking for the kind at every occurrence: the fitted variant's kinds hold it, and each
    // bind's notices carry it, where the base's hold none of it; its making announces nothing.
    let (each, heard) = listening(BTreeSet::from([kind]));
    let (base, fitted) = pipelines(&each);
    each.degrade_render(&fitted, Occurrence::Made);
    assert!(of_kind(&heard, kind).is_empty(), "announced at the making");
    assert!(each.binds_each(&Kinds::of(&fitted.stages, Some(&fitted))));
    assert!(!each.binds_each(&Kinds::of(&base.stages, Some(&base))));
    for _ in 0..2 {
        let notices = each.bound_notices((&fitted.stages, Some(&fitted)), &[]);
        let kinds: Vec<Degradation> = notices.iter().map(|(k, _)| *k).collect();
        assert_eq!(kinds, [kind], "a bind of the fitted variant");
    }
    assert!(each
        .bound_notices((&base.stages, Some(&base)), &[])
        .iter()
        .all(|(k, _)| *k != kind));
}
