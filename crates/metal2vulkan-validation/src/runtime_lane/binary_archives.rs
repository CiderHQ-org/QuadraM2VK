//! Binary archives through the runtime, on the device the cfg selects, under the validation layer.
//! The pipelines are the owned fixture `rt6_render_state`'s fullscreen white triangle
//! (`render_state`'s probe) and the public kernel case `threadgroup-word`.
//!
//! - An archive a render and a compute pipeline were added to, serialized and loaded on a second
//!   device with an empty shader cache, gives that device's creations every translation: its
//!   shader cache counts no lookup, and the draw stores what the first device's does.
//! - Under `FAIL_ON_BINARY_ARCHIVE_MISS`, a creation hits where Metal's does (another depth
//!   format still hits; one add's vertex stage beside another's fragment stage hits, announced
//!   once as [`ARCHIVES`]' `combined-archive-stages`) and is refused where Metal's misses, naming
//!   the function of the stage missed as Metal does: the fragment function for another color
//!   format, the vertex function for another topology class, the kernel under another threadgroup
//!   limit. Without the option, or with no archive listed, the same creations compile.
//! - Metal's errors: an empty archive does not serialize, a missing file does not load, an add
//!   answers its creation's error, and another device's archive is refused.

use super::degrade_notices::assert_once;
use super::render_state::{library, Probe, PASS};
use super::{
    compute_descriptor, pipeline_state, public_case, validated_device, Messages, ScratchCache,
};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::metal::{PixelFormat, PrimitiveTopologyClass, PrimitiveType};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, LinkedFunctions, RenderPipelineDescriptor, StageSource,
};
use metal2vulkan_runtime::{
    ArchivedPipeline, BinaryArchiveDescriptor, BinaryArchiveError, BinaryArchiveMiss, Degradation,
    Device, Error, Invalid, PipelineOption, PipelineOptions, Problem, RenderPipelineState,
};

/// The kind [`an_archive_serves_a_fresh_device_and_fails_where_metal_misses`] takes: a render
/// pipeline whose vertex and fragment stages an archive holds from two adds, not as one add's
/// pair, created twice and announced once.
pub(super) const ARCHIVES: [Degradation; 1] = [Degradation::CombinedArchiveStages];

/// The probe's triangle of `vertex` (`rt6_full`, the fullscreen one), shaded by `fragment`
/// (`rt6_white`), into a target of `format` (`RGBA8Unorm`).
fn shaded(
    probe: &Probe,
    vertex: &str,
    fragment: &str,
    format: PixelFormat,
) -> RenderPipelineDescriptor {
    let function = |name| probe.library.new_function_with_name(name).unwrap();
    let mut descriptor = RenderPipelineDescriptor::new(StageSource::Function(
        function(vertex).function_descriptor(),
    ));
    descriptor.fragment = Some(StageSource::Function(
        function(fragment).function_descriptor(),
    ));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(format));
    descriptor
}

/// The probe's white triangle of `vertex` into an `RGBA8Unorm` target.
fn descriptor(probe: &Probe, vertex: &str) -> RenderPipelineDescriptor {
    shaded(probe, vertex, "rt6_white", PixelFormat::RGBA8Unorm)
}

/// The probe of `device`.
fn probe(device: &Device) -> Probe {
    Probe {
        device: device.clone(),
        queue: device.new_command_queue().unwrap(),
        library: library(device),
    }
}

/// The color `state` stores drawing the probe's triangle into the probe's pass.
fn drawn(probe: &Probe, state: &RenderPipelineState) -> Vec<u8> {
    probe
        .run(PASS, |probe, encoder, held| {
            encoder.set_render_pipeline_state(state).unwrap();
            let depth = probe.depth(0.5, 0.0);
            encoder.set_vertex_buffer(Some(&depth), 0, 0).unwrap();
            encoder
                .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                .unwrap();
            held.push(depth);
        })
        .color
}

/// The options that fail on a miss in `archives`.
fn failing(archives: &[metal2vulkan_runtime::BinaryArchive]) -> PipelineOptions {
    let mut options = PipelineOptions::new(PipelineOption::FAIL_ON_BINARY_ARCHIVE_MISS);
    options.binary_archives = archives.to_vec();
    options
}

#[test]
fn an_archive_serves_a_fresh_device_and_fails_where_metal_misses() {
    let files = ScratchCache::new().unwrap();
    std::fs::create_dir_all(files.path()).unwrap();
    let file = files.path().join("pipelines.metallib-archive");
    let case = public_case("threadgroup-word").unwrap();
    // The case's compute descriptor on `device`, as the Metal oracle states it.
    let kernel = |device: &Device| {
        let function = device
            .new_library_with_air(&case.bitcode)
            .unwrap()
            .new_function_with_name_constant_values(&case.compute.entry, &case.compute.constants)
            .unwrap();
        compute_descriptor(&function, LinkedFunctions::new(), &case.compute)
    };
    let (first_cache, second_cache) = (ScratchCache::new().unwrap(), ScratchCache::new().unwrap());
    let messages = Messages::default();
    {
        let first = validated_device(TEST_DEVICE, &messages, first_cache.path()).unwrap();
        let first_probe = probe(&first);
        let white = descriptor(&first_probe, "rt6_full");
        let archive = first
            .new_binary_archive(&BinaryArchiveDescriptor::default())
            .unwrap();
        assert_eq!(
            archive.serialize_to_url(&file),
            Err(Error::BinaryArchive(BinaryArchiveError::NothingToSerialize))
        );
        archive.set_label("not serialized");
        archive.add_render_pipeline_functions(&white).unwrap();
        archive.add_render_pipeline_functions(&white).unwrap();
        // An add answers its creation's error: a fragment function as the vertex function.
        let wrong = descriptor(&first_probe, "rt6_white");
        assert!(archive.add_render_pipeline_functions(&wrong).is_err());
        assert_eq!(
            archive
                .add_render_pipeline_functions(&wrong)
                .map(|()| "added"),
            first
                .new_render_pipeline_state(&wrong, &PipelineOptions::default())
                .map(|_| "added")
        );
        archive
            .add_render_pipeline_functions(&shaded(
                &first_probe,
                "rt6_flip",
                "rt6_half",
                PixelFormat::BGRA8Unorm,
            ))
            .unwrap();
        archive
            .add_compute_pipeline_functions(&kernel(&first))
            .unwrap();
        archive.serialize_to_url(&file).unwrap();
        let expected = drawn(
            &first_probe,
            &first
                .new_render_pipeline_state(&white, &PipelineOptions::default())
                .unwrap(),
        );

        let second = validated_device(TEST_DEVICE, &messages, second_cache.path()).unwrap();
        let second_probe = probe(&second);
        let white = descriptor(&second_probe, "rt6_full");
        let missing = files.path().join("missing");
        match second.new_binary_archive(&BinaryArchiveDescriptor::new(Some(missing.clone()))) {
            Err(Error::BinaryArchive(BinaryArchiveError::InvalidFile { path })) => {
                assert_eq!(path, missing)
            }
            other => panic!("a missing file: {other:?}"),
        }
        let loaded = second
            .new_binary_archive(&BinaryArchiveDescriptor::new(Some(file.clone())))
            .unwrap();
        assert_eq!(loaded.label(), None, "labels are not serialized");
        let options = failing(std::slice::from_ref(&loaded));
        let state = second.new_render_pipeline_state(&white, &options).unwrap();
        let compute = kernel(&second);
        second
            .new_compute_pipeline_state(&compute, &options)
            .unwrap();
        let s = second.shader_cache_statistics();
        assert_eq!(
            (s.hits, s.misses, s.writes),
            (0, 0, 0),
            "the archive gave every translation: {s:?}"
        );
        assert_eq!(drawn(&second_probe, &state), expected);

        let mut depth = white.clone();
        depth.depth_format = Some(PixelFormat::Depth32Float);
        second.new_render_pipeline_state(&depth, &options).unwrap();
        // The notices so far are the two devices' own; the stages of two adds are announced
        // once, however often a creation combines them.
        messages.take_notices();
        let combined = shaded(
            &second_probe,
            "rt6_full",
            "rt6_half",
            PixelFormat::BGRA8Unorm,
        );
        for _ in 0..2 {
            second
                .new_render_pipeline_state(&combined, &options)
                .unwrap();
        }
        assert_once(&messages.take_notices(), &ARCHIVES);
        let mut color = white.clone();
        color.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::BGRA8Unorm));
        let miss = |function| {
            Err(Error::BinaryArchiveMiss(BinaryArchiveMiss::new(
                function,
                ArchivedPipeline::Render,
            )))
        };
        assert_eq!(
            second
                .new_render_pipeline_state(&color, &options)
                .map(|_| ()),
            miss("rt6_white")
        );
        let mut triangles = white.clone();
        triangles.input_primitive_topology = PrimitiveTopologyClass::Triangle;
        assert_eq!(
            second
                .new_render_pipeline_state(&triangles, &options)
                .map(|_| ()),
            miss("rt6_full")
        );
        assert_eq!(
            Error::BinaryArchiveMiss(BinaryArchiveMiss::new(
                "rt6_white",
                ArchivedPipeline::Render
            ))
            .to_string(),
            "Unable to find function \"rt6_white\" (render pipeline) in binary archives."
        );
        second
            .new_render_pipeline_state(&color, &failing(&[]))
            .unwrap();
        let mut listed = PipelineOptions::default();
        listed.binary_archives = vec![loaded.clone()];
        second.new_render_pipeline_state(&color, &listed).unwrap();

        let mut limited = compute.clone();
        limited.max_total_threads_per_threadgroup = Some(64);
        assert_eq!(
            second
                .new_compute_pipeline_state(&limited, &options)
                .map(|_| ()),
            Err(Error::BinaryArchiveMiss(BinaryArchiveMiss::new(
                &case.compute.entry,
                ArchivedPipeline::Compute,
            )))
        );
        pipeline_state(&second, &case).unwrap();

        assert_eq!(
            second
                .new_render_pipeline_state(&white, &failing(&[archive]))
                .map(|_| ()),
            Err(Error::Invalid(Invalid::new(
                "new_render_pipeline_state",
                Problem::ForeignObject
            )))
        );
    }
    let messages = messages.take();
    assert!(
        messages.is_empty(),
        "the validation layer reported: {messages:#?}"
    );
}
