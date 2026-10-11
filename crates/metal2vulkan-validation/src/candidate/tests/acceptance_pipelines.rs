//! L4 driver acceptance on a device: the unsafe creation path of [`create_acceptance_pipeline`]
//! (set layouts, binding flags, specialization, subgroup size, render pass, graphics pipeline) runs
//! against the lane's Vulkan device. Nothing is bound or dispatched, so no shader executes.
//! `G-linux`'s `m2v_cpu_icd` step runs it on lavapipe.
//!
//! Each row compiles for the opened device's own `Device` (its queried description restricted to
//! what the context enabled, `platform::opened_worker_device`), never for a committed profile
//! named after the lane, so a lane whose device differs from any committed description still
//! checks what it would create.

use super::*;
use crate::acceptance::{
    compile_row_in_worker, retained, Acceptance, Projection, RowCompile, RowPipeline, RowStage,
};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};

/// Compile `entry` of the owned fixture `file` as an L4 row under `projection` for the device the
/// lane opens, in the device tests' compile worker ([`compile_worker`]), as `corpus-pipeline`
/// compiles its rows. A device lane has `llvm-as`, so a missing assembler fails the test rather
/// than skipping it.
fn compiled(file: &str, entry: &str, stage: RowStage, projection: Projection) -> RowPipeline {
    let bitcode = llvm_assemble(&engine_dir().join(file))
        .unwrap_or_else(|error| panic!("assemble {file}: {error:?}"));
    let device = platform::opened_worker_device(crate::gpu::TEST_DEVICE)
        .unwrap_or_else(|error| panic!("open the lane's device: {error:?}"));
    match compile_row_in_worker(
        compile_worker(),
        &device,
        &retained(&bitcode),
        entry,
        stage,
        projection,
        &metal2vulkan::library::FunctionConstantValues::defaults(),
    ) {
        (RowCompile::Compiled(pipeline), _) => *pipeline,
        (other, _) => panic!("{file} {entry} did not compile for the opened device: {other:?}"),
    }
}

/// The driver created the pipeline within the ceiling, or the device lacks something it needs; a
/// refusal or a tooling error (including an unavailable device) is a failure.
fn assert_accepted(pipeline: &RowPipeline) {
    match create_acceptance_pipeline(crate::gpu::TEST_DEVICE, pipeline) {
        Ok(Acceptance::Created { create }) => {
            assert!(
                crate::budgets::creation_within_ceiling("the pipeline", create).is_ok(),
                "{create:?}"
            );
        }
        Ok(Acceptance::NotEnabled(missing)) => eprintln!("NOT_ENABLED: {}", missing.join(", ")),
        Ok(Acceptance::Refused(why)) => panic!("the driver refused the pipeline: {why}"),
        Ok(Acceptance::InvalidUsage(messages)) => {
            panic!("the validation layer reported: {}", messages.join(" | "))
        }
        Err(error) => panic!("acceptance failed: {error}"),
    }
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn compute_and_vertex_only_pipelines_are_created_from_the_binder_layout() {
    let row = |file, entry, stage| compiled(file, entry, stage, Projection::Derived);
    let compute = row("store_const.ll", "store_const", RowStage::Kernel);
    assert!(matches!(compute, RowPipeline::Compute { .. }));
    assert_accepted(&compute);

    let vertex = row("stage_io.ll", "stage_io_vertex_position", RowStage::Vertex);
    assert!(matches!(vertex, RowPipeline::Render { .. }));
    assert_accepted(&vertex);
}

/// Create kernel row `entry` of the owned fixture `file` from the region the binder plans for its
/// representative dispatch, after printing the row's dispatch, its stage's flags and the region's,
/// the required subgroup size, and the imageblock extent: they show whether the lane's tier
/// requires full subgroups.
fn assert_created_from_its_region(file: &str, entry: &str) {
    let pipeline = compiled(file, entry, RowStage::Kernel, Projection::Derived);
    let RowPipeline::Compute {
        pipeline: compute,
        creation,
    } = &pipeline
    else {
        panic!("{entry}: a kernel row is a compute pipeline");
    };
    eprintln!(
        "{entry}: {:?}, stage flags {:?} (the region's {:?}), required subgroup size {:?}, \
         imageblock {:?}",
        creation.dispatch,
        compute.stage.stage_flags,
        creation.region.stage_flags,
        creation.region.required_subgroup_size,
        creation.imageblock
    );
    assert_accepted(&pipeline);
}

/// A kernel whose simd tier can require full subgroups is created from its region. The 897
/// `VUID-VkPipelineShaderStageCreateInfo-flags-02759` rows were such kernels created at their
/// stage's placeholder local size of 1.
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn a_full_subgroup_kernel_row_is_created_from_its_region() {
    assert_created_from_its_region("simd_ops.ll", "simd_shuffle_k");
}

/// A kernel with an explicit imageblock is created from its region, whose specialization sets
/// the representative extent.
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn an_imageblock_kernel_row_is_created_from_its_region() {
    assert_created_from_its_region("a4_imageblock_extent.ll", "a4_imageblock_extent");
}

/// A post-tessellation vertex function's row is a tessellation pipeline (the synthesized vertex
/// and control stages before its evaluation stage), created as a patch list of one control point
/// per patch in Metal's default tessellation state, with the patch attributes and control-point
/// fields the row lays out pulled.
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn post_tessellation_vertex_rows_are_created() {
    for entry in ["tess_quad_buffer", "tess_triangle_points"] {
        let pipeline = compiled(
            "tessellation.ll",
            entry,
            RowStage::Vertex,
            Projection::Derived,
        );
        let RowPipeline::Render {
            pipeline: render, ..
        } = &pipeline
        else {
            panic!("{entry}: a vertex row is a render pipeline");
        };
        assert!(render.graphics.tessellation.is_some(), "{entry}");
        assert_eq!(render.stages.len(), 3, "{entry}");
        assert_accepted(&pipeline);
    }
}

/// A vertex function that reads attributes is created with the descriptor the row synthesizes,
/// fixed-function and with every attribute pulled, and a fragment function with its generated
/// passthrough companion into the attachments its outputs imply.
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn attribute_reading_vertex_and_fragment_rows_are_created() {
    for projection in [Projection::Derived, Projection::Pulled] {
        let vertex = compiled(
            "stage_io.ll",
            "stage_io_vertex",
            RowStage::Vertex,
            projection,
        );
        assert_accepted(&vertex);
    }
    let fragment = compiled(
        "stage_io.ll",
        "stage_io_fragment",
        RowStage::Fragment,
        Projection::Derived,
    );
    let RowPipeline::Render {
        pipeline: render, ..
    } = &fragment
    else {
        panic!("a fragment row compiles to a render pipeline");
    };
    assert!(!render.graphics.rasterizer_discard);
    assert_accepted(&fragment);
}
