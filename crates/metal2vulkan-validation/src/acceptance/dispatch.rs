//! The dispatch region an L4 kernel row's pipeline is created from.
//!
//! A kernel's stage carries its specialization constants at placeholder defaults (a local size
//! of 1) that the binder replaces, and the stage create flags of its simd tier: in a `Direct` or
//! `Cluster32` tier `REQUIRE_FULL_SUBGROUPS`, valid only with a local size of whole subgroups. A
//! pipeline created at the defaults is therefore invalid usage
//! (`VUID-VkPipelineShaderStageCreateInfo-flags-02759`: 897 kernel rows), and one the binder never
//! creates. The executors create each region's pipeline from the region the binder plans (its
//! specialization, stage flags, and required subgroup size); so does the sweep, for a
//! representative dispatch of the row: one threadgroup of the fewest threads that are whole
//! simdgroups (Metal's thread execution width, 32, or a multiple of it where the tier's subgroups
//! are wider), within the pipeline's `max_total_threads_per_threadgroup`, the smallest the binder
//! runs on the compiled pipeline itself rather than on its emulated-simdgroups variant.
//!
//! The binder plans it for the encoder's defaults, as `BindState::default` states them: nothing
//! bound, no threadgroup memory length set (Metal's 0), and a stage-in region of the dispatch's
//! own threads (which only a dispatch's bytes read, never its specialization). A kernel with an
//! explicit imageblock (`RuntimeFact::ImageblockSize`) has no default: Metal's encoder must set its
//! extent (`setImageblockWidth:height:`), so the dispatch sets one cell per thread of the
//! threadgroup, `w` by 1. That extent is this sweep's representative choice, not a binder fact.
//! Another extent is another specialization of the same pipeline (the binder plans no other variant
//! for it), and its `width · height · cell_bytes` bytes join the threadgroup total that must fit
//! `maxComputeSharedMemorySize`: a larger extent can be refused where this one is created. The
//! row's detail names the extent it was created with (`imageblock=WxH`).
//!
//! The row is compiled as the translation sweep compiles it, with no variant, so a row whose plan
//! asks for one, or that no such threadgroup runs on, is [`Unmodelled`] of kind
//! [`UnmodelledKind::Dispatch`]: the representative dispatch does not reach it. A binder error
//! for the representative dispatch is the binder's answer, not a tool limit
//! ([`BinderAnswer`]). A typed refusal the product documents (`BindError::Unsupported`: the
//! device cannot bind what the dispatch needs) is counted. Any other error fails the sweep: a
//! binder defect (`BindError::Internal`), or a state the binder rejects, which the representative
//! state never should be.

use super::row::{Unmodelled, UnmodelledKind};
use metal2vulkan::bind::{BindState, Dispatch, DispatchRegion, ImageblockSize, StageInRegion};
use metal2vulkan::cache::VariantKey;
use metal2vulkan::device::Device;
use metal2vulkan::error::BindError;
use metal2vulkan::pipeline::{ComputePipeline, RuntimeFact};

/// How the driver is asked to create a kernel row's compute pipeline: the representative
/// dispatch and the binder's region of it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComputeCreation {
    /// One threadgroup of `threads_per_group` `[w, 1, 1]` (the module docs).
    pub dispatch: Dispatch,
    /// The binder's region of `dispatch`: its specialization (the plan's constants, the local
    /// size, `MultiRegion`), stage flags, and required subgroup size.
    pub region: DispatchRegion,
    /// The representative imageblock extent the dispatch sets, for a kernel with an explicit
    /// imageblock (the module docs); `None` for any other.
    pub imageblock: Option<ImageblockSize>,
}

/// Why a kernel row has no [`ComputeCreation`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NoCreation {
    /// The representative dispatch does not reach the row: a modelled limit of the tool, of kind
    /// [`UnmodelledKind::Dispatch`].
    Unmodelled(Unmodelled),
    /// The binder's error for the representative dispatch.
    Binder(BinderAnswer),
}

/// The binder's error for a kernel row's representative dispatch (the module docs).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BinderAnswer {
    /// A typed refusal the product documents (`BindError::Unsupported`), which the sweep counts.
    /// Any other binder error fails the sweep.
    pub documented: bool,
    /// The step and the error.
    pub detail: String,
}

impl BinderAnswer {
    fn of(error: &BindError, step: String) -> BinderAnswer {
        BinderAnswer {
            documented: matches!(error, BindError::Unsupported(_)),
            detail: format!("{step}: {error}"),
        }
    }
}

/// The [`ComputeCreation`] of `pipeline`, a kernel row compiled for `device` with no variant (the
/// module docs).
pub fn plan_compute_row(
    device: &Device,
    pipeline: &ComputePipeline,
) -> Result<ComputeCreation, NoCreation> {
    let unmodelled = |detail: String| {
        NoCreation::Unmodelled(Unmodelled {
            kind: UnmodelledKind::Dispatch,
            detail,
        })
    };
    let binder =
        |error: BindError, step: String| NoCreation::Binder(BinderAnswer::of(&error, step));
    let info = &pipeline.metal_info;
    let imageblock = pipeline
        .dependencies
        .iter()
        .any(|dependency| matches!(dependency.fact, RuntimeFact::ImageblockSize { .. }));
    for width in widths(
        info.thread_execution_width,
        info.max_total_threads_per_threadgroup,
    ) {
        let mut state = BindState::default();
        let extent = imageblock.then(|| ImageblockSize::new(width, 1));
        if let Some(extent) = extent {
            state.imageblock_size = extent;
        }
        let plan = pipeline
            .plan(device, &state.shape())
            .map_err(|error| binder(error, format!("plan for a threadgroup of {width}")))?;
        if plan.variant != VariantKey::new() {
            return Err(unmodelled(format!(
                "the binder plans a threadgroup of {width} on variant {:?}, not the row's",
                plan.variant
            )));
        }
        let dispatch = Dispatch::Threadgroups {
            groups: [1, 1, 1],
            threads_per_group: [width, 1, 1],
        };
        let variant = plan
            .dispatch_variant(&dispatch)
            .map_err(|error| binder(error, format!("the variant of a threadgroup of {width}")))?;
        // A threadgroup that is not whole subgroups of the tier runs on the emulated variant.
        if variant != plan.variant {
            continue;
        }
        state.stage_in_region = Some(StageInRegion::new([0; 3], [width, 1, 1]));
        let commands = plan
            .instantiate_dispatch(&state, &dispatch)
            .map_err(|error| binder(error, format!("instantiate a threadgroup of {width}")))?;
        let region = commands.regions.into_iter().next().ok_or_else(|| {
            unmodelled(format!(
                "the binder plans no region for one threadgroup of {width}"
            ))
        })?;
        return Ok(ComputeCreation {
            dispatch,
            region,
            imageblock: extent,
        });
    }
    Err(unmodelled(format!(
        "no threadgroup of whole simdgroups within the pipeline's {} threads runs on the row's \
         variant",
        info.max_total_threads_per_threadgroup
    )))
}

/// The threadgroup widths tried, fewest first: the thread execution width `width` (or `max`, the
/// most threads a threadgroup may have, when it is smaller), then each multiple of `width` up to
/// `max`.
fn widths(width: u32, max: u32) -> impl Iterator<Item = u32> {
    let width = width.max(1);
    std::iter::once(width.min(max).max(1)).chain(
        (2..)
            .map_while(move |k: u32| k.checked_mul(width))
            .take_while(move |&w| w <= max),
    )
}

#[cfg(test)]
mod tests {
    use super::{widths, BinderAnswer};
    use crate::acceptance::{compile_row, Projection, RowCompile, RowPipeline, RowStage};
    use metal2vulkan::bind::Dispatch;
    use metal2vulkan::pipeline::{Axis, SpecKey};
    use metal2vulkan::vk::PipelineShaderStageCreateFlags;
    use metal2vulkan_devtools::{engine_dir, llvm_assemble, AssembleError};

    /// A kernel whose simd tier requires full subgroups on `moltenvk-m3` keeps the flag, and is
    /// created from the binder's region of one threadgroup of 32 threads, not at its stage's
    /// placeholder local size of 1, with which the flag is
    /// `VUID-VkPipelineShaderStageCreateInfo-flags-02759`.
    #[test]
    fn a_full_subgroup_kernel_row_is_created_from_a_region_of_whole_subgroups() {
        let bitcode = match llvm_assemble(&engine_dir().join("simd_ops.ll")) {
            Ok(bitcode) => bitcode,
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                return;
            }
            Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected simd_ops.ll: {why}"),
        };
        let device = crate::engine::load_device("moltenvk-m3").unwrap();
        let compiled = compile_row(
            &bitcode,
            "simd_shuffle_k",
            RowStage::Kernel,
            Projection::Derived,
            &device,
            None,
        );
        let RowCompile::Compiled(row) = compiled else {
            panic!("simd_shuffle_k: {compiled:?}");
        };
        let RowPipeline::Compute { pipeline, creation } = *row else {
            panic!("a kernel row is a compute pipeline");
        };
        let full = PipelineShaderStageCreateFlags::REQUIRE_FULL_SUBGROUPS;
        assert!(pipeline.stage.stage_flags.contains(full));
        assert!(creation.region.stage_flags.contains(full));
        assert_eq!(
            creation.dispatch,
            Dispatch::Threadgroups {
                groups: [1, 1, 1],
                threads_per_group: [32, 1, 1],
            }
        );
        // Every local-size constant: its placeholder default, and its value in the region, where
        // the binder clears the bits the module never reads. The flattened tier's workgroup size is
        // `(FlatLocalSize, 1, 1)`, which the module reads whole.
        let mut local = Vec::new();
        for constant in &pipeline.stage.specialization.constants {
            let size = match constant.key {
                SpecKey::FlatLocalSize => 32,
                SpecKey::LocalSize(Axis::X) => 32,
                SpecKey::LocalSize(_) => 1,
                _ => continue,
            };
            if constant.key == SpecKey::FlatLocalSize {
                assert_eq!(constant.unread_bits, 0, "the workgroup size is read whole");
            }
            let size = size & !constant.unread_bits;
            assert_eq!(constant.default, 1, "{:?}", constant.key);
            let specialization = &creation.region.specialization;
            let entry = specialization
                .entries
                .iter()
                .find(|entry| entry.constant_id == constant.id)
                .unwrap_or_else(|| panic!("the region does not specialize {:?}", constant.key));
            let at = entry.offset as usize;
            let bytes = &specialization.data[at..at + entry.size as usize];
            let mut value = [0u8; 8];
            value[..bytes.len()].copy_from_slice(bytes);
            assert_eq!(u64::from_le_bytes(value), size, "{:?}", constant.key);
            local.push(constant.key);
        }
        assert!(local.contains(&SpecKey::FlatLocalSize), "{local:?}");
    }

    /// A kernel with an explicit imageblock, whose extent Metal's encoder must set, is created from
    /// the region of one threadgroup of 32 threads with an imageblock of one cell per thread, 32 by
    /// 1, not refused for the encoder's default of 0 by 0.
    #[test]
    fn an_imageblock_kernel_row_is_created_with_one_cell_per_thread() {
        let bitcode = match llvm_assemble(&engine_dir().join("a4_imageblock_extent.ll")) {
            Ok(bitcode) => bitcode,
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                return;
            }
            Err(AssembleError::Rejected(why)) => {
                panic!("llvm-as rejected a4_imageblock_extent.ll: {why}")
            }
        };
        let device = crate::engine::load_device("moltenvk-m3").unwrap();
        let compiled = compile_row(
            &bitcode,
            "a4_imageblock_extent",
            RowStage::Kernel,
            Projection::Derived,
            &device,
            None,
        );
        let RowCompile::Compiled(row) = compiled else {
            panic!("a4_imageblock_extent: {compiled:?}");
        };
        let RowPipeline::Compute { pipeline, creation } = *row else {
            panic!("a kernel row is a compute pipeline");
        };
        let Dispatch::Threadgroups {
            threads_per_group, ..
        } = creation.dispatch
        else {
            panic!("{:?}", creation.dispatch);
        };
        let width = threads_per_group[0];
        assert_eq!(
            creation.imageblock,
            Some(metal2vulkan::bind::ImageblockSize::new(width, 1))
        );
        let mut extent = Vec::new();
        for constant in &pipeline.stage.specialization.constants {
            let want = match constant.key {
                SpecKey::ImageblockWidth => u64::from(width),
                SpecKey::ImageblockHeight => 1,
                _ => continue,
            };
            let specialization = &creation.region.specialization;
            let entry = specialization
                .entries
                .iter()
                .find(|entry| entry.constant_id == constant.id)
                .unwrap_or_else(|| panic!("the region does not specialize {:?}", constant.key));
            let at = entry.offset as usize;
            let bytes = &specialization.data[at..at + entry.size as usize];
            let mut value = [0u8; 8];
            value[..bytes.len()].copy_from_slice(bytes);
            assert_eq!(u64::from_le_bytes(value), want, "{:?}", constant.key);
            extent.push(constant.key);
        }
        assert_eq!(
            extent.len(),
            2,
            "the stage specializes the imageblock's extent: {extent:?}"
        );
    }

    /// A binder error for the representative dispatch is a documented refusal only when it is the
    /// product's typed `Unsupported`; a binder defect or a rejected state is not, so it fails the
    /// sweep instead of counting as a tool limit.
    #[test]
    fn only_a_typed_unsupported_binder_error_is_a_documented_refusal() {
        use metal2vulkan::error::{BindError, Feature, InternalError, InternalKind, Unsupported};
        let refused = BindError::Unsupported(Unsupported::not_implemented(Feature::Imageblocks));
        let answer = BinderAnswer::of(&refused, "plan".into());
        assert!(answer.documented, "{answer:?}");
        assert!(answer.detail.starts_with("plan: "), "{answer:?}");
        for error in [
            BindError::Internal(InternalError::new(
                InternalKind::Invariant,
                "a binder defect",
            )),
            BindError::InvalidImageblockSize {
                width: 0,
                height: 0,
            },
            BindError::MissingStageInRegion,
        ] {
            assert!(
                !BinderAnswer::of(&error, "plan".into()).documented,
                "{error:?}"
            );
        }
    }

    #[test]
    fn threadgroup_widths_are_whole_simdgroups_within_the_pipelines_most_threads() {
        assert_eq!(widths(32, 128).collect::<Vec<_>>(), [32, 64, 96, 128]);
        assert_eq!(widths(32, 32).collect::<Vec<_>>(), [32]);
        // A pipeline that allows fewer threads than one simdgroup tries those.
        assert_eq!(widths(32, 16).collect::<Vec<_>>(), [16]);
        assert_eq!(widths(32, 0).collect::<Vec<_>>(), [1]);
    }
}
