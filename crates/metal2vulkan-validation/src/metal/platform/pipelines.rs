//! Native compute/render pipeline construction and indirect pipeline registration.

use super::libraries::companion_air;
use super::*;
use crate::check::interface::AirInterface;
use crate::companion::Companion;
use crate::executor_contract::{self, stage_inputs};
use crate::metal::MetalRunError;
use crate::observation::ReferenceGap;
use metal2vulkan_air::abi::ParamRole;
use objc2_foundation::NSError;
use objc2_metal::{MTLArgumentEncoder, MTLAttributeFormat, MTLGPUFamily};

/// The SDK's `MTLAttributeFormat` of an authored stage-input format: the one authored-to-Metal
/// format table the engine candidate compiles under too (`stage_inputs::attribute_format`),
/// whose Metal enum carries the SDK's values.
fn stage_input_format(format: crate::case::AttributeFormat) -> MTLAttributeFormat {
    MTLAttributeFormat(stage_inputs::attribute_format(format).as_raw() as usize)
}

/// The SDK values of a render pipeline's tessellation fields: the product's Metal enums carry
/// the SDK's values (`regen-metal-enums`), so each is the SDK enum of the same raw value.
struct MetalTessellation {
    partition_mode: MTLTessellationPartitionMode,
    max_factor: usize,
    control_point_index_type: MTLTessellationControlPointIndexType,
    factor_format: MTLTessellationFactorFormat,
    factor_step_function: MTLTessellationFactorStepFunction,
    output_winding: MTLWinding,
    factor_scale_enabled: bool,
}

impl MetalTessellation {
    fn of(descriptor: &metal2vulkan::pipeline::TessellationDescriptor) -> MetalTessellation {
        MetalTessellation {
            partition_mode: MTLTessellationPartitionMode(
                descriptor.partition_mode.as_raw() as usize
            ),
            max_factor: descriptor.max_tessellation_factor as usize,
            control_point_index_type: MTLTessellationControlPointIndexType(
                descriptor.control_point_index_type.as_raw() as usize,
            ),
            factor_format: MTLTessellationFactorFormat(descriptor.factor_format.as_raw() as usize),
            factor_step_function: MTLTessellationFactorStepFunction(
                descriptor.factor_step_function.as_raw() as usize,
            ),
            output_winding: MTLWinding(descriptor.output_winding_order.as_raw() as usize),
            factor_scale_enabled: descriptor.factor_scale_enabled,
        }
    }
}

/// The SDK value of a render encoder's cull mode, which the product's Metal enum carries.
pub(super) fn metal_cull_mode(mode: metal2vulkan::metal::CullMode) -> MTLCullMode {
    MTLCullMode(mode.as_raw() as usize)
}

pub(super) enum NativePipeline {
    Compute(Retained<ProtocolObject<dyn MTLComputePipelineState>>),
    Render(Retained<ProtocolObject<dyn MTLRenderPipelineState>>),
}

impl NativePipeline {
    /// Whether an argument-buffer member of AIR role `role` holds this kind of pipeline.
    pub(super) fn fits(&self, role: &ParamRole) -> bool {
        matches!(
            (self, role),
            (Self::Compute(_), ParamRole::ComputePipelineState(_))
                | (Self::Render(_), ParamRole::RenderPipelineState(_))
        )
    }

    pub(super) fn supports_indirect_commands(&self) -> bool {
        match self {
            Self::Compute(pipeline) => pipeline.supportIndirectCommandBuffers(),
            Self::Render(pipeline) => pipeline.supportIndirectCommandBuffers(),
        }
    }
}
/// The library of a generated `companion`, loaded from its typed AIR as a case's own library is
/// ([`companion_air`], [`load_library`]).
pub(super) fn companion_library(
    device: &ProtocolObject<dyn MTLDevice>,
    companion: &Companion,
) -> Result<Library, String> {
    load_library(device, &companion_air(companion)?, None)
}

/// The function of a generated `companion` of one function, from its library
/// ([`companion_library`]).
fn companion_function(
    device: &ProtocolObject<dyn MTLDevice>,
    companion: &Companion,
) -> Result<Function, String> {
    companion_library(device, companion)?
        .newFunctionWithName(&NSString::from_str(companion.entry()))
        .ok_or_else(|| format!("the generated companion has no entry {}", companion.entry()))
}

/// The vertex and fragment functions of `case`'s render pipeline: its own `function` alone for a
/// rasterization-disabled vertex case, else with its generated `companion` loaded from the
/// companion's typed AIR as the case's own library is ([`companion_air`], [`load_library`]), in
/// stage order.
pub(super) fn make_render_functions(
    device: &ProtocolObject<dyn MTLDevice>,
    case: &AuthoredCase,
    companion: Option<&Companion>,
    function: Function,
) -> Result<(Function, Option<Function>), String> {
    let generated = |companion: &Companion| companion_function(device, companion);
    match (case.stage, case.vertex_observation, companion) {
        (crate::case::Stage::Vertex, None, None) => Ok((function, None)),
        (crate::case::Stage::Vertex, Some(_), Some(observer @ Companion::ObserverFragment(_))) => {
            Ok((function, Some(generated(observer)?)))
        }
        (crate::case::Stage::Fragment, _, Some(passthrough @ Companion::PassthroughVertex(_))) => {
            Ok((generated(passthrough)?, Some(function)))
        }
        (stage, observation, companion) => Err(format!(
            "a {stage:?} case with vertex observation {observation:?} has companion {:?}",
            companion.map(Companion::entry)
        )),
    }
}

pub(super) fn make_render_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    case: &AuthoredCase,
    functions: &(Function, Option<Function>),
    linked: &MetalLinkedFunctions,
    layout: Option<&MTLVertexDescriptor>,
    indirect: bool,
) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>, MetalRunError> {
    let descriptor = MTLRenderPipelineDescriptor::new();
    descriptor.setVertexFunction(Some(&functions.0));
    descriptor.setFragmentFunction(functions.1.as_deref());
    descriptor.setSupportIndirectCommandBuffers(indirect);
    descriptor.setRasterizationEnabled(functions.1.is_some());
    descriptor.setVertexDescriptor(layout);
    if let Some(draw) = &case.draw {
        unsafe {
            descriptor.setInputPrimitiveTopology(metal_primitive_topology_class(draw.primitive));
        }
    }
    // Only a multisampled pass sets it: a pass of one sample keeps the descriptor's default, as
    // every case drew before.
    if let Some(samples) = pipeline_raster_sample_count(case) {
        descriptor.setRasterSampleCount(samples);
    }
    if let Some(linked) = &linked.descriptor {
        match case.stage {
            crate::case::Stage::Vertex => descriptor.setVertexLinkedFunctions(Some(linked)),
            crate::case::Stage::Fragment => descriptor.setFragmentLinkedFunctions(Some(linked)),
            crate::case::Stage::Kernel => {
                return Err("compute shader cannot form a render pipeline".into())
            }
            // An object or mesh case draws a mesh render pipeline of its own (`mesh.rs`).
            crate::case::Stage::Object | crate::case::Stage::Mesh => {
                return Err("an object or mesh function forms no vertex render pipeline".into())
            }
        }
    }
    if let Some(tessellation) = &case.tessellation {
        // The tessellation state the engine candidate compiles too.
        let state =
            MetalTessellation::of(&executor_contract::tessellation::descriptor(tessellation));
        unsafe {
            descriptor.setTessellationPartitionMode(state.partition_mode);
            descriptor.setMaxTessellationFactor(state.max_factor);
            descriptor.setTessellationControlPointIndexType(state.control_point_index_type);
            descriptor.setMaxVertexAmplificationCount(tessellation.amplification_count as usize);
        }
        descriptor.setTessellationFactorFormat(state.factor_format);
        descriptor.setTessellationFactorStepFunction(state.factor_step_function);
        descriptor.setTessellationOutputWindingOrder(state.output_winding);
        descriptor.setTessellationFactorScaleEnabled(state.factor_scale_enabled);
    }
    let attachments = descriptor.colorAttachments();
    for target in &case.render_targets {
        let attachment = unsafe { attachments.objectAtIndexedSubscript(target.index as usize) };
        attachment.setPixelFormat(metal_pixel_format(target.format));
    }
    if case.is_rasterization_disabled_vertex() {
        let attachment = unsafe { attachments.objectAtIndexedSubscript(0) };
        attachment.setPixelFormat(MTLPixelFormat::R8Unorm);
    }
    if case.stage == crate::case::Stage::Fragment {
        if let Some(depth_stencil) = &case.depth_stencil {
            if depth_stencil.initial_depth_b64.is_some() {
                descriptor.setDepthAttachmentPixelFormat(MTLPixelFormat::Depth32Float);
            }
            if depth_stencil.initial_stencil_b64.is_some() {
                descriptor.setStencilAttachmentPixelFormat(MTLPixelFormat::Stencil8);
            }
        }
    }
    compile::bounded("render pipeline creation", || {
        device.newRenderPipelineStateWithDescriptor_error(&descriptor)
    })
    .map_err(|error| pipeline_failure(device, "create render pipeline", &error))
}

/// Why Metal created no pipeline: a reference gap when its message names exactly one Apple GPU
/// family (`GPUFamily<N>`, [`ReferenceGap::named_family`]) and this device does not support
/// `MTLGPUFamilyApple<N>`; any other refusal fails the run as `what`.
pub(super) fn pipeline_failure(
    device: &ProtocolObject<dyn MTLDevice>,
    what: &str,
    error: &NSError,
) -> MetalRunError {
    let message = error.localizedDescription().to_string();
    match ReferenceGap::named_family(&message) {
        Some(apple_family) if !supports_apple_family(device, apple_family) => {
            MetalRunError::ReferenceGap(ReferenceGap {
                apple_family,
                message,
            })
        }
        _ => MetalRunError::Failed(format!("{what}: {message}")),
    }
}

/// Whether `device` supports `MTLGPUFamilyApple<family>`, whose raw value is 1000 + `family`.
fn supports_apple_family(device: &ProtocolObject<dyn MTLDevice>, family: u32) -> bool {
    isize::try_from(family)
        .ok()
        .and_then(|family| family.checked_add(APPLE_FAMILY_BASE))
        .is_some_and(|raw| device.supportsFamily(MTLGPUFamily(raw)))
}

/// `MTLGPUFamilyApple1`'s raw value less one: `MTLGPUFamilyApple<N>` is this plus `N`.
const APPLE_FAMILY_BASE: isize = MTLGPUFamily::Apple1.0 - 1;

/// The `rasterSampleCount` the oracle sets on `case`'s render pipeline: the case's count of a
/// multisampled pass, and none for one sample, which leaves Metal's default (1).
pub(super) fn pipeline_raster_sample_count(case: &AuthoredCase) -> Option<usize> {
    case.raster_sample_count
        .filter(|&samples| samples > 1)
        .map(|samples| samples as usize)
}

/// Compile each pipeline template from the AIR the oracle compiles for it, as compiled.
pub(super) fn make_pipeline_templates(
    device: &ProtocolObject<dyn MTLDevice>,
    templates: &BTreeMap<String, OracleTemplate<'_>>,
) -> Result<BTreeMap<String, NativePipeline>, MetalRunError> {
    templates
        .iter()
        .map(|(id, oracle)| {
            let template = oracle.shader;
            if !template
                .reflection
                .implicit_imageblock_attachments
                .is_empty()
            {
                return Err("native indirect tile pipeline registration is not implemented".into());
            }
            let constants = crate::literal::function_constants(&template.case)?;
            let library = load_library(device, &oracle.air, None)?;
            let function = make_function(
                &library,
                &NSString::from_str(&template.case.entry),
                &constants,
            )?;
            let linked = load_linked_functions(device, &template.linked_functions, &constants)?;
            let pipeline = match template.case.stage {
                crate::case::Stage::Kernel => NativePipeline::Compute(make_compute_pipeline(
                    device,
                    &function,
                    &template.case.kernel_stage_inputs,
                    &template.interface,
                    &linked,
                    true,
                )?),
                crate::case::Stage::Vertex | crate::case::Stage::Fragment => {
                    let functions = make_render_functions(
                        device,
                        &template.case,
                        template.companion.as_ref(),
                        function,
                    )?;
                    let layout = (template.case.stage == crate::case::Stage::Vertex)
                        .then(|| make_vertex_layout(&template.case, &template.interface))
                        .transpose()?;
                    NativePipeline::Render(make_render_pipeline(
                        device,
                        &template.case,
                        &functions,
                        &linked,
                        layout.as_ref().map(|layout| &*layout.descriptor),
                        true,
                    )?)
                }
                crate::case::Stage::Object | crate::case::Stage::Mesh => {
                    NativePipeline::Render(super::mesh::make_template_pipeline(
                        device,
                        &template.case,
                        (&library, &function),
                        &constants,
                        template.companion.as_ref(),
                    )?)
                }
            };
            if !pipeline.supports_indirect_commands() {
                return Err("compiled template does not support indirect commands".into());
            }
            Ok((id.clone(), pipeline))
        })
        .collect()
}

pub(super) fn encode_pipeline_templates(
    function: &ProtocolObject<dyn MTLFunction>,
    buffers: &[(u32, Buffer)],
    resources: &[crate::case::indirect_commands::PipelineResource],
    interface: &AirInterface,
    pipelines: &BTreeMap<String, NativePipeline>,
) -> Result<(), String> {
    let writes = resources
        .iter()
        .map(|resource| {
            let pipeline = pipelines.get(&resource.case_id).ok_or_else(|| {
                format!("missing compiled pipeline template {}", resource.case_id)
            })?;
            let (index, ()) = argument_member(
                interface,
                resource.buffer_binding,
                resource.field_offset,
                "native pipeline handle",
                |role| pipeline.fits(role).then_some(()),
            )?;
            let encoder = argument_encoder(function, buffers, resource.buffer_binding)?;
            Ok((encoder, index, pipeline))
        })
        .collect::<Result<Vec<_>, String>>()?;
    for (encoder, index, pipeline) in writes {
        // The AIR layout supplies the native argument index, not the byte offset.
        // The registry retains the pipeline through producer completion.
        unsafe {
            match pipeline {
                NativePipeline::Compute(pipeline) => {
                    encoder.setComputePipelineState_atIndex(Some(pipeline), index as usize)
                }
                NativePipeline::Render(pipeline) => {
                    encoder.setRenderPipelineState_atIndex(Some(pipeline), index as usize)
                }
            }
        };
    }
    Ok(())
}

pub(super) fn make_compute_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    function: &ProtocolObject<dyn MTLFunction>,
    stage_inputs: &[crate::case::KernelStageInput],
    interface: &AirInterface,
    linked: &MetalLinkedFunctions,
    indirect: bool,
) -> Result<Retained<ProtocolObject<dyn objc2_metal::MTLComputePipelineState>>, MetalRunError> {
    if !indirect && stage_inputs.is_empty() && linked.descriptor.is_none() {
        return compile::bounded("compute pipeline creation", || {
            device.newComputePipelineStateWithFunction_error(function)
        })
        .map_err(|error| pipeline_failure(device, "create compute pipeline", &error));
    }
    let descriptor = MTLComputePipelineDescriptor::new();
    descriptor.setComputeFunction(Some(function));
    descriptor.setSupportIndirectCommandBuffers(indirect);
    if !stage_inputs.is_empty() {
        // The fetch rule the engine candidate compiles under too.
        let stage = MTLStageInputOutputDescriptor::stageInputOutputDescriptor();
        let attributes = stage.attributes();
        let layouts = stage.layouts();
        for fetch in stage_inputs::authored(stage_inputs, interface)? {
            let slot = fetch.slot as usize;
            let attribute = unsafe { attributes.objectAtIndexedSubscript(fetch.location as usize) };
            attribute.setFormat(stage_input_format(fetch.format));
            attribute.setOffset(stage_inputs::OFFSET as usize);
            unsafe { attribute.setBufferIndex(slot) };
            let layout = unsafe { layouts.objectAtIndexedSubscript(slot) };
            // `MTLBufferLayoutStrideDynamic` is `NSUIntegerMax`: the fetch reads the stride the
            // buffer is bound with (`stage_inputs::bound_strides`).
            layout.setStride(if fetch.dynamic {
                usize::MAX
            } else {
                fetch.stride as usize
            });
            // The product's Metal enums carry the SDK's values.
            layout.setStepFunction(MTLStepFunction(
                stage_inputs::STEP_FUNCTION.as_raw() as usize
            ));
            layout.setStepRate(stage_inputs::STEP_RATE as usize);
        }
        descriptor.setStageInputDescriptor(Some(&stage));
    }
    if let Some(linked) = &linked.descriptor {
        descriptor.setLinkedFunctions(Some(linked));
    }
    compile::bounded("compute pipeline creation", || {
        device.newComputePipelineStateWithDescriptor_options_reflection_error(
            &descriptor,
            MTLPipelineOption::None,
            None,
        )
    })
    .map_err(|error| pipeline_failure(device, "create described compute pipeline", &error))
}

#[cfg(test)]
mod tests {
    use super::super::vertex_inputs::metal_vertex_format;
    use super::{
        metal_cull_mode, stage_input_format, MTLCullMode, MTLTessellationControlPointIndexType,
        MTLTessellationFactorFormat, MTLTessellationFactorStepFunction,
        MTLTessellationPartitionMode, MTLWinding, MetalTessellation,
    };
    use crate::case::AttributeFormat;

    /// Every authored stage-input format reaches Metal as the SDK's value of the format of that
    /// name, which `MTLVertexFormat` shares.
    #[test]
    fn stage_input_formats_are_the_sdks() {
        let mut formats = 0;
        for scalar in [
            "char", "uchar", "short", "ushort", "half", "float", "uint", "int",
        ] {
            for lanes in ["", "2", "3", "4"] {
                let name = format!("{scalar}{lanes}");
                let format = AttributeFormat::from_air_type_name(&name).expect(&name);
                formats += 1;
                assert_eq!(
                    stage_input_format(format).0,
                    metal_vertex_format(format).0,
                    "{format:?}"
                );
            }
        }
        assert_eq!(formats, 32);
    }

    /// The shared tessellation state reaches Metal as the SDK values the oracle set before the
    /// state was shared: integer partitioning, a maximum factor of 64, no control-point index
    /// buffer, half factors read per patch, counter-clockwise output, and no factor scale, for a
    /// draw that states nothing; each authored partition mode, winding, and cull mode is the SDK
    /// value of its name.
    #[test]
    fn the_shared_tessellation_state_is_the_oracles() {
        use crate::case::{CullMode, TessellationDraw, TessellationPartition, TessellationWinding};
        use crate::executor_contract::tessellation::{cull_mode, descriptor};
        let mut draw = TessellationDraw {
            factors: vec![],
            instance_count: 1,
            amplification_count: 1,
            control_points: vec![],
            patch_inputs: vec![],
            partition_mode: TessellationPartition::default(),
            output_winding: TessellationWinding::default(),
            cull_mode: CullMode::default(),
        };
        let state = MetalTessellation::of(&descriptor(&draw));
        assert_eq!(state.partition_mode, MTLTessellationPartitionMode::Integer);
        assert_eq!(state.max_factor, 64);
        assert_eq!(
            state.control_point_index_type,
            MTLTessellationControlPointIndexType::None
        );
        assert_eq!(state.factor_format, MTLTessellationFactorFormat::Half);
        assert_eq!(
            state.factor_step_function,
            MTLTessellationFactorStepFunction::PerPatch
        );
        assert_eq!(state.output_winding, MTLWinding::CounterClockwise);
        assert!(!state.factor_scale_enabled);
        assert_eq!(metal_cull_mode(cull_mode(&draw)), MTLCullMode::None);

        for (authored, sdk) in [
            (
                TessellationPartition::Pow2,
                MTLTessellationPartitionMode::Pow2,
            ),
            (
                TessellationPartition::Integer,
                MTLTessellationPartitionMode::Integer,
            ),
            (
                TessellationPartition::FractionalOdd,
                MTLTessellationPartitionMode::FractionalOdd,
            ),
            (
                TessellationPartition::FractionalEven,
                MTLTessellationPartitionMode::FractionalEven,
            ),
        ] {
            draw.partition_mode = authored;
            assert_eq!(
                MetalTessellation::of(&descriptor(&draw)).partition_mode,
                sdk
            );
        }
        for (authored, sdk) in [
            (TessellationWinding::Clockwise, MTLWinding::Clockwise),
            (
                TessellationWinding::CounterClockwise,
                MTLWinding::CounterClockwise,
            ),
        ] {
            draw.output_winding = authored;
            assert_eq!(
                MetalTessellation::of(&descriptor(&draw)).output_winding,
                sdk
            );
        }
        for (authored, sdk) in [
            (CullMode::None, MTLCullMode::None),
            (CullMode::Front, MTLCullMode::Front),
            (CullMode::Back, MTLCullMode::Back),
        ] {
            draw.cull_mode = authored;
            assert_eq!(metal_cull_mode(cull_mode(&draw)), sdk);
        }
    }
}
