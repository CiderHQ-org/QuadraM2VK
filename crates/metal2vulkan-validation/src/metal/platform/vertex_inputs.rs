//! Native stage-input layouts and vertex uploads.

use super::buffers::{upload_buffer, with_shadow_past};
use super::Buffer;
use crate::case::out_of_bounds::OutOfBoundsResource;
use crate::case::AuthoredCase;
use crate::executor_contract::buffer_slots::vertex_input_slots;
use crate::executor_contract::out_of_bounds::reach;
use crate::literal::LiteralResources;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLDevice, MTLVertexDescriptor, MTLVertexFormat, MTLVertexStepFunction};

pub(super) struct MetalVertexLayout {
    pub(super) descriptor: Retained<MTLVertexDescriptor>,
    pub(super) slots: Vec<(usize, u32)>,
}

pub(super) fn make_vertex_layout(
    case: &AuthoredCase,
    interface: &crate::check::interface::AirInterface,
) -> Result<MetalVertexLayout, String> {
    let buffer_slots = vertex_input_slots(case, interface)?;
    let descriptor = MTLVertexDescriptor::vertexDescriptor();
    let attributes = descriptor.attributes();
    let layouts = descriptor.layouts();
    let mut inputs = case
        .vertex_inputs
        .iter()
        .map(|input| (input, MTLVertexStepFunction::PerVertex))
        .chain(case.tessellation.iter().flat_map(|tessellation| {
            tessellation
                .control_points
                .iter()
                .map(|input| (input, MTLVertexStepFunction::PerPatchControlPoint))
                .chain(
                    tessellation
                        .patch_inputs
                        .iter()
                        .map(|input| (input, MTLVertexStepFunction::PerPatch)),
                )
        }))
        .collect::<Vec<_>>();
    inputs.sort_by_key(|(input, _)| input.location);
    let mut slots = Vec::with_capacity(inputs.len());
    for (input, step_function) in inputs {
        let slot = *buffer_slots
            .get(&input.location)
            .ok_or_else(|| format!("no buffer slot for vertex input {}", input.location))?
            as usize;
        let attribute = unsafe { attributes.objectAtIndexedSubscript(input.location as usize) };
        attribute.setFormat(metal_vertex_format(input.format));
        unsafe {
            attribute.setOffset(0);
            attribute.setBufferIndex(slot);
        }
        let layout = unsafe { layouts.objectAtIndexedSubscript(slot) };
        // `MTLBufferLayoutStrideDynamic` is `NSUIntegerMax`: the fetch reads the stride the
        // buffer is bound with ([`make_vertex_buffers`]).
        let stride = if input.dynamic_stride {
            usize::MAX
        } else {
            input.stride as usize
        };
        unsafe { layout.setStride(stride) };
        layout.setStepFunction(step_function);
        unsafe { layout.setStepRate(1) };
        slots.push((slot, input.location));
    }
    Ok(MetalVertexLayout { descriptor, slots })
}

/// The buffer of each vertex input and tessellation input at its slot of `layout`, an
/// out-of-bounds target's with its shadow past its data: Metal's vertex fetch checks nothing, so
/// every fetch past the data stays inside the allocation. With it, the stride a dynamic layout's
/// buffer is bound with (the input's, through the stride form), `None` for a fixed layout's.
pub(super) fn make_vertex_buffers(
    device: &ProtocolObject<dyn MTLDevice>,
    case: &AuthoredCase,
    resources: &LiteralResources,
    layout: &MetalVertexLayout,
) -> Result<Vec<(usize, Buffer, Option<u32>)>, String> {
    layout
        .slots
        .iter()
        .map(|&(slot, location)| {
            let input = resources
                .vertex_inputs
                .iter()
                .chain(
                    resources
                        .tessellation
                        .iter()
                        .flat_map(|t| t.control_points.iter().chain(&t.patch_inputs)),
                )
                .find(|input| input.location == location)
                .ok_or_else(|| format!("missing vertex input {location} for planned slot"))?;
            let label = format!("vertex input {}", input.location);
            let target = OutOfBoundsResource::AttributeBuffer { location };
            let bytes = with_shadow_past(&input.bytes, reach(case, target), &label)?;
            let stride = input.dynamic_stride.then_some(input.stride);
            Ok((slot, upload_buffer(device, &bytes, &label)?, stride))
        })
        .collect()
}

pub(super) fn metal_vertex_format(format: crate::case::AttributeFormat) -> MTLVertexFormat {
    use crate::case::AttributeFormat as F;
    match format {
        F::Char => MTLVertexFormat::Char,
        F::Char2 => MTLVertexFormat::Char2,
        F::Char3 => MTLVertexFormat::Char3,
        F::Char4 => MTLVertexFormat::Char4,
        F::Uchar => MTLVertexFormat::UChar,
        F::Uchar2 => MTLVertexFormat::UChar2,
        F::Uchar3 => MTLVertexFormat::UChar3,
        F::Uchar4 => MTLVertexFormat::UChar4,
        F::Short => MTLVertexFormat::Short,
        F::Short2 => MTLVertexFormat::Short2,
        F::Short3 => MTLVertexFormat::Short3,
        F::Short4 => MTLVertexFormat::Short4,
        F::Ushort => MTLVertexFormat::UShort,
        F::Ushort2 => MTLVertexFormat::UShort2,
        F::Ushort3 => MTLVertexFormat::UShort3,
        F::Ushort4 => MTLVertexFormat::UShort4,
        F::Half => MTLVertexFormat::Half,
        F::Half2 => MTLVertexFormat::Half2,
        F::Half3 => MTLVertexFormat::Half3,
        F::Half4 => MTLVertexFormat::Half4,
        F::Float => MTLVertexFormat::Float,
        F::Float2 => MTLVertexFormat::Float2,
        F::Float3 => MTLVertexFormat::Float3,
        F::Float4 => MTLVertexFormat::Float4,
        F::Uint => MTLVertexFormat::UInt,
        F::Uint2 => MTLVertexFormat::UInt2,
        F::Uint3 => MTLVertexFormat::UInt3,
        F::Uint4 => MTLVertexFormat::UInt4,
        F::Int => MTLVertexFormat::Int,
        F::Int2 => MTLVertexFormat::Int2,
        F::Int3 => MTLVertexFormat::Int3,
        F::Int4 => MTLVertexFormat::Int4,
    }
}
