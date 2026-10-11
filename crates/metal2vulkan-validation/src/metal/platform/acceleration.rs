//! Native acceleration-structure construction and build submission.

use super::{commit_and_wait, new_buffer_from_slice};
use crate::case::{AccelerationStructureKind, AuthoredCase};
use base64::Engine as _;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSArray;
use objc2_metal::{
    MTLAccelerationStructure, MTLAccelerationStructureCommandEncoder,
    MTLAccelerationStructureDescriptor, MTLAccelerationStructureGeometryDescriptor,
    MTLAccelerationStructureInstanceDescriptor, MTLAccelerationStructureInstanceOptions,
    MTLAccelerationStructureTriangleGeometryDescriptor, MTLCommandBuffer, MTLCommandEncoder,
    MTLCommandQueue, MTLDevice, MTLInstanceAccelerationStructureDescriptor, MTLPackedFloat3,
    MTLPackedFloat4x3, MTLPrimitiveAccelerationStructureDescriptor, MTLResourceOptions,
};

pub(super) type AccelerationStructure = Retained<ProtocolObject<dyn MTLAccelerationStructure>>;

pub(super) fn make_acceleration_structures(
    device: &ProtocolObject<dyn MTLDevice>,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    case: &AuthoredCase,
) -> Result<Vec<(u32, AccelerationStructure)>, String> {
    if case.acceleration_structures.is_empty() {
        return Ok(Vec::new());
    }
    if !device.supportsRaytracing() {
        return Err("Metal device does not support acceleration structures".into());
    }

    let canonical_primitive = build_primitive_acceleration_structure(
        device,
        queue,
        &AccelerationStructureKind::INSTANCED_TRIANGLE,
        "canonical triangle",
    )?;

    case.acceleration_structures
        .iter()
        .map(|resource| {
            let acceleration_structure = match resource.kind {
                AccelerationStructureKind::Instance => {
                    let descriptors = resource
                        .child_references
                        .iter()
                        .map(|_| MTLAccelerationStructureInstanceDescriptor {
                            transformationMatrix: identity_transform(),
                            options: MTLAccelerationStructureInstanceOptions::Opaque,
                            mask: u32::MAX,
                            intersectionFunctionTableOffset: 0,
                            accelerationStructureIndex: 0,
                        })
                        .collect::<Vec<_>>();
                    let instance_buffer = new_buffer_from_slice(
                        device,
                        &descriptors,
                        &format!("acceleration structure {} instances", resource.binding),
                    )?;
                    let primitive_array = NSArray::from_slice(&[&*canonical_primitive]);
                    let descriptor = MTLInstanceAccelerationStructureDescriptor::descriptor();
                    descriptor.setInstancedAccelerationStructures(Some(&primitive_array));
                    descriptor.setInstanceDescriptorBuffer(Some(&*instance_buffer));
                    descriptor.setInstanceCount(descriptors.len());
                    let descriptor_base: &MTLAccelerationStructureDescriptor = &descriptor;
                    build_acceleration_structure(device, queue, descriptor_base)?
                }
                AccelerationStructureKind::Primitive => {
                    let encoded = resource.primitive_triangles_b64.as_deref().ok_or_else(|| {
                        format!(
                            "primitive acceleration structure {} has no triangles",
                            resource.binding
                        )
                    })?;
                    let bytes = base64::engine::general_purpose::STANDARD
                        .decode(encoded)
                        .map_err(|error| {
                            format!(
                                "decode primitive acceleration structure {}: {error}",
                                resource.binding
                            )
                        })?;
                    let vertices = bytes
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|word| f32::from_le_bytes(*word))
                        .collect::<Vec<_>>();
                    build_primitive_acceleration_structure(
                        device,
                        queue,
                        &vertices,
                        &format!("primitive acceleration structure {}", resource.binding),
                    )?
                }
            };
            Ok((resource.binding, acceleration_structure))
        })
        .collect()
}

fn build_primitive_acceleration_structure(
    device: &ProtocolObject<dyn MTLDevice>,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    vertices: &[f32],
    label: &str,
) -> Result<AccelerationStructure, String> {
    let vertex_buffer = new_buffer_from_slice(device, vertices, label)?;
    let geometry = MTLAccelerationStructureTriangleGeometryDescriptor::descriptor();
    geometry.setVertexBuffer(Some(&*vertex_buffer));
    geometry.setVertexStride(3 * std::mem::size_of::<f32>());
    geometry.setTriangleCount(vertices.len() / 9);
    let geometry_base: &MTLAccelerationStructureGeometryDescriptor = &geometry;
    let geometries = NSArray::from_slice(&[geometry_base]);
    let descriptor = MTLPrimitiveAccelerationStructureDescriptor::descriptor();
    descriptor.setGeometryDescriptors(Some(&geometries));
    let descriptor_base: &MTLAccelerationStructureDescriptor = &descriptor;
    build_acceleration_structure(device, queue, descriptor_base)
}

fn build_acceleration_structure(
    device: &ProtocolObject<dyn MTLDevice>,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    descriptor: &MTLAccelerationStructureDescriptor,
) -> Result<AccelerationStructure, String> {
    let sizes = device.accelerationStructureSizesWithDescriptor(descriptor);
    let acceleration_structure = device
        .newAccelerationStructureWithSize(sizes.accelerationStructureSize)
        .ok_or_else(|| "create Metal acceleration structure".to_string())?;
    let scratch = device
        .newBufferWithLength_options(
            sizes.buildScratchBufferSize.max(1),
            MTLResourceOptions::StorageModePrivate,
        )
        .ok_or_else(|| "create Metal acceleration-structure scratch buffer".to_string())?;
    let command_buffer = queue
        .commandBuffer()
        .ok_or_else(|| "MTLCommandQueue::commandBuffer returned nil".to_string())?;
    let encoder = command_buffer
        .accelerationStructureCommandEncoder()
        .ok_or_else(|| "accelerationStructureCommandEncoder returned nil".to_string())?;
    encoder.buildAccelerationStructure_descriptor_scratchBuffer_scratchBufferOffset(
        &acceleration_structure,
        descriptor,
        &scratch,
        0,
    );
    encoder.endEncoding();
    commit_and_wait(&command_buffer, "Metal acceleration-structure build")?;
    Ok(acceleration_structure)
}

fn identity_transform() -> MTLPackedFloat4x3 {
    MTLPackedFloat4x3 {
        columns: [
            MTLPackedFloat3 {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            },
            MTLPackedFloat3 {
                x: 0.0,
                y: 1.0,
                z: 0.0,
            },
            MTLPackedFloat3 {
                x: 0.0,
                y: 0.0,
                z: 1.0,
            },
            MTLPackedFloat3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
        ],
    }
}
