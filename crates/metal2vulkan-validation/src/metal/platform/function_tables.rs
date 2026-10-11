//! Native visible/intersection function-table construction and encoder binding.

use super::libraries::{linked_function, MetalLinkedFunctions};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLComputePipelineState, MTLIntersectionFunctionSignature, MTLIntersectionFunctionTable,
    MTLIntersectionFunctionTableDescriptor, MTLRenderCommandEncoder, MTLRenderPipelineState,
    MTLRenderStages, MTLVisibleFunctionTable, MTLVisibleFunctionTableDescriptor,
};

type VisibleFunctionTable = Retained<ProtocolObject<dyn MTLVisibleFunctionTable>>;
type IntersectionFunctionTable = Retained<ProtocolObject<dyn MTLIntersectionFunctionTable>>;

#[derive(Default)]
pub(super) struct MetalFunctionTables {
    pub(super) visible: Vec<(u32, VisibleFunctionTable)>,
    pub(super) intersection: Vec<(
        crate::library_module::ResolvedIntersectionFunctionTableLocation,
        IntersectionFunctionTable,
    )>,
}

fn function_table_count(binding: u32, size: u32) -> Result<usize, String> {
    usize::try_from(size).map_err(|_| format!("function table binding {binding} is too large"))
}

fn metal_intersection_signature(
    flags: &[crate::case::IntersectionFunctionSignature],
) -> MTLIntersectionFunctionSignature {
    use crate::case::IntersectionFunctionSignature as Flag;
    flags
        .iter()
        .fold(MTLIntersectionFunctionSignature::None, |bits, flag| {
            bits | match flag {
                Flag::Instancing => MTLIntersectionFunctionSignature::Instancing,
                Flag::TriangleData => MTLIntersectionFunctionSignature::TriangleData,
                Flag::WorldSpaceData => MTLIntersectionFunctionSignature::WorldSpaceData,
                Flag::InstanceMotion => MTLIntersectionFunctionSignature::InstanceMotion,
                Flag::PrimitiveMotion => MTLIntersectionFunctionSignature::PrimitiveMotion,
                Flag::ExtendedLimits => MTLIntersectionFunctionSignature::ExtendedLimits,
                Flag::MaxLevels => MTLIntersectionFunctionSignature::MaxLevels,
                Flag::IntersectionFunctionBuffer => {
                    MTLIntersectionFunctionSignature::IntersectionFunctionBuffer
                }
                Flag::UserData => MTLIntersectionFunctionSignature::UserData,
            }
        })
}

pub(super) fn make_compute_function_tables(
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    tables: &crate::library_module::ResolvedLinkedFunctions,
    linked: &MetalLinkedFunctions,
) -> Result<MetalFunctionTables, String> {
    let mut result = MetalFunctionTables::default();
    for table in &tables.visible {
        let descriptor = MTLVisibleFunctionTableDescriptor::visibleFunctionTableDescriptor();
        unsafe { descriptor.setFunctionCount(function_table_count(table.binding, table.size)?) };
        let metal_table = pipeline
            .newVisibleFunctionTableWithDescriptor(&descriptor)
            .ok_or_else(|| {
                format!(
                    "create compute visible function table at binding {}",
                    table.binding
                )
            })?;
        for entry in &table.entries {
            let function = linked_function(linked, entry)?;
            let handle = pipeline
                .functionHandleWithFunction(function)
                .ok_or_else(|| format!("create handle for linked function {:?}", entry.function))?;
            unsafe { metal_table.setFunction_atIndex(Some(&handle), entry.index as usize) };
        }
        result.visible.push((table.binding, metal_table));
    }
    for table in &tables.intersection {
        let table_binding = table.location.buffer_binding();
        let descriptor =
            MTLIntersectionFunctionTableDescriptor::intersectionFunctionTableDescriptor();
        descriptor.setFunctionCount(function_table_count(table_binding, table.size)?);
        let metal_table = pipeline
            .newIntersectionFunctionTableWithDescriptor(&descriptor)
            .ok_or_else(|| {
                format!(
                    "create compute intersection function table at binding {}",
                    table_binding
                )
            })?;
        for entry in &table.entries {
            match entry {
                crate::library_module::ResolvedIntersectionFunctionEntry::Linked(entry) => {
                    let function = linked_function(linked, entry)?;
                    let handle =
                        pipeline
                            .functionHandleWithFunction(function)
                            .ok_or_else(|| {
                                format!("create handle for linked function {:?}", entry.function)
                            })?;
                    metal_table.setFunction_atIndex(Some(&handle), entry.index as usize);
                }
                crate::library_module::ResolvedIntersectionFunctionEntry::OpaqueTriangle {
                    index,
                    signature,
                } => unsafe {
                    metal_table.setOpaqueTriangleIntersectionFunctionWithSignature_atIndex(
                        metal_intersection_signature(signature),
                        *index as usize,
                    )
                },
            }
        }
        result.intersection.push((table.location, metal_table));
    }
    Ok(result)
}

pub(super) fn make_render_function_tables(
    pipeline: &ProtocolObject<dyn MTLRenderPipelineState>,
    stage: MTLRenderStages,
    tables: &crate::library_module::ResolvedLinkedFunctions,
    linked: &MetalLinkedFunctions,
) -> Result<MetalFunctionTables, String> {
    let mut result = MetalFunctionTables::default();
    for table in &tables.visible {
        let descriptor = MTLVisibleFunctionTableDescriptor::visibleFunctionTableDescriptor();
        unsafe { descriptor.setFunctionCount(function_table_count(table.binding, table.size)?) };
        let metal_table = pipeline
            .newVisibleFunctionTableWithDescriptor_stage(&descriptor, stage)
            .ok_or_else(|| {
                format!(
                    "create render visible function table at binding {}",
                    table.binding
                )
            })?;
        for entry in &table.entries {
            let function = linked_function(linked, entry)?;
            let handle = unsafe { pipeline.functionHandleWithFunction_stage(function, stage) }
                .ok_or_else(|| format!("create handle for linked function {:?}", entry.function))?;
            unsafe { metal_table.setFunction_atIndex(Some(&handle), entry.index as usize) };
        }
        result.visible.push((table.binding, metal_table));
    }
    for table in &tables.intersection {
        let table_binding = table.location.buffer_binding();
        let descriptor =
            MTLIntersectionFunctionTableDescriptor::intersectionFunctionTableDescriptor();
        descriptor.setFunctionCount(function_table_count(table_binding, table.size)?);
        let metal_table = pipeline
            .newIntersectionFunctionTableWithDescriptor_stage(&descriptor, stage)
            .ok_or_else(|| {
                format!(
                    "create render intersection function table at binding {}",
                    table_binding
                )
            })?;
        for entry in &table.entries {
            match entry {
                crate::library_module::ResolvedIntersectionFunctionEntry::Linked(entry) => {
                    let function = linked_function(linked, entry)?;
                    let handle =
                        unsafe { pipeline.functionHandleWithFunction_stage(function, stage) }
                            .ok_or_else(|| {
                                format!("create handle for linked function {:?}", entry.function)
                            })?;
                    metal_table.setFunction_atIndex(Some(&handle), entry.index as usize);
                }
                crate::library_module::ResolvedIntersectionFunctionEntry::OpaqueTriangle {
                    index,
                    signature,
                } => unsafe {
                    metal_table.setOpaqueTriangleIntersectionFunctionWithSignature_atIndex(
                        metal_intersection_signature(signature),
                        *index as usize,
                    )
                },
            }
        }
        result.intersection.push((table.location, metal_table));
    }
    Ok(result)
}

pub(super) fn bind_fragment_function_tables(
    encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
    tables: &MetalFunctionTables,
) {
    for (binding, table) in &tables.visible {
        unsafe {
            encoder.setFragmentVisibleFunctionTable_atBufferIndex(Some(&**table), *binding as usize)
        };
    }
    for (location, table) in &tables.intersection {
        if let crate::library_module::ResolvedIntersectionFunctionTableLocation::Direct {
            binding,
        } = location
        {
            unsafe {
                encoder.setFragmentIntersectionFunctionTable_atBufferIndex(
                    Some(&**table),
                    *binding as usize,
                )
            };
        }
    }
}

pub(super) fn bind_tile_function_tables(
    encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
    tables: &MetalFunctionTables,
) {
    for (binding, table) in &tables.visible {
        unsafe {
            encoder.setTileVisibleFunctionTable_atBufferIndex(Some(&**table), *binding as usize)
        };
    }
    for (location, table) in &tables.intersection {
        if let crate::library_module::ResolvedIntersectionFunctionTableLocation::Direct {
            binding,
        } = location
        {
            unsafe {
                encoder.setTileIntersectionFunctionTable_atBufferIndex(
                    Some(&**table),
                    *binding as usize,
                )
            };
        }
    }
}

pub(super) fn bind_vertex_function_tables(
    encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
    tables: &MetalFunctionTables,
) {
    for (binding, table) in &tables.visible {
        unsafe {
            encoder.setVertexVisibleFunctionTable_atBufferIndex(Some(&**table), *binding as usize)
        };
    }
    for (location, table) in &tables.intersection {
        if let crate::library_module::ResolvedIntersectionFunctionTableLocation::Direct {
            binding,
        } = location
        {
            unsafe {
                encoder.setVertexIntersectionFunctionTable_atBufferIndex(
                    Some(&**table),
                    *binding as usize,
                )
            };
        }
    }
}
