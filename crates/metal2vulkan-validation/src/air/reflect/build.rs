//! The interface built from one stage's parsed AIR metadata: the old engine's builder, less the
//! fields only its translation filled.

use super::descriptors::{
    buffer_binding, color_input_binding, fragment_imageblock_binding, imageblock_binding,
    sampled_texture_binding, sampler_binding as sampler_descriptor_binding,
    storage_texture_binding, IMAGEBLOCK_BINDING_END, RESOURCE_DESCRIPTOR_SET,
    SAMPLED_TEXTURE_BINDING_END, STORAGE_TEXTURE_BINDING_END,
};
use super::{
    BufferExtent, DescriptorLocation, EmbeddedArgBuffer, FragmentImageblock,
    FragmentImageblockMember, ImageblockLayout, ImplicitImageblockAttachment, RenderTarget,
    ResourceAccess, ResourceBinding, ResourceKind, ShaderInterface, ShaderStage,
    TessellationAttribute, TessellationInterface, Varying, VertexAttribute, VertexBuiltins,
    ADDRESS_SPACE_CONSTANT, ADDRESS_SPACE_THREADGROUP,
};
use crate::air::meta::{
    self, texture_shape_from_name, ArgumentBufferResourceKind, BufferAccess, FragMeta, FragRole,
    KernMeta, KernRole, TextureShape, VertMeta, VertOutRole, VertRole,
};

/// Build reflection for a fragment shader from its parsed meta and (optional) entry name.
pub(crate) fn interface_from_fragment(
    meta: &FragMeta,
    entry_point: Option<&str>,
) -> ShaderInterface {
    let mut bindings = Vec::new();
    for (idx, role) in &meta.roles {
        let idx = *idx;
        let binding = match role {
            FragRole::Buffer(n) => ResourceBinding {
                kind: ResourceKind::Buffer,
                metal_index: *n,
                descriptor: descriptor_at(buffer_binding(*n)),
                param_index: Some(idx),
                stage_input_location: None,
                address_space: meta.buffer_address_spaces.get(&idx).copied(),
                declared_size: meta.buffer_type_sizes.get(&idx).copied(),
                extent: Some(buffer_extent(
                    meta.buffer_object_sizes.get(&idx).copied(),
                    meta.buffer_type_sizes.get(&idx).copied(),
                    meta.buffer_type_names.get(&idx),
                )),
                type_layout: meta.buffer_layouts.get(&idx).cloned(),
                type_name: meta.buffer_type_names.get(&idx).cloned(),
                texture_shape: None,
                embedded_source: None,
                access: buffer_access(
                    meta.buffer_accesses.get(&idx).copied(),
                    meta.buffer_address_spaces.get(&idx).copied(),
                ),
                static_sampler: None,
            },
            FragRole::Texture(n) => texture_binding(*n, Some(idx), &meta.texture_type_names),
            FragRole::Sampler(n) => sampler_binding(*n, Some(idx)),
            FragRole::VisibleFunctionTable(n) => {
                function_table_binding(ResourceKind::VisibleFunctionTable, *n, idx)
            }
            FragRole::IntersectionFunctionTable(n) => {
                function_table_binding(ResourceKind::IntersectionFunctionTable, *n, idx)
            }
            FragRole::ColorInput(n) => ResourceBinding {
                kind: ResourceKind::ColorInput,
                metal_index: *n,
                descriptor: descriptor_at(color_input_binding(*n)),
                param_index: Some(idx),
                stage_input_location: None,
                address_space: None,
                declared_size: None,
                extent: None,
                type_layout: None,
                type_name: meta.color_input_type_names.get(n).cloned(),
                texture_shape: None,
                embedded_source: None,
                access: None,
                static_sampler: None,
            },
            FragRole::Position
            | FragRole::PointCoord
            | FragRole::FrontFacing
            | FragRole::BarycentricCoord { .. }
            | FragRole::PrimitiveId
            | FragRole::SampleId
            | FragRole::SampleMaskIn
            | FragRole::ViewportArrayIndex
            | FragRole::RenderTargetArrayIndex
            | FragRole::AmplificationId
            | FragRole::AmplificationCount
            | FragRole::Varying(_)
            | FragRole::ImageblockData
            | FragRole::ExecutionGroup { .. }
            | FragRole::VariantAbsentTexture
            | FragRole::Other => {
                continue;
            }
        };
        bindings.push(binding);
    }
    append_embedded_resources(
        &mut bindings,
        &meta.embedded_textures,
        &meta.embedded_arguments,
    );
    let varyings = meta
        .varying_types
        .keys()
        .chain(meta.varying_names.keys())
        .chain(meta.varying_user_semantics.keys())
        .copied()
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .map(|loc| Varying {
            location: loc,
            type_name: meta.varying_types.get(&loc).cloned(),
        })
        .collect();
    let render_targets = meta
        .render_target_members
        .iter()
        .map(|(member, location)| RenderTarget {
            location: *location,
            type_name: meta.render_target_type_names.get(member).cloned(),
        })
        .collect();
    ShaderInterface {
        stage: ShaderStage::Fragment,
        entry_point: entry_point.map(str::to_string),
        bindings,
        argument_buffer_fields: embedded_argument_fields(&meta.embedded_arguments),
        vertex_attributes: Vec::new(),
        varyings,
        render_targets,
        depth_members: meta.depth_members.clone(),
        depth_qualifier: meta.depth_qualifier,
        stencil_members: meta.stencil_members.clone(),
        vertex_builtins: None,
        tessellation: None,
        imageblock_layouts: Vec::new(),
        implicit_imageblock_attachments: implicit_imageblock_planes(
            &meta.implicit_imageblock_attachments,
        ),
        fragment_imageblock: meta.fragment_imageblock.as_ref().map(|imageblock| {
            FragmentImageblock {
                sample_size: imageblock.sample_size,
                members: imageblock
                    .members
                    .iter()
                    .enumerate()
                    .map(|(index, member)| {
                        let index = index as u32;
                        let reads = imageblock.inputs.iter().any(|projection| {
                            projection
                                .members
                                .iter()
                                .any(|projected| projected.master_member == index)
                        });
                        let writes = imageblock.outputs.iter().any(|projection| {
                            projection
                                .members
                                .iter()
                                .any(|projected| projected.master_member == index)
                        });
                        FragmentImageblockMember {
                            offset: member.offset,
                            size: member.size,
                            type_name: member.type_name.clone(),
                            semantic: member.semantic.clone(),
                            raster_order_group: member.raster_order_group,
                            binding: (reads || writes)
                                .then(|| fragment_imageblock_binding(index))
                                .flatten(),
                            access: match (reads, writes) {
                                (true, true) => ResourceAccess::ReadWrite,
                                (true, false) => ResourceAccess::ReadOnly,
                                (false, true) => ResourceAccess::WriteOnly,
                                (false, false) => ResourceAccess::Unused,
                            },
                        }
                    })
                    .collect(),
                inputs: imageblock.inputs.clone(),
                outputs: imageblock.outputs.clone(),
            }
        }),
        mesh_stage: None,
        function_constants: Vec::new(),
    }
}

/// Build reflection for a vertex shader from its parsed meta and (optional) entry name.
pub(crate) fn interface_from_vertex(meta: &VertMeta, entry_point: Option<&str>) -> ShaderInterface {
    let is_tessellation = meta.tessellation.is_some();
    let vertex_builtins = VertexBuiltins {
        writes_position: meta.output_roles.contains(&VertOutRole::Position),
    };
    let mut bindings = Vec::new();
    for (idx, role) in &meta.roles {
        let idx = *idx;
        let binding = match role {
            VertRole::Buffer(n) => ResourceBinding {
                kind: ResourceKind::Buffer,
                metal_index: *n,
                descriptor: descriptor_at(buffer_binding(*n)),
                param_index: Some(idx),
                stage_input_location: None,
                address_space: meta.buffer_address_spaces.get(&idx).copied(),
                declared_size: meta.buffer_type_sizes.get(&idx).copied(),
                extent: Some(buffer_extent(
                    meta.buffer_object_sizes.get(&idx).copied(),
                    meta.buffer_type_sizes.get(&idx).copied(),
                    meta.buffer_type_names.get(&idx),
                )),
                type_layout: meta.buffer_layouts.get(&idx).cloned(),
                type_name: meta.buffer_type_names.get(&idx).cloned(),
                texture_shape: None,
                embedded_source: None,
                access: buffer_access(
                    meta.buffer_accesses.get(&idx).copied(),
                    meta.buffer_address_spaces.get(&idx).copied(),
                ),
                static_sampler: None,
            },
            VertRole::Texture(n) => texture_binding(*n, Some(idx), &meta.texture_type_names),
            VertRole::Sampler(n) => sampler_binding(*n, Some(idx)),
            VertRole::VisibleFunctionTable(n) => {
                function_table_binding(ResourceKind::VisibleFunctionTable, *n, idx)
            }
            VertRole::IntersectionFunctionTable(n) => {
                function_table_binding(ResourceKind::IntersectionFunctionTable, *n, idx)
            }
            VertRole::VertexInput(_)
            | VertRole::VertexId
            | VertRole::InstanceId
            | VertRole::PatchControlPoints
            | VertRole::PatchInput(_)
            | VertRole::PositionInPatch
            | VertRole::PatchId
            | VertRole::AmplificationId
            | VertRole::AmplificationCount
            | VertRole::ExecutionGroup { .. }
            | VertRole::VariantAbsentTexture
            | VertRole::Other => continue,
        };
        bindings.push(binding);
    }
    append_embedded_resources(
        &mut bindings,
        &meta.embedded_textures,
        &meta.embedded_arguments,
    );
    let vertex_attributes = meta
        .vertex_input_types
        .keys()
        .chain(meta.vertex_input_names.keys())
        .copied()
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .map(|loc| VertexAttribute {
            location: loc,
            type_name: meta.vertex_input_types.get(&loc).cloned(),
        })
        .collect();
    let varyings = meta
        .output_roles
        .iter()
        .filter_map(|role| match role {
            VertOutRole::Varying(loc) => Some(Varying {
                location: *loc,
                type_name: meta.output_varying_types.get(loc).cloned(),
            }),
            _ => None,
        })
        .collect();
    ShaderInterface {
        stage: if is_tessellation {
            ShaderStage::TessellationEvaluation
        } else {
            ShaderStage::Vertex
        },
        entry_point: entry_point.map(str::to_string),
        bindings,
        argument_buffer_fields: embedded_argument_fields(&meta.embedded_arguments),
        vertex_attributes,
        varyings,
        render_targets: Vec::new(),
        depth_members: Vec::new(),
        depth_qualifier: None,
        stencil_members: Vec::new(),
        vertex_builtins: Some(vertex_builtins),
        tessellation: meta.tessellation.as_ref().map(|tessellation| {
            let mut patch_input_locations = meta
                .roles
                .iter()
                .filter_map(|(_, role)| match role {
                    VertRole::PatchInput(location) => Some(*location),
                    _ => None,
                })
                .collect::<Vec<_>>();
            patch_input_locations.sort_unstable();
            let patch_attributes = patch_input_locations
                .iter()
                .map(|location| TessellationAttribute {
                    location: *location,
                    type_name: meta.patch_input_types.get(location).cloned(),
                })
                .collect();
            TessellationInterface {
                domain: tessellation.domain,
                control_point_count: tessellation.control_point_count,
                control_point_attributes: tessellation
                    .control_point_fields
                    .iter()
                    .map(|field| TessellationAttribute {
                        location: field.location,
                        type_name: field.type_name.clone(),
                    })
                    .collect(),
                patch_attributes,
                instance_id: tessellation_system_attribute(meta, &VertRole::InstanceId),
                amplification_id: tessellation_system_attribute(meta, &VertRole::AmplificationId),
                amplification_count: tessellation_system_attribute(
                    meta,
                    &VertRole::AmplificationCount,
                ),
            }
        }),
        imageblock_layouts: Vec::new(),
        implicit_imageblock_attachments: implicit_imageblock_planes(
            &meta.implicit_imageblock_attachments,
        ),
        fragment_imageblock: None,
        mesh_stage: None,
        function_constants: Vec::new(),
    }
}

/// Build reflection for a compute kernel from its parsed meta and (optional) entry name.
pub(crate) fn interface_from_kernel(meta: &KernMeta, entry_point: Option<&str>) -> ShaderInterface {
    let mut bindings = Vec::new();
    let stage_input_bindings = meta.stage_input_bindings();
    for (idx, role) in &meta.roles {
        let idx = *idx;
        let binding = match role {
            KernRole::Buffer(n) => {
                let address_space = meta.buffer_address_spaces.get(&idx).copied();
                let threadgroup = address_space == Some(ADDRESS_SPACE_THREADGROUP);
                let kind = if threadgroup {
                    ResourceKind::ThreadgroupBuffer
                } else {
                    ResourceKind::Buffer
                };
                ResourceBinding {
                    kind,
                    metal_index: *n,
                    descriptor: if threadgroup {
                        None
                    } else {
                        descriptor_at(buffer_binding(*n))
                    },
                    param_index: Some(idx),
                    stage_input_location: None,
                    address_space,
                    declared_size: meta.buffer_type_sizes.get(&idx).copied(),
                    extent: Some(buffer_extent(
                        meta.buffer_object_sizes.get(&idx).copied(),
                        meta.buffer_type_sizes.get(&idx).copied(),
                        meta.buffer_type_names.get(&idx),
                    )),
                    type_layout: meta.buffer_layouts.get(&idx).cloned(),
                    type_name: meta.buffer_type_names.get(&idx).cloned(),
                    texture_shape: None,
                    embedded_source: None,
                    access: buffer_access(meta.buffer_accesses.get(&idx).copied(), address_space),
                    static_sampler: None,
                }
            }
            KernRole::Texture(n) => texture_binding(*n, Some(idx), &meta.texture_type_names),
            KernRole::Sampler(n) => sampler_binding(*n, Some(idx)),
            KernRole::StageInput(location) => {
                let metal_index = stage_input_bindings
                    .get(&idx)
                    .copied()
                    .expect("stage-input binding was allocated");
                ResourceBinding {
                    kind: ResourceKind::KernelStageInput,
                    metal_index,
                    descriptor: descriptor_at(buffer_binding(metal_index)),
                    param_index: Some(idx),
                    stage_input_location: Some(*location),
                    address_space: None,
                    declared_size: None,
                    extent: Some(BufferExtent::Unbounded),
                    type_layout: None,
                    type_name: meta.stage_input_type_names.get(&idx).cloned(),
                    texture_shape: None,
                    embedded_source: None,
                    access: Some(ResourceAccess::ReadOnly),
                    static_sampler: None,
                }
            }
            KernRole::AccelerationStructureShadow(n) => ResourceBinding {
                kind: ResourceKind::AccelerationStructureShadow,
                metal_index: *n,
                descriptor: descriptor_at(buffer_binding(*n)),
                param_index: Some(idx),
                stage_input_location: None,
                address_space: None,
                declared_size: None,
                extent: None,
                type_layout: None,
                type_name: None,
                texture_shape: None,
                embedded_source: None,
                access: None,
                static_sampler: None,
            },
            KernRole::PrimitiveAccelerationStructure(n) => ResourceBinding {
                kind: ResourceKind::PrimitiveAccelerationStructure,
                metal_index: *n,
                descriptor: None,
                param_index: Some(idx),
                stage_input_location: None,
                address_space: None,
                declared_size: None,
                extent: None,
                type_layout: None,
                type_name: Some("acceleration_structure<>".into()),
                texture_shape: None,
                embedded_source: None,
                access: Some(ResourceAccess::ReadOnly),
                static_sampler: None,
            },
            KernRole::PrimitiveAccelerationStructureShadow(n) => ResourceBinding {
                kind: ResourceKind::PrimitiveAccelerationStructure,
                metal_index: *n,
                descriptor: descriptor_at(buffer_binding(*n)),
                param_index: Some(idx),
                stage_input_location: None,
                address_space: None,
                declared_size: None,
                extent: Some(BufferExtent::Unbounded),
                type_layout: None,
                type_name: Some("acceleration_structure<>".into()),
                texture_shape: None,
                embedded_source: None,
                access: Some(ResourceAccess::ReadOnly),
                static_sampler: None,
            },
            KernRole::VisibleFunctionTable(n) => {
                function_table_binding(ResourceKind::VisibleFunctionTable, *n, idx)
            }
            KernRole::IntersectionFunctionTable(n) => {
                function_table_binding(ResourceKind::IntersectionFunctionTable, *n, idx)
            }
            _ => continue,
        };
        bindings.push(binding);
    }
    append_embedded_resources(
        &mut bindings,
        &meta.embedded_textures,
        &meta.embedded_arguments,
    );
    ShaderInterface {
        stage: ShaderStage::Kernel,
        entry_point: entry_point.map(str::to_string),
        bindings,
        argument_buffer_fields: embedded_argument_fields(&meta.embedded_arguments),
        vertex_attributes: Vec::new(),
        varyings: Vec::new(),
        render_targets: Vec::new(),
        depth_members: Vec::new(),
        depth_qualifier: None,
        stencil_members: Vec::new(),
        vertex_builtins: None,
        tessellation: None,
        imageblock_layouts: {
            let mut ibs: Vec<ImageblockLayout> = meta
                .imageblock_layouts
                .keys()
                .map(|idx| ImageblockLayout { param_index: *idx })
                .collect();
            ibs.sort_by_key(|ib| ib.param_index);
            ibs
        },
        implicit_imageblock_attachments: implicit_imageblock_planes(
            &meta.implicit_imageblock_attachments,
        ),
        fragment_imageblock: None,
        mesh_stage: None,
        function_constants: Vec::new(),
    }
}

/// Build reflection for an object or mesh function (`stage`) from its parsed meta, (optional)
/// entry name and `reached`, the text of what the entry reaches (`air::entry_text`): a kernel's
/// interface, at its own stage, with the stage's own facts.
pub(crate) fn interface_from_mesh_stage(
    stage: crate::case::Stage,
    meta: &KernMeta,
    entry_point: Option<&str>,
    reached: &str,
) -> ShaderInterface {
    let mut interface = interface_from_kernel(meta, entry_point);
    interface.stage = match stage {
        crate::case::Stage::Object => ShaderStage::Object,
        _ => ShaderStage::Mesh,
    };
    let own = &meta.mesh_stage;
    interface.mesh_stage = Some(super::MeshStageInterface {
        payload: own.payload.clone(),
        mesh_grid: own.mesh_grid_param.is_some(),
        sets_mesh_grid: reached.contains("@air.set_threadgroups_per_grid_mesh_properties("),
        outputs: own.outputs.clone(),
        max_total_threads: meta.max_work_group_size,
        max_mesh_threadgroups: own.max_mesh_work_groups,
    });
    interface
}

/// A descriptor in set 0 at `binding`, one descriptor wide, if the band has the index.
fn descriptor_at(binding: Option<u32>) -> Option<DescriptorLocation> {
    Some(DescriptorLocation {
        set: RESOURCE_DESCRIPTOR_SET,
        binding: binding?,
        count: 1,
    })
}

fn tessellation_system_attribute(
    meta: &VertMeta,
    expected_role: &VertRole,
) -> Option<TessellationAttribute> {
    let (parameter, role) = meta
        .roles
        .iter()
        .find(|(_, role)| *role == *expected_role)?;
    Some(TessellationAttribute {
        location: meta.tessellation_system_input_location(role)?,
        type_name: meta.parameter_type_names.get(parameter).cloned(),
    })
}

/// The implicit-imageblock planes a module's `air.load/store.implicit_imageblock.*` calls use.
/// They are found in the module body whatever the stage, so every stage reports them.
fn implicit_imageblock_planes(
    attachments: &[meta::ImplicitImageblockAttachment],
) -> Vec<ImplicitImageblockAttachment> {
    attachments
        .iter()
        .map(|attachment| ImplicitImageblockAttachment {
            attachment: attachment.attachment,
            data_rate: attachment.data_rate,
            binding: imageblock_binding(attachment.attachment, attachment.data_rate)
                .unwrap_or(IMAGEBLOCK_BINDING_END),
            format: attachment.format,
            access: match (attachment.reads, attachment.writes) {
                (true, true) => ResourceAccess::ReadWrite,
                (true, false) => ResourceAccess::ReadOnly,
                (false, true) => ResourceAccess::WriteOnly,
                (false, false) => ResourceAccess::Unused,
            },
        })
        .collect()
}

fn texture_binding(
    n: u32,
    param_index: Option<u32>,
    type_names: &std::collections::HashMap<u32, String>,
) -> ResourceBinding {
    let type_name = param_index.and_then(|idx| type_names.get(&idx).cloned());
    let texture_shape = type_name.as_deref().map(texture_shape_from_name);
    let (kind, access) = classify_texture(texture_shape.as_ref());
    let descriptor_binding = if access == ResourceAccess::Storage {
        storage_texture_binding(n)
    } else {
        sampled_texture_binding(n)
    };
    let mut descriptor = descriptor_at(descriptor_binding);
    if let (Some(location), Some(shape)) = (descriptor.as_mut(), texture_shape.as_ref()) {
        location.count = shape.descriptor_count();
    }
    ResourceBinding {
        kind,
        metal_index: n,
        descriptor,
        param_index,
        stage_input_location: None,
        address_space: None,
        declared_size: None,
        extent: None,
        type_layout: None,
        type_name,
        texture_shape,
        embedded_source: None,
        access: Some(access),
        static_sampler: None,
    }
}

/// A texture argument's kind and declared access: a `write` or `read_write` texture is a storage
/// image, a texture-handle array follows its element's qualifier, and anything else (or a texture
/// whose type name AIR does not carry) is a sampled texture.
fn classify_texture(shape: Option<&TextureShape>) -> (ResourceKind, ResourceAccess) {
    let Some(shape) = shape else {
        return (ResourceKind::Texture, ResourceAccess::Sampled);
    };
    if shape.array_ref {
        (
            ResourceKind::TextureArray,
            if shape.writable {
                ResourceAccess::Storage
            } else {
                ResourceAccess::Sampled
            },
        )
    } else if shape.writable {
        (ResourceKind::StorageImage, ResourceAccess::Storage)
    } else {
        (ResourceKind::Texture, ResourceAccess::Sampled)
    }
}

fn buffer_extent(
    object_size: Option<u32>,
    declared_size: Option<u32>,
    type_name: Option<&String>,
) -> BufferExtent {
    match (object_size, declared_size, type_name) {
        (Some(bytes), _, _) => BufferExtent::Object { bytes },
        (None, Some(_), _) | (None, None, Some(_)) => BufferExtent::Unbounded,
        (None, None, None) => BufferExtent::Unknown,
    }
}

fn buffer_access(
    declared: Option<BufferAccess>,
    address_space: Option<u32>,
) -> Option<ResourceAccess> {
    match declared {
        Some(BufferAccess::ReadOnly) => Some(ResourceAccess::ReadOnly),
        Some(BufferAccess::WriteOnly) => Some(ResourceAccess::WriteOnly),
        Some(BufferAccess::ReadWrite) => Some(ResourceAccess::ReadWrite),
        None if address_space == Some(ADDRESS_SPACE_CONSTANT) => Some(ResourceAccess::ReadOnly),
        None => None,
    }
}

fn append_embedded_resources(
    bindings: &mut Vec<ResourceBinding>,
    textures: &[meta::EmbeddedTexture],
    arguments: &[meta::EmbeddedArgument],
) {
    for embedded in textures {
        let descriptor_binding = if embedded.storage_format.is_some() {
            storage_texture_binding(embedded.synthetic_texture_index)
                .unwrap_or(STORAGE_TEXTURE_BINDING_END)
        } else {
            sampled_texture_binding(embedded.synthetic_texture_index)
                .unwrap_or(SAMPLED_TEXTURE_BINDING_END)
        };
        bindings.push(ResourceBinding {
            kind: ResourceKind::EmbeddedArgBufferTexture,
            metal_index: embedded.synthetic_texture_index,
            descriptor: Some(DescriptorLocation {
                set: RESOURCE_DESCRIPTOR_SET,
                binding: descriptor_binding,
                count: embedded.array_length.unwrap_or(1),
            }),
            param_index: None,
            stage_input_location: None,
            address_space: None,
            declared_size: None,
            extent: None,
            type_layout: None,
            type_name: None,
            texture_shape: Some(TextureShape {
                dimension: embedded.dim,
                arrayed: embedded.arrayed,
                multisampled: false,
                component: embedded.comp,
                writable: embedded.storage_format.is_some(),
                array_ref: embedded.array_length.is_some(),
                array_length: embedded.array_length,
                storage_format: embedded.storage_format,
            }),
            embedded_source: Some(EmbeddedArgBuffer {
                buffer_param_index: embedded.buffer_param_index,
                buffer_index: embedded.buffer_index,
                field_offset: embedded.field_offset,
                field_ordinal: embedded.field_ordinal,
                argument_index: embedded.argument_index,
                resource_kind: Some(ArgumentBufferResourceKind::Texture),
            }),
            access: Some(if embedded.storage_format.is_some() {
                ResourceAccess::Storage
            } else {
                ResourceAccess::Sampled
            }),
            static_sampler: None,
        });
    }
    for argument in arguments {
        if argument.resource_kind != Some(ArgumentBufferResourceKind::Buffer) {
            continue;
        }
        bindings.push(ResourceBinding {
            kind: ResourceKind::EmbeddedArgBufferBuffer,
            metal_index: argument.argument_index,
            descriptor: None,
            param_index: None,
            stage_input_location: None,
            address_space: argument.resource_address_space,
            declared_size: argument.resource_declared_size,
            extent: Some(buffer_extent(None, argument.resource_declared_size, None)),
            type_layout: argument.resource_type_layout.clone(),
            type_name: None,
            texture_shape: None,
            embedded_source: Some(EmbeddedArgBuffer {
                buffer_param_index: argument.buffer_param_index,
                buffer_index: argument.buffer_index,
                field_offset: argument.field_offset,
                field_ordinal: argument.field_ordinal,
                argument_index: argument.argument_index,
                resource_kind: Some(ArgumentBufferResourceKind::Buffer),
            }),
            access: buffer_access(argument.resource_access, argument.resource_address_space),
            static_sampler: None,
        });
    }
}

fn embedded_argument_fields(arguments: &[meta::EmbeddedArgument]) -> Vec<EmbeddedArgBuffer> {
    arguments
        .iter()
        .map(|argument| EmbeddedArgBuffer {
            buffer_param_index: argument.buffer_param_index,
            buffer_index: argument.buffer_index,
            field_offset: argument.field_offset,
            field_ordinal: argument.field_ordinal,
            argument_index: argument.argument_index,
            resource_kind: argument.resource_kind,
        })
        .collect()
}

fn sampler_binding(n: u32, param_index: Option<u32>) -> ResourceBinding {
    ResourceBinding {
        kind: ResourceKind::Sampler,
        metal_index: n,
        descriptor: descriptor_at(sampler_descriptor_binding(n)),
        param_index,
        stage_input_location: None,
        address_space: None,
        declared_size: None,
        extent: None,
        type_layout: None,
        type_name: None,
        texture_shape: None,
        embedded_source: None,
        access: None,
        static_sampler: None,
    }
}

fn function_table_binding(
    kind: ResourceKind,
    metal_index: u32,
    param_index: u32,
) -> ResourceBinding {
    let type_name = match kind {
        ResourceKind::VisibleFunctionTable => "visible_function_table",
        _ => "intersection_function_table",
    };
    ResourceBinding {
        kind,
        metal_index,
        descriptor: None,
        param_index: Some(param_index),
        stage_input_location: None,
        address_space: None,
        declared_size: None,
        extent: None,
        type_layout: None,
        type_name: Some(type_name.into()),
        texture_shape: None,
        embedded_source: None,
        access: Some(ResourceAccess::ReadOnly),
        static_sampler: None,
    }
}
