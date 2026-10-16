//! The Vulkan registry's read accesses, stages and their extensions, as vk.xml of Vulkan-Headers
//! `vulkan-sdk-1.4.341.0` (the registry `scripts/regen-vulkan-requirements` pins) states them in
//! its `sync` section, and the extensions each of those requires (its `depends`, where every
//! alternative requires them). `VK_KHR_synchronization2`'s, promoted to Vulkan 1.3, and the
//! versions' are the core's.

use std::ffi::CStr;

/// Each read access type (but `MEMORY_READ` and `HOST_READ`) with each stage that supports it
/// (`syncaccess`/`syncsupport`), and the extensions both the access and the stage need there:
/// one row for each least such set.
pub(super) const GRANTS: [(&[&CStr], &str, &str); 188] = [
    (
        &[c"VK_KHR_acceleration_structure"],
        "ACCELERATION_STRUCTURE_READ_BIT_KHR",
        "ACCELERATION_STRUCTURE_BUILD_BIT_KHR",
    ),
    (
        &[c"VK_NV_ray_tracing"],
        "ACCELERATION_STRUCTURE_READ_BIT_KHR",
        "ACCELERATION_STRUCTURE_BUILD_BIT_KHR",
    ),
    (
        &[
            c"VK_KHR_acceleration_structure",
            c"VK_KHR_ray_tracing_maintenance1",
        ],
        "ACCELERATION_STRUCTURE_READ_BIT_KHR",
        "ACCELERATION_STRUCTURE_COPY_BIT_KHR",
    ),
    (
        &[c"VK_KHR_ray_tracing_maintenance1", c"VK_NV_ray_tracing"],
        "ACCELERATION_STRUCTURE_READ_BIT_KHR",
        "ACCELERATION_STRUCTURE_COPY_BIT_KHR",
    ),
    (
        &[
            c"VK_HUAWEI_cluster_culling_shader",
            c"VK_KHR_acceleration_structure",
        ],
        "ACCELERATION_STRUCTURE_READ_BIT_KHR",
        "CLUSTER_CULLING_SHADER_BIT_HUAWEI",
    ),
    (
        &[c"VK_HUAWEI_cluster_culling_shader", c"VK_NV_ray_tracing"],
        "ACCELERATION_STRUCTURE_READ_BIT_KHR",
        "CLUSTER_CULLING_SHADER_BIT_HUAWEI",
    ),
    (
        &[c"VK_KHR_acceleration_structure"],
        "ACCELERATION_STRUCTURE_READ_BIT_KHR",
        "COMPUTE_SHADER_BIT",
    ),
    (
        &[c"VK_NV_ray_tracing"],
        "ACCELERATION_STRUCTURE_READ_BIT_KHR",
        "COMPUTE_SHADER_BIT",
    ),
    (
        &[c"VK_KHR_acceleration_structure"],
        "ACCELERATION_STRUCTURE_READ_BIT_KHR",
        "FRAGMENT_SHADER_BIT",
    ),
    (
        &[c"VK_NV_ray_tracing"],
        "ACCELERATION_STRUCTURE_READ_BIT_KHR",
        "FRAGMENT_SHADER_BIT",
    ),
    (
        &[c"VK_KHR_acceleration_structure"],
        "ACCELERATION_STRUCTURE_READ_BIT_KHR",
        "GEOMETRY_SHADER_BIT",
    ),
    (
        &[c"VK_NV_ray_tracing"],
        "ACCELERATION_STRUCTURE_READ_BIT_KHR",
        "GEOMETRY_SHADER_BIT",
    ),
    (
        &[c"VK_EXT_mesh_shader", c"VK_KHR_acceleration_structure"],
        "ACCELERATION_STRUCTURE_READ_BIT_KHR",
        "MESH_SHADER_BIT_EXT",
    ),
    (
        &[c"VK_EXT_mesh_shader", c"VK_NV_ray_tracing"],
        "ACCELERATION_STRUCTURE_READ_BIT_KHR",
        "MESH_SHADER_BIT_EXT",
    ),
    (
        &[c"VK_KHR_acceleration_structure", c"VK_NV_mesh_shader"],
        "ACCELERATION_STRUCTURE_READ_BIT_KHR",
        "MESH_SHADER_BIT_EXT",
    ),
    (
        &[c"VK_NV_mesh_shader", c"VK_NV_ray_tracing"],
        "ACCELERATION_STRUCTURE_READ_BIT_KHR",
        "MESH_SHADER_BIT_EXT",
    ),
    (
        &[
            c"VK_KHR_acceleration_structure",
            c"VK_KHR_ray_tracing_pipeline",
        ],
        "ACCELERATION_STRUCTURE_READ_BIT_KHR",
        "RAY_TRACING_SHADER_BIT_KHR",
    ),
    (
        &[c"VK_NV_ray_tracing"],
        "ACCELERATION_STRUCTURE_READ_BIT_KHR",
        "RAY_TRACING_SHADER_BIT_KHR",
    ),
    (
        &[
            c"VK_HUAWEI_subpass_shading",
            c"VK_KHR_acceleration_structure",
        ],
        "ACCELERATION_STRUCTURE_READ_BIT_KHR",
        "SUBPASS_SHADER_BIT_HUAWEI",
    ),
    (
        &[c"VK_HUAWEI_subpass_shading", c"VK_NV_ray_tracing"],
        "ACCELERATION_STRUCTURE_READ_BIT_KHR",
        "SUBPASS_SHADER_BIT_HUAWEI",
    ),
    (
        &[c"VK_EXT_mesh_shader", c"VK_KHR_acceleration_structure"],
        "ACCELERATION_STRUCTURE_READ_BIT_KHR",
        "TASK_SHADER_BIT_EXT",
    ),
    (
        &[c"VK_EXT_mesh_shader", c"VK_NV_ray_tracing"],
        "ACCELERATION_STRUCTURE_READ_BIT_KHR",
        "TASK_SHADER_BIT_EXT",
    ),
    (
        &[c"VK_KHR_acceleration_structure", c"VK_NV_mesh_shader"],
        "ACCELERATION_STRUCTURE_READ_BIT_KHR",
        "TASK_SHADER_BIT_EXT",
    ),
    (
        &[c"VK_NV_mesh_shader", c"VK_NV_ray_tracing"],
        "ACCELERATION_STRUCTURE_READ_BIT_KHR",
        "TASK_SHADER_BIT_EXT",
    ),
    (
        &[c"VK_KHR_acceleration_structure"],
        "ACCELERATION_STRUCTURE_READ_BIT_KHR",
        "TESSELLATION_CONTROL_SHADER_BIT",
    ),
    (
        &[c"VK_NV_ray_tracing"],
        "ACCELERATION_STRUCTURE_READ_BIT_KHR",
        "TESSELLATION_CONTROL_SHADER_BIT",
    ),
    (
        &[c"VK_KHR_acceleration_structure"],
        "ACCELERATION_STRUCTURE_READ_BIT_KHR",
        "TESSELLATION_EVALUATION_SHADER_BIT",
    ),
    (
        &[c"VK_NV_ray_tracing"],
        "ACCELERATION_STRUCTURE_READ_BIT_KHR",
        "TESSELLATION_EVALUATION_SHADER_BIT",
    ),
    (
        &[c"VK_KHR_acceleration_structure"],
        "ACCELERATION_STRUCTURE_READ_BIT_KHR",
        "VERTEX_SHADER_BIT",
    ),
    (
        &[c"VK_NV_ray_tracing"],
        "ACCELERATION_STRUCTURE_READ_BIT_KHR",
        "VERTEX_SHADER_BIT",
    ),
    (
        &[],
        "COLOR_ATTACHMENT_READ_BIT",
        "COLOR_ATTACHMENT_OUTPUT_BIT",
    ),
    (&[], "COLOR_ATTACHMENT_READ_BIT", "FRAGMENT_SHADER_BIT"),
    (
        &[c"VK_EXT_blend_operation_advanced"],
        "COLOR_ATTACHMENT_READ_NONCOHERENT_BIT_EXT",
        "COLOR_ATTACHMENT_OUTPUT_BIT",
    ),
    (
        &[c"VK_EXT_device_generated_commands"],
        "COMMAND_PREPROCESS_READ_BIT_EXT",
        "COMMAND_PREPROCESS_BIT_EXT",
    ),
    (
        &[c"VK_NV_device_generated_commands"],
        "COMMAND_PREPROCESS_READ_BIT_EXT",
        "COMMAND_PREPROCESS_BIT_EXT",
    ),
    (
        &[c"VK_EXT_conditional_rendering"],
        "CONDITIONAL_RENDERING_READ_BIT_EXT",
        "CONDITIONAL_RENDERING_BIT_EXT",
    ),
    (
        &[c"VK_ARM_data_graph"],
        "DATA_GRAPH_READ_BIT_ARM",
        "DATA_GRAPH_BIT_ARM",
    ),
    (
        &[],
        "DEPTH_STENCIL_ATTACHMENT_READ_BIT",
        "EARLY_FRAGMENT_TESTS_BIT",
    ),
    (
        &[],
        "DEPTH_STENCIL_ATTACHMENT_READ_BIT",
        "FRAGMENT_SHADER_BIT",
    ),
    (
        &[],
        "DEPTH_STENCIL_ATTACHMENT_READ_BIT",
        "LATE_FRAGMENT_TESTS_BIT",
    ),
    (
        &[
            c"VK_EXT_descriptor_buffer",
            c"VK_HUAWEI_cluster_culling_shader",
        ],
        "DESCRIPTOR_BUFFER_READ_BIT_EXT",
        "CLUSTER_CULLING_SHADER_BIT_HUAWEI",
    ),
    (
        &[c"VK_EXT_descriptor_buffer"],
        "DESCRIPTOR_BUFFER_READ_BIT_EXT",
        "COMPUTE_SHADER_BIT",
    ),
    (
        &[c"VK_EXT_descriptor_buffer"],
        "DESCRIPTOR_BUFFER_READ_BIT_EXT",
        "FRAGMENT_SHADER_BIT",
    ),
    (
        &[c"VK_EXT_descriptor_buffer"],
        "DESCRIPTOR_BUFFER_READ_BIT_EXT",
        "GEOMETRY_SHADER_BIT",
    ),
    (
        &[c"VK_EXT_descriptor_buffer", c"VK_EXT_mesh_shader"],
        "DESCRIPTOR_BUFFER_READ_BIT_EXT",
        "MESH_SHADER_BIT_EXT",
    ),
    (
        &[c"VK_EXT_descriptor_buffer", c"VK_NV_mesh_shader"],
        "DESCRIPTOR_BUFFER_READ_BIT_EXT",
        "MESH_SHADER_BIT_EXT",
    ),
    (
        &[c"VK_EXT_descriptor_buffer", c"VK_KHR_ray_tracing_pipeline"],
        "DESCRIPTOR_BUFFER_READ_BIT_EXT",
        "RAY_TRACING_SHADER_BIT_KHR",
    ),
    (
        &[c"VK_EXT_descriptor_buffer", c"VK_NV_ray_tracing"],
        "DESCRIPTOR_BUFFER_READ_BIT_EXT",
        "RAY_TRACING_SHADER_BIT_KHR",
    ),
    (
        &[c"VK_EXT_descriptor_buffer", c"VK_HUAWEI_subpass_shading"],
        "DESCRIPTOR_BUFFER_READ_BIT_EXT",
        "SUBPASS_SHADER_BIT_HUAWEI",
    ),
    (
        &[c"VK_EXT_descriptor_buffer", c"VK_EXT_mesh_shader"],
        "DESCRIPTOR_BUFFER_READ_BIT_EXT",
        "TASK_SHADER_BIT_EXT",
    ),
    (
        &[c"VK_EXT_descriptor_buffer", c"VK_NV_mesh_shader"],
        "DESCRIPTOR_BUFFER_READ_BIT_EXT",
        "TASK_SHADER_BIT_EXT",
    ),
    (
        &[c"VK_EXT_descriptor_buffer"],
        "DESCRIPTOR_BUFFER_READ_BIT_EXT",
        "TESSELLATION_CONTROL_SHADER_BIT",
    ),
    (
        &[c"VK_EXT_descriptor_buffer"],
        "DESCRIPTOR_BUFFER_READ_BIT_EXT",
        "TESSELLATION_EVALUATION_SHADER_BIT",
    ),
    (
        &[c"VK_EXT_descriptor_buffer"],
        "DESCRIPTOR_BUFFER_READ_BIT_EXT",
        "VERTEX_SHADER_BIT",
    ),
    (
        &[c"VK_EXT_fragment_density_map"],
        "FRAGMENT_DENSITY_MAP_READ_BIT_EXT",
        "FRAGMENT_DENSITY_PROCESS_BIT_EXT",
    ),
    (
        &[c"VK_KHR_fragment_shading_rate"],
        "FRAGMENT_SHADING_RATE_ATTACHMENT_READ_BIT_KHR",
        "FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR",
    ),
    (
        &[c"VK_NV_shading_rate_image"],
        "FRAGMENT_SHADING_RATE_ATTACHMENT_READ_BIT_KHR",
        "FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR",
    ),
    (&[], "INDEX_READ_BIT", "INDEX_INPUT_BIT"),
    (&[], "INDEX_READ_BIT", "VERTEX_INPUT_BIT"),
    (
        &[c"VK_KHR_acceleration_structure"],
        "INDIRECT_COMMAND_READ_BIT",
        "ACCELERATION_STRUCTURE_BUILD_BIT_KHR",
    ),
    (
        &[c"VK_NV_ray_tracing"],
        "INDIRECT_COMMAND_READ_BIT",
        "ACCELERATION_STRUCTURE_BUILD_BIT_KHR",
    ),
    (
        &[c"VK_KHR_copy_memory_indirect"],
        "INDIRECT_COMMAND_READ_BIT",
        "COPY_INDIRECT_BIT_KHR",
    ),
    (&[], "INDIRECT_COMMAND_READ_BIT", "DRAW_INDIRECT_BIT"),
    (&[], "INPUT_ATTACHMENT_READ_BIT", "FRAGMENT_SHADER_BIT"),
    (
        &[c"VK_HUAWEI_subpass_shading"],
        "INPUT_ATTACHMENT_READ_BIT",
        "SUBPASS_SHADER_BIT_HUAWEI",
    ),
    (
        &[c"VK_HUAWEI_invocation_mask"],
        "INVOCATION_MASK_READ_BIT_HUAWEI",
        "INVOCATION_MASK_BIT_HUAWEI",
    ),
    (
        &[c"VK_EXT_memory_decompression"],
        "MEMORY_DECOMPRESSION_READ_BIT_EXT",
        "MEMORY_DECOMPRESSION_BIT_EXT",
    ),
    (
        &[c"VK_EXT_opacity_micromap", c"VK_KHR_acceleration_structure"],
        "MICROMAP_READ_BIT_EXT",
        "ACCELERATION_STRUCTURE_BUILD_BIT_KHR",
    ),
    (
        &[c"VK_EXT_opacity_micromap", c"VK_NV_ray_tracing"],
        "MICROMAP_READ_BIT_EXT",
        "ACCELERATION_STRUCTURE_BUILD_BIT_KHR",
    ),
    (
        &[c"VK_EXT_opacity_micromap"],
        "MICROMAP_READ_BIT_EXT",
        "MICROMAP_BUILD_BIT_EXT",
    ),
    (
        &[c"VK_NV_optical_flow"],
        "OPTICAL_FLOW_READ_BIT_NV",
        "OPTICAL_FLOW_BIT_NV",
    ),
    (
        &[
            c"VK_EXT_descriptor_heap",
            c"VK_HUAWEI_cluster_culling_shader",
        ],
        "RESOURCE_HEAP_READ_BIT_EXT",
        "CLUSTER_CULLING_SHADER_BIT_HUAWEI",
    ),
    (
        &[c"VK_EXT_descriptor_heap"],
        "RESOURCE_HEAP_READ_BIT_EXT",
        "COMPUTE_SHADER_BIT",
    ),
    (
        &[c"VK_EXT_descriptor_heap"],
        "RESOURCE_HEAP_READ_BIT_EXT",
        "FRAGMENT_SHADER_BIT",
    ),
    (
        &[c"VK_EXT_descriptor_heap"],
        "RESOURCE_HEAP_READ_BIT_EXT",
        "GEOMETRY_SHADER_BIT",
    ),
    (
        &[c"VK_EXT_descriptor_heap", c"VK_EXT_mesh_shader"],
        "RESOURCE_HEAP_READ_BIT_EXT",
        "MESH_SHADER_BIT_EXT",
    ),
    (
        &[c"VK_EXT_descriptor_heap", c"VK_NV_mesh_shader"],
        "RESOURCE_HEAP_READ_BIT_EXT",
        "MESH_SHADER_BIT_EXT",
    ),
    (
        &[c"VK_EXT_descriptor_heap", c"VK_KHR_ray_tracing_pipeline"],
        "RESOURCE_HEAP_READ_BIT_EXT",
        "RAY_TRACING_SHADER_BIT_KHR",
    ),
    (
        &[c"VK_EXT_descriptor_heap", c"VK_NV_ray_tracing"],
        "RESOURCE_HEAP_READ_BIT_EXT",
        "RAY_TRACING_SHADER_BIT_KHR",
    ),
    (
        &[c"VK_EXT_descriptor_heap", c"VK_HUAWEI_subpass_shading"],
        "RESOURCE_HEAP_READ_BIT_EXT",
        "SUBPASS_SHADER_BIT_HUAWEI",
    ),
    (
        &[c"VK_EXT_descriptor_heap", c"VK_EXT_mesh_shader"],
        "RESOURCE_HEAP_READ_BIT_EXT",
        "TASK_SHADER_BIT_EXT",
    ),
    (
        &[c"VK_EXT_descriptor_heap", c"VK_NV_mesh_shader"],
        "RESOURCE_HEAP_READ_BIT_EXT",
        "TASK_SHADER_BIT_EXT",
    ),
    (
        &[c"VK_EXT_descriptor_heap"],
        "RESOURCE_HEAP_READ_BIT_EXT",
        "TESSELLATION_CONTROL_SHADER_BIT",
    ),
    (
        &[c"VK_EXT_descriptor_heap"],
        "RESOURCE_HEAP_READ_BIT_EXT",
        "TESSELLATION_EVALUATION_SHADER_BIT",
    ),
    (
        &[c"VK_EXT_descriptor_heap"],
        "RESOURCE_HEAP_READ_BIT_EXT",
        "VERTEX_SHADER_BIT",
    ),
    (
        &[
            c"VK_EXT_descriptor_heap",
            c"VK_HUAWEI_cluster_culling_shader",
        ],
        "SAMPLER_HEAP_READ_BIT_EXT",
        "CLUSTER_CULLING_SHADER_BIT_HUAWEI",
    ),
    (
        &[c"VK_EXT_descriptor_heap"],
        "SAMPLER_HEAP_READ_BIT_EXT",
        "COMPUTE_SHADER_BIT",
    ),
    (
        &[c"VK_EXT_descriptor_heap"],
        "SAMPLER_HEAP_READ_BIT_EXT",
        "FRAGMENT_SHADER_BIT",
    ),
    (
        &[c"VK_EXT_descriptor_heap"],
        "SAMPLER_HEAP_READ_BIT_EXT",
        "GEOMETRY_SHADER_BIT",
    ),
    (
        &[c"VK_EXT_descriptor_heap", c"VK_EXT_mesh_shader"],
        "SAMPLER_HEAP_READ_BIT_EXT",
        "MESH_SHADER_BIT_EXT",
    ),
    (
        &[c"VK_EXT_descriptor_heap", c"VK_NV_mesh_shader"],
        "SAMPLER_HEAP_READ_BIT_EXT",
        "MESH_SHADER_BIT_EXT",
    ),
    (
        &[c"VK_EXT_descriptor_heap", c"VK_KHR_ray_tracing_pipeline"],
        "SAMPLER_HEAP_READ_BIT_EXT",
        "RAY_TRACING_SHADER_BIT_KHR",
    ),
    (
        &[c"VK_EXT_descriptor_heap", c"VK_NV_ray_tracing"],
        "SAMPLER_HEAP_READ_BIT_EXT",
        "RAY_TRACING_SHADER_BIT_KHR",
    ),
    (
        &[c"VK_EXT_descriptor_heap", c"VK_HUAWEI_subpass_shading"],
        "SAMPLER_HEAP_READ_BIT_EXT",
        "SUBPASS_SHADER_BIT_HUAWEI",
    ),
    (
        &[c"VK_EXT_descriptor_heap", c"VK_EXT_mesh_shader"],
        "SAMPLER_HEAP_READ_BIT_EXT",
        "TASK_SHADER_BIT_EXT",
    ),
    (
        &[c"VK_EXT_descriptor_heap", c"VK_NV_mesh_shader"],
        "SAMPLER_HEAP_READ_BIT_EXT",
        "TASK_SHADER_BIT_EXT",
    ),
    (
        &[c"VK_EXT_descriptor_heap"],
        "SAMPLER_HEAP_READ_BIT_EXT",
        "TESSELLATION_CONTROL_SHADER_BIT",
    ),
    (
        &[c"VK_EXT_descriptor_heap"],
        "SAMPLER_HEAP_READ_BIT_EXT",
        "TESSELLATION_EVALUATION_SHADER_BIT",
    ),
    (
        &[c"VK_EXT_descriptor_heap"],
        "SAMPLER_HEAP_READ_BIT_EXT",
        "VERTEX_SHADER_BIT",
    ),
    (
        &[
            c"VK_HUAWEI_cluster_culling_shader",
            c"VK_KHR_ray_tracing_maintenance1",
            c"VK_KHR_ray_tracing_pipeline",
        ],
        "SHADER_BINDING_TABLE_READ_BIT_KHR",
        "CLUSTER_CULLING_SHADER_BIT_HUAWEI",
    ),
    (
        &[
            c"VK_KHR_ray_tracing_maintenance1",
            c"VK_KHR_ray_tracing_pipeline",
        ],
        "SHADER_BINDING_TABLE_READ_BIT_KHR",
        "COMPUTE_SHADER_BIT",
    ),
    (
        &[
            c"VK_KHR_ray_tracing_maintenance1",
            c"VK_KHR_ray_tracing_pipeline",
        ],
        "SHADER_BINDING_TABLE_READ_BIT_KHR",
        "FRAGMENT_SHADER_BIT",
    ),
    (
        &[
            c"VK_KHR_ray_tracing_maintenance1",
            c"VK_KHR_ray_tracing_pipeline",
        ],
        "SHADER_BINDING_TABLE_READ_BIT_KHR",
        "GEOMETRY_SHADER_BIT",
    ),
    (
        &[
            c"VK_EXT_mesh_shader",
            c"VK_KHR_ray_tracing_maintenance1",
            c"VK_KHR_ray_tracing_pipeline",
        ],
        "SHADER_BINDING_TABLE_READ_BIT_KHR",
        "MESH_SHADER_BIT_EXT",
    ),
    (
        &[
            c"VK_KHR_ray_tracing_maintenance1",
            c"VK_KHR_ray_tracing_pipeline",
            c"VK_NV_mesh_shader",
        ],
        "SHADER_BINDING_TABLE_READ_BIT_KHR",
        "MESH_SHADER_BIT_EXT",
    ),
    (
        &[
            c"VK_KHR_ray_tracing_maintenance1",
            c"VK_KHR_ray_tracing_pipeline",
        ],
        "SHADER_BINDING_TABLE_READ_BIT_KHR",
        "RAY_TRACING_SHADER_BIT_KHR",
    ),
    (
        &[
            c"VK_HUAWEI_subpass_shading",
            c"VK_KHR_ray_tracing_maintenance1",
            c"VK_KHR_ray_tracing_pipeline",
        ],
        "SHADER_BINDING_TABLE_READ_BIT_KHR",
        "SUBPASS_SHADER_BIT_HUAWEI",
    ),
    (
        &[
            c"VK_EXT_mesh_shader",
            c"VK_KHR_ray_tracing_maintenance1",
            c"VK_KHR_ray_tracing_pipeline",
        ],
        "SHADER_BINDING_TABLE_READ_BIT_KHR",
        "TASK_SHADER_BIT_EXT",
    ),
    (
        &[
            c"VK_KHR_ray_tracing_maintenance1",
            c"VK_KHR_ray_tracing_pipeline",
            c"VK_NV_mesh_shader",
        ],
        "SHADER_BINDING_TABLE_READ_BIT_KHR",
        "TASK_SHADER_BIT_EXT",
    ),
    (
        &[
            c"VK_KHR_ray_tracing_maintenance1",
            c"VK_KHR_ray_tracing_pipeline",
        ],
        "SHADER_BINDING_TABLE_READ_BIT_KHR",
        "TESSELLATION_CONTROL_SHADER_BIT",
    ),
    (
        &[
            c"VK_KHR_ray_tracing_maintenance1",
            c"VK_KHR_ray_tracing_pipeline",
        ],
        "SHADER_BINDING_TABLE_READ_BIT_KHR",
        "TESSELLATION_EVALUATION_SHADER_BIT",
    ),
    (
        &[
            c"VK_KHR_ray_tracing_maintenance1",
            c"VK_KHR_ray_tracing_pipeline",
        ],
        "SHADER_BINDING_TABLE_READ_BIT_KHR",
        "VERTEX_SHADER_BIT",
    ),
    (
        &[c"VK_KHR_acceleration_structure"],
        "SHADER_READ_BIT",
        "ACCELERATION_STRUCTURE_BUILD_BIT_KHR",
    ),
    (
        &[c"VK_NV_ray_tracing"],
        "SHADER_READ_BIT",
        "ACCELERATION_STRUCTURE_BUILD_BIT_KHR",
    ),
    (
        &[c"VK_HUAWEI_cluster_culling_shader"],
        "SHADER_READ_BIT",
        "CLUSTER_CULLING_SHADER_BIT_HUAWEI",
    ),
    (&[], "SHADER_READ_BIT", "COMPUTE_SHADER_BIT"),
    (&[], "SHADER_READ_BIT", "FRAGMENT_SHADER_BIT"),
    (&[], "SHADER_READ_BIT", "GEOMETRY_SHADER_BIT"),
    (
        &[c"VK_EXT_mesh_shader"],
        "SHADER_READ_BIT",
        "MESH_SHADER_BIT_EXT",
    ),
    (
        &[c"VK_NV_mesh_shader"],
        "SHADER_READ_BIT",
        "MESH_SHADER_BIT_EXT",
    ),
    (
        &[c"VK_EXT_opacity_micromap"],
        "SHADER_READ_BIT",
        "MICROMAP_BUILD_BIT_EXT",
    ),
    (
        &[c"VK_KHR_ray_tracing_pipeline"],
        "SHADER_READ_BIT",
        "RAY_TRACING_SHADER_BIT_KHR",
    ),
    (
        &[c"VK_NV_ray_tracing"],
        "SHADER_READ_BIT",
        "RAY_TRACING_SHADER_BIT_KHR",
    ),
    (
        &[c"VK_HUAWEI_subpass_shading"],
        "SHADER_READ_BIT",
        "SUBPASS_SHADER_BIT_HUAWEI",
    ),
    (
        &[c"VK_EXT_mesh_shader"],
        "SHADER_READ_BIT",
        "TASK_SHADER_BIT_EXT",
    ),
    (
        &[c"VK_NV_mesh_shader"],
        "SHADER_READ_BIT",
        "TASK_SHADER_BIT_EXT",
    ),
    (&[], "SHADER_READ_BIT", "TESSELLATION_CONTROL_SHADER_BIT"),
    (&[], "SHADER_READ_BIT", "TESSELLATION_EVALUATION_SHADER_BIT"),
    (&[], "SHADER_READ_BIT", "VERTEX_SHADER_BIT"),
    (
        &[c"VK_HUAWEI_cluster_culling_shader"],
        "SHADER_SAMPLED_READ_BIT",
        "CLUSTER_CULLING_SHADER_BIT_HUAWEI",
    ),
    (&[], "SHADER_SAMPLED_READ_BIT", "COMPUTE_SHADER_BIT"),
    (&[], "SHADER_SAMPLED_READ_BIT", "FRAGMENT_SHADER_BIT"),
    (&[], "SHADER_SAMPLED_READ_BIT", "GEOMETRY_SHADER_BIT"),
    (
        &[c"VK_EXT_mesh_shader"],
        "SHADER_SAMPLED_READ_BIT",
        "MESH_SHADER_BIT_EXT",
    ),
    (
        &[c"VK_NV_mesh_shader"],
        "SHADER_SAMPLED_READ_BIT",
        "MESH_SHADER_BIT_EXT",
    ),
    (
        &[c"VK_KHR_ray_tracing_pipeline"],
        "SHADER_SAMPLED_READ_BIT",
        "RAY_TRACING_SHADER_BIT_KHR",
    ),
    (
        &[c"VK_NV_ray_tracing"],
        "SHADER_SAMPLED_READ_BIT",
        "RAY_TRACING_SHADER_BIT_KHR",
    ),
    (
        &[c"VK_HUAWEI_subpass_shading"],
        "SHADER_SAMPLED_READ_BIT",
        "SUBPASS_SHADER_BIT_HUAWEI",
    ),
    (
        &[c"VK_EXT_mesh_shader"],
        "SHADER_SAMPLED_READ_BIT",
        "TASK_SHADER_BIT_EXT",
    ),
    (
        &[c"VK_NV_mesh_shader"],
        "SHADER_SAMPLED_READ_BIT",
        "TASK_SHADER_BIT_EXT",
    ),
    (
        &[],
        "SHADER_SAMPLED_READ_BIT",
        "TESSELLATION_CONTROL_SHADER_BIT",
    ),
    (
        &[],
        "SHADER_SAMPLED_READ_BIT",
        "TESSELLATION_EVALUATION_SHADER_BIT",
    ),
    (&[], "SHADER_SAMPLED_READ_BIT", "VERTEX_SHADER_BIT"),
    (
        &[c"VK_HUAWEI_cluster_culling_shader"],
        "SHADER_STORAGE_READ_BIT",
        "CLUSTER_CULLING_SHADER_BIT_HUAWEI",
    ),
    (&[], "SHADER_STORAGE_READ_BIT", "COMPUTE_SHADER_BIT"),
    (&[], "SHADER_STORAGE_READ_BIT", "FRAGMENT_SHADER_BIT"),
    (&[], "SHADER_STORAGE_READ_BIT", "GEOMETRY_SHADER_BIT"),
    (
        &[c"VK_EXT_mesh_shader"],
        "SHADER_STORAGE_READ_BIT",
        "MESH_SHADER_BIT_EXT",
    ),
    (
        &[c"VK_NV_mesh_shader"],
        "SHADER_STORAGE_READ_BIT",
        "MESH_SHADER_BIT_EXT",
    ),
    (
        &[c"VK_KHR_ray_tracing_pipeline"],
        "SHADER_STORAGE_READ_BIT",
        "RAY_TRACING_SHADER_BIT_KHR",
    ),
    (
        &[c"VK_NV_ray_tracing"],
        "SHADER_STORAGE_READ_BIT",
        "RAY_TRACING_SHADER_BIT_KHR",
    ),
    (
        &[c"VK_HUAWEI_subpass_shading"],
        "SHADER_STORAGE_READ_BIT",
        "SUBPASS_SHADER_BIT_HUAWEI",
    ),
    (
        &[c"VK_EXT_mesh_shader"],
        "SHADER_STORAGE_READ_BIT",
        "TASK_SHADER_BIT_EXT",
    ),
    (
        &[c"VK_NV_mesh_shader"],
        "SHADER_STORAGE_READ_BIT",
        "TASK_SHADER_BIT_EXT",
    ),
    (
        &[],
        "SHADER_STORAGE_READ_BIT",
        "TESSELLATION_CONTROL_SHADER_BIT",
    ),
    (
        &[],
        "SHADER_STORAGE_READ_BIT",
        "TESSELLATION_EVALUATION_SHADER_BIT",
    ),
    (&[], "SHADER_STORAGE_READ_BIT", "VERTEX_SHADER_BIT"),
    (
        &[c"VK_QCOM_tile_shading"],
        "SHADER_TILE_ATTACHMENT_READ_BIT_QCOM",
        "COMPUTE_SHADER_BIT",
    ),
    (
        &[c"VK_QCOM_tile_shading"],
        "SHADER_TILE_ATTACHMENT_READ_BIT_QCOM",
        "FRAGMENT_SHADER_BIT",
    ),
    (
        &[c"VK_KHR_acceleration_structure"],
        "TRANSFER_READ_BIT",
        "ACCELERATION_STRUCTURE_BUILD_BIT_KHR",
    ),
    (
        &[c"VK_NV_ray_tracing"],
        "TRANSFER_READ_BIT",
        "ACCELERATION_STRUCTURE_BUILD_BIT_KHR",
    ),
    (
        &[c"VK_KHR_ray_tracing_maintenance1"],
        "TRANSFER_READ_BIT",
        "ACCELERATION_STRUCTURE_COPY_BIT_KHR",
    ),
    (&[], "TRANSFER_READ_BIT", "ALL_TRANSFER_BIT"),
    (&[], "TRANSFER_READ_BIT", "BLIT_BIT"),
    (
        &[c"VK_NV_cooperative_vector"],
        "TRANSFER_READ_BIT",
        "CONVERT_COOPERATIVE_VECTOR_MATRIX_BIT_NV",
    ),
    (&[], "TRANSFER_READ_BIT", "COPY_BIT"),
    (
        &[c"VK_EXT_opacity_micromap"],
        "TRANSFER_READ_BIT",
        "MICROMAP_BUILD_BIT_EXT",
    ),
    (&[], "TRANSFER_READ_BIT", "RESOLVE_BIT"),
    (
        &[c"VK_EXT_transform_feedback"],
        "TRANSFORM_FEEDBACK_COUNTER_READ_BIT_EXT",
        "DRAW_INDIRECT_BIT",
    ),
    (
        &[c"VK_EXT_transform_feedback"],
        "TRANSFORM_FEEDBACK_COUNTER_READ_BIT_EXT",
        "TRANSFORM_FEEDBACK_BIT_EXT",
    ),
    (
        &[c"VK_HUAWEI_cluster_culling_shader"],
        "UNIFORM_READ_BIT",
        "CLUSTER_CULLING_SHADER_BIT_HUAWEI",
    ),
    (&[], "UNIFORM_READ_BIT", "COMPUTE_SHADER_BIT"),
    (&[], "UNIFORM_READ_BIT", "FRAGMENT_SHADER_BIT"),
    (&[], "UNIFORM_READ_BIT", "GEOMETRY_SHADER_BIT"),
    (
        &[c"VK_EXT_mesh_shader"],
        "UNIFORM_READ_BIT",
        "MESH_SHADER_BIT_EXT",
    ),
    (
        &[c"VK_NV_mesh_shader"],
        "UNIFORM_READ_BIT",
        "MESH_SHADER_BIT_EXT",
    ),
    (
        &[c"VK_KHR_ray_tracing_pipeline"],
        "UNIFORM_READ_BIT",
        "RAY_TRACING_SHADER_BIT_KHR",
    ),
    (
        &[c"VK_NV_ray_tracing"],
        "UNIFORM_READ_BIT",
        "RAY_TRACING_SHADER_BIT_KHR",
    ),
    (
        &[c"VK_HUAWEI_subpass_shading"],
        "UNIFORM_READ_BIT",
        "SUBPASS_SHADER_BIT_HUAWEI",
    ),
    (
        &[c"VK_EXT_mesh_shader"],
        "UNIFORM_READ_BIT",
        "TASK_SHADER_BIT_EXT",
    ),
    (
        &[c"VK_NV_mesh_shader"],
        "UNIFORM_READ_BIT",
        "TASK_SHADER_BIT_EXT",
    ),
    (&[], "UNIFORM_READ_BIT", "TESSELLATION_CONTROL_SHADER_BIT"),
    (
        &[],
        "UNIFORM_READ_BIT",
        "TESSELLATION_EVALUATION_SHADER_BIT",
    ),
    (&[], "UNIFORM_READ_BIT", "VERTEX_SHADER_BIT"),
    (
        &[],
        "VERTEX_ATTRIBUTE_READ_BIT",
        "VERTEX_ATTRIBUTE_INPUT_BIT",
    ),
    (&[], "VERTEX_ATTRIBUTE_READ_BIT", "VERTEX_INPUT_BIT"),
    (
        &[c"VK_KHR_video_decode_queue"],
        "VIDEO_DECODE_READ_BIT_KHR",
        "VIDEO_DECODE_BIT_KHR",
    ),
    (
        &[c"VK_KHR_video_encode_queue"],
        "VIDEO_ENCODE_READ_BIT_KHR",
        "VIDEO_ENCODE_BIT_KHR",
    ),
];

/// Each stage (`syncstage`): the queue flags of the families that run it (`syncsupport`; none for
/// a stage no queue limits), the extensions it needs (alternatives; empty for the core's), and
/// the stages it is equivalent to (`syncequivalent`).
pub(super) const STAGES: [(&str, u32, &[&[&CStr]], &[&str]); 46] = [
    ("NONE", 0x0, &[&[]], &[]),
    ("TOP_OF_PIPE_BIT", 0x0, &[&[]], &[]),
    ("DRAW_INDIRECT_BIT", 0x3, &[&[]], &[]),
    (
        "COPY_INDIRECT_BIT_KHR",
        0x7,
        &[&[c"VK_KHR_copy_memory_indirect"]],
        &[],
    ),
    (
        "VERTEX_INPUT_BIT",
        0x1,
        &[&[]],
        &["INDEX_INPUT_BIT", "VERTEX_ATTRIBUTE_INPUT_BIT"],
    ),
    ("VERTEX_SHADER_BIT", 0x1, &[&[]], &[]),
    ("TESSELLATION_CONTROL_SHADER_BIT", 0x1, &[&[]], &[]),
    ("TESSELLATION_EVALUATION_SHADER_BIT", 0x1, &[&[]], &[]),
    ("GEOMETRY_SHADER_BIT", 0x1, &[&[]], &[]),
    ("FRAGMENT_SHADER_BIT", 0x1, &[&[]], &[]),
    ("EARLY_FRAGMENT_TESTS_BIT", 0x1, &[&[]], &[]),
    ("LATE_FRAGMENT_TESTS_BIT", 0x1, &[&[]], &[]),
    ("COLOR_ATTACHMENT_OUTPUT_BIT", 0x1, &[&[]], &[]),
    ("COMPUTE_SHADER_BIT", 0x2, &[&[]], &[]),
    (
        "ALL_TRANSFER_BIT",
        0x7,
        &[&[]],
        &[
            "COPY_BIT",
            "BLIT_BIT",
            "RESOLVE_BIT",
            "CLEAR_BIT",
            "ACCELERATION_STRUCTURE_COPY_BIT_KHR",
        ],
    ),
    ("BOTTOM_OF_PIPE_BIT", 0x0, &[&[]], &[]),
    ("HOST_BIT", 0x0, &[&[]], &[]),
    (
        "ALL_GRAPHICS_BIT",
        0x1,
        &[&[]],
        &[
            "DRAW_INDIRECT_BIT",
            "COPY_INDIRECT_BIT_KHR",
            "TASK_SHADER_BIT_EXT",
            "MESH_SHADER_BIT_EXT",
            "VERTEX_INPUT_BIT",
            "VERTEX_SHADER_BIT",
            "TESSELLATION_CONTROL_SHADER_BIT",
            "TESSELLATION_EVALUATION_SHADER_BIT",
            "GEOMETRY_SHADER_BIT",
            "FRAGMENT_SHADER_BIT",
            "EARLY_FRAGMENT_TESTS_BIT",
            "LATE_FRAGMENT_TESTS_BIT",
            "COLOR_ATTACHMENT_OUTPUT_BIT",
            "CONDITIONAL_RENDERING_BIT_EXT",
            "TRANSFORM_FEEDBACK_BIT_EXT",
            "FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR",
            "FRAGMENT_DENSITY_PROCESS_BIT_EXT",
            "SUBPASS_SHADER_BIT_HUAWEI",
            "INVOCATION_MASK_BIT_HUAWEI",
            "CLUSTER_CULLING_SHADER_BIT_HUAWEI",
        ],
    ),
    ("ALL_COMMANDS_BIT", 0x0, &[&[]], &[]),
    ("COPY_BIT", 0x7, &[&[]], &[]),
    ("RESOLVE_BIT", 0x7, &[&[]], &[]),
    ("BLIT_BIT", 0x7, &[&[]], &[]),
    ("CLEAR_BIT", 0x7, &[&[]], &[]),
    ("INDEX_INPUT_BIT", 0x1, &[&[]], &[]),
    ("VERTEX_ATTRIBUTE_INPUT_BIT", 0x1, &[&[]], &[]),
    (
        "PRE_RASTERIZATION_SHADERS_BIT",
        0x1,
        &[&[]],
        &[
            "VERTEX_SHADER_BIT",
            "TESSELLATION_CONTROL_SHADER_BIT",
            "TESSELLATION_EVALUATION_SHADER_BIT",
            "GEOMETRY_SHADER_BIT",
            "TASK_SHADER_BIT_EXT",
            "MESH_SHADER_BIT_EXT",
            "CLUSTER_CULLING_SHADER_BIT_HUAWEI",
        ],
    ),
    (
        "VIDEO_DECODE_BIT_KHR",
        0x20,
        &[&[c"VK_KHR_video_decode_queue"]],
        &[],
    ),
    (
        "VIDEO_ENCODE_BIT_KHR",
        0x40,
        &[&[c"VK_KHR_video_encode_queue"]],
        &[],
    ),
    (
        "TRANSFORM_FEEDBACK_BIT_EXT",
        0x1,
        &[&[c"VK_EXT_transform_feedback"]],
        &[],
    ),
    (
        "CONDITIONAL_RENDERING_BIT_EXT",
        0x3,
        &[&[c"VK_EXT_conditional_rendering"]],
        &[],
    ),
    (
        "COMMAND_PREPROCESS_BIT_EXT",
        0x3,
        &[
            &[c"VK_EXT_device_generated_commands"],
            &[c"VK_NV_device_generated_commands"],
        ],
        &[],
    ),
    (
        "MEMORY_DECOMPRESSION_BIT_EXT",
        0x3,
        &[&[c"VK_EXT_memory_decompression"]],
        &[],
    ),
    (
        "FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR",
        0x1,
        &[
            &[c"VK_KHR_fragment_shading_rate"],
            &[c"VK_NV_shading_rate_image"],
        ],
        &[],
    ),
    (
        "ACCELERATION_STRUCTURE_BUILD_BIT_KHR",
        0x2,
        &[&[c"VK_KHR_acceleration_structure"], &[c"VK_NV_ray_tracing"]],
        &[],
    ),
    (
        "RAY_TRACING_SHADER_BIT_KHR",
        0x2,
        &[&[c"VK_KHR_ray_tracing_pipeline"], &[c"VK_NV_ray_tracing"]],
        &[],
    ),
    (
        "FRAGMENT_DENSITY_PROCESS_BIT_EXT",
        0x1,
        &[&[c"VK_EXT_fragment_density_map"]],
        &[],
    ),
    (
        "TASK_SHADER_BIT_EXT",
        0x1,
        &[&[c"VK_EXT_mesh_shader"], &[c"VK_NV_mesh_shader"]],
        &[],
    ),
    (
        "MESH_SHADER_BIT_EXT",
        0x1,
        &[&[c"VK_EXT_mesh_shader"], &[c"VK_NV_mesh_shader"]],
        &[],
    ),
    (
        "SUBPASS_SHADER_BIT_HUAWEI",
        0x1,
        &[&[c"VK_HUAWEI_subpass_shading"]],
        &[],
    ),
    (
        "INVOCATION_MASK_BIT_HUAWEI",
        0x1,
        &[&[c"VK_HUAWEI_invocation_mask"]],
        &[],
    ),
    (
        "ACCELERATION_STRUCTURE_COPY_BIT_KHR",
        0x7,
        &[&[c"VK_KHR_ray_tracing_maintenance1"]],
        &[],
    ),
    (
        "MICROMAP_BUILD_BIT_EXT",
        0x2,
        &[&[c"VK_EXT_opacity_micromap"]],
        &[],
    ),
    (
        "CLUSTER_CULLING_SHADER_BIT_HUAWEI",
        0x1,
        &[&[c"VK_HUAWEI_cluster_culling_shader"]],
        &[],
    ),
    (
        "OPTICAL_FLOW_BIT_NV",
        0x100,
        &[&[c"VK_NV_optical_flow"]],
        &[],
    ),
    (
        "CONVERT_COOPERATIVE_VECTOR_MATRIX_BIT_NV",
        0x7,
        &[&[c"VK_NV_cooperative_vector"]],
        &[],
    ),
    ("DATA_GRAPH_BIT_ARM", 0x400, &[&[c"VK_ARM_data_graph"]], &[]),
];

/// Each extension a row above names, and the extensions it requires.
pub(super) const DEPENDS: [(&CStr, &[&CStr]); 28] = [
    (
        c"VK_ARM_data_graph",
        &[c"VK_KHR_deferred_host_operations", c"VK_KHR_maintenance5"],
    ),
    (c"VK_EXT_blend_operation_advanced", &[]),
    (c"VK_EXT_conditional_rendering", &[]),
    (c"VK_EXT_descriptor_buffer", &[]),
    (c"VK_EXT_descriptor_heap", &[c"VK_KHR_maintenance5"]),
    (c"VK_EXT_device_generated_commands", &[]),
    (c"VK_EXT_fragment_density_map", &[]),
    (
        c"VK_EXT_memory_decompression",
        &[
            c"VK_KHR_buffer_device_address",
            c"VK_KHR_get_physical_device_properties2",
        ],
    ),
    (c"VK_EXT_mesh_shader", &[]),
    (
        c"VK_EXT_opacity_micromap",
        &[
            c"VK_KHR_acceleration_structure",
            c"VK_KHR_deferred_host_operations",
        ],
    ),
    (c"VK_EXT_transform_feedback", &[]),
    (c"VK_HUAWEI_cluster_culling_shader", &[]),
    (
        c"VK_HUAWEI_invocation_mask",
        &[
            c"VK_KHR_acceleration_structure",
            c"VK_KHR_deferred_host_operations",
            c"VK_KHR_ray_tracing_pipeline",
        ],
    ),
    (c"VK_HUAWEI_subpass_shading", &[]),
    (
        c"VK_KHR_acceleration_structure",
        &[c"VK_KHR_deferred_host_operations"],
    ),
    (c"VK_KHR_copy_memory_indirect", &[]),
    (c"VK_KHR_fragment_shading_rate", &[]),
    (
        c"VK_KHR_ray_tracing_maintenance1",
        &[
            c"VK_KHR_acceleration_structure",
            c"VK_KHR_deferred_host_operations",
        ],
    ),
    (
        c"VK_KHR_ray_tracing_pipeline",
        &[
            c"VK_KHR_acceleration_structure",
            c"VK_KHR_deferred_host_operations",
        ],
    ),
    (c"VK_KHR_video_decode_queue", &[c"VK_KHR_video_queue"]),
    (c"VK_KHR_video_encode_queue", &[c"VK_KHR_video_queue"]),
    (c"VK_NV_cooperative_vector", &[]),
    (c"VK_NV_device_generated_commands", &[]),
    (c"VK_NV_mesh_shader", &[]),
    (c"VK_NV_optical_flow", &[]),
    (c"VK_NV_ray_tracing", &[]),
    (c"VK_NV_shading_rate_image", &[]),
    (c"VK_QCOM_tile_shading", &[c"VK_QCOM_tile_properties"]),
];
