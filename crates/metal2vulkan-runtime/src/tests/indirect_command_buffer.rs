//! Indirect command buffers (`indirect_command_buffer.rs`): what needs no device.

use crate::indirect_command_buffer::metal_allocated_size;
use crate::IndirectCommandBufferDescriptor;
use metal2vulkan::metal::IndirectCommandType;

/// A fresh descriptor answers as Metal's (an M3 Max and an M4): no command types, neither the
/// pipeline state nor the buffers inherited, no buffers, 31 threadgroup lengths, no dynamic
/// strides; and as the macOS 26 SDK states: every render state inherited, no object threadgroup
/// lengths, no color attachment mapping.
#[test]
fn a_descriptor_has_metal_s_defaults() {
    let descriptor = IndirectCommandBufferDescriptor::new();
    assert!(descriptor.command_types.is_empty());
    assert!(!descriptor.inherit_pipeline_state && !descriptor.inherit_buffers);
    assert!(
        descriptor.inherit_depth_stencil_state
            && descriptor.inherit_depth_bias
            && descriptor.inherit_depth_clip_mode
            && descriptor.inherit_cull_mode
            && descriptor.inherit_front_facing_winding
            && descriptor.inherit_triangle_fill_mode
    );
    assert_eq!(descriptor.max_object_threadgroup_memory_bind_count, 0);
    assert!(!descriptor.support_color_attachment_mapping);
    assert_eq!(
        [
            descriptor.max_vertex_buffer_bind_count,
            descriptor.max_fragment_buffer_bind_count,
            descriptor.max_kernel_buffer_bind_count,
            descriptor.max_kernel_threadgroup_memory_bind_count,
        ],
        [0, 0, 0, 31]
    );
    assert!(!descriptor.support_dynamic_attribute_stride);
    // Apple's private properties, as a fresh descriptor answers on an M4 (plain and validated
    // alike).
    assert!(
        descriptor.inherit_stencil_reference_values
            && descriptor.inherit_depth_test_bounds
            && descriptor.inherit_scissor_rects
            && descriptor.inherit_viewports
            && descriptor.inherit_blend_color
    );
    assert_eq!(
        (
            descriptor.max_scissor_rect_count,
            descriptor.max_viewport_count,
            descriptor.allow_override_render_states
        ),
        (1, 1, 0)
    );
}

/// `allocatedSize` is Metal's: rows measured on an M4, at a page's last count and the next where it
/// turns. Each row: the command types, whether the pipeline and the buffers are inherited, dynamic
/// strides, the vertex, fragment, kernel, object and mesh buffer counts, the count of commands, and
/// Metal's answer. Every kind answers one page for no commands.
#[test]
fn a_buffer_answers_metal_s_allocated_size() {
    type Row = (u32, bool, bool, bool, [u32; 5], u32, u64);
    let rows: [Row; 47] = [
        (32, false, false, true, [0, 0, 5, 0, 0], 1000, 802816),
        (64, true, false, false, [0, 0, 31, 0, 0], 1000, 966656),
        (96, false, true, true, [0, 0, 5, 0, 0], 511, 376832),
        (96, false, true, true, [0, 0, 5, 0, 0], 512, 393216),
        (96, false, true, true, [0, 0, 5, 0, 0], 1000, 737280),
        (1, false, false, true, [3, 5, 0, 0, 0], 15, 16384),
        (1, false, false, true, [3, 5, 0, 0, 0], 16, 32768),
        (1, false, false, true, [3, 5, 0, 0, 0], 1000, 1032192),
        (2, true, false, false, [31, 31, 0, 0, 0], 1000, 1196032),
        (4, false, false, true, [1, 0, 0, 0, 0], 15, 16384),
        (4, false, false, true, [1, 0, 0, 0, 0], 16, 32768),
        (4, false, false, true, [1, 0, 0, 0, 0], 1000, 1015808),
        (8, true, false, false, [0, 2, 0, 0, 0], 1000, 737280),
        (128, false, false, false, [0, 0, 0, 0, 0], 9, 16384),
        (128, false, false, false, [0, 0, 0, 0, 0], 10, 32768),
        (128, false, false, false, [0, 0, 0, 0, 0], 1000, 1671168),
        (256, true, false, false, [2, 3, 0, 4, 4], 11, 16384),
        (256, true, false, false, [2, 3, 0, 4, 4], 12, 32768),
        (256, true, false, false, [2, 3, 0, 4, 4], 1000, 1376256),
        (129, false, false, false, [2, 3, 0, 4, 4], 14, 32768),
        (129, false, false, false, [2, 3, 0, 4, 4], 15, 49152),
        (129, false, false, false, [2, 3, 0, 4, 4], 1000, 2179072),
        (132, true, false, false, [0, 0, 0, 7, 0], 9, 16384),
        (132, true, false, false, [0, 0, 0, 7, 0], 10, 32768),
        (132, true, false, false, [0, 0, 0, 7, 0], 1000, 1654784),
        (128, false, false, true, [5, 0, 0, 0, 0], 9, 16384),
        (128, false, false, true, [5, 0, 0, 0, 0], 10, 32768),
        (128, false, false, true, [5, 0, 0, 0, 0], 1000, 1687552),
        // A mesh type's odd count with dynamic strides and its own buffers: 4 bytes more,
        // past a page where the even count ends on one.
        (128, false, false, true, [0, 0, 0, 0, 0], 354, 589824),
        (128, false, false, true, [0, 0, 0, 0, 0], 355, 606208),
        (129, false, false, true, [2, 3, 0, 0, 0], 434, 917504),
        (129, false, false, true, [2, 3, 0, 0, 0], 435, 933888),
        (0, false, false, false, [0, 0, 0, 0, 0], 31, 16384),
        (0, false, false, false, [0, 0, 0, 0, 0], 32, 32768),
        (0, false, false, false, [0, 0, 0, 0, 0], 1000, 524288),
        (0, true, false, true, [5, 0, 0, 0, 0], 43, 16384),
        (0, true, false, true, [5, 0, 0, 0, 0], 44, 32768),
        (0, true, false, true, [5, 0, 0, 0, 0], 1000, 376832),
        (0, false, false, false, [0, 6, 0, 0, 0], 28, 16384),
        (0, false, false, false, [0, 6, 0, 0, 0], 29, 32768),
        (0, false, false, false, [0, 6, 0, 0, 0], 1000, 557056),
        (33, false, false, false, [0, 0, 5, 0, 0], 20, 16384),
        (33, false, false, false, [0, 0, 5, 0, 0], 21, 32768),
        (33, false, false, false, [0, 0, 5, 0, 0], 1000, 786432),
        (160, true, false, true, [3, 4, 5, 6, 7], 21, 16384),
        (160, true, false, true, [3, 4, 5, 6, 7], 22, 32768),
        (160, true, false, true, [3, 4, 5, 6, 7], 1000, 770048),
    ];
    for (
        types,
        pipeline,
        buffers,
        strides,
        [vertex, fragment, kernel, object, mesh],
        count,
        size,
    ) in rows
    {
        let mut descriptor = IndirectCommandBufferDescriptor::new();
        descriptor.command_types = IndirectCommandType::try_from(types).unwrap();
        descriptor.inherit_pipeline_state = pipeline;
        descriptor.inherit_buffers = buffers;
        descriptor.support_dynamic_attribute_stride = strides;
        descriptor.max_vertex_buffer_bind_count = vertex;
        descriptor.max_fragment_buffer_bind_count = fragment;
        descriptor.max_kernel_buffer_bind_count = kernel;
        descriptor.max_object_buffer_bind_count = object;
        descriptor.max_mesh_buffer_bind_count = mesh;
        assert_eq!(
            metal_allocated_size(&descriptor, count),
            size,
            "types {types}, {count} commands"
        );
        assert_eq!(metal_allocated_size(&descriptor, 0), 16384, "types {types}");
    }
}
