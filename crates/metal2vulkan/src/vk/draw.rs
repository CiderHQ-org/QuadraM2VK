//! `vk::DrawCommand`: the Vulkan draw call a draw plan instantiates to.
//!
//! Hand-written plain data, owned (the binder's draw output); the binder produces it and
//! `ash_interop` may add conversions. The buffers themselves are the consumer's: indirect and index
//! buffers are named by the `DrawCall` the consumer passed in.

use super::IndexType;

/// One Vulkan draw command, with the parameters of the matching `vkCmdDraw*` call.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DrawCommand {
    /// `vkCmdDraw`.
    Draw {
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    },
    /// `vkCmdBindIndexBuffer` + `vkCmdDrawIndexed`.
    DrawIndexed {
        index_type: IndexType,
        /// Byte offset of the index buffer binding.
        index_buffer_offset: u64,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    },
    /// `vkCmdDrawIndirect`, reading `VkDrawIndirectCommand` from the indirect buffer (layout-equal
    /// to `MTLDrawPrimitivesIndirectArguments`).
    DrawIndirect {
        offset: u64,
        draw_count: u32,
        stride: u32,
    },
    /// `vkCmdBindIndexBuffer` + `vkCmdDrawIndexedIndirect` (layout-equal to
    /// `MTLDrawIndexedPrimitivesIndirectArguments`).
    DrawIndexedIndirect {
        index_type: IndexType,
        index_buffer_offset: u64,
        offset: u64,
        draw_count: u32,
        stride: u32,
    },
    /// `vkCmdDrawMeshTasksEXT` (`VK_EXT_mesh_shader`) of a mesh render pipeline.
    DrawMeshTasks {
        group_count_x: u32,
        group_count_y: u32,
        group_count_z: u32,
    },
    /// `vkCmdDrawMeshTasksIndirectEXT`, reading `VkDrawMeshTasksIndirectCommandEXT` from the
    /// indirect buffer (layout-equal to `MTLDispatchThreadgroupsIndirectArguments`).
    DrawMeshTasksIndirect {
        offset: u64,
        draw_count: u32,
        stride: u32,
    },
}
