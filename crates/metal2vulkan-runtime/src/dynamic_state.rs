//! The commands that record a render pipeline's extended dynamic states: Vulkan 1.3's core entry
//! points on a device at 1.3 or later, where the functionality is core without a feature, else
//! `VK_EXT_extended_dynamic_state`'s where the device enabled it, with
//! `VK_EXT_extended_dynamic_state2`'s primitive restart where the device enabled that too. vk.xml
//! aliases each command of the extensions to the core one, so the tables hold the same signatures.
//!
//! A render pipeline makes these states dynamic only when the compile device has the
//! `extendedDynamicState` feature (the restart: `extendedDynamicState2`), which it has exactly
//! when one of the two sources is present (`device::enables`, `Implied`); a draw of such a
//! pipeline on a device without the commands is an internal error, never a guess.

use crate::recorder::{through, Command, Slot};
use ash::vk;
use metal2vulkan::device::{EnableSet, VkExtension, VkFeature};
use metal2vulkan::vk::ApiVersion;

/// The extended dynamic state commands a device has.
#[derive(Clone, Copy)]
pub(super) struct ExtendedDynamicState {
    pub(super) cull_mode: vk::PFN_vkCmdSetCullMode,
    pub(super) front_face: vk::PFN_vkCmdSetFrontFace,
    pub(super) primitive_topology: vk::PFN_vkCmdSetPrimitiveTopology,
    pub(super) depth_test_enable: vk::PFN_vkCmdSetDepthTestEnable,
    pub(super) depth_write_enable: vk::PFN_vkCmdSetDepthWriteEnable,
    pub(super) depth_compare_op: vk::PFN_vkCmdSetDepthCompareOp,
    pub(super) stencil_test_enable: vk::PFN_vkCmdSetStencilTestEnable,
    pub(super) stencil_op: vk::PFN_vkCmdSetStencilOp,
    pub(super) bind_vertex_buffers2: vk::PFN_vkCmdBindVertexBuffers2,
    /// `vkCmdSetPrimitiveRestartEnable`, Vulkan 1.3's core command or
    /// `VK_EXT_extended_dynamic_state2`'s where the device enabled it, which a render pipeline
    /// makes dynamic only where the device has `extendedDynamicState2`.
    pub(super) primitive_restart_enable: Option<vk::PFN_vkCmdSetPrimitiveRestartEnable>,
}

impl ExtendedDynamicState {
    /// The commands of a device of effective version `api` (the instance's, at most the
    /// device's) with `enabled` enabled: the core ones from 1.3, else the extensions', else
    /// none. The core ones are `device`'s functions', routed through `slot` where the device has
    /// one (`recorder`); the extensions' are routed through it here.
    ///
    /// # Safety
    ///
    /// `device` was made from `instance` with `enabled` enabled, its functions routed through
    /// `slot` where it is set, and the loader serves `vkGetDeviceProcAddr` for it (its loader
    /// word is in place).
    pub(super) unsafe fn load(
        instance: &ash::Instance,
        device: &ash::Device,
        api: ApiVersion,
        enabled: &EnableSet,
        slot: Option<&Slot>,
    ) -> Option<ExtendedDynamicState> {
        if api >= ApiVersion::V1_3 {
            let f = device.fp_v1_3();
            return Some(ExtendedDynamicState {
                cull_mode: f.cmd_set_cull_mode,
                front_face: f.cmd_set_front_face,
                primitive_topology: f.cmd_set_primitive_topology,
                depth_test_enable: f.cmd_set_depth_test_enable,
                depth_write_enable: f.cmd_set_depth_write_enable,
                depth_compare_op: f.cmd_set_depth_compare_op,
                stencil_test_enable: f.cmd_set_stencil_test_enable,
                stencil_op: f.cmd_set_stencil_op,
                bind_vertex_buffers2: f.cmd_bind_vertex_buffers2,
                primitive_restart_enable: Some(f.cmd_set_primitive_restart_enable),
            });
        }
        if !enabled
            .extensions
            .contains(&VkExtension::ExtExtendedDynamicState)
        {
            return None;
        }
        let extension = ash::ext::extended_dynamic_state::Device::new(instance, device);
        let f = extension.fp();
        let restart = enabled
            .extensions
            .contains(&VkExtension::ExtExtendedDynamicState2)
            .then(|| {
                ash::ext::extended_dynamic_state2::Device::new(instance, device)
                    .fp()
                    .cmd_set_primitive_restart_enable_ext
            });
        // SAFETY: each is the device's command of the routed command's signature (vk.xml aliases
        // each to the core one), as its driver gave it.
        unsafe {
            Some(ExtendedDynamicState {
                cull_mode: through(slot, Command::CmdSetCullMode, f.cmd_set_cull_mode_ext),
                front_face: through(slot, Command::CmdSetFrontFace, f.cmd_set_front_face_ext),
                primitive_topology: through(
                    slot,
                    Command::CmdSetPrimitiveTopology,
                    f.cmd_set_primitive_topology_ext,
                ),
                depth_test_enable: through(
                    slot,
                    Command::CmdSetDepthTestEnable,
                    f.cmd_set_depth_test_enable_ext,
                ),
                depth_write_enable: through(
                    slot,
                    Command::CmdSetDepthWriteEnable,
                    f.cmd_set_depth_write_enable_ext,
                ),
                depth_compare_op: through(
                    slot,
                    Command::CmdSetDepthCompareOp,
                    f.cmd_set_depth_compare_op_ext,
                ),
                stencil_test_enable: through(
                    slot,
                    Command::CmdSetStencilTestEnable,
                    f.cmd_set_stencil_test_enable_ext,
                ),
                stencil_op: through(slot, Command::CmdSetStencilOp, f.cmd_set_stencil_op_ext),
                bind_vertex_buffers2: through(
                    slot,
                    Command::CmdBindVertexBuffers2,
                    f.cmd_bind_vertex_buffers2_ext,
                ),
                primitive_restart_enable: restart
                    .map(|restart| through(slot, Command::CmdSetPrimitiveRestartEnable, restart)),
            })
        }
    }
}

/// `VK_EXT_extended_dynamic_state3`'s `vkCmdSetColorWriteMaskEXT`, which a draw that reads a
/// texture its pass renders through a snapshot records twice (`render/snapshot.rs`): loaded only
/// where the device enabled everything that path takes, `attachmentFeedbackLoopLayout`,
/// `extendedDynamicState3ColorWriteMask` and `fragmentShaderPixelInterlock`, so its presence is
/// the device's answer to whether a draw may take it.
#[derive(Clone, Copy)]
pub(super) struct ColorWriteMasks(pub(super) vk::PFN_vkCmdSetColorWriteMaskEXT);

impl ColorWriteMasks {
    /// The command of a device with `enabled` enabled, where it enabled the three features.
    ///
    /// # Safety
    ///
    /// `device` was made from `instance` with `enabled` enabled, and the loader serves
    /// `vkGetDeviceProcAddr` for it (its loader word is in place).
    pub(super) unsafe fn load(
        instance: &ash::Instance,
        device: &ash::Device,
        enabled: &EnableSet,
    ) -> Option<ColorWriteMasks> {
        let needed = [
            VkFeature::AttachmentFeedbackLoopLayout,
            VkFeature::ExtendedDynamicState3ColorWriteMask,
            VkFeature::FragmentShaderPixelInterlock,
        ];
        if !needed.iter().all(|f| enabled.features.contains(f)) {
            return None;
        }
        // SAFETY: the caller's; a command the device does not have is null.
        let command = unsafe {
            instance.get_device_proc_addr(device.handle(), c"vkCmdSetColorWriteMaskEXT".as_ptr())
        }?;
        // SAFETY: the device gave the command of this name, whose signature this is.
        Some(ColorWriteMasks(unsafe {
            std::mem::transmute::<unsafe extern "system" fn(), vk::PFN_vkCmdSetColorWriteMaskEXT>(
                command,
            )
        }))
    }
}
