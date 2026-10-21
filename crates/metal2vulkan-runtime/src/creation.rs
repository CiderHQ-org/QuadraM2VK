//! Making the Vulkan objects a host step does not make: the runtime's one door to `vkCreateImage`,
//! `vkCreateRenderPass`, `vkCreateRenderPass2` and `vkCreateFramebuffer`.
//!
//! A host step (`exit_guard::host_step`) runs on the scheduler's thread inside the exit guard's
//! section, which reaches only the entry points a host step calls (`exit_guard::HOST_STEP`), primed
//! before the exit hook registers. These four are not among them: a host step makes no image,
//! render pass or framebuffer. What a host step may need of them is made before, on the program's
//! thread (a state's creation, an encoder's end, a command buffer's commit); one nothing made is
//! refused here, before any Vulkan call, with [`Error::HostStepObject`], which fails the command
//! buffer `CommandBufferError::HostStepObject` of the same kind. Outside a host step each call is
//! the entry point's: on the program's thread, or in a section of the device's lowering thread,
//! which lowers render encoders' calls (`render/lowering_worker.rs`), so the four and an image's
//! memory requirements and binding are in the guard's table (`exit_guard::Call`). The device's
//! priming (`device/prime.rs`) makes its own, on the thread that makes the device. No other file
//! calls these entry points: a source test fails when one does (`tests/exit_guard.rs`).

use crate::error::{vulkan, Error, HostStepObject};
use crate::exit_guard;
use ash::vk;

/// Refused in a host step (the module docs): `object` is made before it, on the program's
/// thread, or not at all.
pub(crate) fn outside_host_step(object: HostStepObject) -> Result<(), Error> {
    match exit_guard::in_host_step() {
        true => Err(Error::HostStepObject(object)),
        false => Ok(()),
    }
}

/// A `VkImage` of `info`, made by `device`; refused in a host step.
///
/// # Safety
///
/// As `vkCreateImage`'s: `device` is live, and `info` and its chain are valid for it.
pub(crate) unsafe fn image(
    device: &ash::Device,
    info: &vk::ImageCreateInfo<'_>,
) -> Result<vk::Image, Error> {
    outside_host_step(HostStepObject::Image)?;
    // SAFETY: the caller's.
    unsafe { device.create_image(info, None) }.map_err(vulkan)
}

/// A `VkImage` of `info`, made by `device`, or the driver's refusal of it: an image whose making
/// the driver may refuse for its parameters alone (one of a DRM format modifier's explicit
/// layout, `dma_buf.rs`), whose caller takes another way then. A lost device is
/// `Error::DeviceLost`; refused in a host step.
///
/// # Safety
///
/// As `vkCreateImage`'s: `device` is live, and `info` and its chain are valid for it.
pub(crate) unsafe fn image_or_refusal(
    device: &ash::Device,
    info: &vk::ImageCreateInfo<'_>,
) -> Result<Result<vk::Image, vk::Result>, Error> {
    outside_host_step(HostStepObject::Image)?;
    // SAFETY: the caller's.
    match unsafe { device.create_image(info, None) } {
        Ok(raw) => Ok(Ok(raw)),
        Err(vk::Result::ERROR_DEVICE_LOST) => Err(Error::DeviceLost),
        Err(refusal) => Ok(Err(refusal)),
    }
}

/// A `VkRenderPass` of `info`, made by `device`; refused in a host step.
///
/// # Safety
///
/// As `vkCreateRenderPass`'s: `device` is live, and `info` and its chain are valid for it.
pub(crate) unsafe fn render_pass(
    device: &ash::Device,
    info: &vk::RenderPassCreateInfo<'_>,
) -> Result<vk::RenderPass, Error> {
    outside_host_step(HostStepObject::RenderPass)?;
    // SAFETY: the caller's.
    unsafe { device.create_render_pass(info, None) }.map_err(vulkan)
}

/// A `VkRenderPass` of `info` (`vkCreateRenderPass2`), made by `device`; refused in a host step.
///
/// # Safety
///
/// As `vkCreateRenderPass2`'s: `device` is live and of Vulkan 1.2 or later, and `info` and its
/// chain are valid for it.
pub(crate) unsafe fn render_pass2(
    device: &ash::Device,
    info: &vk::RenderPassCreateInfo2<'_>,
) -> Result<vk::RenderPass, Error> {
    outside_host_step(HostStepObject::RenderPass)?;
    // SAFETY: the caller's.
    unsafe { device.create_render_pass2(info, None) }.map_err(vulkan)
}

/// A `VkFramebuffer` of `info`, made by `device`; refused in a host step.
///
/// # Safety
///
/// As `vkCreateFramebuffer`'s: `device` is live, and `info` and its chain are valid for it.
pub(crate) unsafe fn framebuffer(
    device: &ash::Device,
    info: &vk::FramebufferCreateInfo<'_>,
) -> Result<vk::Framebuffer, Error> {
    outside_host_step(HostStepObject::Framebuffer)?;
    // SAFETY: the caller's.
    unsafe { device.create_framebuffer(info, None) }.map_err(vulkan)
}
