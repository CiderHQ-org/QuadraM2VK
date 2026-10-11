//! The shader-side stencil's shadow: what a consumer keeps for a pipeline whose fragment function
//! tests and writes the stencil itself (`GraphicsState::shadow_stencil`).
//!
//! The executor runs the protocol `ShadowStencilUse` states around its one draw. The shadow is a
//! 2D storage image of the use's format, the stencil attachment's extent, and the use's sample
//! count, sample `s` of texel `(x, y)` for sample `s` of pixel `(x, y)`, kept in `GENERAL` through
//! the draw and bound where the plan writes `WriteSource::ShadowStencil`. Outside the render
//! pass, before it, the attachment's stencil aspect is copied into the shadow; after it, the
//! shadow is copied back into the stencil aspect, so the attachment's readback sees the
//! function's writes. A single-sample shadow's copies pass through a buffer; a multisampled one's
//! are one `vkCmdCopyImage` each between the stencil aspect and the shadow's color, which
//! `VK_KHR_maintenance8` allows between an 8-bit stencil and `R8_UINT` of one sample count
//! (Vulkan copies no multisampled image to a buffer). The attachment leaves its render-pass
//! layout (`DEPTH_STENCIL_ATTACHMENT_OPTIMAL`) for each copy and returns to it, as at a
//! consumer's pass split; its depth aspect is untouched. The pipeline's stencil test is off
//! (`EncoderState::of`), so the pass itself never writes the stencil.

use super::super::buffers::{create_host_buffer, HostBuffer};
use super::super::device::VulkanContext;
use super::super::plan_resources::{barrier, create_image, layers};
use super::DepthStencilTarget;
use crate::check::admission::ExecutorError;
use ash::vk;
use metal2vulkan::pipeline::ShadowStencilUse;

/// The fragment-test stages that access the depth/stencil attachment.
const TESTS: vk::PipelineStageFlags = vk::PipelineStageFlags::from_raw(
    vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS.as_raw()
        | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS.as_raw(),
);

/// The accesses of the attachment in its render-pass layout.
const ATTACHMENT: vk::AccessFlags = vk::AccessFlags::from_raw(
    vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ.as_raw()
        | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE.as_raw(),
);

/// The accesses of the shadow inside the fragment shader's critical section.
const SHADER: vk::AccessFlags = vk::AccessFlags::from_raw(
    vk::AccessFlags::SHADER_READ.as_raw() | vk::AccessFlags::SHADER_WRITE.as_raw(),
);

/// The shadow image, its storage view, and the buffer a single-sample shadow's copies pass
/// through, destroyed after the submission that used them completed (or before any submission).
pub(super) struct ShadowStencil<'a> {
    device: &'a ash::Device,
    image: vk::Image,
    memory: vk::DeviceMemory,
    /// The storage view the plan's `WriteSource::ShadowStencil` binds.
    pub(super) view: vk::ImageView,
    /// One sample: one byte per pixel, the stencil aspect's buffer layout and the shadow's. More
    /// samples: `None`, copied image to image.
    buffer: Option<HostBuffer>,
    extent: vk::Extent3D,
}

impl Drop for ShadowStencil<'_> {
    fn drop(&mut self) {
        // SAFETY: every handle was created on `device` by this run and is destroyed once, after
        // the submission that used it completed (or before any submission).
        unsafe {
            self.device.destroy_image_view(self.view, None);
            self.device.destroy_image(self.image, None);
            self.device.free_memory(self.memory, None);
        }
    }
}

impl<'a> ShadowStencil<'a> {
    /// The shadow `used` names of the stencil of `target`, which must be an attachment with a
    /// stencil aspect of the use's sample count (link states no shadow otherwise).
    pub(super) fn create(
        context: &'a VulkanContext,
        used: &ShadowStencilUse,
        target: &DepthStencilTarget,
    ) -> Result<ShadowStencil<'a>, ExecutorError> {
        let label = "the shader-side stencil's shadow";
        if !target
            .format_aspects
            .contains(vk::ImageAspectFlags::STENCIL)
        {
            return Err(format!("{label}: the depth/stencil attachment holds no stencil").into());
        }
        if target.samples != used.samples {
            return Err(format!(
                "{label}: the attachment has {} samples, and link states a shadow of {}",
                target.samples, used.samples
            )
            .into());
        }
        let samples = match used.samples {
            1 => vk::SampleCountFlags::TYPE_1,
            2 => vk::SampleCountFlags::TYPE_2,
            4 => vk::SampleCountFlags::TYPE_4,
            other => {
                return Err(format!("{label}: a shadow of {other} samples is not drawn").into())
            }
        };
        let [width, height] = target.dimensions;
        let extent = vk::Extent3D {
            width,
            height,
            depth: 1,
        };
        let format: vk::Format = used.format.into();
        let buffer = (used.samples == 1)
            .then(|| {
                create_host_buffer(
                    context,
                    &format!("{label}'s copies"),
                    &vec![0; width as usize * height as usize],
                    vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
                )
            })
            .transpose()?;
        let (image, memory) = create_image(
            context,
            label,
            &vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(format)
                .extent(extent)
                .mip_levels(1)
                .array_layers(1)
                .samples(samples)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(
                    vk::ImageUsageFlags::STORAGE
                        | vk::ImageUsageFlags::TRANSFER_SRC
                        | vk::ImageUsageFlags::TRANSFER_DST,
                )
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED),
        )?;
        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });
        // SAFETY: `view_info` names the live image.
        let view = match unsafe { context.device.create_image_view(&view_info, None) } {
            Ok(view) => view,
            Err(error) => {
                // SAFETY: the image and memory were created above and are unused.
                unsafe {
                    context.device.destroy_image(image, None);
                    context.device.free_memory(memory, None);
                }
                return Err(format!("create {label} view: {error}").into());
            }
        };
        Ok(ShadowStencil {
            device: &context.device,
            image,
            memory,
            view,
            buffer,
            extent,
        })
    }

    /// The copy of one aspect of an image of the shadow's extent from or into the buffer.
    fn region(&self, aspect: vk::ImageAspectFlags) -> vk::BufferImageCopy {
        vk::BufferImageCopy::default()
            .image_subresource(layers(aspect, 1))
            .image_extent(self.extent)
    }

    /// The buffer's bytes written by a transfer, read by the next one.
    fn buffer_handoff(buffer: &HostBuffer) -> vk::BufferMemoryBarrier<'static> {
        vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(buffer.buffer)
            .offset(0)
            .size(vk::WHOLE_SIZE)
    }

    /// Record the copy of every sample of the stencil aspect of `from` into the shadow's color,
    /// or back (`into_shadow` false), image to image (a multisampled shadow).
    ///
    /// # Safety
    ///
    /// `command` is recording, outside a render pass; the source is in `TRANSFER_SRC_OPTIMAL`
    /// and the destination in `TRANSFER_DST_OPTIMAL`.
    unsafe fn copy_samples(
        &self,
        command: vk::CommandBuffer,
        target: vk::Image,
        into_shadow: bool,
    ) {
        let stencil = layers(vk::ImageAspectFlags::STENCIL, 1);
        let color = layers(vk::ImageAspectFlags::COLOR, 1);
        let (source, destination, region) = if into_shadow {
            (
                target,
                self.image,
                vk::ImageCopy::default()
                    .src_subresource(stencil)
                    .dst_subresource(color),
            )
        } else {
            (
                self.image,
                target,
                vk::ImageCopy::default()
                    .src_subresource(color)
                    .dst_subresource(stencil),
            )
        };
        // SAFETY: the caller's contract.
        unsafe {
            self.device.cmd_copy_image(
                command,
                source,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                destination,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region.extent(self.extent)],
            );
        }
    }

    /// Record the copy of `target`'s stencil into the shadow, outside the render pass and before
    /// it: the attachment, in its render-pass layout, is read as a transfer source and returned;
    /// the shadow ends in `GENERAL` for the fragment shader's storage reads and writes.
    ///
    /// # Safety
    ///
    /// `command` is recording, outside a render pass; every handle is live until it completes.
    pub(super) unsafe fn record_copy_in(
        &self,
        command: vk::CommandBuffer,
        target: &DepthStencilTarget,
    ) {
        let device = self.device;
        let aspects = target.format_aspects;
        // SAFETY: the caller's contract.
        unsafe {
            device.cmd_pipeline_barrier(
                command,
                TESTS,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier(
                    target.image,
                    aspects,
                    1,
                    (
                        vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                    ),
                    (
                        vk::AccessFlags::TRANSFER_READ,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    ),
                )],
            );
            let to_destination = barrier(
                self.image,
                vk::ImageAspectFlags::COLOR,
                1,
                (vk::AccessFlags::empty(), vk::ImageLayout::UNDEFINED),
                (
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                ),
            );
            match &self.buffer {
                Some(buffer) => {
                    device.cmd_copy_image_to_buffer(
                        command,
                        target.image,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        buffer.buffer,
                        &[self.region(vk::ImageAspectFlags::STENCIL)],
                    );
                    device.cmd_pipeline_barrier(
                        command,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[Self::buffer_handoff(buffer)],
                        &[to_destination],
                    );
                    device.cmd_copy_buffer_to_image(
                        command,
                        buffer.buffer,
                        self.image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &[self.region(vk::ImageAspectFlags::COLOR)],
                    );
                }
                None => {
                    device.cmd_pipeline_barrier(
                        command,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[to_destination],
                    );
                    self.copy_samples(command, target.image, true);
                }
            }
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER | TESTS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[
                    barrier(
                        self.image,
                        vk::ImageAspectFlags::COLOR,
                        1,
                        (
                            vk::AccessFlags::TRANSFER_WRITE,
                            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        ),
                        (SHADER, vk::ImageLayout::GENERAL),
                    ),
                    barrier(
                        target.image,
                        aspects,
                        1,
                        (
                            vk::AccessFlags::TRANSFER_READ,
                            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        ),
                        (
                            ATTACHMENT,
                            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                        ),
                    ),
                ],
            );
        }
    }

    /// Record the copy of the shadow back into `target`'s stencil, outside the render pass and
    /// after it: the fragment shader's writes and the pass's store reach the transfers, and the
    /// attachment returns to its render-pass layout.
    ///
    /// # Safety
    ///
    /// As for [`ShadowStencil::record_copy_in`], recorded after the render pass ended.
    pub(super) unsafe fn record_copy_out(
        &self,
        command: vk::CommandBuffer,
        target: &DepthStencilTarget,
    ) {
        let device = self.device;
        let aspects = target.format_aspects;
        // SAFETY: the caller's contract.
        unsafe {
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::FRAGMENT_SHADER | TESTS,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[
                    barrier(
                        self.image,
                        vk::ImageAspectFlags::COLOR,
                        1,
                        (vk::AccessFlags::SHADER_WRITE, vk::ImageLayout::GENERAL),
                        (
                            vk::AccessFlags::TRANSFER_READ,
                            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        ),
                    ),
                    barrier(
                        target.image,
                        aspects,
                        1,
                        (
                            vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                        ),
                        (
                            vk::AccessFlags::TRANSFER_WRITE,
                            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        ),
                    ),
                ],
            );
            match &self.buffer {
                Some(buffer) => {
                    device.cmd_copy_image_to_buffer(
                        command,
                        self.image,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        buffer.buffer,
                        &[self.region(vk::ImageAspectFlags::COLOR)],
                    );
                    device.cmd_pipeline_barrier(
                        command,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[Self::buffer_handoff(buffer)],
                        &[],
                    );
                    device.cmd_copy_buffer_to_image(
                        command,
                        buffer.buffer,
                        target.image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &[self.region(vk::ImageAspectFlags::STENCIL)],
                    );
                }
                None => self.copy_samples(command, target.image, false),
            }
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::TRANSFER,
                TESTS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier(
                    target.image,
                    aspects,
                    1,
                    (
                        vk::AccessFlags::TRANSFER_WRITE,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    ),
                    (
                        ATTACHMENT,
                        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                    ),
                )],
            );
        }
    }
}
