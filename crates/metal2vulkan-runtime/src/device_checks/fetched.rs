//! An image a check's kernel fetches from: its bytes copied in from a host buffer, read through a
//! sampled view, and the descriptor set of the kernel over it, recorded as one copy and one
//! dispatch (`block_decode.rs`, `sampled_fetch.rs`).

use super::kit::{CheckError, HostBuffer, Kit, SetBinding, SetResource};
use ash::vk;

/// One image of a fetch and its kernel's set, kept by the kit.
pub(super) struct Fetched {
    bytes: HostBuffer,
    image: vk::Image,
    extent: vk::Extent3D,
    /// The set's layout: every fetch of the same bindings has one defined alike.
    pub(super) set_layout: vk::DescriptorSetLayout,
    set: vk::DescriptorSet,
}

impl Fetched {
    /// A 2D image of `format` and `extent` holding `bytes` (tight rows, as a copy into its level
    /// 0 reads them), its sampled view at `image_binding` of a set whose other bindings are
    /// `buffers`' storage buffers, read and written by a compute kernel.
    pub(super) fn new(
        kit: &mut Kit<'_>,
        format: vk::Format,
        [width, height]: [u32; 2],
        bytes: &[u8],
        image_binding: u32,
        buffers: &[(u32, HostBuffer)],
    ) -> Result<Fetched, CheckError> {
        let words: Vec<u32> = (bytes.chunks(4))
            .map(|b| {
                let mut word = [0u8; 4];
                word[..b.len()].copy_from_slice(b);
                u32::from_le_bytes(word)
            })
            .collect();
        let bytes = kit.host_buffer(&words, vk::BufferUsageFlags::TRANSFER_SRC)?;
        let extent = vk::Extent3D {
            width,
            height,
            depth: 1,
        };
        let image = kit.image(
            &vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(format)
                .extent(extent)
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED),
        )?;
        let view = kit.view(
            &vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format)
                .subresource_range(color_range()),
        )?;
        let stages = vk::ShaderStageFlags::COMPUTE;
        let mut bindings = vec![SetBinding {
            binding: image_binding,
            stages,
            resource: SetResource::SampledImage(view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
        }];
        bindings.extend(buffers.iter().map(|&(binding, buffer)| SetBinding {
            binding,
            stages,
            resource: SetResource::Buffer(buffer),
        }));
        let (set_layout, set) = kit.descriptor_set(&bindings)?;
        Ok(Fetched {
            bytes,
            image,
            extent,
            set_layout,
            set,
        })
    }

    /// Record the copy of the bytes into the image and one dispatch of `workgroups` along x of
    /// the bound kernel over the set.
    ///
    /// # Safety
    ///
    /// `command` is recording outside a render pass with a compute pipeline of `layout` bound,
    /// whose set 0 is compatible with the set's layout.
    pub(super) unsafe fn record(
        &self,
        device: &ash::Device,
        command: vk::CommandBuffer,
        layout: vk::PipelineLayout,
        workgroups: u32,
    ) {
        let barrier = |old, new, src, dst| {
            vk::ImageMemoryBarrier::default()
                .src_access_mask(src)
                .dst_access_mask(dst)
                .old_layout(old)
                .new_layout(new)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(self.image)
                .subresource_range(color_range())
        };
        let fresh = barrier(
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::AccessFlags::empty(),
            vk::AccessFlags::TRANSFER_WRITE,
        );
        let copied = barrier(
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::SHADER_READ,
        );
        let copy = vk::BufferImageCopy {
            buffer_offset: 0,
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
            image_extent: self.extent,
        };
        // SAFETY: the caller's; the image and buffers are the fetch's, live.
        unsafe {
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[fresh],
            );
            device.cmd_copy_buffer_to_image(
                command,
                self.bytes.buffer,
                self.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[copy],
            );
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[copied],
            );
            device.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::COMPUTE,
                layout,
                0,
                &[self.set],
                &[],
            );
            device.cmd_dispatch(command, workgroups, 1, 1);
        }
    }
}

/// Record every fetch of `fetches` with the compute `pipeline` of `layout` bound, then make
/// their writes visible to the host.
///
/// # Safety
///
/// As [`Fetched::record`]'s, `pipeline` a compute pipeline of `layout`.
pub(super) unsafe fn record_all<'a>(
    device: &ash::Device,
    command: vk::CommandBuffer,
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    fetches: impl IntoIterator<Item = (&'a Fetched, u32)>,
) {
    let written = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::HOST_READ);
    // SAFETY: the caller's.
    unsafe {
        device.cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, pipeline);
        for (fetch, workgroups) in fetches {
            fetch.record(device, command, layout, workgroups);
        }
        device.cmd_pipeline_barrier(
            command,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &[written],
            &[],
            &[],
        );
    }
}

/// The color aspect of level 0 and layer 0.
pub(super) fn color_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    }
}
