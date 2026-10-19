//! The checks of five storage-write deviations Metal's rule decides (`DriverDeviation`'s
//! `MetalHalfStorageWrites`, `MetalHalfStorageNans`, `MetalFloatStorageNans`,
//! `MetalHalfTexelWrites` and `MetalNormStorageWrites`): each fact's writes
//! (`metal2vulkan::bind::storage_writes`), one dispatch of each kernel translation's declarations
//! give the format (`storage_write_modules`) writing each texel's input as computed into a storage
//! image, read back and compared with Metal's texels. A write of a format the device does not
//! store to is left out, as no Metal texture of it is written there; a fact none of whose writes
//! runs is not run.

use super::color::combined;
use super::kit::{CheckError, Kit, SetBinding, SetResource};
use super::Target;
use ash::vk;
use metal2vulkan::bind::{
    storage_write_modules, storage_writes, StorageWrite, StorageWriteModule, STORAGE_WRITE_FACTS,
};
use metal2vulkan::device::CheckAnswer;
use std::collections::BTreeMap;
use std::sync::Arc;

/// The five checks' answers on `target`, in [`STORAGE_WRITE_FACTS`]' order.
pub(super) fn run(kit: &mut Kit<'_>, target: &Target<'_>) -> Vec<CheckAnswer> {
    let mut pipelines = BTreeMap::new();
    STORAGE_WRITE_FACTS
        .iter()
        .map(|&fact| {
            let Some(writes) = storage_writes(fact) else {
                return CheckAnswer::Unanswered("the fact has no writes".to_owned());
            };
            let answers: Vec<CheckAnswer> = writes
                .iter()
                .flat_map(|w| answers(kit, target, &mut pipelines, w))
                .collect();
            match answers.is_empty() {
                true => CheckAnswer::NotRun("the device stores to none of its formats".to_owned()),
                false => combined(answers),
            }
        })
        .collect()
}

/// The pipelines made so far, by their module's words.
type Pipelines = BTreeMap<Arc<[u32]>, vk::Pipeline>;

/// The answers of write `w`, one per kernel of its format; none where the device does not store
/// to the format.
fn answers(
    kit: &mut Kit<'_>,
    target: &Target<'_>,
    pipelines: &mut Pipelines,
    w: &StorageWrite,
) -> Vec<CheckAnswer> {
    let format = vk::Format::from_raw(w.format.as_raw());
    let features = match kit.physical_query(|instance, physical| {
        // SAFETY: the physical device is the instance's.
        unsafe { instance.get_physical_device_format_properties(physical, format) }
    }) {
        Ok(properties) => properties.optimal_tiling_features,
        Err(error) => return vec![CheckAnswer::Unanswered(error.0)],
    };
    let needed = vk::FormatFeatureFlags::STORAGE_IMAGE | vk::FormatFeatureFlags::TRANSFER_SRC;
    if !features.contains(needed) {
        return Vec::new();
    }
    let modules = match storage_write_modules(target.compile, w.format) {
        Ok(modules) => modules,
        Err(error) => {
            return vec![CheckAnswer::Unanswered(format!(
                "its kernels do not build: {error}"
            ))]
        }
    };
    modules
        .iter()
        .map(|module| match written(kit, pipelines, module, format, w) {
            Ok(texels) => match w.difference(&texels) {
                None => CheckAnswer::Metals,
                Some(how) => CheckAnswer::NotMetals(how),
            },
            Err(error) => CheckAnswer::Unanswered(error.0),
        })
        .collect()
}

/// The texels `module` leaves in a storage image of `format` written from `w`'s inputs, row by
/// row, tightly packed.
fn written(
    kit: &mut Kit<'_>,
    pipelines: &mut Pipelines,
    module: &StorageWriteModule,
    format: vk::Format,
    w: &StorageWrite,
) -> Result<Vec<u8>, CheckError> {
    let (width, height) = (StorageWrite::WIDTH, w.height);
    let values = kit.host_buffer(&w.values, vk::BufferUsageFlags::STORAGE_BUFFER)?;
    let image = kit.image(
        &vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC)
            .initial_layout(vk::ImageLayout::UNDEFINED),
    )?;
    let range = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    };
    let view = kit.view(
        &vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(range),
    )?;
    let (set_layout, set) = kit.descriptor_set(&[
        SetBinding {
            binding: StorageWriteModule::VALUES_BINDING,
            stages: vk::ShaderStageFlags::COMPUTE,
            resource: SetResource::Buffer(values),
        },
        SetBinding {
            binding: StorageWriteModule::IMAGE_BINDING,
            stages: vk::ShaderStageFlags::COMPUTE,
            resource: SetResource::StorageImage(view),
        },
    ])?;
    let layout = kit.layout(&[set_layout], None)?;
    // Each write's layout is defined alike, so a pipeline made under the first is compatible
    // with every later one.
    let pipeline = match pipelines.get(&module.spirv) {
        Some(&pipeline) => pipeline,
        None => {
            let shader = kit.shader(&module.spirv)?;
            let pipeline = kit.compute_pipeline(shader, layout)?;
            pipelines.insert(Arc::clone(&module.spirv), pipeline);
            pipeline
        }
    };
    let bytes = w.expected.len();
    let read = kit.host_buffer(
        &vec![0; bytes.div_ceil(4)],
        vk::BufferUsageFlags::TRANSFER_DST,
    )?;
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
        image_extent: vk::Extent3D {
            width,
            height,
            depth: 1,
        },
    };
    let general = vk::ImageMemoryBarrier::default()
        .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::GENERAL)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(range);
    let stored = vk::ImageMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
        .old_layout(vk::ImageLayout::GENERAL)
        .new_layout(vk::ImageLayout::GENERAL)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(range);
    let copied = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::HOST_READ);
    let workgroups = width / StorageWrite::LOCAL_SIZE;
    kit.submit(|device, command| {
        // SAFETY: recording, outside a render pass; every object named is live, the image in
        // `GENERAL` for the dispatch and the copy, the buffer large enough for its tightly packed
        // texels.
        unsafe {
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[general],
            );
            device.cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, pipeline);
            device.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::COMPUTE,
                layout,
                0,
                &[set],
                &[],
            );
            device.cmd_dispatch(command, workgroups, height, 1);
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[stored],
            );
            device.cmd_copy_image_to_buffer(
                command,
                image,
                vk::ImageLayout::GENERAL,
                read.buffer,
                &[copy],
            );
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[copied],
                &[],
                &[],
            );
        }
    })?;
    let mut texels: Vec<u8> = (kit.read(&read)?.iter())
        .flat_map(|w| w.to_le_bytes())
        .collect();
    texels.truncate(bytes);
    Ok(texels)
}
