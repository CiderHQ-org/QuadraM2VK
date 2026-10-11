//! A Metal buffer over a `VkBuffer` the caller made: the shape of a consumer that imports memory
//! with plain Vulkan (another process's pages, as a dma-buf) and hands it to the runtime.
//!
//! ```sh
//! cargo run -p metal2vulkan-runtime --example adopted_buffer [-- path/to/library.metallib]
//! ```
//!
//! The example makes the `VkBuffer` and its `VkDeviceMemory` itself, through the runtime device's
//! raw handles, in host-visible coherent memory it maps. A consumer importing a dma-buf does the
//! same, but chains `VkImportMemoryFdInfoKHR` (handle type `DMA_BUF_BIT_EXT`, with
//! `VK_EXT_external_memory_dma_buf` enabled on the device) to the allocation, and passes its own
//! mapping of the same pages as the contents: the runtime never sees the file descriptor. Then
//! `new_buffer_with_vk_buffer` makes a `Shared` Metal buffer over it, which the kernel
//! `scale_and_offset` (of the repository's owned fixture, or of the metallib the argument names)
//! reads and writes, and the caller reads the result through its own mapping.

mod common;

use ash::vk;
use metal2vulkan_runtime::{Device, PipelineOptions, ResourceOptions, Size, VkBufferMemory};
use std::ptr::NonNull;

/// The floats the kernel scales: the input in the buffer's first half, the output in its second.
const COUNT: u64 = 64;

/// The caller's buffer: COUNT input floats, then COUNT output floats.
const SIZE: u64 = 2 * COUNT * 4;

fn main() -> common::Result<()> {
    let device = Device::system_default()?;
    println!("device: {}", device.name());
    let library = common::library(&device)?;
    let function = library.new_function_with_name("scale_and_offset")?;
    let state =
        device.new_compute_pipeline_state_with_function(&function, &PipelineOptions::default())?;

    // SAFETY: the handles are used only while the device lives, and none of them is destroyed.
    let raw = unsafe { device.raw() };
    // The buffer must be one the runtime could have made: every usage its own buffers have.
    let usage = vk::BufferUsageFlags::from(device.vk_buffer_usage());
    let info = vk::BufferCreateInfo::default()
        .size(SIZE)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    // SAFETY: a live device and a valid creation.
    let buffer = unsafe { raw.device.create_buffer(&info, None) }?;
    // SAFETY: a live buffer of the device, and the instance's physical device.
    let (requirements, properties) = unsafe {
        (
            raw.device.get_buffer_memory_requirements(buffer),
            raw.instance
                .get_physical_device_memory_properties(raw.physical_device),
        )
    };
    let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    let type_index = (0..properties.memory_type_count)
        .find(|&index| {
            requirements.memory_type_bits & (1 << index) != 0
                && properties.memory_types[index as usize]
                    .property_flags
                    .contains(host)
        })
        .ok_or("no host-visible coherent memory type for the buffer")?;
    // Where the device enables `bufferDeviceAddress`, the usage names device addresses and the
    // memory must be allocated for them.
    let mut address =
        vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
    let mut allocate = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(type_index);
    if usage.contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS) {
        allocate = allocate.push_next(&mut address);
    }
    // A dma-buf import chains its `VkImportMemoryFdInfoKHR` here.
    // SAFETY: `allocate` and its chain live for the call.
    let memory = unsafe { raw.device.allocate_memory(&allocate, None) }?;
    // SAFETY: unbound memory meeting the buffer's requirements from offset 0.
    unsafe { raw.device.bind_buffer_memory(buffer, memory, 0) }?;
    // SAFETY: host-visible memory, mapped once, whole.
    let mapped = unsafe {
        raw.device
            .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
    }?;
    let mapped = NonNull::new(mapped.cast::<u8>()).ok_or("a null mapping")?;

    // The caller writes the input through its own mapping.
    let input: Vec<u8> = (0..COUNT).flat_map(|i| (i as f32).to_le_bytes()).collect();
    // SAFETY: SIZE mapped bytes, which nothing else uses yet.
    unsafe { std::ptr::copy_nonoverlapping(input.as_ptr(), mapped.as_ptr(), input.len()) };

    // `Shared` storage: `contents` is the caller's mapping.
    // SAFETY: the buffer was made with exactly `info` and bound whole at 0 to `memory` of
    // `type_index` (allocated for device addresses where the usage names them); `mapped` maps its
    // bytes, coherent, for as long as the Metal buffer lives; both outlive the Metal buffer and the
    // command buffer, and the caller touches the bytes only once that has completed.
    let metal = unsafe {
        device.new_buffer_with_vk_buffer(
            buffer,
            &info,
            VkBufferMemory::new(memory, 0, type_index, Some(mapped)),
            SIZE,
            ResourceOptions::shared(),
        )
    }?;
    println!(
        "adopted: {} bytes at Metal address {:#x}, contents at the caller's mapping: {}",
        metal.length(),
        metal.gpu_address().0,
        metal.contents() == Some(mapped)
    );

    let scale = device.new_buffer_with_bytes(&2.0f32.to_le_bytes(), ResourceOptions::shared())?;
    let queue = device.new_command_queue()?;
    let command_buffer = queue.command_buffer()?;
    let mut encoder = command_buffer.compute_command_encoder()?;
    encoder.set_compute_pipeline_state(&state)?;
    encoder.set_buffer(Some(&metal), 0, 0)?;
    encoder.set_buffer(Some(&metal), COUNT * 4, 1)?;
    encoder.set_buffer(Some(&scale), 0, 2)?;
    encoder.dispatch_threads(Size::new(COUNT, 1, 1), Size::new(32, 1, 1))?;
    encoder.end_encoding()?;
    common::run(&command_buffer)?;

    // The caller reads the output through its own mapping: output[i] = i * 2 + i = 3i.
    // SAFETY: SIZE mapped bytes; the command buffer that wrote them completed.
    let bytes = unsafe { std::slice::from_raw_parts(mapped.as_ptr(), SIZE as usize) };
    // The Metal buffer's `contents` are the same bytes.
    if common::bytes(&metal)? != bytes {
        return Err("the Metal buffer's contents differ from the caller's mapping".into());
    }
    let output: Vec<f32> = bytes[(COUNT * 4) as usize..]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect();
    for (i, &value) in output.iter().enumerate() {
        if value != 3.0 * i as f32 {
            return Err(format!("output[{i}] is {value}, not {}", 3 * i).into());
        }
    }
    println!(
        "output[0..4] = {:?}: as expected, in the caller's memory",
        &output[..4]
    );

    // The Metal buffer goes first, then the caller's objects: the runtime never destroys them.
    drop((metal, command_buffer));
    // SAFETY: no Metal buffer or command buffer uses them any more.
    unsafe {
        raw.device.unmap_memory(memory);
        raw.device.destroy_buffer(buffer, None);
        raw.device.free_memory(memory, None);
    }
    Ok(())
}
