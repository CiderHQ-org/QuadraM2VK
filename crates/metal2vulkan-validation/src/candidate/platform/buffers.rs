//! Host-visible buffers: their ownership, mapping, and out-of-bounds guards.

use super::device::VulkanContext;
use crate::candidate::engine::guards::{GuardCount, GuardReach, MIN_GUARD};
use crate::executor_contract::out_of_bounds::{GUARD_FILL, PAD_FILL};
use ash::{vk, Device};

// Every buffer test maps or submits through a live Vulkan device.
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod tests;

pub(super) struct HostBuffer {
    device: Device,
    pub(super) buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    pub(super) len: vk::DeviceSize,
    /// Where the buffer is bound in `memory`: 0, or past the front guard of a guarded buffer.
    offset: vk::DeviceSize,
    guard: Option<Guard>,
}

/// The guard bytes of a guarded host buffer's memory: `[0, offset)` before the buffer and
/// `[bound_end, memory_len)` past the memory bound to it, both filled with [`GUARD_FILL`] when
/// allocated. The memory bound to the buffer (`[offset, bound_end)`, its size requirement) is the
/// buffer's own: past its length it is padding, filled with [`PAD_FILL`], which only a robustness-1
/// out-of-bounds write may change; no out-of-bounds access may change a guard byte.
#[derive(Clone, Copy)]
struct Guard {
    bound_end: vk::DeviceSize,
    memory_len: vk::DeviceSize,
}

pub(super) struct MappedHostBuffer<'a> {
    allocation: &'a HostBuffer,
    pub(super) bytes: &'a mut [u8],
}

impl HostBuffer {
    /// Map host-coherent storage for exclusive host access.
    ///
    /// Safety: submitted GPU uses must have completed, and no other host or GPU access
    /// to the mapped bytes may overlap the returned guard.
    pub(super) unsafe fn map(&self) -> Result<MappedHostBuffer<'_>, String> {
        unsafe { self.map_range(self.offset, self.len) }
    }

    /// Map `len` bytes of the memory at `offset`.
    ///
    /// Safety: as for [`HostBuffer::map`].
    unsafe fn map_range(
        &self,
        offset: vk::DeviceSize,
        len: vk::DeviceSize,
    ) -> Result<MappedHostBuffer<'_>, String> {
        let length =
            usize::try_from(len).map_err(|_| "host buffer length exceeds host address space")?;
        let pointer = unsafe {
            self.device
                .map_memory(self.memory, offset, len, vk::MemoryMapFlags::empty())
        }
        .map_err(|error| format!("map host buffer memory: {error}"))?;
        Ok(MappedHostBuffer {
            allocation: self,
            bytes: unsafe { std::slice::from_raw_parts_mut(pointer.cast::<u8>(), length) },
        })
    }

    /// The guard and padding bytes that changed since allocation: what escaped the buffer, and what
    /// reached the memory bound to it past its length. `None` for a buffer allocated without a
    /// guard.
    ///
    /// Safety: as for [`HostBuffer::map`].
    pub(super) unsafe fn escaped(&self) -> Result<Option<GuardCount>, String> {
        let Some(guard) = self.guard else {
            return Ok(None);
        };
        let memory = unsafe { self.map_range(0, guard.memory_len)? };
        let (front, end, back) = (
            self.offset as usize,
            (self.offset + self.len) as usize,
            guard.bound_end as usize,
        );
        let changed = |bytes: &[u8], fill: u8| bytes.iter().filter(|byte| **byte != fill).count();
        Ok(Some(GuardCount {
            escaped: (changed(&memory.bytes[..front], GUARD_FILL)
                + changed(&memory.bytes[back..], GUARD_FILL)) as u64,
            past_end: changed(&memory.bytes[end..back], PAD_FILL) as u64,
        }))
    }
}

impl Drop for MappedHostBuffer<'_> {
    fn drop(&mut self) {
        unsafe { self.allocation.device.unmap_memory(self.allocation.memory) };
    }
}

impl Drop for HostBuffer {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_buffer(self.buffer, None);
            self.device.free_memory(self.memory, None);
        }
    }
}

pub(super) fn create_host_buffer(
    context: &VulkanContext,
    label: &str,
    bytes: &[u8],
    usage: vk::BufferUsageFlags,
) -> Result<HostBuffer, String> {
    let allocation = allocate_host_buffer(context, label, bytes.len(), usage, None)?;
    // Newly allocated storage has never been submitted to the GPU.
    unsafe { allocation.map()? }.bytes.copy_from_slice(bytes);
    Ok(allocation)
}

/// [`create_host_buffer`] inside a guard of `reach`: one memory allocation holding a front guard,
/// the memory bound to the buffer, and a back guard past it, every guard byte [`GUARD_FILL`] and
/// the bound memory past the buffer's length [`PAD_FILL`] ([`HostBuffer::escaped`]).
pub(super) fn create_guarded_host_buffer(
    context: &VulkanContext,
    label: &str,
    bytes: &[u8],
    usage: vk::BufferUsageFlags,
    reach: GuardReach,
) -> Result<HostBuffer, String> {
    let allocation = allocate_host_buffer(context, label, bytes.len(), usage, Some(reach))?;
    let memory_len = allocation
        .guard
        .map_or(allocation.len, |guard| guard.memory_len);
    // Newly allocated storage has never been submitted to the GPU.
    {
        let memory = unsafe { allocation.map_range(0, memory_len)? };
        memory.bytes.fill(GUARD_FILL);
        let end = (allocation.offset + allocation.len) as usize;
        let bound_end = allocation
            .guard
            .map_or(end, |guard| guard.bound_end as usize);
        memory.bytes[end..bound_end].fill(PAD_FILL);
    }
    unsafe { allocation.map()? }.bytes.copy_from_slice(bytes);
    Ok(allocation)
}

/// Allocate storage for immediate caller initialization before any device or host read, inside a
/// guard of `reach` when one is given ([`create_guarded_host_buffer`]).
fn allocate_host_buffer(
    context: &VulkanContext,
    label: &str,
    len: usize,
    usage: vk::BufferUsageFlags,
    reach: Option<GuardReach>,
) -> Result<HostBuffer, String> {
    if len == 0 {
        return Err(format!("cannot allocate empty {label} buffer"));
    }
    let create_info = vk::BufferCreateInfo::default()
        .size(len as u64)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let buffer = unsafe { context.device.create_buffer(&create_info, None) }
        .map_err(|error| format!("create {label} buffer: {error}"))?;
    let requirements = unsafe { context.device.get_buffer_memory_requirements(buffer) };
    let memory_type = match context.memory_type(
        requirements.memory_type_bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    ) {
        Ok(memory_type) => memory_type,
        Err(error) => {
            unsafe { context.device.destroy_buffer(buffer, None) };
            return Err(error);
        }
    };
    // A guarded buffer is bound past its front guard, at an offset its alignment allows, and its
    // back guard begins where the memory bound to it ends.
    let (offset, guard) = match reach {
        None => (0, None),
        Some(reach) => {
            let alignment = requirements.alignment.max(1);
            let front = u64::from(reach.before).max(MIN_GUARD).div_ceil(alignment) * alignment;
            let bound_end = front + requirements.size;
            let back = u64::from(reach.after).max(MIN_GUARD);
            (
                front,
                Some(Guard {
                    bound_end,
                    memory_len: bound_end + back,
                }),
            )
        }
    };
    let memory_len = guard.map_or(requirements.size, |guard| guard.memory_len);
    let mut address_flags =
        vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
    let mut allocation_info = vk::MemoryAllocateInfo::default()
        .allocation_size(memory_len)
        .memory_type_index(memory_type);
    if usage.contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS) {
        allocation_info = allocation_info.push_next(&mut address_flags);
    }
    let memory = match unsafe { context.device.allocate_memory(&allocation_info, None) } {
        Ok(memory) => memory,
        Err(error) => {
            unsafe { context.device.destroy_buffer(buffer, None) };
            return Err(format!("allocate {label} memory: {error}"));
        }
    };
    let allocation = HostBuffer {
        device: context.device.clone(),
        buffer,
        memory,
        len: len as u64,
        offset,
        guard,
    };
    unsafe { context.device.bind_buffer_memory(buffer, memory, offset) }
        .map_err(|error| format!("bind {label} memory: {error}"))?;
    Ok(allocation)
}
