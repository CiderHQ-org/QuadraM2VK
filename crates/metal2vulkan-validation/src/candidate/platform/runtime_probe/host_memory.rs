//! Host memory without a copy: `VK_EXT_external_memory_host` on a real
//! page-aligned host allocation, and a storage buffer bound to the imported memory, used by a
//! dispatch through a descriptor and through its device address.

use super::{
    buffer_barrier, buffer_device_address, buffer_requirements, device_lists, dispatch_word,
    flag_names, host_coherent_type, host_page_size, memory_types, result_name, MemoryType, Owned,
    VulkanContext,
};
use ash::vk;
use serde::Serialize;

/// The words each dispatch copies.
const WORDS: u32 = 1024;

/// A loop-free kernel: `dst[i] = src[i] * 3 + 1` for `i < count`, `src` and `dst` storage
/// buffers (bindings 0 and 1 of set 0), `count` the push constant at offset 0.
const DESCRIPTOR_COPY: &str = r#"
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main" %gid %src %dst %pc
               OpExecutionMode %main LocalSize 64 1 1
               OpDecorate %gid BuiltIn GlobalInvocationId
               OpDecorate %arr ArrayStride 4
               OpMemberDecorate %Buf 0 Offset 0
               OpDecorate %Buf Block
               OpDecorate %src DescriptorSet 0
               OpDecorate %src Binding 0
               OpDecorate %dst DescriptorSet 0
               OpDecorate %dst Binding 1
               OpMemberDecorate %Pc 0 Offset 0
               OpDecorate %Pc Block
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
       %bool = OpTypeBool
       %uint = OpTypeInt 32 0
     %v3uint = OpTypeVector %uint 3
        %arr = OpTypeRuntimeArray %uint
        %Buf = OpTypeStruct %arr
         %Pc = OpTypeStruct %uint
  %in_v3uint = OpTypePointer Input %v3uint
    %in_uint = OpTypePointer Input %uint
     %sb_Buf = OpTypePointer StorageBuffer %Buf
    %sb_uint = OpTypePointer StorageBuffer %uint
      %pc_Pc = OpTypePointer PushConstant %Pc
    %pc_uint = OpTypePointer PushConstant %uint
     %uint_0 = OpConstant %uint 0
     %uint_1 = OpConstant %uint 1
     %uint_3 = OpConstant %uint 3
        %gid = OpVariable %in_v3uint Input
        %src = OpVariable %sb_Buf StorageBuffer
        %dst = OpVariable %sb_Buf StorageBuffer
         %pc = OpVariable %pc_Pc PushConstant
       %main = OpFunction %void None %fn
      %entry = OpLabel
      %gid_x = OpAccessChain %in_uint %gid %uint_0
          %i = OpLoad %uint %gid_x
    %count_p = OpAccessChain %pc_uint %pc %uint_0
      %count = OpLoad %uint %count_p
     %inside = OpULessThan %bool %i %count
               OpSelectionMerge %done None
               OpBranchConditional %inside %body %done
       %body = OpLabel
      %src_p = OpAccessChain %sb_uint %src %uint_0 %i
      %value = OpLoad %uint %src_p
     %scaled = OpIMul %uint %value %uint_3
     %result = OpIAdd %uint %scaled %uint_1
      %dst_p = OpAccessChain %sb_uint %dst %uint_0 %i
               OpStore %dst_p %result
               OpBranch %done
       %done = OpLabel
               OpReturn
               OpFunctionEnd
"#;

/// [`DESCRIPTOR_COPY`] writing through a device address instead: `src` is binding 0 of set 0,
/// and the push constants are `dst`'s address (offset 0) and `count` (offset 8).
const ADDRESS_COPY: &str = r#"
               OpCapability Shader
               OpCapability PhysicalStorageBufferAddresses
               OpMemoryModel PhysicalStorageBuffer64 GLSL450
               OpEntryPoint GLCompute %main "main" %gid %src %pc
               OpExecutionMode %main LocalSize 64 1 1
               OpDecorate %gid BuiltIn GlobalInvocationId
               OpDecorate %arr ArrayStride 4
               OpMemberDecorate %Buf 0 Offset 0
               OpDecorate %Buf Block
               OpDecorate %src DescriptorSet 0
               OpDecorate %src Binding 0
               OpMemberDecorate %Pc 0 Offset 0
               OpMemberDecorate %Pc 1 Offset 8
               OpDecorate %Pc Block
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
       %bool = OpTypeBool
       %uint = OpTypeInt 32 0
     %v3uint = OpTypeVector %uint 3
        %arr = OpTypeRuntimeArray %uint
        %Buf = OpTypeStruct %arr
    %psb_Buf = OpTypePointer PhysicalStorageBuffer %Buf
         %Pc = OpTypeStruct %psb_Buf %uint
  %in_v3uint = OpTypePointer Input %v3uint
    %in_uint = OpTypePointer Input %uint
     %sb_Buf = OpTypePointer StorageBuffer %Buf
    %sb_uint = OpTypePointer StorageBuffer %uint
   %psb_uint = OpTypePointer PhysicalStorageBuffer %uint
      %pc_Pc = OpTypePointer PushConstant %Pc
    %pc_uint = OpTypePointer PushConstant %uint
     %pc_psb = OpTypePointer PushConstant %psb_Buf
     %uint_0 = OpConstant %uint 0
     %uint_1 = OpConstant %uint 1
     %uint_3 = OpConstant %uint 3
        %gid = OpVariable %in_v3uint Input
        %src = OpVariable %sb_Buf StorageBuffer
         %pc = OpVariable %pc_Pc PushConstant
       %main = OpFunction %void None %fn
      %entry = OpLabel
      %gid_x = OpAccessChain %in_uint %gid %uint_0
          %i = OpLoad %uint %gid_x
    %count_p = OpAccessChain %pc_uint %pc %uint_1
      %count = OpLoad %uint %count_p
     %inside = OpULessThan %bool %i %count
               OpSelectionMerge %done None
               OpBranchConditional %inside %body %done
       %body = OpLabel
      %src_p = OpAccessChain %sb_uint %src %uint_0 %i
      %value = OpLoad %uint %src_p
     %scaled = OpIMul %uint %value %uint_3
     %result = OpIAdd %uint %scaled %uint_1
     %base_p = OpAccessChain %pc_psb %pc %uint_0
       %base = OpLoad %psb_Buf %base_p
      %dst_p = OpAccessChain %psb_uint %base %uint_0 %i
               OpStore %dst_p %result Aligned 4
               OpBranch %done
       %done = OpLabel
               OpReturn
               OpFunctionEnd
"#;

const EXTERNAL_MEMORY_FEATURES: &[(u64, &str)] = &[
    (0x1, "DEDICATED_ONLY"),
    (0x2, "EXPORTABLE"),
    (0x4, "IMPORTABLE"),
];

const HANDLE_TYPES: &[(u64, &str)] = &[
    (0x1, "OPAQUE_FD"),
    (0x2, "OPAQUE_WIN32"),
    (0x4, "OPAQUE_WIN32_KMT"),
    (0x80, "HOST_ALLOCATION"),
    (0x100, "HOST_MAPPED_FOREIGN_MEMORY"),
    (0x200, "DMA_BUF"),
    (0x10000, "MTLBUFFER"),
    (0x20000, "MTLTEXTURE"),
    (0x40000, "MTLHEAP"),
];

/// What the device answered about host memory without a copy.
#[derive(Debug, Default, Serialize)]
pub(super) struct HostMemory {
    /// The device lists `VK_EXT_external_memory_host`.
    listed: bool,
    /// The probe's context enabled it.
    enabled: bool,
    /// `VkPhysicalDeviceExternalMemoryHostPropertiesEXT::minImportedHostPointerAlignment`.
    min_imported_host_pointer_alignment: Option<u64>,
    /// Whether every page-aligned pointer and length (Metal's rule for a no-copy buffer) meets
    /// that alignment: the page size is a multiple of it.
    page_aligned_pointers_suffice: Option<bool>,
    /// `vkGetPhysicalDeviceExternalBufferProperties` for the probe's buffer over a host
    /// allocation.
    external_buffer: Option<ExternalBuffer>,
    /// The import of a real allocation, and the buffer and dispatches over it.
    import: Option<Import>,
    /// What stopped the step.
    error: Option<String>,
    /// The validation layer's messages during the step.
    pub(super) validation_messages: Vec<String>,
}

#[derive(Debug, Serialize)]
struct ExternalBuffer {
    features: Vec<String>,
    compatible_handle_types: Vec<String>,
}

#[derive(Debug, Default, Serialize)]
struct Import {
    /// The bytes of the host allocation imported, and its alignment (the larger of the page size
    /// and the device's minimum).
    bytes: u64,
    allocation_alignment: u64,
    /// The memory types `vkGetMemoryHostPointerPropertiesEXT` allows for the allocation.
    host_pointer_memory_types: Vec<MemoryType>,
    /// The memory types, and the bytes, a storage buffer over a host allocation of that size
    /// requires (`VkMemoryRequirements`).
    buffer_memory_type_bits: u32,
    buffer_requirement_bytes: u64,
    /// `VkMemoryDedicatedRequirements` of that buffer.
    buffer_requires_dedicated: bool,
    buffer_prefers_dedicated: bool,
    /// The memory type imported: the first host-visible, host-coherent one both allow.
    memory_type: Option<u32>,
    /// Whether the buffer bound to the imported memory has a nonzero device address (`None`
    /// where the context has no `bufferDeviceAddress`).
    device_address: Option<bool>,
    /// The dispatch that reads and writes the imported memory through two descriptors.
    descriptor_dispatch: Option<Dispatch>,
    /// The dispatch that reads it through a descriptor and writes it through its device address.
    address_dispatch: Option<Dispatch>,
}

/// One dispatch over the imported memory: the words the host found in its own allocation.
#[derive(Debug, Serialize)]
struct Dispatch {
    words: u32,
    matched: u32,
    /// `[index, expected, found]` of the first word that did not match.
    first_mismatch: Option<[u32; 3]>,
}

/// A host allocation aligned to `alignment` bytes, zeroed, freed when it drops.
struct HostAllocation {
    pointer: *mut u8,
    bytes: usize,
}

impl HostAllocation {
    fn new(bytes: usize, alignment: usize) -> Result<Self, String> {
        let mut pointer = std::ptr::null_mut();
        // SAFETY: `alignment` is a power of two and a multiple of the pointer size (a page size
        // or a device's power-of-two minimum, at least 4096 here); `pointer` is written only on
        // success.
        let status = unsafe { libc::posix_memalign(&mut pointer, alignment, bytes) };
        if status != 0 || pointer.is_null() {
            return Err(format!(
                "posix_memalign({alignment}, {bytes}) failed with status {status}"
            ));
        }
        let pointer = pointer.cast::<u8>();
        // SAFETY: the allocation holds `bytes` bytes.
        unsafe { std::ptr::write_bytes(pointer, 0, bytes) };
        Ok(HostAllocation { pointer, bytes })
    }

    /// The allocation as words. Only while no device work that uses it is in flight.
    fn words(&mut self) -> &mut [u32] {
        // SAFETY: the allocation is live, aligned for `u32`, and holds `bytes` bytes; `&mut self`
        // is the only access from the host, and the caller's contract keeps the device out.
        unsafe { std::slice::from_raw_parts_mut(self.pointer.cast::<u32>(), self.bytes / 4) }
    }
}

impl Drop for HostAllocation {
    fn drop(&mut self) {
        // SAFETY: `pointer` came from `posix_memalign`, and every import of it was freed first
        // (the step's objects drop before its allocation).
        unsafe { libc::free(self.pointer.cast()) };
    }
}

pub(super) fn measure(context: &VulkanContext) -> HostMemory {
    let mut out = HostMemory {
        enabled: context
            .runtime_probe
            .is_some_and(|probe| probe.external_memory_host),
        ..HostMemory::default()
    };
    if let Err(error) = measure_into(context, &mut out) {
        out.error = Some(error);
    }
    out.validation_messages = context.take_messages();
    out
}

fn measure_into(context: &VulkanContext, out: &mut HostMemory) -> Result<(), String> {
    out.listed = device_lists(context, ash::ext::external_memory_host::NAME)?;
    if !out.listed {
        return Ok(());
    }
    let mut host = vk::PhysicalDeviceExternalMemoryHostPropertiesEXT::default();
    let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut host);
    // SAFETY: the physical device belongs to the live instance and lists the extension whose
    // structure is chained.
    unsafe {
        context
            .instance
            .get_physical_device_properties2(context.physical, &mut properties)
    };
    let alignment = host.min_imported_host_pointer_alignment;
    out.min_imported_host_pointer_alignment = Some(alignment);
    let page = host_page_size();
    out.page_aligned_pointers_suffice = Some(alignment != 0 && page.is_multiple_of(alignment));
    let address = buffer_device_address(context);
    let usage = buffer_usage(address);
    let info = vk::PhysicalDeviceExternalBufferInfo::default()
        .usage(usage)
        .handle_type(vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT);
    let mut external = vk::ExternalBufferProperties::default();
    // SAFETY: as above.
    unsafe {
        context
            .instance
            .get_physical_device_external_buffer_properties(context.physical, &info, &mut external)
    };
    let memory = external.external_memory_properties;
    out.external_buffer = Some(ExternalBuffer {
        features: flag_names(
            u64::from(memory.external_memory_features.as_raw()),
            EXTERNAL_MEMORY_FEATURES,
        ),
        compatible_handle_types: flag_names(
            u64::from(memory.compatible_handle_types.as_raw()),
            HANDLE_TYPES,
        ),
    });
    if !out.enabled {
        return Ok(());
    }
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(format!(
            "minImportedHostPointerAlignment {alignment} is not a power of two"
        ));
    }
    let import = out.import.insert(Import::default());
    import_and_dispatch(context, alignment.max(page), address, import)
}

/// The probe buffer's usage: a storage buffer that copies can reach, with a device address where
/// the context has them.
fn buffer_usage(address: bool) -> vk::BufferUsageFlags {
    let usage = vk::BufferUsageFlags::STORAGE_BUFFER
        | vk::BufferUsageFlags::TRANSFER_SRC
        | vk::BufferUsageFlags::TRANSFER_DST;
    if address {
        usage | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
    } else {
        usage
    }
}

fn import_and_dispatch(
    context: &VulkanContext,
    unit: u64,
    address: bool,
    out: &mut Import,
) -> Result<(), String> {
    let properties_of = context
        .memory_host_pointer_properties()
        .ok_or("vkGetMemoryHostPointerPropertiesEXT did not load")?;
    // Four units, at least two regions of `WORDS` words.
    let bytes = unit * 4;
    out.bytes = bytes;
    out.allocation_alignment = unit;
    let size = usize::try_from(bytes).map_err(|_| "the allocation does not fit in usize")?;
    let alignment = usize::try_from(unit).map_err(|_| "the alignment does not fit in usize")?;
    // Declared before the objects, so it is freed after the memory importing it.
    let mut host = HostAllocation::new(size, alignment)?;
    let mut pointer_properties = vk::MemoryHostPointerPropertiesEXT::default();
    // SAFETY: the context enabled the extension; the pointer is aligned to the device's minimum
    // and its allocation outlives every use of it.
    let result = unsafe {
        properties_of(
            context.device.handle(),
            vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT,
            host.pointer.cast(),
            &mut pointer_properties,
        )
    };
    if result != vk::Result::SUCCESS {
        return Err(format!(
            "vkGetMemoryHostPointerPropertiesEXT: {}",
            result_name(result)
        ));
    }
    out.host_pointer_memory_types = memory_types(context, pointer_properties.memory_type_bits);
    let mut owned = Owned::new(context);
    let mut external = vk::ExternalMemoryBufferCreateInfo::default()
        .handle_types(vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT);
    let buffer = owned.buffer(
        &vk::BufferCreateInfo::default()
            .size(bytes)
            .usage(buffer_usage(address))
            .push_next(&mut external),
    )?;
    let dedicated = buffer_requirements(context, buffer);
    let requirements = dedicated.memory;
    out.buffer_memory_type_bits = requirements.memory_type_bits;
    out.buffer_requirement_bytes = requirements.size;
    out.buffer_requires_dedicated = dedicated.requires_dedicated;
    out.buffer_prefers_dedicated = dedicated.prefers_dedicated;
    if dedicated.requires_dedicated {
        // A host-pointer import may not be a dedicated allocation (VUID-VkMemoryAllocateInfo-
        // pNext-01874).
        return Err(
            "a storage buffer over a host allocation requires a dedicated allocation, \
                    which a host-pointer import cannot be"
                .into(),
        );
    }
    if requirements.size > bytes {
        return Err(format!(
            "a buffer of {bytes} bytes over a host allocation requires {} bytes of memory",
            requirements.size
        ));
    }
    let bits = pointer_properties.memory_type_bits & requirements.memory_type_bits;
    let memory_type = host_coherent_type(context, bits).ok_or_else(|| {
        format!(
            "no host-visible, host-coherent memory type among the host pointer's {:#x} and the \
             buffer's {:#x}",
            pointer_properties.memory_type_bits, requirements.memory_type_bits
        )
    })?;
    out.memory_type = Some(memory_type);
    let mut import = vk::ImportMemoryHostPointerInfoEXT::default()
        .handle_type(vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT)
        .host_pointer(host.pointer.cast());
    let mut flags =
        vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
    let mut allocate = vk::MemoryAllocateInfo::default()
        .allocation_size(bytes)
        .memory_type_index(memory_type)
        .push_next(&mut import);
    if address {
        allocate = allocate.push_next(&mut flags);
    }
    let memory = owned.memory(&allocate)?;
    // SAFETY: the buffer and memory are this device's, the buffer is unbound, the type is one
    // its requirements allow, and the memory holds its required size.
    unsafe { context.device.bind_buffer_memory(buffer, memory, 0) }
        .map_err(|error| format!("bind the buffer to imported memory: {}", result_name(error)))?;
    let device_address = address.then(|| {
        // SAFETY: the buffer has device-address usage and is bound to memory allocated with the
        // device-address flag.
        unsafe {
            context
                .device
                .get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer))
        }
    });
    out.device_address = device_address.map(|value| value != 0);
    out.descriptor_dispatch = Some(dispatch(&mut owned, &mut host, buffer, None)?);
    if let Some(base) = device_address.filter(|&value| value != 0) {
        out.address_dispatch = Some(dispatch(&mut owned, &mut host, buffer, Some(base))?);
    }
    Ok(())
}

/// Run one copy dispatch over `buffer` (the imported memory): it reads `WORDS` words at the
/// allocation's start and writes their images at its middle, through a second descriptor or, with
/// `address`, through the buffer's device address. The host then reads the results from its own
/// allocation.
fn dispatch(
    owned: &mut Owned<'_>,
    host: &mut HostAllocation,
    buffer: vk::Buffer,
    address: Option<u64>,
) -> Result<Dispatch, String> {
    let half = host.bytes as u64 / 2;
    let region = u64::from(WORDS) * 4;
    let first_result = (half / 4) as usize;
    {
        let words = host.words();
        for index in 0..WORDS {
            words[index as usize] = dispatch_word(index).0;
            words[first_result + index as usize] = 0;
        }
    }
    let (label, assembly, bindings, push_bytes) = match address {
        None => ("descriptor", DESCRIPTOR_COPY, 2, 4),
        Some(_) => ("address", ADDRESS_COPY, 1, 16),
    };
    let spv = crate::spirv_tools::assemble_validated(assembly, &format!("runtime-probe-{label}"))?;
    let layout_bindings = (0..bindings)
        .map(|binding| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(binding)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        })
        .collect::<Vec<_>>();
    let set_layout = owned.descriptor_set_layout(&layout_bindings)?;
    let layout = owned.pipeline_layout(set_layout, push_bytes)?;
    let pipeline = owned.compute_pipeline(&spv, &format!("runtime probe {label} copy"), layout)?;
    let set = owned.descriptor_set(set_layout, bindings)?;
    let ranges = [(0, region), (half, region)];
    let infos = ranges
        .iter()
        .take(bindings as usize)
        .map(|&(offset, range)| {
            [vk::DescriptorBufferInfo::default()
                .buffer(buffer)
                .offset(offset)
                .range(range)]
        })
        .collect::<Vec<_>>();
    let writes = infos
        .iter()
        .enumerate()
        .map(|(binding, info)| {
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(binding as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(info)
        })
        .collect::<Vec<_>>();
    // SAFETY: the set and buffer are live objects of this device, and nothing uses the set yet.
    unsafe { owned.device().update_descriptor_sets(&writes, &[]) };
    let push = match address {
        None => WORDS.to_ne_bytes().to_vec(),
        Some(base) => [
            (base + half).to_ne_bytes().as_slice(),
            WORDS.to_ne_bytes().as_slice(),
            0u32.to_ne_bytes().as_slice(),
        ]
        .concat(),
    };
    owned.submit(
        &format!("the runtime probe's {label} dispatch"),
        |device, command| {
            let barrier = buffer_barrier(
                buffer,
                vk::AccessFlags::SHADER_WRITE,
                vk::AccessFlags::HOST_READ,
            );
            // SAFETY: the command buffer is recording; every object it names is live.
            unsafe {
                device.cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, pipeline);
                device.cmd_bind_descriptor_sets(
                    command,
                    vk::PipelineBindPoint::COMPUTE,
                    layout,
                    0,
                    &[set],
                    &[],
                );
                device.cmd_push_constants(command, layout, vk::ShaderStageFlags::COMPUTE, 0, &push);
                device.cmd_dispatch(command, WORDS.div_ceil(64), 1, 1);
                device.cmd_pipeline_barrier(
                    command,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[barrier],
                    &[],
                );
            }
        },
    )?;
    let words = host.words();
    let mut matched = 0;
    let mut first_mismatch = None;
    for index in 0..WORDS {
        let expected = dispatch_word(index).1;
        let found = words[first_result + index as usize];
        if found == expected {
            matched += 1;
        } else if first_mismatch.is_none() {
            first_mismatch = Some([index, expected, found]);
        }
    }
    Ok(Dispatch {
        words: WORDS,
        matched,
        first_mismatch,
    })
}
