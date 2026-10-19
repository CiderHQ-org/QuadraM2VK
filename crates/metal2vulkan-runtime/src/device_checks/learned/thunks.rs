//! The loss watch's entry points (`learned.rs`), one per watched call and slot, and each slot's
//! calls as its device's driver gave them: each calls the driver's and, where it returns
//! `VK_ERROR_DEVICE_LOST`, has the slot's watch record the loss before it returns.
//!
//! The calls are every one of `ash`'s Vulkan 1.0 to 1.3 device tables that returns a `VkResult`
//! and takes a `VkDevice` or a `VkQueue` first: the submissions, waits and status reads, and
//! every making, allocation, mapping and binding, since a driver may answer any of them so once
//! its device is lost. A command buffer's calls (its begin and end) and the extension calls the
//! runtime loads itself go to the driver directly.

use super::Watch;
use crate::recorder::SLOTS;
use ash::vk;
use std::ffi::{c_void, CStr};
use std::sync::atomic::{AtomicBool, AtomicPtr, Ordering};
use std::sync::Arc;

/// The slots taken.
static TAKEN: [AtomicBool; SLOTS] = [const { AtomicBool::new(false) }; SLOTS];

/// Each slot's watch: null where no device holds the slot.
static WATCHES: [AtomicPtr<Watch>; SLOTS] = [const { AtomicPtr::new(std::ptr::null_mut()) }; SLOTS];

/// A free slot, taken for `watch`: its index; none where every slot is taken.
pub(super) fn take(watch: &Arc<Watch>) -> Option<usize> {
    let index = TAKEN.iter().position(|taken| {
        taken
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
            .is_ok()
    })?;
    WATCHES[index].store(Arc::as_ptr(watch).cast_mut(), Ordering::Release);
    Some(index)
}

/// Let go of slot `index`: its watch and its calls forgotten, the slot free for the next device.
/// After its device is destroyed, so no call of it is made from here on.
pub(super) fn let_go(index: usize) {
    WATCHES[index].store(std::ptr::null_mut(), Ordering::Release);
    clear(index);
    TAKEN[index].store(false, Ordering::Release);
}

/// A watched call whose driver's entry point was not stored for the slot whose entry point was
/// called: a device's functions routed before its calls were (a defect of the runtime's).
#[cold]
fn missing(name: &CStr) -> ! {
    panic!("the watched call {name:?} has no driver's entry point");
}

/// `call` of slot `slot`'s device returned `VK_ERROR_DEVICE_LOST`: its watch records the loss.
#[cold]
#[inline(never)]
fn lost(slot: usize, call: &'static CStr) {
    let watch = WATCHES[slot].load(Ordering::Acquire);
    if watch.is_null() {
        return;
    }
    // SAFETY: a slot's watch is held by its `Slot`, which lets go of the slot (nulling this
    // pointer) after its device is destroyed, and a call of the device is made only before.
    unsafe { &*watch }.lost(call);
}

/// Declare the watched calls: those of `ash`'s 1.0, 1.1, 1.2 and 1.3 tables, each with its field
/// in its table, its Vulkan name and its parameters.
macro_rules! watched {
    (
        v1_0 { $($f0:ident $n0:literal ($($a0:ident: $t0:ty),*);)* }
        v1_1 { $($f1:ident $n1:literal ($($a1:ident: $t1:ty),*);)* }
        v1_2 { $($f2:ident $n2:literal ($($a2:ident: $t2:ty),*);)* }
        v1_3 { $($f3:ident $n3:literal ($($a3:ident: $t3:ty),*);)* }
    ) => {
        /// Each slot's calls, as their devices' drivers gave them; null where none is stored.
        struct Real {
            $($f0: [AtomicPtr<c_void>; SLOTS],)*
            $($f1: [AtomicPtr<c_void>; SLOTS],)*
            $($f2: [AtomicPtr<c_void>; SLOTS],)*
            $($f3: [AtomicPtr<c_void>; SLOTS],)*
        }

        static REAL: Real = Real {
            $($f0: [const { AtomicPtr::new(std::ptr::null_mut()) }; SLOTS],)*
            $($f1: [const { AtomicPtr::new(std::ptr::null_mut()) }; SLOTS],)*
            $($f2: [const { AtomicPtr::new(std::ptr::null_mut()) }; SLOTS],)*
            $($f3: [const { AtomicPtr::new(std::ptr::null_mut()) }; SLOTS],)*
        };

        watched!(@thunks $($f0 $n0 ($($a0: $t0),*);)*);
        watched!(@thunks $($f1 $n1 ($($a1: $t1),*);)*);
        watched!(@thunks $($f2 $n2 ($($a2: $t2),*);)*);
        watched!(@thunks $($f3 $n3 ($($a3: $t3),*);)*);

        /// Slot `S`'s entry point in place of the driver's `name`, if it is a watched call.
        pub(super) fn entry<const S: usize>(name: &CStr) -> Option<*const c_void> {
            $(if name == $n0 { return Some($f0::<S> as *const c_void); })*
            $(if name == $n1 { return Some($f1::<S> as *const c_void); })*
            $(if name == $n2 { return Some($f2::<S> as *const c_void); })*
            $(if name == $n3 { return Some($f3::<S> as *const c_void); })*
            None
        }

        /// Forget slot `index`'s calls.
        fn clear(index: usize) {
            $(REAL.$f0[index].store(std::ptr::null_mut(), Ordering::Release);)*
            $(REAL.$f1[index].store(std::ptr::null_mut(), Ordering::Release);)*
            $(REAL.$f2[index].store(std::ptr::null_mut(), Ordering::Release);)*
            $(REAL.$f3[index].store(std::ptr::null_mut(), Ordering::Release);)*
        }

        /// Route the tables' watched calls through slot `S`: each one's driver's entry point
        /// stored, the slot's in its place.
        pub(super) fn route_tables<const S: usize>(
            v1_0: &mut ash::DeviceFnV1_0,
            v1_1: &mut ash::DeviceFnV1_1,
            v1_2: &mut ash::DeviceFnV1_2,
            v1_3: &mut ash::DeviceFnV1_3,
        ) {
            $(
                REAL.$f0[S].store((v1_0.$f0 as *const c_void).cast_mut(), Ordering::Release);
                v1_0.$f0 = $f0::<S>;
            )*
            $(
                REAL.$f1[S].store((v1_1.$f1 as *const c_void).cast_mut(), Ordering::Release);
                v1_1.$f1 = $f1::<S>;
            )*
            $(
                REAL.$f2[S].store((v1_2.$f2 as *const c_void).cast_mut(), Ordering::Release);
                v1_2.$f2 = $f2::<S>;
            )*
            $(
                REAL.$f3[S].store((v1_3.$f3 as *const c_void).cast_mut(), Ordering::Release);
                v1_3.$f3 = $f3::<S>;
            )*
        }
    };
    (@thunks $($f:ident $n:literal ($($a:ident: $t:ty),*);)*) => {
        $(
            /// The call through slot `S` (the module docs).
            unsafe extern "system" fn $f<const S: usize>($($a: $t),*) -> vk::Result {
                let real = REAL.$f[S].load(Ordering::Acquire);
                if real.is_null() {
                    missing($n);
                }
                // SAFETY: the slot's device's driver gave this call (`route_tables`), of this
                // signature; the caller's arguments are as the call takes them.
                let result = unsafe {
                    let real = std::mem::transmute::<
                        *mut c_void,
                        unsafe extern "system" fn($($t),*) -> vk::Result,
                    >(real);
                    real($($a),*)
                };
                if result == vk::Result::ERROR_DEVICE_LOST {
                    lost(S, $n);
                }
                result
            }
        )*
    };
}

watched! {
    v1_0 {
        queue_submit c"vkQueueSubmit"
            (
                queue: vk::Queue,
                submit_count: u32,
                p_submits: *const vk::SubmitInfo<'_>,
                fence: vk::Fence
            );
        queue_wait_idle c"vkQueueWaitIdle" (queue: vk::Queue);
        device_wait_idle c"vkDeviceWaitIdle" (device: vk::Device);
        allocate_memory c"vkAllocateMemory"
            (
                device: vk::Device,
                p_allocate_info: *const vk::MemoryAllocateInfo<'_>,
                p_allocator: *const vk::AllocationCallbacks<'_>,
                p_memory: *mut vk::DeviceMemory
            );
        map_memory c"vkMapMemory"
            (
                device: vk::Device,
                memory: vk::DeviceMemory,
                offset: vk::DeviceSize,
                size: vk::DeviceSize,
                flags: vk::MemoryMapFlags,
                pp_data: *mut *mut c_void
            );
        flush_mapped_memory_ranges c"vkFlushMappedMemoryRanges"
            (
                device: vk::Device,
                memory_range_count: u32,
                p_memory_ranges: *const vk::MappedMemoryRange<'_>
            );
        invalidate_mapped_memory_ranges c"vkInvalidateMappedMemoryRanges"
            (
                device: vk::Device,
                memory_range_count: u32,
                p_memory_ranges: *const vk::MappedMemoryRange<'_>
            );
        bind_buffer_memory c"vkBindBufferMemory"
            (
                device: vk::Device,
                buffer: vk::Buffer,
                memory: vk::DeviceMemory,
                memory_offset: vk::DeviceSize
            );
        bind_image_memory c"vkBindImageMemory"
            (
                device: vk::Device,
                image: vk::Image,
                memory: vk::DeviceMemory,
                memory_offset: vk::DeviceSize
            );
        queue_bind_sparse c"vkQueueBindSparse"
            (
                queue: vk::Queue,
                bind_info_count: u32,
                p_bind_info: *const vk::BindSparseInfo<'_>,
                fence: vk::Fence
            );
        create_fence c"vkCreateFence"
            (
                device: vk::Device,
                p_create_info: *const vk::FenceCreateInfo<'_>,
                p_allocator: *const vk::AllocationCallbacks<'_>,
                p_fence: *mut vk::Fence
            );
        reset_fences c"vkResetFences"
            (device: vk::Device, fence_count: u32, p_fences: *const vk::Fence);
        get_fence_status c"vkGetFenceStatus" (device: vk::Device, fence: vk::Fence);
        wait_for_fences c"vkWaitForFences"
            (
                device: vk::Device,
                fence_count: u32,
                p_fences: *const vk::Fence,
                wait_all: vk::Bool32,
                timeout: u64
            );
        create_semaphore c"vkCreateSemaphore"
            (
                device: vk::Device,
                p_create_info: *const vk::SemaphoreCreateInfo<'_>,
                p_allocator: *const vk::AllocationCallbacks<'_>,
                p_semaphore: *mut vk::Semaphore
            );
        create_event c"vkCreateEvent"
            (
                device: vk::Device,
                p_create_info: *const vk::EventCreateInfo<'_>,
                p_allocator: *const vk::AllocationCallbacks<'_>,
                p_event: *mut vk::Event
            );
        get_event_status c"vkGetEventStatus" (device: vk::Device, event: vk::Event);
        set_event c"vkSetEvent" (device: vk::Device, event: vk::Event);
        reset_event c"vkResetEvent" (device: vk::Device, event: vk::Event);
        create_query_pool c"vkCreateQueryPool"
            (
                device: vk::Device,
                p_create_info: *const vk::QueryPoolCreateInfo<'_>,
                p_allocator: *const vk::AllocationCallbacks<'_>,
                p_query_pool: *mut vk::QueryPool
            );
        get_query_pool_results c"vkGetQueryPoolResults"
            (
                device: vk::Device,
                query_pool: vk::QueryPool,
                first_query: u32,
                query_count: u32,
                data_size: usize,
                p_data: *mut c_void,
                stride: vk::DeviceSize,
                flags: vk::QueryResultFlags
            );
        create_buffer c"vkCreateBuffer"
            (
                device: vk::Device,
                p_create_info: *const vk::BufferCreateInfo<'_>,
                p_allocator: *const vk::AllocationCallbacks<'_>,
                p_buffer: *mut vk::Buffer
            );
        create_buffer_view c"vkCreateBufferView"
            (
                device: vk::Device,
                p_create_info: *const vk::BufferViewCreateInfo<'_>,
                p_allocator: *const vk::AllocationCallbacks<'_>,
                p_view: *mut vk::BufferView
            );
        create_image c"vkCreateImage"
            (
                device: vk::Device,
                p_create_info: *const vk::ImageCreateInfo<'_>,
                p_allocator: *const vk::AllocationCallbacks<'_>,
                p_image: *mut vk::Image
            );
        create_image_view c"vkCreateImageView"
            (
                device: vk::Device,
                p_create_info: *const vk::ImageViewCreateInfo<'_>,
                p_allocator: *const vk::AllocationCallbacks<'_>,
                p_view: *mut vk::ImageView
            );
        create_shader_module c"vkCreateShaderModule"
            (
                device: vk::Device,
                p_create_info: *const vk::ShaderModuleCreateInfo<'_>,
                p_allocator: *const vk::AllocationCallbacks<'_>,
                p_shader_module: *mut vk::ShaderModule
            );
        create_pipeline_cache c"vkCreatePipelineCache"
            (
                device: vk::Device,
                p_create_info: *const vk::PipelineCacheCreateInfo<'_>,
                p_allocator: *const vk::AllocationCallbacks<'_>,
                p_pipeline_cache: *mut vk::PipelineCache
            );
        get_pipeline_cache_data c"vkGetPipelineCacheData"
            (
                device: vk::Device,
                pipeline_cache: vk::PipelineCache,
                p_data_size: *mut usize,
                p_data: *mut c_void
            );
        merge_pipeline_caches c"vkMergePipelineCaches"
            (
                device: vk::Device,
                dst_cache: vk::PipelineCache,
                src_cache_count: u32,
                p_src_caches: *const vk::PipelineCache
            );
        create_graphics_pipelines c"vkCreateGraphicsPipelines"
            (
                device: vk::Device,
                pipeline_cache: vk::PipelineCache,
                create_info_count: u32,
                p_create_infos: *const vk::GraphicsPipelineCreateInfo<'_>,
                p_allocator: *const vk::AllocationCallbacks<'_>,
                p_pipelines: *mut vk::Pipeline
            );
        create_compute_pipelines c"vkCreateComputePipelines"
            (
                device: vk::Device,
                pipeline_cache: vk::PipelineCache,
                create_info_count: u32,
                p_create_infos: *const vk::ComputePipelineCreateInfo<'_>,
                p_allocator: *const vk::AllocationCallbacks<'_>,
                p_pipelines: *mut vk::Pipeline
            );
        create_pipeline_layout c"vkCreatePipelineLayout"
            (
                device: vk::Device,
                p_create_info: *const vk::PipelineLayoutCreateInfo<'_>,
                p_allocator: *const vk::AllocationCallbacks<'_>,
                p_pipeline_layout: *mut vk::PipelineLayout
            );
        create_sampler c"vkCreateSampler"
            (
                device: vk::Device,
                p_create_info: *const vk::SamplerCreateInfo<'_>,
                p_allocator: *const vk::AllocationCallbacks<'_>,
                p_sampler: *mut vk::Sampler
            );
        create_descriptor_set_layout c"vkCreateDescriptorSetLayout"
            (
                device: vk::Device,
                p_create_info: *const vk::DescriptorSetLayoutCreateInfo<'_>,
                p_allocator: *const vk::AllocationCallbacks<'_>,
                p_set_layout: *mut vk::DescriptorSetLayout
            );
        create_descriptor_pool c"vkCreateDescriptorPool"
            (
                device: vk::Device,
                p_create_info: *const vk::DescriptorPoolCreateInfo<'_>,
                p_allocator: *const vk::AllocationCallbacks<'_>,
                p_descriptor_pool: *mut vk::DescriptorPool
            );
        reset_descriptor_pool c"vkResetDescriptorPool"
            (
                device: vk::Device,
                descriptor_pool: vk::DescriptorPool,
                flags: vk::DescriptorPoolResetFlags
            );
        allocate_descriptor_sets c"vkAllocateDescriptorSets"
            (
                device: vk::Device,
                p_allocate_info: *const vk::DescriptorSetAllocateInfo<'_>,
                p_descriptor_sets: *mut vk::DescriptorSet
            );
        free_descriptor_sets c"vkFreeDescriptorSets"
            (
                device: vk::Device,
                descriptor_pool: vk::DescriptorPool,
                descriptor_set_count: u32,
                p_descriptor_sets: *const vk::DescriptorSet
            );
        create_framebuffer c"vkCreateFramebuffer"
            (
                device: vk::Device,
                p_create_info: *const vk::FramebufferCreateInfo<'_>,
                p_allocator: *const vk::AllocationCallbacks<'_>,
                p_framebuffer: *mut vk::Framebuffer
            );
        create_render_pass c"vkCreateRenderPass"
            (
                device: vk::Device,
                p_create_info: *const vk::RenderPassCreateInfo<'_>,
                p_allocator: *const vk::AllocationCallbacks<'_>,
                p_render_pass: *mut vk::RenderPass
            );
        create_command_pool c"vkCreateCommandPool"
            (
                device: vk::Device,
                p_create_info: *const vk::CommandPoolCreateInfo<'_>,
                p_allocator: *const vk::AllocationCallbacks<'_>,
                p_command_pool: *mut vk::CommandPool
            );
        reset_command_pool c"vkResetCommandPool"
            (device: vk::Device, command_pool: vk::CommandPool, flags: vk::CommandPoolResetFlags);
        allocate_command_buffers c"vkAllocateCommandBuffers"
            (
                device: vk::Device,
                p_allocate_info: *const vk::CommandBufferAllocateInfo<'_>,
                p_command_buffers: *mut vk::CommandBuffer
            );
    }
    v1_1 {
        bind_buffer_memory2 c"vkBindBufferMemory2"
            (
                device: vk::Device,
                bind_info_count: u32,
                p_bind_infos: *const vk::BindBufferMemoryInfo<'_>
            );
        bind_image_memory2 c"vkBindImageMemory2"
            (
                device: vk::Device,
                bind_info_count: u32,
                p_bind_infos: *const vk::BindImageMemoryInfo<'_>
            );
        create_sampler_ycbcr_conversion c"vkCreateSamplerYcbcrConversion"
            (
                device: vk::Device,
                p_create_info: *const vk::SamplerYcbcrConversionCreateInfo<'_>,
                p_allocator: *const vk::AllocationCallbacks<'_>,
                p_ycbcr_conversion: *mut vk::SamplerYcbcrConversion
            );
        create_descriptor_update_template c"vkCreateDescriptorUpdateTemplate"
            (
                device: vk::Device,
                p_create_info: *const vk::DescriptorUpdateTemplateCreateInfo<'_>,
                p_allocator: *const vk::AllocationCallbacks<'_>,
                p_descriptor_update_template: *mut vk::DescriptorUpdateTemplate
            );
    }
    v1_2 {
        create_render_pass2 c"vkCreateRenderPass2"
            (
                device: vk::Device,
                p_create_info: *const vk::RenderPassCreateInfo2<'_>,
                p_allocator: *const vk::AllocationCallbacks<'_>,
                p_render_pass: *mut vk::RenderPass
            );
        get_semaphore_counter_value c"vkGetSemaphoreCounterValue"
            (device: vk::Device, semaphore: vk::Semaphore, p_value: *mut u64);
        wait_semaphores c"vkWaitSemaphores"
            (device: vk::Device, p_wait_info: *const vk::SemaphoreWaitInfo<'_>, timeout: u64);
        signal_semaphore c"vkSignalSemaphore"
            (device: vk::Device, p_signal_info: *const vk::SemaphoreSignalInfo<'_>);
    }
    v1_3 {
        create_private_data_slot c"vkCreatePrivateDataSlot"
            (
                device: vk::Device,
                p_create_info: *const vk::PrivateDataSlotCreateInfo<'_>,
                p_allocator: *const vk::AllocationCallbacks<'_>,
                p_private_data_slot: *mut vk::PrivateDataSlot
            );
        set_private_data c"vkSetPrivateData"
            (
                device: vk::Device,
                object_type: vk::ObjectType,
                object_handle: u64,
                private_data_slot: vk::PrivateDataSlot,
                data: u64
            );
        queue_submit2 c"vkQueueSubmit2"
            (
                queue: vk::Queue,
                submit_count: u32,
                p_submits: *const vk::SubmitInfo2<'_>,
                fence: vk::Fence
            );
    }
}
