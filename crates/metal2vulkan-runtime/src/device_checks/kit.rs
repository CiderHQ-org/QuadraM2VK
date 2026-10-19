//! What the checks are made of: a command pool on the device's first queue and one command buffer
//! of it, begun again for each submission, host-visible buffers of words, device-local images, the
//! pipelines a check makes, and one submission at a time, waited for at most [`TIMEOUT`] and timed
//! by GPU timestamps where the queue writes them. What a check makes is released as it ends, after
//! its last submission completed; after a submission that timed out nothing is released (the GPU
//! may still use it) and no later check runs.
//!
//! - **As the device is made** ([`Kit::making`]), on the creating thread: the raw device and its
//!   first queue, which nothing else uses until the device is assembled.
//! - **On a running device** ([`Kit::running`]), from its checks thread
//!   (`device_checks/background.rs`): the kit holds the device weakly and holds it for one call
//!   at a time through its checks' gate (`WeakDevice::with`, `live.rs`), each call inside a section
//!   of the exit guard with the device's functions (the guard's table alone, in builds with debug
//!   assertions: a check that calls another entry point there ends the process). Each object is
//!   kept by the device as soon as it is made (`device_checks/live.rs`), so a device dropped
//!   between two calls ends the check, and its teardown releases what the check made after its
//!   idle wait. A submission goes to the device's first queue under its lock
//!   (`DeviceShared::use_queue`), as a check's (`By::Check`): no counter of the queue's counts
//!   it, and a failure the driver relays for it fails the check alone. Its wait is cut into slices
//!   of [`SLICE`], none of which holds the device past it, so the drop of its last handle waits
//!   at most one call.

use crate::device::relay::{By, Relayed};
use crate::device::{DeviceShared, WeakDevice};
use crate::draw_source::SizedIndexBind;
use crate::exit_guard::{self, Call};
use crate::loader_word::LoaderWord;
use crate::release;
use crate::render_state::DepthBiasCommand;
use ash::vk::{self, Handle};
use metal2vulkan::bind::CheckKernel;
use metal2vulkan::device::{ApiVersion, DeviceChecks, DeviceDescription, EnableSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// The longest a check's submission is waited for.
pub(crate) const TIMEOUT: Duration = Duration::from_secs(2);

/// The longest one wait of a running device's check holds the device: its submission's wait is
/// cut into such slices.
const SLICE: Duration = Duration::from_millis(10);

/// The commands the checks record into a submission, which each submission's section on a running
/// device marks as called ([`exit_guard::called`]): a superset of what one submission records, as
/// a host step's are ([`exit_guard::HOST_STEP`]). A check that records another table command adds
/// it here.
pub(crate) const RECORDED: &[Call] = &[
    Call::BeginCommandBuffer,
    Call::EndCommandBuffer,
    Call::QueueSubmit,
    Call::CmdResetQueryPool,
    Call::CmdWriteTimestamp,
    Call::CmdPipelineBarrier,
    Call::CmdBindPipeline,
    Call::CmdBindDescriptorSets,
    Call::CmdPushConstants,
    Call::CmdDispatch,
    Call::CmdBeginRenderPass,
    Call::CmdEndRenderPass,
    Call::CmdSetViewport,
    Call::CmdSetScissor,
    Call::CmdDraw,
    Call::CmdBindIndexBuffer,
    Call::CmdBindIndexBuffer2,
    Call::CmdDrawIndexedIndirect,
    Call::CmdClearColorImage,
    Call::CmdClearDepthStencilImage,
    Call::CmdCopyImageToBuffer,
    Call::CmdCopyBufferToImage,
    Call::CmdSetDepthBias,
    Call::CmdSetDepthBias2,
    Call::CmdCopyImage,
    Call::CmdCopyBuffer,
    Call::CmdPushDescriptorSet,
    Call::CmdBeginQuery,
    Call::CmdEndQuery,
];

/// Why a check found no answer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CheckError(pub(crate) String);

impl From<vk::Result> for CheckError {
    fn from(result: vk::Result) -> CheckError {
        CheckError(format!("{result:?}"))
    }
}

impl From<crate::Error> for CheckError {
    /// The creation door's refusal or failure (`creation.rs`) as a check's.
    fn from(error: crate::Error) -> CheckError {
        CheckError(error.to_string())
    }
}

/// One object a kit made, released as its check ends.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Made {
    Buffer(vk::Buffer),
    Memory(vk::DeviceMemory),
    Image(vk::Image),
    View(vk::ImageView),
    Shader(vk::ShaderModule),
    SetLayout(vk::DescriptorSetLayout),
    DescriptorPool(vk::DescriptorPool),
    PipelineLayout(vk::PipelineLayout),
    Pipeline(vk::Pipeline),
    RenderPass(vk::RenderPass),
    Framebuffer(vk::Framebuffer),
    QueryPool(vk::QueryPool),
    Fence(vk::Fence),
    CommandPool(vk::CommandPool),
    Sampler(vk::Sampler),
}

impl Made {
    /// Release it.
    ///
    /// # Safety
    ///
    /// `device` made it, no pending work uses it, and it is released once.
    pub(crate) unsafe fn release(self, device: &ash::Device) {
        // SAFETY: the caller's.
        unsafe {
            match self {
                Made::Buffer(o) => release::object(device, o),
                Made::Memory(o) => release::object(device, o),
                Made::Image(o) => release::object(device, o),
                Made::View(o) => release::object(device, o),
                Made::Shader(o) => release::object(device, o),
                Made::SetLayout(o) => release::object(device, o),
                Made::DescriptorPool(o) => release::object(device, o),
                Made::PipelineLayout(o) => release::object(device, o),
                Made::Pipeline(o) => release::object(device, o),
                Made::RenderPass(o) => release::object(device, o),
                Made::Framebuffer(o) => release::object(device, o),
                Made::QueryPool(o) => release::object(device, o),
                Made::Fence(o) => release::object(device, o),
                Made::CommandPool(o) => release::object(device, o),
                Made::Sampler(o) => release::object(device, o),
            };
        }
    }
}

/// A host-visible, host-coherent buffer of 4-byte words, mapped; its words are read through the
/// kit ([`Kit::read`]), which holds the device for the read.
#[derive(Clone, Copy)]
pub(crate) struct HostBuffer {
    pub(crate) buffer: vk::Buffer,
    words: *mut u32,
    len: usize,
}

/// What a kit needs of its device beside the device itself, read as the device is made.
#[derive(Clone, Copy)]
pub(crate) struct DeviceFacts {
    pub(crate) physical: vk::PhysicalDevice,
    /// The queue the checks submit to: its family and index.
    pub(crate) queue: (u32, u32),
    memory: vk::PhysicalDeviceMemoryProperties,
    /// Nanoseconds a timestamp tick.
    period: f64,
    /// The queue's family writes timestamps.
    stamps: bool,
}

impl DeviceFacts {
    /// The facts of `physical`, from `instance`, for the checks' queue `queue`.
    ///
    /// # Safety
    ///
    /// `physical` was enumerated from `instance`.
    pub(crate) unsafe fn of(
        instance: &ash::Instance,
        physical: vk::PhysicalDevice,
        queue: (u32, u32),
    ) -> DeviceFacts {
        // SAFETY: the caller's, for each query.
        let (memory, families, limits) = unsafe {
            (
                instance.get_physical_device_memory_properties(physical),
                instance.get_physical_device_queue_family_properties(physical),
                instance.get_physical_device_properties(physical).limits,
            )
        };
        let stamps = (families.get(queue.0 as usize)).is_some_and(|f| f.timestamp_valid_bits > 0);
        DeviceFacts {
            physical,
            queue,
            memory,
            period: f64::from(limits.timestamp_period),
            stamps,
        }
    }
}

/// How a kit reaches its device (the module docs).
enum Reach<'a> {
    /// As the device is made, on the creating thread.
    Making {
        instance: &'a ash::Instance,
        device: &'a ash::Device,
        loader_word: &'a LoaderWord,
        enabled: &'a EnableSet,
        queue: vk::Queue,
        queue_word: LoaderWord,
        /// Whether the device takes its push command, as the answers its checks gave so far
        /// decide it ([`Kit::answered`], before each check).
        takes_push: bool,
    },
    /// A running device, held weakly.
    Running(WeakDevice),
}

/// Why a running device's kit reaches it no more.
#[derive(Clone, Copy, Debug)]
enum Ended {
    /// The device was dropped.
    Dropped,
    /// The process is exiting.
    Exiting,
}

impl From<Ended> for CheckError {
    fn from(ended: Ended) -> CheckError {
        CheckError(
            match ended {
                Ended::Dropped => "the device was dropped",
                Ended::Exiting => "the process is exiting",
            }
            .to_owned(),
        )
    }
}

/// What one call of a kit reaches.
struct Reached<'r> {
    instance: &'r ash::Instance,
    device: &'r ash::Device,
    /// The running device's shared state; `None` as the device is made.
    shared: Option<&'r DeviceShared>,
    /// As the device is made: its first queue and that queue's loader word.
    queue: Option<(vk::Queue, &'r LoaderWord)>,
    /// As the device is made: its loader word, put back before a loader entry point.
    loader_word: Option<&'r LoaderWord>,
}

impl Reached<'_> {
    /// Keep `made`: in `own` as the device is made, on the device while it runs.
    fn keep(&self, own: &mut Vec<Made>, made: Made) {
        match self.shared {
            Some(shared) => shared.checks.keep(made),
            None => own.push(made),
        }
    }
}

impl Reach<'_> {
    /// `f` of what one call reaches (the module docs), each of `calls` marked as called on a
    /// running device.
    fn run<R>(&self, calls: &[Call], f: impl FnOnce(&Reached<'_>) -> R) -> Result<R, Ended> {
        match self {
            Reach::Making {
                instance,
                device,
                loader_word,
                queue,
                queue_word,
                ..
            } => Ok(f(&Reached {
                instance,
                device,
                shared: None,
                queue: Some((*queue, queue_word)),
                loader_word: Some(*loader_word),
            })),
            Reach::Running(weak) => {
                let ran = weak.with(|device| {
                    let busy = exit_guard::enter().ok_or(Ended::Exiting)?;
                    let shared = device.shared();
                    let reached = f(&Reached {
                        instance: &shared.instance.raw,
                        device: shared.vk(),
                        shared: Some(shared),
                        queue: None,
                        loader_word: None,
                    });
                    for &call in calls {
                        exit_guard::called(call);
                    }
                    // The section ends before the call's hold of the device goes.
                    drop(busy);
                    Ok(reached)
                });
                ran.unwrap_or(Err(Ended::Dropped))
            }
        }
    }
}

/// The checks' objects on one device.
pub(crate) struct Kit<'a> {
    reach: Reach<'a>,
    facts: DeviceFacts,
    pool: vk::CommandPool,
    /// The one command buffer, allocated at the first submission.
    command: vk::CommandBuffer,
    fence: vk::Fence,
    timestamps: Option<vk::QueryPool>,
    /// What the kit made as the device is made, released last first; a running device keeps
    /// them itself.
    made: Vec<Made>,
    /// How many of the objects made are the kit's own (its pool, fence and timestamps), which
    /// outlive each check.
    own: usize,
    pipelines: u32,
    gpu: Option<Duration>,
    longest: Option<Duration>,
    timed_out: bool,
    /// The running device was dropped or the process is exiting: no further call reaches it.
    ended: bool,
}

impl<'a> Kit<'a> {
    /// The kit of the device being made from `instance`: `device`, whose loader word is
    /// `loader_word`, enabling `enabled`, and its queue `facts.queue`, which nothing else uses
    /// meanwhile.
    ///
    /// # Safety
    ///
    /// `device` is live, made on `facts.physical` from `instance` with `enabled` and a queue at
    /// `facts.queue`, and `loader_word` is its.
    pub(super) unsafe fn making(
        instance: &'a ash::Instance,
        device: &'a ash::Device,
        loader_word: &'a LoaderWord,
        enabled: &'a EnableSet,
        facts: DeviceFacts,
    ) -> Result<Kit<'a>, String> {
        let (family, index) = facts.queue;
        // SAFETY: the caller's: the device is live and its word the loader's after the restore.
        let queue = unsafe {
            loader_word.restore();
            device.get_device_queue(family, index)
        };
        // SAFETY: the queue was just given: its word is the loader's.
        let queue_word = unsafe { LoaderWord::of_handle(queue.as_raw()) };
        let reach = Reach::Making {
            instance,
            device,
            loader_word,
            enabled,
            queue,
            queue_word,
            takes_push: false,
        };
        Kit::new(reach, facts)
    }

    /// As the device of `description` is made, before each check: the answers its checks gave so
    /// far, `checks`, which decide what the device is assembled with (its push command,
    /// [`Kit::push_command`]) by the rule it is assembled by. A running device's kit reads what
    /// the device took, and nothing of `checks`.
    pub(super) fn answered(&mut self, description: &DeviceDescription, checks: &DeviceChecks) {
        if let Reach::Making { takes_push, .. } = &mut self.reach {
            let defects = crate::driver_defects::taken(description, checks);
            *takes_push = crate::driver_defects::takes_push_command(&defects);
        }
    }

    /// Whether a submission timed out: no later check runs.
    pub(super) fn timed_out(&self) -> bool {
        self.timed_out
    }

    /// Whether the running device was dropped, or the process is exiting: the check that saw it
    /// answers nothing, and no later check runs.
    pub(super) fn ended(&self) -> bool {
        self.ended
    }

    /// Start counting one check's cost.
    pub(super) fn begin_check(&mut self) -> Instant {
        self.pipelines = 0;
        self.gpu = None;
        self.longest = None;
        Instant::now()
    }

    /// The cost of the check begun at `began`; what it made is released (unless a submission
    /// timed out, or the device went).
    pub(super) fn end_check(&mut self, began: Instant) -> super::CheckCost {
        let cost = super::CheckCost {
            wall: began.elapsed(),
            gpu: self.gpu,
            longest: self.longest,
            pipelines: self.pipelines,
        };
        self.release_made();
        cost
    }

    /// Release what the checks made beside the kit's own objects, last first.
    fn release_made(&mut self) {
        if self.timed_out || self.ended {
            return;
        }
        let own = self.own;
        let made = &mut self.made;
        let released = self.reach.run(&[], |at| {
            let objects = match at.shared {
                Some(shared) => shared.checks.take_from(own),
                None => made.split_off(own.min(made.len())),
            };
            for object in objects.into_iter().rev() {
                // SAFETY: every submission completed, so no pending work uses an object; each
                // was made by the device and is released once.
                unsafe { object.release(at.device) };
            }
        });
        if released.is_err() {
            self.ended = true;
        }
    }

    /// The check's error where the device's reach `reached` ended (the module docs).
    fn reached<R>(&mut self, reached: Result<R, Ended>) -> Result<R, CheckError> {
        reached.map_err(|ended| {
            self.ended = true;
            ended.into()
        })
    }

    /// `f` of the device's functions, each of `calls` marked as called on a running device: a call
    /// that makes nothing (a memory requirement, a bind, a descriptor write).
    pub(crate) fn call<R>(
        &mut self,
        calls: &[Call],
        f: impl FnOnce(&ash::Device) -> R,
    ) -> Result<R, CheckError> {
        let reached = self.reach.run(calls, |at| f(at.device));
        self.reached(reached)
    }

    /// One object `make` makes with the device's functions through `call`, kept by the kit as
    /// `kept`, and by a running device too, whose teardown releases it should the device go first.
    pub(crate) fn make<T: Copy>(
        &mut self,
        call: Call,
        make: impl FnOnce(&ash::Device) -> Result<T, CheckError>,
        kept: fn(T) -> Made,
    ) -> Result<T, CheckError> {
        let own = &mut self.made;
        let reached = self.reach.run(&[call], |at| {
            let object = make(at.device)?;
            at.keep(own, kept(object));
            Ok(object)
        });
        self.reached(reached)?
    }

    /// `f` of the instance and the physical device: a query of the physical device's properties
    /// (on a running device inside a call, which holds the device).
    pub(crate) fn physical_query<R>(
        &mut self,
        f: impl FnOnce(&ash::Instance, vk::PhysicalDevice) -> R,
    ) -> Result<R, CheckError> {
        let physical = self.facts.physical;
        let reached = self.reach.run(&[], |at| f(at.instance, physical));
        self.reached(reached)
    }

    /// The command that sets a depth bias in a representation, where the device enabled
    /// `VK_EXT_depth_bias_control` (`render_state.rs`): as the device is made, its own entry point;
    /// on a running device, its routed one. Called only inside [`Kit::submit`]'s recording.
    pub(crate) fn depth_bias_command(&mut self) -> Result<Option<DepthBiasCommand>, CheckError> {
        let reached = match &self.reach {
            Reach::Making {
                instance,
                device,
                loader_word,
                enabled,
                ..
            } => {
                // SAFETY: the device was made from the instance with `enabled`; the loader
                // serves `vkGetDeviceProcAddr` once the word is put back.
                Ok(unsafe {
                    loader_word.restore();
                    crate::render_state::depth_bias_command(instance, device, enabled)
                })
            }
            Reach::Running(_) => self.reach.run(&[], |at| {
                at.shared.and_then(|shared| shared.depth_bias_command)
            }),
        };
        self.reached(reached)
    }

    /// The command that binds an index buffer range of a size, where the device enabled
    /// `maintenance5` (`draw_source.rs`): as the device is made, its own entry point; on a
    /// running device, its routed one. Called only inside [`Kit::submit`]'s recording.
    pub(crate) fn sized_index_bind(&mut self) -> Result<Option<SizedIndexBind>, CheckError> {
        let reached = match &self.reach {
            Reach::Making {
                instance,
                device,
                loader_word,
                enabled,
                ..
            } => {
                // SAFETY: the device was made from the instance with `enabled`; the loader
                // serves `vkGetDeviceProcAddr` once the word is put back.
                Ok(unsafe {
                    loader_word.restore();
                    crate::draw_source::sized_index_bind(instance, device, enabled)
                })
            }
            Reach::Running(_) => self.reach.run(&[], |at| {
                at.shared.and_then(|shared| shared.sized_index_bind)
            }),
        };
        self.reached(reached)
    }

    /// A memory type of `bits` with every flag of `flags`.
    fn memory_type(&self, bits: u32, flags: vk::MemoryPropertyFlags) -> Option<u32> {
        let memory = &self.facts.memory;
        let types = &memory.memory_types[..memory.memory_type_count as usize];
        (0..types.len() as u32)
            .find(|&i| bits & (1 << i) != 0 && types[i as usize].property_flags.contains(flags))
    }

    /// Memory of `requirements` with `flags`, kept by the kit.
    fn allocate(
        &mut self,
        requirements: vk::MemoryRequirements,
        flags: vk::MemoryPropertyFlags,
    ) -> Result<vk::DeviceMemory, CheckError> {
        let index = self
            .memory_type(requirements.memory_type_bits, flags)
            .ok_or_else(|| CheckError(format!("no memory type of flags {:#x}", flags.as_raw())))?;
        let info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(index);
        self.make(
            Call::AllocateMemory,
            // SAFETY: the device is live.
            |device| Ok(unsafe { device.allocate_memory(&info, None) }?),
            Made::Memory,
        )
    }

    /// A host-visible buffer of `usage` holding `words`, mapped and kept by the kit.
    pub(crate) fn host_buffer(
        &mut self,
        words: &[u32],
        usage: vk::BufferUsageFlags,
    ) -> Result<HostBuffer, CheckError> {
        self.host_buffer_in(words, words.len(), usage)
    }

    /// [`Kit::host_buffer`] of `buffer_words` of `words`, its memory holding all of `words`: a
    /// buffer whose memory runs on past its end.
    pub(crate) fn host_buffer_in(
        &mut self,
        words: &[u32],
        buffer_words: usize,
        usage: vk::BufferUsageFlags,
    ) -> Result<HostBuffer, CheckError> {
        let size = (buffer_words.max(1) * 4) as vk::DeviceSize;
        let info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = self.make(
            Call::CreateBuffer,
            // SAFETY: the device is live.
            |device| Ok(unsafe { device.create_buffer(&info, None) }?),
            Made::Buffer,
        )?;
        let mut requirements = self.call(&[Call::GetBufferMemoryRequirements], |device| {
            // SAFETY: the buffer was just made.
            unsafe { device.get_buffer_memory_requirements(buffer) }
        })?;
        let whole = (words.len().max(1) * 4) as vk::DeviceSize;
        requirements.size = requirements.size.max(whole);
        let flags = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let memory = self.allocate(requirements, flags)?;
        let mapped = self.call(&[Call::BindBufferMemory, Call::MapMemory], |device| {
            // SAFETY: the memory is the buffer's type and large enough; nothing maps it yet.
            let mapped = unsafe {
                device.bind_buffer_memory(buffer, memory, 0)?;
                device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())?
            };
            let mapped = mapped.cast::<u32>();
            // SAFETY: the mapping holds at least `words.len()` words, host-coherent, while the
            // device is held for this call.
            unsafe { std::ptr::copy_nonoverlapping(words.as_ptr(), mapped, words.len()) };
            Ok::<_, vk::Result>(mapped)
        })??;
        Ok(HostBuffer {
            buffer,
            words: mapped,
            len: words.len(),
        })
    }

    /// `buffer`'s words, as the device left them.
    pub(crate) fn read(&mut self, buffer: &HostBuffer) -> Result<Vec<u32>, CheckError> {
        self.call(&[], |_| {
            // SAFETY: the mapping holds `len` words while the kit's check runs and the device
            // is held for this call, and no submission is pending (the kit waits for each).
            unsafe { std::slice::from_raw_parts(buffer.words, buffer.len) }.to_vec()
        })
    }

    /// A buffer of `words` 4-byte words and `usage` in device-local memory (where the device has
    /// none left, any), kept by the kit: one a check's shaders read and write throughout, which
    /// host-visible memory would carry across a discrete GPU's bus. Its words reach the host by a
    /// copy into a [`HostBuffer`].
    pub(crate) fn device_buffer(
        &mut self,
        words: usize,
        usage: vk::BufferUsageFlags,
    ) -> Result<vk::Buffer, CheckError> {
        let info = vk::BufferCreateInfo::default()
            .size((words.max(1) * 4) as vk::DeviceSize)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = self.make(
            Call::CreateBuffer,
            // SAFETY: the device is live.
            |device| Ok(unsafe { device.create_buffer(&info, None) }?),
            Made::Buffer,
        )?;
        let requirements = self.call(&[Call::GetBufferMemoryRequirements], |device| {
            // SAFETY: the buffer was just made.
            unsafe { device.get_buffer_memory_requirements(buffer) }
        })?;
        let memory = match self.allocate(requirements, vk::MemoryPropertyFlags::DEVICE_LOCAL) {
            Ok(memory) => memory,
            Err(_) if !self.ended => {
                self.allocate(requirements, vk::MemoryPropertyFlags::empty())?
            }
            Err(error) => return Err(error),
        };
        self.call(&[Call::BindBufferMemory], |device| {
            // SAFETY: the memory is of the buffer's type and size.
            unsafe { device.bind_buffer_memory(buffer, memory, 0) }
        })??;
        Ok(buffer)
    }

    /// An image of `info` in device-local memory, kept by the kit.
    pub(crate) fn image(
        &mut self,
        info: &vk::ImageCreateInfo<'_>,
    ) -> Result<vk::Image, CheckError> {
        let image = self.make(
            Call::CreateImage,
            // SAFETY: the device is live, and `info` valid for it (the caller's).
            |device| Ok(unsafe { crate::creation::image(device, info) }?),
            Made::Image,
        )?;
        let requirements = self.call(&[Call::GetImageMemoryRequirements], |device| {
            // SAFETY: the image was just made.
            unsafe { device.get_image_memory_requirements(image) }
        })?;
        let memory = match self.allocate(requirements, vk::MemoryPropertyFlags::DEVICE_LOCAL) {
            Ok(memory) => memory,
            Err(_) if !self.ended => {
                self.allocate(requirements, vk::MemoryPropertyFlags::empty())?
            }
            Err(error) => return Err(error),
        };
        self.call(&[Call::BindImageMemory], |device| {
            // SAFETY: the memory is of the image's type and size.
            unsafe { device.bind_image_memory(image, memory, 0) }
        })??;
        Ok(image)
    }

    /// A view of `info`, kept by the kit.
    pub(crate) fn view(
        &mut self,
        info: &vk::ImageViewCreateInfo<'_>,
    ) -> Result<vk::ImageView, CheckError> {
        self.make(
            Call::CreateImageView,
            // SAFETY: the device is live.
            |device| Ok(unsafe { device.create_image_view(info, None) }?),
            Made::View,
        )
    }

    /// A sampler of `info`, kept by the kit.
    pub(crate) fn sampler(
        &mut self,
        info: &vk::SamplerCreateInfo<'_>,
    ) -> Result<vk::Sampler, CheckError> {
        self.make(
            Call::CreateSampler,
            // SAFETY: the device is live, and `info` and its chain live for the call.
            |device| Ok(unsafe { device.create_sampler(info, None) }?),
            Made::Sampler,
        )
    }

    /// A query pool of `info`, kept by the kit.
    pub(crate) fn query_pool(
        &mut self,
        info: &vk::QueryPoolCreateInfo<'_>,
    ) -> Result<vk::QueryPool, CheckError> {
        self.make(
            Call::CreateQueryPool,
            // SAFETY: the device is live.
            |device| Ok(unsafe { device.create_query_pool(info, None) }?),
            Made::QueryPool,
        )
    }

    /// The 64-bit results of the first `count` queries of `pool`, each written by a submission
    /// that completed.
    pub(crate) fn query_results(
        &mut self,
        pool: vk::QueryPool,
        count: usize,
    ) -> Result<Vec<u64>, CheckError> {
        let mut results = vec![0u64; count];
        let flags = vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT;
        self.call(&[Call::GetQueryPoolResults], |device| {
            // SAFETY: the pool holds `count` queries, each ended by a completed submission.
            unsafe { device.get_query_pool_results(pool, 0, &mut results, flags) }
        })??;
        Ok(results)
    }

    /// The command the draws and dispatches of a device at `api` (its description's) push set 0
    /// with, none where it takes no push command (`driver_defects::takes_push_command`): as the
    /// device is made, the driver's own ([`Kit::driver_push_command`]) where the answers its
    /// checks gave so far let the device take it ([`Kit::answered`]); on a running device, the
    /// command it took (`device/mod.rs`). Called only inside [`Kit::submit`]'s recording.
    pub(crate) fn push_command(
        &mut self,
        api: ApiVersion,
    ) -> Result<Option<vk::PFN_vkCmdPushDescriptorSetKHR>, CheckError> {
        match self.reach {
            Reach::Making {
                takes_push: false, ..
            } => Ok(None),
            Reach::Making { .. } => self.driver_push_command(api),
            Reach::Running(_) => {
                let reached = (self.reach).run(&[], |at| {
                    at.shared.and_then(|shared| shared.push_descriptor)
                });
                self.reached(reached)
            }
        }
    }

    /// The driver's own `vkCmdPushDescriptorSet` of a device at `api` (its description's), from
    /// Vulkan 1.4 or `VK_KHR_push_descriptor` (`binding::push_descriptor_command`), whether or not
    /// the device takes it: for the check that decides whether it does (`pushed_sizes.rs`), as
    /// the device is made. A running device took its push command as it was made: its kit answers
    /// an error. Called only inside [`Kit::submit`]'s recording.
    pub(crate) fn driver_push_command(
        &self,
        api: ApiVersion,
    ) -> Result<Option<vk::PFN_vkCmdPushDescriptorSetKHR>, CheckError> {
        match &self.reach {
            Reach::Making {
                instance,
                device,
                loader_word,
                enabled,
                ..
            } => {
                // SAFETY: the device was made from the instance with `enabled`; the loader
                // serves `vkGetDeviceProcAddr` once the word is put back.
                Ok(unsafe {
                    loader_word.restore();
                    crate::binding::push_descriptor_command(instance, device, api, enabled)
                })
            }
            Reach::Running(_) => Err(CheckError(
                "a running device took its push command as it was made".to_owned(),
            )),
        }
    }

    /// A shader module of `words`, kept by the kit.
    pub(crate) fn shader(&mut self, words: &[u32]) -> Result<vk::ShaderModule, CheckError> {
        let info = vk::ShaderModuleCreateInfo::default().code(words);
        self.make(
            Call::CreateShaderModule,
            // SAFETY: the device is live; the words are a module the translator validated.
            |device| Ok(unsafe { device.create_shader_module(&info, None) }?),
            Made::Shader,
        )
    }

    /// A set of storage buffers bound `0..` in `buffers`' order, read from `stages`, and the
    /// pipeline layout of that set alone, kept by the kit.
    pub(crate) fn storage_set(
        &mut self,
        buffers: &[HostBuffer],
        stages: vk::ShaderStageFlags,
    ) -> Result<(vk::DescriptorSet, vk::PipelineLayout), CheckError> {
        let bindings: Vec<SetBinding> = (0u32..)
            .zip(buffers)
            .map(|(binding, &buffer)| SetBinding {
                binding,
                stages,
                resource: SetResource::Buffer(buffer),
            })
            .collect();
        let (set_layout, set) = self.descriptor_set(&bindings)?;
        let layout = self.layout(&[set_layout], None)?;
        Ok((set, layout))
    }

    /// A descriptor set of `bindings`, one descriptor each, and its layout, kept by the kit.
    pub(crate) fn descriptor_set(
        &mut self,
        bindings: &[SetBinding],
    ) -> Result<(vk::DescriptorSetLayout, vk::DescriptorSet), CheckError> {
        let set_layout = self.set_layout(bindings, false)?;
        let set = self.set_of(set_layout, bindings)?;
        Ok((set_layout, set))
    }

    /// The set layout of `bindings`, one descriptor each, an immutable sampler baked where one
    /// holds it ([`SetResource::Immutable`]), and a push descriptor set layout where `pushed`
    /// (`VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT`), kept by the kit.
    pub(crate) fn set_layout(
        &mut self,
        bindings: &[SetBinding],
        pushed: bool,
    ) -> Result<vk::DescriptorSetLayout, CheckError> {
        let samplers: Vec<[vk::Sampler; 1]> = (bindings.iter())
            .map(|b| match b.resource {
                SetResource::Immutable(_, sampler) => [sampler],
                _ => [vk::Sampler::null()],
            })
            .collect();
        let layout_bindings: Vec<vk::DescriptorSetLayoutBinding<'_>> = bindings
            .iter()
            .zip(&samplers)
            .map(|(b, sampler)| {
                let binding = vk::DescriptorSetLayoutBinding::default()
                    .binding(b.binding)
                    .descriptor_type(b.resource.descriptor_type())
                    .descriptor_count(1)
                    .stage_flags(b.stages);
                match b.resource {
                    SetResource::Immutable(..) => binding.immutable_samplers(sampler),
                    _ => binding,
                }
            })
            .collect();
        let flags = match pushed {
            true => vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR,
            false => vk::DescriptorSetLayoutCreateFlags::empty(),
        };
        let info = vk::DescriptorSetLayoutCreateInfo::default()
            .flags(flags)
            .bindings(&layout_bindings);
        self.make(
            Call::CreateDescriptorSetLayout,
            // SAFETY: the device is live; every immutable sampler is live.
            |device| Ok(unsafe { device.create_descriptor_set_layout(&info, None) }?),
            Made::SetLayout,
        )
    }

    /// A descriptor set of `set_layout`, which [`Kit::set_layout`] made of `bindings` (not
    /// pushed), written with them, kept by the kit.
    pub(crate) fn set_of(
        &mut self,
        set_layout: vk::DescriptorSetLayout,
        bindings: &[SetBinding],
    ) -> Result<vk::DescriptorSet, CheckError> {
        let mut sizes: Vec<vk::DescriptorPoolSize> = Vec::new();
        for b in bindings {
            let ty = b.resource.descriptor_type();
            match sizes.iter_mut().find(|s| s.ty == ty) {
                Some(size) => size.descriptor_count += 1,
                None => sizes.push(vk::DescriptorPoolSize {
                    ty,
                    descriptor_count: 1,
                }),
            }
        }
        let info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&sizes);
        let pool = self.make(
            Call::CreateDescriptorPool,
            // SAFETY: the device is live.
            |device| Ok(unsafe { device.create_descriptor_pool(&info, None) }?),
            Made::DescriptorPool,
        )?;
        let layouts = [set_layout];
        let info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&layouts);
        let writes = Writes::of(bindings);
        let calls = [Call::AllocateDescriptorSets, Call::UpdateDescriptorSets];
        let set = self.call(&calls, |device| {
            // SAFETY: the pool holds one set of the layout.
            let set = unsafe { device.allocate_descriptor_sets(&info) }?[0];
            // SAFETY: the set and every resource are live, and no submission uses the set.
            unsafe { device.update_descriptor_sets(&writes.writes(set), &[]) };
            Ok::<_, vk::Result>(set)
        })??;
        Ok(set)
    }

    /// A pipeline layout of `layouts` and, where `push` names one, a push-constant range of
    /// `push.1` bytes from offset 0 for the stages `push.0`, kept by the kit.
    pub(crate) fn layout(
        &mut self,
        layouts: &[vk::DescriptorSetLayout],
        push: Option<(vk::ShaderStageFlags, u32)>,
    ) -> Result<vk::PipelineLayout, CheckError> {
        let ranges: Vec<vk::PushConstantRange> = push
            .map(|(stages, size)| vk::PushConstantRange {
                stage_flags: stages,
                offset: 0,
                size,
            })
            .into_iter()
            .collect();
        let info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(layouts)
            .push_constant_ranges(&ranges);
        self.make(
            Call::CreatePipelineLayout,
            // SAFETY: the device is live.
            |device| Ok(unsafe { device.create_pipeline_layout(&info, None) }?),
            Made::PipelineLayout,
        )
    }

    /// The compute pipeline of `module`'s entry `main` under `layout`, kept by the kit; the
    /// driver's refusal as its `VkResult`, a value a check may read as its answer.
    pub(crate) fn compute_pipeline(
        &mut self,
        module: vk::ShaderModule,
        layout: vk::PipelineLayout,
    ) -> Result<vk::Pipeline, vk::Result> {
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(c"main");
        let info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(layout);
        self.pipeline(Call::CreateComputePipelines, |device| {
            // SAFETY: the device, module and layout are live.
            unsafe { device.create_compute_pipelines(vk::PipelineCache::null(), &[info], None) }
        })
    }

    /// The graphics pipeline of `info`, kept by the kit; the driver's refusal as its `VkResult`.
    pub(crate) fn graphics_pipeline(
        &mut self,
        info: &vk::GraphicsPipelineCreateInfo<'_>,
    ) -> Result<vk::Pipeline, vk::Result> {
        self.pipeline(Call::CreateGraphicsPipelines, |device| {
            // SAFETY: the device and every object `info` names are live.
            unsafe {
                device.create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    std::slice::from_ref(info),
                    None,
                )
            }
        })
    }

    /// The one pipeline `create` makes through `call`, a driver's making
    /// ([`exit_guard::driver_making`]), kept; a refusal's `VkResult`, any pipeline made kept too.
    /// On a running device that went, `ERROR_UNKNOWN`: the kit has ended, so the check's answer is
    /// not read ([`Kit::ended`]).
    fn pipeline(
        &mut self,
        call: Call,
        create: impl FnOnce(&ash::Device) -> Result<Vec<vk::Pipeline>, (Vec<vk::Pipeline>, vk::Result)>,
    ) -> Result<vk::Pipeline, vk::Result> {
        self.pipelines += 1;
        let own = &mut self.made;
        let reached = self.reach.run(&[call], |at| {
            let made = exit_guard::driver_making(|| create(at.device));
            let (pipelines, result) = match made {
                Ok(pipelines) => (pipelines, None),
                Err((pipelines, result)) => (pipelines, Some(result)),
            };
            for &pipeline in pipelines.iter().filter(|p| **p != vk::Pipeline::null()) {
                at.keep(own, Made::Pipeline(pipeline));
            }
            match (result, pipelines.first()) {
                (None, Some(&pipeline)) => Ok(pipeline),
                (Some(result), _) => Err(result),
                (None, None) => Err(vk::Result::ERROR_UNKNOWN),
            }
        });
        match self.reached(reached) {
            Ok(made) => made,
            Err(_) => Err(vk::Result::ERROR_UNKNOWN),
        }
    }

    /// The render pass of `info` (`vkCreateRenderPass2`, core in Vulkan 1.2), kept by the kit.
    pub(crate) fn render_pass(
        &mut self,
        info: &vk::RenderPassCreateInfo2<'_>,
    ) -> Result<vk::RenderPass, CheckError> {
        self.make(
            Call::CreateRenderPass2,
            // SAFETY: the device is live, at Vulkan 1.2 or later.
            |device| Ok(unsafe { crate::creation::render_pass2(device, info) }?),
            Made::RenderPass,
        )
    }

    /// The framebuffer of `info`, kept by the kit.
    pub(crate) fn framebuffer(
        &mut self,
        info: &vk::FramebufferCreateInfo<'_>,
    ) -> Result<vk::Framebuffer, CheckError> {
        self.make(
            Call::CreateFramebuffer,
            // SAFETY: the device and every object `info` names are live.
            |device| Ok(unsafe { crate::creation::framebuffer(device, info) }?),
            Made::Framebuffer,
        )
    }

    /// Run `kernel` (`metal2vulkan::bind::CheckKernel`): its buffers made and filled, one
    /// dispatch of its workgroups, and its read-back binding's words after it.
    pub(crate) fn kernel(&mut self, kernel: &CheckKernel) -> Result<Vec<u32>, CheckError> {
        let buffers = kernel
            .buffers
            .iter()
            .map(|words| self.host_buffer(words, vk::BufferUsageFlags::STORAGE_BUFFER))
            .collect::<Result<Vec<HostBuffer>, CheckError>>()?;
        let read = *buffers
            .get(kernel.read as usize)
            .ok_or_else(|| CheckError(format!("the kernel reads back binding {}", kernel.read)))?;
        let module = self.shader(&kernel.spirv)?;
        let (set, layout) = self.storage_set(&buffers, vk::ShaderStageFlags::COMPUTE)?;
        let pipeline = self.compute_pipeline(module, layout)?;
        let workgroups = kernel.workgroups;
        self.submit(|device, command| {
            let written = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ);
            // SAFETY: recording, outside a render pass; every object named is live.
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
                device.cmd_dispatch(command, workgroups, 1, 1);
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
        })?;
        self.read(&read)
    }

    /// The kit's one command buffer, allocated at its first submission.
    fn command_buffer(&mut self) -> Result<vk::CommandBuffer, CheckError> {
        if self.command != vk::CommandBuffer::null() {
            return Ok(self.command);
        }
        let info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let reached = self.reach.run(&[], |at| match (at.shared, at.loader_word) {
            (Some(shared), _) => shared
                .allocate_command_buffers(&info)
                .map_err(CheckError::from),
            (None, Some(word)) => {
                // SAFETY: the loader's entry point reads the device's word, restored first; the
                // pool is the kit's.
                let allocated = unsafe {
                    word.restore();
                    at.device.allocate_command_buffers(&info)
                };
                allocated.map_err(CheckError::from)
            }
            (None, None) => Err(CheckError("a kit reaches no device".to_owned())),
        });
        let allocated = self.reached(reached)??;
        self.command = allocated.first().copied().unwrap_or_default();
        Ok(self.command)
    }

    /// Record the kit's command buffer with `record`, submit it, and wait for it at most
    /// [`TIMEOUT`]; its GPU time is added to the check's where the queue writes timestamps. The
    /// command buffer's pool resets its command buffers, so beginning it again resets it. A
    /// timed-out submission leaves every object to the device (the kit then releases none). On a
    /// running device, `record` records with the exit guard's table alone (the module docs), and
    /// the submission's section marks [`RECORDED`].
    pub(crate) fn submit(
        &mut self,
        record: impl FnOnce(&ash::Device, vk::CommandBuffer),
    ) -> Result<(), CheckError> {
        let command = self.command_buffer()?;
        let (fence, stamps) = (self.fence, self.timestamps);
        let relayed = Arc::new(Relayed::default());
        let by = By::Check(Arc::clone(&relayed));
        let reached = self.reach.run(RECORDED, |at| -> Result<(), vk::Result> {
            let device = at.device;
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            // SAFETY: the command buffer is not pending (the kit waits for each submission), and
            // its pool resets command buffers.
            unsafe { device.begin_command_buffer(command, &begin) }?;
            if let Some(pool) = stamps {
                // SAFETY: recording; the pool holds two queries.
                unsafe {
                    device.cmd_reset_query_pool(command, pool, 0, 2);
                    device.cmd_write_timestamp(
                        command,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        pool,
                        0,
                    );
                }
            }
            record(device, command);
            if let Some(pool) = stamps {
                // SAFETY: recording.
                unsafe {
                    device.cmd_write_timestamp(
                        command,
                        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                        pool,
                        1,
                    );
                }
            }
            // SAFETY: recording.
            unsafe { device.end_command_buffer(command) }?;
            let commands = [command];
            let submit = vk::SubmitInfo::default().command_buffers(&commands);
            match (at.shared, at.queue) {
                // SAFETY: the queue is used under its lock; the command buffer is executable and
                // the fence unsignaled.
                (Some(shared), _) => shared.use_queue(0, by, |queue| unsafe {
                    device.queue_submit(queue, &[submit], fence)
                }),
                // SAFETY: the queue is the kit's alone as the device is made, its word put back
                // first; the fence is unsignaled.
                (None, Some((queue, word))) => unsafe {
                    word.restore();
                    device.queue_submit(queue, &[submit], fence)
                },
                (None, None) => Err(vk::Result::ERROR_UNKNOWN),
            }
        });
        self.reached(reached)??;
        self.wait()?;
        // The driver reports a failed submission before it signals the fence (`relay.rs`).
        if let Some((code, description)) = relayed.failure() {
            let description = description.map_or(String::new(), |text| format!(": {text}"));
            return Err(CheckError(format!(
                "the driver failed the check's submission (MTLCommandBufferError {code}{description})"
            )));
        }
        let Some(pool) = stamps else {
            return Ok(());
        };
        let mut ticks = [0u64; 2];
        let read = self.call(&[Call::GetQueryPoolResults], |device| {
            // SAFETY: both queries were written by the completed submission.
            unsafe {
                device.get_query_pool_results(pool, 0, &mut ticks, vk::QueryResultFlags::TYPE_64)
            }
        })?;
        if read.is_ok() {
            let nanos = ticks[1].saturating_sub(ticks[0]) as f64 * self.facts.period;
            let spent = Duration::from_nanos(nanos as u64);
            self.gpu = Some(self.gpu.unwrap_or_default() + spent);
            self.longest = Some(self.longest.map_or(spent, |longest| longest.max(spent)));
        }
        Ok(())
    }

    /// Wait for the kit's submission at most [`TIMEOUT`] (on a running device, in slices of
    /// [`SLICE`]), then reset its fence.
    fn wait(&mut self) -> Result<(), CheckError> {
        let fence = self.fence;
        let deadline = Instant::now() + TIMEOUT;
        let sliced = matches!(self.reach, Reach::Running(_));
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            let slice = match sliced {
                true => left.min(SLICE),
                false => left,
            };
            let reached = self.reach.run(&[Call::WaitForFences], |at| {
                // SAFETY: the fence was submitted.
                unsafe {
                    at.device
                        .wait_for_fences(&[fence], true, slice.as_nanos() as u64)
                }
            });
            let waited = match self.reached(reached) {
                Ok(waited) => waited,
                Err(error) => {
                    // The submission may still run: what it uses is the device's teardown's.
                    self.timed_out = true;
                    return Err(error);
                }
            };
            match waited {
                Ok(()) => break,
                Err(vk::Result::TIMEOUT) if Instant::now() < deadline => continue,
                Err(vk::Result::TIMEOUT) => {
                    self.timed_out = true;
                    return Err(CheckError(format!(
                        "the submission did not complete within {} s",
                        TIMEOUT.as_secs()
                    )));
                }
                Err(error) => {
                    self.timed_out = true;
                    return Err(error.into());
                }
            }
        }
        self.call(&[Call::ResetFences], |device| {
            // SAFETY: the submission completed: the fence is free.
            unsafe { device.reset_fences(&[fence]) }
        })??;
        Ok(())
    }
}

impl Kit<'static> {
    /// The kit of the running device `device`, from its checks thread (the module docs).
    pub(super) fn running(device: WeakDevice, facts: DeviceFacts) -> Result<Kit<'static>, String> {
        Kit::new(Reach::Running(device), facts)
    }
}

impl<'a> Kit<'a> {
    /// A kit of `reach`: a command pool of the checks' queue's family whose command buffers reset,
    /// a fence, and a timestamp pool where the family writes timestamps.
    fn new(reach: Reach<'a>, facts: DeviceFacts) -> Result<Kit<'a>, String> {
        let mut kit = Kit {
            reach,
            facts,
            pool: vk::CommandPool::null(),
            command: vk::CommandBuffer::null(),
            fence: vk::Fence::null(),
            timestamps: None,
            made: Vec::new(),
            own: 0,
            pipelines: 0,
            gpu: None,
            longest: None,
            timed_out: false,
            ended: false,
        };
        let info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(facts.queue.0);
        kit.pool = kit
            .make(
                Call::CreateCommandPool,
                // SAFETY: the device is live.
                |device| Ok(unsafe { device.create_command_pool(&info, None) }?),
                Made::CommandPool,
            )
            .map_err(|e| format!("vkCreateCommandPool: {}", e.0))?;
        let info = vk::FenceCreateInfo::default();
        kit.fence = kit
            .make(
                Call::CreateFence,
                // SAFETY: the device is live.
                |device| Ok(unsafe { device.create_fence(&info, None) }?),
                Made::Fence,
            )
            .map_err(|e| format!("vkCreateFence: {}", e.0))?;
        if facts.stamps {
            let info = vk::QueryPoolCreateInfo::default()
                .query_type(vk::QueryType::TIMESTAMP)
                .query_count(2);
            kit.timestamps = kit
                .make(
                    Call::CreateQueryPool,
                    // SAFETY: the device is live.
                    |device| Ok(unsafe { device.create_query_pool(&info, None) }?),
                    Made::QueryPool,
                )
                .ok();
            if kit.ended {
                return Err("the device was dropped".to_owned());
            }
        }
        // The pool, the fence and the timestamps, which outlive each check.
        kit.own = 2 + usize::from(kit.timestamps.is_some());
        Ok(kit)
    }
}

impl Drop for Kit<'_> {
    fn drop(&mut self) {
        if let Reach::Making {
            loader_word,
            queue_word,
            ..
        } = &self.reach
        {
            // SAFETY: the words are the device's and the queue's as the loader wrote them.
            unsafe {
                queue_word.restore();
                loader_word.restore();
            }
        }
        if self.timed_out || self.ended {
            // A submission may still run, or the device went: what the kit made stays the
            // device's (a running device's teardown releases it after its idle wait).
            return;
        }
        self.own = 0;
        self.release_made();
    }
}

/// What one binding of a [`Kit::descriptor_set`] holds.
#[derive(Clone, Copy)]
pub(crate) enum SetResource {
    /// A storage buffer: the whole buffer.
    Buffer(HostBuffer),
    /// A storage buffer in device-local memory ([`Kit::device_buffer`]): the whole buffer.
    DeviceBuffer(vk::Buffer),
    /// A storage image: the view, in the `GENERAL` layout wherever the set is used.
    StorageImage(vk::ImageView),
    /// A sampled image: the view, in the layout given wherever the set is used.
    SampledImage(vk::ImageView, vk::ImageLayout),
    /// A combined image sampler: a sampled image's view, in the `SHADER_READ_ONLY_OPTIMAL` layout
    /// wherever the set is used, and its sampler.
    Combined(vk::ImageView, vk::Sampler),
    /// A uniform buffer: the whole buffer.
    UniformBuffer(HostBuffer),
    /// A combined image sampler whose sampler the set layout bakes (an immutable sampler): the
    /// view, in `SHADER_READ_ONLY_OPTIMAL` wherever the set is used, and the sampler.
    Immutable(vk::ImageView, vk::Sampler),
}

impl SetResource {
    fn descriptor_type(self) -> vk::DescriptorType {
        match self {
            SetResource::Buffer(_) | SetResource::DeviceBuffer(_) => {
                vk::DescriptorType::STORAGE_BUFFER
            }
            SetResource::StorageImage(_) => vk::DescriptorType::STORAGE_IMAGE,
            SetResource::SampledImage(..) => vk::DescriptorType::SAMPLED_IMAGE,
            SetResource::Combined(..) => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            SetResource::UniformBuffer(_) => vk::DescriptorType::UNIFORM_BUFFER,
            SetResource::Immutable(..) => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        }
    }
}

/// The descriptor of one binding of a [`Kit::descriptor_set`], as its write names it.
enum WriteInfo {
    Buffer([vk::DescriptorBufferInfo; 1]),
    Image([vk::DescriptorImageInfo; 1]),
}

/// The writes of a set of bindings: into a set ([`Kit::set_of`]), or pushed
/// (`vkCmdPushDescriptorSet`, whose writes name no set).
pub(crate) struct Writes {
    bindings: Vec<SetBinding>,
    infos: Vec<WriteInfo>,
}

impl Writes {
    /// The writes of `bindings`, each binding's whole resource.
    pub(crate) fn of(bindings: &[SetBinding]) -> Writes {
        let image = |view, layout| {
            WriteInfo::Image([vk::DescriptorImageInfo {
                sampler: vk::Sampler::null(),
                image_view: view,
                image_layout: layout,
            }])
        };
        let infos = (bindings.iter())
            .map(|b| match b.resource {
                SetResource::Buffer(buffer) | SetResource::UniformBuffer(buffer) => {
                    WriteInfo::Buffer([vk::DescriptorBufferInfo {
                        buffer: buffer.buffer,
                        offset: 0,
                        range: vk::WHOLE_SIZE,
                    }])
                }
                SetResource::DeviceBuffer(buffer) => {
                    WriteInfo::Buffer([vk::DescriptorBufferInfo {
                        buffer,
                        offset: 0,
                        range: vk::WHOLE_SIZE,
                    }])
                }
                SetResource::StorageImage(view) => image(view, vk::ImageLayout::GENERAL),
                SetResource::SampledImage(view, layout) => image(view, layout),
                SetResource::Combined(view, sampler) => {
                    WriteInfo::Image([vk::DescriptorImageInfo {
                        sampler,
                        image_view: view,
                        image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    }])
                }
                // The layout's sampler is the binding's: the write's is ignored.
                SetResource::Immutable(view, _) => {
                    image(view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                }
            })
            .collect();
        Writes {
            bindings: bindings.to_vec(),
            infos,
        }
    }

    /// The writes into `set`, one per binding; a pushed write's set is null.
    pub(crate) fn writes(&self, set: vk::DescriptorSet) -> Vec<vk::WriteDescriptorSet<'_>> {
        (self.bindings.iter())
            .zip(&self.infos)
            .map(|(b, info)| {
                let write = vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(b.binding)
                    .descriptor_type(b.resource.descriptor_type());
                match info {
                    WriteInfo::Buffer(info) => write.buffer_info(info),
                    WriteInfo::Image(info) => write.image_info(info),
                }
            })
            .collect()
    }
}

/// One binding of a [`Kit::descriptor_set`]: its number, the stages that read it, and what it
/// holds.
#[derive(Clone, Copy)]
pub(crate) struct SetBinding {
    pub(crate) binding: u32,
    pub(crate) stages: vk::ShaderStageFlags,
    pub(crate) resource: SetResource,
}
