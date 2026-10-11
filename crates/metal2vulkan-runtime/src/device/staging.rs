//! The device's staging queue: the first queue of a family of transfers alone, a copy engine
//! (`create::queue_plan`), where a `Shared` or `Managed` texture's CPU copies run on a device that
//! copies none on the host from memory the host caches (`texture/host_copy.rs`). An image that
//! copies here is concurrent among the device's families and this one ([`Device::staging_sharing`]),
//! so no image changes owner; every other resource keeps the device's sharing (`sharing.rs`).
//!
//! - **Why a copy engine.** A copy on the device's first `VkQueue` waits behind every command
//!   buffer submitted there before it: behind a 2 s kernel, a 1080p `get_bytes` returned after
//!   1.99 s on an RTX 3090, and forever behind a command buffer waiting on an event only a later
//!   command of the calling thread signals (the consumer's guest hang). Metal's `getBytes` and
//!   `replaceRegion` of shared storage are CPU copies that wait for no GPU work. Another queue of
//!   the graphics family is not enough: on the 3090 (615.71.09) a readback there waited 1.88 s for
//!   a 2 s kernel running on the first, where the copy engine's took 7.8 to 13.5 ms with the
//!   kernel still running, and 2.65 ms idle (EV XSH-sqp1-3090, raw Vulkan). Here nothing but these
//!   copies runs, each waited for before the next, so a copy waits for itself alone.
//! - **What it waits for.** Each submission waits, on the GPU, for the value through which each
//!   of the device's other `VkQueue`s completed as the host saw it: work that already completed,
//!   so the wait costs nothing, and the writes of every command buffer the program saw complete
//!   are visible to the copy. It never waits for work in flight: a command in flight meanwhile
//!   is the program's race, as on Metal.
//! - **Who uses it.** A texture made to copy through it (`texture/staging_copy.rs`) has its first
//!   move and zeros written here as it is made, and its `replace_region` and `get_bytes` copied
//!   here, each waited for, so none of its work waits in `texture/first_move.rs` for a batch (a
//!   batch submitted later finds it in `GENERAL`, its bytes written).

use super::{lock, Device, LoaderWord};
use crate::error::{internal, vulkan, Error};
use crate::exit_guard::{self, Call};
use crate::release;
use crate::sharing::Sharing;
use ash::vk;
use ash::vk::Handle;
use std::sync::Mutex;
use std::time::Duration;

/// How long one wait for a staging submission lasts before it is repeated.
const WAIT_SLICE: Duration = Duration::from_millis(100);

/// The staging queue and its objects.
pub(super) struct StagingQueue {
    /// The queue's family: one of transfers alone.
    family: u32,
    /// The device's families, then the queue's: those an image that copies here is shared
    /// among.
    families: Vec<u32>,
    /// The queue's loader word, put back before each use (`relay.rs`).
    word: LoaderWord,
    /// The queue, and the objects its submissions use once made: under one lock, held for a
    /// copy's submission and its wait.
    objects: Mutex<Objects>,
}

struct Objects {
    queue: vk::Queue,
    made: Option<Made>,
}

/// The command pool, command buffer and fence of the staging submissions, made on first use.
struct Made {
    pool: vk::CommandPool,
    buffer: vk::CommandBuffer,
    fence: vk::Fence,
}

impl StagingQueue {
    /// The staging queue `queue` of `family`, on a device that submits to `families`.
    ///
    /// # Safety
    ///
    /// `queue` was just returned by the device: its word is the loader's.
    pub(super) unsafe fn new(family: u32, queue: vk::Queue, families: &[u32]) -> StagingQueue {
        StagingQueue {
            family,
            families: families.iter().copied().chain([family]).collect(),
            // SAFETY: the caller's contract.
            word: unsafe { LoaderWord::of_handle(queue.as_raw()) },
            objects: Mutex::new(Objects { queue, made: None }),
        }
    }

    /// Destroy the objects; the device is idle.
    pub(super) fn destroy(&self, device: &ash::Device) {
        if let Some(made) = lock(&self.objects).made.take() {
            // SAFETY: the device is idle and nothing else holds these objects.
            unsafe {
                release::object(device, made.fence);
                release::object(device, made.pool);
            }
        }
    }
}

impl Made {
    fn new(device: &Device, family: u32) -> Result<Made, Error> {
        let raw = device.shared().vk();
        let pool_info = vk::CommandPoolCreateInfo::default()
            .flags(
                vk::CommandPoolCreateFlags::TRANSIENT
                    | vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            )
            .queue_family_index(family);
        // SAFETY: the device is live and `pool_info` valid.
        let pool = unsafe { raw.create_command_pool(&pool_info, None) }.map_err(vulkan)?;
        let made = (|| {
            let allocate = vk::CommandBufferAllocateInfo::default()
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let buffer = device.allocate_command_buffers(&allocate)?[0];
            // SAFETY: the device is live.
            let fence = unsafe { raw.create_fence(&vk::FenceCreateInfo::default(), None) }
                .map_err(vulkan)?;
            Ok(Made {
                pool,
                buffer,
                fence,
            })
        })();
        if made.is_err() {
            // SAFETY: nothing allocated from the pool is in use.
            unsafe { release::object(raw, pool) };
        }
        made
    }
}

impl Device {
    /// Whether the device reserved a staging queue (the module docs): a device it created with a
    /// family of transfers alone whose copies take any texel.
    pub(crate) fn has_staging_queue(&self) -> bool {
        self.shared().staging.is_some()
    }

    /// The sharing of an image that copies on the staging queue: concurrent among the device's
    /// families and the staging queue's; `None` where the device has none.
    pub(crate) fn staging_sharing(&self) -> Option<Sharing<'_>> {
        (self.shared().staging.as_ref()).map(|staging| Sharing::new(&staging.families))
    }

    /// Record `record` into the staging queue's command buffer, submit it there, waiting on the
    /// GPU for what the device's other `VkQueue`s completed as the host saw (the module docs),
    /// and wait for it. `record` runs outside a render pass, for a family of transfers alone: it
    /// records copies, fills and barriers, of color aspects alone.
    pub(crate) fn submit_staging(
        &self,
        record: impl FnOnce(&ash::Device, vk::CommandBuffer),
    ) -> Result<(), Error> {
        if self.is_lost() {
            return Err(Error::DeviceLost);
        }
        let shared = self.shared();
        let staging = (shared.staging.as_ref())
            .ok_or_else(|| internal("a staging copy on a device without a staging queue"))?;
        let raw = shared.vk();
        let mut objects = lock(&staging.objects);
        if objects.made.is_none() {
            objects.made = Some(Made::new(self, staging.family)?);
        }
        let queue = objects.queue;
        let Some(made) = objects.made.as_ref() else {
            unreachable!("made above");
        };
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: the command buffer is not pending (every submission is waited for below) and
        // the staging lock is held; beginning it resets it (the pool resets command buffers).
        unsafe { raw.begin_command_buffer(made.buffer, &begin) }.map_err(vulkan)?;
        record(raw, made.buffer);
        // SAFETY: as above.
        unsafe { raw.end_command_buffer(made.buffer) }.map_err(vulkan)?;
        // SAFETY: the fence is unsignaled or signaled and not pending.
        unsafe { raw.reset_fences(&[made.fence]) }.map_err(vulkan)?;
        // The completed work of every other queue: a wait that costs nothing, for visibility.
        let completed: Vec<(vk::Semaphore, u64)> = (shared.queues.iter())
            .map(|slot| (slot.timeline, slot.completed_value()))
            .filter(|&(_, value)| value > 0)
            .collect();
        let wait_semaphores: Vec<vk::Semaphore> = completed.iter().map(|w| w.0).collect();
        let wait_values: Vec<u64> = completed.iter().map(|w| w.1).collect();
        let wait_stages = vec![vk::PipelineStageFlags::ALL_COMMANDS; completed.len()];
        let mut timeline =
            vk::TimelineSemaphoreSubmitInfo::default().wait_semaphore_values(&wait_values);
        let buffers = [made.buffer];
        let submit = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&buffers)
            .push_next(&mut timeline);
        // SAFETY: the queue is used under the staging lock alone; the command buffer is
        // executable; every value waited for was signalled (the host saw it).
        let submitted = unsafe {
            staging.word.restore();
            raw.queue_submit(queue, &[submit], made.fence)
        };
        exit_guard::called(Call::QueueSubmit);
        submitted.map_err(|error| self.failed(error))?;
        shared.counts.staging_submission();
        loop {
            // SAFETY: the fence was submitted just now.
            match unsafe { raw.wait_for_fences(&[made.fence], true, WAIT_SLICE.as_nanos() as u64) }
            {
                Ok(()) => return Ok(()),
                Err(vk::Result::TIMEOUT) => continue,
                Err(error) => return Err(self.failed(error)),
            }
        }
    }
}
