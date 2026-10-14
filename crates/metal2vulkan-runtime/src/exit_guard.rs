//! The exit guard: no Vulkan call of the runtime overlaps the destructors a process runs as it
//! exits.
//!
//! A process exits (`exit`, or the end of `main`) on one thread, which runs the exit handlers and
//! the loaded libraries' destructors in the reverse order of their registration; every other
//! thread keeps running meanwhile. A driver's or a layer's destructors free what a later Vulkan
//! call needs: RADV ended a process with `SIGSEGV` when the completion thread destroyed a device
//! beside the validation layer's destructors.
//!
//! - **Sections.** Each Vulkan call the runtime makes on its own threads (the scheduler's
//!   submissions, the completion thread's polls), each release of a
//!   Vulkan object (a drop, on whatever thread it runs), and each read of an event's counter runs
//!   inside a section ([`enter`]), which counts itself. A section runs none of the program's code
//!   (handlers, deallocators, notifications) except a device's message sink, which Vulkan calls
//!   inside it, and publishes nothing it did (a status, a handler's turn, an event's value, a
//!   batch to poll for) before it ended. A section started runs to its end, the sections inside
//!   it included, even when the guard closes meanwhile.
//! - **Coverage, by construction and by test.** Four doors, which new code cannot go around:
//!   - every release (`vkDestroy*`, `vkFree*`) goes through `release.rs`, which runs it inside a
//!     section and marks its entry point; a new kind of object implements its `Released` there.
//!     A source test fails when another file calls such an entry point;
//!   - every thread the runtime starts is started by [`spawn`], and every call of the program's
//!     code (handlers, notifications, deallocators, the message sink) goes through [`program`];
//!     source tests fail when a thread is started or a panic caught elsewhere;
//!   - the device's functions are reached only through checked accessors (`DeviceShared::vk`,
//!     private field), which in builds with debug assertions end the process
//!     when one of the runtime's threads reaches them outside a section and outside the
//!     program's code ([`check`]). Every lane runs that way, so a Vulkan call a runtime thread
//!     makes outside a section fails the lane that reaches it;
//!   - the entry points sections call are one table ([`Call`]): in those builds a section reaches
//!     the table's alone, every other one ending the process (`device/functions.rs`), and each
//!     device's priming makes every one it can and checks that it did (`device/prime.rs`). A
//!     new entry point in a section goes into the table and the priming, or fails the lane.
//!
//!   The program's own calls on its own threads (making objects, encoding) take no section:
//!   their race with an exit is the program's.
//! - **A host step makes no image, render pass, framebuffer or neutral**. Those are outside the
//!   table, and a host step ([`host_step`]) runs inside a section: what one may need is made before
//!   it, on the program's thread. Their one door (`creation.rs`, and `neutral.rs` for the neutrals)
//!   refuses one in a host step before any Vulkan call, `Error::HostStepObject`, in every build; a
//!   source test fails when another file makes such an object.
//! - **The hook.** A device made or adopted registers an exit handler ([`register_device`]),
//!   once its driver and layers are loaded and primed (`device/prime.rs`: the first call of each
//!   entry point of the table, made on the creating thread), so it runs before their destructors,
//!   those of statics made at those first calls included: the process's first device, one whose
//!   driver and functions are new since the process last loaded a shared object, and the first
//!   after such a load ([`loaded`]). A device made again on what stays loaded registers nothing.
//!   It closes the guard, so no section starts, and waits for the sections in progress to end: at
//!   most [`BOUND`], and never for the exiting thread's own (an exit from a message sink). A
//!   second run finds nothing to wait for. While a driver's making of a `VkPipeline` is in progress
//!   in a section (a pool's piece's `vkCreate*Pipelines`, [`driver_making`]), at most
//!   [`MAKING_BOUND`], and it says so on standard error as that wait starts: a making outlasts
//!   [`BOUND`], and the program need not wait for the command buffer whose making it is before it
//!   exits. Seen: with two seconds, two lane processes on ANV ended with `SIGSEGV` and `SIGABRT`
//!   inside the validation layer 3 and 20 s after their test passed, where a bicubic kernel's
//!   making took 6.6 s under the lane's load (EV `PRDC-gl1-anv`); a making in progress then is
//!   inferred from the timing.
//! - **Registered again.** Drivers and layers also register destructors later, when a static is
//!   made at its first use. Measured: with the hook registered only at the first device, polls
//!   and device waits crashed inside the validation layer's synchronization code at exit while
//!   the hook still waited for them. So a section that ends after the first call of a Vulkan
//!   entry point the runtime calls in sections ([`Call`], a fixed set) registers the hook again,
//!   after anything that call registered, and so does every section whose count in the process
//!   reaches a power of two, for statics made at a later call with new parameters: at most 64,
//!   and one per entry point of the table at each load. A load is where the first call counts
//!   from: the Vulkan loader unloads the layers with the last instance and loads them again with
//!   the next, which make their statics again at each entry point's first call after the device
//!   that registered at the load; so that device forgets the entry points called before
//!   ([`forget_calls`]). Measured: before, a teardown lane whose six devices each reloaded the
//!   validation layer ended with `SIGSEGV` at exit, a pool thread's `vkDestroyRenderPass` (its
//!   making's pass) calling through a static of the layer's destroyed first, while the hook waited
//!   for that section (EVs PRDP-hl1-anv, PRDP-hl1-radv, PRDP-hl2-anv: in the core, that pool
//!   thread, and the main thread asleep as the hook's wait polls). The hook then runs before every
//!   destructor registered before its last registration. Since a section publishes nothing before
//!   it ended, an exit the program makes once it learned of a section's work (from a handler, a
//!   wait, a status) comes after that section's registration. Measured: while the scheduler posted
//!   a command buffer's scheduling inside its section, an exit from a scheduled handler crashed the
//!   completion thread's poll on MoltenVK in 10 runs of 10. The hole left: a static first made by a
//!   call after the hook's last registration (one in progress as the exit begins included), when
//!   the process exits before the next registration.
//! - **Once closed.** The scheduler and the completion thread end at their next section, so what
//!   is pending stays pending, as Metal's does. A release skips its Vulkan calls (the process's
//!   end frees what they would), and a no-copy buffer released then keeps its deallocator unrun,
//!   with its pages imported. The handler threads, listeners and the shader cache's thread go on;
//!   a making that ends then saves no record (`shader_cache/driver.rs`).
//! - **A CPU device's work.** A device whose queues run in this process (`DeviceKind::Cpu`:
//!   lavapipe, SwiftShader) runs what was submitted to it on the process's own threads, so the
//!   destructors would run under its execution. Once the sections ended, the hook waits for the
//!   work submitted to each such device's queues ([`InProcess`], [`runs_in_process`]), within
//!   what is left of [`BOUND`]; what is not submitted stays pending. A GPU's work runs on the
//!   GPU, and the hook does not wait for it, as Metal's exit does not. Measured on lavapipe
//!   (Mesa 26.2.4, x86_64): with work in flight at exit, its queue thread compiled a dispatch's
//!   shader (LLVM's `MCJIT::finalizeObject`) while the destructors ran, and the process ended in
//!   `free()`'s heap checks, in 5 of 300 exits while a host step waited and 1 of 300 with a
//!   plain dispatch in flight.
//! - **No exit waits for a thread the guard ended.** Those two threads end between sections,
//!   holding no lock. Two things wait for them:
//!   - joins, made only by a device's teardown, which is one section: once the guard closed none
//!     begins, and one already begun returns, since the threads end;
//!   - waits for their work: a command buffer's scheduling and completion, and an event's value
//!     the GPU raises. Each such wait rechecks the guard at least every [`WAKE`] and then answers
//!     at once: `wait_until_completed` and `wait_until_scheduled` with `WaitWouldDeadlock` of
//!     `DeadlockCause::Exiting` (which [`process_exiting`] answers is so),
//!     `wait_until_completed_for` with `TimedOut`, `wait_until_signaled_value` with the value
//!     known. A handler's wait for another queue is one of them.
//!
//!   The hook itself waits only for sections, and a section waits for nothing the exiting thread
//!   holds but through a message sink, which [`BOUND`] covers. So nothing run before the exit
//!   returns (later exit handlers and destructors, or a thread one of them joins) waits for a
//!   thread the guard ended.
//! - **Forks.** The hook acts only in the process it was registered in: a child forked without
//!   exec, which has none of its parent's threads, runs it as a no-op, and registers its own with
//!   its first device.

use super::loaded;
use std::cell::Cell;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError, Weak};
use std::time::{Duration, Instant};

/// The longest the hook waits for the sections in progress, and then for a CPU device's
/// submitted work ([`InProcess`]). A section is one submission, one poll (at most 100 ms), one
/// counter read, one save's read, or one object's release (a device's: its threads' stops,
/// `vkDeviceWaitIdle`, its destruction); it outlasts this only when a driver or a message sink
/// blocks.
pub(crate) const BOUND: Duration = Duration::from_secs(2);

/// The longest the hook waits, while a driver's making of a `VkPipeline` is in progress in a
/// section ([`driver_making`]), for the sections in progress: a making takes 20 to 50 s on
/// NVIDIA (`pipeline/piece_holds.rs`), and Metal's process waited for all of it at the state's
/// creation, before its exit.
pub(crate) const MAKING_BOUND: Duration = Duration::from_secs(60);

/// The longest a blocked wait goes without looking at the guard.
pub(crate) const WAKE: Duration = Duration::from_millis(50);

/// How often the hook looks at the sections in progress.
const POLL: Duration = Duration::from_millis(1);

/// The sections in progress, and whether the guard closed. The process has one ([`GUARD`]).
pub(crate) struct Guard {
    active: AtomicUsize,
    closed: AtomicBool,
    /// The sections in progress that make a `VkPipeline` ([`Guard::enter_making`]).
    making: AtomicUsize,
}

/// The process's guard.
static GUARD: Guard = Guard::new();

/// The process's devices whose queues run in it.
static IN_PROCESS: InProcessDevices = InProcessDevices::new();

/// A device whose queues run in this process (a CPU device): the work submitted to them runs on
/// the process's own threads, which the exit's destructors must not run under.
pub(crate) trait InProcess: Send + Sync {
    /// Wait until the work submitted to its queues completed, at most until `deadline`: whether
    /// it did. Called by the hook once the sections ended, so nothing is submitted meanwhile.
    fn drain(&self, deadline: Instant) -> bool;
}

/// The devices whose queues run in this process, each with the process it was made in (a forked
/// child has none of its parent's threads, its devices' included).
pub(crate) struct InProcessDevices {
    devices: Mutex<Vec<(u32, Weak<dyn InProcess>)>>,
}

impl InProcessDevices {
    /// None.
    pub(crate) const fn new() -> InProcessDevices {
        InProcessDevices {
            devices: Mutex::new(Vec::new()),
        }
    }

    /// Add `device`, made in this process; those gone are dropped from the list.
    pub(crate) fn add(&self, device: Weak<dyn InProcess>) {
        let mut devices = self.devices.lock().unwrap_or_else(PoisonError::into_inner);
        devices.retain(|(_, device)| device.strong_count() > 0);
        devices.push((std::process::id(), device));
    }

    /// Wait for the submitted work of each live device made in process `pid`, at most until
    /// `deadline`: whether every one completed.
    pub(crate) fn drain(&self, pid: u32, deadline: Instant) -> bool {
        let live: Vec<Arc<dyn InProcess>> = self
            .devices
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .filter(|(made, _)| *made == pid)
            .filter_map(|(_, device)| device.upgrade())
            .collect();
        // Every device is waited for, whatever an earlier one answered.
        let mut drained = true;
        for device in &live {
            drained &= device.drain(deadline);
        }
        drained
    }
}

/// Wait at exit for the work submitted to `device`, whose queues run in this process.
pub(crate) fn runs_in_process(device: Weak<dyn InProcess>) {
    IN_PROCESS.add(device);
}

/// The process the hook is registered in; 0 before the first device.
static REGISTERED: AtomicU32 = AtomicU32::new(0);

/// The sections of the process's guard that ended: the hook is registered again when it reaches
/// a power of two.
static SECTIONS: AtomicU64 = AtomicU64::new(0);

/// The entry points ([`Call`]) called in sections that ended since the hook was last registered
/// at a device after a load of a shared object ([`register_device`], they were the process's), one
/// bit each.
static CALLED: CalledWord = CalledWord([AtomicU64::new(0), AtomicU64::new(0)]);

/// One bit per entry point of the table ([`Call::bit`]), as two atomic halves: the table
/// outgrew 64 bits with the render host step's entry points.
struct CalledWord([AtomicU64; 2]);

impl CalledWord {
    /// Set `calls`; the bits set before.
    fn fetch_or(&self, calls: u128) -> u128 {
        let low = self.0[0].fetch_or(calls as u64, Ordering::Relaxed);
        let high = self.0[1].fetch_or((calls >> 64) as u64, Ordering::Relaxed);
        u128::from(low) | (u128::from(high) << 64)
    }

    /// Clear every bit: a load of a shared object since (a driver or layer loaded again makes
    /// its statics again at each entry point's first call).
    fn clear(&self) {
        self.0[0].store(0, Ordering::Relaxed);
        self.0[1].store(0, Ordering::Relaxed);
    }

    /// The bits set.
    #[cfg(test)]
    fn load(&self) -> u128 {
        let low = self.0[0].load(Ordering::Relaxed);
        let high = self.0[1].load(Ordering::Relaxed);
        u128::from(low) | (u128::from(high) << 64)
    }
}

/// The table of the Vulkan entry points the runtime calls inside its sections, each with its
/// name: [`Call`], [`Call::ALL`] and [`Call::name`] come from it alone. Three things read it:
/// a section that ends after the first call of one registers the hook again ([`called`]);
/// inside a section the device's functions are the table's alone (`device/functions.rs`); and
/// each device's priming makes the first call of each before the hook is registered for it
/// (`device/prime.rs`), which checks that it made them all.
macro_rules! calls {
    ($($call:ident => $name:literal,)*) => {
        /// A Vulkan entry point the runtime calls inside its sections (the table above).
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub(crate) enum Call {
            $($call,)*
        }

        impl Call {
            /// Every entry point, in order.
            pub(crate) const ALL: &'static [Call] = &[$(Call::$call,)*];

            /// Its Vulkan name.
            pub(crate) fn name(self) -> &'static std::ffi::CStr {
                match self {
                    $(Call::$call => $name,)*
                }
            }
        }
    };
}

calls! {
    QueueSubmit => c"vkQueueSubmit",
    // A batch's sparse binding (`sparse/bind.rs`), on a device that binds sparse memory alone
    // ([`SPARSE_CALLS`](crate::device::prime::SPARSE_CALLS)).
    QueueBindSparse => c"vkQueueBindSparse",
    WaitSemaphores => c"vkWaitSemaphores",
    GetSemaphoreCounterValue => c"vkGetSemaphoreCounterValue",
    GetPipelineCacheData => c"vkGetPipelineCacheData",
    // A making's own pipeline cache (`shader_cache/driver.rs`), made in a host step's or a pool
    // piece's making as in the program's.
    CreatePipelineCache => c"vkCreatePipelineCache",
    DeviceWaitIdle => c"vkDeviceWaitIdle",
    CreateCommandPool => c"vkCreateCommandPool",
    AllocateCommandBuffers => c"vkAllocateCommandBuffers",
    BeginCommandBuffer => c"vkBeginCommandBuffer",
    CmdPipelineBarrier => c"vkCmdPipelineBarrier",
    CmdResetQueryPool => c"vkCmdResetQueryPool",
    CmdWriteTimestamp => c"vkCmdWriteTimestamp",
    GetQueryPoolResults => c"vkGetQueryPoolResults",
    EndCommandBuffer => c"vkEndCommandBuffer",
    CreateFence => c"vkCreateFence",
    // The timeline of the device's internal submissions (`device/submit.rs`), made at the first,
    // which an indirect sparse update's read in a host step can be.
    CreateSemaphore => c"vkCreateSemaphore",
    WaitForFences => c"vkWaitForFences",
    // The completion thread's drain of a device's queues after a failed wait
    // (`Device::drain_queues`).
    QueueWaitIdle => c"vkQueueWaitIdle",
    ResetFences => c"vkResetFences",
    DestroyCommandPool => c"vkDestroyCommandPool",
    DestroyFence => c"vkDestroyFence",
    DestroySemaphore => c"vkDestroySemaphore",
    DestroySampler => c"vkDestroySampler",
    DestroyBuffer => c"vkDestroyBuffer",
    DestroyBufferView => c"vkDestroyBufferView",
    DestroyImage => c"vkDestroyImage",
    DestroyImageView => c"vkDestroyImageView",
    FreeMemory => c"vkFreeMemory",
    DestroyPipeline => c"vkDestroyPipeline",
    DestroyPipelineLayout => c"vkDestroyPipelineLayout",
    DestroyDescriptorSetLayout => c"vkDestroyDescriptorSetLayout",
    DestroyDescriptorPool => c"vkDestroyDescriptorPool",
    DestroyShaderModule => c"vkDestroyShaderModule",
    DestroyRenderPass => c"vkDestroyRenderPass",
    DestroyFramebuffer => c"vkDestroyFramebuffer",
    DestroyPipelineCache => c"vkDestroyPipelineCache",
    DestroyQueryPool => c"vkDestroyQueryPool",
    // A `VK_EXT_device_generated_commands` object's (`generated_commands.rs`).
    DestroyIndirectExecutionSet => c"vkDestroyIndirectExecutionSetEXT",
    DestroyIndirectCommandsLayout => c"vkDestroyIndirectCommandsLayoutEXT",
    DestroyDevice => c"vkDestroyDevice",
    DestroyDebugUtilsMessenger => c"vkDestroyDebugUtilsMessengerEXT",
    DestroyInstance => c"vkDestroyInstance",
    // A host step's ([`HOST_STEP`]), and a translation's on a device without device addresses.
    CmdBindPipeline => c"vkCmdBindPipeline",
    CmdBindDescriptorSets => c"vkCmdBindDescriptorSets",
    CmdPushConstants => c"vkCmdPushConstants",
    CmdDispatch => c"vkCmdDispatch",
    CmdCopyBuffer => c"vkCmdCopyBuffer",
    CmdCopyBufferToImage => c"vkCmdCopyBufferToImage",
    CmdCopyImageToBuffer => c"vkCmdCopyImageToBuffer",
    // A sparse batch's copy of a move's texels (`sparse/emulated.rs`).
    CmdCopyImage => c"vkCmdCopyImage",
    CreateDescriptorPool => c"vkCreateDescriptorPool",
    AllocateDescriptorSets => c"vkAllocateDescriptorSets",
    UpdateDescriptorSets => c"vkUpdateDescriptorSets",
    CreateBuffer => c"vkCreateBuffer",
    GetBufferMemoryRequirements => c"vkGetBufferMemoryRequirements",
    AllocateMemory => c"vkAllocateMemory",
    BindBufferMemory => c"vkBindBufferMemory",
    MapMemory => c"vkMapMemory",
    GetBufferDeviceAddress => c"vkGetBufferDeviceAddress",
    CreateImageView => c"vkCreateImageView",
    CreateBufferView => c"vkCreateBufferView",
    CreateSampler => c"vkCreateSampler",
    CreateShaderModule => c"vkCreateShaderModule",
    CreateDescriptorSetLayout => c"vkCreateDescriptorSetLayout",
    CreatePipelineLayout => c"vkCreatePipelineLayout",
    CreateComputePipelines => c"vkCreateComputePipelines",
    // A render host step's ([`HOST_STEP`]): an executed render command's draw into a secondary
    // command buffer, and the pass that executes the secondaries.
    CmdBeginRenderPass => c"vkCmdBeginRenderPass",
    CmdEndRenderPass => c"vkCmdEndRenderPass",
    CmdExecuteCommands => c"vkCmdExecuteCommands",
    CmdBindVertexBuffers => c"vkCmdBindVertexBuffers",
    CmdDraw => c"vkCmdDraw",
    CmdSetViewport => c"vkCmdSetViewport",
    CmdSetScissor => c"vkCmdSetScissor",
    CmdSetDepthBias => c"vkCmdSetDepthBias",
    CmdSetBlendConstants => c"vkCmdSetBlendConstants",
    CmdSetStencilCompareMask => c"vkCmdSetStencilCompareMask",
    CmdSetStencilWriteMask => c"vkCmdSetStencilWriteMask",
    CmdSetStencilReference => c"vkCmdSetStencilReference",
    CreateGraphicsPipelines => c"vkCreateGraphicsPipelines",
    // A render host step's execution under a visibility result mode: the query lent to it, begun
    // in its secondary before its first command and ended after its last.
    CmdBeginQuery => c"vkCmdBeginQuery",
    CmdEndQuery => c"vkCmdEndQuery",
    // A render host step's emulated mesh draw whose launch a scan counts: its mesh dispatch and
    // its draw, each from the scan's arguments (`render/mesh.rs`).
    CmdDispatchIndirect => c"vkCmdDispatchIndirect",
    CmdDrawIndirect => c"vkCmdDrawIndirect",
    // A render host step's draw whose pipeline was being made as it was encoded
    // (`render/pending.rs`): an indexed draw, direct or from its arguments.
    CmdBindIndexBuffer => c"vkCmdBindIndexBuffer",
    CmdDrawIndexed => c"vkCmdDrawIndexed",
    CmdDrawIndexedIndirect => c"vkCmdDrawIndexedIndirect",
    // A render host step's pass steps (`render/pass_steps.rs`): the clear of a sample plane
    // (`sample_planes.rs`) or of a shader-side stencil's shadow (`shadow_stencil.rs`).
    CmdClearColorImage => c"vkCmdClearColorImage",
    // A batch's first work: the clear of a neutral depth image, which may be multisampled
    // (`texture/first_move.rs`, `neutral.rs`).
    CmdClearDepthStencilImage => c"vkCmdClearDepthStencilImage",
}

/// The entry points a host step may call (`CommandBufferShared::host_step`): what a dispatch
/// recorded on the program's thread calls (a command buffer of its pool, the barriers, binds, push
/// constants and dispatch, the shadows' copies, descriptor pools and sets, upload chunks, the views
/// and samplers of bound textures, a `VkPipeline` from a compiled variant), on a device without
/// device addresses a variant's translation, and an indirect sparse update's read of its regions.
/// Its section marks them all as called as it ends ([`called`]), a superset of what one step calls,
/// so the process's first step registers the hook again after it.
pub(crate) const HOST_STEP: &[Call] = &[
    Call::CreateCommandPool,
    Call::AllocateCommandBuffers,
    Call::BeginCommandBuffer,
    Call::CmdPipelineBarrier,
    Call::EndCommandBuffer,
    Call::CmdBindPipeline,
    Call::CmdBindDescriptorSets,
    Call::CmdPushConstants,
    Call::CmdDispatch,
    Call::CmdCopyBuffer,
    Call::CmdCopyBufferToImage,
    Call::CmdCopyImageToBuffer,
    Call::CreateDescriptorPool,
    Call::AllocateDescriptorSets,
    Call::UpdateDescriptorSets,
    Call::CreateBuffer,
    Call::GetBufferMemoryRequirements,
    Call::AllocateMemory,
    Call::BindBufferMemory,
    Call::MapMemory,
    Call::GetBufferDeviceAddress,
    Call::CreateImageView,
    Call::CreateBufferView,
    Call::CreateSampler,
    Call::CreateShaderModule,
    Call::CreateDescriptorSetLayout,
    Call::CreatePipelineLayout,
    Call::CreateComputePipelines,
    // A making's own pipeline cache, seeded from its record and read for its save
    // (`shader_cache/driver.rs`).
    Call::CreatePipelineCache,
    Call::GetPipelineCacheData,
    // An indirect sparse update's read of a `Private` buffer (`sparse/indirect.rs`): a copy through
    // the device's internal command buffer, waited for, which makes that buffer's fence and
    // timeline where it is the device's first internal submission.
    Call::CreateFence,
    Call::CreateSemaphore,
    Call::ResetFences,
    Call::QueueSubmit,
    Call::WaitForFences,
    // A render pass whose render commands a host step records (`render/executions.rs`): each
    // command's draw into a secondary command buffer, from a compiled variant's `VkPipeline`, and
    // the pass's instances executing the secondaries.
    Call::CmdBeginRenderPass,
    Call::CmdEndRenderPass,
    Call::CmdExecuteCommands,
    Call::CmdBindVertexBuffers,
    Call::CmdDraw,
    Call::CmdSetViewport,
    Call::CmdSetScissor,
    Call::CmdSetDepthBias,
    Call::CmdSetBlendConstants,
    Call::CmdSetStencilCompareMask,
    Call::CmdSetStencilWriteMask,
    Call::CmdSetStencilReference,
    Call::CreateGraphicsPipelines,
    // An execution recorded under a visibility result mode: its lent query.
    Call::CmdBeginQuery,
    Call::CmdEndQuery,
    // An emulated mesh command whose launch a scan counts: its mesh dispatch and its draw from the
    // scan's arguments.
    Call::CmdDispatchIndirect,
    Call::CmdDrawIndirect,
    // A draw whose pipeline was being made as it was encoded (`render/pending.rs`).
    Call::CmdBindIndexBuffer,
    Call::CmdDrawIndexed,
    Call::CmdDrawIndexedIndirect,
    // A render pass's steps around its instances (`render/pass_steps.rs`), which a pass of
    // such draws records in the host step too: a sample plane's or a stencil shadow's clear.
    Call::CmdClearColorImage,
];

impl Call {
    /// Its bit. The table fills 88 of the word's 128 (widened from 64): when it next outgrows the
    /// word, widen the word (here, [`CALLED`], `PENDING` and the priming's calls); do not add a
    /// second table.
    pub(crate) fn bit(self) -> u128 {
        1 << self as u32
    }
}

thread_local! {
    /// The sections in progress on this thread (of every guard).
    static OWN: Cell<usize> = const { Cell::new(0) };
    /// The entry points ([`Call`]) this thread called since its last section ended.
    static PENDING: Cell<u128> = const { Cell::new(0) };
    /// This thread is one of the runtime's own ([`spawn`]).
    static RUNTIME: Cell<bool> = const { Cell::new(false) };
    /// The program's code this thread is running ([`program`]).
    static PROGRAM: Cell<usize> = const { Cell::new(0) };
    /// The sections in progress on this thread when it began running the program's code it
    /// runs now: those started since are the runtime's again ([`in_section`]).
    static BASE: Cell<usize> = const { Cell::new(0) };
    /// The host steps this thread is recording now ([`host_step`]).
    static HOST_STEPS: Cell<usize> = const { Cell::new(0) };
}

/// Whether this thread runs inside a call of the program's: one of the program's own threads,
/// or a runtime thread running the program's code ([`program`]), which may call the runtime.
pub(crate) fn in_program_call() -> bool {
    !RUNTIME.with(Cell::get) || PROGRAM.with(Cell::get) > 0
}

/// Run `f`, a command buffer's host step, marked as one ([`in_host_step`]): a serial point of the
/// command buffers' stream, where the work before the batch completed and the batch is not yet
/// submitted.
pub(crate) fn host_step<T>(f: impl FnOnce() -> T) -> T {
    /// Ends the mark as the step returns or unwinds.
    struct Mark;
    impl Drop for Mark {
        fn drop(&mut self) {
            HOST_STEPS.with(|steps| steps.set(steps.get() - 1));
        }
    }
    HOST_STEPS.with(|steps| steps.set(steps.get() + 1));
    let _mark = Mark;
    f()
}

/// Whether this thread is recording a host step now ([`host_step`]).
pub(crate) fn in_host_step() -> bool {
    HOST_STEPS.with(Cell::get) > 0
}

/// A section in progress: it ends when dropped, on the thread it started on.
pub(crate) struct Busy<'a> {
    guard: &'a Guard,
    /// A section of the process's guard ([`GUARD`]), counted in [`SECTIONS`] as it ends.
    counted: bool,
    /// It makes a `VkPipeline`, counted in its guard's makings ([`Guard::enter_making`]).
    making: bool,
    /// Counted on this thread ([`OWN`]), so neither `Send` nor `Sync`.
    _thread: PhantomData<*const ()>,
}

impl Guard {
    /// An open guard.
    pub(crate) const fn new() -> Guard {
        Guard {
            active: AtomicUsize::new(0),
            closed: AtomicBool::new(false),
            making: AtomicUsize::new(0),
        }
    }

    /// Start a section; `None` once the guard closed, unless this thread is in a section
    /// already: a section that started runs to its end, the sections inside it included.
    ///
    /// The count rises before the guard is read and [`Guard::close`] closes it before it reads
    /// the count, both sequentially consistent: a section either sees the guard closed and
    /// starts not, or is counted by the time the closing thread reads the count. A section
    /// inside another is counted while the closing thread waits for the outer one.
    pub(crate) fn enter(&self) -> Option<Busy<'_>> {
        self.active.fetch_add(1, Ordering::SeqCst);
        if self.closed.load(Ordering::SeqCst) && OWN.with(Cell::get) == 0 {
            self.active.fetch_sub(1, Ordering::SeqCst);
            return None;
        }
        OWN.with(|own| own.set(own.get() + 1));
        Some(Busy {
            guard: self,
            counted: false,
            making: false,
            _thread: PhantomData,
        })
    }

    /// [`enter`](Self::enter) a section around a driver's making of a `VkPipeline`: while one is
    /// in progress, [`close`](Self::close) waits up to [`MAKING_BOUND`].
    pub(crate) fn enter_making(&self) -> Option<Busy<'_>> {
        let mut busy = self.enter()?;
        busy.making = true;
        self.making.fetch_add(1, Ordering::SeqCst);
        Some(busy)
    }

    /// Whether the guard closed.
    pub(crate) fn is_closed(&self) -> bool {
        self.closed.load(Ordering::SeqCst)
    }

    /// Close the guard, then wait until no more sections are in progress than `own` (the
    /// closing thread's), at most `bound`, or while a section that makes a `VkPipeline` is in
    /// progress, at most [`MAKING_BOUND`]: whether they ended.
    pub(crate) fn close(&self, own: usize, bound: Duration) -> bool {
        self.closed.store(true, Ordering::SeqCst);
        let start = Instant::now();
        let mut said = false;
        loop {
            if self.active.load(Ordering::SeqCst) <= own {
                return true;
            }
            let waited = start.elapsed();
            let making = self.making.load(Ordering::SeqCst);
            if waited >= bound && (making == 0 || waited >= MAKING_BOUND) {
                return false;
            }
            if waited >= bound && !said {
                said = true;
                eprintln!(
                    "metal2vulkan: the exit waits for {making} pipeline making(s) in the driver, \
                     at most {} s more",
                    MAKING_BOUND.saturating_sub(waited).as_secs()
                );
            }
            std::thread::sleep(POLL);
        }
    }
}

impl Drop for Busy<'_> {
    fn drop(&mut self) {
        OWN.with(|own| own.set(own.get() - 1));
        if self.making {
            self.guard.making.fetch_sub(1, Ordering::SeqCst);
        }
        self.guard.active.fetch_sub(1, Ordering::SeqCst);
        if self.counted {
            // After the section's Vulkan calls, which may have registered a destructor.
            let calls = PENDING.with(|pending| {
                let calls = pending.get();
                pending.set(0);
                calls
            });
            let first = calls & !CALLED.fetch_or(calls) != 0;
            let ended = SECTIONS.fetch_add(1, Ordering::Relaxed) + 1;
            if (first || ended.is_power_of_two()) && !self.guard.is_closed() {
                register_again();
            }
        }
    }
}

/// Start a section of the process's guard; `None` once the process is exiting.
pub(crate) fn enter() -> Option<Busy<'static>> {
    let mut busy = GUARD.enter()?;
    busy.counted = true;
    Some(busy)
}

/// Run `make`, a driver's making of `VkPipeline`s, inside a section of its own where this thread
/// is in a section (a pool's piece), which the hook waits for up to [`MAKING_BOUND`]
/// ([`Guard::enter_making`]); on the program's own thread, as it is. A section inside one runs
/// even once the guard closed, so the making is always counted.
pub(crate) fn driver_making<T>(make: impl FnOnce() -> T) -> T {
    let _making = match in_section() {
        true => GUARD.enter_making(),
        false => None,
    };
    make()
}

/// This thread called `call`; the section it is in registers the hook again as it ends if no
/// section that ended had called it.
pub(crate) fn called(call: Call) {
    PENDING.with(|pending| pending.set(pending.get() | call.bit()));
}

/// Whether the process is exiting: its guard closed.
pub(crate) fn closed() -> bool {
    GUARD.is_closed()
}

/// Whether this process is exiting, as the runtime sees it: the exit handler a device made or
/// adopted registers ran (`exit`, or the end of `main`), so the runtime's threads that schedule
/// and finish command buffers have ended and it makes no further Vulkan call. Once true it stays
/// true for the process; before any device was made or adopted it is false, since no handler is
/// registered. A command buffer wait then answers `WaitWouldDeadlock` of
/// [`DeadlockCause::Exiting`](crate::DeadlockCause::Exiting), and the bounded wait `TimedOut`.
///
/// No Metal counterpart: Metal's exit runs no such handler, and its waits for work that never
/// finishes hang. A consumer that copies that (a thread parked for good) asks this to tell an
/// exit from a deadlock the runtime refused.
pub fn process_exiting() -> bool {
    closed()
}

/// Start one of the runtime's own threads, running `f`: every thread the runtime starts is
/// started here, so it is known to [`may_call`]. On macOS it runs at no lower quality of service
/// than the default ([`raise_quality_of_service`]).
pub(crate) fn spawn<F, T>(
    builder: std::thread::Builder,
    f: F,
) -> std::io::Result<std::thread::JoinHandle<T>>
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    builder.spawn(move || {
        RUNTIME.with(|runtime| runtime.set(true));
        raise_quality_of_service();
        f()
    })
}

/// Start one of the runtime's own threads in `scope`, running `f`, as [`spawn`] does: a thread
/// that borrows its caller's arguments and is joined before its caller returns (a compile's own
/// thread, `device/compiles.rs`).
pub(crate) fn spawn_scoped<'scope, F, T>(
    scope: &'scope std::thread::Scope<'scope, '_>,
    builder: std::thread::Builder,
    f: F,
) -> std::io::Result<std::thread::ScopedJoinHandle<'scope, T>>
where
    F: FnOnce() -> T + Send + 'scope,
    T: Send + 'scope,
{
    builder.spawn_scoped(scope, move || {
        RUNTIME.with(|runtime| runtime.set(true));
        raise_quality_of_service();
        f()
    })
}

/// Start one of the runtime's threads that take a command buffer from its commit to its handlers,
/// running `f`, as [`spawn`] does, but at the user-interactive quality of service on macOS, and
/// with the utilization floor at its top on Linux, from its start ([`run_at_user_interactive`]):
/// the scheduler's (`scheduler/`), the completion thread
/// and each `VkQueue`'s waiter (`command_buffer/completion.rs`), and every thread of the handler
/// pool (`contexts/`).
pub(crate) fn spawn_user_interactive<F, T>(
    builder: std::thread::Builder,
    f: F,
) -> std::io::Result<std::thread::JoinHandle<T>>
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    #[cfg(test)]
    let on_start = ON_START.with(|on_start| on_start.borrow().clone());
    spawn(builder, move || {
        run_at_user_interactive();
        #[cfg(test)]
        if let Some(on_start) = on_start {
            on_start();
        }
        f()
    })
}

#[cfg(test)]
thread_local! {
    /// What each thread [`spawn_user_interactive`] starts from this thread runs once its class is
    /// set, before `f`: a test's reading of the class from inside the thread.
    pub(crate) static ON_START: std::cell::RefCell<Option<std::sync::Arc<dyn Fn() + Send + Sync>>> =
        const { std::cell::RefCell::new(None) };
}

/// Raise this thread's quality of service to the default class where it is utility or
/// background, and leave every other class as it is. On macOS a new thread takes the class of
/// the thread that started it, and each of the runtime's threads (the pool's, the scheduler's,
/// a command buffer's completion) serves the whole program for the device's life: one started
/// from a background thread would run every later making, scheduling and completion at
/// background, whoever waits on it. On an Apple M4 a pipeline making on a background thread took
/// 6.2 times as long through MoltenVK as at the default class (EV TTG-qos-m4). Nothing elsewhere,
/// where there is no such class.
///
/// The floor covers only the threads the runtime starts. The pool's threads run at the
/// user-initiated class ([`raise_to_user_initiated`]), and a pipeline the runtime makes on a
/// program's thread below that class is made on a thread of the runtime's at it while the
/// program's thread waits (`device/compiles.rs`). The threads that take command buffers to their
/// handlers run at the user-interactive class ([`spawn_user_interactive`]).
#[cfg(target_os = "macos")]
pub(crate) fn raise_quality_of_service() {
    use libc::qos_class_t as Q;
    let mut class = Q::QOS_CLASS_UNSPECIFIED;
    let mut relative = 0;
    // SAFETY: the calling thread's own handle, and two out-parameters that live for the call.
    let read =
        unsafe { libc::pthread_get_qos_class_np(libc::pthread_self(), &mut class, &mut relative) };
    if read == 0 && matches!(class, Q::QOS_CLASS_UTILITY | Q::QOS_CLASS_BACKGROUND) {
        // SAFETY: sets the calling thread's own class; no pointer.
        unsafe { libc::pthread_set_qos_class_self_np(Q::QOS_CLASS_DEFAULT, 0) };
    }
}

/// Nothing: no quality of service class here ([`raise_quality_of_service`] on macOS).
#[cfg(not(target_os = "macos"))]
pub(crate) fn raise_quality_of_service() {}

/// Raise this thread's quality of service to the user-initiated class where it is lower or
/// unspecified, and leave the user-interactive class as it is: a thread of the runtime's own
/// that issues the driver's compiles (the pool's, `pipeline/pool.rs`; a compile's own,
/// `device/compiles.rs`). MoltenVK runs each
/// compile on a dispatch queue that takes the class of the thread asking, so the class here is the
/// compile's class in Metal's compiler service.
#[cfg(target_os = "macos")]
pub(crate) fn raise_to_user_initiated() {
    if !at_least_user_initiated() {
        // SAFETY: sets the calling thread's own class; no pointer.
        unsafe {
            libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_USER_INITIATED, 0)
        };
    }
}

/// Nothing: no quality of service class here ([`raise_to_user_initiated`] on macOS).
#[cfg(not(target_os = "macos"))]
pub(crate) fn raise_to_user_initiated() {}

/// Run this thread at the user-interactive quality of service, whatever its class: a thread of
/// the runtime's own that takes command buffers to their handlers ([`spawn_user_interactive`]).
#[cfg(target_os = "macos")]
fn run_at_user_interactive() {
    // Metal runs every command buffer's scheduled and completed handlers at the user-interactive
    // class, whatever the class of the thread that committed it: measured on an M3 Max, macOS
    // 27.0.1, committing from threads at each class from background to user-interactive. So the
    // threads that bring a command buffer to its handlers, and run them, run there too. It is a
    // request of the documented `pthread_set_qos_class_self_np` (`<pthread/qos.h>`): the system
    // still places the thread, which is bound to no CPU, and no option or variable turns it off.
    // SAFETY: sets the calling thread's own class; no pointer.
    unsafe {
        libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_USER_INTERACTIVE, 0)
    };
}

/// Ask the scheduler to treat this thread as fully busy whenever it runs: a thread of the
/// runtime's own that takes command buffers to their handlers ([`spawn_user_interactive`]), as
/// macOS runs them at the user-interactive class.
#[cfg(target_os = "linux")]
fn run_at_user_interactive() {
    // `struct sched_attr` of `sched_setattr(2)` (`<linux/sched/types.h>`), its 56-byte version,
    // the first with the utilization clamps.
    #[repr(C)]
    struct SchedAttr {
        size: u32,
        sched_policy: u32,
        sched_flags: u64,
        sched_nice: i32,
        sched_priority: u32,
        sched_runtime: u64,
        sched_deadline: u64,
        sched_period: u64,
        sched_util_min: u32,
        sched_util_max: u32,
    }
    // `SCHED_FLAG_KEEP_POLICY`, `SCHED_FLAG_KEEP_PARAMS` and `SCHED_FLAG_UTIL_CLAMP_MIN`
    // (`<linux/sched.h>`): only the utilization floor changes, not the policy, nice or priority.
    const FLAGS: u64 = 0x08 | 0x10 | 0x20;
    let attr = SchedAttr {
        size: std::mem::size_of::<SchedAttr>() as u32,
        sched_policy: 0,
        sched_flags: FLAGS,
        sched_nice: 0,
        sched_priority: 0,
        sched_runtime: 0,
        sched_deadline: 0,
        sched_period: 0,
        // The top of the scale (`SCHED_CAPACITY_SCALE`, Documentation/scheduler/sched-util-clamp):
        // where the kernel uses the floor (capacity-aware placement, the schedutil governor) the
        // thread is placed and clocked as if fully busy. No CPU affinity: the system still places
        // it, and no option or variable turns this off. An unprivileged thread may raise its own
        // floor; a kernel without the clamps refuses it, and the thread runs as before.
        sched_util_min: 1024,
        sched_util_max: 0,
    };
    // SAFETY: `sched_setattr(0, &attr, 0)` sets the calling thread's own attributes from `attr`,
    // which lives for the call and whose `size` is its own.
    let _ = unsafe { libc::syscall(libc::SYS_sched_setattr, 0, &attr as *const SchedAttr, 0) };
}

/// Nothing: no such hint here ([`run_at_user_interactive`] on macOS and Linux).
#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn run_at_user_interactive() {}

/// Whether this thread runs at the user-initiated quality of service or above: its class is
/// user-initiated or user-interactive. Always where there is no such class.
#[cfg(target_os = "macos")]
pub(crate) fn at_least_user_initiated() -> bool {
    use libc::qos_class_t as Q;
    let mut class = Q::QOS_CLASS_UNSPECIFIED;
    let mut relative = 0;
    // SAFETY: the calling thread's own handle, and two out-parameters that live for the call.
    let read =
        unsafe { libc::pthread_get_qos_class_np(libc::pthread_self(), &mut class, &mut relative) };
    read == 0
        && matches!(
            class,
            Q::QOS_CLASS_USER_INITIATED | Q::QOS_CLASS_USER_INTERACTIVE
        )
}

/// Always: no quality of service class here ([`at_least_user_initiated`] on macOS).
#[cfg(not(target_os = "macos"))]
pub(crate) fn at_least_user_initiated() -> bool {
    true
}

/// Run the program's code `f` (a handler, a notification, a deallocator, a message sink) on
/// this thread, catching a panic: what it calls is the program's, as on the program's own
/// threads. Every call of the program's code goes through here.
pub(crate) fn program(f: impl FnOnce()) {
    PROGRAM.with(|program| program.set(program.get() + 1));
    let base = BASE.with(|base| base.replace(OWN.with(Cell::get)));
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
    BASE.with(|outer| outer.set(base));
    PROGRAM.with(|program| program.set(program.get() - 1));
}

/// Whether this thread is in one of the runtime's sections now, not in the program's code a
/// section called (a message sink): the device's functions are then the table's alone
/// ([`Call`], `device/functions.rs`).
pub(crate) fn in_section() -> bool {
    OWN.with(Cell::get) > BASE.with(Cell::get)
}

/// Whether this thread may reach the device's functions now: the program's threads always (the
/// program's calls are its own race with an exit), the runtime's own only inside a section or
/// while they run the program's code.
pub(crate) fn may_call() -> bool {
    !RUNTIME.with(Cell::get) || OWN.with(Cell::get) > 0 || PROGRAM.with(Cell::get) > 0
}

/// In builds with debug assertions (every test run), end the process when this thread may not
/// reach the device's functions now ([`may_call`]): a Vulkan call of a runtime thread outside a
/// section, which an exit would not wait for.
pub(crate) fn check() {
    if cfg!(debug_assertions) && !may_call() {
        let name = std::thread::current().name().map(str::to_owned);
        eprintln!(
            "metal2vulkan: the runtime's thread {} reached the device outside an exit-guard \
             section",
            name.as_deref().unwrap_or("(unnamed)")
        );
        std::process::abort();
    }
}

/// The devices the hook was registered at since the process last loaded a shared object: that
/// count of loads ([`loaded::count`]), and the devices' identities.
pub(crate) struct Devices {
    loads: Option<u64>,
    identities: Vec<Vec<u64>>,
}

impl Devices {
    /// None yet.
    pub(crate) const fn new() -> Devices {
        Devices {
            loads: None,
            identities: Vec::new(),
        }
    }

    /// Whether a device of `identity` registers the hook ([`register_device`]), the process
    /// having loaded `loads` shared objects, and the hook being registered in this process
    /// (`registered`), and why; noted when it does. The identities kept are those registered
    /// since the last load, so they stay as many as the distinct devices of one load.
    pub(crate) fn admit(
        &mut self,
        registered: bool,
        loads: Option<u64>,
        identity: Vec<u64>,
    ) -> Option<Admitted> {
        let current = registered && loads.is_some() && self.loads == loads;
        if current && self.identities.contains(&identity) {
            return None;
        }
        if !current {
            self.identities.clear();
        }
        self.loads = loads;
        self.identities.push(identity);
        Some(match current {
            true => Admitted::Identity,
            false => Admitted::Load,
        })
    }

    /// The identities kept.
    #[cfg(test)]
    pub(crate) fn kept(&self) -> usize {
        self.identities.len()
    }
}

/// Why a device registers the hook ([`Devices::admit`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Admitted {
    /// The process's first device (a forked child's included), the first after a load of a
    /// shared object, or any where the host keeps no count of loads: what was loaded since makes
    /// its statics at each entry point's first call again, so the entry points called before are
    /// forgotten ([`CALLED`]).
    Load,
    /// A device of an identity new since the last load.
    Identity,
}

/// The process's [`Devices`].
static DEVICES: Mutex<Devices> = Mutex::new(Devices::new());

/// The hook's registrations in this process: at devices ([`register_device`]), and in all.
static AT_DEVICES: AtomicU64 = AtomicU64::new(0);
static INSTALLED: AtomicU64 = AtomicU64::new(0);

/// Register the hook at a device made or adopted, once its driver and layers are loaded and
/// primed (`device/prime.rs`), when the device can have made statics after the hook's last
/// registration: the process's first device (a forked child's included); a device whose
/// `identity` (its driver's and device's UUIDs and the entry points of the table its functions
/// reach, the topmost layer's where layers run) no device registered at since the last load of
/// a shared object; and the first device after such a load, since a driver or layer loaded again
/// makes its statics again. Devices made again on what stays loaded register nothing, so a
/// process that makes devices over and over adds a registration per load, not per device. Where
/// the host keeps no count of loads, every device registers.
pub(crate) fn register_device(identity: Vec<u64>) {
    let loads = loaded::count();
    let mut devices = DEVICES.lock().unwrap_or_else(PoisonError::into_inner);
    let registered = REGISTERED.load(Ordering::SeqCst) == std::process::id();
    let Some(admitted) = devices.admit(registered, loads, identity) else {
        return;
    };
    // A layer the loader unloaded with the last instance and loaded again with this one makes
    // its statics again at the first call of each entry point, after this registration: each
    // such call's section registers the hook again as it ends.
    if admitted == Admitted::Load {
        forget_calls();
    }
    AT_DEVICES.fetch_add(1, Ordering::Relaxed);
    register();
}

/// Forget the entry points called so far ([`CALLED`]): the next section that calls each
/// registers the hook again as it ends.
pub(crate) fn forget_calls() {
    CALLED.clear();
    LOADS.fetch_add(1, Ordering::Relaxed);
}

/// The times [`forget_calls`] ran (the loads devices were made after): each lets the sections
/// register the hook once per entry point of the table.
static LOADS: AtomicU64 = AtomicU64::new(0);

/// How many times this process registered the exit guard's hook, for the lanes: not part of the
/// API.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Registrations {
    /// At devices made or adopted.
    pub at_devices: u64,
    /// In all: at devices, and at the end of sections (after the first call of an entry point
    /// of the table, and at powers of two of the sections' count).
    pub total: u64,
    /// The most the sections can make: one per entry point of the table at each load a device
    /// was made after, the process's first device's included (It was one per entry point), and 64.
    pub section_bound: u64,
}

/// This process's [`Registrations`] so far.
pub fn registrations() -> Registrations {
    Registrations {
        at_devices: AT_DEVICES.load(Ordering::Relaxed),
        total: INSTALLED.load(Ordering::Relaxed),
        section_bound: Call::ALL.len() as u64 * LOADS.load(Ordering::Relaxed).max(1) + 64,
    }
}

/// Register the hook in this process. A child forked from a process that registered registers
/// its own, and counts none of its parent's sections, whose threads it does not have.
pub(crate) fn register() {
    let pid = std::process::id();
    let seen = REGISTERED.load(Ordering::SeqCst);
    if seen == pid {
        install();
        return;
    }
    if REGISTERED
        .compare_exchange(seen, pid, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        register_again();
        return;
    }
    if seen != 0 {
        GUARD.active.store(OWN.with(Cell::get), Ordering::SeqCst);
    }
    install();
}

/// Register the hook again in the process it is registered in: after the destructors registered
/// since. A forked child that has not registered its own does nothing.
fn register_again() {
    if REGISTERED.load(Ordering::SeqCst) == std::process::id() {
        install();
    }
}

/// Register [`on_exit`] with the C library's exit handlers.
#[cfg(unix)]
fn install() {
    INSTALLED.fetch_add(1, Ordering::Relaxed);
    // SAFETY: `on_exit` is a function with no arguments that never unwinds. Registration fails
    // only when the C library is out of memory; the process then exits unguarded, as before.
    unsafe { libc::atexit(on_exit) };
}

/// Nothing to register: the runtime is built for unix hosts, whose exit order the hook relies on.
#[cfg(not(unix))]
fn install() {}

/// The exit handler.
#[cfg(unix)]
extern "C" fn on_exit() {
    hook(
        &GUARD,
        &IN_PROCESS,
        REGISTERED.load(Ordering::SeqCst),
        std::process::id(),
        OWN.with(Cell::get),
        BOUND,
    );
}

/// The exit handler [`install`] registers, run alone, as the C library's `exit` runs it: a forked
/// child's test runs it without the other handlers the child inherits.
#[cfg(all(test, unix))]
pub(crate) fn run_on_exit() {
    on_exit();
}

/// The hook's work in process `pid`, registered in process `registered`, on a thread with `own`
/// sections in progress: close `guard` and wait for the other sections, then for the work
/// submitted to `in_process`'s devices, at most `bound` in all; where a section that makes a
/// `VkPipeline` held the sections' wait past `bound` ([`Guard::close`]), the work's wait gets
/// `bound` from there. `None` (nothing done) in another process than the one registered; else
/// whether the other sections ended and the work completed.
pub(crate) fn hook(
    guard: &Guard,
    in_process: &InProcessDevices,
    registered: u32,
    pid: u32,
    own: usize,
    bound: Duration,
) -> Option<bool> {
    (registered == pid).then(|| {
        let deadline = Instant::now() + bound;
        let ended = guard.close(own, bound);
        // A making's section held the close past the bound: the work gets the bound from there.
        let now = Instant::now();
        let deadline = match now > deadline {
            true => now + bound,
            false => deadline,
        };
        ended && in_process.drain(pid, deadline)
    })
}

/// The sections in progress on this thread.
#[cfg(test)]
pub(crate) fn own() -> usize {
    OWN.with(Cell::get)
}

/// The sections of the process's guard that ended.
#[cfg(test)]
pub(crate) fn sections() -> u64 {
    SECTIONS.load(Ordering::Relaxed)
}

/// Whether a section that ended called `call`.
#[cfg(test)]
pub(crate) fn was_called(call: Call) -> bool {
    CALLED.load() & call.bit() != 0
}
