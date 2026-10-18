//! Vulkan command recording off the program's encoding thread: the encoders record into deferred
//! command buffers, whose commands a log keeps, and the device's recorder thread replays each log
//! into a `VkCommandBuffer` once its command buffer committed, before its submission.
//!
//! - **One door.** The device's functions hold this module's entry point in place of the driver's
//!   for each command a `VkCommandBuffer` takes ([`Command`]), its begin and its end among them. A
//!   device the runtime makes or adopts takes one of [`SLOTS`] slots ([`route`]): its functions
//!   (`device/functions.rs`, the exit guard's table among them) and the extension commands it loads
//!   ([`Slot::route`]) hold the slot's entry points. Each is generic over its slot, so it finds its
//!   device's command with one load, without a lock or a lookup by handle; a slot is let go of
//!   with its device, after `vkDestroyDevice`. A source test fails when a file outside this module
//!   names a command that is not routed.
//! - **Deferred command buffers.** An encoder's command buffers (`CommandBuffer::allocate_raw`
//!   and `allocate_secondary`) are handles of this module's (`handles.rs`), never a driver's: an
//!   entry point called with one logs the command, its arguments copied at the call (`log.rs`), and
//!   calls the driver's with any other. So everything Metal makes observable at encode time stays
//!   at the call, on the program's thread: the checks, refusals and their errors, the copies of
//!   inline bytes and of the commands' arguments, descriptor writes, object and pipeline making,
//!   retention, notices, the tracker's uses. Only the driver's recording moves, and Metal reads
//!   buffer contents as the GPU runs the commands, not as they are encoded, so what the commands
//!   do is the same. A device without a slot, or once every handle is live, records into the
//!   driver's command buffers as they are encoded.
//! - **Replayed once committed** (`replays.rs`). A command buffer's ended logs replay, in the order
//!   they ended (a secondary before the primary that executes it, Vulkan's executable state), into
//!   command buffers of a pool of its own (a spare, `spares.rs`), never shared with the scheduler's
//!   barrier command buffers: on the recorder thread, which its commit wakes, or on the thread
//!   that submits it where the recorder has not yet reached it, which then replays it itself
//!   ([`resolve`]), so a submission never waits for a wake. A log of a command buffer never
//!   committed is never replayed, nor one never ended: a render pass instance's secondary that
//!   nothing will execute is left unended (`CommandBuffer::give_up_raw`).
//! - **What the logs name lives through the replay.** Every object a command names stays live
//!   until its command buffer completes, but for the host pages a command buffer holds only weakly
//!   from its commit to its first submission (`rebind.rs`): its commit hands their holds to its
//!   replays, which let go of them once every ended log replayed. A release, rebind or detach of
//!   pages the replays hold replays first, on its own thread, what the recorder has yet to
//!   ([`RecorderShared::settle_all`]), and a commit whose pages' buffer let go of them (released
//!   or detached while an encoder took them, on any thread) replays its command buffer itself,
//!   so the pages' deallocator runs in the call it runs in without a recorder. Each marks its
//!   side, then reads the other's, so a let-go and a commit that race agree on which replays
//!   (`HostPages::let_go_by_buffer`). Its release (completion, failure, drop) lets go of the
//!   rest, after any replay in progress, replaying nothing more.
//! - **Counted by kind.** Each command a replay calls counts once for its kind, one relaxed
//!   increment into a fixed array of the device's indexed by the command
//!   (`WorkCounts::commands_replayed`).
//! - **A replay's failure** (`vkBeginCommandBuffer` or `vkEndCommandBuffer` out of memory) fails
//!   the submission of what it replayed, as a recording error.
//! - **In sections.** The recorder thread is one of the runtime's: each replay runs inside a
//!   section of the exit guard, and every entry point it calls is the guard's table's, marked as
//!   called as it is ([`exit_guard::called`]), so the first replay of a command the device's
//!   priming could not make registers the exit hook again.

mod handles;
mod log;
mod replays;
mod thunks;

use crate::device::functions::Functions;
use crate::exit_guard;
use crate::loader_word::LoaderWord;
use ash::vk;
use std::ffi::{c_void, CStr};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

pub(crate) use handles::is_deferred;
#[cfg(test)]
pub(crate) use replays::{queue_room, take_leaving_room};
pub(crate) use replays::{resolve, RecorderShared, ReplayCounts, Replays};
pub(crate) use thunks::Command;
pub use thunks::CommandsReplayed;

/// How many devices route their commands here at once; one made past them keeps its driver's.
pub(crate) const SLOTS: usize = 8;

/// The slots taken.
static TAKEN: [AtomicBool; SLOTS] = [const { AtomicBool::new(false) }; SLOTS];

/// Run `$body` with `$s` the const slot index `$index` names.
macro_rules! per_slot {
    ($index:expr, $s:ident => $body:expr) => {
        match $index {
            0 => {
                const $s: usize = 0;
                $body
            }
            1 => {
                const $s: usize = 1;
                $body
            }
            2 => {
                const $s: usize = 2;
                $body
            }
            3 => {
                const $s: usize = 3;
                $body
            }
            4 => {
                const $s: usize = 4;
                $body
            }
            5 => {
                const $s: usize = 5;
                $body
            }
            6 => {
                const $s: usize = 6;
                $body
            }
            _ => {
                const $s: usize = 7;
                $body
            }
        }
    };
}

/// One device's slot (the module docs): its commands, until it is dropped, after the device is
/// destroyed.
pub(crate) struct Slot {
    index: usize,
}

impl Slot {
    /// `real`, the device's command `command` as its driver gave it, routed through the slot:
    /// the slot's entry point, which calls `real`.
    ///
    /// # Safety
    ///
    /// `real` is a function pointer of `command`'s signature, the slot's device's, and not one
    /// of the slot's entry points (a function's address is no identity to compare against, so
    /// the caller knows which it holds: a command of the device's routed functions is routed).
    pub(crate) unsafe fn route<F: Copy>(&self, command: Command, real: F) -> F {
        assert_eq!(
            std::mem::size_of::<F>(),
            std::mem::size_of::<*const c_void>()
        );
        // SAFETY: `F` is a function pointer (the caller's), of a pointer's size.
        let pointer = unsafe { std::mem::transmute_copy::<F, *const c_void>(&real) };
        thunks::store(self.index, command, pointer);
        let thunk = per_slot!(self.index, S => thunks::thunk::<S>(command));
        // SAFETY: the entry point has `command`'s signature, `F`'s (the caller's).
        unsafe { std::mem::transmute_copy::<*const c_void, F>(&thunk) }
    }

    /// The slot's entry point in place of the driver's `name`, if it routes that command: the exit
    /// guard's table (`device/functions.rs`) holds it too.
    pub(crate) fn entry(&self, name: &CStr) -> Option<*const c_void> {
        let command = Command::ALL.iter().find(|command| command.name() == name)?;
        Some(per_slot!(self.index, S => thunks::thunk::<S>(*command)))
    }
}

/// A device's recorder thread and what its replays share (the module docs).
pub(crate) struct Recorder {
    shared: Arc<RecorderShared>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Recorder {
    /// Start the recorder of the device of `slot`, whose functions are `functions` and loader
    /// word `loader_word`; `None` where the host starts no thread (the device records as its
    /// encoders encode). The thread holds the functions, which it ends before the device's
    /// teardown goes on (its drop joins it); its replays borrow them.
    pub(crate) fn start(
        slot: &Slot,
        functions: &Functions,
        loader_word: LoaderWord,
    ) -> Option<Recorder> {
        let shared = RecorderShared::new(slot.index, loader_word);
        let running = Arc::clone(&shared);
        let functions = functions.clone();
        let builder = std::thread::Builder::new().name("metal2vulkan-recorder".to_owned());
        let thread =
            exit_guard::spawn_user_interactive(builder, move || running.run(&functions)).ok()?;
        Some(Recorder {
            shared,
            thread: Some(thread),
        })
    }

    /// What its replays share.
    pub(crate) fn shared(&self) -> &Arc<RecorderShared> {
        &self.shared
    }
}

impl Drop for Recorder {
    fn drop(&mut self) {
        self.shared.stop();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl Drop for Slot {
    fn drop(&mut self) {
        thunks::clear(self.index);
        TAKEN[self.index].store(false, Ordering::Release);
    }
}

/// `device`'s functions with the entry points of a slot of its own in place of its driver's
/// commands, and the slot; the functions as they are, and no slot, where every slot is taken.
///
/// # Safety
///
/// `device` is live, and its functions are its driver's (or its topmost layer's, or routed for
/// its pipeline makings, `compiles.rs`), not already routed here.
pub(crate) unsafe fn route(device: ash::Device) -> (ash::Device, Option<Slot>) {
    let Some(index) = TAKEN.iter().position(|taken| {
        taken
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
            .is_ok()
    }) else {
        return (device, None);
    };
    let slot = Slot { index };
    let mut v1_0 = device.fp_v1_0().clone();
    let mut v1_3 = device.fp_v1_3().clone();
    per_slot!(index, S => thunks::route_tables::<S>(&mut v1_0, &mut v1_3));
    let routed = ash::Device::from_parts_1_3(
        device.handle(),
        v1_0,
        device.fp_v1_1().clone(),
        device.fp_v1_2().clone(),
        v1_3,
    );
    (routed, Some(slot))
}

/// `real`, a device's command `command`, through `slot` where the device has one
/// ([`Slot::route`]), else as it is.
///
/// # Safety
///
/// As [`Slot::route`].
pub(crate) unsafe fn through<F: Copy>(slot: Option<&Slot>, command: Command, real: F) -> F {
    match slot {
        // SAFETY: the caller's.
        Some(slot) => unsafe { slot.route(command, real) },
        None => real,
    }
}

/// Room in `raw`'s log, where it is a deferred command buffer, for the end a commit may fold into
/// it as its command buffer's tail (`CommandBufferShared::close_tail`): a barrier of one memory
/// barrier, then a timestamp. A log made while no spare one was kept (its earlier replays'
/// completions still to come) has no room in its lists until it takes it here, on the encoding
/// thread as its encoder ends, so the commit's logging grows no list whenever completions come.
///
/// # Safety
///
/// The caller records `raw` alone, and `raw` is recording.
pub(crate) unsafe fn reserve_end(raw: vk::CommandBuffer) {
    // SAFETY: the caller's.
    unsafe { handles::record(raw, |log| log.reserve_end()) };
}

/// The extension commands of `VK_EXT_mesh_shader` a draw records, routed (`render.rs`).
#[derive(Clone, Copy)]
pub(crate) struct MeshDraws {
    draw: vk::PFN_vkCmdDrawMeshTasksEXT,
    indirect: vk::PFN_vkCmdDrawMeshTasksIndirectEXT,
}

impl MeshDraws {
    /// The commands of `loaded`, through `slot` where the device has one.
    pub(crate) fn new(loaded: &ash::ext::mesh_shader::Device, slot: Option<&Slot>) -> MeshDraws {
        let functions = loaded.fp();
        let (draw, indirect) = (
            functions.cmd_draw_mesh_tasks_ext,
            functions.cmd_draw_mesh_tasks_indirect_ext,
        );
        match slot {
            // SAFETY: each is the device's command of that name.
            Some(slot) => unsafe {
                MeshDraws {
                    draw: slot.route(Command::CmdDrawMeshTasks, draw),
                    indirect: slot.route(Command::CmdDrawMeshTasksIndirect, indirect),
                }
            },
            None => MeshDraws { draw, indirect },
        }
    }

    /// `vkCmdDrawMeshTasksEXT`.
    ///
    /// # Safety
    ///
    /// As the command's.
    pub(crate) unsafe fn cmd_draw_mesh_tasks(
        &self,
        command_buffer: vk::CommandBuffer,
        group_count_x: u32,
        group_count_y: u32,
        group_count_z: u32,
    ) {
        // SAFETY: the caller's.
        unsafe { (self.draw)(command_buffer, group_count_x, group_count_y, group_count_z) }
    }

    /// `vkCmdDrawMeshTasksIndirectEXT`.
    ///
    /// # Safety
    ///
    /// As the command's.
    pub(crate) unsafe fn cmd_draw_mesh_tasks_indirect(
        &self,
        command_buffer: vk::CommandBuffer,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        draw_count: u32,
        stride: u32,
    ) {
        // SAFETY: the caller's.
        unsafe { (self.indirect)(command_buffer, buffer, offset, draw_count, stride) }
    }
}
