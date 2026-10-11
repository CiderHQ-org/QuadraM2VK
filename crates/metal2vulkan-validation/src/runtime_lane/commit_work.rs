//! Device tests of the commit path's work on the device the cfg selects, under the validation
//! layer: each holds a piece of work the runtime once repeated at every commit, or at every change
//! of its live buffers, to a bound, through the device's work counts (`Device::work_counts`, not
//! part of the API). Deterministic: they count, and time nothing.
//!
//! - **Command pools.** A command buffer that encoded nothing records its submission's barriers
//!   into the spare command pool the one before gave back; before, it made one in the
//!   scheduler's section, where no spare is taken.
//! - **Fences.** Command buffers that each end in one submission complete at their fences where
//!   the driver reads their objects after their values (MoltenVK): the completion thread waits
//!   for no timeline value; before, it waited for one each, which MoltenVK wakes through a shared
//!   event's listener. Elsewhere they end at their values, and the device makes no fence for
//!   them; before, each commit could make one, which NVIDIA's driver held for up to 1.9 s while
//!   other processes' work ran.
//! - **The live buffers.** A buffer made or released between commits costs the next resident set
//!   no ask of a listed buffer and no new buffer to upload it to; before, each asked every
//!   listed buffer for its pages again and uploaded the set to a new buffer.
//! - **Command pools, kept and reused**. Fills in rounds of more command buffers at once than any
//!   earlier round, released as they complete: the command pools made stay bounded by the peak in
//!   flight, each given back is kept and reused by a later round, and the fills read correctly.
//!   Previously a pool given back past the spares (or that made more than 64 command buffers) was
//!   destroyed as its command buffer completed, where NVIDIA 615.71.09's `vkDestroyCommandPool`
//!   intermittently never returned.
//! - **Addressable buffers**. The program's buffers take their device addresses from their memory
//!   blocks, so making them makes no buffer with the device-address usage beyond a new block's;
//!   before, each was one, which MoltenVK made resident one by one in every encoder whose pipeline
//!   reads pointers (2.5 ms a commit at 32768 live buffers on an Apple M4).

use super::degrade_notices::on_device;
use super::{bound_buffers, encode_case, pipeline_state, public_case, CaseResources, WAIT};
use metal2vulkan::error::RuntimeDriverDefect;
use metal2vulkan_runtime::{
    Buffer, CommandBuffer, CommandBufferStatus, CommandQueue, ResourceOptions,
};

/// A kernel whose root table searches the device's live buffers: its pointer walk reaches
/// buffers no encoder named.
const POINTER_WALKER: &str = "walk-a-device-pointer-chain-through-a-loop-phi";

/// Commit `command_buffer` and wait for it to complete.
fn finish(command_buffer: &CommandBuffer) {
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
}

/// A command buffer of a blit that fills `buffer`.
fn fill(queue: &CommandQueue, buffer: &Buffer) -> CommandBuffer {
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.fill_buffer(buffer, 0..buffer.length(), 0x5a).unwrap();
    blit.end_encoding().unwrap();
    command_buffer
}

/// Empty command buffers one after another make no command pool past the first: each takes the
/// spare the one before gave back as it completed.
#[test]
fn empty_command_buffers_take_the_spare_command_pool() {
    on_device(|device, _| {
        let queue = device.new_command_queue().unwrap();
        // The first may make the pool the later ones take.
        finish(&queue.command_buffer().unwrap());
        let before = device.work_counts();
        for _ in 0..8 {
            finish(&queue.command_buffer().unwrap());
        }
        let after = device.work_counts();
        assert_eq!(after.command_pools_made, before.command_pools_made);
    });
}

/// Rounds of 32 command buffers encoded, then committed, together, each a fill of its own
/// buffer, released as they complete: the runtime keeps every command pool given back while the
/// device lives and reuses it, so across the rounds the command pools made stay bounded by the peak
/// in flight (at most one round's), and every round's fills still read correctly. Previously a pool
/// given back past the device's spares was destroyed as its command buffer completed, where NVIDIA
/// 615.71.09's `vkDestroyCommandPool` intermittently never returned (the consumer's hang); and
/// MoltenVK 1.4.1 fills through `vkCmdUpdateBuffer` (`word_fill.rs`), whose scratch buffer a
/// completed handler gives back to its command pool's pool, so a program that destroyed its pool at
/// that value died (`mutex lock failed`).
#[test]
fn fills_across_rounds_reuse_a_bounded_set_of_command_pools() {
    const IN_FLIGHT: usize = 32;
    const ROUNDS: usize = 16;
    on_device(|device, _| {
        let queue = device.new_command_queue().unwrap();
        let buffers: Vec<Buffer> = (0..IN_FLIGHT)
            .map(|_| device.new_buffer(256, ResourceOptions::shared()).unwrap())
            .collect();
        let before = device.work_counts();
        for round in 0..ROUNDS {
            let byte = round as u8 + 1;
            // All encoded before any is committed, so each holds a command pool of its own.
            let committed: Vec<CommandBuffer> = buffers
                .iter()
                .map(|buffer| {
                    let command_buffer = queue.command_buffer().unwrap();
                    let mut blit = command_buffer.blit_command_encoder().unwrap();
                    blit.fill_buffer(buffer, 0..buffer.length(), byte).unwrap();
                    blit.end_encoding().unwrap();
                    command_buffer
                })
                .collect();
            for command_buffer in &committed {
                command_buffer.commit().unwrap();
            }
            for command_buffer in &committed {
                command_buffer.wait_until_completed_for(WAIT).unwrap();
                assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
            }
            drop(committed);
            for buffer in &buffers {
                let contents = buffer.contents().expect("a shared buffer");
                // SAFETY: the buffer holds `length` bytes at `contents`, alive with the buffer;
                // no command buffer uses it now.
                let bytes = unsafe {
                    std::slice::from_raw_parts(contents.as_ptr(), buffer.length() as usize)
                };
                assert!(bytes.iter().all(|&b| b == byte), "round {round}");
            }
        }
        let after = device.work_counts();
        let made = (after.command_pools_made - before.command_pools_made) as usize;
        let by_transfers = after.fills_by_transfers - before.fills_by_transfers;
        eprintln!(
            "FILL-POOL-REUSE {}: {made} command pools made for {} command buffers over {ROUNDS} \
             rounds, {by_transfers} fills by transfers",
            device.name(),
            ROUNDS * IN_FLIGHT
        );
        // The pools made are bounded by the peak in flight and do not grow with the rounds: each
        // given back is kept and reused, not destroyed and remade. Before the fix this was about
        // `ROUNDS * (IN_FLIGHT - 8)` as pools past the spares were destroyed each round.
        assert!(
            made <= IN_FLIGHT,
            "{made} command pools made, more than the peak in flight"
        );
    });
}

/// Command buffers that each end in one submission, empty or of a blit, complete at their
/// fences where the driver reads a command buffer's objects after its value
/// (`RuntimeDriverDefect::ReadsCommandBuffersAfterTheirSignal`): the completion thread waits for
/// no timeline value. Elsewhere they end at their values alone: the device makes no fence for
/// any of them, from its creation on, and the completion thread waits for their values.
#[test]
fn command_buffers_end_at_fences_only_where_the_driver_reads_them_later() {
    on_device(|device, _| {
        let queue = device.new_command_queue().unwrap();
        let buffer = device.new_buffer(4096, ResourceOptions::default()).unwrap();
        let before = device.work_counts();
        for _ in 0..4 {
            finish(&queue.command_buffer().unwrap());
            finish(&fill(&queue, &buffer));
        }
        let after = device.work_counts();
        let late_reads = (device.properties().driver_defects)
            .contains(&RuntimeDriverDefect::ReadsCommandBuffersAfterTheirSignal);
        eprintln!(
            "COMMAND-BUFFER-ENDS {}: reads after the value {late_reads}, {} fences made, {} \
             timeline waits for 8 command buffers",
            device.name(),
            after.fences_made,
            after.timeline_waits - before.timeline_waits
        );
        if late_reads {
            assert_eq!(after.timeline_waits, before.timeline_waits);
        } else {
            assert_eq!(after.fences_made, 0);
            assert!(after.timeline_waits > before.timeline_waits);
        }
    });
}

/// With 64 buffers listed, a buffer released and another made between each pointer walk's
/// commit puts the walk's resident set into the buffer of the set it replaces.
#[test]
fn a_buffer_made_or_released_costs_the_live_buffers_no_upload() {
    on_device(|device, _| {
        let queue = device.new_command_queue().unwrap();
        let case = public_case(POINTER_WALKER).unwrap();
        let state = pipeline_state(device, &case).unwrap();
        let resources = CaseResources::new(device, &case).unwrap();
        let bound = bound_buffers(device, &case.compute, &resources).unwrap();
        let walk = || {
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            encode_case(&mut encoder, &state, &case.compute, &bound, &resources).unwrap();
            encoder.end_encoding().unwrap();
            finish(&command_buffer);
        };
        let new_buffer = || device.new_buffer(4096, ResourceOptions::default()).unwrap();
        let mut listed: Vec<Buffer> = (0..64).map(|_| new_buffer()).collect();
        // The first set, uploaded to a buffer made for it.
        walk();
        let before = device.work_counts();
        for _ in 0..8 {
            listed.remove(0);
            listed.push(new_buffer());
            walk();
        }
        let after = device.work_counts();
        assert_eq!(after.live_tables_uploaded, before.live_tables_uploaded);
        drop(listed);
    });
}

/// 64 `Shared` and 64 `Private` buffers of the program's make at most one buffer with the
/// device-address usage per memory class they or the runtime's own buffers took a new block of:
/// each takes its address from its block's buffer of addresses (`memory.rs`).
#[test]
fn buffers_made_take_their_addresses_from_their_blocks() {
    on_device(|device, _| {
        let before = device.work_counts();
        let made: Vec<Buffer> = [ResourceOptions::shared(), ResourceOptions::private()]
            .into_iter()
            .flat_map(|options| (0..64).map(move |_| options))
            .map(|options| device.new_buffer(256, options).unwrap())
            .collect();
        let after = device.work_counts();
        let addressable = after.addressable_buffers_made - before.addressable_buffers_made;
        assert!(
            addressable <= 4,
            "{} buffers made {addressable} buffers with the device-address usage",
            made.len()
        );
    });
}
