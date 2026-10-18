//! A no-copy buffer no command used, released while a command buffer whose pipeline reads device
//! addresses waits behind a long one, fails neither, on the device the cfg selects, under the
//! validation layer. Vulkan lets the program free memory no submitted command refers to; MoltenVK
//! 1.4.1 makes every buffer of device-address usage resident in such a command buffer's encoders
//! and lets go of it as its memory is freed, which failed the one behind (`Invalid Resource`,
//! Metal's code 9, then the device lost) in the first round, every time, before the runtime waited.
//! Where the device's driver has that defect
//! (`RuntimeDriverDefect::FreedAddressMemoryFailsPendingWork`), the import's memory and its
//! deallocator wait for the batches submitted before the release
//! (`WorkCounts::freed_memory_parked`), announced once (`host-deallocator-delayed`); elsewhere
//! they go at once.

use super::degrade_notices::{assert_made_once, on_device};
use super::host_memory::Seen;
use super::rebind_reach::{buffer_len, buffer_over, deallocated, node, pages, C, POINTER_WALKER};
use super::resource_ids::DELAYED_DEALLOCATORS;
use super::{
    bound_buffers, encode_case, pipeline_state, public_case, shared_kinds, CaseResources, WAIT,
};
use metal2vulkan::error::RuntimeDriverDefect;
use metal2vulkan_runtime::ResourceOptions;
use std::sync::Arc;
use std::time::Instant;

/// The rounds, each a release behind a long command buffer.
const ROUNDS: u32 = 4;

/// The long command buffer: [`FILLS`] fills of [`FILLED`] bytes.
const FILLS: u8 = 32;
const FILLED: u64 = 64 << 20;

/// Each round: a no-copy buffer no command uses, then a command buffer of fills and one behind it
/// that runs the pointer walker, both committed; once the one behind was submitted, the buffer is
/// released. Both complete without error, and the buffer's deallocator runs; where the device's
/// driver has the defect, a release while the fills ran waited for them, and none waits
/// elsewhere.
#[test]
fn a_buffer_released_behind_work_that_reads_device_addresses_fails_none_of_it() {
    on_device(|device, messages| {
        let queue = device.new_command_queue().unwrap();
        let walker = public_case(POINTER_WALKER).unwrap();
        let state = pipeline_state(device, &walker).unwrap();
        let holding = CaseResources::new(device, &walker).unwrap();
        let buffers = bound_buffers(device, &walker.compute, &holding).unwrap();
        let big = device
            .new_buffer(FILLED, ResourceOptions::private())
            .unwrap();
        let parks = (device.properties().driver_defects)
            .contains(&RuntimeDriverDefect::FreedAddressMemoryFailsPendingWork);
        let mut waited = 0;
        for round in 0..ROUNDS {
            let seen = Arc::new(Seen::default());
            let Some(unused) = buffer_over(device, pages(buffer_len(), 0, &node(C)), &seen) else {
                return;
            };
            let long = queue.command_buffer().unwrap();
            let mut blit = long.blit_command_encoder().unwrap();
            for value in 0..FILLS {
                blit.fill_buffer(&big, 0..FILLED, value).unwrap();
            }
            blit.end_encoding().unwrap();
            let behind = queue.command_buffer().unwrap();
            let mut encoder = behind.compute_command_encoder().unwrap();
            encode_case(&mut encoder, &state, &walker.compute, &buffers, &holding).unwrap();
            encoder.end_encoding().unwrap();
            long.commit().unwrap();
            behind.commit().unwrap();
            let deadline = Instant::now() + WAIT;
            // Scheduled once the CPU finished scheduling it (`kernelEndTime`): submitted, or,
            // under `CompletionPolicy::Grouped` on a driver that merges submissions, held for one
            // with later batches; a driver with the defect (MoltenVK) merges none. Its kernel
            // start is stored as scheduling begins, before its batch is replayed and submitted.
            while behind.kernel_end_time() == 0.0 {
                assert!(
                    Instant::now() < deadline,
                    "the walker's batch was never submitted"
                );
                std::thread::yield_now();
            }
            let before = device.work_counts().freed_memory_parked;
            drop(unused);
            let parked = device.work_counts().freed_memory_parked - before;
            assert!(
                parked <= u64::from(parks),
                "round {round}: {parked} blocks waited (the device's driver has the defect: {parks})"
            );
            waited += parked;
            long.wait_until_completed_for(WAIT).unwrap();
            behind.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(long.error(), None, "round {round}: the fills");
            assert_eq!(behind.error(), None, "round {round}: the walker");
            deallocated(&seen);
        }
        eprintln!("FREED-MEMORY: {waited} of {ROUNDS} releases waited (defect: {parks})");
        // The fills outlast the release where the device runs them: in some round, at least.
        assert!(!parks || waited > 0, "no release waited for the fills");
        let kinds: &[_] = match waited {
            0 => &[],
            _ => &DELAYED_DEALLOCATORS,
        };
        assert_made_once(messages, kinds, &shared_kinds(device));
    });
}
