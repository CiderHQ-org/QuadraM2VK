//! Tracked order across command queues (the scheduler's `order.rs`) on the device the cfg selects,
//! under the validation layer. Metal tracks a resource across queues in the order it schedules the
//! command buffers, their commit order (its tracking across queues), whatever order they were made
//! in. Each test makes queue B's command buffer first and encodes its copy of a tracked buffer,
//! then commits queue A's write of the buffer and B's copy, in that order: B's copy holds A's
//! write.
//!
//! - **Submitted at its commit.** A's write is a blit fill.
//! - **Held for its host step.** A's write is an execution of commands a kernel of A wrote
//!   (`indirect_host_steps.rs`), behind fills of a ballast: its batch waits at its gate for A's
//!   earlier work to complete, and for its host step, while B commits.
//! - **Held for its submission check.** A's fill has a check that waits until B committed.

use super::degrade_notices::{assert_made_once, on_device};
use super::indirect_host_steps::{expected, Setup, HOST_STEPS};
use super::shared_kinds;
use super::{buffer_with, read_buffer, WAIT};
use metal2vulkan_runtime::{
    Buffer, CommandBuffer, CommandBufferStatus, CommandQueue, Device, ResourceOptions, Submission,
    SubmissionCheck,
};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// The bytes A's fill writes.
const FILLED: u8 = 0x5a;

/// The bytes of the buffers A writes and B copies.
const BYTES: u64 = 4096;

/// The ballast before the host step's execution, and how many times A fills it.
const BALLAST: u64 = 16 << 20;
const BALLAST_FILLS: usize = 8;

/// Queue B's command buffer, made first: a copy of `source` into a new buffer of 0xff bytes,
/// encoded before anything of queue A's. The copy's destination with it.
fn copy_first(queue: &CommandQueue, source: &Buffer) -> (CommandBuffer, Buffer) {
    let copy = buffer_with(queue.device(), &vec![0xff; source.length() as usize]).unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_buffer_to_buffer(source, 0, &copy, 0, source.length())
        .unwrap();
    blit.end_encoding().unwrap();
    (command_buffer, copy)
}

/// A fill of `buffer` with [`FILLED`] in `command_buffer`.
fn fill(command_buffer: &CommandBuffer, buffer: &Buffer) {
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.fill_buffer(buffer, 0..buffer.length(), FILLED)
        .unwrap();
    blit.end_encoding().unwrap();
}

/// Wait for each committed command buffer, and require it completed.
fn completed(command_buffers: &[&CommandBuffer]) {
    for command_buffer in command_buffers {
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    }
}

/// Two queues of `device`: A's and B's.
fn queues(device: &Device) -> (CommandQueue, CommandQueue) {
    (
        device.new_command_queue().unwrap(),
        device.new_command_queue().unwrap(),
    )
}

/// A's fill, submitted at its commit, comes before B's copy, committed after it: B's command
/// buffer was made and encoded first.
#[test]
fn a_copy_committed_after_a_write_on_another_queue_holds_it() {
    on_device(|device, _| {
        let (a, b) = queues(device);
        let written = buffer_with(device, &[0; BYTES as usize]).unwrap();
        let (reader, copy) = copy_first(&b, &written);
        let writer = a.command_buffer().unwrap();
        fill(&writer, &written);
        writer.commit().unwrap();
        reader.commit().unwrap();
        completed(&[&writer, &reader]);
        assert_eq!(
            read_buffer(&copy, 0, BYTES).unwrap(),
            vec![FILLED; BYTES as usize],
            "on {}",
            device.name()
        );
    });
}

/// A's execution of commands its kernel wrote waits for its host step, behind fills of a
/// ballast; B's copy of what the commands write, committed meanwhile, holds their writes: the
/// held batch keeps its place.
#[test]
fn a_copy_committed_while_a_write_waits_for_its_host_step_holds_it() {
    on_device(|device, messages| {
        let (a, b) = queues(device);
        let setup = Setup::new(device, 2);
        let (reader, copy) = copy_first(&b, setup.out());
        let ballast = device
            .new_buffer(BALLAST, ResourceOptions::private())
            .unwrap();
        let writer = a.command_buffer().unwrap();
        let mut blit = writer.blit_command_encoder().unwrap();
        for fill in 0..BALLAST_FILLS {
            blit.fill_buffer(&ballast, 0..BALLAST, fill as u8).unwrap();
        }
        blit.end_encoding().unwrap();
        let mut encoder = writer.compute_command_encoder().unwrap();
        setup.write(&mut encoder);
        encoder.end_encoding().unwrap();
        let mut encoder = writer.compute_command_encoder().unwrap();
        setup.execute(&mut encoder);
        encoder.end_encoding().unwrap();
        writer.commit().unwrap();
        reader.commit().unwrap();
        completed(&[&writer, &reader]);
        assert_eq!(setup.output(), expected());
        let copied: Vec<u32> = read_buffer(&copy, 0, copy.length())
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_le_bytes(*word))
            .collect();
        assert_eq!(copied, expected(), "on {}", device.name());
        assert_made_once(messages, &HOST_STEPS, &shared_kinds(device));
    });
}

/// A's fill has a submission check that returns only once B committed; B's copy, committed
/// meanwhile, holds the fill: the held batch keeps its place.
#[test]
fn a_copy_committed_while_a_write_waits_for_its_check_holds_it() {
    on_device(|device, _| {
        let (a, b) = queues(device);
        let written = buffer_with(device, &[0; BYTES as usize]).unwrap();
        let (reader, copy) = copy_first(&b, &written);
        let (entered, check_entered) = mpsc::channel();
        let (release, released) = mpsc::channel::<()>();
        let (entered, released) = (Mutex::new(entered), Mutex::new(released));
        let check: SubmissionCheck = Arc::new(move |_: &Submission<'_>| {
            let _ = entered.lock().unwrap().send(());
            let _ = released
                .lock()
                .unwrap()
                .recv_timeout(Duration::from_secs(60));
            None
        });
        let writer = a.command_buffer().unwrap();
        fill(&writer, &written);
        writer.set_submission_check(check).unwrap();
        writer.commit().unwrap();
        check_entered
            .recv_timeout(Duration::from_secs(30))
            .expect("the check runs on the scheduler thread");
        reader.commit().unwrap();
        release.send(()).unwrap();
        completed(&[&writer, &reader]);
        assert_eq!(
            read_buffer(&copy, 0, BYTES).unwrap(),
            vec![FILLED; BYTES as usize],
            "on {}",
            device.name()
        );
    });
}
