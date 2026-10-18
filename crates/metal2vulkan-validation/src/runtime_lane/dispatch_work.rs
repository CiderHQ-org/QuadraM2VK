//! Device tests of a dispatch's set 0 on the device the cfg selects, under the validation layer:
//! pushed (`vkCmdPushDescriptorSet` at the compute bind point) where the device has the push
//! command, else allocated, written and bound, and the same words either way
//! (`WorkCounts::dispatches_pushing_set_zero`). Each runs both ways (`draw_work::on_device_both_ways`).
//!
//! The kernel is the owned fixture `a4_index_word` (`out[t] = in[3t + 1]`).

use super::draw_work::{on_device_both_ways, pushes};
use super::indirect_host_steps::state;
use super::{buffer_with, read_buffer, WAIT};
use metal2vulkan_runtime::{CommandBufferStatus, Size};

/// Threads a dispatch.
const THREADS: u64 = 16;
/// Dispatches, each into its own words of `out`, from its own words of `in`.
const DISPATCHES: u64 = 4;

/// Dispatches that each bind other offsets of the same buffers push their set 0 where the device
/// pushes it, and bind it where it does not: every dispatch's words are its own, both ways.
#[test]
fn dispatches_push_their_set_zero_where_the_device_pushes() {
    on_device_both_ways(|device| {
        let kernel = state(device, "a4_index_word", false);
        let words = 3 * THREADS * DISPATCHES;
        let input: Vec<u8> = (0..words as u32).flat_map(u32::to_le_bytes).collect();
        let input = buffer_with(device, &input).unwrap();
        let out = buffer_with(device, &vec![0; (4 * THREADS * DISPATCHES) as usize]).unwrap();
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&kernel).unwrap();
        let before = device.work_counts();
        for d in 0..DISPATCHES {
            encoder
                .set_buffer(Some(&input), 4 * 3 * THREADS * d, 0)
                .unwrap();
            encoder.set_buffer(Some(&out), 4 * THREADS * d, 1).unwrap();
            encoder
                .dispatch_threads(Size::new(THREADS, 1, 1), Size::new(THREADS, 1, 1))
                .unwrap();
        }
        let pushed =
            device.work_counts().dispatches_pushing_set_zero - before.dispatches_pushing_set_zero;
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        assert_eq!(pushed, if pushes(device) { DISPATCHES } else { 0 });
        let words: Vec<u32> = read_buffer(&out, 0, 4 * THREADS * DISPATCHES)
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_le_bytes(*word))
            .collect();
        let want: Vec<u32> = (0..(THREADS * DISPATCHES) as u32)
            .map(|t| 3 * t + 1)
            .collect();
        assert_eq!(words, want);
    });
}
