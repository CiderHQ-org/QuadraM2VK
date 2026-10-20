//! Device tests of a dispatch's set 0 on the device the cfg selects, under the validation layer:
//! pushed (`vkCmdPushDescriptorSet` at the compute bind point) where the device has the push
//! command, else allocated, written and bound, and the same words either way
//! (`WorkCounts::dispatches_pushing_set_zero`); and what a dispatch settles, taken again by the
//! encoder's next dispatch only where its inputs are the same (`WorkCounts::
//! dispatches_settled_as_before`, `set_zero_sets_reused`). Each runs both ways
//! (`draw_work::on_device_both_ways`).
//!
//! The kernel is the owned fixture `a4_index_word` (`out[t] = in[3t + 1]`).

use super::draw_work::{on_device_both_ways, pushes};
use super::indirect_host_steps::state;
use super::{buffer_with, read_buffer, WAIT};
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::ResourceUsage;
use metal2vulkan_runtime::{CommandBufferStatus, ResourceRef, Size, WorkCounts};

/// `after`'s count of `count` less `before`'s.
fn grew(before: &WorkCounts, after: &WorkCounts, count: fn(&WorkCounts) -> u64) -> u64 {
    count(after) - count(before)
}

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

/// What one dispatch of [`dispatches_settle_again_only_where_their_inputs_changed`] changed
/// before it ran, and the `out` word it writes from.
struct Step {
    change: &'static str,
    at: u64,
    threads: u64,
    /// Whether it takes what the dispatch before it settled.
    as_before: bool,
}

/// A compute encoder's dispatch takes the bind state, variant, plan, commands and made pipelines
/// its last dispatch settled only where it has the same inputs
/// (`WorkCounts::dispatches_settled_as_before`): a dispatch again of an unchanged state takes
/// them; one after a change of a buffer's offset, of the grid, after a sampler state was made, of
/// another pipeline state of the same kernel, or in an encoder that made a buffer resident,
/// settles again. Set 0 is apart from that memo: a dispatch binds the last set 0 wherever it
/// writes the same descriptors into a set of the same layout (`WorkCounts::set_zero_sets_reused`),
/// so each dispatch that settles as before binds it, where the device binds set 0, and the first
/// does not; whether a changed offset or grid changes the set's writes is the variant's (an
/// offset may be push data). Every dispatch writes its words. Both ways
/// ([`on_device_both_ways`]).
#[test]
fn dispatches_settle_again_only_where_their_inputs_changed() {
    on_device_both_ways(|device| {
        let kernel = state(device, "a4_index_word", false);
        let other = state(device, "a4_index_word", false);
        let input: Vec<u8> = (0..3 * 4 * THREADS as u32)
            .flat_map(u32::to_le_bytes)
            .collect();
        let input = buffer_with(device, &input).unwrap();
        const WORDS: u64 = 4 * THREADS;
        let out = buffer_with(device, &vec![0; (4 * WORDS) as usize]).unwrap();
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&kernel).unwrap();
        encoder.set_buffer(Some(&input), 0, 0).unwrap();
        encoder.set_buffer(Some(&out), 0, 1).unwrap();
        let steps = [
            Step {
                change: "the first",
                at: 0,
                threads: THREADS,
                as_before: false,
            },
            Step {
                change: "nothing",
                at: 0,
                threads: THREADS,
                as_before: true,
            },
            Step {
                change: "out's offset",
                at: THREADS,
                threads: THREADS,
                as_before: false,
            },
            Step {
                change: "the grid",
                at: THREADS,
                threads: 2 * THREADS,
                as_before: false,
            },
            Step {
                change: "nothing again",
                at: THREADS,
                threads: 2 * THREADS,
                as_before: true,
            },
            Step {
                change: "a sampler state made",
                at: THREADS,
                threads: 2 * THREADS,
                as_before: false,
            },
            Step {
                change: "the pipeline state",
                at: THREADS,
                threads: 2 * THREADS,
                as_before: false,
            },
            Step {
                change: "a buffer made resident",
                at: THREADS,
                threads: 2 * THREADS,
                as_before: false,
            },
        ];
        let mut sampler = None;
        let mut settled = Vec::new();
        for step in &steps {
            match step.change {
                "out's offset" => encoder.set_buffer_offset(4 * step.at, 1).unwrap(),
                "the pipeline state" => encoder.set_compute_pipeline_state(&other).unwrap(),
                "a sampler state made" => {
                    sampler = Some(device.new_sampler_state(&SamplerDescriptor::new()).unwrap());
                }
                "a buffer made resident" => {
                    encoder
                        .use_resource(ResourceRef::Buffer(&input), ResourceUsage::READ)
                        .unwrap();
                }
                _ => {}
            }
            let before = device.work_counts();
            encoder
                .dispatch_threads(Size::new(step.threads, 1, 1), Size::new(THREADS, 1, 1))
                .unwrap();
            let after = device.work_counts();
            let as_before = grew(&before, &after, |c| c.dispatches_settled_as_before);
            let reused = grew(&before, &after, |c| c.set_zero_sets_reused);
            settled.push((step.change, as_before, reused));
        }
        drop(sampler);
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        let binds = !pushes(device);
        let as_before: Vec<_> = (settled.iter())
            .map(|&(change, as_before, _)| (change, as_before))
            .collect();
        let want: Vec<_> = (steps.iter())
            .map(|step| (step.change, u64::from(step.as_before)))
            .collect();
        assert_eq!(as_before, want);
        for (step, &(change, _, reused)) in steps.iter().zip(&settled) {
            match (step.change, step.as_before) {
                ("the first", _) => assert_eq!(reused, 0, "{change}"),
                (_, true) => assert_eq!(reused, u64::from(binds), "{change}"),
                (_, false) => assert!(reused <= u64::from(binds), "{change}"),
            }
        }
        let words: Vec<u32> = read_buffer(&out, 0, 4 * WORDS)
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_le_bytes(*word))
            .collect();
        let mut expected = vec![0u32; WORDS as usize];
        for step in &steps {
            for t in 0..step.threads {
                expected[(step.at + t) as usize] = 3 * t as u32 + 1;
            }
        }
        assert_eq!(words, expected);
    });
}
