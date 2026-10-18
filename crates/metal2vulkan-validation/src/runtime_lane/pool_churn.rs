//! A frame's buffers, placed each frame and returned once its command buffer completed, on the
//! device the cfg selects, under the validation layer: once warm, a frame makes and frees no
//! pooled block of memory for them (`memory.rs`'s `KEPT_EMPTY_BLOCKS`). Each frame makes private
//! buffers that fill most of a pooled block, fills each in one command buffer, commits it, waits
//! for it and drops them. The frame's block empties as the next frame places its buffers in the
//! block the pool keeps with room for the encodings' placements: a pool that kept one empty block
//! freed the frame's block and made one again every frame.

use super::degrade_notices::on_device;
use super::WAIT;
use metal2vulkan_runtime::{Buffer, CommandBufferStatus, CommandQueue, Device, ResourceOptions};

/// A frame's buffers' lengths: seven of 9 MiB, 63 MiB of a 64 MiB block, beside which no other
/// frame's buffer fits.
const FRAME_BUFFERS: [u64; 7] = [9 << 20; 7];

/// The frames before the counted ones, which place the working set, and the counted frames.
const WARM_FRAMES: usize = 4;
const COUNTED_FRAMES: usize = 32;

/// One frame's buffers, made, filled on `queue` and waited for.
fn frame(device: &Device, queue: &CommandQueue, value: u8) -> Vec<Buffer> {
    let buffers: Vec<Buffer> = (FRAME_BUFFERS.iter())
        .map(|&len| device.new_buffer(len, ResourceOptions::private()).unwrap())
        .collect();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    for buffer in &buffers {
        blit.fill_buffer(buffer, 0..buffer.length(), value).unwrap();
    }
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    buffers
}

/// The frames: once warm, the pooled blocks made and freed over the counted frames are none.
#[test]
fn buffers_returned_each_frame_make_and_free_no_block() {
    on_device(|device, _| {
        let queue = device.new_command_queue().unwrap();
        let mut counted = None;
        for index in 0..WARM_FRAMES + COUNTED_FRAMES {
            if index == WARM_FRAMES {
                counted = Some(device.work_counts());
            }
            drop(frame(device, &queue, index as u8));
        }
        let before = counted.unwrap();
        let after = device.work_counts();
        let made = after.pooled_memory_blocks_made - before.pooled_memory_blocks_made;
        let freed = after.pooled_memory_blocks_freed - before.pooled_memory_blocks_freed;
        eprintln!(
            "POOL-CHURN {}: {:.2} pooled blocks made and {:.2} freed a frame over {COUNTED_FRAMES} \
             frames; {} pooled blocks in use and {} empty after them",
            device.name(),
            made as f64 / COUNTED_FRAMES as f64,
            freed as f64 / COUNTED_FRAMES as f64,
            after.memory_blocks_live,
            after.empty_memory_blocks_live,
        );
        assert_eq!((made, freed), (0, 0), "pooled blocks made and freed");
    });
}
