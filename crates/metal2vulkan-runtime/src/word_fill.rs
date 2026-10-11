//! A fill of a buffer's whole 32-bit words with one word, `vkCmdFillBuffer`'s, as every fill the
//! runtime records goes.
//!
//! - **Where the driver compiles its fill.** MoltenVK runs `vkCmdFillBuffer` as a compute
//!   dispatch of a pipeline it makes at the first fill each `VkCommandPool`'s command buffers
//!   encode (`MVKCommandEncodingPool::getCmdFillBufferMTLComputePipelineState`), inside
//!   `vkQueueSubmit`, through Metal's compiler service, which runs the process's compiles in
//!   order. Behind a storm of pipeline creations, a commit holding a pool's first fill waited
//!   250 to 330 ms in that submit on an M4. On a release measured so
//!   (`DriverFact::FillCompiledPerCommandPool`, MoltenVK 1.4.1), a fill goes as transfers, which
//!   MoltenVK runs as blits and compiles nothing for:
//!   `vkCmdUpdateBuffer` of the first words (at most 65536 bytes, its limit), then copies of the
//!   filled words onto the rest, each copying all that is filled so far, past a barrier.
//! - **Exact.** Each copy reads only words the fill wrote before the barrier ahead of it, and
//!   writes words no earlier step of the fill wrote, so its source and destination never overlap
//!   (as `vkCmdCopyBuffer` requires), and every word ends as the fill's.
//! - **Elsewhere** `vkCmdFillBuffer`, unchanged, on every other driver and release.
//! - **Counted.** Each fill counts as one of the two forms (`WorkCounts::fills_recorded` and
//!   `fills_by_transfers`), so a run shows which its device took.

use crate::Device;
use ash::vk;
use metal2vulkan::device::DriverFact;
use std::ops::Range;

/// The most bytes one `vkCmdUpdateBuffer` writes.
const UPDATE_LIMIT: u64 = 65536;

/// One step of a fill by transfers ([`steps`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Step {
    /// `vkCmdUpdateBuffer` of these bytes with the word.
    Update(Range<u64>),
    /// The transfers before it have written before the ones after it read.
    Barrier,
    /// `vkCmdCopyBuffer` of `len` bytes at `from` to `to`, within the buffer.
    Copy { from: u64, to: u64, len: u64 },
}

/// The steps that fill `words` (its ends multiples of 4) by transfers: the first words updated,
/// then each copy doubles what is filled, from the filled words at its start, past a barrier.
pub(crate) fn steps(words: Range<u64>) -> Vec<Step> {
    let total = words.end.saturating_sub(words.start);
    if total == 0 {
        return Vec::new();
    }
    let first = total.min(UPDATE_LIMIT);
    let mut steps = vec![Step::Update(words.start..words.start + first)];
    let mut filled = first;
    while filled < total {
        let len = filled.min(total - filled);
        steps.push(Step::Barrier);
        steps.push(Step::Copy {
            from: words.start,
            to: words.start + filled,
            len,
        });
        filled += len;
    }
    steps
}

/// Record a fill of `words` of `buffer` with `word` on `owner`'s command buffer
/// `command_buffer`: `vkCmdFillBuffer`, or the [`steps`] by transfers where the device's driver
/// has [`DriverFact::FillCompiledPerCommandPool`]. The caller's barriers order it as one transfer
/// that writes `words` (it also reads them, in the transfer stage).
///
/// # Safety
///
/// `command_buffer` is recording, outside a render pass; `buffer` is live until it completes,
/// made with `TRANSFER_DST` usage (and `TRANSFER_SRC` where `words` holds more than 65536 bytes),
/// and `words` lies in it, its ends multiples of 4.
pub(crate) unsafe fn record(
    owner: &Device,
    command_buffer: vk::CommandBuffer,
    buffer: vk::Buffer,
    words: Range<u64>,
    word: u32,
) {
    let device = owner.shared().vk();
    let by_transfers = owner
        .properties()
        .driver_facts
        .contains(&DriverFact::FillCompiledPerCommandPool);
    if words.is_empty() {
        return;
    }
    owner.shared().counts.fill_recorded(by_transfers);
    if !by_transfers {
        // SAFETY: the caller's contract.
        unsafe {
            device.cmd_fill_buffer(
                command_buffer,
                buffer,
                words.start,
                words.end - words.start,
                word,
            );
        }
        return;
    }
    let first = (words.end - words.start).min(UPDATE_LIMIT) / 4;
    let bytes: Vec<u8> = (0..first).flat_map(|_| word.to_ne_bytes()).collect();
    for step in steps(words) {
        // SAFETY: the caller's contract; each update writes at most 65536 bytes of `bytes`, a
        // multiple of 4 at a multiple of 4, and each copy's regions lie in `words` and do not
        // overlap (the module docs).
        unsafe {
            match step {
                Step::Update(range) => device.cmd_update_buffer(
                    command_buffer,
                    buffer,
                    range.start,
                    &bytes[..(range.end - range.start) as usize],
                ),
                Step::Barrier => {
                    let barrier = vk::MemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
                    device.cmd_pipeline_barrier(
                        command_buffer,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[barrier],
                        &[],
                        &[],
                    );
                }
                Step::Copy { from, to, len } => {
                    let region = vk::BufferCopy::default()
                        .src_offset(from)
                        .dst_offset(to)
                        .size(len);
                    device.cmd_copy_buffer(command_buffer, buffer, buffer, &[region]);
                }
            }
        }
    }
}
