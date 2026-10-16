//! What one kernel submission costs against the imports a device holds, on the device the cfg
//! selects, without the validation layer: report only, in release builds.
//!
//! RADV lists every imported `VkDeviceMemory` of the device in every command submission's BO list:
//! Mesa 26.2.4's `radv_alloc_memory` makes every memory but WSI's resident, unconditionally (26.0.8
//! wherever the kernel has always-valid BOs), and its amdgpu winsys leaves out of the global list
//! only local BOs, memory neither imported nor exported, which the kernel tracks per VM
//! (`radv_amdgpu_winsys_bo_make_resident`); each submission's list is the whole global list
//! (`radv_amdgpu_get_bo_list`). amdgpu's work per submission is per entry (Linux 7.0
//! `amdgpu_cs.c`: lookup and sort, lock, validate, VM update, fence sync and fence add), and a
//! userptr entry also faults its whole range in again. ANV on i915 lists every `VkDeviceMemory`
//! (`submit_costs`). The runtime's own memory is pooled in 64-MiB blocks, which RADV makes local.
//!
//! The lane holds `n` imports live, for each `n` of [`COUNTS`], of two kinds, each
//! [`IMPORT_BYTES`] over pages of its own: dma-bufs (udmabufs of a sealed `memfd`'s pages, as
//! QEMU's guest RAM, which RADV imports as dma-bufs where it refuses their pointer) and the
//! pointers of private anonymous pages (amdgpu userptrs on RADV). It then commits [`COMMITS`]
//! command buffers of one 4-KiB fill one at a time, each waited for, after [`WARM`] untimed;
//! for dma-bufs twice, at once with every import resident (as before the device released idle
//! imports), then once the device's residency thread released them after their first idle window
//! (`host_residency.rs` of the runtime), as many as the descriptors the process keeps allow. Printed as `SUBMIT-PER-IMPORT` lines, per kind, `n` and
//! residency: the imports the device counts (`WorkCounts::dma_buf_imports` made,
//! `WorkCounts::dma_buf_memories_live`, `WorkCounts::host_imports_live`,
//! `WorkCounts::host_imports_resident`) and its pooled blocks; a command buffer's time
//! from its kernelStartTime to its kernelEndTime (the scheduler's submission of it to Vulkan), its
//! 50th percentile and mean; the scheduler thread's CPU time per commit; and, where the
//! kernel-submission counter (`scripts/i915-execbuffers/counter.c`) is preloaded, the kernel
//! submissions per commit, the objects each lists (dma-bufs and userptrs among them) and the
//! microseconds each spent in the ioctl. Its slope over `n` is the kernel's cost per entry.
//!
//! Ignored by default: a measurement, not a check. Run it by name with `--run-ignored all`; on a
//! host without `/dev/udmabuf` access the dma-buf rows say so and are skipped.

use super::commit_latency::named_cpu_us;
use super::dma_buf_imports::{no_copy as dma_buf_no_copy, udmabuf_openable, Calls, Memfd};
use super::host_memory::{no_copy, Mapping};
use super::submit_costs::Execbuffers;
use super::{validated_device_with, Messages, ScratchCache, WAIT};
use crate::gpu::TEST_DEVICE;
use metal2vulkan_runtime::{
    Buffer, CommandBuffer, CommandBufferStatus, CommandQueue, Device, ResourceOptions,
};
use std::time::{Duration, Instant};

/// The live imports of each phase.
const COUNTS: [usize; 5] = [0, 100, 200, 400, 1000];
/// The bytes of each import: one 64-KiB chunk, RADV 26.2.4's sparse granule, so one dma-buf.
const IMPORT_BYTES: usize = 64 << 10;
/// Command buffers committed untimed first.
const WARM: usize = 30;
/// Command buffers measured.
const COMMITS: usize = 300;
/// The longest the probe waits for the residency thread to release the imports it may.
const RELEASED: Duration = Duration::from_secs(30);
/// The linux name (15 bytes) of the device's scheduler thread.
const SCHEDULER: &str = "metal2vulkan-sc";

/// The kind of import a phase holds.
#[derive(Clone, Copy, Debug)]
enum Kind {
    DmaBuf,
    Pointer,
}

/// A phase's live imports: the buffers, then the memory they lie over.
enum Held {
    DmaBuf {
        _buffers: Vec<Buffer>,
        _memfd: Memfd,
    },
    Pointer {
        _buffers: Vec<Buffer>,
        _pages: Mapping,
    },
}

/// `n` live imports of `kind` on `device`, each over pages of its own.
fn hold(device: &Device, kind: Kind, n: usize) -> Held {
    let len = (n * IMPORT_BYTES).max(IMPORT_BYTES);
    match kind {
        Kind::DmaBuf => {
            let memfd = Memfd::new(len);
            let calls = Calls::default();
            let source = memfd.source(&memfd.whole, &calls);
            let buffers = (0..n)
                .map(|i| {
                    let at = i * IMPORT_BYTES;
                    dma_buf_no_copy(device, &memfd.whole, at, IMPORT_BYTES, source.clone()).unwrap()
                })
                .collect();
            Held::DmaBuf {
                _buffers: buffers,
                _memfd: memfd,
            }
        }
        Kind::Pointer => {
            let pages = Mapping::anonymous(len);
            pages.fill(0x5a);
            let buffers = (0..n)
                .map(|i| {
                    // SAFETY: the pages are this process's, mapped read and write, and outlive
                    // the buffers (`Held`'s field order).
                    let at = unsafe { pages.pointer().add(i * IMPORT_BYTES) };
                    // SAFETY: as above.
                    unsafe { no_copy(device, at, IMPORT_BYTES, ResourceOptions::shared(), None) }
                        .unwrap()
                })
                .collect();
            Held::Pointer {
                _buffers: buffers,
                _pages: pages,
            }
        }
    }
}

/// A command buffer of `queue` filling `target`'s 4 KiB with `value`, not committed.
fn fill(queue: &CommandQueue, target: &Buffer, value: u8) -> CommandBuffer {
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.fill_buffer(target, 0..4096, value).unwrap();
    blit.end_encoding().unwrap();
    command_buffer
}

/// Commit `command_buffer` and wait for it: the seconds from its kernelStartTime to its
/// kernelEndTime.
fn submitted(command_buffer: CommandBuffer) -> f64 {
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    command_buffer.kernel_end_time() - command_buffer.kernel_start_time()
}

/// The phase of `n` imports of `kind`: held, measured and printed. Dma-bufs twice: first at
/// once, every import resident; then again once the device's residency thread released those it
/// may (after their first idle window, as many as the descriptors the process keeps allow).
fn phase(device: &Device, queue: &CommandQueue, target: &Buffer, kind: Kind, n: usize) {
    let made = device.work_counts().dma_buf_imports;
    let held = hold(device, kind, n);
    measure(device, queue, target, (kind, n, "at once"), made);
    if matches!(kind, Kind::DmaBuf) {
        let start = Instant::now();
        while device.work_counts().host_imports_resident > 0 {
            assert!(start.elapsed() < RELEASED, "imports still resident");
            std::thread::sleep(Duration::from_millis(50));
        }
        let waited = format!("released after {:.1} s", start.elapsed().as_secs_f64());
        measure(device, queue, target, (kind, n, &waited), made);
    }
    drop(held);
}

/// Commit and time command buffers beside `kind`'s `n` imports in `residency`, and print what
/// they took; `made` the dma-buf chunks the device imported before the imports.
fn measure(
    device: &Device,
    queue: &CommandQueue,
    target: &Buffer,
    (kind, n, residency): (Kind, usize, &str),
    made: u64,
) {
    for value in 0..WARM {
        submitted(fill(queue, target, value as u8));
    }
    let counts = device.work_counts();
    let execbuffers = Execbuffers::read();
    let scheduler = named_cpu_us(SCHEDULER);
    let mut times: Vec<f64> = (0..COMMITS)
        .map(|value| submitted(fill(queue, target, value as u8)))
        .collect();
    let scheduler = (named_cpu_us(SCHEDULER) - scheduler) / COMMITS as f64;
    times.sort_by(f64::total_cmp);
    let p50 = times[times.len() / 2] * 1e6;
    let mean = times.iter().sum::<f64>() / times.len() as f64 * 1e6;
    let kernel = Execbuffers::read()
        .zip(execbuffers)
        .map(|(after, before)| {
            let submissions = after.execbuffers - before.execbuffers;
            let each = |after: u64, before: u64| (after - before) as f64 / submissions as f64;
            format!(
                "; {:.2} kernel submissions a commit, {:.0} objects each ({:.0} dma-bufs, {:.0} \
                 userptrs), {:.1} us each in the ioctl",
                submissions as f64 / COMMITS as f64,
                each(after.objects, before.objects),
                each(after.prime, before.prime),
                each(after.userptr, before.userptr),
                each(after.nanos, before.nanos) / 1e3,
            )
        })
        .unwrap_or_default();
    eprintln!(
        "SUBMIT-PER-IMPORT {}: {kind:?} {n} {residency}: {} dma-buf chunks imported ({} live), \
         {} host imports live ({} releasable resident), {} pooled blocks live; submit p50 \
         {p50:.1} us, mean {mean:.1} us; scheduler thread {scheduler:.1} us a commit{kernel}",
        device.name(),
        counts.dma_buf_imports - made,
        counts.dma_buf_memories_live,
        counts.host_imports_live,
        counts.host_imports_resident,
        counts.memory_blocks_live,
    );
}

/// The phases, printed (the module docs).
#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn submit_per_import() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
        options.layers.clear();
        // A consumer's device keeps no execution record.
        options.record_executions = false;
    })
    .unwrap();
    let queue = device.new_command_queue().unwrap();
    let target = device.new_buffer(4096, ResourceOptions::shared()).unwrap();
    let dma_bufs = udmabuf_openable("submit per import");
    for kind in [Kind::DmaBuf, Kind::Pointer] {
        if matches!(kind, Kind::DmaBuf) && !dma_bufs {
            continue;
        }
        for n in COUNTS {
            phase(&device, &queue, &target, kind, n);
        }
    }
}
