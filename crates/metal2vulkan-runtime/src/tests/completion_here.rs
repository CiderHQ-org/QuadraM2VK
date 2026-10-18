//! `CommandBufferShared::finish_here`: a command buffer that runs none of the program's code as it
//! completes finishes on the queue waiter that saw its work complete, with no handler thread
//! woken, as Metal's completion wakes its waiting thread itself (`command_buffer/completion.rs`);
//! one with a completed handler, and any while the runtime holds a deallocator of the program's,
//! finishes on its queue's handler thread (`WorkCounts::completed_where_seen`). Likewise
//! `CommandBufferShared::schedule`: a command buffer with no scheduled handler is scheduled on the
//! scheduler's thread as its last batch goes, and one with a scheduled handler on its queue's
//! handler thread (`WorkCounts::scheduled_where_submitted`).
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use crate::{
    CommandBuffer, CommandQueue, Device, DeviceKind, DeviceOptions, HostMemory, ResourceOptions,
    ShaderCacheOptions,
};
use std::alloc::Layout;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;

/// The command buffers of each kind.
const ROUNDS: u64 = 20;

/// The no-copy buffer's bytes: a whole page of any host.
const BYTES: usize = 64 << 10;

/// A scratch directory for the device's shader cache, removed when dropped.
struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Scratch {
        let path = std::env::temp_dir().join(format!("m2v-completion-here-{}", std::process::id()));
        std::fs::create_dir_all(&path).unwrap();
        Scratch(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Page-aligned host memory of [`BYTES`], freed when dropped: declared first, after the device
/// that used it.
struct Host(NonNull<u8>);

impl Host {
    fn layout() -> Layout {
        Layout::from_size_align(BYTES, BYTES).unwrap()
    }

    fn new() -> Host {
        // SAFETY: the layout's size is not zero.
        let pointer = unsafe { std::alloc::alloc_zeroed(Host::layout()) };
        Host(NonNull::new(pointer).unwrap())
    }
}

impl Drop for Host {
    fn drop(&mut self) {
        // SAFETY: allocated with this layout in `Host::new`.
        unsafe { std::alloc::dealloc(self.0.as_ptr(), Host::layout()) };
    }
}

/// The device the cfg selects: a CPU device under `m2v_cpu_icd`, else a GPU, its shader cache
/// in `cache`.
fn device(cache: &Path) -> Device {
    let options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        shader_cache: ShaderCacheOptions {
            directory: Some(cache.to_owned()),
            ..ShaderCacheOptions::default()
        },
        ..DeviceOptions::default()
    };
    Device::system_default_with(&options).unwrap()
}

/// [`ROUNDS`] command buffers of `queue`, each a fill of 16 bytes, with a completed handler
/// where `handler`, each committed and waited for before the next: how many finished where
/// their completion was seen.
fn finished_where_seen(device: &Device, queue: &CommandQueue, handler: bool) -> u64 {
    let target = device.new_buffer(16, ResourceOptions::shared()).unwrap();
    let before = device.work_counts().completed_where_seen;
    for round in 0..ROUNDS {
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.fill_buffer(&target, 0..16, round as u8).unwrap();
        blit.end_encoding().unwrap();
        if handler {
            command_buffer
                .add_completed_handler(Box::new(|_: &CommandBuffer| {}))
                .unwrap();
        }
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed().unwrap();
    }
    device.work_counts().completed_where_seen - before
}

/// Command buffers with no handler finish where their completion is seen: at least one of
/// [`ROUNDS`] (each's transition to `Scheduled` may be running on a handler thread as its work
/// completes, and it is then posted behind it). Those with a completed handler never do, nor do
/// any while a no-copy buffer's deallocator of the program's is held.
#[test]
fn a_command_buffer_with_no_handler_finishes_where_its_completion_is_seen() {
    let host = Host::new();
    let scratch = Scratch::new();
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let plain = finished_where_seen(&device, &queue, false);
    println!("finished where seen: {plain} of {ROUNDS} with no handler");
    assert!(plain >= 1, "none of {ROUNDS} finished where seen");
    assert_eq!(
        finished_where_seen(&device, &queue, true),
        0,
        "a command buffer with a completed handler finished off its queue's handler thread"
    );
    // SAFETY: the memory outlives the buffer (dropped after it, below), and nothing else uses it.
    let made = unsafe {
        device.new_buffer_with_bytes_no_copy(
            HostMemory::new(host.0),
            BYTES as u64,
            ResourceOptions::default(),
            Some(Box::new(|_: NonNull<u8>, _: u64| {})),
        )
    };
    match made {
        Ok(buffer) => {
            assert_eq!(
                finished_where_seen(&device, &queue, false),
                0,
                "a command buffer finished off its queue's handler thread while a deallocator \
                 of the program's was held"
            );
            drop(buffer);
        }
        Err(error) => println!("no buffer over host memory: {error:?}"),
    }
    drop(queue);
    drop(device);
}

/// [`ROUNDS`] command buffers of `queue`, each a fill of 16 bytes, with a scheduled handler where
/// `handler`, each committed and waited for before the next: how many were scheduled where their
/// last batch was submitted.
fn scheduled_where_submitted(device: &Device, queue: &CommandQueue, handler: bool) -> u64 {
    let target = device.new_buffer(16, ResourceOptions::shared()).unwrap();
    let before = device.work_counts().scheduled_where_submitted;
    for round in 0..ROUNDS {
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.fill_buffer(&target, 0..16, round as u8).unwrap();
        blit.end_encoding().unwrap();
        if handler {
            command_buffer
                .add_scheduled_handler(Box::new(|_: &CommandBuffer| {}))
                .unwrap();
        }
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed().unwrap();
    }
    device.work_counts().scheduled_where_submitted - before
}

/// Command buffers with no scheduled handler are scheduled where their last batch is submitted:
/// at least one of [`ROUNDS`] (the one before's transition to `Completed` may still hold its
/// queue's context as the next one's last batch goes, which is then posted behind it). Those with
/// a scheduled handler never are.
#[test]
fn a_command_buffer_with_no_scheduled_handler_is_scheduled_where_submitted() {
    let scratch = Scratch::new();
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let plain = scheduled_where_submitted(&device, &queue, false);
    println!("scheduled where submitted: {plain} of {ROUNDS} with no handler");
    assert!(plain >= 1, "none of {ROUNDS} was scheduled where submitted");
    assert_eq!(
        scheduled_where_submitted(&device, &queue, true),
        0,
        "a command buffer with a scheduled handler was scheduled off its queue's handler thread"
    );
    drop(queue);
    drop(device);
}
