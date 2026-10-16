//! Freed memory that waits for the work in flight (`freed_memory.rs`), in both settings: on a
//! device made to park freed memory through the crate's hook (`Device::park_freed_memory`, as
//! where the driver has `RuntimeDriverDefect::FreedAddressMemoryFailsPendingWork`), a host import
//! released while a batch submitted before it runs keeps its memory, and its deallocator waits,
//! until that batch completed, then goes on a handler thread; elsewhere it goes at once, as
//! before. The counts show which path ran (`WorkCounts::freed_memory_parked`,
//! `freed_memory_waiting`, `host_deallocators_delayed`). And what a block notes as it goes holds
//! it for the batches and the device's own submissions made before, and for none after.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use crate::freed_memory::Parking;
use crate::{
    CommandBuffer, CommandBufferStatus, CommandQueue, Device, DeviceKind, DeviceOptions, Error,
    HostMemory, ResourceOptions, ShaderCacheOptions,
};
use std::alloc::Layout;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::ThreadId;
use std::time::{Duration, Instant};

/// The longest a command buffer, or a release after it, may take.
const WAIT: Duration = Duration::from_secs(60);

/// The no-copy buffer's bytes: a whole page of any host.
const BYTES: usize = 64 << 10;

/// The fills that keep the batch in flight: [`FILLS`] of [`FILLED`] bytes each.
const FILLS: u32 = 16;
const FILLED: u64 = 64 << 20;

/// A scratch directory for the device's shader cache, removed when dropped.
struct Scratch(PathBuf);

impl Scratch {
    fn new(case: &str) -> Scratch {
        let name = format!("m2v-freed-memory-{case}-{}", std::process::id());
        let path = std::env::temp_dir().join(name);
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

/// The runs of a deallocator: how many, and the thread of the last.
#[derive(Default)]
struct Runs {
    count: AtomicUsize,
    thread: Mutex<Option<ThreadId>>,
}

/// A no-copy buffer over `host` that no command uses, its deallocator's runs in `runs`; `None`,
/// printed, where the device imports no host memory.
fn unused(device: &Device, host: &Host, runs: &Arc<Runs>) -> Option<crate::Buffer> {
    let runs = Arc::clone(runs);
    // SAFETY: the memory outlives the buffer (dropped after the device), and nothing else uses it.
    let made = unsafe {
        device.new_buffer_with_bytes_no_copy(
            HostMemory::new(host.0),
            BYTES as u64,
            ResourceOptions::default(),
            Some(Box::new(move |_: NonNull<u8>, _: u64| {
                *runs.thread.lock().unwrap() = Some(std::thread::current().id());
                runs.count.fetch_add(1, Ordering::SeqCst);
            })),
        )
    };
    match made {
        Ok(buffer) => Some(buffer),
        Err(Error::Unsupported(gap)) => {
            eprintln!("FREED-MEMORY: the device imports no host memory: {gap:?}");
            None
        }
        Err(error) => panic!("new_buffer_with_bytes_no_copy: {error:?}"),
    }
}

/// A command buffer of `queue`'s that fills `target` [`FILLS`] times, committed, once its batch
/// was submitted.
fn filling(queue: &CommandQueue, target: &crate::Buffer) -> CommandBuffer {
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    for value in 0..FILLS {
        blit.fill_buffer(target, 0..FILLED, value as u8).unwrap();
    }
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    let deadline = Instant::now() + WAIT;
    while command_buffer.kernel_start_time() == 0.0 {
        assert!(Instant::now() < deadline, "the batch was never submitted");
        std::thread::yield_now();
    }
    command_buffer
}

/// Wait until `done`, failing with `what` past [`WAIT`].
fn until(what: &str, done: impl Fn() -> bool) {
    let deadline = Instant::now() + WAIT;
    while !done() {
        assert!(Instant::now() < deadline, "{what}");
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// Where the device parks freed memory, a host import released while a batch submitted before it
/// runs keeps its memory, its deallocator waiting, until that batch completed; then both go, on a
/// thread of the runtime's, and the wait was announced once.
#[test]
fn a_release_waits_for_the_batch_in_flight_where_the_device_parks_freed_memory() {
    let scratch = Scratch::new("parked");
    let host = Host::new();
    let device = device(&scratch.0);
    device.park_freed_memory();
    let queue = device.new_command_queue().unwrap();
    let runs = Arc::new(Runs::default());
    let Some(buffer) = unused(&device, &host, &runs) else {
        return;
    };
    let target = device
        .new_buffer(FILLED, ResourceOptions::private())
        .unwrap();
    let command_buffer = filling(&queue, &target);
    let before = device.work_counts();
    let pending = command_buffer.status() != CommandBufferStatus::Completed;
    drop(buffer);
    let after = device.work_counts();
    eprintln!(
        "FREED-MEMORY: at the release: parked {}, waiting {}, deallocators delayed {}, run {}",
        after.freed_memory_parked - before.freed_memory_parked,
        after.freed_memory_waiting,
        after.host_deallocators_delayed - before.host_deallocators_delayed,
        runs.count.load(Ordering::SeqCst)
    );
    let waited = pending && command_buffer.status() != CommandBufferStatus::Completed;
    match waited {
        true => {
            assert_eq!(after.freed_memory_parked - before.freed_memory_parked, 1);
            assert_eq!(after.freed_memory_waiting, 1);
            assert_eq!(
                after.host_deallocators_delayed - before.host_deallocators_delayed,
                1,
                "the deallocator's wait is announced"
            );
            assert_eq!(
                runs.count.load(Ordering::SeqCst),
                0,
                "the deallocator ran while the batch submitted before its release ran"
            );
        }
        false => eprintln!("FREED-MEMORY: the fills completed before the release: none waited"),
    }
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    until("the deallocator never ran", || {
        runs.count.load(Ordering::SeqCst) == 1
    });
    until("the memory never went", || {
        device.work_counts().freed_memory_waiting == 0
    });
    eprintln!(
        "FREED-MEMORY: after completion: waiting {}, run {}",
        device.work_counts().freed_memory_waiting,
        runs.count.load(Ordering::SeqCst)
    );
    // A release after its batch completed runs the deallocator at once, on the program's thread.
    if waited {
        assert_ne!(
            *runs.thread.lock().unwrap(),
            Some(std::thread::current().id()),
            "the deallocator ran on the program's thread after its batch completed"
        );
    }
}

/// Where the device parks no freed memory, the same release goes at once, on the releasing
/// thread, and nothing waits.
#[test]
fn a_release_goes_at_once_where_the_device_parks_no_freed_memory() {
    let scratch = Scratch::new("at-once");
    let host = Host::new();
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let runs = Arc::new(Runs::default());
    let Some(buffer) = unused(&device, &host, &runs) else {
        return;
    };
    let target = device
        .new_buffer(FILLED, ResourceOptions::private())
        .unwrap();
    let command_buffer = filling(&queue, &target);
    let parks = device
        .properties()
        .driver_defects
        .contains(&metal2vulkan::error::RuntimeDriverDefect::FreedAddressMemoryFailsPendingWork);
    drop(buffer);
    match parks {
        true => eprintln!("FREED-MEMORY: the device's driver has the defect: it parks"),
        false => {
            assert_eq!(
                runs.count.load(Ordering::SeqCst),
                1,
                "the release went at once"
            );
            assert_eq!(
                *runs.thread.lock().unwrap(),
                Some(std::thread::current().id())
            );
            let counts = device.work_counts();
            eprintln!(
                "FREED-MEMORY: at the release: parked {}, waiting {}, run {}",
                counts.freed_memory_parked,
                counts.freed_memory_waiting,
                runs.count.load(Ordering::SeqCst)
            );
            assert_eq!(
                (counts.freed_memory_parked, counts.freed_memory_waiting),
                (0, 0)
            );
        }
    }
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
}

/// A block's note holds it until each queue's timeline reached the value of the last batch
/// submitted there before it, and until every own submission of the device's numbered before it
/// completed; a batch or an own submission made after the note holds it not.
#[test]
fn a_note_holds_a_block_for_the_work_submitted_before_it_alone() {
    let scratch = Scratch::new("noted");
    let device = device(&scratch.0);
    let parking = Parking::new(2);
    let empty = parking.note();
    assert!(parking.reached(&empty), "nothing was submitted");
    parking.submitted(1, 7);
    let batch = parking.note();
    parking.submitted(1, 8);
    assert!(parking.reached(&empty), "a batch after the note");
    assert!(!parking.reached(&batch), "the batch in flight");
    parking.completed(1, 7);
    assert!(parking.reached(&batch), "the batch completed");
    parking.completed(1, 8);
    let own = parking.own_submission(&device);
    let during = parking.note();
    assert!(parking.reached(&batch), "an own submission after the note");
    assert!(!parking.reached(&during), "the own submission in flight");
    let later = parking.own_submission(&device);
    assert!(
        !parking.reached(&during),
        "the own submission still in flight"
    );
    drop(own);
    assert!(
        parking.reached(&during),
        "the own submission completed; the later one is after"
    );
    let last = parking.note();
    assert!(
        !parking.reached(&last),
        "the later own submission in flight"
    );
    drop(later);
    assert!(parking.reached(&last), "every own submission completed");
}
