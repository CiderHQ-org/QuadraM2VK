//! A detach waiting for a submitted command buffer settles when the device is lost (`detach.rs`): a
//! no-copy buffer named by a command buffer whose first batch was submitted and whose next batch
//! waits for an event is detached on another thread, which blocks; the device is then marked lost
//! (as `device/relay.rs` marks it), the event never signaled, and the detach answers
//! `Err(DeviceLost)` within its poll.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use crate::error::Error;
use crate::{
    Device, DeviceKind, DeviceOptions, EventRef, HostMemory, ResourceOptions, ShaderCacheOptions,
};
use std::alloc::Layout;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::time::{Duration, Instant};

/// The longest a command buffer may take to start.
const WAIT: Duration = Duration::from_secs(60);

/// The no-copy buffer's bytes: a whole page of any host.
const BYTES: usize = 64 << 10;

/// A scratch directory for the device's shader cache, removed when dropped.
struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Scratch {
        let path = std::env::temp_dir().join(format!("m2v-detach-lost-{}", std::process::id()));
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

/// A buffer's detach blocked on a submitted command buffer that took it answers
/// `Err(DeviceLost)` once the device is lost, the command buffer never completing.
#[test]
fn a_detach_waiting_for_a_submitted_command_buffer_settles_on_a_lost_device() {
    let host = Host::new();
    let scratch = Scratch::new();
    let device = device(&scratch.0);
    // SAFETY: the memory outlives the device (dropped after it, below), and nothing else uses
    // it meanwhile.
    let made = unsafe {
        device.new_buffer_with_bytes_no_copy(
            HostMemory::new(host.0),
            BYTES as u64,
            ResourceOptions::default(),
            None,
        )
    };
    let buffer = match made {
        Ok(buffer) => buffer,
        Err(error) => {
            eprintln!("detach lost: no buffer over host memory: {error:?}");
            return;
        }
    };
    let queue = device.new_command_queue().unwrap();
    let copied = device.new_buffer(16, ResourceOptions::shared()).unwrap();
    let event = device.new_shared_event().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_buffer_to_buffer(&buffer, 0, &copied, 0, 16)
        .unwrap();
    blit.end_encoding().unwrap();
    command_buffer
        .encode_wait_for_event(EventRef::from(&event), 1)
        .unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.fill_buffer(&copied, 0..16, 0x3c).unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    let deadline = Instant::now() + WAIT;
    while command_buffer.kernel_start_time() == 0.0 {
        assert!(
            Instant::now() < deadline,
            "the first batch was never submitted"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    std::thread::scope(|scope| {
        let detaching = scope.spawn(|| buffer.detach_backing());
        std::thread::sleep(Duration::from_millis(100));
        assert!(
            !detaching.is_finished(),
            "the detach did not wait for the submitted command buffer"
        );
        device.mark_lost();
        assert_eq!(detaching.join().unwrap(), Err(Error::DeviceLost));
    });
}
