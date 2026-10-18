//! The residency of no-copy buffers' pointer imports (`host_residency.rs`), on the device the cfg
//! selects.
//!
//! The memory is guest RAM as QEMU's `memory-backend-memfd,share=on` makes it, a sealed `memfd`
//! mapped shared, imported by its pointer (`HostMemory::new`), as ANV, NVIDIA and lavapipe import
//! - that a kept import reads the pages a remap put at its address, which is the fact itself, on
//!   a device with it; on any other the test prints the pages it read, a measurement of the
//!   driver;
//! - the counts that show the release and the import again ran (`WorkCounts::host_imports_released`,
//!   `host_imports_reimported`, `host_imports_resident`, `host_imports_live`), and none of a
//!   dma-buf's descriptors kept;
//! - the bytes a blit reads and writes after a release, the CPU's writes made while released among
//!   them;
//! - that a released import reads after a remap the pages the kept one would read;
//! - that a buffer whose Metal address was handed out stays imported.
//!
//! A device whose driver is not measured with the fact keeps every pointer import: the hook
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on Linux, on the kind of device the cfg selects.

use super::host_residency::{
    counting, device, gpu_fill, gpu_read, host_page, read, release, residency, Ram, Scratch,
};
use crate::{Buffer, Device, Error, HostMemory, ResourceOptions};
use metal2vulkan::device::DriverFact;
use metal2vulkan::error::Feature;

/// Whether `device`'s driver is measured to have its pointer imports follow their address, so
/// that the device releases them while idle.
pub(super) fn follows(device: &Device) -> bool {
    (device.properties().driver_facts).contains(&DriverFact::PointerImportsFollowTheirAddress)
}

/// Run `test` on the device the cfg selects.
fn on_device(test: impl FnOnce(&Device)) {
    let scratch = Scratch::new();
    let device = device(&scratch.0);
    test(&device);
}

        }
}

#[test]
fn a_kept_pointer_import_reads_the_pages_a_remap_put_at_its_address() {
    const TEST: &str = "pointer import remap";
    on_device(|device| {
        let page = host_page();
        let len = 2 * page;
        let queue = device.new_command_queue().unwrap();
        let old = counting(len, 13);
        let new = counting(len, 131);
            return;
        };
        assert!(read(&queue, &buffer, len) == old, "{TEST}");

        let bytes = read(&queue, &buffer, len);
            _ if bytes == new => "the new pages",
            _ if bytes == old => "the pages it was made over",
            _ => "neither",
        };
        eprintln!(
            device.name(),
            follows(device),
        );
        if follows(device) {
            assert_eq!(residency(&device.work_counts()).0, 0, "{TEST}: no release");
        }
        drop(buffer);
    });
}

/// A no-copy buffer over 4 pages, imported by its pointer, on a device whose driver's pointer
/// imports follow their address:
/// - read by a blit, its import released: the device's live host imports fall by one, no
///   descriptor is kept, and the hook releases it no further while released;
/// - released again, a GPU fill lands in the pages;
/// - released again, its Metal address handed out imports it again at once and for good, and the
///   hook releases it no more.
///
/// Elsewhere the hook releases nothing and the counts stay 0.
#[test]
fn an_idle_pointer_import_is_released_and_imported_again_at_its_next_blit() {
    const TEST: &str = "pointer residency";
    on_device(|device| {
        let page = host_page();
        let len = 4 * page;
        let queue = device.new_command_queue().unwrap();
        let first = counting(len, 7);
            return;
        };
        assert!(read(&queue, &buffer, len) == first, "{TEST}");
        if !follows(device) {
            assert_eq!(device.release_idle_host_imports(), 0, "{TEST}");
            assert_eq!(residency(&device.work_counts()), (0, 0, 0, 0), "{TEST}");
            eprintln!("{TEST}: on {}: the pointer import kept", device.name());
            return;
        }
        let imported = device.work_counts();
        assert_eq!(residency(&imported), (0, 0, 1, 0), "{TEST}");
        let live = imported.host_imports_live;

        release(TEST, device, 1);
        let released = device.work_counts();
        assert_eq!(residency(&released), (1, 0, 0, 0), "{TEST}: released");
        assert_eq!(released.host_imports_live, live - 1, "{TEST}");
        assert_eq!(
            device.release_idle_host_imports(),
            0,
            "{TEST}: released twice"
        );

        let second = counting(len, 59);
        let contents = buffer.contents().unwrap();
        // SAFETY: the buffer's own bytes, which no command reads now.
        unsafe { std::ptr::copy_nonoverlapping(second.as_ptr(), contents.as_ptr(), len / 2) };
        let (bytes, importing) = gpu_read(&queue, &buffer, len);
        assert!(bytes == second, "{TEST}: the bytes written while released");
        let (bytes, imported) = gpu_read(&queue, &buffer, len);
        assert!(bytes == second, "{TEST}");
        let again = device.work_counts();
        assert_eq!(residency(&again), (1, 1, 1, 0), "{TEST}: imported again");
        assert_eq!(again.host_imports_live, live, "{TEST}");

        // A GPU write after a release lands in the pages.
        release(TEST, device, 1);
        gpu_fill(&queue, &buffer, page as u64..3 * page as u64, 0x6d);
        let mut filled = second.clone();
        filled[page..3 * page].fill(0x6d);
        let counts = device.work_counts();
        assert_eq!(residency(&counts), (2, 2, 1, 0), "{TEST}: filled");

        // Its Metal address handed out keeps it imported for good, from the moment it is.
        release(TEST, device, 1);
        let address = buffer.gpu_address();
        assert_ne!(address.0, 0, "{TEST}");
        let counts = device.work_counts();
        assert_eq!(residency(&counts), (3, 3, 1, 0), "{TEST}: exposed");
        assert_eq!(counts.host_imports_live, live, "{TEST}");
        assert!(read(&queue, &buffer, len) == filled, "{TEST}");
        assert_eq!(device.release_idle_host_imports(), 0, "{TEST}: exposed");
        drop(buffer);
        eprintln!(
            "{TEST}: on {}: released 3 times, imported again 3 times; a copy's call took \
             {importing:?} importing it again, {imported:?} imported",
            device.name()
        );
    });
}

/// (`a_kept_pointer_import_reads_the_pages_a_remap_put_at_its_address`), so the release changed
/// nothing a command reads.
#[test]
fn a_released_pointer_import_reads_after_a_remap_what_the_kept_one_would() {
    const TEST: &str = "pointer residency remap";
    on_device(|device| {
        if !follows(device) {
            eprintln!("{TEST}: on {}: pointer imports kept", device.name());
            return;
        }
        let page = host_page();
        let len = 2 * page;
        let queue = device.new_command_queue().unwrap();
        let old = counting(len, 13);
        let new = counting(len, 131);
            return;
        };
        assert!(read(&queue, &buffer, len) == old, "{TEST}");
        release(TEST, device, 1);

        assert!(
            read(&queue, &buffer, len) == new,
            "{TEST}: the released import"
        );
        let counts = device.work_counts();
        assert_eq!(residency(&counts), (1, 1, 1, 0), "{TEST}: imported again");
        drop(buffer);
        eprintln!(
            "{TEST}: on {}: the released import read the new pages, as the kept one does",
            device.name()
        );
    });
}
