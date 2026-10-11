//! A metallib's function count is untrusted input and must not size a reservation.
//!
//! The consumer parses libraries handed over by a guest, so the header's function count is chosen
//! by whoever built the file. The only cheap bound on it counts the smallest possible entry (a size
//! word and `ENDT`), while a parsed function is two orders of magnitude larger in memory than that.
//! Reserving from the count therefore let a crafted file claim a few hundred bytes of memory per
//! byte of input before any entry was read. This checks that a huge claimed count over a large list
//! region fails with a typed error while the parser's peak allocation stays independent of the
//! count.
//!
//! The single `#[test]` is deliberate: the allocator counters are process-wide.

use metal2vulkan::error::{ContainerError, Error, InputError, InputErrorKind};
use metal2vulkan::library::Library;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            let live = LIVE.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            PEAK.fetch_max(live, Ordering::Relaxed);
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let grown = unsafe { System.realloc(pointer, layout, new_size) };
        if !grown.is_null() {
            let live = LIVE
                .fetch_add(new_size, Ordering::Relaxed)
                .saturating_add(new_size)
                .saturating_sub(layout.size());
            LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
            PEAK.fetch_max(live, Ordering::Relaxed);
        }
        grown
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// A metallib whose function list spans `list_len` zero bytes and claims the most functions the
/// list-size bound admits. Every other section is empty.
fn huge_count_library(list_len: usize) -> Vec<u8> {
    let list_off = 0x58usize;
    let file_size = list_off + 4 + list_len;
    let mut out = Vec::with_capacity(file_size);
    out.extend_from_slice(b"MTLB");
    out.extend_from_slice(&0x8001u16.to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&9u16.to_le_bytes());
    out.push(0); // executable
    out.push(0x81);
    out.extend_from_slice(&26u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    for v in [
        file_size, list_off, list_len, list_off, 0, list_off, 0, list_off, 0,
    ] {
        out.extend_from_slice(&(v as u64).to_le_bytes());
    }
    assert_eq!(out.len(), list_off);
    let count = u32::try_from(list_len / 8).unwrap();
    out.extend_from_slice(&count.to_le_bytes());
    out.resize(file_size, 0);
    out
}

#[test]
fn a_huge_function_count_reserves_nothing_in_proportion() {
    // 4 MiB of list claims 512 Ki functions; reserving for them would take well over 100 MiB.
    let list_len = 4 << 20;
    let bytes = huge_count_library(list_len);

    let before = LIVE.load(Ordering::Relaxed);
    PEAK.store(before, Ordering::Relaxed);
    let result = Library::parse(&bytes);
    let peak = PEAK.load(Ordering::Relaxed).saturating_sub(before);

    assert!(
        matches!(
            &result,
            Err(Error::InvalidInput(InputError {
                kind: InputErrorKind::Container(ContainerError::BadTag),
                ..
            }))
        ),
        "a list of zero-sized entries must fail as a bad entry, got {result:?}"
    );
    drop(result);
    // Entry 0 is rejected before anything is pushed, so the parser allocates next to nothing. The
    // bound is far below the input size, let alone the count-proportional reservation.
    assert!(
        peak < 64 << 10,
        "parsing reserved {peak} bytes for a {list_len}-byte function list"
    );
}
