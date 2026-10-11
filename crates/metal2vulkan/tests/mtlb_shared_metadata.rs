//! Metallib metadata records are untrusted input and must not be parsed once per referencing
//! function.
//!
//! Every function-table entry names its public and private metadata records by offset. If many
//! entries could name one large record, each function would hold its own parsed copy of the
//! record's tags, and memory would grow with the product of the entry count and the record size:
//! quadratic in the input. This checks that entries sharing one record fail with a typed error
//! while the parser's peak allocation stays within a small constant multiple of the input size.
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

/// SHA-256 of the empty string: every function's bitcode is empty.
const EMPTY_SHA256: [u8; 32] = [
    0xe3, 0xb0, 0xc4, 0x42, 0x98, 0xfc, 0x1c, 0x14, 0x9a, 0xfb, 0xf4, 0xc8, 0x99, 0x6f, 0xb9, 0x24,
    0x27, 0xae, 0x41, 0xe4, 0x64, 0x9b, 0x93, 0x4c, 0xa4, 0x95, 0x99, 0x1b, 0x78, 0x52, 0xb8, 0x55,
];

fn tag(out: &mut Vec<u8>, fourcc: &[u8; 4], bytes: &[u8]) {
    out.extend_from_slice(fourcc);
    out.extend_from_slice(&(bytes.len() as u16).to_le_bytes());
    out.extend_from_slice(bytes);
}

/// A metallib of `functions` entries with only the required tags and empty bitcode. Every entry
/// names the same public-metadata record, a list of `tags` empty unknown tags, and the same
/// private-metadata record.
fn shared_record_library(functions: usize, tags: usize) -> Vec<u8> {
    let mut entries = Vec::new();
    for i in 0..functions {
        let mut list = Vec::new();
        let mut name = format!("f{i}").into_bytes();
        name.push(0);
        tag(&mut list, b"NAME", &name);
        tag(&mut list, b"TYPE", &[2]);
        tag(&mut list, b"HASH", &EMPTY_SHA256);
        tag(&mut list, b"MDSZ", &0u64.to_le_bytes());
        tag(&mut list, b"OFFT", &[0; 24]);
        tag(&mut list, b"VERS", &[2, 0, 8, 0, 4, 0, 0, 0]);
        list.extend_from_slice(b"ENDT");
        entries.extend_from_slice(&((list.len() + 4) as u32).to_le_bytes());
        entries.extend(list);
    }
    entries.extend_from_slice(b"ENDT"); // no extended-header tags

    let record = |n: usize| {
        let mut list = Vec::new();
        for _ in 0..n {
            tag(&mut list, b"XXXX", &[]);
        }
        list.extend_from_slice(b"ENDT");
        let mut out = (list.len() as u32).to_le_bytes().to_vec();
        out.extend(list);
        out
    };
    let public = record(tags);
    let private = record(0);

    let list_off = 0x58usize;
    let list_len = entries.len() - 4; // the extended-header ENDT is not counted
    let pub_off = list_off + 4 + entries.len();
    let priv_off = pub_off + public.len();
    let bc_off = priv_off + private.len();
    let file_size = bc_off;

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
        file_size,
        list_off,
        list_len,
        pub_off,
        public.len(),
        priv_off,
        private.len(),
        bc_off,
        0,
    ] {
        out.extend_from_slice(&(v as u64).to_le_bytes());
    }
    assert_eq!(out.len(), list_off);
    out.extend_from_slice(&(functions as u32).to_le_bytes());
    out.extend(entries);
    out.extend(public);
    out.extend(private);
    assert_eq!(out.len(), file_size);
    out
}

#[test]
fn a_shared_metadata_record_is_rejected_in_linear_memory() {
    // 500 entries sharing a 10k-tag record: parsing a copy per entry took about 188 MiB.
    let bytes = shared_record_library(500, 10_000);

    let before = LIVE.load(Ordering::Relaxed);
    PEAK.store(before, Ordering::Relaxed);
    let result = Library::parse(&bytes);
    let peak = PEAK.load(Ordering::Relaxed).saturating_sub(before);

    assert!(
        matches!(
            &result,
            Err(Error::InvalidInput(InputError {
                kind: InputErrorKind::Container(ContainerError::BadOffset),
                ..
            }))
        ),
        "entries sharing a metadata record must fail as a bad offset, got {:?}",
        result.as_ref().map(|_| "a parsed library")
    );
    drop(result);
    let len = bytes.len();
    assert!(
        peak < 16 * len,
        "parsing peaked at {peak} bytes for a {len}-byte library"
    );
}
