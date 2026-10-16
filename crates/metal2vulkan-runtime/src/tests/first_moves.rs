//! The work host calls stage for the device's next submission (`texture/first_move.rs`).

use crate::texture::first_move::{FirstMoves, TexelCopy};
use ash::vk;
use std::any::Any;
use std::sync::{Arc, Weak};

/// A host call's copy staged into an image makes a later CPU copy of its texels wait for the
/// submission that makes it; a decode alone staged (`TexelCopy::decode_only`: of what a copy on
/// the host or the staging queue wrote into a BC image, into its decoded twin) does not, as it
/// writes the twin alone (`FirstMoves::texels_wait`).
#[test]
fn a_staged_decode_alone_leaves_the_texels_to_cpu_copies() {
    let firsts = FirstMoves::default();
    let decoding: Arc<u32> = Arc::new(0);
    let copied: Arc<u32> = Arc::new(1);
    let keeper = |image: &Arc<u32>| -> Weak<dyn Any + Send + Sync> {
        let weak: Weak<u32> = Arc::downgrade(image);
        weak
    };
    let at = |image: &Arc<u32>| Arc::as_ptr(image).cast::<()>();
    // A decode's shape, its pass aside: no copy of texels.
    let alone = TexelCopy::new(vk::Buffer::null(), 0, Vec::new(), Box::new(()));
    assert!(!firsts.register_texels(keeper(&decoding), alone));
    let copy = (vk::Image::null(), vk::BufferImageCopy::default());
    let staged = TexelCopy::new(vk::Buffer::null(), 16, vec![copy], Box::new(()));
    assert!(!firsts.register_texels(keeper(&copied), staged));
    assert!(!firsts.texels_wait(at(&decoding)));
    assert!(firsts.texels_wait(at(&copied)));
}

/// The runs of changed bytes are exact: each as long as it runs, across the comparison's 64-byte
/// blocks, and none where nothing changed.
#[test]
fn the_changed_runs_are_the_bytes_that_differ() {
    use crate::texture::first_move::changed_runs;
    let kept: Vec<u8> = (0..300u32).map(|i| (i * 7) as u8).collect();
    assert_eq!(changed_runs(&kept, &kept), Vec::new());
    let mut now = kept.clone();
    for at in [0, 1, 2, 63, 64, 130, 299] {
        now[at] ^= 0x5a;
    }
    assert_eq!(
        changed_runs(&kept, &now),
        vec![0..3, 63..65, 130..131, 299..300]
    );
    let mut now = kept.clone();
    now[10..250].iter_mut().for_each(|byte| *byte = !*byte);
    assert_eq!(changed_runs(&kept, &now), vec![10..250]);
}

/// Fresh private pages, `pages` of them, read and write, holding `fill`; unmapped as it drops.
#[cfg(any(target_os = "linux", target_vendor = "apple"))]
struct Pages {
    at: *mut u8,
    len: usize,
    page: usize,
}

#[cfg(any(target_os = "linux", target_vendor = "apple"))]
impl Pages {
    fn new(pages: usize, fill: u8) -> Pages {
        // SAFETY: `sysconf` reads a system constant.
        let page = usize::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) }).unwrap();
        let len = pages * page;
        // SAFETY: a fresh private mapping, unmapped as `Pages` drops.
        let at = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            )
        };
        assert_ne!(at, libc::MAP_FAILED);
        let at = at.cast::<u8>();
        // SAFETY: the mapping's own bytes.
        unsafe { at.write_bytes(fill, len) };
        Pages { at, len, page }
    }

    /// Make page `index` read-only.
    fn read_only(&self, index: usize) {
        // SAFETY: a page of this mapping.
        let done = unsafe {
            libc::mprotect(
                self.at.add(index * self.page).cast(),
                self.page,
                libc::PROT_READ,
            )
        };
        assert_eq!(done, 0, "mprotect");
    }

    fn bytes(&self) -> Vec<u8> {
        // SAFETY: the mapping's own bytes, readable.
        unsafe { std::slice::from_raw_parts(self.at, self.len) }.to_vec()
    }

    fn write(&self, at: usize, byte: u8) {
        // SAFETY: a byte of the mapping, on a page still writable.
        unsafe { self.at.add(at).write(byte) };
    }
}

#[cfg(any(target_os = "linux", target_vendor = "apple"))]
impl Drop for Pages {
    fn drop(&mut self) {
        // SAFETY: the mapping is this value's.
        unsafe { libc::munmap(self.at.cast(), self.len) };
    }
}

/// The kernel's answer for the process's own pages: a fresh mapping is writable until a page of it
/// is made read-only, and pages no longer mapped are not (`host_memory::host_writable`, on Linux
/// and Apple's systems).
#[cfg(any(target_os = "linux", target_vendor = "apple"))]
#[test]
fn the_kernel_says_which_of_the_process_s_pages_are_writable() {
    use crate::host_memory::host_writable;
    let pages = Pages::new(3, 0);
    let (at, page) = (pages.at as u64, pages.page as u64);
    assert_eq!(host_writable(at, at + 3 * page), Some(true));
    assert_eq!(host_writable(at + 5, at + 6), Some(true));
    pages.read_only(1);
    assert_eq!(host_writable(at, at + page), Some(true));
    assert_eq!(host_writable(at, at + 3 * page), Some(false));
    assert_eq!(host_writable(at + page + 9, at + page + 10), Some(false));
    assert_eq!(host_writable(at + 2 * page, at + 3 * page), Some(true));
    drop(pages);
    assert_eq!(host_writable(at, at + page), Some(false));
}

/// The write back of a first move's kept bytes (`KeptBytes::write_back`, through `put_back`)
/// stores only what the move changed. Over read-only pages it changed nothing on, it stores
/// nothing, so it cannot fault: the reims-vgpu crash (`SIGBUS` on MoltenVK, a texture over guest
/// pages the host maps read-only) stored every byte. Changed bytes on writable pages are put back;
/// changed bytes on a read-only page stay as they are and are counted, which the caller announces
/// (`Degradation::UnrestoredFirstMoveBytes`, a site no lane can meet: no driver's move is made to
/// change bytes on purpose).
#[cfg(any(target_os = "linux", target_vendor = "apple"))]
#[test]
fn the_write_back_stores_only_what_the_move_changed_and_only_where_the_host_may_write() {
    use crate::texture::first_move::{put_back, Restored};
    // Unchanged, read-only: nothing stored, no fault.
    let pages = Pages::new(2, 0x3c);
    let kept = pages.bytes();
    pages.read_only(0);
    pages.read_only(1);
    // SAFETY: the mapping's bytes, readable; nothing else writes them.
    let restored = unsafe { put_back(pages.at, &kept) };
    assert_eq!(restored, Restored::default());
    assert_eq!(pages.bytes(), kept);
    // Changed on writable pages: every changed byte put back, and only those counted.
    let pages = Pages::new(2, 0x3c);
    let kept = pages.bytes();
    for at in [0, 1, 70, pages.page + 3] {
        pages.write(at, 0xa5);
    }
    // SAFETY: as above.
    let restored = unsafe { put_back(pages.at, &kept) };
    assert_eq!(
        restored,
        Restored {
            stored: 4,
            unwritable: 0
        }
    );
    assert_eq!(pages.bytes(), kept);
    // Changed on both pages, the second then made read-only: the first's put back, the second's
    // left and counted.
    let pages = Pages::new(2, 0x3c);
    let kept = pages.bytes();
    for at in [5, pages.page + 7, pages.page + 8] {
        pages.write(at, 0xa5);
    }
    pages.read_only(1);
    // SAFETY: as above.
    let restored = unsafe { put_back(pages.at, &kept) };
    assert_eq!(
        restored,
        Restored {
            stored: 1,
            unwritable: 2
        }
    );
    let now = pages.bytes();
    assert_eq!(now[5], 0x3c);
    assert_eq!(&now[pages.page + 7..pages.page + 9], &[0xa5, 0xa5]);
}
