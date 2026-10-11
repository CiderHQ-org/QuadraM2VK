//! Buffers over host memory (`host_memory.rs`): the pages an import covers, where the buffer
//! binds in them, the memory types an import may use, and the host's page.

use crate::host_memory::{
    based_size, binds_in, buffer_base, host_page, import_refused, import_window, ImportWindow,
    WindowRefusal,
};
use crate::memory::{candidates, MemoryClass};
use crate::{Error, MemoryKind, VkResultCode};
use ash::vk;
use metal2vulkan::error::{Feature, Unsupported};

const PAGE: u64 = 16 << 10;

fn requirements(size: u64, alignment: u64) -> vk::MemoryRequirements {
    vk::MemoryRequirements {
        size,
        alignment,
        memory_type_bits: !0,
    }
}

#[test]
fn a_page_aligned_range_imports_exactly_its_pages() {
    // The consumer's shapes: a guest mapping at a page, a whole number of pages.
    let at = 0x1_2340_0000;
    assert_eq!(
        import_window(at, 3 * PAGE, PAGE, PAGE),
        Ok(ImportWindow {
            start: at,
            size: 3 * PAGE,
            offset: 0,
        })
    );
    // A 4-KiB host whose device imports at 4 KiB takes a 16-KiB-aligned mapping as it is.
    assert_eq!(
        import_window(at, PAGE, 4096, 4096),
        Ok(ImportWindow {
            start: at,
            size: PAGE,
            offset: 0,
        })
    );
}

#[test]
fn an_unaligned_range_imports_the_pages_it_touches() {
    // Metal makes these: the buffer covers the whole pages of its range.
    let page = 0x1_0000_0000;
    for (offset, length, pages) in [
        (1, 4096, 1),
        (16, 100, 1),
        (4096, PAGE, 2),
        (PAGE - 8, 16, 2),
        (0, 100, 1),
        (0, PAGE + 4096, 2),
    ] {
        let window = import_window(page + offset, length, PAGE, PAGE).unwrap();
        assert_eq!(window.start, page, "{offset} {length}");
        assert_eq!(window.size, pages * PAGE, "{offset} {length}");
        assert_eq!(window.offset, offset, "{offset} {length}");
    }
}

#[test]
fn an_import_alignment_above_the_page_rounds_only_within_the_range_s_pages() {
    // A device importing at 64 KiB on a 4-KiB host: a range aligned to 64 KiB imports as it is;
    // any other would take pages the caller did not vouch for.
    let unit = 64 << 10;
    let at = 0x2_0000_0000;
    assert_eq!(
        import_window(at, unit, 4096, unit),
        Ok(ImportWindow {
            start: at,
            size: unit,
            offset: 0,
        })
    );
    assert_eq!(
        import_window(at + 4096, unit - 4096, 4096, unit),
        Err(WindowRefusal::BeyondPages)
    );
    assert_eq!(
        import_window(at, 4096, 4096, unit),
        Err(WindowRefusal::BeyondPages)
    );
}

#[test]
fn a_range_past_the_address_space_is_refused() {
    assert_eq!(
        import_window(u64::MAX - 100, 200, PAGE, PAGE),
        Err(WindowRefusal::Overflow)
    );
    assert_eq!(
        import_window(u64::MAX - PAGE + 1, 100, PAGE, PAGE),
        Err(WindowRefusal::Overflow),
        "the last page's end does not fit in 64 bits"
    );
}

#[test]
fn the_buffer_starts_below_its_bytes_at_its_alignment_and_at_16_bytes() {
    // The pointer's offset modulo the alignment, or modulo 16 where that is larger.
    assert_eq!(buffer_base(0, 256), 0);
    assert_eq!(buffer_base(256, 256), 0);
    assert_eq!(buffer_base(16, 256), 16);
    assert_eq!(buffer_base(PAGE - 8, 64), 56);
    assert_eq!(buffer_base(1, 4), 1);
    assert_eq!(buffer_base(20, 4), 4);
    assert_eq!(buffer_base(16, 16), 0);
    // The `VkBuffer` reaches the padded length past its base, rounded to 16 bytes.
    assert_eq!(based_size(0, 100), Some(112));
    assert_eq!(based_size(16, 4096), Some(4112));
    assert_eq!(based_size(1, 16), Some(32));
    assert_eq!(based_size(56, 16), Some(80));
    assert_eq!(based_size(8, u64::MAX - 4), None);
}

#[test]
fn the_buffer_binds_below_its_bytes_inside_the_window() {
    let window = |offset| ImportWindow {
        start: 0,
        size: PAGE,
        offset,
    };
    // At the pointer's offset where the alignment allows.
    assert_eq!(
        binds_in(&window(0), 0, PAGE, &requirements(PAGE, 256)),
        Ok(0)
    );
    assert_eq!(
        binds_in(&window(256), 0, 4096, &requirements(4096, 256)),
        Ok(256)
    );
    assert_eq!(
        binds_in(&window(16), 0, 112, &requirements(112, 16)),
        Ok(16)
    );
    // Below it otherwise, by the base: Metal's offset-1 and offset-16 shapes.
    assert_eq!(binds_in(&window(1), 1, 4112, &requirements(4112, 4)), Ok(0));
    assert_eq!(
        binds_in(&window(16), 16, 128, &requirements(128, 64)),
        Ok(0)
    );
    assert_eq!(
        binds_in(&window(PAGE - 8), 56, 80, &requirements(80, 64)),
        Err(WindowRefusal::Binding),
        "the bytes run past the window's page"
    );
    // A base that does not reach the alignment below the bytes binds nowhere.
    assert_eq!(
        binds_in(&window(16), 0, 112, &requirements(112, 64)),
        Err(WindowRefusal::Binding)
    );
    assert_eq!(
        binds_in(&window(8), 16, 112, &requirements(112, 16)),
        Err(WindowRefusal::Binding)
    );
    // The buffer, or the driver's size for it, runs past the pages.
    assert_eq!(
        binds_in(&window(PAGE - 16), 0, 32, &requirements(32, 16)),
        Err(WindowRefusal::Binding)
    );
    assert_eq!(
        binds_in(&window(PAGE - 16), 0, 16, &requirements(64, 16)),
        Err(WindowRefusal::Binding)
    );
    assert_eq!(
        binds_in(&window(PAGE - 48), 16, 48, &requirements(80, 64)),
        Err(WindowRefusal::Binding)
    );
    assert_eq!(
        binds_in(&window(PAGE - 48), 16, 48, &requirements(48, 64)),
        Ok(PAGE - 64)
    );
}

#[test]
fn imported_memory_is_host_visible_and_coherent_of_the_storage_mode_s_class() {
    use vk::MemoryPropertyFlags as P;
    let ty = |property_flags| vk::MemoryType {
        property_flags,
        heap_index: 0,
    };
    // NVIDIA's measured types for a host pointer: 2 (coherent) and 3 (also cached).
    let types = [
        ty(P::DEVICE_LOCAL),
        ty(P::DEVICE_LOCAL | P::HOST_VISIBLE | P::HOST_COHERENT),
        ty(P::HOST_VISIBLE | P::HOST_COHERENT),
        ty(P::HOST_VISIBLE | P::HOST_COHERENT | P::HOST_CACHED),
        ty(P::HOST_VISIBLE | P::HOST_CACHED),
    ];
    let pointer = 0b11100;
    // Either storage mode prefers the cached type, as the caller's pages are cached.
    for class in [MemoryClass::Shared, MemoryClass::Managed] {
        assert_eq!(
            candidates(&types, pointer, class, false),
            [3, 2],
            "{class:?}"
        );
    }
    // A pointer only importable as non-coherent memory has no type: refused, never flushed.
    assert!(candidates(&types, 0b10000, MemoryClass::Shared, false).is_empty());
}

#[cfg(unix)]
#[test]
fn the_host_page_is_known_on_unix() {
    let page = host_page().unwrap();
    assert!(page.is_power_of_two() && page >= 4096, "{page}");
}

#[cfg(not(unix))]
#[test]
fn the_host_page_is_unknown_elsewhere() {
    assert_eq!(host_page(), None);
}

/// Lines of `/proc/<pid>/maps` (address order) for the writability check.
#[cfg(target_os = "linux")]
fn maps(lines: &[&str]) -> Vec<String> {
    lines.iter().map(|line| line.to_string()).collect()
}

#[cfg(target_os = "linux")]
#[test]
fn only_pages_mapped_writable_without_a_gap_are_importable() {
    use crate::host_memory::writable;
    let lines = maps(&[
        "1000-3000 rw-p 00000000 00:00 0",
        "3000-5000 rw-s 00000000 00:01 7   /dev/shm/guest (deleted)",
        "5000-6000 r--p 00000000 00:00 0",
        "7000-8000 rwxp 00000000 00:00 0   [stack]",
        "not a mapping line",
        "8000-9000 rw-p 00000000 00:00 0",
    ]);
    // Inside one mapping, and across two adjacent writable ones (private and shared).
    assert!(writable(lines.clone(), 0x1000, 0x3000));
    assert!(writable(lines.clone(), 0x2000, 0x5000));
    // A read-only page, alone or at the end of a range.
    assert!(!writable(lines.clone(), 0x5000, 0x6000));
    assert!(!writable(lines.clone(), 0x4000, 0x6000));
    // A gap (0x6000-0x7000 is unmapped), and pages past the last mapping.
    assert!(!writable(lines.clone(), 0x6000, 0x7000));
    assert!(!writable(lines.clone(), 0x7000, 0xa000));
    // A line that does not parse is skipped.
    assert!(writable(lines.clone(), 0x7000, 0x9000));
    // Nothing mapped at all.
    assert!(!writable(Vec::new(), 0x1000, 0x2000));
}

/// The check reads this process's own mappings: a fresh page is importable until it is made
/// read-only, and a page no longer mapped is not.
#[cfg(target_os = "linux")]
#[test]
fn this_process_s_read_only_and_unmapped_pages_are_refused() {
    use crate::error::Error;
    use crate::host_memory::refuse_unwritable;
    let page = host_page().unwrap();
    let len = page as usize;
    // SAFETY: a fresh private mapping, unmapped below.
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
    let window = ImportWindow {
        start: at as u64,
        size: page,
        offset: 0,
    };
    let refused = |result: Result<(), Error>| match result {
        Err(Error::Unsupported(unsupported)) => {
            unsupported.feature == metal2vulkan::error::Feature::HostMemory
        }
        _ => false,
    };
    assert_eq!(refuse_unwritable(&window), Ok(()));
    // SAFETY: the mapping is this test's.
    assert_eq!(unsafe { libc::mprotect(at, len, libc::PROT_READ) }, 0);
    assert!(refused(refuse_unwritable(&window)));
    // SAFETY: as above.
    assert_eq!(unsafe { libc::munmap(at, len) }, 0);
    assert!(refused(refuse_unwritable(&window)));
}

/// The kernel's answer for a window (`PROCMAP_QUERY`) is the maps lines' answer, over one
/// mapping, two of different permissions, a read-only page alone, and a page no longer mapped;
/// and a kernel of Linux 6.11 or later answers it, so no import reads the maps' text. Both
/// answers read the address space of a child forked for each state ([`window_answers`]), which
/// no other thread changes: in this process, another test's mapping may take the unmapped page
/// between the two readings.
#[cfg(target_os = "linux")]
#[test]
fn the_kernel_s_answer_for_a_window_is_the_maps_lines_answer() {
    let page = host_page().unwrap();
    let len = 3 * page as usize;
    // SAFETY: a fresh private mapping, unmapped below.
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
    let base = at as u64;
    let release = {
        // SAFETY: `utsname` is plain data, all zeros a valid value.
        let mut name: libc::utsname = unsafe { std::mem::zeroed() };
        // SAFETY: the call fills the structure.
        assert_eq!(unsafe { libc::uname(&mut name) }, 0);
        // SAFETY: the kernel ends the release with a NUL inside the field.
        unsafe { std::ffi::CStr::from_ptr(name.release.as_ptr()) }
            .to_string_lossy()
            .into_owned()
    };
    let mut numbers = release
        .split(|c: char| !c.is_ascii_digit())
        .filter_map(|n| n.parse::<u32>().ok());
    let version = (numbers.next().unwrap_or(0), numbers.next().unwrap_or(0));
    let queries = version >= (6, 11);
    let windows = [(0, 1), (0, 2), (1, 2), (1, 3), (0, 3), (2, 3)];
    let spans = windows.map(|(from, to)| (base + from * page, base + to * page));
    let check = |state: &str| {
        for ((from, to), (read, queried)) in windows.into_iter().zip(window_answers(&spans)) {
            if queries {
                assert_eq!(
                    queried,
                    Some(read),
                    "{state}: pages {from}..{to} on {release}"
                );
            } else if let Some(queried) = queried {
                assert_eq!(queried, read, "{state}: pages {from}..{to} on {release}");
            }
        }
    };
    check("all writable");
    let page_at = |index: u64| at.cast::<u8>().wrapping_add((index * page) as usize).cast();
    // SAFETY: the mapping is this test's; its middle page becomes read-only.
    let protected = unsafe { libc::mprotect(page_at(1), page as usize, libc::PROT_READ) };
    assert_eq!(protected, 0);
    check("the middle page read-only");
    // SAFETY: as above; its last page goes.
    assert_eq!(unsafe { libc::munmap(page_at(2), page as usize) }, 0);
    check("the last page unmapped");
    // SAFETY: as above; the rest goes.
    assert_eq!(unsafe { libc::munmap(at, 2 * page as usize) }, 0);
}

/// The answers for windows `[start, end)` in a child forked now: the maps lines' answer
/// ([`writable`](crate::host_memory::writable)) and the kernel's
/// ([`queried_writable`](crate::host_memory::queried_writable)), from one snapshot. The child's
/// address space is this process's at the fork, and its one thread maps nothing. It asks the
/// kernel and copies its maps text to a pipe without allocating (a fork of a multithreaded
/// process leaves only async-signal-safe calls safe; another thread may have held `malloc`'s lock)
/// and ends with `_exit`, which runs no exit handler; this process reads the lines' answers from
/// the copy.
#[cfg(target_os = "linux")]
fn window_answers<const N: usize>(windows: &[(u64, u64); N]) -> [(bool, Option<bool>); N] {
    use crate::host_memory::writable;
    use std::io::Read;
    use std::os::fd::FromRawFd;
    let mut pipe = [0; 2];
    // SAFETY: `pipe` holds the two descriptors the call makes, closed on any `exec` (another
    // test's child process does not keep the write end open).
    assert_eq!(
        unsafe { libc::pipe2(pipe.as_mut_ptr(), libc::O_CLOEXEC) },
        0
    );
    // SAFETY: the child runs only `send_window_answers` (system calls into stack buffers) and
    // `_exit`.
    let child = unsafe { libc::fork() };
    if child == 0 {
        let code = send_window_answers(pipe[1], windows);
        // SAFETY: ends the child at once, running no exit handler.
        unsafe { libc::_exit(code) };
    }
    assert!(child > 0, "fork");
    // SAFETY: the parent's copy of the write end, closed once, so the read ends at the child's.
    assert_eq!(unsafe { libc::close(pipe[1]) }, 0);
    // SAFETY: the parent's copy of the read end, which the `File` owns and closes.
    let mut source = unsafe { std::fs::File::from_raw_fd(pipe[0]) };
    let mut sent = Vec::new();
    let read = source.read_to_end(&mut sent);
    let mut status = 0;
    // SAFETY: `child` is this process's child.
    assert_eq!(unsafe { libc::waitpid(child, &mut status, 0) }, child);
    assert!(
        libc::WIFEXITED(status) && libc::WEXITSTATUS(status) == 0,
        "the child: {status}"
    );
    read.unwrap();
    let (answers, text) = sent.split_at(N);
    let lines: Vec<String> = String::from_utf8_lossy(text)
        .lines()
        .map(str::to_owned)
        .collect();
    std::array::from_fn(|i| {
        let (start, end) = windows[i];
        let queried = (answers[i] & 1 != 0).then_some(answers[i] & 2 != 0);
        (writable(lines.iter().cloned(), start, end), queried)
    })
}

/// The forked child's side of [`window_answers`], which allocates nothing: one byte per window
/// for the kernel's answer (1 where it answers, plus 2 where that is writable), then the maps
/// text, written to `sink`. The exit code: 0, or what failed (2 the open, 3 a write, 4 a read).
#[cfg(target_os = "linux")]
fn send_window_answers<const N: usize>(sink: i32, windows: &[(u64, u64); N]) -> i32 {
    use crate::host_memory::queried_writable;
    use std::os::fd::FromRawFd;
    let send = |mut bytes: &[u8]| {
        while !bytes.is_empty() {
            // SAFETY: writes `bytes`, of their length.
            let wrote = unsafe { libc::write(sink, bytes.as_ptr().cast(), bytes.len()) };
            if wrote > 0 {
                bytes = &bytes[wrote as usize..];
            } else if wrote == 0
                || std::io::Error::last_os_error().kind() != std::io::ErrorKind::Interrupted
            {
                return false;
            }
        }
        true
    };
    // SAFETY: a NUL-terminated path.
    let fd = unsafe {
        libc::open(
            c"/proc/self/maps".as_ptr(),
            libc::O_RDONLY | libc::O_CLOEXEC,
        )
    };
    if fd < 0 {
        return 2;
    }
    // SAFETY: `fd` is the file just opened, which the `File` owns from here.
    let maps = unsafe { std::fs::File::from_raw_fd(fd) };
    let answers = windows.map(|(start, end)| match queried_writable(&maps, start, end) {
        None => 0,
        Some(writable) => 1 | (u8::from(writable) << 1),
    });
    if !send(&answers) {
        return 3;
    }
    let mut chunk = [0_u8; 4096];
    loop {
        // SAFETY: reads into `chunk`, of its length.
        let read = unsafe { libc::read(fd, chunk.as_mut_ptr().cast(), chunk.len()) };
        if read > 0 {
            if !send(&chunk[..read as usize]) {
                return 3;
            }
        } else if read == 0 {
            return 0;
        } else if std::io::Error::last_os_error().kind() != std::io::ErrorKind::Interrupted {
            return 4;
        }
    }
}

#[test]
fn a_driver_s_refused_import_is_a_device_gap_and_other_failures_are_vulkan_s() {
    let refused = Error::Unsupported(Unsupported::not_representable(Feature::HostMemory));
    // Mesa 26.2.4's RADV refuses shared memory as the extension says; 26.0.8's answers UNKNOWN.
    // Both take the caller's copy fallback.
    assert_eq!(
        import_refused(vk::Result::ERROR_INVALID_EXTERNAL_HANDLE),
        refused
    );
    assert_eq!(import_refused(vk::Result::ERROR_UNKNOWN), refused);
    assert_eq!(
        import_refused(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY),
        Error::OutOfMemory(MemoryKind::Device)
    );
    assert_eq!(
        import_refused(vk::Result::ERROR_INITIALIZATION_FAILED),
        Error::Vulkan(VkResultCode(
            vk::Result::ERROR_INITIALIZATION_FAILED.as_raw()
        ))
    );
}
