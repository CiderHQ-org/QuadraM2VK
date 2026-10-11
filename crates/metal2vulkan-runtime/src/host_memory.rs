//! Buffers over the caller's host memory without a copy (`newBufferWithBytesNoCopy:`), through
//! `VK_EXT_external_memory_host`.
//!
//! - **What Metal takes** (measured on an Apple M3 Max and an M4, plain and under API validation):
//!   any pointer and any length but 0; the buffer covers the whole host pages its range touches
//!   (`allocatedSize`), its `contents` is the pointer, and its `gpuAddress` keeps the pointer's
//!   offset in its page. `Private` and `Memoryless` storage are stopped.
//! - **The import.** The runtime imports the same pages, rounded out to the larger of the host page
//!   and the device's `minImportedHostPointerAlignment` ([`import_window`]), as one dedicated
//!   `VkDeviceMemory` of a host-visible coherent type both the pointer and the buffer allow, and
//!   binds the buffer at the pointer's offset in them. Pages are mapped whole, so the caller's
//!   guarantee for its bytes covers every page they touch; the GPU can reach the rest of those
//!   pages through an out-of-range access, as on Metal. The `VkBuffer` holds the rest of the
//!   pages from the pointer where the device's requirements of it fit them, and a copy out of
//!   the buffer reads them past its length, the caller's bytes, as plain Metal reads them
//!   (`Buffer::copy_reach`). The import is never a dedicated allocation in Vulkan's sense
//!   (`VkMemoryDedicatedAllocateInfo`), which host memory cannot be.
//! - **What it refuses**, typed: a device without the extension (`Unsupported` naming it), and
//!   what this device cannot import exactly (`Unsupported`, `not_representable`): an import
//!   alignment above the host page whose rounding would take pages the range does not touch, an
//!   offset in the pages that misses the buffer's memory alignment, no memory type both allow, or
//!   a driver that refuses the pointer.
//! - **Pages the process cannot write** are refused typed on Linux, before the import
//!   ([`refuse_unwritable`] asks the kernel for the process's own mappings over the window, with
//!   `PROCMAP_QUERY` on `/proc/self/maps`, or reads that file's lines where the kernel predates
//!   the query, Linux 6.11). Linux drivers pin
//!   user pages for writing: NVIDIA failed a read-only page as out of device memory, RADV refused
//!   it, and ANV imported it and lost the device at the GPU's first read. lavapipe read one, so
//!   there the rule refuses a little more than the driver would. On macOS, MoltenVK hands the
//!   range to Metal, which reads read-only pages, so there is no check.
//! - **What drivers answered**. Private anonymous memory was imported on every venue measured. RADV
//!   refused shared memory (anonymous shared, or a mapping of a shared memory object) typed: Mesa
//!   26.2.4 answers `VK_ERROR_INVALID_EXTERNAL_HANDLE`, 26.0.8 `VK_ERROR_UNKNOWN`, and both are
//!   refusals ([`import_refused`]).
//! - **When**. The pages are imported when the buffer is made, on every platform, as Metal wires
//!   them then: the GPU uses the pages mapped at the pointer when the buffer was made, whatever is
//!   mapped there later, and memory the device cannot import is refused by the call that makes the
//!   buffer, as Metal answers nil there. The import costs work per page there: on macOS MoltenVK's
//!   `newBufferWithBytesNoCopy:` and its residency set's commit, about 11 to 20 ms per GiB (an M4,
//!   an M3 Max); Metal makes the buffer at once and pays at the first command using it (24 to 88 ms
//!   per GiB). The runtime once deferred the macOS import to the buffer's first GPU use, which
//!   saw pages remapped in between and lost the device where the import then failed.
//! - **The deallocator** runs once, with the pointer and the length, when the buffer's last
//!   reference goes: after the imported memory is freed, on the thread that drops it. A committed
//!   command buffer holds the buffer until it completed and its completed handlers ran, so then it
//!   runs on one of the device's handler threads, in the command buffer's queue's context, after
//!   the GPU's last use, as Metal's runs on one of its threads. A
//!   refused call never runs it.
//! - **A keep-alive** ([`KeepAlive`]). The import can outlive the buffer: a texture over the buffer
//!   that a command buffer submitted before its release may still read through its resource ID
//!   keeps its image, and the image the import, until that command buffer completes
//!   (`heap_sets.rs`). Vulkan requires imported pages to stay mapped while the import lives, so
//!   with no keep-alive the deallocator waits for the import's free, announced
//!   (`host-deallocator-delayed`). With a keep-alive, which the caller's mapping lives in, the
//!   deallocator runs at the release, as Metal's does, and the keep-alive is dropped once the
//!   import is freed: the pages must stay mapped until then. A buffer over the bytes another
//!   live buffer lies in shares its import (`host_aliases.rs`), whose pages that buffer's
//!   contract keeps mapped: its deallocator runs at its release either way.

use super::device::lock;
use super::error::{vulkan, Error};
use super::exit_guard;
use super::loader_word::LoaderWord;
use ash::vk;
use metal2vulkan::device::{EnableSet, VkExtension};
use metal2vulkan::error::{Feature, Unsupported};
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Mutex;

/// Host memory a buffer wraps without a copy: a pointer whose bytes the caller vouches for (the
/// `pointer` of `newBufferWithBytesNoCopy:length:options:deallocator:`).
#[derive(Debug)]
pub struct HostMemory(NonNull<u8>);

// SAFETY: a `HostMemory` is an address and the caller's guarantee for it (`HostMemory::new`); the
// runtime touches the bytes only as a buffer of them, under that guarantee.
unsafe impl Send for HostMemory {}
// SAFETY: as above.
unsafe impl Sync for HostMemory {}

impl HostMemory {
    /// The host memory at `pointer`, for a buffer of some length `length` made from it.
    ///
    /// # Safety
    ///
    /// The `length` bytes at `pointer`:
    /// - are mapped memory of this process, readable, and writable wherever the CPU or the GPU
    ///   writes the buffer;
    /// - stay mapped, at this address and with the same backing: with a keep-alive
    ///   ([`KeepAlive`]), until it is dropped; else until the buffer's deallocator runs; with
    ///   neither, until the buffer's last reference is dropped (a command buffer that uses the
    ///   buffer holds a reference until it completed) and every command buffer submitted before
    ///   a texture over the buffer was released completed;
    /// - are not freed, unmapped or remapped by anyone meanwhile.
    ///
    /// Reading or writing them through the pointer while the GPU writes them is a data race, as
    /// Metal's `contents` is.
    ///
    /// On Linux a buffer over a page this process maps without write permission is refused
    /// (`Unsupported`, `HostMemory`), though Metal reads such pages: Linux drivers pin imported
    /// pages for writing, and one (ANV) imported a read-only page and then lost the device at the
    /// GPU's first read. Some drivers also refuse shared memory (RADV refused anonymous shared
    /// memory and mappings of shared memory objects); private memory imports everywhere measured.
    pub unsafe fn new(pointer: NonNull<u8>) -> HostMemory {
        HostMemory(pointer)
    }

    /// The pointer.
    pub fn pointer(&self) -> NonNull<u8> {
        self.0
    }
}

/// A no-copy buffer's deallocator (Metal's block): called once, with the pointer and the length,
/// when the buffer is released. It may run on a runtime thread, as Metal's runs on any thread. A
/// panic inside it is caught.
pub type Deallocator = Box<dyn FnOnce(NonNull<u8>, u64) + Send + 'static>;

/// What keeps the caller's host memory mapped (a runtime extension: Metal has none): given with a
/// no-copy buffer's or heap's memory, it is dropped once the device's import of the pages is
/// freed, which may be after the deallocator ran. With one, the deallocator runs when the buffer
/// is released, as Metal's does, though a command buffer submitted before a texture over the
/// pages was released may still hold the import; with none, the deallocator waits for the import
/// (the module docs). Dropped on a runtime thread, possibly; a panic inside its drop is caught.
/// Never dropped while the process exits: the pages stay imported to its end (`exit_guard`).
pub type KeepAlive = Box<dyn Send + 'static>;

/// What the caller gives back with its host memory: the deallocator, and the keep-alive.
pub(super) struct HostRelease {
    pub(super) deallocator: Option<Deallocator>,
    pub(super) keep_alive: Option<KeepAlive>,
}

impl HostRelease {
    /// The release of `length` bytes at `pointer` to it, where it gives anything.
    pub(super) fn of(self, pointer: NonNull<u8>, length: u64) -> Option<Release> {
        let HostRelease {
            deallocator,
            keep_alive,
        } = self;
        (deallocator.is_some() || keep_alive.is_some()).then(|| Release {
            pointer,
            length,
            deallocator: Mutex::new(deallocator),
            keep_alive: Mutex::new(keep_alive),
        })
    }
}

/// The host pages an import covers: `size` bytes from `start`, the buffer's bytes at `offset`
/// in them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ImportWindow {
    pub(super) start: u64,
    pub(super) size: u64,
    pub(super) offset: u64,
}

/// Why an import window or binding does not exist.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum WindowRefusal {
    /// The range runs past the end of the address space.
    Overflow,
    /// The device's import alignment exceeds the host page, and rounding the range out to it
    /// would take pages the range does not touch.
    BeyondPages,
    /// The buffer, bound at its memory alignment below its bytes ([`buffer_base`]), runs past
    /// the window, or its memory requirements do.
    Binding,
}

/// The import of `length` bytes at `pointer` on a host of `page`-byte pages and a device that
/// imports at multiples of `alignment`: the range rounded out to the larger of the two. Refused
/// when the rounding would take pages the range does not touch.
pub(super) fn import_window(
    pointer: u64,
    length: u64,
    page: u64,
    alignment: u64,
) -> Result<ImportWindow, WindowRefusal> {
    let unit = page.max(alignment).max(1);
    let end = pointer.checked_add(length).ok_or(WindowRefusal::Overflow)?;
    let round_up = |value: u64, to: u64| value.checked_next_multiple_of(to);
    let start = pointer - pointer % unit;
    let window_end = round_up(end, unit).ok_or(WindowRefusal::Overflow)?;
    let page_start = pointer - pointer % page.max(1);
    let page_end = round_up(end, page.max(1)).ok_or(WindowRefusal::Overflow)?;
    if start < page_start || window_end > page_end {
        return Err(WindowRefusal::BeyondPages);
    }
    Ok(ImportWindow {
        start,
        size: window_end - start,
        offset: pointer - start,
    })
}

/// Where the bytes of a no-copy buffer at `offset` in its window lie in its `VkBuffer`, whose
/// memory alignment is `alignment` (a power of two): `offset` modulo the alignment, or modulo 16
/// where that is larger. The `VkBuffer` is bound that many bytes below the bytes, at a multiple
/// of both in the window (and in the Metal address space, whose ranges start at pages), so its
/// bytes are congruent to their Metal addresses modulo 16, as the runtime's other buffers' are.
pub(super) fn buffer_base(offset: u64, alignment: u64) -> u64 {
    offset % alignment.max(16)
}

/// The size of the `VkBuffer` of a no-copy buffer of `length` bytes whose bytes start at `base`
/// in it: through the padded length, rounded up to 16 bytes, so that a range the binding layer pads
/// to 16 bytes from `base` lies inside it. `None` past `u64`.
pub(super) fn based_size(base: u64, length: u64) -> Option<u64> {
    length
        .checked_next_multiple_of(16)?
        .checked_add(base)?
        .checked_next_multiple_of(16)
}

/// Where a `VkBuffer` of `size` bytes with `requirements`, whose bytes start `base` bytes into
/// it, binds inside `window`: `base` bytes below the window's offset, which must be a multiple
/// of the memory alignment, with room in the window for the buffer and its requirements.
pub(super) fn binds_in(
    window: &ImportWindow,
    base: u64,
    size: u64,
    requirements: &vk::MemoryRequirements,
) -> Result<u64, WindowRefusal> {
    let at = window
        .offset
        .checked_sub(base)
        .ok_or(WindowRefusal::Binding)?;
    let aligned = at.is_multiple_of(requirements.alignment.max(1));
    let room = window.size - at;
    if !aligned || size > room || requirements.size > room {
        return Err(WindowRefusal::Binding);
    }
    Ok(at)
}

/// The host's page size, where the runtime knows how to ask for it.
#[cfg(unix)]
pub(super) fn host_page() -> Option<u64> {
    // SAFETY: `sysconf` reads a system constant.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    u64::try_from(page).ok().filter(|&page| page > 0)
}

/// The host's page size, where the runtime knows how to ask for it: not on this system.
#[cfg(not(unix))]
pub(super) fn host_page() -> Option<u64> {
    None
}

/// The refusal of host memory this device cannot import exactly.
pub(super) fn not_importable() -> Error {
    Error::Unsupported(Unsupported::not_representable(Feature::HostMemory))
}

/// The error of a failed import call (`vkGetMemoryHostPointerPropertiesEXT` or the allocation): a
/// driver that refuses the memory is a device gap, anything else a Vulkan failure. Refusing
/// drivers answer `VK_ERROR_INVALID_EXTERNAL_HANDLE`, but RADV of Mesa 26.0.8 answers
/// `VK_ERROR_UNKNOWN` for the shared memory 26.2.4 refuses with it, so that is a refusal too.
pub(super) fn import_refused(error: vk::Result) -> Error {
    match error {
        vk::Result::ERROR_INVALID_EXTERNAL_HANDLE | vk::Result::ERROR_UNKNOWN => not_importable(),
        other => vulkan(other),
    }
}

/// Refuse the window's pages, typed, unless the process maps every one of them writable: Linux
/// drivers pin imported pages for writing. The kernel answers for the mappings over the window
/// ([`queried_writable`]) at each call, so a page remapped or reprotected since an earlier import
/// is judged as it is now; a kernel without the query (before Linux 6.11) has the lines of
/// `/proc/self/maps` read up to the window instead ([`writable`]), which cost about 4.5 ms per
/// import in a process of 34,000 mappings. When the file cannot be opened, every import is
/// refused as not implemented here.
#[cfg(target_os = "linux")]
pub(super) fn refuse_unwritable(window: &ImportWindow) -> Result<(), Error> {
    use std::io::BufRead;
    let maps = std::fs::File::open("/proc/self/maps")
        .map_err(|_| Error::Unsupported(Unsupported::not_implemented(Feature::HostMemory)))?;
    let end = window.start + window.size;
    let writable = match queried_writable(&maps, window.start, end) {
        Some(writable) => writable,
        None => {
            let lines = std::io::BufReader::new(maps).lines().map_while(Result::ok);
            writable(lines, window.start, end)
        }
    };
    if writable {
        Ok(())
    } else {
        Err(not_importable())
    }
}

/// `struct procmap_query` of Linux's `include/uapi/linux/fs.h` (Linux 6.11): the argument of
/// [`PROCMAP_QUERY`].
#[cfg(target_os = "linux")]
#[repr(C)]
#[derive(Default)]
struct ProcmapQuery {
    size: u64,
    query_flags: u64,
    query_addr: u64,
    vma_start: u64,
    vma_end: u64,
    vma_flags: u64,
    vma_page_size: u64,
    vma_offset: u64,
    inode: u64,
    dev_major: u32,
    dev_minor: u32,
    vma_name_size: u32,
    build_id_size: u32,
    vma_name_addr: u64,
    build_id_addr: u64,
}

#[cfg(target_os = "linux")]
const _: () = assert!(std::mem::size_of::<ProcmapQuery>() == 104);

/// `PROCMAP_QUERY`, `_IOWR('f', 17, struct procmap_query)` in the generic ioctl encoding (an
/// architecture of another encoding answers `ENOTTY`, and its imports take the lines).
#[cfg(target_os = "linux")]
const PROCMAP_QUERY: u64 =
    (3 << 30) | ((std::mem::size_of::<ProcmapQuery>() as u64) << 16) | ((b'f' as u64) << 8) | 17;

/// `PROCMAP_QUERY_VMA_WRITABLE`: the mapping's `VM_WRITE`, the `w` of its line in the maps.
#[cfg(target_os = "linux")]
const PROCMAP_QUERY_VMA_WRITABLE: u64 = 0x02;

/// Whether the process maps `[start, end)` without a gap, each mapping with write permission, as
/// the kernel answers `PROCMAP_QUERY` on `maps` (this process's `/proc/self/maps`): one query per
/// mapping the window touches, each for the mapping covering the first address not yet covered.
/// `None` where the kernel does not answer the query (before Linux 6.11, `ENOTTY`) or fails it
/// otherwise; a missing mapping (`ENOENT`) is an answer, not writable.
#[cfg(target_os = "linux")]
pub(super) fn queried_writable(maps: &std::fs::File, start: u64, end: u64) -> Option<bool> {
    use std::os::fd::AsRawFd;
    let mut covered = start;
    while covered < end {
        let mut query = ProcmapQuery {
            size: std::mem::size_of::<ProcmapQuery>() as u64,
            query_addr: covered,
            ..ProcmapQuery::default()
        };
        // SAFETY: `query` is a `struct procmap_query` of its stated size, which the kernel reads
        // and writes during the call; it asks for no name or build ID (sizes 0).
        let result = unsafe {
            libc::ioctl(
                maps.as_raw_fd(),
                PROCMAP_QUERY as _,
                std::ptr::from_mut(&mut query),
            )
        };
        if result != 0 {
            let error = std::io::Error::last_os_error().raw_os_error();
            return (error == Some(libc::ENOENT)).then_some(false);
        }
        // The mapping covers `covered`; an answer that does not is not taken.
        if query.vma_start > covered || query.vma_end <= covered {
            return None;
        }
        if query.vma_flags & PROCMAP_QUERY_VMA_WRITABLE == 0 {
            return Some(false);
        }
        covered = query.vma_end;
    }
    Some(true)
}

/// Refuse the window's pages unless the process can write them: no check off Linux (MoltenVK
/// hands the range to Metal, which reads read-only pages).
#[cfg(not(target_os = "linux"))]
pub(super) fn refuse_unwritable(_window: &ImportWindow) -> Result<(), Error> {
    Ok(())
}

/// Whether the mappings of `maps` (lines of `/proc/<pid>/maps`, in address order) cover
/// `[start, end)` without a gap, each with write permission. Reading stops at the first line past
/// `end`; a line that does not parse is skipped.
#[cfg(target_os = "linux")]
pub(super) fn writable(maps: impl IntoIterator<Item = String>, start: u64, end: u64) -> bool {
    let mut covered = start;
    for line in maps {
        if covered >= end {
            break;
        }
        let Some((from, to, write)) = mapping(&line) else {
            continue;
        };
        if to <= covered {
            continue;
        }
        if from > covered || !write {
            return false;
        }
        covered = to;
    }
    covered >= end
}

/// One line of `/proc/<pid>/maps`: the mapping's start, its end, and whether it is writable.
#[cfg(target_os = "linux")]
fn mapping(line: &str) -> Option<(u64, u64, bool)> {
    let mut fields = line.split_ascii_whitespace();
    let (from, to) = fields.next()?.split_once('-')?;
    let permissions = fields.next()?;
    Some((
        u64::from_str_radix(from, 16).ok()?,
        u64::from_str_radix(to, 16).ok()?,
        permissions.as_bytes().get(1) == Some(&b'w'),
    ))
}

/// What a device knows to import host memory.
pub(super) struct Importer {
    functions: ash::ext::external_memory_host::DeviceFn,
    /// The host's page size.
    pub(super) page: u64,
    /// `minImportedHostPointerAlignment`.
    pub(super) alignment: u64,
    /// `vkExportMetalObjectsEXT`, where the device enables `VK_EXT_metal_objects`: each import is
    /// then allocated for export of its `MTLBuffer` (`metal_texture.rs`).
    metal: Option<ash::ext::metal_objects::DeviceFn>,
    /// `vkSetDebugUtilsObjectNameEXT`, where the device exports `MTLBuffer`s and the runtime's
    /// instance reports through `VK_EXT_debug_utils`: an image made of a Metal texture is named
    /// [`METAL_TEXTURE_IMAGE`] for the layers' messages.
    names: Option<ash::ext::debug_utils::DeviceFn>,
}

/// The debug name of an image the runtime made of a Metal texture over an import's `MTLBuffer`
/// (`VkImportMetalTextureInfoEXT`, `metal_texture.rs`), where its instance reports messages. The
/// validation layer (1.4.357, and its main of 2026-10-07) does not model that import, which
/// Vulkan treats as bound, and reports such an image as used with no memory bound; the name lets
/// a reader tell those reports from others.
pub(super) const METAL_TEXTURE_IMAGE: &std::ffi::CStr =
    c"metal2vulkan: a Metal texture over imported host memory";

impl Importer {
    /// The importer of a device, or the refusal every no-copy buffer on it gets: the device did
    /// not enable `VK_EXT_external_memory_host`, the runtime does not know the host's page size,
    /// or the device cannot bind a buffer of `usage` to imported host memory.
    ///
    /// # Safety
    ///
    /// `device` is a live device of `physical`, made from `instance` with `enabled` enabled, and
    /// `loader_word` is its loader word.
    pub(super) unsafe fn new(
        instance: &ash::Instance,
        physical: vk::PhysicalDevice,
        device: &ash::Device,
        enabled: &EnableSet,
        usage: vk::BufferUsageFlags,
        loader_word: &LoaderWord,
        metal_objects: bool,
        debug_names: bool,
    ) -> Result<Importer, Unsupported> {
        if !enabled
            .extensions
            .contains(&VkExtension::ExtExternalMemoryHost)
        {
            return Err(Unsupported::needs(
                Feature::HostMemory,
                metal2vulkan::error::Capability::Extension(VkExtension::ExtExternalMemoryHost),
            ));
        }
        let page = host_page().ok_or_else(|| Unsupported::not_implemented(Feature::HostMemory))?;
        let mut host = vk::PhysicalDeviceExternalMemoryHostPropertiesEXT::default();
        let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut host);
        // SAFETY: the caller's contract; the structure belongs to an enabled extension.
        unsafe { instance.get_physical_device_properties2(physical, &mut properties) };
        let alignment = host.min_imported_host_pointer_alignment;
        let info = vk::PhysicalDeviceExternalBufferInfo::default()
            .usage(usage)
            .handle_type(vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT);
        let mut external = vk::ExternalBufferProperties::default();
        // SAFETY: the caller's contract; the query is core in Vulkan 1.1.
        unsafe {
            instance.get_physical_device_external_buffer_properties(physical, &info, &mut external)
        };
        let features = external.external_memory_properties.external_memory_features;
        if !features.contains(vk::ExternalMemoryFeatureFlags::IMPORTABLE)
            || features.contains(vk::ExternalMemoryFeatureFlags::DEDICATED_ONLY)
        {
            return Err(Unsupported::not_representable(Feature::HostMemory));
        }
        let handle = device.handle();
        let load = |name: &std::ffi::CStr| {
            // SAFETY: the device is live, and the loader serves `vkGetDeviceProcAddr` after its
            // word is put back; the extension is enabled.
            let function = unsafe {
                loader_word.restore();
                instance.get_device_proc_addr(handle, name.as_ptr())
            };
            function.map_or(std::ptr::null(), |function| function as *const c_void)
        };
        let functions = ash::ext::external_memory_host::DeviceFn::load(load);
        let metal = metal_objects.then(|| ash::ext::metal_objects::DeviceFn::load(load));
        let names =
            (metal_objects && debug_names).then(|| ash::ext::debug_utils::DeviceFn::load(load));
        Ok(Importer {
            functions,
            page,
            alignment,
            metal,
            names,
        })
    }

    /// Whether the device's imports are allocated for export of their `MTLBuffer`
    /// (`VK_EXT_metal_objects`).
    pub(super) fn exports_metal(&self) -> bool {
        self.metal.is_some()
    }

    /// The `id<MTLBuffer>` behind `memory`, an import of this importer's device allocated for
    /// export; `None` where the device does not export one.
    ///
    /// # Safety
    ///
    /// `device` is the importer's device and `memory` a live import of it.
    pub(super) unsafe fn metal_buffer(
        &self,
        device: &ash::Device,
        memory: vk::DeviceMemory,
    ) -> Option<vk::MTLBuffer_id> {
        let metal = self.metal.as_ref()?;
        let mut buffer = vk::ExportMetalBufferInfoEXT::default().memory(memory);
        let mut info = vk::ExportMetalObjectsInfoEXT::default().push_next(&mut buffer);
        // SAFETY: the caller's; the memory was allocated for export of its `MTLBuffer`
        // (`Memory::import_host`), and the function was loaded for this device.
        unsafe { (metal.export_metal_objects_ext)(device.handle(), &mut info) };
        let exported = buffer.mtl_buffer;
        (!exported.is_null()).then_some(exported)
    }

    /// Name `image`, made of a Metal texture, [`METAL_TEXTURE_IMAGE`] where the instance reports
    /// messages; a failure to name it is ignored (the name serves messages only).
    ///
    /// # Safety
    ///
    /// `device` is the importer's device and `image` a live image of it.
    pub(super) unsafe fn name_metal_image(&self, device: &ash::Device, image: vk::Image) {
        let Some(names) = &self.names else {
            return;
        };
        let info = vk::DebugUtilsObjectNameInfoEXT::default()
            .object_handle(image)
            .object_name(METAL_TEXTURE_IMAGE);
        // SAFETY: the caller's; the instance enables `VK_EXT_debug_utils` (it reports messages),
        // and the function was loaded for this device.
        let _ = unsafe { (names.set_debug_utils_object_name_ext)(device.handle(), &info) };
    }

    /// The memory types `pointer` can be imported into (`vkGetMemoryHostPointerPropertiesEXT`).
    ///
    /// # Safety
    ///
    /// `device` is the importer's device; `pointer` is host memory aligned to its import
    /// alignment.
    pub(super) unsafe fn memory_types(
        &self,
        device: &ash::Device,
        pointer: NonNull<u8>,
    ) -> Result<u32, vk::Result> {
        let mut properties = vk::MemoryHostPointerPropertiesEXT::default();
        // SAFETY: the caller's contract; the function was loaded for this device.
        let result = unsafe {
            (self.functions.get_memory_host_pointer_properties_ext)(
                device.handle(),
                vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT,
                pointer.as_ptr().cast_const().cast(),
                &mut properties,
            )
        };
        result.result_with_success(properties.memory_type_bits)
    }
}

/// What a no-copy buffer gives back when its last reference goes: the caller's range, to its
/// deallocator, and its keep-alive.
pub(super) struct Release {
    pointer: NonNull<u8>,
    length: u64,
    deallocator: Mutex<Option<Deallocator>>,
    keep_alive: Mutex<Option<KeepAlive>>,
}

// SAFETY: the pointer is only handed to the deallocator, which is `Send`; the deallocator and
// the keep-alive are reached only through their locks.
unsafe impl Send for Release {}
// SAFETY: as above.
unsafe impl Sync for Release {}

impl Release {
    /// Never run the deallocator, nor drop it or the keep-alive: the memory stays imported, the
    /// process is exiting (`exit_guard`).
    pub(super) fn keep(&self) {
        if let Some(deallocator) = lock(&self.deallocator).take() {
            std::mem::forget(deallocator);
        }
        if let Some(keep_alive) = lock(&self.keep_alive).take() {
            std::mem::forget(keep_alive);
        }
    }

    /// Its keep-alive, for the import to drop once it is freed; `None` where the caller gave none.
    pub(super) fn keep_alive(&self) -> Option<KeepAlive> {
        lock(&self.keep_alive).take()
    }

    /// Whether its deallocator is yet to run.
    pub(super) fn deallocates(&self) -> bool {
        lock(&self.deallocator).is_some()
    }

    /// The length of the caller's range.
    pub(super) fn length(&self) -> u64 {
        self.length
    }

    /// Run the deallocator, once; a panic inside it is caught.
    pub(super) fn run(&self) {
        let deallocator = lock(&self.deallocator).take();
        if let Some(deallocator) = deallocator {
            let (pointer, length) = (self.pointer, self.length);
            exit_guard::program(move || deallocator(pointer, length));
        }
    }
}
