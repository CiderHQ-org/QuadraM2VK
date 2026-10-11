//! The Vulkan loader's word at the start of the device handle, kept as the loader wrote it
//! (a copy of validation's `LoaderWord`, which is deleted when validation's executors move onto the
//! runtime).
//!
//! The Loader-Driver interface gives the loader a dispatchable object's first pointer-sized
//! entry: its dispatch table. MoltenVK 1.4.1 writes `ICD_LOADER_MAGIC` back into it every time it
//! names the device in a message it reports to a debug-utils messenger. The loader's own entry
//! points for the device (`vkAllocateCommandBuffers`, `vkGetDeviceQueue`, `vkGetDeviceProcAddr`,
//! `vkDestroyDevice`) then read the magic as their table and fault. A device the runtime creates
//! has such a messenger when its options ask for one, and an adopted device when its consumer
//! registered one, so the runtime records the word the loader wrote and puts it back before each
//! of those calls. On every other driver the word never changes and the restore writes nothing.
//! MoltenVK writes the magic over a queue's word too, where a message names the queue
//! (`device/relay.rs`): the queue's word is put back before each use.

use ash::vk::{self, Handle};

/// The loader's word of one live device handle, as it stood when the device reached the runtime.
pub(super) struct LoaderWord {
    entry: *mut usize,
    word: usize,
}

// SAFETY: the pointer names the device's first word, which outlives every use (the device is
// destroyed after the last restore), and every access is a whole-word read or write.
unsafe impl Send for LoaderWord {}
// SAFETY: as above.
unsafe impl Sync for LoaderWord {}

impl LoaderWord {
    /// The word at the start of `device`'s handle now.
    ///
    /// # Safety
    ///
    /// `device` is a live dispatchable handle the Vulkan loader (or a driver loaded directly)
    /// returned, its word is still the loader's, and [`LoaderWord::restore`] is called only while
    /// it lives.
    pub(super) unsafe fn of_device(device: vk::Device) -> LoaderWord {
        // SAFETY: the caller's.
        unsafe { LoaderWord::of_handle(device.as_raw()) }
    }

    /// The word at the start of the dispatchable handle `handle` now (a queue's, which MoltenVK
    /// overwrites as it does the device's, `device/relay.rs`).
    ///
    /// # Safety
    ///
    /// As [`LoaderWord::of_device`]'s, for the object `handle` names.
    pub(super) unsafe fn of_handle(handle: u64) -> LoaderWord {
        let entry: *mut usize = std::ptr::with_exposed_provenance_mut(handle as usize);
        // SAFETY: the caller's contract; a dispatchable handle points at the object's first
        // pointer-sized entry.
        let word = unsafe { entry.read_volatile() };
        LoaderWord { entry, word }
    }

    /// Put the recorded word back where the driver overwrote it.
    ///
    /// # Safety
    ///
    /// The handle is still live.
    pub(super) unsafe fn restore(&self) {
        // SAFETY: the caller's contract: the object is live, and its first entry is the loader's.
        unsafe {
            if self.entry.read_volatile() != self.word {
                self.entry.write_volatile(self.word);
            }
        }
    }
}
