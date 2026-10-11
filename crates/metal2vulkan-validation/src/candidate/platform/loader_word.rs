//! The Vulkan loader's word at the start of a dispatchable handle, kept as the loader wrote it.
//!
//! The Loader-Driver interface reserves a dispatchable object's first pointer-sized entry for
//! the loader: a driver initializes it to `ICD_LOADER_MAGIC`, and "the loader will replace the
//! first entry with a pointer to the dispatch table which is owned by the loader". MoltenVK reads
//! only the entry after it (its object pointer).
//!
//! MoltenVK 1.4.1 writes the magic back into the entry every time it computes the handle of one of
//! its objects (`MVKDispatchableVulkanAPIObject::getVkHandle`). It does so to name the object of
//! each message it reports to a debug-utils messenger, which every validated context registers.
//! A message about the device (each of its activity-timing lines under
//! `MVK_CONFIG_PERFORMANCE_TRACKING`, among others) so leaves `0x01CDC0DE` where the loader's
//! table pointer was.
//!
//! The loader's own entry points for a device then fail. `vkDestroyDevice`,
//! `vkAllocateCommandBuffers`, `vkGetDeviceQueue(2)`, and `vkGetDeviceProcAddr` (which loads every
//! extension's command table) read the word as the loader's table and fault at `0x1cdc0de`:
//! `vkDestroyDevice` at exit. A context records the word the loader wrote when the device came back
//! from `vkCreateDevice`, and puts it back before each of these calls it makes later
//! (`VulkanContext::allocate_command_buffers`, `VulkanContext::acceleration_structure`, and its
//! `Drop`). Validation makes them nowhere else: `validation/clippy.toml` disallows the direct
//! calls.
//!
//! The validation layer keys its state by the same word, but an overwritten device does not stop
//! it. For every device, queue, and command-buffer call it reads the object's first word as its
//! key and first compares that key with the key of the device it served last (1.4.357,
//! `dispatch_object_manual.cpp`). An overwritten device compares equal to itself, so its calls
//! are served normally (19 single-row runs reported device activity, then made layer-served calls,
//! and exited 0). The layer looks the key up in its map, and aborts on a miss ("The VkDevice
//! dispatch handle was not found"), only when that comparison fails:
//!
//! - a different device was served last;
//! - an overwritten queue or command buffer is called while the last device's word is intact.
//!   MoltenVK writes the magic into a queue when a Metal command buffer fails
//!   (`handleMTLCommandBufferError`), and into a command buffer when it reports an error about
//!   it (a feature not present, for one). The context makes no loader call that reads their
//!   words, and does not guard them.
//!
//! With two overwritten devices in one process, the comparison matches the other device's
//! magic, and the layer serves the call with that device's state: its messages are misattributed
//! rather than the run aborted. Validation opens one context per device type and thread, so this
//! needs two such contexts at once. These are recorded limitations, not guarded.

use ash::vk::{self, Handle};

/// The loader's word of one live dispatchable handle, as it stood when the handle reached the
/// application.
pub(crate) struct LoaderWord {
    entry: *mut usize,
    word: usize,
}

impl LoaderWord {
    /// The word at the start of `device`'s handle now.
    ///
    /// # Safety
    ///
    /// `device` is a live dispatchable handle the Vulkan loader (or a driver loaded directly)
    /// returned, and [`LoaderWord::restore`] is called only while it lives.
    pub(crate) unsafe fn of_device(device: vk::Device) -> LoaderWord {
        // SAFETY: the caller's contract; a dispatchable handle points at the object's first
        // pointer-sized entry.
        unsafe {
            LoaderWord::at(std::ptr::with_exposed_provenance_mut(
                device.as_raw() as usize
            ))
        }
    }

    /// The word at `entry` now.
    ///
    /// # Safety
    ///
    /// `entry` points at the first entry of a live dispatchable object, and
    /// [`LoaderWord::restore`] is called only while it lives.
    unsafe fn at(entry: *mut usize) -> LoaderWord {
        // SAFETY: the caller's contract.
        let word = unsafe { entry.read() };
        LoaderWord { entry, word }
    }

    /// Put the recorded word back where the driver overwrote it. Returns whether it had been
    /// overwritten.
    ///
    /// It does not close a write by the driver's own threads. Between this call and the loader
    /// call after it, a MoltenVK thread can still write the magic. After a device loss,
    /// `MVKDevice::markLost` wakes fence waiters before later Metal completion handlers log about
    /// the device, so a context that goes on to call the loader can still read the magic. Not
    /// guarded.
    ///
    /// # Safety
    ///
    /// The handle is still live, and no other thread calls into the loader with it during the
    /// call.
    pub(crate) unsafe fn restore(&self) -> bool {
        // SAFETY: the caller's contract: the object is live, and its first entry is the loader's.
        unsafe {
            if self.entry.read() == self.word {
                return false;
            }
            self.entry.write(self.word);
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::LoaderWord;

    /// `ICD_LOADER_MAGIC` (`vulkan/vk_icd.h`).
    const ICD_LOADER_MAGIC: usize = 0x01CD_C0DE;

    /// A driver that writes the magic back over the loader's dispatch pointer (as MoltenVK's
    /// `getVkHandle` does) has it restored before the loader's own entry point reads it, and a word
    /// the driver left alone is not written.
    #[test]
    fn the_loaders_word_is_put_back_where_the_driver_overwrote_it() {
        // A driver's dispatchable object: the loader's entry, then the driver's own pointer.
        let table = 0x1234_5670_usize;
        let object: *mut [usize; 2] = Box::into_raw(Box::new([table, 0xD1D1]));
        let entry = object.cast::<usize>();
        // SAFETY: `object` lives until the end of the test, and its first entry stands for the
        // loader's; every access goes through `entry` or `object`.
        unsafe {
            let word = LoaderWord::at(entry);
            assert!(!word.restore(), "an intact word is left alone");
            entry.write(ICD_LOADER_MAGIC);
            assert!(word.restore());
            assert_eq!(object.read(), [table, 0xD1D1]);
            assert!(!word.restore());
            drop(Box::from_raw(object));
        }
    }
}
