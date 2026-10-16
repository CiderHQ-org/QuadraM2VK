//! The device's functions, the only door to them: `DeviceShared::vk` hands them out through
//! [`Functions::get`], which checks the caller (`exit_guard::check`) and, inside one of the
//! runtime's sections, hands out the exit guard's table alone ([`Call`]): the same device, loaded
//! with only the table's entry points, every other one ash's stub, which panics where unwinding
//! cannot pass and so ends the process. In builds with debug assertions (every test run), a section
//! that calls an entry point outside the table therefore fails the lane that reaches it, so the
//! table holds every entry point a section calls, and priming, which makes each before the exit
//! hook is registered (`prime.rs`), covers them all. Other builds hand out every function.
//!
//! Each is a [`Vk`]: the `ash::Device`, which it dereferences to, and the reads the commands of the
//! device's queue family for encoders make, which the runtime's global memory barriers name in
//! their destination (`tracker::Barrier::record`, `image_barrier::MemoryReads`).

use crate::exit_guard::{self, Call};
use crate::image_barrier::MemoryReads;
use ash::vk;
use std::ffi::{c_void, CStr};
use std::ops::Deref;

/// A device's functions as the runtime records commands with them: the `ash::Device`, and the
/// reads a global memory barrier recorded into a command buffer of the device's queue family for
/// encoders names in place of `MEMORY_READ` (the module docs).
#[derive(Clone)]
pub(crate) struct Vk {
    device: ash::Device,
    reads: MemoryReads,
}

impl Vk {
    /// `device`'s functions, whose global memory barriers name `reads`: for the work the runtime
    /// records before the device's functions are made (`prime.rs`).
    pub(super) fn new(device: ash::Device, reads: MemoryReads) -> Vk {
        Vk { device, reads }
    }

    /// The reads a global memory barrier's destination names.
    pub(crate) fn reads(&self) -> MemoryReads {
        self.reads
    }
}

impl Deref for Vk {
    type Target = ash::Device;

    fn deref(&self) -> &ash::Device {
        &self.device
    }
}

/// A device's functions: all of them, and the exit guard's table alone.
#[derive(Clone)]
pub(crate) struct Functions {
    all: Vk,
    sectioned: Vk,
    /// The address of each device-level entry point of the table, in its order (0 for an
    /// instance-level one, or one the device has not): the topmost layer's where layers run,
    /// else the driver's.
    entry_points: Vec<u64>,
}

impl Functions {
    /// The functions of `all`'s device, made from `instance`, whose global memory barriers name
    /// `reads`.
    ///
    /// # Safety
    ///
    /// `all` is a live device of `instance`, routed (`compiles::route`), and the loader serves
    /// `vkGetDeviceProcAddr` for it (its loader word is in place).
    pub(super) unsafe fn new(
        instance: &ash::Instance,
        all: ash::Device,
        reads: MemoryReads,
    ) -> Functions {
        let handle = all.handle();
        let get = instance.fp_v1_0().get_device_proc_addr;
        // SAFETY: the caller's contract; a missing function is null, as ash loads it.
        let address = |name: &CStr| unsafe {
            std::mem::transmute::<vk::PFN_vkVoidFunction, *const c_void>(get(handle, name.as_ptr()))
        };
        // Only the device-level functions ash asks for are looked up: asking the device for an
        // instance-level one (the table's instance and messenger releases) is an error the
        // validation layer reports.
        let mut entry_points = vec![0; Call::ALL.len()];
        let load = |name: &CStr| -> *const c_void {
            match Call::ALL.iter().position(|call| call.name() == name) {
                Some(at) => {
                    let pointer = address(name);
                    entry_points[at] = pointer as usize as u64;
                    // The pipeline makings come through `compiles.rs`, as `all`'s do.
                    super::compiles::entry(name).unwrap_or(pointer)
                }
                None => std::ptr::null(),
            }
        };
        // SAFETY: as above: each pointer is the device's function of that name, or null.
        let sectioned = unsafe { ash::Device::load_with(load, handle) };
        Functions {
            all: Vk::new(all, reads),
            sectioned: Vk::new(sectioned, reads),
            entry_points,
        }
    }

    /// The address of each entry point of the exit guard's table, in its order.
    pub(super) fn entry_points(&self) -> &[u64] {
        &self.entry_points
    }

    /// The device's functions for this thread now (see the module).
    pub(crate) fn get(&self) -> &Vk {
        exit_guard::check();
        if cfg!(debug_assertions) && exit_guard::in_section() {
            &self.sectioned
        } else {
            &self.all
        }
    }
}
