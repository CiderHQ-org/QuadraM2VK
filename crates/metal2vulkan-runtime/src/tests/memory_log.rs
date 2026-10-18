//! The memory log's device address (`memory_log.rs`): a buffer made over a block's memory, bound at
//! its start and asked, gives the address the block's buffer of addresses gives its start, the one
//! every buffer placed in the block takes its own from.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use crate::device::lock;
use crate::memory::{allocate_addressed, MemoryClass};
use crate::memory_log::address_of;
use crate::{Device, DeviceKind, DeviceOptions, ShaderCacheOptions};
use ash::vk;
use std::path::PathBuf;

/// A scratch directory for the device's shader cache, removed when dropped.
struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn a_block_s_asked_address_is_its_buffer_of_addresses() {
    let scratch =
        Scratch(std::env::temp_dir().join(format!("m2v-memory-log-{}", std::process::id())));
    std::fs::create_dir_all(&scratch.0).unwrap();
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
            directory: Some(scratch.0.clone()),
            ..ShaderCacheOptions::default()
        },
        ..DeviceOptions::default()
    };
    let device = Device::system_default_with(&options).unwrap();
    let shared = device.shared();
    let raw = shared.vk();
    let requirements = vk::MemoryRequirements {
        size: 4096,
        alignment: 256,
        memory_type_bits: u32::MAX,
    };
    let placement = allocate_addressed(
        &shared.memory,
        raw,
        &requirements,
        MemoryClass::Private,
        &shared.counts,
    )
    .unwrap();
    let Some(placed) = placement.address else {
        eprintln!("MEMORY-LOG: {} gives no device addresses", device.name());
        lock(&shared.memory).free(raw, &placement);
        return;
    };
    let start = placed - placement.offset;
    // SAFETY: the placement's block is the device's, live, allocated with the device-address
    // flag (it has a buffer of addresses), and no dedicated allocation.
    let asked = unsafe {
        address_of(
            raw,
            placement.memory,
            placement.offset + placement.size(),
            placement.type_index(),
            None,
        )
    };
    lock(&shared.memory).free(raw, &placement);
    eprintln!(
        "MEMORY-LOG on {}: the block starts at device address 0x{start:x}, asked 0x{:x}",
        device.name(),
        asked.unwrap_or(0)
    );
    assert_eq!(asked, Some(start), "the asked address is not the block's");
}
