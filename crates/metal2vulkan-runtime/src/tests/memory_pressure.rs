//! Private buffers past the device's own memory: with memory priority and pageable device-local
//! memory enabled (`DeviceOptions::withhold_memory_priority` off) a driver that pages keeps
//! making them, where without them the allocation that fills the heap fails with `OutOfMemory`.
//!
//! The test fills the device on purpose, so it is ignored: run it with `--include-ignored` on a
//! machine nothing else is using the GPU of. It frees everything before it returns.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use super::blit_holds::Scratch;
use crate::ShaderCacheOptions;
use crate::{Device, DeviceKind, DeviceOptions, Error, MemoryKind, ResourceOptions};
use ash::vk;

const CHUNK: u64 = 256 << 20;

/// Make 256 MiB `Private` buffers until `limit` bytes or a refusal: how many and the error.
fn fill(device: &Device, limit: u64) -> (u64, Option<Error>) {
    let mut kept = Vec::new();
    let mut error = None;
    while (kept.len() as u64) * CHUNK < limit {
        match device.new_buffer(CHUNK, ResourceOptions::private()) {
            Ok(buffer) => kept.push(buffer),
            Err(failure) => {
                error = Some(failure);
                break;
            }
        }
    }
    (kept.len() as u64 * CHUNK, error)
}

fn device(scratch: &Scratch, withhold: bool) -> Device {
    let options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![DeviceKind::Discrete],
        },
        withhold_memory_priority: withhold,
        shader_cache: ShaderCacheOptions {
            directory: Some(scratch.0.clone()),
            ..ShaderCacheOptions::default()
        },
        ..DeviceOptions::default()
    };
    Device::system_default_with(&options).unwrap()
}

/// The device-local heap's size.
fn heap(device: &Device) -> u64 {
    // SAFETY: the handles stay valid for the borrow; only properties are read.
    let raw = unsafe { device.raw() };
    let memory = unsafe {
        raw.instance
            .get_physical_device_memory_properties(raw.physical_device)
    };
    memory.memory_heaps[..memory.memory_heap_count as usize]
        .iter()
        .filter(|heap| heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
        .map(|heap| heap.size)
        .max()
        .unwrap_or(0)
}

#[test]
#[ignore = "fills the device's memory; run it on purpose"]
fn pageable_memory_keeps_private_buffers_coming_past_the_heap() {
    let scratch = Scratch::new("memory-pressure");
    let limit;
    {
        let with = device(&scratch, false);
        let paged = with.extension_enabled(c"VK_EXT_pageable_device_local_memory");
        limit = heap(&with) + (1 << 30);
        let (made, error) = fill(&with, limit);
        eprintln!(
            "PRESSURE: with priority ({paged} pageable): {} MiB of {} MiB, {:?}",
            made >> 20,
            limit >> 20,
            error.as_ref().map(ToString::to_string)
        );
        if paged {
            assert!(error.is_none(), "a paging device refused: {error:?}");
            assert_eq!(made, limit.div_ceil(CHUNK) * CHUNK);
        }
    }
    {
        let without = device(&scratch, true);
        let (made, error) = fill(&without, limit);
        eprintln!(
            "PRESSURE: without: {} MiB of {} MiB, {:?}",
            made >> 20,
            limit >> 20,
            error.as_ref().map(ToString::to_string)
        );
        if let Some(error) = error {
            assert!(
                matches!(error, Error::OutOfMemory(MemoryKind::Device)),
                "the refusal is typed out-of-memory: {error:?}"
            );
        }
    }
}
