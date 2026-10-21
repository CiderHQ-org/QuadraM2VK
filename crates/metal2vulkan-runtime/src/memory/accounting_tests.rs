//! Live placements return to their baseline while pooled storage remains reusable.
use super::*;

fn memory() -> Memory {
    let mut properties = vk::PhysicalDeviceMemoryProperties {
        memory_type_count: 1,
        ..Default::default()
    };
    properties.memory_types[0] = vk::MemoryType {
        property_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
        heap_index: 0,
    };
    Memory::new(&properties, BLOCK_BYTES, false, false, vec![0], None, None)
}

fn block(size: u64, imported: bool) -> Block {
    Block {
        memory: vk::DeviceMemory::null(),
        size,
        mapped: None,
        addresses: None,
        imported,
        type_index: 0,
    }
}

#[test]
fn live_memory_follows_releases_and_keeps_device_totals_separate() {
    let mut first = memory();
    let second = memory();
    let index = first.pools[0].add(4096, false, block(4096, false), 128);
    let (same, offset) = first.pools[0].place(64, 256).unwrap();
    assert_eq!(same, index);
    assert_eq!(
        first.statistics().live_bytes,
        192,
        "alignment is free space"
    );
    assert_eq!(first.statistics().reserved_bytes, 4096);
    assert_eq!(second.statistics(), MemoryStatistics::default());
    first.pools[0].release(index, 0, 128, Instant::now());
    assert_eq!(first.statistics().live_bytes, 64);
    first.pools[0].release(index, offset, 64, Instant::now());
    assert_eq!(first.statistics().live_bytes, 0);
    assert_eq!(
        first.statistics().reserved_bytes,
        4096,
        "empty pool is cached"
    );
    let dedicated = first.pools[0].add(8192, true, block(8192, false), 8192);
    assert_eq!(first.statistics().live_bytes, 8192);
    assert!(first.pools[0]
        .release(dedicated, 0, 8192, Instant::now())
        .is_some());
    assert_eq!(first.statistics().reserved_bytes, 4096);
    first.pools[0].add(16384, true, block(16384, true), 16384);
    assert_eq!(first.statistics().imported_bytes, 16384);
    assert_eq!(first.statistics().live_bytes, 0, "imports are separate");
}
