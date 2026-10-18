//! Visible-function tables (`function_table.rs`): Metal's measured answers that need no device.

use crate::function_table::table_allocated_size;
use crate::VisibleFunctionTableDescriptor;

/// `allocatedSize` as an M3 Max and an M4 answer it: 256 bytes and 8 per entry, whole 16-KiB pages
/// past 16 KiB.
#[test]
fn a_table_is_allocated_as_metal_allocates_it() {
    let measured: [(u32, u64); 17] = [
        (0, 256),
        (1, 264),
        (2, 272),
        (3, 280),
        (4, 288),
        (5, 296),
        (8, 320),
        (100, 1056),
        (1000, 8256),
        (4096, 49152),
        (65536, 540672),
        (1 << 20, 8404992),
        (1 << 21, 16793600),
        (1 << 22, 33570816),
        (1 << 24, 134234112),
        (1 << 28, 2147500032),
        (1 << 31, 17179885568),
    ];
    for (count, allocated) in measured {
        assert_eq!(table_allocated_size(count), allocated, "{count} entries");
    }
}

/// `MTLVisibleFunctionTableDescriptor`'s default: no entries.
#[test]
fn a_table_descriptor_has_no_entries_by_default() {
    assert_eq!(VisibleFunctionTableDescriptor::default().function_count, 0);
}
