//! A captured draw's key of a buffer's bind offset (`render/captured.rs`): offsets of one class
//! agree on every alignment a binding asks of an offset, and a ring buffer's offsets that move by
//! 256 bytes keep their class.

use crate::render::offset_class;

/// Two offsets of one class are multiples of the same powers of two up to 256: every alignment
/// predicate Vulkan's limits allow (`minUniformBufferOffsetAlignment` and the like are at most 256)
/// reads them alike.
#[test]
fn offsets_of_one_class_agree_on_every_alignment_up_to_256() {
    for a in 0..1024u64 {
        for b in 0..1024u64 {
            if offset_class(a) != offset_class(b) {
                continue;
            }
            for shift in 0..=8 {
                let alignment = 1u64 << shift;
                assert_eq!(
                    a.is_multiple_of(alignment),
                    b.is_multiple_of(alignment),
                    "offsets {a} and {b} of one class, alignment {alignment}"
                );
            }
        }
    }
}

/// An offset moved by a multiple of 256 bytes keeps its class, so the memo answers a ring
/// buffer's draws at their moving offsets from one key.
#[test]
fn an_offset_moved_by_256_bytes_keeps_its_class() {
    for offset in 0..1024u64 {
        for moved in [256, 4096, 1 << 32] {
            assert_eq!(offset_class(offset), offset_class(offset + moved));
        }
    }
}
