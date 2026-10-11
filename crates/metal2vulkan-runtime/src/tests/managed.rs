//! Managed storage (`managed.rs`): its memory class.

use crate::memory::{candidates, MemoryClass};
use ash::vk;

fn ty(flags: vk::MemoryPropertyFlags) -> vk::MemoryType {
    vk::MemoryType {
        property_flags: flags,
        heap_index: 0,
    }
}

#[test]
fn managed_memory_is_host_coherent_and_cached_first() {
    use vk::MemoryPropertyFlags as P;
    let types = [
        ty(P::DEVICE_LOCAL),
        ty(P::DEVICE_LOCAL | P::HOST_VISIBLE | P::HOST_COHERENT),
        ty(P::HOST_VISIBLE | P::HOST_CACHED),
        ty(P::HOST_VISIBLE | P::HOST_COHERENT),
        ty(P::HOST_VISIBLE | P::HOST_COHERENT | P::HOST_CACHED),
    ];
    // Cached coherent first; never non-coherent memory, which would need flushes; never
    // host-invisible memory.
    assert_eq!(
        candidates(&types, 0b11111, MemoryClass::Managed, false),
        [4, 1, 3]
    );
    // Without cached coherent memory, any host-visible coherent type.
    assert_eq!(
        candidates(&types, 0b01111, MemoryClass::Managed, false),
        [1, 3]
    );
}
