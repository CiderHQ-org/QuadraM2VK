use crate::memory::{
    candidates, host_cached, host_cached_placement, host_incoherent, MemoryClass, Pool,
};
use ash::vk;

fn ty(flags: vk::MemoryPropertyFlags) -> vk::MemoryType {
    vk::MemoryType {
        property_flags: flags,
        heap_index: 0,
    }
}

#[test]
fn classes_pick_their_memory_types_best_first() {
    use vk::MemoryPropertyFlags as P;
    let types = [
        ty(P::DEVICE_LOCAL),
        ty(P::HOST_VISIBLE | P::HOST_COHERENT),
        ty(P::DEVICE_LOCAL | P::HOST_VISIBLE | P::HOST_COHERENT),
        ty(P::HOST_VISIBLE | P::HOST_CACHED),
    ];
    let all = 0b1111;
    // Shared needs host-visible coherent memory and prefers host-cached, then device-local (the
    // cached type here is not coherent).
    assert_eq!(candidates(&types, all, MemoryClass::Shared, false), [2, 1]);
    // Private needs device-local memory and avoids host-visible.
    assert_eq!(candidates(&types, all, MemoryClass::Private, false), [0, 2]);
    // Only the types the resource allows.
    assert_eq!(candidates(&types, 0b0011, MemoryClass::Shared, false), [1]);
    assert!(candidates(&types, 0b1000, MemoryClass::Private, false).is_empty());
    // A device with only unified memory (lavapipe, Apple GPUs' shared types): both classes use it.
    let unified = [ty(P::DEVICE_LOCAL | P::HOST_VISIBLE | P::HOST_COHERENT)];
    assert_eq!(candidates(&unified, 1, MemoryClass::Shared, false), [0]);
    assert_eq!(candidates(&unified, 1, MemoryClass::Private, false), [0]);
    // Memoryless prefers lazily allocated memory, which no other class uses.
    let tiler = [
        ty(P::DEVICE_LOCAL),
        ty(P::DEVICE_LOCAL | P::LAZILY_ALLOCATED),
        ty(P::DEVICE_LOCAL | P::HOST_VISIBLE | P::HOST_COHERENT),
    ];
    assert_eq!(
        candidates(&tiler, 0b111, MemoryClass::Memoryless, false),
        [1, 0, 2]
    );
    assert_eq!(
        candidates(&tiler, 0b111, MemoryClass::Private, false),
        [0, 2]
    );
    // Without lazily allocated memory, as `Private`.
    assert_eq!(
        candidates(&types, all, MemoryClass::Memoryless, false),
        [0, 2]
    );
}

/// `Shared` memory prefers a host-cached coherent type, then a device-local one, with unified
/// memory or without; `WriteCombined` memory is `Shared` with unified memory and prefers a
/// device-local type without. A cached type that is not coherent is neither.
#[test]
fn shared_memory_follows_the_cpu_cache_mode_and_unified_memory() {
    use vk::MemoryPropertyFlags as P;
    let host = P::HOST_VISIBLE | P::HOST_COHERENT;
    let (shared, combined) = (MemoryClass::Shared, MemoryClass::WriteCombined);
    // RADV on a Radeon 780M: the cached type is not device-local.
    let radv = [
        ty(P::DEVICE_LOCAL),
        ty(host),
        ty(P::DEVICE_LOCAL | host),
        ty(host | P::HOST_CACHED),
    ];
    for unified in [true, false] {
        assert_eq!(candidates(&radv, 0b1111, shared, unified), [3, 2, 1]);
    }
    assert_eq!(candidates(&radv, 0b1111, combined, true), [3, 2, 1]);
    assert_eq!(candidates(&radv, 0b1111, combined, false), [2, 1, 3]);
    // ANV on an Iris Xe: every type is device-local.
    let anv = [
        ty(P::DEVICE_LOCAL),
        ty(P::DEVICE_LOCAL | host),
        ty(P::DEVICE_LOCAL | host | P::HOST_CACHED),
    ];
    for unified in [true, false] {
        assert_eq!(candidates(&anv, 0b111, shared, unified), [2, 1]);
    }
    assert_eq!(candidates(&anv, 0b111, combined, true), [2, 1]);
    assert_eq!(candidates(&anv, 0b111, combined, false), [1, 2]);
    // NVIDIA on an RTX 3090 (resizable BAR): system memory, cached or not, and BAR memory.
    let nvidia = [
        ty(P::empty()),
        ty(P::DEVICE_LOCAL),
        ty(P::DEVICE_LOCAL),
        ty(host),
        ty(host | P::HOST_CACHED),
        ty(P::DEVICE_LOCAL | host),
    ];
    assert_eq!(candidates(&nvidia, 0b11_1111, shared, false), [4, 5, 3]);
    assert_eq!(candidates(&nvidia, 0b11_1111, combined, false), [5, 3, 4]);
    // A cached type that is not coherent is never taken.
    let incoherent = [
        ty(P::DEVICE_LOCAL | host),
        ty(P::HOST_VISIBLE | P::HOST_CACHED),
    ];
    for class in [shared, combined] {
        for unified in [true, false] {
            assert_eq!(candidates(&incoherent, 0b11, class, unified), [0]);
        }
    }
    // The other classes do not depend on unified memory.
    for class in [
        MemoryClass::Private,
        MemoryClass::Managed,
        MemoryClass::Memoryless,
    ] {
        assert_eq!(
            candidates(&radv, 0b1111, class, true),
            candidates(&radv, 0b1111, class, false),
            "{class:?}"
        );
    }
}

/// No class takes a device-coherent or device-uncached type of `VK_AMD_device_coherent_memory`,
/// whose allocation needs a feature the runtime does not enable, even where it is the only type
/// the resource allows or ranks best by its other properties.
#[test]
fn no_class_takes_amd_device_coherent_memory() {
    use vk::MemoryPropertyFlags as P;
    let host = P::HOST_VISIBLE | P::HOST_COHERENT;
    let amd = P::DEVICE_COHERENT_AMD | P::DEVICE_UNCACHED_AMD;
    // RADV on a Radeon 780M lists them after its other types; here they come first as well.
    let types = [
        ty(P::DEVICE_LOCAL | host | P::HOST_CACHED | amd),
        ty(P::DEVICE_LOCAL),
        ty(host),
        ty(P::DEVICE_LOCAL | host),
        ty(host | P::HOST_CACHED),
        ty(P::DEVICE_LOCAL | amd),
        ty(P::DEVICE_LOCAL | host | P::DEVICE_COHERENT_AMD),
        ty(host | P::HOST_CACHED | P::DEVICE_UNCACHED_AMD),
    ];
    let all = (1 << types.len()) - 1;
    for unified in [false, true] {
        for class in [
            MemoryClass::Shared,
            MemoryClass::WriteCombined,
            MemoryClass::Private,
            MemoryClass::Managed,
            MemoryClass::Memoryless,
        ] {
            let picked = candidates(&types, all, class, unified);
            assert!(!picked.is_empty(), "{class:?}");
            for index in picked {
                assert!(
                    !types[index as usize].property_flags.intersects(amd),
                    "{class:?} took type {index}"
                );
            }
            assert!(
                candidates(&types, 0b1110_0001, class, unified).is_empty(),
                "{class:?}"
            );
        }
    }
}

/// A texture's image with host-transfer usage takes host-cached coherent memory of its class
/// (`Private`) where the device has it among the image's types, else, with unified memory,
/// host-cached `Shared` memory where the device has it among them, else the class's memory as any
/// image does (a copy through a submission could queue behind the program's work), never a type
/// the host maps without coherence (ANV's cached type on Arrow Lake-S). On the memory types and
/// host-transfer image memory type bits measured (EVs XSH-hrp2-3090, XSH-hrp4-radv), the RTX 3090
/// has none, so its images take resizable BAR, and the Radeon 780M (unified) takes its host-cached
/// system memory, type 5, where it took RADV's uncached device-local type before. MoltenVK's shared
/// type on Apple GPUs (`MVK_VK_MEMORY_TYPE_METAL_SHARED` with `DEVICE_LOCAL`, MoltenVK 1.4.1) and
/// lavapipe's one type are host-cached and device-local; a device with both kinds of device-local
/// host-visible memory takes the cached one where `Private`'s ranking alone would take the first.
#[test]
fn host_transfer_images_prefer_host_cached_memory() {
    let types = |raw: &[u32]| -> Vec<vk::MemoryType> {
        raw.iter()
            .map(|&flags| ty(vk::MemoryPropertyFlags::from_raw(flags)))
            .collect()
    };
    let private = MemoryClass::Private;
    // The class the image takes and the types it may lie in, best first.
    let picked = |types: &[vk::MemoryType], allowed: u32, unified: bool| {
        let (class, taken) =
            host_cached_placement(types, allowed, private, unified).unwrap_or((private, allowed));
        (class, candidates(types, taken, class, unified))
    };
    // NVIDIA 615.71.09 on an RTX 3090: a host-transfer image may lie in types 3 to 5.
    let nvidia = types(&[0x0, 0x1, 0x1, 0x6, 0xe, 0x7]);
    assert_eq!(host_cached(&nvidia), 0b1_0000);
    assert_eq!(
        host_cached_placement(&nvidia, 0b11_1000, private, false),
        None
    );
    assert_eq!(picked(&nvidia, 0b11_1000, false), (private, vec![5]));
    // RADV (Mesa 26.2.4) on a Radeon 780M: types 2, 3, 5 and the AMD device-coherent 14 to 16.
    let radv = types(&[
        0x1, 0x1, 0x6, 0x7, 0x7, 0xe, 0xe, 0x21, 0x21, 0x21, 0x21, 0x20, 0x20, 0xc1, 0xc6, 0xc7,
        0xce,
    ]);
    let allowed = 0b1_1100_0000_0010_1100;
    assert_eq!(picked(&radv, allowed, true), (MemoryClass::Shared, vec![5]));
    // The same memory on a device without unified memory: its device-local uncached type.
    assert_eq!(host_cached_placement(&radv, allowed, private, false), None);
    assert_eq!(picked(&radv, allowed, false), (private, vec![3]));
    // An image that may lie in no host-cached type keeps its class's memory.
    assert_eq!(picked(&radv, 0b1000, true), (private, vec![3]));
    // MoltenVK on an Apple GPU: private, shared (device-local and host-cached), memoryless.
    let moltenvk = types(&[0x1, 0xf, 0x11]);
    assert_eq!(host_cached(&moltenvk), 0b010);
    assert_eq!(picked(&moltenvk, 0b111, true), (private, vec![1]));
    // lavapipe: one type, device-local and host-cached.
    assert_eq!(picked(&types(&[0xf]), 1, true), (private, vec![0]));
    // Device-local host-visible memory both uncached (listed first) and cached.
    let both = types(&[0x1, 0x7, 0xf]);
    assert_eq!(candidates(&both, 0b110, private, true), [1, 2]);
    assert_eq!(picked(&both, 0b110, true), (private, vec![2]));
    // ANV (Mesa 26.2.4) on Arrow Lake-S's Xe-LPG (0x7d67), which shares no cache with the host:
    // coherent write-combined types 0 and 3, cached type 1 without coherence, protected type 2.
    // No type is host-cached and coherent, so the image takes its class's memory, without the
    // incoherent type (`placed_image` strips it): the coherent type 0 first.
    let arl = types(&[0x7, 0xb, 0x21, 0x7]);
    assert_eq!(host_cached(&arl), 0);
    assert_eq!(host_incoherent(&arl), 0b0010);
    assert_eq!(host_cached_placement(&arl, 0b1011, private, true), None);
    let coherent = 0b1011 & !host_incoherent(&arl);
    assert_eq!(picked(&arl, coherent, true), (private, vec![0, 3]));
    // Every device above lists only coherent host-visible types.
    for listed in [nvidia, radv, moltenvk, types(&[0xf]), both] {
        assert_eq!(host_incoherent(&listed), 0);
    }
}

#[test]
fn pooled_blocks_place_until_full_and_one_empty_block_is_kept() {
    let mut pool: Pool<&str> = Pool::default();
    assert_eq!(pool.place(16, 16), None, "no block yet");
    let first = pool.add(256, false, "first", 100);
    assert_eq!(pool.place(100, 64), Some((first, 128)));
    assert_eq!(pool.place(100, 64), None, "full");
    let second = pool.add(256, false, "second", 200);
    assert_ne!(first, second);
    assert_eq!(pool.blocks(), 2);
    // Emptying the first block keeps it: it is the only empty pooled block.
    assert_eq!(pool.release(first, 0, 100), None);
    assert_eq!(pool.release(first, 128, 100), None);
    assert_eq!(pool.blocks(), 2);
    // Emptying the second while the first is empty frees the second.
    assert_eq!(pool.release(second, 0, 200), Some("second"));
    assert_eq!(pool.blocks(), 1);
    // The kept block serves the next request; its slot is reused for a new block.
    assert_eq!(pool.place(256, 1), Some((first, 0)));
    let third = pool.add(256, false, "third", 1);
    assert_eq!(third, second, "the freed slot is reused");
    let mut drained: Vec<&str> = pool.drain().collect();
    drained.sort_unstable();
    assert_eq!(drained, ["first", "third"]);
    assert_eq!(pool.blocks(), 0);
}

#[test]
fn dedicated_blocks_take_no_other_placement_and_free_with_their_resource() {
    let mut pool: Pool<u32> = Pool::default();
    let dedicated = pool.add(1000, true, 7, 1000);
    assert_eq!(pool.place(1, 1), None, "a dedicated block is never shared");
    assert_eq!(pool.release(dedicated, 0, 1000), Some(7));
    assert_eq!(pool.blocks(), 0);
}

/// A pool has room for a placement where a pooled block holds a free range that long at the
/// largest alignment asked of the pool so far; a dedicated block's room counts for none.
#[test]
fn room_is_a_pooled_block_s_free_range_at_the_largest_alignment_asked() {
    let mut pool: Pool<u32> = Pool::default();
    assert!(!pool.has_room(1), "no block yet");
    pool.add(1000, true, 1, 10);
    assert!(
        !pool.has_room(1),
        "a dedicated block's free bytes are no room"
    );
    let block = pool.add(256, false, 2, 0);
    assert!(pool.has_room(256), "an empty block");
    assert_eq!(pool.place(8, 1), Some((block, 0)));
    assert!(pool.has_room(248) && !pool.has_room(249));
    // A placement aligned to 64 asks the pool's later room at 64: 192 bytes from offset 64.
    assert_eq!(pool.place(8, 64), Some((block, 64)));
    assert!(pool.has_room(128) && !pool.has_room(129));
    assert_eq!(pool.release(block, 64, 8), None);
    assert!(pool.has_room(192) && !pool.has_room(193));
}

#[test]
fn buffers_are_placed_at_the_largest_linear_image_alignment() {
    use crate::buffer::{placed, LINEAR_IMAGE_ALIGNMENT};
    let requirements = |alignment| vk::MemoryRequirements {
        size: 100,
        alignment,
        memory_type_bits: 0b101,
    };
    // Each linear-image alignment measured divides it (RADV, NVIDIA, lavapipe, MoltenVK).
    for measured in [256, 128, 64, 16] {
        assert!(LINEAR_IMAGE_ALIGNMENT.is_multiple_of(measured));
    }
    // A smaller alignment is raised to it, a larger one is kept, and the size takes whole units
    // of the alignment; the memory types are kept.
    for (alignment, placed_at) in [(1, 256), (16, 256), (256, 256), (4096, 4096)] {
        let placed = placed(requirements(alignment));
        assert_eq!(placed.alignment, placed_at);
        assert_eq!((placed.size, placed.memory_type_bits), (placed_at, 0b101));
    }
}

/// Buffers shorter than their placement's alignment, placed one after another, leave no gap:
/// each starts where the last ended, so the block's free ranges stay one and the pool's
/// first-fit search does not walk a gap per buffer placed before (65,537 buffers of 16 bytes
/// took 201 s to make on lavapipe, each later one longer).
#[test]
fn short_buffers_are_placed_without_gaps() {
    use crate::buffer::placed;
    let mut pool: Pool<()> = Pool::default();
    let block = pool.add(1 << 20, false, (), 0);
    let mut end = 0;
    for length in [16, 1, 100, 255, 256, 257, 16, 4000] {
        let requirements = placed(vk::MemoryRequirements {
            size: length,
            alignment: 16,
            memory_type_bits: 1,
        });
        let (at, offset) = pool
            .place(requirements.size, requirements.alignment)
            .unwrap();
        assert_eq!((at, offset), (block, end), "{length} bytes");
        end = offset + requirements.size;
    }
}
