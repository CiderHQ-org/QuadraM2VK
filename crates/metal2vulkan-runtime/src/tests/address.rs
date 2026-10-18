//! The Metal address space (`address.rs`): ranges the runtime picks, and pinned ones as Metal
//! honours them (on an M3 Max and an M4).

use crate::address::{
    AddressPlacement, AddressRange, AddressSpace, GpuAddress, Place, PlacementConflict, RecordsId,
    GRANULE, OUTSIDE_END, PAGE, SPACE_BASE, SPACE_END,
};
use crate::error::Error;
use crate::range::RangeAllocator;
use metal2vulkan::bind::indirect_commands::{records_address, RECORDS_ID_LIMIT};

#[test]
fn spans_are_whole_pages() {
    assert_eq!(AddressSpace::span(0), Some(PAGE));
    assert_eq!(AddressSpace::span(1), Some(PAGE));
    assert_eq!(AddressSpace::span(PAGE), Some(PAGE));
    assert_eq!(AddressSpace::span(PAGE + 1), Some(2 * PAGE));
    assert_eq!(AddressSpace::span(u64::MAX), None);
}

#[test]
fn the_space_is_from_2_to_the_32_to_2_to_the_47() {
    assert_eq!(SPACE_BASE, 1 << 32);
    assert_eq!(SPACE_END, 1 << 47);
}

#[test]
fn addresses_are_page_aligned_above_32_bits_and_disjoint() {
    let mut space = AddressSpace::new();
    let lengths = [1, PAGE, PAGE + 1, 3 * PAGE - 5, 64];
    let mut held = Vec::new();
    for length in lengths {
        let (address, hold) = space.allocate(length).unwrap();
        assert!(address.0 >= SPACE_BASE && address.0 < SPACE_END);
        assert_eq!(address.0 % PAGE, 0);
        let span = AddressSpace::span(length).unwrap();
        for &(other, other_length, _) in &held {
            let other_span = AddressSpace::span(other_length).unwrap();
            let other: GpuAddress = other;
            assert!(
                address.0 + span <= other.0 || other.0 + other_span <= address.0,
                "{address:?} overlaps {other:?}"
            );
        }
        held.push((address, length, hold));
    }
    assert_eq!(
        held[0].0,
        GpuAddress(SPACE_BASE),
        "never 0, never below 2^32"
    );
    for (_, _, hold) in held {
        space.free(hold);
    }
    assert!(space.is_empty());
}

#[test]
fn a_freed_range_comes_back_and_the_space_ends() {
    let mut space = AddressSpace::new();
    let (a, a_hold) = space.allocate(10).unwrap();
    let (_, b_hold) = space.allocate(10).unwrap();
    space.free(a_hold);
    let (again, again_hold) = space.allocate(PAGE).unwrap();
    assert_eq!(again, a);
    assert!(
        space.allocate(SPACE_END - SPACE_BASE).is_none(),
        "past the space"
    );
    space.free(again_hold);
    space.free(b_hold);
    assert!(space.is_empty());
}

/// The range a pin of `length` bytes at `start` needs, as a second pin there reports it.
fn range(start: u64, length: u64) -> AddressRange {
    let mut space = AddressSpace::new();
    space.pin(GpuAddress(start), length).unwrap();
    match space.pin(GpuAddress(start), length) {
        Err(PlacementConflict::Held { requested, .. }) => requested,
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_pin_holds_its_pages_and_guard_pages_after_them() {
    let x = 0x110_0011_8000;
    // A one-page range holds one guard page after it (probes 1 to 3): the page after the guard
    // takes a pin, the guard page does not.
    let mut space = AddressSpace::new();
    space.pin(GpuAddress(x), PAGE).unwrap();
    assert!(
        space.pin(GpuAddress(x + PAGE), PAGE).is_err(),
        "the guard page"
    );
    space.pin(GpuAddress(x + 2 * PAGE), PAGE).unwrap();
    // The page before cannot take a pin either: its own guard page is `x`'s first.
    assert!(space.pin(GpuAddress(x - PAGE), PAGE).is_err());
    space.pin(GpuAddress(x - 2 * PAGE), PAGE).unwrap();
    // A longer range of an even count holds two: its guard page and the page that rounds the
    // span to 32 KiB (16 KiB + 1, a 64-KiB heap, a 1-MiB buffer or heap; `a_pin_holds_metal_s_span`
    // takes the odd counts).
    for length in [PAGE + 1, 4 * PAGE, 64 * PAGE] {
        let mut space = AddressSpace::new();
        let pages = AddressSpace::span(length).unwrap();
        space.pin(GpuAddress(x), length).unwrap();
        assert!(space.pin(GpuAddress(x + pages), PAGE).is_err(), "{length}");
        assert!(
            space.pin(GpuAddress(x + pages + PAGE), PAGE).is_err(),
            "{length}"
        );
        space.pin(GpuAddress(x + pages + 2 * PAGE), PAGE).unwrap();
    }
    assert_eq!(range(x, PAGE), AddressRange::of(x, 2 * PAGE));
    assert_eq!(range(x, 100), AddressRange::of(x, 2 * PAGE));
    assert_eq!(range(x, PAGE + 1), AddressRange::of(x, 4 * PAGE));
}

/// A pin of `n` pages holds them and one guard page, rounded up to 32 KiB, as an Apple M4 holds
/// a pinned buffer's and heap's (a 1-page pin is made at the first page after
/// that span and at none before it, for 1 to 8, 15 to 17, 64 and 65 pages, and for each and a
/// byte); the page after an odd count's guard page is free, where the runtime once held two guard
/// pages after a range of more than one page.
#[test]
fn a_pin_holds_metal_s_span() {
    let x = 0x110_0011_8000;
    for (pages, span) in [
        (1, 2),
        (2, 4),
        (3, 4),
        (4, 6),
        (5, 6),
        (6, 8),
        (7, 8),
        (8, 10),
        (15, 16),
        (16, 18),
        (17, 18),
        (64, 66),
        (65, 66),
    ] {
        assert_eq!(
            range(x, pages * PAGE),
            AddressRange::of(x, span * PAGE),
            "{pages}"
        );
        assert_eq!(
            range(x, pages * PAGE - 1),
            AddressRange::of(x, span * PAGE),
            "{pages}, a byte short"
        );
        let one_more = (pages + 2).next_multiple_of(2);
        assert_eq!(
            range(x, pages * PAGE + 1),
            AddressRange::of(x, one_more * PAGE),
            "{pages} and a byte"
        );
        let mut space = AddressSpace::new();
        space.pin(GpuAddress(x), pages * PAGE).unwrap();
        assert!(
            space.pin(GpuAddress(x + (span - 2) * PAGE), PAGE).is_err(),
            "{pages}"
        );
        space.pin(GpuAddress(x + span * PAGE), PAGE).unwrap();
    }
}

/// A pin whose first page is not a 32-KiB granule's is refused `OffGranule`, wherever it lies
/// (Metal answered nil for every one, an M4 and an M3 Max); one at a byte of a granule's first
/// page is made.
#[test]
fn a_pin_off_a_granule_is_refused() {
    let x = 0x110_0011_8000;
    let mut space = AddressSpace::new();
    for at in [x + PAGE, x + PAGE + 0x10, x + 3 * PAGE] {
        assert!(
            matches!(
                space.pin(GpuAddress(at), PAGE),
                Err(PlacementConflict::OffGranule { .. })
            ),
            "{at:#x}"
        );
    }
    assert!(space.is_empty());
    space.pin(GpuAddress(x + 0x10), PAGE).unwrap();
}

#[test]
fn an_unaligned_pin_keeps_its_address_and_holds_the_pages_it_touches() {
    // Metal made a 16-KiB buffer at a page plus 1, 8, 256, 4096 and 8192 bytes, each answering
    // that address, allocated 32 KiB, and blocking two pages after them.
    let x = 0x110_0011_8000;
    for offset in [1, 8, 256, 4096, 8192] {
        let mut space = AddressSpace::new();
        let hold = space.pin(GpuAddress(x + offset), PAGE).unwrap();
        assert_eq!(range(x + offset, PAGE), AddressRange::of(x, 4 * PAGE));
        assert!(
            space.pin(GpuAddress(x + 3 * PAGE), PAGE).is_err(),
            "{offset}"
        );
        space.pin(GpuAddress(x + 4 * PAGE), PAGE).unwrap();
        space.free(hold);
        space.pin(GpuAddress(x), PAGE).unwrap();
    }
}

#[test]
fn a_pin_into_held_bytes_names_them_and_changes_nothing() {
    let mut space = AddressSpace::new();
    // A granule free before the runtime's range, so a pin can overlap its head.
    let (_, pad) = space.allocate(GRANULE).unwrap();
    let (any, _) = space.allocate(4 * PAGE).unwrap();
    space.free(pad);
    // The same address, inside, and overlapping the head: Metal answered nil for each.
    for (at, length) in [
        (any.0, PAGE),
        (any.0 + 2 * PAGE, PAGE),
        (any.0 - 2 * PAGE, 3 * PAGE),
    ] {
        match space.pin(GpuAddress(at), length) {
            Err(PlacementConflict::Held { held, .. }) => {
                assert_eq!(held, AddressRange::of(any.0, 4 * PAGE), "{at:#x}");
            }
            other => panic!("{at:#x}: {other:?}"),
        }
    }
    let x = 0x110_0011_8000;
    let first = space.pin(GpuAddress(x), 2 * PAGE).unwrap();
    assert!(
        space.pin(GpuAddress(x), PAGE).is_err(),
        "the same address twice"
    );
    assert!(
        space.pin(GpuAddress(x + PAGE), 2 * PAGE).is_err(),
        "its tail"
    );
    // A pinned range never comes back to the runtime's picks.
    for _ in 0..64 {
        let (picked, _) = space.allocate(PAGE).unwrap();
        assert!(
            picked.0 + PAGE <= x || picked.0 >= x + 4 * PAGE,
            "{picked:?}"
        );
    }
    // Freed, its guard pages go with it: the address and the page after take pins again.
    space.free(first);
    space.pin(GpuAddress(x), PAGE).unwrap();
    space.pin(GpuAddress(x + 2 * PAGE), PAGE).unwrap();
}

#[test]
fn a_shared_range_is_free_once_every_holder_is_gone() {
    let mut space = AddressSpace::new();
    let x = 0x110_0011_8000;
    let first = space.pin(GpuAddress(x + 0x40), 2 * PAGE).unwrap();
    // Two buffers share it at its address, as each answers.
    for _ in 0..2 {
        let shared = Place::Shared {
            address: GpuAddress(x + 0x40),
            held: first,
        };
        assert_eq!(
            space.place(shared, PAGE).unwrap(),
            (GpuAddress(x + 0x40), first)
        );
    }
    // Held while any holder lives, whichever goes first.
    for gone in 0..3 {
        assert!(space.pin(GpuAddress(x), PAGE).is_err(), "{gone} gone: held");
        space.free(first);
    }
    assert!(space.is_empty(), "free once all three are gone");
    space.pin(GpuAddress(x), PAGE).unwrap();
}

/// A pool holds a pin's span of its length; its slots hold their lengths rounded up to 256 bytes
/// of it, at 256-byte-aligned offsets, as Metal's pools hold buffers shorter than a page (the
/// consumer's `vaprobe smallpool`, an M3 Max): slots share pages, and one that meets a live slot,
/// is off a slot's alignment or passes the pool's end is refused; a freed slot is taken again; a
/// released pool's span is free once its last slot is.
#[test]
fn a_pool_holds_a_pins_span_and_its_slots_their_bytes_rounded_to_256() {
    let mut space = AddressSpace::new();
    let x = 0x100_0008_0000;
    let length = 0x20000;
    let pool = AddressRange::of(x, length);
    assert_eq!(
        space.reserve_pool(AddressPlacement::At(GpuAddress(x)), length),
        Ok(GpuAddress(x))
    );
    // The span of a pin of 0x20000 bytes: 0x28000, its guard page within.
    assert!(space.pin(GpuAddress(x + length), PAGE).is_err(), "guard");
    let beyond = space.pin(GpuAddress(x + 0x28000), PAGE).unwrap();
    space.free(beyond);
    let slot = |space: &mut AddressSpace, offset: u64, length: u64| {
        let place = Place::InPool {
            pool: x,
            offset,
            placement: AddressPlacement::Any,
        };
        space.place(place, length)
    };
    // Metal's measured slot sizes, one after the other.
    let sizes = [
        (1, 0x100),
        (0x40, 0x100),
        (0xff, 0x100),
        (0x100, 0x100),
        (0x101, 0x200),
        (0x180, 0x200),
        (0x300, 0x300),
        (0x1001, 0x1100),
        (0x3fff, 0x4000),
    ];
    let mut offset = 0;
    let mut held = Vec::new();
    for (length, size) in sizes {
        let (address, slot_held) = slot(&mut space, offset, length).unwrap();
        assert_eq!(address, GpuAddress(x + offset), "{length:#x}");
        held.push((offset, slot_held));
        offset += size;
    }
    // The next slot starts right after the last one's rounded bytes; the one before it meets
    // the run of slots held.
    let (_, next) = slot(&mut space, offset, 0x100).unwrap();
    let conflict = |error| match error {
        Error::PlacementConflict(conflict) => conflict,
        other => panic!("{other:?}"),
    };
    assert_eq!(
        conflict(slot(&mut space, offset - 0x100, 1).unwrap_err()),
        PlacementConflict::Held {
            requested: AddressRange::of(x + offset - 0x100, 0x100),
            held: AddressRange::of(x, offset + 0x100),
        }
    );
    // Off a slot's alignment, or named elsewhere by the placement.
    let off = PlacementConflict::OffSlot {
        requested: AddressRange::of(x + 0x10080, 0x100),
        pool,
    };
    assert_eq!(conflict(slot(&mut space, 0x10080, 0x10).unwrap_err()), off);
    for placement in [
        AddressPlacement::At(GpuAddress(x + 0x10100)),
        AddressPlacement::Outside,
    ] {
        let place = Place::InPool {
            pool: x,
            offset: 0x10000,
            placement,
        };
        let refused = conflict(space.place(place, 0x10).unwrap_err());
        assert!(matches!(refused, PlacementConflict::OffSlot { .. }));
    }
    let named = Place::InPool {
        pool: x,
        offset: 0x10000,
        placement: AddressPlacement::At(GpuAddress(x + 0x10000)),
    };
    let (address, named) = space.place(named, 0x10).unwrap();
    assert_eq!(address, GpuAddress(x + 0x10000));
    // Past the pool's end; the last slot is in it.
    assert_eq!(
        conflict(slot(&mut space, length - 0x100, 0x101).unwrap_err()),
        PlacementConflict::OutsidePool {
            requested: AddressRange::of(x + length - 0x100, 0x200),
            pool,
        }
    );
    assert!(matches!(
        conflict(slot(&mut space, u64::MAX - 0xff, 1).unwrap_err()),
        PlacementConflict::OutsidePool { .. }
    ));
    let (_, last) = slot(&mut space, length - 0x100, 0x100).unwrap();
    // A freed slot is taken again, by as many bytes as it held and no more.
    let (offset, freed) = held[4];
    space.free(freed);
    assert!(slot(&mut space, offset, 0x201).is_err());
    let (_, again) = slot(&mut space, offset, 0x200).unwrap();
    held[4] = (offset, again);
    // Released while slots live: its span stays held until the last is gone.
    space.release_pool(GpuAddress(x));
    for (_, slot_held) in held {
        space.free(slot_held);
    }
    space.free(next);
    space.free(named);
    assert!(space.pin(GpuAddress(x), PAGE).is_err(), "a slot lives");
    space.free(last);
    assert!(space.is_empty());
    let pinned = space.pin(GpuAddress(x), PAGE).unwrap();
    space.free(pinned);

    // A pool the runtime picks holds the same span at a granule, in the space or outside it; an
    // empty one released is free at once, and one of 0 bytes holds no slot.
    let picked = space.reserve_pool(AddressPlacement::Any, length).unwrap();
    assert_eq!(picked.0 % GRANULE, 0);
    let (after, after_held) = space.allocate(1).unwrap();
    assert_eq!(after.0, picked.0 + 0x28000);
    space.free(after_held);
    let outside = space
        .reserve_pool(AddressPlacement::Outside, length)
        .unwrap();
    assert!(outside.0 >= SPACE_END);
    let empty = space
        .reserve_pool(AddressPlacement::At(GpuAddress(x)), 0)
        .unwrap();
    let place = Place::InPool {
        pool: empty.0,
        offset: 0,
        placement: AddressPlacement::Any,
    };
    assert!(matches!(
        conflict(space.place(place, 1).unwrap_err()),
        PlacementConflict::OutsidePool { .. }
    ));
    for pool in [picked, outside, empty] {
        space.release_pool(pool);
    }
    assert!(space.is_empty());
}

#[test]
fn pins_outside_the_space_are_refused() {
    let mut space = AddressSpace::new();
    for at in [
        0,
        0x4000,
        0xffff_c000,
        (1 << 32) - PAGE,
        (1 << 47) - PAGE,
        1 << 47,
        (1 << 48) - PAGE,
        1 << 56,
        u64::MAX - 1,
    ] {
        assert!(
            matches!(
                space.pin(GpuAddress(at), PAGE),
                Err(PlacementConflict::OutsideSpace { .. })
            ),
            "{at:#x}"
        );
    }
    // Inside it, every free range is pinned, Metal's process layout no contract.
    for at in [
        1 << 32,
        1 << 33,
        1 << 36,
        1 << 39,
        1 << 42,
        (1 << 47) - 2 * PAGE,
    ] {
        space.pin(GpuAddress(at), PAGE).unwrap();
    }
    assert!(space.pin(GpuAddress((1 << 47) - 3 * PAGE), PAGE).is_err());
}

#[test]
fn allocate_at_takes_exactly_free_bytes_and_names_the_held_run() {
    let mut ranges = RangeAllocator::new(0, 100);
    assert_eq!(ranges.allocate_at(10, 10), Ok(()));
    assert_eq!(ranges.allocate_at(20, 5), Ok(()));
    // Inside the held run [10, 25), from before it, and from its middle.
    assert_eq!(ranges.allocate_at(12, 2), Err((10, 15)));
    assert_eq!(ranges.allocate_at(5, 10), Err((10, 15)));
    assert_eq!(ranges.allocate_at(24, 10), Err((10, 15)));
    assert_eq!(ranges.used(), 15);
    assert_eq!(ranges.allocate_at(0, 10), Ok(()));
    assert_eq!(ranges.allocate_at(0, 1), Err((0, 25)));
    assert_eq!(ranges.allocate_at(25, 75), Ok(()));
    assert_eq!(ranges.allocate_at(99, 1), Err((0, 100)));
    ranges.free(10, 10);
    ranges.free(0, 10);
    ranges.free(20, 5);
    ranges.free(25, 75);
    assert!(ranges.is_empty());
}

/// Ranges outside the space, `[2^47, 2^48)`, the runtime's own buffers' and
/// `AddressPlacement::Outside`'s: page-aligned, never moving where the space's next range lies,
/// given back to their own space, and out of reach of a pin.
#[test]
fn ranges_outside_the_space_leave_its_ranges_alone() {
    assert_eq!(OUTSIDE_END, 1 << 48);
    let mut space = AddressSpace::new();
    let (first, first_held) = space.place(AddressPlacement::Any, PAGE).unwrap();
    space.free(first_held);
    let (outside, outside_held) = space
        .place(AddressPlacement::Outside, 3 * PAGE + 1)
        .unwrap();
    assert!(
        outside.0 >= SPACE_END && outside.0 < OUTSIDE_END,
        "{outside:?}"
    );
    assert_eq!(outside.0 % PAGE, 0);
    // The space's next range is where it was before the outside range was taken.
    let (again, again_held) = space.place(AddressPlacement::Any, PAGE).unwrap();
    assert_eq!(again, first);
    // A pin never reaches the outside range.
    assert!(matches!(
        space.pin(outside, PAGE),
        Err(PlacementConflict::OutsideSpace { .. })
    ));
    let (next, next_held) = space.place(AddressPlacement::Outside, PAGE).unwrap();
    assert!(next.0 >= outside.0 + 4 * PAGE, "{next:?} after {outside:?}");
    space.free(outside_held);
    let (reused, reused_held) = space.place(AddressPlacement::Outside, PAGE).unwrap();
    assert_eq!(reused, outside);
    for held in [again_held, next_held, reused_held] {
        space.free(held);
    }
    assert!(space.is_empty());
}

/// Indirect command buffers' records: each buffer takes the lowest free records ID, from 1, and
/// its records lie at `records_address` of it; a gone one's ID is the lowest free again, whatever
/// order the buffers went in (Metal's measured order); the records move no range of the space or
/// outside it, and hold nothing of them.
#[test]
fn records_take_the_lowest_free_id_and_leave_the_spaces_alone() {
    let mut space = AddressSpace::new();
    let (first, first_held) = space.place(AddressPlacement::Any, PAGE).unwrap();
    space.free(first_held);
    let mut held: Vec<RecordsId> = (1..=4)
        .map(|id| {
            let held = space.take_records_id(None).unwrap();
            assert_eq!(held.id, id);
            let (at, records) = space.place(Place::Records(held.id), 16 + 2248).unwrap();
            assert_eq!(at.0, records_address(id));
            space.free(records);
            held
        })
        .collect();
    let (again, again_held) = space.place(AddressPlacement::Any, PAGE).unwrap();
    assert_eq!(again, first);
    // 3, then 1, gone: the next take 1, 3 and 5.
    space.give_records_id(held.remove(2));
    space.give_records_id(held.remove(0));
    for id in [1, 3, 5] {
        let next = space.take_records_id(None).unwrap();
        assert_eq!(next.id, id);
        held.push(next);
    }
    for id in held {
        space.give_records_id(id);
    }
    space.free(again_held);
    assert!(space.is_empty());
    assert!(
        space.place(Place::Records(RECORDS_ID_LIMIT), PAGE).is_err(),
        "no records address"
    );
    assert!(
        space.place(Place::Records(1), (1 << 40) + 1).is_err(),
        "past one ID's records"
    );
}

/// Indirect command buffers' records IDs after a reservation are the ones Metal hands out, from a
/// pool of 32 (measured on an M4: predicted before it was measured), and only the first reservation
/// with an ID counts.
#[test]
fn records_ids_after_a_reservation_are_metal_s() {
    let cases: [(&[u64], &[(u64, u64)]); 3] = [
        (&[17, 20], &[(18, 19), (21, 31), (1, 16), (32, 74)]),
        (&[1, 40], &[(41, 63), (2, 39), (64, 138)]),
        (
            &[5, 13, 40, 41, 90],
            &[(1, 4), (6, 12), (14, 39), (91, 127), (42, 89), (128, 145)],
        ),
    ];
    for (reserved, runs) in cases {
        let mut space = AddressSpace::new();
        assert!(space.reserve_records(&[]), "a call of no ID");
        assert!(space.reserve_records(reserved));
        assert!(!space.reserve_records(&[2]), "the second reservation");
        let mut held = Vec::new();
        for id in runs.iter().flat_map(|&(first, last)| first..=last) {
            let next = space.take_records_id(None).unwrap();
            assert_eq!(next.id, id, "{reserved:?}");
            held.push(next);
        }
        for id in held {
            space.give_records_id(id);
        }
    }
}

/// A placed indirect command buffer's ID is not the pool's (on an M4, plain and
/// under API validation): plain allocation neither skips it (1, 2, then 3 placed, then 3 and 4
/// plain) nor gets it back once the placed buffer is gone (1 to 4 plain, 2 placed and gone: 5, 6
/// and 7, then 2 once the plain 2 is gone), and a placement counts as an allocation for a later
/// reservation (9 placed, then a reservation of 2 ignored).
#[test]
fn a_placed_records_id_is_not_the_pool_s() {
    let mut space = AddressSpace::new();
    let plain = |space: &mut AddressSpace| space.take_records_id(None).unwrap();
    let ids: Vec<RecordsId> = vec![plain(&mut space), plain(&mut space)];
    let placed = space.take_records_id(Some(3)).unwrap();
    assert_eq!(placed.id, 3);
    let more = [plain(&mut space), plain(&mut space)];
    assert_eq!(
        ids.iter()
            .chain(&more)
            .map(|held| held.id)
            .collect::<Vec<_>>(),
        [1, 2, 3, 4]
    );
    for held in ids.into_iter().chain(more).chain([placed]) {
        space.give_records_id(held);
    }
    assert!(space.is_empty());

    let mut space = AddressSpace::new();
    let mut held: Vec<RecordsId> = (0..4).map(|_| plain(&mut space)).collect();
    let placed = space.take_records_id(Some(2)).unwrap();
    space.give_records_id(placed);
    for id in [5, 6, 7] {
        let next = plain(&mut space);
        assert_eq!(next.id, id);
        held.push(next);
    }
    space.give_records_id(held.remove(1));
    assert_eq!(plain(&mut space).id, 2);

    let mut space = AddressSpace::new();
    space.take_records_id(Some(9)).unwrap();
    assert!(
        !space.reserve_records(&[2]),
        "a reservation after a placement"
    );
    assert_eq!(plain(&mut space).id, 1);
}

/// Each records ID has one slot, the buffer made last with it, placed or plain, which a kernel
/// encoding through the ID reaches; once it is gone the ID reaches none, even while an earlier
/// buffer with it lives, and an earlier one's going changes nothing (on an M4,
/// the `r_` cases: a kernel's encode through the ID, each buffer executed alone afterwards).
#[test]
fn the_buffer_made_last_with_a_records_id_holds_its_slot() {
    // A plain buffer, then one placed at its ID: the placed one is reached, and once it is gone
    // neither is.
    let mut space = AddressSpace::new();
    let a = space.take_records_id(None).unwrap();
    let b = space.take_records_id(Some(a.id)).unwrap();
    assert_eq!((a.id, b.id), (1, 1));
    assert!(!space.holds_records_slot(a));
    assert!(space.holds_records_slot(b));
    space.give_records_id(b);
    assert!(!space.holds_records_slot(a));
    // The plain one gone too: the ID is free, and the next buffer takes it and its slot.
    space.give_records_id(a);
    let c = space.take_records_id(None).unwrap();
    assert_eq!(c.id, 1);
    assert!(space.holds_records_slot(c));
    space.give_records_id(c);
    assert!(space.is_empty());

    // A placed buffer, then the earlier plain one gone: the placed one is still reached; then a
    // plain buffer takes the ID again, and its slot.
    let mut space = AddressSpace::new();
    let a = space.take_records_id(None).unwrap();
    let b = space.take_records_id(Some(a.id)).unwrap();
    space.give_records_id(a);
    assert!(space.holds_records_slot(b));
    let c = space.take_records_id(None).unwrap();
    assert_eq!(c.id, 1);
    assert!(space.holds_records_slot(c));
    assert!(!space.holds_records_slot(b));

    // Placed twice at one ID: the second is reached, and once it is gone, neither.
    let mut space = AddressSpace::new();
    let b = space.take_records_id(Some(6)).unwrap();
    let c = space.take_records_id(Some(6)).unwrap();
    assert!(space.holds_records_slot(c));
    space.give_records_id(c);
    assert!(!space.holds_records_slot(b));
    space.give_records_id(b);
    assert!(space.is_empty());
}
