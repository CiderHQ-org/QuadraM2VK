//! No-copy buffers over the same host memory (`host_aliases.rs`).

use super::super::host_aliases::{Alias, HostAliases, Memories};
use super::super::memory::MemoryClass;
use super::super::resource::ResourceKey;
use super::super::tracker::{Access, Use, UseRange};
use ash::vk;
use std::sync::Arc;

/// The registry, its imports stood in for by numbers.
type Aliases = HostAliases<u32>;

/// Note `key` over `memory`, lying in a new import of `Shared` memory, which is dropped at once.
fn join(aliases: &Aliases, key: ResourceKey, memory: (u64, u64)) -> Option<Alias> {
    aliases.join(key, memory, MemoryClass::Shared, &Arc::new(0))
}

fn copy() -> Access {
    Access::new(
        vk::PipelineStageFlags2::COPY,
        vk::AccessFlags2::TRANSFER_WRITE,
    )
}

/// A write of `bytes` of the buffer `key`.
fn write(key: ResourceKey, bytes: std::ops::Range<u64>) -> Use<ResourceKey> {
    Use::writes(key, true, UseRange::Bytes(bytes), copy())
}

/// A buffer made over exactly the bytes a live one covers joins it, and keeps the key it joined
/// while the first lives or not; one over other bytes, overlapping or inside, does not.
#[test]
fn a_buffer_over_the_bytes_another_covers_joins_it() {
    let aliases = Aliases::default();
    let (first, second, third) = (ResourceKey::new(), ResourceKey::new(), ResourceKey::new());
    assert_eq!(join(&aliases, first, (0x10000, 0x8000)), None);
    assert_eq!(aliases.of(first), None);
    // Inside, overlapping, and of another length from the same byte: none joins.
    for (start, length) in [(0x11000, 0x1000), (0x14000, 0x8000), (0x10000, 0x4000)] {
        let other = ResourceKey::new();
        assert_eq!(
            join(&aliases, other, (start, length)),
            None,
            "{start:#x}+{length:#x}"
        );
        aliases.forget(other);
    }
    assert_eq!(
        join(&aliases, second, (0x10000, 0x8000)),
        Some(Alias { key: first })
    );
    assert_eq!(aliases.of(second), Some(Alias { key: first }));
    // The first gone, a third over the memory joins its key still.
    aliases.forget(first);
    assert_eq!(
        join(&aliases, third, (0x10000, 0x8000)),
        Some(Alias { key: first })
    );
    // The second rebound: it keeps its key, and the memory is the third's to share.
    aliases.leave(second);
    assert_eq!(aliases.of(second), Some(Alias { key: first }));
    aliases.forget(second);
    aliases.forget(third);
    // Every buffer over the memory gone: the next one over it has its own key.
    let fourth = ResourceKey::new();
    assert_eq!(join(&aliases, fourth, (0x10000, 0x8000)), None);
    assert_eq!(aliases.augment(&[write(fourth, 0..16)]), None);
}

/// A command's uses of a buffer that joined carry a use of the memory's key, of the same bytes,
/// access and tracking; others are as they are, and with no buffer joined nothing is added.
#[test]
fn a_joined_buffers_uses_carry_its_memorys_key() {
    let aliases = Aliases::default();
    let (first, second, apart) = (ResourceKey::new(), ResourceKey::new(), ResourceKey::new());
    join(&aliases, first, (0x40000, 0x1000));
    join(&aliases, apart, (0x80000, 0x1000));
    let uses = [write(first, 0..64), write(apart, 0..64)];
    assert_eq!(aliases.augment(&uses), None);
    join(&aliases, second, (0x40000, 0x1000));
    assert_eq!(aliases.augment(&uses), None);
    let untracked = Use::reads(second, false, UseRange::Bytes(128..256), copy());
    let uses = [write(second, 0..64), untracked.clone(), write(apart, 0..64)];
    let augmented = aliases.augment(&uses).unwrap();
    let mut want = uses.to_vec();
    want.push(write(first, 0..64));
    want.push(Use {
        key: first,
        ..untracked
    });
    assert_eq!(augmented, want);
    aliases.forget(second);
    assert_eq!(aliases.augment(&uses), None);
}

/// A buffer over the bytes a live buffer was made over lies in its import while it lives, of its
/// memory class only; the first import noted stays the memory's while it lives, and once every
/// buffer over the memory is gone none is shared.
#[test]
fn a_buffer_over_the_bytes_another_covers_lies_in_its_import() {
    let aliases = Aliases::default();
    let memory = (0x20000, 0x4000);
    let (first, second, managed) = (ResourceKey::new(), ResourceKey::new(), ResourceKey::new());
    assert!(aliases.import(memory, MemoryClass::Shared).is_none());
    let import = Arc::new(1);
    aliases.join(first, memory, MemoryClass::Shared, &import);
    let shared = aliases.import(memory, MemoryClass::Shared).unwrap();
    assert!(Arc::ptr_eq(&shared, &import));
    // Other bytes, or another class, share nothing.
    assert!(aliases
        .import((0x20000, 0x2000), MemoryClass::Shared)
        .is_none());
    assert!(aliases.import(memory, MemoryClass::Managed).is_none());
    // A second over the memory lies in it, and the first import stays the memory's.
    let again = Arc::new(2);
    aliases.join(second, memory, MemoryClass::Shared, &again);
    assert!(Arc::ptr_eq(
        &aliases.import(memory, MemoryClass::Shared).unwrap(),
        &import
    ));
    // A managed buffer's import is the memory's for its class.
    let own = Arc::new(3);
    aliases.join(managed, memory, MemoryClass::Managed, &own);
    assert!(Arc::ptr_eq(
        &aliases.import(memory, MemoryClass::Managed).unwrap(),
        &own
    ));
    // The first import gone (its buffers rebound), the next noted is shared.
    drop((import, shared));
    assert!(aliases.import(memory, MemoryClass::Shared).is_none());
    let third = ResourceKey::new();
    aliases.join(third, memory, MemoryClass::Shared, &again);
    assert!(Arc::ptr_eq(
        &aliases.import(memory, MemoryClass::Shared).unwrap(),
        &again
    ));
    // Every buffer over the memory gone: its imports are not shared, though they live.
    for key in [first, second, managed, third] {
        aliases.forget(key);
    }
    assert!(aliases.import(memory, MemoryClass::Shared).is_none());
    assert!(aliases.import(memory, MemoryClass::Managed).is_none());
    drop((again, own));
}

/// After the caller reports a remap of any byte of a memory, a later buffer over it lies in no
/// import made before and joins no earlier key; the buffers made before keep the key they
/// joined, and forgetting them later leaves the new buffer's memory alone. A remap of other
/// bytes changes nothing.
#[test]
fn a_reported_remap_ends_the_sharing_of_imports_made_before() {
    let aliases = Aliases::default();
    let memory = (0x40000, 0x8000);
    let (first, second, after) = (ResourceKey::new(), ResourceKey::new(), ResourceKey::new());
    let import = Arc::new(1);
    assert_eq!(
        aliases.join(first, memory, MemoryClass::Shared, &import),
        None
    );
    assert_eq!(
        aliases.join(second, memory, MemoryClass::Shared, &import),
        Some(Alias { key: first })
    );
    // Bytes next to the memory, on either side: nothing changes.
    aliases.remapped(0x30000, 0x40000);
    aliases.remapped(0x48000, 0x50000);
    assert!(aliases.import(memory, MemoryClass::Shared).is_some());
    // One page inside the memory: its import is shared no more.
    aliases.remapped(0x44000, 0x45000);
    assert!(aliases.import(memory, MemoryClass::Shared).is_none());
    // A buffer made after imports anew and joins nothing; the second keeps the key it joined.
    let fresh = Arc::new(2);
    assert_eq!(
        aliases.join(after, memory, MemoryClass::Shared, &fresh),
        None
    );
    assert_eq!(aliases.of(second), Some(Alias { key: first }));
    assert!(Arc::ptr_eq(
        &aliases.import(memory, MemoryClass::Shared).unwrap(),
        &fresh
    ));
    // The buffers made before go: the memory stays the new buffer's.
    aliases.forget(first);
    aliases.forget(second);
    assert!(Arc::ptr_eq(
        &aliases.import(memory, MemoryClass::Shared).unwrap(),
        &fresh
    ));
    aliases.forget(after);
    assert!(aliases.import(memory, MemoryClass::Shared).is_none());
    drop((import, fresh));
}

/// Note a value over `memory`.
fn note(memories: &mut Memories<()>, memory: (u64, u64)) {
    memories.get_or_insert_with(memory, || ());
}

/// How many memories the lookup over `start..end` finds, and how many it reads.
fn looked_up(memories: &Memories<()>, start: u64, end: u64) -> (usize, usize) {
    (
        memories.overlapping(start, end).count(),
        memories.candidates(start, end).count(),
    )
}

/// A remap's lookup by address reads the memories it finds and, of each class of length, at
/// most those over one address before the range, never all of them: among 51,065 memories, a
/// free page between noted ones and a free stretch of the address space read none; a whole
/// reservation reads just the 50,000 it covers; each page report of a lazily mapped 32-MiB
/// buffer reads that buffer's memory alone; and a page just past a three-page memory reads that
/// one (its class's longest is four pages), not its neighbours.
#[test]
fn a_remap_s_lookup_reads_only_the_memories_near_its_range() {
    const PAGE: u64 = 0x4000;
    const GIB: u64 = 1 << 30;
    const MIB: u64 = 1 << 20;
    let (pages, lazy, reserved, threes) = (16 * GIB, 64 * GIB, 128 * GIB, 112 * GIB);
    let mut memories = Memories::<()>::default();
    for page in 0..50_000 {
        note(&mut memories, (pages + 2 * page * PAGE, PAGE));
    }
    for buffer in 0..64 {
        note(&mut memories, (lazy + buffer * 64 * MIB, 32 * MIB));
    }
    note(&mut memories, (reserved, 4 * GIB));
    for three in 0..1000 {
        note(&mut memories, (threes + 4 * three * PAGE, 3 * PAGE));
    }
    assert_eq!(memories.len(), 51_065);
    // A free page between two noted ones, and a free stretch of 4 GiB.
    assert_eq!(looked_up(&memories, pages + PAGE, pages + 2 * PAGE), (0, 0));
    assert_eq!(looked_up(&memories, 200 * GIB, 204 * GIB), (0, 0));
    // A reservation over every one-page memory, and the whole address space.
    let end = pages + 100_000 * PAGE;
    assert_eq!(looked_up(&memories, pages - GIB, end), (50_000, 50_000));
    assert_eq!(looked_up(&memories, 0, u64::MAX), (51_065, 51_065));
    // Each page of the sixth lazily mapped buffer, and a page in the gap after it.
    let sixth = lazy + 5 * 64 * MIB;
    for page in 0..(32 * MIB / PAGE) {
        let at = sixth + page * PAGE;
        assert_eq!(looked_up(&memories, at, at + PAGE), (1, 1), "page {page}");
    }
    let gap = sixth + 40 * MIB;
    assert_eq!(looked_up(&memories, gap, gap + PAGE), (0, 0));
    // The page past each three-page memory.
    for three in 0..1000 {
        let past = threes + (4 * three + 3) * PAGE;
        assert_eq!(
            looked_up(&memories, past, past + PAGE),
            (0, 1),
            "three {three}"
        );
    }
}

/// The lookup by address finds exactly the memories with a byte in the range, among memories
/// of many lengths that overlap each other, and the index keeps its values and count through
/// removals.
#[test]
fn a_remap_s_lookup_finds_exactly_the_memories_with_a_byte_in_its_range() {
    const PAGE: u64 = 0x1000;
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut next = move |below: u64| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state % below
    };
    let mut memories = Memories::<()>::default();
    let mut noted = Vec::new();
    for _ in 0..2000 {
        let class = next(13);
        let length = (1 + next(1 << class)) * PAGE;
        let memory = (next(1 << 20) * PAGE, length);
        note(&mut memories, memory);
        noted.push(memory);
    }
    noted.sort_unstable();
    noted.dedup();
    assert_eq!(memories.len(), noted.len());
    for round in 0..2000 {
        if round % 4 == 0 {
            let gone = noted.swap_remove(next(noted.len() as u64) as usize);
            assert!(memories.remove(gone).is_some());
            assert!(memories.remove(gone).is_none());
            assert_eq!(memories.len(), noted.len());
        }
        let start = next(1 << 21) * PAGE;
        let class = next(16);
        let end = start + (1 + next(1 << class)) * PAGE;
        let mut found: Vec<(u64, u64)> = memories.overlapping(start, end).collect();
        found.sort_unstable();
        let mut expected: Vec<(u64, u64)> = noted
            .iter()
            .copied()
            .filter(|&(at, length)| at < end && start < at + length)
            .collect();
        expected.sort_unstable();
        assert_eq!(found, expected, "{start:#x}..{end:#x}");
    }
}

/// Remaps reported from two threads while four others note and forget buffers over eight
/// memories (the second of each pair lying in the first's import, or its own where a remap
/// ended the memory between them) leave the notes whole: once every buffer is gone, a buffer
/// over each memory joins none, as no memory, or an earlier key, lingered.
#[test]
fn remaps_from_any_thread_keep_the_notes_whole() {
    const BASE: u64 = 0x1_0000_0000;
    const PAGE: u64 = 0x4000;
    let aliases = Aliases::default();
    let memory = |round: u64| (BASE + round % 8 * PAGE, PAGE);
    let done = std::sync::atomic::AtomicBool::new(false);
    std::thread::scope(|scope| {
        let workers: Vec<_> = (0..4)
            .map(|worker| {
                let aliases = &aliases;
                scope.spawn(move || {
                    for round in 0..4000 {
                        let memory = memory(round + worker);
                        let import = Arc::new(worker as u32);
                        let (first, second) = (ResourceKey::new(), ResourceKey::new());
                        aliases.join(first, memory, MemoryClass::Shared, &import);
                        let shared = aliases
                            .import(memory, MemoryClass::Shared)
                            .unwrap_or_else(|| Arc::clone(&import));
                        aliases.join(second, memory, MemoryClass::Shared, &shared);
                        aliases.forget(first);
                        aliases.forget(second);
                        assert_eq!(aliases.of(second), None);
                    }
                })
            })
            .collect();
        for _ in 0..2 {
            let (aliases, done) = (&aliases, &done);
            scope.spawn(move || {
                let mut round = 0;
                while !done.load(std::sync::atomic::Ordering::SeqCst) {
                    let (start, length) = memory(round);
                    aliases.remapped(start + PAGE / 2, start + length);
                    aliases.remapped(0, u64::MAX);
                    round += 1;
                }
            });
        }
        for worker in workers {
            worker.join().unwrap();
        }
        done.store(true, std::sync::atomic::Ordering::SeqCst);
    });
    for round in 0..8 {
        let key = ResourceKey::new();
        assert_eq!(join(&aliases, key, memory(round)), None);
        assert!(aliases.import(memory(round), MemoryClass::Shared).is_none());
        aliases.forget(key);
    }
}

/// A read of `bytes` of the buffer `key` by a fragment shader.
fn sample(key: ResourceKey, bytes: std::ops::Range<u64>) -> Use<ResourceKey> {
    let access = Access::new(
        vk::PipelineStageFlags2::FRAGMENT_SHADER,
        vk::AccessFlags2::SHADER_SAMPLED_READ,
    );
    Use::reads(key, true, UseRange::Bytes(bytes), access)
}

/// Keys over one range of host bytes that joined nothing (one rebound there, or made after a
/// remap of them) share the first placed there while any lies there: a use of another's bytes
/// carries a use of its key; the first's uses take none, the others carrying it. Once the first
/// went, the others carry its key still; once all went, the next over the bytes is on its own.
#[test]
fn keys_over_one_range_share_the_first_placed_there() {
    let aliases = Aliases::default();
    let (first, second, third) = (ResourceKey::new(), ResourceKey::new(), ResourceKey::new());
    let surface = (0x7f00_0000_0000, 0x7e_9000);
    aliases.place(first, surface);
    assert_eq!(aliases.augment(&[write(first, 0..0x7e_9000)]), None);
    aliases.place(second, surface);
    assert_eq!(aliases.of(second), None, "joined nothing");
    let uses = [sample(second, 0..0x7e_9000)];
    assert_eq!(
        aliases.augment(&uses),
        Some(vec![uses[0].clone(), sample(first, 0..0x7e_9000)])
    );
    assert_eq!(aliases.augment(&[write(first, 0..0x7e_9000)]), None);
    aliases.forget(first);
    assert_eq!(
        aliases.augment(&uses),
        Some(vec![uses[0].clone(), sample(first, 0..0x7e_9000)])
    );
    aliases.forget(second);
    aliases.place(third, surface);
    assert_eq!(aliases.augment(&[sample(third, 0..64)]), None);
}

/// A use of a key's bytes carries a use of the key over each other range they meet, of the bytes
/// they meet counted from that range's first, with the use's access and tracking; bytes that
/// meet none take nothing, and a rebind moves the key to its new range's company.
#[test]
fn a_use_takes_the_keys_of_the_ranges_its_bytes_meet_clipped() {
    let aliases = Aliases::default();
    let (a, b, c) = (ResourceKey::new(), ResourceKey::new(), ResourceKey::new());
    aliases.place(a, (0x10000, 0x8000));
    aliases.place(b, (0x14000, 0x8000));
    aliases.place(c, (0x40000, 0x1000));
    // a's bytes 0x5000..0x7000 lie at 0x15000..0x17000: b's 0x1000..0x3000.
    let uses = [write(a, 0x5000..0x7000), write(c, 0..64)];
    assert_eq!(
        aliases.augment(&uses),
        Some(vec![
            uses[0].clone(),
            uses[1].clone(),
            write(b, 0x1000..0x3000)
        ])
    );
    // b's untracked read of its first page meets a's last.
    let untracked = Use::reads(b, false, UseRange::Bytes(0..0x1000), copy());
    assert_eq!(
        aliases.augment(std::slice::from_ref(&untracked)),
        Some(vec![
            untracked.clone(),
            Use {
                key: a,
                range: UseRange::Bytes(0x4000..0x5000),
                ..untracked.clone()
            }
        ])
    );
    assert_eq!(aliases.augment(&[write(a, 0..0x1000)]), None);
    // b rebound beside c: a is on its own, b and c meet.
    aliases.place(b, (0x40800, 0x1000));
    assert_eq!(aliases.augment(&[write(a, 0x5000..0x7000)]), None);
    assert_eq!(
        aliases.augment(&[write(c, 0..0x1000)]),
        Some(vec![write(c, 0..0x1000), write(b, 0..0x800)])
    );
    aliases.unplace(b);
    assert_eq!(aliases.augment(&[write(c, 0..0x1000)]), None);
}

/// The consumer's two keys over one surface's pages, neither joined: a render's write through
/// the first on one queue, then a sample through the second on another, is a wait for the
/// render's batch once the second's uses carry the first's key; as they were, nothing ordered
/// them.
#[test]
fn a_sample_through_a_second_key_waits_for_a_render_through_the_first() {
    use super::super::tracker::{EncoderUses, Point, Tracker, Wait};
    let aliases = Aliases::default();
    let (rendered, sampled) = (ResourceKey::new(), ResourceKey::new());
    let surface = (0x7f00_0000_0000, 0x7e_9000);
    aliases.place(rendered, surface);
    aliases.place(sampled, surface);
    let summary = |uses: &[Use<ResourceKey>]| {
        let mut encoder = EncoderUses::new(true);
        let uses = aliases.augment(uses).unwrap_or_else(|| uses.to_vec());
        let barrier = encoder.barrier(&uses);
        encoder.record(vk::PipelineStageFlags2::NONE, &uses, barrier);
        encoder.finish()
    };
    let render = Use::writes(
        rendered,
        true,
        UseRange::Bytes(0..0x7e_9000),
        Access::new(
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
        ),
    );
    let mut tracker = Tracker::default();
    let first = Point { queue: 0, value: 1 };
    let plan = tracker.plan(&[summary(&[render])], first);
    tracker.apply(plan);
    let plan = tracker.plan(
        &[summary(&[sample(sampled, 0..0x7e_9000)])],
        Point { queue: 1, value: 1 },
    );
    assert_eq!(
        plan.waits(),
        [Wait {
            at: first,
            stages: vk::PipelineStageFlags2::FRAGMENT_SHADER
        }]
    );
}

/// The version counts each change of what `augment` reads: a buffer joining another's memory,
/// a key placed, rebound or unplaced, a buffer forgotten. A buffer over a memory of its own, a
/// rebound buffer leaving its memory, a reported remap, and `augment` itself change nothing it
/// reads, and leave it standing.
#[test]
fn the_version_counts_each_change_augment_reads() {
    let aliases = Aliases::default();
    let (first, second, placed) = (ResourceKey::new(), ResourceKey::new(), ResourceKey::new());
    let version = aliases.version();
    join(&aliases, first, (0x40000, 0x1000));
    assert_eq!(aliases.version(), version, "a memory of its own");
    join(&aliases, second, (0x40000, 0x1000));
    let joined = aliases.version();
    assert!(joined > version, "joined");
    aliases.augment(&[write(second, 0..64)]);
    aliases.leave(second);
    assert_eq!(aliases.version(), joined, "augment, leave and a remap");
    let mut last = joined;
    let mut changed = |what: &str| {
        let now = aliases.version();
        assert!(now > last, "{what}");
        last = now;
    };
    aliases.place(placed, (0x80000, 0x2000));
    changed("placed");
    aliases.place(placed, (0x90000, 0x2000));
    changed("rebound");
    aliases.unplace(placed);
    changed("unplaced");
    aliases.forget(second);
    changed("forgotten");
}
