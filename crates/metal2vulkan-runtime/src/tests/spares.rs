//! A device's spare command pools and descriptor pools (`spares.rs`): which command buffers a pool
//! hands out again, and what the bounds keep.

use crate::descriptor_arena::{Room, SetNeeds};
use crate::spares::{
    keep_command_pool, keep_descriptor_pool, take_command_pool, take_descriptor_pool,
    take_to_begin, CommandPool, DescriptorPool, DESCRIPTORS, DESCRIPTOR_POOLS,
};
use ash::vk::{self, Handle};
use std::collections::VecDeque;

/// Needs of `count` storage buffers, with `flags`.
fn storage(count: u32, flags: vk::DescriptorPoolCreateFlags) -> SetNeeds {
    SetNeeds {
        counts: vec![(vk::DescriptorType::STORAGE_BUFFER.as_raw(), count)],
        flags,
    }
}

/// Descriptor pool `n` of `sets` sets of `needs`.
fn descriptor_pool(n: u64, needs: &SetNeeds, sets: u32) -> DescriptorPool {
    DescriptorPool {
        raw: vk::DescriptorPool::from_raw(n),
        room: Room::of(needs, sets),
    }
}

/// A spare command pool that a command buffer took hands out its command buffers again, in the
/// order they were made, each level apart, before any new one is made.
#[test]
fn a_taken_pool_hands_out_its_command_buffers_again() {
    let primary = vk::CommandBufferLevel::PRIMARY;
    let secondary = vk::CommandBufferLevel::SECONDARY;
    let mut pool = CommandPool::new(vk::CommandPool::from_raw(1), 0);
    assert_eq!(pool.reuse(primary), None);
    pool.made(primary, vk::CommandBuffer::from_raw(10));
    pool.made(secondary, vk::CommandBuffer::from_raw(20));
    pool.made(primary, vk::CommandBuffer::from_raw(11));
    assert_eq!(
        pool.reuse(primary),
        None,
        "each made one is handed out already"
    );
    let mut pools = Vec::new();
    keep_command_pool(&mut pools, pool);
    let mut pool = pools.pop().expect("kept");
    pool.taken();
    assert_eq!(pool.reuse(primary), Some(vk::CommandBuffer::from_raw(10)));
    assert_eq!(pool.reuse(secondary), Some(vk::CommandBuffer::from_raw(20)));
    assert_eq!(pool.reuse(primary), Some(vk::CommandBuffer::from_raw(11)));
    assert_eq!(pool.reuse(primary), None);
    assert_eq!(pool.reuse(secondary), None);
}

/// A spare command pool that made no secondary command buffer wants one; stocked with one, the
/// next command buffer that takes it hands that one out before making any; one that made a
/// secondary wants none.
#[test]
fn a_pool_without_a_secondary_is_stocked_with_one_for_its_next_taker() {
    let primary = vk::CommandBufferLevel::PRIMARY;
    let secondary = vk::CommandBufferLevel::SECONDARY;
    let mut pool = CommandPool::new(vk::CommandPool::from_raw(1), 0);
    pool.made(primary, vk::CommandBuffer::from_raw(10));
    assert!(pool.wants_secondary(), "only a primary made");
    pool.stock_secondary(vk::CommandBuffer::from_raw(20));
    assert!(!pool.wants_secondary(), "stocked");
    let mut pools = Vec::new();
    keep_command_pool(&mut pools, pool);
    let mut pool = pools.pop().expect("kept");
    pool.taken();
    assert_eq!(pool.reuse(secondary), Some(vk::CommandBuffer::from_raw(20)));
    assert_eq!(pool.reuse(secondary), None);
    let mut made = CommandPool::new(vk::CommandPool::from_raw(2), 0);
    made.made(secondary, vk::CommandBuffer::from_raw(30));
    assert!(!made.wants_secondary(), "one made already");
}

/// A spare command pool's primary begun ahead goes to the first encoder of the command buffer
/// that takes the pool, handed out among the pool's command buffers (so the pool settles only
/// once it ended too), never to `reuse`, which hands out the others; a pool its taker resets (one
/// given back recording) holds it as an ordinary primary again.
#[test]
fn a_primary_begun_ahead_goes_to_the_first_encoder_alone() {
    let primary = vk::CommandBufferLevel::PRIMARY;
    let mut pool = CommandPool::new(vk::CommandPool::from_raw(1), 0);
    pool.made(primary, vk::CommandBuffer::from_raw(10));
    pool.begun = Some(vk::CommandBuffer::from_raw(11));
    pool.taken();
    assert_eq!(pool.reuse(primary), Some(vk::CommandBuffer::from_raw(10)));
    assert_eq!(pool.reuse(primary), None, "the begun one is not reused");
    assert_eq!(pool.take_begun(), Some(vk::CommandBuffer::from_raw(11)));
    assert_eq!(pool.take_begun(), None, "taken once");
    assert_eq!(pool.handed_out(), 2, "the begun one counts as handed out");
    pool.taken();
    assert_eq!(pool.reuse(primary), Some(vk::CommandBuffer::from_raw(10)));
    assert_eq!(
        pool.reuse(primary),
        Some(vk::CommandBuffer::from_raw(11)),
        "taken, it is an ordinary primary of the pool"
    );
    let mut reset = CommandPool::new(vk::CommandPool::from_raw(2), 0);
    reset.begun = Some(vk::CommandBuffer::from_raw(20));
    reset.recording = true;
    reset.taken();
    assert_eq!(
        reset.take_begun(),
        None,
        "the reset left it in the initial state"
    );
    assert_eq!(reset.reuse(primary), Some(vk::CommandBuffer::from_raw(20)));
}

/// A command buffer takes the newest spare command pool of its queue's family that holds a
/// primary begun ahead, else the newest of that family.
#[test]
fn a_spare_command_pool_holding_a_begun_primary_is_taken_first() {
    let mut pools = Vec::new();
    for (raw, begun) in [(1, false), (2, true), (3, false)] {
        let mut pool = CommandPool::new(vk::CommandPool::from_raw(raw), 0);
        if begun {
            pool.begun = Some(vk::CommandBuffer::from_raw(raw * 10));
        }
        keep_command_pool(&mut pools, pool);
    }
    let raw = |pool: Option<CommandPool>| pool.map(|pool| pool.raw);
    for expected in [2, 3, 1] {
        assert_eq!(
            raw(take_command_pool(&mut pools, 0)),
            Some(vk::CommandPool::from_raw(expected))
        );
    }
}

/// A command buffer takes the newest spare command pool made for its queue's family, never one
/// of another family.
#[test]
fn a_spare_command_pool_is_taken_only_for_its_family() {
    let mut pools = Vec::new();
    for (raw, family) in [(1, 0), (2, 1), (3, 0), (4, 2)] {
        keep_command_pool(
            &mut pools,
            CommandPool::new(vk::CommandPool::from_raw(raw), family),
        );
    }
    let raw = |pool: Option<CommandPool>| pool.map(|pool| pool.raw);
    assert_eq!(
        raw(take_command_pool(&mut pools, 0)),
        Some(vk::CommandPool::from_raw(3))
    );
    assert_eq!(
        raw(take_command_pool(&mut pools, 0)),
        Some(vk::CommandPool::from_raw(1))
    );
    assert_eq!(raw(take_command_pool(&mut pools, 0)), None);
    assert_eq!(raw(take_command_pool(&mut pools, 3)), None);
    assert_eq!(
        raw(take_command_pool(&mut pools, 2)),
        Some(vk::CommandPool::from_raw(4))
    );
    assert_eq!(pools.len(), 1);
}

/// Every command pool given back is kept, past the former bounds too (8 spares, and a pool that
/// made more than 64 command buffers, each released on the completing handler thread before),
/// and each is taken again, newest first: the runtime never releases one while the device lives.
#[test]
fn every_command_pool_given_back_is_kept() {
    let mut pools = Vec::new();
    let mut large = CommandPool::new(vk::CommandPool::from_raw(1000), 0);
    for n in 0..200u64 {
        large.made(
            vk::CommandBufferLevel::PRIMARY,
            vk::CommandBuffer::from_raw(n + 1),
        );
    }
    keep_command_pool(&mut pools, large);
    for n in 0..100u64 {
        keep_command_pool(
            &mut pools,
            CommandPool::new(vk::CommandPool::from_raw(n + 1), 0),
        );
    }
    assert_eq!(pools.len(), 101);
    for n in (0..100u64).rev() {
        let pool = take_command_pool(&mut pools, 0).expect("kept");
        assert_eq!(pool.raw, vk::CommandPool::from_raw(n + 1));
    }
    let mut large = take_command_pool(&mut pools, 0).expect("the large pool kept");
    assert_eq!(large.raw, vk::CommandPool::from_raw(1000));
    large.taken();
    assert_eq!(
        large.reuse(vk::CommandBufferLevel::PRIMARY),
        Some(vk::CommandBuffer::from_raw(1)),
        "its command buffers handed out again"
    );
    assert!(pools.is_empty());
}

/// A set takes the newest spare descriptor pool with room for it, of its flags; past the bounds
/// the oldest spares go first, and a pool that alone holds more descriptors than the bound is not
/// kept.
#[test]
fn spare_descriptor_pools_are_bounded_and_taken_by_room() {
    let plain = vk::DescriptorPoolCreateFlags::empty();
    let after_bind = vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND;
    let mut pools = VecDeque::new();
    assert!(
        keep_descriptor_pool(&mut pools, descriptor_pool(1, &storage(2, plain), 16)).is_empty()
    );
    assert!(keep_descriptor_pool(&mut pools, descriptor_pool(2, &storage(1, plain), 1)).is_empty());
    assert!(
        keep_descriptor_pool(&mut pools, descriptor_pool(3, &storage(2, after_bind), 16))
            .is_empty()
    );
    // Two storage buffers: pool 2 holds one in all (a pool's room is its totals), and pool 3 has
    // other flags.
    let taken = take_descriptor_pool(&mut pools, &storage(2, plain)).expect("pool 1 fits");
    assert_eq!(taken.raw, vk::DescriptorPool::from_raw(1));
    assert!(take_descriptor_pool(&mut pools, &storage(4, plain)).is_none());
    assert_eq!(pools.len(), 2);

    let mut pools = VecDeque::new();
    for n in 0..DESCRIPTOR_POOLS as u64 {
        assert!(
            keep_descriptor_pool(&mut pools, descriptor_pool(n + 1, &storage(1, plain), 16))
                .is_empty()
        );
    }
    let past = keep_descriptor_pool(&mut pools, descriptor_pool(99, &storage(1, plain), 16));
    assert_eq!(past, [vk::DescriptorPool::from_raw(1)]);
    assert_eq!(pools.len(), DESCRIPTOR_POOLS);

    // By descriptors: three pools of a third of the bound each, then a fourth.
    let third = (DESCRIPTORS / 3) as u32;
    let mut pools = VecDeque::new();
    for n in 0..3 {
        assert!(keep_descriptor_pool(
            &mut pools,
            descriptor_pool(n + 1, &storage(third, plain), 1)
        )
        .is_empty());
    }
    let past = keep_descriptor_pool(&mut pools, descriptor_pool(4, &storage(third, plain), 1));
    assert_eq!(past, [vk::DescriptorPool::from_raw(1)]);
    let whole = (DESCRIPTORS + 1) as u32;
    let past = keep_descriptor_pool(&mut pools, descriptor_pool(5, &storage(whole, plain), 1));
    assert_eq!(past, [vk::DescriptorPool::from_raw(5)]);
    assert_eq!(pools.len(), 3);
}

/// A spare command pool a completing thread begins a primary ahead in is out of the spares while
/// it begins, so a taker meanwhile takes another, and never waits for the begin; a pool that may
/// not begin one (recording, or holding one begun) stays; kept again, it is the newest.
#[test]
fn a_spare_command_pool_begins_ahead_out_of_the_spares() {
    let pool = |n: u64| CommandPool::new(vk::CommandPool::from_raw(n), 0);
    let mut pools = vec![pool(1), pool(2)];
    let taken = take_to_begin(&mut pools, vk::CommandPool::from_raw(2)).unwrap();
    assert_eq!(taken.raw, vk::CommandPool::from_raw(2));
    let other = take_command_pool(&mut pools, 0).unwrap();
    assert_eq!(
        other.raw,
        vk::CommandPool::from_raw(1),
        "a taker takes another"
    );
    assert!(
        take_command_pool(&mut pools, 0).is_none(),
        "and finds none being begun"
    );
    keep_command_pool(&mut pools, other);
    keep_command_pool(&mut pools, taken);
    let raws: Vec<_> = pools.iter().map(|pool| pool.raw.as_raw()).collect();
    assert_eq!(raws, [1, 2], "kept again, the newest");
    pools[1].begun = Some(vk::CommandBuffer::from_raw(20));
    assert!(take_to_begin(&mut pools, vk::CommandPool::from_raw(2)).is_none());
    pools[0].recording = true;
    assert!(take_to_begin(&mut pools, vk::CommandPool::from_raw(1)).is_none());
    assert!(take_to_begin(&mut pools, vk::CommandPool::from_raw(3)).is_none());
    assert_eq!(pools.len(), 2, "a pool that may not begin one stays");
}
