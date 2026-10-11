//! A command buffer's GPU times (`gpu_time.rs`): timestamps placed on the host's clock by a
//! calibration pair or by the completion, across a wrap of the timestamps' bits; the host time
//! domain a calibration reads; the extension's preferred form; and the device's timestamp pools,
//! their slots held by the command buffers in flight, made as those need them up to the most.

use crate::gpu_time::{host_domain, host_time, Extension, GpuClock, Pair, Pool, Pools, SLOTS};
use ash::vk;
use std::collections::BTreeSet;
use std::ffi::CString;

/// An uncalibrated clock of `valid_bits` bits, `period` nanoseconds a tick.
fn clock(valid_bits: u32, period: f32) -> GpuClock {
    GpuClock::new(valid_bits, period, None).expect("a clock with timestamps")
}

#[test]
fn a_family_without_timestamp_bits_has_no_clock() {
    assert!(GpuClock::new(0, 1.0, None).is_none());
    assert_eq!(clock(36, 1.0).valid_bits, 36);
    // More bits than a timestamp holds are its 64.
    assert_eq!(clock(70, 1.0).valid_bits, 64);
}

#[test]
fn durations_count_ticks_of_the_period_across_a_wrap() {
    let clock = clock(36, 52.083_332);
    assert_eq!(clock.duration(1_000, 1_000), 0);
    assert_eq!(clock.duration(1_000, 1_096), 5_000);
    // From 10 ticks before the 36 bits wrap to 5 after: 15 ticks.
    let before = (1u64 << 36) - 10;
    assert_eq!(clock.duration(before, 5), 781);
    // An end before the start is no duration.
    assert_eq!(clock.duration(1_096, 1_000), 0);
    let full = self::clock(64, 1.0);
    assert_eq!(full.duration(u64::MAX - 1, 3), 5);
}

#[test]
fn a_pair_places_timestamps_on_the_hosts_clock_either_side_of_it() {
    let clock = clock(64, 2.0);
    let pair = Pair {
        ticks: 1_000,
        host: 5_000_000,
    };
    assert_eq!(clock.place(pair, 1_000), 5_000_000);
    assert_eq!(clock.place(pair, 1_500), 5_001_000);
    assert_eq!(clock.place(pair, 500), 4_999_000);
    // Across the bits' wrap, the nearer direction: 8 bits, 6 ticks after 250 is 0 after the wrap.
    let narrow = self::clock(8, 1.0);
    let pair = Pair {
        ticks: 250,
        host: 1_000,
    };
    assert_eq!(narrow.place(pair, 0), 1_006);
    assert_eq!(narrow.place(pair, 244), 994);
}

#[test]
fn a_span_is_placed_by_the_pair_or_ends_at_the_completion() {
    let clock = clock(64, 1.0);
    let pair = Pair {
        ticks: 100,
        host: 10_000,
    };
    assert_eq!(clock.span(Some(pair), 150, 400, 99_999), (10_050, 10_300));
    // Uncalibrated: the end is the completion seen, the start its exact duration before.
    assert_eq!(clock.span(None, 150, 400, 20_000), (19_750, 20_000));
    assert_eq!(clock.span(None, 150, 400, 100), (0, 100));
}

#[cfg(unix)]
#[test]
fn calibration_reads_the_devices_domain_and_a_monotonic_host_one() {
    use vk::TimeDomainKHR as D;
    // `TimeDomainKHR` has no `Debug` without ash's `debug` feature: compared, not printed.
    let chosen = |offered: &[D]| host_domain(offered);
    assert!(chosen(&[D::DEVICE, D::CLOCK_MONOTONIC]) == Some(D::CLOCK_MONOTONIC));
    // The raw clock (the host's own, off macOS) before the slewed one.
    assert!(
        chosen(&[D::CLOCK_MONOTONIC, D::DEVICE, D::CLOCK_MONOTONIC_RAW])
            == Some(D::CLOCK_MONOTONIC_RAW)
    );
    assert!(chosen(&[D::CLOCK_MONOTONIC_RAW]).is_none());
    assert!(chosen(&[D::DEVICE, D::QUERY_PERFORMANCE_COUNTER]).is_none());
    assert!(chosen(&[]).is_none());
}

#[test]
fn the_khr_form_is_preferred_where_both_are_listed() {
    let both: BTreeSet<CString> = [
        vk::EXT_CALIBRATED_TIMESTAMPS_NAME,
        vk::KHR_CALIBRATED_TIMESTAMPS_NAME,
    ]
    .into_iter()
    .map(|name| name.to_owned())
    .collect();
    let preferred = Extension::among(&both).expect("listed");
    assert_eq!(preferred.name(), vk::KHR_CALIBRATED_TIMESTAMPS_NAME);
    let ext: BTreeSet<CString> = [vk::EXT_CALIBRATED_TIMESTAMPS_NAME.to_owned()].into();
    assert_eq!(
        Extension::among(&ext).expect("listed").name(),
        vk::EXT_CALIBRATED_TIMESTAMPS_NAME
    );
    assert!(Extension::among(&BTreeSet::new()).is_none());
}

#[test]
fn the_host_time_rises() {
    let first = host_time();
    let second = host_time();
    assert!(first > 0.0 && second >= first, "{first} then {second}");
}

#[test]
fn an_uncalibrated_clock_takes_no_pair() {
    assert_eq!(clock(64, 1.0).pair(), None);
}

/// The pool hands out each slot once, two queries apart, slot 0 first; none once every slot is
/// held; and a slot dropped is given back, so the slots held are the command buffers in flight,
/// however many were committed.
#[test]
fn a_slot_is_held_until_dropped_and_then_given_back() {
    let pool = Pool::new(vk::QueryPool::null(), SLOTS);
    let held: Vec<_> = (0..SLOTS).map(|_| pool.take().unwrap()).collect();
    let firsts: Vec<u32> = held.iter().map(|slot| slot.first()).collect();
    assert_eq!(
        firsts,
        (0..SLOTS).map(|index| 2 * index).collect::<Vec<_>>()
    );
    assert!(pool.take().is_none());
    let mut held = held;
    let given = held.swap_remove(7);
    let first = given.first();
    drop(given);
    let again = pool.take().unwrap();
    assert_eq!(again.first(), first);
    assert!(pool.take().is_none());
    drop(held);
    drop(again);
    let all: Vec<_> = (0..SLOTS).map(|_| pool.take().unwrap()).collect();
    assert!(pool.take().is_none());
    drop(all);
    // Commits one in flight at a time never run out.
    for _ in 0..3 * SLOTS {
        assert!(pool.take().is_some());
    }
}

/// A device's pools: the first made at the first slot taken, another only when every slot of
/// those made is held, with no most where the device makes every one asked; a freed slot of any
/// pool serves the next commit, with no pool made; the pool that gave the last slot is asked
/// first.
#[test]
fn pools_are_made_as_the_command_buffers_in_flight_need_them() {
    use ash::vk::Handle;
    const MADE: u64 = 5;
    let made = std::cell::Cell::new(0u64);
    let make = || {
        made.set(made.get() + 1);
        Some(vk::QueryPool::from_raw(made.get()))
    };
    let mut pools = Pools::new(None);
    assert_eq!(pools.slots(), 0);
    let mut held = Vec::new();
    for pool in 1..=MADE {
        for _ in 0..SLOTS {
            let slot = pools.take(make).unwrap();
            assert_eq!(slot.queries().as_raw(), pool);
            held.push(slot);
        }
    }
    assert_eq!(pools.slots(), MADE as usize * SLOTS as usize);
    // A slot of the first pool given back serves the next commit; no pool more is made, and
    // past it, with every slot held again, one is.
    let given = held.swap_remove(5);
    let (queries, first) = (given.queries(), given.first());
    drop(given);
    let again = pools.take(make).unwrap();
    assert_eq!((again.queries(), again.first()), (queries, first));
    assert_eq!(made.get(), MADE);
    held.push(again);
    let sixth = pools.take(make).unwrap();
    assert_eq!((made.get(), sixth.queries().as_raw()), (MADE + 1, MADE + 1));
    held.push(sixth);
    // The pool that gave the last slot is asked first: with a slot of the second pool given
    // back, the next commit's slot is still the sixth's, which has slots free.
    drop(held.swap_remove(SLOTS as usize + 3));
    let next = pools.take(make).unwrap();
    assert_eq!(next.queries().as_raw(), MADE + 1);
    assert_eq!(
        pools
            .queries()
            .map(|queries| queries.as_raw())
            .collect::<Vec<_>>(),
        (1..=MADE + 1).collect::<Vec<_>>()
    );
}

/// A pool the device refuses ends the making: the commit gets no slot, and no later commit tries
/// again while every slot of the pools made is held. A device of scarce pools makes one.
#[test]
fn a_refused_pool_ends_the_making_and_scarce_pools_are_one() {
    use ash::vk::Handle;
    let mut pools = Pools::new(None);
    let held: Vec<_> = (0..SLOTS)
        .map(|_| pools.take(|| Some(vk::QueryPool::from_raw(1))).unwrap())
        .collect();
    assert!(pools.take(|| None).is_none());
    let mut asked = false;
    assert!(pools
        .take(|| {
            asked = true;
            Some(vk::QueryPool::from_raw(2))
        })
        .is_none());
    assert!(!asked);
    assert_eq!(pools.slots(), SLOTS as usize);
    drop(held);
    assert!(pools.take(|| None).is_some());

    let mut scarce = Pools::new(Some(1));
    let held: Vec<_> = (0..SLOTS)
        .map(|_| scarce.take(|| Some(vk::QueryPool::from_raw(1))).unwrap())
        .collect();
    let mut asked = false;
    assert!(scarce
        .take(|| {
            asked = true;
            Some(vk::QueryPool::from_raw(2))
        })
        .is_none());
    assert!(!asked);
    assert_eq!(scarce.slots(), SLOTS as usize);
    drop(held);
}
