//! The device's live buffers kept as they change (`sorted_chunks.rs`): a change costs one chunk,
//! and a copy made after it shares every other chunk with the copies before it, whatever the
//! length.

use crate::sorted_chunks::{SortedChunks, MOST};
use std::cell::Cell;
use std::collections::BTreeMap;
use std::sync::Arc;

thread_local! {
    /// The values cloned on this thread.
    static CLONES: Cell<usize> = const { Cell::new(0) };
}

/// A value that counts its clones.
#[derive(Debug, PartialEq, Eq)]
struct Counted(u64);

impl Clone for Counted {
    fn clone(&self) -> Counted {
        CLONES.with(|clones| clones.set(clones.get() + 1));
        Counted(self.0)
    }
}

/// What `f` returned, and the values cloned while it ran.
fn clones_in<T>(f: impl FnOnce() -> T) -> (T, usize) {
    let before = CLONES.with(Cell::get);
    let made = f();
    (made, CLONES.with(Cell::get) - before)
}

/// The chunks `a` and `b` share.
fn shared(a: &SortedChunks<u64, Counted>, b: &SortedChunks<u64, Counted>) -> usize {
    a.chunks()
        .iter()
        .filter(|chunk| b.chunks().iter().any(|other| Arc::ptr_eq(chunk, other)))
        .count()
}

/// The entries of `map` in order.
fn entries(map: &SortedChunks<u64, Counted>) -> Vec<u64> {
    map.iter().map(|(key, _)| *key).collect()
}

/// The regression check: in a list of 32768 buffers, a copy clones no entry, and each buffer made
/// or gone after it clones one chunk's entries at most, leaving every other chunk shared with the
/// copy, which keeps the list as it was. Before, each resident set made after a change held, sorted
/// and uploaded every listed buffer again.
#[test]
fn a_change_copies_one_chunk_and_the_copies_share_the_rest() {
    let mut map = SortedChunks::default();
    for key in 0..32768u64 {
        assert!(map.insert(2 * key, Counted(key)));
    }
    let chunks = map.chunks().len();
    assert!(chunks >= 32768 / MOST, "{chunks} chunks");
    let (copy, cloned) = clones_in(|| map.clone());
    assert_eq!(cloned, 0, "a copy clones no entry");
    let before = entries(&copy);
    let (inserted, made) = clones_in(|| map.insert(20001, Counted(7)));
    assert!(inserted);
    assert!(made <= MOST, "a buffer made cloned {made} entries");
    assert!(
        shared(&map, &copy) >= chunks - 1,
        "chunks shared after a buffer made"
    );
    let (removed, gone) = clones_in(|| map.remove(&40000));
    assert_eq!(removed, Some(Counted(20000)));
    assert!(gone <= MOST, "a buffer gone cloned {gone} entries");
    assert!(
        shared(&map, &copy) >= chunks - 2,
        "chunks shared after a buffer gone"
    );
    assert_eq!(entries(&copy), before, "the copy keeps the list as it was");
    let mut after = before.clone();
    after.retain(|&key| key != 40000);
    let at = after.partition_point(|&key| key < 20001);
    after.insert(at, 20001);
    assert_eq!(entries(&map), after);
    // A change where no copy shares the chunk clones nothing.
    drop(copy);
    assert_eq!(clones_in(|| map.insert(20003, Counted(8))), (true, 0));
    assert_eq!(clones_in(|| map.remove(&20001)), (Some(Counted(7)), 0));
}

/// Through inserts and removes in a fixed pseudo-random order, against a `BTreeMap`: the entries
/// stay in key order, each chunk holds at most `MOST`, and the lookups answer as the map's.
#[test]
fn entries_stay_in_key_order_through_changes() {
    let mut map = SortedChunks::default();
    let mut model = BTreeMap::new();
    let mut copies = Vec::new();
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    for step in 0..20000u64 {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        let key = seed % 3000;
        if seed % 5 < 3 {
            assert_eq!(map.insert(key, Counted(step)), !model.contains_key(&key));
            model.entry(key).or_insert(Counted(step));
        } else {
            assert_eq!(map.remove(&key), model.remove(&key));
        }
        if step % 997 == 0 {
            copies.push((map.clone(), model.clone()));
        }
    }
    let want: Vec<(u64, Counted)> = model.iter().map(|(k, v)| (*k, v.clone())).collect();
    let got: Vec<(u64, Counted)> = map.iter().cloned().collect();
    assert_eq!(got, want);
    assert!(map
        .chunks()
        .iter()
        .all(|chunk| !chunk.is_empty() && chunk.len() <= MOST));
    for probe in 0..3100u64 {
        assert_eq!(map.contains(&probe), model.contains_key(&probe));
        let below = map.last_where(|&key| key <= probe).map(|(key, _)| *key);
        assert_eq!(
            below,
            model.range(..=probe).next_back().map(|(key, _)| *key)
        );
    }
    // Each copy kept the list as it was when it was made.
    for (copy, model) in &copies {
        let got: Vec<(u64, Counted)> = copy.iter().cloned().collect();
        let want: Vec<(u64, Counted)> = model.iter().map(|(k, v)| (*k, v.clone())).collect();
        assert_eq!(got, want);
    }
    assert!(!map.is_empty());
    for key in model.keys() {
        assert!(map.remove(key).is_some());
    }
    assert!(map.is_empty() && map.chunks().is_empty());
}
