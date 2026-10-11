//! A sorted map whose copies share what they have in common: the device's live buffers
//! (`live_buffers.rs`), of which each resident set the device makes keeps a copy for the command
//! buffers that hold it.
//!
//! The entries lie in key order in chunks of at most [`MOST`] entries, each behind an `Arc`. A
//! copy ([`Clone`]) clones the chunks' `Arc`s, not the entries; an insertion or a removal changes
//! one chunk, which it copies first only where another copy still shares it (`Arc::make_mut`),
//! and splits a chunk past [`MOST`] in two and drops one left empty. So a change costs one
//! chunk's entries at most, whatever the length, and a copy made after it shares every other
//! chunk with the copies made before it.

use std::sync::Arc;

/// The most entries a chunk holds: one that grows past it splits in two.
pub(crate) const MOST: usize = 64;

/// A map from `K` to `V`, sorted by `K`, in shared chunks (the module docs).
pub(crate) struct SortedChunks<K, V> {
    chunks: Vec<Arc<Vec<(K, V)>>>,
    len: usize,
}

impl<K, V> Default for SortedChunks<K, V> {
    fn default() -> SortedChunks<K, V> {
        SortedChunks {
            chunks: Vec::new(),
            len: 0,
        }
    }
}

impl<K, V> Clone for SortedChunks<K, V> {
    /// A copy sharing every chunk.
    fn clone(&self) -> SortedChunks<K, V> {
        SortedChunks {
            chunks: self.chunks.clone(),
            len: self.len,
        }
    }
}

impl<K: Ord + Copy, V: Clone> SortedChunks<K, V> {
    /// Whether it has no entry.
    pub(crate) fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Its entries' count.
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// The chunk `key` belongs in: the last whose first key is at or below it, else the first.
    fn chunk_of(&self, key: &K) -> usize {
        self.chunk_where(|first| first <= key)
    }

    /// The last chunk whose first key `below` holds for, else the first; `below` holds for a
    /// prefix of the keys.
    fn chunk_where(&self, below: impl Fn(&K) -> bool) -> usize {
        self.chunks
            .partition_point(|chunk| chunk.first().is_some_and(|(first, _)| below(first)))
            .saturating_sub(1)
    }

    /// Insert `value` at `key` where no entry has that key: whether it did.
    pub(crate) fn insert(&mut self, key: K, value: V) -> bool {
        if self.chunks.is_empty() {
            self.chunks.push(Arc::new(vec![(key, value)]));
            self.len = 1;
            return true;
        }
        let at = self.chunk_of(&key);
        let Err(place) = self.chunks[at].binary_search_by(|(k, _)| k.cmp(&key)) else {
            return false;
        };
        let chunk = Arc::make_mut(&mut self.chunks[at]);
        chunk.insert(place, (key, value));
        if chunk.len() > MOST {
            let back = chunk.split_off(chunk.len() / 2);
            self.chunks.insert(at + 1, Arc::new(back));
        }
        self.len += 1;
        true
    }

    /// Remove the entry at `key`: its value, where there was one.
    pub(crate) fn remove(&mut self, key: &K) -> Option<V> {
        let at = self.chunk_of(key);
        let place = self
            .chunks
            .get(at)?
            .binary_search_by(|(k, _)| k.cmp(key))
            .ok()?;
        let chunk = Arc::make_mut(&mut self.chunks[at]);
        let (_, value) = chunk.remove(place);
        if chunk.is_empty() {
            self.chunks.remove(at);
        }
        self.len -= 1;
        Some(value)
    }

    /// Whether it has an entry at `key`.
    pub(crate) fn contains(&self, key: &K) -> bool {
        let at = self.chunk_of(key);
        self.chunks
            .get(at)
            .is_some_and(|chunk| chunk.binary_search_by(|(k, _)| k.cmp(key)).is_ok())
    }

    /// The last entry whose key `below` holds for; `below` holds for a prefix of the keys.
    pub(crate) fn last_where(&self, below: impl Fn(&K) -> bool) -> Option<&(K, V)> {
        let chunk = self.chunks.get(self.chunk_where(&below))?;
        let after = chunk.partition_point(|(k, _)| below(k));
        chunk.get(after.checked_sub(1)?)
    }

    /// The entries in key order.
    pub(crate) fn iter(&self) -> impl Iterator<Item = &(K, V)> + '_ {
        self.chunks.iter().flat_map(|chunk| chunk.iter())
    }

    /// The chunks, for a test of what copies share.
    #[cfg(test)]
    pub(crate) fn chunks(&self) -> &[Arc<Vec<(K, V)>>] {
        &self.chunks
    }
}
