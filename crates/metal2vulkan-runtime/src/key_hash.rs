//! The hasher of the runtime's maps keyed by its own identities (`ResourceKey`, sequence numbers):
//! a multiply-and-fold of each word, as the hazard tracker looks up every resource a batch uses
//! as it plans the batch (`tracker.rs`).
//!
//! The standard maps' default hasher (SipHash-1-3 with random keys) resists inputs chosen to
//! collide, which these keys, numbered by the runtime, are not, and costs its rounds at every
//! lookup. The multiply spreads consecutive numbers over the high bits the map's control bytes
//! read, and the fold brings them down to the low bits its buckets index by. Iteration order
//! follows the hash, which none of these maps' users depends on.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

/// A map keyed by the runtime's own identities ([`KeyHasher`]).
pub(crate) type KeyMap<K, V> = HashMap<K, V, BuildHasherDefault<KeyHasher>>;

/// The module docs' hasher.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct KeyHasher(u64);

impl KeyHasher {
    /// Fold `word` into the hash.
    fn mix(&mut self, word: u64) {
        let spread = (self.0 ^ word).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        self.0 = spread ^ (spread >> 32);
    }
}

impl Hasher for KeyHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        let (words, rest) = bytes.as_chunks::<8>();
        for word in words {
            self.mix(u64::from_le_bytes(*word));
        }
        if !rest.is_empty() {
            let mut last = [0u8; 8];
            last[..rest.len()].copy_from_slice(rest);
            self.mix(u64::from_le_bytes(last) ^ ((rest.len() as u64) << 56));
        }
    }

    fn write_u8(&mut self, value: u8) {
        self.mix(u64::from(value));
    }

    fn write_u16(&mut self, value: u16) {
        self.mix(u64::from(value));
    }

    fn write_u32(&mut self, value: u32) {
        self.mix(u64::from(value));
    }

    fn write_u64(&mut self, value: u64) {
        self.mix(value);
    }

    fn write_usize(&mut self, value: usize) {
        self.mix(value as u64);
    }
}
