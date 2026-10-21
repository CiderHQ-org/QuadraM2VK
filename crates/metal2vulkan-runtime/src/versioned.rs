//! A value that counts its changes: every mutable access gives it a new version, so a memo keyed
//! by the version holds only while nothing could have changed the value (`render/draw_shape.rs`).
//! No list of setters is kept: a mutable borrow is the change, whoever takes it.

use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicU64, Ordering};

/// The next version, unique in the process: a value made or replaced whole never takes a version
/// another value had.
fn next_version() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

/// `T` and its version, which every mutable access ([`DerefMut`]) changes. Only a type whose
/// changes all take `&mut` (no interior mutability in what a memo reads of it) is kept so.
pub(crate) struct Versioned<T> {
    value: T,
    version: u64,
}

impl<T> Versioned<T> {
    /// `value` at a version of its own.
    pub(crate) fn new(value: T) -> Versioned<T> {
        Versioned {
            value,
            version: next_version(),
        }
    }

    /// Its version: equal versions, equal values.
    pub(crate) fn version(&self) -> u64 {
        self.version
    }

    /// The value, given up.
    pub(crate) fn into_inner(self) -> T {
        self.value
    }
}

impl<T> Deref for Versioned<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.value
    }
}

impl<T> DerefMut for Versioned<T> {
    /// The value to change: at a new version, whether the caller changes it or not.
    fn deref_mut(&mut self) -> &mut T {
        self.version = next_version();
        &mut self.value
    }
}

#[cfg(test)]
mod tests {
    use super::Versioned;

    #[test]
    fn a_mutable_access_moves_the_version_and_a_read_does_not() {
        let mut kept = Versioned::new(vec![1u32]);
        let made = kept.version();
        assert_eq!(kept.len(), 1);
        assert_eq!(kept.version(), made);
        kept.push(2);
        let pushed = kept.version();
        assert_ne!(pushed, made);
        // A mutable borrow is a change, though nothing changed.
        let _ = &mut *kept;
        assert_ne!(kept.version(), pushed);
        // Another value never takes a version this one had.
        let other = Versioned::new(vec![1u32]);
        assert!(other.version() > pushed);
        assert_eq!(kept.into_inner(), [1, 2]);
    }
}
