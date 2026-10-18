//! A list of at most one element per color attachment of a pass, kept inline: a pass's shape
//! (`render_target::PassShape`) is made, compared and copied as each render pass opens and ends,
//! and its lists allocate nothing.

use super::error::{internal, Error};
use super::render_pass::MAX_COLOR_ATTACHMENTS;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::{Deref, DerefMut};

/// At most [`MAX_COLOR_ATTACHMENTS`] elements, inline; compared, hashed and printed as the
/// slice of them.
#[derive(Clone)]
pub(crate) struct ColorList<T> {
    len: usize,
    /// The elements, then defaults past `len`.
    items: [T; MAX_COLOR_ATTACHMENTS],
}

impl<T: Copy + Default> ColorList<T> {
    /// The empty list.
    pub(crate) fn new() -> ColorList<T> {
        ColorList {
            len: 0,
            items: [T::default(); MAX_COLOR_ATTACHMENTS],
        }
    }

    /// Add `item` at the end; refused (an invariant) where the list is full.
    pub(crate) fn push(&mut self, item: T) -> Result<(), Error> {
        let slot = (self.items.get_mut(self.len))
            .ok_or_else(|| internal("more elements than a pass has color attachments"))?;
        *slot = item;
        self.len += 1;
        Ok(())
    }

    /// The list of `items`; refused (an invariant) past [`MAX_COLOR_ATTACHMENTS`] of them.
    pub(crate) fn of(items: impl IntoIterator<Item = T>) -> Result<ColorList<T>, Error> {
        let mut list = ColorList::new();
        for item in items {
            list.push(item)?;
        }
        Ok(list)
    }
}

impl<T: Copy + Default> Default for ColorList<T> {
    fn default() -> ColorList<T> {
        ColorList::new()
    }
}

impl<T> Deref for ColorList<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        &self.items[..self.len]
    }
}

impl<T> DerefMut for ColorList<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        &mut self.items[..self.len]
    }
}

impl<T: PartialEq> PartialEq for ColorList<T> {
    fn eq(&self, other: &ColorList<T>) -> bool {
        **self == **other
    }
}

impl<T: PartialEq> PartialEq<&[T]> for ColorList<T> {
    fn eq(&self, other: &&[T]) -> bool {
        **self == **other
    }
}

impl<T: Eq> Eq for ColorList<T> {}

impl<'a, T> IntoIterator for &'a ColorList<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> std::slice::Iter<'a, T> {
        self.iter()
    }
}

impl<T: Hash> Hash for ColorList<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state);
    }
}

impl<T: fmt::Debug> fmt::Debug for ColorList<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (**self).fmt(f)
    }
}
