//! A render encoder's bind state, kept between its draws: each draw brings the state the encoder's
//! last draw bound to its own in place, and makes its bind shape again only where a change may have
//! changed it.
//!
//! - **Brought in place.** Each slot's binding is derived again from the encoder's tables at each
//!   draw and compared with the kept one (`ArgumentTable::refresh_reading`), and every other field
//!   is computed as before and compared ([`Reshape::field`]). The new value is always stored, so
//!   the state is the draw's exactly, a float of the other sign (which compares equal) included.
//!   No `set*` call marks a slot: a bound buffer's pages may be rebound or detached between draws,
//!   which changes its binding with no call, so deriving the binding is the check. While the same
//!   slots stay bound, nothing is allocated.
//! - **The shape made again only where it may have changed.** A bind shape is each bound slot's
//!   shape beside the shapes of the state's other fields (`BindState::shape`: a stage's tables
//!   are its slots' shapes, and a stage with none is left out). Where every changed slot keeps its
//!   shape, no slot was bound or unbound, and no other field changed, the shape is the kept one.
//!   Any other change makes it again, whole.

use metal2vulkan::bind::{BindShape, BindState};
use std::collections::btree_map::Entry;
use std::collections::BTreeMap;
use std::sync::Arc;

use super::work_counts::Counters;

/// Whether the changes brought into a kept bind state may have changed its shape.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Reshape(bool);

impl Reshape {
    /// Store `value` at `index` of `map`, a slot table of a kept stage table: its shape may change
    /// where the index was unbound, or where the binding's `shape` changes.
    pub(super) fn slot<V: PartialEq, S: PartialEq>(
        &mut self,
        map: &mut BTreeMap<u32, V>,
        index: u32,
        value: V,
        shape: impl Fn(&V) -> S,
    ) {
        match map.entry(index) {
            Entry::Vacant(entry) => {
                entry.insert(value);
                self.0 = true;
            }
            Entry::Occupied(mut entry) => {
                if !self.0 && *entry.get() != value && shape(entry.get()) != shape(&value) {
                    self.0 = true;
                }
                entry.insert(value);
            }
        }
    }

    /// A slot of a kept stage table unbound.
    pub(super) fn unbound(&mut self) {
        self.0 = true;
    }

    /// Store `value` in `field`, a field of the kept state beside its stage tables: any change
    /// may change the shape.
    pub(super) fn field<T: PartialEq>(&mut self, field: &mut T, value: T) {
        if *field != value {
            self.0 = true;
        }
        *field = value;
    }
}

/// A render encoder's bind state as its last draw bound it, and its shape while no change may
/// have changed it (the module docs).
#[derive(Default)]
pub(super) struct LiveBind {
    /// The state. Each draw brings it to its own through [`Reshape`] (its values past its settle,
    /// which no shape reads, aside), and then hands the reshape to [`reshaped`](Self::reshaped).
    pub(super) state: BindState,
    /// The state's shape, `None` where a change may have changed it.
    shape: Option<Arc<BindShape>>,
}

impl LiveBind {
    /// Take the changes brought into the state since its shape was made.
    pub(super) fn reshaped(&mut self, reshape: Reshape) {
        if reshape.0 {
            self.shape = None;
        }
    }

    /// The state's shape: the kept one, else made again (counted in `counts`).
    pub(super) fn shape(&mut self, counts: &Counters) -> Arc<BindShape> {
        if let Some(shape) = &self.shape {
            debug_assert!(
                **shape == self.state.shape(),
                "a kept bind shape that is not its state's"
            );
            return Arc::clone(shape);
        }
        counts.draw_shape_made();
        let shape = Arc::new(self.state.shape());
        self.shape = Some(Arc::clone(&shape));
        shape
    }
}
