//! Proven slot windows `Bounded | Whole`.
//!
//! A slot's window is the byte range, relative to its bind offset, that every Metal-defined
//! access through the slot stays inside. It is proven from the accesses' extents
//! ([`Extent`], computed by provenance) and the ABI:
//!
//! - Any `Dynamic` root in the function makes every window `Whole`: a pointer loaded from memory
//!   or made from an integer may reach any byte of any bound allocation.
//! - Otherwise, when every access has a constant offset range and length, the window is the hull
//!   of the accessed bytes (it may start below zero: negative GEPs are Metal-defined when they
//!   stay inside the allocation).
//! - Otherwise a reference parameter (`air.buffer_size`: a sized pointee, `constant T&`) bounds
//!   its Metal-defined accesses to `[0, buffer_size)`; a constant access outside the object is
//!   not Metal-defined and is contained, not covered. `air.arg_type_size` does not bound anything:
//!   it is the pointee *element* size of an unsized pointer parameter too.
//! - Otherwise `Whole`.

use super::provenance::Extent;
use crate::pipeline::interface::Window;

/// The window of a slot whose accesses cover `extent`, for a parameter stating `buffer_size`,
/// in a function with (`dynamic`) or without any dynamic root. `None` when the slot is never
/// accessed.
pub fn window(extent: Extent, buffer_size: Option<u32>, dynamic: bool) -> Option<Window> {
    if extent == Extent::Empty {
        return None;
    }
    if dynamic {
        return Some(Window::Whole);
    }
    match (extent, buffer_size) {
        (Extent::Span { lo, hi }, _) => Some(bounded(lo, hi)),
        (Extent::Unknown, Some(size)) => Some(bounded(0, i64::from(size))),
        (Extent::Unknown, None) | (Extent::Empty, _) => Some(Window::Whole),
    }
}

/// `[lo, hi)` as a window; an empty or inverted span is an empty window at `lo`.
fn bounded(lo: i64, hi: i64) -> Window {
    let len = u64::try_from(i128::from(hi) - i128::from(lo)).unwrap_or(0);
    Window::Bounded { start: lo, len }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unaccessed_slot_has_no_window() {
        assert_eq!(window(Extent::Empty, Some(16), false), None);
        assert_eq!(window(Extent::Empty, None, true), None);
    }

    #[test]
    fn a_dynamic_root_makes_every_window_whole() {
        let span = Extent::Span { lo: 0, hi: 4 };
        assert_eq!(window(span, Some(4), true), Some(Window::Whole));
        assert_eq!(window(Extent::Unknown, None, true), Some(Window::Whole));
    }

    #[test]
    fn constant_extents_give_their_hull_including_negative_starts() {
        assert_eq!(
            window(Extent::Span { lo: 8, hi: 20 }, None, false),
            Some(Window::Bounded { start: 8, len: 12 })
        );
        assert_eq!(
            window(Extent::Span { lo: -16, hi: 4 }, Some(4), false),
            Some(Window::Bounded {
                start: -16,
                len: 20
            })
        );
    }

    #[test]
    fn a_reference_bounds_its_dynamic_accesses_and_a_pointer_does_not() {
        assert_eq!(
            window(Extent::Unknown, Some(6), false),
            Some(Window::Bounded { start: 0, len: 6 })
        );
        assert_eq!(window(Extent::Unknown, None, false), Some(Window::Whole));
    }

    #[test]
    fn the_extreme_span_does_not_overflow() {
        assert_eq!(
            window(
                Extent::Span {
                    lo: i64::MIN,
                    hi: i64::MAX
                },
                None,
                false
            ),
            Some(Window::Bounded {
                start: i64::MIN,
                len: u64::MAX
            })
        );
    }
}
