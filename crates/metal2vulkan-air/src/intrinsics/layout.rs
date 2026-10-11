//! Operand layouts: a signature whose positions carry a named role.
//!
//! Families with many operand shapes (textures, simdgroup matrices, async copies) derive their
//! layout from the variant as a list of `(role, operand)` pairs. The signature is the operand
//! column; the role column tells lowering (and the immediate reader) which position is which
//! without re-deriving the grammar.

use super::sig::{Operand, Signature, MAX_OPERANDS};

/// A fixed-capacity list of `(R, Operand)` positions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Layout<R: Copy> {
    ret: Operand,
    slots: [Option<(R, Operand)>; MAX_OPERANDS],
    len: u8,
}

impl<R: Copy + PartialEq> Layout<R> {
    /// An empty layout returning `ret`.
    pub fn new(ret: Operand) -> Layout<R> {
        Layout {
            ret,
            slots: [None; MAX_OPERANDS],
            len: 0,
        }
    }

    /// Append position `role` of shape `op`. More than `MAX_OPERANDS` positions is a table bug
    /// the totality tests catch; the extra position is dropped, so the signature cannot match.
    pub fn push(&mut self, role: R, op: Operand) {
        if let Some(slot) = self.slots.get_mut(usize::from(self.len)) {
            *slot = Some((role, op));
            self.len += 1;
        } else {
            // Poison the layout: no declaration has a `Void` operand.
            if let Some(Some(last)) = self.slots.last_mut() {
                last.1 = Operand::Void;
            }
        }
    }

    /// Append `role` when `cond` holds.
    pub fn push_if(&mut self, cond: bool, role: R, op: Operand) {
        if cond {
            self.push(role, op);
        }
    }

    /// The positions in order.
    pub fn positions(&self) -> impl Iterator<Item = (R, Operand)> + '_ {
        self.slots
            .iter()
            .take(usize::from(self.len))
            .flatten()
            .copied()
    }

    /// The index of the first position with role `role`.
    pub fn index(&self, role: R) -> Option<usize> {
        self.positions().position(|(r, _)| r == role)
    }

    /// The typed signature.
    pub fn signature(&self) -> Signature {
        let mut ops = [Operand::Void; MAX_OPERANDS];
        let mut n = 0;
        for (slot, (_, op)) in ops.iter_mut().zip(self.positions()) {
            *slot = op;
            n += 1;
        }
        Signature::new(self.ret, ops.get(..n).unwrap_or(&[]))
    }
}
