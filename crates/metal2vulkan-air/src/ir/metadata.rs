//! Module metadata: the graph `air::abi` decodes the ABI from.
//!
//! Nodes may refer to later nodes and to themselves (distinct loop ids do), so metadata is an
//! arena with checked references, not an interner. Debug-info records are kept only as
//! [`Md::Debug`] leaves so node numbering stays aligned with the bitcode.

use super::id::{ConstId, MdId, Sym};

/// One metadata entry.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Md {
    /// An `MDString`.
    String(Sym),
    /// A constant wrapped as metadata (`ValueAsMetadata` of a module-level value).
    Value(ConstId),
    /// A tuple node; `None` operands are `null`.
    Node {
        /// A `distinct !{...}` node, never merged with an equal one.
        distinct: bool,
        /// The operands, in order.
        ops: Box<[Option<MdId>]>,
    },
    /// A debug-info record, not interpreted.
    Debug,
}

/// A named metadata list (`!air.kernel = !{...}`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct NamedMd {
    /// The list's name, without the `!` (`air.kernel`).
    pub name: Sym,
    /// The nodes the list holds, in order.
    pub ops: Box<[MdId]>,
}
