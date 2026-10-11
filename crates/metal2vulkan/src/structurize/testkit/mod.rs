//! Structurizer test support: a test [`CfgView`](super::CfgView), a CFG interpreter
//! for trace equivalence, and CFG generators. Rewritten from the old `native::cfg_testkit` on the
//! `graph` module and the new working graph; the old one runs SPIR-V and is not reused.
//!
//! - [`TestCfg`]: an AIR-shaped CFG (entry `bb0`) whose blocks carry a tiny program: executing a
//!   block appends it to the trace and mixes its id into a 64-bit state; every branch condition,
//!   switch selector, and return value is a hash of that state and the terminator's value id.
//!   Decisions are therefore a pure function of the trace so far, so two graphs that execute the
//!   same original blocks in the same order decide identically, and any divergence shows up as a
//!   different trace. [`hash`] is that scheme, for every model that must decide the same way.
//! - [`interp`]: runs a [`TestCfg`], a [`StructCfg`](super::cfg::StructCfg) (synthetic
//!   selector blocks included), or a [`StructTree`](super::tree::StructTree) (P6 or P7 output) on
//!   a seed and compares the first `K` original blocks executed and how the run ended. Prefix
//!   comparison makes non-terminating inputs (arbitrary exhaustive graphs loop forever on some
//!   seeds) testable.
//! - [`tangle`]: the ideal tangles, which convergence equivalence compares SIMT models
//!   against.
//! - [`check`]: the tree-level structural checker.
//! - [`generate`]: exhaustive canonical enumerations, structured programs with goto injection,
//!   reducible DAGs with back edges, irreducible families, and named exit-tail and switch shapes.
//!
//! This is test support compiled with the core (so the other families' tests and the validation
//! crate can use it); an out-of-range index here is a test bug, so it may panic.
#![allow(clippy::indexing_slicing)]

pub mod check;
pub mod generate;
pub mod hash;
pub mod interp;
pub mod tangle;
mod view;

pub use view::{TestCfg, TestKind, SELECTOR_RANGE};
