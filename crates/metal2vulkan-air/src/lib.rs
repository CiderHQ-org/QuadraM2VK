//! Read Metal AIR: the front end of [metal2vulkan](https://docs.rs/metal2vulkan).
//!
//! AIR is the LLVM-based intermediate representation Apple's Metal compiler writes. This crate
//! reads it and hands back a typed, canonical form:
//!
//! - [`container`]: the metallib container (its function list, tags, and each function's bitcode,
//!   hash-checked), or a bare or wrapped bitcode file;
//! - [`bitstream`] and [`bitcode`]: the LLVM bitstream and the bitcode records of the subset AIR
//!   uses, decoded by [`decode`] into a [`Decoded`] module;
//! - [`ir`]: the typed IR the decoder builds (types, constants, globals, functions, blocks and
//!   instructions, with a builder and a verifier);
//! - [`abi`]: the AIR metadata ABI (entry points, their parameters' roles and bindings, struct
//!   layouts, builtins, stages), cross-checked against the IR;
//! - [`intrinsics`]: the `air.*` and `llvm.*` intrinsic families, as typed operations;
//! - [`canon`]: canonicalization of one entry point under its function-constant values and linked
//!   functions ([`specialize_linked`]): linking, constant binding, static initializers, inlining,
//!   devirtualization, SCCP, and pruning, into one [`Entry`];
//! - [`tensor_ops`]: the tensor-operation descriptors;
//! - [`metal`]: the Metal API enums AIR carries ([`metal::DataType`]);
//! - [`graph`]: control-flow graphs, dominators, loops and strongly connected components;
//! - [`budget`]: the per-attempt [`Budget`](budget::Budget) and the [`Meter`] every input-scaled
//!   loop is charged to (a deadline, cancellation, work units, a byte ledger, structural caps);
//! - [`error`]: why reading or canonicalizing stopped ([`error::Error`]).
//!
//! Every reader is bounded by its input and by the meter: malformed bytes are a typed error,
//! never a panic, and nothing is allocated before it is charged.
//!
//! **Versioning.** This crate moves in lockstep with `metal2vulkan`, which depends on it with an
//! exact pin (`=0.x.y`): both carry the same version, and a release of one is a release of both.
//! Within 0.x, a minor version may change any item here.

// Every public item is documented: this crate is published and its whole surface is public.
#![warn(missing_docs)]
// `too_many_arguments` and `type_complexity` are threshold heuristics that fire pervasively and
// benignly across the front end: decoding and canonicalization functions legitimately thread many
// typed parameters, and their return and table shapes are genuinely nested domain types (the
// translator accepts both crate-wide for the same reason).
#![allow(clippy::too_many_arguments, clippy::type_complexity)]

// Panic-lint policy for the new core: the AIR readers and canonicalization may not unwrap or index
// unchecked; tests are exempt. `budget` and `graph`, the translator's own modules before this crate
// and outside that policy there, are outside it here too.
#[cfg_attr(not(test), deny(clippy::unwrap_used, clippy::indexing_slicing))]
pub mod abi;
#[cfg_attr(not(test), deny(clippy::unwrap_used, clippy::indexing_slicing))]
pub mod bitcode;
#[cfg_attr(not(test), deny(clippy::unwrap_used, clippy::indexing_slicing))]
pub mod bitstream;
pub mod budget;
#[cfg_attr(not(test), deny(clippy::unwrap_used, clippy::indexing_slicing))]
pub mod canon;
#[cfg_attr(not(test), deny(clippy::unwrap_used, clippy::indexing_slicing))]
pub mod container;
#[cfg_attr(not(test), deny(clippy::unwrap_used, clippy::indexing_slicing))]
pub mod error;
pub mod graph;
#[cfg_attr(not(test), deny(clippy::unwrap_used, clippy::indexing_slicing))]
pub mod intrinsics;
#[cfg_attr(not(test), deny(clippy::unwrap_used, clippy::indexing_slicing))]
pub mod ir;
#[cfg_attr(not(test), deny(clippy::unwrap_used, clippy::indexing_slicing))]
pub mod metal;
#[cfg_attr(not(test), deny(clippy::unwrap_used, clippy::indexing_slicing))]
pub mod tensor_ops;

pub use bitcode::{decode, Decoded};
pub use canon::{ConstValue, DynamicLinks, DynamicModule, Entry, LinkedCode, LinkedModule};
pub use canon::{LinkedName, LinkedSamplerGroup, Lockstep, SpecRequest};

use crate::budget::Meter;
use crate::error::Error;

/// Specialize a decoded module for one entry point and its function-constant values, using the ABI
/// [`decode`] already cross-checked (`d.abi`). The pointee table is dropped here: nothing after
/// decode reads it.
pub fn specialize(d: Decoded, r: &SpecRequest, m: &mut Meter) -> Result<Entry, Error> {
    let Decoded { module, abi, .. } = d;
    canon::specialize(module, &abi, r, m)
}

/// [`specialize`], resolving the module's visible-function references against the pipeline's
/// `linked` functions.
pub fn specialize_linked(
    d: Decoded,
    r: &SpecRequest,
    linked: &[LinkedModule],
    m: &mut Meter,
) -> Result<Entry, Error> {
    let Decoded { module, abi, .. } = d;
    canon::specialize_linked(module, &abi, r, linked, m)
}

/// [`specialize_linked`], resolving the module's dynamic-library imports (the declarations its
/// `air.dyld_flat_table` names) against the stage's `dynamic` libraries too.
pub fn specialize_dynamic(
    d: Decoded,
    r: &SpecRequest,
    (linked, dynamic): (&[LinkedModule], &DynamicLinks),
    m: &mut Meter,
) -> Result<Entry, Error> {
    let Decoded { module, abi, .. } = d;
    canon::specialize_dynamic(module, &abi, r, (linked, dynamic), m)
}
