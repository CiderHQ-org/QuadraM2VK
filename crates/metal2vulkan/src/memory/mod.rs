//! Memory analysis and access emission.

// Panic-lint policy for the new core: product code in this tree may not unwrap or index unchecked;
// tests are exempt.
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::indexing_slicing))]

pub mod abi_index;
pub mod alloca;
pub mod arena;
pub mod atomic;
pub mod blob;
pub mod contain;
pub mod copy;
pub mod fragment;
pub mod handles;
pub mod leaf;
pub mod plan;
pub mod provenance;
pub mod roots;
pub mod static_root;
pub(crate) mod union_find;
pub mod views;
pub mod window;
