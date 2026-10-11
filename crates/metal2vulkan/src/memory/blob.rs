//! The pipeline constant blob.
//!
//! A program-scope constant that is not a closed, type-consistent `Private` variable (its
//! address merges, escapes, is compared, or it is punned or copied with `memcpy`) lives in the
//! **constant blob**: the initializers' little-endian bytes under the AIR datalayout, each at its
//! alignment (`memory::plan`). The artifact carries the bytes (`FunctionArtifact.blob`); the binder
//! uploads them once per pipeline and binds them as `WriteSource::ConstantBlob` at the stage's
//! `ResourceSlot{stage, Synthetic(ConstantBlob), 0}`, a read-only storage buffer (`StorageBuffer {
//! u32[] }`, the word view of `memory::views`) covering at least the blob rounded up to 16 bytes
//! (the padding contract).
//!
//! A pointer into the blob is its `u32` byte offset from the blob's start: a constant global's
//! address is its blob offset, and GEPs, casts, and `select` are integer operations (the arena
//! pointer form). Loads are word-composed leaves (`memory::leaf`); containment is the descriptor
//! encoding: none with robust descriptors, otherwise the normative check against the bound size
//! (`OpArrayLength · 4`).

use std::sync::Arc;

use crate::memory::views::{self, View, ViewKind};
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage, SyntheticBinding};
use crate::spirv::builder::ModuleBuilder;
use crate::spirv::error::BResult;

/// The blob a function reads: its bytes, which the artifact carries.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlobUse {
    pub bytes: Arc<[u8]>,
}

/// The binding symbol of `stage`'s constant blob.
pub fn symbol(stage: Stage) -> ResourceSlot {
    ResourceSlot {
        stage,
        kind: ResourceKind::Synthetic(SyntheticBinding::ConstantBlob),
        index: 0,
    }
}

/// Declare `stage`'s blob view: a storage word view at the blob's binding symbol.
pub fn declare(mb: &mut ModuleBuilder, stage: Stage) -> BResult<View> {
    views::declare_at(mb, symbol(stage), ViewKind::Storage)
}
