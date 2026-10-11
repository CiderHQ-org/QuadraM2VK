//! Relocation of serialized function modules.
//!
//! `relocate` writes only the literals the serializer reserved (descriptor set and binding,
//! `Location`/`Component`, `SpecId`); a symbol the table does not assign is a `BuildError`
//! (`Internal`). The words of a module with its relocation slots zeroed are therefore the same
//! for every link.

use crate::error::Error;
use crate::library::FunctionArtifact;
use crate::spirv::serialize::SymbolTable;

/// The final SPIR-V words of `a` under link table `link` (a `LinkTable` or the `Linked` result of
/// the pipeline `a` was linked into).
pub fn relocate(a: &FunctionArtifact, link: &impl SymbolTable) -> Result<Vec<u32>, Error> {
    Ok(crate::spirv::serialize::relocate(&a.module, link)?)
}
