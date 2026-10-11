//! Visible-function tables in the engine candidate.
//!
//! A case's `visible_function_tables` are bound as the product's binder says a consumer binds a
//! table (`bind::table`): a buffer the consumer owns, bound at the table's Metal buffer index as a
//! `BoundBuffer::Buffer` of the table's stage, holding the table's `FunctionTableWords`: a bound
//! table of the authored size, each authored entry the `FunctionHandle` of the function it holds,
//! every other entry null. A function's handle is what `LinkedFunctions::function_handle` answers
//! over the linked functions the compile built for the stage's function. Those exist only in the
//! compile worker, so the worker returns the handles with the pipeline
//! ([`crate::acceptance::LinkedHandles`]), and the executor encodes the words here. The Metal
//! oracle builds the same table with `newVisibleFunctionTableWithDescriptor:` and
//! `setFunction:atIndex:` of `functionHandleWithFunction:` over the same linked functions.

use super::LinkedModule;
use crate::acceptance::LinkedHandles;
use crate::case::FunctionTableResource;
use crate::check::admission::{ExecutorError, NotSupported};
use metal2vulkan::bind::FunctionTableWords;
use metal2vulkan::pipeline::Stage as ShaderStage;
use std::collections::BTreeMap;

/// The bytes of each visible-function table's buffer, by stage and Metal buffer index: what a run
/// binds there.
pub type TableBuffers = BTreeMap<(ShaderStage, u32), Vec<u8>>;

/// One authored visible-function table: its size (`functionCount`) and the linked function each
/// non-null entry holds, by entry index.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FunctionTable {
    pub size: u32,
    pub entries: BTreeMap<u32, TableFunction>,
}

/// The linked function a table entry holds: its module, by SHA-256, and its Metal name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TableFunction {
    pub module_sha256: String,
    pub function: String,
}

impl FunctionTable {
    /// The length in bytes of the table's buffer (`FunctionTableWords::byte_len`).
    pub fn byte_len(&self) -> u64 {
        FunctionTableWords::byte_len(self.size)
    }
}

/// The case's visible-function tables (`authored`), by Metal buffer index. A table bound twice,
/// an entry set twice, and an entry past the table's size are refused (the case checker refuses
/// each first).
pub fn function_tables(
    authored: &[FunctionTableResource],
) -> Result<BTreeMap<u32, FunctionTable>, NotSupported> {
    let gap = |what: String| NotSupported::NotImplemented(format!("engine candidate: {what}"));
    let mut tables = BTreeMap::new();
    for table in authored {
        let mut entries = BTreeMap::new();
        for entry in &table.entries {
            if entry.index >= table.size {
                return Err(gap(format!(
                    "visible function table {} entry {} is past its size {}",
                    table.binding, entry.index, table.size
                )));
            }
            let function = TableFunction {
                module_sha256: entry.module_sha256.clone(),
                function: entry.function.clone(),
            };
            if entries.insert(entry.index, function).is_some() {
                return Err(gap(format!(
                    "visible function table {} entry {} set twice",
                    table.binding, entry.index
                )));
            }
        }
        let table_value = FunctionTable {
            size: table.size,
            entries,
        };
        if tables.insert(table.binding, table_value).is_some() {
            return Err(gap(format!(
                "visible function table {} bound twice",
                table.binding
            )));
        }
    }
    Ok(tables)
}

/// The buffer bytes of each of `tables`, a `stage` function's, keyed by `stage` and Metal index:
/// `FunctionTableWords` of the table's size with each entry the handle of the function it holds.
/// `handles` are the function's linked handles ([`crate::acceptance::LinkedHandles`]), which
/// answer its linked libraries `linked` in order; an entry's function is found in them by module
/// and name. A function that is not linked, or that the product gives no handle, fails the run:
/// the executor links every table entry into `functions` ([`super::linked_modules`]), each of
/// which has one.
pub fn table_buffers(
    stage: ShaderStage,
    tables: &BTreeMap<u32, FunctionTable>,
    linked: &[LinkedModule],
    handles: &LinkedHandles,
) -> Result<TableBuffers, ExecutorError> {
    let mut buffers = TableBuffers::new();
    for (&binding, table) in tables {
        let mut words = FunctionTableWords::new(table.size);
        for (&index, entry) in &table.entries {
            let handle = linked
                .iter()
                .zip(handles)
                .find(|(module, _)| module.module_sha256 == entry.module_sha256)
                .and_then(|(module, handles)| {
                    let at = module
                        .library
                        .functions
                        .iter()
                        .position(|name| *name == entry.function)?;
                    handles.get(at).copied()
                })
                .ok_or_else(|| {
                    ExecutorError::Failed(format!(
                        "visible function table {binding} entry {index}: {} of module {} is not \
                         linked",
                        entry.function, entry.module_sha256
                    ))
                })?
                .ok_or_else(|| {
                    ExecutorError::Failed(format!(
                        "visible function table {binding} entry {index}: the product gives linked \
                         function {} no table handle",
                        entry.function
                    ))
                })?;
            if !words.set(index, Some(handle)) {
                return Err(ExecutorError::Failed(format!(
                    "visible function table {binding} has no entry {index}"
                )));
            }
        }
        buffers.insert((stage, binding), words.to_bytes());
    }
    Ok(buffers)
}
