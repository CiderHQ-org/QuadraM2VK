//! Metal libraries and their functions: the AIR input.
//!
//! [`Library::parse`] takes a metallib, a bitcode wrapper, or a raw AIR bitcode module as owned
//! bytes ([`Library::from_air`] takes the latter two only) and reads its function table. LLVM text
//! is not an input: assemble `.ll` with `llvm-as` first. A library is cheap to clone and keeps no
//! decoded IR. [`Library::functions`] lists its functions ([`FunctionInfo`]: name, stage, content
//! hash, declared function constants) without decoding them, and [`Library::function`] finds one
//! by name, like `newFunctionWithName:`.
//!
//! A [`Function`] becomes a pipeline stage through a
//! [`FunctionDescriptor`](crate::pipeline::FunctionDescriptor), with its
//! [`FunctionConstantValues`] and the functions it links. A pipeline compiles its function stages
//! itself. A consumer that caches per function compiles one with
//! [`Device::compile_function`](crate::device::Device::compile_function) into a
//! [`CompiledFunction`]: opaque, keyed by the [`FunctionCompileKey`] that
//! [`Device::function_compile_key`](crate::device::Device::function_compile_key) computes without
//! decoding, encoded with [`CompiledFunction::to_bytes`], and decoded with
//! [`CompiledFunction::from_bytes_for_key`]. A pipeline descriptor takes it back as
//! [`StageSource::Compiled`](crate::pipeline::StageSource::Compiled), and the pipeline links it
//! without decoding or lowering it again.
//!
//! A metallib of type dynamic ([`LibraryType::Dynamic`], `metal -dynamiclib`) becomes a
//! [`DynamicLibrary`], as `newDynamicLibrary:error:` makes one: a pipeline stage links it as a
//! preloaded library or through an install name its library depends on, and its exports resolve
//! the symbols the stage imports.
//!
//! Errors: bytes that are not a library are `Error::InvalidInput`, and a name the table lacks is
//! `Error::InvalidDescriptor(EntryNotFound)`. Function-constant values are checked against the
//! function's declarations when it compiles (`Error::InvalidDescriptor`).
//!
//! The frozen surface is the items named at this root; the submodules are crate-private.

pub(crate) mod compiled;
pub(crate) mod dynamic;
pub(crate) mod function;

pub use compiled::{CompiledFunction, FunctionCompileKey};
// The artifact behind `CompiledFunction`: internal.
pub(crate) use compiled::FunctionArtifact;
pub(crate) use compiled::KernelFacts;
pub use dynamic::DynamicLibrary;
pub use function::{
    Function, FunctionConstant, FunctionConstantValue, FunctionConstantValues, FunctionInfo,
    FunctionStage, Library, LibraryType,
};
