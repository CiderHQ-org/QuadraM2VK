//! `MTLDynamicLibrary` in the compile layer: a library of type dynamic, whose exported functions
//! and variables resolve the symbols other libraries import when a pipeline stage links it, as a
//! preloaded library or as one an install name names.

use super::function::{Library, LibraryType};
use crate::budget::{Budget, Meter};
use crate::error::{DescriptorError, DynamicLibraryProblem, Error};

/// A dynamic library (`MTLDynamicLibrary`): a [`Library`] of [`LibraryType::Dynamic`], checked as
/// `newDynamicLibrary:error:` checks it ([`DynamicLibrary::new`]) or as a pipeline creation loads
/// one an install name names ([`DynamicLibrary::loaded`]). Cheap to clone; keeps no decoded IR.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DynamicLibrary {
    library: Library,
}

/// `InvalidDescriptor(DynamicLibrary { problem })`.
fn refused(problem: DynamicLibraryProblem) -> Error {
    Error::InvalidDescriptor(DescriptorError::DynamicLibrary { problem })
}

impl DynamicLibrary {
    /// `newDynamicLibrary:error:`: `library` as a dynamic library, refused as Metal refuses it
    /// (probes, an M3 Max and an M4) with `InvalidDescriptor(DynamicLibrary { problem })`: a
    /// library of another type ([`DynamicLibraryProblem::NotDynamic`], Metal's "Expected a library
    /// with type MTLLibraryTypeDynamic"), one without an install name
    /// ([`DynamicLibraryProblem::NoInstallName`]), and one whose code reads a function constant
    /// ([`DynamicLibraryProblem::FunctionConstants`], Metal's "cannot lower module with
    /// unresolved function constants"; a constant its code does not read is not in its AIR).
    /// Its module is decoded once under `budget` to check it, and dropped: a module that does
    /// not decode is that error, as Metal's compile of it fails.
    pub fn new(library: Library, budget: &Budget) -> Result<DynamicLibrary, Error> {
        let library = DynamicLibrary::loaded(library)?;
        let constants = crate::pipeline::compile::guarded(|| {
            let mut m = Meter::new(budget);
            let decoded = crate::air::decode(library.module(), &mut m)?;
            Ok(!decoded.abi.function_constants.is_empty())
        })?;
        if constants {
            return Err(refused(DynamicLibraryProblem::FunctionConstants));
        }
        Ok(library)
    }

    /// `library` as a pipeline creation loads the dynamic library an install name names: refused as
    /// [`DynamicLibrary::new`] refuses a library of another type
    /// ([`DynamicLibraryProblem::NotDynamic`]) or one without an install name
    /// ([`DynamicLibraryProblem::NoInstallName`]), with its module not decoded. What a stage
    /// cannot link of it (a module that does not decode, a function constant it reads) is
    /// refused only when the stage's search reaches it, so a stage whose imports specialization
    /// deletes never decodes it, as Metal loads a stage's dependencies lazily.
    pub fn loaded(library: Library) -> Result<DynamicLibrary, Error> {
        if library.library_type() != LibraryType::Dynamic {
            return Err(refused(DynamicLibraryProblem::NotDynamic));
        }
        if library.install_name().is_none() {
            return Err(refused(DynamicLibraryProblem::NoInstallName));
        }
        if library.dynamic_module().is_none() {
            return Err(Error::Internal(crate::error::InternalError::new(
                crate::error::InternalKind::Invariant,
                "a dynamic library without its module",
            )));
        }
        Ok(DynamicLibrary { library })
    }

    /// `installName`: the name a dependency on this library gives.
    pub fn install_name(&self) -> &str {
        self.library.install_name().unwrap_or_default()
    }

    /// The library it was made from: its bytes are what `serializeToURL:error:` writes.
    pub fn library(&self) -> &Library {
        &self.library
    }

    /// Its module, which every function it exports shares.
    pub(crate) fn module(&self) -> &[u8] {
        self.library.dynamic_module().unwrap_or_default()
    }
}
