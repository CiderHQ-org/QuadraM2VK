//! Support shared by the integration tests, which reach the crate only through its public paths:
//! they take `llvm-as` from the dev-tools crate (`devtools/`).

use metal2vulkan_devtools::{llvm_assemble_text, AssembleError};

/// The bitcode `llvm-as` assembles from `text`, or `None` (a visible skip) when `llvm-as` is not
/// installed. The text goes in on standard input, so the bitcode does not depend on where a text
/// lies, and the tool resolves as the crate's own test helpers resolve it: `METAL2VULKAN_LLVM_AS`,
/// then the Homebrew LLVM prefixes, then `PATH`. A rejection fails the test, naming `what`.
pub fn llvm_as(text: &str, what: &str) -> Option<Vec<u8>> {
    match llvm_assemble_text(text) {
        Ok(bitcode) => Some(bitcode),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {what}: {why}"),
    }
}
