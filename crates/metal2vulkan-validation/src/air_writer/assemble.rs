//! `xcrun metal-as`: typed-pointer text to AIR bitcode (macOS only). CPU only, under the
//! compile deadline ([`crate::gpu::compile::xcrun`]).

use std::fmt;

use crate::gpu::compile::{xcrun, ToolError};
use crate::ScratchDir;

/// Why text could not be assembled.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AssembleError {
    /// `xcrun metal-as` is not available on this host (a test skips visibly).
    Unavailable(String),
    /// `metal-as` ran and rejected the text.
    Rejected(String),
    /// `metal-as` outlasted the compile deadline and was killed.
    Stopped(String),
}

impl fmt::Display for AssembleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AssembleError::Unavailable(why) => write!(f, "metal-as unavailable: {why}"),
            AssembleError::Rejected(why) => write!(f, "metal-as rejected the text: {why}"),
            AssembleError::Stopped(why) => write!(f, "metal-as was stopped: {why}"),
        }
    }
}

impl std::error::Error for AssembleError {}

/// Assemble typed-pointer LLVM `text` with `xcrun metal-as` into AIR bitcode. The scratch
/// directory is removed before returning.
pub fn metal_as(text: &str) -> Result<Vec<u8>, AssembleError> {
    if !cfg!(target_os = "macos") {
        return Err(AssembleError::Unavailable("metal-as needs macOS".into()));
    }
    let scratch =
        ScratchDir::new("air-writer").map_err(|e| AssembleError::Unavailable(e.to_string()))?;
    let input = scratch.path().join("module.ll");
    let output = scratch.path().join("module.air");
    std::fs::write(&input, text).map_err(|e| AssembleError::Unavailable(e.to_string()))?;
    let result = xcrun(
        "metal-as",
        [input.as_os_str(), "-o".as_ref(), output.as_os_str()],
    )
    .map_err(|error| match error {
        ToolError::Unavailable(why) => AssembleError::Unavailable(why),
        ToolError::Stopped(why) => AssembleError::Stopped(why),
    })?;
    if !result.success {
        let stderr = result.stderr;
        if stderr.contains("unable to find utility") || stderr.contains("xcrun: error") {
            return Err(AssembleError::Unavailable(stderr));
        }
        return Err(AssembleError::Rejected(stderr + &result.stdout));
    }
    std::fs::read(&output).map_err(|e| AssembleError::Rejected(format!("read output: {e}")))
}
