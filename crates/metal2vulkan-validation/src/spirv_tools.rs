//! The SPIR-V tools validation runs on its own shaders (the executor's helper pipelines, the
//! runtime probe's modules) and its tests' modules: `spirv-as` and `spirv-val` for the Vulkan 1.2
//! environment every module the product emits satisfies. Validation's own runner, so that it reads
//! no tool runner of the product; a tool resolves as the product's runner resolves it.

use crate::ScratchDir;
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// The `--target-env` of every assembly and validation here: Vulkan 1.2, the floor every module
/// the product emits satisfies.
pub const VULKAN_TARGET_ENV: &str = "vulkan1.2";

/// Why a tool did not answer.
#[derive(Debug)]
pub enum ToolError {
    /// The tool could not be started (not installed, or its override names no program).
    Launch(String),
    /// The tool ran and refused the input, or another step failed.
    Failed(String),
}

impl std::fmt::Display for ToolError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Launch(why) | Self::Failed(why) => formatter.write_str(why),
        }
    }
}

impl From<ToolError> for String {
    fn from(error: ToolError) -> String {
        error.to_string()
    }
}

/// `tool`'s program: `METAL2VULKAN_<TOOL>` (uppercased, `-` as `_`), then the Homebrew prefixes,
/// then `PATH`.
fn program(tool: &str) -> PathBuf {
    let name = format!(
        "METAL2VULKAN_{}",
        tool.replace('-', "_").to_ascii_uppercase()
    );
    if let Some(path) = std::env::var_os(name) {
        return PathBuf::from(path);
    }
    [
        "/opt/homebrew/opt/llvm/bin",
        "/usr/local/opt/llvm/bin",
        "/opt/homebrew/bin",
        "/usr/local/bin",
    ]
    .iter()
    .map(|dir| Path::new(dir).join(tool))
    .find(|candidate| candidate.is_file())
    .unwrap_or_else(|| PathBuf::from(tool))
}

/// Run `tool` with `args`, its output collected; an unsuccessful exit is its standard error.
fn run(tool: &str, args: &[&OsStr]) -> Result<(), ToolError> {
    let executable = program(tool);
    let output = Command::new(&executable)
        .args(args)
        .stdin(Stdio::null())
        .output()
        .map_err(|error| {
            ToolError::Launch(format!(
                "cannot run {tool} ({}): {error}",
                executable.display()
            ))
        })?;
    if output.status.success() {
        Ok(())
    } else {
        Err(ToolError::Failed(format!(
            "{tool} failed ({}):\n{}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        )))
    }
}

fn io(what: &str, path: &Path, error: std::io::Error) -> ToolError {
    ToolError::Failed(format!("{what} {}: {error}", path.display()))
}

/// The SPIR-V bytes `spirv-as` assembles from `assembly`, in a scratch directory named for
/// `label`.
pub fn assemble(assembly: &str, label: &str) -> Result<Vec<u8>, ToolError> {
    let scratch = ScratchDir::new(label).map_err(ToolError::Failed)?;
    let asm = scratch.path().join("module.spvasm");
    let spv = scratch.path().join("module.spv");
    std::fs::write(&asm, assembly).map_err(|error| io("write", &asm, error))?;
    run(
        "spirv-as",
        &[
            OsStr::new("--target-env"),
            OsStr::new(VULKAN_TARGET_ENV),
            asm.as_os_str(),
            OsStr::new("-o"),
            spv.as_os_str(),
        ],
    )?;
    std::fs::read(&spv).map_err(|error| io("read", &spv, error))
}

/// Check `spv` with `spirv-val`, in a scratch directory named for `label`.
pub fn validate(spv: &[u8], label: &str) -> Result<(), ToolError> {
    let scratch = ScratchDir::new(label).map_err(ToolError::Failed)?;
    let path = scratch.path().join("module.spv");
    std::fs::write(&path, spv).map_err(|error| io("write", &path, error))?;
    run(
        "spirv-val",
        &[
            OsStr::new("--target-env"),
            OsStr::new(VULKAN_TARGET_ENV),
            path.as_os_str(),
        ],
    )
}

/// [`assemble`], then [`validate`] the result.
pub fn assemble_validated(assembly: &str, label: &str) -> Result<Vec<u8>, ToolError> {
    let bytes = assemble(assembly, label)?;
    validate(&bytes, label)?;
    Ok(bytes)
}
