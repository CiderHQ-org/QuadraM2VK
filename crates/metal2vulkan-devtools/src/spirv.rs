//! `spirv-val` and `spirv-dis` on a module's words: the L3 checks the product's tests and
//! validation run. What a module is checked under, its target environments and the layout
//! relaxations its device enables, is the product's to say (`DeviceProfile::spirv_validation`);
//! this module only runs the tools.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::tools::{run, ToolError};

/// Why L3 did not accept a module: the module's fault (`Requirements`, `Rejected`) or the
/// validator's (`Tooling`: `spirv-val` could not be launched, its scratch file could not be
/// written, or it died without an exit code). Callers that classify outcomes must keep tooling
/// failures apart from invalid modules.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum L3Failure {
    /// The module's requirements exceed the profile.
    Requirements(String),
    /// `spirv-val` ran and rejected the module.
    Rejected(String),
    /// `spirv-val` itself failed.
    Tooling(String),
}

impl L3Failure {
    /// The failure's text alone: the validator's diagnostic, the launch failure, or why the
    /// requirements exceed the profile.
    pub fn into_text(self) -> String {
        match self {
            L3Failure::Requirements(text)
            | L3Failure::Rejected(text)
            | L3Failure::Tooling(text) => text,
        }
    }
}

impl std::fmt::Display for L3Failure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            L3Failure::Requirements(text) | L3Failure::Rejected(text) => f.write_str(text),
            L3Failure::Tooling(text) => write!(f, "spirv-val tooling: {text}"),
        }
    }
}

/// A scratch file for `words`, named for the process and a counter of `next`; the caller removes
/// it once the tool has run.
fn scratch(prefix: &str, next: &AtomicU64, words: &[u32]) -> Result<PathBuf, String> {
    let path = std::env::temp_dir().join(format!(
        "{prefix}_{}_{}.spv",
        std::process::id(),
        next.fetch_add(1, Ordering::Relaxed)
    ));
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    std::fs::write(&path, bytes).map_err(|e| format!("write {}: {e}", path.display()))?;
    Ok(path)
}

/// The `spirv-dis` disassembly of `words`, or the reason there is none (diagnostics only).
pub fn spirv_dis(words: &[u32]) -> String {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let path = match scratch("m2v_spirv_dis", &NEXT, words) {
        Ok(path) => path,
        Err(e) => return format!("<{e}>"),
    };
    let out = spirv_dis_file(&path);
    let _ = std::fs::remove_file(&path);
    out
}

/// [`spirv_dis`] of the module in the file `path`.
pub fn spirv_dis_file(path: &Path) -> String {
    match run("spirv-dis", &[path.as_os_str()]) {
        Ok((stdout, _)) => String::from_utf8_lossy(&stdout).into_owned(),
        Err(e) => format!("<spirv-dis: {e}>"),
    }
}

/// `spirv-val --target-env <env>` with the relaxation arguments `relaxations` on `words`.
pub fn spirv_val(words: &[u32], env: &str, relaxations: &[&str]) -> Result<(), L3Failure> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let path = scratch("m2v_spirv_val", &NEXT, words).map_err(L3Failure::Tooling)?;
    let result = spirv_val_file(&path, env, relaxations);
    // The validator only needs the file for the subprocess lifetime.
    let _ = std::fs::remove_file(&path);
    result
}

/// [`spirv_val`] on the module in the file `path`.
pub fn spirv_val_file(path: &Path, env: &str, relaxations: &[&str]) -> Result<(), L3Failure> {
    let mut args: Vec<std::ffi::OsString> = vec!["--target-env".into(), env.into()];
    args.extend(relaxations.iter().map(Into::into));
    args.push(path.as_os_str().to_owned());
    run("spirv-val", &args)
        .map(|_| ())
        .map_err(l3_of_tool_error)
}

/// `spirv-val` on linked `words` under each of `envs` with `relaxations`: a rejection is
/// reported with the environment, `options` (the relaxations as the product's plan describes
/// them), `requirements` (the module's, described), and the disassembly.
pub fn check_l3(
    words: &[u32],
    envs: &[&str],
    relaxations: &[&str],
    options: &str,
    requirements: &str,
) -> Result<(), L3Failure> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let path = scratch("m2v_spirv_val", &NEXT, words).map_err(L3Failure::Tooling)?;
    let result = check_l3_file(&path, envs, relaxations, options, requirements);
    let _ = std::fs::remove_file(&path);
    result
}

/// [`check_l3`] on the module in the file `path`, which is read only by the tools: a caller
/// that holds nothing of the module while `spirv-val` runs (a worker's L3 stage) keeps it so.
pub fn check_l3_file(
    path: &Path,
    envs: &[&str],
    relaxations: &[&str],
    options: &str,
    requirements: &str,
) -> Result<(), L3Failure> {
    for env in envs {
        spirv_val_file(path, env, relaxations).map_err(|failure| match failure {
            L3Failure::Rejected(e) => L3Failure::Rejected(format!(
                "spirv-val {env} {options}: {e}\nrequirements: {requirements}\n{}",
                spirv_dis_file(path)
            )),
            other => other,
        })?;
    }
    Ok(())
}

/// An exit code is the validator's verdict on the module; a failed launch or death by signal is
/// the validator's own failure.
fn l3_of_tool_error(error: ToolError) -> L3Failure {
    match error {
        ToolError::Exit { output, .. } if output.status.code().is_some() => {
            L3Failure::Rejected(format!(
                "{}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            ))
        }
        other => L3Failure::Tooling(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validator_failures_are_typed_apart_from_rejections() {
        let launch = ToolError::Launch {
            tool: "spirv-val".into(),
            executable: PathBuf::from("/nonexistent/spirv-val"),
            source: std::io::Error::from(std::io::ErrorKind::NotFound),
        };
        assert!(matches!(l3_of_tool_error(launch), L3Failure::Tooling(_)));
    }

    #[cfg(unix)]
    #[test]
    fn an_exit_code_is_a_rejection_and_a_signal_is_tooling() {
        use std::os::unix::process::ExitStatusExt;
        let exit = |status| ToolError::Exit {
            tool: "spirv-val".into(),
            output: std::process::Output {
                status: std::process::ExitStatus::from_raw(status),
                stdout: b"error: bad id".to_vec(),
                stderr: Vec::new(),
            },
        };
        // Raw wait statuses: exit code 1 is `1 << 8`; SIGKILL is 9.
        match l3_of_tool_error(exit(1 << 8)) {
            L3Failure::Rejected(text) => assert!(text.contains("bad id"), "{text}"),
            other => panic!("{other:?}"),
        }
        assert!(matches!(l3_of_tool_error(exit(9)), L3Failure::Tooling(_)));
    }
}
