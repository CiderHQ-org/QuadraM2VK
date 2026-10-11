//! The host-wide GPU hang tripwire.
//!
//! A GPU submission that times out leaves `~/.cache/m2v-gpu/hang` behind. While that file, or
//! the legacy `/tmp/m2v-gpu.hang`, exists no GPU entry point in this crate starts new work: the
//! device may still be executing the stuck submission, and only a human who has followed the
//! suspected-hang protocol in `docs/VALIDATION.md` may remove it. The writer is the hang protocol
//! in [`super::wait`].
//!
//! The tripwire lives under the home directory because a peer session on this host has deleted
//! `/tmp` entries. The legacy path is still read, so a tripwire written before the move keeps
//! refusing GPU work.

use super::wait::SubmissionBackend;
use serde::{Deserialize, Serialize};
use std::ffi::OsStr;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// The per-user GPU state directory, relative to `$HOME`. It holds the tripwire (`hang`).
pub const STATE_DIR: &str = ".cache/m2v-gpu";

/// The tripwire location before it moved under [`STATE_DIR`]. Still read; written only when the
/// current location cannot be.
pub const LEGACY_TRIPWIRE_PATH: &str = "/tmp/m2v-gpu.hang";

/// Where the host-wide tripwire is written and read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TripwirePaths {
    /// `$HOME/.cache/m2v-gpu/hang`: written by the hang protocol, read by every GPU entry point.
    pub current: PathBuf,
    /// [`LEGACY_TRIPWIRE_PATH`]: read by every GPU entry point.
    pub legacy: PathBuf,
}

impl TripwirePaths {
    /// The paths for this process's `$HOME`.
    pub fn for_host() -> Result<Self, String> {
        Self::for_home(std::env::var_os("HOME").as_deref())
    }

    /// The paths for a `$HOME` value. `HOME` must be an absolute path, so every process resolves
    /// the same file.
    pub fn for_home(home: Option<&OsStr>) -> Result<Self, String> {
        let home = home
            .map(Path::new)
            .filter(|home| home.is_absolute())
            .ok_or_else(|| {
                "HOME must be an absolute path: the GPU hang tripwire lives under it; GPU work \
                 is refused"
                    .to_string()
            })?;
        Ok(Self {
            current: home.join(STATE_DIR).join("hang"),
            legacy: PathBuf::from(LEGACY_TRIPWIRE_PATH),
        })
    }

    /// Refuse GPU work while either tripwire exists.
    pub fn refuse_if_tripped(&self) -> Result<(), String> {
        refuse_if_present(&self.current)?;
        refuse_if_present(&self.legacy)
    }
}

/// Refuse GPU work while the host-wide tripwire (current or legacy) exists, or while its location
/// cannot be resolved.
pub fn refuse_if_tripped() -> Result<(), String> {
    TripwirePaths::for_host()?.refuse_if_tripped()
}

/// Write `record` as the host-wide tripwire. When `$HOME` cannot be resolved or the current
/// location cannot be written, the legacy location is used instead: every GPU entry point reads
/// it too, and any tripwire is better than none.
pub fn write_host_record(record: &HangRecord) -> Result<(), String> {
    match TripwirePaths::for_host().and_then(|paths| write_record(&paths.current, record)) {
        Ok(()) => Ok(()),
        Err(error) => write_record(Path::new(LEGACY_TRIPWIRE_PATH), record)
            .map_err(|legacy| format!("{error}; {legacy}")),
    }
}

/// Refuse GPU work while `path` exists. A path that cannot be inspected counts as tripped: an
/// unknown tripwire state is not permission to submit.
pub(crate) fn refuse_if_present(path: &Path) -> Result<(), String> {
    match path.try_exists() {
        Ok(false) => Ok(()),
        Ok(true) => Err(format!(
            "GPU hang tripwire {} exists; GPU work is refused until the suspected-hang protocol \
             in docs/VALIDATION.md is complete and a human removes it",
            path.display()
        )),
        Err(error) => Err(format!(
            "cannot inspect GPU hang tripwire {}: {error}; GPU work is refused",
            path.display()
        )),
    }
}

/// Why a tripwire was written.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HangKind {
    /// A bounded wait elapsed; the submission may still be running.
    Timeout,
    /// The submission ended with a device-fault status.
    SuspectedFault,
    /// The wait failed without showing whether the submission finished; it may still be running.
    Unresolved,
    /// The process that held the device died by a signal: whatever it had submitted may still be
    /// running.
    Crashed,
}

/// The JSON body of the tripwire: what was running, where, and when.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HangRecord {
    pub kind: HangKind,
    pub pid: u32,
    /// The process command line.
    pub command: Vec<String>,
    pub case_id: Option<String>,
    /// SHA-256 of the executed module (SPIR-V for Vulkan, AIR source for Metal).
    pub module_sha256: Option<String>,
    pub backend: SubmissionBackend,
    /// The submission that was waited on.
    pub operation: String,
    pub detail: String,
    pub timeout_ms: u64,
    pub unix_time_ms: u64,
}

impl HangRecord {
    /// A record for this process, stamped with the current time.
    pub fn now(
        kind: HangKind,
        backend: SubmissionBackend,
        operation: &str,
        detail: String,
        timeout: Duration,
        case_id: Option<String>,
        module_sha256: Option<String>,
    ) -> Self {
        let unix_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();
        Self {
            kind,
            pid: std::process::id(),
            command: std::env::args_os()
                .map(|arg| arg.to_string_lossy().into_owned())
                .collect(),
            case_id,
            module_sha256,
            backend,
            operation: operation.to_string(),
            detail,
            timeout_ms: u64::try_from(timeout.as_millis()).unwrap_or(u64::MAX),
            unix_time_ms: u64::try_from(unix_time.as_millis()).unwrap_or(u64::MAX),
        }
    }

    /// One line for logs and error messages.
    pub fn summary(&self) -> String {
        let what = match self.kind {
            HangKind::Timeout => "timed out",
            HangKind::SuspectedFault => "ended with a suspected device fault",
            HangKind::Unresolved => "could not be confirmed finished",
            HangKind::Crashed => "lost its process to a signal",
        };
        format!(
            "{} {what} ({}) on {:?}, case {}, module {}",
            self.operation,
            self.detail,
            self.backend,
            self.case_id.as_deref().unwrap_or("unknown"),
            self.module_sha256.as_deref().unwrap_or("unknown"),
        )
    }
}

/// Write `record` as the tripwire at `path`, creating its directory. An existing tripwire is
/// kept: the first incident is the one the protocol must explain, and the file's presence
/// already refuses GPU work.
pub fn write_record(path: &Path, record: &HangRecord) -> Result<(), String> {
    let json = serde_json::to_vec_pretty(record)
        .map_err(|error| format!("encode GPU hang tripwire: {error}"))?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|error| {
            format!(
                "create GPU hang tripwire directory {}: {error}",
                parent.display()
            )
        })?;
    }
    let mut file = match std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
    {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => return Ok(()),
        Err(error) => {
            return Err(format!(
                "create GPU hang tripwire {}: {error}",
                path.display()
            ))
        }
    };
    file.write_all(&json)
        .and_then(|()| file.write_all(b"\n"))
        .and_then(|()| file.sync_all())
        .map_err(|error| format!("write GPU hang tripwire {}: {error}", path.display()))
}

/// Read the record a tripwire holds.
pub fn read_record(path: &Path) -> Result<HangRecord, String> {
    let bytes = std::fs::read(path)
        .map_err(|error| format!("read GPU hang tripwire {}: {error}", path.display()))?;
    serde_json::from_slice(&bytes)
        .map_err(|error| format!("parse GPU hang tripwire {}: {error}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tripwire_refuses_only_while_present() {
        let scratch = crate::ScratchDir::new("gpu-tripwire").unwrap();
        let path = scratch.path().join("m2v-gpu.hang");
        refuse_if_present(&path).unwrap();
        std::fs::write(&path, "{}").unwrap();
        let error = refuse_if_present(&path).unwrap_err();
        assert!(error.contains("tripwire"), "{error}");
        std::fs::remove_file(&path).unwrap();
        refuse_if_present(&path).unwrap();
    }

    #[test]
    fn tripwire_paths_follow_home_and_keep_the_legacy_location() {
        let paths = TripwirePaths::for_home(Some(OsStr::new("/home/someone"))).unwrap();
        assert_eq!(
            paths.current,
            Path::new("/home/someone/.cache/m2v-gpu/hang")
        );
        assert_eq!(paths.legacy, Path::new("/tmp/m2v-gpu.hang"));
        for home in [None, Some(""), Some("relative/home")] {
            let error = TripwirePaths::for_home(home.map(OsStr::new)).unwrap_err();
            assert!(error.contains("HOME"), "{error}");
        }
    }

    #[test]
    fn either_tripwire_refuses() {
        let scratch = crate::ScratchDir::new("gpu-tripwire-pair").unwrap();
        let paths = TripwirePaths {
            current: scratch.path().join("m2v-gpu").join("hang"),
            legacy: scratch.path().join("legacy.hang"),
        };
        paths.refuse_if_tripped().unwrap();
        for path in [&paths.current, &paths.legacy] {
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, "{}").unwrap();
            let error = paths.refuse_if_tripped().unwrap_err();
            assert!(error.contains(&path.display().to_string()), "{error}");
            std::fs::remove_file(path).unwrap();
            paths.refuse_if_tripped().unwrap();
        }
    }

    #[test]
    fn hang_record_round_trips_and_the_first_incident_is_kept() {
        let scratch = crate::ScratchDir::new("gpu-tripwire-record").unwrap();
        // The state directory does not exist yet: the writer creates it.
        let path = scratch.path().join("m2v-gpu").join("hang");
        let first = HangRecord::now(
            HangKind::Timeout,
            SubmissionBackend::VulkanGpu,
            "Vulkan candidate submission",
            "no completion within 10s".into(),
            Duration::from_secs(10),
            Some("case".into()),
            Some("00".repeat(32)),
        );
        write_record(&path, &first).unwrap();
        assert_eq!(read_record(&path).unwrap(), first);
        let text = std::fs::read_to_string(&path).unwrap();
        for key in [
            "\"pid\"",
            "\"command\"",
            "\"case_id\"",
            "\"module_sha256\"",
            "\"backend\": \"vulkan_gpu\"",
            "\"kind\": \"timeout\"",
            "\"unix_time_ms\"",
        ] {
            assert!(text.contains(key), "{key} missing from {text}");
        }
        let second = HangRecord::now(
            HangKind::SuspectedFault,
            SubmissionBackend::Metal,
            "Metal compute dispatch",
            "page fault".into(),
            Duration::from_secs(10),
            None,
            None,
        );
        write_record(&path, &second).unwrap();
        assert_eq!(read_record(&path).unwrap(), first);
        assert!(refuse_if_present(&path).is_err());
    }
}
