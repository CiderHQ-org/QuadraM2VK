//! Deadlines on the Metal oracle's compilations.
//!
//! The oracle compiles in two places, each bounded by [`compile_timeout()`] (60 s unless a
//! command overrides it with [`COMPILE_TIMEOUT_FLAG`]):
//!
//! - **Host tools.** `metal`, `metal-as`, `metal-opt`, and `metallib` run through `xcrun` here,
//!   each in its own process group (`worker::process`), killed with the group past the deadline.
//!   The caller gets `ToolError::Stopped` and the case fails. Each runs for the deployment target
//!   [`ORACLE_DEPLOYMENT_TARGET`], whatever the host's macOS release.
//! - **Framework compilations.** Metal compiles AIR to GPU code when a pipeline state is created
//!   (`newComputePipelineState*`, `newRenderPipelineState*`), in its compiler service; it may
//!   compile when it specializes a function (`newFunctionWithName:constantValues:`), and it
//!   compiles the oracle's own MSL helper libraries (`newLibraryWithSource:`). None of these
//!   synchronous calls can be cancelled or given a timeout. Q2 met a fuel-instrumented FFT
//!   pipeline whose creation ran past ten minutes and 2.3 GB, which stalls a batch that runs one
//!   case per process. Every such call of the oracle therefore runs through [`bounded`]: a
//!   watchdog thread waits for it at most the deadline. A source test fails if one outside it
//!   appears in the crate.
//!
//! A compile is CPU work, so nothing is in flight on the GPU when it runs long, and no tripwire
//! is written. The only way out of a framework compilation is to leave the process: past the
//! deadline the watchdog prints a record and the process ends through
//! `_exit(`[`COMPILE_TIMEOUT_EXIT_CODE`]`)` (distinct from the hang exit, 86, which means a GPU
//! wait timed out). The compiler service runs on behalf of this process; its request is expected
//! to end with it, which is not verified here.

use super::wait::current_label;
use crate::worker::process::{MonitoredChild, ProcessControl, ProcessFailure};
use crate::ScratchDir;
use std::ffi::OsStr;
use std::fs;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// Default bound on one compilation: one host tool run or one framework compilation.
pub const DEFAULT_COMPILE_TIMEOUT: Duration = Duration::from_secs(60);

/// Largest bound a command may ask for.
pub const MAX_COMPILE_TIMEOUT: Duration = Duration::from_secs(600);

/// Command-line flag with which Metal-executing binaries override [`DEFAULT_COMPILE_TIMEOUT`].
pub const COMPILE_TIMEOUT_FLAG: &str = "--compile-timeout";

/// Exit status of a process a framework compilation outlasted.
pub const COMPILE_TIMEOUT_EXIT_CODE: i32 = 87;

/// The environment variable through which Apple's Metal tools take the macOS release they compile
/// for, their deployment target.
pub const DEPLOYMENT_TARGET_VAR: &str = "MACOSX_DEPLOYMENT_TARGET";

/// The deployment target every host tool of the oracle runs for. Unset, the tools take it from the
/// host, and the AIR they build changes with it: `metal` on the reference Mac (macOS 26.7, SDK
/// 27.0) writes `air64_v28-apple-macosx26.0.0`, and for macOS 27 it writes
/// `air64_v29-apple-macosx27.0.0`, whose atomic intrinsics take one more operand, so a case
/// would compile to other AIR, and verify against another intrinsic table, on another machine.
/// 26.0 is the target the reference device's tools take today: `metal` builds byte-identical AIR
/// under it and unset there.
pub const ORACLE_DEPLOYMENT_TARGET: &str = "26.0";

static COMPILE_TIMEOUT_MS: AtomicU64 = AtomicU64::new(DEFAULT_COMPILE_TIMEOUT.as_millis() as u64);

/// The bound every compilation in this process uses.
pub fn compile_timeout() -> Duration {
    Duration::from_millis(COMPILE_TIMEOUT_MS.load(Ordering::Relaxed))
}

/// Parse a [`COMPILE_TIMEOUT_FLAG`] value (whole seconds, 1 ..= 600) and make it this process's
/// bound. Commands call it once, before compiling anything.
pub fn set_compile_timeout(value: &str) -> Result<(), String> {
    let seconds = value
        .parse::<u64>()
        .map_err(|error| format!("{COMPILE_TIMEOUT_FLAG} {value:?}: {error}"))?;
    if seconds == 0 || Duration::from_secs(seconds) > MAX_COMPILE_TIMEOUT {
        return Err(format!(
            "{COMPILE_TIMEOUT_FLAG} {value:?}: expected whole seconds in 1..={}",
            MAX_COMPILE_TIMEOUT.as_secs()
        ));
    }
    COMPILE_TIMEOUT_MS.store(seconds * 1000, Ordering::Relaxed);
    Ok(())
}

/// A framework compilation at least this long reports how long it took when it returns
/// (`m2v-compile: ... took`), so the bound can be set from what the slow ones take.
pub const REPORTED_COMPILE: Duration = Duration::from_secs(10);

/// Run the compilation `body` (named `operation` in the record) under [`compile_timeout()`]. It
/// returns `body`'s result; a compilation that outlasts the bound does not return, because the
/// process ends (module documentation). One that took at least [`REPORTED_COMPILE`] prints its
/// time.
pub fn bounded<R>(operation: &str, body: impl FnOnce() -> R) -> R {
    let started = Instant::now();
    let watch = Watch::arm(expired(operation), compile_timeout(), leave);
    let result = body();
    drop(watch);
    let took = started.elapsed();
    if took >= REPORTED_COMPILE {
        eprintln!("m2v-compile: {}", Took(expired(operation), took));
    }
    result
}

/// A slow compilation's report: what [`Expired`] names, and how long it took.
struct Took(Expired, Duration);

impl std::fmt::Display for Took {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Took(compilation, took) = self;
        write!(
            f,
            "{} took {:.1} s of its {:?} bound (case {}, module {})",
            compilation.operation,
            took.as_secs_f64(),
            compilation.timeout,
            compilation.case_id.as_deref().unwrap_or("-"),
            compilation.module_sha256.as_deref().unwrap_or("-"),
        )
    }
}

/// What a host tool printed, and whether it exited successfully.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct ToolOutput {
    pub(crate) success: bool,
    pub(crate) stdout: String,
    pub(crate) stderr: String,
}

/// Why a host tool produced no [`ToolOutput`].
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum ToolError {
    /// It could not be started, or its output could not be read.
    Unavailable(String),
    /// Its process group was killed: it outlasted the deadline, or supervising it failed.
    Stopped(String),
}

impl std::fmt::Display for ToolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ToolError::Unavailable(why) | ToolError::Stopped(why) => f.write_str(why),
        }
    }
}

/// `xcrun --sdk macosx`, for [`ORACLE_DEPLOYMENT_TARGET`]: the one place this module spawns
/// `xcrun`, so no host tool runs for another target.
fn xcrun_command() -> Command {
    let mut command = Command::new("xcrun");
    pin_deployment_target(&mut command).args(["--sdk", "macosx"]);
    command
}

/// Set `command`'s deployment target to [`ORACLE_DEPLOYMENT_TARGET`], replacing whatever the
/// environment this process inherited says.
fn pin_deployment_target(command: &mut Command) -> &mut Command {
    command.env(DEPLOYMENT_TARGET_VAR, ORACLE_DEPLOYMENT_TARGET)
}

/// Run `xcrun --sdk macosx <tool> <args>` under [`compile_timeout()`] (module documentation).
pub(crate) fn xcrun(
    tool: &str,
    args: impl IntoIterator<Item = impl AsRef<OsStr>>,
) -> Result<ToolOutput, ToolError> {
    let mut command = xcrun_command();
    command.arg(tool).args(args);
    run_tool(command, tool, compile_timeout())
}

/// Run the host program `program` with `args` under [`compile_timeout()`], as [`xcrun`] runs a
/// tool: the oracle reads its host's identity with it (`sw_vers`, `xcode-select`, `ioreg`), on
/// macOS alone, as its caller (`metal::platform`) is.
#[cfg(target_os = "macos")]
pub(crate) fn host_tool(program: &str, args: &[&str]) -> Result<ToolOutput, ToolError> {
    let mut command = Command::new(program);
    command.args(args);
    run_tool(command, program, compile_timeout())
}

/// Whether `xcrun` finds `tool` in the macOS SDK (`xcrun --sdk macosx -f <tool>`), under the
/// compile deadline: tests that need a host tool skip visibly without it.
#[cfg(test)]
pub(crate) fn has_tool(tool: &str) -> bool {
    let mut command = xcrun_command();
    command.args(["-f", tool]);
    run_tool(command, tool, compile_timeout()).is_ok_and(|output| output.success)
}

/// Run `command` (named `tool` in errors) in its own process group, killing the group past
/// `timeout`. Its output goes to files in a scratch directory removed on return, so a tool that
/// prints a lot cannot block on a full pipe.
fn run_tool(mut command: Command, tool: &str, timeout: Duration) -> Result<ToolOutput, ToolError> {
    let deadline = Instant::now() + timeout;
    let unavailable = |why: String| ToolError::Unavailable(format!("{tool}: {why}"));
    let scratch = ScratchDir::new("host-tool").map_err(unavailable)?;
    let stdout = scratch.path().join("stdout");
    let stderr = scratch.path().join("stderr");
    let create = |path: &Path| {
        fs::File::create(path)
            .map_err(|error| unavailable(format!("create {}: {error}", path.display())))
    };
    command
        .stdin(Stdio::null())
        .stdout(create(&stdout)?)
        .stderr(create(&stderr)?);
    let child = MonitoredChild::spawn(&mut command)
        .map_err(|error| unavailable(format!("spawn: {error}")))?;
    let control = ProcessControl {
        deadline: Some(deadline),
        memory_limit_bytes: None,
        cancelled: None,
    };
    let exit = child.wait(&control).map_err(|stop| match stop.failure {
        ProcessFailure::DeadlineExceeded => ToolError::Stopped(format!(
            "{tool} did not finish within {timeout:?}; its process group was killed"
        )),
        failure => ToolError::Stopped(format!("{tool}: {failure}")),
    })?;
    let read = |path: &Path| {
        fs::read(path)
            .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
            .map_err(|error| unavailable(format!("read {}: {error}", path.display())))
    };
    Ok(ToolOutput {
        success: exit.status.success(),
        stdout: read(&stdout)?,
        stderr: read(&stderr)?,
    })
}

/// What an expired compilation reports.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Expired {
    pub(crate) operation: String,
    pub(crate) case_id: Option<String>,
    pub(crate) module_sha256: Option<String>,
    pub(crate) timeout: Duration,
}

impl std::fmt::Display for Expired {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} did not finish within {:?} (case {}, module {}); the process exits",
            self.operation,
            self.timeout,
            self.case_id.as_deref().unwrap_or("-"),
            self.module_sha256.as_deref().unwrap_or("-"),
        )
    }
}

fn expired(operation: &str) -> Expired {
    let label = current_label();
    Expired {
        operation: operation.to_string(),
        case_id: label.as_ref().map(|label| label.case_id.clone()),
        module_sha256: label.map(|label| label.module_sha256),
        timeout: compile_timeout(),
    }
}

fn leave(expired: Expired) {
    eprintln!("m2v-compile: {expired}");
    // SAFETY: `_exit` ends the process at once. The compiling thread is blocked inside the
    // framework call; nothing on the GPU is in flight, and no destructor needs to run.
    unsafe { libc::_exit(COMPILE_TIMEOUT_EXIT_CODE) }
}

/// A watchdog for one compilation: disarmed (and joined) when dropped, also on unwinding.
struct Watch {
    done: Arc<(Mutex<bool>, Condvar)>,
    thread: Option<JoinHandle<()>>,
}

impl Watch {
    fn arm(expired: Expired, timeout: Duration, on_expiry: fn(Expired)) -> Watch {
        let done = Arc::new((Mutex::new(false), Condvar::new()));
        let watched = Arc::clone(&done);
        let unwatched = format!("{} (no watchdog thread)", expired.operation);
        let thread = std::thread::Builder::new()
            .name("m2v-compile-watch".into())
            .spawn(move || {
                let (lock, condvar) = &*watched;
                let guard = lock.lock().unwrap_or_else(PoisonError::into_inner);
                let (guard, _) = condvar
                    .wait_timeout_while(guard, timeout, |done| !*done)
                    .unwrap_or_else(PoisonError::into_inner);
                if !*guard {
                    drop(guard);
                    on_expiry(expired);
                }
            })
            .ok();
        if thread.is_none() {
            // Without a watchdog thread the compilation is unbounded: refuse to start it.
            on_expiry(Expired {
                operation: unwatched,
                case_id: None,
                module_sha256: None,
                timeout,
            });
        }
        Watch { done, thread }
    }
}

impl Drop for Watch {
    fn drop(&mut self) {
        let (lock, condvar) = &*self.done;
        *lock.lock().unwrap_or_else(PoisonError::into_inner) = true;
        condvar.notify_all();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;

    static EXPIRED: StdMutex<Vec<Expired>> = StdMutex::new(Vec::new());

    fn record(expired: Expired) {
        EXPIRED.lock().unwrap().push(expired);
    }

    fn fake(operation: &str) -> Expired {
        Expired {
            operation: operation.into(),
            case_id: Some("case".into()),
            module_sha256: None,
            timeout: Duration::from_millis(20),
        }
    }

    #[test]
    fn a_compilation_past_its_deadline_expires_and_one_within_it_does_not() {
        {
            let _watch = Watch::arm(fake("slow"), Duration::from_millis(20), record);
            std::thread::sleep(Duration::from_millis(300));
        }
        {
            let _watch = Watch::arm(fake("fast"), Duration::from_secs(60), record);
        }
        let expired = EXPIRED.lock().unwrap().clone();
        assert_eq!(expired, vec![fake("slow")]);
        assert!(expired[0]
            .to_string()
            .contains("slow did not finish within 20ms"));
    }

    fn shell(script: &str) -> Command {
        let mut command = Command::new("sh");
        command.args(["-c", script]);
        command
    }

    /// The pinned deployment target replaces an inherited one, as a macOS 27 host's would be.
    #[test]
    fn a_host_tool_runs_for_the_pinned_deployment_target_whatever_it_inherits() {
        let mut command = shell(&format!("printf %s \"${DEPLOYMENT_TARGET_VAR}\""));
        command.env(DEPLOYMENT_TARGET_VAR, "27.0");
        pin_deployment_target(&mut command);
        let output = run_tool(command, "sh", Duration::from_secs(60)).unwrap();
        assert_eq!(output.stdout, ORACLE_DEPLOYMENT_TARGET);
        // The one xcrun command this module builds carries the pin.
        let pinned = xcrun_command().get_envs().any(|(key, value)| {
            key == DEPLOYMENT_TARGET_VAR && value == Some(OsStr::new(ORACLE_DEPLOYMENT_TARGET))
        });
        assert!(pinned);
        let source = include_str!("compile.rs");
        let runner = &source[..source.find("\nmod tests {").unwrap()];
        let spawn = concat!("Command::new(\"", "xcrun\")");
        assert_eq!(
            runner.matches(spawn).count(),
            1,
            "one xcrun command, the pinned one"
        );
    }

    /// Apple's compiler follows the deployment target: unpinned under macOS 27's it writes AIR
    /// for macOS 27, and pinned it writes the reference's AIR, byte for byte the same whatever
    /// the inherited target. A linked public fixture prepared as the oracle prepares it
    /// (`libraries::shader_air`) is covered in `metal::platform::libraries`.
    #[test]
    fn the_pin_decides_the_air_metal_writes() {
        if !cfg!(target_os = "macos") || !has_tool("metal") {
            eprintln!("SKIP the_pin_decides_the_air_metal_writes: xcrun metal not found");
            return;
        }
        let scratch = ScratchDir::new("deployment-target").unwrap();
        let source = scratch.path().join("k.metal");
        fs::write(
            &source,
            "#include <metal_stdlib>\nusing namespace metal;\n\
             kernel void k(device atomic_uint* o [[buffer(0)]], uint i [[thread_position_in_grid]]) {\n\
             atomic_fetch_xor_explicit(o, i, memory_order_relaxed);\n}\n",
        )
        .unwrap();
        let compile = |inherited: Option<&str>, pin: bool, name: &str| {
            let out = scratch.path().join(name);
            let mut command = Command::new("xcrun");
            if let Some(target) = inherited {
                command.env(DEPLOYMENT_TARGET_VAR, target);
            } else {
                command.env_remove(DEPLOYMENT_TARGET_VAR);
            }
            if pin {
                pin_deployment_target(&mut command);
            }
            command.args(["--sdk", "macosx", "metal", "-S", "-emit-llvm", "-c"]);
            command.arg(&source).arg("-o").arg(&out);
            let output = run_tool(command, "metal", compile_timeout()).unwrap();
            assert!(output.success, "{}", output.stderr);
            fs::read_to_string(&out).unwrap()
        };
        let triple = |text: &str| {
            text.lines()
                .find(|line| line.starts_with("target triple"))
                .unwrap_or_default()
                .to_string()
        };
        let unpinned_27 = compile(Some("27.0"), false, "unpinned-27.ll");
        assert!(
            triple(&unpinned_27).contains("-apple-macosx27.0.0"),
            "{}",
            triple(&unpinned_27)
        );
        let pinned_27 = compile(Some("27.0"), true, "pinned-27.ll");
        let pinned = compile(None, true, "pinned.ll");
        assert!(triple(&pinned).contains(&format!("-apple-macosx{ORACLE_DEPLOYMENT_TARGET}.0")));
        let body = |text: &str| {
            text.lines()
                .filter(|line| {
                    !line.starts_with("; ModuleID") && !line.starts_with("source_filename")
                })
                .collect::<Vec<_>>()
                .join("\n")
        };
        assert_eq!(body(&pinned_27), body(&pinned));
        assert_ne!(body(&unpinned_27), body(&pinned));
    }

    /// A host tool's output and status come back; one past its deadline is killed, with its
    /// process group, well before it would have finished.
    #[test]
    fn a_host_tool_reports_its_output_and_is_killed_past_its_deadline() {
        let output = run_tool(
            shell("echo out; echo err >&2; exit 3"),
            "sh",
            Duration::from_secs(60),
        )
        .unwrap();
        assert_eq!(
            output,
            ToolOutput {
                success: false,
                stdout: "out\n".into(),
                stderr: "err\n".into(),
            }
        );
        let started = Instant::now();
        let stopped = run_tool(
            shell("sleep 30 & sleep 30; wait"),
            "sh",
            Duration::from_millis(200),
        );
        assert!(
            matches!(&stopped, Err(ToolError::Stopped(why)) if why.contains("did not finish")),
            "{stopped:?}"
        );
        assert!(started.elapsed() < Duration::from_secs(20));
    }

    /// A slow compilation's report names it, its time, its bound, and its case.
    #[test]
    fn a_slow_compilation_reports_its_time_and_bound() {
        let report = Took(
            fake("newComputePipelineState"),
            Duration::from_millis(63_400),
        );
        assert_eq!(
            report.to_string(),
            "newComputePipelineState took 63.4 s of its 20ms bound (case case, module -)"
        );
    }

    #[test]
    fn the_compile_timeout_flag_takes_whole_seconds_in_range() {
        assert!(set_compile_timeout("0").is_err());
        assert!(set_compile_timeout("601").is_err());
        assert!(set_compile_timeout("ten").is_err());
        assert_eq!(compile_timeout(), DEFAULT_COMPILE_TIMEOUT);
    }
}
