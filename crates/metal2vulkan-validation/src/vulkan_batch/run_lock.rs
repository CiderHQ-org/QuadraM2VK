//! The host's GPU run lock. A GPU batch holds it around each run: the run's start, its submissions
//! and their bounded waits, and its end, the hang protocol's tripwire included. So batches running
//! at once on one host (a Metal lane beside a MoltenVK lane, or two agents' lanes) put one run on
//! the GPU at a time. A hang then has exactly one run in flight to name, and no batch's run can
//! wait behind another batch's stuck submission: a batch waiting for the lock has not started its
//! run, so no device wait of its own is running. The lock is the host's, beside its tripwire
//! (`~/.cache/m2v-gpu/run.lock`), and is held through `flock`, which the kernel releases when the
//! holder's file is closed, its process killed included. A CPU batch takes no lock.
//!
//! Batches take the lock in turn: a batch waiting for it first takes the turnstile beside it
//! (`run.turn`) and holds that while it waits, so a holder coming back for its next run waits
//! behind it instead of taking the lock again at once. Two batches alternate run by run.
//!
//! While it waits, a batch looks at the tripwire: one the holder's run left (written before the
//! holder lets go) stops it before it starts anything. The holder records its process, its run,
//! and the latest time its run can end (the run's timeout and the session's teardown after its
//! start); a holder that keeps the lock well past that time stops the waiting batch with an error
//! naming it, so no wait for the lock is unbounded.

use super::Run;
use crate::gpu::tripwire::TripwirePaths;
use std::fs::{self, File, OpenOptions};
use std::io::{Seek as _, SeekFrom, Write as _};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Beyond the run's own timeout: the session's teardown and the run's recording.
const TEARDOWN: Duration = Duration::from_secs(30);

/// How long a waiting batch lets the holder keep the lock past the end it recorded.
const GRACE: Duration = Duration::from_secs(60);

/// How often a waiting batch tries the turnstile or the lock again and looks at the tripwire.
const POLL: Duration = Duration::from_millis(20);

/// The run lock of the host whose tripwire is `tripwire`: the file `run.lock` beside it.
pub(crate) fn lock_path(tripwire: &TripwirePaths) -> PathBuf {
    tripwire.current.with_file_name("run.lock")
}

/// The held lock. Dropping it clears the holder's record, then releases the lock.
pub(crate) struct RunLock {
    file: File,
}

impl RunLock {
    /// Hold the run lock of the host whose tripwire is `tripwire` for `run`, which ends within
    /// `timeout` (the batch's per-run timeout): take the turnstile, then the lock, then let the
    /// turnstile go. While another batch holds either, wait, and fail as soon as the tripwire
    /// exists, or once the lock's holder has kept it [`GRACE`] past the end it recorded.
    pub(crate) fn acquire(
        tripwire: &TripwirePaths,
        run: &Run,
        timeout: Duration,
    ) -> Result<Self, String> {
        let path = lock_path(tripwire);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|error| format!("create {}: {error}", parent.display()))?;
        }
        let turnstile_path = path.with_file_name("run.turn");
        let turnstile = open(&turnstile_path)?;
        let mut file = open(&path)?;
        wait_for(&turnstile, &turnstile_path, tripwire, &path)?;
        wait_for(&file, &path, tripwire, &path)?;
        // Closing the turnstile lets the next waiting batch take it.
        drop(turnstile);
        let ends = unix_seconds(SystemTime::now() + timeout + TEARDOWN);
        file.set_len(0)
            .and_then(|()| file.seek(SeekFrom::Start(0)).map(drop))
            .and_then(|()| {
                writeln!(
                    file,
                    "{}\t{}\t{}\t{ends}",
                    std::process::id(),
                    run.case_id,
                    run.pass.label()
                )
            })
            .and_then(|()| file.flush())
            .map_err(|error| {
                format!(
                    "record the GPU run lock's holder {}: {error}",
                    path.display()
                )
            })?;
        Ok(Self { file })
    }
}

/// The lock file at `path`, created when missing (never truncated: it may hold a holder's record).
fn open(path: &Path) -> Result<File, String> {
    OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(path)
        .map_err(|error| format!("open {}: {error}", path.display()))
}

/// Take `file`'s lock (at `path`), trying every [`POLL`]; between tries, fail as soon as the
/// tripwire exists, or once the run lock's holder (its record at `lock`) has kept it [`GRACE`]
/// past the end it recorded.
fn wait_for(file: &File, path: &Path, tripwire: &TripwirePaths, lock: &Path) -> Result<(), String> {
    loop {
        if try_lock(file).map_err(|error| format!("lock {}: {error}", path.display()))? {
            return Ok(());
        }
        tripwire.refuse_if_tripped()?;
        if let Some(holder) = Holder::read(lock) {
            if unix_seconds(SystemTime::now()) > holder.ends + GRACE.as_secs() {
                return Err(format!(
                    "the host's GPU run lock {} is held by process {} for case {} ({}) more than \
                     {} s past the end its run recorded: a run that outlived its own timeout may \
                     still hold the GPU; follow the suspected-hang protocol in docs/VALIDATION.md",
                    lock.display(),
                    holder.pid,
                    holder.case_id,
                    holder.pass,
                    GRACE.as_secs()
                ));
            }
        }
        std::thread::sleep(POLL);
    }
}

impl Drop for RunLock {
    fn drop(&mut self) {
        // A stale record must not outlive the hold; closing the file releases the lock.
        let _ = self.file.set_len(0);
    }
}

/// What the holder recorded: its process, its run, and the latest time (Unix seconds) its run
/// ends.
#[derive(Debug, PartialEq, Eq)]
struct Holder {
    pid: u32,
    case_id: String,
    pass: String,
    ends: u64,
}

impl Holder {
    /// The record at `path`, if it holds a whole one (none between holds, or while it is written).
    fn read(path: &Path) -> Option<Self> {
        let text = fs::read_to_string(path).ok()?;
        let line = text.strip_suffix('\n')?;
        let [pid, case_id, pass, ends] = line.split('\t').collect::<Vec<_>>()[..] else {
            return None;
        };
        Some(Self {
            pid: pid.parse().ok()?,
            case_id: case_id.into(),
            pass: pass.into(),
            ends: ends.parse().ok()?,
        })
    }
}

fn unix_seconds(time: SystemTime) -> u64 {
    time.duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.as_secs())
}

/// Take `file`'s exclusive `flock` without waiting: `true` when taken, `false` when another open
/// file holds it.
#[cfg(unix)]
fn try_lock(file: &File) -> std::io::Result<bool> {
    use std::os::unix::io::AsRawFd as _;
    // SAFETY: a plain syscall on a descriptor `file` owns for the call.
    if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0 {
        return Ok(true);
    }
    let error = std::io::Error::last_os_error();
    match error.kind() {
        std::io::ErrorKind::WouldBlock => Ok(false),
        _ => Err(error),
    }
}

#[cfg(not(unix))]
fn try_lock(_file: &File) -> std::io::Result<bool> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "the GPU run lock needs a Unix host",
    ))
}
