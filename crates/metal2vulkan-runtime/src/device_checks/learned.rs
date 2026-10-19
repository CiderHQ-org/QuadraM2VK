//! Learning from a lost device: where the device is lost while a submission that took a risky
//! path is in flight, that path is off on the setup from the next device on, logged.
//!
//! - **The risky paths** ([`Risky`]): the direct paths a known-bug entry's workaround replaces,
//!   for an entry whose bug loses the device or hangs the GPU, taken on a setup the list leaves
//!   clean or does not name. A bug that crashes the process (`SIGSEGV`) loses no device, so no
//!   watch sees it: those entries are list entries alone.
//! - **Tagged** ([`Risks`]): a command buffer gathers the risky paths its encoders took, a pipeline
//!   state's among them. Once the driver accepted a batch's `vkQueueSubmit`, each path its command
//!   buffers took is noted on the queue at the timeline value the submission signals
//!   ([`Watch::submitted`]); a free of memory a device address lay in, while work may be pending,
//!   notes every queue at the last value submitted there ([`Watch::risked_in_flight`]). Nothing is
//!   kept per submission.
//! - **One door** (`thunks.rs`): the device's functions, the exit guard's table among them, hold a
//!   slot's entry point in place of the driver's for each device and queue call of Vulkan 1.0 to
//!   1.3 that returns a `VkResult`, so a call that returns `VK_ERROR_DEVICE_LOST`, on any thread,
//!   has the watch record the loss before the result reaches the runtime ([`Watch::lost`]), and so
//!   before the program sees an error after which it may end its process at once. The first call
//!   records; another that sees the loss meanwhile waits for the record, at most
//!   [`RECORD_WAIT`]. The record takes no lock of the runtime's (its caller may hold a queue's)
//!   and runs none of the program's code (its sink may take a lock the program holds). A device
//!   made while every slot is taken runs unwatched ([`Slot`]): never refused, counted, and stated.
//! - **The blame**: a path is blamed where some queue's value for it is above the value through
//!   which the host saw that queue's timeline complete: a submission the driver accepted, that took
//!   the path, not seen complete. A submission whose own `vkQueueSubmit` returned the error was
//!   never accepted, and is not blamed; where nothing tagged is in flight (a shader's own endless
//!   loop that the driver timed out), nothing is. A wrong blame costs only speed: each path's
//!   replacement is exact.
//! - **The record** ([`Target::record`]): `<directory>/v1/<key>.learned/<path>`, one file per path,
//!   made by one create-exclusive open and one write, so its name alone is the record; the body
//!   states the setup and the call. The key ([`Setup::key`]) is SHA-256 over the driver, the
//!   vendor, the GPU and the release, and on macOS the OS build (MoltenVK's answers are Metal's):
//!   not the check suite's version, the pipeline cache's UUID or a Linux kernel's release, since a
//!   record forgotten costs the user another lost device, one kept too long only speed. One line
//!   per path goes to stderr at the loss, `metal2vulkan-learned-off`.
//! - **The next device** ([`read`]): lists the directory as it is made, before its compile device
//!   (no listing where the policy distrusts checks), and takes the workaround of every path
//!   recorded. Deleting the directory forgets them all.

mod thunks;

use super::cache;
use super::Learning;
use metal2vulkan::device::{
    DeviceChecks, DeviceChecksPolicy, DeviceDescription, DeviceFact, DevicePolicy,
};
use metal2vulkan::error::RuntimeDriverDefect;
use metal2vulkan::pipeline::{PrecisionNote, StageModule};
use metal2vulkan::vk::DriverId;
use sha2::{Digest, Sha256};
use std::ffi::{c_void, CStr};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// The message id of the line each path learned off writes to stderr at the loss.
pub(crate) const LEARNED_OFF: &str = "metal2vulkan-learned-off";

/// How long a call that sees the loss while another records it waits for that record.
pub(crate) const RECORD_WAIT: Duration = Duration::from_secs(2);

/// The records' layout, which their bodies state: that of the device-checks directory they are in
/// (`cache::Place::dir`).
const LAYOUT: &str = "v1";

/// A path that can lose the device or hang the GPU, taken where its workaround is not applied
/// (the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum Risky {
    /// `vkCmdCopyImage` between the stencil aspect of a multisampled depth/stencil image and a
    /// multisampled colour image, where `RuntimeDriverDefect::MultisampledStencilColourCopies` is
    /// not worked around.
    MultisampledStencilColourCopies,
    /// An execution of generated commands without the explicit preprocess, where
    /// `RuntimeDriverDefect::ImplicitPreprocessAfterExecutionSet` is not worked around.
    ImplicitPreprocessAfterExecutionSet,
    /// Memory a device address lay in freed while work may be pending, where
    /// `RuntimeDriverDefect::FreedAddressMemoryFailsPendingWork` is not worked around.
    FreedAddressMemoryFailsPendingWork,
    /// A pipeline whose vertex or fragment function holds more Function storage a thread than an
    /// unmeasured device is taken to hold (`PrecisionNote::FunctionStorageAssumed`), where
    /// `DeviceFact::FunctionStorageFloor` does not hold.
    FunctionStorageFloor,
}

/// How many risky paths there are.
const RISKY: usize = Risky::ALL.len();

impl Risky {
    /// Every risky path, in its bits' order.
    pub(crate) const ALL: [Risky; 4] = [
        Risky::MultisampledStencilColourCopies,
        Risky::ImplicitPreprocessAfterExecutionSet,
        Risky::FreedAddressMemoryFailsPendingWork,
        Risky::FunctionStorageFloor,
    ];

    /// Its bit in a [`Risks`].
    const fn bit(self) -> u64 {
        1 << self as u32
    }

    /// The fact whose list entry replaces it, which a record of it turns on.
    pub(crate) const fn fact(self) -> DeviceFact {
        match self {
            Risky::MultisampledStencilColourCopies => {
                DeviceFact::Defect(RuntimeDriverDefect::MultisampledStencilColourCopies)
            }
            Risky::ImplicitPreprocessAfterExecutionSet => {
                DeviceFact::Defect(RuntimeDriverDefect::ImplicitPreprocessAfterExecutionSet)
            }
            Risky::FreedAddressMemoryFailsPendingWork => {
                DeviceFact::Defect(RuntimeDriverDefect::FreedAddressMemoryFailsPendingWork)
            }
            Risky::FunctionStorageFloor => DeviceFact::FunctionStorageFloor,
        }
    }

    /// Its record's file name: its fact's kind and name (`defect-MultisampledStencilColourCopies`).
    pub(crate) fn name(self) -> String {
        let fact = self.fact();
        format!("{}-{}", fact.kind(), fact.name())
    }

    /// The path whose record's file name is `name`.
    pub(crate) fn of_name(name: &str) -> Option<Risky> {
        Risky::ALL.into_iter().find(|risky| risky.name() == name)
    }
}

/// A set of risky paths: what a command buffer took, or a submission carried.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Risks(u64);

impl Risks {
    /// No path.
    pub(crate) const NONE: Risks = Risks(0);

    /// `risky` alone.
    pub(crate) const fn of(risky: Risky) -> Risks {
        Risks(risky.bit())
    }

    /// Whether it holds no path.
    pub(crate) const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Both sets.
    #[must_use]
    pub(crate) const fn with(self, other: Risks) -> Risks {
        Risks(self.0 | other.0)
    }

    /// Whether it holds `risky`.
    pub(crate) const fn holds(self, risky: Risky) -> bool {
        self.0 & risky.bit() != 0
    }

    /// Its paths, in their bits' order.
    pub(crate) fn iter(self) -> impl Iterator<Item = Risky> {
        Risky::ALL
            .into_iter()
            .filter(move |&risky| self.holds(risky))
    }
}

/// The risky paths a pipeline of `stages` takes wherever a draw or dispatch binds it: a stage that
/// records `PrecisionNote::FunctionStorageAssumed`, past the Function storage an unmeasured device
/// is taken to hold (where the floor holds, such a stage is refused, and none is made).
pub(crate) fn stage_risks(stages: &[StageModule]) -> Risks {
    let assumed = stages
        .iter()
        .any(|stage| (stage.precision_notes).contains(&PrecisionNote::FunctionStorageAssumed));
    match assumed {
        true => Risks::of(Risky::FunctionStorageFloor),
        false => Risks::NONE,
    }
}

/// The risky paths a command buffer took so far: its encoders add to it as they encode, and each
/// of its batches' submissions reads it.
#[derive(Debug, Default)]
pub(crate) struct TakenRisks(AtomicU64);

impl TakenRisks {
    /// `risky` was taken.
    pub(crate) fn took(&self, risks: Risks) {
        if !risks.is_empty() {
            self.0.fetch_or(risks.0, Ordering::Release);
        }
    }

    /// The paths taken so far.
    pub(crate) fn read(&self) -> Risks {
        Risks(self.0.load(Ordering::Acquire))
    }
}

/// One `VkQueue`'s progress as the device's watch reads it at a loss: the value of the last batch
/// submitted there, the value through which the host saw its timeline complete, and per risky
/// path the value of the last submission there that took it.
#[derive(Debug, Default)]
pub(crate) struct QueueProgress {
    submitted: AtomicU64,
    completed: AtomicU64,
    risked: [AtomicU64; RISKY],
}

impl QueueProgress {
    /// A batch signalling `value` was submitted, accepted: the value of the last batch, after it.
    pub(crate) fn submitted(&self, value: u64) -> u64 {
        self.submitted.fetch_max(value, Ordering::AcqRel)
    }

    /// The value of the last batch submitted; 0 before the first.
    pub(crate) fn submitted_value(&self) -> u64 {
        self.submitted.load(Ordering::Acquire)
    }

    /// The host saw the timeline reach `value`.
    pub(crate) fn completed(&self, value: u64) {
        self.completed.fetch_max(value, Ordering::AcqRel);
    }

    /// The value through which every batch completed, as the host saw.
    pub(crate) fn completed_value(&self) -> u64 {
        self.completed.load(Ordering::Acquire)
    }

    /// `risks` were taken by a submission that signals `value`, accepted.
    fn risked(&self, risks: Risks, value: u64) {
        for risky in risks.iter() {
            self.risked[risky as usize].fetch_max(value, Ordering::AcqRel);
        }
    }

    /// The paths taken by a submission the host has not seen complete.
    fn in_flight(&self) -> Risks {
        let completed = self.completed_value();
        let held = Risky::ALL
            .into_iter()
            .filter(|&risky| self.risked[risky as usize].load(Ordering::Acquire) > completed);
        held.fold(Risks::NONE, |risks, risky| risks.with(Risks::of(risky)))
    }
}

/// The setup a device's records are kept under: the parts of its identity its key covers
/// ([`Setup::key`]), which the records' bodies state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Setup {
    pub(crate) driver_id: Option<DriverId>,
    pub(crate) vendor_id: u32,
    pub(crate) device_id: u32,
    pub(crate) driver_version: u32,
    /// The OS build, on macOS alone: MoltenVK's answers are Metal's, which change with it.
    pub(crate) os_build: Option<String>,
}

impl Setup {
    /// The setup of the device `description` describes.
    pub(crate) fn of(description: &DeviceDescription) -> Setup {
        Setup {
            driver_id: description.driver_id,
            vendor_id: description.vendor_id,
            device_id: description.device_id,
            driver_version: description.driver_version,
            os_build: cfg!(target_os = "macos")
                .then(cache::os_build)
                .filter(|build| !build.is_empty()),
        }
    }

    /// Its key: SHA-256 over the driver, vendor, GPU and release, and the OS build where there is
    /// one, in lowercase hex (the module docs).
    pub(crate) fn key(&self) -> String {
        let mut hash = Sha256::new();
        hash.update(b"metal2vulkan learned key\0");
        hash.update(self.driver_id.map_or(0, |id| id.as_raw()).to_le_bytes());
        hash.update(self.vendor_id.to_le_bytes());
        hash.update(self.device_id.to_le_bytes());
        hash.update(self.driver_version.to_le_bytes());
        if let Some(build) = &self.os_build {
            hash.update((build.len() as u64).to_le_bytes());
            hash.update(build.as_bytes());
        }
        cache::hex(&hash.finalize())
    }

    /// The lines of a record's body that state the setup.
    fn lines(&self, key: &str) -> String {
        let driver = match self.driver_id {
            Some(id) => format!("{} ({id:?})", id.as_raw()),
            None => "none".to_owned(),
        };
        let mut lines = format!(
            "key: {key}\ndriverID: {driver}\nvendorID: 0x{:04x}\ndeviceID: 0x{:04x}\n\
             driverVersion: {} (0x{:08x})\n",
            self.vendor_id, self.device_id, self.driver_version, self.driver_version
        );
        if let Some(build) = &self.os_build {
            lines.push_str(&format!("OS build: {build}\n"));
        }
        lines
    }
}

/// Where a device's records go: its setup's directory, its key and the setup.
#[derive(Clone, Debug)]
pub(crate) struct Target {
    directory: PathBuf,
    key: String,
    setup: Setup,
}

impl Target {
    /// The records of `setup` in `directory`, the device-checks directory's `v1`
    /// (`cache::Place::dir`).
    pub(crate) fn new(directory: &Path, setup: Setup) -> Target {
        let key = setup.key();
        Target {
            directory: records_directory(directory, &key),
            key,
            setup,
        }
    }

    /// The directory its records go in.
    pub(crate) fn directory(&self) -> &Path {
        &self.directory
    }

    /// Its setup's key.
    pub(crate) fn key(&self) -> &str {
        &self.key
    }

    /// Record `risky` as learned off at a loss `call` returned: its file, made where it is not
    /// there yet, and the line on stderr. Whether a file was made; an error where none could be.
    fn record(&self, risky: Risky, call: &CStr) -> std::io::Result<bool> {
        let path = self.directory.join(risky.name());
        let seen = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |since| since.as_secs());
        let body = format!(
            "metal2vulkan learned record {LAYOUT}\npath: {}\ncall: {} returned \
             VK_ERROR_DEVICE_LOST\nseen: {seen}\nruntime: {}\n{}",
            risky.name(),
            call.to_string_lossy(),
            env!("CARGO_PKG_VERSION"),
            self.setup.lines(&self.key),
        );
        let made = make_record(&self.directory, &path, body.as_bytes())?;
        say(format_args!(
            "{LEARNED_OFF}: {} is off on setup {} (vendorID 0x{:04x}, driverVersion 0x{:08x}, \
             deviceID 0x{:04x}) from the next device on: the device was lost ({} returned \
             VK_ERROR_DEVICE_LOST) while a submission that took it was in flight; record {}",
            risky.name(),
            self.key,
            self.setup.vendor_id,
            self.setup.driver_version,
            self.setup.device_id,
            call.to_string_lossy(),
            path.display(),
        ));
        Ok(made)
    }
}

/// Write `line` to stderr, a failure ignored: the record runs inside the driver's call, where a
/// panic would end the process.
fn say(line: std::fmt::Arguments<'_>) {
    let _ = writeln!(std::io::stderr(), "{line}");
}

/// The directory of the records of the setup whose key is `key`, in `directory`, the
/// device-checks directory's `v1`.
pub(crate) fn records_directory(directory: &Path, key: &str) -> PathBuf {
    directory.join(format!("{key}.learned"))
}

/// Make the record at `path`, in `directory`, of `body`, unless one is there: one create-exclusive
/// open and one write, the file and then the directory flushed to the disk, both readable by the
/// user alone, as the checks' entries are. Whether it was made.
fn make_record(directory: &Path, path: &Path, body: &[u8]) -> std::io::Result<bool> {
    cache::create_private_dir_all(directory)?;
    let mut file = match cache::new_private_file(path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => return Ok(false),
        Err(error) => return Err(error),
    };
    // The name is the record: a body cut short still counts.
    file.write_all(body)?;
    file.sync_all()?;
    // Best effort: the entry's own flush, where the platform opens a directory to flush it.
    if let Ok(entries) = fs::File::open(directory) {
        let _ = entries.sync_all();
    }
    Ok(true)
}

/// One record a device found as it was made: the path it turns off, and the record's body.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Record {
    pub(crate) risky: Risky,
    pub(crate) path: PathBuf,
    /// Its body as written (empty where the write was cut short or unreadable).
    pub(crate) body: String,
}

impl Record {
    /// What the record says, for the feature status: the call and the time it states.
    pub(crate) fn why(&self) -> String {
        let field = |name: &str| {
            self.body
                .lines()
                .find_map(|line| line.strip_prefix(name))
                .map(str::trim)
        };
        let call = field("call:").unwrap_or("a call returned VK_ERROR_DEVICE_LOST");
        let seen = field("seen:").map_or(String::new(), |seen| format!(" at {seen}"));
        format!(
            "the device was lost ({call}{seen}) while a submission that took it was in flight \
             (record {})",
            self.path.display()
        )
    }
}

/// The records of the setup whose key is `key` in `directory`, the device-checks directory's `v1`,
/// in their paths' order: none where it has none. A file whose name is no risky path's is left
/// alone.
pub(crate) fn read(directory: &Path, key: &str) -> Vec<Record> {
    let records = records_directory(directory, key);
    let Ok(entries) = fs::read_dir(&records) else {
        return Vec::new();
    };
    let mut found: Vec<Record> = entries
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let risky = Risky::of_name(entry.file_name().to_str()?)?;
            let path = entry.path();
            let body = fs::read_to_string(&path).unwrap_or_default();
            Some(Record { risky, path, body })
        })
        .collect();
    found.sort_by_key(|record| record.risky);
    found.dedup_by_key(|record| record.risky);
    found
}

/// What the device `description` describes learned on its setup, with `place`, the device-checks
/// directory's `v1`, where there is one: where its records go, and its learning status; each
/// record's fact goes into `checks` (`Decided::LearnedOff`) unless `policy` distrusts checks.
/// Reads no file under that policy.
pub(crate) fn settle(
    description: &DeviceDescription,
    place: Option<&Path>,
    policy: DevicePolicy,
    checks: &mut DeviceChecks,
) -> (Option<Target>, Learning) {
    let setup = Setup::of(description);
    let read = policy.device_checks != DeviceChecksPolicy::Distrusted;
    let Some(place) = place else {
        let learning = Learning {
            key: setup.key(),
            records: None,
            watched: false,
            read,
        };
        return (None, learning);
    };
    let learned = Target::new(place, setup);
    if read {
        for record in self::read(place, learned.key()) {
            checks.learn(record.risky.fact(), record.why());
        }
    }
    let learning = Learning {
        key: learned.key().to_owned(),
        records: Some(learned.directory().to_owned()),
        watched: false,
        read,
    };
    (Some(learned), learning)
}

/// What a device's watch saw so far, for the work counts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct WatchCounts {
    /// Whether the device holds a slot (`thunks.rs`): unwatched, it learns nothing.
    pub(crate) watched: bool,
    /// The calls that returned `VK_ERROR_DEVICE_LOST`.
    pub(crate) losses: u64,
    /// The records made at the loss.
    pub(crate) recorded: u64,
    /// The accepted submissions that took a risky path.
    pub(crate) risky_submissions: u64,
}

/// Where a loss is: none seen, being recorded, recorded.
const LIVE: u8 = 0;
const RECORDING: u8 = 1;
const RECORDED: u8 = 2;

/// A device's watch of its losses (the module docs): where its records go, its queues' progress,
/// and whether a loss was seen.
#[derive(Debug)]
pub(crate) struct Watch {
    /// Where its records go; none where the device keeps none (no device-checks directory).
    target: Option<Target>,
    queues: Box<[Arc<QueueProgress>]>,
    state: AtomicU8,
    watched: AtomicBool,
    losses: AtomicU64,
    recorded: AtomicU64,
    risky_submissions: AtomicU64,
}

impl Watch {
    /// The watch of a device with `queues` `VkQueue`s, recording at `target`.
    pub(crate) fn new(target: Option<Target>, queues: usize) -> Arc<Watch> {
        Arc::new(Watch {
            target,
            queues: (0..queues).map(|_| Arc::default()).collect(),
            state: AtomicU8::new(LIVE),
            watched: AtomicBool::new(false),
            losses: AtomicU64::new(0),
            recorded: AtomicU64::new(0),
            risky_submissions: AtomicU64::new(0),
        })
    }

    /// `VkQueue` `queue`'s progress.
    pub(crate) fn queue(&self, queue: usize) -> &Arc<QueueProgress> {
        &self.queues[queue]
    }

    /// A batch submission on `VkQueue` `queue` that signals `value` and whose command buffers took
    /// `risks` was accepted.
    pub(crate) fn submitted(&self, queue: usize, value: u64, risks: Risks) {
        if risks.is_empty() {
            return;
        }
        self.queues[queue].risked(risks, value);
        self.risky_submissions.fetch_add(1, Ordering::Relaxed);
    }

    /// `risky` was taken while every batch submitted so far may be in flight: each queue is noted
    /// at the last value submitted there.
    pub(crate) fn risked_in_flight(&self, risky: Risky) {
        for queue in self.queues.iter() {
            let last = queue.submitted_value();
            if last > queue.completed_value() {
                queue.risked(Risks::of(risky), last);
            }
        }
    }

    /// The paths blamed now: those of a submission accepted and not seen complete, on any queue.
    pub(crate) fn blamed(&self) -> Risks {
        (self.queues.iter()).fold(Risks::NONE, |risks, queue| risks.with(queue.in_flight()))
    }

    /// `call` returned `VK_ERROR_DEVICE_LOST` (the module docs): the first such call records, and
    /// one made while it does waits for its record, at most [`RECORD_WAIT`].
    pub(crate) fn lost(&self, call: &CStr) {
        self.losses.fetch_add(1, Ordering::Relaxed);
        match (self.state).compare_exchange(LIVE, RECORDING, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => {
                self.record(call);
                self.state.store(RECORDED, Ordering::Release);
            }
            Err(RECORDING) => {
                let began = Instant::now();
                while self.state.load(Ordering::Acquire) == RECORDING
                    && began.elapsed() < RECORD_WAIT
                {
                    std::thread::yield_now();
                }
            }
            Err(_) => {}
        }
    }

    /// Record each path blamed for the loss `call` returned.
    fn record(&self, call: &CStr) {
        let blamed = self.blamed();
        let Some(target) = &self.target else {
            for risky in blamed.iter() {
                say(format_args!(
                    "{LEARNED_OFF}: the device was lost ({} returned VK_ERROR_DEVICE_LOST) while \
                     a submission that took {} was in flight; no device-checks directory keeps \
                     the record",
                    call.to_string_lossy(),
                    risky.name()
                ));
            }
            return;
        };
        for risky in blamed.iter() {
            match target.record(risky, call) {
                Ok(true) => {
                    self.recorded.fetch_add(1, Ordering::Relaxed);
                }
                Ok(false) => {}
                Err(error) => say(format_args!(
                    "{LEARNED_OFF}: the device was lost ({} returned VK_ERROR_DEVICE_LOST) while \
                     a submission that took {} was in flight; its record in {} failed: {error}",
                    call.to_string_lossy(),
                    risky.name(),
                    target.directory().display()
                )),
            }
        }
    }

    /// What it saw so far.
    pub(crate) fn counts(&self) -> WatchCounts {
        WatchCounts {
            watched: self.watched.load(Ordering::Relaxed),
            losses: self.losses.load(Ordering::Relaxed),
            recorded: self.recorded.load(Ordering::Relaxed),
            risky_submissions: self.risky_submissions.load(Ordering::Relaxed),
        }
    }
}

/// One device's slot (`thunks.rs`): its watch, until it is dropped, after the device is
/// destroyed, which frees the slot for the next device.
pub(crate) struct Slot {
    index: usize,
    watch: Arc<Watch>,
}

impl Slot {
    /// The slot's entry point in place of the driver's `name`, if the watch routes that call:
    /// the exit guard's table (`device/functions.rs`) holds it too.
    pub(crate) fn entry(&self, name: &CStr) -> Option<*const c_void> {
        crate::recorder::per_slot!(self.index, S => thunks::entry::<S>(name))
    }
}

impl Drop for Slot {
    fn drop(&mut self) {
        thunks::let_go(self.index);
        self.watch.watched.store(false, Ordering::Relaxed);
    }
}

/// A slot for `watch`, its entry points put in place of the driver's calls that return a
/// `VkResult` in a device's Vulkan 1.0 to 1.3 tables `v1_0` to `v1_3` (the module docs), each
/// reporting a lost device to `watch`; the tables left as they are, and no slot, where every slot
/// is taken. The device's functions are made from the tables where they hold (`device/mod.rs`):
/// only the slot's thunks keep the driver's entry points (`thunks.rs`).
///
/// # Safety
///
/// The tables are a live device's functions, its driver's (or its topmost layer's), its pipeline
/// makings routed through `compiles.rs` at most, not already routed here.
pub(crate) unsafe fn route(
    v1_0: &mut ash::DeviceFnV1_0,
    v1_1: &mut ash::DeviceFnV1_1,
    v1_2: &mut ash::DeviceFnV1_2,
    v1_3: &mut ash::DeviceFnV1_3,
    watch: &Arc<Watch>,
) -> Option<Slot> {
    let index = thunks::take(watch)?;
    watch.watched.store(true, Ordering::Relaxed);
    crate::recorder::per_slot!(index, S => thunks::route_tables::<S>(v1_0, v1_1, v1_2, v1_3));
    Some(Slot {
        index,
        watch: Arc::clone(watch),
    })
}
