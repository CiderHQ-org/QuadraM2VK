//! A device's feature status: how it decided each driver fact it takes, the informational message
//! it sends once as it is made (`metal2vulkan-feature-status`), and the one it sends once its
//! checks thread answered every check left to it (`metal2vulkan-feature-status-settled`).

use super::Check;
use metal2vulkan::device::{
    release_name, CheckAnswer, CheckOf, Decided, DeviceDescription, DeviceFact, FactStatus,
};
use std::path::PathBuf;
use std::time::Duration;

/// The id name of the informational message a device sends as it is made, its feature status.
pub(crate) const FEATURE_STATUS: &str = "metal2vulkan-feature-status";

/// The id name of the informational message a device sends once its checks thread answered every
/// check left to it, its feature status then.
pub(crate) const FEATURE_STATUS_SETTLED: &str = "metal2vulkan-feature-status-settled";

/// When a check's answer must be in. No Metal counterpart.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Settle {
    /// Before the device's first object: a fact that decides a refusal or shapes an object's
    /// layout, which must not change under the running device.
    AtCreation,
    /// On the device's checks thread after its making, unless `DeviceChecksOptions::wait`: the
    /// fact holds as the device was made (its baseline) until the answer is in. A fact the runtime
    /// reads as it records then switches to the answer; one translation reads (the compile
    /// profile) keeps it for the setup's next device, through the cache.
    Background,
}

/// How a device came by its checks' answers. No Metal counterpart.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Settled {
    /// The policy distrusts checks (`DeviceChecksPolicy::Distrusted`): none ran or was read, and
    /// every fact is as the curated tables have it.
    Distrusted,
    /// At least one check ran as the device was made, the checks taking `wall` in all (the
    /// cache's reads and writes included).
    AtCreation {
        /// The checks' time at the making.
        wall: Duration,
    },
    /// Every answer was read from the cache, kept when the checks ran `checked_at` seconds after
    /// the Unix epoch.
    FromCache {
        /// When the checks ran.
        checked_at: u64,
    },
    /// The device's checks thread runs the checks left to it after the making
    /// ([`Settle::Background`]), the making's own checks and reads taking `at_creation`.
    Running {
        /// The checks' time at the making.
        at_creation: Duration,
    },
    /// The device's checks thread answered every check left to it, in `wall` after it started,
    /// the making's own checks and reads having taken `at_creation`.
    InBackground {
        /// The checks' time at the making.
        at_creation: Duration,
        /// The checks thread's time.
        wall: Duration,
    },
}

/// Where one check's answer came from. No Metal counterpart.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CheckSource {
    /// It ran on the device, at this cost.
    Ran(CheckCost),
    /// It was read from the cache.
    Cache,
    /// It did not run: no check could (the reason is its answer's).
    Skipped,
    /// It runs on the device's checks thread: its answer is not in yet.
    Pending,
}

/// What one check cost as it ran on the device. No Metal counterpart.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CheckCost {
    /// Its time on the thread it ran on (the creating thread, or the device's checks thread), from
    /// its first object to its answer.
    pub wall: Duration,
    /// Its submissions' time on the GPU, from their timestamps, where the queue writes them.
    pub gpu: Option<Duration>,
    /// Its longest submission's time on the GPU, from the same timestamps.
    pub longest: Option<Duration>,
    /// The pipelines it made.
    pub pipelines: u32,
}

/// One check of a device's suite: one program that decides one fact or more. No Metal
/// counterpart.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CheckRun {
    /// The facts it decides.
    pub facts: Vec<DeviceFact>,
    /// When its answers must be in.
    pub settle: Settle,
    /// Where its answers came from.
    pub source: CheckSource,
}

/// How a device learns from a lost device on its setup (`docs/HOWTO.md`): where a record of a
/// risky path in use at the loss goes, which turns the path off from the next device on. No Metal
/// counterpart: Metal's driver is Metal's own.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Learning {
    /// The setup's learned key, in hex: SHA-256 over the driver, vendor, GPU and release, and on
    /// macOS the OS build.
    pub key: String,
    /// The directory of the setup's records: none where no device-checks directory could be
    /// used, so a loss is logged and kept nowhere.
    pub records: Option<PathBuf>,
    /// Whether the device watches its calls for a lost device: false where every one of the
    /// process's watch slots was taken as it was made, so it learns nothing (never refused).
    pub watched: bool,
    /// Whether the records found were read: false where the policy distrusts checks
    /// (`DeviceChecksPolicy::Distrusted`), which still records a loss.
    pub read: bool,
}

/// How a device decided each driver fact it takes, as Chrome's GPU feature status says per
/// feature: checked on the device (Metal's answer or not), listed in the curated tables (the
/// entry), or the exact path (why). No Metal counterpart: Metal's driver answers as Metal.
///
/// The device announces it once as it is made in an informational message,
/// `metal2vulkan-feature-status`; nothing a guest sees (its Metal properties, families, limits)
/// depends on it.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FeatureStatus {
    /// The setup's key, in hex: the driver, GPU, release, `pipelineCacheUUID`, OS build, suite
    /// version and pre-check profile the checks' cache keeps the answers by.
    pub setup: String,
    /// How the device came by its checks' answers.
    pub settled: Settled,
    /// The checks that ran on the device (each a program, of one fact or more), as it was made or
    /// on its checks thread since.
    pub checks_run: u32,
    /// The checks whose answers were read from the cache.
    pub checks_from_cache: u32,
    /// Each check of the suite, in its order.
    pub checks: Vec<CheckRun>,
    /// Why the cache was not read or not written, where it was not.
    pub cache_error: Option<String>,
    /// How it learns from a lost device; a fact it learned off reads `Decided::LearnedOff`.
    pub learning: Learning,
    /// Every fact the curated tables or the checks decide on the device, as it takes it: the
    /// deviations and translation defects (`metal2vulkan::device::DeviceDescription::fact_status`),
    /// the runtime's driver facts, and its driver defects, in the facts' order.
    pub facts: Vec<FactStatus>,
}

/// How the checks went, before the facts are read off the description.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ChecksRun {
    pub(crate) setup: String,
    pub(crate) settled: Settled,
    pub(crate) checks_run: u32,
    pub(crate) checks_from_cache: u32,
    pub(crate) checks: Vec<CheckRun>,
    pub(crate) cache_error: Option<String>,
    pub(crate) learning: Learning,
}

impl ChecksRun {
    /// No check yet, for the setup `setup`.
    pub(crate) fn new(setup: String) -> ChecksRun {
        ChecksRun {
            setup,
            settled: Settled::Distrusted,
            checks_run: 0,
            checks_from_cache: 0,
            checks: Vec::new(),
            cache_error: None,
            learning: Learning::default(),
        }
    }

    /// `check`'s answer, from `source`.
    pub(crate) fn push(&mut self, check: &Check, source: CheckSource) {
        match source {
            CheckSource::Ran(_) => self.checks_run += 1,
            CheckSource::Cache => self.checks_from_cache += 1,
            CheckSource::Skipped | CheckSource::Pending => {}
        }
        self.checks.push(CheckRun {
            facts: check.facts.to_vec(),
            settle: check.settle,
            source,
        });
    }
}

impl FeatureStatus {
    /// The status of a device made from `description` (its checks' answers in it) after `run`,
    /// with `runtime_defects`, the runtime's defect rows (`driver_defects::status_rows`).
    pub(crate) fn of(
        description: &DeviceDescription,
        run: ChecksRun,
        runtime_defects: Vec<FactStatus>,
    ) -> FeatureStatus {
        let trusted = run.settled != Settled::Distrusted;
        let mut facts = description.fact_status(trusted);
        facts.extend(runtime_defects);
        facts.sort_by_key(|row| row.fact);
        FeatureStatus {
            setup: run.setup,
            settled: run.settled,
            checks_run: run.checks_run,
            checks_from_cache: run.checks_from_cache,
            checks: run.checks,
            cache_error: run.cache_error,
            learning: run.learning,
            facts,
        }
    }

    /// The status of a device no check ran for, every fact as the tables have it: a device whose
    /// properties a test builds alone.
    pub(crate) fn unchecked(description: &DeviceDescription) -> FeatureStatus {
        let run = ChecksRun::new(String::new());
        let defects = crate::driver_defects::status_rows(description, false);
        FeatureStatus::of(description, run, defects)
    }

    /// The row of `fact`, where the status has one.
    pub fn fact(&self, fact: DeviceFact) -> Option<&FactStatus> {
        self.facts.iter().find(|row| row.fact == fact)
    }
}

/// The text of the informational message of id name `id_name` that `status`'s device, named `name`
/// and made from `description`, sends: as it is made ([`FEATURE_STATUS`]), or once its checks
/// thread answered the last check left to it ([`FEATURE_STATUS_SETTLED`]).
pub(crate) fn status_text(
    id_name: &str,
    name: &str,
    description: &DeviceDescription,
    status: &FeatureStatus,
) -> String {
    let settled = match status.settled {
        Settled::Distrusted => {
            "checks distrusted, every fact as the curated tables have it".to_owned()
        }
        Settled::AtCreation { wall } => format!(
            "settled at creation in {:.1} ms ({} checks run, {} from the cache)",
            wall.as_secs_f64() * 1000.0,
            status.checks_run,
            status.checks_from_cache
        ),
        Settled::FromCache { checked_at } => format!(
            "read from the cache ({} checks, checked at {checked_at} s past the Unix epoch)",
            status.checks_from_cache
        ),
        Settled::Running { at_creation } => format!(
            "{} checks settle on the device's checks thread ({} run and {} read from the cache \
             at the making, in {:.1} ms)",
            pending(status),
            status.checks_run,
            status.checks_from_cache,
            at_creation.as_secs_f64() * 1000.0
        ),
        Settled::InBackground { at_creation, wall } => format!(
            "settled on the device's checks thread in {:.1} ms after {:.1} ms at the making ({} \
             checks run, {} from the cache)",
            wall.as_secs_f64() * 1000.0,
            at_creation.as_secs_f64() * 1000.0,
            status.checks_run,
            status.checks_from_cache
        ),
    };
    let cache = match &status.cache_error {
        Some(why) => format!("; the cache was not used: {why}"),
        None => String::new(),
    };
    let learning = &status.learning;
    let learning = match (&learning.records, learning.watched) {
        (_, false) => "; unwatched for a lost device (every watch slot taken)".to_owned(),
        (None, true) => "; a lost device is logged, kept nowhere".to_owned(),
        (Some(_), true) => format!(
            "; learned key {}{}",
            short(&learning.key),
            if learning.read {
                ""
            } else {
                " (records not read)"
            }
        ),
    };
    let facts: Vec<String> = status.facts.iter().map(fact_text).collect();
    let facts = match facts.is_empty() {
        true => "no driver fact".to_owned(),
        false => facts.join("; "),
    };
    format!(
        "metal2vulkan: the device, on {name}: feature status of {:?} {} on {:04x}:{:04x}, setup \
         {}: {settled}{cache}{learning}. {facts} [{id_name}]",
        description.driver_id,
        release_name(description.driver_id, description.driver_version),
        description.vendor_id,
        description.device_id,
        short(&status.setup),
    )
}

/// How many of `status`'s checks have not answered yet.
fn pending(status: &FeatureStatus) -> usize {
    (status.checks.iter())
        .filter(|check| check.source == CheckSource::Pending)
        .count()
}

/// The first 16 hex digits of a setup key.
fn short(setup: &str) -> &str {
    setup.get(..16).unwrap_or(setup)
}

/// One fact's part of the message: its name and kind, whether it is taken, and how that was
/// decided.
fn fact_text(row: &FactStatus) -> String {
    let taken = match (row.holds, row.fact.is_defect()) {
        (true, false) => "taken",
        (false, false) => "not taken",
        (true, true) => "worked around",
        (false, true) => "not worked around",
    };
    let how = match &row.decided {
        Decided::CheckedMetals => "checked: Metal's answer".to_owned(),
        Decided::CheckedNotMetals => "checked: not Metal's answer".to_owned(),
        Decided::Listed(entry) => format!("listed: {entry}"),
        Decided::Assumed(entry) => format!("assumed: {entry}"),
        Decided::Unlisted => "unlisted".to_owned(),
        Decided::Exact(why) => format!("exact: {why}"),
        Decided::LearnedOff(why) => format!("learned off: {why}"),
        other => format!("{other:?}"),
    };
    let check = match (&row.check, &row.decided) {
        (CheckOf::Answered(CheckAnswer::NotMetals(how)), _) if !how.is_empty() => {
            format!(" ({how})")
        }
        (CheckOf::NextDevice(answer), _) => {
            format!(
                " (checked {}, taken from the next device)",
                answer_text(answer)
            )
        }
        (CheckOf::Answered(CheckAnswer::NotRun(why)), _) => {
            format!(" (its check does not run: {why})")
        }
        (CheckOf::Answered(CheckAnswer::Unanswered(why)), decided)
            if !matches!(decided, Decided::Exact(_)) =>
        {
            format!(" (its check is unanswered: {why})")
        }
        (CheckOf::Pending, _) => " (its check has not answered)".to_owned(),
        _ => String::new(),
    };
    format!(
        "{} ({}): {taken}, {how}{check}",
        row.fact.name(),
        row.fact.kind()
    )
}

/// A check's answer in a few words.
fn answer_text(answer: &CheckAnswer) -> String {
    match answer {
        CheckAnswer::Metals => "Metal's answer".to_owned(),
        CheckAnswer::NotMetals(how) if how.is_empty() => "not Metal's answer".to_owned(),
        CheckAnswer::NotMetals(how) => format!("not Metal's answer: {how}"),
        CheckAnswer::Unanswered(why) => format!("unanswered: {why}"),
        CheckAnswer::NotRun(why) => format!("not run: {why}"),
        other => format!("{other:?}"),
    }
}
