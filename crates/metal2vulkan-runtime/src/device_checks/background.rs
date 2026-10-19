//! The device's checks thread: after the making, one of the runtime's threads runs the checks the
//! making left to it ([`Settle::Background`](super::Settle::Background)), in the suite's order, on
//! a kit of the running device (`kit.rs`), keeps each answer in the device-checks cache as its
//! check settles (`cache.rs`, merged into the setup's entry, with every answer it holds) and then
//! hands it to the device (`live.rs`). Once the last answered, it sends the settled feature
//! status, `metal2vulkan-feature-status-settled`.
//!
//! - **Holds nothing of the device between calls.** The thread holds the device weakly, and for
//!   one kit call or one answer at a time through its gate (`WeakDevice::with`, `live.rs`): the
//!   drop of the device's last handle waits for that call, lets no other begin, and tears the
//!   device down where it lets go, the teardown joining the thread (`Live::stop`) before the drop
//!   returns. A device that goes while a check runs ends that check; the thread then answers
//!   nothing more, keeps nothing more and says nothing, and the answers it kept stay: the setup's
//!   next making reads them and runs only the checks left. An answer is kept outside the hold, so
//!   a drop waits for at most that one write. The settled status goes to the device's sink after
//!   the hold, so the sink, the consumer's code, runs while the thread holds nothing.
//! - **One at a time.** Each device has at most one checks thread, started once as the device is
//!   assembled, which ends with its last check.

use super::kit::{DeviceFacts, Kit};
use super::live::Live;
use super::status::{status_text, CheckSource, FeatureStatus, Settled};
use super::{cache, run_check, Check, Target, FEATURE_STATUS_SETTLED};
use crate::device::{Device, WeakDevice};
use crate::driver_facts::inform_through;
use crate::exit_guard;
use metal2vulkan::device::{CheckAnswer, DeviceChecks, DeviceDescription, EnableSet};
use std::time::{Duration, Instant};

/// The checks a making left to the device's checks thread, and what the thread needs to run them.
pub(crate) struct Pending {
    /// The checks, in the suite's order.
    pub(super) checks: Vec<&'static Check>,
    /// The setup's key, and where its answers are kept.
    pub(super) key: cache::Key,
    pub(super) place: Result<cache::Place, String>,
    /// Every answer the making found or read, which the thread's join.
    pub(super) answers: DeviceChecks,
    /// The description, enable set and compile device the checks read (`Target`), as before any
    /// check.
    pub(super) description: DeviceDescription,
    pub(super) enabled: EnableSet,
    pub(super) compile: metal2vulkan::device::Device,
    pub(super) facts: DeviceFacts,
    /// The making's checks' and reads' time.
    pub(super) at_creation: Duration,
}

impl Pending {
    /// The checks it leaves, for the device's [`Live`].
    pub(crate) fn checks(&self) -> &[&'static Check] {
        &self.checks
    }
}

/// Start the checks thread of `device`, just assembled, for `pending`. Where the host starts no
/// thread, each check left answers unanswered, as one that could not run does.
pub(crate) fn start(device: &Device, pending: Pending) {
    let weak = device.downgrade();
    let (checks, at_creation) = (pending.checks.clone(), pending.at_creation);
    let builder = std::thread::Builder::new().name("metal2vulkan-device-checks".into());
    let thread = move || {
        run(&weak, pending);
        // No call follows: the device's drops take no lock from here on.
        weak.gate().end();
    };
    let error = match exit_guard::spawn(builder, thread) {
        Ok(handle) => {
            device.shared().checks.started(handle);
            return;
        }
        Err(error) => error,
    };
    let live: &Live = &device.shared().checks;
    live.gate.end();
    for check in checks {
        let mut found = DeviceChecks::new();
        for &fact in check.facts {
            let why = format!("the device's checks thread did not start: {error}");
            found.insert(fact, CheckAnswer::Unanswered(why));
        }
        live.answered(device, check, &found, CheckSource::Skipped);
    }
    let settled = Settled::InBackground {
        at_creation,
        wall: Duration::ZERO,
    };
    let status = live.settled(settled, None);
    device.inform(FEATURE_STATUS_SETTLED, settled_text(device, &status));
}

/// The cache's side of the checks thread: where the setup's answers are kept, and the first
/// write that failed (the settled status says why).
pub(crate) struct Keeper {
    place: Result<cache::Place, String>,
    key: cache::Key,
    error: Option<String>,
}

impl Keeper {
    /// The keeper of `key`'s answers at `place`.
    pub(crate) fn new(place: Result<cache::Place, String>, key: cache::Key) -> Keeper {
        Keeper {
            place,
            key,
            error: None,
        }
    }

    /// Keep `answers` in the setup's entry, merged into what it holds (`cache::Place::write`).
    pub(crate) fn keep(&mut self, answers: &DeviceChecks) {
        let failed = match &self.place {
            Ok(place) => place.write(&self.key, answers).err(),
            Err(why) => Some(why.clone()),
        };
        if self.error.is_none() {
            self.error = failed;
        }
    }

    /// The first write that failed, or why there is no place.
    pub(crate) fn error(&self) -> Option<String> {
        self.error.clone()
    }
}

/// Answer `checks` in order, apart from the device: `answer` runs one (`None` where the device
/// went during it), its answers join `answers`, `keeper` keeps them all, and `tell` hands the
/// check's own to the device (`false` where it went). Whether every check answered and was told:
/// a check the device went during keeps nothing, and one answered before it went keeps its
/// answers whether or not the device heard them.
pub(crate) fn answer_each(
    checks: &[&'static Check],
    answers: &mut DeviceChecks,
    keeper: &mut Keeper,
    mut answer: impl FnMut(&'static Check) -> Option<(DeviceChecks, CheckSource)>,
    mut tell: impl FnMut(&'static Check, &DeviceChecks, CheckSource) -> bool,
) -> bool {
    for &check in checks {
        let Some((found, source)) = answer(check) else {
            return false;
        };
        for (fact, answer) in found.iter() {
            answers.insert(fact, answer.clone());
        }
        keeper.keep(answers);
        if !tell(check, &found, source) {
            return false;
        }
    }
    true
}

/// The thread's work (the module docs).
fn run(weak: &WeakDevice, pending: Pending) {
    let started = Instant::now();
    let Pending {
        checks,
        key,
        place,
        mut answers,
        description,
        enabled,
        compile,
        facts,
        at_creation,
    } = pending;
    let target = Target {
        physical: facts.physical,
        description: &description,
        enabled: &enabled,
        compile: &compile,
    };
    let mut kit: Option<Result<Kit<'static>, String>> = None;
    let mut keeper = Keeper::new(place, key);
    let answer = |check: &'static Check| {
        let mut found = DeviceChecks::new();
        let make = || Kit::running(weak.clone(), facts);
        let source = run_check(&mut kit, make, check, &target, &mut found);
        // The device went, or the process is exiting: the check is cut short, and nothing more
        // is answered.
        let ended = (kit.as_ref()).is_some_and(|made| made.as_ref().is_ok_and(Kit::ended));
        (!ended).then_some((found, source))
    };
    let tell = |check: &'static Check, found: &DeviceChecks, source: CheckSource| {
        let told = weak.with(|device| {
            let live = &device.shared().checks;
            live.answered(device, check, found, source);
        });
        told.is_some()
    };
    if !answer_each(&checks, &mut answers, &mut keeper, answer, tell) {
        return;
    }
    // What the last check made is released before the device is told.
    drop(kit);
    let told = weak.with(|device| {
        let settled = Settled::InBackground {
            at_creation,
            wall: started.elapsed(),
        };
        let status = device.shared().checks.settled(settled, keeper.error());
        let sink = device.shared().instance.sink().cloned();
        (sink, settled_text(device, &status))
    });
    if let Some((sink, text)) = told {
        inform_through(sink.as_ref(), FEATURE_STATUS_SETTLED, text);
    }
}

/// The text of `device`'s settled feature status, `status` (`metal2vulkan-feature-status-settled`).
fn settled_text(device: &Device, status: &FeatureStatus) -> String {
    let description = &device.properties().description;
    status_text(FEATURE_STATUS_SETTLED, device.name(), description, status)
}
