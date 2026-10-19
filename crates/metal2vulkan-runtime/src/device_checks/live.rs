//! A device's checks as they settle after its making (`background.rs`): its feature status as it
//! stands, the runtime facts whose checks answer on the checks thread, which switch as each answer
//! comes in, the thread, and what a check made while the device runs.
//!
//! - **A runtime fact switches.** A fact the runtime reads as it records (`DeviceFact::Fact`,
//!   `DeviceFact::Defect`) whose check was left to the thread holds as the device was made (its
//!   baseline) until the answer is in, then as the answer decides it, for every reader that reads
//!   it through [`Device::driver_fact`] (a defect's through [`Device::driver_defect`]). A reader
//!   whose choice must hold across several reads (one draw's) reads it once. `DeviceProperties`
//!   stays as made.
//! - **A translation fact waits for the next device.** A fact translation reads
//!   (`DeviceFact::Deviation`, `DeviceFact::Workaround`) is in the compile profile, which never
//!   changes under a running device: its row keeps the fact as made, its check
//!   `CheckOf::NextDevice(answer)`, and the answer goes to the device-checks cache, where the
//!   setup's next device reads it as it is made.
//! - **What a check made** on the running device is kept here as it is made, so a device dropped
//!   while a check runs releases it in its teardown, after its idle wait ([`Live::release`]).
//! - **The last handle tears the device down, never the thread** ([`Gate`]). The checks thread
//!   holds the device only for one call at a time (`WeakDevice::with`), and each such hold, and
//!   every drop of a device handle while the thread may still call, goes through the gate's lock.
//!   The drop of the last handle but a call's waits for the call to end and lets no other begin,
//!   so the teardown runs where the program (or the runtime thread that held the device last) let
//!   go, and joins the thread and releases what its checks made before that drop returns: the
//!   consumer of an adopted device may destroy its `VkDevice` as soon as it has let go. The wait is
//!   one Vulkan call at most (a fence's wait is cut into slices, `kit.rs`).

use super::kit::Made;
use super::status::{CheckSource, FeatureStatus, Settled};
use super::Check;
use crate::device::{lock, Device};
use metal2vulkan::device::{CheckOf, DeviceChecks, DeviceFact, DriverFact};
use metal2vulkan::error::RuntimeDriverDefect;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::thread::{JoinHandle, ThreadId};

/// Whether translation reads `fact` (the compile profile): a deviation or a translation defect's
/// workaround. The runtime reads the rest as it records.
pub(crate) fn read_by_translation(fact: DeviceFact) -> bool {
    matches!(fact, DeviceFact::Deviation(_) | DeviceFact::Workaround(_))
}

/// Between a device's checks thread and the drops of the device's handles (the module docs).
pub(crate) struct Gate {
    /// Whether the checks thread may still call: set as the device is assembled with checks left
    /// to it, cleared under the lock as the thread ends. A drop that reads it cleared takes no lock.
    active: AtomicBool,
    state: Mutex<GateState>,
    /// Signalled as a call ends.
    ended: Condvar,
}

/// What the gate's lock guards.
#[derive(Default)]
struct GateState {
    /// The checks thread, while a call of its holds the device.
    calling: Option<ThreadId>,
    /// The device's last handle goes, or went: no call begins.
    stopped: bool,
}

impl Gate {
    /// The gate of a device whose checks thread runs (`active`) or none.
    pub(crate) fn new(active: bool) -> Gate {
        Gate {
            active: AtomicBool::new(active),
            state: Mutex::new(GateState::default()),
            ended: Condvar::new(),
        }
    }

    /// Whether a handle's drop goes through the gate ([`Gate::let_go`]).
    pub(crate) fn active(&self) -> bool {
        self.active.load(Ordering::Acquire)
    }

    /// A handle's share of the device, `shared`, goes: dropped under the lock where another
    /// handle stays (no other drop runs meanwhile, so it is not the last), or where the calling
    /// thread itself lets go of one within its call (its hold stays); where only a call's hold
    /// stays, after that call ended, no other beginning; and the last, out of the lock, so the
    /// device's teardown runs here.
    pub(crate) fn let_go<T>(&self, shared: Arc<T>) {
        let mut state = lock(&self.state);
        loop {
            let others = Arc::strong_count(&shared) - 1;
            let own = state.calling == Some(std::thread::current().id());
            if own || others > usize::from(state.calling.is_some()) {
                drop(shared);
                return;
            }
            state.stopped = true;
            if state.calling.is_none() {
                break;
            }
            state = (self.ended.wait(state)).unwrap_or_else(PoisonError::into_inner);
        }
        drop(state);
        drop(shared);
    }

    /// Begin a call holding what `hold` makes of the device (its upgrade): `None` once the
    /// device's last handle goes, or the upgrade finds none.
    pub(crate) fn enter<H>(&self, hold: impl FnOnce() -> Option<H>) -> Option<H> {
        let mut state = lock(&self.state);
        if state.stopped {
            return None;
        }
        let held = hold()?;
        state.calling = Some(std::thread::current().id());
        Some(held)
    }

    /// End a call, its hold `shared` let go of under the lock: never the last, since a last
    /// handle's drop waits for the call ([`Gate::let_go`]).
    pub(crate) fn leave<T>(&self, shared: Arc<T>) {
        let mut state = lock(&self.state);
        drop(shared);
        state.calling = None;
        drop(state);
        self.ended.notify_all();
    }

    /// The checks thread ends: no call follows, so drops take no lock from here on.
    pub(crate) fn end(&self) {
        let _state = lock(&self.state);
        self.active.store(false, Ordering::Release);
    }

    /// Whether the device's last handle goes or went (tests: a drop waiting for a call).
    #[cfg(test)]
    pub(crate) fn stopped(&self) -> bool {
        lock(&self.state).stopped
    }
}

/// A device's checks as they settle (the module docs).
pub(crate) struct Live {
    /// Between the checks thread and the drops of the device's handles.
    pub(crate) gate: Arc<Gate>,
    /// The feature status as it stands.
    status: Mutex<FeatureStatus>,
    /// Each runtime fact whose check the checks thread runs, and whether the device takes it now.
    switches: Vec<(DeviceFact, AtomicBool)>,
    /// The checks thread, while it runs.
    thread: Mutex<Option<JoinHandle<()>>>,
    /// What the running check made, released last first.
    left: Mutex<Vec<Made>>,
}

impl Live {
    /// The checks of a device whose status as made is `status`, `pending` left to its checks
    /// thread: each runtime fact of those switches from its row's answer as made.
    pub(crate) fn new(status: FeatureStatus, pending: &[&'static Check]) -> Live {
        let switches = (pending.iter())
            .flat_map(|check| check.facts)
            .filter(|&&fact| !read_by_translation(fact))
            .map(|&fact| {
                let holds = status.fact(fact).is_some_and(|row| row.holds);
                (fact, AtomicBool::new(holds))
            })
            .collect();
        Live {
            gate: Arc::new(Gate::new(!pending.is_empty())),
            status: Mutex::new(status),
            switches,
            thread: Mutex::new(None),
            left: Mutex::new(Vec::new()),
        }
    }

    /// Whether the device takes `fact` now, where its check runs on the checks thread; `None`
    /// for a fact the device holds as made.
    pub(crate) fn holds(&self, fact: DeviceFact) -> Option<bool> {
        (self.switches.iter())
            .find(|(of, _)| *of == fact)
            .map(|(_, holds)| holds.load(Ordering::Relaxed))
    }

    /// The feature status as it stands.
    pub(crate) fn status(&self) -> FeatureStatus {
        lock(&self.status).clone()
    }

    /// The checks thread started: `handle`.
    pub(crate) fn started(&self, handle: JoinHandle<()>) {
        *lock(&self.thread) = Some(handle);
    }

    /// The device's teardown: the checks thread is joined, which begins no call once the device's
    /// last handle went ([`Gate`]) and ends at its next, answering nothing (unless this is it,
    /// which a teardown never is while the thread calls).
    pub(crate) fn stop(&self) {
        let handle = lock(&self.thread).take();
        if let Some(handle) = handle {
            if handle.thread().id() != std::thread::current().id() {
                let _ = handle.join();
            }
        }
    }

    /// Keep `made`, made by the running check.
    pub(crate) fn keep(&self, made: Made) {
        lock(&self.left).push(made);
    }

    /// What the running check made past the first `own` (its kit's own objects), handed back.
    pub(crate) fn take_from(&self, own: usize) -> Vec<Made> {
        let mut left = lock(&self.left);
        let at = own.min(left.len());
        left.split_off(at)
    }

    /// Release what the checks left made: the device's teardown, its checks thread stopped
    /// ([`Live::stop`]) and the device idle.
    ///
    /// # Safety
    ///
    /// `device` is the device that made each object, idle.
    pub(crate) unsafe fn release(&self, device: &ash::Device) {
        let left: Vec<Made> = lock(&self.left).drain(..).collect();
        for made in left.into_iter().rev() {
            // SAFETY: the caller's: the device is idle, so no work uses the object, and each is
            // in the list once.
            unsafe { made.release(device) };
        }
    }

    /// `check`'s answers `found`, from `source`, on `device`, after its making: a runtime fact's
    /// row and switch take its answer, a translation fact's row names it for the next device (the
    /// module docs).
    pub(crate) fn answered(
        &self,
        device: &Device,
        check: &Check,
        found: &DeviceChecks,
        source: CheckSource,
    ) {
        let mut now = device.properties().description.clone();
        for (fact, answer) in found.iter() {
            now.checks.insert(fact, answer.clone());
        }
        let mut fresh = now.fact_status(true);
        fresh.extend(crate::driver_defects::status_rows(&now, true));
        let mut status = lock(&self.status);
        if let CheckSource::Ran(_) = source {
            status.checks_run += 1;
        }
        if let Some(run) = (status.checks.iter_mut()).find(|run| run.facts == check.facts) {
            run.source = source;
        }
        for &fact in check.facts {
            let Some(answer) = found.get(fact) else {
                continue;
            };
            let Some(row) = status.facts.iter_mut().find(|row| row.fact == fact) else {
                continue;
            };
            if read_by_translation(fact) {
                row.check = CheckOf::NextDevice(answer.clone());
                continue;
            }
            if let Some(taken) = fresh.iter().find(|taken| taken.fact == fact) {
                *row = taken.clone();
            }
            if let Some((_, holds)) = self.switches.iter().find(|(of, _)| *of == fact) {
                holds.store(row.holds, Ordering::Relaxed);
            }
        }
    }

    /// Every check left to the thread answered: the status settled as `settled`, the cache not
    /// used where `cache_error` says why; the status then.
    pub(crate) fn settled(&self, settled: Settled, cache_error: Option<String>) -> FeatureStatus {
        let mut status = lock(&self.status);
        status.settled = settled;
        if cache_error.is_some() {
            status.cache_error = cache_error;
        }
        status.clone()
    }
}

impl Device {
    /// Whether the device takes the driver fact `fact` now: as its check answered on the checks
    /// thread where it ran there ([`Live`]), else as the device was made
    /// (`DeviceProperties::driver_facts`).
    pub(crate) fn driver_fact(&self, fact: DriverFact) -> bool {
        let live = &self.shared().checks;
        live.holds(DeviceFact::Fact(fact))
            .unwrap_or_else(|| self.properties().driver_facts.contains(&fact))
    }

    /// Whether the device takes the driver defect `defect` now: as its check answered on the
    /// checks thread where it ran there ([`Live`]), else as the device was made
    /// (`DeviceProperties::driver_defects`).
    pub(crate) fn driver_defect(&self, defect: RuntimeDriverDefect) -> bool {
        let live = &self.shared().checks;
        live.holds(DeviceFact::Defect(defect))
            .unwrap_or_else(|| self.properties().driver_defects.contains(&defect))
    }

    /// How the device takes each driver fact now: its feature status as made
    /// ([`DeviceProperties::feature_status`](crate::DeviceProperties::feature_status)), each
    /// answer its checks thread found since in it ([`Settled::Running`] until the last, then
    /// [`Settled::InBackground`]). No Metal counterpart: Metal's driver answers as Metal.
    pub fn feature_status(&self) -> FeatureStatus {
        self.shared().checks.status()
    }
}
