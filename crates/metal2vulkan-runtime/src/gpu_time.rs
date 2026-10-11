//! A command buffer's GPU and kernel times: Metal's `GPUStartTime`, `GPUEndTime`, `kernelStartTime`
//! and `kernelEndTime`, in seconds of the host's clock ([`host_time`]), as Metal gives them in the
//! host's time base.
//!
//! - **The host's clock.** `CLOCK_UPTIME_RAW` on macOS, the clock of `mach_absolute_time` and so
//!   of Metal's times and `CACurrentMediaTime`; `CLOCK_MONOTONIC_RAW` elsewhere on Unix, its
//!   counterpart (monotonic, never slewed, stopped while the system sleeps). Other platforms count
//!   the runtime's own monotonic clock, which no device can calibrate against.
//! - **GPU times.** The command buffer's first batch writes a timestamp at the top of its work,
//!   and its last batch one after the barrier that ends it (`command_buffer/submit.rs`), into a
//!   slot of two queries of one of the device's timestamp pools ([`Slot`]), which it takes at its
//!   commit.
//!   The completion thread reads the two once the command buffer completed, and gives the slot
//!   back. Their difference, in ticks of `timestampPeriod` nanoseconds, is exact on every device
//!   that writes timestamps.
//! - **Pools made as needed**. Each pool holds [`SLOTS`] slots. The device makes its first at its
//!   first timed commit, and another whenever a command buffer is committed while every slot of the
//!   ones it has is held, until the device refuses one: its own limit, which Vulkan states nowhere
//!   else (a pool is 4096 queries; the runtime bounds none, as Metal times every command buffer in
//!   flight). It releases them at its teardown. A commit asks the pool that gave the last slot
//!   first, and the others only when that one has none free: the pools are searched as one fills,
//!   not at every commit. On a driver whose pools are Metal counter sample buffers
//!   (`DriverFact::TimestampPoolsAreCounterSampleBuffers`: MoltenVK) it makes one, never more:
//!   Metal keeps at most 32 of those live per GPU, each of at most 4096 timestamps (measured
//!   through MoltenVK 1.4.1 on an M3 Max and an M4: the 33rd failed, as did a pool of 8192), so
//!   a pool per command buffer ran out with 32 command buffers in flight, and each pool more is
//!   one fewer for another device on the GPU. One pool of the most timestamps holds a slot for
//!   2048 command buffers in flight. A command buffer committed while every slot is held and no
//!   pool more can be made has no timestamps: its GPU times are refused as on a device that
//!   writes none, and the device announces it once (`Degradation::UntimedCommandBuffer`).
//!   Waiting for a slot instead could wait for work that needs the program to go on.
//! - **Calibrated where the device can.** With `VK_KHR_calibrated_timestamps` (or its
//!   `VK_EXT_` form) and a host time domain the runtime can read (`CLOCK_MONOTONIC_RAW` or
//!   `CLOCK_MONOTONIC`), a pair of GPU and host readings taken at the commit places each
//!   timestamp on the host's clock, within the pair's stated deviation: Metal's answer. A domain
//!   that is not the host's clock is moved onto it by reading both clocks at once (MoltenVK's
//!   `CLOCK_MONOTONIC_RAW` is macOS's, which counts sleep; Metal's does not).
//! - **Uncalibrated otherwise.** Without them, the end is placed at the host time the command
//!   buffer completed and the start the timestamps' difference before it: the duration exact,
//!   both times late by the completion's latency. The device announces it once
//!   (`Degradation::UncalibratedGpuTimes`).
//! - **None without timestamps.** A queue family whose `timestampValidBits` is 0 writes none: the
//!   times are refused (`Unsupported`, `Feature::GpuTimes`). So are a command buffer's that found
//!   no slot free.
//! - **Kernel times** are the host times at which the scheduler began submitting the command
//!   buffer's first batch and finished submitting its last.
//! - **No encoder, no timestamps**. A command buffer that encoded nothing and waits for no event or
//!   semaphore takes no slot and writes no timestamps. Its GPU start and end are both where its
//!   queue reached it: its kernel end time, or the GPU end of the queue's command buffer before it
//!   where that is later. Metal answers so for one, a zero duration (measured on an M4 and an M3
//!   Max): on its own, a microsecond after its `kernelEndTime`; committed behind a 2.5-ms fill, 112
//!   us after the fill's `GPUEndTime`, 2.7 ms after its own `kernelEndTime`; waiting for an event
//!   the host signals, 7 us after the signal, its `kernelEndTime` 6 us after it. The completion
//!   thread places each command buffer as it completes, in its queue's order: each completed
//!   command buffer moves its queue's latest GPU start and end. On MoltenVK the two timestamps cost
//!   such a command buffer about 150 us of latency (each is sampled in a blit pass of its own),
//!   against about 15 us without. One that waits keeps its timestamps: when the GPU passes a wait
//!   for another queue's signal is not a host time the runtime knows.
//! - **From the host, where the device is told to** (`DeviceOptions::gpu_times_from_host`,
//!   `RawDevice::gpu_times_from_host`; off by default), **or where its driver's timestamps wait
//!   for every queue's** (`DriverFact::TimestampsWaitForEveryQueue`: MoltenVK 1.4.1 on Apple's
//!   GPUs, whatever the options say). No command buffer writes timestamps. One with work answers as
//!   its GPU start its kernel end time, or its queue's latest GPU end before it where later, and as
//!   its GPU end the host time the runtime saw it complete (never before the start): a window its
//!   GPU work ran in. The device announces it once, at the first such answer
//!   (`Degradation::GpuTimesFromHost`). A device whose queue family writes no timestamps answers so
//!   too, instead of refusing.
//!
//!   Every answer is one Metal could give. Four rules hold by construction:
//!   1. **Metal's clock.** The times are seconds of the host's clock ([`host_time`]): on macOS
//!      `CLOCK_UPTIME_RAW`, the clock of `mach_absolute_time`, which Metal's `GPUStartTime` and
//!      `GPUEndTime` count on Apple's GPUs.
//!   2. **Inside the command buffer's life.** Its commit <= `kernelStartTime` <= `kernelEndTime`
//!      <= `GPUStartTime` <= `GPUEndTime` <= the moment its completion is observable: the end is
//!      read on the completion thread before the command buffer's status becomes `Completed`
//!      and before its completed handlers run, and the start is never later than the end.
//!   3. **A serial queue.** In a queue's commit order, which is the order its command buffers
//!      complete and are placed in (`command_buffer/completion.rs`), each one's GPU start is at
//!      or after the GPU end of the one before it, so starts and ends only move forward and no
//!      two command buffers of a queue overlap: one after another, as Metal's serial queue can
//!      run them. An encoder-less command buffer, which writes no timestamps on any device, is
//!      placed so too (a zero duration at its queue's latest end or its kernel end).
//!   4. **No counter samples.** The runtime samples no counters (`MTLCounterSampleBuffer`, the
//!      encoders' `sampleCountersInBuffer`, `MTLDevice::sampleTimestamps`:
//!      `RuntimeGap::CounterSampleBuffers`), so no sample can fall outside its command buffer's
//!      window or disagree with the host's clock.
//!
//!   How far they sit from Metal's: the start can be early by the time from the end of the
//!   submission to the GPU's start of the work, or late by the time the runtime took to see the
//!   queue's command buffer before it complete; the end is late by the time the runtime took to
//!   see this one complete; so the duration holds both latencies, not the GPU's own time.
//!
//! The pair a command buffer's times are placed by is the device's latest, taken at a commit at
//! most [`REFRESH`] before: one calibration per device per 100 ms, not one per commit. The GPU's
//! clock and the host's drift apart by parts per million, so a pair that old moves a time by
//! nanoseconds to microseconds, within what the calibration states for itself on the devices
//! measured.

use super::device::lock;
use super::release;
use ash::vk;
use std::collections::BTreeSet;
use std::ffi::{c_char, CStr, CString};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The longest a device's calibration pair serves the commits after it.
pub(crate) const REFRESH: Duration = Duration::from_millis(100);

/// The slots of each of a device's timestamp pools, two queries each: 4096 timestamps, the most
/// one Metal counter sample buffer holds (the module documentation).
pub(crate) const SLOTS: u32 = 2048;

/// The most timestamps MoltenVK made one Metal counter sample buffer of, on an M3 Max and an M4.
const SAMPLE_BUFFER_TIMESTAMPS: u32 = 4096;
const _: () = assert!(2 * SLOTS <= SAMPLE_BUFFER_TIMESTAMPS);

/// The host time now, in seconds of the clock the runtime's GPU and kernel times count (the
/// module documentation): `CLOCK_UPTIME_RAW` on macOS, which `mach_absolute_time` and
/// `CACurrentMediaTime` count, and `CLOCK_MONOTONIC_RAW` elsewhere on Unix.
pub fn host_time() -> f64 {
    host_nanos() as f64 / 1e9
}

/// The host time now, in nanoseconds of the host's clock.
pub(crate) fn host_nanos() -> u64 {
    #[cfg(unix)]
    {
        read_clock(BASE).unwrap_or(0)
    }
    #[cfg(not(unix))]
    {
        use std::sync::OnceLock;
        use std::time::Instant;
        static START: OnceLock<Instant> = OnceLock::new();
        START.get_or_init(Instant::now).elapsed().as_nanos() as u64
    }
}

/// The host's clock.
#[cfg(target_os = "macos")]
const BASE: libc::clockid_t = libc::CLOCK_UPTIME_RAW;
#[cfg(all(unix, not(target_os = "macos")))]
const BASE: libc::clockid_t = libc::CLOCK_MONOTONIC_RAW;

/// `clock` now, in nanoseconds.
#[cfg(unix)]
fn read_clock(clock: libc::clockid_t) -> Option<u64> {
    let mut now = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `now` is a valid timespec for the call to fill.
    if unsafe { libc::clock_gettime(clock, &mut now) } != 0 {
        return None;
    }
    let seconds = u64::try_from(now.tv_sec).ok()?;
    let nanos = u64::try_from(now.tv_nsec).ok()?;
    Some(seconds * 1_000_000_000 + nanos)
}

/// The host clock behind a Vulkan host time domain, where the runtime can read it.
#[cfg(unix)]
fn domain_clock(domain: vk::TimeDomainKHR) -> Option<libc::clockid_t> {
    match domain {
        vk::TimeDomainKHR::CLOCK_MONOTONIC => Some(libc::CLOCK_MONOTONIC),
        vk::TimeDomainKHR::CLOCK_MONOTONIC_RAW => Some(libc::CLOCK_MONOTONIC_RAW),
        _ => None,
    }
}

/// The host time domain a device's calibration uses, from those it offers: the host's clock
/// where offered, else `CLOCK_MONOTONIC_RAW`, else `CLOCK_MONOTONIC`. `None` without the
/// device's own domain or a host domain the runtime can read.
pub(crate) fn host_domain(offered: &[vk::TimeDomainKHR]) -> Option<vk::TimeDomainKHR> {
    if !offered.contains(&vk::TimeDomainKHR::DEVICE) {
        return None;
    }
    #[cfg(unix)]
    {
        let preferred = [
            vk::TimeDomainKHR::CLOCK_MONOTONIC_RAW,
            vk::TimeDomainKHR::CLOCK_MONOTONIC,
        ];
        let base = preferred
            .into_iter()
            .find(|&domain| domain_clock(domain) == Some(BASE));
        base.into_iter()
            .chain(preferred)
            .find(|domain| offered.contains(domain))
    }
    #[cfg(not(unix))]
    {
        None
    }
}

/// How a device times its command buffers on the GPU.
#[derive(Clone, Copy)]
pub(crate) struct GpuClock {
    /// Nanoseconds per timestamp tick (`timestampPeriod`).
    pub(crate) period: f64,
    /// The bits a timestamp holds (`timestampValidBits` of the runtime's queue family): 1 to 64.
    pub(crate) valid_bits: u32,
    /// The calibration, where the device has one.
    pub(crate) calibration: Option<Calibration>,
}

/// A device's calibrated timestamps: its entry point and the host domain it reads.
#[derive(Clone, Copy)]
pub(crate) struct Calibration {
    pub(crate) get: vk::PFN_vkGetCalibratedTimestampsKHR,
    pub(crate) device: vk::Device,
    pub(crate) domain: vk::TimeDomainKHR,
}

/// A GPU timestamp and the host time, on the host's clock, at one moment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Pair {
    pub(crate) ticks: u64,
    pub(crate) host: u64,
}

impl GpuClock {
    /// The clock of a queue family with `valid_bits` timestamp bits and `period` nanoseconds
    /// per tick; `None` where the family writes no timestamps.
    pub(crate) fn new(
        valid_bits: u32,
        period: f32,
        calibration: Option<Calibration>,
    ) -> Option<GpuClock> {
        (valid_bits > 0).then_some(GpuClock {
            period: f64::from(period),
            valid_bits: valid_bits.min(64),
            calibration,
        })
    }

    /// A calibrated pair now, where the device has calibration and the readings succeed.
    pub(crate) fn pair(&self) -> Option<Pair> {
        let calibration = self.calibration?;
        let infos = [
            vk::CalibratedTimestampInfoKHR::default().time_domain(vk::TimeDomainKHR::DEVICE),
            vk::CalibratedTimestampInfoKHR::default().time_domain(calibration.domain),
        ];
        let mut values = [0u64; 2];
        let mut deviation = 0u64;
        #[cfg(unix)]
        let domain = domain_clock(calibration.domain)?;
        // SAFETY: the device is live (the caller holds it) and enabled the extension the entry
        // point belongs to; the arrays hold one value per info.
        let result = unsafe {
            (calibration.get)(
                calibration.device,
                infos.len() as u32,
                infos.as_ptr(),
                values.as_mut_ptr(),
                &mut deviation,
            )
        };
        if result != vk::Result::SUCCESS {
            return None;
        }
        #[cfg(unix)]
        {
            // The domain's clock onto the host's: both read at once, the host's around it.
            let host = if domain == BASE {
                values[1]
            } else {
                let first = read_clock(BASE)?;
                let during = read_clock(domain)?;
                let last = read_clock(BASE)?;
                let middle = first + last.saturating_sub(first) / 2;
                let offset = i128::from(middle) - i128::from(during);
                u64::try_from(i128::from(values[1]) + offset).ok()?
            };
            Some(Pair {
                ticks: values[0],
                host,
            })
        }
        #[cfg(not(unix))]
        {
            None
        }
    }

    /// The host time, in nanoseconds of the host's clock, of the timestamp `ticks`, by `pair`.
    pub(crate) fn place(&self, pair: Pair, ticks: u64) -> u64 {
        let delta = self.signed_ticks(ticks, pair.ticks);
        let nanos = (delta as f64 * self.period).round() as i128;
        u64::try_from(i128::from(pair.host) + nanos).unwrap_or(0)
    }

    /// The host times, in nanoseconds of the host's clock, of a command buffer's timestamps
    /// `start` and `end`: placed by `pair` where calibrated, else ending at `completed`, the
    /// host time its completion was seen, and starting their duration before.
    pub(crate) fn span(
        &self,
        pair: Option<Pair>,
        start: u64,
        end: u64,
        completed: u64,
    ) -> (u64, u64) {
        match pair {
            Some(pair) => (self.place(pair, start), self.place(pair, end)),
            None => (
                completed.saturating_sub(self.duration(start, end)),
                completed,
            ),
        }
    }

    /// The nanoseconds from timestamp `start` to `end`.
    pub(crate) fn duration(&self, start: u64, end: u64) -> u64 {
        let delta = self.signed_ticks(end, start).max(0);
        (delta as f64 * self.period).round() as u64
    }

    /// `to - from` in ticks, modulo the timestamps' bits, as the nearer of the two directions.
    fn signed_ticks(&self, to: u64, from: u64) -> i64 {
        if self.valid_bits >= 64 {
            return to.wrapping_sub(from) as i64;
        }
        let modulus = 1u64 << self.valid_bits;
        let difference = to.wrapping_sub(from) & (modulus - 1);
        if difference >= modulus / 2 {
            difference as i64 - modulus as i64
        } else {
            difference as i64
        }
    }
}

/// The extension's names, each with its commands' names: `VK_KHR_calibrated_timestamps`, then
/// the `VK_EXT_` form it promotes, whose commands vk.xml aliases to the same signatures.
#[derive(Debug)]
struct Names {
    extension: &'static CStr,
    domains: &'static CStr,
    get: &'static CStr,
}

/// The two forms, preferred first.
const FORMS: [Names; 2] = [
    Names {
        extension: vk::KHR_CALIBRATED_TIMESTAMPS_NAME,
        domains: c"vkGetPhysicalDeviceCalibrateableTimeDomainsKHR",
        get: c"vkGetCalibratedTimestampsKHR",
    },
    Names {
        extension: vk::EXT_CALIBRATED_TIMESTAMPS_NAME,
        domains: c"vkGetPhysicalDeviceCalibrateableTimeDomainsEXT",
        get: c"vkGetCalibratedTimestampsEXT",
    },
];

/// One form of the calibrated timestamps extension, which a device enables.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Extension(&'static Names);

impl Extension {
    /// The preferred form among `names`: the extensions a device lists, or enabled.
    pub(crate) fn among(names: &BTreeSet<CString>) -> Option<Extension> {
        FORMS
            .iter()
            .find(|form| names.contains(form.extension))
            .map(Extension)
    }

    /// Its name.
    pub(crate) fn name(self) -> &'static CStr {
        self.0.extension
    }
}

/// Why a device answers GPU times from the host's clock (the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FromHost {
    /// Its options or the consumer's adopted device say so.
    Asked,
    /// Its driver's timestamps wait for every queue's
    /// (`DriverFact::TimestampsWaitForEveryQueue`).
    Driver,
}

impl FromHost {
    /// Why, as the degradation's notice says it.
    pub(crate) fn because(self) -> &'static str {
        match self {
            FromHost::Asked => {
                "the device was made to answer them from the host's clock (gpu_times_from_host)"
            }
            FromHost::Driver => {
                "the driver samples each timestamp in a pass that waits for every queue's \
                 timestamped work (DriverFact::TimestampsWaitForEveryQueue)"
            }
        }
    }
}

/// How a device times its command buffers: its clock, `None` where its queue family writes no
/// timestamps, whether (and why) it answers GPU times from the host instead, its latest
/// calibration pair, and its timestamp pools, as made.
pub(crate) struct Timing {
    clock: Option<GpuClock>,
    from_host: Option<FromHost>,
    latest: Mutex<Option<(Instant, Pair)>>,
    pools: Mutex<Pools>,
}

/// A device's timestamp pools, made as its command buffers in flight need them, and the most it
/// may make, where it has a most (the module docs).
pub(crate) struct Pools {
    made: Vec<Pool>,
    /// The pool the last slot was taken from, asked first.
    at: usize,
    /// One on a driver whose pools are Metal counter sample buffers; else none until the device
    /// refuses a pool, then the pools it made.
    most: Option<usize>,
}

impl Pools {
    /// No pool yet, and at most `most` to be made where given.
    pub(crate) fn new(most: Option<usize>) -> Pools {
        Pools {
            made: Vec::new(),
            at: 0,
            most,
        }
    }

    /// A free slot of a pool made, the one that gave the last slot asked first, else of a pool
    /// `make` makes now where the device has made fewer than its most; `None` when every slot is
    /// held and no pool more can be made. A pool `make` cannot make (`None`) ends the making: the
    /// device makes no more.
    pub(crate) fn take(&mut self, make: impl FnOnce() -> Option<vk::QueryPool>) -> Option<Slot> {
        let count = self.made.len();
        for step in 0..count {
            let at = (self.at + step) % count;
            if let Some(slot) = self.made[at].take() {
                self.at = at;
                return Some(slot);
            }
        }
        if self.most.is_some_and(|most| count >= most) {
            return None;
        }
        let Some(queries) = make() else {
            self.most = Some(count);
            return None;
        };
        self.made.push(Pool::new(queries, SLOTS));
        self.at = count;
        self.made.last()?.take()
    }

    /// The slots of the pools made.
    pub(crate) fn slots(&self) -> usize {
        self.made.len() * SLOTS as usize
    }

    /// The pools made.
    pub(crate) fn queries(&self) -> impl Iterator<Item = vk::QueryPool> + '_ {
        self.made.iter().map(|pool| pool.queries)
    }
}

/// One of a device's timestamp query pools, of [`SLOTS`] slots, and the slots no command buffer
/// holds.
pub(crate) struct Pool {
    queries: vk::QueryPool,
    free: Arc<Mutex<Vec<u32>>>,
}

impl Pool {
    /// The `slots` slots of `queries`, a pool of `2 * slots` timestamp queries, every one free.
    pub(crate) fn new(queries: vk::QueryPool, slots: u32) -> Pool {
        Pool {
            queries,
            // Popped from the end: slot 0 first.
            free: Arc::new(Mutex::new((0..slots).rev().collect())),
        }
    }

    /// A free slot, `None` when every one is held.
    pub(crate) fn take(&self) -> Option<Slot> {
        let index = lock(&self.free).pop()?;
        Some(Slot {
            queries: self.queries,
            index,
            free: Arc::clone(&self.free),
        })
    }
}

/// A slot of one of its device's timestamp pools: two queries, a command buffer's start and end.
/// The command buffer holds it from its commit until the completion thread read them; dropping
/// it gives it back.
#[derive(Debug)]
pub(crate) struct Slot {
    queries: vk::QueryPool,
    index: u32,
    free: Arc<Mutex<Vec<u32>>>,
}

impl Slot {
    /// The timestamp pool it is of.
    pub(crate) fn queries(&self) -> vk::QueryPool {
        self.queries
    }

    /// Its first query, the start; the end is the next.
    pub(crate) fn first(&self) -> u32 {
        self.index * 2
    }
}

impl Drop for Slot {
    fn drop(&mut self) {
        lock(&self.free).push(self.index);
    }
}

impl Timing {
    /// The timing of `device` of `physical`, made from `instance` through `entry`, whose runtime
    /// queue family has `valid_bits` timestamp bits, ticking every `period` nanoseconds, with
    /// `extension` enabled where it is: calibrated when the device offers its own time domain and
    /// one of the host's the runtime reads ([`host_domain`]). With `from_host`, it answers GPU
    /// times from the host and writes no timestamps (the module docs), for that reason. With
    /// `scarce_pools` (`DriverFact::TimestampPoolsAreCounterSampleBuffers`) it makes one
    /// timestamp pool at most, else as many as its command buffers in flight need until the
    /// device refuses one.
    ///
    /// # Safety
    ///
    /// `device` is a live device of `physical`, made from `instance` with `extension` enabled,
    /// and the loader serves `vkGetDeviceProcAddr` for it (its loader word is in place).
    pub(crate) unsafe fn new(
        entry: &ash::Entry,
        instance: &ash::Instance,
        physical: vk::PhysicalDevice,
        device: &ash::Device,
        valid_bits: u32,
        period: f32,
        extension: Option<Extension>,
        from_host: Option<FromHost>,
        scarce_pools: bool,
    ) -> Timing {
        // SAFETY: the caller's.
        let calibration = extension.and_then(|extension| unsafe {
            calibration(entry, instance, physical, device, extension.0)
        });
        Timing {
            clock: GpuClock::new(valid_bits, period, calibration),
            from_host,
            latest: Mutex::new(None),
            pools: Mutex::new(Pools::new(scarce_pools.then_some(1))),
        }
    }

    /// A slot for a command buffer committed now, from the device's pools, a pool made where
    /// every slot of the ones made is held ([`Pools::take`]); `None` where the device writes no
    /// timestamps, or every slot is held and no pool more can be made.
    ///
    /// # Safety
    ///
    /// `device` is the live device of this timing.
    pub(crate) unsafe fn slot(&self, device: &ash::Device) -> Option<Slot> {
        self.clock.as_ref()?;
        lock(&self.pools).take(|| {
            let info = vk::QueryPoolCreateInfo::default()
                .query_type(vk::QueryType::TIMESTAMP)
                .query_count(2 * SLOTS);
            // SAFETY: the caller's; `info` is valid.
            unsafe { device.create_query_pool(&info, None) }.ok()
        })
    }

    /// The slots of the timestamp pools the device made.
    pub(crate) fn slots(&self) -> usize {
        lock(&self.pools).slots()
    }

    /// Release the device's timestamp pools, at its teardown, inside its section.
    ///
    /// # Safety
    ///
    /// `device` is the live device of this timing, and none of its command buffers is pending.
    pub(crate) unsafe fn release(&self, device: &ash::Device) {
        for queries in lock(&self.pools).queries() {
            // SAFETY: the caller's.
            unsafe { release::object(device, queries) };
        }
    }

    /// The device's clock, where its queue family writes timestamps.
    pub(crate) fn clock(&self) -> Option<&GpuClock> {
        self.clock.as_ref()
    }

    /// Whether the device answers GPU times from the host, writing no timestamps (the module
    /// docs).
    pub(crate) fn answers_from_host(&self) -> bool {
        self.from_host.is_some()
    }

    /// Why the device answers GPU times from the host; `None` where it does not.
    pub(crate) fn host_clock_reason(&self) -> Option<FromHost> {
        self.from_host
    }

    /// The device's latest calibration pair, taken now when the last is [`REFRESH`] old or
    /// there is none; `None` where the device has no calibration or the reading fails.
    pub(crate) fn pair(&self) -> Option<Pair> {
        let clock = self.clock.as_ref()?;
        clock.calibration?;
        let mut latest = lock(&self.latest);
        if let Some((at, pair)) = *latest {
            if at.elapsed() < REFRESH {
                return Some(pair);
            }
        }
        let pair = clock.pair()?;
        *latest = Some((Instant::now(), pair));
        Some(pair)
    }
}

/// The calibration of `device` through `names`, where it offers its own time domain and one of
/// the host's the runtime reads.
///
/// # Safety
///
/// As [`Timing::new`]'s, `names` the form enabled.
unsafe fn calibration(
    entry: &ash::Entry,
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
    device: &ash::Device,
    names: &Names,
) -> Option<Calibration> {
    // SAFETY: the instance is live; a command it does not have is null.
    let domains =
        unsafe { entry.get_instance_proc_addr(instance.handle(), names.domains.as_ptr()) }?;
    // SAFETY: the loader gave the command of this name, of this signature.
    let domains = unsafe {
        std::mem::transmute::<
            unsafe extern "system" fn(),
            vk::PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsKHR,
        >(domains)
    };
    let mut count = 0u32;
    // SAFETY: `physical` is the instance's; the first call counts.
    let counted = unsafe { domains(physical, &mut count, std::ptr::null_mut()) };
    if counted != vk::Result::SUCCESS {
        return None;
    }
    let mut offered = vec![vk::TimeDomainKHR::default(); count as usize];
    // SAFETY: as above, into `count` elements.
    let listed = unsafe { domains(physical, &mut count, offered.as_mut_ptr()) };
    if listed != vk::Result::SUCCESS && listed != vk::Result::INCOMPLETE {
        return None;
    }
    offered.truncate(count as usize);
    let domain = host_domain(&offered)?;
    let name: *const c_char = names.get.as_ptr();
    // SAFETY: the caller's: the device is live and the loader serves the call.
    let get = unsafe { (instance.fp_v1_0().get_device_proc_addr)(device.handle(), name) }?;
    Some(Calibration {
        // SAFETY: the device gave the command of this name, of this signature.
        get: unsafe {
            std::mem::transmute::<unsafe extern "system" fn(), vk::PFN_vkGetCalibratedTimestampsKHR>(
                get,
            )
        },
        device: device.handle(),
        domain,
    })
}

/// A command buffer's times, in its state: host times in nanoseconds of the host's clock, 0
/// until known.
#[derive(Debug, Default)]
pub(crate) struct Times {
    /// Its slot of the device's timestamp pool, from its commit until the completion thread read
    /// its timestamps.
    pub(crate) slot: Option<Slot>,
    /// What its timestamps gave.
    pub(crate) stamped: Stamped,
    /// The device's calibration pair at its commit, where the device has one.
    pub(crate) pair: Option<Pair>,
    /// The scheduler began submitting its first batch.
    pub(crate) kernel_start: u64,
    /// The scheduler finished submitting its last batch.
    pub(crate) kernel_end: u64,
    /// The completion thread saw its last batch complete.
    pub(crate) completed: u64,
    /// Its GPU start and end, once read.
    pub(crate) gpu: Option<(u64, u64)>,
    /// Its GPU start and end where it wrote no timestamps: placed among its queue's command
    /// buffers by the completion thread as it completed.
    pub(crate) placed: Option<(u64, u64)>,
}

/// What a command buffer's timestamps gave.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Stamped {
    /// Nothing yet: it has not completed, or the device writes no timestamps.
    #[default]
    Unread,
    /// It encoded nothing and waits for nothing, and writes none: its GPU start and end are
    /// where its queue reached it, as Metal's (the module docs).
    NoWork,
    /// Its device answers GPU times from the host, and it writes none (the module docs).
    FromHost,
    /// Every slot of the device's pools was held at its commit, and no pool more could be made:
    /// it has none.
    NoSlot,
    /// Read once it completed: its start and end ticks, where the GPU wrote both.
    Read(Option<(u64, u64)>),
    /// The read failed with this result.
    Failed(vk::Result),
}

/// `nanos` of the host's clock in seconds; 0 stays 0.
pub(crate) fn seconds(nanos: u64) -> f64 {
    nanos as f64 / 1e9
}
