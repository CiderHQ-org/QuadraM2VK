//! Device tests of the device's own checks of its driver (`metal2vulkan_runtime::FeatureStatus`)
//! on the device the cfg selects, under the validation layer.
//!
//! The device is made six times, each checked making over a device-checks directory and a shader
//! cache of the test's own:
//!
//! 1. under `DeviceChecksPolicy::Distrusted`, first, so what the process's first device costs
//!    (loading the driver, its first instance) is not counted against the checks;
//! 2. distrusted again, cold: the baseline making, whose status is the curated tables' alone;
//! 3. checked and cold with `DeviceChecksOptions::wait`, so every check runs as it is made;
//! 4. checked and warm over the third's directory, every answer read from the cache;
//! 5. checked and cold by default: the making runs the checks of `Settle::AtCreation`, and the
//!    device's checks thread the rest, until the status settles (`Settled::InBackground`);
//! 6. checked over the fifth's directory, so it takes every answer the checks thread found.
//!
//! Each checked fact's answer at the third making is held against the tables (the agreement):
//!
//! - **A listed row** (the tables give the fact on this setup, measured): on a measured identity
//!   the check must answer and agree (a deviation or answer fact taken, a defect worked around),
//!   unless it does not run on the device (a feature it needs not enabled), which is printed. On
//!   a measured identity no check may be left unanswered.
//! - **An unlisted or assumed row** (no measured row names the setup; an assumed one is a rule
//!   for every setup of a kind, `Decided::Assumed`, such as a workaround by the driver's
//!   identity): a check that answers otherwise is a finding the tables lack, printed `NEW`, not a
//!   failure: the tables list measurements, and a setup measured for other facts may simply not
//!   have been measured for this one.
//! - **An unmeasured identity** (the 7900 XTX and the 780M on Mesa 26.0.8, an NVIDIA release
//!   past the measured one): what the device takes is printed alone.
//!
//! The checks thread's answers must be the waited making's: a runtime fact's row takes its answer
//! on the running device, a translation fact's row keeps the tables' and names the answer for the
//! next device, which takes it. Every making, and the checks thread's work, runs under the
//! validation layer, and any message it reports fails the lane.
//!
//! It prints the cost against the baseline making: what the waited, warm and default makings add,
//! the checks thread's time, the checks' own time split by settle class (`AtCreation`,
//! `Background`), and each check's wall time, GPU time, longest submission and pipelines. Debug
//! timings prove nothing: the cost is read from a release build.

use super::{validated_device_with, Messages, ScratchCache};
use crate::gpu::TEST_DEVICE;
use metal2vulkan::device::{
    CheckAnswer, CheckOf, Decided, DeviceChecksPolicy, DeviceFact, FactStatus, CHECKED,
};
use metal2vulkan_runtime::{
    CheckSource, Device, DeviceChecksOptions, FeatureStatus, Settle, Settled,
};
use std::path::Path;
use std::time::{Duration, Instant};

/// How the lane makes a device: its policy distrusting checks, or trusting them and waiting for
/// every check at its making, or by default.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Checks {
    Distrusted,
    Waited,
    Background,
}

/// A device of the lane's options whose shader cache is `cache` and whose checks keep their
/// answers in `checks`, made as `how` says, and how long its making took. The validation layer
/// reports to `messages`: a check's use the layer finds invalid fails the lane here, as it is
/// made.
fn made(messages: &Messages, cache: &Path, checks: &Path, how: Checks) -> (Device, Duration) {
    let started = Instant::now();
    let device = validated_device_with(TEST_DEVICE, messages, cache, |options| {
        let mut kept = DeviceChecksOptions::default();
        kept.directory = Some(checks.to_owned());
        kept.wait = how == Checks::Waited;
        options.device_checks = kept;
        // The lane's policy is `exact_answers()`, which distrusts checks: trust them here.
        options.policy.device_checks = match how {
            Checks::Distrusted => DeviceChecksPolicy::Distrusted,
            Checks::Waited | Checks::Background => DeviceChecksPolicy::Trusted,
        };
    })
    .unwrap_or_else(|error| panic!("{error}"));
    let wall = started.elapsed();
    reported_nothing(messages, &device, "as the device was made");
    (device, wall)
}

/// Fail where the validation layer reported to `messages` about `device`, `when`.
fn reported_nothing(messages: &Messages, device: &Device, when: &str) {
    let failing = messages.take();
    assert!(
        failing.is_empty(),
        "{}: the validation layer reported, {when}: {failing:#?}",
        device.name()
    );
}

/// `device`'s feature status once its checks thread has settled it (`Settled::InBackground`),
/// within a minute, and how long that took from the call.
fn settled(device: &Device) -> (FeatureStatus, Duration) {
    let started = Instant::now();
    loop {
        let status = device.feature_status();
        if matches!(status.settled, Settled::InBackground { .. }) {
            return (status, started.elapsed());
        }
        assert!(
            started.elapsed() < Duration::from_secs(60),
            "{}: the checks thread did not settle in a minute: {:?}",
            device.name(),
            status.settled
        );
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// `row` as one line.
fn row_line(row: &FactStatus) -> String {
    format!(
        "{} ({}): holds {}, {:?}, check {:?}",
        row.fact.name(),
        row.fact.kind(),
        row.holds,
        row.decided,
        row.check
    )
}

/// `answer`'s kind, its detail left out: what two runs of one check must agree on.
fn kind(answer: &CheckAnswer) -> &'static str {
    match answer {
        CheckAnswer::Metals => "metals",
        CheckAnswer::NotMetals(_) => "not metals",
        CheckAnswer::NotRun(_) => "not run",
        CheckAnswer::Unanswered(_) => "unanswered",
        _ => "another answer",
    }
}

/// `duration` in milliseconds, or `none`.
fn ms(duration: Option<Duration>) -> String {
    duration.map_or("none".to_owned(), |duration| {
        format!("{:.3} ms", duration.as_secs_f64() * 1000.0)
    })
}

/// `wall` past `baseline`, in milliseconds (negative where it took less).
fn added(wall: Duration, baseline: Duration) -> f64 {
    (wall.as_secs_f64() - baseline.as_secs_f64()) * 1000.0
}

/// The checks' wall time of `status` by settle class, and each check's line.
fn costs(status: &FeatureStatus) -> (Duration, Duration, Vec<String>) {
    let (mut at_creation, mut background) = (Duration::ZERO, Duration::ZERO);
    let mut lines = Vec::new();
    for check in &status.checks {
        let names: Vec<String> = check.facts.iter().map(|fact| fact.name()).collect();
        let line = match check.source {
            CheckSource::Ran(cost) => {
                match check.settle {
                    Settle::AtCreation => at_creation += cost.wall,
                    _ => background += cost.wall,
                }
                format!(
                    "{names:?} {:?}: wall {:.2} ms, gpu {}, longest {}, {} pipelines",
                    check.settle,
                    cost.wall.as_secs_f64() * 1000.0,
                    ms(cost.gpu),
                    ms(cost.longest),
                    cost.pipelines
                )
            }
            source => format!("{names:?} {:?}: {source:?}", check.settle),
        };
        lines.push(line);
    }
    (at_creation, background, lines)
}

/// Every check runs cold, answers from the cache warm, and agrees with the curated tables where
/// they list its fact on a measured identity; the distrusted device takes the tables alone; the
/// checks thread finds what the waited making found. Prints the agreement and the cost (the module
/// docs).
#[test]
fn the_device_s_checks_agree_with_the_measured_tables() {
    let messages = Messages::default();
    let scratch = || ScratchCache::new().unwrap();
    let (first_cache, first_checks) = (scratch(), scratch());
    let (first, first_wall) = made(
        &messages,
        first_cache.path(),
        first_checks.path(),
        Checks::Distrusted,
    );
    drop(first);
    let (baseline_cache, baseline_checks) = (scratch(), scratch());
    let (distrusted, baseline) = made(
        &messages,
        baseline_cache.path(),
        baseline_checks.path(),
        Checks::Distrusted,
    );
    let tables = distrusted.properties().feature_status.clone();
    drop(distrusted);
    let (cache, checks) = (scratch(), scratch());
    let (device, cold) = made(&messages, cache.path(), checks.path(), Checks::Waited);
    let name = device.name().to_owned();
    let properties = device.properties().clone();
    assert_eq!(
        device.feature_status(),
        properties.feature_status,
        "{name}: a waited device's status is its making's"
    );
    drop(device);
    let status = &properties.feature_status;
    let measured = properties.description.measured_identity();
    eprintln!(
        "device checks on {name} ({:?} {:#x}, {:04x}:{:04x}, measured identity: {measured}): \
         setup {}, {:?}, {} run, {} from the cache, cache error {:?}",
        properties.description.driver_id,
        properties.description.driver_version,
        properties.vendor_id,
        properties.device_id,
        status.setup,
        status.settled,
        status.checks_run,
        status.checks_from_cache,
        status.cache_error
    );
    for row in &status.facts {
        eprintln!("  {}", row_line(row));
    }
    assert!(
        matches!(status.settled, Settled::AtCreation { .. }),
        "{name}: {:?}",
        status.settled
    );
    assert!(status.checks_run > 0, "{name}: no check ran cold");
    assert_eq!(status.checks_from_cache, 0, "{name}");
    assert_eq!(status.cache_error, None, "{name}");

    assert_eq!(tables.settled, Settled::Distrusted, "{name}");
    assert_eq!(
        (tables.checks_run, tables.checks_from_cache),
        (0, 0),
        "{name}"
    );
    // Every row is judged before any fails the lane: an unanswered check of a measured identity
    // is a failure beside the disagreements, never one that stops the loop before them.
    let (mut disagreements, mut unanswered_measured) = (Vec::new(), Vec::new());
    for entry in CHECKED {
        let checked = status.fact(entry.fact).expect("a checked row");
        let table = tables.fact(entry.fact).expect("a table row");
        assert_eq!(table.check, CheckOf::Distrusted, "{name}: {:?}", entry.fact);
        let answered = matches!(
            checked.check,
            CheckOf::Answered(CheckAnswer::Metals | CheckAnswer::NotMetals(_))
        );
        let listed = matches!(table.decided, Decided::Listed(_));
        let line = format!(
            "{}: checked {}, tables {}",
            entry.fact.name(),
            row_line(checked),
            row_line(table)
        );
        let not_run = matches!(checked.check, CheckOf::Answered(CheckAnswer::NotRun(_)));
        let unanswered = matches!(checked.check, CheckOf::Answered(CheckAnswer::Unanswered(_)));
        match (measured, listed) {
            _ if not_run => eprintln!("  not run {line}"),
            (true, _) if unanswered => {
                eprintln!("  UNANSWERED {line}");
                unanswered_measured.push(line);
            }
            (true, true) if !answered || checked.holds != table.holds => {
                eprintln!("  DISAGREES {line}");
                disagreements.push(line);
            }
            (true, false) if answered && checked.holds != table.holds => {
                eprintln!("  NEW {line}");
            }
            (true, _) => eprintln!("  agrees {line}"),
            (false, _) => eprintln!("  unmeasured {line}"),
        }
    }
    assert!(
        disagreements.is_empty() && unanswered_measured.is_empty(),
        "{name}: the checks disagree with the tables: {disagreements:#?}; unanswered on a measured \
         identity: {unanswered_measured:#?}"
    );

    // A check whose every answer is kept reads them warm; one left unanswered runs again.
    let unanswered = |facts: &[DeviceFact]| {
        facts.iter().any(|&fact| {
            status.fact(fact).is_some_and(|row| {
                matches!(row.check, CheckOf::Answered(CheckAnswer::Unanswered(_)))
            })
        })
    };
    let again = (status.checks.iter())
        .filter(|check| unanswered(&check.facts))
        .count();
    let kept = status.checks.len() - again;
    let (warm_device, warm) = made(&messages, cache.path(), checks.path(), Checks::Waited);
    let warm_status = warm_device.properties().feature_status.clone();
    drop(warm_device);
    eprintln!(
        "device checks warm on {name}: {:?}, {} run, {} from the cache",
        warm_status.settled, warm_status.checks_run, warm_status.checks_from_cache
    );
    assert_eq!(warm_status.checks_run as usize, again, "{name}: warm");
    assert_eq!(warm_status.checks_from_cache as usize, kept, "{name}: warm");
    if again == 0 {
        assert!(
            matches!(warm_status.settled, Settled::FromCache { .. }),
            "{name}: warm {:?}",
            warm_status.settled
        );
    }
    for entry in CHECKED {
        let cold_row = status.fact(entry.fact).expect("a row");
        let warm_row = warm_status.fact(entry.fact).expect("a row");
        assert_eq!(
            (cold_row.holds, &cold_row.decided),
            (warm_row.holds, &warm_row.decided),
            "{name}: {:?}",
            entry.fact
        );
    }

    // By default the checks thread answers what the making leaves to it, as the waited making
    // did; the setup's next device takes every answer.
    let (background_cache, background_checks) = (scratch(), scratch());
    let (running, running_wall) = made(
        &messages,
        background_cache.path(),
        background_checks.path(),
        Checks::Background,
    );
    let as_made = running.properties().feature_status.clone();
    let left = (as_made.checks.iter())
        .filter(|check| check.settle == Settle::Background)
        .count();
    match left {
        0 => assert!(
            matches!(as_made.settled, Settled::AtCreation { .. }),
            "{name}: {:?}",
            as_made.settled
        ),
        _ => assert!(
            matches!(as_made.settled, Settled::Running { .. }),
            "{name}: {:?}",
            as_made.settled
        ),
    }
    let (thread_status, thread_wall) = match left {
        0 => (as_made.clone(), Duration::ZERO),
        _ => settled(&running),
    };
    reported_nothing(&messages, &running, "as the checks thread ran");
    drop(running);
    eprintln!(
        "device checks in the background on {name}: {} left to the thread, {:?}, {} run",
        left, thread_status.settled, thread_status.checks_run
    );
    assert_eq!(thread_status.checks_run, status.checks_run, "{name}");
    assert_eq!(thread_status.cache_error, None, "{name}");
    for entry in CHECKED {
        let waited = status.fact(entry.fact).expect("a row");
        let found = thread_status.fact(entry.fact).expect("a row");
        let table = tables.fact(entry.fact).expect("a table row");
        let line = format!("waited {}, found {}", row_line(waited), row_line(found));
        let CheckOf::Answered(answer) = &waited.check else {
            panic!("{name}: {line}");
        };
        let settle = (status.checks.iter())
            .find(|check| check.facts.contains(&entry.fact))
            .map(|check| check.settle);
        let next = matches!(
            entry.fact,
            DeviceFact::Deviation(_) | DeviceFact::Workaround(_)
        ) && settle == Some(Settle::Background);
        match (&found.check, next) {
            (CheckOf::NextDevice(later), true) => {
                assert_eq!(kind(later), kind(answer), "{name}: {line}");
                let as_made_row = as_made.fact(entry.fact).expect("a row");
                assert_eq!(found.holds, as_made_row.holds, "{name}: {line}");
                assert_eq!(found.holds, table.holds, "{name}: {line}");
            }
            (CheckOf::Answered(later), false) => {
                assert_eq!(kind(later), kind(answer), "{name}: {line}");
                assert_eq!(found.holds, waited.holds, "{name}: {line}");
            }
            _ => panic!("{name}: {line}"),
        }
    }
    let (next_device, next_wall) = made(
        &messages,
        background_cache.path(),
        background_checks.path(),
        Checks::Background,
    );
    let next_status = next_device.properties().feature_status.clone();
    drop(next_device);
    assert_eq!(next_status.checks_run as usize, again, "{name}: next");
    for entry in CHECKED {
        let waited = status.fact(entry.fact).expect("a row");
        if let CheckOf::Answered(CheckAnswer::Unanswered(_)) = waited.check {
            // Its check runs again, on the next device's checks thread where it settles there.
            continue;
        }
        let next = next_status.fact(entry.fact).expect("a row");
        assert_eq!(
            (waited.holds, &waited.decided),
            (next.holds, &next.decided),
            "{name}: the next device {:?}",
            entry.fact
        );
    }

    let (at_creation, background, lines) = costs(status);
    let (thread_at_creation, thread_background, _) = costs(&thread_status);
    eprintln!(
        "device checks cost on {name}: the process's first device {:.1} ms, the distrusted \
         making {:.1} ms; past it, the waited cold making {:+.1} ms, the warm making {:+.1} ms, \
         the default cold making {:+.1} ms and its checks thread {:.1} ms after it, the next \
         device {:+.1} ms; the checks at the waited making: AtCreation {:.2} ms, Background \
         {:.2} ms; at the default making AtCreation {:.2} ms, on its checks thread {:.2} ms",
        first_wall.as_secs_f64() * 1000.0,
        baseline.as_secs_f64() * 1000.0,
        added(cold, baseline),
        added(warm, baseline),
        added(running_wall, baseline),
        thread_wall.as_secs_f64() * 1000.0,
        added(next_wall, baseline),
        at_creation.as_secs_f64() * 1000.0,
        background.as_secs_f64() * 1000.0,
        thread_at_creation.as_secs_f64() * 1000.0,
        thread_background.as_secs_f64() * 1000.0
    );
    for line in lines {
        eprintln!("  {line}");
    }
}
