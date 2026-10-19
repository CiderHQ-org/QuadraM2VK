//! The reader order of the facts a device checks itself (`device::checks`): each reader takes a
//! checked fact as its check found it, its baseline until then, and the tables alone where the
//! policy distrusts checks.

use crate::device::checks::{
    decide, Baseline, CheckAnswer, CheckOf, Decided, DeviceChecks, DeviceFact, IfUnanswered,
    CHECKED,
};
use crate::device::profile::DriverWorkaround;
use crate::device::{Device, DeviceChecksPolicy, DeviceDescription, DevicePolicy, DriverFact};
use crate::error::{DriverDeviation, RuntimeDriverDefect};
use crate::spirv::testing::committed_descriptions;

const QUANTIZE: DeviceFact = DeviceFact::Deviation(DriverDeviation::MetalHalfQuantize);
const ZERO: DeviceFact = DeviceFact::Fact(DriverFact::RobustIndexFetchReadsZero);
const RESOLVE: DeviceFact =
    DeviceFact::Defect(RuntimeDriverDefect::DepthStencilResolveFirstSubresource);

/// The committed description of `stem`.
fn description(stem: &str) -> DeviceDescription {
    committed_descriptions()
        .expect("descriptions")
        .iter()
        .find(|(s, _)| s == stem)
        .map(|(_, d)| d.clone())
        .unwrap_or_else(|| panic!("no committed description {stem}"))
}

/// `description` with `answer` recorded for `fact`.
fn with(
    mut description: DeviceDescription,
    fact: DeviceFact,
    answer: CheckAnswer,
) -> DeviceDescription {
    description.checks.insert(fact, answer);
    description
}

fn unanswered() -> CheckAnswer {
    CheckAnswer::Unanswered("timed out".to_owned())
}

fn not_run() -> CheckAnswer {
    CheckAnswer::NotRun("not enabled".to_owned())
}

fn not_metals() -> CheckAnswer {
    CheckAnswer::NotMetals("word 3 read 0x1, Metal's 0x0".to_owned())
}

/// Every checked fact, for every answer and either baseline: Metal's answer takes an answer and
/// clears a defect, the other answer the opposite, an unanswered check the exact path (or the
/// baseline, for a fact whose workaround refuses what the baseline runs), and no answer, a check
/// that does not run, or distrusted checks, the baseline.
#[test]
fn every_checked_fact_follows_the_reader_order() {
    for entry in CHECKED {
        let fact = entry.fact;
        for listed in [false, true] {
            let baseline = match listed {
                true => Baseline::listed("a row".to_owned()),
                false => Baseline::unlisted(),
            };
            let held = |answer: Option<CheckAnswer>, trusted: bool| {
                let mut checks = DeviceChecks::new();
                if let Some(answer) = answer {
                    checks.insert(fact, answer);
                }
                let status = decide(fact, baseline.clone(), trusted.then_some(&checks));
                assert_eq!(status.fact, fact);
                status
            };
            let defect = fact.is_defect();
            let metals = held(Some(CheckAnswer::Metals), true);
            assert_eq!(
                (metals.holds, &metals.decided),
                (!defect, &Decided::CheckedMetals)
            );
            let not = held(Some(not_metals()), true);
            assert_eq!(
                (not.holds, &not.decided),
                (defect, &Decided::CheckedNotMetals)
            );
            let open = held(Some(unanswered()), true);
            match entry.if_unanswered {
                IfUnanswered::Exact => {
                    assert_eq!(open.holds, defect, "{fact:?}");
                    assert!(matches!(open.decided, Decided::Exact(_)), "{fact:?}");
                }
                IfUnanswered::Baseline => {
                    assert_eq!(open.holds, listed, "{fact:?}");
                    assert_eq!(open.decided, baseline.decided, "{fact:?}");
                }
            }
            assert_eq!(open.check, CheckOf::Answered(unanswered()));
            for (status, check) in [
                (held(Some(not_run()), true), CheckOf::Answered(not_run())),
                (held(None, true), CheckOf::Pending),
                (held(Some(not_metals()), false), CheckOf::Distrusted),
                (held(Some(CheckAnswer::Metals), false), CheckOf::Distrusted),
            ] {
                assert_eq!(status.holds, listed, "{fact:?}");
                assert_eq!(status.decided, baseline.decided, "{fact:?}");
                assert_eq!(status.check, check, "{fact:?}");
            }
        }
    }
}

/// A fact no check decides takes its baseline whatever answer is recorded for it. The fact is the
/// first listed deviation no check decides, so a new check does not take it from the test.
#[test]
fn an_unchecked_fact_ignores_answers() {
    let fact = (crate::device::driver_facts::DEVIATIONS.iter())
        .map(|row| DeviceFact::Deviation(row.fact))
        .find(|fact| CHECKED.iter().all(|entry| entry.fact != *fact))
        .expect("a listed deviation no check decides");
    let mut checks = DeviceChecks::new();
    checks.insert(fact, CheckAnswer::Metals);
    let status = decide(fact, Baseline::unlisted(), Some(&checks));
    assert!(!status.holds);
    assert_eq!(status.check, CheckOf::None);
}

/// The answers keep the facts' order, and a second answer replaces the first.
#[test]
fn checks_hold_one_answer_per_fact_in_order() {
    let mut checks = DeviceChecks::new();
    assert!(checks.is_empty());
    assert_eq!(checks.insert(RESOLVE, CheckAnswer::Metals), None);
    assert_eq!(checks.insert(QUANTIZE, not_metals()), None);
    assert_eq!(
        checks.insert(RESOLVE, not_metals()),
        Some(CheckAnswer::Metals)
    );
    assert_eq!(checks.len(), 2);
    let facts: Vec<DeviceFact> = checks.iter().map(|(fact, _)| fact).collect();
    let mut sorted = facts.clone();
    sorted.sort();
    assert_eq!(facts, sorted);
    assert_eq!(checks.get(RESOLVE), Some(&not_metals()));
    assert_eq!(checks.get(ZERO), None);
}

/// The profile takes the checked deviation as its check found it, its baseline where none
/// answered, and the tables alone under `DeviceChecksPolicy::Distrusted`; a different answer is a
/// different fingerprint.
#[test]
fn the_profile_takes_the_checked_deviation_as_its_check_found_it() {
    let distrusted = DevicePolicy {
        device_checks: DeviceChecksPolicy::Distrusted,
        ..DevicePolicy::default()
    };
    let takes = |description: &DeviceDescription, policy: DevicePolicy| {
        let device = Device::new(description.clone(), policy).expect("device");
        let taken = device
            .profile()
            .takes_driver_deviation(DriverDeviation::MetalHalfQuantize);
        (taken, device.profile_fingerprint())
    };
    // NVIDIA's 615.71.09 is not listed: Metal's answer takes it, nothing else does.
    let nvidia = description("nvidia-rtx5080-laptop");
    let (base, base_print) = takes(&nvidia, DevicePolicy::default());
    assert!(!base);
    let metals = with(nvidia.clone(), QUANTIZE, CheckAnswer::Metals);
    let (taken, print) = takes(&metals, DevicePolicy::default());
    assert!(taken);
    assert_ne!(print, base_print);
    assert_eq!(takes(&metals, distrusted), (false, base_print));
    for answer in [not_metals(), unanswered(), not_run()] {
        let checked = with(nvidia.clone(), QUANTIZE, answer.clone());
        assert_eq!(
            takes(&checked, DevicePolicy::default()),
            (false, base_print),
            "{answer:?}"
        );
    }
    // RADV on the 780M is listed: the other answer and an unanswered check clear it.
    let radv = description("amd-780m-radv");
    let (base, base_print) = takes(&radv, DevicePolicy::default());
    assert!(base);
    for answer in [not_metals(), unanswered()] {
        let checked = with(radv.clone(), QUANTIZE, answer.clone());
        let (taken, print) = takes(&checked, DevicePolicy::default());
        assert!(!taken, "{answer:?}");
        assert_ne!(print, base_print, "{answer:?}");
        assert_eq!(
            takes(&checked, distrusted),
            (true, base_print),
            "{answer:?}"
        );
    }
    for answer in [CheckAnswer::Metals, not_run()] {
        let checked = with(radv.clone(), QUANTIZE, answer.clone());
        assert_eq!(
            takes(&checked, DevicePolicy::default()),
            (true, base_print),
            "{answer:?}"
        );
    }
}

/// The runtime's checked fact follows its check: NVIDIA's 615.71.09 on the RTX 5080 Laptop GPU
/// is listed and the other answer clears it; lavapipe is not and Metal's answer takes it.
#[test]
fn the_runtime_s_checked_fact_follows_its_check() {
    let fact = DriverFact::RobustIndexFetchReadsZero;
    let nvidia = description("nvidia-rtx5080-laptop");
    assert!(nvidia.driver_facts().contains(&fact));
    let not = with(nvidia.clone(), ZERO, not_metals());
    assert!(!not.driver_facts().contains(&fact));
    let open = with(nvidia, ZERO, unanswered());
    assert!(!open.driver_facts().contains(&fact));
    let lavapipe = description("lavapipe");
    assert!(!lavapipe.driver_facts().contains(&fact));
    let metals = with(lavapipe, ZERO, CheckAnswer::Metals);
    assert!(metals.driver_facts().contains(&fact));
}

/// A workaround a driver's identity gives follows its check as a measured row's does: NVIDIA's
/// `HalfConversionPairFolded` (every release, by its identity) is cleared by Metal's answer and
/// kept by the other or an unanswered check, its status row assumed by the identity (no measured
/// row names the RTX 5080) while its check is pending; RADV's identity does not give it, and the other answer takes it. The profile and the
/// feature status agree.
#[test]
fn an_identity_workaround_follows_its_check() {
    let workaround = DriverWorkaround::HalfConversionPairFolded;
    let fact = DeviceFact::Workaround(workaround);
    let distrusted = DevicePolicy {
        device_checks: DeviceChecksPolicy::Distrusted,
        ..DevicePolicy::default()
    };
    let takes = |description: &DeviceDescription, policy: DevicePolicy| {
        let device = Device::new(description.clone(), policy).expect("device");
        device.profile().workarounds().contains(&workaround)
    };
    let status = |description: &DeviceDescription| {
        let rows = description.fact_status(true);
        rows.into_iter()
            .find(|row| row.fact == fact)
            .expect("a row")
    };
    let nvidia = description("nvidia-rtx5080-laptop");
    assert!(takes(&nvidia, DevicePolicy::default()));
    let pending = status(&nvidia);
    assert!(pending.holds);
    assert!(
        matches!(pending.decided, Decided::Assumed(_)),
        "{pending:?}"
    );
    let metals = with(nvidia.clone(), fact, CheckAnswer::Metals);
    assert!(!takes(&metals, DevicePolicy::default()));
    assert!(!status(&metals).holds);
    assert!(takes(&metals, distrusted));
    for answer in [not_metals(), unanswered(), not_run()] {
        let checked = with(nvidia.clone(), fact, answer.clone());
        assert!(takes(&checked, DevicePolicy::default()), "{answer:?}");
        assert!(status(&checked).holds, "{answer:?}");
    }
    let radv = description("amd-780m-radv");
    assert!(!takes(&radv, DevicePolicy::default()));
    assert!(!status(&radv).holds);
    let other = with(radv, fact, not_metals());
    assert!(takes(&other, DevicePolicy::default()));
    assert!(status(&other).holds);
}

/// A device's feature status names every checked fact translation or the runtime's driver facts
/// read once, as the readers take it, and the facts the tables give it; a runtime defect's row is
/// the runtime's to add (`DeviceDescription::fact_status`), so none is here.
#[test]
fn the_feature_status_names_every_checked_fact_as_the_readers_take_it() {
    for (stem, description) in committed_descriptions().expect("descriptions") {
        for trusted in [true, false] {
            let status = description.fact_status(trusted);
            for entry in CHECKED {
                let rows: Vec<_> = status.iter().filter(|row| row.fact == entry.fact).collect();
                if let DeviceFact::Defect(_) = entry.fact {
                    assert!(rows.is_empty(), "{stem}: {:?}", entry.fact);
                    continue;
                }
                assert_eq!(rows.len(), 1, "{stem}: {:?}", entry.fact);
                let check = match trusted {
                    true => CheckOf::Pending,
                    false => CheckOf::Distrusted,
                };
                assert_eq!(rows[0].check, check, "{stem}: {:?}", entry.fact);
            }
            let facts = description.driver_facts();
            for row in &status {
                if let DeviceFact::Fact(fact) = row.fact {
                    assert_eq!(row.holds, facts.contains(&fact), "{stem}: {fact:?}");
                }
            }
        }
        let metals = with(description.clone(), ZERO, CheckAnswer::Metals);
        let row = metals
            .fact_status(true)
            .into_iter()
            .find(|row| row.fact == ZERO)
            .expect("row");
        assert_eq!(row.decided, Decided::CheckedMetals, "{stem}");
        assert!(row.holds, "{stem}");
    }
}

/// No checked fact's answer changes what the device answers as Metal: the consumer's guest sees
/// the same device whatever the checks found.
#[test]
fn checked_facts_change_no_metal_device_property() {
    let answers = [CheckAnswer::Metals, not_metals(), unanswered(), not_run()];
    for (stem, description) in committed_descriptions().expect("descriptions") {
        let Ok(base) = Device::new(description.clone(), DevicePolicy::default()) else {
            continue;
        };
        for entry in CHECKED {
            for answer in &answers {
                let checked = with(description.clone(), entry.fact, answer.clone());
                let device = Device::new(checked, DevicePolicy::default()).expect("device");
                assert_eq!(
                    device.metal_properties(),
                    base.metal_properties(),
                    "{stem}: {:?} {answer:?}",
                    entry.fact
                );
            }
        }
    }
}

/// A fact the device learned off takes its exact side whatever its check or its baseline gave (a
/// defect worked around, an answer not taken), and says so; where those already gave the exact
/// side it keeps their reason, and where the policy distrusts checks nothing learned is read.
#[test]
fn a_learned_fact_takes_its_exact_side() {
    let why = "the device was lost (record r)";
    for fact in [QUANTIZE, ZERO, RESOLVE, DeviceFact::FunctionStorageFloor] {
        let exact = fact.is_defect();
        for answer in [None, Some(CheckAnswer::Metals), Some(not_metals())] {
            for baseline in [Baseline::unlisted(), Baseline::listed("a row".to_owned())] {
                let mut checks = DeviceChecks::new();
                if let Some(answer) = &answer {
                    checks.insert(fact, answer.clone());
                }
                let before = decide(fact, baseline.clone(), Some(&checks));
                checks.learn(fact, why.to_owned());
                assert_eq!(checks.learned(fact), Some(why));
                assert!(!checks.is_empty());
                let after = decide(fact, baseline.clone(), Some(&checks));
                assert_eq!(after.holds, exact, "{fact:?} {answer:?} {baseline:?}");
                match before.holds == exact {
                    true => assert_eq!(after, before, "{fact:?} {answer:?} {baseline:?}"),
                    false => assert_eq!(
                        (after.decided, after.check),
                        (Decided::LearnedOff(why.to_owned()), before.check),
                        "{fact:?} {answer:?} {baseline:?}"
                    ),
                }
            }
        }
    }
    let mut checks = DeviceChecks::new();
    checks.learn(RESOLVE, "first".to_owned());
    checks.learn(QUANTIZE, "second".to_owned());
    checks.learn(RESOLVE, "third".to_owned());
    assert_eq!(
        checks.learned_facts().collect::<Vec<_>>(),
        [QUANTIZE, RESOLVE]
    );
    assert_eq!(checks.learned(RESOLVE), Some("third"));
    assert_eq!(checks.len(), 0);
}

/// An unmeasured device that learned the Function storage floor off holds it for every stage, a
/// different profile; under distrusted checks, or measured, the record changes nothing.
#[test]
fn a_learned_floor_holds_function_storage_to_it() {
    use crate::device::profile::FunctionStorage;
    let distrusted = DevicePolicy {
        device_checks: DeviceChecksPolicy::Distrusted,
        ..DevicePolicy::default()
    };
    let storage = |description: &DeviceDescription, policy: DevicePolicy| {
        let device = Device::new(description.clone(), policy).expect("device");
        let storage = device.profile().fields().function_storage;
        (storage, device.profile_fingerprint())
    };
    // The RTX 5080 Laptop GPU's driver on another GPU, whose Function storage is not measured.
    let mut unmeasured = description("nvidia-rtx5080-laptop");
    unmeasured.device_id ^= 0x5a5a;
    let (assumed, print) = storage(&unmeasured, DevicePolicy::default());
    assert_eq!(assumed, FunctionStorage::Assumed(16 * 1024));
    let mut learned = unmeasured.clone();
    let why = "the device was lost (record r)".to_owned();
    learned
        .checks
        .learn(DeviceFact::FunctionStorageFloor, why.clone());
    let (floor, floor_print) = storage(&learned, DevicePolicy::default());
    assert_eq!(floor, FunctionStorage::Floor(16 * 1024));
    assert_ne!(floor_print, print);
    assert_eq!(storage(&learned, distrusted).0, assumed);
    let row = (learned.fact_status(true).into_iter())
        .find(|row| row.fact == DeviceFact::FunctionStorageFloor)
        .expect("a row");
    assert_eq!(
        (row.holds, row.decided),
        (true, Decided::LearnedOff(why.clone()))
    );
    assert!(learned
        .fact_status(false)
        .iter()
        .all(|row| row.fact != DeviceFact::FunctionStorageFloor));
    let mut measured = description("nvidia-rtx5080-laptop");
    let base = storage(&measured, DevicePolicy::default());
    measured.checks.learn(DeviceFact::FunctionStorageFloor, why);
    assert_eq!(storage(&measured, DevicePolicy::default()).0, base.0);
    assert!(measured
        .fact_status(true)
        .iter()
        .all(|row| row.fact != DeviceFact::FunctionStorageFloor));
}

/// With feature `serde`, a description's checks survive a JSON round trip.
#[cfg(feature = "serde")]
#[test]
fn checks_round_trip_through_serde() {
    let mut description = description("lavapipe");
    description.checks.insert(QUANTIZE, CheckAnswer::Metals);
    description.checks.insert(ZERO, unanswered());
    description.checks.insert(RESOLVE, not_run());
    (description.checks).learn(DeviceFact::FunctionStorageFloor, "a record".to_owned());
    let json = serde_json::to_string(&description).expect("json");
    let decoded: DeviceDescription = serde_json::from_str(&json).expect("decoded");
    assert_eq!(decoded, description);
}
