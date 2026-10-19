//! The device's checks of its driver (`device_checks`): the suite against the translator's list,
//! the cache's entries, and the feature status. None of these runs a check on a device; the
//! validation lane `runtime_lane::device_checks` does.

use crate::device_checks::background::{answer_each, Keeper};
use crate::device_checks::cache::{build_fields, parse, render, Key, Place};
use crate::device_checks::live::{read_by_translation, Gate, Live};
use crate::device_checks::{
    status_text, Check, CheckSource, ChecksRun, DeviceChecksOptions, FeatureStatus, Settle,
    Settled, FEATURE_STATUS, FEATURE_STATUS_SETTLED, SUITE,
};
use metal2vulkan::device::{
    CheckAnswer, CheckOf, Decided, DeviceChecks, DeviceDescription, DeviceFact, DriverFact, CHECKED,
};
use metal2vulkan::error::{DriverDeviation, RuntimeDriverDefect};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

const QUANTIZE: DeviceFact = DeviceFact::Deviation(DriverDeviation::MetalHalfQuantize);
const ZERO: DeviceFact = DeviceFact::Fact(DriverFact::RobustIndexFetchReadsZero);
const RESOLVE: DeviceFact =
    DeviceFact::Defect(RuntimeDriverDefect::DepthStencilResolveFirstSubresource);

/// A key of one setup, `n` telling setups apart.
fn key(n: u32) -> Key {
    Key::new(vec![
        ("driver", "Some(MESA_RADV)".to_owned()),
        ("driver-version", format!("{n:#010x}")),
    ])
}

/// Every answer kind, one per fact of the suite.
fn answers() -> DeviceChecks {
    let mut checks = DeviceChecks::new();
    checks.insert(QUANTIZE, CheckAnswer::Metals);
    checks.insert(
        ZERO,
        CheckAnswer::NotRun("robustBufferAccess2 is not enabled".to_owned()),
    );
    checks.insert(RESOLVE, not_metals());
    checks
}

/// The other answer, as a check states it.
fn not_metals() -> CheckAnswer {
    CheckAnswer::NotMetals("the resolved layers held [Pass, Kept]".to_owned())
}

/// A directory of the test's own under the OS temp dir, removed when dropped.
struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Scratch {
        let dir =
            std::env::temp_dir().join(format!("m2v-device-checks-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        Scratch(dir)
    }

    fn place(&self) -> Place {
        let options = DeviceChecksOptions {
            directory: Some(self.0.clone()),
            wait: false,
        };
        Place::of(&options).expect("a place")
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The suite checks exactly the facts the translator decides by a check, in its order.
#[test]
fn the_suite_checks_every_checked_fact() {
    let suite: Vec<DeviceFact> = SUITE
        .iter()
        .flat_map(|check| check.facts)
        .copied()
        .collect();
    let checked: Vec<DeviceFact> = CHECKED.iter().map(|entry| entry.fact).collect();
    assert_eq!(suite, checked);
}

/// An entry reads back what was written, every answer but an unanswered one, which is never kept
/// and leaves the entry's answer of its fact; one of no keepable answer is not written.
#[test]
fn an_entry_round_trips_but_unanswered_checks() {
    let scratch = Scratch::new("round-trip");
    let place = scratch.place();
    let mut checks = answers();
    place.write(&key(1), &checks).expect("written");
    let entry = place.read(&key(1)).expect("read");
    for (fact, answer) in checks.iter() {
        assert_eq!(entry.answer(fact), Some(answer), "{fact:?}");
    }
    checks.insert(RESOLVE, CheckAnswer::Unanswered("timed out".to_owned()));
    place.write(&key(1), &checks).expect("written");
    let entry = place.read(&key(1)).expect("read");
    assert_eq!(entry.answer(RESOLVE), Some(&not_metals()));
    assert_eq!(entry.answer(QUANTIZE), Some(&CheckAnswer::Metals));
    let mut open = DeviceChecks::new();
    open.insert(QUANTIZE, CheckAnswer::Unanswered("no queue".to_owned()));
    assert_eq!(render(&key(2), 0, &open), None);
    place.write(&key(2), &open).expect("nothing to write");
    assert!(place.read(&key(2)).is_none());
}

/// Two writers of one setup, each with answers of its own, leave both in the entry: a write keeps
/// the entry's answers but those of the facts it answers, which take its own.
#[test]
fn writers_of_one_setup_merge_their_answers() {
    let scratch = Scratch::new("merge");
    let (first, second) = (scratch.place(), scratch.place());
    let mut ours = DeviceChecks::new();
    ours.insert(QUANTIZE, CheckAnswer::Metals);
    let mut theirs = DeviceChecks::new();
    theirs.insert(RESOLVE, not_metals());
    first.write(&key(1), &ours).expect("written");
    second.write(&key(1), &theirs).expect("written");
    let entry = first.read(&key(1)).expect("read");
    assert_eq!(entry.answer(QUANTIZE), Some(&CheckAnswer::Metals));
    assert_eq!(entry.answer(RESOLVE), Some(&not_metals()));
    ours.insert(QUANTIZE, not_metals());
    first.write(&key(1), &ours).expect("written");
    let entry = second.read(&key(1)).expect("read");
    assert_eq!(entry.answer(QUANTIZE), Some(&not_metals()));
    assert_eq!(entry.answer(RESOLVE), Some(&not_metals()));
}

/// Each fact of `check` answered as Metal's.
fn metals(check: &Check) -> DeviceChecks {
    let mut checks = DeviceChecks::new();
    for &fact in check.facts {
        checks.insert(fact, CheckAnswer::Metals);
    }
    checks
}

/// A device dropped after `k` of the checks its thread runs answered, during the next check or
/// before the thread told it the `k`th answer, leaves those `k` in the cache, each kept as its
/// check settled, and the setup's next making reads them and runs only the others.
#[test]
fn a_device_dropped_mid_checks_keeps_the_answers_its_thread_finished() {
    let checks: Vec<&'static Check> = (SUITE.iter())
        .filter(|check| check.settle == Settle::Background)
        .collect();
    assert!(
        checks.len() > 2,
        "{} checks run in the background",
        checks.len()
    );
    let k = 2;
    for (name, told_all) in [("dropped-during", true), ("dropped-untold", false)] {
        let scratch = Scratch::new(name);
        let mut keeper = Keeper::new(Ok(scratch.place()), key(1));
        let mut answers = DeviceChecks::new();
        let (mut ran, mut told) = (0, 0);
        let answer = |check: &'static Check| {
            ran += 1;
            (ran <= k).then(|| (metals(check), CheckSource::Skipped))
        };
        let tell = |_: &'static Check, _: &DeviceChecks, _: CheckSource| {
            told += 1;
            told_all || told < k
        };
        assert!(!answer_each(
            &checks,
            &mut answers,
            &mut keeper,
            answer,
            tell
        ));
        assert_eq!(keeper.error(), None, "{name}");
        let entry = scratch.place().read(&key(1)).expect("the answers kept");
        for (at, check) in checks.iter().enumerate() {
            let read = entry.answers_of(check);
            assert_eq!(
                read.is_some(),
                at < k,
                "{name}: check {at} of {}",
                checks.len()
            );
        }
    }
}

/// Another setup's key reads no entry: a setup that changes is checked again.
#[test]
fn a_changed_setup_reads_no_entry() {
    let scratch = Scratch::new("changed");
    let place = scratch.place();
    place.write(&key(1), &answers()).expect("written");
    assert!(place.read(&key(1)).is_some());
    assert!(place.read(&key(2)).is_none());
    let text = render(&key(1), 7, &answers()).expect("an entry");
    assert!(parse(&text, &key(2)).is_none());
}

/// The key carries this build's translator and runtime source digests, so an entry another build
/// wrote, whose checks' modules, inputs or expected answers may differ though no version moved, is
/// not read.
#[test]
fn another_build_s_entry_is_not_read() {
    let fields = build_fields();
    let named = |name: &str| (fields.iter()).find(|(field, _)| *field == name);
    let translator = named("translator-source").expect("the translator's digest");
    assert_eq!(translator.1, metal2vulkan::cache::SOURCE_DIGEST);
    let runtime = named("runtime-source").expect("the runtime's digest");
    assert_eq!(runtime.1, env!("M2V_RUNTIME_SOURCE_ID"));
    let setup = |translator: &str| {
        let mut fields = vec![("driver", "Some(MESA_RADV)".to_owned())];
        fields.extend(build_fields().map(|(name, value)| match name {
            "translator-source" => (name, translator.to_owned()),
            _ => (name, value),
        }));
        Key::new(fields)
    };
    let (this, other) = (setup(metal2vulkan::cache::SOURCE_DIGEST), setup("another"));
    assert_ne!(this, other);
    let text = render(&other, 7, &answers()).expect("an entry");
    assert!(parse(&text, &other).is_some());
    assert!(parse(&text, &this).is_none());
}

/// An entry that is cut short, changed, of another format or naming a fact the suite does not
/// check is not read.
#[test]
fn a_corrupt_entry_is_not_read() {
    let text = render(&key(1), 7, &answers()).expect("an entry");
    let entry = parse(&text, &key(1)).expect("whole");
    assert_eq!(entry.checked_at, 7);
    assert!(parse(&text[..text.len() / 2], &key(1)).is_none());
    assert!(parse(&text.replacen("metals", "metal", 1), &key(1)).is_none());
    assert!(parse(&text.replacen("v1", "v2", 1), &key(1)).is_none());
    assert!(parse("", &key(1)).is_none());
    // A body whose digest is recomputed but names an unchecked fact.
    let (body, _) = text.trim_end().rsplit_once('\n').expect("a digest line");
    let body = format!("{body}\nanswer deviation MetalFragmentSimdgroups metals\n");
    let digest: String = {
        use sha2::Digest;
        sha2::Sha256::digest(body.as_bytes())
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    };
    assert!(parse(&format!("{body}sha256 {digest}\n"), &key(1)).is_none());
    let scratch = Scratch::new("corrupt");
    let place = scratch.place();
    place.write(&key(1), &answers()).expect("written");
    let file = scratch
        .0
        .join("v1")
        .join(format!("{}.checks", key(1).hex()));
    std::fs::write(&file, text.replacen("not-metals", "metals", 1)).expect("rewritten");
    assert!(place.read(&key(1)).is_none());
}

/// A distrusted device's status takes every fact as the tables have it and says so; a trusted
/// one's checked rows carry their answers.
#[test]
fn the_status_names_how_each_checked_fact_was_decided() {
    for (stem, description) in super::capabilities::committed_descriptions() {
        let distrusted = FeatureStatus::unchecked(&description);
        assert_eq!(distrusted.settled, Settled::Distrusted, "{stem}");
        for entry in CHECKED {
            let row = distrusted.fact(entry.fact).expect("a row");
            assert_eq!(row.check, CheckOf::Distrusted, "{stem}");
        }
        let mut checked = description.clone();
        checked.checks = answers();
        let mut run = ChecksRun::new("ab".repeat(32));
        run.settled = Settled::FromCache { checked_at: 7 };
        let defects = crate::driver_defects::status_rows(&checked, true);
        let status = FeatureStatus::of(&checked, run, defects);
        let quantize = status.fact(QUANTIZE).expect("a row");
        assert_eq!(
            (quantize.holds, &quantize.decided),
            (true, &Decided::CheckedMetals),
            "{stem}"
        );
        let resolve = status.fact(RESOLVE).expect("a row");
        assert_eq!(
            (resolve.holds, &resolve.decided),
            (true, &Decided::CheckedNotMetals),
            "{stem}"
        );
        let zero = status.fact(ZERO).expect("a row");
        assert!(
            matches!(zero.check, CheckOf::Answered(CheckAnswer::NotRun(_))),
            "{stem}"
        );
        assert!(
            crate::driver_defects::checked(&checked).contains(&RESOLVE_DEFECT),
            "{stem}"
        );
        let text = status_text(FEATURE_STATUS, "a GPU", &checked, &status);
        assert!(
            text.ends_with(&format!("[{FEATURE_STATUS}]")),
            "{stem}: {text}"
        );
        assert!(text.contains("read from the cache"), "{stem}: {text}");
        assert!(
            text.contains("MetalHalfQuantize (deviation): taken, checked"),
            "{stem}: {text}"
        );
        assert!(text.contains("abababababababab"), "{stem}: {text}");
    }
}

const RESOLVE_DEFECT: RuntimeDriverDefect =
    RuntimeDriverDefect::DepthStencilResolveFirstSubresource;

/// The checked defect follows its check whatever the tables say: lavapipe's, listed, is cleared
/// by Metal's answer, and a driver no row names takes it from the other answer.
#[test]
fn the_checked_defect_follows_its_check() {
    let descriptions = super::capabilities::committed_descriptions();
    let find = |stem: &str| -> DeviceDescription {
        descriptions
            .iter()
            .find(|(s, _)| s == stem)
            .map(|(_, d)| d.clone())
            .expect(stem)
    };
    let mut lavapipe = find("lavapipe");
    assert!(crate::driver_defects::checked(&lavapipe).contains(&RESOLVE_DEFECT));
    lavapipe.checks.insert(RESOLVE, CheckAnswer::Metals);
    assert!(!crate::driver_defects::checked(&lavapipe).contains(&RESOLVE_DEFECT));
    // Unanswered, the table's answer stands: the workaround refuses what the baseline runs.
    lavapipe
        .checks
        .insert(RESOLVE, CheckAnswer::Unanswered("timed out".to_owned()));
    assert!(crate::driver_defects::checked(&lavapipe).contains(&RESOLVE_DEFECT));
    let mut radv = find("amd-780m-radv");
    assert!(!crate::driver_defects::checked(&radv).contains(&RESOLVE_DEFECT));
    radv.checks.insert(RESOLVE, not_metals());
    assert!(crate::driver_defects::checked(&radv).contains(&RESOLVE_DEFECT));
}

/// The checks push as the device is assembled to push: by one rule over the answers so far
/// (`driver_defects::takes_push_command`). MoltenVK, whose pushed sets are listed as losing their
/// buffers' sizes, takes no push command until its check answers Metal's; ANV takes it unless the
/// check answers the other; and the answers so far decide as a description holding them does.
/// The check that decides it settles at the making, before every check that pushes as the device
/// does.
#[test]
fn the_checks_push_by_the_device_s_rule() {
    let pushes = DeviceFact::Defect(RuntimeDriverDefect::PushedBufferSizesUnset);
    let descriptions = super::capabilities::committed_descriptions();
    let find = |stem: &str| -> DeviceDescription {
        descriptions
            .iter()
            .find(|(s, _)| s == stem)
            .map(|(_, d)| d.clone())
            .expect(stem)
    };
    let takes = |description: &DeviceDescription, answer: Option<CheckAnswer>| {
        let mut checks = DeviceChecks::new();
        if let Some(answer) = answer {
            checks.insert(pushes, answer);
        }
        let so_far = crate::driver_defects::taken(description, &checks);
        let mut holding = description.clone();
        holding.checks = checks;
        let assembled = crate::driver_defects::checked(&holding);
        assert_eq!(so_far, assembled, "{:?}", description.driver_id);
        crate::driver_defects::takes_push_command(&so_far)
    };
    let moltenvk = find("moltenvk-m3");
    assert!(!takes(&moltenvk, None));
    assert!(takes(&moltenvk, Some(CheckAnswer::Metals)));
    assert!(!takes(&moltenvk, Some(not_metals())));
    let anv = find("intel-irisxe-anv");
    assert!(takes(&anv, None));
    assert!(takes(&anv, Some(CheckAnswer::Metals)));
    assert!(!takes(&anv, Some(not_metals())));
    let at = |fact: DeviceFact| {
        SUITE
            .iter()
            .position(|check| check.facts.contains(&fact))
            .expect("a check of the fact")
    };
    let deciding = &SUITE[at(pushes)];
    assert_eq!(deciding.settle, Settle::AtCreation);
    for pushing in [
        RuntimeDriverDefect::PushedDescriptorsStaleAfterPipelineBinds,
        RuntimeDriverDefect::ImmutableSamplersLostAfterCopies,
    ] {
        assert!(at(pushes) < at(DeviceFact::Defect(pushing)), "{pushing:?}");
    }
}

/// A check of a fact that decides a refusal or shapes a layout settles at the making; the
/// runtime's checked defect is one (its workaround refuses a memoryless attachment), and the
/// facts translation reads (the compile profile) are the deviations and workarounds.
#[test]
fn each_check_settles_where_its_fact_allows() {
    for check in SUITE {
        if check.facts.contains(&RESOLVE) {
            assert_eq!(check.settle, Settle::AtCreation, "{:?}", check.facts);
        }
    }
    assert!(read_by_translation(QUANTIZE));
    assert!(!read_by_translation(ZERO));
    assert!(!read_by_translation(RESOLVE));
}

/// The other answer's statement survives the cache, so a status read from it names how the
/// driver differs.
#[test]
fn the_other_answer_keeps_its_statement_in_the_cache() {
    let text = render(&key(3), 7, &answers()).expect("an entry");
    let entry = parse(&text, &key(3)).expect("whole");
    assert_eq!(entry.answer(RESOLVE), Some(&not_metals()));
}

/// The runtime facts whose checks a device's checks thread runs switch from their rows as made;
/// a fact translation reads, or one whose check ran at the making, does not.
#[test]
fn only_the_pending_runtime_facts_switch() {
    let pending: Vec<_> = SUITE
        .iter()
        .filter(|check| check.settle == Settle::Background)
        .collect();
    for (stem, description) in super::capabilities::committed_descriptions() {
        let status = FeatureStatus::unchecked(&description);
        let live = Live::new(status.clone(), &pending);
        for check in SUITE {
            for &fact in check.facts {
                let switches = check.settle == Settle::Background && !read_by_translation(fact);
                let row = status.fact(fact).map(|row| row.holds);
                match switches {
                    true => assert_eq!(live.holds(fact), row, "{stem}: {fact:?}"),
                    false => assert_eq!(live.holds(fact), None, "{stem}: {fact:?}"),
                }
            }
        }
        assert_eq!(live.status(), status, "{stem}");
    }
}

/// A status whose checks settle on the checks thread says how many are left, and once settled,
/// a translation fact's answer as the next device's, the other answer with its statement.
#[test]
fn the_status_names_the_checks_left_and_the_next_device_s_answers() {
    let (_, description) = super::capabilities::committed_descriptions()
        .into_iter()
        .find(|(stem, _)| stem == "lavapipe")
        .expect("lavapipe");
    let mut run = ChecksRun::new("cd".repeat(32));
    for check in SUITE {
        run.push(check, CheckSource::Pending);
    }
    run.settled = Settled::Running {
        at_creation: std::time::Duration::from_millis(3),
    };
    let defects = crate::driver_defects::status_rows(&description, true);
    let mut status = FeatureStatus::of(&description, run, defects);
    let text = status_text(FEATURE_STATUS, "a GPU", &description, &status);
    let left = format!(
        "{} checks settle on the device's checks thread",
        SUITE.len()
    );
    assert!(text.contains(&left), "{text}");
    let row = (status.facts.iter_mut())
        .find(|row| row.fact == QUANTIZE)
        .expect("a row");
    row.check = CheckOf::NextDevice(CheckAnswer::NotMetals("word 3".to_owned()));
    status.settled = Settled::InBackground {
        at_creation: std::time::Duration::from_millis(3),
        wall: std::time::Duration::from_millis(40),
    };
    let text = status_text(FEATURE_STATUS_SETTLED, "a GPU", &description, &status);
    assert!(
        text.ends_with(&format!("[{FEATURE_STATUS_SETTLED}]")),
        "{text}"
    );
    assert!(
        text.contains("settled on the device's checks thread"),
        "{text}"
    );
    assert!(
        text.contains("(checked not Metal's answer: word 3, taken from the next device)"),
        "{text}"
    );
}

/// What a share of a device stands for in the gate's tests: its drop records the thread it ran on,
/// as a device's teardown runs where its last share goes.
struct Teardown(Arc<Mutex<Option<String>>>);

impl Drop for Teardown {
    fn drop(&mut self) {
        let thread = std::thread::current();
        *crate::device::lock(&self.0) = Some(thread.name().unwrap_or("unnamed").to_owned());
    }
}

/// A share of a stand-in device, and where its teardown ran.
fn stand_in() -> (Arc<Teardown>, Arc<Mutex<Option<String>>>) {
    let ran = Arc::new(Mutex::new(None));
    (Arc::new(Teardown(Arc::clone(&ran))), ran)
}

/// The last handle goes while a call of the checks thread holds the device: its drop waits for the
/// call, which lets go first, so the teardown runs on the dropping thread (an adopted device's
/// consumer may destroy its `VkDevice` as soon as that drop returns), and no call begins after.
/// The drop is seen waiting (it stopped the gate) before the call ends: the order is the gate's,
/// never a timing.
#[test]
fn the_last_handle_tears_the_device_down_after_a_call_in_progress() {
    let gate = Arc::new(Gate::new(true));
    let (handle, ran) = stand_in();
    let weak = Arc::downgrade(&handle);
    let held = gate.enter(|| weak.upgrade()).expect("a call begins");
    let dropping = {
        let gate = Arc::clone(&gate);
        std::thread::Builder::new()
            .name("the program".to_owned())
            .spawn(move || gate.let_go(handle))
            .expect("a thread")
    };
    while !gate.stopped() {
        std::thread::yield_now();
    }
    assert_eq!(
        *crate::device::lock(&ran),
        None,
        "torn down before the call ended"
    );
    gate.leave(held);
    dropping.join().expect("the drop returns");
    assert_eq!(crate::device::lock(&ran).as_deref(), Some("the program"));
    assert!(
        gate.enter(|| weak.upgrade()).is_none(),
        "a call after the last drop"
    );
}

/// A handle that is not the last goes under the gate's lock and tears nothing down, a call in
/// progress or not; the last, with no call in progress, tears the device down at once and stops
/// the gate.
#[test]
fn only_the_last_handle_tears_the_device_down() {
    let gate = Gate::new(true);
    let (handle, ran) = stand_in();
    let weak = Arc::downgrade(&handle);
    let other = Arc::clone(&handle);
    let held = gate.enter(|| weak.upgrade()).expect("a call begins");
    gate.let_go(other);
    assert_eq!(*crate::device::lock(&ran), None);
    assert!(!gate.stopped());
    gate.leave(held);
    assert_eq!(*crate::device::lock(&ran), None);
    let name = std::thread::current().name().map(str::to_owned);
    gate.let_go(handle);
    assert_eq!(*crate::device::lock(&ran), name);
    assert!(gate.stopped());
    assert!(gate.enter(|| weak.upgrade()).is_none());
}
