//! The loss watch (`device_checks/learned.rs`) with no GPU and no device: a driver's calls that
//! return `VK_ERROR_DEVICE_LOST` behind a slot's entry points, and the records they leave.

use crate::device_checks::learned::{self, read, Risks, Risky, Setup, Target, Watch};
use crate::device_checks::{
    status_text, ChecksRun, FeatureStatus, Learning, Settled, FEATURE_STATUS,
};
use ash::vk;
use metal2vulkan::device::{
    Decided, DeviceChecks, DeviceChecksPolicy, DeviceDescription, DeviceFact, DevicePolicy,
};
use metal2vulkan::error::RuntimeDriverDefect;
use std::ffi::{c_void, CStr};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// A device-checks directory's `v1` of the test's own, removed when it ends.
struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Scratch {
        let path = std::env::temp_dir().join(format!(
            "metal2vulkan-learned-{name}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&path);
        Scratch(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }

    /// Where [`setup`]'s records go in it.
    fn target(&self) -> Target {
        Target::new(self.path(), setup())
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A setup no record names.
fn setup() -> Setup {
    Setup {
        driver_id: Some(metal2vulkan::vk::DriverId::NVIDIA_PROPRIETARY),
        vendor_id: 0x10de,
        device_id: 0x2c19,
        driver_version: 0x99c3_4840,
        os_build: None,
    }
}

/// A driver's `vkQueueSubmit` of a lost device.
unsafe extern "system" fn submit_lost(
    _: vk::Queue,
    _: u32,
    _: *const vk::SubmitInfo<'_>,
    _: vk::Fence,
) -> vk::Result {
    vk::Result::ERROR_DEVICE_LOST
}

/// A driver's `vkWaitForFences` of a lost device.
unsafe extern "system" fn wait_lost(
    _: vk::Device,
    _: u32,
    _: *const vk::Fence,
    _: vk::Bool32,
    _: u64,
) -> vk::Result {
    vk::Result::ERROR_DEVICE_LOST
}

/// A driver's `vkGetFenceStatus` of a live device.
unsafe extern "system" fn status_ready(_: vk::Device, _: vk::Fence) -> vk::Result {
    vk::Result::SUCCESS
}

/// A device of no driver whose submissions and fence waits answer a lost device and whose fence
/// reads answer a live one, every other call ash's stub (which panics), routed through a slot of
/// `watch`'s: the device and its slot.
fn watched_device(watch: &Arc<Watch>) -> (ash::Device, learned::Slot) {
    let mut v1_0 = ash::DeviceFnV1_0::load(|_| std::ptr::null::<c_void>());
    v1_0.queue_submit = submit_lost;
    v1_0.wait_for_fences = wait_lost;
    v1_0.get_fence_status = status_ready;
    let mut v1_1 = ash::DeviceFnV1_1::load(|_| std::ptr::null());
    let mut v1_2 = ash::DeviceFnV1_2::load(|_| std::ptr::null());
    let mut v1_3 = ash::DeviceFnV1_3::load(|_| std::ptr::null());
    // SAFETY: the tables are the test device's own (never called with a real handle: each call
    // the tests make is above), not routed before.
    let slot = unsafe { learned::route(&mut v1_0, &mut v1_1, &mut v1_2, &mut v1_3, watch) };
    let device = ash::Device::from_parts_1_3(vk::Device::null(), v1_0, v1_1, v1_2, v1_3);
    (device, slot.expect("a free slot"))
}

/// The paths recorded for [`setup`] in `scratch`.
fn recorded(scratch: &Scratch) -> Vec<Risky> {
    read(scratch.path(), scratch.target().key())
        .iter()
        .map(|record| record.risky)
        .collect()
}

/// A loss a submission's call returns, while an accepted submission that took a risky path is not
/// seen complete, has its record made before the call returns, naming that path alone.
#[test]
fn a_loss_with_a_tagged_submission_in_flight_records_its_path_before_the_call_returns() {
    let scratch = Scratch::new("in-flight");
    let watch = Watch::new(Some(scratch.target()), 2);
    let (device, _slot) = watched_device(&watch);
    watch.queue(1).submitted(7);
    watch.submitted(1, 7, Risks::of(Risky::MultisampledStencilColourCopies));
    watch.queue(0).submitted(3);
    // SAFETY: the test driver's call, which reads nothing.
    let result = unsafe { device.queue_submit(vk::Queue::null(), &[], vk::Fence::null()) };
    assert_eq!(result, Err(vk::Result::ERROR_DEVICE_LOST));
    assert_eq!(recorded(&scratch), [Risky::MultisampledStencilColourCopies]);
    let records = read(scratch.path(), scratch.target().key());
    assert!(
        records[0]
            .body
            .contains("call: vkQueueSubmit returned VK_ERROR_DEVICE_LOST"),
        "{}",
        records[0].body
    );
    assert!(
        records[0].body.contains("deviceID: 0x2c19"),
        "{}",
        records[0].body
    );
    assert!(
        records[0].why().contains("vkQueueSubmit"),
        "{}",
        records[0].why()
    );
    let counts = watch.counts();
    assert!(counts.watched);
    assert_eq!(
        (counts.losses, counts.recorded, counts.risky_submissions),
        (1, 1, 1)
    );
}

/// A wait that returns the loss records as a submission does.
#[test]
fn a_wait_that_returns_the_loss_records_it() {
    let scratch = Scratch::new("wait");
    let watch = Watch::new(Some(scratch.target()), 1);
    let (device, _slot) = watched_device(&watch);
    watch.queue(0).submitted(2);
    watch.submitted(0, 2, Risks::of(Risky::FunctionStorageFloor));
    // SAFETY: the test driver's call, which reads nothing.
    let result = unsafe { device.wait_for_fences(&[], true, 0) };
    assert_eq!(result, Err(vk::Result::ERROR_DEVICE_LOST));
    assert_eq!(recorded(&scratch), [Risky::FunctionStorageFloor]);
}

/// A call that answers anything but a lost device records nothing.
#[test]
fn a_call_that_answers_otherwise_records_nothing() {
    let scratch = Scratch::new("live");
    let watch = Watch::new(Some(scratch.target()), 1);
    let (device, _slot) = watched_device(&watch);
    watch.queue(0).submitted(2);
    watch.submitted(0, 2, Risks::of(Risky::FunctionStorageFloor));
    // SAFETY: the test driver's call, which reads nothing.
    let status = unsafe { device.get_fence_status(vk::Fence::null()) };
    assert_eq!(status, Ok(true));
    assert_eq!(recorded(&scratch), []);
    assert_eq!(watch.counts().losses, 0);
}

/// With no tagged submission in flight (a lost submission that took no risky path: a shader's own
/// endless loop, or a submission the driver rejected), the loss blames nothing and writes nothing.
#[test]
fn an_untagged_loss_blames_nothing() {
    let scratch = Scratch::new("untagged");
    let watch = Watch::new(Some(scratch.target()), 1);
    let (device, _slot) = watched_device(&watch);
    watch.queue(0).submitted(4);
    watch.submitted(0, 4, Risks::NONE);
    // SAFETY: the test driver's call, which reads nothing.
    let result = unsafe { device.queue_submit(vk::Queue::null(), &[], vk::Fence::null()) };
    assert_eq!(result, Err(vk::Result::ERROR_DEVICE_LOST));
    assert_eq!(recorded(&scratch), []);
    let target = scratch.target();
    let records = target.directory();
    assert!(!records.exists(), "{} was made", records.display());
    let counts = watch.counts();
    assert_eq!(
        (counts.losses, counts.recorded, counts.risky_submissions),
        (1, 0, 0)
    );
}

/// A tagged submission the host saw complete is not blamed: only one still in flight is.
#[test]
fn a_completed_tagged_submission_is_not_blamed() {
    let scratch = Scratch::new("completed");
    let watch = Watch::new(Some(scratch.target()), 1);
    let (device, _slot) = watched_device(&watch);
    let queue = watch.queue(0);
    queue.submitted(5);
    watch.submitted(0, 5, Risks::of(Risky::ImplicitPreprocessAfterExecutionSet));
    queue.submitted(6);
    watch.submitted(0, 6, Risks::NONE);
    queue.completed(5);
    assert_eq!(watch.blamed(), Risks::NONE);
    // SAFETY: the test driver's call, which reads nothing.
    let result = unsafe { device.queue_submit(vk::Queue::null(), &[], vk::Fence::null()) };
    assert_eq!(result, Err(vk::Result::ERROR_DEVICE_LOST));
    assert_eq!(recorded(&scratch), []);
}

/// A free of address memory while work may be pending blames the batches submitted before it,
/// on every queue they are not seen complete on, and none submitted after them completes it.
#[test]
fn a_risk_taken_in_flight_notes_every_queue_at_its_last_value() {
    let watch = Watch::new(None, 2);
    watch.queue(0).submitted(3);
    watch.queue(0).completed(3);
    watch.queue(1).submitted(9);
    watch.risked_in_flight(Risky::FreedAddressMemoryFailsPendingWork);
    assert_eq!(
        watch.blamed(),
        Risks::of(Risky::FreedAddressMemoryFailsPendingWork)
    );
    watch.queue(1).completed(9);
    assert_eq!(watch.blamed(), Risks::NONE);
    // Nothing in flight as the risk is taken: nothing is noted.
    watch.risked_in_flight(Risky::FreedAddressMemoryFailsPendingWork);
    watch.queue(1).submitted(10);
    assert_eq!(watch.blamed(), Risks::NONE);
}

/// Only the first loss records: a record deleted after it is not made again by a later call that
/// returns the loss, and the later calls still count.
#[test]
fn only_the_first_loss_records() {
    let scratch = Scratch::new("first");
    let watch = Watch::new(Some(scratch.target()), 1);
    let (device, _slot) = watched_device(&watch);
    watch.queue(0).submitted(1);
    watch.submitted(0, 1, Risks::of(Risky::MultisampledStencilColourCopies));
    // SAFETY: the test driver's calls, which read nothing.
    let _ = unsafe { device.queue_submit(vk::Queue::null(), &[], vk::Fence::null()) };
    let directory = scratch.target().directory().to_owned();
    std::fs::remove_dir_all(&directory).expect("remove the records");
    // SAFETY: as above.
    let _ = unsafe { device.wait_for_fences(&[], true, 0) };
    assert!(!directory.exists());
    assert_eq!(watch.counts().losses, 2);
    assert_eq!(watch.counts().recorded, 1);
}

/// A record already there is kept as it is: a second device of the setup that loses itself with
/// the path in flight makes none.
#[test]
fn a_record_already_there_is_kept() {
    let scratch = Scratch::new("kept");
    let first = Watch::new(Some(scratch.target()), 1);
    let second = Watch::new(Some(scratch.target()), 1);
    for watch in [&first, &second] {
        watch.queue(0).submitted(1);
        watch.submitted(0, 1, Risks::of(Risky::FunctionStorageFloor));
        watch.lost(c"vkWaitSemaphores");
    }
    assert_eq!((first.counts().recorded, second.counts().recorded), (1, 0));
    let records = read(scratch.path(), scratch.target().key());
    assert_eq!(records.len(), 1);
    assert!(
        records[0].body.contains("vkWaitSemaphores"),
        "{}",
        records[0].body
    );
}

/// The next device reads each record of its setup, in the paths' order, and leaves a file whose
/// name is no path's alone; another setup's records are not its own.
#[test]
fn the_next_device_reads_its_setups_records() {
    let scratch = Scratch::new("next");
    let target = Target::new(scratch.path(), setup());
    std::fs::create_dir_all(target.directory()).expect("make the directory");
    for name in [
        Risky::FunctionStorageFloor.name(),
        Risky::MultisampledStencilColourCopies.name(),
        "defect-SomeLaterPath".to_owned(),
    ] {
        std::fs::write(target.directory().join(name), "").expect("write a record");
    }
    let found: Vec<Risky> = read(scratch.path(), target.key())
        .iter()
        .map(|record| record.risky)
        .collect();
    assert_eq!(
        found,
        [
            Risky::MultisampledStencilColourCopies,
            Risky::FunctionStorageFloor
        ]
    );
    let other = Setup {
        driver_version: setup().driver_version + 1,
        ..setup()
    };
    assert!(read(scratch.path(), &other.key()).is_empty());
    // An empty body (a write cut short) still turns its path off.
    let records = read(scratch.path(), target.key());
    assert!(records[0].why().contains("VK_ERROR_DEVICE_LOST"));
}

/// The key covers the driver, vendor, GPU, release and the OS build, each of them.
#[test]
fn the_key_covers_the_setup_alone() {
    let base = setup();
    let key = base.key();
    assert_eq!(key.len(), 64);
    assert!(key
        .chars()
        .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
    assert_eq!(setup().key(), key);
    let changed = [
        Setup {
            driver_id: Some(metal2vulkan::vk::DriverId::MESA_RADV),
            ..setup()
        },
        Setup {
            vendor_id: 0x1002,
            ..setup()
        },
        Setup {
            device_id: 0x2204,
            ..setup()
        },
        Setup {
            driver_version: 1,
            ..setup()
        },
        Setup {
            os_build: Some("25A354".to_owned()),
            ..setup()
        },
    ];
    for setup in changed {
        assert_ne!(setup.key(), key, "{setup:?}");
    }
}

/// Each path's record name is its fact's kind and name and reads back as the path, and the slot's
/// entry points are the watched calls' alone; a dropped slot leaves its device unwatched.
#[test]
fn names_and_entry_points() {
    for risky in Risky::ALL {
        let fact = risky.fact();
        assert_eq!(risky.name(), format!("{}-{}", fact.kind(), fact.name()));
        assert_eq!(Risky::of_name(&risky.name()), Some(risky));
    }
    assert_eq!(
        Risky::MultisampledStencilColourCopies.name(),
        "defect-MultisampledStencilColourCopies"
    );
    assert_eq!(
        Risky::FunctionStorageFloor.name(),
        "function-storage-FunctionStorageFloor"
    );
    assert_eq!(Risky::of_name("defect-"), None);
    let watch = Watch::new(None, 1);
    let (_device, slot) = watched_device(&watch);
    let watched: [&CStr; 6] = [
        c"vkQueueSubmit",
        c"vkQueueSubmit2",
        c"vkWaitForFences",
        c"vkWaitSemaphores",
        c"vkAllocateMemory",
        c"vkCreateGraphicsPipelines",
    ];
    for name in watched {
        assert!(slot.entry(name).is_some(), "{name:?}");
    }
    for name in [c"vkCmdDraw", c"vkDestroyBuffer", c"vkBeginCommandBuffer"] {
        assert!(slot.entry(name).is_none(), "{name:?}");
    }
    assert!(watch.counts().watched);
    drop(slot);
    assert!(!watch.counts().watched);
}

/// A committed description whose setup lists none of the risky paths' defects.
fn clean_description() -> DeviceDescription {
    let defects: Vec<RuntimeDriverDefect> = (Risky::ALL.into_iter())
        .filter_map(|risky| match risky.fact() {
            DeviceFact::Defect(defect) => Some(defect),
            _ => None,
        })
        .collect();
    (super::capabilities::committed_descriptions().into_iter())
        .map(|(_, description)| description)
        .find(|description| {
            let listed = crate::driver_defects::of(description);
            defects.iter().all(|defect| !listed.contains(defect))
        })
        .expect("a description that lists none of the risky defects")
}

/// The next device of a setup takes the workaround of each path a loss recorded, its feature
/// status says the fact was learned off and why, and its message names the learned key; a device
/// whose policy distrusts checks reads no record and takes the tables' answers.
#[test]
fn the_next_device_takes_the_workaround_of_each_path_recorded() {
    let scratch = Scratch::new("next-device");
    let target = Target::new(scratch.path(), setup());
    let watch = Watch::new(Some(target.clone()), 1);
    watch.queue(0).submitted(1);
    let risks =
        (Risky::ALL.into_iter()).fold(Risks::NONE, |risks, risky| risks.with(Risks::of(risky)));
    watch.submitted(0, 1, risks);
    watch.lost(c"vkQueueSubmit");
    let records = read(scratch.path(), target.key());
    assert_eq!(
        records
            .iter()
            .map(|record| record.risky)
            .collect::<Vec<_>>(),
        Risky::ALL
    );
    let clean = clean_description();
    let mut next = clean.clone();
    for record in &records {
        next.checks.learn(record.risky.fact(), record.why());
    }
    let defects = crate::driver_defects::checked(&next);
    for risky in Risky::ALL {
        if let DeviceFact::Defect(defect) = risky.fact() {
            assert!(defects.contains(&defect), "{defect:?}");
            assert!(!crate::driver_defects::checked(&clean).contains(&defect));
        }
    }
    assert!(crate::generated_commands::preprocess_apart(&next));
    assert!(!crate::generated_commands::preprocess_apart(&clean));
    let mut run = ChecksRun::new(String::new());
    run.settled = Settled::FromCache { checked_at: 1 };
    run.learning = Learning {
        key: target.key().to_owned(),
        records: Some(target.directory().to_owned()),
        watched: true,
        read: true,
    };
    let rows = crate::driver_defects::status_rows(&next, true);
    let status = FeatureStatus::of(&next, run, rows);
    for risky in Risky::ALL {
        let Some(row) = status.fact(risky.fact()) else {
            // The floor's row is a device's whose Function storage is not measured.
            assert_eq!(risky, Risky::FunctionStorageFloor);
            continue;
        };
        assert!(row.holds, "{risky:?}");
        assert!(
            matches!(&row.decided, Decided::LearnedOff(why) if why.contains("vkQueueSubmit")),
            "{risky:?}: {:?}",
            row.decided
        );
    }
    let text = status_text(FEATURE_STATUS, "a test GPU", &next, &status);
    assert!(text.contains("learned off: the device was lost"), "{text}");
    assert!(
        text.contains(&format!("learned key {}", &target.key()[..16])),
        "{text}"
    );
    // Distrusted: the records stay, unread, and every fact is as the tables have it.
    let mut run = ChecksRun::new(String::new());
    run.learning.read = false;
    run.learning.watched = true;
    run.learning.records = Some(target.directory().to_owned());
    let rows = crate::driver_defects::status_rows(&clean, false);
    let distrusted = FeatureStatus::of(&clean, run, rows);
    assert!(distrusted
        .facts
        .iter()
        .all(|row| !matches!(row.decided, Decided::LearnedOff(_))));
    let text = status_text(FEATURE_STATUS, "a test GPU", &clean, &distrusted);
    assert!(text.contains("(records not read)"), "{text}");
}

/// A device past the watch's slots runs unwatched, and its status says so.
#[test]
fn an_unwatched_device_says_so() {
    let description = clean_description();
    let mut run = ChecksRun::new(String::new());
    run.learning.records = Some(std::env::temp_dir());
    run.learning.watched = false;
    let rows = crate::driver_defects::status_rows(&description, true);
    let status = FeatureStatus::of(&description, run, rows);
    let text = status_text(FEATURE_STATUS, "a test GPU", &description, &status);
    assert!(text.contains("unwatched for a lost device"), "{text}");
    let watch = Watch::new(None, 1);
    assert!(!watch.counts().watched);
}

/// The records of a setup are read as its device is made where the policy trusts checks, each
/// path's fact learned off; under `DeviceChecksPolicy::Distrusted` none is read (the learned set
/// stays empty, the status says so), though the place a loss would record in is still kept; with
/// no device-checks directory there is none.
#[test]
fn a_distrusting_policy_reads_no_record() {
    let scratch = Scratch::new("distrusted");
    let description = clean_description();
    let setup = Setup::of(&description);
    let target = Target::new(scratch.path(), setup.clone());
    std::fs::create_dir_all(target.directory()).expect("make the directory");
    let risky = Risky::ImplicitPreprocessAfterExecutionSet;
    std::fs::write(target.directory().join(risky.name()), "").expect("write a record");
    let mut distrusted = DevicePolicy::default();
    distrusted.device_checks = DeviceChecksPolicy::Distrusted;
    let mut checks = DeviceChecks::new();
    let (kept, learning) =
        learned::settle(&description, Some(scratch.path()), distrusted, &mut checks);
    assert!(checks.is_empty(), "{checks:?}");
    assert_eq!(checks.learned_facts().count(), 0);
    assert!(!learning.read);
    assert_eq!(learning.key, setup.key());
    assert_eq!(learning.records.as_deref(), Some(target.directory()));
    assert_eq!(kept.expect("a place").key(), target.key());
    let mut checks = DeviceChecks::new();
    let (_, learning) = learned::settle(
        &description,
        Some(scratch.path()),
        DevicePolicy::default(),
        &mut checks,
    );
    assert!(learning.read);
    assert_eq!(checks.learned_facts().collect::<Vec<_>>(), [risky.fact()]);
    let mut checks = DeviceChecks::new();
    let (none, learning) =
        learned::settle(&description, None, DevicePolicy::default(), &mut checks);
    assert!(none.is_none() && learning.records.is_none() && checks.is_empty());
}
