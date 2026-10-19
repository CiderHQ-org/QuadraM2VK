//! The known-driver list (`device::listed`): its matcher on driver, release range and GPU family,
//! and its tables' shape against the measured table (`device::driver_facts`).

use crate::device::driver_facts::{mesa, moltenvk, nvidia, MEASURED};
use crate::device::listed::{
    listed, Gpus, Listing, Releases, APPLE9, LISTED_DEFECTS, LISTED_FACTS,
};
use crate::device::{DeviceDescription, DriverFact};
use crate::vk::DriverId;
use std::collections::BTreeSet;
use std::path::PathBuf;

const APPLE: u32 = 0x106B;

/// The committed profile `stem`.
fn read(stem: &str) -> DeviceDescription {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../metal2vulkan-validation/profiles");
    let text = std::fs::read_to_string(dir.join(format!("{stem}.json"))).expect(stem);
    DeviceDescription::from_khronos_json(&text, None).expect("profile parses")
}

/// A description of `driver` on the GPU `device` of `vendor` at `release`, the rest the M4's.
fn of(driver: DriverId, vendor: u32, device: u32, release: u32) -> DeviceDescription {
    let mut desc = read("moltenvk-m4");
    desc.driver_id = Some(driver);
    desc.vendor_id = vendor;
    desc.device_id = device;
    desc.driver_version = release;
    desc
}

/// A range of releases holds from its first through the one before its end: every release, from
/// one on, or before one.
#[test]
fn a_range_holds_from_its_first_release_up_to_its_end() {
    assert!(Releases::EVERY.contains(0) && Releases::EVERY.contains(u32::MAX));
    let since = Releases::since(mesa(26, 2, 4));
    assert!(!since.contains(mesa(26, 2, 3)));
    assert!(since.contains(mesa(26, 2, 4)) && since.contains(mesa(27, 0, 0)));
    let before = Releases::before(mesa(26, 2, 4));
    assert!(before.contains(0) && before.contains(mesa(26, 2, 3)));
    assert!(!before.contains(mesa(26, 2, 4)) && !before.contains(mesa(27, 0, 0)));
}

/// A listing names a device of its driver, of its vendor where it names one, on a GPU of its
/// family and at a release of its range, and no other. A GPU that reports more than one
/// `deviceID` is named by the one it was measured under; a MoltenVK family names Apple's GPUs of
/// that family under every macOS release, and no other vendor's GPU of the same low bits.
#[test]
fn a_listing_names_its_driver_vendor_family_and_releases() {
    let radv = |gpus| Listing {
        driver: DriverId::MESA_RADV,
        vendor: Some(0x1002),
        gpus,
        releases: Releases::since(mesa(26, 2, 4)),
    };
    let phoenix = of(DriverId::MESA_RADV, 0x1002, 0x15bf, mesa(26, 2, 4));
    let mut hawk_point = phoenix.clone();
    hawk_point.device_id = 0x1900;
    let only = radv(Gpus::Only(&[0x15bf]));
    assert!(only.names(&phoenix) && only.names(&hawk_point));
    let mut earlier = phoenix.clone();
    earlier.driver_version = mesa(26, 2, 3);
    assert!(!only.names(&earlier));
    let mut other_gpu = phoenix.clone();
    other_gpu.device_id = 0x744c;
    assert!(!only.names(&other_gpu));
    assert!(radv(Gpus::Except(&[0x15bf])).names(&other_gpu));
    assert!(!radv(Gpus::Except(&[0x15bf])).names(&hawk_point));
    assert!(radv(Gpus::Any).names(&other_gpu));
    let mut other_vendor = phoenix.clone();
    other_vendor.vendor_id = 0x8086;
    assert!(!radv(Gpus::Any).names(&other_vendor));
    let mut other_driver = phoenix.clone();
    other_driver.driver_id = Some(DriverId::MESA_LLVMPIPE);
    assert!(!radv(Gpus::Any).names(&other_driver));
    other_driver.driver_id = None;
    assert!(!radv(Gpus::Any).names(&other_driver));

    let family = Listing {
        driver: DriverId::MOLTENVK,
        vendor: None,
        gpus: Gpus::MoltenVkFamily(APPLE9),
        releases: Releases::EVERY,
    };
    for device in [0x1A06_0209, 0x1B00_0209, 0x1C00_0209] {
        assert!(
            family.names(&of(DriverId::MOLTENVK, APPLE, device, moltenvk(1, 4, 1))),
            "{device:#x}"
        );
    }
    assert!(!family.names(&of(
        DriverId::MOLTENVK,
        APPLE,
        0x1B00_0208,
        moltenvk(1, 4, 1)
    )));
    assert!(!family.names(&of(DriverId::MOLTENVK, 0x1002, 0x0209, moltenvk(1, 4, 1))));
}

/// The two tables of runtime driver facts are apart: a fact a check may settle is `MEASURED`'s,
/// taken on a measured identity, and a fact no check settles is the list's, never both. No
/// defect fact is `MEASURED`'s, and the list holds each defect fact open on the affected side.
#[test]
fn the_measured_and_listed_facts_are_apart() {
    let measured: BTreeSet<DriverFact> = MEASURED.iter().map(|row| row.fact).collect();
    let listed: BTreeSet<DriverFact> = LISTED_FACTS.iter().map(|entry| entry.fact).collect();
    assert!(
        measured.is_disjoint(&listed),
        "{:?}",
        measured.intersection(&listed).collect::<Vec<_>>()
    );
    for fact in &measured {
        assert!(!fact.assumed_where_unmeasured(), "{fact:?}");
    }
    for entry in LISTED_FACTS {
        if entry.fact.assumed_where_unmeasured() {
            assert_eq!(entry.on.releases.until, None, "{:?}", entry.fact);
        }
    }
}

/// Each listed answer names the devices it rests on: its driver on each GPU measured, at the last
/// release measured.
#[test]
fn each_listed_answer_names_its_evidence() {
    let vendor = |driver| match driver {
        DriverId::MOLTENVK => APPLE,
        DriverId::MESA_LLVMPIPE => 0x10005,
        DriverId::MESA_RADV => 0x1002,
        DriverId::INTEL_OPEN_SOURCE_MESA => 0x8086,
        DriverId::NVIDIA_PROPRIETARY => 0x10de,
        other => panic!("no vendor for {other:?}"),
    };
    for entry in LISTED_FACTS {
        assert!(!entry.evidence.gpus.is_empty(), "{:?}", entry.fact);
        for &device in entry.evidence.gpus {
            let vendor = entry.on.vendor.unwrap_or(vendor(entry.on.driver));
            let desc = of(entry.on.driver, vendor, device, entry.evidence.last);
            assert!(
                entry.on.names(&desc),
                "{:?} on {:?} does not name {device:#x} at {:#x}",
                entry.fact,
                entry.on.driver,
                entry.evidence.last
            );
        }
    }
}

/// A device taking a listed answer past its evidence is told so, with the last release and the
/// GPUs measured: none on the measured identities whose facts the list rests on (the M4's, the
/// 780M, the gate's lavapipe and the RPL-P); on the RPL-S the first sparse resource's wait, measured
/// on the RPL-P; at MoltenVK 1.4.2 every MoltenVK answer, measured through 1.4.1. A defect is never
/// told so: the RTX 5080 Laptop GPU holds the custom-position depth, measured on the RTX 3090
/// alone, and is told nothing, every answer it takes measured on it.
#[test]
fn a_listed_answer_past_its_evidence_names_it() {
    for stem in [
        "moltenvk-m4",
        "amd-780m-radv",
        "amd-780m-hawkpoint-radv",
        "gate-lavapipe",
        "intel-irisxe-rplp-anv",
    ] {
        assert!(read(stem).listed_past_evidence().is_empty(), "{stem}");
    }
    let past =
        |stem: &str| -> Vec<DriverFact> { read(stem).listed_past_evidence().into_keys().collect() };
    assert_eq!(
        past("intel-uhd770-rpls-anv"),
        [DriverFact::FirstSparseResourceWaitsForQueues]
    );
    let laptop = read("nvidia-rtx5080-laptop");
    let positioned = DriverFact::PositionedDepthReadAtStandardPositions;
    assert!(laptop.driver_facts().contains(&positioned));
    assert!(laptop.listed_past_evidence().is_empty());

    let mut later = read("moltenvk-m4");
    later.driver_version = moltenvk(1, 4, 2);
    let evidence = later.listed_past_evidence();
    let answers: BTreeSet<DriverFact> = (later.driver_facts().into_iter())
        .filter(|fact| !fact.assumed_where_unmeasured())
        .collect();
    assert!(!answers.is_empty());
    assert_eq!(evidence.keys().copied().collect::<BTreeSet<_>>(), answers);
    for evidence in evidence.values() {
        assert_eq!(evidence.last_release, moltenvk(1, 4, 1));
    }
}

/// The RTX 5080 Laptop GPU copies a multisampled stencil to and from colour at 615.71.09 alone,
/// measured clean; every other NVIDIA GPU and release, it at another release among them, takes
/// the exact path. RADV's implicit preprocessing after an execution set holds before 26.2.4,
/// measured clean there on the 780M and the RX 7900 XTX, on every RADV GPU. No other driver has
/// either.
#[test]
fn the_listed_defects_hold_where_their_ranges_say() {
    use crate::error::RuntimeDriverDefect as D;
    let nvidia_at = |device, release| {
        listed(
            LISTED_DEFECTS,
            &of(DriverId::NVIDIA_PROPRIETARY, 0x10de, device, release),
        )
    };
    let laptop = 0x2c19;
    assert!(
        !nvidia_at(laptop, nvidia(615, 71, 9)).contains_key(&D::MultisampledStencilColourCopies)
    );
    for (device, release) in [
        (laptop, nvidia(615, 71, 8)),
        (laptop, nvidia(615, 71, 10)),
        (0x2204, nvidia(615, 71, 9)),
        (0x2684, nvidia(615, 71, 9)),
    ] {
        assert!(
            nvidia_at(device, release).contains_key(&D::MultisampledStencilColourCopies),
            "{device:#x} {release:#x}"
        );
    }
    let radv_at = |device, release| {
        listed(
            LISTED_DEFECTS,
            &of(DriverId::MESA_RADV, 0x1002, device, release),
        )
    };
    for device in [0x15bf, 0x1900, 0x744c, 0x747e] {
        assert!(
            radv_at(device, mesa(26, 0, 8)).contains_key(&D::ImplicitPreprocessAfterExecutionSet)
        );
        assert!(
            radv_at(device, mesa(26, 2, 3)).contains_key(&D::ImplicitPreprocessAfterExecutionSet)
        );
        assert!(
            !radv_at(device, mesa(26, 2, 4)).contains_key(&D::ImplicitPreprocessAfterExecutionSet)
        );
    }
    let lavapipe = listed(
        LISTED_DEFECTS,
        &of(DriverId::MESA_LLVMPIPE, 0x10005, 0, mesa(26, 0, 8)),
    );
    assert!(!lavapipe.contains_key(&D::MultisampledStencilColourCopies));
    assert!(!lavapipe.contains_key(&D::ImplicitPreprocessAfterExecutionSet));
}
