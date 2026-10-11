//! The indirect-clamp module: valid on every committed profile (L3, `G-val`), and exact on the
//! CPU interpreter (clamp and `OVERFLOW`), including the source residual and a short range.

use super::*;
use crate::bind::status::{ExecutionStatus, StatusReport};
use crate::spirv::interp::{run, Buffer, Config, Order, Resources};
use crate::spirv::testing::{assert_valid, committed_profiles};

fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect()
}

/// Run the clamp for `limits` over `source` (bound whole) with push `residual`; the scratch
/// words and the status report.
fn clamp(limits: [u32; 3], source: &[u32], residual: u32) -> ([u32; 3], StatusReport) {
    let m = typed_module(limits).expect("module");
    let status = ExecutionStatus::initial();
    let mut out = Vec::new();
    for order in [Order::Forward, Order::Reverse] {
        let res = Resources {
            buffers: vec![
                Buffer {
                    slot: Some(SOURCE),
                    address: None,
                    bytes: source.iter().flat_map(|w| w.to_le_bytes()).collect(),
                },
                Buffer {
                    slot: Some(SCRATCH),
                    address: None,
                    bytes: vec![0xEE; 16],
                },
                Buffer {
                    slot: Some(STATUS),
                    address: None,
                    bytes: status.to_vec(),
                },
            ],
            push: residual.to_le_bytes().to_vec(),
            spec: Default::default(),
        };
        let cfg = Config {
            order,
            ..Config::default()
        };
        out.push(run(&m, &cfg, res).expect("run").buffers);
    }
    assert_eq!(out[0], out[1], "order independent");
    let b = &out[0];
    let scratch = words(&b[1]);
    assert_eq!(scratch[3], 0xEEEE_EEEE, "the clamp writes 12 bytes only");
    let status: [u8; 32] = b[2].as_slice().try_into().expect("status");
    (
        [scratch[0], scratch[1], scratch[2]],
        ExecutionStatus::decode(&status),
    )
}

#[test]
fn the_clamp_module_is_valid_on_every_committed_profile() {
    for (_, p) in committed_profiles().expect("profiles") {
        let limits = p.fields().limits.max_compute_work_group_count;
        assert_valid(typed_module(limits).expect("module"), &p);
    }
}

#[test]
fn counts_within_the_limits_are_copied_exactly() {
    let limits = [65535, 65535, 65535];
    let (out, report) = clamp(limits, &[1, 65535, 7, 0], 0);
    assert_eq!(out, [1, 65535, 7]);
    assert!(!report.overflow);
    let (out, report) = clamp(limits, &[0, 0, 0], 0);
    assert_eq!(out, [0, 0, 0]);
    assert!(!report.overflow);
}

#[test]
fn counts_above_the_limits_are_clamped_and_reported() {
    let limits = [65535, 1024, 64];
    for (src, want) in [
        ([65536, 1, 1], [65535, 1, 1]),
        ([1, u32::MAX, 1], [1, 1024, 1]),
        ([1, 1, 65], [1, 1, 64]),
        ([u32::MAX; 3], [65535, 1024, 64]),
    ] {
        let (out, report) = clamp(limits, &src, 0);
        assert_eq!(out, want, "{src:?}");
        assert!(report.overflow, "{src:?}");
        assert!(!report.resident_miss);
    }
}

#[test]
fn the_residual_selects_the_arguments_and_a_short_range_reads_zero() {
    let limits = [100, 100, 100];
    // Arguments at byte 8 of the bound range (the range starts at an aligned offset below them).
    let (out, report) = clamp(limits, &[9, 9, 3, 4, 5, 9], 8);
    assert_eq!(out, [3, 4, 5]);
    assert!(!report.overflow);
    // A range that ends inside the arguments: the missing words are 0, never a fault.
    let (out, report) = clamp(limits, &[500, 2], 4);
    assert_eq!(out, [2, 0, 0]);
    assert!(!report.overflow);
}

#[test]
fn the_prepass_binds_an_aligned_source_range() {
    let c = IndirectClamp::new(IndirectRef::new(3, 300), [65535; 3], 256).expect("clamp");
    assert_eq!(c.source_range.offset, 256);
    assert_eq!(c.source_range.range, 56);
    assert_eq!(u32::from_le_bytes(c.push_constants), 44);
    let c = IndirectClamp::new(IndirectRef::new(3, 12), [65535; 3], 1).expect("clamp");
    assert_eq!((c.source_range.offset, c.source_range.range), (12, 12));
    assert_eq!(u32::from_le_bytes(c.push_constants), 0);
    assert_eq!(IndirectClamp::new(IndirectRef::new(3, 2), [1; 3], 4), None);
}

#[test]
fn the_device_module_is_numbered_with_the_fixed_bindings() {
    let (_, p) = committed_profiles()
        .expect("profiles")
        .into_iter()
        .find(|(s, _)| s == "vk12-floor")
        .expect("vk12-floor");
    let r = crate::spirv::serialize::serialize(typed_module([7, 8, 9]).expect("module"))
        .expect("serialize");
    let words = crate::spirv::serialize::relocate(&r, &binding_of).expect("relocate");
    crate::spirv::testing::check_words_valid(&words, &r.requirements, &p).expect("L3");
    let mut bindings: Vec<u32> = r
        .relocs
        .iter()
        .filter(|(s, _)| matches!(s, Symbol::Binding(_)))
        .map(|(_, at)| words[*at as usize])
        .collect();
    bindings.sort_unstable();
    assert_eq!(
        bindings,
        [
            IndirectClampModule::SOURCE_BINDING,
            IndirectClampModule::SCRATCH_BINDING,
            IndirectClampModule::STATUS_BINDING
        ]
    );
}
