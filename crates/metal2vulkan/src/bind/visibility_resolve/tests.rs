//! The visibility resolve module: valid on every committed profile (L3, `G-val`), and exact on
//! the CPU interpreter (Metal's 1 or 0 for any count), in place, for the listed words only,
//! across workgroups and short ranges; and its plan's windows.

use super::*;
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

fn bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// Run the resolve of `targets` over `dest` (each bound whole) in `groups` workgroups, in both
/// invocation orders; the destination's words after it.
fn resolve(targets: &[u32], dest: &[u32], groups: u32) -> Vec<u32> {
    let m = typed_module().expect("module");
    let mut out = Vec::new();
    for order in [Order::Forward, Order::Reverse] {
        let res = Resources {
            buffers: vec![
                Buffer {
                    slot: Some(TARGETS),
                    address: None,
                    bytes: bytes(targets),
                },
                Buffer {
                    slot: Some(DEST),
                    address: None,
                    bytes: bytes(dest),
                },
            ],
            push: Vec::new(),
            spec: Default::default(),
        };
        let cfg = Config {
            order,
            workgroups: [groups, 1, 1],
            ..Config::default()
        };
        out.push(run(&m, &cfg, res).expect("run").buffers);
    }
    assert_eq!(out[0], out[1], "order independent");
    words(&out[0][1])
}

#[test]
fn the_resolve_module_is_valid_on_every_committed_profile() {
    for (_, p) in committed_profiles().expect("profiles") {
        assert_valid(typed_module().expect("module"), &p);
    }
}

/// Any nonzero count, in either half of the 64-bit word, becomes Metal's 1, and zero stays 0;
/// the high half becomes 0. Words not listed keep their bytes.
#[test]
fn a_count_becomes_metals_one_or_zero_in_place() {
    for (count, want) in [
        ([0, 0], 0),
        ([1, 0], 1),
        ([16, 0], 1),
        ([u32::MAX, u32::MAX], 1),
        ([0, 1], 1),
    ] {
        let dest = [0xAAAA_AAAA, 0xAAAA_AAAA, count[0], count[1], 7, 9];
        let out = resolve(&[1], &dest, 1);
        assert_eq!(out, [0xAAAA_AAAA, 0xAAAA_AAAA, want, 0, 7, 9], "{count:?}");
    }
}

/// Every listed word is rewritten, across workgroups, and a counting word between them is
/// left as copied; an invocation past the table does nothing.
#[test]
fn every_listed_word_resolves_across_workgroups() {
    let n = VisibilityResolveModule::LOCAL_SIZE + 3;
    // Word 2k holds the count k (0 for k = 0); the odd words are counting results, untouched.
    let mut dest = Vec::new();
    for k in 0..2 * n {
        dest.extend([k, 0]);
    }
    let targets: Vec<u32> = (0..n).map(|k| 2 * k).collect();
    let out = resolve(&targets, &dest, 2);
    for k in 0..2 * n {
        let want = if k % 2 == 1 {
            [k, 0]
        } else {
            [u32::from(k != 0), 0]
        };
        assert_eq!(out[2 * k as usize..2 * k as usize + 2], want, "word {k}");
    }
}

/// A word whose high half lies past the bound range is skipped, and so is a target past it;
/// nothing faults.
#[test]
fn words_past_the_bound_range_are_skipped() {
    assert_eq!(resolve(&[0, 1, 9], &[5, 0, 6], 1), [1, 0, 6]);
    assert_eq!(resolve(&[0], &[5], 1), [5]);
}

#[test]
fn the_plan_binds_aligned_windows() {
    let plan = VisibilityResolve::plan(&[72, 24, 72], 64, 1 << 16, 65_535).expect("plan");
    assert_eq!(plan.len(), 1);
    assert_eq!(
        (plan[0].dest_range.offset, plan[0].dest_range.range),
        (0, 80)
    );
    assert_eq!(plan[0].targets, [3, 9], "sorted, each once");
    assert_eq!(plan[0].group_count, 1);
    assert_eq!(plan[0].target_bytes(), [3, 0, 0, 0, 9, 0, 0, 0]);
    // Results farther apart than one binding reaches take a dispatch each, from an aligned
    // offset.
    let plan = VisibilityResolve::plan(&[8, 200], 16, 64, 65_535).expect("plan");
    let ranges: Vec<_> = plan
        .iter()
        .map(|r| (r.dest_range.offset, r.dest_range.range, r.targets.clone()))
        .collect();
    assert_eq!(ranges, [(0, 16, vec![1]), (192, 16, vec![1])]);
    // Too many for one dispatch.
    let offsets: Vec<u64> = (0..65).map(|k| 8 * k).collect();
    let plan = VisibilityResolve::plan(&offsets, 8, 1 << 16, 1).expect("plan");
    assert_eq!(plan.len(), 2);
    assert_eq!(plan[0].targets.len(), 64);
    assert_eq!(plan[1].targets, [0]);
    assert_eq!(plan[1].dest_range.offset, 512);
    assert_eq!(VisibilityResolve::plan(&[], 64, 64, 1), Some(Vec::new()));
}

#[test]
fn the_plan_refuses_what_it_cannot_bind() {
    assert_eq!(VisibilityResolve::plan(&[12], 8, 64, 1), None, "misaligned");
    assert_eq!(
        VisibilityResolve::plan(&[u64::MAX - 7], 8, 64, 1),
        None,
        "an overflow"
    );
    assert_eq!(
        VisibilityResolve::plan(&[56], 64, 32, 1),
        None,
        "a word past the range from its aligned offset"
    );
    assert_eq!(VisibilityResolve::plan(&[0], 8, 64, 0), None, "no groups");
}

#[test]
fn the_device_module_is_numbered_with_the_fixed_bindings() {
    let (_, p) = committed_profiles()
        .expect("profiles")
        .into_iter()
        .find(|(s, _)| s == "vk12-floor")
        .expect("vk12-floor");
    let r = crate::spirv::serialize::serialize(typed_module().expect("module")).expect("serialize");
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
            VisibilityResolveModule::TARGETS_BINDING,
            VisibilityResolveModule::DEST_BINDING
        ]
    );
}
