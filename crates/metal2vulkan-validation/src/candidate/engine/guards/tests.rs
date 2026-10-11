use super::*;
use crate::case::AuthoredCase;
use crate::coverage::tier_table::committed_profiles;
use crate::coverage::RobustBufferForm;
use crate::executor_contract::out_of_bounds::{
    matches, Mechanism, ReadRule, TargetRun, GUARD_FILL, PAD_FILL,
};
use crate::observation_contract::ComparisonClass;
use base64::Engine as _;

/// A kernel case writing past `out` (binding 0, 16 bytes), a storage buffer target with no read
/// bytes: its output is the buffer's 16 bytes.
fn case() -> AuthoredCase {
    let b64 = |bytes: &[u8]| base64::engine::general_purpose::STANDARD.encode(bytes);
    serde_json::from_value(serde_json::json!({
        "air_sha256": "00".repeat(32),
        "case_id": "11".repeat(32),
        "name": "writes-past-a-buffer",
        "entry": "main",
        "stage": "kernel",
        "buffers": [{"binding": 0, "role": "output", "initial_bytes_b64": b64(&[0xab; 16])}],
        "dispatch": {"grid": [4, 1, 1], "threads_per_threadgroup": [4, 1, 1]},
        "output": {"kind": "buffer", "binding": 0, "offset": 0, "length": 16},
        "compare": {"kind": "exact"},
        "execution_safety": "loop_free",
        "out_of_bounds": {
            "targets": [{
                "resource": {"kind": "buffer", "binding": 0},
                "form": "storage_buffer",
                "reach_after": 256,
            }],
            "reads": [],
        },
    }))
    .unwrap()
}

/// The run of [`case`]'s one target, contained by `check`, before its guard counts.
fn run(check: Check) -> OutOfBoundsRun {
    OutOfBoundsRun {
        targets: vec![TargetRun {
            resource: OutOfBoundsResource::Buffer { binding: 0 },
            form: ContainmentForm::StorageBuffer,
            mechanism: Mechanism {
                medium: Medium::DescriptorView,
                check,
            },
            written: true,
            escaped: None,
            past_end: None,
            escape_cause: None,
            read_cause: None,
        }],
    }
}

/// A guard byte a write past a target changed in a heap of its own under `robustBufferAccess`
/// alone, which lets the write land anywhere in the memory bound to the heap's one buffer, is
/// an escape whose row says so: its cause recorded, and never a match, even where the output equals
/// Metal's. The same count in an allocation of its own, or under any other check, records no cause,
/// and a run whose guard held records none either.
#[test]
fn a_heap_guard_changed_under_robustness_1_alone_is_an_escape_with_its_cause() {
    let case = case();
    let output = vec![0xab; 16];
    let target = OutOfBoundsResource::Buffer { binding: 0 };
    let counts = |escaped| {
        BTreeMap::from([(
            target,
            GuardCount {
                escaped,
                past_end: 0,
            },
        )])
    };
    let robust = with_escaped(run(Check::Robust), &counts(4), Placement::Heap);
    assert_eq!(
        robust.targets[0].escape_cause,
        Some(EscapeCause::HeapBoundMemory)
    );
    assert_eq!(robust.targets[0].escaped, Some(4));
    assert!(robust.escapes());
    assert!(!matches(
        &case,
        &ComparisonClass::Exact,
        &output,
        &output,
        Some(&robust)
    ));
    let recorded = serde_json::to_value(&robust.targets[0]).unwrap();
    assert_eq!(recorded["escape_cause"], "heap_bound_memory");
    for (check, placement, escaped) in [
        (Check::Robust, Placement::Allocation, 4),
        (Check::Robust2, Placement::Heap, 4),
        (Check::Shader, Placement::Heap, 4),
        (Check::Robust, Placement::Heap, 0),
    ] {
        let ran = with_escaped(run(check), &counts(escaped), placement);
        assert_eq!(
            ran.targets[0].escape_cause, None,
            "{check:?} {placement:?} {escaped}"
        );
        assert_eq!(ran.escapes(), escaped > 0, "{check:?} {placement:?}");
        // A row without a cause records none.
        let recorded = serde_json::to_value(&ran.targets[0]).unwrap();
        assert!(recorded.get("escape_cause").is_none());
    }
    // A target the run did not count keeps no count and no cause.
    let uncounted = with_escaped(run(Check::Robust), &BTreeMap::new(), Placement::Heap);
    assert_eq!(uncounted.targets[0].escaped, None);
    assert_eq!(uncounted.targets[0].escape_cause, None);
}

/// Why no venue's verdict differs between the executor's guards and the runtime path's heaps
/// today. Under `robustBufferAccess` alone, a write past a target may land in memory bound to it:
/// past its end in the executor's allocation (counted as `past_end`, which that check allows), in
/// the heap's guard on the runtime path (an escape). But the only check of robustness 1 alone is
/// the one buffer robustness gives a descriptor view and fixed-function vertex input
/// (`out_of_bounds::robust_check`), and every venue whose leg takes it leaves those forms
/// uncontained, so a guarded target there is refused before it runs, on both paths. Every other
/// venue takes `robustBufferAccess2` or the shader's check. A venue where a guarded target runs
/// under robustness 1 alone fails this, and the entry's argument with it.
#[test]
fn no_venue_runs_a_guarded_target_under_robustness_1_alone() {
    let venues = crate::c3::Venues::load(&crate::c3::Venues::committed_path()).unwrap();
    let owned = venues
        .venues
        .iter()
        .map(|venue| venue.profile.as_str())
        .collect::<std::collections::BTreeSet<_>>();
    let mut level_1 = Vec::new();
    for entry in committed_profiles().unwrap() {
        if !owned.contains(entry.tier.stem.as_str()) {
            continue;
        }
        let label = entry.tier.label();
        let check = out_of_bounds::robust_check(&entry.profile).unwrap();
        if check != Check::Robust {
            assert!(
                matches!(check, Check::Robust2 | Check::Shader),
                "{label}: {check:?}"
            );
            continue;
        }
        let uncontained = entry.tier.robust_buffers_uncontained();
        for form in [
            RobustBufferForm::LoadStore,
            RobustBufferForm::Atomic,
            RobustBufferForm::VertexInput,
        ] {
            assert!(uncontained.contains(&form), "{label}: {form:?} runs");
        }
        level_1.push(label);
    }
    // MoltenVK's stated leg takes robustness 1 alone, on the Apple M3 Max's venue, so the argument
    // has a venue to hold for.
    assert_eq!(level_1, ["moltenvk-m3"]);
}

/// A vertex case fetching attribute 0 past its vertex buffer (four `uint` elements): vertex `v`
/// writes what it fetched to `out[v]` (binding 1), so output bytes 16..32 are the reads past it.
fn vertex_case() -> AuthoredCase {
    let b64 = |bytes: &[u8]| base64::engine::general_purpose::STANDARD.encode(bytes);
    let elements: Vec<u8> = (1..=4u32)
        .flat_map(|w| (0xA000_0000 | w).to_le_bytes())
        .collect();
    serde_json::from_value(serde_json::json!({
        "air_sha256": "00".repeat(32),
        "case_id": "22".repeat(32),
        "name": "fetches-past-a-vertex-buffer",
        "entry": "main",
        "stage": "vertex",
        "buffers": [{"binding": 1, "role": "output", "initial_bytes_b64": b64(&[0xab; 32])}],
        "vertex_inputs": [
            {"location": 0, "format": "uint", "stride": 4, "bytes_b64": b64(&elements)},
        ],
        "draw": {"primitive": "point", "vertex_start": 0, "vertex_count": 8, "instance_count": 1},
        "output": {"kind": "buffer", "binding": 1, "offset": 0, "length": 32},
        "compare": {"kind": "exact"},
        "execution_safety": "loop_free",
        "out_of_bounds": {
            "targets": [{
                "resource": {"kind": "attribute_buffer", "location": 0},
                "form": "vertex_input",
                "reach_after": 16,
            }],
            "reads": [{"offset": 16, "length": 16, "target": 0}],
        },
    }))
    .unwrap()
}

/// [`vertex_case`]'s output with `reads` in the bytes past its vertex buffer: the four elements,
/// then the four words read past them.
fn vertex_output(reads: [u32; 4]) -> Vec<u8> {
    (1..=4u32)
        .map(|w| 0xA000_0000 | w)
        .chain(reads)
        .flat_map(u32::to_le_bytes)
        .collect()
}

/// A fixed-function vertex fetch past an attribute buffer the runtime path placed in a heap of
/// its own reads the heap's memory: the runtime binds a placed vertex buffer to the end of the
/// heap's one buffer, so buffer robustness bounds the fetch there, as Metal's fetch reads a heap
/// neighbour's bytes. Its row records the cause, and its reads may be the heap's words (the guard
/// fill's, the padding fill's, the buffer's own) or Vulkan's out-of-range values, never another
/// value. The same fetch from an allocation of its own, and a descriptor view in a heap, keep their
/// mechanism's rule: zero under robustness 2.
#[test]
fn a_vertex_fetch_past_a_buffer_in_a_heap_reads_the_heaps_memory_with_its_cause() {
    let case = vertex_case();
    let target = OutOfBoundsResource::AttributeBuffer { location: 0 };
    let fetched = |medium| OutOfBoundsRun {
        targets: vec![TargetRun {
            resource: target,
            form: ContainmentForm::VertexInput,
            mechanism: Mechanism {
                medium,
                check: Check::Robust2,
            },
            written: false,
            escaped: None,
            past_end: None,
            escape_cause: None,
            read_cause: None,
        }],
    };
    let held = BTreeMap::from([(
        target,
        GuardCount {
            escaped: 0,
            past_end: 0,
        },
    )]);
    let golden = vertex_output([0; 4]);
    let guard = u32::from_ne_bytes([GUARD_FILL; 4]);
    let padding = u32::from_ne_bytes([PAD_FILL; 4]);
    let heap = with_escaped(fetched(Medium::VertexInput), &held, Placement::Heap);
    let ran = &heap.targets[0];
    assert_eq!(ran.read_cause, Some(ReadCause::HeapBoundRange));
    assert_eq!(ran.read_rule(), ReadRule::HeapWords);
    assert_eq!(ran.escape_cause, None);
    assert!(!heap.escapes());
    assert!(heap.tolerates_differences());
    let recorded = serde_json::to_value(ran).unwrap();
    assert_eq!(recorded["read_cause"], "heap_bound_range");
    let class = ComparisonClass::Exact;
    let judged = |run: &OutOfBoundsRun, reads| {
        matches(&case, &class, &golden, &vertex_output(reads), Some(run))
    };
    for reads in [
        [guard; 4],
        [padding, padding, guard, guard],
        [0xA000_0002, 0, 1, guard],
        [0; 4],
    ] {
        assert!(judged(&heap, reads), "{reads:x?}");
    }
    assert!(!judged(&heap, [guard, 0x1234_5678, guard, guard]));
    // An allocation of its own bounds the fetch at the buffer: zero past it.
    let own = with_escaped(fetched(Medium::VertexInput), &held, Placement::Allocation);
    assert_eq!(own.targets[0].read_cause, None);
    assert_eq!(own.targets[0].read_rule(), ReadRule::Zero);
    assert!(judged(&own, [0; 4]));
    assert!(!judged(&own, [guard; 4]));
    assert!(serde_json::to_value(&own.targets[0])
        .unwrap()
        .get("read_cause")
        .is_none());
    // A descriptor view of a buffer in a heap is bound at the buffer's own range.
    let view = with_escaped(fetched(Medium::DescriptorView), &held, Placement::Heap);
    assert_eq!(view.targets[0].read_cause, None);
    // A target the run did not place between guards records no cause.
    let unplaced = with_escaped(
        fetched(Medium::VertexInput),
        &BTreeMap::new(),
        Placement::Heap,
    );
    assert_eq!(unplaced.targets[0].read_cause, None);
}
