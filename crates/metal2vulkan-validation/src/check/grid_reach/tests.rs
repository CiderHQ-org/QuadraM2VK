//! Synthetic `.ll` kernels, assembled with `llvm-as`; each test skips visibly without it.

use super::*;
use crate::check::interface::tests::of_text;
use serde_json::json;

/// The interface of entry `k` of `ll`; `None` (a visible skip) without `llvm-as`.
fn interface(test: &str, ll: &str) -> Option<AirInterface> {
    match of_text(ll, "k", &BTreeMap::new()) {
        Ok(interface) => Some(interface),
        Err(why) if why.contains("unavailable") => {
            eprintln!("SKIP {test}: {why}");
            None
        }
        Err(why) => panic!("{test}: {why}"),
    }
}

/// A kernel `k` over `coord` (buffer 0, `float2`), `out` (buffer 1) and the grid position `tid`
/// (of type `position`), with `body` after an `entry:` label.
fn kernel(position: &str, body: &str) -> String {
    format!(
        r#"
target datalayout = "e-p:64:64:64-i64:64:64-f32:32:32-v64:64:64-v128:128:128"
target triple = "air64-apple-macosx14.0.0"
define void @k(ptr addrspace(1) %coord, ptr addrspace(1) %out, {position} %tid) {{
entry:
{body}
}}
!air.kernel = !{{!0}}
!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4, !5}}
!3 = !{{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"float2", !"air.arg_name", !"coord"}}
!4 = !{{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}}
!5 = !{{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}}
"#
    )
}

/// `out[tid].z = coord[tid].y`, every thread, `out` an array of three-float structs.
const EVERY_THREAD: &str = "%i = zext i32 %tid to i64
%p = getelementptr inbounds <2 x float>, ptr addrspace(1) %coord, i64 %i
%v = load <2 x float>, ptr addrspace(1) %p, align 8
%y = extractelement <2 x float> %v, i64 1
%q = getelementptr inbounds { float, float, float }, ptr addrspace(1) %out, i64 %i, i32 2
store float %y, ptr addrspace(1) %q, align 4
ret void";

#[test]
fn every_threads_element_of_a_grid_indexed_buffer_is_its_reach() {
    let test = "every_threads_element_of_a_grid_indexed_buffer_is_its_reach";
    let Some(facts) = interface(test, &kernel("i32", EVERY_THREAD)) else {
        return;
    };
    assert_eq!(
        facts.grid_reaches,
        vec![
            GridReach {
                param: 0,
                axis: 0,
                stride: 8,
                reach: 8,
            },
            GridReach {
                param: 1,
                axis: 0,
                stride: 12,
                reach: 12,
            },
        ]
    );
}

/// A component of a vector position names its axis, and only accesses on every path count: the
/// store a guard skips for some threads is not a reach.
#[test]
fn a_guarded_access_reaches_nothing_and_a_vector_position_names_its_axis() {
    let test = "a_guarded_access_reaches_nothing_and_a_vector_position_names_its_axis";
    let body = "%y = extractelement <2 x i32> %tid, i64 1
%i = zext i32 %y to i64
%p = getelementptr inbounds <2 x float>, ptr addrspace(1) %coord, i64 %i
%v = load <2 x float>, ptr addrspace(1) %p, align 8
%x = extractelement <2 x float> %v, i64 0
%small = icmp ult i32 %y, 4
br i1 %small, label %write, label %done
write:
%q = getelementptr inbounds float, ptr addrspace(1) %out, i64 %i
store float %x, ptr addrspace(1) %q, align 4
br label %done
done:
ret void";
    let Some(facts) = interface(test, &kernel("<2 x i32>", body)) else {
        return;
    };
    assert_eq!(
        facts.grid_reaches,
        vec![GridReach {
            param: 0,
            axis: 1,
            stride: 8,
            reach: 8,
        }]
    );
}

/// A case of [`EVERY_THREAD`]'s kernel over `threads` threads, `coord` holding `coord_bytes`
/// bytes and `out` enough, declaring `coord` an out-of-bounds target when `declared`.
fn case(threads: u32, coord_bytes: usize, declared: bool) -> AuthoredCase {
    let b64 = |bytes: Vec<u8>| base64::engine::general_purpose::STANDARD.encode(bytes);
    let mut case = json!({
        "air_sha256": "00".repeat(32),
        "case_id": "11".repeat(32),
        "name": "grid-reach",
        "entry": "k",
        "stage": "kernel",
        "buffers": [
            {"binding": 0, "role": "input", "bytes_b64": b64(vec![0; coord_bytes])},
            {"binding": 1, "role": "output", "initial_bytes_b64": b64(vec![0xab; 12 * threads as usize])},
        ],
        "dispatch": {"grid": [threads, 1, 1], "threads_per_threadgroup": [1, 1, 1]},
        "output": {"kind": "buffer", "binding": 1, "offset": 0, "length": 12 * threads},
        "compare": {"kind": "exact"},
        "execution_safety": "loop_free",
    });
    if declared {
        case["out_of_bounds"] = json!({
            "targets": [{
                "resource": {"kind": "buffer", "binding": 0},
                "form": "storage_buffer",
                "reach_after": 4096,
            }],
        });
    }
    serde_json::from_value(case).unwrap()
}

/// The checker refuses a buffer shorter than the element its last thread indexes (the texel
/// probe's first cases: a 15-byte `float2` buffer read by 256 threads), takes one exactly long
/// enough, and leaves a declared out-of-bounds target to its declaration.
#[test]
fn a_buffer_shorter_than_its_last_threads_element_is_refused() {
    let test = "a_buffer_shorter_than_its_last_threads_element_is_refused";
    let Some(facts) = interface(test, &kernel("i32", EVERY_THREAD)) else {
        return;
    };
    let errors = |case: AuthoredCase| {
        let mut errors = Vec::new();
        validate(&case, &facts, &mut errors);
        errors
    };
    let short = errors(case(256, 15, false));
    assert_eq!(short.len(), 1, "{short:?}");
    assert!(
        short[0].starts_with("buffer 0 holds 15 bytes")
            && short[0].contains("thread 255 reaches byte 2048"),
        "{short:?}"
    );
    assert_eq!(errors(case(256, 2047, false)).len(), 1);
    assert!(errors(case(256, 2048, false)).is_empty());
    assert!(errors(case(256, 15, true)).is_empty());
    // The output's 12-byte elements fit exactly; one byte less is refused too.
    let mut tight = case(4, 32, false);
    tight.buffers[1].initial_bytes_b64 =
        Some(base64::engine::general_purpose::STANDARD.encode([0xab; 47]));
    let tight = errors(tight);
    assert_eq!(tight.len(), 1, "{tight:?}");
    assert!(
        tight[0].starts_with("buffer 1 holds 47 bytes") && tight[0].contains("byte 48"),
        "{tight:?}"
    );
}
