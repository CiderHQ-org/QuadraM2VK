//! Named loop-exit shapes with the loop membership they get: every loop is its natural loop, and
//! no exit tail joins it (retiring the P3). A tail runs after its loop,
//! behind the loop's exit dispatch, with every invocation that left through it, as AGX runs it.

use std::collections::{BTreeMap, BTreeSet};

use super::{check_ok, meter};
use crate::structurize::cfg::Content;
use crate::structurize::prepare;
use crate::structurize::testkit::TestCfg;

/// Every loop of `succ` after P0–P2, as header -> blocks (original block numbers).
fn loops_of(succ: &[Vec<u32>]) -> BTreeMap<u32, BTreeSet<u32>> {
    let g = TestCfg::from_successors(succ);
    check_ok(&g, 0..8);
    let mut m = meter();
    let p = prepare(&g, &mut m).unwrap();
    let original = |n| match p.cfg.block(n).unwrap().content {
        Content::Code(b) => b.raw(),
        other => panic!("synthetic node {other:?} in a reducible input"),
    };
    let loops = p
        .loops
        .loops()
        .map(|l| {
            let h = original(p.loops.header(l).unwrap());
            (h, p.loops.blocks(l).map(original).collect())
        })
        .collect();
    p.release(&mut m);
    loops
}

fn set(blocks: &[u32]) -> BTreeSet<u32> {
    blocks.iter().copied().collect()
}

#[test]
fn a_break_tail_that_rejoins_the_normal_exit_stays_after_the_loop() {
    // while (h) { if (p) { tail; break; } latch }  exit
    // 0 -> 1; h1 -> {2, 5}; 2 -> {3, 4}; tail3 -> 5; latch4 -> 1; 5 ret.
    let loops = loops_of(&[vec![1], vec![2, 5], vec![3, 4], vec![5], vec![1], vec![]]);
    assert_eq!(loops, BTreeMap::from([(1, set(&[1, 2, 4]))]));
}

#[test]
fn a_single_exit_keeps_post_loop_code_after_the_loop() {
    // while (h) { if (p) break; latch }  post; ret
    let loops = loops_of(&[vec![1], vec![2, 4], vec![4, 3], vec![1], vec![5], vec![]]);
    assert_eq!(loops, BTreeMap::from([(1, set(&[1, 2, 3]))]));
}

#[test]
fn a_returning_tail_and_post_loop_code_both_stay_after_the_loop() {
    // while (h) { if (p) { tail; return; } latch }  post; ret
    let loops = loops_of(&[
        vec![1],
        vec![2, 5],
        vec![3, 4],
        vec![],
        vec![1],
        vec![6],
        vec![],
    ]);
    assert_eq!(loops, BTreeMap::from([(1, set(&[1, 2, 4]))]));
}

#[test]
fn two_break_tails_joining_after_the_loop_both_stay_after_it() {
    // for (;;) { if (a) { A; break; } if (b) { B; break; } }  J
    // 0 -> 1; h1 -> {A2, 3}; A2 -> 6; 3 -> {B4, latch5}; B4 -> 6; latch5 -> 1; J6 ret.
    let loops = loops_of(&[
        vec![1],
        vec![2, 3],
        vec![6],
        vec![4, 5],
        vec![6],
        vec![1],
        vec![],
    ]);
    assert_eq!(loops, BTreeMap::from([(1, set(&[1, 3, 5]))]));
}

#[test]
fn a_loop_in_a_tail_is_a_sibling_after_the_loop() {
    // while (h) { if (p) { T; while (c) { body } break; } latch }  E
    // (the inner loop's exit edge goes straight to E: the `break` after it is folded in)
    // 0 -> 1; h1 -> {2, E6}; 2 -> {T3, latch5}; T3 -> h4; h4 -> {body7, E6}; body7 -> h4;
    // latch5 -> 1; E6 ret. The tail {3, 4, 7} stays after loop 1, so loop 4 is not nested in it.
    let loops = loops_of(&[
        vec![1],
        vec![2, 6],
        vec![3, 5],
        vec![4],
        vec![7, 6],
        vec![1],
        vec![],
        vec![4],
    ]);
    assert_eq!(
        loops,
        BTreeMap::from([(1, set(&[1, 2, 5])), (4, set(&[4, 7]))])
    );
}

#[test]
fn continuing_an_outer_loop_leaves_only_the_inner_loop() {
    // outer: while (ho) { inner: while (hi) { if (b) { B; continue outer; } latch } post; }
    // 0 -> 1; ho1 -> {hi2, 8}; hi2 -> {3, post6}; 3 -> {B4, latch5}; B4 -> 1; latch5 -> 2;
    // post6 -> 7; 7 -> 1; 8 ret.
    let loops = loops_of(&[
        vec![1],
        vec![2, 8],
        vec![3, 6],
        vec![4, 5],
        vec![1],
        vec![2],
        vec![7],
        vec![1],
        vec![],
    ]);
    assert_eq!(
        loops,
        BTreeMap::from([(1, set(&[1, 2, 3, 4, 5, 6, 7])), (2, set(&[2, 3, 5]))])
    );
}

#[test]
fn a_multi_level_break_tail_stays_after_both_loops() {
    // outer: while (ho) { inner: while (hi) { if (p) { T; break outer; } latch } post; }  E
    // 0 -> 1; ho1 -> {hi2, E8}; hi2 -> {3, post6}; 3 -> {T4, latch5}; T4 -> 8; latch5 -> 2;
    // post6 -> 7; 7 -> 1; E8 ret. T4 leaves both loops, so it runs after the outer one.
    let loops = loops_of(&[
        vec![1],
        vec![2, 8],
        vec![3, 6],
        vec![4, 5],
        vec![8],
        vec![2],
        vec![7],
        vec![1],
        vec![],
    ]);
    assert_eq!(
        loops,
        BTreeMap::from([(1, set(&[1, 2, 3, 5, 6, 7])), (2, set(&[2, 3, 5]))])
    );
}

#[test]
fn a_front_end_multi_level_break_runs_its_tail_after_the_inner_loop() {
    // The CFG Metal's front end gives a multi-level break (the A3 fixture
    // `a3_multi_level_break_tail`): both inner exits meet in one join that re-tests the break to
    // leave the outer loop. 0 -> ho1; ho1 -> hi2; hi2 -> {T3, latch4}; T3 -> J6 (the tail, with the
    // `simd_or`); latch4 -> {post5, hi2}; post5 -> J6; J6 -> {E8, outer latch7}; 7 -> {ho1, E8};
    // E8 ret. The tail T3 is outside the inner loop {2, 4} and inside the outer one: it runs once
    // per outer iteration with the invocations that broke in it, as AGX answered (032467bc).
    let loops = loops_of(&[
        vec![1],
        vec![2],
        vec![3, 4],
        vec![6],
        vec![5, 2],
        vec![6],
        vec![8, 7],
        vec![1, 8],
        vec![],
    ]);
    assert_eq!(
        loops,
        BTreeMap::from([(1, set(&[1, 2, 3, 4, 5, 6, 7])), (2, set(&[2, 4]))])
    );
}

#[test]
fn switch_in_loop_with_shared_targets_and_a_default_merge() {
    // h1 -> switch { default: latch5, 0: 2, 1: 3, 2: 2 } ; 2 -> 5; 3 -> {4, 5}; 4 -> 6 (break);
    // latch5 -> {1, 6}; 6 ret. Exit targets are 4 (from 3) and 6 (from latch5); both stay after.
    let loops = loops_of(&[
        vec![1],
        vec![5, 2, 3, 2],
        vec![5],
        vec![4, 5],
        vec![6],
        vec![1, 6],
        vec![],
    ]);
    assert_eq!(loops, BTreeMap::from([(1, set(&[1, 2, 3, 5]))]));
}
