//! Structurizer property tests: exhaustive and generated CFGs, CFG-level trace
//! equivalence after every pass that changes edges, tree-level trace equivalence after P6 and P7,
//! the structural checker, the O((n+e)·d) and output size bounds, typed refusals, ledger
//! balance, determinism, and linear work. Every graph `check` accepts runs through all of P0–P7.
//!
//! **Gate and soak**. Every graph of at most five blocks with out-degree at most two and of at most
//! four blocks with switches is checked in every run. The two large families, six blocks with
//! out-degree at most two (710,556 graphs) and five blocks with switches (15,794,698), are
//! enumerated in full (their counts are asserted) and checked on a deterministic hash sample
//! (`generate::in_sample`, one in [`SIX_BLOCK_SAMPLE`] and one in [`FIVE_BLOCK_SWITCH_SAMPLE`]) in
//! release (G-rel). Each sampled graph gets every check above. The ignored `soak_` tests check
//! every graph of both families:
//!
//! ```text
//! cargo test -p metal2vulkan --all-features --release --lib -- --ignored structurizer_properties::soak_
//! ```

use std::collections::BTreeSet;
use std::ops::Range;

use super::cfg::{Content, Node, StructCfg, Term};
use super::irreducible;
use super::loops::{LoopNest, Loops};
use super::prune;
use super::testkit::generate::{self, Rng};
use super::testkit::interp::{check_trace_equivalence, describe};
use super::testkit::TestCfg;
use super::{prepare, Prepared};
use crate::air::ir::BlockId;
use crate::budget::{output_blocks_cap, Budget, Meter, IRREDUCIBLE_DEPTH};
use crate::error::{Error, Feature, Limit, Needs};
use crate::graph::{Cfg, DomTree, Graph, LoopForest};

mod loop_exits_cases;
mod trees;

fn meter() -> Meter {
    Meter::new(&Budget::default())
}

/// The number of original blocks traced per seed.
const EVENTS: usize = 40;

/// The irreducible nesting depth of a graph: the most irreducible loops on one nesting chain.
fn irreducible_depth(sc: &StructCfg) -> u64 {
    let mut m = meter();
    let cfg = Cfg::build(sc, &mut m).unwrap();
    let Ok(forest) = LoopForest::compute(&cfg, &mut m) else {
        return IRREDUCIBLE_DEPTH + 1; // refused by the forest itself
    };
    let mut depth = vec![0u64; forest.len()];
    let mut max = 0;
    for l in forest.loops() {
        let own = u64::from(!forest.is_reducible(l));
        let d = forest.parent(l).map_or(0, |p| depth[p.index()]) + own;
        depth[l.index()] = d;
        max = max.max(d);
    }
    max
}

/// P0 postconditions: every node reachable, each reachable original block in exactly one `Code`
/// node, no `CondBr(c, t, t)`, switches grouped by target without cases to the default.
fn check_p0(g: &TestCfg, sc: &StructCfg) {
    let mut m = meter();
    let original = Cfg::build(g, &mut m).unwrap();
    let pruned = Cfg::build(sc, &mut m).unwrap();
    assert_eq!(
        pruned.reachable_count(),
        sc.len(),
        "P0 kept an unreachable node"
    );
    assert_eq!(sc.len(), original.reachable_count());
    assert_eq!(sc.entry(), Node::new(0));
    let mut seen = BTreeSet::new();
    for (n, b) in sc.blocks() {
        let Content::Code(o) = b.content else {
            panic!("P0 produced a synthetic node {n:?}");
        };
        assert!(original.is_reachable(o));
        assert!(seen.insert(o), "{o:?} placed twice");
        assert_eq!(sc.node_of(o), Some(n));
        match &b.term {
            Term::CondBr { t, f, .. } => assert_ne!(t, f),
            Term::Switch { default, cases, .. } => {
                let targets: BTreeSet<Node> = cases.iter().map(|(_, t)| *t).collect();
                assert_eq!(targets.len(), cases.len(), "switch targets not grouped");
                assert!(!targets.contains(default));
                assert!(cases.iter().all(|(lits, _)| !lits.is_empty()));
            }
            _ => {}
        }
    }
}

/// Checks on the frozen loops against the graph they describe.
fn check_loops(sc: &StructCfg, loops: &Loops) {
    let mut m = meter();
    let cfg = Cfg::build(sc, &mut m).unwrap();
    let dom = DomTree::compute(&cfg, &mut m).unwrap();
    let natural = LoopForest::compute(&cfg, &mut m).unwrap();
    assert!(natural.is_reducible_graph());
    assert_eq!(
        natural.len(),
        loops.len(),
        "P2 must not create or drop loops"
    );
    for l in loops.loops() {
        let h = loops.header(l).unwrap();
        assert_eq!(loops.loop_of_header(h), Some(l));
        let blocks: BTreeSet<Node> = loops.blocks(l).collect();
        assert_eq!(blocks.len(), loops.blocks(l).len(), "a block listed twice");
        // The loop is the natural loop of the same header: no exit tail joins it.
        let nl = natural.loop_of_header(h).expect("header of a natural loop");
        let natural_blocks: BTreeSet<Node> = natural.blocks(nl).collect();
        assert_eq!(blocks, natural_blocks, "{l:?} is not its natural loop");
        for &b in &blocks {
            assert!(loops.contains(l, b));
            assert!(dom.dominates(h, b), "{h:?} does not dominate member {b:?}");
            if b != h {
                for p in cfg.predecessors(b) {
                    assert!(
                        blocks.contains(&p),
                        "{l:?} entered at {b:?} from {p:?}\n{}",
                        describe(sc)
                    );
                }
            }
        }
        if let Some(p) = loops.parent(l) {
            assert!(loops.loop_contains(p, l));
            assert!(blocks.iter().all(|&b| loops.contains(p, b)));
        }
        // Exit targets: exactly the successors outside the loop, in RPO.
        let mut exits: Vec<Node> = blocks
            .iter()
            .flat_map(|&b| sc.successors(b).collect::<Vec<_>>())
            .filter(|s| !blocks.contains(s))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        exits.sort_by_key(|&s| cfg.rpo_number(s));
        assert_eq!(loops.exits(l).collect::<Vec<_>>(), exits);
    }
    for (n, _) in sc.blocks() {
        let member_of: Vec<_> = loops.loops().filter(|&l| loops.contains(l, n)).collect();
        match loops.innermost(n) {
            None => assert!(member_of.is_empty()),
            Some(i) => assert!(member_of.iter().all(|&l| loops.loop_contains(l, i))),
        }
    }
}

/// Run P0–P2 step by step on `g`, checking every pass. Returns the typed refusal, if any.
fn check(g: &TestCfg, seeds: Range<u64>) -> Result<(), Error> {
    let mut m = meter();
    let mut sc = prune::prune(g, &mut m).unwrap();
    assert_eq!(
        m.ledger(),
        sc.ledger_bytes() as u64,
        "P0 scratch not released"
    );
    check_p0(g, &sc);
    check_trace_equivalence(g, &sc, seeds.clone(), EVENTS)
        .unwrap_or_else(|e| panic!("P0 changed the traces: {e}\n{g:?}\n{}", describe(&sc)));

    let (n0, e0) = (sc.len() as u64, sc.edge_count() as u64);
    let depth = irreducible_depth(&sc);
    if let Err(e) = irreducible::resolve(&mut sc, g, &mut m) {
        assert_eq!(
            m.ledger(),
            sc.ledger_bytes() as u64,
            "P1 refusal leaked ledger bytes"
        );
        return Err(e);
    }
    assert_eq!(
        m.ledger(),
        sc.ledger_bytes() as u64,
        "P1 scratch not released"
    );
    assert_eq!(irreducible_depth(&sc), 0, "P1 left the graph irreducible");
    check_trace_equivalence(g, &sc, seeds.clone(), EVENTS)
        .unwrap_or_else(|e| panic!("P1 changed the traces: {e}\n{g:?}\n{}", describe(&sc)));
    let grown = sc.len() as u64;
    assert!(grown <= output_blocks_cap(n0, e0));
    assert!(
        grown <= n0 + 3 * n0 * depth,
        "P1 output {grown} exceeds n + 3·n·d for n = {n0}, d = {depth}"
    );
    if depth == 0 {
        assert_eq!(grown, n0, "P1 changed a reducible graph");
    }

    let before = describe(&sc);
    let nest = LoopNest::compute(&sc, &mut m).unwrap();
    let loops = nest.finish(&sc, &mut m).unwrap();
    assert_eq!(describe(&sc), before, "P2 changed an edge");
    assert_eq!(
        m.ledger(),
        (sc.ledger_bytes() + loops.ledger_bytes()) as u64,
        "P2 scratch not released"
    );
    check_loops(&sc, &loops);
    let prepared = Prepared {
        cfg: sc,
        loops,
        output_cap: output_blocks_cap(n0, e0),
    };
    trees::check_stages(g, &prepared, seeds, &mut m);
    prepared.release(&mut m);
    assert_eq!(m.ledger(), 0);
    Ok(())
}

/// [`check`], requiring success.
fn check_ok(g: &TestCfg, seeds: Range<u64>) {
    if let Err(e) = check(g, seeds) {
        panic!("unexpected refusal {e:?} for {g:?}");
    }
}

/// [`check`], allowing only the depth refusal (for families that may nest deeper than the cap).
fn check_within_depth(g: &TestCfg, seeds: Range<u64>) {
    match check(g, seeds) {
        Ok(()) | Err(Error::LimitExceeded(Limit::IrreducibleDepth)) => {}
        Err(e) => panic!("unexpected refusal {e:?} for {g:?}"),
    }
}

/// One in this many of the 710,556 six-block out-degree-two graphs is checked in the gate.
const SIX_BLOCK_SAMPLE: u64 = 16;
/// One in this many of the 15,794,698 five-block switch graphs is checked in the gate.
const FIVE_BLOCK_SWITCH_SAMPLE: u64 = 256;

/// Check the canonical graphs of `n` blocks over `choices` in the one-in-`every` sample; returns
/// how many were enumerated and how many checked.
fn exhaustive_sampled(n: u32, choices: Vec<Vec<u32>>, every: u64) -> (u64, u64) {
    let mut checked = 0;
    let total = generate::canonical_sampled(
        n,
        &choices,
        |i| generate::in_sample(i, every),
        |g| {
            check_ok(g, 0..3);
            checked += 1;
        },
    );
    (total, checked)
}

/// Check every canonical graph of `n` blocks over `choices`; returns how many.
fn exhaustive(n: u32, choices: Vec<Vec<u32>>) -> u64 {
    let (total, checked) = exhaustive_sampled(n, choices, 1);
    assert_eq!(total, checked);
    total
}

#[test]
fn exhaustive_out_degree_two_up_to_five_blocks() {
    let counts: Vec<u64> = (1..=5)
        .map(|n| exhaustive(n, generate::out_degree_two_choices(n)))
        .collect();
    assert_eq!(counts, [2, 8, 91, 1419, 28752]);
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "44k six-block graphs: runs in release (G-rel: cargo test --release -- structurizer_properties)"
)]
fn sampled_out_degree_two_six_blocks() {
    let (total, checked) =
        exhaustive_sampled(6, generate::out_degree_two_choices(6), SIX_BLOCK_SAMPLE);
    assert_eq!(total, 710_556);
    eprintln!("six-block graphs: {total}, checked {checked}");
    assert!(
        checked * SIX_BLOCK_SAMPLE > total * 9 / 10 && checked * SIX_BLOCK_SAMPLE < total * 11 / 10,
        "sampled {checked} of {total}"
    );
}

#[test]
#[ignore = "soak: all 710,556 six-block graphs (cargo test --release --lib -- --ignored structurizer_properties::soak_)"]
fn soak_exhaustive_out_degree_two_six_blocks() {
    assert_eq!(exhaustive(6, generate::out_degree_two_choices(6)), 710_556);
}

#[test]
fn exhaustive_switches_up_to_four_blocks() {
    let counts: Vec<u64> = (1..=4)
        .map(|n| exhaustive(n, generate::switch_choices(n)))
        .collect();
    assert!(counts.iter().all(|&c| c > 0));
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "62k five-block switch graphs: runs in release (G-rel)"
)]
fn sampled_switches_five_blocks() {
    let (total, checked) =
        exhaustive_sampled(5, generate::switch_choices(5), FIVE_BLOCK_SWITCH_SAMPLE);
    assert_eq!(total, 15_794_698);
    eprintln!("five-block switch graphs: {total}, checked {checked}");
    assert!(
        checked * FIVE_BLOCK_SWITCH_SAMPLE > total * 9 / 10
            && checked * FIVE_BLOCK_SWITCH_SAMPLE < total * 11 / 10,
        "sampled {checked} of {total}"
    );
}

#[test]
#[ignore = "soak: all 15,794,698 five-block switch graphs (cargo test --release --lib -- --ignored structurizer_properties::soak_)"]
fn soak_exhaustive_switches_five_blocks() {
    assert_eq!(exhaustive(5, generate::switch_choices(5)), 15_794_698);
}

#[test]
fn structured_programs_with_goto_injection() {
    for seed in 0..400 {
        let g = generate::structured(seed, 2 + (seed % 4) as u32, 0);
        check_ok(&g, 0..4);
    }
}

#[test]
fn structured_programs_with_irreducible_crossings() {
    let mut irreducible = 0;
    for seed in 0..400 {
        let g = generate::structured(seed, 3 + (seed % 3) as u32, 1 + (seed % 3) as u32);
        let mut m = meter();
        let sc = prune::prune(&g, &mut m).unwrap();
        irreducible += u64::from(irreducible_depth(&sc) > 0);
        check_within_depth(&g, 0..4);
    }
    assert!(
        irreducible > 50,
        "only {irreducible} irreducible programs generated"
    );
}

#[test]
fn random_dags_with_back_edges() {
    let mut rng = Rng::new(0x5eed_12a1);
    let (mut reducible, mut irreducible) = (0, 0);
    for _ in 0..1500 {
        let n = 2 + rng.below(40) as u32;
        let g = generate::dag_with_back_edges(n, rng.next_u64());
        let mut m = meter();
        let sc = prune::prune(&g, &mut m).unwrap();
        if irreducible_depth(&sc) == 0 {
            reducible += 1;
        } else {
            irreducible += 1;
        }
        check_within_depth(&g, 0..3);
    }
    assert!(
        reducible > 300 && irreducible > 40,
        "{reducible} / {irreducible}"
    );
}

#[test]
fn irreducible_families_resolve_within_the_size_bound() {
    for k in 1..=12 {
        check_ok(&generate::irreducible_ladder(k), 0..4);
    }
    for k in 1..=IRREDUCIBLE_DEPTH as u32 {
        check_ok(&generate::nested_irreducible(k), 0..4);
    }
}

#[test]
fn irreducible_nesting_beyond_the_cap_is_refused() {
    let g = generate::nested_irreducible(IRREDUCIBLE_DEPTH as u32 + 1);
    assert_eq!(
        check(&g, 0..1),
        Err(Error::LimitExceeded(Limit::IrreducibleDepth))
    );
    let mut m = meter();
    assert_eq!(
        prepare(&g, &mut m).unwrap_err(),
        Error::LimitExceeded(Limit::IrreducibleDepth)
    );
    assert_eq!(m.ledger(), 0);
}

#[test]
fn tangled_irreducible_regions_are_refused_at_the_block() {
    // bb0 -> {bb1, bb2}, bb1 <-> bb2, bb2 -> bb3.
    let ladder = generate::irreducible_ladder(1);
    for tangled in [1u32, 2] {
        let g = ladder.clone().with_tangled(&[tangled]);
        let err = check(&g, 0..1).unwrap_err();
        let Error::Unsupported(u) = err else {
            panic!("expected Unsupported, got {err:?}");
        };
        assert_eq!(u.feature, Feature::IrreducibleConvergent);
        assert_eq!(u.needs, Needs::NotRepresentable);
        let expected = super::CfgView::location(&g, BlockId::new(tangled));
        assert_eq!(u.location, Some(expected));
    }
    // Tangled code outside the irreducible cycle is fine.
    check_ok(&ladder.clone().with_tangled(&[0, 3]), 0..3);
    // And tangled code in reducible loops is never refused by P1.
    check_ok(
        &generate::break_tail_ladder(3).with_tangled(&[0, 1, 2, 3]),
        0..3,
    );
}

#[test]
fn output_cap_is_checked_before_allocating() {
    let g = generate::irreducible_ladder(3);
    let mut m = meter();
    let mut sc = prune::prune(&g, &mut m).unwrap();
    let (len, bytes) = (sc.len(), sc.ledger_bytes());
    assert_eq!(
        irreducible::resolve_capped(&mut sc, &g, &mut m, len as u64 + 2),
        Err(Error::LimitExceeded(Limit::OutputBlocks))
    );
    assert_eq!(sc.len(), len, "a refused round added nodes");
    assert_eq!(sc.ledger_bytes(), bytes);
    assert_eq!(m.ledger(), bytes as u64);
    // The standard cap is far above what dispatch needs.
    irreducible::resolve(&mut sc, &g, &mut m).unwrap();
}

/// Run `prepare` under a sweep of ledger and work caps: every refusal leaves the ledger empty.
fn check_release_on_error(g: &TestCfg) {
    let budget = Budget::default();
    let mut full = meter();
    let prepared = prepare(g, &mut full).unwrap();
    let (peak, work) = (full.ledger_peak(), full.work());
    prepared.release(&mut full);
    assert_eq!(full.ledger(), 0);
    let mut refusals = 0;
    let steps = 61;
    for i in 0..steps {
        for mut m in [
            Meter::with_caps(&budget, u64::MAX, peak * i / steps),
            Meter::with_caps(&budget, work * i / steps, u64::MAX),
        ] {
            match prepare(g, &mut m) {
                Ok(p) => p.release(&mut m),
                Err(Error::LimitExceeded(Limit::LedgerBytes | Limit::WorkUnits)) => refusals += 1,
                Err(other) => panic!("unexpected error {other:?}"),
            }
            assert_eq!(m.ledger(), 0, "a refused prepare kept ledger bytes");
        }
    }
    assert!(
        refusals > steps,
        "the sweep refused too rarely ({refusals})"
    );
}

#[test]
fn refusals_leave_the_ledger_balanced() {
    check_release_on_error(&generate::structured(7, 5, 0));
    check_release_on_error(&generate::structured(11, 4, 2));
    check_release_on_error(&generate::break_tail_ladder(20));
    check_release_on_error(&generate::irreducible_ladder(20));
    check_release_on_error(&generate::nested_irreducible(IRREDUCIBLE_DEPTH as u32));
}

#[test]
fn preparation_is_deterministic() {
    for seed in 0..50 {
        let g = generate::structured(seed, 4, (seed % 3) as u32);
        let run = || {
            let mut m = meter();
            prepare(&g, &mut m).map(|p| (describe(&p.cfg), format!("{:?}", p.loops)))
        };
        assert_eq!(run(), run());
    }
}

/// Work units of `prepare` on `g`.
fn work(g: &TestCfg) -> u64 {
    let mut m = meter();
    prepare(g, &mut m).unwrap().release(&mut m);
    m.work()
}

#[test]
fn work_and_output_grow_linearly() {
    type Family = fn(u32) -> TestCfg;
    let families: [(&str, Family, u32); 3] = [
        ("break tails", generate::break_tail_ladder, 4_000),
        ("irreducible", generate::irreducible_ladder, 4_000),
        ("chain", |k| TestCfg::from_successors(&chain(k)), 20_000),
    ];
    for (name, family, k) in families {
        let (small, large) = (family(k), family(2 * k));
        let (ws, wl) = (work(&small), work(&large));
        assert!(
            wl <= 2 * ws + ws / 5,
            "{name}: work {ws} -> {wl} when the graph doubles"
        );
        let mut m = meter();
        let p = prepare(&large, &mut m).unwrap();
        let (n, e) = (large.blocks.len() as u64, large.edge_count() as u64);
        assert!((p.cfg.len() as u64) <= output_blocks_cap(n, e));
        p.release(&mut m);
    }
}

fn chain(k: u32) -> Vec<Vec<u32>> {
    (0..k)
        .map(|i| if i + 1 < k { vec![i + 1] } else { vec![] })
        .collect()
}

#[test]
fn deep_graphs_do_not_recurse() {
    // Runs on the default test thread stack: every pass iterates with explicit stacks.
    let g = TestCfg::from_successors(&chain(200_000));
    let mut m = meter();
    prepare(&g, &mut m).unwrap().release(&mut m);
    // A deep loop nest: loop i's header branches into loop i+1 or exits to loop i-1's latch.
    let depth = 2_000u32;
    let mut succ = Vec::new();
    for i in 0..depth {
        let latch = 2 * i + 1;
        let inner = if i + 1 < depth { 2 * (i + 1) } else { latch };
        succ.push(vec![inner, latch]);
        succ.push(if i == 0 {
            vec![0, 2 * depth]
        } else {
            vec![2 * i, 2 * i - 1]
        });
    }
    succ.push(vec![]);
    let g = TestCfg::from_successors(&succ);
    let mut m = meter();
    let p = prepare(&g, &mut m).unwrap();
    assert_eq!(p.loops.len(), depth as usize);
    p.release(&mut m);
}

#[test]
fn prepared_structures_are_send_and_sync() {
    fn check<T: Send + Sync>() {}
    check::<Prepared>();
    check::<StructCfg>();
    check::<Loops>();
}
