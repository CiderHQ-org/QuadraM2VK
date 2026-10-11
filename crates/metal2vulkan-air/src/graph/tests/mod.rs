//! Tests of the graph module against independent references (see `reference.rs`).

use std::cell::Cell;
use std::collections::BTreeSet;

use super::*;
use crate::budget::{Budget, Meter, IRREDUCIBLE_DEPTH};
use crate::error::{BudgetExceeded, Limit};

mod generate;
mod reference;

use generate::Rng;
use reference::RefLoop;

/// A plain adjacency-list graph.
#[derive(Clone, Debug)]
pub struct VecGraph {
    pub succ: Vec<Vec<u32>>,
    pub entry: u32,
}

impl Graph for VecGraph {
    type Node = u32;

    fn node_count(&self) -> usize {
        self.succ.len()
    }

    fn entry(&self) -> u32 {
        self.entry
    }

    fn successors(&self, node: u32) -> impl Iterator<Item = u32> + '_ {
        self.succ[node as usize].iter().copied()
    }
}

fn meter() -> Meter {
    Meter::new(&Budget::default())
}

struct Analyses {
    cfg: Cfg<u32>,
    dom: DomTree<u32>,
    pdom: PostDomTree<u32>,
    /// `None` when the irreducible nesting exceeds `Limit::IrreducibleDepth`.
    loops: Option<LoopForest<u32>>,
    sccs: Sccs<u32>,
    work: u64,
}

impl Analyses {
    fn loops(&self) -> &LoopForest<u32> {
        self.loops
            .as_ref()
            .expect("irreducible nesting within the cap")
    }
}

fn analyze(g: &VecGraph) -> Analyses {
    let mut m = meter();
    let cfg = Cfg::build(g, &mut m).unwrap();
    assert_eq!(
        m.ledger(),
        cfg.ledger_bytes() as u64,
        "Cfg scratch not released"
    );
    let dom = DomTree::compute(&cfg, &mut m).unwrap();
    let pdom = PostDomTree::compute(&cfg, &mut m).unwrap();
    let sccs = Sccs::compute(&cfg, &mut m).unwrap();
    let kept = cfg.ledger_bytes() + dom.ledger_bytes() + pdom.ledger_bytes() + sccs.ledger_bytes();
    assert_eq!(m.ledger(), kept as u64, "scratch buffers not released");
    // The loop forest may refuse a too deep irreducible nest; measure it on its own meter.
    let mut lm = meter();
    let loops = match LoopForest::compute(&cfg, &mut lm) {
        Ok(loops) => {
            assert_eq!(lm.ledger(), loops.ledger_bytes() as u64);
            Some(loops)
        }
        Err(Error::LimitExceeded(Limit::IrreducibleDepth)) => None,
        Err(other) => panic!("loop forest failed: {other:?}"),
    };
    Analyses {
        cfg,
        dom,
        pdom,
        loops,
        sccs,
        work: m.work() + lm.work(),
    }
}

/// The forest as reference loops (blocks and entries sorted).
fn forest_as_reference(loops: &LoopForest<u32>) -> BTreeSet<RefLoop> {
    loops
        .loops()
        .map(|l| {
            let mut blocks: Vec<u32> = loops.blocks(l).collect();
            blocks.sort_unstable();
            let mut entries: Vec<u32> = loops.entries(l).collect();
            entries.sort_unstable();
            RefLoop {
                header: loops.header(l),
                blocks,
                entries,
                parent: loops.parent(l).map(|p| loops.header(p)),
            }
        })
        .collect()
}

/// Node -> component number, from each component's members.
fn components(s: &Sccs<u32>, n: usize) -> Vec<u32> {
    let mut comp = vec![u32::MAX; n];
    for c in 0..s.len() as u32 {
        for m in s.members(c) {
            comp[m as usize] = c;
        }
    }
    comp
}

/// Checks that need no reference: the forest's own consistency. O(n + e + Σ loop sizes).
fn check_forest_shape(g: &VecGraph, a: &Analyses) {
    let loops = a.loops();
    let n = g.succ.len() as u32;
    let mut child_count = vec![0usize; loops.len()];
    let mut top_count = 0usize;
    for l in loops.loops() {
        let h = loops.header(l);
        assert!(loops.contains(l, h));
        assert_eq!(loops.loop_of_header(h), Some(l));
        assert_eq!(
            loops.entries(l).next(),
            Some(h),
            "header is the first entry"
        );
        assert_eq!(loops.is_reducible(l), loops.entries(l).len() == 1);
        match loops.parent(l) {
            Some(p) => {
                assert!(p < l, "parents precede children");
                assert!(loops.loop_contains(p, l));
                child_count[p.index()] += 1;
            }
            None => {
                top_count += 1;
            }
        }
        let hp = a.cfg.preorder_number(h).unwrap();
        for b in loops.blocks(l) {
            assert!(
                a.cfg.preorder_number(b).unwrap() >= hp,
                "header is DFS-first"
            );
            assert!(loops.contains(l, b));
        }
        for e in loops.entries(l) {
            assert!(loops.contains(l, e));
        }
        if loops.is_reducible(l) {
            for b in loops.blocks(l) {
                assert!(a.dom.dominates(h, b), "reducible header dominates its loop");
            }
        }
        let children: Vec<LoopId> = loops.children(l).collect();
        for &c in &children {
            assert_eq!(loops.parent(c), Some(l));
        }
        for pair in children.windows(2) {
            assert!(
                a.cfg.preorder_number(loops.header(pair[0]))
                    < a.cfg.preorder_number(loops.header(pair[1]))
            );
        }
    }
    for l in loops.loops() {
        assert_eq!(loops.children(l).count(), child_count[l.index()]);
    }
    assert!(loops.top_level().all(|l| loops.parent(l).is_none()));
    assert_eq!(loops.top_level().count(), top_count);
    // Parents come after all their descendants in `innermost_first`.
    let mut position = vec![0usize; loops.len()];
    for (i, l) in loops.innermost_first().enumerate() {
        position[l.index()] = i;
    }
    for l in loops.loops() {
        if let Some(p) = loops.parent(l) {
            assert!(position[l.index()] < position[p.index()]);
        }
    }
    // `blocks(l)` is exactly the set of nodes `contains(l, _)`: count membership by walking up
    // from each node's innermost loop.
    let mut members = vec![0usize; loops.len()];
    for v in 0..n {
        let mut at = loops.innermost(v);
        if let Some(l) = at {
            assert!(loops.children(l).all(|c| !loops.contains(c, v)));
        }
        while let Some(l) = at {
            members[l.index()] += 1;
            at = loops.parent(l);
        }
    }
    for l in loops.loops() {
        assert_eq!(loops.blocks(l).len(), members[l.index()]);
    }
    assert_eq!(
        loops.is_reducible_graph(),
        loops.loops().all(|l| loops.is_reducible(l))
    );
}

/// Full brute-force check (small graphs).
fn check_small(g: &VecGraph) {
    let a = analyze(g);
    let n = g.succ.len();
    let pred = reference::preds(g);
    let dfs = reference::dfs(g);

    // Cfg.
    assert_eq!(a.cfg.node_count(), n);
    assert_eq!(
        a.cfg.edge_count(),
        g.succ.iter().map(Vec::len).sum::<usize>()
    );
    for v in 0..n as u32 {
        assert!(a.cfg.successors(v).eq(g.succ[v as usize].iter().copied()));
        assert!(a.cfg.predecessors(v).eq(pred[v as usize].iter().copied()));
        assert_eq!(a.cfg.preorder_number(v), dfs.pre[v as usize]);
        assert_eq!(a.cfg.is_reachable(v), dfs.pre[v as usize].is_some());
        assert_eq!(a.cfg.dfs_parent(v), dfs.parent[v as usize]);
        for w in 0..n as u32 {
            assert_eq!(a.cfg.is_dfs_ancestor(v, w), dfs.is_ancestor(v, w));
        }
    }
    assert!(a.cfg.preorder().eq(dfs.order.iter().copied()));
    assert!(a.cfg.rpo().eq(dfs.rpo.iter().copied()));
    for (i, &v) in dfs.rpo.iter().enumerate() {
        assert_eq!(a.cfg.rpo_number(v), Some(i as u32));
    }

    // Dominators.
    let dom = reference::brute_dominance(g);
    for b in 0..n {
        assert_eq!(
            a.dom.idom(b as u32),
            reference::immediate(&dom, b),
            "idom({b})"
        );
        assert_eq!(a.dom.is_reachable(b as u32), dfs.pre[b].is_some());
        for (x, row) in dom.iter().enumerate() {
            assert_eq!(a.dom.dominates(x as u32, b as u32), row[b], "dom({x},{b})");
        }
    }
    let mut seen = 0;
    for v in a.dom.preorder() {
        seen += 1;
        if let Some(p) = a.dom.idom(v) {
            assert!(a.dom.children(p).any(|c| c == v));
        }
    }
    assert_eq!(seen, dfs.order.len());

    // Post-dominators.
    let (in_tree, pdom) = reference::brute_post_dominance(g);
    for b in 0..n {
        assert_eq!(a.pdom.is_reachable(b as u32), in_tree[b], "pdom tree({b})");
        assert_eq!(
            a.pdom.ipdom(b as u32),
            reference::immediate(&pdom, b),
            "ipdom({b})"
        );
        for (x, row) in pdom.iter().enumerate() {
            assert_eq!(a.pdom.post_dominates(x as u32, b as u32), row[b]);
        }
    }

    // SCCs, by mutual reachability over the whole graph.
    let all = vec![true; n];
    let reach: Vec<Vec<bool>> = (0..n as u32)
        .map(|v| reference::reach(&g.succ, v, None, &all))
        .collect();
    let comp = components(&a.sccs, n);
    for (u, from_u) in reach.iter().enumerate() {
        for (v, from_v) in reach.iter().enumerate() {
            let same = comp[u] == comp[v];
            assert_eq!(same, from_u[v] && from_v[u], "scc({u},{v})");
        }
        for &w in &g.succ[u] {
            assert!(comp[u] >= comp[w as usize]);
        }
        let c = comp[u];
        let cyclic = a.sccs.members(c).len() > 1 || g.succ[u].contains(&(u as u32));
        assert_eq!(a.sccs.is_cyclic(c), cyclic);
    }
    let members: usize = (0..a.sccs.len() as u32)
        .map(|c| a.sccs.members(c).len())
        .sum();
    assert_eq!(members, n);

    // Loops: the definition, natural loops, reducibility.
    check_forest_shape(g, &a);
    assert_eq!(
        forest_as_reference(a.loops()),
        reference::loop_forest(g, &dfs.pre)
    );
    let reducible = (0..n as u32).all(|u| {
        g.succ[u as usize]
            .iter()
            .all(|&v| !dfs.is_ancestor(v, u) || dom[v as usize][u as usize])
    });
    assert_eq!(a.loops().is_reducible_graph(), reducible);
    if reducible {
        for l in a.loops().loops() {
            let mut blocks: Vec<u32> = a.loops().blocks(l).collect();
            blocks.sort_unstable();
            assert_eq!(
                blocks,
                reference::natural_loop(g, &pred, &dom, a.loops().header(l))
            );
        }
    }
}

/// Check against the scalable references (large graphs).
fn check_large(g: &VecGraph, loops_reference: bool) {
    let a = analyze(g);
    let n = g.succ.len();
    let pred = reference::preds(g);
    let dfs = reference::dfs(g);

    assert!(a.cfg.preorder().eq(dfs.order.iter().copied()));
    assert!(a.cfg.rpo().eq(dfs.rpo.iter().copied()));
    for v in 0..n as u32 {
        assert_eq!(a.cfg.dfs_parent(v), dfs.parent[v as usize]);
    }

    let idom = reference::chk_idom(&g.succ, &pred, g.entry);
    for v in 0..n as u32 {
        assert_eq!(a.dom.idom(v), idom[v as usize], "idom({v})");
        if let Some(d) = idom[v as usize] {
            assert!(a.dom.strictly_dominates(d, v));
        }
    }

    // Post-dominators: CHK on the reverse graph with a virtual exit `n`.
    let mut rsucc = vec![Vec::new(); n + 1];
    for v in 0..n {
        if dfs.pre[v].is_none() {
            continue;
        }
        if g.succ[v].is_empty() {
            rsucc[n].push(v as u32);
        }
        for &p in &pred[v] {
            if dfs.pre[p as usize].is_some() {
                rsucc[v].push(p);
            }
        }
    }
    let rgraph = VecGraph {
        succ: rsucc,
        entry: n as u32,
    };
    let rpred = reference::preds(&rgraph);
    let ripdom = reference::chk_idom(&rgraph.succ, &rpred, n as u32);
    let rdfs = reference::dfs(&rgraph);
    for v in 0..n as u32 {
        assert_eq!(a.pdom.is_reachable(v), rdfs.pre[v as usize].is_some());
        let expect = ripdom[v as usize].filter(|&p| p != n as u32);
        assert_eq!(a.pdom.ipdom(v), expect, "ipdom({v})");
    }

    // SCCs against Kosaraju.
    let all = vec![true; n];
    let comps = reference::kosaraju(&g.succ, &pred, &all);
    assert_eq!(comps.len(), a.sccs.len());
    let comp = components(&a.sccs, n);
    for (members, cyclic) in comps {
        let c = comp[members[0] as usize];
        let mut ours: Vec<u32> = a.sccs.members(c).collect();
        ours.sort_unstable();
        assert_eq!(ours, members);
        assert_eq!(a.sccs.is_cyclic(c), cyclic);
    }
    for u in 0..n {
        for &w in &g.succ[u] {
            assert!(comp[u] >= comp[w as usize]);
        }
    }

    let expected = loops_reference.then(|| reference::loop_forest(g, &dfs.pre));
    match &a.loops {
        Some(loops) => {
            check_forest_shape(g, &a);
            if let Some(expected) = expected {
                assert!(reference::irreducible_depth(&expected) <= IRREDUCIBLE_DEPTH);
                assert_eq!(forest_as_reference(loops), expected);
            }
        }
        None => {
            if let Some(expected) = expected {
                assert!(reference::irreducible_depth(&expected) > IRREDUCIBLE_DEPTH);
            }
        }
    }
}

#[test]
fn exhaustive_all_digraphs_up_to_four_nodes() {
    let mut count = 0u64;
    for n in 1..=4 {
        generate::all_digraphs(n, |g| {
            check_small(g);
            count += 1;
        });
    }
    assert_eq!(count, 2 + 16 + 512 + 65536);
}

fn canonical_counts(max: u32) -> Vec<u64> {
    (1..=max)
        .map(|n| {
            let mut count = 0u64;
            generate::canonical_out_degree_2(n, |g| {
                check_small(g);
                count += 1;
            });
            count
        })
        .collect()
}

#[test]
fn exhaustive_out_degree_two_up_to_five_nodes() {
    assert_eq!(canonical_counts(5), [2, 8, 91, 1419, 28752]);
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "710k graphs: runs in release (G-rel: cargo test --release -- graph)"
)]
fn exhaustive_out_degree_two_six_nodes() {
    let mut count = 0u64;
    generate::canonical_out_degree_2(6, |g| {
        check_small(g);
        count += 1;
    });
    assert_eq!(count, 710_556);
}

#[test]
fn random_small_graphs_match_brute_force() {
    let mut rng = Rng::new(0x5eed_0001);
    for _ in 0..3000 {
        let n = 1 + rng.below(12) as u32;
        let g = if rng.below(2) == 0 {
            generate::uniform(n, &mut rng)
        } else {
            let window = 1 + rng.below(4) as u32;
            generate::local(n, window, &mut rng)
        };
        check_small(&g);
    }
}

#[test]
fn random_medium_graphs_match_references() {
    let mut rng = Rng::new(0x5eed_0002);
    for round in 0..40 {
        let n = 50 + rng.below(1500) as u32;
        let g = if round % 2 == 0 {
            generate::uniform(n, &mut rng)
        } else {
            let window = 2 + rng.below(30) as u32;
            generate::local(n, window, &mut rng)
        };
        check_large(&g, true);
    }
}

#[test]
fn random_large_graphs_match_references() {
    let mut rng = Rng::new(0x5eed_0003);
    for &n in &[20_000u32, 100_000] {
        check_large(&generate::uniform(n, &mut rng), false);
        check_large(&generate::local(n, 8, &mut rng), false);
        check_large(&generate::local(n, 200, &mut rng), false);
    }
}

#[test]
fn structured_families_match_references() {
    for g in [
        generate::chain(300),
        generate::nested_loops(300),
        generate::diamonds(100),
        generate::irreducible_ladder(100),
        generate::nested_irreducible(40),
        generate::switch_fan(300),
        generate::sequential_loops(100),
    ] {
        check_large(&g, true);
    }
    for g in [
        generate::nested_loops(12),
        generate::irreducible_ladder(3),
        generate::nested_irreducible(4),
        generate::diamonds(3),
    ] {
        check_small(&g);
    }
}

#[test]
fn parallel_edges_and_self_loops() {
    // CondBr(c, t, t), a switch with a repeated target, and a self-loop.
    let g = VecGraph {
        succ: vec![vec![1, 1], vec![2, 3, 2], vec![2, 3], vec![]],
        entry: 0,
    };
    check_small(&g);
    let a = analyze(&g);
    assert_eq!(a.cfg.edge_count(), 7);
    assert!(a.cfg.predecessors(2).eq([1, 1, 2]));
    assert_eq!(a.loops().len(), 1);
    let l = a.loops().loops().next().unwrap();
    assert_eq!(a.loops().header(l), 2);
    assert!(a.loops().blocks(l).eq([2]));
    let comp = components(&a.sccs, g.succ.len());
    assert!(a.sccs.is_cyclic(comp[2]));
    assert!(!a.sccs.is_cyclic(comp[1]));
}

#[test]
fn classic_irreducible_triangle() {
    // 0 -> {1, 2}, 1 <-> 2, 2 -> 3: one loop entered at both 1 and 2.
    let g = VecGraph {
        succ: vec![vec![1, 2], vec![2], vec![1, 3], vec![]],
        entry: 0,
    };
    check_small(&g);
    let a = analyze(&g);
    assert!(!a.loops().is_reducible_graph());
    assert_eq!(a.loops().irreducible_count(), 1);
    let l = a.loops().loops().next().unwrap();
    assert_eq!(a.loops().header(l), 1, "the DFS-first block heads the loop");
    assert!(a.loops().entries(l).eq([1, 2]));
    assert_eq!(a.dom.idom(1), Some(0));
    assert_eq!(a.dom.idom(2), Some(0));
    assert_eq!(a.pdom.ipdom(1), Some(2));
    assert_eq!(a.pdom.ipdom(0), Some(2));
}

#[test]
fn nested_loops_are_numbered_outer_first() {
    // 0 -> 1 -> 2 -> 3 -> 4, back edges 3 -> 2 (inner) and 4 -> 1 (outer); 4 exits to 5.
    let g = VecGraph {
        succ: vec![vec![1], vec![2], vec![3], vec![2, 4], vec![1, 5], vec![]],
        entry: 0,
    };
    check_small(&g);
    let a = analyze(&g);
    let ids: Vec<LoopId> = a.loops().loops().collect();
    assert_eq!(ids.len(), 2);
    let (outer, inner) = (ids[0], ids[1]);
    assert_eq!(a.loops().header(outer), 1);
    assert_eq!(a.loops().header(inner), 2);
    assert_eq!(a.loops().parent(inner), Some(outer));
    assert!(a.loops().blocks(outer).eq([1, 4, 2, 3]));
    assert!(a.loops().innermost_first().eq([inner, outer]));
    assert_eq!(a.loops().innermost(3), Some(inner));
    assert_eq!(a.loops().innermost(5), None);
}

#[test]
fn unreachable_nodes_are_outside_every_analysis() {
    // 2 is unreachable and has an edge into the reachable part and a self-loop.
    let g = VecGraph {
        succ: vec![vec![1], vec![], vec![2, 1]],
        entry: 0,
    };
    check_small(&g);
    let a = analyze(&g);
    assert!(!a.cfg.is_reachable(2));
    assert_eq!(a.cfg.rpo_number(2), None);
    assert!(!a.dom.is_reachable(2));
    assert!(!a.dom.dominates(2, 1));
    assert!(!a.pdom.is_reachable(2));
    assert!(a.loops().is_empty());
    assert!(
        a.sccs.is_cyclic(components(&a.sccs, g.succ.len())[2]),
        "SCCs cover every node"
    );
    assert_eq!(a.dom.idom(1), Some(0));
}

#[test]
fn infinite_loops_have_no_post_dominators() {
    // 0 -> {1, 3}; 1 <-> 2 never exits; 3 exits.
    let g = VecGraph {
        succ: vec![vec![1, 3], vec![2], vec![1], vec![]],
        entry: 0,
    };
    check_small(&g);
    let a = analyze(&g);
    assert!(a.pdom.is_reachable(0));
    assert!(!a.pdom.is_reachable(1));
    assert!(!a.pdom.is_reachable(2));
    assert_eq!(a.pdom.ipdom(0), Some(3));
    assert_eq!(a.pdom.ipdom(3), None, "the virtual exit is not a node");
    assert!(a.pdom.post_dominates(3, 0));
}

#[test]
fn results_are_deterministic() {
    let mut rng = Rng::new(0x5eed_0004);
    let g = generate::local(5000, 40, &mut rng);
    let (a, b) = (analyze(&g), analyze(&g));
    assert_eq!(format!("{:?}", a.cfg), format!("{:?}", b.cfg));
    assert_eq!(format!("{:?}", a.dom), format!("{:?}", b.dom));
    assert_eq!(format!("{:?}", a.pdom), format!("{:?}", b.pdom));
    assert_eq!(format!("{:?}", a.loops), format!("{:?}", b.loops));
    assert_eq!(format!("{:?}", a.sccs), format!("{:?}", b.sccs));
    assert_eq!(a.work, b.work);
}

/// All analyses on 10⁶-node graphs, on a 256 KiB thread stack: nothing recurses on the input.
#[test]
fn million_node_graphs_do_not_recurse() {
    std::thread::Builder::new()
        .stack_size(256 * 1024)
        .spawn(|| {
            let n = 1_000_000;
            let a = analyze(&generate::chain(n));
            assert_eq!(a.dom.idom(n - 1), Some(n - 2));
            assert!(a.loops().is_empty());
            let a = analyze(&generate::nested_loops(n));
            assert_eq!(a.loops().len() as u32, n / 2);
            let nesting =
                std::iter::successors(a.loops().innermost(n / 2), |&l| a.loops().parent(l));
            assert_eq!(nesting.count() as u32, n / 2);
            assert!(!a.pdom.is_reachable(0), "the outermost loop never exits");
            let a = analyze(&generate::irreducible_ladder(n / 3));
            assert_eq!(a.loops().irreducible_count() as u32, n / 3);
            let a = analyze(&generate::nested_irreducible(n / 2 - 1));
            assert!(
                a.loops.is_none(),
                "irreducible nesting above the cap is refused"
            );
        })
        .unwrap()
        .join()
        .unwrap();
}

/// Work units grow linearly: doubling each family's size at most doubles its work (plus 5%).
#[test]
fn work_units_grow_linearly() {
    type Family = fn(u32) -> VecGraph;
    let families: [(&str, Family, u32); 7] = [
        ("chain", generate::chain, 100_000),
        ("nested_loops", generate::nested_loops, 100_000),
        ("diamonds", generate::diamonds, 33_333),
        ("irreducible_ladder", generate::irreducible_ladder, 33_333),
        ("switch_fan", generate::switch_fan, 100_000),
        ("sequential_loops", generate::sequential_loops, 33_333),
        (
            "local",
            |n| generate::local(n, 16, &mut Rng::new(7)),
            100_000,
        ),
    ];
    for (name, family, n) in families {
        let small = analyze(&family(n)).work;
        let large = analyze(&family(2 * n)).work;
        assert!(
            large <= small * 21 / 10 + 64,
            "{name}: work {small} at n={n} grew to {large} at 2n"
        );
        let g = family(n);
        let size = (g.succ.len() + g.succ.iter().map(Vec::len).sum::<usize>()) as u64;
        assert!(
            small <= 40 * size,
            "{name}: {small} work units for n + e = {size}"
        );
    }
}

#[test]
fn irreducible_nesting_is_capped() {
    let at_cap = analyze(&generate::nested_irreducible(IRREDUCIBLE_DEPTH as u32));
    assert_eq!(at_cap.loops().irreducible_count() as u64, IRREDUCIBLE_DEPTH);
    assert!(
        analyze(&generate::nested_irreducible(IRREDUCIBLE_DEPTH as u32 + 1))
            .loops
            .is_none()
    );
    // Loops complete innermost first, so a deep nest is refused after O(cap) levels of work.
    let g = generate::nested_irreducible(100_000);
    let mut m = meter();
    let cfg = Cfg::build(&g, &mut m).unwrap();
    let before = m.work();
    assert_eq!(
        LoopForest::compute(&cfg, &mut m).unwrap_err(),
        Error::LimitExceeded(Limit::IrreducibleDepth)
    );
    let edges = g.succ.iter().map(Vec::len).sum::<usize>() as u64;
    assert!(m.work() - before < 4 * (g.succ.len() as u64 + edges));
}

#[test]
fn work_cap_is_a_deterministic_limit() {
    let g = generate::nested_loops(10_000);
    let full = analyze(&g).work;
    let budget = Budget::default();
    let mut m = Meter::with_caps(&budget, 1000, u64::MAX);
    assert_eq!(
        Cfg::build(&g, &mut m).unwrap_err(),
        Error::LimitExceeded(Limit::WorkUnits)
    );
    let mut m = meter();
    let cfg = Cfg::build(&g, &mut m).unwrap();
    let after_cfg = m.work();
    let mut m = Meter::with_caps(&budget, after_cfg + 10, u64::MAX);
    let cfg2 = Cfg::build(&g, &mut m).unwrap();
    assert_eq!(m.work(), after_cfg);
    assert_eq!(
        DomTree::compute(&cfg2, &mut m).unwrap_err(),
        Error::LimitExceeded(Limit::WorkUnits)
    );
    for cap in [full / 4, full / 2, full - 1] {
        let mut m = Meter::with_caps(&budget, cap, u64::MAX);
        let result = (|| -> Result<(), Error> {
            let cfg = Cfg::build(&g, &mut m)?;
            DomTree::compute(&cfg, &mut m)?;
            PostDomTree::compute(&cfg, &mut m)?;
            LoopForest::compute(&cfg, &mut m)?;
            Sccs::compute(&cfg, &mut m)?;
            Ok(())
        })();
        assert_eq!(result, Err(Error::LimitExceeded(Limit::WorkUnits)));
    }
    drop(cfg);
}

#[test]
fn ledger_cap_refuses_before_allocating() {
    let g = generate::chain(100_000);
    let budget = Budget::default();
    let mut m = Meter::with_caps(&budget, u64::MAX, 64 * 1024);
    assert_eq!(
        Cfg::build(&g, &mut m).unwrap_err(),
        Error::LimitExceeded(Limit::LedgerBytes)
    );
    assert_eq!(
        Sccs::compute(&g, &mut m).unwrap_err(),
        Error::LimitExceeded(Limit::LedgerBytes)
    );
    // Exactly the Cfg's peak fits; anything more is refused.
    let mut m = meter();
    Cfg::build(&g, &mut m).unwrap();
    let mut m = Meter::with_caps(&budget, u64::MAX, m.ledger_peak());
    let cfg = Cfg::build(&g, &mut m).unwrap();
    assert_eq!(
        DomTree::compute(&cfg, &mut m).unwrap_err(),
        Error::LimitExceeded(Limit::LedgerBytes)
    );
    assert_eq!(
        LoopForest::compute(&cfg, &mut m).unwrap_err(),
        Error::LimitExceeded(Limit::LedgerBytes)
    );
}

#[test]
fn deadline_is_observed() {
    let budget = Budget {
        deadline: Some(std::time::Instant::now()),
        cancel: None,
    };
    let mut m = Meter::new(&budget);
    assert_eq!(
        Cfg::build(&generate::chain(10), &mut m).unwrap_err(),
        Error::Budget(BudgetExceeded::Time)
    );
}

fn invariant_detail(err: Error) -> String {
    match err {
        Error::Internal(e) => {
            assert_eq!(e.kind, InternalKind::Invariant);
            e.detail
        }
        other => panic!("expected an invariant error, got {other:?}"),
    }
}

#[test]
fn broken_graph_contracts_are_typed_errors() {
    let mut m = meter();
    let empty = VecGraph {
        succ: vec![],
        entry: 0,
    };
    assert!(invariant_detail(Cfg::build(&empty, &mut m).unwrap_err()).contains("no nodes"));
    let bad_entry = VecGraph {
        succ: vec![vec![]],
        entry: 1,
    };
    assert!(invariant_detail(Cfg::build(&bad_entry, &mut m).unwrap_err()).contains("entry"));
    let bad_successor = VecGraph {
        succ: vec![vec![5]],
        entry: 0,
    };
    assert!(invariant_detail(Cfg::build(&bad_successor, &mut m).unwrap_err()).contains("5"));
    assert!(invariant_detail(Sccs::compute(&bad_successor, &mut m).unwrap_err()).contains("5"));

    /// A graph whose successor lists grow on every call.
    struct Shifting(Cell<u32>);
    impl Graph for Shifting {
        type Node = u32;
        fn node_count(&self) -> usize {
            2
        }
        fn entry(&self) -> u32 {
            0
        }
        fn successors(&self, node: u32) -> impl Iterator<Item = u32> + '_ {
            let calls = self.0.get();
            self.0.set(calls + 1);
            std::iter::repeat_n(1 - node, calls as usize)
        }
    }
    assert!(
        invariant_detail(Cfg::build(&Shifting(Cell::new(0)), &mut m).unwrap_err())
            .contains("changed")
    );
}

#[test]
fn analysis_structures_are_send_and_sync() {
    fn check<T: Send + Sync>() {}
    check::<Cfg<u32>>();
    check::<DomTree<u32>>();
    check::<PostDomTree<u32>>();
    check::<LoopForest<u32>>();
    check::<Sccs<u32>>();
    check::<LoopId>();
}

/// Run `analysis` (which returns the bytes its result keeps charged) under a sweep of ledger and
/// work caps that refuse it at many points, with `base` bytes already charged. Every refusal must
/// leave the ledger at `base`; every success at `base` plus what the result keeps.
fn check_release_on_error(base: usize, analysis: &dyn Fn(&mut Meter) -> Result<usize, Error>) {
    let budget = Budget::default();
    let fresh = |work_cap: u64, byte_cap: u64| {
        let mut m = Meter::with_caps(&budget, work_cap, byte_cap);
        m.charge(base).unwrap();
        m
    };
    let mut full = fresh(u64::MAX, u64::MAX);
    let kept = analysis(&mut full).unwrap();
    assert_eq!(full.ledger(), (base + kept) as u64);
    let (peak, work) = (full.ledger_peak(), full.work());
    let mut refusals = 0;
    let steps = 97u64;
    for i in 0..steps {
        let byte_cap = base as u64 + (peak - base as u64) * i / steps;
        let work_cap = work * i / steps;
        for mut m in [fresh(u64::MAX, byte_cap), fresh(work_cap, u64::MAX)] {
            match analysis(&mut m) {
                Ok(kept) => assert_eq!(m.ledger(), (base + kept) as u64),
                Err(Error::LimitExceeded(Limit::LedgerBytes | Limit::WorkUnits)) => {
                    refusals += 1;
                    assert_eq!(
                        m.ledger(),
                        base as u64,
                        "a refused analysis kept ledger bytes"
                    );
                }
                Err(other) => panic!("unexpected error {other:?}"),
            }
        }
    }
    assert!(
        refusals > steps,
        "the sweep refused too rarely ({refusals})"
    );
}

#[test]
fn refused_analyses_release_their_ledger() {
    for g in [
        generate::nested_loops(300),
        generate::irreducible_ladder(30),
        generate::nested_irreducible(IRREDUCIBLE_DEPTH as u32),
        generate::local(400, 6, &mut Rng::new(7)),
    ] {
        check_release_on_error(0, &|m| Cfg::build(&g, m).map(|c| c.ledger_bytes()));
        check_release_on_error(0, &|m| Sccs::compute(&g, m).map(|s| s.ledger_bytes()));
        let cfg = Cfg::build(&g, &mut meter()).unwrap();
        let base = cfg.ledger_bytes();
        check_release_on_error(base, &|m| {
            DomTree::compute(&cfg, m).map(|d| d.ledger_bytes())
        });
        check_release_on_error(base, &|m| {
            PostDomTree::compute(&cfg, m).map(|d| d.ledger_bytes())
        });
        check_release_on_error(base, &|m| {
            LoopForest::compute(&cfg, m).map(|l| l.ledger_bytes())
        });
    }
    // A structural refusal (too deep an irreducible nest) releases everything too.
    let g = generate::nested_irreducible(IRREDUCIBLE_DEPTH as u32 + 1);
    let mut m = meter();
    let cfg = Cfg::build(&g, &mut m).unwrap();
    assert_eq!(
        LoopForest::compute(&cfg, &mut m).unwrap_err(),
        Error::LimitExceeded(Limit::IrreducibleDepth)
    );
    assert_eq!(m.ledger(), cfg.ledger_bytes() as u64);
}
