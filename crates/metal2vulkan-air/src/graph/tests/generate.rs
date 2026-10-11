//! Graph generators: exhaustive enumerations, seeded random families, and scaling ladders.

use super::VecGraph;

/// SplitMix64: a fixed-seed generator, so every random test is reproducible.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed)
    }

    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in `0..bound` (`bound > 0`).
    pub fn below(&mut self, bound: u64) -> u64 {
        self.next() % bound
    }
}

/// Every digraph on `n` nodes (no parallel edges), entry 0: `2^(n²)` graphs.
pub fn all_digraphs(n: u32, mut f: impl FnMut(&VecGraph)) {
    let bits = n * n;
    for mask in 0u64..(1u64 << bits) {
        let succ = (0..n)
            .map(|v| (0..n).filter(|&w| mask & (1 << (v * n + w)) != 0).collect())
            .collect();
        f(&VecGraph { succ, entry: 0 });
    }
}

/// Every graph on `n` nodes with out-degree ≤ 2 (successor sets, ascending) in which every node
/// is reachable from entry 0, up to relabeling: exactly the graphs whose DFS preorder (visiting
/// successors in ascending order) is `0, 1, …, n-1`. Every such graph is isomorphic to one of
/// these (relabel by DFS preorder), so dominance, loops, and SCCs are covered exhaustively.
pub fn canonical_out_degree_2(n: u32, mut f: impl FnMut(&VecGraph)) {
    let mut sets: Vec<Vec<u32>> = vec![Vec::new()];
    for a in 0..n {
        sets.push(vec![a]);
    }
    for a in 0..n {
        for b in a + 1..n {
            sets.push(vec![a, b]);
        }
    }
    let mut succ = vec![Vec::new(); n as usize];
    for s in &sets {
        succ[0] = s.clone();
        visit(n, &sets, &mut succ, 1, vec![(0, 0)], &mut f);
    }
}

/// Continue the ascending-order DFS: `k` nodes are numbered, `stack` holds (node, next
/// successor position). An unvisited successor must be exactly `k`, whose successor set is
/// chosen now.
fn visit(
    n: u32,
    sets: &[Vec<u32>],
    succ: &mut Vec<Vec<u32>>,
    k: u32,
    mut stack: Vec<(u32, usize)>,
    f: &mut impl FnMut(&VecGraph),
) {
    loop {
        let Some(&(v, i)) = stack.last() else {
            if k == n {
                f(&VecGraph {
                    succ: succ.clone(),
                    entry: 0,
                });
            }
            return;
        };
        let Some(&s) = succ[v as usize].get(i) else {
            stack.pop();
            continue;
        };
        stack.last_mut().unwrap().1 += 1;
        if s < k {
            continue;
        }
        if s > k {
            return;
        }
        for set in sets {
            succ[k as usize] = set.clone();
            let mut next = stack.clone();
            next.push((k, 0));
            visit(n, sets, succ, k + 1, next, f);
        }
        return;
    }
}

/// A random graph: out-degree 0–3 (skewed to 1–2), uniform targets, duplicates possible.
pub fn uniform(n: u32, rng: &mut Rng) -> VecGraph {
    let succ = (0..n)
        .map(|_| {
            let degree = match rng.below(20) {
                0 => 0,
                1..=9 => 1,
                10..=17 => 2,
                _ => 3,
            };
            (0..degree)
                .map(|_| rng.below(u64::from(n)) as u32)
                .collect()
        })
        .collect();
    VecGraph {
        succ,
        entry: rng.below(u64::from(n)) as u32,
    }
}

/// A random graph with local edges: mostly forward within `window`, some backward. Produces deep
/// dominator trees and nested (sometimes irreducible) loops.
pub fn local(n: u32, window: u32, rng: &mut Rng) -> VecGraph {
    let succ = (0..n)
        .map(|v| {
            let degree = if v + 1 == n {
                0
            } else {
                1 + rng.below(2) as u32
            };
            (0..degree)
                .map(|_| {
                    let back = rng.below(5) == 0;
                    let d = 1 + rng.below(u64::from(window)) as u32;
                    if back {
                        v.saturating_sub(d)
                    } else {
                        (v + d).min(n - 1)
                    }
                })
                .collect()
        })
        .collect();
    VecGraph { succ, entry: 0 }
}

/// `0 -> 1 -> … -> n-1`.
pub fn chain(n: u32) -> VecGraph {
    let succ = (0..n)
        .map(|v| if v + 1 < n { vec![v + 1] } else { vec![] })
        .collect();
    VecGraph { succ, entry: 0 }
}

/// A chain with back edges `n-1-i -> i` for `i < n/2`: `n/2` perfectly nested loops.
pub fn nested_loops(n: u32) -> VecGraph {
    let mut g = chain(n);
    for i in 0..n / 2 {
        let latch = n - 1 - i;
        if latch > i {
            g.succ[latch as usize].push(i);
        }
    }
    g
}

/// `k` if/else diamonds in sequence (`3k + 1` nodes).
pub fn diamonds(k: u32) -> VecGraph {
    let mut succ = Vec::new();
    for i in 0..k {
        let b = 3 * i;
        succ.push(vec![b + 1, b + 2]);
        succ.push(vec![b + 3]);
        succ.push(vec![b + 3]);
    }
    succ.push(vec![]);
    VecGraph { succ, entry: 0 }
}

/// `k` two-entry (irreducible) cycles in sequence (`3k + 1` nodes): `b -> {b+1, b+2}`,
/// `b+1 <-> b+2`, `b+2 -> b+3`.
pub fn irreducible_ladder(k: u32) -> VecGraph {
    let mut succ = Vec::new();
    for i in 0..k {
        let b = 3 * i;
        succ.push(vec![b + 1, b + 2]);
        succ.push(vec![b + 2]);
        succ.push(vec![b + 1, b + 3]);
    }
    succ.push(vec![]);
    VecGraph { succ, entry: 0 }
}

/// `k` nested two-entry cycles: level `i` is `a_i <-> b_i`, entered at both blocks (from the
/// entry for level 0, from `a_{i-1}` otherwise), and `b_i` returns to `b_{i-1}` (level 0 exits).
/// Every level is irreducible and the nesting depth is `k`.
pub fn nested_irreducible(k: u32) -> VecGraph {
    // Node 0 is the entry, level i uses nodes 1 + 2i (a) and 2 + 2i (b), exit is 2k + 1.
    let exit = 2 * k + 1;
    let mut succ = vec![vec![1, 2]];
    for i in 0..k {
        let a = 1 + 2 * i;
        let b = a + 1;
        let mut a_succ = vec![b];
        if i + 1 < k {
            a_succ.extend([a + 2, a + 3]);
        }
        succ.push(a_succ);
        succ.push(vec![a, if i == 0 { exit } else { b - 2 }]);
    }
    succ.push(vec![]);
    VecGraph { succ, entry: 0 }
}

/// A switch from the entry to `n - 2` cases that all reach one exit.
pub fn switch_fan(n: u32) -> VecGraph {
    let exit = n - 1;
    let mut succ = vec![(1..exit).collect::<Vec<u32>>()];
    for _ in 1..exit {
        succ.push(vec![exit]);
    }
    succ.push(vec![]);
    VecGraph { succ, entry: 0 }
}

/// `k` single-block loops in sequence, each with a two-block body.
pub fn sequential_loops(k: u32) -> VecGraph {
    let mut succ = Vec::new();
    for i in 0..k {
        let h = 3 * i;
        succ.push(vec![h + 1, h + 3]);
        succ.push(vec![h + 2]);
        succ.push(vec![h]);
    }
    succ.push(vec![]);
    VecGraph { succ, entry: 0 }
}
