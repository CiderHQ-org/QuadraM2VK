//! Independent reference implementations the graph algorithms are checked against.
//!
//! Brute-force definitions (reachability with a node removed) for small graphs; for large ones,
//! different algorithms (Cooper-Harvey-Kennedy iterative dominators, Kosaraju SCC, recursive SCC
//! decomposition for the loop forest) that share no code with the module under test.

use std::collections::BTreeSet;

use super::VecGraph;

pub fn preds(g: &VecGraph) -> Vec<Vec<u32>> {
    let mut p = vec![Vec::new(); g.succ.len()];
    for (v, s) in g.succ.iter().enumerate() {
        for &w in s {
            p[w as usize].push(v as u32);
        }
    }
    p
}

/// Nodes reachable from `start` without passing through `removed` (`start == removed` reaches
/// nothing).
pub fn reach(succ: &[Vec<u32>], start: u32, removed: Option<u32>, mask: &[bool]) -> Vec<bool> {
    let mut seen = vec![false; succ.len()];
    if Some(start) == removed || !mask[start as usize] {
        return seen;
    }
    let mut stack = vec![start];
    seen[start as usize] = true;
    while let Some(v) = stack.pop() {
        for &w in &succ[v as usize] {
            if Some(w) != removed && mask[w as usize] && !seen[w as usize] {
                seen[w as usize] = true;
                stack.push(w);
            }
        }
    }
    seen
}

/// A DFS from the entry visiting successors in order (explicit stack of slice iterators).
pub struct RefDfs {
    pub pre: Vec<Option<u32>>,
    pub order: Vec<u32>,
    pub parent: Vec<Option<u32>>,
    pub rpo: Vec<u32>,
}

pub fn dfs(g: &VecGraph) -> RefDfs {
    let n = g.succ.len();
    let mut pre = vec![None; n];
    let mut parent = vec![None; n];
    let mut order = Vec::new();
    let mut post = Vec::new();
    let mut stack: Vec<(u32, std::slice::Iter<'_, u32>)> = Vec::new();
    pre[g.entry as usize] = Some(0);
    order.push(g.entry);
    stack.push((g.entry, g.succ[g.entry as usize].iter()));
    while let Some((v, it)) = stack.last_mut() {
        let v = *v;
        match it.next() {
            Some(&w) => {
                if pre[w as usize].is_none() {
                    pre[w as usize] = Some(order.len() as u32);
                    order.push(w);
                    parent[w as usize] = Some(v);
                    stack.push((w, g.succ[w as usize].iter()));
                }
            }
            None => {
                stack.pop();
                post.push(v);
            }
        }
    }
    post.reverse();
    RefDfs {
        pre,
        order,
        parent,
        rpo: post,
    }
}

impl RefDfs {
    pub fn is_ancestor(&self, a: u32, mut b: u32) -> bool {
        if self.pre[a as usize].is_none() || self.pre[b as usize].is_none() {
            return false;
        }
        loop {
            if a == b {
                return true;
            }
            match self.parent[b as usize] {
                Some(p) => b = p,
                None => return false,
            }
        }
    }
}

/// `dom[a][b]`: `a` dominates `b` (both reachable), by definition.
pub fn brute_dominance(g: &VecGraph) -> Vec<Vec<bool>> {
    let n = g.succ.len();
    let all = vec![true; n];
    let reachable = reach(&g.succ, g.entry, None, &all);
    let mut dom = vec![vec![false; n]; n];
    for a in 0..n {
        if !reachable[a] {
            continue;
        }
        let without = reach(&g.succ, g.entry, Some(a as u32), &all);
        for b in 0..n {
            dom[a][b] = reachable[b] && (a == b || !without[b]);
        }
    }
    dom
}

/// The immediate element of a dominance relation: the strict dominator of `b` dominated by
/// every other strict dominator of `b`.
pub fn immediate(rel: &[Vec<bool>], b: usize) -> Option<u32> {
    let strict: Vec<usize> = (0..rel.len()).filter(|&a| a != b && rel[a][b]).collect();
    strict
        .iter()
        .copied()
        .find(|&a| strict.iter().all(|&o| rel[o][a]))
        .map(|a| a as u32)
}

/// `(in_tree, pdom)`: nodes reachable from the entry that reach an exit, and `pdom[a][b]`.
pub fn brute_post_dominance(g: &VecGraph) -> (Vec<bool>, Vec<Vec<bool>>) {
    let n = g.succ.len();
    let all = vec![true; n];
    let reachable = reach(&g.succ, g.entry, None, &all);
    let exits: Vec<usize> = (0..n)
        .filter(|&v| reachable[v] && g.succ[v].is_empty())
        .collect();
    let reaches_exit = |v: usize, removed: Option<u32>| {
        let r = reach(&g.succ, v as u32, removed, &reachable);
        exits.iter().any(|&x| r[x])
    };
    let in_tree: Vec<bool> = (0..n)
        .map(|v| reachable[v] && reaches_exit(v, None))
        .collect();
    let mut pdom = vec![vec![false; n]; n];
    for a in 0..n {
        for b in 0..n {
            pdom[a][b] = in_tree[a] && in_tree[b] && (a == b || !reaches_exit(b, Some(a as u32)));
        }
    }
    (in_tree, pdom)
}

/// Kosaraju SCCs of the subgraph induced by `mask`; each component with a flag "has a cycle".
pub fn kosaraju(succ: &[Vec<u32>], pred: &[Vec<u32>], mask: &[bool]) -> Vec<(Vec<u32>, bool)> {
    let n = succ.len();
    let mut seen = vec![false; n];
    let mut finish = Vec::new();
    for root in 0..n {
        if !mask[root] || seen[root] {
            continue;
        }
        seen[root] = true;
        let mut stack = vec![(root as u32, 0usize)];
        while let Some((v, i)) = stack.last_mut() {
            let v = *v;
            if let Some(&w) = succ[v as usize].get(*i) {
                *i += 1;
                if mask[w as usize] && !seen[w as usize] {
                    seen[w as usize] = true;
                    stack.push((w, 0));
                }
            } else {
                stack.pop();
                finish.push(v);
            }
        }
    }
    let mut comp = vec![usize::MAX; n];
    let mut out = Vec::new();
    for &root in finish.iter().rev() {
        if comp[root as usize] != usize::MAX {
            continue;
        }
        let id = out.len();
        let mut members = vec![root];
        comp[root as usize] = id;
        let mut i = 0;
        while i < members.len() {
            let v = members[i];
            i += 1;
            for &p in &pred[v as usize] {
                if mask[p as usize] && comp[p as usize] == usize::MAX {
                    comp[p as usize] = id;
                    members.push(p);
                }
            }
        }
        let cyclic = members.len() > 1 || succ[root as usize].contains(&root);
        members.sort_unstable();
        out.push((members, cyclic));
    }
    out
}

/// One loop of the reference forest.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct RefLoop {
    pub header: u32,
    pub blocks: Vec<u32>,
    pub entries: Vec<u32>,
    pub parent: Option<u32>,
}

/// The loop forest by definition: cyclic SCCs of the reachable graph; header = least DFS
/// preorder; entries = header + blocks with a reachable predecessor outside; children = cyclic
/// SCCs of the loop without its header. O((n + e)·depth).
pub fn loop_forest(g: &VecGraph, pre: &[Option<u32>]) -> BTreeSet<RefLoop> {
    let n = g.succ.len();
    let pred = preds(g);
    let mask: Vec<bool> = pre.iter().map(Option::is_some).collect();
    let mut out = BTreeSet::new();
    let mut work: Vec<(Vec<bool>, Option<u32>)> = vec![(mask, None)];
    while let Some((mask, parent)) = work.pop() {
        for (members, cyclic) in kosaraju(&g.succ, &pred, &mask) {
            if !cyclic {
                continue;
            }
            let header = *members.iter().min_by_key(|&&v| pre[v as usize]).unwrap();
            let mut inside = vec![false; n];
            for &v in &members {
                inside[v as usize] = true;
            }
            let mut entries: Vec<u32> = members
                .iter()
                .copied()
                .filter(|&v| {
                    v == header
                        || pred[v as usize]
                            .iter()
                            .any(|&p| pre[p as usize].is_some() && !inside[p as usize])
                })
                .collect();
            entries.sort_unstable();
            out.insert(RefLoop {
                header,
                blocks: members.clone(),
                entries,
                parent,
            });
            inside[header as usize] = false;
            work.push((inside, Some(header)));
        }
    }
    out
}

/// Natural loop of `h`: `h` plus every block reaching a latch (a predecessor of `h` that `h`
/// dominates) without passing through `h`.
pub fn natural_loop(g: &VecGraph, pred: &[Vec<u32>], dom: &[Vec<bool>], h: u32) -> Vec<u32> {
    let n = g.succ.len();
    let mut inside = vec![false; n];
    inside[h as usize] = true;
    let mut stack: Vec<u32> = pred[h as usize]
        .iter()
        .copied()
        .filter(|&u| dom[h as usize][u as usize])
        .collect();
    while let Some(v) = stack.pop() {
        if inside[v as usize] {
            continue;
        }
        inside[v as usize] = true;
        stack.extend(
            pred[v as usize]
                .iter()
                .copied()
                .filter(|&p| dom[p as usize][p as usize]),
        );
    }
    (0..n as u32).filter(|&v| inside[v as usize]).collect()
}

/// Cooper-Harvey-Kennedy iterative immediate dominators over `succ`/`pred` from `entry`.
pub fn chk_idom(succ: &[Vec<u32>], pred: &[Vec<u32>], entry: u32) -> Vec<Option<u32>> {
    let n = succ.len();
    let rpo = dfs(&VecGraph {
        succ: succ.to_vec(),
        entry,
    })
    .rpo;
    let mut number = vec![usize::MAX; n];
    for (i, &v) in rpo.iter().enumerate() {
        number[v as usize] = i;
    }
    let mut idom = vec![usize::MAX; n];
    idom[entry as usize] = entry as usize;
    let mut changed = true;
    while changed {
        changed = false;
        for &b in rpo.iter().skip(1) {
            let mut new = usize::MAX;
            for &p in &pred[b as usize] {
                let p = p as usize;
                if idom[p] == usize::MAX {
                    continue;
                }
                if new == usize::MAX {
                    new = p;
                    continue;
                }
                let (mut x, mut y) = (p, new);
                while x != y {
                    while number[x] > number[y] {
                        x = idom[x];
                    }
                    while number[y] > number[x] {
                        y = idom[y];
                    }
                }
                new = x;
            }
            if idom[b as usize] != new {
                idom[b as usize] = new;
                changed = true;
            }
        }
    }
    (0..n)
        .map(|v| {
            if v == entry as usize || idom[v] == usize::MAX {
                None
            } else {
                Some(idom[v] as u32)
            }
        })
        .collect()
}

/// The most irreducible (multi-entry) loops on one nesting chain of the forest.
pub fn irreducible_depth(forest: &BTreeSet<RefLoop>) -> u64 {
    let by_header: std::collections::BTreeMap<u32, &RefLoop> =
        forest.iter().map(|l| (l.header, l)).collect();
    forest
        .iter()
        .map(|l| {
            let mut depth = 0;
            let mut at = Some(l);
            while let Some(x) = at {
                depth += u64::from(x.entries.len() > 1);
                at = x.parent.map(|p| by_header[&p]);
            }
            depth
        })
        .max()
        .unwrap_or(0)
}
