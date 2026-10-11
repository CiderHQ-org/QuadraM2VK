//! CFG generators: exhaustive canonical enumerations and seeded families.
//!
//! Every generator is deterministic from its arguments. Graphs are successor lists (see
//! [`TestCfg::from_successors`]): no successor returns, one branches, two is a conditional
//! branch, three or more is a switch whose first successor is the default.

use super::hash::{mix, MIX};
use super::view::TestCfg;

/// SplitMix64, so every seeded family is reproducible.
#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        let z = mix(self.0);
        self.0 = self.0.wrapping_add(MIX[0]);
        z
    }

    /// Uniform in `0..bound` (`bound > 0`).
    pub fn below(&mut self, bound: u64) -> u64 {
        self.next_u64() % bound
    }

    /// True with probability `numerator / 16`.
    pub fn chance(&mut self, numerator: u64) -> bool {
        self.below(16) < numerator
    }
}

/// Successor choices for out-degree ≤ 2: none, `[a]`, and `[a, b]` with `a < b`.
pub fn out_degree_two_choices(n: u32) -> Vec<Vec<u32>> {
    let mut sets = vec![Vec::new()];
    sets.extend((0..n).map(|a| vec![a]));
    for a in 0..n {
        sets.extend((a + 1..n).map(|b| vec![a, b]));
    }
    sets
}

/// [`out_degree_two_choices`] plus switches: `[a, b, c]` (`a < b < c`, default `a`), `[a, b, b]`
/// (two literals to one target), and `[a, b, a]` (a case that targets the default).
pub fn switch_choices(n: u32) -> Vec<Vec<u32>> {
    let mut sets = out_degree_two_choices(n);
    for a in 0..n {
        for b in a + 1..n {
            sets.push(vec![a, b, b]);
            sets.push(vec![a, b, a]);
            sets.extend((b + 1..n).map(|c| vec![a, b, c]));
        }
    }
    sets
}

/// Every graph on `n` blocks whose successor lists come from `choices` and in which every block
/// is reachable from `bb0`, up to relabeling: exactly those whose DFS preorder (successors in
/// list order) is `0, 1, …, n-1`. Every such graph is isomorphic to one of these.
pub fn canonical(n: u32, choices: &[Vec<u32>], mut f: impl FnMut(&TestCfg)) {
    canonical_lists(n, choices, &mut |succ| f(&TestCfg::from_successors(succ)));
}

/// [`canonical`], building and passing to `f` only the graphs whose enumeration index (0, 1, …
/// in [`canonical`]'s order) `keep` accepts. Returns how many graphs were enumerated.
pub fn canonical_sampled(
    n: u32,
    choices: &[Vec<u32>],
    keep: impl Fn(u64) -> bool,
    mut f: impl FnMut(&TestCfg),
) -> u64 {
    let mut i = 0u64;
    canonical_lists(n, choices, &mut |succ| {
        if keep(i) {
            f(&TestCfg::from_successors(succ));
        }
        i += 1;
    });
    i
}

/// Whether enumeration index `i` is in the deterministic one-in-`every` sample: a hash of the
/// index, so the sample does not alias with the enumeration order (consecutive indices differ
/// only in the last blocks' successors). `every = 1` keeps every index.
pub fn in_sample(i: u64, every: u64) -> bool {
    mix(i).is_multiple_of(every.max(1))
}

/// [`canonical`] over successor lists.
fn canonical_lists(n: u32, choices: &[Vec<u32>], f: &mut impl FnMut(&[Vec<u32>])) {
    let mut succ = vec![Vec::new(); n as usize];
    for s in choices {
        succ[0] = s.clone();
        visit(n, choices, &mut succ, 1, vec![(0, 0)], f);
    }
}

/// Continue the DFS: `k` blocks are numbered, `stack` holds (block, next successor position). An
/// unvisited successor must be exactly `k`, whose successor list is chosen now.
fn visit(
    n: u32,
    choices: &[Vec<u32>],
    succ: &mut Vec<Vec<u32>>,
    k: u32,
    mut stack: Vec<(u32, usize)>,
    f: &mut impl FnMut(&[Vec<u32>]),
) {
    loop {
        let Some(&(v, i)) = stack.last() else {
            if k == n {
                f(succ);
            }
            return;
        };
        let Some(&s) = succ[v as usize].get(i) else {
            stack.pop();
            continue;
        };
        if let Some(top) = stack.last_mut() {
            top.1 += 1;
        }
        if s < k {
            continue;
        }
        if s > k {
            return;
        }
        for set in choices {
            succ[k as usize] = set.clone();
            let mut next = stack.clone();
            next.push((k, 0));
            visit(n, choices, succ, k + 1, next, f);
        }
        return;
    }
}

/// A terminator of the structured generator before it becomes a successor list.
#[derive(Clone, Debug)]
enum Term {
    Branch(u32),
    Conditional(u32, u32),
    Multiway { arms: Vec<u32>, default: u32 },
    Return,
}

/// Structured programs with goto injection: a region tree (sequence /
/// conditional / switch / loop) grown from a seed, perforated with escape edges to an enclosing
/// region's exit (multi-level `break`) or an enclosing loop's header (`continue`). Every escape is
/// forward or a back edge to a dominating header, so the result is reducible. `crossings` edges
/// into the interior of an already written region make it irreducible.
pub fn structured(seed: u64, depth: u32, crossings: u32) -> TestCfg {
    let mut g = Structured {
        rng: Rng::new(seed),
        terms: Vec::new(),
        settled: Vec::new(),
        loop_headers: Vec::new(),
        region_exits: Vec::new(),
        escapes: 0,
        escape_budget: 24,
        cross_budget: crossings,
    };
    let entry = g.allocate();
    let exit = g.allocate();
    let start = g.allocate();
    g.set(entry, Term::Branch(start));
    g.region(start, exit, depth);
    g.set(exit, Term::Return);
    let succ: Vec<Vec<u32>> = g
        .terms
        .into_iter()
        .map(|t| match t.expect("every allocated block is terminated") {
            Term::Branch(t) => vec![t],
            Term::Conditional(t, f) => vec![t, f],
            Term::Multiway { arms, default } => std::iter::once(default).chain(arms).collect(),
            Term::Return => vec![],
        })
        .collect();
    TestCfg::from_successors(&succ)
}

struct Structured {
    rng: Rng,
    terms: Vec<Option<Term>>,
    settled: Vec<u32>,
    loop_headers: Vec<u32>,
    region_exits: Vec<u32>,
    escapes: u32,
    escape_budget: u32,
    cross_budget: u32,
}

impl Structured {
    fn allocate(&mut self) -> u32 {
        self.terms.push(None);
        self.terms.len() as u32 - 1
    }

    fn set(&mut self, block: u32, term: Term) {
        assert!(
            self.terms[block as usize].is_none(),
            "{block} terminated twice"
        );
        self.terms[block as usize] = Some(term);
        self.settled.push(block);
    }

    fn cross_edge(&mut self, from: u32) -> Option<u32> {
        if self.cross_budget == 0 {
            return None;
        }
        let candidates: Vec<u32> = self
            .settled
            .iter()
            .copied()
            .filter(|b| {
                *b != 0
                    && *b != from
                    && !self.loop_headers.contains(b)
                    && !self.region_exits.contains(b)
            })
            .collect();
        if candidates.is_empty() {
            return None;
        }
        self.cross_budget -= 1;
        Some(candidates[self.rng.below(candidates.len() as u64) as usize])
    }

    fn escape_target(&mut self) -> Option<u32> {
        if self.escapes >= self.escape_budget {
            return None;
        }
        let candidates: Vec<u32> = self
            .region_exits
            .iter()
            .chain(&self.loop_headers)
            .copied()
            .collect();
        if candidates.is_empty() {
            return None;
        }
        self.escapes += 1;
        Some(candidates[self.rng.below(candidates.len() as u64) as usize])
    }

    fn flow(&mut self, block: u32, next: u32) {
        if self.rng.chance(4) {
            if let Some(target) = self.cross_edge(block) {
                self.set(block, Term::Conditional(target, next));
                return;
            }
        }
        if self.rng.chance(6) {
            if let Some(target) = self.escape_target() {
                self.set(block, Term::Conditional(target, next));
                return;
            }
        }
        self.set(block, Term::Branch(next));
    }

    fn region(&mut self, entry: u32, exit: u32, depth: u32) {
        if depth == 0 {
            self.flow(entry, exit);
            return;
        }
        match self.rng.below(4) {
            0 => self.sequence(entry, exit, depth),
            1 => self.conditional(entry, exit, depth),
            2 => self.multiway(entry, exit, depth),
            _ => self.loop_region(entry, exit, depth),
        }
    }

    fn multiway(&mut self, entry: u32, exit: u32, depth: u32) {
        let arms: Vec<u32> = (0..3).map(|_| self.allocate()).collect();
        let default = self.allocate();
        self.set(
            entry,
            Term::Multiway {
                arms: arms.clone(),
                default,
            },
        );
        self.region_exits.push(exit);
        for arm in arms.into_iter().chain(std::iter::once(default)) {
            self.region(arm, exit, depth - 1);
        }
        self.region_exits.pop();
    }

    fn sequence(&mut self, entry: u32, exit: u32, depth: u32) {
        let length = 1 + self.rng.below(3);
        let mut current = entry;
        for step in 0..length {
            let next = if step + 1 == length {
                exit
            } else {
                self.allocate()
            };
            self.region_exits.push(next);
            self.region(current, next, depth - 1);
            self.region_exits.pop();
            current = next;
        }
    }

    fn conditional(&mut self, entry: u32, exit: u32, depth: u32) {
        let t = self.allocate();
        let f = self.allocate();
        self.set(entry, Term::Conditional(t, f));
        self.region_exits.push(exit);
        self.region(t, exit, depth - 1);
        self.region(f, exit, depth - 1);
        self.region_exits.pop();
    }

    fn loop_region(&mut self, header: u32, exit: u32, depth: u32) {
        let body = self.allocate();
        let latch = self.allocate();
        self.set(header, Term::Conditional(body, exit));
        self.loop_headers.push(header);
        self.region_exits.push(latch);
        self.region(body, latch, depth - 1);
        self.region_exits.pop();
        self.loop_headers.pop();
        self.set(latch, Term::Branch(header));
    }
}

/// A random DAG over `n` blocks (edges `i -> j`, `i < j`; out-degree 0–3, skewed to 1–2) plus
/// random back edges `i -> j` with `j <= i`. Often irreducible: filter with the loop forest for
/// the reducible family.
pub fn dag_with_back_edges(n: u32, seed: u64) -> TestCfg {
    let mut rng = Rng::new(seed);
    let succ: Vec<Vec<u32>> = (0..n)
        .map(|i| {
            if i + 1 == n {
                return vec![];
            }
            let degree = match rng.below(16) {
                0 => 0,
                1..=7 => 1,
                8..=14 => 2,
                _ => 3,
            };
            let mut s = Vec::new();
            for d in 0..degree {
                let back = d > 0 && rng.chance(3);
                let t = if back {
                    rng.below(u64::from(i) + 1) as u32
                } else {
                    i + 1 + rng.below(u64::from(n - i - 1)) as u32
                };
                if !s.contains(&t) {
                    s.push(t);
                }
            }
            s
        })
        .collect();
    TestCfg::from_successors(&succ)
}

/// `k` two-entry (irreducible) cycles in sequence (`3k + 1` blocks): `b -> {b+1, b+2}`,
/// `b+1 <-> b+2`, `b+2 -> b+3`.
pub fn irreducible_ladder(k: u32) -> TestCfg {
    let mut succ = Vec::new();
    for i in 0..k {
        let b = 3 * i;
        succ.push(vec![b + 1, b + 2]);
        succ.push(vec![b + 2]);
        succ.push(vec![b + 1, b + 3]);
    }
    succ.push(vec![]);
    TestCfg::from_successors(&succ)
}

/// `k` nested two-entry cycles: level `i` is `a_i <-> b_i`, entered at both blocks (from the
/// entry for level 0, from `a_{i-1}` otherwise), and `b_i` returns to `b_{i-1}` (level 0 exits).
/// Every level is irreducible and the nesting depth is `k`.
pub fn nested_irreducible(k: u32) -> TestCfg {
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
    TestCfg::from_successors(&succ)
}

/// `k` loops in sequence, each with a break tail that rejoins the normal exit (`5k + 1`
/// blocks): `h -> {body, e}`, `body -> {tail, latch}`, `tail -> e`, `latch -> h`, `e -> next`.
pub fn break_tail_ladder(k: u32) -> TestCfg {
    let mut succ = Vec::new();
    for i in 0..k {
        let h = 5 * i;
        let (body, tail, latch, e) = (h + 1, h + 2, h + 3, h + 4);
        succ.push(vec![body, e]);
        succ.push(vec![tail, latch]);
        succ.push(vec![e]);
        succ.push(vec![h]);
        succ.push(vec![e + 1]);
    }
    succ.push(vec![]);
    TestCfg::from_successors(&succ)
}
