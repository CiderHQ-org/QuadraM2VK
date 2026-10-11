//! A metered union-find over dense indices.
//!
//! Components are joined by size, the smaller under the larger, and every find halves its path,
//! so `n` operations cost near-linear work. Every step a find takes is a work unit, and the tables
//! are charged to the ledger as they grow; their owner returns [`UnionFind::charged`] when it
//! drops them. Which indices share a component does not depend on the order of the unions.

use super::provenance::{grow_working, push_working};
use crate::budget::Meter;
use crate::error::{Error, InternalError, InternalKind};

/// The union-find: every index's parent, and each component's size at its representative.
#[derive(Debug, Default)]
pub(crate) struct UnionFind {
    parent: Vec<u32>,
    size: Vec<u32>,
    /// The ledger bytes the tables hold.
    charged: usize,
}

impl UnionFind {
    /// The ledger bytes the tables hold, which the owner returns when it drops them.
    pub(crate) fn charged(&self) -> usize {
        self.charged
    }

    /// Add an index that is a component of its own, and return it.
    pub(crate) fn push(&mut self, m: &mut Meter) -> Result<u32, Error> {
        let i = u32::try_from(self.parent.len()).map_err(|_| invariant("more indices than u32"))?;
        push_working(&mut self.parent, i, &mut self.charged, m)?;
        push_working(&mut self.size, 1, &mut self.charged, m)?;
        Ok(i)
    }

    /// The representative of `i`'s component, halving the path to it: one work unit per step.
    pub(crate) fn find(&mut self, mut i: u32, m: &mut Meter) -> Result<u32, Error> {
        loop {
            let p = self.parent_of(i)?;
            if p == i {
                return Ok(i);
            }
            m.tick(1)?;
            let g = self.parent_of(p)?;
            if let Some(slot) = self.parent.get_mut(i as usize) {
                *slot = g;
            }
            i = g;
        }
    }

    /// A union-find of `n` indices, each a component of its own, its tables charged exactly.
    pub(crate) fn with_len(n: usize, m: &mut Meter) -> Result<UnionFind, Error> {
        let len = u32::try_from(n).map_err(|_| invariant("more indices than u32"))?;
        let mut uf = UnionFind::default();
        grow_working(&mut uf.parent, n, &mut uf.charged, m)?;
        grow_working(&mut uf.size, n, &mut uf.charged, m)?;
        uf.parent.extend(0..len);
        uf.size.resize(n, 1);
        Ok(uf)
    }

    /// Join the components of `a` and `b`, the smaller under the larger.
    pub(crate) fn union(&mut self, a: u32, b: u32, m: &mut Meter) -> Result<(), Error> {
        self.join(a, b, m).map(|_| ())
    }

    /// Join the components of `a` and `b`, the smaller under the larger, and return the joined
    /// component's representative.
    pub(crate) fn join(&mut self, a: u32, b: u32, m: &mut Meter) -> Result<u32, Error> {
        let (ra, rb) = (self.find(a, m)?, self.find(b, m)?);
        if ra == rb {
            return Ok(ra);
        }
        let size = |r: u32| self.size.get(r as usize).copied().unwrap_or(0);
        let (big, small) = if size(ra) >= size(rb) {
            (ra, rb)
        } else {
            (rb, ra)
        };
        let joined = size(big).saturating_add(size(small));
        match (
            self.parent.get_mut(small as usize),
            self.size.get_mut(big as usize),
        ) {
            (Some(p), Some(s)) => {
                *p = big;
                *s = joined;
                Ok(big)
            }
            _ => Err(invariant("a union-find index out of range")),
        }
    }

    fn parent_of(&self, i: u32) -> Result<u32, Error> {
        self.parent
            .get(i as usize)
            .copied()
            .ok_or_else(|| invariant("a union-find index out of range"))
    }
}

fn invariant(detail: &str) -> Error {
    Error::Internal(InternalError::new(InternalKind::Invariant, detail))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meter() -> Meter {
        crate::pipeline::testing::meter()
    }

    /// Unions in any order leave the same components, and the tables' charge is what the ledger
    /// holds.
    #[test]
    fn components_follow_the_unions() {
        let mut m = meter();
        let mut uf = UnionFind::default();
        for _ in 0..6 {
            uf.push(&mut m).expect("push");
        }
        for (a, b) in [(4, 5), (0, 1), (5, 3), (1, 2)] {
            uf.union(a, b, &mut m).expect("union");
        }
        let mut rep = |i| uf.find(i, &mut m).expect("find");
        assert_eq!(rep(0), rep(2));
        assert_eq!(rep(3), rep(4));
        assert_ne!(rep(0), rep(3));
        assert_eq!(rep(5), rep(3));
        assert_ne!(rep(1), rep(5));
        assert_eq!(m.ledger() as usize, uf.charged());
    }

    /// Joined by size with path halving, `n` indices chained pairwise from the last to the first
    /// and then all found cost under 4 units an index; joined without ranks the chain would be
    /// `n` deep.
    #[test]
    fn a_chain_of_unions_takes_near_linear_work() {
        let n = 1 << 16;
        let mut m = meter();
        let mut uf = UnionFind::default();
        for _ in 0..n {
            uf.push(&mut m).expect("push");
        }
        for i in (0..n - 1).rev() {
            uf.union(i, i + 1, &mut m).expect("union");
        }
        let root = uf.find(0, &mut m).expect("find");
        for i in 0..n {
            assert_eq!(uf.find(i, &mut m).expect("find"), root);
        }
        assert!(m.work() < 4 * u64::from(n), "{} units", m.work());
    }
}
