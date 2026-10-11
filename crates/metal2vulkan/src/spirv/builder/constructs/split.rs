//! A switch of more literals than one `OpSwitch` may carry.
//!
//! SPIR-V limits an `OpSwitch` to 16,383 `(literal, label)` pairs
//! ([`SWITCH_PAIRS`](super::SWITCH_PAIRS)), and a
//! switch-at-use has one literal per candidate, so a pointer into a table of 16,384 buffers
//! dispatched on its root was a module `spirv-val` rejects. A switch of more literals than its
//! kind's limit is built as two exact steps instead, each `OpSwitch` within the limit:
//!
//! 1. **The chunk index.** The arms are cut, in case order, into chunks of at most `limit`
//!    literals (an arm with more literals than that is a chunk of its own). The selector's chunk
//!    is found by a switch over the sorted literals, cut into groups of `limit` consecutive
//!    literals: the group a selector value may belong to is the number of groups after the
//!    first whose least literal is not above it (unsigned compares), a switch on that number
//!    picks the group's switch, and that switch maps each of its literals to the index of its
//!    arm's chunk and every other value to the chunk count, the miss. Every arm here is a
//!    constant, so cutting by value never splits an arm that runs code.
//! 2. **The dispatch.** A switch on the chunk index runs a chunk of one arm as that arm and a
//!    chunk of several by a switch on the original selector over its literals (whose default is
//!    unreachable: the index says the selector is one of them), and the miss runs the default
//!    arm.
//!
//! Every arm and the default run exactly once, from one block, for exactly the selector values
//! that chose them before, and their fall-through values merge by `OpPhi` as in one switch. The
//! arms keep their order, nested one construct deeper. A switch-at-use stays one throughout:
//! every switch of both steps is one, so no tangled instruction may enter an arm or the default.
//!
//! The work is a sort of the literals, a sort of each group's literals by chunk, and passes
//! over them, and the output adds a pair per literal in step 1 and a few instructions per group
//! and chunk, within the words a switch's strategy estimated per candidate; its blocks are
//! charged as the builder grows, like any other construct's. The working tables (each literal with
//! its chunk, each arm's chunk, the groups, a group's literals by chunk, and the cases regrouped by
//! chunk) are freed when the split returns; the builder counts their bytes into its footprint for
//! the rest of the module (`ModuleBuilder::count_split_tables`), an upper bound of their peak.

use crate::budget::vec_bytes;
use crate::spirv::builder::constructs::{ArmEnd, ArmFn, CaseBody, Fall, Merged, SwitchKind};
use crate::spirv::builder::function::FunctionBuilder;
use crate::spirv::error::{BResult, BuildError, ControlRule};
use crate::spirv::ids::{BlockId, Value};
use crate::spirv::ir::{BinOp, Terminator};

/// A case of a switch: its literals and its body.
type Case<'a, C> = (Vec<u32>, CaseBody<'a, C>);

/// A case of step 1: its literals and its constant arm.
type IndexCase<'a> = (Vec<u32>, ArmFn<'a, ()>);

impl FunctionBuilder<'_> {
    /// The switch on `sel` over `cases` (distinct literals, more than `kind.limit` of them) and
    /// `default`, as the module's two steps (the module comment).
    pub(super) fn split_switch<'a, C>(
        &mut self,
        ctx: &mut C,
        sel: Value,
        cases: Vec<Case<'a, C>>,
        default: ArmFn<'a, C>,
        kind: SwitchKind,
    ) -> BResult<Merged> {
        let limit = kind.limit;
        if limit < 2 {
            return Err(BuildError::Control(ControlRule::SwitchSplit));
        }
        // The chunk of each arm, in case order.
        let mut chunk_of = Vec::with_capacity(cases.len());
        let (mut chunk, mut fill) = (0usize, 0usize);
        for (lits, _) in &cases {
            if fill > 0 && fill.saturating_add(lits.len()) > limit {
                chunk += 1;
                fill = 0;
            }
            fill = fill.saturating_add(lits.len());
            chunk_of.push(chunk);
        }
        let chunks = chunk + 1;
        let total: usize = cases.iter().map(|(l, _)| l.len()).sum();
        // Step 2 switches on the chunks, one literal each, split again when they are more than
        // `limit`. With a limit of at least 2 they are fewer than the literals (two consecutive
        // chunks hold more than `limit`, so a chunk of one literal is next to a larger one), so
        // the splits end.
        if chunks >= total {
            return Err(BuildError::Control(ControlRule::SwitchSplit));
        }
        let miss =
            u32::try_from(chunks).map_err(|_| BuildError::Control(ControlRule::SwitchSplit))?;
        // Every literal with its arm's chunk, sorted by literal.
        let mut lits: Vec<(u32, u32)> = Vec::with_capacity(total);
        for ((ls, _), &j) in cases.iter().zip(&chunk_of) {
            let j = u32::try_from(j).map_err(|_| BuildError::Control(ControlRule::SwitchSplit))?;
            lits.extend(ls.iter().map(|&l| (l, j)));
        }
        lits.sort_unstable();
        // The groups `chunk_index` cuts `lits` into.
        let groups = total
            .div_ceil(limit)
            .saturating_mul(std::mem::size_of::<&[(u32, u32)]>());
        let mut tables = vec_bytes(&chunk_of)
            .saturating_add(vec_bytes(&lits))
            .saturating_add(groups);
        let idx = self.chunk_index(sel, &lits, miss, kind)?;
        drop(lits);

        let mut by_chunk: Vec<Vec<Case<'a, C>>> = (0..chunks).map(|_| Vec::new()).collect();
        for (case, j) in cases.into_iter().zip(chunk_of) {
            by_chunk
                .get_mut(j)
                .ok_or(BuildError::Control(ControlRule::SwitchSplit))?
                .push(case);
        }
        tables = by_chunk
            .iter()
            .fold(tables.saturating_add(vec_bytes(&by_chunk)), |sum, chunk| {
                sum.saturating_add(vec_bytes(chunk))
            });
        let mut outer = Vec::with_capacity(chunks);
        for (j, chunk) in (0u32..).zip(by_chunk) {
            let body = match <[Case<'a, C>; 1]>::try_from(chunk) {
                Ok([(_, body)]) => body,
                Err(cases) => CaseBody::Chunk { sel, cases },
            };
            outer.push((vec![j], body));
        }
        tables = tables
            .saturating_add(vec_bytes(&outer))
            .saturating_add(chunks.saturating_mul(std::mem::size_of::<u32>()));
        self.module().count_split_tables(tables);
        self.switch_bodies(ctx, idx, outer, default, kind)
    }

    /// Step 1: the chunk index of `sel` among `lits` (each literal with its chunk, sorted by
    /// literal), `miss` (the chunk count) when `sel` is none of them.
    fn chunk_index(
        &mut self,
        sel: Value,
        lits: &[(u32, u32)],
        miss: u32,
        kind: SwitchKind,
    ) -> BResult<Value> {
        let groups: Vec<&[(u32, u32)]> = lits.chunks(kind.limit).collect();
        // The group `sel` may belong to: how many groups after the first start at or below it.
        let one = self.u32_const(1)?;
        let one = self.module().const_value(one)?;
        let zero = self.u32_const(0)?;
        let zero = self.module().const_value(zero)?;
        let mut group: Option<Value> = None;
        for g in groups.iter().skip(1) {
            let Some(&(start, _)) = g.first() else {
                continue;
            };
            let start = self.u32_const(start)?;
            let start = self.module().const_value(start)?;
            let at = self.binary(BinOp::UGreaterThanEqual, sel, start)?;
            let step = self.select(at, one, zero)?;
            group = Some(match group {
                None => step,
                Some(sum) => self.binary(BinOp::IAdd, sum, step)?,
            });
        }
        let group = group.ok_or(BuildError::Control(ControlRule::SwitchSplit))?;
        let mut cases: Vec<IndexCase<'_>> = Vec::with_capacity(groups.len());
        for (g, &members) in (0u32..).zip(&groups) {
            let arm: ArmFn<'_, ()> =
                Box::new(move |fb, _| fb.group_index(sel, members, miss, kind));
            cases.push((vec![g], arm));
        }
        // The last group is the default: `group` is never past it.
        let (_, last) = cases
            .pop()
            .ok_or(BuildError::Control(ControlRule::SwitchSplit))?;
        let merged = self.switch_within(&mut (), group, cases, last, kind)?;
        match merged.0.as_slice() {
            [idx] => Ok(*idx),
            _ => Err(BuildError::Control(ControlRule::SwitchSplit)),
        }
    }

    /// One group's switch of step 1: each literal of `members` (sorted, each with its chunk) to
    /// its chunk's index, every other value to `miss` (the chunk count). The cases are the
    /// group's chunks in order, each with its literals in order: the members sorted by chunk,
    /// so the work is in the group's literals, not in the chunk count.
    fn group_index(
        &mut self,
        sel: Value,
        members: &[(u32, u32)],
        miss: u32,
        kind: SwitchKind,
    ) -> BResult<ArmEnd> {
        let mut by_chunk: Vec<(u32, u32)> = members.iter().map(|&(l, j)| (j, l)).collect();
        by_chunk.sort_unstable();
        let mut cases: Vec<IndexCase<'_>> = Vec::new();
        let mut tables = vec_bytes(&by_chunk);
        for run in by_chunk.chunk_by(|a, b| a.0 == b.0) {
            let Some(&(j, _)) = run.first() else {
                continue;
            };
            if j >= miss {
                return Err(BuildError::Control(ControlRule::SwitchSplit));
            }
            let lits: Vec<u32> = run.iter().map(|&(_, l)| l).collect();
            tables = tables.saturating_add(vec_bytes(&lits));
            cases.push((lits, index_arm(j)));
        }
        tables = tables.saturating_add(vec_bytes(&cases));
        drop(by_chunk);
        self.module().count_split_tables(tables);
        let merged = self.switch_within(&mut (), sel, cases, index_arm(miss), kind)?;
        Ok(ArmEnd::FallThrough(merged.0))
    }

    /// Step 2's case for a chunk of several arms, from fresh block `entry` (a fall-through
    /// branches to `merge`): a switch on the original selector `sel` over the chunk's `cases`,
    /// whose default is unreachable.
    pub(super) fn chunk<C>(
        &mut self,
        ctx: &mut C,
        (entry, merge): (BlockId, BlockId),
        sel: Value,
        cases: Vec<Case<'_, C>>,
        kind: SwitchKind,
    ) -> BResult<Option<Fall>> {
        self.enter(entry)?;
        let unreachable: ArmFn<'_, C> = Box::new(|fb, _| {
            fb.unreachable()?;
            Ok(ArmEnd::Exits)
        });
        let merged = self.switch_bodies(ctx, sel, cases, unreachable, kind)?;
        match self.current_block() {
            Some(end) => {
                self.terminate(None, Terminator::Branch(merge))?;
                Ok(Some((merged.0, end)))
            }
            None => Ok(None),
        }
    }
}

/// An arm of step 1 that falls through with chunk index `j`.
fn index_arm<'a>(j: u32) -> ArmFn<'a, ()> {
    Box::new(move |fb, _| {
        let k = fb.u32_const(j)?;
        Ok(ArmEnd::FallThrough(vec![fb.module().const_value(k)?]))
    })
}
