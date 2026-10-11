//! A body under rewrite: instructions keep provisional ids (the original ids, then new ones) and
//! blocks keep their ids (new blocks are appended), so the entry stays block 0. [`Editor::finish`]
//! lays the blocks out in id order and renumbers every instruction operand, as the IR builder
//! does.

use std::collections::BTreeMap;

use metal2vulkan::error::Error;
use metal2vulkan_air::budget::Meter;
use metal2vulkan_air::ir::{
    Block, BlockId, Body, Callee, Idx, Inst, InstId, InstKind, Sym, TailKind, TypeId, Value,
};

pub(super) struct Editor {
    /// Provisional instructions: the original ones at their original ids, then new ones.
    insts: Vec<Inst>,
    /// Per block, its provisional instruction ids in order.
    blocks: Vec<Vec<u32>>,
    names: Vec<Option<Sym>>,
}

impl Editor {
    /// Take `body` apart for editing.
    pub(super) fn new(body: Body) -> Editor {
        let Body { blocks, insts } = body;
        let blocks = blocks.into_vec();
        let names = blocks.iter().map(|b| b.name).collect();
        let runs = blocks
            .iter()
            .map(|b| (b.start.raw()..b.end.raw()).collect())
            .collect();
        Editor {
            insts: insts.into_vec(),
            blocks: runs,
            names,
        }
    }

    /// A new, empty block.
    pub(super) fn new_block(&mut self) -> BlockId {
        self.blocks.push(Vec::new());
        self.names.push(None);
        BlockId::new(self.blocks.len() as u32 - 1)
    }

    /// A new instruction not yet placed in any block.
    pub(super) fn detached(&mut self, ty: TypeId, kind: InstKind) -> (u32, Value) {
        let id = self.insts.len() as u32;
        self.insts.push(Inst { ty, kind });
        (id, Value::Inst(InstId::new(id)))
    }

    /// Append an instruction to `block`.
    pub(super) fn push(&mut self, block: BlockId, ty: TypeId, kind: InstKind) -> Value {
        let (id, value) = self.detached(ty, kind);
        self.blocks[block.index()].push(id);
        value
    }

    /// Place detached instructions at the start of `block` (which must have no PHIs).
    pub(super) fn prepend(&mut self, block: BlockId, run: Vec<u32>) {
        let rest = &mut self.blocks[block.index()];
        let mut out = run;
        out.extend(rest.iter().copied());
        *rest = out;
    }

    /// Append `extra`, the fuel state and the status buffer, to the arguments of the call
    /// `call`, and drop the promises the call made that the callee now breaks: its memory effects
    /// and its `tail` marker. `tail` promises that the callee accesses none of the caller's
    /// allocas (LangRef, `call`), and the state is the entry's alloca. Apple's optimizer acts on
    /// the promise: `metal-opt`'s alias analysis takes such a call to neither read nor write the
    /// state, so dead-store elimination deletes the store of the invocation's fuel before it and
    /// the forced test after it folds away, and the callee reads an uninitialized fuel word.
    /// `notail` promises nothing about the callee's accesses and stays. A `musttail` call makes the
    /// same promise and cannot be followed by the forced test; the plan refuses every function that
    /// takes the state and makes one ([`super::FuelUnsupported::MustTailCall`]), so meeting one
    /// here is an error. The argument list grows in place, with no charge of its own. An indirect
    /// call (its callee closed) calls through `grown`'s prototype for its original one, which its
    /// targets now have.
    pub(super) fn extend_call(
        &mut self,
        call: InstId,
        extra: &[Value],
        grown: &BTreeMap<TypeId, TypeId>,
    ) -> Result<(), String> {
        let InstKind::Call(c) = &mut self.insts[call.index()].kind else {
            return Err(format!("instruction {} is not a call", call.raw()));
        };
        if let Callee::Indirect { fn_ty, .. } = &mut c.callee {
            *fn_ty = grown.get(fn_ty).copied().ok_or_else(|| {
                format!(
                    "the indirect call {} has no grown prototype for its type",
                    call.raw()
                )
            })?;
        }
        c.tail = match c.tail {
            TailKind::Tail | TailKind::None => TailKind::None,
            TailKind::NoTail => TailKind::NoTail,
            TailKind::MustTail => {
                return Err(format!(
                    "the musttail call {} would carry the fuel state",
                    call.raw()
                ))
            }
        };
        let mut args = c.args.to_vec();
        args.extend_from_slice(extra);
        c.args = args.into();
        // The callee now writes the thread's state and the status buffer (the call's enum kinds
        // are the table's, which `emit` edits).
        if let Some(attrs) = &mut c.attrs {
            super::emit::writes_state(&mut attrs.func, None);
        }
        Ok(())
    }

    /// Move the instructions after `inst` in `block` into a new block and return it.
    pub(super) fn split_after(&mut self, block: BlockId, inst: InstId) -> Option<BlockId> {
        let run = &mut self.blocks[block.index()];
        let at = run.iter().position(|&i| i == inst.raw())?;
        let tail = run.split_off(at + 1);
        let new = self.new_block();
        self.blocks[new.index()] = tail;
        Some(new)
    }

    /// Redirect every edge of `block`'s terminator from `from` to `to`.
    pub(super) fn retarget(&mut self, block: BlockId, from: BlockId, to: BlockId) {
        let Some(&last) = self.blocks[block.index()].last() else {
            return;
        };
        let swap = |b: &mut BlockId| {
            if *b == from {
                *b = to;
            }
        };
        match &mut self.insts[last as usize].kind {
            InstKind::Br { dest, .. } => swap(dest),
            InstKind::CondBr { t, f, .. } => {
                swap(t);
                swap(f);
            }
            InstKind::Switch { default, cases, .. } => {
                swap(default);
                for (_, b) in cases.iter_mut() {
                    swap(b);
                }
            }
            _ => {}
        }
    }

    /// Rewrite the incoming blocks of every PHI in `block` with `map`, then drop repeated entries
    /// for the blocks in `single` (a block with one edge to `block` has one entry).
    pub(super) fn remap_phis(
        &mut self,
        block: BlockId,
        map: impl Fn(BlockId) -> BlockId,
        single: impl Fn(BlockId) -> bool,
    ) {
        let ids = self.blocks[block.index()].clone();
        for id in ids {
            let InstKind::Phi { incoming, .. } = &mut self.insts[id as usize].kind else {
                break;
            };
            let mut out: Vec<(Value, BlockId)> = Vec::with_capacity(incoming.len());
            for &(v, b) in incoming.iter() {
                let nb = map(b);
                if single(nb) && out.iter().any(|&(_, seen)| seen == nb) {
                    continue;
                }
                out.push((v, nb));
            }
            *incoming = out.into_boxed_slice();
        }
    }

    /// Lay the blocks out in id order, renumber operands, and rebuild the body. Each instruction
    /// enters it through [`Body::push_inst`], so the body is charged its slots and its payloads
    /// as they are after the edits ([`Editor::extend_call`] lengthens a call's arguments in
    /// place), as every other body is built. Also returns each provisional id's final id (`None`
    /// for an instruction no block holds).
    pub(super) fn finish(self, meter: &mut Meter) -> Result<(Body, Vec<Option<InstId>>), Error> {
        let Editor {
            insts,
            blocks,
            names,
        } = self;
        let mut final_id = vec![u32::MAX; insts.len()];
        let mut next = 0u32;
        for run in &blocks {
            for &t in run {
                final_id[t as usize] = next;
                next += 1;
            }
        }
        // Each provisional id sits in at most one block, so its final id is unique; sorting by
        // final id yields the layout order.
        let mut laid: Vec<(u32, Inst)> = final_id
            .iter()
            .copied()
            .zip(insts)
            .filter(|&(id, _)| id != u32::MAX)
            .collect();
        laid.sort_unstable_by_key(|&(id, _)| id);
        let mut laid = laid.into_iter();
        let mut body = Body::default();
        for (run, name) in blocks.iter().zip(names) {
            let start = InstId::new(body.insts.len() as u32);
            for (_, mut inst) in laid.by_ref().take(run.len()) {
                inst.kind.for_each_value_mut(|v| {
                    if let Value::Inst(i) = v {
                        *v = Value::Inst(InstId::new(final_id[i.index()]));
                    }
                });
                body.push_inst(inst, meter)?;
            }
            let end = InstId::new(body.insts.len() as u32);
            body.blocks.push(Block { start, end, name }, meter)?;
        }
        let renumbered = final_id
            .into_iter()
            .map(|id| (id != u32::MAX).then(|| InstId::new(id)))
            .collect();
        Ok((body, renumbered))
    }
}
