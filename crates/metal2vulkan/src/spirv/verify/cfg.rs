//! Per-function rules: ownership, SSA dominance, structured control flow, canonical loops.

use crate::budget::Meter;
use crate::graph::{Cfg, DomTree, Graph};
use crate::spirv::error::Dangling;
use crate::spirv::ids::{BlockId, FuncId, Value};
use crate::spirv::ir::{Block, Function, Inst, Merge, Terminator, Type, TypedModule, ValueOrigin};

use super::{Ledger, VerifyError, VerifyReport, Violation};

/// "None" in the dense `u32` side tables.
pub(super) const NONE: u32 = u32::MAX;
/// `Defs::index` of a phi result.
const PHI: u32 = u32::MAX - 1;
/// `ModuleConstructs::loop_meta` of a header named by two `LoopMeta`s.
const TWICE: u32 = u32::MAX - 1;

fn ix(v: u32) -> usize {
    v as usize
}

fn at(v: &[u32], i: usize) -> u32 {
    v.get(i).copied().unwrap_or(NONE)
}

fn func_of(m: &TypedModule, f: FuncId) -> Result<&Function, VerifyError> {
    m.functions
        .get(f.index())
        .ok_or(VerifyError::Rule(Violation::Dangling(Dangling::Func(f))))
}

pub(super) fn block_of(m: &TypedModule, b: BlockId) -> Result<&Block, VerifyError> {
    m.blocks
        .get(b.index())
        .ok_or(VerifyError::Rule(Violation::Dangling(Dangling::Block(b))))
}

/// The branch targets of a terminator, in branch order, with duplicates.
pub(super) fn targets(t: &Terminator) -> impl Iterator<Item = BlockId> + '_ {
    let (a, b, cases): (Option<BlockId>, Option<BlockId>, &[(u64, BlockId)]) = match t {
        Terminator::Branch(x) => (Some(*x), None, &[]),
        Terminator::CondBranch { t, f, .. } => (Some(*t), Some(*f), &[]),
        Terminator::Switch { default, cases, .. } => (Some(*default), None, cases.as_slice()),
        Terminator::Return
        | Terminator::ReturnValue(_)
        | Terminator::Kill
        | Terminator::TerminateInvocation
        | Terminator::Unreachable
        | Terminator::EmitMeshTasks { .. } => (None, None, &[]),
    };
    a.into_iter().chain(b).chain(cases.iter().map(|c| c.1))
}

/// Which function lists each block, and at which layout position.
pub(super) struct Layout {
    /// Block -> index of the function that lists it.
    owner: Vec<u32>,
    /// Block -> position in its function's layout.
    pos: Vec<u32>,
}

impl Layout {
    /// Every block is listed by exactly one function, which it names; every target (branch,
    /// merge, continue) is a block of the same function; no function's first block is a target.
    pub(super) fn build(
        m: &TypedModule,
        meter: &mut Meter,
        ledger: &mut Ledger,
    ) -> Result<Layout, VerifyError> {
        let n = m.blocks.len();
        let mut owner = ledger.filled(meter, n, NONE)?;
        let mut pos = ledger.filled(meter, n, NONE)?;
        for (fi, f) in m.functions.iter().enumerate() {
            let fid = FuncId::new(u32::try_from(fi).unwrap_or(NONE));
            if f.blocks.is_empty() {
                return Err(Violation::EmptyFunction(fid).into());
            }
            meter.tick(u32::try_from(f.blocks.len()).unwrap_or(NONE))?;
            for (i, b) in f.blocks.iter().enumerate() {
                if block_of(m, *b)?.func != fid {
                    return Err(Violation::BlockOwnership(*b).into());
                }
                match (owner.get_mut(b.index()), pos.get_mut(b.index())) {
                    (Some(o), Some(p)) if *o == NONE => {
                        *o = fid.0;
                        *p = u32::try_from(i).unwrap_or(NONE);
                    }
                    _ => return Err(Violation::BlockOwnership(*b).into()),
                }
            }
        }
        if let Some(i) = owner.iter().position(|o| *o == NONE) {
            return Err(Violation::BlockOwnership(BlockId::new(i as u32)).into());
        }
        let layout = Layout { owner, pos };
        for (i, b) in m.blocks.iter().enumerate() {
            let from = BlockId::new(i as u32);
            let f = func_of(m, b.func)?;
            let merge_targets = match b.merge {
                Some(Merge::Selection { merge }) => [Some(merge), None],
                Some(Merge::Loop { merge, cont }) => [Some(merge), Some(cont)],
                None => [None, None],
            };
            for to in targets(&b.term).chain(merge_targets.into_iter().flatten()) {
                block_of(m, to)?;
                if layout.owner(to) != b.func.0 {
                    return Err(Violation::ForeignTarget { from, to }.into());
                }
            }
            for to in targets(&b.term) {
                if f.blocks.first() == Some(&to) {
                    return Err(Violation::EntryBlockTarget(to).into());
                }
            }
        }
        Ok(layout)
    }

    fn owner(&self, b: BlockId) -> u32 {
        at(&self.owner, b.index())
    }

    /// The layout position of `b` in its function.
    pub(super) fn pos(&self, b: BlockId) -> u32 {
        at(&self.pos, b.index())
    }
}

/// Where every instruction and phi result is defined.
pub(super) struct Defs {
    /// Value -> defining block, `NONE` for values no instruction or phi defines.
    block: Vec<u32>,
    /// Value -> instruction index in its block, or `PHI`.
    index: Vec<u32>,
}

impl Defs {
    /// Record every definition, checking that each value's origin names its definer and that
    /// nothing is defined twice. Constants, globals, parameters, and variables are checked too.
    pub(super) fn build(
        m: &TypedModule,
        meter: &mut Meter,
        ledger: &mut Ledger,
    ) -> Result<Defs, VerifyError> {
        let n = m.values.len();
        let mut block = ledger.filled(meter, n, NONE)?;
        let mut index = ledger.filled(meter, n, NONE)?;
        let origin = |v: Value| -> Result<ValueOrigin, VerifyError> {
            m.values
                .get(v.index())
                .map(|d| d.origin)
                .ok_or(VerifyError::Rule(Violation::Dangling(Dangling::Value(v))))
        };
        for (bi, b) in m.blocks.iter().enumerate() {
            meter.tick(1)?;
            let bid = BlockId::new(bi as u32);
            let defs = b
                .phis
                .iter()
                .map(|p| (p.result, PHI, ValueOrigin::Phi(bid)))
                .chain(b.insts.iter().enumerate().filter_map(|(i, n)| {
                    n.result
                        .map(|r| (r, u32::try_from(i).unwrap_or(NONE), ValueOrigin::Inst(bid)))
                }));
            for (r, i, want) in defs {
                if origin(r)? != want {
                    return Err(Violation::ResultOrigin(r).into());
                }
                match (block.get_mut(r.index()), index.get_mut(r.index())) {
                    (Some(b), Some(x)) if *b == NONE => {
                        *b = bid.0;
                        *x = i;
                    }
                    _ => return Err(Violation::DefinedTwice(r).into()),
                }
            }
        }
        for (fi, f) in m.functions.iter().enumerate() {
            let func = FuncId::new(fi as u32);
            for (i, p) in f.params.iter().enumerate() {
                let index = u32::try_from(i).unwrap_or(NONE);
                if origin(*p)? != (ValueOrigin::Param { func, index }) {
                    return Err(Violation::ResultOrigin(*p).into());
                }
            }
            for v in &f.vars {
                let var = m
                    .vars
                    .get(v.index())
                    .ok_or(VerifyError::Rule(Violation::Dangling(Dangling::Var(*v))))?;
                if var.func != func || origin(var.value)? != ValueOrigin::Var(*v) {
                    return Err(Violation::ResultOrigin(var.value).into());
                }
            }
        }
        for (gi, g) in m.globals.iter().enumerate() {
            let want = ValueOrigin::Global(crate::spirv::ids::GlobalId::new(gi as u32));
            if origin(g.value)? != want {
                return Err(Violation::ResultOrigin(g.value).into());
            }
        }
        for (ci, v) in m.const_values.iter().enumerate() {
            let want = ValueOrigin::Const(crate::spirv::ids::ConstId::new(ci as u32));
            if origin(*v)? != want {
                return Err(Violation::ResultOrigin(*v).into());
            }
        }
        Ok(Defs { block, index })
    }

    /// The block defining instruction or phi result `v`, if any.
    pub(super) fn block(&self, v: Value) -> Option<BlockId> {
        Some(at(&self.block, v.index()))
            .filter(|b| *b != NONE)
            .map(BlockId::new)
    }

    /// The instruction index of `v` in its block (`None` for a phi or an undefined value).
    pub(super) fn index(&self, v: Value) -> Option<usize> {
        Some(at(&self.index, v.index()))
            .filter(|i| *i != NONE && *i != PHI)
            .map(ix)
    }
}

/// A construct's kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kind {
    If,
    Switch,
    Loop,
}

/// A structured construct: the layout interval `[start, end)` from its header to its merge.
#[derive(Clone, Copy, Debug)]
pub(super) struct Construct {
    pub(super) header: BlockId,
    merge: BlockId,
    cont: Option<BlockId>,
    pub(super) kind: Kind,
    start: u32,
    end: u32,
    /// The innermost enclosing construct, `NONE` at top level.
    pub(super) parent: u32,
    /// The innermost loop construct containing this one (itself for a loop), or `NONE`.
    innermost_loop: u32,
}

impl Construct {
    fn contains(&self, pos: u32) -> bool {
        self.start <= pos && pos < self.end
    }
}

/// Every function's constructs and each block's innermost construct.
pub(super) struct ModuleConstructs {
    pub(super) list: Vec<Construct>,
    /// Block -> index of its innermost construct in `list`, or `NONE`.
    inner: Vec<u32>,
    /// Block -> whether it is already the merge or continue target of a header.
    claimed: Vec<bool>,
    /// Block -> whether it heads a switch-at-use.
    at_use: Vec<bool>,
    /// Block -> index of the `LoopMeta` naming it as header, `NONE`, or `TWICE`.
    loop_meta: Vec<u32>,
}

impl ModuleConstructs {
    /// Empty tables; also checks that every switch-at-use record and every `LoopMeta` names a
    /// header of the right kind.
    pub(super) fn new(
        m: &TypedModule,
        meter: &mut Meter,
        ledger: &mut Ledger,
    ) -> Result<ModuleConstructs, VerifyError> {
        let n = m.blocks.len();
        // A module has at most one construct per block, so `list` never reallocates.
        let list = ledger.with_capacity(meter, n)?;
        let inner = ledger.filled(meter, n, NONE)?;
        let claimed = ledger.filled(meter, n, false)?;
        let mut at_use = ledger.filled(meter, n, false)?;
        let mut loop_meta = ledger.filled(meter, n, NONE)?;
        for h in &m.switches_at_use {
            let b = block_of(m, *h)?;
            let ok = matches!(b.merge, Some(Merge::Selection { .. }))
                && matches!(b.term, Terminator::Switch { .. });
            match at_use.get_mut(h.index()) {
                Some(x) if ok => *x = true,
                _ => return Err(Violation::SwitchAtUseHeader(*h).into()),
            }
        }
        for (li, l) in m.loops.iter().enumerate() {
            let li = u32::try_from(li).unwrap_or(NONE);
            let b = block_of(m, l.header)?;
            if !matches!(b.merge, Some(Merge::Loop { .. })) {
                return Err(Violation::LoopMetaMismatch(li).into());
            }
            if let Some(x) = loop_meta.get_mut(l.header.index()) {
                *x = if *x == NONE { li } else { TWICE };
            }
        }
        Ok(ModuleConstructs {
            list,
            inner,
            claimed,
            at_use,
            loop_meta,
        })
    }

    fn inner_of(&self, b: BlockId) -> u32 {
        at(&self.inner, b.index())
    }

    fn get(&self, c: u32) -> Option<&Construct> {
        self.list.get(ix(c))
    }

    /// The innermost switch-at-use enclosing block `b`, if any.
    fn enclosing_at_use(&self, b: BlockId) -> Option<BlockId> {
        let mut c = self.inner_of(b);
        while let Some(con) = self.get(c) {
            if con.kind == Kind::Switch
                && con.header != b
                && self
                    .at_use
                    .get(con.header.index())
                    .copied()
                    .unwrap_or(false)
            {
                return Some(con.header);
            }
            c = con.parent;
        }
        None
    }
}

/// One function as a `graph::Graph` over layout positions.
struct FnGraph<'a> {
    m: &'a TypedModule,
    f: &'a Function,
    layout: &'a Layout,
}

impl Graph for FnGraph<'_> {
    type Node = u32;

    fn node_count(&self) -> usize {
        self.f.blocks.len()
    }

    fn entry(&self) -> u32 {
        0
    }

    fn successors(&self, node: u32) -> impl Iterator<Item = u32> + '_ {
        // `Layout::build` checked every target is a block of this function, so every position
        // is in range.
        self.f
            .blocks
            .get(ix(node))
            .and_then(|b| self.m.blocks.get(b.index()))
            .into_iter()
            .flat_map(|b| targets(&b.term))
            .map(|t| self.layout.pos(t))
    }
}

/// The analyses of one function.
struct FnCtx<'a> {
    m: &'a TypedModule,
    func: FuncId,
    f: &'a Function,
    layout: &'a Layout,
    defs: &'a Defs,
    cfg: &'a Cfg<u32>,
    dom: &'a DomTree<u32>,
}

impl FnCtx<'_> {
    fn pos(&self, b: BlockId) -> u32 {
        self.layout.pos(b)
    }

    fn reachable(&self, b: BlockId) -> bool {
        self.dom.is_reachable(self.pos(b))
    }

    fn dominates(&self, a: BlockId, b: BlockId) -> bool {
        self.dom.dominates(self.pos(a), self.pos(b))
    }

    fn block_at(&self, p: u32) -> Option<BlockId> {
        self.f.blocks.get(ix(p)).copied()
    }

    /// Distinct predecessors of `b`, sorted by layout position.
    fn preds(&self, b: BlockId) -> Vec<u32> {
        let mut p: Vec<u32> = self.cfg.predecessors(self.pos(b)).collect();
        p.dedup();
        p
    }

    /// The use of `v` by instruction `idx` of block `b` (`idx == insts.len()` for the
    /// terminator) is dominated by its definition.
    fn check_use(&self, v: Value, b: BlockId, idx: usize) -> Result<(), VerifyError> {
        let def = self
            .m
            .values
            .get(v.index())
            .ok_or(VerifyError::Rule(Violation::Dangling(Dangling::Value(v))))?;
        let foreign = Violation::ForeignValue { value: v, block: b };
        match def.origin {
            ValueOrigin::Const(_) | ValueOrigin::Global(_) => Ok(()),
            ValueOrigin::Param { func, .. } if func == self.func => Ok(()),
            ValueOrigin::Var(var) => match self.m.vars.get(var.index()) {
                Some(l) if l.func == self.func => Ok(()),
                Some(_) => Err(foreign.into()),
                None => Err(Violation::Dangling(Dangling::Var(var)).into()),
            },
            ValueOrigin::Param { .. } => Err(foreign.into()),
            ValueOrigin::Removed => Err(Violation::Undefined { value: v, block: b }.into()),
            ValueOrigin::Inst(_) | ValueOrigin::Phi(_) => {
                let d = self
                    .defs
                    .block(v)
                    .ok_or(Violation::Undefined { value: v, block: b })?;
                if block_of(self.m, d)?.func != self.func {
                    return Err(foreign.into());
                }
                let ok = if d == b {
                    // A phi precedes every instruction of its block.
                    self.defs.index(v).is_none_or(|i| i < idx)
                } else {
                    // Every block dominates an unreachable one (no path from the entry reaches
                    // it), as `spirv-val` also reads it.
                    !self.reachable(b) || self.dom.strictly_dominates(self.pos(d), self.pos(b))
                };
                if ok {
                    Ok(())
                } else {
                    Err(Violation::NotDominated { value: v, block: b }.into())
                }
            }
        }
    }

    /// The phi operand `v` arriving from `pred` is available at the end of `pred`.
    fn check_incoming(
        &self,
        v: Value,
        pred: BlockId,
        phi_block: BlockId,
    ) -> Result<(), VerifyError> {
        let def = self
            .m
            .values
            .get(v.index())
            .ok_or(VerifyError::Rule(Violation::Dangling(Dangling::Value(v))))?;
        match def.origin {
            ValueOrigin::Inst(_) | ValueOrigin::Phi(_) => {
                let d = self.defs.block(v).ok_or(Violation::Undefined {
                    value: v,
                    block: phi_block,
                })?;
                if block_of(self.m, d)?.func != self.func {
                    return Err(Violation::ForeignValue {
                        value: v,
                        block: phi_block,
                    }
                    .into());
                }
                if d == pred || !self.reachable(pred) || self.dominates(d, pred) {
                    Ok(())
                } else {
                    Err(Violation::NotDominated {
                        value: v,
                        block: phi_block,
                    }
                    .into())
                }
            }
            // Constants, globals, parameters, and variables do not depend on the path.
            _ => self.check_use(v, pred, usize::MAX),
        }
    }
}

/// The module-wide facts every per-function check reads.
pub(super) struct ModuleFacts<'a> {
    pub(super) m: &'a TypedModule,
    pub(super) layout: &'a Layout,
    pub(super) defs: &'a Defs,
    /// Function -> whether it executes a tangled instruction, directly or through a call.
    pub(super) tangled_fn: &'a [bool],
}

/// Every per-function rule for function `func`.
pub(super) fn check_function(
    facts: &ModuleFacts<'_>,
    func: FuncId,
    cons: &mut ModuleConstructs,
    report: &mut VerifyReport,
    meter: &mut Meter,
) -> Result<(), VerifyError> {
    let ModuleFacts {
        m,
        layout,
        defs,
        tangled_fn,
    } = *facts;
    let f = func_of(m, func)?;
    let graph = FnGraph { m, f, layout };
    let cfg = Cfg::build(&graph, meter)?;
    let dom = match DomTree::compute(&cfg, meter) {
        Ok(d) => d,
        Err(e) => {
            meter.release(cfg.ledger_bytes());
            return Err(e.into());
        }
    };
    let ctx = FnCtx {
        m,
        func,
        f,
        layout,
        defs,
        cfg: &cfg,
        dom: &dom,
    };
    let r = check_function_in(&ctx, cons, tangled_fn, report, meter);
    meter.release(cfg.ledger_bytes());
    meter.release(dom.ledger_bytes());
    r
}

fn check_function_in(
    ctx: &FnCtx<'_>,
    cons: &mut ModuleConstructs,
    tangled_fn: &[bool],
    report: &mut VerifyReport,
    meter: &mut Meter,
) -> Result<(), VerifyError> {
    let m = ctx.m;
    let first = cons.list.len();
    collect_constructs(ctx, cons)?;
    nest_constructs(ctx, cons, first)?;
    for (p, b) in ctx.f.blocks.iter().enumerate() {
        let blk = block_of(m, *b)?;
        let work = 1 + blk.phis.len() + blk.insts.len();
        meter.tick(u32::try_from(work).unwrap_or(NONE))?;
        report.instructions = report.instructions.saturating_add(work as u64);
        check_terminator(ctx, *b, blk)?;
        check_edges(ctx, cons, *b, blk, meter)?;
        if ctx.reachable(*b) {
            report.reachable_blocks = report.reachable_blocks.saturating_add(1);
            if p > 0 {
                let idom = ctx.dom.idom(ctx.pos(*b)).unwrap_or(NONE);
                if ix(idom) >= p {
                    return Err(Violation::LayoutDominance(*b).into());
                }
            }
            if let Some(c) = cons.get(cons.inner_of(*b)) {
                if !ctx.dominates(c.header, *b) {
                    return Err(Violation::HeaderDominance {
                        header: c.header,
                        block: *b,
                    }
                    .into());
                }
            }
        }
        check_ssa(ctx, *b, blk)?;
        check_tangled_uses(cons, tangled_fn, *b, blk)?;
    }
    let mine = cons.list.get(first..).unwrap_or(&[]);
    for c in mine {
        check_construct_dominance(ctx, cons, c)?;
    }
    for c in mine {
        check_construct_shape(ctx, cons, c)?;
    }
    report.constructs = report
        .constructs
        .saturating_add(u32::try_from(cons.list.len() - first).unwrap_or(NONE));
    Ok(())
}

/// Record the constructs of the function, in layout (start) order, and check their merge and
/// continue targets are laid out inside and after them and claimed by no other header.
fn collect_constructs(ctx: &FnCtx<'_>, cons: &mut ModuleConstructs) -> Result<(), VerifyError> {
    for (p, b) in ctx.f.blocks.iter().enumerate() {
        let blk = block_of(ctx.m, *b)?;
        let (merge, cont, kind) = match (blk.merge, &blk.term) {
            (None, _) => continue,
            (Some(Merge::Selection { merge }), Terminator::Switch { .. }) => {
                (merge, None, Kind::Switch)
            }
            (Some(Merge::Selection { merge }), _) => (merge, None, Kind::If),
            (Some(Merge::Loop { merge, cont }), _) => (merge, Some(cont), Kind::Loop),
        };
        let start = u32::try_from(p).unwrap_or(NONE);
        let end = ctx.pos(merge);
        if end <= start {
            return Err(Violation::ConstructLayout(*b).into());
        }
        if let Some(c) = cont {
            let cp = ctx.pos(c);
            if cp <= start || cp >= end {
                return Err(Violation::ConstructLayout(*b).into());
            }
        }
        for t in std::iter::once(merge).chain(cont) {
            match cons.claimed.get_mut(t.index()) {
                Some(x) if !*x => *x = true,
                _ => return Err(Violation::SharedMerge(t).into()),
            }
        }
        cons.list.push(Construct {
            header: *b,
            merge,
            cont,
            kind,
            start,
            end,
            parent: NONE,
            innermost_loop: NONE,
        });
    }
    Ok(())
}

/// Check the function's constructs (from index `first`) nest, set their parents and innermost
/// loops, and record each block's innermost construct.
fn nest_constructs(
    ctx: &FnCtx<'_>,
    cons: &mut ModuleConstructs,
    first: usize,
) -> Result<(), VerifyError> {
    let total = cons.list.len();
    let mut stack: Vec<usize> = Vec::new();
    let mut next = first;
    for (p, b) in ctx.f.blocks.iter().enumerate() {
        let p = u32::try_from(p).unwrap_or(NONE);
        while let Some(&top) = stack.last() {
            if cons.list.get(top).is_some_and(|c| c.end <= p) {
                stack.pop();
            } else {
                break;
            }
        }
        if next < total && cons.list.get(next).is_some_and(|c| c.start == p) {
            let parent = stack.last().copied();
            let (parent_end, parent_loop) = match parent.and_then(|q| cons.list.get(q)) {
                Some(pc) => (pc.end, pc.innermost_loop),
                None => (NONE, NONE),
            };
            let this = u32::try_from(next).unwrap_or(NONE);
            if let Some(c) = cons.list.get_mut(next) {
                if c.end > parent_end {
                    return Err(Violation::ConstructNesting(c.header).into());
                }
                c.parent = parent.map_or(NONE, |q| u32::try_from(q).unwrap_or(NONE));
                c.innermost_loop = if c.kind == Kind::Loop {
                    this
                } else {
                    parent_loop
                };
            }
            stack.push(next);
            next += 1;
        }
        if let Some(x) = cons.inner.get_mut(b.index()) {
            *x = stack
                .last()
                .map_or(NONE, |t| u32::try_from(*t).unwrap_or(NONE));
        }
    }
    Ok(())
}

/// Merge declarations fit terminators; switch literals are distinct and fit the selector.
fn check_terminator(ctx: &FnCtx<'_>, b: BlockId, blk: &Block) -> Result<(), VerifyError> {
    match (&blk.merge, &blk.term) {
        (
            Some(Merge::Selection { .. }),
            Terminator::CondBranch { .. } | Terminator::Switch { .. },
        )
        | (Some(Merge::Loop { .. }), Terminator::Branch(_)) => {}
        (None, Terminator::CondBranch { .. } | Terminator::Switch { .. }) => {
            return Err(Violation::MissingMerge(b).into());
        }
        (Some(_), _) => return Err(Violation::MergeTerminator(b).into()),
        (None, _) => {}
    }
    if let Terminator::Switch { sel, cases, .. } = &blk.term {
        let ty = ctx
            .m
            .values
            .get(sel.index())
            .ok_or(VerifyError::Rule(Violation::Dangling(Dangling::Value(
                *sel,
            ))))?
            .ty;
        let width = match ctx.m.types.get(ty)? {
            Type::Int { width, .. } => u32::from(*width),
            _ => return Err(Violation::SwitchLiterals(b).into()),
        };
        let mut lits: Vec<u64> = cases.iter().map(|c| c.0).collect();
        lits.sort_unstable();
        let fits = |l: u64| width >= 64 || l >> width == 0;
        if lits.windows(2).any(|w| w.first() == w.get(1)) || !lits.iter().all(|l| fits(*l)) {
            return Err(Violation::SwitchLiterals(b).into());
        }
    }
    Ok(())
}

/// Back edges return from a loop's continue block to its header; forward edges enter
/// constructs at their headers and leave them to their merges or the innermost loop's merge or
/// continue target.
fn check_edges(
    ctx: &FnCtx<'_>,
    cons: &ModuleConstructs,
    u: BlockId,
    blk: &Block,
    meter: &mut Meter,
) -> Result<(), VerifyError> {
    let pu = ctx.pos(u);
    let inner_u = cons.inner_of(u);
    let lp = cons.get(inner_u).map_or(NONE, |c| c.innermost_loop);
    let lp = cons.get(lp);
    for v in targets(&blk.term) {
        let pv = ctx.pos(v);
        if pv <= pu {
            let ok = cons
                .get(cons.inner_of(v))
                .is_some_and(|c| c.kind == Kind::Loop && c.header == v && c.cont == Some(u));
            if !ok {
                return Err(Violation::BackEdge { from: u, to: v }.into());
            }
            continue;
        }
        let mut steps = 0u32;
        let mut c = inner_u;
        while let Some(con) = cons.get(c) {
            if con.contains(pv) {
                break;
            }
            let ok = v == con.merge || lp.is_some_and(|l| v == l.merge || Some(v) == l.cont);
            if !ok {
                return Err(Violation::ConstructExit { from: u, to: v }.into());
            }
            c = con.parent;
            steps += 1;
        }
        let mut c = cons.inner_of(v);
        while let Some(con) = cons.get(c) {
            if con.contains(pu) {
                break;
            }
            if con.header != v {
                return Err(Violation::ConstructEntry { from: u, to: v }.into());
            }
            c = con.parent;
            steps += 1;
        }
        if steps > 0 {
            meter.tick(steps)?;
        }
    }
    Ok(())
}

/// Every use in `b` is dominated by its definition; every phi lists exactly the predecessors.
fn check_ssa(ctx: &FnCtx<'_>, b: BlockId, blk: &Block) -> Result<(), VerifyError> {
    if !blk.phis.is_empty() {
        let preds = ctx.preds(b);
        for phi in &blk.phis {
            let mut inc = Vec::with_capacity(phi.incoming.len());
            for (v, p) in &phi.incoming {
                block_of(ctx.m, *p)?;
                if block_of(ctx.m, *p)?.func != ctx.func {
                    return Err(Violation::PhiIncoming {
                        phi: phi.result,
                        block: b,
                    }
                    .into());
                }
                inc.push(ctx.pos(*p));
                ctx.check_incoming(*v, *p, b)?;
            }
            inc.sort_unstable();
            if inc != preds {
                return Err(Violation::PhiIncoming {
                    phi: phi.result,
                    block: b,
                }
                .into());
            }
        }
    }
    let mut err = Ok(());
    for (i, n) in blk.insts.iter().enumerate() {
        n.inst.for_each_value(|v| {
            if err.is_ok() {
                err = ctx.check_use(v, b, i);
            }
        });
        err.clone()?;
    }
    blk.term.for_each_value(|v| {
        if err.is_ok() {
            err = ctx.check_use(v, b, blk.insts.len());
        }
    });
    err
}

/// No tangled instruction, and no call to a function that executes one, inside a
/// switch-at-use.
fn check_tangled_uses(
    cons: &ModuleConstructs,
    tangled_fn: &[bool],
    b: BlockId,
    blk: &Block,
) -> Result<(), VerifyError> {
    let first = blk.insts.iter().find_map(|n| match &n.inst {
        Inst::Call { func, .. } if tangled_fn.get(func.index()).copied().unwrap_or(false) => {
            Some(Some(*func))
        }
        inst if inst.is_tangled() => Some(None),
        _ => None,
    });
    let (Some(callee), Some(header)) = (first, cons.enclosing_at_use(b)) else {
        return Ok(());
    };
    Err(match callee {
        Some(callee) => Violation::TangledCallInSwitchAtUse { header, callee },
        None => Violation::TangledInSwitchAtUse { header, block: b },
    }
    .into())
}

/// A reachable merge, continue target, or nested header is dominated by the construct's header.
fn check_construct_dominance(
    ctx: &FnCtx<'_>,
    cons: &ModuleConstructs,
    c: &Construct,
) -> Result<(), VerifyError> {
    let h = c.header;
    let hd = |block: BlockId| -> Result<(), VerifyError> {
        if ctx.reachable(block) && !ctx.dominates(h, block) {
            Err(Violation::HeaderDominance { header: h, block }.into())
        } else {
            Ok(())
        }
    };
    hd(c.merge)?;
    if let Some(k) = c.cont {
        hd(k)?;
    }
    if let Some(p) = cons.get(c.parent) {
        if ctx.reachable(h) && !ctx.dominates(p.header, h) {
            return Err(Violation::HeaderDominance {
                header: p.header,
                block: h,
            }
            .into());
        }
    }
    Ok(())
}

/// Switch arms have the switch as their only predecessor; loops have canonical shape and their
/// `LoopMeta`.
fn check_construct_shape(
    ctx: &FnCtx<'_>,
    cons: &ModuleConstructs,
    c: &Construct,
) -> Result<(), VerifyError> {
    let h = c.header;
    let blk = block_of(ctx.m, h)?;
    match c.kind {
        Kind::If => Ok(()),
        Kind::Switch => {
            let hp = ctx.pos(h);
            for t in targets(&blk.term) {
                if t == c.merge {
                    continue;
                }
                if ctx.cfg.predecessors(ctx.pos(t)).any(|p| p != hp) {
                    return Err(Violation::CaseFallthrough { header: h, case: t }.into());
                }
            }
            Ok(())
        }
        Kind::Loop => check_loop(ctx, cons, c, blk),
    }
}

/// The canonical loop shape and the loop's `LoopMeta`.
fn check_loop(
    ctx: &FnCtx<'_>,
    cons: &ModuleConstructs,
    c: &Construct,
    blk: &Block,
) -> Result<(), VerifyError> {
    let h = c.header;
    let bad = || VerifyError::Rule(Violation::LoopShape(h));
    let cont = c.cont.ok_or_else(bad)?;
    let first = match blk.term {
        Terminator::Branch(t) => t,
        _ => return Err(bad()),
    };
    if !blk.phis.is_empty() || !blk.insts.is_empty() || first == cont || first == c.merge {
        return Err(bad());
    }
    let cb = block_of(ctx.m, cont)?;
    if !cb.phis.is_empty() || cb.term != Terminator::Branch(h) {
        return Err(bad());
    }
    let preds: Vec<u32> = ctx.cfg.predecessors(ctx.pos(h)).collect();
    let cp = ctx.pos(cont);
    let others: Vec<u32> = preds.iter().copied().filter(|p| *p != cp).collect();
    let preheader = match (preds.len(), others.as_slice()) {
        (2, [p]) => ctx.block_at(*p).ok_or_else(bad)?,
        _ => return Err(bad()),
    };
    if block_of(ctx.m, preheader)?.term != Terminator::Branch(h) {
        return Err(bad());
    }
    let li = at(&cons.loop_meta, h.index());
    let meta = match li {
        NONE => return Err(Violation::LoopMetaMissing(h).into()),
        TWICE => {
            let dup = ctx.m.loops.iter().rposition(|l| l.header == h).unwrap_or(0);
            return Err(Violation::LoopMetaMismatch(u32::try_from(dup).unwrap_or(NONE)).into());
        }
        li => ctx
            .m
            .loops
            .get(ix(li))
            .ok_or(Violation::LoopMetaMismatch(li))?,
    };
    let mismatch = VerifyError::Rule(Violation::LoopMetaMismatch(li));
    if meta.merge != c.merge || meta.cont != cont || meta.preheader != preheader {
        return Err(mismatch);
    }
    if let Some(v) = meta.exit_sel {
        let var = ctx
            .m
            .vars
            .get(v.index())
            .ok_or(Violation::Dangling(Dangling::Var(v)))?;
        let ok = var.func == ctx.func
            && matches!(ctx.m.types.get(var.pointee)?, Type::Int { width: 32, .. });
        if !ok {
            return Err(mismatch);
        }
    }
    Ok(())
}
