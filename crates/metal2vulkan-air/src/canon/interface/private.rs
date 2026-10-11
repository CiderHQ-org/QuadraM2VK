//! Private memory in the observation of absent arguments.
//!
//! The frontend often copies every stage input of a fragment or vertex function into a stack
//! allocation (a `VertexOut`-style struct) and hands the struct's address to a surface or
//! geometry function; after linking and inlining, the stores stay although nothing reads the
//! gated-off input's field. [`Private`] lets the observation ([`super::observed_args`]) count
//! such a store only once an observed read reaches its bytes:
//!
//! - **Locations.** A pointer derived from an `alloca` by `getelementptr` and pointer casts
//!   (`bitcast`, `addrspacecast`) points into that allocation ([`Loc::Private`]) at a byte offset:
//!   exact when every index is a constant, else anywhere in it. Any other pointer (a parameter, a
//!   global, one loaded from memory or made from an integer, a `phi` or `select` of pointers) is
//!   [`Loc::Other`], not followed.
//! - **Writes.** A plain (non-volatile, non-atomic) `store`, and a non-volatile `memset`, `memcpy`,
//!   or `memmove`, whose destination points into an allocation writes a span of its bytes. Only
//!   the bytes it writes are deferred: a store's value, a `memset`'s fill byte, and a copy's
//!   source bytes wait until an observed read overlaps the span or the allocation escapes. Every
//!   other operand (the address and the indices that derive it, a length, a volatile flag) is
//!   observed at once, since it decides where and how much the write writes.
//! - **Reads.** An observed `load`, or the source of an observed `memcpy` or `memmove`, through a
//!   pointer into an allocation reads a span of its bytes (every byte when the offset or length is
//!   not a translation-time constant).
//! - **Escapes.** An allocation escapes when a pointer into it is observed other than as the
//!   address of a load, store, or memory intrinsic, or the base of a derivation: stored as a value
//!   (a pointer cell some observed read reaches), passed to a call, merged by a `phi`, `select`,
//!   or `freeze`, converted to an integer, placed in an aggregate, or returned. Every write of an
//!   escaped allocation is observed.
//!
//! **Why this is sound.** Only reads can make a stored value reach a side effect, a branch, or an
//! output. A read through a pointer into the allocation is seen directly. A pointer this analysis
//! does not follow can reach the allocation only through a use of a pointer into it as a value,
//! and that use is observed by the time such a read is: the read observes its address and so
//! whatever produced it, and a pointer that went through memory was stored either as a root (into
//! memory not followed) or into a followed allocation whose bytes the load reads. Reads are
//! matched to writes by byte overlap regardless of order, which can only observe more.
//!
//! **Cost.** Each allocation's writes of a constant span are sorted by start, with a tree of the
//! greatest end over them. A read descends it to the writes that start before its end and end
//! after its start, which are exactly the ones it overlaps, and removes each as it observes it:
//! every write is visited once overall, and a read that finds none costs one descent.
//!
//! Not done:
//! forwarding a pointer loaded back from a cell (`memory::provenance::CellFacts` does it
//! for lowering); such a load escapes the allocation whole.

use crate::budget::Meter;
use crate::error::Error;
use crate::intrinsics::LlvmIntrinsic;
use crate::ir::{
    Body, Call, Callee, CastOp, Const, ConstTable, Idx, Inst, InstId, InstKind, Type, TypeId,
    TypeLayouts, TypeTable, Value,
};

/// Where a pointer value points.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Loc {
    /// Into the allocation made by `alloca`, at byte `offset` from its start (`None`: anywhere in
    /// it).
    Private { alloca: InstId, offset: Option<i64> },
    /// Nowhere this analysis follows.
    Other,
}

/// The bytes of an allocation one access covers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Span {
    /// `start..end`.
    Bytes { start: i64, end: i64 },
    /// Possibly any byte.
    Any,
}

impl Span {
    /// The `len` bytes at `offset` (`None` in either: any byte).
    fn at(offset: Option<i64>, len: Option<u64>) -> Span {
        let len = len.and_then(|l| i64::try_from(l).ok());
        match (offset, len) {
            (Some(start), Some(len)) => start
                .checked_add(len)
                .map_or(Span::Any, |end| Span::Bytes { start, end }),
            _ => Span::Any,
        }
    }
}

/// A resolution in progress (a derivation chain being walked) or done.
#[derive(Clone, Copy)]
enum Resolution {
    Pending,
    Walking,
    Done(Loc),
}

/// The private-memory facts of one entry body: each pointer's [`Loc`], and each allocation's
/// deferred writes and whether they are all observed.
pub(super) struct Private<'a> {
    body: &'a Body,
    types: &'a TypeTable,
    consts: &'a ConstTable,
    params: &'a [TypeId],
    layouts: &'a TypeLayouts,
    /// Per instruction.
    locs: Vec<Loc>,
    /// Per instruction: a deferred write.
    deferred: Vec<bool>,
    /// Deferred writes of a constant span, `(alloca, start, end, write)`, sorted.
    exact: Vec<(InstId, i64, i64, InstId)>,
    /// A max tree over the ends of `exact` (leaf `i` at `leaves + i`), [`REMOVED`] once a write is
    /// observed.
    ends: Vec<i64>,
    /// The first leaf of `ends`: the least power of two at least `exact.len()`.
    leaves: usize,
    /// A read's descent through `ends`: `(node, first leaf, end leaf)`.
    stack: Vec<(usize, usize, usize)>,
    /// Deferred writes of any byte, `(alloca, write)`, sorted.
    anywhere: Vec<(InstId, InstId)>,
    /// Per allocation (by its instruction): every write observed.
    all: Vec<bool>,
    /// Per allocation: its writes of any byte observed.
    anywhere_done: Vec<bool>,
    /// Ledger bytes charged so far.
    charged: usize,
}

impl<'a> Private<'a> {
    /// The facts of `body`, whose parameters have types `params`.
    pub(super) fn new(
        body: &'a Body,
        types: &'a TypeTable,
        consts: &'a ConstTable,
        params: &'a [TypeId],
        layouts: &'a TypeLayouts,
        m: &mut Meter,
    ) -> Result<Private<'a>, Error> {
        let n = body.insts.len();
        let charged = n
            * (std::mem::size_of::<Loc>()
                + std::mem::size_of::<Resolution>()
                + 3 * std::mem::size_of::<bool>());
        m.charge(charged)?;
        let mut p = Private {
            body,
            types,
            consts,
            params,
            layouts,
            locs: Vec::new(),
            deferred: vec![false; n],
            exact: Vec::new(),
            ends: Vec::new(),
            leaves: 0,
            stack: Vec::new(),
            anywhere: Vec::new(),
            all: vec![false; n],
            anywhere_done: vec![false; n],
            charged,
        };
        p.locs = p.resolve(m)?;
        for (id, inst) in body.insts.iter() {
            m.tick(1)?;
            let Some((alloca, span)) = p.write(inst) else {
                continue;
            };
            if let Some(d) = p.deferred.get_mut(id.index()) {
                *d = true;
            }
            match span {
                Span::Bytes { start, end } => {
                    p.charge_grow_exact(m)?;
                    p.exact.push((alloca, start, end, id));
                }
                Span::Any => {
                    let before = p.anywhere.capacity();
                    m.grow(&mut p.anywhere, 1)?;
                    p.charged +=
                        (p.anywhere.capacity() - before) * std::mem::size_of::<(InstId, InstId)>();
                    p.anywhere.push((alloca, id));
                }
            }
        }
        m.tick(u32::try_from(p.exact.len() + p.anywhere.len()).unwrap_or(u32::MAX))?;
        p.exact.sort_unstable();
        p.anywhere.sort_unstable();
        p.build_ends(m)?;
        Ok(p)
    }

    /// Build [`Private::ends`] over the sorted `exact`.
    fn build_ends(&mut self, m: &mut Meter) -> Result<(), Error> {
        let leaves = self.exact.len().next_power_of_two();
        let bytes = 2 * leaves * std::mem::size_of::<i64>();
        m.charge(bytes)?;
        self.charged += bytes;
        m.tick(u32::try_from(2 * leaves).unwrap_or(u32::MAX))?;
        let mut ends = vec![REMOVED; 2 * leaves];
        for (slot, &(_, _, e, _)) in ends.iter_mut().skip(leaves).zip(&self.exact) {
            *slot = e;
        }
        for i in (1..leaves).rev() {
            let (l, r) = (ends.get(2 * i).copied(), ends.get(2 * i + 1).copied());
            if let (Some(slot), Some(l), Some(r)) = (ends.get_mut(i), l, r) {
                *slot = l.max(r);
            }
        }
        self.ends = ends;
        self.leaves = leaves;
        // A depth-first descent holds at most one pending sibling per level.
        let depth = (usize::BITS - leaves.leading_zeros()) as usize + 2;
        m.grow(&mut self.stack, depth)?;
        self.charged += self.stack.capacity() * std::mem::size_of::<(usize, usize, usize)>();
        Ok(())
    }

    fn charge_grow_exact(&mut self, m: &mut Meter) -> Result<(), Error> {
        let before = self.exact.capacity();
        m.grow(&mut self.exact, 1)?;
        self.charged +=
            (self.exact.capacity() - before) * std::mem::size_of::<(InstId, i64, i64, InstId)>();
        Ok(())
    }

    /// Return every ledger byte these facts charged.
    pub(super) fn release(self, m: &mut Meter) {
        m.release(self.charged);
    }

    /// Whether instruction `id` is a deferred write (not observed by itself).
    pub(super) fn deferred(&self, id: InstId) -> bool {
        self.deferred.get(id.index()).copied().unwrap_or(false)
    }

    /// The allocation `v` points into, if it is a pointer this analysis follows.
    pub(super) fn alloca(&self, v: Value) -> Option<InstId> {
        match self.loc(v) {
            Loc::Private { alloca, .. } => Some(alloca),
            Loc::Other => None,
        }
    }

    /// Whether instruction `id` derives a followed pointer from its pointer operand (a
    /// `getelementptr` or pointer cast into an allocation), so that operand is not used as a value.
    pub(super) fn derives(&self, id: InstId) -> bool {
        let into = matches!(self.locs.get(id.index()), Some(Loc::Private { .. }));
        into && matches!(
            self.body.insts.get(id).map(|i| &i.kind),
            Some(InstKind::Gep { .. } | InstKind::Cast { .. })
        )
    }

    /// How many leading operands of `call` are addresses rather than values: a copy's
    /// destination and source, a `memset`'s destination.
    pub(super) fn addresses(call: &Call) -> usize {
        match call.callee {
            Callee::Llvm {
                op: LlvmIntrinsic::Memcpy { .. } | LlvmIntrinsic::Memmove { .. },
                ..
            } => 2,
            Callee::Llvm {
                op: LlvmIntrinsic::Memset { .. },
                ..
            } => 1,
            _ => 0,
        }
    }

    /// The operand of `call` that holds the bytes it writes, when it is a value: a `memset`'s
    /// fill byte (a copy's bytes are its source's, read when it is observed).
    pub(super) fn fill(call: &Call) -> Option<usize> {
        match call.callee {
            Callee::Llvm {
                op: LlvmIntrinsic::Memset { .. },
                ..
            } => Some(1),
            _ => None,
        }
    }

    /// The allocation and span observed instruction `inst` reads: a load's, a copy's source.
    fn read_of(&self, inst: &Inst) -> Option<(InstId, Span)> {
        let (ptr, len) = match &inst.kind {
            InstKind::Load { ptr, .. } => (*ptr, self.layouts.store_size(inst.ty)),
            InstKind::Call(call) if Private::addresses(call) == 2 => {
                (*call.args.get(1)?, self.len(call.args.get(2)))
            }
            _ => return None,
        };
        match self.loc(ptr) {
            Loc::Private { alloca, offset } => Some((alloca, Span::at(offset, len))),
            Loc::Other => None,
        }
    }

    /// The allocation and span `inst` writes when it is a write this analysis defers.
    fn write(&self, inst: &Inst) -> Option<(InstId, Span)> {
        let (ptr, len) = match &inst.kind {
            InstKind::Store {
                val,
                ptr,
                volatile: false,
                atomic: None,
                ..
            } => (*ptr, self.layouts.store_size(self.ty(*val)?)),
            InstKind::Call(call) => match call.callee {
                Callee::Llvm {
                    op:
                        LlvmIntrinsic::Memset {
                            volatile: false, ..
                        }
                        | LlvmIntrinsic::Memcpy {
                            volatile: false, ..
                        }
                        | LlvmIntrinsic::Memmove {
                            volatile: false, ..
                        },
                    ..
                } => (*call.args.first()?, self.len(call.args.get(2))),
                _ => return None,
            },
            _ => return None,
        };
        match self.loc(ptr) {
            Loc::Private { alloca, offset } => Some((alloca, Span::at(offset, len))),
            Loc::Other => None,
        }
    }

    /// An observed read by instruction `inst`: observe (through `observe`) each deferred write its
    /// span overlaps.
    pub(super) fn read(
        &mut self,
        inst: &Inst,
        m: &mut Meter,
        mut observe: impl FnMut(InstId),
    ) -> Result<(), Error> {
        let Some((alloca, span)) = self.read_of(inst) else {
            return Ok(());
        };
        if self.all.get(alloca.index()).copied().unwrap_or(true) {
            return Ok(());
        }
        let Span::Bytes { start, end } = span else {
            self.escape(alloca, observe);
            return Ok(());
        };
        if let Some(done) = self.anywhere_done.get_mut(alloca.index()) {
            if !*done {
                *done = true;
                let (lo, hi) = group(&self.anywhere, alloca, |&(a, _)| a);
                for &(_, w) in self.anywhere.get(lo..hi).unwrap_or(&[]) {
                    observe(w);
                }
            }
        }
        // The writes overlapping `start..end` are those of `alloca` that start before `end` (the
        // leaves `lo..hi`) and end after `start`: descend the max tree to each, and remove it.
        let lo = self.exact.partition_point(|&(a, _, _, _)| a < alloca);
        let hi = self
            .exact
            .partition_point(|&(a, s, _, _)| a < alloca || (a == alloca && s < end));
        let Private {
            exact,
            ends,
            leaves,
            stack,
            ..
        } = self;
        stack.clear();
        stack.push((1, 0, *leaves));
        while let Some((node, from, to)) = stack.pop() {
            m.tick(1)?;
            let greatest = ends.get(node).copied().unwrap_or(REMOVED);
            if to <= lo || hi <= from || greatest <= start {
                continue;
            }
            if to - from == 1 {
                if let Some(&(_, _, _, w)) = exact.get(from) {
                    observe(w);
                }
                remove(ends, node);
                continue;
            }
            let mid = from + (to - from) / 2;
            stack.push((2 * node, from, mid));
            stack.push((2 * node + 1, mid, to));
        }
        Ok(())
    }

    /// Allocation `alloca` escapes (or a read may cover any of its bytes): observe every deferred
    /// write into it.
    pub(super) fn escape(&mut self, alloca: InstId, mut observe: impl FnMut(InstId)) {
        match self.all.get_mut(alloca.index()) {
            Some(all) if !*all => *all = true,
            _ => return,
        }
        let (lo, hi) = group(&self.anywhere, alloca, |&(a, _)| a);
        for &(_, w) in self.anywhere.get(lo..hi).unwrap_or(&[]) {
            observe(w);
        }
        let (lo, hi) = group(&self.exact, alloca, |&(a, _, _, _)| a);
        for &(_, _, _, w) in self.exact.get(lo..hi).unwrap_or(&[]) {
            observe(w);
        }
    }

    fn loc(&self, v: Value) -> Loc {
        match v {
            Value::Inst(i) => self.locs.get(i.index()).copied().unwrap_or(Loc::Other),
            Value::Arg(_) | Value::Const(_) => Loc::Other,
        }
    }

    /// The type of `v`.
    fn ty(&self, v: Value) -> Option<TypeId> {
        match v {
            Value::Inst(i) => self.body.insts.get(i).map(|x| x.ty),
            Value::Arg(a) => self.params.get(a.index()).copied(),
            Value::Const(c) => self.consts.ty(c),
        }
    }

    /// A memory intrinsic's constant length operand.
    fn len(&self, v: Option<&Value>) -> Option<u64> {
        let Some(Value::Const(c)) = v else {
            return None;
        };
        match &self.consts.get(*c)?.value {
            Const::Int(x) => Some(*x),
            Const::Null => Some(0),
            _ => None,
        }
    }

    /// A constant integer index, sign-extended from its width.
    fn index(&self, v: Value) -> Option<i64> {
        let Value::Const(c) = v else { return None };
        let k = self.consts.get(c)?;
        match (&k.value, self.types.get(k.ty)?) {
            (Const::Int(u), Type::Int(w)) => {
                let shift = 64 - u32::from(*w).clamp(1, 64);
                Some(((*u << shift) as i64) >> shift)
            }
            (Const::Null, Type::Int(_)) => Some(0),
            _ => None,
        }
    }

    /// The constant byte displacement of `getelementptr src_ty, _, idx..` (`None`: not constant,
    /// or not laid out).
    fn displacement(&self, src_ty: TypeId, idx: &[Value]) -> Option<i64> {
        let size = |t: TypeId| i64::try_from(self.layouts.alloc_size(t)?).ok();
        let mut off = 0i64;
        let mut cur = src_ty;
        for (n, &i) in idx.iter().enumerate() {
            let k = self.index(i)?;
            let step = if n == 0 {
                k.checked_mul(size(src_ty)?)?
            } else {
                match self.types.get(cur)? {
                    Type::Struct { fields, .. } => {
                        let f = usize::try_from(k).ok()?;
                        let o = *self.layouts.field_offsets(cur)?.get(f)?;
                        cur = *fields.get(f)?;
                        i64::try_from(o).ok()?
                    }
                    Type::Array { elem, .. } => {
                        cur = *elem;
                        k.checked_mul(size(cur)?)?
                    }
                    // LLVM steps a vector by its element's allocation size, which is where the
                    // element is stored only for byte-sized elements.
                    Type::Vector { elem, .. } => {
                        let l = self.layouts.get(*elem)?;
                        if l.alloc_size != l.store_size {
                            return None;
                        }
                        cur = *elem;
                        k.checked_mul(i64::try_from(l.alloc_size).ok()?)?
                    }
                    _ => return None,
                }
            };
            off = off.checked_add(step)?;
        }
        Some(off)
    }

    /// Whether instruction `inst` produces a (scalar) pointer.
    fn is_pointer(&self, inst: &Inst) -> bool {
        matches!(self.types.get(inst.ty), Some(Type::Ptr { .. }))
    }

    /// The pointer `inst` derives from its operand: the operand of a `getelementptr` or pointer
    /// cast producing a pointer.
    fn derived_from(&self, inst: &Inst) -> Option<Value> {
        match &inst.kind {
            InstKind::Gep { base, .. } if self.is_pointer(inst) => Some(*base),
            InstKind::Cast {
                op: CastOp::Bitcast | CastOp::AddrSpaceCast,
                v,
            } if self.is_pointer(inst) => Some(*v),
            _ => None,
        }
    }

    /// The location `inst` points to, given its derivation base's location `base`.
    fn step(&self, inst: &Inst, base: Loc) -> Loc {
        let Loc::Private { alloca, offset } = base else {
            return Loc::Other;
        };
        match &inst.kind {
            InstKind::Gep {
                src_ty, indices, ..
            } => Loc::Private {
                alloca,
                offset: offset.and_then(|o| o.checked_add(self.displacement(*src_ty, indices)?)),
            },
            _ => base,
        }
    }

    /// Every instruction's [`Loc`]: each derivation chain is walked once, from its top down to an
    /// `alloca` or another origin, then resolved back up. A chain that meets itself (only
    /// unreachable code can form one) resolves to [`Loc::Other`].
    fn resolve(&self, m: &mut Meter) -> Result<Vec<Loc>, Error> {
        let n = self.body.insts.len();
        let mut state = vec![Resolution::Pending; n];
        let mut chain: Vec<InstId> = Vec::new();
        for (id, _) in self.body.insts.iter() {
            m.tick(1)?;
            let mut cur = id;
            let mut base = loop {
                let Some(inst) = self.body.insts.get(cur) else {
                    break Loc::Other;
                };
                match state.get(cur.index()).copied() {
                    Some(Resolution::Done(l)) => break l,
                    Some(Resolution::Walking) | None => break Loc::Other,
                    Some(Resolution::Pending) => {}
                }
                if let InstKind::Alloca { .. } = inst.kind {
                    let l = Loc::Private {
                        alloca: cur,
                        offset: Some(0),
                    };
                    set(&mut state, cur, Resolution::Done(l));
                    break l;
                }
                match self.derived_from(inst) {
                    Some(Value::Inst(b)) => {
                        m.tick(1)?;
                        m.grow(&mut chain, 1)?;
                        set(&mut state, cur, Resolution::Walking);
                        chain.push(cur);
                        cur = b;
                    }
                    _ => {
                        set(&mut state, cur, Resolution::Done(Loc::Other));
                        break Loc::Other;
                    }
                }
            };
            while let Some(d) = chain.pop() {
                if let Some(inst) = self.body.insts.get(d) {
                    base = self.step(inst, base);
                }
                set(&mut state, d, Resolution::Done(base));
            }
        }
        m.release(chain.capacity() * std::mem::size_of::<InstId>());
        Ok(state
            .into_iter()
            .map(|s| match s {
                Resolution::Done(l) => l,
                Resolution::Pending | Resolution::Walking => Loc::Other,
            })
            .collect())
    }
}

/// A removed (observed) write's end in [`Private::ends`]: below every span's start.
const REMOVED: i64 = i64::MIN;

/// Remove the observed write at leaf `node` of the max tree `ends`, and update its ancestors.
fn remove(ends: &mut [i64], mut node: usize) {
    if let Some(slot) = ends.get_mut(node) {
        *slot = REMOVED;
    }
    while node > 1 {
        node /= 2;
        let l = ends.get(2 * node).copied().unwrap_or(REMOVED);
        let r = ends.get(2 * node + 1).copied().unwrap_or(REMOVED);
        if let Some(slot) = ends.get_mut(node) {
            *slot = l.max(r);
        }
    }
}

/// Record `r` as instruction `id`'s resolution.
fn set(state: &mut [Resolution], id: InstId, r: Resolution) {
    if let Some(s) = state.get_mut(id.index()) {
        *s = r;
    }
}

/// The range of the entries of `sorted` (sorted by `key` first) whose key is `alloca`.
fn group<T>(sorted: &[T], alloca: InstId, key: impl Fn(&T) -> InstId) -> (usize, usize) {
    (
        sorted.partition_point(|x| key(x) < alloca),
        sorted.partition_point(|x| key(x) <= alloca),
    )
}
