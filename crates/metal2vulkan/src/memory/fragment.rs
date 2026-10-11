//! Simdgroup matrices held as their lanes' fragments in typed objects.
//!
//! A simdgroup matrix's AIR value is `<64 x T>`, but a lane holds only its fragment, elements 0
//! and 1, and every matrix operation writes zero to elements 2..63. An object that holds matrices
//! therefore needs only each matrix's two elements. Held whole in the private arena, a 16 x 16
//! array of `float` matrices is 64 KiB a thread, past every measured device's Function storage but
//! the Apple GPUs'; held as fragments it is 2 KiB.
//!
//! **Fragments, by structure.** A `<64 x T>` value is a *fragment* when each of its elements
//! 2..63 is zero (every bit clear) or undefined (`undef` or `poison`, which zero refines). The
//! analysis ([`held`]) is a greatest fixpoint over the entry's `<64 x T>` values and the
//! *candidate* allocas (those the memory plan would type but for their `<64 x T>` members):
//!
//! - the result of a simdgroup-matrix `load`, `init_diag`, `init_filled`, or
//!   `multiply_accumulate` is one (`lower::simd::matrix` writes zero to elements 2..63);
//! - a constant is one when each element 2..63 is zero or undefined;
//! - `insertelement` into a fragment is one when its index is 0 or 1 whenever it runs
//!   (`Fixpoint::low_indices`): an integer constant 0 or 1, any `i1` value, an `and` with such an
//!   operand, a `trunc` or `zext` of such, a `sext` of such from more than one bit, and a `phi` or
//!   `select` whose every incoming value is such (a greatest fixpoint, so a loop-carried index
//!   qualifies when every value entering the loop does); `undef` and `poison` indices of more than
//!   one bit do not, as they may name any element;
//! - `shufflevector` is one when each result element 2..63 is undefined, or an element known to
//!   be zero: an element 2..63 of a fragment operand, or a zero or undefined element of a
//!   constant operand;
//! - a `phi` or `select` of fragments is one;
//! - a load is one when every root it reads is a held candidate;
//! - nothing else is.
//!
//! A candidate is *held* while every `<64 x T>` value stored into it is a fragment and no access
//! reaches an element of one of its `<64 x T>` members past the held ones: a load or store of
//! elements 0 and 1 as a `<2 x T>` at the matrix's start, or of one of them as a `T` (MSL's
//! `thread_elements()` through memory), reaches only what the fragment holds.
//!
//! Everything starts as a fragment or held; whatever fails its rule is removed, and what reads it
//! is visited again, until nothing changes.
//! Each value and candidate is removed at most once, so the work is linear in the values, their
//! uses, and each candidate's loads.
//!
//! A held alloca is typed with each `<64 x T>` member as a 2-lane vector of the matrix's held
//! element (`alloca::TypedNode::Fragment`): a store keeps elements 0 and 1, and a load rebuilds
//! the value with zero in elements 2..63, which is what the value it reads back held, or a
//! refinement of it. Its private memory is its members' bytes with each matrix as two elements
//! ([`bytes`]).

use std::collections::{BTreeMap, BTreeSet};

use super::alloca::{self, Step};
use super::provenance::{self, grow_working, map_entry_bytes, Provenance, Root};
use crate::air::intrinsics::{AirIntrinsic, SimdMatrix};
use crate::air::ir::{
    BinOp, Body, Callee, CastOp, Const, ConstId, InstId, InstKind, Type, TypeId, TypeLayouts,
    TypeTable, Value,
};
use crate::air::Entry;
use crate::budget::Meter;
use crate::error::Error;
use crate::spirv::ids::DemoteTy;

/// The elements of a simdgroup matrix's AIR value.
const LANES: u32 = 64;

/// The elements a lane's fragment holds: 0 and 1.
const HELD: u32 = 2;

/// Whether AIR type `ty` is a simdgroup matrix's value, a vector of 64 elements.
pub fn is_matrix(types: &TypeTable, ty: TypeId) -> bool {
    matches!(types.get(ty), Some(Type::Vector { len, .. }) if *len == LANES)
}

/// The held form of the fragment of a matrix held as `d` (its chunked vector): a 2-lane vector of
/// the chunks' element.
pub fn fragment_form(d: &DemoteTy) -> Option<DemoteTy> {
    let DemoteTy::Composite(chunks) = d else {
        return None;
    };
    match chunks.first()? {
        DemoteTy::Vector { elem, len } if u32::from(*len) >= HELD => Some(DemoteTy::Vector {
            elem: *elem,
            len: HELD as u8,
        }),
        _ => None,
    }
}

/// Whether a typed access along `steps` into an object of type `obj` reaches an element of a
/// matrix member, past the member itself (the first step is the object level).
pub fn enters_matrix(types: &TypeTable, obj: TypeId, steps: &[Step]) -> bool {
    let mut ty = obj;
    for step in steps.iter().skip(1) {
        if is_matrix(types, ty) {
            return true;
        }
        let next = match (types.get(ty), step) {
            (Some(Type::Array { elem, .. } | Type::Vector { elem, .. }), _) => Some(*elem),
            (Some(Type::Struct { fields, .. }), Step::Const(i)) => fields.get(*i as usize).copied(),
            _ => None,
        };
        let Some(next) = next else {
            return false;
        };
        ty = next;
    }
    false
}

/// The private memory a held alloca's element of type `ty` holds: its members' bytes, each
/// matrix as its two held elements, with no padding between members (the `Private` variable's
/// layout is the driver's). `None` for a type with no layout.
pub fn bytes(
    types: &TypeTable,
    layouts: &TypeLayouts,
    ty: TypeId,
    m: &mut Meter,
) -> Result<Option<u64>, Error> {
    let mut sizes: BTreeMap<TypeId, u64> = BTreeMap::new();
    let mut charged = 0usize;
    let out = size_of(types, layouts, ty, 0, &mut sizes, &mut charged, m);
    m.release(charged);
    out
}

fn size_of(
    types: &TypeTable,
    layouts: &TypeLayouts,
    ty: TypeId,
    depth: u32,
    sizes: &mut BTreeMap<TypeId, u64>,
    charged: &mut usize,
    m: &mut Meter,
) -> Result<Option<u64>, Error> {
    m.tick(1)?;
    if let Some(&s) = sizes.get(&ty) {
        return Ok(Some(s));
    }
    // Typing the object bounded its nesting (`alloca::typed_ty`).
    if depth > alloca::MAX_DEPTH {
        return Ok(None);
    }
    let size = match types.get(ty) {
        Some(Type::Array { elem, len }) => {
            match size_of(types, layouts, *elem, depth + 1, sizes, charged, m)? {
                Some(s) => s.saturating_mul(*len),
                None => return Ok(None),
            }
        }
        Some(Type::Struct { fields, .. }) => {
            let mut sum = 0u64;
            for f in fields.iter() {
                match size_of(types, layouts, *f, depth + 1, sizes, charged, m)? {
                    Some(s) => sum = sum.saturating_add(s),
                    None => return Ok(None),
                }
            }
            sum
        }
        Some(Type::Vector { elem, .. }) if is_matrix(types, ty) => {
            match layouts.alloc_size(*elem) {
                Some(s) => s.saturating_mul(u64::from(HELD)),
                None => return Ok(None),
            }
        }
        _ => match layouts.alloc_size(ty) {
            Some(s) => s,
            None => return Ok(None),
        },
    };
    m.charge(map_entry_bytes::<TypeId, u64>())?;
    *charged = charged.saturating_add(map_entry_bytes::<TypeId, u64>());
    sizes.insert(ty, size);
    Ok(Some(size))
}

/// Add `x` to `key`'s list in `map`, charging the entry and the list's growth to `charged`.
fn push_to(
    map: &mut BTreeMap<InstId, Vec<InstId>>,
    key: InstId,
    x: InstId,
    charged: &mut usize,
    m: &mut Meter,
) -> Result<(), Error> {
    if !map.contains_key(&key) {
        let entry = map_entry_bytes::<InstId, Vec<InstId>>();
        m.charge(entry)?;
        *charged = charged.saturating_add(entry);
    }
    let list = map.entry(key).or_default();
    grow_working(list, 1, charged, m)?;
    list.push(x);
    Ok(())
}

/// The analysis's state (the module docs).
struct Fixpoint<'a> {
    e: &'a Entry,
    prov: &'a Provenance,
    /// Each `<64 x T>` instruction, and whether it is still a fragment.
    values: BTreeMap<InstId, bool>,
    /// What to visit again when an instruction stops being a fragment: the `<64 x T>`
    /// instructions and the stores into candidates that read it.
    users: BTreeMap<InstId, Vec<InstId>>,
    /// Each store of a `<64 x T>` value into candidates: the value and those candidates.
    stores: BTreeMap<InstId, (Value, Vec<InstId>)>,
    /// What to visit again when a candidate stops being held: the loads of `<64 x T>` from it.
    readers: BTreeMap<InstId, Vec<InstId>>,
    held: BTreeSet<InstId>,
    /// The instructions an `insertelement` into a `<64 x T>` value reaches as its index whose
    /// every value is 0 or 1 (`Fixpoint::low_indices`).
    low: BTreeSet<InstId>,
    /// The ledger bytes the state holds.
    charged: usize,
}

/// What the analysis finds (the module docs).
pub struct Fragments {
    /// The candidates that hold only fragments.
    pub held: BTreeSet<InstId>,
    /// The `<64 x T>` instructions that are fragments, sorted: a shuffle among them moves only
    /// elements 0 and 1 (`lower::arith::chunk::shuffle_fragment`).
    pub values: Vec<InstId>,
}

/// The candidates of `candidates` that hold only fragments, and the `<64 x T>` instructions that
/// are fragments (the module docs). The ledger keeps `values`' bytes.
pub fn held(
    e: &Entry,
    layouts: &TypeLayouts,
    prov: &Provenance,
    candidates: &BTreeSet<InstId>,
    m: &mut Meter,
) -> Result<Fragments, Error> {
    let none = || Fragments {
        held: BTreeSet::new(),
        values: Vec::new(),
    };
    let types = e.module.types();
    let Some(body) = e.module.func(e.func).and_then(|f| f.body.as_ref()) else {
        return Ok(none());
    };
    if candidates.is_empty() && !body.insts.iter().any(|(_, i)| is_matrix(types, i.ty)) {
        return Ok(none());
    }
    let mut f = Fixpoint {
        e,
        prov,
        values: BTreeMap::new(),
        users: BTreeMap::new(),
        stores: BTreeMap::new(),
        readers: BTreeMap::new(),
        held: BTreeSet::new(),
        low: BTreeSet::new(),
        charged: 0,
    };
    let out = f.run(layouts, candidates, m);
    let Fixpoint {
        held,
        values,
        charged,
        ..
    } = f;
    m.release(charged);
    out?;
    let values: Vec<InstId> = values
        .into_iter()
        .filter_map(|(id, fragment)| fragment.then_some(id))
        .collect();
    m.charge(values.len() * std::mem::size_of::<InstId>())?;
    Ok(Fragments { held, values })
}

impl Fixpoint<'_> {
    fn charge(&mut self, bytes: usize, m: &mut Meter) -> Result<(), Error> {
        m.charge(bytes)?;
        self.charged = self.charged.saturating_add(bytes);
        Ok(())
    }

    /// The roots and offset shape of pointer `v`.
    fn fact(&self, v: Value) -> Option<(&[Root], provenance::Shape)> {
        let id = self.prov.values.id(v)?;
        let at = self
            .prov
            .facts
            .binary_search_by_key(&id, |(k, _)| *k)
            .ok()?;
        let (_, f) = self.prov.facts.get(at)?;
        Some((self.prov.sets.get(f.roots), f.shape))
    }

    /// The candidates among `roots`.
    fn candidates_in(roots: &[Root], candidates: &BTreeSet<InstId>) -> Vec<InstId> {
        roots
            .iter()
            .filter_map(|r| match r {
                Root::Alloca(a) if candidates.contains(a) => Some(*a),
                _ => None,
            })
            .collect()
    }

    /// Whether an access of type `a` through pointer `ptr` reaches an element of a matrix
    /// member of candidate `al`, past the held ones: an access of elements 0 and 1, or of one
    /// of them, reaches what the fragment holds.
    fn enters(&self, layouts: &TypeLayouts, al: InstId, ptr: Value, a: TypeId) -> bool {
        let e = self.e;
        let Some(body) = e.module.func(e.func).and_then(|f| f.body.as_ref()) else {
            return true;
        };
        let Some(InstKind::Alloca { ty, count, .. }) = body.insts.get(al).map(|i| &i.kind) else {
            return true;
        };
        let count = match count {
            Value::Const(c) => match e.module.consts().get(*c).map(|k| &k.value) {
                Some(Const::Int(n)) => *n,
                _ => return true,
            },
            _ => return true,
        };
        let shape = match self.fact(ptr).map(|(_, s)| s) {
            Some(provenance::Shape::Exact(c)) => alloca::Shape::Exact(c),
            Some(provenance::Shape::Strided { base, strides }) => {
                alloca::Shape::Strided { base, strides }
            }
            _ => return true,
        };
        if alloca::fragment_path(e, layouts, *ty, count, shape, a).is_some() {
            return false;
        }
        match alloca::path(e, layouts, *ty, count, shape, a) {
            Some(p) => enters_matrix(e.module.types(), *ty, &p.steps),
            None => true,
        }
    }

    fn run(
        &mut self,
        layouts: &TypeLayouts,
        candidates: &BTreeSet<InstId>,
        m: &mut Meter,
    ) -> Result<(), Error> {
        let e = self.e;
        let types = e.module.types();
        let Some(body) = e.module.func(e.func).and_then(|f| f.body.as_ref()) else {
            return Ok(());
        };
        let mut lanes_reached: BTreeSet<InstId> = BTreeSet::new();
        let mut work: Vec<InstId> = Vec::new();
        // The instruction indices of `insertelement`s into `<64 x T>` values.
        let mut indices: Vec<InstId> = Vec::new();
        for (id, inst) in body.insts.iter() {
            m.tick(1)?;
            // An access into a candidate: one reaching a matrix's element disqualifies it; a
            // matrix stored into it, or loaded from it, is recorded.
            let access = match &inst.kind {
                InstKind::Load { ptr, .. } => Some((*ptr, inst.ty, None)),
                InstKind::Store { ptr, val, .. } => e
                    .module
                    .value_ty(e.func, *val)
                    .map(|t| (*ptr, t, Some(*val))),
                _ => None,
            };
            if let Some((ptr, a, val)) = access {
                let into = match self.fact(ptr) {
                    Some((roots, _)) => Self::candidates_in(roots, candidates),
                    None => Vec::new(),
                };
                for &al in &into {
                    if self.enters(layouts, al, ptr, a) && !lanes_reached.contains(&al) {
                        self.charge(map_entry_bytes::<InstId, ()>(), m)?;
                        lanes_reached.insert(al);
                    }
                }
                if is_matrix(types, a) && !into.is_empty() {
                    match val {
                        Some(v) => {
                            if let Value::Inst(i) = v {
                                push_to(&mut self.users, i, id, &mut self.charged, m)?;
                            }
                            self.charge(
                                map_entry_bytes::<InstId, (Value, Vec<InstId>)>()
                                    + into.len() * std::mem::size_of::<InstId>(),
                                m,
                            )?;
                            self.stores.insert(id, (v, into));
                            grow_working(&mut work, 1, &mut self.charged, m)?;
                            work.push(id);
                        }
                        None => {
                            for al in into {
                                push_to(&mut self.readers, al, id, &mut self.charged, m)?;
                            }
                        }
                    }
                }
            }
            if !is_matrix(types, inst.ty) {
                continue;
            }
            self.charge(map_entry_bytes::<InstId, bool>(), m)?;
            self.values.insert(id, true);
            grow_working(&mut work, 1, &mut self.charged, m)?;
            work.push(id);
            if let InstKind::InsertElement {
                idx: Value::Inst(i),
                ..
            } = &inst.kind
            {
                grow_working(&mut indices, 1, &mut self.charged, m)?;
                indices.push(*i);
            }
            let operands: Vec<Value> = match &inst.kind {
                InstKind::InsertElement { vec, .. } => vec![*vec],
                InstKind::ShuffleVector { a, b, .. } => vec![*a, *b],
                InstKind::Select { t, f, .. } => vec![*t, *f],
                InstKind::Phi { incoming, .. } => incoming.iter().map(|(v, _)| *v).collect(),
                _ => Vec::new(),
            };
            for v in operands {
                if let Value::Inst(i) = v {
                    push_to(&mut self.users, i, id, &mut self.charged, m)?;
                }
            }
        }
        let held: BTreeSet<InstId> = candidates.difference(&lanes_reached).copied().collect();
        self.charge(held.len() * map_entry_bytes::<InstId, ()>(), m)?;
        self.held = held;
        self.low_indices(body, indices, m)?;
        while let Some(id) = work.pop() {
            m.tick(1)?;
            if let Some((v, _)) = self.stores.get(&id) {
                if self.fragment(*v, m)? {
                    continue;
                }
                let Fixpoint {
                    stores,
                    held,
                    readers,
                    charged,
                    ..
                } = &mut *self;
                for al in stores.get(&id).into_iter().flat_map(|(_, into)| into) {
                    if held.remove(al) {
                        let reads = readers.get(al).map_or(&[][..], Vec::as_slice);
                        grow_working(&mut work, reads.len(), charged, m)?;
                        work.extend_from_slice(reads);
                    }
                }
                continue;
            }
            if self.values.get(&id) != Some(&true) || self.rule(id, m)? {
                continue;
            }
            self.values.insert(id, false);
            let Fixpoint { users, charged, .. } = &mut *self;
            let next = users.get(&id).map_or(&[][..], Vec::as_slice);
            grow_working(&mut work, next.len(), charged, m)?;
            work.extend_from_slice(next);
        }
        Ok(())
    }

    /// Whether every value of integer `v` is 0 or 1, given the instructions `low` holds as such:
    /// an `i1` value, the constant 0 or 1, or an instruction of `low`.
    fn low(&self, v: Value, low: &BTreeSet<InstId>) -> bool {
        let e = self.e;
        if matches!(
            e.module
                .value_ty(e.func, v)
                .and_then(|t| e.module.types().get(t)),
            Some(Type::Int(1))
        ) {
            return true;
        }
        match v {
            Value::Const(c) => matches!(
                e.module.consts().get(c).map(|k| &k.value),
                Some(Const::Int(0 | 1) | Const::Null)
            ),
            Value::Inst(i) => low.contains(&i),
            Value::Arg(_) => false,
        }
    }

    /// The operands through which instruction `inst` passes a value of 0 or 1 on: a `phi`'s
    /// incoming values, a `select`'s arms, a `trunc`'s, `zext`'s or `sext`'s operand, and an
    /// `and`'s operands.
    fn passed(inst: &InstKind) -> Vec<Value> {
        match inst {
            InstKind::Phi { incoming, .. } => incoming.iter().map(|(v, _)| *v).collect(),
            InstKind::Select { t, f, .. } => vec![*t, *f],
            InstKind::Cast {
                op: CastOp::Trunc | CastOp::ZExt | CastOp::SExt,
                v,
            } => vec![*v],
            InstKind::BinOp {
                op: BinOp::And,
                a,
                b,
                ..
            } => vec![*a, *b],
            _ => Vec::new(),
        }
    }

    /// Whether instruction `inst` (of type `ty`) yields only 0 or 1 when the instructions of
    /// `low` do (the module docs' index rule).
    fn low_rule(&self, inst: &InstKind, ty: TypeId, low: &BTreeSet<InstId>) -> bool {
        let e = self.e;
        let types = e.module.types();
        if matches!(types.get(ty), Some(Type::Int(1))) {
            return true;
        }
        match inst {
            InstKind::Phi { incoming, .. } => incoming.iter().all(|(v, _)| self.low(*v, low)),
            InstKind::Select { t, f, .. } => self.low(*t, low) && self.low(*f, low),
            InstKind::Cast {
                op: CastOp::Trunc | CastOp::ZExt,
                v,
            } => self.low(*v, low),
            // A `sext` of an `i1` 1 is all ones.
            InstKind::Cast {
                op: CastOp::SExt,
                v,
            } => {
                self.low(*v, low)
                    && !matches!(
                        e.module.value_ty(e.func, *v).and_then(|t| types.get(t)),
                        Some(Type::Int(1))
                    )
            }
            InstKind::BinOp {
                op: BinOp::And,
                a,
                b,
                ..
            } => self.low(*a, low) || self.low(*b, low),
            _ => false,
        }
    }

    /// Set `Fixpoint::low`: of the instructions the `insertelement` indices `indices` reach
    /// through `Fixpoint::passed` operands, those whose every value is 0 or 1. A greatest
    /// fixpoint: every reached instruction starts low, one that fails `Fixpoint::low_rule`
    /// stops being low and its users are visited again. Each instruction is reached once and
    /// removed at most once, so the work is linear in the reached instructions and their
    /// operands.
    fn low_indices(
        &mut self,
        body: &Body,
        mut work: Vec<InstId>,
        m: &mut Meter,
    ) -> Result<(), Error> {
        let mut users: BTreeMap<InstId, Vec<InstId>> = BTreeMap::new();
        let mut low: BTreeSet<InstId> = BTreeSet::new();
        while let Some(id) = work.pop() {
            m.tick(1)?;
            let Some(inst) = body.insts.get(id) else {
                continue;
            };
            if low.contains(&id) {
                continue;
            }
            self.charge(map_entry_bytes::<InstId, ()>(), m)?;
            low.insert(id);
            for v in Self::passed(&inst.kind) {
                m.tick(1)?;
                if let Value::Inst(o) = v {
                    push_to(&mut users, o, id, &mut self.charged, m)?;
                    grow_working(&mut work, 1, &mut self.charged, m)?;
                    work.push(o);
                }
            }
        }
        let mut work: Vec<InstId> = Vec::new();
        grow_working(&mut work, low.len(), &mut self.charged, m)?;
        work.extend(low.iter().copied());
        while let Some(id) = work.pop() {
            m.tick(1)?;
            let Some(inst) = body.insts.get(id) else {
                continue;
            };
            if !low.contains(&id) || self.low_rule(&inst.kind, inst.ty, &low) {
                continue;
            }
            low.remove(&id);
            let next = users.get(&id).map_or(&[][..], Vec::as_slice);
            grow_working(&mut work, next.len(), &mut self.charged, m)?;
            work.extend_from_slice(next);
        }
        self.low = low;
        Ok(())
    }

    /// Whether `v` is a fragment.
    fn fragment(&self, v: Value, m: &mut Meter) -> Result<bool, Error> {
        Ok(match v {
            Value::Inst(i) => self.values.get(&i) == Some(&true),
            Value::Const(c) => {
                let mut all = true;
                for j in HELD..LANES {
                    m.tick(1)?;
                    all &= self.const_zero(c, j);
                }
                all
            }
            Value::Arg(_) => false,
        })
    }

    /// Whether element `j` of constant `c` is zero or undefined.
    fn const_zero(&self, c: ConstId, j: u32) -> bool {
        let consts = self.e.module.consts();
        match consts.get(c).map(|k| &k.value) {
            Some(Const::Null | Const::Undef | Const::Poison) => true,
            Some(Const::Aggregate(elems)) => matches!(
                elems
                    .get(j as usize)
                    .and_then(|x| consts.get(*x))
                    .map(|k| &k.value),
                Some(Const::Int(0) | Const::Float(0) | Const::Null | Const::Undef | Const::Poison)
            ),
            _ => false,
        }
    }

    /// Whether element `j` of operand `v` is known zero or undefined.
    fn zero(&self, v: Value, j: u32) -> bool {
        match v {
            Value::Const(c) => self.const_zero(c, j),
            Value::Inst(i) => j >= HELD && self.values.get(&i) == Some(&true),
            Value::Arg(_) => false,
        }
    }

    /// Whether `<64 x T>` instruction `id` is a fragment, its operands as they stand.
    fn rule(&self, id: InstId, m: &mut Meter) -> Result<bool, Error> {
        let e = self.e;
        let types = e.module.types();
        let Some(inst) = e
            .module
            .func(e.func)
            .and_then(|f| f.body.as_ref())
            .and_then(|b| b.insts.get(id))
        else {
            return Ok(false);
        };
        Ok(match &inst.kind {
            InstKind::Call(call) => matches!(
                call.callee,
                Callee::Air {
                    op: AirIntrinsic::SimdMatrix(
                        SimdMatrix::Load8x8 { .. }
                            | SimdMatrix::InitDiag8x8 { .. }
                            | SimdMatrix::InitFilled8x8 { .. }
                            | SimdMatrix::MultiplyAccumulate8x8 { .. }
                    ),
                    ..
                }
            ),
            InstKind::InsertElement { vec, idx, .. } => {
                self.low(*idx, &self.low) && self.fragment(*vec, m)?
            }
            InstKind::ShuffleVector { a, b, mask } => {
                let width = match e.module.value_ty(e.func, *a).and_then(|t| types.get(t)) {
                    Some(Type::Vector { len, .. }) => *len,
                    _ => return Ok(false),
                };
                let mut all = mask.len() == LANES as usize;
                for lane in mask.iter().skip(HELD as usize) {
                    m.tick(1)?;
                    all &= match lane {
                        None => true,
                        Some(j) if *j < width => self.zero(*a, *j),
                        Some(j) => self.zero(*b, *j - width),
                    };
                }
                all
            }
            InstKind::Select { t, f, .. } => self.fragment(*t, m)? && self.fragment(*f, m)?,
            InstKind::Phi { incoming, .. } => {
                let mut all = true;
                for (v, _) in incoming.iter() {
                    all &= self.fragment(*v, m)?;
                }
                all
            }
            InstKind::Load {
                ptr,
                volatile: false,
                atomic: None,
                ..
            } => match self.fact(*ptr) {
                Some((roots, _)) => {
                    !roots.is_empty()
                        && roots
                            .iter()
                            .all(|r| matches!(r, Root::Alloca(a) if self.held.contains(a)))
                }
                None => false,
            },
            _ => false,
        })
    }
}
