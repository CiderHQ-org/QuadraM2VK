//! Atomics on memory: `air.atomic.{global,local,private}.<op>[.<s|u>].<type>`, and coherent plain
//! accesses `air.{load,store}.{device,system}_coherent[.volatile].<type>.p<space><...>`.
//!
//! Operand layout (Metal's atomic functions): read-modify-write `T (ptr, T, i32 order, i32 scope,
//! i1 volatile)`; load `T (ptr, i32 order, i32 scope, i1 volatile)`; store `void (ptr, T, i32
//! order, i32 scope, i1 volatile)`; weak compare-exchange `T (ptr, ptr expected, T desired, i32
//! success order, i32 failure order, i32 scope, i1 volatile)`, where `expected` is a thread
//! pointer the old value is written back through. Orders and scopes are immediates; a value
//! outside the enumerations below is not a known intrinsic (decode keeps the call as unknown)
//! rather than a guessed one.
//!
//! Metal's 64-bit `atomic_min_explicit` and `atomic_max_explicit` on a `device atomic_ulong`
//! return nothing (MSL §6.16.4.6), so `air.atomic.global.{min,max}.u.i64` may also be declared
//! `void (ptr addrspace(1), i64, i32 order, i32 scope, i1 volatile)`: the same operation with the
//! same immediates, without the old value (`Atomic::fetch` is `false`).
//!
//! **AIR 2.9** (deployment target macOS 27): every one of these declarations takes one more `i32`
//! immediate between the scope and the volatile flag, the call's Metal `mem_flags` (`T (ptr, i32
//! order, i32 scope, i32 flags, i1 volatile)` for a load, and so on). Metal's headers pass a
//! `mem_flags` to every atomic builtin in every language version (`__METAL_MEMORY_FLAGS_NONE__`
//! with a relaxed order before MSL 4.1); AIR 2.8 drops it. Which form a declaration has is read
//! from its type: [`Atomic::flags`] is `None` for the AIR 2.8 form and the operand's mask for the
//! AIR 2.9 one, which MSL 4.1's `_explicit(..., order, flags)` functions (MSL §6.16.4) and its
//! non-explicit ones (`seq_cst` with every flag) fill.
//!
//! **Thread memory** (MSL 4.1, so AIR 2.9 only): `air.atomic.private.<op>` is an atomic on a
//! `thread` object through a thread pointer, in the AIR 2.9 layout, with Apple's compiler passing
//! scope `thread_scope_thread`. MSL 4.1 declares thread atomics of 32-bit integers and `float`
//! (`bool` is an `i32`), not of 64-bit integers, and `float` ones without a minimum or maximum; no
//! other private name is decoded.

use super::sig::{ImmKind, Immediates, Operand, Signature};
use super::sync::MemFlags;
use super::ty::{Elem, NumTy, Sign};
use crate::error::IntrinsicError;
use crate::ir::AddrSpace;

/// The memory an atomic names: `global` (device), `local` (threadgroup), or `private` (thread).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AtomicSpace {
    /// `global`: device memory.
    Global,
    /// `local`: threadgroup memory.
    Local,
    /// `private`: thread memory (AIR 2.9 only).
    Private,
}

impl AtomicSpace {
    /// The pointer operand's address space.
    pub const fn space(self) -> AddrSpace {
        match self {
            AtomicSpace::Global => AddrSpace::Device,
            AtomicSpace::Local => AddrSpace::Threadgroup,
            AtomicSpace::Private => AddrSpace::Thread,
        }
    }
}

/// The operation of an [`Atomic`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AtomicOp {
    /// `add`: atomic addition.
    Add,
    /// `sub`: atomic subtraction.
    Sub,
    /// `min`: atomic minimum.
    Min,
    /// `max`: atomic maximum.
    Max,
    /// `and`: atomic bitwise AND.
    And,
    /// `or`: atomic bitwise OR.
    Or,
    /// `xor`: atomic bitwise XOR.
    Xor,
    /// `xchg`: atomic exchange.
    Xchg,
    /// `load`: atomic load.
    Load,
    /// `store`: atomic store.
    Store,
    /// `cmpxchg.weak`: weak compare-exchange.
    CmpXchgWeak,
}

/// Every operation with its name tokens.
pub(super) const OPS: &[(&str, AtomicOp)] = &[
    ("add", AtomicOp::Add),
    ("sub", AtomicOp::Sub),
    ("min", AtomicOp::Min),
    ("max", AtomicOp::Max),
    ("and", AtomicOp::And),
    ("or", AtomicOp::Or),
    ("xor", AtomicOp::Xor),
    ("xchg", AtomicOp::Xchg),
    ("load", AtomicOp::Load),
    ("store", AtomicOp::Store),
    ("cmpxchg.weak", AtomicOp::CmpXchgWeak),
];

/// A C11 memory order, as Metal's headers pass it (`__ATOMIC_*`: relaxed 0, acquire 2, release 3,
/// acq_rel 4, seq_cst 5). `consume` (1) has no Metal spelling and is not decoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MemoryOrder {
    /// `memory_order_relaxed` (0).
    Relaxed,
    /// `memory_order_acquire` (2).
    Acquire,
    /// `memory_order_release` (3).
    Release,
    /// `memory_order_acq_rel` (4).
    AcqRel,
    /// `memory_order_seq_cst` (5).
    SeqCst,
}

impl MemoryOrder {
    /// The order numbered `v`.
    pub fn from_raw(v: u64) -> Option<MemoryOrder> {
        Some(match v {
            0 => MemoryOrder::Relaxed,
            2 => MemoryOrder::Acquire,
            3 => MemoryOrder::Release,
            4 => MemoryOrder::AcqRel,
            5 => MemoryOrder::SeqCst,
            _ => return None,
        })
    }
}

/// A Metal `thread_scope` (thread 0, threadgroup 1, device 2, simdgroup 4), plus the
/// OpenCL-derived all-devices scope (3) older front ends still pass to `air.atomic.fence`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ThreadScope {
    /// `thread_scope_thread` (0).
    Thread,
    /// `thread_scope_threadgroup` (1).
    Threadgroup,
    /// `thread_scope_device` (2).
    Device,
    /// The OpenCL-derived all-devices scope (3), which only older front ends pass, to
    /// `air.atomic.fence`.
    AllDevices,
    /// `thread_scope_simdgroup` (4).
    Simdgroup,
}

impl ThreadScope {
    /// The scope numbered `v`.
    pub fn from_raw(v: u64) -> Option<ThreadScope> {
        Some(match v {
            0 => ThreadScope::Thread,
            1 => ThreadScope::Threadgroup,
            2 => ThreadScope::Device,
            3 => ThreadScope::AllDevices,
            4 => ThreadScope::Simdgroup,
            _ => return None,
        })
    }
}

/// A decoded memory atomic.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Atomic {
    /// The memory the name spells, which gives the pointer operand's address space.
    pub space: AtomicSpace,
    /// The operation.
    pub op: AtomicOp,
    /// The sign token, when the name spells one (required for integer `min`/`max`).
    pub sign: Option<Sign>,
    /// The value type: `i32`, `i64`, or `f32` (`f32` only for `add`/`sub`/`min`/`max`/`xchg`/
    /// `load`/`store`).
    pub ty: NumTy,
    /// The memory order immediate (a compare-exchange's success order).
    pub order: MemoryOrder,
    /// The failure order of a compare-exchange.
    pub failure: Option<MemoryOrder>,
    /// The `thread_scope` immediate.
    pub scope: ThreadScope,
    /// The object's `volatile` qualifier (Metal's atomic functions take volatile objects).
    pub volatile: bool,
    /// Whether the call returns the value it read from memory: a load's value, the old value of
    /// a read-modify-write or compare-exchange. A store reads nothing, and Metal's result-less
    /// 64-bit `atomic_{min,max}_explicit` (`air.atomic.global.{min,max}.u.i64` declared `void`)
    /// returns nothing; its memory operation is the fetching form's.
    pub fetch: bool,
    /// The memories the order applies to: the `mem_flags` immediate of the AIR 2.9 form, `None`
    /// for the AIR 2.8 form, which has no such operand.
    pub flags: Option<MemFlags>,
}

impl Atomic {
    /// Parse the part of a name after `air.atomic.`.
    pub(super) fn parse(rest: &str) -> Option<Atomic> {
        let (space, rest) = match rest.split_once('.')? {
            ("global", r) => (AtomicSpace::Global, r),
            ("local", r) => (AtomicSpace::Local, r),
            ("private", r) => (AtomicSpace::Private, r),
            _ => return None,
        };
        let private = space == AtomicSpace::Private;
        let (op, rest) = OPS.iter().find_map(|&(name, op)| {
            let r = rest.strip_prefix(name)?.strip_prefix('.')?;
            Some((op, r))
        })?;
        let (sign, ty) = match rest.split_once('.') {
            Some((s, ty)) => (Some(Sign::parse(s)?), ty),
            None => (None, rest),
        };
        let ty = NumTy::parse(ty)?;
        let scalar_ok = match ty.elem {
            Elem::I32 => true,
            Elem::I64 => !private,
            Elem::F32 => {
                sign.is_none()
                    && match op {
                        AtomicOp::Add
                        | AtomicOp::Sub
                        | AtomicOp::Xchg
                        | AtomicOp::Load
                        | AtomicOp::Store => true,
                        AtomicOp::Min | AtomicOp::Max => !private,
                        _ => false,
                    }
            }
            _ => false,
        };
        let int_min_max = ty.elem.is_int() && matches!(op, AtomicOp::Min | AtomicOp::Max);
        if ty.is_vector() || !scalar_ok || (int_min_max && sign.is_none()) {
            return None;
        }
        if sign.is_some() && matches!(op, AtomicOp::Xchg | AtomicOp::CmpXchgWeak) {
            return None;
        }
        Some(Atomic {
            space,
            op,
            sign,
            ty,
            order: MemoryOrder::Relaxed,
            failure: (op == AtomicOp::CmpXchgWeak).then_some(MemoryOrder::Relaxed),
            scope: ThreadScope::Device,
            volatile: false,
            fetch: op != AtomicOp::Store,
            // A thread atomic exists in the AIR 2.9 form only.
            flags: private.then(MemFlags::default),
        })
    }

    /// The next declaration the same name may have: the AIR 2.9 form of an AIR 2.8 one (its
    /// `mem_flags` operand), then Metal's result-less 64-bit `atomic_{min,max}_explicit` on a
    /// `device atomic_ulong` (MSL §6.16.4.6) in either form. No other operation, type, sign, or
    /// memory is declared without its result (none is observed), so each has the two forms of its
    /// fetching declaration; a thread atomic has the AIR 2.9 form alone.
    pub(super) fn alternative(self) -> Option<Atomic> {
        if self.flags.is_none() {
            return Some(Atomic {
                flags: Some(MemFlags::default()),
                ..self
            });
        }
        let ulong_min_max = self.space == AtomicSpace::Global
            && matches!(self.op, AtomicOp::Min | AtomicOp::Max)
            && self.sign == Some(Sign::Unsigned)
            && self.ty == NumTy::scalar(Elem::I64);
        (ulong_min_max && self.fetch).then_some(Atomic {
            fetch: false,
            flags: None,
            ..self
        })
    }

    /// The typed signature: the AIR 2.9 form inserts the `mem_flags` immediate before the
    /// volatile flag.
    pub fn signature(&self) -> Signature {
        let ptr = Operand::Ptr(self.space.space());
        let t = Operand::Num(self.ty);
        let i32s = Operand::Imm(ImmKind::I32);
        let flag = Operand::Imm(ImmKind::Bool);
        let (head, ret): (&[Operand], Operand) = match self.op {
            AtomicOp::Load => (&[ptr, i32s, i32s], t),
            AtomicOp::Store => (&[ptr, t, i32s, i32s], Operand::Void),
            AtomicOp::CmpXchgWeak => (
                &[ptr, Operand::Ptr(AddrSpace::Thread), t, i32s, i32s, i32s],
                t,
            ),
            _ => (
                &[ptr, t, i32s, i32s],
                if self.fetch { t } else { Operand::Void },
            ),
        };
        let tail: &[Operand] = match self.flags {
            Some(_) => &[i32s, flag],
            None => &[flag],
        };
        // The longest is a compare-exchange's six leading operands and both trailing ones.
        let mut params = [Operand::Void; 8];
        for (slot, p) in params.iter_mut().zip(head.iter().chain(tail)) {
            *slot = *p;
        }
        Signature::new(ret, params.get(..head.len() + tail.len()).unwrap_or(&[]))
    }

    /// Read the call's order, scope, `mem_flags`, and volatile immediates. `Ok(None)`: an order,
    /// scope, or flag outside the enumerations.
    pub(super) fn bind(self, imm: &Immediates<'_>) -> Result<Option<Atomic>, IntrinsicError> {
        let (order, failure, scope) = match self.op {
            AtomicOp::Load => (1, None, 2),
            AtomicOp::CmpXchgWeak => (3, Some(4), 5),
            _ => (2, None, 3),
        };
        // The AIR 2.9 form's `mem_flags` follows the scope; the volatile flag is last.
        let flags_at = self.flags.map(|_| scope + 1);
        let volatile = imm.flag(scope + 1 + usize::from(flags_at.is_some()))?;
        let (order, scope) = (imm.get(order)?, imm.get(scope)?);
        let failure = match failure {
            Some(i) => Some(imm.get(i)?),
            None => None,
        };
        let flags = match flags_at {
            Some(i) => match MemFlags::from_raw(imm.get(i)?) {
                Some(f) => Some(f),
                None => return Ok(None),
            },
            None => None,
        };
        let (Some(order), Some(scope)) =
            (MemoryOrder::from_raw(order), ThreadScope::from_raw(scope))
        else {
            return Ok(None);
        };
        let failure = match failure.map(MemoryOrder::from_raw) {
            None => None,
            Some(Some(f)) => Some(f),
            Some(None) => return Ok(None),
        };
        Ok(Some(Atomic {
            order,
            failure,
            scope,
            volatile,
            flags,
            ..self
        }))
    }
}

/// The coherence a coherent access names.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Coherence {
    Device,
    System,
}

/// Whether a coherent access loads or stores.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AccessKind {
    /// `air.load.*`: a coherent load, `T (ptr)`.
    Load,
    /// `air.store.*`: a coherent store, `void (T, ptr)`.
    Store,
}

/// A plain (non-atomic) access with `coherent(device|system)` semantics.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Coherent {
    /// Whether the access loads or stores.
    pub kind: AccessKind,
    /// `device_coherent` or `system_coherent`, as the name spells.
    pub coherence: Coherence,
    /// `.volatile`: the access is volatile.
    pub volatile: bool,
    /// The value type the name spells.
    pub ty: NumTy,
    /// The pointer operand's address space, the name's `p<space>`.
    pub space: AddrSpace,
}

impl Coherent {
    /// Parse the part of a name after `air.load.` / `air.store.`: `<coherence>[.volatile].<type>.
    /// p<space><pointee>`. The pointee spelling is the mangler's and is not interpreted.
    pub(super) fn parse(kind: AccessKind, rest: &str) -> Option<Coherent> {
        let (coherence, rest) = match rest.split_once('.')? {
            ("device_coherent", r) => (Coherence::Device, r),
            ("system_coherent", r) => (Coherence::System, r),
            _ => return None,
        };
        let (volatile, rest) = match rest.strip_prefix("volatile.") {
            Some(r) => (true, r),
            None => (false, rest),
        };
        let (ty, ptr) = rest.split_once('.')?;
        let digits = ptr.strip_prefix('p')?;
        let n = digits.bytes().take_while(u8::is_ascii_digit).count();
        let space: u32 = digits.get(..n)?.parse().ok()?;
        Some(Coherent {
            kind,
            coherence,
            volatile,
            ty: NumTy::parse(ty)?,
            space: AddrSpace::from_raw(space).ok()?,
        })
    }

    /// `T (ptr)` or `void (T, ptr)`.
    pub fn signature(&self) -> Signature {
        let (t, ptr) = (Operand::Num(self.ty), Operand::Ptr(self.space));
        match self.kind {
            AccessKind::Load => Signature::new(t, &[ptr]),
            AccessKind::Store => Signature::new(Operand::Void, &[t, ptr]),
        }
    }
}
