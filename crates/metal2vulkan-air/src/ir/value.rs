//! Dense AIR value indices.
//!
//! [`ValueId`] is the one index type for AIR values: dense over an `Entry`, covering arguments,
//! instructions (void ones included), and constants. Side tables keyed by value
//! (`MemoryPlan.repr`, `Plan.strategy`, `ValueEnv`) index by it. [`ValueIndex`] is the numbering.

use super::id::{ArgIdx, ConstId, InstId};
use super::inst::Value;

/// A dense index of an AIR value within one `Entry`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ValueId(u32);

impl ValueId {
    /// The value with this dense index.
    pub const fn new(index: u32) -> ValueId {
        ValueId(index)
    }

    /// The value at a `usize` position, if it fits the `u32` index space.
    pub fn from_index(index: usize) -> Option<ValueId> {
        u32::try_from(index).ok().map(ValueId)
    }

    /// The dense index.
    pub const fn index(self) -> usize {
        self.0 as usize
    }

    /// The raw `u32` index.
    pub const fn raw(self) -> u32 {
        self.0
    }
}

/// The numbering of one function's values: arguments first, then every instruction (void ones
/// included) in arena order, then every module constant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ValueIndex {
    args: u32,
    insts: u32,
    consts: u32,
}

impl ValueIndex {
    /// The numbering for `args` arguments, `insts` instructions, and `consts` constants, if the
    /// total fits the `u32` index space.
    pub fn new(args: usize, insts: usize, consts: usize) -> Option<ValueIndex> {
        let total = args.checked_add(insts)?.checked_add(consts)?;
        u32::try_from(total).ok()?;
        Some(ValueIndex {
            args: u32::try_from(args).ok()?,
            insts: u32::try_from(insts).ok()?,
            consts: u32::try_from(consts).ok()?,
        })
    }

    /// The number of ids (every id is below it).
    pub fn len(&self) -> usize {
        self.args as usize + self.insts as usize + self.consts as usize
    }

    /// Whether there is no id at all.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The id of an argument, instruction, or constant, if in range.
    pub fn id(&self, v: Value) -> Option<ValueId> {
        match v {
            Value::Arg(a) if a.raw() < self.args => Some(ValueId(a.raw())),
            Value::Inst(i) if i.raw() < self.insts => Some(ValueId(self.args + i.raw())),
            Value::Const(c) if c.raw() < self.consts => {
                Some(ValueId(self.args + self.insts + c.raw()))
            }
            _ => None,
        }
    }

    /// The id of an instruction (void or not), if in range.
    pub fn inst(&self, i: InstId) -> Option<ValueId> {
        self.id(Value::Inst(i))
    }

    /// The value an id numbers.
    pub fn value(&self, id: ValueId) -> Option<Value> {
        let n = id.raw();
        if n < self.args {
            Some(Value::Arg(ArgIdx::new(n)))
        } else if n - self.args < self.insts {
            Some(Value::Inst(InstId::new(n - self.args)))
        } else if n - self.args - self.insts < self.consts {
            Some(Value::Const(ConstId::new(n - self.args - self.insts)))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn value_ids_round_trip_their_index() {
        let v = ValueId::new(42);
        assert_eq!(v.index(), 42);
        assert_eq!(v.raw(), 42);
        assert_eq!(ValueId::from_index(42), Some(v));
        assert_eq!(
            ValueId::from_index(u32::MAX as usize),
            Some(ValueId::new(u32::MAX))
        );
        if usize::BITS > 32 {
            assert_eq!(ValueId::from_index(u32::MAX as usize + 1), None);
        }
    }

    #[test]
    fn value_index_numbers_args_then_insts_then_consts() {
        let ix = ValueIndex::new(2, 3, 4).unwrap();
        assert_eq!(ix.len(), 9);
        let all = [
            Value::Arg(ArgIdx::new(0)),
            Value::Arg(ArgIdx::new(1)),
            Value::Inst(InstId::new(0)),
            Value::Inst(InstId::new(1)),
            Value::Inst(InstId::new(2)),
            Value::Const(ConstId::new(0)),
            Value::Const(ConstId::new(1)),
            Value::Const(ConstId::new(2)),
            Value::Const(ConstId::new(3)),
        ];
        for (n, v) in all.into_iter().enumerate() {
            let id = ix.id(v).unwrap();
            assert_eq!(id.index(), n);
            assert_eq!(ix.value(id), Some(v));
        }
        assert_eq!(ix.value(ValueId::new(9)), None);
        assert_eq!(ix.id(Value::Arg(ArgIdx::new(2))), None);
        assert_eq!(ix.id(Value::Inst(InstId::new(3))), None);
        assert_eq!(ix.id(Value::Const(ConstId::new(4))), None);
        assert_eq!(ValueIndex::new(u32::MAX as usize, 1, 0), None);
    }
}
