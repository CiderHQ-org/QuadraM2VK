//! Simdgroup and quad operations: `air.{simd,quad}_<op>[.<s|u>].<type>`, plus `air.is_uniform`,
//! `air.get_simdgroup_size`, and the ballot, `is_first`, and active-mask queries.
//!
//! A value operand of integer type spells its sign (`.s.`/`.u.`); a float one spells none. Lane
//! indices, deltas, masks, and widths are `i16` operands. One declarative table ([`OPS`]) lists
//! every operation with its operand shape and the scopes (simdgroup, quad) it exists in.

use super::sig::{Operand, Signature};
use super::ty::{Elem, NumTy, Sign};

/// The execution group an operation spans.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GroupScope {
    /// `simd_`: the simdgroup.
    Simdgroup,
    /// `quad_`: a quad of four threads.
    Quad,
}

/// A group operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GroupOp {
    /// `broadcast`: the value in the lane the `i16` operand names, the same lane for every thread.
    Broadcast,
    /// `broadcast_first`: the value in the first active lane.
    BroadcastFirst,
    /// `shuffle`: the value in the lane the `i16` operand names, per thread.
    Shuffle,
    /// `shuffle_up`: the value in the lane `delta` below this one.
    ShuffleUp,
    /// `shuffle_down`: the value in the lane `delta` above this one.
    ShuffleDown,
    /// `shuffle_xor`: the value in the lane whose index is this one's XOR the `i16` mask.
    ShuffleXor,
    /// `shuffle_rotate_down`: the value in the lane `delta` above this one, wrapping around.
    ShuffleRotateDown,
    /// `shuffle_rotate_up`: the value in the lane `delta` below this one, wrapping around.
    ShuffleRotateUp,
    /// `shuffle_and_fill_up`: `shuffle_up` within segments of `modulo` lanes, the lowest lanes
    /// filled from `fill` (a quad's `modulo` is 4 where the source spells none).
    ShuffleAndFillUp,
    /// `shuffle_and_fill_down`: `shuffle_down` within segments of `modulo` lanes, the highest
    /// lanes filled from `fill`.
    ShuffleAndFillDown,
    /// `sum`: the sum over the active lanes.
    Sum,
    /// `min`: the minimum over the active lanes.
    Min,
    /// `max`: the maximum over the active lanes.
    Max,
    /// `and`: the bitwise AND over the active lanes.
    And,
    /// `or`: the bitwise OR over the active lanes.
    Or,
    /// `xor`: the bitwise XOR over the active lanes.
    Xor,
    /// `product`: the product over the active lanes.
    Product,
    /// `prefix_exclusive_sum`: the sum over the active lanes below this one.
    PrefixExclusiveSum,
    /// `prefix_inclusive_sum`: the sum over the active lanes up to and including this one.
    PrefixInclusiveSum,
    /// `prefix_exclusive_product`: the product over the active lanes below this one (1 for the
    /// first).
    PrefixExclusiveProduct,
    /// `prefix_inclusive_product`: the product over the active lanes up to and including this
    /// one.
    PrefixInclusiveProduct,
    /// `all`: whether the `i1` operand is true in every active lane.
    All,
    /// `any`: whether the `i1` operand is true in some active lane.
    Any,
    /// `ballot`: the mask of the active lanes whose `i1` operand is true.
    Ballot,
    /// `is_first`: whether this is the first active lane.
    IsFirst,
    /// `active_threads_mask`: the mask of the active lanes.
    ActiveThreadsMask,
}

impl GroupOp {
    /// Whether the operation's answer depends on which of its group's lanes are active: a
    /// reduction, a scan, a vote, a ballot, the active mask, the first lane, or a broadcast from
    /// it. A shuffle or a broadcast from a named lane reads that lane's value, which a helper
    /// invocation computes alike, so it is not one.
    pub fn reads_active_lanes(self) -> bool {
        use GroupOp as G;
        match self {
            G::BroadcastFirst
            | G::Sum
            | G::Min
            | G::Max
            | G::And
            | G::Or
            | G::Xor
            | G::Product
            | G::PrefixExclusiveSum
            | G::PrefixInclusiveSum
            | G::PrefixExclusiveProduct
            | G::PrefixInclusiveProduct
            | G::All
            | G::Any
            | G::Ballot
            | G::IsFirst
            | G::ActiveThreadsMask => true,
            G::Broadcast
            | G::Shuffle
            | G::ShuffleUp
            | G::ShuffleDown
            | G::ShuffleXor
            | G::ShuffleRotateDown
            | G::ShuffleRotateUp
            | G::ShuffleAndFillUp
            | G::ShuffleAndFillDown => false,
        }
    }
}

/// The operand shape of a table entry, with `T` the spelled value type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Shape {
    /// `T (T)`.
    Reduce,
    /// `T (T, i16)`.
    Lane,
    /// `T (T, T fill, i16 delta, i16 width)`.
    Fill,
    /// `i1 (i1)`: no type token.
    Vote,
    /// `iN (i1)`: the token spells the mask type.
    Ballot,
    /// `i16 (i1)`: no type token (`quad_ballot`).
    QuadBallot,
    /// `i1 ()`: no type token.
    Predicate,
    /// `i16 ()`: no type token (`quad_active_threads_mask`).
    Mask,
    /// `iN ()`: the token spells the mask type (`simd_active_threads_mask`).
    SimdMask,
}

/// Which value types an entry admits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Values {
    /// Integers and floats.
    Any,
    /// Integers only.
    Int,
    /// No value type (the shape fixes the types).
    None,
}

/// Every operation: name stem, variant, shape, admitted values, and scopes. A stem may have one
/// entry per scope when the scopes spell it differently (the ballot and the active mask: a
/// simdgroup's mask is the spelled `i64`, a quad's an unspelled `i16`, as Apple's compiler
/// emits them).
pub(super) const OPS: &[(&str, GroupOp, Shape, Values, &[GroupScope])] = {
    use GroupOp as G;
    use GroupScope::{Quad as Q, Simdgroup as S};
    use Shape::*;
    use Values::{Any as A, Int as I, None as N};
    &[
        ("broadcast", G::Broadcast, Lane, A, &[S, Q]),
        ("broadcast_first", G::BroadcastFirst, Reduce, A, &[S, Q]),
        ("shuffle", G::Shuffle, Lane, A, &[S, Q]),
        ("shuffle_up", G::ShuffleUp, Lane, A, &[S, Q]),
        ("shuffle_down", G::ShuffleDown, Lane, A, &[S, Q]),
        ("shuffle_xor", G::ShuffleXor, Lane, A, &[S, Q]),
        (
            "shuffle_rotate_down",
            G::ShuffleRotateDown,
            Lane,
            A,
            &[S, Q],
        ),
        ("shuffle_rotate_up", G::ShuffleRotateUp, Lane, A, &[S, Q]),
        ("shuffle_and_fill_up", G::ShuffleAndFillUp, Fill, A, &[S, Q]),
        (
            "shuffle_and_fill_down",
            G::ShuffleAndFillDown,
            Fill,
            A,
            &[S, Q],
        ),
        ("sum", G::Sum, Reduce, A, &[S, Q]),
        ("min", G::Min, Reduce, A, &[S, Q]),
        ("max", G::Max, Reduce, A, &[S, Q]),
        ("and", G::And, Reduce, I, &[S, Q]),
        ("or", G::Or, Reduce, I, &[S, Q]),
        ("xor", G::Xor, Reduce, I, &[S, Q]),
        ("product", G::Product, Reduce, A, &[S, Q]),
        (
            "prefix_exclusive_sum",
            G::PrefixExclusiveSum,
            Reduce,
            A,
            &[S, Q],
        ),
        (
            "prefix_inclusive_sum",
            G::PrefixInclusiveSum,
            Reduce,
            A,
            &[S, Q],
        ),
        (
            "prefix_exclusive_product",
            G::PrefixExclusiveProduct,
            Reduce,
            A,
            &[S, Q],
        ),
        (
            "prefix_inclusive_product",
            G::PrefixInclusiveProduct,
            Reduce,
            A,
            &[S, Q],
        ),
        ("all", G::All, Vote, N, &[S, Q]),
        ("any", G::Any, Vote, N, &[S, Q]),
        ("ballot", G::Ballot, Ballot, N, &[S]),
        ("ballot", G::Ballot, QuadBallot, N, &[Q]),
        ("is_first", G::IsFirst, Predicate, N, &[S, Q]),
        ("active_threads_mask", G::ActiveThreadsMask, Mask, N, &[Q]),
        (
            "active_threads_mask",
            G::ActiveThreadsMask,
            SimdMask,
            N,
            &[S],
        ),
    ]
};

/// A decoded group operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Group {
    /// The group the operation spans: the name's `simd_` or `quad_` prefix.
    pub scope: GroupScope,
    /// The operation.
    pub op: GroupOp,
    /// The value type (for a ballot, the mask type); `None` when the name spells none.
    pub ty: Option<NumTy>,
    /// The sign token of an integer value.
    pub sign: Option<Sign>,
}

impl Group {
    /// Parse the part of a name after `air.`.
    pub(super) fn parse(rest: &str) -> Option<Group> {
        let (scope, rest) = if let Some(r) = rest.strip_prefix("simd_") {
            (GroupScope::Simdgroup, r)
        } else {
            (GroupScope::Quad, rest.strip_prefix("quad_")?)
        };
        let mut tokens = rest.split('.');
        let stem = tokens.next()?;
        let &(_, op, shape, values, _) = OPS
            .iter()
            .find(|(s, _, _, _, scopes)| *s == stem && scopes.contains(&scope))?;
        let rest: Vec<&str> = tokens.collect();
        let spelled = matches!(shape, Shape::Ballot | Shape::SimdMask);
        let (sign, ty) = match (values, shape, &rest[..]) {
            (Values::None, _, [ty]) if spelled => {
                let ty = NumTy::parse(ty)?;
                (ty.elem == Elem::I64 && !ty.is_vector()).then_some(())?;
                (None, Some(ty))
            }
            (Values::None, _, []) if !spelled => (None, None),
            (Values::Any | Values::Int, _, [ty]) => {
                let ty = NumTy::parse(ty)?;
                (ty.elem.is_float() && values == Values::Any).then_some(())?;
                (None, Some(ty))
            }
            (Values::Any | Values::Int, _, [sign, ty]) => {
                let ty = NumTy::parse(ty)?;
                ty.elem.is_int().then_some(())?;
                (Some(Sign::parse(sign)?), Some(ty))
            }
            _ => return None,
        };
        if ty.is_some_and(|t| t.elem == Elem::BF16) {
            return None;
        }
        Some(Group {
            scope,
            op,
            ty,
            sign,
        })
    }

    fn shape(&self) -> Shape {
        OPS.iter()
            .find(|(_, op, _, _, scopes)| *op == self.op && scopes.contains(&self.scope))
            .map_or(Shape::Reduce, |&(_, _, shape, ..)| shape)
    }

    /// The typed signature.
    pub fn signature(&self) -> Signature {
        let bool1 = Operand::Num(NumTy::scalar(Elem::I1));
        let i16s = Operand::Num(NumTy::scalar(Elem::I16));
        let t = self.ty.map_or(Operand::Void, Operand::Num);
        match self.shape() {
            Shape::Reduce => Signature::new(t, &[t]),
            Shape::Lane => Signature::new(t, &[t, i16s]),
            Shape::Fill => Signature::new(t, &[t, t, i16s, i16s]),
            Shape::Vote => Signature::new(bool1, &[bool1]),
            Shape::Ballot => Signature::new(t, &[bool1]),
            Shape::QuadBallot => Signature::new(i16s, &[bool1]),
            Shape::Predicate => Signature::new(bool1, &[]),
            Shape::Mask => Signature::new(i16s, &[]),
            Shape::SimdMask => Signature::new(t, &[]),
        }
    }
}

/// `air.is_uniform.<type>`: `i1 (T)`, whether a value is the same across the simdgroup: Metal's
/// `uniform<T>` assertion over any scalar or vector `T`, an integer, `i1`, or a float (`xcrun
/// metal` 32023.921 spells `uniform<half>` `air.is_uniform.f16`, and `bfloat2`'s
/// `air.is_uniform.v2bf16`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct IsUniform {
    /// The value type the name spells.
    pub ty: NumTy,
}

impl IsUniform {
    pub(super) fn parse(rest: &str) -> Option<IsUniform> {
        let ty = NumTy::parse(rest.strip_prefix("is_uniform.")?)?;
        Some(IsUniform { ty })
    }

    /// The typed signature.
    pub fn signature(&self) -> Signature {
        Signature::new(
            Operand::Num(NumTy::scalar(Elem::I1)),
            &[Operand::Num(self.ty)],
        )
    }
}

/// `air.get_simdgroup_size.<i16|i32>`: `T ()`, the execution width.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SimdgroupSize {
    /// The result element the name spells: `I16` or `I32`.
    pub ty: Elem,
}

impl SimdgroupSize {
    pub(super) fn parse(rest: &str) -> Option<SimdgroupSize> {
        let ty = NumTy::parse(rest.strip_prefix("get_simdgroup_size.")?)?;
        (matches!(ty.elem, Elem::I16 | Elem::I32) && !ty.is_vector())
            .then_some(SimdgroupSize { ty: ty.elem })
    }

    /// The typed signature.
    pub fn signature(&self) -> Signature {
        Signature::new(Operand::Num(NumTy::scalar(self.ty)), &[])
    }
}
