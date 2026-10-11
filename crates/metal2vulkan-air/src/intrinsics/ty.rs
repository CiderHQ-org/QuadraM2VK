//! Numeric type tokens shared by every intrinsic family.
//!
//! AIR and LLVM spell an intrinsic's overload types in its name: `f32`, `v4f16`, `i1`, `bf16`.
//! [`NumTy::parse`] reads one such token; [`NumTy::of`] reads the same shape from an IR type, so a
//! name and its declaration are compared structurally, never as strings.

use crate::ir::{Type, TypeId, TypeTable};

/// The scalar element of an intrinsic's numeric type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Elem {
    /// `i1`: a boolean.
    I1,
    /// `i8`: an 8-bit integer.
    I8,
    /// `i16`: a 16-bit integer.
    I16,
    /// `i32`: a 32-bit integer.
    I32,
    /// `i64`: a 64-bit integer.
    I64,
    /// `f16`: `half`.
    F16,
    /// `bf16`: `bfloat`.
    BF16,
    /// `f32`: `float`.
    F32,
}

impl Elem {
    /// Every element.
    #[cfg(test)]
    pub const ALL: [Elem; 8] = [
        Elem::I1,
        Elem::I8,
        Elem::I16,
        Elem::I32,
        Elem::I64,
        Elem::F16,
        Elem::BF16,
        Elem::F32,
    ];

    /// Whether this is a floating-point element.
    pub const fn is_float(self) -> bool {
        matches!(self, Elem::F16 | Elem::BF16 | Elem::F32)
    }

    /// Whether this is an integer element of at least 8 bits (`i1` is a boolean).
    pub const fn is_int(self) -> bool {
        matches!(self, Elem::I8 | Elem::I16 | Elem::I32 | Elem::I64)
    }

    /// The width in bits.
    pub const fn bits(self) -> u32 {
        match self {
            Elem::I1 => 1,
            Elem::I8 => 8,
            Elem::I16 | Elem::F16 | Elem::BF16 => 16,
            Elem::I32 | Elem::F32 => 32,
            Elem::I64 => 64,
        }
    }

    fn parse(token: &str) -> Option<Elem> {
        Some(match token {
            "i1" => Elem::I1,
            "i8" => Elem::I8,
            "i16" => Elem::I16,
            "i32" => Elem::I32,
            "i64" => Elem::I64,
            "f16" => Elem::F16,
            "bf16" => Elem::BF16,
            "f32" => Elem::F32,
            _ => return None,
        })
    }

    fn of(ty: &Type) -> Option<Elem> {
        Some(match ty {
            Type::Int(1) => Elem::I1,
            Type::Int(8) => Elem::I8,
            Type::Int(16) => Elem::I16,
            Type::Int(32) => Elem::I32,
            Type::Int(64) => Elem::I64,
            Type::Half => Elem::F16,
            Type::BFloat => Elem::BF16,
            Type::Float => Elem::F32,
            _ => return None,
        })
    }

    /// Whether IR type `ty` is this element.
    fn is(self, ty: &Type) -> bool {
        Elem::of(ty) == Some(self)
    }
}

/// A scalar (`lanes == 1`) or fixed vector of an [`Elem`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NumTy {
    /// The scalar element.
    pub elem: Elem,
    /// The lane count: 1 for a scalar, at least 2 for a vector.
    pub lanes: u32,
}

impl NumTy {
    /// The scalar of `elem`.
    pub const fn scalar(elem: Elem) -> NumTy {
        NumTy { elem, lanes: 1 }
    }

    /// The same shape with element `elem`.
    pub const fn with_elem(self, elem: Elem) -> NumTy {
        NumTy {
            elem,
            lanes: self.lanes,
        }
    }

    /// Whether this is a vector.
    pub const fn is_vector(self) -> bool {
        self.lanes > 1
    }

    /// A name token: `f32`, `v4f16`, `i1`. A vector has at least two lanes; `v1` and leading
    /// zeros are not spellings AIR or LLVM produce.
    pub fn parse(token: &str) -> Option<NumTy> {
        let Some(rest) = token.strip_prefix('v') else {
            return Elem::parse(token).map(NumTy::scalar);
        };
        let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
        let (count, elem) = rest.split_at_checked(digits)?;
        if count.starts_with('0') {
            return None;
        }
        let lanes: u32 = count.parse().ok()?;
        (lanes >= 2).then_some(())?;
        Some(NumTy {
            elem: Elem::parse(elem)?,
            lanes,
        })
    }

    /// The shape of IR type `id`, when it is a scalar or vector of an [`Elem`].
    pub fn of(types: &TypeTable, id: TypeId) -> Option<NumTy> {
        match types.get(id)? {
            Type::Vector { elem, len } if *len >= 2 => Some(NumTy {
                elem: Elem::of(types.get(*elem)?)?,
                lanes: *len,
            }),
            Type::Vector { .. } => None,
            scalar => Elem::of(scalar).map(NumTy::scalar),
        }
    }

    /// Whether IR type `id` has exactly this shape.
    pub fn matches(self, types: &TypeTable, id: TypeId) -> bool {
        match types.get(id) {
            Some(Type::Vector { elem, len }) => {
                self.lanes >= 2
                    && *len == self.lanes
                    && types.get(*elem).is_some_and(|e| self.elem.is(e))
            }
            Some(scalar) => self.lanes == 1 && self.elem.is(scalar),
            None => false,
        }
    }
}

/// The signedness an integer family spells in its name (`.s.` / `.u.`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Sign {
    /// `s`: signed.
    Signed,
    /// `u`: unsigned.
    Unsigned,
}

impl Sign {
    /// Both signs.
    pub const ALL: [Sign; 2] = [Sign::Signed, Sign::Unsigned];

    /// `s` or `u`.
    pub fn parse(token: &str) -> Option<Sign> {
        match token {
            "s" => Some(Sign::Signed),
            "u" => Some(Sign::Unsigned),
            _ => None,
        }
    }

    /// The name token.
    #[cfg(test)]
    pub const fn token(self) -> &'static str {
        match self {
            Sign::Signed => "s",
            Sign::Unsigned => "u",
        }
    }
}

/// The name token of `ty` (the inverse of [`NumTy::parse`]).
#[cfg(test)]
pub(super) fn token(ty: NumTy) -> String {
    let elem = match ty.elem {
        Elem::I1 => "i1",
        Elem::I8 => "i8",
        Elem::I16 => "i16",
        Elem::I32 => "i32",
        Elem::I64 => "i64",
        Elem::F16 => "f16",
        Elem::BF16 => "bf16",
        Elem::F32 => "f32",
    };
    if ty.is_vector() {
        format!("v{}{elem}", ty.lanes)
    } else {
        elem.to_string()
    }
}
