//! The Metal API enums AIR carries: [`DataType`] (`MTLDataType`), the type of a function
//! constant, which the ABI states (`air.function_constants`) and a specialization request's
//! values name.
//!
//! Like the translator's own Metal enums (which re-export this one), it mirrors the Metal API's
//! numeric values, generated from the macOS SDK headers (names and values only;
//! `scripts/regen-metal-enums/`, never hand-edited), and its `TryFrom<u32>` rejects an ordinal
//! outside the known set with
//! [`Error::InvalidRequest`](crate::error::Error::InvalidRequest)`(`[`RequestError::UnknownEnumValue`](crate::error::RequestError::UnknownEnumValue)`)`:
//! an unknown value is never guessed.

/// Defines one Metal `NS_ENUM` as a closed Rust enum with a checked `TryFrom<u32>`, as the
/// translator's `metal` module does. The generated invocation states `#[non_exhaustive]` itself.
macro_rules! metal_enum {
    (
        $(#[$meta:meta])*
        $name:ident = $metal:literal {
            $($variant:ident = $value:literal, $mname:literal;)*
        }
        aliases {
            $($alias:literal => $target:ident;)*
        }
    ) => {
        $(#[$meta])*
        #[repr(u32)]
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
        pub enum $name {
            $(
                #[doc = concat!("`", $mname, "`.")]
                $variant = $value,
            )*
        }

        impl $name {
            /// The Metal type name.
            pub const METAL_NAME: &'static str = $metal;

            /// Every value, in declaration order.
            pub const ALL: &'static [$name] = &[$($name::$variant),*];

            const ALIASES: &'static [(&'static str, $name)] = &[$(($alias, $name::$target)),*];

            /// The raw Metal ordinal.
            pub const fn as_raw(self) -> u32 {
                self as u32
            }

            /// The Metal enumerator name.
            pub const fn name(self) -> &'static str {
                match self {
                    $($name::$variant => $mname,)*
                }
            }

            /// The value with this Metal enumerator name (current or deprecated spelling).
            pub fn from_name(name: &str) -> Option<$name> {
                Self::ALL
                    .iter()
                    .copied()
                    .find(|v| v.name() == name)
                    .or_else(|| Self::ALIASES.iter().find(|(n, _)| *n == name).map(|(_, v)| *v))
            }
        }

        impl From<$name> for u32 {
            fn from(value: $name) -> u32 {
                value.as_raw()
            }
        }

        impl TryFrom<u32> for $name {
            type Error = $crate::error::Error;

            fn try_from(value: u32) -> Result<$name, $crate::error::Error> {
                match value {
                    $($value => Ok($name::$variant),)*
                    _ => Err($crate::error::Error::InvalidRequest(
                        $crate::error::RequestError::UnknownEnumValue { ty: $metal, value },
                    )),
                }
            }
        }
    };
}

// The generated file invokes the macro above, so it is declared after it (textual scoping).
mod generated;

pub use generated::*;
