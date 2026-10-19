//! The first level read in place ([`super::MipmapFilterChain::plan`]'s `fetch`): the first later
//! level's kernel fetches each tap from a sampled view of the texture's first level instead of
//! reading the scratch buffer's copy of it, and recovers the texel's bytes in the format the
//! runtime holds it in, so that every later step reads the words the copy would have given it.
//!
//! **What a fetch returns** ([`components`]): a sampled view of a normalized format returns each
//! code `c` of `b` bits as a `float` near `c / (2^b - 1)` (unorm) or `max(c / (2^(b-1) - 1), -1)`
//! (snorm); of a 16-bit float format, each half's value as a `float`; of an unsigned integer
//! format, the bits. Vulkan states no precision for the first two, so a fetch recovers a texel's
//! codes only where the device is measured to return values its recovery reads back as every
//! code (the runtime's driver fact); the integer formats' bits are exact by construction.
//!
//! **The recovery.** A unorm code is `RoundEven(v (2^b - 1))`; a snorm code `RoundEven(v (2^(b-1)
//! - 1))` as `b` two's-complement bits (the code `-2^(b-1)` reads as `-(2^(b-1) - 1)`, both -1.0,
//! which the filter reads alike: the module docs); a half's bits the `float`'s converted exactly
//! (each half is a `float`; a NaN keeps its top ten mantissa bits, set to a quiet NaN's); an
//! integer component its bits. Each lands at its bit offset in the texel's little-endian bytes,
//! as [`components`] places it.

use super::layout::{Layout, Rule};
use crate::vk::Format;

/// How a fetched component's code is recovered (the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) enum Recover {
    /// A unorm code: the `float` times `2^bits - 1`, rounded.
    Unorm,
    /// A snorm code: the `float` times `2^(bits-1) - 1`, rounded, in `bits` two's-complement
    /// bits.
    Snorm,
    /// A 16-bit float: the `float` converted to a half's bits.
    Half,
    /// An unsigned integer: its bits.
    Bits,
}

/// One component of a fetched texel (R, G, B, A in order): how its code is recovered, its
/// width, and the bit of the texel's little-endian bytes it starts at.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct Component {
    pub(crate) recover: Recover,
    pub(crate) bits: u32,
    pub(crate) offset: u32,
}

const fn at(recover: Recover, bits: u32, offset: u32) -> Option<Component> {
    Some(Component {
        recover,
        bits,
        offset,
    })
}

/// The components a sampled view of `format` returns, where a fetch recovers its texels (the
/// module docs): the 8-bit unorm (`A8_UNORM` among them) and snorm formats, the 10-bit unorm
/// ones with their 2-bit alpha, the 4- and 5-bit packed unorm ones, the 16-bit unorm, snorm and
/// float ones, and the unsigned integer carriers of the extended-range formats. `None` for
/// another format: the 32-bit floats, the packed floats and the sRGB formats (whose views
/// decode; the runtime reads them through their UNORM twin) among them.
pub(crate) fn components(format: Format) -> Option<[Option<Component>; 4]> {
    COMPONENTS
        .iter()
        .find(|(f, _)| *f == format)
        .map(|(_, components)| *components)
}

/// [`components`]' table: each format a fetch recovers, and its components.
pub(crate) const COMPONENTS: &[(Format, [Option<Component>; 4])] = {
    use Recover::{Bits as B, Half as H, Snorm as S, Unorm as U};
    &[
        (Format::R8_UNORM, [at(U, 8, 0), None, None, None]),
        (Format::A8_UNORM, [None, None, None, at(U, 8, 0)]),
        (Format::R8G8_UNORM, [at(U, 8, 0), at(U, 8, 8), None, None]),
        (
            Format::R8G8B8A8_UNORM,
            [at(U, 8, 0), at(U, 8, 8), at(U, 8, 16), at(U, 8, 24)],
        ),
        (
            Format::B8G8R8A8_UNORM,
            [at(U, 8, 16), at(U, 8, 8), at(U, 8, 0), at(U, 8, 24)],
        ),
        (Format::R8_SNORM, [at(S, 8, 0), None, None, None]),
        (Format::R8G8_SNORM, [at(S, 8, 0), at(S, 8, 8), None, None]),
        (
            Format::R8G8B8A8_SNORM,
            [at(S, 8, 0), at(S, 8, 8), at(S, 8, 16), at(S, 8, 24)],
        ),
        (
            Format::A2B10G10R10_UNORM_PACK32,
            [at(U, 10, 0), at(U, 10, 10), at(U, 10, 20), at(U, 2, 30)],
        ),
        (
            Format::A2R10G10B10_UNORM_PACK32,
            [at(U, 10, 20), at(U, 10, 10), at(U, 10, 0), at(U, 2, 30)],
        ),
        (
            Format::R4G4B4A4_UNORM_PACK16,
            [at(U, 4, 12), at(U, 4, 8), at(U, 4, 4), at(U, 4, 0)],
        ),
        (
            Format::B4G4R4A4_UNORM_PACK16,
            [at(U, 4, 4), at(U, 4, 8), at(U, 4, 12), at(U, 4, 0)],
        ),
        (
            Format::R5G5B5A1_UNORM_PACK16,
            [at(U, 5, 11), at(U, 5, 6), at(U, 5, 1), at(U, 1, 0)],
        ),
        (
            Format::B5G5R5A1_UNORM_PACK16,
            [at(U, 5, 1), at(U, 5, 6), at(U, 5, 11), at(U, 1, 0)],
        ),
        (Format::R16_UNORM, [at(U, 16, 0), None, None, None]),
        (
            Format::R16G16_UNORM,
            [at(U, 16, 0), at(U, 16, 16), None, None],
        ),
        (
            Format::R16G16B16A16_UNORM,
            [at(U, 16, 0), at(U, 16, 16), at(U, 16, 32), at(U, 16, 48)],
        ),
        (Format::R16_SNORM, [at(S, 16, 0), None, None, None]),
        (
            Format::R16G16_SNORM,
            [at(S, 16, 0), at(S, 16, 16), None, None],
        ),
        (
            Format::R16G16B16A16_SNORM,
            [at(S, 16, 0), at(S, 16, 16), at(S, 16, 32), at(S, 16, 48)],
        ),
        (Format::R16_SFLOAT, [at(H, 16, 0), None, None, None]),
        (
            Format::R16G16_SFLOAT,
            [at(H, 16, 0), at(H, 16, 16), None, None],
        ),
        (
            Format::R16G16B16A16_SFLOAT,
            [at(H, 16, 0), at(H, 16, 16), at(H, 16, 32), at(H, 16, 48)],
        ),
        (Format::R32_UINT, [at(B, 32, 0), None, None, None]),
        (
            Format::R16G16B16A16_UINT,
            [at(B, 16, 0), at(B, 16, 16), at(B, 16, 32), at(B, 16, 48)],
        ),
    ]
};

/// Whether the components are integers, fetched from a view of `uint` sampled type, rather than
/// `float`s.
pub(super) fn integer(components: &[Option<Component>; 4]) -> bool {
    components
        .iter()
        .flatten()
        .all(|c| c.recover == Recover::Bits)
}

/// Whether a fetch of `components` recovers the bytes of a texel of `layout`: each field the
/// filter reads lies in one component of its kind (a normalized field in a unorm or snorm
/// component of its sign and exactly its bits, a half in a half, an extended-range code in an
/// integer component), and no component lies past the texel.
pub(super) fn fits(layout: &Layout, components: &[Option<Component>; 4]) -> bool {
    let texel_bits = 8 * layout.bytes;
    let inside = components
        .iter()
        .flatten()
        .all(|c| c.offset + c.bits <= texel_bits);
    let covered = layout.fields.iter().all(|field| {
        let exact = |c: &Component| c.offset == field.offset && c.bits == field.width;
        let within = |c: &Component| {
            c.offset <= field.offset && field.offset + field.width <= c.offset + c.bits
        };
        components
            .iter()
            .flatten()
            .any(|c| match (field.rule, c.recover) {
                (Rule::Constant(_), _) => true,
                (
                    Rule::Grid { signed: false, .. }
                    | Rule::Srgb8
                    | Rule::Fixed16 { signed: false },
                    Recover::Unorm,
                )
                | (
                    Rule::Grid { signed: true, .. } | Rule::Fixed16 { signed: true },
                    Recover::Snorm,
                )
                | (Rule::Float16, Recover::Half) => exact(c),
                (Rule::XrColour | Rule::XrAlpha | Rule::XrSrgb, Recover::Bits) => within(c),
                _ => false,
            })
    });
    inside && covered
}
