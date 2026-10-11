//! Constexpr sampler states.
//!
//! AIR stores a `constexpr sampler` as a constant global: one `i64` (AIR 2.5 and 2.6) or
//! `[2 x i64]` (AIR 2.7 and later). Apple's metalfe-32023.886 also writes the one-word form in an
//! AIR 2.7 module (a macOS 15 deployment target), and says so with the module flag
//! `air.workaround.sampler_state_variable_length_encoding`; the global's type decides the form
//! here, so either decodes in any module. Every field below is what the Metal frontend emits for
//! the matching `constexpr sampler` option, measured one option at a time (the table lives in the
//! tests):
//!
//! | `words[0]` bits | Field |
//! |---|---|
//! | 0-2, 3-5, 6-8 | address mode s / t / r |
//! | 9-10, 11-12, 13-14 | mag / min / mip filter |
//! | 15 | coordinates |
//! | 16-19 | compare function |
//! | 20-23 | max anisotropy − 1 |
//! | 24-39, 40-55 | LOD min / max clamp (half) |
//! | 56-57, 58-59 | border color / reduction |
//! | 60-62 | must be zero |
//! | 63 | ignored: set in every one-word sampler measured and in no two-word one |
//!
//! | `words[1]` bits | Field |
//! |---|---|
//! | 0-15 | LOD bias (half) |
//! | 16-63 | must be zero |
//!
//! An unknown code in a decoded field, or a set bit that must be zero, is
//! `AbiError::BadSamplerState`: the alternative would be inventing a filter or address mode.
//! Whether the device can reproduce a state (anisotropy, reduction, border, bias) is decided at
//! compile time from the profile, not here.

use super::md::Site;
use crate::error::AbiError;
use crate::error::Error;
use crate::ir::{Const, GlobalId, Module, Type};

/// A half-precision value, kept as its exact bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Half(pub u16);

impl Half {
    /// The value as `f32` (exact: every half is representable).
    pub fn to_f32(self) -> f32 {
        let bits = u32::from(self.0);
        let sign = (bits & 0x8000) << 16;
        let exp = (bits >> 10) & 0x1f;
        let frac = bits & 0x3ff;
        let magnitude = match (exp, frac) {
            (0, 0) => 0,
            // Subnormal: frac * 2^-24, normalized into f32.
            (0, _) => {
                let shift = frac.leading_zeros() - 21;
                let frac = (frac << shift) & 0x3ff;
                ((127 - 15 + 1 - shift) << 23) | (frac << 13)
            }
            (0x1f, _) => 0x7f80_0000 | (frac << 13),
            _ => ((exp + 127 - 15) << 23) | (frac << 13),
        };
        f32::from_bits(sign | magnitude)
    }
}

/// Texture-coordinate addressing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AddressMode {
    /// `address::clamp_to_zero` (code 0 with a transparent black border): reads outside the
    /// texture return zero.
    ClampToZero,
    /// `address::clamp_to_edge` (code 1).
    ClampToEdge,
    /// `address::repeat` (code 2).
    Repeat,
    /// `address::mirrored_repeat` (code 3).
    MirroredRepeat,
    /// Code 0 with an opaque border color.
    ClampToBorder,
}

/// Minification or magnification filtering.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Filter {
    /// Nearest-texel sampling (code 0).
    Nearest,
    /// Linear filtering (code 1).
    Linear,
    /// Bicubic filtering (code 2).
    Bicubic,
}

/// Mipmap filtering.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MipFilter {
    /// `mip_filter::none` (code 0): only the base level is sampled.
    None,
    /// `mip_filter::nearest` (code 1).
    Nearest,
    /// `mip_filter::linear` (code 2).
    Linear,
}

/// Coordinate normalization.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Coordinates {
    /// `coord::normalized` (bit clear): coordinates run from 0 to 1 across the texture.
    Normalized,
    /// `coord::pixel` (bit set): coordinates are in texels.
    Pixel,
}

/// Depth comparison. `None` (code 0) is never written by the frontend; `compare_func::never` and
/// an unspecified comparison both encode `Never`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CompareFunction {
    /// Code 0, which the frontend never writes.
    None,
    /// `compare_func::less` (code 1).
    Less,
    /// `compare_func::less_equal` (code 2).
    LessEqual,
    /// `compare_func::greater` (code 3).
    Greater,
    /// `compare_func::greater_equal` (code 4).
    GreaterEqual,
    /// `compare_func::equal` (code 5).
    Equal,
    /// `compare_func::not_equal` (code 6).
    NotEqual,
    /// `compare_func::always` (code 7).
    Always,
    /// `compare_func::never`, or no comparison stated (code 8).
    Never,
}

/// Border color.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BorderColor {
    /// `border_color::transparent_black` (code 0).
    TransparentBlack,
    /// `border_color::opaque_black` (code 1).
    OpaqueBlack,
    /// `border_color::opaque_white` (code 2).
    OpaqueWhite,
}

/// Min/max reduction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Reduction {
    /// The filter's weighted average of the texels (code 0,
    /// `MTLSamplerReductionModeWeightedAverage`).
    WeightedAverage,
    /// The texels' componentwise minimum (code 1, `MTLSamplerReductionModeMinimum`).
    Minimum,
    /// The texels' componentwise maximum (code 2, `MTLSamplerReductionModeMaximum`).
    Maximum,
}

/// A decoded constexpr sampler state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SamplerState {
    /// Addressing of the `s` (width) coordinate (`s_address`).
    pub address_s: AddressMode,
    /// Addressing of the `t` (height) coordinate (`t_address`).
    pub address_t: AddressMode,
    /// Addressing of the `r` (depth) coordinate (`r_address`).
    pub address_r: AddressMode,
    /// Magnification filter (`mag_filter::`).
    pub mag_filter: Filter,
    /// Minification filter (`min_filter::`).
    pub min_filter: Filter,
    /// Filtering between mipmap levels.
    pub mip_filter: MipFilter,
    /// Whether coordinates are normalized or in pixels.
    pub coordinates: Coordinates,
    /// The depth comparison of a comparison sample.
    pub compare: CompareFunction,
    /// `max_anisotropy(n)`: 1 through 16 (1: no anisotropic filtering).
    pub max_anisotropy: u32,
    /// The least level of detail sampled (`lod_clamp(min, ..)`).
    pub lod_min_clamp: Half,
    /// The greatest level of detail sampled (`lod_clamp(.., max)`).
    pub lod_max_clamp: Half,
    /// The color read outside the texture under [`AddressMode::ClampToBorder`].
    pub border_color: BorderColor,
    /// How filtered texels combine.
    pub reduction: Reduction,
    /// `bias(b)`: added to the computed level of detail (`words[1]`; 0 in an AIR 2.6 sampler).
    pub lod_bias: Half,
}

impl SamplerState {
    /// Decode the two AIR sampler words (see the module docs). `None`: an undecodable pattern.
    pub fn from_words(words: [u64; 2]) -> Option<SamplerState> {
        let [w, w1] = words;
        if (w >> 60) & 0x7 != 0 || w1 >> 16 != 0 {
            return None;
        }
        let field = |shift: u32, mask: u64| (w >> shift) & mask;
        let border_color = match field(56, 0x3) {
            0 => BorderColor::TransparentBlack,
            1 => BorderColor::OpaqueBlack,
            2 => BorderColor::OpaqueWhite,
            _ => return None,
        };
        let address = |shift: u32| match field(shift, 0x7) {
            0 if border_color != BorderColor::TransparentBlack => Some(AddressMode::ClampToBorder),
            0 => Some(AddressMode::ClampToZero),
            1 => Some(AddressMode::ClampToEdge),
            2 => Some(AddressMode::Repeat),
            3 => Some(AddressMode::MirroredRepeat),
            _ => None,
        };
        let filter = |shift: u32| match field(shift, 0x3) {
            0 => Some(Filter::Nearest),
            1 => Some(Filter::Linear),
            2 => Some(Filter::Bicubic),
            _ => None,
        };
        let mip_filter = match field(13, 0x3) {
            0 => MipFilter::None,
            1 => MipFilter::Nearest,
            2 => MipFilter::Linear,
            _ => return None,
        };
        let coordinates = if field(15, 0x1) == 0 {
            Coordinates::Normalized
        } else {
            Coordinates::Pixel
        };
        let compare = match field(16, 0xf) {
            0 => CompareFunction::None,
            1 => CompareFunction::Less,
            2 => CompareFunction::LessEqual,
            3 => CompareFunction::Greater,
            4 => CompareFunction::GreaterEqual,
            5 => CompareFunction::Equal,
            6 => CompareFunction::NotEqual,
            7 => CompareFunction::Always,
            8 => CompareFunction::Never,
            _ => return None,
        };
        let reduction = match field(58, 0x3) {
            0 => Reduction::WeightedAverage,
            1 => Reduction::Minimum,
            2 => Reduction::Maximum,
            _ => return None,
        };
        let half = |v: u64| Half((v & 0xffff) as u16);
        Some(SamplerState {
            address_s: address(0)?,
            address_t: address(3)?,
            address_r: address(6)?,
            mag_filter: filter(9)?,
            min_filter: filter(11)?,
            mip_filter,
            coordinates,
            compare,
            max_anisotropy: field(20, 0xf) as u32 + 1,
            lod_min_clamp: half(w >> 24),
            lod_max_clamp: half(w >> 40),
            border_color,
            reduction,
            lod_bias: half(w1),
        })
    }
}

/// Decode the state stored in constexpr-sampler global `g`: a constant with an `i64` or
/// `[2 x i64]` initializer.
pub(super) fn decode_global(m: &Module, g: GlobalId, site: &Site) -> Result<SamplerState, Error> {
    let bad = || site.err(AbiError::BadSamplerState, None);
    let global = m.global(g).ok_or_else(bad)?;
    let init = global.init.filter(|_| global.constant).ok_or_else(bad)?;
    let word = |c| -> Option<u64> {
        let c = m.consts().get(c)?;
        match (m.types().get(c.ty)?, &c.value) {
            (Type::Int(64), Const::Int(v)) => Some(*v),
            (Type::Int(64), Const::Null) => Some(0),
            _ => None,
        }
    };
    let constant = m.consts().get(init).ok_or_else(bad)?;
    let words = match (m.types().get(constant.ty), &constant.value) {
        (Some(Type::Int(64)), _) => [word(init).ok_or_else(bad)?, 0],
        (Some(Type::Array { len: 2, .. }), Const::Aggregate(elems)) => match &**elems {
            [a, b] => [word(*a).ok_or_else(bad)?, word(*b).ok_or_else(bad)?],
            _ => return Err(bad()),
        },
        (Some(Type::Array { len: 2, elem }), Const::Null)
            if m.types().get(*elem) == Some(&Type::Int(64)) =>
        {
            [0, 0]
        }
        _ => return Err(bad()),
    };
    // The one-word form sets bit 63; it carries no sampler property.
    let [w0, w1] = words;
    SamplerState::from_words([w0 & !(1 << 63), w1]).ok_or_else(bad)
}
