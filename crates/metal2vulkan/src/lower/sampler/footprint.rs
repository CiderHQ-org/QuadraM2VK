//! Metal's footprint of a sample or gather whose residency the function observes.
//!
//! `sparse_color::resident()` of a filtered sample is whether every texel the filter weighs is
//! resident. Metal counts only the texels its filter gives a nonzero weight, and Vulkan's linear
//! footprint holds the 2x2 texels around the position even where one weighs zero (lavapipe counts
//! it). Apple's rule, measured on apple-m3-max-macos26.7 (cases 39503d65 and efe06aaf at a width of
//! 256, both axes and the clamp-to-zero border, 363e39a2 at 384, off the sweep, and 203caac1 along
//! a 3D texture's z): per axis of size `N` and normalized coordinate `u`,
//!
//! - the position is `X = f32(u · N)`, the product rounded to `float` with ties away from zero
//!   (the probes: these cases hold no tie; `position`);
//! - a linear filter's sample position `X - 1/2` is rounded to 1/256 of a texel, ties up:
//!   `q = floor(256·X + 1/2) - 128`; its texels are `i0 = floor(q / 256)` with weight
//!   `(256 - q mod 256) / 256`, never 0, and `i0 + 1` with weight `(q mod 256) / 256`, counted only
//!   where it is not 0;
//! - a gather reads the same `i0` and `i0 + 1`, all four counted;
//! - a nearest filter reads `floor(X)`, not rounded to the quantum.
//!
//! A clamp-to-edge axis reads the texel clamped into the level, which counts; a clamp-to-zero or
//! clamp-to-border axis's texel outside the level is the border, which is no texel (Metal answers
//! resident across it; fd292944 measured both modes). [`of`] states where the rule decides the footprint: a constexpr normalized sampler, no
//! comparison or anisotropy (a lowered sampler's reduction is the weighted average), one filter for
//! minification and magnification, clamp addressing, level 0 (the sampler not mipmapped, or a
//! level-0 sample whose sampler neither biases nor clamps the level), and no texel offset.
//! Elsewhere (a runtime sampler, mip levels, repeat or mirrored addressing, an offset, none of them
//! measured) the footprint is the device's, as before.
//!
//! [`word`] emits the status word of such an access: the residency of each counted texel at
//! level 0, through the slot's sparse arm (`OpImageSparseFetch` of the texel) or its tile map,
//! under the slot's `SparseTexture` specialization, and the residency of each texel; the sample
//! itself is the device's plain sample. Where the bound texture is a sparse image, a nearest
//! sample's or a gather's texel that is not resident then reads zero, Metal's unmapped value,
//! before the slot's constant lanes are set (`emit::unmapped_value`).

use crate::air::abi::SamplerState;
use crate::air::abi::{AddressMode, Coordinates, Filter, MipFilter};
use crate::air::intrinsics::{ImageDim, ImageShape};
use crate::lower::emit::EmitCx;
use crate::lower::texture::TileLevel;
use crate::memory::views::View;
use crate::spirv::builder::constructs::{ArmEnd, ArmFn, Merged};
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::Value as SValue;
use crate::spirv::ir::{
    BinOp, ConstBits, Constant, ConvOp, GlslOp, ImageOperands, ImageQuery, Type as SType, UnOp,
};

use super::{Lod, SampleOp};

/// How the footprint's texels follow from the position.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    /// `floor(X)`.
    Nearest,
    /// `i0`, and `i0 + 1` where its quantized weight is not 0.
    Linear,
    /// `i0` and `i0 + 1`, every one.
    Gather,
}

/// Metal's footprint of one sample or gather.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Footprint {
    pub kind: Kind,
    /// The axes (2 for a 2D or 2D-array texture, 3 for a 3D one).
    pub axes: u8,
    /// Per axis, whether a texel outside the level is the border (clamp-to-zero or
    /// clamp-to-border) rather than the edge texel (clamp-to-edge).
    pub border: [bool; 3],
}

/// The largest texel distance the position keeps (`2^22`): any position past it is outside every
/// texture, so under clamp addressing it reads the same texels as at the limit.
const LIMIT: f32 = 4_194_304.0;

/// Metal's footprint of `op` on a texture of `shape` through the constexpr sampler `state`, where
/// the rule decides it; `None` elsewhere. `integer`: the texture's texels are integers (a sample of
/// one is not filtered).
pub fn of(
    state: &SamplerState,
    op: &SampleOp,
    shape: ImageShape,
    integer: bool,
) -> Option<Footprint> {
    if shape.ms || shape.depth {
        return None;
    }
    let axes: u8 = match (shape.dim, shape.array) {
        (ImageDim::D2, _) => 2,
        (ImageDim::D3, false) => 3,
        _ => return None,
    };
    // A comparison is the op's (`dref`, a depth texture's): the sampler's compare function
    // applies only there.
    if state.coordinates != Coordinates::Normalized || state.max_anisotropy > 1 {
        return None;
    }
    let mut border = [false; 3];
    let modes = [state.address_s, state.address_t, state.address_r];
    for (b, mode) in border.iter_mut().zip(modes).take(usize::from(axes)) {
        *b = match mode {
            AddressMode::ClampToEdge => false,
            AddressMode::ClampToZero | AddressMode::ClampToBorder => true,
            AddressMode::Repeat | AddressMode::MirroredRepeat => return None,
        };
    }
    let kind = match op {
        SampleOp::Gather { offset: None, .. } if axes == 2 => Kind::Gather,
        SampleOp::Sample {
            lod,
            dref: None,
            min_lod,
            offset: None,
        } if !integer => {
            let kind = match (state.min_filter, state.mag_filter) {
                (Filter::Nearest, Filter::Nearest) => Kind::Nearest,
                (Filter::Linear, Filter::Linear) => Kind::Linear,
                _ => return None,
            };
            // Level 0: the sampler is not mipmapped, or the sample reads level 0 and the sampler
            // neither biases nor clamps it.
            let base = matches!(
                lod,
                Lod::Level(None)
                    | Lod::Base {
                        bias: None,
                        min: None,
                        ..
                    }
            ) && min_lod.is_none()
                && state.lod_min_clamp.to_f32() == 0.0
                && state.lod_bias.to_f32() == 0.0;
            if state.mip_filter != MipFilter::None && !base {
                return None;
            }
            kind
        }
        _ => return None,
    };
    Some(Footprint { kind, axes, border })
}

/// Where an access reads a texel's residency: the slot's sparse arm (the combined binding's
/// variable, a fetch of the texel), or its tile map.
#[derive(Clone, Copy, Debug)]
pub enum Probe {
    Sparse,
    Tiled(View),
}

/// Whether an axis's `i0 + 1` is in the footprint.
#[derive(Clone, Copy)]
enum Second {
    /// Never (a nearest filter).
    No,
    /// Always (a gather).
    Yes,
    /// Where this `bool` holds (a linear filter: its quantized weight is not 0).
    If(SValue),
}

/// One axis of the footprint: its texel `i0`, whether `i0 + 1` counts, and the level's size.
struct Axis {
    i0: SValue,
    second: Second,
    size: SValue,
}

fn f32_const(fb: &mut FunctionBuilder<'_>, x: f32) -> BResult<SValue> {
    let f = fb.module().ty(SType::Float { width: 32 })?;
    let c = fb.module().constant(Constant::Scalar {
        ty: f,
        bits: ConstBits(u64::from(x.to_bits())),
    })?;
    fb.module().const_value(c)
}

fn i32_const(fb: &mut FunctionBuilder<'_>, x: i32) -> BResult<SValue> {
    let i = fb.module().ty(SType::Int {
        width: 32,
        signed: true,
    })?;
    let c = fb.module().constant(Constant::Scalar {
        ty: i,
        bits: ConstBits(u64::from(x as u32)),
    })?;
    fb.module().const_value(c)
}

fn u32_const(fb: &mut FunctionBuilder<'_>, x: u32) -> BResult<SValue> {
    let c = fb.u32_const(x)?;
    fb.module().const_value(c)
}

/// Component `i` of `v`, or `v` itself for a one-lane value.
fn lane(fb: &mut FunctionBuilder<'_>, v: SValue, lanes: u8, i: u32) -> BResult<SValue> {
    if lanes == 1 {
        Ok(v)
    } else {
        fb.composite_extract(v, &[i])
    }
}

/// One axis's texels from normalized coordinate `u` and the level's size `size` (a `u32`).
fn axis(fb: &mut FunctionBuilder<'_>, kind: Kind, u: SValue, size: SValue) -> BResult<Axis> {
    let i = fb.module().ty(SType::Int {
        width: 32,
        signed: true,
    })?;
    // Metal's position on a clamping axis: the product rounded with ties away from zero.
    let (x, _) = super::position::position(fb, u, size, super::bicubic::TapAddress::Edge)?;
    let (i0, second) = match kind {
        Kind::Nearest => {
            let lo = f32_const(fb, -LIMIT)?;
            let hi = f32_const(fb, LIMIT)?;
            let fl = fb.glsl(GlslOp::Floor, &[x])?;
            let fl = fb.glsl(GlslOp::FClamp, &[fl, lo, hi])?;
            (fb.convert(ConvOp::ConvertFToS, fl, i)?, Second::No)
        }
        Kind::Linear | Kind::Gather => {
            // q = floor(256·X + 1/2) - 128, exactly: 256·X is exact, and its fraction decides
            // the rounding (ties up).
            let k = f32_const(fb, 256.0)?;
            let y = fb.binary_no_contraction(BinOp::FMul, x, k)?;
            let lo = f32_const(fb, -LIMIT * 256.0)?;
            let hi = f32_const(fb, LIMIT * 256.0)?;
            let y = fb.glsl(GlslOp::FClamp, &[y, lo, hi])?;
            let fy = fb.glsl(GlslOp::Floor, &[y])?;
            let r = fb.binary_no_contraction(BinOp::FSub, y, fy)?;
            let half = f32_const(fb, 0.5)?;
            let up = fb.binary(BinOp::FOrdGreaterThanEqual, r, half)?;
            let q = fb.convert(ConvOp::ConvertFToS, fy, i)?;
            let one = i32_const(fb, 1)?;
            let zero = i32_const(fb, 0)?;
            let carry = fb.select(up, one, zero)?;
            let q = fb.binary(BinOp::IAdd, q, carry)?;
            let k128 = i32_const(fb, 128)?;
            let q = fb.binary(BinOp::ISub, q, k128)?;
            let eight = i32_const(fb, 8)?;
            let i0 = fb.binary(BinOp::ShiftRightArithmetic, q, eight)?;
            let second = match kind {
                Kind::Gather => Second::Yes,
                _ => {
                    let mask = i32_const(fb, 255)?;
                    let frac = fb.binary(BinOp::BitwiseAnd, q, mask)?;
                    Second::If(fb.binary(BinOp::INotEqual, frac, zero)?)
                }
            };
            (i0, second)
        }
    };
    Ok(Axis { i0, second, size })
}

/// Texel `i` (an `i32`) of an axis of size `size`: clamped into the level (a `u32`, the texel
/// read), and whether it is inside the level (`None` where the axis clamps to its edge, every
/// texel counted). A negative texel is outside: as a `u32` it is at least `2^31`, past any size.
fn texel(
    fb: &mut FunctionBuilder<'_>,
    i: SValue,
    size: SValue,
    border: bool,
) -> BResult<(SValue, Option<SValue>)> {
    let u = fb.module().ty(SType::Int {
        width: 32,
        signed: false,
    })?;
    let zero = i32_const(fb, 0)?;
    let one = u32_const(fb, 1)?;
    let last = fb.binary(BinOp::ISub, size, one)?;
    let low = fb.glsl(GlslOp::SMax, &[i, zero])?;
    let low = fb.convert(ConvOp::Bitcast, low, u)?;
    let clamped = fb.glsl(GlslOp::UMin, &[low, last])?;
    if !border {
        return Ok((clamped, None));
    }
    let raw = fb.convert(ConvOp::Bitcast, i, u)?;
    Ok((clamped, Some(fb.binary(BinOp::ULessThan, raw, size)?)))
}

/// Where a sample or gather reads: the combined binding, the normalized plane coordinate `coord`
/// (`lanes` floats), and the array layer (a `u32`, clamped into the array as sampling clamps it).
#[derive(Clone, Copy, Debug)]
pub struct Site {
    pub binding: super::Binding,
    pub coord: SValue,
    pub lanes: u8,
    pub layer: Option<SValue>,
}

/// The status word of a sample or gather and the residency of its texels ([`counted`]).
pub struct Counted {
    /// The status word (a `u32`): 1 where a texel Metal counts is not resident, else 0.
    pub word: SValue,
    /// Per texel of the footprint, in [`Footprint::texels`]' order, whether it is resident as
    /// far as the word goes (a `bool`): a texel that does not count (its weight 0, or the
    /// border) is.
    pub resident: Vec<SValue>,
}

impl Footprint {
    /// The texels of the footprint, each a combination of each axis's texels: bit `a` of a
    /// combination picks axis `a`'s `i0 + 1` over its `i0`. A nearest filter reads combination 0
    /// alone; a linear filter and a gather read every one, a 2D gather's taps `x`, `y`, `z` and
    /// `w` (`(i0, j0 + 1)`, `(i0 + 1, j0 + 1)`, `(i0 + 1, j0)`, `(i0, j0)`) its combinations 2,
    /// 3, 1 and 0 ([`GATHER_TEXELS`]).
    pub fn texels(&self) -> usize {
        match self.kind {
            Kind::Nearest => 1,
            Kind::Linear | Kind::Gather => 1 << self.axes,
        }
    }
}

/// The combination ([`Footprint::texels`]) of each tap of a 2D gather, in its lanes' order.
pub const GATHER_TEXELS: [usize; 4] = [2, 3, 1, 0];

/// The status word of a sample or gather with footprint `fp` at `site`, on a slot whose
/// `SparseTexture` specialization is `spec`, and the residency of each of its texels:
/// [`counted`]'s where the bound texture is sparse, and 0 and every texel resident where it is
/// not.
pub fn word(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    fp: Footprint,
    spec: SValue,
    probe: Probe,
    site: Site,
) -> BResult<Counted> {
    let n = fp.texels();
    let sparse: ArmFn<'_, ()> = Box::new(move |fb, _| {
        let c = counted(fb, fp, probe, site)?;
        Ok(ArmEnd::FallThrough(
            std::iter::once(c.word).chain(c.resident).collect(),
        ))
    });
    let plain: ArmFn<'_, ()> = Box::new(move |fb, _| {
        let word = u32_const(fb, 0)?;
        let b = fb.module().ty(SType::Bool)?;
        let yes = fb.module().constant(Constant::Scalar {
            ty: b,
            bits: ConstBits(1),
        })?;
        let yes = fb.module().const_value(yes)?;
        Ok(ArmEnd::FallThrough(
            std::iter::once(word)
                .chain(std::iter::repeat_n(yes, n))
                .collect(),
        ))
    });
    let Merged(merged) = fb.if_(&mut (), spec, sparse, Some(plain))?;
    match merged.split_first() {
        Some((&word, resident)) if resident.len() == n => Ok(Counted {
            word,
            resident: resident.to_vec(),
        }),
        _ => Err(cx.fail("a footprint arm without its word and texels")),
    }
}

/// The status word (a `u32`) of a sample or gather of a sparse texture with footprint `fp` at
/// `site`: 1 where a texel Metal counts is not resident at level 0, read through `probe`, else 0;
/// and the residency of each texel of the footprint ([`Counted::resident`]).
pub fn counted(
    fb: &mut FunctionBuilder<'_>,
    fp: Footprint,
    probe: Probe,
    site: Site,
) -> BResult<Counted> {
    let Site {
        binding,
        coord,
        lanes,
        layer,
    } = site;
    let img = binding.image(fb)?;
    let zero = u32_const(fb, 0)?;
    let size = fb.query(ImageQuery::SizeLod(zero), img)?;
    let size_lanes = fp.axes + u8::from(layer.is_some());
    let mut axes = Vec::with_capacity(usize::from(fp.axes));
    for a in 0..u32::from(fp.axes) {
        let u = lane(fb, coord, lanes, a)?;
        let n = lane(fb, size, size_lanes, a)?;
        axes.push(axis(fb, fp.kind, u, n)?);
    }
    // The layer, clamped into the array.
    let slice = match layer {
        Some(l) => {
            let layers = fb.composite_extract(size, &[u32::from(fp.axes)])?;
            let one = u32_const(fb, 1)?;
            let last = fb.binary(BinOp::ISub, layers, one)?;
            Some(fb.glsl(GlslOp::UMin, &[l, last])?)
        }
        None => None,
    };
    // Each axis's texels, once: `i0`, and `i0 + 1` where the footprint has it, each clamped into
    // the level, with whether it counts (its weight is not 0, and it is not the border).
    let tiles = match probe {
        Probe::Tiled(map) => Some(TileLevel::new(fb, map, zero, slice.unwrap_or(zero))?),
        Probe::Sparse => None,
    };
    let mut picks: Vec<Vec<Pick>> = Vec::with_capacity(axes.len());
    for (a, ax) in axes.iter().enumerate() {
        let border = fp.border.get(a).copied().unwrap_or(false);
        let mut both = Vec::with_capacity(2);
        let (t, inside) = texel(fb, ax.i0, ax.size, border)?;
        both.push(pick(fb, tiles.as_ref(), a, t, inside)?);
        if !matches!(ax.second, Second::No) {
            let one = i32_const(fb, 1)?;
            let i = fb.binary(BinOp::IAdd, ax.i0, one)?;
            let (t, inside) = texel(fb, i, ax.size, border)?;
            let counts = match (ax.second, inside) {
                (Second::If(weighed), Some(inside)) => {
                    Some(fb.binary(BinOp::LogicalAnd, weighed, inside)?)
                }
                (Second::If(weighed), None) => Some(weighed),
                (_, inside) => inside,
            };
            both.push(pick(fb, tiles.as_ref(), a, t, counts)?);
        }
        picks.push(both);
    }
    // A 2D texture's tiles are in its one z tile.
    let tz = match &tiles {
        Some(l) if picks.len() < 3 => Some(l.tile(fb, 2, zero)?),
        _ => None,
    };
    // Every combination of each axis's texels (bit `a` of `combo` picks axis `a`'s second); a
    // combination that picks a second an axis lacks is not in the footprint.
    let mut all: Option<SValue> = None;
    let mut resident_texels = Vec::with_capacity(fp.texels());
    for combo in 0..(1usize << picks.len()) {
        let chosen: Option<Vec<Pick>> = picks
            .iter()
            .enumerate()
            .map(|(a, p)| p.get(combo >> a & 1).copied())
            .collect();
        let Some(chosen) = chosen else {
            continue;
        };
        let mut counts: Option<SValue> = None;
        for p in &chosen {
            if let Some(c) = p.counts {
                counts = Some(and(fb, counts, c)?);
            }
        }
        let resident = match (&tiles, probe) {
            (Some(l), _) => {
                let mut at = [tz.unwrap_or(zero); 3];
                for (slot, p) in at.iter_mut().zip(&chosen) {
                    *slot = p.tile.unwrap_or(zero);
                }
                l.mapped(fb, at)?
            }
            (None, _) => {
                let u = fb.module().ty(SType::Int {
                    width: 32,
                    signed: false,
                })?;
                let mut parts: Vec<SValue> = chosen.iter().map(|p| p.texel).collect();
                parts.extend(slice);
                let n = u8::try_from(parts.len()).unwrap_or(u8::MAX);
                let ty = fb.module().ty(SType::Vector { elem: u, len: n })?;
                let at = fb.composite_construct(ty, &parts)?;
                let ops = ImageOperands {
                    lod: Some(zero),
                    ..ImageOperands::default()
                };
                let r = fb.sparse_fetch(img, at, &ops)?;
                let code = fb.composite_extract(r, &[0])?;
                fb.texels_resident(code)?
            }
        };
        // A texel that does not count is resident as far as the word goes.
        let ok = match counts {
            Some(c) => {
                let not = fb.unary(UnOp::LogicalNot, c)?;
                fb.binary(BinOp::LogicalOr, not, resident)?
            }
            None => resident,
        };
        all = Some(and(fb, all, ok)?);
        resident_texels.push(ok);
    }
    // Combination 0 (every axis's `i0`) is always in the footprint, so `all` is set.
    let one = u32_const(fb, 1)?;
    let word = match all {
        Some(all) => fb.select(all, zero, one)?,
        None => zero,
    };
    Ok(Counted {
        word,
        resident: resident_texels,
    })
}

/// One texel of an axis of the footprint: its coordinate clamped into the level, its tile under a
/// tile map, and whether it counts (`None`: always).
#[derive(Clone, Copy)]
struct Pick {
    texel: SValue,
    tile: Option<SValue>,
    counts: Option<SValue>,
}

fn pick(
    fb: &mut FunctionBuilder<'_>,
    tiles: Option<&TileLevel>,
    axis: usize,
    texel: SValue,
    counts: Option<SValue>,
) -> BResult<Pick> {
    let tile = match tiles {
        Some(l) => Some(l.tile(fb, axis, texel)?),
        None => None,
    };
    Ok(Pick {
        texel,
        tile,
        counts,
    })
}

/// `a ∧ b`, or `b` alone.
fn and(fb: &mut FunctionBuilder<'_>, a: Option<SValue>, b: SValue) -> BResult<SValue> {
    match a {
        Some(a) => fb.binary(BinOp::LogicalAnd, a, b),
        None => Ok(b),
    }
}

#[cfg(test)]
mod tests;
