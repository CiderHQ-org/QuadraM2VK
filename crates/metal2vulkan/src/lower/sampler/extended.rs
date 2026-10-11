//! Samples of an extended-range texture: Metal's nearest and linear filters of a `BGR10_XR`,
//! `BGRA10_XR` or `RGB10A8_2P_XR10` texture (or an `_sRGB` twin), emulated by fetches of the raw
//! texels the slot's variant holds (`LoweringInputs::extended_range`).
//!
//! **Why fetches on every device.** No Vulkan format holds these codes: the image holds the
//! texels' bytes in an unsigned integer format, which no device filters, so the shader reads
//! the taps and filters them itself, on a device that samples through Metal's own sampler too.
//!
//! **Metal's answers** (EVs TXN-xrprobe3 and TXN-xrprobe4, an Apple M3 Max on macOS 26.7 and an
//! Apple M4 on macOS 27.0.1, byte-identical): a nearest sample is the read of the texel at
//! Metal's position, decoded (`extended_range::decode`, its `half` rounded with ties away from
//! zero). A linear sample's footprint is the other formats' (`linear`, the position quantized to
//! 1/256 of a texel, each tap addressed on its own, a border tap the sampler's border color through
//! the format's channels), and its sum is the exact routine's on the format's grid
//! (`bicubic::routines`, `BicubicTexels::EXTENDED`): each tap's code `16 (c - 384)` (an `_sRGB`
//! color step `±s`, alpha `8 (c - 384)` of its clamped code), each term of the 2D product weights
//! truncated (onto `2^-7` of a color lane's steps, `2^-4` of alpha's), the step rounded half up,
//! divided by the grid exactly, and a `half` result rounded to nearest even but an `_sRGB` color
//! lane's, which rounds the exact `f32` with ties away from zero. Every lane of 4,096 random 2D
//! footprints per format and result type matched (EV TXN-xrprobe4).
//!
//! **The two-plane formats** (EV XRR-tx1-m4, an Apple M4 on macOS 27.0.1: every lane of 4,096
//! samples of a 16 x 16 texture per format, filter, address mode and result type): the color
//! lanes are `BGR10_XR`'s (or its twin's), and the alpha byte `b` reads `b / 255` (a nearest
//! sample's `half` rounded with ties away from zero, as the decode rounds it) and filters as an
//! 8-bit unorm code on `1/(16 · 255)`, which is alpha's grid `8 · 510` here: the step
//! `⌊Σ 16 b w / 2^16 + 1/2⌋` of the product weights `w`, its `half` rounded to nearest even.
//!
//! **A runtime sampler** reads its slot's `SamplerState` word: the sample is linear where the word
//! states linear minification and magnification, else nearest; the taps follow the word's address
//! modes, border color, and coordinates. The binder refuses a bound sampler whose word states
//! anything else (`ResourcePlanner::pair`).
//!
//! **The bfloat formats** (EV FPV-pf5-m4, an Apple M4 under macOS 27.0.1, every code of
//! `R16Bfloat`, `RG16Bfloat` and `RGBA16Bfloat`; EV FPV-pf8-m4, 18, 432 positions of each under
//! every address mode and border color): a sample is nearest under either filter, Metal's linear
//! filter reading the nearest texel, the texel at `floor` of the unsnapped coordinate, so a runtime
//! sampler's word selects no filter, and its `half` result is the `float` result toward zero
//! (saturating at `65504`); a gather of a component reads the four taps of the linear footprint
//! (`TapBase::Quantized`), each decoded, that component of each in Vulkan's order `(x0, y1)`, `(x1,
//! y1)`, `(x1, y0)`, `(x0, y0)`, its `half` result rounded to nearest even.
//!
//! **Gathers and swizzles** (EV FPV-sw1-m4, an Apple M4 under macOS 27.0.1: reads, nearest and
//! linear samples under an opaque-white border, and gathers of each component, of an 8 x 4 texture
//! of each `BGR10_XR`, `BGRA10_XR`, padded and bfloat format and `_sRGB` twin, at 540 positions,
//! unswizzled and under four swizzles). An extended-range or padded form's `float` gather is the
//! bfloat form's: the four taps of the linear footprint, each the texel its read gives or the
//! border texel through the format's channels. A swizzled texture's answers are the unswizzled
//! texture's, swizzled: the swizzle applies to the decoded texel, a border texel included, and a
//! gather of a component gathers the component the swizzle names (a `Zero` or `One` 0 or 1 at every
//! tap). The slot's view holds raw codes, so it takes no swizzle: the slot joins
//! `LoweringInputs::swizzled_storage` and the function swizzles under its `TextureSwizzle`
//! specialization. A two-plane form's gather, and another non-bfloat form's `half` gather, are not
//! measured.

use crate::air::abi::BorderColor;
use crate::air::ir::ValueId;
use crate::lower::emit::EmitCx;
use crate::metal::bicubic::BicubicTexels;
use crate::metal::format::ExtendedRange;
use crate::metal::sampler::SamplerWord;
use crate::metal::{PixelFormat, TextureSwizzleChannels};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn, Merged};
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::{DemoteTy, FuncId, GlobalId, Value as SValue};
use crate::spirv::ir::{BinOp, ConstBits, Constant};

use super::bicubic::{
    footprint_of_image, missing_routine, scalar, value, Addressing, Bits, Footprint, Tap,
    TapAddress, TapBase, Taps, Tys,
};
use super::{CombinedImage, Pixel};

/// A sample of an extended-range texture slot, emulated by fetches of its raw texels (module
/// docs).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sample {
    /// The texture slot, whose variant binds [`form`](Self::form).
    pub texture: u32,
    pub form: ExtendedRange,
    /// The sampler's state the sample follows.
    pub state: State,
    /// A gather of this component (0 to 3, red first) instead of a sample: a bfloat form's,
    /// or another's but a two-plane form's, with a `float` result.
    pub gather: Option<u32>,
    /// The slot is one of `LoweringInputs::swizzled_storage`: the sample swizzles its decoded
    /// texel, a border texel included, and a gather each decoded tap before it takes its
    /// component, under the slot's `TextureSwizzle` specialization.
    pub swizzle: bool,
    /// The constant texel offset Metal applies, per plane axis.
    pub offset: Option<Vec<i32>>,
    /// The offset moves an axis that may repeat or mirror: the sample places it by the slot's
    /// `SpecKey::OffsetBaseAxes` tests (`texture::OffsetBase`), which state where Metal's rule
    /// reads a view's parent's base-level size (`position::offset_dropped`).
    pub offset_base: bool,
    /// The plane coordinate: two floats.
    pub coord: ValueId,
    /// The result is `half`, else `float`.
    pub half: bool,
    /// The lowered type of the result (`{T, i8}`, its status word 0).
    pub result: DemoteTy,
}

/// The state of the sampler an extended-range sample follows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum State {
    /// A constexpr sampler's, known at translation.
    Constant {
        /// Linear minification and magnification, else nearest.
        linear: bool,
        /// The addressing of the `s` and `t` axes.
        address: Vec<TapAddress>,
        /// The border color every axis clamping to zero or to the border reads.
        border: BorderColor,
        /// Pixel coordinates: the plane coordinate is already in texels.
        pixel: bool,
    },
    /// The runtime sampler in slot `sampler`: its `SamplerState` specialization's word states
    /// the filter, the address modes, the border color, and the coordinates.
    Slot { sampler: u32 },
}

/// The plane axes a sample reads: a 2D texture's.
pub(super) const AXES: usize = 2;

/// The census words of a nearest sample's tap beyond its position: the size query, the floors, the
/// tap's addressing, the coordinate, the fetch, and the border's select.
pub(super) const NEAREST_WORDS: u32 = 192;

/// The census words of a runtime sampler's branch: the filter's test and the merge.
pub(super) const BRANCH_WORDS: u32 = 48;

/// The census words of one swizzle of a decoded texel: the word's four fields, their tests, and the
/// selects of each lane's source.
pub(super) const SWIZZLE_WORDS: u32 = 192;

impl Sample {
    /// The census name.
    pub(crate) fn name(&self) -> &'static str {
        if self.gather.is_some() {
            return "sampler.extended_range.gather";
        }
        match self.state {
            State::Constant { linear: true, .. } => "sampler.extended_range.linear",
            State::Constant { linear: false, .. } => "sampler.extended_range.nearest",
            State::Slot { .. } => "sampler.extended_range.runtime",
        }
    }

    /// The estimated words: per filter the sample takes, each axis's position (which a pixel
    /// coordinate does not take, and which a runtime sampler's word adds its mode to), the
    /// footprint and its taps' addressing (a runtime sampler's under every mode), the fetches and
    /// their decodes, the routine's call (whose own words the module counts once,
    /// `Routine::words`), and Metal's offset rule on each axis an offset moves (with the slot's
    /// parent tests, `position::PARENT_RULE_WORDS`); a runtime sampler's word's tests and its
    /// branch; and the status composite.
    pub(crate) fn words(&self) -> u32 {
        let axes = AXES as u32;
        let decode = crate::lower::arith::extended_range::DECODE_WORDS;
        let moved = self
            .offset
            .as_deref()
            .map_or(0, |o| o.iter().filter(|o| **o != 0).count() as u32);
        let rule = match self.offset_base {
            true => super::position::OFFSET_RULE_WORDS + super::position::PARENT_RULE_WORDS,
            false => super::position::OFFSET_RULE_WORDS,
        };
        let offset = moved * rule;
        let linear = super::linear::PLANE_WORDS + 4 * decode + offset;
        let nearest = NEAREST_WORDS + decode + offset;
        let position = axes * super::position::POSITION_WORDS;
        // A gather's footprint and four decodes, or a bfloat form's nearest sample alone, under
        // its addressing.
        let one = match (self.gather, self.form.bfloat()) {
            (Some(_), _) => Some((linear, 2)),
            (None, true) => Some((nearest, 1)),
            (None, false) => None,
        };
        // A swizzle of the sample's texel, or of each of a gather's four taps.
        let swizzle = match (self.swizzle, self.gather) {
            (false, _) => 0,
            (true, None) => SWIZZLE_WORDS,
            (true, Some(_)) => 4 * SWIZZLE_WORDS,
        };
        if let Some((words, taps)) = one {
            let addressing = match &self.state {
                State::Constant { pixel: true, .. } => 0,
                State::Constant { pixel: false, .. } => position,
                State::Slot { .. } => {
                    super::linear::WORD_WORDS
                        + position
                        + axes * super::position::MODE_POSITION_WORDS
                        + taps * axes * super::linear::WORD_TAP_WORDS
                }
            };
            return words + addressing + swizzle + 16;
        }
        let filters = match &self.state {
            State::Constant {
                linear: true,
                pixel,
                ..
            } => linear + if *pixel { 0 } else { position },
            State::Constant {
                linear: false,
                pixel,
                ..
            } => nearest + if *pixel { 0 } else { position },
            State::Slot { .. } => {
                let word = position + axes * super::position::MODE_POSITION_WORDS;
                let taps = axes * super::linear::WORD_TAP_WORDS;
                super::linear::WORD_WORDS
                    + BRANCH_WORDS
                    + (linear + word + 2 * taps)
                    + (nearest + word + taps)
            }
        };
        filters + swizzle + 16
    }
}

/// What a sample's fetches read: the slot's image (image facts `image`), the format's bits for
/// the exact routine and the border texel ([`flags`]), and how its raw texels decode; and the
/// slot's swizzle word, for a swizzled texture of its variant.
pub(super) struct Texture {
    pub img: SValue,
    pub image: CombinedImage,
    pub decode: Decode,
    pub flags: SValue,
    pub swizzle: Option<SValue>,
    /// The slot's `SpecKey::OffsetBaseAxes` tests, by which a texel offset is placed
    /// (`position::offset_dropped`), where the sample reads them ([`Sample::offset_base`]).
    pub parent: Option<crate::lower::texture::OffsetBase>,
}

/// How a sample's raw texels decode: by one form, a slot's variant's, through the form's sRGB
/// table; or by the form a heap handle's class index names (`sampler::heap_raw`), through the
/// heap's decode routine of each result type (`texture::heap_raw::define`).
#[derive(Clone, Copy)]
pub(super) enum Decode {
    Form {
        form: ExtendedRange,
        table: Option<GlobalId>,
    },
    Heap {
        /// The form's index in `ExtendedRange::HEAP_READ` (a `u32`).
        form: SValue,
        /// Whether it is a bfloat form (a `bool`).
        bfloat: SValue,
        /// The routine of a `float` result, and of a `half` one where the access rounds to it.
        routines: (FuncId, Option<FuncId>),
    },
}

/// Where a sample reads: its plane coordinate (two floats), the constant texel offset Metal
/// applies, per plane axis, and whether its result is `half`.
#[derive(Clone, Copy)]
pub(super) struct At<'a> {
    pub coord: ValueId,
    pub offset: Option<&'a [i32]>,
    pub half: bool,
}

/// The texel of extended-range sample `e` in the open block, with its status word 0 (module
/// docs).
pub(crate) fn emit(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    e: &Sample,
) -> BResult<SValue> {
    let (decl, img) = crate::lower::texture::slot_image(cx, fb, e.texture)?;
    let table = crate::lower::texture::extended_table(cx, e.form)?;
    let t = Tys::new(fb)?;
    let flags = scalar(fb, t.u32_, u64::from(flags(e.form)))?;
    let swizzle = match e.swizzle {
        true => match cx.decl.families.texture.swizzle.get(&e.texture).copied() {
            Some(word) => Some(word),
            None => {
                return Err(cx.fail(format!(
                    "texture slot {} has no swizzle constant",
                    e.texture
                )))
            }
        },
        false => None,
    };
    let parent = match e.offset_base {
        true => match cx
            .decl
            .families
            .texture
            .offset_base
            .get(&e.texture)
            .copied()
        {
            Some(base) => Some(base),
            None => {
                return Err(cx.fail(format!(
                    "texture slot {} has no OffsetBaseAxes specialization",
                    e.texture
                )))
            }
        },
        false => None,
    };
    let tex = Texture {
        img,
        image: CombinedImage {
            plane: AXES as u8,
            arrayed: false,
            texel: decl.texel,
        },
        decode: Decode::Form {
            form: e.form,
            table,
        },
        flags: value(fb, flags)?,
        swizzle,
        parent,
    };
    let at = At {
        coord: e.coord,
        offset: e.offset.as_deref(),
        half: e.half,
    };
    // A gather, and a bfloat form's sample, follow no filter.
    if e.gather.is_some() || e.form.bfloat() {
        let addressing = match &e.state {
            State::Constant {
                address,
                border,
                pixel,
                ..
            } => Addressing::Modes {
                address,
                border: *border,
                pixel: *pixel,
            },
            State::Slot { sampler } => {
                let word = super::emit::sampler_word(cx, fb, Pixel::Spec(*sampler))?;
                Addressing::Word { word, axes: AXES }
            }
        };
        let x = match e.gather {
            // A gather swizzles each tap itself.
            Some(component) => gathered(cx, fb, &t, &at, &tex, addressing, component)?,
            None => {
                let x = nearest(cx, fb, &t, &at, &tex, addressing)?;
                swizzled(fb, &tex, x)?
            }
        };
        return crate::lower::texture::with_word(cx, fb, &e.result, x, None);
    }
    let x = match &e.state {
        State::Constant {
            linear,
            address,
            border,
            pixel,
        } => {
            let addressing = Addressing::Modes {
                address,
                border: *border,
                pixel: *pixel,
            };
            match linear {
                true => filtered(cx, fb, &t, &at, &tex, addressing)?,
                false => nearest(cx, fb, &t, &at, &tex, addressing)?,
            }
        }
        State::Slot { sampler } => {
            let word = super::emit::sampler_word(cx, fb, Pixel::Spec(*sampler))?;
            let addressing = Addressing::Word { word, axes: AXES };
            let mask = scalar(fb, t.u32_, u64::from(SamplerWord::LINEAR_FILTER))?;
            let mask = value(fb, mask)?;
            let want = scalar(fb, t.u32_, u64::from(SamplerWord::LINEAR))?;
            let want = value(fb, want)?;
            let have = fb.binary(BinOp::BitwiseAnd, word, mask)?;
            let linear = fb.binary(BinOp::IEqual, have, want)?;
            let (t, tex, at) = (&t, &tex, &at);
            let filtered: ArmFn<'_, EmitCx<'_>> = Box::new(move |fb, cx| {
                let x = filtered(cx, fb, t, at, tex, addressing)?;
                Ok(ArmEnd::FallThrough(vec![x]))
            });
            let nearest: ArmFn<'_, EmitCx<'_>> = Box::new(move |fb, cx| {
                let x = nearest(cx, fb, t, at, tex, addressing)?;
                Ok(ArmEnd::FallThrough(vec![x]))
            });
            let Merged(merged) = fb.if_(cx, linear, filtered, Some(nearest))?;
            match merged[..] {
                [x] => x,
                _ => return Err(cx.fail("an extended-range sample's branch without its texel")),
            }
        }
    };
    let x = swizzled(fb, &tex, x)?;
    crate::lower::texture::with_word(cx, fb, &e.result, x, None)
}

/// `x`, a sample's or a gather tap's decoded texel (a `vec4` of `f32`, a border texel included),
/// swizzled by `tex`'s swizzle word where it has one, as plain Metal swizzles a sampled texture's
/// texels after decoding them (EV FPV-sw1-m4, an Apple M4 under macOS 27.0.1: each read, nearest
/// and linear sample under an opaque-white border, and gather of each component, of an 8 x 4
/// texture of each extended-range, padded and bfloat format under four swizzles, is the
/// unswizzled texture's, swizzled).
fn swizzled(fb: &mut FunctionBuilder<'_>, tex: &Texture, x: SValue) -> BResult<SValue> {
    match tex.swizzle {
        Some(word) => crate::lower::texture::swizzled_float(fb, x, word),
        None => Ok(x),
    }
}

/// The linear sample at `at` of `tex` under `addressing`: the exact 2D routine's sum of the four
/// taps' decoded values, a merged axis's fraction taken as 0.
pub(super) fn filtered(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    t: &Tys,
    at: &At<'_>,
    tex: &Texture,
    addressing: Addressing<'_>,
) -> BResult<SValue> {
    let taps = Taps {
        addressing,
        offset: at.offset,
        span: &[0, 1],
        base: TapBase::Quantized,
        level: None,
    };
    let fp = footprint_of_image(
        cx,
        fb,
        t,
        (at.coord, None),
        (tex.img, tex.image),
        Bits::Value(tex.flags),
        (&taps, tex.parent),
    )?;
    let [(kx, xs), (ky, ys)] = &fp.axes[..] else {
        return Err(cx.fail("an extended-range linear sample without two plane axes"));
    };
    let (&[x0, x1], &[y0, y1]) = (xs.as_slice(), ys.as_slice()) else {
        return Err(cx.fail("an extended-range linear axis without two taps"));
    };
    let (kx, ky) = (*kx, *ky);
    let Some(quad) = cx.decl.families.sampler.linear_quad else {
        return Err(cx.fail("an extended-range linear sample without its routines"));
    };
    let mut args = Vec::with_capacity(8);
    for site in [[x0, y0], [x1, y0], [x0, y1], [x1, y1]] {
        args.push(texel(cx, fb, &fp, tex, &site, false)?);
    }
    args.push(super::linear::merged(fb, t, kx, [x0, x1])?);
    args.push(super::linear::merged(fb, t, ky, [y0, y1])?);
    args.push(tex.flags);
    let half = fb.module().constant(Constant::Scalar {
        ty: t.bool_,
        bits: ConstBits(u64::from(at.half)),
    })?;
    args.push(value(fb, half)?);
    fb.call(quad.exact, &args)?.ok_or_else(missing_routine)
}

/// The nearest sample at `at` of `tex` under `addressing`: the texel at Metal's position, decoded.
pub(super) fn nearest(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    t: &Tys,
    at: &At<'_>,
    tex: &Texture,
    addressing: Addressing<'_>,
) -> BResult<SValue> {
    let taps = Taps {
        addressing,
        offset: at.offset,
        span: &[0],
        base: TapBase::Floor,
        level: None,
    };
    let fp = footprint_of_image(
        cx,
        fb,
        t,
        (at.coord, None),
        (tex.img, tex.image),
        Bits::Value(tex.flags),
        (&taps, tex.parent),
    )?;
    let [(_, xs), (_, ys)] = &fp.axes[..] else {
        return Err(cx.fail("an extended-range nearest sample without two plane axes"));
    };
    let (&[x], &[y]) = (xs.as_slice(), ys.as_slice()) else {
        return Err(cx.fail("an extended-range nearest axis without its tap"));
    };
    // A bfloat form's `half` result is its `float` result's toward zero, border texels included
    // (EV FPV-pf8-m4), where its reads and gathers round to nearest even.
    let site = [x, y];
    match (at.half, tex.decode) {
        (false, _) => texel(cx, fb, &fp, tex, &site, false),
        (true, Decode::Form { form, .. }) if form.bfloat() => {
            let v = texel(cx, fb, &fp, tex, &site, false)?;
            crate::lower::arith::extended_range::bfloat_sample_half(fb, v)
        }
        (true, Decode::Form { .. }) => texel(cx, fb, &fp, tex, &site, true),
        // A heap handle's form: both roundings of the one raw texel, chosen by the form.
        (true, Decode::Heap { bfloat, .. }) => {
            let (raw, inside) = tap(fb, &fp, tex, &site)?;
            let v = decoded(cx, fb, tex, raw, false)?;
            let v = fp.border(fb, v, inside)?;
            let toward = crate::lower::arith::extended_range::bfloat_sample_half(fb, v)?;
            let rounded = decoded(cx, fb, tex, raw, true)?;
            let rounded = fp.border(fb, rounded, inside)?;
            fb.select(bfloat, toward, rounded)
        }
    }
}

/// The gather `e` of `component` of `tex` under `addressing`: the four taps of the linear
/// footprint, each decoded (rounded to `half` for a `half` result) or the border texel, and
/// swizzled by `tex`'s swizzle word where it has one, that component of each in Vulkan's order
/// `(x0, y1)`, `(x1, y1)`, `(x1, y0)`, `(x0, y0)`: a swizzle's `Zero` or `One` 0 or 1 at every tap,
/// another source that channel's taps.
pub(super) fn gathered(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    t: &Tys,
    at: &At<'_>,
    tex: &Texture,
    addressing: Addressing<'_>,
    component: u32,
) -> BResult<SValue> {
    let taps = Taps {
        addressing,
        offset: at.offset,
        span: &[0, 1],
        base: TapBase::Quantized,
        level: None,
    };
    let fp = footprint_of_image(
        cx,
        fb,
        t,
        (at.coord, None),
        (tex.img, tex.image),
        Bits::Value(tex.flags),
        (&taps, tex.parent),
    )?;
    let [(_, xs), (_, ys)] = &fp.axes[..] else {
        return Err(cx.fail("an extended-range gather without two plane axes"));
    };
    let (&[x0, x1], &[y0, y1]) = (xs.as_slice(), ys.as_slice()) else {
        return Err(cx.fail("an extended-range gather axis without two taps"));
    };
    let mut lanes = Vec::with_capacity(4);
    for site in [[x0, y1], [x1, y1], [x1, y0], [x0, y0]] {
        let v = texel(cx, fb, &fp, tex, &site, at.half)?;
        let v = swizzled(fb, tex, v)?;
        lanes.push(fb.composite_extract(v, &[component])?);
    }
    fb.composite_construct(t.vec4, &lanes)
}

/// The value of the tap at `site`: the raw texel fetched from the base level, decoded
/// (`extended_range::decode`, rounded to `half` when `half`), or the border texel where the tap
/// lies outside a border axis.
fn texel(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    fp: &Footprint,
    tex: &Texture,
    site: &[Tap],
    half: bool,
) -> BResult<SValue> {
    let (raw, inside) = tap(fb, fp, tex, site)?;
    let v = decoded(cx, fb, tex, raw, half)?;
    fp.border(fb, v, inside)
}

/// The raw texel of the tap at `site`, fetched from the base level, and whether the tap lies
/// inside every border axis (none where no axis has a border).
fn tap(
    fb: &mut FunctionBuilder<'_>,
    fp: &Footprint,
    tex: &Texture,
    site: &[Tap],
) -> BResult<(SValue, Option<SValue>)> {
    let (at, inside) = fp.site(fb, site)?;
    Ok((fp.read(fb, tex.img, at)?, inside))
}

/// Raw texel `raw` decoded by `tex`'s form (`extended_range::decode`, rounded to `half` when
/// `half`), or by the form its heap handle names, through the heap's routine.
fn decoded(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    tex: &Texture,
    raw: SValue,
    half: bool,
) -> BResult<SValue> {
    match tex.decode {
        Decode::Form { form, table } => {
            crate::lower::arith::extended_range::decode(fb, form, raw, table, half)
        }
        Decode::Heap { form, routines, .. } => {
            let routine = match (half, routines) {
                (false, (f, _)) => f,
                (true, (_, Some(h))) => h,
                (true, (_, None)) => return Err(cx.fail("a raw heap sample without its routine")),
            };
            let bool_ = fb.module().ty(crate::spirv::ir::Type::Bool)?;
            let inside = fb.module().constant(Constant::Scalar {
                ty: bool_,
                bits: ConstBits(1),
            })?;
            let inside = fb.module().const_value(inside)?;
            fb.call(routine, &[raw, form, inside])?
                .ok_or_else(missing_routine)
        }
    }
}

/// The bits of `form` the exact routine and the border texel read: `EXACT` and `EXTENDED` (a
/// padded format's `UNORM` and `RGB10A2Unorm`'s packed field instead), `SRGB` for an `_sRGB` twin,
/// its channels (`BGR10_XR` has no alpha), and Metal's border texels through them
/// (`metal::sampler::border_texel`: a channel the format lacks reads 0, alpha 1). A two-plane
/// texel's alpha byte `b` arrives as `b / 255`, which the routine recovers as the code `16 b` on
/// alpha's grid `8 · 510 = 16 · 255`: its truncated terms `b · w` are exact, so the step is
/// Metal's, `⌊Σ 16 b w / 2^16 + 1/2⌋` (EV XRR-tx1-m4).
pub(super) fn flags(form: ExtendedRange) -> u32 {
    use crate::metal::texel_format::PackedTexels;
    let extended = BicubicTexels::EXTENDED;
    // A padded format's lanes sum as `RGB10A2Unorm`'s 10-bit ones (its field's grid `4 · 1023`,
    // each 2D term truncated onto `2^-4` of a step): Metal's linear filter of `R10Unorm_X6` and
    // `RG10Unorm_X12` (EVs XRR2-x6d and -x6e, an Apple M4 and an Apple M3 Max, byte-identical:
    // every channel of 303,104 samples at 256 x 256, and of 110,592 at each of 1298 x 1298,
    // 300 x 77, 17 x 9 and 1 x 1 under clamp-to-edge, normalized and pixel coordinates). Each tap
    // arrives as `c / 1023`, recovered on the scale 1023.
    let padded = BicubicTexels::UNORM | PackedTexels::Rgb10A2.class().0;
    // `R10Unorm_X6_sRGB`'s red sums as a YCbCr field's on the grid `8192`, each tap its decode
    // step `s / 8192` ([`PackedTexels::YCbCr8192`]: each 2D term `s · wx · wy` truncated by
    // `2^7`, the step rounded half up, a `half` result rounded with ties away from zero), Metal's
    // linear filter of it (EVs XRR2-x6ds, -x6es and -x6hs, an Apple M4 and an Apple M3 Max,
    // byte-identical); `RG10Unorm_X12_sRGB`'s reads and filter are `RG10Unorm_X12`'s.
    let padded_srgb = BicubicTexels::UNORM | PackedTexels::YCbCr8192.class().0;
    let (pf, channels, family) = match form {
        ExtendedRange::Bgr10 { srgb: false } => (PixelFormat::BGR10XR, 0b0111, extended),
        ExtendedRange::Bgr10 { srgb: true } => (PixelFormat::BGR10XRSRGB, 0b0111, extended),
        ExtendedRange::Bgra10 { srgb: false } => (PixelFormat::BGRA10XR, 0b1111, extended),
        ExtendedRange::Bgra10 { srgb: true } => (PixelFormat::BGRA10XRSRGB, 0b1111, extended),
        ExtendedRange::Planes { srgb: false } => (PixelFormat::RGB10A82PXR10, 0b1111, extended),
        ExtendedRange::Planes { srgb: true } => (PixelFormat::RGB10A82PXR10SRGB, 0b1111, extended),
        ExtendedRange::R10X6 => (PixelFormat::R10UnormX6, 0b0001, padded),
        ExtendedRange::Rg10X12 => (PixelFormat::RG10UnormX12, 0b0011, padded),
        ExtendedRange::R10X6Srgb => (PixelFormat::R10UnormX6SRGB, 0b0001, padded_srgb),
        ExtendedRange::Rg10X12Srgb => (PixelFormat::RG10UnormX12SRGB, 0b0011, padded),
        // A bfloat form's samples and gathers read its texels unfiltered: its bits name its
        // channels and border texels alone.
        ExtendedRange::R16Bfloat => (PixelFormat::R16Bfloat, 0b0001, 0),
        ExtendedRange::Rg16Bfloat => (PixelFormat::RG16Bfloat, 0b0011, 0),
        ExtendedRange::Rgba16Bfloat => (PixelFormat::RGBA16Bfloat, 0b1111, 0),
    };
    let srgb = if form.srgb() { BicubicTexels::SRGB } else { 0 };
    let bits = BicubicTexels::EXACT | family | srgb | channels << BicubicTexels::PRESENT;
    BicubicTexels(bits)
        .with_border(pf, TextureSwizzleChannels::DEFAULT)
        .0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A four-channel format's border texels are the colors (a two-plane format's too);
    /// `BGR10_XR`'s alpha reads 1 under each color, as its texels' does.
    #[test]
    fn border_texels_follow_the_channels() {
        let border = |form| flags(form) >> BicubicTexels::BORDER & 0xfff;
        // Channel `c`'s bits per color (transparent black, opaque black, opaque white).
        let four = 0b100 | 0b100 << 3 | 0b100 << 6 | 0b110 << 9;
        let three = 0b100 | 0b100 << 3 | 0b100 << 6 | 0b111 << 9;
        for srgb in [false, true] {
            assert_eq!(border(ExtendedRange::Bgra10 { srgb }), four);
            assert_eq!(border(ExtendedRange::Planes { srgb }), four);
            assert_eq!(border(ExtendedRange::Bgr10 { srgb }), three);
            let f = flags(ExtendedRange::Bgra10 { srgb });
            assert_ne!(f & BicubicTexels::EXACT, 0);
            assert_ne!(f & BicubicTexels::EXTENDED, 0);
            assert_eq!(f & BicubicTexels::SRGB != 0, srgb);
        }
    }
}
