//! Metal's linear filter of a cube texture, emulated by fetches.
//!
//! **The filter Metal applies** (Apple M4, macOS 27.0.1, and Apple M3 Max, macOS 26.7,
//! byte-identical, each run with and without Metal's validation layer; with their M3 twins). A
//! sample through a sampler with linear minification and magnification and no mip filter reads the
//! face of its direction's major axis (`face::metal_ties`'s ties) at Metal's position on each face
//! axis, `X = RN(RN(1 + sc * r) / 2 * n)` (`face::position`), quantized as in 2D: `q = floor(256
//! X + 1/2) - 128`, the base texel `i = q >> 8` and the fraction's 1/256 steps `k = q & 255`. The
//! four taps `(i, j)`, `(i + 1, j)`, `(i, j + 1)`, and `(i + 1, j + 1)` weigh `(256 - k)` and `k`
//! per axis, each tap `wx · wy` in units of `2^-16`; an axis's two taps are always two texels, so
//! no axis merges. A tap past one edge of the face reads the adjacent face's texel along that edge
//! (`face::define_fold`). A tap past both edges, at the cube's corner, reads a texel that depends
//! on the format: a 32-bit float or half-float format's is the texel at the corner of the face of
//! the `x` axis; an 8- or 16-bit normalized or 8-bit sRGB format's is made up from the corner
//! texels of the three faces there, `⌊(6 X + 5 Y + 5 Z + 8) / 16⌋` of their codes per channel (an
//! sRGB color channel's table steps, a signed format's least code read as the next one;
//! `bicubic::routines::Cube::corner`). The sum is the 2D sum's: a normalized format's terms
//! truncated onto its grid, a float format's Metal's linear accumulator. Metal reads the sampler's
//! address modes and border color as clamping (the probe's `cm` runs: every lane of 4x4 cubes reads
//! this rule under clamp-to-edge, repeat, mirrored repeat, zero, and a white border), but for
//! `repeat` on a face axis of a size that is not a power of two, which wraps a face coordinate of
//! exactly 1 to 0 (`face::FaceWrap`; our probe's linear samples of 3x3, 5x5, and 6x6
//! cubes), reads level 0 whatever a `level` or `bias` names (the probe's `clv`
//! run, a two-level cube), and clamps a cube array's index into the array as an unsigned integer,
//! so a negative index reads the last cube (its `cla` run).
//!
//! **The emulation** reads the slot's `TexelFormat` specialization as the 2D emulation does
//! (`linear`), and fetches the taps from the slot's 2D-array view of the cube's faces (its raw
//! view under its `SrgbTexture` variant), layer `6 c + face` of cube `c`: per tap the fold's
//! texel, and at the corner candidate the `y` and `z` faces' texels too, six fetches. The corner
//! routine replaces the tap past both edges where there is one, and the 2D routines sum the four
//! taps under the unmerged fractions.
//!
//! **Where it is exact.** Every `float` and `half` sample channel of every lane whose major
//! magnitude is 1: our probe's 21 runs (seven formats, an identity texture among them, on faces of
//! 2, 5, and 8 texels, 4,096 lanes each), our probe's 27 `cl` runs (nine formats on faces of 1, 3,
//! and 16 texels, 2,048 such lanes each) and its `clv` and `cla` runs, and our probe's `cm` runs.
//! Elsewhere Metal's texture unit takes an approximate reciprocal `r` of the major magnitude, which
//! the emulation takes correctly rounded (`PrecisionNote::CubeFaceReciprocal`): of the `cl` runs'
//! 6,144 lanes of `rgba32Float` with a scaled major, 5,957 are the emulation's, and each of the
//! other 187 is its rule under an `r` one or two ulps from `RN(1 / |m_a|)`, a position within about
//! `n * 2^-23` of a quantization step reading the next step. A direction with an infinite or NaN
//! component gives an unspecified footprint, whose fetches stay on the cube.
//!
//! **Gathers** (the probes, Apple M4 on macOS 27.0.1, each run with and without Metal's validation
//! layer). A gather reads these four taps at the same position, in gather order `x` `(i, j + 1)`,
//! `y` `(i + 1, j + 1)`, `z` `(i + 1, j)` and `w` `(i, j)`, whatever the sampler's filters, mip
//! filter and anisotropy, from level 0, and wraps as a sample does. The tap past both edges is the
//! format's: the `x` face's corner texel for a 32-bit float, half-float or integer format, the
//! made-up texel for a normalized one (an sRGB color channel's a step `k`, gathered as
//! `RN(k / 4095)`). A `half` gather is the `float` texel rounded to nearest even. A depth cube's
//! taps are its depth texels, `depth32Float`'s corner the `x` face's and `depth16Unorm`'s made up;
//! a comparison compares each tap with the reference (`compare`). Outside those families the corner
//! is measured for the formats of a `metal::texel_format::CubeCorner` (the probe): the `x` face's
//! texel for `depth32Float_stencil8`'s depth (compared as `depth32Float`'s), the small floats,
//! BC6H, and ASTC HDR, and for the packed normalized and compressed formats the made-up texel of
//! each channel's codes on its decoded grid. [`CubeGather`] answers a cube gather through a
//! constexpr normalized sampler with the same fetches (`fetch_taps`): an integer result's taps as
//! fetched, a `float` or `half` result's corner by the corner routine under the slot's
//! `TexelFormat` specialization, an sRGB variant's from the raw view, each step decoded exactly
//! (`texture::srgb_raw_gather`), and a comparison's taps compared in the shader. Every lane of both
//! probes fits but one scaled lane per run, the reciprocal's residue above. A gather through a
//! runtime sampler slot is answered alike: its face axes wrap by the bound sampler's word's address
//! modes, and a comparison takes the word's compare function; an argument buffer's texture keeps
//! the device's gather.

use crate::air::abi::{CompareFunction, Coordinates, SamplerState};
use crate::air::intrinsics::{ImageDim, ImageShape};
use crate::error::Error;
use crate::lower::emit::EmitCx;
use crate::lower::select::{NoteClause, NoteOn, NoteTest, SelectCx};
use crate::lower::texture::{ImageRef, TexelConv, TexelForm};
use crate::metal::bicubic::BicubicTexels;
use crate::metal::texel_format::texel_format;
use crate::pipeline::sym::SamplerRef;
use crate::spirv::builder::constructs::{ArmEnd, ArmFn, Merged};
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::{ConstId, Value as SValue};
use crate::spirv::ir::{
    BinOp, ConvOp, GlslOp, ImageOperands, ImageQuery, MemAccess, Type as SType, UnOp,
};

use super::super::bicubic::routines::Routine;
use super::super::bicubic::{self, quantize, scalar, value, Bits, Tys};
use super::super::face::{self, FaceRoutines, FaceWrap, Wraps};
use super::super::{SampleOp, Sampling};
use super::compare::{TapCompare, WORD_REFERENCE_WORDS, WORD_TAP_WORDS};
use super::Comparison;
use super::{bool_value, Linear, LinearImage, LinearState};

/// The census words of a cube's emulation beside its face position's routines
/// ([`FaceRoutines::use_words`]) and a raw sample's tap conversions: the view's load and size
/// query, the cube's first layer, `face::position`, the two quantizations, the six folds and
/// fetches, the corner's test, candidate, and call, the four taps' selects, and the two sums'
/// calls and their select (`the_cube_estimate_bounds_its_emission`).
pub(super) const WORDS: u32 = 1024;

/// The texels a cube's emulation fetches: the four taps, and the corner's `y` and `z` faces'.
pub(super) const TAPS: u32 = 6;

/// The plan of a linear cube sample through sampler `s` of texture slot `texture` (module docs):
/// its face position's routines (`face::plan`, which records `PrecisionNote::CubeFaceReciprocal`),
/// the addressing of its face's `s` and `t` axes, which wraps a repeating one (`face::FaceWrap`),
/// and the 2D sums' and the cube's routines. Metal reads a cube's coordinate as a direction through
/// a pixel sampler too, as through a normalized one. A depth cube's comparison compares each tap
/// under `compare`.
pub(super) fn select(
    cx: &mut SelectCx<'_>,
    texture: u32,
    s: &SamplerState,
    compare: Option<Comparison>,
) -> Result<Option<Linear>, Error> {
    let face_address = super::super::position::address;
    let l = Linear {
        image: LinearImage::Slot(texture),
        state: LinearState::Constant {
            address: vec![face_address(s.address_s), face_address(s.address_t)],
            border: s.border_color,
            pixel: false,
        },
        offset: None,
        raw: false,
        cube: Some(face::plan(cx)?),
        compare,
        fetch: None,
        exact_only: false,
        native: false,
    };
    cx.families
        .sampler
        .bicubic_routines
        .extend(l.routines().iter().copied());
    Ok(Some(l))
}

/// The emulation's texel (module docs) of the cube sample `s` by `l` (a runtime sampler's
/// through its `word`), along direction `p` (with Metal's ties broken), its face position's
/// routines `routines`, under the bound format's bits `bits` (the slot's `TexelFormat` value, which
/// a function reads as data), a `half` result when `half`. A depth cube's comparison (`l.compare`,
/// against reference `dref`) compares each tap's depth, the corner's by the format's rule
/// (`Depth32Float`'s the `x` face's texel, `Depth16Unorm`'s the made-up code), and sums the passes
/// as `r8Unorm` codes 0 and 255 through the exact routine: its value is the sum's first channel, a
/// scalar (3,072 directions, `less` and `greater_equal`, cubes and a cube array,
/// every lane).
pub(super) fn emulate(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    t: &Tys,
    layer: Option<SValue>,
    (l, word): (&Linear, Option<SValue>),
    (routines, p): (FaceRoutines, SValue),
    (bits, half, dref): (Bits, bool, Option<SValue>),
) -> BResult<SValue> {
    let (Some(cube), Some(quad)) = (
        cx.decl.families.sampler.linear_cube,
        cx.decl.families.sampler.linear_quad,
    ) else {
        return Err(cx.fail("a linear cube sample without its routines"));
    };
    let post = |cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>, v: SValue| match l.raw {
        true => crate::lower::texture::srgb_raw_tap(cx, fb, v),
        false => Ok(v),
    };
    let wrap = match (&l.state, word) {
        (LinearState::Constant { address, .. }, _) => Wraps::Axes(FaceWrap::of(
            address.first().copied(),
            address.get(1).copied(),
        )),
        (LinearState::Slot { .. } | LinearState::Argument { .. }, Some(word)) => Wraps::Word(word),
        (LinearState::Slot { .. } | LinearState::Argument { .. }, None) => {
            return Err(cx.fail("a runtime sampler's cube sample without its word"))
        }
    };
    let Some(texture) = l.image.slot() else {
        return Err(cx.fail("a heap texture's cube sample"));
    };
    let site = Site {
        texture,
        raw: l.raw,
        routines,
        wrap,
        p,
    };
    let taps = fetch_taps(cx, fb, t, layer, &site, (true, &post))?;
    let [x, y, z] = taps.corner_texels(cx)?;
    let flags = bits.value(fb)?;
    let flags = match l.raw {
        true => {
            let exact = scalar(fb, t.u32_, u64::from(BicubicTexels::EXACT))?;
            let exact = value(fb, exact)?;
            fb.binary(BinOp::BitwiseOr, flags, exact)?
        }
        false => flags,
    };
    let made = fb
        .call(cube.corner, &[x, y, z, flags])?
        .ok_or_else(bicubic::missing_routine)?;
    let mut args = Vec::with_capacity(8);
    for (tap, here) in taps.texels.into_iter().zip(taps.corner) {
        args.push(fb.select(here, made, tap)?);
    }
    let h = bool_value(fb, t, half)?;
    let compare = match (l.compare, dref) {
        (Some(Comparison::Function(f)), Some(r)) => Some(TapCompare::new(cx, fb, t, f, (r, bits))?),
        (None, None) => None,
        (Some(Comparison::Word), _) => {
            return Err(cx.fail("a linear cube comparison through a runtime sampler"))
        }
        _ => return Err(cx.fail("a linear cube comparison without its reference")),
    };
    if let Some(c) = compare {
        for tap in args.iter_mut() {
            let depth = fb.composite_extract(*tap, &[0])?;
            let pass = c.pass(fb, depth)?;
            *tap = fb.composite_construct(t.vec4, &[pass; 4])?;
        }
        let pass = texel_format(super::compare::PASS_FORMAT).0;
        let pass = scalar(fb, t.u32_, u64::from(pass))?;
        args.extend([taps.kx, taps.ky, value(fb, pass)?, h]);
        let e = fb
            .call(quad.exact, &args)?
            .ok_or_else(bicubic::missing_routine)?;
        return fb.composite_extract(e, &[0]);
    }
    args.extend([taps.kx, taps.ky, flags, h]);
    if l.raw {
        return fb
            .call(quad.exact, &args)?
            .ok_or_else(bicubic::missing_routine);
    }
    // One sum runs, by a branch on the class, where both ran and a select took one (the linear
    // sum's rule, `linear::sum`).
    let is_exact = bits.test(fb, t, BicubicTexels::EXACT)?;
    let float_args = args.clone();
    let exact_arm: ArmFn<'_, EmitCx<'_>> = Box::new(move |fb, _| {
        let e = fb
            .call(quad.exact, &args)?
            .ok_or_else(bicubic::missing_routine)?;
        Ok(ArmEnd::FallThrough(vec![e]))
    });
    let float_arm: ArmFn<'_, EmitCx<'_>> = Box::new(move |fb, _| {
        let f = fb
            .call(quad.float, &float_args)?
            .ok_or_else(bicubic::missing_routine)?;
        Ok(ArmEnd::FallThrough(vec![f]))
    });
    let Merged(merged) = fb.if_(cx, is_exact, exact_arm, Some(float_arm))?;
    match merged[..] {
        [x] => Ok(x),
        _ => Err(cx.fail("a linear cube sum's branch without its texel")),
    }
}

/// Where a cube footprint's taps are fetched: texture slot `texture`'s view of its faces (its raw
/// view where `raw`), at Metal's position along direction `p` (its ties broken) by the face
/// position's routines `routines`, wrapped on the face axes `wrap` names.
struct Site {
    texture: u32,
    raw: bool,
    routines: FaceRoutines,
    wrap: Wraps,
    p: SValue,
}

/// A cube footprint's fetched texels (module docs).
struct Taps {
    /// The taps `(i, j)`, `(i + 1, j)`, `(i, j + 1)`, and `(i + 1, j + 1)`, each its fold's
    /// texel: past both edges, the `x` face's corner texel.
    texels: [SValue; 4],
    /// Per tap whether it is the one past both edges (a `bool`).
    corner: [SValue; 4],
    /// The `y` and `z` faces' texels at the cube's corner the tap past both edges would read,
    /// where fetched.
    yz: Option<[SValue; 2]>,
    /// The candidate's own tap, the `x` face's corner texel.
    x: SValue,
    /// The fractions' 1/256 steps on the face's `s` and `t` axes.
    kx: SValue,
    ky: SValue,
}

impl Taps {
    /// The three faces' texels at the cube's corner: `x`, `y`, and `z`.
    fn corner_texels(&self, cx: &mut EmitCx<'_>) -> BResult<[SValue; 3]> {
        match self.yz {
            Some([y, z]) => Ok([self.x, y, z]),
            None => Err(cx.fail("a cube corner's texels not fetched")),
        }
    }
}

/// The four taps of the cube footprint of a sampling of cube array layer `layer` (`None`: a cube)
/// at `site` (module docs), each fetched texel passed through `post`, and where `faces` the `y`
/// and `z` faces' corner texels beside them.
fn fetch_taps(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    t: &Tys,
    layer: Option<SValue>,
    site: &Site,
    (faces, post): (bool, &Post<'_>),
) -> BResult<Taps> {
    let view = match site.raw {
        true => cx.decl.families.texture.raw.get(&site.texture).copied(),
        false => cx
            .decl
            .families
            .texture
            .images
            .get(&site.texture)
            .map(|(g, _)| *g),
    };
    let Some(view) = view else {
        return Err(cx.fail(format!(
            "texture slot {} has no view of its cube's faces",
            site.texture
        )));
    };
    let Some(cube) = cx.decl.families.sampler.linear_cube else {
        return Err(cx.fail("a cube footprint without its fold"));
    };
    let routines = site.routines.declared(cx)?;
    let i32_ = fb.module().ty(SType::Int {
        width: 32,
        signed: true,
    })?;
    let u = |fb: &mut FunctionBuilder<'_>, v: u32| -> BResult<SValue> {
        let c = scalar(fb, t.u32_, u64::from(v))?;
        value(fb, c)
    };
    let zero = u(fb, 0)?;
    let ptr = fb.module().global_value(view)?;
    let img = fb.load(ptr, MemAccess::default())?;
    let size = fb.query(ImageQuery::SizeLod(zero), img)?;
    let n = fb.composite_extract(size, &[0])?;
    // The cube's first layer: its index clamped into the array, as Metal clamps it, times 6.
    let six = u(fb, 6)?;
    let base = match layer {
        Some(index) => {
            let layers = fb.composite_extract(size, &[2])?;
            let cubes = fb.binary(BinOp::UDiv, layers, six)?;
            let one = u(fb, 1)?;
            let last = fb.binary(BinOp::ISub, cubes, one)?;
            let c = fb.glsl(GlslOp::UMin, &[index, last])?;
            fb.binary(BinOp::IMul, c, six)?
        }
        None => zero,
    };
    let at = face::position(fb, site.p, n, (routines, site.wrap))?;
    let (i, kx) = quantize(fb, t, at.s, None)?;
    let (j, ky) = quantize(fb, t, at.t, None)?;
    let ni = fb.convert(ConvOp::Bitcast, n, i32_)?;
    let one = {
        let c = scalar(fb, i32_, 1)?;
        value(fb, c)?
    };
    let i1 = fb.binary(BinOp::IAdd, i, one)?;
    let j1 = fb.binary(BinOp::IAdd, j, one)?;
    let lod = ImageOperands {
        lod: Some(zero),
        ..ImageOperands::default()
    };
    let fetch = |cx: &mut EmitCx<'_>,
                 fb: &mut FunctionBuilder<'_>,
                 (ti, tj): (SValue, SValue),
                 pick: u32|
     -> BResult<SValue> {
        let pick = u(fb, pick)?;
        let site = fb
            .call(cube.fold, &[at.face, ti, tj, ni, pick, base])?
            .ok_or_else(bicubic::missing_routine)?;
        let texel = fb.fetch(img, site, &lod)?;
        post(cx, fb, texel)
    };
    let t00 = fetch(cx, fb, (i, j), 0)?;
    let t10 = fetch(cx, fb, (i1, j), 0)?;
    let t01 = fetch(cx, fb, (i, j1), 0)?;
    let t11 = fetch(cx, fb, (i1, j1), 0)?;
    // The tap that may lie past both edges: per axis the tap past its edge, `i` where it is
    // negative and `i + 1` otherwise (the base tap lies in `[-1, n - 1]`).
    let zero_i = {
        let c = scalar(fb, i32_, 0)?;
        value(fb, c)?
    };
    let low_i = fb.binary(BinOp::SLessThan, i, zero_i)?;
    let low_j = fb.binary(BinOp::SLessThan, j, zero_i)?;
    let past = |fb: &mut FunctionBuilder<'_>, low: SValue, next: SValue| -> BResult<SValue> {
        let high = fb.binary(BinOp::SGreaterThanEqual, next, ni)?;
        fb.binary(BinOp::LogicalOr, low, high)
    };
    let past_i = past(fb, low_i, i1)?;
    let past_j = past(fb, low_j, j1)?;
    let corner = fb.binary(BinOp::LogicalAnd, past_i, past_j)?;
    let yz = match faces {
        true => {
            let ci = fb.select(low_i, i, i1)?;
            let cj = fb.select(low_j, j, j1)?;
            Some([fetch(cx, fb, (ci, cj), 1)?, fetch(cx, fb, (ci, cj), 2)?])
        }
        false => None,
    };
    // The candidate's own tap is its `x` face's texel.
    let row_j = fb.select(low_i, t00, t10)?;
    let row_j1 = fb.select(low_i, t01, t11)?;
    let x = fb.select(low_j, row_j, row_j1)?;
    let high_i = fb.unary(UnOp::LogicalNot, low_i)?;
    let high_j = fb.unary(UnOp::LogicalNot, low_j)?;
    let mut here = Vec::with_capacity(4);
    for (on_i, on_j) in [
        (low_i, low_j),
        (high_i, low_j),
        (low_i, high_j),
        (high_i, high_j),
    ] {
        let tap = fb.binary(BinOp::LogicalAnd, on_i, on_j)?;
        here.push(fb.binary(BinOp::LogicalAnd, corner, tap)?);
    }
    let [c00, c10, c01, c11] = here[..] else {
        return Err(cx.fail("a cube footprint without its four taps"));
    };
    Ok(Taps {
        texels: [t00, t10, t01, t11],
        corner: [c00, c10, c01, c11],
        yz,
        x,
        kx,
        ky,
    })
}

/// How a fetched texel becomes a tap: a raw sample's sRGB conversion, or the texel as fetched.
type Post<'a> = dyn Fn(&mut EmitCx<'_>, &mut FunctionBuilder<'_>, SValue) -> BResult<SValue> + 'a;

// ---- Gathers ----------------------------------------------------------------------------------

/// A gather of a cube or cube array that Metal's footprint, emulated by fetches, answers (module
/// docs): the four taps of the linear filter at Metal's position, in gather order, fetched from the
/// slot's view of its faces.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CubeGather {
    /// The texture slot gathered, whose faces are fetched and whose `TexelFormat` specialization
    /// states a `float` or `half` result's corner rule.
    pub texture: u32,
    /// The gather reads an 8-bit sRGB texture's raw codes, its slot in the `SrgbTexture` variant:
    /// each tap Metal's table step, the corner a made-up step.
    pub raw: bool,
    /// The face position's routines.
    pub face: FaceRoutines,
    /// The face axes the sampler repeats: a constexpr sampler's, or a runtime sampler's by its
    /// word.
    pub wrap: GatherWrap,
    /// An integer result: every tap a texel, the corner's the `x` face's, whatever the format.
    pub integer: bool,
    /// A comparison gather of a depth texture: its sampler's compare function (a runtime
    /// sampler's by its word), under which each tap's texel (the corner's by the format's rule)
    /// is compared with the reference, `1` where it passes and `0` elsewhere.
    pub compare: Option<Comparison>,
}

/// The face axes a cube gather's sampler repeats (`face::FaceWrap`): a constexpr sampler's, or a
/// runtime sampler's, whose word's `s` and `t` address modes the function reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GatherWrap {
    Axes(FaceWrap),
    Word,
}

impl CubeGather {
    /// Where the gather leaves its footprint to the device
    /// (`PrecisionNote::DeviceGatherFootprint`): a `float` or `half` result's on a slot bound in a
    /// format whose class states neither a measured family nor a corner rule (`linear::gather`'s
    /// device arm). An integer result and a raw gather fetch every tap, with no device arm.
    pub(crate) fn footprint_when(&self) -> Vec<NoteClause> {
        if self.integer || self.raw {
            return Vec::new();
        }
        vec![[NoteTest::equal(
            NoteOn::Texel(self.texture),
            BicubicTexels::EXACT
                | BicubicTexels::LINEAR
                | BicubicTexels::HALF
                | crate::metal::texel_format::TexelFormat::CORNER_MASK,
            0,
        )]
        .into()]
    }

    /// The census words beyond the device's gather: the footprint ([`GATHER_WORDS`]), its face
    /// position's routine uses and wrap, a raw gather's corner conversions and decode, and a
    /// comparison's ([`COMPARE_WORDS`]).
    pub(in crate::lower::sampler) fn words(&self) -> u32 {
        let raw = match self.raw {
            true => 3 * crate::lower::texture::SRGB_TAP_WORDS + RAW_CORNER_WORDS,
            false => 0,
        };
        let compare = match self.compare {
            Some(Comparison::Function(_)) => COMPARE_WORDS,
            Some(Comparison::Word) => COMPARE_WORDS + WORD_REFERENCE_WORDS + 4 * WORD_TAP_WORDS,
            None => 0,
        };
        let wrap = match self.wrap {
            GatherWrap::Axes(w) => w.words(),
            GatherWrap::Word => face::WORD_WRAP_WORDS,
        };
        GATHER_WORDS + self.face.use_words() + wrap + raw + compare
    }
}

/// The census words of a cube gather's footprint beside its face position's routines: the view's
/// load and size query, the cube's first layer, `face::position`, the two quantizations, the six
/// folds and fetches, the corner's test, candidate, and call, the four channels' extractions and
/// selects, the result's assembly, and the format's branch
/// (`the_cube_gather_estimate_bounds_its_emission`).
const GATHER_WORDS: u32 = 1024;

/// A raw gather's words for its corner's step: its extraction, scaling, rounding, conversion,
/// splat, and the four lanes' selects.
const RAW_CORNER_WORDS: u32 = 96;

/// A comparison gather's words: its reference's operand, the constants, the reference's code, and
/// per tap the texel's code, the comparison, and its select (`compare`).
const COMPARE_WORDS: u32 = 256;

/// The plan of gather `op` of `image` (shape `shape`) through the static sampler `stat`, with
/// texel form `form` (`observed`: the function observes its residency), where the emulation reaches
/// it (module docs): a gather of a cube or cube array slot through a constexpr sampler (with
/// normalized or pixel coordinates, which a cube reads alike) or a runtime sampler slot
/// (whose word's address modes wrap the face axes and whose compare function a comparison takes),
/// whose residency is not observed, on a device without `VkProperty::MetalSampler` (where Metal's
/// own sampler gathers): a color one with a `float`, `half`, or integer result, a depth one, and a
/// depth one's comparison under its sampler's compare function. It records
/// `PrecisionNote::CubeFaceReciprocal` (`face::plan`); a `float` or `half` result's
/// `PrecisionNote::DeviceGatherFootprint`, where its format's corner rule is not measured (its
/// gather keeps the device's footprint), is the caller's once it knows whether the gather reads raw
/// codes (`CubeGather::footprint_when`). Any other gather (an argument buffer's texture's among
/// them) is `None`.
pub(in crate::lower::sampler) fn select_gather(
    cx: &mut SelectCx<'_>,
    op: &SampleOp,
    form: Option<TexelForm>,
    (stat, image, shape): (Option<(SamplerRef, Coordinates)>, ImageRef, ImageShape),
    observed: bool,
) -> Result<Option<CubeGather>, Error> {
    let SampleOp::Gather { dref, .. } = op else {
        return Ok(None);
    };
    // Metal compares a depth texture's texels only.
    if shape.dim != ImageDim::Cube || shape.ms || observed || (dref.is_some() && !shape.depth) {
        return Ok(None);
    }
    // A pixel sampler's cube coordinate is a direction too.
    let (Some((sampler, _)), ImageRef::Slot(texture)) = (stat, image) else {
        return Ok(None);
    };
    let integer = match form.map(|f| f.conv) {
        Some(TexelConv::F32 | TexelConv::F16) => false,
        Some(TexelConv::I32 | TexelConv::I16 { .. }) => true,
        _ => return Ok(None),
    };
    if cx.profile.device_samples() {
        cx.families.sampler.device_sampling = true;
        return Ok(None);
    }
    let (wrap, compare) = match sampler {
        SamplerRef::Constexpr(words) => {
            let s = super::super::state_of(words)?;
            // The frontend encodes an unspecified comparison as `never` and writes no `None`.
            let compare = match (dref, s.compare) {
                (None, _) => None,
                (Some(_), CompareFunction::None) => return Ok(None),
                (Some(_), f) => Some(Comparison::Function(f)),
            };
            let face_address = super::super::position::address;
            let wrap = FaceWrap::of(
                Some(face_address(s.address_s)),
                Some(face_address(s.address_t)),
            );
            (GatherWrap::Axes(wrap), compare)
        }
        // The function reads the word's `s` and `t` address modes, and a comparison its compare
        // function.
        SamplerRef::Slot(n) => {
            let Some(slot) = cx.families.sampler.slots.get_mut(&n) else {
                return Err(cx.invariant("an unplanned sampler slot"));
            };
            slot.faces = true;
            slot.compared |= dref.is_some();
            (GatherWrap::Word, dref.map(|_| Comparison::Word))
        }
    };
    let g = CubeGather {
        texture,
        raw: false,
        face: face::plan(cx)?,
        wrap,
        integer,
        compare,
    };
    cx.families.sampler.bicubic_routines.insert(Routine::Cube);
    Ok(Some(g))
}

/// The gathered texel of cube gather `s` by `g` of channel `component` (module docs), along
/// direction `p` (its ties broken), its face position's routines `routines`: the four taps'
/// channel in gather order, `x` `(i, j + 1)`, `y` `(i + 1, j + 1)`, `z` `(i + 1, j)`, and `w`
/// `(i, j)`. An integer result's taps are the fetched texels (past both edges the `x` face's).
/// A `float` or `half` result's tap past both edges is the corner routine's under the slot's
/// `TexelFormat` specialization `spec`: a normalized format's made-up texel, a float format's `x`
/// face's, a `CubeCorner`'s by its field. The value is the view's texel as a device gather returns
/// it (a `vec4` of its sampled type), but for a raw gather, whose value is the held texel of
/// `form`: each tap's Metal decode (`texture::srgb_raw_texel`'s), and the corner's made-up step
/// over 4095 (alpha's code over 255); and for a comparison, whose value is per tap `1` where
/// reference `dref` passes `g.compare` against the tap's texel and `0` elsewhere (`TapCompare`).
pub(super) fn gather(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    t: &Tys,
    s: &Sampling,
    g: &CubeGather,
    (p, component, dref): (SValue, u32, Option<SValue>),
    spec: Option<ConstId>,
) -> BResult<SValue> {
    // A runtime sampler's word (its slot's `SamplerState` specialization), where its address modes
    // wrap the face axes or its compare function compares.
    let word = match (g.wrap, g.compare) {
        (GatherWrap::Word, _) | (_, Some(Comparison::Word)) => {
            match super::super::emit::word_pixel(s) {
                Some(pixel) => Some(super::super::emit::sampler_word(cx, fb, pixel)?),
                None => return Err(cx.fail("a runtime sampler's cube gather without its word")),
            }
        }
        _ => None,
    };
    let wrap = match (g.wrap, word) {
        (GatherWrap::Axes(w), _) => Wraps::Axes(w),
        (GatherWrap::Word, Some(w)) => Wraps::Word(w),
        (GatherWrap::Word, None) => {
            return Err(cx.fail("a runtime sampler's cube gather without its word"))
        }
    };
    let site = Site {
        texture: g.texture,
        raw: g.raw,
        routines: g.face,
        wrap,
        p,
    };
    let layer = match s.layer {
        Some(l) => Some(cx.operand(fb, l)?),
        None => None,
    };
    let taps = fetch_taps(cx, fb, t, layer, &site, (spec.is_some(), &|_, _, v| Ok(v)))?;
    // The taps in gather order.
    let order = [
        taps.texels[2],
        taps.texels[3],
        taps.texels[1],
        taps.texels[0],
    ];
    let corner = [
        taps.corner[2],
        taps.corner[3],
        taps.corner[1],
        taps.corner[0],
    ];
    let mut lanes = Vec::with_capacity(4);
    for tap in order {
        lanes.push(fb.composite_extract(tap, &[component])?);
    }
    let fetched = fb.value_ty(order[0])?;
    let Some(spec) = spec else {
        return fb.composite_construct(fetched, &lanes);
    };
    let Some(cube) = cx.decl.families.sampler.linear_cube else {
        return Err(cx.fail("a cube gather without its corner routine"));
    };
    let [x, y, z] = taps.corner_texels(cx)?;
    let flags = Bits::Spec(spec).value(fb)?;
    let (x, y, z, flags) = match g.raw {
        true => {
            let tap = crate::lower::texture::srgb_raw_tap;
            let exact = scalar(fb, t.u32_, u64::from(BicubicTexels::EXACT))?;
            let exact = value(fb, exact)?;
            let flags = fb.binary(BinOp::BitwiseOr, flags, exact)?;
            (tap(cx, fb, x)?, tap(cx, fb, y)?, tap(cx, fb, z)?, flags)
        }
        false => (x, y, z, flags),
    };
    let made = fb
        .call(cube.corner, &[x, y, z, flags])?
        .ok_or_else(bicubic::missing_routine)?;
    let made = fb.composite_extract(made, &[component])?;
    if !g.raw {
        for (lane, here) in lanes.iter_mut().zip(corner) {
            *lane = fb.select(here, made, *lane)?;
        }
        if let Some(c) = g.compare {
            let Some(r) = dref else {
                return Err(cx.fail("a comparison cube gather without its reference"));
            };
            let compare = TapCompare::of(cx, fb, t, (c, word), (r, Bits::Spec(spec)))?;
            for lane in lanes.iter_mut() {
                *lane = compare.pass(fb, *lane)?;
            }
        }
        return fb.composite_construct(fetched, &lanes);
    }
    let Some(form) = s.texel else {
        return Err(cx.fail("a raw cube gather without its texel form"));
    };
    let codes = fb.composite_construct(fetched, &lanes)?;
    crate::lower::texture::srgb_raw_gather(cx, fb, form, (codes, component), (made, corner))
}
