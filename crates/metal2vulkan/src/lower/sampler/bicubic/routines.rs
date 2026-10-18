//! The bicubic filter's arithmetic after the fetches, as module routines that every bicubic sample
//! of the module calls: one definition per module, whatever the number of samples.
//!
//! Each routine takes the bound format's `BicubicTexels` bits as an argument, the texture slot's
//! specialization constant at every call, so once a device specializes the module and inlines a
//! call, the paths of the other formats fold away. A sample calls both the exact and the emulated
//! routine of its dimension and selects on `BicubicTexels::EXACT`, a specialization constant
//! operation: the routine its format does not take is dead after specialization (a 2D sample
//! selects among its format classes' routines the same way, [`Plane::sums`]). Each routine also
//! takes whether the sample returns `half`, a constant at every call. Each is started by
//! `ModuleBuilder::routine`, so on a profile whose driver converts every function to one inlined
//! at each call (`DriverWorkaround::FunctionsInlinedAtEveryCall`, MoltenVK 1.4.1) it states
//! `DontInline`; a converter that honoured it would keep the calls and fold only the selection,
//! compiling each routine once for every format.
//!
//! - [`Plane`] (a 2D sample): `weights` is Metal's weight table at a fraction
//!   (`metal::bicubic::weights`), in integer operations.
//!   `exact` is the `EXACT` formats' sum: each tap's code recovered from
//!   its normalized value, the codes weighed and summed exactly in two-word integers (a row's
//!   four products through `OpSMulExtended` and `OpIAddCarry`, the rows' through the two-word
//!   multiply), rounded half up onto the format's grid, clamped, and divided by the grid exactly
//!   ([`texel_round::ratio_to_f32`], or [`texel_round::ratio_to_half`] for a `half` result).
//!   Every other format's result is its class's: a half format's
//!   (`HALF`) Catmull-Rom through its two-pass accumulator (below) in
//!   32-bit integer operations on the taps' bits, `row` on each row as the sample's loop over
//!   the rows reads it and `half` on the rows' sums; a 32-bit float format's (`LINEAR`)
//!   bilinear over the inner taps through Metal's linear accumulator, `linear` ([`Quad`]'s
//!   `float`); and any other format's Catmull-Rom in `f32` operations with `NoContraction` and
//!   the format's clamp, `cubic`. The half and 32-bit float formats' non-finite results are
//!   Metal's, decided on the taps' bits, never by float operations a device may assume finite.
//! - [`Line`] (a 1D sample, bilinear over taps `i` and `i + 1` whatever the format): `exact`
//!   sums the two codes exactly in one word. `float` sums a 32-bit float or half format's two
//!   taps through Metal's linear accumulator (`accumulate`), and any other format's in `f32`
//!   operations. Its non-finite results follow IEEE on the taps of nonzero weight: read for a
//!   32-bit float format, inferred for a half one.
//! - [`Span`] (a linear 1D sample's emulation, bilinear over taps `i` and `i + 1`). `exact` is
//!   [`Line::exact`]. `float` is a 32-bit float or half format's sum through Metal's linear
//!   accumulator (`accumulate`), and Metal's non-finite results on the taps of nonzero weight.
//! - [`Quad`] (a linear 2D sample's emulation, bilinear over the footprint's taps `(i, j)`, `(i +
//!   1, j)`, `(i, j + 1)`, and `(i + 1, j + 1)`). `exact` weighs each tap `wx · wy` in units of
//!   `2^-16`, the integer weights `256 - k` and `k` per axis (a merged axis's fraction arrives as
//!   0), and truncates each code's term before the half-up step, as Metal's 2D sums do: onto `2^-4`
//!   of a 12-bit grid, which moves only an sRGB color lane's steps (an 8-bit format's `16 · code`
//!   term is already on it), and onto `2^-8` of a 16-bit format's codes; an
//!   extended-range format's onto `2^-7` of a color lane's steps and `2^-4` of alpha's
//!   (`BicubicTexels::EXTENDED`); a packed field's by its own `s`
//!   (`PackedTexels::filter_truncation`: 7 on YCbCr's grid `2048`). `float` is a 32-bit float or
//!   half format's sum through Metal's linear accumulator under the weights' products, and Metal's
//!   non-finite results on the taps of nonzero weight. [`define_quad_steps`] is `exact` over taps
//!   given as the integers `exact` recovers from their values, an extended-range sample's grid
//!   steps from its codes ([`Routine::QuadSteps`]).
//! - [`Cube`] (a linear cube sample's emulation, beside [`Quad`]'s sums): `fold` is the texel a tap
//!   reads across the faces' edges, and `corner` the tap past both edges of its face, Metal's
//!   made-up texel of an exact format's codes.
//! - [`Block`] (a linear 3D sample's emulation, over the footprint's eight corners). Both routines
//!   weigh each tap `wx · wy · wz / 256` rounded to nearest in units of `2^-16`, a tie up on the
//!   `t` axis's first tap in Metal's frame (`block_weights`). `exact` truncates each code's term
//!   onto `2^-4` of a 12-bit grid or `2^-8` of a 16-bit format's codes before the half-up step, as
//!   Metal's 3D sums do. `float` is Metal's linear accumulator under those weights, and Metal's
//!   non-finite results on the taps of nonzero product weight (a tap whose weight rounds to zero
//!   among them).
//!
//! Metal's linear accumulator (every lane of the P2a and P2c probes) truncates each term `v · w`
//! toward zero onto a grid `b` bits below the largest present term's `e(v) + e(w)` (the tap's and
//! the weight's exponents; 38 bits for a 32-bit float format, 17 for a half one), sums the
//! truncated terms exactly, and rounds the sum once with ties away from zero, to `f32` (a denormal
//! `+0`) or `half` (a zero `+0`). It is not the exact sum: under cancellation the truncated bits
//! show. A 1D bicubic sample, and a 32-bit float format's 2D one, sum through it (every lane of the
//! P2d probe and of the float fixtures). A half format's 2D Catmull-Rom sum takes it twice
//! (`define_half_row`, `define_half_plane`): each row's terms onto a grid 17 bits below the row's
//! reference (a unit weight's exponent read as `-1`), then the rows' sums under their weights onto
//! a grid 20 bits below the largest row reference plus its weight's exponent, and one rounding to
//! `half`. Neither model was fitted on the held-out P2e and P2f probes, and both answer every lane
//! of them. An axis whose four taps read one texel (indices clamped onto one past an edge) is
//! weighed as that texel alone at the unit weight, Metal's weights at fraction 0, not as four
//! copies of it under the fraction's weights, whose terms the grids truncate apart (every lane of a
//! probe on the half and small-float formats; [`Plane::sums`]'s `whole`).
//!
//! The codes are recovered exactly: Vulkan converts a normalized code `c` to `c / (2^b - 1)`
//! rounded to one of the two nearest `f32` values (or exactly to `±1`), so `v · D` on the grid
//! `D` (the code scale `2^b - 1`, times 16 for an 8-bit format) lies within `0.006` of the
//! integer `c · D / (2^b - 1)`, and `RoundEven` recovers it. A signed format's code `-2^(b-1)`
//! reads as `-1` there, the code `-(2^(b-1) - 1)`: Metal's sum reads it the same way. An 8-bit sRGB
//! format's raw taps (`BicubicTexels::SRGB`) arrive as `k / 4095` for the color code's table step
//! `k` and `c / 255` for the alpha code `c`, recovered the same way on the grids `4095` and `16 ·
//! 255`. An extended-range format's taps (`BicubicTexels::EXTENDED`) arrive as their decoded
//! values, `(c - 384) / 510` correctly rounded (an `_sRGB` twin's color lanes `±s / 8192`, exact),
//! recovered on the grids `16 · 510` (`8192`) and `8 · 510`: the error of `v · D` stays below
//! `0.001` of the integer. Their steps reach past `D` (to `2D`), so the quotient divides by `2D`
//! and doubles, exactly. A packed normalized format's lanes
//! (`metal::texel_format::PackedTexels::filter_grids`) sum on their own grids, `16 · 255`, `4 ·
//! 1023`, or `4095`, as an 8-bit format's do (the 2D and 3D terms truncated onto `2^-4` of a step),
//! each tap's code recovered on its channel's code scale `S` and multiplied by `D / S`: a device
//! that widens a 5- or 6-bit channel by replicating its bits misses `c / S` by up to 0.177 of a
//! code, which is many steps of `D`. A YCbCr field's channels (`PackedTexels::YCbCr2048`,
//! `YCbCr8192`, `YCbCrExtended8192`) arrive as their steps `k / D`, exact, and sum on the grid
//! `2048` or `8192` (`S = D`), each 2D term `k · w` truncated by `2^7`; the extended-range field's
//! steps are signed and reach `2D`, so its quotient divides by `2D` and doubles, exactly, and a
//! YCbCr field's `half` result rounds the exact `f32` with ties away from zero.

use crate::lower::arith::legal::{FloatFmt, Form, Held};
use crate::lower::arith::ops::splat;
use crate::lower::arith::texel_round::{self, HalfRounding};
use crate::lower::arith::wide::{self, Pair};
use crate::metal::bicubic::BicubicTexels;
use crate::metal::texel_format::{CubeCorner, PackedTexels, TexelFormat};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn, LoopFn, LoopSpec};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{FuncId, TypeId, Value};
use crate::spirv::ir::{BinOp, ConvOp, GlslOp, MemAccess, Type, UnOp};

/// The routines a module defines for its bicubic samples, by the samples' dimension.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Routine {
    /// A 2D (or 2D array) texture's: [`Plane`].
    Plane,
    /// A 1D (or 1D array) texture's: [`Line`].
    Line,
    /// A linear 2D (or 2D array) sample's emulation: [`Quad`].
    Quad,
    /// A linear 1D (or 1D array) sample's emulation: [`Span`].
    Span,
    /// A linear 3D sample's emulation: [`Block`].
    Block,
    /// A linear cube sample's taps across the faces' edges: [`Cube`], beside [`Routine::Quad`]'s
    /// sums.
    Cube,
    /// A linear mip filter's blend of an extended-range sample's two levels: [`define_levels`].
    Levels,
    /// A linear 2D (or 2D array) extended-range sample's exact sum of its taps' grid steps:
    /// [`define_quad_steps`].
    QuadSteps,
}

impl Routine {
    /// An upper estimate of the routines' SPIR-V words, counted once per module. Measured 9062
    /// words for [`Routine::Plane`] and 3663 for [`Routine::Line`] since the extended-range grids,
    /// and 4954 for [`Routine::Quad`], 3750 for [`Routine::Span`], and 7839 for [`Routine::Block`]
    /// since YCbCr's grid and truncation (`bicubic_routines_words_bound_their_emission`).
    pub fn words(self) -> u32 {
        match self {
            Routine::Plane => 9216,
            Routine::Line => 3712,
            Routine::Quad => 5184,
            Routine::Span => 3968,
            Routine::Block => BLOCK_WORDS,
            Routine::Cube => CUBE_WORDS,
            Routine::Levels => LEVELS_WORDS,
            Routine::QuadSteps => 5184,
        }
    }

    /// The census name of the routines.
    pub fn census_name(self) -> &'static str {
        match self {
            Routine::Plane => "sampler.bicubic.plane",
            Routine::Line => "sampler.bicubic.line",
            Routine::Quad => "sampler.linear.quad",
            Routine::Span => "sampler.linear.span",
            Routine::Block => "sampler.linear.block",
            Routine::Cube => "sampler.linear.cube",
            Routine::Levels => "sampler.linear.levels",
            Routine::QuadSteps => "sampler.linear.quad_steps",
        }
    }
}

/// A 2D sample's routines (module docs), composed by [`Plane::sums`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plane {
    /// `fn(k: u32) -> uvec4`: Metal's four weights at fraction `k / 256`, as `i32` bits, in units
    /// of `2^-16`.
    pub weights: FuncId,
    /// `fn(16 × vec4, wx: uvec4, wy: uvec4, flags: u32, half: bool) -> vec4`: the exact sum of
    /// the footprint `taps[4j + i]` (row `j`, column `i`) under weights `wx` and `wy`.
    pub exact: FuncId,
    /// `fn(t0: vec4, t1: vec4, t2: vec4, t3: vec4, wx: uvec4, smax: uvec4, umax: uvec4) ->
    /// { uvec4, uvec4, uvec4 }`: a half format's row accumulator over one row of the footprint
    /// (the row's sum and reference in one word, and the taps' largest bits so far).
    pub row: FuncId,
    /// `fn(row0 .. row3: uvec4, wy: uvec4, smax: uvec4, umax: uvec4) -> vec4`: a half format's
    /// result from its rows' sums and references.
    pub half: FuncId,
    /// [`Quad::float`]'s routine on the footprint's inner taps: a 32-bit float format's result.
    pub linear: FuncId,
    /// `fn(16 × vec4, wx: uvec4, wy: uvec4, flags: u32, half: bool) -> vec4`: any other format's
    /// result.
    pub cubic: FuncId,
}

/// A 2D sample's results ([`Plane::sums`]).
#[derive(Clone, Copy, Debug)]
pub struct PlaneSums {
    /// The `EXACT` formats' ([`Plane::exact`]).
    pub exact: Value,
    /// Every other format's, by class; `None` for a sample that takes the exact sum alone (a raw
    /// sample, an extended-range texture's).
    pub float: Option<FloatSums>,
}

/// A 2D sample's emulated results by the format's class ([`PlaneSums`]).
#[derive(Clone, Copy, Debug)]
pub struct FloatSums {
    /// A half format's ([`Plane::half`]).
    pub half: Value,
    /// A 32-bit float format's ([`Plane::linear`]).
    pub linear: Value,
    /// Any other format's ([`Plane::cubic`]).
    pub cubic: Value,
}

impl FloatSums {
    /// The result the format takes: `half` where `is_half`, else `linear` where `is_linear`,
    /// else `cubic` (`bool` scalars, the format's `HALF` and
    /// `LINEAR` bits).
    pub fn select(
        self,
        fb: &mut FunctionBuilder<'_>,
        is_half: Value,
        is_linear: Value,
    ) -> BResult<Value> {
        let f = fb.select(is_linear, self.linear, self.cubic)?;
        fb.select(is_half, self.half, f)
    }
}

/// The row of a 2D sample's footprint a read takes ([`FetchRow`]): row `j` (`0 ..= 3`, the `t`
/// axis's taps `j - 1 .. j + 2`), or the row a loop's counter (a `u32` value, `0 ..= 3`) names.
#[derive(Clone, Copy, Debug)]
pub enum RowIndex {
    At(u32),
    Counter(Value),
}

/// Reads one row of a 2D sample's footprint, its four `vec4` taps by column ([`Plane::sums`]).
pub type FetchRow<'a> = dyn FnMut(&mut FunctionBuilder<'_>, RowIndex) -> BResult<[Value; 4]> + 'a;

impl Plane {
    /// A 2D sample's results (module docs) at the fractions `kx` and `ky` (`u32` 1/256 steps),
    /// for the format's `flags` (`u32`) and a `half` result (`bool`), reading the footprint
    /// through `fetch`. A sample of a format known as it is emitted (`exact_only`: a raw sample,
    /// an extended-range texture's) takes the exact sum alone. The 32-bit float formats'
    /// bilinear sum ([`Plane::linear`]) takes the fractions `linear_k`: the sample's own, or, where
    /// the caller computes its small level's accumulated sum by the same call, those it selects
    /// for that level (`bicubic::filter`). `whole` says, per axis (`bool`), whether the axis's
    /// four taps read one texel: a half format's sum weighs that texel alone at the unit weight
    /// there (module docs).
    ///
    /// The sample selects its format's result on the specialization, so a device drops the other
    /// classes' routines and the reads only they take. The exact, 32-bit float, and other
    /// formats' routines take the footprint read row by row in straight-line code. A half
    /// format's takes one loop over the rows ([`Plane::half_sum`]), reading each row once and
    /// carrying the rows' sums and references, so a device that keeps the loop compiles one
    /// row's arithmetic and holds one row's taps at a time, and one that unrolls it reads each
    /// tap once (holding the footprint, and then four inlined rows of it, a half format's sample
    /// took above the 5 s to create on an Intel Iris Xe, ANV).
    pub fn sums(
        self,
        fb: &mut FunctionBuilder<'_>,
        fetch: &mut FetchRow<'_>,
        (kx, ky): (Value, Value),
        (lkx, lky): (Value, Value),
        whole: (Value, Value),
        (flags, half): (Value, Value),
        exact_only: bool,
    ) -> BResult<PlaneSums> {
        let call = |fb: &mut FunctionBuilder<'_>, f: FuncId, args: &[Value]| -> BResult<Value> {
            fb.call(f, args)?.ok_or_else(super::missing_routine)
        };
        let wx = call(fb, self.weights, &[kx])?;
        let wy = call(fb, self.weights, &[ky])?;
        let grid = [
            fetch(fb, RowIndex::At(0))?,
            fetch(fb, RowIndex::At(1))?,
            fetch(fb, RowIndex::At(2))?,
            fetch(fb, RowIndex::At(3))?,
        ];
        let mut args: Vec<Value> = grid.iter().flatten().copied().collect();
        args.extend([wx, wy, flags, half]);
        let exact = call(fb, self.exact, &args)?;
        if exact_only {
            return Ok(PlaneSums { exact, float: None });
        }
        let [_, [_, a11, a12, _], [_, a21, a22, _], _] = grid;
        let linear = call(
            fb,
            self.linear,
            &[a11, a12, a21, a22, lkx, lky, flags, half],
        )?;
        let cubic = call(fb, self.cubic, &args)?;
        // A half format's axis whose four taps read one texel: Metal's weights at fraction 0, the
        // texel at the unit weight and its three copies at none.
        let t = types(fb.module())?;
        let (none, unit) = (u1(fb, 0)?, u1(fb, 0x1_0000)?);
        let alone = fb.composite_construct(t.vu, &[none, unit, none, none])?;
        let hwx = fb.select(whole.0, alone, wx)?;
        let hwy = fb.select(whole.1, alone, wy)?;
        let half_sum = self.half_sum(fb, fetch, hwx, hwy)?;
        Ok(PlaneSums {
            exact,
            float: Some(FloatSums {
                half: half_sum,
                linear,
                cubic,
            }),
        })
    }

    /// A half format's 2D result under the weights `wx` and `wy`: a loop over the rows passes
    /// each through the row accumulator as `fetch` reads it ([`Plane::row`]), keeping the row's
    /// sum and reference (one word a lane) in the slot of its index (the slots shift down one
    /// place a row, so row `j` ends in slot `j`), and the taps' largest bits; then the rows' pass
    /// and the result ([`Plane::half`]).
    fn half_sum(
        self,
        fb: &mut FunctionBuilder<'_>,
        fetch: &mut FetchRow<'_>,
        wx: Value,
        wy: Value,
    ) -> BResult<Value> {
        let t = types(fb.module())?;
        let mem = MemAccess::default();
        let var = |fb: &mut FunctionBuilder<'_>, ty: TypeId, init: Value| -> BResult<Value> {
            let v = fb.local_var(ty, None)?;
            let p = fb.var_ptr(v)?;
            fb.store(p, init, mem)?;
            Ok(p)
        };
        let call = |fb: &mut FunctionBuilder<'_>, f: FuncId, args: &[Value]| -> BResult<Value> {
            fb.call(f, args)?.ok_or_else(super::missing_routine)
        };
        let z = u4(fb, 0)?;
        let rows = [
            var(fb, t.vu, z)?,
            var(fb, t.vu, z)?,
            var(fb, t.vu, z)?,
            var(fb, t.vu, z)?,
        ];
        let least = u4(fb, 0x8000_0000)?;
        let hi = var(fb, t.vu, least)?;
        let uhi = var(fb, t.vu, z)?;
        each_row(fb, &mut *fetch, &mut |fb, [t0, t1, t2, t3], _| {
            let (a, b) = (fb.load(hi, mem)?, fb.load(uhi, mem)?);
            let r = call(fb, self.row, &[t0, t1, t2, t3, wx, a, b])?;
            for (to, from) in rows.iter().zip(rows.iter().skip(1)) {
                let v = fb.load(*from, mem)?;
                fb.store(*to, v, mem)?;
            }
            for (i, p) in (0u32..).zip([rows[3], hi, uhi]) {
                let v = fb.composite_extract(r, &[i])?;
                fb.store(p, v, mem)?;
            }
            Ok(())
        })?;
        let mut args = Vec::with_capacity(7);
        for p in rows {
            args.push(fb.load(p, mem)?);
        }
        args.push(wy);
        args.push(fb.load(hi, mem)?);
        args.push(fb.load(uhi, mem)?);
        call(fb, self.half, &args)
    }
}

/// One pass over a 2D footprint's rows: a loop whose counter `k` (`u32`) runs `0 ..= 3`, each
/// iteration reading row `k` through `fetch` and handing its taps and `k` to `pass`.
fn each_row(
    fb: &mut FunctionBuilder<'_>,
    fetch: &mut FetchRow<'_>,
    pass: &mut dyn FnMut(&mut FunctionBuilder<'_>, [Value; 4], Value) -> BResult<()>,
) -> BResult<()> {
    let mem = MemAccess::default();
    let t = types(fb.module())?;
    let v = fb.local_var(t.u, None)?;
    let counter = fb.var_ptr(v)?;
    let zero = u1(fb, 0)?;
    fb.store(counter, zero, mem)?;
    let body: LoopFn<'_, ()> = Box::new(move |fb, cx, lc| {
        let k = fb.load(counter, mem)?;
        let rows = u1(fb, 4)?;
        let done = bin(fb, BinOp::UGreaterThanEqual, k, rows)?;
        let exit: ArmFn<'_, ()> = Box::new(move |fb, _| {
            lc.break_(fb)?;
            Ok(ArmEnd::Exits)
        });
        fb.if_(cx, done, exit, None)?;
        let one = u1(fb, 1)?;
        let next = bin(fb, BinOp::IAdd, k, one)?;
        fb.store(counter, next, mem)?;
        let row = fetch(fb, RowIndex::Counter(k))?;
        pass(fb, row, k)
    });
    fb.loop_(&mut (), LoopSpec::default(), body)
}

/// A 1D sample's routines (module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Line {
    /// `fn(t0: vec4, t1: vec4, k: u32, flags: u32, half: bool) -> vec4`: the exact sum.
    pub exact: FuncId,
    /// `fn(t0: vec4, t1: vec4, k: u32, flags: u32, half: bool) -> vec4`: the emulated sum.
    pub float: FuncId,
}

/// How a definition shared by the bicubic and the linear samples starts its routine: a bicubic
/// sample's (`Shared`, [`ModuleBuilder::routine`]) or a linear sample's emulation's (`Kept`,
/// [`ModuleBuilder::texel_routine`], which every profile declares `DontInline`, so a driver that
/// honours the control compiles it once rather than inside each sample).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Calls {
    Shared,
    Kept,
}

/// Start `calls`'s kind of routine returning `ret` in `mb`.
fn start(calls: Calls, mb: &mut ModuleBuilder, ret: TypeId) -> BResult<FunctionBuilder<'_>> {
    match calls {
        Calls::Shared => mb.routine(ret),
        Calls::Kept => mb.texel_routine(ret),
    }
}

/// Define the 2D sample's routines in `mb`.
pub fn define_plane(mb: &mut ModuleBuilder) -> BResult<Plane> {
    Ok(Plane {
        weights: define_weights(mb)?,
        exact: define_exact_plane(mb)?,
        row: define_half_row(mb)?,
        half: define_half_plane(mb)?,
        linear: define_float_quad(mb, Calls::Shared)?,
        cubic: define_cubic_plane(mb)?,
    })
}

/// Define the 1D sample's routines in `mb`.
pub fn define_line(mb: &mut ModuleBuilder) -> BResult<Line> {
    Ok(Line {
        exact: define_exact_line(mb, Packed::No, Calls::Shared)?,
        float: define_float_line(mb)?,
    })
}

/// A linear 1D sample's routines (module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    /// `fn(t0: vec4, t1: vec4, k: u32, flags: u32, half: bool) -> vec4`: the exact sum.
    pub exact: FuncId,
    /// `fn(t0: vec4, t1: vec4, k: u32, flags: u32, half: bool) -> vec4`: Metal's accumulated sum.
    pub float: FuncId,
}

/// Define a linear 1D sample's routines in `mb`.
pub fn define_span(mb: &mut ModuleBuilder) -> BResult<Span> {
    Ok(Span {
        exact: define_exact_line(mb, Packed::Field, Calls::Kept)?,
        float: define_float_span(mb)?,
    })
}

/// A linear 2D sample's routines (module docs). The taps are the footprint's row `j`, then row `j +
/// 1`, each from column `i`; `kx` and `ky` are the fractions' 1/256 steps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Quad {
    /// `fn(t00: vec4, t01: vec4, t10: vec4, t11: vec4, kx: u32, ky: u32, flags: u32, half: bool)
    /// -> vec4`: an `EXACT` format's sum, taking a merged axis's
    /// fraction as 0.
    pub exact: FuncId,
    /// `fn(t00: vec4, t01: vec4, t10: vec4, t11: vec4, kx: u32, ky: u32, flags: u32, half: bool)
    /// -> vec4`: Metal's accumulated sum, taking a merged axis's fraction as 0 too.
    pub float: FuncId,
}

/// Define a linear 2D sample's routines in `mb`.
pub fn define_quad(mb: &mut ModuleBuilder) -> BResult<Quad> {
    Ok(Quad {
        exact: define_exact_quad(mb)?,
        float: define_float_quad(mb, Calls::Kept)?,
    })
}

/// A linear 3D sample's routines (module docs). The eight taps are the footprint's corners with the
/// `s` axis's fastest, then the `t` axis's, then the `r` axis's: `(i, j, l)`, `(i + 1, j, l)`, `(i,
/// j + 1, l)`, `(i + 1, j + 1, l)`, then the same at `l + 1`. `kx`, `ky`, and `kz` are the
/// fractions' 1/256 steps, `0` on an axis whose two taps are one texel; `flip` is whether the `t`
/// axis's second tap is the first in Metal's frame (a mirrored axis's reflected half).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Block {
    /// `fn(8 × vec4, kx: u32, ky: u32, kz: u32, flip: bool, flags: u32, half: bool) -> vec4`: the
    /// exact sum.
    pub exact: FuncId,
    /// `fn(8 × vec4, kx: u32, ky: u32, kz: u32, flip: bool, flags: u32, half: bool) -> vec4`:
    /// Metal's accumulated sum.
    pub float: FuncId,
}

/// Define a linear 3D sample's routines in `mb`.
pub fn define_block(mb: &mut ModuleBuilder) -> BResult<Block> {
    Ok(Block {
        exact: define_exact_block(mb)?,
        float: define_float_block(mb)?,
    })
}

/// [`Routine::words`] of [`Routine::Block`].
const BLOCK_WORDS: u32 = 8064;

/// A linear cube sample's routines, beside [`Quad`]'s sums of the four taps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cube {
    /// `fn(face: u32, ti: i32, tj: i32, n: i32, pick: u32, base: u32) -> uvec3`: the 2D-array
    /// coordinate of the texel a tap reads, across the faces' edges (`face::define_fold`).
    pub fold: FuncId,
    /// `fn(x: vec4, y: vec4, z: vec4, flags: u32) -> vec4`: the tap past both edges of a face,
    /// from the texels of the three faces at the cube's corner (`define_corner`).
    pub corner: FuncId,
}

/// Define a linear cube sample's routines in `mb`.
pub fn define_cube(mb: &mut ModuleBuilder) -> BResult<Cube> {
    Ok(Cube {
        fold: super::super::face::define_fold(mb)?,
        corner: define_corner(mb)?,
    })
}

/// [`Routine::words`] of [`Routine::Cube`] (1,523 measured since the extended-range grids).
const CUBE_WORDS: u32 = 1536;

// ---- Types and constants ----

/// The four lanes of a texel.
const LANES: u8 = 4;

fn uform(lanes: u8) -> Form {
    Form {
        held: Held::Int {
            bits: 32,
            container: 32,
        },
        lanes,
    }
}

fn fform(lanes: u8) -> Form {
    Form {
        held: Held::Float(FloatFmt::F32),
        lanes,
    }
}

fn bform(lanes: u8) -> Form {
    Form {
        held: Held::Bool,
        lanes,
    }
}

/// The routines' types.
#[derive(Clone, Copy)]
struct Tys {
    f: TypeId,
    u: TypeId,
    b: TypeId,
    vf: TypeId,
    vu: TypeId,
}

fn types(mb: &mut ModuleBuilder) -> BResult<Tys> {
    let f = mb.ty(Type::Float { width: 32 })?;
    let u = mb.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let b = mb.ty(Type::Bool)?;
    let vf = mb.ty(Type::Vector {
        elem: f,
        len: LANES,
    })?;
    let vu = mb.ty(Type::Vector {
        elem: u,
        len: LANES,
    })?;
    Ok(Tys { f, u, b, vf, vu })
}

fn bin(fb: &mut FunctionBuilder<'_>, op: BinOp, a: Value, b: Value) -> BResult<Value> {
    fb.binary(op, a, b)
}

/// A float operation no device may contract or regroup.
fn exact_op(fb: &mut FunctionBuilder<'_>, op: BinOp, a: Value, b: Value) -> BResult<Value> {
    fb.binary_no_contraction(op, a, b)
}

/// The `u32` scalar `v`.
fn u1(fb: &mut FunctionBuilder<'_>, v: u32) -> BResult<Value> {
    splat(fb, uform(1), u64::from(v))
}

/// The `uvec4` of `v` in every lane.
fn u4(fb: &mut FunctionBuilder<'_>, v: u32) -> BResult<Value> {
    splat(fb, uform(LANES), u64::from(v))
}

/// The `f32` scalar `x`.
fn f1(fb: &mut FunctionBuilder<'_>, x: f32) -> BResult<Value> {
    splat(fb, fform(1), u64::from(x.to_bits()))
}

/// The `vec4` of `x` in every lane.
fn f4(fb: &mut FunctionBuilder<'_>, x: f32) -> BResult<Value> {
    splat(fb, fform(LANES), u64::from(x.to_bits()))
}

/// The vector of scalar `x` in every lane.
fn broadcast(fb: &mut FunctionBuilder<'_>, ty: TypeId, x: Value) -> BResult<Value> {
    fb.composite_construct(ty, &[x; LANES as usize])
}

/// Whether bit `bit` of `flags` (a `u32`) is set.
fn flag(fb: &mut FunctionBuilder<'_>, flags: Value, bit: u32) -> BResult<Value> {
    let m = u1(fb, bit)?;
    let z = u1(fb, 0)?;
    let and = bin(fb, BinOp::BitwiseAnd, flags, m)?;
    bin(fb, BinOp::INotEqual, and, z)
}

/// The function's return value `r`, and the function.
fn finish(mut fb: FunctionBuilder<'_>, r: Value) -> BResult<FuncId> {
    fb.ret(Some(r))?;
    fb.finish()
}

// ---- The weights ----

/// `fn(k: u32) -> uvec4` (module docs): `metal::bicubic::weights` in integer
/// operations. The outer weights are never positive and the inner ones never negative
/// (`metal::bicubic::tests::the_weights_are_the_measured_table`), so each rounds away from zero
/// as `(|a| + 1) >> 1` with its known sign.
fn define_weights(mb: &mut ModuleBuilder) -> BResult<FuncId> {
    let t = types(mb)?;
    let mut fb = mb.routine(t.vu)?;
    let k = fb.param(t.u)?;
    let fb_ = &mut fb;
    let k2 = bin(fb_, BinOp::IMul, k, k)?;
    let k3 = bin(fb_, BinOp::IMul, k2, k)?;
    // The fine ranges: 76 ..= 95, 161 ..= 176, and 255 (as unsigned differences).
    let in_range = |fb: &mut FunctionBuilder<'_>, lo: u32, len: u32| -> BResult<Value> {
        let lo = u1(fb, lo)?;
        let d = bin(fb, BinOp::ISub, k, lo)?;
        let len = u1(fb, len)?;
        bin(fb, BinOp::ULessThanEqual, d, len)
    };
    let r1 = in_range(fb_, 76, 19)?;
    let r2 = in_range(fb_, 161, 15)?;
    let r3 = in_range(fb_, 255, 0)?;
    let fine = bin(fb_, BinOp::LogicalOr, r1, r2)?;
    let fine = bin(fb_, BinOp::LogicalOr, fine, r3)?;
    let (k128, k8) = (u1(fb_, 128)?, u1(fb_, 8)?);
    let cf = bin(fb_, BinOp::IAdd, k3, k128)?;
    let cf = bin(fb_, BinOp::ShiftRightLogical, cf, k8)?;
    let (k2048, k12, k4) = (u1(fb_, 2048)?, u1(fb_, 12)?, u1(fb_, 4)?);
    let cc = bin(fb_, BinOp::IAdd, k3, k2048)?;
    let cc = bin(fb_, BinOp::ShiftRightLogical, cc, k12)?;
    let cc = bin(fb_, BinOp::ShiftLeftLogical, cc, k4)?;
    let c = fb_.select(fine, cf, cc)?;
    let k256 = bin(fb_, BinOp::ShiftLeftLogical, k, k8)?;
    let one = u1(fb_, 1)?;
    let two_k2 = bin(fb_, BinOp::ShiftLeftLogical, k2, one)?;
    let (three, k5) = (u1(fb_, 3)?, u1(fb_, 5)?);
    let three_c = bin(fb_, BinOp::IMul, c, three)?;
    let five_k2 = bin(fb_, BinOp::IMul, k2, k5)?;
    let two = u1(fb_, 2)?;
    let four_k2 = bin(fb_, BinOp::ShiftLeftLogical, k2, two)?;
    // |a0| = C + 256k - 2k², a1 = 2^17 - 5k² + 3C, a2 = 4k² + 256k - 3C, |a3| = k² - C.
    let m0 = bin(fb_, BinOp::IAdd, c, k256)?;
    let m0 = bin(fb_, BinOp::ISub, m0, two_k2)?;
    let k17 = u1(fb_, 1 << 17)?;
    let a1 = bin(fb_, BinOp::ISub, k17, five_k2)?;
    let a1 = bin(fb_, BinOp::IAdd, a1, three_c)?;
    let a2 = bin(fb_, BinOp::IAdd, four_k2, k256)?;
    let a2 = bin(fb_, BinOp::ISub, a2, three_c)?;
    let m3 = bin(fb_, BinOp::ISub, k2, c)?;
    let halve = |fb: &mut FunctionBuilder<'_>, a: Value| -> BResult<Value> {
        let a = bin(fb, BinOp::IAdd, a, one)?;
        bin(fb, BinOp::ShiftRightLogical, a, one)
    };
    let w0 = halve(fb_, m0)?;
    let w0 = fb_.unary(UnOp::SNegate, w0)?;
    let w1 = halve(fb_, a1)?;
    let w2 = halve(fb_, a2)?;
    let w3 = halve(fb_, m3)?;
    let w3 = fb_.unary(UnOp::SNegate, w3)?;
    let w = fb_.composite_construct(t.vu, &[w0, w1, w2, w3])?;
    finish(fb, w)
}

// ---- The exact sums ----

/// The grid of an `EXACT` format: `D` (`16 · 255`, `16 · 127`, `65535`,
/// or `32767`; with `SRGB` `4095` for the color lanes and `16 · 255` for
/// alpha; with `EXTENDED` `16 · 510` for the color lanes, `8192` an `_sRGB` twin's, and `8 · 510`
/// for alpha) as `u32` lanes, the bounds of a tap's code (`±D`, `±2D` for `EXTENDED`) as `f32`
/// lanes, the bounds of a sum's step (`[0, D]`, `[-D, D]` for a signed format,
/// `[-2D, 2D]` for `EXTENDED`) as `i32` bits, and what the result's quotient reads: the grid it
/// divides by (`D`, or `2D` for `EXTENDED`, whose steps reach past `D`), the factor that restores
/// it (1 or 2, exact), and the lanes whose `half` result rounds ties away from zero (an
/// `EXTENDED` `_sRGB` twin's color lanes, a YCbCr field's). A tap's code is recovered on `scale`
/// (`f32` lanes) and
/// multiplied by `per` (`u32` lanes): `D` and 1 (no `per`), but on a packed normalized format's
/// lanes (its `TexelFormat::PACKED` field with `PackedTexels::filter_grids`, where the routine
/// reads such a class: [`Packed`]), which take the field's grid `D` each, their channel's code
/// scale `S` (also the code's bound) and `D / S`. Such a routine's grid also states whether the
/// field names its grids and the field's truncation `s` per lane
/// (`PackedTexels::filter_truncation`), which [`Truncation::of`] takes there. A field whose steps
/// reach `±2D` (`PackedTexels::filter_extended`) bounds its codes by `±2S` and divides by `2D`.
struct Grid {
    d: Value,
    scale: Value,
    per: Option<Value>,
    cap: Value,
    neg_cap: Value,
    lo: Value,
    hi: Value,
    divisor: Value,
    factor: Value,
    away: Value,
    packed: Option<(Value, Value)>,
}

/// Whether a routine's treatment may name a packed normalized format's grids: a `TexelFormat`
/// class's routine (the linear filter's), not a `BicubicTexels` one's nor the cube corner's,
/// whose `CubeCorner` grids a packed class's corner takes.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Packed {
    Field,
    No,
}

fn grid(fb: &mut FunctionBuilder<'_>, t: Tys, flags: Value, packed: Packed) -> BResult<Grid> {
    let wide16 = flag(fb, flags, BicubicTexels::WIDE16)?;
    let snorm = flag(fb, flags, BicubicTexels::SNORM)?;
    let (d16s, d16u) = (u1(fb, 32767)?, u1(fb, 65535)?);
    let (d8s, d8u) = (u1(fb, 16 * 127)?, u1(fb, 16 * 255)?);
    let d16 = fb.select(snorm, d16s, d16u)?;
    let d8 = fb.select(snorm, d8s, d8u)?;
    let d = fb.select(wide16, d16, d8)?;
    let d = broadcast(fb, t.vu, d)?;
    let srgb = flag(fb, flags, BicubicTexels::SRGB)?;
    let (color, alpha) = (u1(fb, 4095)?, u1(fb, 16 * 255)?);
    let srgb_d = fb.composite_construct(t.vu, &[color, color, color, alpha])?;
    let d = fb.select(srgb, srgb_d, d)?;
    let extended = flag(fb, flags, BicubicTexels::EXTENDED)?;
    let (linear, steps, alpha) = (u1(fb, 16 * 510)?, u1(fb, 8192)?, u1(fb, 8 * 510)?);
    let color = fb.select(srgb, steps, linear)?;
    let extended_d = fb.composite_construct(t.vu, &[color, color, color, alpha])?;
    let d = fb.select(extended, extended_d, d)?;
    let (one, two) = (u1(fb, 1)?, u1(fb, 2)?);
    let factor_u = fb.select(extended, two, one)?;
    let factor_u = broadcast(fb, t.vu, factor_u)?;
    let divisor = bin(fb, BinOp::IMul, d, factor_u)?;
    let factor = fb.convert(ConvOp::ConvertUToF, factor_u, t.vf)?;
    let lo = |fb: &mut FunctionBuilder<'_>, divisor: Value| -> BResult<Value> {
        let neg = fb.unary(UnOp::SNegate, divisor)?;
        let z = u4(fb, 0)?;
        let signed = bin(fb, BinOp::LogicalOr, snorm, extended)?;
        fb.select(signed, neg, z)
    };
    let srgb_steps = bin(fb, BinOp::LogicalAnd, extended, srgb)?;
    let (yes, no) = (splat(fb, bform(1), 1)?, splat(fb, bform(1), 0)?);
    let lanes = crate::lower::arith::ops::ty(fb, bform(LANES))?;
    let color_lanes = fb.composite_construct(lanes, &[yes, yes, yes, no])?;
    let none = splat(fb, bform(LANES), 0)?;
    let away = fb.select(srgb_steps, color_lanes, none)?;
    let Some(p) = packed_grids(fb, t, flags, packed)? else {
        let cap = fb.convert(ConvOp::ConvertUToF, divisor, t.vf)?;
        return Ok(Grid {
            d,
            scale: fb.convert(ConvOp::ConvertUToF, d, t.vf)?,
            per: None,
            cap,
            neg_cap: fb.unary(UnOp::FNegate, cap)?,
            lo: lo(fb, divisor)?,
            hi: divisor,
            divisor,
            factor,
            away,
            packed: None,
        });
    };
    let packed = p.named;
    // A packed field's lanes: its grids, each tap's code recovered on its channel's scale
    // (clamped there, within `±2S` where the field's steps reach `±2D`) and moved onto the grid;
    // such a field's step divided by `2D` and doubled, and a YCbCr field's `half` rounded with
    // ties away from zero.
    let scale = fb.select(packed, p.scales, d)?;
    let scale = fb.convert(ConvOp::ConvertUToF, scale, t.vf)?;
    let wide = bin(fb, BinOp::LogicalAnd, packed, p.wide)?;
    let span = fb.select(wide, two, one)?;
    let span = broadcast(fb, t.vu, span)?;
    let field_cap = bin(fb, BinOp::IMul, p.scales, span)?;
    let cap = fb.select(packed, field_cap, divisor)?;
    let cap = fb.convert(ConvOp::ConvertUToF, cap, t.vf)?;
    let neg_cap = fb.unary(UnOp::FNegate, cap)?;
    let ones = u4(fb, 1)?;
    let per = Some(fb.select(packed, p.per, ones)?);
    let d = fb.select(packed, p.d, d)?;
    let field_divisor = bin(fb, BinOp::IMul, p.d, span)?;
    let divisor = fb.select(packed, field_divisor, divisor)?;
    let span = fb.convert(ConvOp::ConvertUToF, span, t.vf)?;
    let factor = fb.select(packed, span, factor)?;
    let rounds_away = bin(fb, BinOp::LogicalAnd, packed, p.away)?;
    let away = fb.select(rounds_away, color_lanes, away)?;
    Ok(Grid {
        d,
        scale,
        per,
        cap,
        neg_cap,
        lo: lo(fb, divisor)?,
        hi: divisor,
        divisor,
        factor,
        away,
        packed: Some((packed, p.truncation)),
    })
}

/// The code of normalized texel `x` on grid `g`, as `i32` bits: `RoundEven(x · D)` (module docs),
/// or on a packed normalized format's lane `RoundEven(x · S) · (D / S)`, its code on its channel's
/// scale `S` moved onto the grid (the device's value may miss `c / S` by more than half a grid
/// step, never by half a code). The product is clamped into the code bounds first (`[-D, D]`,
/// `[-2D, 2D]` for an extended-range lane, `[-S, S]` on a packed lane), which changes no texel's of
/// the format, so that a texel of a format this routine does not serve (its result is selected
/// away) still converts and shifts within range.
fn code(fb: &mut FunctionBuilder<'_>, g: &Grid, x: Value) -> BResult<Value> {
    let y = fb.binary_no_contraction(BinOp::FMul, x, g.scale)?;
    let y = fb.glsl(GlslOp::NClamp, &[y, g.neg_cap, g.cap])?;
    let y = fb.glsl(GlslOp::RoundEven, &[y])?;
    let it = crate::lower::arith::ops::ty(fb, uform(LANES))?;
    let c = fb.convert(ConvOp::ConvertFToS, y, it)?;
    match g.per {
        Some(per) => bin(fb, BinOp::IMul, c, per),
        None => Ok(c),
    }
}

/// The value `n / D` of grid step `n` (`i32` bits, within the grid's step bounds): `+0` for
/// zero, the magnitude divided exactly, rounded to `f32`, or to `half` when `half`, and the sign
/// set. An extended-range lane divides by `2D` and doubles, exactly; its `_sRGB` color lanes'
/// `half` is the exact `f32` rounded with ties away from zero.
fn ratio(fb: &mut FunctionBuilder<'_>, g: &Grid, n: Value, half: Value) -> BResult<Value> {
    let rf = texel_round::grid_value(fb, uform(LANES), n, g.divisor, |fb, m, d| {
        texel_round::ratio_to_f32(fb, uform(LANES), m, d)
    })?;
    let rh = texel_round::grid_value(fb, uform(LANES), n, g.divisor, |fb, m, d| {
        texel_round::ratio_to_half(fb, uform(LANES), m, d)
    })?;
    let rf = exact_op(fb, BinOp::FMul, rf, g.factor)?;
    let rh = exact_op(fb, BinOp::FMul, rh, g.factor)?;
    let ra = texel_round::f32_to_half(fb, fform(LANES), rf, HalfRounding::NearestAway)?;
    let rh = fb.select(g.away, ra, rh)?;
    fb.select(half, rh, rf)
}

/// The signed 64-bit product of `i32` lanes `a` and `b`, as a pair.
fn smul(fb: &mut FunctionBuilder<'_>, a: Value, b: Value) -> BResult<Pair> {
    let p = bin(fb, BinOp::SMulExtended, a, b)?;
    Ok(Pair {
        lo: fb.composite_extract(p, &[0])?,
        hi: fb.composite_extract(p, &[1])?,
    })
}

/// The sum of pairs `terms` modulo `2^64` (zero for none).
fn total(fb: &mut FunctionBuilder<'_>, terms: &[Pair]) -> BResult<Pair> {
    let Some((first, rest)) = terms.split_first() else {
        return Ok(Pair {
            lo: u4(fb, 0)?,
            hi: u4(fb, 0)?,
        });
    };
    let mut acc = *first;
    for p in rest {
        acc = wide::add(fb, acc, *p)?;
    }
    Ok(acc)
}

/// `fn(16 × vec4, wx, wy, flags, half) -> vec4` (module docs, [`Plane::exact`]).
///
/// A row's sum `Σ cᵢ · wxᵢ` stays below `2^33` in magnitude (codes below `2^16`, the weights'
/// magnitudes summing below `1.2 · 2^16`), and the footprint's below `2^49`: both fit the
/// two-word pairs, whose arithmetic is exact modulo `2^64`. The step is `⌊(S + 2^31) / 2^32⌋`,
/// the high word of `S + 2^31`.
fn define_exact_plane(mb: &mut ModuleBuilder) -> BResult<FuncId> {
    let t = types(mb)?;
    let mut fb = mb.routine(t.vf)?;
    let mut taps = Vec::with_capacity(16);
    for _ in 0..16 {
        taps.push(fb.param(t.vf)?);
    }
    let wx = fb.param(t.vu)?;
    let wy = fb.param(t.vu)?;
    let flags = fb.param(t.u)?;
    let half = fb.param(t.b)?;
    let fb_ = &mut fb;
    let g = grid(fb_, t, flags, Packed::No)?;
    let mut codes = Vec::with_capacity(taps.len());
    for x in &taps {
        codes.push(code(fb_, &g, *x)?);
    }
    let mut rows = Vec::with_capacity(4);
    for (j, row) in (0u32..).zip(codes.chunks(4)) {
        let mut products = Vec::with_capacity(4);
        for (i, c) in (0u32..).zip(row) {
            let w = fb_.composite_extract(wx, &[i])?;
            let w = broadcast(fb_, t.vu, w)?;
            products.push(smul(fb_, *c, w)?);
        }
        let sum = total(fb_, &products)?;
        let w = fb_.composite_extract(wy, &[j])?;
        let w = broadcast(fb_, t.vu, w)?;
        let k31 = u4(fb_, 31)?;
        let hi = bin(fb_, BinOp::ShiftRightArithmetic, w, k31)?;
        rows.push(wide::mul(fb_, sum, Pair { lo: w, hi })?);
    }
    let total = total(fb_, &rows)?;
    let bias = Pair {
        lo: u4(fb_, 0x8000_0000)?,
        hi: u4(fb_, 0)?,
    };
    let n = wide::add(fb_, total, bias)?.hi;
    let n = fb_.glsl(GlslOp::SClamp, &[n, g.lo, g.hi])?;
    let r = ratio(fb_, &g, n, half)?;
    finish(fb, r)
}

/// `fn(t0, t1, k, flags, half) -> vec4` (module docs, [`Line::exact`], and [`Span::exact`] where
/// `packed` names a packed normalized format's grids): the step
/// `⌊(c₀ · (256 - k) + c₁ · k + 128) / 256⌋`, below `2^25` in magnitude before the shift.
fn define_exact_line(mb: &mut ModuleBuilder, packed: Packed, calls: Calls) -> BResult<FuncId> {
    let t = types(mb)?;
    let mut fb = start(calls, mb, t.vf)?;
    let t0 = fb.param(t.vf)?;
    let t1 = fb.param(t.vf)?;
    let k = fb.param(t.u)?;
    let flags = fb.param(t.u)?;
    let half = fb.param(t.b)?;
    let fb_ = &mut fb;
    let g = grid(fb_, t, flags, packed)?;
    let c0 = code(fb_, &g, t0)?;
    let c1 = code(fb_, &g, t1)?;
    let kv = broadcast(fb_, t.vu, k)?;
    let k256 = u4(fb_, 256)?;
    let w0 = bin(fb_, BinOp::ISub, k256, kv)?;
    let a = bin(fb_, BinOp::IMul, c0, w0)?;
    let b = bin(fb_, BinOp::IMul, c1, kv)?;
    let s = bin(fb_, BinOp::IAdd, a, b)?;
    let k128 = u4(fb_, 128)?;
    let s = bin(fb_, BinOp::IAdd, s, k128)?;
    let k8 = u4(fb_, 8)?;
    let n = bin(fb_, BinOp::ShiftRightArithmetic, s, k8)?;
    // A no-op for codes of one format, which average within its range; it keeps `ratio`'s
    // operands in range whatever the texels.
    let n = fb_.glsl(GlslOp::SClamp, &[n, g.lo, g.hi])?;
    let r = ratio(fb_, &g, n, half)?;
    finish(fb, r)
}

/// The parameters of a [`Quad`] routine, in order.
struct QuadParams {
    t00: Value,
    t01: Value,
    t10: Value,
    t11: Value,
    kx: Value,
    ky: Value,
    flags: Value,
    half: Value,
}

fn quad_params(fb: &mut FunctionBuilder<'_>, t: Tys) -> BResult<QuadParams> {
    Ok(QuadParams {
        t00: fb.param(t.vf)?,
        t01: fb.param(t.vf)?,
        t10: fb.param(t.vf)?,
        t11: fb.param(t.vf)?,
        kx: fb.param(t.u)?,
        ky: fb.param(t.u)?,
        flags: fb.param(t.u)?,
        half: fb.param(t.b)?,
    })
}

/// `fn(t00, t01, t10, t11, kx, ky, flags, half) -> vec4` (module docs, [`Quad::exact`]): Metal's
/// sum over the four taps under the weights `w = wx · wy` in units of `2^-16`, the axes' weights
/// `256 - k` and `k` (a merged axis's fraction arrives as 0), each term truncated: `T = Σ ⌊c · w /
/// 2^s⌋` for the codes `c` on the format's grid, then the step `⌊(T + 2^(15 - s)) / 2^(16 -
/// s)⌋`, clamped and divided by the grid exactly. `s` is 8 for a 16-bit format's codes and 4 for
/// every 12-bit grid: an sRGB color lane's step is truncated, and an 8-bit lane's `16 ·
/// code` term is a multiple of 16, so its sum is the exact one, rounded half up as [`Plane::exact`]
/// rounds it under the weights `[0, (256 - k) · 256, k · 256, 0]`. An extended-range format's `s`
/// is 7 on a color lane and 4 on alpha (every lane of random 2D footprints of the
/// four formats). An unsigned format's product `c · w` is below `2^32` (codes below `2^16`, weights
/// at most `2^16`), a `u32` word shifted logically; a signed format's codes (at most `2^15 - 1` in
/// magnitude) keep it below `2^31`, an `i32` shifted arithmetically (toward negative infinity), as
/// an extended-range format's do (below `2^14`). The terms' sum is below `2^24` in magnitude (the
/// weights sum to `2^16`, and the codes are below `2^12` where `s` is 4, below `2^14` where it is
/// 7, and below `2^16` where it is 8), and so is the biased one.
fn define_exact_quad(mb: &mut ModuleBuilder) -> BResult<FuncId> {
    let t = types(mb)?;
    let mut fb = mb.texel_routine(t.vf)?;
    let q = quad_params(&mut fb, t)?;
    let fb_ = &mut fb;
    let g = grid(fb_, t, q.flags, Packed::Field)?;
    let n = quad_sum(fb_, t, &g, &q, |fb, x| code(fb, &g, x))?;
    let r = ratio(fb_, &g, n, q.half)?;
    finish(fb, r)
}

/// `fn(c00, c01, c10, c11: uvec4, kx, ky, flags, half) -> vec4` ([`Routine::QuadSteps`]): the
/// exact quad's sum ([`define_exact_quad`]) of an extended-range sample's four taps given as
/// their grid steps (`i32` bits, `arith::extended_range::steps`, a border tap's its value times
/// the lane's grid), the integers [`code`] recovers from their decoded values: each tap's step
/// enters the sum as the recovered code did, so the result is the same, without a decode at the
/// tap or a recovery here.
pub fn define_quad_steps(mb: &mut ModuleBuilder) -> BResult<FuncId> {
    let t = types(mb)?;
    let mut fb = mb.texel_routine(t.vf)?;
    let q = QuadParams {
        t00: fb.param(t.vu)?,
        t01: fb.param(t.vu)?,
        t10: fb.param(t.vu)?,
        t11: fb.param(t.vu)?,
        kx: fb.param(t.u)?,
        ky: fb.param(t.u)?,
        flags: fb.param(t.u)?,
        half: fb.param(t.b)?,
    };
    let fb_ = &mut fb;
    let g = grid(fb_, t, q.flags, Packed::Field)?;
    let n = quad_sum(fb_, t, &g, &q, |_, c| Ok(c))?;
    let r = extended_ratio(fb_, t, &g, q.flags, n, q.half)?;
    finish(fb, r)
}

/// [`ratio`] of an extended-range format's step `n` (`BicubicTexels::EXTENDED`: [`grid`]'s
/// divisor `2D` and factor 2, so the value is `|n| / D` correctly rounded, `n`'s sign set), its
/// `f32` result without [`ratio`]'s general quotient: on a lane of grid `D = 255 · 2^k` (a color
/// lane's `16 · 510`, `k = 5`; alpha's `8 · 510`, `k = 4`) `RN(|n| / 255)`
/// ([`texel_round::ratio_by_255_to_f32`]) times `2^-k`, and on an `_sRGB` twin's color lane (`D =
/// 8192`) `|n| · 2^-13`, each product exact (`|n| <= 2D`, every quotient normal or zero). A `half`
/// result is that `f32` rounded to `half`, which no quotient here rounds twice (below).
fn extended_ratio(
    fb: &mut FunctionBuilder<'_>,
    t: Tys,
    g: &Grid,
    flags: Value,
    n: Value,
    half: Value,
) -> BResult<Value> {
    let m = fb.glsl(GlslOp::SAbs, &[n])?;
    let q = texel_round::ratio_by_255_to_f32(fb, uform(LANES), m)?;
    let (color, alpha) = (f1(fb, 1.0 / 32.0)?, f1(fb, 1.0 / 16.0)?);
    let scale = fb.composite_construct(t.vf, &[color, color, color, alpha])?;
    let by255 = exact_op(fb, BinOp::FMul, q, scale)?;
    let x = fb.convert(ConvOp::ConvertUToF, m, t.vf)?;
    let step = f4(fb, 1.0 / 8192.0)?;
    let steps = exact_op(fb, BinOp::FMul, x, step)?;
    let srgb = flag(fb, flags, BicubicTexels::SRGB)?;
    let no = splat(fb, bform(1), 0)?;
    let lanes = crate::lower::arith::ops::ty(fb, bform(LANES))?;
    let stepped = fb.composite_construct(lanes, &[srgb, srgb, srgb, no])?;
    let v = fb.select(stepped, steps, by255)?;
    let bits = fb.convert(ConvOp::Bitcast, v, t.vu)?;
    // A `half` result: no `f32` quotient on these grids lies on a `half` tie, so rounding it
    // rounds the quotient once (`the_extended_grids_half_rounds_once_through_f32`), to nearest
    // even, an `_sRGB` twin's color lane with ties away from zero; every one is a normal `half`
    // or zero (at least `1 / 8160`, at most 2).
    let rounded = texel_round::normal_half_bits(fb, uform(LANES), bits, g.away)?;
    let bits = fb.select(half, rounded, bits)?;
    let z = u4(fb, 0)?;
    let neg = bin(fb, BinOp::SLessThan, n, z)?;
    let sign = u4(fb, 0x8000_0000)?;
    let sign = fb.select(neg, sign, z)?;
    let bits = bin(fb, BinOp::BitwiseOr, bits, sign)?;
    fb.convert(ConvOp::Bitcast, bits, t.vf)
}

/// The exact quad's step ([`define_exact_quad`]) over grid `g` of `q`'s taps (`(i, j)`, `(i + 1,
/// j)`, `(i, j + 1)`, `(i + 1, j + 1)`), each tap's code `code_of` it (taken where its term is),
/// under `q`'s fractions and flags: the terms' step rounded half up and clamped into the grid's
/// step bounds, which the caller divides.
fn quad_sum(
    fb_: &mut FunctionBuilder<'_>,
    t: Tys,
    g: &Grid,
    q: &QuadParams,
    mut code_of: impl FnMut(&mut FunctionBuilder<'_>, Value) -> BResult<Value>,
) -> BResult<Value> {
    let (kx, ky, flags) = (q.kx, q.ky, q.flags);
    let k256 = u1(fb_, 256)?;
    let wx = [bin(fb_, BinOp::ISub, k256, kx)?, kx];
    let wy = [bin(fb_, BinOp::ISub, k256, ky)?, ky];
    let tr = Truncation::of(fb_, t, flags, g)?;
    let rows = [[q.t00, q.t01], [q.t10, q.t11]];
    let mut total = u4(fb_, 0)?;
    for (row, b) in rows.into_iter().zip(wy) {
        for (x, a) in row.into_iter().zip(wx) {
            let c = code_of(fb_, x)?;
            let w = bin(fb_, BinOp::IMul, a, b)?;
            let w = broadcast(fb_, t.vu, w)?;
            let term = tr.term(fb_, c, w)?;
            total = bin(fb_, BinOp::IAdd, total, term)?;
        }
    }
    let n = tr.step(fb_, total)?;
    fb_.glsl(GlslOp::SClamp, &[n, g.lo, g.hi])
}

/// An upper estimate of [`define_levels`]'s words: eight codes' recovery, the weights and terms,
/// the step, the clamp, and the ratio's two roundings.
const LEVELS_WORDS: u32 = 4096;

/// `fn(a00, a01, a10, a11, b00, b01, b10, b11: vec4, kxa, kya, kxb, kyb, kz: u32, flags: u32,
/// half: bool) -> vec4`: an extended-range sample's linear mip filter (`super::super::extended`),
/// Metal's trilinear sum over its two levels' footprints, the level as a third axis. `a..` are
/// level `L`'s four taps (`(i, j)`, `(i + 1, j)`, `(i, j + 1)`, `(i + 1, j + 1)`, each decoded,
/// a `float`, which [`code`] recovers exactly) and `kxa`, `kya` its fractions in 256ths (a merged
/// axis's 0), `b..`, `kxb`, `kyb` level `L + 1`'s, and `kz` the second level's weight in 256ths
/// (four times its sixty-fourths). A nearest filter's level is its texel at every tap, at
/// fractions 0. Each tap weighs the product `P = wx · wy · wz` of its axes' weights `256 - k` and
/// `k`, rounded to `P / 256` to nearest, a tie up on the `t` axis's first tap and down on its
/// second, as a 3D footprint's ([`block_weights`]); its term is truncated as a 2D sum's
/// ([`Truncation`]: onto `2^-7` of a color lane's steps, `2^-4` of alpha's), the terms' step
/// rounded half up, clamped, and divided by the grid exactly ([`ratio`]: a `half` result rounded
/// to nearest even, an `_sRGB` color lane's the exact `f32` rounded with ties away from zero).
/// Every lane of 8,192 samples per form (`BGR10_XR`, `BGRA10_XR` and their `_sRGB` twins),
/// filter, result type, 2D and 2D array textures of five levels, at levels from -0.6 to 5.5,
/// read so on an Apple M3 Max under macOS 27.0.1; at the weight 0 the sum is level `L`'s 2D one.
pub fn define_levels(mb: &mut ModuleBuilder) -> BResult<FuncId> {
    let t = types(mb)?;
    let mut fb = mb.texel_routine(t.vf)?;
    let mut quads = Vec::with_capacity(2);
    for _ in 0..2 {
        let quad: [Value; 4] = [
            fb.param(t.vf)?,
            fb.param(t.vf)?,
            fb.param(t.vf)?,
            fb.param(t.vf)?,
        ];
        quads.push(quad);
    }
    let mut ks = Vec::with_capacity(2);
    for _ in 0..2 {
        ks.push((fb.param(t.u)?, fb.param(t.u)?));
    }
    let kz = fb.param(t.u)?;
    let flags = fb.param(t.u)?;
    let half = fb.param(t.b)?;
    let fb_ = &mut fb;
    let g = grid(fb_, t, flags, Packed::No)?;
    let tr = Truncation::of(fb_, t, flags, &g)?;
    let k256 = u1(fb_, 256)?;
    let pair = |fb: &mut FunctionBuilder<'_>, k: Value| -> BResult<[Value; 2]> {
        Ok([bin(fb, BinOp::ISub, k256, k)?, k])
    };
    let wz = pair(fb_, kz)?;
    let (k8, k255, k128) = (u1(fb_, 8)?, u1(fb_, 255)?, u1(fb_, 128)?);
    let (z, one) = (u1(fb_, 0)?, u1(fb_, 1)?);
    let mut total = u4(fb_, 0)?;
    for ((c, [t00, t01, t10, t11]), (kx, ky)) in wz.into_iter().zip(quads).zip(ks) {
        let (wx, wy) = (pair(fb_, kx)?, pair(fb_, ky)?);
        for (row, (b, taps)) in wy.into_iter().zip([[t00, t01], [t10, t11]]).enumerate() {
            for (a, tap) in wx.into_iter().zip(taps) {
                let p = bin(fb_, BinOp::IMul, a, b)?;
                let p = bin(fb_, BinOp::IMul, p, c)?;
                let q = bin(fb_, BinOp::ShiftRightLogical, p, k8)?;
                let r = bin(fb_, BinOp::BitwiseAnd, p, k255)?;
                let above = bin(fb_, BinOp::UGreaterThan, r, k128)?;
                let up = match row {
                    0 => {
                        let tie = bin(fb_, BinOp::IEqual, r, k128)?;
                        bin(fb_, BinOp::LogicalOr, above, tie)?
                    }
                    _ => above,
                };
                let inc = fb_.select(up, one, z)?;
                let w = bin(fb_, BinOp::IAdd, q, inc)?;
                let w = broadcast(fb_, t.vu, w)?;
                let code = code(fb_, &g, tap)?;
                let term = tr.term(fb_, code, w)?;
                total = bin(fb_, BinOp::IAdd, total, term)?;
            }
        }
    }
    let n = tr.step(fb_, total)?;
    let n = fb_.glsl(GlslOp::SClamp, &[n, g.lo, g.hi])?;
    let r = ratio(fb_, &g, n, half)?;
    finish(fb, r)
}

/// How a 2D or 3D exact sum truncates its terms: `s` per lane (8 for a 16-bit format's codes, 4 for
/// every 12-bit grid, for an extended-range format 7 on a color lane and 4 on alpha, and on a
/// packed field's grids the field's own, `PackedTexels::filter_truncation`: 7 on YCbCr's grid
/// `2048`), the bias `2^(15 - s)` and shift `16 - s` of the step, and whether the codes are signed
/// (a signed normalized or an extended-range format's), whose terms and sum shift arithmetically
/// (toward negative infinity).
struct Truncation {
    s: Value,
    bias: Value,
    shift: Value,
    signed: Value,
}

impl Truncation {
    fn of(fb: &mut FunctionBuilder<'_>, t: Tys, flags: Value, g: &Grid) -> BResult<Truncation> {
        let wide16 = flag(fb, flags, BicubicTexels::WIDE16)?;
        let snorm = flag(fb, flags, BicubicTexels::SNORM)?;
        let extended = flag(fb, flags, BicubicTexels::EXTENDED)?;
        let (a, b) = (u1(fb, 8)?, u1(fb, 4)?);
        let plain = fb.select(wide16, a, b)?;
        let plain = broadcast(fb, t.vu, plain)?;
        let seven = u1(fb, 7)?;
        let lanes = fb.composite_construct(t.vu, &[seven, seven, seven, b])?;
        let s = fb.select(extended, lanes, plain)?;
        let s = match g.packed {
            Some((named, own)) => fb.select(named, own, s)?,
            None => s,
        };
        let (k15, k16, one) = (u4(fb, 15)?, u4(fb, 16)?, u4(fb, 1)?);
        let up = bin(fb, BinOp::ISub, k15, s)?;
        let bias = bin(fb, BinOp::ShiftLeftLogical, one, up)?;
        let shift = bin(fb, BinOp::ISub, k16, s)?;
        let signed = bin(fb, BinOp::LogicalOr, snorm, extended)?;
        Ok(Truncation {
            s,
            bias,
            shift,
            signed,
        })
    }

    /// The term `⌊c · w / 2^s⌋` of code `c` under weight `w` (`uvec4` lanes).
    fn term(&self, fb: &mut FunctionBuilder<'_>, c: Value, w: Value) -> BResult<Value> {
        let prod = bin(fb, BinOp::IMul, c, w)?;
        let unsigned = bin(fb, BinOp::ShiftRightLogical, prod, self.s)?;
        let signed = bin(fb, BinOp::ShiftRightArithmetic, prod, self.s)?;
        fb.select(self.signed, signed, unsigned)
    }

    /// The grid step `⌊(T + 2^(15 - s)) / 2^(16 - s)⌋` of the terms' sum `T`.
    fn step(&self, fb: &mut FunctionBuilder<'_>, total: Value) -> BResult<Value> {
        let total = bin(fb, BinOp::IAdd, total, self.bias)?;
        let unsigned = bin(fb, BinOp::ShiftRightLogical, total, self.shift)?;
        let signed = bin(fb, BinOp::ShiftRightArithmetic, total, self.shift)?;
        fb.select(self.signed, signed, unsigned)
    }
}

/// `fn(x, y, z, flags) -> vec4` ([`Cube::corner`]): the tap a linear cube sample reads past both
/// edges of its face, from `x`, `y`, and `z`, the texels at the cube's corner of the faces of the
/// `x`, `y`, and `z` axes. An `EXACT` format's tap is a texel Metal makes up: on each lane `⌊(6 X +
/// 5 Y + 5 Z + 8) / 16⌋` of the three texels' codes (an sRGB color lane's table steps, a signed
/// format's least code read as the next one, as [`code`] reads it), handed on as its value on the
/// format's grid, which the sums recover exactly. In grid units an 8-bit lane's code is `16 ·
/// code`, so its sum is shifted by 4 more and back. A format of a `CubeCorner` with grids (its
/// `TexelFormat::CORNER` field) makes up the same texel of its codes on each lane's grid. Any other
/// format's is `x`.
fn define_corner(mb: &mut ModuleBuilder) -> BResult<FuncId> {
    let t = types(mb)?;
    let mut fb = mb.routine(t.vf)?;
    let x = fb.param(t.vf)?;
    let y = fb.param(t.vf)?;
    let z = fb.param(t.vf)?;
    let flags = fb.param(t.u)?;
    let fb_ = &mut fb;
    let g = grid(fb_, t, flags, Packed::No)?;
    let wide16 = flag(fb_, flags, BicubicTexels::WIDE16)?;
    let srgb = flag(fb_, flags, BicubicTexels::SRGB)?;
    // A cube corner outside the families: its codes on each lane's grid `D` of the field's
    // `CubeCorner` (`RoundEven(v · D)`, clamped into `[-D, D]`), whose made-up code is the texel.
    let (coded, corner_d) = corner_grids(fb_, &t, flags)?;
    let d = fb_.select(coded, corner_d, g.d)?;
    let df = fb_.convert(ConvOp::ConvertUToF, d, t.vf)?;
    let neg_df = fb_.unary(UnOp::FNegate, df)?;
    // Per lane, the constant of a 16-bit lane or an sRGB color lane (`fine`), of an 8-bit lane
    // (`coarse`), or of a cube corner's lane (`corner`).
    let lanes = |fb: &mut FunctionBuilder<'_>, fine: u32, coarse: u32, corner: u32| {
        let (f, c) = (u1(fb, fine)?, u1(fb, coarse)?);
        let plain = fb.select(wide16, f, c)?;
        let plain = broadcast(fb, t.vu, plain)?;
        let steps = fb.composite_construct(t.vu, &[f, f, f, c])?;
        let family = fb.select(srgb, steps, plain)?;
        let corner = u4(fb, corner)?;
        fb.select(coded, corner, family)
    };
    let mut sum = lanes(fb_, 1 << 3, 1 << 7, 1 << 3)?;
    for (v, w) in [(x, 6), (y, 5), (z, 5)] {
        let c = texel_round::grid_step(fb_, fform(LANES), v, df, neg_df)?;
        let w = u4(fb_, w)?;
        let term = bin(fb_, BinOp::IMul, c, w)?;
        sum = bin(fb_, BinOp::IAdd, sum, term)?;
    }
    let shift = lanes(fb_, 4, 8, 4)?;
    let back = lanes(fb_, 0, 4, 0)?;
    let n = bin(fb_, BinOp::ShiftRightArithmetic, sum, shift)?;
    let n = bin(fb_, BinOp::ShiftLeftLogical, n, back)?;
    let made = texel_round::grid_value(fb_, uform(LANES), n, d, |fb, m, d| {
        texel_round::ratio_to_f32(fb, uform(LANES), m, d)
    })?;
    let exact = flag(fb_, flags, BicubicTexels::EXACT)?;
    let made_up = bin(fb_, BinOp::LogicalOr, exact, coded)?;
    let r = fb_.select(made_up, made, x)?;
    finish(fb, r)
}

/// Whether treatment `flags` states a `CubeCorner` with grids in its `TexelFormat::CORNER` field
/// (every corner after `CubeCorner::XFace`), and that corner's grid `D` per lane (`uvec4`; each
/// lane 1 for any other field).
fn corner_grids(fb: &mut FunctionBuilder<'_>, t: &Tys, flags: Value) -> BResult<(Value, Value)> {
    let shift = u1(fb, TexelFormat::CORNER)?;
    let field = bin(fb, BinOp::ShiftRightLogical, flags, shift)?;
    let four_bits = u1(fb, 0xF)?;
    let field = bin(fb, BinOp::BitwiseAnd, field, four_bits)?;
    let x_face = u1(fb, CubeCorner::XFace as u32)?;
    let coded = bin(fb, BinOp::UGreaterThan, field, x_face)?;
    let mut d = u4(fb, 1)?;
    for corner in CubeCorner::ALL {
        let Some(grids) = corner.grids() else {
            continue;
        };
        let k = u1(fb, corner as u32)?;
        let is = bin(fb, BinOp::IEqual, field, k)?;
        let mut lanes = Vec::with_capacity(4);
        for g in grids {
            lanes.push(u1(fb, g)?);
        }
        let g = fb.composite_construct(t.vu, &lanes)?;
        d = fb.select(is, g, d)?;
    }
    Ok((coded, d))
}

/// A packed field's grids ([`packed_grids`]), each `uvec4` lanes.
struct PackedGrids {
    /// Whether the treatment names the field's grids.
    named: Value,
    /// The grids `D`.
    d: Value,
    /// The channels' code scales `S`.
    scales: Value,
    /// `D / S`.
    per: Value,
    /// The 2D and 3D sums' truncation `s` (`PackedTexels::filter_truncation`).
    truncation: Value,
    /// Whether the field's codes and steps reach `±2D` (`PackedTexels::filter_extended`), a
    /// `bool`.
    wide: Value,
    /// Whether a `half` result rounds the step's value with ties away from zero
    /// (`PackedTexels::ycbcr`), a `bool`.
    away: Value,
}

/// Whether treatment `flags` names a packed field's grids (its `TexelFormat::PACKED` field with
/// `PackedTexels::filter_grids`, under `BicubicTexels::EXACT`: `TexelFormat::packed_exact`), and
/// per lane that field's grids (each lane 1 for any other field); `None` for a routine whose
/// treatment names none.
fn packed_grids(
    fb: &mut FunctionBuilder<'_>,
    t: Tys,
    flags: Value,
    packed: Packed,
) -> BResult<Option<PackedGrids>> {
    if packed == Packed::No {
        return Ok(None);
    }
    let shift = u1(fb, TexelFormat::PACKED)?;
    let field = bin(fb, BinOp::ShiftRightLogical, flags, shift)?;
    let four_bits = u1(fb, 0xF)?;
    let field = bin(fb, BinOp::BitwiseAnd, field, four_bits)?;
    let no = splat(fb, bform(1), 0)?;
    let (mut named, mut wide, mut away) = (no, no, no);
    let ones = u4(fb, 1)?;
    let (mut d, mut s, mut per, mut truncation) = (ones, ones, ones, ones);
    for p in PackedTexels::ALL {
        let (Some(grids), Some(cut)) = (p.filter_grids(), p.filter_truncation()) else {
            continue;
        };
        // A field without code scales (a YCbCr field's exact steps) recovers its codes on the
        // grids.
        let scales = p.scales().unwrap_or(grids);
        let k = u1(fb, p as u32)?;
        let is = bin(fb, BinOp::IEqual, field, k)?;
        let lanes = |fb: &mut FunctionBuilder<'_>, v: [u32; 4]| -> BResult<Value> {
            let mut l = Vec::with_capacity(4);
            for x in v {
                l.push(u1(fb, x)?);
            }
            fb.composite_construct(t.vu, &l)
        };
        let mut steps = grids;
        for (g, s) in steps.iter_mut().zip(scales) {
            *g /= s.max(1);
        }
        let these_d = lanes(fb, grids)?;
        let these_s = lanes(fb, scales)?;
        let these_per = lanes(fb, steps)?;
        let these_cut = lanes(fb, [cut; 4])?;
        d = fb.select(is, these_d, d)?;
        s = fb.select(is, these_s, s)?;
        per = fb.select(is, these_per, per)?;
        truncation = fb.select(is, these_cut, truncation)?;
        named = bin(fb, BinOp::LogicalOr, named, is)?;
        if p.filter_extended() {
            wide = bin(fb, BinOp::LogicalOr, wide, is)?;
        }
        if p.ycbcr() {
            away = bin(fb, BinOp::LogicalOr, away, is)?;
        }
    }
    let exact = flag(fb, flags, BicubicTexels::EXACT)?;
    let named = bin(fb, BinOp::LogicalAnd, named, exact)?;
    Ok(Some(PackedGrids {
        named,
        d,
        scales: s,
        per,
        truncation,
        wide,
        away,
    }))
}

/// The parameters of a [`Block`] routine, in order.
struct BlockParams {
    taps: Vec<Value>,
    kx: Value,
    ky: Value,
    kz: Value,
    flip: Value,
    flags: Value,
    half: Value,
}

fn block_params(fb: &mut FunctionBuilder<'_>, t: Tys) -> BResult<BlockParams> {
    let mut taps = Vec::with_capacity(8);
    for _ in 0..8 {
        taps.push(fb.param(t.vf)?);
    }
    Ok(BlockParams {
        taps,
        kx: fb.param(t.u)?,
        ky: fb.param(t.u)?,
        kz: fb.param(t.u)?,
        flip: fb.param(t.b)?,
        flags: fb.param(t.u)?,
        half: fb.param(t.b)?,
    })
}

/// Metal's weights of a linear 3D footprint's eight taps ([`Block`]'s order), `u32` scalars in
/// units of `2^-16` (probes): per tap the product `P = wx · wy · wz` of the axes' weights `256 - k`
/// and `k`, rounded to `P / 256` to nearest, a tie (`P mod 256 = 128`) up on the `t` axis's first
/// tap in Metal's frame (its second where `flip`) and down on the other. The weights sum to `2^16`:
/// a tie's partners on the `t` axis tie together.
fn block_weights(
    fb: &mut FunctionBuilder<'_>,
    (kx, ky, kz): (Value, Value, Value),
    flip: Value,
) -> BResult<Vec<Value>> {
    let k256 = u1(fb, 256)?;
    let pair = |fb: &mut FunctionBuilder<'_>, k: Value| -> BResult<[Value; 2]> {
        Ok([bin(fb, BinOp::ISub, k256, k)?, k])
    };
    let (wx, wy, wz) = (pair(fb, kx)?, pair(fb, ky)?, pair(fb, kz)?);
    let keep = fb.unary(UnOp::LogicalNot, flip)?;
    let (k8, k255, k128) = (u1(fb, 8)?, u1(fb, 255)?, u1(fb, 128)?);
    let (z, one) = (u1(fb, 0)?, u1(fb, 1)?);
    let mut w = Vec::with_capacity(8);
    for c in wz {
        for (b, first) in wy.into_iter().zip([keep, flip]) {
            for a in wx {
                let p = bin(fb, BinOp::IMul, a, b)?;
                let p = bin(fb, BinOp::IMul, p, c)?;
                let q = bin(fb, BinOp::ShiftRightLogical, p, k8)?;
                let r = bin(fb, BinOp::BitwiseAnd, p, k255)?;
                let above = bin(fb, BinOp::UGreaterThan, r, k128)?;
                let tie = bin(fb, BinOp::IEqual, r, k128)?;
                let tie = bin(fb, BinOp::LogicalAnd, tie, first)?;
                let up = bin(fb, BinOp::LogicalOr, above, tie)?;
                let inc = fb.select(up, one, z)?;
                w.push(bin(fb, BinOp::IAdd, q, inc)?);
            }
        }
    }
    Ok(w)
}

/// `fn(8 × vec4, kx, ky, kz, flip, flags, half) -> vec4` (module docs, [`Block::exact`]): Metal's
/// sum over the eight taps under their weights `w` ([`block_weights`]), each term truncated: `T =
/// Σ ⌊c · w / 2^s⌋` for the codes `c` on the format's grid, `s` 4 for a 12-bit grid (an 8-bit
/// format's `16 · code` and an sRGB step) and 8 for a 16-bit format's codes, then the step `⌊(T +
/// 2^(15 - s)) / 2^(16 - s)⌋`, clamped and divided by the grid exactly (an extended-range format's
/// `s` is 7 on a color lane and 4 on alpha). An unsigned format's product `c · w`
/// is below `2^32` (codes below `2^16`, weights at most `2^16`), a `u32` word shifted logically; a
/// signed format's codes (at most `2^15 - 1` in magnitude) keep it below `2^31`, an `i32` shifted
/// arithmetically (toward negative infinity). The terms' sum is below `2^24` in magnitude (the
/// weights sum to `2^16`).
fn define_exact_block(mb: &mut ModuleBuilder) -> BResult<FuncId> {
    let t = types(mb)?;
    let mut fb = mb.texel_routine(t.vf)?;
    let p = block_params(&mut fb, t)?;
    let fb_ = &mut fb;
    let g = grid(fb_, t, p.flags, Packed::Field)?;
    let w = block_weights(fb_, (p.kx, p.ky, p.kz), p.flip)?;
    let tr = Truncation::of(fb_, t, p.flags, &g)?;
    let mut total = u4(fb_, 0)?;
    for (x, w) in p.taps.into_iter().zip(w) {
        let c = code(fb_, &g, x)?;
        let w = broadcast(fb_, t.vu, w)?;
        let term = tr.term(fb_, c, w)?;
        total = bin(fb_, BinOp::IAdd, total, term)?;
    }
    let n = tr.step(fb_, total)?;
    let n = fb_.glsl(GlslOp::SClamp, &[n, g.lo, g.hi])?;
    let r = ratio(fb_, &g, n, p.half)?;
    finish(fb, r)
}

// ---- The emulated sums ----

/// Which lanes of the taps read so far hold a NaN, `+inf`, or `-inf` (`bool` lanes), decided on
/// the bits: a device that is not asked to preserve them may assume no float operand is one.
#[derive(Clone, Copy)]
struct Specials {
    nan: Value,
    pos: Value,
    neg: Value,
}

impl Specials {
    fn of(fb: &mut FunctionBuilder<'_>, t: Tys, x: Value) -> BResult<Specials> {
        let b = fb.convert(ConvOp::Bitcast, x, t.vu)?;
        let abs = u4(fb, 0x7FFF_FFFF)?;
        let mag = bin(fb, BinOp::BitwiseAnd, b, abs)?;
        let inf = u4(fb, 0x7F80_0000)?;
        let ninf = u4(fb, 0xFF80_0000)?;
        Ok(Specials {
            nan: bin(fb, BinOp::UGreaterThan, mag, inf)?,
            pos: bin(fb, BinOp::IEqual, b, inf)?,
            neg: bin(fb, BinOp::IEqual, b, ninf)?,
        })
    }

    /// No special in any lane.
    fn none(fb: &mut FunctionBuilder<'_>) -> BResult<Specials> {
        let f = splat(fb, bform(LANES), 0)?;
        Ok(Specials {
            nan: f,
            pos: f,
            neg: f,
        })
    }

    fn or(self, fb: &mut FunctionBuilder<'_>, o: Specials) -> BResult<Specials> {
        Ok(Specials {
            nan: bin(fb, BinOp::LogicalOr, self.nan, o.nan)?,
            pos: bin(fb, BinOp::LogicalOr, self.pos, o.pos)?,
            neg: bin(fb, BinOp::LogicalOr, self.neg, o.neg)?,
        })
    }

    /// `self` where `c` (a `bool` scalar), else `o`.
    fn or_else(self, fb: &mut FunctionBuilder<'_>, c: Value, o: Specials) -> BResult<Specials> {
        Ok(Specials {
            nan: fb.select(c, self.nan, o.nan)?,
            pos: fb.select(c, self.pos, o.pos)?,
            neg: fb.select(c, self.neg, o.neg)?,
        })
    }

    /// `r` where no tap is special, else Metal's answer: NaN for a NaN or for both infinities,
    /// else the infinity present.
    fn apply(self, fb: &mut FunctionBuilder<'_>, t: Tys, r: Value) -> BResult<Value> {
        let both = bin(fb, BinOp::LogicalAnd, self.pos, self.neg)?;
        let nan = bin(fb, BinOp::LogicalOr, self.nan, both)?;
        let bits = fb.convert(ConvOp::Bitcast, r, t.vu)?;
        let ninf = u4(fb, 0xFF80_0000)?;
        let bits = fb.select(self.neg, ninf, bits)?;
        let inf = u4(fb, 0x7F80_0000)?;
        let bits = fb.select(self.pos, inf, bits)?;
        let qnan = u4(fb, 0x7FC0_0000)?;
        let bits = fb.select(nan, qnan, bits)?;
        fb.convert(ConvOp::Bitcast, bits, t.vf)
    }
}

/// The bilinear weights `(256 - k) / 256` and `k / 256` of fraction `k` (exact `f32`), and
/// whether `k` is 0 (the second tap weighs nothing).
fn bilinear(fb: &mut FunctionBuilder<'_>, t: Tys, k: Value) -> BResult<(Value, Value, Value)> {
    let k256 = u1(fb, 256)?;
    let rest = bin(fb, BinOp::ISub, k256, k)?;
    let step = f1(fb, 1.0 / 256.0)?;
    let a = fb.convert(ConvOp::ConvertUToF, rest, t.f)?;
    let a = exact_op(fb, BinOp::FMul, a, step)?;
    let b = fb.convert(ConvOp::ConvertUToF, k, t.f)?;
    let b = exact_op(fb, BinOp::FMul, b, step)?;
    let z = u1(fb, 0)?;
    let zero = bin(fb, BinOp::IEqual, k, z)?;
    Ok((a, b, zero))
}

/// `a · wa + b · wb`, or `a` alone when `skip` (the second tap's weight is zero, and Metal skips
/// it, a NaN or an infinity there included).
fn lerp(
    fb: &mut FunctionBuilder<'_>,
    (a, b): (Value, Value),
    (wa, wb, skip): (Value, Value, Value),
) -> BResult<Value> {
    let x = exact_op(fb, BinOp::VectorTimesScalar, a, wa)?;
    let y = exact_op(fb, BinOp::VectorTimesScalar, b, wb)?;
    let s = exact_op(fb, BinOp::FAdd, x, y)?;
    fb.select(skip, a, s)
}

// ---- Two-word shifts ----

/// The low word of two-word `r` shifted right by `s` (`uvec4`, at most 64; `0` at 64).
fn shr_low(fb: &mut FunctionBuilder<'_>, r: Pair, s: Value) -> BResult<Value> {
    let (k31, k32, k64) = (u4(fb, 31)?, u4(fb, 32)?, u4(fb, 64)?);
    let one = u4(fb, 1)?;
    let sl = bin(fb, BinOp::BitwiseAnd, s, k31)?;
    let lo = bin(fb, BinOp::ShiftRightLogical, r.lo, sl)?;
    let back = bin(fb, BinOp::ISub, k31, sl)?;
    let cross = bin(fb, BinOp::ShiftLeftLogical, r.hi, back)?;
    let cross = bin(fb, BinOp::ShiftLeftLogical, cross, one)?;
    let near = bin(fb, BinOp::BitwiseOr, lo, cross)?;
    let far = bin(fb, BinOp::ShiftRightLogical, r.hi, sl)?;
    let big = bin(fb, BinOp::UGreaterThanEqual, s, k32)?;
    let v = fb.select(big, far, near)?;
    let gone = bin(fb, BinOp::UGreaterThanEqual, s, k64)?;
    let z = u4(fb, 0)?;
    fb.select(gone, z, v)
}

// ---- Metal's linear accumulator ----

/// The bits Metal's linear accumulator keeps below its reference exponent: 38 for a 32-bit float
/// format's sum, 17 for a half format's ([`accumulate`]).
const F32_GRID_BITS: u32 = 38;
const HALF_GRID_BITS: u32 = 17;

/// A tap's term for [`accumulate`] (`uvec4` and `bvec4` lanes): whether the term is negative,
/// the tap's biased exponent `x` (1 for a denormal), the product `p = m · W` of its significand
/// `m` (with the leading bit) and the integer weight's magnitude `W` (units of `2^-16`), so the
/// term's magnitude is `p · 2^(x - 166)`; `lead = x + msb(m) + msb(W)`, the term's reference
/// exponent plus 166 (the tap's exponent plus the weight's); and whether the term is present (a
/// nonzero tap of nonzero weight).
#[derive(Clone, Copy)]
struct AccTerm {
    neg: Value,
    x: Value,
    p: Pair,
    lead: Value,
    present: Value,
}

/// The leading bit of two-word `p` (`0` for zero).
fn msb2(fb: &mut FunctionBuilder<'_>, p: Pair) -> BResult<Value> {
    let (z, one, k32) = (u4(fb, 0)?, u4(fb, 1)?, u4(fb, 32)?);
    let hi = fb.glsl(GlslOp::FindUMsb, &[p.hi])?;
    let hi = bin(fb, BinOp::IAdd, hi, k32)?;
    let lo = fb.glsl(GlslOp::UMax, &[p.lo, one])?;
    let lo = fb.glsl(GlslOp::FindUMsb, &[lo])?;
    let upper = bin(fb, BinOp::INotEqual, p.hi, z)?;
    fb.select(upper, hi, lo)
}

/// The term ([`AccTerm`]) of tap `tap` (`vec4`) under weight `w` (a `u32` scalar, at most
/// `2^16`).
fn acc_term(fb: &mut FunctionBuilder<'_>, t: Tys, tap: Value, w: Value) -> BResult<AccTerm> {
    let b = fb.convert(ConvOp::Bitcast, tap, t.vu)?;
    let (z, one) = (u4(fb, 0)?, u4(fb, 1)?);
    let sign = u4(fb, 0x8000_0000)?;
    let sign = bin(fb, BinOp::BitwiseAnd, b, sign)?;
    let neg = bin(fb, BinOp::INotEqual, sign, z)?;
    let k23 = u4(fb, 23)?;
    let e = bin(fb, BinOp::ShiftRightLogical, b, k23)?;
    let byte = u4(fb, 0xFF)?;
    let e = bin(fb, BinOp::BitwiseAnd, e, byte)?;
    let frac = u4(fb, 0x007F_FFFF)?;
    let f = bin(fb, BinOp::BitwiseAnd, b, frac)?;
    let normal = bin(fb, BinOp::INotEqual, e, z)?;
    let lead_bit = u4(fb, 0x0080_0000)?;
    let lead_bit = fb.select(normal, lead_bit, z)?;
    let m = bin(fb, BinOp::BitwiseOr, f, lead_bit)?;
    let x = fb.glsl(GlslOp::UMax, &[e, one])?;
    let w = broadcast(fb, t.vu, w)?;
    let prod = bin(fb, BinOp::UMulExtended, m, w)?;
    let p = Pair {
        lo: fb.composite_extract(prod, &[0])?,
        hi: fb.composite_extract(prod, &[1])?,
    };
    let m1 = fb.glsl(GlslOp::UMax, &[m, one])?;
    let msb_m = fb.glsl(GlslOp::FindUMsb, &[m1])?;
    let w1 = fb.glsl(GlslOp::UMax, &[w, one])?;
    let msb_w = fb.glsl(GlslOp::FindUMsb, &[w1])?;
    let lead = bin(fb, BinOp::IAdd, x, msb_m)?;
    let lead = bin(fb, BinOp::IAdd, lead, msb_w)?;
    let some = bin(fb, BinOp::INotEqual, m, z)?;
    let weighed = bin(fb, BinOp::INotEqual, w, z)?;
    let present = bin(fb, BinOp::LogicalAnd, some, weighed)?;
    Ok(AccTerm {
        neg,
        x,
        p,
        lead,
        present,
    })
}

/// Metal's linear sum of the finite taps `taps` (`vec4`) under the integer weights (`u32`
/// scalars, units of `2^-16`) (measured on every lane of the P2a and P2c probes): each term `v · w`
/// truncated toward zero onto the grid `2^(E - b)`, `E` the largest of the present terms' `e(v) +
/// e(w)` (`e` the exponent, `⌊log2 |·|⌋`; a tap of zero weight and a zero tap are absent), `b` 17
/// for a half format (`half_format`, a `bool` scalar) and 38 for a 32-bit float one; the truncated
/// terms summed exactly; and the sum rounded once with ties away from zero, to `half` for a half
/// format, else to `f32`. A zero result, and an `f32` denormal one, is `+0`. Integer operations on
/// the taps' bits only, so every device and float mode answers alike. A NaN or an infinity
/// decomposes like a finite value; its lane's result is Metal's non-finite one instead
/// ([`Specials`]).
///
/// - Each term is `p · 2^(x - 166)` ([`AccTerm`]), `p < 2^41` (`m < 2^24`, `W <= 2^16`). The
///   grid is `2^g` for `g = L - 166 - b`, `L` the largest `lead` among the present terms. A term
///   in grid units is `p · 2^s` for `s = x + b - L <= b - msb(m) - msb(W)`, below `2^(b + 2)`
///   (`p < 2^(msb(m) + msb(W) + 2)`), and the sum of at most eight below `2^(b + 5)`: two-word
///   pairs, exact modulo `2^64`. A right shift truncates the magnitude (toward zero), 64 places or more to
///   zero.
/// - The rounding is [`round_scaled`]'s.
fn accumulate(
    fb: &mut FunctionBuilder<'_>,
    t: Tys,
    taps: &[(Value, Value)],
    half_format: Value,
) -> BResult<Value> {
    let mut terms = Vec::with_capacity(taps.len());
    for (tap, w) in taps {
        terms.push(acc_term(fb, t, *tap, *w)?);
    }
    let top = lead_top(fb, &terms)?;
    let (bh, bf) = (u1(fb, HALF_GRID_BITS)?, u1(fb, F32_GRID_BITS)?);
    let bits = fb.select(half_format, bh, bf)?;
    let bits = broadcast(fb, t.vu, bits)?;
    let sum = truncated_sum(fb, &terms, bits, top)?;
    // g = L - 166 - b.
    let k166 = u4(fb, 166)?;
    let g = bin(fb, BinOp::ISub, top, k166)?;
    let g = bin(fb, BinOp::ISub, g, bits)?;
    round_scaled(fb, t, sum, g, half_format)
}

/// The bits Metal's Catmull-Rom accumulator keeps below its references: 17 below each row's
/// ([`define_half_row`]), 20 below the rows' ([`define_half_plane`]).
const ROW_GRID_BITS: u32 = 17;
const ROWS_GRID_BITS: u32 = 20;

/// The low bits of the `f32` significand a half value leaves zero: its 10 fraction bits are the
/// top of `f32`'s 23, a half denormal's included (an `f32` normal).
const HALF_LOW_BITS: u32 = 13;

/// The leading bit of a nonzero half tap's significand without its zero bits ([`half_terms`]).
const LEAD_BIT: u32 = 23 - HALF_LOW_BITS;

/// The low bits of a half row's word that hold its reference `top_j` ([`define_half_row`]): a
/// row has a term of nonzero weight (Metal's weights at a fraction sum to one), so `top_j` is
/// `max(e, 50) + 10 + msb(min(w, 2^16 - 1))` of one, in `[60, 280]`, below `2^9`; the row's sum,
/// below `2^21` in magnitude, takes the word's high bits, below `2^30` once shifted.
const TOP_BITS: u32 = 9;

/// The reference of a term of zero weight and of a row of zero weight ([`half_terms`],
/// [`define_half_plane`]): below every other by more than any other's range, and twice it still
/// an `i32`.
const ABSENT: i32 = -(1 << 30);

/// The biased `f32` exponent a zero tap's reference reads ([`half_terms`]): above every zero
/// tap's shift that matters and below every nonzero half value's exponent (at least 103, `2^-24`).
const ZERO_EXPONENT: u32 = 50;

/// Weight `i` of `w4` (`uvec4` of `i32` bits) as its magnitude and whether it is negative (`u32`
/// and `bool` scalars).
fn weight_split(fb: &mut FunctionBuilder<'_>, w4: Value, i: u32) -> BResult<(Value, Value)> {
    let w = fb.composite_extract(w4, &[i])?;
    let zero = u1(fb, 0)?;
    let negative = bin(fb, BinOp::SLessThan, w, zero)?;
    let minus = fb.unary(UnOp::SNegate, w)?;
    Ok((fb.select(negative, minus, w)?, negative))
}

/// The weight `w` (an `i32` scalar's bits) as its magnitude and whether it is negative (`u32`
/// and `bool` scalars).
fn weight_of(fb: &mut FunctionBuilder<'_>, w: Value) -> BResult<(Value, Value)> {
    let zero = u1(fb, 0)?;
    let negative = bin(fb, BinOp::SLessThan, w, zero)?;
    let minus = fb.unary(UnOp::SNegate, w)?;
    Ok((fb.select(negative, minus, w)?, negative))
}

/// The `i32` scalar `v`, as `u32` bits.
fn i1(fb: &mut FunctionBuilder<'_>, v: i32) -> BResult<Value> {
    u1(fb, v as u32)
}

/// The `uvec4` of the `i32` `v` in every lane, as `u32` bits.
fn i4(fb: &mut FunctionBuilder<'_>, v: i32) -> BResult<Value> {
    u4(fb, v as u32)
}

/// `mag` (`uvec4`) with the sign of the `i32` lanes `v`.
fn signed_of(fb: &mut FunctionBuilder<'_>, v: Value, mag: Value) -> BResult<Value> {
    // `(mag ^ s) - s` for `s` the sign spread over the word (0 or -1).
    let k31 = u4(fb, 31)?;
    let s = bin(fb, BinOp::ShiftRightArithmetic, v, k31)?;
    let x = bin(fb, BinOp::BitwiseXor, mag, s)?;
    bin(fb, BinOp::ISub, x, s)
}

/// `max(u, v)` of `uvec4` lanes `u` and `v` read as `i32`.
fn smax(fb: &mut FunctionBuilder<'_>, u: Value, v: Value) -> BResult<Value> {
    fb.glsl(GlslOp::SMax, &[u, v])
}

/// One half row's terms ([`half_terms`], `uvec4` lanes): each term's product `p` as a
/// sign-magnitude word and the right shift's reference `at`, and the row's reference `top`.
struct HalfRow {
    terms: Vec<(Value, Value)>,
    top: Value,
}

/// The terms of row `taps` (`uvec4`, the taps' `f32` bits) of a half format under the weights
/// `wx` (`uvec4` of `i32` bits, units of `2^-16`), for Metal's row accumulator
/// ([`define_half_row`]).
///
/// A nonzero tap of a half format is an `f32` normal whose significand `m` has its low
/// [`HALF_LOW_BITS`] bits zero: with `m' = m >> 13` (11 bits, the leading one at bit 10) and the
/// biased exponent `e` (at least 103: `2^-24`, the least half value), the tap is `m' · 2^(e -
/// 137)`, and its term under weight `w` is `m' · w · 2^(e - 153)`, whose reference exponent
/// `e(v) + e(w)` is `lead - 153` for `lead = e + 10 + msb(min(w, 2^16 - 1))` (a cap of `2^16 - 1`
/// reads a unit weight's exponent as `-1`, Metal's row reference). The row's grid is `2^(top - 170)`
/// for `top` the largest `lead`, so the term in grid units is `m' · w · 2^s` for `s = e + 17 -
/// top <= 7 - msb(min(w, 2^16 - 1))`, below `2^19`, and the row's sum below `2^21`.
///
/// Each term is one right shift: `p = m' · (w << k)` for `k = max(7 - msb(min(w, 2^16 - 1)),
/// 0)` (below `2^28`), shifted right by `top - at` for `at = e + 10 + min(msb(min(w, 2^16 - 1)),
/// 7)` (at most `lead`, so never negative). `p` carries the term's sign at bit 31 (the tap's and
/// the weight's differing). A term of zero weight has `p = 0` and its `lead` [`ABSENT`] or below.
/// A zero tap's `lead` reads its exponent as [`ZERO_EXPONENT`]: below every nonzero tap's, so it
/// never sets a row's `top` beside one, and in a row of none its term `0x400 · w`, shifted right
/// 31 places (`top - at >= 50 - 17`), is 0, as is the row's sum. A NaN or an infinity decomposes
/// like a finite tap; its lane's result is Metal's non-finite one ([`define_half_plane`]).
fn half_terms(
    fb: &mut FunctionBuilder<'_>,
    t: Tys,
    taps: [Value; 4],
    wx: Value,
) -> BResult<HalfRow> {
    let (z, k8, k23) = (u1(fb, 0)?, u1(fb, 8)?, u1(fb, 23)?);
    let (low, k10) = (u1(fb, HALF_LOW_BITS)?, u1(fb, LEAD_BIT)?);
    // A term's left shift room below the row's grid: `17 - 10`.
    let room = ROW_GRID_BITS - LEAD_BIT;
    let (cap, one, seven, ten) = (
        u1(fb, 0xFFFF)?,
        u1(fb, 1)?,
        u1(fb, room)?,
        u1(fb, LEAD_BIT)?,
    );
    let (sign, unsigned, absent) = (u1(fb, 0x8000_0000)?, u1(fb, 0)?, i1(fb, ABSENT)?);
    let lead_bit = u4(fb, 1 << LEAD_BIT)?;
    let (sign4, zero_e) = (u4(fb, 0x8000_0000)?, u4(fb, ZERO_EXPONENT)?);
    let mut terms = Vec::with_capacity(4);
    let mut top = i4(fb, ABSENT)?;
    for (i, b) in (0u32..).zip(taps) {
        // The weight's facts (scalars): `w << k`, the sign flip, and the references' parts.
        let (w, negative) = weight_split(fb, wx, i)?;
        let wc = fb.glsl(GlslOp::UMin, &[w, cap])?;
        let wc = fb.glsl(GlslOp::UMax, &[wc, one])?;
        let msb = fb.glsl(GlslOp::FindUMsb, &[wc])?;
        let k = bin(fb, BinOp::ISub, seven, msb)?;
        let k = smax(fb, k, z)?;
        let wk = bin(fb, BinOp::ShiftLeftLogical, w, k)?;
        let flip = fb.select(negative, sign, unsigned)?;
        let c = bin(fb, BinOp::IAdd, msb, ten)?;
        let weighed = bin(fb, BinOp::INotEqual, w, z)?;
        let c = fb.select(weighed, c, absent)?;
        let low7 = fb.glsl(GlslOp::UMin, &[msb, seven])?;
        let c_at = bin(fb, BinOp::IAdd, low7, ten)?;
        let (wk, flip) = (broadcast(fb, t.vu, wk)?, broadcast(fb, t.vu, flip)?);
        let (c, c_at) = (broadcast(fb, t.vu, c)?, broadcast(fb, t.vu, c_at)?);
        // The tap's: `m'`, `e`, the product, its sign, and the references.
        let e = fb.bit_field_extract(false, b, k23, k8)?;
        let f = fb.bit_field_extract(false, b, low, k10)?;
        let m = bin(fb, BinOp::BitwiseOr, f, lead_bit)?;
        let p = bin(fb, BinOp::IMul, m, wk)?;
        let s = bin(fb, BinOp::BitwiseXor, b, flip)?;
        let s = bin(fb, BinOp::BitwiseAnd, s, sign4)?;
        let p = bin(fb, BinOp::BitwiseOr, p, s)?;
        let ez = fb.glsl(GlslOp::UMax, &[e, zero_e])?;
        let lead = bin(fb, BinOp::IAdd, ez, c)?;
        top = smax(fb, top, lead)?;
        terms.push((p, bin(fb, BinOp::IAdd, e, c_at)?));
    }
    Ok(HalfRow { terms, top })
}

/// The bits of `taps` (`vec4`).
fn tap_bits(fb: &mut FunctionBuilder<'_>, t: Tys, taps: [Value; 4]) -> BResult<[Value; 4]> {
    let [a, b, c, d] = taps;
    Ok([
        fb.convert(ConvOp::Bitcast, a, t.vu)?,
        fb.convert(ConvOp::Bitcast, b, t.vu)?,
        fb.convert(ConvOp::Bitcast, c, t.vu)?,
        fb.convert(ConvOp::Bitcast, d, t.vu)?,
    ])
}

/// The rows' pass's parts of row weight `wy` (an `i32` scalar's bits): `c`, `msb(|wy|)` or
/// [`ABSENT`] for a zero weight; `k = max(3 - msb(|wy|), 0)` and `min(msb(|wy|), 3)`; and the
/// sign flip (bit 31 where negative) ([`define_half_plane`]).
struct RowWeight {
    mag: Value,
    c: Value,
    k: Value,
    c_at: Value,
    flip: Value,
}

fn row_weight(fb: &mut FunctionBuilder<'_>, wy: Value) -> BResult<RowWeight> {
    let (w, negative) = weight_of(fb, wy)?;
    // A row term's left shift room below the rows' grid: `20 - 17`.
    let room = ROWS_GRID_BITS - ROW_GRID_BITS;
    let (z, one, three) = (u1(fb, 0)?, u1(fb, 1)?, u1(fb, room)?);
    let w1 = fb.glsl(GlslOp::UMax, &[w, one])?;
    let msb = fb.glsl(GlslOp::FindUMsb, &[w1])?;
    let weighed = bin(fb, BinOp::INotEqual, w, z)?;
    let absent = i1(fb, ABSENT)?;
    let c = fb.select(weighed, msb, absent)?;
    let k = bin(fb, BinOp::ISub, three, msb)?;
    let k = smax(fb, k, z)?;
    let c_at = fb.glsl(GlslOp::UMin, &[msb, three])?;
    let (sign, unsigned) = (u1(fb, 0x8000_0000)?, u1(fb, 0)?);
    let flip = fb.select(negative, sign, unsigned)?;
    Ok(RowWeight {
        mag: w,
        c,
        k,
        c_at,
        flip,
    })
}

/// `fn(t0: vec4, t1: vec4, t2: vec4, t3: vec4, wx: uvec4, smax: uvec4, umax: uvec4) -> { uvec4,
/// uvec4, uvec4 }` ([`Plane::row`]): a half format's row accumulator over row `j` (taps `t0 ..
/// t3`; measured on every lane of the P2d probe and of the float fixtures, and answering every lane
/// of the held-out P2e and P2f): the row's sum `R_j` and reference `top_j` as one word `R_j · 2^9 +
/// top_j` ([`TOP_BITS`]), and the largest bits of the taps read so far as `i32` (`smax`) and as
/// `u32` (`umax`), which decide Metal's non-finite result ([`define_half_plane`]).
///
/// The row's terms `v · wx_i` are truncated toward zero onto the grid `2^(E_j - 17)`, `E_j` the
/// largest present `e(v) + e(wx_i)` of the row (`e` the exponent, `⌊log2 |·|⌋`; a zero tap and a
/// tap of zero weight are absent; `e(w)` of a unit weight, at an `x` fraction of 0, read as
/// `-1`), and summed exactly: `R_j`, in units of `2^(top_j - 170)` for `top_j = E_j + 153`
/// ([`half_terms`]), each term `p` shifted right by `top_j - at` (at most 31 places).
///
/// Every tap of a half format is a half value, the small floats' (`RG11B10Float`,
/// `RGB9E5Float`) and a border's included, which Vulkan converts to `f32` exactly: every quantity
/// is one word per lane.
fn define_half_row(mb: &mut ModuleBuilder) -> BResult<FuncId> {
    let t = types(mb)?;
    let out = mb.ty(Type::Struct {
        members: [t.vu, t.vu, t.vu].into(),
        layout: None,
    })?;
    let mut fb = mb.routine(out)?;
    let taps = [
        fb.param(t.vf)?,
        fb.param(t.vf)?,
        fb.param(t.vf)?,
        fb.param(t.vf)?,
    ];
    let wx = fb.param(t.vu)?;
    let mut hi = fb.param(t.vu)?;
    let mut uhi = fb.param(t.vu)?;
    let fb_ = &mut fb;
    let bits = tap_bits(fb_, t, taps)?;
    for b in bits {
        hi = smax(fb_, hi, b)?;
        uhi = fb_.glsl(GlslOp::UMax, &[uhi, b])?;
    }
    let row = half_terms(fb_, t, bits, wx)?;
    let (z, k31) = (u4(fb_, 0)?, u4(fb_, 31)?);
    let magnitude = u4(fb_, 0x7FFF_FFFF)?;
    let mut r = z;
    for (p, at) in row.terms {
        let d = bin(fb_, BinOp::ISub, row.top, at)?;
        let d = fb_.glsl(GlslOp::UMin, &[d, k31])?;
        let a = bin(fb_, BinOp::BitwiseAnd, p, magnitude)?;
        let mag = bin(fb_, BinOp::ShiftRightLogical, a, d)?;
        let signed = signed_of(fb_, p, mag)?;
        r = bin(fb_, BinOp::IAdd, r, signed)?;
    }
    let top_bits = u4(fb_, TOP_BITS)?;
    let r = bin(fb_, BinOp::ShiftLeftLogical, r, top_bits)?;
    let r = bin(fb_, BinOp::BitwiseOr, r, row.top)?;
    let r = fb_.composite_construct(out, &[r, hi, uhi])?;
    finish(fb, r)
}

/// `fn(row0 .. row3: uvec4, wy: uvec4, smax: uvec4, umax: uvec4) -> vec4` ([`Plane::half`]): a
/// half format's Catmull-Rom result from its rows' sums `R_j` and references `top_j`
/// ([`define_half_row`]'s words) under the rows' weights `wy` (`i32` bits).
///
/// - The rows' terms `R_j · wy_j` are truncated toward zero onto the grid `2^(E - 20)`, `E` the
///   largest `E_j + e(wy_j)` of the rows with a present tap and a nonzero `wy_j` (the rows'
///   inputs' exponents, not their sums'; `top = E + 169` is the largest `top_j + msb(|wy_j|)`, a
///   row of zero weight never raising it), and summed exactly. `R_j · |wy_j|` (below `2^37`,
///   `OpUMulExtended`) counts `2^(top_j - 186)` and the grid is `2^(top - 189)`, so the term is
///   `R_j · |wy_j| · 2^(top_j - top + 3)`, below `2^25`, and the four's sum below `2^27`. It is
///   one right shift: `R_j · (|wy_j| << max(3 - msb(|wy_j|), 0))` (below `2^38`) by
///   `top - top_j - min(msb(|wy_j|), 3)`, never negative for a row with a present tap and a
///   nonzero weight (any other row's term is 0).
/// - The sum is rounded once to `half` with ties away from zero ([`round_half`]); then Metal's
///   non-finite result where a tap of the footprint is one, whatever its weight (the reading): NaN
///   for a NaN or for both infinities, else the infinity present. On the footprint's largest bits:
///   a NaN makes `smax` (as `i32`) above `+inf`'s bits or `umax` above `-inf`'s; without one,
///   `+inf` is present exactly where `smax` is its bits, and `-inf` where `umax` is. The result can
///   exceed `half`'s range (the weights are not all positive): an infinity. A half format's sample
///   returns the `half` value whether the sample's result is `half` or `float`.
///
/// The unit weight's exponent is read on the rows' pass only: a unit `wy_j` (a `y` fraction of
/// 0) is the one row of nonzero weight, whose `R_j` lies on its row's grid, far coarser than the
/// rows' grid either reading gives.
fn define_half_plane(mb: &mut ModuleBuilder) -> BResult<FuncId> {
    let t = types(mb)?;
    let mut fb = mb.routine(t.vf)?;
    let rows = [
        fb.param(t.vu)?,
        fb.param(t.vu)?,
        fb.param(t.vu)?,
        fb.param(t.vu)?,
    ];
    let wy = fb.param(t.vu)?;
    let hi = fb.param(t.vu)?;
    let uhi = fb.param(t.vu)?;
    let fb_ = &mut fb;
    let (top_bits, top_mask) = (u4(fb_, TOP_BITS)?, u4(fb_, (1 << TOP_BITS) - 1)?);
    let mut sums = Vec::with_capacity(4);
    let mut tops = Vec::with_capacity(4);
    for row in rows {
        sums.push(bin(fb_, BinOp::ShiftRightArithmetic, row, top_bits)?);
        tops.push(bin(fb_, BinOp::BitwiseAnd, row, top_mask)?);
    }
    // The rows' reference.
    let mut weights = Vec::with_capacity(4);
    let mut top = i4(fb_, ABSENT)?;
    for (j, &top_j) in (0u32..).zip(&tops) {
        let w = fb_.composite_extract(wy, &[j])?;
        let rw = row_weight(fb_, w)?;
        let c = broadcast(fb_, t.vu, rw.c)?;
        let l = bin(fb_, BinOp::IAdd, top_j, c)?;
        top = smax(fb_, top, l)?;
        weights.push(rw);
    }
    // The rows' terms on the rows' grid.
    let k31 = u4(fb_, 31)?;
    let mut sum = u4(fb_, 0)?;
    for ((r, top_j), rw) in sums.into_iter().zip(tops).zip(weights) {
        let spread = bin(fb_, BinOp::ShiftRightArithmetic, r, k31)?;
        let r_abs = bin(fb_, BinOp::BitwiseXor, r, spread)?;
        let r_abs = bin(fb_, BinOp::ISub, r_abs, spread)?;
        let wk = bin(fb_, BinOp::ShiftLeftLogical, rw.mag, rw.k)?;
        let wk = broadcast(fb_, t.vu, wk)?;
        let prod = bin(fb_, BinOp::UMulExtended, r_abs, wk)?;
        let p = Pair {
            lo: fb_.composite_extract(prod, &[0])?,
            hi: fb_.composite_extract(prod, &[1])?,
        };
        let c_at = broadcast(fb_, t.vu, rw.c_at)?;
        let at = bin(fb_, BinOp::IAdd, top_j, c_at)?;
        let d = bin(fb_, BinOp::ISub, top, at)?;
        let mag = shr_low(fb_, p, d)?;
        let flip = broadcast(fb_, t.vu, rw.flip)?;
        let sign = bin(fb_, BinOp::BitwiseXor, r, flip)?;
        let term = signed_of(fb_, sign, mag)?;
        sum = bin(fb_, BinOp::IAdd, sum, term)?;
    }
    // g = top - 189: the term's 153, the weight's 16, and the rows' grid bits.
    let k189 = u4(fb_, 153 + 16 + ROWS_GRID_BITS)?;
    let g = bin(fb_, BinOp::ISub, top, k189)?;
    let r = round_half(fb_, t, sum, g)?;
    let (inf, ninf) = (u4(fb_, 0x7F80_0000)?, u4(fb_, 0xFF80_0000)?);
    let pos = bin(fb_, BinOp::IEqual, hi, inf)?;
    let neg = bin(fb_, BinOp::IEqual, uhi, ninf)?;
    let nan_pos = bin(fb_, BinOp::SGreaterThan, hi, inf)?;
    let nan_neg = bin(fb_, BinOp::UGreaterThan, uhi, ninf)?;
    let nan = bin(fb_, BinOp::LogicalOr, nan_pos, nan_neg)?;
    let found = Specials { nan, pos, neg };
    let r = found.apply(fb_, t, r)?;
    finish(fb, r)
}

/// `S · 2^g` (`S` an `i32` sum below `2^31` in magnitude, `g` signed `i32` lanes) rounded once to
/// `half` with ties away from zero, as an `f32` value: [`round_scaled`]'s rounding on one word
/// (`p` 11, `floor` `-24`), a zero result `+0`, and one past `half`'s range (at least `2^16`
/// once rounded) the infinity of its sign, as Metal's `half` result.
fn round_half(fb: &mut FunctionBuilder<'_>, t: Tys, sum: Value, g: Value) -> BResult<Value> {
    let (z, one) = (u4(fb, 0)?, u4(fb, 1)?);
    let negative = bin(fb, BinOp::SLessThan, sum, z)?;
    let minus = fb.unary(UnOp::SNegate, sum)?;
    let a = fb.select(negative, minus, sum)?;
    let a1 = fb.glsl(GlslOp::UMax, &[a, one])?;
    let msb = fb.glsl(GlslOp::FindUMsb, &[a1])?;
    let keep = u4(fb, 10)?;
    let at = bin(fb, BinOp::ISub, msb, keep)?;
    let floor = u4(fb, (-24i32) as u32)?;
    let floor = bin(fb, BinOp::ISub, floor, g)?;
    let at = fb.glsl(GlslOp::SMax, &[at, floor])?;
    let k31 = u4(fb, 31)?;
    let s = bin(fb, BinOp::ISub, at, one)?;
    let s = fb.glsl(GlslOp::SClamp, &[s, z, k31])?;
    let rounded = bin(fb, BinOp::ShiftRightLogical, a, s)?;
    let rounded = bin(fb, BinOp::IAdd, rounded, one)?;
    let rounded = bin(fb, BinOp::ShiftRightLogical, rounded, one)?;
    // `at <= 0` only where `msb <= 10`: `a · 2^-at` is below `2^11`.
    let neg_at = fb.unary(UnOp::SNegate, at)?;
    let neg_at = fb.glsl(GlslOp::SClamp, &[neg_at, z, k31])?;
    let exact = bin(fb, BinOp::ShiftLeftLogical, a, neg_at)?;
    let rounds = bin(fb, BinOp::SGreaterThan, at, z)?;
    let q = fb.select(rounds, rounded, exact)?;
    // The bits: exponent field `n + at + g + 127` for q's leading bit n, the significand below n.
    let q1 = fb.glsl(GlslOp::UMax, &[q, one])?;
    let lead = fb.glsl(GlslOp::FindUMsb, &[q1])?;
    let field = bin(fb, BinOp::IAdd, lead, at)?;
    let field = bin(fb, BinOp::IAdd, field, g)?;
    let k127 = u4(fb, 127)?;
    let field = bin(fb, BinOp::IAdd, field, k127)?;
    let k24 = u4(fb, 24)?;
    let up = bin(fb, BinOp::ISub, k24, lead)?;
    let sig = bin(fb, BinOp::ShiftLeftLogical, q, up)?;
    let sig = bin(fb, BinOp::ShiftRightLogical, sig, one)?;
    let frac = u4(fb, 0x007F_FFFF)?;
    let sig = bin(fb, BinOp::BitwiseAnd, sig, frac)?;
    let k23 = u4(fb, 23)?;
    let e = bin(fb, BinOp::ShiftLeftLogical, field, k23)?;
    let bits = bin(fb, BinOp::BitwiseOr, e, sig)?;
    // Past `half`'s range: `2^16` and above, the field `127 + 16`.
    let inf = u4(fb, 0x7F80_0000)?;
    let k143 = u4(fb, 127 + 16)?;
    let past = bin(fb, BinOp::SGreaterThanEqual, field, k143)?;
    let bits = fb.select(past, inf, bits)?;
    let sign = u4(fb, 0x8000_0000)?;
    let signed = bin(fb, BinOp::BitwiseOr, bits, sign)?;
    let bits = fb.select(negative, signed, bits)?;
    let none = bin(fb, BinOp::IEqual, q, z)?;
    let tiny = bin(fb, BinOp::SLessThanEqual, field, z)?;
    let zero = bin(fb, BinOp::LogicalOr, none, tiny)?;
    let bits = fb.select(zero, z, bits)?;
    fb.convert(ConvOp::Bitcast, bits, t.vf)
}

/// The largest `lead` of the present terms among `terms` (`uvec4`; `0` where none is present:
/// every present term's `lead` is at least 1).
fn lead_top(fb: &mut FunctionBuilder<'_>, terms: &[AccTerm]) -> BResult<Value> {
    let z = u4(fb, 0)?;
    let mut top = z;
    for term in terms {
        let l = fb.select(term.present, term.lead, z)?;
        top = fb.glsl(GlslOp::UMax, &[top, l])?;
    }
    Ok(top)
}

/// The sum of `terms`, each truncated toward zero onto the grid `2^(top - 166 - bits)` (`bits`
/// and `top` `uvec4` lanes, [`lead_top`]), as a sign-extended two-word integer in grid units: a
/// term is `p · 2^s` for `s = x + bits - top`, shifted left for `s >= 0`, else right (toward
/// zero; 64 places or more to zero).
fn truncated_sum(
    fb: &mut FunctionBuilder<'_>,
    terms: &[AccTerm],
    bits: Value,
    top: Value,
) -> BResult<Pair> {
    let z = u4(fb, 0)?;
    let k64 = u4(fb, 64)?;
    let zero = Pair { lo: z, hi: z };
    let mut sum = zero;
    for term in terms {
        // s = x + b - L (signed).
        let s = bin(fb, BinOp::IAdd, term.x, bits)?;
        let s = bin(fb, BinOp::ISub, s, top)?;
        let left = bin(fb, BinOp::SGreaterThanEqual, s, z)?;
        let up = wide::shl(fb, LANES, term.p, s)?;
        let back = fb.unary(UnOp::SNegate, s)?;
        let down = wide::lshr(fb, LANES, term.p, back)?;
        let gone = bin(fb, BinOp::UGreaterThanEqual, back, k64)?;
        let down = wide::sel(fb, gone, zero, down)?;
        let mag = wide::sel(fb, left, up, down)?;
        let mag = wide::sel(fb, term.present, mag, zero)?;
        let neg = wide::neg(fb, LANES, mag)?;
        let signed = wide::sel(fb, term.neg, neg, mag)?;
        sum = wide::add(fb, sum, signed)?;
    }
    Ok(sum)
}

/// `S · 2^g` (`S` a sign-extended two-word sum, `g` signed `i32` lanes) rounded once with ties
/// away from zero to `half` when `half_format` (a `bool` scalar), else to `f32`; a zero result,
/// and an `f32` denormal one, is `+0`.
///
/// The magnitude `A` rounds at bit `r = max(M - p + 1, floor - g)` (`M` its leading bit, `p` 24
/// or 11 significand bits, `floor` the format's least denormal exponent, `-149` or `-24`): `q =
/// (⌊A / 2^(r - 1)⌋ + 1) / 2` when `r > 0` (below `2^25`), else `A · 2^-r` (`A` then below
/// `2^24`). The result is `q · 2^(r + g)`, assembled from `q`'s leading bit as an `f32` (a `half`
/// result is an `f32` value), its exponent field at most 0 giving `+0`. A finite convex sum lies
/// within its taps' range, so no result overflows.
fn round_scaled(
    fb: &mut FunctionBuilder<'_>,
    t: Tys,
    sum: Pair,
    g: Value,
    half_format: Value,
) -> BResult<Value> {
    let (z, one) = (u4(fb, 0)?, u4(fb, 1)?);
    let negative = bin(fb, BinOp::SLessThan, sum.hi, z)?;
    let flipped = wide::neg(fb, LANES, sum)?;
    let a = wide::sel(fb, negative, flipped, sum)?;
    let msb = msb2(fb, a)?;
    let (p10, p23) = (u1(fb, 10)?, u1(fb, 23)?);
    let keep = fb.select(half_format, p10, p23)?;
    let keep = broadcast(fb, t.vu, keep)?;
    let (f24, f149) = (u1(fb, (-24i32) as u32)?, u1(fb, (-149i32) as u32)?);
    let floor = fb.select(half_format, f24, f149)?;
    let floor = broadcast(fb, t.vu, floor)?;
    let at = bin(fb, BinOp::ISub, msb, keep)?;
    let floor = bin(fb, BinOp::ISub, floor, g)?;
    let at = fb.glsl(GlslOp::SMax, &[at, floor])?;
    let k64 = u4(fb, 64)?;
    let s = bin(fb, BinOp::ISub, at, one)?;
    let s = fb.glsl(GlslOp::SClamp, &[s, z, k64])?;
    let rounded = shr_low(fb, a, s)?;
    let rounded = bin(fb, BinOp::IAdd, rounded, one)?;
    let rounded = bin(fb, BinOp::ShiftRightLogical, rounded, one)?;
    let neg_at = fb.unary(UnOp::SNegate, at)?;
    let neg_at = fb.glsl(GlslOp::SMax, &[neg_at, z])?;
    let exact = bin(fb, BinOp::ShiftLeftLogical, a.lo, neg_at)?;
    let rounds = bin(fb, BinOp::SGreaterThan, at, z)?;
    let q = fb.select(rounds, rounded, exact)?;
    // The bits: exponent field `n + r + g + 127` for q's leading bit n, the significand below n.
    let q1 = fb.glsl(GlslOp::UMax, &[q, one])?;
    let lead = fb.glsl(GlslOp::FindUMsb, &[q1])?;
    let field = bin(fb, BinOp::IAdd, lead, at)?;
    let field = bin(fb, BinOp::IAdd, field, g)?;
    let k127 = u4(fb, 127)?;
    let field = bin(fb, BinOp::IAdd, field, k127)?;
    let k24 = u4(fb, 24)?;
    let up = bin(fb, BinOp::ISub, k24, lead)?;
    let sig = bin(fb, BinOp::ShiftLeftLogical, q, up)?;
    let sig = bin(fb, BinOp::ShiftRightLogical, sig, one)?;
    let frac = u4(fb, 0x007F_FFFF)?;
    let sig = bin(fb, BinOp::BitwiseAnd, sig, frac)?;
    let k23 = u4(fb, 23)?;
    let e = bin(fb, BinOp::ShiftLeftLogical, field, k23)?;
    let bits = bin(fb, BinOp::BitwiseOr, e, sig)?;
    let sign = u4(fb, 0x8000_0000)?;
    let signed = bin(fb, BinOp::BitwiseOr, bits, sign)?;
    let bits = fb.select(negative, signed, bits)?;
    let none = bin(fb, BinOp::IEqual, q, z)?;
    let tiny = bin(fb, BinOp::SLessThanEqual, field, z)?;
    let zero = bin(fb, BinOp::LogicalOr, none, tiny)?;
    let bits = fb.select(zero, z, bits)?;
    fb.convert(ConvOp::Bitcast, bits, t.vf)
}

/// Whether the format's bits state a half format and a 32-bit float format (`bool` scalars).
#[derive(Clone, Copy)]
struct Kind {
    half: Value,
    linear: Value,
}

impl Kind {
    fn of(fb: &mut FunctionBuilder<'_>, flags: Value) -> BResult<Kind> {
        Ok(Kind {
            half: flag(fb, flags, BicubicTexels::HALF)?,
            linear: flag(fb, flags, BicubicTexels::LINEAR)?,
        })
    }
}

/// The format's rounding of the emulated sum `r` and Metal's non-finite results (`specials`
/// of the taps it reads), then the `half` result's conversion when `half`: a 32-bit float
/// format's result truncated toward zero and saturated (Metal's conversion of an `f32` sum to a
/// `half` sample), a half format's already a `half`, any other rounded to nearest even.
fn round_result(
    fb: &mut FunctionBuilder<'_>,
    t: Tys,
    flags: Value,
    r: Value,
    specials: Specials,
    half: Value,
) -> BResult<Value> {
    let kind = Kind::of(fb, flags)?;
    let r = format_round(fb, t, kind, r, specials)?;
    result_of(fb, kind, r, half)
}

/// The format's value of the emulated sum `r`: a half format's rounded to `half` with ties away
/// from zero, then, for a 32-bit float or half format, Metal's non-finite result where `specials`
/// (of the taps the sum reads) holds one.
fn format_round(
    fb: &mut FunctionBuilder<'_>,
    t: Tys,
    kind: Kind,
    r: Value,
    specials: Specials,
) -> BResult<Value> {
    let f = fform(LANES);
    let rh = texel_round::f32_to_half(fb, f, r, HalfRounding::NearestAway)?;
    let r = fb.select(kind.half, rh, r)?;
    let float = bin(fb, BinOp::LogicalOr, kind.half, kind.linear)?;
    let none = Specials::none(fb)?;
    let specials = specials.or_else(fb, float, none)?;
    specials.apply(fb, t, r)
}

/// The sample's result of the format's value `r`: `r`, or when `half` (a `bool` scalar) its
/// `half` conversion ([`round_result`]).
fn result_of(fb: &mut FunctionBuilder<'_>, kind: Kind, r: Value, half: Value) -> BResult<Value> {
    let f = fform(LANES);
    let hz = texel_round::f32_to_half(fb, f, r, HalfRounding::TowardZeroSaturating)?;
    let he = texel_round::f32_to_half(fb, f, r, HalfRounding::NearestEven)?;
    let other = fb.select(kind.half, r, he)?;
    let as_half = fb.select(kind.linear, hz, other)?;
    fb.select(half, as_half, r)
}

/// `fn(16 × vec4, wx: uvec4, wy: uvec4, flags: u32, half: bool) -> vec4` ([`Plane::cubic`]): the
/// Catmull-Rom sum of a format neither half nor 32-bit float (module docs) in `f32` operations
/// with `NoContraction`: each row's horizontal sum, then the rows', under Metal's weights as
/// exact `f32` values (`w · 2^-16`); the normalized formats' clamps; and a `half` result rounded
/// to nearest even. Its non-finite taps take the `f32` operations' result.
fn define_cubic_plane(mb: &mut ModuleBuilder) -> BResult<FuncId> {
    let t = types(mb)?;
    let mut fb = mb.routine(t.vf)?;
    let row = |fb: &mut FunctionBuilder<'_>| -> BResult<[Value; 4]> {
        Ok([
            fb.param(t.vf)?,
            fb.param(t.vf)?,
            fb.param(t.vf)?,
            fb.param(t.vf)?,
        ])
    };
    let grid = [row(&mut fb)?, row(&mut fb)?, row(&mut fb)?, row(&mut fb)?];
    let wx = fb.param(t.vu)?;
    let wy = fb.param(t.vu)?;
    let flags = fb.param(t.u)?;
    let half = fb.param(t.b)?;
    let fb_ = &mut fb;
    // Metal's weights as exact f32 values: `w · 2^-16`.
    let unit = f1(fb_, 1.0 / 65536.0)?;
    let scaled = |fb: &mut FunctionBuilder<'_>, w: Value| -> BResult<Vec<Value>> {
        let w = fb.convert(ConvOp::ConvertSToF, w, t.vf)?;
        let w = exact_op(fb, BinOp::VectorTimesScalar, w, unit)?;
        (0u32..4).map(|i| fb.composite_extract(w, &[i])).collect()
    };
    let fwx = scaled(fb_, wx)?;
    let fwy = scaled(fb_, wy)?;
    let weighted = |fb: &mut FunctionBuilder<'_>, xs: &[Value], ws: &[Value]| -> BResult<Value> {
        let mut acc: Option<Value> = None;
        for (x, w) in xs.iter().zip(ws) {
            let term = exact_op(fb, BinOp::VectorTimesScalar, *x, *w)?;
            acc = Some(match acc {
                Some(a) => exact_op(fb, BinOp::FAdd, a, term)?,
                None => term,
            });
        }
        match acc {
            Some(a) => Ok(a),
            None => f4(fb, 0.0),
        }
    };
    // Each row's horizontal sum, then the rows'.
    let mut rows = Vec::with_capacity(4);
    for row in &grid {
        rows.push(weighted(fb_, row, &fwx)?);
    }
    let r = weighted(fb_, &rows, &fwy)?;
    // The normalized formats' clamps.
    let (zero, one, neg) = (f4(fb_, 0.0)?, f4(fb_, 1.0)?, f4(fb_, -1.0)?);
    let unorm = fb_.glsl(GlslOp::FClamp, &[r, zero, one])?;
    let is_unorm = flag(fb_, flags, BicubicTexels::UNORM)?;
    let r = fb_.select(is_unorm, unorm, r)?;
    let snorm = fb_.glsl(GlslOp::FClamp, &[r, neg, one])?;
    let is_snorm = flag(fb_, flags, BicubicTexels::SNORM)?;
    let r = fb_.select(is_snorm, snorm, r)?;
    let he = texel_round::f32_to_half(fb_, fform(LANES), r, HalfRounding::NearestEven)?;
    let r = fb_.select(half, he, r)?;
    finish(fb, r)
}

/// `fn(t0, t1, k, flags, half) -> vec4` (module docs, [`Line::float`]): a 32-bit float or half
/// format's sum through Metal's linear accumulator ([`accumulate`], under [`Span`]'s weights),
/// every other format's bilinear in `f32` operations; non-finite taps of nonzero weight decide
/// the result as IEEE arithmetic would (measured for a 32-bit float format, inferred for a half
/// one).
fn define_float_line(mb: &mut ModuleBuilder) -> BResult<FuncId> {
    let t = types(mb)?;
    let mut fb = mb.routine(t.vf)?;
    let t0 = fb.param(t.vf)?;
    let t1 = fb.param(t.vf)?;
    let k = fb.param(t.u)?;
    let flags = fb.param(t.u)?;
    let half = fb.param(t.b)?;
    let fb_ = &mut fb;
    let w = bilinear(fb_, t, k)?;
    let r = lerp(fb_, (t0, t1), w)?;
    let kind = Kind::of(fb_, flags)?;
    let float = bin(fb_, BinOp::LogicalOr, kind.half, kind.linear)?;
    let (k256, k8) = (u1(fb_, 256)?, u1(fb_, 8)?);
    let rest = bin(fb_, BinOp::ISub, k256, k)?;
    let w0 = bin(fb_, BinOp::ShiftLeftLogical, rest, k8)?;
    let w1 = bin(fb_, BinOp::ShiftLeftLogical, k, k8)?;
    let accumulated = accumulate(fb_, t, &[(t0, w0), (t1, w1)], kind.half)?;
    let r = fb_.select(float, accumulated, r)?;
    let (_, _, zero) = w;
    let s0 = Specials::of(fb_, t, t0)?;
    let s1 = Specials::of(fb_, t, t1)?;
    let none = Specials::none(fb_)?;
    let s1 = none.or_else(fb_, zero, s1)?;
    let specials = s0.or(fb_, s1)?;
    let r = round_result(fb_, t, flags, r, specials, half)?;
    finish(fb, r)
}

/// The specials of the taps of nonzero weight among `taps` (each with whether its weight is
/// zero, a `bool` scalar).
fn weighed_specials(
    fb: &mut FunctionBuilder<'_>,
    t: Tys,
    taps: &[(Value, Option<Value>)],
) -> BResult<Specials> {
    let none = Specials::none(fb)?;
    let mut acc = none;
    for (x, zero) in taps {
        let s = Specials::of(fb, t, *x)?;
        let s = match zero {
            Some(z) => none.or_else(fb, *z, s)?,
            None => s,
        };
        acc = acc.or(fb, s)?;
    }
    Ok(acc)
}

/// Metal's result of the accumulated sum `r` (`accumulate`) for format kind `kind`: Metal's
/// non-finite result where `specials` holds one, then the `half` result's conversion when `half`
/// ([`result_of`]).
fn accumulated_result(
    fb: &mut FunctionBuilder<'_>,
    t: Tys,
    kind: Kind,
    (r, specials): (Value, Specials),
    half: Value,
) -> BResult<Value> {
    let r = specials.apply(fb, t, r)?;
    result_of(fb, kind, r, half)
}

/// `fn(t00, t01, t10, t11, kx, ky, flags, half) -> vec4` (module docs, [`Quad::float`]): Metal's
/// accumulator ([`accumulate`]) over the four taps under the weights `(256 - kx) · (256 - ky)`,
/// `kx · (256 - ky)`, `(256 - kx) · ky`, and `kx · ky`; Metal's non-finite results on the taps of
/// nonzero weight; and the `half` result's conversion.
fn define_float_quad(mb: &mut ModuleBuilder, calls: Calls) -> BResult<FuncId> {
    let t = types(mb)?;
    let mut fb = start(calls, mb, t.vf)?;
    let q = quad_params(&mut fb, t)?;
    let fb_ = &mut fb;
    let k256 = u1(fb_, 256)?;
    let rx = bin(fb_, BinOp::ISub, k256, q.kx)?;
    let ry = bin(fb_, BinOp::ISub, k256, q.ky)?;
    let w00 = bin(fb_, BinOp::IMul, rx, ry)?;
    let w01 = bin(fb_, BinOp::IMul, q.kx, ry)?;
    let w10 = bin(fb_, BinOp::IMul, rx, q.ky)?;
    let w11 = bin(fb_, BinOp::IMul, q.kx, q.ky)?;
    let kind = Kind::of(fb_, q.flags)?;
    let taps = [(q.t00, w00), (q.t01, w01), (q.t10, w10), (q.t11, w11)];
    let r = accumulate(fb_, t, &taps, kind.half)?;
    let z = u1(fb_, 0)?;
    let x0 = bin(fb_, BinOp::IEqual, q.kx, z)?;
    let y0 = bin(fb_, BinOp::IEqual, q.ky, z)?;
    let xy0 = bin(fb_, BinOp::LogicalOr, x0, y0)?;
    let specials = weighed_specials(
        fb_,
        t,
        &[
            (q.t00, None),
            (q.t01, Some(x0)),
            (q.t10, Some(y0)),
            (q.t11, Some(xy0)),
        ],
    )?;
    let r = accumulated_result(fb_, t, kind, (r, specials), q.half)?;
    finish(fb, r)
}

/// `fn(8 × vec4, kx, ky, kz, flip, flags, half) -> vec4` (module docs, [`Block::float`]): Metal's
/// accumulator ([`accumulate`]) over the eight taps under their weights ([`block_weights`]);
/// Metal's non-finite results on the taps of nonzero product weight `wx · wy · wz`, that is on
/// every tap but the second of an axis whose fraction is 0 (a tap whose product rounds to a zero
/// weight is one of them: an Apple M3 Max's 3D samples of `RG11B10Float` texels
/// holding infinities, where two such taps' infinities are the result and the 37 of zero product
/// weight are not; inferred for the other float formats, whose sums are the same accumulator's);
/// and the `half` result's conversion.
fn define_float_block(mb: &mut ModuleBuilder) -> BResult<FuncId> {
    let t = types(mb)?;
    let mut fb = mb.texel_routine(t.vf)?;
    let p = block_params(&mut fb, t)?;
    let fb_ = &mut fb;
    let w = block_weights(fb_, (p.kx, p.ky, p.kz), p.flip)?;
    let kind = Kind::of(fb_, p.flags)?;
    let taps: Vec<(Value, Value)> = p.taps.into_iter().zip(w).collect();
    let r = accumulate(fb_, t, &taps, kind.half)?;
    let z = u1(fb_, 0)?;
    // An axis's second tap weighs nothing where its fraction is 0 (its first weighs 256 - k, never
    // 0).
    let mut still = Vec::with_capacity(3);
    for k in [p.kx, p.ky, p.kz] {
        still.push(bin(fb_, BinOp::IEqual, k, z)?);
    }
    let mut weighed = Vec::with_capacity(taps.len());
    for (n, (x, _)) in (0u32..).zip(&taps) {
        let mut dead = None;
        for (axis, s) in (0u32..).zip(&still) {
            if n >> axis & 1 == 1 {
                dead = Some(match dead {
                    Some(d) => bin(fb_, BinOp::LogicalOr, d, *s)?,
                    None => *s,
                });
            }
        }
        weighed.push((*x, dead));
    }
    let specials = weighed_specials(fb_, t, &weighed)?;
    let r = accumulated_result(fb_, t, kind, (r, specials), p.half)?;
    finish(fb, r)
}

/// `fn(t0, t1, k, flags, half) -> vec4` (module docs, [`Span::float`]): Metal's accumulator
/// ([`accumulate`]) over the two taps under the weights `(256 - k) · 256` and `k · 256`; Metal's
/// non-finite results on the taps of nonzero weight; and the `half` result's conversion.
fn define_float_span(mb: &mut ModuleBuilder) -> BResult<FuncId> {
    let t = types(mb)?;
    let mut fb = mb.texel_routine(t.vf)?;
    let t0 = fb.param(t.vf)?;
    let t1 = fb.param(t.vf)?;
    let k = fb.param(t.u)?;
    let flags = fb.param(t.u)?;
    let half = fb.param(t.b)?;
    let fb_ = &mut fb;
    let (k256, k8) = (u1(fb_, 256)?, u1(fb_, 8)?);
    let rest = bin(fb_, BinOp::ISub, k256, k)?;
    let w0 = bin(fb_, BinOp::ShiftLeftLogical, rest, k8)?;
    let w1 = bin(fb_, BinOp::ShiftLeftLogical, k, k8)?;
    let kind = Kind::of(fb_, flags)?;
    let r = accumulate(fb_, t, &[(t0, w0), (t1, w1)], kind.half)?;
    let z = u1(fb_, 0)?;
    let zero = bin(fb_, BinOp::IEqual, k, z)?;
    let specials = weighed_specials(fb_, t, &[(t0, None), (t1, Some(zero))])?;
    let r = accumulated_result(fb_, t, kind, (r, specials), half)?;
    finish(fb, r)
}
