//! Correctly rounded fused multiply-add on `float` and `half`.
//!
//! Metal's `fma` (`air.fma`, `llvm.fma`) rounds the exact `a * b + c` once. Vulkan's
//! `GLSL.std.450 Fma` does not promise that (its precision is inherited from an `OpFMul`
//! followed by an `OpFAdd`), so a `float` fused multiply-add takes one of three routes, by
//! what the profile offers ([`float_route`]):
//!
//! - [`Route::Khr`]: `OpFmaKHR` (`SPV_KHR_fma`, `shaderFmaFloat32` of `VK_KHR_shader_fma`),
//!   which the Vulkan precision table lists as correctly rounded;
//! - [`Route::Wide`]: in `double` (`shaderFloat64` and `shaderInt64`): the product of two
//!   `float`s is exact in `double`, the sum rounds once, and its rounding error is exact
//!   (FastTwoSum), so the sum is moved to the `double` neighbour of odd significand when it was
//!   inexact (round to odd). One conversion to `float` then rounds the exact sum correctly,
//!   because `double` keeps more than `24 + 2` significand bits (Boldo and Melquiond);
//! - [`Route::Soft`]: in 32-bit integers on every other profile: the 48-bit product of the
//!   significands, the addend aligned to it with a sticky bit, one 64-bit sum or difference, and
//!   one rounding to nearest even, with overflow to infinity.
//!
//! The two routes the translation computes itself answer as Metal's `fma` does (a probe cases
//! e98b103f and f7850ba7, every lane): an `f32` denormal operand is read as a zero of its sign, a
//! result whose exact value is below `2^-126` in magnitude is a zero of its sign, flushed before
//! it is rounded (Metal's `fma(-2^-126, 2^-126, 2^-126)` is `+0`, where the
//! correctly rounded sum is `2^-126`), and a NaN result is `0x7FC00000` (as Metal's functions read
//! their operands). `OpFmaKHR` keeps the device's denormal mode and NaN encoding, as the device's
//! other `float` arithmetic does; on MoltenVK it is Metal's own `fma`. Elsewhere it rounds before
//! it flushes, if it flushes: an exact value just below `2^-126` answers `2^-126`. Where the
//! profile asks Metal's answers (`FloatMathsPolicy::Exact`), a result of `OpFmaKHR` at most
//! `2^-126` in magnitude is answered by a routine ([`tiny_route`], [`checked`]); every larger one
//! is Metal's already. Where the profile has `double`, that routine decides only what such a
//! result needs, the exact sum's sign and whether it lies below `2^-126` ([`small`]), about 40
//! instructions where the whole `double` route is about 90: the device's compiler inlines a call,
//! so each checked use compiles its copy.
//!
//! The wide and integer routes are one module-scope function per lane count ([`Routine`]),
//! which every use calls, so a module holds each once (as the wide divisions are). A module that
//! calls a route on one lane and on more holds the route's sequence once, on one lane: its routines
//! on more lanes call that one on each lane ([`complete`], [`define_per_lane`]), and so does one
//! that calls a route only on two or more vector lane counts where the sequences that saves
//! outweigh the per-lane routines. The sequence is lane-wise, so each lane's answer is the same
//! either way; only the module's size changes (the integer route's sequence is about 1,200 to 1,800
//! words, a per-lane routine about 90 to 140). Both answer an infinite or NaN operand without their
//! core: the addend where the factors are finite (a finite product plus an infinity or NaN is
//! that), else the unfused `a * b + c` (an infinite or NaN product, and the sum IEEE's fused result
//! shares with it); and an exact zero sum or a zero factor with the unfused result, which rounds
//! nothing there.
//!
//! A `half` fused multiply-add needs no route ([`Route::HalfOdd`]): its product is exact in
//! `float` and within the normal range, so the device's `float` `Fma` is the exact sum rounded
//! once, fused or not, and that sum with its exact error, moved to odd as above, rounds once
//! correctly to `half` (`float` keeps `11 + 2` bits and more). It is a module-scope function per
//! lane count too (on one lane and per lane, as above), which every use calls: the `half` pairs
//! Metal contracts make it as frequent as the `half` arithmetic itself, and its sequence is about
//! 40 instructions. A `bfloat` one keeps its own sequence (`narrow::bf16_fma`): its product is
//! exact in `float`, so the device's `Fma`, fused or not, rounds the exact sum once, and its tie is
//! corrected.
//!
//! Where the driver is measured to compute `GLSL.std.450 Fma` on 16-bit floats as Metal's own
//! `half` `fma` (`DriverDeviation::MetalHalfFma`: MoltenVK 1.4.1 on Apple's GPUs) and the profile
//! has `shaderFloat16`, a `half` one takes [`Route::F16`] instead: its operands, each a `half` held
//! in a `float`, convert to 16 bits exactly, the device's `Fma` rounds the exact sum once as Metal
//! does, and its result widens exactly; the module then requires `VkProperty::MetalHalfFma`. It is
//! a few instructions per use and no routine, where MoltenVK inlines [`Route::HalfOdd`]'s routine
//! at every call and Metal's back end compiles each copy: one corpus kernel with 800 `half` fused
//! multiply-adds took 6 s to create through the routine.
//!
//! Where the profile has `shaderFmaFloat16` and rounds to `half` under the 16-bit float controls
//! ([`QuantizeForm::Controlled`], [`narrow::controlled_modes`]), a `half` one takes
//! [`Route::F16Khr`]: its operands convert to 16 bits exactly, `OpFmaKHR` on them rounds the exact
//! sum once, and its result widens exactly. Vulkan's precision table lists `OpFmaKHR` as "Correctly
//! rounded", "according to IEEE 754 roundTiesToEven" under `RoundingModeRTE`, with denormals kept
//! under `DenormPreserve` and zero signs under `SignedZeroInfNanPreserve`, which the entry then
//! declares (`narrow::computes_in_16_bits`): the one rounding [`Route::HalfOdd`]'s round-to-odd
//! sequence emulates, and Metal's `half` `fma`, with no fact. A NaN's bits are the device's, as the
//! controlled rounding's are; a fused pair's NaN is selected as on every route
//! (`super::contract`). It replaces that sequence, or native device maths' `float` `Fma` and its
//! rounding to `half` ([`Route::Native`], noted `NativeFma`), with one instruction whose
//! conversions the device folds into its neighbours'.

use crate::device::profile::FloatFunctions;
use crate::device::DeviceProfile;
use crate::error::DriverDeviation;
use crate::lower::arith::legal::{FloatFmt, Form, Held};
use crate::lower::arith::narrow::{self, u32_form};
use crate::lower::arith::ops::{fsplat, splat, ty};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn, Merged};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder, QuantizeForm};
use crate::spirv::error::{BResult, BuildError, TypeRule};
use crate::spirv::ids::{FuncId, Value};
use crate::spirv::ir::{BinOp, Contraction, ConvOp, GlslOp, Type, UnOp};
use std::collections::{BTreeMap, BTreeSet};

/// How a fused multiply-add computes its exact result.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Route {
    /// `OpFmaKHR` on `float`.
    Khr,
    /// The `double` round-to-odd routine ([`Routine`]).
    Wide,
    /// The 32-bit integer routine ([`Routine`]).
    Soft,
    /// A `half` fused multiply-add: the exact `float` sum, rounded to odd, then to `half`.
    HalfOdd,
    /// A `half` fused multiply-add by the device's `Fma` on 16-bit floats, Metal's own
    /// ([`half_route`]).
    F16,
    /// A `half` fused multiply-add by `OpFmaKHR` on 16-bit floats under the 16-bit float
    /// controls, correctly rounded by the Vulkan spec ([`half_route`]).
    F16Khr,
    /// Native device maths (`FloatFunctions::Native`): the device's `GLSL.std.450 Fma` on the
    /// `float` lanes (a `half`'s held in `float`, its result rounded to `half`), within
    /// Vulkan's precision rather than rounded once (`PrecisionNote::NativeFma`).
    Native,
}

impl Route {
    /// Whether the route is the device's own `float` instruction, at the device's denormals and
    /// NaN encoding (`OpFmaKHR`, or native device maths' `Fma`), where the routines answer as
    /// Metal does.
    pub fn device_float(self) -> bool {
        matches!(self, Route::Khr | Route::Native)
    }
}

/// The census names of the fused multiply-add strategies (one per [`Route`]).
pub const KHR: &str = "arith.fma.khr";
/// See [`KHR`]: `OpFmaKHR` with the routine of its results at most `2^-126` ([`tiny_route`]).
pub const KHR_CHECKED: &str = "arith.fma.khr.checked";
/// See [`KHR`].
pub const WIDE: &str = "arith.fma.double";
/// See [`KHR`].
pub const SOFT: &str = "arith.fma.integer";
/// See [`KHR`].
pub const HALF: &str = "arith.fma.half";
/// See [`KHR`].
pub const F16: &str = "arith.fma.f16";
/// See [`KHR`].
pub const F16_KHR: &str = "arith.fma.f16.khr";
/// See [`KHR`].
pub const NATIVE: &str = "arith.fma.native";

/// Every route [`float_route`] answers, which the tests run each of.
#[cfg(test)]
pub const FLOAT_ROUTES: [Route; 3] = [Route::Khr, Route::Wide, Route::Soft];

/// The `float` route profile `p` offers: `OpFmaKHR` where `shaderFmaFloat32` is enabled, else the
/// device's `Fma` under native device maths, else `double` where `shaderFloat64` and `shaderInt64`
/// are, else the integer routine.
pub fn float_route(p: &DeviceProfile) -> Route {
    let leaf = &p.fields().leaf;
    if leaf.fma32 {
        Route::Khr
    } else if p.fields().float_functions == FloatFunctions::Native {
        Route::Native
    } else if leaf.float64 && leaf.int64 {
        Route::Wide
    } else {
        Route::Soft
    }
}

/// The route of the routine that answers a `float` fused multiply-add by `OpFmaKHR` whose result
/// is at most `2^-126` in magnitude ([`checked`]), on profile `p`: where its `float` route is
/// [`Route::Khr`] and its functions are to answer as Metal's (`FloatFunctions::Device`, the
/// policy's `FloatMathsPolicy::Exact`), [`Route::Wide`] where `shaderFloat64` and `shaderInt64` are
/// enabled (the routine of those results alone, [`small`]), else [`Route::Soft`]. A device that reads `f32` denormals as Metal does
/// (`denormal::device_flushes`) is checked too: it rounds before it flushes, as every device off
/// MoltenVK does (`denormal::checks_tiny_products`). `None` elsewhere: `OpFmaKHR` is then Metal's
/// `fma` (MoltenVK), or the device's rounding before its flush is left to it (native device maths,
/// `docs/FAST-PATHS.md`).
pub fn tiny_route(p: &DeviceProfile) -> Option<Route> {
    let fields = p.fields();
    if float_route(p) != Route::Khr || fields.float_functions != FloatFunctions::Device {
        return None;
    }
    Some(if fields.leaf.float64 && fields.leaf.int64 {
        Route::Wide
    } else {
        Route::Soft
    })
}

/// The `half` route profile `p` offers: [`Route::F16`] where the driver is measured to compute a
/// 16-bit `Fma` as Metal's own `half` `fma` (`DriverDeviation::MetalHalfFma`) and `shaderFloat16`
/// is enabled, else [`Route::F16Khr`] where `shaderFmaFloat16` is and the profile rounds to `half`
/// under the 16-bit float controls (which need `shaderFloat16`), under either policy since it is
/// Metal's, else the device's `float` `Fma` rounded to `half` under native device maths, else
/// [`Route::HalfOdd`].
pub fn half_route(p: &DeviceProfile) -> Route {
    if p.takes_driver_deviation(DriverDeviation::MetalHalfFma) && p.fields().leaf.float16 {
        Route::F16
    } else if p.fields().leaf.fma16 && narrow::quantize_form(p) == QuantizeForm::Controlled {
        Route::F16Khr
    } else if p.fields().float_functions == FloatFunctions::Native {
        Route::Native
    } else {
        Route::HalfOdd
    }
}

/// The routes a profile's fused multiply-adds take, by format ([`routes`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Routes {
    /// A `float` one's: [`float_route`].
    pub float: Route,
    /// A `half` one's: [`half_route`].
    pub half: Route,
    /// The routine route a `float` one by [`Route::Khr`] recomputes its results at most `2^-126`
    /// in magnitude by: [`tiny_route`].
    pub tiny: Option<Route>,
}

/// The routes profile `p` offers ([`float_route`], [`half_route`]).
pub fn routes(p: &DeviceProfile) -> Routes {
    Routes {
        float: float_route(p),
        half: half_route(p),
        tiny: tiny_route(p),
    }
}

/// A fused multiply-add on floats of `form` (`float` or `half` lanes) by `route`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Fma {
    pub form: Form,
    pub route: Route,
    /// On [`Route::Khr`], the route of the routine that recomputes a result at most `2^-126` in
    /// magnitude ([`tiny_route`], [`checked`]); `None` on every other route.
    pub tiny: Option<Route>,
}

impl Fma {
    /// The fused multiply-add on `form` on profile `p`, by the profile's [`routes`]. `None` for
    /// any other form.
    pub fn on(p: &DeviceProfile, form: Form) -> Option<Fma> {
        Fma::by(routes(p), form)
    }

    /// The fused multiply-add on `form` by `routes` (a profile's [`routes`]): `half` by its
    /// `half` route, `float` by its `float` route. `None` for any other form.
    pub fn by(routes: Routes, form: Form) -> Option<Fma> {
        let (route, tiny) = match form.float_fmt()? {
            FloatFmt::Half => (routes.half, None),
            FloatFmt::F32 => (
                routes.float,
                routes.tiny.filter(|_| routes.float == Route::Khr),
            ),
            FloatFmt::BFloat | FloatFmt::F64 => return None,
        };
        Some(Fma { form, route, tiny })
    }

    /// Whether it is `OpFmaKHR` whose results at most `2^-126` in magnitude its routine recomputes
    /// ([`checked`]): on operands read as Metal reads them, Metal's `fma` exactly, never a
    /// denormal.
    pub fn checked(self) -> bool {
        self.route == Route::Khr && self.tiny.is_some()
    }

    /// This fused multiply-add without the check of its small results ([`checked`]), for a use
    /// whose exact result is never below `2^-126` in magnitude but zero, where `OpFmaKHR` is
    /// Metal's `fma` already, and which must stay in one block.
    pub fn unchecked(self) -> Fma {
        Fma { tiny: None, ..self }
    }

    /// The shared routine the operation calls, if its route has one: the route's, or on
    /// [`Route::Khr`] the routine of its small results ([`Fma::tiny`]): on [`Route::Wide`] the
    /// one that decides only those ([`small`]), named by [`Route::Khr`].
    pub fn routine(self) -> Option<Routine> {
        let route = match (self.route, self.tiny) {
            (Route::Wide | Route::Soft | Route::HalfOdd, _) => self.route,
            (Route::Khr, Some(Route::Wide)) => Route::Khr,
            (Route::Khr, Some(tiny)) => tiny,
            (Route::Khr | Route::F16 | Route::F16Khr | Route::Native, _) => return None,
        };
        Some(Routine {
            route,
            lanes: self.form.lanes,
        })
    }

    /// The census name.
    pub fn name(self) -> &'static str {
        match self.route {
            Route::Khr if self.checked() => KHR_CHECKED,
            Route::Khr => KHR,
            Route::Wide => WIDE,
            Route::Soft => SOFT,
            Route::HalfOdd => HALF,
            Route::F16 => F16,
            Route::F16Khr => F16_KHR,
            Route::Native => NATIVE,
        }
    }

    /// An upper estimate of the SPIR-V words per use: the instruction or the call; on
    /// [`Route::F16`] and [`Route::F16Khr`] the three conversions, the instruction and the
    /// widening; checked
    /// ([`checked`]), the instruction, the test of its magnitude, the selection, the call and the
    /// merge. A routine's own words are the module's, charged once ([`Routine::words`]).
    pub fn words(self) -> u32 {
        match self.route {
            Route::Khr if self.checked() => 112,
            Route::Khr => 16,
            Route::Wide | Route::Soft | Route::HalfOdd => 24,
            Route::F16 | Route::F16Khr => 32,
            // The instruction, and a `half` result's rounding.
            Route::Native => 64,
        }
    }
}

/// A module-scope fused multiply-add routine on `float` lanes (`half` values held in `float` for
/// [`Route::HalfOdd`]): `route` is [`Route::Wide`], [`Route::Soft`], or [`Route::HalfOdd`]; or
/// [`Route::Khr`], the routine of `OpFmaKHR`'s results at most `2^-126` in magnitude through
/// `double` ([`small`]), which takes that result too.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Routine {
    pub route: Route,
    pub lanes: u8,
}

impl Routine {
    /// An upper estimate of the routine's SPIR-V words, with its constants and types, defined by
    /// [`define`] in a module whose `half` rounding takes form `q`.
    pub fn words(self, q: QuantizeForm) -> u32 {
        match self.route {
            Route::Wide => 640,
            // 43 instructions of at most six words, eight constants (a vector's carry their lanes),
            // the `double` types, and the function's type, frame and four parameters.
            Route::Khr => 340 + 25 * u32::from(self.lanes),
            // The sequence, its rounding to `half`, and the function's frame.
            Route::HalfOdd => 272 + narrow::round_words_in(q, half_form(self.lanes)) + 64,
            _ => 1800,
        }
    }

    /// An upper estimate of the routine's SPIR-V words defined per lane ([`define_per_lane`]):
    /// its types, frame and three parameters, and per lane three extractions and a call (5 and
    /// 7 words), and the result's construction.
    pub fn per_lane_words(self) -> u32 {
        48 + 24 * u32::from(self.lanes)
    }

    /// An upper estimate of the routine's SPIR-V words among a module's routines `fmas`
    /// ([`complete`]), its `half` rounding in form `q`: per lane where [`per_lane`] says so, else
    /// its sequence's.
    pub fn words_among(self, fmas: &BTreeSet<Routine>, q: QuantizeForm) -> u32 {
        if per_lane(self, fmas) {
            self.per_lane_words()
        } else {
            self.words(q)
        }
    }

    /// The float container widths the routine computes in: 32, and 64 for those in `double`
    /// ([`Route::Wide`], and [`Route::Khr`]'s [`small`]). It answers Metal's zero of its sign
    /// through float instructions at those widths (an exact zero sum by the device's `Fma` or the
    /// unfused `a * b + c`, a `float` result below `2^-126` flushed to the zero of its sign),
    /// whose zero signs Vulkan lets a device ignore without `SignedZeroInfNanPreserve`: where the
    /// profile's functions answer as Metal's, the module keeps them at each width
    /// (`ArithPlan::keep_zero_signs`).
    pub fn widths(self) -> &'static [u8] {
        match self.route {
            Route::Wide | Route::Khr => &[32, 64],
            _ => &[32],
        }
    }

    /// The routine of its route on one lane.
    fn one_lane(self) -> Routine {
        Routine {
            route: self.route,
            lanes: 1,
        }
    }

    /// The census name of the strategies that call it.
    pub fn name(self) -> &'static str {
        match self.route {
            Route::Khr => KHR_CHECKED,
            Route::Wide => WIDE,
            Route::HalfOdd => HALF,
            _ => SOFT,
        }
    }
}

/// Complete a module's routines `fmas`, each a route on a lane count its uses call, so that the
/// module holds a route's sequence once where that makes it smaller. A route the module calls on
/// one lane and on more already does: its routines on more lanes call the one-lane routine on each
/// lane ([`per_lane`]), each smaller than a sequence. A route called only on `k` (two or more)
/// vector lane counts holds `k` sequences, or one and `k` per-lane routines with its one-lane
/// routine, which it gets where the `k - 1` sequences that saves are, by their estimate, at least
/// twice the per-lane routines together: the estimates are upper bounds, and in a module a sequence
/// shares its constants and types, so a smaller margin grew modules (`half`'s sequence, called on
/// two and four lanes), and weighing one sequence where `k - 1` go grew those of the `double` route
/// on two, three and four lanes. A route called on one lane count keeps its one routine. A `half`
/// sequence holds its rounding to `half`, weighed in the module's form `q`: the integer rounding
/// (`QuantizeForm::Integer`) makes it the larger, so that its routines on two and four lanes call
/// the one-lane routine there.
pub fn complete(fmas: &mut BTreeSet<Routine>, q: QuantizeForm) {
    let add: Vec<Routine> = fmas
        .iter()
        .filter(|r| r.route != Route::Khr)
        .map(|r| r.one_lane())
        .filter(|one| !fmas.contains(one))
        .filter(|one| {
            let vectors: Vec<Routine> = fmas
                .iter()
                .copied()
                .filter(|r| r.route == one.route)
                .collect();
            let per_lane: u32 = vectors.iter().map(|r| r.per_lane_words()).sum();
            let saved = vectors.len().saturating_sub(1) as u32;
            saved > 0 && saved * one.words(q) >= 2 * per_lane
        })
        .collect();
    fmas.extend(add);
}

/// Whether routine `r`, among a module's routines `fmas` ([`complete`]), is defined per lane
/// ([`define_per_lane`]): it is on two or more lanes, and the module holds its route's routine
/// on one lane. The routine of small results ([`Route::Khr`], [`small`]) never is.
pub fn per_lane(r: Routine, fmas: &BTreeSet<Routine>) -> bool {
    r.lanes > 1 && r.route != Route::Khr && fmas.contains(&r.one_lane())
}

/// Define routine `r` among a module's routines `fmas` ([`complete`]), given the functions of
/// those `defined` before it, in `fmas`' order (a route's routine on one lane sorts before its
/// routines on more lanes): per lane where [`per_lane`] says so ([`define_per_lane`]), else its
/// sequence ([`define`]).
pub fn define_among(
    mb: &mut ModuleBuilder,
    r: Routine,
    fmas: &BTreeSet<Routine>,
    defined: &BTreeMap<Routine, FuncId>,
) -> BResult<FuncId> {
    if !per_lane(r, fmas) {
        return define(mb, r);
    }
    let one_lane = defined
        .get(&r.one_lane())
        .copied()
        .ok_or(BuildError::Type(TypeRule::FunctionSignature))?;
    define_per_lane(mb, r, one_lane)
}

/// Define routine `r`: a function of three `float` values of `r.lanes` lanes returning their
/// correctly rounded fused multiply-add; on [`Route::Khr`], of those three and their `OpFmaKHR`,
/// returning Metal's `fma` where that is at most `2^-126` in magnitude ([`small`]).
pub fn define(mb: &mut ModuleBuilder, r: Routine) -> BResult<FuncId> {
    let f = f32_form(r.lanes);
    let e = mb.ty(Type::Float { width: 32 })?;
    let t = if r.lanes == 1 {
        e
    } else {
        mb.ty(Type::Vector {
            elem: e,
            len: r.lanes,
        })?
    };
    let mut fb = mb.routine(t)?;
    let (a, b, c) = (fb.param(t)?, fb.param(t)?, fb.param(t)?);
    let v = match r.route {
        Route::Khr => {
            let fused = fb.param(t)?;
            small(&mut fb, f, [a, b, c], fused)?
        }
        Route::Wide => wide(&mut fb, f, a, b, c)?,
        Route::Soft => soft(&mut fb, f, a, b, c)?,
        Route::HalfOdd => {
            let h = half_form(r.lanes);
            let s = half_sum(&mut fb, h, a, b, c)?;
            narrow::round(&mut fb, h, s)?
        }
        Route::F16 | Route::F16Khr | Route::Native => {
            return Err(BuildError::Type(TypeRule::FunctionSignature))
        }
    };
    fb.ret(Some(v))?;
    fb.finish()
}

/// Define routine `r`, on two or more lanes, per lane: a function of three `float` values of
/// `r.lanes` lanes that calls `one_lane`, its route's routine on one lane ([`define`]), on each
/// lane, and returns the lanes' answers. Each lane's answer is `one_lane`'s, which is the lane's
/// answer of [`define`]'s routine on `r.lanes` lanes (its sequence is lane-wise).
pub fn define_per_lane(mb: &mut ModuleBuilder, r: Routine, one_lane: FuncId) -> BResult<FuncId> {
    let e = mb.ty(Type::Float { width: 32 })?;
    let t = mb.ty(Type::Vector {
        elem: e,
        len: r.lanes,
    })?;
    let mut fb = mb.routine(t)?;
    let params = [fb.param(t)?, fb.param(t)?, fb.param(t)?];
    let mut lanes = Vec::with_capacity(usize::from(r.lanes));
    for l in 0..u32::from(r.lanes) {
        let xs = params
            .iter()
            .map(|p| fb.composite_extract(*p, &[l]))
            .collect::<BResult<Vec<_>>>()?;
        let v = fb
            .call(one_lane, &xs)?
            .ok_or(BuildError::Type(TypeRule::FunctionSignature))?;
        lanes.push(v);
    }
    let v = fb.composite_construct(t, &lanes)?;
    fb.ret(Some(v))?;
    fb.finish()
}

/// Emit `fma` on `args` (three floats of `fma.form`); `routine` is the declared function of
/// [`Fma::routine`] when it has one. The result is a value of the form's format.
pub fn emit(
    fb: &mut FunctionBuilder<'_>,
    fma: Fma,
    args: &[Value],
    routine: Option<FuncId>,
) -> BResult<Value> {
    let [a, b, c] = *args else {
        return Err(BuildError::Type(TypeRule::FunctionSignature));
    };
    match (fma.route, fma.form.float_fmt()) {
        (Route::HalfOdd, Some(FloatFmt::Half))
        | (Route::Khr | Route::Wide | Route::Soft, Some(FloatFmt::F32)) => {
            float(fb, fma, a, b, c, routine)
        }
        (Route::F16, Some(FloatFmt::Half)) => f16(fb, fma.form.lanes, [a, b, c], false),
        (Route::F16Khr, Some(FloatFmt::Half)) => f16(fb, fma.form.lanes, [a, b, c], true),
        // Native device maths: the device's `Fma`; on `half` operands held in `float` the product
        // is exact, so only the sum's rounding and the rounding to `half` differ from Metal's one
        // rounding.
        (Route::Native, Some(FloatFmt::F32)) => fb.glsl(GlslOp::Fma, &[a, b, c]),
        (Route::Native, Some(FloatFmt::Half)) => {
            let r = fb.glsl(GlslOp::Fma, &[a, b, c])?;
            narrow::round(fb, fma.form, r)
        }
        _ => Err(BuildError::Type(TypeRule::FunctionSignature)),
    }
}

/// The `half` fused multiply-add of `a`, `b`, `c` (`half` values held in `float` lanes of
/// `lanes` lanes) by the device's `Fma` on 16-bit floats ([`Route::F16`]), or by `OpFmaKHR` on
/// them where `khr` ([`Route::F16Khr`]): each operand converted to 16 bits, exactly since it is a
/// `half`, the instruction, and its result widened, exactly. The device rounds the exact sum once,
/// as Metal's `half` `fma` does: measured where the profile takes `DriverDeviation::MetalHalfFma`,
/// by the Vulkan spec for `OpFmaKHR` under the 16-bit float controls.
fn f16(
    fb: &mut FunctionBuilder<'_>,
    lanes: u8,
    [a, b, c]: [Value; 3],
    khr: bool,
) -> BResult<Value> {
    let lanes_of = |fb: &mut FunctionBuilder<'_>, width: u8| -> BResult<_> {
        let e = fb.module().ty(Type::Float { width })?;
        match lanes {
            1 => Ok(e),
            len => fb.module().ty(Type::Vector { elem: e, len }),
        }
    };
    let (h, w) = (lanes_of(fb, 16)?, lanes_of(fb, 32)?);
    let narrow = |fb: &mut FunctionBuilder<'_>, x: Value| fb.convert(ConvOp::FConvert, x, h);
    let (a, b, c) = (narrow(fb, a)?, narrow(fb, b)?, narrow(fb, c)?);
    let r = if khr {
        fb.fma(a, b, c)?
    } else {
        fb.glsl(GlslOp::Fma, &[a, b, c])?
    };
    fb.convert(ConvOp::FConvert, r, w)
}

/// The `half` lanes of `lanes` lanes.
fn half_form(lanes: u8) -> Form {
    Form {
        held: Held::Float(FloatFmt::Half),
        lanes,
    }
}

/// The `float` lanes of `lanes` lanes.
fn f32_form(lanes: u8) -> Form {
    Form {
        held: Held::Float(FloatFmt::F32),
        lanes,
    }
}

/// The correctly rounded fused multiply-add of `a`, `b`, `c` by `fma`'s route: `OpFmaKHR`
/// ([`checked`] where [`Fma::checked`]), or a call of the route's routine.
fn float(
    fb: &mut FunctionBuilder<'_>,
    fma: Fma,
    a: Value,
    b: Value,
    c: Value,
    routine: Option<FuncId>,
) -> BResult<Value> {
    match (fma.route, fma.tiny, routine) {
        (Route::Khr, None, _) => fb.fma(a, b, c),
        (Route::Khr, Some(tiny), Some(r)) => {
            checked(fb, fma.form, [a, b, c], (r, tiny == Route::Wide))
        }
        (Route::Wide | Route::Soft | Route::HalfOdd, _, Some(r)) => fb
            .call(r, &[a, b, c])?
            .ok_or(BuildError::Type(TypeRule::FunctionSignature)),
        _ => Err(BuildError::Type(TypeRule::FunctionSignature)),
    }
}

/// Metal's `fma` of `a`, `b` and `c` (`float` lanes of form `f`, each read as Metal reads it: no
/// denormal) by `OpFmaKHR`, its result `r` answered by `routine` ([`tiny_route`]) where any lane's
/// is at most `2^-126` in magnitude, zero included: where `small`, the routine of those results
/// ([`small`]), given `r`, on every lane; else Metal's `fma`, taken on those lanes.
///
/// Every other lane's `r` is Metal's answer. `OpFmaKHR` is correctly rounded (Vulkan's precision
/// table), and the device may only replace a denormal it reads or answers by a zero (Vulkan's
/// float controls without `DenormPreserve`). Its operands are no denormal, so with `v` the exact
/// `a * b + c`, `r` is the correctly rounded `v`, or a zero. A finite `r` above `2^-126` in
/// magnitude is no flushed zero, so it is the correctly rounded `v` and at least the float after
/// `2^-126`, `2^-126 + 2^-149`. Rounding to nearest gives that only for `|v|` above the midpoint,
/// `2^-126 + 2^-150`, which is above `2^-126`. Metal flushes only an exact value below `2^-126`,
/// so it rounds `v` too and answers `r`. An infinite or NaN `r` is likewise kept: Metal's
/// infinity, and a NaN Metal's NaN answer replaces. The lanes at most `2^-126` are those the
/// device's rounding and its denormal mode decide: `v` just below `2^-126` rounded up to it, a
/// denormal kept or flushed, a zero whose sign the device's flush chose, and an exact zero. The
/// routine answers each of them as Metal does. The test is on `r`'s bits, so no float mode
/// changes it.
fn checked(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    [a, b, c]: [Value; 3],
    (routine, small): (FuncId, bool),
) -> BResult<Value> {
    let r = fb.fma(a, b, c)?;
    let u = u32_form(f);
    let ut = ty(fb, u)?;
    let bits = fb.convert(ConvOp::Bitcast, r, ut)?;
    let magnitude = splat(fb, u, 0x7FFF_FFFF)?;
    let least = splat(fb, u, 0x0080_0000)?;
    let m = bin(fb, BinOp::BitwiseAnd, bits, magnitude)?;
    let at_most = bin(fb, BinOp::ULessThanEqual, m, least)?;
    let any = match f.lanes {
        1 => at_most,
        _ => fb.unary(UnOp::Any, at_most)?,
    };
    let lanes = f.lanes;
    let recompute: ArmFn<'_, ()> = Box::new(move |fb, _| {
        let all = [a, b, c, r];
        let args = if small { &all[..] } else { &all[..3] };
        let metal = fb
            .call(routine, args)?
            .ok_or(BuildError::Type(TypeRule::FunctionSignature))?;
        let v = match (small, lanes) {
            (true, _) | (_, 1) => metal,
            _ => fb.select(at_most, metal, r)?,
        };
        Ok(ArmEnd::FallThrough(vec![v]))
    });
    let keep: ArmFn<'_, ()> = Box::new(move |_, _| Ok(ArmEnd::FallThrough(vec![r])));
    let Merged(merged) = fb.if_(&mut (), any, recompute, Some(keep))?;
    merged
        .first()
        .copied()
        .ok_or(BuildError::Type(TypeRule::FunctionSignature))
}

fn bin(fb: &mut FunctionBuilder<'_>, op: BinOp, a: Value, b: Value) -> BResult<Value> {
    fb.binary(op, a, b)
}

/// Float arithmetic decorated `NoContraction`: rounded on its own, never fused or regrouped.
fn exact_op(fb: &mut FunctionBuilder<'_>, op: BinOp, a: Value, b: Value) -> BResult<Value> {
    fb.binary_with(op, a, b, Contraction::Forbidden)
}

/// `s = x + y` (floats of form `f`) and its rounding error `e`, exactly ([`sum_error`]), for
/// finite `x` and `y` whose sum does not overflow.
fn two_sum(fb: &mut FunctionBuilder<'_>, x: Value, y: Value) -> BResult<(Value, Value)> {
    let s = exact_op(fb, BinOp::FAdd, x, y)?;
    let e = sum_error(fb, s, x, y)?;
    Ok((s, e))
}

/// The rounding error of `s`, the sum of `x` and `y` rounded to nearest, exactly (FastTwoSum on
/// the operands ordered by magnitude: `t = s - big` is exact, and so is `small - t`), for finite
/// `x` and `y` whose sum does not overflow.
fn sum_error(fb: &mut FunctionBuilder<'_>, s: Value, x: Value, y: Value) -> BResult<Value> {
    let ax = fb.glsl(GlslOp::FAbs, &[x])?;
    let ay = fb.glsl(GlslOp::FAbs, &[y])?;
    let x_big = bin(fb, BinOp::FOrdGreaterThanEqual, ax, ay)?;
    let big = fb.select(x_big, x, y)?;
    let small = fb.select(x_big, y, x)?;
    let t = exact_op(fb, BinOp::FSub, s, big)?;
    exact_op(fb, BinOp::FSub, small, t)
}

/// `s` (floats of form `f`, bits of integer form `u`) moved to round to odd given its exact
/// rounding error `e`: where `e` is nonzero and `s`'s lowest significand bit is clear, the
/// neighbour on `e`'s side, whose lowest bit is set (consecutive encodings alternate it). `s`
/// is then never zero (an inexact sum is not), so the step stays within its sign. A NaN `e`
/// (from an infinite or NaN `s`) leaves `s`.
fn to_odd(fb: &mut FunctionBuilder<'_>, f: Form, u: Form, s: Value, e: Value) -> BResult<Value> {
    let ut = ty(fb, u)?;
    let ft = ty(fb, f)?;
    let sb = fb.convert(ConvOp::Bitcast, s, ut)?;
    let (zu, one) = (splat(fb, u, 0)?, splat(fb, u, 1)?);
    let low = bin(fb, BinOp::BitwiseAnd, sb, one)?;
    let even = bin(fb, BinOp::IEqual, low, zu)?;
    let zf = fsplat(fb, f, 0.0)?;
    let inexact = bin(fb, BinOp::FOrdNotEqual, e, zf)?;
    let s_neg = bin(fb, BinOp::FOrdLessThan, s, zf)?;
    let e_neg = bin(fb, BinOp::FOrdLessThan, e, zf)?;
    // The exact value lies beyond `s` (away from zero) when `e` has `s`'s sign.
    let grow = bin(fb, BinOp::LogicalEqual, s_neg, e_neg)?;
    let up = bin(fb, BinOp::IAdd, sb, one)?;
    let down = bin(fb, BinOp::ISub, sb, one)?;
    let moved = fb.select(grow, up, down)?;
    let fix = bin(fb, BinOp::LogicalAnd, inexact, even)?;
    let r = fb.select(fix, moved, sb)?;
    fb.convert(ConvOp::Bitcast, r, ft)
}

/// The exact `a * b + c` of `half` values held in `float` (form `f`), rounded to odd in
/// `float`. The product `p` (two 11-bit significands, between `2^-48` and `2^32` in magnitude)
/// is exact in `float`, so the device's `Fma` is the exact sum rounded once, `s`, whether it
/// fuses or not; `s`'s exact error is FastTwoSum's on `p` and `c` ([`sum_error`]), and
/// [`to_odd`] keeps the rounding direction. An infinite or NaN `s` (only from such an operand)
/// is `Fma`'s, unmoved, decided on its bits. Rounded to `half` to nearest even, it is the `half`
/// fused multiply-add correctly rounded (an extended-range blend's).
///
/// Both choices keep the result off two renderings of SPIRV-Cross, which MoltenVK uses (`f7850ba7`
/// lanes 2, 3, and 7): a `NoContraction` `OpFMul` becomes `fma(l, r, 0)`, which answers `+0` for a
/// `-0` product, so `s`'s signed zeros are `Fma`'s, never `p`'s (a zero `p` leaves the error zero);
/// and `OpFOrdNotEqual` becomes the unordered `!=`, true for the NaN error of an infinite `s`, so
/// [`to_odd`]'s move is not taken there.
pub(crate) fn half_sum(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    a: Value,
    b: Value,
    c: Value,
) -> BResult<Value> {
    let wide = f.with(Held::Float(FloatFmt::F32));
    let s = fb.glsl(GlslOp::Fma, &[a, b, c])?;
    let p = exact_op(fb, BinOp::FMul, a, b)?;
    let e = sum_error(fb, s, p, c)?;
    let odd = to_odd(fb, wide, u32_form(f), s, e)?;
    let finite = all_finite(fb, wide, &[s])?;
    fb.select(finite, odd, s)
}

/// `a * b + c` of finite `float`s (lanes of form `f`) rounded once to `float`, nearest even, in
/// plain `float` arithmetic, whatever the device's `Fma` fuses: Dekker's exact product
/// `a * b = ph + pl` (each factor split at 12 bits by Veltkamp's constant, every step
/// `NoContraction`), then Boldo and Melquiond's emulation of a fused multiply-add through rounding
/// to odd (IEEE TC 57(4), 2008): `(th, tl) = TwoSum(c, ph)`, `v = RO(tl + pl)` ([`half_sum`]'s
/// odd sum, its product `tl · 1` exact), and `RN(th + v)`. Exact where no part underflows: an
/// extended-range blend of a `float4` output, whose operands are at most 2 in magnitude and whose
/// result is rounded to `half` next, far above a part that small. An infinite or NaN result (an
/// infinite or NaN operand's: the operands lie far below `float`'s overflow) is the device's
/// `Fma`'s, decided on its bits, as [`half_sum`]'s is: an infinity of the exact sum's sign, or a
/// NaN, whose encoding the caller chooses.
///
/// A zero result has the sign IEEE 754-2008 (6.3) gives a fused multiply-add's exact zero, put on
/// the bits as an integer ([`zero_sign`]). The emulation's own zero has no defined sign: `th + v`
/// adds a `+0` `v` to a `-0` `th` (`-0 · 1 + -0`), whose sum is `+0` under roundTiesToEven;
/// SPIRV-Cross renders the `NoContraction` product `ph` as `fma(a, b, 0)`, a `+0` for a `-0`
/// product ([`half_sum`]); and a module without `SignedZeroInfNanPreserve` lets the device ignore
/// the sign of a zero in any float instruction (Vulkan's Precision and Operation of SPIR-V
/// Instructions). `OpBitcast` keeps the bits, and the integer instructions are exact.
pub(crate) fn float_fma_exact(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    a: Value,
    b: Value,
    c: Value,
) -> BResult<Value> {
    let (ah, al) = veltkamp(fb, f, a)?;
    let (bh, bl) = veltkamp(fb, f, b)?;
    let ph = exact_op(fb, BinOp::FMul, a, b)?;
    let hh = exact_op(fb, BinOp::FMul, ah, bh)?;
    let t = exact_op(fb, BinOp::FSub, hh, ph)?;
    let hl = exact_op(fb, BinOp::FMul, ah, bl)?;
    let t = exact_op(fb, BinOp::FAdd, t, hl)?;
    let lh = exact_op(fb, BinOp::FMul, al, bh)?;
    let t = exact_op(fb, BinOp::FAdd, t, lh)?;
    let ll = exact_op(fb, BinOp::FMul, al, bl)?;
    let pl = exact_op(fb, BinOp::FAdd, t, ll)?;
    let (th, tl) = two_sum(fb, c, ph)?;
    let one = fsplat(fb, f, 1.0)?;
    let v = half_sum(fb, f.with(Held::Float(FloatFmt::Half)), tl, one, pl)?;
    let r = exact_op(fb, BinOp::FAdd, th, v)?;
    let r = zero_sign(fb, f, [a, b, c], r)?;
    // An infinite or NaN result, only from such an operand, is the device's `Fma`'s, which IEEE 754
    // gives exactly: the emulation's own steps subtract an infinity from itself there, a NaN.
    let s = fb.glsl(GlslOp::Fma, &[a, b, c])?;
    let finite = all_finite(fb, f, &[s])?;
    fb.select(finite, r, s)
}

/// `r`, a multiply-add of `a`, `b`, `c` (`float` lanes of form `f`) rounded once, with a zero
/// result given IEEE 754-2008's sign (6.3): where the exact `a · b + c` is zero, a fused
/// multiply-add's zero is the sign of that sum, the product signed by the rules for
/// multiplication, so under roundTiesToEven `-0` only where the product's sign (`a`'s and `b`'s
/// differing) and `c`'s are both negative, and `+0` otherwise: opposite zeros, or a product that
/// `c` cancels, whose signs differ. The test and the sign are on the bits, as integers
/// (`OpBitcast`, `OpBitwiseAnd`, `OpBitwiseXor`, `OpIEqual`, and `OpSelect` of integers), so the
/// sign is the same whatever the device's float modes.
fn zero_sign(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    [a, b, c]: [Value; 3],
    r: Value,
) -> BResult<Value> {
    let u = u32_form(f);
    let (ut, ft) = (ty(fb, u)?, ty(fb, f)?);
    let [ab, bb, cb, rb] = [a, b, c, r].map(|x| fb.convert(ConvOp::Bitcast, x, ut));
    let (ab, bb, cb, rb) = (ab?, bb?, cb?, rb?);
    let (mag, sign, z) = (
        splat(fb, u, 0x7FFF_FFFF)?,
        splat(fb, u, 0x8000_0000)?,
        splat(fb, u, 0)?,
    );
    let m = bin(fb, BinOp::BitwiseAnd, rb, mag)?;
    let zero = bin(fb, BinOp::IEqual, m, z)?;
    let p = bin(fb, BinOp::BitwiseXor, ab, bb)?;
    let both = bin(fb, BinOp::BitwiseAnd, p, cb)?;
    let signed = bin(fb, BinOp::BitwiseAnd, both, sign)?;
    let w = fb.select(zero, signed, rb)?;
    fb.convert(ConvOp::Bitcast, w, ft)
}

/// `x`'s high 12 significand bits and the rest, `x = hi + lo` exactly (Veltkamp's split by
/// `2^12 + 1`), for finite `x` far below `float`'s overflow.
fn veltkamp(fb: &mut FunctionBuilder<'_>, f: Form, x: Value) -> BResult<(Value, Value)> {
    let k = fsplat(fb, f, 4097.0)?;
    let t = exact_op(fb, BinOp::FMul, k, x)?;
    let u = exact_op(fb, BinOp::FSub, t, x)?;
    let hi = exact_op(fb, BinOp::FSub, t, u)?;
    let lo = exact_op(fb, BinOp::FSub, x, hi)?;
    Ok((hi, lo))
}

/// `x` (`float` lanes of form `f`) as Metal reads an operand: a denormal is a zero of its sign
/// (decided on the exponent field, whatever the device's denormal mode).
fn flush(fb: &mut FunctionBuilder<'_>, f: Form, x: Value) -> BResult<Value> {
    let u = u32_form(f);
    let (ut, ft) = (ty(fb, u)?, ty(fb, f)?);
    let bits = fb.convert(ConvOp::Bitcast, x, ut)?;
    let (exp, sign, z) = (
        splat(fb, u, 0x7F80_0000)?,
        splat(fb, u, 0x8000_0000)?,
        splat(fb, u, 0)?,
    );
    let e = bin(fb, BinOp::BitwiseAnd, bits, exp)?;
    let tiny = bin(fb, BinOp::IEqual, e, z)?;
    let s = bin(fb, BinOp::BitwiseAnd, bits, sign)?;
    let r = fb.select(tiny, s, bits)?;
    fb.convert(ConvOp::Bitcast, r, ft)
}

/// `x` (`float` lanes of form `f`, never a denormal) as Metal answers a result: every NaN
/// `0x7FC00000`.
fn metal_result(fb: &mut FunctionBuilder<'_>, f: Form, x: Value) -> BResult<Value> {
    let u = u32_form(f);
    let (ut, ft) = (ty(fb, u)?, ty(fb, f)?);
    let bits = fb.convert(ConvOp::Bitcast, x, ut)?;
    let (mag, inf, quiet) = (
        splat(fb, u, 0x7FFF_FFFF)?,
        splat(fb, u, 0x7F80_0000)?,
        splat(fb, u, 0x7FC0_0000)?,
    );
    let m = bin(fb, BinOp::BitwiseAnd, bits, mag)?;
    let nan = bin(fb, BinOp::UGreaterThan, m, inf)?;
    let r = fb.select(nan, quiet, bits)?;
    fb.convert(ConvOp::Bitcast, r, ft)
}

/// Whether every lane of `float` values `xs` (form `f`) is finite, from their bits (exact
/// whatever the device does with infinities and NaNs).
fn all_finite(fb: &mut FunctionBuilder<'_>, f: Form, xs: &[Value]) -> BResult<Value> {
    let u = u32_form(f);
    let ut = ty(fb, u)?;
    let exp = splat(fb, u, 0x7F80_0000)?;
    let mut all: Option<Value> = None;
    for x in xs {
        let b = fb.convert(ConvOp::Bitcast, *x, ut)?;
        let e = bin(fb, BinOp::BitwiseAnd, b, exp)?;
        let fin = bin(fb, BinOp::INotEqual, e, exp)?;
        all = Some(match all {
            Some(v) => bin(fb, BinOp::LogicalAnd, v, fin)?,
            None => fin,
        });
    }
    all.ok_or(BuildError::Type(TypeRule::FunctionSignature))
}

/// What a route answers outside its exact core.
struct Outside {
    /// Every operand is finite (the core's domain).
    finite: Value,
    /// The unfused `a * b + c` in `float`, the fused result where the core leaves an exact
    /// zero sum or a zero factor to it (no rounding happens there).
    unfused: Value,
    /// The result for an infinite or NaN operand: the addend itself when both factors are
    /// finite (their exact product, however large, added to an infinity or NaN), else the
    /// unfused result, whose infinite or NaN product IEEE's fused one shares.
    special: Value,
}

fn outside(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    a: Value,
    b: Value,
    c: Value,
) -> BResult<Outside> {
    let p = exact_op(fb, BinOp::FMul, a, b)?;
    let unfused = exact_op(fb, BinOp::FAdd, p, c)?;
    let factors = all_finite(fb, f, &[a, b])?;
    let addend = all_finite(fb, f, &[c])?;
    let special = fb.select(factors, c, unfused)?;
    let finite = bin(fb, BinOp::LogicalAnd, factors, addend)?;
    Ok(Outside {
        finite,
        unfused,
        special,
    })
}

/// The `float` fused multiply-add through `double` ([`Route::Wide`]): `p = a * b` is exact in
/// `double` (48 bits, between `2^-298` and `2^256`), `s = p + c` with its exact error is
/// FastTwoSum's (no `double` denormal or overflow is reachable), [`to_odd`] in `u64` bits
/// makes it the round-to-odd sum, and its conversion to `float` is then the correctly rounded
/// result, overflow included. The round-to-odd sum is below `2^-126` in magnitude exactly where
/// the exact one is (it rounds no value onto `2^-126`, whose significand is even), and there the
/// result is a zero of its sign, as Metal flushes it before rounding. An exact zero sum takes the
/// unfused `float` result, which is the fused one there: it is of two exact zeros (signed as IEEE
/// adds them) or of a product that is a `float` cancelling `c` (`+0`), and needs no `double`
/// signed zero. Infinite and NaN operands take [`Outside::special`].
fn wide(fb: &mut FunctionBuilder<'_>, f: Form, a: Value, b: Value, c: Value) -> BResult<Value> {
    let d = f.with(Held::Float(FloatFmt::F64));
    let w = f.with(Held::Int {
        bits: 64,
        container: 64,
    });
    let (td, tf) = (ty(fb, d)?, ty(fb, f)?);
    let (a, b, c) = (flush(fb, f, a)?, flush(fb, f, b)?, flush(fb, f, c)?);
    let out = outside(fb, f, a, b, c)?;
    let x = fb.convert(ConvOp::FConvert, a, td)?;
    let y = fb.convert(ConvOp::FConvert, b, td)?;
    let z = fb.convert(ConvOp::FConvert, c, td)?;
    let p = exact_op(fb, BinOp::FMul, x, y)?;
    let (s, e) = two_sum(fb, p, z)?;
    let odd = to_odd(fb, d, w, s, e)?;
    let r = fb.convert(ConvOp::FConvert, odd, tf)?;
    let magnitude = fb.glsl(GlslOp::FAbs, &[odd])?;
    let normal = fsplat(fb, d, f64::from(f32::MIN_POSITIVE))?;
    let tiny = bin(fb, BinOp::FOrdLessThan, magnitude, normal)?;
    let u = u32_form(f);
    let ut = ty(fb, u)?;
    let bits = fb.convert(ConvOp::Bitcast, r, ut)?;
    let sign = splat(fb, u, 0x8000_0000)?;
    let signed_zero = bin(fb, BinOp::BitwiseAnd, bits, sign)?;
    let bits = fb.select(tiny, signed_zero, bits)?;
    let r = fb.convert(ConvOp::Bitcast, bits, tf)?;
    let zd = fsplat(fb, d, 0.0)?;
    let zero = bin(fb, BinOp::FOrdEqual, s, zd)?;
    let r = fb.select(zero, out.unfused, r)?;
    let r = fb.select(out.finite, r, out.special)?;
    metal_result(fb, f, r)
}

/// Metal's `fma` of `a`, `b` and `c` (`float` lanes of form `f`) from `r`, their `OpFmaKHR`, on
/// each lane where `r` is at most `2^-126` in magnitude, through `double`; `r` on every other lane
/// ([`checked`]'s rare path, [`Route::Khr`]'s routine).
///
/// On such a lane every operand is finite, and with each read as Metal reads it (a denormal as a
/// zero of its sign) `r` is the device's rounding of the exact `v = a · b + c`: correctly rounded,
/// then kept, flushed to a zero of either sign, or flushed before rounding (Vulkan's float
/// controls), its zero's sign the device's to drop without `SignedZeroInfNanPreserve`. Metal
/// answers a zero where `v` is below `2^-126` in magnitude: of `v`'s sign where `v` is nonzero,
/// and where it is zero the sign IEEE 754-2008 (6.3) gives a fused multiply-add's exact zero
/// (`-0` only where the product's sign and `c`'s are both negative, [`zero_sign`]). Elsewhere it
/// answers the correctly rounded `v`, which there is `r`, `±2^-126` (`|v|` at least `2^-126`
/// rounds to no less, and `r` is no more). So only `v`'s sign and whether it lies below `2^-126`
/// are computed: `p = a · b` is exact in `double` (48 bits), `s = p + c` rounds once and Knuth's
/// TwoSum gives its error `e` exactly (no `double` overflow, and every value is a multiple of
/// `2^-298`, far above `double`'s denormals). `s` is zero only where `v` is and has its sign
/// elsewhere; rounding is monotone and `2^-126` a `double`, so `|v| < 2^-126` where
/// `|s| < 2^-126`, never where `|s| > 2^-126`, and where `|s| == 2^-126` exactly where `e` points
/// inward (`e · s < 0`, a normal `double`). The test of `r`, the zero's sign and the zero are on
/// bits, so no float mode changes the answer.
fn small(fb: &mut FunctionBuilder<'_>, f: Form, [a, b, c]: [Value; 3], r: Value) -> BResult<Value> {
    let d = f.with(Held::Float(FloatFmt::F64));
    let u = u32_form(f);
    let (ut, td, tf) = (ty(fb, u)?, ty(fb, d)?, ty(fb, f)?);
    let rb = fb.convert(ConvOp::Bitcast, r, ut)?;
    let magnitude = splat(fb, u, 0x7FFF_FFFF)?;
    let least = splat(fb, u, 0x0080_0000)?;
    let m = bin(fb, BinOp::BitwiseAnd, rb, magnitude)?;
    let at_most = bin(fb, BinOp::ULessThanEqual, m, least)?;
    // The operands as Metal reads them, in `double`: a denormal as a zero, its sign on its bits.
    let exponent = splat(fb, u, 0x7F80_0000)?;
    let zu = splat(fb, u, 0)?;
    let zf = fsplat(fb, f, 0.0)?;
    let mut read = |x: Value| -> BResult<(Value, Value)> {
        let bits = fb.convert(ConvOp::Bitcast, x, ut)?;
        let e = bin(fb, BinOp::BitwiseAnd, bits, exponent)?;
        let denormal = bin(fb, BinOp::IEqual, e, zu)?;
        let x = fb.select(denormal, zf, x)?;
        Ok((bits, fb.convert(ConvOp::FConvert, x, td)?))
    };
    let ((ab, x), (bb, y), (cb, z)) = (read(a)?, read(b)?, read(c)?);
    let p = exact_op(fb, BinOp::FMul, x, y)?;
    let s = exact_op(fb, BinOp::FAdd, p, z)?;
    // Knuth's TwoSum: `e = (p - (s - z)) + (z - (s - (s - z)))`, exactly `v - s`.
    let sp = exact_op(fb, BinOp::FSub, s, z)?;
    let sz = exact_op(fb, BinOp::FSub, s, sp)?;
    let dp = exact_op(fb, BinOp::FSub, p, sp)?;
    let dz = exact_op(fb, BinOp::FSub, z, sz)?;
    let e = exact_op(fb, BinOp::FAdd, dp, dz)?;
    let normal = fsplat(fb, d, f64::from(f32::MIN_POSITIVE))?;
    let zd = fsplat(fb, d, 0.0)?;
    let size = fb.glsl(GlslOp::FAbs, &[s])?;
    let below = bin(fb, BinOp::FOrdLessThan, size, normal)?;
    let at = bin(fb, BinOp::FOrdEqual, size, normal)?;
    let es = exact_op(fb, BinOp::FMul, e, s)?;
    let inward = bin(fb, BinOp::FOrdLessThan, es, zd)?;
    let edge = bin(fb, BinOp::LogicalAnd, at, inward)?;
    let under = bin(fb, BinOp::LogicalOr, below, edge)?;
    let tiny = bin(fb, BinOp::LogicalAnd, at_most, under)?;
    // The zero's sign: an exact zero sum's IEEE sign, else `v`'s.
    let sign = splat(fb, u, 0x8000_0000)?;
    let product = bin(fb, BinOp::BitwiseXor, ab, bb)?;
    let both = bin(fb, BinOp::BitwiseAnd, product, cb)?;
    let exact = bin(fb, BinOp::BitwiseAnd, both, sign)?;
    let negative = bin(fb, BinOp::FOrdLessThan, s, zd)?;
    let signed = fb.select(negative, sign, zu)?;
    let cancelled = bin(fb, BinOp::FOrdEqual, s, zd)?;
    let zero = fb.select(cancelled, exact, signed)?;
    let out = fb.select(tiny, zero, rb)?;
    fb.convert(ConvOp::Bitcast, out, tf)
}

// ---- The integer routine ----

/// 64-bit shifts and tests on (high, low) `u32` word pairs of form `u`, every 32-bit shift in
/// range for an amount in `[0, 63]`.
struct Words {
    u: Form,
}

impl Words {
    fn k(&self, fb: &mut FunctionBuilder<'_>, v: u32) -> BResult<Value> {
        splat(fb, self.u, u64::from(v))
    }

    /// The amount's low five bits and whether it is at least 32.
    fn split(&self, fb: &mut FunctionBuilder<'_>, n: Value) -> BResult<(Value, Value)> {
        let (k31, k32) = (self.k(fb, 31)?, self.k(fb, 32)?);
        let nl = bin(fb, BinOp::BitwiseAnd, n, k31)?;
        let big = bin(fb, BinOp::UGreaterThanEqual, n, k32)?;
        Ok((nl, big))
    }

    /// `(hi, lo) << n` for `n` in `[0, 63]`.
    fn shl(
        &self,
        fb: &mut FunctionBuilder<'_>,
        (hi, lo): (Value, Value),
        n: Value,
    ) -> BResult<(Value, Value)> {
        let (nl, big) = self.split(fb, n)?;
        let (one, k31, z) = (self.k(fb, 1)?, self.k(fb, 31)?, self.k(fb, 0)?);
        let t_lo = bin(fb, BinOp::ShiftLeftLogical, lo, nl)?;
        let t_hi = bin(fb, BinOp::ShiftLeftLogical, hi, nl)?;
        // The bits of `lo` that cross into `hi`: `lo >> (32 - nl)`, as two in-range shifts.
        let back = bin(fb, BinOp::ISub, k31, nl)?;
        let cross = bin(fb, BinOp::ShiftRightLogical, lo, one)?;
        let cross = bin(fb, BinOp::ShiftRightLogical, cross, back)?;
        let t_hi = bin(fb, BinOp::BitwiseOr, t_hi, cross)?;
        let hi = fb.select(big, t_lo, t_hi)?;
        let lo = fb.select(big, z, t_lo)?;
        Ok((hi, lo))
    }

    /// `(hi, lo) >> n` for `n` in `[0, 63]`.
    fn shr(
        &self,
        fb: &mut FunctionBuilder<'_>,
        (hi, lo): (Value, Value),
        n: Value,
    ) -> BResult<(Value, Value)> {
        let (nl, big) = self.split(fb, n)?;
        let (one, k31, z) = (self.k(fb, 1)?, self.k(fb, 31)?, self.k(fb, 0)?);
        let t_hi = bin(fb, BinOp::ShiftRightLogical, hi, nl)?;
        let t_lo = bin(fb, BinOp::ShiftRightLogical, lo, nl)?;
        let back = bin(fb, BinOp::ISub, k31, nl)?;
        let cross = bin(fb, BinOp::ShiftLeftLogical, hi, one)?;
        let cross = bin(fb, BinOp::ShiftLeftLogical, cross, back)?;
        let t_lo = bin(fb, BinOp::BitwiseOr, t_lo, cross)?;
        let lo = fb.select(big, t_hi, t_lo)?;
        let hi = fb.select(big, z, t_hi)?;
        Ok((hi, lo))
    }

    /// Whether any bit of `(hi, lo)` below bit `n` (in `[0, 63]`) is set.
    fn sticky(
        &self,
        fb: &mut FunctionBuilder<'_>,
        (hi, lo): (Value, Value),
        n: Value,
    ) -> BResult<Value> {
        let (nl, big) = self.split(fb, n)?;
        let (one, z) = (self.k(fb, 1)?, self.k(fb, 0)?);
        let unit = bin(fb, BinOp::ShiftLeftLogical, one, nl)?;
        let below = bin(fb, BinOp::ISub, unit, one)?;
        let lo_part = bin(fb, BinOp::BitwiseAnd, lo, below)?;
        let lo_set = bin(fb, BinOp::INotEqual, lo_part, z)?;
        let hi_part = bin(fb, BinOp::BitwiseAnd, hi, below)?;
        let hi_set = bin(fb, BinOp::INotEqual, hi_part, z)?;
        let lo_any = bin(fb, BinOp::INotEqual, lo, z)?;
        let big_set = bin(fb, BinOp::LogicalOr, lo_any, hi_set)?;
        fb.select(big, big_set, lo_set)
    }

    /// The index of the highest set bit of `(hi, lo)` (all ones for zero, as `FindUMsb`).
    fn msb(&self, fb: &mut FunctionBuilder<'_>, (hi, lo): (Value, Value)) -> BResult<Value> {
        let (z, k32) = (self.k(fb, 0)?, self.k(fb, 32)?);
        let hi_set = bin(fb, BinOp::INotEqual, hi, z)?;
        let hm = fb.glsl(GlslOp::FindUMsb, &[hi])?;
        let hm = bin(fb, BinOp::IAdd, hm, k32)?;
        let lm = fb.glsl(GlslOp::FindUMsb, &[lo])?;
        fb.select(hi_set, hm, lm)
    }
}

/// The `float` fused multiply-add in 32-bit integers ([`Route::Soft`]); exponents are held
/// biased by 512 so every compare is unsigned, except the rounding position, a signed count.
///
/// For finite operands with a nonzero product: the significands `ma`, `mb` (hidden bit set for
/// a normal, exponent field 1 for a denormal) multiply exactly into 48 bits (`OpUMulExtended`),
/// normalized to bit 61 of a 64-bit pair; the addend's significand is placed at bit 61 too.
/// The larger magnitude is `big`, the other is shifted right to `big`'s exponent with the bits
/// shifted out kept as a sticky bit (below the 64 kept, where it cannot reach a rounding
/// decision), both are doubled so the sticky bit sits below the smaller one's last bit, and one
/// 64-bit sum or difference (of operands below `2^63`, so no carry is lost) holds the exact
/// result up to that sticky bit. It rounds to nearest even at the bit that keeps 24 significant
/// bits, into the exponent on a carry and to infinity past the largest finite `float`. An exact
/// cancellation is `+0`; a result whose exact value is below `2^-126` (its leading bit's, the
/// sticky bit keeping it below where the exact one is) is a zero of the sum's sign, as Metal
/// flushes it before rounding. A zero factor takes the unfused result, which is the
/// fused one there (`±0 + c`, no rounding); infinite and NaN operands take
/// [`Outside::special`].
fn soft(fb: &mut FunctionBuilder<'_>, f: Form, a: Value, b: Value, c: Value) -> BResult<Value> {
    use BinOp as B;
    let u = u32_form(f);
    let w = Words { u };
    let (ut, ft) = (ty(fb, u)?, ty(fb, f)?);
    let (a, b, c) = (flush(fb, f, a)?, flush(fb, f, b)?, flush(fb, f, c)?);
    let out = outside(fb, f, a, b, c)?;
    let k = |fb: &mut FunctionBuilder<'_>, v: u32| splat(fb, u, u64::from(v));
    let (z, one, k23, k31) = (k(fb, 0)?, k(fb, 1)?, k(fb, 23)?, k(fb, 31)?);
    let (fmask, emask, hidden, mag) = (
        k(fb, 0x7F_FFFF)?,
        k(fb, 0xFF)?,
        k(fb, 0x80_0000)?,
        k(fb, 0x7FFF_FFFF)?,
    );
    // Sign, biased exponent field, significand (hidden bit set for a normal), and the
    // exponent of its last bit plus 150 (`max(e, 1)`).
    let unpack = |fb: &mut FunctionBuilder<'_>, x: Value| -> BResult<[Value; 4]> {
        let bits = fb.convert(ConvOp::Bitcast, x, ut)?;
        let s = bin(fb, B::ShiftRightLogical, bits, k31)?;
        let e = bin(fb, B::ShiftRightLogical, bits, k23)?;
        let e = bin(fb, B::BitwiseAnd, e, emask)?;
        let m = bin(fb, B::BitwiseAnd, bits, fmask)?;
        let normal = bin(fb, B::INotEqual, e, z)?;
        let h = fb.select(normal, hidden, z)?;
        let m = bin(fb, B::BitwiseOr, m, h)?;
        let e1 = fb.glsl(GlslOp::UMax, &[e, one])?;
        Ok([bits, s, m, e1])
    };
    let [ab, sa, ma, ea] = unpack(fb, a)?;
    let [bb, sb, mb, eb] = unpack(fb, b)?;
    let [cb, sc, mc, ec] = unpack(fb, c)?;
    let zero_mag = |fb: &mut FunctionBuilder<'_>, bits: Value| -> BResult<Value> {
        let m = bin(fb, B::BitwiseAnd, bits, mag)?;
        bin(fb, B::IEqual, m, z)
    };
    let (za, zb, zc) = (zero_mag(fb, ab)?, zero_mag(fb, bb)?, zero_mag(fb, cb)?);
    let sp = bin(fb, B::BitwiseXor, sa, sb)?;

    // The product, its leading bit moved to bit 61; its exponent `pe` (of bit 0, + 512).
    let prod = bin(fb, B::UMulExtended, ma, mb)?;
    let plo = fb.composite_extract(prod, &[0])?;
    let phi = fb.composite_extract(prod, &[1])?;
    let k61 = k(fb, 61)?;
    let mp = w.msb(fb, (phi, plo))?;
    let l = bin(fb, B::ISub, k61, mp)?;
    let k63 = k(fb, 63)?;
    let l = fb.glsl(GlslOp::UMin, &[l, k63])?;
    let (phi, plo) = w.shl(fb, (phi, plo), l)?;
    // pe = ea + eb - 300 - l + 512.
    let pe = bin(fb, B::IAdd, ea, eb)?;
    let k212 = k(fb, 212)?;
    let pe = bin(fb, B::IAdd, pe, k212)?;
    let pe = bin(fb, B::ISub, pe, l)?;
    // The addend at bit 61 (a 24-bit significand: its pair's low word is zero); ce = ec - 150
    // - lc + 512. A zero addend shifts by a harmless 30 and is not used.
    let mcm = fb.glsl(GlslOp::FindUMsb, &[mc])?;
    let lc = bin(fb, B::ISub, k61, mcm)?;
    let k32 = k(fb, 32)?;
    let lc32 = bin(fb, B::ISub, lc, k32)?;
    let k30 = k(fb, 30)?;
    let lc32 = fb.glsl(GlslOp::UMin, &[lc32, k30])?;
    let chi = bin(fb, B::ShiftLeftLogical, mc, lc32)?;
    let k362 = k(fb, 362)?;
    let ce = bin(fb, B::IAdd, ec, k362)?;
    let ce = bin(fb, B::ISub, ce, lc)?;

    // The larger magnitude (the product when the addend is zero).
    let pe_gt = bin(fb, B::UGreaterThan, pe, ce)?;
    let pe_eq = bin(fb, B::IEqual, pe, ce)?;
    let hi_ge = bin(fb, B::UGreaterThanEqual, phi, chi)?;
    let tie = bin(fb, B::LogicalAnd, pe_eq, hi_ge)?;
    let p_big = bin(fb, B::LogicalOr, pe_gt, tie)?;
    let p_big = bin(fb, B::LogicalOr, p_big, zc)?;
    let bhi = fb.select(p_big, phi, chi)?;
    let blo = fb.select(p_big, plo, z)?;
    let ebig = fb.select(p_big, pe, ce)?;
    let sbig = fb.select(p_big, sp, sc)?;
    let xhi = fb.select(p_big, chi, phi)?;
    let xlo = fb.select(p_big, z, plo)?;
    let es = fb.select(p_big, ce, pe)?;
    let d = bin(fb, B::ISub, ebig, es)?;
    let d = fb.glsl(GlslOp::UMin, &[d, k63])?;
    let (shi, slo) = w.shr(fb, (xhi, xlo), d)?;
    let st = w.sticky(fb, (xhi, xlo), d)?;
    let shi = fb.select(zc, z, shi)?;
    let slo = fb.select(zc, z, slo)?;
    let nzc = fb.unary(UnOp::LogicalNot, zc)?;
    let st = bin(fb, B::LogicalAnd, st, nzc)?;

    // Double both and jam the sticky bit below the smaller one.
    let (bhi, blo) = w.shl(fb, (bhi, blo), one)?;
    let (shi, slo) = w.shl(fb, (shi, slo), one)?;
    let stb = fb.select(st, one, z)?;
    let slo = bin(fb, B::BitwiseOr, slo, stb)?;

    // One 64-bit sum or difference (`big >= small`).
    let lo_add = bin(fb, B::IAdd, blo, slo)?;
    let carry = bin(fb, B::ULessThan, lo_add, blo)?;
    let carry = fb.select(carry, one, z)?;
    let hi_add = bin(fb, B::IAdd, bhi, shi)?;
    let hi_add = bin(fb, B::IAdd, hi_add, carry)?;
    let lo_sub = bin(fb, B::ISub, blo, slo)?;
    let borrow = bin(fb, B::ULessThan, blo, slo)?;
    let borrow = fb.select(borrow, one, z)?;
    let hi_sub = bin(fb, B::ISub, bhi, shi)?;
    let hi_sub = bin(fb, B::ISub, hi_sub, borrow)?;
    let sub = bin(fb, B::INotEqual, sp, sc)?;
    let rlo = fb.select(sub, lo_sub, lo_add)?;
    let rhi = fb.select(sub, hi_sub, hi_add)?;
    let er = bin(fb, B::ISub, ebig, one)?;
    let r_any = bin(fb, B::BitwiseOr, rlo, rhi)?;
    let cancelled = bin(fb, B::IEqual, r_any, z)?;
    let m = w.msb(fb, (rhi, rlo))?;
    // The exponent of the leading bit (+ 512); below 2^-126 the result is a signed zero, so a
    // result kept is normal.
    let emsb = bin(fb, B::IAdd, m, er)?;
    let k386 = k(fb, 512 - 126)?;
    let tiny = bin(fb, B::ULessThan, emsb, k386)?;

    // Bits to drop: k = m - 23, a signed count in [-23, 40].
    let kk = bin(fb, B::ISub, m, k23)?;
    let pos = bin(fb, B::SGreaterThan, kk, z)?;
    // k <= 0: the sum is below 2^24 and moves left by -k.
    let nk = bin(fb, B::ISub, z, kk)?;
    let nk = fb.glsl(GlslOp::SMax, &[nk, z])?;
    let nk = fb.glsl(GlslOp::UMin, &[nk, k31])?;
    let q_left = bin(fb, B::ShiftLeftLogical, rlo, nk)?;
    // k > 0: shifted right by k, rounded to nearest even on the dropped bits.
    let kp = fb.glsl(GlslOp::SMax, &[kk, one])?;
    let kp = fb.glsl(GlslOp::UMin, &[kp, k63])?;
    let (_, q_right) = w.shr(fb, (rhi, rlo), kp)?;
    let kh = bin(fb, B::ISub, kp, one)?;
    let (_, half_word) = w.shr(fb, (rhi, rlo), kh)?;
    let half_bit = bin(fb, B::BitwiseAnd, half_word, one)?;
    let half = bin(fb, B::INotEqual, half_bit, z)?;
    let rest = w.sticky(fb, (rhi, rlo), kh)?;
    let q_low = bin(fb, B::BitwiseAnd, q_right, one)?;
    let q_odd = bin(fb, B::INotEqual, q_low, z)?;
    let above = bin(fb, B::LogicalOr, rest, q_odd)?;
    let up = bin(fb, B::LogicalAnd, half, above)?;
    let inc = fb.select(up, one, z)?;
    let q_right = bin(fb, B::IAdd, q_right, inc)?;
    let q = fb.select(pos, q_right, q_left)?;

    // Pack: the leading bit lands on the exponent field's lowest bit, so the field is emsb + 126
    // (+ the carry of a rounding); a tiny result's wrapped field is selected away below.
    let base = bin(fb, B::ISub, emsb, k386)?;
    let base = bin(fb, B::ShiftLeftLogical, base, k23)?;
    let bits = bin(fb, B::IAdd, base, q)?;
    let inf = k(fb, 0x7F80_0000)?;
    let over = bin(fb, B::UGreaterThanEqual, bits, inf)?;
    let bits = fb.select(over, inf, bits)?;
    let bits = fb.select(tiny, z, bits)?;
    let sign = bin(fb, B::ShiftLeftLogical, sbig, k31)?;
    let bits = bin(fb, B::BitwiseOr, bits, sign)?;
    let bits = fb.select(cancelled, z, bits)?;
    let r = fb.convert(ConvOp::Bitcast, bits, ft)?;
    let zero_factor = bin(fb, B::LogicalOr, za, zb)?;
    let r = fb.select(zero_factor, out.unfused, r)?;
    let r = fb.select(out.finite, r, out.special)?;
    metal_result(fb, f, r)
}
