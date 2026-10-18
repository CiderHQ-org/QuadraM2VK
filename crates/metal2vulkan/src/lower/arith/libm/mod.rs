//! The translation's own `float` functions: routines that answer within Apple's stated bound on
//! every device, for the functions whose `GLSL.std.450` instruction a device may answer outside it.
//!
//! Vulkan's precision table bounds `Atan`, `Asin`, and `Acos` at 4096 ulp and `Sin` and `Cos` at
//! `2^-11` absolute, about what Metal's fast-math table allows, while Metal's precise functions
//! are bounded at 4 to 6 ulp (MSL Table 8.1) and Mesa's devices answer `atan` 3,528 ulp off
//! (on lavapipe, RADV, and ANV). Where the device is Metal's own (`FloatFunctions::Metal`: MoltenVK
//! on an Apple GPU), its instruction is Metal's function in the math mode Metal compiles the module
//! with ([`module_mode`]), so a call in that mode is left to it, and a call in the other mode (a
//! `precise::` call in a fast-math module, or a `fast::` one in a module that keeps NaNs) takes the
//! routine. Elsewhere every such call takes the routine. Metal's `sinpi`, `cospi`, and `tanpi` are
//! no device's instruction, so every precise call of them takes their routines, and every fast one
//! but where the device's functions are Metal's.
//!
//! A routine is one module-scope function of one `float` per [`Fun`] (as the wide divisions and
//! fused multiply-adds are), which a call of any lane count calls once per lane: GPU code runs a
//! lane per invocation anyway, and a scalar routine is defined once whatever the vector widths. Its
//! arithmetic is the device's correctly rounded `float` addition, subtraction, and multiplication,
//! each `NoContraction` so that no device fuses or regroups it, with comparisons, selects, and bit
//! operations, and its divisions and roots call the module's routines ([`Needs`], [`Helpers`]):
//! Metal's precise division (`divide.rs`) and Metal's inverse root ([`Fun::FastRsqrt`], from the
//! correctly rounded one). So its answer does not depend on the device. Before, a routine took the
//! device's `OpFDiv` and `Sqrt` (Vulkan allows 2.5 ulp and 3 ulp) and corrected their error where
//! it was found to reach the result; `tan`'s answers still differed among RADV, lavapipe and
//! NVIDIA. No intermediate is an `f32` denormal where the result depends on it, so a
//! device that flushes them answers the same. A routine reads a denormal operand as Metal does
//! ([`denormal`]), and answers Metal's values at its special operands where they are measured,
//! Metal's NaN at a domain error among them (`E::metal_nan`); a NaN operand's result is the
//! device's arithmetic's (`PrecisionNote::DeviceNanEncoding`).

pub(crate) mod atan;
pub(crate) mod exp;
mod fmod;
pub(crate) mod hyper;
pub(crate) mod log;
mod pair;
mod pi;
pub(crate) mod quarter;
pub(crate) mod rcp;
mod root;
pub(crate) mod rsqrt;
pub(crate) mod sincos;
pub(crate) mod tan;
pub(crate) mod trig;

use std::collections::BTreeMap;

use crate::lower::arith::legal::{FloatFmt, Form, Held};
use crate::lower::arith::math::FloatFn;
use crate::lower::arith::narrow::u32_form;
use crate::lower::arith::ops::{splat, ty};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::{BResult, BuildError, TypeRule};
use crate::spirv::ids::{FuncId, GlobalId, Value};
use crate::spirv::ir::{
    BinOp, ConstBits, Constant, Contraction, ConvOp, GlslOp, StorageClass, Type, UnOp,
};

#[cfg(doc)]
use crate::lower::arith::denormal;

/// The quiet NaN Metal answers at the routines' functions' domain errors ([`E::metal_nan`]).
pub(crate) const METAL_NAN: u32 = 0x7FC0_0000;

/// A math mode: Metal's precise functions (MSL Table 8.1) or its fast ones (Table 8.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Mode {
    Precise,
    Fast,
}

/// The mode Metal compiles the `float` functions of an entry in on a device whose functions are
/// Metal's (MoltenVK: `MVKDevice::getMTLCompileOptions` under its default on-demand fast math):
/// precise where the entry declares a `SignedZeroInfNanPreserve` execution mode of any width
/// (`declared`, the widths [`super::ArithPlan::modes`] holds), fast otherwise.
pub fn module_mode(declared: &std::collections::BTreeSet<u8>) -> Mode {
    if declared.is_empty() {
        Mode::Fast
    } else {
        Mode::Precise
    }
}

/// A function a routine computes on `float` operands.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Fun {
    /// Metal's `atan` in a mode: within 5 ulp in both (Tables 8.1 and 8.2).
    Atan(Mode),
    /// Metal's `asin` in a mode: within 4 ulp precise, 5 fast.
    Asin(Mode),
    /// Metal's `acos` in a mode: within 4 ulp precise, 5 fast.
    Acos(Mode),
    /// The generic part of Metal's precise `atan2(y, x)` (within 6 ulp): finite nonzero normal
    /// operands, whose special values `math::atan2_special` answers around it.
    Atan2,
    /// Metal's precise `sin` (within 4 ulp): its own form, `t S(t)` of its reduction's folded
    /// quarter turns and its hardware's `sinc` through a model.
    Sin,
    /// Metal's precise `cos` (within 4 ulp), as [`Fun::Sin`].
    Cos,
    /// Metal's precise `tan` (within 6 ulp).
    Tan,
    /// Metal's fast `sin` below its flush point, which is its precise `sin`: the device's `Sin`
    /// inside `[-π, π]` (Apple's `2^-13` bound, MSL Table 8.2), [`Fun::Sin`]'s routine past it.
    FastSin,
    /// Metal's fast `cos` below its flush point; as [`Fun::FastSin`].
    FastCos,
    /// Metal's `sinpi`, its own form of half turns and its hardware's `sinc` through a model, for a
    /// precise call (`air.sinpi`) or a fast one (`air.fast_sinpi`): Metal's fast function is its
    /// precise one, and only a device whose functions are Metal's tells them apart
    /// ([`Fun::device_answers`]).
    SinPi(Mode),
    /// Metal's `cospi`, as [`Fun::SinPi`].
    CosPi(Mode),
    /// Metal's `tanpi`, as [`Fun::SinPi`], its quotient by Metal's reciprocal ([`Fun::Recip`]).
    TanPi(Mode),
    /// Metal's precise `sqrt`: correctly rounded (MSL Tables 8.1 and 8.3).
    Sqrt,
    /// Metal's precise `rsqrt`: correctly rounded (Tables 8.1 and 8.3).
    Rsqrt,
    /// Metal's precise `fmod`: exact (0 ulp, Tables 8.1 and 8.3).
    Fmod,
    /// Metal's `exp2` (within 4 ulp, Table 8.1), which is also its fast `exp2`: the two answer
    /// alike on every lane of cases 7b283404 and f6524207.
    Exp2,
    /// Metal's precise `exp`: within 4 ulp.
    Exp,
    /// Metal's precise `exp10`: within 4 ulp. No `GLSL.std.450` instruction is it.
    Exp10,
    /// Metal's `log2` (within 4 ulp), also its fast `log2`, as [`Fun::Exp2`], with Metal's error
    /// table (its word on 99.97 % of the positive normal operands): the routine of `air.log2`,
    /// `air.fast_log2`, and the fast functions' formulas.
    Log2,
    /// Metal's precise `log`: within 4 ulp.
    Log,
    /// Metal's precise `log10`: within 4 ulp. No `GLSL.std.450` instruction is it.
    Log10,
    /// The magnitude of Metal's precise `powr(x, y)` and `pow(x, y)` (within 16 ulp): `|x|^y`,
    /// Metal's own form (`log::powr`), whose NaN and zero-exponent cases `math::powr_nan` and
    /// `math::pow_special` answer around it.
    Powr,
    /// Metal's precise `sinh`: within 4 ulp.
    Sinh,
    /// Metal's precise `cosh`: within 4 ulp.
    Cosh,
    /// Metal's precise `tanh`: within 5 ulp.
    Tanh,
    /// Metal's precise `asinh`: within 4 ulp.
    Asinh,
    /// Metal's precise `acosh`: within 4 ulp.
    Acosh,
    /// Metal's precise `atanh`: within 5 ulp.
    Atanh,
    /// Metal's fast `asinh`, MSL Table 8.2's `log(x + sqrt(x x + 1))` in Metal's own form
    /// (`hyper::fast_arc`): its hardware's inverse root, fused into the sum, and its `log2`.
    FastAsinh,
    /// Metal's fast `acosh`, `log(x + sqrt(x x - 1))`, as [`Fun::FastAsinh`].
    FastAcosh,
    /// Metal's reciprocal `1 / y`: its fast division's, the one its precise `sinh`, `cosh`, and
    /// `tanh` compose with, and its texture unit's (`rcp::metal_rcp`). The routine other routines
    /// call for it (`E::recip`), and any lowering per lane ([`Lib::call`]).
    Recip,
    /// Metal's fast `rsqrt`, its hardware's inverse square root (`rsqrt::metal_rsqrt`): the one
    /// Metal's `asin`, `acos`, `asinh`, and `acosh` in both modes take inside, the routine they
    /// call for it (`E::fast_rsqrt`). A standalone `fast::rsqrt` keeps the device's `InverseSqrt`.
    FastRsqrt,
    /// `a b + c` rounded once, the fused multiply-add of Metal's polynomials, emulated: the routine
    /// the others call for a fused step where the profile has no `OpFmaKHR` (`E::fused`), so
    /// their many steps share one emulation.
    Fused,
    /// Metal's `half` `atan` (both modes) before its rounding, of a `half` held as `float`: its own
    /// form, not its `float` function's.
    HalfAtan,
    /// The generic part of Metal's `half` `atan2(y, x)` before its rounding (finite nonzero
    /// operands; `math::atan2_special` answers the rest).
    HalfAtan2,
    /// Metal's `half` `tan` before its rounding, its own form (`trig::half_tan`): its `half`
    /// reduction and polynomial, and Metal's reciprocal in an odd quarter turn.
    HalfTan,
    /// Metal's `half` `sin` before its rounding, its own form (`sincos::half_sin_cos`): a shorter
    /// reduction than its `float` function's, and its hardware's `sinc`.
    HalfSin,
    /// Metal's `half` `cos` before its rounding, as [`Fun::HalfSin`].
    HalfCos,
    /// Metal's `half` `asinh` before its rounding, its own form (`hyper::half_asinh`).
    HalfAsinh,
}

/// What a routine calls or reads besides its own body ([`Fun::needs`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Needs {
    /// Metal's precise division, the precise division routine on one `float` lane (`divide.rs`).
    pub division: bool,
    /// Metal's `exp2` table (`exp::TABLE`).
    pub exp2: bool,
    /// Metal's `log2` table (`log::TABLE`).
    pub log2: bool,
    /// Metal's reciprocal table (`rcp::TABLE`).
    pub rcp: bool,
    /// Metal's reciprocal, [`Fun::Recip`]'s routine.
    pub recip: bool,
    /// Metal's precise inverse square root, [`Fun::Rsqrt`]'s routine.
    pub inverse_root: bool,
    /// Metal's fast inverse square root's table (`rsqrt::TABLE`).
    pub rsqrt: bool,
    /// Metal's fast inverse square root, [`Fun::FastRsqrt`]'s routine.
    pub fast_inverse_root: bool,
    /// The fused multiply-add, [`Fun::Fused`]'s routine where the profile has no `OpFmaKHR`.
    pub fused: bool,
    /// Metal's reduction table from `2^21` up (`quarter::QUARTER_TABLE`).
    pub quarter: bool,
    /// The model of Metal's hardware `sinc` (`sincos::SINC`).
    pub sinc: bool,
    /// Metal's `powr` reduction table (`log::POWR_TABLE`).
    pub powr: bool,
}

impl Needs {
    /// What either needs.
    pub fn or(self, other: Needs) -> Needs {
        Needs {
            division: self.division || other.division,
            exp2: self.exp2 || other.exp2,
            log2: self.log2 || other.log2,
            rcp: self.rcp || other.rcp,
            recip: self.recip || other.recip,
            inverse_root: self.inverse_root || other.inverse_root,
            rsqrt: self.rsqrt || other.rsqrt,
            fast_inverse_root: self.fast_inverse_root || other.fast_inverse_root,
            fused: self.fused || other.fused,
            quarter: self.quarter || other.quarter,
            sinc: self.sinc || other.sinc,
            powr: self.powr || other.powr,
        }
    }
}

/// The module's functions a routine calls and tables it reads for what it [`Needs`], each
/// defined once in the module before the routines that use them, so that no routine's answer
/// depends on the device's division or square root.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Helpers {
    /// The precise division routine on one `float` lane (`divide::define`).
    pub division: Option<FuncId>,
    /// Metal's `exp2` table (`exp::declare`).
    pub exp2: Option<GlobalId>,
    /// Metal's `log2` table (`log::declare`).
    pub log2: Option<GlobalId>,
    /// Metal's reciprocal table (`rcp::declare`).
    pub rcp: Option<GlobalId>,
    /// [`Fun::Recip`]'s routine.
    pub recip: Option<FuncId>,
    /// [`Fun::Rsqrt`]'s routine.
    pub inverse_root: Option<FuncId>,
    /// Metal's fast inverse square root's table (`rsqrt::declare`).
    pub rsqrt: Option<GlobalId>,
    /// [`Fun::FastRsqrt`]'s routine.
    pub fast_inverse_root: Option<FuncId>,
    /// [`Fun::Fused`]'s routine.
    pub fused: Option<FuncId>,
    /// Metal's reduction table from `2^21` up (`quarter::declare`).
    pub quarter: Option<GlobalId>,
    /// The model of Metal's hardware `sinc` (`sincos::declare`).
    pub sinc: Option<GlobalId>,
    /// Metal's `powr` reduction table (`log::declare_powr`).
    pub powr: Option<GlobalId>,
}

impl Fun {
    /// The operand count.
    pub fn arity(self) -> usize {
        match self {
            Fun::Fused => 3,
            Fun::Atan2 | Fun::Fmod | Fun::Powr | Fun::HalfAtan2 => 2,
            Fun::Atan(_)
            | Fun::Asin(_)
            | Fun::Acos(_)
            | Fun::Sin
            | Fun::Cos
            | Fun::Tan
            | Fun::FastSin
            | Fun::FastCos
            | Fun::SinPi(_)
            | Fun::CosPi(_)
            | Fun::TanPi(_)
            | Fun::Sqrt
            | Fun::Rsqrt
            | Fun::Exp2
            | Fun::Exp
            | Fun::Exp10
            | Fun::Log2
            | Fun::Log
            | Fun::Log10
            | Fun::Sinh
            | Fun::Cosh
            | Fun::Tanh
            | Fun::Asinh
            | Fun::Acosh
            | Fun::Atanh
            | Fun::FastAsinh
            | Fun::FastAcosh
            | Fun::Recip
            | Fun::FastRsqrt
            | Fun::HalfAtan
            | Fun::HalfTan
            | Fun::HalfSin
            | Fun::HalfCos
            | Fun::HalfAsinh => 1,
        }
    }

    /// The module's routines `self`'s routine calls and tables it reads ([`Helpers`]): Metal's
    /// precise division, its `exp2` table, which `exp2`, `exp`, `exp10`, `powr`, `sinh`, `cosh`,
    /// and `tanh` read, its `powr` reduction table, which `powr` reads, its `log2` table, which
    /// `log2`, `log`, `log10`, both modes' `asinh` and `acosh`, the `half` `asinh`, and the precise
    /// `atanh` read, its reciprocal ([`Fun::Recip`]'s routine), which `sinh`, `cosh`, `tanh`,
    /// `acosh`, `atanh`, both `atan`s, the precise `atan2`, `tanpi`, and the `half` `tan` call,
    /// the reciprocal's table, which that routine reads, its fast inverse square root
    /// ([`Fun::FastRsqrt`]'s routine), which `asin`, `acos`, both modes' `asinh` and `acosh`, and
    /// the `half` `asinh` call, with its table and its precise inverse square root
    /// ([`Fun::Rsqrt`]'s routine), which that routine reads and calls, the fused step
    /// ([`Fun::Fused`]'s routine), which the routines with Metal's fused steps call where the
    /// profile has no `OpFmaKHR`, Metal's reduction table from `2^21` up, which the precise `sin`,
    /// `cos` and `tan` read (and the fast `sin` and `cos`, past `π`), and the model of its
    /// hardware `sinc`, which `sin`, `cos`, `sinpi`, `cospi`, `tanpi`, and the `half` `sin` and
    /// `cos` read. `sqrt` and `rsqrt` correct the device's own exactly, so they call neither.
    pub fn needs(self) -> Needs {
        let division = match self {
            // Metal's `atan` takes its reciprocal, not a quotient; its `asin` and `acos` take a
            // root of its inverse root, not a square root.
            Fun::Atan(_) | Fun::Asin(_) | Fun::Acos(_) => false,
            Fun::Atan2
            | Fun::Tan
            | Fun::Log2
            | Fun::Log
            | Fun::Log10
            | Fun::Asinh
            | Fun::Acosh
            | Fun::Atanh
            | Fun::FastAsinh
            | Fun::FastAcosh
            | Fun::Recip
            | Fun::HalfAtan
            | Fun::HalfAtan2
            | Fun::HalfAsinh => true,
            Fun::Sinh
            | Fun::Cosh
            | Fun::Tanh
            | Fun::Sin
            | Fun::Cos
            | Fun::FastSin
            | Fun::FastCos
            | Fun::SinPi(_)
            | Fun::CosPi(_)
            | Fun::TanPi(_)
            | Fun::Sqrt
            | Fun::Rsqrt
            | Fun::FastRsqrt
            | Fun::Fused
            | Fun::Fmod
            | Fun::Exp2
            | Fun::Exp
            | Fun::Exp10
            | Fun::Powr
            | Fun::HalfTan
            | Fun::HalfSin
            | Fun::HalfCos => false,
        };
        Needs {
            division,
            exp2: matches!(
                self,
                Fun::Exp2 | Fun::Exp | Fun::Exp10 | Fun::Powr | Fun::Sinh | Fun::Cosh | Fun::Tanh
            ),
            log2: matches!(
                self,
                Fun::Log2
                    | Fun::Log
                    | Fun::Log10
                    | Fun::Asinh
                    | Fun::Acosh
                    | Fun::Atanh
                    | Fun::FastAsinh
                    | Fun::FastAcosh
                    | Fun::HalfAsinh
            ),
            rcp: self == Fun::Recip,
            recip: matches!(
                self,
                Fun::Sinh
                    | Fun::Cosh
                    | Fun::Tanh
                    | Fun::Atan(_)
                    | Fun::Atan2
                    | Fun::Acosh
                    | Fun::Atanh
                    | Fun::TanPi(_)
                    | Fun::HalfTan
            ),
            inverse_root: self == Fun::FastRsqrt,
            rsqrt: self == Fun::FastRsqrt,
            fast_inverse_root: matches!(
                self,
                Fun::Asin(_)
                    | Fun::Acos(_)
                    | Fun::Asinh
                    | Fun::Acosh
                    | Fun::FastAsinh
                    | Fun::FastAcosh
                    | Fun::HalfAsinh
            ),
            fused: matches!(
                self,
                Fun::Atan(_)
                    | Fun::Asin(_)
                    | Fun::Acos(_)
                    | Fun::Atan2
                    | Fun::Exp
                    | Fun::Exp10
                    | Fun::Log
                    | Fun::Log10
                    | Fun::Powr
                    | Fun::Sinh
                    | Fun::Cosh
                    | Fun::Tanh
                    | Fun::Asinh
                    | Fun::Acosh
                    | Fun::Atanh
                    | Fun::FastAsinh
                    | Fun::FastAcosh
                    | Fun::Tan
                    | Fun::HalfTan
                    | Fun::HalfSin
                    | Fun::HalfCos
                    | Fun::HalfAsinh
                    | Fun::Sin
                    | Fun::Cos
                    | Fun::FastSin
                    | Fun::FastCos
                    | Fun::TanPi(_)
            ),
            quarter: matches!(
                self,
                Fun::Tan | Fun::Sin | Fun::Cos | Fun::FastSin | Fun::FastCos
            ),
            sinc: matches!(
                self,
                Fun::Sin
                    | Fun::Cos
                    | Fun::HalfSin
                    | Fun::HalfCos
                    | Fun::FastSin
                    | Fun::FastCos
                    | Fun::SinPi(_)
                    | Fun::CosPi(_)
                    | Fun::TanPi(_)
            ),
            powr: self == Fun::Powr,
        }
    }

    /// Whether a device whose `float` functions are Metal's answers `self` with its own
    /// instruction in an entry whose functions Metal compiles in mode `module`: the
    /// instruction is Metal's function in that mode, and `atan2` is always the precise one
    /// (SPIRV-Cross emits `precise::atan2` whatever the flags). The precise-only functions (`sin`,
    /// `cos`, `tan`, `sqrt`, `rsqrt`, and `fmod`, whose fast calls are their own functions) are the
    /// device's only in a precise module. Metal's fast `sin` and `cos` are the device's in either:
    /// its fast functions in a fast-math module, and in a precise one its precise functions, which
    /// are the fast ones below the flush point. Metal's `sinpi`, `cospi`, and `tanpi` are none of
    /// the instructions SPIRV-Cross emits (it emits no `sinpi`), so no device answers a precise
    /// call: the device's `Sin` and `Cos` of `float(π) r` matched Metal on 849, 922 and 637 of case
    /// 087758cb's 1,024 lanes, where Metal's own forms through the `sinc`'s model
    /// are its word at 99.5 % to 99.99 % of every measured range. A fast call is that composition,
    /// in either mode: within Apple's fast bound (MSL Table 8.2), the near-equivalent the user's
    /// fast-math sign-off accepts, at a third to a sixth of the routine's instructions per call
    /// (the orchestrator's ruling of 2026-10-07).
    ///
    /// Of the second stage's: `exp2` and `log2` are the device's in either mode (Metal's fast ones
    /// are its precise ones), and so is `tanh` (SPIRV-Cross emits `precise::tanh` whatever the
    /// flags); `powr` and the other hyperbolic functions only in a precise module. No
    /// `GLSL.std.450` instruction reaches Metal's `exp10` and `log10`, but Metal's own compositions
    /// of its `exp2` and `log2` do, in either mode (`math::metal_exp`, `math::metal_log`): its
    /// precise `exp10` is `fma(E, d ln 2, E)` of `E = exp2(RN(x c))` and the product's error `d`,
    /// `c = log2(10)` (every lane of case 7b283404 on the M3 and the M4), and its `log10` is
    /// `log2(x) RN(log10(2))` with its own polynomial next to 1 (every `float` of our probe's
    /// binades). Its precise `exp` and `log` are the device's own in a
    /// precise module and the same compositions, with `log2(e)` and `RN(ln 2)` and `log`'s
    /// polynomial, in a fast-math one. Its reciprocal is the device's `1 / y` in a fast-math
    /// module, where Metal compiles the division with its fast reciprocal, and not in a precise
    /// one, whose quotient is correctly rounded.
    pub fn device_answers(self, module: Mode) -> bool {
        match self {
            Fun::FastSin | Fun::FastCos => true,
            Fun::SinPi(m) | Fun::CosPi(m) | Fun::TanPi(m) => m == Mode::Fast,
            Fun::Atan(m) | Fun::Asin(m) | Fun::Acos(m) => m == module,
            Fun::Sin
            | Fun::Cos
            | Fun::Tan
            | Fun::Sqrt
            | Fun::Rsqrt
            | Fun::Fmod
            | Fun::Powr
            | Fun::Sinh
            | Fun::Cosh
            | Fun::Asinh
            | Fun::Acosh
            | Fun::Atanh => module == Mode::Precise,
            // Metal's fast `rsqrt` is the device's `InverseSqrt` in a fast-math module, and its
            // fast `asinh` and `acosh` the device's `Asinh` and `Acosh`.
            Fun::Recip | Fun::FastRsqrt | Fun::FastAsinh | Fun::FastAcosh => module == Mode::Fast,
            // The device's instruction on a `half` held as `float` is its `float` function; the
            // fused step is no float function.
            Fun::HalfAtan
            | Fun::HalfAtan2
            | Fun::HalfTan
            | Fun::HalfSin
            | Fun::HalfCos
            | Fun::HalfAsinh
            | Fun::Fused => false,
            Fun::Atan2
            | Fun::Exp2
            | Fun::Log2
            | Fun::Tanh
            | Fun::Exp
            | Fun::Exp10
            | Fun::Log
            | Fun::Log10 => true,
        }
    }

    /// Whether native device maths (`FloatFunctions::Native`) takes the device's instruction for
    /// the routine's function: every function but precise `fmod`, whose device form
    /// (`OpFRem`, `x - y · trunc(x / y)` at the device's precision) Vulkan leaves unbounded
    /// once the quotient passes `2^24`, where Metal's is exact, and the fused step, which only
    /// the routines call. The rest are within Vulkan's precision table
    /// (`PrecisionNote::NativeTranscendentals`).
    pub fn native(self) -> bool {
        !matches!(self, Fun::Fmod | Fun::Fused)
    }

    /// Whether the device answers the routine's function, where it does
    /// ([`Fun::device_answers`]), by a near-equivalent of Metal's function rather than by Metal's
    /// function itself: Metal's fast `sinpi`, `cospi`, and `tanpi`, which the device answers by
    /// its `Sin` and `Cos` of `float(π) r`, within Apple's fast bound but off Metal's own forms
    /// (case f6524207 differs at 175, 102, and 264 of its 1, 024 lanes through MoltenVK). Every
    /// other function the device answers is Metal's own, word for word
    /// (`DriverDeviation::MetalFloatFunctions`).
    pub fn device_near_equivalent(self) -> bool {
        matches!(
            self,
            Fun::SinPi(Mode::Fast) | Fun::CosPi(Mode::Fast) | Fun::TanPi(Mode::Fast)
        )
    }

    /// Whether Metal's answer to the routine's function is the one right value at every operand,
    /// so that the routine's answer, and the device's where it answers the function
    /// ([`Fun::device_answers`]: Metal's own precise function), is Metal's on every device in
    /// either math mode. Metal documents its precise `sqrt` and `rsqrt` correctly rounded and its
    /// `fmod` exact (MSL Table 8.1 for `float`, Table 8.3 for `half`), and its `half` functions
    /// answer so at every `half` operand: these routines, rounded to `half`, are Metal's word at
    /// all 65,536. The routines compute that value, the correctly rounded root and inverse root and
    /// the exact remainder. Case 7b283404's lanes, which match on every venue, are a check of it,
    /// not its proof. Every other routine is a model of a function Apple bounds by ulps (its
    /// reciprocal and fast inverse root among them), or the fused step the models call, which is no
    /// call's function.
    pub fn correctly_rounded(self) -> bool {
        match self {
            Fun::Sqrt | Fun::Rsqrt | Fun::Fmod => true,
            Fun::Atan(_)
            | Fun::Asin(_)
            | Fun::Acos(_)
            | Fun::Atan2
            | Fun::Sin
            | Fun::Cos
            | Fun::Tan
            | Fun::FastSin
            | Fun::FastCos
            | Fun::SinPi(_)
            | Fun::CosPi(_)
            | Fun::TanPi(_)
            | Fun::Exp2
            | Fun::Exp
            | Fun::Exp10
            | Fun::Log2
            | Fun::Log
            | Fun::Log10
            | Fun::Powr
            | Fun::Sinh
            | Fun::Cosh
            | Fun::Tanh
            | Fun::Asinh
            | Fun::Acosh
            | Fun::Atanh
            | Fun::FastAsinh
            | Fun::FastAcosh
            | Fun::Recip
            | Fun::FastRsqrt
            | Fun::Fused
            | Fun::HalfAtan
            | Fun::HalfAtan2
            | Fun::HalfTan
            | Fun::HalfSin
            | Fun::HalfCos
            | Fun::HalfAsinh => false,
        }
    }

    /// The float function whose operand table ([`denormal::operand`]) the routine reads its
    /// operands by; `None` for the reciprocal and the fast inverse square root, which answer a
    /// denormal operand themselves (as a zero).
    pub fn float_fn(self) -> Option<FloatFn> {
        Some(match self {
            Fun::Atan(Mode::Precise) => FloatFn::Atan,
            Fun::Atan(Mode::Fast) => FloatFn::FastAtan,
            Fun::Asin(Mode::Precise) => FloatFn::Asin,
            Fun::Asin(Mode::Fast) => FloatFn::FastAsin,
            Fun::Acos(Mode::Precise) => FloatFn::Acos,
            Fun::Acos(Mode::Fast) => FloatFn::FastAcos,
            Fun::Atan2 => FloatFn::Atan2,
            Fun::Sin => FloatFn::Sin,
            Fun::Cos => FloatFn::Cos,
            Fun::Tan => FloatFn::Tan,
            Fun::FastSin => FloatFn::FastSin,
            Fun::FastCos => FloatFn::FastCos,
            Fun::SinPi(Mode::Precise) => FloatFn::SinPi,
            Fun::CosPi(Mode::Precise) => FloatFn::CosPi,
            Fun::TanPi(Mode::Precise) => FloatFn::TanPi,
            Fun::SinPi(Mode::Fast) => FloatFn::FastSinPi,
            Fun::CosPi(Mode::Fast) => FloatFn::FastCosPi,
            Fun::TanPi(Mode::Fast) => FloatFn::FastTanPi,
            Fun::Sqrt => FloatFn::Sqrt,
            Fun::Rsqrt => FloatFn::Rsqrt,
            Fun::Fmod => FloatFn::Fmod,
            Fun::Exp2 => FloatFn::Exp2,
            Fun::Exp => FloatFn::Exp,
            Fun::Exp10 => FloatFn::Exp10,
            Fun::Log2 => FloatFn::Log2,
            Fun::Log => FloatFn::Log,
            Fun::Log10 => FloatFn::Log10,
            Fun::Powr => FloatFn::Powr,
            Fun::Sinh => FloatFn::Sinh,
            Fun::Cosh => FloatFn::Cosh,
            Fun::Tanh => FloatFn::Tanh,
            Fun::Asinh => FloatFn::Asinh,
            Fun::Acosh => FloatFn::Acosh,
            Fun::Atanh => FloatFn::Atanh,
            Fun::FastAsinh => FloatFn::FastAsinh,
            Fun::FastAcosh => FloatFn::FastAcosh,
            // A `half` held as `float` is never an `f32` denormal.
            Fun::Recip
            | Fun::FastRsqrt
            | Fun::Fused
            | Fun::HalfAtan
            | Fun::HalfAtan2
            | Fun::HalfTan
            | Fun::HalfSin
            | Fun::HalfCos
            | Fun::HalfAsinh => return None,
        })
    }

    /// An upper estimate of the routine's SPIR-V words, with its constants and types, without
    /// the routines it calls ([`Fun::needs`]), which a module holds once and charges on their own.
    pub fn words(self) -> u32 {
        match self {
            // The polynomial's fused steps (each one `OpFmaKHR`, or a call of the fused step's
            // routine without it), the reduction's arms, and the selects (Metal's reciprocal and
            // inverse root are their own routines): about 270, 240, 330, and 350 words; precise
            // `asin`'s five polynomials about 660, precise `acos`'s three about 410.
            Fun::Atan(_) => 300,
            Fun::Asin(Mode::Precise) => 700,
            Fun::Acos(Mode::Precise) => 450,
            Fun::Asin(Mode::Fast) | Fun::Acos(Mode::Fast) => 400,
            // Precise `atan`'s steps after the quotient (the precise division is its own
            // routine).
            Fun::Atan2 => 350,
            // Metal's reduction in fused steps with its table (`quarter::TABLE_WORDS`), the fold,
            // and the `sinc`'s model in integers with its table (`sincos::TABLE_WORDS`).
            Fun::Sin | Fun::Cos => 600 + quarter::TABLE_WORDS + sincos::TABLE_WORDS,
            // Metal's reduction in fused steps, its two polynomials, and the selects (the
            // quotient is the precise division's routine): about 380 words, with its table
            // (`quarter::TABLE_WORDS`).
            Fun::Tan => 500 + quarter::TABLE_WORDS,
            Fun::FastSin | Fun::FastCos => 700 + quarter::TABLE_WORDS + sincos::TABLE_WORDS,
            // The half turns, the selects, and the `sinc`'s model in integers (twice for `tanpi`,
            // with its fused step and the call of Metal's reciprocal) with its table
            // (`sincos::TABLE_WORDS`).
            Fun::SinPi(_) | Fun::CosPi(_) => 560 + sincos::TABLE_WORDS,
            Fun::TanPi(_) => 750 + sincos::TABLE_WORDS,
            Fun::Sqrt => 700,
            Fun::Rsqrt => 900,
            Fun::Fmod => 600,
            // Metal's `exp2` and its table (`exp::TABLE_WORDS`), and its compositions.
            Fun::Exp2 => 500 + exp::TABLE_WORDS,
            Fun::Exp | Fun::Exp10 => 1300 + exp::TABLE_WORDS,
            // Metal's `log2` and its table (`log::TABLE_WORDS`), and its products.
            Fun::Log2 => 1900 + log::TABLE_WORDS,
            Fun::Log | Fun::Log10 => 2100 + log::TABLE_WORDS,
            // Metal's logarithm pair through its table (`log::POWR_TABLE_WORDS`), the product,
            // its `exp2` with that table (`exp::TABLE_WORDS`), the growth, and the selects.
            Fun::Powr => 1900 + exp::TABLE_WORDS + log::POWR_TABLE_WORDS,
            // Their exponentials with `exp2`'s table; Metal's reciprocal is its own routine.
            Fun::Sinh | Fun::Cosh => 1200 + exp::TABLE_WORDS,
            Fun::Tanh => 1400 + exp::TABLE_WORDS,
            // Metal's polynomials (fused steps: `OpFmaKHR`, or calls of the fused step's routine)
            // and its forms, its `log2` inline with the table (`log::TABLE_WORDS`); the inverse
            // root and Metal's reciprocal are their own routines.
            Fun::Asinh => 2200 + log::TABLE_WORDS,
            Fun::Acosh | Fun::Atanh => 2100 + log::TABLE_WORDS,
            // The root's operand and its fused sum, the selects, and Metal's `log2` inline with
            // its table (`log::TABLE_WORDS`); the inverse root is its own routine.
            Fun::FastAsinh | Fun::FastAcosh => 2000 + log::TABLE_WORDS,
            // Metal's reciprocal and its table (`rcp::TABLE_WORDS`).
            Fun::Recip => 700 + rcp::TABLE_WORDS,
            // Metal's fast inverse square root and its table (`rsqrt::TABLE_WORDS`); the
            // precise inverse root is its own routine.
            Fun::FastRsqrt => 900 + rsqrt::TABLE_WORDS,
            // The emulated fused step: an exact product's splits, three exact sums, and the odd
            // rounding, about 400 words.
            Fun::Fused => 450,
            // Their polynomials' products and sums and the selects (the precise division is its
            // own routine): about 200 and 270 words.
            Fun::HalfAtan => 300,
            Fun::HalfAtan2 => 400,
            // The reduction's three fused steps, the polynomial's three, and the selects (Metal's
            // reciprocal is its own routine): about 180 words.
            Fun::HalfTan => 200,
            // The reduction's three fused steps, the fold, the selects, and the `sinc`'s model in
            // integers (about 510 words) with its table (`sincos::TABLE_WORDS`).
            Fun::HalfSin | Fun::HalfCos => 540 + sincos::TABLE_WORDS,
            // Two fused steps, the selects, and Metal's `log2` inline with its table
            // (`log::TABLE_WORDS`); the inverse root is its own routine.
            Fun::HalfAsinh => 2000 + log::TABLE_WORDS,
        }
    }

    /// The census name of the strategies that call it: the float functions at the device's
    /// precision class, whose answer it replaces.
    pub fn name(self) -> &'static str {
        "arith.float_fn.device_transcendental"
    }
}

/// The routines a call of float function `op` on form `f` may call: its own and those of the
/// functions it composes. `float` lanes take routines, and so do `half` lanes, which are held
/// as `float` and computed by a `float` function before their rounding: Metal's `half` functions
/// are the same in both math modes (the half probe, cases dc0fee8f and 3ef7c1c5; every `half` of
/// our probe), so `refine_float` refines only `float` calls, and the `float` function of a
/// `half` one is `math::metal_half`'s, whose `tanh` calls Metal's `exp2` and reciprocal
/// (`math::half_tanh`); its `atan`, `atan2`, `tan`, `sin`, `cos`, and `asinh` are Metal's own
/// `half` forms ([`Fun::HalfAtan`], [`Fun::HalfAtan2`], [`Fun::HalfTan`]; [`Fun::HalfSin`],
/// [`Fun::HalfCos`], [`Fun::HalfAsinh`]). Another format keeps the device's instruction.
pub fn uses(op: FloatFn, f: Form) -> &'static [Fun] {
    if !matches!(f.float_fmt(), Some(FloatFmt::F32 | FloatFmt::Half)) {
        return &[];
    }
    if f.float_fmt() == Some(FloatFmt::Half) {
        match op {
            FloatFn::Tanh => return &[Fun::Exp2, Fun::Recip],
            FloatFn::Atan => return &[Fun::HalfAtan],
            FloatFn::Atan2 => return &[Fun::HalfAtan2],
            FloatFn::Tan => return &[Fun::HalfTan],
            FloatFn::Sin => return &[Fun::HalfSin],
            FloatFn::Cos => return &[Fun::HalfCos],
            FloatFn::Asinh => return &[Fun::HalfAsinh],
            _ => {}
        }
    }
    match crate::lower::arith::math::metal_half(op, f) {
        FloatFn::Atan => &[Fun::Atan(Mode::Precise)],
        FloatFn::FastAtan => &[Fun::Atan(Mode::Fast)],
        FloatFn::Asin => &[Fun::Asin(Mode::Precise)],
        FloatFn::FastAsin => &[Fun::Asin(Mode::Fast)],
        FloatFn::Acos => &[Fun::Acos(Mode::Precise)],
        FloatFn::FastAcos => &[Fun::Acos(Mode::Fast)],
        FloatFn::Atan2 => &[Fun::Atan2],
        // `atan(y · rcp(x))`, Metal's fast `atan` of its fast quotient.
        FloatFn::FastAtan2 { .. } => &[Fun::Atan(Mode::Fast), Fun::Recip],
        FloatFn::Sin => &[Fun::Sin],
        FloatFn::Cos => &[Fun::Cos],
        FloatFn::Tan => &[Fun::Tan],
        FloatFn::FastSin => &[Fun::FastSin],
        FloatFn::FastCos => &[Fun::FastCos],
        // Apple's formula `sin(x) · (1 / cos(x))` of Metal's fast functions.
        FloatFn::FastTan { .. } => &[Fun::FastSin, Fun::FastCos],
        FloatFn::SinPi => &[Fun::SinPi(Mode::Precise)],
        FloatFn::CosPi => &[Fun::CosPi(Mode::Precise)],
        FloatFn::TanPi => &[Fun::TanPi(Mode::Precise)],
        FloatFn::FastSinPi => &[Fun::SinPi(Mode::Fast)],
        FloatFn::FastCosPi => &[Fun::CosPi(Mode::Fast)],
        FloatFn::FastTanPi => &[Fun::TanPi(Mode::Fast)],
        FloatFn::Sqrt => &[Fun::Sqrt],
        FloatFn::Rsqrt => &[Fun::Rsqrt],
        FloatFn::Fmod => &[Fun::Fmod],
        // The second stage: the precise functions' routines, and the fast ones' formulas of Metal's
        // `exp2` and `log2` (MSL Table 8.2), the latter the fast `log2`'s routine.
        FloatFn::Exp2 => &[Fun::Exp2],
        FloatFn::Exp => &[Fun::Exp],
        FloatFn::Exp10 => &[Fun::Exp10],
        FloatFn::Log2 | FloatFn::FastLog2 => &[Fun::Log2],
        FloatFn::Log => &[Fun::Log],
        FloatFn::Log10 => &[Fun::Log10],
        FloatFn::Pow | FloatFn::Powr => &[Fun::Powr],
        FloatFn::Sinh => &[Fun::Sinh],
        FloatFn::Cosh => &[Fun::Cosh],
        FloatFn::Tanh => &[Fun::Tanh],
        FloatFn::Asinh => &[Fun::Asinh],
        FloatFn::Acosh => &[Fun::Acosh],
        FloatFn::Atanh => &[Fun::Atanh],
        FloatFn::FastExp
        | FloatFn::FastExp10
        | FloatFn::FastSinh
        | FloatFn::FastCosh
        | FloatFn::FastTanh { .. } => &[Fun::Exp2],
        FloatFn::FastLog | FloatFn::FastLog10 | FloatFn::FastAtanh { .. } => &[Fun::Log2],
        FloatFn::FastAsinh => &[Fun::FastAsinh],
        FloatFn::FastAcosh => &[Fun::FastAcosh],
        FloatFn::FastPow | FloatFn::FastPowr => &[Fun::Exp2, Fun::Log2],
        _ => &[],
    }
}

/// An upper estimate of the SPIR-V words per use of a routine of `arity` operands on `lanes`
/// lanes: each lane's operand extractions and call, and the result's construction.
pub fn use_words(lanes: u8, arity: usize) -> u32 {
    let arity = u32::try_from(arity).unwrap_or(u32::MAX);
    let lanes = u32::from(lanes);
    lanes
        .saturating_mul(6u32.saturating_add(arity.saturating_mul(6)))
        .saturating_add(8)
}

/// Which routines an entry's calls take, as emission reads them: the mode of the device's own
/// functions where they are Metal's, and the routines the module defines.
#[derive(Clone, Copy, Debug)]
pub struct Lib<'a> {
    answers: Answers,
    funcs: &'a BTreeMap<Fun, FuncId>,
    /// Whether the device's `NMin`, `NMax` and `NClamp` are Metal's `fmin`, `fmax` and `clamp`
    /// ([`Lib::with_min_max`]).
    min_max: bool,
    /// Whether `mix` and `fast::sqrt` are the device's `FMix` and `Sqrt`
    /// ([`Lib::with_native_order`]).
    native_order: bool,
}

/// Which functions the device's instructions answer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Answers {
    /// Metal's, in the entry's mode ([`module_mode`]).
    Metal(Mode),
    /// The device's own: every [`Fun`] takes its routine.
    Device,
    /// Native device maths (`FloatFunctions::Native`): every function the device's instruction,
    /// or Metal's composition of the device's instructions, whatever its mode
    /// ([`Fun::native`]).
    Native,
    /// Every function, whatever its mode: the instruction itself, for unit tests of the
    /// sequences around it.
    #[cfg(test)]
    Instructions,
}

/// The routines of no module, for [`Lib::instructions`].
#[cfg(test)]
static NO_ROUTINES: BTreeMap<Fun, FuncId> = BTreeMap::new();

impl<'a> Lib<'a> {
    /// The routines of a module whose device functions are Metal's in mode `metal` (`None`:
    /// the device's own), defined as `funcs`.
    pub fn new(metal: Option<Mode>, funcs: &'a BTreeMap<Fun, FuncId>) -> Lib<'a> {
        let answers = match metal {
            Some(m) => Answers::Metal(m),
            None => Answers::Device,
        };
        Lib {
            answers,
            funcs,
            min_max: false,
            native_order: false,
        }
    }

    /// This library, where the device's `GLSL.std.450` `NMin`, `NMax` and `NClamp` are Metal's
    /// own `fmin`, `fmax` and `clamp` if `device` (`ArithPlan::device_min_max`): Metal's minimum
    /// and maximum family on `float` and `half` then takes those instructions
    /// (`math::metal_min_max` otherwise).
    pub fn with_min_max(self, device: bool) -> Lib<'a> {
        Lib {
            min_max: device,
            ..self
        }
    }

    /// Whether the device's `NMin`, `NMax` and `NClamp` are Metal's ([`Lib::with_min_max`]).
    pub fn device_min_max(&self) -> bool {
        self.min_max
    }

    /// A library that leaves every function to its instruction (a unit test's view of the
    /// sequences around one).
    #[cfg(test)]
    pub fn instructions() -> Lib<'static> {
        Lib {
            answers: Answers::Instructions,
            funcs: &NO_ROUTINES,
            min_max: false,
            native_order: false,
        }
    }

    /// The library of native device maths (`FloatFunctions::Native`): the device's instruction
    /// for every function [`Fun::native`] gives it, `funcs`' routines for the rest.
    pub fn native(funcs: &'a BTreeMap<Fun, FuncId>) -> Lib<'a> {
        Lib {
            answers: Answers::Native,
            funcs,
            min_max: false,
            native_order: false,
        }
    }

    /// This library, where Metal's `mix` and `fast::sqrt` round in the device's order if
    /// `native` (`ArithPlan::native_order`): a `float` or `half` `mix` is the device's `FMix` and
    /// `fast::sqrt` its `Sqrt`.
    pub fn with_native_order(self, native: bool) -> Lib<'a> {
        Lib {
            native_order: native,
            ..self
        }
    }

    /// Whether `mix` and `fast::sqrt` round in the device's order ([`Lib::with_native_order`]).
    pub fn orders_natively(&self) -> bool {
        self.native_order
    }

    /// Whether the device's instructions are Metal's functions in mode `m` (a unit test's
    /// [`Lib::instructions`] are, in either).
    pub fn metal_in(&self, m: Mode) -> bool {
        match self.answers {
            Answers::Metal(x) => x == m,
            Answers::Device => false,
            // Native device maths takes the device's instruction wherever Metal's in either mode
            // would.
            Answers::Native => true,
            #[cfg(test)]
            Answers::Instructions => true,
        }
    }

    /// Whether the device's instruction answers `fun` (no routine is called).
    pub fn device(&self, fun: Fun) -> bool {
        match self.answers {
            Answers::Metal(m) => fun.device_answers(m),
            Answers::Device => false,
            Answers::Native => fun.native(),
            #[cfg(test)]
            Answers::Instructions => true,
        }
    }

    /// `fun` of `args` (`float` lanes of form `f`): one call of its routine per lane.
    pub fn call(
        &self,
        fb: &mut FunctionBuilder<'_>,
        fun: Fun,
        f: Form,
        args: &[Value],
    ) -> BResult<Value> {
        let func = *self
            .funcs
            .get(&fun)
            .ok_or(BuildError::Type(TypeRule::FunctionSignature))?;
        if args.len() != fun.arity() {
            return Err(BuildError::Type(TypeRule::FunctionSignature));
        }
        per_lane(fb, func, f, args)
    }
}

/// `func` (a function of `float` values returning one) of `args`, lanes of form `f`: one call
/// per lane.
fn per_lane(fb: &mut FunctionBuilder<'_>, func: FuncId, f: Form, args: &[Value]) -> BResult<Value> {
    let call = |fb: &mut FunctionBuilder<'_>, xs: &[Value]| -> BResult<Value> {
        fb.call(func, xs)?
            .ok_or(BuildError::Type(TypeRule::FunctionSignature))
    };
    if f.lanes == 1 {
        return call(fb, args);
    }
    let mut lanes = Vec::with_capacity(usize::from(f.lanes));
    for l in 0..u32::from(f.lanes) {
        let xs = args
            .iter()
            .map(|a| fb.composite_extract(*a, &[l]))
            .collect::<BResult<Vec<_>>>()?;
        lanes.push(call(fb, &xs)?);
    }
    let t = ty(fb, f)?;
    fb.composite_construct(t, &lanes)
}

/// Define the routine of `fun`: a function of [`Fun::arity`] `float` values returning one, which
/// calls the module's `helpers` for what it needs ([`Fun::needs`]; a missing one is an error).
/// `fma`: the profile's `float` fused multiply-add is `OpFmaKHR`, which the trigonometric and the
/// second stage's routines take for a product's exact error (`E::exact_error`).
pub fn define(mb: &mut ModuleBuilder, fun: Fun, fma: bool, helpers: Helpers) -> BResult<FuncId> {
    define_with(mb, fun, false, fma, helpers)
}

/// [`define`] with the helpers `fun` needs defined first in `mb`, the precise division by the
/// path a profile of `fma` takes (a unit test's module of one routine).
#[cfg(test)]
pub fn define_alone(mb: &mut ModuleBuilder, fun: Fun, fma: bool) -> BResult<FuncId> {
    let helpers = define_helpers(mb, fun.needs(), false, fma)?;
    define_with(mb, fun, false, fma, helpers)
}

/// [`define_alone`], with the device's division and square root moved to the edge of what the
/// routines correct (a unit test's model of a device whose quotient and root are not correctly
/// rounded): a normal quotient 2 ulp and a normal root 3 ulp either way, by its bits, in the
/// routine and its helpers.
#[cfg(test)]
pub fn define_rough(mb: &mut ModuleBuilder, fun: Fun, fma: bool) -> BResult<FuncId> {
    let helpers = define_helpers(mb, fun.needs(), true, fma)?;
    define_with(mb, fun, true, fma, helpers)
}

/// Define the helpers `needs` asks for (the reciprocal's own division included), rough or not.
#[cfg(test)]
pub fn define_helpers(
    mb: &mut ModuleBuilder,
    needs: Needs,
    rough: bool,
    fma: bool,
) -> BResult<Helpers> {
    use crate::lower::arith::divide::{self, Path, Routine};
    let mut helpers = Helpers::default();
    if needs.fused && !fma {
        helpers.fused = Some(define_with(mb, Fun::Fused, rough, fma, helpers)?);
    }
    let recip_needs = Fun::Recip.needs();
    if needs.division || (needs.recip && recip_needs.division) {
        let r = Routine {
            lanes: 1,
            path: if fma { Path::Fused } else { Path::Integer },
        };
        let moved = |fb: &mut FunctionBuilder<'_>, a: Value, b: Value| -> BResult<Value> {
            let mut e = E {
                fb,
                rough,
                fma,
                helpers: Helpers::default(),
            };
            e.device_div(a, b)
        };
        helpers.division = Some(if rough {
            divide::define_with(mb, r, &moved)?
        } else {
            divide::define(mb, r)?
        });
    }
    if needs.exp2 {
        helpers.exp2 = Some(exp::declare(mb)?);
    }
    if needs.log2 {
        helpers.log2 = Some(log::declare(mb)?);
    }
    if needs.rcp || (needs.recip && recip_needs.rcp) {
        helpers.rcp = Some(rcp::declare(mb)?);
    }
    if needs.recip {
        helpers.recip = Some(define_with(mb, Fun::Recip, rough, fma, helpers)?);
    }
    let fast_root_needs = Fun::FastRsqrt.needs();
    if needs.inverse_root || (needs.fast_inverse_root && fast_root_needs.inverse_root) {
        helpers.inverse_root = Some(define_with(mb, Fun::Rsqrt, rough, fma, helpers)?);
    }
    if needs.rsqrt || (needs.fast_inverse_root && fast_root_needs.rsqrt) {
        helpers.rsqrt = Some(rsqrt::declare(mb)?);
    }
    if needs.quarter {
        helpers.quarter = Some(quarter::declare(mb)?);
    }
    if needs.sinc {
        helpers.sinc = Some(sincos::declare(mb)?);
    }
    if needs.powr {
        helpers.powr = Some(log::declare_powr(mb)?);
    }
    if needs.fast_inverse_root {
        helpers.fast_inverse_root = Some(define_with(mb, Fun::FastRsqrt, rough, fma, helpers)?);
    }
    Ok(helpers)
}

fn define_with(
    mb: &mut ModuleBuilder,
    fun: Fun,
    rough: bool,
    fma: bool,
    helpers: Helpers,
) -> BResult<FuncId> {
    let t = mb.ty(Type::Float { width: 32 })?;
    let mut fb = mb.routine(t)?;
    let mut args = Vec::with_capacity(fun.arity());
    for _ in 0..fun.arity() {
        args.push(fb.param(t)?);
    }
    let mut e = E {
        fb: &mut fb,
        rough,
        fma,
        helpers,
    };
    let v = match (fun, args.as_slice()) {
        (Fun::Atan(m), [x]) => atan::atan(&mut e, m, *x)?,
        (Fun::Asin(m), [x]) => atan::asin(&mut e, m, *x)?,
        (Fun::Acos(m), [x]) => atan::acos(&mut e, m, *x)?,
        (Fun::Atan2, [y, x]) => atan::atan2(&mut e, *y, *x)?,
        (Fun::Sin, [x]) => sincos::sin_cos(&mut e, trig::Trig::Sin, *x)?,
        (Fun::Cos, [x]) => sincos::sin_cos(&mut e, trig::Trig::Cos, *x)?,
        (Fun::Tan, [x]) => tan::tan(&mut e, *x)?,
        (Fun::FastSin, [x]) => trig::fast(&mut e, trig::Trig::Sin, *x)?,
        (Fun::FastCos, [x]) => trig::fast(&mut e, trig::Trig::Cos, *x)?,
        (Fun::SinPi(_), [x]) => pi::pi(&mut e, fun, pi::Pi::Sin, *x)?,
        (Fun::CosPi(_), [x]) => pi::pi(&mut e, fun, pi::Pi::Cos, *x)?,
        (Fun::TanPi(_), [x]) => pi::pi(&mut e, fun, pi::Pi::Tan, *x)?,
        (Fun::Sqrt, [x]) => root::root(&mut e, root::Root::Sqrt, *x)?,
        (Fun::Rsqrt, [x]) => root::root(&mut e, root::Root::Rsqrt, *x)?,
        (Fun::Fmod, [x, y]) => fmod::fmod(&mut e, *x, *y)?,
        (Fun::Exp2, [x]) => exp::exp2(&mut e, *x)?,
        (Fun::Exp, [x]) => exp::exp_base(&mut e, *x, false)?,
        (Fun::Exp10, [x]) => exp::exp_base(&mut e, *x, true)?,
        (Fun::Log2, [x]) => log::metal_log2(&mut e, *x)?,
        (Fun::Log, [x]) => log::log(&mut e, log::Base::E, *x)?,
        (Fun::Log10, [x]) => log::log(&mut e, log::Base::Ten, *x)?,
        (Fun::Powr, [x, y]) => log::powr(&mut e, *x, *y)?,
        (Fun::Sinh, [x]) => hyper::hyper(&mut e, hyper::Hyper::Sinh, *x)?,
        (Fun::Cosh, [x]) => hyper::hyper(&mut e, hyper::Hyper::Cosh, *x)?,
        (Fun::Tanh, [x]) => hyper::hyper(&mut e, hyper::Hyper::Tanh, *x)?,
        (Fun::Asinh, [x]) => hyper::arc(&mut e, hyper::Arc::Asinh, *x)?,
        (Fun::Acosh, [x]) => hyper::arc(&mut e, hyper::Arc::Acosh, *x)?,
        (Fun::Atanh, [x]) => hyper::arc(&mut e, hyper::Arc::Atanh, *x)?,
        (Fun::FastAsinh, [x]) => hyper::fast_arc(&mut e, hyper::FastArc::Asinh, *x)?,
        (Fun::FastAcosh, [x]) => hyper::fast_arc(&mut e, hyper::FastArc::Acosh, *x)?,
        (Fun::Recip, [y]) => rcp::metal_rcp(&mut e, *y)?,
        (Fun::FastRsqrt, [y]) => rsqrt::metal_rsqrt(&mut e, *y)?,
        (Fun::Fused, [a, b, c]) => e.fused_inline(*a, *b, *c)?,
        (Fun::HalfAtan, [x]) => atan::half_atan(&mut e, *x)?,
        (Fun::HalfAtan2, [y, x]) => atan::half_atan2(&mut e, *y, *x)?,
        (Fun::HalfTan, [x]) => trig::half_tan(&mut e, *x)?,
        (Fun::HalfSin, [x]) => sincos::half_sin_cos(&mut e, trig::Trig::Sin, *x)?,
        (Fun::HalfCos, [x]) => sincos::half_sin_cos(&mut e, trig::Trig::Cos, *x)?,
        (Fun::HalfAsinh, [x]) => hyper::half_asinh(&mut e, *x)?,
        _ => return Err(BuildError::Type(TypeRule::FunctionSignature)),
    };
    fb.ret(Some(v))?;
    fb.finish()
}

/// `v` as `hi + lo`, `hi` the `float` nearest `v`, `lo` the `float` nearest the rest.
pub(crate) fn split(v: f64) -> (f32, f32) {
    let hi = v as f32;
    (hi, (v - f64::from(hi)) as f32)
}

/// The multiple of words every table's array is a whole number of ([`declare_words`]): 16 bytes.
pub(crate) const TABLE_ALIGN: usize = 4;

/// Declare a table of `words` in the module: a `Private` `u32` array initialized to them (Metal's
/// `exp2` and `log2` tables), followed by zeros up to a multiple of [`TABLE_ALIGN`] words.
///
/// The zeros keep every table a whole number of 16-byte rows, so a table declared after another
/// starts where a 16-byte load can reach it. NVIDIA's driver (615.71.09, RTX 5080 laptop) read
/// every word of the `powr` table one index low in case 7b283404, where that table followed the
/// 165 words of `quarter::QUARTER_TABLE` (the 1,221st word of the tables declared before it):
/// as if it reads the four words of a row as one 16-byte load from a 16-byte boundary, and the
/// tables lie end to end. A host replay of `powr` with every table word taken one index low is
/// that device's word at all 675 positive normal lanes; with the table itself, RADV's.
pub(crate) fn declare_words(mb: &mut ModuleBuilder, words: &[u32]) -> BResult<GlobalId> {
    let u32_ = mb.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let padded = words.len().div_ceil(TABLE_ALIGN) * TABLE_ALIGN;
    let len = u64::try_from(padded).map_err(|_| BuildError::Type(TypeRule::FunctionSignature))?;
    let len = mb.constant(Constant::Scalar {
        ty: u32_,
        bits: ConstBits(len),
    })?;
    let arr = mb.ty(Type::Array {
        elem: u32_,
        len,
        stride: None,
    })?;
    let mut parts = Vec::with_capacity(padded);
    let zeros = std::iter::repeat_n(&0, padded - words.len());
    for w in words.iter().chain(zeros) {
        parts.push(mb.constant(Constant::Scalar {
            ty: u32_,
            bits: ConstBits(u64::from(*w)),
        })?);
    }
    let init = mb.constant(Constant::Composite {
        ty: arr,
        parts: parts.into(),
    })?;
    mb.plain_global(StorageClass::Private, arr, Some(init))
}

/// The `float` scalar.
const F32: Form = Form {
    held: Held::Float(FloatFmt::F32),
    lanes: 1,
};

/// A routine body's emission on `float` scalars: constants by their bits, and the correctly
/// rounded arithmetic, each operation `NoContraction`.
pub(crate) struct E<'f, 'm> {
    pub fb: &'f mut FunctionBuilder<'m>,
    /// The device's division and square root moved ([`define_rough`]); `false` in the product.
    rough: bool,
    /// The profile's `float` fused multiply-add is `OpFmaKHR`: the trigonometric and the second
    /// stage's routines take a product's exact error as one `OpFmaKHR` ([`E::exact_error`]).
    fma: bool,
    /// The module's functions [`E::div`] and [`E::sqrt`] call.
    helpers: Helpers,
}

impl E<'_, '_> {
    /// The `float` constant `v`.
    pub fn c(&mut self, v: f32) -> BResult<Value> {
        splat(self.fb, F32, u64::from(v.to_bits()))
    }

    /// Metal's NaN at a domain error of a routine's function, the quiet NaN `0x7fc00000` by its
    /// bits, with `x`'s sign where `signed`: an infinite operand of `sin`, `cos`, and `tan`,
    /// `|x| > 1` of `asin` (`fast::asin`'s NaN has the operand's sign) and `acos`, a negative
    /// normal or infinite operand of `sqrt` and `rsqrt`, and an infinite dividend or a zero or
    /// denormal divisor of `fmod` (Metal's rows of cases 087758cb and cc234732, every such lane).
    /// Selected as a constant, so no device's NaN encoding reaches it; a NaN operand's answer stays
    /// the device's arithmetic's (`PrecisionNote::DeviceNanEncoding`).
    pub fn metal_nan(&mut self, signed: Option<Value>) -> BResult<Value> {
        let nan = self.c(f32::from_bits(METAL_NAN))?;
        match signed {
            Some(x) => self.copysign(nan, x),
            None => Ok(nan),
        }
    }

    /// Metal's `exp2` table ([`Helpers::exp2`]); an error where the module has none.
    pub fn exp2_table(&self) -> BResult<GlobalId> {
        self.helpers
            .exp2
            .ok_or(BuildError::Type(TypeRule::FunctionSignature))
    }

    /// Metal's `log2` table ([`Helpers::log2`]); an error where the module has none.
    pub fn log2_table(&self) -> BResult<GlobalId> {
        self.helpers
            .log2
            .ok_or(BuildError::Type(TypeRule::FunctionSignature))
    }

    /// Metal's reciprocal table ([`Helpers::rcp`]); an error where the module has none.
    pub fn rcp_table(&self) -> BResult<GlobalId> {
        self.helpers
            .rcp
            .ok_or(BuildError::Type(TypeRule::FunctionSignature))
    }

    /// Metal's fast inverse square root's table ([`Helpers::rsqrt`]); an error where the module
    /// has none.
    pub fn rsqrt_table(&self) -> BResult<GlobalId> {
        self.helpers
            .rsqrt
            .ok_or(BuildError::Type(TypeRule::FunctionSignature))
    }

    /// Metal's reduction table from `2^21` up ([`Helpers::quarter`]); an error where the module
    /// has none.
    pub fn quarter_table(&self) -> BResult<GlobalId> {
        self.helpers
            .quarter
            .ok_or(BuildError::Type(TypeRule::FunctionSignature))
    }

    /// The model of Metal's hardware `sinc` ([`Helpers::sinc`]); an error where the module has
    /// none.
    pub fn sinc_table(&self) -> BResult<GlobalId> {
        self.helpers
            .sinc
            .ok_or(BuildError::Type(TypeRule::FunctionSignature))
    }

    /// Metal's `powr` reduction table ([`Helpers::powr`]); an error where the module has none.
    pub fn powr_table(&self) -> BResult<GlobalId> {
        self.helpers
            .powr
            .ok_or(BuildError::Type(TypeRule::FunctionSignature))
    }

    /// The `float` integer at or below `a` (`GLSL.std.450 Floor`, exact).
    pub fn floor(&mut self, a: Value) -> BResult<Value> {
        self.fb.glsl(GlslOp::Floor, &[a])
    }

    /// The `u32` constant `v`.
    pub fn u(&mut self, v: u32) -> BResult<Value> {
        splat(self.fb, u32_form(F32), u64::from(v))
    }

    fn op(&mut self, op: BinOp, a: Value, b: Value) -> BResult<Value> {
        self.fb.binary_with(op, a, b, Contraction::Forbidden)
    }

    pub fn add(&mut self, a: Value, b: Value) -> BResult<Value> {
        self.op(BinOp::FAdd, a, b)
    }

    pub fn sub(&mut self, a: Value, b: Value) -> BResult<Value> {
        self.op(BinOp::FSub, a, b)
    }

    pub fn mul(&mut self, a: Value, b: Value) -> BResult<Value> {
        self.op(BinOp::FMul, a, b)
    }

    /// `a / b`, Metal's precise quotient: a call of the module's precise division routine
    /// (`divide.rs`, correctly rounded on every profile), so that no routine's answer depends on
    /// the device's division (Vulkan allows `OpFDiv` 2.5 ulp, and `tan` answered differently on
    /// RADV and lavapipe). Its operands and quotient are normal wherever a routine's answer depends
    /// on them, where Metal's precise division is the correctly rounded quotient.
    pub fn div(&mut self, a: Value, b: Value) -> BResult<Value> {
        let f = self
            .helpers
            .division
            .ok_or(BuildError::Type(TypeRule::FunctionSignature))?;
        self.fb
            .call(f, &[a, b])?
            .ok_or(BuildError::Type(TypeRule::FunctionSignature))
    }

    /// The device's quotient (within 2.5 ulp for a divisor in `[2^-126, 2^126]`): only for a
    /// routine that corrects it exactly whatever its error (`root.rs`).
    pub fn device_div(&mut self, a: Value, b: Value) -> BResult<Value> {
        let q = self.op(BinOp::FDiv, a, b)?;
        self.moved(q, 2)
    }

    /// Metal's reciprocal of `a`: a call of the module's [`Fun::Recip`] routine.
    pub fn recip(&mut self, a: Value) -> BResult<Value> {
        let f = self
            .helpers
            .recip
            .ok_or(BuildError::Type(TypeRule::FunctionSignature))?;
        self.fb
            .call(f, &[a])?
            .ok_or(BuildError::Type(TypeRule::FunctionSignature))
    }

    /// Metal's precise inverse square root of `a` (correctly rounded): a call of the module's
    /// [`Fun::Rsqrt`] routine, which Metal's fast one starts from (`rsqrt::metal_rsqrt`).
    pub fn rsqrt(&mut self, a: Value) -> BResult<Value> {
        let f = self
            .helpers
            .inverse_root
            .ok_or(BuildError::Type(TypeRule::FunctionSignature))?;
        self.fb
            .call(f, &[a])?
            .ok_or(BuildError::Type(TypeRule::FunctionSignature))
    }

    /// Metal's fast inverse square root of `a`, its hardware's: a call of the module's
    /// [`Fun::FastRsqrt`] routine, which `asin` and `acos` in both modes and `asinh` and `acosh` in
    /// both compose.
    pub fn fast_rsqrt(&mut self, a: Value) -> BResult<Value> {
        let f = self
            .helpers
            .fast_inverse_root
            .ok_or(BuildError::Type(TypeRule::FunctionSignature))?;
        self.fb
            .call(f, &[a])?
            .ok_or(BuildError::Type(TypeRule::FunctionSignature))
    }

    /// The device's square root: only for a routine that corrects it exactly whatever its error
    /// (`root.rs`).
    pub fn device_sqrt(&mut self, a: Value) -> BResult<Value> {
        let s = self.fb.glsl(GlslOp::Sqrt, &[a])?;
        self.moved(s, 3)
    }

    /// `v`, or where the emission is [`define_rough`]'s, a normal `v` moved `ulps` units of its
    /// last place up or down by one of its own bits.
    fn moved(&mut self, v: Value, ulps: u32) -> BResult<Value> {
        if !self.rough {
            return Ok(v);
        }
        let b = self.bits(v)?;
        let (field, low, inf) = (
            self.u(0x7F80_0000)?,
            self.u(0x0100_0000)?,
            self.u(0x7F00_0000)?,
        );
        let e = self.iop(BinOp::BitwiseAnd, b, field)?;
        let above = self.iop(BinOp::UGreaterThanEqual, e, low)?;
        let below = self.iop(BinOp::ULessThan, e, inf)?;
        let normal = self.iop(BinOp::LogicalAnd, above, below)?;
        let (bit, zero) = (self.u(4)?, self.u(0)?);
        let set = self.iop(BinOp::BitwiseAnd, b, bit)?;
        let up = self.iop(BinOp::INotEqual, set, zero)?;
        let (plus, minus) = (self.u(ulps)?, self.u(ulps.wrapping_neg())?);
        let d = self.sel(up, plus, minus)?;
        let d = self.sel(normal, d, zero)?;
        let r = self.iop(BinOp::IAdd, b, d)?;
        self.float(r)
    }

    pub fn lt(&mut self, a: Value, b: Value) -> BResult<Value> {
        self.fb.binary(BinOp::FOrdLessThan, a, b)
    }

    pub fn le(&mut self, a: Value, b: Value) -> BResult<Value> {
        self.fb.binary(BinOp::FOrdLessThanEqual, a, b)
    }

    pub fn gt(&mut self, a: Value, b: Value) -> BResult<Value> {
        self.fb.binary(BinOp::FOrdGreaterThan, a, b)
    }

    pub fn ge(&mut self, a: Value, b: Value) -> BResult<Value> {
        self.fb.binary(BinOp::FOrdGreaterThanEqual, a, b)
    }

    pub fn sel(&mut self, c: Value, t: Value, f: Value) -> BResult<Value> {
        self.fb.select(c, t, f)
    }

    /// The bits of `float` `a`.
    pub fn bits(&mut self, a: Value) -> BResult<Value> {
        let ut = ty(self.fb, u32_form(F32))?;
        self.fb.convert(ConvOp::Bitcast, a, ut)
    }

    /// The `float` of bits `b`.
    pub fn float(&mut self, b: Value) -> BResult<Value> {
        let ft = ty(self.fb, F32)?;
        self.fb.convert(ConvOp::Bitcast, b, ft)
    }

    pub fn iop(&mut self, op: BinOp, a: Value, b: Value) -> BResult<Value> {
        self.fb.binary(op, a, b)
    }

    /// `|a|`, by its bits (exact on every device, a denormal included).
    pub fn abs(&mut self, a: Value) -> BResult<Value> {
        let b = self.bits(a)?;
        let m = self.u(0x7FFF_FFFF)?;
        let r = self.iop(BinOp::BitwiseAnd, b, m)?;
        self.float(r)
    }

    /// `a`'s magnitude with `s`'s sign bit, by their bits.
    pub fn copysign(&mut self, a: Value, s: Value) -> BResult<Value> {
        let (ab, sb) = (self.bits(a)?, self.bits(s)?);
        let (m, sign) = (self.u(0x7FFF_FFFF)?, self.u(0x8000_0000)?);
        let ab = self.iop(BinOp::BitwiseAnd, ab, m)?;
        let sb = self.iop(BinOp::BitwiseAnd, sb, sign)?;
        let r = self.iop(BinOp::BitwiseOr, ab, sb)?;
        self.float(r)
    }

    /// `P(v)` of the coefficients `c` (`f32` bits, lowest first) by Horner's rule from the highest,
    /// each step one fused multiply-add (Metal's polynomials).
    pub fn fused_poly(&mut self, c: &[u32], v: Value) -> BResult<Value> {
        let (last, rest) = c
            .split_last()
            .ok_or(BuildError::Type(TypeRule::FunctionSignature))?;
        let mut acc = self.c(f32::from_bits(*last))?;
        for k in rest.iter().rev() {
            let k = self.c(f32::from_bits(*k))?;
            acc = self.fused(acc, v, k)?;
        }
        Ok(acc)
    }

    /// The polynomial `c[0] + z (c[1] + z (...))` by Horner's rule.
    pub fn horner(&mut self, c: &[f32], z: Value) -> BResult<Value> {
        let (last, rest) = c
            .split_last()
            .ok_or(BuildError::Type(TypeRule::FunctionSignature))?;
        let mut q = self.c(*last)?;
        for k in rest.iter().rev() {
            let kz = self.mul(z, q)?;
            let kc = self.c(*k)?;
            q = self.add(kc, kz)?;
        }
        Ok(q)
    }

    /// `a`'s two halves of at most 12 significant bits each, `hi + lo = a` exactly (Veltkamp's
    /// split by `2^12 + 1`), for `|a|` below `2^115`.
    fn split(&mut self, a: Value) -> BResult<(Value, Value)> {
        let k = self.c(4097.0)?;
        let c = self.mul(k, a)?;
        let d = self.sub(c, a)?;
        let hi = self.sub(c, d)?;
        let lo = self.sub(a, hi)?;
        Ok((hi, lo))
    }

    /// The rounding error of the product `p = a * b` (`a * b = p + e` exactly, Dekker's
    /// product), where no partial product is an `f32` denormal: `|a|` and `|b|` below `2^115`
    /// and their product above `2^-100` or zero.
    pub fn product_error(&mut self, a: Value, b: Value, p: Value) -> BResult<Value> {
        let (ah, al) = self.split(a)?;
        let (bh, bl) = self.split(b)?;
        let hh = self.mul(ah, bh)?;
        let hl = self.mul(ah, bl)?;
        let lh = self.mul(al, bh)?;
        let ll = self.mul(al, bl)?;
        let e = self.sub(hh, p)?;
        let e = self.add(e, hl)?;
        let e = self.add(e, lh)?;
        self.add(e, ll)
    }

    /// The rounding error of the product `p = a * b` where [`E::product_error`]'s is exact: one
    /// `OpFmaKHR` of `a * b - p` where the profile has it ([`E::fma`]), else Dekker's product.
    /// Both are that exact error: it is a normal `float` (above `2^-124`, or zero) for a product
    /// above `2^-100`, so the fused operation rounds nothing and a device that flushes
    /// denormals keeps it.
    pub fn exact_error(&mut self, a: Value, b: Value, p: Value) -> BResult<Value> {
        if !self.fma {
            return self.product_error(a, b, p);
        }
        let minus_p = self.fb.unary(UnOp::FNegate, p)?;
        self.fb.fma(a, b, minus_p)
    }

    /// `a * b` of `u32` values as `(low word, high word)` (`OpUMulExtended`).
    pub fn umul(&mut self, a: Value, b: Value) -> BResult<(Value, Value)> {
        let p = self.fb.binary(BinOp::UMulExtended, a, b)?;
        Ok((
            self.fb.composite_extract(p, &[0])?,
            self.fb.composite_extract(p, &[1])?,
        ))
    }

    /// `a + b` of `u32` values as `(sum, carry)` (`OpIAddCarry`).
    pub fn uadd(&mut self, a: Value, b: Value) -> BResult<(Value, Value)> {
        let s = self.fb.binary(BinOp::IAddCarry, a, b)?;
        Ok((
            self.fb.composite_extract(s, &[0])?,
            self.fb.composite_extract(s, &[1])?,
        ))
    }

    /// The `float` of the `u32` `a`, exact where `a` has at most 24 significant bits.
    pub fn ufloat(&mut self, a: Value) -> BResult<Value> {
        let ft = ty(self.fb, F32)?;
        self.fb.convert(ConvOp::ConvertUToF, a, ft)
    }

    /// The `u32` of the non-negative `float` `a`, truncated (`OpConvertFToU`).
    pub fn utrunc(&mut self, a: Value) -> BResult<Value> {
        let ut = ty(self.fb, u32_form(F32))?;
        self.fb.convert(ConvOp::ConvertFToU, a, ut)
    }

    /// `a + b` as the exact two-term sum `(s, e)`, for `|a| >= |b|` or `a = 0` (Fast2Sum).
    pub fn fast_two_sum(&mut self, a: Value, b: Value) -> BResult<(Value, Value)> {
        let s = self.add(a, b)?;
        let d = self.sub(s, a)?;
        let l = self.sub(b, d)?;
        Ok((s, l))
    }

    /// The exact remainder `n - q * d` of a quotient `q` close to `n / d` (within a few ulp),
    /// as one `float`: `n - p` is exact by Sterbenz's lemma, and the product's error is
    /// [`E::product_error`]'s.
    pub fn remainder(&mut self, n: Value, q: Value, d: Value) -> BResult<Value> {
        let p = self.mul(q, d)?;
        let e = self.product_error(q, d, p)?;
        let r = self.sub(n, p)?;
        self.sub(r, e)
    }
}
