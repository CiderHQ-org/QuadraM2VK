//! Metal's `Depth16Unorm` codes, stored by the fragment stage (`LoweringInputs::depth16_codes`).
//!
//! Metal stores a depth `d` of `[0, 1]` in a `Depth16Unorm` attachment as the exact product
//! `65535 d` rounded to nearest, a tie up (probes on the M3 Max under macOS 26.7 and 27.0.1; the
//! one tie an `f32` of `[0, 1]` makes is 0.5's 32767.5, stored 32768). Vulkan converts the depth a
//! fragment stores by `convertFloatToUint(f × (2^b - 1), b)`, which "returns one of the two
//! unsigned binary integer values with exactly b bits which are closest to the floating-point
//! value r. Implementations should round to nearest. If r is equal to an integer, then that
//! integer value must be returned." (the Vulkan specification, "Conversion From Floating-Point to
//! Normalized Fixed-Point"). A device measured to store Metal's codes so takes
//! `DriverDeviation::MetalDepth16Stores`, and one measured to store Metal's codes of biased depths
//! too `DriverDeviation::MetalDepth16BiasedStores`, a biased draw on a device of only the first
//! running the variant that stores them here (`VariantKey::depth16_biased`); elsewhere the device's own rounding may store the code
//! next to Metal's near a midpoint (NVIDIA's 615.71.09 stored the code below up to 0.55 of a code
//! past one, lavapipe rounds the `float` product with ties to even). A stage compiled under the
//! input writes in place of each depth the `float` nearest Metal's code `c` over 65535
//! ([`code_depth`]): its product with 65535 lies within 2^-7 of `c`, so a conversion that rounds
//! to nearest, as the specification asks, stores `c`, as it stores every color channel the
//! translation rounds so (`arith::color_write`).
//!
//! - **A returned depth** (`[[depth(...)]]`) is clamped as Metal stores it (`clamp_depth`, to
//!   `[0, 1]` on every profile here) and rounded to its code: Metal biases no depth a function
//!   returns (probe `depthout`, the M3 Max under macOS 27.0.1: 16,020 rows of depths, constants
//!   and slopes, every row but the two positive NaNs, which store nothing, Metal's code of the
//!   clamped depth). Its `DepthGreater` or `DepthLess` mode is dropped, which the rounding can
//!   break.
//! - **A function that returns no depth**, and a pipeline without a fragment function
//!   ([`fragment_module`]), stores at entry the depth Metal stores for the fragment
//!   ([`biased_depth`]): `FragCoord.z` (whose Vulkan bias the consumer records as zero) plus
//!   Metal's bias, `o = s m + u` with `m = max(|dz/dx|, |dz/dy|)`, each operation rounded on its
//!   own, clamped by Metal's clamp factor, added rounding toward zero, clamped to `[0, 1]`, and
//!   rounded to its code (probe `depth16`, the M3 Max under macOS 27.0.1: 705 table rows and 24,000
//!   random rows of constants exact, and 2,877 of 2,880 sloped rows, the other three within 0.002
//!   of a code's midpoint). `u` is Metal's `c ± 1` units of 1/65535 (`metal::state::depth16_bias`)
//!   and `s` and the clamp Metal's factors, from `PushField::DepthBias`. `m` is taken from the
//!   quad's differences of `FragCoord.z` at entry, where the quad runs whole: Metal's slope is the
//!   triangle's own, so a sloped bias can differ near a midpoint (the runtime announces it).
//!
//! A function that declares or takes early fragment tests, or runs its draw's tests itself, is not
//! compiled so (`LoweringInputs::depth16_codes`), nor is the store at entry made in one that reads
//! its position's depth: whether Metal's position takes the draw's bias is unmeasured, and with the
//! Vulkan bias recorded as zero it would not.
//!
//! A stage that writes its depth keeps a device from testing the depth before the stage runs, so
//! every fragment the depth hides runs it. Under a bias whose side is known
//! (`LoweringInputs::depth16_bias_raises`) the store at entry declares it, `DepthGreater` for a
//! bias that raises the depth and `DepthLess` for one that lowers it, which lets the device reject
//! a fragment the depth hides before the stage runs where its test compares that way
//! ([`bias_side_mode`], [`side_bounded`]): on the Iris Xe of Raptor Lake-P, 16 full-screen layers
//! drawn front to back took 1.85 ms so, against 9.42 ms without the declaration and 1.46 ms
//! through the device's own conversion (docs/FAST-PATHS.md, costs of matching Metal).

use crate::lower::arith::color_write::round_scaled_product;
use crate::lower::arith::legal::{FloatFmt, Form, Held};
use crate::lower::arith::ops::fsplat;
use crate::pipeline::interface::PushPlacement;
use crate::pipeline::sym::{PushField, Stage};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::Value as SValue;
use crate::spirv::ir::{
    BinOp, ConstBits, Constant, ConvOp, DerivOp, ExecMode, ExecModel, GlslOp, IoDir, MemAccess,
    Type, TypedModule,
};
use crate::spv::BuiltIn as B;

/// The words [`code_depth`] adds: the exact product's fourteen instructions, their constants and
/// nine `NoContraction` decorations, and the scale.
pub(crate) const CODE_WORDS: u64 = 30 * 6;

/// The words [`biased_depth`] adds beside [`code_depth`]'s: the field's three extracts and
/// conversions, the two derivatives and the slope, the bias and its clamp's comparisons and
/// selection, the sum's error and its step toward zero, the clamp to `[0, 1]`, their constants and
/// decorations.
pub(crate) const BIAS_WORDS: u64 = 40 * 6;

/// The words a function's synthesized depth adds at entry: its two objects' declarations and
/// interface entries, the push field's load, the `FragCoord` load and extract, the
/// [`biased_depth`] and [`code_depth`] arithmetic, and the store.
pub(crate) const STORE_WORDS: u64 = 4 * 6 + BIAS_WORDS + CODE_WORDS;

/// The scalar `f32` form.
const F32: Form = Form {
    held: Held::Float(FloatFmt::F32),
    lanes: 1,
};

/// `depth` (an `f32` of `[0, 1]`) as the `f32` the device stores as Metal's code `c`: `c` is the
/// exact product with 65535 rounded to nearest (`round_scaled_product`, ties to even, which
/// on `[0, 1]` meets only 0.5 and takes 32768, Metal's tie up), and the value is `c` times the
/// `f32` nearest 1/65535, within two `f32` rounding errors of `c / 65535`. A NaN passes as a NaN.
pub(crate) fn code_depth(fb: &mut FunctionBuilder<'_>, depth: SValue) -> BResult<SValue> {
    let scale = fsplat(fb, F32, 65536.0)?;
    let code = round_scaled_product(fb, F32, depth, scale)?;
    let unit = fsplat(fb, F32, f64::from(1.0f32 / 65535.0))?;
    fb.binary(BinOp::FMul, code, unit)
}

/// The depth Metal stores for a fragment at window depth `z` (`FragCoord.z`, unbiased) under the
/// draw's depth bias `bias` (`PushField::DepthBias`'s `uvec4` array), as [`code_depth`] writes it
/// (the module docs): [`offset_depth`] of Metal's slope, the larger of the depth's two screen
/// derivatives' magnitudes. The derivatives read the quad, so the caller emits this where every
/// invocation of it runs.
pub(crate) fn biased_depth(
    fb: &mut FunctionBuilder<'_>,
    z: SValue,
    bias: SValue,
) -> BResult<SValue> {
    let f32_ = fb.module().ty(Type::Float { width: 32 })?;
    let word = |fb: &mut FunctionBuilder<'_>, i: u32| -> BResult<SValue> {
        let bits = fb.composite_extract(bias, &[0, i])?;
        fb.convert(ConvOp::Bitcast, bits, f32_)
    };
    let factors = [word(fb, 0)?, word(fb, 1)?, word(fb, 2)?];
    let dx = fb.derivative(DerivOp::DPdx, z)?;
    let dy = fb.derivative(DerivOp::DPdy, z)?;
    let dx = fb.glsl(GlslOp::FAbs, &[dx])?;
    let dy = fb.glsl(GlslOp::FAbs, &[dy])?;
    let m = fb.glsl(GlslOp::NMax, &[dx, dy])?;
    offset_depth(fb, z, m, factors)
}

/// The depth Metal stores for a fragment at window depth `z` of slope `m` under Metal's bias
/// `[units, slope, clamp]` (`metal::state::depth16_bias`), as [`code_depth`] writes it: the
/// offset `slope m + units`, bounded by a nonzero clamp from its side, added to `z` rounding toward
/// zero, clamped to `[0, 1]` (a NaN to 0). Every addition and product is decorated
/// `NoContraction`, so no device fuses the bias's product into its sum or the sum's error terms.
pub(crate) fn offset_depth(
    fb: &mut FunctionBuilder<'_>,
    z: SValue,
    m: SValue,
    [units, slope, clamp]: [SValue; 3],
) -> BResult<SValue> {
    let f32_ = fb.module().ty(Type::Float { width: 32 })?;
    let u32_ = fb.module().ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let sloped = fb.binary_no_contraction(BinOp::FMul, slope, m)?;
    let offset = fb.binary_no_contraction(BinOp::FAdd, sloped, units)?;
    // A clamp above 0 bounds the bias from above, one below 0 from below, and 0 not at all.
    let zero = fsplat(fb, F32, 0.0)?;
    let up = fb.binary(BinOp::FOrdGreaterThan, clamp, zero)?;
    let past_up = fb.binary(BinOp::FOrdGreaterThan, offset, clamp)?;
    let over = fb.binary(BinOp::LogicalAnd, up, past_up)?;
    let down = fb.binary(BinOp::FOrdLessThan, clamp, zero)?;
    let past_down = fb.binary(BinOp::FOrdLessThan, offset, clamp)?;
    let under = fb.binary(BinOp::LogicalAnd, down, past_down)?;
    let clamped = fb.binary(BinOp::LogicalOr, over, under)?;
    let offset = fb.select(clamped, clamp, offset)?;
    // `z + offset` rounded toward zero: the nearest sum and its exact error (TwoSum), and one step
    // down where a positive sum lies above the exact one. A negative sum stores 0 whichever way it
    // rounds.
    let sum = fb.binary_no_contraction(BinOp::FAdd, z, offset)?;
    let b = fb.binary_no_contraction(BinOp::FSub, sum, z)?;
    let a = fb.binary_no_contraction(BinOp::FSub, sum, b)?;
    let da = fb.binary_no_contraction(BinOp::FSub, z, a)?;
    let db = fb.binary_no_contraction(BinOp::FSub, offset, b)?;
    let error = fb.binary_no_contraction(BinOp::FAdd, da, db)?;
    let positive = fb.binary(BinOp::FOrdGreaterThan, sum, zero)?;
    let above = fb.binary(BinOp::FOrdLessThan, error, zero)?;
    let step = fb.binary(BinOp::LogicalAnd, positive, above)?;
    let bits = fb.convert(ConvOp::Bitcast, sum, u32_)?;
    let one = fb.u32_const(1)?;
    let one = fb.module().const_value(one)?;
    let below = fb.binary(BinOp::ISub, bits, one)?;
    let below = fb.convert(ConvOp::Bitcast, below, f32_)?;
    let toward_zero = fb.select(step, below, sum)?;
    let unit = fsplat(fb, F32, 1.0)?;
    let depth = fb.glsl(GlslOp::NClamp, &[toward_zero, zero, unit])?;
    code_depth(fb, depth)
}

/// The execution mode a stage that stores [`biased_depth`] declares under a bias of known side
/// (`LoweringInputs::depth16_bias_raises`): `DepthGreater` where it raises the depth, `DepthLess`
/// where it lowers it ([`side_bounded`] makes the declaration hold).
pub(crate) fn bias_side_mode(raises: bool) -> ExecMode {
    match raises {
        true => ExecMode::DepthGreater,
        false => ExecMode::DepthLess,
    }
}

/// The words [`side_bounded`] and [`bias_side_mode`] add: the mode's declaration and the bound.
pub(crate) const SIDE_WORDS: u64 = 3 + 6;

/// The depth a stage that declares [`bias_side_mode`] stores in place of [`biased_depth`]'s
/// `stored`: `NMax(stored, z)` where the bias raises the depth, `NMin(stored, z)` where it lowers
/// it, of the fragment's `z` (`FragCoord.z`). `DepthGreater` promises a depth no lower than
/// `FragCoord.z` and `DepthLess` one no higher, and a stage that breaks the promise is undefined
/// in Vulkan, so the bound keeps it by construction; it changes no code Metal stores.
///
/// The runtime gives a draw a side (`BindState::depth16_bias_raises`) only where Metal's units `u`
/// (`metal::state::depth16_bias`) lie above 0, the slope factor `s` is finite and not below 0, the
/// clamp `k` is not above 0 or is at least `u`, and the depth range of every Vulkan viewport of the
/// draw, and its Vulkan depth clamp range where it sets one (`VK_EXT_depth_clamp_control`), lie
/// inside `[0, 1]`; `DepthLess` mirrors each condition. Then, for raising, with
/// `q = fl(1/65535) = 2^-16 (1 + 2^-16)`, just below `1/65535`:
///
/// 1. `u >= q`: `u` is `(c + 1) / 65535` for a constant `c` of 0 or more, rounded to `float`, and
///    rounding keeps order.
/// 2. The offset `o >= q`: the quad's depths are finite, so the slope `m = max(|dz/dx|, |dz/dy|)`
///    is finite and not below 0, and so is `fl(s m)` (or `+inf`); `fl(fl(s m) + u) >= u` (a slope
///    factor of 0 adds 0). A clamp above 0 takes `min(o, k) >= min(u, k) = u`; one of 0 or below
///    bounds nothing from above. An infinite `o` saturates the depth at 1.
/// 3. For a `z` in `[0, 1]`, `stored >= z`: the sum rounded toward zero, `t`, is at least `z`, a
///    `float` not above the exact sum. Where `t >= 1` the depth clamps to 1, whose code 65535 is
///    written as `fl(65535 q) = fl(1 - 2^-32) = 1`. Below 1, `t > z + q - 2^-24`, so
///    `65535 t > 65535 z + 0.996`, Metal's code (nearest) is above `65535 z + 0.496`, and the
///    written `fl(code q)` lies within `2^-25` of `code (1 - 2^-32) / 65535`, above `z + 7.5e-6`.
/// 4. Without a Vulkan depth clamp the device clips each primitive's depth to the view volume, so
///    `z` lies in its Vulkan viewport's range, inside `[0, 1]`, and the bound is `stored`. Under
///    one (Metal's depth clamp mode, or a compensated range) `z` may leave `[0, 1]`: below 0
///    `stored >= 0 > z`, and above 1 the bound writes `z` where Metal stores 1, but the device
///    clamps the written depth to its clamp range, inside `[0, 1]`, so both store that range's top.
///
/// Lowering mirrors it: `u <= -q`, `o <= -q`, a sum at or below 0 stores 0, and below `z`
/// otherwise; a `z` below 0 under a Vulkan depth clamp stores the clamp range's bottom, as Metal's
/// 0 does.
pub(crate) fn side_bounded(
    fb: &mut FunctionBuilder<'_>,
    stored: SValue,
    z: SValue,
    raises: bool,
) -> BResult<SValue> {
    let op = match raises {
        true => GlslOp::NMax,
        false => GlslOp::NMin,
    };
    fb.glsl(op, &[stored, z])
}

/// The push placement of a synthesized fragment stage ([`fragment_module`]): `PushField::DepthBias`
/// alone, at the fragment window's start.
pub(crate) fn fragment_placement() -> PushPlacement {
    crate::lower::push::place(Stage::Fragment, &[PushField::DepthBias].into())
}

/// The fragment stage of a pipeline without a fragment function whose depth attachment is
/// `Depth16Unorm`, compiled where a function would be under `LoweringInputs::depth16_codes`: it
/// writes no color and stores at entry the depth Metal stores for the fragment
/// ([`biased_depth`]), reading the draw's bias from `PushField::DepthBias` at
/// [`fragment_placement`]'s offset. Under a bias of known side (`raises`) it declares the side
/// ([`bias_side_mode`]) and stores [`side_bounded`].
pub(crate) fn fragment_module(raises: Option<bool>) -> BResult<TypedModule> {
    let mut mb = ModuleBuilder::new();
    let f32_ = mb.ty(Type::Float { width: 32 })?;
    let u32_ = mb.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let vec4 = mb.ty(Type::Vector { elem: f32_, len: 4 })?;
    let uvec4 = mb.ty(Type::Vector { elem: u32_, len: 4 })?;
    let len = mb.constant(Constant::Scalar {
        ty: u32_,
        bits: ConstBits(1),
    })?;
    let field = mb.ty(Type::Array {
        elem: uvec4,
        len,
        stride: Some(16),
    })?;
    let void = mb.ty(Type::Void)?;
    let coord = mb.builtin(B::FragCoord, IoDir::Input, vec4)?;
    let depth = mb.builtin(B::FragDepth, IoDir::Output, f32_)?;
    // The one field, at the window's start ([`fragment_placement`]).
    let offset = crate::lower::push::window(Stage::Fragment).0;
    let member = mb.push_member(PushField::DepthBias, field, offset)?.index;
    let push = mb.push_global()?;
    let mut fb = mb.function(void)?;
    let p = fb.module().global_value(coord)?;
    let frag = fb.load(p, MemAccess::default())?;
    let z = fb.composite_extract(frag, &[2])?;
    let base = fb.module().global_value(push)?;
    let index = fb.u32_const(member)?;
    let index = fb.module().const_value(index)?;
    let at = fb.access_chain(base, &[index])?;
    let bias = fb.load(at, MemAccess::default())?;
    let mut stored = biased_depth(&mut fb, z, bias)?;
    if let Some(raises) = raises {
        stored = side_bounded(&mut fb, stored, z, raises)?;
    }
    let out = fb.module().global_value(depth)?;
    fb.store(out, stored, MemAccess::default())?;
    fb.ret(None)?;
    let f = fb.finish()?;
    let mut modes = vec![ExecMode::OriginUpperLeft, ExecMode::DepthReplacing];
    modes.extend(raises.map(bias_side_mode));
    mb.entry(ExecModel::Fragment, f, modes)?;
    mb.finalize()
}
