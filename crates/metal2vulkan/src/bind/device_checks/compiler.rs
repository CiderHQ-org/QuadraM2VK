//! The device's checks of two shader-compiler defects (`DriverWorkaround`'s
//! `NarrowMaskAfterSignedClamp` and `HalfConversionPairFolded`): each kernel holds the
//! instructions translation writes where the profile does not take the workaround, over inputs
//! whose answers the defect changes, and stores every result. A device whose compiler keeps the
//! instructions' meaning leaves [`CheckKernel::expected`]; one with the defect leaves other words
//! (NVIDIA's 615.71.09 on the RTX 3090 has both).
//!
//! - **A narrow mask after a signed clamp** ([`narrow_mask_check`]): two values zero-extended from
//!   `N` bits are sign-extended (`ops::cast`'s `SExt`: a shift left and an arithmetic shift right),
//!   added or subtracted, clamped signed to exactly `N` bits' range (`SClamp`, or `SMin` then
//!   `SMax`) and truncated to `N` bits (`ops::cast`'s `Trunc`: the mask `OpBitwiseAnd` with
//!   `2^N - 1`, the profile's [`MaskForm::And`]), at `N` = 8 and 16, scalar and three lanes: the
//!   result's low `N` bits, zero above. The 3090's compiler drops the mask there and answers the
//!   clamp sign-extended.
//! - **A `half` round trip folded** ([`half_round_trip_check`]): an `f32` narrowed to 16 bits and
//!   widened back, each `OpFConvert` decorated `NoContraction`, in an entry declaring
//!   `RoundingModeRTE`, `DenormPreserve` and `SignedZeroInfNanPreserve` for 16 bits and the modes
//!   the device's independence ties to them (`narrow::controlled_modes_of`), in each math mode the
//!   device has: Metal's `half(float)` by the Vulkan spec but for a NaN's bits, which Vulkan leaves
//!   to the device and Metal makes `0x7E00`'s widening. Every input here but the zeros, the
//!   infinities and the canonical NaN changes through it, the NaNs quiet and signalling with the
//!   payloads a conversion may keep. The 3090's compiler folds the pair into the `f32` it started
//!   from; through the RTX 5080 Laptop GPU's pair a NaN keeps its payload
//!   (`DriverWorkaround::HalfConversionPairFolded`'s docs).

use super::kernel::{finish, MathMode, LOCAL_SIZE};
use super::{metal_half_of_f32, CheckKernel};
use crate::device::Device;
use crate::error::Error;
use crate::lower::arith::legal::{Form, Held};
use crate::lower::arith::narrow::controlled_modes_of;
use crate::lower::arith::ops::{cast, ones, splat, CastKind};
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage};
use crate::spirv::builder::{FunctionBuilder, MaskForm, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, TypeId, Value};
use crate::spirv::ir::{
    natural_layout, BinOp, ConstBits, Constant, ConvOp, ExecMode, ExecModel, GlslOp, IoDir,
    LayoutRule, MemAccess, StorageClass, Type, TypedModule,
};

/// The binding symbol of storage buffer `index`.
fn buffer_slot(index: u32) -> ResourceSlot {
    ResourceSlot {
        stage: Stage::Kernel,
        kind: ResourceKind::Buffer,
        index,
    }
}

/// A `Block { u32 data[]; }` storage buffer (std430) at buffer slot `index`.
fn words(b: &mut ModuleBuilder, index: u32, u32_: TypeId) -> BResult<GlobalId> {
    let rta = b.ty(Type::RuntimeArray {
        elem: u32_,
        stride: Some(4),
    })?;
    let layout = natural_layout(b.types(), LayoutRule::Std430, true, &[rta])?;
    let block = b.ty(Type::Struct {
        members: [rta].into(),
        layout: Some(layout),
    })?;
    b.resource(buffer_slot(index), StorageClass::StorageBuffer, block)
}

/// A `u32` constant as a value.
fn word(f: &mut FunctionBuilder<'_>, u32_: TypeId, bits: u32) -> BResult<Value> {
    let c = f.module().constant(Constant::Scalar {
        ty: u32_,
        bits: ConstBits(u64::from(bits)),
    })?;
    f.module().const_value(c)
}

/// A pointer to word `i` of storage buffer `g`.
fn at(f: &mut FunctionBuilder<'_>, g: GlobalId, u32_: TypeId, i: Value) -> BResult<Value> {
    let zero = word(f, u32_, 0)?;
    let base = f.module().global_value(g)?;
    f.access_chain(base, &[zero, i])
}

/// `gl_GlobalInvocationID.x`.
fn invocation(f: &mut FunctionBuilder<'_>, u32_: TypeId) -> BResult<Value> {
    let uvec3 = f.module().ty(Type::Vector { elem: u32_, len: 3 })?;
    let g = f
        .module()
        .builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)?;
    let p = f.module().global_value(g)?;
    let v = f.load(p, MemAccess::default())?;
    f.composite_extract(v, &[0])
}

/// `base + i * stride + k` as a value.
fn index(
    f: &mut FunctionBuilder<'_>,
    u32_: TypeId,
    i: Value,
    stride: u32,
    k: u32,
) -> BResult<Value> {
    let stride = word(f, u32_, stride)?;
    let k = word(f, u32_, k)?;
    let scaled = f.binary(BinOp::IMul, i, stride)?;
    f.binary(BinOp::IAdd, scaled, k)
}

// ---------------------------------------------------------------------------------------------
// A narrow mask after a signed clamp.

/// The narrow widths the mask check takes, each one of its input words' pairs.
const WIDTHS: [u8; 2] = [8, 16];

/// The narrow mask check's input words per invocation: its `(a, b)` pair at 8 bits, then at 16.
const MASK_INPUTS: u32 = 4;

/// How the mask check combines a pair, in its output order: `(subtracted, chained)`, a sum or a
/// difference, clamped by `SClamp` or by `SMin` then `SMax`.
const COMBINATIONS: [(bool, bool); 4] =
    [(false, false), (false, true), (true, false), (true, true)];

/// The mask check's output words per invocation: for each width and combination, the scalar
/// result, then the three lanes of the vector one (the pairs of this invocation and the two
/// after it).
const MASK_OUTPUTS: u32 = (WIDTHS.len() * COMBINATIONS.len() * 4) as u32;

/// The invocations of the mask check: one workgroup.
const MASK_INVOCATIONS: u32 = LOCAL_SIZE;

/// An `i<bits>` held zero-extended in a `u32`, `lanes` wide.
fn narrow(bits: u8, lanes: u8) -> Form {
    Form {
        held: Held::Int {
            bits,
            container: 32,
        },
        lanes,
    }
}

/// The value of an `N`-bit input pair's word `k` (0 or 1) of invocation `i`: the range's edges
/// (0, 1, -1, the largest, the smallest and their neighbours) for the first 16 invocations, the
/// rest from a fixed sequence, as `N` bits zero-extended.
fn mask_input(bits: u8, i: u32, k: u32) -> u32 {
    let m = ones(bits) as u32;
    let top = 1u32 << (bits - 1);
    let edges = [0, 1, m, top - 1, top, top - 2, top + 1, 2];
    let n = i * 2 + k;
    match n {
        n if n < 16 => edges[((n * 3 + k) % 8) as usize],
        n => (n.wrapping_mul(0x9e37_79b9).rotate_left(7) ^ 0x5bd1_e995) & m,
    }
}

/// `x` (`N` bits, zero-extended) sign-extended.
fn sign_extended(x: u32, bits: u8) -> i64 {
    let shift = 32 - u32::from(bits);
    i64::from(((x << shift) as i32) >> shift)
}

/// The answer of the mask check's combination of `a` and `b` at `bits`: the sum or difference,
/// clamped to `bits`' signed range, its low `bits` bits.
fn mask_answer(bits: u8, subtracted: bool, a: u32, b: u32) -> u32 {
    let (a, b) = (sign_extended(a, bits), sign_extended(b, bits));
    let s = if subtracted { a - b } else { a + b };
    let limit = (1i64 << (bits - 1)) - 1;
    let c = s.clamp(-limit - 1, limit);
    (c as u32) & (ones(bits) as u32)
}

/// The check of `DriverWorkaround::NarrowMaskAfterSignedClamp` on `device` (the module docs):
/// storage buffer 0 holds each invocation's input pairs (at 8 bits, then 16), buffer 1 receives
/// its results (for each width and combination of the clamp, the scalar result and a vector's
/// three lanes); one workgroup of [`CheckKernel::LOCAL_SIZE`].
pub fn narrow_mask_check(device: &Device) -> Result<CheckKernel, Error> {
    let n = MASK_INVOCATIONS;
    let inputs: Vec<u32> = (0..n)
        .flat_map(|i| {
            WIDTHS
                .iter()
                .flat_map(move |&bits| [mask_input(bits, i, 0), mask_input(bits, i, 1)])
        })
        .collect();
    let pair = |bits: u8, i: u32| {
        let w = if bits == 8 { 0 } else { 2 };
        let at = (i % n * MASK_INPUTS + w) as usize;
        (inputs[at], inputs[at + 1])
    };
    let mut expected = Vec::with_capacity((n * MASK_OUTPUTS) as usize);
    for i in 0..n {
        for bits in WIDTHS {
            for (subtracted, _) in COMBINATIONS {
                let (a, b) = pair(bits, i);
                expected.push(mask_answer(bits, subtracted, a, b));
                for lane in 0..3 {
                    let (a, b) = pair(bits, i + lane);
                    expected.push(mask_answer(bits, subtracted, a, b));
                }
            }
        }
    }
    let outputs = vec![0; (n * MASK_OUTPUTS) as usize];
    let spirv = finish(device, mask_kernel()?, "narrow mask check")?;
    Ok(CheckKernel::new(
        spirv,
        1,
        vec![inputs, outputs],
        1,
        expected,
    ))
}

/// [`narrow_mask_check`]'s kernel.
fn mask_kernel() -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    b.set_mask_form(MaskForm::And);
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
    let input = words(&mut b, 0, u32_)?;
    let output = words(&mut b, 1, u32_)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let i = invocation(&mut f, u32_)?;
    let count = word(&mut f, u32_, MASK_INVOCATIONS)?;
    // The pair of width slot `w` (0 for 8 bits, 2 for 16) of invocation `i + lane`, wrapped.
    let load_pair = |f: &mut FunctionBuilder<'_>, w: u32, lane: u32| -> BResult<(Value, Value)> {
        let lane = word(f, u32_, lane)?;
        let j = f.binary(BinOp::IAdd, i, lane)?;
        let j = f.binary(BinOp::UMod, j, count)?;
        let a = index(f, u32_, j, MASK_INPUTS, w)?;
        let b = index(f, u32_, j, MASK_INPUTS, w + 1)?;
        let a = at(f, input, u32_, a)?;
        let b = at(f, input, u32_, b)?;
        Ok((
            f.load(a, MemAccess::default())?,
            f.load(b, MemAccess::default())?,
        ))
    };
    let mut out = 0;
    for (slot, bits) in WIDTHS.into_iter().enumerate() {
        let w = 2 * slot as u32;
        let (a, b) = load_pair(&mut f, w, 0)?;
        let mut lanes_a = Vec::new();
        let mut lanes_b = Vec::new();
        for lane in 0..3 {
            let (x, y) = load_pair(&mut f, w, lane)?;
            lanes_a.push(x);
            lanes_b.push(y);
        }
        let va = f.composite_construct(uvec3, &lanes_a)?;
        let vb = f.composite_construct(uvec3, &lanes_b)?;
        for (subtracted, chained) in COMBINATIONS {
            let scalar = clamped(&mut f, bits, 1, subtracted, chained, a, b)?;
            let k = index(&mut f, u32_, i, MASK_OUTPUTS, out)?;
            let p = at(&mut f, output, u32_, k)?;
            f.store(p, scalar, MemAccess::default())?;
            let vector = clamped(&mut f, bits, 3, subtracted, chained, va, vb)?;
            for lane in 0..3 {
                let e = f.composite_extract(vector, &[lane])?;
                let k = index(&mut f, u32_, i, MASK_OUTPUTS, out + 1 + lane)?;
                let p = at(&mut f, output, u32_, k)?;
                f.store(p, e, MemAccess::default())?;
            }
            out += 4;
        }
    }
    f.ret(None)?;
    let main = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        main,
        vec![ExecMode::LocalSize {
            x: LOCAL_SIZE,
            y: 1,
            z: 1,
        }],
    )?;
    b.finalize()
}

/// `a` and `b` (`bits` wide, zero-extended, `lanes` lanes) sign-extended, added or subtracted,
/// clamped signed to `bits`' range by `SClamp` or (`chained`) `SMin` then `SMax`, and truncated:
/// the instructions translation writes for the narrow arithmetic (the arith tests'
/// `a_narrow_mask_after_a_signed_clamp_is_an_extract_under_the_workaround`).
fn clamped(
    f: &mut FunctionBuilder<'_>,
    bits: u8,
    lanes: u8,
    subtracted: bool,
    chained: bool,
    a: Value,
    b: Value,
) -> BResult<Value> {
    let form = narrow(bits, lanes);
    let wide = narrow(32, lanes);
    let a = cast(f, CastKind::SExt, form, wide, a)?;
    let b = cast(f, CastKind::SExt, form, wide, b)?;
    let op = if subtracted { BinOp::ISub } else { BinOp::IAdd };
    let s = f.binary(op, a, b)?;
    let lo = splat(f, wide, ones(32) & !(ones(bits) >> 1))?;
    let hi = splat(f, wide, ones(bits) >> 1)?;
    let c = if chained {
        let m = f.glsl(GlslOp::SMin, &[s, hi])?;
        f.glsl(GlslOp::SMax, &[m, lo])?
    } else {
        f.glsl(GlslOp::SClamp, &[s, lo, hi])?
    };
    cast(f, CastKind::Trunc, wide, form, c)
}

// ---------------------------------------------------------------------------------------------
// A `half` round trip folded.

/// The workgroups of the round-trip check, each of [`LOCAL_SIZE`] invocations, one input each.
const ROUND_TRIP_WORKGROUPS: u32 = 64;

/// The round-trip check's inputs: `f32`s none of which is a `half`, but the zeros, the
/// infinities and the canonical NaN: the largest half's neighbours, values between two halves of
/// every half exponent, every tie between two halves of each binade, values in the half denormal
/// range, values past 65504 up to and past 65520, values below the smallest half denormal, and
/// NaNs, quiet and signalling, whose payloads lie above, across and below the ten bits a `half`
/// keeps, each of both signs. The entry's `SignedZeroInfNanPreserve` for 16 bits keeps a NaN a NaN
/// through the conversion, its bits the device's. No `f32` denormal (the entry declares no mode for
/// them).
fn round_trip_inputs() -> Vec<u32> {
    let count = (ROUND_TRIP_WORKGROUPS * LOCAL_SIZE) as usize;
    let mut out = vec![
        0x0000_0000,
        0x8000_0000,
        0x7f80_0000,
        0xff80_0000,
        0x477f_e001,
        0x477f_efff,
        0x477f_f000,
        0x477f_f001,
        0x4780_0000,
        0xc77f_efff,
        0xc77f_f000,
        0x7fc0_0000,
        0xffc0_0000,
        0x7f80_0001,
        0xff80_0001,
        0x7fbf_ffff,
        0xffbf_ffff,
        0x7fc0_0001,
        0xffc0_0001,
        0x7fff_ffff,
        0xffff_ffff,
        0x7fc1_2345,
        0x7fe0_0000,
        0x7f80_2000,
        0xff80_2000,
        0x7fc0_2000,
        0x7fa0_0000,
        0x7fd0_0000,
        0xfff0_0000,
        0x7f80_1fff,
        0xff80_1000,
    ];
    let mut state = 0x2545_f491_u32;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    let mut k = 0u32;
    while out.len() < count {
        let r = next();
        let sign = (k & 1) << 31;
        let bits = match k % 5 {
            // Between two halves: a normal-half exponent (2^-14 to 2^15), a significand whose
            // low 13 bits are neither 0 nor the tie.
            0 => {
                let exponent = 113 + (r >> 23) % 30;
                let low = 1 + (r % 0x0fff);
                (exponent << 23) | (r & 0x007f_e000) | low
            }
            // A tie between two halves of a normal binade.
            1 => {
                let exponent = 113 + (r >> 23) % 30;
                (exponent << 23) | (r & 0x007f_e000) | 0x1000
            }
            // The half denormal range (2^-24 to 2^-14), and just below its smallest.
            2 => {
                let exponent = 102 + (r >> 23) % 11;
                (exponent << 23) | (r & 0x007f_ffff) | 1
            }
            // Past the largest half and up to 2^17, where the rounding gives infinity.
            3 => 0x477f_e001 + (r % 0x0010_1fff),
            // A NaN of any payload, quiet or signalling.
            _ => 0x7f80_0000 | (r & 0x007f_ffff).max(1),
        };
        out.push(sign | bits);
        k += 1;
    }
    out
}

/// Why `device` cannot run [`half_round_trip_check`]: it has no `shaderFloat16`, or its float
/// controls do not make the round trip Metal's conversion (`narrow::controlled_modes_of`), so no
/// module of it rounds in that form and the workaround changes nothing there.
pub fn half_round_trip_unsupported(device: &Device) -> Option<&'static str> {
    let fields = device.profile().fields();
    if !fields.leaf.float16 {
        return Some("the device has no shaderFloat16");
    }
    if controlled_modes_of(&fields.float_controls).is_none() {
        return Some(
            "the device's float controls lack RoundingModeRTE, DenormPreserve or \
             SignedZeroInfNanPreserve for 16 bits, or the modes their independence ties to them",
        );
    }
    None
}

/// The check of `DriverWorkaround::HalfConversionPairFolded` on `device` in math mode `mode` (the
/// module docs): storage buffer 0 holds the inputs (the module docs), buffer 1 receives each
/// one's round trip, against Metal's `half(float)` widened; one input per invocation of workgroups
/// of [`CheckKernel::LOCAL_SIZE`]. Errs where [`half_round_trip_unsupported`] names a reason.
pub fn half_round_trip_check(device: &Device, mode: MathMode) -> Result<CheckKernel, Error> {
    let modes = controlled_modes_of(&device.profile().fields().float_controls);
    let mut modes = match (half_round_trip_unsupported(device), modes) {
        (None, Some(modes)) => modes,
        (why, _) => {
            return Err(Error::Internal(crate::error::InternalError::new(
                crate::error::InternalKind::RequirementsExceedProfile,
                format!(
                    "the half round-trip check: {}",
                    why.unwrap_or("no float controls")
                ),
            )))
        }
    };
    modes.extend(mode.exec_modes());
    let inputs = round_trip_inputs();
    let expected = inputs.iter().map(|&x| metal_half_of_f32(x)).collect();
    let outputs = vec![0; inputs.len()];
    let spirv = finish(device, round_trip_kernel(modes)?, "half round-trip check")?;
    Ok(CheckKernel::new(
        spirv,
        ROUND_TRIP_WORKGROUPS,
        vec![inputs, outputs],
        1,
        expected,
    ))
}

/// [`half_round_trip_check`]'s kernel, its entry declaring `modes`.
fn round_trip_kernel(modes: Vec<ExecMode>) -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let f32_ = b.ty(Type::Float { width: 32 })?;
    let f16_ = b.ty(Type::Float { width: 16 })?;
    let input = words(&mut b, 0, u32_)?;
    let output = words(&mut b, 1, u32_)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let i = invocation(&mut f, u32_)?;
    let from = at(&mut f, input, u32_, i)?;
    let x = f.load(from, MemAccess::default())?;
    let x = f.convert(ConvOp::Bitcast, x, f32_)?;
    let h = f.convert_no_contraction(x, f16_)?;
    let y = f.convert_no_contraction(h, f32_)?;
    let y = f.convert(ConvOp::Bitcast, y, u32_)?;
    let to = at(&mut f, output, u32_, i)?;
    f.store(to, y, MemAccess::default())?;
    f.ret(None)?;
    let main = f.finish()?;
    let mut entry_modes = vec![ExecMode::LocalSize {
        x: LOCAL_SIZE,
        y: 1,
        z: 1,
    }];
    entry_modes.extend(modes);
    b.entry(ExecModel::GLCompute, main, entry_modes)?;
    b.finalize()
}

#[cfg(test)]
mod tests {
    use super::{
        mask_answer, mask_kernel, metal_half_of_f32, round_trip_inputs, round_trip_kernel,
        sign_extended, MathMode,
    };
    use crate::float16::{f16_bits_to_f32, f32_to_f16_bits};
    use crate::lower::arith::narrow::controlled_modes_of;
    use crate::spirv::testing::{assert_valid, committed_profiles};

    /// The mask kernel is valid on every committed profile, and the round-trip kernel, in each
    /// math mode, on every one whose float controls declare the modes.
    #[test]
    fn the_compiler_check_kernels_are_valid_on_the_committed_profiles() {
        let mut round_trips = 0;
        for (_, profile) in committed_profiles().expect("profiles") {
            assert_valid(mask_kernel().expect("mask kernel"), &profile);
            let fields = profile.fields();
            if let (true, Some(modes)) = (
                fields.leaf.float16,
                controlled_modes_of(&fields.float_controls),
            ) {
                for mode in MathMode::of_profile(&profile) {
                    let mut modes = modes.clone();
                    modes.extend(mode.exec_modes());
                    assert_valid(round_trip_kernel(modes).expect("round trip"), &profile);
                    round_trips += 1;
                }
            }
        }
        assert!(round_trips > 0, "no committed profile declares the modes");
    }

    /// The mask answers at the edges: a sum past the range clamps, a negative result keeps only
    /// its low bits.
    #[test]
    fn the_mask_answers_clamp_and_truncate() {
        assert_eq!(sign_extended(0xff, 8), -1);
        assert_eq!(mask_answer(8, false, 0x7f, 0x01), 0x7f);
        assert_eq!(mask_answer(8, false, 0x80, 0xff), 0x80);
        assert_eq!(mask_answer(8, true, 0x00, 0x01), 0xff);
        assert_eq!(mask_answer(16, true, 0x8000, 0x0001), 0x8000);
        assert_eq!(mask_answer(16, false, 0xffff, 0xffff), 0xfffe);
    }

    /// Every round-trip input but the zeros, the infinities and the canonical NaN changes through
    /// Metal's conversion, so a folded pair answers each of them wrong.
    #[test]
    fn every_round_trip_input_but_the_specials_is_no_half() {
        let inputs = round_trip_inputs();
        assert_eq!(inputs.len(), 4096);
        let kept: Vec<u32> = inputs
            .iter()
            .copied()
            .filter(|&x| metal_half_of_f32(x) == x)
            .collect();
        assert_eq!(
            kept,
            [
                0x0000_0000,
                0x8000_0000,
                0x7f80_0000,
                0xff80_0000,
                0x7fc0_0000
            ]
        );
    }

    /// The round-trip inputs hold NaNs of both signs, quiet and signalling, each answered as
    /// Metal's `0x7E00` widened; a conversion that keeps a NaN's sign and payload answers most of
    /// them otherwise.
    #[test]
    fn the_round_trip_inputs_hold_nans_of_every_class() {
        let nans: Vec<u32> = round_trip_inputs()
            .into_iter()
            .filter(|&x| x & 0x7fff_ffff > 0x7f80_0000)
            .collect();
        for (sign, quiet) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
            assert!(
                nans.iter()
                    .any(|&x| x >> 31 == sign && (x >> 22) & 1 == quiet),
                "no NaN of sign {sign}, quiet bit {quiet}"
            );
        }
        assert!(nans.iter().all(|&x| metal_half_of_f32(x) == 0x7fc0_0000));
        let kept = |x: u32| f16_bits_to_f32(f32_to_f16_bits(f32::from_bits(x))).to_bits();
        let other = nans.iter().filter(|&&x| kept(x) != 0x7fc0_0000).count();
        assert!(2 * other > nans.len(), "{other} of {} NaNs", nans.len());
    }
}
