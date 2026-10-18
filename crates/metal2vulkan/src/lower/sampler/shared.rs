//! The shared emulation of a switch-at-use (SEMANTICS 1012).
//!
//! A switch-at-use whose every arm takes the same emulated sampling on its own texture slot
//! ([`Sampling::shared_shape`]): a bicubic sample, a linear sample Metal's filter answers, a
//! snapped gather or nearest sample, or a plain gather or nearest sample whose base-level texels
//! the shader fetches. Emitted per arm, each arm repeats the whole emulation (its position,
//! weights, sums, and routine calls), so the switch's words grow with arms × emulation, and so does
//! the driver's work once the bound formats make every arm's emulation live (an Intel ANV pipeline
//! of 32 such arms took 6 s to create).
//!
//! **The shared form** runs the emulation once, outside the dispatch, on the named candidate's
//! facts, and dispatches only what reads a candidate's own image:
//!
//! - a facts dispatch, whose arms yield their level-0 size (`OpImageQuerySizeLod` 0), the bound
//!   format's bits (the slot's `BicubicTexels` or `TexelFormat` specialization) and `true`, and
//!   whose null arm yields a size of ones, bits 0 and `false`;
//! - the emulation on those facts (its footprint, weights and routine calls once), whose taps a
//!   dispatch reads from each arm's image (the null arm's are zero);
//! - a linear sample's device sample, where the named format is not measured, dispatched the same
//!   way, converted for a `half` result by the named format's class;
//! - a snapped sample's coordinate snapped once on the named size, and the device's access and
//!   texel conversion in each arm, the null arm yielding the switch's null value;
//! - a plain fetch's taps on the named size, fetched in one dispatch where the named format is of
//!   the 32-bit float class, else the device's access (its coordinate snapped once) in another.
//!
//! Where a handle names a candidate the facts are that candidate's, so every value is the one its
//! own arm computed; elsewhere the switch's null value is selected. The emulation of a null handle
//! runs on a size of ones and zero taps, whose value is discarded.

use crate::lower::emit::EmitCx;
use crate::lower::texture::{dispatch, with_word, NullLeaf};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn, Merged};
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::{ConstId, Value as SValue};
use crate::spirv::ir::{ConstBits, Constant, ImageOperands, ImageQuery, Type as SType};

use super::emit::{
    device_access, held_value, layered, offset_operands, op_offset, pixel_coordinates,
    plane_and_gradients, sample_access, sample_operands, sampler_word, scaled, snap_offsets,
    texel_value, word_level, word_level_size,
};
use super::{Binding, CombinedImage, SampleOp, Sampling, Source, Unmapped};

/// The candidates of a switch-at-use that emulates once: the handles' values dispatched on,
/// outermost first, and each arm's case literals and sampling.
pub(crate) struct Candidates<'a> {
    pub(crate) sels: Vec<SValue>,
    pub(crate) arms: Vec<(&'a [u32], &'a Sampling)>,
}

impl Candidates<'_> {
    /// One dispatch on the handles: `read` yields each arm's values from its sampling in the arm,
    /// `null` the null arm's; the merged values, in order.
    pub(super) fn each(
        &self,
        cx: &mut EmitCx<'_>,
        fb: &mut FunctionBuilder<'_>,
        read: &dyn Fn(&mut EmitCx<'_>, &mut FunctionBuilder<'_>, &Sampling) -> BResult<Vec<SValue>>,
        null: &NullLeaf<'_>,
    ) -> BResult<Vec<SValue>> {
        let leaf =
            |cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>, a: &Sampling| read(cx, fb, a);
        dispatch(cx, fb, &self.sels, &self.arms, &leaf, null)
    }

    /// The sampled image of arm sampling `a`'s binding, made in the open block.
    pub(super) fn sampled_image(
        &self,
        cx: &mut EmitCx<'_>,
        fb: &mut FunctionBuilder<'_>,
        a: &Sampling,
    ) -> BResult<SValue> {
        binding(cx, a)?.0.sampled_image(fb)
    }

    /// The sampled image of arm sampling `a`'s binding its linear taps are fetched through
    /// (`Linear::fetch`; its own binding where it names none), made in the open block: a raw
    /// emulation's raw view, through the arm's runtime sampler.
    pub(super) fn fetch_sampled_image(
        &self,
        cx: &mut EmitCx<'_>,
        fb: &mut FunctionBuilder<'_>,
        a: &Sampling,
    ) -> BResult<SValue> {
        fetch_binding(cx, a)?.sampled_image(fb)
    }

    /// The image of arm sampling `a`'s binding, loaded in the open block: for an emulated linear
    /// comparison the binding its taps are fetched through (`Linear::fetch`).
    pub(super) fn image(
        &self,
        cx: &mut EmitCx<'_>,
        fb: &mut FunctionBuilder<'_>,
        a: &Sampling,
    ) -> BResult<SValue> {
        fetch_binding(cx, a)?.image(fb)
    }
}

/// The combined binding sampling `a` reads, and its image facts.
fn binding(cx: &mut EmitCx<'_>, a: &Sampling) -> BResult<(Binding, CombinedImage)> {
    let Source::Combined(key) = a.source else {
        return Err(cx.fail("a shared sampling of a heap element"));
    };
    match cx.decl.families.sampler.combined.get(&key) {
        Some(&b) => Ok(b),
        None => Err(cx.fail(format!("combined binding {key:?} is not declared"))),
    }
}

/// The binding arm sampling `a`'s linear taps are fetched through (`Linear::fetch`), else its own.
fn fetch_binding(cx: &mut EmitCx<'_>, a: &Sampling) -> BResult<Binding> {
    let key = match (&a.linear, a.source) {
        (Some(super::linear::Linear { fetch: Some(k), .. }), _) => *k,
        (_, Source::Combined(key)) => key,
        _ => return Err(cx.fail("a shared sampling of a heap element")),
    };
    match cx.decl.families.sampler.combined.get(&key) {
        Some(&(b, _)) => Ok(b),
        None => Err(cx.fail(format!("combined binding {key:?} is not declared"))),
    }
}

/// The emulation every arm of a shared switch takes.
#[derive(Clone, Copy)]
enum Emulation<'a> {
    Bicubic(&'a super::bicubic::Bicubic),
    Linear(&'a super::linear::Linear),
    Snap(&'a super::position::Snap),
    /// A plain fetch of the base level, its device arm snapped where it has a snap.
    Fetched(&'a super::fetched::Plain),
}

impl<'a> Emulation<'a> {
    fn of(s: &'a Sampling) -> Option<Emulation<'a>> {
        if let Some(p) = &s.fetched {
            return Some(Emulation::Fetched(p));
        }
        match (&s.op, &s.linear, &s.snap) {
            (SampleOp::Bicubic(b), None, None) => Some(Emulation::Bicubic(b)),
            (SampleOp::Sample { .. }, Some(l), snap)
                if snap.as_ref().is_none_or(super::position::Snap::by_word) =>
            {
                Some(Emulation::Linear(l))
            }
            (SampleOp::Sample { .. } | SampleOp::Gather { .. }, None, Some(p)) if !p.by_word() => {
                Some(Emulation::Snap(p))
            }
            _ => None,
        }
    }

    /// The specialization holding the bound format's bits of arm sampling `a`'s slot, where the
    /// emulation reads them: a bicubic sample's `BicubicTexels`, a linear one's `TexelFormat`.
    fn bits(self, cx: &mut EmitCx<'_>, a: &Sampling) -> BResult<Option<ConstId>> {
        let spec = match (self, Emulation::of(a)) {
            (Emulation::Bicubic(_), Some(Emulation::Bicubic(b))) => {
                cx.decl.families.sampler.bicubic.get(&b.texture).copied()
            }
            (Emulation::Linear(_), Some(Emulation::Linear(l))) => l
                .image
                .slot()
                .and_then(|t| crate::lower::texture::texel_format_const(cx, t)),
            (Emulation::Fetched(_), Some(Emulation::Fetched(p))) => p
                .image
                .slot()
                .and_then(|t| crate::lower::texture::texel_format_const(cx, t)),
            (Emulation::Snap(_), Some(Emulation::Snap(_))) => return Ok(None),
            _ => return Err(cx.fail("a shared switch's arms of different emulations")),
        };
        match spec {
            Some(c) => Ok(Some(c)),
            None => Err(cx.fail("a shared switch's arm without its format specialization")),
        }
    }
}

/// The value of the shared switch over `c` (module docs): its sampling's value where a handle
/// names a candidate, else `null`'s; `grads` are the gradients its rule-2 prelude hoisted.
pub(crate) fn shared_value(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    c: &Candidates<'_>,
    grads: Option<(SValue, SValue)>,
    null: &NullLeaf<'_>,
) -> BResult<SValue> {
    let Some(&(_, first)) = c.arms.first() else {
        return Err(cx.fail("a shared switch without arms"));
    };
    let Some(kind) = Emulation::of(first) else {
        return Err(cx.fail("a shared switch without an emulation"));
    };
    let (_, image) = binding(cx, first)?;
    for &(_, a) in &c.arms {
        if binding(cx, a)?.1 != image {
            return Err(cx.fail("a shared switch's candidates of different images"));
        }
    }
    // A runtime sampler's level by its word: the level handed to the device, and the one a
    // leveled snap reads, whose size each arm's facts give.
    let (lambda, nearest) = word_level(cx, fb, first)?;
    let leveled = match nearest {
        Some(nearest) => Some((sampler_word(cx, fb, first.pixel)?, nearest)),
        None => None,
    };
    // A leveled snap of an access with a texel offset reads the base level's size after its
    // level's: Metal's offset rule reads both.
    let based = leveled.is_some() && op_offset(&first.op).is_some();
    let facts = facts(cx, fb, c, kind, (image, leveled, based))?;
    let (size, base, rest) = match (based, &facts[..]) {
        (true, [size, base, rest @ ..]) => (*size, Some(*base), rest),
        (false, [size, rest @ ..]) => (*size, None, rest),
        _ => return Err(cx.fail("a shared switch's facts without their sizes")),
    };
    let (bits, named, lanes) = match (kind, rest) {
        (Emulation::Snap(_), &[]) => (None, None, None),
        (Emulation::Bicubic(_) | Emulation::Linear(_) | Emulation::Fetched(_), &[bits, named]) => {
            (Some(bits), Some(named), None)
        }
        (
            Emulation::Linear(_) | Emulation::Fetched(_),
            &[bits, named, c0, c1, c2, c3, o0, o1, o2, o3],
        ) => (
            Some(bits),
            Some(named),
            Some(crate::lower::texture::LaneConstants {
                constant: [c0, c1, c2, c3],
                one: [o0, o1, o2, o3],
            }),
        ),
        _ => return Err(cx.fail("a shared switch's facts without their values")),
    };
    // The lanes Metal reads as constants, by the named candidate's tests: each arm's own conversion
    // is the shared one, so they are set after it.
    let constants = |cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>, x: SValue| match lanes {
        Some(tests) => crate::lower::texture::constant_lanes_by(cx, fb, tests, x, None),
        None => Ok(x),
    };
    let x = match (kind, bits) {
        (Emulation::Bicubic(b), Some(bits)) => {
            super::bicubic::emit_shared(cx, fb, first, (size, image), b, bits, c)?
        }
        (Emulation::Linear(l), Some(bits)) => {
            let SampleOp::Sample {
                lod,
                dref,
                min_lod,
                offset,
            } = &first.op
            else {
                return Err(cx.fail("a shared linear sample without its sample"));
            };
            // The device's sample, which a runtime sampler's word snap snaps.
            let (coord, grad) = device_coordinate(
                cx,
                fb,
                first,
                grads,
                ((size, base), image),
                first.snap.as_ref(),
            )?;
            let (ops, dref, beyond) = sample_operands(
                cx,
                fb,
                first,
                (coord.0, grad, lambda, false),
                (lod, *min_lod, offset, *dref),
            )?;
            // A native class's device sample reads Metal's footprint step on the named
            // candidate's size and bits (`native`).
            let facts = (size, super::bicubic::Bits::Value(bits));
            let (at, native) = match super::native::coordinate(cx, fb, first, l, coord, facts)? {
                Some((at, native)) => (at, Some(native)),
                None => (coord.1, None),
            };
            let device = device_access(at, dref, ops, beyond);
            let x = super::linear::emit_shared(
                cx,
                fb,
                first,
                (size, image, dref),
                l,
                (bits, c),
                &device,
                native,
            )?;
            // A depth sample without comparison reads the first component.
            if first.depth && dref.is_none() {
                fb.composite_extract(x, &[0])?
            } else {
                x
            }
        }
        (Emulation::Snap(p), _) => {
            return snapped(
                cx,
                fb,
                c,
                (first, p),
                grads,
                ((size, base), image, lambda),
                null,
            );
        }
        (Emulation::Fetched(p), Some(bits)) => {
            let x = fetched(
                cx,
                fb,
                c,
                (first, p),
                grads,
                ((size, base), image, lambda),
                bits,
            )?;
            // A depth sample without comparison reads the first component.
            let x = if first.depth && matches!(first.op, SampleOp::Sample { .. }) {
                fb.composite_extract(x, &[0])?
            } else {
                x
            };
            // The conversion, then the lanes Metal reads as constants by the merged tests, not
            // the named candidate's own.
            let x = held_value(cx, fb, first, image, x)?;
            let x = constants(cx, fb, x)?;
            let x = with_word(cx, fb, &first.result, x, None)?;
            let Some(named) = named else {
                return Err(cx.fail("a shared fetch without its named test"));
            };
            let [n] = null(cx, fb)?[..] else {
                return Err(cx.fail("a shared switch without its null value"));
            };
            return fb.select(named, x, n);
        }
        _ => return Err(cx.fail("a shared emulation without its format bits")),
    };
    let x = match first.depth {
        true => x,
        false => constants(cx, fb, x)?,
    };
    let x = with_word(cx, fb, &first.result, x, None)?;
    let Some(named) = named else {
        return Err(cx.fail("a shared emulation without its named test"));
    };
    let [n] = null(cx, fb)?[..] else {
        return Err(cx.fail("a shared switch without its null value"));
    };
    fb.select(named, x, n)
}

/// The facts dispatch (module docs): each arm's level-0 size, or where a runtime sampler's
/// leveled snap reads another level (`leveled`: its word and `level::by_word`'s level) that level's
/// size, then where `based` the level-0 size (Metal's offset rule reads both), and where `kind`
/// reads them its format's bits and `true`, then where the shared value takes the lanes Metal reads
/// as constants (`shared_constants`) its slot's eight lane tests (`LaneConstants`); the null arm's
/// ones, 0, and `false`.
fn facts(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    c: &Candidates<'_>,
    kind: Emulation<'_>,
    (image, leveled, based): (CombinedImage, Option<(SValue, SValue)>, bool),
) -> BResult<Vec<SValue>> {
    let u32_ = fb.module().ty(SType::Int {
        width: 32,
        signed: false,
    })?;
    let bool_ = fb.module().ty(SType::Bool)?;
    let scalar = |fb: &mut FunctionBuilder<'_>, ty, bits: u64| {
        fb.module().constant(Constant::Scalar {
            ty,
            bits: ConstBits(bits),
        })
    };
    let one = scalar(fb, u32_, 1)?;
    let lanes = image.plane + u8::from(image.arrayed);
    let ones = if lanes == 1 {
        one
    } else {
        let ty = fb.module().ty(SType::Vector {
            elem: u32_,
            len: lanes,
        })?;
        let parts = vec![one; usize::from(lanes)].into_boxed_slice();
        fb.module().constant(Constant::Composite { ty, parts })?
    };
    let zero = scalar(fb, u32_, 0)?;
    let yes = scalar(fb, bool_, 1)?;
    let no = scalar(fb, bool_, 0)?;
    let with_bits = !matches!(kind, Emulation::Snap(_));
    let lanes_read = shared_constants(kind, c);
    let read = |cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>, a: &Sampling| {
        let img = c.image(cx, fb, a)?;
        let level0 = |fb: &mut FunctionBuilder<'_>| {
            let level = fb.module().const_value(zero)?;
            fb.query(ImageQuery::SizeLod(level), img)
        };
        let mut v = match leveled {
            Some((word, nearest)) => vec![word_level_size(fb, word, nearest, img)?],
            None => vec![level0(fb)?],
        };
        if based {
            v.push(level0(fb)?);
        }
        if let Some(bits) = kind.bits(cx, a)? {
            v.push(fb.module().const_value(bits)?);
            v.push(fb.module().const_value(yes)?);
        }
        if lanes_read {
            let tests = lane_tests(cx, a)?;
            v.extend(tests.constant);
            v.extend(tests.one);
        }
        Ok(v)
    };
    let null_facts = |_: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>| {
        let mut v = vec![fb.module().const_value(ones)?];
        if based {
            v.push(fb.module().const_value(ones)?);
        }
        if with_bits {
            v.push(fb.module().const_value(zero)?);
            v.push(fb.module().const_value(no)?);
        }
        if lanes_read {
            let no = fb.module().const_value(no)?;
            v.extend([no; 8]);
        }
        Ok(v)
    };
    c.each(cx, fb, &read, &null_facts)
}

/// The plane coordinate of sampling `s` as the device reads it, on the named candidate's size
/// `size` (image facts `image`; the level-0 size, or a leveled word snap's level's, `facts`, with
/// the level-0 size `base`): the plane coordinate scaled from pixel coordinates over the level-0
/// size, then snapped by `snap` (a pixel sampler's snap from the coordinate as given), and
/// with the layer appended; and its gradients. The first of the pair is the plane coordinate before
/// the snap, which a bias past the device's limit differentiates.
fn device_coordinate(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    s: &Sampling,
    grads: Option<(SValue, SValue)>,
    ((size, base), image): ((SValue, Option<SValue>), CombinedImage),
    snap: Option<&super::position::Snap>,
) -> BResult<((SValue, SValue), Option<(SValue, SValue)>)> {
    let (mut coord, mut grad) = plane_and_gradients(cx, fb, s, grads)?;
    // A pixel sampler's snap reads the coordinate as given.
    let given = coord;
    if let Some(spec) = pixel_coordinates(cx, fb, s)? {
        // Pixel coordinates scale by the level-0 size, a leveled snap's included.
        let size_lanes = image.plane + u8::from(image.arrayed);
        let level0 = base.unwrap_or(size);
        (coord, grad) = scaled(fb, s.lanes, (level0, size_lanes), spec, (coord, grad))?;
    }
    let plane = coord;
    if let Some(snap) = snap {
        // The device adds the texel offset to the snapped coordinate; each snap moves its index
        // back where Metal drops it.
        let offsets = snap_offsets(cx, fb, op_offset(&s.op), s.lanes)?;
        coord = if let super::position::SnapKind::Word { leveled, .. } = snap.kind {
            let word = sampler_word(cx, fb, s.pixel)?;
            let size_lanes = image.plane + u8::from(image.arrayed);
            // A shared access reads no slot's `OffsetBaseAxes` (`Sampling::shared_shape`).
            super::position::snapped_by_word(
                fb,
                (word, leveled),
                (coord, &offsets, None),
                s.lanes,
                (size, base, size_lanes),
            )?
        } else {
            let from = if snap.pixel { given } else { coord };
            let at = (from, offsets.as_slice(), None);
            super::position::snapped_on(cx, fb, snap, at, s.lanes, (size, image))?
        };
    }
    let (coord, _) = layered(cx, fb, s, coord)?;
    Ok(((plane, coord), grad))
}

/// A shared snapped gather or nearest sample (module docs): the coordinate snapped once on the
/// named size, and in each arm the device's access and the texel's conversion; the null arm
/// yields `null`'s value.
fn snapped(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    c: &Candidates<'_>,
    (first, snap): (&Sampling, &super::position::Snap),
    grads: Option<(SValue, SValue)>,
    (size, image, lambda): ((SValue, Option<SValue>), CombinedImage, Option<SValue>),
    null: &NullLeaf<'_>,
) -> BResult<SValue> {
    let ((plane, coord), grad) =
        device_coordinate(cx, fb, first, grads, (size, image), Some(snap))?;
    let values = match &first.op {
        SampleOp::Sample {
            lod,
            dref,
            min_lod,
            offset,
        } => {
            let (ops, dref, beyond) = sample_operands(
                cx,
                fb,
                first,
                (plane, grad, lambda, false),
                (lod, *min_lod, offset, *dref),
            )?;
            let read = |cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>, a: &Sampling| {
                let si = c.sampled_image(cx, fb, a)?;
                let x = sample_access(coord, dref, ops)(fb, si, false)?;
                // Past the device's limit the sample with the scaled derivatives replaces the
                // implicit one; the implicit sample stays where it was, unconditional.
                let x = match beyond {
                    None => x,
                    Some((test, grads)) => {
                        let mut grad_ops = ops;
                        grad_ops.bias = None;
                        grad_ops.grad = Some(grads);
                        let past: ArmFn<'_, EmitCx<'_>> = Box::new(move |fb, cx| {
                            // A sampled image is consumed in the block that makes it.
                            let si = c.sampled_image(cx, fb, a)?;
                            let x = sample_access(coord, dref, grad_ops)(fb, si, false)?;
                            Ok(ArmEnd::FallThrough(vec![x]))
                        });
                        let within: ArmFn<'_, EmitCx<'_>> =
                            Box::new(move |_, _| Ok(ArmEnd::FallThrough(vec![x])));
                        let Merged(merged) = fb.if_(cx, test, past, Some(within))?;
                        match merged[..] {
                            [x] => x,
                            _ => return Err(cx.fail("a biased sample without its merged texel")),
                        }
                    }
                };
                // A depth sample without comparison reads the first component.
                let x = if a.depth && dref.is_none() {
                    fb.composite_extract(x, &[0])?
                } else {
                    x
                };
                Ok(vec![texel_value(cx, fb, a, image, x, None)?])
            };
            c.each(cx, fb, &read, null)?
        }
        SampleOp::Gather {
            component,
            dref,
            offset,
        } => {
            let mut ops = ImageOperands::default();
            offset_operands(cx, fb, offset, first.lanes, &mut ops)?;
            let component = *component;
            let dref = match dref {
                Some(r) => Some(cx.operand(fb, *r)?),
                None => None,
            };
            let read = |cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>, a: &Sampling| {
                let si = c.sampled_image(cx, fb, a)?;
                let x = match dref {
                    Some(r) => fb.dref_gather(si, coord, r, &ops)?,
                    None => fb.gather(si, coord, component, &ops)?,
                };
                Ok(vec![texel_value(cx, fb, a, image, x, None)?])
            };
            c.each(cx, fb, &read, null)?
        }
        _ => return Err(cx.fail("a shared snapped access without its sample or gather")),
    };
    match values[..] {
        [x] => Ok(x),
        _ => Err(cx.fail("a shared snapped access without its value")),
    }
}

/// A shared plain fetch of the base level (module docs): the device's coordinate snapped once on
/// the named size where the sampling has a snap, and `fetched::emit_shared`'s fetches of each arm's
/// texels where the named format `bits` is of the 32-bit float class, else each arm's device
/// access: a sample's one texel, or a gather's component `component` of its four.
fn fetched(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    c: &Candidates<'_>,
    (first, p): (&Sampling, &super::fetched::Plain),
    grads: Option<(SValue, SValue)>,
    ((size, base), image, lambda): ((SValue, Option<SValue>), CombinedImage, Option<SValue>),
    bits: SValue,
) -> BResult<SValue> {
    let ((plane, coord), grad) = device_coordinate(
        cx,
        fb,
        first,
        grads,
        ((size, base), image),
        first.snap.as_ref(),
    )?;
    match &first.op {
        SampleOp::Sample {
            lod,
            dref: None,
            min_lod,
            offset,
        } => {
            let (ops, dref, beyond) = sample_operands(
                cx,
                fb,
                first,
                (plane, grad, lambda, false),
                (lod, *min_lod, offset, None),
            )?;
            let device = device_access(coord, dref, ops, beyond);
            super::fetched::emit_shared(
                cx,
                fb,
                first,
                (size, image),
                p,
                (bits, c),
                &super::fetched::one,
                &device,
            )
        }
        SampleOp::Gather {
            component,
            dref: None,
            offset,
        } => {
            let mut ops = ImageOperands::default();
            offset_operands(cx, fb, offset, first.lanes, &mut ops)?;
            let component = *component;
            let vec4 = super::bicubic::Tys::new(fb)?.vec4;
            let gathered = move |fb: &mut FunctionBuilder<'_>, texels: &[SValue]| {
                let mut parts = Vec::with_capacity(texels.len());
                for &x in texels {
                    parts.push(fb.composite_extract(x, &[component])?);
                }
                fb.composite_construct(vec4, &parts)
            };
            let device =
                |fb: &mut FunctionBuilder<'_>, si: SValue| fb.gather(si, coord, component, &ops);
            super::fetched::emit_shared(
                cx,
                fb,
                first,
                (size, image),
                p,
                (bits, c),
                &gathered,
                &device,
            )
        }
        _ => Err(cx.fail("a shared fetch without its plain sample or gather")),
    }
}

/// Whether the shared value of `kind` over `c` sets the lanes Metal reads as constants by its
/// arms' merged lane tests: a linear sample's or a plain fetch's, whose arms take them
/// ([`Unmapped`]). A snapped access converts each arm's texel in its arm, with its own slot's; a
/// bicubic sample's fetches take none.
fn shared_constants(kind: Emulation<'_>, c: &Candidates<'_>) -> bool {
    matches!(kind, Emulation::Linear(_) | Emulation::Fetched(_))
        && c.arms.iter().any(|(_, a)| a.unmapped != Unmapped::Device)
}

/// The lane tests of arm sampling `a`'s texture slot.
fn lane_tests(cx: &mut EmitCx<'_>, a: &Sampling) -> BResult<crate::lower::texture::LaneConstants> {
    let Source::Combined(key) = a.source else {
        return Err(cx.fail("a shared sampling of a heap element"));
    };
    match cx.decl.families.texture.constant_lanes.get(&key.texture) {
        Some(&tests) => Ok(tests),
        None => Err(cx.fail(format!(
            "texture slot {} has no constant lanes",
            key.texture
        ))),
    }
}
