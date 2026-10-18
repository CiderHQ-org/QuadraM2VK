//! The sampling routines: the emulation a combined binding's sample shape takes, defined once in
//! the module and called by every sample of that shape with its plain data.
//!
//! **Why.** On a device without `VkProperty::MetalSampler` each linear sample of a measured
//! format is Metal's filter emulated by fetches (`linear`), and a nearest sample of a 32-bit float
//! texture through a runtime sampler fetches its texel (`fetched`). Written at every site, that
//! code made the consumer's CoreAnimation fragment `Xgc` 246,000 instructions from 15,800 AIR
//! instructions, about 574 instructions a sample site and 78% of the module; NVIDIA's driver
//! compiled every copy, 13 to 26 s a pipeline on the RTX 5080. The same emulation as one routine
//! that 256 sites call, declared `DontInline`, made in 0.39 s on the RTX 3090 where the 256 inlined
//! copies took 10.2 s.
//!
//! **What a routine holds.** One per shape ([`Shape`]):
//! - [`Emulation`]: a linear sample's emulated arm (`linear::emit`): Metal's position of the
//!   coordinate and its quantization, each tap's addressing (a runtime sampler's under every mode
//!   its word may state), the border, the fetches, and the sums' calls (`linear::sum`), a cube's
//!   face position and fold, and a comparison's tap tests.
//! - [`Fetched`]: a sample's nearest-exact fetch arm (`fetched::branch`): the level its sampler
//!   reads, the texel's position and addressing, the border, and the fetch.
//! - [`Filter`]: Metal's anisotropic filter emulated (`anisotropic`): the level-0 size and level
//!   count, the sampler's degree and mip filter, the footprint, and the taps' samples at explicit
//!   levels and their sum.
//! - [`Extended`]: an extended-range texture's sample (`extended`): the levels its mip filter
//!   reads, the filter's footprint and addressing, the fetches and their decodes, the sums' calls,
//!   the border and the swizzle.
//!
//! The routine reads its binding's image and sampler variables and its slots' specialization
//! constants (the runtime sampler's word, the offset tests) itself, so no image handle crosses a
//! call, and a device's specialization folds its arithmetic as it folded each site's. The call
//! passes what the site computes in its own position: the coordinate, the array layer, a cube's
//! direction (its ties broken), a comparison's reference, and a fetch's level operand; a constant
//! texel offset with a lane off zero, whose every lane the routine places by Metal's rule, where a
//! 0 lane adds nothing, so the sites of one binding at different offsets call one routine; and the
//! bound format's class, which a function reads as data (`SpecKey::TexelFormat`), as the sums'
//! routines take it. The site keeps the class and word tests, the branch, the device's sample with
//! its level, bias or gradients, and the texel's conversion: no implicit level crosses a call, and
//! each arm is the code it was, so every answer is the one the inline code gave.
//!
//! An anisotropic sample's call passes its coordinate (a cube's direction, its ties broken), its
//! gradients (a cube's on its face) and a cube's face steps, the array layer and the
//! `min_lod_clamp`, and its texel offset, constant or dynamic, where the profile admits the
//! `Offset` operand on a sample (`maintenance8` and `shaderImageGatherExtended`), which the taps'
//! samples take, so the samples at every offset call one routine; elsewhere a constant offset is
//! the shape's, as the taps' `ConstOffset`. The site keeps the gradients (no implicit derivative
//! crosses a call), the word's test, the plain arm and the texel's conversion of the sum. An
//! extended-range sample's call passes its coordinate, the array layer and the level its mip
//! filter reads at (an explicit level, or the level of detail its site takes from derivatives or
//! gradients), and its constant texel offset, which the routine places on each lane, so the
//! samples at every offset call one routine; the site keeps the result's status word.
//!
//! **Not routines.** A heap image's sample (an argument buffer's), whose `NonUniform` handle a
//! call would carry (a later step's check); an extended-range sample on a profile whose driver is
//! not known to keep a `DontInline` call a call (`extended::Sample::routine`), where the routine
//! measured slower to make than the site code; a switch-at-use's shared emulation
//! (`sampler::shared_value`), which runs once already; a gather's fetches; an anisotropic sample
//! at a dynamic texel offset on a profile without the `Offset` operand on a sample, whose arms its
//! site's access reads; and the bicubic filter's site code.
//!
//! **The call's control.** A sampling routine is started by
//! `ModuleBuilder::sampling_routine`: `DontInline` where the profile states that its driver keeps
//! such a call a call (`DriverWorkaround::DontInlineKeepsCalls`, a cost fact keyed on the driver
//! release and GPU measured) or that it inlines every function at each call
//! (`DriverWorkaround::FunctionsInlinedAtEveryCall`, MoltenVK's); no control elsewhere, the
//! default, whose code and answers are the same. Mesa's drivers inline every function whatever it
//! states, so there the module is smaller and the pipeline the same.
//!
//! **Census**. A routine's words are counted once per shape the plan notes ([`note`],
//! `sampler::routine_words`), under `sampler.linear.routine` and `sampler.fetch.routine`; the
//! site counts its call ([`CALL_WORDS`]) where it counted the code; an extended-range sample's
//! routine is counted under `sampler.extended_range.routine` (`extended::Sample::words` its
//! site's). The census counts no anisotropic emulation, at its sites or in a routine.

use std::collections::HashMap;

use crate::lower::texture::TexelConv;
use crate::metal::format::ExtendedRange;
use crate::spirv::ids::{FuncId, TypeId};

use super::anisotropic::Anisotropic;
use super::bicubic::Bicubic;
use super::extended;
use super::fetched::{Fetch, FetchAddress, FetchLevel};
use super::level::{LevelRule, LevelSnap};
use super::linear::{Linear, LinearState};
use super::{BindingKey, SampleOp, SamplerPlan, Sampling, Source};

/// The estimated words a routine adds besides its body: `OpFunction`, its parameters, the entry
/// label, the return, and `OpFunctionEnd`.
pub const ROUTINE_WORDS: u32 = 48;

/// The estimated words a sample spends at its site on a routine's call: the call with its
/// arguments, and the arm's label and branch.
pub const CALL_WORDS: u32 = 32;

/// The shape of a sampling routine (module docs): what its body depends on, besides its call's
/// argument types.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Shape {
    /// A linear sample's emulated arm (`linear::emit`).
    Emulation(Emulation),
    /// A sample's nearest-exact fetch arm (`fetched::branch`).
    Fetched(Fetched),
    /// Metal's anisotropic filter emulated (`anisotropic`).
    Anisotropic(Filter),
    /// An extended-range texture's sample (`extended`).
    Extended(Box<Extended>),
}

impl Shape {
    /// The census name of the shape's routines.
    pub fn census_name(&self) -> &'static str {
        match self {
            Shape::Emulation(_) => "sampler.linear.routine",
            Shape::Fetched(_) => "sampler.fetch.routine",
            Shape::Anisotropic(_) => "sampler.anisotropic.routine",
            Shape::Extended(_) => "sampler.extended_range.routine",
        }
    }

    /// An upper estimate of the routine's words: its body's, which the site counted previously, and
    /// the function's own.
    pub fn words(&self) -> u32 {
        match self {
            Shape::Emulation(e) => e.linear.words() + offset_words(e.offset) + ROUTINE_WORDS,
            Shape::Fetched(f) => f.words + offset_words(f.offset) + ROUTINE_WORDS,
            Shape::Anisotropic(_) => FILTER_WORDS + ROUTINE_WORDS,
            Shape::Extended(e) => e.words + ROUTINE_WORDS,
        }
    }
}

/// A linear sample's emulated arm (`linear::emit`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Emulation {
    /// The sample's own combined binding, whose image facts the footprint reads.
    pub own: BindingKey,
    /// The binding the taps are fetched through: a comparison's without comparison, a runtime
    /// sampler's raw view (`Linear::fetch`), else the sample's own.
    pub taps: BindingKey,
    /// The emulation's plan, without its constant texel offset, which the call passes.
    pub linear: Linear,
    /// The lanes of the texel offset the call passes, where the sample's has a lane off zero.
    pub offset: Option<u8>,
    /// The result is `half`.
    pub half: bool,
    /// The emulation places a texel offset by its slot's `OffsetBaseAxes` tests.
    pub parent: bool,
    /// The sample reads an array's layer.
    pub layer: bool,
}

impl Emulation {
    /// The emulated arm sample `s` calls: a linear sample of a texture slot through its combined
    /// binding that Metal's filter answers by fetches (`Sampling::linear`); `None` for any other.
    pub(super) fn of(s: &Sampling) -> Option<Emulation> {
        let (Source::Combined(own), Some(l)) = (s.source, s.linear.as_ref()) else {
            return None;
        };
        l.image.slot()?;
        let half = s.texel?.conv == TexelConv::F16;
        // A constexpr sampler's raw emulation reads the sample's own binding, the raw view.
        let taps = match (l.raw, &l.state, l.fetch) {
            (true, LinearState::Constant { .. }, _) | (_, _, None) => own,
            (_, _, Some(k)) => k,
        };
        Some(Emulation {
            own,
            taps,
            linear: Linear {
                offset: None,
                ..l.clone()
            },
            offset: passed(l.offset.as_deref()),
            half,
            parent: s.offset_base,
            layer: s.layer.is_some(),
        })
    }
}

/// The estimated words of an anisotropic emulation's body (`anisotropic`), its taps past the first
/// in one loop: about 2,100 measured on a runtime slot's.
pub const FILTER_WORDS: u32 = 2_400;

/// Metal's anisotropic filter emulated (`anisotropic`), as a sample of a texture slot through its
/// combined binding takes it.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Filter {
    /// The sample's combined binding.
    pub binding: BindingKey,
    /// The sampler's degree, mip filter and the device's anisotropy.
    pub filter: Anisotropic,
    /// The coordinate's plane lanes (2; a cube direction's 3).
    pub lanes: u8,
    /// The sample reads a cube: its call passes the face's steps.
    pub cube: bool,
    /// The sample reads an array's layer.
    pub layer: bool,
    /// The sample states a `min_lod_clamp`.
    pub min_lod: bool,
    /// The constant texel offset the taps' samples take as `ConstOffset`, where the sample has one
    /// and the profile admits no `Offset` operand on a sample.
    pub offset: Option<Vec<i32>>,
    /// The call passes the sample's texel offset, constant or dynamic, which the taps' samples
    /// take as their `Offset` operand: on a profile that admits it on a sample (`maintenance8`
    /// and `shaderImageGatherExtended`), so the samples at every offset call one routine.
    pub passed_offset: bool,
}

/// An extended-range texture's sample (`extended::Sample`) without the operands its call passes:
/// its coordinate, layer and the level its mip filter reads at.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Extended {
    /// The texture slot, whose variant binds [`form`](Self::form).
    pub texture: u32,
    /// The extended-range form the slot's texels are decoded from.
    pub form: ExtendedRange,
    /// The sampler's state the sample follows.
    pub state: extended::State,
    /// Metal's bicubic filter of the decoded texels, where the sampler's state is cubic.
    pub cubic: Option<Bicubic>,
    /// A gather of this component instead of a sample.
    pub gather: Option<u32>,
    /// The sample swizzles its decoded texel under the slot's `TextureSwizzle` specialization.
    pub swizzle: bool,
    /// The lanes of the sample's constant texel offset, which the call passes, where it has one.
    pub offset: Option<u8>,
    /// The offset is placed by the slot's `SpecKey::OffsetBaseAxes` tests.
    pub offset_base: bool,
    /// A 3D texture: the sample reads three plane axes.
    pub volume: bool,
    /// The sample reads an array's layer, which the call passes.
    pub layer: bool,
    /// The mip filter the level is read by; a level the call passes but at the base level.
    pub level: extended::Mip,
    /// The result is `half`, else `float`.
    pub half: bool,
    /// The census words of the routine's body (`extended::Sample::body_words`).
    pub words: u32,
}

impl Extended {
    /// The routine extended-range sample `e` calls.
    pub(super) fn of(e: &extended::Sample) -> Extended {
        Extended {
            texture: e.texture,
            form: e.form,
            state: e.state.clone(),
            cubic: e.cubic.clone(),
            gather: e.gather,
            swizzle: e.swizzle,
            offset: e.offset.as_ref().map(|o| o.len() as u8),
            offset_base: e.offset_base,
            volume: e.volume,
            layer: e.layer.is_some(),
            level: e.level.mip(),
            half: e.half,
            words: e.body_words(),
        }
    }

    /// The plane axes the sample reads: a 2D texture's two, a 3D texture's three.
    pub(super) fn axes(&self) -> usize {
        match self.volume {
            true => 3,
            false => extended::AXES,
        }
    }
}

/// Note the routine of extended-range sample `e` in `plan`, its shape once, for the census.
pub(super) fn note_extended(plan: &mut SamplerPlan, e: &extended::Sample) {
    let shape = Shape::Extended(Box::new(Extended::of(e)));
    if !plan.routines.contains(&shape) {
        plan.routines.push(shape);
    }
}

/// A sample's nearest-exact fetch arm (`fetched::branch`): one texel of the level its sampler
/// reads.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Fetched {
    /// The combined binding whose image the texel is fetched from.
    pub binding: BindingKey,
    /// The texture slot whose `TexelFormat` specialization its border texel reads.
    pub texture: u32,
    /// How the tap is addressed.
    pub address: FetchAddress,
    /// The lanes of Metal's constant texel offset, which the call passes, where it has a lane off
    /// zero.
    pub offset: Option<u8>,
    /// The level read.
    pub level: Level,
    /// The fetch places a texel offset by its slot's `OffsetBaseAxes` tests.
    pub parent: bool,
    /// The sample reads an array's layer.
    pub layer: bool,
    /// The fetch's estimated words without its offset (`Fetch::words`).
    pub words: u32,
}

/// The level a [`Fetched`] routine reads: `fetched::FetchLevel` without the call's operand,
/// which the call passes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Level {
    /// A level known when the function compiles.
    Fixed(u32),
    /// `floor` of the level a constexpr sampler's snap hands the device for the operand.
    Snapped(LevelSnap),
    /// The level a runtime sampler's word reads at the operand, or at level 0 without one.
    Word { operand: bool },
}

impl Fetched {
    /// The fetch arm of sample `s` fetching as `f` from texture slot `texture` through combined
    /// binding `binding`; `None` for a gather (four texels) or a blend of two levels.
    pub(super) fn of(
        s: &Sampling,
        binding: BindingKey,
        texture: u32,
        f: &Fetch,
    ) -> Option<Fetched> {
        if !matches!(s.op, SampleOp::Sample { .. }) {
            return None;
        }
        let level = match f.level {
            FetchLevel::Fixed(n) => Level::Fixed(n),
            FetchLevel::Snapped(_) => match s.level {
                Some(LevelRule::Stated(snap)) => Level::Snapped(snap),
                _ => return None,
            },
            FetchLevel::Word(v) => Level::Word {
                operand: v.is_some(),
            },
            FetchLevel::Blend(..) => return None,
        };
        let words = Fetch {
            offset: None,
            ..f.clone()
        }
        .words(&s.op);
        Some(Fetched {
            binding,
            texture,
            address: f.address.clone(),
            offset: passed(f.offset.as_deref()),
            level,
            parent: s.offset_base,
            layer: s.layer.is_some(),
            words,
        })
    }

    /// The fetch arm a plain nearest sample `s` calls (`Sampling::fetched` on its slot, through
    /// its own combined binding), which a linear sample's device arm calls too.
    pub(super) fn plain(s: &Sampling) -> Option<Fetched> {
        let (Source::Combined(key), Some(p)) = (s.source, s.fetched.as_ref()) else {
            return None;
        };
        let texture = p.image.slot()?;
        Fetched::of(s, key, texture, &p.fetch)
    }
}

/// The lanes of constant texel offset `offset` a routine's call passes: `None` where no lane is off
/// zero, which places nothing.
fn passed(offset: Option<&[i32]>) -> Option<u8> {
    let o = offset.filter(|o| o.iter().any(|o| *o != 0))?;
    u8::try_from(o.len()).ok()
}

/// The estimated words of placing a routine's offset of `lanes` lanes: Metal's offset rule on
/// each (`position::OFFSET_RULE_WORDS`).
fn offset_words(lanes: Option<u8>) -> u32 {
    lanes.map_or(0, u32::from) * super::position::OFFSET_RULE_WORDS
}

/// Note the routines sampling `s` calls in `plan`, each shape once, for the census.
pub(super) fn note(plan: &mut SamplerPlan, s: &Sampling) {
    let shapes = [
        Emulation::of(s).map(Shape::Emulation),
        Fetched::plain(s).map(Shape::Fetched),
    ];
    for shape in shapes.into_iter().flatten() {
        if !plan.routines.contains(&shape) {
            plan.routines.push(shape);
        }
    }
}

/// The census words routed sampling `s` saves at its site: the bodies of the routines it calls,
/// which [`note`] counts once in the module, less the calls ([`CALL_WORDS`]).
pub(super) fn routed_words(s: &Sampling) -> u32 {
    let emulation = Emulation::of(s)
        .and(s.linear.as_ref())
        .map_or(0, |l| l.words().saturating_sub(CALL_WORDS));
    let fetch = Fetched::plain(s)
        .and(s.fetched.as_ref())
        .map_or(0, |p| p.fetch.words(&s.op).saturating_sub(CALL_WORDS));
    emulation + fetch
}

/// The sampling routines one function's emission has defined, by shape and by the types of their
/// call's arguments: each defined at its first call.
#[derive(Debug, Default)]
pub(crate) struct Routines {
    defined: HashMap<(Shape, Vec<TypeId>), FuncId>,
}

impl Routines {
    /// The routine of `shape` taking arguments of `args`' types, where defined.
    pub(super) fn get(&self, shape: &Shape, args: &[TypeId]) -> Option<FuncId> {
        self.defined.get(&(shape.clone(), args.to_vec())).copied()
    }

    /// Record `f` as the routine of `shape` taking arguments of `args`' types.
    pub(super) fn insert(&mut self, shape: Shape, args: Vec<TypeId>, f: FuncId) {
        self.defined.insert((shape, args), f);
    }
}
