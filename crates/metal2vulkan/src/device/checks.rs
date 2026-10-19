//! The device's own checks of what its driver answers, and the order every driver fact is read in.
//!
//! A fact of [`CHECKED`] is decided on the device itself: the runtime runs a short check of the
//! driver's answer (a bounded dispatch, draw or copy, read back and compared with Metal's answer)
//! and hands what it found to the description
//! ([`DeviceDescription::checks`](crate::device::DeviceDescription::checks)). Every reader of a
//! driver fact, the deviations translation reads, translation's defects, the runtime's driver facts
//! and its defects, takes it in this order ([`decide`]):
//!
//! 1. **Checked here.** A check that answered Metal's ([`CheckAnswer::Metals`]) takes a deviation
//!    or an answer fact and clears a defect; one that answered otherwise
//!    ([`CheckAnswer::NotMetals`]) does the opposite.
//! 2. **Unanswered.** A check that errored, timed out or answered neither
//!    ([`CheckAnswer::Unanswered`]) takes the exact path: no deviation or answer fact, a defect
//!    worked around. A fact whose workaround refuses work the baseline runs takes the baseline
//!    instead ([`IfUnanswered::Baseline`]).
//! 3. **Baseline.** Otherwise (no check answered yet, its check does not run on the device,
//!    [`CheckAnswer::NotRun`], or the policy distrusts checks,
//!    [`DeviceChecksPolicy::Distrusted`](crate::device::DeviceChecksPolicy::Distrusted)) the
//!    curated tables decide, as measured (`driver_facts.rs`): a fact a row names on the setup, a
//!    defect a row names, and elsewhere the exact path for an answer and Vulkan's own for a defect.
//! 4. **Learned off.** A fact the device lost itself on ([`DeviceChecks::learn`]: the runtime
//!    read a record of a lost device while a submission that took the fact's direct path was in
//!    flight) takes the exact side where the steps above left it on the direct one: a defect worked
//!    around, an answer not taken ([`Decided::LearnedOff`]). Not where the policy distrusts checks.
//!
//! A deviation a check decides is in the compile profile as any deviation is, so a different
//! answer is a different profile, fingerprint and cache key. A passed check records no precision
//! note: the answer is Metal's on the device.

use crate::device::driver_facts::DriverFact;
use crate::device::profile::DriverWorkaround;
use crate::error::{DriverDeviation, RuntimeDriverDefect};

/// A fact of a driver that one of the readers takes: a deviation translation reads, a
/// translation's defect, a runtime's driver fact or a runtime's defect.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DeviceFact {
    /// A deviation translation reads: the profile takes it.
    Deviation(DriverDeviation),
    /// A defect translation works around.
    Workaround(DriverWorkaround),
    /// A runtime's driver fact.
    Fact(DriverFact),
    /// A runtime's defect.
    Defect(RuntimeDriverDefect),
    /// The floor of Function storage, 16 KiB a thread, held by every stage of a device whose
    /// Function storage is not measured (`ProfileFields::function_storage`), where a vertex or
    /// fragment function otherwise runs past it with `PrecisionNote::FunctionStorageAssumed`:
    /// held only where the device learned it off ([`DeviceChecks::learn`]).
    FunctionStorageFloor,
}

impl DeviceFact {
    /// Whether the fact, held, is the driver doing otherwise than Metal, which the device works
    /// around, so a check that answers Metal's clears it: a translation's defect, a runtime's
    /// defect, and a driver fact assumed where unmeasured
    /// ([`DriverFact::assumed_where_unmeasured`]). Held, any other fact is Metal's answer, taken.
    pub const fn is_defect(self) -> bool {
        match self {
            DeviceFact::Deviation(_) => false,
            DeviceFact::Workaround(_) | DeviceFact::Defect(_) => true,
            DeviceFact::Fact(fact) => fact.assumed_where_unmeasured(),
            DeviceFact::FunctionStorageFloor => true,
        }
    }

    /// The fact's name: its variant's.
    pub fn name(self) -> String {
        match self {
            DeviceFact::Deviation(fact) => format!("{fact:?}"),
            DeviceFact::Workaround(fact) => format!("{fact:?}"),
            DeviceFact::Fact(fact) => format!("{fact:?}"),
            DeviceFact::Defect(fact) => format!("{fact:?}"),
            DeviceFact::FunctionStorageFloor => "FunctionStorageFloor".to_owned(),
        }
    }

    /// The kind's name: `deviation`, `workaround`, `fact`, `defect` or `function-storage`.
    pub const fn kind(self) -> &'static str {
        match self {
            DeviceFact::Deviation(_) => "deviation",
            DeviceFact::Workaround(_) => "workaround",
            DeviceFact::Fact(_) => "fact",
            DeviceFact::Defect(_) => "defect",
            DeviceFact::FunctionStorageFloor => "function-storage",
        }
    }
}

/// What a device's check of one fact found.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CheckAnswer {
    /// The driver answered Metal's answer on every input of the check.
    Metals,
    /// The driver answered the other answer the check tells apart, as it states in a few words:
    /// the first word that differs from Metal's, or a sweep's count of differing inputs and the
    /// least of them.
    NotMetals(String),
    /// The check errored, timed out or found an answer that is neither, which it states.
    Unanswered(String),
    /// The check does not run on the device, for the reason it states (a feature the fact needs
    /// is not enabled, so no reader takes the fact there).
    NotRun(String),
}

/// What a device's checks found
/// ([`DeviceDescription::checks`](crate::device::DeviceDescription::checks)): one answer per fact, in the
/// facts' order, and the facts the device learned off, each with what its record says. Empty for a
/// description no check ran for and no record named: every fact then takes its baseline.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DeviceChecks {
    answers: Vec<(DeviceFact, CheckAnswer)>,
    #[cfg_attr(feature = "serde", serde(default))]
    learned: Vec<(DeviceFact, String)>,
}

impl DeviceChecks {
    /// No answer.
    pub fn new() -> DeviceChecks {
        DeviceChecks::default()
    }

    /// Record `answer` for `fact`, in place of the answer it had: that answer.
    pub fn insert(&mut self, fact: DeviceFact, answer: CheckAnswer) -> Option<CheckAnswer> {
        match self.answers.binary_search_by(|(f, _)| f.cmp(&fact)) {
            Ok(at) => Some(std::mem::replace(&mut self.answers[at].1, answer)),
            Err(at) => {
                self.answers.insert(at, (fact, answer));
                None
            }
        }
    }

    /// The answer recorded for `fact`.
    pub fn get(&self, fact: DeviceFact) -> Option<&CheckAnswer> {
        let at = self.answers.binary_search_by(|(f, _)| f.cmp(&fact)).ok()?;
        Some(&self.answers[at].1)
    }

    /// Every fact and its answer, in the facts' order.
    pub fn iter(&self) -> impl Iterator<Item = (DeviceFact, &CheckAnswer)> {
        self.answers.iter().map(|(fact, answer)| (*fact, answer))
    }

    /// Whether no answer and no learned fact is recorded.
    pub fn is_empty(&self) -> bool {
        self.answers.is_empty() && self.learned.is_empty()
    }

    /// How many answers are recorded.
    pub fn len(&self) -> usize {
        self.answers.len()
    }

    /// Record `fact` as learned off, `why` being what its record says: the runtime read a record
    /// of a lost device while a submission that took the fact's direct path was in flight, so the
    /// fact takes its exact side from now on (the module docs).
    pub fn learn(&mut self, fact: DeviceFact, why: String) {
        match self.learned.binary_search_by(|(f, _)| f.cmp(&fact)) {
            Ok(at) => self.learned[at].1 = why,
            Err(at) => self.learned.insert(at, (fact, why)),
        }
    }

    /// What the record of `fact` says, where the device learned it off.
    pub fn learned(&self, fact: DeviceFact) -> Option<&str> {
        let at = self.learned.binary_search_by(|(f, _)| f.cmp(&fact)).ok()?;
        Some(&self.learned[at].1)
    }

    /// Every fact learned off, in the facts' order.
    pub fn learned_facts(&self) -> impl Iterator<Item = DeviceFact> + '_ {
        self.learned.iter().map(|(fact, _)| *fact)
    }
}

/// What a fact takes where its check is unanswered ([`CheckAnswer::Unanswered`]).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IfUnanswered {
    /// The exact path: no deviation or answer fact, a defect worked around.
    Exact,
    /// The baseline: the fact's workaround refuses work that the baseline runs where no row names
    /// the setup, so a check that could not answer leaves the fact as the tables have it.
    Baseline,
}

/// A fact the device decides by its own check ([`CHECKED`]).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CheckedFact {
    /// The fact.
    pub fact: DeviceFact,
    /// What it takes where its check is unanswered.
    pub if_unanswered: IfUnanswered,
}

/// Every fact the device decides by its own check, which the runtime's suite checks (its test
/// holds the two lists equal):
/// - `MetalHalfQuantize`: the device's `OpQuantizeToF16`, in translation's form, over the
///   overflow edge, every normal half exponent's ties and the specials, against Metal's
///   conversion to `half`.
/// - `RobustIndexFetchReadsZero`: an indexed indirect draw's fetch past its bound index range
///   under `robustBufferAccess2`.
/// - `DepthStencilResolveFirstSubresource`: a render pass's depth resolve of a second layer into
///   a second layer. Its workaround refuses a memoryless attachment at another layer, which a
///   driver no row names runs, so an unanswered check takes the baseline.
/// - The shader converter's defects `CompositeInsertsWrittenInPlace`, `LoadsForwardedIntoLoops`
///   and `LocalizedLoadsForwardedPastAtomicsAndCalls`, and the shader compiler's
///   `NarrowMaskAfterSignedClamp` and `HalfConversionPairFolded`: a kernel of the instructions
///   translation writes where the profile does not take the workaround, over inputs whose answer
///   the defect changes (`bind::composite_inserts_check` and its siblings). An unanswered check
///   takes the workaround.
/// - The color-write deviations `MetalColorWrites`, `MetalFloatColorNans`, `MetalHalfColorNans`,
///   `HalfColorOutputBits`, `MetalSmallFloatColorWrites` and `MetalNorm16ColorWrites`: draws of a
///   fragment that writes each pixel's input as computed into a target of each format the
///   deviation covers, plain and blended, against Metal's texels (`bind::color_write_draws`).
/// - The storage-write deviations `MetalHalfStorageWrites`, `MetalHalfStorageNans`,
///   `MetalFloatStorageNans`, `MetalHalfTexelWrites` and `MetalNormStorageWrites`: a kernel that
///   writes each texel's input as computed into a storage image of each format the deviation
///   covers, through each declaration translation gives it, against Metal's texels
///   (`bind::storage_writes`).
/// - The arithmetic deviations `MetalCanonicalNans` and `MetalFloatDenormals`: a kernel of the
///   device's own instructions over rows of operands, in each math mode the device has, against
///   Metal's results by its rule (`bind::canonical_nans_check`, `bind::float_denormals_check`).
/// - The block-compressed formats' decodes, `ColorBlockDecodeIsMetals` (BC1 to BC3),
///   `ChannelBlockDecodeIsMetals` (BC4 and BC5), `Bc6hBlockDecodeIsMetals`,
///   `Bc7BlockDecodeIsMetals` and `Bc7ReservedBlockIsMetals` (BC7's blocks of the reserved mode,
///   apart from its others): every texel of fixed random blocks of each unorm, snorm or float
///   format fetched, against Metal's reads of them (`bind::block_decode_cases`). An unanswered
///   check takes the exact path: a BC1 to BC5 texture reads a decoded twin, a BC6H or BC7 one
///   announces its decode (of a reserved block, where a host write holds one).
/// - `SampledFetchRecoversCodes`: every code of each normalized or 16-bit float format the mipmap
///   filter reads in place, fetched through a sampled view and recovered as the filter recovers
///   it (`bind::sampled_fetch_cases`). An unanswered check takes the exact path: the first level
///   is copied into the filter's scratch buffer.
/// - `CenterFragCoordDepth` and `SampleFragCoordDepth`, one draw: a per-sample fragment's
///   `FragCoord.z` and the sample depth translation computes from the window depth, against the
///   depth the device's own test stored at each sample of a flat, a sloped and a biased triangle.
/// - `MetalCenterVaryings`: a per-sample fragment's undecorated varyings, against the pixel
///   centre's value.
/// - `MemoryWritingFragmentsRunPerSample` and `ClippedTrianglesSplitPixelFragments`, one draw: the
///   invocations and coverage masks a memory-writing fragment reading its mask has at each pixel of
///   a triangle inside the clip volume and of two the device clips. The second not where the
///   device holds no `ClippedCoverageNotMetals`: its rasterizer is Apple's, whose pieces are
///   Metal's.
/// - `EarlyTestsDropMultisampledImageWrites`: a multisampled draw's early-tests fragment's storage
///   image writes beside its buffer writes.
/// - `BothBarycentricFormsRejected`: the driver's pipeline of a fragment reading both barycentric
///   builtins beside one reading the perspective one alone. Not on MoltenVK, whose converter logs
///   its refusal as an error: there MoltenVK's identity holds the defect at every release.
/// - `GraphicsCubeGradientRoute` and `GraphicsCubeGradientTangents`, one draw and one dispatch: a
///   fragment's cube samples with explicit gradients in each graphics route's form, against a
///   kernel's of the same lanes with the gradients as given, on a device whose kernels hand them to
///   Metal's own reading (`CubeGradientRoute`).
/// - `MetalDepth16Stores` and `MetalDepth16BiasedStores`, one draw: the codes the device stores in a
///   16-bit unorm depth for Metal's probes' depths and constant biases, against Metal's rule.
/// - `MultisampledCopiesConvertFloatNans`: a copy of 16-bit patterns, NaNs of both signs, quiet
///   and signalling, with and without payloads, beside numbers, from a multisampled integer image
///   into a multisampled `R16_SFLOAT` attachment of no other view format, as a NaN clear copies,
///   every sample copied back and fetched (`bind::texel_fetch_check_module`).
/// - The descriptor defects, each in the command order the device records:
///   `PushedBufferSizesUnset`, a pushed storage buffer's runtime array length after a bound set's
///   buffer of another length (`bind::array_length_check_module`);
///   `PushedDescriptorsStaleAfterPipelineBinds`, a pushed uniform buffer read by pipelines bound
///   after the push with no push after their bind, one at an offset of its `VertexIndex`
///   (`bind::pushed_uniform_check_module`); `ImmutableSamplersLostAfterCopies`, an immutable
///   clamp-to-edge sampler sampled outside `[0, 1]` after copies in the same command buffer
///   (`bind::sampler_clamp_check_module`).
/// - The deviations measured on every input of a 32-bit domain, `MetalHalfConversion`,
///   `MetalMinMax` and `MetalHalfArithmetic`: a module that computes the device's form and the
///   exact form of each input and counts those where they differ, over all 2^32 inputs in each
///   math mode the device has (`bind::Sweep`). Only the whole sweep answers Metal's.
/// - `OcclusionCountPerColorOutput`: a precise occlusion query of a draw whose fragment writes
///   three color outputs, beside one of one output (`bind::color_outputs_check_fragment`).
/// - `NormalizedDepthPastUnitRange`: a depth-clamped draw of a triangle whose normalized depth runs
///   from 0.5 to 2, as the compensation draws a viewport of `[0.5, 2]`, its samples below 1
///   against their plane's depth. Its workaround refuses a viewport depth range past `[0, 1]`,
///   which a driver no row names draws, so an unanswered check takes the baseline.
/// - `DepthTestKeepsDenormals`: depth tests at depth 0 against a `D32_SFLOAT` target's stored
///   denormals, beside the least normals: the positive ones, and the negative ones where the
///   device enables `VK_EXT_depth_range_unrestricted`, without which a copy may store no value
///   outside `[0, 1]` in a depth image. Not run where the device's copy does not keep them, nor,
///   where the positive ones test as Metal's, without the extension.
/// - `FragmentDepthStoresUnchanged` and `FragmentDepthKeepsDepth16Codes`: a per-sample draw that
///   writes chosen words as its `FragDepth` into a 4-sample depth
///   (`bind::depth_writes_check_fragment`), every sample fetched, then the fetched words written
///   again alike: signed zero, denormals, 1, values outside `[0, 1]`, infinities and NaNs into a
///   `D32_SFLOAT` depth, and every code of a `D16_UNORM` one. Their workaround refuses a
///   multisampled depth copy or resolve the draws make where no row names the driver, so an
///   unanswered check takes the baseline.
/// - `MultisampledDepthStencilColourCopies`: the depth aspect of a 4-sample
///   `D32_SFLOAT_S8_UINT` image, a different value at each sample, copied into a 4-sample
///   `R32_SFLOAT` image and back into a second such depth, every sample fetched. Not where the
///   known-driver list names the defect. Its workaround refuses the copies where the device's
///   draws cannot make them, so an unanswered check takes the baseline.
/// - `ColorResolveAverageIsMetals`: the device's `AVERAGE` resolve of random texels of each color
///   format with Metal's rule at 2 and 4 samples (`bind::ColorResolveFormat`), against the
///   crate's exact resolve of the same samples (`bind::color_resolve_modules`). It answers for
///   those formats alone: a format without such a rule takes the fact where the curated tables
///   hold it too. Not on a portability-subset device, whose views the exact resolve's reads need.
///
/// The refusals of the exact paths of the facts from `CenterFragCoordDepth` to
/// `BothBarycentricFormsRejected` (a multisampled in-shader depth test, a per-sample centre varying
/// without a centred offset read, the coverage loop's, an early-tests storage form, a fragment of
/// both barycentric forms) are work a driver no row names runs, so an unanswered check of any takes
/// the baseline.
///
/// An unanswered check of `MultisampledCopiesConvertFloatNans` or of a descriptor defect takes its
/// workaround, which refuses nothing.
pub const CHECKED: &[CheckedFact] = &[
    CheckedFact {
        fact: DeviceFact::Deviation(DriverDeviation::MetalHalfQuantize),
        if_unanswered: IfUnanswered::Exact,
    },
    CheckedFact {
        fact: DeviceFact::Fact(DriverFact::RobustIndexFetchReadsZero),
        if_unanswered: IfUnanswered::Exact,
    },
    CheckedFact {
        fact: DeviceFact::Defect(RuntimeDriverDefect::DepthStencilResolveFirstSubresource),
        if_unanswered: IfUnanswered::Baseline,
    },
    CheckedFact {
        fact: DeviceFact::Workaround(DriverWorkaround::CompositeInsertsWrittenInPlace),
        if_unanswered: IfUnanswered::Exact,
    },
    CheckedFact {
        fact: DeviceFact::Workaround(DriverWorkaround::LoadsForwardedIntoLoops),
        if_unanswered: IfUnanswered::Exact,
    },
    CheckedFact {
        fact: DeviceFact::Workaround(DriverWorkaround::LocalizedLoadsForwardedPastAtomicsAndCalls),
        if_unanswered: IfUnanswered::Exact,
    },
    CheckedFact {
        fact: DeviceFact::Workaround(DriverWorkaround::NarrowMaskAfterSignedClamp),
        if_unanswered: IfUnanswered::Exact,
    },
    CheckedFact {
        fact: DeviceFact::Workaround(DriverWorkaround::HalfConversionPairFolded),
        if_unanswered: IfUnanswered::Exact,
    },
    CheckedFact {
        fact: DeviceFact::Deviation(DriverDeviation::MetalColorWrites),
        if_unanswered: IfUnanswered::Exact,
    },
    CheckedFact {
        fact: DeviceFact::Deviation(DriverDeviation::MetalFloatColorNans),
        if_unanswered: IfUnanswered::Exact,
    },
    CheckedFact {
        fact: DeviceFact::Deviation(DriverDeviation::MetalHalfColorNans),
        if_unanswered: IfUnanswered::Exact,
    },
    CheckedFact {
        fact: DeviceFact::Deviation(DriverDeviation::HalfColorOutputBits),
        if_unanswered: IfUnanswered::Exact,
    },
    CheckedFact {
        fact: DeviceFact::Deviation(DriverDeviation::MetalSmallFloatColorWrites),
        if_unanswered: IfUnanswered::Exact,
    },
    CheckedFact {
        fact: DeviceFact::Deviation(DriverDeviation::MetalNorm16ColorWrites),
        if_unanswered: IfUnanswered::Exact,
    },
    CheckedFact {
        fact: DeviceFact::Deviation(DriverDeviation::MetalHalfStorageWrites),
        if_unanswered: IfUnanswered::Exact,
    },
    CheckedFact {
        fact: DeviceFact::Deviation(DriverDeviation::MetalHalfStorageNans),
        if_unanswered: IfUnanswered::Exact,
    },
    CheckedFact {
        fact: DeviceFact::Deviation(DriverDeviation::MetalFloatStorageNans),
        if_unanswered: IfUnanswered::Exact,
    },
    CheckedFact {
        fact: DeviceFact::Deviation(DriverDeviation::MetalHalfTexelWrites),
        if_unanswered: IfUnanswered::Exact,
    },
    CheckedFact {
        fact: DeviceFact::Deviation(DriverDeviation::MetalNormStorageWrites),
        if_unanswered: IfUnanswered::Exact,
    },
    CheckedFact {
        fact: DeviceFact::Deviation(DriverDeviation::MetalCanonicalNans),
        if_unanswered: IfUnanswered::Exact,
    },
    CheckedFact {
        fact: DeviceFact::Deviation(DriverDeviation::MetalFloatDenormals),
        if_unanswered: IfUnanswered::Exact,
    },
    CheckedFact {
        fact: DeviceFact::Fact(DriverFact::ColorBlockDecodeIsMetals),
        if_unanswered: IfUnanswered::Exact,
    },
    CheckedFact {
        fact: DeviceFact::Fact(DriverFact::ChannelBlockDecodeIsMetals),
        if_unanswered: IfUnanswered::Exact,
    },
    CheckedFact {
        fact: DeviceFact::Fact(DriverFact::Bc6hBlockDecodeIsMetals),
        if_unanswered: IfUnanswered::Exact,
    },
    CheckedFact {
        fact: DeviceFact::Fact(DriverFact::Bc7BlockDecodeIsMetals),
        if_unanswered: IfUnanswered::Exact,
    },
    CheckedFact {
        fact: DeviceFact::Fact(DriverFact::Bc7ReservedBlockIsMetals),
        if_unanswered: IfUnanswered::Exact,
    },
    CheckedFact {
        fact: DeviceFact::Fact(DriverFact::SampledFetchRecoversCodes),
        if_unanswered: IfUnanswered::Exact,
    },
    CheckedFact {
        fact: DeviceFact::Deviation(DriverDeviation::CenterFragCoordDepth),
        if_unanswered: IfUnanswered::Baseline,
    },
    CheckedFact {
        fact: DeviceFact::Deviation(DriverDeviation::SampleFragCoordDepth),
        if_unanswered: IfUnanswered::Baseline,
    },
    CheckedFact {
        fact: DeviceFact::Deviation(DriverDeviation::MetalCenterVaryings),
        if_unanswered: IfUnanswered::Baseline,
    },
    CheckedFact {
        fact: DeviceFact::Workaround(DriverWorkaround::MemoryWritingFragmentsRunPerSample),
        if_unanswered: IfUnanswered::Baseline,
    },
    CheckedFact {
        fact: DeviceFact::Workaround(DriverWorkaround::ClippedTrianglesSplitPixelFragments),
        if_unanswered: IfUnanswered::Baseline,
    },
    CheckedFact {
        fact: DeviceFact::Workaround(DriverWorkaround::EarlyTestsDropMultisampledImageWrites),
        if_unanswered: IfUnanswered::Baseline,
    },
    CheckedFact {
        fact: DeviceFact::Workaround(DriverWorkaround::BothBarycentricFormsRejected),
        if_unanswered: IfUnanswered::Baseline,
    },
    CheckedFact {
        fact: DeviceFact::Deviation(DriverDeviation::GraphicsCubeGradientRoute),
        if_unanswered: IfUnanswered::Exact,
    },
    CheckedFact {
        fact: DeviceFact::Deviation(DriverDeviation::GraphicsCubeGradientTangents),
        if_unanswered: IfUnanswered::Exact,
    },
    CheckedFact {
        fact: DeviceFact::Deviation(DriverDeviation::MetalDepth16Stores),
        if_unanswered: IfUnanswered::Exact,
    },
    CheckedFact {
        fact: DeviceFact::Deviation(DriverDeviation::MetalDepth16BiasedStores),
        if_unanswered: IfUnanswered::Exact,
    },
    CheckedFact {
        fact: DeviceFact::Defect(RuntimeDriverDefect::MultisampledCopiesConvertFloatNans),
        if_unanswered: IfUnanswered::Exact,
    },
    CheckedFact {
        fact: DeviceFact::Defect(RuntimeDriverDefect::PushedBufferSizesUnset),
        if_unanswered: IfUnanswered::Exact,
    },
    CheckedFact {
        fact: DeviceFact::Defect(RuntimeDriverDefect::PushedDescriptorsStaleAfterPipelineBinds),
        if_unanswered: IfUnanswered::Exact,
    },
    CheckedFact {
        fact: DeviceFact::Defect(RuntimeDriverDefect::ImmutableSamplersLostAfterCopies),
        if_unanswered: IfUnanswered::Exact,
    },
    CheckedFact {
        fact: DeviceFact::Deviation(DriverDeviation::MetalHalfConversion),
        if_unanswered: IfUnanswered::Exact,
    },
    CheckedFact {
        fact: DeviceFact::Deviation(DriverDeviation::MetalMinMax),
        if_unanswered: IfUnanswered::Exact,
    },
    CheckedFact {
        fact: DeviceFact::Deviation(DriverDeviation::MetalHalfArithmetic),
        if_unanswered: IfUnanswered::Exact,
    },
    CheckedFact {
        fact: DeviceFact::Defect(RuntimeDriverDefect::OcclusionCountPerColorOutput),
        if_unanswered: IfUnanswered::Exact,
    },
    CheckedFact {
        fact: DeviceFact::Defect(RuntimeDriverDefect::NormalizedDepthPastUnitRange),
        if_unanswered: IfUnanswered::Baseline,
    },
    CheckedFact {
        fact: DeviceFact::Fact(DriverFact::DepthTestKeepsDenormals),
        if_unanswered: IfUnanswered::Exact,
    },
    CheckedFact {
        fact: DeviceFact::Fact(DriverFact::FragmentDepthStoresUnchanged),
        if_unanswered: IfUnanswered::Baseline,
    },
    CheckedFact {
        fact: DeviceFact::Fact(DriverFact::FragmentDepthKeepsDepth16Codes),
        if_unanswered: IfUnanswered::Baseline,
    },
    CheckedFact {
        fact: DeviceFact::Defect(RuntimeDriverDefect::MultisampledDepthStencilColourCopies),
        if_unanswered: IfUnanswered::Baseline,
    },
    CheckedFact {
        fact: DeviceFact::Fact(DriverFact::ColorResolveAverageIsMetals),
        if_unanswered: IfUnanswered::Exact,
    },
];

/// The entry of [`CHECKED`] for `fact`, where the device checks it.
pub fn checked(fact: DeviceFact) -> Option<CheckedFact> {
    CHECKED.iter().copied().find(|checked| checked.fact == fact)
}

/// How the device decided one fact ([`FactStatus::decided`]).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Decided {
    /// Its check answered Metal's on the device.
    CheckedMetals,
    /// Its check answered otherwise on the device.
    CheckedNotMetals,
    /// A row of the curated tables names the setup: the row, as `driver, releases, GPU`.
    Listed(String),
    /// The curated tables give it to every setup of a kind (a driver by its identity, every
    /// driver but one rasterizer's), no measured row naming this setup: the rule, as text. A
    /// check that answers otherwise overturns an assumption, not a measurement.
    Assumed(String),
    /// No row of the curated tables names the setup: the exact path for an answer, Vulkan's own
    /// for a defect.
    Unlisted,
    /// The exact path, for the reason stated: an answer the device does not take, or a defect it
    /// works around.
    Exact(String),
    /// The exact path, learned: the device was lost while a submission that took the direct path
    /// was in flight, as its record states ([`DeviceChecks::learn`]).
    LearnedOff(String),
}

/// What became of a fact's check on the device ([`FactStatus::check`]).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum CheckOf {
    /// The fact has no check ([`CHECKED`]): the tables decide it.
    None,
    /// Its check has not answered on the device yet.
    Pending,
    /// The policy distrusts checks: none ran or was read.
    Distrusted,
    /// Its check's answer, run on the device or read from its cache.
    Answered(CheckAnswer),
    /// Its check answered on the running device after the device was made, which keeps the fact
    /// as it was made (translation reads it in the compile profile): the answer is the setup's
    /// next device's, from the device-checks cache.
    NextDevice(CheckAnswer),
}

/// How the device takes one fact: a row of its feature status.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FactStatus {
    /// The fact.
    pub fact: DeviceFact,
    /// Whether the device takes it: a deviation or an answer fact taken, a defect worked around.
    pub holds: bool,
    /// How that was decided.
    pub decided: Decided,
    /// What became of its check.
    pub check: CheckOf,
}

/// A fact as the curated tables alone have it ([`decide`]'s baseline).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Baseline {
    /// Whether the tables hold the fact on the setup.
    pub holds: bool,
    /// How: [`Decided::Listed`], [`Decided::Assumed`], [`Decided::Unlisted`], or
    /// [`Decided::Exact`] for an answer an unmeasured identity withholds.
    pub decided: Decided,
}

impl Baseline {
    /// Held by the row `entry` names.
    pub fn listed(entry: String) -> Baseline {
        Baseline {
            holds: true,
            decided: Decided::Listed(entry),
        }
    }

    /// Held by the rule `entry` names for every setup of its kind, no measured row naming this one.
    pub fn assumed(entry: String) -> Baseline {
        Baseline {
            holds: true,
            decided: Decided::Assumed(entry),
        }
    }

    /// Named by no row.
    pub fn unlisted() -> Baseline {
        Baseline {
            holds: false,
            decided: Decided::Unlisted,
        }
    }

    /// Not held, for the reason `why`.
    pub fn withheld(why: String) -> Baseline {
        Baseline {
            holds: false,
            decided: Decided::Exact(why),
        }
    }
}

/// `fact` as the device takes it, by the reader order (the module docs): its check's answer in
/// `checks` (`None` where the policy distrusts checks), else its `baseline`, and then the exact
/// side where `checks` holds it learned off and those left it on the direct one.
pub fn decide(fact: DeviceFact, baseline: Baseline, checks: Option<&DeviceChecks>) -> FactStatus {
    let status = checked_or_baseline(fact, baseline, checks);
    let exact = fact.is_defect();
    match checks.and_then(|checks| checks.learned(fact)) {
        Some(why) if status.holds != exact => FactStatus {
            holds: exact,
            decided: Decided::LearnedOff(why.to_owned()),
            ..status
        },
        _ => status,
    }
}

/// `fact` by its check's answer in `checks`, else its `baseline` ([`decide`]'s first steps).
fn checked_or_baseline(
    fact: DeviceFact,
    baseline: Baseline,
    checks: Option<&DeviceChecks>,
) -> FactStatus {
    let of_baseline = |check: CheckOf| FactStatus {
        fact,
        holds: baseline.holds,
        decided: baseline.decided.clone(),
        check,
    };
    let Some(entry) = checked(fact) else {
        return of_baseline(CheckOf::None);
    };
    let Some(checks) = checks else {
        return of_baseline(CheckOf::Distrusted);
    };
    let Some(answer) = checks.get(fact) else {
        return of_baseline(CheckOf::Pending);
    };
    let check = CheckOf::Answered(answer.clone());
    let (holds, decided) = match answer {
        CheckAnswer::Metals => (!fact.is_defect(), Decided::CheckedMetals),
        CheckAnswer::NotMetals(_) => (fact.is_defect(), Decided::CheckedNotMetals),
        CheckAnswer::Unanswered(why) => match entry.if_unanswered {
            IfUnanswered::Exact => (
                fact.is_defect(),
                Decided::Exact(format!("its check is unanswered: {why}")),
            ),
            IfUnanswered::Baseline => return of_baseline(check),
        },
        CheckAnswer::NotRun(_) => return of_baseline(check),
    };
    FactStatus {
        fact,
        holds,
        decided,
        check,
    }
}
