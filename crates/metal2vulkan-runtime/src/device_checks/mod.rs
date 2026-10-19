//! The device's own checks of what its driver answers (`metal2vulkan::device::checks`): each fact
//! of the suite is checked on the device by a short, bounded dispatch, draw or copy, read back and
//! compared with Metal's answer, and the answer goes to the description the compile device is made
//! from ([`DeviceDescription::checks`]), where every reader takes it before the curated tables.
//!
//! - **When.** A check of [`Settle::AtCreation`] (a fact that decides a refusal or shapes a
//!   layout) runs as the device is made, between `vkCreateDevice` and the device's first object,
//!   on the creating thread, through the device's primary queue (an adopted device's queue, which
//!   the consumer does not use meanwhile). One of [`Settle::Background`] runs after the making on
//!   the device's checks thread (`background.rs`), so a making waits for none of them; until its
//!   answer is in, its fact holds as the device was made (its baseline). A runtime fact then
//!   switches to the answer, and a translation fact keeps it for the setup's next device
//!   (`live.rs`). [`DeviceChecksOptions::wait`] runs every check at the making instead. A check
//!   that errors, times out (2 s) or reads an answer that is neither takes the exact path, and its
//!   status says why; after a time-out no later check runs.
//! - **The cache.** The answers are kept per setup in the device-checks directory
//!   ([`DeviceChecksOptions::directory`], by default the platform's per-user cache directory's
//!   `metal2vulkan-device-checks`), keyed by the driver, GPU, release, `pipelineCacheUUID`, the
//!   OS build, the suite's version, the translator's and the runtime's source digests and the
//!   compile device's profile before any check (`cache.rs`).
//!   A device of a setup whose answers are kept reads them and runs no check; a corrupt or
//!   unreadable entry is checked again. An unanswered check is never kept. The making writes the
//!   answers it found, and the checks thread each answer as its check settles, so a device dropped
//!   before its thread answered the last keeps every answer the thread finished, and the next
//!   making runs only the checks left. Each write merges into the entry.
//! - **Distrusted.** Under `DeviceChecksPolicy::Distrusted` no check runs and none is read: every
//!   fact is as the curated tables have it.
//! - **Learned.** A path in use as the device was lost is off on the setup from the next device on
//!   (`learned.rs`): its record in the same directory is read here, before any check, unless the
//!   policy distrusts checks.
//! - **The status.** What the device found and how it takes each fact is its feature status
//!   ([`FeatureStatus`], [`DeviceProperties::feature_status`] as made,
//!   [`Device::feature_status`](crate::Device::feature_status) as it stands), which the device
//!   announces once as it is made in an informational message, `metal2vulkan-feature-status`, and
//!   once its checks thread answered the last check left to it,
//!   `metal2vulkan-feature-status-settled` (`status.rs`).
//!
//! Each check's shader is a module the translator builds itself
//! (`metal2vulkan::bind::half_quantize_check`, `index_fetch_check_module`, the kernels of the
//! shader converter's and compiler's defects, `converter.rs` and `compiler.rs`, and the raster
//! checks' `raster_check_vertex` with their fragment stages and kernels), and its expected answer a
//! constant of Metal's, a closed rule of Metal's (`metal::state::depth16_code`) or the device's own
//! consistency (a fragment's depth against the depth its test stored, a fragment's cube sample
//! against a kernel's); the color writes' (`color.rs`) are draws of the translator's modules into
//! targets of each format, and the storage writes' (`storage.rs`) dispatches of its kernels into
//! storage images of each format, against Metal's texels by its rule. The arithmetic checks
//! (`arith.rs`) run once in each math mode the device has, and the sweeps (`sweep.rs`) over every
//! input of a 32-bit domain in each, one chunk a submission. Nothing a guest sees (the device's
//! Metal properties, families, limits, layouts) depends on an answer.
//!
//! [`DeviceDescription::checks`]: metal2vulkan::device::DeviceDescription::checks
//! [`DeviceProperties::feature_status`]: crate::DeviceProperties::feature_status

mod arith;
pub(crate) mod background;
mod barycentric;
mod block_decode;
pub(crate) mod cache;
mod center_varyings;
mod color;
mod colour_copies;
mod compiler;
mod converter;
mod coverage;
mod cube_gradients;
mod depth16;
mod depth_denormals;
mod depth_draws;
mod depth_range;
mod depth_resolve;
mod early_writes;
mod fetched;
mod fragcoord_depth;
mod fragment_depth;
mod half_quantize;
mod immutable_sampler;
mod index_fetch;
pub(crate) mod kit;
pub(crate) mod learned;
pub(crate) mod live;
mod nan_copy;
mod occlusion;
mod pushed_sizes;
mod pushed_stale;
mod raster;
mod resolve_average;
mod sampled_fetch;
mod status;
mod storage;
mod sweep;

pub(crate) use status::{status_text, ChecksRun, FEATURE_STATUS, FEATURE_STATUS_SETTLED};
pub use status::{CheckCost, CheckRun, CheckSource, FeatureStatus, Learning, Settle, Settled};

use crate::loader_word::LoaderWord;
use ash::vk;
use metal2vulkan::device::{
    CheckAnswer, DeviceChecks, DeviceChecksPolicy, DeviceDescription, DeviceFact, DevicePolicy,
    DriverFact, DriverWorkaround, EnableSet,
};
use metal2vulkan::error::{DriverDeviation, RuntimeDriverDefect};
use std::path::PathBuf;
use std::time::Instant;

/// Where a device keeps what its checks of its driver found, and whether its making waits for
/// every check. No Metal counterpart: Metal's driver is Metal's own.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DeviceChecksOptions {
    /// The directory the answers are kept in, one file per setup under its `v1`: by default the
    /// platform's per-user cache directory's `metal2vulkan-device-checks` (`~/Library/Caches` on
    /// macOS, else `$XDG_CACHE_HOME` or `~/.cache`), beside the shader cache's `metal2vulkan`.
    /// Where no directory can be used the checks run at every making, and the feature status says
    /// why.
    pub directory: Option<PathBuf>,
    /// Run every check as the device is made, those that would settle on its checks thread
    /// ([`Settle::Background`]) included: every fact is as its check found it from the device's
    /// first object, and no thread runs checks beside the program's work. For a test or a consumer
    /// that needs the same answers from the start of every run; a making whose setup is not in the
    /// cache then takes the checks' whole time. By default (`false`) the making runs only the
    /// checks of [`Settle::AtCreation`].
    pub wait: bool,
}

/// The version of the suite: part of every cache key, raised whenever a check, its inputs or its
/// expected answer change, so no answer of an earlier suite is read.
pub(crate) const CHECK_SUITE_VERSION: u32 = 24;

/// One check of the suite: one program on the device that decides one fact or more.
pub(crate) struct Check {
    /// The facts it decides, in the order of its answers.
    pub(crate) facts: &'static [DeviceFact],
    /// When its answers must be in.
    pub(crate) settle: Settle,
    /// Run it on the device: one answer per fact of [`facts`](Self::facts), in their order (a
    /// fact it gives no answer for is unanswered).
    pub(crate) run: fn(&mut kit::Kit<'_>, &Target<'_>) -> Vec<CheckAnswer>,
}

/// Every check the runtime runs: each fact of `metal2vulkan::device::CHECKED` decided by exactly
/// one, in that order (a test holds the lists equal).
pub(crate) const SUITE: &[Check] = &[
    Check {
        facts: &[DeviceFact::Deviation(DriverDeviation::MetalHalfQuantize)],
        settle: Settle::Background,
        run: half_quantize::run,
    },
    Check {
        facts: &[DeviceFact::Fact(DriverFact::RobustIndexFetchReadsZero)],
        settle: Settle::Background,
        run: index_fetch::run,
    },
    Check {
        facts: &[DeviceFact::Defect(
            RuntimeDriverDefect::DepthStencilResolveFirstSubresource,
        )],
        // Its workaround refuses a memoryless attachment at another layer: decided before the
        // device's first object.
        settle: Settle::AtCreation,
        run: depth_resolve::run,
    },
    Check {
        facts: &[
            DeviceFact::Workaround(DriverWorkaround::CompositeInsertsWrittenInPlace),
            DeviceFact::Workaround(DriverWorkaround::LoadsForwardedIntoLoops),
            DeviceFact::Workaround(DriverWorkaround::LocalizedLoadsForwardedPastAtomicsAndCalls),
        ],
        settle: Settle::Background,
        run: converter::run,
    },
    Check {
        facts: &[
            DeviceFact::Workaround(DriverWorkaround::NarrowMaskAfterSignedClamp),
            DeviceFact::Workaround(DriverWorkaround::HalfConversionPairFolded),
        ],
        settle: Settle::Background,
        run: compiler::run,
    },
    Check {
        facts: &[
            DeviceFact::Deviation(DriverDeviation::MetalColorWrites),
            DeviceFact::Deviation(DriverDeviation::MetalFloatColorNans),
            DeviceFact::Deviation(DriverDeviation::MetalHalfColorNans),
            DeviceFact::Deviation(DriverDeviation::HalfColorOutputBits),
            DeviceFact::Deviation(DriverDeviation::MetalSmallFloatColorWrites),
            DeviceFact::Deviation(DriverDeviation::MetalNorm16ColorWrites),
        ],
        settle: Settle::Background,
        run: color::run,
    },
    Check {
        facts: &[
            DeviceFact::Deviation(DriverDeviation::MetalHalfStorageWrites),
            DeviceFact::Deviation(DriverDeviation::MetalHalfStorageNans),
            DeviceFact::Deviation(DriverDeviation::MetalFloatStorageNans),
            DeviceFact::Deviation(DriverDeviation::MetalHalfTexelWrites),
            DeviceFact::Deviation(DriverDeviation::MetalNormStorageWrites),
        ],
        settle: Settle::Background,
        run: storage::run,
    },
    Check {
        facts: &[
            DeviceFact::Deviation(DriverDeviation::MetalCanonicalNans),
            DeviceFact::Deviation(DriverDeviation::MetalFloatDenormals),
        ],
        settle: Settle::Background,
        run: arith::run,
    },
    Check {
        facts: &[
            DeviceFact::Fact(block_decode::FACTS[0]),
            DeviceFact::Fact(block_decode::FACTS[1]),
            DeviceFact::Fact(block_decode::FACTS[2]),
            DeviceFact::Fact(block_decode::FACTS[3]),
            DeviceFact::Fact(block_decode::FACTS[4]),
        ],
        settle: Settle::Background,
        run: block_decode::run,
    },
    Check {
        facts: &[DeviceFact::Fact(DriverFact::SampledFetchRecoversCodes)],
        settle: Settle::Background,
        run: sampled_fetch::run,
    },
    Check {
        facts: &[
            DeviceFact::Deviation(DriverDeviation::CenterFragCoordDepth),
            DeviceFact::Deviation(DriverDeviation::SampleFragCoordDepth),
        ],
        settle: Settle::AtCreation,
        run: fragcoord_depth::run,
    },
    Check {
        facts: &[DeviceFact::Deviation(DriverDeviation::MetalCenterVaryings)],
        settle: Settle::AtCreation,
        run: center_varyings::run,
    },
    Check {
        facts: &[
            DeviceFact::Workaround(DriverWorkaround::MemoryWritingFragmentsRunPerSample),
            DeviceFact::Workaround(DriverWorkaround::ClippedTrianglesSplitPixelFragments),
        ],
        settle: Settle::AtCreation,
        run: coverage::run,
    },
    Check {
        facts: &[DeviceFact::Workaround(
            DriverWorkaround::EarlyTestsDropMultisampledImageWrites,
        )],
        settle: Settle::AtCreation,
        run: early_writes::run,
    },
    Check {
        facts: &[DeviceFact::Workaround(
            DriverWorkaround::BothBarycentricFormsRejected,
        )],
        settle: Settle::AtCreation,
        run: barycentric::run,
    },
    Check {
        facts: &[
            DeviceFact::Deviation(DriverDeviation::GraphicsCubeGradientRoute),
            DeviceFact::Deviation(DriverDeviation::GraphicsCubeGradientTangents),
        ],
        settle: Settle::Background,
        run: cube_gradients::run,
    },
    Check {
        facts: &[
            DeviceFact::Deviation(DriverDeviation::MetalDepth16Stores),
            DeviceFact::Deviation(DriverDeviation::MetalDepth16BiasedStores),
        ],
        settle: Settle::Background,
        run: depth16::run,
    },
    Check {
        facts: &[DeviceFact::Defect(
            RuntimeDriverDefect::MultisampledCopiesConvertFloatNans,
        )],
        settle: Settle::Background,
        run: nan_copy::run,
    },
    Check {
        facts: &[DeviceFact::Defect(
            RuntimeDriverDefect::PushedBufferSizesUnset,
        )],
        // It decides whether the device takes the push command, so every pipeline layout's
        // kind: decided before the device's first object, and before the checks that push as
        // the device does (`kit::Kit::push_command`) run as it is made.
        settle: Settle::AtCreation,
        run: pushed_sizes::run,
    },
    Check {
        facts: &[DeviceFact::Defect(
            RuntimeDriverDefect::PushedDescriptorsStaleAfterPipelineBinds,
        )],
        settle: Settle::Background,
        run: pushed_stale::run,
    },
    Check {
        facts: &[DeviceFact::Defect(
            RuntimeDriverDefect::ImmutableSamplersLostAfterCopies,
        )],
        settle: Settle::Background,
        run: immutable_sampler::run,
    },
    Check {
        facts: &[
            DeviceFact::Deviation(DriverDeviation::MetalHalfConversion),
            DeviceFact::Deviation(DriverDeviation::MetalMinMax),
            DeviceFact::Deviation(DriverDeviation::MetalHalfArithmetic),
        ],
        settle: Settle::Background,
        run: sweep::run,
    },
    Check {
        facts: &[DeviceFact::Defect(
            RuntimeDriverDefect::OcclusionCountPerColorOutput,
        )],
        settle: Settle::Background,
        run: occlusion::run,
    },
    Check {
        facts: &[DeviceFact::Defect(
            RuntimeDriverDefect::NormalizedDepthPastUnitRange,
        )],
        // It decides whether `set_viewport` refuses a depth range past [0, 1]: decided before
        // the device's first object.
        settle: Settle::AtCreation,
        run: depth_range::run,
    },
    Check {
        facts: &[DeviceFact::Fact(DriverFact::DepthTestKeepsDenormals)],
        // The texture state reads it as the device is assembled (the depth writes' notices, the
        // magnitude stand-in's switch): decided before the device's first object.
        settle: Settle::AtCreation,
        run: depth_denormals::run,
    },
    Check {
        facts: &[
            DeviceFact::Fact(DriverFact::FragmentDepthStoresUnchanged),
            DeviceFact::Fact(DriverFact::FragmentDepthKeepsDepth16Codes),
        ],
        // Each decides whether a multisampled depth copy or resolve the runtime draws is
        // refused, read as the device is assembled: decided before the device's first object.
        settle: Settle::AtCreation,
        run: fragment_depth::run,
    },
    Check {
        facts: &[DeviceFact::Defect(
            RuntimeDriverDefect::MultisampledDepthStencilColourCopies,
        )],
        // It decides whether a multisampled depth or stencil copy goes by colour copies, by
        // draws or is refused, read as the device is assembled: decided before the device's
        // first object.
        settle: Settle::AtCreation,
        run: colour_copies::run,
    },
    Check {
        facts: &[DeviceFact::Fact(DriverFact::ColorResolveAverageIsMetals)],
        settle: Settle::Background,
        run: resolve_average::run,
    },
];

/// The device as it is made, which the checks of its making run on.
pub(crate) struct Making<'a> {
    pub(crate) instance: &'a ash::Instance,
    pub(crate) device: &'a ash::Device,
    /// The device's loader word, put back before each loader entry point (`loader_word.rs`).
    pub(crate) loader_word: &'a LoaderWord,
    /// The queue the checks submit to: its family and index.
    pub(crate) queue: (u32, u32),
}

/// What a check reads of its device, as it is made or running.
pub(crate) struct Target<'a> {
    pub(crate) physical: vk::PhysicalDevice,
    /// What the device enables, as the compile device was made from it (no check's answer yet).
    pub(crate) description: &'a DeviceDescription,
    pub(crate) enabled: &'a EnableSet,
    /// The compile device made from `description` before any check: the checks' modules are its.
    pub(crate) compile: &'a metal2vulkan::device::Device,
}

/// What a device's checks found as it was made, and how it came by them.
pub(crate) struct Settlement {
    /// The answers, for the description the device is made from.
    pub(crate) checks: DeviceChecks,
    /// How they were found, for the feature status.
    pub(crate) run: status::ChecksRun,
    /// Where a record of its lost device goes (`learned.rs`): none where no directory is used.
    pub(crate) learned: Option<learned::Target>,
    /// The checks left to the device's checks thread, where any are.
    pub(crate) pending: Option<background::Pending>,
}

/// Run `check` on the kit `kit` holds, made by `make` at the first check that runs, for `target`,
/// its answers into `checks`, which hold the answers so far as the device is made
/// ([`kit::Kit::answered`]); where they came from.
fn run_check<'a>(
    kit: &mut Option<Result<kit::Kit<'a>, String>>,
    make: impl FnOnce() -> Result<kit::Kit<'a>, String>,
    check: &Check,
    target: &Target<'_>,
    checks: &mut DeviceChecks,
) -> CheckSource {
    let facts = check.facts.len();
    let (answers, source) = match kit.get_or_insert_with(make) {
        Err(why) => (
            vec![CheckAnswer::Unanswered(format!("no check could run: {why}")); facts],
            CheckSource::Skipped,
        ),
        Ok(kit) if kit.timed_out() => (
            vec![CheckAnswer::Unanswered("an earlier check timed out".to_owned()); facts],
            CheckSource::Skipped,
        ),
        Ok(kit) => {
            kit.answered(target.description, checks);
            let began = kit.begin_check();
            let answers = (check.run)(kit, target);
            (answers, CheckSource::Ran(kit.end_check(began)))
        }
    };
    let given = answers.len();
    let mut answers = answers.into_iter();
    for &fact in check.facts {
        let answer = answers.next().unwrap_or_else(|| {
            CheckAnswer::Unanswered(format!("the check gave {given} answers for {facts} facts"))
        });
        checks.insert(fact, answer);
    }
    source
}

/// The checks of `target`, made as `making`, under `policy`: from `options`' cache where it keeps
/// them, else run on the device now, or left to its checks thread (the module docs).
///
/// # Safety
///
/// `making`'s handles are live: `device` was made on `target.physical` from `instance` with
/// `target.enabled` enabled and a queue at `queue`, which nothing else uses until this returns,
/// and `loader_word` is the device's.
pub(crate) unsafe fn settle(
    making: &Making<'_>,
    target: &Target<'_>,
    options: &DeviceChecksOptions,
    policy: DevicePolicy,
) -> Settlement {
    let started = Instant::now();
    // SAFETY: the caller's.
    let key = unsafe { cache::Key::of(making.instance, target, policy) };
    let mut run = status::ChecksRun::new(key.hex());
    let place = cache::Place::of(options);
    let mut checks = DeviceChecks::new();
    // What the setup learned from a lost device: its records' place even where the policy
    // distrusts checks, since a loss is recorded all the same, and what they say where it does not.
    let (learned, learning) = {
        let place = place.as_ref().ok().map(cache::Place::dir);
        learned::settle(target.description, place, policy, &mut checks)
    };
    run.learning = learning;
    if policy.device_checks == DeviceChecksPolicy::Distrusted {
        run.settled = Settled::Distrusted;
        return Settlement {
            checks,
            run,
            learned,
            pending: None,
        };
    }
    let kept = match &place {
        Ok(place) => place.read(&key),
        Err(why) => {
            run.cache_error = Some(why.clone());
            None
        }
    };
    // SAFETY: the caller's.
    let facts = unsafe { kit::DeviceFacts::of(making.instance, target.physical, making.queue) };
    let mut kit = None;
    let mut pending = Vec::new();
    for check in SUITE {
        if let Some(answers) = kept.as_ref().and_then(|entry| entry.answers_of(check)) {
            for (&fact, answer) in check.facts.iter().zip(answers) {
                checks.insert(fact, answer);
            }
            run.push(check, CheckSource::Cache);
            continue;
        }
        if check.settle == Settle::Background && !options.wait {
            run.push(check, CheckSource::Pending);
            pending.push(check);
            continue;
        }
        let make = || {
            // SAFETY: the caller's.
            unsafe {
                kit::Kit::making(
                    making.instance,
                    making.device,
                    making.loader_word,
                    target.enabled,
                    facts,
                )
            }
        };
        let source = run_check(&mut kit, make, check, target, &mut checks);
        run.push(check, source);
    }
    // Every object of the checks is gone before the device's own are made.
    drop(kit);
    let ran = run.checks_run > 0;
    let at_creation = started.elapsed();
    run.settled = match (pending.is_empty(), ran, &kept) {
        (false, _, _) => Settled::Running { at_creation },
        (true, false, Some(entry)) => Settled::FromCache {
            checked_at: entry.checked_at,
        },
        (true, _, _) => Settled::AtCreation { wall: at_creation },
    };
    if let (true, Ok(place)) = (ran, &place) {
        if let Err(why) = place.write(&key, &checks) {
            run.cache_error = Some(why);
        }
    }
    let pending = (!pending.is_empty()).then(|| background::Pending {
        checks: pending,
        key,
        place,
        answers: checks.clone(),
        description: target.description.clone(),
        enabled: target.enabled.clone(),
        compile: target.compile.clone(),
        facts,
        at_creation,
    });
    Settlement {
        checks,
        run,
        learned,
        pending,
    }
}
