//! `SamplerState` (`MTLSamplerState`) and `Device::new_sampler_state`.
//!
//! A sampler state is its descriptor (the compile layer's [`SamplerDescriptor`], Metal's
//! properties) and the `VkSampler`s its binds need. One Metal sampler is several Vulkan ones: how
//! a function samples through it (with depth comparison or not) and the texture it meets (an
//! integer one is never filtered) decide fields Metal keeps out of the descriptor
//! (`bind::sampler_desc`), and a bind's LOD clamps (`set_sampler_state_with_lod_clamps`) replace
//! the descriptor's. Each distinct Vulkan sampler is made at its first bind and kept by the state,
//! at most [`KEPT`] of them: past that the least recently bound one is forgotten, and the command
//! buffers that bound it keep it until they complete.
//!
//! **Metal's rules** (probe 7: an Apple M3 Max on macOS 26.7 and an M4 on 27.0.1, each case run
//! without and with API validation). Metal's API validation stops, at
//! `newSamplerStateWithDescriptor:`:
//!
//! - a `maxAnisotropy` outside 1 to 16;
//! - a pixel-coordinate sampler (`normalizedCoordinates` false) with a `maxAnisotropy` other than
//!   1, a mip filter other than `NotMipmapped`, or an address mode other than `ClampToEdge`,
//!   `ClampToZero` and `ClampToBorderColor` on any axis. Its two filters may differ, and its LOD
//!   clamps, comparison and `lodAverage` are free.
//!
//! Plain Metal makes both, and so does the runtime ([`taken_descriptor`]). A `maxAnisotropy` of 1
//! to 16 samples as that value rounded up to a power of two: 3 as 4, 5 to 7 as 8, 9 to 15 as 16
//! (an M4 on macOS 27.0.1: every value through runtime and constexpr samplers at 1, 268 footprints
//! of ratios 1 to 32, each answer bit for bit the power of two's, the powers told apart at 388 to
//! 1, 191 of them). A `maxAnisotropy` outside 1 to 16 samples as the low five bits of the value
//! taken through a 32-bit float, saturated to 32 bits, 0 as 1 and 17 to 31 as 32, a value of 1 to
//! 16 rounded up so ([`metal_anisotropy`]; an M4 on macOS 27.0.1: 0, 17 to 2^64 - 1, each sampling
//! as one of 1, 2, 4, 8, 16 and 32 at gradients that tell those apart, footprints up to 256 to 1,
//! two runs alike). An anisotropy past the device's largest (`maxSamplerAnisotropy`, 16 on every
//! venue) filters with that largest, a near-equivalent announced once
//! (`Degradation::CappedAnisotropy`). A pixel-coordinate sampler Metal makes and samples its repeat
//! and mirrored-repeat axes as `ClampToEdge`, at level 0 whatever its mip filter, LOD or gradients,
//! and without anisotropy (an M4 on macOS 27.0.1: nearest and linear filters at 22 points inside,
//! on and past the edges, every answer the `ClampToEdge` sampler's): so does the runtime
//! ([`taken_descriptor`]). A `MirrorClampToEdge` `s` or `t` axis it mirrors once about 0 and then
//! clamps, by texel index: each tap's index `i`, a constant texel offset added first (but dropped
//! where the base level is 3, 5, 6 or 7 texels wide, as on a repeating axis), reads texel `-1 - i`
//! where it is negative, then clamped into the level, as a normalized sampler's axis does (an M4 on
//! macOS 27.0.1: nearest and linear samples, gathers and depth comparisons of textures 1 to 8
//! texels wide, offsets -7 to 5, two runs alike): so does the runtime, keeping the mode, whose
//! Vulkan sampler mirrors once and clamps (`MIRROR_CLAMP_TO_EDGE`) at the coordinate the function
//! divides by the texture's size, the function placing the offset and filtering linearly as Metal
//! does. Address modes that hold both `ClampToZero` and `ClampToBorderColor`, which Metal's
//! validation layer stops where the border color is not transparent black, plain Metal makes, and
//! it reads the border color on every border axis: so does the runtime, as it reads a `ClampToZero`
//! axis's border on any sampler. Metal takes any LOD clamps, at creation and in a bind (inverted,
//! NaN and infinite ones included): so does the runtime. Metal raises a sample's level to the lower
//! clamp and then lowers it to the upper one, so inverted clamps read the upper clamp's level,
//! filtering as clamps of the upper clamp alone do, a descriptor's and a bind's alike: so does the
//! runtime ([`metal_lod_clamps`]; an M4 on macOS 27.0.1, and the M3 Max on macOS 26.7). A sampler
//! whose clamps where it is bound are NaN is refused where it is used, by the binding plan, since
//! what Metal samples then is not measured. Metal takes any raw border color and keeps its low two
//! bits, sampling 3 as transparent black on every format measured ([`sampler_border_color`], the
//! same probes). What the device cannot do with the state on any use (a mirror-clamp address mode
//! it lacks) is refused here, `Unsupported` ([`check_sampler`]). Anisotropy on a device without
//! `samplerAnisotropy` is made: its Vulkan samplers are isotropic, and a function samples through
//! the state with Metal's anisotropic filter emulated where the emulation reaches the sample
//! (`Degradation::AnisotropicFilter`) and the device's isotropic filter elsewhere
//! (`Degradation::DeviceAnisotropicFilter`), each announced; an argument buffer that lists it is
//! refused where it is used, `Unsupported` (`samplerAnisotropy`), since a function reading it
//! there knows not its anisotropy. A minimum or maximum reduction reads the weighted average, as
//! on an Apple M4 without Metal's validation layer (which refuses one), and
//! `DeviceProperties::supports_sampler_reduction_mode` answers no.
//!
//! A min or mag filter `Cubic` (`MTLSamplerMinMagFilterCubic`, raw value 2) is made. Without a
//! mip filter, Metal samples through such a state as through a constexpr `filter::bicubic`
//! sampler of its address modes, border color and coordinates, whatever its other filter (our
//! probe, the M3 Max and an M4 alike): a pipeline samples through it on its cubic-sampler variant,
//! which emulates that filter. With a mip filter, `MirrorClampToEdge`, a minimum or maximum
//! reduction, anisotropy, or through an argument buffer, the dispatch or draw that samples through
//! it is refused `NotImplemented(SamplerBicubicFilter)`: Metal's level rule through it is not
//! modelled, and the others are not measured.

use super::degradation::Degradation;
use super::device::lock;
use super::error::{Error, SamplerLimit};
use super::exit_guard;
use super::resource_ids::{ElementHold, Holder, IdHold, IdPlacement, IdSpace, ResourceId};
use super::Device;
use crate::release;
use ash::vk;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::device::VkFeature;
use metal2vulkan::error::DriverDeviation;
use metal2vulkan::metal::sampler::{check_sampler, SamplerUse};
use metal2vulkan::metal::{
    CompareFunction, SamplerAddressMode, SamplerBorderColor, SamplerMinMagFilter, SamplerMipFilter,
    SamplerReductionMode,
};
use std::any::Any;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

/// How a sampler state is made beyond its descriptor: the rest of `MTLSamplerDescriptor`.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SamplerStateOptions {
    /// `supportArgumentBuffers`. It decides only whether two requests share a state (probe 5):
    /// every state has a `gpuResourceID`, which an argument buffer can hold to name it, and
    /// Metal samples through the ID of a state without the support too (probe 7).
    pub support_argument_buffers: bool,
    /// `resourceIndex` with `forceResourceIndex` (Apple's private properties, a runtime
    /// extension): where the state goes in the space of sampler resource IDs
    /// (`resource_ids.rs`). A request equal to a live state's gets that state, its ID whatever
    /// this asks, as Metal's does (probe 5).
    pub placement: IdPlacement,
}

/// What decides whether two requests share one state (probe 5, an M3 Max and an M4: each
/// descriptor field changed alone, and `supportArgumentBuffers`, gave another state; the
/// resource index never did): every field of the descriptor, its LOD clamps by their bits, and
/// argument-buffer support. Metal's label decides too; the runtime's states have none.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct SamplerKey {
    filters: (SamplerMinMagFilter, SamplerMinMagFilter, SamplerMipFilter),
    address_modes: [SamplerAddressMode; 3],
    border_color: SamplerBorderColor,
    normalized_coordinates: bool,
    lod_clamps: [u32; 2],
    lod_average: bool,
    max_anisotropy: u32,
    compare_function: CompareFunction,
    reduction_mode: SamplerReductionMode,
    support_argument_buffers: bool,
}

impl SamplerKey {
    fn of(descriptor: &SamplerDescriptor, options: SamplerStateOptions) -> SamplerKey {
        let d = descriptor;
        SamplerKey {
            filters: (d.min_filter, d.mag_filter, d.mip_filter),
            address_modes: [d.s_address_mode, d.t_address_mode, d.r_address_mode],
            border_color: d.border_color,
            normalized_coordinates: d.normalized_coordinates,
            lod_clamps: [d.lod_min_clamp.to_bits(), d.lod_max_clamp.to_bits()],
            lod_average: d.lod_average,
            max_anisotropy: d.max_anisotropy,
            compare_function: d.compare_function,
            reduction_mode: d.reduction_mode,
            support_argument_buffers: options.support_argument_buffers,
        }
    }
}

/// The most Vulkan samplers a sampler state keeps.
const KEPT: usize = 16;

/// The device's live Vulkan samplers, at most `maxSamplerAllocationCount`: every sampler the
/// runtime makes (a state's, a pipeline layout's immutable one, a neutral one) is counted from its
/// creation to its destruction, and one past the limit is refused, `Error::SamplerLimit`, rather
/// than made, which Vulkan does not allow. A state's samplers are at most [`KEPT`] in its cache
/// plus those the command buffers that bound them still hold; one per view mapping (an opaque-black
/// border on a device without `borderColorSwizzleFromImage`) is one more such sampler, so a
/// workload cycling views is held to the device's limit too.
pub(super) struct SamplerCount {
    live: AtomicU32,
    limit: u32,
}

impl SamplerCount {
    /// No sampler live, of `limit`.
    pub(super) fn new(limit: u32) -> SamplerCount {
        SamplerCount {
            live: AtomicU32::new(0),
            limit,
        }
    }

    /// Count one more live sampler, or refuse it where `limit` are live.
    pub(super) fn take(&self) -> Result<(), Error> {
        self.live
            .try_update(Ordering::AcqRel, Ordering::Acquire, |live| {
                (live < self.limit).then_some(live + 1)
            })
            .map(|_| ())
            .map_err(|_| Error::SamplerLimit(SamplerLimit { limit: self.limit }))
    }

    /// One counted sampler was destroyed.
    pub(super) fn give(&self) {
        self.live.fetch_sub(1, Ordering::AcqRel);
    }

    /// The samplers live now (the device tests).
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn live(&self) -> u32 {
        self.live.load(Ordering::Acquire)
    }
}

/// The entry of a full cache of samplers to forget, `kept` least recently bound first: the first
/// that only the cache holds, whose sampler is destroyed as it goes, else the first, which the
/// command buffers that bound it keep until they complete. No sampler a pending command buffer
/// holds is destroyed.
pub(super) fn forgotten<K, T>(kept: &[(K, Arc<T>)]) -> usize {
    kept.iter()
        .position(|(_, sampler)| Arc::strong_count(sampler) == 1)
        .unwrap_or(0)
}

/// Metal's largest `maxAnisotropy`.
const MAX_ANISOTROPY: u32 = 16;

/// The anisotropy plain Metal samples a `maxAnisotropy` of 17 to 31 with (at
/// footprints of 32, 64 and 256 to 1 each reads the level an anisotropy of 32 reads, exactly).
const PAST_RANGE_ANISOTROPY: u32 = 32;

/// The anisotropy plain Metal samples a `maxAnisotropy` of `value` with: from 1 to 16, `value`
/// rounded up to a power of two (3 as 4, 5 to 7 as 8, 9 to 15 as 16); outside, the low five bits of
/// `value` taken through a 32-bit float (rounded to nearest) and saturated to 32 bits, 0 as 1 and
/// 17 to 31 as 32, and 1 to 16 rounded up as before (the module's rules). A Metal value past
/// `u32::MAX` takes the float's saturation too, as `u32::MAX` does: a caller may saturate it.
pub(super) fn metal_anisotropy(value: u32) -> u32 {
    // Rust's `as` rounds to nearest and saturates, as the measured conversion does.
    let low = ((value as f32) as u32) & 31;
    match low {
        0 => 1,
        1..=MAX_ANISOTROPY => low.next_power_of_two(),
        _ => PAST_RANGE_ANISOTROPY,
    }
}

/// The LOD clamps plain Metal samples with for clamps `[min, max]`, a descriptor's or a bind's
/// (the module's rules): Metal raises a sample's level to `min` and then lowers it to `max`, so
/// where `min` is past `max` every sample reads `max`'s level, and magnifies or minifies as at
/// `max`: the clamps `[max, max]`, which Vulkan takes (its `minLod` may not pass its `maxLod`).
/// NaN clamps are kept, for the binding plan to refuse where they are used.
pub(super) fn metal_lod_clamps([min, max]: [f32; 2]) -> [f32; 2] {
    if min > max {
        [max, max]
    } else {
        [min, max]
    }
}

/// The border color plain Metal samples through a sampler state whose `MTLSamplerDescriptor`'s
/// `borderColor` is Metal's raw value `raw`. Metal takes any value, its validation layer
/// included, and keeps its low two bits (the descriptor reads back 0 for 4 and 3 for 7); it
/// samples 3, which names no color, as transparent black (samples, linear blends at the edge and
/// depth comparisons of eight formats, unorm, float, integer and depth, every answer border 0's).
/// `SamplerBorderColor`'s `TryFrom<u32>` takes Metal's three named values alone.
pub fn sampler_border_color(raw: u64) -> SamplerBorderColor {
    match raw & 3 {
        1 => SamplerBorderColor::OpaqueBlack,
        2 => SamplerBorderColor::OpaqueWhite,
        _ => SamplerBorderColor::TransparentBlack,
    }
}

/// The descriptor a state of `descriptor` samples with (the module's rules): `descriptor` itself
/// with plain Metal's anisotropy ([`metal_anisotropy`]) and LOD clamps ([`metal_lod_clamps`]),
/// and for a pixel-coordinate one its repeat and mirrored-repeat axes `ClampToEdge`, its `r`
/// axis's `MirrorClampToEdge` too (`r` addresses nothing such a sampler samples), mip filter
/// `NotMipmapped` and anisotropy 1, as plain Metal samples it. A pixel-coordinate
/// `MirrorClampToEdge` `s` or `t` axis keeps its mode, which Metal mirrors once by texel index and
/// then clamps.
pub(super) fn taken_descriptor(descriptor: &SamplerDescriptor) -> SamplerDescriptor {
    use SamplerAddressMode as A;
    let mut taken = *descriptor;
    taken.max_anisotropy = metal_anisotropy(descriptor.max_anisotropy);
    [taken.lod_min_clamp, taken.lod_max_clamp] =
        metal_lod_clamps([descriptor.lod_min_clamp, descriptor.lod_max_clamp]);
    if !descriptor.normalized_coordinates {
        for mode in [&mut taken.s_address_mode, &mut taken.t_address_mode] {
            if matches!(*mode, A::Repeat | A::MirrorRepeat) {
                *mode = A::ClampToEdge;
            }
        }
        // `r` addresses nothing a pixel-coordinate sampler samples (1D and 2D textures).
        if matches!(
            taken.r_address_mode,
            A::Repeat | A::MirrorRepeat | A::MirrorClampToEdge
        ) {
            taken.r_address_mode = A::ClampToEdge;
        }
        taken.mip_filter = SamplerMipFilter::NotMipmapped;
        taken.max_anisotropy = 1;
    }
    taken
}

/// A `VkSampler` the runtime made, destroyed when the last holder lets it go.
pub(super) struct OwnedSampler {
    device: Device,
    raw: vk::Sampler,
}

impl OwnedSampler {
    /// The `VkSampler`.
    pub(super) fn raw(&self) -> vk::Sampler {
        self.raw
    }
}

impl Drop for OwnedSampler {
    fn drop(&mut self) {
        // Once the process is exiting, the sampler is left to its end (`exit_guard`).
        let Some(_busy) = exit_guard::enter() else {
            return;
        };
        // SAFETY: no pending command buffer uses the sampler: each one that bound it holds it
        // until it completes.
        unsafe { release::object(self.device.shared().vk(), self.raw) };
        self.device.shared().sampler_destroyed();
    }
}

/// A Metal sampler state (`MTLSamplerState`). Cheap to clone (a retain); `Send + Sync`. Two
/// handles are equal when they are the same state.
#[derive(Clone)]
pub struct SamplerState(Arc<SamplerShared>);

pub(super) struct SamplerShared {
    device: Device,
    /// The descriptor the state samples with ([`taken_descriptor`]).
    descriptor: SamplerDescriptor,
    /// The state's `gpuResourceID`.
    id: IdHold,
    /// The state's element of the set-1 sampler arrays (`resource_ids.rs`).
    element: ElementHold,
    /// The Vulkan samplers made for binds, least recently bound first.
    kept: Mutex<Vec<(metal2vulkan::vk::SamplerDesc, Arc<OwnedSampler>)>>,
}

impl std::fmt::Debug for SamplerState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SamplerState")
            .field("descriptor", &self.0.descriptor)
            .finish()
    }
}

impl PartialEq for SamplerState {
    fn eq(&self, other: &SamplerState) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for SamplerState {}

impl std::hash::Hash for SamplerState {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.0).hash(state);
    }
}

impl Device {
    /// `newSamplerStateWithDescriptor:`. A request equal to a live state's (every field of its
    /// descriptor, and its options but the placement) gets that state, as Metal's does.
    ///
    /// A descriptor Metal's validation stops, of a `max_anisotropy` outside 1 to 16 or of pixel
    /// coordinates, is made, and samples as plain Metal samples it (the module's rules); an
    /// anisotropy past the largest of a device with `samplerAnisotropy` filters with that
    /// largest, announced once (`Degradation::CappedAnisotropy`). An anisotropic state without a
    /// mip filter whose two filters differ, on a device that samples through Metal's own sampler
    /// (MoltenVK), filters another footprint than Metal's, announced once
    /// (`Degradation::NotMipmappedAnisotropicSampler`). Anisotropy on a device without it is made,
    /// and samples as the module's docs say. Refused: what the device cannot do on
    /// any use of the state (`Unsupported`, naming the feature: a mirror-clamp address mode, a
    /// pixel-coordinate one's included). A minimum or maximum reduction reads the weighted
    /// average.
    pub fn new_sampler_state(&self, descriptor: &SamplerDescriptor) -> Result<SamplerState, Error> {
        self.sampler_state(
            "new_sampler_state",
            descriptor,
            SamplerStateOptions::default(),
        )
    }

    /// `newSamplerStateWithDescriptor:` with the rest of Metal's descriptor in `options`. Every
    /// state has a [`gpu_resource_id`](SamplerState::gpu_resource_id), which an argument buffer
    /// holds to name it, and every function an encoder runs can name it so while it lives. A
    /// state at an ID no set-1 sampler array of the device can hold is made, and announced:
    /// a function reading its ID reads the neutral sampler (`resource_ids.rs`). Refused as
    /// [`new_sampler_state`](Self::new_sampler_state) is, and a placement at an ID of 16381 or
    /// more (`Invalid`, `Problem::ResourceId`), which Metal ends the process for.
    pub fn new_sampler_state_with_options(
        &self,
        descriptor: &SamplerDescriptor,
        options: SamplerStateOptions,
    ) -> Result<SamplerState, Error> {
        self.sampler_state("new_sampler_state_with_options", descriptor, options)
    }

    fn sampler_state(
        &self,
        call: &'static str,
        descriptor: &SamplerDescriptor,
        options: SamplerStateOptions,
    ) -> Result<SamplerState, Error> {
        let taken = taken_descriptor(descriptor);
        // The LOD clamps are checked where they are used: a bind's clamps replace them.
        let mut shape_of = taken;
        shape_of.lod_min_clamp = 0.0;
        shape_of.lod_max_clamp = f32::MAX;
        let profile = self.compile_device().profile();
        check_sampler(&shape_of.shape(), SamplerUse::default(), profile).map_err(|error| {
            match error {
                metal2vulkan::error::Error::Unsupported(unsupported) => {
                    Error::Unsupported(unsupported)
                }
                other => super::error::internal(&format!("a sampler check refused {other:?}")),
            }
        })?;
        self.check_sampler_placement(call, options.placement)?;
        // Metal samples an anisotropy past 16 with 32 (`metal_anisotropy`): a device whose
        // samplers take less filters with its largest (`bind::sampler_desc` bounds it so). A
        // device without `samplerAnisotropy` filters with none, Metal's filter emulated where the
        // function reaches the sample and the isotropic one recorded elsewhere.
        let largest = f32::from_bits(profile.limits().max_sampler_anisotropy_bits);
        let anisotropy = (profile.enabled().features).contains(&VkFeature::SamplerAnisotropy);
        if anisotropy && taken.max_anisotropy as f32 > largest {
            self.degrade(Degradation::CappedAnisotropy, || {
                (
                    format!(
                        "a sampler state of maxAnisotropy {}",
                        descriptor.max_anisotropy
                    ),
                    format!(
                        "plain Metal samples it with an anisotropy of {}, and the device's \
                         samplers take at most {largest} (maxSamplerAnisotropy)",
                        taken.max_anisotropy
                    ),
                )
            });
        }
        // Metal's own sampler handed anisotropy reads no level clamp of a sampler without a mip
        // filter whose filters differ as Metal's `NotMipmapped` (`metal::sampler::vk_sampler`).
        // A pixel-coordinate state has no anisotropy (`taken_descriptor`).
        if anisotropy
            && taken.max_anisotropy > 1
            && taken.mip_filter == SamplerMipFilter::NotMipmapped
            && taken.min_filter != taken.mag_filter
            && profile.takes_driver_deviation(DriverDeviation::MetalSampler)
        {
            self.degrade(Degradation::NotMipmappedAnisotropicSampler, || {
                (
                    format!(
                        "a sampler state of maxAnisotropy {} without a mip filter, minification \
                         {:?} and magnification {:?}",
                        descriptor.max_anisotropy, taken.min_filter, taken.mag_filter
                    ),
                    "the device samples through Metal's own sampler, whose anisotropic filter \
                     reads no level clamp as Metal's NotMipmapped where the filters differ"
                        .to_owned(),
                )
            });
        }
        let shared = self.shared();
        let key = SamplerKey::of(descriptor, options);
        // Under the map's lock, so equal requests on two threads get one state.
        let mut live = lock(&shared.samplers);
        if let Some(state) = live.get(&key).and_then(std::sync::Weak::upgrade) {
            return Ok(SamplerState(state));
        }
        let id = IdHold::take(&shared.resource_ids, IdSpace::Sampler, options.placement);
        let element = ElementHold::take(&shared.resource_ids);
        let state = SamplerState(Arc::new(SamplerShared {
            device: self.clone(),
            descriptor: taken,
            id,
            element,
            kept: Mutex::new(Vec::new()),
        }));
        live.retain(|_, state| state.strong_count() > 0);
        live.insert(key, Arc::downgrade(&state.0));
        let hold = &state.0.id;
        // Sampler slots have no readers: the device counts the command buffers that list them.
        hold.hold(Holder::Sampler(Arc::downgrade(&state.0)));
        drop(live);
        let id = hold.id().0;
        let element = state.heap_element();
        let (available, limit) = self.sampler_array();
        if element >= available {
            self.degrade(Degradation::UnlistedArgumentSampler, || {
                (
                    format!("sampler state at resource ID {id}"),
                    format!(
                        "more sampler states live than a set-1 sampler array of the device \
                         holds ({limit:?} is {available}), so none holds its element {element}, \
                         and a command buffer whose commands list the argument samplers while it \
                         lives fails (CommandBufferError::UnlistedArgumentSampler)"
                    ),
                )
            });
        }
        if shared.sampler_tables.load(Ordering::Acquire) > 0 {
            self.degrade(Degradation::ResourceIdTakeover, || {
                (
                    format!("sampler state made at resource ID {id}"),
                    "a committed command buffer that has not completed lists the argument \
                     samplers as it committed, which this state is not among"
                        .to_owned(),
                )
            });
        }
        Ok(state)
    }

    /// The live sampler states, in resource-ID order, each the holder of its ID
    /// (`resource_ids.rs`): what an argument buffer can name (`BindState::argument_samplers`).
    pub(super) fn argument_samplers(&self) -> Vec<SamplerState> {
        self.argument_samplers_at().1
    }

    /// [`argument_samplers`](Self::argument_samplers), with the count of the sampler IDs' slots'
    /// changes they are live at ([`sampler_slots_changed`](Self::sampler_slots_changed)).
    fn argument_samplers_at(&self) -> (u64, Vec<SamplerState>) {
        let (changed, states) = lock(&self.shared().resource_ids).samplers();
        (changed, states.into_iter().map(SamplerState).collect())
    }

    /// [`argument_samplers_at`](Self::argument_samplers_at), for a draw's listing: counted
    /// (`WorkCounts::argument_samplers_taken`).
    fn argument_samplers_for_draw(&self) -> (u64, Vec<SamplerState>) {
        self.shared().counts.argument_samplers_taken();
        self.argument_samplers_at()
    }

    /// The count of the changes of the sampler IDs' slots so far: while it holds, the live
    /// sampler states are the ones [`argument_samplers`](Self::argument_samplers) answered at it.
    pub(super) fn sampler_slots_changed(&self) -> u64 {
        self.shared()
            .sampler_ids
            .slots_changed
            .load(Ordering::Acquire)
    }
}

/// The device's live sampler states a draw lists (`BindState::argument_samplers`), each at the
/// heap element of its resource ID, taken from the device only where the draw needs them
/// (`RenderCommandEncoder::settle_draw`): a kept bind state listed at the count of the sampler
/// IDs' slots' changes that still holds lists them already (`LiveBind::listed_at`), and a draw
/// that writes no heap element and names no state in a notice takes none.
pub(super) struct ArgumentSamplers {
    /// The states and the count of slot changes they were live at, once taken.
    taken: Option<(u64, Vec<SamplerState>)>,
    /// Those the draw's variant's set-1 sampler arrays leave out were dropped
    /// (`heap_sets::fit_samplers`): the states are no longer every live one.
    trimmed: bool,
}

impl ArgumentSamplers {
    /// None taken yet.
    pub(super) fn untaken() -> ArgumentSamplers {
        ArgumentSamplers {
            taken: None,
            trimmed: false,
        }
    }

    /// `device`'s, taken now.
    pub(super) fn taken(device: &Device) -> ArgumentSamplers {
        ArgumentSamplers {
            taken: Some(device.argument_samplers_for_draw()),
            trimmed: false,
        }
    }

    /// The states, taken from `device` now where none were yet.
    pub(super) fn take(&mut self, device: &Device) -> &mut Vec<SamplerState> {
        let (_, states) = self
            .taken
            .get_or_insert_with(|| device.argument_samplers_for_draw());
        states
    }

    /// The states, where they were taken.
    pub(super) fn states(&self) -> Option<&[SamplerState]> {
        self.taken.as_ref().map(|(_, states)| &states[..])
    }

    /// The count of slot changes every live state is listed at: where the states were taken
    /// and not trimmed.
    pub(super) fn listed_at(&self) -> Option<u64> {
        match (&self.taken, self.trimmed) {
            (Some((changed, _)), false) => Some(*changed),
            _ => None,
        }
    }

    /// Keep those of the taken states that `set`, the variant's set 1, holds
    /// (`heap_sets::fit_samplers`, `unlisted` saying so): whether every one fits.
    pub(super) fn fit(
        &mut self,
        set: Option<&metal2vulkan::pipeline::DescriptorSetLayout>,
        unlisted: &mut Option<(u64, u64)>,
    ) -> bool {
        let Some((_, states)) = &mut self.taken else {
            return set.is_none();
        };
        let fits = super::heap_sets::fit_samplers(set, states, unlisted);
        self.trimmed |= !fits;
        fits
    }
}

impl SamplerState {
    /// Another reference to the sampler state, which a command buffer holds it by
    /// (`EncoderCore::retain_shared`).
    pub(super) fn held(&self) -> Arc<dyn Any + Send + Sync> {
        self.0.clone()
    }

    /// The sampler state's address: no other object's while it lives.
    pub(super) fn address(&self) -> usize {
        Arc::as_ptr(&self.0) as usize
    }

    /// `device`: the device that made the state.
    pub fn device(&self) -> &Device {
        &self.0.device
    }

    /// `gpuResourceID`: what an argument buffer holds to name the state (`resource_ids.rs`).
    pub fn gpu_resource_id(&self) -> ResourceId {
        self.0.id.id()
    }

    /// The state's element of the set-1 sampler arrays, which a sampler table maps its ID to
    /// (`resource_ids.rs`).
    pub(super) fn heap_element(&self) -> u64 {
        self.0.element.element()
    }

    /// The descriptor the state samples with: its request's as plain Metal takes it
    /// ([`taken_descriptor`]).
    pub(super) fn descriptor(&self) -> &SamplerDescriptor {
        &self.0.descriptor
    }

    /// Whether an access through an argument buffer that names this state may hand the device a
    /// pixel coordinate scaled by the texture's size (`Device::degrade_argument_pixel_samplers`):
    /// its coordinates are pixels, and it filters (a minification or magnification filter other
    /// than nearest) or compares. A nearest pixel state's sample reads Metal's texel, by its
    /// flagged fetch or Metal's own sampler.
    pub(super) fn scales_in_an_argument_buffer(&self) -> bool {
        let d = &self.0.descriptor;
        !d.normalized_coordinates
            && (d.min_filter != SamplerMinMagFilter::Nearest
                || d.mag_filter != SamplerMinMagFilter::Nearest
                || d.compare_function != CompareFunction::Never)
    }

    /// The Vulkan sampler of `desc` (one of this state's binds, its view mapping included): the
    /// kept one, else a new one, kept in place of the one [`forgotten`] names when the state keeps
    /// [`KEPT`]; refused, `Error::SamplerLimit`, where the device's samplers are at its limit
    /// ([`SamplerCount`]).
    pub(super) fn vk_sampler(
        &self,
        desc: &metal2vulkan::vk::SamplerDesc,
    ) -> Result<Arc<OwnedSampler>, Error> {
        let mut kept = lock(&self.0.kept);
        if let Some(at) = kept.iter().position(|(d, _)| d == desc) {
            let entry = kept.remove(at);
            let sampler = Arc::clone(&entry.1);
            kept.push(entry);
            return Ok(sampler);
        }
        if kept.len() == KEPT {
            let at = forgotten(&kept);
            kept.remove(at);
        }
        let mut reduction = vk::SamplerReductionModeCreateInfo::default();
        let mut border = vk::SamplerBorderColorComponentMappingCreateInfoEXT::default();
        let info =
            metal2vulkan::ash_interop::sampler_create_info(desc, &mut reduction, &mut border);
        let device = &self.0.device;
        // SAFETY: the create info is valid for the device: `check_sampler` admitted its
        // features, its LOD range is ordered (`bind::sampler_desc`), and a border mapping is
        // chained only where the device has `borderColorSwizzle`
        // (`metal::sampler::opaque_black_border`).
        let raw = unsafe { device.shared().create_sampler(&info) }?;
        let sampler = Arc::new(OwnedSampler {
            device: device.clone(),
            raw,
        });
        kept.push((*desc, Arc::clone(&sampler)));
        Ok(sampler)
    }
}
