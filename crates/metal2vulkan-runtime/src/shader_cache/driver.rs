//! The Vulkan driver's pipeline cache data: one record per pipeline making, each making with a
//! `VkPipelineCache` of its own (it replaced the one entry per device and the caches per variant).
//!
//! - **One making, one cache, one record.** Every `vkCreate*Pipelines` call of the runtime
//!   (`ShaderCache::make_pipelines`) is made with a `VkPipelineCache` of its own: made from the
//!   making's record where one loads, else empty, and destroyed after the call. Where the making
//!   added to the data (no record, or its pipeline not in it), the data is written as the record;
//!   where the data is the record's, byte for byte, nothing is written. A boot reads the records
//!   of the pipelines it makes, and nothing else.
//! - **Named by what the making reads** ([`Making`]). A record is named by the SHA-256 of
//!   [`VERSION`], the making's description, the driver's identity (vendor, device, driver
//!   version, `pipelineCacheUUID`) and the digest of the layer settings the instance was created
//!   with; and by the device's profile fingerprint (`store::entry_name`). A program's pipeline is
//!   described by its variant's stages (SPIR-V words, entry points, stage flags, required subgroup
//!   sizes, specialization), its layout (its immutable samplers among it), and the state its
//!   making adds (a graphics key; a compute pipeline's specialization and flags); a helper of the
//!   runtime's by its SPIR-V and its parameters. Records live in the content directory, which no
//!   build owns (`store.rs`): a new build of the runtime or the translator that leaves a
//!   pipeline's SPIR-V and state as they were finds its record. The driver keys each pipeline by
//!   everything its creation reads, so a description that covers a making loosely costs a
//!   compile, never a wrong pipeline.
//! - **No figure of the driver's to meet.** A record holds one making's data (an Xgc pipeline's
//!   1.7 MB on an RTX 3090), so how much of a large cache's initial data a
//!   driver keeps does not decide which pipelines are kept. The shader cache's cap and its scans,
//!   the least recently used first (`store.rs`), do: a hit refreshes a record's access time.
//! - **One making's libraries.** A driver that compiles every shader of a cache's initial data
//!   when the cache is made (`RuntimeDriverDefect::PipelineCacheDataCompiledAtLoad`, MoltenVK)
//!   compiles from a record the making's own stages alone, which the making converts anyway without
//!   one; host steps and pool pieces load and save records as the program's threads do.
//! - **Makings at once.** A record is written to a temporary file renamed over it
//!   (`Store::write_record`), so a reader reads the whole old record or the whole new one, never a
//!   mix. Two processes, or two threads, making the same pipeline at once each write that
//!   pipeline's data from the same driver under the same name: the same bytes, or either one is
//!   fine.
//! - **Cold on purpose.** A program that wants every pipeline compiled (a cold boot measured as
//!   one) names an empty directory in `ShaderCacheOptions::directory`: nothing else of the
//!   runtime's holds the driver's data. A driver's own shader disk cache is the driver's to turn
//!   off (NVIDIA's `__GL_SHADER_DISK_CACHE=0`, Mesa's `MESA_SHADER_CACHE_DISABLE=true`).
//!
//! ```text
//! magic        8 bytes   "M2VPPLC\0"
//! identity     28 bytes  vendorID, deviceID, driverVersion (u32 each), pipelineCacheUUID
//! data length  u64
//! data         vkGetPipelineCacheData's bytes
//! checksum     32 bytes  SHA-256 of every byte before it
//! ```
//!
//! Integers are little-endian. A load takes the data only if the identity equals the device's,
//! the checksum holds, and the data's own header (`VkPipelineCacheHeaderVersionOne`) names the
//! same vendor, device and UUID; anything else is refused and the record removed before the
//! driver sees it. A record the driver refuses is counted, and the making starts empty. A save
//! checks the driver's data the same way and keeps nothing it would refuse to load, counting the
//! reason ([`Refusal`]): some drivers return data that ends inside the header it states, at
//! `VK_SUCCESS`.

use super::store::{self, count, entry_name, Lookup, Store};
use super::EntryKind;
use crate::exit_guard::{self, Call};
use crate::release;
use ash::vk;
use metal2vulkan::cache::Hash256;
use sha2::{Digest, Sha256};
use std::ops::Range;

const MAGIC: [u8; 8] = *b"M2VPPLC\0";
/// The version of what a record's name covers and how a record is framed: a change to either
/// bumps it, so no build opens a record another build names or frames otherwise.
pub(crate) const VERSION: u32 = 1;
/// The identity's bytes.
const IDENTITY: usize = 4 + 4 + 4 + 16;
/// Bytes before the data.
const HEADER: usize = 8 + IDENTITY + 8;
/// Bytes after the data.
const CHECKSUM: usize = 32;
/// `sizeof(VkPipelineCacheHeaderVersionOne)`.
const VULKAN_HEADER: usize = 32;

/// Why the driver's pipeline cache data is not kept.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Refusal {
    /// The data ends before the end of the header it states, or of a version-one header.
    Truncated,
    /// The header is smaller than a version-one header, or its version, vendor, device or UUID
    /// is not this device's.
    Mismatched,
}

/// A Vulkan device's identity for its pipeline cache data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Identity([u8; IDENTITY]);

impl Identity {
    /// The identity of a device of `properties`.
    pub(super) fn of(properties: &vk::PhysicalDeviceProperties) -> Identity {
        Identity::new(
            properties.vendor_id,
            properties.device_id,
            properties.driver_version,
            properties.pipeline_cache_uuid,
        )
    }

    /// The identity of these values.
    pub(crate) fn new(vendor: u32, device: u32, driver: u32, uuid: [u8; 16]) -> Identity {
        let mut bytes = [0; IDENTITY];
        bytes[..4].copy_from_slice(&vendor.to_le_bytes());
        bytes[4..8].copy_from_slice(&device.to_le_bytes());
        bytes[8..12].copy_from_slice(&driver.to_le_bytes());
        bytes[12..].copy_from_slice(&uuid);
        Identity(bytes)
    }

    /// Whether `data` begins with the `VkPipelineCacheHeaderVersionOne` of this identity, whole:
    /// its size, its version, then vendor, device and UUID (not the driver version, which it does
    /// not hold).
    pub(crate) fn check_vulkan_header(&self, data: &[u8]) -> Result<(), Refusal> {
        let word = |at: usize| {
            data.get(at..at + 4)
                .and_then(|bytes| bytes.try_into().ok())
                .map(u32::from_le_bytes)
        };
        let size = word(0)
            .and_then(|size| usize::try_from(size).ok())
            .ok_or(Refusal::Truncated)?;
        if size < VULKAN_HEADER {
            return Err(Refusal::Mismatched);
        }
        if data.len() < size {
            return Err(Refusal::Truncated);
        }
        let ours = word(4) == Some(vk::PipelineCacheHeaderVersion::ONE.as_raw() as u32)
            && data[8..12] == self.0[..4]
            && data[12..16] == self.0[4..8]
            && data[16..32] == self.0[12..];
        if ours {
            Ok(())
        } else {
            Err(Refusal::Mismatched)
        }
    }
}

/// `data` framed for `identity`.
pub(crate) fn seal(identity: &Identity, data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEADER + data.len() + CHECKSUM);
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&identity.0);
    out.extend_from_slice(&(data.len() as u64).to_le_bytes());
    out.extend_from_slice(data);
    let checksum: Hash256 = Sha256::digest(&out).into();
    out.extend_from_slice(&checksum);
    out
}

/// Where the data of a frame for `identity` lies in `bytes`, or `None` when the frame or the
/// data's own header does not hold.
pub(crate) fn open(identity: &Identity, bytes: &[u8]) -> Option<Range<usize>> {
    if bytes.get(..8) != Some(&MAGIC[..]) || bytes.get(8..8 + IDENTITY) != Some(&identity.0[..]) {
        return None;
    }
    let length = bytes
        .get(HEADER - 8..HEADER)?
        .try_into()
        .ok()
        .map(u64::from_le_bytes)?;
    let end = usize::try_from(length).ok()?.checked_add(HEADER)?;
    if bytes.len() != end.checked_add(CHECKSUM)? {
        return None;
    }
    let checksum: Hash256 = Sha256::digest(&bytes[..end]).into();
    if bytes[end..] != checksum {
        return None;
    }
    identity
        .check_vulkan_header(&bytes[HEADER..end])
        .is_ok()
        .then_some(HEADER..end)
}

/// What a making reads, which names its record (the module docs): a tag naming the code that
/// makes it, then the parts its caller adds, in order.
pub(crate) struct Making(Sha256);

impl Making {
    /// A making by the code `kind` names.
    pub(crate) fn new(kind: &str) -> Making {
        let mut hash = Sha256::new();
        hash.update(b"metal2vulkan pipeline making\0");
        hash.update(VERSION.to_le_bytes());
        Making(hash).part(kind.as_bytes())
    }

    /// And `bytes`, after their length.
    fn part(mut self, bytes: &[u8]) -> Making {
        self.0.update((bytes.len() as u64).to_le_bytes());
        self.0.update(bytes);
        self
    }

    /// And the SPIR-V `words`.
    pub(crate) fn words(mut self, words: &[u32]) -> Making {
        self.0.update(((words.len() as u64) * 4).to_le_bytes());
        let mut chunk = [0u8; 4096];
        for words in words.chunks(chunk.len() / 4) {
            for (at, word) in words.iter().enumerate() {
                chunk[at * 4..at * 4 + 4].copy_from_slice(&word.to_le_bytes());
            }
            self.0.update(&chunk[..words.len() * 4]);
        }
        self
    }

    /// And `value`'s `Debug` text: equal values give equal text, and a build that prints them
    /// otherwise names other records, which costs a compile.
    pub(crate) fn text(self, value: &dyn std::fmt::Debug) -> Making {
        self.part(format!("{value:?}").as_bytes())
    }

    /// And `digest`, of parts described as these are.
    pub(crate) fn digest(self, digest: &Hash256) -> Making {
        self.part(digest)
    }

    /// The digest of the parts so far, to describe the makings that share them at once.
    pub(crate) fn finish(self) -> Hash256 {
        self.0.finalize().into()
    }
}

/// What names a device's records: its driver's identity, its instance's layer settings' digest,
/// and its profile fingerprint.
pub(crate) struct Records {
    pub(crate) identity: Identity,
    pub(crate) settings: Hash256,
    pub(crate) profile: Hash256,
}

impl Records {
    /// The name of `making`'s record.
    pub(crate) fn name(&self, making: Making) -> String {
        let key = making
            .digest(&Sha256::digest(self.identity.0).into())
            .digest(&self.settings)
            .finish();
        entry_name(
            &store::content(),
            EntryKind::PipelineData,
            &key,
            &self.profile,
        )
    }
}

/// What a making's cache started from.
struct Seed {
    /// The record's bytes.
    bytes: Vec<u8>,
    /// Where its data lies in them.
    data: Range<usize>,
}

/// Make with `create`, handed a `VkPipelineCache` of the making's own on the device of `raw`
/// (the module docs), seeded from `making`'s record in `store` where one loads, and write the
/// data the making added as the record; `create` is handed a null cache where the driver makes
/// none. `raw` is the device's checked accessor's (`DeviceShared::vk`).
///
/// Once the process is exiting (`exiting`: the exit guard's [`closed`](exit_guard::closed), or a
/// test's own), the making's own `vkCreate*Pipelines` is the one Vulkan call it makes: a pool
/// piece's section, begun before the close, runs on past it (`exit_guard`), and what the making
/// would add to it (its cache's creation, its data's read for the record, its release) is left
/// out. Asked before the cache is made, where `create` is then handed a null cache, and again once
/// `create` returned, where the cache is left to the process's end, unread.
pub(crate) fn make<T, E>(
    store: &Store,
    records: &Records,
    raw: &ash::Device,
    making: Making,
    exiting: impl Fn() -> bool,
    create: impl FnOnce(vk::PipelineCache) -> Result<T, E>,
) -> Result<T, E> {
    if exiting() {
        return create(vk::PipelineCache::null());
    }
    let name = records.name(making);
    let identity = &records.identity;
    let seed = match store.load_content(&name, |bytes| {
        let data = open(identity, &bytes)?;
        Some(Seed { bytes, data })
    }) {
        Lookup::Hit(seed) => Some(seed),
        Lookup::Absent | Lookup::Unreadable | Lookup::Rejected => None,
    };
    let seeded = seed.as_ref().and_then(|seed| {
        let info =
            vk::PipelineCacheCreateInfo::default().initial_data(&seed.bytes[seed.data.clone()]);
        // SAFETY: the device is live and `info` lives for the call.
        let made = unsafe { raw.create_pipeline_cache(&info, None) };
        exit_guard::called(Call::CreatePipelineCache);
        match made {
            Ok(cache) => {
                count(&store.counters.driver_loads);
                Some(cache)
            }
            Err(_) => {
                // The driver refused data that passed the checks; the making's save replaces it.
                count(&store.counters.rejected);
                None
            }
        }
    });
    // The creation log's split names whether the making started from its record.
    crate::pipeline::creation_log::record_loaded(seeded.is_some());
    let loaded = if seeded.is_some() {
        seed.as_ref()
    } else {
        None
    };
    let cache = match seeded {
        Some(cache) => cache,
        None => {
            let info = vk::PipelineCacheCreateInfo::default();
            // SAFETY: as above.
            let made = unsafe { raw.create_pipeline_cache(&info, None) };
            exit_guard::called(Call::CreatePipelineCache);
            match made {
                Ok(cache) => cache,
                Err(_) => return create(vk::PipelineCache::null()),
            }
        }
    };
    let made = create(cache);
    if exiting() {
        return made;
    }
    if made.is_ok() {
        keep(store, (identity, &name), raw, cache, loaded);
    }
    // SAFETY: the making returned, and nothing else holds the cache.
    unsafe { release::object(raw, cache) };
    made
}

/// Write `cache`'s data as the record `name` unless it is `loaded`'s, byte for byte, or the
/// checks refuse it (counted). The read is a section of the exit guard ([`exit_guard`]): on the
/// program's own thread, once the process is exiting, nothing is read or written.
fn keep(
    store: &Store,
    (identity, name): (&Identity, &str),
    raw: &ash::Device,
    cache: vk::PipelineCache,
    loaded: Option<&Seed>,
) {
    let Some(busy) = exit_guard::enter() else {
        return;
    };
    // SAFETY: the cache is live, and no making uses it any more.
    let read = unsafe { raw.get_pipeline_cache_data(cache) };
    exit_guard::called(Call::GetPipelineCacheData);
    drop(busy);
    let Ok(data) = read else {
        store.write_failed();
        return;
    };
    if loaded.is_some_and(|seed| seed.bytes[seed.data.clone()] == data[..]) {
        count(&store.counters.driver_data_unchanged);
        return;
    }
    if let Err(refusal) = identity.check_vulkan_header(&data) {
        store.counters.refused(refusal);
        return;
    }
    match store.write_record(name, &seal(identity, &data)) {
        Ok(_) => count(&store.counters.driver_saves),
        Err(_) => store.write_failed(),
    }
}
